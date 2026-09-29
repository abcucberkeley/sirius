#include "core/agent_protocol.hpp"

#include "core/agent_server.hpp"

#include <algorithm>
#include <exception>
#include <utility>

// What the session and the MCP server share: the error vocabulary, and the
// Server loop. The reader thread parses lines and queues requests (answering
// the few that must not wait behind a running tool itself); the main thread
// takes them one at a time, in order, and pumps the dispatcher between them.

namespace sirius::app::agent {

    using json = nlohmann::json;

    const char* const kLatestLegacyVersion = "2025-11-25";

    ToolResult failure(std::string code, std::string message, std::string hint, json data) {
        ToolResult r;
        r.ok = false;
        r.error = ToolError{std::move(code), std::move(message), std::move(hint), std::move(data)};
        return r;
    }

    int exitCodeFor(const std::string& errorCode) noexcept {
        // Section 3.4 of the CLI's contract; sirius-cli exits with these, and
        // `schema` prints the same table. A code nobody listed is a failure
        // (1), and so are failed, run_failed, export_failed, io_error and
        // internal. The protocols' own request errors are the client's usage
        // errors.
        static const char* const kUsage[] = {"usage", "unknown_tool", "parse_error", "invalid_request", "unknown_method"};
        static const char* const kInput[] = {"not_found", "open_failed", "invalid_argument", "unknown_step", "unknown_operation",
                                             "no_dataset", "validation", "not_computed", "unsupported", "too_large",
                                             "busy", "stale_workspace"};
        static const char* const kWorker[] = {"worker_unavailable", "python_not_found"};
        const auto in = [&](const auto& codes) {
            for (const char* code : codes)
                if (errorCode == code) return true;
            return false;
        };
        if (errorCode.empty()) return 0;
        if (in(kUsage)) return 2;
        if (in(kInput)) return 3;
        if (in(kWorker)) return 4;
        if (errorCode == "consent_required") return 5;
        if (errorCode == "timeout") return 124;
        if (errorCode == "cancelled") return 130;
        return 1;
    }

    json Status::toJson() const {
        return {{"running", running},
                {"fraction", fraction},
                {"step", step >= 0 ? json(step) : json(nullptr)},
                {"message", message},
                {"run_id", runId.empty() ? json(nullptr) : json(runId)},
                {"workspace", workspace},
                {"active_tool", activeTool.empty() ? json(nullptr) : json(activeTool)}};
    }

    // --- shared helpers ------------------------------------------------------------

    std::string compactJson(const json& value) { return value.dump(-1, ' ', false, json::error_handler_t::replace); }

    json errorJson(const ToolError& error) {
        return {{"code", error.code}, {"message", error.message}, {"hint", error.hint}, {"data", error.data}};
    }

    json valueObject(const json& value) {
        if (value.is_object()) return value;
        if (value.is_null()) return json::object();
        if (value.is_array()) return {{"items", value}};
        return {{"value", value}};
    }

    std::string base64(const std::vector<std::uint8_t>& bytes) {
        static constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out;
        out.reserve((bytes.size() + 2) / 3 * 4);
        std::size_t i = 0;
        for (; i + 2 < bytes.size(); i += 3) {
            const std::uint32_t v = (std::uint32_t{bytes[i]} << 16) | (std::uint32_t{bytes[i + 1]} << 8) | bytes[i + 2];
            out += kAlphabet[(v >> 18) & 63];
            out += kAlphabet[(v >> 12) & 63];
            out += kAlphabet[(v >> 6) & 63];
            out += kAlphabet[v & 63];
        }
        if (i < bytes.size()) {
            const bool two = i + 1 < bytes.size();
            const std::uint32_t v = (std::uint32_t{bytes[i]} << 16) | (two ? std::uint32_t{bytes[i + 1]} << 8 : 0u);
            out += kAlphabet[(v >> 18) & 63];
            out += kAlphabet[(v >> 12) & 63];
            out += two ? kAlphabet[(v >> 6) & 63] : '=';
            out += '=';
        }
        return out;
    }

    json toolJson(const ToolDescriptor& tool, const std::string& protocolVersion) {
        // Versions are dates, so they compare as strings. Annotations came
        // with 2025-03-26, title and _meta with 2025-06-18; a client of an
        // older version gets neither rather than fields it may reject. (The
        // header's default version is kLatestLegacyVersion.)
        json schema = tool.inputSchema.is_object() ? tool.inputSchema : json{{"type", "object"}, {"properties", json::object()}};
        json o = {{"name", tool.name}, {"description", tool.description}, {"inputSchema", std::move(schema)}};
        if (protocolVersion >= "2025-03-26")
            o["annotations"] = {{"readOnlyHint", tool.hints.readOnly},
                                {"destructiveHint", tool.hints.destructive},
                                {"idempotentHint", tool.hints.idempotent},
                                {"openWorldHint", tool.hints.openWorld}};
        if (protocolVersion >= "2025-06-18") {
            if (!tool.title.empty()) o["title"] = tool.title;
            if (tool.meta.is_object() && !tool.meta.empty()) o["_meta"] = tool.meta;
        }
        return o;
    }

    // --- Server::Impl --------------------------------------------------------------

    void Server::Impl::send(const json& message) noexcept {
        try {
            out(compactJson(message));
        } catch (...) {
            broken = true;
            wake.notify_all();
        }
    }

    bool Server::Impl::enqueue(Request request) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (ended || terminated) return false;
            queue.push_back(std::move(request));
        }
        wake.notify_all();
        return true;
    }

    std::optional<Server::Impl::Request> Server::Impl::takeQueued(const json& id) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it = std::find_if(queue.begin(), queue.end(), [&](const Request& r) { return r.id == id; });
        if (it == queue.end()) return std::nullopt;
        std::optional<Request> taken(std::move(*it));
        queue.erase(it);
        return taken;
    }

    void Server::Impl::pumpEvents() {
        dispatcher->pump();
        for (const json& event : dispatcher->takeEvents()) onEvent(event);
    }

    void Server::Impl::pause(std::chrono::milliseconds wait) {
        if (wait.count() <= 0) return;
        std::unique_lock<std::mutex> lock(mutex);
        wake.wait_for(lock, wait, [&] { return terminated || broken.load(); });
    }

    // --- Server ----------------------------------------------------------------------

    Server::Server(ToolDispatcher& dispatcher, LineSink out, ServerOptions options, std::unique_ptr<Impl> impl)
        : impl_(std::move(impl)) {
        impl_->dispatcher = &dispatcher;
        impl_->out = std::move(out);
        impl_->options = std::move(options);
    }

    Server::~Server() = default;

    void Server::receive(const std::string& line) {
        Impl& s = *impl_;
        std::lock_guard<std::mutex> guard(s.receiveMutex);
        if (s.closed) return;
        {
            // After end of input nothing more is read: a termination signal
            // ends input while the reader may still be delivering lines. A
            // session's shutdown does not end input, so that the client can
            // still ask for the status of a run, or cancel it, while the
            // session waits for it.
            std::lock_guard<std::mutex> lock(s.mutex);
            if (s.inputEnded || s.terminated) return;
        }
        // The line end may still be there, "\r\n" from a Windows client
        // included; blank lines are no requests.
        const std::size_t last = line.find_last_not_of(" \t\r\n");
        if (last == std::string::npos) return;
        const std::string text = line.substr(0, last + 1);
        try {
            s.onLine(text);
        } catch (...) {
            // The reader thread has nobody to hand an exception to; a line the
            // mode could not even answer is dropped rather than ending the process.
        }
    }

    void Server::endOfInput() {
        Impl& s = *impl_;
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            s.ended = s.inputEnded = true;
            if (!s.inputEndedAt) s.inputEndedAt = std::chrono::steady_clock::now();
        }
        s.wake.notify_all();
    }

    void Server::terminate() {
        Impl& s = *impl_;
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            s.ended = s.inputEnded = s.terminated = true;
            if (!s.inputEndedAt) s.inputEndedAt = std::chrono::steady_clock::now();
            if (s.active) *s.active->cancelled = true;
            s.queue.clear();
        }
        s.dispatcher->cancelActive();
        s.wake.notify_all();
    }

    bool Server::step(std::chrono::milliseconds idleWait) {
        Impl& s = *impl_;
        if (s.finished) return false;
        const auto finish = [&](int code) {
            if (code != 0) s.exitCode = code;
            s.finished = true;
            return false;
        };
        try {
            s.pumpEvents();
            std::optional<Impl::Request> next;
            bool ended = false;
            {
                std::unique_lock<std::mutex> lock(s.mutex);
                const auto ready = [&] { return !s.queue.empty() || s.ended || s.terminated || s.broken.load(); };
                if (!ready() && idleWait.count() > 0) s.wake.wait_for(lock, idleWait, ready);
                if (s.terminated) return finish(0);
                if (!s.queue.empty()) {
                    next = std::move(s.queue.front());
                    s.queue.pop_front();
                    s.active = *next;   // shares the flags, so a cancel still reaches it
                } else {
                    ended = s.ended;
                }
            }
            if (s.broken) return finish(1);
            if (next) {
                s.handle(*next);
                {
                    std::lock_guard<std::mutex> lock(s.mutex);
                    s.active.reset();
                }
                s.pumpEvents();
                return s.broken ? finish(1) : true;
            }
            if (ended && !s.onEnd(idleWait)) return finish(s.broken ? 1 : 0);
            return s.broken ? finish(1) : true;
        } catch (...) {
            // Only the dispatcher's own machinery (pump, takeEvents, status)
            // gets here -- a tool's failure is an answer. Nothing sensible can
            // follow a workbench that cannot be pumped.
            return finish(1);
        }
    }

    void Server::close() {
        Impl& s = *impl_;
        std::lock_guard<std::mutex> guard(s.receiveMutex);
        s.closed = true;
    }

    int Server::exitCode() const noexcept { return impl_->exitCode.load(); }

} // namespace sirius::app::agent
