#include "core/agent_protocol.hpp"

#include "core/agent_server.hpp"

#include <algorithm>
#include <cctype>
#include <exception>
#include <utility>

// The JSON-lines session ("sirius-session/1"): requests {"id", "method",
// "params"} where the method is a tool name or one of the control methods,
// answered with the same envelope and error codes as the one-shot commands,
// plus progress, log and run_finished events. End of input drains the queue
// and then waits for a run that is still going (D31), so that
// `sirius-cli session < script.jsonl` does not cancel the runs it started.

namespace sirius::app::agent {

    using json = nlohmann::json;
    using Clock = std::chrono::steady_clock;

    namespace {

        constexpr const char* kProtocol = "sirius-session/1";
        constexpr const char* kRequestShape = "{\"id\":1,\"method\":\"<tool or control>\",\"params\":{}}";
        constexpr const char* kMethods = "the methods are the tool names (send {\"id\":1,\"method\":\"tools\"}) "
                                         "and tools, status, cancel, subscribe, ping and shutdown";
        // How long end of input keeps pumping after it has cancelled a run
        // that outlived eofRunWait, so that the run_finished of the cancelled
        // run can still be written.
        constexpr std::chrono::seconds kCancelGrace{10};

        // The headless run driver reports "Step 03 · SIM reconstruction · <message>";
        // the event carries the step and its name as fields of their own. Any
        // other text is the message as it is.
        struct ProgressText {
            json step;
            json stepName;
            std::string message;
        };
        ProgressText splitProgress(const std::string& text) {
            static const std::string kSeparator = " \xC2\xB7 ";
            ProgressText r{nullptr, nullptr, text};
            if (text.compare(0, 5, "Step ") != 0) return r;
            std::size_t i = 5;
            int step = 0;
            while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i])) && step < 100000)
                step = step * 10 + (text[i++] - '0');
            if (i == 5) return r;
            if (i == text.size()) {
                r.step = step;
                r.message.clear();
                return r;
            }
            if (text.compare(i, kSeparator.size(), kSeparator) != 0) return r;
            i += kSeparator.size();
            const std::size_t end = text.find(kSeparator, i);
            r.step = step;
            r.stepName = text.substr(i, end == std::string::npos ? std::string::npos : end - i);
            r.message = end == std::string::npos ? std::string() : text.substr(end + kSeparator.size());
            return r;
        }

        // Throttling state of one request's progress, shared with the callback
        // so a dispatcher that keeps the callback cannot reach a dead frame.
        struct ProgressGate {
            std::mutex mutex;
            Clock::time_point last;
            bool any = false;
            bool done = false;   // the response is out: nothing more for this request

            void close() {
                std::lock_guard<std::mutex> lock(mutex);
                done = true;
            }
        };

    } // namespace

    struct Server::Impl::SessionMode final : Server::Impl {
        std::atomic<bool> progressEvents{true};
        std::atomic<bool> logEvents{false};
        // Main thread: when end of input first found the queue empty, and
        // when it then cancelled a run that outlived eofRunWait.
        std::optional<Clock::time_point> drainedAt, cancelledAt;
        // The request whose answer first reported the current run as
        // running, and that run's id (guarded by mutex): `cancel` with that
        // id cancels the run. A later run_status that reports the same run
        // does not replace it, so cancelling a wait never cancels the run.
        json runRequestId;
        std::string runRequestRunId;

        void answer(const json& id, json result) { send({{"id", id}, {"ok", true}, {"result", std::move(result)}}); }
        void fail(const json& id, const ToolError& error, const std::vector<std::string>& warnings = {}) {
            json m = {{"id", id}, {"ok", false}, {"error", errorJson(error)}};
            if (!warnings.empty()) m["warnings"] = warnings;
            send(m);
        }
        void fail(const json& id, std::string code, std::string message, std::string hint = {}) {
            fail(id, ToolError{std::move(code), std::move(message), std::move(hint), nullptr});
        }

        void start() {
            const Status st = dispatcher->status();
            send({{"event", "ready"},
                  {"protocol", kProtocol},
                  {"version", options.version},
                  {"workspace", st.workspace},
                  {"tools", dispatcher->tools().size()}});
        }

        json statusJson() {
            Status st = dispatcher->status();
            if (st.activeTool.empty()) {
                std::lock_guard<std::mutex> lock(mutex);
                if (active) st.activeTool = active->method;
            }
            return st.toJson();
        }

        // --- reader thread

        void onLine(const std::string& line) override {
            json message;
            try {
                message = json::parse(line);
            } catch (const json::parse_error& e) {
                fail(nullptr, "parse_error", std::string("the line is not JSON: ") + e.what(),
                     std::string("send one JSON object per line: ") + kRequestShape);
                return;
            }
            if (!message.is_object()) {
                fail(nullptr, "invalid_request", "a request must be a JSON object", kRequestShape);
                return;
            }
            const auto id = message.find("id");
            if (id == message.end() || !(id->is_string() || id->is_number())) {
                fail(nullptr, "invalid_request", "every request needs an id, a number or a string", kRequestShape);
                return;
            }
            const auto method = message.find("method");
            if (method == message.end() || !method->is_string()) {
                fail(*id, "invalid_request", "the request has no method", kMethods);
                return;
            }
            Request r;
            r.id = *id;
            r.method = method->get<std::string>();
            const auto params = message.find("params");
            if (params != message.end() && !params->is_null()) {
                if (!params->is_object()) {
                    fail(r.id, "invalid_request", "params must be an object");
                    return;
                }
                r.params = *params;
            }
            // "inline" may sit beside the params as well as in them, since
            // it is an option of the request rather than of the tool; the
            // one in params wins.
            if (const auto it = message.find("inline"); it != message.end() && !r.params.contains("inline"))
                r.params["inline"] = *it;
            // These three are answered here, not queued, so they work while
            // a long tool holds the main thread, and after a shutdown while
            // the session waits for a run.
            if (r.method == "ping") {
                answer(r.id, json::object());
            } else if (r.method == "status") {
                answer(r.id, statusJson());
            } else if (r.method == "cancel") {
                cancel(r);
            } else {
                const json requestId = r.id;
                if (!enqueue(std::move(r))) fail(requestId, "cancelled", "the session is shutting down");
            }
        }

        // cancel {id}: a queued request with that id is dropped (and
        // answered as cancelled); the request being answered has its cancel
        // flag set (a `run` that is still waiting cancels the job it
        // started; any other tool only stops); the request that reported
        // the current run as running cancels that run. Any other id names
        // nothing that can still be cancelled -- a quick request answered a
        // moment ago, say -- and leaves the run alone.
        // cancel {} cancels whatever runs: the request and the run.
        void cancel(const Request& r) {
            const json target = r.params.value("id", json());
            if (target.is_null()) {
                bool cancelledRequest = false;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (active) {
                        *active->cancelled = true;
                        cancelledRequest = true;
                    }
                }
                const bool running = dispatcher->status().running;
                dispatcher->cancelActive();
                answer(r.id, {{"cancelled", cancelledRequest || running}});
                return;
            }
            if (std::optional<Request> dropped = takeQueued(target)) {
                fail(dropped->id, "cancelled", "cancelled before it started");
                answer(r.id, {{"cancelled", true}});
                return;
            }
            bool cancelledRequest = false, startedRun = false;
            std::string runId;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (active && active->id == target) {
                    *active->cancelled = true;
                    cancelledRequest = true;
                } else if (!runRequestId.is_null() && runRequestId == target) {
                    startedRun = true;
                    runId = runRequestRunId;
                }
            }
            bool cancelledRun = false;
            if (startedRun) {
                // Only the run that request started, and only while it runs.
                const Status st = dispatcher->status();
                cancelledRun = st.running && (runId.empty() || st.runId.empty() || st.runId == runId);
                if (cancelledRun) dispatcher->cancelActive();
            }
            answer(r.id, {{"cancelled", cancelledRequest || cancelledRun}});
        }

        // --- main thread

        void handle(Request& r) override {
            try {
                if (r.method == "tools") {
                    json list = json::array();
                    for (const ToolDescriptor& t : dispatcher->tools()) list.push_back(toolJson(t, kLatestLegacyVersion));
                    answer(r.id, {{"tools", std::move(list)}});
                } else if (r.method == "subscribe") {
                    subscribe(r);
                } else if (r.method == "shutdown") {
                    shutdown(r);
                } else if (dispatcher->hasTool(r.method)) {
                    callTool(r);
                } else {
                    fail(r.id, "unknown_method", "unknown method '" + r.method + "'", kMethods);
                }
            } catch (const std::exception& e) {
                fail(r.id, "internal", e.what());
            } catch (...) {
                fail(r.id, "internal", "unknown exception");
            }
        }

        void subscribe(const Request& r) {
            for (const char* key : {"progress", "log"}) {
                const auto it = r.params.find(key);
                if (it != r.params.end() && !it->is_boolean()) {
                    fail(r.id, "invalid_argument", std::string("subscribe: '") + key + "' must be true or false");
                    return;
                }
            }
            progressEvents = r.params.value("progress", progressEvents.load());
            logEvents = r.params.value("log", logEvents.load());
            answer(r.id, {{"progress", progressEvents.load()}, {"log", logEvents.load()}});
        }

        // shutdown ends the requests: what was queued behind it, and every
        // later request but the reader thread's three, is answered as
        // cancelled, and the session then waits for a run as at EOF. Lines
        // are still read, so that the client can ask for the run's status
        // or cancel it rather than wait without a bound.
        void shutdown(const Request& r) {
            std::deque<Request> dropped;
            {
                std::lock_guard<std::mutex> lock(mutex);
                ended = true;
                dropped.swap(queue);
            }
            answer(r.id, json::object());
            for (const Request& d : dropped) fail(d.id, "cancelled", "the session is shutting down");
        }

        void callTool(Request& r) {
            // "inline" is the session's own option (the image's bytes in the
            // response), not the tool's.
            bool inlineImages = false;
            if (const auto it = r.params.find("inline"); it != r.params.end()) {
                inlineImages = it->is_boolean() && it->get<bool>();
                r.params.erase(it);
            }
            auto gate = std::make_shared<ProgressGate>();
            CallContext ctx;
            ctx.cancelled = [flag = r.cancelled] { return flag->load(); };
            ctx.progress = [this, gate, id = r.id](double fraction, const std::string& text) {
                {
                    // The gate comes first: once it is closed the server may
                    // be gone, and nothing of `this` may be read.
                    std::lock_guard<std::mutex> lock(gate->mutex);
                    if (gate->done || !progressEvents) return;
                    const Clock::time_point now = Clock::now();
                    if (gate->any && now - gate->last < options.progressInterval) return;
                    gate->any = true;
                    gate->last = now;
                }
                const ProgressText p = splitProgress(text);
                send({{"event", "progress"},
                      {"id", id},
                      {"fraction", fraction},
                      {"step", p.step},
                      {"step_name", p.stepName},
                      {"message", p.message}});
            };

            ToolResult res;
            try {
                res = dispatcher->call(r.method, r.params, ctx);
            } catch (const std::exception& e) {
                res = failure("internal", e.what());
            } catch (...) {
                res = failure("internal", "unknown exception");
            }
            gate->close();
            // The log lines and finished runs of the call go out before
            // its answer, which is the order they happened in.
            pumpEvents();

            if (!res.ok) {
                fail(r.id, res.error, res.warnings);
                return;
            }
            json value = valueObject(res.value);
            if (!res.images.empty()) {
                if (inlineImages) {
                    // The limit holds for the images together: they all go
                    // out on the one line of the answer.
                    std::size_t total = 0;
                    for (const Attachment& a : res.images) total += a.bytes.size();
                    const std::size_t limit = options.maxInlineImageBytes;
                    if (total > limit) {
                        const std::string what = res.images.size() == 1 ? "the image is " : "the images are ";
                        const std::string message =
                            what + std::to_string(total) + " bytes, more than the " + std::to_string(limit) + " sent inline";
                        const json data = {{"path", res.images.front().path}, {"bytes", total}, {"limit", limit}};
                        const std::string hint = "read the file at data.path instead, or lower max_size";
                        fail(r.id, ToolError{"too_large", message, hint, data}, res.warnings);
                        return;
                    }
                }
                json images = json::array();
                for (const Attachment& a : res.images) {
                    json image = {{"path", a.path},
                                  {"mime_type", a.mimeType},
                                  {"width", a.width},
                                  {"height", a.height},
                                  {"bytes", a.bytes.size()}};
                    if (inlineImages) image["base64"] = base64(a.bytes);
                    images.push_back(std::move(image));
                }
                value["image"] = images.front();
                if (images.size() > 1) value["images"] = std::move(images);
            }
            // Remembered before the answer goes out, so that a client that
            // cancels the run as soon as it reads the answer finds it.
            const auto status = value.find("status");
            if (status != value.end() && *status == "running") {
                const auto run = value.find("run_id");
                const std::string runId = run != value.end() && run->is_string() ? run->get<std::string>() : std::string();
                std::lock_guard<std::mutex> lock(mutex);
                if (runRequestId.is_null() || runId.empty() || runId != runRequestRunId) {
                    runRequestId = r.id;
                    runRequestRunId = runId;
                }
            }
            send({{"id", r.id},
                  {"ok", true},
                  {"result", std::move(value)},
                  {"changes", res.changes},
                  {"undoable", res.undoable},
                  {"warnings", res.warnings}});
        }

        void onEvent(const json& event) override {
            const std::string kind = event.is_object() ? event.value("event", std::string()) : std::string();
            if (kind == "log" && !logEvents) return;
            if (kind == "progress" && !progressEvents) return;
            send(event);
        }

        bool onEnd(std::chrono::milliseconds idleWait) override {
            const Clock::time_point now = Clock::now();
            if (!drainedAt) drainedAt = now;
            if (!dispatcher->status().running) {
                // One more pump: a run that finished since the last one is
                // folded back here, and its run_finished written.
                pumpEvents();
                return false;
            }
            if (!cancelledAt && options.eofRunWait.count() >= 0 && now - *drainedAt >= options.eofRunWait) {
                dispatcher->cancelActive();
                cancelledAt = now;
            }
            if (cancelledAt && now - *cancelledAt >= kCancelGrace) return false;
            pause(std::min(idleWait, std::chrono::milliseconds(50)));
            return true;
        }
    };

    SessionServer::SessionServer(ToolDispatcher& dispatcher, LineSink out, ServerOptions options)
        : Server(dispatcher, std::move(out), std::move(options), std::make_unique<Impl::SessionMode>()) {}

    void SessionServer::start() { static_cast<Impl::SessionMode&>(*impl_).start(); }

} // namespace sirius::app::agent
