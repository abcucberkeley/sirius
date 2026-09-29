#include "core/agent_protocol.hpp"

#include "core/agent_server.hpp"

#include <algorithm>
#include <exception>
#include <utility>

// The Model Context Protocol server on stdio: JSON-RPC 2.0, one message per
// line, tools only. It speaks both eras (D15): the initialize handshake of
// 2024-11-05 to 2025-11-25, with the fields each version knows, and 2026-07-28,
// where every request carries its version in _meta and server/discover
// replaces the handshake. Log lines and finished runs are not MCP messages
// and are dropped (D32).
//
// End of input (D31) cancels, but not at once: what the client sent before
// closing stdin is still answered, in order, for a grace of eofRunWait (10 s
// when unset), because a client may write its requests and close its end
// straight away (a transcript fed from a file does). Once the grace is over,
// the request in progress is cancelled, the rest of the queue is dropped
// unanswered, and a run still going is cancelled -- so a client that leaves
// during `run_status {wait_s:-1}` does not keep the server, the worker and
// the CPU busy until the run ends.

namespace sirius::app::agent {

    using json = nlohmann::json;
    using Clock = std::chrono::steady_clock;

    namespace {

        constexpr const char* kModernVersion = "2026-07-28";
        constexpr const char* kVersionKey = "io.modelcontextprotocol/protocolVersion";
        constexpr const char* kCapabilitiesKey = "io.modelcontextprotocol/clientCapabilities";
        constexpr const char* kServerInfoKey = "io.modelcontextprotocol/serverInfo";
        constexpr const char* kLegacyVersions[] = {"2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"};

        // JSON-RPC and MCP error codes. -32000 to -32019 and the rest of the
        // implementation range are never used.
        constexpr int kParseError = -32700;
        constexpr int kInvalidRequest = -32600;
        constexpr int kMethodNotFound = -32601;
        constexpr int kInvalidParams = -32602;
        constexpr int kInternalError = -32603;
        constexpr int kUnsupportedVersion = -32022;

        constexpr std::int64_t kTtlMs = 3600000;
        // How long end of input keeps answering when eofRunWait is not set.
        constexpr std::chrono::seconds kEofGrace{10};

        json supportedVersions() {
            json v = json::array({kModernVersion});
            for (const char* legacy : kLegacyVersions) v.push_back(legacy);
            return v;
        }

        // Throttling and monotonicity of one request's progress notifications.
        struct ProgressGate {
            std::mutex mutex;
            Clock::time_point last;
            double sent = -1.0;   // the last progress value sent
            bool done = false;    // the response is out (or suppressed): nothing more

            void close() {
                std::lock_guard<std::mutex> lock(mutex);
                done = true;
            }
        };

    } // namespace

    struct Server::Impl::McpMode final : Server::Impl {
        McpOptions mcp;
        std::string negotiated;   // main thread: the version initialize settled on

        explicit McpMode(McpOptions m) : mcp(std::move(m)) {}

        // --- writing

        void error(const std::optional<json>& id, int code, const std::string& message, const json& data = nullptr) {
            json e = {{"code", code}, {"message", message}};
            if (!data.is_null()) e["data"] = data;
            json m = {{"jsonrpc", "2.0"}};
            if (id) m["id"] = *id;   // omitted when there is no usable id
            m["error"] = std::move(e);
            send(m);
        }
        void reply(const json& id, json result) { send({{"jsonrpc", "2.0"}, {"id", id}, {"result", std::move(result)}}); }
        // Main thread: the answer to a queued request, unless the client
        // cancelled it, which means it gets no answer at all.
        void answer(const Request& r, json result) {
            if (!*r.silenced) reply(r.id, std::move(result));
        }
        void refuse(const Request& r, int code, const std::string& message, const json& data = nullptr) {
            if (!*r.silenced) error(r.id, code, message, data);
        }

        // Any thread: true once end of input is older than the grace. From
        // then on the request in progress counts as cancelled (and answers
        // with what it did), and what is still queued is dropped.
        bool pastEofGrace() {
            const std::chrono::milliseconds grace =
                options.eofRunWait.count() >= 0 ? options.eofRunWait : std::chrono::milliseconds(kEofGrace);
            std::lock_guard<std::mutex> lock(mutex);
            return inputEndedAt && Clock::now() - *inputEndedAt >= grace;
        }

        json serverMeta() const {
            return {{kServerInfoKey, {{"name", mcp.serverName}, {"version", options.version}}}};
        }
        // Every modern result is complete (this server never answers in
        // parts) and names the server.
        void modernize(json& result) const {
            result["resultType"] = "complete";
            result["_meta"] = serverMeta();
        }

        // --- reader thread

        void onLine(const std::string& line) override {
            json message;
            try {
                message = json::parse(line);
            } catch (const json::parse_error& e) {
                error(std::nullopt, kParseError, std::string("Parse error: ") + e.what());
                return;
            }
            if (message.is_array()) {
                error(std::nullopt, kInvalidRequest, "Invalid request: batches are not supported");
                return;
            }
            if (!message.is_object()) {
                error(std::nullopt, kInvalidRequest, "Invalid request: a message must be a JSON object");
                return;
            }
            std::optional<json> id;
            if (const auto it = message.find("id"); it != message.end()) {
                if (!(it->is_string() || it->is_number())) {
                    error(std::nullopt, kInvalidRequest, "Invalid request: the id must be a string or a number");
                    return;
                }
                id = *it;
            }
            const auto method = message.find("method");
            if (method == message.end()) {
                // A response carries no method; this server sends no
                // requests, so there is nothing it could answer.
                if (message.contains("result") || message.contains("error")) return;
                error(id, kInvalidRequest, "Invalid request: no method");
                return;
            }
            if (!method->is_string()) {
                error(id, kInvalidRequest, "Invalid request: the method must be a string");
                return;
            }
            const auto version = message.find("jsonrpc");
            if (version == message.end() || !version->is_string() || version->get<std::string>() != "2.0") {
                error(id, kInvalidRequest, "Invalid request: jsonrpc must be \"2.0\"");
                return;
            }
            json params = json::object();
            if (const auto it = message.find("params"); it != message.end() && !it->is_null()) {
                if (!it->is_object()) {
                    if (id) error(id, kInvalidParams, "Invalid params: params must be an object");
                    return;
                }
                params = *it;
            }
            const std::string name = method->get<std::string>();
            if (!id) {
                // Notifications are never answered. Only a cancel needs
                // acting on; notifications/initialized changes nothing here.
                if (name == "notifications/cancelled") cancel(params);
                return;
            }
            if (name == "ping") {
                reply(*id, json::object());   // in both eras, whatever else is going on
                return;
            }
            Request r;
            r.id = std::move(*id);
            r.method = name;
            r.params = std::move(params);
            enqueue(std::move(r));
        }

        // notifications/cancelled {requestId}: the request being answered
        // has its cancel flag set and is never answered; a queued one is
        // dropped. Unknown ids and malformed cancels are ignored. The
        // dispatcher's run is not touched: `run` itself cancels its job
        // when its request is cancelled, and run_status only stops waiting.
        void cancel(const json& params) {
            const auto it = params.find("requestId");
            if (it == params.end() || !(it->is_string() || it->is_number())) return;
            // The queue is looked at first: step() moves a request from the
            // queue to active under one lock, so a request missed in the
            // queue here is already active below, never neither.
            if (takeQueued(*it)) return;
            std::lock_guard<std::mutex> lock(mutex);
            if (active && active->id == *it) {
                *active->silenced = true;
                *active->cancelled = true;
            }
        }

        // --- main thread

        void handle(Request& r) override {
            // The client closed stdin longer than the grace ago: what is
            // still queued is dropped, like a request it cancelled.
            if (pastEofGrace()) return;

            // The era is decided per request: modern metadata wins, then
            // the legacy handshake, and a request with neither is refused.
            const json* meta = nullptr;
            if (const auto it = r.params.find("_meta"); it != r.params.end() && it->is_object()) meta = &*it;
            const bool modern = meta && meta->contains(kVersionKey);
            std::string version;
            if (modern) {
                const json& requested = (*meta)[kVersionKey];
                if (!requested.is_string() || requested.get<std::string>() != kModernVersion) {
                    refuse(r, kUnsupportedVersion, "Unsupported protocol version",
                           {{"supported", json::array({kModernVersion})}, {"requested", requested}});
                    return;
                }
                if (!meta->contains(kCapabilitiesKey)) {
                    refuse(r, kInvalidParams, std::string("Invalid params: _meta has no ") + kCapabilitiesKey);
                    return;
                }
                version = kModernVersion;
            } else if (r.method == "initialize") {
                initialize(r);
                return;
            } else if (!negotiated.empty()) {
                version = negotiated;
            } else {
                refuse(r, kInvalidParams, "missing _meta protocol fields (send initialize first or use 2026-07-28 request metadata)");
                return;
            }

            try {
                if (r.method == "tools/list") {
                    listTools(r, version, modern);
                } else if (r.method == "tools/call") {
                    callTool(r, meta, version, modern);
                } else if (modern && r.method == "server/discover") {
                    discover(r);
                } else if (r.method == "ping") {
                    answer(r, json::object());
                } else {
                    refuse(r, kMethodNotFound, "Method not found: " + r.method);
                }
            } catch (const std::exception& e) {
                refuse(r, kInternalError, std::string("Internal error: ") + e.what());
            } catch (...) {
                refuse(r, kInternalError, "Internal error");
            }
        }

        void initialize(const Request& r) {
            std::string requested;
            if (const auto it = r.params.find("protocolVersion"); it != r.params.end() && it->is_string())
                requested = it->get<std::string>();
            // An unknown version is not an error: the server answers with
            // the newest it speaks and the client decides.
            negotiated = kLatestLegacyVersion;
            for (const char* legacy : kLegacyVersions)
                if (requested == legacy) negotiated = requested;
            json serverInfo = {{"name", mcp.serverName}, {"version", options.version}};
            if (negotiated >= "2025-06-18") serverInfo["title"] = mcp.serverTitle;
            json result = {{"protocolVersion", negotiated},
                           {"capabilities", {{"tools", json::object()}}},
                           {"serverInfo", std::move(serverInfo)}};
            if (!mcp.instructions.empty()) result["instructions"] = mcp.instructions;
            answer(r, std::move(result));
        }

        void discover(const Request& r) {
            json result = {{"supportedVersions", supportedVersions()},
                           {"capabilities", {{"tools", json::object()}}},
                           {"ttlMs", kTtlMs},
                           {"cacheScope", "public"}};
            if (!mcp.instructions.empty()) result["instructions"] = mcp.instructions;
            modernize(result);
            answer(r, std::move(result));
        }

        void listTools(const Request& r, const std::string& version, bool modern) {
            // Every tool fits one page, so no cursor is ever handed out and
            // any cursor a client sends back is not one of ours.
            if (const auto it = r.params.find("cursor"); it != r.params.end() && !it->is_null()) {
                if (!it->is_string() || !it->get<std::string>().empty()) {
                    refuse(r, kInvalidParams, "Invalid params: unknown cursor (every tool is on the first page)");
                    return;
                }
            }
            json list = json::array();
            for (const ToolDescriptor& t : dispatcher->tools()) list.push_back(toolJson(t, version));
            json result = {{"tools", std::move(list)}};
            if (modern) {
                result["ttlMs"] = kTtlMs;
                result["cacheScope"] = "public";
                modernize(result);
            }
            answer(r, std::move(result));
        }

        void callTool(Request& r, const json* meta, const std::string& version, bool modern) {
            const auto name = r.params.find("name");
            if (name == r.params.end() || !name->is_string()) {
                refuse(r, kInvalidParams, "Invalid params: tools/call needs the tool's name");
                return;
            }
            const std::string tool = name->get<std::string>();
            if (!dispatcher->hasTool(tool)) {
                refuse(r, kInvalidParams, "Unknown tool: " + tool);
                return;
            }
            json args = json::object();
            if (const auto it = r.params.find("arguments"); it != r.params.end() && !it->is_null()) {
                if (!it->is_object()) {
                    refuse(r, kInvalidParams, "Invalid params: arguments must be an object");
                    return;
                }
                args = *it;
            }

            // Progress goes out only for a request that asked with a token.
            json token;
            if (meta && meta->contains("progressToken")) {
                const json& t = (*meta)["progressToken"];
                if (t.is_string() || t.is_number()) token = t;
            }
            auto gate = std::make_shared<ProgressGate>();
            CallContext ctx;
            // Cancelled by the client, or by end of input once the grace is
            // over (see the top of this file).
            ctx.cancelled = [this, flag = r.cancelled] { return flag->load() || pastEofGrace(); };
            const bool withMessage = version >= "2025-03-26";   // progress messages came with 2025-03-26
            ctx.progress = [this, gate, token, silenced = r.silenced, withMessage](double fraction, const std::string& message) {
                if (token.is_null() || *silenced) return;
                const double progress = 100.0 * std::clamp(fraction, 0.0, 1.0);
                {
                    std::lock_guard<std::mutex> lock(gate->mutex);
                    const Clock::time_point now = Clock::now();
                    if (gate->done || !(progress > gate->sent)) return;
                    if (gate->sent >= 0 && now - gate->last < options.progressInterval) return;
                    gate->sent = progress;
                    gate->last = now;
                }
                json params = {{"progressToken", token}, {"progress", progress}, {"total", 100}};
                if (withMessage && !message.empty()) params["message"] = message;
                send({{"jsonrpc", "2.0"}, {"method", "notifications/progress"}, {"params", std::move(params)}});
            };

            ToolResult res;
            try {
                res = dispatcher->call(tool, args, ctx);
            } catch (const std::exception& e) {
                // The dispatcher answers a tool's own failure as a result;
                // an exception is its machinery failing, a protocol error.
                gate->close();
                refuse(r, kInternalError, std::string("Internal error: ") + e.what());
                return;
            } catch (...) {
                gate->close();
                refuse(r, kInternalError, "Internal error");
                return;
            }
            gate->close();
            if (*r.silenced) return;

            // The limit holds for the images together, since they all go out
            // base64-encoded on the one line of the response.
            std::size_t imageBytes = 0;
            for (const Attachment& a : res.images) imageBytes += a.bytes.size();
            if (res.ok && imageBytes > options.maxInlineImageBytes) {
                const std::string what = res.images.size() == 1 ? "the image is " : "the images are ";
                const std::vector<std::string> warnings = std::move(res.warnings);
                const std::string path = res.images.front().path;
                res = failure("too_large",
                              what + std::to_string(imageBytes) + " bytes, more than the " +
                                  std::to_string(options.maxInlineImageBytes) + " this server sends",
                              "lower max_size or pick a region",
                              {{"bytes", imageBytes}, {"limit", options.maxInlineImageBytes}, {"path", path}});
                res.warnings = warnings;
            }

            const bool structured = version >= "2025-06-18";
            json content = json::array();
            json result;
            if (res.ok) {
                json value = valueObject(res.value);
                content.push_back({{"type", "text"}, {"text", compactJson(value)}});
                for (const Attachment& a : res.images)
                    content.push_back({{"type", "image"}, {"data", base64(a.bytes)}, {"mimeType", a.mimeType}});
                result = {{"isError", false}};
                if (structured) result["structuredContent"] = std::move(value);
            } else {
                std::string text = res.error.code + ": " + res.error.message;
                if (!res.error.hint.empty()) text += "\nhint: " + res.error.hint;
                content.push_back({{"type", "text"}, {"text", std::move(text)}});
                result = {{"isError", true}};
                if (structured) result["structuredContent"] = {{"error", errorJson(res.error)}};
            }
            // Warnings (arguments the tool ignored, a stale output) go last,
            // after any image, where the model reads them with the result.
            for (const std::string& w : res.warnings) content.push_back({{"type", "text"}, {"text", "warning: " + w}});
            result["content"] = std::move(content);
            if (modern) modernize(result);
            answer(r, std::move(result));
        }

        void onEvent(const json& /*event*/) override {}

        bool onEnd(std::chrono::milliseconds /*idleWait*/) override {
            // Everything queued is answered (or dropped after the grace). An
            // MCP client closes stdin when it is done with the server, so
            // nobody waits for a run any more.
            if (dispatcher->status().running) dispatcher->cancelActive();
            return false;
        }
    };

    McpServer::McpServer(ToolDispatcher& dispatcher, LineSink out, ServerOptions options, McpOptions mcp)
        : Server(dispatcher, std::move(out), std::move(options), std::make_unique<Impl::McpMode>(std::move(mcp))) {}

} // namespace sirius::app::agent
