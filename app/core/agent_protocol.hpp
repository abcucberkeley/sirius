#ifndef SIRIUS_APP_AGENT_PROTOCOL_HPP
#define SIRIUS_APP_AGENT_PROTOCOL_HPP

// sirius-cli's machine interfaces -- the JSON-lines session and the MCP server -- over any
// ToolDispatcher. No I/O: the host feeds lines in and gives a sink for lines out.

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace sirius::app::agent {
    struct ToolHints {
        bool readOnly = false, destructive = false, idempotent = false, openWorld = false;
    };
    struct ToolDescriptor {
        std::string name, title, description;
        nlohmann::json inputSchema;                               // {"type":"object",...}
        ToolHints hints;
        nlohmann::json meta = nlohmann::json::object();           // Tool._meta
    };
    struct ToolError {
        std::string code, message, hint;
        nlohmann::json data;
    };
    struct Attachment {
        std::string mimeType;                                     // image/png | image/jpeg
        std::vector<std::uint8_t> bytes;
        std::string path;                                         // the copy in the server's scratch directory
        int width = 0, height = 0;
    };
    struct ToolResult {
        bool ok = true;
        nlohmann::json value = nlohmann::json::object();          // always an object
        ToolError error;
        std::vector<Attachment> images;
        std::vector<std::string> changes;
        bool undoable = false;
        std::vector<std::string> warnings;
    };
    ToolResult failure(std::string code, std::string message, std::string hint = {}, nlohmann::json data = nullptr);
    int exitCodeFor(const std::string& errorCode) noexcept;       // section 3.4
    // The MCP Tool object of a descriptor, with only the fields that MCP version knows (annotations from
    // 2025-03-26, title and _meta from 2025-06-18). MCP tools/list, the session's `tools` and `sirius-cli tools`
    // all list tools with it, so that the three agree.
    nlohmann::json toolJson(const ToolDescriptor& tool, const std::string& protocolVersion = "2025-11-25");
    struct CallContext {
        std::function<bool()> cancelled;                          // never null from a server
        std::function<void(double fraction, const std::string& message)> progress;
    };
    struct Status {
        bool running = false;
        double fraction = 0.0;
        int step = -1;
        std::string message, runId, workspace, activeTool;
        nlohmann::json toJson() const;
    };
    class ToolDispatcher {
    public:
        virtual ~ToolDispatcher() = default;
        virtual std::vector<ToolDescriptor> tools() const = 0;                        // main thread, stable order
        virtual bool hasTool(const std::string& name) const = 0;
        virtual ToolResult call(const std::string& name, const nlohmann::json& args, const CallContext& ctx) = 0;  // main thread
        virtual Status status() const = 0;                                           // any thread
        virtual void cancelActive() = 0;                                             // any thread
        virtual void pump() = 0;                                                     // main thread
        // main thread: {"event":"run_finished","run_id",...,"result":RunOutcome}
        //              {"event":"log","source":"workbench"|"worker","line":...}
        virtual std::vector<nlohmann::json> takeEvents() = 0;
    };
    using LineSink = std::function<void(const std::string& line)>;                   // thread-safe, one message, no '\n'
    struct ServerOptions {
        std::string version;
        std::chrono::milliseconds progressInterval{250};
        std::size_t maxInlineImageBytes = std::size_t{4} << 20;  // checked only; renderOutput keeps within it
        // Session: how long EOF waits for an active run (<0 = unbounded). MCP: how long EOF keeps answering
        // what was queued before it (<0 = 10 s).
        std::chrono::milliseconds eofRunWait{-1};
    };
    class Server {
    public:
        virtual ~Server();
        void receive(const std::string& line);            // the reader thread
        void endOfInput();                                // EOF or a termination signal (then terminate())
        void terminate();                                 // cancel whatever runs; finish without waiting
        bool step(std::chrono::milliseconds idleWait);    // main thread; false when finished
        void close();                                     // waits for a receive() in progress; after it, receive() does nothing
        int exitCode() const noexcept;

    protected:
        struct Impl;
        Server(ToolDispatcher& dispatcher, LineSink out, ServerOptions options, std::unique_ptr<Impl> impl);
        std::unique_ptr<Impl> impl_;
    };
    class SessionServer final : public Server {
    public:
        SessionServer(ToolDispatcher& dispatcher, LineSink out, ServerOptions options);
        void start();                                     // emits the "ready" event
    };
    struct McpOptions {
        std::string instructions;
        std::string serverName = "sirius";
        std::string serverTitle = "SIRIUS microscopy workbench";
    };
    // At EOF it answers what was queued before it for at most eofRunWait (10 s when unset), then cancels the
    // request in progress, drops the rest and cancels a run (D31). Log and run_finished events are dropped.
    class McpServer final : public Server {
    public:
        McpServer(ToolDispatcher& dispatcher, LineSink out, ServerOptions options, McpOptions mcp);
    };
} // namespace sirius::app::agent

#endif // SIRIUS_APP_AGENT_PROTOCOL_HPP
