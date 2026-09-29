#ifndef SIRIUS_APP_AGENT_SERVER_HPP
#define SIRIUS_APP_AGENT_SERVER_HPP

// Private to the agent_protocol unit (agent_protocol.cpp, agent_session.cpp,
// agent_mcp.cpp): the state a Server keeps. The half both protocols share --
// the request queue between the reader thread and the main thread, the
// request being answered with its cancel flag, end of input, termination and
// close() -- is defined in agent_protocol.cpp. Each mode derives from Impl in
// its own file and says what goes over the wire: how a line is read, how a
// request is answered, which events are forwarded and what end of input waits
// for.

#include "core/agent_protocol.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace sirius::app::agent {

    struct Server::Impl {
        // One request on its way from the reader thread, which parsed it, to
        // the main thread, which answers it. The flags are shared so that the
        // reader thread can still reach a request the main thread has taken.
        struct Request {
            nlohmann::json id;                                // echoed with its JSON type
            std::string method;
            nlohmann::json params = nlohmann::json::object(); // always an object
            std::shared_ptr<std::atomic<bool>> cancelled = std::make_shared<std::atomic<bool>>(false);
            // MCP: the client cancelled it, so it is never answered.
            std::shared_ptr<std::atomic<bool>> silenced = std::make_shared<std::atomic<bool>>(false);
        };

        // The two modes, each defined in its own file. They are nested because
        // Impl is a protected member of Server, which only Server, the classes
        // derived from it and Impl's own members may name.
        struct SessionMode;
        struct McpMode;

        virtual ~Impl() = default;

        ToolDispatcher* dispatcher = nullptr;
        LineSink out;
        ServerOptions options;

        // --- what a mode supplies
        // Reader thread: parse one line (without its line end), then answer it
        // at once or enqueue() it.
        virtual void onLine(const std::string& line) = 0;
        // Main thread: answer a request taken from the queue.
        virtual void handle(Request& request) = 0;
        // Main thread: one event from ToolDispatcher::takeEvents(), to forward or drop.
        virtual void onEvent(const nlohmann::json& event) = 0;
        // Main thread, once input has ended and the queue is empty: false when
        // the server is finished, true to be stepped again (waiting at most
        // idleWait before returning).
        virtual bool onEnd(std::chrono::milliseconds idleWait) = 0;

        // --- shared helpers (agent_protocol.cpp)
        // Any thread. A sink that throws means nobody reads the output any
        // more, which ends the server with exit code 1.
        void send(const nlohmann::json& message) noexcept;
        // False when nothing more is queued (after a session's shutdown, end
        // of input or a signal); the caller then says what became of it.
        bool enqueue(Request request);
        // Removes the queued request with this id, if there is one.
        std::optional<Request> takeQueued(const nlohmann::json& id);
        // Main thread: dispatcher->pump(), then every event through onEvent().
        void pumpEvents();
        // Main thread: waits up to `wait`, returning early on termination.
        void pause(std::chrono::milliseconds wait);

        // Guards the queue, the active request and the end flags.
        std::mutex mutex;
        std::condition_variable wake;
        std::deque<Request> queue;
        std::optional<Request> active;   // the request handle() is answering
        bool ended = false;              // nothing more is queued: end of input, a session's shutdown, or a signal
        bool inputEnded = false;         // endOfInput() or terminate(): nothing more is read
        std::optional<std::chrono::steady_clock::time_point> inputEndedAt;
        bool terminated = false;         // terminate(): finish without waiting

        // Held for the whole of receive(), so close() can wait for one in progress.
        std::mutex receiveMutex;
        bool closed = false;

        std::atomic<bool> broken{false}; // the sink threw
        std::atomic<int> exitCode{0};
        bool finished = false;           // main thread
    };

    // Shared by the modes (agent_protocol.cpp).
    // A protocol line: compact, one line, invalid UTF-8 replaced rather than thrown on.
    std::string compactJson(const nlohmann::json& value);
    nlohmann::json errorJson(const ToolError& error);     // {code, message, hint, data}
    // A tool value as an object: arrays become {"items": [...]}, like the headless tools do.
    nlohmann::json valueObject(const nlohmann::json& value);
    std::string base64(const std::vector<std::uint8_t>& bytes);
    // The newest MCP version this server speaks with the initialize handshake.
    extern const char* const kLatestLegacyVersion;

} // namespace sirius::app::agent

#endif // SIRIUS_APP_AGENT_SERVER_HPP
