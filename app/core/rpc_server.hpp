#ifndef SIRIUS_APP_RPC_SERVER_HPP
#define SIRIUS_APP_RPC_SERVER_HPP

// The serving half of the worker protocol (core/rpc.hpp), in C++: what
// app/python/sirius_worker/server.py does, for SIRIUS's own engine
// (core/engine_server.hpp, `sirius-cli serve`). The same frames, the same
// handshake and the same rules, so RemoteWorker cannot tell the two apart:
//
//   * nothing is served before `hello` (protocol version, the client's nonce;
//     answered with this side's nonce and proof) and `auth` (the client's
//     proof) -- HandshakeResponder -- and until then a peer is held to
//     kMaxPreauthFrame bytes per frame and `preauthTimeout`, and at most
//     `maxPreauth` such peers are served at once;
//   * `maxClients` authenticated connections at once: with one, the next
//     connection's `auth` is answered once the one before has gone; with
//     more, one too many is refused as busy;
//   * every reply carries its request's id; a job streams "progress" frames
//     and is cancelled by `cancel {id}` or by its connection going away;
//   * built in: ping, cancel, shutdown. Everything else is a handler: inline
//     on the connection's thread (a dataset view), a job (one at a time
//     across the server, "busy" otherwise) or concurrent (a call relayed to
//     another worker).
//
// A connection is a Transport, so a server is tested in-process over
// rpc::loopbackPair() and served for real from a Listener.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/rpc.hpp"

namespace sirius::app::rpc {

    // The largest frame a peer may send before it has authenticated
    // (server.py: MAX_PREAUTH_FRAME): a hello is a few hundred bytes.
    inline constexpr std::uint64_t kMaxPreauthFrame = 16u << 10;

    struct Request {
        nlohmann::json id;                         // as the client sent it; every reply echoes it
        std::string method;
        nlohmann::json params = nlohmann::json::object();
        std::vector<Tensor> tensors;
        std::string peer;                          // "address:port", "loopback"
    };

    struct Reply {
        nlohmann::json result = nlohmann::json::object();
        std::vector<Tensor> tensors;
    };

    // What a handler sees of its request while it runs.
    class CallContext {
    public:
        virtual ~CallContext() = default;
        // A "progress" frame {fraction, message} with `extra`'s fields added
        // (the engine's step / state).
        virtual void progress(double fraction, const std::string& message, const nlohmann::json& extra = nlohmann::json()) = 0;
        // The client cancelled this request, its connection went away, or the
        // server is stopping.
        virtual bool cancelled() const = 0;
        std::function<bool()> cancelFlag() const {
            return [this] { return cancelled(); };
        }
    };

    enum class Dispatch {
        Inline,       // on the connection's thread: quick requests, served while a job runs on another connection
        Job,          // a thread of its own, one job at a time across the server; cancellable
        Concurrent,   // a thread of its own, any number at once; cancellable
    };

    class Server {
    public:
        using Handler = std::function<Reply(const Request&, CallContext&)>;

        struct Options {
            std::string token;                                  // "" = the empty token (a loopback-only server)
            int maxClients = 1;                                 // authenticated connections at once
            std::chrono::milliseconds preauthTimeout{5000};     // from connecting to a completed handshake
            int maxPreauth = 8;                                 // connections in their handshake at once
            std::chrono::milliseconds idleTimeout{3600000};     // an idle authenticated connection is closed; 0 = never
            std::function<void(const std::string&)> log;        // one line per event (connections, refusals, shutdown)
        };

        explicit Server(Options options);
        // stop(), then waits for every connection and job thread.
        ~Server();
        Server(const Server&) = delete;
        Server& operator=(const Server&) = delete;

        // Registers `method`; set them up before serving.
        void handle(const std::string& method, Handler handler, Dispatch dispatch = Dispatch::Inline);
        // What serves a method nobody registered (a relay); without one such a
        // request is answered "unknown method '<m>'".
        void setFallback(Handler handler, Dispatch dispatch = Dispatch::Concurrent);
        // What `auth` answers with; called once per connection, after the proof.
        void setCapabilities(std::function<nlohmann::json()> capabilities);

        // Serves one connection on the calling thread until it ends or the
        // server stops (in-process clients, tests).
        void serveConnection(std::unique_ptr<Transport> transport, const std::string& peer = "loopback");
        // Accepts from `listener`, one thread per connection, until stop().
        void serve(Listener& listener);
        // Stops accepting and ends every connection (their jobs are
        // cancelled); any thread, also a handler's.
        void stop();
        bool stopping() const noexcept { return stop_.load(); }
        // Blocks until stop() has been called.
        void waitForStop();

        // The authenticated connections now (tests, the engine's status).
        int clients() const;

    private:
        struct Connection;
        struct Job;
        class Context;
        struct Entry {
            Handler handler;
            Dispatch dispatch = Dispatch::Inline;
        };

        void serveOne(const std::shared_ptr<Connection>& conn);
        bool takeSlot(Connection& conn, std::vector<std::byte>& inbox);
        void releaseSlot();
        void startJob(const std::shared_ptr<Connection>& conn, Request request, const Entry& entry);
        void cancelJobs(const Connection* owner, const nlohmann::json* id);
        void reapJobs();
        void log(const std::string& line) const;

        Options options_;
        std::map<std::string, Entry> handlers_;
        std::optional<Entry> fallback_;
        std::function<nlohmann::json()> capabilities_;

        std::atomic<bool> stop_{false};
        mutable std::mutex stateMutex_;
        std::condition_variable stateChanged_;
        int clients_ = 0;
        int preauth_ = 0;

        std::mutex jobsMutex_;
        std::vector<std::shared_ptr<Job>> jobs_;
        std::shared_ptr<Job> slotJob_;   // the one Dispatch::Job running

        std::mutex threadsMutex_;
        std::vector<std::pair<std::thread, std::shared_ptr<std::atomic<bool>>>> connectionThreads_;
    };

} // namespace sirius::app::rpc

#endif // SIRIUS_APP_RPC_SERVER_HPP
