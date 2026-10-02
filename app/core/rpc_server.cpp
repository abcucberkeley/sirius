#include "core/rpc_server.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "core/cancel.hpp"
#include "core/errors.hpp"

namespace sirius::app::rpc {

    using json = nlohmann::json;

    namespace {
        std::uint32_t u32At(const std::vector<std::byte>& b, std::size_t at) {
            std::uint32_t v = 0;
            for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(std::to_integer<unsigned>(b[at + static_cast<std::size_t>(i)])) << (8 * i);
            return v;
        }
        std::uint64_t u64At(const std::vector<std::byte>& b, std::size_t at) {
            std::uint64_t v = 0;
            for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(std::to_integer<unsigned>(b[at + static_cast<std::size_t>(i)])) << (8 * i);
            return v;
        }

        // A frame larger than `cap` at the front of `buffer`, judged from
        // its lengths as soon as they have arrived (before its bytes are
        // waited for): the message to refuse it with, or "".
        std::string oversizeFrame(const std::vector<std::byte>& buffer, std::uint64_t cap) {
            if (buffer.size() < 4) return {};
            const std::uint32_t hlen = u32At(buffer, 0);
            if (hlen > cap) return "header length " + std::to_string(hlen) + " exceeds " + std::to_string(cap);
            if (buffer.size() < 4 + static_cast<std::size_t>(hlen) + 8) return {};
            const std::uint64_t plen = u64At(buffer, 4 + hlen);
            if (plen > cap) return "payload length " + std::to_string(plen) + " exceeds " + std::to_string(cap);
            return {};
        }

        double unixSeconds() {
            return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        }

        std::vector<TensorRef> refsOf(const std::vector<Tensor>& tensors) {
            std::vector<TensorRef> out;
            out.reserve(tensors.size());
            for (const Tensor& t : tensors) out.push_back(TensorRef{t.name, t.dtype, t.shape, t.bytes.data(), t.bytes.size()});
            return out;
        }
    } // namespace

    // --- one connection --------------------------------------------------------------------

    struct Server::Connection {
        std::unique_ptr<Transport> transport;
        std::string peer;
        std::mutex sendMutex;
        bool closed = false;   // under sendMutex: nothing is sent once the connection is given up

        // false when the connection is gone
        bool send(const json& header, const std::vector<Tensor>& tensors = {}) {
            std::vector<std::byte> bytes = encodeFrame(header, refsOf(tensors));
            const std::lock_guard<std::mutex> g(sendMutex);
            if (closed || !transport || !transport->isOpen()) return false;
            try {
                transport->send(bytes);
                return true;
            } catch (const std::exception&) {
                return false;
            }
        }
        bool reply(const json& id, json result, const std::vector<Tensor>& tensors = {}) {
            return send({{"id", id}, {"type", "result"}, {"result", std::move(result)}}, tensors);
        }
        bool error(const json& id, const std::string& message) { return send({{"id", id}, {"type", "error"}, {"message", message}}); }
        void close() {
            const std::lock_guard<std::mutex> g(sendMutex);
            closed = true;
            if (transport) transport->close();
        }
        bool isClosed() {
            const std::lock_guard<std::mutex> g(sendMutex);
            return closed;
        }
    };

    struct Server::Job {
        json id;
        const Connection* owner = nullptr;
        bool slot = false;                // the server's one Dispatch::Job
        std::atomic<bool> cancel{false};
        std::atomic<bool> done{false};
        std::thread thread;
    };

    class Server::Context final : public CallContext {
    public:
        Context(const Server& server, std::shared_ptr<Connection> conn, json id, std::shared_ptr<Job> job)
            : server_(server), conn_(std::move(conn)), id_(std::move(id)), job_(std::move(job)) {}
        void progress(double fraction, const std::string& message, const json& extra) override {
            json h = {{"id", id_}, {"type", "progress"}, {"fraction", fraction}, {"message", message}};
            if (extra.is_object())
                for (auto it = extra.begin(); it != extra.end(); ++it) h[it.key()] = it.value();
            conn_->send(h);
        }
        bool cancelled() const override {
            return (job_ && job_->cancel.load()) || server_.stopping() || conn_->isClosed();
        }

    private:
        const Server& server_;
        std::shared_ptr<Connection> conn_;
        json id_;
        std::shared_ptr<Job> job_;
    };

    // --- the server ----------------------------------------------------------------------

    Server::Server(Options options) : options_(std::move(options)) {
        options_.maxClients = std::max(options_.maxClients, 1);
        options_.maxPreauth = std::max(options_.maxPreauth, 1);
    }

    Server::~Server() {
        stop();
        {
            const std::lock_guard<std::mutex> g(threadsMutex_);
            for (auto& [t, done] : connectionThreads_)
                if (t.joinable()) t.join();
            connectionThreads_.clear();
        }
        std::vector<std::shared_ptr<Job>> jobs;
        {
            const std::lock_guard<std::mutex> g(jobsMutex_);
            jobs.swap(jobs_);
        }
        for (auto& j : jobs) {
            j->cancel.store(true);
            if (j->thread.joinable()) j->thread.join();
        }
    }

    void Server::handle(const std::string& method, Handler handler, Dispatch dispatch) {
        handlers_[method] = Entry{std::move(handler), dispatch};
    }

    void Server::setFallback(Handler handler, Dispatch dispatch) { fallback_ = Entry{std::move(handler), dispatch}; }

    void Server::setCapabilities(std::function<json()> capabilities) { capabilities_ = std::move(capabilities); }

    void Server::log(const std::string& line) const {
        if (options_.log) options_.log(line);
    }

    void Server::stop() {
        if (stop_.exchange(true)) return;
        {
            const std::lock_guard<std::mutex> g(stateMutex_);
            stateChanged_.notify_all();
        }
        const std::lock_guard<std::mutex> g(jobsMutex_);
        for (auto& j : jobs_) j->cancel.store(true);
    }

    void Server::waitForStop() {
        std::unique_lock<std::mutex> lk(stateMutex_);
        stateChanged_.wait(lk, [this] { return stop_.load(); });
    }

    int Server::clients() const {
        const std::lock_guard<std::mutex> g(stateMutex_);
        return clients_;
    }

    void Server::serve(Listener& listener) {
        log("listening on " + (listener.host().empty() ? std::string("0.0.0.0") : listener.host()) + ":" + std::to_string(listener.port()));
        while (!stopping()) {
            std::string peer;
            std::unique_ptr<Transport> t;
            try {
                t = listener.accept(std::chrono::milliseconds(250), &peer);
            } catch (const std::exception& e) {
                log(std::string("accept failed: ") + e.what());
                break;
            }
            {
                // threads of connections that have ended
                const std::lock_guard<std::mutex> g(threadsMutex_);
                for (auto it = connectionThreads_.begin(); it != connectionThreads_.end();) {
                    if (it->second->load()) {
                        if (it->first.joinable()) it->first.join();
                        it = connectionThreads_.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
            if (!t) {
                if (!listener.isOpen()) break;
                continue;
            }
            bool crowded;
            {
                const std::lock_guard<std::mutex> g(stateMutex_);
                crowded = preauth_ >= options_.maxPreauth;
                if (!crowded) ++preauth_;
            }
            if (crowded) {
                log("connection from " + peer + " closed: " + std::to_string(options_.maxPreauth) + " connections are in their handshake already");
                t->close();
                continue;
            }
            auto conn = std::make_shared<Connection>();
            conn->transport = std::move(t);
            conn->peer = peer;
            auto done = std::make_shared<std::atomic<bool>>(false);
            std::thread th([this, conn, done] {
                serveOne(conn);
                done->store(true);
            });
            const std::lock_guard<std::mutex> g(threadsMutex_);
            connectionThreads_.emplace_back(std::move(th), std::move(done));
        }
        listener.close();
        stop();
        const std::lock_guard<std::mutex> g(threadsMutex_);
        for (auto& [th, done] : connectionThreads_)
            if (th.joinable()) th.join();
        connectionThreads_.clear();
    }

    void Server::serveConnection(std::unique_ptr<Transport> transport, const std::string& peer) {
        {
            const std::lock_guard<std::mutex> g(stateMutex_);
            ++preauth_;
        }
        auto conn = std::make_shared<Connection>();
        conn->transport = std::move(transport);
        conn->peer = peer;
        serveOne(conn);
    }

    bool Server::takeSlot(Connection& conn, std::vector<std::byte>& inbox) {
        std::unique_lock<std::mutex> lk(stateMutex_);
        if (options_.maxClients > 1) {
            if (clients_ >= options_.maxClients) return false;
            ++clients_;
            return true;
        }
        // One client at a time: this one waits for the one before it, as
        // long as it takes, unless it hangs up or the server stops.
        while (!stopping()) {
            if (clients_ < 1) {
                ++clients_;
                return true;
            }
            stateChanged_.wait_for(lk, std::chrono::milliseconds(250));
            lk.unlock();
            try {
                conn.transport->receive(inbox, std::chrono::milliseconds(0));   // a peer that left throws
            } catch (const std::exception&) {
                return false;
            }
            lk.lock();
        }
        return false;
    }

    void Server::releaseSlot() {
        const std::lock_guard<std::mutex> g(stateMutex_);
        --clients_;
        stateChanged_.notify_all();
    }

    void Server::cancelJobs(const Connection* owner, const json* id) {
        const std::lock_guard<std::mutex> g(jobsMutex_);
        for (auto& j : jobs_)
            if (j->owner == owner && !j->done.load() && (!id || id->is_null() || j->id == *id)) {
                j->cancel.store(true);
                log("cancel requested for " + j->id.dump());
            }
    }

    void Server::reapJobs() {
        std::vector<std::shared_ptr<Job>> finished;
        {
            const std::lock_guard<std::mutex> g(jobsMutex_);
            for (auto it = jobs_.begin(); it != jobs_.end();) {
                if ((*it)->done.load()) {
                    finished.push_back(*it);
                    it = jobs_.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (auto& j : finished)
            if (j->thread.joinable()) j->thread.join();
    }

    void Server::startJob(const std::shared_ptr<Connection>& conn, Request request, const Entry& entry) {
        reapJobs();
        auto job = std::make_shared<Job>();
        job->id = request.id;
        job->owner = conn.get();
        job->slot = entry.dispatch == Dispatch::Job;
        const Handler handler = entry.handler;
        const bool timed = job->slot;
        auto body = [this, conn, job, handler, timed, req = std::move(request)]() mutable {
            const auto t0 = std::chrono::steady_clock::now();
            Context ctx(*this, conn, job->id, job);
            try {
                Reply r = handler(req, ctx);
                if (job->cancel.load()) {
                    conn->error(job->id, "cancelled");
                } else {
                    if (timed && r.result.is_object())
                        r.result["seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    conn->reply(job->id, std::move(r.result), r.tensors);
                }
            } catch (const std::exception& e) {
                const bool cancelled = job->cancel.load() || isCancellation(e);
                if (!cancelled) log(req.method + " failed: " + e.what());
                conn->error(job->id, cancelled ? std::string("cancelled") : std::string(e.what()));
            } catch (...) {
                conn->error(job->id, job->cancel.load() ? std::string("cancelled") : std::string("internal error"));
            }
            {
                const std::lock_guard<std::mutex> g(jobsMutex_);
                if (slotJob_ == job) slotJob_.reset();
            }
            job->done.store(true);
        };
        json busy;
        {
            // The job is published and its thread started under one lock, so
            // two requests arriving together cannot both take the slot.
            const std::lock_guard<std::mutex> g(jobsMutex_);
            if (job->slot && slotJob_ && !slotJob_->done.load()) {
                busy = slotJob_->id;
            } else {
                if (job->slot) slotJob_ = job;
                jobs_.push_back(job);
                job->thread = std::thread(std::move(body));
            }
        }
        if (!busy.is_null()) conn->error(job->id, "busy: request " + busy.dump() + " is still running");
    }

    void Server::serveOne(const std::shared_ptr<Connection>& conn) {
        Connection& c = *conn;
        log("client " + c.peer + " connected");
        bool preauth = true, slot = false, authenticated = false;
        HandshakeResponder handshake;
        handshake.token = options_.token;
        const auto leavePreauth = [&] {
            if (!preauth) return;
            preauth = false;
            const std::lock_guard<std::mutex> g(stateMutex_);
            --preauth_;
        };
        const auto preauthDeadline = std::chrono::steady_clock::now() + options_.preauthTimeout;
        auto lastActivity = std::chrono::steady_clock::now();
        std::vector<std::byte> inbox;
        const auto ownJobRunning = [&] {
            const std::lock_guard<std::mutex> g(jobsMutex_);
            for (auto& j : jobs_)
                if (j->owner == &c && !j->done.load()) return true;
            return false;
        };

        while (!stopping()) {
            std::optional<Message> msg;
            try {
                if (!authenticated) {
                    // a peer that has not proved the token is not waited for, or buffered for, past a small frame
                    const std::string over = oversizeFrame(inbox, kMaxPreauthFrame);
                    if (!over.empty()) throw ProtocolError(over);
                }
                msg = decodeFrame(inbox);
            } catch (const std::exception& e) {
                log("protocol error from " + c.peer + ": " + e.what());
                c.error(nullptr, std::string("protocol error: ") + e.what());
                break;
            }
            if (!msg) {
                const auto now = std::chrono::steady_clock::now();
                if (!authenticated && now > preauthDeadline) {
                    log("client " + c.peer + " did not complete the handshake within " + std::to_string(options_.preauthTimeout.count()) +
                        " ms; dropped");
                    break;
                }
                if (authenticated && ownJobRunning()) {
                    lastActivity = now;
                } else if (authenticated && options_.idleTimeout.count() > 0 && now - lastActivity > options_.idleTimeout) {
                    log("client " + c.peer + " sent nothing for " + std::to_string(options_.idleTimeout.count() / 1000) + " s; closed");
                    break;
                }
                try {
                    c.transport->receive(inbox, std::chrono::milliseconds(250));
                } catch (const std::exception&) {
                    break;   // the peer closed
                }
                continue;
            }
            lastActivity = std::chrono::steady_clock::now();
            const json& h = msg->header;
            const json rid = h.contains("id") ? h["id"] : json(nullptr);
            const std::string method = h.contains("method") && h["method"].is_string() ? h["method"].get<std::string>() : std::string();
            json params = h.contains("params") && h["params"].is_object() ? h["params"] : json::object();
            const std::string type = h.contains("type") && h["type"].is_string() ? h["type"].get<std::string>() : std::string("request");
            if (type != "request") {
                c.error(rid, "unexpected frame type '" + type + "'");
                continue;
            }
            if (method == "hello" || (method == "auth" && !authenticated)) {
                std::string error;
                const std::optional<json> r = handshake.answer(method, params, json::object(), error);
                if (!r) {
                    log(c.peer + ": " + error);
                    c.error(rid, error);
                    break;
                }
                if (method == "hello") {
                    c.reply(rid, *r);
                    continue;
                }
                leavePreauth();
                slot = takeSlot(c, inbox);
                if (!slot) {
                    if (!stopping()) {
                        log("client " + c.peer + " refused: " + std::to_string(options_.maxClients) + " clients are connected already");
                        c.error(rid, "busy: " + std::to_string(options_.maxClients) + " clients are connected already");
                    }
                    break;
                }
                authenticated = true;
                lastActivity = std::chrono::steady_clock::now();
                json caps = json::object();
                try {
                    if (capabilities_) caps = capabilities_();
                } catch (const std::exception& e) {
                    log(std::string("capabilities failed: ") + e.what());
                }
                if (!caps.contains("protocol_version")) caps["protocol_version"] = kProtocolVersion;
                c.reply(rid, caps);
                continue;
            }
            if (!authenticated) {
                c.error(rid, "not authenticated: complete the handshake ('hello', then 'auth') first");
                continue;
            }
            if (method == "ping") {
                c.reply(rid, {{"time", unixSeconds()}});
                continue;
            }
            if (method == "shutdown") {
                // Privileged: it ends every job here; logged so the log says who did it.
                log("privileged request: shutdown, from " + c.peer);
                c.reply(rid, json::object());
                stop();
                break;
            }
            if (method == "cancel") {
                const json target = params.contains("id") ? params["id"] : (h.contains("target") ? h["target"] : json(nullptr));
                cancelJobs(&c, &target);
                c.reply(rid, {{"cancelled", target}});
                continue;
            }
            const Entry* entry = nullptr;
            if (auto it = handlers_.find(method); it != handlers_.end()) entry = &it->second;
            else if (fallback_) entry = &*fallback_;
            if (!entry) {
                c.error(rid, "unknown method '" + method + "'");
                continue;
            }
            Request req;
            req.id = rid;
            req.method = method;
            req.params = std::move(params);
            req.tensors = std::move(msg->tensors);
            req.peer = c.peer;
            if (entry->dispatch == Dispatch::Inline) {
                Context ctx(*this, conn, rid, nullptr);
                try {
                    Reply r = entry->handler(req, ctx);
                    if (!c.reply(rid, std::move(r.result), r.tensors)) break;
                } catch (const std::exception& e) {
                    log(method + " failed: " + e.what());
                    if (!c.error(rid, isCancellation(e) ? std::string("cancelled") : std::string(e.what()))) break;
                }
                continue;
            }
            startJob(conn, std::move(req), *entry);
        }

        // The connection is gone (or the server stops): cancel what it still
        // runs, and give that a while to end before the connection is closed.
        cancelJobs(&c, nullptr);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (ownJobRunning() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        c.close();
        leavePreauth();
        if (slot) releaseSlot();
        log("client " + c.peer + " disconnected");
    }

} // namespace sirius::app::rpc
