#include "core/rpc.hpp"

#include "core/cancel.hpp"

#include "core/errors.hpp"

#include <sirius/checked_math.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
using socket_t = SOCKET;
#define SIRIUS_INVALID_SOCKET INVALID_SOCKET
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
#define SIRIUS_INVALID_SOCKET (-1)
#endif

namespace sirius::app::rpc {

    // Frame size limits. Both lengths in a frame header come from the peer,
    // so they are bounded before they are used for arithmetic or indexing.
    constexpr std::uint64_t kMaxHeaderBytes = 64ull << 20;    // 64 MiB of JSON
    constexpr std::uint64_t kMaxPayloadBytes = 32ull << 30;   // 32 GiB of tensors


    using json = nlohmann::json;

    namespace {
        void putU32(std::vector<std::byte>& out, std::uint32_t v) {
            for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xff));
        }
        void putU64(std::vector<std::byte>& out, std::uint64_t v) {
            for (int i = 0; i < 8; ++i) out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xff));
        }
        std::uint32_t getU32(const std::byte* p) {
            std::uint32_t v = 0;
            for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(std::to_integer<unsigned>(p[i])) << (8 * i);
            return v;
        }
        std::uint64_t getU64(const std::byte* p) {
            std::uint64_t v = 0;
            for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(std::to_integer<unsigned>(p[i])) << (8 * i);
            return v;
        }
        std::size_t dtypeSize(const std::string& dtype) {
            if (dtype == "float32" || dtype == "uint32" || dtype == "int32") return 4;
            if (dtype == "float64" || dtype == "int64" || dtype == "uint64") return 8;
            if (dtype == "uint16" || dtype == "int16") return 2;
            if (dtype == "uint8" || dtype == "int8" || dtype == "bool") return 1;
            throw std::invalid_argument("rpc: unsupported dtype '" + dtype + "'");
        }
    } // namespace

    Index Tensor::numel() const {
        // The shape arrives on the wire, so the product is checked: a wrapped
        // one would pass the nbytes validation below and then let a consumer
        // that walks the shape run off the end of the payload.
        return sirius::detail::checkedProduct(shape.begin(), shape.end(), "rpc: tensor shape");
    }

    const float* Tensor::asFloat32() const {
        if (dtype != "float32") throw ProtocolError("rpc: tensor '" + name + "' is " + dtype + ", not float32");
        return reinterpret_cast<const float*>(bytes.data());
    }

    const std::uint32_t* Tensor::asUInt32() const {
        if (dtype != "uint32") throw ProtocolError("rpc: tensor '" + name + "' is " + dtype + ", not uint32");
        return reinterpret_cast<const std::uint32_t*>(bytes.data());
    }

    std::vector<std::byte> encodeFrame(const json& headerIn, const std::vector<TensorRef>& tensors) {
        json header = headerIn;
        json list = json::array();
        std::uint64_t offset = 0;
        for (const TensorRef& t : tensors) {
            Index n = 1;
            for (Index d : t.shape) n *= d;
            const std::size_t expected = static_cast<std::size_t>(n) * dtypeSize(t.dtype);
            if (expected != t.nbytes)
                throw std::invalid_argument("rpc: tensor '" + t.name + "' has " + std::to_string(t.nbytes) +
                                            " bytes, shape and dtype imply " + std::to_string(expected));
            list.push_back({{"name", t.name}, {"dtype", t.dtype}, {"shape", t.shape}, {"offset", offset}, {"nbytes", t.nbytes}});
            offset += t.nbytes;
        }
        if (!tensors.empty()) header["tensors"] = list;
        const std::string h = header.dump();
        std::vector<std::byte> out;
        out.reserve(4 + h.size() + 8 + offset);
        putU32(out, static_cast<std::uint32_t>(h.size()));
        for (char c : h) out.push_back(static_cast<std::byte>(c));
        putU64(out, offset);
        for (const TensorRef& t : tensors) {
            const auto* p = static_cast<const std::byte*>(t.data);
            out.insert(out.end(), p, p + t.nbytes);
        }
        return out;
    }

    std::optional<Message> decodeFrame(std::vector<std::byte>& buffer) {
        if (buffer.size() < 4) return std::nullopt;
        const std::uint32_t hlen = getU32(buffer.data());
        if (hlen > kMaxHeaderBytes) throw ProtocolError("rpc: header of " + std::to_string(hlen) + " bytes is not plausible");
        if (buffer.size() < 4 + hlen + 8) return std::nullopt;
        const std::uint64_t plen = getU64(buffer.data() + 4 + hlen);
        // Both lengths are attacker-controlled. Without the cap the total
        // below can wrap, the "is the frame complete" test then passes, and
        // the payload pointer runs past the buffer.
        if (plen > kMaxPayloadBytes)
            throw ProtocolError("rpc: payload of " + std::to_string(plen) + " bytes exceeds the limit");
        const std::uint64_t total = 4 + hlen + 8 + plen;
        if (buffer.size() < total) return std::nullopt;
        Message m;
        const std::string h(reinterpret_cast<const char*>(buffer.data() + 4), hlen);
        try {
            m.header = json::parse(h);
        } catch (const json::exception& e) {
            throw ProtocolError(std::string("rpc: bad header JSON: ") + e.what());
        }
        const std::byte* payload = buffer.data() + 4 + hlen + 8;
        if (m.header.contains("tensors") && m.header["tensors"].is_array()) {
            for (const json& tj : m.header["tensors"]) {
                Tensor t;
                t.name = tj.value("name", "");
                t.dtype = tj.value("dtype", "float32");
                if (tj.contains("shape")) t.shape = tj["shape"].get<std::vector<Index>>();
                const std::uint64_t off = tj.value("offset", 0ull), n = tj.value("nbytes", 0ull);
                // Written so neither sum nor product can wrap.
                if (n > plen || off > plen - n) throw ProtocolError("rpc: tensor '" + t.name + "' exceeds the payload");
                if (sirius::detail::checkedBytes(t.numel(), dtypeSize(t.dtype), "rpc: tensor size") != n)
                    throw ProtocolError("rpc: tensor '" + t.name + "' size does not match its shape");
                t.bytes.assign(payload + off, payload + off + n);
                m.tensors.push_back(std::move(t));
            }
        }
        buffer.erase(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(total));
        return m;
    }

    // --- TCP transport ------------------------------------------------------------------

    namespace {
#ifdef _WIN32
        struct WinsockInit {
            WinsockInit() {
                WSADATA d;
                WSAStartup(MAKEWORD(2, 2), &d);
            }
            ~WinsockInit() { WSACleanup(); }
        };
        void ensureWinsock() { static WinsockInit init; }
        void closeSocket(socket_t s) { closesocket(s); }
        int lastError() { return WSAGetLastError(); }
        bool wouldBlock(int e) { return e == WSAEWOULDBLOCK || e == WSAEINPROGRESS; }
#else
        void ensureWinsock() {}
        void closeSocket(socket_t s) { ::close(s); }
        int lastError() { return errno; }
        bool wouldBlock(int e) { return e == EINPROGRESS || e == EWOULDBLOCK || e == EAGAIN; }
#endif

        void setBlocking(socket_t s, bool blocking) {
#ifdef _WIN32
            u_long mode = blocking ? 0 : 1;
            ioctlsocket(s, FIONBIO, &mode);
#else
            const int flags = fcntl(s, F_GETFL, 0);
            fcntl(s, F_SETFL, blocking ? (flags & ~O_NONBLOCK) : (flags | O_NONBLOCK));
#endif
        }

        // poll() for readability / writability with a timeout
        bool waitFor(socket_t s, bool write, std::chrono::milliseconds timeout) {
#ifdef _WIN32
            fd_set set;
            FD_ZERO(&set);
            FD_SET(s, &set);
            timeval tv;
            tv.tv_sec = static_cast<long>(timeout.count() / 1000);
            tv.tv_usec = static_cast<long>((timeout.count() % 1000) * 1000);
            const int r = select(0, write ? nullptr : &set, write ? &set : nullptr, nullptr, &tv);
            return r > 0;
#else
            pollfd p;
            p.fd = s;
            p.events = write ? POLLOUT : POLLIN;
            p.revents = 0;
            const int r = ::poll(&p, 1, static_cast<int>(timeout.count()));
            return r > 0 && (p.revents & (write ? POLLOUT : (POLLIN | POLLHUP | POLLERR)));
#endif
        }

        class TcpTransport final : public Transport {
        public:
            explicit TcpTransport(socket_t s) : sock_(s) {}
            ~TcpTransport() override { close(); }

            void send(const std::vector<std::byte>& bytes) override {
                if (sock_ == SIRIUS_INVALID_SOCKET) throw ProtocolError("rpc: connection is closed");
                std::size_t sent = 0;
                while (sent < bytes.size()) {
                    if (!waitFor(sock_, true, std::chrono::seconds(30))) throw ProtocolError("rpc: send timed out");
                    const auto n = ::send(sock_, reinterpret_cast<const char*>(bytes.data() + sent),
                                          static_cast<int>(std::min<std::size_t>(bytes.size() - sent, 1u << 20)), 0);
                    if (n < 0) {
                        if (wouldBlock(lastError())) continue;
                        throw ProtocolError("rpc: send failed");
                    }
                    sent += static_cast<std::size_t>(n);
                }
            }

            bool receive(std::vector<std::byte>& into, std::chrono::milliseconds timeout) override {
                if (sock_ == SIRIUS_INVALID_SOCKET) throw ProtocolError("rpc: connection is closed");
                if (!waitFor(sock_, false, timeout)) return false;
                std::byte buf[1 << 16];
                const auto n = ::recv(sock_, reinterpret_cast<char*>(buf), sizeof buf, 0);
                if (n == 0) throw ProtocolError("rpc: the worker closed the connection");
                if (n < 0) {
                    if (wouldBlock(lastError())) return false;
                    throw ProtocolError("rpc: receive failed");
                }
                into.insert(into.end(), buf, buf + n);
                return true;
            }

            void close() override {
                if (sock_ != SIRIUS_INVALID_SOCKET) {
                    closeSocket(sock_);
                    sock_ = SIRIUS_INVALID_SOCKET;
                }
            }
            bool isOpen() const noexcept override { return sock_ != SIRIUS_INVALID_SOCKET; }

        private:
            socket_t sock_;
        };

        // In-memory transport pair for tests.
        struct Pipe {
            std::mutex m;
            std::condition_variable cv;
            std::deque<std::byte> data;
            bool closed = false;
        };

        class LoopbackTransport final : public Transport {
        public:
            LoopbackTransport(std::shared_ptr<Pipe> in, std::shared_ptr<Pipe> out) : in_(std::move(in)), out_(std::move(out)) {}
            ~LoopbackTransport() override { close(); }

            void send(const std::vector<std::byte>& bytes) override {
                std::lock_guard<std::mutex> g(out_->m);
                if (out_->closed) throw ProtocolError("rpc: connection is closed");
                out_->data.insert(out_->data.end(), bytes.begin(), bytes.end());
                out_->cv.notify_all();
            }
            bool receive(std::vector<std::byte>& into, std::chrono::milliseconds timeout) override {
                std::unique_lock<std::mutex> lock(in_->m);
                if (!in_->cv.wait_for(lock, timeout, [&] { return !in_->data.empty() || in_->closed; })) return false;
                if (in_->data.empty() && in_->closed) throw ProtocolError("rpc: the worker closed the connection");
                into.insert(into.end(), in_->data.begin(), in_->data.end());
                in_->data.clear();
                return true;
            }
            void close() override {
                if (!open_) return;
                open_ = false;
                for (auto& p : {in_, out_}) {
                    std::lock_guard<std::mutex> g(p->m);
                    p->closed = true;
                    p->cv.notify_all();
                }
            }
            bool isOpen() const noexcept override { return open_; }

        private:
            std::shared_ptr<Pipe> in_, out_;
            bool open_ = true;
        };
    } // namespace

    std::unique_ptr<Transport> connectTcp(const std::string& host, int port, std::chrono::milliseconds timeout) {
        ensureWinsock();
        addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        addrinfo* res = nullptr;
        if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res)
            throw ProtocolError("rpc: cannot resolve " + host);
        std::string lastErr = "no address";
        for (addrinfo* ai = res; ai; ai = ai->ai_next) {
            const socket_t s = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
            if (s == SIRIUS_INVALID_SOCKET) continue;
            setBlocking(s, false);
            int r = ::connect(s, ai->ai_addr, static_cast<int>(ai->ai_addrlen));
            bool ok = r == 0;
            if (!ok && wouldBlock(lastError())) {
                if (waitFor(s, true, timeout)) {
                    int err = 0;
                    socklen_t len = sizeof err;
                    getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&err), &len);
                    ok = err == 0;
                    if (!ok) lastErr = std::strerror(err);
                } else {
                    lastErr = "connection timed out";
                }
            } else if (!ok) {
                lastErr = std::strerror(lastError());
            }
            if (ok) {
                int one = 1;
                setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof one);
                freeaddrinfo(res);
                return std::make_unique<TcpTransport>(s);
            }
            closeSocket(s);
        }
        freeaddrinfo(res);
        throw ProtocolError("rpc: cannot connect to " + host + ":" + std::to_string(port) + " (" + lastErr + ")");
    }

    std::pair<std::unique_ptr<Transport>, std::unique_ptr<Transport>> loopbackPair() {
        auto a = std::make_shared<Pipe>(), b = std::make_shared<Pipe>();
        return {std::make_unique<LoopbackTransport>(a, b), std::make_unique<LoopbackTransport>(b, a)};
    }

} // namespace sirius::app::rpc

namespace sirius::app {

    using json = nlohmann::json;

    RemoteWorker::RemoteWorker(std::unique_ptr<rpc::Transport> transport, std::string token)
        : transport_(std::move(transport)), token_(std::move(token)) {
        if (!transport_) throw std::invalid_argument("RemoteWorker: no transport");
        const WorkerResult hello = call("hello", {{"token", token_}, {"protocol_version", rpc::kProtocolVersion}});
        // Same version on both ends or nothing: the framing and the method set
        // are versioned together, so a mismatch is reported here rather than
        // as a puzzling failure in the middle of a run. A worker predating the
        // handshake sends no field at all and counts as version 0.
        if (hello.result.contains("protocol_version") && hello.result["protocol_version"].is_number_integer())
            caps_.protocolVersion = hello.result["protocol_version"].get<int>();
        if (caps_.protocolVersion != rpc::kProtocolVersion)
            throw ProtocolError(
                "worker: protocol version mismatch. This application speaks version " +
                std::to_string(rpc::kProtocolVersion) + ", the worker speaks version " +
                std::to_string(caps_.protocolVersion) +
                (caps_.protocolVersion < rpc::kProtocolVersion
                     ? "; update sirius_worker (app/python) on the machine running the worker."
                     : "; update SIRIUS on this machine."));
        caps_.version = hello.result.value("version", "");
        caps_.cuda = hello.result.value("cuda", false);
        caps_.device = hello.result.value("device", "");
        caps_.hostname = hello.result.value("hostname", "");
        caps_.python = hello.result.value("python", "");
        if (hello.result.contains("methods") && hello.result["methods"].is_array())
            for (const json& m : hello.result["methods"]) caps_.methods.push_back(m.get<std::string>());
    }

    RemoteWorker::~RemoteWorker() { close(); }

    std::unique_ptr<RemoteWorker> RemoteWorker::connect(const std::string& host, int port, const std::string& token,
                                                        std::chrono::milliseconds timeout) {
        return std::make_unique<RemoteWorker>(rpc::connectTcp(host, port, timeout), token);
    }

    bool RemoteWorker::supports(const std::string& kind) const noexcept {
        return std::find(caps_.methods.begin(), caps_.methods.end(), "run:" + kind) != caps_.methods.end() ||
               std::find(caps_.methods.begin(), caps_.methods.end(), kind) != caps_.methods.end();
    }

    WorkerResult RemoteWorker::call(const std::string& method, const json& params,
                                    const std::vector<rpc::TensorRef>& tensors,
                                    const std::function<void(double, const std::string&)>& progress,
                                    const std::function<bool()>& cancelled) {
        if (!transport_ || !transport_->isOpen()) throw ProtocolError("worker: not connected");
        const std::uint64_t id = nextId_++;
        json header = {{"id", id}, {"type", "request"}, {"method", method}, {"params", params}};
        if (!token_.empty()) header["token"] = token_;
        const auto t0 = std::chrono::steady_clock::now();
        transport_->send(rpc::encodeFrame(header, tensors));
        bool cancelSent = false;
        // After a cancel the worker gets cancelGrace_ to answer (its job
        // polls the flag between tiles); then the connection is given up so
        // a hung worker cannot hold the run thread, and with it every edit
        // and the application's exit, forever.
        std::chrono::steady_clock::time_point cancelDeadline{};
        for (;;) {
            if (cancelled && cancelled() && !cancelSent) {
                transport_->send(rpc::encodeFrame({{"id", nextId_++}, {"type", "request"}, {"method", "cancel"}, {"params", {{"id", id}}}}, {}));
                cancelSent = true;
                cancelDeadline = std::chrono::steady_clock::now() + cancelGrace_;
            }
            if (cancelSent && std::chrono::steady_clock::now() > cancelDeadline) {
                transport_->close();
                throw CancelledError();
            }
            std::optional<rpc::Message> msg = rpc::decodeFrame(inbox_);
            if (!msg) {
                transport_->receive(inbox_, std::chrono::milliseconds(250));
                continue;
            }
            const json& h = msg->header;
            const std::string type = h.value("type", "");
            const auto idField = h.find("id");
            const bool hasId = idField != h.end() && idField->is_number_unsigned();
            if (!hasId) {
                // an error the worker could not attribute to a request (a
                // malformed frame, an oversize one) ends the exchange
                if (type == "error") throw std::runtime_error("worker: " + h.value("message", std::string("unknown error")));
                continue;
            }
            if (idField->get<std::uint64_t>() != id) continue;   // a stale reply
            if (type == "progress") {
                if (progress) progress(h.value("fraction", 0.0), h.value("message", ""));
                continue;
            }
            if (type == "error") {
                const std::string message = h.value("message", std::string("unknown error"));
                if (cancelSent && message == "cancelled") throw CancelledError();   // what we asked for
                throw std::runtime_error("worker: " + message);
            }
            if (type == "result") {
                WorkerResult r;
                r.result = h.value("result", json::object());
                r.tensors = std::move(msg->tensors);
                r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                return r;
            }
        }
    }

    void RemoteWorker::close() {
        if (transport_) transport_->close();
    }

    bool RemoteWorker::isOpen() const noexcept { return transport_ && transport_->isOpen(); }

    std::string workerScriptPath(const std::string& scriptDir) {
        namespace fs = std::filesystem;
        std::vector<fs::path> candidates;
        if (!scriptDir.empty()) candidates.push_back(fs::path(scriptDir));
        if (const char* env = std::getenv("SIRIUS_WORKER_DIR")) candidates.push_back(fs::path(env));
        candidates.push_back(fs::current_path() / "python");
#ifdef SIRIUS_APP_SOURCE_DIR
        candidates.push_back(fs::path(SIRIUS_APP_SOURCE_DIR) / "python");
#endif
        for (const fs::path& dir : candidates) {
            std::error_code ec;
            if (fs::exists(dir / "sirius_worker" / "__main__.py", ec)) return dir.string();
        }
        return {};
    }

} // namespace sirius::app
