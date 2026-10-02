#ifndef SIRIUS_APP_RPC_HPP
#define SIRIUS_APP_RPC_HPP

// Wire protocol to the Python compute worker (app/python/sirius_worker), the
// same worker that serves the HPC backend from a Slurm job. One TCP
// connection, request/response with streamed progress, no third-party
// dependency on either side:
//
//   frame := u32 header_len | header (UTF-8 JSON) | u64 payload_len | payload
//
// "hello" exchanges kProtocolVersion below; a peer answering with another
// version is refused, so a framing change is never silently misread.
//
// The token never crosses the wire. "hello" carries the client's random
// nonce and is answered with the worker's nonce and the worker's proof,
// HMAC-SHA256(token, "sirius-worker-auth/2|worker|<client>|<worker>"); the
// client checks that proof before it sends anything else, then sends its own
// ("auth", role "client") and only then gets the capabilities. A stand-in
// that does not know the token therefore learns nothing it could replay, and
// the client never talks to it past the handshake (handshakeProof below).
//
// header: {"id": n, "type": "request"|"progress"|"result"|"error",
//          "method": "...", "params": {...}, "tensors": [{"name", "dtype",
//          "shape", "offset", "nbytes"}], "message", "fraction"}
// Tensors are raw little-endian arrays concatenated in the payload.
// Methods: "hello" (capabilities), "model_info", "run" (kind, params) and
// "cancel". Integers, tokens and paths are all JSON; nothing is pickled.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <sirius/buffer.hpp>

namespace sirius::app::rpc {

    // Version of this wire protocol, sent in "hello" and echoed in its reply.
    // Both ends must speak the same number; bump it when the framing or the
    // method set changes in a way an older peer cannot understand. The Python
    // worker defines the same constant as PROTOCOL_VERSION in
    // app/python/sirius_worker/protocol.py.
    inline constexpr int kProtocolVersion = 2;

    // Most tensors one frame may describe, on both ends.
    inline constexpr std::size_t kMaxTensors = 64;

    // The largest payload a frame this side reads may announce. 8 GiB unless
    // $SIRIUS_RPC_MAX_PAYLOAD_GIB says otherwise (read once); the setter is
    // for tests and for a caller with a reason to change it.
    std::uint64_t maxPayloadBytes() noexcept;
    void setMaxPayloadBytes(std::uint64_t bytes) noexcept;

    // The proof of one side of the handshake: lower-case hex of
    // HMAC-SHA256(token, "sirius-worker-auth/2|" + role + "|" + clientNonce +
    // "|" + serverNonce), role "worker" or "client". sirius_worker/protocol.py
    // computes the same (handshake_proof).
    std::string handshakeProof(const std::string& token, const std::string& role, const std::string& clientNonce,
                               const std::string& serverNonce);
    // A nonce as the handshake takes one: 32 to 128 lower-case hex digits.
    bool validNonce(const nlohmann::json& value);
    // 16 bytes from the operating system's generator, as 32 hex digits.
    std::string randomNonce();

    // The worker's half of the handshake, for a stand-in worker (the tests'):
    // `answer` takes a "hello" or "auth" request's params and returns the
    // result to send, or nullopt with `error` set (and the connection should
    // then be closed). `capabilities` is what "auth" answers with.
    struct HandshakeResponder {
        std::string token;
        int protocolVersion = kProtocolVersion;   // what this stand-in claims
        std::string clientNonce, serverNonce;
        bool authenticated = false;
        std::optional<nlohmann::json> answer(const std::string& method, const nlohmann::json& params, const nlohmann::json& capabilities,
                                             std::string& error);
    };

    struct TensorRef {
        std::string name;
        std::string dtype;                 // "float32", "uint32", "uint8", "float64", "int64"
        std::vector<Index> shape;
        const void* data = nullptr;
        std::size_t nbytes = 0;
    };

    struct Tensor {                        // owned, decoded from a frame
        std::string name;
        std::string dtype;
        std::vector<Index> shape;
        std::vector<std::byte> bytes;
        Index numel() const;               // throws std::overflow_error on a wrapped shape
        const float* asFloat32() const;    // throws on dtype mismatch
        const std::uint32_t* asUInt32() const;
    };

    struct Message {
        nlohmann::json header;
        std::vector<Tensor> tensors;
    };

    // --- framing (pure, unit-tested) ---------------------------------------
    std::vector<std::byte> encodeFrame(const nlohmann::json& header, const std::vector<TensorRef>& tensors);
    // Consumes one complete frame from the front of `buffer` (erasing it) and
    // returns it; nullopt when the buffer holds less than a frame. Throws on
    // a malformed frame.
    std::optional<Message> decodeFrame(std::vector<std::byte>& buffer);

    // --- transport ----------------------------------------------------------
    class Transport {
    public:
        virtual ~Transport() = default;
        virtual void send(const std::vector<std::byte>& bytes) = 0;
        // Appends whatever arrived to `into`; false on timeout, throws when closed.
        virtual bool receive(std::vector<std::byte>& into, std::chrono::milliseconds timeout) = 0;
        virtual void close() = 0;
        virtual bool isOpen() const noexcept = 0;
    };

    // Blocking TCP client (POSIX / Winsock). Throws ProtocolError when the
    // connection fails.
    std::unique_ptr<Transport> connectTcp(const std::string& host, int port, std::chrono::milliseconds timeout);

    // The same through a SOCKS5 proxy on this machine (the `ssh -D` of a
    // cluster session, core/remote_host.hpp): CONNECT without
    // authentication, the host sent as a name so that the far end resolves
    // it -- a compute node's name means something only on the cluster.
    // Throws ProtocolError naming what failed: the proxy, the far end
    // refusing (the worker not listening yet), or the reply.
    std::unique_ptr<Transport> connectSocks5(const std::string& proxyHost, int proxyPort, const std::string& host, int port,
                                             std::chrono::milliseconds timeout);
    // connectTcp, or connectSocks5 through 127.0.0.1:`socksPort` when it is > 0.
    std::unique_ptr<Transport> connectEndpoint(const std::string& host, int port, int socksPort, std::chrono::milliseconds timeout);

    // In-memory pair for tests: what one end sends, the other receives.
    std::pair<std::unique_ptr<Transport>, std::unique_ptr<Transport>> loopbackPair();

} // namespace sirius::app::rpc

namespace sirius::app {

    struct WorkerCapabilities {
        std::string version;                   // the worker package's version, e.g. "0.1.0"
        int protocolVersion = 0;               // rpc::kProtocolVersion the worker answered with
        std::vector<std::string> methods;      // "run:torch_segment", "model_info", ...
        bool cuda = false;
        std::string device;                    // "cuda:0 · RTX 4000 · 20 GB"
        std::string hostname;
        std::string python;
        // What dataset_read / dataset_view replies may be compressed with,
        // best first ("zstd", "zlib"); empty for a worker without them.
        std::vector<std::string> encodings;
        int maxClients = 1;                    // connections the worker serves at once (--max-clients)
        std::string tifffile;                  // its version; "" when the worker cannot read TIFF itself
    };

    struct WorkerResult {
        nlohmann::json result;
        std::vector<rpc::Tensor> tensors;
        double seconds = 0.0;
    };

    // Client for one worker connection; every call is synchronous and may be
    // cancelled from another thread.
    class RemoteWorker {
    public:
        // How long the handshake of a caller that cannot cancel waits by
        // default: a worker's first answer imports torch and sets up CUDA,
        // which takes minutes on a cluster's shared filesystem.
        static constexpr std::chrono::milliseconds kHelloTimeout{300000};

        // The handshake (see the top of this file), then the capabilities.
        // A worker that cannot prove it holds `token` is refused before
        // anything but the nonce has been sent to it. The worker serves one
        // client at a time and reads a connection's hello only once the
        // client before it (a run, the model hub) has gone, so the wait may
        // be long. With `cancelled` it lasts until the answer comes or
        // `cancelled` says to stop (CancelledError); a caller that cannot
        // cancel is given `helloTimeout` instead, then ProtocolError.
        explicit RemoteWorker(std::unique_ptr<rpc::Transport> transport, std::string token = {},
                              const std::function<bool()>& cancelled = {}, std::chrono::milliseconds helloTimeout = kHelloTimeout);
        ~RemoteWorker();

        // `timeout` bounds the TCP connect; `cancelled` the handshake, as above.
        // A `socksPort` > 0 connects through that local SOCKS5 proxy (a cluster
        // session's SSH connection), which resolves `host` on the far side.
        static std::unique_ptr<RemoteWorker> connect(const std::string& host, int port, const std::string& token,
                                                     std::chrono::milliseconds timeout = std::chrono::seconds(5),
                                                     const std::function<bool()>& cancelled = {}, int socksPort = 0);

        const WorkerCapabilities& capabilities() const noexcept { return caps_; }
        bool supports(const std::string& kind) const noexcept;

        WorkerResult call(const std::string& method, const nlohmann::json& params,
                          const std::vector<rpc::TensorRef>& tensors = {},
                          const std::function<void(double, const std::string&)>& progress = {},
                          const std::function<bool()>& cancelled = {});
        void close();
        bool isOpen() const noexcept;
        // How long a cancelled call waits for the worker's answer before
        // the connection is given up (a stuck worker must not hold the run
        // thread forever); 15 s by default, shorter in tests. Zero gives the
        // connection up at once, without asking the worker to stop.
        void setCancelGrace(std::chrono::milliseconds grace) noexcept { cancelGrace_ = grace; }

    private:
        void handshake(const std::string& token, const std::function<bool()>& cancelled, std::chrono::milliseconds helloTimeout);

        std::unique_ptr<rpc::Transport> transport_;
        WorkerCapabilities caps_;
        std::vector<std::byte> inbox_;
        std::uint64_t nextId_ = 1;
        std::chrono::milliseconds cancelGrace_{15000};
    };

    // The first directory holding sirius_worker/__main__.py of: `scriptDir`,
    // an installed tree's share/sirius/python, the copy beside the executable
    // (core/app_paths.hpp), $SIRIUS_WORKER_DIR, ./python, the source tree's
    // app/python; "" when none does.
    std::string workerScriptPath(const std::string& scriptDir = {});

} // namespace sirius::app

#endif // SIRIUS_APP_RPC_HPP
