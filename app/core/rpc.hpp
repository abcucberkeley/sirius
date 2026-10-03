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

    // The worker's half of the handshake, as sirius_worker/server.py does it:
    // the C++ server (core/rpc_server.hpp) and the tests' stand-in workers.
    // `answer` takes a "hello" or "auth" request's params and returns the
    // result to send, or nullopt with `error` set (and the connection should
    // then be closed). `capabilities` is what "auth" answers with. A second
    // "hello", or an "auth" before any, is refused as the worker refuses it.
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

    // A listening TCP socket: the server side of connectTcp (core/rpc_server.hpp
    // serves what it accepts). IPv4, as the Python worker binds; "" or
    // "0.0.0.0" is every interface, port 0 a free port. On Windows the port is
    // bound exclusively (SO_EXCLUSIVEADDRUSE), so no second socket can share
    // it and receive a client's hello. Throws ProtocolError when the bind fails.
    class Listener {
    public:
        Listener(const std::string& host, int port, int backlog = 16);
        ~Listener();
        Listener(const Listener&) = delete;
        Listener& operator=(const Listener&) = delete;

        int port() const noexcept { return port_; }
        const std::string& host() const noexcept { return host_; }
        // The next connection, or null when none came within `timeout` (or the
        // listener is closed). `peer` gets "address:port".
        std::unique_ptr<Transport> accept(std::chrono::milliseconds timeout, std::string* peer = nullptr);
        // Stops accepting; any thread.
        void close() noexcept;
        bool isOpen() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
        std::string host_;
        int port_ = 0;
    };

    // True for an address only this machine can reach ("127.0.0.1", "::1",
    // "localhost"); "" and "0.0.0.0" are every interface, and any other name
    // counts as public (server.py: is_loopback).
    bool isLoopbackHost(const std::string& host);

} // namespace sirius::app::rpc

namespace sirius::app {

    // One GPU of a worker's node, as its hello names it ("gpus"): the
    // hardware, whether or not the worker can compute on it.
    struct GpuInfo {
        std::string name;              // "NVIDIA A100-SXM4-80GB"
        std::int64_t memoryMb = 0;     // MiB, as nvidia-smi reports it
    };

    struct WorkerCapabilities {
        std::string version;                   // the worker package's version, e.g. "0.1.0"
        int protocolVersion = 0;               // rpc::kProtocolVersion the worker answered with
        std::vector<std::string> methods;      // "run:torch_segment", "model_info", ...
        bool cuda = false;
        std::string device;                    // "cuda:0 · RTX 4000 · 20 GB"
        // The node's GPU hardware (the Python worker asks nvidia-smi, the
        // engine its CUDA runtime), whether this worker can compute on a GPU
        // at all, and why not ("no CUDA library in the worker's
        // environment: ..."). A worker older than these fields: no GPUs
        // listed, `cudaUsable` = `cuda`, no reason.
        std::vector<GpuInfo> gpus;
        bool cudaUsable = false;
        std::string cudaReason;
        int cpuThreads = 0;                    // 0 = not said
        std::string hostname;
        std::string python;
        std::string torch;                     // hello's "torch": its version in the worker ("" none, or not said)
        // What dataset_read / dataset_view replies may be compressed with,
        // best first ("zstd", "zlib"); empty for a worker without them.
        std::vector<std::string> encodings;
        int maxClients = 1;                    // connections the worker serves at once (--max-clients)
        // hello's "tiff_reader": the version of the sirius package the worker
        // reads TIFF datasets with ("" when it has none and cannot open TIFF),
        // and whether it decodes them on the GPU (nvTIFF)
        std::string tiffReader;
        bool nvtiff = false;
        // hello's "engine" block: present when the peer is SIRIUS's C++
        // engine (sirius-cli serve, core/engine_server.hpp) rather than the
        // Python worker -- its build (core/build_info.hpp), devices and the
        // Python worker it runs beside it. Null for the Python worker.
        nlohmann::json engine;
    };

    // The hardware fields of a hello's capabilities ("gpus", "cuda_usable",
    // "cuda_reason", "cpu_threads") into `caps`, whose `cuda` is already
    // read; a field that is missing or of another type is left as it is.
    void parseWorkerHardware(const nlohmann::json& hello, WorkerCapabilities& caps);

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
        // The same, handing each progress frame's header over whole: the
        // engine's frames carry the step and its state besides the fraction.
        WorkerResult callWithFrames(const std::string& method, const nlohmann::json& params, const std::vector<rpc::TensorRef>& tensors,
                                    const std::function<void(const nlohmann::json& frame)>& progressFrame,
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
        WorkerResult exchange(const std::string& method, const nlohmann::json& params, const std::vector<rpc::TensorRef>& tensors,
                              const std::function<void(double, const std::string&)>& progress,
                              const std::function<void(const nlohmann::json&)>& progressFrame, const std::function<bool()>& cancelled);

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
