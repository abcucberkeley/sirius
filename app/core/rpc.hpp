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
    inline constexpr int kProtocolVersion = 1;

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
        explicit RemoteWorker(std::unique_ptr<rpc::Transport> transport, std::string token = {});
        ~RemoteWorker();

        static std::unique_ptr<RemoteWorker> connect(const std::string& host, int port, const std::string& token,
                                                     std::chrono::milliseconds timeout = std::chrono::seconds(5));

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
        // thread forever); 15 s by default, shorter in tests.
        void setCancelGrace(std::chrono::milliseconds grace) noexcept { cancelGrace_ = grace; }

    private:
        std::unique_ptr<rpc::Transport> transport_;
        std::string token_;
        WorkerCapabilities caps_;
        std::vector<std::byte> inbox_;
        std::uint64_t nextId_ = 1;
        std::chrono::milliseconds cancelGrace_{15000};
    };

    // Launches the bundled worker as a child process on a free local port
    // (Backend::Cuda / Cpu steps that need Python, e.g. Torch models).
    struct LocalWorkerConfig {
        std::string python = "python3";        // interpreter; env SIRIUS_PYTHON overrides
        std::string scriptDir;                 // directory holding sirius_worker/; empty = next to the executable
        int port = 0;                          // 0 = pick a free port
        std::string device = "auto";           // "cuda", "cpu", "auto"
    };
    std::string workerScriptPath(const std::string& scriptDir = {});

} // namespace sirius::app

#endif // SIRIUS_APP_RPC_HPP
