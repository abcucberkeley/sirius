#ifndef SIRIUS_APP_ENGINE_SERVER_HPP
#define SIRIUS_APP_ENGINE_SERVER_HPP

// SIRIUS's C++ engine as a worker-protocol server: what `sirius-cli serve`
// runs on a cluster node (docs: the HPC engine plan). To the application it
// is the Python worker's superset -- the same frames, handshake and replies
// (core/rpc_server.hpp) -- plus an "engine" block in its hello:
//
//   "engine": {"build", "version", "commit", "dirty", "ops_schema", "api"   (core/build_info.hpp)
//              "cuda": {"devices": [{"index", "name", "memory_gb"}], "nvtiff"},
//              "cpu_threads", "view_cache_used", "cache_used", "scratch",
//              "session" (this process's: every output handle starts with it),
//              "job": {"id"},
//              "python": {"state": "disabled|starting|ready|failed", "caps"?, "error"?}}
//
// Served here, in C++: ping, cancel, shutdown, the cluster datasets
// (dataset_info / _read / _view / _stats, core/dataset_service.hpp), with
// SIRIUS's own TIFF reader (nvTIFF on a CUDA device), and the application's
// pipelines (pipeline_run, step_preview, step_validate, output_stats,
// put_file, stat_file, outputs_release, cache_status: core/engine_node.hpp),
// whose outputs stay here and are drawn through dataset_* by their handles.
// Every other request is
// relayed verbatim to a Python worker the engine runs as its child on
// 127.0.0.1 with a token of its own (core/local_worker.hpp) -- progress,
// cancel and tensors included -- or, without one, refused with a message
// that says so. The child starts at once, in the background: the engine is
// ready without waiting for torch to import.

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "core/dataset_service.hpp"
#include "core/rpc.hpp"
#include "core/rpc_server.hpp"

namespace sirius::app {

    class EngineNode;

    struct EngineOptions {
        std::string token;                                  // the shared secret; "" only on a loopback address
        std::string host = "127.0.0.1";                     // what the announce line and hello report
        int maxClients = 8;
        std::string device = "auto";                        // auto | cpu | cuda | cuda:N: the default decode device
        std::chrono::milliseconds idleTimeout{3600000};
        std::string scratch;                                // node scratch: the step cache and uploads go in a folder of their own there ("" = the temp dir)
        long long viewCacheBytes = -1;                      // < 0: $SIRIUS_WORKER_VIEW_CACHE_MB, default 4 GiB
        // The Python worker for what the engine does not serve itself.
        bool pythonWorker = true;
        std::string python, workerDir;                      // "" = the usual search (core/local_worker.hpp)
        // Instead of starting a child: how to connect to a worker (tests).
        std::function<std::unique_ptr<RemoteWorker>(const std::function<bool()>& cancelled)> connectPython;
        std::function<void(const std::string&)> log;
        // Tests only: fields reported in the "engine" block in place of this
        // build's (an engine of other operations, refused at the hello).
        // `sirius-cli serve` takes them from $SIRIUS_TEST_ENGINE_BUILD.
        nlohmann::json buildOverride;
    };

    class EngineServer {
    public:
        explicit EngineServer(EngineOptions options);
        ~EngineServer();   // stops serving and the Python child
        EngineServer(const EngineServer&) = delete;
        EngineServer& operator=(const EngineServer&) = delete;

        // What `auth` answers with.
        nlohmann::json capabilities() const;
        // The one line `sirius-cli serve` prints once it listens:
        // {"port", "pid", "host", "hostname", "device", "engine": {build...}},
        // "port" first: the job's log is searched for ^{"port" (core/cluster.cpp).
        nlohmann::ordered_json announce(int port) const;
        // "cuda:0" or "cpu": the --device, resolved against the GPUs there are.
        std::string resolvedDevice() const;
        // {"state", "caps"?, "error"?} of the Python child.
        nlohmann::json pythonStatus() const;

        // One in-process connection, served on the calling thread (tests).
        void serveConnection(std::unique_ptr<rpc::Transport> transport, const std::string& peer = "loopback");
        // Accepts and serves until stop() (or a client's shutdown).
        void serve(rpc::Listener& listener);
        void stop();
        bool stopping() const noexcept;

        DatasetService& datasets() noexcept;
        EngineNode& node() noexcept;
        rpc::Server& server() noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_ENGINE_SERVER_HPP
