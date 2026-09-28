#ifndef SIRIUS_IMGUI_WORKER_LAUNCHER_HPP
#define SIRIUS_IMGUI_WORKER_LAUNCHER_HPP

// Starts the bundled Python worker (app/python/sirius_worker) as a child
// process the first time a step needs it and hands out connections to it.
// The process is kept for the rest of the session: loading torch and a
// model takes seconds, connecting takes milliseconds. Installed into the
// workbench through Workbench::setLocalWorkerLauncher.
//
// Threading: connect() and stop() may be called from any thread -- a run's
// worker thread, the model hub's thread, the GUI -- and block the caller
// while the process starts or stops (one at a time). The RemoteWorker a
// call returns belongs to the calling thread. The log handler is called on
// the process's stderr reader thread. The launcher must outlive every
// thread that may still call it.

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "core/rpc.hpp"
#include "imgui/process.hpp"

namespace sirius::app::gui {

    class WorkerLauncher {
    public:
        WorkerLauncher();
        ~WorkerLauncher();
        WorkerLauncher(const WorkerLauncher&) = delete;
        WorkerLauncher& operator=(const WorkerLauncher&) = delete;

        // Interpreter and worker directory; empty = $SIRIUS_PYTHON (then the
        // setting "worker/python", then "python3") and the directory next
        // to the executable / the source tree.
        void setPython(const std::string& python);
        void setScriptDir(const std::string& dir);
        void setDevice(const std::string& device);   // "auto", "cuda", "cpu"
        std::string python() const;
        std::string scriptDir() const;

        // Every line the worker writes to stderr (any thread).
        void setLogHandler(std::function<void(const std::string& line)> handler);

        // Starts the process when needed and connects; throws std::runtime_error
        // with the worker's stderr when it fails to come up. Blocks the caller
        // for the start-up (up to about a minute).
        std::unique_ptr<RemoteWorker> connect();
        bool isRunning() const;
        int port() const noexcept { return port_.load(); }
        void stop();
        std::string lastLog() const;

    private:
        struct Launch {
            std::string python, dir, device, token;
        };
        Launch launchSettings() const;
        void start();                        // caller holds processMutex_
        void stopLocked();                   // caller holds processMutex_
        void appendLog(const std::string& text);

        std::mutex processMutex_;            // one start / stop at a time
        std::unique_ptr<ChildProcess> process_;
        mutable std::mutex mutex_;           // python_, scriptDir_, device_, log_, handler_
        std::string python_;
        std::string scriptDir_;
        std::string device_ = "auto";
        std::string token_;                  // fixed at construction
        std::string log_;
        std::function<void(const std::string&)> handler_;
        std::atomic<int> port_{0};
        std::atomic<bool> running_{false};
    };

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_WORKER_LAUNCHER_HPP
