#ifndef SIRIUS_APP_LOCAL_WORKER_HPP
#define SIRIUS_APP_LOCAL_WORKER_HPP

// Starts the bundled Python worker as a child process the first time a step needs
// it and hands out connections (was gui::WorkerLauncher). Threading: connect() and
// stop() from any thread, one start at a time; the log handler runs on the stderr
// reader thread; the start-failure handler on the connecting thread with the start
// lock held (it may only record or post). Must outlive every thread that calls it.

#include <functional>
#include <memory>
#include <string>

#include "core/python_env.hpp"
#include "core/rpc.hpp"
#include "core/worker_error.hpp"

namespace sirius::app {
    class LocalWorker {
    public:
        LocalWorker();
        virtual ~LocalWorker();                                  // stop()
        LocalWorker(const LocalWorker&) = delete;
        LocalWorker& operator=(const LocalWorker&) = delete;
        void setPython(const std::string& python);              // explicit ("" = pyenv order)
        void setScriptDir(const std::string& dir);
        void setDevice(const std::string& device);              // auto | cpu | cuda | cuda:N
        void setAllowInstall(bool allow);                        // --allow-install; default false
        void setMaxClients(int clients);                         // --max-clients; default 1 (one connection at a time)
        void setConfiguredPython(std::function<std::string()> source);
        void setConfiguredScriptDir(std::function<std::string()> source);
        void setSetupHint(std::string hint);
        void setStartFailureHandler(std::function<void(const WorkerStartError&)> handler);
        void setLogHandler(std::function<void(const std::string& line)> handler);
        pyenv::Interpreter interpreter() const;
        std::string python() const;                              // the explicit value
        std::string scriptDir() const;
        std::string runningPython() const;
        // Throws WorkerStartError when the process does not come up, CancelledError when
        // `cancelled` ends the wait (also during the start) or a stop() from another
        // thread ends the start, std::runtime_error when a running worker refuses.
        std::unique_ptr<RemoteWorker> connect(const std::function<bool()>& cancelled = {});
        bool isRunning() const;
        int port() const noexcept;
        void stop();
        std::string lastLog() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace sirius::app

#endif // SIRIUS_APP_LOCAL_WORKER_HPP
