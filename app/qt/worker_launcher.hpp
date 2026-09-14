#ifndef SIRIUS_APP_WORKER_LAUNCHER_HPP
#define SIRIUS_APP_WORKER_LAUNCHER_HPP

// Starts the bundled Python worker (app/python/sirius_worker) as a child
// process the first time a step needs it and hands out connections to it.
// The process is kept for the rest of the session: loading torch and a
// model takes seconds, connecting takes milliseconds. Installed into the
// workbench through Workbench::setLocalWorkerLauncher.
//
// Threading: the QProcess is created and driven on a thread of the
// launcher's own (a QProcess must be used from the thread that owns it),
// so connect() and stop() may be called from any other thread -- a run's
// worker thread, the model hub's thread, the GUI -- and block the caller
// while that thread starts or stops the process. The RemoteWorker a call
// returns is created on the calling thread and belongs to it. The signals
// are emitted from the launcher's thread; receivers elsewhere get them
// queued. The launcher must outlive every thread that may still call it.

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>

#include <QObject>
#include <QString>
#include <QThread>

#include "core/rpc.hpp"

namespace sirius::app {

    class WorkerLauncher : public QObject {
        Q_OBJECT
    public:
        explicit WorkerLauncher(QObject* parent = nullptr);
        ~WorkerLauncher() override;

        // Interpreter and worker directory; empty = $SIRIUS_PYTHON (then
        // QSettings "worker/python", then "python3") and the directory next
        // to the executable / the source tree.
        void setPython(const QString& python);
        void setScriptDir(const QString& dir);
        void setDevice(const QString& device);   // "auto", "cuda", "cpu"
        QString python() const;
        QString scriptDir() const;

        // Starts the process when needed and connects; throws std::runtime_error
        // with the worker's stderr when it fails to come up. Blocks the caller
        // for the start-up (up to about a minute) but never the GUI thread
        // unless it is the caller.
        std::unique_ptr<RemoteWorker> connect();
        bool isRunning() const;
        int port() const noexcept { return port_.load(); }
        void stop();
        QString lastLog() const;

    signals:
        void started(int port);
        void stopped();
        void logged(const QString& line);

    private:
        class Host;   // owns the QProcess, lives on thread_
        friend class Host;
        struct Launch {
            QString python, dir, device, token;
        };
        Launch launchSettings() const;
        void start();
        // Runs `fn` on thread_ and waits; an exception it throws is rethrown here.
        void runOnHost(const std::function<void()>& fn);

        QThread thread_;
        Host* host_ = nullptr;
        mutable std::mutex mutex_;                // python_, scriptDir_, device_, log_
        QString python_;
        QString scriptDir_;
        QString device_ = QStringLiteral("auto");
        QString token_;                           // fixed at construction
        QString log_;
        std::atomic<int> port_{0};
        std::atomic<bool> running_{false};
    };

} // namespace sirius::app

#endif // SIRIUS_APP_WORKER_LAUNCHER_HPP
