#include "qt/worker_launcher.hpp"

#include <exception>
#include <stdexcept>

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMetaObject>
#include <QProcess>
#include <QProcessEnvironment>
#include <QRandomGenerator>
#include <QSettings>
#include <QStandardPaths>

#include "qt/qt_strings.hpp"
#include "qt/secret_store.hpp"

namespace sirius::app {

    // The object that owns the process. Everything in here runs on the
    // launcher's thread (runOnHost); the launcher's atomics and mutex are
    // how the outcome reaches the other threads.
    class WorkerLauncher::Host final : public QObject {
    public:
        explicit Host(WorkerLauncher& launcher) : launcher_(launcher) {}
        ~Host() override { stop(); }

        void start(const Launch& cfg) {
            stop();   // a dead or half-started process from before
            if (cfg.dir.isEmpty() || !QFileInfo::exists(cfg.dir + QStringLiteral("/sirius_worker/__main__.py")))
                throw std::runtime_error("the Python worker (sirius_worker) was not found next to the application; "
                                         "set Preferences ▸ Worker ▸ directory");
            process_ = std::make_unique<QProcess>();
            process_->setWorkingDirectory(cfg.dir);
            process_->setProcessChannelMode(QProcess::SeparateChannels);
            // the Hugging Face token from Preferences (kept in the secret store,
            // not in QSettings) reaches huggingface_hub as HF_TOKEN
            QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
            const QString hfToken = secrets::read(QStringLiteral("hub/token")).trimmed();
            if (!hfToken.isEmpty() && !env.contains(QStringLiteral("HF_TOKEN"))) env.insert(QStringLiteral("HF_TOKEN"), hfToken);
            // The shared secret goes through the environment: a command line
            // is readable by every user of the machine (ps, /proc), the
            // environment of a process only by its owner.
            env.insert(QStringLiteral("SIRIUS_TOKEN"), cfg.token);
            process_->setProcessEnvironment(env);
            // --exit-with-parent: the worker holds our end of its stdin pipe
            // and stops when it closes, so a crash here leaves no orphan
            // holding the GPU.
            QStringList args{QStringLiteral("-m"), QStringLiteral("sirius_worker"), QStringLiteral("--host"),
                             QStringLiteral("127.0.0.1"), QStringLiteral("--port"), QStringLiteral("0"),
                             QStringLiteral("--device"), cfg.device, QStringLiteral("--exit-with-parent"),
                             QStringLiteral("--allow-install")};   // the model hub may install packages
            {
                std::lock_guard<std::mutex> g(launcher_.mutex_);
                launcher_.log_.clear();
            }
            QObject::connect(process_.get(), &QProcess::readyReadStandardError, this, [this] {
                const QString text = QString::fromUtf8(process_->readAllStandardError());
                appendLog(text);
                for (const QString& line : text.split('\n', Qt::SkipEmptyParts)) emit launcher_.logged(line);
            });
            QObject::connect(process_.get(), qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
                             [this](int, QProcess::ExitStatus) {
                                 launcher_.port_.store(0);
                                 launcher_.running_.store(false);
                                 emit launcher_.stopped();
                             });
            process_->start(cfg.python, args);
            if (!process_->waitForStarted(5000))
                throw std::runtime_error("cannot start " + toStd(cfg.python) + ": " + toStd(process_->errorString()));
            // The worker prints one JSON line with its port once it listens.
            QByteArray line;
            while (line.isEmpty() && process_->state() == QProcess::Running) {
                if (!process_->waitForReadyRead(60000)) break;
                while (process_->canReadLine()) {
                    line = process_->readLine().trimmed();
                    if (!line.isEmpty()) break;
                }
            }
            const QJsonObject obj = QJsonDocument::fromJson(line).object();
            const int port = obj.value(QStringLiteral("port")).toInt(0);
            if (port <= 0) {
                const QString err = QString::fromUtf8(process_->readAllStandardError());
                appendLog(err);
                const QString log = launcher_.lastLog();
                stop();
                throw std::runtime_error("the Python worker did not start: " + toStd(log.trimmed().right(800)));
            }
            launcher_.port_.store(port);
            launcher_.running_.store(true);
            emit launcher_.started(port);
        }

        void stop() {
            if (!process_) return;
            if (process_->state() == QProcess::Running) {
                process_->terminate();
                if (!process_->waitForFinished(3000)) {
                    process_->kill();
                    process_->waitForFinished(1000);
                }
            }
            process_.reset();
            launcher_.port_.store(0);
            launcher_.running_.store(false);
        }

    private:
        void appendLog(const QString& text) {
            std::lock_guard<std::mutex> g(launcher_.mutex_);
            launcher_.log_ += text;
            if (launcher_.log_.size() > 20000) launcher_.log_ = launcher_.log_.right(10000);
        }

        WorkerLauncher& launcher_;
        std::unique_ptr<QProcess> process_;
    };

    WorkerLauncher::WorkerLauncher(QObject* parent) : QObject(parent) {
        // A per-session token so only this app talks to its worker.
        token_ = QString::number(QRandomGenerator::global()->generate64(), 16) +
                 QString::number(QRandomGenerator::global()->generate64(), 16);
        host_ = new Host(*this);
        host_->moveToThread(&thread_);
        thread_.setObjectName(QStringLiteral("sirius-worker-launcher"));
        thread_.start();
    }

    WorkerLauncher::~WorkerLauncher() {
        stop();
        runOnHost([h = host_] { delete h; });
        host_ = nullptr;
        thread_.quit();
        thread_.wait();
    }

    void WorkerLauncher::setPython(const QString& python) {
        std::lock_guard<std::mutex> g(mutex_);
        python_ = python;
    }
    void WorkerLauncher::setScriptDir(const QString& dir) {
        std::lock_guard<std::mutex> g(mutex_);
        scriptDir_ = dir;
    }
    void WorkerLauncher::setDevice(const QString& device) {
        std::lock_guard<std::mutex> g(mutex_);
        device_ = device;
    }

    QString WorkerLauncher::python() const {
        {
            std::lock_guard<std::mutex> g(mutex_);
            if (!python_.isEmpty()) return python_;
        }
        const QString fromSettings = QSettings().value(QStringLiteral("worker/python")).toString();
        if (!fromSettings.isEmpty()) return fromSettings;
        const QByteArray env = qgetenv("SIRIUS_PYTHON");
        if (!env.isEmpty()) return QString::fromLocal8Bit(env);
        return QStringLiteral("python3");
    }

    QString WorkerLauncher::scriptDir() const {
        {
            std::lock_guard<std::mutex> g(mutex_);
            if (!scriptDir_.isEmpty()) return scriptDir_;
        }
        const QString fromSettings = QSettings().value(QStringLiteral("worker/dir")).toString();
        if (!fromSettings.isEmpty()) return fromSettings;
        // next to the executable (the build copies app/python there), then the source tree
        const QString beside = QCoreApplication::applicationDirPath() + QStringLiteral("/python");
        if (QFileInfo::exists(beside + QStringLiteral("/sirius_worker/__main__.py"))) return beside;
        return fromStd(workerScriptPath());
    }

    QString WorkerLauncher::lastLog() const {
        std::lock_guard<std::mutex> g(mutex_);
        return log_;
    }

    WorkerLauncher::Launch WorkerLauncher::launchSettings() const {
        Launch cfg;
        cfg.python = python();
        cfg.dir = scriptDir();
        std::lock_guard<std::mutex> g(mutex_);
        cfg.device = device_;
        cfg.token = token_;
        return cfg;
    }

    bool WorkerLauncher::isRunning() const { return running_.load() && port_.load() > 0; }

    void WorkerLauncher::runOnHost(const std::function<void()>& fn) {
        if (!host_) return;
        if (QThread::currentThread() == &thread_) {   // a blocking call onto itself would deadlock
            fn();
            return;
        }
        std::exception_ptr error;
        QMetaObject::invokeMethod(
            host_,
            [&fn, &error] {
                try {
                    fn();
                } catch (...) {
                    error = std::current_exception();
                }
            },
            Qt::BlockingQueuedConnection);
        if (error) std::rethrow_exception(error);
    }

    void WorkerLauncher::start() {
        const Launch cfg = launchSettings();
        runOnHost([this, &cfg] { host_->start(cfg); });
    }

    std::unique_ptr<RemoteWorker> WorkerLauncher::connect() {
        if (!isRunning()) start();
        const std::string token = toStd(token_);
        try {
            return RemoteWorker::connect("127.0.0.1", port_.load(), token);
        } catch (const std::exception&) {
            // the process may have died between runs: start once more
            stop();
            start();
            return RemoteWorker::connect("127.0.0.1", port_.load(), token);
        }
    }

    void WorkerLauncher::stop() {
        runOnHost([this] { host_->stop(); });
    }

} // namespace sirius::app
