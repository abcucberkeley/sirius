#include "imgui/worker_launcher.hpp"

#include <chrono>
#include <exception>
#include <random>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "core/cancel.hpp"
#include "imgui/platform.hpp"
#include "imgui/settings.hpp"
#include "imgui/strings.hpp"

namespace sirius::app::gui {

    WorkerLauncher::WorkerLauncher() {
        // A per-session token so only this app talks to its worker.
        std::random_device rd;
        std::uniform_int_distribution<unsigned long long> dist;
        token_ = format("%llx%llx", dist(rd), dist(rd));
    }

    WorkerLauncher::~WorkerLauncher() { stop(); }

    void WorkerLauncher::setPython(const std::string& python) {
        const std::lock_guard<std::mutex> g(mutex_);
        python_ = python;
    }
    void WorkerLauncher::setScriptDir(const std::string& dir) {
        const std::lock_guard<std::mutex> g(mutex_);
        scriptDir_ = dir;
    }
    void WorkerLauncher::setDevice(const std::string& device) {
        const std::lock_guard<std::mutex> g(mutex_);
        device_ = device;
    }
    void WorkerLauncher::setLogHandler(std::function<void(const std::string& line)> handler) {
        const std::lock_guard<std::mutex> g(handlerMutex_);
        handler_ = std::move(handler);
    }

    std::string WorkerLauncher::python() const {
        {
            const std::lock_guard<std::mutex> g(mutex_);
            if (!python_.empty()) return python_;
        }
        // The environment first, as Preferences and RemoteConfig say.
        const std::string env = platform::environment("SIRIUS_PYTHON");
        if (!env.empty()) return env;
        const std::string fromSettings = trimmed(settings().getString("worker/python"));
        if (!fromSettings.empty()) return fromSettings;
        const std::string found = platform::findPython();
        if (!found.empty()) return found;
#ifdef _WIN32
        return "python";
#else
        return "python3";
#endif
    }

    std::string WorkerLauncher::scriptDir() const {
        {
            const std::lock_guard<std::mutex> g(mutex_);
            if (!scriptDir_.empty()) return scriptDir_;
        }
        const std::string fromSettings = settings().getString("worker/dir");
        if (!fromSettings.empty()) return fromSettings;
        // an installed tree, next to the executable (the build copies
        // app/python there), then the environment and the source tree
        return workerScriptPath();
    }

    std::string WorkerLauncher::lastLog() const {
        const std::lock_guard<std::mutex> g(mutex_);
        return log_;
    }

    void WorkerLauncher::appendLog(const std::string& text) {
        const std::lock_guard<std::mutex> g(mutex_);
        log_ += text;
        if (log_.size() > 20000) log_ = log_.substr(log_.size() - 10000);
    }

    WorkerLauncher::Launch WorkerLauncher::launchSettings() const {
        Launch cfg;
        cfg.python = python();
        cfg.dir = scriptDir();
        const std::lock_guard<std::mutex> g(mutex_);
        cfg.device = device_;
        cfg.token = token_;
        return cfg;
    }

    bool WorkerLauncher::isRunning() const { return running_.load() && port_.load() > 0; }

    void WorkerLauncher::stopLocked() {
        if (process_) process_->stop();
        process_.reset();
        port_.store(0);
        running_.store(false);
    }

    void WorkerLauncher::start() {
        stopLocked();   // a dead or half-started process from before
        const Launch cfg = launchSettings();
        if (cfg.dir.empty() || !isFile(cfg.dir + "/sirius_worker/__main__.py"))
            throw std::runtime_error("the Python worker (sirius_worker) was not found next to the application; "
                                     "set Preferences \xE2\x96\xB8 Worker \xE2\x96\xB8 directory");
        {
            const std::lock_guard<std::mutex> g(mutex_);
            log_.clear();
        }
        process_ = std::make_unique<ChildProcess>();
        process_->setErrorHandler([this](const std::string& line) {
            appendLog(line + "\n");
            if (line.empty()) return;
            // Called with the lock held, not on a copy: the worker still
            // writes while it stops at exit, and a copy could post into the
            // Bridge after main() has detached the handler and destroyed it.
            // A separate lock, so that the handler may still ask lastLog().
            const std::lock_guard<std::mutex> g(handlerMutex_);
            if (handler_) handler_(line);
        });
        ChildProcess::Options o;
        o.program = cfg.python;
        // --exit-with-parent: the worker holds our end of its stdin pipe and
        // stops when it closes, so a crash here leaves no orphan holding the GPU.
        o.arguments = {"-m", "sirius_worker", "--host", "127.0.0.1", "--port", "0", "--device", cfg.device,
                       "--exit-with-parent", "--allow-install"};   // the model hub may install packages
        o.workingDirectory = cfg.dir;
        // The shared secret goes through the environment: a command line is
        // readable by every user of the machine (ps, /proc), the environment
        // of a process only by its owner. The Hugging Face token is not put
        // there: every request that needs it carries it (SECURITY.md).
        o.environment = {{"SIRIUS_TOKEN", cfg.token}, {"PYTHONUNBUFFERED", "1"}};
        std::string error;
        if (!process_->start(o, &error)) {
            process_.reset();
            throw std::runtime_error("cannot start " + cfg.python + ": " + error);
        }
        // The worker prints one JSON line with its port once it listens.
        std::string line;
        int port = 0;
        while (process_->readLine(line, 60000)) {
            line = trimmed(line);
            if (line.empty()) continue;
            const nlohmann::json j = nlohmann::json::parse(line, nullptr, false);
            if (j.is_object() && j.contains("port") && j["port"].is_number_integer()) port = j["port"].get<int>();
            break;
        }
        if (port <= 0) {
            stopLocked();   // joins the stderr reader: the log is complete
            const std::string log = trimmed(lastLog());
            throw std::runtime_error("the Python worker did not start: " + (log.size() > 800 ? log.substr(log.size() - 800) : log));
        }
        port_.store(port);
        running_.store(true);
    }

    std::unique_ptr<RemoteWorker> WorkerLauncher::connect(const std::function<bool()>& cancelled) {
        const std::lock_guard<std::mutex> lock(processMutex_);
        if (process_ && !process_->running()) {   // it died between runs
            port_.store(0);
            running_.store(false);
        }
        if (!isRunning()) start();
        const auto timeout = std::chrono::seconds(5);
        try {
            return RemoteWorker::connect("127.0.0.1", port_.load(), token_, timeout, cancelled);
        } catch (const CancelledError&) {
            throw;   // the caller stopped waiting; the worker is fine
        } catch (const std::exception&) {
            // The process may have died between runs: start once more. One
            // that still runs answered and refused, or did not answer before
            // the handshake's deadline (still starting, or serving another
            // client); starting it again would not help, and would end that
            // start or that client's work.
            if (process_ && process_->running()) throw;
            start();
            return RemoteWorker::connect("127.0.0.1", port_.load(), token_, timeout, cancelled);
        }
    }

    void WorkerLauncher::stop() {
        const std::lock_guard<std::mutex> lock(processMutex_);
        stopLocked();
    }

} // namespace sirius::app::gui
