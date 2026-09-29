#include "core/local_worker.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <random>
#include <thread>
#include <utility>

#include <nlohmann/json.hpp>

#include "core/cancel.hpp"
#include "core/host.hpp"
#include "core/process.hpp"

namespace sirius::app {

    namespace {
        // How long the worker may take to print its port: importing torch
        // from a cluster's network filesystem is the slow case.
        constexpr auto kPortTimeout = std::chrono::seconds(60);
        // The wait for that line is cut into slices this long, so that a
        // cancel ends it at once instead of after the minute.
        constexpr int kSliceMs = 200;
        // How long a worker that printed something else than its port gets to
        // exit before it is stopped (the missing_packages line comes last).
        constexpr int kExitGraceMs = 3000;
        // How long a worker gets to leave by itself once its stdin has closed.
        constexpr int kStopGraceMs = 3000;
        // How long `<python> -I -c "import sys"` may take.
        constexpr int kBasicRunMs = 5000;

        std::string trimmed(const std::string& s) {
            const auto space = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
            std::size_t b = 0, e = s.size();
            while (b < e && space(s[b])) ++b;
            while (e > b && space(s[e - 1])) --e;
            return s.substr(b, e - b);
        }

        // A per-launcher token, so only this process talks to its worker.
        std::string newToken() {
            std::random_device rd;
            std::uniform_int_distribution<unsigned long long> dist;
            char buffer[40];
            std::snprintf(buffer, sizeof buffer, "%llx%llx", dist(rd), dist(rd));
            return buffer;
        }

        // The line the worker prints once it listens: {"port": N, ...}.
        int portOf(const std::string& line) {
            const nlohmann::json j = nlohmann::json::parse(line, nullptr, false);
            if (j.is_object() && j.contains("port") && j["port"].is_number_integer()) return j["port"].get<int>();
            return 0;
        }

        // How a wait for a process to leave ended.
        enum class Waited { Exited,
                            TimedOut,
                            Cancelled };

        // Waits up to `timeoutMs` for `p` to leave, in the slices of the wait
        // for the port line: a cancel is seen within one of them here too.
        Waited waitForExit(ChildProcess& p, int timeoutMs, const std::function<bool()>& cancelled) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
            for (;;) {
                if (cancelled && cancelled()) return Waited::Cancelled;
                const long long left =
                    std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
                if (p.waitForExit(left <= 0 ? 0 : static_cast<int>(std::min<long long>(left, kSliceMs)))) return Waited::Exited;
                if (left <= kSliceMs) return Waited::TimedOut;
            }
        }

        // `<python> -I -c "import sys"`: whether the interpreter runs at all,
        // apart from the worker and whatever the environment says (-I).
        // Throws CancelledError when `cancelled` ends the wait.
        bool basicRunFails(const std::string& python, const std::function<bool()>& cancelled) {
            ChildProcess p;
            ChildProcess::Options o;
            o.program = python;
            o.arguments = {"-I", "-c", "import sys"};
            // Like the worker: a console Ctrl+C would otherwise end this run
            // too, and its exit code would read as a broken interpreter.
            o.ownProcessGroup = true;
            if (!p.start(o)) return true;
            const Waited waited = waitForExit(p, kBasicRunMs, cancelled);
            const int code = p.exitCode();
            p.stop(0);
            if (waited == Waited::Cancelled) throw CancelledError();
            return waited != Waited::Exited || code != 0;
        }
    } // namespace

    struct LocalWorker::Impl {
        // What one start uses, read once so that a setter called meanwhile
        // does not change a start half-way.
        struct Launch {
            pyenv::Interpreter interpreter;
            std::string dir, device, hint;
            bool allowInstall = false;
        };

        const std::string token = newToken();   // fixed for the launcher's life

        mutable std::mutex mutex;   // the settings, the log and runningPython
        std::string python, scriptDir, device = "auto", setupHint;
        bool allowInstall = false;
        std::function<std::string()> configuredPython, configuredScriptDir;
        std::string log, runningPython;

        // Each handler is called with its own lock held, so that replacing it
        // waits for a call in progress: the caller may destroy what the old
        // one captured right after (see main.cpp).
        std::mutex failureMutex;
        std::function<void(const WorkerStartError&)> failureHandler;
        std::mutex logMutex;
        std::function<void(const std::string&)> logHandler;

        std::mutex processMutex;   // one start / stop at a time; guards process
        // stop() calls waiting for processMutex. A start in progress sees them
        // as a cancel, so that stop() from another thread (the Preferences'
        // environment removal, the host shutting down) does not wait for the
        // whole port timeout of a start that it is about to end anyway.
        std::atomic<int> stopsWaiting{0};
        std::unique_ptr<ChildProcess> process;
        std::atomic<int> port{0};
        std::atomic<bool> running{false};

        std::string lastLog() const {
            const std::lock_guard<std::mutex> g(mutex);
            return log;
        }

        void appendLog(const std::string& text) {
            const std::lock_guard<std::mutex> g(mutex);
            log += text;
            if (log.size() > 20000) log = log.substr(log.size() - 10000);
        }

        // Caller holds processMutex. The exit code of what was stopped, -1 for nothing.
        int stopLocked(int graceMs = kStopGraceMs) {
            int code = -1;
            if (process) {
                process->stop(graceMs);   // joins the stderr reader: the log is complete
                code = process->exitCode();
            }
            process.reset();
            port.store(0);
            running.store(false);
            const std::lock_guard<std::mutex> g(mutex);
            runningPython.clear();
            return code;
        }

        // Hands the error to the host's handler, then throws it. Caller holds processMutex.
        [[noreturn]] void fail(WorkerStartError e, const Launch& cfg) {
            if (e.setupWouldHelp()) e.hint = cfg.hint;
            {
                const std::lock_guard<std::mutex> g(failureMutex);
                if (failureHandler) failureHandler(e);
            }
            throw e;
        }

        // A cancel during a start: the process never served, so there is
        // nothing to wait for, and it is no failure to start either (the
        // failure handler does not hear of it). Caller holds processMutex.
        [[noreturn]] void cancelStart() {
            stopLocked(0);
            throw CancelledError();
        }

        void start(const Launch& cfg, const std::function<bool()>& cancelled);
    };

    void LocalWorker::Impl::start(const Launch& cfg, const std::function<bool()>& cancelled) {
        stopLocked();   // a dead or half-started process from before
        {
            const std::lock_guard<std::mutex> g(mutex);
            log.clear();
        }
        StartFailure failure;
        failure.interpreter = cfg.interpreter.path;
        failure.source = pyenv::toString(cfg.interpreter.source);

        if (cfg.dir.empty() || !host::isFile(cfg.dir + "/sirius_worker/__main__.py")) {
            const std::string where = cfg.dir.empty()
                                          ? std::string("next to the application; SIRIUS_WORKER_DIR may name their directory")
                                          : "in " + cfg.dir;
            WorkerStartError e(WorkerStartError::Kind::NoWorkerScripts,
                               "the Python worker cannot start: its scripts (sirius_worker/__main__.py) were not found " + where);
            e.interpreter = failure.interpreter;
            e.source = failure.source;
            fail(std::move(e), cfg);
        }

        process = std::make_unique<ChildProcess>();
        process->setErrorHandler([this](const std::string& line) {
            appendLog(line + "\n");
            if (line.empty()) return;
            // Called with the lock held, not on a copy: the worker still
            // writes while it stops at exit, and a copy could post into the
            // host after it has detached the handler and destroyed its target.
            const std::lock_guard<std::mutex> g(logMutex);
            if (logHandler) logHandler(line);
        });
        ChildProcess::Options o;
        o.program = cfg.interpreter.path;
        // --exit-with-parent: the worker holds our end of its stdin pipe and
        // stops when it closes, so a crash here leaves no orphan holding the GPU.
        o.arguments = {"-m", "sirius_worker", "--host", "127.0.0.1", "--port", "0", "--device", cfg.device, "--exit-with-parent"};
        // Installing packages into the interpreter is the model hub's (the
        // GUI's); a worker nobody asked to install for does not accept it.
        if (cfg.allowInstall) o.arguments.emplace_back("--allow-install");
        o.workingDirectory = cfg.dir;
        // The shared secret goes through the environment: a command line is
        // readable by every user of the machine (ps, /proc), the environment
        // of a process only by its owner. The Hugging Face token is not put
        // there: every request that needs it carries it (SECURITY.md).
        o.environment = {{"SIRIUS_TOKEN", token}, {"PYTHONUNBUFFERED", "1"}};
        // A PYTHONHOME meant for another Python would break SIRIUS's own
        // environment, and nobody set it for that one; CPython ignores an
        // empty value.
        if (cfg.interpreter.source == pyenv::Source::Managed) o.environment.emplace_back("PYTHONHOME", "");
        // Its own process group: a Ctrl+C in sirius-cli's console cancels the
        // run and leaves the worker alone (--exit-with-parent still ends it
        // with its parent).
        o.ownProcessGroup = true;

        std::string error;
        if (!process->start(o, &error)) {
            failure.startError = error.empty() ? std::string("unknown error") : error;
            failure.programNotFound = process->programNotFound();
            process.reset();
            fail(classifyStartFailure(failure), cfg);
        }

        // The worker prints one JSON line with its port once it listens.
        const auto deadline = std::chrono::steady_clock::now() + kPortTimeout;
        std::string line;
        int workerPort = 0;
        bool exited = false;
        for (;;) {
            if (cancelled && cancelled()) cancelStart();
            const auto sliceStart = std::chrono::steady_clock::now();
            if (process->readLine(line, exited ? 500 : kSliceMs)) {
                line = trimmed(line);
                if (line.empty()) continue;
                workerPort = portOf(line);
                if (workerPort <= 0) failure.firstStdoutLine = line;
                break;
            }
            if (exited) break;   // it ended, and nothing more came
            if (!process->running()) {
                exited = true;   // what it printed last may still be on its way from the reader
                continue;
            }
            if (std::chrono::steady_clock::now() >= deadline) break;
            // readLine answers at once when stdout has closed while the
            // process runs on; the slice is then slept here instead.
            const auto halfSlice = std::chrono::milliseconds(kSliceMs / 2);
            if (std::chrono::steady_clock::now() - sliceStart < halfSlice) std::this_thread::sleep_for(halfSlice);
        }

        if (workerPort > 0) {
            port.store(workerPort);
            running.store(true);
            const std::lock_guard<std::mutex> g(mutex);
            runningPython = cfg.interpreter.path;
            return;
        }

        // It did not come up: find out why, from what it left behind. Every
        // wait for that polls `cancelled` as well, so that a cancel coming
        // now still ends the start within a slice, and as a cancel.
        if (!exited && !failure.firstStdoutLine.empty()) {
            const Waited waited = waitForExit(*process, kExitGraceMs, cancelled);
            if (waited == Waited::Cancelled) cancelStart();
            exited = waited == Waited::Exited;
        }
        failure.exitedBeforePort = exited;
        if (!exited) {
            // What stop() does with its grace, in slices: the closed stdin
            // makes the worker (--exit-with-parent) leave by itself.
            process->closeInput();
            if (waitForExit(*process, kStopGraceMs, cancelled) == Waited::Cancelled) cancelStart();
        }
        failure.exitCode = stopLocked(0);   // it has left, or is ended now
        failure.stderrLog = lastLog();
        if (cfg.interpreter.source == pyenv::Source::Managed && failure.exitedBeforePort &&
            failure.firstStdoutLine.find("missing_packages") == std::string::npos)
            failure.basicRunFailed = basicRunFails(cfg.interpreter.path, cancelled);
        // A cancel that came during the last slice of one of those waits is a
        // cancel as well, not a failure to start.
        if (cancelled && cancelled()) throw CancelledError();
        fail(classifyStartFailure(failure), cfg);
    }

    LocalWorker::LocalWorker() : impl_(std::make_unique<Impl>()) {}

    LocalWorker::~LocalWorker() { stop(); }

    void LocalWorker::setPython(const std::string& python) {
        const std::lock_guard<std::mutex> g(impl_->mutex);
        impl_->python = python;
    }

    void LocalWorker::setScriptDir(const std::string& dir) {
        const std::lock_guard<std::mutex> g(impl_->mutex);
        impl_->scriptDir = dir;
    }

    void LocalWorker::setDevice(const std::string& device) {
        const std::lock_guard<std::mutex> g(impl_->mutex);
        impl_->device = device;
    }

    void LocalWorker::setAllowInstall(bool allow) {
        const std::lock_guard<std::mutex> g(impl_->mutex);
        impl_->allowInstall = allow;
    }

    void LocalWorker::setConfiguredPython(std::function<std::string()> source) {
        const std::lock_guard<std::mutex> g(impl_->mutex);
        impl_->configuredPython = std::move(source);
    }

    void LocalWorker::setConfiguredScriptDir(std::function<std::string()> source) {
        const std::lock_guard<std::mutex> g(impl_->mutex);
        impl_->configuredScriptDir = std::move(source);
    }

    void LocalWorker::setSetupHint(std::string hint) {
        const std::lock_guard<std::mutex> g(impl_->mutex);
        impl_->setupHint = std::move(hint);
    }

    void LocalWorker::setStartFailureHandler(std::function<void(const WorkerStartError&)> handler) {
        const std::lock_guard<std::mutex> g(impl_->failureMutex);
        impl_->failureHandler = std::move(handler);
    }

    void LocalWorker::setLogHandler(std::function<void(const std::string& line)> handler) {
        const std::lock_guard<std::mutex> g(impl_->logMutex);
        impl_->logHandler = std::move(handler);
    }

    pyenv::Interpreter LocalWorker::interpreter() const {
        std::string explicitPython;
        std::function<std::string()> configured;
        {
            const std::lock_guard<std::mutex> g(impl_->mutex);
            explicitPython = impl_->python;
            configured = impl_->configuredPython;
        }
        // Outside the lock: the host's source may take its own (the GUI's settings).
        return pyenv::workerInterpreter(explicitPython, configured ? configured() : std::string());
    }

    std::string LocalWorker::python() const {
        const std::lock_guard<std::mutex> g(impl_->mutex);
        return impl_->python;
    }

    std::string LocalWorker::scriptDir() const {
        std::string dir;
        std::function<std::string()> configured;
        {
            const std::lock_guard<std::mutex> g(impl_->mutex);
            dir = impl_->scriptDir;
            configured = impl_->configuredScriptDir;
        }
        if (!dir.empty()) return dir;
        if (configured) {
            dir = configured();
            if (!dir.empty()) return dir;
        }
        // an installed tree, next to the executable (the build copies
        // app/python there), then the environment and the source tree
        return workerScriptPath();
    }

    std::string LocalWorker::runningPython() const {
        const std::lock_guard<std::mutex> g(impl_->mutex);
        return impl_->runningPython;
    }

    std::unique_ptr<RemoteWorker> LocalWorker::connect(const std::function<bool()>& callerCancelled) {
        const std::lock_guard<std::mutex> lock(impl_->processMutex);
        const std::function<bool()> cancelled = [this, &callerCancelled] {
            return impl_->stopsWaiting.load() > 0 || (callerCancelled && callerCancelled());
        };
        const auto launch = [this] {
            Impl::Launch cfg;
            cfg.interpreter = interpreter();
            cfg.dir = scriptDir();
            const std::lock_guard<std::mutex> g(impl_->mutex);
            cfg.device = impl_->device;
            cfg.hint = impl_->setupHint;
            cfg.allowInstall = impl_->allowInstall;
            return cfg;
        };
        if (impl_->process && !impl_->process->running()) {   // it died between runs
            impl_->port.store(0);
            impl_->running.store(false);
        }
        if (!isRunning()) impl_->start(launch(), cancelled);
        const auto timeout = std::chrono::seconds(5);
        try {
            return RemoteWorker::connect("127.0.0.1", impl_->port.load(), impl_->token, timeout, cancelled);
        } catch (const CancelledError&) {
            throw;   // the caller stopped waiting; the worker is fine
        } catch (const std::exception&) {
            // The process may have died between runs: start once more. One
            // that still runs answered and refused, or did not answer before
            // the handshake's deadline (still starting, or serving another
            // client); starting it again would not help, and would end that
            // start or that client's work.
            if (impl_->process && impl_->process->running()) throw;
            impl_->start(launch(), cancelled);
            return RemoteWorker::connect("127.0.0.1", impl_->port.load(), impl_->token, timeout, cancelled);
        }
    }

    bool LocalWorker::isRunning() const { return impl_->running.load() && impl_->port.load() > 0; }

    int LocalWorker::port() const noexcept { return impl_->port.load(); }

    void LocalWorker::stop() {
        impl_->stopsWaiting.fetch_add(1);
        const std::lock_guard<std::mutex> lock(impl_->processMutex);
        impl_->stopsWaiting.fetch_sub(1);
        impl_->stopLocked();
    }

    std::string LocalWorker::lastLog() const { return impl_->lastLog(); }

} // namespace sirius::app
