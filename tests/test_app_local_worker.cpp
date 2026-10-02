// The worker launcher (app/core/local_worker.hpp) and why a worker did not
// come up (app/core/worker_error.hpp).
//
// The classification is pure and runs everywhere. The cases that start a
// process use the fake worker in tests/data/fake_worker_missing, which needs
// only the standard library, with a Python from $SIRIUS_PYTHON or else the
// one host::findPython finds; without either they skip. Only the last case
// starts the real worker, and only with $SIRIUS_PYTHON (which must have
// numpy). Every case that could reach SIRIUS's own environment points
// $SIRIUS_PYTHON_ENV into a temporary directory first: the developer's own
// environment is never looked at.

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "core/cancel.hpp"
#include "core/host.hpp"
#include "core/local_worker.hpp"
#include "core/process.hpp"
#include "core/python_env.hpp"
#include "core/worker_error.hpp"
#include "temp_path.hpp"

using namespace sirius::app;
using Kind = WorkerStartError::Kind;

namespace {

    namespace fs = std::filesystem;

    bool has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

    // An environment variable of this process; nullopt when it is not set.
    std::optional<std::string> environmentValue(const char* name) {
#ifdef _WIN32
        char* value = nullptr;
        std::size_t size = 0;
        if (_dupenv_s(&value, &size, name) != 0 || !value) return std::nullopt;
        std::string copy(value);
        std::free(value);
        return copy;
#else
        const char* value = std::getenv(name);
        return value ? std::optional<std::string>(value) : std::nullopt;
#endif
    }

    // Sets an environment variable (nullopt: removes it) for the scope, then
    // puts back what was there.
    class ScopedEnv {
    public:
        ScopedEnv(const char* name, std::optional<std::string> value) : name_(name), old_(environmentValue(name)) { set(value); }
        ~ScopedEnv() { set(old_); }
        ScopedEnv(const ScopedEnv&) = delete;
        ScopedEnv& operator=(const ScopedEnv&) = delete;

    private:
        void set(const std::optional<std::string>& value) const {
#ifdef _WIN32
            _putenv_s(name_, value ? value->c_str() : "");   // "" removes it
#else
            if (value) ::setenv(name_, value->c_str(), 1);
            else ::unsetenv(name_);
#endif
        }
        const char* name_;
        std::optional<std::string> old_;
    };

    fs::path madeDirectory(fs::path dir) {
        fs::create_directories(dir);
        return dir;
    }

    // A temporary directory with SIRIUS's own environment pointed into it,
    // no fake-worker mode left over, and no bytecode written into the source
    // tree by the workers a case starts.
    struct Sandbox {
        fs::path dir = madeDirectory(sirius::test::uniqueTempPath("local_worker", ""));
        ScopedEnv environment{"SIRIUS_PYTHON_ENV", (dir / "env").generic_u8string()};
        ScopedEnv mode{"SIRIUS_FAKE_WORKER", std::nullopt};
        ScopedEnv noBytecode{"PYTHONDONTWRITEBYTECODE", std::string("1")};

        Sandbox() = default;
        Sandbox(const Sandbox&) = delete;
        Sandbox& operator=(const Sandbox&) = delete;
        ~Sandbox() {
            std::error_code ec;
            fs::remove_all(dir, ec);
        }
        std::string path(const char* name) const { return (dir / name).generic_u8string(); }
    };

    // A Python for the fake worker; the case skips without one.
    std::string testPython() {
        const std::string python = environmentValue("SIRIUS_PYTHON").value_or(std::string());
        if (!python.empty()) return python;
        const std::string found = host::findPython();
        if (found.empty()) SKIP("no Python interpreter: set SIRIUS_PYTHON");
        return found;
    }

    // A launcher of the fake worker with `python`.
    std::unique_ptr<LocalWorker> fakeWorker(const std::string& python) {
        auto worker = std::make_unique<LocalWorker>();
        worker->setPython(python);
        worker->setScriptDir(SIRIUS_TEST_FAKE_WORKER_DIR);
        worker->setDevice("cpu");
        return worker;
    }

    // What connect() threw; the case fails when the worker started.
    WorkerStartError startError(LocalWorker& worker) {
        std::optional<WorkerStartError> error;
        try {
            (void)worker.connect();
        } catch (const WorkerStartError& e) {
            error = e;
        }
        REQUIRE(error.has_value());
        return *error;
    }

    // The fake worker's pid, from the line it logs first; 0 before it has.
    int loggedPid(const std::string& log) {
        const std::string key = "fake worker: pid ";
        const std::size_t at = log.find(key);
        return at == std::string::npos ? 0 : std::atoi(log.c_str() + at + key.size());
    }

    // Whether the process `pid` has gone within three seconds: a launcher in
    // between (a uv trampoline, the venv launcher) takes its child along,
    // which may take a moment.
    bool goneSoon(int pid) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (host::processAlive(pid) && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        return !host::processAlive(pid);
    }

    // SIRIUS's own environment, planted at $SIRIUS_PYTHON_ENV: a virtual
    // environment made from `python` without pip (offline, well under a
    // second), with the marker setup writes last. False when this Python
    // cannot make one.
    bool plantManagedEnvironment(const std::string& python, const std::string& envDir) {
        ChildProcess venv;
        ChildProcess::Options o;
        o.program = python;
        o.arguments = {"-m", "venv", "--without-pip", envDir};
        if (!venv.start(o)) return false;
        const bool made = venv.waitForExit(120000) && venv.exitCode() == 0;
        venv.stop(0);
        if (!made) return false;
        pyenv::Marker marker;
        marker.createdBy = "test_app_local_worker";
        marker.baseExecutable = python;
        marker.fingerprint = pyenv::requirementsFingerprint(SIRIUS_TEST_WORKER_DIR, false);
        std::ofstream(fs::u8path(envDir) / pyenv::kMarkerFile) << marker.toJson().dump();
        return host::isFile(pyenv::environmentPython(envDir));
    }

    StartFailure failure(const std::string& interpreter, const std::string& source) {
        StartFailure f;
        f.interpreter = interpreter;
        f.source = source;
        return f;
    }

} // namespace

// --- classifyStartFailure: each way a start fails --------------------------------------------

TEST_CASE("local worker: a missing_packages line is MissingPackages, with what the worker reported", "[app][local_worker]") {
    StartFailure f = failure("C:/py/python.exe", "discovered");
    f.firstStdoutLine = R"({"error":"missing_packages","missing":["numpy"],"python":"C:/py/python.exe","version":"3.14.5"})";
    f.stderrLog = "missing packages: numpy (not installed in C:/py/python.exe)\n";
    f.exitCode = 3;
    f.exitedBeforePort = true;
    const WorkerStartError e = classifyStartFailure(f);
    CHECK(e.kind == Kind::MissingPackages);
    CHECK(e.missing == std::vector<std::string>{"numpy"});
    CHECK(e.pythonVersion == "3.14.5");
    CHECK(e.interpreter == "C:/py/python.exe");
    CHECK(e.source == "discovered");
    CHECK(std::string(e.what()) ==
          "the Python worker cannot start: numpy is not installed in C:/py/python.exe (found on this computer)");
    CHECK(e.hint.empty());   // the launcher adds the host's
    CHECK(has(e.log, "missing packages: numpy"));

    // Several names read as a list; from SIRIUS's own environment it is
    // still MissingPackages (an outdated environment), even when a plain
    // start fails too.
    f.source = "managed";
    f.basicRunFailed = true;
    f.firstStdoutLine = R"({"error":"missing_packages","missing":["numpy","scipy","skimage"]})";
    const WorkerStartError several = classifyStartFailure(f);
    CHECK(several.kind == Kind::MissingPackages);
    CHECK(several.missing == std::vector<std::string>{"numpy", "scipy", "skimage"});
    CHECK(several.pythonVersion.empty());
    CHECK(has(several.what(),
              "numpy, scipy and skimage are not installed in C:/py/python.exe (SIRIUS's own Python environment)"));
}

TEST_CASE("local worker: a ModuleNotFoundError from an older worker is MissingPackages as well", "[app][local_worker]") {
    StartFailure f = failure("/usr/bin/python3", "environment");
    f.exitCode = 1;
    f.exitedBeforePort = true;
    f.stderrLog = "Traceback (most recent call last):\n"
                  "  File \"/opt/sirius/python/sirius_worker/__main__.py\", line 58, in main\n"
                  "    from . import models\n"
                  "ModuleNotFoundError: No module named 'numpy'\n";
    WorkerStartError e = classifyStartFailure(f);
    CHECK(e.kind == Kind::MissingPackages);
    CHECK(e.missing == std::vector<std::string>{"numpy"});
    CHECK(has(e.what(), "numpy is not installed in /usr/bin/python3 (named by $SIRIUS_PYTHON)"));

    // A submodule names its package.
    f.stderrLog = "ModuleNotFoundError: No module named 'scipy.ndimage'\n";
    e = classifyStartFailure(f);
    CHECK(e.kind == Kind::MissingPackages);
    CHECK(e.missing == std::vector<std::string>{"scipy"});

    // Not a missing package: the worker's own package (a wrong directory),
    // and an interpreter that could not find its standard library.
    f.stderrLog = "ModuleNotFoundError: No module named 'sirius_worker'\n";
    CHECK(classifyStartFailure(f).kind == Kind::Failed);
    f.stderrLog = "Fatal Python error: Failed to import encodings module\n"
                  "Python runtime state: core initialized\n"
                  "ModuleNotFoundError: No module named 'encodings'\n";
    CHECK(classifyStartFailure(f).kind == Kind::Failed);
}

TEST_CASE("local worker: the venv launcher's missing base is a broken managed environment", "[app][local_worker]") {
    StartFailure f = failure("C:/Users/u/AppData/Local/sirius/python-env/Scripts/python.exe", "managed");
    f.exitCode = 103;
    f.exitedBeforePort = true;
    for (const char* said : {"No Python at '\"C:\\Users\\u\\AppData\\Roaming\\uv\\python\\cpython-3.12\\python.exe\"'",
                             "did not find executable at 'C:\\Python314\\python.exe': "
                             "The system cannot find the file specified."}) {
        f.stderrLog = std::string(said) + "\n";
        const WorkerStartError e = classifyStartFailure(f);
        CHECK(e.kind == Kind::BrokenEnvironment);
        CHECK(has(e.what(), "SIRIUS's own Python environment no longer runs"));
        CHECK(has(e.what(), said));
    }
    // The same words from an interpreter the user named are the user's to fix.
    f.source = "explicit";
    CHECK(classifyStartFailure(f).kind == Kind::Failed);
}

TEST_CASE("local worker: a program that does not exist is NoInterpreter, or a broken managed environment",
          "[app][local_worker]") {
    StartFailure f = failure("python", "fallback");
    f.startError = "The system cannot find the file specified.";
    f.programNotFound = true;
    WorkerStartError e = classifyStartFailure(f);
    CHECK(e.kind == Kind::NoInterpreter);
    CHECK(std::string(e.what()) == "the Python worker cannot start: no Python 3 interpreter was found on this computer");

    f = failure("D:/tools/python.exe", "configured");
    f.startError = "No such file or directory";
    f.programNotFound = true;
    e = classifyStartFailure(f);
    CHECK(e.kind == Kind::NoInterpreter);
    CHECK(has(e.what(), "D:/tools/python.exe (the configured interpreter) does not exist"));

    f.source = "managed";
    CHECK(classifyStartFailure(f).kind == Kind::BrokenEnvironment);

    // A start that failed for another reason is not about the interpreter.
    f.source = "explicit";
    f.programNotFound = false;
    f.startError = "Access is denied.";
    e = classifyStartFailure(f);
    CHECK(e.kind == Kind::Failed);
    CHECK(has(e.what(), "Access is denied."));

    // Windows' App Execution Alias for python.exe runs, and only points to the Store.
    f = failure("python", "fallback");
    f.exitCode = 9009;
    f.exitedBeforePort = true;
    f.stderrLog = "Python was not found; run without arguments to install from the Microsoft Store, or disable this "
                  "shortcut from Settings > Apps > Advanced app settings > App execution aliases.\n";
    CHECK(classifyStartFailure(f).kind == Kind::NoInterpreter);
    // Named by the user it exists all the same: the message does not say it is missing.
    f.interpreter = "C:/Users/u/AppData/Local/Microsoft/WindowsApps/python.exe";
    f.source = "explicit";
    e = classifyStartFailure(f);
    CHECK(e.kind == Kind::NoInterpreter);
    CHECK(has(e.what(), "no Python 3 interpreter was found"));
    CHECK(has(e.what(), "placeholder for the Microsoft Store"));
    CHECK_FALSE(has(e.what(), "does not exist"));
}

TEST_CASE("local worker: a managed environment that exits early is broken when a plain start fails too", "[app][local_worker]") {
    StartFailure f = failure("/home/u/.local/share/sirius/python-env/bin/python", "managed");
    f.exitCode = 1;
    f.exitedBeforePort = true;
    f.basicRunFailed = true;
    WorkerStartError e = classifyStartFailure(f);
    CHECK(e.kind == Kind::BrokenEnvironment);
    CHECK(has(e.what(), "/home/u/.local/share/sirius/python-env/bin/python does not run (exit code 1)"));

    // The interpreter itself runs: the worker failed on its own.
    f.basicRunFailed = false;
    f.stderrLog = "RuntimeError: something else\n";
    e = classifyStartFailure(f);
    CHECK(e.kind == Kind::Failed);
    CHECK(has(e.what(), "RuntimeError: something else"));
}

TEST_CASE("local worker: anything else is Failed, with the end of the worker's stderr", "[app][local_worker]") {
    StartFailure f = failure("/usr/bin/python3", "discovered");
    f.exitCode = 2;
    f.exitedBeforePort = true;
    f.stderrLog = std::string(3000, 'x') + "\nOSError: [Errno 98] Address already in use\n";
    WorkerStartError e = classifyStartFailure(f);
    CHECK(e.kind == Kind::Failed);
    CHECK(e.log.size() <= 800);
    CHECK(has(e.log, "Address already in use"));
    CHECK(has(e.what(), "the Python worker did not start (exit code 2): "));
    CHECK(has(e.what(), "Address already in use"));
    CHECK(e.missing.empty());

    // Something else on stdout than the port line, and no line in time.
    f.stderrLog.clear();
    f.firstStdoutLine = "hello from sitecustomize";
    CHECK(has(classifyStartFailure(f).what(), "it printed \"hello from sitecustomize\" instead of its port"));
    f.firstStdoutLine.clear();
    f.exitedBeforePort = false;
    CHECK(has(classifyStartFailure(f).what(), "it did not report its port in time"));
}

TEST_CASE("local worker: which failures setup would help, the kind names and the JSON form", "[app][local_worker]") {
    const struct {
        Kind kind;
        const char* name;
        bool setupHelps;
    } kinds[] = {{Kind::NoInterpreter, "no_interpreter", true},
                 {Kind::MissingPackages, "missing_packages", true},
                 {Kind::BrokenEnvironment, "broken_environment", true},
                 {Kind::NoWorkerScripts, "no_worker_scripts", false},
                 {Kind::Failed, "failed", false}};
    for (const auto& k : kinds) {
        const WorkerStartError e(k.kind, "why");
        CHECK(std::string(toString(k.kind)) == k.name);
        CHECK(e.setupWouldHelp() == k.setupHelps);
    }

    WorkerStartError e(Kind::MissingPackages, "the Python worker cannot start: numpy is not installed in py");
    e.interpreter = "py";
    e.source = "managed";
    e.pythonVersion = "3.12.4";
    e.missing = {"numpy"};
    e.hint = "set it up";
    const nlohmann::json j = e.toJson();
    CHECK(j.at("kind") == "missing_packages");
    CHECK(j.at("message") == e.what());
    CHECK(j.at("interpreter") == "py");
    CHECK(j.at("source") == "managed");
    CHECK(j.at("python_version") == "3.12.4");
    CHECK(j.at("missing") == nlohmann::json::array({"numpy"}));
    CHECK(j.at("hint") == "set it up");
}

// --- LocalWorker without a process -----------------------------------------------------------

TEST_CASE("local worker: the interpreter follows the pyenv order, the directory the host's", "[app][local_worker]") {
    const Sandbox box;   // SIRIUS's own environment: none there
    const ScopedEnv noEnvironmentPython("SIRIUS_PYTHON", std::nullopt);
    LocalWorker worker;
    CHECK(worker.python().empty());
    CHECK_FALSE(worker.isRunning());
    CHECK(worker.port() == 0);
    CHECK(worker.runningPython().empty());
    worker.stop();   // nothing to stop

    worker.setConfiguredPython([] { return std::string("C:/configured/python.exe"); });
    CHECK(worker.interpreter().source == pyenv::Source::Configured);
    CHECK(worker.interpreter().path == "C:/configured/python.exe");
    {
        const ScopedEnv environmentPython("SIRIUS_PYTHON", std::string("C:/environment/python.exe"));
        CHECK(worker.interpreter().source == pyenv::Source::Environment);
        CHECK(worker.interpreter().path == "C:/environment/python.exe");
    }
    worker.setPython("C:/explicit/python.exe");
    CHECK(worker.python() == "C:/explicit/python.exe");
    CHECK(worker.interpreter().source == pyenv::Source::Explicit);
    CHECK(worker.interpreter().path == "C:/explicit/python.exe");

    worker.setConfiguredScriptDir([] { return std::string("C:/configured/python"); });
    CHECK(worker.scriptDir() == "C:/configured/python");
    worker.setScriptDir("C:/explicit/python");
    CHECK(worker.scriptDir() == "C:/explicit/python");
}

TEST_CASE("local worker: a directory without sirius_worker is NoWorkerScripts", "[app][local_worker]") {
    const Sandbox box;
    LocalWorker worker;
    worker.setPython(box.path("python.exe"));   // never started
    worker.setScriptDir(box.dir.generic_u8string());
    worker.setSetupHint("set it up");
    int heard = 0;
    worker.setStartFailureHandler([&heard](const WorkerStartError& e) {
        CHECK(e.kind == Kind::NoWorkerScripts);
        ++heard;
    });
    const WorkerStartError e = startError(worker);
    CHECK(e.kind == Kind::NoWorkerScripts);
    CHECK(has(e.what(), box.dir.generic_u8string()));
    CHECK_FALSE(e.setupWouldHelp());
    CHECK(e.hint.empty());   // setting up Python would not find the scripts
    CHECK(heard == 1);
    CHECK_FALSE(worker.isRunning());
}

TEST_CASE("local worker: an interpreter that does not exist is NoInterpreter", "[app][local_worker]") {
    const Sandbox box;
    LocalWorker worker;
    const std::string missing = box.path("no-such-python.exe");
    worker.setPython(missing);
    worker.setScriptDir(SIRIUS_TEST_FAKE_WORKER_DIR);
    worker.setSetupHint("set it up");
    int heard = 0;
    worker.setStartFailureHandler([&heard](const WorkerStartError&) { ++heard; });
    const WorkerStartError e = startError(worker);
    CHECK(e.kind == Kind::NoInterpreter);
    CHECK(e.interpreter == missing);
    CHECK(e.source == "explicit");
    CHECK(has(e.what(), missing + " (given explicitly) does not exist"));
    CHECK(e.hint == "set it up");
    CHECK(heard == 1);
}

// --- LocalWorker with the fake worker --------------------------------------------------------

TEST_CASE("local worker: a worker without numpy fails with MissingPackages, and the handler hears it once",
          "[app][local_worker]") {
    const std::string python = testPython();
    const Sandbox box;
    const auto worker = fakeWorker(python);
    worker->setSetupHint("set it up");
    std::vector<WorkerStartError> heard;
    worker->setStartFailureHandler([&heard](const WorkerStartError& e) { heard.push_back(e); });
    std::vector<std::string> logged;
    worker->setLogHandler([&logged](const std::string& line) { logged.push_back(line); });

    const WorkerStartError e = startError(*worker);
    CHECK(e.kind == Kind::MissingPackages);
    CHECK(e.missing == std::vector<std::string>{"numpy"});
    CHECK(e.interpreter == python);
    CHECK(e.source == "explicit");
    CHECK_FALSE(e.pythonVersion.empty());
    CHECK(e.hint == "set it up");
    CHECK(has(e.what(), "numpy is not installed in " + python + " (given explicitly)"));
    CHECK(has(e.log, "missing packages: numpy"));
    REQUIRE(heard.size() == 1);
    CHECK(heard.front().kind == Kind::MissingPackages);
    CHECK(heard.front().hint == "set it up");
    CHECK_FALSE(worker->isRunning());
    CHECK(worker->port() == 0);
    CHECK(has(worker->lastLog(), "missing packages: numpy"));
    bool sawLine = false;
    for (const std::string& line : logged) sawLine = sawLine || has(line, "missing packages: numpy");
    CHECK(sawLine);
    worker->setLogHandler({});
}

TEST_CASE("local worker: the worker may install packages only when the host allows it", "[app][local_worker]") {
    const std::string python = testPython();
    const Sandbox box;
    const auto worker = fakeWorker(python);
    (void)startError(*worker);
    const std::string arguments = worker->lastLog();
    CHECK(has(arguments, R"("--device", "cpu")"));
    CHECK(has(arguments, R"("--exit-with-parent")"));
    CHECK_FALSE(has(arguments, "--allow-install"));

    worker->setAllowInstall(true);
    (void)startError(*worker);
    CHECK(has(worker->lastLog(), R"("--allow-install")"));
}

TEST_CASE("local worker: the application's secrets stay out of the worker's environment", "[app][local_worker]") {
    const std::string python = testPython();
    const Sandbox box;
    const ScopedEnv hpc("SIRIUS_HPC_TOKEN", std::string("hpc-secret"));
    const ScopedEnv llm("SIRIUS_LLM_API_KEY", std::string("llm-secret"));
    const ScopedEnv openai("OPENAI_API_KEY", std::string("openai-secret"));
    const ScopedEnv hf("HF_TOKEN", std::string("hf-token"));
    const auto worker = fakeWorker(python);
    (void)startError(*worker);
    // HF_TOKEN stays: the worker downloads gated models with it
    CHECK(has(worker->lastLog(), R"(fake worker: secrets ["HF_TOKEN"])"));
    CHECK_FALSE(has(worker->lastLog(), "secret\""));
}

TEST_CASE("local worker: an older worker's traceback and other failures reach the error", "[app][local_worker]") {
    const std::string python = testPython();
    const Sandbox box;
    const auto worker = fakeWorker(python);
    worker->setSetupHint("set it up");
    {
        const ScopedEnv mode("SIRIUS_FAKE_WORKER", std::string("traceback"));
        const WorkerStartError e = startError(*worker);
        CHECK(e.kind == Kind::MissingPackages);
        CHECK(e.missing == std::vector<std::string>{"sirius_test_absent_module"});
    }
    {
        const ScopedEnv mode("SIRIUS_FAKE_WORKER", std::string("exit"));
        const WorkerStartError e = startError(*worker);
        CHECK(e.kind == Kind::Failed);
        CHECK(has(e.what(), "failing on purpose"));
        CHECK(has(e.log, "failing on purpose"));
        CHECK(e.hint.empty());
    }
}

TEST_CASE("local worker: a cancel ends a start that never prints its port", "[app][local_worker]") {
    const std::string python = testPython();
    const Sandbox box;
    const ScopedEnv mode("SIRIUS_FAKE_WORKER", std::string("sleep"));
    const auto worker = fakeWorker(python);
    int heard = 0;
    worker->setStartFailureHandler([&heard](const WorkerStartError&) { ++heard; });

    // Cancelled once the fake has logged its pid (so it surely runs), or
    // after ten seconds whatever it did.
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    std::optional<Clock::time_point> cancelledAt;
    const std::function<bool()> cancelled = [&] {
        if (!cancelledAt && (loggedPid(worker->lastLog()) > 0 || Clock::now() - started > std::chrono::seconds(10)))
            cancelledAt = Clock::now();
        return cancelledAt.has_value();
    };
    CHECK_THROWS_AS(worker->connect(cancelled), CancelledError);
    REQUIRE(cancelledAt);
    CHECK(Clock::now() - *cancelledAt < std::chrono::seconds(1));
    CHECK(heard == 0);   // a cancel is not a failure to start
    CHECK_FALSE(worker->isRunning());
    const int pid = loggedPid(worker->lastLog());   // gone with it
    CHECK(pid > 0);
    if (pid > 0) CHECK(goneSoon(pid));
}

TEST_CASE("local worker: a cancel also ends the wait for a worker that did not come up to leave", "[app][local_worker]") {
    const std::string python = testPython();
    const Sandbox box;
    const ScopedEnv mode("SIRIUS_FAKE_WORKER", std::string("linger"));
    const auto worker = fakeWorker(python);
    int heard = 0;
    worker->setStartFailureHandler([&heard](const WorkerStartError&) { ++heard; });

    // Cancelled once the fake says it lingers: the launcher has then read
    // its line, which is not a port, and waits for it to leave (the grace
    // for that, then the one of the stop, took six seconds a cancel could
    // not shorten). Or after ten seconds, whatever it did.
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    std::optional<Clock::time_point> cancelledAt;
    const std::function<bool()> cancelled = [&] {
        if (!cancelledAt &&
            (has(worker->lastLog(), "fake worker: lingering") || Clock::now() - started > std::chrono::seconds(10)))
            cancelledAt = Clock::now();
        return cancelledAt.has_value();
    };
    CHECK_THROWS_AS(worker->connect(cancelled), CancelledError);
    REQUIRE(cancelledAt);
    CHECK(Clock::now() - *cancelledAt < std::chrono::seconds(1));
    CHECK(heard == 0);
    CHECK_FALSE(worker->isRunning());
    CHECK(worker->port() == 0);
    const int pid = loggedPid(worker->lastLog());
    CHECK(pid > 0);
    if (pid > 0) CHECK(goneSoon(pid));
}

TEST_CASE("local worker: a stop from another thread ends a start in progress", "[app][local_worker]") {
    const std::string python = testPython();
    const Sandbox box;
    const ScopedEnv mode("SIRIUS_FAKE_WORKER", std::string("sleep"));
    const auto worker = fakeWorker(python);
    int heard = 0;
    worker->setStartFailureHandler([&heard](const WorkerStartError&) { ++heard; });

    // The stop comes once the fake has logged its pid, while connect() still
    // waits for its port; it used to wait for the minute of that timeout.
    using Clock = std::chrono::steady_clock;
    std::optional<Clock::time_point> stoppedAt, stopReturned;
    std::thread stopper([&] {
        const auto started = Clock::now();
        while (loggedPid(worker->lastLog()) <= 0 && Clock::now() - started < std::chrono::seconds(10))
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        stoppedAt = Clock::now();
        worker->stop();
        stopReturned = Clock::now();
    });
    CHECK_THROWS_AS(worker->connect(), CancelledError);
    stopper.join();
    REQUIRE(stoppedAt);
    REQUIRE(stopReturned);
    CHECK(*stopReturned - *stoppedAt < std::chrono::seconds(2));
    CHECK(heard == 0);   // stopped, not failed
    CHECK_FALSE(worker->isRunning());
    const int pid = loggedPid(worker->lastLog());
    CHECK(pid > 0);
    if (pid > 0) CHECK(goneSoon(pid));
}

// --- SIRIUS's own environment ----------------------------------------------------------------

TEST_CASE("local worker: SIRIUS's own environment is chosen, and runs with PYTHONHOME emptied", "[app][local_worker]") {
    const std::string python = testPython();
    const Sandbox box;
    const ScopedEnv noEnvironmentPython("SIRIUS_PYTHON", std::nullopt);
    if (!plantManagedEnvironment(python, box.path("env"))) SKIP(python << " cannot make a virtual environment");

    // A PYTHONHOME meant for another Python: had the launcher passed it on,
    // the environment's interpreter would not even find its standard library.
    const ScopedEnv strayHome("PYTHONHOME", box.path("another-python"));
    LocalWorker worker;
    worker.setScriptDir(SIRIUS_TEST_FAKE_WORKER_DIR);
    CHECK(worker.interpreter().source == pyenv::Source::Managed);
    CHECK(worker.interpreter().path == pyenv::environmentPython(box.path("env")));
    const WorkerStartError e = startError(worker);
    CHECK(e.kind == Kind::MissingPackages);
    CHECK(e.source == "managed");
    CHECK(has(e.what(), "(SIRIUS's own Python environment)"));
    CHECK(has(worker.lastLog(), R"(fake worker: PYTHONHOME "")"));
}

TEST_CASE("local worker: a managed environment whose Python has gone is BrokenEnvironment", "[app][local_worker]") {
    const std::string python = testPython();
    const Sandbox box;
    const ScopedEnv noEnvironmentPython("SIRIUS_PYTHON", std::nullopt);
    const std::string envDir = box.path("env");
    if (!plantManagedEnvironment(python, envDir)) SKIP(python << " cannot make a virtual environment");
#ifdef _WIN32
    // The venv launcher (Scripts/python.exe) starts the Python that
    // pyvenv.cfg names, which is no longer there.
    const fs::path config = fs::u8path(envDir) / "pyvenv.cfg";
    std::ofstream(config, std::ios::trunc) << "home = " << (box.dir / "gone").string()
                                           << "\ninclude-system-site-packages = false\n";
#else
    // An interpreter that runs nothing (a uv trampoline whose base has gone
    // says nothing either): only the plain `-I -c` start tells.
    const char* fails = fs::exists("/usr/bin/false") ? "/usr/bin/false" : "/bin/false";
    if (!fs::exists(fails)) SKIP("no false(1) to stand in for a broken interpreter");
    const fs::path interpreter = fs::u8path(pyenv::environmentPython(envDir));
    fs::remove(interpreter);
    fs::create_symlink(fails, interpreter);
#endif
    LocalWorker worker;
    worker.setScriptDir(SIRIUS_TEST_FAKE_WORKER_DIR);
    worker.setSetupHint("repair it");
    REQUIRE(worker.interpreter().source == pyenv::Source::Managed);
    const WorkerStartError e = startError(worker);
    CHECK(e.kind == Kind::BrokenEnvironment);
    CHECK(e.source == "managed");
    CHECK(e.hint == "repair it");
    CHECK(has(e.what(), "SIRIUS's own Python environment no longer runs"));
}

// --- the real worker -------------------------------------------------------------------------

TEST_CASE("local worker: a worker with numpy starts, answers hello, is reused and stops", "[app][local_worker][worker]") {
    const std::string python = environmentValue("SIRIUS_PYTHON").value_or(std::string());
    if (python.empty()) SKIP("SIRIUS_PYTHON is not set");
    const Sandbox box;
    LocalWorker worker;
    worker.setPython(python);
    worker.setScriptDir(SIRIUS_TEST_WORKER_DIR);
    worker.setDevice("cpu");
    std::unique_ptr<RemoteWorker> remote;
    try {
        remote = worker.connect();
    } catch (const WorkerStartError& e) {
        if (e.kind == Kind::MissingPackages) SKIP("SIRIUS_PYTHON cannot run the worker: " << e.what());
        throw;
    }
    REQUIRE(remote);
    CHECK(remote->capabilities().protocolVersion == rpc::kProtocolVersion);
    CHECK(worker.isRunning());
    const int port = worker.port();
    CHECK(port > 0);
    CHECK(worker.runningPython() == python);
    remote->close();
    remote.reset();

    // The process is kept: the next connection goes to the same one.
    remote = worker.connect();
    CHECK(worker.port() == port);
    remote->close();
    remote.reset();

    worker.stop();
    CHECK_FALSE(worker.isRunning());
    CHECK(worker.port() == 0);
    CHECK(worker.runningPython().empty());
}
