// Tests of SIRIUS's own Python environment for the worker
// (app/core/python_env.hpp): where it lives, the requirement lists and their
// fingerprint, the states of planted environments, which interpreter the
// worker runs, how installer output is classified and turned into progress,
// the marker, the lock, what reaches the installers' command lines (absolute
// paths, packages as spelled, no options), and a probe of a real interpreter
// when there is one.
//
// Every case that could reach the managed environment points
// SIRIUS_PYTHON_ENV at a directory of its own first (in-process, put back
// when the case ends): the developer's environment is never read or changed.
// A real setup downloads numpy, so it runs only with SIRIUS_TEST_PYTHON_SETUP=1.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/host.hpp"
#include "core/python_env.hpp"

using namespace sirius::app;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

    // Sets a variable (or, with nullopt, removes it) for as long as it
    // lives, then puts back what was there.
    class ScopedVariable {
    public:
        ScopedVariable(const char* name, std::optional<std::string> value) : name_(name) {
            if (host::hasEnvironment(name)) saved_ = host::environment(name);
            set(value);
        }
        ~ScopedVariable() { set(saved_); }
        ScopedVariable(const ScopedVariable&) = delete;
        ScopedVariable& operator=(const ScopedVariable&) = delete;

    private:
        void set(const std::optional<std::string>& value) const {
#ifdef _WIN32
            // Changes the process's environment as well as the C runtime's
            // copy; "" removes the variable.
            (void)_putenv_s(name_.c_str(), value ? value->c_str() : "");
#else
            if (value) ::setenv(name_.c_str(), value->c_str(), 1);
            else ::unsetenv(name_.c_str());
#endif
        }

        std::string name_;
        std::optional<std::string> saved_;
    };

    // A directory of the case's own, removed with what it holds.
    struct TempDir {
        std::string path = host::makeTempDirectory("sirius-pyenv-test-");
        TempDir() { REQUIRE_FALSE(path.empty()); }
        ~TempDir() {
            std::error_code ec;
            fs::remove_all(fs::u8path(path), ec);
        }
        TempDir(const TempDir&) = delete;
        TempDir& operator=(const TempDir&) = delete;
    };

    // SIRIUS_PYTHON_ENV at <temp>/env (not created), for the whole case.
    struct ScratchEnvironment {
        TempDir dir;
        std::string env = dir.path + "/env";
        ScopedVariable variable{"SIRIUS_PYTHON_ENV", env};
    };

    void writeText(const std::string& path, const std::string& text) {
        fs::create_directories(fs::u8path(path).parent_path());
        std::ofstream f(fs::u8path(path), std::ios::binary);
        f << text;
        REQUIRE(f.good());
    }

    std::string readText(const std::string& path) {
        std::string text;
        REQUIRE(host::readFile(path, text));
        return text;
    }

    bool has(const std::string& s, const std::string& needle) { return s.find(needle) != std::string::npos; }

    const std::string kWorkerDir = SIRIUS_TEST_WORKER_DIR;

    // A fake environment the way a finished setup leaves it, as far as the
    // states without running Python can tell: the marker (with `fingerprint`,
    // and what it was set up with besides the requirements) and an
    // interpreter file that does not run.
    void plantEnvironment(const std::string& env, const std::string& fingerprint, bool extras = false,
                          const std::vector<std::string>& extraPackages = {}) {
        pyenv::Marker marker;
        marker.createdBy = "test";
        marker.created = "2026-09-29T10:12:03Z";
        marker.baseExecutable = "C:/Python314/python.exe";
        marker.pythonVersion = "3.14.5";
        marker.installer = "uv 0.12.18";
        marker.index = "https://pypi.org/simple";
        marker.fingerprint = fingerprint;
        marker.extras = extras;
        marker.extraPackages = extraPackages;
        marker.packages = {{"numpy", "2.5.3"}, {"pip", "26.2.1"}};
        writeText(env + "/" + pyenv::kMarkerFile, marker.toJson().dump(2));
        writeText(env + "/pyvenv.cfg", "home = C:/Python314\n");
        writeText(pyenv::environmentPython(env), "");
    }

    // An interpreter for the cases that run one: $SIRIUS_PYTHON, else the
    // one on PATH; "" when there is none.
    std::string availablePython() {
        const std::string named = host::environment("SIRIUS_PYTHON");
        return !named.empty() ? named : host::findPython();
    }

} // namespace

TEST_CASE("python env: the directory follows SIRIUS_PYTHON_ENV, else the data directory", "[app][python_env]") {
    TempDir dir;
    {
        const ScopedVariable variable("SIRIUS_PYTHON_ENV", dir.path + "/custom/");
        CHECK(pyenv::environmentDirectory() == dir.path + "/custom");
    }
    {
        const ScopedVariable variable("SIRIUS_PYTHON_ENV", std::nullopt);
        const std::string data = host::dataDirectory();
        REQUIRE_FALSE(data.empty());
        CHECK(pyenv::environmentDirectory() == data + "/sirius/python-env");
    }
}

TEST_CASE("python env: the interpreter inside an environment", "[app][python_env]") {
#ifdef _WIN32
    CHECK(pyenv::environmentPython("C:/x/env") == "C:/x/env/Scripts/python.exe");
#else
    CHECK(pyenv::environmentPython("/x/env") == "/x/env/bin/python");
#endif
    CHECK(pyenv::environmentPython("").empty());
}

TEST_CASE("python env: the requirements come from the worker's files, or numpy", "[app][python_env]") {
    using List = std::vector<std::string>;
    CHECK(pyenv::requirements(kWorkerDir, false) == List{"numpy"});
    CHECK(pyenv::requirements(kWorkerDir, true) == List{"numpy", "scikit-image", "scipy"});
    CHECK(pyenv::requirements(kWorkerDir, false, {" Torch ", "numpy", "btrack  # tracking"}) == List{"btrack", "numpy", "torch"});
    TempDir empty;
    CHECK(pyenv::requirements(empty.path, false) == List{"numpy"});
    CHECK(pyenv::requirements(empty.path, true) == List{"numpy", "scikit-image", "scipy"});
    CHECK(pyenv::requirements("", false) == List{"numpy"});
}

TEST_CASE("python env: the fingerprint ignores line endings, order and comments", "[app][python_env]") {
    TempDir lf, crlf, reordered, renamed;
    writeText(lf.path + "/requirements.txt", "# required\nnumpy\n");
    writeText(lf.path + "/requirements-extra.txt", "# optional\nscipy\nscikit-image\n");
    writeText(crlf.path + "/requirements.txt", "# required\r\nnumpy\r\n");
    writeText(crlf.path + "/requirements-extra.txt", "# optional\r\nscipy\r\nscikit-image\r\n");
    const std::string bom = "\xEF\xBB\xBF";
    writeText(reordered.path + "/requirements.txt", bom + "NumPy   # the array\n\n");
    writeText(reordered.path + "/requirements-extra.txt", "scikit-image\n  scipy\nscipy\n");
    writeText(renamed.path + "/requirements.txt", "numpy\n");
    writeText(renamed.path + "/requirements-extra.txt", "scipy\nscikit-learn\n");

    for (const bool extras : {false, true}) {
        INFO("extras " << extras);
        const std::string f = pyenv::requirementsFingerprint(lf.path, extras);
        CHECK(f.size() == 16);
        CHECK(f.find_first_not_of("0123456789abcdef") == std::string::npos);
        CHECK(pyenv::requirementsFingerprint(crlf.path, extras) == f);
        CHECK(pyenv::requirementsFingerprint(reordered.path, extras) == f);
    }
    CHECK(pyenv::requirementsFingerprint(renamed.path, false) == pyenv::requirementsFingerprint(lf.path, false));
    CHECK(pyenv::requirementsFingerprint(renamed.path, true) != pyenv::requirementsFingerprint(lf.path, true));
    CHECK(pyenv::requirementsFingerprint(lf.path, true) != pyenv::requirementsFingerprint(lf.path, false));
    CHECK(pyenv::requirementsFingerprint(lf.path, false, {"torch"}) != pyenv::requirementsFingerprint(lf.path, false));
    CHECK(pyenv::requirementsFingerprint(lf.path, false, {"Torch "}) == pyenv::requirementsFingerprint(lf.path, false, {"torch"}));
    // the files that ship are what the fallback says, too
    CHECK(pyenv::requirementsFingerprint(kWorkerDir, true) == pyenv::requirementsFingerprint(lf.path, true));
}

TEST_CASE("python env: the optional distributions are sirius_worker.OPTIONAL's", "[app][python_env]") {
    const std::string init = readText(kWorkerDir + "/sirius_worker/__init__.py");
    REQUIRE(has(init, "OPTIONAL = {"));
    REQUIRE_FALSE(pyenv::optionalDistributions().empty());
    for (const std::string& d : pyenv::optionalDistributions()) {
        INFO(d);
        CHECK(has(init, ": \"" + d + "\""));
    }
    CHECK(std::find(pyenv::optionalDistributions().begin(), pyenv::optionalDistributions().end(), "numpy") ==
          pyenv::optionalDistributions().end());
}

TEST_CASE("python env: credentials in URLs are redacted", "[app][python_env]") {
    CHECK(pyenv::redactUrl("https://u:p@host/simple") == "https://***@host/simple");
    CHECK(pyenv::redactUrl("https://token@host.example:8443/simple/") == "https://***@host.example:8443/simple/");
    CHECK(pyenv::redactUrl("https://pypi.org/simple") == "https://pypi.org/simple");
    CHECK(pyenv::redactUrl("https://host/path@version") == "https://host/path@version");
    CHECK(pyenv::redactUrl("--index-url=https://a:b@one/simple --find-links https://c:d@two/wheels") ==
          "--index-url=https://***@one/simple --find-links https://***@two/wheels");
    CHECK(pyenv::redactUrl("D:/wheels") == "D:/wheels");
    CHECK(pyenv::redactUrl(pyenv::redactUrl("https://u:p@host/simple")) == "https://***@host/simple");
    // query values: an index's token is often one
    CHECK(pyenv::redactUrl("https://host/simple?token=abc&x=1#frag") == "https://host/simple?token=***&x=***#frag");
    CHECK(pyenv::redactUrl("see https://u:p@host/s/?sig=zzz and https://two/?k=v") == "see https://***@host/s/?sig=*** and https://two/?k=***");
    CHECK(pyenv::redactUrl("https://host/?flag") == "https://host/?flag");
    CHECK(pyenv::redactUrl(pyenv::redactUrl("https://host/?a=b&c=d")) == "https://host/?a=***&c=***");
}

TEST_CASE("python env: the marker survives a JSON round trip", "[app][python_env]") {
    pyenv::Marker m;
    m.createdBy = "sirius-cli 0.1.0";
    m.created = "2026-09-29T10:12:03Z";
    m.baseExecutable = "C:/Users/Velat/AppData/Roaming/uv/python/cpython-3.14-windows-x86_64-none/python.exe";
    m.pythonVersion = "3.14.5";
    m.installer = "uv 0.12.18";
    m.index = "https://pypi.org/simple";
    m.fingerprint = "9c1e0000aaaabbbb";
    m.extras = true;
    m.extraPackages = {"torch"};
    m.packages = {{"numpy", "2.5.3"}, {"pip", "26.2.1"}, {"scipy", "1.16.2"}};
    const json j = m.toJson();
    CHECK(j["schema"] == 1);
    CHECK(j["created_by"] == "sirius-cli 0.1.0");
    CHECK(j["packages"]["numpy"] == "2.5.3");
    const std::optional<pyenv::Marker> back = pyenv::Marker::fromJson(json::parse(j.dump()));
    REQUIRE(back);
    CHECK(back->toJson() == j);
    CHECK(back->extraPackages == m.extraPackages);
    CHECK(back->packages == m.packages);

    CHECK_FALSE(pyenv::Marker::fromJson(json::array()));
    CHECK_FALSE(pyenv::Marker::fromJson(json{{"fingerprint", "x"}}));
    CHECK(pyenv::Marker::fromJson(json{{"schema", 2}, {"future", true}}));   // a newer SIRIUS's marker still reads
    pyenv::Marker secret = m;
    secret.index = "https://user:pass@mirror/simple";
    CHECK(secret.toJson()["index"] == "https://***@mirror/simple");
}

TEST_CASE("python env: the states of planted environments", "[app][python_env]") {
    const ScratchEnvironment scratch;
    const std::string current = pyenv::requirementsFingerprint(kWorkerDir, false);
    const auto status = [] { return pyenv::environmentStatus(kWorkerDir, false); };

    CHECK(status().state == pyenv::State::Absent);
    CHECK(status().dir == scratch.env);
    CHECK(status().python == pyenv::environmentPython(scratch.env));

    fs::create_directories(fs::u8path(scratch.env));
    writeText(scratch.env + "/pyvenv.cfg", "home = x\n");
    CHECK(status().state == pyenv::State::Incomplete);
    CHECK(has(status().problem, "did not finish"));

    plantEnvironment(scratch.env, current);
    CHECK(status().state == pyenv::State::Ready);
    CHECK(status().requirementsCurrent);
    REQUIRE(status().marker);
    CHECK(status().marker->pythonVersion == "3.14.5");
    const json j = status().toJson();
    CHECK(j["state"] == "ready");
    CHECK(j["marker"]["packages"]["numpy"] == "2.5.3");

    // The interpreter file does not run: only a check that runs it can tell.
    const pyenv::EnvironmentStatus checked = pyenv::environmentStatus(kWorkerDir, true);
    CHECK(checked.state == pyenv::State::Broken);
    CHECK(has(checked.problem, "does not run"));

    plantEnvironment(scratch.env, "0000000000000000");
    CHECK(status().state == pyenv::State::Outdated);
    CHECK_FALSE(status().requirementsCurrent);
    CHECK(status().problem == "the requirements changed");

    fs::remove(fs::u8path(pyenv::environmentPython(scratch.env)));
    CHECK(status().state == pyenv::State::Incomplete);
    CHECK(has(status().problem, "Python is missing"));
}

TEST_CASE("python env: a directory that is not an environment is left alone", "[app][python_env]") {
    const ScratchEnvironment scratch;
    writeText(scratch.env + "/thesis.docx", "mine");
    const pyenv::EnvironmentStatus s = pyenv::environmentStatus(kWorkerDir, false);
    CHECK(s.state == pyenv::State::Incomplete);
    CHECK(has(s.problem, "not a Python environment"));
    CHECK(has(s.problem, "thesis.docx"));   // what to move, named

    const pyenv::SetupResult removed = pyenv::remove(scratch.env);
    CHECK_FALSE(removed.ok);
    CHECK(removed.failure == pyenv::Failure::Failed);
    CHECK(has(removed.message, "thesis.docx"));
    CHECK(host::isFile(scratch.env + "/thesis.docx"));

    pyenv::SetupOptions options;
    options.useUv = false;
    options.basePython = scratch.dir.path + "/no-such-python";
    const pyenv::SetupResult set = pyenv::setup(options, kWorkerDir, {}, {}, {});
    CHECK_FALSE(set.ok);
    CHECK(host::isFile(scratch.env + "/thesis.docx"));
    CHECK_FALSE(fs::exists(fs::u8path(scratch.env + ".old")));
    CHECK_FALSE(fs::exists(fs::u8path(scratch.env + ".lock")));

    // With a base that works, it is the directory itself that stops the
    // setup, before anything is moved or made (and without the network).
    const std::string python = availablePython();
    if (python.empty()) SKIP("no Python interpreter: set SIRIUS_PYTHON");
    const std::optional<pyenv::PythonInfo> info = pyenv::probe(python);
    if (!info || !info->problem.empty() || !info->hasEnsurepip) SKIP("the Python found cannot be a base without uv: " + python);
    options.basePython = python;
    const pyenv::SetupPlan plan = pyenv::planSetup(options, kWorkerDir);
    CHECK(plan.mode == pyenv::Mode::Recreate);
    CHECK(std::any_of(plan.warnings.begin(), plan.warnings.end(), [](const std::string& w) { return has(w, "not a Python environment"); }));
    const pyenv::SetupResult refused = pyenv::setup(options, kWorkerDir, {}, {}, {});
    INFO(refused.toJson().dump(2));
    CHECK(refused.failure == pyenv::Failure::Failed);
    CHECK(has(refused.message, "not a Python environment"));
    CHECK(host::isFile(scratch.env + "/thesis.docx"));
    CHECK_FALSE(fs::exists(fs::u8path(scratch.env + ".old")));
    CHECK_FALSE(fs::exists(fs::u8path(scratch.env + ".lock")));
}

TEST_CASE("python env: what Finder and Explorer leave in an environment does not count", "[app][python_env]") {
    const ScratchEnvironment scratch;
    plantEnvironment(scratch.env, "0000000000000000");
    for (const char* name : {".DS_Store", "desktop.ini", "Thumbs.db"}) writeText(scratch.env + "/" + name, "");
    const pyenv::SetupResult removed = pyenv::remove(scratch.env);
    INFO(removed.toJson().dump(2));
    CHECK(removed.ok);
    CHECK_FALSE(fs::exists(fs::u8path(scratch.env)));
}

TEST_CASE("python env: a previous environment left at .old is put back first", "[app][python_env]") {
    // A recreate whose rollback could not delete the new folder: the working
    // environment at .old, an unfinished one without a marker in its place.
    const ScratchEnvironment scratch;
    plantEnvironment(scratch.env + ".old", "0000000000000000");
    writeText(scratch.env + "/pyvenv.cfg", "home = x\n");
    pyenv::SetupOptions options;
    options.useUv = false;
    options.basePython = scratch.dir.path + "/no-such-python";   // stops the setup right after planning
    std::vector<std::string> lines;
    const pyenv::SetupResult r = pyenv::setup(options, kWorkerDir, [&](const std::string& line) { lines.push_back(line); }, {}, {});
    INFO(r.toJson().dump(2));
    CHECK(r.failure == pyenv::Failure::UnsupportedPython);
    CHECK(host::isFile(scratch.env + "/" + pyenv::kMarkerFile));
    CHECK_FALSE(fs::exists(fs::u8path(scratch.env + ".old")));
    CHECK(std::any_of(lines.begin(), lines.end(), [](const std::string& l) { return has(l, "put back the previous environment"); }));
}

TEST_CASE("python env: the worker's interpreter, in order", "[app][python_env]") {
    const ScratchEnvironment scratch;
    const ScopedVariable noPython("SIRIUS_PYTHON", std::nullopt);

    CHECK(pyenv::workerInterpreter("C:/explicit/python.exe", "C:/configured/python.exe").source == pyenv::Source::Explicit);
    CHECK(pyenv::workerInterpreter("C:/explicit/python.exe", "").path == "C:/explicit/python.exe");
    {
        const ScopedVariable named("SIRIUS_PYTHON", std::string("C:/environment/python.exe"));
        const pyenv::Interpreter i = pyenv::workerInterpreter("", "C:/configured/python.exe");
        CHECK(i.source == pyenv::Source::Environment);
        CHECK(i.path == "C:/environment/python.exe");
    }
    CHECK(pyenv::workerInterpreter("", "C:/configured/python.exe").source == pyenv::Source::Configured);

    // Nothing named: SIRIUS's environment once it has a marker and a python,
    // whether or not it is current.
    const pyenv::Interpreter before = pyenv::workerInterpreter("", "");
    CHECK(before.source == (host::findPython().empty() ? pyenv::Source::Fallback : pyenv::Source::Discovered));
    plantEnvironment(scratch.env, "0000000000000000");
    const pyenv::Interpreter managed = pyenv::workerInterpreter("", "");
    CHECK(managed.source == pyenv::Source::Managed);
    CHECK(managed.path == pyenv::environmentPython(scratch.env));
    CHECK(pyenv::workerInterpreter("", "C:/configured/python.exe").source == pyenv::Source::Configured);
    CHECK(std::string(pyenv::toString(pyenv::Source::Managed)) == "managed");
}

TEST_CASE("python env: installer output is classified", "[app][python_env]") {
    using pyenv::Failure;
    const auto classify = [](const std::vector<std::string>& lines, int exitCode = 1) {
        std::string hint;
        const Failure f = pyenv::classifyInstallerOutput(lines, exitCode, &hint);
        if (f != Failure::None) CHECK_FALSE(hint.empty());
        return f;
    };
    CHECK(classify({"Resolved 1 package in 480ms"}, 0) == Failure::None);
    // offline: uv, and pip, whose retries end in "no matching distribution"
    CHECK(classify({"error: Request failed after 3 retries", "  Caused by: Failed to fetch: `https://pypi.org/simple/numpy/`",
                    "  Caused by: error sending request for url (https://pypi.org/simple/numpy/)", "  Caused by: client error (Connect)",
                    "  Caused by: tcp connect error: Network is unreachable (os error 101)"}) == Failure::Offline);
    CHECK(classify({"WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by "
                    "'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x0000>: Failed to establish a new "
                    "connection: [Errno 11001] getaddrinfo failed')': /simple/numpy/",
                    "ERROR: Could not find a version that satisfies the requirement numpy (from versions: none)",
                    "ERROR: No matching distribution found for numpy"}) == Failure::Offline);
    // TLS interception
    CHECK(classify({"error: Request failed after 3 retries", "  Caused by: Failed to fetch: `https://pypi.org/simple/numpy/`",
                    "  Caused by: invalid peer certificate: UnknownIssuer"}) == Failure::Tls);
    CHECK(classify({"Could not fetch URL https://pypi.org/simple/numpy/: There was a problem confirming the ssl certificate: "
                    "HTTPSConnectionPool(host='pypi.org', port=443): Max retries exceeded with url: /simple/numpy/ (Caused by "
                    "SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get "
                    "local issuer certificate (_ssl.c:1006)'))) - skipping",
                    "ERROR: No matching distribution found for numpy"}) == Failure::Tls);
    // no wheel for this Python
    CHECK(classify({"  x No solution found when resolving dependencies:",
                    "  `-> Because numpy==2.3.4 has no wheels with a matching Python ABI tag (e.g., `cp315`) and you require numpy==2.3.4, "
                    "we can conclude that your requirements are unsatisfiable."}) == Failure::NoWheel);
    CHECK(classify({"Collecting scikit-image", "ERROR: Could not find a version that satisfies the requirement scikit-image (from versions: 0.19.3)",
                    "ERROR: No matching distribution found for scikit-image"}) == Failure::NoWheel);
    // disk full, POSIX and Windows
    CHECK(classify({"ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device"}) == Failure::DiskFull);
    CHECK(classify({"error: Failed to install: numpy-2.3.4-cp314-cp314-win_amd64.whl (numpy==2.3.4)",
                    "  Caused by: There is not enough space on the disk. (os error 112)"}) == Failure::DiskFull);
    // a file in use (a worker still running from the environment)
    CHECK(classify({"ERROR: Could not install packages due to an OSError: [WinError 32] The process cannot access the file because it is "
                    "being used by another process: 'C:\\\\env\\\\Lib\\\\site-packages\\\\numpy\\\\_core\\\\_multiarray_umath.cp314-win_amd64.pyd'"}) ==
          Failure::InUse);
    // a Debian Python without python3-venv; the message is wrapped
    CHECK(classify({"The virtual environment was not created successfully because ensurepip is not",
                    "available.  On Debian/Ubuntu systems, you need to install the python3-venv", "package using the following command."}) ==
          Failure::NoEnsurepip);
    // A private index that answers 401: uv says "Failed to fetch" for it too,
    // but it is the password (or the package), not the network.
    std::string refusedHint;
    CHECK(pyenv::classifyInstallerOutput({"error: Failed to fetch: `https://mirror.example/simple/numpy/`",
                                          "  Caused by: HTTP status client error (401 Unauthorized) for url (https://mirror.example/simple/numpy/)"},
                                         2, &refusedHint) == Failure::Failed);
    CHECK(has(refusedHint, "refused"));
    CHECK(classify({"ERROR: HTTP error 403 while getting https://mirror.example/packages/numpy-2.3.4.whl"}) == Failure::Failed);
    // An Update that failed half-way may have changed the environment: the
    // hint does not claim otherwise.
    std::string offlineHint;
    pyenv::classifyInstallerOutput({"  Caused by: tcp connect error: Network is unreachable (os error 101)"}, 1, &offlineHint);
    CHECK_FALSE(has(offlineHint, "Nothing was changed"));
    CHECK(classify({"something else went wrong"}, 2) == Failure::Failed);
    CHECK(pyenv::classifyInstallerOutput({"No space left on device"}, 1, nullptr) == Failure::DiskFull);
}

TEST_CASE("python env: progress from installer lines", "[app][python_env]") {
    // What `uv pip install` writes, stderr and stdout in the order they came.
    const std::vector<std::string> uv{"Using Python 3.14.5 environment at: C:\\Users\\x\\env",
                                      "Resolved 13 packages in 1.21s",
                                      "Downloading scipy (36.5MiB)",
                                      "Downloading numpy (12.3MiB)",
                                      "Downloading scikit-image (12.2MiB)",
                                      " Downloaded scikit-image",
                                      " Downloaded numpy",
                                      " Downloaded scipy",
                                      "Prepared 13 packages in 5.42s",
                                      "Installed 13 packages in 1.10s",
                                      " + numpy==2.3.4",
                                      " + scipy==1.16.2"};
    std::vector<double> fractions;
    for (const std::string& line : uv)
        if (const auto f = pyenv::progressFromLine(line, true)) fractions.push_back(*f);
    REQUIRE(fractions.size() == 9);
    CHECK(std::is_sorted(fractions.begin(), fractions.end()));
    CHECK(fractions.front() == 0.25);
    CHECK(fractions.back() == 0.95);
    CHECK_FALSE(pyenv::progressFromLine(" + numpy==2.3.4", true));
    CHECK_FALSE(pyenv::progressFromLine("Collecting numpy", true));

    CHECK(pyenv::progressFromLine("Collecting numpy", false) == 0.20);
    CHECK(pyenv::progressFromLine("  Downloading numpy-2.3.4-cp314-cp314-win_amd64.whl (12.9 MB)", false) == 0.40);
    CHECK(pyenv::progressFromLine("Installing collected packages: numpy", false) == 0.80);
    CHECK(pyenv::progressFromLine("Successfully installed numpy-2.3.4", false) == 0.95);
    CHECK_FALSE(pyenv::progressFromLine("Resolved 1 package in 480ms", false));
}

TEST_CASE("python env: the lock keeps two setups apart", "[app][python_env]") {
    const ScratchEnvironment scratch;
    const std::string lock = scratch.env + ".lock";
    const auto plantLock = [&](const std::string& content) { writeText(lock, content); };
    const auto removeFresh = [&] {
        plantEnvironment(scratch.env, "0000000000000000");
        return pyenv::remove(scratch.env);
    };

    SECTION("a live holder refuses") {
        plantLock(json{{"pid", host::processId()}, {"started", "2026-09-29T10:12:03Z"}}.dump());
        const pyenv::SetupResult r = removeFresh();
        CHECK(r.failure == pyenv::Failure::Locked);
        CHECK(has(r.message, "pid " + std::to_string(host::processId())));
        CHECK(host::isDirectory(scratch.env));
        CHECK(host::isFile(lock));   // somebody else's; it stays
    }
    SECTION("a holder that is gone is taken over") {
        plantLock(json{{"pid", 2147483600}, {"started", "2026-09-29T10:12:03Z"}}.dump());
        const pyenv::SetupResult r = removeFresh();
        CHECK(r.ok);
        CHECK_FALSE(fs::exists(fs::u8path(scratch.env)));
        CHECK_FALSE(fs::exists(fs::u8path(lock)));
        // The stale lock was moved aside before it was deleted: nothing of
        // it is left next to the environment.
        std::vector<std::string> left;
        for (const fs::directory_entry& entry : fs::directory_iterator(fs::u8path(scratch.dir.path))) left.push_back(entry.path().filename().u8string());
        CHECK(left.empty());
    }
    SECTION("a live holder's lock older than an hour is stale") {
        plantLock(json{{"pid", host::processId()}}.dump());
        fs::last_write_time(fs::u8path(lock), fs::file_time_type::clock::now() - std::chrono::hours(2));
        CHECK(removeFresh().ok);
    }
    SECTION("another machine's holder is not looked up") {
        plantLock(json{{"pid", 2147483600}, {"host", "some-other-node-of-the-cluster"}}.dump());
        CHECK(removeFresh().failure == pyenv::Failure::Locked);
    }
    SECTION("a half-written lock is only taken over once it has aged") {
        plantLock("{\"pi");
        const pyenv::SetupResult fresh = removeFresh();
        CHECK(fresh.failure == pyenv::Failure::Locked);
        fs::last_write_time(fs::u8path(lock), fs::file_time_type::clock::now() - std::chrono::minutes(1));
        CHECK(removeFresh().ok);
    }
    SECTION("a stale .old is removed under the lock") {
        writeText(scratch.env + ".old/pyvenv.cfg", "home = x\n");
        writeText(scratch.env + ".old/Lib/site-packages/numpy/__init__.py", "");
        const pyenv::SetupResult r = removeFresh();
        CHECK(r.ok);
        CHECK_FALSE(fs::exists(fs::u8path(scratch.env + ".old")));
        CHECK_FALSE(fs::exists(fs::u8path(scratch.env)));
    }
    SECTION("nothing to remove") {
        const pyenv::SetupResult r = pyenv::remove(scratch.env);
        CHECK(r.ok);
        CHECK(has(r.message, "no environment"));
        CHECK_FALSE(fs::exists(fs::u8path(lock)));
    }
}

TEST_CASE("python env: a base that is not Python stops the setup before anything changes", "[app][python_env]") {
    const ScratchEnvironment scratch;
    const ScopedVariable noIndex("PIP_INDEX_URL", std::nullopt);
    pyenv::SetupOptions options;
    options.useUv = false;
    options.basePython = scratch.dir.path + "/no-such-python";

    const pyenv::SetupPlan plan = pyenv::planSetup(options, kWorkerDir);
    CHECK(plan.envDir == scratch.env);
    CHECK(plan.mode == pyenv::Mode::Create);
    CHECK_FALSE(plan.nothingToDo);
    CHECK(plan.packages == std::vector<std::string>{"numpy"});
    CHECK(plan.approxDownloadBytes == 13'000'000ULL);
    CHECK(plan.installer == "pip");
    CHECK(plan.index == "https://pypi.org/simple");
    CHECK_FALSE(plan.warnings.empty());
    const json j = plan.toJson();
    CHECK(j["mode"] == "create");
    CHECK(j["env_dir"] == scratch.env);

    std::vector<std::string> lines;
    const pyenv::SetupResult r = pyenv::setup(options, kWorkerDir, [&](const std::string& line) { lines.push_back(line); }, {}, {});
    CHECK_FALSE(r.ok);
    CHECK(r.failure == pyenv::Failure::UnsupportedPython);
    CHECK_FALSE(r.hint.empty());
    CHECK_FALSE(fs::exists(fs::u8path(scratch.env)));
    CHECK_FALSE(fs::exists(fs::u8path(scratch.env + ".lock")));
    REQUIRE_FALSE(lines.empty());
    CHECK(has(lines.back(), "Python environment: not set up: "));
    CHECK(r.toJson()["failure"] == "unsupported_python");
}

TEST_CASE("python env: the index is named without its credentials", "[app][python_env]") {
    const ScratchEnvironment scratch;
    pyenv::SetupOptions options;
    options.useUv = false;
    options.basePython = scratch.dir.path + "/no-such-python";
    options.indexUrl = "https://user:secret@mirror.example/simple";
    const pyenv::SetupPlan plan = pyenv::planSetup(options, kWorkerDir);
    CHECK(plan.index == "https://***@mirror.example/simple");
    CHECK_FALSE(has(plan.toJson().dump(), "secret"));

    // --no-index: the wheel directories, or a name that says there is none
    // (the prompt would otherwise read "from pypi.org").
    pyenv::SetupOptions offline = options;
    offline.indexUrl.clear();
    offline.noIndex = true;
    CHECK(pyenv::planSetup(offline, kWorkerDir).index == "none (--no-index)");
    offline.findLinks = {"https://user:secret@mirror.example/wheels/"};
    CHECK(pyenv::planSetup(offline, kWorkerDir).index == "https://***@mirror.example/wheels/");
}

TEST_CASE("python env: the index reaches the installer through its environment, not its command line", "[app][python_env]") {
    const ScratchEnvironment scratch;
    pyenv::SetupOptions options;
    options.useUv = false;
    options.basePython = scratch.dir.path + "/no-such-python";
    options.indexUrl = "https://user:secret@mirror.example/simple?token=abc";
    const pyenv::SetupPlan plan = pyenv::planSetup(options, kWorkerDir);
    for (const auto& command : plan.commands)
        for (const std::string& a : command) {
            CHECK_FALSE(has(a, "--index-url"));
            CHECK_FALSE(has(a, "mirror.example"));
        }
    const auto pip = pyenv::installerIndexEnvironment(options, false);
    REQUIRE(pip.size() == 1);
    CHECK(pip.front().first == "PIP_INDEX_URL");
    CHECK(pip.front().second == options.indexUrl);
    const auto uv = pyenv::installerIndexEnvironment(options, true);
    REQUIRE(uv.size() == 1);
    CHECK(uv.front().first == "UV_INDEX_URL");
    options.indexUrl.clear();
    CHECK(pyenv::installerIndexEnvironment(options, false).empty());
}

TEST_CASE("python env: the application's secrets are named for the children to drop", "[app][python_env]") {
    const std::vector<std::string>& names = pyenv::secretEnvironmentNames();
    for (const char* name : {"SIRIUS_HPC_TOKEN", "SIRIUS_LLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"})
        CHECK(std::find(names.begin(), names.end(), std::string(name)) != names.end());
    CHECK(std::find(names.begin(), names.end(), std::string("HF_TOKEN")) == names.end());
}

TEST_CASE("python env: a directory of venv-like folders without pyvenv.cfg or the marker is not removed", "[app][python_env]") {
    const ScratchEnvironment scratch;
    // what ~/.local looks like: bin, lib, share, include -- and nothing that says venv
    writeText(scratch.env + "/bin/tool", "mine");
    writeText(scratch.env + "/lib/data.txt", "mine");
    writeText(scratch.env + "/share/notes.txt", "mine");
    const pyenv::SetupResult removed = pyenv::remove(scratch.env);
    CHECK_FALSE(removed.ok);
    CHECK(removed.failure == pyenv::Failure::Failed);
    CHECK(has(removed.message, "pyvenv.cfg"));
    CHECK(host::isFile(scratch.env + "/bin/tool"));

    // with pyvenv.cfg it is an environment, and goes
    writeText(scratch.env + "/pyvenv.cfg", "home = /usr/bin\n");
    const pyenv::SetupResult again = pyenv::remove(scratch.env);
    INFO(again.message);
    CHECK(again.ok);
    CHECK_FALSE(fs::exists(fs::u8path(scratch.env)));
}

TEST_CASE("python env: a package given as a URL is named without its credentials", "[app][python_env]") {
    const ScratchEnvironment scratch;
    const std::string wheel = "https://user:secret@wheels.example/x-1.0-py3-none-any.whl";
    const std::string shown = "https://***@wheels.example/x-1.0-py3-none-any.whl";
    const auto warned = [&](const pyenv::SetupPlan& plan) {
        return std::any_of(plan.warnings.begin(), plan.warnings.end(), [&](const std::string& w) { return has(w, shown); });
    };
    pyenv::SetupOptions options;
    options.useUv = false;
    options.basePython = scratch.dir.path + "/no-such-python";
    options.extraPackages = {wheel};

    // Asked for: its download size is unknown, and says so without the password.
    const pyenv::SetupPlan plan = pyenv::planSetup(options, kWorkerDir);
    CHECK_FALSE(has(plan.toJson().dump(), "secret"));
    CHECK(warned(plan));
    std::vector<std::string> lines;
    const pyenv::SetupResult r = pyenv::setup(options, kWorkerDir, [&](const std::string& line) { lines.push_back(line); }, {}, {});
    CHECK_FALSE(r.ok);
    CHECK_FALSE(has(r.toJson().dump(), "secret"));
    for (const std::string& line : lines) CHECK_FALSE(has(line, "secret"));

    // Kept from the marker by Auto, and left out by a Recreate that did not
    // ask for it again.
    plantEnvironment(scratch.env, "0000000000000000", false, {wheel});
    options.extraPackages.clear();
    const pyenv::SetupPlan kept = pyenv::planSetup(options, kWorkerDir);
    CHECK_FALSE(has(kept.toJson().dump(), "secret"));
    CHECK(warned(kept));
    options.mode = pyenv::Mode::Recreate;
    const pyenv::SetupPlan dropped = pyenv::planSetup(options, kWorkerDir);
    CHECK_FALSE(has(dropped.toJson().dump(), "secret"));
    CHECK(std::any_of(dropped.warnings.begin(), dropped.warnings.end(), [&](const std::string& w) { return has(w, "without " + shown); }));
    CHECK_FALSE(has(pyenv::environmentStatus(kWorkerDir, false).toJson().dump(), "secret"));
}

TEST_CASE("python env: a probe of something that is not Python", "[app][python_env]") {
    std::string error;
    CHECK_FALSE(pyenv::probe(SIRIUS_TEST_CHILD, 15000, &error));
    CHECK_FALSE(error.empty());
    TempDir dir;
    error.clear();
    CHECK_FALSE(pyenv::probe(dir.path + "/no-such-python", 15000, &error));
    CHECK_FALSE(error.empty());
    CHECK_FALSE(pyenv::probe("", 15000, nullptr));
}

TEST_CASE("python env: a probe of a real interpreter", "[app][python_env]") {
    const std::string python = availablePython();
    if (python.empty()) SKIP("no Python interpreter: set SIRIUS_PYTHON");
    std::string error;
    const std::optional<pyenv::PythonInfo> info = pyenv::probe(python, 30000, &error);
    REQUIRE(info);
    INFO(info->toJson().dump());
    CHECK(info->major == 3);
    CHECK(info->minor > 0);
    CHECK(info->version.rfind("3." + std::to_string(info->minor) + ".", 0) == 0);
    CHECK((info->bits == 32 || info->bits == 64));
    CHECK_FALSE(info->executable.empty());
    CHECK_FALSE(info->baseExecutable.empty());
    CHECK(info->problem.empty() == (info->minor >= pyenv::kMinMinor && !info->freeThreaded));
    const json j = info->toJson();
    for (const char* key : {"executable", "base_executable", "version", "major", "minor", "bits", "free_threaded", "venv",
                            "externally_managed", "pip", "ensurepip", "problem"})
        CHECK(j.contains(key));

    // It is one of the candidates a setup would offer, unless it came from
    // somewhere the candidates do not look.
    const std::vector<std::string> candidates = pyenv::pythonCandidates();
    for (const std::string& c : candidates) {
        INFO(c);
        CHECK(host::isFile(c));
        CHECK(c.find('\\') == std::string::npos);
    }
}

TEST_CASE("python env: a plan for a real base interpreter", "[app][python_env]") {
    const ScratchEnvironment scratch;
    const std::string python = availablePython();
    if (python.empty()) SKIP("no Python interpreter: set SIRIUS_PYTHON");
    pyenv::SetupOptions options;
    options.useUv = false;
    options.basePython = python;
    options.extras = true;
    const pyenv::SetupPlan plan = pyenv::planSetup(options, kWorkerDir);
    INFO(plan.toJson().dump(2));
    CHECK(plan.mode == pyenv::Mode::Create);
    CHECK(plan.basePython == python);
    CHECK_FALSE(plan.basePythonVersion.empty());
    CHECK(plan.packages == std::vector<std::string>{"numpy", "scikit-image", "scipy"});
    CHECK(plan.approxDownloadBytes == 70'000'000ULL);
    REQUIRE(plan.commands.size() == 3);
    CHECK(plan.commands[0] == std::vector<std::string>{python, "-m", "venv", scratch.env});
    CHECK(std::find(plan.commands[1].begin(), plan.commands[1].end(), ":all:") != plan.commands[1].end());
    CHECK(std::find(plan.commands[1].begin(), plan.commands[1].end(), kWorkerDir + "/requirements-extra.txt") != plan.commands[1].end());
    CHECK(plan.commands[2].back() == "--check");
    CHECK_FALSE(fs::exists(fs::u8path(scratch.env)));   // planning changes nothing
}

TEST_CASE("python env: relative paths reach the installers as absolute ones", "[app][python_env]") {
    // The installers run in the environment's parent directory, not in this
    // one: --worker-dir app/python, --find-links ./wheels, a relative
    // --base-python and a local wheel must still name the same files there.
    const ScratchEnvironment scratch;
    const std::string python = availablePython();
    if (python.empty() || !fs::u8path(python).is_absolute()) SKIP("no Python interpreter named by its path: set SIRIUS_PYTHON");
    std::error_code ec;
    const fs::path cwd = fs::current_path(ec);
    const fs::path worker = fs::relative(fs::u8path(kWorkerDir), cwd, ec);
    const fs::path base = fs::relative(fs::u8path(python), cwd, ec);
    if (ec || worker.empty() || base.empty() || worker.is_absolute()) SKIP("the checkout and the working directory are on different drives");
    const auto absolute = [&cwd](const std::string& relative) { return (cwd / fs::u8path(relative)).lexically_normal().generic_u8string(); };

    pyenv::SetupOptions options;
    options.useUv = false;
    options.basePython = (fs::path(".") / base).generic_u8string();
    options.findLinks = {"wheels-here", "https://user:secret@mirror.example/wheels/"};
    options.extraPackages = {"./local/Foo-1.0-py3-none-any.whl", "Foo @ https://Host.example/Foo-1.0-py3-none-any.whl  # pinned", " Torch "};
    const pyenv::SetupPlan plan = pyenv::planSetup(options, worker.generic_u8string());
    INFO(plan.toJson().dump(2));
    REQUIRE(plan.commands.size() == 3);
    CHECK(fs::u8path(plan.basePython).is_absolute());
    CHECK(fs::equivalent(fs::u8path(plan.basePython), fs::u8path(python)));
    CHECK(plan.commands[0].front() == plan.basePython);

    const std::vector<std::string>& install = plan.commands[1];
    const auto after = [&install](const std::string& option) {
        std::vector<std::string> values;
        for (std::size_t i = 0; i + 1 < install.size(); ++i)
            if (install[i] == option) values.push_back(install[i + 1]);
        return values;
    };
    const auto contains = [&install](const std::string& argument) { return std::find(install.begin(), install.end(), argument) != install.end(); };
    const std::vector<std::string> files = after("-r");
    REQUIRE(files.size() == 1);
    CHECK(fs::u8path(files.front()).is_absolute());
    CHECK(fs::equivalent(fs::u8path(files.front()), fs::u8path(kWorkerDir + "/requirements.txt")));
    CHECK(after("--find-links") == std::vector<std::string>{absolute("wheels-here"), "https://***@mirror.example/wheels/"});
    // A local wheel made absolute; a direct reference as it was spelled
    // (without its comment); a name as it was spelled, too.
    CHECK(contains(absolute("local/Foo-1.0-py3-none-any.whl")));
    CHECK(contains("Foo @ https://Host.example/Foo-1.0-py3-none-any.whl"));
    CHECK(contains("Torch"));
    CHECK(std::find(plan.packages.begin(), plan.packages.end(), "torch") != plan.packages.end());
}

TEST_CASE("python env: a package that is an installer option is refused", "[app][python_env]") {
    const ScratchEnvironment scratch;
    const auto warns = [](const pyenv::SetupPlan& plan, const std::string& text) {
        return std::any_of(plan.warnings.begin(), plan.warnings.end(), [&text](const std::string& w) { return has(w, text); });
    };
    pyenv::SetupOptions options;
    options.useUv = false;
    options.extraPackages = {"numpy", "--index-url=https://evil.example/simple"};
    const pyenv::SetupPlan plan = pyenv::planSetup(options, kWorkerDir);
    INFO(plan.toJson().dump(2));
    CHECK(warns(plan, "is not a package"));
    CHECK_FALSE(has(plan.toJson()["commands"].dump(), "evil.example"));
    const pyenv::SetupResult r = pyenv::setup(options, kWorkerDir, {}, {}, {});
    CHECK(r.failure == pyenv::Failure::Failed);
    CHECK(has(r.message, "is not a package"));
    CHECK_FALSE(fs::exists(fs::u8path(scratch.env)));
    CHECK_FALSE(fs::exists(fs::u8path(scratch.env + ".lock")));

    // One in the marker (written by hand) is left out of the next setup,
    // with a warning, rather than stopping every setup from then on.
    plantEnvironment(scratch.env, "0000000000000000", false, {"-e ."});
    pyenv::SetupOptions again;
    again.useUv = false;
    again.basePython = scratch.dir.path + "/no-such-python";
    const pyenv::SetupPlan planned = pyenv::planSetup(again, kWorkerDir);
    INFO(planned.toJson().dump(2));
    CHECK(warns(planned, "\"-e .\" in sirius-env.json is not a package"));
    for (const std::vector<std::string>& command : planned.commands) CHECK(std::find(command.begin(), command.end(), "-e .") == command.end());
}

TEST_CASE("python env: a recreate says what it leaves out", "[app][python_env]") {
    // What was asked for is what is made (the dialog offers the extras
    // again); what the environment has besides is named, not dropped silently.
    const ScratchEnvironment scratch;
    plantEnvironment(scratch.env, "0000000000000000", true, {"torch"});
    pyenv::SetupOptions options;
    options.useUv = false;
    options.basePython = scratch.dir.path + "/no-such-python";
    options.mode = pyenv::Mode::Recreate;
    const auto dropped = [&options] {
        const pyenv::SetupPlan plan = pyenv::planSetup(options, kWorkerDir);
        for (const std::string& w : plan.warnings)
            if (has(w, "made again without")) return w;
        return std::string();
    };
    const std::string all = dropped();
    CHECK(has(all, "scikit-image, scipy, torch"));
    options.extras = true;
    CHECK(dropped() == "The environment is made again without torch, which it has now; ask for it again to keep it.");
    options.extraPackages = {"Torch"};
    CHECK(dropped().empty());
    // An update keeps them without being asked.
    options = pyenv::SetupOptions();
    options.mode = pyenv::Mode::Update;
    options.useUv = false;
    const pyenv::SetupPlan update = pyenv::planSetup(options, kWorkerDir);
    CHECK(update.packages == std::vector<std::string>{"numpy", "scikit-image", "scipy", "torch"});
}

TEST_CASE("python env: a cancel while the setup plans changes nothing", "[app][python_env]") {
    const ScratchEnvironment scratch;
    const std::string python = availablePython();
    if (python.empty()) SKIP("no Python interpreter: set SIRIUS_PYTHON");
    pyenv::SetupOptions options;
    options.useUv = false;
    options.basePython = python;
    const pyenv::SetupResult r = pyenv::setup(options, kWorkerDir, {}, {}, [] { return true; });
    INFO(r.toJson().dump(2));
    CHECK(r.failure == pyenv::Failure::Cancelled);
    CHECK(r.message == "Cancelled; nothing was changed.");
    CHECK_FALSE(fs::exists(fs::u8path(scratch.env)));
    CHECK_FALSE(fs::exists(fs::u8path(scratch.env + ".lock")));
}

// --- a real setup (network) -----------------------------------------------------

namespace {

    bool realSetupEnabled() { return host::environment("SIRIUS_TEST_PYTHON_SETUP") == "1"; }

    // A setup into the scratch environment, checked the way the GUI and the
    // CLI use it afterwards, then removed.
    void setUpAndRemove(bool useUv) {
        const ScratchEnvironment scratch;
        // The base is chosen while $SIRIUS_PYTHON may still name it; the
        // variable then goes, so that the worker's interpreter is SIRIUS's own.
        const std::string python = availablePython();
        const ScopedVariable noPython("SIRIUS_PYTHON", std::nullopt);
        if (python.empty()) SKIP("no Python interpreter");
        pyenv::SetupOptions options;
        options.useUv = useUv;
        options.basePython = python;
        options.createdBy = "test_app_python_env";
        std::vector<std::string> lines;
        std::vector<double> fractions;
        const pyenv::SetupResult r = pyenv::setup(
            options, kWorkerDir, [&](const std::string& line) { lines.push_back(line); },
            [&](double f, const std::string&) { fractions.push_back(f); }, {});
        std::string log;
        for (const std::string& line : lines) log += line + "\n";
        INFO(log);
        INFO(r.toJson().dump(2));
        REQUIRE(r.ok);
        REQUIRE(r.marker);
        CHECK(r.marker->packages.count("numpy") == 1);
        CHECK(r.marker->installer.rfind(useUv ? "uv" : "pip", 0) == 0);
        CHECK(r.marker->createdBy == "test_app_python_env");
        CHECK(std::is_sorted(fractions.begin(), fractions.end()));
        REQUIRE_FALSE(fractions.empty());
        CHECK(fractions.back() == 1.0);
        CHECK(has(lines.back(), "Python environment: ready: numpy "));
        CHECK(host::isFile(scratch.env + "/README.txt"));
        CHECK_FALSE(fs::exists(fs::u8path(scratch.env + ".lock")));

        const pyenv::EnvironmentStatus status = pyenv::environmentStatus(kWorkerDir, true);
        INFO(status.toJson().dump(2));
        CHECK(status.state == pyenv::State::Ready);
        CHECK(pyenv::workerInterpreter("", "").source == pyenv::Source::Managed);
        pyenv::SetupOptions again;
        again.useUv = useUv;
        CHECK(pyenv::planSetup(again, kWorkerDir).nothingToDo);

        if (useUv) {
            // An update runs in the environment as it is; a recreate moves it
            // aside, makes it anew and only then deletes the old one.
            pyenv::SetupOptions update = again;
            update.mode = pyenv::Mode::Update;
            const pyenv::SetupResult updated = pyenv::setup(update, kWorkerDir, {}, {}, {});
            INFO(updated.toJson().dump(2));
            CHECK(updated.ok);
            CHECK(updated.plan.mode == pyenv::Mode::Update);
            REQUIRE(updated.marker);
            CHECK(updated.marker->created == r.marker->created);

            pyenv::SetupOptions recreate = again;
            recreate.mode = pyenv::Mode::Recreate;
            const pyenv::SetupResult recreated = pyenv::setup(recreate, kWorkerDir, {}, {}, {});
            INFO(recreated.toJson().dump(2));
            CHECK(recreated.ok);
            CHECK(recreated.plan.mode == pyenv::Mode::Recreate);
            CHECK(recreated.plan.basePython == r.marker->baseExecutable);   // the one it was made from
            CHECK_FALSE(fs::exists(fs::u8path(scratch.env + ".old")));
            CHECK(pyenv::environmentStatus(kWorkerDir, true).state == pyenv::State::Ready);
        }

        const pyenv::SetupResult removed = pyenv::remove(scratch.env);
        INFO(removed.toJson().dump(2));
        CHECK(removed.ok);
        CHECK_FALSE(fs::exists(fs::u8path(scratch.env)));
        CHECK_FALSE(fs::exists(fs::u8path(scratch.env + ".old")));
    }

} // namespace

TEST_CASE("python env: a real setup with uv, then removed", "[app][python_env]") {
    if (!realSetupEnabled()) SKIP("downloads numpy: set SIRIUS_TEST_PYTHON_SETUP=1");
    if (pyenv::findUv().empty()) SKIP("uv is not installed");
    setUpAndRemove(true);
}

TEST_CASE("python env: a real setup with venv and pip, then removed", "[app][python_env]") {
    if (!realSetupEnabled()) SKIP("downloads numpy: set SIRIUS_TEST_PYTHON_SETUP=1");
    setUpAndRemove(false);
}

TEST_CASE("python env: a cancelled setup leaves nothing behind", "[app][python_env]") {
    if (!realSetupEnabled()) SKIP("downloads numpy: set SIRIUS_TEST_PYTHON_SETUP=1");
    const ScratchEnvironment scratch;
    const std::string python = availablePython();
    if (python.empty()) SKIP("no Python interpreter");
    pyenv::SetupOptions options;
    options.useUv = false;
    options.basePython = python;
    // Cancelled once pip has said its first word about the packages: the
    // environment exists by then, and must go again with every process that
    // ran from it.
    bool installing = false, cancel = false;
    std::vector<std::string> lines;
    const pyenv::SetupResult r = pyenv::setup(
        options, kWorkerDir,
        [&](const std::string& line) {
            lines.push_back(line);
            if (line.rfind("$ ", 0) == 0 && has(line, " pip install ")) installing = true;
            else if (installing && line.rfind("python-env: ", 0) == 0) cancel = true;
        },
        {}, [&] { return cancel; });
    std::string log;
    for (const std::string& line : lines) log += line + "\n";
    INFO(log);
    CHECK(r.failure == pyenv::Failure::Cancelled);
    CHECK(r.message == "Cancelled; nothing was changed.");
    // On Windows a folder a process still runs from cannot be deleted, so
    // its absence also shows that pip was ended.
    CHECK_FALSE(fs::exists(fs::u8path(scratch.env)));
    CHECK_FALSE(fs::exists(fs::u8path(scratch.env + ".old")));
    CHECK_FALSE(fs::exists(fs::u8path(scratch.env + ".lock")));
}

TEST_CASE("python env: an untrusted certificate is retried once, then named", "[app][python_env]") {
    if (!realSetupEnabled()) SKIP("reaches a test server: set SIRIUS_TEST_PYTHON_SETUP=1");
    if (pyenv::findUv().empty()) SKIP("uv is not installed");
    const ScratchEnvironment scratch;
    const std::string python = availablePython();
    if (python.empty()) SKIP("no Python interpreter");
    pyenv::SetupOptions options;
    options.basePython = python;
    // A server whose certificate no store trusts (badssl.com's test host):
    // `uv venv --seed` makes the environment, then fails to fetch pip from
    // it, and fails again with the system's certificates. The retry must
    // start from an empty place, or uv refuses to make the environment.
    options.indexUrl = "https://self-signed.badssl.com/simple";
    std::vector<std::string> lines;
    const pyenv::SetupResult r = pyenv::setup(options, kWorkerDir, [&](const std::string& line) { lines.push_back(line); }, {}, {});
    std::string log;
    std::vector<std::string> creates;
    for (const std::string& line : lines) {
        log += line + "\n";
        if (line.rfind("$ ", 0) == 0 && has(line, " venv ")) creates.push_back(line);
    }
    INFO(log);
    CHECK(r.failure == pyenv::Failure::Tls);
    CHECK(has(r.hint, "SSL_CERT_FILE"));
    REQUIRE(creates.size() == 2);
    CHECK_FALSE(has(creates[0], "--native-tls"));
    CHECK(has(creates[1], "--native-tls"));
    CHECK_FALSE(has(log, "already exists"));
    CHECK_FALSE(fs::exists(fs::u8path(scratch.env)));
    CHECK_FALSE(fs::exists(fs::u8path(scratch.env + ".old")));
    CHECK_FALSE(fs::exists(fs::u8path(scratch.env + ".lock")));
}
