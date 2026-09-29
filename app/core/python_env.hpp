#ifndef SIRIUS_APP_PYTHON_ENV_HPP
#define SIRIUS_APP_PYTHON_ENV_HPP

// SIRIUS's own Python environment for the worker (a venv in the user's data directory)
// and which interpreter the worker runs. GUI-free; the GUI and sirius-cli share it.

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace sirius::app::pyenv {
    constexpr const char* kMarkerFile = "sirius-env.json";
    constexpr int kMarkerSchema = 1;
    constexpr int kMinMinor = 9;

    std::string environmentDirectory();                              // $SIRIUS_PYTHON_ENV or <data>/sirius/python-env
    std::string environmentPython(const std::string& envDir);        // <env>/Scripts/python.exe | <env>/bin/python
    // Normalised: comments / blank lines dropped, names lower-cased, deduplicated, sorted.
    std::vector<std::string> requirements(const std::string& scriptDir, bool extras,
                                          const std::vector<std::string>& extraPackages = {});
    // FNV-1a 64 (hex) of the normalised list + extras flag + extra packages (line endings never matter).
    std::string requirementsFingerprint(const std::string& scriptDir, bool extras,
                                        const std::vector<std::string>& extraPackages = {});
    const std::vector<std::string>& optionalDistributions();         // mirrors sirius_worker.OPTIONAL's values
    std::string redactUrl(const std::string& url);                   // scheme://user:pass@host -> scheme://***@host

    struct Marker {
        int schema = kMarkerSchema;
        std::string createdBy, created, baseExecutable, pythonVersion, installer, index, fingerprint;
        bool extras = false;
        std::vector<std::string> extraPackages;
        std::map<std::string, std::string> packages;
        nlohmann::json toJson() const;
        static std::optional<Marker> fromJson(const nlohmann::json& j);
    };
    std::optional<Marker> readMarker(const std::string& envDir);

    enum class State { Absent,
                       Incomplete,
                       Ready,
                       Outdated,
                       Broken };
    const char* toString(State s) noexcept;
    struct EnvironmentStatus {
        State state = State::Absent;
        std::string dir, python, problem;
        std::optional<Marker> marker;
        bool requirementsCurrent = false;
        nlohmann::json toJson() const;                               // URLs redacted
    };
    // runPython: also `<envpy> -I -c "import sys"` and `<envpy> -m sirius_worker --check` (cwd scriptDir).
    EnvironmentStatus environmentStatus(const std::string& scriptDir, bool runPython);

    enum class Source { Explicit,
                        Environment,
                        Configured,
                        Managed,
                        Discovered,
                        Fallback };
    const char* toString(Source s) noexcept;
    struct Interpreter {
        std::string path;
        Source source = Source::Fallback;
    };
    Interpreter workerInterpreter(const std::string& explicitPython, const std::string& configured);

    struct PythonInfo {
        std::string executable, baseExecutable, version;
        int major = 0, minor = 0, bits = 64;
        bool freeThreaded = false, venv = false, externallyManaged = false, hasPip = false, hasEnsurepip = false;
        std::string problem;                                         // "" = usable as a base
        nlohmann::json toJson() const;
    };
    // `<python> -I -c "<inline stdlib script>"`; never imports sirius_worker.
    std::optional<PythonInfo> probe(const std::string& python, int timeoutMs = 15000, std::string* error = nullptr);
    std::vector<std::string> pythonCandidates();                     // resolved, deduplicated, not probed
    std::string findUv();                                            // $SIRIUS_UV, then PATH
    std::string uvVersion(const std::string& uv);

    enum class Mode { Auto,
                      Create,
                      Update,
                      Recreate };
    struct SetupOptions {
        std::string basePython;
        bool extras = false;
        std::vector<std::string> extraPackages;
        Mode mode = Mode::Auto;
        bool useUv = true;
        std::string indexUrl;                                        // CLI only (never from agent tools)
        std::vector<std::string> findLinks;                          // CLI only
        bool noIndex = false;
        std::string createdBy;
    };
    struct SetupPlan {
        std::string envDir, basePython, basePythonVersion, installer, uv, index;   // index redacted
        Mode mode = Mode::Create;
        bool nothingToDo = false;
        std::vector<std::string> packages;
        std::uint64_t approxDownloadBytes = 0;
        bool externallyManagedBase = false;
        std::vector<std::vector<std::string>> commands;              // redacted
        std::vector<std::string> warnings;
        nlohmann::json toJson() const;
    };
    enum class Failure { None,
                         NoPython,
                         UnsupportedPython,
                         NoEnsurepip,
                         Offline,
                         Tls,
                         NoWheel,
                         DiskFull,
                         InUse,
                         Locked,
                         Cancelled,
                         Failed };
    const char* toString(Failure f) noexcept;
    struct SetupResult {
        bool ok = false;
        Failure failure = Failure::None;
        std::string message, hint;
        std::vector<std::string> logTail;
        SetupPlan plan;
        std::optional<Marker> marker;
        double seconds = 0.0;
        nlohmann::json toJson() const;
    };
    SetupPlan planSetup(const SetupOptions& options, const std::string& scriptDir);
    // Installer processes run with killTree + mergeErrorLines; both streams reach onLine.
    SetupResult setup(const SetupOptions& options, const std::string& scriptDir,
                      const std::function<void(const std::string& line)>& onLine,
                      const std::function<void(double fraction, const std::string& message)>& onProgress,
                      const std::function<bool()>& cancelled);
    SetupResult remove(const std::string& envDir);
    Failure classifyInstallerOutput(const std::vector<std::string>& lines, int exitCode, std::string* hint);
    std::optional<double> progressFromLine(const std::string& line, bool uv);
} // namespace sirius::app::pyenv

#endif // SIRIUS_APP_PYTHON_ENV_HPP
