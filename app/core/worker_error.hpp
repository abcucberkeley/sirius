#ifndef SIRIUS_APP_WORKER_ERROR_HPP
#define SIRIUS_APP_WORKER_ERROR_HPP

// Why the local Python worker did not come up, in a form the hosts can act
// on: the GUI offers to set up SIRIUS's own Python environment, sirius-cli
// names the command that does, and both log the message with their hint.
// LocalWorker::connect throws it; classifyStartFailure reads what the
// failed start left behind and is pure, so each way a start can fail is
// tested without a process.

#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace sirius::app {
    // Why the local Python worker did not come up (LocalWorker::connect).
    class WorkerStartError : public std::runtime_error {
    public:
        enum class Kind { NoInterpreter,
                          MissingPackages,
                          BrokenEnvironment,
                          NoWorkerScripts,
                          Failed };
        WorkerStartError(Kind kind, const std::string& message);
        Kind kind = Kind::Failed;
        std::string interpreter;              // the program started
        std::string source;                   // pyenv::toString(Source): "explicit" "environment" "configured"
                                              // "managed" "discovered" "fallback"
        std::string pythonVersion;            // when the worker reported it
        std::vector<std::string> missing;     // import names, {"numpy"}
        std::string log;                      // last ~800 bytes of the worker's stderr
        std::string hint;                     // host-specific next step (LocalWorker::setSetupHint), set
                                              // only when setupWouldHelp(): "" otherwise
        bool setupWouldHelp() const noexcept; // NoInterpreter, MissingPackages, BrokenEnvironment
        nlohmann::json toJson() const;        // {kind, message, interpreter, source, python_version, missing, hint}
    };
    // "no_interpreter" "missing_packages" "broken_environment" "no_worker_scripts" "failed"
    const char* toString(WorkerStartError::Kind k) noexcept;
    struct StartFailure {
        std::string startError;               // ChildProcess::start's error, "" when it started
        bool programNotFound = false;
        std::string firstStdoutLine;          // what came instead of {"port": ...}
        std::string stderrLog;
        int exitCode = -1;
        bool exitedBeforePort = false;
        bool basicRunFailed = false;          // managed only: `<envpy> -I -c "import sys"` failed afterwards
        std::string interpreter, source;
    };
    WorkerStartError classifyStartFailure(const StartFailure& f);   // pure; the message without the hint
} // namespace sirius::app

#endif // SIRIUS_APP_WORKER_ERROR_HPP
