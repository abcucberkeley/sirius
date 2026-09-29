#include "core/worker_error.hpp"

#include <cstddef>
#include <string_view>

#include <nlohmann/json.hpp>

namespace sirius::app {

    namespace {
        // How much of the worker's stderr an error carries: the end of a
        // traceback says why, the start of a long one only where.
        constexpr std::size_t kLogTail = 800;

        std::string trimmed(std::string_view s) {
            const auto space = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
            while (!s.empty() && space(s.front())) s.remove_prefix(1);
            while (!s.empty() && space(s.back())) s.remove_suffix(1);
            return std::string(s);
        }

        // The last kLogTail bytes of the log, not starting inside a UTF-8
        // sequence (a path in the traceback may have any characters).
        std::string tail(const std::string& log) {
            const std::string t = trimmed(log);
            if (t.size() <= kLogTail) return t;
            std::size_t from = t.size() - kLogTail;
            while (from < t.size() && (static_cast<unsigned char>(t[from]) & 0xC0) == 0x80) ++from;
            return t.substr(from);
        }

        bool contains(const std::string& text, std::string_view needle) { return text.find(needle) != std::string::npos; }

        // The whole line of `log` that holds `needle`, trimmed; "" when none does.
        std::string lineWith(const std::string& log, std::string_view needle) {
            const std::size_t at = log.find(needle);
            if (at == std::string::npos) return {};
            const std::size_t begin = log.rfind('\n', at);
            const std::size_t end = log.find('\n', at);
            const std::size_t from = begin == std::string::npos ? 0 : begin + 1;
            return trimmed(std::string_view(log).substr(from, end == std::string::npos ? std::string::npos : end - from));
        }

        // "numpy", "numpy and scipy", "numpy, scipy and torch".
        std::string listed(const std::vector<std::string>& names) {
            std::string out;
            for (std::size_t i = 0; i < names.size(); ++i) {
                if (i > 0) out += i + 1 == names.size() ? " and " : ", ";
                out += names[i];
            }
            return out;
        }

        // How the interpreter was chosen (pyenv::Source), for the message.
        std::string whereFrom(const std::string& source) {
            if (source == "explicit") return "given explicitly";
            if (source == "environment") return "named by $SIRIUS_PYTHON";
            if (source == "configured") return "the configured interpreter";
            if (source == "managed") return "SIRIUS's own Python environment";
            if (source == "discovered") return "found on this computer";
            if (source == "fallback") return "the default on PATH";
            return {};
        }

        std::string described(const StartFailure& f) {
            const std::string from = whereFrom(f.source);
            return from.empty() ? f.interpreter : f.interpreter + " (" + from + ")";
        }

        // The first line on stdout when it is the worker's own report of what
        // it lacks: {"error":"missing_packages","missing":[...],"python":...,"version":...}.
        bool missingPackagesLine(const std::string& line, std::vector<std::string>& missing, std::string& version) {
            const nlohmann::json j = nlohmann::json::parse(line, nullptr, false);
            if (!j.is_object() || j.value("error", std::string()) != "missing_packages") return false;
            if (const auto m = j.find("missing"); m != j.end() && m->is_array())
                for (const nlohmann::json& name : *m)
                    if (name.is_string() && !name.get<std::string>().empty()) missing.push_back(name.get<std::string>());
            if (const auto v = j.find("version"); v != j.end() && v->is_string()) version = v->get<std::string>();
            return true;
        }

        // The top-level module of the first "ModuleNotFoundError: No module
        // named 'x.y'" in the log: what a worker from before the
        // missing_packages line printed when numpy was not there. The worker's
        // own package is not a missing requirement but a wrong directory, and
        // an interpreter that could not initialise (a PYTHONHOME meant for
        // another Python: "Fatal Python error: Failed to import encodings
        // module") lacks its standard library, not a package.
        std::string missingModule(const std::string& log) {
            static constexpr std::string_view key = "ModuleNotFoundError: No module named '";
            if (contains(log, "Fatal Python error")) return {};
            const std::size_t at = log.find(key);
            if (at == std::string::npos) return {};
            const std::size_t begin = at + key.size();
            const std::size_t end = log.find('\'', begin);
            if (end == std::string::npos) return {};
            std::string name = log.substr(begin, end - begin);
            name = name.substr(0, name.find('.'));
            return name == "sirius_worker" ? std::string() : name;
        }

        std::string noInterpreterMessage(const StartFailure& f) {
            // What was looked for rather than found (python / python3 on PATH)
            // names no file of its own.
            if (f.interpreter.empty() || f.source == "fallback" || f.source == "discovered")
                return "no Python 3 interpreter was found on this computer";
            return described(f) + " does not exist";
        }

        WorkerStartError make(WorkerStartError::Kind kind, const std::string& why, const StartFailure& f) {
            WorkerStartError e(kind, "the Python worker cannot start: " + why);
            e.interpreter = f.interpreter;
            e.source = f.source;
            e.log = tail(f.stderrLog);
            return e;
        }
    } // namespace

    WorkerStartError::WorkerStartError(Kind k, const std::string& message) : std::runtime_error(message), kind(k) {}

    bool WorkerStartError::setupWouldHelp() const noexcept {
        return kind == Kind::NoInterpreter || kind == Kind::MissingPackages || kind == Kind::BrokenEnvironment;
    }

    nlohmann::json WorkerStartError::toJson() const {
        return {{"kind", toString(kind)},
                {"message", what()},
                {"interpreter", interpreter},
                {"source", source},
                {"python_version", pythonVersion},
                {"missing", missing},
                {"hint", hint}};
    }

    const char* toString(WorkerStartError::Kind k) noexcept {
        switch (k) {
            case WorkerStartError::Kind::NoInterpreter: return "no_interpreter";
            case WorkerStartError::Kind::MissingPackages: return "missing_packages";
            case WorkerStartError::Kind::BrokenEnvironment: return "broken_environment";
            case WorkerStartError::Kind::NoWorkerScripts: return "no_worker_scripts";
            case WorkerStartError::Kind::Failed: return "failed";
        }
        return "failed";
    }

    WorkerStartError classifyStartFailure(const StartFailure& f) {
        using Kind = WorkerStartError::Kind;
        const bool managed = f.source == "managed";
        const std::string brokenEnvironment = "SIRIUS's own Python environment no longer runs: ";

        // The worker's own report comes first: it ran, and says exactly what
        // it lacks. From the managed environment this is an outdated one.
        std::vector<std::string> missing;
        std::string version;
        if (missingPackagesLine(trimmed(f.firstStdoutLine), missing, version)) {
            const std::string lacking = missing.empty() ? std::string("the packages it needs are")
                                                        : listed(missing) + (missing.size() == 1 ? " is" : " are");
            WorkerStartError e = make(Kind::MissingPackages, lacking + " not installed in " + described(f), f);
            e.missing = std::move(missing);
            e.pythonVersion = std::move(version);
            return e;
        }

        if (f.programNotFound) {
            if (managed) return make(Kind::BrokenEnvironment, brokenEnvironment + f.interpreter + " is missing", f);
            return make(Kind::NoInterpreter, noInterpreterMessage(f), f);
        }
        if (!f.startError.empty()) return make(Kind::Failed, "cannot run " + described(f) + ": " + f.startError, f);

        // The standard library's Windows venv launcher, when the Python the
        // environment was made from has gone (uninstalled, or upgraded to
        // another directory): "No Python at '...'" from older ones, "did not
        // find executable at '...'" from newer ones (3.14 here).
        for (const std::string_view said : {std::string_view("No Python at"), std::string_view("did not find executable at")})
            if (managed && contains(f.stderrLog, said))
                return make(Kind::BrokenEnvironment,
                            brokenEnvironment + "the Python it was made from is gone (" + lineWith(f.stderrLog, said) + ")", f);
        // A uv environment's trampoline says nothing when its base has gone;
        // a plain `-I -c "import sys"` that fails as well tells a broken
        // environment from a worker that failed on its own.
        if (managed && f.exitedBeforePort && f.basicRunFailed) {
            std::string why = brokenEnvironment + f.interpreter + " does not run";
            if (f.exitCode >= 0) why += " (exit code " + std::to_string(f.exitCode) + ")";
            return make(Kind::BrokenEnvironment, why, f);
        }

        // Windows' App Execution Alias for python.exe: a placeholder that
        // only points to the Microsoft Store. The program exists, so the
        // message says what it is rather than that it is missing.
        if (!managed && (contains(f.stderrLog, "Python was not found") || contains(f.firstStdoutLine, "Python was not found")))
            return make(Kind::NoInterpreter,
                        "no Python 3 interpreter was found on this computer (" + described(f) +
                            " is only Windows' placeholder for the Microsoft Store)",
                        f);

        if (std::string module = missingModule(f.stderrLog); !module.empty()) {
            WorkerStartError e = make(Kind::MissingPackages, module + " is not installed in " + described(f), f);
            e.missing = {std::move(module)};
            return e;
        }

        // Anything else: what the worker said is the best account there is.
        const std::string log = tail(f.stderrLog);
        const std::string first = trimmed(f.firstStdoutLine);
        std::string message = "the Python worker did not start";
        if (!first.empty()) message += " (it printed \"" + first + "\" instead of its port)";
        else if (!f.exitedBeforePort) message += " (it did not report its port in time)";
        else if (f.exitCode >= 0) message += " (exit code " + std::to_string(f.exitCode) + ")";
        if (!log.empty()) message += ": " + log;
        WorkerStartError e(Kind::Failed, message);
        e.interpreter = f.interpreter;
        e.source = f.source;
        e.log = log;
        return e;
    }

} // namespace sirius::app
