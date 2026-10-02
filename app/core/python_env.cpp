#include "core/python_env.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <deque>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <mutex>
#include <set>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#include <share.h>
#else
#include <unistd.h>
#endif

#include <nlohmann/json.hpp>

#include "core/host.hpp"
#include "core/process.hpp"

namespace sirius::app::pyenv {

    using json = nlohmann::json;

    namespace {
        namespace fs = std::filesystem;

        // --- strings and paths ---------------------------------------------------

        fs::path fsPath(const std::string& p) { return fs::u8path(p); }

        // Windows paths as the rest of the application writes them. Elsewhere
        // a backslash is an ordinary character of a file name, left alone.
        std::string forwardSlashes(std::string s) {
#ifdef _WIN32
            std::replace(s.begin(), s.end(), '\\', '/');
#endif
            return s;
        }

        // A path as UTF-8. On Windows u8string() throws for a name that is
        // not valid UTF-16 (host.cpp says the same); such a name, met in a
        // PATH directory or in the environment, reads as empty here instead
        // of throwing out of the functions that list directories.
        std::string utf8(const fs::path& p) {
            try {
                return p.u8string();
            } catch (...) {
                return std::string();
            }
        }

        std::string pathText(const fs::path& p) { return forwardSlashes(utf8(p)); }

        std::string lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        }

        std::string trim(const std::string& s) {
            const auto space = [](unsigned char c) { return std::isspace(c) != 0; };
            std::size_t b = 0, e = s.size();
            while (b < e && space(static_cast<unsigned char>(s[b]))) ++b;
            while (e > b && space(static_cast<unsigned char>(s[e - 1]))) --e;
            return s.substr(b, e - b);
        }

        bool startsWith(std::string_view s, std::string_view prefix) { return s.substr(0, prefix.size()) == prefix; }

        std::string join(const std::vector<std::string>& parts, const std::string& separator) {
            std::string out;
            for (const std::string& p : parts) {
                if (!out.empty()) out += separator;
                out += p;
            }
            return out;
        }

        // The directory that holds `path` ("" for a bare name).
        std::string parentOf(const std::string& path) { return pathText(fsPath(path).parent_path()); }

        bool exists(const std::string& path) {
            std::error_code ec;
            return !path.empty() && fs::exists(fsPath(path), ec);
        }

        // "C:/x/env/" -> "C:/x/env"; a root keeps its slash.
        std::string withoutTrailingSlash(std::string s) {
            while (s.size() > 1 && s.back() == '/' && !(s.size() == 3 && s[1] == ':')) s.pop_back();
            return s;
        }

        bool isUrl(const std::string& s) { return s.find("://") != std::string::npos; }

        // `path` made absolute against this process's working directory; an
        // absolute one is left as it is spelled. The installers run in the
        // environment's parent directory (so that a `pip` or `numpy` folder
        // where SIRIUS was started never shadows theirs), where a relative
        // path would name another file, or none.
        std::string absolutePath(const std::string& path) {
            if (path.empty() || !fsPath(path).is_relative()) return path;
            std::error_code ec;
            const fs::path a = fs::absolute(fsPath(path), ec);
            return ec ? path : withoutTrailingSlash(pathText(a.lexically_normal()));
        }

        // A program made absolute when it names a directory; a bare name is
        // left to be looked up on PATH.
        std::string absoluteProgram(const std::string& program) {
            return fsPath(program).has_parent_path() ? absolutePath(program) : program;
        }

        // "2026-09-29T10:12:03Z"
        std::string utcNow() {
            const std::time_t now = std::time(nullptr);
            std::tm tm{};
#ifdef _WIN32
            ::gmtime_s(&tm, &now);
#else
            ::gmtime_r(&now, &tm);
#endif
            char buffer[32];
            std::strftime(buffer, sizeof buffer, "%Y-%m-%dT%H:%M:%SZ", &tm);
            return buffer;
        }

        std::string hostName() {
#ifdef _WIN32
            return host::environment("COMPUTERNAME");
#else
            char buffer[256] = {};
            if (::gethostname(buffer, sizeof buffer - 1) != 0) return std::string();
            return buffer;
#endif
        }

        std::string secondsText(double s) {
            char buffer[32];
            std::snprintf(buffer, sizeof buffer, "%.1f s", s);
            return buffer;
        }

        // "3.12.4" -> 12; 0 for anything that is not a Python 3 version.
        int minorOf(std::string_view version) {
            if (!startsWith(version, "3.")) return 0;
            int minor = 0;
            for (std::size_t i = 2; i < version.size() && i < 5 && std::isdigit(static_cast<unsigned char>(version[i])); ++i)
                minor = minor * 10 + (version[i] - '0');
            return minor;
        }

        // "pip 24.0 from ... (python 3.12)" -> 12; 0 for anything else.
        int pipPythonMinor(const std::string& line) {
            const std::size_t at = line.rfind("(python ");
            return at == std::string::npos ? 0 : minorOf(std::string_view(line).substr(at + 8));
        }

        // "pip 24.0 from ... (python 3.12)" -> {24, 0}; {0, 0} for anything else.
        std::pair<int, int> pipVersion(const std::string& line) {
            const std::string t = trim(line);
            if (!startsWith(t, "pip ")) return {0, 0};
            int parts[2] = {0, 0};
            std::size_t i = 4;
            for (int& part : parts) {
                if (i >= t.size() || !std::isdigit(static_cast<unsigned char>(t[i]))) return {0, 0};
                while (i < t.size() && std::isdigit(static_cast<unsigned char>(t[i]))) part = part * 10 + (t[i++] - '0');
                if (i < t.size() && t[i] == '.') ++i;
            }
            return {parts[0], parts[1]};
        }

        // One command as it would be typed, for the "$ ..." log line; URLs
        // with credentials are redacted.
        std::string commandLine(const std::vector<std::string>& command) {
            std::string out;
            for (const std::string& raw : command) {
                const std::string a = redactUrl(raw);
                if (!out.empty()) out += ' ';
                out += a.empty() || a.find_first_of(" \t\"") != std::string::npos ? "\"" + a + "\"" : a;
            }
            return out;
        }

        std::vector<std::string> redacted(const std::vector<std::string>& command) {
            std::vector<std::string> out;
            out.reserve(command.size());
            for (const std::string& a : command) out.push_back(redactUrl(a));
            return out;
        }

        // --- requirement files ----------------------------------------------------

        // A requirement line without its comment. A '#' starts one only at
        // the start or after whitespace, as pip reads it (a URL fragment is
        // not one).
        std::string withoutComment(std::string s) {
            for (std::size_t i = 0; i < s.size(); ++i) {
                if (s[i] == '#' && (i == 0 || std::isspace(static_cast<unsigned char>(s[i - 1])))) {
                    s.erase(i);
                    break;
                }
            }
            return s;
        }

        // One requirement line as it is compared and hashed: without its
        // comment and without whitespace, lower-cased. "" for a line that
        // holds nothing else.
        std::string normaliseRequirement(const std::string& line) {
            std::string out;
            for (const char c : withoutComment(line))
                if (!std::isspace(static_cast<unsigned char>(c))) out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            return out;
        }

        // Whether a package argument names a local file or directory rather
        // than a requirement, by pip's own rule: a path separator or a
        // leading dot, or the name of an archive. A URL, a direct reference
        // ("name @ url") and one with a marker ("; python_version...") are not.
        bool isLocalPackage(const std::string& spec) {
            if (spec.empty() || isUrl(spec) || spec.find_first_of("@;") != std::string::npos) return false;
            if (spec.find_first_of("/\\") != std::string::npos || spec.front() == '.') return true;
            const std::string l = lower(spec);
            for (const std::string_view archive : {".whl", ".zip", ".tar.gz", ".tgz", ".tar.bz2"})
                if (l.size() > archive.size() && std::string_view(l).substr(l.size() - archive.size()) == archive) return true;
            return false;
        }

        // A package as the installer gets it and the marker keeps it: without
        // a comment and the whitespace around it, a local file made
        // absolute, and otherwise as it was spelled (a URL or a file name can
        // be case-sensitive; only the fingerprint compares normalised names).
        std::string packageArgument(const std::string& spec) {
            const std::string s = trim(withoutComment(spec));
            return isLocalPackage(s) ? absolutePath(s) : s;
        }

        // One entry per package however often and however it is spelled
        // (the first spelling wins), in the order of the normalised names.
        std::vector<std::string> uniquePackages(const std::vector<std::string>& specs) {
            std::map<std::string, std::string> byName;
            for (const std::string& s : specs)
                if (!s.empty()) byName.emplace(normaliseRequirement(s), s);
            std::vector<std::string> out;
            for (const auto& entry : byName) out.push_back(entry.second);
            return out;
        }

        // The entries of a requirements file, normalised; nullopt when the
        // file cannot be read. CRLF and a UTF-8 byte order mark are the same
        // file as LF and none (core.autocrlf checkouts, Notepad).
        std::optional<std::vector<std::string>> readRequirementFile(const std::string& path) {
            std::string text;
            if (path.empty() || !host::readFile(path, text)) return std::nullopt;
            if (startsWith(text, "\xEF\xBB\xBF")) text.erase(0, 3);
            std::vector<std::string> out;
            std::size_t begin = 0;
            while (begin <= text.size()) {
                std::size_t end = text.find('\n', begin);
                if (end == std::string::npos) end = text.size();
                const std::string entry = normaliseRequirement(text.substr(begin, end - begin));
                if (!entry.empty()) out.push_back(entry);
                begin = end + 1;
            }
            return out;
        }

        std::string requirementsFile(const std::string& scriptDir) { return scriptDir.empty() ? std::string() : scriptDir + "/requirements.txt"; }
        std::string extraRequirementsFile(const std::string& scriptDir) {
            return scriptDir.empty() ? std::string() : scriptDir + "/requirements-extra.txt";
        }

        // What the worker needs when the files are not where the worker is
        // (a stripped install): the same lists as the files ship with.
        const std::vector<std::string> kFallbackRequired{"numpy"};
        const std::vector<std::string> kFallbackExtras{"scipy", "scikit-image"};

        std::vector<std::string> normalisedPackages(const std::vector<std::string>& packages) {
            std::set<std::string> unique;
            for (const std::string& p : packages) {
                const std::string n = normaliseRequirement(p);
                if (!n.empty()) unique.insert(n);
            }
            return {unique.begin(), unique.end()};
        }

        // The distribution a requirement names ("numpy>=2" -> "numpy"),
        // spelled as PEP 503 compares names, so that it can be looked up in
        // the marker's package versions.
        // A direct URL or a path ("https://user:pass@host/x.whl") names no
        // distribution before it is fetched; it stands for itself, redacted,
        // because this name reaches the plan's warnings and the log, and cut
        // at its '@' it would keep the password and lose the "@" that
        // redactUrl looks for.
        std::string distributionName(const std::string& requirement) {
            std::string name;
            for (const char c : requirement) {
                if (std::string_view("<>=!~;[@ (").find(c) != std::string_view::npos) break;
                if (c == ':' || c == '/' || c == '\\') return redactUrl(trim(requirement));
                name.push_back(c == '_' || c == '.' ? '-' : static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            return name;
        }

        // A version from `packages` (the marker's, keyed as the worker names
        // the distributions) by a requirement's distribution name.
        std::optional<std::string> packageVersion(const std::map<std::string, std::string>& packages, const std::string& distribution) {
            for (const auto& [name, version] : packages)
                if (distributionName(name) == distribution) return version;
            return std::nullopt;
        }

        // Rough download sizes for the prompts ("about 13 MB"): the wheels of
        // the current releases for 64-bit Windows and Linux, dependencies
        // included. Anything else is "size unknown".
        std::optional<std::uint64_t> approximateSize(const std::string& distribution) {
            if (distribution == "numpy") return 13'000'000ULL;
            if (distribution == "scipy") return 38'000'000ULL;
            if (distribution == "scikit-image") return 19'000'000ULL;   // with pillow, imageio, networkx, tifffile
            return std::nullopt;
        }

        // --- child processes ---------------------------------------------------------

        // A setup that is cancelled first closes the installer's stdin and
        // waits this long before the whole tree is ended (section 2.4 of the
        // design): pip and uv never read stdin, so in practice this is how
        // long a cancel takes at most.
        constexpr int kStopGraceMs = 3000;
        constexpr int kProbeTimeoutMs = 15000;
        constexpr int kCheckTimeoutMs = 60000;
        constexpr int kInstallerTimeoutMs = 30 * 60 * 1000;   // a slow mirror, scipy over a bad link

        // What one child process did.
        struct RunOutcome {
            bool started = false;
            bool notFound = false;
            bool timedOut = false;
            bool cancelled = false;
            int exitCode = -1;
            std::string error;                // why it did not start
            std::vector<std::string> lines;   // stdout, and stderr too when merged
            std::vector<std::string> errors;  // stderr when not merged
            bool ok() const { return started && !timedOut && !cancelled && exitCode == 0; }
        };

        // Runs a child to its end, `timeoutMs` at most (0 = no limit),
        // handing each line to `onLine` on this thread. `cancelled` is asked
        // between reads; a cancel or a timeout stops the child (its whole
        // tree with killTree).
        RunOutcome runChild(const ChildProcess::Options& options, int timeoutMs,
                            const std::function<void(const std::string&)>& onLine = {},
                            const std::function<bool()>& cancelled = {}) {
            RunOutcome r;
            std::mutex errorMutex;
            std::vector<std::string> errors;
            ChildProcess child;
            if (!options.mergeErrorLines)
                child.setErrorHandler([&errorMutex, &errors](const std::string& line) {
                    const std::lock_guard<std::mutex> g(errorMutex);
                    errors.push_back(line);
                });
            std::string error;
            if (!child.start(options, &error)) {
                r.error = error.empty() ? "cannot start " + options.program : error;
                r.notFound = child.programNotFound();
                return r;
            }
            r.started = true;
            child.closeInput();
            const auto take = [&](std::string& line) {
                if (onLine) onLine(line);
                r.lines.push_back(std::move(line));
            };
            const auto begin = std::chrono::steady_clock::now();
            std::string line;
            for (;;) {
                if (cancelled && cancelled()) {
                    r.cancelled = true;
                    break;
                }
                if (timeoutMs > 0 && std::chrono::steady_clock::now() - begin >= std::chrono::milliseconds(timeoutMs)) {
                    r.timedOut = true;
                    break;
                }
                if (child.readLine(line, 200)) {
                    take(line);
                    continue;
                }
                if (!child.running()) {
                    while (child.readLine(line, 200)) take(line);   // what was still in the pipe
                    break;
                }
            }
            int grace = 0;   // it has ended by itself
            if (r.cancelled) grace = kStopGraceMs;
            else if (r.timedOut) grace = 500;
            child.stop(grace);
            while (child.readLine(line, 0)) take(line);   // stderr lines queued while the readers finished
            r.exitCode = child.exitCode();
            {
                const std::lock_guard<std::mutex> g(errorMutex);
                r.errors = std::move(errors);
            }
            return r;
        }

        // What no installer or probe needs from this process's environment:
        // the application's secrets, and the Hugging Face token (pip and uv
        // fetch nothing from the Hub).
        std::vector<std::string> installerUnset() {
            std::vector<std::string> out = secretEnvironmentNames();
            out.insert(out.end(), {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"});
            return out;
        }

        // A Python child: UTF-8 output, unbuffered, so lines arrive as they
        // are written and decode the same everywhere.
        ChildProcess::Options pythonOptions(const std::string& python, std::vector<std::string> arguments,
                                            const std::string& workingDirectory = std::string()) {
            ChildProcess::Options o;
            o.program = python;
            o.arguments = std::move(arguments);
            o.workingDirectory = workingDirectory;
            o.environment = {{"PYTHONIOENCODING", "utf-8"}, {"PYTHONUNBUFFERED", "1"}};
            o.unsetEnvironment = installerUnset();
            return o;
        }

        // The environment's own interpreter. A PYTHONHOME left in the
        // environment (by another Python installation) would point it at a
        // foreign standard library.
        ChildProcess::Options environmentPythonOptions(const std::string& envPython, std::vector<std::string> arguments,
                                                       const std::string& workingDirectory = std::string()) {
            ChildProcess::Options o = pythonOptions(envPython, std::move(arguments), workingDirectory);
            o.unsetEnvironment.emplace_back("PYTHONHOME");
            return o;
        }

        // The last line of `lines` that is a JSON object; null when none is.
        json lastJsonObject(const std::vector<std::string>& lines) {
            for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
                const std::string t = trim(*it);
                if (t.empty() || t.front() != '{') continue;
                json j = json::parse(t, nullptr, false);
                if (j.is_object()) return j;
            }
            return json();
        }

        // The last few lines a child wrote, for a message.
        std::string tail(const RunOutcome& r, std::size_t count = 3) {
            const std::vector<std::string>& source = r.errors.empty() ? r.lines : r.errors;
            std::vector<std::string> kept;
            for (auto it = source.rbegin(); it != source.rend() && kept.size() < count; ++it) {
                const std::string t = trim(*it);
                if (!t.empty()) kept.insert(kept.begin(), t);
            }
            return join(kept, " | ");
        }

        // Whether `<python> -I -c "import sys"` runs: a venv whose base
        // interpreter is gone fails here ("No Python at ...", or a uv
        // trampoline that cannot find its target).
        bool basicRun(const std::string& python, std::string* problem, const std::function<bool()>& cancelled = {}) {
            const RunOutcome r = runChild(pythonOptions(python, {"-I", "-c", "import sys"}), kProbeTimeoutMs, {}, cancelled);
            if (r.ok()) return true;
            if (problem) {
                if (r.cancelled) *problem = "cancelled";
                else if (!r.started) *problem = r.error;
                else if (r.timedOut) *problem = "it did not answer within " + std::to_string(kProbeTimeoutMs / 1000) + " s";
                else {
                    const std::string t = tail(r);
                    *problem = "exit code " + std::to_string(r.exitCode) + (t.empty() ? std::string() : ": " + t);
                }
            }
            return false;
        }

        bool hasWorkerScripts(const std::string& scriptDir) {
            return !scriptDir.empty() && host::isFile(scriptDir + "/sirius_worker/__main__.py");
        }

        // `<envpy> -m sirius_worker --check` in `scriptDir` (without -I, so
        // that the worker is found there): what the environment has for the
        // worker, as the worker itself sees it. Null when it did not answer.
        json workerCheck(const std::string& envPython, const std::string& scriptDir, std::string* problem,
                         const std::function<bool()>& cancelled = {}) {
            const RunOutcome r = runChild(environmentPythonOptions(envPython, {"-m", "sirius_worker", "--check"}, scriptDir),
                                          kCheckTimeoutMs, {}, cancelled);
            json j = lastJsonObject(r.lines);
            if (!r.cancelled && j.is_object() && j.contains("missing") && j["missing"].is_array()) return j;
            if (problem) {
                if (r.cancelled) *problem = "cancelled";
                else if (!r.started) *problem = r.error;
                else if (r.timedOut) *problem = "the worker's check did not answer within " + std::to_string(kCheckTimeoutMs / 1000) + " s";
                else {
                    const std::string t = tail(r);
                    *problem = "the worker's check failed (exit code " + std::to_string(r.exitCode) + ")" + (t.empty() ? std::string() : ": " + t);
                }
            }
            return json();
        }

        std::vector<std::string> stringList(const json& j) {
            std::vector<std::string> out;
            if (j.is_array())
                for (const json& v : j)
                    if (v.is_string()) out.push_back(v.get<std::string>());
            return out;
        }

        // --- the lock -----------------------------------------------------------------

        // 1 when the file was created with `content`, 0 when it exists
        // already, -1 on any other error (in `error`).
        int createExclusive(const std::string& path, const std::string& content, std::string& error) {
#ifdef _WIN32
            int fd = -1;
            const errno_t e = ::_wsopen_s(&fd, fsPath(path).c_str(), _O_CREAT | _O_EXCL | _O_WRONLY | _O_BINARY, _SH_DENYNO,
                                          _S_IREAD | _S_IWRITE);
            if (e != 0) {
                if (e == EEXIST) return 0;
                error = std::error_code(e, std::generic_category()).message();
                return -1;
            }
            const int written = ::_write(fd, content.data(), static_cast<unsigned>(content.size()));
            ::_close(fd);
            if (written != static_cast<int>(content.size())) error = "short write";
#else
            const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
            if (fd < 0) {
                if (errno == EEXIST) return 0;
                error = std::error_code(errno, std::generic_category()).message();
                return -1;
            }
            const ssize_t written = ::write(fd, content.data(), content.size());
            ::close(fd);
            if (written != static_cast<ssize_t>(content.size())) error = "short write";
#endif
            return 1;
        }

        // Renames `from` to `to` unless `to` exists, keeping the file itself:
        // a holder that is still writing it writes into it where it belongs.
        bool renameNoReplace(const std::string& from, const std::string& to) {
#ifdef _WIN32
            // The C runtime's rename never replaces an existing file.
            return ::_wrename(fsPath(from).c_str(), fsPath(to).c_str()) == 0;
#else
            if (::link(from.c_str(), to.c_str()) != 0) return false;
            ::unlink(from.c_str());
            return true;
#endif
        }

        // <env>.lock, held while a setup or a removal runs, so that two SIRIUS
        // processes (the GUI and sirius-cli, or two windows) never change the
        // environment at once. It is created exclusively and names its
        // holder; one whose holder is gone (a crash) or that is older than an
        // hour is stale, is taken over, and the lock is tried again.
        class SetupLock {
        public:
            SetupLock() = default;
            SetupLock(const SetupLock&) = delete;
            SetupLock& operator=(const SetupLock&) = delete;
            ~SetupLock() { release(); }

            // False with `holder` set to the pid that holds it (0 when that
            // is unknown), or with `error` set when it cannot be made at all.
            bool acquire(const std::string& envDir, int& holder, std::string& error) {
                path_ = envDir + ".lock";
                const json content = {{"pid", host::processId()}, {"started", utcNow()}, {"host", hostName()}};
                for (int attempt = 0; attempt < 3; ++attempt) {
                    const int made = createExclusive(path_, content.dump(), error);
                    if (made > 0) {
                        held_ = true;
                        content_ = content.dump();
                        touched_ = std::chrono::steady_clock::now();
                        return true;
                    }
                    if (made < 0) return false;
                    Seen seen;
                    if (!stale(holder, seen) || !takeOver(seen)) return false;
                }
                return false;
            }

            // Deletes the lock file only while it is still this process's:
            // one that another process took over (stale by age, or by the
            // race in takeOver) is that process's lock now.
            void release() {
                if (!held_) return;
                held_ = false;
                if (!ours()) return;
                std::error_code ec;
                fs::remove(fsPath(path_), ec);
            }

            // Keeps the lock from reading as stale by age (an hour) while a
            // long setup runs: its file's time is brought forward at most
            // once a minute, at every installer step and output line.
            void touch() {
                const auto now = std::chrono::steady_clock::now();
                if (!held_ || now - touched_ < std::chrono::minutes(1)) return;
                touched_ = now;
                if (!ours()) return;
                std::error_code ec;
                fs::last_write_time(fsPath(path_), fs::file_time_type::clock::now(), ec);
            }

        private:
            // The lock file as stale() judged it.
            struct Seen {
                std::string text;
                fs::file_time_type written{};
            };

            // Two processes can find the same lock stale (the GUI and
            // sirius-cli after a crash). The first takes it over and makes its
            // own; the second must not delete that fresh one. So the stale
            // file is first moved to a name of this process's own, and only
            // deleted when it is still the file that was judged: anything else
            // is somebody's fresh lock, which goes back. False in that case.
            bool takeOver(const Seen& seen) const {
                const std::string aside = path_ + "." + std::to_string(host::processId()) + ".stale";
                std::error_code ec;
                fs::rename(fsPath(path_), fsPath(aside), ec);
                if (ec) return true;   // gone already, or still being written: the next attempt tells
                std::string text;
                std::error_code timeError;
                const auto written = fs::last_write_time(fsPath(aside), timeError);
                const bool same = !timeError && written == seen.written && host::readFile(aside, text) && text == seen.text;
                if (same) {
                    fs::remove(fsPath(aside), ec);
                    return true;
                }
                // Somebody's fresh lock goes back. When it cannot (a third
                // process made yet another lock meanwhile, or a filesystem
                // without hard links), it stays where it is rather than being
                // deleted: its holder still runs, and its release() finds
                // the lock file no longer its own and leaves the other alone.
                renameNoReplace(aside, path_);
                return false;
            }

            bool stale(int& holder, Seen& seen) const {
                holder = 0;
                std::error_code ec;
                seen.written = fs::last_write_time(fsPath(path_), ec);
                if (ec) return true;   // gone in the meantime
                const auto age = fs::file_time_type::clock::now() - seen.written;
                std::string& text = seen.text;
                const json j = host::readFile(path_, text) ? json::parse(text, nullptr, false) : json();
                if (!j.is_object() || !j.contains("pid") || !j["pid"].is_number_integer()) {
                    // Being written this very moment, or left half-written by
                    // a crash; the second only once it has aged a little.
                    return age > std::chrono::seconds(10);
                }
                holder = j["pid"].get<int>();
                if (age > std::chrono::hours(1)) return true;
                // Another machine's pid (a home directory shared across the
                // nodes of a cluster) cannot be looked up here.
                const std::string owner = j.contains("host") && j["host"].is_string() ? j["host"].get<std::string>() : std::string();
                const bool here = owner.empty() || lower(owner) == lower(hostName());
                return here && holder > 0 && !host::processAlive(holder);
            }

            bool ours() const {
                std::string text;
                return host::readFile(path_, text) && text == content_;
            }

            std::string path_;
            std::string content_;
            std::chrono::steady_clock::time_point touched_{};
            bool held_ = false;
        };

        // Whether `dir` holds nothing but what a virtual environment (and
        // this module) put there. Recreate and Remove delete such a directory
        // only: one that holds anything else is somebody's own
        // ($SIRIUS_PYTHON_ENV pointed at the wrong place) and is never touched.
        // `foreign` gets the first entry that does not belong, for the message.
        bool looksLikeEnvironment(const std::string& dir, std::string* foreign = nullptr) {
            // .lock is uv's, taken while `uv pip install` changes the environment.
            // .DS_Store, desktop.ini and Thumbs.db are what Finder and Explorer
            // leave in a folder that was opened (Preferences' [Open folder]).
            static const std::set<std::string> names{"bin", "include", "lib", "lib64", "scripts", "share",
                                                     "etc", "man", "pyvenv.cfg", ".gitignore", "cachedir.tag",
                                                     ".lock", ".ds_store", "desktop.ini", "thumbs.db"};
            std::error_code ec;
            fs::directory_iterator it(fsPath(dir), ec);
            bool any = false, identified = false;
            // increment(ec), not a range-for, whose increment throws.
            for (; !ec && it != fs::directory_iterator(); it.increment(ec)) {
                const std::string given = utf8(it->path().filename());
                const std::string name = lower(given);
                any = true;
                // the marker and the README, and what an interrupted atomic write leaves
                if (names.count(name) == 0 && !startsWith(name, kMarkerFile) && !startsWith(name, "readme.txt")) {
                    if (foreign) *foreign = given.empty() ? std::string("a file whose name is not valid Unicode") : given;
                    return false;
                }
                if (name == "pyvenv.cfg" || name == kMarkerFile) identified = true;
            }
            // Names alone are not enough: ~/.local holds bin, lib, share and
            // include too. A directory is an environment when a venv
            // (pyvenv.cfg) or this module (the marker) says so; an empty one
            // loses nothing.
            if (!ec && any && !identified) {
                if (foreign) *foreign = "no pyvenv.cfg or " + std::string(kMarkerFile);
                return false;
            }
            return !ec;
        }

        // " (such as .vscode)" for the messages that refuse a directory.
        std::string foreignText(const std::string& foreign) {
            if (foreign.empty()) return std::string();
            return startsWith(foreign, "no pyvenv.cfg") ? " (it has " + foreign + ")" : " (such as " + foreign + ")";
        }

        // The message that refuses to `action` ("remove", "replace") `dir`,
        // or nothing when it is a Python environment.
        std::optional<std::string> refuseDirectory(const std::string& dir, const char* action) {
            std::string foreign;
            if (looksLikeEnvironment(dir, &foreign)) return std::nullopt;
            return dir + " holds files that are not a Python environment" + foreignText(foreign) + "; SIRIUS does not " + action + " it.";
        }

        // Deletes a directory tree, trying again for a moment: on Windows the
        // files of a process that was just ended stay locked for a while.
        bool removeTreeRetry(const std::string& dir, std::string* error = nullptr) {
            for (int attempt = 0;; ++attempt) {
                if (!exists(dir) || host::removeTree(dir, error)) return true;
                if (attempt >= 10) return false;
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        }

        // --- failures -------------------------------------------------------------------

        std::string noPythonHint() {
#if defined(_WIN32)
            return "Install Python 3 from python.org (tick \"Add python.exe to PATH\") or with `winget install Python.Python.3.13`, "
                   "then try again.";
#elif defined(__APPLE__)
            return "Install Python 3 from python.org or with `brew install python`, then try again.";
#else
            return "Install Python 3 with its venv module (for example `sudo apt install python3 python3-venv`), then try again.";
#endif
        }

        std::string noEnsurepipHint(int minor) {
#if defined(_WIN32) || defined(__APPLE__)
            (void)minor;
            return "Install uv (https://docs.astral.sh/uv/), or a Python from python.org, then try again.";
#else
            const std::string package = minor > 0 ? "python3." + std::to_string(minor) + "-venv" : std::string("python3-venv");
            return "Install its venv support (`sudo apt install " + package + "`), or install uv, then try again.";
#endif
        }

        constexpr const char* kUnsupportedHint = "Choose another interpreter (--base-python, or the Python list in the dialog).";
        constexpr const char* kInUseMessage = "Another SIRIUS window or sirius-cli is using the environment; close it and try again.";

        // The host an index URL points at, for "Could not reach <host>"; a
        // local directory (--find-links with --no-index) as it is.
        std::string indexHost(const std::string& index) {
            if (!index.empty() && !isUrl(index)) return index;
            std::string h = index;
            const std::size_t scheme = h.find("://");
            if (scheme != std::string::npos) h = h.substr(scheme + 3);
            const std::size_t at = h.find('@');
            const std::size_t slash = h.find('/');
            if (at != std::string::npos && (slash == std::string::npos || at < slash)) h = h.substr(at + 1);
            h = h.substr(0, h.find('/'));
            return h.empty() ? std::string("pypi.org") : h;
        }

        // --- installer progress ---------------------------------------------------------

        // progressFromLine with memory: uv's downloads advance from 0.30 to
        // 0.80 as "Downloaded" lines follow their "Downloading" ones, and the
        // fraction never goes back (pip collects packages one after another).
        class InstallProgress {
        public:
            explicit InstallProgress(bool uv) : uv_(uv) {}
            std::optional<double> feed(const std::string& raw) {
                const std::string line = trim(raw);
                std::optional<double> f = progressFromLine(line, uv_);
                if (!f) return std::nullopt;
                if (uv_ && startsWith(line, "Downloading ")) ++downloads_;
                if (uv_ && startsWith(line, "Downloaded ")) ++downloaded_;
                if (uv_ && (startsWith(line, "Downloading ") || startsWith(line, "Downloaded ")))
                    f = 0.30 + 0.50 * static_cast<double>(downloaded_) / static_cast<double>(std::max(downloads_, downloaded_));
                if (*f <= last_) return std::nullopt;
                last_ = *f;
                return f;
            }

        private:
            bool uv_ = false;
            int downloads_ = 0, downloaded_ = 0;
            double last_ = 0.0;
        };

        // --- planning -----------------------------------------------------------------

        // probe(), which a cancel stops (defined with it below).
        std::optional<PythonInfo> probeWith(const std::string& python, int timeoutMs, std::string* error,
                                            const std::function<bool()>& cancelled);

        const char* modeName(Mode m) {
            switch (m) {
                case Mode::Auto: return "auto";
                case Mode::Create: return "create";
                case Mode::Update: return "update";
                case Mode::Recreate: return "recreate";
            }
            return "create";
        }

        // Everything setup() needs besides the public plan.
        struct Planned {
            SetupPlan plan;
            std::string envPython;
            bool extras = false;
            std::vector<std::string> extraPackages;   // as the installer gets them (packageArgument)
            std::optional<PythonInfo> base;
            std::optional<Marker> marker;
            std::vector<std::string> createCommand, installCommand;
            Failure failure = Failure::None;   // what stops the setup before it changes anything
            std::string message, hint;
        };

        std::vector<std::string> indexArguments(const SetupOptions& options) {
            std::vector<std::string> out;
            // The index URL is not among them: installerIndexEnvironment.
            for (const std::string& d : options.findLinks) out.insert(out.end(), {"--find-links", d});
            if (options.noIndex) out.push_back("--no-index");
            return out;
        }

        // The index the packages come from, as the prompt and the marker name
        // it (redacted).
        std::string effectiveIndex(const SetupOptions& options, bool uv) {
            if (options.noIndex) {
                std::vector<std::string> links;
                for (const std::string& d : options.findLinks) links.push_back(redactUrl(d));
                return links.empty() ? std::string("none (--no-index)") : join(links, ", ");
            }
            if (!options.indexUrl.empty()) return redactUrl(options.indexUrl);
            for (const char* name : uv ? std::vector<const char*>{"UV_DEFAULT_INDEX", "UV_INDEX_URL"} : std::vector<const char*>{"PIP_INDEX_URL"}) {
                const std::string v = trim(host::environment(name));
                if (!v.empty()) return redactUrl(v);
            }
            return "https://pypi.org/simple";
        }

        // The base interpreter for a new environment, in the order of the
        // design (2.4): the one asked for, the one the environment was made
        // from, the one found on PATH, then the other candidates. The first
        // usable one wins; when none is, the first that answered is kept so
        // that the failure can say why it cannot be used. A cancel stops the
        // search with nothing chosen.
        void chooseBase(Planned& p, const SetupOptions& options, const std::function<bool()>& cancelled) {
            if (!options.basePython.empty()) {
                std::string error;
                p.plan.basePython = options.basePython;
                p.base = probeWith(options.basePython, kProbeTimeoutMs, &error, cancelled);
                if (!p.base) {
                    p.failure = Failure::UnsupportedPython;
                    p.message = options.basePython + " is not a usable Python 3 (" + error + ").";
                    p.hint = kUnsupportedHint;
                }
                return;
            }
            std::vector<std::string> order;
            if (p.marker && !p.marker->baseExecutable.empty()) order.push_back(p.marker->baseExecutable);
            const std::string found = host::findPython();
            if (!found.empty()) order.push_back(found);
            bool candidatesAdded = false;
            std::set<std::string> tried;
            std::optional<PythonInfo> firstAnswer;
            std::string firstAnswerPath;
            for (std::size_t i = 0; i < order.size() || !candidatesAdded; ++i) {
                if (cancelled && cancelled()) return;
                if (i >= order.size()) {
                    const std::vector<std::string> more = pythonCandidates();
                    order.insert(order.end(), more.begin(), more.end());
                    candidatesAdded = true;
                    if (i >= order.size()) break;
                }
                const std::string& python = order[i];
                if (!tried.insert(lower(python)).second) continue;
                std::optional<PythonInfo> info = probeWith(python, kProbeTimeoutMs, nullptr, cancelled);
                if (!info) continue;
                if (info->problem.empty()) {
                    p.plan.basePython = python;
                    p.base = std::move(info);
                    return;
                }
                if (!firstAnswer) {
                    firstAnswer = std::move(info);
                    firstAnswerPath = python;
                }
            }
            if (firstAnswer) {
                p.plan.basePython = firstAnswerPath;
                p.base = std::move(firstAnswer);
                return;
            }
            p.failure = Failure::NoPython;
            p.message = "No Python 3 interpreter was found.";
            p.hint = noPythonHint();
        }

        Planned planInternal(const SetupOptions& given, const std::string& givenScriptDir, const std::function<bool()>& cancelled) {
            Planned p;
            SetupPlan& plan = p.plan;
            plan.envDir = environmentDirectory();
            p.envPython = environmentPython(plan.envDir);
            if (plan.envDir.empty()) {
                p.failure = Failure::Failed;
                p.message = "There is no directory for SIRIUS's Python environment (the user's data directory is unknown).";
                p.hint = "Set SIRIUS_PYTHON_ENV to the directory it should use.";
                return p;
            }
            const bool dirExists = host::isDirectory(plan.envDir);
            if (dirExists) p.marker = readMarker(plan.envDir);
            const bool complete = p.marker.has_value() && host::isFile(p.envPython);

            // Every path the installers get, as they can read it from where
            // they run (absolutePath). A URL stays as it is.
            const std::string scriptDir = absolutePath(givenScriptDir);
            SetupOptions options = given;
            options.basePython = absoluteProgram(given.basePython);
            for (std::string& d : options.findLinks)
                if (!isUrl(d)) d = absolutePath(d);

            // What to install: what was asked for, plus, when the environment
            // is only brought up to date, what it already has; an update
            // never takes packages away. Something that starts with '-' would
            // reach the installer as one of its options (--index-url=...).
            p.extras = options.extras;
            std::vector<std::string> extra, refused;
            for (const std::string& spec : given.extraPackages) {
                const std::string a = packageArgument(spec);
                if (a.empty()) continue;
                if (a.front() == '-') refused.push_back(a);
                else extra.push_back(a);
            }
            const bool keepsPackages = options.mode == Mode::Auto || options.mode == Mode::Update;
            if (p.marker && keepsPackages) {
                p.extras = p.extras || p.marker->extras;
                for (const std::string& spec : p.marker->extraPackages) {
                    const std::string a = packageArgument(spec);
                    if (a.empty()) continue;
                    if (a.front() == '-') plan.warnings.push_back("\"" + redactUrl(a) + "\" in " + kMarkerFile + " is not a package; it is left out.");
                    else extra.push_back(a);
                }
            }
            p.extraPackages = uniquePackages(extra);
            plan.packages = requirements(scriptDir, p.extras, p.extraPackages);
            const std::string fingerprint = requirementsFingerprint(scriptDir, p.extras, p.extraPackages);
            // A create or a recreate that was asked for makes what was asked
            // for (the dialog offers the extras again), which is said when it
            // leaves out what the environment has now.
            if (p.marker && !keepsPackages) {
                std::vector<std::string> dropped;
                for (const std::string& r : requirements(scriptDir, p.marker->extras, p.marker->extraPackages))
                    if (!std::binary_search(plan.packages.begin(), plan.packages.end(), r)) dropped.push_back(redactUrl(distributionName(r)));
                if (!dropped.empty())
                    plan.warnings.push_back("The environment is made again without " + join(dropped, ", ") + ", which it has now; ask for " +
                                            (dropped.size() == 1 ? "it" : "them") + " again to keep " + (dropped.size() == 1 ? "it." : "them."));
            }

            // The mode (2.4, step 2).
            std::string runProblem;
            switch (options.mode) {
                case Mode::Auto:
                    if (!dirExists) plan.mode = Mode::Create;
                    else if (!complete) plan.mode = Mode::Recreate;
                    else if (!basicRun(p.envPython, &runProblem, cancelled)) {
                        plan.mode = Mode::Recreate;
                        plan.warnings.push_back("The environment no longer runs (" + runProblem + "); it is made again.");
                    } else if (p.marker->fingerprint != fingerprint) {
                        plan.mode = Mode::Update;
                    } else {
                        // Up to date but lacking a required package is Broken,
                        // made again (2.4, step 2) rather than installed into:
                        // what took the package away may have damaged its pip too.
                        plan.mode = Mode::Update;
                        std::string checkProblem;
                        const json check = hasWorkerScripts(scriptDir) ? workerCheck(p.envPython, scriptDir, &checkProblem, cancelled) : json();
                        const std::vector<std::string> missing = check.is_object() ? stringList(check["missing"]) : std::vector<std::string>();
                        plan.nothingToDo = check.is_object() && missing.empty();
                        if (!missing.empty()) {
                            plan.mode = Mode::Recreate;
                            plan.warnings.push_back("The environment lacks " + join(missing, ", ") + "; it is made again.");
                        }
                    }
                    break;
                case Mode::Create:
                    plan.mode = dirExists ? Mode::Recreate : Mode::Create;
                    if (dirExists) plan.warnings.push_back("The environment exists already; it is made again.");
                    break;
                case Mode::Update:
                    if (!dirExists) {
                        plan.mode = Mode::Create;
                        plan.warnings.push_back("There is no environment to update; it is created.");
                    } else if (!complete) {
                        plan.mode = Mode::Recreate;
                        plan.warnings.push_back("The environment is incomplete; it is made again.");
                    } else if (!basicRun(p.envPython, &runProblem, cancelled)) {
                        plan.mode = Mode::Recreate;
                        plan.warnings.push_back("The environment no longer runs (" + runProblem + "); it is made again.");
                    } else {
                        plan.mode = Mode::Update;
                    }
                    break;
                case Mode::Recreate: plan.mode = dirExists ? Mode::Recreate : Mode::Create; break;
            }
            const bool creates = plan.mode != Mode::Update;
            if (!refused.empty()) {
                p.failure = Failure::Failed;
                p.message = "\"" + redactUrl(refused.front()) + "\" is not a package; the installer's options cannot be given as packages.";
                p.hint = "Name packages as pip does: numpy, torch==2.5, or the path of a wheel.";
            }

            // The installer.
            const std::string uv = options.useUv ? findUv() : std::string();
            plan.uv = uv;
            if (!uv.empty()) {
                const std::string version = uvVersion(uv);
                plan.installer = version.empty() ? std::string("uv") : "uv " + version;
            } else {
                plan.installer = "pip";
            }
            plan.index = effectiveIndex(options, !uv.empty());

            // The base interpreter, for a new environment only; an update
            // runs in the environment's own.
            if (creates && !plan.nothingToDo && p.failure == Failure::None) {
                chooseBase(p, options, cancelled);
                if (p.base) {
                    plan.basePythonVersion = p.base->version;
                    plan.externallyManagedBase = p.base->externallyManaged;
                    if (p.failure == Failure::None && !p.base->problem.empty()) {
                        p.failure = Failure::UnsupportedPython;
                        p.message = p.base->problem + ".";
                        p.message[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(p.message[0])));
                        p.hint = kUnsupportedHint;
                    }
                    if (p.failure == Failure::None && uv.empty() && !p.base->hasEnsurepip) {
                        p.failure = Failure::NoEnsurepip;
                        p.message = "Python " + p.base->version + " (" + plan.basePython + ") lacks venv support (ensurepip).";
                        p.hint = noEnsurepipHint(p.base->minor);
                    }
                    if (p.base->bits == 32)
                        plan.warnings.push_back("Python " + p.base->version + " is a 32-bit build; recent numpy releases have no 32-bit wheels.");
                }
            } else if (p.marker) {
                plan.basePython = p.marker->baseExecutable;
                plan.basePythonVersion = p.marker->pythonVersion;
            }
            if (!p.message.empty() && p.failure != Failure::None) plan.warnings.push_back(p.message);

            if (const auto refusal = creates && dirExists && p.failure == Failure::None ? refuseDirectory(plan.envDir, "replace") : std::nullopt) {
                p.failure = Failure::Failed;
                p.message = *refusal;
                p.hint = "Remove or move that directory yourself, or point SIRIUS_PYTHON_ENV at another one.";
                plan.warnings.push_back(p.message);
            }
            if (!hasWorkerScripts(scriptDir) && p.failure == Failure::None && !plan.nothingToDo) {
                p.failure = Failure::Failed;
                p.message = "The Python worker (sirius_worker) was not found in " + (scriptDir.empty() ? std::string("\"\"") : scriptDir) + ".";
                p.hint = "Reinstall SIRIUS, or name the worker's directory (--worker-dir).";
                plan.warnings.push_back(p.message);
            }
            {
                // Only a note for a dry run: setup() takes the lock itself
                // before it plans, and then this is its own.
                std::string text;
                if (host::readFile(plan.envDir + ".lock", text)) {
                    const json j = json::parse(text, nullptr, false);
                    const int holder = j.is_object() && j.contains("pid") && j["pid"].is_number_integer() ? j["pid"].get<int>() : 0;
                    if (holder != host::processId())
                        plan.warnings.push_back("Another setup holds " + plan.envDir + ".lock" +
                                                (holder > 0 ? " (pid " + std::to_string(holder) + ")" : std::string()) + ".");
                }
            }

            // What would be downloaded: everything for a new environment, what
            // the environment does not have yet for an update.
            std::vector<std::string> unknown;
            for (const std::string& requirement : plan.packages) {
                const std::string name = distributionName(requirement);
                if (!creates && p.marker && packageVersion(p.marker->packages, name)) continue;
                if (const auto size = approximateSize(name)) plan.approxDownloadBytes += *size;
                else unknown.push_back(name);
            }
            if (!unknown.empty() && !plan.nothingToDo) plan.warnings.push_back("The download size of " + join(unknown, ", ") + " is not known.");

            // The commands (2.4, steps 3 to 5).
            const std::vector<std::string> index = indexArguments(options);
            const std::string base = plan.basePython;
            if (creates && !base.empty()) {
                if (!uv.empty()) {
                    p.createCommand = {uv, "venv", "--no-config", "--seed", "--python", base};
                    p.createCommand.insert(p.createCommand.end(), index.begin(), index.end());
                    p.createCommand.push_back(plan.envDir);
                } else {
                    p.createCommand = {base, "-m", "venv", plan.envDir};
                }
            }
            if (!uv.empty())
                p.installCommand = {uv, "pip", "install", "--no-config", "--python", p.envPython, "--only-binary", ":all:"};
            else
                p.installCommand = {p.envPython, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--progress-bar", "off",
                                    "--retries", "2", "--timeout", "15", "--only-binary", ":all:"};
            const auto requirementSource = [&](const std::string& file, const std::vector<std::string>& fallback) {
                if (host::isFile(file)) p.installCommand.insert(p.installCommand.end(), {"-r", file});
                else p.installCommand.insert(p.installCommand.end(), fallback.begin(), fallback.end());
            };
            requirementSource(requirementsFile(scriptDir), kFallbackRequired);
            if (p.extras) requirementSource(extraRequirementsFile(scriptDir), kFallbackExtras);
            p.installCommand.insert(p.installCommand.end(), p.extraPackages.begin(), p.extraPackages.end());
            p.installCommand.insert(p.installCommand.end(), index.begin(), index.end());
            if (!plan.nothingToDo) {
                if (!p.createCommand.empty()) plan.commands.push_back(redacted(p.createCommand));
                plan.commands.push_back(redacted(p.installCommand));
                plan.commands.push_back({p.envPython, "-m", "sirius_worker", "--check"});
            }
            return p;
        }

        // --- macOS: the Xcode stub ------------------------------------------------------

#ifdef __APPLE__
        // /usr/bin/python3 is a stub until the Command Line Tools are
        // installed, and running it opens their installer instead.
        bool commandLineToolsInstalled() {
            static const bool installed = [] {
                ChildProcess::Options o;
                o.program = "/usr/bin/xcode-select";
                o.arguments = {"-p"};
                return runChild(o, 5000).ok();
            }();
            return installed;
        }
#endif

        // "python3.<minor><suffix>" (python3.13.exe, python3.12): the minor
        // version, or -1 for any other name (python3.13t.exe, python3.12-config).
        int versionedMinor(const std::string& name, std::string_view prefix, std::string_view suffix) {
            if (!startsWith(name, prefix)) return -1;
            std::size_t n = prefix.size();
            int minor = 0;
            while (n < name.size() && n - prefix.size() < 3 && std::isdigit(static_cast<unsigned char>(name[n])))
                minor = minor * 10 + (name[n++] - '0');
            if (n == prefix.size() || std::string_view(name).substr(n) != suffix) return -1;
            return minor;
        }

        // The minor version a directory name spells after `prefix`
        // ("cpython-3." 14 "-windows-x86_64-none", "Python3" 13 "-arm64"),
        // or -1. Free-threaded builds are left out: they cannot be a base.
        int directoryMinor(const std::string& name, std::string_view prefix) {
            if (!startsWith(name, prefix) || name.find("freethreaded") != std::string::npos) return -1;
            std::size_t n = prefix.size();
            int minor = 0;
            while (n < name.size() && n - prefix.size() < 3 && std::isdigit(static_cast<unsigned char>(name[n])))
                minor = minor * 10 + (name[n++] - '0');
            if (n == prefix.size() || (n < name.size() && name[n] == 't')) return -1;
            return minor;
        }

        // Entries of `dir` whose lower-cased name `minorOf` accepts, newest
        // first and, within one version, by name (uv's "cpython-3.14-..."
        // link before the "cpython-3.14.5-..." directory it points at).
        std::vector<fs::path> newestFirst(const fs::path& dir, const std::function<int(const std::string&)>& minorOf) {
            std::vector<std::pair<int, fs::path>> found;
            std::error_code ec;
            // increment(ec), not a range-for, whose increment throws.
            for (fs::directory_iterator it(dir, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
                const int minor = minorOf(lower(utf8(it->path().filename())));
                if (minor >= 0) found.emplace_back(minor, it->path());
            }
            std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) {
                return a.first != b.first ? a.first > b.first : utf8(a.second.filename()) < utf8(b.second.filename());
            });
            std::vector<fs::path> out;
            for (auto& f : found) out.push_back(std::move(f.second));
            return out;
        }

        std::vector<std::string> pathDirectories() {
#ifdef _WIN32
            const char separator = ';';
#else
            const char separator = ':';
#endif
            std::vector<std::string> out;
            const std::string path = host::environment("PATH");
            std::size_t begin = 0;
            while (begin <= path.size()) {
                std::size_t end = path.find(separator, begin);
                if (end == std::string::npos) end = path.size();
                std::string d = trim(path.substr(begin, end - begin));
                if (d.size() > 1 && d.front() == '"' && d.back() == '"') d = d.substr(1, d.size() - 2);
                if (!d.empty()) out.push_back(d);
                begin = end + 1;
            }
            return out;
        }

        // Where uv keeps the Pythons it installs.
        std::string uvPythonDirectory() {
            const std::string configured = host::environment("UV_PYTHON_INSTALL_DIR");
            if (!configured.empty()) return forwardSlashes(configured);
#ifdef _WIN32
            const std::string appdata = host::environment("APPDATA");
            return appdata.empty() ? std::string() : forwardSlashes(appdata) + "/uv/python";
#else
            const std::string xdg = host::environment("XDG_DATA_HOME");
            if (!xdg.empty() && xdg.front() == '/') return xdg + "/uv/python";
            const std::string home = host::homeDirectory();
            return home.empty() ? std::string() : home + "/.local/share/uv/python";
#endif
        }

    } // namespace

    // --- location and requirements --------------------------------------------------

    std::string environmentDirectory() {
        const std::string configured = trim(host::environment("SIRIUS_PYTHON_ENV"));
        if (!configured.empty()) {
            // A venv holds absolute paths: a relative name is fixed against
            // the working directory once, here.
            std::error_code ec;
            fs::path p = fs::absolute(fsPath(configured), ec);
            if (ec) p = fsPath(configured);
            std::string out = pathText(p.lexically_normal());
            while (out.size() > 1 && out.back() == '/' && !(out.size() == 3 && out[1] == ':')) out.pop_back();
            return out;
        }
        const std::string data = host::dataDirectory();
        if (data.empty()) return std::string();
        return forwardSlashes(data) + "/sirius/python-env";
    }

    std::string environmentPython(const std::string& envDir) {
        if (envDir.empty()) return std::string();
#ifdef _WIN32
        return envDir + "/Scripts/python.exe";
#else
        return envDir + "/bin/python";
#endif
    }

    std::vector<std::string> requirements(const std::string& scriptDir, bool extras, const std::vector<std::string>& extraPackages) {
        std::vector<std::string> all = readRequirementFile(requirementsFile(scriptDir)).value_or(kFallbackRequired);
        if (extras) {
            const std::vector<std::string> more = readRequirementFile(extraRequirementsFile(scriptDir)).value_or(kFallbackExtras);
            all.insert(all.end(), more.begin(), more.end());
        }
        all.insert(all.end(), extraPackages.begin(), extraPackages.end());
        return normalisedPackages(all);
    }

    std::string requirementsFingerprint(const std::string& scriptDir, bool extras, const std::vector<std::string>& extraPackages) {
        std::string text = "requirements\n";
        for (const std::string& r : requirements(scriptDir, extras, extraPackages)) text += r + "\n";
        text += extras ? "extras=1\n" : "extras=0\n";
        for (const std::string& p : normalisedPackages(extraPackages)) text += "package=" + p + "\n";
        std::uint64_t hash = 14695981039346656037ULL;   // FNV-1a 64
        for (const unsigned char c : text) {
            hash ^= c;
            hash *= 1099511628211ULL;
        }
        char buffer[17];
        std::snprintf(buffer, sizeof buffer, "%016llx", static_cast<unsigned long long>(hash));
        return buffer;
    }

    const std::vector<std::string>& optionalDistributions() {
        // The values of sirius_worker.OPTIONAL (app/python/sirius_worker/__init__.py);
        // a test checks that each one is spelled there.
        static const std::vector<std::string> distributions{"scipy", "scikit-image", "torch", "huggingface_hub", "onnxruntime",
                                                            "cellpose", "micro_sam", "btrack"};
        return distributions;
    }

    std::string redactUrl(const std::string& url) {
        constexpr std::size_t npos = std::string::npos;
        std::string out = url;
        std::size_t from = 0;
        for (;;) {
            const std::size_t scheme = out.find("://", from);
            if (scheme == npos) return out;
            const std::size_t start = scheme + 3;
            std::size_t end = out.find_first_of("/?# \t\"'", start);
            if (end == npos) end = out.size();
            const std::size_t at = out.rfind('@', end == 0 ? 0 : end - 1);
            if (at != npos && at >= start && at < end) out.replace(start, at - start, "***");
            // The query: every value, whatever its name (token=, key=, sig=, X-Amz-Signature=).
            std::size_t urlEnd = out.find_first_of(" \t\"'", start);
            if (urlEnd == npos) urlEnd = out.size();
            const std::size_t q = out.find('?', start);
            if (q != npos && q < urlEnd) {
                std::size_t limit = out.find('#', q);
                if (limit == npos || limit > urlEnd) limit = urlEnd;
                std::size_t i = q + 1;
                while (i < limit) {
                    std::size_t amp = out.find('&', i);
                    if (amp == npos || amp > limit) amp = limit;
                    const std::size_t eq = out.find('=', i);
                    if (eq != npos && eq < amp) {
                        const std::size_t length = amp - eq - 1;
                        if (out.compare(eq + 1, length, "***") != 0) {
                            out.replace(eq + 1, length, "***");
                            limit = limit + 3 - length;
                            urlEnd = urlEnd + 3 - length;
                            amp = eq + 4;
                        }
                    }
                    i = amp + 1;
                }
            }
            from = urlEnd;
        }
    }

    const std::vector<std::string>& secretEnvironmentNames() {
        static const std::vector<std::string> names{"SIRIUS_TOKEN", "SIRIUS_HPC_TOKEN", "SIRIUS_LLM_API_KEY", "OPENROUTER_API_KEY",
                                                    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY", "GEMINI_API_KEY",
                                                    "GOOGLE_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY",
                                                    "XAI_API_KEY", "TOGETHER_API_KEY", "COHERE_API_KEY", "CO_API_KEY"};
        return names;
    }

    std::vector<std::pair<std::string, std::string>> installerIndexEnvironment(const SetupOptions& options, bool uv) {
        if (options.indexUrl.empty()) return {};
        return {{uv ? "UV_INDEX_URL" : "PIP_INDEX_URL", options.indexUrl}};
    }

    // --- the marker ---------------------------------------------------------------------

    json Marker::toJson() const {
        return json{{"schema", schema},
                    {"created_by", createdBy},
                    {"created", created},
                    {"base_executable", baseExecutable},
                    {"python_version", pythonVersion},
                    {"installer", installer},
                    {"index", redactUrl(index)},
                    {"fingerprint", fingerprint},
                    {"extras", extras},
                    {"extra_packages", extraPackages},
                    {"packages", packages}};
    }

    std::optional<Marker> Marker::fromJson(const json& j) {
        if (!j.is_object() || !j.contains("schema") || !j["schema"].is_number_integer() || j["schema"].get<int>() < 1) return std::nullopt;
        Marker m;
        m.schema = j["schema"].get<int>();
        const auto text = [&j](const char* key) {
            const auto it = j.find(key);
            return it != j.end() && it->is_string() ? it->get<std::string>() : std::string();
        };
        m.createdBy = text("created_by");
        m.created = text("created");
        m.baseExecutable = text("base_executable");
        m.pythonVersion = text("python_version");
        m.installer = text("installer");
        m.index = text("index");
        m.fingerprint = text("fingerprint");
        if (const auto it = j.find("extras"); it != j.end() && it->is_boolean()) m.extras = it->get<bool>();
        if (const auto it = j.find("extra_packages"); it != j.end()) m.extraPackages = stringList(*it);
        if (const auto it = j.find("packages"); it != j.end() && it->is_object())
            for (const auto& [name, version] : it->items())
                if (version.is_string()) m.packages[name] = version.get<std::string>();
        return m;
    }

    std::optional<Marker> readMarker(const std::string& envDir) {
        std::string text;
        if (envDir.empty() || !host::readFile(envDir + "/" + kMarkerFile, text)) return std::nullopt;
        return Marker::fromJson(json::parse(text, nullptr, false));
    }

    namespace {
        // The marker as status and results show it. The file keeps a package
        // URL as it was given, since an update installs from it again; what
        // is shown or logged never holds its credentials.
        json shownMarker(const std::optional<Marker>& marker) {
            if (!marker) return json();
            json j = marker->toJson();
            j["extra_packages"] = redacted(marker->extraPackages);
            return j;
        }
    } // namespace

    // --- status ---------------------------------------------------------------------------

    const char* toString(State s) noexcept {
        switch (s) {
            case State::Absent: return "absent";
            case State::Incomplete: return "incomplete";
            case State::Ready: return "ready";
            case State::Outdated: return "outdated";
            case State::Broken: return "broken";
        }
        return "absent";
    }

    json EnvironmentStatus::toJson() const {
        return json{{"state", toString(state)},
                    {"dir", dir},
                    {"python", python},
                    {"problem", redactUrl(problem)},
                    {"requirements_current", requirementsCurrent},
                    {"marker", shownMarker(marker)}};
    }

    EnvironmentStatus environmentStatus(const std::string& scriptDir, bool runPython) {
        EnvironmentStatus s;
        s.dir = environmentDirectory();
        s.python = environmentPython(s.dir);
        if (s.dir.empty() || !host::isDirectory(s.dir)) {
            s.state = State::Absent;
            return s;
        }
        s.marker = readMarker(s.dir);
        if (!s.marker) {
            s.state = State::Incomplete;
            std::string foreign;
            s.problem = looksLikeEnvironment(s.dir, &foreign) ? "a setup did not finish (no " + std::string(kMarkerFile) + ")"
                                                              : "the directory holds files that are not a Python environment" + foreignText(foreign);
            return s;
        }
        if (!host::isFile(s.python)) {
            s.state = State::Incomplete;
            s.problem = "its Python is missing (" + s.python + ")";
            return s;
        }
        s.requirementsCurrent = s.marker->fingerprint == requirementsFingerprint(scriptDir, s.marker->extras, s.marker->extraPackages);
        s.state = s.requirementsCurrent ? State::Ready : State::Outdated;
        if (!s.requirementsCurrent) s.problem = "the requirements changed";
        if (!runPython) return s;
        // A broken environment needs to be made again, an outdated one only
        // updated: what runs decides first.
        std::string problem;
        if (!basicRun(s.python, &problem)) {
            s.state = State::Broken;
            s.problem = "its Python does not run (" + problem + ")";
            return s;
        }
        if (!hasWorkerScripts(scriptDir)) return s;
        const json check = workerCheck(s.python, scriptDir, &problem);
        if (!check.is_object()) {
            s.state = State::Broken;
            s.problem = problem;
            return s;
        }
        const std::vector<std::string> missing = stringList(check["missing"]);
        if (!missing.empty()) {
            s.state = State::Broken;
            s.problem = "missing packages: " + join(missing, ", ");
        }
        return s;
    }

    // --- which interpreter the worker runs --------------------------------------------------

    const char* toString(Source s) noexcept {
        switch (s) {
            case Source::Explicit: return "explicit";
            case Source::Environment: return "environment";
            case Source::Configured: return "configured";
            case Source::Managed: return "managed";
            case Source::Discovered: return "discovered";
            case Source::Fallback: return "fallback";
        }
        return "fallback";
    }

    Interpreter workerInterpreter(const std::string& explicitPython, const std::string& configured) {
        if (!trim(explicitPython).empty()) return {trim(explicitPython), Source::Explicit};
        const std::string fromEnvironment = trim(host::environment("SIRIUS_PYTHON"));
        if (!fromEnvironment.empty()) return {fromEnvironment, Source::Environment};
        if (!trim(configured).empty()) return {trim(configured), Source::Configured};
        // SIRIUS's own environment, even when it is outdated or broken: the
        // user is then offered an update or a repair, rather than the worker
        // falling back silently to another Python that lacks numpy.
        const std::string envDir = environmentDirectory();
        const std::string envPython = environmentPython(envDir);
        if (!envDir.empty() && host::isFile(envDir + "/" + kMarkerFile) && host::isFile(envPython)) return {envPython, Source::Managed};
        const std::string found = host::findPython();
        if (!found.empty()) return {found, Source::Discovered};
#ifdef _WIN32
        return {"python", Source::Fallback};
#else
        return {"python3", Source::Fallback};
#endif
    }

    // --- base interpreters --------------------------------------------------------------------

    json PythonInfo::toJson() const {
        return json{{"executable", executable},
                    {"base_executable", baseExecutable},
                    {"version", version},
                    {"major", major},
                    {"minor", minor},
                    {"bits", bits},
                    {"free_threaded", freeThreaded},
                    {"venv", venv},
                    {"externally_managed", externallyManaged},
                    {"pip", hasPip},
                    {"ensurepip", hasEnsurepip},
                    {"problem", problem}};
    }

    namespace {
        std::optional<PythonInfo> probeWith(const std::string& python, int timeoutMs, std::string* error, const std::function<bool()>& cancelled) {
        // Only syntax every Python 3 accepts, and only the standard library:
        // an old interpreter still reports its version, so that it can be
        // turned down by name. -I keeps the user's site-packages, PYTHON*
        // variables and the working directory out of it.
            static const char* kScript = R"PY(import sys, os, json
d = {"version": "%d.%d.%d" % tuple(sys.version_info[:3]), "major": sys.version_info[0], "minor": sys.version_info[1],
     "executable": sys.executable, "base_executable": getattr(sys, "_base_executable", None) or sys.executable,
     "bits": 64 if sys.maxsize > 2 ** 32 else 32, "venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
     "free_threaded": False, "externally_managed": False}
try:
    import sysconfig
    d["free_threaded"] = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    d["externally_managed"] = os.path.isfile(os.path.join(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED"))
except Exception:
    pass
def has(name):
    try:
        import importlib.util
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False
d["pip"] = has("pip")
d["ensurepip"] = has("ensurepip") and has("venv")
sys.stdout.write(json.dumps(d) + "\n")
)PY";
            if (trim(python).empty()) {
                if (error) *error = "no interpreter was named";
                return std::nullopt;
            }
            const RunOutcome r = runChild(pythonOptions(python, {"-I", "-c", kScript}), timeoutMs > 0 ? timeoutMs : kProbeTimeoutMs, {}, cancelled);
            if (r.cancelled) {
                if (error) *error = "cancelled";
                return std::nullopt;
            }
            if (!r.started) {
                if (error) *error = r.error;
                return std::nullopt;
            }
            if (r.timedOut) {
                if (error) *error = "it did not answer within " + std::to_string((timeoutMs > 0 ? timeoutMs : kProbeTimeoutMs) / 1000) + " s";
                return std::nullopt;
            }
            const json j = lastJsonObject(r.lines);
            if (!j.is_object() || !j.contains("version") || !j["version"].is_string()) {
                if (error) {
                    const std::string t = tail(r);
                    *error = "it is not a Python 3 that runs (exit code " + std::to_string(r.exitCode) + (t.empty() ? std::string() : ": " + t) + ")";
                }
                return std::nullopt;
            }
            PythonInfo info;
            const auto text = [&j](const char* key) { return j.contains(key) && j[key].is_string() ? j[key].get<std::string>() : std::string(); };
            const auto flag = [&j](const char* key) { return j.contains(key) && j[key].is_boolean() && j[key].get<bool>(); };
            const auto number = [&j](const char* key, int fallback) { return j.contains(key) && j[key].is_number_integer() ? j[key].get<int>() : fallback; };
            info.executable = forwardSlashes(text("executable"));
            info.baseExecutable = forwardSlashes(text("base_executable"));
            info.version = text("version");
            info.major = number("major", 0);
            info.minor = number("minor", 0);
            info.bits = number("bits", 64);
            info.freeThreaded = flag("free_threaded");
            info.venv = flag("venv");
            info.externallyManaged = flag("externally_managed");
            info.hasPip = flag("pip");
            info.hasEnsurepip = flag("ensurepip");
            if (info.major != 3 || info.minor < kMinMinor)
                info.problem = "Python " + std::to_string(info.major) + "." + std::to_string(info.minor) + " is older than 3." + std::to_string(kMinMinor);
            else if (info.freeThreaded)
                info.problem = "free-threaded builds are not supported";
            return info;
        }
    } // namespace

    std::optional<PythonInfo> probe(const std::string& python, int timeoutMs, std::string* error) { return probeWith(python, timeoutMs, error, {}); }

    std::vector<std::string> pythonCandidates() {
        std::vector<std::string> out;
        std::set<std::string> seen;
        // One interpreter once, however many names lead to it (a uv link and
        // the directory it points at, /usr/bin/python3 and python3.12).
        const auto add = [&](const fs::path& p) {
            const std::string text = pathText(p);
#ifdef _WIN32
            // %LOCALAPPDATA%\Microsoft\WindowsApps\python.exe is the Store's
            // alias, which prints an advertisement instead of running
            if (lower(text).find("/windowsapps/") != std::string::npos) return;
#endif
            std::error_code ec;
            if (!fs::is_regular_file(p, ec)) return;
            const fs::path resolved = fs::canonical(p, ec);
            std::string key = ec ? text : pathText(resolved);
#ifdef _WIN32
            key = lower(key);
#endif
            if (seen.insert(key).second) out.push_back(text);
        };
        const std::string home = host::homeDirectory();
#ifdef _WIN32
        const std::string exe = ".exe";
#else
        const std::string exe;
#endif
        const auto versioned = [&exe](const std::string& name) { return versionedMinor(name, "python3.", exe); };
        const auto pathEntries = [&]() {
            for (const std::string& dir : pathDirectories()) {
                const fs::path d = fsPath(dir);
#ifdef __APPLE__
                if (!commandLineToolsInstalled() && lower(pathText(d.lexically_normal())) == "/usr/bin") continue;
#endif
                for (const fs::path& p : newestFirst(d, versioned)) add(p);
                add(d / ("python3" + exe));
#ifdef _WIN32
                add(d / "python.exe");
#endif
            }
        };
        const auto uvEntries = [&]() {
            const std::string root = uvPythonDirectory();
            if (root.empty()) return;
            for (const fs::path& d : newestFirst(fsPath(root), [](const std::string& n) { return directoryMinor(n, "cpython-3."); }))
#ifdef _WIN32
                add(d / "python.exe");
#else
                add(d / "bin" / "python3");
#endif
        };
        const auto localBin = [&]() {
            if (!home.empty())
                for (const fs::path& p : newestFirst(fsPath(home) / ".local" / "bin", versioned)) add(p);
        };
#if defined(_WIN32)
        pathEntries();
        const std::string localAppData = host::environment("LOCALAPPDATA");
        std::string programFiles = host::environment("ProgramFiles");
        if (programFiles.empty()) programFiles = "C:/Program Files";
        for (const std::string& root : {localAppData.empty() ? std::string() : localAppData + "/Programs/Python", programFiles}) {
            if (root.empty()) continue;
            for (const fs::path& d : newestFirst(fsPath(root), [](const std::string& n) { return directoryMinor(n, "python3"); })) add(d / "python.exe");
        }
        uvEntries();
        localBin();
#elif defined(__APPLE__)
        add("/opt/homebrew/bin/python3");
        add("/usr/local/bin/python3");
        for (const fs::path& d : newestFirst("/Library/Frameworks/Python.framework/Versions", [](const std::string& n) { return directoryMinor(n, "3."); }))
            add(d / "bin" / "python3");
        uvEntries();
        localBin();
        pathEntries();
        if (commandLineToolsInstalled()) add("/usr/bin/python3");
#else
        pathEntries();
        uvEntries();
        localBin();
#endif
        return out;
    }

    std::string findUv() {
        // Absolute: uv runs in the environment's parent directory, where a
        // relative name would not be found (on POSIX the child changes
        // directory before it starts the program).
        const std::string configured = trim(host::environment("SIRIUS_UV"));
        if (!configured.empty()) return host::isFile(configured) ? absolutePath(forwardSlashes(configured)) : std::string();
        const std::string found = host::findExecutable("uv");
        if (!found.empty()) return absolutePath(found);
        // uv's own installer puts it in ~/.local/bin, which a session started
        // from the desktop does not always have on PATH.
        const std::string home = host::homeDirectory();
#ifdef _WIN32
        const std::string name = "uv.exe";
#else
        const std::string name = "uv";
#endif
        if (!home.empty())
            for (const std::string& dir : {home + "/.local/bin", home + "/.cargo/bin"})
                if (host::isFile(dir + "/" + name)) return dir + "/" + name;
        return std::string();
    }

    std::string uvVersion(const std::string& uv) {
        if (uv.empty()) return std::string();
        ChildProcess::Options o;
        o.program = uv;
        o.arguments = {"--version"};
        const RunOutcome r = runChild(o, 10000);
        for (const std::string& line : r.lines) {
            // "uv 0.12.18 (01cb90c1a 2026-09-22 x86_64-pc-windows-msvc)"
            const std::string t = trim(line);
            if (!startsWith(t, "uv ")) continue;
            const std::string rest = t.substr(3);
            return rest.substr(0, rest.find(' '));
        }
        return std::string();
    }

    // --- setting it up --------------------------------------------------------------------------

    json SetupPlan::toJson() const {
        json cmds = json::array();
        for (const std::vector<std::string>& c : commands) cmds.push_back(redacted(c));
        return json{{"env_dir", envDir},
                    {"python", environmentPython(envDir)},
                    {"base_python", basePython},
                    {"base_python_version", basePythonVersion},
                    {"installer", installer},
                    {"uv", uv.empty() ? json() : json(uv)},
                    {"index", redactUrl(index)},
                    {"mode", nothingToDo ? "none" : modeName(mode)},
                    {"nothing_to_do", nothingToDo},
                    {"packages", redacted(packages)},
                    {"approx_download_bytes", approxDownloadBytes},
                    {"approx_download_mb", static_cast<double>(approxDownloadBytes) / 1e6},
                    {"externally_managed_base", externallyManagedBase},
                    {"commands", cmds},
                    {"warnings", warnings}};
    }

    const char* toString(Failure f) noexcept {
        switch (f) {
            case Failure::None: return "none";
            case Failure::NoPython: return "no_python";
            case Failure::UnsupportedPython: return "unsupported_python";
            case Failure::NoEnsurepip: return "no_ensurepip";
            case Failure::Offline: return "offline";
            case Failure::Tls: return "tls";
            case Failure::NoWheel: return "no_wheel";
            case Failure::DiskFull: return "disk_full";
            case Failure::InUse: return "in_use";
            case Failure::Locked: return "locked";
            case Failure::Cancelled: return "cancelled";
            case Failure::Failed: return "failed";
        }
        return "failed";
    }

    json SetupResult::toJson() const {
        json logLines = json::array();
        for (const std::string& line : logTail) logLines.push_back(redactUrl(line));
        return json{{"ok", ok},
                    {"failure", toString(failure)},
                    {"message", redactUrl(message)},
                    {"hint", hint},
                    {"log_tail", logLines},
                    {"env_dir", plan.envDir},
                    {"python", environmentPython(plan.envDir)},
                    {"base_python", marker && !marker->baseExecutable.empty() ? marker->baseExecutable : plan.basePython},
                    {"python_version", marker ? marker->pythonVersion : plan.basePythonVersion},
                    {"installer", marker ? marker->installer : plan.installer},
                    {"mode", plan.nothingToDo ? "none" : modeName(plan.mode)},
                    {"packages", marker ? json(marker->packages) : json::object()},
                    {"extras", marker ? marker->extras : false},
                    {"seconds", seconds},
                    {"plan", plan.toJson()},
                    {"marker", shownMarker(marker)}};
    }

    SetupPlan planSetup(const SetupOptions& options, const std::string& scriptDir) { return planInternal(options, scriptDir, {}).plan; }

    SetupResult setup(const SetupOptions& options, const std::string& scriptDir,
                      const std::function<void(const std::string& line)>& onLine,
                      const std::function<void(double fraction, const std::string& message)>& onProgress,
                      const std::function<bool()>& cancelled) {
        const auto begin = std::chrono::steady_clock::now();
        // As the installers, which run elsewhere, can read it (absolutePath).
        const std::string workerDir = absolutePath(scriptDir);
        SetupResult result;
        std::deque<std::string> logTail;
        const auto note = [&](const std::string& line) {
            const std::string clean = redactUrl(line);
            logTail.push_back(clean);
            while (logTail.size() > 25) logTail.pop_front();
            if (onLine) onLine(clean);
        };
        const auto progress = [&](double fraction, const std::string& message) {
            if (onProgress) onProgress(fraction, message);
        };
        const auto isCancelled = [&] { return cancelled && cancelled(); };
        const auto finish = [&](Failure failure, std::string message, std::string hint) -> SetupResult {
            result.ok = failure == Failure::None;
            result.failure = failure;
            result.message = std::move(message);
            result.hint = std::move(hint);
            result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            if (!result.ok) note("Python environment: not set up: " + result.message);
            result.logTail.assign(logTail.begin(), logTail.end());
            return std::move(result);
        };

        std::string envDir, oldDir;
        bool createdHere = false, renamedOld = false;
        // Puts back what was there: the new folder goes, the previous
        // environment returns. An update leaves the environment as it is.
        // When the new folder cannot be deleted (a virus scanner holds its
        // files), the previous one stays at .old, and the next setup puts it
        // back before anything else.
        const auto rollback = [&] {
            if (!createdHere) return;
            removeTreeRetry(envDir);
            if (renamedOld && !exists(envDir)) host::renamePath(oldDir, envDir);
        };
        // Declared outside the try block, so that a rollback after an
        // exception still runs under it.
        SetupLock lock;
        try {
            progress(0.02, "Checking the Python\xE2\x80\xA6");
            envDir = environmentDirectory();
            if (envDir.empty())
                return finish(Failure::Failed, "There is no directory for SIRIUS's Python environment (the user's data directory is unknown).",
                              "Set SIRIUS_PYTHON_ENV to the directory it should use.");
            oldDir = envDir + ".old";
            host::makePath(parentOf(envDir));

            // 1. The lock, and what an earlier crash left behind.
            int holder = 0;
            std::string lockError;
            if (!lock.acquire(envDir, holder, lockError)) {
                if (!lockError.empty())
                    return finish(Failure::Failed, "Cannot create " + envDir + ".lock: " + lockError + ".", "Check that the directory is writable.");
                return finish(Failure::Locked, "Another setup is running" + (holder > 0 ? " (pid " + std::to_string(holder) + ")" : std::string()) + ".",
                              "Another SIRIUS window or sirius-cli is setting it up; wait for it to finish, then try again.");
            }
            if (exists(oldDir)) {
                if (const auto refusal = refuseDirectory(oldDir, "remove"))
                    return finish(Failure::Failed, *refusal, "Remove or move that directory yourself, then try again.");
                // A finished environment at .old and an unfinished one in its
                // place is a recreate whose rollback could not delete the new
                // folder: the working one comes back, it is not deleted.
                if (readMarker(oldDir) && !readMarker(envDir)) {
                    if (const auto refusal = exists(envDir) ? refuseDirectory(envDir, "replace") : std::nullopt)
                        return finish(Failure::Failed, *refusal, "Remove or move that directory yourself, then try again.");
                    if (!removeTreeRetry(envDir) || !host::renamePath(oldDir, envDir))
                        return finish(Failure::InUse, kInUseMessage, "Stop what runs from " + envDir + ", then try again.");
                    note("Python environment: put back the previous environment from " + oldDir);
                } else if (!removeTreeRetry(oldDir)) {
                    return finish(Failure::InUse, kInUseMessage, "Stop what runs from " + oldDir + ", then try again.");
                }
            }

            // 2. The plan. Probing interpreters and running the environment's
            // own take a while each; a cancel meanwhile ends here.
            Planned p = planInternal(options, workerDir, cancelled);
            result.plan = p.plan;
            const SetupPlan& plan = result.plan;
            if (isCancelled()) return finish(Failure::Cancelled, "Cancelled; nothing was changed.", std::string());
            if (p.failure != Failure::None) return finish(p.failure, p.message, p.hint);
            if (plan.nothingToDo) {
                result.marker = p.marker;
                note("Python environment: ready, nothing to do: " + envDir);
                progress(1.0, "Ready");
                return finish(Failure::None, "SIRIUS's Python environment is ready; nothing needed to be done.", std::string());
            }
            const bool uv = !plan.uv.empty();
            const bool creates = plan.mode != Mode::Update;
            if (creates)
                note("Python environment: " + std::string(plan.mode == Mode::Recreate ? "recreating " : "creating ") + envDir + " from " + plan.basePython +
                     " (Python " + plan.basePythonVersion + ") with " + plan.installer);
            else
                note("Python environment: updating " + envDir + " (Python " + plan.basePythonVersion + ") with " + plan.installer);

            // Runs one installer step: killTree, so that a cancel also ends
            // ensurepip's pip and uv's workers; both streams are logged,
            // classified and, while installing, turned into progress.
            std::vector<std::string> stepLines;
            InstallProgress installProgress(uv);
            bool feedProgress = false;
            const auto runStep = [&](const std::vector<std::string>& command) {
                ChildProcess::Options o;
                o.program = command.front();
                o.arguments.assign(command.begin() + 1, command.end());
                o.workingDirectory = parentOf(envDir);
                o.killTree = true;
                o.mergeErrorLines = true;
                o.environment = {{"PYTHONIOENCODING", "utf-8"}, {"PYTHONUNBUFFERED", "1"}, {"UV_PYTHON_DOWNLOADS", "never"}};
                for (auto& kv : installerIndexEnvironment(options, uv)) o.environment.push_back(std::move(kv));
                o.unsetEnvironment = installerUnset();
                o.unsetEnvironment.emplace_back("PYTHONHOME");
                note("$ " + commandLine(command));
                stepLines.clear();
                lock.touch();
                return runChild(
                    o, kInstallerTimeoutMs,
                    [&](const std::string& line) {
                        lock.touch();
                        stepLines.push_back(line);
                        note("python-env: " + line);
                        if (!feedProgress) return;
                        if (const auto f = installProgress.feed(line)) progress(*f, redactUrl(trim(line)));
                    },
                    cancelled);
            };
            // TLS interception gets one retry where the installer can use the
            // system's certificate store instead: uv with --native-tls; pip
            // 24.2 and later use that store already, and its truststore
            // feature before that needs Python 3.10. A retry that fails for a
            // reason of its own (an installer that no longer knows the
            // option) leaves the certificate as what the user hears about.
            const auto runWithTlsRetry = [&](std::vector<std::string> command, bool creating) {
                const RunOutcome first = runStep(command);
                if (first.ok() || first.cancelled || !first.started || first.timedOut) return first;
                if (classifyInstallerOutput(stepLines, first.exitCode, nullptr) != Failure::Tls) return first;
                if (uv) {
                    command.push_back("--native-tls");
                } else {
                    if (creating) return first;   // the standard library's venv downloads nothing
                    const RunOutcome v =
                        runChild(environmentPythonOptions(p.envPython, {"-m", "pip", "--version"}), kProbeTimeoutMs, {}, cancelled);
                    const std::string versionLine = v.lines.empty() ? std::string() : v.lines.front();
                    const auto [pipMajor, pipMinor] = pipVersion(versionLine);
                    const bool from222 = pipMajor > 22 || (pipMajor == 22 && pipMinor >= 2);
                    const bool before242 = pipMajor < 24 || (pipMajor == 24 && pipMinor < 2);
                    // The environment's Python as pip names it; else the base's,
                    // or for an update (which has no base) the marker's.
                    int pythonMinor = pipPythonMinor(versionLine);
                    if (pythonMinor == 0 && p.base) pythonMinor = p.base->minor;
                    if (pythonMinor == 0 && p.marker) pythonMinor = minorOf(p.marker->pythonVersion);
                    if (!from222 || !before242 || pythonMinor < 10) return first;
                    command.push_back("--use-feature=truststore");
                }
                // `uv venv --seed` makes the environment before it fetches pip
                // into it, and refuses to make one where one exists. The folder
                // is this setup's own (a recreate's previous one is at .old).
                if (creating && !removeTreeRetry(envDir)) return first;
                note("Python environment: the server's certificate was not trusted; trying again with the system's certificate store");
                const std::vector<std::string> firstLines = stepLines;
                const RunOutcome second = runStep(command);
                if (!second.ok() && !second.cancelled && second.started && !second.timedOut &&
                    classifyInstallerOutput(stepLines, second.exitCode, nullptr) == Failure::Failed) {
                    stepLines = firstLines;
                    return first;
                }
                return second;
            };
            const auto cancelledResult = [&] {
                rollback();
                return finish(Failure::Cancelled, creates ? "Cancelled; nothing was changed." : "Cancelled; the environment keeps its previous packages.",
                              std::string());
            };
            // The failure of a step, as the user reads it.
            const auto stepFailure = [&](const RunOutcome& r, const std::string& program, const std::string& what) -> SetupResult {
                if (r.cancelled) return cancelledResult();
                if (!r.started) {
                    rollback();
                    return finish(Failure::Failed, "Cannot start " + program + ": " + r.error + ".", std::string());
                }
                if (r.timedOut) {
                    rollback();
                    return finish(Failure::Failed, what + " did not finish within " + std::to_string(kInstallerTimeoutMs / 60000) + " minutes.",
                                  "Check the network, then try again.");
                }
                std::string hint;
                const Failure f = classifyInstallerOutput(stepLines, r.exitCode, &hint);
                rollback();
                const std::string server = indexHost(plan.index);
                switch (f) {
                    case Failure::Offline: return finish(f, "Could not reach " + server + ".", hint);
                    case Failure::Tls: return finish(f, "The secure connection to " + server + " could not be verified.", hint);
                    case Failure::NoWheel: {
                        std::vector<std::string> others;
                        for (const std::string& c : pythonCandidates())
                            if (lower(c) != lower(plan.basePython) && others.size() < 3) others.push_back(c);
                        const std::string version = plan.basePythonVersion.substr(0, plan.basePythonVersion.rfind('.'));
                        if (!others.empty()) hint = "Set up from an older Python (--base-python, or the Python list in the dialog): " + join(others, ", ") + ".";
                        return finish(f, "No ready-made package for Python " + version + ".", hint);
                    }
                    case Failure::DiskFull: return finish(f, "The disk is full at " + envDir + ".", hint);
                    case Failure::InUse: return finish(f, kInUseMessage, hint);
                    case Failure::NoEnsurepip:
                        return finish(f, "Python " + plan.basePythonVersion + " (" + plan.basePython + ") lacks venv support (ensurepip).",
                                      noEnsurepipHint(p.base ? p.base->minor : 0));
                    default: return finish(Failure::Failed, what + " failed (exit code " + std::to_string(r.exitCode) + ").", hint);
                }
            };

            // 3. Create (a Recreate first moves the working environment aside).
            if (creates) {
                if (plan.mode == Mode::Recreate && exists(envDir)) {
                    if (!host::renamePath(envDir, oldDir)) return finish(Failure::InUse, kInUseMessage, "Stop the worker that runs from it, then try again.");
                    renamedOld = true;
                }
                createdHere = true;
                progress(0.10, "Creating the environment (installing pip)\xE2\x80\xA6");
                const RunOutcome r = runWithTlsRetry(p.createCommand, true);
                if (!r.ok()) return stepFailure(r, p.createCommand.front(), "Creating the environment");
            }
            if (isCancelled()) return cancelledResult();

            // 4. Install.
            progress(0.20, "Downloading " + join(redacted(plan.packages), ", ") + "\xE2\x80\xA6");
            feedProgress = true;
            const RunOutcome installed = runWithTlsRetry(p.installCommand, false);
            feedProgress = false;
            if (!installed.ok()) return stepFailure(installed, p.installCommand.front(), "Installing the packages");
            if (isCancelled()) return cancelledResult();

            // 5. Verify, as the worker will start: from its own directory.
            progress(0.97, "Checking the environment\xE2\x80\xA6");
            note("$ " + commandLine({p.envPython, "-m", "sirius_worker", "--check"}));
            std::string checkProblem;
            const json check = workerCheck(p.envPython, workerDir, &checkProblem, cancelled);
            if (isCancelled()) return cancelledResult();
            if (!check.is_object()) {
                rollback();
                return finish(Failure::Failed, "The new environment does not run the worker: " + checkProblem + ".", std::string());
            }
            const std::vector<std::string> missing = stringList(check["missing"]);
            if (!missing.empty()) {
                rollback();
                return finish(Failure::Failed, "The environment still lacks " + join(missing, ", ") + " after the installation.", std::string());
            }

            // 6. Finish: the README, then the marker, written last.
            Marker marker;
            marker.createdBy = options.createdBy.empty() ? std::string("sirius") : options.createdBy;
            marker.created = !creates && p.marker && !p.marker->created.empty() ? p.marker->created : utcNow();
            const auto checkText = [&check](const char* key) {
                return check.contains(key) && check[key].is_string() ? forwardSlashes(check[key].get<std::string>()) : std::string();
            };
            if (creates && p.base) marker.baseExecutable = !p.base->baseExecutable.empty() ? p.base->baseExecutable : p.base->executable;
            else if (p.marker) marker.baseExecutable = p.marker->baseExecutable;
            if (marker.baseExecutable.empty()) marker.baseExecutable = checkText("base_executable");
            marker.pythonVersion = checkText("version");
            if (check.contains("packages") && check["packages"].is_object())
                for (const auto& [name, version] : check["packages"].items())
                    if (version.is_string()) marker.packages[name] = version.get<std::string>();
            marker.installer = uv || marker.packages.count("pip") == 0 ? plan.installer : "pip " + marker.packages["pip"];
            marker.index = plan.index;
            marker.fingerprint = requirementsFingerprint(workerDir, p.extras, p.extraPackages);
            marker.extras = p.extras;
            marker.extraPackages = p.extraPackages;
            host::writeFileAtomic(envDir + "/README.txt",
                                  "Created by SIRIUS for its Python worker. Safe to delete. Recreate it in Preferences \xE2\x96\xB8 Compute "
                                  "or with `sirius-cli worker setup`.\n");
            if (!host::writeFileAtomic(envDir + "/" + kMarkerFile, marker.toJson().dump(2) + "\n")) {
                rollback();
                return finish(Failure::Failed, "Cannot write " + envDir + "/" + kMarkerFile + ".", "Check that the disk has space and the directory is writable.");
            }
            if (renamedOld) removeTreeRetry(oldDir);   // what stays is removed by the next setup
            result.marker = marker;
            result.plan.basePythonVersion = marker.pythonVersion;
            std::vector<std::string> versions;
            for (const std::string& requirement : plan.packages) {
                const std::string name = distributionName(requirement);
                const std::optional<std::string> version = packageVersion(marker.packages, name);
                versions.push_back(version ? name + " " + *version : name);
            }
            progress(1.0, "Ready");
            note("Python environment: ready: " + join(versions, ", ") + " (" +
                 secondsText(std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count()) + ")");
            return finish(Failure::None, "Ready: " + join(versions, ", ") + " (Python " + marker.pythonVersion + ").", std::string());
        } catch (const std::exception& e) {
            rollback();
            return finish(Failure::Failed, e.what(), std::string());
        } catch (...) {
            rollback();
            return finish(Failure::Failed, "unknown error", std::string());
        }
    }

    SetupResult remove(const std::string& envDir) {
        const auto begin = std::chrono::steady_clock::now();
        SetupResult result;
        result.plan.envDir = envDir;
        result.plan.nothingToDo = true;
        const auto finish = [&](Failure failure, std::string message, std::string hint) {
            result.ok = failure == Failure::None;
            result.failure = failure;
            result.message = std::move(message);
            result.hint = std::move(hint);
            result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            return result;
        };
        if (envDir.empty()) return finish(Failure::Failed, "No environment directory was named.", std::string());
        const std::string oldDir = envDir + ".old";
        if (!host::isDirectory(parentOf(envDir))) return finish(Failure::None, "There is no environment at " + envDir + ".", std::string());
        SetupLock lock;
        int holder = 0;
        std::string lockError;
        if (!lock.acquire(envDir, holder, lockError)) {
            if (!lockError.empty())
                return finish(Failure::Failed, "Cannot create " + envDir + ".lock: " + lockError + ".", "Check that the directory is writable.");
            return finish(Failure::Locked, "Another setup is running" + (holder > 0 ? " (pid " + std::to_string(holder) + ")" : std::string()) + ".",
                          "Wait for it to finish, then try again.");
        }
        if (exists(oldDir)) {
            if (const auto refusal = refuseDirectory(oldDir, "remove")) return finish(Failure::Failed, *refusal, "Remove or move that directory yourself.");
            if (!removeTreeRetry(oldDir)) return finish(Failure::InUse, kInUseMessage, "Stop what runs from " + oldDir + ", then try again.");
        }
        if (!exists(envDir)) return finish(Failure::None, "There is no environment at " + envDir + ".", std::string());
        if (const auto refusal = refuseDirectory(envDir, "remove")) return finish(Failure::Failed, *refusal, "Remove or move that directory yourself.");
        // Moved aside first: a folder that cannot be renamed is in use (a
        // worker runs from it, on Windows), and nothing of it is deleted.
        if (!host::renamePath(envDir, oldDir)) return finish(Failure::InUse, kInUseMessage, "Stop the worker that runs from it, then try again.");
        std::string error;
        if (!removeTreeRetry(oldDir, &error))
            return finish(Failure::None, "Removed " + envDir + "; some files of " + oldDir + " could not be deleted yet (" + error + ").",
                          "They are removed by the next setup.");
        return finish(Failure::None, "Removed " + envDir + ".", std::string());
    }

    Failure classifyInstallerOutput(const std::vector<std::string>& lines, int exitCode, std::string* hint) {
        const auto setHint = [hint](const char* text) {
            if (hint) *hint = text;
        };
        if (exitCode == 0) {
            setHint("");
            return Failure::None;
        }
        // One lower-cased text with every run of whitespace a single space,
        // so that a message the installer wrapped over two lines still reads
        // as written.
        std::string all;
        bool space = true;
        for (const std::string& line : lines) {
            for (const char c : line + "\n") {
                if (std::isspace(static_cast<unsigned char>(c))) {
                    if (!space) all.push_back(' ');
                    space = true;
                } else {
                    all.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
                    space = false;
                }
            }
        }
        const auto any = [&all](std::initializer_list<const char*> needles) {
            for (const char* n : needles)
                if (all.find(lower(n)) != std::string::npos) return true;
            return false;
        };
        // Most specific first: a TLS failure also reads "Failed to fetch",
        // and pip ends an unreachable index with "No matching distribution".
        if (any({"ensurepip is not available", "no module named ensurepip", "no module named 'ensurepip'", "no module named venv",
                 "no module named 'venv'"})) {
            setHint("Install the Python's venv support (for example `sudo apt install python3-venv`), or install uv, then try again.");
            return Failure::NoEnsurepip;
        }
        if (any({"no space left", "errno 28", "there is not enough space", "os error 112"})) {
            setHint("Free some space on that disk, then try again.");
            return Failure::DiskFull;
        }
        if (any({"certificate verify failed", "invalid peer certificate", "unknownissuer", "self-signed certificate", "self signed certificate"})) {
            setHint("Your network intercepts TLS; set SSL_CERT_FILE / REQUESTS_CA_BUNDLE to its certificate, then try again.");
            return Failure::Tls;
        }
        // An index that answered with an error status was reached: uv also
        // says "Failed to fetch" for a 401 from a private index, which is a
        // password or a package the index lacks, not the network.
        if (any({"http status client error", "http error 401", "http error 403", "http error 404", "401 client error", "403 client error",
                 "404 client error", "401 unauthorized", "403 forbidden"})) {
            setHint("The package index refused the request: check its address and credentials (--index-url, PIP_INDEX_URL / UV_INDEX_URL) "
                    "and that it has the packages, then try again.");
            return Failure::Failed;
        }
        if (any({"failed to establish a new connection", "getaddrinfo", "name or service not known", "tcp connect error", "failed to fetch",
                 "dns error", "temporary failure in name resolution", "no such host is known", "network is unreachable",
                 "could not resolve host"})) {
            setHint("Check the network or proxy (HTTPS_PROXY), or use a mirror (PIP_INDEX_URL / UV_INDEX_URL, --index-url), then try again.");
            return Failure::Offline;
        }
        if (any({"no matching distribution", "no wheels with a matching", "has no usable wheels", "could not find a version that satisfies"})) {
            setHint("Set up from an older Python (--base-python, or the Python list in the dialog).");
            return Failure::NoWheel;
        }
        if (any({"access is denied", "winerror 5]", "winerror 32", "being used by another process"})) {
            setHint("Close the other SIRIUS windows and sirius-cli sessions (their worker runs from the environment), then try again.");
            return Failure::InUse;
        }
        setHint("The last lines of the installer's output say more.");
        return Failure::Failed;
    }

    std::optional<double> progressFromLine(const std::string& line, bool uv) {
        const std::string t = trim(line);
        if (uv) {
            if (startsWith(t, "Resolved ")) return 0.25;
            if (startsWith(t, "Downloading ")) return 0.30;
            if (startsWith(t, "Downloaded ")) return 0.55;
            if (startsWith(t, "Prepared ")) return 0.80;
            if (startsWith(t, "Installed ") || startsWith(t, "Audited ")) return 0.95;
            return std::nullopt;
        }
        if (startsWith(t, "Collecting ") || startsWith(t, "Requirement already satisfied")) return 0.20;
        if (startsWith(t, "Downloading ") || startsWith(t, "Using cached ")) return 0.40;
        if (startsWith(t, "Installing collected packages")) return 0.80;
        if (startsWith(t, "Successfully installed")) return 0.95;
        return std::nullopt;
    }

} // namespace sirius::app::pyenv
