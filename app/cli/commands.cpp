#include "cli/commands.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <sirius/device.hpp>

#include "cli/args.hpp"
#include "cli/stdio.hpp"
#include "core/agent_protocol.hpp"
#include "core/app_paths.hpp"
#include "core/array_source.hpp"
#include "core/build_info.hpp"
#include "core/cancel.hpp"
#include "core/export.hpp"
#include "core/headless.hpp"
#include "core/help_pages.hpp"
#include "core/host.hpp"
#include "core/image_encode.hpp"
#include "core/local_worker.hpp"
#include "core/ops/builtin.hpp"
#include "core/python_env.hpp"
#include "core/rpc.hpp"
#include "core/worker_error.hpp"

namespace sirius::cli {

    using json = nlohmann::json;
    namespace fs = std::filesystem;
    namespace app = sirius::app;
    namespace agent = sirius::app::agent;
    namespace pyenv = sirius::app::pyenv;
    using Clock = std::chrono::steady_clock;

    const std::vector<std::string>& mcpVersions() {
        static const std::vector<std::string> versions = {"2026-07-28", "2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"};
        return versions;
    }

    namespace {

        // A command that did not do what it was asked: the failure envelope.
        struct CliError {
            std::string code, message, hint;
            json data;
            CliError(std::string c, std::string m, std::string h = {}, json d = nullptr)
                : code(std::move(c)), message(std::move(m)), hint(std::move(h)), data(std::move(d)) {}
        };

        // --- what the signal threads reach -----------------------------------
        //
        // Ctrl+C and the termination signals arrive on a thread of their own
        // (stdio.cpp); they set these flags, which every wait of a command
        // polls, and cancel what the workspace is doing.

        std::atomic<bool> interrupted{false};
        std::atomic<bool> terminated{false};
        std::atomic<bool> documentClaimed{false};

        std::mutex& activeMutex() {
            static std::mutex m;
            return m;
        }
        app::HeadlessWorkbench* activeWorkspace = nullptr;   // guarded by activeMutex()

        void cancelActiveWorkspace() {
            const std::lock_guard<std::mutex> g(activeMutex());
            if (activeWorkspace) activeWorkspace->cancelActive();
        }

        // Only one document ever goes to stdout: the command's, or the
        // timeout watchdog's when the command does not end in time.
        bool claimDocument() { return !documentClaimed.exchange(true); }

        int exitFor(const std::string& code) { return agent::exitCodeFor(code); }

        std::string dump(const json& j, bool pretty) { return j.dump(pretty ? 2 : -1, ' ', false, json::error_handler_t::replace); }

        // --- small helpers ------------------------------------------------------

        std::string lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        }

        std::string trim(const std::string& s) {
            const std::size_t b = s.find_first_not_of(" \t\r\n");
            if (b == std::string::npos) return {};
            return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
        }

        bool endsWith(const std::string& s, const std::string& suffix) {
            return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
        }

        std::vector<std::string> split(const std::string& s, char sep) {
            std::vector<std::string> out;
            std::size_t start = 0;
            for (;;) {
                const std::size_t pos = s.find(sep, start);
                out.push_back(trim(s.substr(start, pos == std::string::npos ? std::string::npos : pos - start)));
                if (pos == std::string::npos) return out;
                start = pos + 1;
            }
        }

        std::string join(const std::vector<std::string>& parts, const std::string& sep) {
            std::string out;
            for (const std::string& p : parts) out += (out.empty() ? "" : sep) + p;
            return out;
        }

        std::string slashes(std::string s) {
            std::replace(s.begin(), s.end(), '\\', '/');
            return s;
        }

        // Absolute, with forward slashes: how every path is reported.
        std::string displayPath(const std::string& p) {
            if (p.empty()) return p;
            std::error_code ec;
            const fs::path abs = fs::absolute(fs::u8path(p), ec);
            if (ec) return slashes(p);
            return abs.lexically_normal().generic_u8string();
        }

        json pathOrNull(const std::string& p) { return p.empty() ? json(nullptr) : json(displayPath(p)); }

        void appendUtf8(std::string& out, unsigned cp) {
            if (cp < 0x80) {
                out += static_cast<char>(cp);
            } else if (cp < 0x800) {
                out += static_cast<char>(0xC0 | (cp >> 6));
                out += static_cast<char>(0x80 | (cp & 0x3F));
            } else if (cp < 0x10000) {
                out += static_cast<char>(0xE0 | (cp >> 12));
                out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                out += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
                out += static_cast<char>(0xF0 | (cp >> 18));
                out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                out += static_cast<char>(0x80 | (cp & 0x3F));
            }
        }

        // JSON handed over as a file or on stdin, as UTF-8. Windows PowerShell
        // writes UTF-16 with a byte order mark (`> steps.json`), and many
        // editors a UTF-8 one.
        std::string asUtf8(const std::string& bytes) {
            const auto byte = [&](std::size_t i) { return static_cast<unsigned>(static_cast<unsigned char>(bytes[i])); };
            const std::size_t n = bytes.size();
            if (n >= 3 && byte(0) == 0xEF && byte(1) == 0xBB && byte(2) == 0xBF) return bytes.substr(3);
            const bool le = n >= 2 && byte(0) == 0xFF && byte(1) == 0xFE;
            const bool be = n >= 2 && byte(0) == 0xFE && byte(1) == 0xFF;
            if (!le && !be) return bytes;
            const auto unit = [&](std::size_t i) { return le ? byte(i) | (byte(i + 1) << 8) : (byte(i) << 8) | byte(i + 1); };
            std::string out;
            for (std::size_t i = 2; i + 1 < n; i += 2) {
                unsigned cp = unit(i);
                if (cp >= 0xD800 && cp <= 0xDBFF && i + 3 < n && unit(i + 2) >= 0xDC00 && unit(i + 2) <= 0xDFFF) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (unit(i + 2) - 0xDC00);
                    i += 2;
                } else if (cp >= 0xD800 && cp <= 0xDFFF) {
                    cp = 0xFFFD;
                }
                appendUtf8(out, cp);
            }
            return out;
        }

        bool readWholeFile(const std::string& path, std::string& out) {
            std::ifstream in(fs::u8path(path), std::ios::binary);
            if (!in) return false;
            out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
            return !in.bad();
        }

        long long toInteger(const std::string& option, const std::string& value) {
            const std::string v = trim(value);
            char* end = nullptr;
            const long long n = std::strtoll(v.c_str(), &end, 10);
            if (v.empty() || end != v.c_str() + v.size()) throw UsageError("--" + option + " expects a whole number, not '" + value + "'");
            return n;
        }

        double toNumber(const std::string& option, const std::string& value) {
            const std::string v = trim(value);
            char* end = nullptr;
            const double d = std::strtod(v.c_str(), &end);
            if (v.empty() || end != v.c_str() + v.size() || !std::isfinite(d))
                throw UsageError("--" + option + " expects a number, not '" + value + "'");
            return d;
        }

        json integerList(const std::string& option, const std::string& value, std::size_t count = 0) {
            json out = json::array();
            for (const std::string& part : split(value, ',')) out.push_back(toInteger(option, part));
            if (count && out.size() != count)
                throw UsageError("--" + option + " expects " + std::to_string(count) + " comma-separated numbers, not '" + value + "'");
            return out;
        }

        json numberList(const std::string& option, const std::string& value, std::size_t count = 0) {
            json out = json::array();
            for (const std::string& part : split(value, ',')) out.push_back(toNumber(option, part));
            if (count && out.size() != count)
                throw UsageError("--" + option + " expects " + std::to_string(count) + " comma-separated numbers, not '" + value + "'");
            return out;
        }

        // A step: a number (1 = Load) or a name.
        json stepValue(const std::string& s) {
            const std::string t = trim(s);
            if (!t.empty() && t.size() < 9 && std::all_of(t.begin(), t.end(), [](unsigned char c) { return std::isdigit(c) != 0; }))
                return std::atoi(t.c_str());
            return t;
        }

        // "a:b" -> [a, b], "a:" -> [a, -1] (to the end), "a" -> [a, a + 1]: the
        // half-open ranges the export takes.
        json rangeValue(const std::string& option, const std::string& value) {
            const std::size_t colon = value.find(':');
            if (colon == std::string::npos) {
                const long long a = toInteger(option, value);
                return json::array({a, a + 1});
            }
            const long long a = toInteger(option, value.substr(0, colon));
            const std::string rest = trim(value.substr(colon + 1));
            return json::array({a, rest.empty() ? -1LL : toInteger(option, rest)});
        }

        // NaN and infinity have no JSON form; they are written as null, and
        // the envelope says so.
        bool replaceNonFinite(json& j) {
            if (j.is_number_float()) {
                if (std::isfinite(j.get<double>())) return false;
                j = nullptr;
                return true;
            }
            bool any = false;
            if (j.is_array() || j.is_object())
                for (json& v : j) any = replaceNonFinite(v) || any;
            return any;
        }

        std::string slug(const std::string& s) {
            std::string out;
            for (unsigned char c : s) {
                if (std::isalnum(c)) out += static_cast<char>(std::tolower(c));
                else if (!out.empty() && out.back() != '-') out += '-';
            }
            while (!out.empty() && out.back() == '-') out.pop_back();
            return out.empty() ? std::string("image") : out;
        }

        // --- stderr: progress and log lines -----------------------------------

        class Reporter {
        public:
            enum class Mode { None,
                              Text,
                              Json };

            Reporter(Mode mode, bool quiet, bool terminal) : mode_(mode), quiet_(quiet), terminal_(terminal) {}

            // Any thread (the worker's log comes on its stderr reader). In
            // text a line is marked with its source ("worker: ...") unless it
            // is the workbench's own, already marked, or `prefix` is false.
            void log(const std::string& source, const std::string& line, bool prefix = true) {
                if (quiet_) return;
                const std::lock_guard<std::mutex> g(mutex_);
                if (mode_ == Mode::Json) {
                    writeError(dump({{"event", "log"}, {"source", source}, {"line", line}}, false) + "\n");
                    return;
                }
                clearStatus();
                const bool marked = !prefix || source == "workbench" || line.compare(0, source.size() + 2, source + ": ") == 0;
                writeError((marked ? line : source + ": " + line) + "\n");
            }

            void progress(double fraction, const std::string& message) {
                if (mode_ == Mode::None) return;
                const std::lock_guard<std::mutex> g(mutex_);
                const Clock::time_point now = Clock::now();
                const bool changed = message != lastMessage_;
                if (now - lastProgress_ < std::chrono::milliseconds(250) && !changed) return;
                if (mode_ == Mode::Json) {
                    writeError(dump({{"event", "progress"}, {"fraction", fraction}, {"message", message}}, false) + "\n");
                } else {
                    const int percent = static_cast<int>(std::lround(std::clamp(fraction, 0.0, 1.0) * 100.0));
                    std::string number = std::to_string(percent);
                    if (number.size() < 3) number.insert(0, 3 - number.size(), ' ');
                    std::string text = "[" + number + "%] " + message;
                    if (terminal_) {
                        // One status line, redrawn in place; kept short enough
                        // not to wrap (a wrapped line cannot be redrawn with \r).
                        if (text.size() > 76) {
                            std::size_t cut = 76;
                            while (cut > 0 && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80) --cut;
                            text = text.substr(0, cut);
                        }
                        std::string padded = "\r" + text;
                        if (text.size() < statusLength_) padded += std::string(statusLength_ - text.size(), ' ');
                        writeError(padded);
                        statusLength_ = text.size();
                    } else if (now - lastPrinted_ >= std::chrono::seconds(1) || fraction >= 1.0) {
                        // A log file: one line a second at most.
                        writeError(text + "\n");
                        lastPrinted_ = now;
                    }
                }
                lastProgress_ = now;
                lastMessage_ = message;
            }

            // A line for the person at the terminal (not in --quiet).
            void note(const std::string& text) {
                if (quiet_) return;
                const std::lock_guard<std::mutex> g(mutex_);
                clearStatus();
                writeError(text + "\n");
            }

            // The one line a failure always leaves on stderr.
            void summary(const std::string& code, const std::string& message) {
                const std::lock_guard<std::mutex> g(mutex_);
                clearStatus();
                writeError("sirius-cli: " + code + ": " + message + "\n");
            }

            void finish() {
                const std::lock_guard<std::mutex> g(mutex_);
                clearStatus();
            }

        private:
            void clearStatus() {
                if (statusLength_ == 0) return;
                writeError("\r" + std::string(statusLength_, ' ') + "\r");
                statusLength_ = 0;
            }

            std::mutex mutex_;
            Mode mode_;
            bool quiet_, terminal_;
            std::size_t statusLength_ = 0;
            std::string lastMessage_;
            Clock::time_point lastProgress_{}, lastPrinted_{};
        };

        Reporter::Mode reporterMode(const GlobalOptions& g) {
            if (g.progress == "json") return Reporter::Mode::Json;
            if (g.progress == "text") return Reporter::Mode::Text;
            if (g.progress == "none" || g.quiet) return Reporter::Mode::None;
            return stderrIsTerminal() ? Reporter::Mode::Text : Reporter::Mode::None;
        }

        // --- the envelope --------------------------------------------------------

        json successEnvelope(const std::string& command, json result, const std::vector<std::string>& warnings) {
            return {{"ok", true}, {"schema", kOutputSchema}, {"command", command}, {"result", std::move(result)}, {"warnings", warnings}};
        }

        json failureEnvelope(const std::string& command, const CliError& e, int exitCode, const std::vector<std::string>& warnings) {
            return {{"ok", false},
                    {"schema", kOutputSchema},
                    {"command", command},
                    {"error", {{"code", e.code}, {"message", e.message}, {"hint", e.hint}, {"data", e.data}}},
                    {"exit_code", exitCode},
                    {"warnings", warnings}};
        }

        // --- tool schemas, for `call <tool> --<param> value` -----------------

        void collectTypes(const json& schema, std::set<std::string>& types) {
            if (!schema.is_object()) return;
            if (schema.contains("type")) {
                const json& t = schema["type"];
                if (t.is_string()) types.insert(t.get<std::string>());
                else if (t.is_array())
                    for (const json& x : t)
                        if (x.is_string()) types.insert(x.get<std::string>());
            } else if (schema.contains("enum") && schema["enum"].is_array()) {
                for (const json& v : schema["enum"]) {
                    if (v.is_string()) types.insert("string");
                    else if (v.is_boolean()) types.insert("boolean");
                    else if (v.is_number_integer()) types.insert("integer");
                    else if (v.is_number()) types.insert("number");
                }
            }
            for (const char* key : {"oneOf", "anyOf"})
                if (schema.contains(key) && schema[key].is_array())
                    for (const json& branch : schema[key]) collectTypes(branch, types);
        }

        const json* arrayItems(const json& schema) {
            if (!schema.is_object()) return nullptr;
            if (schema.contains("items")) return &schema["items"];
            for (const char* key : {"oneOf", "anyOf"})
                if (schema.contains(key) && schema[key].is_array())
                    for (const json& branch : schema[key])
                        if (const json* items = arrayItems(branch)) return items;
            return nullptr;
        }

        bool matchesType(const json& v, const std::set<std::string>& types) {
            for (const std::string& t : types) {
                if (t == "string" && v.is_string()) return true;
                if (t == "boolean" && v.is_boolean()) return true;
                if (t == "number" && v.is_number()) return true;
                if (t == "integer" && (v.is_number_integer() || (v.is_number_float() && std::floor(v.get<double>()) == v.get<double>()))) return true;
                if (t == "array" && v.is_array()) return true;
                if (t == "object" && v.is_object()) return true;
                if (t == "null" && v.is_null()) return true;
            }
            return false;
        }

        // --- signals and the timeout -------------------------------------------

        void installOneShotHandlers() {
            installInterruptHandler([] {
                interrupted.store(true);
                cancelActiveWorkspace();
            });
            installTerminationHandler([] {
                terminated.store(true);
                cancelActiveWorkspace();
            });
        }

        // The session and MCP servers, as the stdin reader and the signal
        // threads reach them. Created once and never destroyed: the detached
        // reader may still call in when the main thread is done, so after
        // close() every entry point does nothing.
        struct ServerGate {
            std::mutex mutex;
            agent::Server* server = nullptr;
            bool closed = false, ended = false;

            void receive(const std::string& line) {
                const std::lock_guard<std::mutex> g(mutex);
                if (closed) return;
                try {
                    server->receive(line);
                } catch (const std::exception& e) {
                    writeError(std::string("sirius-cli: internal: ") + e.what() + "\n");
                }
            }
            void endOfInput() {
                const std::lock_guard<std::mutex> g(mutex);
                if (closed) return;
                try {
                    server->endOfInput();
                } catch (const std::exception& e) {
                    writeError(std::string("sirius-cli: internal: ") + e.what() + "\n");
                }
            }
            // A termination signal, or a client that closed our stdout: end
            // the input, then cancel whatever runs.
            void terminate() {
                const std::lock_guard<std::mutex> g(mutex);
                if (closed || ended) return;
                ended = true;
                try {
                    server->endOfInput();
                    server->terminate();
                } catch (const std::exception& e) {
                    writeError(std::string("sirius-cli: internal: ") + e.what() + "\n");
                }
            }
            void close() {
                {
                    const std::lock_guard<std::mutex> g(mutex);
                    closed = true;
                }
                server->close();
            }
        };

        // --- one invocation ------------------------------------------------------

        class Runner {
        public:
            explicit Runner(const Args& args)
                : args_(args), reporter_(reporterMode(args.global), args.global.quiet, stderrIsTerminal()),
                  pretty_(args.global.pretty.value_or(stdoutIsTerminal())) {}

            int execute();

        private:
            std::optional<json> dispatch();

            // the workspace (HeadlessWorkbench) and its scratch directory
            app::HeadlessWorkbench& workspace();
            void closeWorkspace();
            void applyState();
            std::string scriptDir() const;

            // tool calls
            bool cancelled(bool honourDeadline = true) const;
            bool deadlinePassed() const { return deadline_ && Clock::now() >= *deadline_; }
            agent::CallContext context(bool honourDeadline = true);
            CliError cancellation() const;
            CliError toCliError(const agent::ToolError& e) const;
            agent::ToolResult callRaw(const std::string& tool, const json& args, bool honourDeadline = true);
            json call(const std::string& tool, const json& args);
            json runPipeline(json runArgs);
            json readJsonOption(const std::string& option, const std::string& value);
            json openArgs(const std::vector<Option>& open) const;
            json renderArgs(const std::vector<Option>& options) const;
            json statsArgs(const std::vector<Option>& options) const;
            json exportArgs(const std::vector<Option>& options, const std::string& path) const;
            json renderTo(const std::string& path, json renderArguments, bool base64);
            void defaultToPipelineEnd(json& toolArguments);
            void checkRenderTargets() const;
            json imageInfo(const agent::ToolResult& r);

            // commands
            json cmdVersion();
            json cmdInfo();
            json cmdOps();
            std::optional<json> cmdHelp();
            json cmdValidate();
            json cmdRun();
            json cmdRender();
            json cmdDiagnostics();
            json cmdExportTraining();
            json cmdCall();
            json cmdTools();
            json cmdWorkerStatus();
            json cmdWorkerCheck();
            json cmdWorkerSetup();
            json cmdWorkerRemove();
            int serve(bool mcp);
            std::string toolHelp();

            CliError workerUnavailable(const app::WorkerStartError& e) const;
            CliError setupFailure(const pyenv::SetupResult& r) const;
            bool confirm(const std::string& question);
            void startWatchdog();

            const Args& args_;
            std::string command_;              // "version" for a bare --version
            Reporter reporter_;
            bool pretty_;
            bool rawOutput_ = false;           // help --markdown: the page, not JSON
            bool stdinUsed_ = false;
            std::vector<std::string> warnings_;
            std::optional<Clock::time_point> deadline_;
            std::string scratch_;
            bool removeScratch_ = false;
            std::unique_ptr<app::HeadlessWorkbench> hw_;
        };

        // --- the workspace ------------------------------------------------------

        app::HeadlessWorkbench& Runner::workspace() {
            if (hw_) return *hw_;
            const GlobalOptions& g = args_.global;
            if (!g.scratch.empty()) {
                const fs::path p = fs::u8path(g.scratch);
                std::error_code ec;
                const bool existed = fs::exists(p, ec);
                fs::create_directories(p, ec);
                if (!fs::is_directory(p, ec)) throw CliError("io_error", "cannot create the scratch directory " + displayPath(g.scratch));
                scratch_ = displayPath(g.scratch);
                // A directory someone named may hold files of theirs: it is
                // removed at exit only when this run created it.
                removeScratch_ = !existed && !g.keepScratch;
            } else {
                scratch_ = app::host::makeTempDirectory("sirius-cli-");
                if (scratch_.empty())
                    throw CliError("io_error", "cannot create a scratch directory in " + app::host::tempDirectory(), "pass --scratch <dir>");
                scratch_ = slashes(scratch_);
                removeScratch_ = !g.keepScratch;
            }
            if (removeScratch_) {
                setEmergencyCleanup([dir = scratch_] {
                    std::error_code ec;
                    fs::remove_all(fs::u8path(dir), ec);
                });
            }

            app::HeadlessOptions o;
            o.scratchDir = fs::u8path(scratch_);
            o.python = g.python;
            o.workerDir = g.workerDir;
            o.plugins = g.plugins == "on"    ? app::HeadlessOptions::Plugins::On
                        : g.plugins == "off" ? app::HeadlessOptions::Plugins::Off
                                             : app::HeadlessOptions::Plugins::Auto;
            o.backend = g.backend;
            o.cudaDevice = g.cudaDevice;
            o.hpcDevice = g.hpcDevice;
            if (!g.hpcHost.empty()) {
                app::RemoteConfig hpc;
                hpc.host = g.hpcHost;
                hpc.port = g.hpcPort;
                hpc.token = app::host::environment("SIRIUS_HPC_TOKEN");
                o.hpc = hpc;
            }
            o.hubToken = app::host::environment("HF_TOKEN");
            o.recordPath = g.record;
            o.allowWorkerSetup = args_.has("allow-worker-setup");
            o.readOnly = args_.has("read-only");
            // The servers' tools take paths an agent chose, which must be local
            // unless the user says otherwise; a one-shot command's paths are
            // the ones its user typed, network shares included.
            o.allowNetworkPaths = (command_ != "session" && command_ != "mcp") || args_.has("allow-network-paths");
            o.createdBy = std::string("sirius-cli ") + SIRIUS_VERSION;
            o.logSink = [this](const std::string& source, const std::string& line) { reporter_.log(source, line); };
            hw_ = std::make_unique<app::HeadlessWorkbench>(std::move(o));
            {
                const std::lock_guard<std::mutex> lock(activeMutex());
                activeWorkspace = hw_.get();
            }
            return *hw_;
        }

        void Runner::closeWorkspace() {
            {
                const std::lock_guard<std::mutex> lock(activeMutex());
                activeWorkspace = nullptr;
            }
            // Cancels and joins a run, stops the worker.
            hw_.reset();
            if (scratch_.empty()) return;
            if (removeScratch_) {
                std::error_code ec;
                fs::remove_all(fs::u8path(scratch_), ec);
                setEmergencyCleanup({});
            } else if (args_.global.keepScratch) {
                reporter_.note("scratch directory kept: " + scratch_);
            }
        }

        std::string Runner::scriptDir() const {
            // A --worker-dir is taken as given, so a wrong one is reported
            // rather than silently replaced by another copy of the worker.
            if (!args_.global.workerDir.empty()) return slashes(args_.global.workerDir);
            return slashes(app::workerScriptPath());
        }

        // --- tool calls ---------------------------------------------------------

        bool Runner::cancelled(bool honourDeadline) const {
            return interrupted.load() || terminated.load() || (honourDeadline && deadlinePassed());
        }

        agent::CallContext Runner::context(bool honourDeadline) {
            agent::CallContext ctx;
            ctx.cancelled = [this, honourDeadline] { return cancelled(honourDeadline); };
            ctx.progress = [this](double fraction, const std::string& message) { reporter_.progress(fraction, message); };
            return ctx;
        }

        CliError Runner::cancellation() const {
            if (deadlinePassed() && !interrupted.load() && !terminated.load()) {
                std::ostringstream s;
                s << "timed out after " << *args_.global.timeoutSeconds << " s; what was running was cancelled";
                return CliError("timeout", s.str(), "raise --timeout");
            }
            return CliError("cancelled", terminated.load() ? "terminated; what was running was cancelled" : "interrupted; what was running was cancelled");
        }

        CliError Runner::toCliError(const agent::ToolError& e) const {
            if (e.code == "cancelled" && cancelled()) {
                CliError c = cancellation();
                c.data = e.data;
                return c;
            }
            return CliError(e.code.empty() ? std::string("failed") : e.code, e.message, e.hint, e.data);
        }

        agent::ToolResult Runner::callRaw(const std::string& tool, const json& args, bool honourDeadline) {
            if (cancelled(honourDeadline)) throw cancellation();
            agent::ToolResult r = workspace().call(tool, args, context(honourDeadline));
            for (const std::string& w : r.warnings) warnings_.push_back(tool + ": " + w);
            return r;
        }

        json Runner::call(const std::string& tool, const json& args) {
            agent::ToolResult r = callRaw(tool, args);
            if (!r.ok) throw toCliError(r.error);
            return std::move(r.value);
        }

        // `run` until it ends, or until --timeout: then cancel, wait up to
        // 10 s for the job to stop, and report a timeout.
        json Runner::runPipeline(json runArgs) {
            double wait = -1.0;
            if (deadline_) wait = std::max(0.0, std::chrono::duration<double>(*deadline_ - Clock::now()).count());
            runArgs["wait_s"] = wait;
            json outcome = call("run", runArgs);
            if (outcome.value("status", "") != "running") return outcome;
            callRaw("cancel_run", json::object(), false);
            agent::ToolResult last = callRaw("run_status", {{"wait_s", 10}}, false);
            CliError e = cancellation();
            if (e.code != "timeout") e = CliError("timeout", "the run did not finish within --timeout; it was cancelled", "raise --timeout");
            e.data = last.ok ? last.value : outcome;
            throw e;
        }

        json Runner::readJsonOption(const std::string& option, const std::string& value) {
            std::string text;
            if (value == "-") {
                if (stdinUsed_) throw UsageError("only one option can read stdin (-)");
                stdinUsed_ = true;
                if (!readStandardInput(text)) throw CliError("io_error", "cannot read stdin for --" + option);
            } else if (!value.empty() && value[0] == '@') {
                if (!readWholeFile(value.substr(1), text))
                    throw CliError("not_found", "cannot read " + displayPath(value.substr(1)) + " (--" + option + ")");
            } else {
                text = value;
            }
            try {
                return json::parse(asUtf8(text));
            } catch (const json::exception& e) {
                throw UsageError("--" + option + " is not valid JSON: " + e.what());
            }
        }

        json Runner::openArgs(const std::vector<Option>& open) const {
            json a = json::object();
            for (const Option& o : open) {
                if (o.name == "page-order") a["page_order"] = lower(trim(o.value));
                else if (o.name == "page-c") a["c"] = toInteger(o.name, o.value);
                else if (o.name == "page-t") a["t"] = toInteger(o.name, o.value);
                else if (o.name == "page-z") a["z"] = toInteger(o.name, o.value);
                else if (o.name == "voxel") a["voxel_um"] = numberList(o.name, o.value, 3);
                else if (o.name == "sim") {
                    const std::vector<std::string> parts = split(o.value, ',');
                    if (parts.size() < 2 || parts.size() > 3 || (parts.size() == 3 && lower(parts[2]) != "fast"))
                        throw UsageError("--sim expects directions,phases[,fast], not '" + o.value + "'");
                    a["sim"] = {{"ndirs", toInteger(o.name, parts[0])}, {"nphases", toInteger(o.name, parts[1])}, {"fast", parts.size() == 3}};
                } else if (o.name == "no-sim") a["sim"] = false;
                else if (o.name == "dataset-tile") a["tile"] = toInteger(o.name, o.value);
                else if (o.name == "full-load") a["full_load"] = true;
            }
            return a;
        }

        json Runner::renderArgs(const std::vector<Option>& options) const {
            json a = json::object();
            for (const Option& o : options) {
                const std::string& n = o.name;
                const std::string& v = o.value;
                if (n == "step") a["step"] = stepValue(v);
                else if (n == "plane" || n == "layout" || n == "format") a[n] = lower(trim(v));
                else if (n == "z") a["z"] = v.find(',') == std::string::npos ? json(toInteger(n, v)) : integerList(n, v);
                else if (n == "t" || n == "y" || n == "x" || n == "label") a[n] = toInteger(n, v);
                else if (n == "channels") a["channels"] = integerList(n, v);
                else if (n == "window") {
                    const std::string w = lower(trim(v));
                    if (w == "auto" || w == "full") {
                        a["window"] = w;
                        continue;
                    }
                    // c=lo:hi[:gamma],...
                    json windows = json::array();
                    for (const std::string& part : split(v, ',')) {
                        const std::size_t eq = part.find('=');
                        const std::vector<std::string> range = split(eq == std::string::npos ? std::string() : part.substr(eq + 1), ':');
                        if (eq == std::string::npos || range.size() < 2 || range.size() > 3)
                            throw UsageError("--window expects auto, full or channel=lo:hi[:gamma],..., not '" + v + "'");
                        json cw = {{"channel", toInteger(n, part.substr(0, eq))}, {"lo", toNumber(n, range[0])}, {"hi", toNumber(n, range[1])}};
                        if (range.size() == 3) cw["gamma"] = toNumber(n, range[2]);
                        windows.push_back(std::move(cw));
                    }
                    a["windows"] = std::move(windows);
                } else if (n == "labels") a["labels"] = true;
                else if (n == "no-labels") a["labels"] = false;
                else if (n == "solo") a["solo"] = true;
                else if (n == "label-opacity") a["label_opacity"] = toNumber(n, v);
                else if (n == "region") a["region"] = integerList(n, v, 4);
                else if (n == "max-size") a["max_size"] = toInteger(n, v);
                else if (n == "no-physical-z") a["physical_z"] = false;
            }
            return a;
        }

        json Runner::statsArgs(const std::vector<Option>& options) const {
            json a = json::object();
            for (const Option& o : options) {
                const std::string& n = o.name;
                if (n == "step") a["step"] = stepValue(o.value);
                else if (n == "t") a["t"] = lower(trim(o.value)) == "all" ? json("all") : json(toInteger(n, o.value));
                else if (n == "channels") a["channels"] = integerList(n, o.value);
                else if (n == "percentiles") a["percentiles"] = numberList(n, o.value);
                else if (n == "histogram") a["histogram_bins"] = toInteger(n, o.value);
                else if (n == "no-labels") a["labels"] = false;
            }
            return a;
        }

        json Runner::exportArgs(const std::vector<Option>& options, const std::string& path) const {
            json a = {{"path", path}};
            json tiff = json::object(), zarr = json::object();
            std::string format;
            for (const Option& o : options)
                if (o.name == "format") format = lower(trim(o.value));
            const std::string lowerPath = lower(slashes(path));
            const bool chunked = format == "zarr" || format == "n5" ||
                                 (format.empty() && (endsWith(lowerPath, ".zarr") || endsWith(lowerPath, ".zarr/") ||
                                                     endsWith(lowerPath, ".n5") || endsWith(lowerPath, ".n5/")));
            for (const Option& o : options) {
                const std::string& n = o.name;
                const std::string& v = o.value;
                if (n == "step") a["step"] = stepValue(v);
                else if (n == "format") a["format"] = format;
                else if (n == "dtype") a["dtype"] = lower(trim(v));
                else if (n == "scaling") a["scaling"] = lower(trim(v));
                else if (n == "range") a["range"] = numberList(n, v, 2);
                else if (n == "percentiles") a["percentiles"] = numberList(n, v, 2);
                else if (n == "t" || n == "z") a[n] = rangeValue(n, v);
                else if (n == "channels") a["channels"] = integerList(n, v);
                else if (n == "compression") tiff["compression"] = lower(trim(v));
                else if (n == "tiled") tiff["tiled"] = true;
                else if (n == "tile") tiff["tile"] = integerList(n, v, 2);
                else if (n == "bigtiff") tiff["bigtiff"] = true;
                else if (n == "no-bigtiff") tiff["bigtiff"] = false;
                else if (n == "level") (chunked ? zarr : tiff)["level"] = toInteger(n, v);
                else if (n == "pyramid") (chunked ? zarr : tiff)["pyramid_levels"] = toInteger(n, v);
                else if (n == "chunk") zarr["chunk"] = integerList(n, v, 5);
                else if (n == "codec") zarr["codec"] = lower(trim(v));
                else if (n == "zarr-version") zarr["version"] = toInteger(n, v);
                else if (n == "labels") a["include_labels"] = true;
                else if (n == "labels-only") a["labels_only"] = true;
                else if (n == "pipeline-sidecar") a["include_pipeline"] = true;
            }
            if (!tiff.empty()) a["tiff"] = std::move(tiff);
            if (!zarr.empty()) a["zarr"] = std::move(zarr);
            return a;
        }

        json Runner::imageInfo(const agent::ToolResult& r) {
            json images = json::array();
            for (const agent::Attachment& a : r.images)
                images.push_back({{"path", pathOrNull(a.path)}, {"mime_type", a.mimeType}, {"width", a.width}, {"height", a.height}, {"bytes", a.bytes.size()}});
            if (!r.images.empty() && removeScratch_)
                warnings_.push_back("the image is in the scratch directory, which is removed at exit: pass --keep-scratch, or use render --out");
            return images;
        }

        // Never over an existing file that is not an image: `render --out` must
        // not be a way to overwrite a document with a picture by a slip of the
        // name.
        void checkRenderTarget(const std::string& path) {
            const fs::path target = fs::u8path(path);
            const std::string ext = lower(target.extension().u8string());
            std::error_code ec;
            if (!fs::exists(target, ec)) return;
            if (fs::is_directory(target, ec)) throw CliError("invalid_argument", displayPath(path) + " is a directory");
            if (ext != ".png" && ext != ".jpg" && ext != ".jpeg")
                throw CliError("invalid_argument", "will not write over " + displayPath(path) + ": it exists and is not an image",
                               "choose a new file name, or one ending in .png or .jpg");
        }

        // The targets are also checked before the state options and the run: a
        // slip in a file name should not cost a full load or a long run first.
        void Runner::checkRenderTargets() const {
            if (command_ == "render" && args_.has("out")) checkRenderTarget(args_.value("out"));
            if (command_ != "run") return;
            for (const Action& act : args_.actions) {
                if (act.name != "render") continue;
                try {
                    checkRenderTarget(act.argument);
                } catch (CliError& e) {
                    e.message = "--render " + act.argument + ": " + e.message;
                    throw;
                }
            }
        }

        // Draws with the render tool and writes the image to `path`; the file
        // may have appeared since the first check.
        json Runner::renderTo(const std::string& path, json a, bool base64) {
            checkRenderTarget(path);
            const std::string ext = lower(fs::u8path(path).extension().u8string());
            // The file name says what the file holds; the JPEG fallback of an
            // unnamed format is only for images that are not saved.
            if (!a.contains("format")) {
                if (ext == ".png") a["format"] = "png";
                else if (ext == ".jpg" || ext == ".jpeg") a["format"] = "jpeg";
            }
            agent::ToolResult r = callRaw("render", a);
            if (!r.ok) throw toCliError(r.error);
            if (r.images.empty()) throw CliError("internal", "the render tool returned no image");
            const agent::Attachment& image = r.images.front();
            std::string error;
            if (!app::writeBinaryFile(path, image.bytes, &error)) throw CliError("io_error", "cannot write " + displayPath(path) + ": " + error);
            json caption = r.value;
            if (args_.global.keepScratch && !image.path.empty()) caption["scratch_path"] = displayPath(image.path);
            caption["path"] = displayPath(path);
            caption["mime_type"] = image.mimeType;
            caption["width"] = image.width;
            caption["height"] = image.height;
            caption["bytes"] = image.bytes.size();
            if (base64) caption["base64"] = app::base64Encode(image.bytes);
            return caption;
        }

        // --- the state options -----------------------------------------------

        void Runner::applyState() {
            const StateOptions& s = args_.state;
            if (!s.any()) return;
            const json open = openArgs(s.open);
            if (!s.pipeline.empty()) {
                json a = {{"path", s.pipeline}};
                // The pipeline's Load step says how its dataset opens; open
                // options on the command line are applied by a second open.
                if (!s.dataset.empty() && open.empty()) a["dataset"] = s.dataset;
                call("load_pipeline", a);
            }
            if (!s.dataset.empty() && (s.pipeline.empty() || !open.empty())) {
                json a = open;
                a["path"] = s.dataset;
                call("open_dataset", a);
            } else if (s.dataset.empty() && !open.empty()) {
                warnings_.push_back("the open options apply to --dataset, which was not given; ignored");
            }
            if (!s.steps.empty()) {
                const json steps = readJsonOption("steps", s.steps);
                if (!steps.is_array()) throw UsageError("--steps expects a JSON array of {kind, preset?, params?, enabled?, name?}");
                for (const json& entry : steps) {
                    const json e = entry.is_string() ? json{{"kind", entry}} : entry;
                    if (!e.is_object() || !e.contains("kind") || !e["kind"].is_string())
                        throw UsageError("every --steps entry needs a kind: " + entry.dump());
                    json a = {{"kind", e["kind"]}};
                    for (const char* key : {"params", "preset", "name", "at"})
                        if (e.contains(key)) a[key] = e[key];
                    const json step = call("add_step", a);
                    const json number = step.is_object() ? step.value("step", json()) : json();
                    if (e.contains("enabled") && e["enabled"].is_boolean() && !e["enabled"].get<bool>())
                        call("set_step_enabled", {{"step", number}, {"enabled", false}});
                }
            }
            for (const std::string& set : s.sets) {
                const std::size_t eq = set.find('=');
                const std::string left = eq == std::string::npos ? std::string() : set.substr(0, eq);
                // Keys never contain a dot, names might: the last dot splits.
                const std::size_t dot = left.rfind('.');
                if (eq == std::string::npos || dot == std::string::npos || dot == 0 || dot + 1 == left.size())
                    throw UsageError("--set expects <step>.<key>=<value>, not '" + set + "'", "for example --set 2.gamma=0.8 or --set Contrast.mode=auto");
                const std::string text = set.substr(eq + 1);
                json value;
                try {
                    value = json::parse(text);
                } catch (const json::exception&) {
                    value = text;   // not JSON: text, which the parameter's spec coerces
                }
                call("set_params", {{"step", stepValue(left.substr(0, dot))}, {"params", {{left.substr(dot + 1), value}}}});
            }
        }

        // --- commands -------------------------------------------------------------

        json Runner::cmdVersion() {
            json formats = json::array();
            if (app::exportFormatAvailable(app::ExportFormat::Tiff)) {
                formats.push_back("tiff");
                formats.push_back("ome-tiff");
            }
            if (app::exportFormatAvailable(app::ExportFormat::Zarr)) formats.push_back("zarr");
            if (app::exportFormatAvailable(app::ExportFormat::N5)) formats.push_back("n5");
            if (app::exportFormatAvailable(app::ExportFormat::Raw)) formats.push_back("raw");
            std::string env;
            try {
                env = pyenv::environmentDirectory();
            } catch (const std::exception&) {
                // Reported as null: version must answer even when the data
                // directory cannot be worked out.
            }
            return {{"name", "sirius-cli"},
                    {"version", SIRIUS_VERSION},
                    // which build: what an engine image's BUILD.json holds (core/build_info.hpp)
                    {"build", app::toJson(app::buildInfo())},
                    {"schema", kOutputSchema},
                    {"protocols", {{"session", kSessionProtocol}, {"mcp", mcpVersions()}}},
                    {"features",
                     {{"cuda", sirius::cudaAvailable()},
                      {"cuda_devices", sirius::cudaDeviceCount()},
                      {"zarr", app::zarrSupported()},
                      {"export_formats", formats},
                      {"readable_extensions", app::readableExtensions()}}},
                    {"paths",
                     {{"executable_dir", pathOrNull(app::applicationDirectory())},
                      {"help", pathOrNull(app::helpDirectory())},
                      {"worker", pathOrNull(scriptDir())},
                      {"python_env", pathOrNull(env)}}}};
        }

        json Runner::cmdInfo() {
            json a = openArgs(args_.state.open);
            a["path"] = args_.positionals.front();
            if (args_.has("open")) a["open"] = true;
            return call("dataset_info", a);
        }

        json Runner::cmdOps() {
            const std::vector<std::string>& kinds = args_.positionals;
            if (kinds.size() == 1 && args_.has("detail")) return call("describe_operation", {{"kind", kinds.front()}});
            json a = json::object();
            if (args_.has("group")) a["group"] = args_.value("group");
            if (args_.has("detail")) a["detail"] = true;
            if (args_.has("plugins")) a["include_plugins"] = true;
            if (kinds.size() == 1) a["kind"] = kinds.front();
            json r = call("list_operations", a);
            if (kinds.empty()) return r;
            const char* key = r.contains("operations") ? "operations" : "items";
            json kept = json::array();
            for (const std::string& kind : kinds) {
                bool found = false;
                if (r.contains(key) && r[key].is_array())
                    for (const json& op : r[key])
                        if (op.is_object() && op.value("kind", "") == kind) {
                            kept.push_back(op);
                            found = true;
                        }
                if (!found) throw CliError("unknown_operation", "no operation of kind '" + kind + "'", "sirius-cli ops lists them");
            }
            r[key] = std::move(kept);
            return r;
        }

        std::optional<json> Runner::cmdHelp() {
            if (args_.positionals.empty()) {
                if (args_.has("markdown")) throw UsageError("help --markdown needs a page", "sirius-cli help lists them");
                return call("get_help", json::object());
            }
            const std::string page = args_.positionals.front();
            rawOutput_ = args_.has("markdown");
            json r;
            try {
                r = call("get_help", {{"page", page}});
            } catch (const CliError& e) {
                if (e.code != "not_found" && e.code != "invalid_argument") throw;
                // An operation kind whose page has another name (the einsum
                // kinds share one, a plugin's may differ): the kind finds it.
                try {
                    r = call("get_help", {{"kind", page}});
                } catch (const CliError& again) {
                    if (again.code == "cancelled" || again.code == "timeout") throw;
                    throw e;   // the page's own error names what was asked
                }
            }
            if (!rawOutput_) return r;
            // The page as it is on disk: the JSON form cuts long pages short.
            std::string markdown;
            const std::string path = r.is_object() ? r.value("path", std::string()) : std::string();
            if (path.empty() || !readWholeFile(path, markdown)) markdown = r.is_object() ? r.value("markdown", std::string()) : std::string();
            if (trim(markdown).empty()) throw CliError("not_found", "no help page '" + page + "'", "sirius-cli help lists them");
            if (markdown.back() != '\n') markdown += '\n';
            writeRaw(markdown);
            return std::nullopt;
        }

        json Runner::cmdValidate() {
            json r = call("validate", json::object());
            if (!r.is_object() || r.value("ok", true)) return r;
            if (!r.value("has_dataset", true))
                throw CliError("no_dataset", "no dataset is open", "pass --dataset <path>, or a --pipeline whose Load step names one", r);
            std::string first;
            if (r.contains("steps") && r["steps"].is_array())
                for (const json& s : r["steps"]) {
                    if (!s.is_object() || !s.contains("errors") || !s["errors"].is_array() || s["errors"].empty()) continue;
                    first = "step " + s.value("step", json()).dump() + " (" + s.value("name", s.value("kind", std::string())) + "): " +
                            (s["errors"].front().is_string() ? s["errors"].front().get<std::string>() : s["errors"].front().dump());
                    break;
                }
            throw CliError("validation", first.empty() ? std::string("the pipeline cannot run as it is") : first,
                           "fix the step (sirius-cli ops <kind> --detail lists its parameters), then validate again", r);
        }

        json Runner::cmdRun() {
            json runArgs = json::object();
            if (args_.has("to")) runArgs["step"] = stepValue(args_.value("to"));
            if (args_.has("force")) runArgs["force"] = true;
            json result = {{"run", runPipeline(runArgs)}, {"renders", json::array()}, {"exports", json::array()}};
            for (const Action& act : args_.actions) {
                try {
                    if (act.name == "render") {
                        json a = renderArgs(act.options);
                        a["run"] = true;
                        result["renders"].push_back(renderTo(act.argument, a, act.has("base64")));
                    } else if (act.name == "export") {
                        json a = exportArgs(act.options, act.argument);
                        a["run"] = true;
                        result["exports"].push_back(call("export_result", a));
                    } else if (act.name == "stats") {
                        json a = statsArgs(act.options);
                        a["run"] = true;
                        result["statistics"] = call("statistics", a);
                    } else if (act.name == "save-pipeline") {
                        result["pipeline"] = call("save_pipeline", {{"path", act.argument}});
                    } else if (act.name == "export-python") {
                        result["python"] = call("export_python", {{"path", act.argument}});
                    }
                } catch (CliError& e) {
                    // Which action failed, when there are several.
                    e.message = "--" + act.name + (act.argument.empty() ? "" : " " + act.argument) + ": " + e.message;
                    throw;
                }
            }
            return result;
        }

        // A one-shot command that runs has not run anything before it, so the
        // tools' default (the last run's target, else the last step with an
        // output) would always be the Load step. What it runs and shows is the
        // end of the pipeline instead, as `run --render` does.
        void Runner::defaultToPipelineEnd(json& a) {
            if (a.contains("step") || !a.value("run", false)) return;
            const app::Pipeline& p = workspace().workbench().pipeline();
            int last = 0;
            for (int i = 1; i < p.size(); ++i)
                if (p.at(i).enabled) last = i;
            a["step"] = last + 1;
        }

        json Runner::cmdRender() {
            if (!args_.has("out")) throw UsageError("render needs --out <file.png>", "the image goes to a file; `call render` keeps it in the scratch directory");
            json a = renderArgs(args_.options);
            a["run"] = !args_.has("no-run");
            defaultToPipelineEnd(a);
            return renderTo(args_.value("out"), a, args_.has("base64"));
        }

        json Runner::cmdDiagnostics() {
            json step;
            if (args_.has("step")) step = stepValue(args_.value("step"));
            if (!args_.has("no-run")) {
                json runArgs = json::object();
                if (!step.is_null()) runArgs["step"] = step;
                runPipeline(runArgs);
            }
            json a = json::object();
            if (!step.is_null()) a["step"] = step;
            if (args_.has("detail")) a["detail"] = true;
            json d = call("get_diagnostics", a);
            if (!args_.has("images")) return d;

            const std::string dir = args_.value("images");
            std::error_code ec;
            fs::create_directories(fs::u8path(dir), ec);
            if (!fs::is_directory(fs::u8path(dir), ec)) throw CliError("io_error", "cannot create " + displayPath(dir));
            json files = json::array();
            // One image of render_diagnostics into DIR; false when there is no
            // such image (the end of a tab).
            auto renderOne = [&](json request, const json& listedTab) {
                if (!step.is_null()) request["step"] = step;
                agent::ToolResult r = callRaw("render_diagnostics", request);
                if (!r.ok) {
                    if (r.error.code == "cancelled") throw toCliError(r.error);
                    return false;
                }
                if (r.images.empty()) return false;
                const agent::Attachment& image = r.images.front();
                const json& v = r.value;
                // The file is named after the tab the image is shown under.
                const json& t = listedTab.is_string() ? listedTab : v.is_object() && v.contains("tab") ? v["tab"]
                                                                                                       : listedTab;
                const std::string tab = t.is_string() ? t.get<std::string>() : std::string("images");
                const std::string stepText = v.is_object() && v.contains("step") ? v["step"].dump() : std::string("0");
                const std::string index = v.is_object() && v.contains("index") ? v["index"].dump() : request.value("index", json(0)).dump();
                const std::string name = "step" + stepText + "-" + slug(tab) + "-" + index + (image.mimeType == "image/jpeg" ? ".jpg" : ".png");
                const std::string path = (fs::u8path(dir) / fs::u8path(name)).u8string();
                std::string error;
                if (!app::writeBinaryFile(path, image.bytes, &error)) throw CliError("io_error", "cannot write " + displayPath(path) + ": " + error);
                json entry = v.is_object() ? v : json::object();
                entry["path"] = displayPath(path);
                files.push_back(std::move(entry));
                return true;
            };
            if (d.is_object() && d.contains("images") && d["images"].is_array()) {
                // The listed index counts every image of the step; with a tab
                // it would count that tab's, so the tab is left out.
                std::size_t i = 0;
                for (const json& img : d["images"]) {
                    const json request = {{"index", img.is_object() && img.contains("index") ? img["index"] : json(i)}};
                    if (!renderOne(request, img.is_object() && img.contains("tab") ? img["tab"] : json())) warnings_.push_back("diagnostic image " + std::to_string(i) + " could not be drawn");
                    ++i;
                }
            } else {
                // Without a list, each tab is walked until an index has no image.
                json tabs = d.is_object() && d.contains("tabs") && d["tabs"].is_array() ? d["tabs"] : json::array();
                if (tabs.empty()) tabs.push_back(nullptr);
                for (const json& tab : tabs)
                    for (int index = 0; index < 256; ++index) {
                        json request = {{"index", index}};
                        if (!tab.is_null()) request["tab"] = tab;
                        if (!renderOne(request, tab)) break;
                    }
            }
            d["image_files"] = std::move(files);
            return d;
        }

        json Runner::cmdExportTraining() {
            if (!args_.has("dir")) throw UsageError("export-training needs --dir <folder>");
            json step;
            if (args_.has("step")) step = stepValue(args_.value("step"));
            if (!args_.has("no-run")) {
                json runArgs = json::object();
                if (!step.is_null()) runArgs["step"] = step;
                runPipeline(runArgs);
            }
            json a = {{"directory", args_.value("dir")}};
            if (!step.is_null()) a["step"] = step;
            if (args_.has("sample")) a["sample"] = args_.value("sample");
            if (args_.has("slices")) a["slices"] = true;
            if (args_.has("min-voxels")) a["min_voxels"] = toInteger("min-voxels", args_.value("min-voxels"));
            if (args_.has("image-dtype")) a["image_dtype"] = lower(args_.value("image-dtype"));
            if (args_.has("image-scaling")) a["image_scaling"] = lower(args_.value("image-scaling"));
            return call("export_training_data", a);
        }

        json Runner::cmdCall() {
            app::HeadlessWorkbench& ws = workspace();
            const std::string& tool = args_.tool;
            if (!ws.hasTool(tool)) throw CliError("unknown_tool", "no tool '" + tool + "'", "sirius-cli tools --names lists them");
            json schema = json::object();
            for (const agent::ToolDescriptor& d : ws.tools())
                if (d.name == tool) schema = d.inputSchema;
            const json properties = schema.is_object() && schema.contains("properties") ? schema["properties"] : json::object();
            auto parameterList = [&] {
                std::vector<std::string> names;
                for (auto it = properties.begin(); it != properties.end(); ++it) {
                    std::string n = it.key();
                    std::replace(n.begin(), n.end(), '_', '-');
                    names.push_back("--" + n);
                }
                return names.empty() ? std::string("it takes none") : "its parameters: " + join(names, " ");
            };

            json base = json::object(), flags = json::object();
            const std::vector<std::string>& in = args_.toolArgs;
            for (std::size_t i = 0; i < in.size(); ++i) {
                const std::string& arg = in[i];
                if (arg.size() < 3 || arg.compare(0, 2, "--") != 0)
                    throw UsageError("unexpected argument '" + arg + "' for call " + tool, "parameters are given as --name value; " + parameterList());
                std::string name = arg.substr(2), inlineValue;
                bool inlineGiven = false;
                if (const std::size_t eq = name.find('='); eq != std::string::npos) {
                    inlineValue = name.substr(eq + 1);
                    name.resize(eq);
                    inlineGiven = true;
                }
                auto nextValue = [&]() -> std::string {
                    if (inlineGiven) return inlineValue;
                    if (i + 1 >= in.size()) throw UsageError("--" + name + " needs a value");
                    return in[++i];
                };
                if (name == "args") {
                    base = readJsonOption("args", nextValue());
                    if (!base.is_object()) throw UsageError("--args expects a JSON object");
                    continue;
                }
                std::string key = name;
                std::replace(key.begin(), key.end(), '-', '_');
                bool negated = false;
                if (!properties.contains(key) && key.compare(0, 3, "no_") == 0 && properties.contains(key.substr(3))) {
                    key = key.substr(3);
                    negated = true;
                }
                if (!properties.contains(key)) throw UsageError("unknown parameter --" + name + " for " + tool, parameterList());
                const json& spec = properties[key];
                std::set<std::string> types;
                collectTypes(spec, types);
                const bool booleanAllowed = types.empty() || types.count("boolean") > 0;
                if (negated) {
                    if (!booleanAllowed || inlineGiven) throw UsageError("--" + name + " is not a switch");
                    flags[key] = false;
                    continue;
                }
                // A switch: --flag, --flag=true|false. A parameter that may be
                // a boolean or something else takes a value when one follows.
                if (types.size() == 1 && booleanAllowed && !types.empty()) {
                    if (!inlineGiven) flags[key] = true;
                    else if (lower(inlineValue) == "true" || lower(inlineValue) == "false") flags[key] = lower(inlineValue) == "true";
                    else throw UsageError("--" + name + " is a switch (--" + name + " or --no-" + name + ")");
                    continue;
                }
                if (booleanAllowed && !types.empty() && !inlineGiven && (i + 1 >= in.size() || in[i + 1].compare(0, 2, "--") == 0)) {
                    flags[key] = true;
                    continue;
                }
                const std::string text = nextValue();
                // Arrays: a comma list or JSON; objects: JSON, @file or -.
                std::function<json(const std::string&, const json&)> convert = [&](const std::string& t, const json& s) -> json {
                    std::set<std::string> allowed;
                    collectTypes(s, allowed);
                    if (allowed.size() == 1 && allowed.count("string")) return t;
                    if ((allowed.count("object") || allowed.count("array")) && (t == "-" || (!t.empty() && t[0] == '@')))
                        return readJsonOption(name, t);
                    try {
                        json parsed = json::parse(t);
                        if (allowed.empty() || matchesType(parsed, allowed)) {
                            if (allowed.count("integer") && !allowed.count("number") && parsed.is_number_float())
                                parsed = static_cast<long long>(parsed.get<double>());
                            return parsed;
                        }
                    } catch (const json::exception&) {
                        // not JSON: a list, or text
                    }
                    if (allowed.count("array") && (t.empty() || t[0] != '[')) {
                        const json* items = arrayItems(s);
                        json out = json::array();
                        for (const std::string& part : split(t, ',')) out.push_back(convert(part, items ? *items : json::object()));
                        return out;
                    }
                    if (allowed.empty() || allowed.count("string")) return t;
                    std::vector<std::string> names(allowed.begin(), allowed.end());
                    throw UsageError("--" + name + " expects " + join(names, " or ") + ", not '" + t + "'");
                };
                flags[key] = convert(text, spec);
            }
            for (auto it = flags.begin(); it != flags.end(); ++it) base[it.key()] = it.value();

            agent::ToolResult r = callRaw(tool, base);
            if (!r.ok) throw toCliError(r.error);
            json value = std::move(r.value);
            if (!r.images.empty()) {
                const json images = imageInfo(r);
                if (value.is_object()) {
                    value["image"] = images.front();
                    if (images.size() > 1) value["images"] = images;
                }
            }
            return value;
        }

        json Runner::cmdTools() {
            json list = json::array();
            for (const agent::ToolDescriptor& d : workspace().tools()) {
                if (args_.has("names")) {
                    list.push_back(d.name);
                    continue;
                }
                // The same objects as MCP's tools/list, so the two never disagree.
                list.push_back(agent::toolJson(d));
            }
            return list;
        }

        std::string Runner::toolHelp() {
            try {
                app::HeadlessWorkbench& ws = workspace();
                for (const agent::ToolDescriptor& d : ws.tools()) {
                    if (d.name != args_.tool) continue;
                    std::string s = "\nTool " + d.name + (d.title.empty() ? "" : " - " + d.title) + "\n  " + d.description + "\n";
                    if (d.inputSchema.is_object() && d.inputSchema.contains("properties")) {
                        s += "Parameters:\n";
                        const json& props = d.inputSchema["properties"];
                        for (auto it = props.begin(); it != props.end(); ++it) {
                            std::string n = it.key();
                            std::replace(n.begin(), n.end(), '_', '-');
                            std::set<std::string> types;
                            collectTypes(it.value(), types);
                            std::vector<std::string> names(types.begin(), types.end());
                            // A choice shows its values rather than "string".
                            if (it.value().contains("enum") && it.value()["enum"].is_array()) {
                                names.clear();
                                for (const json& v : it.value()["enum"]) names.push_back(v.is_string() ? v.get<std::string>() : v.dump());
                            }
                            std::string left = "  --" + n + (names.empty() ? "" : " <" + join(names, "|") + ">");
                            if (left.size() < 32) left.resize(32, ' ');
                            else left += "  ";
                            s += left + it.value().value("description", std::string()) + "\n";
                        }
                    }
                    return s;
                }
                return "\nThere is no tool '" + args_.tool + "' (sirius-cli tools --names lists them).\n";
            } catch (const std::exception&) {
                return {};
            } catch (const CliError&) {
                return {};
            }
        }

        // --- the Python worker ----------------------------------------------------

        CliError Runner::workerUnavailable(const app::WorkerStartError& e) const {
            json data;
            try {
                data = e.toJson();
            } catch (const std::exception&) {
                data = {{"kind", app::toString(e.kind)}, {"message", e.what()}};
            }
            data["python"] = e.interpreter;
            if (e.setupWouldHelp()) data["fix"] = "sirius-cli worker setup --yes";
            return CliError("worker_unavailable", e.what(), e.hint.empty() ? app::HeadlessOptions{}.setupHint : e.hint, data);
        }

        CliError Runner::setupFailure(const pyenv::SetupResult& r) const {
            std::string code = "failed";
            switch (r.failure) {
                case pyenv::Failure::NoPython:
                case pyenv::Failure::UnsupportedPython:
                case pyenv::Failure::NoEnsurepip: code = "python_not_found"; break;
                case pyenv::Failure::Locked:
                case pyenv::Failure::InUse: code = "busy"; break;
                case pyenv::Failure::DiskFull: code = "io_error"; break;
                case pyenv::Failure::Cancelled: return CliError(cancellation().code, r.message.empty() ? cancellation().message : r.message, r.hint, r.toJson());
                default: break;
            }
            return CliError(code, r.message.empty() ? std::string("the Python environment was not set up") : r.message, r.hint, r.toJson());
        }

        // y/N on the terminal; false anywhere else (the caller then asks for --yes).
        bool Runner::confirm(const std::string& question) {
            reporter_.finish();
            writeError(question + " [y/N] ");
            // The answer is read on a thread of its own while this one watches
            // for Ctrl+C: a POSIX read restarts after the signal, so it would
            // only return at Enter. After a Ctrl+C the reader stays blocked
            // until the process ends, which the cancellation soon makes it do.
            struct Pending {
                std::mutex mutex;
                std::condition_variable done;
                bool finished = false, answered = false;
                std::string answer;
            };
            auto pending = std::make_shared<Pending>();
            std::thread([pending] {
                std::string line;
                const bool ok = readAnswer(line);
                {
                    const std::lock_guard<std::mutex> g(pending->mutex);
                    pending->answer = std::move(line);
                    pending->answered = ok;
                    pending->finished = true;
                }
                pending->done.notify_all();
            }).detach();
            std::unique_lock<std::mutex> lock(pending->mutex);
            while (!pending->finished && !cancelled(false)) pending->done.wait_for(lock, std::chrono::milliseconds(50));
            const bool finished = pending->finished, answered = pending->answered;
            std::string answer = pending->answer;
            lock.unlock();
            if (!finished || !answered || trim(answer).empty()) {
                // A console read ends at Ctrl+C before the control handler's
                // thread has run: give it a moment to say so.
                for (int i = 0; i < 10 && !cancelled(false); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            if (!finished || !answered) writeError("\n");
            // Ctrl+C while the question waits is a no that says so.
            if (cancelled(false)) throw cancellation();
            answer = lower(trim(answer));
            return answered && (answer == "y" || answer == "yes");
        }

        json Runner::cmdWorkerStatus() {
            const std::string dir = scriptDir();
            const pyenv::Interpreter interpreter = pyenv::workerInterpreter(args_.global.python, "");
            const std::string uv = pyenv::findUv();
            const std::vector<std::string> required = pyenv::requirements(dir, false);
            json extras = json::array();
            for (const std::string& r : pyenv::requirements(dir, true))
                if (std::find(required.begin(), required.end(), r) == required.end()) extras.push_back(r);
            json candidates = json::array();
            for (const std::string& c : pyenv::pythonCandidates()) candidates.push_back(slashes(c));
            if (!args_.global.workerDir.empty() && !app::host::isFile(dir + "/sirius_worker/__main__.py"))
                warnings_.push_back("--worker-dir " + dir + " holds no sirius_worker/__main__.py");
            return {{"interpreter", {{"path", slashes(interpreter.path)}, {"source", pyenv::toString(interpreter.source)}}},
                    {"environment", pyenv::environmentStatus(dir, false).toJson()},
                    {"uv", uv.empty() ? json(nullptr) : json{{"path", slashes(uv)}, {"version", pyenv::uvVersion(uv)}}},
                    {"candidates", candidates},
                    {"worker_dir", pathOrNull(dir)},
                    {"requirements", {{"required", required}, {"extras", extras}}}};
        }

        json Runner::cmdWorkerCheck() {
            app::LocalWorker worker;
            worker.setPython(args_.global.python);
            worker.setScriptDir(scriptDir());
            const std::string& backend = args_.global.backend;
            worker.setDevice(backend == "cpu"    ? std::string("cpu")
                             : backend == "cuda" ? (args_.global.cudaDevice < 0 ? std::string("cuda") : "cuda:" + std::to_string(args_.global.cudaDevice))
                                                 : std::string("auto"));
            worker.setAllowInstall(false);
            worker.setSetupHint(app::HeadlessOptions{}.setupHint);
            worker.setLogHandler([this](const std::string& line) { reporter_.log("worker", line); });
            // The log handler runs on the worker's stderr thread; it goes
            // before this frame's reporter does.
            struct Detach {
                app::LocalWorker& w;
                ~Detach() {
                    w.setLogHandler({});
                    w.stop();
                }
            } detach{worker};

            const Clock::time_point started = Clock::now();
            std::unique_ptr<app::RemoteWorker> remote;
            try {
                remote = worker.connect([this] { return cancelled(); });
            } catch (const app::WorkerStartError& e) {
                throw workerUnavailable(e);
            } catch (const std::exception& e) {
                if (app::isCancellation(e)) throw cancellation();
                throw CliError("worker_unavailable", std::string("the Python worker did not answer: ") + e.what(), app::HeadlessOptions{}.setupHint,
                               {{"kind", "failed"}, {"message", e.what()}});
            }
            const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
            const app::WorkerCapabilities caps = remote->capabilities();
            remote->close();
            remote.reset();
            json interpreter = {{"path", slashes(worker.runningPython())}, {"source", nullptr}};
            try {
                const pyenv::Interpreter i = worker.interpreter();
                if (interpreter["path"].get<std::string>().empty()) interpreter["path"] = slashes(i.path);
                interpreter["source"] = pyenv::toString(i.source);
            } catch (const std::exception&) {
                // the interpreter's source is a detail; the handshake worked
            }
            return {{"interpreter", interpreter},
                    {"capabilities",
                     {{"version", caps.version},
                      {"protocol", caps.protocolVersion},
                      {"methods", caps.methods},
                      {"cuda", caps.cuda},
                      {"device", caps.device},
                      {"hostname", caps.hostname},
                      {"python", caps.python}}},
                    {"seconds", seconds}};
        }

        json Runner::cmdWorkerSetup() {
            pyenv::SetupOptions o;
            o.basePython = args_.value("base-python");
            o.extras = args_.has("extras");
            o.extraPackages = args_.values("package");
            if (args_.has("update") && args_.has("recreate")) throw UsageError("--update and --recreate exclude each other");
            o.mode = args_.has("update") ? pyenv::Mode::Update : args_.has("recreate") ? pyenv::Mode::Recreate
                                                                                       : pyenv::Mode::Auto;
            o.useUv = !args_.has("no-uv");
            o.indexUrl = args_.value("index-url");
            o.findLinks = args_.values("find-links");
            o.noIndex = args_.has("no-index");
            o.createdBy = std::string("sirius-cli ") + SIRIUS_VERSION;
            const std::string dir = scriptDir();

            const pyenv::SetupPlan plan = pyenv::planSetup(o, dir);
            for (const std::string& w : plan.warnings) warnings_.push_back(w);
            if (args_.has("dry-run")) return plan.toJson();
            if (!plan.nothingToDo) {
                if (plan.basePython.empty()) {
                    throw CliError("python_not_found", plan.warnings.empty() ? std::string("No Python 3 interpreter was found.") : plan.warnings.front(),
                                   "install Python 3.9 or newer (python.org, winget, apt, brew), or name one with --base-python", plan.toJson());
                }
                std::string size = "size unknown";
                if (plan.approxDownloadBytes > 0) size = "about " + std::to_string((plan.approxDownloadBytes + 999999) / 1000000) + " MB";
                const std::string index = plan.index.empty() ? std::string("pypi.org") : plan.index;
                // A package may be a URL with credentials in it; the prompt and
                // the message show it as the plan's JSON does, redacted.
                std::vector<std::string> packages;
                for (const std::string& p : plan.packages) packages.push_back(pyenv::redactUrl(p));
                const std::string what = (packages.empty() ? std::string("the worker's packages") : join(packages, ", ")) + " (" + size + ")";
                if (!args_.has("yes")) {
                    const bool canAsk = stdinIsTerminal() && stderrIsTerminal();
                    if (!canAsk || !confirm("Download " + what + " from " + index + " into " + plan.envDir + "?")) {
                        throw CliError("consent_required",
                                       canAsk ? "the download was declined"
                                              : "setting up the environment downloads " + what + " from " + index + "; that needs consent",
                                       "run `sirius-cli worker setup --yes` (a person should agree to the download first)", plan.toJson());
                    }
                }
            }
            const pyenv::SetupResult r = pyenv::setup(
                o, dir,
                [this](const std::string& line) {
                    // The setup's own lines ("Python environment: ...", "$ uv ...") as
                    // they are; the installers' output marked as theirs.
                    const bool own = line.compare(0, 18, "Python environment") == 0 || line.compare(0, 2, "$ ") == 0;
                    reporter_.log("python-env", line, !own);
                },
                [this](double fraction, const std::string& message) { reporter_.progress(fraction, message); },
                [this] { return cancelled(); });
            reporter_.finish();
            if (!r.ok) throw setupFailure(r);
            const std::string envDir = r.plan.envDir.empty() ? plan.envDir : r.plan.envDir;
            json packages = json::object();
            std::string baseExecutable = r.plan.basePython, version = r.plan.basePythonVersion, installer = r.plan.installer;
            bool extras = o.extras;
            if (r.marker) {
                for (const auto& p : r.marker->packages) packages[p.first] = p.second;
                if (!r.marker->baseExecutable.empty()) baseExecutable = r.marker->baseExecutable;
                if (!r.marker->pythonVersion.empty()) version = r.marker->pythonVersion;
                if (!r.marker->installer.empty()) installer = r.marker->installer;
                extras = r.marker->extras;
            }
            const char* mode = r.plan.nothingToDo                     ? "none"
                               : r.plan.mode == pyenv::Mode::Update   ? "update"
                               : r.plan.mode == pyenv::Mode::Recreate ? "recreate"
                                                                      : "create";
            return {{"env_dir", pathOrNull(envDir)},
                    {"python", pathOrNull(envDir.empty() ? std::string() : pyenv::environmentPython(envDir))},
                    {"base_python", pathOrNull(baseExecutable)},
                    {"python_version", version},
                    {"installer", installer},
                    {"mode", mode},
                    {"packages", packages},
                    {"extras", extras},
                    {"seconds", r.seconds}};
        }

        json Runner::cmdWorkerRemove() {
            const std::string envDir = pyenv::environmentDirectory();
            const pyenv::EnvironmentStatus status = pyenv::environmentStatus(scriptDir(), false);
            if (status.state == pyenv::State::Absent) {
                warnings_.push_back("there is no environment at " + displayPath(envDir));
                return {{"removed", nullptr}, {"env_dir", pathOrNull(envDir)}};
            }
            if (!args_.has("yes")) {
                const bool canAsk = stdinIsTerminal() && stderrIsTerminal();
                if (!canAsk || !confirm("Remove SIRIUS's Python environment (" + displayPath(envDir) + ")?"))
                    throw CliError("consent_required", canAsk ? "the removal was declined" : "removing the environment needs consent",
                                   "run `sirius-cli worker remove --yes`", {{"env_dir", displayPath(envDir)}});
            }
            const pyenv::SetupResult r = pyenv::remove(envDir);
            if (!r.ok) throw setupFailure(r);
            return {{"removed", displayPath(envDir)}};
        }

        // --- session and MCP ------------------------------------------------------

        int Runner::serve(bool mcp) {
            // stdin is the protocol here.
            if (args_.state.steps == "-") throw UsageError("--steps - would read the protocol's stdin; use --steps @file");
            // Until the server exists, the signals are handled as in a
            // one-shot command: a termination during a long --dataset open
            // cancels it and ends in order (exit 0, scratch removed) rather
            // than killing the process where it stands.
            installOneShotHandlers();
            app::HeadlessWorkbench* opened = nullptr;
            try {
                opened = &workspace();
                applyState();
            } catch (const std::exception&) {
                if (!terminated.load()) throw;
            }
            if (terminated.load()) {
                reporter_.note("terminated before the " + std::string(mcp ? "MCP server" : "session") + " started");
                return 0;
            }
            app::HeadlessWorkbench& ws = *opened;

            agent::ServerOptions so;
            so.version = SIRIUS_VERSION;
            if (!mcp && args_.global.timeoutSeconds)
                so.eofRunWait = std::chrono::milliseconds(static_cast<long long>(*args_.global.timeoutSeconds * 1000.0));
            agent::LineSink sink = [](const std::string& line) { writeLine(line); };
            // Never destroyed (see ServerGate).
            static ServerGate gate;
            agent::SessionServer* session = nullptr;
            if (mcp) {
                agent::McpOptions mo;
                mo.instructions = mcpInstructions();
                gate.server = new agent::McpServer(ws, sink, so, mo);
            } else {
                session = new agent::SessionServer(ws, sink, so);
                gate.server = session;
            }
            installInterruptHandler([] { cancelActiveWorkspace(); });
            installTerminationHandler([] { gate.terminate(); });
            // One that landed between the state options and the line above.
            if (terminated.load()) gate.terminate();
            if (session) session->start();
            startStdinReader([](std::string line) { gate.receive(line); }, [] { gate.endOfInput(); });
            int code = 0;
            try {
                while (gate.server->step(std::chrono::milliseconds(50))) {
                    // A client that closed our stdout is gone: finish as at
                    // the end of input, cancelling what runs.
                    if (outputClosed()) gate.terminate();
                }
                code = gate.server->exitCode();
            } catch (const std::exception& e) {
                reporter_.summary("internal", e.what());
                code = 1;
            }
            gate.close();
            return code;
        }

        // --- the timeout ------------------------------------------------------

        // At the deadline whatever runs is cancelled (the commands also poll
        // it); a command that still has not ended 10 s later -- a full load,
        // say, which cannot be cancelled -- is ended here with exit 124.
        void Runner::startWatchdog() {
            const Clock::time_point deadline = *deadline_;
            const std::string command = command_;
            const bool pretty = pretty_;
            const double seconds = *args_.global.timeoutSeconds;
            std::thread([deadline, command, pretty, seconds] {
                std::this_thread::sleep_until(deadline);
                cancelActiveWorkspace();
                std::this_thread::sleep_until(deadline + std::chrono::seconds(10));
                if (claimDocument()) {
                    std::ostringstream s;
                    s << "timed out after " << seconds << " s, and what was running did not stop";
                    const CliError e("timeout", s.str(), "raise --timeout");
                    writeLine(dump(failureEnvelope(command, e, exitFor("timeout"), {}), pretty));
                    writeError("sirius-cli: timeout: " + e.message + "\n");
                } else {
                    // The command answered but its clean-up hangs; give it as
                    // long again before ending the process anyway.
                    std::this_thread::sleep_until(deadline + std::chrono::seconds(20));
                }
                // exitProcess removes the scratch directory on the way out.
                exitProcess(exitFor("timeout"));
            }).detach();
        }

        // --- dispatch ---------------------------------------------------------------

        std::optional<json> Runner::dispatch() {
            const std::string& c = command_;
            if (c == "version") return cmdVersion();
            if (c == "devices") return call("list_devices", json::object());
            if (c == "info") return cmdInfo();
            if (c == "ops") return cmdOps();
            if (c == "help") return cmdHelp();
            if (c == "validate") return cmdValidate();
            if (c == "run") return cmdRun();
            if (c == "render") return cmdRender();
            if (c == "stats") {
                json a = statsArgs(args_.options);
                a["run"] = !args_.has("no-run");
                defaultToPipelineEnd(a);
                return call("statistics", a);
            }
            if (c == "diagnostics") return cmdDiagnostics();
            if (c == "export") {
                if (!args_.has("out")) throw UsageError("export needs --out <path>");
                json a = exportArgs(args_.options, args_.value("out"));
                a["run"] = !args_.has("no-run");
                defaultToPipelineEnd(a);
                return call("export_result", a);
            }
            if (c == "export-training") return cmdExportTraining();
            if (c == "export-python") {
                if (!args_.has("out")) throw UsageError("export-python needs --out <file.py>");
                return call("export_python", {{"path", args_.value("out")}});
            }
            if (c == "call") return cmdCall();
            if (c == "tools") return cmdTools();
            if (c == "schema") return schemaJson();
            if (c == "worker status") return cmdWorkerStatus();
            if (c == "worker check") return cmdWorkerCheck();
            if (c == "worker setup") return cmdWorkerSetup();
            if (c == "worker remove") return cmdWorkerRemove();
            throw UsageError("unknown command '" + c + "'", "sirius-cli --help lists the commands");
        }

        int Runner::execute() {
            if (args_.help) {
                std::string text = args_.command.empty() ? usageText() : commandHelp(args_.command);
                if (args_.command == "call" && !args_.tool.empty()) {
                    text += toolHelp();
                    closeWorkspace();
                }
                writeRaw(text);
                return 0;
            }
            if (args_.command.empty() && !args_.version) {
                writeRaw(usageText());
                return 0;
            }
            command_ = args_.command.empty() ? std::string("version") : args_.command;
            const std::string& command = command_;
            warnings_ = args_.warnings;
            // The engine: its stdout is the announce line, then nothing.
            if (command == "serve") return serveEngine(args_);
            const bool server = command == "session" || command == "mcp";

            if (server) {
                int code = 0;
                try {
                    code = serve(command == "mcp");
                } catch (const UsageError& e) {
                    reporter_.summary("usage", e.what());
                    code = exitFor("usage");
                } catch (const CliError& e) {
                    // No envelope: stdout is the protocol, and there is none yet.
                    reporter_.summary(e.code, e.message + (e.hint.empty() ? "" : " (" + e.hint + ")"));
                    code = exitFor(e.code);
                } catch (const app::ToolFailure& e) {
                    reporter_.summary(e.code(), std::string(e.what()) + (e.hint().empty() ? "" : " (" + e.hint() + ")"));
                    code = exitFor(e.code());
                } catch (const std::exception& e) {
                    reporter_.summary("internal", e.what());
                    code = exitFor("internal");
                }
                closeWorkspace();
                return code;
            }

            installOneShotHandlers();
            // Setup and removal roll back when cancelled; the watchdog must not
            // end them half way, so --timeout does not apply to them.
            if (args_.global.timeoutSeconds && command != "worker setup" && command != "worker remove") {
                deadline_ = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(*args_.global.timeoutSeconds));
                startWatchdog();
            }

            std::optional<json> result;
            std::optional<CliError> failure;
            try {
                const CommandSpec* spec = findCommand(command);
                checkRenderTargets();
                if (spec && spec->state) applyState();
                result = dispatch();
            } catch (const UsageError& e) {
                failure = CliError("usage", e.what(), e.hint());
            } catch (const CliError& e) {
                failure = e;
            } catch (const app::ToolFailure& e) {
                // the core's shared helpers (and the workspace's constructor) report this way
                failure = CliError(e.code(), e.what(), e.hint(), e.data());
            } catch (const app::WorkerStartError& e) {
                failure = workerUnavailable(e);
            } catch (const std::exception& e) {
                if (app::isCancellation(e)) failure = cancellation();
                else failure = CliError("internal", e.what());
            }
            reporter_.finish();

            int code = 0;
            if (failure) {
                code = exitFor(failure->code);
                if (!rawOutput_ && claimDocument()) {
                    if (replaceNonFinite(failure->data)) warnings_.push_back("NaN or infinite numbers were written as null");
                    writeLine(dump(failureEnvelope(command, *failure, code, warnings_), pretty_));
                }
                reporter_.summary(failure->code, failure->message);
            } else if (result && claimDocument()) {
                if (replaceNonFinite(*result)) warnings_.push_back("NaN or infinite numbers were written as null");
                writeLine(dump(successEnvelope(command, std::move(*result), warnings_), pretty_));
            }
            closeWorkspace();
            return code;
        }

    } // namespace

    int run(const std::vector<std::string>& argv) {
        // The help pages, the worker and the plugins are found relative to
        // the executable (core/app_paths.hpp).
        app::setApplicationDirectory(app::host::executableDirectory());
        app::registerBuiltinOperations();

        Args args;
        try {
            args = parseArgs(argv);
        } catch (const UsageError& e) {
            const CliError error("usage", e.what(), e.hint());
            const int code = exitFor("usage");
            // The command word, when there is one, names the envelope.
            std::string command;
            for (std::size_t i = 0; i < argv.size() && command.empty(); ++i) {
                if (argv[i] == "worker" && i + 1 < argv.size() && findCommand("worker " + argv[i + 1])) command = "worker " + argv[i + 1];
                else if (findCommand(argv[i])) command = argv[i];
            }
            // No envelope for the servers: their stdout is the protocol, and a
            // client would read a line that is not one of its messages.
            const bool server = command == "session" || command == "mcp" || command == "serve";
            if (!server && claimDocument()) writeLine(dump(failureEnvelope(command, error, code, {}), stdoutIsTerminal()));
            writeError("sirius-cli: usage: " + std::string(e.what()) + "\n");
            return code;
        }
        Runner runner(args);
        return runner.execute();
    }

} // namespace sirius::cli
