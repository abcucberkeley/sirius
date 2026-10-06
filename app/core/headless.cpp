#include "core/headless.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <system_error>
#include <thread>
#include <utility>

#include <sirius/device.hpp>

#include "core/cancel.hpp"
#include "core/host.hpp"
#include "core/image_encode.hpp"
#include "core/python_env.hpp"
#include "core/rpc.hpp"
#include "core/statistics.hpp"

// The workbench without a window: its tool table, the run driver that waits
// for or polls runs, and the log and event plumbing the protocol servers read.

namespace sirius::app {

    using json = nlohmann::json;

    namespace {
        namespace fs = std::filesystem;
        using Clock = std::chrono::steady_clock;

        // What a worker that cannot start asks of the user, as the error data
        // says it (section 3.4 of the CLI's reference).
        constexpr const char* kSetupCommand = "sirius-cli worker setup --yes";
        // What the workbench appends to a local worker failure it has no hint for.
        constexpr const char* kLocalWorkerHint =
            "`sirius-cli worker check` shows why the Python worker does not start; --python names another interpreter.";
        constexpr const char* kMiddot = " \xC2\xB7 ";   // " · "
        constexpr const char* kArrow = " \xE2\x86\x92 ";  // " → "

        // How each tool of the table behaves, in the order tools() lists them.
        struct ToolTraits {
            const char* name;
            const char* title;
            bool whileRunning;          // answered during a run; every other tool says "busy"
            bool readOnly, destructive, idempotent, openWorld;
            bool longResult;            // MCP clients may keep up to 200 000 characters of it
            bool userInteraction;       // a person has to agree first
        };

        const std::vector<ToolTraits>& toolTraits() {
            static const std::vector<ToolTraits> table = {
                // dataset and workspace
                {"open_dataset", "Open a dataset", false, false, false, false, false, false, false},
                {"dataset_info", "Describe a dataset file", true, true, false, false, false, false, false},
                {"load_pipeline", "Load a pipeline", false, false, false, false, false, false, false},
                {"save_pipeline", "Save the pipeline", true, false, true, false, false, false, false},
                {"clear_pipeline", "Clear the pipeline", false, false, false, false, false, false, false},
                {"get_state", "Workspace state", true, true, false, false, false, false, false},
                // pipeline editing
                {"add_step", "Add a step", false, false, false, false, false, false, false},
                {"remove_step", "Remove a step", false, false, false, false, false, false, false},
                {"move_step", "Move a step", false, false, false, false, false, false, false},
                {"set_step_enabled", "Enable or skip a step", false, false, false, true, false, false, false},
                {"set_params", "Set step parameters", false, false, false, true, false, false, false},
                {"apply_preset", "Apply a preset", false, false, false, true, false, false, false},
                {"set_cache", "Set a step's cache policy", false, false, false, true, false, false, false},
                {"undo", "Undo", false, false, false, false, false, false, false},
                {"redo", "Redo", false, false, false, false, false, false, false},
                {"load_example_pipeline", "Load the example pipeline", false, false, false, false, false, false, false},
                {"get_step", "Describe a step", true, true, false, false, false, false, false},
                // operations and help
                {"list_operations", "List operations", true, true, false, false, false, true, false},
                {"describe_operation", "Describe an operation", true, true, false, false, false, true, false},
                {"get_help", "Help pages", true, true, false, false, false, true, false},
                // running
                {"validate", "Validate the pipeline", true, true, false, false, false, true, false},
                // open-world: a step may download model weights from Hugging Face, and the hpc backend sends the data to a remote worker
                {"run", "Run the pipeline", false, false, false, false, true, false, false},
                {"run_status", "Run status", true, true, false, false, false, false, false},
                {"cancel_run", "Cancel the run", true, false, false, false, false, false, false},
                {"set_backend", "Set the compute backend", true, false, false, true, false, false, false},
                {"list_devices", "List compute devices", true, true, false, false, false, false, false},
                // inspecting
                {"render", "Render a view", false, true, false, false, false, false, false},
                {"render_diagnostics", "Render a diagnostics image", false, true, false, false, false, false, false},
                {"probe", "Probe a voxel", false, true, false, false, false, false, false},
                {"statistics", "Intensity statistics", false, true, false, false, false, false, false},
                {"get_diagnostics", "Step diagnostics", false, true, false, false, false, false, false},
                {"list_tracks", "List tracks", false, true, false, false, false, false, false},
                {"list_labels", "List labels", false, true, false, false, false, true, false},
                {"get_log", "Workbench log", true, true, false, false, false, true, false},
                // output
                {"export_result", "Export a result", false, false, true, false, false, false, false},
                {"export_training_data", "Export training data", false, false, true, false, false, false, false},
                {"export_labels", "Export labels", false, false, true, false, false, false, false},
                // label editing (the viewer's paint tools, by step and time point)
                {"paint_label", "Paint a label", false, false, false, false, false, false, false},
                {"fill_label", "Fill a label", false, false, false, false, false, false, false},
                {"merge_labels", "Merge labels", false, false, false, false, false, false, false},
                {"split_label", "Split a label", false, false, false, false, false, false, false},
                {"delete_label", "Delete a label", false, false, false, false, false, false, false},
                {"clear_labels", "Clear the labels", false, false, false, false, false, false, false},
                {"set_label_reviewed", "Mark a label reviewed", false, false, false, true, false, false, false},
                {"export_python", "Export a Python script", true, false, true, false, false, false, false},
                // the Python worker and plugins
                {"list_plugins", "List plugins", false, true, false, false, false, false, false},
                {"worker_status", "Python worker status", true, true, false, false, false, false, false},
                {"setup_worker_env", "Set up SIRIUS's Python environment", false, false, true, false, true, false, true},
            };
            return table;
        }

        // The view tools only move what a window shows; every headless tool
        // takes explicit steps and coordinates instead.
        const char* const kViewTools[] = {"view_step", "select_step", "set_view", "focus_track"};

        [[noreturn]] void invalid(const std::string& message, const std::string& hint = {}, json data = nullptr) {
            throw ToolFailure("invalid_argument", message, hint, std::move(data));
        }

        std::string lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        }

        // --- arguments ---------------------------------------------------------------

        bool has(const json& a, const char* key) { return a.contains(key) && !a[key].is_null(); }

        std::string stringArg(const json& a, const char* key, const std::string& def = {}) {
            if (!has(a, key)) return def;
            if (!a[key].is_string()) invalid(std::string("'") + key + "' must be a string");
            return a[key].get<std::string>();
        }

        std::string requiredString(const json& a, const char* key) {
            if (!has(a, key) || !a[key].is_string() || a[key].get<std::string>().empty())
                invalid(std::string("'") + key + "' is required (a non-empty string)");
            return a[key].get<std::string>();
        }

        bool boolArg(const json& a, const char* key, bool def) {
            if (!has(a, key)) return def;
            if (!a[key].is_boolean()) invalid(std::string("'") + key + "' must be true or false");
            return a[key].get<bool>();
        }

        std::int64_t integerArg(const json& a, const char* key, std::int64_t def, std::int64_t min, std::int64_t max) {
            if (!has(a, key)) return def;
            const json& v = a[key];
            if (!v.is_number() || (!v.is_number_integer() && v.get<double>() != std::floor(v.get<double>())))
                invalid(std::string("'") + key + "' must be an integer");
            const std::int64_t i = v.is_number_integer() ? v.get<std::int64_t>() : static_cast<std::int64_t>(v.get<double>());
            if (i < min || i > max)
                invalid(std::string("'") + key + "' must be within " + std::to_string(min) + ".." + std::to_string(max));
            return i;
        }

        double numberArg(const json& a, const char* key, double def) {
            if (!has(a, key)) return def;
            if (!a[key].is_number()) invalid(std::string("'") + key + "' must be a number");
            return a[key].get<double>();
        }

        // The absolute path, with forward slashes: what every reply reports.
        std::string reported(const fs::path& p) {
            std::error_code ec;
            const fs::path abs = fs::absolute(p, ec);
            return (ec ? p : abs).lexically_normal().generic_u8string();
        }

        std::string reported(const std::string& path) {
            if (path.empty()) return path;
            try {
                return reported(fs::u8path(path));
            } catch (const std::exception&) {
                return path;
            }
        }

        // A path argument, made absolute against the working directory. A
        // network path is refused unless the server was started with
        // --allow-network-paths: opening \\server\share connects to that
        // server with the user's Windows credentials.
        std::string pathArg(const json& a, const char* key, bool allowNetwork) {
            const std::string given = requiredString(a, key);
            if (!allowNetwork && isNetworkPath(given))
                invalid(std::string("'") + key + "' is a network path (" + given + "), which this server does not open",
                        "use a local path, or ask the user to restart sirius-cli with --allow-network-paths");
            try {
                return reported(fs::u8path(given));
            } catch (const std::exception&) {
                invalid(std::string("'") + key + "' is not a usable path");
            }
        }

        std::string existingPath(const json& a, const char* key, bool allowNetwork) {
            const std::string path = pathArg(a, key, allowNetwork);
            std::error_code ec;
            if (!fs::exists(fs::u8path(path), ec))
                throw ToolFailure("not_found", "no such file or directory: " + path,
                                  "relative paths resolve against " + reported(fs::current_path(ec)) + "; prefer absolute paths");
            return path;
        }

        // --- the tool schemas --------------------------------------------------------

        json schema(json properties = json::object(), std::vector<std::string> required = {}) {
            json s = {{"type", "object"}, {"properties", std::move(properties)}, {"additionalProperties", false}};
            if (!required.empty()) s["required"] = std::move(required);
            return s;
        }

        json prop(const char* type, const char* description) { return {{"type", type}, {"description", description}}; }

        json stepProp(const char* description = "Step number (1 = Load) or a step's name or kind") {
            return {{"type", json::array({"integer", "string"})}, {"description", description}};
        }

        json intList(const char* description) { return {{"type", "array"}, {"items", {{"type", "integer"}}}, {"description", description}}; }

        // The tools accept these values in any case, so they are named in the
        // description rather than as a JSON Schema enum, which a client that
        // validates arguments would hold to the lowercase spelling.
        json enumProp(const std::vector<std::string>& values, const char* description) {
            std::string names;
            for (const std::string& v : values) names += (names.empty() ? "" : ", ") + v;
            return {{"type", "string"}, {"description", std::string(description) + ". One of: " + names + " (any case)"}};
        }

        // The options a dataset opens with, shared by open_dataset and dataset_info.
        json openProps() {
            return {{"page_order", prop("string", "Plain multi-page TIFF: the order of the pages, fastest axis first, such as \"czt\"")},
                    {"c", prop("integer", "Plain TIFF: number of channels in the page order")},
                    {"t", prop("integer", "Plain TIFF: number of time points in the page order")},
                    {"z", prop("integer", "Plain TIFF: number of z planes (0 = from the page count)")},
                    {"voxel_um", {{"type", "array"}, {"items", {{"type", "number"}}}, {"description", "Voxel size [x, y, z] in micrometres, over what the file says"}}},
                    {"sim", {{"type", json::array({"object", "boolean"})}, {"description", "Raw SIM layout {ndirs, nphases, fast}, or false for none"}}},
                    {"tile", prop("integer", "Multi-file datasets: the tile to open")},
                    {"full_load", prop("boolean", "Read everything into memory now (default false: planes are read when needed)")}};
        }

        std::string backendName(Backend b) { return lower(toString(b)); }

        json cudaDeviceJson(int device) { return device < 0 ? json("all") : json(device); }
        std::string hpcDeviceName(HpcDevice d) { return d == HpcDevice::Cpu ? "cpu" : "gpu"; }

        json dimsJson(const Dims5& d) { return {{"c", d.c}, {"t", d.t}, {"z", d.z}, {"y", d.y}, {"x", d.x}}; }

        // ToolApi::stepJson carries the GUI's selection, which a headless
        // workspace does not have, and the step's number as the GUI prints it
        // ("04"), next to `step`, the same number as an integer.
        void scrubSteps(json& v) {
            if (v.is_array()) {
                for (json& e : v) scrubSteps(e);
                return;
            }
            if (!v.is_object()) return;
            if (v.contains("kind") && v.contains("number") && v.contains("params")) {
                v.erase("selected");
                v.erase("viewed");
                v.erase("number");
            }
            if (v.contains("steps")) scrubSteps(v["steps"]);
        }

        std::string randomHex(int digits) {
            std::random_device rd;
            std::mt19937_64 gen((static_cast<std::uint64_t>(rd()) << 32) ^ rd() ^
                                static_cast<std::uint64_t>(Clock::now().time_since_epoch().count()));
            static const char* hex = "0123456789abcdef";
            std::string s;
            for (int i = 0; i < digits; ++i) s += hex[gen() & 15u];
            return s;
        }

        // "0.1", "50", "99.9": a percentile as a key.
        std::string percentileKey(double p) {
            char buf[32];
            std::snprintf(buf, sizeof buf, "%g", p);
            return buf;
        }

        const char* stateName(StepReport::State s) {
            switch (s) {
                case StepReport::State::Ran: return "ran";
                case StepReport::State::Cached: return "cached";
                case StepReport::State::Skipped: return "skipped";
                case StepReport::State::Failed: return "failed";
                case StepReport::State::Running: return "running";
            }
            return "ran";
        }

        const char* modeName(pyenv::Mode m) {
            switch (m) {
                case pyenv::Mode::Auto: return "auto";
                case pyenv::Mode::Create: return "create";
                case pyenv::Mode::Update: return "update";
                case pyenv::Mode::Recreate: return "recreate";
            }
            return "create";
        }

        // "Step 02 · SIM reconstruction · phase 3 of 5": where a run is.
        std::string progressMessage(RunJob& job) {
            const int s = job.progress().stepIndex.load();
            const std::string m = job.progress().messageCopy();
            if (s < 0 || s >= job.pipeline().size()) return m.empty() ? std::string("Preparing") : m;
            std::string out = "Step " + Step::number(s) + kMiddot + job.pipeline().at(s).name;
            if (!m.empty()) out += kMiddot + m;
            return out;
        }

        json lastLines(const std::vector<std::string>& lines, std::size_t n) {
            json out = json::array();
            for (std::size_t i = lines.size() > n ? lines.size() - n : 0; i < lines.size(); ++i) out.push_back(lines[i]);
            return out;
        }

        // The error data of a worker that did not start (section 3.4): what it
        // was, which Python, and the command that sets one up.
        json workerErrorData(const WorkerStartError& e) {
            json d = e.toJson();
            d["python"] = e.interpreter;
            d["fix"] = e.setupWouldHelp() ? json(kSetupCommand) : json(nullptr);
            return d;
        }
    } // namespace

    // --- the state -----------------------------------------------------------------------

    struct HeadlessWorkbench::Impl {
        explicit Impl(HeadlessOptions o);
        ~Impl();
        Impl(const Impl&) = delete;
        Impl& operator=(const Impl&) = delete;

        HeadlessOptions options;
        // The worker's stderr lines, from its reader thread, until pump() logs
        // them. Declared before the worker, which may still write here while
        // it is being destroyed.
        std::mutex workerLogMutex;
        std::deque<std::string> workerLog;
        // Declared before the workbench: a run's thread may be inside
        // connect() until the destructor has joined it.
        LocalWorker worker;
        Workbench wb;
        ToolApi api;
        display::DisplayModel model;
        std::string workspace;

        // Every workbench log line, on the main thread (Workbench::Observer::logged),
        // and every dataset the workbench installs.
        struct LogObserver final : Workbench::Observer {
            Impl* impl = nullptr;
            void logged(const std::string& line) override { impl->onLogged(line); }
            void datasetChanged() override { ++impl->datasetChanges; }
        } observer;
        std::uint64_t datasetChanges = 0;

        // The call in progress (main thread only): what the tool functions,
        // which ToolApi calls as json(const json&), report through.
        const agent::CallContext* ctx = nullptr;
        std::vector<agent::Attachment> images;
        std::vector<std::string> warnings;
        int renderCounter = 0;

        // What other threads read or set: status(), cancelActive().
        mutable std::mutex stateMutex;
        std::shared_ptr<RunJob> statusJob;
        std::string statusRunId;
        std::string activeTool;
        std::atomic<bool> workerCancel{false};      // ends a worker start (D33)
        std::atomic<bool> cancelRequested{false};   // ends the synchronous tool in progress
        std::atomic<bool> loadingPlugins{false};    // ensurePlugins is on the main thread, in a call

        // Main thread.
        std::vector<json> events;
        bool forwardingWorkerLine = false;
        bool capturingRun = false;
        std::vector<std::string> runLog;
        struct ActiveRun {
            std::shared_ptr<RunJob> job;
            std::thread thread;
            std::string id;
            int target = 0;
            StepId targetId = 0;
            std::string backend;
            bool announce = false;   // its run call returned "running": a run_finished event follows
            Clock::time_point started;
        };
        std::unique_ptr<ActiveRun> run;
        json lastOutcome;            // null until a run finished
        std::optional<WorkerStartError> lastWorkerFailure;
        StepId lastRunTarget = 0;    // the target of the last run that succeeded
        int runCounter = 0;
        bool pluginsAttempted = false, pluginsLoaded = false;
        OpenResult lastOpened;       // its summary fields only (no source)
        std::string lastOpenedPath;

        // --- plumbing ------------------------------------------------------------------
        void onLogged(const std::string& line);
        void emitLog(const std::string& source, const std::string& line);
        void pushEvent(json e);
        void drainWorkerLog();
        void syncWorkerDevice();
        bool cancelled() const { return (ctx && ctx->cancelled && ctx->cancelled()) || cancelRequested.load(); }
        void progress(double fraction, const std::string& message) const {
            if (ctx && ctx->progress) ctx->progress(fraction, message);
        }
        std::function<bool()> cancelFn() {
            return [this] { return cancelled(); };
        }
        std::function<void(double, const std::string&)> progressFn(std::string fallback) {
            return [this, fallback](double f, const std::string& m) { progress(f, m.empty() ? fallback : m); };
        }
        std::string scriptDir() const { return worker.scriptDir(); }   // --worker-dir, else where the worker ships
        // What to look at when the HPC endpoint does not answer or refuses the
        // token. It is not the local Python, which `worker setup` would set up.
        std::string hpcHint(bool sentence) const {
            const std::string where = options.hpc ? options.hpc->host + ":" + std::to_string(options.hpc->port) : std::string("--hpc");
            return std::string(sentence ? "Check" : "check") + " that the HPC worker at " + where +
                   " is running and takes the token this server was given ($SIRIUS_HPC_TOKEN); set_backend cpu runs the steps here instead" +
                   (sentence ? "." : "");
        }
        bool ensurePlugins(bool reload);
        bool hidden(const ToolSpec& spec) const { return options.readOnly && spec.destructive; }
        agent::ToolDescriptor descriptorOf(const ToolSpec& spec) const;
        void addTool(const char* name, std::string description, json parameters, std::function<json(const json&)> fn);
        void installTools();

        // --- steps and outputs ---------------------------------------------------------
        int resolveStep(const json& a) const { return ToolApi::resolveStepIndex(wb.pipeline(), a); }
        int defaultInspectStep() const;
        int inspectStep(const json& a) const { return has(a, "step") ? resolveStep(a) : defaultInspectStep(); }
        // The last step that is not skipped (0, Load, at least): a skipped
        // step at the end would only pass its input through.
        int lastEnabledStep() const {
            int last = 0;
            for (int i = 1; i < wb.pipeline().size(); ++i)
                if (wb.pipeline().at(i).enabled) last = i;
            return last;
        }
        std::shared_ptr<const StepOutput> outputFor(int index, bool runIfNeeded);
        json stepsJson() const;
        json datasetJson() const;
        void adopt(OpenResult opened, const std::string& path, OpenOptions options);
        void attach(const RenderResult& r, json& caption);

        // --- runs -------------------------------------------------------------------------
        void startRun(int target, bool force);
        ToolFailure refusal(int target) const;
        // The validation refusal createRun would give for the steps up to `target`, if any.
        std::optional<ToolFailure> firstInvalidStep(int target) const;
        json waitRun(double waitSeconds, bool cancelOnRequest);
        json finishActiveRun();
        json runningOutcome() const;
        json outcomeOf(const ActiveRun& r) const;
        void raiseForOutcome(const json& outcome) const;
        json runSync(int target);

        // --- the worker ---------------------------------------------------------------------
        json workerInterpreterJson() const;
        json workerStatusJson() const;

        // --- the tools that are not ToolApi's -------------------------------------------------
        json openDatasetTool(const json& a);
        json datasetInfoTool(const json& a);
        json loadPipelineTool(const json& a);
        json getStateTool(const json& a);
        json listOperationsTool(const json& a);
        json describeOperationTool(const json& a);
        json getHelpTool(const json& a);
        json validateTool(const json& a);
        json runTool(const json& a);
        json setBackendTool(const json& a);
        json listDevicesTool(const json& a);
        json renderTool(const json& a);
        json renderDiagnosticsTool(const json& a);
        json probeTool(const json& a);
        json statisticsTool(const json& a);
        json exportResultTool(const json& a);
        json exportPythonTool(const json& a);
        json listPluginsTool(const json& a);
        json workerStatusTool(const json& a);
        json setupWorkerEnvTool(const json& a);
    };

    HeadlessWorkbench::Impl::Impl(HeadlessOptions o) : options(std::move(o)), wb(options.scratchDir), api(wb), workspace("ws_" + randomHex(12)) {
        // A predictable start (D19): the Load step alone, nothing to undo. The
        // workbench's default Contrast step is for a person looking at data.
        wb.replacePipeline(Pipeline(), "Start");
        wb.history().clear();
        api.setAllowNetworkPaths(options.allowNetworkPaths);

        const std::string backend = lower(options.backend);
        if (backend == "cpu") {
            wb.setBackend(Backend::Cpu);
        } else if (backend == "cuda") {
            if (!cudaAvailable()) throw ToolFailure("unsupported", "no CUDA device is available on this computer", "use --backend cpu, or auto");
            wb.setBackend(Backend::Cuda);
        } else if (backend == "hpc") {
            if (!options.hpc) throw ToolFailure("invalid_argument", "the hpc backend needs the endpoint to run on", "start sirius-cli with --hpc host:port");
            wb.setBackend(Backend::Hpc);
        } else if (!backend.empty() && backend != "auto") {
            throw ToolFailure("invalid_argument", "the backend must be auto, cpu, cuda or hpc, not '" + options.backend + "'");
        }
        // auto: the workbench starts on CUDA when there is a GPU, else on the CPU
        if (options.hpc) wb.setRemoteConfig(*options.hpc);
        wb.setCudaDevice(options.cudaDevice);
        if (const auto d = hpcDeviceFromString(options.hpcDevice)) wb.setHpcDevice(*d);
        else throw ToolFailure("invalid_argument", "the HPC device must be gpu or cpu, not '" + options.hpcDevice + "'");

        worker.setPython(options.python);
        worker.setScriptDir(options.workerDir);
        worker.setSetupHint(options.setupHint);
        worker.setLogHandler([this](const std::string& line) {
            const std::lock_guard<std::mutex> g(workerLogMutex);
            workerLog.push_back(line);
            if (workerLog.size() > 4000) workerLog.pop_front();
        });
        syncWorkerDevice();
        // D33: a cancel ends the wait for the worker's port line, not only the run.
        // A plugin load runs on this thread inside a tool call, so that call's own
        // cancel (a cancelled request) may end it too; a run's thread reads only the flag.
        wb.setLocalWorkerLauncher([this] {
            return worker.connect([this] { return workerCancel.load() || (loadingPlugins.load() && cancelled()); });
        });
        wb.setWorkerHint(kLocalWorkerHint);
        wb.setHubTokenProvider([token = options.hubToken] { return token; });

        observer.impl = this;
        wb.addObserver(&observer);
        // No destructor runs for a constructor that throws, and the observer
        // goes before the workbench does: the workbench must not keep it.
        try {
            if (!options.recordPath.empty()) {
                try {
                    wb.startRecording(options.recordPath);
                } catch (const std::exception& e) {
                    throw ToolFailure("io_error", "cannot record to " + options.recordPath + ": " + e.what());
                }
            }
            installTools();
            if (options.plugins == HeadlessOptions::Plugins::On) ensurePlugins(false);
        } catch (...) {
            wb.removeObserver(&observer);
            worker.setLogHandler({});
            throw;
        }
    }

    HeadlessWorkbench::Impl::~Impl() {
        // Nothing reaches the host's sink once it is going away.
        wb.removeObserver(&observer);
        worker.setLogHandler({});
        if (run) {
            run->job->cancel();
            workerCancel = true;
            if (run->thread.joinable()) run->thread.join();
            {
                const std::lock_guard<std::mutex> g(stateMutex);
                statusJob.reset();
            }
            wb.finishRun(run->job);
            run.reset();
        }
        worker.stop();
        if (wb.recording()) wb.stopRecording();
    }

    // --- plumbing ------------------------------------------------------------------------

    void HeadlessWorkbench::Impl::onLogged(const std::string& line) {
        if (capturingRun) {
            runLog.push_back(line);
            if (runLog.size() > 1000) runLog.erase(runLog.begin(), runLog.begin() + 500);
        }
        // a worker line is forwarded as itself by drainWorkerLog, not as "worker: ..."
        if (!forwardingWorkerLine) emitLog("workbench", line);
    }

    void HeadlessWorkbench::Impl::emitLog(const std::string& source, const std::string& line) {
        if (options.logSink) {
            try {
                options.logSink(source, line);
            } catch (const std::exception&) {
                // a sink that cannot write (a closed stderr) loses the line, nothing else
            }
        }
        pushEvent({{"event", "log"}, {"source", source}, {"line", line}});
    }

    void HeadlessWorkbench::Impl::pushEvent(json e) {
        // Nobody may be taking them (a one-shot command): the oldest go first.
        if (events.size() >= 10000) events.erase(events.begin(), events.begin() + 1000);
        events.push_back(std::move(e));
    }

    void HeadlessWorkbench::Impl::drainWorkerLog() {
        std::deque<std::string> lines;
        {
            const std::lock_guard<std::mutex> g(workerLogMutex);
            lines.swap(workerLog);
        }
        for (const std::string& line : lines) {
            forwardingWorkerLine = true;
            wb.logLine("worker: " + line);
            forwardingWorkerLine = false;
            emitLog("worker", line);
        }
    }

    void HeadlessWorkbench::Impl::syncWorkerDevice() {
        if (wb.backend() != Backend::Cuda) worker.setDevice("cpu");
        else if (wb.cudaDevice() < 0) worker.setDevice("cuda");
        else worker.setDevice("cuda:" + std::to_string(wb.cudaDevice()));
    }

    bool HeadlessWorkbench::Impl::ensurePlugins(bool reload) {
        // D20: never with --plugins off, and never during a run (the registry is in use).
        if (options.plugins == HeadlessOptions::Plugins::Off || wb.running()) return false;
        // A failed load is remembered too: each attempt may start Python and wait
        // for it, so only list_plugins(reload), set_backend and setup_worker_env,
        // which may have changed the answer, try again.
        if (pluginsAttempted && !reload) return pluginsLoaded;
        workerCancel = false;
        // The launcher reads the call's cancel while this is set (see the constructor).
        struct Loading {
            std::atomic<bool>& flag;
            explicit Loading(std::atomic<bool>& f) : flag(f) { flag = true; }
            ~Loading() { flag = false; }
            Loading(const Loading&) = delete;
            Loading& operator=(const Loading&) = delete;
        };
        {
            const Loading loading(loadingPlugins);
            wb.loadPlugins(reload);
        }
        drainWorkerLog();
        pluginsAttempted = true;
        pluginsLoaded = wb.pluginError().empty() && !wb.pluginWorkerFailure();
        // A load the call's cancel ended is the call's end, not a worker that is
        // unavailable; the next call tries again.
        if (!pluginsLoaded && cancelled()) {
            pluginsAttempted = false;
            throw ToolFailure("cancelled", "the plugins were not loaded: the call was cancelled");
        }
        return pluginsLoaded;
    }

    agent::ToolDescriptor HeadlessWorkbench::Impl::descriptorOf(const ToolSpec& spec) const {
        agent::ToolDescriptor d;
        d.name = spec.name;
        d.title = spec.title;
        d.description = spec.description;
        json s = spec.parameters.is_object() ? spec.parameters : json::object();
        s["type"] = "object";
        if (!s.contains("properties") || !s["properties"].is_object()) s["properties"] = json::object();
        s["properties"]["workspace"] = {{"type", "string"},
                                        {"description", "The workspace id get_state returns; a call meant for another workspace (the server restarted) is refused"}};
        s["additionalProperties"] = false;
        d.inputSchema = std::move(s);
        d.hints = agent::ToolHints{spec.readOnly, spec.destructive, spec.idempotent, spec.openWorld};
        d.meta = spec.meta.is_object() ? spec.meta : json::object();
        return d;
    }

    void HeadlessWorkbench::Impl::addTool(const char* name, std::string description, json parameters, std::function<json(const json&)> fn) {
        ToolSpec spec;
        spec.name = name;
        spec.description = std::move(description);
        spec.parameters = std::move(parameters);
        spec.fn = std::move(fn);
        api.addTool(std::move(spec));
    }

    // --- steps and outputs ------------------------------------------------------------

    int HeadlessWorkbench::Impl::defaultInspectStep() const {
        // the target of the last run while it has an output, else the last
        // enabled step that has one, else Load
        const Pipeline& p = wb.pipeline();
        if (lastRunTarget != 0) {
            const int i = p.indexOf(lastRunTarget);
            if (i >= 0 && wb.output(i)) return i;
        }
        for (int i = p.size() - 1; i > 0; --i)
            if (p.at(i).enabled && wb.output(i)) return i;
        return 0;
    }

    std::shared_ptr<const StepOutput> HeadlessWorkbench::Impl::outputFor(int index, bool runIfNeeded) {
        if (!wb.hasDataset()) throw ToolFailure("no_dataset", "no dataset is open", "open_dataset or load_pipeline first");
        const std::string which = "step " + std::to_string(index + 1) + " (" + wb.pipeline().at(index).name + ")";
        std::shared_ptr<const StepOutput> out = wb.output(index);
        const bool usable = out && (out->array || out->source);
        if (!usable || (runIfNeeded && !wb.outputFresh(index))) {
            if (!runIfNeeded) throw ToolFailure("not_computed", which + " has not been computed", "run it first, or pass run:true");
            runSync(index);
            out = wb.output(index);
            if (!out || !(out->array || out->source)) throw ToolFailure("not_computed", which + " produced no output", "get_log says what the run did");
        } else if (!wb.outputFresh(index)) {
            warnings.push_back(which + "'s output is stale: its parameters or an earlier step changed since it ran (run it again, or pass run:true)");
        }
        return out;
    }

    json HeadlessWorkbench::Impl::stepsJson() const {
        json steps = json::array();
        for (int i = 0; i < wb.pipeline().size(); ++i) steps.push_back(api.stepJson(i));
        scrubSteps(steps);
        return steps;
    }

    json HeadlessWorkbench::Impl::datasetJson() const {
        if (!wb.hasDataset()) return nullptr;
        const bool same = !lastOpenedPath.empty() && lastOpenedPath == reported(wb.dataset().sourcePath);
        return datasetInfo(wb.dataset(), same ? &lastOpened : nullptr);
    }

    void HeadlessWorkbench::Impl::adopt(OpenResult opened, const std::string& path, OpenOptions openOptions) {
        // What the open said about the file, kept for DatasetInfo (the source
        // itself goes to the workbench).
        OpenResult summary;
        summary.meta = opened.meta;
        summary.metadataSummary = opened.metadataSummary;
        summary.dimsFromMetadata = opened.dimsFromMetadata;
        summary.fullLoadSkipped = opened.fullLoadSkipped;
        openOptions.progress = {};
        wb.adoptDataset(std::move(opened), path, openOptions);
        lastOpened = std::move(summary);
        lastOpenedPath = reported(wb.dataset().sourcePath);
        lastRunTarget = 0;
        // the display model would keep the previous dataset's output alive
        model.setOutput(nullptr);
    }

    void HeadlessWorkbench::Impl::attach(const RenderResult& r, json& caption) {
        // The copy a client can open later: <scratch>/renders/render-NNNN.<ext>,
        // valid until the server ends.
        char name[48];
        std::snprintf(name, sizeof name, "render-%04d.%s", ++renderCounter, r.mimeType == "image/jpeg" ? "jpg" : "png");
        const fs::path file = options.scratchDir / "renders" / name;
        const std::string path = reported(file);
        std::string error;
        if (!writeBinaryFile(path, r.bytes, &error)) throw ToolFailure("io_error", "cannot write the image to " + path + ": " + error);
        caption["path"] = path;
        agent::Attachment a;
        a.mimeType = r.mimeType;
        a.bytes = r.bytes;
        a.path = path;
        a.width = r.width;
        a.height = r.height;
        images.push_back(std::move(a));
    }

    // --- runs (section 3.8) ---------------------------------------------------------------

    void HeadlessWorkbench::Impl::startRun(int target, bool force) {
        if (run) throw ToolFailure("busy", "a run is in progress", "wait for it with run_status, or cancel_run");
        if (!wb.hasDataset()) throw ToolFailure("no_dataset", "no dataset is open", "open_dataset or load_pipeline first");
        // a step whose plugin is not loaded yet (D20)
        for (int i = 1; i <= target && i < wb.pipeline().size(); ++i)
            if (wb.pipeline().at(i).enabled && wb.pipeline().at(i).op().info().missing) {
                ensurePlugins(false);
                break;
            }
        // A forced run drops every cached output, so it first passes the checks
        // createRun makes: a refused run must not have cost the caches.
        if (force) {
            if (std::optional<ToolFailure> invalidStep = firstInvalidStep(target)) throw *invalidStep;
            wb.clearAllCaches();
        }
        capturingRun = true;
        runLog.clear();
        // The job keeps the hint it is made with: an HPC endpoint that does not
        // answer is not something `worker check` would explain.
        wb.setWorkerHint(wb.backend() == Backend::Hpc ? hpcHint(true) : std::string(kLocalWorkerHint));
        std::shared_ptr<RunJob> job = wb.createRun(target);
        wb.setWorkerHint(kLocalWorkerHint);
        if (!job) {
            capturingRun = false;
            throw refusal(target);
        }
        auto r = std::make_unique<ActiveRun>();
        r->job = job;
        r->id = "r" + std::to_string(++runCounter);
        r->target = job->target();
        r->targetId = wb.pipeline().at(job->target()).id;
        r->backend = backendName(wb.backend());
        r->started = Clock::now();
        workerCancel = false;
        {
            const std::lock_guard<std::mutex> g(stateMutex);
            statusJob = job;
            statusRunId = r->id;
        }
        try {
            r->thread = std::thread([job] { job->execute(); });
        } catch (const std::exception& e) {
            {
                const std::lock_guard<std::mutex> g(stateMutex);
                statusJob.reset();
                statusRunId.clear();
            }
            wb.finishRun(job);   // never executed: logged as abandoned
            capturingRun = false;
            throw ToolFailure("failed", std::string("the run could not start: ") + e.what());
        }
        run = std::move(r);
    }

    std::optional<ToolFailure> HeadlessWorkbench::Impl::firstInvalidStep(int target) const {
        const int last = target < 0 || target >= wb.pipeline().size() ? wb.pipeline().size() - 1 : target;
        for (int i = 1; i <= last; ++i) {
            if (!wb.pipeline().at(i).enabled) continue;
            const Validation v = wb.stepValidation(i);
            if (!v.ok())
                return ToolFailure("validation", "step " + std::to_string(i + 1) + " " + wb.pipeline().at(i).name + " cannot run: " + v.firstError(),
                                   "validate lists every step's errors; set_params fixes them", {{"step", i + 1}, {"errors", v.errors}});
        }
        return std::nullopt;
    }

    ToolFailure HeadlessWorkbench::Impl::refusal(int target) const {
        const RunRefusal& r = wb.lastRunRefusal();
        switch (r.kind) {
            case RunRefusal::Kind::Running: return ToolFailure("busy", r.message, "wait for the run with run_status, or cancel_run");
            case RunRefusal::Kind::NoDataset: return ToolFailure("no_dataset", "no dataset is open", "open_dataset or load_pipeline first");
            case RunRefusal::Kind::Invalid:
                return ToolFailure("validation", r.message, "validate lists every step's errors; set_params fixes them",
                                   {{"step", r.step + 1}, {"errors", wb.stepValidation(r.step).errors}});
            case RunRefusal::Kind::NoLauncher: {
                // the data every other worker_unavailable answer carries (section 3.4)
                const json in = workerInterpreterJson();
                return ToolFailure("worker_unavailable", r.message, options.setupHint,
                                   {{"kind", "no_launcher"},
                                    {"message", r.message},
                                    {"python", in["path"] == "" ? json(nullptr) : in["path"]},
                                    {"source", in["source"]},
                                    {"fix", kSetupCommand}});
            }
            case RunRefusal::Kind::NoEngine:
                return ToolFailure("no_engine", r.message, "set_backend CPU or CUDA runs it on this computer", {{"step", r.step + 1}});
            case RunRefusal::Kind::EngineMismatch: return ToolFailure("engine_mismatch", r.message, "connect to an engine built from this SIRIUS");
            case RunRefusal::Kind::NeedsUpload: {
                json files = json::array();
                for (const UploadFile& f : r.uploads) files.push_back({{"path", f.path}, {"bytes", f.bytes}});
                return ToolFailure("needs_upload", r.message, "open the data from the cluster (cluster://...), or set_backend CPU or CUDA",
                                   {{"files", files}});
            }
            case RunRefusal::Kind::None: break;
        }
        // The workbench did not say why: find the reason the way createRun does.
        if (!wb.hasDataset()) return ToolFailure("no_dataset", "no dataset is open", "open_dataset or load_pipeline first");
        if (std::optional<ToolFailure> invalidStep = firstInvalidStep(target)) return *invalidStep;
        const std::string why = wb.log().empty() ? std::string("the run could not start") : wb.log().back();
        return ToolFailure("failed", why, "get_log says more");
    }

    json HeadlessWorkbench::Impl::waitRun(double waitSeconds, bool cancelOnRequest) {
        const Clock::time_point deadline =
            waitSeconds < 0 ? Clock::time_point::max()
                            : Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(std::min(waitSeconds, 1e7)));
        Clock::time_point lastReport{};
        double lastFraction = -1.0;
        bool cancelSent = false;
        while (run) {
            drainWorkerLog();
            if (run->job->finished()) return finishActiveRun();
            if (cancelled()) {
                // run_status only stops waiting; run cancels the job it started
                if (!cancelOnRequest) return runningOutcome();
                if (!cancelSent) {
                    wb.cancelRun();
                    workerCancel = true;
                    cancelSent = true;
                }
            }
            const Clock::time_point now = Clock::now();
            const double fraction = run->job->progress().fraction.load();
            if (now - lastReport >= std::chrono::milliseconds(250) && fraction != lastFraction) {
                progress(fraction, progressMessage(*run->job));
                lastReport = now;
                lastFraction = fraction;
            }
            if (now >= deadline) return runningOutcome();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        return lastOutcome.is_null() ? json{{"status", "idle"}} : lastOutcome;
    }

    json HeadlessWorkbench::Impl::finishActiveRun() {
        std::unique_ptr<ActiveRun> r = std::move(run);
        if (r->thread.joinable()) r->thread.join();
        {
            const std::lock_guard<std::mutex> g(stateMutex);
            statusJob.reset();
            statusRunId.clear();
        }
        wb.finishRun(r->job);
        drainWorkerLog();
        json outcome = outcomeOf(*r);
        capturingRun = false;
        lastOutcome = outcome;
        lastWorkerFailure = r->job->workerFailure();
        if (r->job->succeeded()) lastRunTarget = r->targetId;
        if (r->announce) pushEvent({{"event", "run_finished"}, {"run_id", r->id}, {"result", outcome}});
        return outcome;
    }

    json HeadlessWorkbench::Impl::runningOutcome() const {
        const ActiveRun& r = *run;
        RunProgress& p = r.job->progress();
        const int step = p.stepIndex.load();
        return {{"status", "running"},
                {"run_id", r.id},
                {"target_step", r.target + 1},
                {"seconds", std::chrono::duration<double>(Clock::now() - r.started).count()},
                {"backend", r.backend},
                {"worker", nullptr},
                {"steps", json::array()},
                {"output", nullptr},
                {"progress", {{"fraction", p.fraction.load()}, {"step", step >= 0 ? json(step + 1) : json(nullptr)}, {"message", progressMessage(*r.job)}}},
                {"log", lastLines(runLog, 50)}};
    }

    json HeadlessWorkbench::Impl::outcomeOf(const ActiveRun& r) const {
        RunJob& job = *r.job;
        const std::string status = job.succeeded() ? "succeeded" : job.wasCancelled() ? "cancelled"
                                                                                      : "failed";
        json steps = json::array();
        for (const StepReport& rep : job.reports()) {
            if (rep.state == StepReport::State::Running) continue;
            const Step* s = rep.index >= 0 && rep.index < job.pipeline().size() ? &job.pipeline().at(rep.index) : nullptr;
            steps.push_back({{"step", rep.index + 1},
                             {"kind", s ? s->kind : std::string()},
                             {"name", s ? s->name : std::string()},
                             {"state", stateName(rep.state)},
                             {"seconds", rep.seconds},
                             {"note", rep.note},
                             {"error", rep.error}});
        }
        json output = nullptr;
        if (job.succeeded()) {
            if (const std::shared_ptr<const StepOutput> out = job.output()) {
                // A skipped target passes its input through: the output is that
                // of the last step above it that ran, and is reported as such.
                int from = r.target;
                while (from > 0 && from < job.pipeline().size() && !job.pipeline().at(from).enabled) --from;
                const int now = from < job.pipeline().size() ? wb.pipeline().indexOf(job.pipeline().at(from).id) : -1;
                output = {{"step", from + 1},
                          {"shape", out->meta.shapeString()},
                          {"dims", dimsJson(out->meta.dims)},
                          {"fresh", now >= 0 && wb.outputFresh(now)},
                          {"labels", out->labels && !out->labels->empty() ? json{{"count", out->labels->stats().size()}} : json(nullptr)},
                          {"diagnostics_summary", out->diagnostics.summary}};
            }
        }
        // finishRun logs where the steps ran ("Local worker: cuda:0 · ..."): the stamp, then the note
        json ranOn = nullptr;
        for (const std::string& line : runLog)
            for (const char* prefix : {"Local worker: ", "HPC worker: "})
                if (const std::size_t at = line.find(prefix); at != std::string::npos && at <= 9) ranOn = line.substr(at);
        json o = {{"status", status},
                  {"run_id", r.id},
                  {"target_step", r.target + 1},
                  {"seconds", job.seconds()},
                  {"backend", r.backend},
                  {"worker", ranOn},
                  {"steps", steps},
                  {"output", output},
                  {"progress", {{"fraction", 1.0}, {"step", nullptr}, {"message", ""}}},
                  {"log", lastLines(runLog, 50)}};
        if (status == "failed") o["error"] = job.error();
        if (job.workerFailure()) o["worker_error"] = workerErrorData(*job.workerFailure());
        return o;
    }

    void HeadlessWorkbench::Impl::raiseForOutcome(const json& o) const {
        const std::string status = o.value("status", std::string());
        if (status == "cancelled") throw ToolFailure("cancelled", "the run was cancelled", {}, o);
        if (status != "failed") return;
        if (lastWorkerFailure) {
            const WorkerStartError& e = *lastWorkerFailure;
            json data = o;
            data.update(workerErrorData(e));
            throw ToolFailure("worker_unavailable", e.what(), e.hint.empty() ? options.setupHint : e.hint, data);
        }
        // worker_unavailable is a worker that did not start (RunJob::workerFailure),
        // which a Python environment may fix. An HPC endpoint that does not answer
        // or refuses the token, or a local worker that is busy with another client,
        // is a failed run: a download would not help, and its hint says what would.
        const std::string error = o.value("error", std::string());
        std::string hint = "the steps' errors and get_log say why; set_params fixes a step";
        if (error.rfind("Worker unavailable:", 0) == 0)
            hint = o.value("backend", std::string()) == "hpc" ? hpcHint(false) : "get_log and worker_status (check:true) say why the Python worker did not answer";
        throw ToolFailure("run_failed", "the run failed: " + error, hint, o);
    }

    json HeadlessWorkbench::Impl::runSync(int target) {
        startRun(target, false);
        json outcome = waitRun(-1.0, true);
        raiseForOutcome(outcome);
        return outcome;
    }

    // --- the worker ---------------------------------------------------------------------------

    json HeadlessWorkbench::Impl::workerInterpreterJson() const {
        try {
            const pyenv::Interpreter in = worker.interpreter();
            return {{"path", in.path}, {"source", pyenv::toString(in.source)}};
        } catch (const std::exception&) {
            return {{"path", ""}, {"source", ""}};
        }
    }

    json HeadlessWorkbench::Impl::workerStatusJson() const {
        const std::string dir = scriptDir();
        json uv = nullptr;
        if (const std::string path = pyenv::findUv(); !path.empty()) uv = {{"path", reported(path)}, {"version", pyenv::uvVersion(path)}};
        const std::vector<std::string> required = pyenv::requirements(dir, false);
        json extras = json::array();
        for (const std::string& r : pyenv::requirements(dir, true))
            if (std::find(required.begin(), required.end(), r) == required.end()) extras.push_back(r);
        json candidates = json::array();
        for (const std::string& c : pyenv::pythonCandidates()) candidates.push_back(reported(c));
        return {{"interpreter", workerInterpreterJson()},
                {"environment", pyenv::environmentStatus(dir, false).toJson()},
                {"uv", uv},
                {"candidates", candidates},
                {"worker_dir", reported(dir)},
                {"requirements", {{"required", required}, {"extras", extras}}},
                {"running", worker.isRunning()}};
    }

    // --- the tools ---------------------------------------------------------------------------

    json HeadlessWorkbench::Impl::openDatasetTool(const json& a) {
        const std::string path = existingPath(a, "path", options.allowNetworkPaths);
        OpenOptions o = openOptionsFromJson(a);
        if (o.readAll) o.progress = progressFn("Reading the dataset");
        OpenResult opened;
        try {
            opened = sirius::app::openDataset(path, o);
        } catch (const std::exception& e) {
            throw ToolFailure("open_failed", "cannot open " + path + ": " + e.what(),
                              "dataset_info shows what the file holds; page_order, c, t and z tell a plain TIFF's layout");
        }
        adopt(std::move(opened), path, o);
        if (!lastOpened.fullLoadSkipped.empty()) warnings.push_back("full load skipped: " + lastOpened.fullLoadSkipped);
        api.noteAction({ActionRecord::Kind::Info, "Opened " + path + kMiddot + wb.dataset().shapeString(), "", {}, "open_dataset"});
        json info = datasetInfo(wb.dataset(), &lastOpened);
        info["workspace"] = workspace;
        return info;
    }

    json HeadlessWorkbench::Impl::datasetInfoTool(const json& a) {
        if (!has(a, "path")) {
            if (!wb.hasDataset()) throw ToolFailure("no_dataset", "no dataset is open and no path was given", "pass path, or open_dataset first");
            return datasetJson();
        }
        const std::string path = existingPath(a, "path", options.allowNetworkPaths);
        OpenOptions o = openOptionsFromJson(a);
        o.readAll = false;
        try {
            if (boolArg(a, "open", false)) {
                // a lazy open reads the metadata the probe does not; the workspace keeps its dataset
                const OpenResult r = sirius::app::openDataset(path, o);
                return datasetInfo(r.meta, &r);
            }
            return datasetInfo(probeDataset(path, o));
        } catch (const ToolFailure&) {
            throw;
        } catch (const std::exception& e) {
            throw ToolFailure("open_failed", "cannot read " + path + ": " + e.what(), "page_order, c, t and z tell a plain TIFF's layout");
        }
    }

    json HeadlessWorkbench::Impl::loadPipelineTool(const json& a) {
        const std::string path = existingPath(a, "path", options.allowNetworkPaths);
        const std::string dataset = has(a, "dataset") ? existingPath(a, "dataset", options.allowNetworkPaths) : std::string();
        // The file's own Load parameters, read before the workbench loads it. When
        // the dataset they name cannot be opened, the workbench puts the open
        // dataset's parameters back into the Load step, and those describe another
        // file: its page order, voxel size, SIM layout and tile.
        ParamSet fileLoad;
        const std::uint64_t changesBefore = datasetChanges;
        try {
            const Pipeline file = Pipeline::load(path);
            fileLoad = file.at(0).params;
            // The file's paths (the dataset, a PSF, a model) are opened as if
            // an agent had named them: no network path without the flag.
            if (!options.allowNetworkPaths)
                for (const Step& s : file.steps())
                    if (const Operation* op = findOperation(s.kind))
                        for (const ParamSpec& spec : op->info().params)
                            if (spec.type == ParamType::Path && isNetworkPath(s.params.getString(spec.key)))
                                throw ToolFailure("invalid_argument", "its step " + s.name + " names the network path " + s.params.getString(spec.key),
                                                  "use local paths in the pipeline, or ask the user to restart sirius-cli with --allow-network-paths");
            wb.loadPipeline(path);
        } catch (const std::exception& e) {
            throw ToolFailure("invalid_argument", "cannot load the pipeline " + path + ": " + e.what(),
                              "a pipeline is a .sirius.toml file that save_pipeline or the application wrote");
        }
        const bool openedItsOwn = datasetChanges != changesBefore;
        lastRunTarget = 0;
        model.setOutput(nullptr);   // the pipeline may have opened another dataset
        const std::string named = fileLoad.getString("path");
        if (!dataset.empty()) {
            // The pipeline's Load parameters say how to open it; a light-sheet
            // angle is not an open option and goes with the dataset's own.
            const OpenOptions o = Workbench::openOptionsFromLoadParams(fileLoad);
            try {
                adopt(sirius::app::openDataset(dataset, o), dataset, o);
            } catch (const std::exception& e) {
                throw ToolFailure("open_failed", "the pipeline loaded, but " + dataset + " cannot be opened: " + e.what());
            }
        } else if (!named.empty() && !openedItsOwn) {
            // Nothing was opened. Either the open dataset already is the pipeline's
            // (the same file, opened the same way), or the open failed and the
            // Load step went back to the open dataset's parameters.
            std::string wanted = reported(named);
            try {
                // where the workbench looks: beside the pipeline file, unless the
                // name only exists relative to the working directory
                const fs::path given = fs::u8path(named);
                std::error_code ec;
                const fs::path beside = fs::u8path(path).parent_path() / given;
                if (given.is_relative() && (fs::exists(beside, ec) || !fs::exists(given, ec))) wanted = reported(beside);
            } catch (const std::exception&) {
                // a name that is not a path is not the open dataset's either
            }
            const auto withoutPath = [](const ParamSet& p) {
                json j = p.toJson();
                if (j.is_object()) j.erase("path");
                return j;
            };
            const bool itsOwn = wb.hasDataset() && reported(wb.dataset().sourcePath) == wanted &&
                                withoutPath(wb.pipeline().at(0).params) == withoutPath(fileLoad);
            if (!itsOwn)
                warnings.push_back("the pipeline's dataset " + wanted + " could not be opened (get_log says why)" +
                                   (wb.hasDataset() ? "; the open dataset stays, with its own Load parameters" : std::string()) +
                                   "; open_dataset, or load_pipeline with dataset, opens another");
        }
        const auto missingKinds = [this] {
            json kinds = json::array();
            for (int i = 1; i < wb.pipeline().size(); ++i)
                if (wb.pipeline().at(i).op().info().missing) kinds.push_back(wb.pipeline().at(i).kind);
            return kinds;
        };
        json missing = missingKinds();
        bool loaded = false;
        if (!missing.empty() && options.plugins != HeadlessOptions::Plugins::Off) {
            // the steps of a kind a plugin provides are fixed up once it is loaded (D20)
            try {
                loaded = ensurePlugins(false);
            } catch (const ToolFailure& e) {
                // the pipeline is loaded all the same: a cancel only ends the plugin load
                if (e.code() != "cancelled") throw;
                warnings.push_back("the plugins were not loaded: the call was cancelled");
            }
            missing = missingKinds();
        }
        if (!missing.empty())
            warnings.push_back("steps of kinds nothing provides here: " + missing.dump() +
                               (options.plugins == HeadlessOptions::Plugins::Off ? " (plugins are off)" : ""));
        return {{"workspace", workspace},
                {"pipeline_path", reported(path)},
                {"steps", stepsJson()},
                {"dataset", datasetJson()},
                {"missing_kinds", missing},
                {"plugins_loaded", loaded}};
    }

    json HeadlessWorkbench::Impl::getStateTool(const json&) {
        json steps = stepsJson();
        for (int i = 0; i < wb.pipeline().size(); ++i) {
            const std::shared_ptr<const StepOutput> out = wb.output(i);
            json& s = steps[static_cast<std::size_t>(i)];
            s["has_output"] = out && (out->array || out->source);
            s["fresh"] = wb.outputFresh(i);
            s["output_shape"] = wb.outputMetaOf(i).shapeString();
        }
        json runJson = nullptr;
        if (run) {
            runJson = {{"run_id", run->id}, {"status", "running"}, {"fraction", run->job->progress().fraction.load()}};
        } else if (!lastOutcome.is_null()) {
            runJson = {{"run_id", lastOutcome["run_id"]}, {"status", lastOutcome["status"]}, {"fraction", 1.0}};
        }
        json workerJson = workerInterpreterJson();
        workerJson["interpreter"] = workerJson["path"];
        workerJson.erase("path");
        workerJson["running"] = worker.isRunning();
        std::error_code ec;
        return {{"workspace", workspace},
                {"dataset", datasetJson()},
                {"pipeline_path", wb.pipelinePath().empty() ? json(nullptr) : json(reported(wb.pipelinePath()))},
                {"steps", steps},
                {"backend", backendName(wb.backend())},
                {"cuda_device", cudaDeviceJson(wb.cudaDevice())},
                {"hpc_device", hpcDeviceName(wb.hpcDevice())},
                {"hpc_configured", options.hpc.has_value()},
                {"running", wb.running()},
                {"run", runJson},
                {"can_undo", wb.history().canUndo()},
                {"undo_label", wb.history().undoLabel()},
                {"can_redo", wb.history().canRedo()},
                {"redo_label", wb.history().redoLabel()},
                {"plugins", {{"loaded", pluginsAttempted && pluginsLoaded}, {"count", wb.plugins().size()}, {"error", wb.pluginError()}}},
                {"worker", workerJson},
                {"cwd", reported(fs::current_path(ec))},
                {"scratch", reported(options.scratchDir)}};
    }

    json HeadlessWorkbench::Impl::listOperationsTool(const json& a) {
        const bool detail = boolArg(a, "detail", false);
        if (boolArg(a, "include_plugins", false)) ensurePlugins(false);
        const std::string kind = stringArg(a, "kind");
        const std::string group = lower(stringArg(a, "group"));
        json ops = json::array();
        for (const Operation* op : allOperations()) {
            if (op->kind() == "load") continue;
            if (!kind.empty() && op->kind() != kind) continue;
            if (!group.empty() && lower(op->info().group) != group) continue;
            ops.push_back(operationJson(*op, detail));
        }
        if (!kind.empty() && ops.empty())
            throw ToolFailure("unknown_operation", "there is no operation '" + kind + "'", "list_operations without kind lists them all");
        return {{"operations", ops}};
    }

    json HeadlessWorkbench::Impl::describeOperationTool(const json& a) {
        const std::string kind = requiredString(a, "kind");
        // Not a plugin trigger (D20): a mistyped kind must not start Python.
        const Operation* op = findOperation(kind);
        if (!op || op->info().missing || kind == "load")
            throw ToolFailure("unknown_operation", "there is no operation '" + kind + "'",
                              "list_operations gives the kinds that can be steps; list_plugins loads the plugins' kinds");
        return describeOperation(*op);
    }

    json HeadlessWorkbench::Impl::getHelpTool(const json& a) {
        if (has(a, "page")) return helpPageJson(stringArg(a, "page"));
        std::string kind;
        if (has(a, "kind")) {
            kind = stringArg(a, "kind");
            if (!isHelpPageName(kind) && !findOperation(kind)) invalid("'" + kind + "' is not an operation kind", "list_operations gives the kinds");
        } else if (has(a, "step")) {
            kind = wb.pipeline().at(resolveStep(a)).kind;
        } else {
            return helpPageList();
        }
        // an operation names its page (OpInfo::helpPage); otherwise it is the kind's
        const Operation* op = findOperation(kind);
        json page = helpPageJson(op && !op->info().helpPage.empty() ? op->info().helpPage : kind);
        page["kind"] = kind;
        return page;
    }

    json HeadlessWorkbench::Impl::validateTool(const json&) {
        json steps = json::array();
        bool ok = wb.hasDataset(), needsWorker = false;
        for (int i = 0; i < wb.pipeline().size(); ++i) {
            const Step& s = wb.pipeline().at(i);
            const Validation v = wb.stepValidation(i);
            const Operation* op = findOperation(s.kind);
            // A kind whose plugin is not loaded yet runs in the worker: the
            // plugins load there before the run starts.
            bool stepNeedsWorker = true;
            if (op && !op->info().missing) {
                try {
                    stepNeedsWorker = op->needsWorker(s.params);
                } catch (const std::exception&) {
                    stepNeedsWorker = op->info().remoteCapable;
                }
            }
            const bool enabled = i == 0 || s.enabled;
            const bool fresh = wb.outputFresh(i);
            if (enabled && !v.ok()) ok = false;
            if (enabled && stepNeedsWorker && !fresh) needsWorker = true;
            const std::shared_ptr<const StepOutput> out = wb.output(i);
            steps.push_back({{"step", i + 1},
                             {"kind", s.kind},
                             {"name", s.name},
                             {"enabled", enabled},
                             {"errors", v.errors},
                             {"warnings", v.warnings},
                             {"input_shape", i > 0 ? wb.inputMetaOf(i).shapeString() : std::string()},
                             {"output_shape", wb.outputMetaOf(i).shapeString()},
                             {"estimated_bytes", wb.estimatedBytesOf(i)},
                             {"needs_worker", stepNeedsWorker},
                             {"has_output", out && (out->array || out->source)},
                             {"fresh", fresh},
                             {"missing", !op || op->info().missing}});
        }
        json interpreter = workerInterpreterJson();
        interpreter["interpreter"] = interpreter["path"];
        interpreter.erase("path");
        try {
            interpreter["environment_state"] = pyenv::toString(pyenv::environmentStatus(scriptDir(), false).state);
        } catch (const std::exception&) {
            interpreter["environment_state"] = nullptr;
        }
        return {{"ok", ok}, {"has_dataset", wb.hasDataset()}, {"needs_worker", needsWorker}, {"worker", interpreter}, {"steps", steps}};
    }

    json HeadlessWorkbench::Impl::runTool(const json& a) {
        const int target = has(a, "step") ? resolveStep(a) : lastEnabledStep();
        const double waitSeconds = numberArg(a, "wait_s", 50.0);
        startRun(target, boolArg(a, "force", false));
        json outcome = waitRun(waitSeconds, true);
        if (outcome.value("status", std::string()) == "running") {
            run->announce = true;
            outcome["hint"] = "the run goes on: poll run_status (with wait_s) until its status is no longer running";
            return outcome;
        }
        std::string text = "Ran to step " + Step::number(target) + kMiddot + wb.pipeline().at(target).name;
        char seconds[32];
        std::snprintf(seconds, sizeof seconds, "%.1f s", outcome.value("seconds", 0.0));
        text += kMiddot + std::string(seconds) + (outcome["status"] == "succeeded" ? std::string() : kMiddot + outcome["status"].get<std::string>());
        api.noteAction({ActionRecord::Kind::Run, text, "log", {}, "run"});
        raiseForOutcome(outcome);
        return outcome;
    }

    json HeadlessWorkbench::Impl::setBackendTool(const json& a) {
        const std::string b = lower(requiredString(a, "backend"));
        Backend backend = Backend::Cpu;
        if (b == "cpu") {
            backend = Backend::Cpu;
        } else if (b == "cuda") {
            if (!cudaAvailable()) throw ToolFailure("unsupported", "no CUDA device is available on this computer", "use backend cpu");
            backend = Backend::Cuda;
        } else if (b == "hpc") {
            // D27: the endpoint is fixed when the server starts, never by a tool
            if (!options.hpc) throw ToolFailure("invalid_argument", "no HPC endpoint was given when this server started", "start sirius-cli with --hpc host:port");
            backend = Backend::Hpc;
        } else {
            invalid("'backend' must be cpu, cuda or hpc");
        }
        std::optional<HpcDevice> hpcDevice;
        if (has(a, "hpc_device")) {
            hpcDevice = hpcDeviceFromString(requiredString(a, "hpc_device"));
            if (!hpcDevice) invalid("'hpc_device' must be gpu or cpu");
        }
        if (has(a, "cuda_device")) {
            const json& v = a["cuda_device"];
            if (v.is_string() && lower(v.get<std::string>()) == "all") {
                wb.setCudaDevice(Workbench::kAllCudaDevices);
            } else {
                const int count = cudaDeviceCount();
                const std::int64_t device = integerArg(a, "cuda_device", 0, 0, std::max(count - 1, 0));
                wb.setCudaDevice(static_cast<int>(device));
            }
        }
        wb.setBackend(backend);
        if (hpcDevice) wb.setHpcDevice(*hpcDevice);
        syncWorkerDevice();
        // the plugins may load now where they could not before
        if (!pluginsLoaded) pluginsAttempted = false;
        std::string text = "Backend" + std::string(kArrow) + backendName(backend);
        if (backend == Backend::Hpc) text += kMiddot + hpcDeviceName(wb.hpcDevice());
        api.noteAction({ActionRecord::Kind::Param, text, "", {}, "set_backend"});
        return {{"backend", backendName(wb.backend())}, {"cuda_device", cudaDeviceJson(wb.cudaDevice())}, {"hpc_device", hpcDeviceName(wb.hpcDevice())}};
    }

    json HeadlessWorkbench::Impl::listDevicesTool(const json&) {
        json devices = json::array();
        const int count = cudaDeviceCount();
        for (int i = 0; i < count; ++i) {
            try {
                const DeviceProperties p = deviceProperties(Device::cuda(i));
                char compute[32];
                std::snprintf(compute, sizeof compute, "%d.%d", p.computeMajor, p.computeMinor);
                devices.push_back({{"index", i}, {"name", p.name}, {"memory_gb", static_cast<double>(p.totalMemoryBytes) / 1e9}, {"compute", compute}});
            } catch (const std::exception& e) {
                devices.push_back({{"index", i}, {"name", ""}, {"error", e.what()}});
            }
        }
        return {{"backend", backendName(wb.backend())},
                {"cuda_available", count > 0},
                {"cuda_device", cudaDeviceJson(wb.cudaDevice())},
                {"devices", devices}};
    }

    json HeadlessWorkbench::Impl::renderTool(const json& a) {
        const int i = inspectStep(a);
        const std::shared_ptr<const StepOutput> out = outputFor(i, boolArg(a, "run", false));
        RenderRequest r = renderRequestFromJson(a);
        r.step = i;
        const RenderResult result = renderOutput(out, i, r, model, cancelFn());
        // The viewer draws no labels over the projection, and neither does
        // this. Labels that were asked for, or expected by default, and are
        // missing from the image need a reason and a way to see them.
        if (r.plane == "mip" && out && out->labels && r.labels.value_or(true))
            warnings.push_back("labels are not drawn over a projection (plane mip); plane xy draws them, and several z planes "
                               "give a grid");
        json caption = result.caption;
        caption["step"] = i + 1;
        caption["fresh"] = wb.outputFresh(i);
        attach(result, caption);
        return caption;
    }

    json HeadlessWorkbench::Impl::renderDiagnosticsTool(const json& a) {
        const int i = inspectStep(a);
        const Diagnostics d = wb.diagnosticsOf(i);
        const std::string which = "step " + std::to_string(i + 1) + " (" + wb.pipeline().at(i).name + ")";
        if (d.images.empty())
            throw ToolFailure("not_found", which + " has no diagnostics images", "get_diagnostics lists what it reports; a step that has not run shows a preview at most");
        // Without a tab, `index` counts every image (get_diagnostics' images);
        // with one, the images that tab shows.
        std::vector<int> list;
        std::string tab;
        if (has(a, "tab")) {
            const json& t = a["tab"];
            const DiagnosticTab* found = nullptr;
            if (t.is_number_integer() && t.get<std::int64_t>() >= 0 && static_cast<std::size_t>(t.get<std::int64_t>()) < d.tabs.size())
                found = &d.tabs[static_cast<std::size_t>(t.get<std::int64_t>())];
            for (const DiagnosticTab& dt : d.tabs)
                if (!found && t.is_string() && lower(dt.name) == lower(t.get<std::string>())) found = &dt;
            if (!found) {
                json names = json::array();
                for (const DiagnosticTab& dt : d.tabs) names.push_back(dt.name);
                invalid("no tab " + t.dump() + " in " + which + "'s diagnostics", "tab takes one of the names get_diagnostics lists", {{"tabs", names}});
            }
            list = found->images;
            tab = found->name;
        } else {
            for (std::size_t k = 0; k < d.images.size(); ++k) list.push_back(static_cast<int>(k));
        }
        const std::int64_t index = integerArg(a, "index", 0, 0, std::numeric_limits<int>::max());
        if (static_cast<std::size_t>(index) >= list.size() || list[static_cast<std::size_t>(index)] < 0 ||
            static_cast<std::size_t>(list[static_cast<std::size_t>(index)]) >= d.images.size())
            invalid("image " + std::to_string(index) + " does not exist (there are " + std::to_string(list.size()) + ")");
        const int image = list[static_cast<std::size_t>(index)];
        // capped as render caps it: a larger size is what a client may send, not an error
        const int maxSize = static_cast<int>(std::min<std::int64_t>(integerArg(a, "max_size", 768, 0, std::numeric_limits<int>::max()), 1568));
        const RenderResult result = renderDiagnosticImage(d.images[static_cast<std::size_t>(image)], maxSize);
        json caption = result.caption;
        caption["step"] = i + 1;
        caption["tab"] = tab.empty() ? json(nullptr) : json(tab);
        caption["index"] = index;
        caption["image"] = image;
        attach(result, caption);
        return caption;
    }

    json HeadlessWorkbench::Impl::probeTool(const json& a) {
        const int i = inspectStep(a);
        const std::shared_ptr<const StepOutput> out = outputFor(i, false);
        const Dims5 d = out->meta.dims;
        if (!has(a, "x") || !has(a, "y")) invalid("probe needs x and y (voxels of the xy plane)");
        const Index x = static_cast<Index>(integerArg(a, "x", 0, 0, d.x - 1));
        const Index y = static_cast<Index>(integerArg(a, "y", 0, 0, d.y - 1));
        const Index z = static_cast<Index>(integerArg(a, "z", d.z / 2, 0, d.z - 1));
        const Index t = static_cast<Index>(integerArg(a, "t", 0, 0, d.t - 1));
        model.setOutput(out);
        json values = json::array();
        for (Index c = 0; c < d.c; ++c) {
            const std::optional<float> v = model.valueAt(c, t, z, y, x);
            const std::string label = static_cast<std::size_t>(c) < out->meta.channels.size() ? out->meta.channels[static_cast<std::size_t>(c)].label : std::string();
            values.push_back({{"channel", c}, {"label", label}, {"value", v ? json(*v) : json(nullptr)}});
        }
        json label = nullptr;
        if (const std::shared_ptr<const LabelVolume> labels = out->labels; labels && !labels->empty() && t < labels->t() && z < labels->z() &&
                                                                           y < labels->y() && x < labels->x()) {
            const std::uint32_t id = labels->at(t, z, y, x);
            if (id != 0) {
                const LabelStats* s = labels->statsT() == t ? labels->statsOf(id) : nullptr;
                const LabelAnnotation note = labels->annotationOf(t, id);
                label = {{"id", id},
                         {"class", s ? s->cls : note.cls},
                         {"voxels", s ? json(s->voxels) : json(nullptr)},
                         {"flags", s ? json(s->flags) : json::array()},
                         {"reviewed", s ? s->reviewed : note.reviewed}};
            }
        }
        return {{"step", i + 1}, {"fresh", wb.outputFresh(i)}, {"x", x}, {"y", y}, {"z", z}, {"t", t}, {"values", values}, {"label", label}};
    }

    json HeadlessWorkbench::Impl::statisticsTool(const json& a) {
        const int i = inspectStep(a);
        const std::shared_ptr<const StepOutput> out = outputFor(i, boolArg(a, "run", false));
        const DatasetMeta& meta = out->meta;
        StatisticsOptions o;
        if (has(a, "t") && a["t"].is_string()) {
            if (lower(a["t"].get<std::string>()) != "all") invalid("'t' is a time point or \"all\"");
            o.t = -1;
        } else {
            o.t = static_cast<Index>(integerArg(a, "t", 0, 0, meta.dims.t - 1));
        }
        if (has(a, "channels")) {
            if (!a["channels"].is_array()) invalid("'channels' must be a list of channel indices");
            for (const json& c : a["channels"]) {
                const json one = {{"channels", c}};
                o.channels.push_back(static_cast<Index>(integerArg(one, "channels", 0, 0, meta.dims.c - 1)));
            }
        }
        if (has(a, "percentiles")) {
            if (!a["percentiles"].is_array()) invalid("'percentiles' must be a list of numbers within 0..100");
            o.percentiles.clear();
            for (const json& p : a["percentiles"]) {
                if (!p.is_number() || p.get<double>() < 0.0 || p.get<double>() > 100.0) invalid("'percentiles' must be numbers within 0..100");
                o.percentiles.push_back(p.get<double>());
            }
        }
        o.histogramBins = static_cast<int>(integerArg(a, "histogram_bins", 0, 0, 4096));
        o.maxSamples = static_cast<std::uint64_t>(integerArg(a, "max_samples", std::int64_t{1} << 22, 1000, std::int64_t{1} << 32));
        // what the camera clipped: the Load step of an integer type only
        if (i == 0) o.saturationLevel = pixelTypeMaximum(meta);
        const std::vector<ChannelStatistics> stats =
            channelStatistics(*out, o, [this](double f) { progress(f, "Measuring intensities"); }, cancelFn());
        json channels = json::array();
        bool sampled = false;
        for (const ChannelStatistics& s : stats) {
            json percentiles = json::object();
            for (const auto& [p, v] : s.percentiles) percentiles[percentileKey(p)] = v;
            const std::string label = static_cast<std::size_t>(s.channel) < meta.channels.size() ? meta.channels[static_cast<std::size_t>(s.channel)].label : std::string();
            json c = {{"channel", s.channel},
                      {"label", label},
                      {"min", s.min},
                      {"max", s.max},
                      {"mean", s.mean},
                      {"std", s.stddev},
                      {"count", s.count},
                      {"nan", s.nanCount},
                      {"percentiles", percentiles}};
            if (s.saturatedFraction) c["saturated_fraction"] = *s.saturatedFraction;
            if (!s.histogram.empty()) c["histogram"] = {{"lo", s.histLo}, {"hi", s.histHi}, {"counts", s.histogram}};
            sampled = sampled || s.sampled;
            channels.push_back(std::move(c));
        }
        json result = {{"step", i + 1},
                       {"fresh", wb.outputFresh(i)},
                       {"shape", meta.shapeString()},
                       {"t", o.t < 0 ? json("all") : json(o.t)},
                       {"sampled", sampled},
                       {"channels", channels}};
        if (boolArg(a, "labels", true) && out->labels && !out->labels->empty())
            result["labels"] = labelStatistics(*out->labels, o.t < 0 ? 0 : o.t, meta.voxelUm);
        return result;
    }

    json HeadlessWorkbench::Impl::exportResultTool(const json& a) {
        (void)pathArg(a, "path", options.allowNetworkPaths);   // refused before anything runs
        const int i = inspectStep(a);
        const std::shared_ptr<const StepOutput> out = outputFor(i, boolArg(a, "run", false));
        const ExportOptions o = exportOptionsFromJson(a, out->meta);
        const bool labelsOnly = boolArg(a, "labels_only", false);
        if (!labelsOnly) {
            if (!exportFormatAvailable(o.format))
                throw ToolFailure("unsupported", "this build cannot write zarr or N5 stores", "export to .ome.tif or .tif instead");
            if (const std::string why = validateExport(o, out->meta.dims); !why.empty()) invalid(why);
        } else {
            // The labels are written whole, as they are: say which options that leaves unused.
            std::string ignored;
            for (const char* key : {"dtype", "scaling", "range", "percentiles", "t", "z", "channels", "tiff", "zarr", "include_labels", "include_pipeline"})
                if (has(a, key)) ignored += (ignored.empty() ? "" : ", ") + std::string(key);
            if (!ignored.empty()) warnings.push_back("labels_only writes every label as uint32 and ignores " + ignored);
        }
        json r = exportStepOutput(out, wb.pipeline(), o, labelsOnly, progressFn("Exporting"), cancelFn());
        for (const json& w : r["warnings"]) warnings.push_back(w.get<std::string>());
        r.erase("warnings");
        r["step"] = i + 1;
        const std::string path = r["path"].get<std::string>();
        wb.logLine("Exported step " + Step::number(i) + " to " + path);
        wb.recordEvent("export", {{"step", i + 1}, {"path", path}, {"format", r["format"]}, {"labels_only", labelsOnly}});
        api.noteAction({ActionRecord::Kind::Run, "Exported step " + Step::number(i) + kMiddot + wb.pipeline().at(i).name + kArrow + path, "log", {}, "export_result"});
        return r;
    }

    json HeadlessWorkbench::Impl::exportPythonTool(const json& a) {
        const std::string script = wb.pipeline().toPythonScript(wb.hasDataset() ? wb.dataset().sourcePath : std::string());
        if (!has(a, "path")) return {{"script", script}};
        const std::string path = pathArg(a, "path", options.allowNetworkPaths);
        std::ofstream f(fs::u8path(path), std::ios::binary);
        if (!(f << script)) throw ToolFailure("io_error", "cannot write " + path);
        f.close();
        wb.logLine("Python script written to " + path);
        return {{"path", path}};
    }

    json HeadlessWorkbench::Impl::listPluginsTool(const json& a) {
        if (options.plugins == HeadlessOptions::Plugins::Off)
            throw ToolFailure("unsupported", "plugins are off in this server (--plugins off)", "start sirius-cli with --plugins auto to use them");
        const bool reload = boolArg(a, "reload", false);
        ensurePlugins(reload);
        if (!pluginsLoaded) {
            if (const std::optional<WorkerStartError>& f = wb.pluginWorkerFailure())
                throw ToolFailure("worker_unavailable", f->what(), f->hint.empty() ? options.setupHint : f->hint, workerErrorData(*f));
            const std::string& error = wb.pluginError();
            throw ToolFailure("worker_unavailable", error.empty() ? std::string("the plugins could not be loaded (get_log says why)") : error,
                              options.setupHint);
        }
        json plugins = json::array();
        int registered = 0;
        for (const Workbench::PluginInfo& p : wb.plugins()) {
            plugins.push_back({{"kind", p.kind}, {"name", p.name}, {"file", reported(p.file)}, {"error", p.error}});
            registered += p.error.empty() ? 1 : 0;
        }
        json dirs = json::array();
        for (const std::string& d : wb.pluginDirs()) dirs.push_back(reported(d));
        return {{"plugins", plugins}, {"dirs", dirs}, {"registered", registered}};
    }

    json HeadlessWorkbench::Impl::workerStatusTool(const json& a) {
        const bool check = boolArg(a, "check", false);
        if (check && wb.running())
            throw ToolFailure("busy", "the worker is checked between runs only", "wait for the run with run_status, or cancel_run");
        json status = workerStatusJson();
        if (!check) return status;
        workerCancel = false;
        const Clock::time_point started = Clock::now();
        try {
            const std::unique_ptr<RemoteWorker> w = worker.connect([this] { return workerCancel.load() || cancelled(); });
            const WorkerCapabilities& caps = w->capabilities();
            status["capabilities"] = {{"version", caps.version},
                                      {"protocol", caps.protocolVersion},
                                      {"methods", caps.methods},
                                      {"cuda", caps.cuda},
                                      {"device", caps.device},
                                      {"hostname", caps.hostname},
                                      {"python", caps.python}};
        } catch (const WorkerStartError& e) {
            drainWorkerLog();
            throw ToolFailure("worker_unavailable", e.what(), e.hint.empty() ? options.setupHint : e.hint, workerErrorData(e));
        }
        drainWorkerLog();
        status["seconds"] = std::chrono::duration<double>(Clock::now() - started).count();
        status["running"] = worker.isRunning();
        status["interpreter"] = workerInterpreterJson();
        return status;
    }

    json HeadlessWorkbench::Impl::setupWorkerEnvTool(const json& a) {
        if (!options.allowWorkerSetup)
            throw ToolFailure("consent_required", "this server may not download packages: it was started without --allow-worker-setup",
                              "Ask the user to run `sirius-cli worker setup --yes`, or to restart the server with --allow-worker-setup.");
        // D29: an agent installs only what the worker knows it may use, from
        // an interpreter that is on this machine anyway, from the default index.
        pyenv::SetupOptions so;
        so.createdBy = options.createdBy;
        so.extras = boolArg(a, "extras", false);
        if (has(a, "packages")) {
            const std::vector<std::string>& allowed = pyenv::optionalDistributions();
            if (!a["packages"].is_array()) invalid("'packages' must be a list", "packages takes the worker's optional packages", {{"allowed", allowed}});
            for (const json& p : a["packages"]) {
                const std::string name = p.is_string() ? lower(p.get<std::string>()) : p.dump();
                if (std::find(allowed.begin(), allowed.end(), name) == allowed.end())
                    invalid("'" + name + "' is not one of the packages SIRIUS may install for the worker",
                            "packages takes only the worker's optional packages; anything else the user installs themselves", {{"allowed", allowed}});
                so.extraPackages.push_back(name);
            }
        }
        if (has(a, "base_python")) {
            const std::string base = stringArg(a, "base_python");
            std::vector<std::string> allowed = pyenv::pythonCandidates();
            if (const std::string found = host::findPython(); !found.empty() && std::find(allowed.begin(), allowed.end(), found) == allowed.end())
                allowed.push_back(found);
            const auto same = [&base](const std::string& c) { return c == base || reported(c) == reported(base); };
            const auto match = std::find_if(allowed.begin(), allowed.end(), same);
            if (match == allowed.end())
                invalid("'" + base + "' is not a Python found on this computer", "base_python takes one of the interpreters listed in data.allowed",
                        {{"allowed", allowed}});
            // D29: what runs is the entry found on this computer, never the agent's spelling of it.
            so.basePython = *match;
        }
        const std::string dir = scriptDir();
        if (!boolArg(a, "confirm", false)) {
            json plan = nullptr;
            try {
                plan = pyenv::planSetup(so, dir).toJson();
            } catch (const std::exception&) {
                // the plan is what the user would be asked about; without it the question stands
            }
            throw ToolFailure("consent_required", "setting up the environment downloads packages from the Python Package Index: it needs confirm:true",
                              "Ask the user whether SIRIUS may download them (data is the plan), then call again with confirm:true.", plan);
        }
        // a loaded numpy keeps an update or a rename from happening on Windows
        worker.stop();
        std::mutex lineMutex;
        std::vector<std::string> lines;
        const auto flush = [&] {
            std::vector<std::string> take;
            {
                const std::lock_guard<std::mutex> g(lineMutex);
                take.swap(lines);
            }
            for (const std::string& l : take) wb.logLine("python-env: " + l);
        };
        const pyenv::SetupResult r = pyenv::setup(
            so, dir,
            [&](const std::string& line) {
                const std::lock_guard<std::mutex> g(lineMutex);
                lines.push_back(line);
            },
            [&](double f, const std::string& m) {
                flush();
                progress(f, m);
            },
            cancelFn());
        flush();
        if (!r.ok) {
            std::string code = "failed";
            switch (r.failure) {
                case pyenv::Failure::NoPython:
                case pyenv::Failure::UnsupportedPython:
                case pyenv::Failure::NoEnsurepip: code = "python_not_found"; break;
                case pyenv::Failure::Cancelled: code = "cancelled"; break;
                case pyenv::Failure::Locked:
                case pyenv::Failure::InUse: code = "busy"; break;
                default: break;
            }
            wb.logLine("Python environment: not set up: " + r.message);
            throw ToolFailure(code, r.message, r.hint, r.toJson());
        }
        const pyenv::SetupPlan& plan = r.plan;
        json packages = json::object();
        if (r.marker)
            for (const auto& [name, version] : r.marker->packages) packages[name] = version;
        wb.logLine("Python environment: ready in " + plan.envDir);
        // the plugins may load now where they could not before
        if (!pluginsLoaded) pluginsAttempted = false;
        return {{"env_dir", reported(plan.envDir)},
                {"python", reported(pyenv::environmentPython(plan.envDir))},
                {"base_python", reported(plan.basePython)},
                {"python_version", r.marker ? r.marker->pythonVersion : plan.basePythonVersion},
                {"installer", plan.installer},
                {"mode", plan.nothingToDo ? "none" : modeName(plan.mode)},
                {"packages", packages},
                {"extras", so.extras},
                {"seconds", r.seconds}};
    }

    // --- the table -----------------------------------------------------------------------------

    void HeadlessWorkbench::Impl::installTools() {
        for (const char* name : kViewTools) api.removeTool(name);

        // ToolApi's tools, adapted: a missing kind loads the plugins first, the
        // tracks are those of a step rather than of the viewer.
        if (const ToolSpec* s = api.findTool("add_step")) {
            ToolSpec spec = *s;
            spec.description = "Add a processing step of the given kind (list_operations gives the kinds) at the end, or at a 1-based "
                               "position; optional parameters are applied. Undoable.";
            spec.fn = [this, base = s->fn](const json& a) {
                const std::string kind = requiredString(a, "kind");
                const Operation* op = findOperation(kind);
                if ((!op || op->info().missing) && ensurePlugins(false)) op = findOperation(kind);
                if (!op || op->info().missing || kind == "load")
                    throw ToolFailure("unknown_operation", "there is no operation '" + kind + "'", "list_operations gives the kinds that can be steps");
                return base(a);
            };
            api.addTool(std::move(spec));
        }
        if (const ToolSpec* s = api.findTool("list_tracks")) {
            ToolSpec spec = *s;
            spec.description = "The tracks of a tracked step's labels (default: the last computed step): id, first and last frame, frames "
                               "present, gaps (where identity may have been lost), um per frame, net displacement, parent and children. "
                               "Division counts are the tracker's estimate, not a measurement.";
            spec.parameters = schema({{"step", stepProp()}, {"limit", prop("integer", "Rows to return, those with the most gaps first (default 50)")}});
            spec.fn = [this, base = s->fn](const json& a) {
                const int i = inspectStep(a);
                outputFor(i, false);
                wb.view(i);
                json rest = a;
                rest.erase("step");
                return base(rest);
            };
            api.addTool(std::move(spec));
        }
        // The label tools default to the viewed step, which nothing moves here:
        // without a step they take the one every inspecting tool takes.
        for (const char* name : {"list_labels", "paint_label", "fill_label", "merge_labels", "split_label", "delete_label", "clear_labels",
                                 "set_label_reviewed", "export_labels"}) {
            const ToolSpec* s = api.findTool(name);
            if (!s) continue;
            ToolSpec spec = *s;
            spec.parameters["properties"]["step"] = stepProp("The step whose labels (default: the last computed)");
            spec.fn = [this, base = s->fn](const json& a) {
                if (has(a, "step")) return base(a);
                json b = a;
                b["step"] = defaultInspectStep() + 1;
                return base(b);
            };
            api.addTool(std::move(spec));
        }
        if (const ToolSpec* s = api.findTool("get_step")) {
            ToolSpec spec = *s;
            spec.description = "Details of one step: parameters, summary, validation errors and warnings, output shape, whether it has a "
                               "(fresh) output, and its diagnostics summary.";
            api.addTool(std::move(spec));
        }

        const json openP = openProps();
        const auto withOpen = [&openP](json props) {
            for (auto it = openP.begin(); it != openP.end(); ++it) props[it.key()] = it.value();
            return props;
        };

        addTool("open_dataset",
                "Open a microscopy dataset (TIFF, OME-TIFF, a zarr or N5 store, a folder with a manifest) as the workspace's data; it "
                "becomes the Load step's output. Returns what it is (dimensions, pixel type, voxel size, channels, SIM layout) and the "
                "workspace id. Planes are read when needed unless full_load is set.",
                schema(withOpen({{"path", prop("string", "The dataset; relative paths resolve against the server's working directory")}}), {"path"}),
                [this](const json& a) { return openDatasetTool(a); });
        addTool("dataset_info",
                "Describe a dataset file without opening it in the workspace (dimensions, pixel type, voxel size, channels), or the open "
                "dataset when no path is given. open:true also reads the metadata summary (a lazy open).",
                schema(withOpen({{"path", prop("string", "The file to describe; none = the open dataset")},
                                 {"open", prop("boolean", "Open it lazily to also report metadata_summary and dims_from_metadata")}})),
                [this](const json& a) { return datasetInfoTool(a); });
        addTool("load_pipeline",
                "Load a pipeline file (.sirius.toml) as the workspace's steps, and the dataset its Load step names; with 'dataset', "
                "that one is opened with the pipeline's Load options instead (the pipeline's own dataset, when it exists here, is "
                "still read first). Steps of kinds a plugin provides load the plugins first. Undoable only when no dataset was "
                "opened: opening one starts a new undo history.",
                schema({{"path", prop("string", "The .sirius.toml file")},
                        {"dataset", prop("string", "Open this dataset instead of the one the pipeline names, with the pipeline's Load options")}},
                       {"path"}),
                [this](const json& a) { return loadPipelineTool(a); });
        addTool("save_pipeline", "Save the workspace's steps and their parameters as a pipeline file (.sirius.toml), overwriting it.",
                schema({{"path", prop("string", "The file to write, normally ending in .sirius.toml")}}, {"path"}), [this](const json& a) {
                    const std::string path = pathArg(a, "path", options.allowNetworkPaths);
                    try {
                        wb.savePipeline(path);
                    } catch (const std::exception& e) {
                        throw ToolFailure("io_error", "cannot save the pipeline to " + path + ": " + e.what());
                    }
                    return json{{"path", path}};
                });
        addTool("clear_pipeline", "Remove every step but Load (one undoable change).", schema(), [this](const json&) {
            wb.replacePipeline(Pipeline(), "Clear the pipeline");
            api.noteAction({ActionRecord::Kind::Param, "Cleared the pipeline", "undo", {}, "clear_pipeline"});
            return json{{"steps", stepsJson()}};
        });
        addTool("get_state",
                "The workspace at a glance: its id, the dataset, the steps (with whether each has a fresh output), the backend, the "
                "active or last run, undo and redo, the plugins, the Python worker, the working and scratch directories.",
                schema(), [this](const json& a) { return getStateTool(a); });

        addTool("list_operations",
                "The operations that can be added as steps: kind, name, group, parameter count, presets, and whether they produce or "
                "need labels, need the Python worker or can use a GPU. detail lists each parameter with its schema.",
                schema({{"kind", prop("string", "Only this kind")},
                        {"group", prop("string", "Only this group (Reconstruct, Reduce, Segment, ...)")},
                        {"detail", prop("boolean", "Each operation's parameters with their schemas (default false)")},
                        {"include_plugins", prop("boolean", "Load the user operations (plugins) first; starts the Python worker (default false)")}}),
                [this](const json& a) { return listOperationsTool(a); });
        addTool("describe_operation",
                "Everything about one operation: its parameters (type, default, choices, range, unit, help), presets with their values, "
                "whether it produces or needs labels, needs the Python worker or can use a GPU, and its help page's title and intro.",
                schema({{"kind", prop("string", "The operation kind, as list_operations gives it")}}, {"kind"}),
                [this](const json& a) { return describeOperationTool(a); });
        addTool("get_help",
                "Help pages (Markdown with $...$ LaTeX): an operation's page by kind, the page of a step, or any page by name; without "
                "arguments, the list of pages.",
                schema({{"kind", prop("string", "An operation kind")},
                        {"page", prop("string", "A page name as the list gives it (letters, digits, '_' and '-')")},
                        {"step", stepProp("The page of this step's operation")}}),
                [this](const json& a) { return getHelpTool(a); });

        addTool("validate",
                "Check the pipeline against the dataset without running it: each step's errors and warnings, input and output shapes, "
                "estimated size, whether it needs the Python worker, and whether it already has a fresh output.",
                schema(), [this](const json& a) { return validateTool(a); });
        addTool("run",
                "Run the pipeline up to a step (default: the last) and wait up to wait_s seconds. A run still going then returns status "
                "\"running\"; poll run_status. Steps whose outputs are fresh are not recomputed unless force is set.",
                schema({{"step", stepProp("The step to run to (default the last)")},
                        {"wait_s", prop("number", "Seconds to wait (default 50; 0 = return at once; -1 = until it ends)")},
                        {"force", prop("boolean", "Drop every cached output first (default false)")}}),
                [this](const json& a) { return runTool(a); });
        addTool("run_status", "The active run's progress, or the outcome of the last run; waits up to wait_s seconds for the active one to end.",
                schema({{"wait_s", prop("number", "Seconds to wait for the run to end (default 0; -1 = until it ends)")}}), [this](const json& a) {
                    if (run) return waitRun(numberArg(a, "wait_s", 0.0), false);
                    return lastOutcome.is_null() ? json{{"status", "idle"}} : lastOutcome;
                });
        addTool("cancel_run", "Cancel the active run; its steps stop at their next check.", schema(), [this](const json&) {
            if (!run) return json{{"cancelled", false}, {"status", lastOutcome.is_null() ? json("idle") : lastOutcome["status"]}};
            wb.cancelRun();
            workerCancel = true;
            const Clock::time_point until = Clock::now() + std::chrono::milliseconds(1500);
            while (run && !run->job->finished() && Clock::now() < until) {
                drainWorkerLog();
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            if (run && run->job->finished()) {
                const json outcome = finishActiveRun();
                return json{{"cancelled", true}, {"status", outcome["status"]}, {"run", outcome}};
            }
            return json{{"cancelled", true}, {"status", "cancelling"}};
        });
        addTool("set_backend",
                "Choose where runs compute: cpu, cuda (a GPU, or all of them) or hpc (only the endpoint the server was started with).",
                schema({{"backend", enumProp({"cpu", "cuda", "hpc"}, "The backend")},
                        {"cuda_device", {{"type", json::array({"integer", "string"})}, {"description", "The GPU's index, or \"all\""}}},
                        {"hpc_device", enumProp({"gpu", "cpu"}, "hpc: where the worker computes, its job's GPU or its CPU; kept until changed, "
                                                                "and switched without a new job")}},
                       {"backend"}),
                [this](const json& a) { return setBackendTool(a); });
        addTool("list_devices", "The compute devices: whether CUDA is available, the GPUs with their memory, and the backend in use.", schema(),
                [this](const json& a) { return listDevicesTool(a); });

        addTool("render",
                "Look at a step's output as an image, drawn as the viewer draws it: an xy plane (or a grid of several z), an xz or yz "
                "re-slice, or the maximum projection, with the channels blended in their colours or side by side in grey, and the "
                "labels on top. Returns the image and a caption with its geometry, windows and file.",
                schema({{"step", stepProp("The step to show (default: the last computed)")},
                        {"plane", enumProp({"xy", "xz", "yz", "mip"}, "The view (default xy)")},
                        {"z", {{"type", json::array({"integer", "array"})}, {"items", {{"type", "integer"}}}, {"description", "xy: a plane, or up to 16 planes as a grid (default the middle)"}}},
                        {"t", prop("integer", "The time point (default 0)")},
                        {"y", prop("integer", "xz: the row to re-slice at (default the middle)")},
                        {"x", prop("integer", "yz: the column to re-slice at (default the middle)")},
                        {"channels", intList("The channels to draw (default all)")},
                        {"layout", enumProp({"blend", "channels"}, "Blend the channels in colour, or one grey panel per channel")},
                        {"window", enumProp({"auto", "full"}, "Robust percentiles (auto) or each volume's full range")},
                        {"windows", {{"type", "array"}, {"items", {{"type", "object"}}}, {"description", "Explicit windows [{channel, lo, hi, gamma}]"}}},
                        {"labels", prop("boolean", "Draw the labels (default on when the output has them; never over a mip)")},
                        {"label_opacity", prop("number", "Opacity of the label fill, 0..1 (default 0.45)")},
                        {"label", prop("integer", "A label id to outline")},
                        {"solo", prop("boolean", "Draw only that label")},
                        {"region", intList("[x, y, w, h] in voxels of the plane")},
                        {"max_size", prop("integer", "The image's longer side at most (default 1024, at most 1568; 0 = native); never enlarged")},
                        {"physical_z", prop("boolean", "xz / yz: stretch z by the voxel aspect (default true)")},
                        {"format", enumProp({"png", "jpeg"}, "Default: PNG, or JPEG when the PNG would be over 1 MiB")},
                        {"inline", prop("boolean", "Session protocol: also return the image as base64")},
                        {"run", prop("boolean", "Run the step first when it has no fresh output (default false)")}}),
                [this](const json& a) { return renderTool(a); });
        addTool("render_diagnostics",
                "Look at one of a step's diagnostics images (spectra, fits, previews) as the diagnostics panel draws it, marks included. "
                "get_diagnostics lists them.",
                schema({{"step", stepProp("The step (default: the last computed)")},
                        {"tab", {{"type", json::array({"string", "integer"})}, {"description", "A diagnostics tab by name or number; index then counts its images"}}},
                        {"index", prop("integer", "The image (default 0)")},
                        {"max_size", prop("integer", "The image's longer side at most (default 768, at most 1568; 0 = native)")}}),
                [this](const json& a) { return renderDiagnosticsTool(a); });
        addTool("probe", "Read the values of every channel at one voxel of a step's output, and the label there with its statistics.",
                schema({{"step", stepProp("The step (default: the last computed)")},
                        {"x", prop("integer", "Column")},
                        {"y", prop("integer", "Row")},
                        {"z", prop("integer", "Plane (default the middle)")},
                        {"t", prop("integer", "Time point (default 0)")}},
                       {"x", "y"}),
                [this](const json& a) { return probeTool(a); });
        addTool("statistics",
                "Measure a step's output: per channel the exact minimum, maximum, mean, standard deviation and NaN count, percentiles, "
                "optionally a histogram and (Load step) the saturated fraction; for labels their count, sizes, classes and flags.",
                schema({{"step", stepProp("The step (default: the last computed)")},
                        {"t", {{"type", json::array({"integer", "string"})}, {"description", "A time point, or \"all\" (default 0)"}}},
                        {"channels", intList("The channels (default all)")},
                        {"percentiles", {{"type", "array"}, {"items", {{"type", "number"}}}, {"description", "Default [0.1, 1, 50, 99, 99.9]"}}},
                        {"histogram_bins", prop("integer", "Bins of a histogram per channel (default 0 = none)")},
                        {"labels", prop("boolean", "Also the label statistics (default true)")},
                        {"max_samples", prop("integer", "Values the percentiles and histogram are taken from, at most (default 4194304)")},
                        {"run", prop("boolean", "Run the step first when it has no fresh output (default false)")}}),
                [this](const json& a) { return statisticsTool(a); });
        addTool("get_diagnostics",
                "What a step reports about itself: summary, facts, table, curves, histograms, warnings and its images (which "
                "render_diagnostics draws). detail adds the curves' points and the histograms' bins.",
                schema({{"step", stepProp("The step (default: the last computed)")},
                        {"detail", prop("boolean", "Curve points (at most 200) and histogram bins (default false)")}}),
                [this](const json& a) {
                    const int i = inspectStep(a);
                    return diagnosticsJson(wb.diagnosticsOf(i), i, boolArg(a, "detail", false));
                });
        addTool("get_log", "The most recent lines of the workbench log, the worker's included.",
                schema({{"lines", prop("integer", "How many (default 30, at most 500)")}}), [this](const json& a) {
                    const std::int64_t n = std::clamp<std::int64_t>(has(a, "lines") ? integerArg(a, "lines", 30, 1, 1000000) : 30, 1, 500);
                    const std::vector<std::string>& log = wb.log();
                    json lines = json::array();
                    for (std::size_t i = log.size() > static_cast<std::size_t>(n) ? log.size() - static_cast<std::size_t>(n) : 0; i < log.size(); ++i)
                        lines.push_back(log[i]);
                    return json{{"lines", lines}};
                });

        addTool("export_result",
                "Write a step's output to a file: OME-TIFF or TIFF (tiles, compression, BigTIFF, pyramid), a zarr or N5 store, or raw, "
                "in any pixel type with a scaling rule and an optional t / z / channel range; the labels and the pipeline beside it "
                "on request. The format follows the extension (.ome.tif, .tif, .zarr, .n5, .raw) unless named. Overwrites.",
                schema({{"path", prop("string", "The file or store to write")},
                        {"step", stepProp("The step (default: the last computed)")},
                        {"format", enumProp({"ome-tiff", "tiff", "zarr", "n5", "raw"}, "The container (default from the extension)")},
                        {"dtype", enumProp({"uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"}, "Pixel type (default float32)")},
                        {"scaling", enumProp({"cast", "minmax", "fixed", "percentile"}, "How values map into the pixel type (default cast)")},
                        {"range", {{"type", "array"}, {"items", {{"type", "number"}}}, {"description", "scaling fixed: [lo, hi]"}}},
                        {"percentiles", {{"type", "array"}, {"items", {{"type", "number"}}}, {"description", "scaling percentile: [lo, hi]"}}},
                        {"t", intList("[first, end) of the time points; end -1 = to the last")},
                        {"z", intList("[first, end) of the planes; end -1 = to the last")},
                        {"channels", intList("The channels (default all)")},
                        {"tiff", {{"type", "object"}, {"description", "{tiled, tile:[w, h], compression: none|lzw|deflate, level, bigtiff, ome, pyramid_levels, downsample}"}}},
                        {"zarr", {{"type", "object"}, {"description", "{version: 2|3, chunk:[c, t, z, y, x], codec, level, shard, pyramid_levels, downsample, ome_ngff}"}}},
                        {"include_labels", prop("boolean", "Write the step's labels beside it (default false)")},
                        {"include_pipeline", prop("boolean", "Write <path>.pipeline.toml beside it (default false)")},
                        {"labels_only", prop("boolean", "Write only the labels, as one 32-bit TIFF (default false)")},
                        {"run", prop("boolean", "Run the step first when it has no fresh output (default false)")}},
                       {"path"}),
                [this](const json& a) { return exportResultTool(a); });
        addTool("export_python",
                "The pipeline as a Python script that reproduces it with the sirius package; written to path, or returned as text.",
                schema({{"path", prop("string", "The .py file to write (overwritten); none = return the script")}}),
                [this](const json& a) { return exportPythonTool(a); });

        addTool("list_plugins", "The user operations (plugins) the Python worker serves, and the files that failed to load; starts the worker.",
                schema({{"reload", prop("boolean", "Import the plugin files again (default false)")}}),
                [this](const json& a) { return listPluginsTool(a); });
        addTool("worker_status",
                "Which Python interpreter the worker runs and why, the state of SIRIUS's own Python environment, uv, the candidate "
                "interpreters and the requirements. check:true starts the worker and asks it what it can do.",
                schema({{"check", prop("boolean", "Start the worker and report its capabilities (default false)")}}),
                [this](const json& a) { return workerStatusTool(a); });
        addTool("setup_worker_env",
                "Create or update SIRIUS's own Python environment for the worker, downloading numpy (and, on request, the worker's "
                "optional packages) from the Python Package Index. Only with the user's agreement: the server needs "
                "--allow-worker-setup and the call confirm:true.",
                schema({{"confirm", prop("boolean", "True once the user agreed to the download")},
                        {"extras", prop("boolean", "Also scipy and scikit-image (about 57 MB)")},
                        {"packages", {{"type", "array"}, {"items", {{"type", "string"}}}, {"description", "More of the worker's optional packages"}}},
                        {"base_python", prop("string", "The interpreter to build it from: one worker_status lists as a candidate")}},
                       {"confirm"}),
                [this](const json& a) { return setupWorkerEnvTool(a); });

        // How each behaves, whatever ToolApi said: the busy gate, the hints, the metadata.
        for (const ToolTraits& t : toolTraits()) {
            const ToolSpec* s = api.findTool(t.name);
            if (!s) continue;
            ToolSpec spec = *s;
            spec.title = t.title;
            spec.refusedWhileRunning = !t.whileRunning;
            spec.readOnly = t.readOnly;
            spec.destructive = t.destructive;
            spec.idempotent = t.idempotent;
            spec.openWorld = t.openWorld;
            spec.meta = json::object();
            if (t.longResult) spec.meta["anthropic/maxResultSizeChars"] = 200000;
            if (t.userInteraction) spec.meta["anthropic/requiresUserInteraction"] = true;
            api.addTool(std::move(spec));
        }
    }

    // --- HeadlessWorkbench ---------------------------------------------------------------------

    HeadlessWorkbench::HeadlessWorkbench(HeadlessOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}

    HeadlessWorkbench::~HeadlessWorkbench() = default;

    Workbench& HeadlessWorkbench::workbench() noexcept { return impl_->wb; }

    ToolApi& HeadlessWorkbench::toolApi() noexcept { return impl_->api; }

    LocalWorker& HeadlessWorkbench::worker() noexcept { return impl_->worker; }

    const std::string& HeadlessWorkbench::workspaceId() const noexcept { return impl_->workspace; }

    std::vector<agent::ToolDescriptor> HeadlessWorkbench::tools() const {
        const Impl& m = *impl_;
        std::vector<agent::ToolDescriptor> out;
        for (const ToolTraits& t : toolTraits())
            if (const ToolSpec* s = m.api.findTool(t.name); s && !m.hidden(*s)) out.push_back(m.descriptorOf(*s));
        // anything a caller added to the ToolApi after the table, in its order
        for (const ToolSpec& s : m.api.tools()) {
            const bool listed = std::any_of(toolTraits().begin(), toolTraits().end(), [&s](const ToolTraits& t) { return s.name == t.name; });
            if (!listed && !m.hidden(s)) out.push_back(m.descriptorOf(s));
        }
        return out;
    }

    bool HeadlessWorkbench::hasTool(const std::string& name) const {
        const ToolSpec* s = impl_->api.findTool(name);
        return s && !impl_->hidden(*s);
    }

    agent::ToolResult HeadlessWorkbench::call(const std::string& name, const nlohmann::json& args, const agent::CallContext& ctx) {
        Impl& m = *impl_;
        // A run that ended since the last look is folded back first, or its
        // workbench would still read as running and refuse the call (busy).
        pump();
        const ToolSpec* spec = m.api.findTool(name);
        if (!spec || m.hidden(*spec))
            return agent::failure("unknown_tool", "there is no tool '" + name + "'" + (spec ? " in a read-only server" : ""),
                                  "tools lists the tools this server offers");
        if (!args.is_null() && !args.is_object()) return agent::failure("invalid_argument", "the arguments must be an object");
        json a = args.is_object() ? args : json::object();
        // D16: a call meant for another workspace (a restarted server) is refused.
        // A null one is not given, as with every other argument (clients send
        // optional fields as null).
        if (a.contains("workspace")) {
            if (!a["workspace"].is_null() && (!a["workspace"].is_string() || a["workspace"].get<std::string>() != m.workspace))
                return agent::failure("stale_workspace", "this server's workspace is " + m.workspace + ", not " + a["workspace"].dump(),
                                      "the server was restarted and its state is gone: get_state, then open the dataset again",
                                      {{"workspace", m.workspace}});
            a.erase("workspace");
        }
        std::vector<std::string> warnings;
        if (spec->parameters.is_object() && spec->parameters.contains("properties") && spec->parameters["properties"].is_object()) {
            const json& known = spec->parameters["properties"];
            for (auto it = a.begin(); it != a.end();) {
                if (known.contains(it.key())) {
                    ++it;
                    continue;
                }
                warnings.push_back("unknown argument '" + it.key() + "' ignored");
                it = a.erase(it);
            }
        }

        m.ctx = &ctx;
        m.images.clear();
        m.warnings = std::move(warnings);
        m.cancelRequested = false;
        {
            const std::lock_guard<std::mutex> g(m.stateMutex);
            m.activeTool = name;
        }
        const std::uint64_t before = m.wb.history().revision();
        agent::ToolResult result;
        try {
            const json r = m.api.call(name, a);
            if (r.is_object() && r.contains("error_kind")) {
                // D25: a failure is a result with error_kind, and nothing else is
                const std::string code = r["error_kind"].is_string() ? r["error_kind"].get<std::string>() : std::string("failed");
                const std::string message = r.contains("error") && r["error"].is_string() ? r["error"].get<std::string>() : code;
                const std::string hint = r.contains("hint") && r["hint"].is_string() ? r["hint"].get<std::string>() : std::string();
                result = agent::failure(code, message, hint, r.contains("data") ? r["data"] : json(nullptr));
            } else {
                result.ok = true;
                result.value = r.is_object() ? r : r.is_array() ? json{{"items", r}}
                                                                : json{{"value", r}};
                scrubSteps(result.value);
                result.images = std::move(m.images);
                // A value clamped into its range is a warning like any other,
                // where a client looks for them, not only a field of the answer.
                if (result.value.contains("clamped") && result.value["clamped"].is_array())
                    for (const json& w : result.value["clamped"])
                        if (w.is_string()) m.warnings.push_back(w.get<std::string>());
            }
        } catch (const std::exception& e) {
            result = agent::failure("internal", e.what());
        }
        for (const ActionRecord& action : m.api.takeActions()) result.changes.push_back(action.text);
        // The revision moved, and the history still holds the change: opening a
        // dataset (load_pipeline) moves it and then starts a new history.
        result.undoable = m.wb.history().revision() != before && (m.wb.history().canUndo() || m.wb.history().canRedo());
        result.warnings = std::move(m.warnings);
        m.images.clear();
        m.warnings.clear();
        m.ctx = nullptr;
        {
            const std::lock_guard<std::mutex> g(m.stateMutex);
            m.activeTool.clear();
        }
        m.drainWorkerLog();
        if (m.wb.recording()) m.wb.recordEvent("agent_tool", {{"name", name}, {"args", a}, {"ok", result.ok}});
        return result;
    }

    agent::Status HeadlessWorkbench::status() const {
        const Impl& m = *impl_;
        agent::Status s;
        const std::lock_guard<std::mutex> g(m.stateMutex);
        s.workspace = m.workspace;
        s.activeTool = m.activeTool;
        if (m.statusJob) {
            s.running = true;
            s.fraction = m.statusJob->progress().fraction.load();
            const int step = m.statusJob->progress().stepIndex.load();
            s.step = step >= 0 ? step + 1 : -1;
            s.message = progressMessage(*m.statusJob);
            s.runId = m.statusRunId;
        }
        return s;
    }

    void HeadlessWorkbench::cancelActive() {
        Impl& m = *impl_;
        m.cancelRequested = true;
        m.workerCancel = true;
        const std::lock_guard<std::mutex> g(m.stateMutex);
        if (m.statusJob) m.statusJob->cancel();
    }

    void HeadlessWorkbench::pump() {
        Impl& m = *impl_;
        m.drainWorkerLog();
        if (m.run && m.run->job->finished()) m.finishActiveRun();
    }

    std::vector<nlohmann::json> HeadlessWorkbench::takeEvents() {
        std::vector<json> out = std::move(impl_->events);
        impl_->events.clear();
        return out;
    }

} // namespace sirius::app
