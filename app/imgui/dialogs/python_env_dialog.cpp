// "Set up Python for SIRIUS": SIRIUS's own Python environment for the worker
// (core/python_env.hpp) from the GUI. The same dialog is the offer made when
// the worker cannot start for want of packages or of a Python at all, and
// what Preferences ▸ Compute opens to set the environment up, update, repair
// or recreate it. It says what would be downloaded, from where and into
// which folder, and nothing is downloaded until its button is pressed: a
// worker that starts on its own never installs anything.
//
// Nothing here holds the GUI thread. The plan (the interpreters there are,
// the one the environment would be made from, whether uv is installed) is
// worked out on a DialogThread, since probing an interpreter starts it, and
// a dialog closed meanwhile does not wait for it. The setup runs as a Bridge
// task on the worker thread; the dialog reads its progress and installer
// lines every frame. What has to happen on the GUI thread when it ends (the
// worker restarted, the plugins loaded again) is posted to the Bridge, so it
// happens as well when the dialog was closed in the meantime.

#include "imgui/dialogs/dialogs.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <imgui.h>

#include "core/python_env.hpp"
#include "core/rpc.hpp"
#include "core/worker_error.hpp"
#include "imgui/dialogs/export_dialog_support.hpp"
#include "imgui/platform.hpp"
#include "imgui/settings.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui {

    namespace {

        using namespace dialog_support;
        using theme::px;

        constexpr std::size_t kDetailLines = 200;   // what "Show details" keeps of the installer's output

        constexpr const char* kWhatTheWorkerDoes = "SIRIUS runs user operations (plugins), segmentation models, tracking and "
                                                   "the model hub in a Python worker.";

        // Where a machine without Python gets one, per system.
        const char* installPythonHint() {
#if defined(_WIN32)
            return "Install Python from python.org (tick \"Add python.exe to PATH\"), or run: winget install Python.Python.3.13";
#elif defined(__APPLE__)
            return "Install Python from python.org, or run: brew install python";
#else
            return "Install it with the system's package manager, for example: sudo apt install python3 python3-venv";
#endif
        }

        bool samePath(const std::string& a, const std::string& b) {
            if (a.empty() || b.empty()) return false;
            const std::string x = replaceAll(a, "\\", "/"), y = replaceAll(b, "\\", "/");
#ifdef _WIN32
            return toLower(x) == toLower(y);   // Windows paths ignore case
#else
            return x == y;
#endif
        }

        // "numpy", "scipy and scikit-image", "a, b and c".
        std::string andList(const std::vector<std::string>& items) {
            if (items.empty()) return std::string();
            if (items.size() == 1) return items.front();
            return join(std::vector<std::string>(items.begin(), items.end() - 1), ", ") + " and " + items.back();
        }

        // The worker reports import names; what pip installs is the distribution.
        std::vector<std::string> distributions(const std::vector<std::string>& importNames) {
            std::vector<std::string> out;
            for (const std::string& m : importNames) out.push_back(m == "skimage" ? std::string("scikit-image") : m);
            if (out.empty()) out.emplace_back("numpy");
            return out;
        }

        // "numpy is" / "numpy and scipy are", before "not installed".
        std::string missingText(const std::vector<std::string>& importNames) {
            const std::vector<std::string> names = distributions(importNames);
            return andList(names) + (names.size() == 1 ? " is" : " are");
        }

        // "about 13 MB, " before what follows it; nothing when nothing is
        // known to be downloaded (an update of packages the marker lists).
        std::string aboutMegabytes(std::uint64_t bytes, const char* then = "") {
            if (bytes == 0) return std::string();
            return format("about %.0f MB%s", std::max(1.0, std::round(static_cast<double>(bytes) / 1e6)), then);
        }

        // The error's message without its lead-ins, to follow one of ours.
        std::string reasonOf(const WorkerStartError& e) {
            std::string s = e.what();
            for (const char* lead : {"the Python worker cannot start: ", "the Python worker did not start: ",
                                     "SIRIUS's own Python environment no longer runs: "})
                if (startsWith(s, lead)) s = s.substr(std::strlen(lead));
            return trimmed(s);
        }

        std::string workerDirectory(App& app) {
            std::string dir = app.launcher().scriptDir();
            if (dir.empty()) dir = settings().getString("worker/dir");
            if (dir.empty()) dir = workerScriptPath();
            return dir;
        }

        // "numpy 2.5.3 (Python 3.14.5)": the required packages as the marker
        // recorded them.
        std::string readyText(const pyenv::SetupResult& r, const std::vector<std::string>& required) {
            std::vector<std::string> parts;
            for (const std::string& name : required.empty() ? std::vector<std::string>{"numpy"} : required) {
                std::string part = name;
                if (r.marker) {
                    const auto it = r.marker->packages.find(name);
                    if (it != r.marker->packages.end() && !it->second.empty()) part += " " + it->second;
                }
                parts.push_back(std::move(part));
            }
            std::string version = r.marker ? r.marker->pythonVersion : std::string();
            if (version.empty()) version = r.plan.basePythonVersion;
            return join(parts, ", ") + (version.empty() ? std::string() : " (Python " + version + ")");
        }

        // --- what the threads share with the dialog ------------------------------------

        struct Candidate {
            std::string path, version;
        };

        // What the planning thread found.
        struct Planned {
            pyenv::EnvironmentStatus status;
            pyenv::SetupPlan plan;
            std::vector<Candidate> candidates;           // usable as a base, in pythonCandidates() order
            std::optional<pyenv::PythonInfo> failing;    // the interpreter the worker could not start in
            std::vector<std::string> required, extras;   // requirements.txt, and what requirements-extra.txt adds
            bool extrasOn = false;                       // what the plan was made with
            std::vector<std::string> extraPackages;
            std::string error;                           // no plan could be made
        };

        struct PlanInput {
            std::string scriptDir, failingInterpreter;
            pyenv::SetupOptions options;
            bool keepPackages = false;   // Update, Repair, Recreate: what the environment has now
            // Another "Python to use" was chosen: only the plan is made again
            // (options.basePython names it), the rest is kept from this one.
            std::optional<Planned> previous;
        };

        // On the planning thread: probing an interpreter starts it. `stop`
        // (the dialog has closed) ends the work before the next interpreter
        // is started; the answer is not wanted then.
        Planned makePlan(const PlanInput& in, const std::atomic<bool>& stop) {
            Planned p;
            if (in.previous) {
                p = *in.previous;
                try {
                    p.plan = pyenv::planSetup(in.options, in.scriptDir);
                    p.error.clear();
                } catch (const std::exception& e) {
                    p.error = e.what();
                }
                return p;
            }
            try {
                p.status = pyenv::environmentStatus(in.scriptDir, false);
                pyenv::SetupOptions o = in.options;
                if (in.keepPackages && p.status.marker) {
                    o.extras = p.status.marker->extras;
                    o.extraPackages = p.status.marker->extraPackages;
                }
                p.extrasOn = o.extras;
                p.extraPackages = o.extraPackages;
                p.required = pyenv::requirements(in.scriptDir, false);
                for (const std::string& r : pyenv::requirements(in.scriptDir, true))
                    if (std::find(p.required.begin(), p.required.end(), r) == p.required.end()) p.extras.push_back(r);
                if (stop.load()) return p;
                p.plan = pyenv::planSetup(o, in.scriptDir);
                bool planBaseListed = p.plan.basePython.empty();
                for (const std::string& c : pyenv::pythonCandidates()) {
                    if (stop.load()) return p;
                    const std::optional<pyenv::PythonInfo> info = pyenv::probe(c);
                    if (!info) continue;
                    if (info->problem.empty()) p.candidates.push_back({c, info->version});
                    if (samePath(c, in.failingInterpreter)) p.failing = info;
                    planBaseListed = planBaseListed || samePath(c, p.plan.basePython);
                }
                if (stop.load()) return p;
                if (!in.failingInterpreter.empty() && !p.failing) p.failing = pyenv::probe(in.failingInterpreter);
                // The plan's base may come from elsewhere (the marker, the py
                // launcher): offered first when it can be one.
                if (!planBaseListed && !stop.load()) {
                    const std::optional<pyenv::PythonInfo> info = pyenv::probe(p.plan.basePython);
                    if (info && info->problem.empty())
                        p.candidates.insert(p.candidates.begin(), Candidate{p.plan.basePython, info->version});
                }
            } catch (const std::exception& e) {
                p.error = e.what();
            }
            return p;
        }

        // Shared by the dialog, its planning thread, the setup task and what
        // the task posts to the GUI thread. It outlives the dialog: a setup
        // goes on, and ends, when the dialog was closed.
        struct Shared {
            std::mutex mutex;
            // Set (under the mutex) as the dialog goes. The planning thread
            // then stops and leaves the Bridge alone, which may be going too;
            // a setup, on the Bridge's own thread, goes on and says itself
            // how it ended.
            std::atomic<bool> gone{false};
            std::optional<Planned> planned;   // the planning thread's answer, taken by the dialog
            double fraction = 0.0;
            std::string message;
            std::deque<std::string> lines;    // the last kDetailLines lines of the installers
            std::uint64_t lineCount = 0;
            std::optional<pyenv::SetupResult> result;
            enum class Plugins { NotYet,
                                 Loading,
                                 Loaded } plugins = Plugins::NotYet;
            int pluginCount = 0;
            std::string pluginError;

            void addLine(const std::string& line) {
                const std::lock_guard<std::mutex> g(mutex);
                lines.push_back(line);
                while (lines.size() > kDetailLines) lines.pop_front();
                ++lineCount;
            }
        };

        // Loads the plugins again, from the next frame on: the dialog first says
        // that it does, since loading them starts the worker, which holds the
        // GUI thread while it comes up. GUI thread.
        void loadPluginsLater(App& app, const std::shared_ptr<Shared>& shared) {
            {
                const std::lock_guard<std::mutex> g(shared->mutex);
                shared->plugins = Shared::Plugins::Loading;
            }
            app.bridge().post([&app, shared] {
                int count = 0;
                std::string error;
                // The registry is in use during a run (loadPlugins refuses);
                // the next load, or Window ▸ User operations, brings them.
                if (app.wb().canEdit()) {
                    count = app.wb().loadPlugins(false);
                    error = app.wb().pluginError();
                } else {
                    error = "a run is in progress; Window \xE2\x96\xB8 User operations loads them once it has finished";
                }
                const std::lock_guard<std::mutex> g(shared->mutex);
                shared->plugins = Shared::Plugins::Loaded;
                shared->pluginCount = count;
                shared->pluginError = error;
            });
        }

        // The GUI-thread half of a setup, posted by the task when it ends. The
        // setup has logged its outcome itself.
        void setupEnded(App& app, const std::shared_ptr<Shared>& shared, const pyenv::SetupResult& r, bool instead) {
            if (!r.ok) return;
            if (instead) settings().remove("worker/python");
            // The next start runs the new environment: a worker that a step or
            // the model hub started meanwhile would go on with the old one.
            app.launcher().stop();
            loadPluginsLater(app, shared);
        }

        // --- controls -------------------------------------------------------------------
        //
        // The progress bar and the button row used to be here, and a second
        // copy of the bar was in the model hub. Both are
        // export_dialog_support.hpp's now (dialog_support::progressBar,
        // dialog_support::buttonRow), so a new dialog that needs either takes
        // the one that exists.

        void paragraph(const std::string& text, ImU32 color = theme::kText) { widgets::textWrapped(text, 13, color); }

        // --- the dialog -------------------------------------------------------------------

        // What the dialog is for, which decides what it says and offers.
        enum class Variant { SetUp,         // create it: the Python found lacks numpy, or Preferences' Set up
                             NoPython,      // there is nothing to make it from
                             Configured,    // the interpreter set in Preferences lacks packages
                             Environment,   // the interpreter $SIRIUS_PYTHON names lacks packages
                             Repair,        // it no longer runs, or a setup did not finish
                             Update,        // the requirements changed, or packages went missing
                             Recreate };    // Preferences: a new one in place of a working one

        Variant variantFor(const PythonEnvRequest& r) {
            if (!r.failure) {
                switch (r.state) {
                    case pyenv::State::Absent: return Variant::SetUp;
                    case pyenv::State::Outdated: return Variant::Update;
                    case pyenv::State::Incomplete:
                    case pyenv::State::Broken: return Variant::Repair;
                    case pyenv::State::Ready: return Variant::Recreate;
                }
                return Variant::SetUp;
            }
            const WorkerStartError& e = *r.failure;
            if (e.kind == WorkerStartError::Kind::BrokenEnvironment) return Variant::Repair;
            if (e.source == "configured") return Variant::Configured;
            if (e.source == "environment" || e.source == "explicit") return Variant::Environment;
            if (e.source == "managed") return Variant::Update;
            if (e.kind == WorkerStartError::Kind::NoInterpreter) return Variant::NoPython;
            return Variant::SetUp;
        }

        class PythonEnvDialog final : public Dialog {
        public:
            PythonEnvDialog(App& app, PythonEnvRequest request)
                : app_(app), request_(std::move(request)), variant_(variantFor(request_)), shared_(std::make_shared<Shared>()) {
                scriptDir_ = workerDirectory(app);
                extras_ = settings().getBool("worker/environmentExtras", false);
                try {
                    envDir_ = pyenv::environmentDirectory();
                } catch (const std::exception&) {
                    // the plan says where, or says why it cannot
                }
                startPlanning();
            }

            // A plan still being made is not waited for (planner_ hands its
            // thread over): it stops before the next interpreter.
            ~PythonEnvDialog() override {
                const std::lock_guard<std::mutex> g(shared_->mutex);
                shared_->gone.store(true);
            }

            PythonEnvDialog(const PythonEnvDialog&) = delete;
            PythonEnvDialog& operator=(const PythonEnvDialog&) = delete;

            std::string title() const override { return "Set up Python for SIRIUS"; }
            ImVec2 size() const override { return ImVec2(600, 0); }

            void closed(App&) override {
                if (dontAsk_) settings().set("worker/offerEnvironment", false);
            }

            void draw(App& app) override {
                takePlan();
                if (page_ == Page::Running) {
                    const std::lock_guard<std::mutex> g(shared_->mutex);
                    if (shared_->result) page_ = Page::Done;
                }
                const Spacing spacing(8, 10);
                widgets::vspace(2);
                switch (page_) {
                    case Page::Ask: drawAsk(app); break;
                    case Page::Running: drawRunning(app); break;
                    case Page::Done: drawDone(app); break;
                }
            }

        private:
            enum class Page { Ask,
                              Running,
                              Done };

            bool offer() const { return request_.failure.has_value(); }

            pyenv::Mode mode() const {
                switch (variant_) {
                    case Variant::Update: return pyenv::Mode::Update;
                    case Variant::Repair:
                    case Variant::Recreate: return pyenv::Mode::Recreate;
                    default: return pyenv::Mode::Auto;
                }
            }
            // An update or a repair keeps what the environment had (the
            // extras, packages added on the command line); a recreate offers
            // the choice again, starting from what it had.
            bool keepsPackages() const {
                return variant_ == Variant::Update || variant_ == Variant::Repair || variant_ == Variant::Recreate;
            }
            bool offersExtras() const { return variant_ == Variant::SetUp || variant_ == Variant::Recreate; }
            bool offersBase() const {
                return variant_ == Variant::SetUp || variant_ == Variant::Recreate || variant_ == Variant::Repair;
            }

            std::string chosenBase() const {
                if (bases_.empty()) return std::string();
                return bases_[static_cast<std::size_t>(std::clamp(base_, 0, static_cast<int>(bases_.size()) - 1))].path;
            }
            std::string chosenBaseVersion() const {
                if (bases_.empty()) return std::string();
                return bases_[static_cast<std::size_t>(std::clamp(base_, 0, static_cast<int>(bases_.size()) - 1))].version;
            }
            std::string envDir() const {
                if (planned_ && !planned_->plan.envDir.empty()) return planned_->plan.envDir;
                if (planned_ && !planned_->status.dir.empty()) return planned_->status.dir;
                return envDir_;
            }
            std::string installer() const {
                const std::string s = planned_ ? planned_->plan.installer : std::string();
                return s.empty() ? std::string("pip") : s;
            }
            bool nothingToDo() const { return planned_ && planned_->error.empty() && planned_->plan.nothingToDo; }
            // Something to make the environment from (an update installs into the one there is).
            bool canSetUp() const {
                if (!planned_ || !planned_->error.empty()) return false;
                return nothingToDo() || variant_ == Variant::Update || !chosenBase().empty();
            }

            pyenv::SetupOptions setupOptions(const std::string& base) const {
                pyenv::SetupOptions o;
                o.basePython = variant_ == Variant::Update ? std::string() : base;
                o.mode = mode();
                o.useUv = request_.useUv;
                o.extras = extras_;
                if (planned_) o.extraPackages = planned_->extraPackages;
                o.createdBy = std::string("sirius-app ") + SIRIUS_VERSION;
                return o;
            }

            // --- the plan ---------------------------------------------------------------

            // Everything (on opening, Retry), or with `baseOnly` the plan
            // alone, for the "Python to use" just chosen: what the page says
            // (the installer, the warnings) holds for the interpreter the
            // setup will use. Only once the last plan has answered.
            void startPlanning(bool baseOnly = false) {
                PlanInput in;
                in.scriptDir = scriptDir_;
                in.failingInterpreter = request_.failure ? request_.failure->interpreter : std::string();
                in.keepPackages = keepsPackages();
                if (baseOnly && planned_) {
                    in.options = setupOptions(chosenBase());
                    in.previous = planned_;
                    replanning_ = true;
                } else {
                    in.options = setupOptions(std::string());
                    planning_ = true;
                    planned_.reset();
                }
                failNote_.clear();
                const std::shared_ptr<Shared> shared = shared_;
                Bridge& bridge = app_.bridge();   // only while the dialog is there (Shared::gone)
                planner_.start([in, shared, &bridge] {
                    Planned p = makePlan(in, shared->gone);
                    const std::lock_guard<std::mutex> g(shared->mutex);
                    if (shared->gone.load()) return;
                    shared->planned = std::move(p);
                    bridge.wake();
                });
            }

            void takePlan() {
                if (!planning_ && !replanning_) return;
                std::optional<Planned> p;
                {
                    const std::lock_guard<std::mutex> g(shared_->mutex);
                    p.swap(shared_->planned);
                }
                if (!p) return;
                planning_ = replanning_ = false;
                planned_ = std::move(p);
                if (!extrasTouched_) extras_ = planned_->extrasOn;
                // The interpreters to choose from, the plan's own chosen. One
                // the plan chose but cannot use (too old, free-threaded) is
                // not offered: the first usable one is, and the plan's
                // warning says why its own choice was not.
                bases_ = planned_->candidates;
                base_ = 0;
                for (std::size_t i = 0; i < bases_.size(); ++i)
                    if (samePath(bases_[i].path, planned_->plan.basePython)) base_ = static_cast<int>(i);
                // A Python after all (not on PATH, where the worker looks): offer to set up from it.
                if (variant_ == Variant::NoPython && planned_->error.empty() && !chosenBase().empty()) variant_ = Variant::SetUp;
            }

            // --- the first page: what the dialog proposes ------------------------------------

            void drawAsk(App& app) {
                switch (variant_) {
                    case Variant::SetUp:
                    case Variant::Recreate: drawSetUp(app); break;
                    case Variant::NoPython: drawNoPython(); break;
                    case Variant::Configured: drawConfigured(app); break;
                    case Variant::Environment: drawEnvironment(); break;
                    case Variant::Repair:
                    case Variant::Update: drawRepairOrUpdate(app); break;
                }
            }

            // The plan is not there yet, or could not be made: true when the
            // page has nothing more to say.
            bool planPending() {
                if (planning_) {
                    note("Looking for Python interpreters and uv\xE2\x80\xA6");
                    return true;
                }
                if (!planned_ || !planned_->error.empty()) {
                    const std::string why = planned_ ? planned_->error : std::string("no plan");
                    note("Cannot work out the setup: " + why, theme::kAccentText);
                    return true;
                }
                return false;
            }

            std::string versioned(const std::string& path, const std::string& version) const {
                return version.empty() ? path : path + " (Python " + version + ")";
            }

            std::string setUpIntro() const {
                std::string s = kWhatTheWorkerDoes;
                if (instead_) {
                    const std::string configured = request_.failure ? request_.failure->interpreter : std::string();
                    return s + " Once SIRIUS's own environment is set up, the worker runs in it instead of the Python set in "
                               "Preferences \xE2\x96\xB8 Compute" +
                           (configured.empty() ? std::string() : " (" + configured + ")") + ".";
                }
                if (!offer()) return s;
                const WorkerStartError& e = *request_.failure;
                if (e.kind == WorkerStartError::Kind::NoInterpreter)
                    return s + " The worker could not start: it found no Python 3 interpreter on PATH.";
                std::string version = e.pythonVersion;
                if (version.empty() && planned_ && planned_->failing) version = planned_->failing->version;
                s += " The worker could not start: " + missingText(e.missing) + " not installed in the Python it found (" +
                     e.interpreter + (version.empty() ? std::string() : ", Python " + version) + ")";
                if (planned_ && planned_->failing && planned_->failing->externallyManaged)
                    s += ", and that Python does not accept packages installed into it directly";
                return s + ".";
            }

            void drawPackages() {
                std::vector<std::string> required = planned_->required;
                if (required.empty()) required.emplace_back("numpy");
                const bool numpyOnly = required.size() == 1 && required.front() == "numpy";
                const std::string size = numpyOnly ? " (required, about 13 MB)" : " (required)";
                widgets::text("\xE2\x80\xA2  " + join(required, ", ") + size, 13);
                if (!planned_->extras.empty()) {
                    if (offersExtras()) {
                        const std::string label = andList(planned_->extras) + " (about 57 MB)";
                        if (widgets::checkbox(label.c_str(), &extras_)) extrasTouched_ = true;
                        ImGui::Indent(px(22));
                        note("Label clean-up in the worker, the foundation step's labels and some plugins.");
                        ImGui::Unindent(px(22));
                    } else if (extras_) {
                        widgets::text("\xE2\x80\xA2  " + andList(planned_->extras), 13);
                    }
                }
                if (!planned_->extraPackages.empty())
                    widgets::text("\xE2\x80\xA2  " + andList(planned_->extraPackages) + " (added earlier; size unknown)", 13);
            }

            void drawBaseChoice() {
                if (!offersBase() || bases_.size() < 2) return;
                std::vector<std::string> items;
                for (const Candidate& c : bases_) items.push_back(versioned(c.path, c.version));
                const Field f("Python to use");
                widgets::FieldOpts fo;
                fo.enabled = !replanning_;
                if (widgets::combo("##basePython", &base_, items, fo) && !samePath(chosenBase(), planned_->plan.basePython))
                    startPlanning(true);
            }

            // The plan's warnings, or that it is being made for another interpreter.
            void drawWarnings() {
                if (replanning_) {
                    note("Looking at " + chosenBase() + "\xE2\x80\xA6");
                    return;
                }
                for (const std::string& w : planned_->plan.warnings) note(w);
            }

            void noPythonNote() {
                note(format("No Python 3.%d or newer that SIRIUS can make its environment from was found. ", pyenv::kMinMinor) +
                         installPythonHint(),
                     theme::kAccentText);
            }

            // The primary button while a run or another task holds the worker
            // thread, which the setup needs.
            ButtonSpec setupButton(App& app, const std::string& label, bool enabled) const {
                ButtonSpec b{label, enabled, {}};
                if (app.bridge().running()) {
                    b.enabled = false;
                    b.tooltip = "Finish or cancel the run first";
                } else if (app.bridge().taskRunning()) {
                    b.enabled = false;
                    b.tooltip = "Wait for " + app.bridge().taskLabel() + " to finish";
                }
                return b;
            }

            std::string dismissLabel() const { return offer() ? "Not now" : "Cancel"; }

            // The worker found a Python without numpy; Preferences' Set up… and Recreate…
            void drawSetUp(App& app) {
                const bool recreate = variant_ == Variant::Recreate;
                if (!recreate) paragraph(setUpIntro());
                bool ready = false;
                if (!planPending()) {
                    if (nothingToDo()) {
                        paragraph("SIRIUS's Python environment (" + envDir() +
                                  ") is set up and current: nothing needs downloading.");
                        ready = true;
                    } else if (chosenBase().empty()) {
                        noPythonNote();
                    } else {
                        const std::string base = chosenBase();
                        const bool same = offer() && !instead_ && samePath(base, request_.failure->interpreter);
                        const std::string from = same ? std::string("that interpreter") : versioned(base, chosenBaseVersion());
                        if (recreate)
                            paragraph("Recreate SIRIUS's Python environment (" + envDir() + ") from " + from +
                                      " and download what the worker needs again from the Python Package Index (pypi.org). The "
                                      "environment there now stays until the new one works:");
                        else
                            paragraph("SIRIUS can create its own Python environment for the worker from " + from +
                                      " and download what the worker needs from the Python Package Index (pypi.org):");
                        drawPackages();
                        note("Location: " + envDir() + ". Nothing outside that folder is changed." +
                             (offer() ? " Preferences \xE2\x96\xB8 Compute can update, repair or remove it later." : "") +
                             " Downloads with " + installer() + ".");
                        drawBaseChoice();
                        drawWarnings();
                        ready = !replanning_;
                    }
                }
                if (!failNote_.empty()) note(failNote_, theme::kAccentText);
                widgets::vspace(4);
                const std::string primary = nothingToDo() ? "Use it" : (recreate ? "Recreate" : "Download and set up");
                const int pressed = buttonRow(onTop(), {{dismissLabel()}, setupButton(app, primary, ready && canSetUp())},
                                              offer() ? "Don't ask again" : nullptr, offer() ? &dontAsk_ : nullptr);
                if (pressed == 0) close();
                else if (pressed == 1) nothingToDo() ? useEnvironment(app) : startSetup(app);
            }

            // No Python at all.
            void drawNoPython() {
                paragraph(std::string("The worker could not start: no Python 3 interpreter was found on this computer. ") +
                          installPythonHint() + ". Once Python is installed, SIRIUS sets up its own environment from it.");
                if (planning_) note("Looking for Python interpreters\xE2\x80\xA6");
                else if (planned_ && !planned_->error.empty())
                    note("Cannot work out the setup: " + planned_->error, theme::kAccentText);
                widgets::vspace(4);
                const ButtonSpec retry{"Retry", !planning_, "Look for an interpreter again"};
                const int pressed = buttonRow(onTop(), {{"Not now"}, retry}, "Don't ask again", &dontAsk_);
                if (pressed == 0) close();
                else if (pressed == 1) startPlanning();
            }

            // The interpreter set in Preferences lacks packages.
            void drawConfigured(App& app) {
                const WorkerStartError& e = *request_.failure;
                const std::string& path = e.interpreter;
                if (e.kind == WorkerStartError::Kind::NoInterpreter)
                    paragraph("The worker could not start: the Python set in Preferences \xE2\x96\xB8 Compute (" + path +
                              ") was not found. Correct it there, or let SIRIUS set up its own environment and use that "
                              "instead.");
                else
                    paragraph("The worker could not start: " + missingText(e.missing) +
                              " not installed in the Python set in Preferences \xE2\x96\xB8 Compute (" + path +
                              "). Install it there (" + path + " -m pip install " + join(distributions(e.missing), " ") +
                              "), or let SIRIUS set up its own environment and use that instead.");
                if (planning_) note("Looking at SIRIUS's own environment\xE2\x80\xA6");
                widgets::vspace(4);
                const ButtonSpec instead{"Use SIRIUS's environment instead", !planning_ && planned_ && planned_->error.empty()};
                const int pressed = buttonRow(onTop(), {{"Not now"}, {"Open Preferences"}, instead}, "Don't ask again", &dontAsk_);
                if (pressed == 0) {
                    close();
                } else if (pressed == 1) {
                    close();
                    app.defer([&app] { app.preferences(); });
                } else if (pressed == 2) {
                    instead_ = true;
                    if (nothingToDo()) useEnvironment(app);   // it is there already: only the setting goes
                    else variant_ = Variant::SetUp;
                }
            }

            // The interpreter $SIRIUS_PYTHON names lacks packages: only the user can change that.
            void drawEnvironment() {
                const WorkerStartError& e = *request_.failure;
                const std::string& path = e.interpreter;
                const bool variable = e.source == "environment";
                const std::string names = variable ? std::string("which $SIRIUS_PYTHON names")
                                                   : std::string("the interpreter this session was told to use");
                const std::string otherwise = std::string(variable ? "unset SIRIUS_PYTHON" : "name none") +
                                              " so that SIRIUS can use its own environment";
                if (e.kind == WorkerStartError::Kind::NoInterpreter)
                    paragraph("The worker could not start: " + path + ", " + names + ", was not found. Correct it, or " +
                              otherwise + ".");
                else
                    paragraph("The worker could not start: " + missingText(e.missing) + " not installed in " + path + ", " +
                              names + ". Install it there (" + path + " -m pip install " + join(distributions(e.missing), " ") +
                              "), or " + otherwise + ".");
                widgets::vspace(4);
                if (buttonRow(onTop(), {{"Close"}}, "Don't ask again", &dontAsk_) == 0) close();
            }

            // The environment no longer runs, or needs an update.
            void drawRepairOrUpdate(App& app) {
                const bool repair = variant_ == Variant::Repair;
                const std::string dir = envDir();
                bool ready = false;
                if (repair) {
                    const bool incomplete = !offer() && request_.state == pyenv::State::Incomplete;
                    std::string problem = offer() ? reasonOf(*request_.failure) : request_.problem;
                    if (incomplete)
                        paragraph("SIRIUS's Python environment (" + dir + ") is incomplete: a setup did not finish.");
                    else
                        paragraph("SIRIUS's Python environment (" + dir + ") no longer runs" +
                                  (problem.empty() ? std::string() : ": " + problem) +
                                  ". The Python it was made from may have been removed or upgraded.");
                    if (!planPending()) {
                        if (chosenBase().empty()) {
                            noPythonNote();
                        } else {
                            const std::string again = incomplete ? std::string() : std::string(" again");
                            paragraph("Repair recreates it from " + versioned(chosenBase(), chosenBaseVersion()) +
                                      " and downloads its packages" + again + " (" +
                                      aboutMegabytes(planned_->plan.approxDownloadBytes, ", ") + "with " + installer() + ").");
                            drawBaseChoice();
                            drawWarnings();
                            ready = !replanning_;
                        }
                    }
                } else {
                    std::string what;
                    if (offer() && !request_.failure->missing.empty())
                        what = missingText(request_.failure->missing) + " not installed in it";
                    else if (!offer() && !request_.problem.empty()) what = request_.problem;
                    else what = "the requirements changed";
                    paragraph("SIRIUS's Python environment needs an update: " + what + ".");
                    if (!planPending()) {
                        if (nothingToDo()) {
                            paragraph("It is set up and current now: nothing needs downloading.");
                        } else {
                            const std::vector<std::string>& packages = planned_->plan.packages;
                            const std::string names = packages.empty() ? std::string("its packages") : andList(packages);
                            paragraph("Update downloads " + names + " into " + dir + " (" +
                                      aboutMegabytes(planned_->plan.approxDownloadBytes, ", ") + "with " + installer() + ").");
                            drawWarnings();
                        }
                        ready = true;
                    }
                }
                if (!failNote_.empty()) note(failNote_, theme::kAccentText);
                widgets::vspace(4);
                const std::string primary = nothingToDo() ? "Use it" : (repair ? "Repair" : "Update");
                const int pressed = buttonRow(onTop(), {{dismissLabel()}, setupButton(app, primary, ready && canSetUp())},
                                              offer() ? "Don't ask again" : nullptr, offer() ? &dontAsk_ : nullptr);
                if (pressed == 0) close();
                else if (pressed == 1) nothingToDo() ? useEnvironment(app) : startSetup(app);
            }

            // --- running it -----------------------------------------------------------------

            // The environment is there and current: the worker only has to run
            // in it (and the configured interpreter goes, when it replaces that).
            void useEnvironment(App& app) {
                if (instead_) settings().remove("worker/python");
                app.launcher().stop();
                app.wb().logLine("Python environment: the worker runs SIRIUS's environment (" + envDir() + ")");
                pyenv::SetupResult r;
                r.ok = true;
                r.plan = planned_->plan;
                r.marker = planned_->status.marker;
                {
                    const std::lock_guard<std::mutex> g(shared_->mutex);
                    shared_->result = r;
                }
                required_ = planned_->required;
                page_ = Page::Done;
                // one frame first, which says the plugins are loading
                const std::shared_ptr<Shared> shared = shared_;
                app.bridge().post([&app, shared] { loadPluginsLater(app, shared); });
                if (request_.finished) request_.finished();
            }

            void startSetup(App& app) {
                const pyenv::SetupOptions o = setupOptions(chosenBase());
                if (offersExtras()) settings().set("worker/environmentExtras", extras_);
                {
                    const std::lock_guard<std::mutex> g(shared_->mutex);
                    shared_->fraction = 0.0;
                    shared_->message.clear();
                    shared_->lines.clear();
                    shared_->result.reset();
                    shared_->plugins = Shared::Plugins::NotYet;
                }
                required_ = planned_->required;
                const std::shared_ptr<Shared> shared = shared_;
                const std::string scriptDir = scriptDir_;
                const bool instead = instead_;
                const std::function<void()> finished = request_.finished;
                Bridge& bridge = app.bridge();
                WorkerLauncher& worker = app.launcher();
                const bool started = bridge.startTask(
                    "Setting up Python",
                    [&app, &bridge, &worker, o, scriptDir, shared, instead, finished](const Bridge::TaskProgress& progress,
                                                                                      const Bridge::TaskCancelled& cancelled) {
                        // A worker holding numpy blocks an update or a rename on Windows: it goes first.
                        worker.stop();
                        pyenv::SetupResult r;
                        try {
                            // Its lines are the log's as they come: what it does,
                            // the commands, the installers' output, the outcome.
                            r = pyenv::setup(
                                o, scriptDir,
                                [&app, &bridge, &shared](const std::string& line) {
                                    shared->addLine(line);
                                    bridge.post([&app, line] { app.wb().logLine(line); });
                                },
                                [&shared, &progress](double fraction, const std::string& message) {
                                    {
                                        const std::lock_guard<std::mutex> g(shared->mutex);
                                        shared->fraction = fraction;
                                        shared->message = message;
                                    }
                                    progress(fraction, message);
                                },
                                cancelled);
                        } catch (const std::exception& e) {
                            // setup() answers every failure itself; this is one it did not foresee
                            r.ok = false;
                            r.failure = pyenv::Failure::Failed;
                            r.message = e.what();
                            const std::string line = "Python environment: not set up: " + r.message;
                            bridge.post([&app, line] { app.wb().logLine(line); });
                        }
                        {
                            const std::lock_guard<std::mutex> g(shared->mutex);
                            shared->result = r;
                        }
                        // Posted rather than a completion: those are skipped once
                        // the task was cancelled, and a setup cancelled a moment
                        // too late has still made the environment.
                        bridge.post([&app, shared, r, instead, finished] {
                            setupEnded(app, shared, r, instead);
                            if (finished) finished();
                        });
                        // A failure is the task's as well once the dialog that
                        // shows it has closed: the application's box says it,
                        // and the log's last word on it is not "done". While
                        // the dialog is open that box would only repeat its page.
                        if (!r.ok && r.failure != pyenv::Failure::Cancelled && shared->gone.load())
                            throw std::runtime_error(r.message + (r.hint.empty() ? std::string() : " " + r.hint));
                    });
                if (!started) {
                    failNote_ = "The setup could not start: a run or another task holds the worker thread (see the log).";
                    return;
                }
                cancelling_ = false;
                shownLines_ = 0;
                page_ = Page::Running;
            }

            void drawDetails() {
                if (widgets::linkButton(details_ ? "Hide details" : "Show details")) {
                    details_ = !details_;
                    followTail_ = true;
                }
                if (!details_) return;
                std::deque<std::string> lines;
                std::uint64_t count = 0;
                {
                    const std::lock_guard<std::mutex> g(shared_->mutex);
                    lines = shared_->lines;
                    count = shared_->lineCount;
                }
                ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::kSurface);
                if (ImGui::BeginChild("##details", ImVec2(0.0f, px(180)), ImGuiChildFlags_Borders,
                                      ImGuiWindowFlags_HorizontalScrollbar)) {
                    // The tail is followed while the view is at the bottom (as it
                    // was last frame), not once the user has scrolled up.
                    const bool atBottom = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f;
                    const theme::FontScope f(11, theme::mono() ? theme::mono() : ImGui::GetFont());
                    ImGui::PushStyleColor(ImGuiCol_Text, theme::kNeutral800);
                    if (lines.empty()) ImGui::TextUnformatted("(nothing yet)");
                    for (const std::string& line : lines) {
                        // the installers' own lines without the log's prefix
                        const std::size_t from = startsWith(line, "python-env: ") ? std::strlen("python-env: ") : 0;
                        ImGui::TextUnformatted(line.c_str() + from, line.c_str() + line.size());
                    }
                    ImGui::PopStyleColor();
                    if (followTail_ || (atBottom && count != shownLines_)) ImGui::SetScrollHereY(1.0f);
                    followTail_ = false;
                    shownLines_ = count;
                }
                ImGui::EndChild();
                ImGui::PopStyleColor();
            }

            void drawRunning(App& app) {
                double fraction = 0.0;
                std::string message;
                {
                    const std::lock_guard<std::mutex> g(shared_->mutex);
                    fraction = shared_->fraction;
                    message = shared_->message;
                }
                const char* doing =
                    variant_ == Variant::Update ? "Updating" : (mode() == pyenv::Mode::Recreate ? "Recreating" : "Setting up");
                paragraph(std::string(doing) + " SIRIUS's Python environment in " + envDir() + "\xE2\x80\xA6");
                progressBar(fraction, ImGui::GetContentRegionAvail().x);
                note(message.empty() ? std::string("Starting\xE2\x80\xA6") : message);
                drawDetails();
                widgets::vspace(4);
                // Not the default: an Enter pressed again after the one that
                // started the setup must not roll the download back.
                const ButtonSpec cancel{cancelling_ ? "Cancelling\xE2\x80\xA6" : "Cancel", !cancelling_,
                                        "Stop the setup; nothing is left half done", true};
                if (buttonRow(onTop(), {cancel}) == 0) {
                    app.bridge().cancelTask();
                    cancelling_ = true;
                }
            }

            void drawDone(App& app) {
                pyenv::SetupResult r;
                Shared::Plugins plugins = Shared::Plugins::NotYet;
                int pluginCount = 0;
                std::string pluginError;
                {
                    const std::lock_guard<std::mutex> g(shared_->mutex);
                    if (shared_->result) r = *shared_->result;
                    plugins = shared_->plugins;
                    pluginCount = shared_->pluginCount;
                    pluginError = shared_->pluginError;
                }
                if (r.ok) {
                    // setup()'s own words ("Ready: numpy 2.5.3 (Python 3.14.5)."), or the environment as it is
                    std::string text = r.message.empty() ? "Ready: " + readyText(r, required_) + "." : r.message;
                    text += " ";
                    if (plugins != Shared::Plugins::Loaded) text += "Loading plugins\xE2\x80\xA6";
                    else if (!pluginError.empty()) text += "The user operations did not load: " + pluginError;
                    else text += format("%d user operation%s loaded.", pluginCount, pluginCount == 1 ? "" : "s");
                    paragraph(text);
                    drawDetails();
                    widgets::vspace(4);
                    if (buttonRow(onTop(), {{"Close"}}) == 0) close();
                    return;
                }
                paragraph(r.message.empty() ? std::string("The setup failed.") : r.message,
                          r.failure == pyenv::Failure::Cancelled ? theme::kText : theme::kAccentText);
                if (!r.hint.empty()) note(r.hint);
                drawDetails();
                widgets::vspace(4);
                const int pressed = buttonRow(onTop(), {{"Copy log"}, {"Retry", true, "Plan the setup again"}, {"Close"}});
                if (pressed == 0) {
                    std::string text = r.message + (r.hint.empty() ? std::string() : "\n" + r.hint) + "\n\n";
                    {
                        const std::lock_guard<std::mutex> g(shared_->mutex);
                        const std::vector<std::string> tail(shared_->lines.begin(), shared_->lines.end());
                        text += join(tail.empty() ? r.logTail : tail, "\n");
                    }
                    ImGui::SetClipboardText(text.c_str());
                    app.wb().logLine("Python environment: the setup's log is on the clipboard");
                } else if (pressed == 1) {
                    page_ = Page::Ask;
                    startPlanning();
                } else if (pressed == 2) {
                    close();
                }
            }

            App& app_;
            PythonEnvRequest request_;
            Variant variant_;
            Page page_ = Page::Ask;
            bool instead_ = false;         // the environment replaces the configured interpreter
            std::string scriptDir_, envDir_;
            std::shared_ptr<Shared> shared_;
            bool planning_ = false;     // the page waits for the plan
            bool replanning_ = false;   // the plan for another "Python to use"; the page stays
            std::optional<Planned> planned_;
            std::vector<Candidate> bases_;   // "Python to use"
            int base_ = 0;
            bool extras_ = false, extrasTouched_ = false;
            std::vector<std::string> required_;
            bool dontAsk_ = false;
            bool details_ = false, followTail_ = false;
            std::uint64_t shownLines_ = 0;
            bool cancelling_ = false;
            std::string failNote_;
            DialogThread planner_;
        };

        // The session's offer (GUI thread only): made once, and not stacked.
        bool offered = false;
        std::weak_ptr<Dialog> openOffer;

        // The threads of dialogs that closed before their work had ended,
        // each with the flag it sets as it ends. Joined once ended, and all
        // of them at the latest when the process exits.
        struct LeftBehind {
            std::mutex mutex;
            std::vector<std::pair<std::thread, std::shared_ptr<std::atomic<bool>>>> threads;

            void join(bool all) {
                std::vector<std::thread> ended;
                {
                    const std::lock_guard<std::mutex> g(mutex);
                    for (auto it = threads.begin(); it != threads.end();) {
                        if (!all && !it->second->load()) {
                            ++it;
                            continue;
                        }
                        ended.push_back(std::move(it->first));
                        it = threads.erase(it);
                    }
                }
                for (std::thread& t : ended) t.join();
            }
            ~LeftBehind() { join(true); }
        };

        LeftBehind& leftBehind() {
            static LeftBehind threads;
            return threads;
        }

    } // namespace

    DialogThread::~DialogThread() {
        if (!thread_.joinable()) return;
        LeftBehind& left = leftBehind();
        left.join(false);
        if (ended_->load()) {
            thread_.join();   // its work is done: only its last instructions are waited for
            return;
        }
        const std::lock_guard<std::mutex> g(left.mutex);
        left.threads.emplace_back(std::move(thread_), ended_);
    }

    void DialogThread::start(std::function<void()> work) {
        if (thread_.joinable()) thread_.join();
        ended_ = std::make_shared<std::atomic<bool>>(false);
        thread_ = std::thread([work = std::move(work), ended = ended_] {
            try {
                work();
            } catch (...) {
                // the work answers its own failures; nothing may leave a thread
            }
            ended->store(true);
        });
    }

    void finishDialogThreads() { leftBehind().join(true); }

    std::shared_ptr<Dialog> makePythonEnvDialog(App& app, PythonEnvRequest request) {
        return std::make_shared<PythonEnvDialog>(app, std::move(request));
    }

    void offerPythonEnvironment(App& app, const WorkerStartError& error) {
        if (!error.setupWouldHelp()) return;
        const std::string override = toLower(trimmed(platform::environment("SIRIUS_PYTHON_OFFER")));
        std::string why;
        if (override == "never") why = "not offered (SIRIUS_PYTHON_OFFER=never)";
        else if (override != "always") {
            if (app.unattended()) why = "not offered in an unattended session";
            else if (!settings().getBool("worker/offerEnvironment", true)) why = "not offered (turned off)";
            else if (offered) why = "offered once in this session already";
        }
        if (!why.empty()) {
            app.wb().logLine("Python environment: " + why + "; Preferences \xE2\x96\xB8 Compute sets it up.");
            return;
        }
        if (const std::shared_ptr<Dialog> open = openOffer.lock(); open && open->isOpen()) {
            app.showDialog(open);   // raised, not shown twice
            return;
        }
        offered = true;
        PythonEnvRequest request;
        request.failure = error;
        request.useUv = settings().getBool("worker/useUv", true);
        std::shared_ptr<Dialog> dialog = makePythonEnvDialog(app, std::move(request));
        openOffer = dialog;
        app.showDialog(std::move(dialog));
    }

} // namespace sirius::app::gui
