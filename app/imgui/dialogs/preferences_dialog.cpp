// File ▸ Preferences…: default backend and CUDA device, the HPC worker
// connection, the Python interpreter for the local worker and SIRIUS's own
// Python environment, and the assistant provider (Ollama / OpenRouter /
// custom OpenAI-compatible endpoint). Values live in the settings; the
// workbench is updated on Save. The environment's buttons act at once: they
// open the setup dialog over this one, or remove the environment.

#include "imgui/dialogs/dialogs.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>

#include <sirius/device.hpp>

#include "core/python_env.hpp"
#include "core/rpc.hpp"
#include "imgui/cluster_link.hpp"
#include "imgui/dialogs/export_dialog_support.hpp"
#include "imgui/panels/assistant_panel.hpp"
#include "imgui/panels/llm_client.hpp"
#include "imgui/platform.hpp"
#include "imgui/secret_store.hpp"
#include "imgui/settings.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui {

    namespace {

        using namespace dialog_support;
        using theme::px;

        const char* const kProviders[] = {"ollama", "openrouter", "custom"};

        std::string deviceLabel(int i) {
            try {
                const DeviceProperties p = deviceProperties(Device::cuda(i));
                return format("cuda:%d · %s · %.1f GB", i, p.name.c_str(),
                              static_cast<double>(p.totalMemoryBytes) / (1024.0 * 1024.0 * 1024.0));
            } catch (const std::exception&) {
                return format("cuda:%d", i);
            }
        }

        // The backend a default runs on: CUDA falls back to the CPU on a
        // machine without a CUDA device. The stored default stays as chosen,
        // so the GPU is used again once there is one.
        Backend usableBackend(int stored) {
            const auto b = static_cast<Backend>(std::clamp(stored, 0, 2));
            return b == Backend::Cuda && !cudaAvailable() ? Backend::Cpu : b;
        }

        // The stored HPC device ("compute/hpcDevice": gpu | cpu); the GPU by default.
        HpcDevice storedHpcDevice() { return hpcDeviceFromString(settings().getString("compute/hpcDevice", "gpu")).value_or(HpcDevice::Gpu); }

        // "the requirements changed." -> "the requirements changed", to go before one of ours.
        std::string withoutStop(std::string s) {
            s = trimmed(s);
            while (!s.empty() && s.back() == '.') s.pop_back();
            return s;
        }

        // "Removed C:/…" -> "removed C:/…", to follow "Python environment: ".
        // A first word in capitals (a name, a path) stays as it is.
        std::string continuing(std::string s) {
            if (s.size() > 1 && s[0] >= 'A' && s[0] <= 'Z' && s[1] >= 'a' && s[1] <= 'z') s[0] = static_cast<char>(s[0] - 'A' + 'a');
            return s;
        }

        // What the Python environment's section shares with the threads and
        // dialogs that work for it; it outlives the Preferences dialog.
        struct EnvironmentShared {
            std::atomic<bool> stale{false};   // an action changed the environment: look again
            std::mutex mutex;
            std::optional<pyenv::EnvironmentStatus> checked;   // Check's answer
            std::string checkError;
            // The dialog has closed: a Check still running then leaves the
            // Bridge alone, which may be going too.
            bool gone = false;
        };

        class PreferencesDialog : public Dialog {
        public:
            explicit PreferencesDialog(App& app) {
                const Workbench& wb = app.wb();
                backend_ = std::clamp(static_cast<int>(wb.backend()), 0, 2);
                // A stored CUDA default runs on the CPU here (usableBackend);
                // the field still shows it, so a Save made for anything else
                // does not replace it.
                if (!cudaAvailable() && wb.backend() == Backend::Cpu && settings().getInt("compute/backend", 1) == 0)
                    backend_ = static_cast<int>(Backend::Cuda);
                // Likewise a stored GPU the cluster's job cannot give (the session then runs on its CPU).
                hpcDevice_ = static_cast<int>(wb.hpcDevice());
                if (!app.cluster().hpcGpuUsable() && storedHpcDevice() == HpcDevice::Gpu) hpcDevice_ = static_cast<int>(HpcDevice::Gpu);
                const int n = cudaDeviceCount();
                for (int i = 0; i < n; ++i) {
                    deviceNames_.push_back(deviceLabel(i));
                    deviceValues_.push_back(i);
                }
                if (n > 1) {
                    deviceNames_.push_back(format("All GPUs (%d) · round-robin volumes", n));
                    deviceValues_.push_back(Workbench::kAllCudaDevices);
                }
                if (n == 0) {
                    deviceNames_.emplace_back("no CUDA device");
                    deviceValues_.push_back(0);
                    deviceEnabled_ = false;
                }
                for (std::size_t i = 0; i < deviceValues_.size(); ++i)
                    if (deviceValues_[i] == wb.cudaDevice()) device_ = static_cast<int>(i);

                host_ = wb.remoteConfig().host;
                port_ = wb.remoteConfig().port;
                token_ = wb.remoteConfig().token;
                openedToken_ = token_;
                // Empty unless the user chose one: a default written back by Save
                // is a choice nobody made (see pyenv::workerInterpreter).
                python_ = settings().getString("worker/python");
                pythonTaken_ = python_;
                envPython_ = platform::environment("SIRIUS_PYTHON");
                hfToken_ = secrets::read("hub/token");
                openedHfToken_ = hfToken_;

                const AssistantSettings as = AssistantSettings::load();
                for (int i = 0; i < 3; ++i)
                    if (as.provider == kProviders[i]) provider_ = i;
                baseUrl_ = as.baseUrl;
                model_ = as.model;
                if (!as.model.empty()) modelItems_.push_back(as.model);
                // a key from the environment stays out of the field (and out of the store)
                apiKey_ = as.apiKeyVariable.empty() ? as.apiKey : std::string();
                openedApiKey_ = apiKey_;
                askFirst_ = as.askBeforeActing;

                // the tab last looked at comes back first (the assistant's settings
                // get revisited far more often than the compute ones)
                tab_ = std::clamp(settings().getInt("prefs/tab", 0), 0, 1);
                refreshModelList();   // the list on opening, without blocking the dialog

                useUv_ = settings().getBool("worker/useUv", true);
                offerEnvironment_ = settings().getBool("worker/offerEnvironment", true);
                offerTaken_ = offerEnvironment_;
                scriptDir_ = app.launcher().scriptDir();
                if (scriptDir_.empty()) scriptDir_ = settings().getString("worker/dir");
                if (scriptDir_.empty()) scriptDir_ = workerScriptPath();
                lookAtEnvironment();
            }

            // A Check still running is not waited for (checker_ hands its thread over).
            ~PreferencesDialog() override {
                const std::lock_guard<std::mutex> g(env_->mutex);
                env_->gone = true;
            }

            PreferencesDialog(const PreferencesDialog&) = delete;
            PreferencesDialog& operator=(const PreferencesDialog&) = delete;

            std::string title() const override { return "Preferences"; }
            ImVec2 size() const override { return ImVec2(560, 0); }

            void closed(App&) override { settings().set("prefs/tab", tab_); }

            void draw(App& app) override {
                widgets::vspace(2);
                widgets::tabRow("##prefsTabs", {"Compute", "Assistant"}, &tab_);
                // the rule the tabs stand on
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() - ImGui::GetStyle().ItemSpacing.y);
                widgets::rule(theme::kHairline);
                widgets::vspace(4);
                {
                    const Spacing spacing(8, 12);
                    if (tab_ == 0) drawCompute(app);
                    else drawAssistant();
                }
                widgets::vspace(4);
                // the one settings file, to read or edit as text
                widgets::rule(theme::kHairline);
                widgets::text("Settings file: " + settings().filePath(), 11, theme::kNeutral600);
                {
                    widgets::ButtonOpts b;
                    b.small = true;
                    b.tooltip = "Open sirius-app.toml in SIRIUS's editor: checked as you type, used as soon as it is saved";
                    // Preferences is modal: it closes first, the editor takes its place
                    if (widgets::button("Edit settings file\xE2\x80\xA6", b)) {
                        close();
                        app.defer([&app] { app.showDialog(makeSettingsEditor(app)); });
                    }
                    ImGui::SameLine(0.0f, px(8));
                    b.tooltip = "Show the folder of the settings file in the file manager";
                    if (widgets::button("Open settings folder", b)) platform::openInFileManager(settings().directory());
                }
                if (const std::string bad = settings().loadError(); !bad.empty())
                    note("The settings file does not read (" + bad + "): SIRIUS runs with its defaults and leaves the file as it is until it is "
                                                                     "fixed. Edit settings file\xE2\x80\xA6 says where.",
                         theme::kAccentText);
                widgets::vspace(6);
                switch (actionRow("Save", true)) {
                    case Action::Cancel: close(); break;
                    case Action::Accept:
                        apply(app);
                        close();
                        break;
                    case Action::None: break;
                }
            }

        private:
            std::string providerKey() const { return kProviders[std::clamp(provider_, 0, 2)]; }

            // Dialogs over this one change settings it shows: the setup's "Use
            // SIRIUS's environment instead" takes the configured interpreter
            // away, an offer's "Don't ask again" turns the offer off. A field
            // the user has not touched follows them, and Save writes only what
            // the user changed, so it never puts back what they took away.
            void followSettings() {
                if (python_ == pythonTaken_) python_ = pythonTaken_ = settings().getString("worker/python");
                if (offerEnvironment_ == offerTaken_)
                    offerEnvironment_ = offerTaken_ = settings().getBool("worker/offerEnvironment", true);
            }

            void drawCompute(App& app) {
                followSettings();
                {
                    const float w = columnWidth(2, 10);
                    widgets::FieldOpts fo;
                    fo.width = design(w);
                    {
                        const Field f("Default backend");
                        widgets::combo("##backend", &backend_, {"CUDA", "CPU", "HPC (remote worker)"}, fo);
                    }
                    ImGui::SameLine(0.0f, px(10));
                    const Field f("CUDA device");
                    widgets::FieldOpts dev = fo;
                    dev.enabled = deviceEnabled_;
                    widgets::combo("##device", &device_, deviceNames_, dev);
                }
                if (backend_ == static_cast<int>(Backend::Cuda) && !cudaAvailable())
                    note("No CUDA device is available in this build / machine: runs use the CPU until there is one.");
                if (backend_ == static_cast<int>(Backend::Hpc)) {
                    // where the HPC worker computes, sent with each step: no new job to switch
                    std::string why;
                    const bool gpu = app.cluster().hpcGpuUsable(&why);
                    const Field f("Cluster device");
                    widgets::SegmentedOpts so;
                    so.optionEnabled = {gpu, true};
                    so.tooltips = {gpu ? "Run the worker's steps on the job's GPU" : why,
                                   "Run the worker's steps on the job's CPU (the GPU stays allocated)"};
                    widgets::segmented("##hpcDevice", {"GPU", "CPU"}, &hpcDevice_, so);
                }
                widgets::rule(theme::kRule);
                widgets::caption("HPC worker");
                {
                    // the cluster session: one login, the job, the tunnel, all from here
                    ClusterLink& link = app.cluster();
                    ImU32 color = theme::kNeutral600;
                    std::string state = link.indicator(color);
                    if (state.empty()) state = "HPC: not connected to a cluster";
                    widgets::text(state, 12, color);
                    ImGui::SameLine(0.0f, px(10));
                    widgets::ButtonOpts b;
                    b.small = true;
                    b.tooltip = "Log in once, submit the worker job and connect through the SSH tunnel: no terminals";
                    // Preferences is modal: a dialog opened over it would get no input,
                    // so it closes first and the cluster dialog takes its place.
                    if (widgets::button("Connect to cluster\xE2\x80\xA6", b)) {
                        close();
                        app.defer([&app] { app.clusterDialog(); });
                    }
                    if (link.connected())
                        note("While the cluster session is connected the HPC backend goes through it; the fields below are for a worker "
                             "you started and tunnelled yourself.");
                }
                {
                    const float w = columnWidth(2, 10);
                    widgets::FieldOpts fo;
                    fo.width = design(w);
                    {
                        const Field f("Host");
                        widgets::inputText("##host", &host_, fo);
                    }
                    ImGui::SameLine(0.0f, px(10));
                    {
                        const Field f("Port");
                        widgets::inputInt("##port", &port_, 1, 65535, 1, fo);
                    }
                    const Field f("Token");
                    widgets::FieldOpts secret;
                    secret.password = true;
                    widgets::inputText("##token", &token_, secret);
                }
                note("By hand instead: launch the worker with app/python/slurm/sirius_worker.sbatch and forward its port "
                     "(ssh -L); see app/python/slurm/README.md. The token is kept for this session only, never saved.");
                widgets::rule(theme::kRule);
                {
                    const Field f("Python for the local worker");
                    widgets::FieldOpts fo;
                    fo.hint = pythonHint_;
                    widgets::inputText("##python", &python_, fo);
                    widgets::tooltip("Interpreter for the local worker. Empty = SIRIUS's own Python environment when it is "
                                     "set up, otherwise the first python3 on PATH. $SIRIUS_PYTHON, when set, overrides this "
                                     "field.");
                }
                drawEnvironment(app);
                {
                    const Field f("Hugging Face access token (optional)");
                    widgets::FieldOpts fo;
                    fo.password = true;
                    widgets::inputText("##hfToken", &hfToken_, fo);
                    widgets::tooltip("Access token for gated or private Hugging Face repositories (huggingface.co ▸ Settings ▸ "
                                     "Access Tokens); sent with each request that downloads a model");
                }
            }

            // --- SIRIUS's Python environment -------------------------------------------------
            //
            // Looked at on opening and after each action, never every frame:
            // the status reads the marker and the requirement files.

            void lookAtEnvironment() {
                try {
                    envStatus_ = pyenv::environmentStatus(scriptDir_, false);
                    envError_.clear();
                } catch (const std::exception& e) {
                    envStatus_ = pyenv::EnvironmentStatus();
                    envError_ = e.what();
                }
                pyenv::Interpreter unset;
                try {
                    unset = pyenv::workerInterpreter(std::string(), std::string());
                } catch (const std::exception&) {
                    // the hint falls back to what is on PATH
                }
                envDirExists_ = !envStatus_.dir.empty() && isDirectory(envStatus_.dir);
                found_ = platform::findPython();
                // what an empty field means, as the worker would pick it now
                const std::string version = envStatus_.marker ? envStatus_.marker->pythonVersion : std::string();
                if (!envPython_.empty()) pythonHint_ = envPython_ + " (from $SIRIUS_PYTHON)";
                else if (unset.source == pyenv::Source::Managed)
                    pythonHint_ = "SIRIUS environment" + (version.empty() ? std::string() : " \xC2\xB7 Python " + version);
                else if (!unset.path.empty()) pythonHint_ = unset.path;
                else pythonHint_ = found_.empty() ? std::string("python3") : found_;
            }

            // What the worker runs without the environment: the environment
            // variable, the field, the first Python on PATH.
            std::string interpreterWithout() const {
                if (!envPython_.empty()) return envPython_;
                if (std::string field = trimmed(python_); !field.empty()) return field;
                if (!found_.empty()) return found_;
#ifdef _WIN32
                return "python";
#else
                return "python3";
#endif
            }

            std::string problemOr(const char* otherwise) const {
                const std::string problem = withoutStop(envStatus_.problem);
                return problem.empty() ? std::string(otherwise) : problem;
            }

            std::string environmentLine() const {
                if (!envError_.empty()) return "Cannot tell: " + envError_;
                const pyenv::EnvironmentStatus& s = envStatus_;
                switch (s.state) {
                    case pyenv::State::Ready: {
                        std::string line = "Ready";
                        if (s.marker) {
                            if (!s.marker->pythonVersion.empty()) line += " \xC2\xB7 Python " + s.marker->pythonVersion;
                            for (const char* name : {"numpy", "scipy"}) {
                                const auto it = s.marker->packages.find(name);
                                const bool known = it != s.marker->packages.end() && !it->second.empty();
                                const std::string version = known ? it->second : std::string("\xE2\x80\x94");
                                line += std::string(" \xC2\xB7 ") + name + " " + version;
                            }
                        }
                        return line + " \xC2\xB7 " + s.dir;
                    }
                    case pyenv::State::Absent: return "Not set up. The worker runs " + interpreterWithout() + ".";
                    case pyenv::State::Incomplete: return "Incomplete (a setup did not finish).";
                    case pyenv::State::Outdated: return "Update needed: " + problemOr("the requirements changed") + ".";
                    case pyenv::State::Broken: return "Needs repair: " + problemOr("it does not run") + ".";
                }
                return std::string();
            }

            // Check's answer, or an action that changed the environment.
            void takeEnvironmentNews() {
                if (checking_) {
                    std::optional<pyenv::EnvironmentStatus> checked;
                    std::string error;
                    {
                        const std::lock_guard<std::mutex> g(env_->mutex);
                        checked.swap(env_->checked);
                        error = env_->checkError;
                    }
                    if (checked || !error.empty()) {
                        checking_ = false;
                        lookAtEnvironment();
                        if (checked) envStatus_ = *checked;   // with what running its Python found
                        if (!error.empty()) envError_ = error;
                    }
                }
                if (env_->stale.exchange(false)) lookAtEnvironment();
            }

            void drawEnvironment(App& app) {
                takeEnvironmentNews();
                const pyenv::EnvironmentStatus& s = envStatus_;
                // Closer together than the fields around it: the tab is tall
                // enough already, and these lines belong together.
                const Spacing section(6, 6);
                widgets::caption("SIRIUS's Python environment");
                widgets::textWrapped(checking_ ? std::string("Checking\xE2\x80\xA6") : environmentLine(), 12, theme::kText);
                if (s.state != pyenv::State::Absent && envError_.empty() && (!envPython_.empty() || !trimmed(python_).empty())) {
                    const char* who = envPython_.empty() ? "the field above" : "$SIRIUS_PYTHON";
                    note(std::string("Not used: ") + who + " names the interpreter.");
                }
                {
                    widgets::ButtonOpts o;
                    o.small = true;
                    const char* action = "Recreate\xE2\x80\xA6";
                    if (s.state == pyenv::State::Absent) action = "Set up\xE2\x80\xA6";
                    else if (s.state == pyenv::State::Outdated) action = "Update\xE2\x80\xA6";
                    else if (s.state == pyenv::State::Incomplete || s.state == pyenv::State::Broken)
                        action = "Repair\xE2\x80\xA6";
                    o.tooltip = "Say what would be downloaded and where, then do it";
                    if (widgets::button(action, o)) openSetup(app);
                    ImGui::SameLine();
                    o.enabled = s.state != pyenv::State::Absent && !checking_;
                    o.tooltip = "Run the environment's Python and see that the worker's packages import";
                    if (widgets::button("Check", o)) startCheck(app);
                    ImGui::SameLine();
                    o.enabled = s.state != pyenv::State::Absent && !app.bridge().busy();
                    o.tooltip = app.bridge().running()       ? std::string("Finish or cancel the run first")
                                : app.bridge().taskRunning() ? "Wait for " + app.bridge().taskLabel() + " to finish"
                                                             : std::string("Delete the environment's folder");
                    if (widgets::button("Remove", o)) askRemove(app);
                    ImGui::SameLine();
                    o.enabled = envDirExists_;
                    o.tooltip = s.dir;
                    if (widgets::button("Open folder", o)) platform::openInFileManager(s.dir);
                }
                widgets::checkbox("Use uv to download when it is installed", &useUv_);
                {
                    // the field below keeps its usual distance
                    const Spacing field(8, 12);
                    widgets::checkbox("Offer to set up Python when the worker cannot start", &offerEnvironment_);
                }
            }

            void openSetup(App& app) {
                PythonEnvRequest r;
                r.state = envStatus_.state;
                r.problem = envStatus_.problem;
                r.useUv = useUv_;   // as ticked here, saved or not
                const std::shared_ptr<EnvironmentShared> env = env_;
                r.finished = [env] { env->stale.store(true); };
                app.showDialog(makePythonEnvDialog(app, std::move(r)));
            }

            // `<envpy> -m sirius_worker --check` starts a Python: on a thread.
            void startCheck(App& app) {
                if (checking_) return;   // one at a time: start() joins the last, which has answered
                {
                    const std::lock_guard<std::mutex> g(env_->mutex);
                    env_->checked.reset();
                    env_->checkError.clear();
                }
                checking_ = true;
                const std::shared_ptr<EnvironmentShared> env = env_;
                const std::string scriptDir = scriptDir_;
                Bridge& bridge = app.bridge();   // only while the dialog is there (EnvironmentShared::gone)
                checker_.start([env, scriptDir, &bridge] {
                    std::optional<pyenv::EnvironmentStatus> status;
                    std::string error;
                    try {
                        status = pyenv::environmentStatus(scriptDir, true);
                    } catch (const std::exception& e) {
                        error = e.what();
                    }
                    const std::lock_guard<std::mutex> g(env->mutex);
                    env->checked = std::move(status);
                    env->checkError = error.empty() && !env->checked ? std::string("no answer") : error;
                    if (!env->gone) bridge.wake();
                });
            }

            void askRemove(App& app) {
                const std::string dir = envStatus_.dir;
                const std::shared_ptr<EnvironmentShared> env = env_;
                app.ask("Remove the Python environment",
                        "Remove SIRIUS's Python environment (" + dir + ")? The worker then runs " + interpreterWithout() +
                            " until it is set up again.",
                        {"Cancel", "Remove"}, [&app, dir, env](int answer) {
                            if (answer != 1) return;
                            WorkerLauncher& worker = app.launcher();
                            // What remove() did, in its words: a folder removed, one
                            // with files left for the next setup, or none there.
                            const auto outcome = std::make_shared<std::string>();
                            const bool started = app.bridge().startTask(
                                "Removing the Python environment",
                                [&worker, dir, env, outcome](const Bridge::TaskProgress&, const Bridge::TaskCancelled&) {
                                    // A worker running from it holds its files (on Windows the DLLs cannot go).
                                    worker.stop();
                                    const pyenv::SetupResult r = pyenv::remove(dir);
                                    env->stale.store(true);
                                    const std::string text = r.message + (r.hint.empty() ? std::string() : " " + r.hint);
                                    if (!r.ok) throw std::runtime_error(text);
                                    *outcome = text;   // read by the completion, after the task has ended
                                },
                                [&app, outcome] { app.wb().logLine("Python environment: " + continuing(*outcome)); });
                            if (!started) env->stale.store(true);
                        });
            }

            void drawAssistant() {
                const bool ollama = providerKey() == "ollama";
                {
                    const float w = columnWidth(2, 10);
                    {
                        const Field f("Provider");
                        widgets::FieldOpts fo;
                        fo.width = design(w);
                        if (widgets::combo("##provider", &provider_, {"Ollama (local)", "OpenRouter", "Custom OpenAI-compatible"}, fo)) {
                            const std::string p = providerKey();
                            if (p == "ollama") baseUrl_ = "http://localhost:11434/v1";
                            else if (p == "openrouter") baseUrl_ = "https://openrouter.ai/api/v1";
                            refreshModelList(false);
                        }
                    }
                    ImGui::SameLine(0.0f, px(10));
                    const Field f("Model");
                    const Spacing row(6, 8);
                    widgets::ButtonOpts refresh;
                    refresh.enabled = !fetching_;
                    refresh.tooltip = "Ask the server at the base URL which models it offers";
                    const float refreshW = buttonWidth("Refresh", widgets::ButtonKind::Secondary);
                    widgets::FieldOpts fo;
                    fo.width = design(std::max(px(60), ImGui::GetContentRegionAvail().x - refreshW - px(6)));
                    fo.hint = "e.g. llama3.1:8b or anthropic/claude-sonnet-4";
                    // a group, so the tool tip answers anywhere on the field
                    beginGroupAtTop();
                    widgets::editableCombo("##model", &model_, modelItems_, fo);
                    settleCursor();   // the dropdown puts the cursor back by hand as well
                    ImGui::EndGroup();
                    widgets::tooltip("The models the server lists (Refresh asks it again), or any name typed here");
                    ImGui::SameLine();
                    if (widgets::button("Refresh", refresh)) refreshModelList();
                }
                {
                    const Field f("Base URL");
                    widgets::inputText("##baseUrl", &baseUrl_);
                }
                if (!modelNote_.empty()) note(modelNote_);
                {
                    // The key field holds a stored or typed key; a key from the
                    // environment is shown as a placeholder naming its variable.
                    std::string variable;
                    AssistantSettings::environmentKey(providerKey(), &variable);
                    const Field f("API key");
                    widgets::FieldOpts fo;
                    fo.password = true;
                    fo.enabled = !ollama;
                    fo.hint = ollama ? std::string("Ollama takes no key")
                                     : (variable.empty() ? std::string() : "from $" + variable + " (not stored)");
                    ImGui::BeginGroup();
                    widgets::inputText("##apiKey", &apiKey_, fo);
                    ImGui::EndGroup();
                    if (ollama) widgets::tooltip("Never sent to Ollama; a stored key stays for OpenRouter and custom servers");
                }
                widgets::checkbox("Ask before acting (the assistant proposes, you confirm)", &askFirst_);
                note("Ollama needs a model with tool calling (ollama pull llama3.1). OpenRouter keys start with sk-or-; "
                     "the key is kept in the secret store and never sent to Ollama.");
            }

            std::string typedOrEnvironmentKey() const {
                if (!apiKey_.empty()) return apiKey_;
                return AssistantSettings::environmentKey(providerKey());
            }

            // The server's model list into the dropdown, keeping whatever is
            // typed; for Ollama the models held in memory are marked, since
            // the first answer from any other one waits for a load.
            // `fieldKey` false leaves the key field out: after a provider
            // switch it still holds the key meant for the previous server,
            // which must not reach the new one unasked (Refresh sends it).
            void refreshModelList(bool fieldKey = true) {
                const std::string base = trimmed(baseUrl_);
                const bool ollama = providerKey() == "ollama";
                std::string key;   // Ollama takes no key
                if (!ollama) key = fieldKey ? typedOrEnvironmentKey() : AssistantSettings::environmentKey(providerKey());
                listedModels_.clear();
                fetching_ = true;
                // A provider switched twice asks twice; only the last answer
                // may fill the list and the note.
                const int generation = ++generation_;
                modelNote_ = "Asking " + base + " for its models…";
                client_.fetchModels(base, key, [this, base, ollama, generation](std::vector<std::string> ids, std::string error) {
                    if (generation != generation_) return;
                    fetching_ = false;
                    if (ids.empty()) {
                        modelNote_ = error.empty() ? base + " lists no models."
                                                   : "Cannot list the models at " + base + " (" + error + "); type a name.";
                        return;
                    }
                    std::sort(ids.begin(), ids.end(), [](const std::string& a, const std::string& b) { return toLower(a) < toLower(b); });
                    listedModels_ = ids;
                    modelItems_ = ids;
                    // what is in the field now: a name typed while the server
                    // was answering is the user's choice
                    const std::string typed = trimmed(model_);
                    model_ = typed.empty() ? ids.front() : typed;
                    modelNote_ = format("%d model(s) at %s.", static_cast<int>(ids.size()), base.c_str());
                    if (!ollama) return;
                    client_.fetchLoadedModels(base, [this, generation](std::vector<std::string> loaded, std::string) {
                        if (generation != generation_) return;
                        if (loaded.empty()) {
                            modelNote_ += " None is loaded yet: the first answer waits for a load, a minute or two for a large model.";
                            return;
                        }
                        modelNote_ += " In memory now: " + join(loaded, ", ") + ".";
                    });
                });
            }

            void apply(App& app) {
                Workbench& wb = app.wb();
                Settings& s = settings();
                const int device = deviceValues_[static_cast<std::size_t>(std::clamp(device_, 0, static_cast<int>(deviceValues_.size()) - 1))];
                s.set("compute/backend", backend_);
                s.set("compute/cudaDevice", device);
                const HpcDevice hpcDevice = hpcDevice_ == static_cast<int>(HpcDevice::Cpu) ? HpcDevice::Cpu : HpcDevice::Gpu;
                s.set("compute/hpcDevice", std::string(hpcDevice == HpcDevice::Cpu ? "cpu" : "gpu"));
                s.set("hpc/host", trimmed(host_));
                s.set("hpc/port", static_cast<int>(port_));
                // Save writes only the secrets the user changed: rewriting an
                // untouched token into a store that refuses it is how a token that
                // still worked got lost, and a key that came from the environment
                // is not the user's to store.
                std::vector<std::string> notStored;
                // The HPC worker's token is not stored: it belongs to one worker job
                // (Connect to cluster makes a new one per session), and a stored
                // copy would only outlive the job while still opening it.
                // These two only when changed here (followSettings).
                if (python_ != pythonTaken_) {
                    if (const std::string python = trimmed(python_); python.empty()) s.remove("worker/python");
                    else s.set("worker/python", python);
                }
                s.set("worker/useUv", useUv_);
                if (offerEnvironment_ != offerTaken_) s.set("worker/offerEnvironment", offerEnvironment_);
                if (hfToken_ != openedHfToken_ && !secrets::write("hub/token", trimmed(hfToken_)))
                    notStored.emplace_back("the Hugging Face token");
                AssistantSettings as;
                as.provider = providerKey();
                as.baseUrl = trimmed(baseUrl_);
                as.model = LlmClient::resolveModel(model_, listedModels_);
                as.askBeforeActing = askFirst_;
                as.save();
                if (apiKey_ != openedApiKey_ && !AssistantSettings::storeApiKey(apiKey_)) notStored.emplace_back("the assistant's API key");
                wb.setBackend(usableBackend(backend_));
                wb.setCudaDevice(device);
                // a GPU the job does not have stays the stored default only
                if (hpcDevice == HpcDevice::Cpu || app.cluster().hpcGpuUsable()) wb.setHpcDevice(hpcDevice);
                RemoteConfig rc;
                rc.host = trimmed(host_);
                rc.port = static_cast<int>(port_);
                rc.token = token_;
                // a connected cluster session keeps the backend on its tunnel
                wb.setRemoteConfig(app.cluster().connected() ? app.cluster().remoteConfig() : rc);
                // The local worker's launcher reads "worker/python" itself each time
                // it starts the worker, after $SIRIUS_PYTHON: handing it the field
                // here would put the setting above the environment.
                app.assistant().setSettings(AssistantSettings::load());
                if (!notStored.empty())
                    app.message("Preferences",
                                "Could not store " + join(notStored, " and ") +
                                    " in the secret store: it will not be there at the next launch.",
                                MessageIcon::Warning);
            }

            int tab_ = 0;
            int backend_ = 0;
            int device_ = 0;
            int hpcDevice_ = 0;                       // HpcDevice: 0 GPU, 1 CPU
            std::vector<std::string> deviceNames_;
            std::vector<int> deviceValues_;
            bool deviceEnabled_ = true;
            std::string host_;
            std::int64_t port_ = 7645;
            std::string token_;
            std::string python_;
            std::string envPython_;
            std::string hfToken_;
            int provider_ = 0;
            std::string baseUrl_;
            std::string model_;                        // editable: the server's list, or any name typed
            std::vector<std::string> modelItems_;
            std::vector<std::string> listedModels_;   // what the server said it has, empty until it answers
            std::string modelNote_;
            std::string apiKey_;
            bool askFirst_ = false;
            // The secrets as the dialog opened with them.
            std::string openedToken_, openedHfToken_, openedApiKey_;
            bool fetching_ = false;                   // Refresh is disabled until the server answers
            int generation_ = 0;
            // SIRIUS's Python environment
            std::string scriptDir_;
            pyenv::EnvironmentStatus envStatus_;
            std::string envError_;                    // the status could not be read
            bool envDirExists_ = false;
            std::string found_;                       // the first Python on PATH
            std::string pythonHint_;                  // what an empty Python field means
            bool useUv_ = true, offerEnvironment_ = true;
            // The settings as the Python field and the offer's checkbox last took them.
            std::string pythonTaken_;
            bool offerTaken_ = true;
            bool checking_ = false;
            std::shared_ptr<EnvironmentShared> env_ = std::make_shared<EnvironmentShared>();
            DialogThread checker_;
            // Last, so its requests are cancelled before what their callbacks write to goes.
            LlmClient client_;
        };

    } // namespace

    std::shared_ptr<Dialog> makePreferencesDialog(App& app) { return std::make_shared<PreferencesDialog>(app); }

    void applyStoredPreferences(Workbench& wb) {
        const Settings& s = settings();
        wb.setBackend(usableBackend(s.getInt("compute/backend", cudaAvailable() ? 0 : 1)));
        wb.setCudaDevice(s.getInt("compute/cudaDevice", 0));
        wb.setHpcDevice(storedHpcDevice());
        RemoteConfig rc;
        rc.host = s.getString("hpc/host", "localhost");
        rc.port = s.getInt("hpc/port", 7645);
        // Earlier versions stored the token typed in Preferences; a token is
        // one worker job's, so a stored one is removed rather than reused.
        secrets::remove("hpc/token");
        wb.setRemoteConfig(rc);
    }

} // namespace sirius::app::gui
