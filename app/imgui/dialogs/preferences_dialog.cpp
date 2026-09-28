// File ▸ Preferences…: default backend and CUDA device, the HPC worker
// connection, the Python interpreter for the local worker, and the
// assistant provider (Ollama / OpenRouter / custom OpenAI-compatible
// endpoint). Values live in the settings; the workbench is updated on Save.
// (app/qt/dialogs/preferences_dialog.cpp)

#include "imgui/dialogs/dialogs.hpp"

#include <algorithm>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>

#include <sirius/device.hpp>

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

        class PreferencesDialog : public Dialog {
        public:
            explicit PreferencesDialog(App& app) {
                const Workbench& wb = app.wb();
                backend_ = std::clamp(static_cast<int>(wb.backend()), 0, 2);
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
                // is a choice nobody made (see WorkerLauncher::python).
                python_ = settings().getString("worker/python");
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
            }

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
                    if (tab_ == 0) drawCompute();
                    else drawAssistant();
                }
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

            void drawCompute() {
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
                widgets::rule(theme::kRule);
                widgets::caption("HPC worker");
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
                note("Launch the worker on the cluster with app/python/slurm/sirius_worker.sbatch and forward its port "
                     "(ssh -L). The same worker runs Torch models locally.");
                widgets::rule(theme::kRule);
                {
                    const Field f("Python for the local worker");
                    widgets::FieldOpts fo;
                    fo.hint = envPython_.empty() ? std::string("python3") : envPython_ + " (from $SIRIUS_PYTHON)";
                    widgets::inputText("##python", &python_, fo);
                    widgets::tooltip(envPython_.empty() ? "Interpreter with numpy (and torch for segmentation); empty = python3. "
                                                          "$SIRIUS_PYTHON, when set, overrides this field"
                                                        : "Interpreter with numpy (and torch for segmentation). $SIRIUS_PYTHON is set "
                                                          "and overrides this field");
                }
                {
                    const Field f("Hugging Face access token (optional)");
                    widgets::FieldOpts fo;
                    fo.password = true;
                    widgets::inputText("##hfToken", &hfToken_, fo);
                    widgets::tooltip("Access token for gated or private Hugging Face repositories (huggingface.co ▸ Settings ▸ "
                                     "Access Tokens); sent with each request that downloads a model");
                }
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
                            refreshModelList();
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
            void refreshModelList() {
                const std::string keep = trimmed(model_);
                const std::string base = trimmed(baseUrl_);
                const bool ollama = providerKey() == "ollama";
                const std::string key = ollama ? std::string() : typedOrEnvironmentKey();   // Ollama takes no key
                listedModels_.clear();
                fetching_ = true;
                // A provider switched twice asks twice; only the last answer
                // may fill the list and the note.
                const int generation = ++generation_;
                modelNote_ = "Asking " + base + " for its models…";
                client_.fetchModels(base, key, [this, keep, base, ollama, generation](std::vector<std::string> ids, std::string error) {
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
                    model_ = keep.empty() ? ids.front() : keep;
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
                s.set("hpc/host", trimmed(host_));
                s.set("hpc/port", static_cast<int>(port_));
                // Save writes only the secrets the user changed: rewriting an
                // untouched token into a store that refuses it is how a token that
                // still worked got lost, and a key that came from the environment
                // is not the user's to store.
                std::vector<std::string> notStored;
                if (token_ != openedToken_ && !secrets::write("hpc/token", token_)) notStored.emplace_back("the HPC token");
                if (const std::string python = trimmed(python_); python.empty()) s.remove("worker/python");
                else s.set("worker/python", python);
                if (hfToken_ != openedHfToken_ && !secrets::write("hub/token", trimmed(hfToken_)))
                    notStored.emplace_back("the Hugging Face token");
                AssistantSettings as;
                as.provider = providerKey();
                as.baseUrl = trimmed(baseUrl_);
                as.model = LlmClient::resolveModel(model_, listedModels_);
                as.askBeforeActing = askFirst_;
                as.save();
                if (apiKey_ != openedApiKey_ && !AssistantSettings::storeApiKey(apiKey_)) notStored.emplace_back("the assistant's API key");
                wb.setBackend(static_cast<Backend>(backend_));
                wb.setCudaDevice(device);
                RemoteConfig rc;
                rc.host = trimmed(host_);
                rc.port = static_cast<int>(port_);
                rc.token = token_;
                wb.setRemoteConfig(rc);
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
            // Last, so its requests are cancelled before what their callbacks write to goes.
            LlmClient client_;
        };

    } // namespace

    std::shared_ptr<Dialog> makePreferencesDialog(App& app) { return std::make_shared<PreferencesDialog>(app); }

    void applyStoredPreferences(Workbench& wb) {
        const Settings& s = settings();
        const int backend = s.getInt("compute/backend", cudaAvailable() ? 0 : 1);
        wb.setBackend(static_cast<Backend>(std::max(0, std::min(backend, 2))));
        wb.setCudaDevice(s.getInt("compute/cudaDevice", 0));
        RemoteConfig rc;
        rc.host = s.getString("hpc/host", "localhost");
        rc.port = s.getInt("hpc/port", 7645);
        rc.token = secrets::read("hpc/token");
        wb.setRemoteConfig(rc);
    }

} // namespace sirius::app::gui
