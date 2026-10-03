// Preferences ▸ Edit settings file…: the settings file (sirius-app.toml) in
// the application's code editor. Every edit is checked as TOML (toml++ says
// where it stops) and against what the cluster profiles may hold
// (core/cluster_profiles.hpp), each problem with its line and column; Save
// is refused while an error is left, and otherwise writes the text as it is
// (atomically) and takes it as the settings at once (Settings::adoptText).
// No secret is in the file, so none is shown: the secret store keeps them.

#include "imgui/dialogs/dialogs.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include <imgui.h>
#include <nlohmann/json.hpp>

#include "core/cluster_profiles.hpp"
#include "core/settings_toml.hpp"
#include "imgui/dialogs/export_dialog_support.hpp"
#include "imgui/platform.hpp"
#include "imgui/settings.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/widgets/code_editor.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui {

    namespace {

        using namespace dialog_support;
        using theme::px;

        constexpr ImU32 kAmber = theme::rgb(0xb2, 0x6a, 0x00);

        // The file's text; the settings as SIRIUS would write them when there is no file yet.
        std::string currentText() {
            std::string text;
            if (platform::readFile(settings().filePath(), text)) return text;
            nlohmann::json flat = nlohmann::json::object();
            for (const std::string& k : settings().keys())
                if (k.rfind("secrets/", 0) != 0) flat[k] = settings().value(k);
            return settings_toml::toToml(flat);
        }

        class SettingsEditor final : public Dialog {
        public:
            SettingsEditor() {
                load();
                editor_.focus();
            }

            std::string title() const override { return "Settings file"; }
            ImVec2 size() const override { return ImVec2(780, 600); }
            bool resizable() const override { return true; }
            bool modal() const override { return false; }

            bool canClose(App& app) override {
                if (!modified_) return true;
                app.ask("Settings file", "Discard your changes to the settings file?", {"Keep editing", "Discard"}, [this](int answer) {
                    if (answer != 1) return;
                    modified_ = false;
                    close(); }, 0);
                return false;
            }

            void draw(App& app) override {
                const std::string path = settings().filePath();
                widgets::text(widgets::elideText(path, ImGui::GetContentRegionAvail().x, 11), 11, theme::kNeutral700);
                widgets::tooltip(path);
                if (widgets::linkButton("Open settings folder")) platform::openInFileManager(settings().directory());
                widgets::tooltip("Show the folder in the file manager: the settings file, imgui.ini (the window layout) and the files migrated from before");
                ImGui::SameLine(0.0f, px(10));
                if (widgets::linkButton("Reload from the file")) load();
                widgets::tooltip("Read the file again, dropping the edits here");
                widgets::textWrapped("TOML: [group] tables, key = value. Each change is checked as you type; Save writes the text as it is and SIRIUS uses it "
                                     "at once. Cluster profiles are the [cluster.<name>] tables (docs/clusters.example.toml explains their keys). "
                                     "Passwords and tokens are never in this file.",
                                     11, theme::kNeutral600);
                widgets::vspace(4);
                // the problems below the editor take what they need
                const float problemsH = problems_.empty() ? px(22) : std::min(px(130), px(18) * static_cast<float>(problems_.size()) + px(10));
                const float buttonsH = ImGui::GetFrameHeightWithSpacing() + px(30);
                const ImVec2 avail = ImGui::GetContentRegionAvail();
                const float editorH = std::max(px(120), avail.y - problemsH - buttonsH - px(8));
                if (editor_.draw("##settingsEditor", ImVec2(avail.x, editorH))) {
                    modified_ = true;
                    check();
                }
                widgets::vspace(4);
                ImGui::BeginChild("##problems", ImVec2(avail.x, problemsH), ImGuiChildFlags_None);
                if (problems_.empty()) {
                    widgets::text("No problems found.", 11, theme::rgb(0x2e, 0x7d, 0x32));
                } else {
                    for (const cluster::SettingsProblem& p : problems_) {
                        const std::string where = p.line > 0 ? "line " + std::to_string(p.line) + ", column " + std::to_string(p.column) + ": " : std::string();
                        widgets::textWrapped(std::string(p.error ? "Error \xC2\xB7 " : "Note \xC2\xB7 ") + where + p.message, 11,
                                             p.error ? theme::kAccentText : kAmber);
                    }
                }
                ImGui::EndChild();
                widgets::vspace(4);
                if (!saveError_.empty()) widgets::textWrapped(saveError_, 11, theme::kAccentText);
                const bool errors = std::any_of(problems_.begin(), problems_.end(), [](const cluster::SettingsProblem& p) { return p.error; });
                const Action a = actionRow("Save", modified_ && !errors);
                if (a == Action::Cancel) {
                    if (canClose(app)) close();
                } else if (a == Action::Accept) {
                    save(app);
                }
            }

        private:
            void load() {
                const std::string text = currentText();
                editor_.setText(text);
                modified_ = false;
                saveError_.clear();
                problems_ = cluster::checkSettingsText(text);
            }

            void check() { problems_ = cluster::checkSettingsText(editor_.text()); }

            void save(App& app) {
                const std::string text = editor_.text();
                check();
                if (std::any_of(problems_.begin(), problems_.end(), [](const cluster::SettingsProblem& p) { return p.error; })) return;
                std::string error;
                if (!settings().adoptText(text, &error)) {
                    saveError_ = "Not saved: " + error + ".";
                    return;
                }
                // what the workbench takes from the settings at start-up, again
                applyStoredPreferences(app.wb());
                app.wb().logLine("Settings: " + settings().filePath() + " saved and in use.");
                modified_ = false;
                close();
            }

            widgets::CodeEditor editor_;
            std::vector<cluster::SettingsProblem> problems_;
            bool modified_ = false;
            std::string saveError_;
        };

    } // namespace

    std::shared_ptr<Dialog> makeSettingsEditor(App&) { return std::make_shared<SettingsEditor>(); }

} // namespace sirius::app::gui
