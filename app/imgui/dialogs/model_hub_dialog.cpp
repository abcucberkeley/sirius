// Models for the steps that need one: the local model cache, Hugging Face
// (search, the files of a repository, download into the cache) and the
// registry of foundation bundles (.ltb). The chosen model spec is what the
// step's "model" parameter accepts. (app/qt/dialogs/model_hub_dialog.cpp)
//
// Where the Qt dialog asks the Python worker for everything, this one asks it
// only for what the worker alone can answer:
//   * the cache is a directory on this machine and is read here
//     (model_hub_cache.hpp, the worker's layout and rules);
//   * Hugging Face is reached over http::Fetch, one Fetch per kind of request
//     (search, file list, download), with the access token as a bearer that
//     never follows a redirect to another host;
//   * the bundles of a registry are listed by the worker, since on a cluster
//     the worker is what sees the filesystem they are on, and so is what a
//     downloaded file holds (the one-line summary of model_info).
// So the dialog works, short of the bundles, on a machine whose worker does
// not start.
#include "imgui/dialogs/dialogs.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <imgui.h>
#include <nlohmann/json.hpp>

#include "core/ops/torch_model.hpp"
#include "core/rpc.hpp"
#include "core/workbench.hpp"
#include "imgui/dialogs/model_hub_cache.hpp"
#include "imgui/http.hpp"
#include "imgui/platform.hpp"
#include "imgui/secret_store.hpp"
#include "imgui/settings.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui {

    namespace {

        using theme::px;
        using widgets::ButtonKind;

        // --- the worker -----------------------------------------------------------
        //
        // Calls to the Python worker block while it starts and while it
        // answers, so they run on a thread of the dialog's own, one after the
        // other, and their results come back through Bridge::post. The
        // connections belong to that thread. Destroying the object cancels the
        // call in flight and waits for the thread; what was posted before
        // finds `alive` cleared and does nothing.
        class HubWorker {
        public:
            using Done = std::function<void(const nlohmann::json& result, const std::string& error)>;

            HubWorker(Bridge& bridge, WorkerLauncher& launcher) : bridge_(bridge), launcher_(launcher) {}
            HubWorker(const HubWorker&) = delete;
            HubWorker& operator=(const HubWorker&) = delete;

            ~HubWorker() {
                shared_->alive.store(false);
                shared_->cancel.store(true);
                {
                    const std::lock_guard<std::mutex> g(mutex_);
                    quit_ = true;
                    jobs_.clear();
                }
                ready_.notify_all();
                if (thread_.joinable()) thread_.join();
            }

            // `remote` with a host: the configured HPC worker instead of the
            // local one, for the calls where "which machine" is the whole
            // question (a bundle registry is a directory on the cluster).
            void call(const std::string& method, nlohmann::json params, const RemoteConfig* remote, Done done) {
                Job job;
                job.method = method;
                job.params = std::move(params);
                if (remote) {
                    job.useRemote = !remote->host.empty();
                    job.remote = *remote;
                }
                job.done = std::move(done);
                {
                    const std::lock_guard<std::mutex> g(mutex_);
                    jobs_.push_back(std::move(job));
                }
                if (!thread_.joinable()) thread_ = std::thread([this] { loop(); });
                ready_.notify_all();
            }

        private:
            struct Job {
                std::string method;
                nlohmann::json params;
                bool useRemote = false;
                RemoteConfig remote;
                Done done;
            };
            struct Shared {
                std::atomic<bool> alive{true};
                std::atomic<bool> cancel{false};
            };

            void loop() {
                std::unique_ptr<RemoteWorker> local, hpc;
                for (;;) {
                    Job job;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        ready_.wait(lock, [this] { return quit_ || !jobs_.empty(); });
                        if (quit_) break;
                        job = std::move(jobs_.front());
                        jobs_.pop_front();
                    }
                    std::string text, error;
                    try {
                        RemoteWorker* worker = nullptr;
                        if (job.useRemote) {
                            if (!hpc || !hpc->isOpen()) hpc = RemoteWorker::connect(job.remote.host, job.remote.port, job.remote.token);
                            worker = hpc.get();
                        } else {
                            if (!local || !local->isOpen()) local = launcher_.connect();
                            worker = local.get();
                        }
                        const std::shared_ptr<Shared> shared = shared_;
                        const WorkerResult r = worker->call(job.method, job.params, {}, {}, [shared] { return shared->cancel.load(); });
                        text = r.result.dump();   // results cross threads as JSON text
                    } catch (const std::exception& e) {
                        error = e.what();
                        if (error.empty()) error = "the worker gave no reason";
                    }
                    bridge_.post([shared = shared_, done = std::move(job.done), text, error] {
                        if (!shared->alive.load()) return;
                        if (!done) return;
                        if (!error.empty()) {
                            done(nlohmann::json(), error);
                            return;
                        }
                        const nlohmann::json result = nlohmann::json::parse(text, nullptr, false);
                        if (result.is_discarded()) done(nlohmann::json(), "the worker's answer is not JSON");
                        else done(result, std::string());
                    });
                }
            }

            Bridge& bridge_;
            WorkerLauncher& launcher_;
            std::shared_ptr<Shared> shared_ = std::make_shared<Shared>();
            std::thread thread_;
            std::mutex mutex_;
            std::condition_variable ready_;
            std::deque<Job> jobs_;
            bool quit_ = false;
        };

        // --- text -------------------------------------------------------------------

        std::string countText(long long n) {
            if (n >= 1000000) return format("%.1fM", static_cast<double>(n) / 1e6);
            if (n >= 10000) return format("%.0fk", static_cast<double>(n) / 1e3);
            if (n >= 1000) return format("%.1fk", static_cast<double>(n) / 1e3);
            return std::to_string(n);
        }

        std::string sizeText(long long n) { return n < 0 ? std::string("?") : bytesText(static_cast<std::uint64_t>(n)); }

        std::string hubToken() { return trimmed(secrets::read("hub/token")); }

        std::string firstLine(const std::string& s) {
            const std::size_t end = s.find_first_of("\r\n");
            return trimmed(end == std::string::npos ? s : s.substr(0, end));
        }

        long long integerOf(const nlohmann::json& j, const char* key, long long fallback) {
            if (!j.is_object()) return fallback;
            const auto it = j.find(key);
            if (it == j.end() || !it->is_number()) return fallback;
            return it->is_number_float() ? static_cast<long long>(it->get<double>()) : it->get<long long>();
        }

        std::string stringOf(const nlohmann::json& j, const char* key) {
            if (!j.is_object()) return std::string();
            const auto it = j.find(key);
            return it != j.end() && it->is_string() ? it->get<std::string>() : std::string();
        }

        std::vector<double> numbersOf(const nlohmann::json& j, const char* key) {
            std::vector<double> out;
            if (!j.is_object()) return out;
            const auto it = j.find(key);
            if (it == j.end() || !it->is_array()) return out;
            for (const nlohmann::json& v : *it)
                if (v.is_number()) out.push_back(v.get<double>());
            return out;
        }

        // A Hugging Face failure as one sentence that says what to do (the
        // worker's _hub_error, from the answer's status and error headers).
        std::string hubError(const http::Response& r, const std::string& repo) {
            if (r.status == 0) return "Hugging Face is unreachable from this machine (" + r.message() + ").";
            const auto header = [&r](const char* name) {
                const auto it = r.headers.find(name);
                return it == r.headers.end() ? std::string() : it->second;
            };
            const std::string code = header("x-error-code");
            std::string text = header("x-error-message");
            if (text.empty()) {
                const nlohmann::json body = nlohmann::json::parse(r.body, nullptr, false);
                text = stringOf(body, "error");
            }
            text = firstLine(text);
            if (code == "GatedRepo" || containsNoCase(text, "gated"))
                return repo + " is a gated repository: sign in at https://huggingface.co/" + repo +
                       ", accept its terms, then paste an access token (Hugging Face settings > Access Tokens) into the hub's "
                       "Token field or Preferences > Compute.";
            if (code == "RepoNotFound" || r.status == 401)
                return repo + " was not found on Hugging Face (a private repository needs your access token).";
            if (code == "EntryNotFound" || code == "RevisionNotFound")
                return repo + ": no such file in the repository (" + (text.empty() ? r.message() : text) + ").";
            if (!r.error.empty()) return repo + ": " + r.error;
            return repo + ": " + (text.empty() ? r.message() : text);
        }

        // "owner/repo" + "sub dir/net.onnx" -> the URL the file is fetched from.
        std::string fileUrl(const std::string& repo, const std::string& file) {
            std::string url = "https://huggingface.co/" + repo + "/resolve/main";
            for (const std::string& part : split(file, '/', true)) url += "/" + http::urlEncode(part);
            return url;
        }

        // --- controls ----------------------------------------------------------------

        float scaleOrOne() { return std::max(theme::scale(), 0.01f); }

        // widgets::tooltip's look (12 px, ink border, after the usual delay),
        // also over a disabled button. Local because that one pops its font
        // after EndTooltip(), which Dear ImGui reports as an error box.
        void tip(const std::string& s) {
            if (s.empty() || !ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip | ImGuiHoveredFlags_AllowWhenDisabled)) return;
            ImGui::PushStyleColor(ImGuiCol_Border, theme::kText);
            ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, theme::crispPen(theme::kBorder));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, px(8, 4));
            if (ImGui::BeginTooltip()) {
                {
                    const theme::FontScope f(12);
                    ImGui::PushStyleColor(ImGuiCol_Text, theme::kText);
                    ImGui::PushTextWrapPos(px(360));
                    ImGui::TextUnformatted(s.c_str(), s.c_str() + s.size());
                    ImGui::PopTextWrapPos();
                    ImGui::PopStyleColor();
                }
                ImGui::EndTooltip();
            }
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor();
        }

        // The height of a small button (widgets::button's own arithmetic).
        float smallButtonHeight() {
            const float text = theme::textSize("Xg", 12, theme::Weight::SemiBold).y;
            return theme::snap(std::max(px(14), text) + 2 * px(4) + 2 * px(theme::kBorder));
        }

        bool smallButton(const char* label, ButtonKind kind, float width, bool enabled = true, const std::string& tooltip = {}) {
            widgets::ButtonOpts o;
            o.kind = kind;
            o.small = true;
            o.width = width;
            o.enabled = enabled;
            o.centered = true;
            const bool pressed = widgets::button(label, o);
            tip(tooltip);
            return pressed;
        }

        // Moves the cursor down so that an item `itemH` high sits in the middle of a row `rowH` high.
        void centreInRow(float rowH, float itemH) {
            if (itemH < rowH) ImGui::SetCursorPosY(ImGui::GetCursorPosY() + std::floor((rowH - itemH) * 0.5f));
        }

        float wrappedHeight(const std::string& s, float designPx, float width) {
            const theme::FontScope f(designPx);
            return ImGui::CalcTextSize(s.c_str(), s.c_str() + s.size(), false, std::max(1.0f, width)).y;
        }

        bool enterPressed() { return ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false); }

        // The 8 px bar of a download: a groove, the accent up to `fraction`.
        void progressBar(float fraction, float width, float rowH) {
            const ImVec2 at = ImGui::GetCursorScreenPos();
            const float h = theme::snap(px(8));
            const float y = theme::snap(at.y + (rowH - h) * 0.5f);
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(ImVec2(at.x, y), ImVec2(at.x + width, y + h), theme::kNeutral300);
            const float f = std::clamp(fraction, 0.0f, 1.0f);
            if (f > 0.0f) dl->AddRectFilled(ImVec2(at.x, y), ImVec2(at.x + theme::snap(width * f), y + h), theme::kAccent);
            ImGui::Dummy(ImVec2(width, rowH));
        }

        // --- tables ------------------------------------------------------------------
        //
        // Rows that select as a whole, no grid, headers in caption case. The
        // stretch columns cut their text with an ellipsis (the tooltip says
        // the rest); the others are as wide as what they hold.

        struct Column {
            const char* title;
            float stretch;   // 0: as wide as its content
        };
        struct Cell {
            std::string text;
            ImU32 color = theme::kText;
            std::string tip;
        };
        struct Picked {
            int clicked = -1;
            int doubleClicked = -1;
        };

        Picked table(const char* id, const std::vector<Column>& columns, int rows, int selected, float height,
                     const std::function<Cell(int row, int column)>& cellAt) {
            Picked out;
            const float rowH = theme::snap(px(26));
            const float headerH = theme::snap(px(24));
            const float padX = px(8);
            ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(padX, 0.0f));
            const ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_NoSavedSettings;
            if (ImGui::BeginTable(id, static_cast<int>(columns.size()), flags, ImVec2(0.0f, std::max(height, headerH + rowH)))) {
                ImGui::TableSetupScrollFreeze(0, 1);
                for (const Column& c : columns)
                    ImGui::TableSetupColumn(c.title, c.stretch > 0.0f ? ImGuiTableColumnFlags_WidthStretch : ImGuiTableColumnFlags_WidthFixed,
                                            c.stretch);
                const float captionH = theme::textSize("X", theme::kCaptionPx).y;
                ImGui::TableNextRow(ImGuiTableRowFlags_None, headerH);
                for (int c = 0; c < static_cast<int>(columns.size()); ++c) {
                    ImGui::TableSetColumnIndex(c);
                    const ImVec2 at = ImGui::GetCursorScreenPos();
                    const float w = ImGui::GetContentRegionAvail().x;
                    ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(at.x - padX, at.y + headerH - theme::crispPen(1)),
                                                              ImVec2(at.x + w + padX, at.y + headerH), theme::kDivider);
                    ImGui::SetCursorScreenPos(ImVec2(at.x, at.y + std::floor((headerH - captionH) * 0.5f)));
                    widgets::caption(columns[static_cast<std::size_t>(c)].title);
                }
                const float textH = theme::textSize("Xg", theme::kBodyPx).y;
                ImGuiListClipper clipper;
                clipper.Begin(rows, rowH);
                while (clipper.Step()) {
                    for (int r = clipper.DisplayStart; r < clipper.DisplayEnd; ++r) {
                        ImGui::TableNextRow(ImGuiTableRowFlags_None, rowH);
                        ImGui::PushID(r);
                        for (int c = 0; c < static_cast<int>(columns.size()); ++c) {
                            ImGui::TableSetColumnIndex(c);
                            const ImVec2 at = ImGui::GetCursorScreenPos();
                            if (c == 0) {
                                const ImGuiSelectableFlags sf = ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick |
                                                                ImGuiSelectableFlags_AllowOverlap;
                                if (ImGui::Selectable("##row", r == selected, sf, ImVec2(0.0f, rowH))) out.clicked = r;
                                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) out.doubleClicked = r;
                            }
                            const Cell cell = cellAt(r, c);
                            ImGui::SetCursorScreenPos(ImVec2(at.x, at.y + std::floor((rowH - textH) * 0.5f)));
                            if (columns[static_cast<std::size_t>(c)].stretch > 0.0f) {
                                const std::string shown = widgets::elideText(cell.text, ImGui::GetContentRegionAvail().x, theme::kBodyPx);
                                widgets::text(shown, theme::kBodyPx, cell.color);
                                tip(!cell.tip.empty() ? cell.tip : shown != cell.text ? cell.text
                                                                                      : std::string());
                            } else {
                                widgets::text(cell.text, theme::kBodyPx, cell.color);
                                tip(cell.tip);
                            }
                        }
                        ImGui::PopID();
                    }
                }
                ImGui::EndTable();
            }
            ImGui::PopStyleVar();
            return out;
        }

        // --- the token ---------------------------------------------------------------

        // QInputDialog::getText with a password field.
        class TokenPrompt final : public Dialog {
        public:
            TokenPrompt(std::string initial, std::function<void(const std::string&)> accepted)
                : value_(std::move(initial)), accepted_(std::move(accepted)) {}

            std::string title() const override { return "Hugging Face access token"; }
            ImVec2 size() const override { return ImVec2(460, 0); }

            void draw(App&) override {
                widgets::textWrapped("Token for gated or private repositories (huggingface.co ▸ Settings ▸ Access Tokens).\n"
                                     "Kept in the secret store and sent with each request that downloads a model.",
                                     12, theme::kText);
                widgets::vspace(2);
                if (first_) ImGui::SetKeyboardFocusHere();
                first_ = false;
                widgets::FieldOpts f;
                f.password = true;
                f.enterReturnsTrue = true;
                const bool enter = widgets::inputText("##token", &value_, f);
                widgets::vspace(8);
                const float w = px(84);
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - 2 * w - px(8));
                widgets::ButtonOpts cancel;
                cancel.width = 84;
                cancel.centered = true;
                if (widgets::button("Cancel", cancel)) close();
                ImGui::SameLine(0.0f, px(8));
                widgets::ButtonOpts ok;
                ok.kind = ButtonKind::Primary;
                ok.width = 84;
                ok.centered = true;
                if (widgets::button("OK", ok) || enter) {
                    ok_ = true;
                    close();
                }
            }

            void closed(App&) override {
                if (ok_ && accepted_) accepted_(value_);
            }

        private:
            std::string value_;
            std::function<void(const std::string&)> accepted_;
            bool first_ = true;
            bool ok_ = false;
        };

        // --- the dialog ----------------------------------------------------------------

        struct HubModel {
            std::string id;
            bool gated = false;
            long long downloads = 0, likes = 0;
            std::vector<std::string> tags;
        };
        struct HubFile {
            std::string name;
            long long size = -1;
            bool model = false;
        };
        struct Bundle {
            std::string name, path, task, voxel, threshold, size;
            nlohmann::json facts;   // what the worker said about it
        };

        enum Tab { kLocal = 0,
                   kHuggingFace = 1,
                   kBundles = 2 };

        class ModelHubDialog final : public Dialog {
        public:
            ModelHubDialog(App& app, bool bundles, std::function<void(const std::string&)> chosen)
                : app_(app), accept_(std::move(chosen)), worker_(app.bridge(), app.launcher()) {
                tokenStored_ = !hubToken().empty();
                registryDir_ = storedRegistry();
                // the Local tab is up first
                listCache();
                if (bundles) {
                    tab_ = kBundles;
                    // Listing spawns a worker, so it waits for the tab that needs it
                    // rather than happening when any tab is opened.
                    if (!trimmed(registryDir_).empty()) listBundles();
                }
            }

            ~ModelHubDialog() override {
                *alive_ = false;
                // the requests in flight are cancelled by their Fetch, the
                // worker's call by the HubWorker: no callback arrives after this
            }

            std::string title() const override { return "Model hub"; }
            ImVec2 size() const override { return ImVec2(720, 660); }
            bool resizable() const override { return true; }

            void draw(App& app) override {
                enterUsed_ = false;
                widgets::textWrapped("Models for the step that runs them. Segmentation takes a TorchScript / ONNX file from Hugging "
                                     "Face, one already in the cache, or a file on this machine; a package family the worker provides "
                                     "(cellpose:cpsam, microsam:vit_b_lm) can be typed straight into the step's Model field. The "
                                     "foundation model takes a bundle from the registry, under Bundles.",
                                     11, theme::kNeutral600);
                widgets::vspace(2);
                {
                    // the tabs sit on a hairline, as a document-mode tab bar does
                    const ImVec2 at = ImGui::GetCursorScreenPos();
                    const float w = ImGui::GetContentRegionAvail().x;
                    const float h = theme::snap(px(26));
                    ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(at.x, at.y + h - theme::crispPen(1)), ImVec2(at.x + w, at.y + h),
                                                              theme::kDivider);
                    widgets::tabRow("##tabs", {"Local", "Hugging Face", "Bundles"}, &tab_);
                }

                // what stays under the tabs: the status, the rule, the chosen model and the buttons
                const float spacing = ImGui::GetStyle().ItemSpacing.y;
                const float width = ImGui::GetContentRegionAvail().x;
                const float statusFull = wrappedHeight(statusShown_, 11, width);
                const float statusH = std::min(statusFull, theme::snap(px(84)));
                const float buttonsH = theme::snap(px(35));
                // The rest of the footer as the last frame measured it (an estimate on the first).
                const float restH = footerRestH_ > 0.0f ? footerRestH_ : 3 * spacing + theme::crispPen(theme::kRule) + buttonsH;
                const float pageH = std::max(px(120), ImGui::GetContentRegionAvail().y - statusH - restH);

                if (ImGui::BeginChild("##page", ImVec2(0.0f, pageH), ImGuiChildFlags_None,
                                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
                    widgets::vspace(4);   // with the item spacing, the 12 px above a page
                    if (tab_ == kLocal) drawLocal(app);
                    else if (tab_ == kHuggingFace) drawHuggingFace(app);
                    else drawBundles(app);
                }
                ImGui::EndChild();
                const float footerTop = ImGui::GetCursorPosY();

                // A long error (the worker's stderr) scrolls rather than push the buttons out.
                if (ImGui::BeginChild("##status", ImVec2(0.0f, statusH), ImGuiChildFlags_None,
                                      statusFull > statusH ? ImGuiWindowFlags_None : ImGuiWindowFlags_NoScrollbar)) {
                    widgets::textWrapped(statusShown_, 11, statusError_ ? theme::kAccentText : theme::kNeutral600);
                    if (statusShown_ != status_) tip(status_);
                }
                ImGui::EndChild();
                widgets::rule(theme::kRule);

                const float buttonW = px(84), gap = px(8);
                const float labelW = std::max(px(40), ImGui::GetContentRegionAvail().x - 2 * buttonW - 2 * gap);
                {
                    const float textH = theme::textSize("Xg", 12).y;
                    const ImVec2 at = ImGui::GetCursorPos();
                    ImGui::SetCursorPosY(at.y + std::floor((buttonsH - textH) * 0.5f));
                    widgets::elided(chosen_.empty() ? std::string("No model chosen") : "Model: " + chosen_, labelW, 12, theme::kText);
                    ImGui::SameLine(0.0f, 0.0f);
                    ImGui::SetCursorPos(ImVec2(at.x + labelW + gap, at.y));
                }
                widgets::ButtonOpts cancel;
                cancel.kind = ButtonKind::Ghost;
                cancel.width = 84;
                cancel.centered = true;
                if (widgets::button("Cancel", cancel)) close();
                ImGui::SameLine(0.0f, gap);
                widgets::ButtonOpts ok;
                ok.kind = ButtonKind::Primary;
                ok.width = 84;
                ok.centered = true;
                ok.enabled = !chosen_.empty();
                // OK is the default button: Enter presses it, unless a field took the key
                const bool enter = ok.enabled && !enterUsed_ && enterPressed() && !ImGui::GetIO().WantTextInput && !ImGui::IsAnyItemActive();
                if (widgets::button("OK", ok) || enter) accept();
                footerRestH_ = ImGui::GetCursorPosY() - footerTop - statusH;
            }

        private:
            // --- outcome ---------------------------------------------------------------

            void choose(const std::string& spec) { chosen_ = spec; }

            void accept() {
                if (chosen_.empty()) return;
                const std::string spec = chosen_;
                if (accept_) accept_(spec);
                close();
            }

            void setStatus(const std::string& text, bool error) {
                status_ = text;
                statusError_ = error;
                // A worker that did not start reports its whole traceback. The
                // line says what failed and the last line of the traceback why;
                // the rest is in the tool tip.
                std::vector<std::string> lines;
                for (const std::string& line : split(text, '\n', true))
                    if (!trimmed(line).empty()) lines.push_back(line);
                statusShown_ = lines.size() > 2 ? trimmed(lines.front()) + "\n" + trimmed(lines.back()) : text;
            }

            // A callback that outlives the dialog does nothing.
            template <class F>
            auto guarded(F fn) {
                return [alive = alive_, fn = std::move(fn)](auto&&... args) {
                    if (*alive) fn(std::forward<decltype(args)>(args)...);
                };
            }

            // --- the worker ------------------------------------------------------------

            // Calls `method` on the worker's thread and `done(result)` back
            // here; a throw becomes a status line.
            void callWorker(const std::string& what, nlohmann::json params, const std::string& method,
                            std::function<void(const nlohmann::json&)> done, bool preferRemote = false) {
                setStatus(what + "…", false);
                const RemoteConfig remote = app_.wb().remoteConfig();
                worker_.call(method, std::move(params), preferRemote ? &remote : nullptr,
                             [this, what, done = std::move(done)](const nlohmann::json& result, const std::string& error) {
                                 if (!error.empty()) {
                                     setStatus(what + " failed: " + error, true);
                                     return;
                                 }
                                 setStatus(std::string(), false);
                                 try {
                                     done(result);
                                 } catch (const std::exception& e) {
                                     setStatus(what + ": " + e.what(), true);
                                 }
                             });
            }

            // --- the local cache ---------------------------------------------------------

            void listCache() {
                const std::string selectedPath = cacheRow_ >= 0 && cacheRow_ < static_cast<int>(cache_.size())
                                                     ? cache_[static_cast<std::size_t>(cacheRow_)].path
                                                     : std::string();
                cache_ = modelhub::listCachedModels();
                cacheRow_ = -1;
                for (std::size_t i = 0; i < cache_.size(); ++i)
                    if (cache_[i].path == selectedPath) cacheRow_ = static_cast<int>(i);
                cacheNote_ = "Cache: " + modelhub::cacheDirectory() + " (SIRIUS_MODEL_CACHE overrides)";
            }

            static std::string cacheName(const modelhub::CachedModel& m) { return startsWith(m.spec, "hf:") ? m.spec : fileName(m.path); }

            // Only what the hub itself downloaded: a path outside the cache is
            // refused, since a model chosen from elsewhere on the machine is
            // the user's file and not ours to remove.
            void removeCached(const modelhub::CachedModel& model) {
                const std::string path = model.path, name = cacheName(model), size = sizeText(static_cast<long long>(model.bytes));
                app_.defer(guarded([this, path, name, size] {
                    // Cancel is the last button, and so the default: deleting is not the safe one
                    app_.ask("Delete model",
                             "Delete " + name + " from the model cache?\n\n" + path + " (" + size +
                                 ") is removed from this machine. Any step still pointing at it will download it again, or fail if "
                                 "it came from elsewhere.",
                             {"Delete", "Cancel"}, guarded([this, path, name](int answer) {
                                 if (answer != 0) return;
                                 try {
                                     const modelhub::Deleted gone = modelhub::deleteCachedModel(path);
                                     setStatus("Deleted " + name + " · " + sizeText(static_cast<long long>(gone.bytes)) + " freed", false);
                                     // the chosen model may be the one that just went
                                     if (chosen_ == gone.path || chosen_ == path) choose(std::string());
                                     if (downloadedPath_ == gone.path || downloadedPath_ == path) {
                                         downloadedPath_.clear();
                                         progress_ = 0.0f;
                                     }
                                 } catch (const std::exception& e) {
                                     setStatus("Deleting " + name + " failed: " + e.what(), true);
                                 }
                                 listCache();
                             }));
                }));
            }

            void drawLocal(App& app) {
                const float spacing = ImGui::GetStyle().ItemSpacing.y;
                widgets::caption("Model cache");
                const float width = ImGui::GetContentRegionAvail().x;
                const float noteH = wrappedHeight(cacheNote_, 11, width);
                const float buttonH = smallButtonHeight();
                const float tableH = ImGui::GetContentRegionAvail().y - noteH - buttonH - 2 * spacing;
                const Picked picked = table("##cache", {{"Model", 0.0f}, {"Path", 1.0f}, {"Size", 0.0f}}, static_cast<int>(cache_.size()),
                                            cacheRow_, tableH, [this](int row, int column) {
                                                const modelhub::CachedModel& m = cache_[static_cast<std::size_t>(row)];
                                                Cell c;
                                                c.text = column == 0   ? cacheName(m)
                                                         : column == 1 ? m.path
                                                                       : sizeText(static_cast<long long>(m.bytes));
                                                return c;
                                            });
                if (picked.clicked >= 0) cacheRow_ = picked.clicked;
                widgets::textWrapped(cacheNote_, 11, theme::kNeutral600);
                tip("Everything the hub has downloaded. Delete frees the disk it uses.");

                const bool any = cacheRow_ >= 0 && cacheRow_ < static_cast<int>(cache_.size());
                if (smallButton("Browse…##local", ButtonKind::Secondary, 84)) {
                    app.defer(guarded([this] {
                        const std::string f = platform::openFileDialog(
                            "Choose model", std::string(), platform::filtersFromQt("Models (*.pt *.pts *.pth *.onnx);;All files (*)"));
                        if (!f.empty()) choose(f);
                    }));
                }
                ImGui::SameLine(0.0f, 0.0f);
                const float deleteW = px(70), useW = px(64), gap = px(8);
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - deleteW - useW - gap));
                if (smallButton("Delete##local", ButtonKind::Ghost, 70, any, "Remove the selected model from the cache on this machine"))
                    removeCached(cache_[static_cast<std::size_t>(cacheRow_)]);
                ImGui::SameLine(0.0f, gap);
                if (smallButton("Use##local", ButtonKind::Primary, 64, any)) choose(cache_[static_cast<std::size_t>(cacheRow_)].path);
                if (picked.doubleClicked >= 0 && picked.doubleClicked < static_cast<int>(cache_.size())) {
                    choose(cache_[static_cast<std::size_t>(picked.doubleClicked)].path);
                    accept();
                }
            }

            // --- Hugging Face -------------------------------------------------------------

            http::Request hubRequest(const std::string& url) const {
                http::Request request;
                request.url = url;
                request.headers.emplace_back("Accept", "application/json");
                request.timeoutSeconds = 60;
                // gated / private repositories: the access token from the secret store
                request.bearer = hubToken();
                return request;
            }

            void runSearch() {
                const std::string q = trimmed(query_);
                results_.clear();
                resultRow_ = -1;
                files_.clear();
                fileRow_ = -1;
                repo_.clear();
                downloadedPath_.clear();
                filesFetch_.cancel();
                const std::string what = "Searching Hugging Face";
                setStatus(what + "…", false);
                // sorted by downloads; the fields a search omits unless asked for
                std::string url = "https://huggingface.co/api/models?limit=40&sort=downloads&direction=-1";
                if (!q.empty()) url += "&search=" + http::urlEncode(q);
                for (const char* field : {"gated", "private", "downloads", "likes", "tags", "pipeline_tag", "lastModified", "library_name"})
                    url += std::string("&expand%5B%5D=") + field;
                http::Fetch::Handlers handlers;
                handlers.done = [this, what, q](const http::Response& r) {
                    if (!r.ok()) {
                        setStatus(what + " failed: " + hubError(r, q.empty() ? std::string("search") : q), true);
                        return;
                    }
                    const nlohmann::json models = nlohmann::json::parse(r.body, nullptr, false);
                    if (!models.is_array()) {
                        setStatus(what + ": the answer is not a list of models", true);
                        return;
                    }
                    results_.clear();
                    resultRow_ = -1;
                    for (const nlohmann::json& m : models) {
                        if (!m.is_object()) continue;
                        HubModel model;
                        model.id = stringOf(m, "id");
                        if (model.id.empty()) model.id = stringOf(m, "modelId");
                        if (model.id.empty()) continue;
                        // "auto" / "manual" for a repository whose terms must be accepted, else false
                        const auto gated = m.find("gated");
                        model.gated = gated != m.end() && !gated->is_null() && !(gated->is_boolean() && !gated->get<bool>());
                        model.downloads = integerOf(m, "downloads", 0);
                        model.likes = integerOf(m, "likes", 0);
                        const auto tags = m.find("tags");
                        if (tags != m.end() && tags->is_array())
                            for (const nlohmann::json& t : *tags)
                                if (t.is_string() && model.tags.size() < 8) model.tags.push_back(t.get<std::string>());
                        results_.push_back(std::move(model));
                    }
                    setStatus(results_.empty() ? std::string("No models found.") : std::to_string(results_.size()) + " models", false);
                };
                searchFetch_.start(hubRequest(url), std::move(handlers));
            }

            void listFiles(const std::string& id, bool gated) {
                if (id == repo_) return;
                repo_ = id;
                repoGated_ = gated;
                files_.clear();
                fileRow_ = -1;
                downloadedPath_.clear();
                const std::string what = "Listing files of " + id;
                setStatus(what + "…", false);
                http::Fetch::Handlers handlers;
                handlers.done = [this, what, id](const http::Response& r) {
                    if (id != repo_) return;
                    if (!r.ok()) {
                        setStatus(what + " failed: " + hubError(r, id), true);
                        return;
                    }
                    const nlohmann::json info = nlohmann::json::parse(r.body, nullptr, false);
                    if (!info.is_object()) {
                        setStatus(what + ": the answer is not a repository", true);
                        return;
                    }
                    setStatus(std::string(), false);
                    files_.clear();
                    const auto siblings = info.find("siblings");
                    if (siblings != info.end() && siblings->is_array())
                        for (const nlohmann::json& s : *siblings) {
                            HubFile f;
                            f.name = stringOf(s, "rfilename");
                            if (f.name.empty()) continue;
                            f.size = integerOf(s, "size", -1);
                            f.model = modelhub::isModelFile(f.name);
                            files_.push_back(std::move(f));
                        }
                    // the model files first, each group by name
                    std::sort(files_.begin(), files_.end(), [](const HubFile& a, const HubFile& b) {
                        if (a.model != b.model) return a.model;
                        return a.name < b.name;
                    });
                    const bool anyModel = !files_.empty() && files_.front().model;
                    fileNote_ = anyModel ? "Model files (.pt, .pts, .pth, .onnx) are highlighted; the rest is shown for reference."
                                         : "This repository has no TorchScript / ONNX file; SIRIUS cannot run its weights directly.";
                    if (repoGated_)
                        fileNote_ += " Gated repository: accept its terms at https://huggingface.co/" + id +
                                     " while signed in, then add your access token with Token….";
                    fileRow_ = -1;
                    if (anyModel) selectFile(0);
                };
                filesFetch_.start(hubRequest("https://huggingface.co/api/models/" + id + "?blobs=true"), std::move(handlers));
            }

            bool downloading() const { return downloadFetch_.busy(); }

            void selectFile(int row) {
                fileRow_ = row;
                downloadedPath_.clear();
                progress_ = 0.0f;
                if (row < 0 || row >= static_cast<int>(files_.size())) return;
                // already in the cache? then "Use" is available right away
                const std::string file = files_[static_cast<std::size_t>(row)].name;
                const std::string path = modelhub::cachedPath(repo_, file);
                if (path.empty()) return;
                downloadedPath_ = path;
                progress_ = 1.0f;
                setStatus("In the cache: " + path, false);
                describeCached(path);
            }

            // What the file holds, for the status line: only the worker can
            // open a model. The file is usable without the answer, so a worker
            // that does not start is not reported here (and not asked twice).
            void describeCached(const std::string& path) {
                if (workerFailed_) return;
                worker_.call("model_info", {{"path", path}, {"model", path}, {"spec", path}}, nullptr,
                             [this, path](const nlohmann::json& info, const std::string& error) {
                                 if (!error.empty()) {
                                     workerFailed_ = true;
                                     return;
                                 }
                                 if (downloadedPath_ != path || status_ != "In the cache: " + path) return;
                                 setStatus("In the cache: " + path + " · " + torchModelSummary(info), false);
                             });
            }

            void startDownload() {
                if (fileRow_ < 0 || fileRow_ >= static_cast<int>(files_.size()) || repo_.empty() || downloading()) return;
                const std::string file = files_[static_cast<std::size_t>(fileRow_)].name;
                const std::string id = repo_;
                const std::string what = "Downloading " + file;
                progress_ = 0.0f;
                downloadedPath_.clear();
                // files already present are not fetched again
                const std::string have = modelhub::cachedPath(id, file);
                if (!have.empty()) {
                    downloaded(id, file, have);
                    return;
                }
                std::string target;
                try {
                    // the name comes from the network: it must not reach outside the repository's directory
                    target = modelhub::downloadTarget(id, file);
                } catch (const std::exception& e) {
                    setStatus(what + " failed: " + e.what(), true);
                    return;
                }
                setStatus(what + "… downloading " + file, false);
                http::Request request = hubRequest(fileUrl(id, file));
                request.headers.clear();
                request.headers.emplace_back("Accept-Encoding", "identity");   // the size is the size on the wire
                request.timeoutSeconds = 0;
                request.stallSeconds = 10;   // a connection that delivers nothing is given up
                http::Fetch::Handlers handlers;
                handlers.onProgress = [this, what, file](double received, double total) {
                    progress_ = total > 0.0 ? static_cast<float>(std::min(0.99, received / total)) : 0.0f;
                    const double mb = 1024.0 * 1024.0;
                    setStatus(what + "… " + file + ": " +
                                  (total > 0.0 ? format("%.0f / %.0f MB", received / mb, total / mb) : format("%.0f MB", received / mb)),
                              false);
                };
                handlers.done = [this, what, id, file, target](const http::Response& r) {
                    if (!r.ok()) {
                        progress_ = 0.0f;
                        http::Response answer = r;
                        if (answer.status >= 400) answer.error.clear();
                        setStatus(what + " failed: " + hubError(answer, id), true);
                        return;
                    }
                    downloaded(id, file, target);
                };
                downloadFetch_.start(request, std::move(handlers), target);
            }

            void downloaded(const std::string& id, const std::string& file, const std::string& path) {
                const bool selected = id == repo_ && fileRow_ >= 0 && fileRow_ < static_cast<int>(files_.size()) &&
                                      files_[static_cast<std::size_t>(fileRow_)].name == file;
                if (selected) {
                    progress_ = 1.0f;
                    downloadedPath_ = path;
                }
                listCache();   // the download belongs in the Local tab straight away
                long long bytes = -1;
                for (const modelhub::CachedModel& m : cache_)
                    if (m.path == path) bytes = static_cast<long long>(m.bytes);
                if (bytes < 0)
                    for (const HubFile& f : files_)
                        if (id == repo_ && f.name == file) bytes = f.size;
                setStatus("Downloaded " + file + " (" + sizeText(bytes) + ") to " + path, false);
            }

            void cancelDownload() {
                if (!downloading()) return;
                downloadFetch_.cancel();   // what arrived so far is removed with the partial file
                progress_ = 0.0f;
                setStatus("Download cancelled.", false);
            }

            void askToken() {
                app_.defer(guarded([this] {
                    app_.showDialog(std::make_shared<TokenPrompt>(hubToken(), guarded([this](const std::string& typed) {
                                                                      const std::string token = trimmed(typed);
                                                                      const bool stored = secrets::write("hub/token", token);
                                                                      tokenStored_ = !token.empty() && stored;
                                                                      setStatus(!stored         ? "The token could not be stored (secret store refused); it is not kept."
                                                                                : token.empty() ? "Token cleared."
                                                                                                : "Token stored.",
                                                                                !stored);
                                                                  })));
                }));
            }

            void drawHuggingFace(App&) {
                const float spacing = ImGui::GetStyle().ItemSpacing.y;
                const float fieldH = theme::snap(px(theme::kInputH));
                const float buttonH = smallButtonHeight();
                const float gap = px(6);

                // --- the search row
                const float searchW = px(72), tokenW = px(82);
                const float rowTop = ImGui::GetCursorPosY();
                widgets::FieldOpts field;
                field.width = (ImGui::GetContentRegionAvail().x - searchW - tokenW - 2 * gap) / scaleOrOne();
                field.hint = "search models… (e.g. nuclei segmentation 3d, cellpose, unet)";
                field.enterReturnsTrue = true;
                bool search = widgets::inputText("##query", &query_, field);
                if (search) enterUsed_ = true;
                ImGui::SameLine(0.0f, gap);
                ImGui::SetCursorPosY(rowTop);
                centreInRow(fieldH, buttonH);
                if (smallButton("Search", ButtonKind::Secondary, 72)) search = true;
                ImGui::SameLine(0.0f, gap);
                ImGui::SetCursorPosY(rowTop);
                centreInRow(fieldH, buttonH);
                if (smallButton(tokenStored_ ? "Token ✓##token" : "Token…##token", ButtonKind::Ghost, 82, true,
                                "Hugging Face access token for gated or private repositories"))
                    askToken();
                ImGui::SetCursorPosY(rowTop + fieldH + spacing);
                if (search) runSearch();

                // --- results over files, two to one
                const float width = ImGui::GetContentRegionAvail().x;
                const float captionH = theme::textSize("X", theme::kCaptionPx).y;
                const float noteH = wrappedHeight(fileNote_, 11, width);
                const float flexible = ImGui::GetContentRegionAvail().y - captionH - buttonH - noteH - 4 * spacing;
                const float resultsH = std::floor(std::max(px(80), flexible * 0.6f));
                const float filesH = std::floor(std::max(px(60), flexible - resultsH));

                const Picked model =
                    table("##results", {{"Model", 1.3f}, {"Downloads", 0.0f}, {"Likes", 0.0f}, {"Tags", 1.0f}}, static_cast<int>(results_.size()),
                          resultRow_, resultsH, [this](int row, int column) {
                              const HubModel& m = results_[static_cast<std::size_t>(row)];
                              Cell c;
                              switch (column) {
                                  case 0:
                                      c.text = m.gated ? m.id + "  (gated)" : m.id;
                                      c.tip = m.gated ? m.id + "\nGated: accept the terms on huggingface.co while signed in, "
                                                               "then add your access token (Token…)"
                                                      : m.id;
                                      if (m.gated) c.color = theme::kAccentText;
                                      break;
                                  case 1: c.text = countText(m.downloads); break;
                                  case 2: c.text = countText(m.likes); break;
                                  default:
                                      c.text = join(m.tags, ", ");
                                      c.tip = join(m.tags, "\n");
                                      break;
                              }
                              return c;
                          });
                if (model.clicked >= 0 && model.clicked < static_cast<int>(results_.size())) {
                    resultRow_ = model.clicked;
                    const HubModel& m = results_[static_cast<std::size_t>(model.clicked)];
                    listFiles(m.id, m.gated);
                }

                widgets::caption("Files");
                const Picked file = table("##files", {{"File", 1.0f}, {"Size", 0.0f}}, static_cast<int>(files_.size()), fileRow_, filesH,
                                          [this](int row, int column) {
                                              const HubFile& f = files_[static_cast<std::size_t>(row)];
                                              Cell c;
                                              c.text = column == 0 ? f.name : sizeText(f.size);
                                              if (column == 0 && !f.model) c.color = theme::kNeutral500;
                                              return c;
                                          });
                if (file.clicked >= 0 && file.clicked != fileRow_) selectFile(file.clicked);

                // --- progress, Download, Use
                const bool anyFile = fileRow_ >= 0 && fileRow_ < static_cast<int>(files_.size());
                const bool busy = downloading();
                const float downloadW = px(88), useW = px(64), gap8 = px(8);
                progressBar(progress_, std::max(px(40), ImGui::GetContentRegionAvail().x - downloadW - useW - 2 * gap8), buttonH);
                ImGui::SameLine(0.0f, gap8);
                bool download = false;
                if (busy) {
                    // the one way to stop a download short of closing the dialog
                    if (smallButton("Cancel##download", ButtonKind::Secondary, 88, true, "Stop the download; what arrived is removed"))
                        cancelDownload();
                } else if (smallButton("Download", ButtonKind::Secondary, 88, anyFile)) {
                    download = true;
                }
                ImGui::SameLine(0.0f, gap8);
                if (smallButton("Use##file", ButtonKind::Primary, 64, !downloadedPath_.empty() && !busy)) choose(downloadedPath_);
                if (file.doubleClicked >= 0 && file.doubleClicked == fileRow_) {
                    if (!downloadedPath_.empty()) {
                        choose(downloadedPath_);
                        accept();
                    } else if (!busy) {
                        download = true;
                    }
                }
                if (download) startDownload();
                widgets::textWrapped(fileNote_, 11, theme::kNeutral600);
            }

            // --- the foundation-bundle registry ------------------------------------------

            // Where the bundles are. Remembered, and seeded from the environment so
            // that a cluster deployment can point every user at one directory
            // instead of each of them having to find it.
            static std::string storedRegistry() {
                const std::string saved = settings().getString("foundation/registry");
                if (!saved.empty()) return saved;
                return platform::environment("SIRIUS_BUNDLE_REGISTRY");
            }

            // Listed by the worker, not by this process: on a cluster the worker is
            // what can see the filesystem the bundles are on, and it is also what
            // reads a manifest out of one.
            void listBundles() {
                const std::string dir = trimmed(registryDir_);
                bundles_.clear();
                bundleRow_ = -1;
                if (dir.empty()) {
                    bundleNote_ = "Give the directory the .ltb bundles are in. On a cluster that is a directory the worker can read, "
                                  "which need not be one this machine can.";
                    return;
                }
                settings().set("foundation/registry", dir);
                bundleNote_.clear();
                const bool onHpc = app_.wb().backend() == Backend::Hpc;
                callWorker(
                    "Listing bundles in " + dir, {{"dir", dir}}, "list_bundles", [this, dir](const nlohmann::json& r) { fillBundles(dir, r); },
                    onHpc);
            }

            void fillBundles(const std::string& dir, const nlohmann::json& r) {
                bundles_.clear();
                bundleRow_ = -1;
                const auto list = r.find("bundles");
                if (list == r.end() || !list->is_array() || list->empty()) {
                    bundleNote_ = "No .ltb bundles in " + dir + ".";
                    return;
                }
                for (const nlohmann::json& b : *list) {
                    if (!b.is_object()) continue;
                    Bundle bundle;
                    bundle.facts = b;
                    bundle.path = stringOf(b, "path");
                    bundle.name = stringOf(b, "name");
                    bundle.task = stringOf(b, "task");
                    if (bundle.task.empty()) bundle.task = "?";
                    for (double v : numbersOf(b, "voxel_um")) bundle.voxel += (bundle.voxel.empty() ? "" : " x ") + format("%.3g", v);
                    // The unit goes in the cells: a caption-case header would turn the micro sign into a capital mu.
                    bundle.voxel = bundle.voxel.empty() ? std::string("?") : bundle.voxel + " µm";
                    const auto threshold = b.find("peak_threshold");
                    bundle.threshold = threshold != b.end() && threshold->is_number() ? format("%.3g", threshold->get<double>()) : std::string("?");
                    bundle.size = sizeText(integerOf(b, "size_bytes", 0));
                    bundles_.push_back(std::move(bundle));
                }
                bundleNote_ = std::to_string(bundles_.size()) + " bundle(s) in " + dir +
                              ". A '?' is a bundle whose manifest could not be read: it can still be chosen, but the step cannot "
                              "default to the thresholds it was validated at.";
            }

            // What the manifest says about the bundle now selected, under the table.
            void bundleSelected() {
                if (bundleRow_ < 0 || bundleRow_ >= static_cast<int>(bundles_.size())) return;
                const Bundle& bundle = bundles_[static_cast<std::size_t>(bundleRow_)];
                const nlohmann::json& b = bundle.facts;
                std::vector<std::string> facts;
                const auto separation = b.find("min_separation_um");
                if (separation != b.end() && separation->is_number()) facts.push_back("min. separation " + format("%.3g", separation->get<double>()) + " um");
                const std::vector<double> patch = numbersOf(b, "patch");
                if (patch.size() == 3) facts.push_back(format("patch %.3g x %.3g x %.3g", patch[0], patch[1], patch[2]));
                const auto channels = b.find("channels");
                if (channels != b.end() && channels->is_array() && !channels->empty()) {
                    std::vector<std::string> names;
                    for (const nlohmann::json& c : *channels) names.push_back(c.is_string() ? c.get<std::string>() : c.dump());
                    facts.push_back("channels " + join(names, ", "));
                }
                std::string text = join(facts, " · ");
                const std::string notes = stringOf(b, "notes");
                if (!notes.empty()) text += (text.empty() ? "" : "\n") + notes;
                const auto manifest = b.find("manifest");
                if (manifest == b.end() || !manifest->is_boolean() || !manifest->get<bool>()) text = "The manifest could not be read. " + text;
                bundleNote_ = text.empty() ? bundle.path : text;
            }

            void chooseSelectedBundle() {
                if (bundleRow_ < 0 || bundleRow_ >= static_cast<int>(bundles_.size())) return;
                choose(bundles_[static_cast<std::size_t>(bundleRow_)].path);
            }

            void drawBundles(App& app) {
                const float spacing = ImGui::GetStyle().ItemSpacing.y;
                const float fieldH = theme::snap(px(theme::kInputH));
                const float buttonH = smallButtonHeight();
                const float gap = px(6);

                // --- the registry row
                const float rowTop = ImGui::GetCursorPosY();
                const float browseW = px(84), refreshW = px(76);
                centreInRow(fieldH, theme::textSize("X", theme::kCaptionPx).y);
                widgets::caption("Registry");
                ImGui::SameLine(0.0f, gap);
                ImGui::SetCursorPosY(rowTop);
                widgets::FieldOpts field;
                field.width = (ImGui::GetContentRegionAvail().x - browseW - refreshW - 2 * gap) / scaleOrOne();
                field.hint = "directory of .ltb bundles, as the worker sees it";
                field.enterReturnsTrue = true;
                bool refresh = widgets::inputText("##registry", &registryDir_, field);
                tip("Where the bundles are. Read by the worker, so on a cluster this is a path on the cluster. "
                    "SIRIUS_BUNDLE_REGISTRY sets the default.");
                if (refresh) enterUsed_ = true;
                ImGui::SameLine(0.0f, gap);
                ImGui::SetCursorPosY(rowTop);
                centreInRow(fieldH, buttonH);
                if (smallButton("Browse…##registry", ButtonKind::Secondary, 84, true, "Only useful when the worker runs on this machine")) {
                    app.defer(guarded([this] {
                        const std::string d = platform::pickFolderDialog("Bundle registry", registryDir_);
                        if (d.empty()) return;
                        registryDir_ = d;
                        listBundles();
                    }));
                }
                ImGui::SameLine(0.0f, gap);
                ImGui::SetCursorPosY(rowTop);
                centreInRow(fieldH, buttonH);
                if (smallButton("Refresh", ButtonKind::Secondary, 76)) refresh = true;
                ImGui::SetCursorPosY(rowTop + fieldH + spacing);
                if (refresh) listBundles();

                // --- the bundles, what the selected one says, Use
                const float width = ImGui::GetContentRegionAvail().x;
                // no note, no line for it (a failed listing leaves none)
                const float noteH = bundleNote_.empty() ? 0.0f : wrappedHeight(bundleNote_, 11, width) + spacing;
                const float tableH = ImGui::GetContentRegionAvail().y - noteH - buttonH - spacing;
                const Picked picked =
                    table("##bundles", {{"Bundle", 1.0f}, {"Task", 0.0f}, {"Voxel", 0.0f}, {"Threshold", 0.0f}, {"Size", 0.0f}},
                          static_cast<int>(bundles_.size()), bundleRow_, tableH, [this](int row, int column) {
                              const Bundle& b = bundles_[static_cast<std::size_t>(row)];
                              Cell c;
                              switch (column) {
                                  case 0:
                                      c.text = b.name;
                                      c.tip = b.path;
                                      break;
                                  case 1: c.text = b.task; break;
                                  case 2: c.text = b.voxel; break;
                                  case 3: c.text = b.threshold; break;
                                  default: c.text = b.size; break;
                              }
                              return c;
                          });
                if (picked.clicked >= 0 && picked.clicked != bundleRow_) {
                    bundleRow_ = picked.clicked;
                    bundleSelected();
                }
                if (!bundleNote_.empty()) widgets::textWrapped(bundleNote_, 11, theme::kNeutral600);

                const bool any = bundleRow_ >= 0 && bundleRow_ < static_cast<int>(bundles_.size());
                const float useW = px(64);
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - useW));
                if (smallButton("Use##bundle", ButtonKind::Primary, 64, any)) chooseSelectedBundle();
                if (picked.doubleClicked >= 0 && picked.doubleClicked == bundleRow_ && any) {
                    chooseSelectedBundle();
                    accept();
                }
            }

            App& app_;
            std::function<void(const std::string&)> accept_;
            std::shared_ptr<bool> alive_ = std::make_shared<bool>(true);

            int tab_ = kLocal;
            std::string chosen_;
            std::string status_;
            std::string statusShown_;   // status_, a traceback cut to its first and last line
            bool statusError_ = false;
            bool enterUsed_ = false;   // a field took this frame's Enter
            float footerRestH_ = 0.0f;   // below the status line, display px, measured

            // the local cache
            std::vector<modelhub::CachedModel> cache_;
            int cacheRow_ = -1;
            std::string cacheNote_;

            // Hugging Face
            std::string query_;
            bool tokenStored_ = false;
            std::vector<HubModel> results_;
            int resultRow_ = -1;
            std::vector<HubFile> files_;
            int fileRow_ = -1;
            std::string repo_;
            bool repoGated_ = false;
            std::string downloadedPath_;   // of the selected file, once it is in the cache
            float progress_ = 0.0f;
            std::string fileNote_ = "Downloads land in $SIRIUS_MODEL_CACHE or ~/.sirius/models.";

            // the foundation-bundle registry
            std::string registryDir_;
            std::vector<Bundle> bundles_;
            int bundleRow_ = -1;
            std::string bundleNote_;

            // Last, so that they go first: their callbacks use what is above.
            bool workerFailed_ = false;
            http::Fetch searchFetch_, filesFetch_, downloadFetch_;
            HubWorker worker_;
        };

    } // namespace

    std::shared_ptr<Dialog> makeModelHubDialog(App& app, bool bundles, std::function<void(const std::string&)> chosen) {
        return std::make_shared<ModelHubDialog>(app, bundles, std::move(chosen));
    }

} // namespace sirius::app::gui
