// Models for the steps that need one: the local model cache, Hugging Face
// (search, the files of a repository, download into the cache) and the
// Foundation step's model folders (core/model_folder.hpp) under the models
// folders of this computer and of the cluster. The chosen model spec is what
// the step's "model" parameter accepts.
//
// The dialog asks the Python worker only for what the worker alone can
// answer:
//   * the cache is a directory on this machine and is read here
//     (model_hub_cache.hpp, the worker's layout and rules);
//   * Hugging Face is reached over http::Fetch, one Fetch per kind of request
//     (search, file list, download), with the access token as a bearer that
//     never follows a redirect to another host;
//   * this computer's model folders are read here (model.json is plain
//     JSON); the cluster's are listed by the worker beside the engine on the
//     node, the process that sees that filesystem, and so is what a
//     downloaded file holds (the one-line summary of model_info).
// So the dialog works, short of the cluster's models, on a machine whose
// worker does not start.
#include "imgui/dialogs/dialogs.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
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

#include "core/model_folder.hpp"
#include "core/ops/torch_model.hpp"
#include "core/remote_source.hpp"
#include "core/rpc.hpp"
#include "core/workbench.hpp"
#include "imgui/cluster_link.hpp"
#include "imgui/dialogs/export_dialog_support.hpp"
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
        // other, and their results come back through Bridge::post. Each call
        // has a connection of its own, closed when it is answered: the worker
        // serves one client at a time, and a connection kept open here would
        // hold a run, or the plugins loading on the GUI thread, until the hub
        // closed. Destroying the object cancels the call in flight, and a
        // handshake still waiting for the worker, and waits for the thread;
        // what was posted before finds `alive` cleared and does nothing.
        class HubWorker {
        public:
            // `unreachable`: the error is that no worker could be reached (it
            // did not start, or refused the handshake), not the call's.
            using Done = std::function<void(const nlohmann::json& result, const std::string& error, bool unreachable)>;

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
            // question (a models folder on the cluster).
            void call(const std::string& method, nlohmann::json params, const RemoteConfig* remote, Done done) {
                Job job;
                job.method = method;
                job.params = std::move(params);
                if (remote) {
                    job.useRemote = !remote->host.empty() || static_cast<bool>(remote->connect);
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
                const std::shared_ptr<Shared> shared = shared_;
                const std::function<bool()> cancelled = [shared] { return shared->cancel.load(); };
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
                    bool unreachable = false;
                    try {
                        // With `cancelled` the handshake has no deadline: it waits for a run that holds the
                        // worker, or for a worker's first answer, which imports torch, until the dialog closes.
                        // A worker that is only busy or slow is therefore never taken for an unreachable one.
                        std::unique_ptr<RemoteWorker> worker;
                        try {
                            // the configured endpoint as it connects: through the cluster
                            // session (its SOCKS proxy, a reconnect) when it has one
                            worker = job.useRemote ? job.remote.open(cancelled) : launcher_.connect(cancelled);
                        } catch (const std::exception&) {
                            unreachable = true;
                            throw;
                        }
                        // Cancelled only when the dialog is gone: nothing waits for the answer then.
                        worker->setCancelGrace(std::chrono::milliseconds(0));
                        const WorkerResult r = worker->call(job.method, job.params, {}, {}, cancelled);
                        text = r.result.dump();   // results cross threads as JSON text
                    } catch (const std::exception& e) {
                        error = e.what();
                        if (error.empty()) error = "the worker gave no reason";
                    }
                    bridge_.post([shared, done = std::move(job.done), text, error, unreachable] {
                        if (!shared->alive.load()) return;
                        if (!done) return;
                        if (!error.empty()) {
                            done(nlohmann::json(), error, unreachable);
                            return;
                        }
                        const nlohmann::json result = nlohmann::json::parse(text, nullptr, false);
                        if (result.is_discarded()) done(nlohmann::json(), "the worker's answer is not JSON", false);
                        else done(result, std::string(), false);
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
        // `withToken`: the request carried the stored access token, which
        // makes a 401 a rejected token rather than a repository that needs one.
        std::string hubError(const http::Response& r, const std::string& repo, bool withToken) {
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
            if (code == "RepoNotFound" || (r.status == 401 && !withToken))
                return repo + " was not found on Hugging Face (a private repository needs your access token).";
            // Hugging Face turns a bad bearer away on every endpoint, public ones and the search included,
            // so nothing succeeds until the token is replaced or cleared.
            if (r.status == 401)
                return "Hugging Face rejected the stored access token" + (text.empty() ? std::string() : " (" + text + ")") +
                       ": replace or clear it with Token….";
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

        // A one-line prompt with a password field.
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
        struct ModelRow {
            ModelFolderFacts facts;
            bool onCluster = false;
            std::string spec;   // what the step's Model becomes: the folder, or cluster://host/folder
        };

        enum Tab { kLocal = 0,
                   kHuggingFace = 1,
                   kModels = 2 };

        class ModelHubDialog final : public Dialog {
        public:
            ModelHubDialog(App& app, bool models, std::function<void(const std::string&)> chosen)
                : app_(app), accept_(std::move(chosen)), worker_(app.bridge(), app.launcher()) {
                tokenStored_ = !hubToken().empty();
                foldersText_ = join(storedFolders(), "; ");
                // the Local tab is up first
                listCache();
                if (models) {
                    tab_ = kModels;
                    // Listing the cluster's asks a worker, so it waits for the tab
                    // that needs it rather than happening when any tab is opened.
                    listModels();
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
                                     "Foundation step takes a model folder, under Models.",
                                     11, theme::kNeutral600);
                widgets::vspace(2);
                {
                    // the tabs sit on a hairline, as a document-mode tab bar does
                    const ImVec2 at = ImGui::GetCursorScreenPos();
                    const float w = ImGui::GetContentRegionAvail().x;
                    const float h = theme::snap(px(26));
                    ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(at.x, at.y + h - theme::crispPen(1)), ImVec2(at.x + w, at.y + h),
                                                              theme::kDivider);
                    if (widgets::tabRow("##tabs", {"Local", "Hugging Face", "Models"}, &tab_) && tab_ == kModels && models_.empty()) listModels();
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
                    else drawModels(app);
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
                // Deliberately not dialog_support::actionRow or buttonRow. This
                // pair is 84 px wide whatever its labels measure, it shares its
                // line with the elided "Model: ..." label whose width decides
                // where it starts, and its Enter test is this dialog's own
                // (enterUsed_, WantTextInput: the search field and the tables
                // take the key first). Either shared row would change the
                // widths, the narrow-window behaviour and the key handling, so
                // moving it would not be the behaviour-preserving promotion
                // this change is. The bar above IS shared.
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
                // OK is the default button: Enter presses it, unless a field took the key, or it answers a popup
                // above the hub (a message box, the token prompt). The hub is drawn before them, and their focus
                // counts as the hub's (the focus test follows the popup hierarchy), so the Enter that answered
                // "Delete model" also applied the chosen model and closed the hub under the box.
                const bool enter = ok.enabled && !enterUsed_ && !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId) &&
                                   ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && enterPressed() &&
                                   !ImGui::GetIO().WantTextInput && !ImGui::IsAnyItemActive();
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
            // here; a throw becomes a status line. During a run it may wait:
            // the worker serves one client at a time, and a run that uses it
            // keeps it until the run ends. The call goes ahead then by itself
            // (or is dropped when the hub closes); a run that does not use
            // that worker does not hold it up at all.
            void callWorker(const std::string& what, nlohmann::json params, const std::string& method,
                            std::function<void(const nlohmann::json&)> done, bool preferRemote = false) {
                setStatus(what + (app_.bridge().running() ? "… (after the run, if it is using the worker)" : "…"), false);
                const RemoteConfig remote = app_.wb().remoteConfig();
                worker_.call(method, std::move(params), preferRemote ? &remote : nullptr,
                             [this, what, done = std::move(done)](const nlohmann::json& result, const std::string& error, bool) {
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
                            "Choose model", std::string(), platform::parseFileFilters("Models (*.pt *.pts *.pth *.onnx);;All files (*)"));
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
                // Whether the token went along is fixed here: it may be changed while the request runs.
                const http::Request request = hubRequest(url);
                const bool withToken = !request.bearer.empty();
                http::Fetch::Handlers handlers;
                handlers.done = [this, what, q, withToken](const http::Response& r) {
                    if (!r.ok()) {
                        setStatus(what + " failed: " + hubError(r, q.empty() ? std::string("search") : q, withToken), true);
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
                searchFetch_.start(request, std::move(handlers));
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
                const http::Request request = hubRequest("https://huggingface.co/api/models/" + id + "?blobs=true");
                const bool withToken = !request.bearer.empty();
                http::Fetch::Handlers handlers;
                handlers.done = [this, what, id, withToken](const http::Response& r) {
                    if (id != repo_) return;
                    if (!r.ok()) {
                        setStatus(what + " failed: " + hubError(r, id, withToken), true);
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
                filesFetch_.start(request, std::move(handlers));
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
                // only a model file can be opened by the worker; a README would only "fail to load"
                if (files_[static_cast<std::size_t>(row)].model) describeCached(path);
            }

            // What the file holds, for the status line: only the worker can
            // open a model. The file is usable without the answer, so a worker
            // that does not start is not reported here (and not asked twice);
            // a model the worker cannot load is, since choosing it would fail
            // the step. Behind a run that uses the worker, the answer comes
            // when the run ends.
            void describeCached(const std::string& path) {
                if (workerFailed_) return;
                worker_.call("model_info", {{"path", path}, {"model", path}, {"spec", path}}, nullptr,
                             [this, path](const nlohmann::json& info, const std::string& error, bool unreachable) {
                                 if (unreachable) {
                                     workerFailed_ = true;
                                     return;
                                 }
                                 if (downloadedPath_ != path || status_ != "In the cache: " + path) return;
                                 if (!error.empty()) {
                                     setStatus("In the cache: " + path + " · cannot be loaded: " + error, true);
                                     return;
                                 }
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
                const bool withToken = !request.bearer.empty();
                handlers.done = [this, what, id, file, target, withToken](const http::Response& r) {
                    if (!r.ok()) {
                        progress_ = 0.0f;
                        // http::download could not create the file: a problem on this machine, not the network's
                        if (!r.cancelled && startsWith(r.error, "cannot write ")) {
                            setStatus(what + " failed: " + r.error + " (is the model cache " + modelhub::cacheDirectory() +
                                          " writable, with room to spare? SIRIUS_MODEL_CACHE chooses another)",
                                      true);
                            return;
                        }
                        http::Response answer = r;
                        if (answer.status >= 400) answer.error.clear();
                        setStatus(what + " failed: " + hubError(answer, id, withToken), true);
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
                // The bar is dialog_support's (export_dialog_support.hpp); this
                // one shares its line with Download and Use, so it is centred
                // in a row as high as those buttons.
                dialog_support::progressBar(progress_,
                                            std::max(px(40), ImGui::GetContentRegionAvail().x - downloadW - useW - 2 * gap8),
                                            buttonH);
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

            // --- the models: folders of model folders -----------------------------------

            // The models folders of this computer: sirius-app.toml's [models]
            // folders = [...] (a single string is read too).
            static std::vector<std::string> storedFolders() {
                std::vector<std::string> out = settings().getStringList("models/folders");
                if (out.empty()) {
                    const std::string one = settings().getString("models/folders");
                    if (!trimmed(one).empty()) out.push_back(one);
                }
                return out;
            }

            // The cluster's models folder: the current cluster profile's `models`.
            std::string clusterFolder() const { return trimmed(app_.cluster().storedProfile().models); }

            // This computer's folders are read here (model.json is plain JSON);
            // the cluster's by the worker beside the engine on the node, the
            // process that sees that filesystem.
            void listModels() {
                models_.clear();
                modelRow_ = -1;
                modelNote_.clear();
                std::vector<std::string> dirs;
                for (const std::string& d : split(foldersText_, ';', true))
                    if (!trimmed(d).empty()) dirs.push_back(trimmed(d));
                if (dirs != storedFolders()) settings().set("models/folders", dirs);
                const ModelListing here = listModelFolders(dirs);
                for (const ModelFolderFacts& f : here.models) models_.push_back({f, false, f.path});
                std::vector<std::string> notes = here.errors;
                const std::string remote = clusterFolder();
                const int generation = ++listGeneration_;
                if (!remote.empty()) {
                    if (app_.wb().remoteConfig().hasEngine()) {
                        const std::string host = app_.cluster().status().host;
                        callWorker(
                            "Listing the models in " + remote + " on the cluster", {{"dirs", {remote}}}, "list_bundles",
                            [this, host, generation](const nlohmann::json& r) {
                                if (generation != listGeneration_) return;   // a later listing replaced this one
                                const ModelListing there = modelListingFromJson(r);
                                for (const ModelFolderFacts& f : there.models) models_.push_back({f, true, makeClusterPath(host, f.path)});
                                for (const std::string& e : there.errors) modelNote_ += (modelNote_.empty() ? "" : "\n") + ("cluster: " + e);
                                if (models_.empty() && modelNote_.empty()) modelNote_ = "No models in these folders.";
                            },
                            true);
                    } else {
                        notes.push_back("The cluster's models folder (" + remote + ") is listed once you are connected to the cluster with "
                                                                                   "SIRIUS's engine.");
                    }
                }
                if (dirs.empty() && remote.empty())
                    notes.push_back("Give the folder your models are in (<models>/<name>/<version>/ with model.py and model.json), or set "
                                    "[models] folders = [...] in sirius-app.toml, and models = \"...\" in a cluster profile for the cluster's.");
                else if (models_.empty() && notes.empty() && remote.empty())
                    notes.push_back("No models in these folders.");
                modelNote_ = join(notes, "\n");
            }

            // What model.json says about the model now selected, under the table.
            void modelSelected() {
                if (modelRow_ < 0 || modelRow_ >= static_cast<int>(models_.size())) return;
                const ModelRow& row = models_[static_cast<std::size_t>(modelRow_)];
                const ModelFolderFacts& m = row.facts;
                if (!m.error.empty()) {
                    modelNote_ = m.error;
                    return;
                }
                std::vector<std::string> facts;
                if (m.voxelUm.size() == 3) facts.push_back(format("trained at %.3g x %.3g x %.3g um (z, y, x)", m.voxelUm[2], m.voxelUm[1], m.voxelUm[0]));
                facts.push_back(m.channels > 1 ? std::to_string(m.channels) + " channels" + (m.channelMerge.empty() ? std::string() : " (" + m.channelMerge + ")")
                                               : std::string("one channel"));
                facts.push_back(m.promptable ? "promptable" : "automatic only");
                std::string text = join(facts, " \xC2\xB7 ");
                if (!m.description.empty()) text += "\n" + m.description;
                text += "\n" + row.spec;
                modelNote_ = text;
            }

            void chooseSelectedModel() {
                if (modelRow_ < 0 || modelRow_ >= static_cast<int>(models_.size())) return;
                const ModelRow& row = models_[static_cast<std::size_t>(modelRow_)];
                if (row.facts.error.empty()) choose(row.spec);
            }

            void drawModels(App& app) {
                const float spacing = ImGui::GetStyle().ItemSpacing.y;
                const float fieldH = theme::snap(px(theme::kInputH));
                const float buttonH = smallButtonHeight();
                const float gap = px(6);

                // --- the folders row
                const float rowTop = ImGui::GetCursorPosY();
                const float browseW = px(84), refreshW = px(76);
                centreInRow(fieldH, theme::textSize("X", theme::kCaptionPx).y);
                widgets::caption("Folders");
                ImGui::SameLine(0.0f, gap);
                ImGui::SetCursorPosY(rowTop);
                widgets::FieldOpts field;
                field.width = (ImGui::GetContentRegionAvail().x - browseW - refreshW - 2 * gap) / scaleOrOne();
                field.hint = "models folders on this computer, separated by ;";
                field.enterReturnsTrue = true;
                bool refresh = widgets::inputText("##folders", &foldersText_, field);
                tip("This computer's models folders (sirius-app.toml: [models] folders = [...]). Each holds <name>/<version>/ model "
                    "folders, as latents scripts/export_model.py writes them. The cluster's is the cluster profile's models = \"...\".");
                if (refresh) enterUsed_ = true;
                ImGui::SameLine(0.0f, gap);
                ImGui::SetCursorPosY(rowTop);
                centreInRow(fieldH, buttonH);
                if (smallButton("Browse…##folders", ButtonKind::Secondary, 84, true, "Add a models folder of this computer")) {
                    app.defer(guarded([this] {
                        const std::string d = platform::pickFolderDialog("Models folder", app_.lastDir());
                        if (d.empty()) return;
                        foldersText_ = trimmed(foldersText_).empty() ? d : foldersText_ + "; " + d;
                        listModels();
                    }));
                }
                ImGui::SameLine(0.0f, gap);
                ImGui::SetCursorPosY(rowTop);
                centreInRow(fieldH, buttonH);
                if (smallButton("Refresh", ButtonKind::Secondary, 76)) refresh = true;
                ImGui::SetCursorPosY(rowTop + fieldH + spacing);
                if (refresh) listModels();
                if (const std::string remote = clusterFolder(); !remote.empty()) {
                    widgets::textWrapped("Cluster: " + remote + (app_.wb().remoteConfig().hasEngine() ? "" : " (listed once connected)"), 11,
                                         theme::kNeutral600);
                }

                // --- the models, what the selected one says, Use
                const float width = ImGui::GetContentRegionAvail().x;
                const float noteH = modelNote_.empty() ? 0.0f : wrappedHeight(modelNote_, 11, width) + spacing;
                const float tableH = ImGui::GetContentRegionAvail().y - noteH - buttonH - spacing;
                const Picked picked =
                    table("##models", {{"Model", 0.0f}, {"Version", 0.0f}, {"Tasks", 0.0f}, {"Where", 0.0f}, {"What it is", 1.0f}},
                          static_cast<int>(models_.size()), modelRow_, tableH, [this](int row, int column) {
                              const ModelRow& r = models_[static_cast<std::size_t>(row)];
                              const ModelFolderFacts& m = r.facts;
                              Cell c;
                              switch (column) {
                                  case 0:
                                      c.text = m.name;
                                      c.tip = r.spec;
                                      break;
                                  case 1: c.text = m.version; break;
                                  case 2: c.text = m.error.empty() ? join(m.tasks, ", ") : std::string("?"); break;
                                  case 3: c.text = r.onCluster ? "cluster" : "this computer"; break;
                                  default:
                                      c.text = m.error.empty() ? firstLine(m.description) : m.error;
                                      c.tip = m.error.empty() ? m.description : m.error;
                                      break;
                              }
                              return c;
                          });
                if (picked.clicked >= 0 && picked.clicked != modelRow_) {
                    modelRow_ = picked.clicked;
                    modelSelected();
                }
                if (!modelNote_.empty()) widgets::textWrapped(modelNote_, 11, theme::kNeutral600);

                const bool any = modelRow_ >= 0 && modelRow_ < static_cast<int>(models_.size()) &&
                                 models_[static_cast<std::size_t>(modelRow_)].facts.error.empty();
                const float useW = px(64);
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - useW));
                if (smallButton("Use##model", ButtonKind::Primary, 64, any)) chooseSelectedModel();
                if (picked.doubleClicked >= 0 && picked.doubleClicked == modelRow_ && any) {
                    chooseSelectedModel();
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

            // the Foundation step's model folders
            std::string foldersText_;   // this computer's models folders, "; " between them
            std::vector<ModelRow> models_;
            int modelRow_ = -1;
            std::string modelNote_;
            int listGeneration_ = 0;

            // Last, so that they go first: their callbacks use what is above.
            bool workerFailed_ = false;
            http::Fetch searchFetch_, filesFetch_, downloadFetch_;
            HubWorker worker_;
        };

    } // namespace

    std::shared_ptr<Dialog> makeModelHubDialog(App& app, bool models, std::function<void(const std::string&)> chosen) {
        return std::make_shared<ModelHubDialog>(app, models, std::move(chosen));
    }

} // namespace sirius::app::gui
