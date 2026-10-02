// Process ▸ Connect to cluster…: the profile (SSH host, the SIRIUS checkout
// and Python environment on the cluster, the Slurm partition, account, QoS,
// time and resources), Connect, and the steps it takes as a checklist with
// what each found -- the queue's state and wait, the node, the worker's own
// account of itself -- or, when one fails, why in plain words, the cluster's
// own output and the command that fixes it. Not modal: the window stays
// usable while a job waits in the queue.
//
// Also here: the box for one ssh prompt (a password, a one-time code, a host
// key question), and the cluster's file browser that the Open dataset
// dialog and the file parameters use.

#include "imgui/dialogs/dialogs.hpp"

#include <algorithm>
#include <cfloat>
#include <atomic>
#include <chrono>
#include <ctime>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>

#include "core/remote_source.hpp"
#include "imgui/cluster_link.hpp"
#include "imgui/dialogs/export_dialog_support.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui {

    namespace {

        using namespace dialog_support;
        using theme::px;

        constexpr ImU32 kGreen = theme::rgb(0x2e, 0x7d, 0x32);
        constexpr ImU32 kAmber = theme::rgb(0xb2, 0x6a, 0x00);

        void mark(cluster::StepStatus s) {
            const float d = px(10);
            const ImVec2 at = ImGui::GetCursorScreenPos();
            const float lineH = ImGui::GetTextLineHeight();
            const ImVec2 c(at.x + d * 0.5f, at.y + lineH * 0.5f + px(1));
            ImDrawList* dl = ImGui::GetWindowDrawList();
            switch (s) {
                case cluster::StepStatus::Pending: dl->AddCircle(c, d * 0.4f, theme::kNeutral400, 0, px(1.5f)); break;
                case cluster::StepStatus::Running: {
                    // a turning arc: something is happening
                    const float a0 = static_cast<float>(ImGui::GetTime() * 4.0);
                    dl->PathArcTo(c, d * 0.4f, a0, a0 + 4.2f, 16);
                    dl->PathStroke(theme::kAccent, 0, px(2));
                    break;
                }
                case cluster::StepStatus::Done: dl->AddCircleFilled(c, d * 0.45f, kGreen); break;
                case cluster::StepStatus::Warning: dl->AddCircleFilled(c, d * 0.45f, kAmber); break;
                case cluster::StepStatus::Failed: dl->AddCircleFilled(c, d * 0.45f, theme::kAccent); break;
            }
            ImGui::Dummy(ImVec2(d, lineH));
        }

        // A read-only, monospace, selectable block with a Copy button.
        void outputBlock(const char* id, const std::string& text, float heightPx) {
            std::string copy = text;
            widgets::FieldOpts fo;
            fo.readOnly = true;
            fo.monospace = true;
            widgets::inputTextMultiline(id, &copy, heightPx, fo);
        }

        // --- the connect dialog ---------------------------------------------------------

        class ClusterDialog final : public Dialog {
        public:
            explicit ClusterDialog(App& app) : profile_(app.cluster().storedProfile()) {
                gpus_ = profile_.gpus;
                cpus_ = profile_.cpus;
                port_ = profile_.port;
            }

            std::string title() const override { return "Connect to cluster"; }
            ImVec2 size() const override { return ImVec2(620, 0); }
            bool modal() const override { return false; }

            void draw(App& app) override {
                ClusterLink& link = app.cluster();
                const cluster::Status st = link.status();
                const bool busy = st.state == cluster::State::Connecting;
                const bool connected = st.state == cluster::State::Connected;
                const bool editable = !busy && !connected;
                if (busy) app.requestRedraw(2);   // the running mark turns, the waits count up
                // the body scrolls on a small screen; the actions stay in sight
                ImGui::SetNextWindowSizeConstraints(ImVec2(0.0f, 0.0f),
                                                    ImVec2(FLT_MAX, std::max(px(200), ImGui::GetMainViewport()->WorkSize.y * 0.8f - px(150))));
                ImGui::BeginChild("##clusterBody", ImVec2(0.0f, 0.0f), ImGuiChildFlags_AutoResizeY);
                if (editable) {
                    {
                        const Spacing spacing(8, 10);
                        drawProfile(editable);
                    }
                    widgets::vspace(4);
                    note("Your password and one-time codes are asked for in a separate box when the cluster asks, handed to ssh and "
                         "never stored. A wrong one costs one attempt; Connect again to retry.");
                } else {
                    // the profile in one line while it is in use
                    const cluster::Profile p = link.session().profile();
                    std::string line = p.host + " \xC2\xB7 " + p.checkout;
                    for (const std::string& part : {p.partition, p.account, p.qos, p.time})
                        if (!part.empty()) line += " \xC2\xB7 " + part;
                    line += " \xC2\xB7 " + std::to_string(p.gpus) + (p.gpus == 1 ? " GPU" : " GPUs") + " \xC2\xB7 " + std::to_string(p.cpus) +
                            " CPUs \xC2\xB7 " + p.mem;
                    widgets::textWrapped(line, 12, theme::kNeutral700);
                }
                widgets::rule(theme::kRule);
                drawSteps(st);
                drawOutcome(app, st);
                ImGui::EndChild();
                widgets::vspace(6);
                drawActions(app, link, st, busy, connected);
            }

        private:
            void drawProfile(bool editable) {
                widgets::FieldOpts fo;
                fo.enabled = editable;
                {
                    const float w = columnWidth(3, 10);
                    fo.width = design(w * 2 + px(10));
                    {
                        const Field f("SSH host");
                        fo.hint = "fiona, or user@login.cluster.org";
                        widgets::inputText("##host", &profile_.host, fo);
                        widgets::tooltip("A host of your ~/.ssh/config works, with its user, ProxyJump and the rest.");
                    }
                    ImGui::SameLine(0.0f, px(10));
                    fo.width = design(w);
                    fo.hint.clear();
                    const Field f("Worker port");
                    spinInt("##port", &port_, 1024, 65535, 1, fo);
                }
                {
                    const float w = columnWidth(2, 10);
                    fo.width = design(w);
                    {
                        const Field f("SIRIUS checkout on the cluster");
                        widgets::inputText("##checkout", &profile_.checkout, fo);
                        widgets::tooltip("Needs app/python (the worker) and app/python/slurm (the job script), the same version as "
                                         "this application.");
                    }
                    ImGui::SameLine(0.0f, px(10));
                    const Field f("Python environment (venv)");
                    fo.hint = "none";
                    widgets::inputText("##venv", &profile_.venv, fo);
                    widgets::tooltip("Activated for the worker; it needs numpy (and torch for models, tifffile to open cluster "
                                     "datasets). Empty: the python of the job's modules.");
                    fo.hint.clear();
                }
                {
                    const float w = columnWidth(3, 10);
                    fo.width = design(w);
                    {
                        const Field f("Partition");
                        widgets::inputText("##partition", &profile_.partition, fo);
                    }
                    ImGui::SameLine(0.0f, px(10));
                    {
                        const Field f("Account");
                        widgets::inputText("##account", &profile_.account, fo);
                    }
                    ImGui::SameLine(0.0f, px(10));
                    const Field f("QoS");
                    widgets::inputText("##qos", &profile_.qos, fo);
                }
                {
                    const float w = columnWidth(4, 10);
                    fo.width = design(w);
                    {
                        const Field f("Time limit");
                        fo.hint = "01:00:00";
                        widgets::inputText("##time", &profile_.time, fo);
                        fo.hint.clear();
                    }
                    ImGui::SameLine(0.0f, px(10));
                    {
                        const Field f("GPUs");
                        spinInt("##gpus", &gpus_, 0, 16, 1, fo);
                    }
                    ImGui::SameLine(0.0f, px(10));
                    {
                        const Field f("CPUs");
                        spinInt("##cpus", &cpus_, 1, 256, 1, fo);
                    }
                    ImGui::SameLine(0.0f, px(10));
                    const Field f("Memory");
                    widgets::inputText("##mem", &profile_.mem, fo);
                }
            }

            void drawSteps(const cluster::Status& st) {
                for (int i = 0; i < cluster::kStepCount; ++i) {
                    const cluster::StepState& s = st.steps[static_cast<std::size_t>(i)];
                    ImGui::PushID(i);
                    mark(s.status);
                    ImGui::SameLine(0.0f, px(8));
                    ImGui::BeginGroup();
                    const ImU32 ink = s.status == cluster::StepStatus::Pending ? theme::kNeutral500 : theme::kText;
                    widgets::text(cluster::stepTitle(static_cast<cluster::Step>(i)), 12, ink, theme::Weight::SemiBold);
                    if (!s.detail.empty())
                        widgets::textWrapped(s.detail, 11, s.status == cluster::StepStatus::Failed ? theme::kAccentText : theme::kNeutral600);
                    ImGui::EndGroup();
                    ImGui::PopID();
                }
            }

            void drawOutcome(App& app, const cluster::Status& st) {
                if (st.state == cluster::State::Connected) {
                    widgets::rule(theme::kRule);
                    widgets::text("Connected to the worker on " + st.node + " (job " + st.jobId + ")", 12, kGreen, theme::Weight::SemiBold);
                    const WorkerCapabilities& c = st.caps;
                    widgets::textWrapped("sirius_worker " + c.version + " \xC2\xB7 Python " + c.python + " \xC2\xB7 " + c.device + " on " +
                                             c.hostname,
                                         11, theme::kNeutral700);
                    std::string kinds;
                    for (const std::string& m : c.methods)
                        if (m.rfind("run:", 0) == 0) kinds += (kinds.empty() ? "" : ", ") + m.substr(4);
                    widgets::textWrapped("Steps it runs: " + kinds, 11, theme::kNeutral600);
                    if (c.tifffile.empty())
                        widgets::textWrapped("tifffile is not installed in the worker's Python: TIFF datasets on the cluster cannot be "
                                             "opened (" +
                                                 st.fix + ").",
                                             11, kAmber);
                    widgets::textWrapped("The HPC backend runs there now; File \xE2\x96\xB8 Open dataset\xE2\x80\xA6 \xE2\x96\xB8 Cluster opens "
                                         "datasets that stay on the cluster.",
                                         11, theme::kNeutral600);
                    return;
                }
                if (st.state != cluster::State::Disconnected || st.reason.empty()) return;
                widgets::rule(theme::kRule);
                // the failed step says why already; a disconnect after the connect has no step to say it
                bool told = false;
                for (const cluster::StepState& s : st.steps) told = told || (s.status == cluster::StepStatus::Failed && s.detail == st.reason);
                if (!told) widgets::textWrapped(st.reason, 12, theme::kAccentText);
                if (!st.remoteOutput.empty()) {
                    widgets::caption("What the cluster said");
                    outputBlock("##remote", st.remoteOutput, 90);
                }
                if (!st.fix.empty()) {
                    widgets::caption("To fix it, on the cluster");
                    outputBlock("##fix", st.fix, 52);
                    if (widgets::chipButton("Copy command")) ImGui::SetClipboardText(st.fix.c_str());
                }
                (void)app;
            }

            void drawActions(App& app, ClusterLink& link, const cluster::Status& st, bool busy, bool connected) {
                widgets::ButtonOpts b;
                b.small = true;
                b.enabled = st.sshUp;
                b.tooltip = st.sshUp ? std::string("The cluster's files, through this SSH session") : std::string("Log in first");
                if (widgets::button("Browse cluster files\xE2\x80\xA6", b)) {
                    app.defer([&app] {
                        app.showDialog(makeClusterBrowser(app, {}, false, [&app](const std::string& path) { app.openDatasetPath(path); }));
                    });
                }
                std::string primary = "Connect";
                if (busy) primary = "Stop";
                else if (connected) primary = "Disconnect\xE2\x80\xA6";
                else if (st.state == cluster::State::Disconnected) primary = "Connect again";
                const float gap = px(8);
                const float total = buttonWidth("Close", widgets::ButtonKind::Ghost) + gap + buttonWidth(primary, widgets::ButtonKind::Primary);
                ImGui::SameLine();
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - total));
                widgets::ButtonOpts ghost;
                ghost.kind = widgets::ButtonKind::Ghost;
                if (widgets::button("Close##clusterClose", ghost)) close();
                ImGui::SameLine(0.0f, gap);
                const bool valid = !trimmed(profile_.host).empty() && !trimmed(profile_.checkout).empty();
                if (widgets::primaryButton((primary + "##clusterPrimary").c_str(), 0.0f, busy || connected || valid)) {
                    if (busy) {
                        link.session().cancelConnect();
                    } else if (connected) {
                        link.disconnectAsking();
                    } else {
                        profile_.host = trimmed(profile_.host);
                        profile_.checkout = trimmed(profile_.checkout);
                        profile_.venv = trimmed(profile_.venv);
                        profile_.gpus = static_cast<int>(gpus_);
                        profile_.cpus = static_cast<int>(cpus_);
                        profile_.port = static_cast<int>(port_);
                        link.connect(profile_);
                    }
                }
            }

            cluster::Profile profile_;
            std::int64_t gpus_ = 1, cpus_ = 8, port_ = 7645;
        };

        // --- one ssh prompt ------------------------------------------------------------

        class PromptDialog final : public Dialog {
        public:
            explicit PromptDialog(std::shared_ptr<ClusterPrompt> p) : p_(std::move(p)) {}
            ~PromptDialog() override { finish(true); }

            std::string title() const override { return "Log in to " + (p_->host.empty() ? std::string("the cluster") : p_->host); }
            ImVec2 size() const override { return ImVec2(440, 0); }

            void draw(App&) override {
                if (p_->abandoned.load()) {
                    finish(true);
                    close();
                    return;
                }
                widgets::textWrapped(p_->prompt.text, 13, theme::kText);
                widgets::vspace(4);
                widgets::FieldOpts fo;
                fo.password = !p_->prompt.echo;
                fo.enterReturnsTrue = true;
                if (!focused_) {
                    ImGui::SetKeyboardFocusHere();
                    focused_ = true;
                }
                const bool enter = widgets::inputText("##answer", &answer_, fo);
                note("Handed to ssh for this login only: not stored, not logged. Cancel stops the login without sending anything.");
                widgets::vspace(6);
                Action a = actionRow("Continue", true);
                if (enter) a = Action::Accept;
                if (a == Action::Cancel) {
                    finish(true);
                    close();
                } else if (a == Action::Accept) {
                    finish(false);
                    close();
                }
            }

            void closed(App&) override { finish(true); }

        private:
            void finish(bool cancelled) {
                if (done_) return;
                done_ = true;
                if (!cancelled) p_->answer = answer_;
                p_->cancelled = cancelled;
                std::fill(answer_.begin(), answer_.end(), '\0');
                answer_.clear();
                p_->answered.store(true);
            }

            std::shared_ptr<ClusterPrompt> p_;
            std::string answer_;
            bool focused_ = false;
            bool done_ = false;
        };

        // --- the cluster's files ------------------------------------------------------------

        struct BrowserShared {
            std::mutex m;
            bool loading = false;
            std::optional<cluster::Listing> listing;
            std::string error;
            bool gone = false;
        };

        std::string whenText(double mtime) {
            if (mtime <= 0) return {};
            const std::time_t t = static_cast<std::time_t>(mtime);
            std::tm tm{};
#ifdef _WIN32
            localtime_s(&tm, &t);
#else
            localtime_r(&t, &tm);
#endif
            char buf[32];
            std::strftime(buf, sizeof buf, "%Y-%m-%d %H:%M", &tm);
            return buf;
        }

        std::string parentOf(const std::string& p) {
            if (p.size() <= 1) return p;
            std::string s = p;
            while (s.size() > 1 && (s.back() == '/' || s.back() == '\\')) s.pop_back();
            const std::size_t slash = s.find_last_of("/\\");
            if (slash == std::string::npos) return s;
            if (slash == 0) return "/";
            if (slash == 2 && s[1] == ':') return s.substr(0, 3);
            return s.substr(0, slash);
        }

        std::string joinPath(const std::string& dir, const std::string& name) {
            if (dir.empty()) return name;
            return dir.back() == '/' || dir.back() == '\\' ? dir + name : dir + "/" + name;
        }

        class ClusterBrowser final : public Dialog {
        public:
            ClusterBrowser(App& app, std::string start, bool folders, std::function<void(const std::string&)> chosen)
                : folders_(folders), chosen_(std::move(chosen)) {
                const std::vector<std::string> recent = app.cluster().recentFolders();
                go(app, !start.empty() ? start : (!recent.empty() ? recent.front() : std::string("~")));
            }
            ~ClusterBrowser() override {
                const std::lock_guard<std::mutex> g(shared_->m);
                shared_->gone = true;
            }

            std::string title() const override { return "Cluster files" + (host_.empty() ? std::string() : " \xC2\xB7 " + host_); }
            ImVec2 size() const override { return ImVec2(640, 520); }
            bool resizable() const override { return true; }

            void draw(App& app) override {
                ClusterLink& link = app.cluster();
                host_ = link.status().host;
                if (!link.sshUp()) {
                    widgets::textWrapped("Not logged in to a cluster. Connect first (Process \xE2\x96\xB8 Connect to cluster\xE2\x80\xA6): browsing works "
                                         "as soon as the SSH login is done, before the worker job runs.",
                                         12, theme::kNeutral700);
                    if (widgets::linkButton("Connect to cluster\xE2\x80\xA6")) app.defer([&app] { app.clusterDialog(); });
                    widgets::vspace(6);
                    if (actionRow("Open", false) == Action::Cancel) close();
                    return;
                }
                std::optional<cluster::Listing> listing;
                bool loading = false;
                std::string error;
                {
                    const std::lock_guard<std::mutex> g(shared_->m);
                    listing = shared_->listing;
                    loading = shared_->loading;
                    error = shared_->error;
                }
                if (listing && listing->path != shown_) {
                    shown_ = listing->path;
                    pathField_ = shown_;
                    home_ = listing->home;
                    selected_.clear();
                }
                if (loading) app.requestRedraw(2);
                drawBar(app);
                widgets::vspace(4);
                const float footer = ImGui::GetFrameHeightWithSpacing() + px(44);
                ImGui::BeginChild("##entries", ImVec2(0, std::max(px(120), ImGui::GetContentRegionAvail().y - footer)), ImGuiChildFlags_Borders);
                if (!error.empty()) widgets::textWrapped(error, 12, theme::kAccentText);
                else if (loading && !listing) widgets::text("Listing\xE2\x80\xA6", 12, theme::kNeutral600);
                if (listing) drawEntries(app, *listing);
                ImGui::EndChild();
                if (listing && listing->truncated)
                    note("Only the first " + std::to_string(listing->entries.size()) + " entries are listed: type a path to go deeper.");
                widgets::checkbox("Show hidden files", &hidden_);
                ImGui::SameLine();
                const std::string target = folders_ ? shown_ : (selected_.empty() ? std::string() : joinPath(shown_, selected_));
                widgets::text(target.empty() ? std::string("Choose a file") : target, 11, theme::kNeutral600);
                const Action a = actionRow("Open", !target.empty());
                if (a == Action::Cancel) close();
                else if (a == Action::Accept) accept(app, target);
            }

        private:
            void accept(App& app, const std::string& remotePath) {
                app.cluster().addRecentFolder(folders_ ? remotePath : shown_);
                const std::string name = makeClusterPath(host_, remotePath);
                auto chosen = chosen_;
                close();
                if (chosen) app.defer([chosen, name] { chosen(name); });
            }

            void go(App& app, const std::string& path) {
                {
                    const std::lock_guard<std::mutex> g(shared_->m);
                    if (shared_->loading) return;
                    shared_->loading = true;
                    shared_->error.clear();
                }
                auto shared = shared_;
                cluster::Session* session = &app.cluster().session();
                Bridge* bridge = &app.bridge();
                worker_.start([shared, session, bridge, path] {
                    std::optional<cluster::Listing> l;
                    std::string error;
                    try {
                        l = session->list(path);
                    } catch (const ssh::SshError& e) {
                        error = std::string(e.what()) + (e.detail.empty() ? std::string() : ": " + e.detail);
                    } catch (const std::exception& e) {
                        error = e.what();
                    }
                    const std::lock_guard<std::mutex> g(shared->m);
                    shared->loading = false;
                    if (shared->gone) return;
                    if (l) shared->listing = std::move(l);
                    shared->error = error;
                    bridge->post([] {});
                });
            }

            void drawBar(App& app) {
                widgets::ButtonOpts b;
                b.small = true;
                if (widgets::button("Up", b)) go(app, parentOf(shown_));
                ImGui::SameLine(0.0f, px(6));
                if (widgets::button("Home", b)) go(app, home_.empty() ? std::string("~") : home_);
                ImGui::SameLine(0.0f, px(6));
                if (widgets::button("Refresh", b)) go(app, shown_.empty() ? std::string("~") : shown_);
                ImGui::SameLine(0.0f, px(6));
                const std::vector<std::string> recent = app.cluster().recentFolders();
                int pick = -1;
                widgets::FieldOpts rf;
                rf.width = 140;
                std::vector<std::string> items = recent;
                if (items.empty()) items.emplace_back("(no recent folders)");
                if (widgets::combo("##recent", &pick, items, rf) && pick >= 0 && !recent.empty())
                    go(app, recent[static_cast<std::size_t>(pick)]);
                widgets::tooltip("Cluster folders opened recently");
                ImGui::SameLine(0.0f, px(6));
                widgets::FieldOpts fo;
                fo.enterReturnsTrue = true;
                fo.monospace = true;
                if (widgets::inputText("##path", &pathField_, fo)) go(app, trimmed(pathField_));
            }

            void drawEntries(App& app, const cluster::Listing& l) {
                const float sizeX = ImGui::GetContentRegionAvail().x - px(210);
                for (const cluster::Entry& e : l.entries) {
                    if (!hidden_ && !e.name.empty() && e.name.front() == '.') continue;
                    if (folders_ && !e.dir) continue;
                    ImGui::PushID(e.name.c_str());
                    const std::string label = (e.dir ? "\xE2\x96\xB8 " : "   ") + e.name + (e.link ? " \xE2\x86\x92" : "");
                    const bool sel = !e.dir && selected_ == e.name;
                    if (ImGui::Selectable(label.c_str(), sel, ImGuiSelectableFlags_AllowDoubleClick, ImVec2(sizeX, 0))) {
                        if (e.dir) {
                            ImGui::PopID();
                            go(app, joinPath(l.path, e.name));
                            return;
                        } else {
                            selected_ = e.name;
                            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                                ImGui::PopID();
                                accept(app, joinPath(l.path, e.name));
                                return;
                            }
                        }
                    }
                    ImGui::SameLine(sizeX + px(8));
                    widgets::text(e.dir ? std::string() : bytesText(e.size), 11, theme::kNeutral600);
                    ImGui::SameLine(sizeX + px(80));
                    widgets::text(whenText(e.mtime), 11, theme::kNeutral600);
                    ImGui::PopID();
                }
            }

            bool folders_;
            std::function<void(const std::string&)> chosen_;
            std::shared_ptr<BrowserShared> shared_ = std::make_shared<BrowserShared>();
            DialogThread worker_;
            std::string shown_, pathField_, home_, selected_, host_;
            bool hidden_ = false;
        };

    } // namespace

    std::shared_ptr<Dialog> makeClusterDialog(App& app) { return std::make_shared<ClusterDialog>(app); }

    std::shared_ptr<Dialog> makeClusterPromptDialog(App&, std::shared_ptr<ClusterPrompt> prompt) {
        return std::make_shared<PromptDialog>(std::move(prompt));
    }

    std::shared_ptr<Dialog> makeClusterBrowser(App& app, const std::string& start, bool folders,
                                               std::function<void(const std::string& clusterPath)> chosen) {
        return std::make_shared<ClusterBrowser>(app, start, folders, std::move(chosen));
    }

} // namespace sirius::app::gui
