// Process ▸ Connect to cluster…, in two steps under the profile:
//
//   the profile   one per cluster (a dropdown, New cluster…, Rename, Delete,
//                 Import / Export as a small .toml), kept in the settings
//                 file as [cluster.<name>]; Basic: the SSH host, the worker
//                 image, the data folders, each with a line of what it is;
//                 Advanced (folded): the job (dropdowns of the profile's
//                 partitions, accounts, QoS and times, and what the cluster
//                 reports, marked so), the software, the node cache folder
//   1 The job     Connect: the login, a job that holds a node, its queue
//   2 The worker  Start worker: the image checked, the worker started in
//                 the job, its hello; or Build an image in the job
//
// Each step's checklist says what it found or, when one fails, why in plain
// words, the cluster's own output and what to do. Everything stays editable
// while connected: what changed is named with what it takes -- a new job,
// or the worker restarted in the same job. Not modal: the window stays
// usable while a job waits in the queue.
//
// Also here: the box for one ssh prompt (a password, a one-time code, a host
// key question), and the cluster's file browser that the Open dataset
// dialog and the file parameters use.

#include "imgui/dialogs/dialogs.hpp"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <ctime>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>
#include <nlohmann/json.hpp>

#include "core/cluster_profiles.hpp"
#include "core/host.hpp"
#include "core/remote_source.hpp"
#include "core/secure_wipe.hpp"
#include "imgui/cluster_link.hpp"
#include "imgui/dialogs/export_dialog_support.hpp"
#include "imgui/platform.hpp"
#include "imgui/settings.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/widgets/controls.hpp"
#include "imgui/widgets/icons.hpp"

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

        // A Refresh of the partitions on the dialog's thread.
        struct RefreshShared {
            std::mutex m;
            bool running = false;
            std::string error;
            bool gone = false;
        };

        // One line of plain words under a field: what it is for.
        void describe(const std::string& text) { widgets::textWrapped(text, 11, theme::kNeutral600, theme::Weight::Regular, ImGui::GetContentRegionAvail().x); }

        // A group's heading inside Advanced, with a line of what it is for.
        void group(const char* title, const std::string& what) {
            widgets::vspace(6);
            widgets::text(title, 12, theme::kText, theme::Weight::SemiBold);
            describe(what);
            widgets::vspace(2);
        }

        // A heading that folds what is under it; true while open.
        bool disclosure(const char* id, const std::string& label, bool* open) {
            ImGui::PushID(id);
            const ImVec2 at = ImGui::GetCursorScreenPos();
            const float h = ImGui::GetTextLineHeight() + px(6);
            const float w = theme::textSize(label, 13, theme::Weight::SemiBold).x + px(26);
            if (ImGui::InvisibleButton("##fold", ImVec2(w, h))) *open = !*open;
            const bool hov = ImGui::IsItemHovered();
            if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            ImDrawList* dl = ImGui::GetWindowDrawList();
            drawIcon(dl, ImVec2(at.x, at.y + (h - px(14)) * 0.5f), ImVec2(at.x + px(14), at.y + (h + px(14)) * 0.5f),
                     *open ? Icon::ChevronDown : Icon::ChevronRight, hov ? theme::kAccent : theme::kText);
            widgets::drawTextIn(dl, ImVec2(at.x + px(20), at.y), ImVec2(at.x + w, at.y + h), label, 13, hov ? theme::kAccent : theme::kText,
                                theme::Weight::SemiBold, 0.0f, 0.5f);
            ImGui::PopID();
            return *open;
        }

        class ClusterDialog final : public Dialog {
        public:
            explicit ClusterDialog(App& app) {
                ClusterLink& link = app.cluster();
                book_ = link.profiles();
                // the profile a session runs, when there is one: what the dialog edits
                const cluster::Status st = link.status();
                const bool inUse = st.state == cluster::State::JobReady || st.state == cluster::State::Connected ||
                                   st.state == cluster::State::Connecting || st.state == cluster::State::Starting;
                if (inUse && book_.find(link.session().profile().name)) book_.current = link.session().profile().name;
                if (book_.profiles.empty()) {
                    cluster::Profile fresh;
                    fresh.name = book_.uniqueName("New cluster");
                    book_.profiles.push_back(fresh);
                    book_.current = fresh.name;
                    autoName_ = true;
                }
                select(book_.current);
                savedBook_ = bookJson(book_);
                // Screenshots only: the partition list open once it is in, Advanced open.
                openList_ = app.unattended() && !host::environment("SIRIUS_TEST_OPEN_PARTITIONS").empty();
                advanced_ = app.unattended() && !host::environment("SIRIUS_TEST_CLUSTER_ADVANCED").empty();
                buildOpen_ = app.unattended() && !host::environment("SIRIUS_TEST_CLUSTER_BUILD").empty();
                scrollTo_ = app.unattended() ? host::environment("SIRIUS_TEST_CLUSTER_SCROLL") : std::string();   // "advanced", or the end
                scrollDown_ = scrollTo_.empty() ? 0 : 60;
            }
            ~ClusterDialog() override {
                *alive_ = false;
                const std::lock_guard<std::mutex> g(refresh_->m);
                refresh_->gone = true;
            }

            std::string title() const override { return "Connect to cluster"; }
            ImVec2 size() const override { return ImVec2(680, 0); }
            bool modal() const override { return false; }

            bool canClose(App& app) override {
                if (!dirty()) return true;
                askToSave(app, [this] { close(); });
                return false;
            }

            void draw(App& app) override {
                ClusterLink& link = app.cluster();
                const cluster::Status st = link.status();
                const bool busy = st.state == cluster::State::Connecting || st.state == cluster::State::Starting;
                if (busy || st.build.phase == cluster::BuildStatus::Phase::Probing || st.build.phase == cluster::BuildStatus::Phase::Building)
                    app.requestRedraw(2);   // the running marks turn, the waits count up
                // the partitions as last listed, for this profile's host
                info_.reset();
                if (std::optional<cluster::ClusterInfo> ci = link.session().clusterInfo(); ci && ci->host == trimmed(profile_.host)) info_ = std::move(ci);
                fillFromCluster(link);
                // the body scrolls on a small screen; the actions stay in sight
                ImGui::SetNextWindowSizeConstraints(ImVec2(0.0f, 0.0f),
                                                    ImVec2(FLT_MAX, std::max(px(240), ImGui::GetMainViewport()->WorkSize.y * 0.82f - px(140))));
                ImGui::BeginChild("##clusterBody", ImVec2(0.0f, 0.0f), ImGuiChildFlags_AutoResizeY);
                {
                    const Spacing spacing(8, 8);
                    drawProfiles(app, link, st, busy);
                    widgets::rule(theme::kRule);
                    drawBasic(app, st, !busy);
                    widgets::vspace(4);
                    if (scrollDown_ > 0 && scrollTo_ == "advanced") {
                        ImGui::SetScrollHereY(0.0f);   // screenshots only: Advanced at the top
                        --scrollDown_;
                    }
                    if (disclosure("##advanced", "Advanced settings", &advanced_)) drawAdvanced(app, link, st, !busy);
                    widgets::rule(theme::kRule);
                    drawApply(app, link, st);
                    drawJob(app, link, st, busy);
                    drawWorker(app, link, st, busy);
                    drawOutcome(st);
                }
                if (scrollDown_ > 0 && scrollTo_ != "advanced") {
                    // screenshots only: the steps in sight, under the profile
                    ImGui::SetScrollHereY(1.0f);
                    --scrollDown_;
                }
                ImGui::EndChild();
                widgets::vspace(6);
                drawActions(app, link, st);
            }

        private:
            // --- the profiles ---------------------------------------------------------------

            static nlohmann::json bookJson(const cluster::ProfileBook& b) {
                nlohmann::json j = nlohmann::json::object();
                for (const auto& [k, v] : b.toSettings()) j[k] = v;
                return j;
            }

            // The book with the fields as they are now.
            cluster::ProfileBook edited() const {
                cluster::ProfileBook b = book_;
                cluster::Profile p = profile_;
                p.gpus = static_cast<int>(gpus_);
                p.cpus = static_cast<int>(cpus_);
                if (cluster::Profile* there = b.find(name_)) *there = p;
                return b;
            }

            bool dirty() const { return bookJson(edited()) != savedBook_; }

            void select(const std::string& name) {
                const cluster::Profile* p = book_.find(name);
                if (!p) return;
                name_ = name;
                profile_ = *p;
                book_.current = name;
                gpus_ = profile_.gpus;
                cpus_ = profile_.cpus;
                lastHost_ = trimmed(profile_.host);
                filledFor_.clear();
                filled_.clear();
                custom_.clear();
                renaming_ = false;
            }

            // The fields into the book (in memory).
            void keep() { book_ = edited(); }

            // The book into the settings file.
            void save(ClusterLink& link) {
                keep();
                // a profile made here and still called "New cluster": named after its host
                if (autoName_ && !trimmed(profile_.host).empty() && name_.rfind("New cluster", 0) == 0) {
                    const std::string to = book_.uniqueName(trimmed(profile_.host));
                    if (book_.rename(name_, to)) {
                        name_ = to;
                        profile_.name = to;
                    }
                    autoName_ = false;
                }
                link.saveProfiles(book_);
                savedBook_ = bookJson(book_);
            }

            void askToSave(App& app, std::function<void()> then) {
                auto self = alive_;
                app.ask("Unsaved cluster settings", "Save the changes to \"" + name_ + "\" in the settings file?", {"Discard", "Save"}, [this, self, &app, then](int answer) {
                            if (!*self || answer < 0) return;
                            if (answer == 1) save(app.cluster());
                            else savedBook_ = bookJson(edited());   // let go
                            if (then) then(); }, 1);
            }

            void drawProfiles(App& app, ClusterLink& link, const cluster::Status& st, bool busy) {
                widgets::FieldOpts fo;
                fo.enabled = !busy;
                const float gap = px(8);
                if (renaming_) {
                    const Field f("Rename the profile");
                    fo.width = design(columnWidth(2, 8));
                    fo.enterReturnsTrue = true;
                    const bool enter = widgets::inputText("##rename", &renameText_, fo);
                    ImGui::SameLine(0.0f, gap);
                    const std::string problem = book_.nameProblem(renameText_, name_);
                    if ((widgets::linkButton("Rename##do", problem.empty()) || (enter && problem.empty()))) {
                        keep();
                        const std::string to = trimmed(renameText_);
                        if (book_.rename(name_, to)) {
                            name_ = to;
                            profile_.name = to;
                            autoName_ = false;
                        }
                        renaming_ = false;
                    }
                    ImGui::SameLine(0.0f, gap);
                    if (widgets::linkButton("Cancel##rename")) renaming_ = false;
                    if (!problem.empty()) describe(problem);
                    return;
                }
                {
                    const Field f("Cluster profile");
                    std::vector<std::string> names;
                    int current = 0;
                    for (std::size_t i = 0; i < book_.profiles.size(); ++i) {
                        names.push_back(book_.profiles[i].name);
                        if (book_.profiles[i].name == name_) current = static_cast<int>(i);
                    }
                    names.emplace_back("New cluster\xE2\x80\xA6");
                    fo.width = 230;
                    // the profile in use stays while a session runs on it
                    widgets::FieldOpts po = fo;
                    po.enabled = fo.enabled && !(st.state == cluster::State::JobReady || st.state == cluster::State::Connected);
                    int pick = current;
                    if (widgets::combo("##profile", &pick, names, po) && pick != current) {
                        keep();
                        if (pick == static_cast<int>(names.size()) - 1) {
                            cluster::Profile fresh;
                            fresh.name = book_.uniqueName("New cluster");
                            book_.profiles.push_back(fresh);
                            select(fresh.name);
                            autoName_ = true;
                        } else {
                            select(names[static_cast<std::size_t>(pick)]);
                        }
                    }
                    widgets::tooltip("One profile per cluster: its host, image, data folders and the job it asks for. They are kept in the "
                                     "settings file as [cluster.<name>] tables.");
                }
                ImGui::SameLine(0.0f, gap);
                ImGui::BeginGroup();
                widgets::fieldLabel(" ");
                widgets::ButtonOpts b;
                b.small = true;
                b.enabled = !busy;
                b.tooltip = "Give this profile another name";
                if (widgets::button("Rename\xE2\x80\xA6", b)) {
                    renaming_ = true;
                    renameText_ = name_;
                }
                ImGui::SameLine(0.0f, px(6));
                const bool inUse = st.state == cluster::State::JobReady || st.state == cluster::State::Connected;
                b.enabled = !busy && !inUse;
                b.tooltip = inUse ? std::string("Disconnect first: a job runs with this profile") : std::string("Remove this profile from the settings file");
                if (widgets::button("Delete\xE2\x80\xA6", b)) {
                    auto self = alive_;
                    const std::string doomed = name_;
                    app.ask("Delete a cluster profile", "Remove \"" + doomed + "\" from the settings file? Nothing on the cluster is touched.", {"Keep it", "Delete"}, [this, self, &link, doomed](int answer) {
                                if (!*self || answer != 1) return;
                                keep();
                                book_.remove(doomed);
                                if (book_.profiles.empty()) {
                                    cluster::Profile fresh;
                                    fresh.name = book_.uniqueName("New cluster");
                                    book_.profiles.push_back(fresh);
                                    autoName_ = true;
                                }
                                select(book_.current.empty() ? book_.profiles.front().name : book_.current);
                                link.saveProfiles(book_);
                                savedBook_ = bookJson(book_); }, 0);
                }
                ImGui::SameLine(0.0f, px(6));
                b.enabled = !busy;
                b.tooltip = "Add the profiles of a file a colleague exported (.toml)";
                if (widgets::button("Import\xE2\x80\xA6", b)) {
                    auto self = alive_;
                    app.defer([this, self, &app] {
                        const std::string path = platform::openFileDialog("Import a cluster profile", {}, {{"Cluster profile", "toml"}, {"All files", "*"}});
                        if (!*self || path.empty()) return;
                        std::string text;
                        std::vector<cluster::Profile> found;
                        try {
                            if (!platform::readFile(path, text)) throw std::runtime_error("it could not be read");
                            found = cluster::importProfiles(text);
                        } catch (const std::exception& e) {
                            app.message("Import a cluster profile", path + " was not imported: " + e.what() + ".", MessageIcon::Warning);
                            return;
                        }
                        keep();
                        std::string last;
                        for (cluster::Profile p : found) {
                            p.name = book_.uniqueName(p.name);
                            last = p.name;
                            book_.profiles.push_back(p);
                        }
                        select(last);
                        app.cluster().saveProfiles(book_);
                        savedBook_ = bookJson(book_);
                        app.wb().logLine("Cluster: imported " + std::to_string(found.size()) + " profile(s) from " + path);
                    });
                }
                ImGui::SameLine(0.0f, px(6));
                b.tooltip = "Save this profile as a small .toml file to share (no password, no token is in it)";
                if (widgets::button("Export\xE2\x80\xA6", b)) {
                    const cluster::Profile p = edited().currentProfile();
                    app.defer([p, &app] {
                        const std::string path = platform::saveFileDialog("Export the cluster profile", {}, p.name + ".toml", {{"Cluster profile", "toml"}});
                        if (path.empty()) return;
                        if (platform::writeFileAtomic(path, cluster::exportProfile(p))) app.wb().logLine("Cluster: exported " + p.name + " to " + path);
                        else app.message("Export the cluster profile", "Could not write " + path + ".", MessageIcon::Warning);
                    });
                }
                ImGui::EndGroup();
                if (!trimmed(profile_.host).empty() && trimmed(profile_.host) != lastHost_ && !hostEditing_) {
                    // another host: the partition, account, QoS, time and binds last used there
                    lastHost_ = trimmed(profile_.host);
                    if (profile_.recall(lastHost_)) filled_ = {"the partition, account, QoS, time and data folders last used on " + lastHost_};
                }
            }

            // --- basic ----------------------------------------------------------------------

            void drawBasic(App& app, const cluster::Status& st, bool editable) {
                widgets::FieldOpts fo;
                fo.enabled = editable;
                const float gap = px(10);
                const bool here = st.sshUp && st.host == trimmed(profile_.host);
                {
                    const Field f("Cluster (SSH host)");
                    fo.width = design(ImGui::GetContentRegionAvail().x);
                    fo.hint = "an alias from your ~/.ssh/config, or user@login.example.org";
                    widgets::inputText("##host", &profile_.host, fo);
                    hostEditing_ = ImGui::IsItemActive();
                    fo.hint.clear();
                    widgets::tooltip("The login node you submit jobs from. A Host of your ~/.ssh/config works with its user, ProxyJump and keys. "
                                     "Changing it needs a new job.");
                }
                describe("The cluster's login node, as you would type it after ssh: an alias from your ~/.ssh/config, or user@host.");
                // the image: a dropdown of the ones used before, typed, or picked among the cluster's files
                {
                    const float browse = buttonWidth("Browse\xE2\x80\xA6", widgets::ButtonKind::Secondary);
                    fo.width = design(std::max(px(160), ImGui::GetContentRegionAvail().x - browse - gap));
                    {
                        const Field f("Worker image");
                        fo.hint = "/path/on/the/cluster/sirius-worker.sif";
                        std::vector<widgets::ComboItem> items;
                        for (const std::string& i : profile_.images) items.push_back({i, "used before", false});
                        bool picked = false;
                        widgets::editableCombo("##image", &profile_.container, items, fo, &picked);
                        fo.hint.clear();
                        widgets::tooltip("An Apptainer/Singularity image (.sif) on the cluster with SIRIUS's worker environment (the compiled sirius "
                                         "package, numpy, torch). The worker and the engine run in it. Changing it restarts the worker in the same job.");
                    }
                    ImGui::SameLine(0.0f, gap);
                    const Field f(" ");
                    widgets::ButtonOpts b;
                    b.enabled = editable && here;
                    b.tooltip = here ? std::string("Pick the image (*.sif) among the cluster's files") : std::string("Connect (or log in) first: then the cluster's files are listed");
                    if (widgets::button("Browse\xE2\x80\xA6##image", b)) {
                        std::string start;
                        const std::string c = trimmed(profile_.container);
                        if (const std::size_t slash = c.find_last_of('/'); slash != std::string::npos && slash > 0) start = c.substr(0, slash);
                        auto self = alive_;
                        app.defer([&app, start, self, this] {
                            app.showDialog(makeClusterBrowser(
                                app, start, false,
                                [self, this](const std::string& chosen) {
                                    std::string h, path;
                                    if (*self && splitClusterPath(chosen, h, path)) profile_.container = path;
                                },
                                ".sif"));
                        });
                    }
                }
                if (trimmed(profile_.container).empty())
                    describe("Required: an Apptainer/Singularity image (.sif) with SIRIUS's worker in it. None yet? Once the job runs, \"Build an image\" "
                             "below makes one.");
                else
                    describe("The Apptainer/Singularity image (.sif) SIRIUS's worker runs in.");
                // the data folders: the sets used before, typed, or added from the cluster's files
                {
                    const float add = buttonWidth("Add folder\xE2\x80\xA6", widgets::ButtonKind::Secondary);
                    fo.width = design(std::max(px(160), ImGui::GetContentRegionAvail().x - add - gap));
                    {
                        const Field f("Data folders");
                        fo.hint = "/data/lab, /scratch/me   (optional)";
                        std::vector<widgets::ComboItem> items;
                        for (const std::string& set : profile_.bindSets) items.push_back({set, "used before", false});
                        bool picked = false;
                        widgets::editableCombo("##binds", &profile_.bind, items, fo, &picked);
                        fo.hint.clear();
                        widgets::tooltip("Folders on the cluster made visible inside the image (apptainer --bind; comma separated, src[:dst[:ro]]). "
                                         "Each is checked before the worker starts. Changing them restarts the worker in the same job.");
                    }
                    ImGui::SameLine(0.0f, gap);
                    const Field f(" ");
                    widgets::ButtonOpts b;
                    b.enabled = editable && here;
                    b.tooltip = here ? std::string("Pick a folder among the cluster's files and add it") : std::string("Connect (or log in) first: then the cluster's files are listed");
                    if (widgets::button("Add folder\xE2\x80\xA6##bind", b)) {
                        const std::vector<std::string> binds = cluster::bindHostPaths(profile_.bind);
                        const std::string start = binds.empty() ? std::string() : binds.back();
                        auto self = alive_;
                        app.defer([&app, start, self, this] {
                            app.showDialog(makeClusterBrowser(app, start, true, [self, this](const std::string& chosen) {
                                std::string h, path;
                                if (!*self || !splitClusterPath(chosen, h, path) || path.empty()) return;
                                for (const std::string& b : cluster::bindHostPaths(profile_.bind))
                                    if (b == path) return;
                                std::string bind = trimmed(profile_.bind);
                                while (!bind.empty() && bind.back() == ',') bind.pop_back();
                                profile_.bind = bind.empty() ? path : bind + "," + path;
                            }));
                        });
                    }
                }
                describe("Optional: the folders on the cluster your datasets live in, made visible to the image. Your home folder always is.");
                if (const std::string w = cluster::emptyBindWarning(profile_); !w.empty() && !trimmed(profile_.container).empty() && !profile_.bind.empty())
                    describe(w + ".");
            }

            // --- advanced ----------------------------------------------------------------------

            // A dropdown of `items` (each a value and a line about it) with
            // "Custom…" at its end; a value not among them shows as the first
            // entry. Custom… turns it into a text field (a "List" link back).
            bool choice(const char* id, std::string* value, std::vector<widgets::ComboItem> items, widgets::FieldOpts fo, float popupWidth,
                        const std::string& emptyLabel) {
                bool& custom = custom_[id];
                if (custom) {
                    const float link = theme::textSize("List", 11).x + px(10);
                    fo.width = std::max(40.0f, fo.width - design(link));
                    ImGui::BeginGroup();
                    const bool changed = widgets::inputText(id, value, fo);
                    ImGui::SameLine(0.0f, px(4));
                    if (widgets::linkButton((std::string("List##") + id).c_str(), fo.enabled)) custom = false;
                    widgets::tooltip("Back to the list");
                    ImGui::EndGroup();
                    return changed;
                }
                std::vector<std::string> values;
                for (const widgets::ComboItem& i : items) values.push_back(i.value);
                int current = -1;
                for (std::size_t i = 0; i < values.size(); ++i)
                    if (values[i] == trimmed(*value)) current = static_cast<int>(i);
                if (current < 0) {
                    const std::string v = trimmed(*value);
                    items.insert(items.begin(), widgets::ComboItem{v.empty() ? emptyLabel : v, v.empty() ? "Slurm decides" : "set in this profile", false});
                    values.insert(values.begin(), v);
                    current = 0;
                }
                items.push_back(widgets::ComboItem{"Custom\xE2\x80\xA6", "type a value of your own", false});
                int pick = current;
                if (widgets::detailCombo(id, &pick, items, fo, popupWidth) && pick != current && pick >= 0) {
                    if (pick == static_cast<int>(items.size()) - 1) {
                        custom = true;
                        return false;
                    }
                    *value = values[static_cast<std::size_t>(pick)];
                    return true;
                }
                return false;
            }

            void drawAdvanced(App& app, ClusterLink& link, const cluster::Status& st, bool editable) {
                widgets::FieldOpts fo;
                fo.enabled = editable;
                const float gap = px(10);
                // --- the job ---
                group("Job", "What the job asks Slurm for. Log in fills these from the cluster; the lists are your settings' choices and, "
                             "marked so, what the cluster reports. Changing any of them needs a new job.");
                drawSlurmRow(fo);
                const cluster::PartitionChoice* mine = profile_.choice(trimmed(profile_.partition));
                const cluster::Partition* part = info_ ? cluster::findPartition(*info_, trimmed(profile_.partition)) : nullptr;
                const std::int64_t maxGpus = mine && mine->maxGpus >= 0 ? mine->maxGpus : (part && part->gpusPerNode > 0 ? part->gpusPerNode : 16);
                const std::int64_t maxCpus = mine && mine->maxCpus > 0 ? mine->maxCpus : (part && part->cpusPerNode > 0 ? part->cpusPerNode : 256);
                gpus_ = std::min(gpus_, maxGpus);
                cpus_ = std::min(cpus_, maxCpus);
                {
                    const float w = columnWidth(4, 10);
                    fo.width = design(w);
                    {
                        const Field f("Time limit");
                        std::string limit = part ? part->maxTime : std::string();
                        if (info_)
                            if (const auto it = info_->qosMaxWall.find(trimmed(profile_.qos)); it != info_->qosMaxWall.end() && !it->second.empty()) {
                                const long long q = cluster::slurmTimeSeconds(it->second), l = cluster::slurmTimeSeconds(limit);
                                if (q >= 0 && (l < 0 || q < l)) limit = it->second;
                            }
                        std::vector<widgets::ComboItem> items;
                        for (const std::string& t : cluster::timeChoices(mine, limit, profile_.time)) items.push_back({t, timeWords(t), false});
                        choice("##time", &profile_.time, items, fo, 0.0f, "(Slurm's default)");
                        widgets::tooltip("How long the job may hold its node: it ends then, and the worker with it. The most allowed: " +
                                         (limit.empty() ? std::string("not known (log in)") : limit) + ". Needs a new job.");
                    }
                    ImGui::SameLine(0.0f, gap);
                    {
                        const Field f("GPUs");
                        spinInt("##gpus", &gpus_, 0, maxGpus, 1, fo);
                        widgets::tooltip("GPUs the job asks for (0: none, the worker computes on the CPU). At most " + std::to_string(maxGpus) +
                                         " here. Needs a new job.");
                    }
                    ImGui::SameLine(0.0f, gap);
                    {
                        const Field f("CPUs");
                        spinInt("##cpus", &cpus_, 1, maxCpus, 1, fo);
                        widgets::tooltip("CPU cores the job asks for. At most " + std::to_string(maxCpus) + " here. Needs a new job.");
                    }
                    ImGui::SameLine(0.0f, gap);
                    const Field f("Memory");
                    widgets::inputText("##mem", &profile_.mem, fo);
                    std::string most = mine && !mine->maxMem.empty() ? mine->maxMem : std::string();
                    if (most.empty() && part && part->memPerNodeMB > 0) most = std::to_string(part->memPerNodeMB / 1024) + "G";
                    widgets::tooltip("Memory the job asks for: \"64G\", \"500M\". " + (most.empty() ? std::string() : "A node here has " + most + ". ") +
                                     "Needs a new job.");
                    if (!most.empty() && cluster::memoryMB(profile_.mem) > cluster::memoryMB(most))
                        describe("More memory than a node here has (" + most + "): the job would never start. Ask for less.");
                }
                drawPartitionNotes(app, link, st, editable);
                // --- the software ---
                group("Software", "Where SIRIUS is on the cluster and how the image is started. Changing these restarts the worker in the same job.");
                {
                    const float w = columnWidth(2, 10);
                    fo.width = design(w);
                    {
                        const Field f("SIRIUS checkout on the cluster");
                        fo.hint = "filled in at the login: <home>/sirius";
                        widgets::inputText("##checkout", &profile_.checkout, fo);
                        fo.hint.clear();
                        widgets::tooltip("A clone of this SIRIUS repository on the cluster, the same version as this application: the worker's Python "
                                         "code (app/python) and its start script come from there. Change it when you cloned it elsewhere.");
                    }
                    ImGui::SameLine(0.0f, gap);
                    const Field f("Container launcher");
                    fo.hint = "apptainer";
                    widgets::inputText("##launcher", &profile_.launcher, fo);
                    fo.hint.clear();
                    widgets::tooltip("apptainer, singularity, or a full path. When it is not on the PATH, `module load` of it is tried, then the "
                                     "other one. Change it only when your cluster names it otherwise.");
                }
                {
                    fo.width = design(ImGui::GetContentRegionAvail().x);
                    const Field f("Extra Python path (optional)");
                    fo.hint = "entries added to PYTHONPATH inside the image, : separated";
                    widgets::inputText("##pythonPath", &profile_.containerPythonPath, fo);
                    fo.hint.clear();
                    widgets::tooltip("Paths as the image sees them, put after the worker's own code. For Python packages you keep outside the image; "
                                     "usually empty.");
                }
                widgets::checkbox("Run SIRIUS's C++ engine on the node##engine", &profile_.engine, editable);
                widgets::tooltip("On: every step of a pipeline runs on the node with SIRIUS's own engine (CUDA where SIRIUS has it), and the "
                                 "results stay there until you look at them or export them. Off: only the Python steps run there. On unless an "
                                 "image has no engine.");
                if (profile_.engine) {
                    const float w = columnWidth(2, 10);
                    fo.width = design(w);
                    {
                        const Field f("Engine builds folder (optional)");
                        fo.hint = "empty: the image's own engine";
                        widgets::inputText("##engineBuilds", &profile_.engineBuilds, fo);
                        fo.hint.clear();
                        widgets::tooltip("A folder with one engine build per SIRIUS commit (<commit>/bin/sirius-cli and BUILD.json). The checks pick "
                                         "the build of this application, or one with the same operations, and bind it into the image. Empty: the "
                                         "engine inside the image.");
                    }
                    ImGui::SameLine(0.0f, gap);
                    const Field f("Engine executable (optional)");
                    fo.hint = "empty: picked from the builds";
                    widgets::inputText("##engineBin", &profile_.engineBin, fo);
                    fo.hint.clear();
                    widgets::tooltip("Names a sirius-cli outright, overriding the builds folder: for testing a build of your own. Usually empty.");
                }
                // --- storage ---
                group("Storage", "Where the engine keeps the steps' results on the node.");
                {
                    fo.width = design(ImGui::GetContentRegionAvail().x);
                    const Field f("Node cache folder (optional)");
                    fo.hint = "empty: the node's temporary folder";
                    fo.enabled = editable && profile_.engine;
                    widgets::inputText("##cache", &profile_.scratch, fo);
                    fo.hint.clear();
                    widgets::tooltip("A folder on the node (or a file system it sees) with room for the engine's cache and uploads: a node's local "
                                     "disk is fastest. Checked before the worker starts, bound into the image. Empty: the node's temporary folder. "
                                     "Changing it restarts the worker.");
                }
            }

            static std::string timeWords(const std::string& t) {
                const long long s = cluster::slurmTimeSeconds(t);
                return s < 0 ? std::string() : cluster::durationText(s);
            }

            // Partition, Account and QoS: the profile's choices, then what the cluster reports.
            void drawSlurmRow(widgets::FieldOpts fo) {
                const float w = columnWidth(3, 10);
                fo.width = design(w);
                const float listWidth = design(w * 3 + px(20));
                {
                    const Field f("Partition");
                    std::vector<widgets::ComboItem> items;
                    for (const cluster::PartitionChoice& c : profile_.choices) {
                        std::string detail = c.isDefault ? "your default" : "your settings";
                        if (info_)
                            if (const cluster::Partition* p = cluster::findPartition(*info_, c.name)) detail += " \xC2\xB7 " + cluster::partitionSummary(*p);
                        items.push_back({c.name, detail, false});
                    }
                    if (info_)
                        for (int pass = 0; pass < 2; ++pass)
                            for (const cluster::Partition& p : info_->partitions) {
                                if (profile_.choice(p.name)) continue;
                                const bool usable = cluster::hasAssociation(*info_, p.name);
                                if (usable != (pass == 0)) continue;
                                items.push_back({p.name, std::string("from the cluster \xC2\xB7 ") + (usable ? "" : "no association \xC2\xB7 ") + cluster::partitionSummary(p),
                                                 false});
                            }
                    if (openList_ && info_) {
                        // a screenshot: the list as the user sees it on a click
                        // (the dropdown's own popup, as Dear ImGui's BeginCombo names it)
                        ImGui::OpenPopupEx(ImHashStr("##ComboPopup", 0, ImGui::GetID("##partition")));
                        openList_ = false;
                    }
                    if (choice("##partition", &profile_.partition, items, fo, listWidth, "(the cluster's default)")) choosePartition();
                    widgets::tooltip("Where the job runs. Your settings' partitions come first, then those the cluster lists (after Log in). "
                                     "Picking one fills the account, QoS and limits that go with it. Needs a new job.");
                }
                ImGui::SameLine(0.0f, px(10));
                {
                    const Field f("Account");
                    std::vector<widgets::ComboItem> items;
                    std::set<std::string> seen;
                    if (const cluster::PartitionChoice* c = profile_.choice(trimmed(profile_.partition)))
                        for (const std::string& a : c->accounts)
                            if (seen.insert(a).second) items.push_back({a, "your settings", false});
                    if (info_)
                        for (const std::string& a : cluster::accountsFor(*info_, trimmed(profile_.partition)))
                            if (seen.insert(a).second) items.push_back({a, "from the cluster", false});
                    choice("##account", &profile_.account, items, fo, 0.0f, "(your default)");
                    widgets::tooltip("The account the job is charged to: one you have an association with for this partition. Empty: your "
                                     "default account. Needs a new job.");
                }
                ImGui::SameLine(0.0f, px(10));
                const Field f("QoS");
                std::vector<widgets::ComboItem> items;
                std::set<std::string> seen;
                auto wall = [this](const std::string& q) {
                    if (!info_) return std::string();
                    const auto it = info_->qosMaxWall.find(q);
                    return it == info_->qosMaxWall.end() || it->second.empty() ? std::string() : " \xC2\xB7 up to " + it->second;
                };
                if (const cluster::PartitionChoice* c = profile_.choice(trimmed(profile_.partition)))
                    for (const std::string& q : c->qos)
                        if (seen.insert(q).second) items.push_back({q, "your settings" + wall(q), false});
                if (info_) {
                    std::vector<std::string> qos = cluster::qosFor(*info_, trimmed(profile_.partition), trimmed(profile_.account));
                    if (!info_->associationsKnown)
                        for (const auto& entry : info_->qosMaxWall) qos.push_back(entry.first);
                    for (const std::string& q : qos)
                        if (seen.insert(q).second) items.push_back({q, "from the cluster" + wall(q), false});
                }
                choice("##qos", &profile_.qos, items, fo, design(w * 1.4f), "(the default QoS)");
                widgets::tooltip("The quality of service: it sets the job's priority and longest time. Empty: the association's default. "
                                 "Needs a new job.");
            }

            // A partition picked: the rest filled in from the profile's
            // choice, or from the association, and brought within its limits.
            void choosePartition() {
                profile_.gpus = static_cast<int>(gpus_);
                profile_.cpus = static_cast<int>(cpus_);
                if (const cluster::PartitionChoice* c = profile_.choice(trimmed(profile_.partition))) {
                    const cluster::PartitionChoice copy = *c;
                    filled_ = cluster::applyChoice(profile_, copy);
                } else if (info_ && cluster::findPartition(*info_, trimmed(profile_.partition))) {
                    filled_ = cluster::choosePartition(profile_, *info_, trimmed(profile_.partition));
                }
                gpus_ = profile_.gpus;
                cpus_ = profile_.cpus;
            }

            // Once the cluster has said what it has (after a login): what the
            // profile leaves empty, as the session fills it.
            void fillFromCluster(ClusterLink& link) {
                if (!info_ || filledFor_ == info_->host) return;
                filledFor_ = info_->host;
                profile_.gpus = static_cast<int>(gpus_);
                profile_.cpus = static_cast<int>(cpus_);
                const std::vector<std::string> filled = cluster::fillFromCluster(profile_, *info_);
                gpus_ = profile_.gpus;
                cpus_ = profile_.cpus;
                if (filled.empty()) return;
                filled_ = filled;
                (void)link;
            }

            // Under the Slurm rows: what the chosen partition is, what to know
            // before submitting there, and where the lists came from.
            void drawPartitionNotes(App& app, ClusterLink& link, const cluster::Status& st, bool editable) {
                const std::string host = trimmed(profile_.host);
                bool refreshing = false;
                std::string refreshError;
                {
                    const std::lock_guard<std::mutex> g(refresh_->m);
                    refreshing = refresh_->running;
                    refreshError = refresh_->error;
                }
                const bool querying = refreshing || link.session().queryingClusterInfo();
                if (querying) app.requestRedraw(2);
                ImGui::BeginGroup();
                const std::string partName = trimmed(profile_.partition);
                if (info_) {
                    if (const cluster::Partition* part = cluster::findPartition(*info_, partName)) {
                        widgets::textWrapped(part->name + ": " + cluster::partitionSummary(*part, true), 11, theme::kNeutral700);
                        if (!cluster::hasAssociation(*info_, part->name))
                            widgets::textWrapped("You have no association for " + part->name + " (sacctmgr): sbatch will most likely refuse the job. Pick "
                                                                                               "another partition, or ask your cluster's support for access.",
                                                 11, theme::kAccentText);
                        const std::string warning = cluster::partitionWarning(*part);
                        if (!warning.empty()) widgets::textWrapped(warning, 11, kAmber, theme::Weight::SemiBold);
                        if (!profile_.choice(part->name)) {
                            if (widgets::linkButton(("Add " + part->name + " to my settings##addChoice").c_str(), editable))
                                profile_.choices.push_back(cluster::choiceFromCluster(*info_, part->name));
                            widgets::tooltip("Keep " + part->name + " in this profile's list of partitions, with its accounts, QoS and limits as the "
                                                                    "cluster reports them (Save writes it to the settings file).");
                        }
                    } else if (!partName.empty() && info_->error.empty() && !profile_.choice(partName)) {
                        widgets::textWrapped(partName + " is not one of " + host + "'s partitions: sbatch will say whether it exists.", 11, kAmber);
                    }
                    if (!info_->error.empty()) widgets::textWrapped(info_->error + ".", 11, theme::kAccentText);
                    for (const std::string& n : info_->notes) widgets::textWrapped(n + ".", 11, theme::kNeutral600);
                }
                if (!filled_.empty()) {
                    std::string line;
                    for (const std::string& f : filled_) line += (line.empty() ? "" : ", ") + f;
                    widgets::textWrapped("Filled in: " + line + ".", 11, theme::kNeutral600);
                }
                if (!refreshError.empty()) widgets::textWrapped(refreshError, 11, theme::kAccentText);
                // where the lists come from, and asking again
                std::string line;
                if (querying) line = "Listing " + host + "'s partitions\xE2\x80\xA6";
                else if (info_)
                    line = std::to_string(info_->partitions.size()) + " partitions on " + host + (info_->user.empty() ? std::string() : " for " + info_->user);
                else
                    line = "Log in to fill these from the cluster: its partitions, your accounts and QoS.";
                widgets::text(line, 11, theme::kNeutral600);
                ImGui::SameLine(0.0f, px(8));
                const bool loggedIn = st.sshUp && st.host == host;
                const bool held = st.state == cluster::State::JobReady || st.state == cluster::State::Connected;
                const char* label = info_ ? "Refresh##partitions" : (loggedIn ? "List partitions##partitions" : "Log in##partitions");
                if (widgets::linkButton(label, (editable || held) && !querying && !host.empty())) {
                    if (loggedIn) {
                        startRefresh(app, link);
                    } else {
                        commit(link);
                        link.logIn(profileToUse());
                    }
                }
                widgets::tooltip(loggedIn ? "Ask " + host + " again: sinfo, sacctmgr, scontrol"
                                          : std::string("The SSH login only, then sinfo and sacctmgr: nothing is submitted"));
                ImGui::EndGroup();
            }

            void startRefresh(App& app, ClusterLink& link) {
                {
                    const std::lock_guard<std::mutex> g(refresh_->m);
                    if (refresh_->running) return;
                    refresh_->running = true;
                    refresh_->error.clear();
                }
                filledFor_.clear();
                auto shared = refresh_;
                cluster::Session* session = &link.session();
                Bridge* bridge = &app.bridge();
                refreshThread_.start([shared, session, bridge] {
                    std::string error;
                    try {
                        session->refreshClusterInfo();
                    } catch (const ssh::SshError& e) {
                        error = std::string("The partitions could not be listed: ") + e.what() + (e.detail.empty() ? std::string() : ": " + e.detail);
                    } catch (const std::exception& e) {
                        error = std::string("The partitions could not be listed: ") + e.what();
                    }
                    const std::lock_guard<std::mutex> g(shared->m);
                    shared->running = false;
                    if (shared->gone) return;
                    shared->error = error;
                    bridge->post([] {});
                });
            }

            // The profile as the fields have it, for a login, a job or a worker.
            cluster::Profile profileToUse() {
                for (std::string* s : {&profile_.host, &profile_.checkout, &profile_.container, &profile_.launcher, &profile_.bind,
                                       &profile_.containerPythonPath, &profile_.engineBuilds, &profile_.engineBin, &profile_.scratch,
                                       &profile_.partition, &profile_.account, &profile_.qos, &profile_.time, &profile_.mem})
                    *s = trimmed(*s);
                if (profile_.launcher.empty()) profile_.launcher = "apptainer";
                profile_.gpus = static_cast<int>(gpus_);
                profile_.cpus = static_cast<int>(cpus_);
                profile_.name = name_;
                // the image and the data folders, first in their dropdowns from now on
                auto remember = [](std::vector<std::string>& list, const std::string& v) {
                    if (v.empty()) return;
                    list.erase(std::remove(list.begin(), list.end(), v), list.end());
                    list.insert(list.begin(), v);
                    if (list.size() > 8) list.resize(8);
                };
                remember(profile_.images, profile_.container);
                remember(profile_.bindSets, profile_.bind);
                lastHost_ = profile_.host;
                return profile_;
            }

            // The fields saved before the session gets them (what Connect uses is what the file has).
            void commit(ClusterLink& link) {
                profileToUse();
                save(link);
            }

            // --- the steps ---------------------------------------------------------------------

            void drawSteps(const cluster::Status& st, int from, int to) {
                for (int i = from; i < to; ++i) {
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

            // A step heading: its number, its title, a line of what it does, and its button flush right.
            bool stepHeading(const char* number, const std::string& title, const std::string& what, const std::string& button, bool primary,
                             bool enabled, const std::string& tip) {
                ImGui::BeginGroup();
                widgets::text(std::string(number) + "  " + title, 13, theme::kText, theme::Weight::ExtraBold);
                describe(what);
                ImGui::EndGroup();
                if (button.empty()) return false;
                const widgets::ButtonKind kind = primary ? widgets::ButtonKind::Primary : widgets::ButtonKind::Secondary;
                const float bw = buttonWidth(button, kind);
                ImGui::SameLine();
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - bw));
                widgets::ButtonOpts b;
                b.kind = kind;
                b.enabled = enabled;
                b.tooltip = tip;
                return widgets::button(button.c_str(), b);
            }

            void drawJob(App& app, ClusterLink& link, const cluster::Status& st, bool busy) {
                (void)app;
                const bool connecting = st.state == cluster::State::Connecting;
                const bool held = st.state == cluster::State::JobReady || st.state == cluster::State::Connected || st.state == cluster::State::Starting;
                std::string button = "Connect##job";
                std::string tip = "Log in, then ask Slurm for a job with the settings above and wait until it runs. Nothing of SIRIUS has to be on "
                                  "the cluster for this.";
                if (connecting) {
                    button = "Stop##job";
                    tip = "Stop logging in or waiting (a job already submitted is left for Connect to take up)";
                } else if (held) {
                    button = "Disconnect\xE2\x80\xA6##job";
                    tip = "Close the connection; you are asked whether to cancel the job";
                } else if (st.state == cluster::State::Disconnected) {
                    button = "Connect again##job";
                }
                const bool valid = !trimmed(profile_.host).empty();
                const std::string what = held ? "Job " + st.jobId + " holds " + st.node + (st.jobLimitSeconds >= 0 ? timeLeft(st) : std::string()) + "."
                                              : "Log in and get a job on the cluster that holds a node for SIRIUS.";
                if (stepHeading("1", "The job", what, button, !held && !connecting, connecting || held || (valid && !busy), valid || held ? tip : "Fill in the cluster's name first")) {
                    if (connecting) {
                        link.session().cancelConnect();
                    } else if (held) {
                        link.disconnectAsking();
                    } else {
                        commit(link);
                        link.connectJob(profileToUse());
                    }
                }
                if (!valid && !held) describe("Fill in Cluster (SSH host) to connect.");
                drawSteps(st, 0, cluster::kJobStepCount);
            }

            static std::string timeLeft(const cluster::Status& st) {
                if (st.jobStarted == std::chrono::steady_clock::time_point{}) return {};
                const long long used = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - st.jobStarted).count();
                return " \xC2\xB7 " + cluster::durationText(std::max(0LL, st.jobLimitSeconds - used)) + " left";
            }

            void drawWorker(App& app, ClusterLink& link, const cluster::Status& st, bool busy) {
                widgets::vspace(6);
                const bool held = st.state == cluster::State::JobReady || st.state == cluster::State::Connected;
                const bool starting = st.state == cluster::State::Starting;
                const bool connected = st.state == cluster::State::Connected;
                std::string button = "Start worker##worker";
                std::string tip = "Check the image and start SIRIUS's worker in the job (srun): then the HPC backend runs there.";
                if (starting) {
                    button = "Stop##worker";
                    tip = "Stop starting the worker; the job stays";
                } else if (connected) {
                    button = "Stop worker##worker";
                    tip = "End the worker; the job stays, and Start worker starts it again (with other settings, a new image)";
                }
                const bool imageSet = !trimmed(profile_.container).empty();
                const std::string what = connected ? "SIRIUS's worker answers on " + st.node + ": the HPC backend runs there."
                                                   : "Start SIRIUS's worker in the job, in the worker image.";
                const bool enabled = starting || connected || (held && imageSet);
                std::string why = tip;
                if (!held && !starting) why = "Connect first: the worker starts in the job";
                else if (!imageSet && !connected) why = "Set Worker image first (or build one below)";
                if (stepHeading("2", "The worker", what, button, held && !connected, enabled, why)) {
                    if (starting) link.session().cancelConnect();
                    else if (connected) link.stopWorker();
                    else {
                        commit(link);
                        link.startWorker(profileToUse());
                    }
                }
                if (held && !imageSet && !connected) describe("Set Worker image above to start the worker, or build an image below.");
                drawSteps(st, cluster::kJobStepCount, cluster::kStepCount);
                drawBuild(app, link, st, held && !busy);
            }

            // Build an image in the held job, when the cluster lets the user.
            void drawBuild(App& app, ClusterLink& link, const cluster::Status& st, bool held) {
                const cluster::BuildStatus& b = st.build;
                const bool running = b.phase == cluster::BuildStatus::Phase::Probing || b.phase == cluster::BuildStatus::Phase::Building;
                widgets::vspace(4);
                if (!disclosure("##build", "Build an image", &buildOpen_)) return;
                describe("Builds SIRIUS's worker image with apptainer build --fakeroot inside the job, from the definition file in the SIRIUS "
                         "checkout. Takes a while; only where the cluster allows unprivileged builds (checked first).");
                if (!held && !running) {
                    describe("Connect first: the build runs in the job.");
                    return;
                }
                if (b.supported && !*b.supported) {
                    widgets::textWrapped("This cluster does not let you build images: " + b.why, 11, theme::kAccentText);
                    describe("Use an image someone built: pick it under Worker image.");
                    return;
                }
                widgets::FieldOpts fo;
                fo.enabled = !running;
                const float w = columnWidth(2, 10);
                fo.width = design(w);
                if (defFile_.empty()) defFile_ = trimmed(profile_.defFile);
                if (imageOut_.empty()) {
                    std::string home = st.home;
                    if (home.empty() && info_) home = info_->home;
                    imageOut_ = (home.empty() ? std::string("~") : home) + "/sirius-worker.sif";
                }
                {
                    const Field f("Definition file");
                    fo.hint = trimmed(profile_.checkout).empty() ? std::string("<checkout>/containers/sirius-worker.def")
                                                                 : trimmed(profile_.checkout) + "/containers/sirius-worker.def";
                    widgets::inputText("##defFile", &defFile_, fo);
                    fo.hint.clear();
                    widgets::tooltip("The Apptainer definition the image is built from: SIRIUS's own (containers/sirius-worker.def in the checkout) "
                                     "unless you have another. On the cluster.");
                }
                ImGui::SameLine(0.0f, px(10));
                {
                    const Field f("New image");
                    widgets::inputText("##imageOut", &imageOut_, fo);
                    widgets::tooltip("Where the new image goes, on the cluster. An existing file is never written over.");
                }
                widgets::ButtonOpts bo;
                bo.small = true;
                bo.enabled = held && !running && !trimmed(imageOut_).empty();
                bo.tooltip = "Check that this cluster allows the build, then build the image in the job";
                if (widgets::button("Build##image", bo)) {
                    profile_.defFile = trimmed(defFile_);
                    commit(link);
                    link.buildImage(profileToUse(), trimmed(defFile_), trimmed(imageOut_));
                }
                if (running) {
                    ImGui::SameLine(0.0f, px(8));
                    widgets::ButtonOpts so;
                    so.small = true;
                    so.kind = widgets::ButtonKind::Ghost;
                    if (widgets::button("Stop the build", so)) link.session().cancelBuild();
                }
                ImGui::SameLine(0.0f, px(10));
                std::string state;
                switch (b.phase) {
                    case cluster::BuildStatus::Phase::None: break;
                    case cluster::BuildStatus::Phase::Probing: state = "Checking that this cluster allows the build\xE2\x80\xA6"; break;
                    case cluster::BuildStatus::Phase::Building:
                        state = "Building " + b.image + " \xC2\xB7 " +
                                cluster::durationText(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - b.started).count());
                        break;
                    case cluster::BuildStatus::Phase::Done: state = "Built " + b.image + "."; break;
                    case cluster::BuildStatus::Phase::Failed: state = b.error; break;
                }
                if (!state.empty())
                    widgets::textWrapped(state, 11, b.phase == cluster::BuildStatus::Phase::Failed ? theme::kAccentText : theme::kNeutral700);
                if (!b.log.empty()) outputBlock("##buildLog", b.log, 90);
                // the result into the profile, once
                if (b.phase == cluster::BuildStatus::Phase::Done && b.image != adopted_) {
                    adopted_ = b.image;
                    profile_.container = b.image;
                    profileToUse();
                    keep();
                    app.wb().logLine("Cluster: the new image " + b.image + " is this profile's worker image now (Save keeps it).");
                }
            }

            // Connected (or a job held) and the fields differ from what runs:
            // what it takes, and the button that does it.
            void drawApply(App& app, ClusterLink& link, const cluster::Status& st) {
                (void)app;
                const bool held = st.state == cluster::State::JobReady || st.state == cluster::State::Connected;
                if (!held) return;
                cluster::Profile now = profile_;
                now.gpus = static_cast<int>(gpus_);
                now.cpus = static_cast<int>(cpus_);
                const cluster::ProfileChange c = cluster::profileChange(link.session().profile(), now);
                if (c.fields.empty()) return;
                std::string fields;
                for (const std::string& f : c.fields) fields += (fields.empty() ? "" : ", ") + f;
                widgets::beginCard("##apply", true, 8);
                if (c.newJob) {
                    widgets::textWrapped("Changed: " + fields + ". That takes a new job: the current one is cancelled first (you are asked).", 12,
                                         theme::kText);
                    widgets::ButtonOpts b;
                    b.kind = widgets::ButtonKind::Primary;
                    b.small = true;
                    b.tooltip = "Cancel job " + st.jobId + " and ask for a new one with these settings";
                    if (widgets::button("Apply: new job", b)) {
                        commit(link);
                        link.newJobAsking(profileToUse());
                    }
                } else if (st.state == cluster::State::Connected) {
                    widgets::textWrapped("Changed: " + fields + ". The worker restarts in the same job (job " + st.jobId + " stays).", 12, theme::kText);
                    widgets::ButtonOpts b;
                    b.kind = widgets::ButtonKind::Primary;
                    b.small = true;
                    b.tooltip = "Stop the worker and start it again with these settings; what it held is lost";
                    if (widgets::button("Apply: restart the worker", b)) {
                        commit(link);
                        link.startWorker(profileToUse());
                    }
                } else {
                    widgets::textWrapped("Changed: " + fields + ". Start worker uses them.", 12, theme::kText);
                }
                widgets::endCard();
                widgets::vspace(4);
            }

            void drawOutcome(const cluster::Status& st) {
                if (st.state == cluster::State::Connected) {
                    widgets::rule(theme::kRule);
                    widgets::text("Connected to the worker on " + st.node + " (job " + st.jobId + ")", 12, kGreen, theme::Weight::SemiBold);
                    const WorkerCapabilities& c = st.caps;
                    if (c.engine.is_object())
                        widgets::textWrapped("SIRIUS engine " + c.engine.value("build", c.version) + " \xC2\xB7 " + c.device + " on " + c.hostname +
                                                 ": every step of a pipeline runs there",
                                             11, theme::kNeutral700);
                    else
                        widgets::textWrapped("sirius_worker " + c.version + " \xC2\xB7 Python " + c.python + " \xC2\xB7 " + c.device + " on " + c.hostname, 11,
                                             theme::kNeutral700);
                    std::string kinds;
                    for (const std::string& m : c.methods)
                        if (m.rfind("run:", 0) == 0) kinds += (kinds.empty() ? "" : ", ") + m.substr(4);
                    if (!kinds.empty()) widgets::textWrapped("Python steps it runs: " + kinds, 11, theme::kNeutral600);
                    if (c.tiffReader.empty())
                        widgets::textWrapped("The sirius package is not in the worker's image: TIFF datasets on the cluster cannot be opened. Build the "
                                             "image again from the SIRIUS checkout.",
                                             11, kAmber);
                    if (const std::string why = cluster::gpuUnusableReason(st.node, c); !why.empty() && !c.gpus.empty())
                        widgets::textWrapped(why + ". Until it can, the steps run on the CPU of the job.", 11, kAmber);
                    widgets::textWrapped("The HPC backend runs there now; File \xE2\x96\xB8 Open dataset\xE2\x80\xA6 \xE2\x96\xB8 Cluster opens "
                                         "datasets that stay on the cluster.",
                                         11, theme::kNeutral600);
                    return;
                }
                if (st.reason.empty() || (st.state != cluster::State::Disconnected && st.state != cluster::State::JobReady)) return;
                if (st.state == cluster::State::Disconnected && st.reason.rfind("disconnected", 0) == 0) return;   // asked for: nothing to explain
                widgets::rule(theme::kRule);
                // the failed step says why already; a drop after the connect has no step to say it
                bool told = false;
                for (const cluster::StepState& s : st.steps) told = told || (s.status == cluster::StepStatus::Failed && s.detail == st.reason);
                if (!told) widgets::textWrapped(st.reason, 12, theme::kAccentText);
                if (!st.remoteOutput.empty()) {
                    widgets::caption("What the cluster said");
                    outputBlock("##remote", st.remoteOutput, 90);
                }
                if (!st.fix.empty()) {
                    widgets::caption("What to do");
                    widgets::textWrapped(st.fix, 12, theme::kText);
                    if (widgets::chipButton("Copy")) ImGui::SetClipboardText(st.fix.c_str());
                }
            }

            void drawActions(App& app, ClusterLink& link, const cluster::Status& st) {
                widgets::ButtonOpts b;
                b.small = true;
                b.enabled = st.sshUp;
                b.tooltip = st.sshUp ? std::string("The cluster's files, through this SSH session") : std::string("Connect (or log in) first");
                if (widgets::button("Browse cluster files\xE2\x80\xA6", b)) {
                    app.defer([&app] {
                        app.showDialog(makeClusterBrowser(app, {}, false, [&app](const std::string& path) { app.openDatasetPath(path); }));
                    });
                }
                const bool changed = dirty();
                const float gap = px(8);
                const float total = buttonWidth("Close", widgets::ButtonKind::Ghost) + gap + buttonWidth("Save", widgets::ButtonKind::Primary);
                ImGui::SameLine();
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - total));
                widgets::ButtonOpts ghost;
                ghost.kind = widgets::ButtonKind::Ghost;
                if (widgets::button("Close##clusterClose", ghost) && canClose(app)) close();
                ImGui::SameLine(0.0f, gap);
                widgets::ButtonOpts save;
                save.kind = widgets::ButtonKind::Primary;
                save.enabled = changed;
                save.tooltip = changed ? "Write the cluster profiles to the settings file (" + settings().filePath() + ")" : std::string("Nothing changed");
                if (widgets::button("Save##clusterSave", save)) {
                    profileToUse();
                    this->save(link);
                }
            }

            cluster::ProfileBook book_;
            nlohmann::json savedBook_;
            std::string name_;
            cluster::Profile profile_;
            std::int64_t gpus_ = 1, cpus_ = 8;
            std::optional<cluster::ClusterInfo> info_;   // this frame's, for the profile's host
            std::vector<std::string> filled_;            // what the last pick or login filled in
            std::string filledFor_;                      // the host whose info filled the profile already
            std::string lastHost_;
            bool hostEditing_ = false;
            bool autoName_ = false;                      // made here as "New cluster": named after its host on save
            bool renaming_ = false;
            std::string renameText_;
            std::map<std::string, bool> custom_;         // a choice field typed into instead of picked
            bool openList_ = false;
            bool advanced_ = false;
            bool buildOpen_ = false;
            int scrollDown_ = 0;   // frames left to keep the body scrolled (screenshots)
            std::string scrollTo_;
            std::string defFile_, imageOut_, adopted_;
            std::shared_ptr<RefreshShared> refresh_ = std::make_shared<RefreshShared>();
            DialogThread refreshThread_;
            std::shared_ptr<bool> alive_ = std::make_shared<bool>(true);   // GUI thread: a browser's pick reaches the dialog only while it lives
        };

        // --- one ssh prompt ------------------------------------------------------------

        class PromptDialog final : public Dialog {
        public:
            explicit PromptDialog(std::shared_ptr<ClusterPrompt> p) : p_(std::move(p)) {
                // room for any answer up front: a std::string that grows
                // leaves its old buffer, password and all, to the heap
                answer_.reserve(1024);
            }
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
                fieldId_ = ImGui::GetItemID();
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
                // Dear ImGui's own copies of the field's text, then ours
                widgets::forgetInputText(fieldId_);
                secureWipe(answer_);
                p_->answered.store(true);
            }

            std::shared_ptr<ClusterPrompt> p_;
            std::string answer_;
            ImGuiID fieldId_ = 0;
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
            ClusterBrowser(App& app, std::string start, bool folders, std::function<void(const std::string&)> chosen, std::string extension)
                : folders_(folders), chosen_(std::move(chosen)), extension_(std::move(extension)) {
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
                    if (!e.dir && !extension_.empty() &&
                        (e.name.size() < extension_.size() || e.name.compare(e.name.size() - extension_.size(), extension_.size(), extension_) != 0))
                        continue;
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
            std::string extension_;   // "" = every file
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
                                               std::function<void(const std::string& clusterPath)> chosen, const std::string& extension) {
        return std::make_shared<ClusterBrowser>(app, start, folders, std::move(chosen), extension);
    }

} // namespace sirius::app::gui
