// Process ▸ Connect to cluster…, the cluster button in the title bar: a
// wizard of three pages, and a summary once it is set up.
//
//   1 Connect   the cluster's name (an SSH config Host, or user@host) and
//               Connect: the login, its outcome in plain words (ssh's own
//               output under Details). The ⋯ menu holds the profiles (one
//               per cluster, kept in the settings file as [cluster.<name>]:
//               switch, new, rename, delete, import, export) and the
//               settings file itself.
//   2 Job       the node type (the partitions, each with a line of what its
//               nodes have), account and QoS, GPUs, CPUs, memory and time
//               bounded by it; the worker image and the data folders (the
//               cluster browser picks them); Start job and its live queue
//               status. More options: the software, the node cache, building
//               an image.
//   3 Worker    Start worker in the job, then its health report: the node,
//               the GPUs and CUDA, torch, the sirius package, nvTIFF, the
//               engine's build, the CPUs, memory, data folders, time left.
//               Finish switches the backend to HPC.
//   Summary     a session set up already: where it runs and how it is, with
//               Disconnect, Change job, Restart worker and Another cluster.
//
// Which page opens, and when Next and Finish may be pressed, is decided in
// core/cluster_wizard.hpp (GUI-free, tested). Back keeps what was entered.
// The choices are saved to the settings file on Connect, Start job and
// Start worker (and on closing). Not modal: the window stays usable while a
// job waits in the queue.
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

#include "core/build_info.hpp"
#include "core/cluster_profiles.hpp"
#include "core/cluster_wizard.hpp"
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
        using cluster::wizard::Page;

        constexpr ImU32 kGreen = theme::rgb(0x2e, 0x7d, 0x32);
        constexpr ImU32 kAmber = theme::rgb(0xb2, 0x6a, 0x00);
        constexpr ImU32 kWhite = theme::rgb(0xff, 0xff, 0xff);

        // A round mark at the start of a line: done, failed, a warning, a
        // note, something running, or not yet.
        enum class Ink { Ok,
                         Fail,
                         Warn,
                         Info,
                         Busy,
                         Pending };

        Ink inkOf(cluster::wizard::Mark m) {
            switch (m) {
                case cluster::wizard::Mark::Ok: return Ink::Ok;
                case cluster::wizard::Mark::Warn: return Ink::Warn;
                case cluster::wizard::Mark::Fail: return Ink::Fail;
                case cluster::wizard::Mark::Info: return Ink::Info;
            }
            return Ink::Info;
        }

        void drawMark(ImDrawList* dl, ImVec2 c, float d, Ink k) {
            const float r = d * 0.5f;
            switch (k) {
                case Ink::Ok:
                    dl->AddCircleFilled(c, r, kGreen);
                    drawIcon(dl, c, d * 0.8f, Icon::Check, kWhite, px(2));
                    break;
                case Ink::Fail:
                    dl->AddCircleFilled(c, r, theme::kAccent);
                    drawIcon(dl, c, d * 0.7f, Icon::Close, kWhite, px(2));
                    break;
                case Ink::Warn:
                    dl->AddCircleFilled(c, r, kAmber);
                    dl->AddLine(ImVec2(c.x, c.y - r * 0.5f), ImVec2(c.x, c.y + r * 0.1f), kWhite, px(2));
                    dl->AddCircleFilled(ImVec2(c.x, c.y + r * 0.45f), px(1.2f), kWhite);
                    break;
                case Ink::Info:
                    dl->AddCircle(c, r - px(0.75f), theme::kNeutral500, 0, px(1.5f));
                    dl->AddLine(ImVec2(c.x, c.y - r * 0.05f), ImVec2(c.x, c.y + r * 0.5f), theme::kNeutral600, px(1.5f));
                    dl->AddCircleFilled(ImVec2(c.x, c.y - r * 0.42f), px(1.1f), theme::kNeutral600);
                    break;
                case Ink::Busy: {
                    const float a0 = static_cast<float>(ImGui::GetTime() * 4.0);
                    dl->PathArcTo(c, r * 0.8f, a0, a0 + 4.2f, 16);
                    dl->PathStroke(theme::kAccent, 0, px(2));
                    break;
                }
                case Ink::Pending: dl->AddCircle(c, r * 0.8f, theme::kNeutral400, 0, px(1.5f)); break;
            }
        }

        // A mark as an item `lineHeight` tall, centred on it.
        void mark(Ink k, float side = 14.0f, float lineHeight = 0.0f) {
            const float d = px(side);
            const float h = std::max(d, lineHeight > 0.0f ? lineHeight : ImGui::GetTextLineHeight());
            const ImVec2 at = ImGui::GetCursorScreenPos();
            drawMark(ImGui::GetWindowDrawList(), ImVec2(at.x + d * 0.5f, at.y + h * 0.5f), d, k);
            ImGui::Dummy(ImVec2(d, h));
        }

        // A read-only, monospace, selectable block.
        void outputBlock(const char* id, const std::string& text, float heightPx) {
            std::string copy = text;
            widgets::FieldOpts fo;
            fo.readOnly = true;
            fo.monospace = true;
            widgets::inputTextMultiline(id, &copy, heightPx, fo);
        }

        // One line of plain words under a field: what it is for.
        void describe(const std::string& text, ImU32 color = theme::kNeutral600) {
            widgets::textWrapped(text, 11, color, theme::Weight::Regular, ImGui::GetContentRegionAvail().x);
        }

        // A heading that folds what is under it; true while open.
        bool disclosure(const char* id, const std::string& label, bool* open, float fontPx = 12.0f) {
            ImGui::PushID(id);
            const ImVec2 at = ImGui::GetCursorScreenPos();
            const float h = theme::textSize(label, fontPx, theme::Weight::SemiBold).y + px(6);
            const float w = theme::textSize(label, fontPx, theme::Weight::SemiBold).x + px(22);
            if (ImGui::InvisibleButton("##fold", ImVec2(w, h))) *open = !*open;
            const bool hov = ImGui::IsItemHovered();
            if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            ImDrawList* dl = ImGui::GetWindowDrawList();
            drawIcon(dl, ImVec2(at.x, at.y + (h - px(12)) * 0.5f), ImVec2(at.x + px(12), at.y + (h + px(12)) * 0.5f),
                     *open ? Icon::ChevronDown : Icon::ChevronRight, hov ? theme::kAccent : theme::kNeutral700);
            widgets::drawTextIn(dl, ImVec2(at.x + px(16), at.y), ImVec2(at.x + w, at.y + h), label, fontPx, hov ? theme::kAccent : theme::kNeutral700,
                                theme::Weight::SemiBold, 0.0f, 0.5f);
            ImGui::PopID();
            return *open;
        }

        // A section's heading inside a page.
        void section(const char* title) {
            widgets::vspace(2);
            widgets::text(title, 12, theme::kText, theme::Weight::Bold);
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
                // Screenshots only: a page, the partition list open, More options, Details.
                if (app.unattended()) {
                    const std::string page = host::environment("SIRIUS_TEST_CLUSTER_PAGE");
                    if (page == "connect") forcePage_ = Page::Connect;
                    else if (page == "job") forcePage_ = Page::Job;
                    else if (page == "worker") forcePage_ = Page::Worker;
                    else if (page == "summary") forcePage_ = Page::Summary;
                    openList_ = !host::environment("SIRIUS_TEST_OPEN_PARTITIONS").empty();
                    moreOpen_ = !host::environment("SIRIUS_TEST_CLUSTER_MORE").empty();
                    buildOpen_ = !host::environment("SIRIUS_TEST_CLUSTER_BUILD").empty();
                    detailsOpen_ = !host::environment("SIRIUS_TEST_CLUSTER_DETAILS").empty();
                }
            }
            ~ClusterDialog() override { *alive_ = false; }

            std::string title() const override { return "Connect to cluster"; }
            ImVec2 size() const override { return ImVec2(600, 0); }
            bool modal() const override { return false; }

            // What was entered is kept: closing saves it (nothing secret is in it).
            bool canClose(App& app) override {
                if (dirty()) {
                    profileToUse();
                    save(app.cluster());
                }
                return true;
            }
            void closed(App& app) override {
                if (dirty()) {
                    profileToUse();
                    save(app.cluster());
                }
            }

            void draw(App& app) override {
                ClusterLink& link = app.cluster();
                const cluster::Status st = link.status();
                const auto now = std::chrono::steady_clock::now();
                if (cluster::wizard::busy(st) || st.build.phase == cluster::BuildStatus::Phase::Probing ||
                    st.build.phase == cluster::BuildStatus::Phase::Building)
                    app.requestRedraw(2);   // the running marks turn, the waits count up
                // the partitions as last listed, for this profile's host
                info_.reset();
                if (std::optional<cluster::ClusterInfo> ci = link.session().clusterInfo(); ci && ci->host == host()) info_ = std::move(ci);
                fillFromCluster();
                querying_ = link.session().queryingClusterInfo();
                if (querying_) app.requestRedraw(2);
                reloadIfEditedElsewhere(link, now);
                // the summary of a session that went away: the first page again
                if (pageChosen_ && page_ == Page::Summary && !cluster::wizard::jobHeld(st) && !cluster::wizard::busy(st)) page_ = Page::Connect;
                if (!pageChosen_) {
                    page_ = forcePage_ ? *forcePage_ : cluster::wizard::openingPage(st, host());
                    pageChosen_ = true;
                    follow_ = !forcePage_;
                } else if (follow_ && cluster::wizard::busy(st) && page_ != Page::Summary) {
                    // opened while the login or the job was under way: the page of what runs now
                    page_ = cluster::wizard::openingPage(st, host());
                }
                drawHeader(st);
                widgets::vspace(2);
                widgets::rule(theme::kRule);
                // the body scrolls on a small screen; the actions stay in sight
                ImGui::SetNextWindowSizeConstraints(ImVec2(0.0f, 0.0f),
                                                    ImVec2(FLT_MAX, std::max(px(240), ImGui::GetMainViewport()->WorkSize.y * 0.86f - px(150))));
                ImGui::BeginChild("##clusterBody", ImVec2(0.0f, 0.0f), ImGuiChildFlags_AutoResizeY);
                {
                    const Spacing spacing(8, 7);
                    switch (page_) {
                        case Page::Connect: drawConnectPage(app, link, st); break;
                        case Page::Job: drawJobPage(app, link, st, now); break;
                        case Page::Worker: drawWorkerPage(app, link, st, now); break;
                        case Page::Summary: drawSummaryPage(app, link, st, now); break;
                    }
                }
                ImGui::EndChild();
                widgets::rule(theme::kRule);
                widgets::vspace(2);
                drawFooter(app, st, now);
            }

        private:
            std::string host() const { return trimmed(profile_.host); }

            // A page chosen here: from now on it stays until the next choice.
            void goTo(Page p) {
                page_ = p;
                follow_ = false;
            }

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
                lastHost_ = host();
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
                if (autoName_ && !host().empty() && name_.rfind("New cluster", 0) == 0) {
                    const std::string to = book_.uniqueName(host());
                    if (book_.rename(name_, to)) {
                        name_ = to;
                        profile_.name = to;
                    }
                    autoName_ = false;
                }
                link.saveProfiles(book_);
                savedBook_ = bookJson(book_);
            }

            // Saved, and said so for a moment.
            void saveSaying(ClusterLink& link) {
                save(link);
                savedAt_ = std::chrono::steady_clock::now();
            }

            // The settings file edited elsewhere (Edit settings file…) while
            // nothing is pending here: its profiles, once a second.
            void reloadIfEditedElsewhere(ClusterLink& link, std::chrono::steady_clock::time_point now) {
                if (now - lastReload_ < std::chrono::seconds(1)) return;
                lastReload_ = now;
                if (dirty() || renaming_) return;
                cluster::ProfileBook there = link.profiles();
                if (there.profiles.empty() || bookJson(there) == savedBook_) return;
                book_ = there;
                select(book_.find(name_) ? name_ : (book_.current.empty() ? book_.profiles.front().name : book_.current));
                savedBook_ = bookJson(book_);
            }

            void newProfile() {
                keep();
                cluster::Profile fresh;
                fresh.name = book_.uniqueName("New cluster");
                book_.profiles.push_back(fresh);
                select(fresh.name);
                autoName_ = true;
                goTo(Page::Connect);
            }

            // The ⋯ menu of page 1: the profiles and the settings file.
            void drawMoreMenu(App& app, ClusterLink& link, const cluster::Status& st) {
                const bool held = cluster::wizard::jobHeld(st) || cluster::wizard::busy(st);
                widgets::GlyphOpts o;
                o.tooltip = "Cluster profiles, import and export, the settings file";
                const ImVec2 at = ImGui::GetCursorScreenPos();
                if (widgets::glyphButton("##clusterMore", Icon::More, ImVec2(32, 32), o)) ImGui::OpenPopup("##clusterMenu");
                ImGui::SetNextWindowPos(ImVec2(at.x + px(32), at.y + px(34)), ImGuiCond_Appearing, ImVec2(1.0f, 0.0f));
                ImGui::PushStyleColor(ImGuiCol_Border, theme::kText);
                ImGui::PushStyleColor(ImGuiCol_PopupBg, theme::kBg);
                ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, theme::crispPen(2));
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, px(4, 4));
                std::string action, switchTo;
                if (ImGui::BeginPopup("##clusterMenu")) {
                    const theme::FontScope f(12);
                    const float w = px(250), h = px(24);
                    auto item = [&](const char* label, const char* key, bool enabled, const std::string& tip = {}) {
                        if (ImGui::Selectable(label, false, enabled ? ImGuiSelectableFlags_None : ImGuiSelectableFlags_Disabled, ImVec2(w, h))) action = key;
                        if (!tip.empty()) widgets::tooltip(tip);
                    };
                    widgets::caption("Profiles");
                    for (const cluster::Profile& p : book_.profiles) {
                        ImGui::PushID(p.name.c_str());
                        const std::string label = (p.name == name_ ? "\xE2\x97\x8F  " : "     ") + p.name +
                                                  (trimmed(p.host).empty() || trimmed(p.host) == p.name ? std::string() : "  (" + trimmed(p.host) + ")");
                        if (ImGui::Selectable(label.c_str(), p.name == name_, held ? ImGuiSelectableFlags_Disabled : ImGuiSelectableFlags_None, ImVec2(w, h)))
                            switchTo = p.name;
                        ImGui::PopID();
                    }
                    item("New cluster profile", "new", !held, "A profile for another cluster: its own host, job, image and folders");
                    item("Rename this profile\xE2\x80\xA6", "rename", true);
                    item("Delete this profile\xE2\x80\xA6", "delete", !held, held ? "Disconnect first: a job runs with this profile" : std::string());
                    ImGui::Separator();
                    item("Import\xE2\x80\xA6", "import", !held, "Add the profiles of a file a colleague exported (.toml)");
                    item("Export\xE2\x80\xA6", "export", true, "Save this profile as a small .toml file to share (no password, no token, no SSH client)");
                    ImGui::Separator();
                    item("Edit settings file\xE2\x80\xA6", "edit", true, settings().filePath());
                    item("Open settings folder", "folder", true, settings().directory());
                    if (st.sshUp || st.state != cluster::State::Idle) {
                        ImGui::Separator();
                        item("Disconnect\xE2\x80\xA6", "disconnect", st.sshUp || cluster::wizard::jobHeld(st), "Close the connection; you are asked whether to cancel the job");
                    }
                    ImGui::EndPopup();
                }
                ImGui::PopStyleVar(2);
                ImGui::PopStyleColor(2);
                if (!switchTo.empty() && switchTo != name_) {
                    keep();
                    select(switchTo);
                    link.saveProfiles(book_);   // the current profile
                    savedBook_ = bookJson(book_);
                }
                if (action == "new") newProfile();
                else if (action == "rename") {
                    renaming_ = true;
                    renameText_ = name_;
                } else if (action == "delete") askDelete(app, link);
                else if (action == "import") importProfiles(app);
                else if (action == "export") {
                    const cluster::Profile p = edited().currentProfile();
                    app.defer([p, &app] {
                        const std::string path = platform::saveFileDialog("Export the cluster profile", {}, p.name + ".toml", {{"Cluster profile", "toml"}});
                        if (path.empty()) return;
                        if (platform::writeFileAtomic(path, cluster::exportProfile(p))) app.wb().logLine("Cluster: exported " + p.name + " to " + path);
                        else app.message("Export the cluster profile", "Could not write " + path + ".", MessageIcon::Warning);
                    });
                } else if (action == "edit") {
                    if (dirty()) {
                        profileToUse();
                        save(link);   // what is here first: the editor shows the file
                    }
                    app.defer([&app] { app.showDialog(makeSettingsEditor(app)); });
                } else if (action == "folder") platform::openInFileManager(settings().directory());
                else if (action == "disconnect") link.disconnectAsking();
            }

            void askDelete(App& app, ClusterLink& link) {
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

            void importProfiles(App& app) {
                auto self = alive_;
                app.defer([this, self, &app] {
                    const std::string path = platform::openFileDialog("Import a cluster profile", {}, {{"Cluster profile", "toml"}, {"All files", "*"}});
                    if (!*self || path.empty()) return;
                    std::string text;
                    std::vector<cluster::Profile> found;
                    bool ignoredSsh = false;
                    try {
                        if (!platform::readFile(path, text)) throw std::runtime_error("it could not be read");
                        found = cluster::importProfiles(text, &ignoredSsh);
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
                    const std::string note = "Cluster: imported " + std::to_string(found.size()) + " profile(s) from " + path;
                    app.wb().logLine(note);
                    if (ignoredSsh)
                        app.message("Import a cluster profile",
                                    "Imported " + std::to_string(found.size()) + " profile(s). An ssh program in the file was ignored; SIRIUS uses the system SSH client.",
                                    MessageIcon::Warning);
                });
            }

            void drawRename() {
                widgets::FieldOpts fo;
                const float gap = px(8);
                const Field f("Rename the profile \"" + name_ + "\"");
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
                if (!problem.empty()) describe(problem, theme::kAccentText);
            }

            // --- the header and the footer ----------------------------------------------------

            // "1 Connect · 2 Job · 3 Worker": the current one in ink, the done ones ticked;
            // a page reached already (or whose step before is done) is a click away.
            void drawHeader(const cluster::Status& st) {
                static const char* const kNames[] = {"Connect", "Job", "Worker"};
                ImDrawList* dl = ImGui::GetWindowDrawList();
                const float d = px(20), gap = px(8), line = px(28);
                const float h = std::max(d, ImGui::GetTextLineHeight());
                for (int i = 0; i < 3; ++i) {
                    const Page pg = static_cast<Page>(i);
                    const bool current = page_ == pg;
                    const bool done = cluster::wizard::pageDone(pg, st, host());
                    const bool reachable = !current && (i == 0 || page_ == Page::Summary || static_cast<int>(page_) > i ||
                                                        cluster::wizard::pageDone(static_cast<Page>(i - 1), st, host()));
                    const std::string name = kNames[i];
                    const theme::Weight weight = current ? theme::Weight::ExtraBold : theme::Weight::SemiBold;
                    const float tw = theme::textSize(name, 13, weight).x;
                    if (i > 0) {
                        ImGui::SameLine(0.0f, gap);
                        const ImVec2 at = ImGui::GetCursorScreenPos();
                        dl->AddLine(ImVec2(at.x, at.y + h * 0.5f), ImVec2(at.x + line, at.y + h * 0.5f), theme::kNeutral400, px(1.5f));
                        ImGui::Dummy(ImVec2(line, h));
                        ImGui::SameLine(0.0f, gap);
                    }
                    ImGui::PushID(i);
                    const ImVec2 at = ImGui::GetCursorScreenPos();
                    const ImVec2 size(d + px(6) + tw, h);
                    if (ImGui::InvisibleButton("##page", size) && reachable) goTo(pg);
                    const bool hov = reachable && ImGui::IsItemHovered();
                    if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                    if (reachable) widgets::tooltip("Go to " + name);
                    const ImVec2 c(at.x + d * 0.5f, at.y + h * 0.5f);
                    if (done && !current) {
                        dl->AddCircleFilled(c, d * 0.5f, kGreen);
                        drawIcon(dl, c, d * 0.75f, Icon::Check, kWhite, px(2));
                    } else {
                        if (current) dl->AddCircleFilled(c, d * 0.5f, done ? kGreen : theme::kText);
                        else dl->AddCircle(c, d * 0.5f - px(0.75f), theme::kNeutral500, 0, px(1.5f));
                        widgets::drawTextIn(dl, ImVec2(c.x - d * 0.5f, at.y), ImVec2(c.x + d * 0.5f, at.y + h), std::to_string(i + 1), 11,
                                            current ? kWhite : theme::kNeutral600, theme::Weight::Bold, 0.5f, 0.5f);
                    }
                    const ImU32 ink = hov ? theme::kAccent : (current ? theme::kText : (done ? theme::kNeutral800 : theme::kNeutral600));
                    widgets::drawTextIn(dl, ImVec2(at.x + d + px(6), at.y), ImVec2(at.x + size.x, at.y + h), name, 13, ink, weight, 0.0f, 0.5f);
                    ImGui::PopID();
                }
                if (page_ == Page::Summary) {
                    ImGui::SameLine(0.0f, px(18));
                    const ImVec2 at = ImGui::GetCursorScreenPos();
                    const std::string s = "Summary";
                    widgets::drawTextIn(dl, at, ImVec2(at.x + theme::textSize(s, 13, theme::Weight::ExtraBold).x, at.y + h), s, 13, theme::kText,
                                        theme::Weight::ExtraBold, 0.0f, 0.5f);
                    ImGui::Dummy(ImVec2(theme::textSize(s, 13, theme::Weight::ExtraBold).x, h));
                }
            }

            void drawFooter(App& app, const cluster::Status& st, std::chrono::steady_clock::time_point now) {
                const float gap = px(8);
                bool onLine = false;
                if (page_ == Page::Job || page_ == Page::Worker) {
                    widgets::ButtonOpts b;
                    b.kind = widgets::ButtonKind::Ghost;
                    b.tooltip = "Back to " + std::string(page_ == Page::Job ? "Connect" : "Job") + " (what you entered stays)";
                    if (widgets::button("\xE2\x80\xB9 Back##clusterBack", b)) goTo(page_ == Page::Job ? Page::Connect : Page::Job);
                    onLine = true;
                }
                if (now - savedAt_ < std::chrono::seconds(4)) {
                    app.requestRedraw(2);
                    if (onLine) ImGui::SameLine(0.0f, gap);
                    onLine = true;
                    const float y = ImGui::GetCursorPosY();
                    ImGui::SetCursorPosY(y + std::max(0.0f, (ImGui::GetFrameHeight() - ImGui::GetTextLineHeight()) * 0.5f));
                    widgets::text("Saved to " + std::string("sirius-app.toml"), 11, kGreen);
                    widgets::tooltip(settings().filePath());
                    ImGui::SameLine(0.0f, gap);
                    ImGui::SetCursorPosY(y);
                    onLine = false;   // on the line already
                }
                std::string primary;
                cluster::wizard::Gate gate;
                switch (page_) {
                    case Page::Connect: primary = "Next \xE2\x80\xBA##clusterNext"; break;
                    case Page::Job: primary = "Next \xE2\x80\xBA##clusterNext"; break;
                    case Page::Worker: primary = "Finish##clusterFinish"; break;
                    case Page::Summary: primary = "Close##clusterDone"; break;
                }
                gate = cluster::wizard::nextGate(page_, st, host());
                const bool closeToo = page_ != Page::Summary;
                const float total = buttonWidth(primary, widgets::ButtonKind::Primary) +
                                    (closeToo ? gap + buttonWidth("Close", widgets::ButtonKind::Ghost) : 0.0f);
                if (onLine) ImGui::SameLine();
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - total));
                if (closeToo) {
                    widgets::ButtonOpts ghost;
                    ghost.kind = widgets::ButtonKind::Ghost;
                    ghost.tooltip = "Close this window: what runs on the cluster stays as it is";
                    if (widgets::button("Close##clusterClose", ghost) && canClose(app)) close();
                    ImGui::SameLine(0.0f, gap);
                }
                widgets::ButtonOpts b;
                b.kind = widgets::ButtonKind::Primary;
                b.enabled = gate.enabled;
                b.tooltip = gate.enabled ? (page_ == Page::Worker ? std::string("Use the cluster: the HPC backend runs there") : std::string()) : gate.why;
                if (widgets::button(primary.c_str(), b)) {
                    switch (page_) {
                        case Page::Connect: goTo(Page::Job); break;
                        case Page::Job: goTo(Page::Worker); break;
                        case Page::Worker:
                            app.wb().setBackend(Backend::Hpc);
                            if (canClose(app)) close();
                            break;
                        case Page::Summary:
                            if (canClose(app)) close();
                            break;
                    }
                }
            }

            // --- page 1: connect -----------------------------------------------------------

            void drawConnectPage(App& app, ClusterLink& link, const cluster::Status& st) {
                if (renaming_) {
                    drawRename();
                    widgets::vspace(2);
                }
                const bool loginRunning = st.state == cluster::State::Connecting &&
                                          st.steps[static_cast<std::size_t>(cluster::Step::Login)].status == cluster::StepStatus::Running;
                const bool held = cluster::wizard::jobHeld(st);
                const float gap = px(8);
                const std::string connectLabel = loginRunning ? "Stop##login" : "Connect##login";
                const float bw = buttonWidth(connectLabel, loginRunning ? widgets::ButtonKind::Secondary : widgets::ButtonKind::Primary);
                {
                    widgets::FieldOpts fo;
                    fo.enabled = !cluster::wizard::busy(st) && !held;
                    fo.width = design(std::max(px(160), ImGui::GetContentRegionAvail().x - bw - px(32) - 2 * gap));
                    fo.hint = "fiona   or   me@login.example.org";
                    const Field f("Cluster");
                    // the profiles and the hosts used before
                    std::vector<widgets::ComboItem> items;
                    std::set<std::string> seen{host()};
                    for (const cluster::Profile& p : book_.profiles)
                        if (p.name != name_ && seen.insert(p.name).second)
                            items.push_back({p.name, "profile" + (trimmed(p.host).empty() || trimmed(p.host) == p.name ? std::string() : " \xC2\xB7 " + trimmed(p.host)), false});
                    for (const auto& entry : profile_.perHost)
                        if (seen.insert(entry.first).second) items.push_back({entry.first, "used before", false});
                    std::string field = profile_.host;
                    bool picked = false;
                    widgets::editableCombo("##host", &field, items, fo, &picked);
                    hostEditing_ = ImGui::IsItemActive();
                    widgets::tooltip("The login node you submit jobs from, as you would type it after ssh. A Host of your ~/.ssh/config works "
                                     "with its user, ProxyJump and keys.");
                    const cluster::Profile* other = picked ? book_.find(trimmed(field)) : nullptr;
                    if (other && other->name != name_) {
                        keep();
                        select(other->name);
                    } else {
                        profile_.host = field;
                    }
                }
                ImGui::SameLine(0.0f, gap);
                {
                    const Field f(" ");
                    const cluster::wizard::Gate g = cluster::wizard::connectGate(st, host());
                    widgets::ButtonOpts b;
                    b.kind = loginRunning || cluster::wizard::loggedIn(st, host()) ? widgets::ButtonKind::Secondary : widgets::ButtonKind::Primary;
                    b.enabled = loginRunning || g.enabled;
                    b.tooltip = loginRunning ? std::string("Stop the login (nothing more is sent)")
                                             : (g.enabled ? (cluster::wizard::loggedIn(st, host()) ? "Log in again and list " + host() + "'s partitions anew"
                                                                                                   : "Log in to " + host() + ", then list its partitions. Nothing is submitted.")
                                                          : g.why);
                    if (widgets::button(connectLabel.c_str(), b)) {
                        if (loginRunning) link.session().cancelConnect();
                        else {
                            commit(link);
                            link.logIn(profileToUse());
                        }
                    }
                }
                ImGui::SameLine(0.0f, gap);
                {
                    const Field f(" ");
                    drawMoreMenu(app, link, st);
                }
                describe("An SSH config name such as fiona, or user@host. A password or a one-time code is asked for in a box of its own; "
                         "it is never stored.");
                widgets::text("Profile: " + name_, 11, theme::kNeutral600);
                widgets::tooltip("The settings of this cluster are kept under this name in " + settings().filePath() +
                                 ". The \xE2\x8B\xAF menu switches, adds, renames or shares profiles.");
                if (!trimmed(profile_.host).empty() && host() != lastHost_ && !hostEditing_) {
                    // another host: the partition, account, QoS, time and binds last used there
                    lastHost_ = host();
                    if (profile_.recall(lastHost_)) filled_ = {"the job and data folders last used on " + lastHost_};
                }
                widgets::vspace(6);
                drawLoginOutcome(st);
                if (held && st.host != host() && !st.host.empty()) {
                    widgets::vspace(4);
                    describe("A job is held on " + st.host + ": disconnect first (\xE2\x8B\xAF \xE2\x96\xB8 Disconnect\xE2\x80\xA6) to log in to another cluster.", kAmber);
                }
            }

            void drawLoginOutcome(const cluster::Status& st) {
                const std::string user = info_ ? info_->user : std::string();
                const cluster::wizard::LoginOutcome o = cluster::wizard::loginOutcome(st, host(), user);
                using K = cluster::wizard::LoginOutcome::Kind;
                if (o.kind == K::None) {
                    describe("Not connected yet: Connect logs in. Next is enabled once you are logged in.");
                    return;
                }
                const float lineH = theme::textSize(std::string("Xg"), 13, theme::Weight::SemiBold).y;
                widgets::beginCard("##loginOutcome", false, 10);
                mark(o.kind == K::Ok ? Ink::Ok : (o.kind == K::Busy ? Ink::Busy : Ink::Fail), 16, lineH);
                ImGui::SameLine(0.0f, px(8));
                ImGui::BeginGroup();
                const ImU32 ink = o.kind == K::Ok ? kGreen : (o.kind == K::Failed ? theme::kAccentText : theme::kText);
                // a failure is copied into a report: with its details, when there are some
                if (o.kind == K::Failed)
                    widgets::copyableText("##loginFailure", o.text, 13, ink, theme::Weight::SemiBold, ImGui::GetContentRegionAvail().x,
                                          o.details.empty() || o.details == o.text ? std::string() : o.text + "\n\n" + o.details);
                else
                    widgets::textWrapped(o.text, 13, ink, theme::Weight::SemiBold, ImGui::GetContentRegionAvail().x);
                if (o.kind == K::Ok) {
                    if (info_) {
                        const std::string what = info_->error.empty() ? std::to_string(info_->partitions.size()) + " partitions found" + (info_->home.empty() ? std::string() : " \xC2\xB7 home " + info_->home)
                                                                      : info_->error;
                        describe(what + ". Next: the job.");
                    } else if (querying_) {
                        describe("Listing the partitions\xE2\x80\xA6");
                    }
                } else if (o.kind == K::Busy) {
                    describe("Answer the password or code prompt when it appears.");
                } else if (!o.details.empty() && o.details != o.text) {
                    disclosure("##loginDetails", "Details", &detailsOpen_, 11);
                    if (detailsOpen_) outputBlock("##loginOutput", o.details, 70);
                }
                ImGui::EndGroup();
                widgets::endCard();
            }

            // --- page 2: the job ----------------------------------------------------------------

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

            static std::string timeWords(const std::string& t) {
                const long long s = cluster::slurmTimeSeconds(t);
                return s < 0 ? std::string() : cluster::durationText(s);
            }

            // The partitions: the profile's choices, then what the cluster reports.
            std::vector<widgets::ComboItem> partitionItems() const {
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
                            items.push_back({p.name, (usable ? std::string() : std::string("no association \xC2\xB7 ")) + cluster::partitionSummary(p), false});
                        }
                return items;
            }

            std::vector<widgets::ComboItem> accountItems() const {
                std::vector<widgets::ComboItem> items;
                std::set<std::string> seen;
                if (const cluster::PartitionChoice* c = profile_.choice(trimmed(profile_.partition)))
                    for (const std::string& a : c->accounts)
                        if (seen.insert(a).second) items.push_back({a, "your settings", false});
                if (info_)
                    for (const std::string& a : cluster::accountsFor(*info_, trimmed(profile_.partition)))
                        if (seen.insert(a).second) items.push_back({a, "from the cluster", false});
                return items;
            }

            std::vector<widgets::ComboItem> qosItems() const {
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
                return items;
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
            void fillFromCluster() {
                if (!info_ || filledFor_ == info_->host) return;
                filledFor_ = info_->host;
                profile_.gpus = static_cast<int>(gpus_);
                profile_.cpus = static_cast<int>(cpus_);
                const std::vector<std::string> filled = cluster::fillFromCluster(profile_, *info_);
                gpus_ = profile_.gpus;
                cpus_ = profile_.cpus;
                if (!filled.empty()) filled_ = filled;
            }

            void drawJobPage(App& app, ClusterLink& link, const cluster::Status& st, std::chrono::steady_clock::time_point now) {
                const bool held = cluster::wizard::jobHeld(st);
                const bool busy = cluster::wizard::busy(st);
                const bool editable = !held && !busy;
                const float gap = px(10);
                if (!cluster::wizard::loggedIn(st, host()) && !held) describe("Not logged in: Connect on page 1 first. The lists below fill from the cluster then.", kAmber);
                // a job of the user's own, in place of a new one
                drawYourJobs(link, st);
                widgets::FieldOpts fo;
                fo.enabled = editable;
                // the node type
                {
                    const float w = ImGui::GetContentRegionAvail().x;
                    fo.width = design(w);
                    const Field f("Node type (partition)");
                    if (openList_ && info_) {
                        // a screenshot: the list as the user sees it on a click
                        ImGui::OpenPopupEx(ImHashStr("##ComboPopup", 0, ImGui::GetID("##partition")));
                        openList_ = false;
                    }
                    if (choice("##partition", &profile_.partition, partitionItems(), fo, 0.0f, "(the cluster's default)")) choosePartition();
                    widgets::tooltip("Where the job runs: each line says how many nodes it has, how many are idle, their GPUs and the longest "
                                     "time. Picking one fills the account, QoS and limits that go with it.");
                }
                const cluster::PartitionChoice* mine = profile_.choice(trimmed(profile_.partition));
                const cluster::Partition* part = info_ ? cluster::findPartition(*info_, trimmed(profile_.partition)) : nullptr;
                drawPartitionLine(part, editable);
                // account and QoS: only where there is a choice
                const std::vector<widgets::ComboItem> accounts = accountItems();
                const std::vector<widgets::ComboItem> qos = qosItems();
                const bool pickAccount = accounts.size() > 1 || (accounts.size() == 1 && trimmed(profile_.account) != accounts.front().value);
                const bool pickQos = qos.size() > 1 || (qos.size() == 1 && trimmed(profile_.qos) != qos.front().value);
                if (pickAccount || pickQos) {
                    const int n = (pickAccount ? 1 : 0) + (pickQos ? 1 : 0);
                    fo.width = design(columnWidth(n, 10));
                    if (pickAccount) {
                        const Field f("Account");
                        choice("##account", &profile_.account, accounts, fo, 0.0f, "(your default)");
                        widgets::tooltip("The account the job is charged to.");
                    }
                    if (pickQos) {
                        if (pickAccount) ImGui::SameLine(0.0f, gap);
                        const Field f("QoS");
                        choice("##qos", &profile_.qos, qos, fo, 0.0f, "(the default QoS)");
                        widgets::tooltip("The quality of service: it sets the job's priority and longest time.");
                    }
                }
                // the size and the time
                const std::int64_t maxGpus = mine && mine->maxGpus >= 0 ? mine->maxGpus : (part && part->gpusPerNode > 0 ? part->gpusPerNode : 16);
                const std::int64_t maxCpus = mine && mine->maxCpus > 0 ? mine->maxCpus : (part && part->cpusPerNode > 0 ? part->cpusPerNode : 256);
                if (editable) {
                    gpus_ = std::min(gpus_, maxGpus);
                    cpus_ = std::min(cpus_, maxCpus);
                }
                std::string mostMem = mine && !mine->maxMem.empty() ? mine->maxMem : std::string();
                if (mostMem.empty() && part && part->memPerNodeMB > 0) mostMem = std::to_string(part->memPerNodeMB / 1024) + "G";
                std::string limit = mine && !mine->maxTime.empty() ? mine->maxTime : (part ? part->maxTime : std::string());
                if (info_)
                    if (const auto it = info_->qosMaxWall.find(trimmed(profile_.qos)); it != info_->qosMaxWall.end() && !it->second.empty()) {
                        const long long q = cluster::slurmTimeSeconds(it->second), l = cluster::slurmTimeSeconds(limit);
                        if (q >= 0 && (l < 0 || q < l)) limit = it->second;
                    }
                {
                    fo.width = design(columnWidth(4, 10));
                    {
                        const Field f("GPUs");
                        spinInt("##gpus", &gpus_, 0, maxGpus, 1, fo);
                        widgets::tooltip("GPUs per job, at most " + std::to_string(maxGpus) + " here (0: the steps run on the CPU).");
                    }
                    ImGui::SameLine(0.0f, gap);
                    {
                        const Field f("CPUs");
                        spinInt("##cpus", &cpus_, 1, maxCpus, 1, fo);
                        widgets::tooltip("CPU cores, at most " + std::to_string(maxCpus) + " here.");
                    }
                    ImGui::SameLine(0.0f, gap);
                    {
                        const Field f("Memory");
                        std::vector<widgets::ComboItem> items;
                        const long long most = mostMem.empty() ? -1 : cluster::memoryMB(mostMem);
                        for (const char* m : {"8G", "16G", "32G", "64G", "128G", "256G", "512G"})
                            if (most < 0 || cluster::memoryMB(m) <= most) items.push_back({m, {}, false});
                        if (!mostMem.empty()) items.push_back({mostMem, "a whole node's", false});
                        bool picked = false;
                        widgets::editableCombo("##mem", &profile_.mem, items, fo, &picked);
                        widgets::tooltip("Memory for the job: \"64G\", \"500M\"." + (mostMem.empty() ? std::string() : " A node here has " + mostMem + "."));
                    }
                    ImGui::SameLine(0.0f, gap);
                    {
                        const Field f("Time");
                        std::vector<widgets::ComboItem> items;
                        for (const std::string& t : cluster::timeChoices(mine, limit, profile_.time)) items.push_back({t, timeWords(t), false});
                        choice("##time", &profile_.time, items, fo, 200.0f, "(Slurm's default)");
                        widgets::tooltip("How long the job may hold its node: it ends then, and the worker with it. The most allowed here: " +
                                         (limit.empty() ? std::string("not known") : limit) + ".");
                    }
                }
                if (!mostMem.empty() && cluster::memoryMB(profile_.mem) > cluster::memoryMB(mostMem))
                    describe("More memory than a node here has (" + mostMem + "): the job would never start.", theme::kAccentText);
                if (!filled_.empty() && editable) {
                    std::string line;
                    for (const std::string& s : filled_) line += (line.empty() ? "" : ", ") + s;
                    describe("Filled in: " + line + ".");
                }
                // the image and the data folders
                widgets::vspace(2);
                drawImageAndFolders(app, st);
                // start the job
                widgets::vspace(4);
                drawJobStatus(link, st, now);
                // the rest, folded
                widgets::vspace(2);
                if (disclosure("##more", "More options\xE2\x80\xA6", &moreOpen_)) drawMoreOptions(app, link, st);
            }

            // The user's jobs that run or wait on the cluster (any of them),
            // each with "Use this job": SIRIUS's worker then runs in it as a
            // step, and SIRIUS never cancels it. A job taken up so says so.
            void drawYourJobs(ClusterLink& link, const cluster::Status& st) {
                const bool held = cluster::wizard::jobHeld(st);
                if (held && st.adopted) {
                    describe("Job " + st.jobId + " was yours before SIRIUS: SIRIUS's worker runs in it as a step, and SIRIUS never cancels it "
                                                 "(Disconnect leaves it running).");
                    return;
                }
                if (held || !cluster::wizard::loggedIn(st, host())) return;
                ClusterLink::UserJobs jobs = link.userJobs();
                if ((!jobs.known && !jobs.loading) || jobs.host != st.host) {
                    link.refreshJobs();
                    jobs = link.userJobs();
                }
                if (!disclosure("##yourJobs", "Your jobs on " + st.host + (jobs.known ? " (" + std::to_string(jobs.jobs.size()) + ")" : std::string()), &yourJobsOpen_))
                    return;
                const bool busy = cluster::wizard::busy(st);
                {
                    widgets::ButtonOpts r;
                    r.small = true;
                    r.kind = widgets::ButtonKind::Ghost;
                    r.enabled = !jobs.loading;
                    r.tooltip = "Ask the cluster again (squeue)";
                    if (widgets::button(jobs.loading ? "Asking\xE2\x80\xA6##jobsRefresh" : "Refresh##jobsRefresh", r)) link.refreshJobs();
                }
                if (!jobs.error.empty()) describe("Your jobs could not be listed: " + jobs.error, theme::kAccentText);
                else if (jobs.known && jobs.jobs.empty())
                    describe("None runs or waits: Start job below asks for one.");
                for (const cluster::ClusterJob& j : jobs.jobs) {
                    ImGui::PushID(j.id.c_str());
                    widgets::ButtonOpts b;
                    b.small = true;
                    b.enabled = !busy;
                    b.tooltip = "Take up job " + j.id + " as it is, in place of a new job: SIRIUS's worker runs in it as a step (srun --overlap) "
                                                        "with its " +
                                (j.gpus > 0 ? std::to_string(j.gpus) + (j.gpus == 1 ? " GPU" : " GPUs") : std::string("CPUs")) +
                                ", after the checks on its node. SIRIUS never cancels it.";
                    if (widgets::button("Use this job", b)) {
                        profileToUse();
                        saveSaying(link);
                        link.adoptJob(profileToUse(), j.id);
                    }
                    ImGui::SameLine(0.0f, px(8));
                    widgets::textWrapped(j.id + " \xC2\xB7 " + j.name + " \xC2\xB7 " + j.partition + " \xC2\xB7 " + cluster::jobSummary(j), 12,
                                         j.running() ? theme::kText : theme::kNeutral700, theme::Weight::Regular, ImGui::GetContentRegionAvail().x);
                    ImGui::PopID();
                }
                widgets::vspace(4);
            }

            // Under the node type: what it has, and what to know before submitting there.
            void drawPartitionLine(const cluster::Partition* part, bool editable) {
                if (part) {
                    std::string line = cluster::partitionSummary(*part, true);
                    std::string acct;
                    if (!trimmed(profile_.account).empty()) acct += "account " + trimmed(profile_.account);
                    if (!trimmed(profile_.qos).empty()) acct += (acct.empty() ? "" : " \xC2\xB7 ") + std::string("QoS ") + trimmed(profile_.qos);
                    describe(line + (acct.empty() ? std::string() : " \xC2\xB7 " + acct));
                    if (info_ && !cluster::hasAssociation(*info_, part->name))
                        describe("You have no association for " + part->name + ": sbatch will most likely refuse the job. Pick another node type.",
                                 theme::kAccentText);
                    if (const std::string warning = cluster::partitionWarning(*part); !warning.empty()) describe(warning, kAmber);
                    if (editable && info_ && !profile_.choice(part->name)) {
                        if (widgets::linkButton(("Add " + part->name + " to my settings##addChoice").c_str()))
                            profile_.choices.push_back(cluster::choiceFromCluster(*info_, part->name));
                        widgets::tooltip("Keep " + part->name + " in this profile's list of node types, with its accounts, QoS and limits as the "
                                                                "cluster reports them.");
                    }
                } else if (info_ && !info_->error.empty()) {
                    describe(info_->error + ".", theme::kAccentText);
                } else if (!info_) {
                    describe(querying_ ? std::string("Listing the cluster's partitions\xE2\x80\xA6") : std::string("The list fills from the cluster once you are logged in."));
                }
            }

            void drawImageAndFolders(App& app, const cluster::Status& st) {
                widgets::FieldOpts fo;
                fo.enabled = st.state != cluster::State::Starting;
                const float gap = px(10);
                const bool here = cluster::wizard::loggedIn(st, host());
                {
                    const float browse = buttonWidth("Browse\xE2\x80\xA6", widgets::ButtonKind::Secondary);
                    fo.width = design(std::max(px(160), ImGui::GetContentRegionAvail().x - browse - gap));
                    {
                        const Field f("Worker image (.sif)");
                        fo.hint = "/path/on/the/cluster/sirius-worker.sif   (required)";
                        std::vector<widgets::ComboItem> items;
                        for (const std::string& i : profile_.images) items.push_back({i, "used before", false});
                        bool picked = false;
                        widgets::editableCombo("##image", &profile_.container, items, fo, &picked);
                        fo.hint.clear();
                        widgets::tooltip("An Apptainer/Singularity image on the cluster with SIRIUS's worker environment (the compiled sirius "
                                         "package, numpy, torch). The worker and the engine run in it.");
                    }
                    ImGui::SameLine(0.0f, gap);
                    const Field f(" ");
                    widgets::ButtonOpts b;
                    b.enabled = fo.enabled && here;
                    b.tooltip = here ? std::string("Pick the image among the cluster's .sif files") : std::string("Log in first: then the cluster's files are listed");
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
                    describe("Required. None yet? More options \xE2\x96\xB8 Build an image makes one in the job.", kAmber);
                {
                    const float add = buttonWidth("Add folder\xE2\x80\xA6", widgets::ButtonKind::Secondary);
                    fo.width = design(std::max(px(160), ImGui::GetContentRegionAvail().x - add - gap));
                    {
                        const Field f("Data folders");
                        fo.hint = "/data/lab, /scratch/me   (optional: your home folder is always there)";
                        std::vector<widgets::ComboItem> items;
                        for (const std::string& set : profile_.bindSets) items.push_back({set, "used before", false});
                        bool picked = false;
                        widgets::editableCombo("##binds", &profile_.bind, items, fo, &picked);
                        fo.hint.clear();
                        widgets::tooltip("Folders on the cluster your datasets live in, made visible inside the image (comma separated). Each is "
                                         "checked before the worker starts.");
                    }
                    ImGui::SameLine(0.0f, gap);
                    const Field f(" ");
                    widgets::ButtonOpts b;
                    b.enabled = fo.enabled && here;
                    b.tooltip = here ? std::string("Pick a folder among the cluster's files and add it") : std::string("Log in first: then the cluster's files are listed");
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
            }

            // Start job / Stop / Change job…, and the job's live line.
            void drawJobStatus(ClusterLink& link, const cluster::Status& st, std::chrono::steady_clock::time_point now) {
                const bool held = cluster::wizard::jobHeld(st);
                const bool connecting = st.state == cluster::State::Connecting;
                const cluster::wizard::JobLine line = cluster::wizard::jobLine(st, now);
                widgets::beginCard("##jobStatus", false, 10);
                std::string label = "Start job##job";
                widgets::ButtonOpts b;
                b.kind = widgets::ButtonKind::Primary;
                if (connecting) {
                    label = "Stop##job";
                    b.kind = widgets::ButtonKind::Secondary;
                    b.tooltip = "Stop waiting (a job already submitted is taken up again by Start job)";
                } else if (held && st.adopted) {
                    label = "Let go of this job\xE2\x80\xA6##job";
                    b.kind = widgets::ButtonKind::Secondary;
                    b.tooltip = "Job " + st.jobId + " was yours before SIRIUS: it is never cancelled. Let go of it (it keeps running) to use another; you stay logged in";
                } else if (held) {
                    label = "Change job\xE2\x80\xA6##job";
                    b.kind = widgets::ButtonKind::Secondary;
                    b.tooltip = "Cancel job " + st.jobId + " (you are asked first) to ask for one with other settings; you stay logged in";
                } else {
                    const cluster::wizard::Gate g = cluster::wizard::startJobGate(st, host());
                    b.enabled = g.enabled;
                    b.tooltip = g.enabled ? "Save these choices and ask Slurm for a job that holds a node for SIRIUS" : g.why;
                }
                if (widgets::button(label.c_str(), b)) {
                    if (connecting) link.session().cancelConnect();
                    else if (held) link.changeJobAsking();
                    else {
                        profileToUse();
                        saveSaying(link);
                        link.connectJob(profileToUse());
                    }
                }
                ImGui::SameLine(0.0f, px(10));
                ImGui::BeginGroup();
                const float lineH = ImGui::GetFrameHeight();
                Ink ink = Ink::Pending;
                ImU32 color = theme::kNeutral600;
                switch (line.kind) {
                    case cluster::wizard::JobLine::Kind::None: break;
                    case cluster::wizard::JobLine::Kind::Busy:
                        ink = Ink::Busy;
                        color = theme::kText;
                        break;
                    case cluster::wizard::JobLine::Kind::Running:
                        ink = Ink::Ok;
                        color = kGreen;
                        break;
                    case cluster::wizard::JobLine::Kind::Failed:
                        ink = Ink::Fail;
                        color = theme::kAccentText;
                        break;
                }
                mark(ink, 14, lineH);
                ImGui::SameLine(0.0f, px(6));
                const float y = ImGui::GetCursorPosY();
                ImGui::SetCursorPosY(y + std::max(0.0f, (lineH - ImGui::GetTextLineHeight()) * 0.5f));
                if (line.kind == cluster::wizard::JobLine::Kind::Failed)
                    widgets::copyableText("##jobFailure", line.text, 12, color, theme::Weight::SemiBold, ImGui::GetContentRegionAvail().x,
                                          line.text + (st.fix.empty() ? std::string() : "\n" + st.fix) +
                                              (st.remoteOutput.empty() ? std::string() : "\n\n" + st.remoteOutput));
                else
                    widgets::textWrapped(line.text, 12, color, line.kind == cluster::wizard::JobLine::Kind::None ? theme::Weight::Regular : theme::Weight::SemiBold,
                                         ImGui::GetContentRegionAvail().x);
                ImGui::EndGroup();
                if (line.kind == cluster::wizard::JobLine::Kind::Failed) {
                    if (!st.fix.empty()) describe(st.fix, theme::kText);
                    if (!st.remoteOutput.empty()) {
                        disclosure("##jobDetails", "Details", &detailsOpen_, 11);
                        if (detailsOpen_) outputBlock("##jobOutput", st.remoteOutput, 70);
                    }
                }
                widgets::endCard();
            }

            void drawMoreOptions(App& app, ClusterLink& link, const cluster::Status& st) {
                widgets::FieldOpts fo;
                fo.enabled = !cluster::wizard::busy(st);
                const float gap = px(10);
                {
                    fo.width = design(columnWidth(2, 10));
                    {
                        const Field f("SIRIUS checkout on the cluster");
                        fo.hint = "filled in at the login: <home>/sirius";
                        widgets::inputText("##checkout", &profile_.checkout, fo);
                        fo.hint.clear();
                        widgets::tooltip("A clone of this SIRIUS repository on the cluster: the worker's Python code (app/python) comes from there.");
                    }
                    ImGui::SameLine(0.0f, gap);
                    const Field f("Container launcher");
                    fo.hint = "apptainer";
                    widgets::inputText("##launcher", &profile_.launcher, fo);
                    fo.hint.clear();
                    widgets::tooltip("apptainer, singularity, or a full path. When it is not on the PATH, `module load` of it is tried.");
                }
                {
                    fo.width = design(columnWidth(2, 10));
                    {
                        const Field f("Extra Python path");
                        fo.hint = "usually empty";
                        widgets::inputText("##pythonPath", &profile_.containerPythonPath, fo);
                        fo.hint.clear();
                        widgets::tooltip("Entries added to PYTHONPATH inside the image (: separated), after the worker's own code.");
                    }
                    ImGui::SameLine(0.0f, gap);
                    const Field f("Node cache folder");
                    fo.hint = "empty: the node's temporary folder";
                    widgets::FieldOpts co = fo;
                    co.enabled = fo.enabled && profile_.engine;
                    widgets::inputText("##cache", &profile_.scratch, co);
                    fo.hint.clear();
                    widgets::tooltip("A folder on the node with room for the engine's cache and uploads: a node's local disk is fastest.");
                }
                widgets::checkbox("Run SIRIUS's C++ engine on the node##engine", &profile_.engine, fo.enabled);
                widgets::tooltip("On: every step of a pipeline runs on the node with SIRIUS's own engine, and the results stay there. Off: only "
                                 "the Python steps run there.");
                if (profile_.engine) {
                    fo.width = design(columnWidth(2, 10));
                    {
                        const Field f("Engine builds folder");
                        fo.hint = "empty: the image's own engine";
                        widgets::inputText("##engineBuilds", &profile_.engineBuilds, fo);
                        fo.hint.clear();
                        widgets::tooltip("A folder with one engine build per SIRIUS commit (<commit>/bin/sirius-cli and BUILD.json): the build of "
                                         "this application, or one with the same operations, is bound into the image.");
                    }
                    ImGui::SameLine(0.0f, gap);
                    const Field f("Engine executable");
                    fo.hint = "empty: picked from the builds";
                    widgets::inputText("##engineBin", &profile_.engineBin, fo);
                    fo.hint.clear();
                    widgets::tooltip("Names a sirius-cli outright, overriding the builds folder. Usually empty.");
                }
                widgets::vspace(2);
                if (disclosure("##build", "Build an image", &buildOpen_, 11)) drawBuild(app, link, st);
            }

            // Build an image in the held job, when the cluster lets the user.
            void drawBuild(App& app, ClusterLink& link, const cluster::Status& st) {
                const cluster::BuildStatus& b = st.build;
                const bool running = b.phase == cluster::BuildStatus::Phase::Probing || b.phase == cluster::BuildStatus::Phase::Building;
                const bool held = (st.state == cluster::State::JobReady || st.state == cluster::State::Connected);
                describe("apptainer build --fakeroot inside the job, from the definition file in the SIRIUS checkout. Takes a while; only where "
                         "the cluster allows unprivileged builds (checked first).");
                if (!held && !running) {
                    describe("Start the job first: the build runs in it.");
                    return;
                }
                if (b.supported && !*b.supported) {
                    describe("This cluster does not let you build images: " + b.why, theme::kAccentText);
                    return;
                }
                widgets::FieldOpts fo;
                fo.enabled = !running;
                fo.width = design(columnWidth(2, 10));
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
                if (!state.empty()) describe(state, b.phase == cluster::BuildStatus::Phase::Failed ? theme::kAccentText : theme::kNeutral700);
                if (!b.log.empty()) outputBlock("##buildLog", b.log, 80);
                // the result into the profile, once
                if (b.phase == cluster::BuildStatus::Phase::Done && b.image != adopted_) {
                    adopted_ = b.image;
                    profile_.container = b.image;
                    profileToUse();
                    saveSaying(link);
                    app.wb().logLine("Cluster: the new image " + b.image + " is this profile's worker image now.");
                }
            }

            // --- page 3: the worker -------------------------------------------------------------

            void drawWorkerPage(App& app, ClusterLink& link, const cluster::Status& st, std::chrono::steady_clock::time_point now) {
                (void)app;
                const bool connected = st.state == cluster::State::Connected;
                const bool starting = st.state == cluster::State::Starting;
                const cluster::wizard::Gate g = cluster::wizard::startWorkerGate(st, trimmed(profile_.container));
                // what is started, and where
                std::string where = cluster::wizard::jobHeld(st) ? "job " + st.jobId + " on " + st.node : std::string("the job (start it on page 2)");
                describe("Runs " + (trimmed(profile_.container).empty() ? std::string("the worker image") : trimmed(profile_.container)) + " in " + where +
                         (profile_.engine ? ", with SIRIUS's C++ engine." : ", the Python worker only."));
                // what changed since it started
                std::string changed;
                if (connected) {
                    cluster::Profile edited = profile_;
                    edited.gpus = static_cast<int>(gpus_);
                    edited.cpus = static_cast<int>(cpus_);
                    const cluster::ProfileChange c = cluster::profileChange(link.session().profile(), edited);
                    if (c.newWorker && !c.newJob)
                        for (const std::string& f : c.fields) changed += (changed.empty() ? "" : ", ") + f;
                }
                {
                    widgets::ButtonOpts b;
                    std::string label = connected ? "Restart worker##worker" : "Start worker##worker";
                    b.kind = connected && changed.empty() ? widgets::ButtonKind::Secondary : widgets::ButtonKind::Primary;
                    b.enabled = starting || g.enabled;
                    b.tooltip = g.enabled ? (connected ? std::string("Stop the worker and start it again (what it holds is lost)")
                                                       : std::string("Save these choices and start SIRIUS's worker in the job (srun --overlap)"))
                                          : g.why;
                    if (starting) {
                        label = "Stop##worker";
                        b.kind = widgets::ButtonKind::Secondary;
                        b.tooltip = "Stop starting the worker; the job stays";
                    }
                    if (widgets::button(label.c_str(), b)) {
                        if (starting) link.session().cancelConnect();
                        else {
                            profileToUse();
                            saveSaying(link);
                            link.startWorker(profileToUse());
                        }
                    }
                    if (connected) {
                        ImGui::SameLine(0.0f, px(8));
                        widgets::ButtonOpts s;
                        s.kind = widgets::ButtonKind::Ghost;
                        s.tooltip = "End the worker; the job stays held";
                        if (widgets::button("Stop worker##stopWorker", s)) link.stopWorker();
                    }
                }
                if (!changed.empty()) describe("Changed since it started: " + changed + ". Restart worker uses them.", kAmber);
                widgets::vspace(4);
                drawHealth(cluster::wizard::healthReport(st, connected ? link.session().profile() : profile_, buildInfo(), now));
            }

            void drawHealth(const cluster::wizard::HealthReport& r) {
                using V = cluster::wizard::HealthReport::Verdict;
                widgets::beginCard("##health", false, 10);
                {
                    const float lineH = theme::textSize(std::string("Xg"), 14, theme::Weight::Bold).y;
                    Ink ink = Ink::Pending;
                    ImU32 color = theme::kNeutral700;
                    switch (r.verdict) {
                        case V::None: break;
                        case V::Busy:
                            ink = Ink::Busy;
                            color = theme::kText;
                            break;
                        case V::Ready:
                            ink = Ink::Ok;
                            color = kGreen;
                            break;
                        case V::Failed:
                            ink = Ink::Fail;
                            color = theme::kAccentText;
                            break;
                    }
                    mark(ink, 18, lineH);
                    ImGui::SameLine(0.0f, px(8));
                    if (r.verdict == V::Failed) {
                        // the whole report goes to the clipboard: headline, what to do, the rows, the details
                        std::string report = r.headline;
                        if (!r.fix.empty()) report += "\n" + r.fix;
                        for (const cluster::wizard::HealthRow& row : r.rows) report += "\n" + row.label + ": " + row.value;
                        if (!r.details.empty()) report += "\n\n" + r.details;
                        widgets::copyableText("##healthFailure", r.headline, 14, color, theme::Weight::Bold, ImGui::GetContentRegionAvail().x, report);
                    } else {
                        widgets::textWrapped(r.headline, 14, color, theme::Weight::Bold, ImGui::GetContentRegionAvail().x);
                    }
                }
                if (r.verdict == V::Failed && !r.fix.empty()) {
                    widgets::text("What to do", 11, theme::kText, theme::Weight::SemiBold);
                    describe(r.fix, theme::kText);
                }
                if (r.verdict == V::Failed && !r.details.empty()) {
                    disclosure("##workerDetails", "Details", &detailsOpen_, 11);
                    if (detailsOpen_) outputBlock("##workerOutput", r.details, 80);
                }
                widgets::vspace(2);
                healthTable("##healthRows", r.rows);
                widgets::endCard();
            }

            // Rows of a mark, a label and a value (wrapped).
            static void healthTable(const char* id, const std::vector<cluster::wizard::HealthRow>& rows) {
                if (rows.empty()) return;
                const ImGuiTableFlags flags = ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoPadOuterX;
                ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, px(4, 2));
                if (ImGui::BeginTable(id, 3, flags)) {
                    ImGui::TableSetupColumn("mark", ImGuiTableColumnFlags_WidthFixed, px(16));
                    ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed, px(104));
                    ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
                    for (std::size_t i = 0; i < rows.size(); ++i) {
                        const cluster::wizard::HealthRow& row = rows[i];
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        mark(inkOf(row.mark), 12);
                        ImGui::TableSetColumnIndex(1);
                        widgets::text(row.label, 12, theme::kText, theme::Weight::SemiBold);
                        ImGui::TableSetColumnIndex(2);
                        const ImU32 color = row.mark == cluster::wizard::Mark::Fail ? theme::kAccentText
                                                                                    : (row.mark == cluster::wizard::Mark::Warn ? kAmber : theme::kNeutral800);
                        widgets::textWrapped(row.value, 12, color, theme::Weight::Regular, ImGui::GetContentRegionAvail().x);
                        ImGui::PushID(static_cast<int>(i));
                        widgets::copyOnRightClick("##rowCopy", row.label + ": " + row.value);
                        ImGui::PopID();
                    }
                    ImGui::EndTable();
                }
                ImGui::PopStyleVar();
            }

            // --- the summary -------------------------------------------------------------------

            void drawSummaryPage(App& app, ClusterLink& link, const cluster::Status& st, std::chrono::steady_clock::time_point now) {
                (void)app;
                const cluster::Profile running = link.session().profile();
                const cluster::wizard::HealthReport r = cluster::wizard::healthReport(st, running, buildInfo(), now);
                const bool connected = st.state == cluster::State::Connected;
                {
                    const float lineH = theme::textSize(std::string("Xg"), 14, theme::Weight::Bold).y;
                    mark(connected ? Ink::Ok : (cluster::wizard::busy(st) ? Ink::Busy : Ink::Fail), 18, lineH);
                    ImGui::SameLine(0.0f, px(8));
                    const std::string head = connected ? "The cluster is set up: the HPC backend runs on " + cluster::shortNodeName(st.node)
                                                       : (cluster::wizard::jobHeld(st) ? "The job runs, but no worker answers" : "Not connected");
                    widgets::textWrapped(head, 14, connected ? kGreen : theme::kAccentText, theme::Weight::Bold, ImGui::GetContentRegionAvail().x);
                }
                std::vector<cluster::wizard::HealthRow> rows;
                using M = cluster::wizard::Mark;
                rows.push_back({"Cluster", st.host + (info_ && !info_->user.empty() ? " as " + info_->user : std::string()), M::Ok});
                if (!st.jobId.empty())
                    rows.push_back({"Job", st.jobId + (running.partition.empty() ? std::string() : " \xC2\xB7 " + running.partition) + " \xC2\xB7 " + std::to_string(running.gpus) + " GPU \xC2\xB7 " + std::to_string(running.cpus) + " CPUs \xC2\xB7 " + running.mem,
                                    M::Ok});
                if (!st.node.empty()) rows.push_back({"Node", st.node, M::Ok});
                if (connected) {
                    const std::string gpus = cluster::gpuSummary(st.caps.gpus);
                    rows.push_back({"Device", cluster::gpuUsable(st.caps) ? (gpus.empty() ? st.caps.device : gpus) : st.caps.device + (gpus.empty() ? std::string() : " (" + gpus + " not usable)"),
                                    cluster::gpuUsable(st.caps) || running.gpus <= 0 ? M::Ok : M::Warn});
                }
                rows.push_back({"Worker", r.headline, r.verdict == cluster::wizard::HealthReport::Verdict::Ready ? (r.headline == "Ready" ? M::Ok : M::Warn) : (r.verdict == cluster::wizard::HealthReport::Verdict::Failed ? M::Fail : M::Info)});
                if (const std::string left = cluster::wizard::timeLeftText(st, now); !left.empty()) rows.push_back({"Time left", left, M::Info});
                widgets::vspace(2);
                healthTable("##summaryRows", rows);
                widgets::vspace(6);
                const float gap = px(8);
                widgets::ButtonOpts b;
                b.small = true;
                b.tooltip = "Close the connection; you are asked whether to cancel the job";
                if (widgets::button("Disconnect\xE2\x80\xA6", b)) link.disconnectAsking();
                ImGui::SameLine(0.0f, gap);
                b.tooltip = "The job's settings (page 2): Change job… there cancels this one first";
                if (widgets::button("Change job", b)) goTo(Page::Job);
                ImGui::SameLine(0.0f, gap);
                b.tooltip = "The worker page: restart the worker, see its full report";
                b.enabled = cluster::wizard::jobHeld(st);
                if (widgets::button("Restart worker", b)) {
                    goTo(Page::Worker);
                    if (connected) {
                        profileToUse();
                        saveSaying(link);
                        link.startWorker(profileToUse());
                    }
                }
                ImGui::SameLine(0.0f, gap);
                b.enabled = true;
                b.tooltip = "Page 1: another cluster or profile";
                if (widgets::button("Another cluster\xE2\x80\xA6", b)) goTo(Page::Connect);
            }

            // --- what the fields hold ---------------------------------------------------------------

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

            // The fields saved before the session gets them (what it uses is what the file has).
            void commit(ClusterLink& link) {
                profileToUse();
                save(link);
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
            Page page_ = Page::Connect;
            bool follow_ = true;   // the page follows what the session does, until one is chosen here
            bool pageChosen_ = false;
            std::optional<Page> forcePage_;              // screenshots
            bool openList_ = false;
            bool moreOpen_ = false;
            bool yourJobsOpen_ = true;   // the user's own jobs on the Job page
            bool buildOpen_ = false;
            bool detailsOpen_ = false;
            bool querying_ = false;                      // the partitions being listed
            std::chrono::steady_clock::time_point savedAt_{};
            std::chrono::steady_clock::time_point lastReload_{};
            std::string defFile_, imageOut_, adopted_;
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
