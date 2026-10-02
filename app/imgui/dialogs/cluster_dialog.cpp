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

#include "core/host.hpp"
#include "core/remote_source.hpp"
#include "core/secure_wipe.hpp"
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

        // A Refresh of the partitions on the dialog's thread.
        struct RefreshShared {
            std::mutex m;
            bool running = false;
            std::string error;
            bool gone = false;
        };

        class ClusterDialog final : public Dialog {
        public:
            explicit ClusterDialog(App& app) : profile_(app.cluster().storedProfile()) {
                gpus_ = profile_.gpus;
                cpus_ = profile_.cpus;
                lastHost_ = trimmed(profile_.host);
                // Screenshots only: the partition list open once it is in.
                openList_ = app.unattended() && !host::environment("SIRIUS_TEST_OPEN_PARTITIONS").empty();
            }
            ~ClusterDialog() override {
                *alive_ = false;
                const std::lock_guard<std::mutex> g(refresh_->m);
                refresh_->gone = true;
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
                // the partitions as last listed, for this profile's host
                info_.reset();
                if (std::optional<cluster::ClusterInfo> ci = link.session().clusterInfo(); ci && ci->host == trimmed(profile_.host))
                    info_ = std::move(ci);
                if (editable) {
                    {
                        const Spacing spacing(8, 10);
                        drawProfile(app, link, st, editable);
                    }
                    widgets::vspace(4);
                    note("Your password and one-time codes are asked for in a separate box when the cluster asks, handed to ssh and "
                         "never stored. A wrong one costs one attempt; Connect again to retry.");
                } else {
                    // the profile in one line while it is in use
                    const cluster::Profile p = link.session().profile();
                    std::string line = p.host + " \xC2\xB7 " + p.checkout;
                    if (!p.container.empty()) line += " \xC2\xB7 " + p.container + (p.bind.empty() ? std::string() : " (bind " + p.bind + ")");
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
            void drawProfile(App& app, ClusterLink& link, const cluster::Status& st, bool editable) {
                widgets::FieldOpts fo;
                fo.enabled = editable;
                {
                    // No port: the worker takes a free one on its node and
                    // says which in its private log.
                    const float w = columnWidth(3, 10);
                    fo.width = design(w * 3 + px(20));
                    const Field f("SSH host");
                    fo.hint = "fiona, or user@login.cluster.org";
                    widgets::inputText("##host", &profile_.host, fo);
                    const ImGuiID hostId = ImGui::GetItemID();
                    widgets::tooltip("A host of your ~/.ssh/config works, with its user, ProxyJump and the rest.");
                    fo.hint.clear();
                    // another host: the partition, account, QoS and time last used there
                    const std::string h = trimmed(profile_.host);
                    if (h != lastHost_ && ImGui::GetActiveID() != hostId) {
                        lastHost_ = h;
                        if (profile_.recall(h)) filled_ = {"the partition, account, QoS, time and binds last used on " + h};
                        else filled_.clear();
                    }
                }
                drawContainer(app, link, st, fo);
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
                    // the image has its own Python: the venv is not used with one
                    const bool inImage = !trimmed(profile_.container).empty();
                    widgets::FieldOpts vo = fo;
                    vo.enabled = fo.enabled && !inImage;
                    vo.hint = inImage ? "not used: the worker runs in the image" : "none";
                    std::string shown = inImage ? std::string() : profile_.venv;
                    widgets::inputText("##venv", inImage ? &shown : &profile_.venv, vo);
                    widgets::tooltip(inImage ? std::string("The worker runs in the container image, with the image's own Python; clear the image to use a venv.")
                                             : std::string("Activated for the worker; it needs numpy (and torch for models, the sirius package to open "
                                                           "cluster TIFF datasets). Empty: the python of the job's modules."));
                }
                note("Container and venv are alternatives: with an image set, the venv is not used.");
                if (const std::string w = cluster::emptyBindWarning(profile_); !w.empty()) note("With no Bind, " + w + ".", kAmber);
                drawSlurmRow(fo);
                drawPartitionNotes(app, link, st, editable);
                const cluster::Partition* part = info_ ? cluster::findPartition(*info_, profile_.partition) : nullptr;
                // the spinners within one node of the partition (nodes that
                // list no GPUs leave the GPUs free: a DGX's may be unmanaged)
                const std::int64_t maxGpus = part && part->gpusPerNode > 0 ? part->gpusPerNode : 16;
                const std::int64_t maxCpus = part && part->cpusPerNode > 0 ? part->cpusPerNode : 256;
                gpus_ = std::min(gpus_, maxGpus);
                cpus_ = std::min(cpus_, maxCpus);
                {
                    const float w = columnWidth(4, 10);
                    fo.width = design(w);
                    {
                        const Field f("Time limit");
                        fo.hint = "01:00:00";
                        widgets::inputText("##time", &profile_.time, fo);
                        const std::string limits = timeLimits(part);
                        if (!limits.empty()) widgets::tooltip(limits);
                        fo.hint.clear();
                    }
                    ImGui::SameLine(0.0f, px(10));
                    {
                        const Field f("GPUs");
                        spinInt("##gpus", &gpus_, 0, maxGpus, 1, fo);
                    }
                    ImGui::SameLine(0.0f, px(10));
                    {
                        const Field f("CPUs");
                        spinInt("##cpus", &cpus_, 1, maxCpus, 1, fo);
                    }
                    ImGui::SameLine(0.0f, px(10));
                    const Field f("Memory");
                    widgets::inputText("##mem", &profile_.mem, fo);
                    if (part && part->memPerNodeMB > 0)
                        widgets::tooltip("A node of " + part->name + " has " + std::to_string(part->memPerNodeMB / 1024) + "G.");
                }
                if (part && part->memPerNodeMB > 0 && cluster::memoryMB(profile_.mem) > part->memPerNodeMB)
                    note("More memory than a node of " + part->name + " has (" + std::to_string(part->memPerNodeMB / 1024) +
                             "G): the job would never start.",
                         kAmber);
            }

            // The container image the worker runs in (Apptainer or
            // Singularity), its launcher, and the cluster's files to pick it from.
            void drawContainer(App& app, ClusterLink& link, const cluster::Status& st, widgets::FieldOpts fo) {
                const float gap = px(10);
                const float browse = buttonWidth("Browse\xE2\x80\xA6", widgets::ButtonKind::Secondary);
                const float launcherW = px(120);
                const float w = std::max(px(120), ImGui::GetContentRegionAvail().x - launcherW - browse - gap * 2);
                fo.width = design(w);
                {
                    const Field f("Container image");
                    fo.hint = "empty: the venv below; or ~/sirius-worker.sif";
                    widgets::inputText("##container", &profile_.container, fo);
                    widgets::tooltip("An Apptainer (Singularity) image with the compiled sirius package, numpy and torch: the worker runs in "
                                     "it (apptainer exec --nv), the checkout and ~/.sirius/run bound into it.");
                    fo.hint.clear();
                }
                ImGui::SameLine(0.0f, gap);
                {
                    const Field f("Launcher");
                    fo.width = design(launcherW);
                    fo.hint = "apptainer";
                    widgets::inputText("##launcher", &profile_.launcher, fo);
                    widgets::tooltip("apptainer, singularity or a full path; when it is not found, `module load` of it is tried, then the "
                                     "other one.");
                    fo.hint.clear();
                }
                ImGui::SameLine(0.0f, gap);
                {
                    const Field f(" ");
                    const bool here = st.sshUp && st.host == trimmed(profile_.host);
                    widgets::ButtonOpts b;
                    b.enabled = fo.enabled && here;
                    b.tooltip = here ? std::string("Pick the image (*.sif) among the cluster's files") : std::string("Log in first (below)");
                    if (widgets::button("Browse\xE2\x80\xA6##container", b)) {
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
                (void)link;
                drawBind(app, st, fo);
                drawEngine(fo);
            }

            // SIRIUS's engine as the job: every step on the node, the results
            // kept there (the image has it at /opt/sirius/bin/sirius-cli).
            void drawEngine(widgets::FieldOpts fo) {
                widgets::checkbox("Run SIRIUS's engine on the node##engine", &profile_.engine);
                widgets::tooltip("The job runs SIRIUS's own engine (sirius-cli serve) with the Python worker beside it: every step of "
                                 "a pipeline then runs on the node, with CUDA where SIRIUS has it, and its results stay there until "
                                 "you look at them or export them. Off: the Python worker alone, which runs only the Python steps.");
                if (!profile_.engine) return;
                fo.width = design(ImGui::GetContentRegionAvail().x);
                const Field f("Engine executable (optional)");
                fo.hint = trimmed(profile_.container).empty() ? "sirius-cli on the job's PATH" : "/opt/sirius/bin/sirius-cli in the image";
                widgets::inputText("##engineBin", &profile_.engineBin, fo);
                widgets::tooltip("Where sirius-cli is on the node (or in the image): a build of the same SIRIUS as this application, or "
                                 "one whose operations are the same; another is refused when the job answers.");
            }

            // Under the image: the host paths bound into it (apptainer --bind)
            // and, folded away, extra entries for the worker's PYTHONPATH.
            void drawBind(App& app, const cluster::Status& st, widgets::FieldOpts fo) {
                const bool inImage = !trimmed(profile_.container).empty();
                const float gap = px(10);
                const float browse = buttonWidth("Add folder\xE2\x80\xA6", widgets::ButtonKind::Secondary);
                fo.enabled = fo.enabled && inImage;
                fo.width = design(std::max(px(120), ImGui::GetContentRegionAvail().x - browse - gap));
                {
                    const Field f("Bind");
                    fo.hint = inImage ? "host paths the worker may read, e.g. /clusterfs" : "used with a container image";
                    widgets::inputText("##bind", &profile_.bind, fo);
                    widgets::tooltip("apptainer --bind: comma separated, src[:dst[:ro]], e.g. /clusterfs:/clusterfs,/global/scratch. The "
                                     "container sees only the image and your home folder without them; each path is checked before the job "
                                     "is submitted.");
                    fo.hint.clear();
                }
                ImGui::SameLine(0.0f, gap);
                {
                    const Field f(" ");
                    const bool here = st.sshUp && st.host == trimmed(profile_.host);
                    widgets::ButtonOpts b;
                    b.enabled = fo.enabled && here;
                    b.tooltip = here ? std::string("Pick a folder among the cluster's files and add it to Bind") : std::string("Log in first (below)");
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
                // the Python path, folded away unless it has something
                if (!trimmed(profile_.containerPythonPath).empty()) showPythonPath_ = true;
                if (!showPythonPath_) {
                    if (widgets::linkButton("Python path (optional)\xE2\x80\xA6##showPyPath", fo.enabled)) showPythonPath_ = true;
                } else {
                    fo.width = design(ImGui::GetContentRegionAvail().x);
                    const Field f("Python path (optional)");
                    fo.hint = "extra entries for PYTHONPATH inside the image, : separated";
                    widgets::inputText("##containerPythonPath", &profile_.containerPythonPath, fo);
                    widgets::tooltip("Appended after the worker's own code (SIRIUS_CONTAINER_PYTHONPATH); paths as the container sees them.");
                }
            }

            // Partition, Account and QoS: lists once the cluster has said
            // what it has, free text all the same.
            void drawSlurmRow(widgets::FieldOpts fo) {
                const float w = columnWidth(3, 10);
                fo.width = design(w);
                const float listWidth = design(w * 3 + px(20));
                bool picked = false;
                {
                    const Field f("Partition");
                    if (info_ && !info_->partitions.empty()) {
                        std::vector<widgets::ComboItem> items;
                        // the ones the user may submit to first, each in sinfo's order
                        for (int pass = 0; pass < 2; ++pass)
                            for (const cluster::Partition& p : info_->partitions) {
                                const bool usable = cluster::hasAssociation(*info_, p.name);
                                if (usable != (pass == 0)) continue;
                                std::string detail = cluster::partitionSummary(p);
                                if (!usable) detail = "no association \xC2\xB7 " + detail;
                                items.push_back(widgets::ComboItem{p.name, detail, !usable});
                            }
                        if (openList_) {
                            // a screenshot: the list as the user sees it on a click
                            ImGui::PushID("##partition");
                            ImGui::OpenPopup("##items");
                            ImGui::PopID();
                            openList_ = false;
                        }
                        widgets::editableCombo("##partition", &profile_.partition, items, fo, &picked, listWidth);
                    } else {
                        widgets::inputText("##partition", &profile_.partition, fo);
                    }
                }
                if (picked) choose();
                ImGui::SameLine(0.0f, px(10));
                {
                    const Field f("Account");
                    const std::vector<std::string> accounts = info_ ? cluster::accountsFor(*info_, profile_.partition) : std::vector<std::string>{};
                    if (accounts.size() > 1) {
                        std::vector<widgets::ComboItem> items;
                        for (const std::string& a : accounts) items.push_back(widgets::ComboItem{a, {}, false});
                        bool pickedAccount = false;
                        widgets::editableCombo("##account", &profile_.account, items, fo, &pickedAccount);
                        if (pickedAccount) choose();
                    } else {
                        widgets::inputText("##account", &profile_.account, fo);
                    }
                }
                ImGui::SameLine(0.0f, px(10));
                const Field f("QoS");
                std::vector<std::string> qos;
                if (info_) {
                    qos = cluster::qosFor(*info_, profile_.partition, profile_.account);
                    if (!info_->associationsKnown)
                        for (const auto& [name, wall] : info_->qosMaxWall) qos.push_back(name);
                }
                if (qos.size() > 1) {
                    std::vector<widgets::ComboItem> items;
                    for (const std::string& q : qos) {
                        std::string detail;
                        if (const auto it = info_->qosMaxWall.find(q); it != info_->qosMaxWall.end() && !it->second.empty())
                            detail = "up to " + it->second;
                        items.push_back(widgets::ComboItem{q, detail, false});
                    }
                    bool pickedQos = false;
                    widgets::editableCombo("##qos", &profile_.qos, items, fo, &pickedQos, design(w * 1.4f));
                    if (pickedQos) choose();
                } else {
                    widgets::inputText("##qos", &profile_.qos, fo);
                }
            }

            // A partition, account or QoS picked from a list: the rest filled
            // in from the association and brought within the partition's nodes.
            void choose() {
                if (!info_ || !cluster::findPartition(*info_, profile_.partition)) return;
                profile_.gpus = static_cast<int>(gpus_);
                profile_.cpus = static_cast<int>(cpus_);
                filled_ = cluster::choosePartition(profile_, *info_, profile_.partition);
                gpus_ = profile_.gpus;
                cpus_ = profile_.cpus;
            }

            std::string timeLimits(const cluster::Partition* part) const {
                std::string s;
                if (part && !part->maxTime.empty()) s = part->name + ": " + part->maxTime;
                if (info_)
                    if (const auto it = info_->qosMaxWall.find(profile_.qos); it != info_->qosMaxWall.end() && !it->second.empty())
                        s += (s.empty() ? "" : "; ") + ("QoS " + profile_.qos + ": " + it->second);
                return s.empty() ? s : "The most allowed \xE2\x80\x94 " + s + ".";
            }

            // Under the Slurm row: what the chosen partition is, what to know
            // before submitting there, and where the list came from.
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
                if (info_) {
                    const cluster::Partition* part = cluster::findPartition(*info_, profile_.partition);
                    if (part) {
                        widgets::textWrapped(part->name + ": " + cluster::partitionSummary(*part, true), 11, theme::kNeutral700);
                        if (!cluster::hasAssociation(*info_, part->name))
                            widgets::textWrapped("You have no association for " + part->name + " (sacctmgr): sbatch will most likely refuse the job.",
                                                 11, theme::kAccentText);
                        const std::string warning = cluster::partitionWarning(*part);
                        if (!warning.empty()) widgets::textWrapped(warning, 11, kAmber, theme::Weight::SemiBold);
                    } else if (!trimmed(profile_.partition).empty() && info_->error.empty()) {
                        widgets::textWrapped(trimmed(profile_.partition) + " is not one of " + host + "'s partitions: sbatch will say whether it exists.",
                                             11, kAmber);
                    }
                    if (!filled_.empty()) {
                        std::string line;
                        for (const std::string& f : filled_) line += (line.empty() ? "" : ", ") + f;
                        widgets::textWrapped("Filled in: " + line + ".", 11, theme::kNeutral600);
                    }
                    if (!info_->error.empty()) widgets::textWrapped(info_->error + ".", 11, theme::kAccentText);
                    for (const std::string& n : info_->notes) widgets::textWrapped(n + ".", 11, theme::kNeutral600);
                } else if (!filled_.empty()) {
                    widgets::textWrapped("Filled in: " + filled_.front() + ".", 11, theme::kNeutral600);
                }
                if (!refreshError.empty()) widgets::textWrapped(refreshError, 11, theme::kAccentText);
                // where the list comes from, and asking again
                std::string line;
                if (querying) line = "Listing " + host + "'s partitions\xE2\x80\xA6";
                else if (info_)
                    line = std::to_string(info_->partitions.size()) + " partitions on " + host +
                           (info_->user.empty() ? std::string() : " for " + info_->user);
                else
                    line = "Log in to list this cluster's partitions, your accounts and QoS.";
                widgets::text(line, 11, theme::kNeutral600);
                ImGui::SameLine(0.0f, px(8));
                const bool loggedIn = st.sshUp && st.host == host;
                const char* label = info_ ? "Refresh##partitions" : (loggedIn ? "List partitions##partitions" : "Log in##partitions");
                if (widgets::linkButton(label, editable && !querying && !host.empty())) {
                    if (loggedIn) startRefresh(app, link);
                    else link.logIn(profileToUse());
                }
                widgets::tooltip(loggedIn ? "Ask " + host + " again: sinfo, sacctmgr, scontrol"
                                          : std::string("The SSH login only, then sinfo and sacctmgr: no job is submitted"));
                ImGui::EndGroup();
            }

            void startRefresh(App& app, ClusterLink& link) {
                {
                    const std::lock_guard<std::mutex> g(refresh_->m);
                    if (refresh_->running) return;
                    refresh_->running = true;
                    refresh_->error.clear();
                }
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

            // The profile as the fields have it, for a login or a connect.
            cluster::Profile profileToUse() {
                profile_.host = trimmed(profile_.host);
                profile_.checkout = trimmed(profile_.checkout);
                profile_.venv = trimmed(profile_.venv);
                profile_.container = trimmed(profile_.container);
                profile_.launcher = trimmed(profile_.launcher);
                if (profile_.launcher.empty()) profile_.launcher = "apptainer";
                profile_.bind = trimmed(profile_.bind);
                profile_.containerPythonPath = trimmed(profile_.containerPythonPath);
                profile_.engineBin = trimmed(profile_.engineBin);
                profile_.partition = trimmed(profile_.partition);
                profile_.account = trimmed(profile_.account);
                profile_.qos = trimmed(profile_.qos);
                profile_.gpus = static_cast<int>(gpus_);
                profile_.cpus = static_cast<int>(cpus_);
                lastHost_ = profile_.host;
                return profile_;
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
                    if (c.tiffReader.empty())
                        widgets::textWrapped("The sirius package is not in the worker's Python: TIFF datasets on the cluster cannot be "
                                             "opened (" +
                                                 st.fix + ").",
                                             11, kAmber);
                    // the job's GPU that the worker cannot compute on, and why
                    if (const std::string why = cluster::gpuUnusableReason(st.node, c); !why.empty() && !c.gpus.empty())
                        widgets::textWrapped(why + ". Until it can, the steps run on the CPU of the job.", 11, kAmber);
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
                        link.connect(profileToUse());
                    }
                }
            }

            cluster::Profile profile_;
            std::int64_t gpus_ = 1, cpus_ = 8;
            std::optional<cluster::ClusterInfo> info_;   // this frame's, for the profile's host
            std::vector<std::string> filled_;            // what the last pick filled in
            std::string lastHost_;
            bool openList_ = false;
            bool showPythonPath_ = false;                // the Python path field unfolded
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
