#include "imgui/panels/params_panel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>
#include <imgui_internal.h>

#include <sirius/device.hpp>

#include "core/labels.hpp"
#include "core/ops/common.hpp"
#include "core/ops/contrast.hpp"
#include "core/ops/load.hpp"
#include "core/workbench.hpp"
#include "imgui/app.hpp"
#include "imgui/cluster_link.hpp"
#include "imgui/dialogs/dialogs.hpp"
#include "imgui/platform.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/viewer/viewer.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui {

    namespace {

        using theme::px;
        using theme::Weight;

        const char* const kFrozen = "Not while a run or load is in progress — cancel it (Esc) or wait";

        // Display pixels as the design pixels the widgets take.
        float dp(float displayPx) { return displayPx / std::max(theme::scale(), 0.01f); }

        float lineHeight(float designPx, Weight w = Weight::Regular) { return theme::textSize("Ag", designPx, w).y; }

        float captionHeight() {
            const theme::FontScope f(theme::kCaptionPx, theme::captionFont());
            return ImGui::CalcTextSize("AG").y;
        }

        float captionWidth(const std::string& s) {
            const std::string t = captionCase(s);
            const theme::FontScope f(theme::kCaptionPx, theme::captionFont());
            return ImGui::CalcTextSize(t.c_str(), t.c_str() + t.size()).x;
        }

        // A caption cut to `width` display pixels, on code point boundaries.
        std::string fitCaption(const std::string& s, float width) {
            if (captionWidth(s) <= width) return s;
            std::vector<std::size_t> ends;
            for (std::size_t i = 0; i < s.size();) {
                nextCodepoint(s, i);
                ends.push_back(i);
            }
            for (std::size_t n = ends.size(); n-- > 0;) {
                std::string cut = s.substr(0, n == 0 ? 0 : ends[n - 1]);
                while (!cut.empty() && cut.back() == ' ') cut.pop_back();
                cut += "…";
                if (captionWidth(cut) <= width || n == 0) return cut;
            }
            return "…";
        }

        float wrappedHeight(const std::string& s, float designPx, float wrapWidth, Weight w = Weight::Regular) {
            const theme::FontScope f(designPx, w);
            return ImGui::CalcTextSize(s.c_str(), s.c_str() + s.size(), false, std::max(1.0f, wrapWidth)).y;
        }

        // The cursor, moved by hand; callers submit an item afterwards.
        void place(float x, float y) { ImGui::SetCursorScreenPos(ImVec2(theme::snap(x), theme::snap(y))); }
        // ... and tell the layout where the hand-placed content ended.
        void placeEnd(float x, float y) {
            place(x, y);
            ImGui::Dummy(ImVec2(0, 0));
            place(x, y);
        }

        // The tooltip of widgets::tooltip, for the last item even when it is
        // disabled: a control frozen by a run says why it does not answer.
        void tip(const std::string& s) {
            if (s.empty() || !ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip | ImGuiHoveredFlags_AllowWhenDisabled)) return;
            ImGui::PushStyleColor(ImGuiCol_Border, theme::kText);
            ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, theme::crispPen(theme::kBorder));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, px(8, 4));
            if (ImGui::BeginTooltip()) {
                {   // the font is popped inside the tooltip it was pushed in
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

        // The heights and natural widths widgets::button gives its buttons.
        float buttonHeight(bool small = false) {
            if (small) return theme::snap(std::max(px(14), lineHeight(12, Weight::SemiBold)) + 2 * px(4) + 2 * px(theme::kBorder));
            return theme::snap(std::max(px(18), lineHeight(13, Weight::ExtraBold)) + 2 * px(7) + 2 * px(theme::kBorder));
        }
        float buttonWidth(const std::string& label, bool small, float padX, Weight w = Weight::SemiBold) {
            return theme::snap(theme::textSize(label, small ? 12.0f : 13.0f, w).x + 2 * px(padX) + 2 * px(theme::kBorder));
        }

        // core's formatBytes for the number, an em dash for "not known yet".
        std::string bytesOrDash(std::size_t bytes) { return bytes == 0 ? std::string("—") : bytesText(bytes); }

        bool isNumeric(ParamType t) { return t == ParamType::Int || t == ParamType::Double || t == ParamType::Channel; }

        ImU32 colourOfHex(const std::string& hex, ImU32 fallback) {
            try {
                return theme::fromFloat(colorFromHex(hex));
            } catch (const std::exception&) {
                return fallback;
            }
        }

        std::string wavelengthText(const ChannelInfo& ch) {
            return ch.wavelengthNm > 0 ? std::to_string(static_cast<int>(std::lround(ch.wavelengthNm))) : std::string("—");
        }

        // What a text or number field holds while it is being typed into: the
        // value it shows is the parameter's until the field takes the
        // keyboard, and the edit is committed when it lets go of it, not on
        // every keystroke.
        struct FieldBuf {
            double d = 0.0;
            std::int64_t i = 0;
            std::string s;
            int decimals = -2;   // -2: not decided yet
            bool active = false;
            bool edited = false;   // the number changed while the field had the keyboard

            // After a number field is drawn: `changed` is what the widget
            // returned, `nowActive` whether the field has the keyboard now.
            // True when the value is an edit to commit. The arrows and the
            // wheel commit at once, typing when the field lets go of the
            // keyboard -- and only if something was typed while it had it:
            // the value the field was given when it took the keyboard is no
            // edit, and may be stale by then (the assistant edits steps too),
            // so writing it back would undo the newer one. Escape drops what
            // was typed.
            bool settle(bool changed, bool nowActive) {
                const bool was = active;
                active = nowActive;
                if (active) {
                    edited = (was && edited) || changed;
                    return false;
                }
                const bool typed = was && edited;
                edited = false;
                if (was && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) return false;
                return changed || typed;
            }
        };

    } // namespace

    struct ParamsPanel::Impl {
        App& app;
        int runFinishedSlot = 0;
        std::uint64_t runsFinished = 0;

        // What resets the form's buffers when it changes: the step, by id (a
        // step added at the selected place takes the place over), its kind,
        // the values its visibility rules read, and every dataset change and
        // finished run. The contrast range is read from the input, so it is
        // read again when the step's place changes too.
        int builtFor = -2;
        StepId builtStep = 0;
        std::string builtKind, builtVisibility;
        std::uint64_t builtDataset = 0, builtRuns = 0;
        // per form: the field buffers and which "More parameters" are open
        std::map<std::string, FieldBuf> bufs;
        std::map<std::string, bool> moreOpen;
        // a path field's Browse, chosen by hand: this computer (0) or the cluster (1)
        std::map<std::pair<StepId, std::string>, int> pathWhere;
        // The Load step's Source: 0 a file, 1 a folder dataset, as chosen by
        // hand (per step and field); else what the value names (loadSourceIsFolder).
        std::map<std::pair<StepId, std::string>, int> pathKind;
        // derived once per form
        bool haveUpstream = false;
        double dataMin = 0.0, dataMax = 1.0;   // contrast: the input's intensity range
        int contrastDecimals = 3;
        // contrast: the window an automatic (empty) min / max resolves to
        ParamSet effParams;
        std::uint64_t effOutputs = 0;
        bool effValid = false;
        double effLo = 0.0, effHi = 1.0;
        // contrast: an input that failed to read. It is treated as no input,
        // and neither read nor reported again until the form is rebuilt or
        // the input is another output: a drag on an automatic window would
        // otherwise read it, and log the same error, on every frame.
        std::weak_ptr<const StepOutput> unreadable;

        // what the core derives for the selected step without running it, kept
        // until the pipeline, the dataset or the outputs move
        std::uint64_t derivedStamp = 0;
        int derivedFor = -2;
        DatasetMeta derivedInput;
        Validation derivedValidation;
        std::size_t derivedBytes = 0;
        Diagnostics diagnostics;   // SIM warnings, segmentation facts

        void refreshDerived() {
            const Revisions& r = app.bridge().rev();
            const std::uint64_t stamp = r.anyPipeline() + r.dataset + r.outputs + r.backend;
            const int i = index();
            if (stamp == derivedStamp && i == derivedFor) return;
            derivedStamp = stamp;
            derivedFor = i;
            const Step* st = step();
            if (!st) return;
            derivedInput = wb().inputMetaOf(i);
            derivedValidation = wb().stepValidation(i);
            derivedBytes = wb().estimatedBytesOf(i);
            // The step's validation is part of these, and so is whether its
            // result is older than its parameters: they follow every edit.
            diagnostics = st->kind == "sim" || st->kind == "seg" ? wb().selectedDiagnostics() : Diagnostics{};
        }

        // edits made while drawing, run once the frame's form is drawn
        std::vector<std::function<void()>> actions;
        bool firstItem = true;
        // Every parameter edit is refused while a run holds the pipeline, so
        // the whole form is drawn disabled.
        bool formEnabled = true;
        ImU32 dim(ImU32 c) const { return formEnabled ? c : theme::withAlpha(c, 0.45f); }
        float formX = 0.0f, formW = 1.0f;

        // the channel colour picker (Merge)
        ImGuiID colourPopup = 0;
        struct Pick {
            StepId step = 0;
            std::size_t channel = 0;
            float rgb[3] = {1, 1, 1};
            std::string key;
            std::vector<std::string> defaults;
        } pick;
        // Merge without a colour parameter: the chips remember what was chosen.
        std::map<std::pair<StepId, std::size_t>, ImU32> chipColour;
        ImVec2 pickAnchor{0, 0};

        explicit Impl(App& a) : app(a) {
            runFinishedSlot = app.bridge().runFinished.connect([this](bool, const std::string&) { ++runsFinished; });
        }
        ~Impl() { app.bridge().runFinished.disconnect(runFinishedSlot); }
        Impl(const Impl&) = delete;
        Impl& operator=(const Impl&) = delete;

        Workbench& wb() { return app.wb(); }
        int index() const { return app.wb().selectedIndex(); }
        const Step* step() const {
            const Pipeline& p = app.wb().pipeline();
            const int i = index();
            return i >= 0 && i < p.size() ? &p.at(i) : nullptr;
        }
        // A file, colour or model dialog outlives the frame it was opened in,
        // in which a run finishing or another selection may point this form at
        // another step. What the dialog returns applies only while the step it
        // was opened for is still the selected one.
        StepId selectedId() const {
            const Step* st = step();
            return st ? st->id : 0;
        }

        void later(std::function<void()> fn) { actions.push_back(std::move(fn)); }

        // An edit of the step the form shows, run with its index once the
        // frame's form is drawn. The step is found again by id then, and the
        // edit dropped if it is gone.
        void onStep(std::function<void(int)> fn) {
            const StepId id = selectedId();
            later([this, id, fn = std::move(fn)] {
                const int i = wb().pipeline().indexOf(id);
                if (i >= 0) fn(i);
            });
        }

        void setParam(const std::string& key, ParamValue v, bool merge) {
            onStep([this, key, v, merge](int i) { wb().setStepParam(i, key, v, merge ? key : std::string()); });
        }

        // 12 px between the items of the form.
        void gap() {
            if (!firstItem) widgets::vspace(12);
            firstItem = false;
        }

        // The 11 px label above an input.
        void fieldLabel(const std::string& label, float width) {
            if (label.empty()) return;
            widgets::elided(label, width, 11, theme::kNeutral600);
            widgets::vspace(4);
        }

        // The unit a spin box carries as its suffix, drawn inside the field
        // left of its arrows while it is not being typed into.
        static void unitSuffix(ImVec2 at, float width, const std::string& unit) {
            if (unit.empty()) return;
            const ImVec2 ts = theme::textSize(unit, 13);
            const float x = at.x + width - px(16) - px(6) - ts.x;
            if (x < at.x + px(48)) return;
            widgets::drawText(ImGui::GetWindowDrawList(), ImVec2(x, at.y + (px(theme::kInputH) - ts.y) * 0.5f), unit, 13,
                              theme::kNeutral600);
        }

        FieldBuf& buf(const std::string& key) { return bufs[key]; }

        bool readable(const std::shared_ptr<const StepOutput>& input) const { return input && unreadable.lock() != input; }
        void readFailed(const std::shared_ptr<const StepOutput>& input, const std::exception& e) {
            unreadable = input;
            wb().logLine(std::string("Contrast: ") + e.what());
        }

        // --- form shape ------------------------------------------------------------
        // The values every visibility rule of this step depends on. The form is
        // rebuilt when one of them moves, since that is what decides which
        // fields exist at all.
        static std::string visibilitySignature(const OpInfo& info, const ParamSet& params) {
            std::string out;
            for (const ParamSpec& s : info.params)
                for (const ParamSpec::Visibility& rule : s.visibility)
                    if (const ParamValue* v = params.find(rule.key)) out += rule.key + "=" + toDisplayString(*v) + ";";
            return out;
        }

        void checkShape(const Step* st, int i) {
            const Revisions& r = app.bridge().rev();
            const std::string kind = st ? st->kind : std::string();
            const std::string vis = st ? visibilitySignature(st->op().info(), st->params) : std::string();
            const StepId id = st ? st->id : 0;
            const bool sameForm = id == builtStep && kind == builtKind && vis == builtVisibility && r.dataset == builtDataset &&
                                  runsFinished == builtRuns;
            if (sameForm && i == builtFor) return;
            builtFor = i;
            builtStep = id;
            builtKind = kind;
            builtVisibility = vis;
            builtDataset = r.dataset;
            builtRuns = runsFinished;
            // A step moved by an edit above it keeps its fields, one of which
            // may be being typed into; only its input, and so the contrast
            // range, is another one.
            if (!sameForm) {
                bufs.clear();
                moreOpen.clear();
            }
            effValid = false;
            haveUpstream = false;
            unreadable.reset();
            if (!st) return;
            if (kind == "contrast") {
                // the input's intensity range, for the slider extents
                dataMin = 0.0;
                dataMax = 1.0;
                std::shared_ptr<const StepOutput> upstream = wb().upstreamOutput(i);
                haveUpstream = static_cast<bool>(upstream);
                if (upstream) {
                    // A lazily read input reads planes here, and one that cannot
                    // be read (a truncated file, a share gone away) leaves the
                    // sliders on 0 .. 1 rather than taking the panel down.
                    // (an input on the cluster is measured there: Workbench::contrastWindowOf)
                    try {
                        float mn = std::numeric_limits<float>::infinity(), mx = -mn;
                        for (Index c = 0; c < upstream->meta.dims.c; ++c) {
                            const std::optional<ContrastWindow> w = wb().contrastWindowOf(i, st->params, c, true);
                            if (!w) {
                                haveUpstream = false;   // asked of the node: the panel follows when it answers
                                break;
                            }
                            mn = std::min(mn, w->dataMin);
                            mx = std::max(mx, w->dataMax);
                        }
                        if (mn < mx) {
                            dataMin = mn;
                            dataMax = mx;
                        }
                    } catch (const std::exception& e) {
                        haveUpstream = false;
                        readFailed(upstream, e);
                    }
                }
                const double span = dataMax - dataMin;
                contrastDecimals = span >= 100.0 ? 1 : span >= 10.0 ? 2
                                                   : span >= 1.0    ? 3
                                                                    : 4;
            }
        }

        // --- generic editors -----------------------------------------------------------
        // A file / directory field with Browse, and optionally one more button
        // after it (Hub…, Bundles…). `opensDataset`: the Load step's Source,
        // where a path chosen while no dataset is open opens it -- nothing
        // else would, and the form says to choose a file there. It goes
        // through the Open dataset dialog, as File ▸ Open dataset… does, so a
        // plain TIFF asks how its pages map onto (c, t, z). With a dataset
        // open, the path is an edit like the tile or the page order, and the
        // next run of the Load step opens it; so is an emptied field.
        static void openSource(App* a, const std::string& path) {
            a->showDialog(makeOpenDatasetDialog(*a, path, [a](const std::string& p, const OpenOptions& o) { a->openWith(p, o); }));
        }

        // Where a path field's Browse looks while a cluster session is up:
        // 0 this computer, 1 the cluster. Chosen by hand it stays (per step
        // and field); else a cluster value, or the HPC backend on a
        // connected cluster, says the cluster.
        int browseWhere(const std::string& key, const std::string& value) {
            const auto it = pathWhere.find({selectedId(), key});
            if (it != pathWhere.end()) return it->second;
            if (isRemoteDatasetPath(value)) return 1;
            if (value.empty() && wb().backend() == Backend::Hpc && app.cluster().connected()) return 1;
            return 0;
        }

        // The cluster's files (the browser of File ▸ Open from cluster); the
        // choice is the field's value as "cluster://<host>/<path>", which the
        // engine on the node reads where it is. The Load step's Source with
        // no dataset open opens it, as File ▸ Open from cluster does.
        void browseCluster(const std::string& key, const std::string& current, bool dir, bool opensDataset) {
            App* a = &app;
            const StepId forStep = selectedId();
            std::string start, host, remote;
            if (splitClusterPath(current, host, remote)) {
                const std::size_t slash = remote.find_last_of('/');
                start = slash == std::string::npos || slash == 0 ? std::string("/") : remote.substr(0, slash);
            }
            a->defer([this, a, key, dir, start, forStep, opensDataset] {
                a->showDialog(makeClusterBrowser(*a, start, dir, [this, a, key, forStep, opensDataset](const std::string& chosen) {
                    if (chosen.empty() || selectedId() != forStep) return;
                    bufs.erase(key);
                    if (opensDataset && !wb().hasDataset()) a->openDatasetPath(chosen);
                    else wb().setStepParam(index(), key, chosen);
                }));
            });
        }

        // The Load step's Source: File or Folder, as chosen, else as the value says.
        int sourceKind(const std::string& key, const std::string& value) {
            const auto it = pathKind.find({selectedId(), key});
            if (it != pathKind.end()) return it->second;
            return loadSourceIsFolder(value) ? 1 : 0;
        }

        void pathEditor(const ParamSpec& s, const ParamSet& params, float width, const char* extraLabel = nullptr,
                        const std::string& extraTip = {}, std::function<void()> extra = {}, bool opensDataset = false) {
            const std::string key = s.key;
            ImGui::PushID("path");
            const float spacing = px(6);
            const float browseW = buttonWidth("Browse", true, 10);
            const float extraW = extraLabel ? buttonWidth(extraLabel, true, 10) : 0.0f;
            const float editW = std::max(px(40), width - browseW - spacing - (extraLabel ? extraW + spacing : 0.0f));
            FieldBuf& b = buf(key);
            if (!b.active) b.s = params.getString(key);
            ImVec2 at = ImGui::GetCursorScreenPos();
            // Logged in to a cluster: Browse looks on this computer or on the
            // cluster, as the Open dataset dialog's switch does
            const bool clusterUp = app.cluster().sshUp();
            int where = clusterUp ? browseWhere(key, b.s) : 0;
            // the Load step's Source: one file, or a folder as one dataset
            // (its manifest, or the folder dialog that writes one; on the
            // cluster the engine opens it there)
            int kind = opensDataset ? sourceKind(key, b.s) : 0;
            if (clusterUp || opensDataset) {
                widgets::SegmentedOpts so;
                so.enabled = !s.readOnly && formEnabled;
                if (clusterUp) {
                    so.tooltips = {"Browse this computer's files",
                                   "Browse the cluster's files (" + app.cluster().status().host + "): the step reads the file there"};
                    if (widgets::segmented("##where", {"This computer", "Cluster"}, &where, so)) pathWhere[{selectedId(), key}] = where;
                }
                if (opensDataset) {
                    if (clusterUp) ImGui::SameLine(0.0f, px(10));
                    so.tooltips = {"One file: a TIFF / OME-TIFF stack",
                                   "A folder as one dataset: TIFF stacks per channel, time point and tile (its manifest, or the folder dialog makes "
                                   "one), or a zarr / N5 store"};
                    if (widgets::segmented("##kind", {"File", "Folder"}, &kind, so)) pathKind[{selectedId(), key}] = kind;
                }
                at.y += theme::snap(px(26)) + px(6);
                place(at.x, at.y);
            }
            const bool pickDir = s.directory || kind == 1;
            widgets::FieldOpts fo;
            fo.width = dp(editW);
            fo.enabled = formEnabled;
            fo.readOnly = s.readOnly;
            fo.hint = pickDir ? (opensDataset ? "folder…" : "directory…") : "file…";
            widgets::inputText("##edit", &b.s, fo);
            b.active = ImGui::IsItemActive();
            if (ImGui::IsItemDeactivatedAfterEdit() && b.s != params.getString(key)) {
                if (opensDataset && !wb().hasDataset() && !b.s.empty()) {
                    App* a = &app;
                    const std::string path = b.s;
                    a->defer([a, path] { openSource(a, path); });
                } else {
                    setParam(key, b.s, false);
                }
            }
            tip(s.help);
            const float btnY = at.y + (px(theme::kInputH) - buttonHeight(true)) * 0.5f;
            place(at.x + editW + spacing, btnY);
            widgets::ButtonOpts bo;
            bo.small = true;
            bo.enabled = !s.readOnly && formEnabled;
            if (where == 1) bo.tooltip = "The cluster's files, through the SSH session";
            const bool browse = widgets::button("Browse##browse", bo);
            if (browse && where == 1) {
                browseCluster(key, b.s, pickDir, opensDataset);
            } else if (browse && opensDataset && kind == 1) {
                // a folder on this computer: described already (a manifest, a
                // zarr / N5 store) it is the Source; else the folder dialog
                // builds its manifest and opens it, as File ▸ Open folder as dataset does
                App* a = &app;
                const std::string current = b.s;
                const StepId forStep = selectedId();
                a->defer([this, a, key, current, forStep] {
                    const std::string start = current.empty() ? a->lastDir() : (isDirectory(current) ? current : parentPath(current));
                    const std::string path = platform::pickFolderDialog("Choose the dataset's folder", start);
                    if (path.empty() || selectedId() != forStep) return;
                    a->setLastDir(path);
                    bufs.erase(key);
                    bool store = false;
                    for (const char* marker : {".zarray", ".zgroup", "zarr.json", "attributes.json"})
                        if (pathExists(path + "/" + marker)) store = true;
                    if (wb().hasDataset() && (isFolderDataset(path) || store)) wb().setStepParam(index(), key, path);
                    else a->openDatasetPath(path);
                });
            } else if (browse) {
                App* a = &app;
                const bool dir = s.directory;
                const std::string filter = s.fileFilter;
                const std::string current = b.s;
                const StepId forStep = selectedId();
                a->defer([this, a, key, dir, filter, current, forStep, opensDataset] {
                    const std::string start = current.empty() ? a->lastDir() : parentPath(current);
                    const std::string path =
                        dir ? platform::pickFolderDialog("Choose directory", start)
                            : platform::openFileDialog("Choose file", start,
                                                       platform::parseFileFilters(filter.empty() ? std::string("All files (*)") : filter));
                    if (path.empty() || selectedId() != forStep) return;
                    a->setLastDir(dir ? path : parentPath(path));
                    bufs.erase(key);
                    if (opensDataset && !wb().hasDataset()) openSource(a, path);
                    else wb().setStepParam(index(), key, path);
                });
            }
            if (extraLabel) {
                place(at.x + editW + spacing + browseW + spacing, btnY);
                widgets::ButtonOpts eo;
                eo.small = true;
                eo.enabled = formEnabled;
                eo.tooltip = extraTip;
                if (widgets::button((std::string(extraLabel) + "##extra").c_str(), eo) && extra) extra();
            }
            placeEnd(at.x, at.y + px(theme::kInputH));
            ImGui::PopID();
        }

        void editor(const ParamSpec& s, const ParamSet& params, const DatasetMeta& input, float width) {
            const std::string key = s.key;
            ImGui::PushID(key.c_str());
            widgets::FieldOpts fo;
            fo.width = dp(width);
            fo.enabled = formEnabled;
            switch (s.type) {
                case ParamType::Bool: {
                    const bool on = params.getBool(key);
                    if (widgets::tokenCheck((s.label + "##bool").c_str(), on, nullptr, !s.readOnly && formEnabled)) setParam(key, !on, false);
                    tip(s.help);
                    break;
                }
                case ParamType::Channel: {
                    std::vector<std::string> items;
                    for (const ChannelInfo& ch : input.channels) items.push_back(ch.shortName() + " " + ch.label);
                    if (items.empty()) items.emplace_back("ch 0");
                    // A channel the input does not have shows no entry, not the
                    // nearest one: then picking any entry is a change, and
                    // repairs the step.
                    const std::int64_t stored = params.getInt(key);
                    int cur = stored >= 0 && stored < static_cast<std::int64_t>(items.size()) ? static_cast<int>(stored) : -1;
                    if (widgets::combo("##channel", &cur, items, fo)) setParam(key, static_cast<std::int64_t>(cur), false);
                    tip(s.help);
                    break;
                }
                case ParamType::Int: {
                    FieldBuf& b = buf(key);
                    const ImGuiID fid = ImGui::GetID("##int");
                    if (!b.active) b.i = params.getInt(key);
                    const std::int64_t lo = std::isfinite(s.min) ? static_cast<std::int64_t>(s.min) : -1000000000;
                    const std::int64_t hi = std::isfinite(s.max) ? static_cast<std::int64_t>(s.max) : 1000000000;
                    const std::int64_t step = s.step > 0 ? static_cast<std::int64_t>(s.step) : 1;
                    fo.readOnly = s.readOnly;
                    const ImVec2 at = ImGui::GetCursorScreenPos();
                    ImGui::BeginGroup();
                    const bool changed = widgets::inputInt("##int", &b.i, lo, hi, step, fo);
                    const bool commit = b.settle(changed, ImGui::GetActiveID() == fid);
                    if (!b.active) unitSuffix(at, width, s.unit);
                    ImGui::EndGroup();
                    tip(s.help);
                    // a run of arrow clicks is one undo entry
                    if (commit && b.i != params.getInt(key)) setParam(key, b.i, true);
                    break;
                }
                case ParamType::Double: {
                    FieldBuf& b = buf(key);
                    const ImGuiID fid = ImGui::GetID("##double");
                    if (!b.active) b.d = params.getDouble(key);
                    if (b.decimals == -2) {
                        int decimals = s.decimals;
                        if (decimals < 0) {
                            const double mag = std::abs(b.d) > 0 ? std::abs(b.d) : (s.step > 0 ? s.step : 1.0);
                            decimals = mag >= 100 ? 1 : mag >= 1  ? 2
                                                    : mag >= 0.01 ? 4
                                                                  : 6;
                        }
                        b.decimals = decimals;
                    }
                    const double lo = std::isfinite(s.min) ? s.min : -1e12;
                    const double hi = std::isfinite(s.max) ? s.max : 1e12;
                    const double step = s.step > 0 ? s.step : std::pow(10.0, -std::max(b.decimals - 1, 0));
                    fo.readOnly = s.readOnly;
                    const ImVec2 at = ImGui::GetCursorScreenPos();
                    ImGui::BeginGroup();
                    const bool changed = widgets::inputDouble("##double", &b.d, lo, hi, step, b.decimals, fo);
                    const bool commit = b.settle(changed, ImGui::GetActiveID() == fid);
                    if (!b.active) unitSuffix(at, width, s.unit);
                    ImGui::EndGroup();
                    tip(s.help);
                    if (commit && b.d != params.getDouble(key)) setParam(key, b.d, true);
                    break;
                }
                case ParamType::Choice: {
                    int cur = -1;
                    const std::string value = params.getString(key);
                    for (std::size_t c = 0; c < s.choices.size(); ++c)
                        if (s.choices[c] == value) cur = static_cast<int>(c);
                    fo.enabled = !s.readOnly && formEnabled;
                    if (widgets::combo("##choice", &cur, s.choices, fo) && cur >= 0)
                        setParam(key, s.choices[static_cast<std::size_t>(cur)], false);
                    tip(s.help);
                    break;
                }
                case ParamType::Path: pathEditor(s, params, width); break;
                case ParamType::Prompts: break;   // placed in the viewer, listed by promptList()
                case ParamType::String:
                case ParamType::DoubleList:
                case ParamType::StringList:
                case ParamType::Axes: {
                    FieldBuf& b = buf(key);
                    const ParamValue* v = params.find(key);
                    const std::string shown = v ? toDisplayString(*v) : std::string();
                    if (!b.active) b.s = shown;
                    fo.readOnly = s.readOnly;
                    if (s.type == ParamType::DoubleList) fo.hint = "z, y, x";
                    widgets::inputText("##text", &b.s, fo);
                    b.active = ImGui::IsItemActive();
                    tip(s.help);
                    if (ImGui::IsItemDeactivatedAfterEdit() && b.s != shown) {
                        const std::string text = b.s;
                        if (s.type == ParamType::DoubleList) {
                            ParamSet tmp;
                            tmp.set("v", text);
                            setParam(key, tmp.getDoubleList("v"), false);
                        } else if (s.type == ParamType::StringList) {
                            ParamSet tmp;
                            tmp.set("v", text);
                            setParam(key, tmp.getStringList("v"), false);
                        } else {
                            setParam(key, text, false);
                        }
                    }
                    break;
                }
            }
            ImGui::PopID();
        }

        // Generic form: numeric fields in pairs, everything else full width.
        void generic(const std::vector<ParamSpec>& specs, const ParamSet& params, const DatasetMeta& input, bool includeAdvanced,
                     const std::vector<std::string>& skip = {}, const std::string& block = "main") {
            std::vector<ParamSpec> advanced;
            std::string group;
            int col = -1;   // position in the current pair grid; -1 = none
            float rowTop = 0.0f;
            const float colGap = px(10);
            const float colW = std::floor((formW - colGap) * 0.5f);
            const float cellH = lineHeight(11) + px(4) + px(theme::kInputH);
            for (const ParamSpec& s : specs) {
                if (std::find(skip.begin(), skip.end(), s.key) != skip.end()) continue;
                // a field the current mode ignores is not shown at all, not
                // even folded away under "More parameters"
                if (!s.visibleFor(params)) continue;
                // Prompts are placed in the viewer. The list of them stands
                // where the parameter does, under the Task that asks for them,
                // since for a Prompt step they are the point of the form.
                if (s.type == ParamType::Prompts) {
                    col = -1;
                    if (const Step* st = step(); st && isPromptStep(params)) promptList(*st, params);
                    continue;
                }
                if (s.advanced && !includeAdvanced) {
                    ParamSpec c = s;
                    c.advanced = false;
                    advanced.push_back(std::move(c));
                    continue;
                }
                if (!s.group.empty() && s.group != group) {
                    group = s.group;
                    col = -1;
                    gap();
                    widgets::rule(theme::kRule);
                    gap();
                    widgets::caption(fitCaption(group, formW));
                }
                if (s.readOnly && (s.type == ParamType::String || isNumeric(s.type))) {
                    col = -1;
                    gap();
                    const ParamValue* v = params.find(s.key);
                    const std::string value = v ? toDisplayString(*v) : std::string();
                    const float y = ImGui::GetCursorScreenPos().y;
                    const float valueW = std::min(theme::textSize(value, 12).x, formW * 0.6f);
                    widgets::elided(s.label, formW - valueW - px(8), 12, theme::kNeutral600);
                    tip(s.help);
                    place(formX + formW - valueW, y);
                    widgets::elided(value, valueW, 12, theme::kText);
                    placeEnd(formX, y + lineHeight(12));
                    continue;
                }
                if (isNumeric(s.type)) {
                    if (col < 0) col = 0;
                    const bool second = col % 2 == 1;
                    if (!second) {
                        gap();
                        rowTop = ImGui::GetCursorScreenPos().y;
                    }
                    const float x = second ? formX + colW + colGap : formX;
                    const float w = second ? formW - colW - colGap : colW;
                    place(x, rowTop);
                    ImGui::BeginGroup();
                    fieldLabel(s.label, w);
                    editor(s, params, input, w);
                    ImGui::EndGroup();
                    placeEnd(formX, rowTop + cellH);
                    ++col;
                    continue;
                }
                col = -1;
                gap();
                if (s.type != ParamType::Bool) fieldLabel(s.label, formW);
                editor(s, params, input, formW);
            }
            if (!advanced.empty()) {
                gap();
                bool& open = moreOpen[block];
                ImGui::PushID(block.c_str());
                if (widgets::linkButton(open ? "Fewer parameters##more" : "More parameters…##more", formEnabled)) open = !open;
                if (open) generic(advanced, params, input, true, {}, block + "/more");
                ImGui::PopID();
            }
        }

        // --- kind decorations ------------------------------------------------------------
        void factsTable(const std::vector<std::pair<std::string, std::string>>& rows) {
            gap();
            widgets::rule(theme::kRule);
            gap();
            const float keyW = px(90);
            const float h12 = lineHeight(12);
            float y = ImGui::GetCursorScreenPos().y;
            for (const auto& [k, v] : rows) {
                place(formX, y + px(6));
                widgets::elided(k, keyW - px(4), 12, theme::kNeutral600);
                place(formX + keyW, y + px(6));
                widgets::textWrapped(v, 12, theme::kText, Weight::Regular, formW - keyW);
                const float valueH = wrappedHeight(v, 12, formW - keyW);
                y += px(6) + std::max(h12, valueH) + px(6);
                place(formX, y);
                widgets::rule(1);
                y = ImGui::GetCursorScreenPos().y;
            }
            placeEnd(formX, y);
        }

        void channelList(const DatasetMeta& meta) {
            if (meta.channels.empty()) return;
            gap();
            widgets::text("Channels", 11, theme::kNeutral600);
            const float h12 = lineHeight(12);
            float y = ImGui::GetCursorScreenPos().y;
            ImDrawList* dl = ImGui::GetWindowDrawList();
            for (const ChannelInfo& ch : meta.channels) {
                const float top = y + px(5);
                const float chip = px(10);
                const float cy = top + h12 * 0.5f;
                dl->AddRectFilled(ImVec2(theme::snap(formX), theme::snap(cy - chip * 0.5f)),
                                  ImVec2(theme::snap(formX + chip), theme::snap(cy - chip * 0.5f) + theme::snap(chip)),
                                  colourOfHex(ch.hexColor(), theme::kNeutral500));
                widgets::drawText(dl, ImVec2(formX + chip + px(10), top), wavelengthText(ch), 12, theme::kText);
                const float labelX = formX + chip + px(10) + px(36) + px(10);
                place(labelX, top);
                widgets::elided(ch.label, formX + formW - labelX, 12, theme::kText);
                y = top + h12 + px(5);
                place(formX, y);
                widgets::rule(1);
                y = ImGui::GetCursorScreenPos().y;
            }
            placeEnd(formX, y);
        }

        void buildLoad(const Step& step, const ParamSet& params, const OpInfo& info) {
            Workbench& w = wb();
            const DatasetMeta& ds = w.dataset();
            std::vector<std::string> done;
            // the path field first, then facts, channels, then the rest
            for (const ParamSpec& s : info.params)
                if (s.type == ParamType::Path) {
                    gap();
                    fieldLabel(s.label, formW);
                    ImGui::PushID(s.key.c_str());
                    pathEditor(s, params, formW, nullptr, {}, {}, true);
                    ImGui::PopID();
                    done.push_back(s.key);
                    break;
                }
            if (w.hasDataset()) {
                std::vector<std::pair<std::string, std::string>> facts{
                    {"Shape", ds.shapeString()},
                    {"Acquisition", ds.acquisition.empty() ? std::string("—") : ds.acquisition},
                    {"Voxel", ds.voxelString()},
                    {"Dtype", std::string(toString(ds.sourceType)) + " · " + bytesOrDash(ds.bytesOnDisk)}};
                if (ds.hasTiles()) {
                    // grid extent from the tiles' grid indices (rows × columns, layers when > 1)
                    Index rows = 0, cols = 0, layers = 0;
                    for (const TileInfo& t : ds.tiles) {
                        layers = std::max(layers, t.gridIndex[0] + 1);
                        rows = std::max(rows, t.gridIndex[1] + 1);
                        cols = std::max(cols, t.gridIndex[2] + 1);
                    }
                    std::string tiles = std::to_string(ds.tiles.size());
                    if (rows * cols > 1) tiles += format(" · %lld × %lld grid", static_cast<long long>(rows), static_cast<long long>(cols));
                    if (layers > 1) tiles += format(" · %lld layers", static_cast<long long>(layers));
                    facts.emplace_back("Tiles", tiles);
                }
                factsTable(facts);
                channelList(ds);
                // the tile chooser, bound to Load ▸ tile like the viewer toolbar's
                const bool hasTileParam =
                    std::any_of(info.params.begin(), info.params.end(), [](const ParamSpec& s) { return s.key == "tile"; });
                if (ds.hasTiles() && hasTileParam) {
                    std::vector<std::string> items;
                    for (std::size_t i = 0; i < ds.tiles.size(); ++i) items.push_back(std::to_string(i + 1) + " · " + ds.tiles[i].name);
                    // a tile the dataset does not have shows no entry (see the Channel field)
                    const std::int64_t stored = params.getInt("tile");
                    int cur = stored >= 0 && stored < static_cast<std::int64_t>(items.size()) ? static_cast<int>(stored) : -1;
                    gap();
                    fieldLabel("Tile", formW);
                    widgets::FieldOpts fo;
                    fo.width = dp(formW);
                    fo.enabled = formEnabled;
                    if (widgets::combo("##tile", &cur, items, fo)) {
                        const std::int64_t tile = cur;
                        onStep([this, tile](int i) {
                            wb().setStepParam(i, "tile", tile);
                            Bridge& bridge = app.bridge();
                            if (!bridge.running()) bridge.startRun(wb().viewedIndex());
                        });
                    }
                    tip("Which tile of the multi-file dataset the pipeline reads; the viewed step is re-run on it");
                    done.emplace_back("tile");
                }
            } else {
                gap();
                widgets::textWrapped("No dataset loaded. Choose a file above or use File ▸ Open dataset…", 12, theme::kNeutral600,
                                     Weight::Regular, formW);
            }
            (void)step;
            generic(info.params, params, ds, false, done);
        }

        // Einsum axis tiles: kept = outlined, reduced = accent-filled with the
        // reduction name underneath. Click toggles.
        bool axisTiles(std::string& kept, const std::string& reduction) {
            const int n = 5;
            const float h = theme::snap(px(46));
            const float g = theme::snap(px(2));
            const float w = std::floor((formW - g * static_cast<float>(n - 1)) / static_cast<float>(n));
            const ImVec2 origin = ImGui::GetCursorScreenPos();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            bool changed = false;
            for (int i = 0; i < n; ++i) {
                const char ax = "ctzyx"[i];
                const bool keep = kept.find(ax) != std::string::npos;
                const float x = origin.x + static_cast<float>(i) * (w + g);
                const float x1 = i == n - 1 ? origin.x + formW : x + w;
                const ImVec2 a(x, origin.y), b(x1, origin.y + h);
                place(a.x, a.y);
                ImGui::PushID(i);
                const bool pressed = ImGui::InvisibleButton("##axis", ImVec2(std::max(1.0f, b.x - a.x), h));
                const bool hovered = ImGui::IsItemHovered();
                ImGui::PopID();
                if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                if (!keep) dl->AddRectFilled(a, b, dim(theme::kAccent));
                widgets::crispRect(dl, a, b, dim(keep ? (hovered && formEnabled ? theme::kAccent : theme::kDivider) : theme::kAccent), theme::kBorder);
                const ImU32 ink = dim(keep ? theme::kText : theme::kBg);
                widgets::drawTextIn(dl, ImVec2(a.x, a.y + px(6)), ImVec2(b.x, b.y - px(14)), std::string(1, ax), 16, ink,
                                    Weight::ExtraBold, 0.5f, 0.0f);
                {
                    const std::string cap = captionCase(keep ? std::string("keep") : reduction);
                    const theme::FontScope f(9, theme::captionFont());
                    const ImVec2 ts = ImGui::CalcTextSize(cap.c_str());
                    dl->AddText(ImGui::GetFont(), ImGui::GetFontSize(),
                                ImVec2(theme::snap((a.x + b.x - ts.x) * 0.5f), theme::snap(b.y - px(5) - ts.y)), ink, cap.c_str());
                }
                if (pressed) {
                    std::string k;
                    for (const char c : std::string("ctzyx")) {
                        const bool on = kept.find(c) != std::string::npos;
                        if ((c == ax) != on) k += c;   // toggle the clicked axis
                    }
                    kept = k;
                    changed = true;
                }
            }
            placeEnd(origin.x, origin.y + h);
            return changed;
        }

        void buildEinsum(const ParamSet& params, const OpInfo& info, const DatasetMeta& input) {
            const ParamSpec* axesSpec = nullptr;
            const ParamSpec* redSpec = nullptr;
            for (const ParamSpec& s : info.params) {
                if (!axesSpec && s.type == ParamType::Axes) axesSpec = &s;
                if (!redSpec && s.type == ParamType::Choice && std::find(s.choices.begin(), s.choices.end(), "mean") != s.choices.end())
                    redSpec = &s;
            }
            if (!axesSpec) {
                generic(info.params, params, input, false);
                return;
            }
            const std::string axesKey = axesSpec->key;
            const std::string redKey = redSpec ? redSpec->key : std::string();
            const auto normalizeKept = [](const std::string& raw) {
                std::string kept;
                for (const char c : std::string("ctzyx"))
                    if (raw.find(c) != std::string::npos) kept += c;
                return kept;
            };
            std::string kept = normalizeKept(params.getString(axesKey, "ctzyx"));
            const std::string reduction = redKey.empty() ? std::string("mean") : params.getString(redKey, "mean");
            gap();
            fieldLabel(axesSpec->label.empty() ? std::string("Axes — click to keep or reduce") : axesSpec->label, formW);
            if (axisTiles(kept, reduction)) setParam(axesKey, kept, false);
            std::vector<std::string> done{axesKey};
            if (redSpec) {
                gap();
                fieldLabel(redSpec->label, formW);
                int cur = -1;
                for (std::size_t c = 0; c < redSpec->choices.size(); ++c)
                    if (redSpec->choices[c] == reduction) cur = static_cast<int>(c);
                widgets::SegmentedOpts so;
                so.tiles = true;
                so.enabled = formEnabled;
                so.width = dp(formW);
                if (widgets::segmented("##reduction", redSpec->choices, &cur, so) && cur >= 0)
                    setParam(redKey, redSpec->choices[static_cast<std::size_t>(cur)], false);
                done.push_back(redKey);
            }
            // the expression, monospace on the surface
            gap();
            fieldLabel("Expression", formW);
            {
                const std::string expr = "ctzyx -> " + (kept.empty() ? std::string("·") : kept);
                const ImVec2 at = ImGui::GetCursorScreenPos();
                const float th = theme::textSize("Ag", 15).y;
                const float h = px(10) + th + px(10);
                ImDrawList* dl = ImGui::GetWindowDrawList();
                dl->AddRectFilled(at, ImVec2(at.x + formW, at.y + h), theme::kSurface);
                place(at.x + px(12), at.y + px(10));
                widgets::mono(expr, 15, theme::kText);
                placeEnd(at.x, at.y + h);
            }
            generic(info.params, params, input, false, done);
        }

        void buildSim(const ParamSet& params, const OpInfo& info, const DatasetMeta& input) {
            std::vector<std::string> done;
            // first Choice = mode -> segmented control
            for (const ParamSpec& s : info.params) {
                if (s.type != ParamType::Choice || s.advanced) continue;
                bool modeLike = false;
                for (const std::string& c : s.choices)
                    if (c.find("stimate") != std::string::npos || c.find("anual") != std::string::npos) modeLike = true;
                if (!modeLike) continue;
                int cur = -1;
                const std::string value = params.getString(s.key);
                for (std::size_t c = 0; c < s.choices.size(); ++c)
                    if (s.choices[c] == value) cur = static_cast<int>(c);
                gap();
                widgets::SegmentedOpts modeOpts;
                modeOpts.enabled = formEnabled;
                if (widgets::segmented("##simMode", s.choices, &cur, modeOpts) && cur >= 0)
                    setParam(s.key, s.choices[static_cast<std::size_t>(cur)], false);
                tip(s.help);
                done.push_back(s.key);
                break;
            }
            generic(info.params, params, input, false, done);
            if (!diagnostics.warnings.empty()) {
                gap();
                widgets::rule(theme::kRule);
                for (const std::string& w : diagnostics.warnings) {
                    gap();
                    widgets::textWrapped(w, 11, theme::kNeutral600, Weight::Regular, formW);
                }
            }
        }

        // The foundation step's model is a bundle, and a bundle is picked from
        // the registry rather than found on disk: what distinguishes two .ltb
        // files is inside them (task, voxel size, the thresholds they were
        // validated at), and a file dialog shows none of it.
        void buildFoundation(const ParamSet& params, const OpInfo& info, const DatasetMeta& input) {
            std::vector<std::string> done;
            for (const ParamSpec& s : info.params) {
                if (s.type != ParamType::Path || s.advanced) continue;
                const std::string key = s.key;
                gap();
                fieldLabel(s.label, formW);
                ImGui::PushID(key.c_str());
                pathEditor(s, params, formW, "Bundles…",
                           "Choose from the bundles in the registry, with what each was trained for and calibrated at",
                           [this, key] { openHub(key, true); });
                ImGui::PopID();
                done.push_back(key);
                break;
            }
            generic(info.params, params, input, false, done);
        }

        // A Prompt step's prompts: what the viewer's Prompt tool places (Box,
        // Click, Scribble, the viewer's own setting), how, and the objects:
        // one row per object in its label colour ("Object 3 · box + 2 points
        // + 1 correction · score 0.81") with a button that removes it, under
        // it one row per prompt with a button that removes that one, and
        // Clear all. Every change is an undoable edit of the step, which then
        // re-runs as after a click in the viewer.
        void promptList(const Step& st, const ParamSet& params) {
            const std::vector<Prompt> prompts = promptsOf(params);
            const std::vector<PromptObject> objects = promptObjects(prompts);
            const bool planar = st.op().info().promptPlanar;
            gap();
            widgets::rule(theme::kRule);
            gap();
            {
                const float y = ImGui::GetCursorScreenPos().y;
                const float h10 = captionHeight(), h12 = lineHeight(12);
                const float headH = std::max(h10, h12);
                place(formX, y + (headH - h10) * 0.5f);
                widgets::caption("Prompts");
                const std::string count = objects.empty()       ? std::string("none")
                                          : objects.size() == 1 ? std::string("1 object")
                                                                : std::to_string(objects.size()) + " objects";
                place(formX + formW - theme::textSize(count, 12).x, y + (headH - h12) * 0.5f);
                widgets::text(count, 12, theme::kNeutral700);
                placeEnd(formX, y + headH);
            }
            gap();
            {
                static const std::vector<std::string> modes = {"Box", "Click", "Scribble"};
                int mode = static_cast<int>(wb().viewState().promptMode);
                widgets::SegmentedOpts so;
                so.tiles = true;
                so.width = dp(formW);
                so.enabled = formEnabled;
                so.tooltips = {"Drag a box around a new object: the best single prompt",
                               "Click an object; a click inside its mask grows it, Alt or right click there corrects it",
                               "Draw a stroke over a new object"};
                if (widgets::segmented("##promptMode", modes, &mode, so) && mode >= 0) {
                    const PromptMode m = static_cast<PromptMode>(mode);
                    later([this, m] {
                        ViewState vs = wb().viewState();
                        vs.promptMode = m;
                        vs.tool = ViewerTool::Prompt;
                        wb().setViewState(vs);
                    });
                }
            }
            widgets::vspace(6);
            std::string how = "A box, a click or a scribble starts an object; a click inside its mask grows it, and Alt or right click "
                              "there is a correction that refines that mask (Shift starts a new object; a click on a prompt removes it).";
            if (planar) how += " micro-SAM is 2-D: an object and its corrections stay on one plane.";
            widgets::textWrapped(how, 11, theme::kNeutral600, Weight::Regular, formW);
            if (prompts.empty()) return;
            const auto edit = [this](std::vector<Prompt> keep, std::string label) {
                onStep([this, keep = std::move(keep), label = std::move(label)](int i) {
                    ParamSet p = wb().pipeline().at(i).params;
                    p.set(kPromptsKey, promptsValue(keep));
                    wb().setStepParams(i, p, label);
                    app.viewer().promptsEdited(wb().pipeline().at(i).id);
                });
            };
            // the model's score of each object's mask, from the last run
            const int stepIndex = wb().pipeline().indexOf(st.id);
            const std::shared_ptr<const StepOutput> out = stepIndex >= 0 ? wb().output(stepIndex) : nullptr;
            const auto scoreOf = [&](const PromptObject& o) -> std::optional<double> {
                if (!out || o.times.empty()) return std::nullopt;
                for (const auto& [id, score] : promptScores(out->diagnostics, o.times.front()))
                    if (id == o.id) return score;
                return std::nullopt;
            };
            bool manyTimes = false;
            for (const Prompt& p : prompts) manyTimes = manyTimes || p.t != 0;

            widgets::vspace(6);
            const float h12 = lineHeight(12);
            const float rowH = std::max(theme::snap(px(22)), h12 + px(6));
            const float btn = theme::snap(px(18));
            const float indent = px(14);
            ImDrawList* dl = ImGui::GetWindowDrawList();
            float y = ImGui::GetCursorScreenPos().y;
            for (const PromptObject& o : objects) {
                const ImU32 colour = theme::fromFloat(labelColor(o.id));
                ImGui::PushID(static_cast<int>(o.id));
                {
                    // the object: its colour (its mask's in the label overlay), what it holds, its score
                    const float textY = y + (rowH - lineHeight(12, Weight::SemiBold)) * 0.5f;
                    const float chip = px(10);
                    const ImVec2 c0(formX, y + (rowH - chip) * 0.5f);
                    dl->AddRectFilled(c0, ImVec2(c0.x + chip, c0.y + chip), dim(colour), px(2));
                    std::string line = "Object " + std::to_string(o.id) + " \xC2\xB7 " + promptObjectText(o);
                    if (manyTimes) line = "t " + std::to_string(o.times.front()) + " \xC2\xB7 " + line;
                    if (!o.sent()) line += " \xC2\xB7 not sent: no object prompt";
                    else if (const std::optional<double> s = scoreOf(o)) line += " \xC2\xB7 score " + formatNumber(*s, 2);
                    const float textX = formX + chip + px(8);
                    place(textX, textY);
                    widgets::elided(line, formX + formW - btn - px(8) - textX, 12, dim(o.sent() ? theme::kText : theme::kNeutral600), Weight::SemiBold);
                    place(formX + formW - btn, y + (rowH - btn) * 0.5f);
                    widgets::GlyphOpts go;
                    go.borderless = true;
                    go.enabled = formEnabled;
                    go.tooltip = "Remove object " + std::to_string(o.id) + " and all of its prompts";
                    if (widgets::glyphButton("##removeObject", Icon::Close, 18, go))
                        edit(removePromptObject(prompts, o.id), st.name + " \xC2\xB7 removed object " + std::to_string(o.id));
                    y += rowH;
                }
                for (const std::size_t k : o.prompts) {
                    const Prompt& p = prompts[k];
                    const float textY = y + (rowH - h12) * 0.5f;
                    // what it is, drawn as the viewer draws it: an object point a
                    // disc in the object's colour, a correction a dark disc ringed
                    // in it, a box a square, a scribble a stroke
                    const float r = px(4.0f);
                    const ImVec2 c(formX + indent + r + px(1), y + rowH * 0.5f);
                    const ImU32 ink = dim(p.positive ? colour : theme::kNeutral900);
                    if (p.kind == Prompt::Kind::Point) {
                        dl->AddCircleFilled(c, r + px(1.0f), dim(p.positive ? theme::kText : colour));
                        dl->AddCircleFilled(c, r, ink);
                    } else if (p.kind == Prompt::Kind::Box) {
                        dl->AddRect(ImVec2(c.x - r, c.y - r), ImVec2(c.x + r, c.y + r), ink, 0.0f, ImDrawFlags_None, px(1.5f));
                    } else {
                        dl->AddBezierCubic(ImVec2(c.x - r, c.y + r * 0.6f), ImVec2(c.x - r * 0.3f, c.y - r * 1.4f), ImVec2(c.x + r * 0.3f, c.y + r * 1.4f),
                                           ImVec2(c.x + r, c.y - r * 0.6f), ink, px(1.5f));
                    }
                    std::string where;
                    switch (p.kind) {
                        case Prompt::Kind::Point: where = format("point x %g  y %g  z %g", p.at[0], p.at[1], p.at[2]); break;
                        case Prompt::Kind::Box:
                            where = format("box x %g\xE2\x80\x93%g  y %g\xE2\x80\x93%g  z %g\xE2\x80\x93%g", p.box[0], p.box[3] - 1, p.box[1], p.box[4] - 1,
                                           p.box[2], p.box[5] - 1);
                            break;
                        case Prompt::Kind::Scribble:
                            where = format("scribble, %zu voxels at z %g", p.stroke.size(), p.stroke.empty() ? 0.0 : p.stroke.front()[2]);
                            break;
                    }
                    if (o.times.size() > 1) where = format("t %lld \xC2\xB7 ", static_cast<long long>(p.t)) + where;
                    const float textX = formX + indent + 2 * r + px(10);
                    const char* kind = p.positive ? "object" : "correction";
                    const float kindW = theme::textSize("correction", 11).x;
                    place(textX, textY);
                    widgets::elided(where, formX + formW - btn - px(16) - kindW - textX, 12, dim(theme::kText));
                    place(formX + formW - btn - px(8) - theme::textSize(kind, 11).x, y + (rowH - lineHeight(11)) * 0.5f);
                    widgets::text(kind, 11, dim(theme::kNeutral600));
                    place(formX + formW - btn, y + (rowH - btn) * 0.5f);
                    ImGui::PushID(static_cast<int>(k));
                    widgets::GlyphOpts go;
                    go.borderless = true;
                    go.enabled = formEnabled;
                    go.tooltip = p.positive ? "Remove this prompt (the object's last object prompt takes the object with it)" : "Remove this correction";
                    if (widgets::glyphButton("##removePrompt", Icon::Close, 18, go)) {
                        static const char* const names[] = {"a point", "the box", "a scribble"};
                        edit(removePrompt(prompts, k), st.name + " \xC2\xB7 removed " + (p.positive ? names[static_cast<int>(p.kind)] : "a correction") +
                                                           " of object " + std::to_string(o.id));
                    }
                    ImGui::PopID();
                    y += rowH;
                }
                ImGui::PopID();
                place(formX, y);
                widgets::rule(1);
                y = ImGui::GetCursorScreenPos().y;
            }
            placeEnd(formX, y);
            gap();
            if (widgets::linkButton("Clear all prompts##clearPrompts", formEnabled)) edit({}, st.name + " \xC2\xB7 cleared the prompts");
        }

        // Hugging Face, the model cache and the model families (or the
        // foundation bundles) in one dialog; its choice becomes the step's model.
        void openHub(const std::string& key, bool bundles) {
            App* a = &app;
            const StepId forStep = selectedId();
            a->defer([this, a, key, bundles, forStep] {
                a->showDialog(makeModelHubDialog(*a, bundles, [this, key, forStep](const std::string& chosen) {
                    if (chosen.empty() || selectedId() != forStep) return;
                    bufs.erase(key);
                    wb().setStepParam(index(), key, chosen);
                }));
            });
        }

        void buildSeg(const ParamSet& params, const OpInfo& info, const DatasetMeta& input) {
            std::vector<std::string> done;
            for (const ParamSpec& s : info.params)
                if (s.type == ParamType::Path && !s.advanced) {
                    // the path editor (file + Browse) plus "Hub…": Hugging Face
                    // downloads and the local model cache in one dialog
                    const std::string key = s.key;
                    gap();
                    fieldLabel(s.label, formW);
                    ImGui::PushID(key.c_str());
                    pathEditor(s, params, formW, "Hub…", "Search Hugging Face, pick a Cellpose / micro-SAM model, or a cached file",
                               [this, key] { openHub(key, false); });
                    ImGui::PopID();
                    done.push_back(key);
                    break;
                }
            std::string facts;
            for (const DiagnosticFact& f : diagnostics.facts) {
                if (!facts.empty()) facts += " · ";
                facts += f.key.empty() ? f.value : f.key + " " + f.value;
            }
            if (!facts.empty()) {
                gap();
                widgets::textWrapped(facts, 11, theme::kNeutral600, Weight::Regular, formW);
            }
            generic(info.params, params, input, false, done);
            gap();
            widgets::rule(theme::kRule);
            gap();
            widgets::vspace(10);
            const float y = ImGui::GetCursorScreenPos().y;
            widgets::text("Label opacity", 12, theme::kNeutral600);
            const std::string pct = format("%d %%", static_cast<int>(std::lround(wb().viewState().labelOpacity * 100.0)));
            place(formX + formW - theme::textSize(pct, 12).x, y);
            widgets::text(pct, 12, theme::kText);
            placeEnd(formX, y + lineHeight(12));
        }

        // Per-channel rows with a colour chip; a StringList spec (if any)
        // receives the hex colours, otherwise the chips are informative.
        void buildMerge(const Step& step, const ParamSet& params, const OpInfo& info, const DatasetMeta& input) {
            const ParamSpec* colorSpec = nullptr;
            for (const ParamSpec& s : info.params)
                if (s.type == ParamType::StringList) {
                    colorSpec = &s;
                    break;
                }
            const std::vector<std::string> colors = colorSpec ? params.getStringList(colorSpec->key) : std::vector<std::string>{};
            std::vector<std::string> defaults;
            for (const ChannelInfo& other : input.channels) defaults.push_back(other.hexColor());
            if (!input.channels.empty()) gap();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const float h12 = lineHeight(12);
            float y = ImGui::GetCursorScreenPos().y;
            for (std::size_t c = 0; c < input.channels.size(); ++c) {
                const ChannelInfo& ch = input.channels[c];
                const std::string hex = c < colors.size() && !colors[c].empty() ? colors[c] : ch.hexColor();
                ImU32 colour = colourOfHex(hex, theme::kNeutral500);
                if (!colorSpec) {
                    const auto it = chipColour.find({step.id, c});
                    if (it != chipColour.end()) colour = it->second;
                }
                const float rowH = px(22);
                const float top = y + px(6);
                const float textY = top + (rowH - h12) * 0.5f;
                widgets::drawText(dl, ImVec2(formX, textY), wavelengthText(ch), 12, theme::kText);
                const float chipsW = px(44) + px(2) + px(44);
                const float labelX = formX + px(44) + px(10);
                const float labelW = formW - px(44) - px(10) - px(10) - chipsW;
                if (labelW > px(8)) {
                    place(labelX, textY);
                    widgets::elided(ch.label, labelW, 12, theme::kText);
                }
                const float chipX = formX + formW - chipsW;
                dl->AddRectFilled(ImVec2(theme::snap(chipX), theme::snap(top)), ImVec2(theme::snap(chipX + px(44)), theme::snap(top + rowH)),
                                  colour);
                place(chipX + px(44) + px(2), top);
                ImGui::PushID(static_cast<int>(c));
                widgets::GlyphOpts go;
                go.tooltip = "Choose display colour";
                go.glyphPx = 13;
                go.enabled = formEnabled;
                if (widgets::glyphTextButton("##pick", "…", ImVec2(44, 22), go)) {
                    pick = Pick{};
                    pick.step = step.id;
                    pick.channel = c;
                    const ImVec4 v = theme::vec(colour);
                    pick.rgb[0] = v.x;
                    pick.rgb[1] = v.y;
                    pick.rgb[2] = v.z;
                    pick.key = colorSpec ? colorSpec->key : std::string();
                    pick.defaults = defaults;
                    pickAnchor = ImVec2(ImGui::GetItemRectMax().x, ImGui::GetItemRectMax().y + px(4));
                    ImGui::OpenPopup(colourPopup);
                }
                ImGui::PopID();
                y = top + rowH + px(6);
                place(formX, y);
                widgets::rule(1);
                y = ImGui::GetCursorScreenPos().y;
            }
            if (!input.channels.empty()) placeEnd(formX, y);
            std::vector<std::string> done;
            if (colorSpec) done.push_back(colorSpec->key);
            generic(info.params, params, input, false, done);
        }

        // "Channel colour": the picker the "…" of a Merge row opens.
        void drawColourPopup() {
            ImGui::PushStyleColor(ImGuiCol_Border, theme::kText);
            ImGui::PushStyleColor(ImGuiCol_PopupBg, theme::kBg);
            ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, theme::crispPen(2));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, px(12, 12));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, px(8, 8));
            // under the "…" that opened it, right-aligned with it
            ImGui::SetNextWindowPos(pickAnchor, ImGuiCond_Appearing, ImVec2(1.0f, 0.0f));
            if (ImGui::BeginPopupEx(colourPopup, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar |
                                                     ImGuiWindowFlags_NoSavedSettings)) {
                widgets::caption("Channel colour");
                ImGui::SetNextItemWidth(px(220));
                ImGui::ColorPicker3("##picker", pick.rgb,
                                    ImGuiColorEditFlags_NoSidePreview | ImGuiColorEditFlags_PickerHueBar | ImGuiColorEditFlags_DisplayHex |
                                        ImGuiColorEditFlags_InputRGB | ImGuiColorEditFlags_NoAlpha);
                if (widgets::ghostButton("Cancel##colour", false)) ImGui::CloseCurrentPopup();
                ImGui::SameLine();
                if (widgets::primaryButton("OK##colour")) {
                    const Pick chosen = pick;
                    later([this, chosen] { applyColour(chosen); });
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndPopup();
            }
            ImGui::PopStyleVar(3);
            ImGui::PopStyleColor(2);
        }

        void applyColour(const Pick& chosen) {
            if (selectedId() != chosen.step) return;
            const ImU32 c = ImGui::ColorConvertFloat4ToU32(ImVec4(chosen.rgb[0], chosen.rgb[1], chosen.rgb[2], 1.0f));
            chipColour[{chosen.step, chosen.channel}] = c;
            if (chosen.key.empty()) return;
            const Step* st = step();
            if (!st) return;
            std::vector<std::string> cols = st->params.getStringList(chosen.key);
            cols.resize(std::max(cols.size(), std::max(chosen.defaults.size(), chosen.channel + 1)));
            for (std::size_t k = 0; k < cols.size(); ++k)
                if (cols[k].empty() && k < chosen.defaults.size()) cols[k] = chosen.defaults[k];
            cols[chosen.channel] = theme::hex(c);
            wb().setStepParam(index(), chosen.key, cols);
        }

        // Contrast: min / max sliders over the input's range, gamma, Auto /
        // Reset. Every edit is a parameter change, so the viewer's live
        // preview follows it and it is undoable.
        // The window an automatic (empty) min / max resolves to on channel 0.
        bool effective(const ParamSet& p, double& lo, double& hi) {
            const std::uint64_t outs = app.bridge().rev().outputs;
            if (!effValid || effOutputs != outs || effParams != p) {
                effValid = true;
                effOutputs = outs;
                effParams = p;
                std::shared_ptr<const StepOutput> up = wb().upstreamOutput(index());
                haveUpstream = readable(up);
                if (haveUpstream) {
                    // a plane that cannot be read leaves the window unresolved
                    // (see checkShape)
                    try {
                        if (const std::optional<ContrastWindow> eff = wb().contrastWindowOf(index(), p, 0, false)) {
                            effLo = eff->lo;
                            effHi = eff->hi;
                        } else {
                            haveUpstream = false;   // asked of the node
                        }
                    } catch (const std::exception& e) {
                        haveUpstream = false;
                        readFailed(up, e);
                    }
                }
            }
            if (!haveUpstream) return false;
            lo = effLo;
            hi = effHi;
            return true;
        }

        void contrastCommit(const std::string& k, double value, bool merge) {
            onStep([this, k, value, merge](int i) {
                const Pipeline& p = wb().pipeline();
                ParamSet np = p.at(i).params;
                // Leaving automatic pins the other bound to what it resolves
                // to. An input that cannot be read has none, as with no input.
                if (!(np.getDouble("max", 0.0) > np.getDouble("min", 0.0))) {
                    const std::shared_ptr<const StepOutput> up = wb().upstreamOutput(i);
                    if (readable(up)) {
                        try {
                            if (const std::optional<ContrastWindow> eff = wb().contrastWindowOf(i, np, 0, false)) {
                                np.set("min", static_cast<double>(eff->lo));
                                np.set("max", static_cast<double>(eff->hi));
                            }
                        } catch (const std::exception& e) {
                            readFailed(up, e);
                        }
                    }
                }
                np.set(k, value);
                // a drag is one undo entry, and a drag on another Contrast step another one
                const std::string mergeKey = merge ? "contrast/" + k + "#" + std::to_string(p.at(i).id) : std::string();
                wb().setStepParams(i, np, "Step " + Step::number(i) + " · " + (k == "min" ? "Min" : "Max"), mergeKey);
            });
        }

        void buildContrast(const ParamSet& params, const OpInfo& info, const DatasetMeta& input) {
            const std::vector<std::string> done{"min", "max", "gamma"};
            const auto specOf = [&](const char* key) -> const ParamSpec* {
                for (const ParamSpec& sp : info.params)
                    if (sp.key == key) return &sp;
                return nullptr;
            };
            const double lo = dataMin, hi = dataMax;
            const float spinW = px(96), spacing = px(8);
            const float sliderW = std::max(px(40), formW - spinW - spacing);
            const float sliderDy = (px(theme::kInputH) - theme::snap(px(18))) * 0.5f;
            // an empty window (the default) is automatic: show what it resolves to
            double effLoV = 0.0, effHiV = 0.0;
            const bool automatic = !(params.getDouble("max", 0.0) > params.getDouble("min", 0.0));
            const bool resolved = automatic && effective(params, effLoV, effHiV);

            // manual min / max: slider + spin box over the data range
            gap();
            for (const char* key : {"min", "max"}) {
                const std::string k = key;
                if (k == "max") widgets::vspace(10);
                double v = params.getDouble(k, lo);
                if (resolved) v = k == "min" ? effLoV : effHiV;
                ImGui::PushID(key);
                fieldLabel(k == "min" ? "Min" : "Max", formW);
                const ImVec2 at = ImGui::GetCursorScreenPos();
                place(at.x, at.y + sliderDy);
                double sv = std::clamp(v, std::min(lo, hi), std::max(lo, hi));
                widgets::SliderOpts so;
                so.width = dp(sliderW);
                so.enabled = formEnabled;
                if (widgets::slider("##slider", &sv, lo, hi, so)) contrastCommit(k, sv, true);   // one undo entry per drag
                place(at.x + sliderW + spacing, at.y);
                FieldBuf& b = buf("contrast/" + k);
                const ImGuiID fid = ImGui::GetID("##spin");
                if (!b.active) b.d = v;
                widgets::FieldOpts fo;
                fo.width = dp(spinW);
                fo.enabled = formEnabled;
                // the slider spans the data; typed values are not clamped
                const bool changed = widgets::inputDouble("##spin", &b.d, -1e12, 1e12, (hi - lo) / 200.0, contrastDecimals, fo);
                if (b.settle(changed, ImGui::GetActiveID() == fid) && b.d != v) contrastCommit(k, b.d, false);
                placeEnd(at.x, at.y + px(theme::kInputH));
                ImGui::PopID();
            }

            // gamma: slider (0.1 .. 5) with the generic spin box
            if (const ParamSpec* g = specOf("gamma")) {
                gap();
                fieldLabel(g->label, formW);
                const ImVec2 at = ImGui::GetCursorScreenPos();
                place(at.x, at.y + sliderDy);
                double gv = std::clamp(params.getDouble("gamma", 1.0), 0.1, 5.0);
                widgets::SliderOpts so;
                so.width = dp(sliderW);
                so.enabled = formEnabled;
                if (widgets::slider("##gammaSlider", &gv, 0.1, 5.0, so)) setParam("gamma", std::round(gv * 100.0) / 100.0, true);
                place(at.x + sliderW + spacing, at.y);
                editor(*g, params, input, spinW);
                placeEnd(at.x, at.y + px(theme::kInputH));
            }

            // Auto / Reset
            gap();
            {
                const ImVec2 at = ImGui::GetCursorScreenPos();
                widgets::ButtonOpts ao;
                ao.small = true;
                ao.enabled = formEnabled;
                ao.tooltip = "Min / max on the input's percentiles (see More parameters)";
                if (widgets::button("Auto##auto", ao))
                    onStep([this](int i) {
                        if (const std::optional<ParamSet> p = wb().contrastAutoOf(i, wb().pipeline().at(i).params))
                            wb().setStepParams(i, *p, "Auto contrast");
                        else if (wb().upstreamOutput(i))
                            wb().logLine("Auto contrast: the cluster node is measuring the input; press Auto again in a moment.");
                    });
                place(at.x + buttonWidth("Auto", true, 10) + px(8), at.y);
                widgets::ButtonOpts ro;
                ro.kind = widgets::ButtonKind::Ghost;
                ro.small = true;
                ro.enabled = formEnabled;
                ro.tooltip = "Min / max over the input's full range, gamma 1";
                if (widgets::button("Reset##reset", ro))
                    onStep([this](int i) {
                        if (const std::optional<ParamSet> p = wb().contrastResetOf(i, wb().pipeline().at(i).params))
                            wb().setStepParams(i, *p, "Reset contrast");
                        else if (wb().upstreamOutput(i))
                            wb().logLine("Reset contrast: the cluster node is measuring the input; press Reset again in a moment.");
                    });
                placeEnd(at.x, at.y + buttonHeight(true));
            }

            generic(info.params, params, input, false, done);
        }

        // Offered above the fields by any operation that has them. Choosing
        // one writes its values into the step -- an ordinary undoable
        // parameter change -- and the control returns to its caption, because
        // what a step holds afterwards is a set of values and not a mode.
        void buildPresets(const OpInfo& info) {
            if (info.presets.empty()) return;
            gap();
            fieldLabel("Preset", formW);
            const float fontSize = 13;
            ImGui::PushFont(theme::font(), fontSize);
            const float fh = ImGui::GetFontSize();
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(px(8), std::max(2.0f, std::floor((px(theme::kInputH) - fh) * 0.5f))));
            ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, theme::crispPen(theme::kBorder));
            ImGui::PushStyleColor(ImGuiCol_FrameBg, theme::kBg);
            ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, theme::kBg);
            ImGui::PushStyleColor(ImGuiCol_FrameBgActive, theme::kBg);
            ImGui::PushStyleColor(ImGuiCol_Border, theme::kText);
            ImGui::PushStyleColor(ImGuiCol_Header, theme::kSurface);
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, theme::kNeutral200);
            ImGui::PushStyleColor(ImGuiCol_HeaderActive, theme::kNeutral300);
            ImGui::PushStyleColor(ImGuiCol_PopupBg, theme::kBg);
            ImGui::SetNextItemWidth(formW);
            ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, theme::crispPen(2));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, px(0, 2));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, px(0, 0));
            const bool open = ImGui::BeginCombo("##preset", "", ImGuiComboFlags_NoArrowButton | ImGuiComboFlags_HeightLarge);
            std::string chosen;
            if (open) {
                for (std::size_t k = 0; k < info.presets.size(); ++k) {
                    const ParamPreset& preset = info.presets[k];
                    ImGui::PushID(static_cast<int>(k));
                    const ImVec2 p = ImGui::GetCursorScreenPos();
                    const float h = theme::snap(px(26));
                    if (ImGui::Selectable("##item", false, ImGuiSelectableFlags_None, ImVec2(0, h))) chosen = preset.name;
                    widgets::drawTextIn(ImGui::GetWindowDrawList(), ImVec2(p.x + px(8), p.y), ImVec2(p.x + ImGui::GetItemRectSize().x, p.y + h),
                                        preset.name, 12, theme::kText, Weight::Regular, 0.0f, 0.5f);
                    tip(preset.summary);
                    ImGui::PopID();
                }
                ImGui::EndCombo();
            }
            ImGui::PopStyleVar(3);
            const ImVec2 min = ImGui::GetItemRectMin(), max = ImGui::GetItemRectMax();
            ImGui::PopStyleColor(8);
            ImGui::PopStyleVar(2);
            ImGui::PopFont();
            if (!open) {
                ImDrawList* dl = ImGui::GetWindowDrawList();
                widgets::drawTextIn(dl, ImVec2(min.x + px(8), min.y), ImVec2(max.x - px(24), max.y), "Start from…", 13, dim(theme::kText),
                                    Weight::Regular, 0.0f, 0.5f);
                drawIcon(dl, ImVec2(max.x - px(13), (min.y + max.y) * 0.5f), px(12), Icon::ChevronDown, dim(theme::kText), px(1.5f));
                if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                tip("Fill the fields below for a kind of structure; every one stays editable");
            }
            if (!chosen.empty()) onStep([this, chosen](int i) { wb().applyPreset(i, chosen); });
        }

        // --- the form ---------------------------------------------------------------------
        void drawForm(const Step* st) {
            firstItem = true;
            formX = ImGui::GetCursorScreenPos().x;
            formW = std::max(px(60), ImGui::GetContentRegionAvail().x);
            colourPopup = ImGui::GetID("##channelColour");
            if (!st) {
                widgets::text("No step selected.", 12, theme::kNeutral600);
                return;
            }
            const ParamSet params = st->params;   // a copy: edits land after the frame
            const OpInfo& info = st->op().info();
            const DatasetMeta& input = derivedInput;
            // The fields' ids carry the step's: when another step is selected
            // (the assistant selects and adds steps too), a field of the one
            // before that still has the keyboard, a drag or an open popup is
            // not taken for this step's field of the same key, and its edit
            // is dropped rather than written into this step.
            ImGui::PushID(static_cast<int>(st->id));
            buildPresets(info);
            const std::string& kind = st->kind;
            if (kind == "load") buildLoad(*st, params, info);
            else if (kind == "einsum") buildEinsum(params, info, input);
            else if (kind == "sim") buildSim(params, info, input);
            else if (kind == "seg") buildSeg(params, info, input);
            else if (kind == "foundation") buildFoundation(params, info, input);
            else if (kind == "merge") buildMerge(*st, params, info, input);
            else if (kind == "contrast") buildContrast(params, info, input);
            else generic(info.params, params, input, false);
            ImGui::PopID();

            // validation: errors in the accent, warnings in grey
            const Validation& v = derivedValidation;
            std::string text;
            for (const std::string& e : v.errors) text += (text.empty() ? "" : "\n") + e;
            for (const std::string& w : v.warnings) text += (text.empty() ? "" : "\n") + w;
            if (!text.empty()) {
                gap();
                widgets::textWrapped(text, 11, v.ok() ? theme::kNeutral600 : theme::kAccentText, Weight::Regular, formW);
            }
            drawColourPopup();
        }

        // --- the panel ----------------------------------------------------------------------
        void draw() {
            Workbench& w = wb();
            Bridge& bridge = app.bridge();
            const int i = index();
            const Step* st = step();
            checkShape(st, i);
            refreshDerived();
            const bool running = bridge.running();
            const bool busy = running || bridge.taskRunning();
            // Every parameter edit is refused while a run holds the pipeline
            // (Workbench::canEdit), so the whole form goes with it rather
            // than accepting values that are dropped.
            const bool editable = w.canEdit() && !bridge.taskRunning();

            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
            const ImVec2 origin = ImGui::GetCursorScreenPos();
            const ImVec2 avail = ImGui::GetContentRegionAvail();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const float x = origin.x + px(18);
            const float width = std::max(px(40), avail.x - px(36));
            const float rule = theme::crispPen(theme::kRule);
            const float h10 = captionHeight(), h11 = lineHeight(11), h12 = lineHeight(12);

            // --- header ---
            float headerH = 0.0f;
            {
                const float kickH = std::max(h10, h11);
                const float nameH = std::max(theme::snap(px(24)), lineHeight(theme::kH4Px, Weight::ExtraBold));
                const float y0 = origin.y + px(14);
                const std::string kicker = st ? "Step " + Step::number(i) + " · " + st->op().info().kindLabel : std::string("Step");
                const std::string state = st ? (st->enabled ? "enabled" : "skipped") : std::string();
                const float stateW = state.empty() ? 0.0f : theme::textSize(state, 11).x;
                place(x, y0 + (kickH - h10) * 0.5f);
                // The accent at 10 / 11 px is 3.8:1 on this background; the
                // darkened one (theme::kAccentText) is the same red at 5.2:1.
                widgets::caption(fitCaption(kicker, width - stateW - px(8)), theme::kAccentText);
                if (!state.empty()) {
                    place(x + width - stateW, y0 + (kickH - h11) * 0.5f);
                    widgets::text(state, 11, st->enabled ? theme::kAccentText : theme::kNeutral600);
                }
                const float ny = y0 + kickH + px(4);
                const float helpS = theme::snap(px(24));
                const std::string name = st ? st->name : std::string("—");
                const float nameW = width - px(10) - helpS;
                const std::string shown = widgets::elideText(name, nameW, theme::kH4Px, Weight::ExtraBold);
                place(x, ny + (nameH - lineHeight(theme::kH4Px, Weight::ExtraBold)) * 0.5f);
                widgets::text(shown, theme::kH4Px, theme::kText, Weight::ExtraBold);
                if (st) {
                    tip(shown != name ? name + "\nDouble-click to rename" : std::string("Double-click to rename"));
                    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                        App* a = &app;
                        const StepId id = st->id;
                        const std::string current = st->name;
                        a->defer([a, id, current] {
                            a->promptText("Rename step", "Name", current, [a, id](const std::string& text) {
                                const int now = a->wb().pipeline().indexOf(id);
                                if (now >= 0) a->wb().renameStep(now, text);
                            });
                        });
                    }
                }
                place(x + width - helpS, ny + (nameH - helpS) * 0.5f);
                widgets::GlyphOpts ho;
                ho.active = app.helpOpen();
                ho.iconPx = 14;
                ho.tooltip = widgets::withShortcut("Explain this step", shortcutText(keys::helpForStep));
                if (widgets::glyphButton("##help", Icon::Help, 24, ho)) {
                    App* a = &app;
                    a->defer([a] { a->toggleHelp(); });
                }
                headerH = theme::snap(px(14) + kickH + px(4) + nameH + px(10));
            }

            // --- the fixed sections' metrics: backend, cache, footer ---
            std::string backendNote;
            // the HPC backend without SIRIUS's engine runs nothing: the one reason, here too
            const RunGate gate = w.runGate();
            if (st && w.backend() == Backend::Hpc) {
                // With SIRIUS's engine in the job every step runs on the node.
                const RemoteConfig& rc = w.remoteConfig();
                if (rc.hasEngine())
                    backendNote = "Runs on the cluster node" + (rc.where.empty() ? std::string() : " (" + rc.where + ")") +
                                  ", by SIRIUS's engine; the result stays there.";
                else if (!gate.enabled)
                    backendNote = gate.why + ". Nothing runs on HPC until SIRIUS's engine answers; choose CPU/CUDA to run on this computer.";
                else
                    backendNote = "Runs on the HPC worker.";
            }
            // Where the step's last result was computed (and whether it is still there).
            if (st && i > 0)
                if (const std::shared_ptr<const StepOutput> out = w.output(i); out && !w.placementOf(i).empty()) {
                    char secs[32];
                    std::snprintf(secs, sizeof secs, "%.1f s", out->seconds);
                    std::string last = "Last run " + placementText(*out) + ", " + secs + ".";
                    if (!out->gone.empty()) last += " Its result is gone (" + out->gone + "): run the step again.";
                    backendNote = backendNote.empty() ? last : backendNote + "\n" + last;
                }
            static const char* const kCacheNotes[] = {
                "Fastest scrubbing; evicted first when GPU/RAM fills.",
                "Survives restarts; written to the zarr scratch directory. Best for slow steps like reconstruction.",
                "Nothing stored; recomputed from the previous step on demand. Good for cheap steps."};
            const std::string cacheNote = st ? kCacheNotes[static_cast<int>(st->cache)] : std::string();
            const float tileH = theme::snap(px(36));
            // HPC: the "Cluster device: GPU | CPU" row under the tiles
            const bool hpc = w.backend() == Backend::Hpc;
            const float deviceRowH = hpc ? px(8) + theme::snap(px(26)) : 0.0f;
            const float backendH = px(16) + h10 + px(8) + tileH + deviceRowH + (backendNote.empty() ? 0.0f : px(8) + wrappedHeight(backendNote, 12, width)) + px(16);
            const float cacheH = px(16) + std::max(h10, h12) + px(8) + tileH + (cacheNote.empty() ? 0.0f : px(8) + wrappedHeight(cacheNote, 12, width)) + px(16);
            const float btnH = buttonHeight();
            const float footerH = rule + px(14) + btnH + px(14);
            const float sectionsH = rule + backendH + rule + cacheH;
            const float bodyH = std::max(px(60), avail.y - headerH - sectionsH - footerH);

            // --- body ---
            place(origin.x, origin.y + headerH);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(px(18), 0));
            const bool bodyOpen = ImGui::BeginChild("##form", ImVec2(avail.x, bodyH), ImGuiChildFlags_AlwaysUseWindowPadding,
                                                    ImGuiWindowFlags_None);
            ImGui::PopStyleVar();
            if (bodyOpen) {
                formEnabled = editable;
                ImGui::BeginDisabled(!editable);
                drawForm(st);
                ImGui::EndDisabled();
                widgets::vspace(16);
            }
            ImGui::EndChild();
            if (!editable) tip(kFrozen);

            // --- backend ---
            float y = origin.y + headerH + bodyH;
            dl->AddRectFilled(ImVec2(x, theme::snap(y)), ImVec2(x + width, theme::snap(y) + rule), theme::kDivider);
            y = theme::snap(y) + rule + px(16);
            place(x, y);
            widgets::caption("Backend");
            y += h10 + px(8);
            place(x, y);
            {
                int backend = static_cast<int>(w.backend());
                widgets::SegmentedOpts so;
                so.tiles = true;
                so.width = dp(width);
                const bool cuda = cudaAvailable();
                so.optionEnabled = {cuda, true, true};
                so.tooltips = {cuda ? "Run on the selected CUDA device" : "No CUDA device is available in this build / machine", "",
                               "Run on the remote worker (Preferences ▸ HPC)"};
                if (widgets::segmented("##backend", {"CUDA", "CPU", "HPC"}, &backend, so)) {
                    const Backend b = static_cast<Backend>(backend);
                    later([this, b] { wb().setBackend(b); });
                }
            }
            y += tileH;
            if (hpc) {
                // Where the HPC worker computes, sent with each step: a
                // switch needs no new job. The GPU only when the job has one.
                const float rowH = theme::snap(px(26));
                y += px(8);
                place(x, y + (rowH - theme::textSize("Cluster device", 12).y) * 0.5f);
                widgets::text("Cluster device", 12, theme::kNeutral700);
                const std::vector<std::string> devices = {"GPU", "CPU"};
                place(x + width - widgets::segmentedWidth(devices), y);
                std::string why;
                const bool gpu = app.cluster().hpcGpuUsable(&why);
                int device = static_cast<int>(w.hpcDevice());
                widgets::SegmentedOpts so;
                so.optionEnabled = {gpu, true};
                so.tooltips = {gpu ? "Run the worker's steps on the job's GPU" : why,
                               "Run the worker's steps on the job's CPU (the GPU stays allocated)"};
                if (widgets::segmented("##hpcDevice", devices, &device, so)) {
                    const HpcDevice d = static_cast<HpcDevice>(device);
                    later([this, d] { wb().setHpcDevice(d); });
                }
                y += rowH;
            }
            if (!backendNote.empty()) {
                place(x, y + px(8));
                widgets::textWrapped(backendNote, 12, gate.enabled ? theme::kNeutral600 : theme::kAccentText, Weight::Regular, width);
                y += px(8) + wrappedHeight(backendNote, 12, width);
            }
            y += px(16);

            // --- cache output ---
            dl->AddRectFilled(ImVec2(x, theme::snap(y)), ImVec2(x + width, theme::snap(y) + rule), theme::kDivider);
            y = theme::snap(y) + rule + px(16);
            {
                const float headH = std::max(h10, h12);
                place(x, y + (headH - h10) * 0.5f);
                widgets::caption("Cache output");
                if (st) {
                    const std::string size = "≈ " + bytesOrDash(derivedBytes);
                    place(x + width - theme::textSize(size, 12).x, y + (headH - h12) * 0.5f);
                    widgets::text(size, 12, theme::kNeutral700);
                }
                y += headH + px(8);
            }
            place(x, y);
            {
                int cache = st ? static_cast<int>(st->cache) : -1;
                widgets::SegmentedOpts so;
                so.tiles = true;
                so.width = dp(width);
                so.enabled = editable && st;
                if (editable) so.tooltips = {"Cached in GPU/RAM", "Cached on disk (zarr scratch)", "Recomputed on demand"};
                if (widgets::segmented("##cache", {"Memory", "Disk", "Recompute"}, &cache, so) && cache >= 0) {
                    const CachePolicy c = static_cast<CachePolicy>(cache);
                    onStep([this, c](int now) { wb().setStepCache(now, c); });
                }
                if (!editable) tip(kFrozen);
            }
            y += tileH;
            if (!cacheNote.empty()) {
                place(x, y + px(8));
                widgets::textWrapped(cacheNote, 12, theme::kNeutral600, Weight::Regular, width);
            }

            // --- footer: Run step / View / Remove ---
            {
                const float fy = origin.y + avail.y - footerH;
                dl->AddRectFilled(ImVec2(origin.x, theme::snap(fy)), ImVec2(origin.x + avail.x, theme::snap(fy) + rule), theme::kDivider);
                const float by = fy + rule + px(14);
                const bool showRemove = st && !st->pinned;
                const float viewW = buttonWidth("View", false, 12);
                const float removeW = buttonWidth("Remove", false, 8);
                const float runW = std::max(px(40), width - px(8) - viewW - (showRemove ? px(8) + removeW : 0.0f));
                App* a = &app;
                place(x, by);
                widgets::ButtonOpts run;
                run.kind = widgets::ButtonKind::Primary;
                run.width = dp(runW);
                run.enabled = st && !busy && w.hasDataset() && gate.enabled;
                if (widgets::button("Run step##run", run)) a->defer([a] { a->runSelectedStep(); });
                tip(gate.enabled ? widgets::withShortcut("Run this step; its input has to be computed already", shortcutText(keys::runSelected)) : gate.why);
                place(x + runW + px(8), by);
                widgets::ButtonOpts view;
                view.enabled = st != nullptr;
                view.tooltip = "Show this step's output in the viewer";
                if (widgets::button("View##view", view)) later([this, i] { wb().view(i); });
                if (showRemove) {
                    place(x + runW + px(8) + viewW + px(8), by);
                    widgets::ButtonOpts remove;
                    remove.kind = widgets::ButtonKind::Ghost;
                    remove.enabled = editable;
                    // Between frames an assistant call deferred ahead of this one
                    // may have moved or removed steps: the step is found again by id.
                    const StepId id = st->id;
                    if (widgets::button("Remove##remove", remove))
                        a->defer([a, id] {
                            const int now = a->wb().pipeline().indexOf(id);
                            if (now >= 0) a->removeStepAt(now);
                        });
                    if (!editable) tip(kFrozen);
                }
                placeEnd(origin.x, origin.y + avail.y);
            }
            ImGui::PopStyleVar();

            // the edits of this frame, now that nothing is drawn from the old values;
            // one may read the input (Contrast's Auto, Reset, a bound leaving
            // automatic), and a plane of a lazily read input can fail
            std::vector<std::function<void()>> pendingActions;
            pendingActions.swap(actions);
            for (auto& fn : pendingActions) {
                try {
                    fn();
                } catch (const std::exception& e) {
                    w.logLine(std::string("Error: ") + e.what());
                }
            }
        }
    };

    ParamsPanel::ParamsPanel(App& app) : impl_(std::make_unique<Impl>(app)) {}
    ParamsPanel::~ParamsPanel() = default;

    void ParamsPanel::draw() { impl_->draw(); }

} // namespace sirius::app::gui
