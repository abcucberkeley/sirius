#include "imgui/panels/diagnostics_panel.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>

#include "core/operation.hpp"
#include "core/workbench.hpp"
#include "imgui/app.hpp"
#include "imgui/panels/diagnostic_cells.hpp"
#include "imgui/panels/diagnostic_cleanup.hpp"
#include "imgui/panels/track_table.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/widgets/controls.hpp"
#include "imgui/widgets/icons.hpp"

// The bottom dock: a 34 px header (▼/▶ toggle, "DIAGNOSTICS · <step>", the
// tab row, a hint) over the page for the selected step: the per-kind grid of
// cells (DiagnosticsBody), the segmentation cleanup tools, or the track table
// of tracked labels. Dock, float and maximise are the window controls on the
// panel's tab bar, as for every panel (App).
//
// Immediate mode: what the page shows is derived again when the revisions it
// came from move. Selection, outputs and the run state refresh at once; edits
// of the pipeline, a step, the dataset or the labels arrive in bursts (a
// slider being dragged, a brush stroke) and refresh 150 ms after the last
// one.

namespace sirius::app::gui {

    using theme::px;
    using theme::Weight;

    namespace {
        constexpr double kRefreshDelay = 0.150;   // seconds after the last burst edit
        const char* const kDiagnostics = "DIAGNOSTICS";

        // The width widgets::tabRow gives a tab: measured in the heavier face
        // so it does not move when chosen.
        float tabWidth(const std::string& name) {
            return theme::snap(std::max(px(54), theme::textSize(name, 12, Weight::ExtraBold).x + px(20)));
        }

        float captionWidth(const std::string& text) {
            ImFont* f = theme::captionFont() ? theme::captionFont() : ImGui::GetFont();
            return f->CalcTextSizeA(cells::fontPx(theme::kCaptionPx), FLT_MAX, 0.0f, text.c_str(), text.c_str() + text.size()).x;
        }
    } // namespace

    struct DiagnosticsPanel::Impl {
        enum class Page { Body,
                          Segment,
                          Tracks };

        App& app;
        bool collapsed = false;
        int tab = 0;

        // what the header shows
        std::string captionText = kDiagnostics;
        std::vector<std::string> tabs;
        DiagnosticsKind kind = DiagnosticsKind::Generic;
        Page page = Page::Body;

        DiagnosticsBody body;
        SegmentCleanupView segment;
        TrackTable tracks;   // the "Tracks" tab of a segment step whose labels are tracked

        // the revisions the page was derived from
        bool fresh = false;
        Revisions seen;
        bool seenTask = false;
        double refreshAt = -1.0;   // a burst refresh is due then (ImGui::GetTime), < 0: none

        explicit Impl(App& a) : app(a), segment(a) {}

        void refresh();
        void watch();
        void drawHeader(float width);
        void drawPage();
        void setTab(int index);
        void setCollapsed(bool on);
    };

    void DiagnosticsPanel::Impl::refresh() {
        fresh = true;
        refreshAt = -1.0;
        const Workbench& wb = app.wb();
        const int sel = wb.selectedIndex();
        const Pipeline& p = wb.pipeline();
        if (sel < 0 || sel >= p.size()) {
            captionText = kDiagnostics;
            tabs.clear();
            kind = DiagnosticsKind::Generic;
            body.setDiagnostics(Diagnostics{}, DiagnosticsKind::Generic, {});
            page = Page::Body;
            return;
        }
        const Step& step = p.at(sel);
        captionText = std::string(kDiagnostics) + " · " + captionCase(step.name);
        const Operation* op = findOperation(step.kind);
        kind = op ? op->info().diagnostics : DiagnosticsKind::Generic;
        Diagnostics d;
        try {
            d = wb.selectedDiagnostics();
        } catch (const std::exception& e) {
            d.warnings.push_back(std::string("No diagnostics: ") + e.what());   // shown above the cells, copyable
        }
        std::vector<std::string> names = DiagnosticsBody::tabNames(d, kind);
        // tracked labels add a table of their tracks beside the cleanup tools
        bool tracked = false;
        if (kind == DiagnosticsKind::Segment) {
            const std::shared_ptr<LabelVolume> labels = wb.viewedLabels();
            tracked = labels && labels->tracked() && labels->tracks();
        }
        if (tracked) names.insert(names.begin(), "Tracks");   // what a tracking step is for: first
        tab = std::clamp(tab, 0, std::max(0, static_cast<int>(names.size()) - 1));
        tabs = std::move(names);
        if (collapsed) return;   // the page is derived when it is expanded (setCollapsed)
        if (kind == DiagnosticsKind::Segment) {
            if (tracked && tab == 0) {
                tracks.setTracks(wb.viewedTrackSummaries());
                page = Page::Tracks;
                return;
            }
            segment.setLabels(wb.viewedLabels());
            page = Page::Segment;
            return;
        }
        DiagnosticsBody::Context ctx;
        try {
            ctx.stepSummary = wb.stepSummary(sel);
            ctx.inputShape = wb.inputMetaOf(sel).shapeString();
            ctx.outputShape = wb.outputMetaOf(sel).shapeString();
            ctx.estimate = "≈ " + bytesText(wb.estimatedBytesOf(sel)) + " output · cache " + toString(step.cache);
        } catch (const std::exception& e) {
            ctx.stepSummary = e.what();
        }
        body.setDiagnostics(std::move(d), kind, std::move(ctx));
        page = Page::Body;
    }

    // Compares the revisions with those the page was derived from.
    void DiagnosticsPanel::Impl::watch() {
        const Revisions& r = app.bridge().rev();
        const bool task = app.bridge().taskRunning();
        bool now = !fresh;
        if (r.selection != seen.selection) {
            tab = 0;   // another step: its first tab
            now = true;
        }
        // a run's results, and what the cleanup tools may do while one is active
        if (r.outputs != seen.outputs || r.runState != seen.runState || task != seenTask || r.viewedStep != seen.viewedStep) now = true;
        const bool burst = r.pipeline != seen.pipeline || r.step != seen.step || r.dataset != seen.dataset || r.labels != seen.labels ||
                           r.operations != seen.operations;
        seen = r;
        seenTask = task;
        if (now) {
            refresh();
            return;
        }
        const double t = ImGui::GetTime();
        if (burst) refreshAt = t + kRefreshDelay;   // restarted by every edit of the burst
        if (refreshAt >= 0.0) {
            if (t >= refreshAt) refresh();
            else app.requestRedraw();   // the frame loop sleeps otherwise
        }
    }

    void DiagnosticsPanel::Impl::setTab(int index) {
        tab = std::max(0, index);
        // The cells follow the tab without deriving anything again; a
        // segment step's tabs change the page itself.
        if (kind == DiagnosticsKind::Segment) refresh();
        app.requestRedraw();
    }

    void DiagnosticsPanel::Impl::setCollapsed(bool on) {
        if (collapsed == on) return;
        collapsed = on;
        refresh();
        app.requestRedraw();
    }

    void DiagnosticsPanel::Impl::drawHeader(float width) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const float h = theme::snap(px(theme::kDiagnosticsHeaderH));
        // 2 px rule on top (the region divider)
        dl->AddRectFilled(origin, ImVec2(origin.x + width, origin.y + theme::crispPen(theme::kRule)), theme::kDivider);
        const float margin = px(14), spacing = px(16);
        const float left = origin.x + margin, right = origin.x + width - margin;
        const float cy = origin.y + h * 0.5f;
        // the "more tabs" button's box
        const float bw = theme::snap(px(24)), bh = theme::snap(px(22));

        // --- the toggle and the caption on the left ---------------------------
        const float chevron = theme::snap(px(12));
        const std::string cap = captionCase(captionText);
        const float leftRoom = std::max(px(20), right - left - chevron - px(8));
        ImFont* cf = theme::captionFont() ? theme::captionFont() : ImGui::GetFont();
        const std::string shownCap = cells::elideIn(cf, theme::kCaptionPx, cap, leftRoom);
        const float capW = std::ceil(captionWidth(shownCap));
        const float leftW = chevron + px(8) + capW;
        {
            ImGui::SetCursorScreenPos(ImVec2(left, origin.y));
            if (ImGui::InvisibleButton("##toggle", ImVec2(std::max(1.0f, leftW), h))) setCollapsed(!collapsed);
            if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            if (shownCap != cap) widgets::tooltip(cap);
            drawIcon(dl, ImVec2(left + chevron * 0.5f, cy), chevron, collapsed ? Icon::ChevronRight : Icon::ChevronDown, theme::kAccent,
                     std::max(1.0f, px(1.5f)));
            const float size = cells::fontPx(theme::kCaptionPx);
            dl->AddText(cf, size, ImVec2(theme::snap(left + chevron + px(8)), theme::snap(cy - std::ceil(size * 1.15f) * 0.5f)),
                        theme::kNeutral600, shownCap.c_str(), shownCap.c_str() + shownCap.size());
        }

        // --- the tabs: in order while they fit, always the active one ----------
        // The header clips rather than widening the window, so it decides
        // itself what to show when squeezed: tabs that do not fit are dropped
        // from the right (behind "…"), the hint goes first.
        const float avail = width - 2.0f * margin - leftW - spacing;
        std::vector<float> widths(tabs.size());
        float total = 0.0f;
        for (std::size_t i = 0; i < tabs.size(); ++i) {
            widths[i] = tabWidth(tabs[i]);
            total += widths[i] + (i ? px(2) : 0.0f);
        }
        const bool overflow = total > avail;
        const float room = overflow ? avail - px(26) - px(2) : avail;   // the "…" button's share
        std::vector<int> visible;
        float used = 0.0f;
        bool activeShown = false;
        const std::size_t active = static_cast<std::size_t>(std::max(tab, 0));
        for (std::size_t i = 0; i < tabs.size(); ++i) {
            const float w = widths[i] + (visible.empty() ? 0.0f : px(2));
            const bool isActive = i == active;
            bool fits = !collapsed && tabs.size() > 1 && used + w <= room;
            if (fits && isActive) activeShown = true;
            // reserve the active tab's width so it is never the one dropped
            if (fits && !isActive && !activeShown && active < tabs.size() && i < active && used + w + widths[active] + px(2) > room)
                fits = false;
            if (fits) {
                visible.push_back(static_cast<int>(i));
                used += w;
            }
        }
        const bool hidden = !collapsed && tabs.size() > 1 && visible.size() < tabs.size();
        float x = left + leftW + spacing;
        if (!visible.empty()) {
            std::vector<std::string> names;
            int current = -1;
            for (std::size_t k = 0; k < visible.size(); ++k) {
                names.push_back(tabs[static_cast<std::size_t>(visible[k])]);
                if (visible[k] == tab) current = static_cast<int>(k);
            }
            ImGui::SetCursorScreenPos(ImVec2(x, theme::snap(cy - px(26) * 0.5f)));
            int chosen = current;
            if (widgets::tabRow("##tabs", names, &chosen) && chosen >= 0 && chosen < static_cast<int>(visible.size()))
                setTab(visible[static_cast<std::size_t>(chosen)]);
            x += used + spacing;
        }
        // Tabs that do not fit stay reachable through this menu.
        if (hidden) {
            ImGui::SetCursorScreenPos(ImVec2(x, theme::snap(cy - bh * 0.5f)));
            widgets::GlyphOpts o;
            o.tooltip = "More tabs";
            if (widgets::glyphButton("##more", Icon::More, ImVec2(24, 22), o)) ImGui::OpenPopup("##moreTabs");
            ImGui::SetNextWindowPos(ImVec2(x, theme::snap(cy + bh * 0.5f)));
            ImGui::PushStyleColor(ImGuiCol_Border, theme::kText);
            ImGui::PushStyleColor(ImGuiCol_PopupBg, theme::kBg);
            ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, theme::crispPen(2));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, px(4, 4));
            int picked = -1;
            if (ImGui::BeginPopup("##moreTabs")) {
                {   // the font is popped inside the popup it was pushed in
                    const theme::FontScope f(12);
                    for (std::size_t i = 0; i < tabs.size(); ++i) {
                        if (std::find(visible.begin(), visible.end(), static_cast<int>(i)) != visible.end()) continue;
                        ImGui::PushID(static_cast<int>(i));
                        if (ImGui::Selectable(tabs[i].c_str(), static_cast<int>(i) == tab, ImGuiSelectableFlags_None,
                                              ImVec2(std::max(px(140), theme::textSize(tabs[i], 12).x + px(16)), px(24))))
                            picked = static_cast<int>(i);
                        ImGui::PopID();
                    }
                }
                ImGui::EndPopup();
            }
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor(2);
            if (picked >= 0) setTab(picked);
            x += bw + px(4);
        }

        // --- the hint, when there is room for it -------------------------------
        const std::string hint = collapsed ? "Click to expand" : "Updates live as parameters change";
        const float rest = right - x;
        if (rest >= px(80)) {
            const std::string shown = widgets::elideText(hint, rest, theme::kSmallPx);
            widgets::drawTextIn(dl, ImVec2(x, origin.y), ImVec2(x + rest, origin.y + h), shown, theme::kSmallPx, theme::kNeutral600,
                                Weight::Regular, 0.0f, 0.5f);
        }

        ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + h));
    }

    void DiagnosticsPanel::Impl::drawPage() {
        switch (page) {
            case Page::Body: body.draw(tab); break;
            case Page::Segment: segment.draw(); break;
            case Page::Tracks: {
                Workbench& wb = app.wb();
                const ViewState& vs = wb.viewState();
                const TrackTable::Events ev = tracks.draw(vs.selectedLabel, vs.followTrack);
                if (ev.chosen) wb.focusTrack(ev.chosen);
                if (ev.follow) wb.setFollowTrack(*ev.follow);
                break;
            }
        }
    }

    DiagnosticsPanel::DiagnosticsPanel(App& app) : impl_(std::make_unique<Impl>(app)) {}
    DiagnosticsPanel::~DiagnosticsPanel() = default;

    void DiagnosticsPanel::draw() {
        Impl& d = *impl_;
        d.watch();
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        if (avail.x < 2.0f || avail.y < 2.0f) return;
        ImGui::GetWindowDrawList()->AddRectFilled(origin, ImVec2(origin.x + avail.x, origin.y + avail.y), theme::kBg);
        ImGui::PushID("diagnostics");
        d.drawHeader(avail.x);
        if (!d.collapsed) {
            widgets::rule(theme::kHairline);
            const ImVec2 rest = ImGui::GetContentRegionAvail();
            if (rest.y >= 2.0f) {
                // The cursor is placed by hand only when the page follows: a
                // window too short for it must not end on a moved cursor with
                // no item after it.
                ImGui::SetCursorScreenPos(ImVec2(origin.x, ImGui::GetCursorScreenPos().y));
                // the page in a region of its own: whatever it lays out stays inside
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
                ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::kBg);
                const bool open = ImGui::BeginChild("##page", rest, ImGuiChildFlags_None,
                                                    ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
                ImGui::PopStyleColor();
                ImGui::PopStyleVar();
                if (open) d.drawPage();
                ImGui::EndChild();
            }
        } else {
            ImGui::Dummy(ImVec2(0, 0));   // the header positioned the cursor under itself
        }
        ImGui::PopID();
    }

    bool DiagnosticsPanel::isCollapsed() const { return impl_->collapsed; }

    void DiagnosticsPanel::setCollapsed(bool collapsed) { impl_->setCollapsed(collapsed); }

    void DiagnosticsPanel::setTab(int index) { impl_->setTab(index); }

    int DiagnosticsPanel::tabCount() const { return static_cast<int>(impl_->tabs.size()); }

} // namespace sirius::app::gui
