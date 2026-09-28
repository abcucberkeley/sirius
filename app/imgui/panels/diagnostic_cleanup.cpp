#include "imgui/panels/diagnostic_cleanup.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

#include <imgui.h>

#include "core/workbench.hpp"
#include "imgui/app.hpp"
#include "imgui/panels/diagnostic_cells.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/widgets/controls.hpp"
#include "imgui/widgets/icons.hpp"

namespace sirius::app::gui {

    using theme::px;
    using theme::Weight;

    namespace {

        struct SegTool {
            PaintTool tool;
            Icon icon;
            const char* name;
        };
        constexpr std::array<SegTool, 8> kSegTools{{
            {PaintTool::Brush, Icon::Brush, "Brush"},
            {PaintTool::Erase, Icon::Erase, "Erase"},
            {PaintTool::Fill, Icon::Fill, "Fill region"},
            {PaintTool::Pick, Icon::Pick, "Pick label"},
            {PaintTool::Merge, Icon::Merge, "Merge labels"},
            {PaintTool::Split, Icon::Split, "Split (watershed seed)"},
            {PaintTool::Delete, Icon::Trash, "Delete label"},
            {PaintTool::Lasso, Icon::Lasso, "Lasso"},
        }};

        std::size_t toolIndex(PaintTool t) {
            for (std::size_t i = 0; i < kSegTools.size(); ++i)
                if (kSegTools[i].tool == t) return i;
            return 0;
        }

        const char* const kFrozen = "Not while a run or load is in progress — cancel it (Esc) or wait";
        const char* const kToolsCaption = "CLEANUP TOOLS · PAINT IN VIEWER";
        const char* const kLinks = "merge · split · ";

        std::string groupThousands(Index n) {
            const std::string digits = std::to_string(n < 0 ? -n : n);
            std::string s = n < 0 ? "-" : "";
            for (std::size_t i = 0; i < digits.size(); ++i) {
                if (i > 0 && (digits.size() - i) % 3 == 0) s += "\xE2\x80\x89";   // thin space
                s += digits[i];
            }
            return s;
        }

        // The caption face at 10 px: its width and a draw at (x, y).
        float captionWidth(const std::string& text) {
            ImFont* f = theme::captionFont() ? theme::captionFont() : ImGui::GetFont();
            return f->CalcTextSizeA(cells::fontPx(theme::kCaptionPx), FLT_MAX, 0.0f, text.c_str(), text.c_str() + text.size()).x;
        }
        float captionHeight() { return std::ceil(cells::fontPx(theme::kCaptionPx) * 1.3f); }
        void drawCaption(ImDrawList* dl, ImVec2 at, const std::string& text, float width) {
            ImFont* f = theme::captionFont() ? theme::captionFont() : ImGui::GetFont();
            const std::string shown = cells::elideIn(f, theme::kCaptionPx, text, width);
            dl->AddText(f, cells::fontPx(theme::kCaptionPx), ImVec2(theme::snap(at.x), theme::snap(at.y)), theme::kNeutral600,
                        shown.c_str(), shown.c_str() + shown.size());
        }

        // The height of a `tiny` button (widgets::button's own arithmetic).
        float tinyButtonHeight() {
            return theme::snap(std::max(px(12), theme::textSize("Ag", 11, Weight::SemiBold).y) + 2 * px(5) + 2 * px(theme::kBorder));
        }

        std::string cellText(const LabelStats& s, int column) {
            switch (column) {
                case 0: return format("%04u", static_cast<unsigned>(s.id));
                case 1: return s.cls;
                case 2: return groupThousands(s.voxels);
                case 3: return format("%.2f", s.confidence);
                case 4: return s.flagText();
                default: return {};
            }
        }

    } // namespace

    void SegmentCleanupView::setLabels(std::shared_ptr<LabelVolume> labels) {
        labels_ = std::move(labels);
        orderedFor_ = nullptr;   // the rows are read again
        // The table follows the view state's label, as the Qt table does on
        // every refresh: that row alone is selected and brought into view.
        const std::uint32_t id = app_.wb().viewState().selectedLabel;
        seenSelected_ = primary_ = id;
        selected_.clear();
        if (id && labels_ && labels_->statsOf(id)) selected_.insert(id);
        anchorRow_ = -1;
        scrollToSelected_ = true;
    }

    void SegmentCleanupView::refreshOrder() {
        const std::vector<LabelStats>& st = labels_->stats();
        const Revisions& rev = app_.bridge().rev();
        // A sorted table re-sorts when a statistic may have moved: after an
        // edit, and after a change of time point (the statistics describe the
        // frame on screen).
        const std::uint64_t key = rev.labels + (sortColumn_ >= 0 ? rev.viewState : 0);
        if (orderedFor_ == labels_.get() && orderedSize_ == st.size() && orderedRev_ == key) return;
        orderedFor_ = labels_.get();
        orderedSize_ = st.size();
        orderedRev_ = key;
        order_.resize(st.size());
        for (std::size_t i = 0; i < order_.size(); ++i) order_[i] = static_cast<int>(i);
        if (sortColumn_ < 0) return;
        const int column = sortColumn_;
        const bool descending = sortDescending_;
        auto less = [&](int ia, int ib) {
            const LabelStats& a = st[static_cast<std::size_t>(ia)];
            const LabelStats& b = st[static_cast<std::size_t>(ib)];
            switch (column) {
                case 0: return a.id < b.id;
                case 1: return a.cls < b.cls;
                case 2: return a.voxels < b.voxels;
                case 3: return a.confidence < b.confidence;
                case 4: return a.flagText() < b.flagText();
                default: return false;
            }
        };
        std::stable_sort(order_.begin(), order_.end(), [&](int a, int b) { return descending ? less(b, a) : less(a, b); });
    }

    void SegmentCleanupView::clickRow(int displayRow, std::uint32_t id) {
        const ImGuiIO& io = ImGui::GetIO();
        const std::vector<LabelStats>& st = labels_->stats();
        const bool plain = !io.KeyCtrl && !(io.KeyShift && anchorRow_ >= 0);
        if (io.KeyCtrl) {
            if (!selected_.erase(id)) selected_.insert(id);
            anchorRow_ = displayRow;
        } else if (io.KeyShift && anchorRow_ >= 0) {
            selected_.clear();
            const int a = std::min(anchorRow_, displayRow), b = std::max(anchorRow_, displayRow);
            for (int r = a; r <= b && r < static_cast<int>(order_.size()); ++r) {
                const std::size_t i = static_cast<std::size_t>(order_[static_cast<std::size_t>(r)]);
                if (i < st.size()) selected_.insert(st[i].id);
            }
        } else {
            selected_ = {id};
            anchorRow_ = displayRow;
        }
        // The view state names one label: the row clicked alone, else the
        // first row selected (as the Qt table's selectedRows().first()) --
        // kept while it stays selected, so a multi-selection survives.
        if (plain) {
            primary_ = id;
        } else if (!selected_.count(primary_)) {
            primary_ = 0;
            for (int r : order_) {
                const std::size_t i = static_cast<std::size_t>(r);
                if (i < st.size() && selected_.count(st[i].id)) {
                    primary_ = st[i].id;
                    break;
                }
            }
        }
        if (!primary_) return;
        ViewState vs = app_.wb().viewState();
        seenSelected_ = primary_;
        if (vs.selectedLabel == primary_) return;
        vs.selectedLabel = primary_;
        app_.wb().setViewState(vs);
    }

    void SegmentCleanupView::act(const std::string& link, std::uint32_t id) {
        Workbench& wb = app_.wb();
        if (!wb.canEdit() || app_.bridge().taskRunning()) {
            wb.logLine("Label edits are refused while a run or load is in progress.");
            return;
        }
        if (link == "delete") {
            wb.deleteLabel(id);
        } else if (link == "merge") {
            std::vector<std::uint32_t> ids{id};
            for (std::uint32_t other : selected_)
                if (other != id) ids.push_back(other);
            if (ids.size() < 2 && wb.viewState().selectedLabel != 0 && wb.viewState().selectedLabel != id)
                ids.push_back(wb.viewState().selectedLabel);
            if (ids.size() >= 2) wb.mergeLabels(ids);
            else wb.logLine("Merge: select another label row first.");
        } else if (link == "split") {
            std::shared_ptr<LabelVolume> labels = wb.viewedLabels();
            const LabelStats* s = labels ? labels->statsOf(id) : nullptr;
            if (!s) return;
            // two seeds at a quarter and three quarters of the longest bbox axis
            const std::array<Index, 3> extent{s->bbox[1] - s->bbox[0], s->bbox[3] - s->bbox[2], s->bbox[5] - s->bbox[4]};
            const std::size_t ax = static_cast<std::size_t>(std::max_element(extent.begin(), extent.end()) - extent.begin());
            const std::array<Index, 3> centre{(s->bbox[0] + s->bbox[1]) / 2, (s->bbox[2] + s->bbox[3]) / 2, (s->bbox[4] + s->bbox[5]) / 2};
            std::array<Index, 3> a = centre, b = centre;
            a[ax] = s->bbox[2 * ax] + extent[ax] / 4;
            b[ax] = s->bbox[2 * ax] + (3 * extent[ax]) / 4;
            wb.splitLabel(id, a, b);
        }
    }

    // --- the tools -----------------------------------------------------------------

    void SegmentCleanupView::drawTools(float x0, float x1, float y0, float y1, bool editable) {
        (void)y1;
        Workbench& wb = app_.wb();
        const ViewState& vs = wb.viewState();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const float left = x0 + px(14), right = x1 - px(14);
        const float innerW = std::max(px(40), right - left);
        float y = y0 + px(10);
        drawCaption(dl, ImVec2(left, y), kToolsCaption, innerW);
        y += captionHeight() + px(8);

        // the 4 x 2 grid, 2 px apart, 34 px high
        const float gap = px(2);
        const float cellW = std::max(px(10), (innerW - 3.0f * gap) / 4.0f);
        const float cellH = px(34);
        const PaintTool current = vs.paintTool;
        for (std::size_t i = 0; i < kSegTools.size(); ++i) {
            const float cx = left + static_cast<float>(i % 4) * (cellW + gap);
            const float cy = y + static_cast<float>(i / 4) * (cellH + gap);
            ImGui::SetCursorScreenPos(ImVec2(theme::snap(cx), theme::snap(cy)));
            ImGui::PushID(static_cast<int>(i));
            widgets::GlyphOpts o;
            o.active = kSegTools[i].tool == current;
            o.enabled = editable;
            o.iconPx = 16;
            o.tooltip = kSegTools[i].name;
            if (widgets::glyphButton("##tool", kSegTools[i].icon, ImVec2(cellW / theme::scale(), 34), o)) {
                wb.setPaintTool(kSegTools[i].tool);
                wb.setTool(ViewerTool::Paint);
            }
            ImGui::PopID();
        }
        y += 2.0f * cellH + gap + px(8);

        // tool name and brush size
        const float lineH = theme::textSize("Ag", theme::kSmallPx).y;
        const std::string brush = format("%d px", vs.brushPx);
        const float brushW = theme::textSize(brush, theme::kSmallPx).x;
        widgets::drawTextIn(dl, ImVec2(left, y), ImVec2(right - brushW - px(8), y + lineH),
                            widgets::elideText(kSegTools[toolIndex(current)].name, std::max(px(10), innerW - brushW - px(8)), theme::kSmallPx),
                            theme::kSmallPx, theme::kNeutral600, Weight::Regular, 0.0f, 0.5f);
        widgets::drawTextIn(dl, ImVec2(left, y), ImVec2(right, y + lineH), brush, theme::kSmallPx, theme::kText, Weight::Regular, 1.0f, 0.5f);
        y += lineH + px(8);

        // brush size 2 - 60 px
        ImGui::SetCursorScreenPos(ImVec2(left, y));
        std::int64_t size = vs.brushPx;
        widgets::SliderOpts so;
        so.width = innerW / theme::scale();
        so.enabled = editable;
        if (widgets::sliderInt("##brush", &size, 2, 60, so)) {
            ViewState next = wb.viewState();
            next.brushPx = static_cast<int>(size);
            wb.setViewState(next);
        }
        y += px(18) + px(8);

        ImGui::SetCursorScreenPos(ImVec2(left, y));
        const std::string label = format("Paint in 3D (±%d z)###paint3d", std::max(1, static_cast<int>(std::lround(vs.brushPx / 6.0))));
        if (widgets::tokenCheck(label.c_str(), vs.paint3d, nullptr, editable)) {
            ViewState next = wb.viewState();
            next.paint3d = !next.paint3d;
            wb.setViewState(next);
        }
    }

    // --- the label table --------------------------------------------------------------

    void SegmentCleanupView::drawTable(float x0, float x1, float y0, float y1) {
        const cells::Rect r{ImVec2(x0, y0), ImVec2(x1, y1)};
        if (r.width() < 4.0f || r.height() < 4.0f) return;
        static const std::vector<LabelStats> none;
        const std::vector<LabelStats>& st = labels_ ? labels_->stats() : none;

        ImGui::SetCursorScreenPos(r.min);
        cells::pushTableStyle();
        const float headerH = cells::tableHeaderHeight();
        const float rowH = cells::tableRowHeight();
        const ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_PadOuterX |
                                      ImGuiTableFlags_Sortable | ImGuiTableFlags_SortTristate |
                                      ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoSavedSettings;
        bool drawn = false;
        if (ImGui::BeginTable("##labels", 6, flags, ImVec2(r.width(), r.height()))) {
            drawn = true;
            ImGui::TableSetupScrollFreeze(0, 1);
            // fixed widths: measuring every row would be the 20 s the Qt model avoided
            auto width = [&](const char* sample, float extra) {
                return std::max(px(10), theme::textSize(sample, theme::kSmallPx).x + px(extra) - 2.0f * px(6));
            };
            ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, width("00000", 46));
            ImGui::TableSetupColumn("CLASS", ImGuiTableColumnFlags_WidthFixed, width("nucleus", 28));
            ImGui::TableSetupColumn("VOXELS", ImGuiTableColumnFlags_WidthFixed, width("1 000 000", 20));
            ImGui::TableSetupColumn("CONF.", ImGuiTableColumnFlags_WidthFixed, width("0.00", 30));
            ImGui::TableSetupColumn("FLAG", ImGuiTableColumnFlags_WidthFixed, width("touching border", 32));
            // With a horizontal scroll a stretching column has no width of
            // its own: the links get theirs, and the table scrolls to them.
            ImGui::TableSetupColumn("##actions", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoSort,
                                    theme::textSize(kLinks, theme::kSmallPx).x + px(11) + px(4));
            cells::tableHeaders();
            if (ImGuiTableSortSpecs* specs = ImGui::TableGetSortSpecs()) {
                if (specs->SpecsDirty) {
                    sortColumn_ = specs->SpecsCount > 0 ? specs->Specs[0].ColumnIndex : -1;
                    sortDescending_ = specs->SpecsCount > 0 && specs->Specs[0].SortDirection == ImGuiSortDirection_Descending;
                    orderedFor_ = nullptr;
                    specs->SpecsDirty = false;
                }
            }
            if (labels_) refreshOrder();
            else order_.clear();

            if (scrollToSelected_) {
                scrollToSelected_ = false;
                if (primary_) {
                    for (std::size_t row = 0; row < order_.size(); ++row) {
                        const std::size_t i = static_cast<std::size_t>(order_[row]);
                        if (i >= st.size() || st[i].id != primary_) continue;
                        anchorRow_ = static_cast<int>(row);
                        const float scroll = ImGui::GetScrollY(), visible = ImGui::GetWindowHeight();
                        const float rowTop = headerH + static_cast<float>(row) * rowH;
                        if (rowTop - scroll < headerH) ImGui::SetScrollY(rowTop - headerH);
                        else if (rowTop + rowH - scroll > visible) ImGui::SetScrollY(rowTop + rowH - visible);
                        break;
                    }
                }
            }

            ImDrawList* dl = ImGui::GetWindowDrawList();
            const float wMerge = theme::textSize("merge", theme::kSmallPx).x;
            const float wSep = theme::textSize(" · ", theme::kSmallPx).x;
            const float wSplit = theme::textSize("split", theme::kSmallPx).x;
            const float wLinks = theme::textSize(kLinks, theme::kSmallPx).x;
            int clickedRow = -1;
            std::uint32_t clickedId = 0;
            std::string link;
            std::uint32_t linkId = 0;
            ImGuiListClipper clipper;
            clipper.Begin(static_cast<int>(order_.size()), rowH);
            while (clipper.Step()) {
                for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                    const std::size_t i = static_cast<std::size_t>(order_[static_cast<std::size_t>(row)]);
                    if (i >= st.size()) continue;
                    const LabelStats& s = st[i];
                    ImGui::TableNextRow(ImGuiTableRowFlags_None, rowH);
                    ImGui::PushID(static_cast<int>(s.id));
                    for (int c = 0; c < 6; ++c) {
                        if (!ImGui::TableSetColumnIndex(c)) continue;
                        if (c == 0) {
                            if (ImGui::Selectable("##row", selected_.count(s.id) > 0,
                                                  ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap)) {
                                clickedRow = row;
                                clickedId = s.id;
                            }
                            ImGui::SameLine(0.0f, 0.0f);
                            widgets::colorChip(theme::fromFloat(labelColor(s.id)), 10, 10);
                            ImGui::SameLine(0.0f, px(6));
                        }
                        if (c == 5) {
                            // "merge · split ·" and a bin, in accent: a click on a word (or the bin)
                            const ImVec2 p = ImGui::GetCursorScreenPos();
                            const float lh = ImGui::GetTextLineHeight();
                            auto hit = [&](const char* id, float x, float w) {
                                ImGui::SetCursorScreenPos(ImVec2(p.x + x, p.y));
                                const bool pressed = ImGui::InvisibleButton(id, ImVec2(std::max(1.0f, w), lh));
                                if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                                return pressed;
                            };
                            if (hit("##merge", 0.0f, wMerge)) {
                                link = "merge";
                                linkId = s.id;
                            }
                            if (hit("##split", wMerge + wSep, wSplit)) {
                                link = "split";
                                linkId = s.id;
                            }
                            // the bin alone, not the blank right of the row: a click there
                            // used to delete that row's label
                            if (hit("##delete", wLinks - px(2), px(11) + px(4))) {
                                link = "delete";
                                linkId = s.id;
                            }
                            ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(), ImGui::GetFontSize(), p, theme::kAccentText, kLinks);
                            drawIcon(dl, ImVec2(p.x + wLinks, p.y + (lh - px(11)) * 0.5f),
                                     ImVec2(p.x + wLinks + px(11), p.y + (lh + px(11)) * 0.5f), Icon::Trash, theme::kAccentText,
                                     std::max(1.0f, px(1.25f)));
                            continue;
                        }
                        const bool accent = (c == 3 && s.confidence < 0.6) || (c == 4 && !s.flags.empty());
                        const Weight w = c == 4 && !s.flags.empty() ? Weight::ExtraBold : Weight::Regular;
                        widgets::text(cellText(s, c), theme::kSmallPx, accent ? theme::kAccentText : theme::kText, w);
                    }
                    ImGui::PopID();
                }
            }
            ImGui::EndTable();
            if (!link.empty()) act(link, linkId);
            else if (clickedId) clickRow(clickedRow, clickedId);
        }
        cells::popTableStyle();
        if (drawn) cells::tableHeaderRule(r.min, r.width(), headerH);
    }

    // --- the review queue ---------------------------------------------------------

    void SegmentCleanupView::drawQueue(float x0, float x1, float y0, float y1, bool editable) {
        Workbench& wb = app_.wb();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const float left = x0 + px(14), right = x1 - px(14);
        float y = y0 + px(10);
        drawCaption(dl, ImVec2(left, y), "REVIEW QUEUE", std::max(px(10), right - left));
        y += captionHeight() + px(6);

        // Two rows of buttons: the three labels never fitted across 280 px,
        // and "Accept all reviewed" was the one that lost its ending.
        const float bh = tinyButtonHeight();
        const float buttonsTop = std::max(y, y1 - px(10) - (2.0f * bh + px(6)));

        std::vector<DiagnosticFact> facts;
        std::string trailer;
        if (!labels_ || labels_->stats().empty()) {
            facts.push_back({"Labels", "none yet"});
            trailer = "Run the segmentation step to fill the review queue.";
        } else {
            facts.push_back({"Low confidence (< 0.6)", std::to_string(labels_->flaggedCount("low conf"))});
            facts.push_back({"Touching border", std::to_string(labels_->flaggedCount("touching border"))});
            facts.push_back({"Size outliers", std::to_string(labels_->flaggedCount("small") + labels_->flaggedCount("merged?"))});
            facts.push_back({"Reviewed", std::to_string(labels_->reviewedCount()) + " / " + std::to_string(labels_->stats().size())});
        }
        // the facts lay out their own 14 px margins
        const cells::Rect fr{ImVec2(x0, y), ImVec2(x1, std::max(y, buttonsTop - px(6)))};
        if (fr.height() > 4.0f) cells::facts("##queue", fr, facts, {}, trailer);

        auto tiny = [&](const char* label, widgets::ButtonKind kind) {
            widgets::ButtonOpts o;
            o.kind = kind;
            o.tiny = true;
            o.enabled = editable;
            const bool pressed = widgets::button(label, o);
            if (!editable) cells::tooltipAlways(kFrozen);
            return pressed;
        };
        ImGui::SetCursorScreenPos(ImVec2(left, buttonsTop));
        if (tiny("Next flagged →", widgets::ButtonKind::Secondary)) wb.nextFlaggedLabel(true);
        ImGui::SameLine(0.0f, px(6));
        if (tiny("Undo", widgets::ButtonKind::Ghost)) wb.undo();
        ImGui::SetCursorScreenPos(ImVec2(left, buttonsTop + bh + px(6)));
        if (tiny("Accept all reviewed", widgets::ButtonKind::Ghost)) wb.acceptAllReviewed();
    }

    // --- the page ---------------------------------------------------------------------

    void SegmentCleanupView::draw() {
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        if (avail.x < 2.0f || avail.y < 2.0f) return;
        Workbench& wb = app_.wb();
        // The view state's label moved elsewhere (the viewer's Pick tool, the
        // assistant, "Next flagged"): that row alone, brought into view.
        const std::uint32_t vsLabel = wb.viewState().selectedLabel;
        if (vsLabel != seenSelected_) {
            seenSelected_ = primary_ = vsLabel;
            selected_.clear();
            if (vsLabel) selected_.insert(vsLabel);
            anchorRow_ = -1;
            scrollToSelected_ = true;
        }
        // Label edits are refused while a run is active, painting included:
        // the tools say so instead of doing nothing.
        const bool editable = wb.canEdit() && !app_.bridge().taskRunning();

        ImGui::GetWindowDrawList()->AddRectFilled(origin, ImVec2(origin.x + avail.x, origin.y + avail.y), theme::kDivider);
        const float gap = theme::crispPen(2);
        // 230 px was narrower than the caption, which then clipped to
        // "CLEANUP TOOLS · PAINT IN V": the cell takes whichever is wider.
        float toolsW = theme::snap(std::max(px(230), captionWidth(kToolsCaption) + px(30)));
        float queueW = theme::snap(px(280));
        // a narrow dock squeezes the side cells before the table disappears
        const float minTable = px(120);
        const float room = avail.x - 2.0f * gap;
        if (toolsW + queueW + minTable > room) {
            const float k = std::max(0.1f, (room - minTable) / (toolsW + queueW));
            toolsW = theme::snap(toolsW * k);
            queueW = theme::snap(queueW * k);
        }
        const float y0 = origin.y, y1 = origin.y + avail.y;
        const float tx0 = origin.x, tx1 = tx0 + toolsW;
        const float qx1 = origin.x + avail.x, qx0 = qx1 - queueW;
        const float mx0 = tx1 + gap, mx1 = qx0 - gap;

        cells::beginBox("##tools", ImVec2(tx0, y0), ImVec2(tx1, y1));
        drawTools(tx0, tx1, y0, y1, editable);
        cells::endCell();
        if (mx1 - mx0 >= 4.0f) {
            cells::beginBox("##table", ImVec2(mx0, y0), ImVec2(mx1, y1));
            drawTable(mx0, mx1, y0, y1);
            cells::endCell();
        }
        cells::beginBox("##queue", ImVec2(qx0, y0), ImVec2(qx1, y1));
        drawQueue(qx0, qx1, y0, y1, editable);
        cells::endCell();

        ImGui::SetCursorScreenPos(origin);
        ImGui::Dummy(avail);
    }

} // namespace sirius::app::gui
