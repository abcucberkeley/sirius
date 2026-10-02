#include "imgui/panels/ops_panel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>
#include <imgui_internal.h>

#include "core/help_pages.hpp"
#include "core/workbench.hpp"
#include "imgui/app.hpp"
#include "imgui/settings.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui {

    namespace {

        using theme::px;
        using theme::Weight;

        const char* const kFrozenSuffix = " — not while a run or load is in progress";
        const char* const kFrozen = "Not while a run or load is in progress — cancel it (Esc) or wait";

        // Display pixels as the design pixels the widgets take.
        float dp(float displayPx) { return displayPx / std::max(theme::scale(), 0.01f); }

        float lineHeight(float designPx, Weight w = Weight::Regular) { return theme::textSize("Ag", designPx, w).y; }

        float captionWidth(const std::string& s) {
            const std::string t = captionCase(s);
            const theme::FontScope f(theme::kCaptionPx, theme::captionFont());
            return ImGui::CalcTextSize(t.c_str(), t.c_str() + t.size()).x;
        }

        float captionHeight() {
            const theme::FontScope f(theme::kCaptionPx, theme::captionFont());
            return ImGui::CalcTextSize("AG").y;
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

        // The cursor, moved by hand; every caller submits an item afterwards.
        void place(float x, float y) { ImGui::SetCursorScreenPos(ImVec2(theme::snap(x), theme::snap(y))); }

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

        // The height widgets::button gives a full-size button.
        float buttonHeight() {
            return theme::snap(std::max(px(18), lineHeight(13, Weight::ExtraBold)) + 2 * px(7) + 2 * px(theme::kBorder));
        }

        // The cache glyph of the row and of the legend: M and D really are
        // letters in the design, Recompute is the circular arrow.
        struct CacheLook {
            std::string glyph;
            Icon icon;
            ImU32 color;
            std::string title;
        };

        CacheLook cacheLook(CachePolicy c) {
            switch (c) {
                case CachePolicy::Memory: return {"M", Icon::None, theme::kAccentText, "Cached in GPU/RAM"};
                case CachePolicy::Disk: return {"D", Icon::None, theme::kText, "Cached on disk (zarr scratch)"};
                case CachePolicy::Recompute: break;
            }
            return {std::string(), Icon::Recompute, theme::kNeutral500, "Recomputed on demand"};
        }

        // The 14 × 14 enable box as the rows and the legend paint it.
        void paintEnableBox(ImDrawList* dl, ImVec2 min, bool checked, bool enabled, bool hovered) {
            const float s = theme::snap(px(14));
            const ImVec2 max(min.x + s, min.y + s);
            if (checked) dl->AddRectFilled(min, max, enabled ? theme::kAccent : theme::kNeutral400);
            const ImU32 line = !enabled ? theme::kNeutral400 : (hovered ? theme::kAccent : theme::kNeutral700);
            widgets::crispRect(dl, min, max, line, theme::kBorder);
        }

        // One sentence about an operation, from the first paragraph of its help
        // page, with the markdown and inline maths stripped.
        std::string operationBlurb(const std::string& kind) {
            std::string intro;
            try {
                intro = loadHelpPage(kind).intro;
            } catch (const std::exception&) {
                return {};
            }
            std::string out;
            bool math = false;
            for (const char c : intro) {
                if (c == '$') {
                    math = !math;
                    continue;
                }
                if (math || c == '*' || c == '`' || c == '_' || c == '\n' || c == '\r') {
                    if (c == '\n' || c == '\r') out += ' ';
                    continue;
                }
                out += c;
            }
            // the first sentence, or a trimmed line
            const std::size_t stop = out.find(". ");
            if (stop != std::string::npos && stop > 30) out = out.substr(0, stop + 1);
            if (out.size() > 190) {
                // cut on a code point boundary
                std::size_t cut = 187;
                while (cut > 0 && (static_cast<unsigned char>(out[cut]) & 0xC0) == 0x80) --cut;
                out = out.substr(0, cut) + "…";
            }
            return simplified(out);
        }

        // What a row shows, derived from the workbench when something changed.
        struct RowData {
            std::string name;
            std::string kind;       // caption case
            std::string summary;
            bool ok = true;
            bool enabled = true;
            bool pinned = false;
            CachePolicy cache = CachePolicy::Recompute;
            // where its last output was computed: "node A100", "this computer · CPU"
            std::string placement;
            std::string placementTip;
            bool gone = false;
        };

        // One entry of the add menu.
        struct MenuItem {
            std::string kind;
            std::string name;
            std::string blurb;
            bool localOnly = false;
        };
        struct MenuGroup {
            std::string name;
            std::vector<MenuItem> items;
        };

    } // namespace

    struct OpsPanel::Impl {
        App& app;

        // rows
        std::uint64_t rowsStamp = 0;
        std::vector<RowData> rows;

        // the add menu
        bool openRequested = false;      // openAddMenu(): open and bring into view
        bool menuOpen = false;
        bool details = false;            // every item carries its one-line blurb (a setting)
        std::uint64_t menuStamp = 0;
        std::vector<MenuGroup> groups;
        // An edit picked in the menu, run once the frame's rows are drawn.
        std::function<void()> pending;

        explicit Impl(App& a) : app(a) { details = settings().getBool("ops/addMenuDetails", false); }

        void refreshRows() {
            const Revisions& r = app.bridge().rev();
            const std::uint64_t stamp = r.dataset + r.pipeline + r.step + r.operations + r.outputs + r.runState + r.backend;
            const Workbench& wb = app.wb();
            const Pipeline& p = wb.pipeline();
            if (stamp == rowsStamp && static_cast<int>(rows.size()) == p.size()) return;
            rowsStamp = stamp;
            rows.clear();
            for (int i = 0; i < p.size(); ++i) {
                const Step& step = p.at(i);
                RowData d;
                d.name = step.name;
                // The kind label keeps its natural width (it is a short, fixed
                // vocabulary); the name is what gives way in a narrow dock.
                d.kind = captionCase(step.op().info().kindLabel);
                d.summary = wb.stepSummary(i);
                const Validation v = wb.stepValidation(i);
                d.ok = v.ok();
                if (!d.ok) d.summary = v.firstError();
                else if (!step.enabled) d.summary += " — skipped";
                d.enabled = step.enabled;
                d.pinned = step.pinned;
                d.cache = step.cache;
                d.placement = wb.placementOf(i);
                if (!d.placement.empty())
                    if (const std::shared_ptr<const StepOutput> out = wb.output(i)) {
                        char secs[32];
                        std::snprintf(secs, sizeof secs, "%.1f s", out->seconds);
                        d.gone = !out->gone.empty();
                        d.placementTip = "Computed " + placementText(*out) + " in " + secs +
                                         (d.gone               ? ". Its result is gone (" + out->gone + "): run the step again."
                                          : out->where.empty() ? std::string(".")
                                                               : ". The result stays there: the viewer shows it at screen size, nothing else is downloaded.");
                    }
                rows.push_back(std::move(d));
            }
        }

        // Plugins register operations after start-up, and the HPC notes depend
        // on the backend: the list is rebuilt when either moved.
        void refreshMenu() {
            const Revisions& r = app.bridge().rev();
            const std::uint64_t stamp = r.operations + r.backend + (details ? 1u : 0u) * 0x100000000ull;
            if (stamp == menuStamp && !groups.empty()) return;
            menuStamp = stamp;
            groups.clear();
            const bool hpc = app.wb().backend() == Backend::Hpc;
            // SIRIUS's engine on the node runs every step there; a job of the
            // Python worker alone runs only what the worker implements
            const bool engine = app.wb().remoteConfig().hasEngine();
            for (const auto& [group, ops] : operationGroups()) {
                MenuGroup g;
                g.name = group;
                for (const Operation* op : ops) {
                    MenuItem item;
                    item.kind = op->kind();
                    item.name = op->info().name;
                    // Without SIRIUS's engine in the job only the operations the
                    // Python worker implements (OpInfo::remoteCapable) can run
                    // there; the rest are refused on HPC. Say so where the step is chosen.
                    item.localOnly = hpc && !engine && !op->info().remoteCapable;
                    if (details) {
                        item.blurb = operationBlurb(op->kind());
                        if (op->info().plugin && !op->info().source.empty())
                            item.blurb = (item.blurb.empty() ? std::string() : item.blurb + " · ") + "user operation · " +
                                         fileName(op->info().source);
                    }
                    g.items.push_back(std::move(item));
                }
                groups.push_back(std::move(g));
            }
        }

        // --- one step row: grid 22 | 1fr | auto -------------------------------------
        void drawRow(int index, const RowData& d, bool editable, std::function<void()>& action) {
            Workbench& wb = app.wb();
            const bool selected = wb.selectedIndex() == index;
            const bool viewed = wb.viewedIndex() == index;
            const int count = wb.pipeline().size();
            const float h13 = lineHeight(13, Weight::ExtraBold), h11 = lineHeight(11);
            const float rowH = px(9) + h13 + h11 + px(9);

            ImGui::PushID(index);
            widgets::RowOpts ro;
            ro.selected = selected;
            ro.edge = true;
            const widgets::Row row = widgets::beginRow("##step", dp(rowH), ro);
            if (row.clicked) action = [&wb, index] { wb.select(index); };
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const float width = row.max.x - row.min.x;
            const float midY = (row.min.y + row.max.y) * 0.5f;

            // right: reorder chevrons, view, remove. In a dock too narrow for
            // them and a name, the view button stays and the rest is left to
            // the Edit menu and its shortcuts.
            const float fullRight = px(14) + px(4) + px(20) + px(4) + px(20);
            const float fixed = px(10) + px(22) + px(10) + px(10) + px(14);   // margins, left cell, spacings
            const bool controls = !d.pinned && width - fixed - fullRight >= px(48);
            const float rightW = controls ? fullRight : px(20);
            const bool showView = width - fixed - rightW >= 0.0f;
            const float rightX = row.max.x - px(14) - rightW;

            // left: the enable box, or the pin of the Load step
            {
                const float cellX = row.min.x + px(10);
                if (d.pinned) {
                    const float s = px(12);
                    place(cellX + (px(22) - s) * 0.5f, midY - s * 0.5f);
                    const ImVec2 at = ImGui::GetCursorScreenPos();
                    ImGui::Dummy(ImVec2(s, s));
                    drawIcon(dl, ImVec2(at.x + s * 0.5f, at.y + s * 0.5f), s, Icon::Pin, theme::kNeutral500, px(1.25f));
                    tip("Always first, always enabled");
                } else {
                    const float s = theme::snap(px(14));
                    place(cellX + (px(22) - s) * 0.5f, midY - s * 0.5f);
                    const ImVec2 at = ImGui::GetCursorScreenPos();
                    ImGui::BeginDisabled(!editable);
                    const bool pressed = ImGui::InvisibleButton("##enable", ImVec2(s, s));
                    const bool hovered = ImGui::IsItemHovered();
                    ImGui::EndDisabled();
                    if (hovered && editable) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                    paintEnableBox(dl, at, d.enabled, editable, hovered && editable);
                    if (ImGui::IsItemFocused() && ImGui::GetIO().NavVisible)
                        widgets::crispRect(dl, at, ImVec2(at.x + s, at.y + s), theme::kAccent, 2.0f);
                    tip(widgets::withShortcut("Enable / skip this step", shortcutText(keys::enableStep)) +
                        (editable ? "" : kFrozenSuffix));
                    const bool on = !d.enabled;
                    if (pressed && editable) action = [&wb, index, on] { wb.setStepEnabled(index, on); };
                }
            }

            // middle: name + kind label, cache glyph + summary
            {
                const float opacity = d.enabled ? 1.0f : 0.45f;   // a skipped step is drawn faded
                const float bodyX = row.min.x + px(10) + px(22) + px(10);
                const float bodyW = (showView ? rightX - px(10) : row.max.x - px(14)) - bodyX;
                const float top = row.min.y + px(9);
                if (bodyW > px(12)) {
                    const float kindW = captionWidth(d.kind);
                    // the kind label goes before the name is cut to nothing
                    const bool showKind = !d.kind.empty() && bodyW - kindW - px(8) >= px(40);
                    const float nameW = showKind ? bodyW - kindW - px(8) : bodyW;
                    place(bodyX, top);
                    widgets::elided(d.name, nameW, 13, theme::withAlpha(theme::kText, opacity), Weight::ExtraBold);
                    if (showKind) {
                        // on the name's baseline
                        place(bodyX + bodyW - kindW, top + px((13.0f - theme::kCaptionPx) * 0.88f));
                        widgets::caption(d.kind, theme::withAlpha(theme::kNeutral600, opacity));
                    }
                    const float sumY = top + h13;
                    const CacheLook look = cacheLook(d.cache);
                    place(bodyX, sumY);
                    const ImVec2 cell = ImGui::GetCursorScreenPos();
                    ImGui::Dummy(ImVec2(px(12), h11));
                    const ImU32 cacheColor = theme::withAlpha(look.color, opacity);
                    if (look.icon == Icon::None)
                        widgets::drawTextIn(dl, cell, ImVec2(cell.x + px(12), cell.y + h11), look.glyph, 11, cacheColor,
                                            Weight::ExtraBold, 0.0f, 0.5f);
                    else
                        drawIcon(dl, ImVec2(cell.x + px(5.5f), cell.y + h11 * 0.5f), px(11), look.icon, cacheColor, px(1.25f));
                    tip(look.title);
                    float sumW = bodyW - px(12) - px(6);
                    // where it ran, at the right end of the summary line
                    const float tagW = d.placement.empty() ? 0.0f : theme::textSize(d.placement, 11).x;
                    if (tagW > 0.0f && sumW - tagW - px(8) >= px(40)) {
                        sumW -= tagW + px(8);
                        place(bodyX + bodyW - tagW, sumY);
                        widgets::text(d.placement, 11, theme::withAlpha(d.gone ? theme::kAccentText : theme::kNeutral500, opacity));
                        tip(d.placementTip);
                    }
                    if (sumW > px(12)) {
                        place(bodyX + px(12) + px(6), sumY);
                        const ImU32 base = d.ok ? theme::kNeutral600 : theme::kAccentText;
                        widgets::elided(d.summary, sumW, 11, d.enabled ? base : theme::withAlpha(theme::kNeutral600, opacity));
                    }
                }
            }

            // While a run is active the workbench refuses every pipeline
            // edit, so the row's controls say so instead of doing nothing.
            if (controls) {
                // The chevrons stack into one 14 px column: at the design's
                // 290 px dock every pixel of the row belongs to the name.
                widgets::GlyphOpts chevron;
                chevron.borderless = true;
                chevron.iconPx = 11;
                chevron.idle = theme::kNeutral500;
                chevron.enabled = editable && index > 1;
                place(rightX, midY - px(11));
                if (widgets::glyphButton("##up", Icon::ChevronUp, ImVec2(14, 11), chevron))
                    action = [&wb, index] { wb.moveStep(index, -1); };
                tip(widgets::withShortcut("Move up", shortcutText(keys::moveUp)) + (editable ? "" : kFrozenSuffix));
                chevron.enabled = editable && index < count - 1;
                place(rightX, midY);
                if (widgets::glyphButton("##down", Icon::ChevronDown, ImVec2(14, 11), chevron))
                    action = [&wb, index] { wb.moveStep(index, +1); };
                tip(widgets::withShortcut("Move down", shortcutText(keys::moveDown)) + (editable ? "" : kFrozenSuffix));
            }
            if (showView) {
                widgets::GlyphOpts view;
                view.active = viewed;
                view.iconPx = 14;
                view.idle = theme::kNeutral400;
                place(controls ? rightX + px(14) + px(4) : rightX, midY - px(10));
                if (widgets::glyphButton("##view", Icon::Eye, 20, view)) action = [&wb, index] { wb.view(index); };
                tip("Show this step's output in the viewer");
            }
            if (controls) {
                widgets::GlyphOpts remove;
                remove.iconPx = 14;
                remove.idle = theme::kNeutral400;
                remove.border = theme::kNeutral400;
                remove.enabled = editable;
                place(rightX + px(14) + px(4) + px(20) + px(4), midY - px(10));
                // The window handles removal (cache warning, undo hint). It runs
                // between frames, after any assistant call deferred ahead of it
                // that may move or remove steps, so the step goes by its id.
                App* a = &app;
                if (widgets::glyphButton("##remove", Icon::Trash, 20, remove)) {
                    const StepId id = wb.pipeline().at(index).id;
                    action = [a, id] {
                        a->defer([a, id] {
                            const int now = a->wb().pipeline().indexOf(id);
                            if (now >= 0) a->removeStepAt(now);
                        });
                    };
                }
                tip(widgets::withShortcut("Remove this step", shortcutText(keys::removeStep)) + (editable ? "" : kFrozenSuffix));
            }

            widgets::endRow(row);
            ImGui::PopID();
        }

        // --- the add row ---------------------------------------------------------------
        // Returns the row's rectangle, under which the menu opens.
        widgets::Row drawAddRow(int count, bool editable) {
            const float h13 = lineHeight(13, Weight::ExtraBold), h11 = lineHeight(11);
            const float rowH = px(12) + h13 + h11 + px(12);
            // Adding a step is an edit like any other: refused while a run
            // holds the pipeline, so the row (and the menu it opens) go with it.
            widgets::RowOpts ro;
            ro.selected = menuOpen;
            ro.hoverable = editable;
            ImGui::BeginDisabled(!editable);
            const widgets::Row row = widgets::beginRow("##add", dp(rowH), ro);
            ImGui::EndDisabled();
            if (!editable) tip(kFrozen);
            // the row toggles the menu under it
            if (row.clicked && editable) menuOpen = !menuOpen;

            ImDrawList* dl = ImGui::GetWindowDrawList();
            const float opacity = editable ? 1.0f : 0.45f;
            const float midY = (row.min.y + row.max.y) * 0.5f;
            // the dashed "+" square; the row is the control, this is its sign
            const float s = theme::snap(px(16));
            const ImVec2 g0(theme::snap(row.min.x + px(10) + (px(22) - s) * 0.5f), theme::snap(midY - s * 0.5f));
            const ImVec2 g1(g0.x + s, g0.y + s);
            widgets::dashedRect(dl, g0, g1, theme::withAlpha(row.hovered && editable ? theme::kAccent : theme::kDivider, opacity));
            drawIcon(dl, ImVec2((g0.x + g1.x) * 0.5f, (g0.y + g1.y) * 0.5f), px(10), Icon::Plus,
                     theme::withAlpha(theme::kNeutral600, opacity), px(1.25f));

            const float bodyX = row.min.x + px(10) + px(22) + px(10);
            const float bodyW = row.max.x - px(14) - bodyX;
            if (bodyW > px(12)) {
                const ImU32 ink = theme::withAlpha(theme::kNeutral600, opacity);
                const std::string hint = count < 2 ? std::string("Reconstruct, reduce, adjust, combine, segment…")
                                                   : "Runs after step " + Step::number(count - 1);
                widgets::drawText(dl, ImVec2(bodyX, row.min.y + px(12)),
                                  widgets::elideText("Add a processing step", bodyW, 13, Weight::ExtraBold), 13, ink,
                                  Weight::ExtraBold);
                widgets::drawText(dl, ImVec2(bodyX, row.min.y + px(12) + h13), widgets::elideText(hint, bodyW, 11), 11, ink);
            }
            widgets::endRow(row);
            return row;
        }

        // --- the add menu -----------------------------------------------------------------
        // An accent link of the menu: one line, or wrapped over several.
        bool menuLink(const char* id, const std::string& text, float x, float& y, float width, float padV, const std::string& tooltip) {
            const float textW = std::max(px(20), width - px(20));
            const float h = wrappedHeight(text, 11, textW) + 2 * padV;
            place(x, y);
            widgets::RowOpts ro;
            ro.topRule = 0.0f;
            ro.width = dp(width);
            const widgets::Row row = widgets::beginRow(id, dp(h), ro);
            tip(tooltip);
            place(row.min.x + px(10), row.min.y + padV);
            widgets::textWrapped(text, 11, theme::kAccentText, Weight::Regular, textW);
            widgets::endRow(row);
            y = row.max.y;
            return row.clicked;
        }

        void menuRule(ImDrawList* dl, float x, float& y, float width) {
            const float t = theme::crispPen(1);
            dl->AddRectFilled(ImVec2(x, theme::snap(y)), ImVec2(x + width, theme::snap(y) + t), theme::kDivider);
            y = theme::snap(y) + t;
        }

        // Grouped dropdown, inline under the add row: it
        // pushes the legend down and scrolls with the rows, so a long list
        // (descriptions on, many user operations) is never cut by the window.
        // 2 px ink border, the design's shadow-md, 8 px in from the dock.
        // Returns the bottom of the menu.
        float drawMenu(float contentX, float contentW) {
            const float border = theme::crispPen(2);
            const float x = theme::snap(contentX + px(8));
            const float w = theme::snap(std::max(px(120), contentW - px(16)));
            const float top = theme::snap(ImGui::GetCursorScreenPos().y);
            ImDrawList* dl = ImGui::GetWindowDrawList();
            // the content first (channel 1), then the ground under it (channel
            // 0), whose height is known only afterwards
            dl->ChannelsSplit(2);
            dl->ChannelsSetCurrent(1);
            const float bottom = theme::snap(drawMenuContent(x + border, top + border, w - 2 * border) + border);
            dl->ChannelsSetCurrent(0);
            drawMenuShadow(dl, ImVec2(x, top), ImVec2(x + w, bottom));
            dl->AddRectFilled(ImVec2(x, top), ImVec2(x + w, bottom), theme::kBg);
            dl->ChannelsMerge();
            widgets::crispRect(dl, ImVec2(x, top), ImVec2(x + w, bottom), theme::kText, 2.0f);
            place(contentX, bottom);
            ImGui::Dummy(ImVec2(0, 0));
            return bottom;
        }

        // shadow-md of the design (0 3px 10px, 16 %): the menu floats.
        static void drawMenuShadow(ImDrawList* dl, ImVec2 a, ImVec2 b) {
            const float drop = theme::snap(px(3));
            const int layers = std::max(4, static_cast<int>(std::lround(px(8))));
            for (int i = 0; i < layers; ++i) {
                const float f = 1.0f - static_cast<float>(i) / static_cast<float>(layers);
                const ImU32 c = theme::withAlpha(theme::kNeutral900, 0.09f * f * f);
                const float d = static_cast<float>(i);
                const float l = a.x - d - 1.0f, r = b.x + d, t = a.y + drop, bt = b.y + drop + d;
                dl->AddRectFilled(ImVec2(l, t), ImVec2(l + 1.0f, bt + 1.0f), c);           // left
                dl->AddRectFilled(ImVec2(r, t), ImVec2(r + 1.0f, bt + 1.0f), c);           // right
                dl->AddRectFilled(ImVec2(l + 1.0f, bt), ImVec2(r, bt + 1.0f), c);          // bottom
            }
        }

        // Draws the menu's rows from (x0, y0), `width` wide; returns their bottom.
        float drawMenuContent(float x0, float y0, float width) {
            refreshMenu();
            Workbench& wb = app.wb();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const float h10 = captionHeight(), h11 = lineHeight(11), h12 = lineHeight(12, Weight::ExtraBold);
            float y = y0;
            App* a = &app;

            // header: what this is, and the descriptions toggle
            {
                const std::string label = details ? "Hide descriptions" : "Show descriptions";
                const float toggleW = theme::textSize(label, 11).x + px(12);
                const float toggleH = h11 + px(4);
                const float rowH = px(6) + std::max(h10, toggleH) + px(6);
                const float room = width - px(10) - px(10) - toggleW - px(8);
                if (room > px(24)) {
                    place(x0 + px(10), y + (rowH - h10) * 0.5f);
                    widgets::caption(fitCaption("Add a step", room));
                }
                place(x0 + width - px(10) - toggleW, y + (rowH - toggleH) * 0.5f);
                widgets::RowOpts ro;
                ro.topRule = 0.0f;
                ro.width = dp(toggleW);
                const widgets::Row toggle = widgets::beginRow("##details", dp(toggleH), ro);
                tip("Show a sentence about every operation (kept in the settings)");
                widgets::drawTextIn(dl, toggle.min, toggle.max, label, 11, theme::kAccentText);
                widgets::endRow(toggle);
                if (toggle.clicked) {
                    details = !details;
                    settings().set("ops/addMenuDetails", details);
                    app.requestRedraw();
                }
                y += rowH;
                menuRule(dl, x0, y, width);
            }

            // 72 rather than the design's 82: the dock is 290 px wide and
            // "Volume reconstruction" has to fit beside the caption.
            float captionCol = px(72);
            for (const MenuGroup& g : groups) captionCol = std::max(captionCol, px(10) + captionWidth(g.name) + px(8));
            captionCol = std::min(captionCol, width * 0.45f);
            const float itemsX = x0 + captionCol;
            const float itemsW = width - captionCol;
            const float textW = std::max(px(20), itemsW - px(20));
            const float padV = px(details ? 7.0f : 6.0f);

            const auto manageLink = [&](const char* id, float x, float w) {
                if (menuLink(id, "Manage user operations…", x, y, w, px(6),
                             "Browse, edit and create the Python files that define user operations")) {
                    menuOpen = false;
                    a->defer([a] { a->pluginManager(); });
                }
            };

            bool linked = false;
            std::string picked;
            for (std::size_t gi = 0; gi < groups.size(); ++gi) {
                const MenuGroup& g = groups[gi];
                ImGui::PushID(static_cast<int>(gi));
                const float groupTop = y;
                place(x0 + px(10), groupTop + px(8));
                widgets::caption(fitCaption(g.name, captionCol - px(10) - px(4)));
                for (std::size_t k = 0; k < g.items.size(); ++k) {
                    const MenuItem& item = g.items[k];
                    ImGui::PushID(static_cast<int>(k));
                    float h = padV + h12 + padV;
                    if (item.localOnly) h += px(2) + h11;
                    float blurbH = 0.0f;
                    if (!item.blurb.empty()) {
                        blurbH = wrappedHeight(item.blurb, 11, textW);
                        h += px(2) + blurbH;
                    }
                    place(itemsX, y);
                    widgets::RowOpts ro;
                    ro.topRule = 0.0f;
                    ro.width = dp(itemsW);
                    const widgets::Row row = widgets::beginRow("##item", dp(h), ro);
                    if (row.clicked) picked = item.kind;
                    float ty = row.min.y + padV;
                    place(row.min.x + px(10), ty);
                    widgets::elided(item.name, textW, 12, theme::kText, Weight::ExtraBold);
                    ty += h12;
                    if (item.localOnly) {
                        ty += px(2);
                        place(row.min.x + px(10), ty);
                        widgets::text(widgets::elideText("needs the SIRIUS engine on HPC", textW, 11), 11, theme::kNeutral600);
                        tip(item.name + " runs on the cluster only with SIRIUS's engine in the job; this job runs the Python worker alone, so "
                                        "it is refused on HPC. Choose CPU/CUDA to run it here, or reconnect with an engine image.");
                        ty += h11;
                    }
                    if (!item.blurb.empty()) {
                        ty += px(2);
                        place(row.min.x + px(10), ty);
                        widgets::textWrapped(item.blurb, 11, theme::kNeutral600, Weight::Regular, textW);
                    }
                    widgets::endRow(row);
                    y = row.max.y;
                    ImGui::PopID();
                }
                // "Manage user operations…" sits in the User group, or under
                // the last group while there are no user operations yet.
                if (g.name == "User") {
                    manageLink("##manage", itemsX, itemsW);
                    linked = true;
                }
                y = std::max(y, groupTop + px(8) + h10 + px(8));
                menuRule(dl, x0, y, width);
                ImGui::PopID();
            }
            if (!linked) {
                manageLink("##manage", x0, width);
                menuRule(dl, x0, y, width);
            }
            if (menuLink("##example", "Load example pipeline (SIM → einsum → contrast → merge → segment → volume)", x0, y, width,
                         px(8), {})) {
                // after the frame: the rows above were drawn from the old pipeline
                pending = [&wb] { wb.loadExamplePipeline(); };
                menuOpen = false;
            }
            if (!picked.empty()) {
                pending = [&wb, picked] { wb.addStep(picked); };
                menuOpen = false;
            }
            return y;
        }

        // --- the legend -----------------------------------------------------------------------
        void drawLegend(float x0, float width) {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const float iconCol = px(36);
            const float textX = x0 + px(14) + iconCol + px(8);
            const float textW = x0 + width - px(14) - textX;
            if (textW < px(40)) return;   // a dock this narrow has no room to explain itself
            const float h12 = lineHeight(12);
            float y = ImGui::GetCursorScreenPos().y + px(12);
            const float cx = x0 + px(14) + iconCol * 0.5f;

            const auto entry = [&](float iconH, const std::string& text, const std::function<void(float top)>& icon) {
                const float textH = wrappedHeight(text, 12, textW);
                icon(y + std::max(0.0f, (h12 - iconH) * 0.5f));
                place(textX, y + std::max(0.0f, (iconH - h12) * 0.5f));
                widgets::textWrapped(text, 12, theme::kNeutral700, Weight::Regular, textW);
                y += std::max(iconH, textH + std::max(0.0f, (iconH - h12) * 0.5f)) + px(8);
            };

            const float box = theme::snap(px(14));
            entry(box, "Enabled — runs and passes its output on", [&](float top) {
                paintEnableBox(dl, ImVec2(theme::snap(cx - box * 0.5f), theme::snap(top)), true, true, false);
            });
            entry(box, "Skipped — data passes through unchanged", [&](float top) {
                paintEnableBox(dl, ImVec2(theme::snap(cx - box * 0.5f), theme::snap(top)), false, true, false);
            });
            const float eye = theme::snap(px(20));
            entry(eye, "Shown in the viewer (click any step's eye)", [&](float top) {
                const ImVec2 a(theme::snap(cx - eye * 0.5f), theme::snap(top));
                const ImVec2 b(a.x + eye, a.y + eye);
                dl->AddRectFilled(a, b, theme::kAccent);
                drawIcon(dl, ImVec2((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f), px(14), Icon::Eye, theme::kBg, px(1.5f));
            });
            entry(px(22), "Reorder — steps run top to bottom", [&](float top) {
                drawIcon(dl, ImVec2(cx, top + px(5.5f)), px(11), Icon::ChevronUp, theme::kNeutral500, px(1.25f));
                drawIcon(dl, ImVec2(cx, top + px(16.5f)), px(11), Icon::ChevronDown, theme::kNeutral500, px(1.25f));
            });
            entry(h12, "Output cached in memory · on disk · recomputed", [&](float top) {
                const float mW = theme::textSize("M", 11, Weight::ExtraBold).x, dW = theme::textSize("D", 11, Weight::ExtraBold).x;
                const float total = mW + px(3) + dW + px(3) + px(11);
                float x = cx - total * 0.5f;
                const float h11 = lineHeight(11, Weight::ExtraBold);
                const float ty = top + (h12 - h11) * 0.5f;
                widgets::drawText(dl, ImVec2(x, ty), "M", 11, theme::kAccentText, Weight::ExtraBold);
                x += mW + px(3);
                widgets::drawText(dl, ImVec2(x, ty), "D", 11, theme::kText, Weight::ExtraBold);
                x += dW + px(3);
                drawIcon(dl, ImVec2(x + px(5.5f), top + h12 * 0.5f), px(11), Icon::Recompute, theme::kNeutral500, px(1.25f));
            });
            place(x0, y - px(8) + px(12));
            ImGui::Dummy(ImVec2(width, 0.0f));
        }

        // --- the panel ---------------------------------------------------------------------------
        void draw() {
            Workbench& wb = app.wb();
            Bridge& bridge = app.bridge();
            refreshRows();
            const int count = static_cast<int>(rows.size());
            const bool running = bridge.running();
            const bool busy = running || bridge.taskRunning();
            const bool editable = wb.canEdit() && !bridge.taskRunning();

            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
            const ImVec2 origin = ImGui::GetCursorScreenPos();
            const ImVec2 avail = ImGui::GetContentRegionAvail();
            ImDrawList* dl = ImGui::GetWindowDrawList();

            // header
            float headerH = 0.0f;
            {
                const std::string steps = format("%d %s", count, count == 1 ? "step" : "steps");
                const float h10 = captionHeight(), h11 = lineHeight(11);
                const float h = std::max(h10, h11);
                const float stepsW = theme::textSize(steps, 11).x;
                const float y = origin.y + px(12);
                // the count gives way before the caption does
                const bool showCount = avail.x - px(28) - stepsW - px(8) >= px(60);
                const float room = avail.x - px(28) - (showCount ? stepsW + px(8) : 0.0f);
                if (room > px(12)) {
                    place(origin.x + px(14), y + (h - h10) * 0.5f);
                    widgets::caption(fitCaption("Operations · any order", room));
                }
                if (showCount) {
                    place(origin.x + avail.x - px(14) - stepsW, y + (h - h11) * 0.5f);
                    widgets::text(steps, 11, theme::kNeutral600);
                }
                headerH = theme::snap(px(12) + h + px(8));
                place(origin.x, origin.y + headerH);
                ImGui::Dummy(ImVec2(0, 0));
            }

            // footer metrics: it keeps its place at the bottom of the dock
            const float rule = theme::crispPen(theme::kRule);
            const float btnH = buttonHeight();
            const float footerH = rule + px(12) + btnH + px(8) + btnH + px(12);
            const float listH = std::max(px(40), avail.y - headerH - footerH);

            // scrolling rows + add + legend
            place(origin.x, origin.y + headerH);
            if (ImGui::BeginChild("##steps", ImVec2(avail.x, listH), ImGuiChildFlags_None, ImGuiWindowFlags_None)) {
                std::function<void()> action;
                for (int i = 0; i < count; ++i) drawRow(i, rows[static_cast<std::size_t>(i)], editable, action);

                const float contentX = ImGui::GetCursorScreenPos().x;
                const float contentW = std::max(1.0f, ImGui::GetContentRegionAvail().x);
                const widgets::Row addRow = drawAddRow(count, editable);

                // The menu belongs to the add row: refused with it while a run
                // holds the pipeline.
                const bool reveal = openRequested && editable;
                if (reveal) menuOpen = true;
                openRequested = false;
                if (!editable) menuOpen = false;
                float below = addRow.max.y;
                if (menuOpen) {
                    const float bottom = drawMenu(contentX, contentW);
                    // Process ▸ Add operation… brings the whole menu into view
                    if (reveal) {
                        ImGui::ScrollToRect(ImGui::GetCurrentWindow(), ImRect(addRow.min, ImVec2(addRow.max.x, bottom + px(14))));
                        app.requestRedraw();
                    }
                    below = bottom;
                }

                // legend, 14 px under the menu's room
                place(contentX, below + px(14));
                widgets::rule(theme::kRule);
                drawLegend(contentX, contentW);

                // after the rows: an edit changes what the loop was walking
                if (action) action();
                if (pending) {
                    pending();
                    pending = nullptr;
                }
            }
            ImGui::EndChild();

            // footer
            {
                const float y = origin.y + avail.y - footerH;
                dl->AddRectFilled(ImVec2(origin.x, theme::snap(y)), ImVec2(origin.x + avail.x, theme::snap(y) + rule), theme::kDivider);
                const float width = std::max(px(40), avail.x - px(28));
                App* a = &app;

                place(origin.x + px(14), y + rule + px(12));
                widgets::ButtonOpts run;
                run.kind = widgets::ButtonKind::Primary;
                run.width = dp(width);
                run.enabled = !busy && wb.hasDataset();
                const std::string label = running ? format("Running · %d %%", static_cast<int>(bridge.runFraction() * 100.0 + 0.5))
                                                  : std::string("Run all enabled");
                if (widgets::button((label + "##runAll").c_str(), run)) app.runAll();
                tip(widgets::withShortcut("Run every enabled step top to bottom", shortcutText(keys::runAll)));

                place(origin.x + px(14), y + rule + px(12) + btnH + px(8));
                widgets::ButtonOpts exp;
                exp.width = dp(width);
                exp.enabled = wb.hasDataset() && !busy;
                if (widgets::button("Export result…##export", exp)) a->defer([a] { a->exportResultDialog(); });

                place(origin.x, origin.y + avail.y);
                ImGui::Dummy(ImVec2(0, 0));
            }
            ImGui::PopStyleVar();
        }
    };

    OpsPanel::OpsPanel(App& app) : impl_(std::make_unique<Impl>(app)) {}
    OpsPanel::~OpsPanel() = default;

    void OpsPanel::draw() { impl_->draw(); }

    void OpsPanel::openAddMenu() { impl_->openRequested = true; }

} // namespace sirius::app::gui
