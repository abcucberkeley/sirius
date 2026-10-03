#include "imgui/widgets/controls.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <imgui_internal.h>
#include <imgui_stdlib.h>

#include "imgui/strings.hpp"

namespace sirius::app::gui::widgets {

    using theme::px;
    using theme::Weight;

    namespace {

        // Rounded the way Dear ImGui rounds the size it draws PushFont text at
        // (UpdateCurrentFontSize), so text drawn here matches theme::textSize.
        float fontPx(float designPx) {
            const ImGuiStyle& st = ImGui::GetStyle();
            return std::max(1.0f, ImGui::GetRoundedFontSize(designPx * st.FontScaleMain * st.FontScaleDpi));
        }

        ImU32 dim(ImU32 c, bool enabled) { return enabled ? c : theme::withAlpha(c, 0.45f); }

        // The part of a label that is shown ("Run##step" -> "Run").
        const char* labelEnd(const char* label) { return ImGui::FindRenderedTextEnd(label); }

        float lineWidth(float designWidth) {
            return designWidth < 0.0f ? std::max(1.0f, ImGui::GetContentRegionAvail().x)
                                      : (designWidth == 0.0f ? 0.0f : px(designWidth));
        }

        void focusRing(ImDrawList* dl, ImVec2 min, ImVec2 max) {
            if (ImGui::IsItemFocused() && ImGui::GetIO().NavVisible) crispRect(dl, min, max, theme::kAccent, 2.0f);
        }

        struct CardState {
            ImVec2 min;
            float width;
            float padding;
            bool accent;
        };
        std::vector<CardState>& cards() {
            static std::vector<CardState> stack;
            return stack;
        }

    } // namespace

    // --- text ----------------------------------------------------------------

    void text(const std::string& s, float pxSize, ImU32 color, Weight w) {
        const theme::FontScope f(pxSize, w);
        ImGui::PushStyleColor(ImGuiCol_Text, color);
        ImGui::TextUnformatted(s.c_str(), s.c_str() + s.size());
        ImGui::PopStyleColor();
    }

    void textWrapped(const std::string& s, float pxSize, ImU32 color, Weight w, float wrapWidth) {
        const theme::FontScope f(pxSize, w);
        ImGui::PushStyleColor(ImGuiCol_Text, color);
        ImGui::PushTextWrapPos(wrapWidth > 0.0f ? ImGui::GetCursorPosX() + wrapWidth : 0.0f);
        ImGui::TextUnformatted(s.c_str(), s.c_str() + s.size());
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
    }

    void heading(const std::string& s, float pxSize) { text(s, pxSize, theme::kText, Weight::ExtraBold); }

    void caption(const std::string& s, ImU32 color) {
        const std::string t = captionCase(s);
        const theme::FontScope f(theme::kCaptionPx, theme::captionFont());
        ImGui::PushStyleColor(ImGuiCol_Text, color);
        ImGui::TextUnformatted(t.c_str(), t.c_str() + t.size());
        ImGui::PopStyleColor();
    }

    void mono(const std::string& s, float pxSize, ImU32 color) {
        const theme::FontScope f(pxSize, theme::mono());
        ImGui::PushStyleColor(ImGuiCol_Text, color);
        ImGui::TextUnformatted(s.c_str(), s.c_str() + s.size());
        ImGui::PopStyleColor();
    }

    std::string elideText(const std::string& s, float width, float pxSize, Weight w) {
        if (theme::textSize(s, pxSize, w).x <= width) return s;
        static const std::string dots = "\xE2\x80\xA6";   // …
        const float room = width - theme::textSize(dots, pxSize, w).x;
        if (room <= 0.0f) return dots;
        // longest prefix (on code point boundaries) that fits
        std::vector<std::size_t> ends;
        for (std::size_t i = 0; i < s.size();) {
            nextCodepoint(s, i);
            ends.push_back(i);
        }
        std::size_t lo = 0, hi = ends.size();
        while (lo < hi) {
            const std::size_t mid = (lo + hi + 1) / 2;
            if (theme::textSize(s.c_str(), pxSize, w, s.c_str() + ends[mid - 1]).x <= room) lo = mid;
            else hi = mid - 1;
        }
        std::string out = lo == 0 ? std::string() : s.substr(0, ends[lo - 1]);
        while (!out.empty() && out.back() == ' ') out.pop_back();
        return out + dots;
    }

    void elided(const std::string& s, float width, float pxSize, ImU32 color, Weight w) {
        if (width <= 0.0f) width = ImGui::GetContentRegionAvail().x;
        const std::string shown = elideText(s, width, pxSize, w);
        text(shown, pxSize, color, w);
        if (shown != s) tooltip(s);
    }

    void drawText(ImDrawList* dl, ImVec2 pos, const std::string& s, float pxSize, ImU32 color, Weight w) {
        ImFont* f = theme::font(w);
        if (!f) f = ImGui::GetFont();
        dl->AddText(f, fontPx(pxSize), ImVec2(theme::snap(pos.x), theme::snap(pos.y)), color, s.c_str(), s.c_str() + s.size());
    }

    void drawTextIn(ImDrawList* dl, ImVec2 min, ImVec2 max, const std::string& s, float pxSize, ImU32 color, Weight w,
                    float ax, float ay) {
        const ImVec2 size = theme::textSize(s, pxSize, w);
        const ImVec2 pos(min.x + (max.x - min.x - size.x) * ax, min.y + (max.y - min.y - size.y) * ay);
        drawText(dl, pos, s, pxSize, color, w);
    }

    // --- layout ---------------------------------------------------------------

    void rule(float pxSize, ImU32 color) {
        const float t = theme::crispPen(pxSize);
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float w = std::max(1.0f, ImGui::GetContentRegionAvail().x);
        ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(p.x, theme::snap(p.y)), ImVec2(p.x + w, theme::snap(p.y) + t), color);
        ImGui::Dummy(ImVec2(w, t));
    }

    void ruleAt(ImDrawList* dl, ImVec2 a, ImVec2 b, float pxSize, ImU32 color) {
        const float t = theme::crispPen(pxSize);
        if (std::abs(a.y - b.y) < 0.5f) dl->AddRectFilled(ImVec2(a.x, theme::snap(a.y)), ImVec2(b.x, theme::snap(a.y) + t), color);
        else dl->AddRectFilled(ImVec2(theme::snap(a.x), a.y), ImVec2(theme::snap(a.x) + t, b.y), color);
    }

    void vspace(float designPx) { ImGui::Dummy(ImVec2(0.0f, px(designPx))); }

    void colorChip(ImU32 color, float w, float h) {
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const ImVec2 size = px(w, h);
        // centred on the text line it shares
        const float lineH = ImGui::GetTextLineHeight();
        const float dy = size.y < lineH ? std::floor((lineH - size.y) * 0.5f) : 0.0f;
        ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(p.x, p.y + dy), ImVec2(p.x + size.x, p.y + dy + size.y), color);
        ImGui::Dummy(ImVec2(size.x, std::max(size.y, dy + size.y)));
    }

    void tooltip(const std::string& s) {
        if (s.empty() || !ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) return;
        ImGui::PushStyleColor(ImGuiCol_Border, theme::kText);
        ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, theme::crispPen(theme::kBorder));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, px(8, 4));
        if (ImGui::BeginTooltip()) {
            {   // popped inside the tooltip window it was pushed in
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

    std::string withShortcut(const std::string& t, const std::string& shortcut) {
        return shortcut.empty() ? t : t + " (" + shortcut + ")";
    }

    void crispRect(ImDrawList* dl, ImVec2 min, ImVec2 max, ImU32 color, float designPx) {
        const float t = theme::crispPen(designPx);
        const ImVec2 a(theme::snap(min.x), theme::snap(min.y)), b(theme::snap(max.x), theme::snap(max.y));
        if (b.x - a.x < 2 * t || b.y - a.y < 2 * t) {
            dl->AddRectFilled(a, b, color);
            return;
        }
        dl->AddRectFilled(a, ImVec2(b.x, a.y + t), color);                       // top
        dl->AddRectFilled(ImVec2(a.x, b.y - t), b, color);                       // bottom
        dl->AddRectFilled(ImVec2(a.x, a.y + t), ImVec2(a.x + t, b.y - t), color);   // left
        dl->AddRectFilled(ImVec2(b.x - t, a.y + t), ImVec2(b.x, b.y - t), color);   // right
    }

    void dashedRect(ImDrawList* dl, ImVec2 min, ImVec2 max, ImU32 color, float thickness, float dash) {
        const float t = theme::crispPen(thickness), d = px(dash);
        const ImVec2 a(theme::snap(min.x), theme::snap(min.y)), b(theme::snap(max.x), theme::snap(max.y));
        for (float x = a.x; x < b.x; x += 2 * d) {
            const float x1 = std::min(x + d, b.x);
            dl->AddRectFilled(ImVec2(x, a.y), ImVec2(x1, a.y + t), color);
            dl->AddRectFilled(ImVec2(x, b.y - t), ImVec2(x1, b.y), color);
        }
        for (float y = a.y; y < b.y; y += 2 * d) {
            const float y1 = std::min(y + d, b.y);
            dl->AddRectFilled(ImVec2(a.x, y), ImVec2(a.x + t, y1), color);
            dl->AddRectFilled(ImVec2(b.x - t, y), ImVec2(b.x, y1), color);
        }
    }

    // --- buttons -------------------------------------------------------------

    bool button(const char* label, const ButtonOpts& o) {
        float fontSize = 13, padX = 12, padY = 7, minH = 18;
        Weight weight = Weight::SemiBold;
        switch (o.kind) {
            case ButtonKind::Primary: weight = Weight::ExtraBold; break;
            case ButtonKind::Ghost: padX = 8; break;
            case ButtonKind::Link:
                fontSize = 11;
                padX = padY = 0;
                minH = 0;
                weight = Weight::Regular;
                break;
            case ButtonKind::Chip:
                fontSize = 11;
                padX = 8;
                padY = 3;
                minH = 0;
                weight = Weight::Regular;
                break;
            case ButtonKind::Secondary: break;
        }
        if (o.kind != ButtonKind::Link && o.kind != ButtonKind::Chip) {
            if (o.small) {
                fontSize = 12;
                padX = o.kind == ButtonKind::Ghost ? 8.0f : 10.0f;
                padY = 4;
                minH = 14;
            } else if (o.tiny) {
                fontSize = 11;
                padX = 9;
                padY = 5;
                minH = 12;
            }
        }
        const float border = o.kind == ButtonKind::Link ? 0.0f : theme::kBorder;
        const char* end = labelEnd(label);
        const ImVec2 ts = theme::textSize(label, fontSize, weight, end);
        const float contentH = std::max(px(minH), ts.y);
        ImVec2 size(ts.x + 2 * px(padX) + 2 * px(border), contentH + 2 * px(padY) + 2 * px(border));
        const float want = lineWidth(o.width);
        if (want > 0.0f) size.x = want;
        size.x = theme::snap(size.x);
        size.y = theme::snap(size.y);

        ImGui::BeginDisabled(!o.enabled);
        const ImVec2 min = ImGui::GetCursorScreenPos();
        const bool pressed = ImGui::InvisibleButton(label, size);
        const bool hovered = ImGui::IsItemHovered(), held = ImGui::IsItemActive();
        if (hovered && o.enabled) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        ImGui::EndDisabled();
        const ImVec2 max(min.x + size.x, min.y + size.y);
        ImDrawList* dl = ImGui::GetWindowDrawList();

        ImU32 fill = theme::kTransparent, line = theme::kText, fg = theme::kText;
        switch (o.kind) {
            case ButtonKind::Primary:
                fill = held ? theme::kAccent700 : (hovered ? theme::kAccent600 : theme::kAccent);
                line = fill;
                fg = theme::kBg;
                if (!o.enabled) {
                    fill = line = theme::kNeutral300;
                    fg = theme::kBg;
                }
                break;
            case ButtonKind::Secondary:
                if (held) fill = theme::kNeutral200;
                if (hovered || held) line = fg = theme::kAccent;
                if (!o.enabled) {
                    line = theme::kNeutral300;
                    fg = theme::kNeutral500;
                }
                break;
            case ButtonKind::Ghost:
                line = theme::kTransparent;
                if (held) fill = theme::kNeutral200;
                if (hovered || held) fg = theme::kAccent;
                if (!o.enabled) fg = theme::kNeutral500;
                break;
            case ButtonKind::Link:
                line = theme::kTransparent;
                fg = hovered ? theme::kAccent700 : theme::kAccentText;
                if (!o.enabled) fg = theme::kNeutral500;
                break;
            case ButtonKind::Chip:
                line = hovered ? theme::kAccent : theme::kDivider;
                fg = hovered ? theme::kAccent : theme::kText;
                if (!o.enabled) {
                    line = theme::kNeutral300;
                    fg = theme::kNeutral500;
                }
                break;
        }
        if (fill & 0xFF000000u) dl->AddRectFilled(min, max, fill);
        if ((line & 0xFF000000u) && border > 0.0f) crispRect(dl, min, max, line, theme::kBorder);
        focusRing(dl, min, max);
        const std::string shown(label, end);
        // the label may be wider than a button of a given width
        const float room = size.x - 2 * px(padX) - 2 * px(border);
        const std::string fitted = ts.x > room + 0.5f && room > 0 ? elideText(shown, room, fontSize, weight) : shown;
        const float ax = o.centered ? 0.5f : 0.0f;
        drawTextIn(dl, ImVec2(min.x + px(padX) + px(border), min.y), ImVec2(max.x - px(padX) - px(border), max.y), fitted, fontSize,
                   fg, weight, ax, 0.5f);
        if (!o.tooltip.empty()) tooltip(o.tooltip);
        else if (fitted != shown) tooltip(shown);
        return pressed && o.enabled;
    }

    float segmentedWidth(const std::vector<std::string>& options) {
        float w = 0.0f;
        for (std::size_t i = 0; i < options.size(); ++i)
            w += theme::textSize(options[i], 12).x + px(22) + (i ? px(2) : 0.0f);
        return w;
    }

    bool segmented(const char* id, const std::vector<std::string>& options, int* current, const SegmentedOpts& o) {
        if (options.empty()) return false;
        const int n = static_cast<int>(options.size());
        const float fontSize = o.tiles ? 13.0f : 12.0f;
        const float h = theme::snap(px(o.tiles ? 36.0f : 26.0f));
        const float gap = theme::snap(px(2));
        std::vector<float> widths(options.size());
        float total = 0.0f;
        if (o.tiles) {
            total = theme::snap(lineWidth(o.width));
            const float base = std::floor((total - gap * static_cast<float>(n - 1)) / static_cast<float>(n));
            for (int i = 0; i < n; ++i) widths[static_cast<std::size_t>(i)] = base;
            widths.back() = total - gap * static_cast<float>(n - 1) - base * static_cast<float>(n - 1);
        } else {
            for (int i = 0; i < n; ++i) {
                widths[static_cast<std::size_t>(i)] = theme::snap(theme::textSize(options[static_cast<std::size_t>(i)], fontSize).x + px(22));
                total += widths[static_cast<std::size_t>(i)] + (i ? gap : 0.0f);
            }
        }
        bool changed = false;
        ImGui::PushID(id);
        ImGui::BeginGroup();
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImU32 selFill = o.accent ? theme::kAccent : theme::kText;
        float x = origin.x;
        for (int i = 0; i < n; ++i) {
            const std::size_t k = static_cast<std::size_t>(i);
            const bool en = o.enabled && (o.optionEnabled.empty() || (k < o.optionEnabled.size() && o.optionEnabled[k]));
            const bool sel = current && *current == i;
            const ImVec2 min(x, origin.y), max(x + widths[k], origin.y + h);
            ImGui::SetCursorScreenPos(min);
            ImGui::PushID(i);
            ImGui::BeginDisabled(!en);
            if (ImGui::InvisibleButton("##opt", ImVec2(std::max(1.0f, widths[k]), h)) && en && !sel) {
                if (current) *current = i;
                changed = true;
            }
            const bool hovered = ImGui::IsItemHovered();
            if (hovered && en) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            ImGui::EndDisabled();
            const ImU32 line = sel ? selFill : (hovered && en ? theme::kAccent : theme::kDivider);
            if (sel) dl->AddRectFilled(min, max, dim(selFill, en));
            crispRect(dl, min, max, dim(line, en), theme::kBorder);
            focusRing(dl, min, max);
            const float room = widths[k] - px(4);
            const std::string shown = elideText(options[k], room, fontSize);
            drawTextIn(dl, min, max, shown, fontSize, dim(sel ? theme::kBg : theme::kText, en));
            std::string tip = k < o.tooltips.size() ? o.tooltips[k] : std::string();
            // an elided tile says what it could not draw
            if (shown != options[k]) tip = tip.empty() ? options[k] : options[k] + " \xC2\xB7 " + tip;
            if (!tip.empty()) {
                ImGui::BeginDisabled(false);
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip | ImGuiHoveredFlags_AllowWhenDisabled)) tooltip(tip);
                ImGui::EndDisabled();
            }
            ImGui::PopID();
            x += widths[k] + gap;
        }
        ImGui::SetCursorScreenPos(origin);
        ImGui::Dummy(ImVec2(total, h));
        ImGui::EndGroup();
        ImGui::PopID();
        return changed;
    }

    namespace {
        bool glyphButtonImpl(const char* id, Icon icon, const char* glyph, ImVec2 designSize, const GlyphOpts& o) {
            const ImVec2 size(theme::snap(px(designSize.x)), theme::snap(px(designSize.y)));
            ImGui::BeginDisabled(!o.enabled);
            const ImVec2 min = ImGui::GetCursorScreenPos();
            const bool pressed = ImGui::InvisibleButton(id, size);
            const bool hovered = ImGui::IsItemHovered();
            if (hovered && o.enabled) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            ImGui::EndDisabled();
            const ImVec2 max(min.x + size.x, min.y + size.y);
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const float opacity = o.enabled ? 1.0f : 0.35f;
            const bool on = o.active;
            const bool live = hovered && o.enabled;
            ImU32 border = on || live ? theme::kAccent : o.border;
            if (o.onDark && !on && !live) border = theme::withAlpha(theme::kViewerText, 0.5f);
            else if (o.dimmed && !on && !live) border = theme::kNeutral400;
            if (o.borderless && !on) border = theme::kTransparent;
            if (on) dl->AddRectFilled(min, max, theme::withAlpha(theme::kAccent, opacity));
            if (border & 0xFF000000u) {
                if (o.dashed && !on) dashedRect(dl, min, max, theme::withAlpha(border, opacity));
                else crispRect(dl, min, max, theme::withAlpha(border, opacity), theme::kBorder);
            }
            focusRing(dl, min, max);
            ImU32 fg = o.idle;
            if (on) fg = theme::kBg;
            else if (live && o.borderless) fg = theme::kAccent;
            else if (o.onDark) fg = theme::kViewerText;
            else if (o.dimmed) fg = theme::kNeutral600;
            fg = theme::withAlpha(fg, opacity);
            if (icon != Icon::None) {
                const float side = o.iconPx > 0.0f ? px(o.iconPx) : std::max(px(10.0f), std::min(size.x, size.y) * 0.68f);
                const float design = side / std::max(theme::scale(), 0.01f);
                const float stroke = px(design >= 20.0f ? 1.8f : (design >= 14.0f ? 1.5f : 1.25f));
                drawIcon(dl, ImVec2((min.x + max.x) * 0.5f, (min.y + max.y) * 0.5f), side, icon, fg, stroke);
            } else if (glyph && *glyph) {
                drawTextIn(dl, min, max, glyph, o.glyphPx, fg);
            }
            if (!o.tooltip.empty()) {
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip | ImGuiHoveredFlags_AllowWhenDisabled)) tooltip(o.tooltip);
            }
            return pressed && o.enabled;
        }
    } // namespace

    bool glyphButton(const char* id, Icon icon, float size, const GlyphOpts& opts) {
        return glyphButtonImpl(id, icon, nullptr, ImVec2(size, size), opts);
    }

    bool glyphButton(const char* id, Icon icon, ImVec2 size, const GlyphOpts& opts) {
        return glyphButtonImpl(id, icon, nullptr, size, opts);
    }

    bool glyphTextButton(const char* id, const char* glyph, ImVec2 size, const GlyphOpts& opts) {
        return glyphButtonImpl(id, Icon::None, glyph, size, opts);
    }

    // --- messages to copy --------------------------------------------------------

    namespace {
        constexpr double kCopiedSeconds = 1.5;
        // The copy button clicked last, and until when it says so.
        struct CopiedMark {
            ImGuiID id = 0;
            double until = 0.0;
        };
        CopiedMark& copiedMark() {
            static CopiedMark mark;
            return mark;
        }
        float copySide(float lineH) { return theme::snap(std::max(px(16), lineH)); }
    } // namespace

    void copyToClipboard(const std::string& text) { ImGui::SetClipboardText(text.c_str()); }

    bool copyButton(const char* id, const std::string& text, float side, bool onDark) {
        const ImGuiID key = ImGui::GetID(id);
        CopiedMark& mark = copiedMark();
        const bool copied = mark.id == key && ImGui::GetTime() < mark.until;
        GlyphOpts o;
        o.borderless = true;
        o.onDark = onDark;
        o.idle = onDark ? theme::kViewerText : theme::kNeutral500;
        o.iconPx = std::max(10.0f, side - 4.0f);
        o.tooltip = copied ? "Copied" : "Copy to the clipboard";
        const bool clicked = glyphButton(id, copied ? Icon::Check : Icon::Copy, ImVec2(side, side), o);
        if (clicked) {
            copyToClipboard(text);
            mark.id = key;
            mark.until = ImGui::GetTime() + kCopiedSeconds;
        }
        return clicked;
    }

    void copyOnRightClick(const char* id, const std::string& text, const std::vector<std::pair<std::string, std::string>>& more) {
        // by the item's rectangle: text drawn over a row's button is never the
        // hovered item itself, and it is the text that is right-clicked
        const bool over = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) &&
                          ImGui::IsMouseHoveringRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
        if (over && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) ImGui::OpenPopup(id);
        ImGui::PushStyleColor(ImGuiCol_Border, theme::kText);
        ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, theme::crispPen(2));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, px(0, 4));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, px(0, 0));
        if (ImGui::BeginPopup(id)) {
            auto item = [](const std::string& label) {
                const ImVec2 p = ImGui::GetCursorScreenPos();
                const float h = theme::snap(px(26));
                const float w = std::max(px(160), theme::textSize(label, 12).x + px(36));
                const bool clicked = ImGui::Selectable(("##" + label).c_str(), false, ImGuiSelectableFlags_None, ImVec2(w, h));
                drawTextIn(ImGui::GetWindowDrawList(), ImVec2(p.x + px(12), p.y), ImVec2(p.x + w, p.y + h), label, 12, theme::kText,
                           Weight::Regular, 0.0f, 0.5f);
                return clicked;
            };
            if (item("Copy")) copyToClipboard(text);
            for (const auto& [label, value] : more)
                if (item(label)) copyToClipboard(value);
            ImGui::EndPopup();
        }
        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor();
    }

    float copyableTextHeight(const std::string& s, float pxSize, float wrapWidth, Weight w) {
        const theme::FontScope f(pxSize, w);
        const float lineH = ImGui::GetTextLineHeight();
        const float side = copySide(lineH);
        const float textW = std::max(px(20), wrapWidth - side - px(4));
        const float h = ImGui::CalcTextSize(s.c_str(), s.c_str() + s.size(), false, textW).y;
        return std::max(h, side);
    }

    void copyableText(const char* id, const std::string& s, float pxSize, ImU32 color, Weight w, float wrapWidth, const std::string& copied) {
        ImGui::PushID(id);
        const float total = wrapWidth > 0.0f ? wrapWidth : std::max(px(40), ImGui::GetContentRegionAvail().x);
        float lineH = 0.0f;
        {
            const theme::FontScope f(pxSize, w);
            lineH = ImGui::GetTextLineHeight();
        }
        const float side = copySide(lineH);
        const float textW = std::max(px(20), total - side - px(4));
        const std::string& text = copied.empty() ? s : copied;
        // the button first, at the end of the first line; then the text, which leaves the cursor
        const ImVec2 at = ImGui::GetCursorScreenPos();
        ImGui::SetCursorScreenPos(ImVec2(at.x + total - side, theme::snap(at.y + (lineH - side) * 0.5f)));
        copyButton("##copy", text, side / std::max(theme::scale(), 0.01f));
        ImGui::SetCursorScreenPos(at);
        textWrapped(s, pxSize, color, w, textW);
        copyOnRightClick("##copyMenu", text);
        ImGui::PopID();
    }

    bool tokenCheck(const char* label, bool checked, const char* cap, bool enabled, bool onDark) {
        const char* end = labelEnd(label);
        const std::string shown(label, end);
        const std::string capText = cap ? captionCase(cap) : std::string();
        const float box = theme::snap(px(14));
        const ImVec2 ts = theme::textSize(shown, 12);
        const ImVec2 cs = capText.empty() ? ImVec2(0, 0) : theme::textSize(capText, 10);
        const float h = std::max(box, ts.y) + px(4);
        float w = box;
        if (!shown.empty()) w += px(8) + ts.x;
        if (!capText.empty()) w += px(6) + cs.x + px(static_cast<float>(capText.size()));
        ImGui::BeginDisabled(!enabled);
        const ImVec2 min = ImGui::GetCursorScreenPos();
        const bool pressed = ImGui::InvisibleButton(label, ImVec2(w, h));
        const bool hovered = ImGui::IsItemHovered();
        if (hovered && enabled) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        ImGui::EndDisabled();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 b0(min.x, theme::snap(min.y + (h - box) * 0.5f)), b1(b0.x + box, b0.y + box);
        const ImU32 ink = onDark ? theme::kViewerText : theme::kText;
        if (checked) dl->AddRectFilled(b0, b1, dim(theme::kAccent, enabled));
        crispRect(dl, b0, b1, dim(checked ? theme::kAccent : (hovered ? theme::kAccent : (onDark ? theme::kViewerText : theme::kNeutral700)), enabled),
                  theme::kBorder);
        float x = b1.x + px(8);
        if (!shown.empty()) {
            drawTextIn(dl, ImVec2(x, min.y), ImVec2(x + ts.x, min.y + h), shown, 12, dim(ink, enabled), Weight::Regular, 0.0f, 0.5f);
            x += ts.x + px(6);
        }
        if (!capText.empty()) {
            ImFont* f = theme::captionFont();
            const float size = fontPx(10);
            dl->AddText(f, size, ImVec2(theme::snap(x), theme::snap(min.y + (h - cs.y) * 0.5f)),
                        dim(onDark ? theme::withAlpha(theme::kViewerText, 0.7f) : theme::kNeutral600, enabled), capText.c_str());
        }
        focusRing(dl, min, ImVec2(min.x + w, min.y + h));
        return pressed && enabled;
    }

    bool checkbox(const char* label, bool* v, bool enabled) {
        if (tokenCheck(label, v && *v, nullptr, enabled)) {
            if (v) *v = !*v;
            return true;
        }
        return false;
    }

    bool radio(const char* label, bool selected, bool enabled) {
        const char* end = labelEnd(label);
        const std::string shown(label, end);
        const float d = theme::snap(px(12));
        const ImVec2 ts = theme::textSize(shown, 12);
        const float h = std::max(d, ts.y) + px(4);
        const float w = d + (shown.empty() ? 0.0f : px(8) + ts.x);
        ImGui::BeginDisabled(!enabled);
        const ImVec2 min = ImGui::GetCursorScreenPos();
        const bool pressed = ImGui::InvisibleButton(label, ImVec2(w, h));
        const bool hovered = ImGui::IsItemHovered();
        if (hovered && enabled) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        ImGui::EndDisabled();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 c(min.x + d * 0.5f, min.y + h * 0.5f);
        if (selected) dl->AddCircleFilled(c, d * 0.5f, dim(theme::kAccent, enabled));
        dl->AddCircle(c, d * 0.5f - 0.5f, dim(hovered ? theme::kAccent : theme::kNeutral700, enabled), 0, theme::crispPen(theme::kBorder));
        if (!shown.empty())
            drawTextIn(dl, ImVec2(min.x + d + px(8), min.y), ImVec2(min.x + w, min.y + h), shown, 12, dim(theme::kText, enabled),
                       Weight::Regular, 0.0f, 0.5f);
        return pressed && enabled;
    }

    bool channelSwatch(const char* id, const std::string& label, ImU32 color, bool on, const std::string& tip) {
        const float s = theme::snap(px(22));
        const ImVec2 min = ImGui::GetCursorScreenPos();
        const bool pressed = ImGui::InvisibleButton(id, ImVec2(s, s));
        const bool hovered = ImGui::IsItemHovered();
        if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        const ImVec2 max(min.x + s, min.y + s);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        if (on) dl->AddRectFilled(min, max, color);
        crispRect(dl, min, max, hovered && !on ? theme::kAccent : color, theme::kBorder);
        // ink on a light fill, the colour itself on the page when hidden
        const float r = static_cast<float>(color & 0xFF), g = static_cast<float>((color >> 8) & 0xFF),
                    b = static_cast<float>((color >> 16) & 0xFF);
        const float luma = 0.299f * r + 0.587f * g + 0.114f * b;
        const ImU32 fg = on ? (luma > 140.0f ? theme::kText : theme::kBg) : theme::kNeutral700;
        const std::string shown = elideText(label, s - px(2), 9, Weight::Bold);
        drawTextIn(dl, min, max, shown == "\xE2\x80\xA6" ? std::string() : shown, 9, fg, Weight::Bold);
        focusRing(dl, min, max);
        if (!tip.empty()) tooltip(tip);
        return pressed;
    }

    // --- sliders ---------------------------------------------------------------

    namespace {
        struct Track {
            ImVec2 min;
            float w, h;
            float x0, x1;   // where the handle centre can go
        };

        Track beginTrack(const char* id, const SliderOpts& o, bool& active, bool& hovered) {
            Track t;
            t.w = std::max(px(40), lineWidth(o.width < 0.0f ? -1.0f : (o.width == 0.0f ? 120.0f : o.width)));
            t.h = theme::snap(px(18));
            t.min = ImGui::GetCursorScreenPos();
            ImGui::BeginDisabled(!o.enabled);
            ImGui::InvisibleButton(id, ImVec2(t.w, t.h));
            active = ImGui::IsItemActive() && o.enabled;
            hovered = ImGui::IsItemHovered() && o.enabled;
            ImGui::EndDisabled();
            const float half = px(5);
            t.x0 = t.min.x + half;
            t.x1 = t.min.x + t.w - half;
            return t;
        }

        void drawGroove(ImDrawList* dl, const Track& t, float from, float to, const SliderOpts& o) {
            const float cy = theme::snap(t.min.y + t.h * 0.5f), th = theme::crispPen(2);
            const ImU32 groove = o.onDark ? theme::withAlpha(theme::kViewerText, 0.35f) : theme::kNeutral300;
            dl->AddRectFilled(ImVec2(t.min.x, cy - th * 0.5f), ImVec2(t.min.x + t.w, cy + th * 0.5f), groove);
            dl->AddRectFilled(ImVec2(from, cy - th * 0.5f), ImVec2(to, cy + th * 0.5f), o.enabled ? theme::kAccent : theme::kNeutral400);
        }

        void drawHandle(ImDrawList* dl, const Track& t, float x, bool hot, const SliderOpts& o) {
            const float hw = px(5), hh = px(7), cy = theme::snap(t.min.y + t.h * 0.5f);
            const ImU32 c = !o.enabled ? theme::kNeutral400 : (hot ? theme::kAccent600 : theme::kAccent);
            dl->AddRectFilled(ImVec2(theme::snap(x - hw), cy - hh), ImVec2(theme::snap(x + hw), cy + hh), c);
        }

        double fromMouse(const Track& t) {
            const float x = ImGui::GetIO().MousePos.x;
            return std::clamp(static_cast<double>((x - t.x0) / std::max(1.0f, t.x1 - t.x0)), 0.0, 1.0);
        }

        // The arrows step a focused slider: the slider (the last item) owns them,
        // so a plain Left / Right menu shortcut, which asks for keys nobody owns,
        // stands back. Ownership set now holds on the frame the key goes down.
        void claimArrows() {
            const ImGuiID id = ImGui::GetItemID();
            ImGui::SetKeyOwner(ImGuiKey_LeftArrow, id);
            ImGui::SetKeyOwner(ImGuiKey_RightArrow, id);
        }
    } // namespace

    bool slider(const char* id, double* v, double lo, double hi, const SliderOpts& o) {
        bool active = false, hovered = false;
        const Track t = beginTrack(id, o, active, hovered);
        bool changed = false;
        const double span = hi - lo;
        if (active && span > 0.0) {
            const double nv = lo + fromMouse(t) * span;
            if (nv != *v) {
                *v = nv;
                changed = true;
            }
        }
        if (ImGui::IsItemFocused() && o.enabled && span > 0.0) {
            claimArrows();
            const double step = span / 100.0;
            if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) {
                *v = std::max(lo, *v - step);
                changed = true;
            }
            if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) {
                *v = std::min(hi, *v + step);
                changed = true;
            }
        }
        const double f = span > 0.0 ? std::clamp((*v - lo) / span, 0.0, 1.0) : 0.0;
        const float x = t.x0 + static_cast<float>(f) * (t.x1 - t.x0);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        drawGroove(dl, t, t.min.x, x, o);
        drawHandle(dl, t, x, active || hovered, o);
        if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        return changed;
    }

    bool sliderInt(const char* id, std::int64_t* v, std::int64_t lo, std::int64_t hi, const SliderOpts& o) {
        bool active = false, hovered = false;
        const Track t = beginTrack(id, o, active, hovered);
        bool changed = false;
        const std::int64_t span = hi - lo;
        std::int64_t value = std::clamp(*v, lo, std::max(lo, hi));
        if (active && span > 0) {
            const std::int64_t nv = lo + static_cast<std::int64_t>(std::llround(fromMouse(t) * static_cast<double>(span)));
            if (nv != value) {
                value = nv;
                changed = true;
            }
        }
        if (ImGui::IsItemFocused() && o.enabled && span > 0) {
            claimArrows();
            if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow) && value > lo) {
                --value;
                changed = true;
            }
            if (ImGui::IsKeyPressed(ImGuiKey_RightArrow) && value < hi) {
                ++value;
                changed = true;
            }
        }
        if (hovered && span > 0) {
            const float wheel = ImGui::GetIO().MouseWheel;
            if (wheel != 0.0f && ImGui::GetIO().KeyCtrl) {
                value = std::clamp<std::int64_t>(value + (wheel > 0 ? 1 : -1), lo, hi);
                changed = true;
            }
        }
        if (changed) *v = value;
        const double f = span > 0 ? static_cast<double>(value - lo) / static_cast<double>(span) : 0.0;
        const float x = t.x0 + static_cast<float>(f) * (t.x1 - t.x0);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        drawGroove(dl, t, t.min.x, x, o);
        drawHandle(dl, t, x, active || hovered, o);
        if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        return changed;
    }

    bool rangeSlider(const char* id, double* lo, double* hi, const SliderOpts& o) {
        bool active = false, hovered = false;
        const Track t = beginTrack(id, o, active, hovered);
        ImGuiStorage* store = ImGui::GetStateStorage();
        const ImGuiID key = ImGui::GetID(id);
        int drag = store->GetInt(key, 0);   // 1 = low handle, 2 = high handle
        bool changed = false;
        if (ImGui::IsItemActivated()) {
            const double f = fromMouse(t);
            drag = std::abs(f - *lo) <= std::abs(f - *hi) ? 1 : 2;
            store->SetInt(key, drag);
        }
        if (active && drag) {
            const double f = fromMouse(t);
            double nlo = *lo, nhi = *hi;
            if (drag == 1) nlo = std::min(f, *hi - 0.01);
            else nhi = std::max(f, *lo + 0.01);
            nlo = std::clamp(nlo, 0.0, 1.0);
            nhi = std::clamp(nhi, 0.0, 1.0);
            if (nlo != *lo || nhi != *hi) {
                *lo = nlo;
                *hi = nhi;
                changed = true;
            }
        }
        if (!active && drag) store->SetInt(key, 0);
        const float xl = t.x0 + static_cast<float>(*lo) * (t.x1 - t.x0), xh = t.x0 + static_cast<float>(*hi) * (t.x1 - t.x0);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        drawGroove(dl, t, xl, xh, o);
        drawHandle(dl, t, xl, (active && drag == 1) || hovered, o);
        drawHandle(dl, t, xh, (active && drag == 2) || hovered, o);
        if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        return changed;
    }

    // --- tabs ------------------------------------------------------------------

    bool tabRow(const char* id, const std::vector<std::string>& tabs, int* current) {
        bool changed = false;
        ImGui::PushID(id);
        ImGui::BeginGroup();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const float h = theme::snap(px(26));
        for (std::size_t i = 0; i < tabs.size(); ++i) {
            const bool sel = current && *current == static_cast<int>(i);
            // measured in the heavier face so a tab does not move when it is chosen
            const float w = theme::snap(std::max(px(54), theme::textSize(tabs[i], 12, Weight::ExtraBold).x + px(20)));
            if (i) ImGui::SameLine(0.0f, px(2));
            ImGui::PushID(static_cast<int>(i));
            const ImVec2 min = ImGui::GetCursorScreenPos();
            if (ImGui::InvisibleButton("##tab", ImVec2(w, h)) && !sel) {
                if (current) *current = static_cast<int>(i);
                changed = true;
            }
            const bool hovered = ImGui::IsItemHovered();
            if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            const ImVec2 max(min.x + w, min.y + h);
            drawTextIn(dl, min, max, tabs[i], 12, hovered && !sel ? theme::kAccent : theme::kText,
                       sel ? Weight::ExtraBold : Weight::Regular);
            if (sel) dl->AddRectFilled(ImVec2(min.x, max.y - theme::crispPen(2)), max, theme::kAccent);
            focusRing(dl, min, max);
            ImGui::PopID();
        }
        ImGui::EndGroup();
        ImGui::PopID();
        return changed;
    }

    // --- fields ---------------------------------------------------------------

    void fieldLabel(const std::string& label, const std::string& unit) {
        const std::string s = unit.empty() ? label : label + " (" + unit + ")";
        const float spacing = ImGui::GetStyle().ItemSpacing.y;
        // A label placed beside a field with SameLine() would otherwise take
        // the field's text baseline offset and sit lower than its neighbour's.
        ImGui::GetCurrentWindow()->DC.CurrLineTextBaseOffset = 0.0f;
        text(s, 11, theme::kNeutral700);
        // the label sits 4 px above its input, whatever the item spacing is
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - spacing + px(4));
    }

    namespace {
        // Pushes the field look; returns the item width.
        struct FieldFrame {
            bool mono;
            explicit FieldFrame(const FieldOpts& o, float rightInset = 0.0f) : mono(o.monospace) {
                const float fontSize = o.monospace ? 12.0f : 13.0f;
                ImGui::PushFont(o.monospace ? theme::mono() : theme::font(), fontSize);
                const float fh = fontPx(fontSize);
                const float pad = std::max(2.0f, std::floor((px(o.height) - fh) * 0.5f));
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(px(8), pad));
                ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, theme::crispPen(theme::kBorder));
                ImGui::PushStyleColor(ImGuiCol_FrameBg, o.readOnly ? theme::kSurface : theme::kBg);
                ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, o.readOnly ? theme::kSurface : theme::kBg);
                ImGui::PushStyleColor(ImGuiCol_FrameBgActive, o.readOnly ? theme::kSurface : theme::kBg);
                ImGui::PushStyleColor(ImGuiCol_Border, o.enabled ? theme::kDivider : theme::kNeutral300);
                ImGui::SetNextItemWidth(std::max(px(24), lineWidth(o.width == 0.0f ? 120.0f : o.width) - rightInset));
                ImGui::BeginDisabled(!o.enabled);
            }
            ~FieldFrame() {
                ImGui::EndDisabled();
                ImGui::PopStyleColor(4);
                ImGui::PopStyleVar(2);
                ImGui::PopFont();
            }
        };

        // The 2 px accent ring of a field that has the keyboard.
        void activeRing() {
            if (ImGui::IsItemActive() || (ImGui::IsItemFocused() && ImGui::GetIO().NavVisible))
                crispRect(ImGui::GetWindowDrawList(), ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), theme::kAccent, 2.0f);
        }

        std::string numberFormat(double step, int decimals) {
            if (decimals >= 0) return format("%%.%df", decimals);
            if (step > 0.0) {
                int d = 0;
                double s = step;
                while (d < 8 && std::abs(s - std::round(s)) > 1e-9) {
                    s *= 10.0;
                    ++d;
                }
                return format("%%.%df", d);
            }
            return "%.6g";
        }

        // The layout as the last item left it. The spin arrows and the combo's
        // drop button are items placed over a field that is already laid out;
        // SetCursorScreenPos back to where the field ended would leave Dear
        // ImGui thinking the cursor was moved by hand (the "SetCursorPos
        // extends boundaries" error when a group or cell ends there) and would
        // make SameLine() start from the arrow. Restoring the whole state makes
        // what follows see the field as the last item, as if the extra
        // buttons were never there -- IsItemDeactivatedAfterEdit() included.
        struct LayoutSnapshot {
            ImVec2 cursor, prevLine, maxPos, idealMax, currLineSize, prevLineSize;
            float currBase, prevBase;
            bool sameLine, setPos;
            ImGuiLastItemData last;
            LayoutSnapshot() {
                const ImGuiWindowTempData& dc = ImGui::GetCurrentWindow()->DC;
                cursor = dc.CursorPos;
                prevLine = dc.CursorPosPrevLine;
                maxPos = dc.CursorMaxPos;
                idealMax = dc.IdealMaxPos;
                currLineSize = dc.CurrLineSize;
                prevLineSize = dc.PrevLineSize;
                currBase = dc.CurrLineTextBaseOffset;
                prevBase = dc.PrevLineTextBaseOffset;
                sameLine = dc.IsSameLine;
                setPos = dc.IsSetPos;
                last = ImGui::GetCurrentContext()->LastItemData;
            }
            void restore() const {
                ImGuiWindowTempData& dc = ImGui::GetCurrentWindow()->DC;
                dc.CursorPos = cursor;
                dc.CursorPosPrevLine = prevLine;
                dc.CursorMaxPos = ImMax(dc.CursorMaxPos, maxPos);
                dc.IdealMaxPos = ImMax(dc.IdealMaxPos, idealMax);
                dc.CurrLineSize = currLineSize;
                dc.PrevLineSize = prevLineSize;
                dc.CurrLineTextBaseOffset = currBase;
                dc.PrevLineTextBaseOffset = prevBase;
                dc.IsSameLine = sameLine;
                dc.IsSetPos = setPos;
                ImGui::GetCurrentContext()->LastItemData = last;
            }
        };

        // Up / down chevrons on the right of the last item; returns -1, 0, +1.
        int spinArrows(const char* id, ImVec2 min, ImVec2 max, bool enabled) {
            const float w = theme::snap(px(16));
            const ImVec2 a(max.x - w - theme::crispPen(theme::kBorder), min.y), b(max.x - theme::crispPen(theme::kBorder), max.y);
            const float mid = theme::snap((a.y + b.y) * 0.5f);
            int dir = 0;
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const LayoutSnapshot layout;
            ImGui::PushID(id);
            ImGui::BeginDisabled(!enabled);
            ImGui::PushItemFlag(ImGuiItemFlags_ButtonRepeat, true);
            ImGui::SetCursorScreenPos(a);
            if (ImGui::InvisibleButton("##up", ImVec2(w, mid - a.y))) dir = 1;
            const bool hotUp = ImGui::IsItemHovered();
            ImGui::SetCursorScreenPos(ImVec2(a.x, mid));
            if (ImGui::InvisibleButton("##down", ImVec2(w, b.y - mid))) dir = -1;
            const bool hotDown = ImGui::IsItemHovered();
            ImGui::PopItemFlag();
            ImGui::EndDisabled();
            ImGui::PopID();
            layout.restore();
            const float s = px(9);
            drawIcon(dl, ImVec2((a.x + b.x) * 0.5f, (a.y + mid) * 0.5f + px(2)), s, Icon::ChevronUp,
                     dim(hotUp ? theme::kAccent : theme::kNeutral700, enabled), px(1.25f));
            drawIcon(dl, ImVec2((a.x + b.x) * 0.5f, (mid + b.y) * 0.5f - px(2)), s, Icon::ChevronDown,
                     dim(hotDown ? theme::kAccent : theme::kNeutral700, enabled), px(1.25f));
            return dir;
        }
    } // namespace

    bool inputText(const char* id, std::string* value, const FieldOpts& o) {
        const FieldFrame frame(o);
        ImGuiInputTextFlags flags = ImGuiInputTextFlags_None;
        if (o.readOnly) flags |= ImGuiInputTextFlags_ReadOnly;
        // a password keeps no undo history: nothing of it outlives the field
        if (o.password) flags |= ImGuiInputTextFlags_Password | ImGuiInputTextFlags_NoUndoRedo;
        if (o.enterReturnsTrue) flags |= ImGuiInputTextFlags_EnterReturnsTrue;
        ImGui::PushStyleColor(ImGuiCol_TextDisabled, theme::kNeutral500);
        const bool changed = o.hint.empty() ? ImGui::InputText(id, value, flags)
                                            : ImGui::InputTextWithHint(id, o.hint.c_str(), value, flags);
        ImGui::PopStyleColor();
        if (!o.readOnly) activeRing();
        return changed;
    }

    void forgetInputText(ImGuiID id) {
        if (id == 0 || !ImGui::GetCurrentContext()) return;
        ImGuiContext& g = *GImGui;
        const auto wipe = [](ImVector<char>& v) {
            if (v.Data && v.Capacity > 0) {
                volatile char* p = v.Data;
                for (int i = 0; i < v.Capacity; ++i) p[i] = 0;
            }
            v.clear();
        };
        if (g.ActiveId == id) ImGui::ClearActiveID();
        if (g.InputTextState.ID == id) {
            wipe(g.InputTextState.TextA);
            wipe(g.InputTextState.TextToRevertTo);
            wipe(g.InputTextState.CallbackTextBackup);
            g.InputTextState.TextLen = 0;
            g.InputTextState.ID = 0;
        }
        if (g.InputTextDeactivatedState.ID == id) {
            wipe(g.InputTextDeactivatedState.TextA);
            g.InputTextDeactivatedState.ID = 0;
        }
    }

    bool inputTextMultiline(const char* id, std::string* value, float heightPx, const FieldOpts& o) {
        const float fontSize = o.monospace ? 12.0f : 13.0f;
        ImGui::PushFont(o.monospace ? theme::mono() : theme::font(), fontSize);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, px(8, 6));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, theme::crispPen(theme::kBorder));
        ImGui::PushStyleColor(ImGuiCol_FrameBg, o.readOnly ? theme::kSurface : theme::kBg);
        ImGui::PushStyleColor(ImGuiCol_Border, theme::kDivider);
        ImGui::BeginDisabled(!o.enabled);
        ImGuiInputTextFlags flags = ImGuiInputTextFlags_None;
        if (o.readOnly) flags |= ImGuiInputTextFlags_ReadOnly;
        if (o.enterReturnsTrue) flags |= ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CtrlEnterForNewLine;
        const float w = std::max(px(24), lineWidth(o.width == 0.0f ? 240.0f : o.width));
        const bool changed = ImGui::InputTextMultiline(id, value, ImVec2(w, px(heightPx)), flags);
        if (!o.readOnly) activeRing();
        ImGui::EndDisabled();
        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(2);
        ImGui::PopFont();
        return changed;
    }

    bool inputInt(const char* id, std::int64_t* value, std::int64_t lo, std::int64_t hi, std::int64_t step, const FieldOpts& o) {
        bool changed = false;
        ImVec2 min, max;
        {
            const FieldFrame frame(o);
            ImGuiInputTextFlags flags = o.readOnly ? ImGuiInputTextFlags_ReadOnly : ImGuiInputTextFlags_None;
            std::int64_t v = *value;
            // the spin arrows are items over the field's right end
            if (step > 0 && !o.readOnly) ImGui::SetNextItemAllowOverlap();
            // live edit, as inputDouble: typing reports on the frames it happens, both alike
            ImGui::PushItemFlag(ImGuiItemFlags_LiveEditOnInputScalar, true);
            const bool edited = ImGui::InputScalar(id, ImGuiDataType_S64, &v, nullptr, nullptr, "%lld", flags);
            ImGui::PopItemFlag();
            if (edited) {
                v = std::clamp(v, lo, std::max(lo, hi));
                if (v != *value) {
                    *value = v;
                    changed = true;
                }
            }
            min = ImGui::GetItemRectMin();
            max = ImGui::GetItemRectMax();
            // the wheel steps over the whole field, the spin arrows on it included
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenOverlappedByItem) && ImGui::IsItemFocused() && step > 0 &&
                !o.readOnly) {
                const float wheel = ImGui::GetIO().MouseWheel;
                if (wheel != 0.0f) {
                    *value = std::clamp(*value + (wheel > 0 ? step : -step), lo, std::max(lo, hi));
                    changed = true;
                }
            }
            if (!o.readOnly) activeRing();
        }
        if (step > 0 && !o.readOnly) {
            const int dir = spinArrows(id, min, max, o.enabled);
            if (dir != 0) {
                const std::int64_t v = std::clamp(*value + dir * step, lo, std::max(lo, hi));
                if (v != *value) {
                    *value = v;
                    changed = true;
                }
            }
        }
        return changed;
    }

    bool inputDouble(const char* id, double* value, double lo, double hi, double step, int decimals, const FieldOpts& o) {
        bool changed = false;
        ImVec2 min, max;
        const std::string fmt = numberFormat(step, decimals);
        {
            const FieldFrame frame(o);
            ImGuiInputTextFlags flags = o.readOnly ? ImGuiInputTextFlags_ReadOnly : ImGuiInputTextFlags_None;
            double v = *value;
            // the spin arrows are items over the field's right end
            if (step > 0.0 && !o.readOnly) ImGui::SetNextItemAllowOverlap();
            // Without live edit, Dear ImGui parses the field's text whenever it lets
            // go, typed into or not: a value with more digits than it shows would
            // come back rounded to them just for being focused.
            ImGui::PushItemFlag(ImGuiItemFlags_LiveEditOnInputScalar, true);
            const bool edited = ImGui::InputScalar(id, ImGuiDataType_Double, &v, nullptr, nullptr, fmt.c_str(), flags);
            ImGui::PopItemFlag();
            // Escape after typing puts back the text the field showed when it took
            // the keyboard, which live edit then parses: the value rounded to
            // `decimals`. The value itself is kept to put back instead. Only one
            // item has the keyboard at a time, so one slot serves every field.
            static struct {
                ImGuiID id;
                double value;
            } typedFrom = {0, 0.0};
            const ImGuiID fieldId = ImGui::GetItemID();
            if (ImGui::IsItemActivated()) typedFrom = {fieldId, *value};
            if (edited) {
                if (std::isfinite(lo)) v = std::max(v, lo);
                if (std::isfinite(hi)) v = std::min(v, hi);
                if (v != *value) {
                    *value = v;
                    changed = true;
                }
            }
            if (ImGui::IsItemDeactivated() && typedFrom.id == fieldId) {
                typedFrom.id = 0;
                if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) && *value != typedFrom.value) {
                    *value = typedFrom.value;
                    changed = true;
                }
            }
            min = ImGui::GetItemRectMin();
            max = ImGui::GetItemRectMax();
            if (!o.readOnly) activeRing();
        }
        if (step > 0.0 && !o.readOnly) {
            const int dir = spinArrows(id, min, max, o.enabled);
            if (dir != 0) {
                double v = *value + dir * step;
                if (std::isfinite(lo)) v = std::max(v, lo);
                if (std::isfinite(hi)) v = std::min(v, hi);
                if (v != *value) {
                    *value = v;
                    changed = true;
                }
            }
        }
        return changed;
    }

    namespace {
        // The popup of a dropdown: 2 px ink border, rows that highlight in neutral-200
        // and touch. Without `rowSpacing` the rows' spacing is left to the caller,
        // for BeginCombo, which lays out its field under the same pushes.
        struct PopupLook {
            int vars;
            explicit PopupLook(bool rowSpacing = true) : vars(rowSpacing ? 3 : 2) {
                ImGui::PushStyleColor(ImGuiCol_Border, theme::kText);
                ImGui::PushStyleColor(ImGuiCol_Header, theme::kSurface);
                ImGui::PushStyleColor(ImGuiCol_HeaderHovered, theme::kNeutral200);
                ImGui::PushStyleColor(ImGuiCol_HeaderActive, theme::kNeutral300);
                ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, theme::crispPen(2));
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, px(0, 2));
                if (rowSpacing) ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, px(0, 0));
            }
            ~PopupLook() {
                ImGui::PopStyleVar(vars);
                ImGui::PopStyleColor(4);
            }
        };

        bool popupItem(const std::string& label, bool selected) {
            const ImVec2 p = ImGui::GetCursorScreenPos();
            const float h = theme::snap(px(26));
            const bool clicked = ImGui::Selectable(("##" + label).c_str(), selected, ImGuiSelectableFlags_None, ImVec2(0, h));
            const float w = ImGui::GetItemRectSize().x;
            drawTextIn(ImGui::GetWindowDrawList(), ImVec2(p.x + px(8), p.y), ImVec2(p.x + w, p.y + h), label, 12, theme::kText,
                       selected ? Weight::SemiBold : Weight::Regular, 0.0f, 0.5f);
            if (selected) ImGui::SetItemDefaultFocus();
            return clicked;
        }

        void chevron(ImVec2 min, ImVec2 max, bool enabled) {
            drawIcon(ImGui::GetWindowDrawList(), ImVec2(max.x - px(13), (min.y + max.y) * 0.5f), px(12), Icon::ChevronDown,
                     dim(theme::kText, enabled), px(1.5f));
        }
    } // namespace

    bool combo(const char* id, int* current, const std::vector<std::string>& items, const FieldOpts& o) {
        bool changed = false;
        const int n = static_cast<int>(items.size());
        const std::string preview = current && *current >= 0 && *current < n ? items[static_cast<std::size_t>(*current)] : std::string();
        ImVec2 min, max;
        {
            const FieldFrame frame(o);
            {
                // BeginCombo draws the field and begins the popup in one call, and the
                // popup's ink border is read there: the field is drawn without a border
                // and gets FieldFrame's below. The rows' spacing is pushed inside the
                // popup, so the field keeps the usual spacing under it.
                const PopupLook look(false);
                ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
                // ten rows before the list scrolls
                ImGui::SetNextWindowSizeConstraints(ImVec2(0.0f, 0.0f), ImVec2(FLT_MAX, 10.0f * theme::snap(px(26)) + px(4)));
                // the preview is drawn below, elided, so a long choice never runs under the chevron
                if (ImGui::BeginCombo(id, "", ImGuiComboFlags_NoArrowButton)) {
                    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, px(0, 0));
                    for (int i = 0; i < n; ++i) {
                        ImGui::PushID(i);
                        if (popupItem(items[static_cast<std::size_t>(i)], current && *current == i)) {
                            if (current && *current != i) {
                                *current = i;
                                changed = true;
                            }
                        }
                        ImGui::PopID();
                    }
                    ImGui::PopStyleVar();
                    ImGui::EndCombo();
                }
                ImGui::PopStyleVar();
            }
            // The popup's End() made the field the last item again: its rectangle
            // is the field's whether the list is open or not.
            min = ImGui::GetItemRectMin();
            max = ImGui::GetItemRectMax();
            ImGui::RenderFrameBorder(min, max, 0.0f);
        }
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const float room = (max.x - min.x) - px(8) - px(24);
        drawTextIn(dl, ImVec2(min.x + px(8), min.y), ImVec2(max.x - px(24), max.y), elideText(preview, room, 13), 13,
                   dim(theme::kText, o.enabled), Weight::Regular, 0.0f, 0.5f);
        chevron(min, max, o.enabled);
        if (ImGui::IsItemHovered() && o.enabled) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        return changed;
    }

    bool detailCombo(const char* id, int* current, const std::vector<ComboItem>& items, const FieldOpts& o, float popupWidth) {
        bool changed = false;
        const int n = static_cast<int>(items.size());
        const std::string preview = current && *current >= 0 && *current < n ? items[static_cast<std::size_t>(*current)].value : std::string();
        ImVec2 min, max;
        {
            const FieldFrame frame(o);
            {
                const PopupLook look(false);
                ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
                const float fieldW = ImGui::CalcItemWidth();
                const float w = std::max(fieldW, popupWidth > 0.0f ? px(popupWidth) : 0.0f);
                ImGui::SetNextWindowSizeConstraints(ImVec2(w, 0.0f), ImVec2(w, px(360)));
                if (ImGui::BeginCombo(id, "", ImGuiComboFlags_NoArrowButton)) {
                    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, px(0, 0));
                    for (int i = 0; i < n; ++i) {
                        const ComboItem& item = items[static_cast<std::size_t>(i)];
                        const bool selected = current && *current == i;
                        ImGui::PushID(i);
                        const ImVec2 p = ImGui::GetCursorScreenPos();
                        const float h = theme::snap(px(item.detail.empty() ? 26.0f : 40.0f));
                        const ImGuiSelectableFlags flags = item.dimmed ? ImGuiSelectableFlags_Disabled : ImGuiSelectableFlags_None;
                        const bool clicked = ImGui::Selectable("##row", selected, flags, ImVec2(0, h));
                        const float rw = ImGui::GetItemRectSize().x;
                        ImDrawList* dl = ImGui::GetWindowDrawList();
                        const float top = item.detail.empty() ? p.y : p.y + px(3);
                        const float nameH = item.detail.empty() ? h : px(19);
                        drawTextIn(dl, ImVec2(p.x + px(8), top), ImVec2(p.x + rw - px(8), top + nameH), elideText(item.value, rw - px(16), 12), 12,
                                   item.dimmed ? theme::kNeutral500 : theme::kText, selected ? Weight::SemiBold : Weight::Regular, 0.0f, 0.5f);
                        if (!item.detail.empty()) {
                            drawTextIn(dl, ImVec2(p.x + px(8), top + nameH), ImVec2(p.x + rw - px(8), p.y + h - px(3)),
                                       elideText(item.detail, rw - px(16), 11), 11, item.dimmed ? theme::kAccentText : theme::kNeutral600,
                                       Weight::Regular, 0.0f, 0.5f);
                            tooltip(item.detail);
                        }
                        if (selected) ImGui::SetItemDefaultFocus();
                        if (clicked && !item.dimmed && current && *current != i) {
                            *current = i;
                            changed = true;
                        }
                        ImGui::PopID();
                    }
                    ImGui::PopStyleVar();
                    ImGui::EndCombo();
                }
                ImGui::PopStyleVar();
            }
            min = ImGui::GetItemRectMin();
            max = ImGui::GetItemRectMax();
            ImGui::RenderFrameBorder(min, max, 0.0f);
        }
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const float room = (max.x - min.x) - px(8) - px(24);
        drawTextIn(dl, ImVec2(min.x + px(8), min.y), ImVec2(max.x - px(24), max.y), elideText(preview, room, 13), 13,
                   dim(theme::kText, o.enabled), Weight::Regular, 0.0f, 0.5f);
        chevron(min, max, o.enabled);
        if (ImGui::IsItemHovered() && o.enabled) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        return changed;
    }

    bool editableCombo(const char* id, std::string* value, const std::vector<std::string>& items, const FieldOpts& o) {
        std::vector<ComboItem> rows;
        rows.reserve(items.size());
        for (const std::string& i : items) rows.push_back(ComboItem{i, {}, false});
        return editableCombo(id, value, rows, o, nullptr);
    }

    bool editableCombo(const char* id, std::string* value, const std::vector<ComboItem>& items, const FieldOpts& o, bool* picked,
                       float popupWidth) {
        bool changed = false;
        if (picked) *picked = false;
        ImGui::PushID(id);
        const float arrow = theme::snap(px(24));
        ImVec2 min, max;
        {
            const FieldFrame frame(o, 0.0f);
            ImGui::PushStyleColor(ImGuiCol_TextDisabled, theme::kNeutral500);
            ImGuiInputTextFlags flags = o.enterReturnsTrue ? ImGuiInputTextFlags_EnterReturnsTrue : ImGuiInputTextFlags_None;
            // the drop button is an item over the field's right end
            ImGui::SetNextItemAllowOverlap();
            if (o.hint.empty()) changed = ImGui::InputText("##edit", value, flags);
            else changed = ImGui::InputTextWithHint("##edit", o.hint.c_str(), value, flags);
            ImGui::PopStyleColor();
            activeRing();
            min = ImGui::GetItemRectMin();
            max = ImGui::GetItemRectMax();
        }
        const LayoutSnapshot layout;
        ImGui::SetCursorScreenPos(ImVec2(max.x - arrow, min.y));
        ImGui::BeginDisabled(!o.enabled);
        if (ImGui::InvisibleButton("##drop", ImVec2(arrow, max.y - min.y))) ImGui::OpenPopup("##items");
        if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        ImGui::EndDisabled();
        ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(max.x - arrow, min.y + theme::crispPen(2)),
                                                  ImVec2(max.x - theme::crispPen(2), max.y - theme::crispPen(2)), theme::kBg);
        chevron(min, max, o.enabled);
        layout.restore();
        {
            const PopupLook look;
            const float w = std::max(max.x - min.x, popupWidth > 0.0f ? px(popupWidth) : 0.0f);
            ImGui::SetNextWindowPos(ImVec2(min.x, max.y));
            ImGui::SetNextWindowSizeConstraints(ImVec2(w, 0), ImVec2(w, px(360)));
            if (ImGui::BeginPopup("##items")) {
                if (items.empty()) {
                    ImGui::Dummy(px(0, 4));
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + px(8));
                    text("No entries", 12, theme::kNeutral600);
                    ImGui::Dummy(px(0, 4));
                }
                for (std::size_t i = 0; i < items.size(); ++i) {
                    const ComboItem& item = items[i];
                    ImGui::PushID(static_cast<int>(i));
                    const bool selected = item.value == *value;
                    bool clicked = false;
                    if (item.detail.empty() && !item.dimmed) {
                        clicked = popupItem(item.value, selected);
                    } else {
                        // the name, and its detail on a line of its own below
                        const ImVec2 p = ImGui::GetCursorScreenPos();
                        const float h = theme::snap(px(item.detail.empty() ? 26.0f : 40.0f));
                        clicked = ImGui::Selectable("##row", selected, ImGuiSelectableFlags_None, ImVec2(0, h));
                        const float rw = ImGui::GetItemRectSize().x;
                        ImDrawList* dl = ImGui::GetWindowDrawList();
                        const ImU32 ink = item.dimmed ? theme::kNeutral500 : theme::kText;
                        const float top = item.detail.empty() ? p.y : p.y + px(3);
                        const float nameH = item.detail.empty() ? h : px(19);
                        drawTextIn(dl, ImVec2(p.x + px(8), top), ImVec2(p.x + rw - px(8), top + nameH), item.value, 12, ink,
                                   selected ? Weight::SemiBold : Weight::Regular, 0.0f, 0.5f);
                        if (!item.detail.empty())
                            drawTextIn(dl, ImVec2(p.x + px(8), top + nameH), ImVec2(p.x + rw - px(8), p.y + h - px(3)),
                                       elideText(item.detail, rw - px(16), 11), 11, item.dimmed ? theme::kNeutral500 : theme::kNeutral600,
                                       Weight::Regular, 0.0f, 0.5f);
                        if (selected) ImGui::SetItemDefaultFocus();
                    }
                    if (clicked) {
                        if (*value != item.value) {
                            *value = item.value;
                            changed = true;
                        }
                        if (picked) *picked = true;
                        ImGui::CloseCurrentPopup();
                    }
                    ImGui::PopID();
                }
                ImGui::EndPopup();
            }
        }
        ImGui::PopID();
        return changed;
    }

    // --- rows -----------------------------------------------------------------

    Row beginRow(const char* id, float height, const RowOpts& o) {
        Row row;
        ImGui::PushID(id);
        row.min = ImGui::GetCursorScreenPos();
        const float w = lineWidth(o.width);
        const float h = theme::snap(px(height));
        row.max = ImVec2(row.min.x + w, row.min.y + h);
        ImGui::SetNextItemAllowOverlap();
        row.clicked = ImGui::InvisibleButton("##row", ImVec2(w, h));
        row.hovered = ImGui::IsItemHovered();
        row.doubleClicked = row.hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);
        row.rightClicked = ImGui::IsItemClicked(ImGuiMouseButton_Right);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        if (o.selected) dl->AddRectFilled(row.min, row.max, theme::kSurface);
        else if (row.hovered && o.hoverable) dl->AddRectFilled(row.min, row.max, theme::kNeutral200);
        else if (o.fill & 0xFF000000u) dl->AddRectFilled(row.min, row.max, o.fill);
        if (o.topRule > 0.0f)
            dl->AddRectFilled(row.min, ImVec2(row.max.x, row.min.y + theme::crispPen(o.topRule)), theme::kDivider);
        if (o.edge && o.selected) dl->AddRectFilled(row.min, ImVec2(row.min.x + theme::snap(px(3)), row.max.y), theme::kAccent);
        focusRing(dl, row.min, row.max);
        if (row.hovered && o.hoverable) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        ImGui::SetCursorScreenPos(row.min);
        return row;
    }

    void endRow(const Row& row) {
        ImGui::SetCursorScreenPos(ImVec2(row.min.x, row.max.y));
        // the cursor was moved by hand: tell the layout where the row ended
        ImGui::Dummy(ImVec2(0.0f, 0.0f));
        ImGui::SetCursorScreenPos(ImVec2(row.min.x, row.max.y));
        ImGui::PopID();
    }

    // --- panels ---------------------------------------------------------------

    void beginCard(const char* id, bool accent, float padding) {
        ImGui::PushID(id);
        CardState s;
        s.min = ImGui::GetCursorScreenPos();
        s.width = std::max(1.0f, ImGui::GetContentRegionAvail().x);
        s.padding = px(padding);
        s.accent = accent;
        cards().push_back(s);
        ImGui::SetCursorScreenPos(ImVec2(s.min.x + s.padding, s.min.y + s.padding));
        ImGui::BeginGroup();
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + s.width - 2 * s.padding);
    }

    void endCard() {
        if (cards().empty()) return;
        const CardState s = cards().back();
        cards().pop_back();
        ImGui::PopTextWrapPos();
        ImGui::EndGroup();
        const float bottom = ImGui::GetItemRectMax().y + s.padding;
        crispRect(ImGui::GetWindowDrawList(), s.min, ImVec2(s.min.x + s.width, bottom), s.accent ? theme::kAccent : theme::kDivider,
                  theme::kBorder);
        ImGui::SetCursorScreenPos(ImVec2(s.min.x, bottom));
        ImGui::Dummy(ImVec2(s.width, 0.0f));
        ImGui::PopID();
    }

} // namespace sirius::app::gui::widgets
