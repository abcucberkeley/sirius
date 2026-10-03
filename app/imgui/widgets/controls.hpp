#ifndef SIRIUS_IMGUI_WIDGETS_CONTROLS_HPP
#define SIRIUS_IMGUI_WIDGETS_CONTROLS_HPP

// The controls of docs/design as immediate-mode functions: the segmented
// control / tile row (outlined options, selected = ink fill or accent fill),
// the square icon button (the view / help / dock chrome and the tool strip),
// the buttons (primary / secondary / ghost / link / chip), caption labels
// (10 px uppercase, 0.1 em tracking), rules between regions, the 14 px token
// checkbox, sliders, the underlined tab row, the labelled input fields and a
// clickable row with hover / selected states. Everything draws from theme::
// tokens and the icon table in widgets/icons.hpp.
//
// Conventions:
//   * sizes and font sizes are design pixels (theme::px() scales them);
//     positions handed to or read from Dear ImGui are display pixels;
//   * every widget takes a string id / label with the usual "##id" rules and
//     returns true on the frame the user changed or pressed it;
//   * a width of 0 means "as wide as the content", a negative width "the
//     rest of the line" (like ImGui's -FLT_MIN).

#include <imgui.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "imgui/theme.hpp"
#include "imgui/widgets/icons.hpp"

namespace sirius::app::gui::widgets {

    // --- text ----------------------------------------------------------------
    void text(const std::string& s, float px = theme::kBodyPx, ImU32 color = theme::kText,
              theme::Weight w = theme::Weight::Regular);
    // Wrapped at `wrapWidth` display pixels from the cursor (0 = the rest of the line).
    void textWrapped(const std::string& s, float px = theme::kBodyPx, ImU32 color = theme::kText,
                     theme::Weight w = theme::Weight::Regular, float wrapWidth = 0.0f);
    void heading(const std::string& s, float px);                                  // weight 800
    void caption(const std::string& s, ImU32 color = theme::kNeutral600);          // 10 px, caption case, tracked
    void mono(const std::string& s, float px = 11, ImU32 color = theme::kNeutral700);
    // `s` cut to `width` display pixels with an ellipsis; the tooltip says
    // the whole text when it was cut. Width <= 0: the rest of the line.
    void elided(const std::string& s, float width = 0.0f, float px = theme::kBodyPx, ImU32 color = theme::kText,
                theme::Weight w = theme::Weight::Regular);
    std::string elideText(const std::string& s, float width, float px = theme::kBodyPx,
                          theme::Weight w = theme::Weight::Regular);
    // Text drawn straight into a draw list at `pos` (display pixels).
    void drawText(ImDrawList* dl, ImVec2 pos, const std::string& s, float px, ImU32 color,
                  theme::Weight w = theme::Weight::Regular);
    // Centred in (min, max), or aligned: ax, ay in 0..1 (0 = left / top).
    void drawTextIn(ImDrawList* dl, ImVec2 min, ImVec2 max, const std::string& s, float px, ImU32 color,
                    theme::Weight w = theme::Weight::Regular, float ax = 0.5f, float ay = 0.5f);

    // --- layout ---------------------------------------------------------------
    void rule(float px = theme::kRule, ImU32 color = theme::kDivider);      // across the rest of the line
    void ruleAt(ImDrawList* dl, ImVec2 a, ImVec2 b, float px = theme::kRule, ImU32 color = theme::kDivider);
    void vspace(float designPx);
    void colorChip(ImU32 color, float w = 10, float h = 10);
    // Tooltip for the last item (12 px, ink border), shown after the usual delay.
    void tooltip(const std::string& s);
    // "Move step up" + "Alt+Up" -> "Move step up (Alt+Up)".
    std::string withShortcut(const std::string& text, const std::string& shortcut);

    // --- messages to copy --------------------------------------------------------
    // An error, a failure's detail, a reason something cannot run: text the
    // user pastes into a report or a search. Every such message is drawn
    // with these, the same way everywhere.
    //
    // Wrapped text (`wrapWidth` display pixels in all, the button included;
    // 0 = the rest of the line) with a small Copy button at the end of its
    // first line, and "Copy" on a right-click of the text. `copied` is what
    // goes to the clipboard ("" = the text itself). The cursor ends where
    // textWrapped leaves it.
    void copyableText(const char* id, const std::string& s, float px = theme::kBodyPx, ImU32 color = theme::kText,
                      theme::Weight w = theme::Weight::Regular, float wrapWidth = 0.0f, const std::string& copied = {});
    // Its height at `wrapWidth`, for a layout placed by hand.
    float copyableTextHeight(const std::string& s, float px, float wrapWidth, theme::Weight w = theme::Weight::Regular);
    // The small Copy button alone (a `side` design-px square, borderless):
    // copies `text`, then shows a tick and "Copied" for a moment. True on the
    // click.
    bool copyButton(const char* id, const std::string& text, float side = 16.0f, bool onDark = false);
    // A right-click on the last item offers "Copy" of `text` (and the
    // `more` entries after it: label, text to copy).
    void copyOnRightClick(const char* id, const std::string& text,
                          const std::vector<std::pair<std::string, std::string>>& more = {});
    // Puts `text` on the clipboard (the one place every copy goes through).
    void copyToClipboard(const std::string& text);

    // --- buttons -------------------------------------------------------------
    enum class ButtonKind { Secondary,   // ink outline (the default look)
                            Primary,     // accent fill, paper text, 800
                            Ghost,       // text only
                            Link,        // 11 px accent text, no box
                            Chip };      // 11 px, thin outline
    struct ButtonOpts {
        ButtonKind kind = ButtonKind::Secondary;
        bool small = false;          // 12 px, tighter padding
        bool tiny = false;           // 11 px
        float width = 0.0f;          // design px; 0 = content, < 0 = rest of the line
        bool enabled = true;
        bool centered = false;       // labels are flush left unless asked
        std::string tooltip;
    };
    bool button(const char* label, const ButtonOpts& opts = {});
    inline bool primaryButton(const char* label, float width = 0.0f, bool enabled = true) {
        ButtonOpts o;
        o.kind = ButtonKind::Primary;
        o.width = width;
        o.enabled = enabled;
        return button(label, o);
    }
    inline bool ghostButton(const char* label, bool small = true, bool enabled = true) {
        ButtonOpts o;
        o.kind = ButtonKind::Ghost;
        o.small = small;
        o.enabled = enabled;
        return button(label, o);
    }
    inline bool linkButton(const char* label, bool enabled = true) {
        ButtonOpts o;
        o.kind = ButtonKind::Link;
        o.enabled = enabled;
        return button(label, o);
    }
    inline bool chipButton(const char* label, bool enabled = true) {
        ButtonOpts o;
        o.kind = ButtonKind::Chip;
        o.enabled = enabled;
        return button(label, o);
    }

    // "Ortho | 3D | Compare", "CUDA | CPU | HPC", "sum | mean | max | min".
    struct SegmentedOpts {
        // Tiles: equal-width, 36 px high, 13 px text (Backend / Cache rows).
        // Segmented: compact, 26 px, 12 px text, hugs its content.
        bool tiles = false;
        bool accent = false;                         // selected drawn with the accent instead of ink
        float width = -1.0f;                         // tiles only: design px, < 0 = rest of the line
        bool enabled = true;
        std::vector<bool> optionEnabled;             // empty = all
        std::vector<std::string> tooltips;           // empty = none
    };
    bool segmented(const char* id, const std::vector<std::string>& options, int* current, const SegmentedOpts& opts = {});
    // The width a (non-tile) segmented control takes, display pixels.
    float segmentedWidth(const std::vector<std::string>& options);

    // Square button carrying one drawn icon (or a short text glyph): 1.5 px
    // border, accent fill when active.
    struct GlyphOpts {
        bool active = false;         // accent fill + paper icon
        bool enabled = true;
        bool borderless = false;     // no border when idle (the reorder chevrons)
        bool dashed = false;         // dashed border (the "+" add square)
        bool onDark = false;         // sitting on the viewer ground
        bool dimmed = false;         // available, but not the one in play
        ImU32 idle = theme::kText;
        ImU32 border = theme::kDivider;
        float iconPx = 0.0f;         // side of the icon's box; 0 scales it to the button
        float glyphPx = 11.0f;       // text glyph size
        std::string tooltip;
    };
    bool glyphButton(const char* id, Icon icon, float size = 24, const GlyphOpts& opts = {});
    bool glyphButton(const char* id, Icon icon, ImVec2 size, const GlyphOpts& opts = {});
    bool glyphTextButton(const char* id, const char* glyph, ImVec2 size, const GlyphOpts& opts = {});

    // 14 x 14 square box (accent fill when checked) + 12 px label + optional
    // 10 px uppercase caption ("LOCKED"). Returns true when clicked; the
    // caller owns the state.
    bool tokenCheck(const char* label, bool checked, const char* caption = nullptr, bool enabled = true,
                    bool onDark = false);
    // The same, toggling `*v`.
    bool checkbox(const char* label, bool* v, bool enabled = true);
    bool radio(const char* label, bool selected, bool enabled = true);
    // 22 x 22 channel square labelled with the wavelength: filled when the
    // channel is visible, outlined when hidden.
    bool channelSwatch(const char* id, const std::string& label, ImU32 color, bool on, const std::string& tip = {});

    // --- sliders ---------------------------------------------------------------
    // 2 px groove, accent up to the 10 x 14 handle. Width in design px (< 0: rest).
    struct SliderOpts {
        float width = -1.0f;
        bool enabled = true;
        bool onDark = false;
    };
    bool slider(const char* id, double* v, double lo, double hi, const SliderOpts& opts = {});
    bool sliderInt(const char* id, std::int64_t* v, std::int64_t lo, std::int64_t hi, const SliderOpts& opts = {});
    // Two handles over [0, 1] (the 3D clip bar).
    bool rangeSlider(const char* id, double* lo, double* hi, const SliderOpts& opts = {});

    // --- tabs ------------------------------------------------------------------
    // 12 px labels, the active one 800 with a 2 px accent underline.
    bool tabRow(const char* id, const std::vector<std::string>& tabs, int* current);

    // --- fields ---------------------------------------------------------------
    // The 11 px label above an input.
    void fieldLabel(const std::string& label, const std::string& unit = {});
    struct FieldOpts {
        float width = -1.0f;         // design px; < 0 = rest of the line
        float height = theme::kInputH;   // design px (single-line fields)
        bool enabled = true;
        bool readOnly = false;
        std::string hint;            // placeholder
        bool password = false;
        bool monospace = false;
        bool enterReturnsTrue = false;   // true only when Enter was pressed (else on every edit)
    };
    bool inputText(const char* id, std::string* value, const FieldOpts& opts = {});
    // For a field that held a secret (a password field's ImGui id, from
    // ImGui::GetItemID() after inputText): ends its editing and overwrites
    // the copies Dear ImGui keeps of its text (the edit buffer, the text to
    // revert to, the deactivated-field copy). The caller wipes its own string.
    void forgetInputText(ImGuiID id);
    bool inputTextMultiline(const char* id, std::string* value, float heightPx, const FieldOpts& opts = {});
    // Spin boxes; `step` 0 hides the arrows' effect (plain number field).
    bool inputInt(const char* id, std::int64_t* value, std::int64_t lo, std::int64_t hi, std::int64_t step = 1,
                  const FieldOpts& opts = {});
    bool inputDouble(const char* id, double* value, double lo, double hi, double step = 0.0, int decimals = -1,
                     const FieldOpts& opts = {});
    bool combo(const char* id, int* current, const std::vector<std::string>& items, const FieldOpts& opts = {});
    // A dropdown one can also type into (the assistant's model). Returns true
    // when the text changed (typed or picked).
    bool editableCombo(const char* id, std::string* value, const std::vector<std::string>& items, const FieldOpts& opts = {});
    // The same with a line of detail under each entry, and entries drawn
    // greyed (still pickable) when `dimmed`. `picked` (optional) is set when
    // the change came from the list rather than the keyboard; `popupWidth`
    // (design px, 0 = the field's) lets the list be wider than the field.
    struct ComboItem {
        std::string value;
        std::string detail;
        bool dimmed = false;
    };
    bool editableCombo(const char* id, std::string* value, const std::vector<ComboItem>& items, const FieldOpts& opts,
                       bool* picked, float popupWidth = 0.0f);
    // A dropdown of fixed entries like combo(), each with its line of detail
    // under it; a `dimmed` entry is drawn greyed and cannot be picked (its
    // detail says why; the whole of it is the entry's tooltip).
    bool detailCombo(const char* id, int* current, const std::vector<ComboItem>& items, const FieldOpts& opts = {},
                     float popupWidth = 0.0f);

    // --- rows -----------------------------------------------------------------
    // A row that answers a click anywhere on it and paints its own hover /
    // selected / accent-edge states. Widgets submitted between beginRow and
    // endRow lie on top of it and take their own clicks.
    struct RowOpts {
        bool selected = false;
        bool hoverable = true;
        bool edge = false;               // 3 px accent left edge when selected
        float topRule = 1.0f;            // design px, 0 = none
        ImU32 fill = theme::kTransparent;   // idle background
        float width = -1.0f;             // design px; < 0 = rest of the line
    };
    struct Row {
        bool clicked = false;
        bool hovered = false;
        bool doubleClicked = false;
        bool rightClicked = false;
        ImVec2 min, max;                 // display pixels
    };
    Row beginRow(const char* id, float height, const RowOpts& opts = {});
    void endRow(const Row& row);

    // --- panels ---------------------------------------------------------------
    // A bordered card (1.5 px divider or accent) around what is drawn between
    // begin and end; the content is laid out at `padding` from the border.
    void beginCard(const char* id, bool accent = false, float padding = 10.0f);
    void endCard();

    // The standard frame of a dashed rectangle (drop zones, the add square).
    void dashedRect(ImDrawList* dl, ImVec2 min, ImVec2 max, ImU32 color, float thickness = 1.5f, float dash = 4.0f);
    // A rectangle outline whose stroke lies inside (min, max) on whole pixels.
    void crispRect(ImDrawList* dl, ImVec2 min, ImVec2 max, ImU32 color, float designPx = theme::kBorder);

} // namespace sirius::app::gui::widgets

#endif // SIRIUS_IMGUI_WIDGETS_CONTROLS_HPP
