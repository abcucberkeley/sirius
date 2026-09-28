#ifndef SIRIUS_IMGUI_EXPORT_DIALOG_SUPPORT_HPP
#define SIRIUS_IMGUI_EXPORT_DIALOG_SUPPORT_HPP

// What the export, training export and preferences dialogs lay their forms
// out with: labelled fields side by side in equal columns, a spin box with
// a prefix inside its frame ("t 0", "level 6"), the
// wrapped 11 px notes, and the row of actions at the bottom (Cancel, then
// the primary action, flush right, Enter accepting as a default button does).
//
// Widths are display pixels here unless a name says otherwise; the controls
// of widgets/controls.hpp take design pixels, which design() converts to.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

#include <imgui.h>
#include <imgui_internal.h>

#include "imgui/theme.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui::dialog_support {

    inline float design(float displayPx) { return displayPx / std::max(theme::scale(), 0.01f); }

    // The width of one of `n` equal columns across the rest of the line,
    // `gap` design pixels apart.
    inline float columnWidth(int n, float gap) {
        const float avail = ImGui::GetContentRegionAvail().x;
        return std::max(theme::px(24), std::floor((avail - theme::px(gap) * static_cast<float>(n - 1)) / static_cast<float>(n)));
    }

    // Spacing between the items submitted while it lives (design pixels).
    class Spacing {
    public:
        Spacing(float x, float y) { ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, theme::px(x, y)); }
        ~Spacing() { ImGui::PopStyleVar(); }
        Spacing(const Spacing&) = delete;
        Spacing& operator=(const Spacing&) = delete;
    };

    // A group begun here lies on the line of the item before it when that
    // came with SameLine(), and Dear ImGui then lowers the first text of the
    // group to the baseline of that item's frame: a field label beside a
    // field would sit a frame padding below its neighbour's.
    inline void beginGroupAtTop() {
        ImGui::BeginGroup();
        ImGui::GetCurrentWindow()->DC.CurrLineTextBaseOffset = 0.0f;
    }

    // widgets::inputInt / inputDouble draw their arrows as items of their own
    // and put the cursor back by hand below the field. When a group or a
    // window ends right there, Dear ImGui takes that for a layout extended
    // with SetCursorPos and complains; one empty item at the bottom of what
    // was laid out settles it without adding any height.
    inline void settleCursor() {
        ImGuiWindow* w = ImGui::GetCurrentWindow();
        if (!w->DC.IsSetPos || w->DC.CursorPos.y <= w->DC.CursorMaxPos.y) return;
        ImGui::SetCursorScreenPos(ImVec2(w->DC.CursorPos.x, w->DC.CursorMaxPos.y));
        ImGui::Dummy(ImVec2(0.0f, 0.0f));
    }

    // A label with the editor(s) submitted while it lives below it, as one
    // item: `ImGui::SameLine()` after it puts the next field beside it.
    class Field {
    public:
        explicit Field(const std::string& label, const std::string& unit = {}) {
            beginGroupAtTop();
            widgets::fieldLabel(label, unit);
        }
        ~Field() {
            settleCursor();
            ImGui::EndGroup();
        }
        Field(const Field&) = delete;
        Field& operator=(const Field&) = delete;
    };

    // An 11 px note wrapped at the rest of the line.
    inline void note(const std::string& text, ImU32 color = theme::kNeutral600) {
        widgets::textWrapped(text, 11, color, theme::Weight::Regular, ImGui::GetContentRegionAvail().x);
    }

    // widgets::inputInt as one item: the last item it submits is its lower
    // arrow, so a SameLine() straight after it would start at the arrow's y.
    inline bool spinInt(const char* id, std::int64_t* value, std::int64_t lo, std::int64_t hi, std::int64_t step,
                        const widgets::FieldOpts& o) {
        beginGroupAtTop();
        const bool changed = widgets::inputInt(id, value, lo, hi, step, o);
        settleCursor();
        ImGui::EndGroup();
        return changed;
    }

    // A spin box `width` wide with `prefix` before the number.
    inline bool prefixedInt(const char* id, const std::string& prefix, std::int64_t* value, std::int64_t lo, std::int64_t hi,
                            float width, std::int64_t step = 1, bool enabled = true) {
        beginGroupAtTop();
        const ImVec2 pos = ImGui::GetCursorScreenPos();
        const ImVec2 ts = theme::textSize(prefix, 12);
        const float lead = ts.x + theme::px(5);
        widgets::drawTextIn(ImGui::GetWindowDrawList(), pos, ImVec2(pos.x + ts.x, pos.y + theme::px(theme::kInputH)), prefix, 12,
                            enabled ? theme::kNeutral600 : theme::withAlpha(theme::kNeutral600, 0.45f), theme::Weight::Regular, 0.0f,
                            0.5f);
        ImGui::SetCursorScreenPos(ImVec2(pos.x + lead, pos.y));
        widgets::FieldOpts o;
        o.width = design(std::max(theme::px(24), width - lead));
        o.enabled = enabled;
        const bool changed = widgets::inputInt(id, value, lo, hi, step, o);
        settleCursor();
        ImGui::EndGroup();
        return changed;
    }

    // A checkbox centred on the line of a 32 px input beside it.
    inline bool checkboxOnFieldLine(const char* label, bool* v, bool enabled = true) {
        ImGui::BeginGroup();
        const float box = std::max(theme::px(14), theme::textSize(std::string("X"), 12).y) + theme::px(4);
        ImGui::Dummy(ImVec2(0.0f, 0.0f));
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - ImGui::GetStyle().ItemSpacing.y +
                             std::max(0.0f, std::floor((theme::px(theme::kInputH) - box) * 0.5f)));
        const bool changed = widgets::checkbox(label, v, enabled);
        ImGui::EndGroup();
        return changed;
    }

    // The width widgets::button gives a full-size button with this label.
    inline float buttonWidth(const std::string& label, widgets::ButtonKind kind) {
        const bool primary = kind == widgets::ButtonKind::Primary;
        const float padX = kind == widgets::ButtonKind::Ghost ? 8.0f : 12.0f;
        // what is shown: the label up to its "##id"
        const std::string shown = label.substr(0, label.find("##"));
        return theme::snap(theme::textSize(shown, 13, primary ? theme::Weight::ExtraBold : theme::Weight::SemiBold).x +
                           2 * theme::px(padX) + 2 * theme::px(theme::kBorder));
    }

    enum class Action { None,
                        Cancel,
                        Accept };

    // "Cancel" (ghost) and the primary action, flush right. Enter accepts
    // when nothing is being edited, as a dialog's default button does.
    //
    // Not while a popup is open above the dialog: a message box shown over
    // it or one of its own dropdowns. Their focus counts as the dialog's (the
    // focus test follows the popup hierarchy), and the dialog is drawn before
    // them, so the Enter that answers the box started the export and dropped
    // the box unanswered.
    inline Action actionRow(const std::string& primary, bool enabled) {
        const float gap = theme::px(8);
        const float total = buttonWidth("Cancel", widgets::ButtonKind::Ghost) + gap + buttonWidth(primary, widgets::ButtonKind::Primary);
        const float avail = ImGui::GetContentRegionAvail().x;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, avail - total));
        Action action = Action::None;
        widgets::ButtonOpts cancel;
        cancel.kind = widgets::ButtonKind::Ghost;
        if (widgets::button("Cancel##dialogCancel", cancel)) action = Action::Cancel;
        ImGui::SameLine(0.0f, gap);
        if (widgets::primaryButton((primary + "##dialogAccept").c_str(), 0.0f, enabled)) action = Action::Accept;
        if (action == Action::None && enabled && !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId) &&
            ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && !ImGui::IsAnyItemActive() &&
            (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false)))
            action = Action::Accept;
        return action;
    }

} // namespace sirius::app::gui::dialog_support

#endif // SIRIUS_IMGUI_EXPORT_DIALOG_SUPPORT_HPP
