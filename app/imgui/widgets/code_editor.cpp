#include "imgui/widgets/code_editor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

#include <imgui_internal.h>

#include <TextEditor.h>

#include "imgui/strings.hpp"
#include "imgui/theme.hpp"

namespace sirius::app::gui::widgets {

    using theme::px;

    namespace {

        constexpr float kCodePx = 13;   // the editor's monospace size

        // Python as the editor ships it, plus the names worth singling out:
        // the builtins (and numpy) in a quieter ink, and the
        // plugin contract (STEP, run) in the accent.
        const TextEditor::Language* python() {
            static TextEditor::Language language = [] {
                TextEditor::Language l = *TextEditor::Language::Python();
                l.name = "Python";
                for (const char* b : {"abs", "all", "any", "dict", "enumerate", "float", "int", "isinstance", "len", "list", "map",
                                      "max", "min", "print", "range", "round", "set", "sorted", "str", "sum", "tuple", "zip", "np",
                                      "numpy"})
                    l.identifiers.insert(b);
                for (const char* entry : {"STEP", "run"}) l.declarations.insert(entry);
                return l;
            }();
            return &language;
        }

        // The theme's colours in the editor's palette (keywords accent-700, numbers accent-600,
        // strings neutral-700, comments neutral-500, the contract accent).
        TextEditor::Palette palette() {
            TextEditor::Palette p = TextEditor::GetLightPalette();
            auto set = [&p](TextEditor::Color c, ImU32 v) { p[static_cast<std::size_t>(c)] = v; };
            using C = TextEditor::Color;
            set(C::text, theme::kText);
            set(C::keyword, theme::kAccent700);
            set(C::declaration, theme::kAccent);
            set(C::number, theme::kAccent600);
            set(C::string, theme::kNeutral700);
            set(C::punctuation, theme::kText);
            set(C::preprocessor, theme::kNeutral600);
            set(C::identifier, theme::kText);
            set(C::knownIdentifier, theme::kNeutral800);
            set(C::comment, theme::kNeutral500);
            // transparent: the gutter and the paper are drawn underneath
            set(C::background, theme::kTransparent);
            set(C::cursor, theme::kText);
            set(C::selection, theme::withAlpha(theme::kAccent, 0.22f));
            set(C::whitespace, theme::kNeutral300);
            set(C::matchingBracketBackground, theme::kNeutral300);
            set(C::matchingBracketActive, theme::kAccent);
            set(C::matchingBracketLevel1, theme::kText);
            set(C::matchingBracketLevel2, theme::kNeutral700);
            set(C::matchingBracketLevel3, theme::kNeutral600);
            set(C::matchingBracketError, theme::kAccent);
            set(C::lineNumber, theme::kNeutral500);
            set(C::currentLineNumber, theme::kText);
            set(C::currentLineHighlight, theme::kSurface);
            set(C::currentLineHighlightBorder, theme::kTransparent);
            return p;
        }

        // The first `glyphs` code points of `line`.
        std::string prefixOf(const std::string& line, std::size_t glyphs) {
            std::size_t i = 0;
            for (std::size_t n = 0; n < glyphs && i < line.size(); ++n) nextCodepoint(line, i);
            return line.substr(0, i);
        }

    } // namespace

    struct CodeEditor::Impl {
        TextEditor editor;
        std::string placeholder;
        bool changed = false;
        bool swallowChange = false;   // setText is not an edit
        bool focused = false;

        Impl() {
            editor.SetLanguage(python());
            editor.SetPalette(palette());
            editor.SetTabSize(4);
            editor.SetInsertSpacesOnTabs(true);
            // The editor's own auto-indent adds a tab after '{' and '[' and
            // nothing after ':'; Python wants the opposite, so the indent is
            // done here (draw()).
            editor.SetAutoIndentEnabled(false);
            editor.SetShowWhitespacesEnabled(false);
            editor.SetShowLineNumbersEnabled(true);
            editor.SetShowScrollbarMiniMapEnabled(false);
            editor.SetShowPanScrollIndicatorEnabled(false);
            editor.SetCompletePairedGlyphs(false);
            editor.SetLineFoldingEnabled(false);
            editor.SetShowMatchingBrackets(true);
            editor.SetLineNumberLeftMargin(1);
            editor.SetTextLeftMargin(2);
            editor.SetChangeCallback([this] { changed = true; }, 0);
        }
    };

    CodeEditor::CodeEditor() : impl_(std::make_unique<Impl>()) {}
    CodeEditor::~CodeEditor() = default;

    void CodeEditor::setText(const std::string& text) {
        impl_->editor.SetText(text);
        impl_->editor.SetCursor(TextEditor::DocPos(0, 0));
        impl_->swallowChange = true;
        impl_->changed = false;
    }

    std::string CodeEditor::text() const { return impl_->editor.GetText(); }

    void CodeEditor::setReadOnly(bool on) {
        impl_->editor.SetReadOnlyEnabled(on);
        // the current line is only marked where one can type
        impl_->editor.SetShowCurrentLineHighlightEnabled(!on);
    }

    bool CodeEditor::readOnly() const { return impl_->editor.IsReadOnlyEnabled(); }
    void CodeEditor::setPlaceholder(const std::string& text) { impl_->placeholder = text; }
    void CodeEditor::focus() { impl_->editor.SetFocus(); }
    bool CodeEditor::focused() const { return impl_->focused; }

    bool CodeEditor::draw(const char* id, ImVec2 size) {
        Impl& d = *impl_;
        TextEditor& ed = d.editor;
        const theme::FontScope font(kCodePx, theme::mono());
        const ImVec2 min = ImGui::GetCursorScreenPos();
        const ImVec2 max(min.x + size.x, min.y + size.y);
        ImDrawList* dl = ImGui::GetWindowDrawList();

        // The paper, and the gutter on the surface as wide as the numbers
        // the editor draws (its margins are counted in glyphs).
        const float glyph = ImGui::CalcTextSize("#").x;
        const std::size_t lines = std::max<std::size_t>(ed.GetLineCount(), 1);
        const float digits = std::floor(std::log10(static_cast<float>(lines + 1))) + 1.0f;
        const float gutter = std::floor((1.0f + digits) * glyph + glyph * 0.8f);
        dl->AddRectFilled(min, max, theme::kBg);
        dl->AddRectFilled(min, ImVec2(std::min(max.x, min.x + gutter), max.y), theme::kSurface);

        // What Enter is about to split, for the indent that follows it.
        const bool enter = d.focused && !ed.IsReadOnlyEnabled() && ImGui::GetIO().KeyMods == ImGuiMod_None &&
                           (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter));
        TextEditor::DocPos before;
        std::string head;
        if (enter) {
            const TextEditor::DocSelection sel = ed.GetCurrentCursorSelection();
            before = sel.start < sel.end ? sel.start : sel.end;
            head = prefixOf(ed.GetLineText(before.line), before.index);
        }

        const ImGuiID childId = ImGui::GetID(id);
        ed.Render(id, size, ImGuiChildFlags_None, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_HorizontalScrollbar);
        {
            const ImGuiContext& g = *ImGui::GetCurrentContext();
            d.focused = g.NavWindow && (g.NavWindow->ChildId == childId ||
                                        (g.NavWindow->ParentWindow && g.NavWindow->ParentWindow->ChildId == childId));
        }

        if (enter && d.changed) {
            // the new line takes the indentation of the one it was split
            // from, one level more after a ':'
            const TextEditor::DocPos now = ed.GetCurrentCursorPosition();
            if (now.line == before.line + 1 && now.index == 0) {
                std::string indent;
                for (char c : head) {
                    if (c == ' ' || c == '\t') indent += c;
                    else break;
                }
                const std::string t = trimmed(head);
                if (!t.empty() && t.back() == ':') indent += "    ";
                if (!indent.empty()) ed.ReplaceTextInCurrentCursor(indent);
            }
        }

        if (ed.IsEmpty() && !d.placeholder.empty()) {
            const ImVec2 at(min.x + gutter + glyph * 1.2f, min.y);
            dl->AddText(ImGui::GetFont(), ImGui::GetFontSize(), at, theme::kNeutral500, d.placeholder.c_str());
        }

        const bool changed = d.changed && !d.swallowChange;
        d.changed = false;
        d.swallowChange = false;
        return changed;
    }

} // namespace sirius::app::gui::widgets
