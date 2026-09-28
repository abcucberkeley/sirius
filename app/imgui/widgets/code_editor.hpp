#ifndef SIRIUS_IMGUI_WIDGETS_CODE_EDITOR_HPP
#define SIRIUS_IMGUI_WIDGETS_CODE_EDITOR_HPP

// A small code editor for plugin files: ImGuiColorTextEdit with a
// line-number gutter, the theme's monospace face, Python syntax colours
// from the theme and the few editing habits a quick edit needs (Tab = four
// spaces, auto-indent after ':', Ctrl+/ toggles comments). Not an IDE: no
// completion, no folding, no diagnostics. (app/qt/widgets/code_editor.cpp)
//
// The editor's own header stays out of this one: the dialogs that embed an
// editor only see this class.

#include <imgui.h>

#include <memory>
#include <string>

namespace sirius::app::gui::widgets {

    class CodeEditor {
    public:
        CodeEditor();
        ~CodeEditor();
        CodeEditor(const CodeEditor&) = delete;
        CodeEditor& operator=(const CodeEditor&) = delete;

        // Replaces the text (and the undo history); the cursor goes to the
        // start. Not a change: draw() does not report it.
        void setText(const std::string& text);
        std::string text() const;
        void setReadOnly(bool on);
        bool readOnly() const;
        // Shown in the empty editor.
        void setPlaceholder(const std::string& text);
        // Takes the keyboard next frame.
        void focus();

        // Draws the editor filling `size` (display pixels) at the cursor.
        // True on the frame the user changed the text (typing, paste, undo).
        bool draw(const char* id, ImVec2 size);
        // Whether the editor had the keyboard in the last frame drawn.
        bool focused() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app::gui::widgets

#endif // SIRIUS_IMGUI_WIDGETS_CODE_EDITOR_HPP
