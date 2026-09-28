#ifndef SIRIUS_IMGUI_HELP_WINDOW_HPP
#define SIRIUS_IMGUI_HELP_WINDOW_HPP

// Floating help page (520 x <= 760): "HELP · <step>", Edit page, ✕; the
// operation's Markdown + LaTeX page. (app/qt/panels/help_window.cpp)
//
// The window is its own: draw() begins and ends it (the application calls
// draw() every frame; nothing is drawn while it is hidden).

#include <memory>
#include <string>

namespace sirius::app::gui {

    class App;

    class HelpWindow {
    public:
        explicit HelpWindow(App& app);
        ~HelpWindow();
        HelpWindow(const HelpWindow&) = delete;
        HelpWindow& operator=(const HelpWindow&) = delete;

        void draw();
        void showKind(const std::string& kind);   // page for an operation kind; shows the window
        void showManual();                        // the general manual page
        void showShortcuts();                     // keyboard shortcuts page
        std::string currentKind() const;
        bool visible() const;
        void setVisible(bool on);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_HELP_WINDOW_HPP
