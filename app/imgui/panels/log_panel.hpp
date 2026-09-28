#ifndef SIRIUS_IMGUI_LOG_PANEL_HPP
#define SIRIUS_IMGUI_LOG_PANEL_HPP

// The session log as a dock: everything Workbench::logLine records.
// Monospace, selectable, copy and clear, and an auto-scroll that stops
// following as soon as the reader scrolls up.

#include <memory>

namespace sirius::app::gui {

    class App;

    class LogPanel {
    public:
        explicit LogPanel(App& app);
        ~LogPanel();
        LogPanel(const LogPanel&) = delete;
        LogPanel& operator=(const LogPanel&) = delete;

        void draw();
        // Scrolls to the newest line and resumes following.
        void showLatest();
        int lineCount() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_LOG_PANEL_HPP
