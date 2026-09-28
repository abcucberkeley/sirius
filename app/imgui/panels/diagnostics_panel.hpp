#ifndef SIRIUS_IMGUI_DIAGNOSTICS_PANEL_HPP
#define SIRIUS_IMGUI_DIAGNOSTICS_PANEL_HPP

// Bottom dock content: header (▼/▶ toggle, "DIAGNOSTICS · <step>", tab row,
// hint, ▁ ❐ ⛶ controls) and the per-kind body built from the selected
// step's Diagnostics (image cells, table, curves, histograms, facts) plus
// the dedicated segmentation-cleanup, track and volume panels.
// (app/qt/panels/diagnostics_panel.cpp, diagnostic_cells.cpp, track_table.cpp)
//
// Docked, floating and maximised are the application's business
// (App::floatDiagnostics, App::dockDiagnostics,
// App::setDiagnosticsMaximized): the header's controls call those.

#include <memory>

namespace sirius::app::gui {

    class App;

    class DiagnosticsPanel {
    public:
        explicit DiagnosticsPanel(App& app);
        ~DiagnosticsPanel();
        DiagnosticsPanel(const DiagnosticsPanel&) = delete;
        DiagnosticsPanel& operator=(const DiagnosticsPanel&) = delete;

        void draw();
        // Collapsed: only the 34 px header is drawn.
        bool isCollapsed() const;
        void setCollapsed(bool collapsed);
        void setTab(int index);
        int tabCount() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_DIAGNOSTICS_PANEL_HPP
