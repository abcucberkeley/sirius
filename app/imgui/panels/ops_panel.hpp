#ifndef SIRIUS_IMGUI_OPS_PANEL_HPP
#define SIRIUS_IMGUI_OPS_PANEL_HPP

// Left dock: "OPERATIONS · ANY ORDER" header, the step rows (enable box /
// pin, name + kind label, cache glyph + summary, ▲▼, ◉), the "Add a
// processing step" row with its grouped dropdown, the legend and the
// Run all / Export footer. (app/qt/panels/ops_panel.cpp)

#include <memory>

namespace sirius::app::gui {

    class App;

    class OpsPanel {
    public:
        explicit OpsPanel(App& app);
        ~OpsPanel();
        OpsPanel(const OpsPanel&) = delete;
        OpsPanel& operator=(const OpsPanel&) = delete;

        // The panel's content, inside the dock window the application began.
        void draw();
        void openAddMenu();                 // Process ▸ Add operation…

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_OPS_PANEL_HPP
