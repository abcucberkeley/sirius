#ifndef SIRIUS_IMGUI_PARAMS_PANEL_HPP
#define SIRIUS_IMGUI_PARAMS_PANEL_HPP

// Right dock: "STEP 05 · INTENSITY" kicker + state, step name + ? help
// button, the per-kind parameter body (generic form from ParamSpecs plus
// bespoke editors for Load, SIM, Einsum, Segmentation and Merge), the
// BACKEND and CACHE OUTPUT tile rows and the Run step / View / Remove
// footer. (app/qt/panels/params_panel.cpp)

#include <memory>

namespace sirius::app::gui {

    class App;

    class ParamsPanel {
    public:
        explicit ParamsPanel(App& app);
        ~ParamsPanel();
        ParamsPanel(const ParamsPanel&) = delete;
        ParamsPanel& operator=(const ParamsPanel&) = delete;

        void draw();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_PARAMS_PANEL_HPP
