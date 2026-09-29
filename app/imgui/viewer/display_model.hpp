#ifndef SIRIUS_IMGUI_VIEWER_DISPLAY_MODEL_HPP
#define SIRIUS_IMGUI_VIEWER_DISPLAY_MODEL_HPP

// The viewer's display model lives in the core (core/display_model.hpp) so sirius-cli
// renders what the viewer draws; the GUI keeps its names.

#include "core/display_model.hpp"

namespace sirius::app::gui {
    using display::DisplayModel;
    using display::DisplayWindow;
    using display::Image;
    using display::RectI;
} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_VIEWER_DISPLAY_MODEL_HPP
