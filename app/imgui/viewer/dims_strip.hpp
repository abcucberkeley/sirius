#ifndef SIRIUS_IMGUI_VIEWER_DIMS_STRIP_HPP
#define SIRIUS_IMGUI_VIEWER_DIMS_STRIP_HPP

// The strip below the viewer: Z with its µm readout, slider and "n / max";
// T with the 20 x 20 play / pause button, the seconds readout, slider and
// "n / max". The T row hides itself when the data has one time point.
// (app/qt/viewer/dims_strip.*)

#include <functional>

#include <imgui.h>

#include <sirius/buffer.hpp>

namespace sirius::app::gui {

    class DimsStrip {
    public:
        // Extents and physical scales; t <= 1 hides the T row.
        void setExtents(Index nz, Index nt, double dzUm, double frameIntervalS);
        void setPosition(Index z, Index t);
        void setPlaying(bool on) { playing_ = on; }
        bool playing() const noexcept { return playing_; }

        // The strip's height for the current extents, display pixels.
        float height() const;
        // Draws the strip in (min, max) of the current window.
        void draw(ImVec2 min, ImVec2 max);

        std::function<void(Index)> zRequested;
        std::function<void(Index)> tRequested;
        std::function<void(bool)> playToggled;

    private:
        Index nz_ = 1, nt_ = 1, z_ = 0, t_ = 0;
        double dz_ = 0.0, dt_ = 0.0;
        bool playing_ = false;
    };

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_VIEWER_DIMS_STRIP_HPP
