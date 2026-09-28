#ifndef SIRIUS_IMGUI_VIEWER_CONSTANTS_HPP
#define SIRIUS_IMGUI_VIEWER_CONSTANTS_HPP

// The numbers the viewer's files share: the zoom steps and clamps of
// docs/design/README.md ("Clamp zoom 0.5-16x"), the playback rate ("loops
// at ~8 fps"), the ortho pane targets ("Grid 1fr 220px / 1fr 170px, 2 px
// gaps") and the insets the pane overlays are drawn at. Values only used
// inside one file stay there; these are the ones two files would otherwise
// disagree about. Theme tokens (colours, fonts, dock metrics) live in
// imgui/theme.hpp -- this header is the viewer's own geometry, in design
// pixels. (app/qt/viewer/viewer_constants.hpp)

#include <sirius/buffer.hpp>

namespace sirius::app::gui::viewer {

    // --- zoom ---------------------------------------------------------------
    // One wheel step multiplies the zoom by this; the +/- buttons use the
    // larger factor.
    inline constexpr double kWheelZoomBase = 1.15;
    inline constexpr double kButtonZoomFactor = 1.5;
    inline constexpr double kMinZoom = 0.5, kMaxZoom = 16.0;
    // The 3D camera has its own, wider, range (distance, not pixels).
    inline constexpr double kMinVolumeZoom = 0.25, kMaxVolumeZoom = 8.0;

    // --- playback -----------------------------------------------------------
    inline constexpr int kPlayIntervalMs = 125;   // ~8 fps

    // --- ortho layout -------------------------------------------------------
    // Targets for the 1600 x 960 default window; the splitters start here and
    // the user rebalances them from there.
    inline constexpr int kYzWidth = 220;
    inline constexpr int kXzHeight = 170;
    inline constexpr int kPaneGap = 2;            // 2 px gaps on neutral-900
    inline constexpr int kSidePaneMin = 80;       // YZ / XZ never vanish
    inline constexpr int kMainPaneMin = 160;      // nor does XY

    // --- pane overlays ------------------------------------------------------
    inline constexpr double kOverlayInset = 10.0;     // label / hint from the left
    inline constexpr double kOverlayTop = 8.0;        // label from the top
    inline constexpr double kOverlayBottom = 8.0;     // hint / scale bar from the bottom
    inline constexpr double kOverlayGap = 12.0;       // between the label's two parts
    inline constexpr double kScaleBarMaxPx = 140.0;   // longest scale bar drawn

    // --- brush --------------------------------------------------------------
    inline constexpr int kBrushMinPx = 2, kBrushMaxPx = 60, kBrushStepPx = 2;

    // --- 3D textures --------------------------------------------------------
    // The ray caster's volume textures are reduced to at most this many
    // texels per axis, for at most this many channels.
    inline constexpr Index kVolumeTexelsMax = 256;
    inline constexpr int kVolumeMaxChannels = 4;

} // namespace sirius::app::gui::viewer

#endif // SIRIUS_IMGUI_VIEWER_CONSTANTS_HPP
