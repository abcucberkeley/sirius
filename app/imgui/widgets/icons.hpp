#ifndef SIRIUS_IMGUI_WIDGETS_ICONS_HPP
#define SIRIUS_IMGUI_WIDGETS_ICONS_HPP

// The icon set of docs/design/README.md ("Icons: Lucide (thin, 1.5-2 px
// stroke)"), drawn into an ImDrawList from a table of paths on Lucide's
// 24 x 24 grid: flat strokes, no gradients, no radius, one colour. Drawn,
// not taken from a font, so the glyphs are the same whatever fonts the
// platform has.

#include <imgui.h>

namespace sirius::app::gui {

    enum class Icon {
        None,
        // viewer tools
        Navigate,
        Probe,
        Measure,
        Roi,
        Brush,
        Prompt,   // a pointer at an object: the Prompt tool
        // label cleanup tools
        Erase,
        Fill,
        Pick,
        Merge,
        Split,
        Lasso,
        // zoom / view
        Plus,
        Minus,
        ZoomIn,
        ZoomOut,
        Fit,
        Eye,
        Pin,
        // transport and dock chrome
        Play,
        Pause,
        Maximize,
        Float,
        Dock,
        ChevronUp,
        ChevronDown,
        ChevronRight,
        // panels
        Trash,
        Sparkle,
        Pencil,
        Info,
        Check,
        Close,
        Enter,
        Recompute,
        More,
        Help,
        Server,   // a rack of two: the cluster
    };

    // Draws `icon` centred in the box (min, max), scaled from the 24 x 24
    // design grid, with a `strokePx` pen (display pixels) in `colour`.
    void drawIcon(ImDrawList* dl, ImVec2 min, ImVec2 max, Icon icon, ImU32 colour, float strokePx = 1.5f);
    // The same in a square of `side` display pixels centred on `centre`.
    void drawIcon(ImDrawList* dl, ImVec2 centre, float side, Icon icon, ImU32 colour, float strokePx = 1.5f);

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_WIDGETS_ICONS_HPP
