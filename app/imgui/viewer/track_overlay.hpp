#ifndef SIRIUS_IMGUI_VIEWER_TRACK_OVERLAY_HPP
#define SIRIUS_IMGUI_VIEWER_TRACK_OVERLAY_HPP

// Trajectories of a tracked label volume drawn over a slice pane: each track
// is the path of its centroid through time, in the colour its mask has, so a
// track that breaks and restarts under a new id shows as a change of colour
// along one path. The part up to the time point on screen is solid, the part
// still to come faint, a frame the track is missing from is a dotted segment,
// and a dot marks where the track is now.
//
// Built once per label change (trackPaths), drawn every frame
// (paintTrackPaths): the pane never walks the label index itself.
// (app/qt/viewer/track_overlay.*)

#include <cstdint>
#include <functional>
#include <vector>

#include <imgui.h>

#include "core/tracks.hpp"

namespace sirius::app::gui {

    // A point of a pane's plane in voxels (column, row), or on screen.
    struct DPoint {
        double x = 0.0, y = 0.0;
        DPoint() = default;
        DPoint(double px, double py) : x(px), y(py) {}
        DPoint operator+(const DPoint& o) const noexcept { return {x + o.x, y + o.y}; }
        DPoint operator-(const DPoint& o) const noexcept { return {x - o.x, y - o.y}; }
        DPoint operator*(double f) const noexcept { return {x * f, y * f}; }
    };

    // An axis-aligned rectangle of voxels; "null" (the Qt sense) when it has
    // neither width nor height.
    struct DRect {
        double x0 = 0.0, y0 = 0.0, x1 = 0.0, y1 = 0.0;
        double width() const noexcept { return x1 - x0; }
        double height() const noexcept { return y1 - y0; }
        bool isNull() const noexcept { return width() == 0.0 && height() == 0.0; }
        // The rectangle spanned by two corners, in any order.
        static DRect spanning(const DPoint& a, const DPoint& b) {
            DRect r;
            r.x0 = a.x < b.x ? a.x : b.x;
            r.x1 = a.x < b.x ? b.x : a.x;
            r.y0 = a.y < b.y ? a.y : b.y;
            r.y1 = a.y < b.y ? b.y : a.y;
            return r;
        }
    };

    struct TrackPath {
        std::uint32_t id = 0;
        ImU32 color = 0;
        std::vector<DPoint> points;   // voxel coordinates of the pane's plane (column, row)
        std::vector<Index> frames;    // the time point of each point
        std::vector<double> depths;   // each point along the axis the plane is a slice of (voxels)
        DRect bounds;                 // of the points, for skipping paths off screen
    };

    // Which plane a pane shows, as columns and rows: XY (and the z projection)
    // is (x, y), XZ is (x, z), YZ is (z, y).
    enum class TrackPlane { XY,
                            XZ,
                            YZ };

    // The paths as seen in `plane`. `only` (non-zero) builds that track alone.
    std::vector<TrackPath> trackPaths(const TrackIndex& index, TrackPlane plane, std::uint32_t only = 0);

    struct TrackPaintOptions {
        Index t = 0;                    // the time point on screen
        std::uint32_t selected = 0;     // drawn heavier; the others dim while one is selected
        Index tail = 0;                 // frames of history drawn before t; 0 = all
        bool future = true;             // draw the part after t, faint
        // A slice shows the tracks near it: a path is drawn when, at t (or the
        // closest frame it exists in), its centroid lies within `depthRange`
        // voxels of `depth`. A negative range draws every path (projections).
        double depth = 0.0;
        double depthRange = -1.0;
    };

    // `toScreen` maps the pane's voxel coordinates to display pixels
    // (SlicePane::toScreenAbs); `visibleMin` / `visibleMax` is the pane.
    // Widths are design pixels.
    void paintTrackPaths(ImDrawList* dl, const std::vector<TrackPath>& paths, const TrackPaintOptions& options,
                         const std::function<ImVec2(const DPoint&)>& toScreen, ImVec2 visibleMin, ImVec2 visibleMax);

    // A polyline drawn in dashes (`dash` on, `gap` off, display pixels).
    void dashedPolyline(ImDrawList* dl, const std::vector<ImVec2>& points, ImU32 color, float thickness, float dash, float gap);

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_VIEWER_TRACK_OVERLAY_HPP
