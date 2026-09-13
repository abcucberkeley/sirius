#ifndef SIRIUS_APP_QT_VIEWER_TRACK_OVERLAY_HPP
#define SIRIUS_APP_QT_VIEWER_TRACK_OVERLAY_HPP

// Trajectories of a tracked label volume drawn over a slice pane: each track
// is the path of its centroid through time, in the colour its mask has, so a
// track that breaks and restarts under a new id shows as a change of colour
// along one path. The part up to the time point on screen is solid, the part
// still to come faint, a frame the track is missing from is a dotted segment,
// and a dot marks where the track is now.
//
// Built once per label change (trackPaths), painted on every repaint
// (paintTrackPaths): the pane never walks the label index itself.

#include <cstdint>
#include <functional>

#include <QColor>
#include <QPointF>
#include <QRectF>
#include <QVector>

#include "core/tracks.hpp"
#include "qt/viewer/slice_pane.hpp"

class QPainter;

namespace sirius::app {

    struct TrackPath {
        std::uint32_t id = 0;
        QColor color;
        QVector<QPointF> points;   // voxel coordinates of the pane's plane (column, row)
        QVector<Index> frames;     // the time point of each point
        QRectF bounds;             // of the points, for skipping paths off screen
    };

    // The paths as seen by a pane of `kind`: XY and MIP draw (x, y), XZ (x, z)
    // and YZ (z, y), matching the panes' column and row axes. `only` (non-zero)
    // builds that track alone.
    QVector<TrackPath> trackPaths(const TrackIndex& index, SlicePane::Kind kind, std::uint32_t only = 0);

    struct TrackPaintOptions {
        Index t = 0;                    // the time point on screen
        std::uint32_t selected = 0;     // drawn heavier; the others dim while one is selected
        Index tail = 0;                 // frames of history drawn before t; 0 = all
        bool future = true;             // draw the part after t, faint
    };

    // `toScreen` maps the pane's voxel coordinates to widget pixels (SlicePane::toScreen);
    // `visible` is the widget rectangle.
    void paintTrackPaths(QPainter& p, const QVector<TrackPath>& paths, const TrackPaintOptions& options,
                         const std::function<QPointF(const QPointF&)>& toScreen, const QRectF& visible);

} // namespace sirius::app

#endif // SIRIUS_APP_QT_VIEWER_TRACK_OVERLAY_HPP
