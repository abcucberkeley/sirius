#include "qt/viewer/track_overlay.hpp"

#include "core/labels.hpp"   // labelColor

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

#include <QPainter>
#include <QPen>
#include <QPolygonF>

#include "qt/theme.hpp"

namespace sirius::app {

    namespace {
        QPointF planeOf(const TrackPoint& p, TrackPlane plane) {
            const double z = p.centroid[0], y = p.centroid[1], x = p.centroid[2];
            switch (plane) {
                case TrackPlane::XZ: return {x, z};
                case TrackPlane::YZ: return {z, y};
                case TrackPlane::XY: break;
            }
            return {x, y};
        }

        double depthOf(const TrackPoint& p, TrackPlane plane) {
            switch (plane) {
                case TrackPlane::XZ: return p.centroid[1];
                case TrackPlane::YZ: return p.centroid[2];
                case TrackPlane::XY: break;
            }
            return p.centroid[0];
        }

        // Where the path is at t, or at the frame closest to it.
        int nearestFrame(const TrackPath& path, Index t) {
            const auto it = std::lower_bound(path.frames.begin(), path.frames.end(), t);
            if (it == path.frames.end()) return path.frames.size() - 1;
            const int i = static_cast<int>(it - path.frames.begin());
            if (*it == t || i == 0) return i;
            return (t - path.frames[i - 1] <= *it - t) ? i - 1 : i;
        }

        // The pane's voxel (i, j) covers [i, i + 1): a centroid at 3.0 is the
        // middle of voxel 3 only after the half-voxel shift.
        constexpr double kVoxelCentre = 0.5;

        enum class Style { Past,
                           PastGap,
                           Future,
                           FutureGap };
    } // namespace

    QVector<TrackPath> trackPaths(const TrackIndex& index, TrackPlane plane, std::uint32_t only) {
        std::map<std::uint32_t, TrackPath> byId;
        index.forEachPoint([&](std::uint32_t id, const TrackPoint& point) {
            if (only && id != only) return;
            TrackPath& path = byId[id];
            const QPointF at = planeOf(point, plane) + QPointF(kVoxelCentre, kVoxelCentre);
            if (path.points.isEmpty()) {
                path.id = id;
                const auto c = labelColor(id);
                path.color = QColor::fromRgbF(c[0], c[1], c[2]);
                path.bounds = QRectF(at, QSizeF(0, 0));
            } else {
                path.bounds = path.bounds.united(QRectF(at, QSizeF(0, 0)));
            }
            path.points.push_back(at);
            path.frames.push_back(point.t);
            path.depths.push_back(depthOf(point, plane));
        });
        QVector<TrackPath> out;
        out.reserve(static_cast<int>(byId.size()));
        for (auto& [id, path] : byId) out.push_back(std::move(path));
        return out;
    }

    void paintTrackPaths(QPainter& p, const QVector<TrackPath>& paths, const TrackPaintOptions& options,
                         const std::function<QPointF(const QPointF&)>& toScreen, const QRectF& visible) {
        if (paths.isEmpty()) return;
        p.save();
        p.setRenderHint(QPainter::Antialiasing, true);
        p.setBrush(Qt::NoBrush);
        const QRectF margin = visible.adjusted(-8, -8, 8, 8);
        const Index oldest = options.tail > 0 ? options.t - options.tail : std::numeric_limits<Index>::min();

        // the selected track last, so it is on top
        QVector<const TrackPath*> order;
        order.reserve(paths.size());
        const TrackPath* selected = nullptr;
        for (const TrackPath& path : paths) {
            if (path.id == options.selected) selected = &path;
            else order.push_back(&path);
        }
        if (selected) order.push_back(selected);

        for (const TrackPath* path : order) {
            if (path->points.isEmpty()) continue;
            if (options.depthRange >= 0.0 && path->id != options.selected &&
                std::abs(path->depths[nearestFrame(*path, options.t)] - options.depth) > options.depthRange)
                continue;   // not near this slice (the selected track is always drawn)
            const QRectF screenBounds = QRectF(toScreen(path->bounds.topLeft()), toScreen(path->bounds.bottomRight())).normalized();
            if (!screenBounds.adjusted(-1, -1, 1, 1).intersects(margin)) continue;

            const bool isSelected = path->id == options.selected && options.selected != 0;
            const bool dimmed = options.selected != 0 && !isSelected;
            const double width = isSelected ? 2.5 : 1.5;

            QPolygonF run;
            Style runStyle = Style::Past;
            const auto flush = [&] {
                if (run.size() >= 2) {
                    QColor c = path->color;
                    const bool future = runStyle == Style::Future || runStyle == Style::FutureGap;
                    double alpha = future ? 0.3 : 0.9;
                    if (dimmed) alpha *= 0.45;
                    c.setAlphaF(static_cast<float>(alpha));
                    QPen pen(c, future ? width - 0.5 : width);
                    pen.setCapStyle(Qt::RoundCap);
                    pen.setJoinStyle(Qt::RoundJoin);
                    if (runStyle == Style::PastGap || runStyle == Style::FutureGap) pen.setStyle(Qt::DotLine);
                    p.setPen(pen);
                    p.drawPolyline(run);
                }
                run.clear();
            };

            for (int i = 0; i + 1 < path->points.size(); ++i) {
                const Index f0 = path->frames[i], f1 = path->frames[i + 1];
                if (f1 < oldest) continue;
                const bool future = f0 >= options.t;
                if (future && !options.future) break;
                const bool gap = f1 - f0 > 1;
                const Style style = future ? (gap ? Style::FutureGap : Style::Future) : (gap ? Style::PastGap : Style::Past);
                if (run.isEmpty() || style != runStyle) {
                    flush();
                    runStyle = style;
                    run << toScreen(path->points[i]);
                }
                run << toScreen(path->points[i + 1]);
            }
            flush();

            // where the track is now
            const auto now = std::find(path->frames.begin(), path->frames.end(), options.t);
            if (now != path->frames.end()) {
                const QPointF at = toScreen(path->points[static_cast<int>(now - path->frames.begin())]);
                const double r = isSelected ? 4.5 : 3.0;
                QColor fill = path->color;
                if (dimmed) fill.setAlphaF(0.5f);
                p.setPen(QPen(isSelected ? theme::kViewerText : theme::kViewerGround, isSelected ? 1.5 : 1.0));
                p.setBrush(fill);
                p.drawEllipse(at, r, r);
                p.setBrush(Qt::NoBrush);
            }
        }
        p.restore();
    }

} // namespace sirius::app
