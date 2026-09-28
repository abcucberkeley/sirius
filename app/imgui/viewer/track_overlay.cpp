#include "imgui/viewer/track_overlay.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

#include "core/labels.hpp"   // labelColor
#include "imgui/theme.hpp"

namespace sirius::app::gui {

    using theme::px;

    namespace {
        DPoint planeOf(const TrackPoint& p, TrackPlane plane) {
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
        std::size_t nearestFrame(const TrackPath& path, Index t) {
            const auto it = std::lower_bound(path.frames.begin(), path.frames.end(), t);
            if (it == path.frames.end()) return path.frames.size() - 1;
            const std::size_t i = static_cast<std::size_t>(it - path.frames.begin());
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

    void dashedPolyline(ImDrawList* dl, const std::vector<ImVec2>& points, ImU32 color, float thickness, float dash, float gap) {
        if (points.size() < 2 || dash <= 0.0f) return;
        bool on = true;
        float left = dash;   // of the current dash or gap
        for (std::size_t i = 0; i + 1 < points.size(); ++i) {
            ImVec2 a = points[i];
            const ImVec2 b = points[i + 1];
            float len = std::hypot(b.x - a.x, b.y - a.y);
            if (len <= 0.0f) continue;
            const ImVec2 dir((b.x - a.x) / len, (b.y - a.y) / len);
            while (len > 0.0f) {
                const float step = std::min(left, len);
                const ImVec2 c(a.x + dir.x * step, a.y + dir.y * step);
                if (on) dl->AddLine(a, c, color, thickness);
                a = c;
                len -= step;
                left -= step;
                if (left <= 0.0f) {
                    on = !on;
                    left = on ? dash : gap;
                }
            }
        }
    }

    std::vector<TrackPath> trackPaths(const TrackIndex& index, TrackPlane plane, std::uint32_t only) {
        std::map<std::uint32_t, TrackPath> byId;
        index.forEachPoint([&](std::uint32_t id, const TrackPoint& point) {
            if (only && id != only) return;
            TrackPath& path = byId[id];
            const DPoint at = planeOf(point, plane) + DPoint(kVoxelCentre, kVoxelCentre);
            if (path.points.empty()) {
                path.id = id;
                path.color = theme::fromFloat(labelColor(id));
                path.bounds = DRect{at.x, at.y, at.x, at.y};
            } else {
                path.bounds.x0 = std::min(path.bounds.x0, at.x);
                path.bounds.y0 = std::min(path.bounds.y0, at.y);
                path.bounds.x1 = std::max(path.bounds.x1, at.x);
                path.bounds.y1 = std::max(path.bounds.y1, at.y);
            }
            path.points.push_back(at);
            path.frames.push_back(point.t);
            path.depths.push_back(depthOf(point, plane));
        });
        std::vector<TrackPath> out;
        out.reserve(byId.size());
        for (auto& [id, path] : byId) out.push_back(std::move(path));
        return out;
    }

    void paintTrackPaths(ImDrawList* dl, const std::vector<TrackPath>& paths, const TrackPaintOptions& options,
                         const std::function<ImVec2(const DPoint&)>& toScreen, ImVec2 visibleMin, ImVec2 visibleMax) {
        if (paths.empty()) return;
        const float margin = px(8);
        const ImVec2 mMin(visibleMin.x - margin, visibleMin.y - margin), mMax(visibleMax.x + margin, visibleMax.y + margin);
        const Index oldest = options.tail > 0 ? options.t - options.tail : std::numeric_limits<Index>::min();

        // the selected track last, so it is on top
        std::vector<const TrackPath*> order;
        order.reserve(paths.size());
        const TrackPath* selected = nullptr;
        for (const TrackPath& path : paths) {
            if (path.id == options.selected) selected = &path;
            else order.push_back(&path);
        }
        if (selected) order.push_back(selected);

        for (const TrackPath* path : order) {
            if (path->points.empty()) continue;
            if (options.depthRange >= 0.0 && path->id != options.selected &&
                std::abs(path->depths[nearestFrame(*path, options.t)] - options.depth) > options.depthRange)
                continue;   // not near this slice (the selected track is always drawn)
            const ImVec2 a = toScreen(DPoint(path->bounds.x0, path->bounds.y0)), b = toScreen(DPoint(path->bounds.x1, path->bounds.y1));
            const ImVec2 sMin(std::min(a.x, b.x) - 1.0f, std::min(a.y, b.y) - 1.0f), sMax(std::max(a.x, b.x) + 1.0f, std::max(a.y, b.y) + 1.0f);
            if (sMax.x < mMin.x || sMin.x > mMax.x || sMax.y < mMin.y || sMin.y > mMax.y) continue;

            const bool isSelected = path->id == options.selected && options.selected != 0;
            const bool dimmed = options.selected != 0 && !isSelected;
            const float width = isSelected ? 2.5f : 1.5f;

            std::vector<ImVec2> run;
            Style runStyle = Style::Past;
            const auto flush = [&] {
                if (run.size() >= 2) {
                    const bool future = runStyle == Style::Future || runStyle == Style::FutureGap;
                    float alpha = future ? 0.3f : 0.9f;
                    if (dimmed) alpha *= 0.45f;
                    const ImU32 c = theme::withAlpha(path->color, alpha);
                    const float w = px(future ? width - 0.5f : width);
                    if (runStyle == Style::PastGap || runStyle == Style::FutureGap) {
                        // dotted: a dot a pen wide, two pens of gap
                        dashedPolyline(dl, run, c, w, w, 2.0f * w);
                    } else {
                        dl->AddPolyline(run.data(), static_cast<int>(run.size()), c, w, ImDrawFlags_None);
                    }
                }
                run.clear();
            };

            for (std::size_t i = 0; i + 1 < path->points.size(); ++i) {
                const Index f0 = path->frames[i], f1 = path->frames[i + 1];
                if (f1 < oldest) continue;
                const bool future = f0 >= options.t;
                if (future && !options.future) break;
                const bool gap = f1 - f0 > 1;
                const Style style = future ? (gap ? Style::FutureGap : Style::Future) : (gap ? Style::PastGap : Style::Past);
                if (run.empty() || style != runStyle) {
                    flush();
                    runStyle = style;
                    run.push_back(toScreen(path->points[i]));
                }
                run.push_back(toScreen(path->points[i + 1]));
            }
            flush();

            // where the track is now
            const auto now = std::find(path->frames.begin(), path->frames.end(), options.t);
            if (now != path->frames.end()) {
                const ImVec2 at = toScreen(path->points[static_cast<std::size_t>(now - path->frames.begin())]);
                const float r = px(isSelected ? 4.5f : 3.0f);
                const ImU32 fill = dimmed ? theme::withAlpha(path->color, 0.5f) : path->color;
                dl->AddCircleFilled(at, r, fill);
                dl->AddCircle(at, r, isSelected ? theme::kViewerText : theme::kViewerGround, 0, px(isSelected ? 1.5f : 1.0f));
            }
        }
    }

} // namespace sirius::app::gui
