// Tracks as the reviewer sees them: a tracked label volume (one id naming the
// same object at every time point, LabelVolume::tracked) turned into what the
// track table, the trajectory overlay and "follow this track" need -- where
// each track is at every frame, when it starts and ends, how far it moves,
// and which tracks it came from or divided into.
//
// Two pieces. `TrackIndex` holds the centroid sums of every (frame, id) pair.
// It is built with one pass over the voxels and then kept current from the
// LabelDiff of each edit, because centroid sums are additive: an edit costs
// the voxels it changed, not a rescan of the clip. `summarizeTracks` turns the
// index and a lineage map into one row per track; it touches only the index,
// so it is cheap enough to redo after every edit.
//
// Lineage (core/labels.hpp) is kept beside the labels as {child id: parent id}, the form the
// model and btrack return it in. Nothing here assumes it is complete or even
// consistent: ids that no longer exist in the labels (deleted, merged away)
// are ignored rather than reported, and a cycle cannot hang anything because
// nothing walks the tree.
#ifndef SIRIUS_APP_TRACKS_HPP
#define SIRIUS_APP_TRACKS_HPP

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "core/labels.hpp"

namespace sirius::app {

    // One track at one time point. Centroid in voxels (z, y, x).
    struct TrackPoint {
        Index t = 0;
        std::array<double, 3> centroid{0.0, 0.0, 0.0};
        Index voxels = 0;
    };

    class TrackIndex {
    public:
        TrackIndex() = default;
        // One pass over every frame of `labels` (frames in parallel).
        explicit TrackIndex(const LabelVolume& labels);

        Index frames() const noexcept { return static_cast<Index>(frames_.size()); }
        bool empty() const noexcept;

        // Brings the index up to date after `diff` was applied to the labels
        // (forward) or reverted (backward), in the same order LabelVolume::apply
        // uses, so a stroke that touched one voxel twice is counted once.
        void apply(const LabelDiff& diff, bool forward = true);
        // Recounts one frame from the voxels: for writes that bypass the edits
        // (an operation filling LabelVolume::volume directly).
        void rescanFrame(const LabelVolume& labels, Index t);

        // Every id present in at least one frame, ascending.
        std::vector<std::uint32_t> ids() const;
        // The track's points in time order; empty when the id is nowhere.
        std::vector<TrackPoint> points(std::uint32_t id) const;
        std::optional<TrackPoint> pointAt(std::uint32_t id, Index t) const;
        // The point at `t`, else at the closest frame the track exists in
        // (the earlier one on a tie): where to send the time cursor when a
        // track is chosen that is not in the frame on screen.
        std::optional<TrackPoint> nearestPoint(std::uint32_t id, Index t) const;
        // Every (id, point) in time order; the order of ids within a frame is unspecified.
        void forEachPoint(const std::function<void(std::uint32_t id, const TrackPoint& point)>& fn) const;

    private:
        struct Sum {
            double z = 0.0, y = 0.0, x = 0.0;
            Index n = 0;
        };
        using Frame = std::unordered_map<std::uint32_t, Sum>;
        static void countFrame(const LabelVolume& labels, Index t, Frame& out);
        static TrackPoint pointOf(Index t, const Sum& s);
        void move(Frame& frame, Index linear, std::uint32_t from, std::uint32_t to);

        std::vector<Frame> frames_;
        Index y_ = 0, x_ = 0;   // plane extent, to turn a diff's linear index into (z, y, x)
    };

    struct TrackSummary {
        std::uint32_t id = 0;
        Index first = 0, last = 0;       // first and last frame present
        Index frames = 0;                // frames present
        Index gaps = 0;                  // frames missing between first and last
        double pathUm = 0.0;             // summed centroid steps between frames present
        double netUm = 0.0;              // first to last centroid
        double umPerFrame = 0.0;         // pathUm / (last - first); 0 for a single frame
        double meanVoxels = 0.0;
        std::uint32_t parent = 0;        // 0: none, or not in the labels any more
        std::vector<std::uint32_t> children;   // ascending; present in the labels

        Index span() const noexcept { return last - first + 1; }
        bool divides() const noexcept { return children.size() >= 2; }
    };

    // One row per track, ascending id. Distances in microns from `voxelUm`
    // (z, y, x), as everywhere else: a voxel step is not comparable between
    // the axes of anisotropic data.
    std::vector<TrackSummary> summarizeTracks(const TrackIndex& index, const Lineage& lineage,
                                              const std::array<double, 3>& voxelUm);

    // The lineage a worker reports, {"child id": parent id} with the keys as
    // strings (JSON object keys are). Entries that are not two positive ids
    // are skipped rather than failing the step: the labels are the result,
    // the lineage an annotation of them.
    Lineage lineageFromJson(const nlohmann::json& j);

    // Parents with two or more children present: the divisions the lineage
    // still describes after any edits.
    Index countDivisions(const std::vector<TrackSummary>& tracks);

} // namespace sirius::app

#endif // SIRIUS_APP_TRACKS_HPP
