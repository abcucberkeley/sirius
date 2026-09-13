// Tracking: linking segmented objects across time.
//
// Two pieces. `solveAssignment` is the optimal assignment (Hungarian /
// Jonker-Volgenant shortest augmenting paths, O(n^3)) that answers "which
// object in this frame is which object in the next" for the whole frame at
// once, rather than greedily nearest-first, which swaps identities whenever
// two objects pass close to each other. `linkTracks` builds the cost from
// centroid distance and voxel overlap, gates it, runs the solver on every
// consecutive pair, then optionally closes gaps over frames where an object
// was missed -- a second assignment problem between track ends and starts.
#ifndef SIRIUS_APP_TRACKING_HPP
#define SIRIUS_APP_TRACKING_HPP

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

#include "core/labels.hpp"

namespace sirius::app {

    // Cost that forbids a pair; `solveAssignment` never returns one and
    // `linkTracks` treats it as "these cannot be the same object".
    inline constexpr double kNoAssignment = std::numeric_limits<double>::infinity();

    // Minimum-cost matching of `rows` rows to `cols` columns; `cost` is row
    // major and may hold kNoAssignment. Returns one entry per row: the column
    // it takes, or -1 when it is left unmatched (no feasible column, or more
    // rows than columns). `poll` is called once per row, O(rows x cols) of
    // work apart; throw from it to stop.
    std::vector<int> solveAssignment(const std::vector<double>& cost, int rows, int cols,
                                     const std::function<void()>& poll = {});

    // One segmented object in one frame.
    struct TrackObject {
        std::uint32_t label = 0;
        std::array<double, 3> centroid{0.0, 0.0, 0.0};   // z, y, x in voxels
        Index voxels = 0;
    };

    struct TrackOptions {
        double maxDistanceUm = 10.0;   // gate: no link moves further than this
        double overlapWeight = 0.5;    // how much shared voxels count against distance
        Index maxGap = 1;              // frames an object may vanish for and still be the same track
        Index minLength = 1;           // tracks shorter than this are dropped
        // Called between pieces of work -- every frame pair, every row of an
        // assignment -- so a gap closing over thousands of tracks, O(n^3),
        // can be cancelled. Throw from it to stop.
        std::function<void()> poll;
    };

    // One object followed through time: (frame, label in that frame).
    struct Track {
        std::uint32_t id = 0;
        std::vector<std::pair<Index, std::uint32_t>> points;

        Index first() const noexcept { return points.empty() ? 0 : points.front().first; }
        Index last() const noexcept { return points.empty() ? 0 : points.back().first; }
        std::size_t length() const noexcept { return points.size(); }
    };

    struct TrackResult {
        std::vector<Track> tracks;
        Index gapsClosed = 0;
        Index links = 0;          // frame-to-frame links accepted
    };

    // `byFrame[t]` are the objects of frame t, `overlap[t]` the voxel counts
    // shared between frame t and t + 1 (row = object of t, column = object of
    // t + 1, row major; empty to ignore overlap).
    TrackResult linkTracks(const std::vector<std::vector<TrackObject>>& byFrame,
                           const std::vector<std::vector<Index>>& overlap, const std::array<double, 3>& voxelUm,
                           const TrackOptions& options);

    // Objects and their centroids from one frame of a label volume.
    std::vector<TrackObject> objectsOfFrame(const LabelVolume& labels, Index t);

    // Voxels shared by each (label of t, label of t + 1) pair, row major over
    // the two frames' object lists.
    std::vector<Index> overlapBetween(const LabelVolume& labels, Index t, const std::vector<TrackObject>& a,
                                      const std::vector<TrackObject>& b);

} // namespace sirius::app

#endif // SIRIUS_APP_TRACKING_HPP
