#ifndef SIRIUS_APP_OPS_SEGMENT_COMMON_HPP
#define SIRIUS_APP_OPS_SEGMENT_COMMON_HPP

// The post-processing every segmentation step shares: a probability (or any
// intensity) volume and an optional boundary map turned into instance labels,
// with the statistics and review flags the label table reads.

#include <cstdint>
#include <functional>
#include <string>

#include "core/labels.hpp"

namespace sirius::app {

    // Instance labels from a (z, y, x) foreground probability (or any
    // intensity: `threshold` applies to it) and an optional boundary map,
    // shared by the segmentation steps: threshold, connected components or a
    // seeded watershed, small-object removal, statistics and flags. Fills
    // labels.volume(t); returns the label count.
    struct LabelPostOptions {
        std::string post = "Connected components";   // "Watershed on boundary channel" | "Connected components" | "None (raw probabilities)"
        double threshold = 0.5;
        Index minVoxels = 0;
        double seedMinDistance = 5.0;                // voxels between watershed seeds
        // "Distance maxima" (the peaks of the distance map, kept apart by
        // seedMinDistance) or "H-maxima" (peaks that stand seedDepth above
        // their surroundings, which does not split a lumpy object).
        std::string seeds = "Distance maxima";
        double seedDepth = 2.0;
        // Seeds the caller worked out itself (blob centres, say). When set the
        // watershed starts from these instead of computing its own.
        const std::uint32_t* externalSeeds = nullptr;
        std::uint32_t externalSeedCount = 0;
        LabelFlagRules flags;
        std::string className = "object";
        // Called during the long loops (the distance seeds); the step sets it
        // to throw when its run is cancelled.
        std::function<void()> poll;
    };
    std::uint32_t labelsFromProbabilities(const float* foreground, const float* boundary, Index z, Index y, Index x,
                                          const LabelPostOptions& options, LabelVolume& labels, Index t);

} // namespace sirius::app

#endif // SIRIUS_APP_OPS_SEGMENT_COMMON_HPP
