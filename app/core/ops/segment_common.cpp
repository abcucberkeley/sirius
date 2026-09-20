// Post-processing shared by the segmentation steps: probabilities / an
// intensity threshold -> mask -> instances -> statistics and review flags.
#include "core/ops/segment_common.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace sirius::app {

    std::uint32_t labelsFromProbabilities(const float* foreground, const float* boundary, Index z, Index y, Index x,
                                          const LabelPostOptions& options, LabelVolume& labels, Index t) {
        const Index n = z * y * x;
        if (labels.z() != z || labels.y() != y || labels.x() != x || t < 0 || t >= labels.t())
            throw std::invalid_argument("labelsFromProbabilities: label volume does not match (z, y, x, t)");
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(n));
        for (Index i = 0; i < n; ++i) mask[static_cast<std::size_t>(i)] = foreground[i] > options.threshold ? 1 : 0;

        std::uint32_t* out = labels.volume(t);
        std::uint32_t count = 0;
        if (options.post == "None (raw probabilities)") {
            // no instances: the foreground is one semantic label
            for (Index i = 0; i < n; ++i) out[i] = mask[static_cast<std::size_t>(i)];
            count = std::any_of(mask.begin(), mask.end(), [](std::uint8_t m) { return m != 0; }) ? 1 : 0;
        } else if (options.post.rfind("Watershed", 0) == 0) {
            std::fill(out, out + n, 0u);
            std::vector<float> distance(static_cast<std::size_t>(n));
            distanceTransform(mask.data(), z, y, x, distance.data());
            std::uint32_t seeds = 0;
            if (options.externalSeeds) {
                // seeds the caller found in the image itself; keep only the
                // ones the mask actually covers
                seeds = 0;
                for (Index i = 0; i < n; ++i) {
                    out[i] = mask[static_cast<std::size_t>(i)] ? options.externalSeeds[i] : 0u;
                    seeds = std::max(seeds, out[i]);
                }
            } else if (options.seeds == "H-maxima") {
                seeds = hMaximaSeeds(distance.data(), mask.data(), z, y, x, options.seedDepth, out);
            } else {
                seeds = distanceSeeds(mask.data(), z, y, x, options.seedMinDistance, out, options.poll);
            }
            if (seeds == 0) {
                count = connectedComponents(mask.data(), z, y, x, out);
            } else {
                std::vector<float> landscape(static_cast<std::size_t>(n));
                if (boundary) {
                    std::copy_n(boundary, n, landscape.data());
                } else {
                    // classic distance watershed: ridges are far from the object centres
                    for (Index i = 0; i < n; ++i) landscape[static_cast<std::size_t>(i)] = -distance[static_cast<std::size_t>(i)];
                }
                watershed(landscape.data(), mask.data(), z, y, x, out);
                count = seeds;
                // The flood only reaches what is connected to a seed. A
                // component no seed landed in -- a thin bar the h-maxima
                // flattened, a nucleus closer to another than the seed
                // distance -- is still an object, not background: numbered
                // after the seeds, in raster order of its first voxel.
                std::vector<std::uint8_t> unseeded(static_cast<std::size_t>(n), 0);
                bool any = false;
                for (Index i = 0; i < n; ++i) {
                    if (!mask[static_cast<std::size_t>(i)] || out[i] != 0) continue;
                    unseeded[static_cast<std::size_t>(i)] = 1;
                    any = true;
                }
                if (any) {
                    std::vector<std::uint32_t> extra(static_cast<std::size_t>(n));
                    const std::uint32_t more = connectedComponents(unseeded.data(), z, y, x, extra.data());
                    for (Index i = 0; i < n; ++i)
                        if (extra[static_cast<std::size_t>(i)]) out[i] = seeds + extra[static_cast<std::size_t>(i)];
                    count = seeds + more;
                }
            }
        } else {
            count = connectedComponents(mask.data(), z, y, x, out);
        }
        count = removeSmall(out, n, options.minVoxels);
        labels.recomputeStats(t, foreground);
        for (LabelStats& s : labels.stats()) s.cls = options.className;
        labels.applyFlags(options.flags);
        return count;
    }

} // namespace sirius::app
