#ifndef SIRIUS_APP_STATISTICS_HPP
#define SIRIUS_APP_STATISTICS_HPP

// Numbers about a step's output, for an agent that cannot look at a histogram
// panel: per-channel intensity statistics and a summary of the labels. What
// sirius-cli's `stats` command and the `statistics` tool report.
//
// The moments are exact and the order statistics are not. Min, max, mean,
// standard deviation and the counts come from every value, one plane at a
// time, so a lazy dataset of any size is streamed rather than loaded; the
// percentiles and the histogram need the values themselves, and those come
// from at most maxSamples of them taken at a fixed stride, which bounds the
// memory and keeps the result the same from one call to the next.

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "core/labels.hpp"
#include "core/operation.hpp"

namespace sirius::app {
    struct StatisticsOptions {
        std::vector<Index> channels;                              // empty = all
        Index t = 0;                                              // -1 = every time point
        std::vector<double> percentiles{0.1, 1.0, 50.0, 99.0, 99.9};
        int histogramBins = 0;
        std::uint64_t maxSamples = std::uint64_t{1} << 22;       // taken as 1 << 26 above it
        double saturationLevel = 0.0;                             // > 0: fraction of values >= it
    };
    // A channel without a single value that is not NaN has count 0 and NaN for
    // min, max, mean, stddev, every percentile and the saturated fraction (null
    // once written as JSON): no number is the honest answer, and 0 would read
    // as a measurement.
    struct ChannelStatistics {
        Index channel = 0;
        double min = 0, max = 0, mean = 0, stddev = 0;             // stddev: population (ddof 0), as numpy's default
        std::uint64_t count = 0, nanCount = 0;                      // count: the values that are not NaN
        // (p, value), in the order asked for: linear between the two nearest
        // samples, numpy's default, except that an infinite one gives that
        // infinity where numpy's form gives NaN.
        std::vector<std::pair<double, double>> percentiles;
        std::optional<double> saturatedFraction;                    // exact; set when saturationLevel > 0
        // Equal-width bins over [histLo, histHi] = [min, max] (the finite
        // extremes of the samples when min or max is infinite), filled from
        // the samples: the counts sum to `count` unless `sampled`.
        double histLo = 0, histHi = 0;
        std::vector<std::uint64_t> histogram;
        bool sampled = false;                                       // the percentiles and histogram come from a subsample
    };
    // Streams planes from out.array or out.source; exact min/max/mean/std/count/NaN,
    // percentiles and histogram from at most maxSamples strided values. More
    // than 1 << 26 samples (256 MB, held one channel at a time) are never kept.
    // Throws std::invalid_argument for a channel or t outside the output, a
    // percentile outside [0, 100] or histogramBins outside [0, 65536];
    // std::runtime_error when the output holds no data; CancelledError
    // (core/cancel.hpp) once `cancelled` returns true, checked between planes.
    std::vector<ChannelStatistics> channelStatistics(const StepOutput& out, const StatisticsOptions& o,
                                                     const std::function<void(double)>& progress = {},
                                                     const std::function<bool()>& cancelled = {});
    // The maximum of a source pixel type (lib/pixel_type), for saturationLevel; 0 for float types.
    double pixelTypeMaximum(const DatasetMeta& meta);
    // {count, t, voxels:{min,median,mean,max}, volume_um3:{...}, flags:{..}, classes:{..}, reviewed, tracked,
    //  tracks?:{total, with_gaps, divisions}}
    // Frame t of the labels; voxelUm is (x, y, z), as DatasetMeta::voxelUm. The
    // table the volume holds is used when it describes frame t, and otherwise
    // one is made for t without touching the volume. flags and classes count
    // the objects per flag and per class; reviewed counts the reviewed ones;
    // tracks is there only for tracked labels. An empty volume gives count 0;
    // otherwise a t outside the labels throws std::invalid_argument.
    nlohmann::json labelStatistics(const LabelVolume& labels, Index t, const std::array<double, 3>& voxelUm);
} // namespace sirius::app

#endif // SIRIUS_APP_STATISTICS_HPP
