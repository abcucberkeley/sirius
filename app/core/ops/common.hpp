#ifndef SIRIUS_APP_OPS_COMMON_HPP
#define SIRIUS_APP_OPS_COMMON_HPP

// What every operation implementation needs: the (c, t) volume loops that
// report progress and honour cancellation, the small formatters its summary
// is built from, and the two standard Diagnostics.

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>

#include "core/operation.hpp"

namespace sirius::app {

    // Runs `fn(c, t, progress)` for every (c, t) volume, reporting progress
    // 0..1 and honouring cancellation.
    void forEachVolume(const DatasetMeta& meta, const StepContext& ctx,
                       const std::function<void(Index c, Index t)>& fn);
    // Like forEachVolume, but when the context asks for every GPU the (c, t)
    // volumes run in parallel (one OpenMP thread per device) and `fn` is
    // given the device for that volume.
    void forEachVolumeOnGpus(const DatasetMeta& meta, const StepContext& ctx,
                             const std::function<void(Index c, Index t, Device device)>& fn);
    // "3 angles · 5 phases" style joining with " · ".
    std::string joinSummary(std::initializer_list<std::string> parts);
    // Channel label "488 α-actinin" for summaries.
    std::string channelName(const DatasetMeta& meta, Index c);
    // "12.8 GB", "412 MB"
    std::string formatBytes(std::uint64_t bytes);
    std::string formatNumber(double v, int decimals);
    // Otsu's threshold of `n` values (256-bin histogram over the finite ones; NaN and +-inf ignored).
    float otsuThreshold(const float* values, Index n);
    // "~9 s" for `bytes` at a nominal throughput.
    std::string estimatedTime(std::uint64_t bytes, double bytesPerSecond);
    // Input / Output thumbnails, summary and cost facts (the "Einsum / other" panel).
    Diagnostics genericDiagnostics(const StepInput& input, const StepOutput& output, const std::string& summary,
                                   double bytesPerSecond = 2.0e9);
    std::shared_ptr<Array5> allocateLike(const DatasetMeta& meta);
    // Label table + review-queue facts (the segmentation panel's data).
    Diagnostics labelDiagnostics(const LabelVolume& labels, const std::string& summary);

} // namespace sirius::app

#endif // SIRIUS_APP_OPS_COMMON_HPP
