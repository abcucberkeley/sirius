#ifndef SIRIUS_APP_OPS_COMMON_HPP
#define SIRIUS_APP_OPS_COMMON_HPP

// What every operation implementation needs: the (c, t) volume loops that
// report progress and honour cancellation, the small formatters its summary
// is built from, and the two standard Diagnostics.

#include <array>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

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

    // One time point of a Prompt step as the worker is asked it, and what
    // becomes of the mask the worker returns for each prompt. The worker
    // answers one mask per prompt -- the points, then the boxes, then the
    // scribbles -- and lets a later mask win where two overlap. So the
    // background points are sent first: an object mask then wins over the
    // mask a background point produced, and that mask is dropped (a point on
    // the background names no object). Object masks are numbered 1..n in the
    // order they are sent.
    struct FramePrompt {
        struct Stroke {
            std::vector<std::array<double, 3>> points;   // a few, evenly spaced along the stroke
            int label = 1;
        };
        std::vector<std::array<double, 3>> points;      // (x, y, z), background first
        std::vector<int> pointLabels;                   // 1 object, 0 background
        std::vector<std::array<double, 6>> boxes;       // (x0, y0, z0, x1, y1, z1)
        std::vector<Stroke> scribbles;
        std::vector<std::uint32_t> ids;                 // per mask, in the worker's order: the label it becomes, 0 drops it
        std::vector<std::size_t> placed;                // per mask: the prompt's index in the step's list
        bool empty() const noexcept { return ids.empty(); }
        std::size_t objects() const noexcept;
    };
    // A scribble is sent as at most this many points, evenly spaced along the
    // stroke and including both ends: the prompt decoder was trained on a few
    // points per stroke (three, at random), and a stroke's every voxel would
    // make one decoder call no better and much slower.
    inline constexpr std::size_t kScribblePointsSent = 8;
    std::vector<std::array<double, 3>> scribbleSample(const std::vector<std::array<double, 3>>& stroke, std::size_t at_most = kScribblePointsSent);
    FramePrompt framePrompt(const std::vector<Prompt>& prompts, Index t);
    // Masks numbered as the worker returned them (i + 1 for the i-th mask)
    // renumbered to FramePrompt::ids, in place; an id beyond the list becomes 0.
    void applyPromptIds(std::uint32_t* labels, Index n, const FramePrompt& frame);
    // Every prompt inside the image and on one of its time points: the worker
    // refuses one outside the volume, and one placed on other data (a dataset
    // swapped under the step) would otherwise be a failed run rather than a
    // line in the panel. No prompts, or only background ones, is a warning:
    // the step runs and segments nothing.
    void validatePrompts(const std::vector<Prompt>& prompts, const DatasetMeta& in, Validation& v);
    // "2 boxes · 3 object points · 1 background point": what a run was given.
    std::string promptCounts(const std::vector<Prompt>& prompts);

    // The "device" of a request to the Python worker: where this run was
    // asked to go, as the launcher names it when it starts a local worker.
    // "auto" would mean the worker's own, fixed when its process started
    // and kept for the session, so a backend or GPU chosen since never
    // reached it. The HPC worker keeps its own device.
    inline std::string workerDevice(const StepContext& ctx) {
        if (ctx.backend == Backend::Cpu) return "cpu";
        if (ctx.backend == Backend::Cuda)
            return ctx.device.isCuda() && ctx.device.index >= 0 ? "cuda:" + std::to_string(ctx.device.index) : std::string("cuda");
        return "auto";
    }

} // namespace sirius::app

#endif // SIRIUS_APP_OPS_COMMON_HPP
