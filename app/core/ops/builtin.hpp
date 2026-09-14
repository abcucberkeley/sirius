#ifndef SIRIUS_APP_OPS_BUILTIN_HPP
#define SIRIUS_APP_OPS_BUILTIN_HPP

// Factories of the built-in operations, in menu order. Every ops/*.cpp
// defines one `std::unique_ptr<Operation> makeXxxOperation()`; the list in
// builtin_list.cpp is what registerBuiltinOperations() walks.

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <sirius/sim_parameters.hpp>

#include "core/array_source.hpp"
#include "core/operation.hpp"
#include "core/rpc.hpp"

namespace sirius::app {

    using OperationFactory = std::unique_ptr<Operation> (*)();
    std::vector<OperationFactory> builtinOperationFactories();

    std::unique_ptr<Operation> makeLoadOperation();
    // The Load step's parameters as the options it opens the dataset with
    // (page order, voxel size, SIM layout, tile, full read); 0 keeps what the
    // file says for that axis or size.
    OpenOptions loadOpenOptions(const ParamSet& loadParams);
    std::unique_ptr<Operation> makeSimOperation();
    std::unique_ptr<Operation> makeDeconvolveOperation();
    std::unique_ptr<Operation> makeVolumeOperation();
    std::unique_ptr<Operation> makeEinsumOperation();
    std::unique_ptr<Operation> makeMaxProjectionOperation();
    std::unique_ptr<Operation> makeMeanOverTimeOperation();
    std::unique_ptr<Operation> makeContrastOperation();
    std::unique_ptr<Operation> makeFlatFieldOperation();
    std::unique_ptr<Operation> makeBleachOperation();
    std::unique_ptr<Operation> makeDeskewOperation();
    std::unique_ptr<Operation> makeCropPadOperation();
    std::unique_ptr<Operation> makeResampleOperation();
    std::unique_ptr<Operation> makeMergeOperation();
    std::unique_ptr<Operation> makeStitchOperation();
    std::unique_ptr<Operation> makeRegisterOperation();
    std::unique_ptr<Operation> makeTorchSegmentationOperation();
    std::unique_ptr<Operation> makeFoundationOperation();
    std::unique_ptr<Operation> makeThresholdOperation();
    std::unique_ptr<Operation> makeClassicalSegmentationOperation();
    std::unique_ptr<Operation> makeSkimageSegmentationOperation();
    std::unique_ptr<Operation> makeLabelCleanupOperation();
    std::unique_ptr<Operation> makeTrackOperation();

    // Shared helpers for operation implementations (ops/common.cpp).
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

    // Live preview of the contrast step's histograms without running it
    // (sub-sampled so it stays under ~100 ms on large stacks).
    Diagnostics contrastPreview(const StepInput& input, const ParamSet& params);

    // The window the Contrast step applies (min / max / gamma parameters);
    // dataMin / dataMax (the range of at most `maxPlanes` sampled planes of
    // channel `c`, 0 = every plane) are filled when `wantRange`.
    struct ContrastWindow {
        float lo = 0.0f, hi = 1.0f;
        float gamma = 1.0f;
        float dataMin = 0.0f, dataMax = 1.0f;
    };
    ContrastWindow contrastWindow(const StepInput& input, const ParamSet& params, Index c, Index maxPlanes,
                                  bool wantRange = false);
    // Parameter sets behind the Auto and Reset buttons: min / max on the
    // lo / hi percentiles of the input (all channels), or its full range.
    ParamSet contrastAutoParams(const ParamSet& current, const StepInput& input);
    ParamSet contrastResetParams(const ParamSet& current, const StepInput& input);
    // "TorchScript · in (1, 1, Z, Y, X) float32 · out (1, 3, Z, Y, X) · 41 MB"
    // from the worker's model_info; throws when the worker cannot load it.
    nlohmann::json torchModelInfo(RemoteWorker& worker, const std::string& modelPath);
    std::string torchModelSummary(const nlohmann::json& info);
    // SIMParameters the SIM step would reconstruct with (pixel sizes from the
    // input); the viewer's frequency-space overlays use it before any run.
    SIMParameters simParametersFromStep(const ParamSet& params, const DatasetMeta& input);

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

#endif // SIRIUS_APP_OPS_BUILTIN_HPP
