#include "core/ops/builtin.hpp"

namespace sirius::app {

    std::vector<OperationFactory> builtinOperationFactories() {
        return {
            &makeLoadOperation,
            // Reconstruct
            &makeSimOperation,
            &makeDeconvolveOperation,
            &makeVolumeOperation,
            // Reduce
            &makeEinsumOperation,
            &makeMaxProjectionOperation,
            &makeMeanOverTimeOperation,
            // Intensity
            &makeContrastOperation,
            &makeFlatFieldOperation,
            &makeBleachOperation,
            // Geometry
            &makeDeskewOperation,
            &makeCropPadOperation,
            &makeResampleOperation,
            // Combine
            &makeMergeOperation,
            &makeStitchOperation,
            &makeRegisterOperation,
            // Segment
            &makeFoundationOperation,
            &makeTorchSegmentationOperation,
            &makeThresholdOperation,
            &makeClassicalSegmentationOperation,
            &makeSkimageSegmentationOperation,
            &makeLabelCleanupOperation,
            &makeTrackOperation,
        };
    }

} // namespace sirius::app
