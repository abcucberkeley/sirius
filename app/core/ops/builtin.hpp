#ifndef SIRIUS_APP_OPS_BUILTIN_HPP
#define SIRIUS_APP_OPS_BUILTIN_HPP

// Factories of the built-in operations, in menu order. Every ops/*.cpp
// defines one `std::unique_ptr<Operation> makeXxxOperation()`; the list in
// builtin_list.cpp is what registerBuiltinOperations() walks.
//
// Only the factories: what an operation *implementation* shares with the
// others is ops/common.hpp, and the few helpers the GUI calls directly have
// headers of their own (ops/contrast.hpp, ops/load.hpp, ops/torch_model.hpp,
// ops/segment_common.hpp, ops/sim_params.hpp). So this header -- and, through
// the factories, the whole set of operations behind it -- is included where
// operations are registered, not by every operation.

#include <memory>
#include <vector>

#include "core/operation.hpp"

namespace sirius::app {

    // Registers every built-in operation (idempotent).
    void registerBuiltinOperations();

    using OperationFactory = std::unique_ptr<Operation> (*)();
    std::vector<OperationFactory> builtinOperationFactories();

    std::unique_ptr<Operation> makeLoadOperation();
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

} // namespace sirius::app

#endif // SIRIUS_APP_OPS_BUILTIN_HPP
