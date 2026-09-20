#ifndef SIRIUS_APP_OPS_SIM_PARAMS_HPP
#define SIRIUS_APP_OPS_SIM_PARAMS_HPP

// The SIM step's parameters as the library's SIMParameters, before any run.

#include <sirius/sim_parameters.hpp>

#include "core/dataset.hpp"
#include "core/params.hpp"

namespace sirius::app {

    // SIMParameters the SIM step would reconstruct with (pixel sizes from the
    // input); the viewer's frequency-space overlays use it before any run.
    SIMParameters simParametersFromStep(const ParamSet& params, const DatasetMeta& input);

} // namespace sirius::app

#endif // SIRIUS_APP_OPS_SIM_PARAMS_HPP
