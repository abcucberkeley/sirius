#ifndef SIRIUS_APP_OPS_CONTRAST_HPP
#define SIRIUS_APP_OPS_CONTRAST_HPP

// The Contrast step's window and histograms, computed without running it: the
// viewer shows the step live on its input while the percentiles are dragged
// (OpInfo::livePreview), and the parameter panel's Auto and Reset buttons are
// parameter sets, not a mode.

#include "core/operation.hpp"

namespace sirius::app {

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

} // namespace sirius::app

#endif // SIRIUS_APP_OPS_CONTRAST_HPP
