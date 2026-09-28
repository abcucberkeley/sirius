#ifndef SIRIUS_APP_DISPLAY_MAPPING_HPP
#define SIRIUS_APP_DISPLAY_MAPPING_HPP

// Intensity mapping for on-screen display of double-precision volumes. Pure
// functions over raw pointers (no GUI types) so they are unit-testable and
// the GUI only has to display the resulting 8-bit planes.

#include <cstddef>
#include <cstdint>

#include <sirius/buffer.hpp>

namespace sirius::app {

    // Display window: values <= lo map to black, >= hi to white.
    struct DisplayRange {
        double lo = 0.0;
        double hi = 1.0;

        bool valid() const noexcept { return hi > lo; }
    };

    // Exact min/max over n values. NaNs are ignored; an all-NaN or empty input
    // yields {0, 0} (invalid).
    DisplayRange minMaxRange(const double* data, Index n) noexcept;

    // Robust window from the [lowFrac, highFrac] quantiles (e.g. 0.001 and
    // 0.999 clip hot pixels). The quantiles are estimated from at most
    // maxSamples values taken at a fixed stride, so the cost is bounded for
    // arbitrarily large volumes. NaNs are ignored. Falls back to min/max when
    // the quantile window collapses.
    DisplayRange percentileRange(const double* data, Index n, double lowFrac, double highFrac,
                                 Index maxSamples = Index{1} << 20);

    // dst[i] = clamp(round((src[i] - lo) / (hi - lo) * 255), 0, 255).
    // An invalid range maps everything to 0. NaN maps to 0.
    void mapToGray8(const double* src, Index n, DisplayRange range, std::uint8_t* dst) noexcept;

} // namespace sirius::app

#endif // SIRIUS_APP_DISPLAY_MAPPING_HPP
