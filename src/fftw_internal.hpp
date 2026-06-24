#ifndef SIRIUS_FFTW_INTERNAL_HPP
#define SIRIUS_FFTW_INTERNAL_HPP

#include "sirius/fft_common.hpp"

#include <cstddef>
#include <mutex>
#include <vector>

#include <fftw3.h>

namespace sirius::detail {

    std::mutex& fftwPlannerMutex();

    unsigned int toFFTWFlag(PlanRigor rigor);

    int checkedProduct(const std::vector<int>& dims, const char* what);
    int checkedMultiply(int a, int b, const char* what);

    // Caller must hold fftwPlannerMutex().
    void ensureDoubleThreadsInitializedLocked();

    void* checkedFftwMalloc(std::size_t bytes);

    template <typename T>
    struct FftwTypedFree {
        void operator()(T* p) const { fftw_free(p); }
    };

} // namespace sirius::detail

#endif // SIRIUS_FFTW_INTERNAL_HPP
