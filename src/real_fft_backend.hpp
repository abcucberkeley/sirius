#ifndef SIRIUS_REAL_FFT_BACKEND_HPP
#define SIRIUS_REAL_FFT_BACKEND_HPP

// Internal: one backend per device type behind sirius::RealFFT (same pattern
// as fft_backend.hpp for the complex transform).

#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

#include "sirius/device.hpp"
#include "sirius/fft_common.hpp"

namespace sirius::detail {

    class RealFftBackend {
    public:
        virtual ~RealFftBackend() = default;
        virtual void rfft(const double* in, std::complex<double>* out,
                          const Stream& stream) const = 0;
        // The complex input is preserved (both backends copy it internally:
        // FFTW's and cuFFT's multi-dimensional c2r transforms overwrite it).
        virtual void irfft(const std::complex<double>* in, double* out,
                           const Stream& stream) const = 0;
        // out[i] *= scale for the full real batch (irfft normalization).
        virtual void scaleReal(double* out, std::size_t n, double scale,
                               const Stream& stream) const = 0;
    };

    std::unique_ptr<RealFftBackend> makeFftwRealBackend(const std::vector<int>& dims, int howmany,
                                                        PlanRigor rigor);
    // Defined in real_fft_cufft.cpp (only compiled with SIRIUS_HAS_CUDA).
    std::unique_ptr<RealFftBackend> makeCufftRealBackend(const std::vector<int>& dims, int howmany,
                                                         Device device);

} // namespace sirius::detail

#endif // SIRIUS_REAL_FFT_BACKEND_HPP
