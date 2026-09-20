// cuFFT backend of sirius::RealFFT (compiled only with SIRIUS_ENABLE_CUDA);
// the complex transform's is fft_cufft.cpp.

#include "real_fft_backend.hpp"
#include "cuda_check.hpp"
#include "cufft_check.hpp"
#include "cuda/fft_kernels.hpp"

#include <cufft.h>

#include <numeric>
#include <stdexcept>
#include <string>

namespace sirius::detail {

    namespace {
        class CufftRealBackend final : public RealFftBackend {
        public:
            CufftRealBackend(const std::vector<int>& dims, int howmany, Device device) : device_(device) {
                cuda::DeviceGuard g(device.index);
                const int real_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>{});
                const int complex_size = real_size / dims.back() * (dims.back() / 2 + 1);
                full_real_ = static_cast<std::size_t>(real_size) * static_cast<std::size_t>(howmany);
                full_complex_ = static_cast<std::size_t>(complex_size) * static_cast<std::size_t>(howmany);

                cufftCheck(cufftPlanMany(&forward_, static_cast<int>(dims.size()), const_cast<int*>(dims.data()),
                                         nullptr, 1, real_size, nullptr, 1, complex_size, CUFFT_D2Z, howmany),
                           "cufftPlanMany D2Z");
                cufftResult r = cufftPlanMany(&inverse_, static_cast<int>(dims.size()),
                                              const_cast<int*>(dims.data()),
                                              nullptr, 1, complex_size, nullptr, 1, real_size,
                                              CUFFT_Z2D, howmany);
                if (r != CUFFT_SUCCESS) {
                    (void)cufftDestroy(forward_);
                    cufftCheck(r, "cufftPlanMany Z2D");
                }
                // cuFFT's multi-dimensional Z2D overwrites its input; keep a
                // plan-owned staging copy so irfft can preserve the caller's.
                cudaError_t e = cudaMalloc(&staging_, full_complex_ * sizeof(cufftDoubleComplex));
                if (e != cudaSuccess) {
                    (void)cufftDestroy(forward_);
                    (void)cufftDestroy(inverse_);
                    cuda::check(e, "RealFFT staging alloc");
                }
            }

            ~CufftRealBackend() override {
                cuda::DeviceGuardNoThrow g(device_.index);
                (void)cufftDestroy(forward_);
                (void)cufftDestroy(inverse_);
                (void)cudaFree(staging_);
            }

            void rfft(const double* in, std::complex<double>* out, const Stream& stream) const override {
                checkStream(stream);
                cuda::DeviceGuard g(device_.index);
                cufftCheck(cufftSetStream(forward_, cuda::handle(stream)), "cufftSetStream");
                cufftCheck(cufftExecD2Z(forward_, const_cast<double*>(in),
                                        reinterpret_cast<cufftDoubleComplex*>(out)),
                           "cufftExecD2Z");
            }

            void irfft(const std::complex<double>* in, double* out, const Stream& stream) const override {
                checkStream(stream);
                cuda::DeviceGuard g(device_.index);
                cuda::check(cudaMemcpyAsync(staging_, in, full_complex_ * sizeof(cufftDoubleComplex),
                                            cudaMemcpyDeviceToDevice, cuda::handle(stream)),
                            "RealFFT staging copy");
                cufftCheck(cufftSetStream(inverse_, cuda::handle(stream)), "cufftSetStream");
                cufftCheck(cufftExecZ2D(inverse_, staging_, out), "cufftExecZ2D");
            }

            void scaleReal(double* out, std::size_t n, double s, const Stream& stream) const override {
                cuda::DeviceGuard g(device_.index);
                cuda::scaleDouble(out, n, s, cuda::handle(stream));
                cuda::check(cudaGetLastError(), "scaleReal kernel launch");
            }

        private:
            void checkStream(const Stream& stream) const {
                if (stream.device().isCuda() && stream.device() != device_)
                    throw std::invalid_argument("RealFFT: stream on " + toString(stream.device()) +
                                                " used with a plan on " + toString(device_));
            }

            cufftHandle forward_{};
            cufftHandle inverse_{};
            cufftDoubleComplex* staging_ = nullptr;
            Device device_;
            std::size_t full_real_ = 0;
            std::size_t full_complex_ = 0;
        };
    } // namespace

    std::unique_ptr<RealFftBackend> makeCufftRealBackend(const std::vector<int>& dims, int howmany,
                                                         Device device) {
        return std::make_unique<CufftRealBackend>(dims, howmany, device);
    }

} // namespace sirius::detail
