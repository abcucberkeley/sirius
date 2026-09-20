#ifndef SIRIUS_CUDA_FFT_KERNELS_HPP
#define SIRIUS_CUDA_FFT_KERNELS_HPP

// The normalization kernels of the cuFFT backends (cuFFT, like FFTW, leaves
// the inverse transform unscaled). Plain C++ types, as in cuda/kernels.hpp:
// only fft_kernels.cu needs nvcc. Every function enqueues on `stream` and
// returns.

#include <cuda_runtime.h>

#include <complex>
#include <cstddef>

namespace sirius::cuda {

    // Multiply n complex<double> by a real scalar (used by ifft normalization).
    void scaleComplexDouble(std::complex<double>* p, std::size_t n, double scale, cudaStream_t stream);

    // Multiply n doubles by a scalar (used by irfft normalization).
    void scaleDouble(double* p, std::size_t n, double scale, cudaStream_t stream);

} // namespace sirius::cuda

#endif // SIRIUS_CUDA_FFT_KERNELS_HPP
