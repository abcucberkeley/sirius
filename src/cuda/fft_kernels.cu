#include "cuda/fft_kernels.hpp"

#include <cuComplex.h>

namespace sirius::cuda {

    namespace {
        constexpr int kBlock = 256;

        // Grid-stride loops over a fixed, modest grid, as in kernels.cu.
        inline unsigned gridFor(std::size_t n) {
            const std::size_t blocks = (n + kBlock - 1) / kBlock;
            return static_cast<unsigned>(blocks < 4096 ? (blocks == 0 ? 1 : blocks) : 4096);
        }

        __global__ void scaleKernel(cuDoubleComplex* __restrict__ p, std::size_t n, double s) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)gridDim.x * blockDim.x) {
                p[i].x *= s;
                p[i].y *= s;
            }
        }

        __global__ void scaleRealKernel(double* __restrict__ p, std::size_t n, double s) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)gridDim.x * blockDim.x)
                p[i] *= s;
        }
    } // namespace

    void scaleComplexDouble(std::complex<double>* p, std::size_t n, double scale, cudaStream_t stream) {
        if (n == 0) return;
        scaleKernel<<<gridFor(n), kBlock, 0, stream>>>(reinterpret_cast<cuDoubleComplex*>(p), n, scale);
    }

    void scaleDouble(double* p, std::size_t n, double scale, cudaStream_t stream) {
        if (n == 0) return;
        scaleRealKernel<<<gridFor(n), kBlock, 0, stream>>>(p, n, scale);
    }

} // namespace sirius::cuda
