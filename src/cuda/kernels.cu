#include "cuda/kernels.hpp"

#include <cuComplex.h>

#include <cstring>

namespace sirius::cuda {

    namespace {
        constexpr int kBlock = 256;

        // Grid-stride loops: a fixed, modest grid handles any n without
        // per-launch grid-size arithmetic overflowing, and keeps enough
        // blocks in flight to saturate memory bandwidth (these kernels are
        // all bandwidth bound).
        inline unsigned gridFor(std::size_t n) {
            const std::size_t blocks = (n + kBlock - 1) / kBlock;
            return static_cast<unsigned>(blocks < 4096 ? (blocks == 0 ? 1 : blocks) : 4096);
        }

        template <typename T>
        __global__ void fillKernel(T* __restrict__ dst, std::size_t n, T value) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)gridDim.x * blockDim.x)
                dst[i] = value;
        }

        template <typename From, typename To>
        __global__ void convertKernel(const From* __restrict__ src, To* __restrict__ dst, std::size_t n) {
            for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
                 i += (std::size_t)gridDim.x * blockDim.x)
                dst[i] = detail::convertScalar<To>(src[i]);
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

        // std::complex is not usable in device code; fill it via the
        // layout-compatible CUDA complex types.
        template <typename T> struct DeviceRep {
            using type = T;
        };
        template <> struct DeviceRep<std::complex<float>> {
            using type = cuFloatComplex;
        };
        template <> struct DeviceRep<std::complex<double>> {
            using type = cuDoubleComplex;
        };
    } // namespace

    template <typename T>
    void fillDevice(T* dst, std::size_t n, T value, cudaStream_t stream) {
        if (n == 0) return;
        using R = typename DeviceRep<T>::type;
        R v;
        static_assert(sizeof(R) == sizeof(T), "device representation must match");
        std::memcpy(&v, &value, sizeof(T));   // host-side bit copy; MSVC has no __builtin_memcpy
        fillKernel<R><<<gridFor(n), kBlock, 0, stream>>>(reinterpret_cast<R*>(dst), n, v);
    }

    template <typename From, typename To>
    void convertDevice(const From* src, To* dst, std::size_t n, cudaStream_t stream) {
        if (n == 0) return;
        convertKernel<From, To><<<gridFor(n), kBlock, 0, stream>>>(src, dst, n);
    }

    void scaleComplexDouble(std::complex<double>* p, std::size_t n, double scale, cudaStream_t stream) {
        if (n == 0) return;
        scaleKernel<<<gridFor(n), kBlock, 0, stream>>>(reinterpret_cast<cuDoubleComplex*>(p), n, scale);
    }

    void scaleDouble(double* p, std::size_t n, double scale, cudaStream_t stream) {
        if (n == 0) return;
        scaleRealKernel<<<gridFor(n), kBlock, 0, stream>>>(p, n, scale);
    }

    // Explicit instantiations for the SIRIUS pixel/scalar types.
    // clang-format off: one instantiation per line (the formatter does not
    // settle on a layout for a chain of statement-like macros)
#define SIRIUS_FILL(T) template void fillDevice<T>(T*, std::size_t, T, cudaStream_t);
    SIRIUS_FILL(std::uint8_t)
    SIRIUS_FILL(std::int8_t)
    SIRIUS_FILL(std::uint16_t)
    SIRIUS_FILL(std::int16_t)
    SIRIUS_FILL(std::uint32_t)
    SIRIUS_FILL(std::int32_t)
    SIRIUS_FILL(float)
    SIRIUS_FILL(double)
    SIRIUS_FILL(std::complex<float>)
    SIRIUS_FILL(std::complex<double>)
#undef SIRIUS_FILL

#define SIRIUS_CONVERT_TO(From, To) template void convertDevice<From, To>(const From*, To*, std::size_t, cudaStream_t);
#define SIRIUS_CONVERT_FROM(From) \
    SIRIUS_CONVERT_TO(From, std::uint8_t) \
    SIRIUS_CONVERT_TO(From, std::int8_t) \
    SIRIUS_CONVERT_TO(From, std::uint16_t) \
    SIRIUS_CONVERT_TO(From, std::int16_t) \
    SIRIUS_CONVERT_TO(From, std::uint32_t) \
    SIRIUS_CONVERT_TO(From, std::int32_t) \
    SIRIUS_CONVERT_TO(From, float) \
    SIRIUS_CONVERT_TO(From, double)
    SIRIUS_CONVERT_FROM(std::uint8_t)
    SIRIUS_CONVERT_FROM(std::int8_t)
    SIRIUS_CONVERT_FROM(std::uint16_t)
    SIRIUS_CONVERT_FROM(std::int16_t)
    SIRIUS_CONVERT_FROM(std::uint32_t)
    SIRIUS_CONVERT_FROM(std::int32_t)
    SIRIUS_CONVERT_FROM(float)
    SIRIUS_CONVERT_FROM(double)
#undef SIRIUS_CONVERT_FROM
#undef SIRIUS_CONVERT_TO
    // clang-format on

} // namespace sirius::cuda
