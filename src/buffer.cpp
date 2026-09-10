#include "sirius/buffer.hpp"

#include "sirius/errors.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <stdexcept>

#ifdef SIRIUS_HAS_CUDA
#include "cuda_check.hpp"
#include "cuda/kernels.hpp"
#endif

namespace sirius {

    // --- Shape ------------------------------------------------------------

    Shape::Shape(std::initializer_list<Index> dims) {
        for (Index d : dims) push(d);
    }

    void Shape::push(Index dim) {
        if (rank_ >= kMaxRank)
            throw std::invalid_argument("Shape: rank exceeds " + std::to_string(kMaxRank));
        if (dim < 0)
            throw std::invalid_argument("Shape: negative dimension " + std::to_string(dim));
        dims_[static_cast<std::size_t>(rank_++)] = dim;
    }

    Index Shape::numel() const {
        if (rank_ == 0) return 0;
        return detail::checkedProduct(dims_.data(), dims_.data() + rank_, "Shape::numel");
    }

    Shape Shape::asStack() const {
        switch (rank_) {
            case 0: return Shape{0, 0, 0};
            case 1: return Shape{1, 1, dims_[0]};
            case 2: return Shape{1, dims_[0], dims_[1]};
            case 3: return *this;
            default: {
                const Index lead = detail::checkedProduct(dims_.data(), dims_.data() + (rank_ - 2), "Shape::asStack");
                return Shape{lead, dims_[static_cast<std::size_t>(rank_ - 2)],
                             dims_[static_cast<std::size_t>(rank_ - 1)]};
            }
        }
    }

    std::string Shape::toString() const {
        std::string s = "(";
        for (int i = 0; i < rank_; ++i) {
            if (i) s += ", ";
            s += std::to_string(dims_[static_cast<std::size_t>(i)]);
        }
        return s + ")";
    }

    // --- untyped memory primitives -----------------------------------------

    namespace detail {

        namespace {
            constexpr std::size_t kHostAlignment = 64;   // cache line / AVX-512

            void* allocateHost(std::size_t bytes) {
                // Round up so the aligned-new contract (size % alignment == 0) holds.
                const std::size_t rounded = (bytes + kHostAlignment - 1) / kHostAlignment * kHostAlignment;
                return ::operator new(rounded, std::align_val_t{kHostAlignment});
            }
            void deallocateHost(void* p) noexcept {
                ::operator delete(p, std::align_val_t{kHostAlignment});
            }

#ifdef SIRIUS_HAS_CUDA
            // Keep freed device memory in the pool instead of returning it to
            // the driver; re-allocation then costs no cudaMalloc round trip.
            void configurePoolOnce(int deviceIndex) {
                static std::once_flag flags[64];
                if (deviceIndex < 0 || deviceIndex >= 64) return;
                std::call_once(flags[deviceIndex], [deviceIndex] {
                    cudaMemPool_t pool = nullptr;
                    if (cudaDeviceGetDefaultMemPool(&pool, deviceIndex) != cudaSuccess) {
                        (void)cudaGetLastError();
                        return;
                    }
                    std::uint64_t threshold = UINT64_MAX;
                    if (cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold) != cudaSuccess)
                        (void)cudaGetLastError();
                });
            }

            cudaMemcpyKind kindFor(Device src, Device dst) noexcept {
                if (src.isCpu() && dst.isCpu()) return cudaMemcpyHostToHost;
                if (src.isCpu()) return cudaMemcpyHostToDevice;
                if (dst.isCpu()) return cudaMemcpyDeviceToHost;
                return cudaMemcpyDeviceToDevice;
            }
#endif

            // The CUDA stream to enqueue on, validated against the device the
            // operation touches. A CPU/null Stream maps to the legacy default
            // stream of that device.
            [[maybe_unused]] void checkStreamDevice(const Stream& stream, Device device) {
                if (stream.device().isCuda() && device.isCuda() && stream.device() != device)
                    throw std::runtime_error("Stream on " + toString(stream.device()) +
                                             " cannot be used with memory on " + toString(device));
            }
        } // namespace

        void* allocateBytes(std::size_t bytes, Device device, HostMemory host, const Stream& stream) {
            if (bytes == 0) return nullptr;
            requireDevice(device);
            if (device.isCpu() && host == HostMemory::Pageable)
                return allocateHost(bytes);
#ifdef SIRIUS_HAS_CUDA
            if (device.isCpu()) {   // pinned host memory
                void* p = nullptr;
                cuda::check(cudaHostAlloc(&p, bytes, cudaHostAllocPortable), "cudaHostAlloc");
                return p;
            }
            checkStreamDevice(stream, device);
            cuda::DeviceGuard g(device.index);
            configurePoolOnce(device.index);
            void* p = nullptr;
            cuda::check(cudaMallocAsync(&p, bytes, cuda::handle(stream)),
                        "cudaMallocAsync");
            return p;
#else
            (void)stream;
            throw std::runtime_error("SIRIUS was built without CUDA support; cannot allocate " +
                                     std::string(device.isCpu() ? "pinned host" : "device") + " memory");
#endif
        }

        void deallocateBytes(void* p, Device device, HostMemory host) noexcept {
            if (!p) return;
            if (device.isCpu() && host == HostMemory::Pageable) {
                deallocateHost(p);
                return;
            }
#ifdef SIRIUS_HAS_CUDA
            if (device.isCpu()) {
                (void)cudaFreeHost(p);
                return;
            }
            // Freed on the legacy NULL stream: it is ordered after all work
            // previously enqueued on any blocking stream of this device, so a
            // buffer can safely outlive the Stream it was allocated on (a
            // common pattern: allocate inside a reader on a scratch stream,
            // return the buffer). No host synchronization is involved.
            int previous = 0;
            if (cudaGetDevice(&previous) == cudaSuccess && previous != device.index)
                (void)cudaSetDevice(device.index);
            else
                previous = device.index;
            (void)cudaFreeAsync(p, nullptr);
            if (previous != device.index) (void)cudaSetDevice(previous);
            (void)cudaGetLastError();
#endif
        }

        void copyBytes(const void* src, Device srcDevice, void* dst, Device dstDevice,
                       std::size_t bytes, const Stream& stream) {
            if (bytes == 0) return;
            if (srcDevice.isCpu() && dstDevice.isCpu()) {
                std::memcpy(dst, src, bytes);
                return;
            }
            requireDevice(srcDevice);
            requireDevice(dstDevice);
#ifdef SIRIUS_HAS_CUDA
            const Device ctx = srcDevice.isCuda() ? srcDevice : dstDevice;
            checkStreamDevice(stream, ctx);
            cuda::DeviceGuard g(ctx.index);
            if (srcDevice.isCuda() && dstDevice.isCuda() && srcDevice != dstDevice) {
                cuda::check(cudaMemcpyPeerAsync(dst, dstDevice.index, src, srcDevice.index, bytes,
                                                cuda::handle(stream)),
                            "cudaMemcpyPeerAsync");
                return;
            }
            // cudaMemcpyAsync degrades to a synchronous staged copy for
            // pageable host memory and stays asynchronous for pinned memory --
            // exactly the behaviour HostMemory documents.
            cuda::check(cudaMemcpyAsync(dst, src, bytes, kindFor(srcDevice, dstDevice), cuda::handle(stream)),
                        "cudaMemcpyAsync");
#else
            (void)stream;
#endif
        }

        void memsetBytes(void* dst, Device device, int value, std::size_t bytes, const Stream& stream) {
            if (bytes == 0) return;
            if (device.isCpu()) {
                std::memset(dst, value, bytes);
                return;
            }
            requireDevice(device);
#ifdef SIRIUS_HAS_CUDA
            checkStreamDevice(stream, device);
            cuda::DeviceGuard g(device.index);
            cuda::check(cudaMemsetAsync(dst, value, bytes, cuda::handle(stream)), "cudaMemsetAsync");
#else
            (void)stream;
#endif
        }

        void throwShapeMismatch(const char* what, const Shape& a, const Shape& b) {
            throw ShapeError(std::string(what) + ": shape mismatch " + a.toString() +
                             " vs " + b.toString());
        }

        void checkSameDevice(Device a, Device b, const char* what) {
            if (a != b)
                throw std::invalid_argument(std::string(what) + ": operands live on different devices (" +
                                            toString(a) + " vs " + toString(b) + ")");
        }

    } // namespace detail

    // --- fill / convert -----------------------------------------------------

    template <typename T>
    void fill(BufferView<T> dst, T value, const Stream& stream) {
        if (dst.empty()) return;
        if (dst.device().isCpu()) {
            std::fill(dst.data(), dst.data() + dst.size(), value);
            return;
        }
        requireDevice(dst.device());
#ifdef SIRIUS_HAS_CUDA
        detail::checkStreamDevice(stream, dst.device());
        cuda::DeviceGuard g(dst.device().index);
        cuda::fillDevice<T>(dst.data(), static_cast<std::size_t>(dst.size()), value, cuda::handle(stream));
        cuda::check(cudaGetLastError(), "fill kernel launch");
#else
        (void)stream;
#endif
    }

    template <typename From, typename To>
    void convert(BufferView<const From> src, BufferView<To> dst, const Stream& stream) {
        if (src.shape() != dst.shape()) detail::throwShapeMismatch("convert", src.shape(), dst.shape());
        detail::checkSameDevice(src.device(), dst.device(), "convert");
        if (src.empty()) return;
        const auto n = static_cast<std::size_t>(src.size());
        if (src.device().isCpu()) {
            const From* s = src.data();
            To* d = dst.data();
            if constexpr (std::is_same_v<From, To>) {
                if (s != d) std::memcpy(d, s, n * sizeof(To));
            } else {
                // Large conversions parallelize across cores; the loop
                // vectorizes on its own for every scalar pair.
                const auto ni = static_cast<std::ptrdiff_t>(n);
#pragma omp parallel for if (ni > (1 << 20)) schedule(static)
                for (std::ptrdiff_t i = 0; i < ni; ++i) d[i] = detail::convertScalar<To>(s[i]);
            }
            return;
        }
        requireDevice(src.device());
#ifdef SIRIUS_HAS_CUDA
        detail::checkStreamDevice(stream, src.device());
        cuda::DeviceGuard g(src.device().index);
        if constexpr (std::is_same_v<From, To>) {
            cuda::check(cudaMemcpyAsync(dst.data(), src.data(), n * sizeof(To), cudaMemcpyDeviceToDevice,
                                        cuda::handle(stream)),
                        "cudaMemcpyAsync");
        } else {
            cuda::convertDevice<From, To>(src.data(), dst.data(), n, cuda::handle(stream));
            cuda::check(cudaGetLastError(), "convert kernel launch");
        }
#else
        (void)stream;
#endif
    }

    // Explicit instantiations for the supported element types.
    // clang-format off: one instantiation per line (the formatter does not
    // settle on a layout for a chain of statement-like macros)
#define SIRIUS_FILL(T) template void fill<T>(BufferView<T>, T, const Stream&);
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

#define SIRIUS_CONVERT_TO(From, To) \
    template void convert<From, To>(BufferView<const From>, BufferView<To>, const Stream&);
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

} // namespace sirius
