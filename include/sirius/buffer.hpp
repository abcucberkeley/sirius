#ifndef SIRIUS_BUFFER_HPP
#define SIRIUS_BUFFER_HPP

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <type_traits>
#include <utility>

#include <unsupported/Eigen/CXX11/Tensor>

#include "sirius/checked_math.hpp"
#include "sirius/device.hpp"

// Buffer<T>: owning, contiguous, row-major, typed storage on a Device.
// BufferView<T>: non-owning pointer + Shape + Device, the currency algorithms
// accept so they don't care who owns the memory (a Buffer, an Eigen tensor, a
// numpy/torch array from the bindings, an MPI window later).
//
// Design notes (following the usual performance rules):
//  * No implicit copies. Buffer is move-only; clone()/to() copy explicitly.
//  * Allocation is the expensive operation: device memory comes from the
//    per-device CUDA memory pool (cudaMallocAsync) so repeated alloc/free of
//    working buffers is cheap, and host memory is 64-byte aligned for SIMD.
//  * Everything taking a Stream is asynchronous and stream-ordered on CUDA.
//    Host<->device copies are truly asynchronous only when the host side is
//    pinned (HostMemory::Pinned); pageable memory forces a staging copy.
//  * Row-major, innermost dimension contiguous -- the same layout as the Eigen
//    RowMajor tensors used throughout SIRIUS and as TIFF scanlines, so
//    Eigen<->Buffer interop is a pointer reinterpretation, not a transpose.

namespace sirius {

    using Index = Eigen::Index;   // ptrdiff_t: matches Eigen tensors and dev-side helpers

    // --- Shape ------------------------------------------------------------
    // Up to 4 dimensions: e.g. {z, y, x} for a stack, {phase*dir, z, y, x}
    // for a raw SIM acquisition. Rank 0 denotes an empty buffer.
    class Shape {
    public:
        static constexpr int kMaxRank = 4;

        constexpr Shape() noexcept = default;
        Shape(std::initializer_list<Index> dims);
        template <typename It>
        Shape(It first, It last) {
            for (; first != last; ++first) push(static_cast<Index>(*first));
        }
        template <typename T, int Rank, int Options>
        explicit Shape(const Eigen::Tensor<T, Rank, Options>& t) {
            for (int i = 0; i < Rank; ++i) push(t.dimension(i));
        }

        int rank() const noexcept { return rank_; }
        Index operator[](int i) const noexcept { return dims_[static_cast<std::size_t>(i)]; }
        Index& operator[](int i) noexcept { return dims_[static_cast<std::size_t>(i)]; }
        const Index* data() const noexcept { return dims_.data(); }
        // Element count; throws std::overflow_error when the extents do not
        // multiply within Index.
        Index numel() const;
        bool empty() const noexcept {   // numel() == 0, without the product
            if (rank_ == 0) return true;
            for (int i = 0; i < rank_; ++i)
                if (dims_[static_cast<std::size_t>(i)] == 0) return true;
            return false;
        }
        void push(Index dim);   // append a dimension (throws past kMaxRank)

        // Rank-3 shape {d0..d(rank-1) with leading dims collapsed}. Buffers that
        // model image stacks are (depth, rows, cols); a rank-2 image is (1, rows, cols).
        Shape asStack() const;

        std::string toString() const;

        friend bool operator==(const Shape& a, const Shape& b) noexcept {
            if (a.rank_ != b.rank_) return false;
            for (int i = 0; i < a.rank_; ++i)
                if (a.dims_[static_cast<std::size_t>(i)] != b.dims_[static_cast<std::size_t>(i)]) return false;
            return true;
        }
        friend bool operator!=(const Shape& a, const Shape& b) noexcept { return !(a == b); }

    private:
        std::array<Index, kMaxRank> dims_{};
        int rank_ = 0;
    };

    enum class HostMemory : std::uint8_t {
        Pageable, // std::malloc-style; cheapest to allocate, synchronous copies to/from the GPU
        Pinned    // cudaHostAlloc; required for asynchronous, DMA-driven H2D/D2H transfers
    };

    // --- BufferView -------------------------------------------------------
    template <typename T>
    class BufferView {
    public:
        using value_type = std::remove_const_t<T>;

        constexpr BufferView() noexcept = default;
        BufferView(T* data, Shape shape, Device device) noexcept
            : data_(data), shape_(std::move(shape)), device_(device) {}

        // BufferView<T> -> BufferView<const T>
        template <typename U, std::enable_if_t<std::is_same_v<const U, T> && !std::is_const_v<U>, int> = 0>
        BufferView(const BufferView<U>& other) noexcept
            : data_(other.data()), shape_(other.shape()), device_(other.device()) {}

        T* data() const noexcept { return data_; }
        const Shape& shape() const noexcept { return shape_; }
        Device device() const noexcept { return device_; }
        int rank() const noexcept { return shape_.rank(); }
        Index dim(int i) const noexcept { return shape_[i]; }
        Index size() const { return shape_.numel(); }   // throws std::overflow_error, see Shape::numel
        std::size_t bytes() const { return detail::checkedBytes(size(), sizeof(T), "BufferView::bytes"); }
        bool empty() const noexcept { return shape_.empty(); }

        // Contiguous sub-block along dimension 0: elements [first, first+count).
        BufferView slice(Index first, Index count) const;
        // Same memory, different shape (element count must match).
        BufferView reshape(Shape shape) const;
        // Rank-3 view (depth, rows, cols); see Shape::asStack.
        BufferView asStack() const { return reshape(shape_.asStack()); }

    private:
        T* data_ = nullptr;
        Shape shape_;
        Device device_ = Device::cpu();
    };

    template <typename T>
    using ConstBufferView = BufferView<const T>;

    // --- Buffer -----------------------------------------------------------
    template <typename T>
    class Buffer {
        static_assert(std::is_trivially_copyable_v<T>, "Buffer<T> requires a trivially copyable T");

    public:
        using value_type = T;

        Buffer() noexcept = default;
        // Uninitialized storage of `shape` on `device`. CUDA allocations are
        // stream-ordered on `stream` (pass the stream that will first touch
        // the memory to avoid an implicit device synchronization).
        explicit Buffer(Shape shape, Device device = Device::cpu(),
                        HostMemory host = HostMemory::Pageable,
                        const Stream& stream = Stream::null());
        ~Buffer() { reset(); }

        Buffer(Buffer&& other) noexcept { swap(other); }
        Buffer& operator=(Buffer&& other) noexcept {
            if (this != &other) {
                reset();
                swap(other);
            }
            return *this;
        }
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;

        // Wrap memory owned elsewhere; `deleter` runs on destruction (may be empty).
        static Buffer adopt(T* data, Shape shape, Device device, std::function<void(T*)> deleter,
                            HostMemory host = HostMemory::Pageable);

        T* data() noexcept { return data_; }
        const T* data() const noexcept { return data_; }
        const Shape& shape() const noexcept { return shape_; }
        int rank() const noexcept { return shape_.rank(); }
        Index dim(int i) const noexcept { return shape_[i]; }
        Index size() const { return shape_.numel(); }   // throws std::overflow_error, see Shape::numel
        std::size_t bytes() const { return detail::checkedBytes(size(), sizeof(T), "Buffer::bytes"); }
        bool empty() const noexcept { return data_ == nullptr; }
        Device device() const noexcept { return device_; }
        HostMemory hostMemory() const noexcept { return host_; }
        bool pinned() const noexcept { return device_.isCpu() && host_ == HostMemory::Pinned; }

        BufferView<T> view() noexcept { return {data_, shape_, device_}; }
        BufferView<const T> view() const noexcept { return {data_, shape_, device_}; }
        operator BufferView<T>() noexcept { return view(); }
        operator BufferView<const T>() const noexcept { return view(); }

        // Explicit copies.
        Buffer clone(const Stream& stream = Stream::null()) const;
        Buffer to(Device device, const Stream& stream = Stream::null(),
                  HostMemory host = HostMemory::Pageable) const;

        // Reinterpret the same allocation with another shape of equal element count.
        void reshape(Shape shape);

        void swap(Buffer& other) noexcept {
            std::swap(data_, other.data_);
            std::swap(shape_, other.shape_);
            std::swap(device_, other.device_);
            std::swap(host_, other.host_);
            std::swap(deleter_, other.deleter_);
        }
        void reset() noexcept;

    private:
        T* data_ = nullptr;
        Shape shape_;
        Device device_ = Device::cpu();
        HostMemory host_ = HostMemory::Pageable;
        std::function<void(T*)> deleter_;   // set only by adopt()
    };

    // --- untyped primitives (implemented in buffer.cpp) ---------------------
    namespace detail {
        void* allocateBytes(std::size_t bytes, Device device, HostMemory host, const Stream& stream);
        void deallocateBytes(void* p, Device device, HostMemory host) noexcept;
        // Any src/dst device combination. Asynchronous on CUDA streams when
        // both sides are device or pinned memory.
        void copyBytes(const void* src, Device srcDevice, void* dst, Device dstDevice,
                       std::size_t bytes, const Stream& stream);
        void memsetBytes(void* dst, Device device, int value, std::size_t bytes, const Stream& stream);
        // Every rank / extent mismatch in the library funnels through here
        // and arrives as a sirius::ShapeError (sirius/errors.hpp).
        [[noreturn]] void throwShapeMismatch(const char* what, const Shape& a, const Shape& b);
        void checkSameDevice(Device a, Device b, const char* what);
    } // namespace detail

    // --- view adaptors ----------------------------------------------------
    // Everything below accepts a Buffer, a BufferView, or a row-major Eigen
    // tensor (host memory) so that call sites read naturally:
    //     copy(eigenStack, deviceBuffer, stream);
    template <typename T> BufferView<const T> toConstView(const Buffer<T>& b) noexcept { return b.view(); }
    template <typename T> BufferView<const T> toConstView(BufferView<T> v) noexcept { return v; }
    template <typename T> BufferView<const T> toConstView(BufferView<const T> v) noexcept { return v; }
    template <typename T, int Rank>
    BufferView<const T> toConstView(const Eigen::Tensor<T, Rank, Eigen::RowMajor>& t) noexcept {
        return {t.data(), Shape(t), Device::cpu()};
    }
    template <typename T> BufferView<T> toView(Buffer<T>& b) noexcept { return b.view(); }
    template <typename T> BufferView<T> toView(BufferView<T> v) noexcept { return v; }
    template <typename T, int Rank>
    BufferView<T> toView(Eigen::Tensor<T, Rank, Eigen::RowMajor>& t) noexcept {
        return {t.data(), Shape(t), Device::cpu()};
    }

    // Zero-copy Eigen tensor maps over host views (throws for device memory
    // or a rank mismatch).
    template <typename T, int Rank>
    Eigen::TensorMap<Eigen::Tensor<T, Rank, Eigen::RowMajor>> asTensor(BufferView<T> v);
    template <typename T, int Rank>
    Eigen::TensorMap<Eigen::Tensor<const T, Rank, Eigen::RowMajor>> asTensor(BufferView<const T> v);

    // --- algorithms -------------------------------------------------------
    // copy: element-for-element, shapes must match, any device combination.
    template <typename Src, typename Dst>
    void copy(const Src& src, Dst&& dst, const Stream& stream = Stream::null()) {
        auto s = toConstView(src);
        auto d = toView(dst);
        static_assert(std::is_same_v<typename decltype(s)::value_type, typename decltype(d)::value_type>,
                      "copy() requires identical element types; use convert()");
        if (s.shape() != d.shape()) detail::throwShapeMismatch("copy", s.shape(), d.shape());
        detail::copyBytes(s.data(), s.device(), d.data(), d.device(), s.bytes(), stream);
    }

    // fill: every element = value (a kernel on CUDA, std::fill on the host).
    template <typename T> void fill(BufferView<T> dst, T value, const Stream& stream = Stream::null());
    template <typename T> void fill(Buffer<T>& dst, T value, const Stream& stream = Stream::null()) {
        fill(dst.view(), value, stream);
    }

    // convert: elementwise static_cast<To>(from) between the scalar pixel
    // types (u/int 8..32, float, double). src and dst must be on the same device.
    template <typename From, typename To>
    void convert(BufferView<const From> src, BufferView<To> dst, const Stream& stream = Stream::null());
    template <typename Src, typename Dst>
    void convert(const Src& src, Dst&& dst, const Stream& stream = Stream::null()) {
        auto s = toConstView(src);
        auto d = toView(dst);
        convert<typename decltype(s)::value_type, typename decltype(d)::value_type>(s, d, stream);
    }

    // Allocate on `device` and copy `src` into it.
    template <typename Src>
    auto toDevice(const Src& src, Device device, const Stream& stream = Stream::null(),
                  HostMemory host = HostMemory::Pageable) {
        auto s = toConstView(src);
        using T = typename decltype(s)::value_type;
        Buffer<T> out(s.shape(), device, host, stream);
        copy(s, out, stream);
        return out;
    }

    // Copy to the host as an owning row-major Eigen tensor (synchronizes `stream`).
    template <int Rank, typename Src>
    auto toEigen(const Src& src, const Stream& stream = Stream::null()) {
        auto s = toConstView(src);
        using T = typename decltype(s)::value_type;
        if (s.rank() != Rank) detail::throwShapeMismatch("toEigen rank", s.shape(), Shape{});
        Eigen::DSizes<Eigen::Index, Rank> dims;
        for (int i = 0; i < Rank; ++i) dims[i] = s.dim(i);
        Eigen::Tensor<T, Rank, Eigen::RowMajor> t(dims);
        copy(s, t, stream);
        stream.synchronize();
        return t;
    }

    // ---------------------------------------------------------------------
    // Template member definitions
    // ---------------------------------------------------------------------
    template <typename T>
    BufferView<T> BufferView<T>::slice(Index first, Index count) const {
        if (rank() == 0 || first < 0 || count < 0 || first + count > dim(0))
            throw std::out_of_range("BufferView::slice: [" + std::to_string(first) + ", " +
                                    std::to_string(first + count) + ") exceeds dimension 0 of " +
                                    shape_.toString());
        Shape s = shape_;
        s[0] = count;
        const Index inner = dim(0) == 0 ? 0 : size() / dim(0);
        return BufferView(data_ + first * inner, s, device_);
    }

    template <typename T>
    BufferView<T> BufferView<T>::reshape(Shape shape) const {
        if (shape.numel() != size()) detail::throwShapeMismatch("reshape", shape_, shape);
        return BufferView(data_, shape, device_);
    }

    template <typename T>
    Buffer<T>::Buffer(Shape shape, Device device, HostMemory host, const Stream& stream)
        : shape_(std::move(shape)), device_(device), host_(device.isCpu() ? host : HostMemory::Pageable) {
        if (shape_.numel() > 0)
            data_ = static_cast<T*>(detail::allocateBytes(bytes(), device_, host_, stream));
    }

    template <typename T>
    Buffer<T> Buffer<T>::adopt(T* data, Shape shape, Device device, std::function<void(T*)> deleter,
                               HostMemory host) {
        Buffer b;
        b.data_ = data;
        b.shape_ = std::move(shape);
        b.device_ = device;
        b.host_ = host;
        b.deleter_ = deleter ? std::move(deleter) : [](T*) {};
        return b;
    }

    template <typename T>
    void Buffer<T>::reset() noexcept {
        if (data_) {
            if (deleter_) deleter_(data_);
            else detail::deallocateBytes(data_, device_, host_);
        }
        data_ = nullptr;
        shape_ = Shape{};
        deleter_ = nullptr;
    }

    template <typename T>
    Buffer<T> Buffer<T>::clone(const Stream& stream) const {
        Buffer out(shape_, device_, host_, stream);
        if (size() > 0) detail::copyBytes(data_, device_, out.data_, out.device_, bytes(), stream);
        return out;
    }

    template <typename T>
    Buffer<T> Buffer<T>::to(Device device, const Stream& stream, HostMemory host) const {
        Buffer out(shape_, device, host, stream);
        if (size() > 0) detail::copyBytes(data_, device_, out.data_, out.device_, bytes(), stream);
        return out;
    }

    template <typename T>
    void Buffer<T>::reshape(Shape shape) {
        if (shape.numel() != size()) detail::throwShapeMismatch("reshape", shape_, shape);
        shape_ = std::move(shape);
    }

    template <typename T, int Rank>
    Eigen::TensorMap<Eigen::Tensor<T, Rank, Eigen::RowMajor>> asTensor(BufferView<T> v) {
        if (!v.device().isCpu())
            throw std::runtime_error("asTensor: view lives on " + toString(v.device()) + ", not host memory");
        if (v.rank() != Rank) detail::throwShapeMismatch("asTensor rank", v.shape(), Shape{});
        Eigen::DSizes<Eigen::Index, Rank> dims;
        for (int i = 0; i < Rank; ++i) dims[i] = v.dim(i);
        return Eigen::TensorMap<Eigen::Tensor<T, Rank, Eigen::RowMajor>>(v.data(), dims);
    }
    template <typename T, int Rank>
    Eigen::TensorMap<Eigen::Tensor<const T, Rank, Eigen::RowMajor>> asTensor(BufferView<const T> v) {
        if (!v.device().isCpu())
            throw std::runtime_error("asTensor: view lives on " + toString(v.device()) + ", not host memory");
        if (v.rank() != Rank) detail::throwShapeMismatch("asTensor rank", v.shape(), Shape{});
        Eigen::DSizes<Eigen::Index, Rank> dims;
        for (int i = 0; i < Rank; ++i) dims[i] = v.dim(i);
        return Eigen::TensorMap<Eigen::Tensor<const T, Rank, Eigen::RowMajor>>(v.data(), dims);
    }

} // namespace sirius

#endif // SIRIUS_BUFFER_HPP
