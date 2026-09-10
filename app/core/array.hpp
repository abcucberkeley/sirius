#ifndef SIRIUS_APP_ARRAY_HPP
#define SIRIUS_APP_ARRAY_HPP

// The workbench's array model. Every dataset and every step output is a
// five-dimensional (c, t, z, y, x) float32 array in host memory; x is the
// fastest axis, exactly as the library's (planes, rows, cols) buffers and the
// TIFF pages that feed them. Axis semantics never move: reducing an axis
// leaves a length-1 axis behind, so downstream steps and the viewer never
// have to guess which axis is which.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <sirius/buffer.hpp>
#include <sirius/checked_math.hpp>

namespace sirius::app {

    enum class Axis : int { C = 0,
                            T = 1,
                            Z = 2,
                            Y = 3,
                            X = 4 };
    constexpr int kAxisCount = 5;
    constexpr std::array<Axis, kAxisCount> kAxes{Axis::C, Axis::T, Axis::Z, Axis::Y, Axis::X};

    char axisLetter(Axis a) noexcept;                    // 'c' 't' 'z' 'y' 'x'
    std::optional<Axis> axisFromLetter(char letter) noexcept;

    struct Dims5 {
        Index c = 1, t = 1, z = 1, y = 1, x = 1;

        Index operator[](Axis a) const noexcept;
        Index& operator[](Axis a) noexcept;
        // Checked: the extents come from a file header or a step's output
        // metadata, so a wrapped product would size an allocation far too
        // small. Throws std::overflow_error instead (sirius/checked_math.hpp).
        Index numel() const {
            const std::array<Index, kAxisCount> e{c, t, z, y, x};
            return sirius::detail::checkedProduct(e.begin(), e.end(), "Dims5::numel");
        }
        Index planes() const {   // checked like numel(): the extents come from files
            const std::array<Index, 3> e{c, t, z};
            return sirius::detail::checkedProduct(e.begin(), e.end(), "Dims5::planes");
        }
        Index planeIndex(Index ci, Index ti, Index zi) const noexcept { return (ci * t + ti) * z + zi; }
        Index planeSize() const noexcept { return y * x; }
        std::size_t bytes() const { return sirius::detail::checkedBytes(numel(), sizeof(float), "Dims5::bytes"); }
        // "c2 t40 z48 y2048 x2048"
        std::string toString() const;
        // "2 x 40 x 48 x 2048 x 2048"
        std::string toProduct() const;

        friend bool operator==(const Dims5& a, const Dims5& b) noexcept {
            return a.c == b.c && a.t == b.t && a.z == b.z && a.y == b.y && a.x == b.x;
        }
        friend bool operator!=(const Dims5& a, const Dims5& b) noexcept { return !(a == b); }
    };

    // Owning (c, t, z, y, x) float32 host array. Storage is a rank-3
    // (planes, y, x) Buffer so any (c, t) volume or (c, t, z) plane is a
    // contiguous view the library's algorithms accept directly.
    class Array5 {
    public:
        Array5() = default;
        explicit Array5(Dims5 dims);                 // uninitialized
        static Array5 zeros(Dims5 dims);
        static Array5 filled(Dims5 dims, float value);
        // Adopt an existing (z, y, x) or (planes, y, x) host buffer as (1, 1, planes, y, x)
        // unless `dims` says otherwise (its plane count must match).
        static Array5 fromBuffer(Buffer<float> buffer, std::optional<Dims5> dims = std::nullopt);

        const Dims5& dims() const noexcept { return dims_; }
        bool empty() const noexcept { return data_.empty(); }
        Index numel() const { return dims_.numel(); }          // may throw, see Dims5::numel
        std::size_t bytes() const { return data_.bytes(); }

        float* data() noexcept { return data_.data(); }
        const float* data() const noexcept { return data_.data(); }
        float* plane(Index c, Index t, Index z) noexcept;
        const float* plane(Index c, Index t, Index z) const noexcept;
        float& at(Index c, Index t, Index z, Index y, Index x) noexcept;
        float at(Index c, Index t, Index z, Index y, Index x) const noexcept;

        // (z, y, x) view of one channel at one time point.
        BufferView<float> volume(Index c, Index t) noexcept;
        BufferView<const float> volume(Index c, Index t) const noexcept;
        // (planes, y, x) view of everything.
        BufferView<float> stack() noexcept { return data_.view(); }
        BufferView<const float> stack() const noexcept { return data_.view(); }
        Buffer<float>& buffer() noexcept { return data_; }
        const Buffer<float>& buffer() const noexcept { return data_; }

        Array5 clone() const;

    private:
        Dims5 dims_;
        Buffer<float> data_;
    };

    using ArrayPtr = std::shared_ptr<const Array5>;

    // Min / max over an array, NaN ignored (both 0 when empty).
    std::pair<float, float> minMax(const Array5& a) noexcept;

} // namespace sirius::app

#endif // SIRIUS_APP_ARRAY_HPP
