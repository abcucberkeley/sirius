#ifndef SIRIUS_DOWNSAMPLE_HPP
#define SIRIUS_DOWNSAMPLE_HPP

// Box-mean down-sampling shared by the TIFF pyramid writer, the OME-NGFF
// multiscale writer and the public image_ops::downsampleBox. One arithmetic
// for all of them: every box is accumulated in double, in C order, and the
// mean is rounded to nearest for integer pixels (std::llround) and converted
// as-is for floating-point ones. Partial boxes at the far edges average what
// is inside them.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <type_traits>
#include <vector>

#include "sirius/checked_math.hpp"
#include "sirius/image_ops.hpp"

namespace sirius::detail {

    // Shape of the down-sampled array: ceil(shape / factor) per axis.
    inline std::vector<Index> downsampledShape(const std::vector<Index>& shape, const std::vector<int>& factors) {
        std::vector<Index> out(shape.size());
        for (std::size_t i = 0; i < shape.size(); ++i) out[i] = downsampledExtent(shape[i], factors[i]);
        return out;
    }

    // Element count of `shape`; throws std::overflow_error rather than wrap.
    inline Index elementCount(const std::vector<Index>& shape) {
        return static_cast<Index>(checkedProduct(shape.begin(), shape.end(), "downsample"));
    }

    // Mean of one box, stored as T: integers round to nearest, floats convert.
    template <typename T>
    inline T boxMean(double acc, Index count) noexcept {
        const double v = count ? acc / static_cast<double>(count) : 0.0;
        if constexpr (std::is_integral_v<T>) return static_cast<T>(std::llround(v));
        else return static_cast<T>(v);
    }

    // Box-mean of `in` (C order over `shape`, every extent >= 1) by integer
    // `factors` (>= 1) per axis into `out`, which holds
    // elementCount(downsampledShape(shape, factors)) values. Parallel over
    // the output lines along the last axis.
    template <typename T>
    void downsampleBoxMean(const T* in, const std::vector<Index>& shape, const std::vector<int>& factors, T* out) {
        const std::size_t r = shape.size();
        if (r == 0) {   // a scalar: the one box holds the one value
            out[0] = boxMean<T>(static_cast<double>(in[0]), 1);
            return;
        }
        const std::vector<Index> outShape = downsampledShape(shape, factors);
        const std::size_t last = r - 1;
        const Index lastIn = shape[last], lastOut = outShape[last];
        const int lastF = factors[last];
        // strides of the input in elements and of the output in lines along the last axis
        std::vector<Index> inStride(r, 1), lineStride(last, 1);
        for (std::size_t k = last; k-- > 0;) inStride[k] = inStride[k + 1] * shape[k + 1];
        for (std::size_t k = last; k-- > 1;) lineStride[k - 1] = lineStride[k] * outShape[k];
        const Index lines = last ? lineStride[0] * outShape[0] : Index{1};

#pragma omp parallel
        {
            std::vector<Index> oi(last, 0);   // output multi-index over the outer axes
            std::vector<Index> bi(last, 0);   // input multi-index walking one box's lines
            std::vector<Index> bases;         // input offsets of the box's lines, C order
#pragma omp for schedule(static)
            for (Index o = 0; o < lines; ++o) {
                Index rem = o;
                for (std::size_t k = 0; k < last; ++k) {
                    oi[k] = rem / lineStride[k];
                    rem -= oi[k] * lineStride[k];
                    bi[k] = oi[k] * factors[k];
                }
                bases.clear();
                for (bool more = true; more;) {
                    Index base = 0;
                    for (std::size_t k = 0; k < last; ++k) base += bi[k] * inStride[k];
                    bases.push_back(base);
                    more = false;   // advance the box odometer, innermost outer axis first
                    for (std::size_t k = last; k-- > 0;) {
                        if (++bi[k] < std::min<Index>((oi[k] + 1) * factors[k], shape[k])) {
                            more = true;
                            break;
                        }
                        bi[k] = oi[k] * factors[k];
                    }
                }
                T* row = out + o * lastOut;
                for (Index x = 0; x < lastOut; ++x) {
                    const Index x0 = x * lastF, x1 = std::min<Index>(x0 + lastF, lastIn);
                    double acc = 0.0;
                    Index count = 0;
                    for (Index base : bases) {
                        const T* line = in + base;
                        for (Index sx = x0; sx < x1; ++sx, ++count) acc += static_cast<double>(line[sx]);
                    }
                    row[x] = boxMean<T>(acc, count);
                }
            }
        }
    }

} // namespace sirius::detail

#endif // SIRIUS_DOWNSAMPLE_HPP
