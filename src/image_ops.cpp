#include "sirius/image_ops.hpp"
#include "sirius/constants.hpp"
#include "downsample.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Generic host array operations. Every routine is parallel over its output
// planes (or rows when there is a single plane) and accumulates in double:
// float stores are cheap, float sums over a 4096² plane are not accurate.

namespace sirius {

    namespace {
        constexpr double kInf = std::numeric_limits<double>::infinity();

        void requireExtent(const Extent5& e, const char* what) {
            for (Index d : e)
                if (d < 1) throw std::invalid_argument(std::string(what) + ": every extent must be >= 1");
        }

        void requirePositive(Index z, Index y, Index x, const char* what) {
            if (z < 1 || y < 1 || x < 1)
                throw std::invalid_argument(std::string(what) + ": extents must be >= 1, got " + std::to_string(z) +
                                            " x " + std::to_string(y) + " x " + std::to_string(x));
        }

        // Fold one input value into an accumulator for the reduction.
        inline void accumulate(double& acc, double v, ReduceOp op) noexcept {
            switch (op) {
                case ReduceOp::Sum:
                case ReduceOp::Mean: acc += v; break;
                // `v > acc` is false for NaN, so NaN never replaces the running extreme
                case ReduceOp::Max: acc = v > acc ? v : acc; break;
                case ReduceOp::Min: acc = v < acc ? v : acc; break;
            }
        }

        inline double initialAccumulator(ReduceOp op) noexcept {
            switch (op) {
                case ReduceOp::Max: return -kInf;
                case ReduceOp::Min: return kInf;
                default: return 0.0;
            }
        }

        // Catmull-Rom weights for a fractional offset f in [0, 1) at taps -1..2.
        inline void cubicWeights(double f, double w[4]) noexcept {
            const double f2 = f * f, f3 = f2 * f;
            w[0] = 0.5 * (-f3 + 2.0 * f2 - f);
            w[1] = 0.5 * (3.0 * f3 - 5.0 * f2 + 2.0);
            w[2] = 0.5 * (-3.0 * f3 + 4.0 * f2 + f);
            w[3] = 0.5 * (f3 - f2);
        }

        // One axis of an interpolation: the integer taps and their weights.
        // An axis of extent 1 accepts coordinates within half a voxel of 0.
        struct AxisTaps {
            Index index[4];
            double weight[4];
            int count = 0;   // 0 = the coordinate is outside the input
        };

        inline AxisTaps axisTaps(double p, Index n, Interpolation interp) noexcept {
            AxisTaps t;
            if (n == 1) {
                if (p < -0.5 || p > 0.5) return t;
                t.index[0] = 0;
                t.weight[0] = 1.0;
                t.count = 1;
                return t;
            }
            if (interp == Interpolation::Nearest) {
                // a voxel covers [i - 0.5, i + 0.5): the grid reaches half a voxel past its centres
                if (!(p >= -0.5) || p >= static_cast<double>(n) - 0.5) return t;   // also rejects NaN
                t.index[0] = std::clamp<Index>(static_cast<Index>(std::lround(p)), 0, n - 1);
                t.weight[0] = 1.0;
                t.count = 1;
                return t;
            }
            if (!(p >= 0.0) || p > static_cast<double>(n - 1)) return t;   // also rejects NaN
            const double fl = std::floor(p);
            const Index i0 = static_cast<Index>(fl);
            const double f = p - fl;
            switch (interp) {
                case Interpolation::Nearest:
                    break;   // handled above
                case Interpolation::Linear:
                    t.index[0] = i0;
                    t.index[1] = std::min<Index>(i0 + 1, n - 1);
                    t.weight[0] = 1.0 - f;
                    t.weight[1] = f;
                    t.count = 2;
                    break;
                case Interpolation::Cubic: {
                    double w[4];
                    cubicWeights(f, w);
                    for (int k = 0; k < 4; ++k) {
                        t.index[k] = std::clamp<Index>(i0 - 1 + k, 0, n - 1);   // clamped taps at the edges
                        t.weight[k] = w[k];
                    }
                    t.count = 4;
                    break;
                }
            }
            return t;
        }
    } // namespace

    // --- reductions ---------------------------------------------------------

    Extent5 reducedExtent(const Extent5& extent, const std::array<bool, 5>& reduce) noexcept {
        Extent5 out = extent;
        for (std::size_t i = 0; i < 5; ++i)
            if (reduce[i]) out[i] = 1;
        return out;
    }

    void reduceAxes(const float* in, const Extent5& extent, const std::array<bool, 5>& reduce, ReduceOp op,
                    float* out) {
        requireExtent(extent, "reduceAxes");
        const Extent5 oe = reducedExtent(extent, reduce);
        const Index ic = extent[0], it = extent[1], iz = extent[2], iy = extent[3], ix = extent[4];
        const Index oc = oe[0], ot = oe[1], oz = oe[2], oy = oe[3], ox = oe[4];
        const Index inPlane = iy * ix, outPlane = oy * ox;
        const Index outPlanes = oc * ot * oz;
        // Every output value folds this many input values.
        double count = 1.0;
        for (std::size_t i = 0; i < 5; ++i)
            if (reduce[i]) count *= static_cast<double>(extent[i]);
        const double init = initialAccumulator(op);

#pragma omp parallel
        {
            std::vector<double> acc(static_cast<std::size_t>(outPlane));
#pragma omp for schedule(dynamic)
            for (Index op_ = 0; op_ < outPlanes; ++op_) {
                const Index ocI = op_ / (ot * oz), otI = (op_ / oz) % ot, ozI = op_ % oz;
                std::fill(acc.begin(), acc.end(), init);
                // ranges of the input planes this output plane draws from
                const Index c0 = reduce[0] ? 0 : ocI, c1 = reduce[0] ? ic : ocI + 1;
                const Index t0 = reduce[1] ? 0 : otI, t1 = reduce[1] ? it : otI + 1;
                const Index z0 = reduce[2] ? 0 : ozI, z1 = reduce[2] ? iz : ozI + 1;
                for (Index c = c0; c < c1; ++c)
                    for (Index t = t0; t < t1; ++t)
                        for (Index z = z0; z < z1; ++z) {
                            const float* plane = in + ((c * it + t) * iz + z) * inPlane;
                            for (Index y = 0; y < iy; ++y) {
                                const float* row = plane + y * ix;
                                double* accRow = acc.data() + (reduce[3] ? 0 : y * ox);
                                if (reduce[4]) {
                                    double a = accRow[0];
                                    for (Index x = 0; x < ix; ++x) accumulate(a, row[x], op);
                                    accRow[0] = a;
                                } else {
                                    for (Index x = 0; x < ix; ++x) accumulate(accRow[x], row[x], op);
                                }
                            }
                        }
                float* dst = out + op_ * outPlane;
                if (op == ReduceOp::Mean)
                    for (Index i = 0; i < outPlane; ++i) dst[i] = static_cast<float>(acc[static_cast<std::size_t>(i)] / count);
                else if (op == ReduceOp::Max || op == ReduceOp::Min)
                    // an all-NaN run leaves the sentinel: report NaN, not +-inf
                    for (Index i = 0; i < outPlane; ++i) {
                        const double a = acc[static_cast<std::size_t>(i)];
                        dst[i] = std::isinf(a) ? std::numeric_limits<float>::quiet_NaN() : static_cast<float>(a);
                    }
                else
                    for (Index i = 0; i < outPlane; ++i) dst[i] = static_cast<float>(acc[static_cast<std::size_t>(i)]);
            }
        }
    }

    // --- resampling -----------------------------------------------------------

    void resampleAffine(const float* in, Index iz, Index iy, Index ix, const std::array<double, 9>& A,
                        const std::array<double, 3>& b, float* out, Index oz, Index oy, Index ox,
                        Interpolation interp, float fill) {
        requirePositive(iz, iy, ix, "resampleAffine input");
        requirePositive(oz, oy, ox, "resampleAffine output");
        const Index inPlane = iy * ix;
#pragma omp parallel for collapse(2) schedule(dynamic, 4)
        for (Index z = 0; z < oz; ++z)
            for (Index y = 0; y < oy; ++y) {
                float* row = out + (z * oy + y) * ox;
                const double zd = static_cast<double>(z), yd = static_cast<double>(y);
                // p = A * (z, y, x) + b, evaluated incrementally along x
                double pz = A[0] * zd + A[1] * yd + b[0];
                double py = A[3] * zd + A[4] * yd + b[1];
                double px = A[6] * zd + A[7] * yd + b[2];
                for (Index x = 0; x < ox; ++x, pz += A[2], py += A[5], px += A[8]) {
                    const AxisTaps tz = axisTaps(pz, iz, interp);
                    const AxisTaps ty = tz.count ? axisTaps(py, iy, interp) : AxisTaps{};
                    const AxisTaps tx = ty.count ? axisTaps(px, ix, interp) : AxisTaps{};
                    if (!tx.count) {
                        row[x] = fill;
                        continue;
                    }
                    double v = 0.0;
                    for (int kz = 0; kz < tz.count; ++kz) {
                        const float* plane = in + tz.index[kz] * inPlane;
                        for (int ky = 0; ky < ty.count; ++ky) {
                            const float* line = plane + ty.index[ky] * ix;
                            double s = 0.0;
                            for (int kx = 0; kx < tx.count; ++kx) s += tx.weight[kx] * line[tx.index[kx]];
                            v += tz.weight[kz] * ty.weight[ky] * s;
                        }
                    }
                    row[x] = static_cast<float>(v);
                }
            }
    }

    // Geometry of a light-sheet acquisition, in words:
    //
    //   The camera sees planes (y, x) perpendicular to the detection axis z'.
    //   Between exposures the stage moves the sample by `stageStep` along the
    //   coverslip, which is inclined by `angle` to the image plane, so plane
    //   k is displaced by k * stageStep * cos(angle) along x (a shear) and by
    //   k * stageStep * sin(angle) along z' (the true axial spacing).
    //
    //        z' (detection)                 coverslip
    //        ^          plane 2 -----      /
    //        |       plane 1 -----        / angle
    //        |    plane 0 -----          /________ x
    //
    //   Deskew (no rotation) puts every plane back where the sample was: the
    //   output keeps the (z', y, x) axes, plane k shifted by the shear so the
    //   sample is straight, on a grid of (stageStep sin angle, dx, dx).
    //   Rotate-to-coverslip additionally rotates the sheared volume about y by
    //   -angle so the coverslip is the output x-z plane and z its normal, and
    //   resamples onto an isotropic dx grid over the bounding box of the
    //   rotated corners. y is untouched by either (square pixels assumed).
    ResampleGeometry deskewGeometry(Index iz, Index iy, Index ix, double dxUm, double dzUm, double angleDeg,
                                    double stageStepUm, bool rotateToCoverslip) {
        requirePositive(iz, iy, ix, "deskewGeometry");
        if (dxUm <= 0.0) throw std::invalid_argument("deskewGeometry: dx must be positive");
        const double step = stageStepUm > 0.0 ? stageStepUm : dzUm;
        if (step <= 0.0) throw std::invalid_argument("deskewGeometry: stage step (or dz) must be positive");
        const double theta = angleDeg * kPi / 180.0;
        const double ct = std::cos(theta), st = std::sin(theta);
        const double shear = step * ct / dxUm;              // pixels of x per plane
        // axial spacing of the sheared stack; a sheet parallel to the image
        // plane (angle 0) has no axial travel, keep the nominal dz then
        const double dzOut = std::abs(st) > 1e-9 ? step * std::abs(st) : (dzUm > 0.0 ? dzUm : step);

        ResampleGeometry g;
        if (!rotateToCoverslip) {
            const double totalShift = static_cast<double>(iz - 1) * shear;
            const double xmin = std::min(0.0, totalShift);
            g.oz = iz;
            g.oy = iy;
            g.ox = ix + static_cast<Index>(std::ceil(std::abs(totalShift)));
            // input plane = output plane, input x = output x + xmin - z * shear
            g.A = {1, 0, 0, 0, 1, 0, -shear, 0, 1};
            g.b = {0, 0, xmin};
            g.outVoxelUm = {dzOut, dxUm, dxUm};
            return g;
        }

        // Physical position (um) of input voxel (k, y, x) in the sheared
        // frame: X = (x + k shear) dx, Z' = k dzOut. Rotate about y by -angle:
        // Xr = X cos + Z' sin, Zr = -X sin + Z' cos.
        double xrMin = kInf, xrMax = -kInf, zrMin = kInf, zrMax = -kInf;
        for (int corner = 0; corner < 4; ++corner) {
            const double k = (corner & 1) ? static_cast<double>(iz - 1) : 0.0;
            const double x = (corner & 2) ? static_cast<double>(ix - 1) : 0.0;
            const double X = (x + k * shear) * dxUm, Zp = k * dzOut;
            const double xr = X * ct + Zp * st, zr = -X * st + Zp * ct;
            xrMin = std::min(xrMin, xr);
            xrMax = std::max(xrMax, xr);
            zrMin = std::min(zrMin, zr);
            zrMax = std::max(zrMax, zr);
        }
        // enough voxels that the far corner is still inside the last one
        g.ox = static_cast<Index>(std::ceil((xrMax - xrMin) / dxUm - 1e-9)) + 1;
        g.oz = static_cast<Index>(std::ceil((zrMax - zrMin) / dxUm - 1e-9)) + 1;
        g.oy = iy;
        // Output voxel (z, y, x) sits at Xr = xrMin + x dx, Zr = zrMin + z dx.
        // Undo the rotation, X = Xr cos - Zr sin and Z' = Xr sin + Zr cos,
        // then k = Z' / dzOut and input x = X / dx - k shear.
        const double a00 = dxUm * ct / dzOut, a02 = dxUm * st / dzOut;     // k from z, x
        const double b0 = (xrMin * st + zrMin * ct) / dzOut;
        g.A = {a00, 0, a02,
               0, 1, 0,
               -st - shear * a00, 0, ct - shear * a02};
        g.b = {b0, 0, (xrMin * ct - zrMin * st) / dxUm - shear * b0};
        g.outVoxelUm = {dxUm, dxUm, dxUm};
        return g;
    }

    ResampleGeometry resampleGeometry(Index iz, Index iy, Index ix, double dzUm, double dyUm, double dxUm,
                                      double tzUm, double tyUm, double txUm) {
        requirePositive(iz, iy, ix, "resampleGeometry");
        if (dzUm <= 0.0 || dyUm <= 0.0 || dxUm <= 0.0)
            throw std::invalid_argument("resampleGeometry: input voxel sizes must be positive");
        const double tz = tzUm > 0.0 ? tzUm : dzUm, ty = tyUm > 0.0 ? tyUm : dyUm, tx = txUm > 0.0 ? txUm : dxUm;
        // keep the physical field: (n - 1) * d um from the first to the last centre
        auto extent = [](Index n, double d, double t) {
            return n == 1 ? Index{1} : static_cast<Index>(std::floor(static_cast<double>(n - 1) * d / t + 1e-9)) + 1;
        };
        ResampleGeometry g;
        g.oz = extent(iz, dzUm, tz);
        g.oy = extent(iy, dyUm, ty);
        g.ox = extent(ix, dxUm, tx);
        g.A = {tz / dzUm, 0, 0, 0, ty / dyUm, 0, 0, 0, tx / dxUm};
        g.b = {0, 0, 0};
        g.outVoxelUm = {tz, ty, tx};
        return g;
    }

    void cropPad(const float* in, Index iz, Index iy, Index ix, Index z0, Index y0, Index x0, float* out,
                 Index oz, Index oy, Index ox, float fill) {
        requirePositive(iz, iy, ix, "cropPad input");
        requirePositive(oz, oy, ox, "cropPad output");
        // columns of the output that map inside the input
        const Index xa = std::clamp<Index>(-x0, 0, ox), xb = std::clamp<Index>(ix - x0, 0, ox);
#pragma omp parallel for collapse(2) schedule(static)
        for (Index z = 0; z < oz; ++z)
            for (Index y = 0; y < oy; ++y) {
                float* row = out + (z * oy + y) * ox;
                const Index sz = z + z0, sy = y + y0;
                if (sz < 0 || sz >= iz || sy < 0 || sy >= iy || xa >= xb) {
                    std::fill(row, row + ox, fill);
                    continue;
                }
                std::fill(row, row + xa, fill);
                std::copy_n(in + (sz * iy + sy) * ix + (xa + x0), xb - xa, row + xa);
                std::fill(row + xb, row + ox, fill);
            }
    }

    // --- intensities --------------------------------------------------------------

    std::pair<float, float> percentiles(const float* values, Index n, double loPercent, double hiPercent,
                                        Index maxSamples) {
        if (n <= 0) return {0.0f, 0.0f};
        loPercent = std::clamp(loPercent, 0.0, 100.0);
        hiPercent = std::clamp(hiPercent, loPercent, 100.0);
        maxSamples = std::max<Index>(maxSamples, 1);
        // fixed-stride subsample: bounded work, deterministic result
        const Index stride = std::max<Index>(1, (n + maxSamples - 1) / maxSamples);
        std::vector<float> samples;
        samples.reserve(static_cast<std::size_t>(n / stride + 1));
        for (Index i = 0; i < n; i += stride)
            if (!std::isnan(values[i])) samples.push_back(values[i]);
        if (samples.empty()) return {0.0f, 0.0f};
        auto rank = [&](double pct) {
            return static_cast<std::ptrdiff_t>(std::llround(pct / 100.0 * static_cast<double>(samples.size() - 1)));
        };
        const std::ptrdiff_t kLo = rank(loPercent), kHi = rank(hiPercent);
        std::nth_element(samples.begin(), samples.begin() + kLo, samples.end());
        const float lo = samples[static_cast<std::size_t>(kLo)];
        std::nth_element(samples.begin() + kLo, samples.begin() + kHi, samples.end());
        const float hi = samples[static_cast<std::size_t>(kHi)];
        if (hi > lo) return {lo, hi};
        // flat quantiles (mostly-zero data): fall back to the full range
        float mn = kInf, mx = -kInf;
        for (Index i = 0; i < n; ++i) {
            const float v = values[i];
            if (std::isnan(v)) continue;
            mn = v < mn ? v : mn;
            mx = v > mx ? v : mx;
        }
        if (!(mn <= mx)) return {0.0f, 0.0f};
        return {mn, mx};
    }

    void rescaleGamma(float* values, Index n, float lo, float hi, float gamma) {
        if (n <= 0) return;
        const double span = static_cast<double>(hi) - static_cast<double>(lo);
        const double invGamma = gamma > 0.0f ? 1.0 / static_cast<double>(gamma) : 1.0;
        if (!(span > 0.0)) {
#pragma omp parallel for schedule(static)
            for (Index i = 0; i < n; ++i) values[i] = std::isnan(values[i]) ? values[i] : (values[i] > hi ? 1.0f : 0.0f);
            return;
        }
#pragma omp parallel for schedule(static)
        for (Index i = 0; i < n; ++i) {
            const double t = std::clamp((static_cast<double>(values[i]) - lo) / span, 0.0, 1.0);
            values[i] = static_cast<float>(invGamma == 1.0 ? t : std::pow(t, invGamma));
        }
    }

    void downsampleBox(const float* in, Index iz, Index iy, Index ix, int fz, int fy, int fx, float* out) {
        requirePositive(iz, iy, ix, "downsampleBox");
        if (fz < 1 || fy < 1 || fx < 1) throw std::invalid_argument("downsampleBox: factors must be >= 1");
        detail::downsampleBoxMean<float>(in, {iz, iy, ix}, {fz, fy, fx}, out);
    }

    std::vector<double> histogram(const float* values, Index n, int bins, float lo, float hi) {
        bins = std::max(bins, 1);
        std::vector<double> counts(static_cast<std::size_t>(bins), 0.0);
        if (n <= 0 || !(hi > lo)) return counts;
        const double scale = bins / (static_cast<double>(hi) - static_cast<double>(lo));
        // per-thread histograms merged at the end: no atomics on the hot loop
#pragma omp parallel
        {
            std::vector<double> local(counts.size(), 0.0);
#pragma omp for schedule(static) nowait
            for (Index i = 0; i < n; ++i) {
                const float v = values[i];
                if (!(v >= lo) || v > hi) continue;   // NaN and out-of-range values are not counted
                int b = static_cast<int>((static_cast<double>(v) - lo) * scale);
                if (b >= bins) b = bins - 1;   // v == hi lands in the last bin
                local[static_cast<std::size_t>(b)] += 1.0;
            }
#pragma omp critical
            for (std::size_t b = 0; b < counts.size(); ++b) counts[b] += local[b];
        }
        return counts;
    }

    void equalizeFrames(float* stack, Index frames, Index planeSize, bool toMean) {
        if (frames <= 0 || planeSize <= 0) return;
        std::vector<double> sums(static_cast<std::size_t>(frames), 0.0);
#pragma omp parallel for schedule(static)
        for (Index f = 0; f < frames; ++f) {
            const float* p = stack + f * planeSize;
            double s = 0.0;
            for (Index i = 0; i < planeSize; ++i) s += p[i];
            sums[static_cast<std::size_t>(f)] = s;
        }
        double target = sums[0];
        if (toMean) {
            target = 0.0;
            for (double s : sums) target += s;
            target /= static_cast<double>(frames);
        }
#pragma omp parallel for schedule(static)
        for (Index f = 0; f < frames; ++f) {
            const double s = sums[static_cast<std::size_t>(f)];
            if (s == 0.0 || !std::isfinite(s)) continue;   // an empty frame cannot be scaled
            const float scale = static_cast<float>(target / s);
            float* p = stack + f * planeSize;
            for (Index i = 0; i < planeSize; ++i) p[i] *= scale;
        }
    }

    void flatField(float* values, Index planes, Index planeSize, const float* flat, const float* dark) {
        if (planes <= 0 || planeSize <= 0) return;
        // gain = flat - dark, normalized so the mean brightness is unchanged
        std::vector<float> gain(static_cast<std::size_t>(planeSize));
        double mean = 0.0;
        for (Index i = 0; i < planeSize; ++i) {
            const float g = flat[i] - (dark ? dark[i] : 0.0f);
            gain[static_cast<std::size_t>(i)] = g;
            mean += g;
        }
        mean /= static_cast<double>(planeSize);
        if (!(mean > 0.0)) throw std::invalid_argument("flatField: the flat image must be brighter than the dark image");
        // pixels with no gain (dead, vignetted to black) would explode; floor them
        const float floor_ = static_cast<float>(1e-6 * mean);
        for (float& g : gain) g = static_cast<float>(mean / std::max(g, floor_));
#pragma omp parallel for schedule(static)
        for (Index p = 0; p < planes; ++p) {
            float* v = values + p * planeSize;
            for (Index i = 0; i < planeSize; ++i)
                v[i] = (v[i] - (dark ? dark[i] : 0.0f)) * gain[static_cast<std::size_t>(i)];
        }
    }

} // namespace sirius
