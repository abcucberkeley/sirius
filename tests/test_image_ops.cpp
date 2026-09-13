// Generic array operations: reductions, affine resampling and the deskew /
// resample geometries, crop / pad, intensity helpers and box down-sampling.
// Every check compares against a direct evaluation of the definition.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

#include "sirius/image_ops.hpp"
#include "downsample.hpp"

using namespace sirius;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

    // value = a linear ramp over the five coordinates, so every reduction has
    // a closed form and every voxel is distinguishable
    struct Array {
        Extent5 e;
        std::vector<float> v;
        explicit Array(Extent5 extent) : e(extent), v(static_cast<std::size_t>(numel(extent))) {}
        Index at(Index c, Index t, Index z, Index y, Index x) const {
            return (((c * e[1] + t) * e[2] + z) * e[3] + y) * e[4] + x;
        }
        float& operator()(Index c, Index t, Index z, Index y, Index x) { return v[static_cast<std::size_t>(at(c, t, z, y, x))]; }
        float operator()(Index c, Index t, Index z, Index y, Index x) const { return v[static_cast<std::size_t>(at(c, t, z, y, x))]; }
    };

    Array ramp(Extent5 e) {
        Array a(e);
        for (Index c = 0; c < e[0]; ++c)
            for (Index t = 0; t < e[1]; ++t)
                for (Index z = 0; z < e[2]; ++z)
                    for (Index y = 0; y < e[3]; ++y)
                        for (Index x = 0; x < e[4]; ++x)
                            a(c, t, z, y, x) = static_cast<float>(1000 * c + 100 * t + 10 * z + 2 * y + 0.5 * x);
        return a;
    }

    // linear function of (z, y, x): trilinear interpolation reproduces it exactly
    float linearField(double z, double y, double x) { return static_cast<float>(3.0 + 0.5 * z - 0.25 * y + 0.125 * x); }

} // namespace

TEST_CASE("reduceAxes matches a direct evaluation for every op and axis subset", "[image_ops]") {
    const Extent5 e{2, 3, 4, 5, 6};
    const Array a = ramp(e);
    for (int mask = 0; mask < 32; ++mask) {
        std::array<bool, 5> reduce{};
        for (int i = 0; i < 5; ++i) reduce[static_cast<std::size_t>(i)] = (mask >> i) & 1;
        const Extent5 oe = reducedExtent(e, reduce);
        for (ReduceOp op : {ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min}) {
            std::vector<float> out(static_cast<std::size_t>(numel(oe)));
            reduceAxes(a.v.data(), e, reduce, op, out.data());
            // direct: for every output voxel, fold the input voxels that map to it
            Array expected(oe);
            std::vector<double> acc(expected.v.size(), op == ReduceOp::Max ? -1e300 : op == ReduceOp::Min ? 1e300
                                                                                                          : 0.0);
            double count = 1;
            for (int i = 0; i < 5; ++i)
                if (reduce[static_cast<std::size_t>(i)]) count *= static_cast<double>(e[static_cast<std::size_t>(i)]);
            for (Index c = 0; c < e[0]; ++c)
                for (Index t = 0; t < e[1]; ++t)
                    for (Index z = 0; z < e[2]; ++z)
                        for (Index y = 0; y < e[3]; ++y)
                            for (Index x = 0; x < e[4]; ++x) {
                                const Index o = expected.at(reduce[0] ? 0 : c, reduce[1] ? 0 : t, reduce[2] ? 0 : z,
                                                            reduce[3] ? 0 : y, reduce[4] ? 0 : x);
                                const double v = a(c, t, z, y, x);
                                double& s = acc[static_cast<std::size_t>(o)];
                                if (op == ReduceOp::Max) s = std::max(s, v);
                                else if (op == ReduceOp::Min) s = std::min(s, v);
                                else s += v;
                            }
            for (std::size_t i = 0; i < out.size(); ++i) {
                const double want = op == ReduceOp::Mean ? acc[i] / count : acc[i];
                INFO("mask " << mask << " op " << static_cast<int>(op) << " index " << i);
                CHECK_THAT(out[i], WithinRel(want, 1e-5));
            }
        }
    }
}

TEST_CASE("reduceAxes max and min ignore NaN", "[image_ops]") {
    const Extent5 e{1, 1, 1, 1, 4};
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> in{nan, 2.0f, -1.0f, nan};
    float out = 0.0f;
    reduceAxes(in.data(), e, {false, false, false, false, true}, ReduceOp::Max, &out);
    CHECK(out == 2.0f);
    reduceAxes(in.data(), e, {false, false, false, false, true}, ReduceOp::Min, &out);
    CHECK(out == -1.0f);
    std::vector<float> allNan{nan, nan};
    reduceAxes(allNan.data(), Extent5{1, 1, 1, 1, 2}, {false, false, false, false, true}, ReduceOp::Max, &out);
    CHECK(std::isnan(out));
}

TEST_CASE("resampleAffine identity and translation", "[image_ops]") {
    const Index iz = 3, iy = 4, ix = 5;
    std::vector<float> in(static_cast<std::size_t>(iz * iy * ix));
    for (Index z = 0; z < iz; ++z)
        for (Index y = 0; y < iy; ++y)
            for (Index x = 0; x < ix; ++x) in[static_cast<std::size_t>((z * iy + y) * ix + x)] = linearField(z, y, x);
    const std::array<double, 9> I{1, 0, 0, 0, 1, 0, 0, 0, 1};

    SECTION("identity reproduces the input for every interpolation") {
        for (Interpolation interp : {Interpolation::Nearest, Interpolation::Linear, Interpolation::Cubic}) {
            std::vector<float> out(in.size(), -1.0f);
            resampleAffine(in.data(), iz, iy, ix, I, {0, 0, 0}, out.data(), iz, iy, ix, interp);
            for (std::size_t i = 0; i < in.size(); ++i) CHECK_THAT(out[i], WithinAbs(in[i], 1e-5));
        }
    }
    SECTION("half-voxel shift interpolates the linear field exactly; outside reads fill") {
        std::vector<float> out(in.size());
        resampleAffine(in.data(), iz, iy, ix, I, {0.5, 0.25, 0.75}, out.data(), iz, iy, ix, Interpolation::Linear, -9.0f);
        for (Index z = 0; z < iz; ++z)
            for (Index y = 0; y < iy; ++y)
                for (Index x = 0; x < ix; ++x) {
                    const float got = out[static_cast<std::size_t>((z * iy + y) * ix + x)];
                    if (z == iz - 1 || y == iy - 1 || x == ix - 1) CHECK(got == -9.0f);
                    else CHECK_THAT(got, WithinAbs(linearField(z + 0.5, y + 0.25, x + 0.75), 1e-5));
                }
    }
    SECTION("cubic reproduces a linear field away from the edges") {
        std::vector<float> out(in.size());
        resampleAffine(in.data(), iz, iy, ix, I, {0.0, 0.5, 0.5}, out.data(), iz, iy, ix, Interpolation::Cubic, -9.0f);
        CHECK_THAT(out[static_cast<std::size_t>((1 * iy + 1) * ix + 2)], WithinAbs(linearField(1, 1.5, 2.5), 1e-4));
    }
    SECTION("nearest rounds to the closest voxel") {
        std::vector<float> out(in.size());
        resampleAffine(in.data(), iz, iy, ix, I, {0.0, 0.0, 0.4}, out.data(), iz, iy, ix, Interpolation::Nearest);
        CHECK(out[1] == in[1]);
        resampleAffine(in.data(), iz, iy, ix, I, {0.0, 0.0, 0.6}, out.data(), iz, iy, ix, Interpolation::Nearest);
        CHECK(out[1] == in[2]);
    }
    SECTION("a single-plane input is sampled at z = 0 only") {
        std::vector<float> out(static_cast<std::size_t>(iy * ix));
        resampleAffine(in.data(), 1, iy, ix, I, {0.3, 0, 0}, out.data(), 1, iy, ix, Interpolation::Linear, -9.0f);
        CHECK_THAT(out[0], WithinAbs(in[0], 1e-6));
        resampleAffine(in.data(), 1, iy, ix, I, {0.7, 0, 0}, out.data(), 1, iy, ix, Interpolation::Linear, -9.0f);
        CHECK(out[0] == -9.0f);
    }
}

TEST_CASE("deskewGeometry shears each plane by the stage travel along x", "[image_ops]") {
    const Index iz = 4, iy = 2, ix = 6;
    const double dx = 0.1, step = 0.4, angle = 31.8;
    const ResampleGeometry g = deskewGeometry(iz, iy, ix, dx, step, angle, step, false);
    const double shear = step * std::cos(angle * 3.14159265358979323846 / 180.0) / dx;   // pixels per plane
    CHECK(g.oz == iz);
    CHECK(g.oy == iy);
    CHECK(g.ox == ix + static_cast<Index>(std::ceil((iz - 1) * shear)));
    CHECK_THAT(g.outVoxelUm[0], WithinRel(step * std::sin(angle * 3.14159265358979323846 / 180.0), 1e-12));
    CHECK(g.outVoxelUm[2] == dx);

    // plane k of the input lands at output x = x_in + k * shear
    std::vector<float> in(static_cast<std::size_t>(iz * iy * ix), 0.0f);
    for (Index z = 0; z < iz; ++z) in[static_cast<std::size_t>((z * iy + 0) * ix + 0)] = 1.0f;   // a marker at x = 0 of each plane
    std::vector<float> out(static_cast<std::size_t>(g.oz * g.oy * g.ox));
    resampleAffine(in.data(), iz, iy, ix, g.A, g.b, out.data(), g.oz, g.oy, g.ox, Interpolation::Nearest);
    for (Index z = 0; z < iz; ++z) {
        const Index expectX = static_cast<Index>(std::lround(z * shear));
        const float* row = out.data() + (z * g.oy + 0) * g.ox;
        Index argmax = 0;
        for (Index x = 1; x < g.ox; ++x)
            if (row[x] > row[argmax]) argmax = x;
        CHECK(argmax == expectX);
    }

    SECTION("rotating to the coverslip yields an isotropic grid that covers every input voxel") {
        const ResampleGeometry r = deskewGeometry(iz, iy, ix, dx, step, angle, step, true);
        CHECK(r.outVoxelUm == std::array<double, 3>{dx, dx, dx});
        CHECK(r.oy == iy);
        // every input corner maps back inside the output grid: invert the
        // affine numerically by checking the output box brackets the corners
        double zmin = 1e300, zmax = -1e300, xmin = 1e300, xmax = -1e300;
        const double th = angle * 3.14159265358979323846 / 180.0, dzOut = step * std::sin(th);
        for (int corner = 0; corner < 4; ++corner) {
            const double k = (corner & 1) ? iz - 1 : 0, x = (corner & 2) ? ix - 1 : 0;
            const double X = (x + k * shear) * dx, Zp = k * dzOut;
            const double xr = X * std::cos(th) + Zp * std::sin(th), zr = -X * std::sin(th) + Zp * std::cos(th);
            zmin = std::min(zmin, zr);
            zmax = std::max(zmax, zr);
            xmin = std::min(xmin, xr);
            xmax = std::max(xmax, xr);
        }
        CHECK(r.ox == static_cast<Index>(std::ceil((xmax - xmin) / dx - 1e-9)) + 1);
        CHECK(r.oz == static_cast<Index>(std::ceil((zmax - zmin) / dx - 1e-9)) + 1);
        // the map is invertible in the (z, x) plane and sends every input
        // corner inside the output grid
        const double det = r.A[0] * r.A[8] - r.A[2] * r.A[6];
        REQUIRE(std::abs(det) > 1e-9);
        auto toOutput = [&](double pz, double px) {
            // solve A_zx * (oz, ox) = (pz, px) - b_zx
            const double rz = pz - r.b[0], rx = px - r.b[2];
            return std::array<double, 2>{(rz * r.A[8] - r.A[2] * rx) / det, (r.A[0] * rx - r.A[6] * rz) / det};
        };
        for (int corner = 0; corner < 4; ++corner) {
            const auto o = toOutput((corner & 1) ? iz - 1 : 0, (corner & 2) ? ix - 1 : 0);
            CHECK(o[0] >= -1e-6);
            CHECK(o[0] <= r.oz - 1 + 1e-6);
            CHECK(o[1] >= -1e-6);
            CHECK(o[1] <= r.ox - 1 + 1e-6);
        }
        // a marker at input (1, 0, 2) lands where the inverse map predicts
        std::vector<float> marker(static_cast<std::size_t>(iz * iy * ix), 0.0f);
        marker[static_cast<std::size_t>((1 * iy + 0) * ix + 2)] = 1.0f;
        std::vector<float> rotated(static_cast<std::size_t>(r.oz * r.oy * r.ox));
        resampleAffine(marker.data(), iz, iy, ix, r.A, r.b, rotated.data(), r.oz, r.oy, r.ox, Interpolation::Linear);
        const auto o = toOutput(1.0, 2.0);
        Index best = 0;
        for (Index i = 1; i < r.oz * r.oy * r.ox; ++i)
            if (rotated[static_cast<std::size_t>(i)] > rotated[static_cast<std::size_t>(best)]) best = i;
        CHECK(rotated[static_cast<std::size_t>(best)] > 0.0f);
        const Index bz = best / (r.oy * r.ox), by = (best / r.ox) % r.oy, bx = best % r.ox;
        CHECK(by == 0);
        CHECK(std::abs(static_cast<double>(bz) - o[0]) <= 1.0);
        CHECK(std::abs(static_cast<double>(bx) - o[1]) <= 1.0);
    }
    SECTION("a missing stage step falls back to dz") {
        const ResampleGeometry d = deskewGeometry(iz, iy, ix, dx, 0.3, angle, 0.0, false);
        CHECK_THAT(d.outVoxelUm[0], WithinRel(0.3 * std::sin(angle * 3.14159265358979323846 / 180.0), 1e-12));
    }
}

TEST_CASE("resampleGeometry maps onto a grid with the target voxel size", "[image_ops]") {
    const ResampleGeometry g = resampleGeometry(5, 10, 20, 0.4, 0.1, 0.1, 0.1, 0.1, 0.0);
    CHECK(g.oz == 17);   // (5 - 1) * 0.4 / 0.1 + 1
    CHECK(g.oy == 10);
    CHECK(g.ox == 20);
    CHECK(g.outVoxelUm == std::array<double, 3>{0.1, 0.1, 0.1});
    CHECK_THAT(g.A[0], WithinRel(0.25, 1e-12));
    CHECK(g.A[4] == 1.0);
    CHECK(g.A[8] == 1.0);
    CHECK(resampleGeometry(1, 4, 4, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1).oz == 1);
}

TEST_CASE("cropPad copies the overlap and fills the rest", "[image_ops]") {
    const Index iz = 2, iy = 3, ix = 4;
    std::vector<float> in(static_cast<std::size_t>(iz * iy * ix));
    std::iota(in.begin(), in.end(), 0.0f);
    const Index oz = 3, oy = 4, ox = 5;
    std::vector<float> out(static_cast<std::size_t>(oz * oy * ox));
    cropPad(in.data(), iz, iy, ix, -1, 1, -2, out.data(), oz, oy, ox, -1.0f);
    for (Index z = 0; z < oz; ++z)
        for (Index y = 0; y < oy; ++y)
            for (Index x = 0; x < ox; ++x) {
                const Index sz = z - 1, sy = y + 1, sx = x - 2;
                const float got = out[static_cast<std::size_t>((z * oy + y) * ox + x)];
                if (sz < 0 || sz >= iz || sy < 0 || sy >= iy || sx < 0 || sx >= ix) CHECK(got == -1.0f);
                else CHECK(got == in[static_cast<std::size_t>((sz * iy + sy) * ix + sx)]);
            }
}

TEST_CASE("percentiles, rescaleGamma and histogram", "[image_ops]") {
    std::vector<float> v(1000);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i);
    v[0] = -1e9f;
    v[999] = 1e9f;
    const auto [lo, hi] = percentiles(v.data(), static_cast<Index>(v.size()), 1.0, 99.0);
    CHECK(lo > 0.0f);
    CHECK(lo < 20.0f);
    CHECK(hi > 980.0f);
    CHECK(hi < 999.0f);

    SECTION("flat data falls back to the full range") {
        std::vector<float> flat(50, 4.0f);
        flat[10] = 9.0f;
        const auto [a, b] = percentiles(flat.data(), 50, 10.0, 90.0);
        CHECK(a == 4.0f);
        CHECK(b == 9.0f);
    }
    SECTION("rescaleGamma clamps and applies 1/gamma") {
        std::vector<float> w{-1.0f, 0.0f, 0.25f, 1.0f, 3.0f};
        rescaleGamma(w.data(), 5, 0.0f, 1.0f, 2.0f);
        CHECK(w[0] == 0.0f);
        CHECK(w[1] == 0.0f);
        CHECK_THAT(w[2], WithinAbs(0.5f, 1e-6));
        CHECK(w[3] == 1.0f);
        CHECK(w[4] == 1.0f);
    }
    SECTION("histogram counts values in [lo, hi] into equal bins") {
        std::vector<float> h{0.0f, 0.1f, 0.5f, 1.0f, 2.0f, std::numeric_limits<float>::quiet_NaN()};
        const std::vector<double> bins = histogram(h.data(), 6, 4, 0.0f, 1.0f);
        REQUIRE(bins.size() == 4);
        CHECK(bins[0] == 2.0);   // 0.0, 0.1
        CHECK(bins[1] == 0.0);
        CHECK(bins[2] == 1.0);   // 0.5
        CHECK(bins[3] == 1.0);   // 1.0 lands in the last bin; 2.0 and NaN are dropped
    }
}

TEST_CASE("downsampleBox averages full and partial boxes", "[image_ops]") {
    const Index iz = 1, iy = 3, ix = 5;
    std::vector<float> in(static_cast<std::size_t>(iy * ix));
    std::iota(in.begin(), in.end(), 1.0f);   // 1..15
    const Index oy = downsampledExtent(iy, 2), ox = downsampledExtent(ix, 2);
    CHECK(oy == 2);
    CHECK(ox == 3);
    std::vector<float> out(static_cast<std::size_t>(oy * ox));
    downsampleBox(in.data(), iz, iy, ix, 1, 2, 2, out.data());
    CHECK_THAT(out[0], WithinAbs((1 + 2 + 6 + 7) / 4.0f, 1e-6));   // full box
    CHECK_THAT(out[2], WithinAbs((5 + 10) / 2.0f, 1e-6));          // partial in x
    CHECK_THAT(out[3], WithinAbs((11 + 12) / 2.0f, 1e-6));         // partial in y
    CHECK_THAT(out[5], WithinAbs(15.0f, 1e-6));                    // corner
}

namespace {
    // The definition the TIFF and zarr writers used before sharing one
    // implementation: walk the input in C order, fold every voxel into its
    // box, then round the mean for integers and convert it for floats.
    template <typename T>
    std::vector<T> referenceBoxMean(const std::vector<T>& in, const std::vector<Index>& shape,
                                    const std::vector<int>& factors) {
        const std::size_t r = shape.size();
        std::vector<Index> outShape(r), outStride(r, 1);
        for (std::size_t i = 0; i < r; ++i) outShape[i] = (shape[i] + factors[i] - 1) / factors[i];
        for (std::size_t k = r - 1; k-- > 0;) outStride[k] = outStride[k + 1] * outShape[k + 1];
        const std::size_t m = static_cast<std::size_t>(detail::elementCount(outShape));
        std::vector<double> acc(m, 0.0);
        std::vector<Index> cnt(m, 0), idx(r, 0);
        for (const T v : in) {
            Index o = 0;
            for (std::size_t k = 0; k < r; ++k) o += (idx[k] / factors[k]) * outStride[k];
            acc[static_cast<std::size_t>(o)] += static_cast<double>(v);
            ++cnt[static_cast<std::size_t>(o)];
            for (std::size_t k = r; k-- > 0;) {
                if (++idx[k] < shape[k]) break;
                idx[k] = 0;
            }
        }
        std::vector<T> out(m);
        for (std::size_t i = 0; i < m; ++i) {
            const double mean = acc[i] / static_cast<double>(cnt[i]);
            if constexpr (std::is_integral_v<T>) out[i] = static_cast<T>(std::llround(mean));
            else out[i] = static_cast<T>(mean);
        }
        return out;
    }

    template <typename T>
    std::vector<T> sharedBoxMean(const std::vector<T>& in, const std::vector<Index>& shape,
                                 const std::vector<int>& factors) {
        const std::vector<Index> outShape = detail::downsampledShape(shape, factors);
        std::vector<T> out(static_cast<std::size_t>(detail::elementCount(outShape)));
        detail::downsampleBoxMean<T>(in.data(), shape, factors, out.data());
        return out;
    }
} // namespace

TEST_CASE("shared box-mean matches the writers' previous behaviour", "[image_ops][downsample]") {
    SECTION("uint16 rounds the mean to nearest, halves away from zero") {
        const std::vector<Index> shape{3, 5};
        std::vector<std::uint16_t> in{1, 2, 7, 9, 100,   //
                                      2, 2, 8, 9, 101,   //
                                      5, 6, 3, 4, 65535};
        const std::vector<std::uint16_t> out = sharedBoxMean(in, shape, {2, 2});
        REQUIRE(out.size() == 6);
        CHECK(out[0] == 2);       // (1 + 2 + 2 + 2) / 4 = 1.75
        CHECK(out[1] == 8);       // (7 + 9 + 8 + 9) / 4 = 8.25
        CHECK(out[2] == 101);     // (100 + 101) / 2 = 100.5, partial in x
        CHECK(out[3] == 6);       // (5 + 6) / 2 = 5.5, partial in y
        CHECK(out[4] == 4);       // (3 + 4) / 2 = 3.5
        CHECK(out[5] == 65535);   // the corner alone, no overflow
        CHECK(out == referenceBoxMean(in, shape, {2, 2}));
    }
    SECTION("float keeps the mean, odd extents leave partial boxes at the far edges") {
        const std::vector<Index> shape{5, 7};
        std::vector<float> in(35);
        for (std::size_t i = 0; i < in.size(); ++i) in[i] = 0.1f * static_cast<float>(i) - 1.3f;
        for (int f : {2, 3, 4}) {
            const std::vector<float> out = sharedBoxMean(in, shape, {f, f});
            const std::vector<Index> outShape = detail::downsampledShape(shape, {f, f});
            CHECK(outShape[0] == (5 + f - 1) / f);
            CHECK(outShape[1] == (7 + f - 1) / f);
            CHECK(out == referenceBoxMean(in, shape, {f, f}));   // bit-exact, same summation order
        }
        // the public 3-D entry point is the same arithmetic
        std::vector<float> pub(static_cast<std::size_t>(downsampledExtent(5, 2) * downsampledExtent(7, 3)));
        downsampleBox(in.data(), 1, 5, 7, 1, 2, 3, pub.data());
        CHECK(pub == referenceBoxMean(in, shape, {2, 3}));
    }
    SECTION("axes that do not shrink may sit between those that do, as in an OME-NGFF (c, z, y, x) level") {
        const std::vector<Index> shape{2, 3, 5, 4};
        std::vector<std::int32_t> in(120);
        for (std::size_t i = 0; i < in.size(); ++i) in[i] = static_cast<std::int32_t>((i * 37) % 23) - 11;
        for (const std::vector<int>& f : {std::vector<int>{1, 2, 2, 2}, std::vector<int>{1, 1, 2, 2}, std::vector<int>{2, 1, 3, 1}})
            CHECK(sharedBoxMean(in, shape, f) == referenceBoxMean(in, shape, f));
        // rank 1 and a rank-3 stack of double planes
        const std::vector<double> line{1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0};
        CHECK(sharedBoxMean(line, {7}, {3}) == std::vector<double>{7.0 / 3.0, 56.0 / 3.0, 64.0});
        std::vector<double> stack(3 * 4 * 5);
        for (std::size_t i = 0; i < stack.size(); ++i) stack[i] = std::sqrt(static_cast<double>(i));
        CHECK(sharedBoxMean(stack, {3, 4, 5}, {2, 2, 2}) == referenceBoxMean(stack, {3, 4, 5}, {2, 2, 2}));
    }
}

TEST_CASE("equalizeFrames and flatField", "[image_ops]") {
    SECTION("frames are scaled to the first frame's total or to the mean") {
        std::vector<float> s{1, 1, 2, 2, 0, 0, 4, 4};   // four frames of two pixels: sums 2, 4, 0, 8
        std::vector<float> first = s;
        equalizeFrames(first.data(), 4, 2, false);
        CHECK_THAT(first[2] + first[3], WithinAbs(2.0f, 1e-6));
        CHECK(first[4] == 0.0f);   // an empty frame stays empty
        CHECK_THAT(first[6] + first[7], WithinAbs(2.0f, 1e-6));
        std::vector<float> mean = s;
        equalizeFrames(mean.data(), 4, 2, true);
        CHECK_THAT(mean[0] + mean[1], WithinAbs(3.5f, 1e-6));   // (2 + 4 + 0 + 8) / 4
    }
    SECTION("flat-field divides by the normalized gain and subtracts the dark frame") {
        const std::vector<float> flat{2, 4, 6, 8}, dark{1, 1, 1, 1};   // gain 1, 3, 5, 7, mean 4
        std::vector<float> v{3, 7, 11, 15};                             // (v - dark) / gain = 2 everywhere
        flatField(v.data(), 1, 4, flat.data(), dark.data());
        for (float x : v) CHECK_THAT(x, WithinAbs(8.0f, 1e-5));       // 2 * mean gain
        std::vector<float> w{1, 3, 5, 7};
        flatField(w.data(), 1, 4, flat.data(), nullptr);              // gain 2, 4, 6, 8, mean 5
        CHECK_THAT(w[0], WithinAbs(2.5f, 1e-5));
        CHECK_THAT(w[3], WithinAbs(7.0f * 5.0f / 8.0f, 1e-5));
    }
}
