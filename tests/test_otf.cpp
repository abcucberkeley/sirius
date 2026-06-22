#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <complex>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

#include "sirius/otf.hpp"
#include "sirius/tiff_io.hpp"

using namespace sirius;
using Catch::Matchers::WithinAbs;
using Cplx = std::complex<double>;

namespace {
    // Build a (nkr, nzotf) radial OTF plane from f(ir, iz) -> complex.
    template <typename F>
    Eigen::Tensor<Cplx, 2, Eigen::RowMajor> makeRadial(int nkr, int nzotf, F&& f) {
        Eigen::Tensor<Cplx, 2, Eigen::RowMajor> m(nkr, nzotf);
        for (int r = 0; r < nkr; ++r)
            for (int z = 0; z < nzotf; ++z)
                m(r, z) = f(r, z);
        return m;
    }

    // RAII temp file: removed on scope exit.
    struct TempFile {
        std::string path;
        explicit TempFile(std::string p) : path(std::move(p)) {}
        ~TempFile() { std::error_code ec; std::filesystem::remove(path, ec); }
    };

    std::string tempTiffPath(const char* tag) {
        return (std::filesystem::temp_directory_path() /
                ("sirius_otf_" + std::string(tag) + ".tif")).string();
    }
}

// --- resampleOTF -----------------------------------------------------------

TEST_CASE("resampleOTF output has (nz, ny, nx) shape", "[otf]") {
    auto radial = makeRadial(4, 4, [](int, int) { return Cplx(1.0, 0.0); });
    auto out = resampleOTF(radial, /*nx*/8, /*ny*/6, /*nz*/5, 1.0, 1.0, 1.0, 1.0);
    REQUIRE(out.dimension(0) == 5);
    REQUIRE(out.dimension(1) == 6);
    REQUIRE(out.dimension(2) == 8);
}

TEST_CASE("resampleOTF: DC is in-band, far corner is zero", "[otf]") {
    // constant OTF over the whole (kr, kz) grid
    auto radial = makeRadial(4, 4, [](int, int) { return Cplx(2.0, -1.0); });
    auto out = resampleOTF(radial, 16, 16, 1, /*dkx*/1.0, /*dky*/1.0, /*dkrotf*/1.0, /*kzscale*/0.0);

    // DC voxel: kx=ky=kz=0 -> krindex=0, in band -> the constant value
    CHECK_THAT(out(0, 0, 0).real(), WithinAbs(2.0, 1e-12));
    CHECK_THAT(out(0, 0, 0).imag(), WithinAbs(-1.0, 1e-12));

    // ix=8 -> kx=8 -> krindex=8 >= nkr=4 -> out of band -> zero
    CHECK_THAT(out(0, 0, 8).real(), WithinAbs(0.0, 1e-12));
    CHECK_THAT(out(0, 0, 8).imag(), WithinAbs(0.0, 1e-12));
}

TEST_CASE("resampleOTF interpolates radially (2D case)", "[otf]") {
    // value depends only on the radial index: otf(ir, *) = ir
    auto radial = makeRadial(8, 4, [](int ir, int) { return Cplx(ir, 0.0); });
    // nz=1 collapses kz; rxscale = dkx/dkrotf = 0.5
    auto out = resampleOTF(radial, 16, 16, 1, /*dkx*/0.5, /*dky*/0.5, /*dkrotf*/1.0, /*kzscale*/0.0);

    // ix=5 -> kx=5 -> krindex = 5*0.5 = 2.5 -> lerp(otf[2], otf[3], 0.5) = 2.5
    CHECK_THAT(out(0, 0, 5).real(), WithinAbs(2.5, 1e-12));
    CHECK_THAT(out(0, 0, 5).imag(), WithinAbs(0.0, 1e-12));
}

TEST_CASE("resampleOTF interpolates kz circularly across the wrap", "[otf]") {
    const int nzotf = 4;
    // along kz at ir=0: otf(0, iz) = iz ; other rings zero
    auto radial = makeRadial(4, nzotf, [](int ir, int iz) {
        return ir == 0 ? Cplx(iz, 0.0) : Cplx(0.0, 0.0);
    });
    // kx=ky=0 -> ir=0. iz=nz-1 -> kz=-1 ; kzscale=0.5 -> kzindex=-0.5+4=3.5
    // -> iz0=3, iz1=0 (wrap), az=0.5 -> 0.5*otf(0,3) + 0.5*otf(0,0) = 1.5
    auto out = resampleOTF(radial, 4, 4, 4, 1.0, 1.0, 1.0, /*kzscale*/0.5);
    CHECK_THAT(out(3, 0, 0).real(), WithinAbs(1.5, 1e-12));
    CHECK_THAT(out(3, 0, 0).imag(), WithinAbs(0.0, 1e-12));
}

TEST_CASE("resampleOTF: full bilinear in kr and kz", "[otf]") {
    // f(ir, iz) = (ir, iz). For an in-band, non-wrapping voxel the real part
    // interpolates to krindex and the imag part to kzindex.
    auto radial = makeRadial(8, 4, [](int ir, int iz) { return Cplx(ir, iz); });
    auto out = resampleOTF(radial, 16, 16, 8, /*dkx*/0.5, /*dky*/0.5, /*dkrotf*/1.0, /*kzscale*/1.5);

    // ix=3 -> kx=3 -> kxt=1.5, ky=0 -> krindex=1.5
    // iz=1 -> kz=1 -> kzindex=1.5 (iz0=1, iz1=2, no wrap)
    CHECK_THAT(out(1, 0, 3).real(), WithinAbs(1.5, 1e-12));   // krindex
    CHECK_THAT(out(1, 0, 3).imag(), WithinAbs(1.5, 1e-12));   // kzindex
}

TEST_CASE("OTFRadiallyAveraged::plane extracts one order", "[otf]") {
    // (norders=2, nkr=2, nzotf=3), value encodes its (order, ir, iz)
    Eigen::Tensor<Cplx, 3, Eigen::RowMajor> d(2, 2, 3);
    for (int o = 0; o < 2; ++o)
        for (int r = 0; r < 2; ++r)
            for (int z = 0; z < 3; ++z)
                d(o, r, z) = Cplx(o * 100 + r * 10 + z, 0.0);

    OTFRadiallyAveraged otf(d, 1.0, 1.0);

    auto p = otf.plane(1);
    REQUIRE(p.dimension(0) == 2);   // nkr
    REQUIRE(p.dimension(1) == 3);   // nzotf
    CHECK(p(0, 0) == Cplx(100, 0));
    CHECK(p(1, 2) == Cplx(112, 0));

    // the extracted plane feeds resampleOTF directly
    auto out = resampleOTF(p, 4, 4, 1, 1.0, 1.0, 1.0, 0.0);
    REQUIRE(out.dimension(0) == 1);

    REQUIRE_THROWS_AS(otf.plane(2),  std::out_of_range);
    REQUIRE_THROWS_AS(otf.plane(-1), std::out_of_range);
}

// --- loadOTF ---------------------------------------------------------------

TEST_CASE("loadOTF de-interleaves real/imag columns", "[otf]") {
    // raw stack (norders=1, nkr=2, 2*nzotf=4): cols = [re0, im0, re1, im1]
    ImageStack<float> raw(1, 2, 4);
    raw(0, 0, 0) = 1; raw(0, 0, 1) = 2; raw(0, 0, 2) = 3; raw(0, 0, 3) = 4;
    raw(0, 1, 0) = 5; raw(0, 1, 1) = 6; raw(0, 1, 2) = 7; raw(0, 1, 3) = 8;

    TempFile tf(tempTiffPath("load"));
    writeTiffStack<float>(tf.path, raw);

    auto otf = loadOTF(tf.path, /*dkrotf*/0.25, /*dkzotf*/0.5);
    const auto& d = otf.data();

    REQUIRE(d.dimension(0) == 1);
    REQUIRE(d.dimension(1) == 2);
    REQUIRE(d.dimension(2) == 2);

    CHECK(d(0, 0, 0) == Cplx(1, 2));
    CHECK(d(0, 0, 1) == Cplx(3, 4));
    CHECK(d(0, 1, 0) == Cplx(5, 6));
    CHECK(d(0, 1, 1) == Cplx(7, 8));

    CHECK(otf.dkrotf() == 0.25);
    CHECK(otf.dkzotf() == 0.5);
}

TEST_CASE("loadOTF rejects an odd last dimension", "[otf]") {
    // 3 columns cannot pair into real/imag
    ImageStack<float> raw(1, 2, 3);
    raw.setZero();

    TempFile tf(tempTiffPath("odd"));
    writeTiffStack<float>(tf.path, raw);

    REQUIRE_THROWS_AS(loadOTF(tf.path, 1.0, 1.0), std::runtime_error);
}
