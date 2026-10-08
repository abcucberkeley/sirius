// loadOTF: the radially averaged OTF table read from a TIFF.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <stdexcept>
#include <utility>
#include <vector>

#include "sirius/otf_io.hpp"
#include "sirius/tiff_io.hpp"

#include "temp_path.hpp"

using namespace sirius;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using Cplx = std::complex<double>;

TEST_CASE("loadOTF de-interleaves real/imag columns", "[otf]") {
    // raw stack (norders=1, nkr=2, 2*nzotf=4): cols = [re0, im0, re1, im1]
    ImageStack<float> raw(1, 2, 4);
    raw(0, 0, 0) = 1;
    raw(0, 0, 1) = 2;
    raw(0, 0, 2) = 3;
    raw(0, 0, 3) = 4;
    raw(0, 1, 0) = 5;
    raw(0, 1, 1) = 6;
    raw(0, 1, 2) = 7;
    raw(0, 1, 3) = 8;

    test::TempFile tf("otf_load", ".tif");
    writeTiffStack<float>(tf.str, raw);

    auto otf = loadOTF(tf.str, /*dkrotf*/ 0.25, /*dkzotf*/ 0.5);
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

    test::TempFile tf("otf_odd", ".tif");
    writeTiffStack<float>(tf.str, raw);

    REQUIRE_THROWS_AS(loadOTF(tf.str, 1.0, 1.0), std::runtime_error);
}

TEST_CASE("loadOTF reads cudasirecon's radially averaged OTF TIFF as cudasirecon reads it", "[otf][data]") {
    // tests/data/otf.tif is cudasirecon's test OTF, written by its makeotf
    // (radialft.cpp: a CImg of width nz*2, height nx/2+1, depth norders) as
    // 3 pages of 129 rows x 130 float32 columns. cudasirecon reads it back
    // (determine_otf_dimensions, otfRA=1 on a 3D stack) as nzotf = width / 2
    // = 65 complex kz samples, (re, im) interleaved along the columns,
    // nxotf = height = 129 radial samples, one page per order, with
    // dkzotf = 1 / (zresPSF * nzotf) and dkrotf = 1 / (xyres * (nxotf - 1) * 2);
    // its kernel indexes otf[ir * nzotf + iz] with kz in FFT order (negative
    // kz wraps to the top). Its log for this file and config.txt reads
    // "nzotf=65, dkzotf=0.123077, nxotf=129, nyotf=1, dkrotf=0.048828".
    const std::string file = (std::filesystem::path(SIRIUS_TEST_DATA_DIR) / "otf.tif").string();
    SIMParameters p;   // config.txt: xyres=0.08 zres=0.125 zresPSF=0.125
    p.dx = p.dy = 0.08;
    p.dz = 0.125;
    p.dz_psf = 0.125;
    const OTFRadiallyAveraged otf = loadOTF(file, p);
    const auto& d = otf.data();
    REQUIRE(d.dimension(0) == 3);     // orders 0, 1, 2
    REQUIRE(d.dimension(1) == 129);   // kr
    REQUIRE(d.dimension(2) == 65);    // kz, complex
    CHECK_THAT(otf.dkrotf(), WithinAbs(0.048828, 1e-6));
    CHECK_THAT(otf.dkzotf(), WithinAbs(0.123077, 1e-6));
    CHECK_THAT(otf.dkrotf(), WithinRel(1.0 / (0.08 * 128.0 * 2.0), 1e-12));
    CHECK_THAT(otf.dkzotf(), WithinRel(1.0 / (0.125 * 65.0), 1e-12));

    // The complex samples are the file's (re, im) column pairs: page = order, row = kr.
    const auto raw = readTiffStack<float>(file);
    REQUIRE(raw.dimension(0) == 3);
    REQUIRE(raw.dimension(1) == 129);
    REQUIRE(raw.dimension(2) == 130);
    for (const auto& [o, ir, iz] : std::vector<std::array<Eigen::Index, 3>>{{0, 0, 0}, {0, 0, 1}, {1, 0, 9}, {2, 5, 7}, {2, 128, 64}}) {
        INFO("order " << o << " kr " << ir << " kz " << iz);
        CHECK(d(o, ir, iz) == Cplx(raw(o, ir, 2 * iz), raw(o, ir, 2 * iz + 1)));
    }

    auto peak = [&](Eigen::Index order) {
        std::array<Eigen::Index, 2> at{0, 0};
        double best = -1.0;
        for (Eigen::Index ir = 0; ir < d.dimension(1); ++ir)
            for (Eigen::Index iz = 0; iz < d.dimension(2); ++iz) {
                const double m = std::abs(d(order, ir, iz));
                if (m > best) {
                    best = m;
                    at = {ir, iz};
                }
            }
        return std::pair{best, at};
    };

    // Order 0 is the widefield OTF, normalised: its peak is 1 at kr = kz = 0.
    const auto [peak0, at0] = peak(0);
    CHECK(d(0, 0, 0) == Cplx(1.0, 0.0));
    CHECK_THAT(peak0, WithinAbs(1.0, 1e-6));
    CHECK(at0 == std::array<Eigen::Index, 2>{0, 0});
    // kz is in FFT order: sample 64 is kz = -1 and mirrors sample 1 in every order.
    CHECK_THAT(std::abs(d(0, 0, 64)), WithinRel(std::abs(d(0, 0, 1)), 1e-4));
    CHECK_THAT(std::abs(d(1, 0, 64)), WithinRel(std::abs(d(1, 0, 1)), 1e-4));
    CHECK_THAT(std::abs(d(2, 0, 64)), WithinRel(std::abs(d(2, 0, 1)), 1e-4));
    CHECK_THAT(std::abs(d(0, 0, 1)), WithinAbs(0.1219, 1e-3));
    // Its support is a fraction of the table, which is what makes the steps
    // matter: at kz = 0 the magnitude falls under cudasirecon's default
    // otfcutoff (0.006) at kr index 69 (3.37 cycles/um at this dkr) and along
    // kz at index 5 (0.62 cycles/um). Read with the 0.03125 step of a 0.125 um
    // pixel, the same table would end at 2.2 cycles/um, short of the side
    // bands of a 0.2035 um pattern (order 2 at 4.9 cycles/um).
    CHECK(std::abs(d(0, 68, 0)) > 0.006);
    CHECK(std::abs(d(0, 69, 0)) < 0.006);
    CHECK(std::abs(d(0, 0, 4)) > 0.006);
    CHECK(std::abs(d(0, 0, 5)) < 0.006);
    CHECK_THAT(otf.dkrotf() * 69.0, WithinAbs(3.37, 0.01));

    // Order 1 is the axial side band of three-beam SIM: it peaks away from kz = 0.
    const auto [peak1, at1] = peak(1);
    CHECK_THAT(peak1, WithinAbs(0.164666, 1e-5));
    CHECK(at1 == std::array<Eigen::Index, 2>{0, 9});
    CHECK(std::abs(d(1, 0, 0)) < 0.02);

    // Order 2 peaks at DC, at 0.225 of the widefield peak.
    const auto [peak2, at2] = peak(2);
    CHECK_THAT(peak2, WithinAbs(0.22525, 1e-5));
    CHECK(at2 == std::array<Eigen::Index, 2>{0, 0});

    // The explicit-step overload decodes the same table.
    const OTFRadiallyAveraged explicitSteps = loadOTF(file, 0.048828, 0.123077);
    REQUIRE(explicitSteps.data().dimension(2) == 65);
    CHECK(explicitSteps.data()(2, 5, 7) == d(2, 5, 7));
    CHECK(explicitSteps.dkrotf() == 0.048828);
}
