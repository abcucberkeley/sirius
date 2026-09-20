// idealOTF: the theoretical widefield OTF in the radially averaged layout.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>
#include <stdexcept>

#include "sirius/constants.hpp"
#include "sirius/otf_ideal.hpp"

using namespace sirius;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using Cplx = std::complex<double>;

namespace {
    SIMParameters lowNaParams() {
        SIMParameters p;
        p.na = 0.3;
        p.nimm = 1.33;
        p.wavelength_nm = 520.0;
        p.nphases = 5;
        p.norders = 3;
        return p;
    }
} // namespace

TEST_CASE("idealOTF 2D matches the paraxial pupil autocorrelation", "[otf][ideal]") {
    // At low NA the sine-condition apodization is ~1, so the in-focus OTF is
    // the classic (2/pi)(acos(r) - r sqrt(1 - r^2)) with r = kr / (2 NA / lambda).
    const SIMParameters p = lowNaParams();
    const OTFRadiallyAveraged otf = idealOTF(p, /*threeD=*/false);
    const auto& d = otf.data();
    REQUIRE(d.dimension(0) == 3);
    REQUIRE(d.dimension(2) == 1);
    const Eigen::Index nkr = d.dimension(1);
    REQUIRE(nkr > 64);

    const double cutoff = 2.0 * p.na / (p.wavelength_nm * 1e-3);
    CHECK_THAT(d(0, 0, 0).real(), WithinAbs(1.0, 1e-9));
    CHECK_THAT(d(0, 0, 0).imag(), WithinAbs(0.0, 1e-9));
    for (Eigen::Index ir = 0; ir < nkr; ++ir) {
        const double r = ir * otf.dkrotf() / cutoff;
        const double expected = r < 1.0 ? (2.0 / kPi) * (std::acos(r) - r * std::sqrt(1.0 - r * r)) : 0.0;
        INFO("ir " << ir << " r " << r);
        CHECK_THAT(d(0, ir, 0).real(), WithinAbs(expected, 0.02));
        CHECK_THAT(std::abs(d(0, ir, 0).imag()), WithinAbs(0.0, 1e-6));
        // every order is the widefield OTF for a 2D table
        CHECK(d(1, ir, 0) == d(0, ir, 0));
        CHECK(d(2, ir, 0) == d(0, ir, 0));
    }
    // monotonically decreasing inside the support
    for (Eigen::Index ir = 1; ir * otf.dkrotf() < cutoff; ++ir)
        CHECK(d(0, ir, 0).real() <= d(0, ir - 1, 0).real() + 1e-9);
}

TEST_CASE("idealOTF 3D has a missing cone, a symmetric kz axis and a shifted order 1", "[otf][ideal]") {
    SIMParameters p = lowNaParams();
    p.na = 1.2;
    p.nimm = 1.515;
    p.dz_psf = 0.1;
    p.linespacing_um = 0.2;
    IdealOtfOptions opts;
    opts.lateralSamples = 128;
    opts.axialSamples = 32;
    const OTFRadiallyAveraged otf = idealOTF(p, /*threeD=*/true, opts);
    const auto& d = otf.data();
    REQUIRE(d.dimension(0) == 3);
    REQUIRE(d.dimension(1) == 65);
    REQUIRE(d.dimension(2) == 32);
    CHECK_THAT(otf.dkzotf(), WithinRel(1.0 / (32 * 0.1), 1e-12));

    CHECK_THAT(d(0, 0, 0).real(), WithinAbs(1.0, 1e-9));
    // widefield: DC is the peak, and the kz axis is symmetric (real PSF, even in z)
    for (Eigen::Index iz = 1; iz < 16; ++iz) {
        CHECK(std::abs(d(0, 0, iz)) < 1.0);
        CHECK_THAT(std::abs(d(0, 0, iz)), WithinAbs(std::abs(d(0, 0, 32 - iz)), 1e-9));
    }
    // missing cone: on the kz axis the OTF drops far below the in-focus value
    // at the same kr for moderate defocus frequencies
    CHECK(std::abs(d(0, 0, 4)) < 0.05);
    // order 2 equals order 0, order 1 is different (axially shifted) and symmetric in kz
    for (Eigen::Index ir = 0; ir < 65; ++ir)
        for (Eigen::Index iz = 0; iz < 32; ++iz) {
            CHECK(d(2, ir, iz) == d(0, ir, iz));
            CHECK_THAT(std::abs(d(1, ir, iz)), WithinAbs(std::abs(d(1, ir, (32 - iz) % 32)), 1e-9));
        }
    CHECK(std::abs(d(1, 0, 0)) < std::abs(d(0, 0, 0)));
    // order 1 peaks away from kz = 0 (the +-kz1 lobes)
    Eigen::Index peak = 0;
    for (Eigen::Index iz = 0; iz < 32; ++iz)
        if (std::abs(d(1, 0, iz)) > std::abs(d(1, 0, peak))) peak = iz;
    CHECK(peak != 0);
}

TEST_CASE("idealOTF tabulates the derived order count", "[otf][ideal][orders]") {
    SIMParameters p = lowNaParams();
    p.norders = 0;
    p.nphases = 3;
    CHECK(idealOTF(p, false).data().dimension(0) == 2);
    p.nphases = 5;
    CHECK(idealOTF(p, false).data().dimension(0) == 3);
}

TEST_CASE("idealOTF rejects unphysical inputs", "[otf][ideal]") {
    SIMParameters p = lowNaParams();
    // an NA above the immersion index is invalid parameters for everything
    p.na = 1.5;
    p.nimm = 1.33;
    REQUIRE_THROWS_AS(idealOTF(p, false), std::runtime_error);
    // an NA equal to it is valid, but the ideal pupil needs it strictly below
    p.na = 1.33;
    REQUIRE_NOTHROW(p.validate());
    REQUIRE_THROWS_AS(idealOTF(p, false), std::invalid_argument);
    IdealOtfOptions bad;
    bad.lateralSamples = 15;
    REQUIRE_THROWS_AS(idealOTF(lowNaParams(), false, bad), std::invalid_argument);
}
