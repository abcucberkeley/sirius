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

TEST_CASE("idealOTF order-1 shift follows resolvedOrders, not nphases/2+1", "[otf][ideal]") {
    // nphases = 5 makes (nphases/2+1)-1 = 2 for both of these. resolvedOrders()-1
    // is 2 when norders is 3 and 1 when norders is 2, so the order-1 lobe moves.
    SIMParameters p = lowNaParams();
    p.nphases = 5;
    p.linespacing_um = 0.2;
    p.dz_psf = 0.2;
    IdealOtfOptions opts;
    opts.lateralSamples = 64;
    opts.axialSamples = 32;
    opts.dzPsf = 0.2;
    p.norders = 3;
    const auto three = idealOTF(p, true, opts).data();
    p.norders = 2;
    const auto two = idealOTF(p, true, opts).data();
    REQUIRE(three.dimension(0) >= 2);
    REQUIRE(two.dimension(0) >= 2);
    double diff = 0;
    for (Eigen::Index ir = 0; ir < three.dimension(1); ++ir)
        for (Eigen::Index iz = 0; iz < three.dimension(2); ++iz)
            diff += std::abs(three(1, ir, iz) - two(1, ir, iz));
    CHECK(diff > 1e-3);
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

// --------------------------------------------------------------------------
// The illumination's axial component as a stated parameter
// (SIMParameters::illumination_has_axial_component).
//
// The order-1 shift above models THREE-BEAM interference: the two first
// diffraction orders beat against the undiffracted central beam, so the
// illumination is modulated in z and the side band's support is displaced
// along kz. A TWO-BEAM pattern has no central beam, its intensity is
// 1 + m cos(2 pi k0 . r + phi) with no z term, and its order-1 band therefore
// rides the plain widefield OTF. The iSOAR2 stacks and the lab's mmmSIM
// calibration data are that case; cudasirecon's 3-angle 5-phase test stack is
// not. The cases below pin the default (three-beam, so no existing result
// moves) and the new setting separately.
// --------------------------------------------------------------------------

TEST_CASE("idealOtfShiftsOrderOne states the rule in one place", "[otf][ideal][axial]") {
    SIMParameters p = lowNaParams();   // norders 3
    REQUIRE(p.illumination_has_axial_component);          // the default
    CHECK(idealOtfShiftsOrderOne(p, /*threeD=*/true));
    // a 2D table has one kz plane, so there is nowhere to shift to
    CHECK_FALSE(idealOtfShiftsOrderOne(p, /*threeD=*/false));
    // one order resolved means there is no order 1 at all
    SIMParameters one = p;
    one.norders = 1;
    CHECK_FALSE(idealOtfShiftsOrderOne(one, true));
    // and the stated parameter, which is the point of this change
    SIMParameters twoBeam = p;
    twoBeam.illumination_has_axial_component = false;
    CHECK_FALSE(idealOtfShiftsOrderOne(twoBeam, true));
    // two orders alone does NOT turn it off: a 5-phase three-beam stack
    // reconstructed with norders = 2 still has an axial component
    SIMParameters twoOrders = p;
    twoOrders.norders = 2;
    CHECK(idealOtfShiftsOrderOne(twoOrders, true));
}

TEST_CASE("a 3D two-beam table's order 1 is the plain widefield OTF", "[otf][ideal][axial]") {
    // The isoar acquisition's shape: 1 direction, 3 phases -> 2 orders, 3D.
    SIMParameters p = lowNaParams();
    p.na = 1.35;
    p.nimm = 1.405;
    p.wavelength_nm = 604.0;
    p.nphases = 3;
    p.norders = 0;            // -> 2
    p.linespacing_um = 0.491;
    p.dz_psf = 0.1;
    IdealOtfOptions opts;
    opts.lateralSamples = 128;
    opts.axialSamples = 32;
    REQUIRE(p.resolvedOrders() == 2);

    SIMParameters threeBeam = p;                                  // the default
    SIMParameters twoBeam = p;
    twoBeam.illumination_has_axial_component = false;
    const auto shifted = idealOTF(threeBeam, true, opts).data();
    const auto plain = idealOTF(twoBeam, true, opts).data();
    REQUIRE(shifted.dimension(0) == 2);
    REQUIRE(plain.dimension(0) == 2);

    // two-beam: order 1 IS order 0, sample for sample, not merely close
    for (Eigen::Index ir = 0; ir < plain.dimension(1); ++ir)
        for (Eigen::Index iz = 0; iz < plain.dimension(2); ++iz) {
            INFO("ir " << ir << " iz " << iz);
            REQUIRE(plain(1, ir, iz) == plain(0, ir, iz));
        }
    // order 0 is untouched by the parameter: the widefield OTF is the
    // widefield OTF whichever way the sample was illuminated
    for (Eigen::Index ir = 0; ir < plain.dimension(1); ++ir)
        for (Eigen::Index iz = 0; iz < plain.dimension(2); ++iz)
            REQUIRE(plain(0, ir, iz) == shifted(0, ir, iz));
    // and the two settings really do differ on order 1 for this acquisition,
    // so the parameter is not a no-op here
    double diff = 0.0;
    for (Eigen::Index ir = 0; ir < plain.dimension(1); ++ir)
        for (Eigen::Index iz = 0; iz < plain.dimension(2); ++iz)
            diff += std::abs(plain(1, ir, iz) - shifted(1, ir, iz));
    CHECK(diff > 1e-3);
    // the three-beam table's order 1 peaks off kz = 0; the two-beam one peaks
    // at kz = 0, where the widefield OTF does
    Eigen::Index peakShifted = 0, peakPlain = 0;
    for (Eigen::Index iz = 0; iz < 32; ++iz) {
        if (std::abs(shifted(1, 0, iz)) > std::abs(shifted(1, 0, peakShifted))) peakShifted = iz;
        if (std::abs(plain(1, 0, iz)) > std::abs(plain(1, 0, peakPlain))) peakPlain = iz;
    }
    CHECK(peakShifted != 0);
    CHECK(peakPlain == 0);
    CHECK_THAT(plain(1, 0, 0).real(), WithinAbs(1.0, 1e-9));   // DC of the widefield OTF
}

TEST_CASE("the axial-component parameter leaves a 2D table alone", "[otf][ideal][axial]") {
    // A 2D table's orders are all the widefield OTF already, so the parameter
    // has nothing to change and must change nothing.
    SIMParameters p = lowNaParams();
    SIMParameters twoBeam = p;
    twoBeam.illumination_has_axial_component = false;
    const auto a = idealOTF(p, false).data();
    const auto b = idealOTF(twoBeam, false).data();
    REQUIRE(a.dimension(0) == b.dimension(0));
    REQUIRE(a.dimension(1) == b.dimension(1));
    REQUIRE(a.dimension(2) == b.dimension(2));
    for (Eigen::Index o = 0; o < a.dimension(0); ++o)
        for (Eigen::Index ir = 0; ir < a.dimension(1); ++ir)
            REQUIRE(a(o, ir, 0) == b(o, ir, 0));
}
