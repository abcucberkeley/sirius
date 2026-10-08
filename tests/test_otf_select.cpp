// selectOTF: the one choice between a measured OTF file and the theoretical
// OTF, and the stack arithmetic that choice depends on.
//
// Why this is a unit of its own: until 2026-10-08 the choice was a ternary
// inside app/core/session.cpp, which the Python bindings do not link, so
// `step_sim` raised NotAvailable without a measured OTF file -- and since the
// application's export_python writes a script that calls run_pipeline, every
// default-OTF SIM step exported from the GUI or the CLI died on its own
// pipeline (docs/findings.md 9k.50, finding 4).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <complex>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "sirius/otf_io.hpp"
#include "sirius/otf_select.hpp"

using namespace sirius;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {
    // config.txt, the cudasirecon config of tests/data/raw.tif
    SIMParameters referenceParams() {
        SIMParameters p;
        p.ndirs = 3;
        p.nphases = 5;
        p.na = 1.42;
        p.nimm = 1.515;
        p.wavelength_nm = 510.0;
        p.linespacing_um = 0.2035;
        p.dx = p.dy = 0.08;
        p.dz = 0.125;
        p.dz_psf = 0.125;
        return p;
    }
} // namespace

TEST_CASE("an empty OTF path is the theoretical OTF, and threeD picks the table", "[otf][select]") {
    const SIMParameters p = referenceParams();

    const OTFRadiallyAveraged three = selectOTF("", p, /*threeD=*/true);
    const OTFRadiallyAveraged ideal3 = idealOTF(p, /*threeD=*/true);
    REQUIRE(three.data().dimension(0) == ideal3.data().dimension(0));
    REQUIRE(three.data().dimension(1) == ideal3.data().dimension(1));
    REQUIRE(three.data().dimension(2) == ideal3.data().dimension(2));
    CHECK(three.data().dimension(2) == 64);   // IdealOtfOptions::axialSamples
    CHECK_THAT(three.dkrotf(), WithinRel(ideal3.dkrotf(), 1e-15));
    CHECK_THAT(three.dkzotf(), WithinRel(ideal3.dkzotf(), 1e-15));
    // the same table, value for value: selectOTF calls idealOTF, it does not
    // reimplement it
    for (Eigen::Index o = 0; o < three.data().dimension(0); ++o)
        for (Eigen::Index ir = 0; ir < three.data().dimension(1); ir += 17)
            for (Eigen::Index iz = 0; iz < three.data().dimension(2); iz += 7) {
                INFO("order " << o << " kr " << ir << " kz " << iz);
                CHECK(three.data()(o, ir, iz) == ideal3.data()(o, ir, iz));
            }

    // threeD=false is the in-focus 2D OTF: one kz sample, not 64. A front that
    // guesses this wrong reconstructs a 2D stack with a 3D OTF and says
    // nothing.
    const OTFRadiallyAveraged two = selectOTF("", p, /*threeD=*/false);
    CHECK(two.data().dimension(2) == 1);
    CHECK(two.data().dimension(2) != three.data().dimension(2));
}

TEST_CASE("a named OTF path is loadOTF, and threeD does not touch it", "[otf][select][data]") {
    const std::string file = (std::filesystem::path(SIRIUS_TEST_DATA_DIR) / "otf.tif").string();
    const SIMParameters p = referenceParams();
    const OTFRadiallyAveraged measured = selectOTF(file, p, /*threeD=*/true);
    const OTFRadiallyAveraged loaded = loadOTF(file, p);
    REQUIRE(measured.data().dimension(0) == 3);
    REQUIRE(measured.data().dimension(1) == 129);
    REQUIRE(measured.data().dimension(2) == 65);
    CHECK_THAT(measured.dkrotf(), WithinAbs(0.048828, 1e-6));
    CHECK_THAT(measured.dkrotf(), WithinRel(loaded.dkrotf(), 1e-15));
    CHECK_THAT(measured.dkzotf(), WithinRel(loaded.dkzotf(), 1e-15));
    CHECK(measured.data()(1, 0, 9) == loaded.data()(1, 0, 9));
    // threeD is for the theoretical table only
    const OTFRadiallyAveraged asTwoD = selectOTF(file, p, /*threeD=*/false);
    CHECK(asTwoD.data().dimension(2) == 65);
    CHECK(asTwoD.data()(1, 0, 9) == loaded.data()(1, 0, 9));

    const std::string missing = (std::filesystem::path(SIRIUS_TEST_DATA_DIR) / "no-such-otf.tif").string();
    CHECK_THROWS(selectOTF(missing, p, true));
}

TEST_CASE("SIMParameters::planes is the stack arithmetic every front shares", "[otf][select][sim_parameters]") {
    SIMParameters p = referenceParams();           // 3 angles x 5 phases
    CHECK(p.sectionsPerPlane() == 15);
    CHECK(p.planes(135) == 9);                     // tests/data/raw.tif
    CHECK(p.planes(15) == 1);                      // one plane: a 2D stack
    CHECK(p.planes(134) == 0);                     // not a whole number of planes
    CHECK(p.planes(0) == 0);
    // the 2D / 3D question the theoretical OTF asks, in the form the fronts
    // ask it (app/core/session.cpp's threeD, step_sim's three_d)
    CHECK(p.planes(135) > 1);
    CHECK_FALSE(p.planes(15) > 1);
    p.ndirs = 1;
    p.nphases = 3;
    CHECK(p.sectionsPerPlane() == 3);
    CHECK(p.planes(303) == 101);                   // the isoar crop: 1 angle x 3 phases x 101 z
    p.nphases = 0;                                 // degenerate parameters divide by nothing
    CHECK(p.sectionsPerPlane() == 0);
    CHECK(p.planes(135) == 0);
}
