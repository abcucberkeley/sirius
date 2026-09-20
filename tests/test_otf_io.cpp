// loadOTF: the radially averaged OTF table read from a TIFF.

#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <stdexcept>

#include "sirius/otf_io.hpp"
#include "sirius/tiff_io.hpp"

#include "temp_path.hpp"

using namespace sirius;
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
