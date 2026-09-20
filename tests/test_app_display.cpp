// The display mapping: the window and the 8-bit gray the viewer paints from a
// float volume, and the ranges it offers for one.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <vector>

#include "core/array.hpp"
#include "core/display_mapping.hpp"

using namespace sirius;
using namespace sirius::app;
using Catch::Matchers::WithinAbs;


TEST_CASE("minMaxRange ignores NaN and reports empty input as invalid", "[app][display]") {
    const double v[] = {3.0, std::numeric_limits<double>::quiet_NaN(), -1.5, 7.25};
    const DisplayRange r = minMaxRange(v, 4);
    CHECK(r.lo == -1.5);
    CHECK(r.hi == 7.25);
    CHECK(r.valid());

    CHECK_FALSE(minMaxRange(v, 0).valid());
    const double nan = std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(minMaxRange(&nan, 1).valid());
}

TEST_CASE("percentileRange clips outliers and falls back to min/max on a constant", "[app][display]") {
    std::vector<double> v(1000);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<double>(i);   // 0..999
    v[0] = -1e9;    // one cold and one hot pixel
    v[999] = 1e9;

    const DisplayRange r = percentileRange(v.data(), static_cast<Index>(v.size()), 0.01, 0.99);
    CHECK(r.lo > 0.0);
    CHECK(r.lo < 20.0);
    CHECK(r.hi > 980.0);
    CHECK(r.hi < 999.0);

    SECTION("subsampling keeps the estimate bounded and close") {
        const DisplayRange sub = percentileRange(v.data(), static_cast<Index>(v.size()), 0.01, 0.99, 100);
        CHECK(sub.valid());
        CHECK(sub.lo > -1e8);
        CHECK(sub.hi < 1e8);
    }
    SECTION("degenerate window widens to min/max") {
        std::vector<double> flat(50, 4.0);
        flat[10] = 9.0;
        const DisplayRange f = percentileRange(flat.data(), 50, 0.1, 0.9);
        CHECK(f.lo == 4.0);
        CHECK(f.hi == 9.0);
    }
}

TEST_CASE("mapToGray8 clamps, rounds and zeroes NaN", "[app][display]") {
    const double src[] = {-10.0, 0.0, 0.5, 1.0, 20.0, std::numeric_limits<double>::quiet_NaN()};
    std::uint8_t dst[6];
    mapToGray8(src, 6, DisplayRange{0.0, 1.0}, dst);
    CHECK(dst[0] == 0);
    CHECK(dst[1] == 0);
    CHECK(dst[2] == 128);   // round(127.5)
    CHECK(dst[3] == 255);
    CHECK(dst[4] == 255);
    CHECK(dst[5] == 0);

    SECTION("an invalid range maps everything to black") {
        mapToGray8(src, 6, DisplayRange{1.0, 1.0}, dst);
        for (std::uint8_t g : dst) CHECK(g == 0);
    }
}

TEST_CASE("Dims5::planeIndex refuses a plane outside the extents", "[app][array]") {
    using namespace sirius::app;
    const Dims5 d{2, 3, 4, 5, 6};
    CHECK(d.planeIndex(1, 2, 3) == d.planes() - 1);
    CHECK(d.planeIndex(0, 0, 0) == 0);
    CHECK_THROWS_AS(d.planeIndex(2, 0, 0), std::out_of_range);
    CHECK_THROWS_AS(d.planeIndex(0, 3, 0), std::out_of_range);
    CHECK_THROWS_AS(d.planeIndex(0, 0, -1), std::out_of_range);
}
