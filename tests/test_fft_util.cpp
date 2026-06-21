#include "sirius/fft_util.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("fftfreq matches numpy convention", "[fft_util]") {
    SECTION("n=4, d=1") {
        const auto f = sirius::fftfreq(4);
        REQUIRE(f.size() == 4);
        REQUIRE_THAT(f(0), Catch::Matchers::WithinAbs( 0.0,  1e-15));
        REQUIRE_THAT(f(1), Catch::Matchers::WithinAbs( 0.25, 1e-15));
        REQUIRE_THAT(f(2), Catch::Matchers::WithinAbs(-0.5,  1e-15));
        REQUIRE_THAT(f(3), Catch::Matchers::WithinAbs(-0.25,1e-15));
    }

    SECTION("n=5, d=2") {
        const auto f = sirius::fftfreq(5, 2.0);
        // [0, 0.1, 0.2, -0.2, -0.1]
        REQUIRE_THAT(f(0), Catch::Matchers::WithinAbs( 0.0, 1e-15));
        REQUIRE_THAT(f(1), Catch::Matchers::WithinAbs( 0.1, 1e-15));
        REQUIRE_THAT(f(2), Catch::Matchers::WithinAbs( 0.2, 1e-15));
        REQUIRE_THAT(f(3), Catch::Matchers::WithinAbs(-0.2, 1e-15));
        REQUIRE_THAT(f(4), Catch::Matchers::WithinAbs(-0.1, 1e-15));
    }

    SECTION("preallocated overload") {
        Eigen::VectorXd out(8);
        sirius::fftfreq(8, 0.1, out);
        const auto f = sirius::fftfreq(8, 0.1);
        REQUIRE(out.isApprox(f));
    }
}