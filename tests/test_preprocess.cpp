#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sirius/preprocess.hpp"

using namespace sirius;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {
    using Stack = Eigen::Tensor<double, 5, Eigen::RowMajor>;

    Stack makeStack(int nd, int np, int nz, int ny, int nx) {
        Stack s(nd, np, nz, ny, nx);
        s.setZero();
        return s;
    }
    void fillSection(Stack& s, int d, int p, int z, double val) {
        for (int y = 0; y < s.dimension(3); ++y)
            for (int x = 0; x < s.dimension(4); ++x)
                s(d, p, z, y, x) = val;
    }
    double sectionSum(const Stack& s, int d, int p, int z) {
        double acc = 0.0;
        for (int y = 0; y < s.dimension(3); ++y)
            for (int x = 0; x < s.dimension(4); ++x)
                acc += s(d, p, z, y, x);
        return acc;
    }

    // 2 dirs, 3 phases, 2 z, 2x2 sections (sum = 4 * value). Direction 0 /
    // phase 0 is the reference anchor: S(0,0,0)=40, S(0,0,1)=20. Direction 1 is
    // deliberately on a different brightness scale to exercise cross-direction
    // tying.
    Stack bleachedStack() {
        auto s = makeStack(2, 3, 2, 2, 2);
        fillSection(s, 0, 0, 0, 10.0); fillSection(s, 0, 1, 0, 8.0); fillSection(s, 0, 2, 0, 6.0);
        fillSection(s, 0, 0, 1,  5.0); fillSection(s, 0, 1, 1, 4.0); fillSection(s, 0, 2, 1, 3.0);
        fillSection(s, 1, 0, 0, 20.0); fillSection(s, 1, 1, 0, 16.0); fillSection(s, 1, 2, 0, 12.0);
        fillSection(s, 1, 0, 1,  2.0); fillSection(s, 1, 1, 1,  1.0); fillSection(s, 1, 2, 1,  0.5);
        return s;
    }
}

TEST_CASE("bleach_rescale ties all dirs/phases to dir0/phase0 per plane (equalizez=false)", "[preprocess]") {
    Stack s = bleachedStack();

    bleach_rescale(s, /*equalizez=*/false);

    const double refZ0 = 40.0; // S(0,0,0)
    const double refZ1 = 20.0; // S(0,0,1)
    for (int d = 0; d < 2; ++d)
        for (int p = 0; p < 3; ++p) {
            CHECK_THAT(sectionSum(s, d, p, 0), WithinRel(refZ0, 1e-12));
            CHECK_THAT(sectionSum(s, d, p, 1), WithinRel(refZ1, 1e-12));
        }
    // (dir 0, phase 0) is the anchor, so it is unchanged
    CHECK_THAT(sectionSum(s, 0, 0, 0), WithinRel(40.0, 1e-12));
    // axial profile of the anchor is preserved (z=0 brighter than z=1)
    CHECK(refZ0 > refZ1);
}

TEST_CASE("bleach_rescale ties every plane to S(0,0,0) (equalizez=true)", "[preprocess]") {
    Stack s = bleachedStack();

    bleach_rescale(s, /*equalizez=*/true);

    const double ref = 40.0; // S(0,0,0)
    for (int d = 0; d < 2; ++d)
        for (int z = 0; z < 2; ++z)
            for (int p = 0; p < 3; ++p)
                CHECK_THAT(sectionSum(s, d, p, z), WithinRel(ref, 1e-12));
}

TEST_CASE("bleach_rescale leaves a dark section untouched (no divide-by-zero)", "[preprocess]") {
    auto s = makeStack(1, 2, 1, 2, 2);
    fillSection(s, 0, 0, 0, 10.0); // anchor, sum 40; phase 1 stays dark
    bleach_rescale(s, false);
    CHECK_THAT(sectionSum(s, 0, 0, 0), WithinRel(40.0, 1e-12)); // anchor: factor 1
    CHECK(sectionSum(s, 0, 1, 0) == 0.0);                       // dark: skipped, no NaN
}

TEST_CASE("edge_apodization blends opposite edges (exact 2x2, napodize=1)", "[preprocess]") {
    // Hand-computed: x-pass blends rows then y-pass blends cols, fact(0) =
    // 1 - sin(pi/4) = 0.2928932188. See the kernel math in the reference.
    auto s = makeStack(1, 1, 1, 2, 2);
    s(0, 0, 0, 0, 0) = 1.0; s(0, 0, 0, 0, 1) = 2.0;
    s(0, 0, 0, 1, 0) = 3.0; s(0, 0, 0, 1, 1) = 4.0;

    edge_apodization(s, 1);

    CHECK_THAT(s(0, 0, 0, 0, 0), WithinAbs(1.4393398282, 1e-9));
    CHECK_THAT(s(0, 0, 0, 0, 1), WithinAbs(2.1464466094, 1e-9));
    CHECK_THAT(s(0, 0, 0, 1, 0), WithinAbs(2.8535533906, 1e-9));
    CHECK_THAT(s(0, 0, 0, 1, 1), WithinAbs(3.5606601718, 1e-9));
}

TEST_CASE("edge_apodization leaves a constant image unchanged", "[preprocess]") {
    // Opposite edges are equal -> diff is zero everywhere -> no change.
    auto s = makeStack(1, 1, 1, 3, 3);
    fillSection(s, 0, 0, 0, 5.0);
    edge_apodization(s, 1);
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 3; ++x)
            CHECK_THAT(s(0, 0, 0, y, x), WithinAbs(5.0, 1e-12));
}

TEST_CASE("edge_apodization is a no-op for napodize <= 0", "[preprocess]") {
    auto s = makeStack(1, 1, 1, 2, 2);
    s(0, 0, 0, 0, 0) = 1.0; s(0, 0, 0, 0, 1) = 2.0;
    s(0, 0, 0, 1, 0) = 3.0; s(0, 0, 0, 1, 1) = 4.0;
    const Stack before = s;
    edge_apodization(s, 0);
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x)
            CHECK(s(0, 0, 0, y, x) == before(0, 0, 0, y, x));
}

TEST_CASE("cosine_apodization applies a separable sine window", "[preprocess]") {
    // 2x2: every factor is sin(pi*(i+0.5)/2) = sin(pi/4); product = 0.5.
    auto s = makeStack(1, 1, 1, 2, 2);
    fillSection(s, 0, 0, 0, 1.0);
    cosine_apodization(s);
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x)
            CHECK_THAT(s(0, 0, 0, y, x), WithinAbs(0.5, 1e-12));
}
