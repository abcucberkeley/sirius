#include "sirius/tensor_util.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <array>
#include <complex>
#include <initializer_list>

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

namespace {
    template <int Rank> using RTensor = Eigen::Tensor<double, Rank, Eigen::RowMajor>;

    RTensor<1> vec1d(std::initializer_list<double> vals) {
        RTensor<1> t(static_cast<Eigen::Index>(vals.size()));
        Eigen::Index i = 0;
        for (double v : vals) t(i++) = v;
        return t;
    }

    RTensor<2> mat2d(std::initializer_list<std::initializer_list<double>> rows) {
        const Eigen::Index nr = static_cast<Eigen::Index>(rows.size());
        const Eigen::Index nc = static_cast<Eigen::Index>(rows.begin()->size());
        RTensor<2> t(nr, nc);
        Eigen::Index r = 0;
        for (const auto& row : rows) {
            Eigen::Index c = 0;
            for (double v : row) t(r, c++) = v;
            ++r;
        }
        return t;
    }

    // Exact element-wise equality. roll only moves data (no arithmetic), so
    // results are bit-identical and exact comparison is the right check —
    // any tolerance would mask indexing bugs.
    template <typename T, int Rank>
    bool exact_equal(const Eigen::Tensor<T, Rank, Eigen::RowMajor>& a,
                     const Eigen::Tensor<T, Rank, Eigen::RowMajor>& b) {
        if (a.dimensions() != b.dimensions()) return false;
        for (Eigen::Index i = 0; i < a.size(); ++i)
            if (a.data()[i] != b.data()[i]) return false;
        return true;
    }
}

// -----------------------------------------------------------------------
// roll - 1D against numpy.roll
// -----------------------------------------------------------------------

TEST_CASE("roll 1D matches numpy.roll", "[tensor_util][roll]") {
    const auto x = vec1d({0, 1, 2, 3, 4});

    SECTION("positive shift") {
        REQUIRE(exact_equal(sirius::roll(x, std::array<int, 1>{2}),
                            vec1d({3, 4, 0, 1, 2})));
    }
    SECTION("negative shift") {
        REQUIRE(exact_equal(sirius::roll(x, std::array<int, 1>{-1}),
                            vec1d({1, 2, 3, 4, 0})));
    }
    SECTION("zero shift is identity") {
        REQUIRE(exact_equal(sirius::roll(x, std::array<int, 1>{0}), x));
    }
    SECTION("shift by n is identity") {
        REQUIRE(exact_equal(sirius::roll(x, std::array<int, 1>{5}), x));
    }
    SECTION("shift larger than n wraps (7 == 2 mod 5)") {
        REQUIRE(exact_equal(sirius::roll(x, std::array<int, 1>{7}),
                            sirius::roll(x, std::array<int, 1>{2})));
    }
    SECTION("very negative shift wraps (-12 == 3 mod 5)") {
        REQUIRE(exact_equal(sirius::roll(x, std::array<int, 1>{-12}),
                            sirius::roll(x, std::array<int, 1>{3})));
    }
}

// -----------------------------------------------------------------------
// roll - multi-dimensional, each axis shifted independently
// -----------------------------------------------------------------------

TEST_CASE("roll 2D shifts each axis independently", "[tensor_util][roll]") {
    // in(r,c) = r*4 + c on a 3x4 grid; roll rows by +1, cols by +2.
    // out(r,c) == in((r-1) mod 3, (c-2) mod 4).
    const auto in = mat2d({{0, 1, 2, 3},
                           {4, 5, 6, 7},
                           {8, 9, 10, 11}});
    const auto out = sirius::roll(in, std::array<int, 2>{1, 2});
    const auto expected = mat2d({{10, 11, 8, 9},
                                 { 2,  3, 0, 1},
                                 { 6,  7, 4, 5}});
    REQUIRE(exact_equal(out, expected));
}

TEST_CASE("roll 2D with zero shift on one axis", "[tensor_util][roll]") {
    const auto in = mat2d({{0, 1, 2, 3},
                           {4, 5, 6, 7},
                           {8, 9, 10, 11}});
    // shift columns only
    const auto out = sirius::roll(in, std::array<int, 2>{0, 1});
    const auto expected = mat2d({{3, 0, 1, 2},
                                 {7, 4, 5, 6},
                                 {11, 8, 9, 10}});
    REQUIRE(exact_equal(out, expected));
}

// -----------------------------------------------------------------------
// roll - inverse property and rank-3 coverage
// -----------------------------------------------------------------------

TEST_CASE("roll by s then by -s recovers the original", "[tensor_util][roll]") {
    Eigen::Tensor<std::complex<double>, 3, Eigen::RowMajor> x(3, 5, 4);
    x.setRandom();

    const std::array<int, 3> s{1, -2, 3};
    const std::array<int, 3> neg{-1, 2, -3};
    REQUIRE(exact_equal(sirius::roll(sirius::roll(x, s), neg), x));
}

TEST_CASE("roll works for any scalar type", "[tensor_util][roll]") {
    // Same geometric operation must hold for complex data.
    Eigen::Tensor<std::complex<double>, 1, Eigen::RowMajor> x(4);
    x(0) = {1, -1};
    x(1) = {2, -2};
    x(2) = {3, -3};
    x(3) = {4, -4};

    const auto out = sirius::roll(x, std::array<int, 1>{1});
    REQUIRE(out(0) == std::complex<double>(4, -4));
    REQUIRE(out(1) == std::complex<double>(1, -1));
    REQUIRE(out(2) == std::complex<double>(2, -2));
    REQUIRE(out(3) == std::complex<double>(3, -3));
}
