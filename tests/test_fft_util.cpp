#include "sirius/fft_util.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <array>
#include <complex>
#include <initializer_list>

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

    // Exact equality: a shift only moves data, so results are bit-identical.
    template <typename T, int Rank>
    bool exact_equal(const Eigen::Tensor<T, Rank, Eigen::RowMajor>& a,
                     const Eigen::Tensor<T, Rank, Eigen::RowMajor>& b) {
        if (a.dimensions() != b.dimensions()) return false;
        for (Eigen::Index i = 0; i < a.size(); ++i)
            if (a.data()[i] != b.data()[i]) return false;
        return true;
    }

    template <int Rank>
    Eigen::Tensor<std::complex<double>, Rank, Eigen::RowMajor>
    random_complex(const Eigen::array<Eigen::Index, Rank>& dims) {
        Eigen::Tensor<std::complex<double>, Rank, Eigen::RowMajor> t(dims);
        t.setRandom();
        return t;
    }
}

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

// -----------------------------------------------------------------------
// fftshift / ifftshift - value checks against the numpy convention
// -----------------------------------------------------------------------

TEST_CASE("fftshift 1D matches numpy (even and odd)", "[fft_util][fftshift]") {
    SECTION("even n=4") {
        REQUIRE(exact_equal(sirius::fftshift(vec1d({0, 1, 2, 3})),
                            vec1d({2, 3, 0, 1})));
    }
    SECTION("odd n=5") {
        REQUIRE(exact_equal(sirius::fftshift(vec1d({0, 1, 2, 3, 4})),
                            vec1d({3, 4, 0, 1, 2})));
    }
}

TEST_CASE("ifftshift 1D matches numpy (even and odd)", "[fft_util][fftshift]") {
    SECTION("even n=4 (same as fftshift)") {
        REQUIRE(exact_equal(sirius::ifftshift(vec1d({0, 1, 2, 3})),
                            vec1d({2, 3, 0, 1})));
    }
    SECTION("odd n=5 (differs from fftshift)") {
        REQUIRE(exact_equal(sirius::ifftshift(vec1d({0, 1, 2, 3, 4})),
                            vec1d({2, 3, 4, 0, 1})));
    }
}

TEST_CASE("fftshift moves the zero-frequency bin to the center", "[fft_util][fftshift]") {
    // An impulse at index 0 must land at floor(n/2).
    auto n = GENERATE(2, 3, 7, 8);
    INFO("n = " << n);
    RTensor<1> impulse(n);
    impulse.setZero();
    impulse(0) = 1.0;

    const auto shifted = sirius::fftshift(impulse);
    const Eigen::Index center = n / 2;
    for (Eigen::Index i = 0; i < n; ++i)
        REQUIRE(shifted(i) == (i == center ? 1.0 : 0.0));
}

TEST_CASE("fftshift 2D matches numpy on a 3x4 grid", "[fft_util][fftshift]") {
    const auto in = mat2d({{0, 1, 2, 3},
                           {4, 5, 6, 7},
                           {8, 9, 10, 11}});
    // fftshift = roll by (floor(3/2), floor(4/2)) = (1, 2)
    const auto expected = mat2d({{10, 11, 8, 9},
                                 { 2,  3, 0, 1},
                                 { 6,  7, 4, 5}});
    REQUIRE(exact_equal(sirius::fftshift(in), expected));
}

// -----------------------------------------------------------------------
// Round-trip / inverse properties - the key correctness guarantees
// -----------------------------------------------------------------------

TEST_CASE("ifftshift undoes fftshift for odd lengths (1D/2D/3D)", "[fft_util][fftshift]") {
    SECTION("1D odd") {
        const auto x = random_complex<1>({7});
        REQUIRE(exact_equal(sirius::ifftshift(sirius::fftshift(x)), x));
        REQUIRE(exact_equal(sirius::fftshift(sirius::ifftshift(x)), x));
    }
    SECTION("2D mixed even/odd") {
        const auto x = random_complex<2>({5, 8});
        REQUIRE(exact_equal(sirius::ifftshift(sirius::fftshift(x)), x));
        REQUIRE(exact_equal(sirius::fftshift(sirius::ifftshift(x)), x));
    }
    SECTION("3D all-odd") {
        const auto x = random_complex<3>({3, 5, 7});
        REQUIRE(exact_equal(sirius::ifftshift(sirius::fftshift(x)), x));
        REQUIRE(exact_equal(sirius::fftshift(sirius::ifftshift(x)), x));
    }
}

TEST_CASE("fftshift equals ifftshift only when all axes are even", "[fft_util][fftshift]") {
    SECTION("even axes: identical, and applying twice is identity") {
        const auto x = random_complex<2>({4, 6});
        REQUIRE(exact_equal(sirius::fftshift(x), sirius::ifftshift(x)));
        REQUIRE(exact_equal(sirius::fftshift(sirius::fftshift(x)), x));
    }
    SECTION("odd axis: fftshift and ifftshift differ") {
        const auto x = random_complex<1>({5});
        REQUIRE_FALSE(exact_equal(sirius::fftshift(x), sirius::ifftshift(x)));
    }
}