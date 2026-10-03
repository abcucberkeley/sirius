#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>

#include "sirius/constants.hpp"
#include "sirius/separation.hpp"

using namespace sirius;
using Catch::Approx;

TEST_CASE("Ideal separation matrix has the makematrix cos/sin structure", "[separation]") {
    const int nphases = 5, norders = 3;
    const Eigen::MatrixXd m = separationMatrix(nphases, norders);

    REQUIRE(m.rows() == 2 * norders - 1);
    REQUIRE(m.cols() == nphases);

    const double phi = 2.0 * kPi / nphases;
    for (int j = 0; j < nphases; ++j) {
        CHECK(m(0, j) == Approx(1.0));
        for (int order = 1; order < norders; ++order) {
            CHECK(m(2 * order - 1, j) == Approx(std::cos(j * order * phi)).margin(1e-14));
            CHECK(m(2 * order, j) == Approx(std::sin(j * order * phi)).margin(1e-14));
        }
    }
}

TEST_CASE("General-phase separation matrix reproduces the ideal one for ideal phases",
          "[separation]") {
    for (int nphases : {5, 7}) {
        const int norders = nphases / 2 + 1;
        const Eigen::MatrixXd ideal = separationMatrix(nphases, norders);
        const Eigen::MatrixXd general = separationMatrix(idealPhases(nphases), norders);
        REQUIRE(general.rows() == ideal.rows());
        REQUIRE(general.cols() == ideal.cols());
        CHECK((general - ideal).cwiseAbs().maxCoeff() < 1e-10);
    }
}

TEST_CASE("Separation recovers known bands from mixed phase images", "[separation]") {
    // forward-mix known bands with phi_j, un-mix, compare against the
    // makematrix normalization: band0 x nphases, side bands x nphases/2
    const int nphases = 5, norders = 3;
    const Eigen::Index nz = 2, ny = 4, nx = 4;

    Eigen::Tensor<double, 4, Eigen::RowMajor> bandsTrue(2 * norders - 1, nz, ny, nx);
    bandsTrue.setRandom();

    const Eigen::VectorXd phases = steppedPhases(nphases, 2.0 * kPi / nphases);
    Eigen::Tensor<double, 4, Eigen::RowMajor> stack(nphases, nz, ny, nx);
    stack.setZero();
    for (int j = 0; j < nphases; ++j)
        for (Eigen::Index z = 0; z < nz; ++z)
            for (Eigen::Index y = 0; y < ny; ++y)
                for (Eigen::Index x = 0; x < nx; ++x) {
                    double v = bandsTrue(0, z, y, x);
                    for (int o = 1; o < norders; ++o)
                        v += std::cos(o * phases(j)) * bandsTrue(2 * o - 1, z, y, x) +
                             std::sin(o * phases(j)) * bandsTrue(2 * o, z, y, x);
                    stack(j, z, y, x) = v;
                }

    const auto separated = separateBands(stack, separationMatrix(nphases, norders));
    REQUIRE(separated.dimension(0) == 2 * norders - 1);
    for (int b = 0; b < 2 * norders - 1; ++b) {
        const double scale = b == 0 ? nphases : nphases / 2.0;
        for (Eigen::Index z = 0; z < nz; ++z)
            for (Eigen::Index y = 0; y < ny; ++y)
                for (Eigen::Index x = 0; x < nx; ++x)
                    CHECK(separated(b, z, y, x) ==
                          Approx(scale * bandsTrue(b, z, y, x)).margin(1e-10));
    }
}

TEST_CASE("Separation recovers bands mixed with unequal phases", "[separation]") {
    const int nphases = 5, norders = 3;
    Eigen::VectorXd phases(nphases);
    phases << 0.0, 0.4, 1.3, 2.2, 4.5;
    const Eigen::Index nz = 2, ny = 3, nx = 3;
    Eigen::Tensor<double, 4, Eigen::RowMajor> bandsTrue(2 * norders - 1, nz, ny, nx);
    bandsTrue.setRandom();
    Eigen::Tensor<double, 4, Eigen::RowMajor> stack(nphases, nz, ny, nx);
    stack.setZero();
    for (int j = 0; j < nphases; ++j)
        for (Eigen::Index z = 0; z < nz; ++z)
            for (Eigen::Index y = 0; y < ny; ++y)
                for (Eigen::Index x = 0; x < nx; ++x) {
                    double v = bandsTrue(0, z, y, x);
                    for (int o = 1; o < norders; ++o)
                        v += std::cos(o * phases(j)) * bandsTrue(2 * o - 1, z, y, x) +
                             std::sin(o * phases(j)) * bandsTrue(2 * o, z, y, x);
                    stack(j, z, y, x) = v;
                }
    const auto separated = separateBands(stack, separationMatrix(phases, norders));
    for (int b = 0; b < 2 * norders - 1; ++b) {
        const double scale = b == 0 ? nphases : nphases / 2.0;
        for (Eigen::Index z = 0; z < nz; ++z)
            for (Eigen::Index y = 0; y < ny; ++y)
                for (Eigen::Index x = 0; x < nx; ++x)
                    CHECK(separated(b, z, y, x) == Approx(scale * bandsTrue(b, z, y, x)).margin(1e-8));
    }
}

TEST_CASE("Separation matrix rejects too few phases", "[separation]") {
    CHECK_THROWS(separationMatrix(4, 3));   // needs 2*3-1 = 5 phases
    CHECK_THROWS(separationMatrix(idealPhases(4), 3));
}
