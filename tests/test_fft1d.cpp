#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <Eigen/Core>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include "sirius/fft.hpp"

using namespace sirius;

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

namespace {
    constexpr double pi = 3.14159265358979323846;

    // Tolerance for double-precision FFT comparisons.
    // FFTW is accurate to ~machine epsilon * log2(N), this is comfortably above that.
    constexpr double kTol = 1e-10;

    // View a rank-1 tensor as an Eigen vector (no copy) so we can reuse Eigen algebra.
    Eigen::Map<Eigen::VectorXcd> as_vec(TensorXcd<1>& t) {
        return Eigen::Map<Eigen::VectorXcd>(t.data(), t.size());
    }
    Eigen::Map<const Eigen::VectorXcd> as_vec(const TensorXcd<1>& t) {
        return Eigen::Map<const Eigen::VectorXcd>(t.data(), t.size());
    }

    // Copy an Eigen vector / expression into a freshly-allocated rank-1 tensor.
    template <typename Derived>
    TensorXcd<1> to_tensor(const Eigen::MatrixBase<Derived>& expr) {
        const Eigen::VectorXcd v = expr;
        TensorXcd<1> t(v.size());
        std::copy(v.data(), v.data() + v.size(), t.data());
        return t;
    }

    TensorXcd<1> random_tensor(Eigen::Index n) {
        return to_tensor(Eigen::VectorXcd::Random(n));
    }

    // Build a complex sinusoid: x[k] = exp(2*pi*i * freq * k / n)
    // Its FFT has a single unit spike at bin `freq` and zeros elsewhere.
    TensorXcd<1> make_sinusoid(Eigen::Index n, int freq) {
        TensorXcd<1> t(n);
        for (Eigen::Index k = 0; k < n; ++k)
            t(k) = std::exp(std::complex<double>(0.0,
                2.0 * pi * freq * k / static_cast<double>(n)));
        return t;
    }

    double max_abs_error(const TensorXcd<1>& a, const TensorXcd<1>& b) {
        return (as_vec(a) - as_vec(b)).cwiseAbs().maxCoeff();
    }
}

// -----------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------

TEST_CASE("FFT 1D construction", "[fft1d]") {
    SECTION("Valid sizes construct without throwing") {
        auto n = GENERATE(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024);
        REQUIRE_NOTHROW(FFT({n}));
    }

    SECTION("All PlanRigor values construct successfully") {
        auto rigor = GENERATE(
            PlanRigor::Estimate,
            PlanRigor::Measure,
            PlanRigor::Patient,
            PlanRigor::Exhaustive
        );
        REQUIRE_NOTHROW(FFT({64}, 1, rigor));
    }

    SECTION("Non-power-of-two sizes are valid for FFTW") {
        // FFTW handles arbitrary sizes, not just powers of two
        auto n = GENERATE(3, 5, 7, 100, 127, 1000);
        REQUIRE_NOTHROW(FFT({n}));
    }

    SECTION("Empty dims throws invalid_argument") {
        REQUIRE_THROWS_AS(FFT({}), std::invalid_argument);
    }

    SECTION("howmany < 1 throws invalid_argument") {
        REQUIRE_THROWS_AS(FFT({64}, 0), std::invalid_argument);
        REQUIRE_THROWS_AS(FFT({64}, -1), std::invalid_argument);
    }
}

// -----------------------------------------------------------------------
// Move semantics
// -----------------------------------------------------------------------

TEST_CASE("FFT 1D move semantics", "[fft1d]") {
    SECTION("Move construction leaves object in valid state") {
        FFT a({64});
        FFT b(std::move(a));

        TensorXcd<1> in  = make_sinusoid(64, 3);
        TensorXcd<1> out(64);
        REQUIRE_NOTHROW(b.fft(in, out));
    }

    SECTION("Move assignment leaves object in valid state") {
        FFT a({64});
        FFT b({32});
        b = std::move(a);

        TensorXcd<1> in  = make_sinusoid(64, 3);
        TensorXcd<1> out(64);
        REQUIRE_NOTHROW(b.fft(in, out));
    }
}

// -----------------------------------------------------------------------
// Linearity (superposition)
// -----------------------------------------------------------------------

TEST_CASE("FFT 1D linearity - fft(a*x + b*y) == a*fft(x) + b*fft(y)", "[fft1d][correctness]") {
    const Eigen::Index n = 128;
    FFT fft({static_cast<int>(n)});

    Eigen::VectorXcd x = Eigen::VectorXcd::Random(n);
    Eigen::VectorXcd y = Eigen::VectorXcd::Random(n);
    const std::complex<double> a(2.0, -1.0);
    const std::complex<double> b(0.5,  3.0);

    TensorXcd<1> x_t        = to_tensor(x);
    TensorXcd<1> y_t        = to_tensor(y);
    TensorXcd<1> combined_t = to_tensor(a * x + b * y);

    TensorXcd<1> out_combined(n), out_x(n), out_y(n);
    fft.fft(combined_t, out_combined);
    fft.fft(x_t,        out_x);
    fft.fft(y_t,        out_y);

    Eigen::VectorXcd expected = a * as_vec(out_x) + b * as_vec(out_y);
    REQUIRE((as_vec(out_combined) - expected).cwiseAbs().maxCoeff() < kTol);
}

// -----------------------------------------------------------------------
// Impulse response - delta function
// -----------------------------------------------------------------------

TEST_CASE("FFT 1D of delta function is flat spectrum", "[fft1d][correctness]") {
    // x[0]=1, x[k]=0 for k>0  =>  X[f] = 1 for all f
    const Eigen::Index n = 256;
    FFT fft({static_cast<int>(n)});

    TensorXcd<1> in(n); in.setZero();
    in(0) = 1.0;

    TensorXcd<1> out(n);
    fft.fft(in, out);

    for (Eigen::Index f = 0; f < n; ++f)
        REQUIRE_THAT(std::abs(out(f)), Catch::Matchers::WithinAbs(1.0, kTol));
}

// -----------------------------------------------------------------------
// Single-frequency sinusoid - spike in spectrum
// -----------------------------------------------------------------------

TEST_CASE("FFT 1D of sinusoid produces spike at correct bin", "[fft1d][correctness]") {
    const Eigen::Index n = 128;
    FFT fft({static_cast<int>(n)});

    auto freq = GENERATE(0, 1, 5, 10, 63);
    INFO("Frequency bin: " << freq);

    TensorXcd<1> in  = make_sinusoid(n, freq);
    TensorXcd<1> out(n);
    fft.fft(in, out);

    for (Eigen::Index f = 0; f < n; ++f) {
        double expected_mag = (f == freq) ? static_cast<double>(n) : 0.0;
        REQUIRE_THAT(std::abs(out(f)), Catch::Matchers::WithinAbs(expected_mag, kTol * n));
    }
}

// -----------------------------------------------------------------------
// Forward-inverse round-trip
// -----------------------------------------------------------------------

TEST_CASE("FFT 1D forward-inverse round-trip recovers original signal", "[fft1d][correctness]") {
    auto n = GENERATE(16, 64, 128, 256, 1024);
    INFO("N = " << n);

    FFT fft({n});
    TensorXcd<1> original = random_tensor(n);

    TensorXcd<1> freq_domain(n), recovered(n);
    fft.fft(original, freq_domain);
    fft.ifft(freq_domain, recovered, /*normalize=*/true);

    REQUIRE(max_abs_error(recovered, original) < kTol);
}

// -----------------------------------------------------------------------
// Parseval's theorem - energy conservation
// -----------------------------------------------------------------------

TEST_CASE("FFT 1D satisfies Parseval's theorem", "[fft1d][correctness]") {
    // sum|x[k]|^2 == (1/N) * sum|X[f]|^2
    const Eigen::Index n = 256;
    FFT fft({static_cast<int>(n)});

    TensorXcd<1> in  = random_tensor(n);
    TensorXcd<1> out(n);
    fft.fft(in, out);

    double energy_time = as_vec(in).squaredNorm();
    double energy_freq = as_vec(out).squaredNorm() / static_cast<double>(n);

    REQUIRE_THAT(energy_freq, Catch::Matchers::WithinRel(energy_time, 1e-10));
}

// -----------------------------------------------------------------------
// Shift theorem - time shift = linear phase in frequency
// -----------------------------------------------------------------------

TEST_CASE("FFT 1D shift theorem - time shift is linear phase in frequency", "[fft1d][correctness]") {
    // If y[k] = x[k - d] (circular), then Y[f] = X[f] * exp(-2*pi*i*f*d/N)
    const Eigen::Index n = 64;
    const int d = 5; // shift amount
    FFT fft({static_cast<int>(n)});

    Eigen::VectorXcd x = Eigen::VectorXcd::Random(n);

    // circular shift by d
    Eigen::VectorXcd y(n);
    for (Eigen::Index k = 0; k < n; ++k)
        y[k] = x[(k - d + n) % n];

    TensorXcd<1> x_t = to_tensor(x), y_t = to_tensor(y);
    TensorXcd<1> X(n), Y(n);
    fft.fft(x_t, X);
    fft.fft(y_t, Y);

    // verify Y[f] == X[f] * exp(-2*pi*i*f*d/N) for each bin f
    for (Eigen::Index f = 0; f < n; ++f) {
        std::complex<double> phase = std::exp(std::complex<double>(0.0, -2.0 * pi * f * d / static_cast<double>(n)));
        std::complex<double> expected = X(f) * phase;
        REQUIRE_THAT(std::abs(Y(f) - expected), Catch::Matchers::WithinAbs(0.0, kTol * n));
    }
}

// -----------------------------------------------------------------------
// Conjugate symmetry - real-valued input
// -----------------------------------------------------------------------

TEST_CASE("FFT 1D conjugate symmetry for real-valued input", "[fft1d][correctness]") {
    // If x is real, then X[N-f] == conj(X[f])
    const Eigen::Index n = 128;
    FFT fft({static_cast<int>(n)});

    // real-valued input stored as complex with zero imaginary part
    TensorXcd<1> in(n);
    for (Eigen::Index k = 0; k < n; ++k)
        in(k) = std::complex<double>(std::cos(2.0 * pi * 3 * k / n) + 0.5, 0.0);

    TensorXcd<1> out(n);
    fft.fft(in, out);

    for (Eigen::Index f = 1; f < n / 2; ++f) {
        std::complex<double> diff = out(n - f) - std::conj(out(f));
        REQUIRE_THAT(std::abs(diff), Catch::Matchers::WithinAbs(0.0, kTol * n));
    }
}

// -----------------------------------------------------------------------
// DC component
// -----------------------------------------------------------------------

TEST_CASE("FFT 1D DC bin equals sum of input", "[fft1d][correctness]") {
    // X[0] = sum(x[k])
    const Eigen::Index n = 64;
    FFT fft({static_cast<int>(n)});

    TensorXcd<1> in  = random_tensor(n);
    TensorXcd<1> out(n);
    fft.fft(in, out);

    std::complex<double> dc_expected = as_vec(in).sum();
    REQUIRE_THAT(std::abs(out(0) - dc_expected), Catch::Matchers::WithinAbs(0.0, kTol * n));
}

// -----------------------------------------------------------------------
// Misaligned input via offset raw pointer (exercises execute_safe fallback)
// -----------------------------------------------------------------------

TEST_CASE("FFT 1D handles misaligned input via offset pointer", "[fft1d][alignment]") {
    // A pointer offset by one complex element is likely misaligned for SIMD.
    // execute_safe should detect this and use the copy fallback.
    const Eigen::Index n = 64;
    FFT fft({static_cast<int>(n)});

    // allocate n+1, transform starting at index 1 to force an offset address
    Eigen::VectorXcd padded = Eigen::VectorXcd::Random(n + 1);

    TensorXcd<1> in_contiguous(n);
    for (Eigen::Index k = 0; k < n; ++k) in_contiguous(k) = padded[k + 1];

    TensorXcd<1> out_ref(n), out_mis(n);
    fft.fft(in_contiguous, out_ref);
    // raw-pointer overload with potentially misaligned pointer
    fft.fft(padded.data() + 1, out_mis.data());

    REQUIRE(max_abs_error(out_ref, out_mis) < kTol);
}

// -----------------------------------------------------------------------
// Multiple independent FFT objects at the same size
// -----------------------------------------------------------------------

TEST_CASE("Multiple FFT objects at same 1D size produce identical results", "[fft1d]") {
    const Eigen::Index n = 128;
    FFT fft_a({static_cast<int>(n)});
    FFT fft_b({static_cast<int>(n)}, 1, PlanRigor::Estimate);

    TensorXcd<1> in = random_tensor(n);
    TensorXcd<1> out_a(n), out_b(n);

    fft_a.fft(in, out_a);
    fft_b.fft(in, out_b);

    REQUIRE(max_abs_error(out_a, out_b) < kTol);
}

// -----------------------------------------------------------------------
// Reuse - same plan, multiple executions
// -----------------------------------------------------------------------

TEST_CASE("FFT 1D plan can be reused across multiple calls", "[fft1d]") {
    const Eigen::Index n = 64;
    FFT fft({static_cast<int>(n)});

    for (int iter = 0; iter < 5; ++iter) {
        TensorXcd<1> in  = make_sinusoid(n, iter);
        TensorXcd<1> out(n);
        fft.fft(in, out);

        // spike should be at bin `iter`
        REQUIRE_THAT(std::abs(out(iter)),
            Catch::Matchers::WithinAbs(static_cast<double>(n), kTol * n));
    }
}

// -----------------------------------------------------------------------
// Normalization flag (now per-call on ifft, not on constructor)
// -----------------------------------------------------------------------

TEST_CASE("FFT 1D ifft(normalize=true): ifft(fft(x)) == x without manual scaling", "[fft1d][normalize]") {
    auto n = GENERATE(16, 64, 256);
    INFO("N = " << n);

    FFT fft({n});
    TensorXcd<1> original = random_tensor(n);

    TensorXcd<1> freq_domain(n), recovered(n);
    fft.fft(original, freq_domain);
    fft.ifft(freq_domain, recovered, /*normalize=*/true);

    REQUIRE(max_abs_error(recovered, original) < kTol);
}

TEST_CASE("FFT 1D ifft(normalize=false) default: inverse is unnormalized", "[fft1d][normalize]") {
    // With normalize=false, IFFT(FFT(x)) = N*x — the raw FFTW convention.
    const Eigen::Index n = 64;
    FFT fft({static_cast<int>(n)});

    TensorXcd<1> original = random_tensor(n);
    TensorXcd<1> freq_domain(n), recovered(n);
    fft.fft(original, freq_domain);
    fft.ifft(freq_domain, recovered); // default normalize=false

    // recovered should equal N * original
    Eigen::VectorXcd scaled = static_cast<double>(n) * as_vec(original);
    REQUIRE((as_vec(recovered) - scaled).cwiseAbs().maxCoeff() < kTol);
}

// -----------------------------------------------------------------------
// Wisdom save / load (static on FFT, not per-rank anymore)
// -----------------------------------------------------------------------

namespace {
    // Use a fixed temp path; clean up in the test.
    const std::string kWisdomPath =
        (std::filesystem::temp_directory_path() / "sirius_test_wisdom.fftw").string();
}

TEST_CASE("FFT saveWisdom writes a file and loadWisdom reads it back", "[fft1d][wisdom]") {
    std::remove(kWisdomPath.c_str());

    // Plan and save
    FFT fft({128});
    REQUIRE_NOTHROW(FFT::saveWisdom(kWisdomPath));

    // File must exist after saving
    FILE* f = std::fopen(kWisdomPath.c_str(), "r");
    REQUIRE(f != nullptr);
    if (f) std::fclose(f);

    // Load and verify a subsequent plan still produces correct results
    REQUIRE_NOTHROW(FFT::loadWisdom(kWisdomPath));

    FFT fft2({128});
    TensorXcd<1> in(128); in.setZero();
    in(0) = 1.0;
    TensorXcd<1> out(128);
    fft2.fft(in, out);

    // delta input -> flat spectrum with magnitude 1
    for (Eigen::Index i = 0; i < 128; ++i)
        REQUIRE_THAT(std::abs(out(i)), Catch::Matchers::WithinAbs(1.0, kTol));

    std::remove(kWisdomPath.c_str());
}

TEST_CASE("FFT loadWisdom on missing file does not throw", "[fft1d][wisdom]") {
    const std::string missing =
        (std::filesystem::temp_directory_path() / "sirius_nonexistent_wisdom.fftw").string();
    REQUIRE_NOTHROW(FFT::loadWisdom(missing));
}

TEST_CASE("FFT saveWisdom to invalid path throws", "[fft1d][wisdom]") {
    const std::string bad =
        (std::filesystem::temp_directory_path() / "sirius_nonexistent_subdir" / "wisdom.fftw").string();
    REQUIRE_THROWS_AS(FFT::saveWisdom(bad), std::runtime_error);
}
