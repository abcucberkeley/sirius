// End-to-end SIM reconstruction of the bundled test data: raw.tif (3 dirs x
// 5 phases x 9 z of 64x64) + otf.tif must reproduce raw_proc.tif, the output
// of the cudasirecon reference binary (9 x 128 x 128, float32). Runs on the
// CPU always and on the GPU when a CUDA device is available.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <filesystem>
#include <stdexcept>

#include "sirius/buffer.hpp"
#include "sirius/legacy_config.hpp"
#include "sirius/otf.hpp"
#include "sirius/sim_reconstruction.hpp"
#include "sirius/tiff_io.hpp"

using namespace sirius;
using namespace std::filesystem;

namespace {

    struct TestData {
        SIMParameters params;
        OTFRadiallyAveraged otf;
        Eigen::Tensor<double, 3, Eigen::RowMajor> raw;
        Eigen::Tensor<float, 3, Eigen::RowMajor> expected;
    };

    TestData loadTestData() {
        const path dir = SIRIUS_TEST_DATA_DIR;
        SIMParameters params = fromLegacy(loadLegacyConfig((dir / "config.txt").string()));
        OTFRadiallyAveraged otf = loadOTF((dir / "otf.tif").string(), params);
        return TestData{
            std::move(params),
            std::move(otf),
            readTiffStack<double>((dir / "raw.tif").string()),
            readTiffStack<float>((dir / "raw_proc.tif").string()),
        };
    }

    // max |a-b| over the volume, relative to the expected volume's peak
    template <typename TensorA>
    double maxRelDiff(const TensorA& actual, const Eigen::Tensor<float, 3, Eigen::RowMajor>& expected) {
        REQUIRE(actual.dimension(0) == expected.dimension(0));
        REQUIRE(actual.dimension(1) == expected.dimension(1));
        REQUIRE(actual.dimension(2) == expected.dimension(2));
        double peak = 0.0, diff = 0.0;
        for (Eigen::Index i = 0; i < expected.size(); ++i) {
            peak = std::max(peak, std::abs(static_cast<double>(expected.data()[i])));
            diff = std::max(diff, std::abs(static_cast<double>(actual.data()[i]) -
                                           static_cast<double>(expected.data()[i])));
        }
        REQUIRE(peak > 0.0);
        return diff / peak;
    }

    // Pattern vectors: the reference fit lands at ~0.407 um in all three
    // directions, at the angles of the config.
    void checkK0(const SimFit& fit, const SIMParameters& params) {
        REQUIRE(fit.k0.size() == 3);
        REQUIRE(params.k0_angles.has_value());
        for (int d = 0; d < 3; ++d) {
            const double mag = std::hypot(fit.k0[d][0], fit.k0[d][1]);
            const double spacing = 1.0 / mag;
            INFO("direction " << d << ": spacing " << spacing << " um");
            CHECK(spacing > 0.40);
            CHECK(spacing < 0.415);
            const double angle = std::atan2(fit.k0[d][1], fit.k0[d][0]);
            CHECK(std::abs(angle - (*params.k0_angles)[d]) < 0.05);
        }
    }

    // Modulation amplitudes are measured relative to the OTF, so these
    // bounds hold for the measured OTF the reference used.
    void checkFit(const SimFit& fit, const SIMParameters& params) {
        checkK0(fit, params);
        for (int d = 0; d < 3; ++d) {
            INFO("direction " << d);
            // |amp1| ~ 0.21-0.24, |amp2| ~ 0.72-0.77 for this data
            CHECK(std::abs(fit.amps[d][0]) == 1.0);
            CHECK(std::abs(fit.amps[d][1]) > 0.1);
            CHECK(std::abs(fit.amps[d][1]) < 0.4);
            CHECK(std::abs(fit.amps[d][2]) > 0.6);
            CHECK(std::abs(fit.amps[d][2]) < 0.9);
        }
    }

} // namespace

TEST_CASE("CPU reconstruction reproduces the cudasirecon reference output", "[reconstruction]") {
    TestData t = loadTestData();

    SimReconstructor recon(t.params, t.otf, Device::cpu(), PlanRigor::Estimate);
    Buffer<double> out = recon.reconstruct(t.raw);

    REQUIRE(out.shape() == Shape({9, 128, 128}));
    checkFit(recon.lastFit(), t.params);

    const auto actual = toEigen<3>(out);
    const double rel = maxRelDiff(actual, t.expected);
    INFO("max |actual-expected| / max |expected| = " << rel);
    CHECK(rel < 1e-4);
}

TEST_CASE("Repeated CPU reconstructions of the same input are bit-identical",
          "[reconstruction]") {
    // The k0 bracket search maximizes |modamp|^2, so a reduction whose
    // rounding depends on the OpenMP thread schedule moves the fitted pattern
    // vector and, through it, every output voxel. Reproducibility is required
    // by the Python API contract (recon.reconstruct(x) twice) and is what
    // makes CPU/GPU comparisons meaningful.
    TestData t = loadTestData();

    SimReconstructor recon(t.params, t.otf, Device::cpu(), PlanRigor::Estimate);
    const auto first = toEigen<3>(recon.reconstruct(t.raw));
    const SimFit fit = recon.lastFit();
    const auto second = toEigen<3>(recon.reconstruct(t.raw));

    REQUIRE(first.size() == second.size());
    Eigen::Index differing = 0;
    for (Eigen::Index i = 0; i < first.size(); ++i)
        if (first.data()[i] != second.data()[i]) ++differing;
    INFO(differing << " of " << first.size() << " voxels differ between runs");
    CHECK(differing == 0);

    for (std::size_t d = 0; d < fit.k0.size(); ++d) {
        CHECK(fit.k0[d][0] == recon.lastFit().k0[d][0]);
        CHECK(fit.k0[d][1] == recon.lastFit().k0[d][1]);
    }
}

TEST_CASE("A cancel callback aborts the reconstruction promptly and changes nothing otherwise",
          "[reconstruction][cancel]") {
    // The contract of SimReconstructor::setCancelCallback: a predicate that
    // never fires must not perturb a single output bit (the reconstruction is
    // bit-reproducible -- see the determinism case above -- so "unchanged" is
    // checkable exactly), and one that fires must end the call by throwing
    // instead of running the pipeline out.
    TestData t = loadTestData();

    SimReconstructor reference(t.params, t.otf, Device::cpu(), PlanRigor::Estimate);
    const auto expected = toEigen<3>(reference.reconstruct(t.raw));
    const SimFit expectedFit = reference.lastFit();

    SECTION("a callback that always returns false is bit-identical to no callback") {
        SimReconstructor recon(t.params, t.otf, Device::cpu(), PlanRigor::Estimate);
        int polls = 0;
        recon.setCancelCallback([&polls] { ++polls; return false; });
        const auto actual = toEigen<3>(recon.reconstruct(t.raw));

        REQUIRE(actual.size() == expected.size());
        Eigen::Index differing = 0;
        for (Eigen::Index i = 0; i < expected.size(); ++i)
            if (actual.data()[i] != expected.data()[i]) ++differing;
        INFO(differing << " of " << expected.size() << " voxels differ; " << polls << " polls");
        CHECK(differing == 0);
        CHECK(polls > 0);   // the stages really are polling
        for (std::size_t d = 0; d < expectedFit.k0.size(); ++d) {
            CHECK(recon.lastFit().k0[d][0] == expectedFit.k0[d][0]);
            CHECK(recon.lastFit().k0[d][1] == expectedFit.k0[d][1]);
        }
    }

    SECTION("cancelling after the first stage throws without finishing the pipeline") {
        SimReconstructor recon(t.params, t.otf, Device::cpu(), PlanRigor::Estimate);
        int polls = 0;
        recon.setCancelCallback([&polls] { return ++polls > 1; });
        CHECK_THROWS_WITH(recon.reconstruct(t.raw), Catch::Matchers::Equals("cancelled"));
        // It stopped at the second boundary, nowhere near the ~200 polls a
        // whole reconstruction of this stack makes.
        INFO(polls << " polls before the throw");
        CHECK(polls == 2);
    }

    SECTION("cancelling during the fit and during assembly both throw") {
        // 3 lands in the per-direction band separation, 6 in findK0's
        // overlaps and 12 inside the k0 bracket search, so every kind of
        // stage boundary the pipeline polls at is exercised.
        const int after = GENERATE(3, 6, 12);
        SimReconstructor recon(t.params, t.otf, Device::cpu(), PlanRigor::Estimate);
        int polls = 0;
        recon.setCancelCallback([&polls, after] { return ++polls > after; });
        INFO("cancel after " << after << " polls");
        CHECK_THROWS_AS(recon.reconstruct(t.raw), std::runtime_error);
    }

    SECTION("the reconstructor is reusable after a cancelled call") {
        SimReconstructor recon(t.params, t.otf, Device::cpu(), PlanRigor::Estimate);
        recon.setCancelCallback([] { return true; });
        CHECK_THROWS_WITH(recon.reconstruct(t.raw), Catch::Matchers::Equals("cancelled"));
        recon.setCancelCallback({});
        const auto actual = toEigen<3>(recon.reconstruct(t.raw));
        Eigen::Index differing = 0;
        for (Eigen::Index i = 0; i < expected.size(); ++i)
            if (actual.data()[i] != expected.data()[i]) ++differing;
        INFO(differing << " voxels differ from a run that was never cancelled");
        CHECK(differing == 0);
    }
}

TEST_CASE("GPU reconstruction reproduces the cudasirecon reference output",
          "[reconstruction][cuda]") {
    if (!cudaAvailable()) SKIP("no CUDA device available");
    const Device gpu = Device::cuda(0);
    TestData t = loadTestData();

    SimReconstructor recon(t.params, t.otf, gpu, PlanRigor::Estimate);
    Stream stream(gpu);
    Buffer<double> dRaw = toDevice(t.raw, gpu, stream);
    stream.synchronize();

    Buffer<double> dOut = recon.reconstruct(dRaw);
    REQUIRE(dOut.device() == gpu);
    REQUIRE(dOut.shape() == Shape({9, 128, 128}));
    checkFit(recon.lastFit(), t.params);

    const auto actual = toEigen<3>(dOut);
    const double rel = maxRelDiff(actual, t.expected);
    INFO("max |actual-expected| / max |expected| = " << rel);
    CHECK(rel < 1e-4);
}

TEST_CASE("CPU and GPU reconstructions agree closely", "[reconstruction][cuda]") {
    if (!cudaAvailable()) SKIP("no CUDA device available");
    const Device gpu = Device::cuda(0);
    TestData t = loadTestData();

    SimReconstructor cpuRecon(t.params, t.otf, Device::cpu(), PlanRigor::Estimate);
    const auto cpuOut = toEigen<3>(cpuRecon.reconstruct(t.raw));

    SimReconstructor gpuRecon(t.params, t.otf, gpu, PlanRigor::Estimate);
    Stream stream(gpu);
    Buffer<double> dRaw = toDevice(t.raw, gpu, stream);
    stream.synchronize();
    const auto gpuOut = toEigen<3>(gpuRecon.reconstruct(dRaw));

    double peak = 0.0, diff = 0.0;
    for (Eigen::Index i = 0; i < cpuOut.size(); ++i) {
        peak = std::max(peak, std::abs(cpuOut.data()[i]));
        diff = std::max(diff, std::abs(cpuOut.data()[i] - gpuOut.data()[i]));
    }
    INFO("max |cpu-gpu| / max |cpu| = " << diff / peak);
    CHECK(diff / peak < 1e-6);
}

// --- diagnostics and the ideal OTF ------------------------------------------

TEST_CASE("Diagnostics capture the separated and filtered band spectra", "[reconstruction][diagnostics]") {
    TestData t = loadTestData();
    SimReconstructor recon(t.params, t.otf, Device::cpu(), PlanRigor::Estimate);

    // off by default: nothing captured, and the result is unaffected
    const auto plain = toEigen<3>(recon.reconstruct(t.raw));
    CHECK_FALSE(recon.lastDiagnostics().captured);
    CHECK(recon.lastDiagnostics().separated.empty());

    recon.setCaptureDiagnostics(true);
    const auto withDiag = toEigen<3>(recon.reconstruct(t.raw));
    for (Eigen::Index i = 0; i < plain.size(); ++i) REQUIRE(withDiag.data()[i] == plain.data()[i]);

    const SimDiagnostics& d = recon.lastDiagnostics();
    REQUIRE(d.captured);
    CHECK(d.ndirs == 3);
    CHECK(d.nbands == 5);
    CHECK(d.nx == 64);
    CHECK(d.ny == 64);
    CHECK(d.nz == 9);
    CHECK(d.dkx > 0.0);
    CHECK(d.rdistcutoff > 0.0);
    const Shape expected{3 * 5 * 9, 64, 33};
    REQUIRE(d.separated.shape() == expected);
    REQUIRE(d.filtered.shape() == expected);
    REQUIRE(d.separated.device().isCpu());

    // Band 0 of direction 0 at DC is the (scaled) sum of the frames: real and
    // positive. The filter changes the bands, so the two captures differ.
    const std::complex<double> dc = d.separated.data()[0];
    CHECK(dc.real() > 0.0);
    CHECK(std::abs(dc.imag()) < 1e-9 * dc.real());
    bool differ = false;
    for (Index i = 0; i < d.separated.size() && !differ; ++i)
        differ = d.separated.data()[i] != d.filtered.data()[i];
    CHECK(differ);
    for (Index i = 0; i < d.filtered.size(); ++i)
        REQUIRE(std::isfinite(std::abs(d.filtered.data()[i])));

    SimDiagnostics taken = recon.takeDiagnostics();
    CHECK(taken.captured);
    CHECK_FALSE(recon.lastDiagnostics().captured);
    CHECK(recon.lastDiagnostics().separated.empty());
}

TEST_CASE("Reconstruction with the ideal OTF resembles the reference", "[reconstruction][ideal]") {
    TestData t = loadTestData();
    const OTFRadiallyAveraged ideal = idealOTF(t.params, /*threeD=*/true);
    REQUIRE(ideal.data().dimension(0) >= 3);

    SimReconstructor recon(t.params, ideal, Device::cpu(), PlanRigor::Estimate);
    const auto out = toEigen<3>(recon.reconstruct(t.raw));
    REQUIRE(out.dimension(0) == 9);
    REQUIRE(out.dimension(1) == 128);
    REQUIRE(out.dimension(2) == 128);
    // the pattern vectors do not depend on the OTF's fine shape; the amplitudes do
    checkK0(recon.lastFit(), t.params);
    for (const auto& amps : recon.lastFit().amps) {
        REQUIRE(amps.size() == 3);
        CHECK(std::abs(amps[0]) == 1.0);
        CHECK(std::abs(amps[1]) > 0.0);
        CHECK(std::abs(amps[2]) > 0.0);
    }

    // Pearson correlation with the measured-OTF reference: same object, so
    // the two reconstructions must agree strongly even though the ideal OTF
    // ignores aberrations of the real system.
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    const double n = static_cast<double>(out.size());
    for (Eigen::Index i = 0; i < out.size(); ++i) {
        const double a = out.data()[i], b = t.expected.data()[i];
        REQUIRE(std::isfinite(a));
        sa += a;
        sb += b;
        saa += a * a;
        sbb += b * b;
        sab += a * b;
    }
    const double corr = (n * sab - sa * sb) / std::sqrt((n * saa - sa * sa) * (n * sbb - sb * sb));
    INFO("correlation with the reference reconstruction: " << corr);
    CHECK(corr > 0.8);
}
