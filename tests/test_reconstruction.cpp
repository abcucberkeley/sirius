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
#include "sirius/errors.hpp"
#include "sirius/legacy_config.hpp"
#include "sirius/otf.hpp"
#include "sirius/sim_reconstruction.hpp"
#include "sirius/tiff_io.hpp"

#include "sim_synthetic.hpp"

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

TEST_CASE("The reconstructor refuses an order count it cannot fit", "[reconstruction][orders]") {
    // One order used to reach the k0 fit (a division by order 0, then a read
    // through the absent sine band: a segfault); one phase divided the 3D
    // pattern guess by zero. Both are now parameter errors.
    TestData t = loadTestData();
    SIMParameters p = t.params;
    p.norders = 1;
    CHECK_THROWS_AS(SimReconstructor(p, t.otf, Device::cpu(), PlanRigor::Estimate), std::runtime_error);
    p.norders = 0;
    p.nphases = 1;
    CHECK_THROWS_AS(SimReconstructor(p, t.otf, Device::cpu(), PlanRigor::Estimate), std::runtime_error);
}

TEST_CASE("The reconstructor refuses optics and zooms that used to crash it", "[reconstruction][validation]") {
    TestData t = loadTestData();
    auto construct = [&](const SIMParameters& p) { return SimReconstructor(p, t.otf, Device::cpu(), PlanRigor::Estimate); };

    SECTION("an NA above the immersion index, or no immersion index, with a measured OTF") {
        // asin(na / nimm) and wavelength / nimm went NaN, the axial cutoff
        // INT_MIN, and the filter zeroed planes far outside the bands (SIGSEGV)
        SIMParameters p = t.params;
        p.na = 1.6;
        CHECK_THROWS_AS(construct(p), std::runtime_error);
        p = t.params;
        p.nimm = 0.0;
        CHECK_THROWS_AS(construct(p), std::runtime_error);
        p.nimm = -1.515;
        CHECK_THROWS_AS(construct(p), std::runtime_error);
    }
    SECTION("an NA equal to the immersion index still reconstructs") {
        SIMParameters p = t.params;
        p.na = p.nimm;
        SimReconstructor recon = construct(p);
        const Buffer<double> out = recon.reconstruct(t.raw);
        Index bad = 0;
        for (Index i = 0; i < out.size(); ++i) bad += std::isfinite(out.data()[i]) ? 0 : 1;
        CHECK(bad == 0);
    }
    SECTION("a zoom below 1") {
        // the output grid was smaller than the frequencies written into it
        SIMParameters p = t.params;
        p.zoomfact = 0.5;
        CHECK_THROWS_AS(construct(p), std::runtime_error);
    }
    SECTION("a pixel size whose frequency step overflows") {
        // valid on its own, but nz * dz is infinite: the axial cutoff is not a
        // number, and is reported instead of being cast to a plane index
        SIMParameters p = t.params;
        p.dz = 1e308;
        SimReconstructor recon = construct(p);
        // twice: the failed first call must not leave the shape bound to
        // buffers it never finished allocating (the retry was a double free)
        for (int attempt = 0; attempt < 2; ++attempt) {
            INFO("attempt " << attempt);
            try {
                recon.reconstruct(t.raw);
                FAIL("reconstructed with an infinite axial cutoff");
            } catch (const std::invalid_argument& e) {
                CHECK_THAT(e.what(), Catch::Matchers::ContainsSubstring("not finite"));
            }
        }
    }
    SECTION("a stack without sections") {
        SimReconstructor recon = construct(t.params);
        const Buffer<double> empty(Shape{0, 64, 64});
        CHECK_THROWS_AS(recon.reconstruct(empty.view()), std::invalid_argument);
    }
}

TEST_CASE("A shape that fails to bind is rebuilt on the next call, not reused half-built",
          "[reconstruction][rebind]") {
    // bindShape stored the new shape before rebuilding the plans and buffers.
    // When a later step threw (out of memory, a failed plan) the retry with the
    // same shape took the "already bound" early return and ran on the previous
    // shape's buffers: a heap-use-after-free under ASan. Here the plan for the
    // second shape throws deterministically (its FFT size overflows int), so
    // no real allocation failure is needed; its data is never read.
    TestData t = loadTestData();
    SimReconstructor recon(t.params, t.otf, Device::cpu(), PlanRigor::Estimate);
    const auto first = toEigen<3>(recon.reconstruct(t.raw));

    const std::vector<double> tiny(64);
    const BufferView<const double> huge(tiny.data(), Shape{135, 65536, 65536}, Device::cpu());
    CHECK_THROWS(recon.reconstruct(huge));
    CHECK_THROWS(recon.reconstruct(huge));   // was: no rebuild, reads of the tiny buffer as 65536 x 65536

    // and the reconstructor still works for the shape it had before
    const auto again = toEigen<3>(recon.reconstruct(t.raw));
    REQUIRE(again.size() == first.size());
    Eigen::Index differing = 0;
    for (Eigen::Index i = 0; i < first.size(); ++i)
        if (again.data()[i] != first.data()[i]) ++differing;
    CHECK(differing == 0);
}

TEST_CASE("An overlap with nothing in it is reported instead of fitted into NaN", "[reconstruction][overlap]") {
    // The modulation amplitude divided by the overlap's energy unguarded:
    // each of these produced a 100 % NaN volume without an error.
    TestData t = loadTestData();
    auto requireEmptyOverlap = [&](const SIMParameters& p, const Eigen::Tensor<double, 3, Eigen::RowMajor>& raw,
                                   Device device) {
        SimReconstructor recon(p, t.otf, device, PlanRigor::Estimate);
        Buffer<double> input = toDevice(raw, device);
        synchronizeDevice(device);
        try {
            recon.reconstruct(input.view());
            FAIL("reconstructed from an empty overlap");
        } catch (const SiriusError& e) {
            CHECK_THAT(e.what(), Catch::Matchers::ContainsSubstring("holds no signal"));
        }
    };
    const Device gpu = cudaAvailable() ? Device::cuda(0) : Device::cpu();

    SECTION("a constant stack") {
        Eigen::Tensor<double, 3, Eigen::RowMajor> constant = t.raw;
        constant.setConstant(100.0);
        requireEmptyOverlap(t.params, constant, Device::cpu());
    }
    SECTION("a line spacing that puts the side band outside the OTF") {
        SIMParameters p = t.params;
        p.linespacing_um = 0.05;
        requireEmptyOverlap(p, t.raw, Device::cpu());
        if (gpu.isCuda()) requireEmptyOverlap(p, t.raw, gpu);
    }
    SECTION("an otfcutoff nothing clears") {
        SIMParameters p = t.params;
        p.otfcutoff = 1.0;
        requireEmptyOverlap(p, t.raw, Device::cpu());
    }
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

// --- 2D and thin stacks ---------------------------------------------------------

namespace {

    // 2D-SIM optics of the synthetic scene: 3 directions x 3 phases and a
    // 0.30 um pattern inside the 1.2 NA passband. Everything else is a
    // library default -- no_kz0 on, the order count derived.
    SIMParameters params2d() {
        SIMParameters p;
        p.ndirs = 3;
        p.nphases = 3;
        p.na = 1.2;
        p.nimm = 1.33;
        p.wavelength_nm = 530.0;
        p.linespacing_um = 0.30;
        p.k0_start_angle = 0.3;
        p.dx = 0.08;
        p.dy = 0.08;
        return p;
    }

    Index nonFinite(const Buffer<double>& v) {
        Index bad = 0;
        for (Index i = 0; i < v.size(); ++i) bad += std::isfinite(v.data()[i]) ? 0 : 1;
        return bad;
    }

    // The simulated pattern, to a fraction of a percent. The data do not fix
    // the sign of k0, so the angle is compared modulo pi.
    void checkPattern2d(const SimFit& fit, const SIMParameters& p) {
        REQUIRE(fit.k0.size() == 3);
        for (int d = 0; d < 3; ++d) {
            const double mag = std::hypot(fit.k0[d][0], fit.k0[d][1]);
            const double angle = std::atan2(fit.k0[d][1], fit.k0[d][0]);
            const double off = std::remainder(angle - (p.k0_start_angle + d * kPi / 3.0), kPi);
            REQUIRE(fit.amps[d].size() == 2);
            INFO("direction " << d << ": spacing " << 1.0 / mag << " um, angle off by " << off << " rad, |amp1| "
                              << std::abs(fit.amps[d][1]));
            CHECK(std::abs(1.0 / mag - p.linespacing_um) < 0.005 * p.linespacing_um);
            CHECK(std::abs(off) < 0.01);
            // a real modulation was measured (the unmodulated background keeps
            // it well below the simulated 0.8 in the reference's convention)
            CHECK(std::abs(fit.amps[d][1]) > 0.05);
            CHECK(std::abs(fit.amps[d][1]) < 1.0);
        }
    }

    // nz planes from the middle of the 9-plane test stack
    Eigen::Tensor<double, 3, Eigen::RowMajor> thinStack(const TestData& t, int nz) {
        Eigen::Tensor<double, 3, Eigen::RowMajor> sub(15 * nz, 64, 64);
        const int z0 = (9 - nz) / 2;
        for (int d = 0; d < 3; ++d)
            for (int z = 0; z < nz; ++z)
                for (int ph = 0; ph < 5; ++ph)
                    for (int y = 0; y < 64; ++y)
                        for (int x = 0; x < 64; ++x)
                            sub((d * nz + z) * 5 + ph, y, x) = t.raw((d * 9 + z0 + z) * 5 + ph, y, x);
        return sub;
    }

} // namespace

TEST_CASE("A 2D stack reconstructs at the default settings", "[reconstruction][2d]") {
    // no_kz0 (on by default) used to skip the only kz plane a 2D stack has:
    // every overlap came out empty, the modulation amplitudes 0/0 and every
    // output voxel NaN. The order-0 damping the application turns on divided
    // by its zero axial limit and did the same.
    SIMParameters p = params2d();
    REQUIRE(p.no_kz0);
    REQUIRE(p.resolvedOrders() == 2);
    const Buffer<double> raw = test::syntheticSim2d(p, 128);
    const OTFRadiallyAveraged otf = idealOTF(p, /*threeD=*/false);

    SECTION("library defaults") {}
    SECTION("the order-0 damping on") { p.dampen_order0 = true; }
    SECTION("no_kz0 off gives the same pattern") { p.no_kz0 = false; }

    SimReconstructor recon(p, otf, Device::cpu(), PlanRigor::Estimate);
    const Buffer<double> out = recon.reconstruct(raw.view());
    REQUIRE(out.shape() == Shape({1, 256, 256}));
    CHECK(nonFinite(out) == 0);
    checkPattern2d(recon.lastFit(), p);
}

TEST_CASE("Thin z stacks reconstruct at the settings of the test data", "[reconstruction][thin]") {
    // The config has no_kz0 (default) and dampenOrder0 on. With 2 or 4 planes
    // kz = +-1 lies outside the OTF's axial support, with 3 the filter keeps
    // kz = 0 alone, and with 5 kz = +-1 is inside the support but holds no
    // overlap sample above otfcutoff: all four used to come out 100 % NaN.
    TestData t = loadTestData();
    const int nz = GENERATE(2, 3, 4, 5);
    INFO(nz << " planes");
    const auto raw = thinStack(t, nz);
    SimReconstructor recon(t.params, t.otf, Device::cpu(), PlanRigor::Estimate);
    const Buffer<double> out = recon.reconstruct(raw);
    REQUIRE(out.shape() == Shape({nz, 128, 128}));
    CHECK(nonFinite(out) == 0);
    checkK0(recon.lastFit(), t.params);
}

TEST_CASE("CPU and GPU agree on 2D and thin stacks", "[reconstruction][cuda][2d][thin]") {
    if (!cudaAvailable()) SKIP("no CUDA device available");
    const Device gpu = Device::cuda(0);
    auto compare = [&](const SIMParameters& p, const OTFRadiallyAveraged& otf, BufferView<const double> raw) {
        SimReconstructor cpuRecon(p, otf, Device::cpu(), PlanRigor::Estimate);
        const Buffer<double> cpuOut = cpuRecon.reconstruct(raw);
        SimReconstructor gpuRecon(p, otf, gpu, PlanRigor::Estimate);
        Stream stream(gpu);
        const Buffer<double> dRaw = toDevice(raw, gpu, stream);
        stream.synchronize();
        const Buffer<double> gpuOut = gpuRecon.reconstruct(dRaw.view()).to(Device::cpu());
        REQUIRE(gpuOut.shape() == cpuOut.shape());
        CHECK(nonFinite(cpuOut) == 0);
        double peak = 0.0, diff = 0.0;
        for (Index i = 0; i < cpuOut.size(); ++i) {
            peak = std::max(peak, std::abs(cpuOut.data()[i]));
            diff = std::max(diff, std::abs(cpuOut.data()[i] - gpuOut.data()[i]));
        }
        REQUIRE(peak > 0.0);
        INFO("max |cpu-gpu| / max |cpu| = " << diff / peak);
        CHECK(diff / peak < 1e-6);
    };

    SECTION("2D, order-0 damping on") {
        SIMParameters p = params2d();
        p.dampen_order0 = true;
        const Buffer<double> raw = test::syntheticSim2d(p, 128);
        compare(p, idealOTF(p, /*threeD=*/false), raw.view());
    }
    SECTION("3 planes of the test data") {
        TestData t = loadTestData();
        const Eigen::Tensor<double, 3, Eigen::RowMajor> raw = thinStack(t, 3);
        compare(t.params, t.otf, BufferView<const double>(raw.data(), Shape(raw), Device::cpu()));
    }
}
