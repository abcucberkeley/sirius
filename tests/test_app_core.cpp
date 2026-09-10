// Tests of the GUI's Qt-free model (app/core): display mapping, parameter
// format detection, fit summary and the ReconSession, including an
// end-to-end reconstruction of the bundled test data through the session
// and the reconstructor/upload caching it promises.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <complex>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "core/display_mapping.hpp"
#include "core/array.hpp"
#include "core/session.hpp"
#include "core/volume_ops.hpp"

#include "sirius/constants.hpp"
#include "sirius/fft.hpp"
#include "sirius/legacy_config.hpp"
#include "sirius/otf.hpp"
#include "sirius/real_fft.hpp"
#include "sirius/tiff_io.hpp"

#include "temp_path.hpp"

using namespace sirius;
using namespace sirius::app;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

    const std::filesystem::path kData = SIRIUS_TEST_DATA_DIR;

    // Writes `text` to a unique temp file that is removed on destruction.
    struct TempFile : test::TempFile {
        TempFile(const char* suffix, const std::string& text) : test::TempFile("app", suffix) {
            std::ofstream(path) << text;
        }
    };

    SIMParameters testParameters() {
        return fromLegacy(loadLegacyConfig((kData / "config.txt").string()));
    }

} // namespace

// --- display mapping -----------------------------------------------------

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

// --- parameter files -----------------------------------------------------

TEST_CASE("detectParameterFormat distinguishes TOML from legacy configs", "[app][params]") {
    const TempFile toml(".toml", "ndirs = 3\n");
    const TempFile tomlNoExt(".cfg", "# comment\n\n[geometry]\nndirs = 3\n");
    const TempFile legacy(".txt", "# comment\nndirs=3\nnphases=5\n");
    const TempFile empty(".cfg", "");

    CHECK(detectParameterFormat(toml.path.string()) == ParameterFormat::Toml);
    CHECK(detectParameterFormat(tomlNoExt.path.string()) == ParameterFormat::Toml);
    CHECK(detectParameterFormat(legacy.path.string()) == ParameterFormat::Legacy);
    CHECK(detectParameterFormat(empty.path.string()) == ParameterFormat::Legacy);
    CHECK_THROWS(detectParameterFormat((kData / "does_not_exist.cfg").string()));
}

TEST_CASE("loadParametersAuto reads the bundled legacy config and a TOML round trip", "[app][params]") {
    ParameterFormat format{};
    const SIMParameters legacy = loadParametersAuto((kData / "config.txt").string(), &format);
    CHECK(format == ParameterFormat::Legacy);
    CHECK(legacy.ndirs == 3);
    CHECK(legacy.nphases == 5);

    const test::TempFile toml("app_roundtrip", ".toml");
    saveParameters(toml.str, legacy);
    const SIMParameters again = loadParametersAuto(toml.str, &format);
    CHECK(format == ParameterFormat::Toml);
    CHECK(again.ndirs == legacy.ndirs);
    CHECK(again.nphases == legacy.nphases);
    CHECK_THAT(again.wiener, WithinRel(legacy.wiener, 1e-12));
    CHECK_THAT(again.linespacing_um, WithinRel(legacy.linespacing_um, 1e-12));
}

// --- fit summary ---------------------------------------------------------

TEST_CASE("summarizeFit converts k0 to spacing and angle", "[app][fit]") {
    SimFit fit;
    fit.k0 = {{2.0, 0.0}, {0.0, -4.0}, {0.0, 0.0}};
    fit.amps = {{{1.0, 0.0}, {0.0, 0.5}}, {{1.0, 0.0}, {-0.25, 0.0}}};   // third direction has no amps

    const std::vector<FitRow> rows = summarizeFit(fit);
    REQUIRE(rows.size() == 3);
    CHECK(rows[0].direction == 0);
    CHECK_THAT(rows[0].spacingUm, WithinRel(0.5, 1e-12));
    CHECK_THAT(rows[0].angleDeg, WithinAbs(0.0, 1e-12));
    REQUIRE(rows[0].ampMagnitude.size() == 2);
    CHECK_THAT(rows[0].ampMagnitude[1], WithinRel(0.5, 1e-12));

    CHECK_THAT(rows[1].spacingUm, WithinRel(0.25, 1e-12));
    CHECK_THAT(rows[1].angleDeg, WithinAbs(-90.0, 1e-12));
    CHECK_THAT(rows[1].ampMagnitude[1], WithinRel(0.25, 1e-12));

    CHECK(rows[2].spacingUm == 0.0);   // zero vector: no division by zero
    CHECK(rows[2].ampMagnitude.empty());
}

// --- session -------------------------------------------------------------

TEST_CASE("ReconSession validates its inputs before reconstructing", "[app][session]") {
    ReconSession s;
    CHECK_FALSE(s.hasRaw());
    CHECK(s.inferredNz() == 0);
    CHECK_FALSE(s.validate().empty());
    CHECK_THROWS(s.reconstruct(Device::cpu(), PlanRigor::Estimate));

    SECTION("rejects non-stack and device buffers") {
        CHECK_THROWS_AS(s.setRaw(Buffer<double>(Shape{8, 8})), std::invalid_argument);
    }

    s.setRaw(Buffer<double>(Shape{15, 8, 8}), "synthetic");
    CHECK(s.hasRaw());
    CHECK(s.rawPath() == "synthetic");
    CHECK(s.inferredNz() == 1);   // 15 sections / (3 dirs * 5 phases)
    CHECK(s.validate().empty());  // no OTF file needed: the ideal OTF stands in
    CHECK(s.usesIdealOtf());

    s.setOtfPath((kData / "otf.tif").string());
    CHECK_FALSE(s.usesIdealOtf());
    CHECK(s.validate().empty());

    SECTION("section count must match ndirs * nphases") {
        s.setRaw(Buffer<double>(Shape{14, 8, 8}));
        CHECK(s.inferredNz() == 0);
        CHECK_FALSE(s.validate().empty());
    }
    SECTION("odd image sizes are rejected") {
        s.setRaw(Buffer<double>(Shape{15, 7, 8}));
        CHECK_FALSE(s.validate().empty());
    }
    SECTION("invalid parameters are reported") {
        SIMParameters p = s.parameters();
        p.nphases = 0;
        s.setParameters(p);
        CHECK(s.validate().rfind("Invalid parameters", 0) == 0);
    }
}

TEST_CASE("ReconSession reconstructs the test data and reuses its plans", "[app][session][reconstruction]") {
    ReconSession s;
    s.setParameters(testParameters());
    s.loadRaw((kData / "raw.tif").string());
    s.setOtfPath((kData / "otf.tif").string());
    REQUIRE(s.raw().shape() == Shape({135, 64, 64}));
    REQUIRE(s.inferredNz() == 9);
    REQUIRE(s.validate().empty());

    ReconResult first = s.reconstruct(Device::cpu(), PlanRigor::Estimate);
    REQUIRE(first.volume.shape() == Shape({9, 128, 128}));
    CHECK(first.device == Device::cpu());
    CHECK_FALSE(first.plansReused);
    CHECK(first.seconds > 0.0);
    REQUIRE(first.fit.k0.size() == 3);

    // same output as the reference, as in test_reconstruction
    const auto expected = readTiffStack<float>((kData / "raw_proc.tif").string());
    double peak = 0.0, diff = 0.0;
    for (Index i = 0; i < first.volume.size(); ++i) {
        peak = std::max(peak, std::abs(static_cast<double>(expected.data()[i])));
        diff = std::max(diff, std::abs(first.volume.data()[i] - static_cast<double>(expected.data()[i])));
    }
    CHECK(diff / peak < 1e-4);

    SECTION("a second run with the same setup reuses the reconstructor") {
        ReconResult second = s.reconstruct(Device::cpu(), PlanRigor::Estimate);
        CHECK(second.plansReused);
        CHECK(second.volume.shape() == first.volume.shape());
        for (Index i = 0; i < first.volume.size(); i += 97)
            CHECK(second.volume.data()[i] == first.volume.data()[i]);
    }
    SECTION("changing the parameters rebuilds it") {
        SIMParameters p = s.parameters();
        p.wiener *= 2.0;
        s.setParameters(p);
        ReconResult second = s.reconstruct(Device::cpu(), PlanRigor::Estimate);
        CHECK_FALSE(second.plansReused);
    }
    SECTION("changing the rigor rebuilds it") {
        ReconResult second = s.reconstruct(Device::cpu(), PlanRigor::Measure);
        CHECK_FALSE(second.plansReused);
    }
    SECTION("re-setting the same OTF path does not invalidate") {
        s.setOtfPath((kData / "otf.tif").string());
        CHECK(s.reconstruct(Device::cpu(), PlanRigor::Estimate).plansReused);
    }
}

TEST_CASE("ReconSession reconstructs on the GPU and returns a host volume", "[app][session][cuda]") {
    if (!cudaAvailable()) SKIP("no CUDA device available");
    ReconSession s;
    s.setParameters(testParameters());
    s.loadRaw((kData / "raw.tif").string());
    s.setOtfPath((kData / "otf.tif").string());

    const ReconResult gpu = s.reconstruct(Device::cuda(0), PlanRigor::Estimate);
    REQUIRE(gpu.volume.device().isCpu());
    REQUIRE(gpu.volume.shape() == Shape({9, 128, 128}));
    CHECK(gpu.device == Device::cuda(0));

    // second run: plans and the uploaded raw stack are reused
    CHECK(s.reconstruct(Device::cuda(0), PlanRigor::Estimate).plansReused);

    const ReconResult cpu = s.reconstruct(Device::cpu(), PlanRigor::Estimate);
    double peak = 0.0, diff = 0.0;
    for (Index i = 0; i < cpu.volume.size(); ++i) {
        peak = std::max(peak, std::abs(cpu.volume.data()[i]));
        diff = std::max(diff, std::abs(cpu.volume.data()[i] - gpu.volume.data()[i]));
    }
    CHECK(diff / peak < 1e-6);
}

// --- volume helpers ------------------------------------------------------

namespace {
    Buffer<double> rampVolume(Index nz, Index ny, Index nx) {
        Buffer<double> v(Shape{nz, ny, nx});
        for (Index z = 0; z < nz; ++z)
            for (Index y = 0; y < ny; ++y)
                for (Index x = 0; x < nx; ++x)
                    v.data()[(z * ny + y) * nx + x] = static_cast<double>(z * 10000 + y * 100 + x);
        return v;
    }
} // namespace

TEST_CASE("cropVolume copies the box and rejects bad boxes", "[app][volume]") {
    const Buffer<double> v = rampVolume(3, 5, 7);
    const Buffer<double> c = cropVolume(v.view(), 1, 3, 2, 4, 3, 7);
    REQUIRE(c.shape() == Shape{2, 2, 4});
    CHECK(c.data()[0] == 1 * 10000 + 2 * 100 + 3);
    CHECK(c.data()[c.size() - 1] == 2 * 10000 + 3 * 100 + 6);
    REQUIRE_THROWS_AS(cropVolume(v.view(), 0, 3, 0, 5, 0, 8), std::out_of_range);
    REQUIRE_THROWS_AS(cropVolume(v.view(), 0, 3, 2, 2, 0, 7), std::out_of_range);
    REQUIRE_THROWS_AS(cropVolume(v.view(), -1, 3, 0, 5, 0, 7), std::out_of_range);
}

TEST_CASE("sliceXZ and sliceYZ re-slice in display layout", "[app][volume]") {
    const Buffer<double> v = rampVolume(3, 5, 7);
    std::vector<double> xz(3 * 7), yz(5 * 3);
    sliceXZ(v.view(), 4, xz.data());                 // (nz, nx) at y = 4
    CHECK(xz[0] == 0 * 10000 + 4 * 100 + 0);
    CHECK(xz[2 * 7 + 6] == 2 * 10000 + 4 * 100 + 6);
    sliceYZ(v.view(), 3, yz.data());                 // (ny, nz) at x = 3
    CHECK(yz[0] == 0 * 10000 + 0 * 100 + 3);
    CHECK(yz[4 * 3 + 2] == 2 * 10000 + 4 * 100 + 3);
    REQUIRE_THROWS_AS(sliceXZ(v.view(), 5, xz.data()), std::out_of_range);
    REQUIRE_THROWS_AS(sliceYZ(v.view(), 7, yz.data()), std::out_of_range);
}

TEST_CASE("PlaneSpectrum centers the zero frequency", "[app][volume][spectrum]") {
    PlaneSpectrum spec;
    // constant plane: all energy at DC, which must land at (rows/2, cols/2)
    std::vector<double> plane(6 * 8, 2.0), out(6 * 8);
    spec.magnitude(plane.data(), 6, 8, out.data());
    for (Index r = 0; r < 6; ++r)
        for (Index c = 0; c < 8; ++c) {
            const double expected = (r == 3 && c == 4) ? 2.0 * 48 : 0.0;
            CHECK_THAT(out[static_cast<std::size_t>(r * 8 + c)], WithinAbs(expected, 1e-9));
        }
    // a delta: flat spectrum; the plan is reused for the same size and rebuilt for another
    std::fill(plane.begin(), plane.end(), 0.0);
    plane[0] = 1.0;
    spec.magnitude(plane.data(), 6, 8, out.data());
    for (double v : out) CHECK_THAT(v, WithinAbs(1.0, 1e-12));
    std::vector<double> small(4 * 4, 1.0), smallOut(16);
    spec.magnitude(small.data(), 4, 4, smallOut.data());
    CHECK_THAT(smallOut[2 * 4 + 2], WithinAbs(16.0, 1e-9));
}

TEST_CASE("bandPlaneMagnitude expands a half spectrum like a full FFT", "[app][volume][spectrum]") {
    // a real (nz, ny, nx) volume: its r2c half spectrum, expanded with
    // conjugate symmetry, must match the centered magnitude of the full FFT
    const Index nz = 3, ny = 4, nx = 6, nxh = nx / 2 + 1;
    std::vector<double> vol(static_cast<std::size_t>(nz * ny * nx));
    std::mt19937 gen(3);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (double& v : vol) v = d(gen);

    RealFFT rfft({static_cast<int>(nz), static_cast<int>(ny), static_cast<int>(nx)}, 1, PlanRigor::Estimate);
    std::vector<std::complex<double>> half(static_cast<std::size_t>(nz * ny * nxh));
    rfft.rfft(vol.data(), half.data());

    FFT fft({static_cast<int>(nz), static_cast<int>(ny), static_cast<int>(nx)}, 1, PlanRigor::Estimate);
    std::vector<std::complex<double>> in(vol.begin(), vol.end()), full(in.size());
    fft.fft(in.data(), full.data());

    std::vector<double> plane(static_cast<std::size_t>(ny * nx));
    for (Index z = 0; z < nz; ++z) {
        bandPlaneMagnitude(half.data(), nullptr, nz, ny, nx, z, BandSide::ReOnly, plane.data());
        for (Index y = 0; y < ny; ++y)
            for (Index x = 0; x < nx; ++x) {
                const double expected = std::abs(full[static_cast<std::size_t>((z * ny + y) * nx + x)]);
                const double got = plane[static_cast<std::size_t>(((y + ny / 2) % ny) * nx + (x + nx / 2) % nx)];
                CHECK_THAT(got, WithinAbs(expected, 1e-9));
            }
    }
    // a side band combines the two parts: with im = re the +side doubles the
    // magnitude where re is real... just check Plus/Minus differ from ReOnly
    std::vector<double> plus(plane.size()), minus(plane.size());
    bandPlaneMagnitude(half.data(), half.data(), nz, ny, nx, 0, BandSide::Plus, plus.data());
    bandPlaneMagnitude(half.data(), half.data(), nz, ny, nx, 0, BandSide::Minus, minus.data());
    bandPlaneMagnitude(half.data(), nullptr, nz, ny, nx, 0, BandSide::ReOnly, plane.data());
    // re + i re = (1 + i) re and re - i re = (1 - i) re: both sqrt(2) |re|
    for (std::size_t i = 0; i < plane.size(); ++i) {
        CHECK_THAT(plus[i], WithinAbs(std::sqrt(2.0) * plane[i], 1e-9));
        CHECK_THAT(minus[i], WithinAbs(std::sqrt(2.0) * plane[i], 1e-9));
    }
    REQUIRE_THROWS_AS(bandPlaneMagnitude(half.data(), nullptr, nz, ny, nx, 0, BandSide::Plus, plane.data()),
                      std::invalid_argument);
}

TEST_CASE("predictedK0 follows the reconstruction's initial guess", "[app][volume][overlay]") {
    SIMParameters p = testParameters();   // 3 dirs, explicit angles, ls 0.2035
    const auto k2d = predictedK0(p, 1);
    REQUIRE(k2d.size() == 3);
    for (std::size_t d = 0; d < 3; ++d) {
        CHECK_THAT(std::hypot(k2d[d][0], k2d[d][1]), WithinRel(1.0 / p.linespacing_um, 1e-12));
        CHECK_THAT(std::atan2(k2d[d][1], k2d[d][0]), WithinAbs((*p.k0_angles)[d], 1e-12));
    }
    const auto k3d = predictedK0(p, 9);   // 3D: order-1 spacing is half the finest pattern's
    CHECK_THAT(std::hypot(k3d[0][0], k3d[0][1]), WithinRel(0.5 / p.linespacing_um, 1e-12));

    p.k0_angles.reset();
    p.k0_start_angle = 0.25;
    const auto k = predictedK0(p, 1);
    CHECK_THAT(std::atan2(k[1][1], k[1][0]), WithinAbs(0.25 + kPi / 3.0, 1e-12));
    CHECK_THAT(otfSupportRadius(p), WithinRel(2.0 * p.na / (p.wavelength_nm * 1e-3), 1e-12));
}

TEST_CASE("SpectrumGeometry maps frequencies to centered pixels", "[app][volume][overlay]") {
    const SpectrumGeometry g{64, 128, 0.1, 0.2};
    const auto dc = g.pixelOf(0.0, 0.0);
    CHECK(dc[0] == 64.0);
    CHECK(dc[1] == 32.0);
    const auto p = g.pixelOf(1.0, -0.4);
    CHECK_THAT(p[0], WithinAbs(74.0, 1e-12));
    CHECK_THAT(p[1], WithinAbs(30.0, 1e-12));
    const auto r = g.radiusPixels(2.0);
    CHECK_THAT(r[0], WithinAbs(20.0, 1e-12));
    CHECK_THAT(r[1], WithinAbs(10.0, 1e-12));
}

TEST_CASE("otfDisplayVolume renders a centered OTF whose peak is the DC voxel", "[app][volume][otf]") {
    SIMParameters p = testParameters();
    const OTFRadiallyAveraged otf = idealOTF(p, /*threeD=*/false);
    const Buffer<double> v = otfDisplayVolume(otf, 0, p, 64, 48, 1);
    REQUIRE(v.shape() == Shape{1, 48, 64});
    const double* d = v.data();
    Index best = 0;
    for (Index i = 1; i < v.size(); ++i)
        if (d[i] > d[best]) best = i;
    CHECK(best == 24 * 64 + 32);
    CHECK_THAT(d[best], WithinAbs(1.0, 1e-9));
    // outside the support the OTF is zero: the corner voxel
    CHECK_THAT(d[0], WithinAbs(0.0, 1e-9));
    REQUIRE_THROWS_AS(otfDisplayVolume(otf, 7, p, 64, 48, 1), std::out_of_range);
}

// --- session without an OTF file ------------------------------------------

TEST_CASE("ReconSession falls back to the ideal OTF and captures diagnostics", "[app][session][ideal]") {
    ReconSession s;
    s.setParameters(testParameters());
    CHECK(s.usesIdealOtf());
    s.loadRaw((kData / "raw.tif").string());
    CHECK(s.validate().empty());   // no OTF file is not an error any more

    // the OTF shown to the user is the ideal 3D one for this 9-plane stack
    auto otf = s.otf();
    REQUIRE(otf);
    CHECK(otf->data().dimension(0) == 3);
    CHECK(otf->data().dimension(2) > 1);
    CHECK(s.otf() == otf);   // cached until the setup changes
    s.setParameters(testParameters());
    CHECK(s.otf() != otf);

    s.setCaptureDiagnostics(true);
    ReconResult r = s.reconstruct(Device::cpu(), PlanRigor::Estimate);
    CHECK(r.idealOtf);
    REQUIRE(r.volume.shape() == Shape{9, 128, 128});
    REQUIRE(r.diagnostics.captured);
    CHECK(r.diagnostics.separated.shape() == Shape{3 * 5 * 9, 64, 33});
    CHECK(r.diagnostics.filtered.shape() == Shape{3 * 5 * 9, 64, 33});
    for (Index i = 0; i < r.volume.size(); ++i) REQUIRE(std::isfinite(r.volume.data()[i]));

    // the captured bands feed the viewer's band volumes
    const Buffer<double> band = bandMagnitudeVolume(r.diagnostics, r.diagnostics.separated, 0, 0, BandSide::ReOnly);
    REQUIRE(band.shape() == Shape{9, 64, 64});
    const Buffer<double> side = bandMagnitudeVolume(r.diagnostics, r.diagnostics.filtered, 2, 1, BandSide::Minus);
    REQUIRE(side.shape() == Shape{9, 64, 64});
    REQUIRE_THROWS_AS(bandMagnitudeVolume(r.diagnostics, r.diagnostics.separated, 3, 0, BandSide::ReOnly),
                      std::out_of_range);

    // a measured OTF path switches back and the next run reports it
    s.setOtfPath((kData / "otf.tif").string());
    CHECK_FALSE(s.usesIdealOtf());
    s.setCaptureDiagnostics(false);
    ReconResult m = s.reconstruct(Device::cpu(), PlanRigor::Estimate);
    CHECK_FALSE(m.idealOtf);
    CHECK_FALSE(m.diagnostics.captured);
    CHECK_FALSE(m.plansReused);
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
