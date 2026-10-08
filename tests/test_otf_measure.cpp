// measureOTF: turning a bead stack into the radially averaged table loadOTF
// reads. sirius/otf_measure.hpp carries the reasoning and the measurements
// from the user's own sparse bead field; this file pins them, on synthetic
// scenes whose answer is known analytically, so correctness does not depend on
// any file outside the repository.
//
// The scene: a point object at r0 imaged under a two-beam pattern gives, for
// phase p, (1 + m cos(2 pi k0 . r0 + phi_p)) h(r - r0) -- the pattern enters
// as a SCALAR, because a delta samples the illumination at one point. Band
// separation therefore returns order 0 = h and order 1 = (m/2) e^{-i theta} h
// with theta the illumination phase at that bead, and combine_reim rotates
// that constant phase into the real part. So a synthetic stack has an exactly
// known answer: order 1 is order 0 times m/2, and order 0 is the PSF's own
// OTF. With a Gaussian PSF of width sigma that is exp(-2 pi^2 sigma^2 k^2),
// in closed form.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

#include "sirius/otf_io.hpp"
#include "sirius/otf_measure.hpp"
#include "sirius/tiff_io.hpp"

#include "temp_path.hpp"

using namespace sirius;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using Cplx = std::complex<double>;
using Stack = Eigen::Tensor<double, 3, Eigen::RowMajor>;

namespace {

    struct Bead {
        double x = 0.0, y = 0.0, z = 0.0;   // voxels, sub-pixel
        double amplitude = 1000.0;
        double sigmaScale = 1.0;            // > 1 widens this bead (an aggregate)
    };

    struct Scene {
        int nz = 24, ny = 64, nx = 64, nphases = 3;
        double dxy = 0.1, dz = 0.2;         // um
        double sigmaXY = 1.6, sigmaZ = 2.4; // voxels
        double depth = 0.7;                 // m in 1 + m cos(.)
        double periodUm = 0.8, angleRad = 0.6;
        double offset = 100.0;              // the camera's dark level
        double noise = 0.0;                 // uniform, deterministic
        std::vector<Bead> beads{Bead{32.3, 31.6, 11.4, 1000.0, 1.0}};
        BeadPhasePacking packing = BeadPhasePacking::PhaseFastest;
    };

    // (nz * nphases, ny, nx), laid out as the scene's packing says.
    Stack render(const Scene& s) {
        const double k0 = 1.0 / s.periodUm;
        const double k0x = k0 * std::cos(s.angleRad), k0y = k0 * std::sin(s.angleRad);
        Stack out(s.nz * s.nphases, s.ny, s.nx);
        out.setZero();
        unsigned rng = 12345u;
        for (int z = 0; z < s.nz; ++z)
            for (int p = 0; p < s.nphases; ++p) {
                const int section = s.packing == BeadPhasePacking::PhaseFastest ? z * s.nphases + p
                                                                               : p * s.nz + z;
                const double phi = 2.0 * 3.14159265358979323846 * p / s.nphases;
                for (int y = 0; y < s.ny; ++y)
                    for (int x = 0; x < s.nx; ++x) {
                        double v = s.offset;
                        for (const Bead& b : s.beads) {
                            const double theta = 2.0 * 3.14159265358979323846 *
                                                 (k0x * b.x * s.dxy + k0y * b.y * s.dxy);
                            const double sx = s.sigmaXY * b.sigmaScale, sz = s.sigmaZ * b.sigmaScale;
                            const double ex = (x - b.x) / sx, ey = (y - b.y) / sx, ez = (z - b.z) / sz;
                            const double g = std::exp(-0.5 * (ex * ex + ey * ey + ez * ez));
                            v += b.amplitude * (1.0 + s.depth * std::cos(theta + phi)) * g;
                        }
                        if (s.noise > 0.0) {
                            rng = rng * 1664525u + 1013904223u;
                            v += s.noise * ((rng >> 8) / 8388608.0 - 1.0);
                        }
                        out(section, y, x) = v;
                    }
            }
        return out;
    }

    OtfMeasureOptions baseOptions(const Scene& s) {
        OtfMeasureOptions o;
        o.nphases = s.nphases;
        o.dxy = s.dxy;
        o.dz = s.dz;
        o.packing = s.packing;
        o.beadDiameterUm = 0.0;   // the scene's object is a true point: nothing to divide out
        // makeotf's border_size is 20 px whatever the section is, and on a
        // 64 px one that frame is 86% of the image and holds the beads. The
        // scenes here are 64 px, so the border is sized to them.
        o.backgroundBorder = 6;
        o.patternPeriodUm = s.periodUm;
        o.patternAngleRad = s.angleRad;
        return o;
    }

    // exp(-2 pi^2 sigma^2 k^2), the OTF of the scene's Gaussian PSF.
    double gaussianOtf(double sigmaUm, double k) {
        const double pi = 3.14159265358979323846;
        return std::exp(-2.0 * pi * pi * sigmaUm * sigmaUm * k * k);
    }

    double maxAbsDiff(const OTFRadiallyAveraged& a, const OTFRadiallyAveraged& b) {
        const auto& x = a.data();
        const auto& y = b.data();
        REQUIRE(x.dimensions() == y.dimensions());
        double worst = 0.0;
        for (Eigen::Index i = 0; i < x.size(); ++i) worst = std::max(worst, std::abs(x.data()[i] - y.data()[i]));
        return worst;
    }

    double peakOf(const OTFRadiallyAveraged& a) {
        double m = 0.0;
        for (Eigen::Index i = 0; i < a.data().size(); ++i) m = std::max(m, std::abs(a.data().data()[i]));
        return m;
    }


    // --- reading options from the environment ---------------------------------
    // The measurement has no tool, no CLI flag and no Python binding yet: those
    // live in files another chain owns this round (app/core/tool_api.cpp,
    // bindings/src/bind_sim.cpp). Until they land, this is how a job measures an
    // OTF from a stack -- a stopgap, said plainly, and the reason the last case
    // in this file reads its options from the environment.
    std::string envStr(const char* name, const std::string& fallback = std::string()) {
        const char* v = std::getenv(name);
        return (v != nullptr && *v != '\0') ? std::string(v) : fallback;
    }
    double envNum(const char* name, double fallback) {
        const std::string v = envStr(name);
        return v.empty() ? fallback : std::stod(v);
    }
    int envInt(const char* name, int fallback) {
        const std::string v = envStr(name);
        return v.empty() ? fallback : std::stoi(v);
    }
} // namespace

TEST_CASE("measureOTF on one synthetic bead is the PSF's own OTF", "[otf_measure]") {
    Scene s;
    const Stack stack = render(s);
    const OtfMeasureResult r = measureOTF(stack, baseOptions(s));

    REQUIRE(r.norders == 2);
    REQUIRE(r.nkr == 33);                    // min(nx, ny) / 2 + 1
    REQUIRE(r.nzotf == s.nz);
    CHECK_THAT(r.dkr, WithinRel(1.0 / (s.nx * s.dxy), 1e-12));
    CHECK_THAT(r.dkz, WithinRel(1.0 / (s.nz * s.dz), 1e-12));

    // the scale step leaves order 0's DC at 1 whichever reference it used, so
    // the table reaches the reconstruction on the one scale an absolute
    // otfcutoff means anything against
    CHECK_THAT(r.otf.data()(0, 0, 0).real(), WithinAbs(1.0, 1e-12));
    CHECK(r.otf.atReferenceScale());

    // order 0's kz = 0 profile against the closed form. The first two radial
    // samples are the ones makeotf's fixorigin repairs and are left out.
    const double sigmaUm = s.sigmaXY * s.dxy;
    for (int ir = 2; ir <= 8; ++ir) {
        const double want = gaussianOtf(sigmaUm, ir * r.dkr);
        CHECK_THAT(r.otf.data()(0, ir, 0).real(), WithinRel(want, 0.02));
    }
    // further out the closed form is an idealisation the grid departs from (it
    // is the transform of a continuous infinite Gaussian, this is a radial
    // average of a 64-point one), so the bound widens rather than pretending
    for (int ir = 9; ir <= 14; ++ir)
        CHECK_THAT(r.otf.data()(0, ir, 0).real(), WithinRel(gaussianOtf(sigmaUm, ir * r.dkr), 0.08));

    // order 1 is order 0 times m/2 -- the depth convention findings 9k.55
    // established, here from first principles rather than by comparison. The
    // band ratio is that factor directly; modulationDepth carries one extra
    // radial sample of the PSF, because step 9 replaced order 1's kr = 0.
    CHECK_THAT(r.bandRatio, WithinRel(s.depth / 2.0, 0.01));
    CHECK_THAT(r.modulationDepth, WithinRel(s.depth / 2.0, 0.02));
    for (int ir = 2; ir <= 10; ++ir)
        CHECK_THAT(r.otf.data()(1, ir, 0).real(),
                   WithinRel((s.depth / 2.0) * r.otf.data()(0, ir, 0).real(), 0.02));

    // radialft forces this exactly, which is what loadOTF's structural check
    // measures a real makeotf table by
    CHECK(r.hermitianKzError == 0.0);
    CHECK(r.found == 1);
    CHECK(r.kept == 1);
    REQUIRE(r.beads.size() == 1);
    CHECK_THAT(r.beads[0].x, WithinAbs(s.beads[0].x, 0.05));
    CHECK_THAT(r.beads[0].y, WithinAbs(s.beads[0].y, 0.05));
    CHECK_THAT(r.beads[0].z, WithinAbs(s.beads[0].z, 0.05));
}

TEST_CASE("the DC scale is set by the background estimate, and the band ratio is not", "[otf_measure]") {
    // THE measurement behind OtfMeasureScale and behind bandRatio. makeotf's
    // rescale() divides every order by order 0's kr = kz = 0 sample, which is
    // the integral of the background-subtracted band 0, so a background that
    // is off by a little is a table that is off by a lot. The two arms below
    // differ in NOTHING but the background they are told to subtract.
    Scene s;
    const Stack stack = render(s);
    // the bead's own integral, which is what order 0's DC should be:
    // A (2 pi)^{3/2} sigma_xy^2 sigma_z
    const double pi = 3.14159265358979323846;
    const double beadIntegral = s.beads[0].amplitude * std::pow(2.0 * pi, 1.5) * s.sigmaXY * s.sigmaXY * s.sigmaZ;
    const double perAdu = static_cast<double>(s.nz) * s.nphases * s.ny * s.nx / s.nphases;

    OtfMeasureOptions right = baseOptions(s);
    right.background = s.offset;              // exactly the dark level
    OtfMeasureOptions off = baseOptions(s);
    off.background = s.offset + 0.5;          // half an ADU too much

    const OtfMeasureResult a = measureOTF(stack, right);
    const OtfMeasureResult b = measureOTF(stack, off);
    WARN("DC with the right background " << a.order0Dc << " (the bead's integral is " << beadIntegral
         << "), with 0.5 ADU too much " << b.order0Dc << "; 1 ADU is " << perAdu
         << " of DC, i.e. " << 100.0 * a.scaleSensitivityPerAdu << "% per ADU");
    CHECK_THAT(a.order0Dc, WithinRel(beadIntegral, 0.01));
    CHECK_THAT(a.scaleSensitivityPerAdu, WithinRel(perAdu / beadIntegral, 0.02));

    // the table moves, and by the amount the sensitivity predicts
    const double moved = maxAbsDiff(a.otf, b.otf) / peakOf(a.otf);
    CHECK(moved > 0.2);
    CHECK(std::abs(b.modulationDepth - a.modulationDepth) > 0.2 * a.modulationDepth);

    // and the band ratio does not: every order is divided by the same number
    CHECK_THAT(b.bandRatio, WithinRel(a.bandRatio, 1e-9));
    CHECK_THAT(a.bandRatio, WithinRel(s.depth / 2.0, 0.02));
    CHECK(a.bandRatioSamples > 100);
    CHECK(a.bandRatioIqr < 0.02 * a.bandRatio);

    // the two estimators are both reported, whichever was used
    OtfMeasureOptions estimated = baseOptions(s);
    const OtfMeasureResult e = measureOTF(stack, estimated);
    CHECK_THAT(e.backgroundBorderMean, WithinAbs(s.offset, 1e-6));
    CHECK_THAT(e.backgroundDarkest, WithinAbs(s.offset, 1e-6));
    CHECK_THAT(e.backgroundMean, WithinAbs(s.offset, 1e-6));
    CHECK_THAT(e.summary(), ContainsSubstring("band ratio"));
}

TEST_CASE("makeotf's fixorigin is background proof and off the DC scale", "[otf_measure]") {
    // The repair the design of this unit started from as its normalisation
    // reference, measured rather than argued. It IS exactly invariant to the
    // background -- the fit runs over kr >= 1 and the background enters only
    // the kr = 0 column -- but its sum[0] is the DC alone where its sum[i >= 1]
    // are kz SUMS, so the line it extrapolates lives on another scale and the
    // table it produces is not the OTF over its DC. Here the right table is
    // known in closed form, so the factor can be quoted.
    Scene s;
    const Stack stack = render(s);
    OtfMeasureOptions fix = baseOptions(s);
    fix.scale = OtfMeasureScale::MakeotfFixOrigin;
    OtfMeasureOptions fixOff = fix;
    fix.background = s.offset;
    fixOff.background = s.offset + 0.5;

    const OtfMeasureResult a = measureOTF(stack, fix);
    const OtfMeasureResult b = measureOTF(stack, fixOff);
    CHECK(maxAbsDiff(a.otf, b.otf) / peakOf(a.otf) < 1e-12);     // exactly invariant

    OtfMeasureOptions dc = baseOptions(s);
    dc.background = s.offset;
    const OtfMeasureResult d = measureOTF(stack, dc);
    const double sigmaUm = s.sigmaXY * s.dxy;
    const double factor = d.otf.data()(0, 4, 0).real() / a.otf.data()(0, 4, 0).real();
    WARN("fixorigin puts the table " << factor << "x below the DC scale; nz / (sqrt(2 pi) sigma_z) is "
         << s.nz / (std::sqrt(2.0 * 3.14159265358979323846) * s.sigmaZ)
         << "; its divisor " << a.scaleDivisor << " against the DC " << a.order0Dc);
    CHECK(factor > 1.5);                                          // not the DC scale
    // the DC-scaled table is the analytic Gaussian and the repaired one is not
    for (int ir = 3; ir <= 8; ++ir) {
        const double want = gaussianOtf(sigmaUm, ir * d.dkr);
        CHECK_THAT(d.otf.data()(0, ir, 0).real(), WithinRel(want, 0.02));
    }
    CHECK(std::abs(a.otf.data()(0, 4, 0).real() - gaussianOtf(sigmaUm, 4 * a.dkr)) >
          0.2 * gaussianOtf(sigmaUm, 4 * a.dkr));
    // both numbers are reported whichever is used
    CHECK(d.lineFitToOrigin != 0.0);
    CHECK_THAT(d.scaleDivisor, WithinRel(d.order0Dc, 1e-12));
    CHECK_THAT(a.scaleDivisor, WithinRel(a.lineFitToOrigin, 1e-9));
    CHECK_THAT(a.summary(), ContainsSubstring("fixorigin"));
}

TEST_CASE("a field of beads needs the field path, and the single-bead path fails on it", "[otf_measure]") {
    // What the field path buys, as a controlled comparison: the same five
    // beads, measured once the way makeotf would (one global maximum, one
    // phase ramp) and once per bead. The truth is the one-bead table, which
    // the first test has already checked against the closed form.
    Scene one;
    const OtfMeasureResult truth = measureOTF(render(one), baseOptions(one));

    Scene many = one;
    many.beads = {Bead{16.4, 15.2, 11.5, 1000.0, 1.0}, Bead{47.6, 16.8, 11.5, 900.0, 1.0},
                  Bead{16.1, 47.3, 11.5, 1100.0, 1.0}, Bead{47.2, 47.9, 11.5, 800.0, 1.0},
                  Bead{31.7, 31.4, 11.5, 1050.0, 1.0}};
    const Stack stack = render(many);

    OtfMeasureOptions single = baseOptions(many);
    const OtfMeasureResult asOneBead = measureOTF(stack, single);

    OtfMeasureOptions fieldOpts = baseOptions(many);
    fieldOpts.field = true;
    fieldOpts.detect.minSeparationLateralUm = 0.8;
    fieldOpts.detect.boundaryMarginLateralUm = 0.4;
    fieldOpts.detect.roiLateralUm = 1.2;
    fieldOpts.detect.sigmaMaxLateralUm = 0.4;
    fieldOpts.detect.sigmaMinLateralUm = 0.05;
    const OtfMeasureResult asField = measureOTF(stack, fieldOpts);

    CHECK(asField.found >= 5);
    CHECK(asField.kept == 5);

    // The side band is where the difference lives: every bead sits at its own
    // illumination phase, so one global ramp leaves four beads' phase ramps in
    // the average and the order-1 band is destroyed. Order 0 survives much
    // better, because its interference terms are not phase-shifted.
    const auto order1Error = [&](const OtfMeasureResult& r) {
        double worst = 0.0;
        for (int ir = 2; ir <= 10; ++ir)
            worst = std::max(worst, std::abs(r.otf.data()(1, ir, 0).real() -
                                             truth.otf.data()(1, ir, 0).real()));
        return worst / std::abs(truth.otf.data()(1, 2, 0).real());
    };
    const double fieldErr = order1Error(asField);
    const double singleErr = order1Error(asOneBead);
    {
        std::string o0, o1, f0, f1, s0, s1;
        for (int ir = 0; ir <= 10; ++ir) {
            o0 += " " + std::to_string(truth.otf.data()(0, ir, 0).real());
            o1 += " " + std::to_string(truth.otf.data()(1, ir, 0).real());
            f0 += " " + std::to_string(asField.otf.data()(0, ir, 0).real());
            f1 += " " + std::to_string(asField.otf.data()(1, ir, 0).real());
            s0 += " " + std::to_string(asOneBead.otf.data()(0, ir, 0).real());
            s1 += " " + std::to_string(asOneBead.otf.data()(1, ir, 0).real());
        }
        WARN("kz=0 profiles, kr 0..10\n  truth  order0" << o0 << "\n  truth  order1" << o1
             << "\n  field  order0" << f0 << "\n  field  order1" << f1 << "\n  single order0" << s0
             << "\n  single order1" << s1);
    }
    WARN("order 1 against the one-bead truth: field path " << fieldErr << ", single-bead path " << singleErr
         << "; band ratio truth " << truth.bandRatio << ", field " << asField.bandRatio << ", single "
         << asOneBead.bandRatio << " (m/2 = " << many.depth / 2.0 << ")");
    CHECK(fieldErr < 0.12);
    CHECK(singleErr > 2.0 * fieldErr);

    // the band ratio -- the scale-free depth -- is recovered by the field path
    // and not by the single-bead one
    CHECK_THAT(asField.bandRatio, WithinRel(many.depth / 2.0, 0.1));
    CHECK(std::abs(asOneBead.bandRatio - many.depth / 2.0) >
          2.0 * std::abs(asField.bandRatio - many.depth / 2.0));

    // the per-bead scatter is the diagnostic a user judges the measurement by
    CHECK(asField.modulationDepthSpread >= 0.0);
    CHECK_THAT(asField.summary(), ContainsSubstring("5 of"));
}

TEST_CASE("with one bead the field path reduces to makeotf's own", "[otf_measure]") {
    // The mask around a bead reaches half way to its nearest neighbour, which
    // for a lone bead is the whole field: the field path is then the
    // single-bead path with the centre from a Gaussian fit instead of three
    // parabolas, and the two tables have to agree.
    Scene s;
    const Stack stack = render(s);
    OtfMeasureOptions fieldOpts = baseOptions(s);
    fieldOpts.field = true;
    const OtfMeasureResult f = measureOTF(stack, fieldOpts);
    const OtfMeasureResult m = measureOTF(stack, baseOptions(s));
    REQUIRE(f.kept == 1);
    CHECK(maxAbsDiff(f.otf, m.otf) / peakOf(m.otf) < 2e-3);
    CHECK_THAT(f.modulationDepth, WithinRel(m.modulationDepth, 0.01));
}

TEST_CASE("every bead the field path drops is reported with its reason", "[otf_measure]") {
    Scene s;
    s.beads = {
        Bead{32.3, 31.6, 11.5, 1000.0, 1.0},   // the reference bead
        Bead{48.2, 31.4, 11.5, 1000.0, 3.0},   // three times as wide: an aggregate
        Bead{16.4, 31.5, 11.5, 20.0, 1.0},     // 2% of the brightest: below the floor
        Bead{1.5, 1.5, 11.5, 1000.0, 1.0},     // against the corner
    };
    const Stack stack = render(s);
    OtfMeasureOptions o = baseOptions(s);
    o.field = true;
    o.detect.minSeparationLateralUm = 0.8;
    o.detect.roiLateralUm = 1.2;
    o.detect.boundaryMarginLateralUm = 0.4;
    o.detect.sigmaMinLateralUm = 0.05;
    o.detect.sigmaMaxLateralUm = 0.25;
    o.detect.minAmplitudeFraction = 0.1;
    const OtfMeasureResult r = measureOTF(stack, o);

    CHECK(r.kept == 1);
    CHECK(r.rejected[static_cast<std::size_t>(BeadRejection::WidthTooLarge)] >= 1);
    CHECK(r.rejected[static_cast<std::size_t>(BeadRejection::Amplitude)] >= 1);
    CHECK(r.rejected[static_cast<std::size_t>(BeadRejection::NearBoundary)] >= 1);
    // the inventory names every candidate, kept or not
    int described = 0;
    for (const BeadFit& b : r.beads) {
        CHECK(std::string(beadRejectionName(b.rejection)).size() > 0);
        if (!b.kept) ++described;
    }
    CHECK(described == static_cast<int>(r.beads.size()) - r.kept);

}


TEST_CASE("a saturated bead is dropped on its level, and the rest is measured", "[otf_measure]") {
    Scene s;
    // 4000 peaks at 1300 at worst and 600 at 1120 at best, so one bead is over
    // a 1000 level whatever pattern phase each sits at and the other is not
    s.beads = {Bead{20.3, 31.6, 11.5, 4000.0, 1.0}, Bead{44.4, 31.5, 11.5, 600.0, 1.0}};
    const Stack stack = render(s);
    OtfMeasureOptions o = baseOptions(s);
    o.field = true;
    o.detect.minSeparationLateralUm = 0.8;
    o.detect.roiLateralUm = 1.2;
    o.detect.boundaryMarginLateralUm = 0.4;
    o.detect.minAmplitudeFraction = 0.02;
    o.detect.sigmaMaxLateralUm = 0.4;
    o.detect.saturationLevel = 1000.0;
    const OtfMeasureResult r = measureOTF(stack, o);
    CHECK(r.rejected[static_cast<std::size_t>(BeadRejection::Saturated)] == 1);
    CHECK(r.kept == 1);
    CHECK_THAT(r.bandRatio, WithinRel(s.depth / 2.0, 0.1));
}

TEST_CASE("the measurement writes makeotf's layout, and loadOTF reads it back unchanged", "[otf_measure]") {
    Scene s;
    const OtfMeasureResult r = measureOTF(render(s), baseOptions(s));

    sirius::test::TempFile out("otf_measure", ".tif");
    const std::vector<std::string> written = writeMeasuredOTF(out.str, r);
    REQUIRE(written.size() == 2);
    CHECK(written[0] == out.str);
    CHECK(written[1] == out.str + ".toml");
    const std::filesystem::path side(written[1]);

    // makeotf's layout: norders pages of nkr rows by 2 * nzotf columns
    const ImageStack<float> pages = readTiffStack<float>(out.str);
    CHECK(pages.dimension(0) == r.norders);
    CHECK(pages.dimension(1) == r.nkr);
    CHECK(pages.dimension(2) == 2 * r.nzotf);

    SIMParameters p;
    p.ndirs = 1;
    p.nphases = s.nphases;
    p.na = 1.35;
    p.nimm = 1.405;
    p.wavelength_nm = 515.0;
    p.dx = p.dy = s.dxy;
    p.dz = s.dz;
    p.dz_psf = s.dz;

    OtfLoadReport rep;
    const OTFRadiallyAveraged back = loadOTF(out.str, p, OtfLoadOptions{}, &rep);
    CHECK(rep.layout == OtfLayout::CudasireconRadial);
    CHECK(rep.norders == r.norders);
    CHECK(rep.nkr == r.nkr);
    CHECK(rep.nzotf == r.nzotf);
    // the sidecar, not a derivation from whatever pixel size the run uses
    CHECK(rep.samplingSource == OtfSamplingSource::Sidecar);
    CHECK(rep.sidecarPath == written[1]);
    CHECK_THAT(back.dkrotf(), WithinRel(r.dkr, 1e-9));
    CHECK_THAT(back.dkzotf(), WithinRel(r.dkz, 1e-9));
    // Auto normalisation finds the table already at the reference and does
    // nothing, which is why writing on makeotf's scale is what keeps the
    // interchange honest
    CHECK(rep.normalizationApplied == false);
    CHECK(rep.kzRolled == false);
    CHECK(rep.hermitianErrorAsStored == 0.0);
    // the values survive the float32 the layout is defined in
    CHECK(maxAbsDiff(back, r.otf) <= 1e-6 * peakOf(r.otf));

    std::error_code ec;
    std::filesystem::remove(side, ec);
}

TEST_CASE("the phase packing is a stated option, and the wrong one is visible", "[otf_measure]") {
    Scene a;
    const Stack fastest = render(a);
    Scene b = a;
    b.packing = BeadPhasePacking::PhaseSlowest;
    const Stack slowest = render(b);

    OtfMeasureOptions oa = baseOptions(a);
    OtfMeasureOptions ob = baseOptions(b);
    const OtfMeasureResult ra = measureOTF(fastest, oa);
    const OtfMeasureResult rb = measureOTF(slowest, ob);
    CHECK(maxAbsDiff(ra.otf, rb.otf) < 1e-12);

    // read with the other convention the bands are mixed and the table is a
    // different table, which is the point: the packing is not guessable from
    // the file and has to be stated
    OtfMeasureOptions wrong = baseOptions(a);
    wrong.packing = BeadPhasePacking::PhaseSlowest;
    const OtfMeasureResult bad = measureOTF(fastest, wrong);
    WARN("the wrong packing: band ratio " << bad.bandRatio << " against " << ra.bandRatio
         << ", table differs by " << maxAbsDiff(bad.otf, ra.otf) / peakOf(ra.otf) << " of peak");
    CHECK(maxAbsDiff(bad.otf, ra.otf) / peakOf(ra.otf) > 0.05);
}

TEST_CASE("the finite bead size is divided out, and only then", "[otf_measure]") {
    Scene s;
    const Stack stack = render(s);
    OtfMeasureOptions none = baseOptions(s);
    OtfMeasureOptions compensated = baseOptions(s);
    compensated.beadDiameterUm = 0.12;
    const OtfMeasureResult a = measureOTF(stack, none);
    const OtfMeasureResult b = measureOTF(stack, compensated);
    // dividing by the transform of a 0.12 um sphere lifts the high radial
    // samples and leaves the origin alone
    CHECK_THAT(b.otf.data()(0, 0, 0).real(), WithinAbs(1.0, 1e-12));
    CHECK(b.otf.data()(0, 20, 0).real() > a.otf.data()(0, 20, 0).real());
    CHECK(maxAbsDiff(a.otf, b.otf) > 1e-3);
}

TEST_CASE("validateOtfMeasure refuses what it cannot measure, with the numbers", "[otf_measure]") {
    OtfMeasureOptions o;
    o.dxy = 0.1;
    o.dz = 0.2;
    o.nphases = 3;
    CHECK(validateOtfMeasure(o, 72, 64, 64).empty());
    CHECK_THAT(validateOtfMeasure(o, 70, 64, 64), ContainsSubstring("70 sections"));
    CHECK_THAT(validateOtfMeasure(o, 72, 4, 64), ContainsSubstring("at least 8 px"));
    {
        OtfMeasureOptions q = o;
        q.dxy = 0.0;
        CHECK_THAT(validateOtfMeasure(q, 72, 64, 64), ContainsSubstring("dxy"));
    }
    {
        OtfMeasureOptions q = o;
        q.norders = 3;
        CHECK_THAT(validateOtfMeasure(q, 72, 64, 64), ContainsSubstring("at least 5"));
    }
    {
        OtfMeasureOptions q = o;
        q.backgroundBorder = 40;
        CHECK_THAT(validateOtfMeasure(q, 72, 64, 64), ContainsSubstring("backgroundBorder 40"));
    }
    {
        OtfMeasureOptions q = o;
        q.phases = {0.0, 1.0};
        CHECK_THAT(validateOtfMeasure(q, 72, 64, 64), ContainsSubstring("2 values for 3 phases"));
    }
    // and measureOTF refuses the same thing with the same words
    Scene s;
    Stack stack = render(s);
    OtfMeasureOptions bad = baseOptions(s);
    bad.nphases = 5;
    CHECK_THROWS_WITH(measureOTF(stack, bad), ContainsSubstring("5 phases do not divide"));
}

// --- the real bead field, when it is at hand ---------------------------------
// The user's sparse bead field is 4.6 MB and is not committed; the environment
// names it instead. SIRIUS_OTF_BEAD_STACK is the raw stack and
// SIRIUS_OTF_BEAD_REFERENCE a makeotf table made from it, which is compared
// sample by sample after one global scale -- the only quantity the two can
// legitimately differ in, since the scale is the one thing makeotf reads off
// the background estimate.
TEST_CASE("the measured table equals makeotf's on the same real bead stack", "[otf_measure][real]") {
    const char* stackPath = std::getenv("SIRIUS_OTF_BEAD_STACK");
    const char* refPath = std::getenv("SIRIUS_OTF_BEAD_REFERENCE");
    if (stackPath == nullptr || *stackPath == '\0')
        SKIP("set SIRIUS_OTF_BEAD_STACK (and SIRIUS_OTF_BEAD_REFERENCE) to a raw bead stack");

    const ImageStack<double> raw = readTiffStack<double>(stackPath);
    OtfMeasureOptions o;
    o.nphases = 3;
    o.dxy = 0.085526315789473686;   // the acquisition's own camera pixel size
    o.dz = 0.1;                      // its SampleMotion step
    // makeotf's own defaults for everything the colleague did not pass, which
    // is what the comparison below establishes they used
    o.beadDiameterUm = 0.12;
    o.patternPeriodUm = 0.2;
    o.patternAngleRad = 1.57;
    o.scale = OtfMeasureScale::Order0Dc;   // makeotf's bare rescale, for the comparison
    // The bead-size division is the only step that reads a pixel size (dr
    // cancels out of the radial binning for a square section), and makeotf's
    // dr DEFAULTS to 0.106 um. Reproducing a table made without -xyres
    // therefore needs 0.106 there while dkr still needs the instrument's own
    // 0.0855263: 3.5e-3 of peak against 1.9e-7 separates the two.
    o.beadCompensationPixelUm = 0.106;
    OtfMeasureOptions asMakeotfRan = o;
    const OtfMeasureResult mine = measureOTF(raw, o);
    INFO("measured: " << mine.summary());
    CHECK(mine.norders == 2);
    CHECK(mine.hermitianKzError == 0.0);

    if (refPath != nullptr && *refPath != '\0') {
        SIMParameters p;
        p.ndirs = 1;
        p.nphases = 3;
        p.na = 1.35;
        p.nimm = 1.405;
        p.wavelength_nm = 515.0;
        p.dx = p.dy = o.dxy;
        p.dz = o.dz;
        p.dz_psf = o.dz;
        OtfLoadOptions lo;
        lo.normalization = OtfNormalization::AsStored;
        const OTFRadiallyAveraged ref = loadOTF(refPath, p, lo, nullptr);
        REQUIRE(ref.data().dimension(0) == mine.otf.data().dimension(0));
        REQUIRE(ref.data().dimension(1) == mine.otf.data().dimension(1));
        REQUIRE(ref.data().dimension(2) == mine.otf.data().dimension(2));
        // the best single real scale, then the residual it leaves -- order 0's
        // kr = kz = 0 sample is 1 in both tables by construction and is left
        // out of the fit
        double num = 0.0, den = 0.0, refPeak = 0.0;
        for (int ord = 0; ord < ref.data().dimension(0); ++ord)
            for (int ir = 0; ir < ref.data().dimension(1); ++ir)
                for (int k = 0; k < ref.data().dimension(2); ++k) {
                    if (ord == 0 && ir == 0 && k == 0) continue;
                    const double a = std::abs(mine.otf.data()(ord, ir, k));
                    const double b = std::abs(ref.data()(ord, ir, k));
                    num += a * b;
                    den += b * b;
                    refPeak = std::max(refPeak, b);
                }
        const double scale = den > 0.0 ? num / den : 1.0;
        double worst = 0.0, sum = 0.0, refSum = 0.0;
        long long n = 0;
        for (int ord = 0; ord < ref.data().dimension(0); ++ord)
            for (int ir = 0; ir < ref.data().dimension(1); ++ir)
                for (int k = 0; k < ref.data().dimension(2); ++k) {
                    if (ord == 0 && ir == 0 && k == 0) continue;
                    const double d = std::abs(mine.otf.data()(ord, ir, k) / scale - ref.data()(ord, ir, k));
                    worst = std::max(worst, d);
                    sum += d;
                    refSum += std::abs(ref.data()(ord, ir, k));
                    ++n;
                }
        WARN("one global scale " << scale << "; max|diff| / max|ref| " << worst / refPeak
                                 << "; mean|diff| / mean|ref| " << (sum / n) / (refSum / n)
                                 << "; our DC " << mine.order0Dc << " line fit " << mine.lineFitToOrigin
                                 << " ratio " << mine.order0Dc / mine.lineFitToOrigin);
        CHECK(worst / refPeak < 1e-5);

        // the same measurement with the instrument's own pixel size in the
        // division, which is the right number physically and the wrong one for
        // reaching this file
        OtfMeasureOptions truePixel = o;
        truePixel.beadCompensationPixelUm = 0.0;
        const OtfMeasureResult t = measureOTF(raw, truePixel);
        double tworst = 0.0, tnum = 0.0, tden = 0.0;
        for (int ord = 0; ord < ref.data().dimension(0); ++ord)
            for (int ir = 0; ir < ref.data().dimension(1); ++ir)
                for (int k = 0; k < ref.data().dimension(2); ++k) {
                    if (ord == 0 && ir == 0 && k == 0) continue;
                    tnum += std::abs(t.otf.data()(ord, ir, k)) * std::abs(ref.data()(ord, ir, k));
                    tden += std::abs(ref.data()(ord, ir, k)) * std::abs(ref.data()(ord, ir, k));
                }
        const double tscale = tden > 0.0 ? tnum / tden : 1.0;
        for (int ord = 0; ord < ref.data().dimension(0); ++ord)
            for (int ir = 0; ir < ref.data().dimension(1); ++ir)
                for (int k = 0; k < ref.data().dimension(2); ++k) {
                    if (ord == 0 && ir == 0 && k == 0) continue;
                    tworst = std::max(tworst, std::abs(t.otf.data()(ord, ir, k) / tscale -
                                                       ref.data()(ord, ir, k)));
                }
        WARN("with the instrument's own 0.0855263 um in the bead-size division: scale " << tscale
             << ", max|diff| / max|ref| " << tworst / refPeak);
        CHECK(tworst > 100.0 * worst);

        // makeotf's fixorigin repair on the same stack, for the record
        OtfMeasureOptions lf = o;
        lf.scale = OtfMeasureScale::MakeotfFixOrigin;
        const OtfMeasureResult lfr = measureOTF(raw, lf);
        WARN("with makeotf's fixorigin repair instead: " << lfr.summary());
        WARN("the table's depth as stored: ours " << mine.modulationDepth << ", the colleague's file "
             << ref.data()(1, 0, 0).real() / ref.data()(0, 0, 0).real() << ", fixorigin "
             << lfr.modulationDepth << " -- against the band ratio, ours " << mine.bandRatio
             << ", fixorigin " << lfr.bandRatio);
        // the band ratio is the same table's own, so it cannot move with the scale
        CHECK_THAT(lfr.bandRatio, WithinRel(mine.bandRatio, 1e-9));
    }

    SECTION("the two background estimators on the same stack") {
        OtfMeasureOptions dark = asMakeotfRan;
        dark.backgroundEstimate = BackgroundEstimate::DarkestFraction;
        const OtfMeasureResult dr = measureOTF(raw, dark);
        WARN("border mean " << mine.backgroundBorderMean << " -> DC " << mine.order0Dc << ", depth "
             << mine.modulationDepth << "; darkest tenth " << dr.backgroundDarkest << " -> DC "
             << dr.order0Dc << ", depth " << dr.modulationDepth << "; band ratio " << mine.bandRatio
             << " vs " << dr.bandRatio);
        CHECK_THAT(dr.bandRatio, WithinRel(mine.bandRatio, 1e-9));
    }

    SECTION("the measured tables are written where a run can use them") {
        const char* outDir = std::getenv("SIRIUS_OTF_OUT_DIR");
        if (outDir == nullptr || *outDir == '\0') SKIP("set SIRIUS_OTF_OUT_DIR to keep the tables");
        const std::filesystem::path dir(outDir);
        std::filesystem::create_directories(dir);
        const std::string stem = std::filesystem::path(stackPath).stem().string();
        OtfMeasureOptions keep = asMakeotfRan;
        keep.beadCompensationPixelUm = 0.0;   // the instrument's own pixel size, not makeotf's default
        const OtfMeasureResult r = measureOTF(raw, keep);
        OtfWriteOptions w;
        w.note = "measured from " + std::string(stackPath);
        const std::vector<std::string> files =
            writeMeasuredOTF((dir / (stem + "_sirius_OTF.tif")).string(), r, w);
        for (const std::string& f : files) WARN("wrote " << f);
        WARN("that table: " << r.summary());
        CHECK(files.size() == 2);
    }

    SECTION("the field path on the same stack, for the comparison the brief asks for") {
        OtfMeasureOptions f = asMakeotfRan;
        f.field = true;
        f.detect.minSeparationLateralUm = 0.6;
        f.detect.roiLateralUm = 1.2;
        f.detect.boundaryMarginLateralUm = 0.4;
        f.detect.sigmaMaxLateralUm = 0.4;
        const OtfMeasureResult asField = measureOTF(raw, f);
        WARN("field path: " << asField.summary());
        for (const BeadFit& b : asField.beads)
            WARN("  bead x " << b.x << " y " << b.y << " z " << b.z << " amp " << b.amplitude << " sxy "
                             << 0.5 * (b.sigmaX + b.sigmaY) << " sz " << b.sigmaZ << " residual " << b.residual
                             << " -> " << beadRejectionName(b.rejection) << (b.kept ? " (kept)" : ""));
        CHECK(asField.kept >= 1);
    }
}

// --- measuring any stack a job names -----------------------------------------
// Runs only when SIRIUS_OTF_MEASURE_STACK and SIRIUS_OTF_OUT are set, so it
// SKIPs in ctest. $S/sim/otf-measure/measure_otf.sbatch drives it.
TEST_CASE("measure an OTF from the stack the environment names", "[otf_measure][measure]") {
    const std::string stackPath = envStr("SIRIUS_OTF_MEASURE_STACK");
    const std::string outPath = envStr("SIRIUS_OTF_OUT");
    if (stackPath.empty() || outPath.empty())
        SKIP("set SIRIUS_OTF_MEASURE_STACK and SIRIUS_OTF_OUT to measure a stack");

    OtfMeasureOptions o;
    o.nphases = envInt("SIRIUS_OTF_NPHASES", 3);
    o.norders = envInt("SIRIUS_OTF_NORDERS", 0);
    o.dxy = envNum("SIRIUS_OTF_DXY", 0.0);
    o.dz = envNum("SIRIUS_OTF_DZ", 0.0);
    o.packing = envStr("SIRIUS_OTF_PACKING", "phase-fastest") == "phase-slowest"
                    ? BeadPhasePacking::PhaseSlowest
                    : BeadPhasePacking::PhaseFastest;
    o.background = envNum("SIRIUS_OTF_BACKGROUND", -1.0);
    o.backgroundBorder = envInt("SIRIUS_OTF_BORDER", 20);
    o.backgroundEstimate = envStr("SIRIUS_OTF_BG", "border") == "darkest"
                               ? BackgroundEstimate::DarkestFraction
                               : BackgroundEstimate::BorderMean;
    o.beadDiameterUm = envNum("SIRIUS_OTF_BEAD_UM", 0.12);
    o.patternPeriodUm = envNum("SIRIUS_OTF_PERIOD_UM", 0.2);
    o.patternAngleRad = envNum("SIRIUS_OTF_ANGLE_RAD", 1.57);
    o.beadCompensationPixelUm = envNum("SIRIUS_OTF_COMP_PIXEL_UM", 0.0);
    const std::string scale = envStr("SIRIUS_OTF_SCALE", "dc");
    o.scale = scale == "fixorigin" ? OtfMeasureScale::MakeotfFixOrigin
                                   : (scale == "none" ? OtfMeasureScale::AsMeasured
                                                      : OtfMeasureScale::Order0Dc);
    o.field = envInt("SIRIUS_OTF_FIELD", 0) != 0;
    o.detect.minSeparationLateralUm = envNum("SIRIUS_OTF_MIN_SEP_UM", 1.0);
    o.detect.roiLateralUm = envNum("SIRIUS_OTF_ROI_UM", 1.5);
    o.detect.boundaryMarginLateralUm = envNum("SIRIUS_OTF_MARGIN_UM", 1.0);
    o.detect.sigmaMaxLateralUm = envNum("SIRIUS_OTF_SIGMA_MAX_UM", 0.2);
    o.detect.sigmaMinLateralUm = envNum("SIRIUS_OTF_SIGMA_MIN_UM", 0.05);
    o.detect.minAmplitudeFraction = envNum("SIRIUS_OTF_MIN_AMP_FRAC", 0.05);
    o.detect.maxBeads = envInt("SIRIUS_OTF_MAX_BEADS", 64);

    const ImageStack<double> raw = readTiffStack<double>(stackPath);
    WARN("stack " << stackPath << " is " << raw.dimension(0) << " x " << raw.dimension(1) << " x "
                  << raw.dimension(2) << " (sections, rows, columns)");
    const std::string bad = validateOtfMeasure(o, static_cast<int>(raw.dimension(0)),
                                               static_cast<int>(raw.dimension(1)),
                                               static_cast<int>(raw.dimension(2)));
    if (!bad.empty()) FAIL("these options do not fit the stack: " << bad);

    const OtfMeasureResult r = measureOTF(raw, o);
    WARN(r.summary());
    for (const BeadFit& b : r.beads)
        WARN("  bead x " << b.x << " y " << b.y << " z " << b.z << " amp " << b.amplitude << " sxy "
                         << 0.5 * (b.sigmaX + b.sigmaY) << " sz " << b.sigmaZ << " residual " << b.residual
                         << " -> " << beadRejectionName(b.rejection) << (b.kept ? " (kept)" : ""));
    OtfWriteOptions w;
    w.note = "measured from " + stackPath;
    const std::vector<std::string> files = writeMeasuredOTF(outPath, r, w);
    for (const std::string& f : files) WARN("wrote " << f);
    CHECK(files.size() == 2);
    CHECK(r.kept >= 1);
}
