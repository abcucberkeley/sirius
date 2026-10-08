// loadOTF: the radially averaged OTF table read from a TIFF, a .dv or a
// .mrc, validated against the shapes an OTF is actually stored in, and put
// on the one scale the reconstruction's absolute thresholds mean anything
// against. sirius/otf_io.hpp carries the reasoning and the measurements;
// this file pins them.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "sirius/mrc_io.hpp"
#include "sirius/otf_io.hpp"
#include "sirius/tiff_io.hpp"

#include "temp_path.hpp"

using namespace sirius;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using Cplx = std::complex<double>;

namespace {
    std::string dataDir() { return std::string(SIRIUS_TEST_DATA_DIR); }
    std::string dataFile(const std::string& name) {
        return (std::filesystem::path(SIRIUS_TEST_DATA_DIR) / name).string();
    }

    // Deletes a path it did not create, for the sidecar a test writes beside
    // a TempFile (whose own name is generated, so it cannot be asked for).
    struct Scratch {
        std::filesystem::path p;
        explicit Scratch(std::filesystem::path q) : p(std::move(q)) {}
        ~Scratch() {
            std::error_code ec;
            std::filesystem::remove(p, ec);
        }
        Scratch(const Scratch&) = delete;
        Scratch& operator=(const Scratch&) = delete;
    };

    // The user's own 2026-04-21 iSOAR2 configuration, which is what the three
    // iSOAR2 fixtures were measured under: 3 phases (so 2 orders), xyres
    // 0.085, zres 0.250, zresPSF 0.1.
    SIMParameters isoar2Params() {
        SIMParameters p;
        p.ndirs = 1;
        p.nphases = 3;
        p.na = 1.35;
        p.nimm = 1.405;
        p.wavelength_nm = 515.0;
        p.linespacing_um = 0.504;
        p.dx = p.dy = 0.085;
        p.dz = 0.250;
        p.dz_psf = 0.1;
        return p;
    }

    bool anyNote(const OtfLoadReport& r, const std::string& needle) {
        return std::any_of(r.notes.begin(), r.notes.end(),
                           [&](const std::string& n) { return n.find(needle) != std::string::npos; });
    }

    // One page of a radially decaying OTF image, peak `peak` at the centre,
    // as OpenSIM / SIM4codes distribute one.
    ImageStack<float> otfImage(int n, double peak) {
        ImageStack<float> img(1, n, n);
        const int c = n / 2;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                const double r = std::hypot(j - c, i - c) / (0.5 * n);
                img(0, i, j) = static_cast<float>(peak * std::max(0.0, 1.0 - r) * std::exp(-2.0 * r));
            }
        return img;
    }

    // The same page with DC moved to index (0, 0) -- FFT order. A roll, so
    // the two pages hold the same values at the same radii and must radially
    // average to the same profile.
    ImageStack<float> rollToCorner(const ImageStack<float>& img) {
        const int n = static_cast<int>(img.dimension(1));
        const int c = n / 2;
        ImageStack<float> out(1, n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) out(0, i, j) = img(0, (i + c) % n, (j + c) % n);
        return out;
    }
} // namespace

// ---------------------------------------------------------------------------
// the format itself
// ---------------------------------------------------------------------------

TEST_CASE("loadOTF de-interleaves real/imag columns, with normalisation switched off", "[otf]") {
    // THIS CASE USED TO ASSERT d(0, 0, 0) == Cplx(1, 2) on a (1, 2, 4) stack
    // under the plain loadOTF(file, dkr, dkz), and that assertion could not
    // survive this reader gaining a scale: the loader now divides every order
    // by order 0's kr = kz = 0 sample when a table is not already at that
    // reference, so the first sample of ANY table that is loaded normalised
    // is 1 by construction and tells you nothing about the de-interleaving.
    //
    // The contract is therefore split in two, deliberately: this case asserts
    // the de-interleaving EXACTLY with OtfNormalization::AsStored, which is
    // the mode whose promise is "the file's own numbers", and the cases below
    // assert the scale separately. The fixture is also no longer 2 radial
    // samples wide -- 2 is not a shape an OTF is stored in, and the reader
    // now says so.
    const int nkr = 16, nzotf = 4;
    ImageStack<float> raw(1, nkr, 2 * nzotf);
    for (int ir = 0; ir < nkr; ++ir)
        for (int iz = 0; iz < nzotf; ++iz) {
            raw(0, ir, 2 * iz) = static_cast<float>(100 * ir + iz);        // real
            raw(0, ir, 2 * iz + 1) = static_cast<float>(-(100 * ir + iz)); // imaginary
        }

    test::TempFile tf("otf_load", ".tif");
    writeTiffStack<float>(tf.str, raw);

    OtfLoadOptions opts;
    opts.normalization = OtfNormalization::AsStored;
    opts.hermitianTolerance = 0.0;   // this fixture is not a physical OTF
    OtfLoadReport rep;
    const OTFRadiallyAveraged otf = loadOTF(tf.str, /*dkrotf*/ 0.25, /*dkzotf*/ 0.5, opts, &rep);
    const auto& d = otf.data();

    REQUIRE(d.dimension(0) == 1);
    REQUIRE(d.dimension(1) == nkr);
    REQUIRE(d.dimension(2) == nzotf);
    for (int ir = 0; ir < nkr; ++ir)
        for (int iz = 0; iz < nzotf; ++iz) {
            INFO("kr " << ir << " kz " << iz);
            const double v = 100.0 * ir + iz;
            CHECK(d(0, ir, iz) == Cplx(v, -v));
        }
    CHECK(otf.dkrotf() == 0.25);
    CHECK(otf.dkzotf() == 0.5);
    CHECK(otf.normalizationDivisor() == 1.0);
    CHECK_FALSE(otf.atReferenceScale());   // its DC is 0, and AsStored left it there
    CHECK(rep.layout == OtfLayout::CudasireconRadial);
    CHECK(rep.samplingSource == OtfSamplingSource::ExplicitArgument);
    CHECK(rep.normalizationApplied == false);
    CHECK(anyNote(rep, "read as stored"));
}

TEST_CASE("loadOTF refuses a shape no OTF is stored in, and says what it got and wanted", "[otf]") {
    SECTION("an odd last axis cannot pair into real and imaginary parts") {
        ImageStack<float> raw(1, 16, 7);
        raw.setZero();
        test::TempFile tf("otf_odd", ".tif");
        writeTiffStack<float>(tf.str, raw);
        REQUIRE_THROWS_AS(loadOTF(tf.str, 1.0, 1.0), IoError);
        try {
            loadOTF(tf.str, 1.0, 1.0);
        } catch (const IoError& e) {
            const std::string m = e.what();
            CHECK_THAT(m, ContainsSubstring("(1, 16, 7)"));          // what it got
            CHECK_THAT(m, ContainsSubstring("(norders, nkr, 2 * nzotf)"));   // what it wanted
            CHECK_THAT(m, ContainsSubstring("EVEN last axis"));
        }
    }

    SECTION("two radial samples is not a radial OTF") {
        ImageStack<float> raw(1, 2, 4);
        raw.setZero();
        test::TempFile tf("otf_tiny", ".tif");
        writeTiffStack<float>(tf.str, raw);
        REQUIRE_THROWS_AS(loadOTF(tf.str, 1.0, 1.0), IoError);
        try {
            loadOTF(tf.str, 1.0, 1.0);
        } catch (const IoError& e) {
            CHECK_THAT(std::string(e.what()), ContainsSubstring("nkr >= 9"));
        }
    }

    SECTION("more pages than any SIM has orders") {
        ImageStack<float> raw(12, 16, 4);
        raw.setZero();
        test::TempFile tf("otf_many", ".tif");
        writeTiffStack<float>(tf.str, raw);
        REQUIRE_THROWS_AS(loadOTF(tf.str, 1.0, 1.0), IoError);
    }

    SECTION("fewer orders than the reconstruction resolves, named with the configuration") {
        // THE CHECK THE FORMAT TEST USED TO BE: dimension(2) % 2, and nothing
        // else. A 2-order table under a 5-phase configuration used to load
        // here and fail three stages later in SimReconstructor.
        SIMParameters p = isoar2Params();
        p.nphases = 5;   // 3 orders
        REQUIRE_THROWS_AS(loadOTF(dataFile("isoar2_sparse_OTF_488.tif"), p), IoError);
        try {
            loadOTF(dataFile("isoar2_sparse_OTF_488.tif"), p);
        } catch (const IoError& e) {
            const std::string m = e.what();
            CHECK_THAT(m, ContainsSubstring("holds 2 orders"));
            CHECK_THAT(m, ContainsSubstring("resolves 3"));
            CHECK_THAT(m, ContainsSubstring("nphases 5"));
        }
    }

    SECTION("a table whose kz axis is not Hermitian is not a radial OTF at all") {
        // This is the structural check that catches a misdecoded file: every
        // real makeotf table measures EXACTLY 0 here (asserted on four files
        // below), because radialft fills the negative kz half from the
        // positive one. A page of image data read as (nkr, 2 * nzotf)
        // measures of order 1.
        ImageStack<float> raw(1, 16, 64);
        for (int ir = 0; ir < 16; ++ir)
            for (int ic = 0; ic < 64; ++ic) raw(0, ir, ic) = static_cast<float>((ir * 37 + ic * 11) % 23);
        test::TempFile tf("otf_noherm", ".tif");
        writeTiffStack<float>(tf.str, raw);
        REQUIRE_THROWS_AS(loadOTF(tf.str, 1.0, 1.0), IoError);
        try {
            loadOTF(tf.str, 1.0, 1.0);
        } catch (const IoError& e) {
            const std::string m = e.what();
            CHECK_THAT(m, ContainsSubstring("not Hermitian"));
            CHECK_THAT(m, ContainsSubstring("as stored"));
        }
        // and the check is a tolerance, not a law of the reader
        OtfLoadOptions opts;
        opts.hermitianTolerance = 0.0;
        opts.normalization = OtfNormalization::AsStored;
        REQUIRE_NOTHROW(loadOTF(tf.str, 1.0, 1.0, opts, nullptr));
    }
}

// ---------------------------------------------------------------------------
// a plain 2D OTF image
// ---------------------------------------------------------------------------

TEST_CASE("loadOTF accepts a plain 2D OTF image by radially averaging it", "[otf]") {
    // What OpenSIM, SIM4codes and most 2D-SIM codebases distribute: one real
    // (N, N) page centred on DC. Before this, a 512 x 512 page of one loaded
    // SILENTLY as 1 order x 512 radial samples x 256 kz planes (9k.51).
    const int n = 64;
    test::TempFile tf("otf_2d", ".tif");
    writeTiffStack<float>(tf.str, otfImage(n, 1000.0));

    SIMParameters p = isoar2Params();   // 3 phases -> 2 orders
    OtfLoadReport rep;
    const OTFRadiallyAveraged otf = loadOTF(tf.str, p, OtfLoadOptions{}, &rep);
    const auto& d = otf.data();

    CHECK(rep.layout == OtfLayout::Plain2dImage);
    CHECK(rep.storedShape == std::array<int, 3>{1, n, n});
    REQUIRE(d.dimension(0) == 2);        // the configuration's orders, same profile in each
    REQUIRE(d.dimension(1) == n / 2 + 1);
    REQUIRE(d.dimension(2) == 1);        // one kz plane, as idealOTF's 2D table has
    CHECK(anyNote(rep, "radially averaged a plain 2D OTF image"));

    // the radial step is the page's own: 1/(dx * N), which is exactly what
    // dkr = 1/(dx * (nkr - 1) * 2) comes to for nkr = N/2 + 1
    CHECK_THAT(otf.dkrotf(), WithinRel(1.0 / (p.dx * n), 1e-12));

    // normalised, decaying, and identical in every order
    CHECK(otf.atReferenceScale());
    CHECK_THAT(d(0, 0, 0).real(), WithinAbs(1.0, 1e-9));
    CHECK_THAT(otf.normalizationDivisor(), WithinRel(1000.0, 1e-5));
    CHECK(rep.reference == OtfNormalizationReference::Order0Dc);
    for (Eigen::Index ir = 1; ir < d.dimension(1); ++ir) {
        INFO("kr " << ir);
        CHECK(std::abs(d(0, ir, 0)) <= std::abs(d(0, ir - 1, 0)));
        CHECK(d(1, ir, 0) == d(0, ir, 0));
        CHECK(d(0, ir, 0).imag() == 0.0);   // a 2D OTF image is a magnitude
    }

    SECTION("the same page in FFT order, DC at the corner, is read and reported as such") {
        test::TempFile shifted("otf_2d_fft", ".tif");
        writeTiffStack<float>(shifted.str, rollToCorner(otfImage(n, 1000.0)));
        OtfLoadReport r2;
        const OTFRadiallyAveraged o2 = loadOTF(shifted.str, p, OtfLoadOptions{}, &r2);
        CHECK(r2.layout == OtfLayout::Plain2dImage);
        CHECK(anyNote(r2, "FFT order"));
        for (Eigen::Index ir = 0; ir < o2.data().dimension(1); ++ir) {
            INFO("kr " << ir);
            CHECK_THAT(o2.data()(0, ir, 0).real(), WithinAbs(d(0, ir, 0).real(), 1e-9));
        }
    }

    SECTION("a square page whose maximum is neither centred nor at DC is refused") {
        ImageStack<float> img = otfImage(n, 1000.0);
        img(0, 8, 9) = 5000.0f;   // a bright something a third of the way out
        test::TempFile odd("otf_2d_off", ".tif");
        writeTiffStack<float>(odd.str, img);
        REQUIRE_THROWS_AS(loadOTF(odd.str, p), IoError);
        try {
            loadOTF(odd.str, p);
        } catch (const IoError& e) {
            const std::string m = e.what();
            CHECK_THAT(m, ContainsSubstring("row 8, column 9"));
            CHECK_THAT(m, ContainsSubstring("centred on DC"));
        }
    }

    SECTION("and the 2D path can be switched off, which brings the structural check down on it") {
        OtfLoadOptions opts;
        opts.accept2dImage = false;
        REQUIRE_THROWS_AS(loadOTF(tf.str, p, opts, nullptr), IoError);
    }
}

// ---------------------------------------------------------------------------
// the scale
// ---------------------------------------------------------------------------

TEST_CASE("loadOTF normalises to one reference and never twice", "[otf][data]") {
    SIMParameters p;   // config.txt: 5 phases -> 3 orders, xyres 0.08, zresPSF 0.125
    p.dx = p.dy = 0.08;
    p.dz = 0.125;
    p.dz_psf = 0.125;
    const std::string file = dataFile("otf.tif");

    SECTION("a table already at the reference is left EXACTLY as stored") {
        // This is what keeps the agreement with cudasirecon on raw.tif, which
        // stands at 1.6e-6 of peak (9k.48, 9k.49): makeotf's rescale() has
        // already divided this table by order 0's DC, so Auto must change
        // nothing at all. Bit-for-bit against AsStored, not "close".
        OtfLoadOptions stored;
        stored.normalization = OtfNormalization::AsStored;
        OtfLoadReport repAuto, repStored;
        const OTFRadiallyAveraged a = loadOTF(file, p, OtfLoadOptions{}, &repAuto);
        const OTFRadiallyAveraged s = loadOTF(file, p, stored, &repStored);
        REQUIRE(a.data().dimension(0) == s.data().dimension(0));
        REQUIRE(a.data().dimension(1) == s.data().dimension(1));
        REQUIRE(a.data().dimension(2) == s.data().dimension(2));
        for (Eigen::Index o = 0; o < a.data().dimension(0); ++o)
            for (Eigen::Index ir = 0; ir < a.data().dimension(1); ++ir)
                for (Eigen::Index iz = 0; iz < a.data().dimension(2); ++iz) {
                    INFO("order " << o << " kr " << ir << " kz " << iz);
                    REQUIRE(a.data()(o, ir, iz) == s.data()(o, ir, iz));
                }
        CHECK(a.normalizationDivisor() == 1.0);
        CHECK(repAuto.normalizationApplied == false);
        CHECK(repAuto.reference == OtfNormalizationReference::Order0Dc);
        CHECK_THAT(repAuto.referenceValue, WithinAbs(1.0, 1e-6));
        CHECK(anyNote(repAuto, "already at the reference scale"));
        CHECK(a.atReferenceScale());
    }

    SECTION("the fixorigin line fit is reported, and is NOT what makeotf's files are on") {
        // The design this stage implements proposed the line fit as the
        // invariant ("order 0 extrapolated to kr = 0 equals 1"). Measured on
        // this file, cudasirecon's own test OTF, the kz-summed fixorigin
        // extrapolation is 1.0300, not 1: makeotf's fixorigin is a repair of
        // the samples below kr = kx1 and is OFF by default (radialft.cpp:104
        // sets interpkr to 0 and :401 gates on interpkr[0] > 0), so no
        // shipped table is on that scale. Dividing by it would move this
        // table by 3% and break the cudasirecon agreement, which is why the
        // DC is the reference wherever it is usable.
        OtfLoadReport rep;
        loadOTF(file, p, OtfLoadOptions{}, &rep);
        CHECK_THAT(rep.lineFitValue, WithinAbs(1.030027, 1e-4));
        CHECK(rep.reference == OtfNormalizationReference::Order0Dc);
        CHECK(rep.normalizationApplied == false);
    }

    SECTION("Reference forces the division even on a table that says it is normalised") {
        OtfLoadOptions forced;
        forced.normalization = OtfNormalization::Reference;
        OtfLoadReport rep;
        const OTFRadiallyAveraged f = loadOTF(file, p, forced, &rep);
        CHECK(rep.normalizationApplied == true);
        CHECK_THAT(f.normalizationDivisor(), WithinAbs(0.99999994, 1e-9));
        CHECK_THAT(f.data()(0, 0, 0).real(), WithinAbs(1.0, 1e-12));
    }
}

TEST_CASE("loadOTF puts an unnormalised table on the reference scale, ratios intact", "[otf]") {
    // The defect this fixes, measured in 9k.51: a loaded table was used on
    // whatever scale its file was on, while idealOTF normalises itself, so
    // one calibration stack gave a fitted modulation amplitude of 2.76 where
    // the truth was 0.85 (and 0.17 theoretical, 0.64 rescaled). otfcutoff is
    // an ABSOLUTE threshold and the Wiener constant is added to |OTF|^2, so a
    // table scaled by s is not a scaled table, it is a different filter.
    const int nkr = 33, nzotf = 8, norders = 2;
    const double scale = 7500.0, depth = 0.3;
    ImageStack<float> raw(norders, nkr, 2 * nzotf);
    raw.setZero();
    for (int o = 0; o < norders; ++o)
        for (int ir = 0; ir < nkr; ++ir)
            for (int iz = 0; iz < nzotf; ++iz) {
                // Hermitian in kz by construction (kz and -kz equal and real),
                // decaying in kr, and order 1 is `depth` times order 0.
                const int kz = std::min(iz, nzotf - iz);
                const double v = scale * (o == 0 ? 1.0 : depth) * std::exp(-0.08 * ir - 0.3 * kz);
                raw(o, ir, 2 * iz) = static_cast<float>(v);
            }
    test::TempFile tf("otf_unnorm", ".tif");
    writeTiffStack<float>(tf.str, raw);

    SIMParameters p = isoar2Params();
    OtfLoadReport rep;
    const OTFRadiallyAveraged otf = loadOTF(tf.str, p, OtfLoadOptions{}, &rep);
    CHECK(rep.normalizationApplied == true);
    CHECK(rep.reference == OtfNormalizationReference::Order0Dc);
    CHECK_THAT(rep.referenceValue, WithinRel(scale, 1e-5));
    CHECK_THAT(otf.data()(0, 0, 0).real(), WithinAbs(1.0, 1e-9));
    CHECK(otf.atReferenceScale());
    // the modulation depth is the ratio between orders, and it survives
    CHECK_THAT(std::abs(otf.data()(1, 0, 0)) / std::abs(otf.data()(0, 0, 0)), WithinRel(depth, 1e-5));
    CHECK_THAT(otf.data()(1, 0, 0).real(), WithinRel(depth, 1e-5));
    CHECK(anyNote(rep, "ratios between orders"));

    SECTION("a table whose DC has been zeroed falls back to the fixorigin line fit, and says so") {
        ImageStack<float> zeroed = raw;
        for (int o = 0; o < norders; ++o)
            for (int iz = 0; iz < nzotf; ++iz) zeroed(o, 0, 2 * iz) = 0.0f;   // the whole kr = 0 column
        test::TempFile zf("otf_nodc", ".tif");
        writeTiffStack<float>(zf.str, zeroed);
        OtfLoadReport zrep;
        const OTFRadiallyAveraged z = loadOTF(zf.str, p, OtfLoadOptions{}, &zrep);
        CHECK(zrep.reference == OtfNormalizationReference::Order0LineFit);
        CHECK(zrep.normalizationApplied == true);
        CHECK(anyNote(zrep, "fixorigin line fit"));
        // the fit extrapolates the kz-summed profile of this exponential to
        // kr = 0, so it lands near the (zeroed) column's own kz sum
        CHECK(zrep.referenceValue > 0.0);
        CHECK_THAT(z.normalizationDivisor(), WithinRel(zrep.referenceValue, 1e-12));
        // and order 1 is still `depth` times order 0 at every sample
        for (int ir = 1; ir < nkr; ++ir) {
            INFO("kr " << ir);
            CHECK_THAT(z.data()(1, ir, 0).real() / z.data()(0, ir, 0).real(), WithinRel(depth, 1e-5));
        }
    }
}

// ---------------------------------------------------------------------------
// sampling
// ---------------------------------------------------------------------------

TEST_CASE("loadOTF takes its sampling from the file, then a sidecar, then the parameters", "[otf][data]") {
    SIMParameters p = isoar2Params();

    SECTION("an MRC OTF states its own steps in its cell lengths") {
        OtfLoadReport rep;
        const OTFRadiallyAveraged otf = loadOTF(dataFile("isoar2_488_3d_otf.mrc"), p, OtfLoadOptions{}, &rep);
        CHECK(rep.samplingSource == OtfSamplingSource::FileCellDimensions);
        // measured: cell = (0.09900989, 0.02297794, 0) = (dkz, dkr)
        CHECK_THAT(otf.dkrotf(), WithinAbs(0.02297794, 1e-8));
        CHECK_THAT(otf.dkzotf(), WithinAbs(0.09900989, 1e-8));
        // which is 1/(0.085 * 512) and 1/(0.1 * 101): the file remembers the
        // pixel sizes it was measured at, to 7 digits
        CHECK_THAT(otf.dkrotf(), WithinRel(1.0 / (0.085 * 512.0), 1e-6));
        CHECK_THAT(otf.dkzotf(), WithinRel(1.0 / (0.1 * 101.0), 1e-6));
        CHECK(anyNote(rep, "cell lengths"));
    }

    SECTION("a TIFF beside a .toml takes the sidecar's steps") {
        // the real sparse-field OTF, whose sidecar carries the pixel size and
        // z step from that acquisition's own settings file
        OtfLoadReport rep;
        const OTFRadiallyAveraged otf = loadOTF(dataFile("isoar2_sparse_OTF_488.tif"), p, OtfLoadOptions{}, &rep);
        CHECK(rep.samplingSource == OtfSamplingSource::Sidecar);
        CHECK_THAT(rep.sidecarPath, ContainsSubstring("isoar2_sparse_OTF_488.toml"));
        CHECK_THAT(otf.dkrotf(), WithinRel(1.0 / (0.08552631578947369 * 128.0), 1e-12));
        CHECK_THAT(otf.dkzotf(), WithinRel(1.0 / (0.1 * 101.0), 1e-12));
        // and the sidecar is why that matters: deriving dkr from the config's
        // rounded xyres = 0.085 instead would be 0.62% high
        const double derived = 1.0 / (p.dx * 128.0);
        CHECK_THAT(100.0 * (derived - otf.dkrotf()) / otf.dkrotf(), WithinAbs(0.62, 0.01));
    }

    SECTION("the <file>.toml spelling is found too, and is looked for first") {
        test::TempFile tf("otf_side", ".tif");
        ImageStack<float> raw(1, 16, 8);
        raw.setZero();
        for (int ir = 0; ir < 16; ++ir) raw(0, ir, 0) = static_cast<float>(16 - ir);
        writeTiffStack<float>(tf.str, raw);
        Scratch side(tf.str + ".toml");
        {
            std::ofstream out(side.p);
            out << "[sampling]\ndkr = 0.5\ndkz = 0.125\n";
        }
        OtfLoadOptions opts;
        opts.normalization = OtfNormalization::AsStored;
        OtfLoadReport rep;
        const OTFRadiallyAveraged otf = loadOTF(tf.str, isoar2Params(), opts, &rep);
        CHECK(rep.samplingSource == OtfSamplingSource::Sidecar);
        CHECK(rep.sidecarPath == side.p.string());
        CHECK(otf.dkrotf() == 0.5);
        CHECK(otf.dkzotf() == 0.125);
    }

    SECTION("a sidecar may give the measurement's pixel sizes instead of the steps") {
        test::TempFile tf("otf_side_px", ".tif");
        ImageStack<float> raw(1, 33, 8);
        raw.setZero();
        for (int ir = 0; ir < 33; ++ir) raw(0, ir, 0) = static_cast<float>(33 - ir);
        writeTiffStack<float>(tf.str, raw);
        Scratch side(tf.str + ".toml");
        {
            std::ofstream out(side.p);
            out << "[psf]\nxyres = 0.1\nzres = 0.2\n";
        }
        OtfLoadReport rep;
        const OTFRadiallyAveraged otf = loadOTF(tf.str, isoar2Params(), OtfLoadOptions{}, &rep);
        CHECK(rep.samplingSource == OtfSamplingSource::Sidecar);
        CHECK_THAT(otf.dkrotf(), WithinRel(1.0 / (0.1 * 32.0 * 2.0), 1e-12));
        CHECK_THAT(otf.dkzotf(), WithinRel(1.0 / (0.2 * 4.0), 1e-12));
    }

    SECTION("with neither, the steps are derived from this run's pixel sizes and the report says so") {
        OtfLoadOptions opts;
        opts.readSidecar = false;
        OtfLoadReport rep;
        const OTFRadiallyAveraged otf = loadOTF(dataFile("isoar2_sparse_OTF_488.tif"), p, opts, &rep);
        CHECK(rep.samplingSource == OtfSamplingSource::DerivedFromParameters);
        CHECK_THAT(otf.dkrotf(), WithinRel(1.0 / (p.dx * 64.0 * 2.0), 1e-12));
        CHECK_THAT(otf.dkzotf(), WithinRel(1.0 / (p.dz_psf * 101.0), 1e-12));
        CHECK(anyNote(rep, "DERIVED"));
        CHECK(anyNote(rep, "right only if the OTF was measured at those pixel sizes"));
    }

    SECTION("an unreadable sidecar is a refusal, not a silent fallback") {
        test::TempFile tf("otf_side_bad", ".tif");
        ImageStack<float> raw(1, 16, 8);
        raw.setZero();
        writeTiffStack<float>(tf.str, raw);
        Scratch side(tf.str + ".toml");
        {
            std::ofstream out(side.p);
            out << "[sampling\ndkr = ";
        }
        REQUIRE_THROWS_AS(loadOTF(tf.str, p), IoError);
    }
}

// ---------------------------------------------------------------------------
// the real files, before and after
// ---------------------------------------------------------------------------

TEST_CASE("loadOTF reads cudasirecon's radially averaged OTF TIFF as cudasirecon reads it", "[otf][data]") {
    // tests/data/otf.tif is cudasirecon's test OTF, written by its makeotf
    // (radialft.cpp: a CImg of width nz*2, height nx/2+1, depth norders) as
    // 3 pages of 129 rows x 130 float32 columns. cudasirecon reads it back
    // (determine_otf_dimensions, otfRA=1 on a 3D stack) as nzotf = width / 2
    // = 65 complex kz samples, (re, im) interleaved along the columns,
    // nxotf = height = 129 radial samples, one page per order, with
    // dkzotf = 1 / (zresPSF * nzotf) and dkrotf = 1 / (xyres * (nxotf - 1) * 2);
    // its kernel indexes otf[ir * nzotf + iz] with kz in FFT order (negative
    // kz wraps to the top). Its log for this file and config.txt reads
    // "nzotf=65, dkzotf=0.123077, nxotf=129, nyotf=1, dkrotf=0.048828".
    const std::string file = dataFile("otf.tif");
    SIMParameters p;   // config.txt: xyres=0.08 zres=0.125 zresPSF=0.125
    p.dx = p.dy = 0.08;
    p.dz = 0.125;
    p.dz_psf = 0.125;
    OtfLoadReport rep;
    const OTFRadiallyAveraged otf = loadOTF(file, p, OtfLoadOptions{}, &rep);
    const auto& d = otf.data();
    REQUIRE(d.dimension(0) == 3);     // orders 0, 1, 2
    REQUIRE(d.dimension(1) == 129);   // kr
    REQUIRE(d.dimension(2) == 65);    // kz, complex
    CHECK(rep.layout == OtfLayout::CudasireconRadial);
    CHECK(rep.samplingSource == OtfSamplingSource::DerivedFromParameters);   // a makeotf TIFF says nothing
    CHECK(rep.kzRolled == false);
    CHECK(rep.hermitianErrorAsStored == 0.0);   // exactly: radialft forces it
    CHECK_THAT(otf.dkrotf(), WithinAbs(0.048828, 1e-6));
    CHECK_THAT(otf.dkzotf(), WithinAbs(0.123077, 1e-6));
    CHECK_THAT(otf.dkrotf(), WithinRel(1.0 / (0.08 * 128.0 * 2.0), 1e-12));
    CHECK_THAT(otf.dkzotf(), WithinRel(1.0 / (0.125 * 65.0), 1e-12));

    // The complex samples are the file's (re, im) column pairs: page = order, row = kr.
    const auto raw = readTiffStack<float>(file);
    REQUIRE(raw.dimension(0) == 3);
    REQUIRE(raw.dimension(1) == 129);
    REQUIRE(raw.dimension(2) == 130);
    for (const auto& [o, ir, iz] : std::vector<std::array<Eigen::Index, 3>>{{0, 0, 0}, {0, 0, 1}, {1, 0, 9}, {2, 5, 7}, {2, 128, 64}}) {
        INFO("order " << o << " kr " << ir << " kz " << iz);
        CHECK(d(o, ir, iz) == Cplx(raw(o, ir, 2 * iz), raw(o, ir, 2 * iz + 1)));
    }

    auto peak = [&](Eigen::Index order) {
        std::array<Eigen::Index, 2> at{0, 0};
        double best = -1.0;
        for (Eigen::Index ir = 0; ir < d.dimension(1); ++ir)
            for (Eigen::Index iz = 0; iz < d.dimension(2); ++iz) {
                const double m = std::abs(d(order, ir, iz));
                if (m > best) {
                    best = m;
                    at = {ir, iz};
                }
            }
        return std::pair{best, at};
    };

    // Order 0 is the widefield OTF, normalised: its peak is 1 at kr = kz = 0
    // (to float32 rounding: the file stores 0.99999994, which is inside the
    // tolerance Auto calls "already normalised", so the stored value is what
    // is here).
    const auto [peak0, at0] = peak(0);
    CHECK_THAT(d(0, 0, 0).real(), WithinAbs(1.0, 1e-6));
    CHECK_THAT(d(0, 0, 0).imag(), WithinAbs(0.0, 1e-6));
    CHECK_THAT(peak0, WithinAbs(1.0, 1e-6));
    CHECK(at0 == std::array<Eigen::Index, 2>{0, 0});
    // kz is in FFT order: sample 64 is kz = -1 and mirrors sample 1 in every order.
    CHECK_THAT(std::abs(d(0, 0, 64)), WithinRel(std::abs(d(0, 0, 1)), 1e-4));
    CHECK_THAT(std::abs(d(1, 0, 64)), WithinRel(std::abs(d(1, 0, 1)), 1e-4));
    CHECK_THAT(std::abs(d(2, 0, 64)), WithinRel(std::abs(d(2, 0, 1)), 1e-4));
    CHECK_THAT(std::abs(d(0, 0, 1)), WithinAbs(0.1219, 1e-3));
    // Its support is a fraction of the table, which is what makes the steps
    // matter: at kz = 0 the magnitude falls under cudasirecon's default
    // otfcutoff (0.006) at kr index 69 (3.37 cycles/um at this dkr) and along
    // kz at index 5 (0.62 cycles/um). Read with the 0.03125 step of a 0.125 um
    // pixel, the same table would end at 2.2 cycles/um, short of the side
    // bands of a 0.2035 um pattern (order 2 at 4.9 cycles/um). Those indices
    // are also why the cutoff has to be read against a normalised table: they
    // are where |OTF| crosses 0.006 OF ORDER 0's DC, and only a table whose
    // DC is 1 makes the absolute number mean that.
    CHECK(std::abs(d(0, 68, 0)) > 0.006);
    CHECK(std::abs(d(0, 69, 0)) < 0.006);
    CHECK(std::abs(d(0, 0, 4)) > 0.006);
    CHECK(std::abs(d(0, 0, 5)) < 0.006);
    CHECK_THAT(otf.dkrotf() * 69.0, WithinAbs(3.37, 0.01));

    // Order 1 is the axial side band of three-beam SIM: it peaks away from kz = 0.
    const auto [peak1, at1] = peak(1);
    CHECK_THAT(peak1, WithinAbs(0.164666, 1e-5));
    CHECK(at1 == std::array<Eigen::Index, 2>{0, 9});
    CHECK(std::abs(d(1, 0, 0)) < 0.02);

    // Order 2 peaks at DC, at 0.225 of the widefield peak.
    const auto [peak2, at2] = peak(2);
    CHECK_THAT(peak2, WithinAbs(0.22525, 1e-5));
    CHECK(at2 == std::array<Eigen::Index, 2>{0, 0});

    // The explicit-step overload decodes the same table.
    const OTFRadiallyAveraged explicitSteps = loadOTF(file, 0.048828, 0.123077);
    REQUIRE(explicitSteps.data().dimension(2) == 65);
    CHECK(explicitSteps.data()(2, 5, 7) == d(2, 5, 7));
    CHECK(explicitSteps.dkrotf() == 0.048828);

    INFO(rep.summary());
    CHECK_THAT(rep.summary(), ContainsSubstring("cudasirecon radial table"));
}

TEST_CASE("loadOTF reads the DeltaVision container makeotf writes (otf.dv)", "[otf][data]") {
    // otf.dv is what cudasirecon's makeotf writes beside otf.tif: the same
    // 3 orders x 129 radial samples x 65 complex kz samples, mode 4 (float32
    // complex), so its rows are the (re, im) pairs loadOTF de-interleaves.
    // The header's cell lengths are the table's own steps, dkzotf then
    // dkrotf, which is how cudasirecon recovers them from a .dv OTF -- and
    // now how loadOTF does, without being told.
    const std::string data = dataDir();
    const MrcInfo head = inspectMrc(data + "/otf.dv");
    CHECK(head.mode == 4);
    CHECK(head.complex);
    CHECK(head.width == 130);      // 65 complex pairs
    CHECK(head.height == 129);
    CHECK(head.sections == 3);
    CHECK_THAT(static_cast<double>(head.cell[0]), WithinAbs(0.123077, 1e-6));   // dkzotf
    CHECK_THAT(static_cast<double>(head.cell[1]), WithinAbs(0.048828, 1e-6));   // dkrotf

    SIMParameters p;
    p.dx = p.dy = 0.08;
    p.dz = 0.125;
    p.dz_psf = 0.125;
    OtfLoadReport rep;
    const OTFRadiallyAveraged otf = loadOTF(data + "/otf.dv", p, OtfLoadOptions{}, &rep);
    const auto& d = otf.data();
    REQUIRE(d.dimension(0) == 3);
    REQUIRE(d.dimension(1) == 129);
    REQUIRE(d.dimension(2) == 65);
    CHECK(rep.samplingSource == OtfSamplingSource::FileCellDimensions);
    CHECK_THAT(otf.dkrotf(), WithinAbs(0.048828, 1e-6));
    CHECK_THAT(otf.dkzotf(), WithinAbs(0.123077, 1e-6));
    CHECK(rep.kzRolled == false);
    CHECK(rep.hermitianErrorAsStored == 0.0);
    // Order 0 is normalised at the origin, and in this container the DC
    // sample is exactly 1 (otf.tif's is 0.99999994), so Auto leaves it.
    CHECK(d(0, 0, 0) == Cplx(1.0, 0.0));
    CHECK(rep.normalizationApplied == false);
    CHECK_THAT(d(0, 0, 1).real(), WithinAbs(0.137122, 1e-6));
    // This is a different measurement from otf.tif, not the same table in
    // another container: its imaginary parts are ~0 throughout, where
    // otf.tif's are not.
    double maxImagDv = 0.0, maxImagTif = 0.0;
    const OTFRadiallyAveraged tif = loadOTF(data + "/otf.tif", head.cell[1], head.cell[0]);
    REQUIRE(tif.data().dimension(0) == 3);
    REQUIRE(tif.data().dimension(1) == 129);
    REQUIRE(tif.data().dimension(2) == 65);
    for (Eigen::Index o = 0; o < 3; ++o)
        for (Eigen::Index ir = 0; ir < 129; ++ir)
            for (Eigen::Index iz = 0; iz < 65; ++iz) {
                maxImagDv = std::max(maxImagDv, std::abs(d(o, ir, iz).imag()));
                maxImagTif = std::max(maxImagTif, std::abs(tif.data()(o, ir, iz).imag()));
            }
    CHECK(maxImagDv < 1e-6);
    CHECK(maxImagTif > 0.01);
    CHECK_THAT(tif.data()(0, 0, 0).real(), WithinAbs(1.0, 1e-6));
}

TEST_CASE("the iSOAR2 3D OTF in MRC form has its kz axis rotated by one, and is rolled back", "[otf][data]") {
    // isoar2_488_3d_otf.mrc is the user's own
    // 488_20px_2beam_3phase_PSF_3D_OTF.mrc (md5
    // 578ba01e1d1196e23147e18594f4e51c), 2 orders x 257 kr x 101 complex kz,
    // mode 4, no DeltaVision magic. ITS kz AXIS IS STORED DC-LAST, and three
    // independent facts say so rather than one guess:
    //   * Hermitian error 0.45 as stored, EXACTLY 0 shifted one plane later,
    //     where every other makeotf table measures 0 as stored;
    //   * order 0's maximum, which is the normalisation DC, sits at stored kz
    //     index 100, i.e. index 0 after the roll;
    //   * after the roll the kr = 0 column is zero at every kz but kz = 0,
    //     which is the missing cone of a 3D OTF, and the kz = 0 profile
    //     decays from 1 (1, 0.833, 0.803, 0.570, ...) instead of rising from 0.
    // Rolling it is reported, never silent: the report carries kzRolled, the
    // roll, and the Hermitian error before and after.
    SIMParameters p = isoar2Params();
    OtfLoadReport rep;
    const OTFRadiallyAveraged otf = loadOTF(dataFile("isoar2_488_3d_otf.mrc"), p, OtfLoadOptions{}, &rep);
    const auto& d = otf.data();
    REQUIRE(d.dimension(0) == 2);
    REQUIRE(d.dimension(1) == 257);
    REQUIRE(d.dimension(2) == 101);
    CHECK(rep.storedShape == std::array<int, 3>{2, 257, 202});   // interleaved complex columns

    CHECK(rep.kzRolled == true);
    CHECK(rep.kzRoll == 1);
    CHECK_THAT(rep.hermitianErrorAsStored, WithinAbs(0.45, 0.05));
    CHECK(rep.hermitianErrorApplied < 1e-12);
    CHECK(anyNote(rep, "kz axis was rotated"));
    CHECK(anyNote(rep, "detected, not assumed"));

    // DC first after the roll, and the table is already at the reference
    CHECK(d(0, 0, 0) == Cplx(1.0, 0.0));
    CHECK(otf.atReferenceScale());
    CHECK(rep.normalizationApplied == false);
    CHECK_THAT(d(0, 1, 0).real(), WithinAbs(0.83333, 1e-4));
    CHECK_THAT(d(0, 2, 0).real(), WithinAbs(0.80280, 1e-4));
    // the side band's own DC is the modulation depth, and it is untouched
    CHECK_THAT(d(1, 0, 0).real(), WithinAbs(0.1275316, 1e-6));
    // the missing cone: the kr = 0 column is zero away from kz = 0
    for (Eigen::Index iz = 1; iz < 101; ++iz) {
        INFO("kz " << iz);
        CHECK(std::abs(d(0, 0, iz)) == 0.0);
    }

    SECTION("the roll can be stated, and a statement that is wrong is refused") {
        // Saying DC is first does not make it first: with the roll
        // suppressed the table is not Hermitian, and that is now a refusal
        // rather than a silent misread.
        const std::string file = dataFile("isoar2_488_3d_otf.mrc");
        OtfLoadOptions stateFirst;
        stateFirst.kzOrigin = OtfKzOrigin::DcFirst;
        REQUIRE_THROWS_AS(loadOTF(file, p, stateFirst, nullptr), IoError);
        try {
            loadOTF(file, p, stateFirst, nullptr);
        } catch (const IoError& e) {
            CHECK_THAT(std::string(e.what()), ContainsSubstring("not Hermitian in kz"));
        }

        // WHAT THIS FILE USED TO LOAD AS, reproduced by switching both new
        // checks off: the kz axis a plane out, so the sample the whole
        // reconstruction reads as the DC is the missing-cone zero of kz = 1,
        // and no normalisation reference at all.
        OtfLoadOptions beforeThisChange;
        beforeThisChange.kzOrigin = OtfKzOrigin::DcFirst;
        beforeThisChange.hermitianTolerance = 0.0;
        beforeThisChange.normalization = OtfNormalization::AsStored;
        OtfLoadReport r2;
        const OTFRadiallyAveraged o2 = loadOTF(file, p, beforeThisChange, &r2);
        CHECK(r2.kzRolled == false);
        CHECK(o2.data()(0, 0, 0) == Cplx(0.0, 0.0));
        CHECK_FALSE(o2.atReferenceScale());

        OtfLoadOptions stateLast;
        stateLast.kzOrigin = OtfKzOrigin::DcLast;
        OtfLoadReport r3;
        const OTFRadiallyAveraged o3 = loadOTF(file, p, stateLast, &r3);
        CHECK(r3.kzRolled == true);
        CHECK(anyNote(r3, "asked for: DC last"));
        CHECK(o3.data()(0, 0, 0) == Cplx(1.0, 0.0));
    }
}

TEST_CASE("the two iSOAR2 OTF TIFFs load as what they are, scale included", "[otf][data]") {
    SIMParameters p = isoar2Params();

    SECTION("the sparse bead field's OTF_488.tif is a clean 2-order table at the reference") {
        // The colleague's makeotf output from the SPARSE bead field (the
        // stack the user asked measurements to come from), md5
        // 162fcd263c7db0d8dd52bb4e29df9eb7: 2 orders x 65 kr x 101 kz.
        OtfLoadReport rep;
        const OTFRadiallyAveraged otf = loadOTF(dataFile("isoar2_sparse_OTF_488.tif"), p, OtfLoadOptions{}, &rep);
        const auto& d = otf.data();
        REQUIRE(d.dimension(0) == 2);
        REQUIRE(d.dimension(1) == 65);
        REQUIRE(d.dimension(2) == 101);
        CHECK(rep.hermitianErrorAsStored == 0.0);
        CHECK(rep.kzRolled == false);
        CHECK(d(0, 0, 0) == Cplx(1.0, 0.0));
        CHECK(rep.normalizationApplied == false);
        // a monotone decaying kz = 0 profile: 1, 0.353, 0.235, 0.194, ...
        CHECK_THAT(d(0, 1, 0).real(), WithinAbs(0.35309, 1e-4));
        CHECK_THAT(d(0, 2, 0).real(), WithinAbs(0.23490, 1e-4));
        for (Eigen::Index ir = 2; ir < 20; ++ir) {
            INFO("kr " << ir);
            CHECK(d(0, ir, 0).real() < d(0, ir - 1, 0).real());
        }
        // and the modulation depth it measured
        CHECK_THAT(d(1, 0, 0).real(), WithinAbs(0.2096378, 1e-6));
        // the fixorigin line fit is 1.45 here, which is the second file
        // saying the line fit is not the scale makeotf leaves tables on
        CHECK_THAT(rep.lineFitValue, WithinAbs(1.4524, 1e-3));
    }

    SECTION("488OTF.tif loads, and its scale rests on a sample sitting among noise") {
        // The user's own 488OTF.tif (md5 89b2d0c0aefa156940ed3b1587aac65d),
        // 2 orders x 257 kr x 201 kz, made from 488nm_512px.tif. Its order 0
        // kz = 0 profile runs 1, 0.964, 0.738, 3.779, 0.809, -1.116, -1.044:
        // the kr = 3 and kr = 5 samples are the file's own max and min tags,
        // so makeotf set this table's scale from the DC sample next to them.
        // Nothing can repair that in a reader -- renormalising would only
        // move the arbitrariness around -- so the load reports it.
        OtfLoadReport rep;
        const OTFRadiallyAveraged otf = loadOTF(dataFile("isoar2_488OTF.tif"), p, OtfLoadOptions{}, &rep);
        const auto& d = otf.data();
        REQUIRE(d.dimension(0) == 2);
        REQUIRE(d.dimension(1) == 257);
        REQUIRE(d.dimension(2) == 201);
        CHECK(rep.hermitianErrorAsStored == 0.0);
        CHECK(rep.kzRolled == false);
        CHECK(rep.samplingSource == OtfSamplingSource::DerivedFromParameters);
        CHECK(d(0, 0, 0) == Cplx(1.0, 0.0));
        CHECK(rep.normalizationApplied == false);
        CHECK_THAT(d(0, 3, 0).real(), WithinAbs(3.77939, 1e-4));
        CHECK_THAT(d(0, 5, 0).real(), WithinAbs(-1.11599, 1e-4));
        CHECK(anyNote(rep, "sitting among noise"));
        // the line fit is no refuge either: 2.12 on this file
        CHECK_THAT(rep.lineFitValue, WithinAbs(2.1153, 1e-3));
    }
}
