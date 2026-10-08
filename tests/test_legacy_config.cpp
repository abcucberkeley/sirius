// The legacy cudasirecon config: parsing it, and mapping it onto SIMParameters.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <filesystem>
#include <fstream>
#include <string>

#include "sirius/legacy_config.hpp"
#include "sirius/sim_parameters.hpp"

#include "temp_path.hpp"

using namespace sirius;
using Catch::Approx;

namespace {

    // RAII temp file: writes `contents` (if any) and removes the file on scope exit.
    struct TempFile {
        std::filesystem::path path;

        explicit TempFile(const std::string& suffix, const std::string& contents = "")
            : path(test::uniqueTempPath("legacy", suffix.c_str())) {
            if (!contents.empty()) {
                std::ofstream f(path);
                f << contents;
            }
        }
        ~TempFile() {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
        std::string str() const { return path.string(); }
    };

    // The example config from the legacy cudasirecon docs.
    const char* kExampleConfig =
        "nimm=1.515\n"
        "fastSI=0\n"
        "background=0\n"
        "wiener=0.001\n"
        "k0angles=0.804300,1.8555,-0.238800\n"
        "ls=0.2035\n"
        "ndirs=3\n"
        "nphases=5\n"
        "na=1.42\n"
        "otfRA=1\n"
        "dampenOrder0=1\n"
        "xyres=0.08\n"
        "zres=0.125\n"
        "zresPSF=0.125\n";

} // namespace

// --------------------------------------------------------------------------
// Legacy config parsing
// --------------------------------------------------------------------------

TEST_CASE("loadLegacyConfig parses the example config", "[legacy]") {
    TempFile tf(".cfg", kExampleConfig);
    LegacyReconConfig c = loadLegacyConfig(tf.str());

    REQUIRE(c.nimm == Approx(1.515f));
    REQUIRE(c.bFastSIM == false);
    REQUIRE(c.constbkgd == Approx(0.0f));
    REQUIRE(c.wiener == Approx(0.001f));
    REQUIRE(c.linespacing == Approx(0.2035f));
    REQUIRE(c.ndirs == 3);
    REQUIRE(c.nphases == 5);
    REQUIRE(c.na == Approx(1.42f));
    REQUIRE(c.bRadAvgOTF == true);     // otfRA=1
    REQUIRE(c.bDampenOrder0 == true);  // dampenOrder0=1
    REQUIRE(c.dxy == Approx(0.08f));
    REQUIRE(c.dz == Approx(0.125f));
    REQUIRE(c.dzPSF == Approx(0.125f));
    // to the file's precision, not a float's: the SIM step reconstructs with
    // these and derives the output voxel and the OTF sampling from them
    CHECK(c.dxy == 0.08);

    REQUIRE(c.k0angles.size() == 3);
    REQUIRE(c.k0angles[0] == Approx(0.804300f));
    REQUIRE(c.k0angles[1] == Approx(1.8555f));
    REQUIRE(c.k0angles[2] == Approx(-0.238800f));
}

TEST_CASE("loadLegacyConfig ignores comments and blank lines", "[legacy]") {
    TempFile tf(".cfg", "# a comment\n\n; another comment\nna=1.3\n");
    LegacyReconConfig c = loadLegacyConfig(tf.str());
    REQUIRE(c.na == Approx(1.3f));
}

TEST_CASE("loadLegacyConfig throws on unknown key (strict)", "[legacy]") {
    TempFile tf(".cfg", "na=1.3\nnot_a_real_key=5\n");
    REQUIRE_THROWS_AS(loadLegacyConfig(tf.str()), std::runtime_error);
}

TEST_CASE("loadLegacyConfig throws on malformed line", "[legacy]") {
    TempFile tf(".cfg", "na=1.3\nthis_line_has_no_equals\n");
    REQUIRE_THROWS_AS(loadLegacyConfig(tf.str()), std::runtime_error);
}

TEST_CASE("loadLegacyConfig throws on bad value type", "[legacy]") {
    TempFile tf(".cfg", "ndirs=not_an_int\n");
    REQUIRE_THROWS_AS(loadLegacyConfig(tf.str()), std::runtime_error);
}

TEST_CASE("legacy inverted flags map correctly", "[legacy]") {
    SECTION("nosuppress=1 disables suppression") {
        TempFile tf(".cfg", "nosuppress=1\n");
        LegacyReconConfig c = loadLegacyConfig(tf.str());
        REQUIRE(c.bSuppress_singularities == 0);
    }
    SECTION("nosuppress=0 keeps suppression on") {
        TempFile tf(".cfg", "nosuppress=0\n");
        LegacyReconConfig c = loadLegacyConfig(tf.str());
        REQUIRE(c.bSuppress_singularities == 1);
    }
    SECTION("norescale=1 disables rescale") {
        TempFile tf(".cfg", "norescale=1\n");
        LegacyReconConfig c = loadLegacyConfig(tf.str());
        REQUIRE(c.do_rescale == 0);
    }
    SECTION("nofilteroverlaps=1 disables overlap filtering") {
        TempFile tf(".cfg", "nofilteroverlaps=1\n");
        LegacyReconConfig c = loadLegacyConfig(tf.str());
        REQUIRE(c.bFilteroverlaps == false);
    }
}

TEST_CASE("usecorr sets file and enables the flag", "[legacy]") {
    TempFile tf(".cfg", "usecorr=/path/to/corr.tif\n");
    LegacyReconConfig c = loadLegacyConfig(tf.str());
    REQUIRE(c.bUsecorr == 1);
    REQUIRE(c.corrfiles == "/path/to/corr.tif");
}

// --------------------------------------------------------------------------
// fromLegacy conversion
// --------------------------------------------------------------------------

TEST_CASE("fromLegacy maps the example config into SIMParameters", "[legacy][convert]") {
    TempFile tf(".cfg", kExampleConfig);
    SIMParameters p = fromLegacy(loadLegacyConfig(tf.str()));

    REQUIRE(p.ndirs == 3);
    REQUIRE(p.nphases == 5);
    REQUIRE(p.na == Approx(1.42));
    REQUIRE(p.nimm == Approx(1.515));
    REQUIRE(p.linespacing_um == Approx(0.2035));
    REQUIRE(p.dx == Approx(0.08));   // both lateral sizes derive from xyres
    REQUIRE(p.dy == Approx(0.08));
    REQUIRE(p.dz == Approx(0.125));
    REQUIRE(p.dz_psf == Approx(0.125));
    REQUIRE(p.wiener == Approx(0.001));
    REQUIRE(p.background == Approx(0.0));
    REQUIRE(p.dampen_order0 == true);
    REQUIRE(p.fast_si == false);
    REQUIRE(p.k0_angles.has_value());
    REQUIRE(p.k0_angles->size() == 3);
}

TEST_CASE("fromLegacy carries the order count of the config", "[legacy][convert][orders]") {
    SECTION("no orders key derives them, so a 3-phase config converts") {
        TempFile tf(".cfg", "nphases=3\n");
        const SIMParameters p = fromLegacy(loadLegacyConfig(tf.str()));
        CHECK(p.norders == 0);
        CHECK(p.resolvedOrders() == 2);
    }
    SECTION("nordersout, what cudasirecon configs use") {
        TempFile tf(".cfg", "nphases=5\nnordersout=2\n");
        const SIMParameters p = fromLegacy(loadLegacyConfig(tf.str()));
        CHECK(p.norders == 2);
    }
    SECTION("an explicit norders wins over nordersout") {
        TempFile tf(".cfg", "nphases=5\nnordersout=2\nnorders=3\n");
        const SIMParameters p = fromLegacy(loadLegacyConfig(tf.str()));
        CHECK(p.norders == 3);
    }
    SECTION("an order count the phases cannot separate fails validation") {
        TempFile tf(".cfg", "nphases=3\nnorders=3\n");
        REQUIRE_THROWS_AS(fromLegacy(loadLegacyConfig(tf.str())), std::runtime_error);
    }
}

TEST_CASE("fromLegacy converts apodizeoutput int to enum", "[legacy][convert]") {
    auto convert = [](int apo) {
        LegacyReconConfig c;
        c.apodizeoutput = apo;
        return fromLegacy(c).apodize_output;
    };
    REQUIRE(convert(0) == ApodizationType::None);
    REQUIRE(convert(1) == ApodizationType::Cosine);
    REQUIRE(convert(2) == ApodizationType::Triangle);
}

TEST_CASE("fromLegacy maps napodize sign to apodize_input", "[legacy][convert]") {
    auto convert = [](int napodize) {
        LegacyReconConfig c;
        c.napodize = napodize;
        return fromLegacy(c);
    };
    SECTION("negative napodize selects the cosine window, width cleared") {
        SIMParameters p = convert(-1);
        REQUIRE(p.apodize_input == ApodizationType::Cosine);
        REQUIRE(p.napodize == 0);
    }
    SECTION("zero napodize selects none") {
        SIMParameters p = convert(0);
        REQUIRE(p.apodize_input == ApodizationType::None);
        REQUIRE(p.napodize == 0);
    }
    SECTION("napodize < -1 selects none (only -1 means cosine)") {
        SIMParameters p = convert(-2);
        REQUIRE(p.apodize_input == ApodizationType::None);
        REQUIRE(p.napodize == 0);
    }
    SECTION("positive napodize selects triangle and keeps the width") {
        SIMParameters p = convert(15);
        REQUIRE(p.apodize_input == ApodizationType::Triangle);
        REQUIRE(p.napodize == 15);
    }
}

TEST_CASE("fromLegacy rejects invalid apodizeoutput", "[legacy][convert]") {
    LegacyReconConfig c;
    c.apodizeoutput = 5;
    REQUIRE_THROWS_AS(fromLegacy(c), std::runtime_error);
}

TEST_CASE("fromLegacy keeps phaseSteps and forcemodamp, and rejects a bad length", "[legacy][convert]") {
    SECTION("phase steps of length nphases become the separation phases") {
        TempFile tf(".cfg", "nphases=5\nphaseSteps=0,0.4,1.2,2.0,3.5\n");
        const SIMParameters p = fromLegacy(loadLegacyConfig(tf.str()));
        REQUIRE(p.phase_steps);
        REQUIRE(p.phase_steps->size() == 5);
        CHECK((*p.phase_steps)[1] == Approx(0.4));
        CHECK((*p.phase_steps)[4] == Approx(3.5));
    }
    SECTION("the wrong number of phase steps is an error") {
        TempFile tf(".cfg", "nphases=5\nphaseSteps=0,1\n");
        REQUIRE_THROWS_AS(fromLegacy(loadLegacyConfig(tf.str())), std::runtime_error);
    }
    SECTION("forcemodamp of length norders is kept") {
        TempFile tf(".cfg", "nphases=5\nnorders=3\nforcemodamp=1,0.5,0.2\n");
        const SIMParameters p = fromLegacy(loadLegacyConfig(tf.str()));
        REQUIRE(p.force_mod_amp);
        REQUIRE(p.force_mod_amp->size() == 3);
        CHECK((*p.force_mod_amp)[1] == Approx(0.5));
    }
    SECTION("forcemodamp of length ndirs*norders is kept") {
        TempFile tf(".cfg", "nphases=5\nndirs=2\nnorders=3\nforcemodamp=1,0.5,0.2,1,0.4,0.1\n");
        const SIMParameters p = fromLegacy(loadLegacyConfig(tf.str()));
        REQUIRE(p.force_mod_amp);
        CHECK(p.force_mod_amp->size() == 6);
    }
    SECTION("any other forcemodamp length is an error") {
        TempFile tf(".cfg", "nphases=5\nforcemodamp=1,0.5\n");
        REQUIRE_THROWS_AS(fromLegacy(loadLegacyConfig(tf.str())), std::runtime_error);
    }
}

TEST_CASE("patternFundamental divides a 3D line spacing by resolvedOrders()-1", "[params]") {
    SIMParameters p;
    p.nphases = 5;
    p.linespacing_um = 0.2;
    p.norders = 0;   // derived: 5/2+1 = 3, so divide by 2
    CHECK(p.patternFundamental(true) == Approx((1.0 / 0.2) / 2.0));
    CHECK(p.patternFundamental(false) == Approx(1.0 / 0.2));
    p.norders = 2;   // explicit: divide by 1, not by (nphases/2+1)-1
    CHECK(p.resolvedOrders() == 2);
    CHECK(p.patternFundamental(true) == Approx(1.0 / 0.2));
}

TEST_CASE("fromLegacy validates the result", "[legacy][convert]") {
    // k0angles count (2) != ndirs (3) must fail validation inside fromLegacy.
    LegacyReconConfig c;
    c.ndirs = 3;
    c.k0angles = {0.1f, 0.2f};
    REQUIRE_THROWS_AS(fromLegacy(c), std::runtime_error);
}

// --------------------------------------------------------------------------
// cudasirecon's own vocabulary (1.1.1 and 1.2.0) is read in full: the isoar
// configuration of 2026-10-08 was refused on 'k0searchAll' while cudasirecon
// 1.1.1 refused the same file on 'otfcutoff' -- the two parsers had drifted
// into dialects and neither accepted the other's. These three keys were the
// whole difference on the cudasirecon side.
// --------------------------------------------------------------------------

TEST_CASE("loadLegacyConfig reads every key cudasirecon 1.2.0 writes", "[legacy]") {
    TempFile tf(".cfg",
                "ndirs=1\n"
                "nphases=3\n"
                "angle0=1.5961\n"
                "ls=0.491\n"
                "na=1.35\n"
                "nimm=1.405\n"
                "wiener=0.001\n"
                "otfcutoff=.006\n"
                "background=100\n"
                "otfPerAngle=0\n"
                "fastSI=0\n"
                "k0searchAll=1\n"
                "dampenOrder0=0\n"
                "gammaApo=1\n"
                "xyres=0.085\n"
                "zres=0.1\n"
                "wavelength=604\n"
                "besselExWave=0.488\n"
                "version=1\n");
    const auto c = loadLegacyConfig(tf.str());
    REQUIRE(c.ndirs == 1);
    REQUIRE(c.nphases == 3);
    REQUIRE(c.k0searchAll == 1);
    REQUIRE(c.BesselLambdaEx == Approx(0.488f));     // besselExWave is cudasirecon's spelling of besselLambdaEx
    REQUIRE(c.otfcutoff == Approx(0.006f));
    REQUIRE(c.wavelengthNm == Approx(604.0f));
}
