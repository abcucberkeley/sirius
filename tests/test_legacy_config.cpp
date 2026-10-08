// The legacy cudasirecon config: parsing it, and mapping it onto SIMParameters.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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
    SECTION("forcemodamp of length norders-1, which is cudasirecon's own, is kept") {
        // cudasirecon indexes forceamp[order - 1] over order = 1..norders-1,
        // so its list covers the side bands alone. nphases=5 derives 3 orders,
        // so two values. This length used to be refused, and
        // `forcemodamp=0.5` in a 3-phase config -- which is what the user runs
        // -- is the norders-1 == 1 case of it.
        TempFile tf(".cfg", "nphases=5\nforcemodamp=0.5,0.2\n");
        const SIMParameters p = fromLegacy(loadLegacyConfig(tf.str()));
        REQUIRE(p.force_mod_amp);
        CHECK(p.force_mod_amp->size() == 2);
        CHECK(p.resolvedOrders() == 3);
    }
    SECTION("forcemodamp of length ndirs*(norders-1) is kept") {
        TempFile tf(".cfg", "nphases=5\nndirs=2\nforcemodamp=0.5,0.2,0.4,0.1\n");
        const SIMParameters p = fromLegacy(loadLegacyConfig(tf.str()));
        REQUIRE(p.force_mod_amp);
        CHECK(p.force_mod_amp->size() == 4);
    }
    SECTION("any other forcemodamp length is an error") {
        // ndirs defaults to 3 and nphases=5 derives 3 orders, so the four
        // accepted lengths are 2, 6, 3 and 9. Four values is none of them.
        TempFile tf(".cfg", "nphases=5\nforcemodamp=1,0.5,0.2,0.1\n");
        REQUIRE_THROWS_AS(fromLegacy(loadLegacyConfig(tf.str())), std::runtime_error);
    }
}

// --------------------------------------------------------------------------
// forcemodamp is a floor on the side bands, which is what cudasirecon does
// --------------------------------------------------------------------------

TEST_CASE("forcedModAmpFloor resolves every accepted forcemodamp length", "[params][forcemodamp]") {
    SIMParameters p;
    p.ndirs = 3;
    p.nphases = 5;          // 3 orders
    REQUIRE(p.resolvedOrders() == 3);

    SECTION("no list forces nothing") {
        for (int d = 0; d < 3; ++d)
            for (int o = 0; o < 3; ++o) CHECK(p.forcedModAmpFloor(d, o) == 0.0);
    }

    SECTION("norders-1: the side bands, shared by every direction") {
        p.force_mod_amp = std::vector<double>{0.5, 0.2};
        p.validate();
        for (int d = 0; d < 3; ++d) {
            CHECK(p.forcedModAmpFloor(d, 0) == 0.0);   // order 0 is never forced
            CHECK(p.forcedModAmpFloor(d, 1) == 0.5);
            CHECK(p.forcedModAmpFloor(d, 2) == 0.2);
        }
    }

    SECTION("ndirs*(norders-1): the side bands, direction-major") {
        p.force_mod_amp = std::vector<double>{0.5, 0.2, 0.4, 0.1, 0.3, 0.05};
        p.validate();
        CHECK(p.forcedModAmpFloor(0, 1) == 0.5);
        CHECK(p.forcedModAmpFloor(0, 2) == 0.2);
        CHECK(p.forcedModAmpFloor(1, 1) == 0.4);
        CHECK(p.forcedModAmpFloor(1, 2) == 0.1);
        CHECK(p.forcedModAmpFloor(2, 1) == 0.3);
        CHECK(p.forcedModAmpFloor(2, 2) == 0.05);
        for (int d = 0; d < 3; ++d) CHECK(p.forcedModAmpFloor(d, 0) == 0.0);
    }

    SECTION("norders: SIRIUS's older spelling, whose leading entry is order 0's") {
        p.force_mod_amp = std::vector<double>{1.0, 0.5, 0.2};
        p.validate();
        CHECK(p.forcedModAmpFloor(0, 0) == 0.0);   // the 1.0 is not applied
        CHECK(p.forcedModAmpFloor(0, 1) == 0.5);
        CHECK(p.forcedModAmpFloor(2, 2) == 0.2);
    }

    SECTION("ndirs*norders, direction-major") {
        p.force_mod_amp = std::vector<double>{1.0, 0.5, 0.2, 1.0, 0.4, 0.1, 1.0, 0.3, 0.05};
        p.validate();
        CHECK(p.forcedModAmpFloor(0, 1) == 0.5);
        CHECK(p.forcedModAmpFloor(1, 2) == 0.1);
        CHECK(p.forcedModAmpFloor(2, 1) == 0.3);
    }

    SECTION("a non-positive first entry switches the whole feature off") {
        // cudasirecon's gate: `if (params->forceamp[0] > 0.0)`. A later
        // positive entry does not reopen it.
        p.force_mod_amp = std::vector<double>{0.0, 0.2};
        p.validate();
        for (int d = 0; d < 3; ++d)
            for (int o = 0; o < 3; ++o) CHECK(p.forcedModAmpFloor(d, o) == 0.0);
    }

    SECTION("an out-of-range direction or order answers 0 rather than reading past the end") {
        p.force_mod_amp = std::vector<double>{0.5, 0.2};
        p.validate();
        CHECK(p.forcedModAmpFloor(-1, 1) == 0.0);
        CHECK(p.forcedModAmpFloor(3, 1) == 0.0);
        CHECK(p.forcedModAmpFloor(0, -1) == 0.0);
        CHECK(p.forcedModAmpFloor(0, 3) == 0.0);
    }
}

TEST_CASE("a length that is both norders and ndirs*(norders-1) keeps the meaning it already had", "[params][forcemodamp]") {
    // ndirs == 2 with 2 orders makes norders == 2 and ndirs*(norders-1) == 2.
    // Before forcedModAmpFloor existed, only the first was a legal length, so
    // that is the one a list of 2 still means: entry 0 is order 0's (ignored)
    // and entry 1 is order 1's, shared by both directions -- not one entry per
    // direction. tests/test_sim_parameters.cpp's round-trip is exactly this
    // shape, and so is the 2D reconstruction case.
    SIMParameters p;
    p.ndirs = 2;
    p.nphases = 7;
    p.norders = 2;
    p.force_mod_amp = std::vector<double>{1.0, 0.42};
    p.validate();
    CHECK(p.forcedModAmpFloor(0, 1) == 0.42);
    CHECK(p.forcedModAmpFloor(1, 1) == 0.42);   // not 1.0, which a per-direction read would give
    CHECK(p.forcedModAmpFloor(0, 0) == 0.0);
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

// --------------------------------------------------------------------------
// The configs the user actually runs, copied byte for byte into tests/data
// from /clusterfs/nvme2/Data/iSOAR2_nvme2/OTF (read-only acquisition data, so
// the fixtures are copies and the test never reads the instrument tree).
// Both are dated 2026-04-21 and are named by EXCITATION line while their
// `wavelength` key is the EMISSION wavelength -- 488 declares 515, 560
// declares 605 -- so a config and an OTF are never paired by name
// (findings 9k.52).
//
// Before this, SIRIUS refused both outright: loadLegacyConfig throws on an
// unknown key and `uint16` is the first one it had never heard of.
// --------------------------------------------------------------------------

namespace {
    const std::filesystem::path kData = SIRIUS_TEST_DATA_DIR;
    std::string isoarCfg(const char* name) { return (kData / name).string(); }
    bool has(const std::vector<std::string>& v, const std::string& s) {
        return std::find(v.begin(), v.end(), s) != v.end();
    }
} // namespace

TEST_CASE("the user's 488 iSOAR2 config loads, key for key", "[legacy][isoar]") {
    const LegacyReconConfig c = loadLegacyConfig(isoarCfg("isoar2_mount2a_2026-04-21_488.cfg"));

    CHECK(c.nimm == Approx(1.405f));
    CHECK(c.constbkgd == Approx(100.0f));
    CHECK(c.wiener == Approx(0.01f));
    REQUIRE(c.k0angles.size() == 1);
    CHECK(c.k0angles[0] == Approx(-1.57f));
    CHECK(c.linespacing == Approx(0.504f));
    CHECK(c.ndirs == 1);
    CHECK(c.nphases == 3);
    CHECK(c.na == Approx(1.35f));
    CHECK(c.bRadAvgOTF == true);
    CHECK(c.bDampenOrder0 == true);
    REQUIRE(c.forceamp.size() == 1);           // norders-1 for a 3-phase config
    CHECK(c.forceamp[0] == Approx(0.5f));
    CHECK(c.bFastSIM == false);
    CHECK(c.wavelengthNm == Approx(515.0f));   // EMISSION; the file is named 488
    CHECK(c.dxy == 0.085);
    CHECK(c.dz == 0.250);
    CHECK(c.dzPSF == 0.1);
    CHECK(c.bUint16Output == true);
    CHECK(c.zoomfact == Approx(1.0f));
    CHECK(c.cropXmin == 3);
    CHECK(c.cropXmax == 2303);
    CHECK(c.cropYmin == 775);
    CHECK(c.cropYmax == 1533);                 // a trailing space on the line
    CHECK(c.cropZmin == -1);                   // the file sets no z bounds
    CHECK(c.cropZmax == -1);
    CHECK(c.chunkX == 0);                      // and no chunking
    CHECK(c.chunkOverlap == 0);

    // 22 distinct keys, every one of them recorded in file order.
    REQUIRE(c.keysPresent.size() == 22);
    CHECK(c.keysPresent.front() == "nimm");
    CHECK(c.keysPresent.back() == "cropYmax");

    // The crop is 0-indexed and INCLUSIVE, so its width is max - min + 1 --
    // and both lateral extents are ODD, which is the case SIRIUS's even-size
    // gates refuse (findings 9k.50). A caller that applies this crop needs the
    // odd-size work.
    CHECK(c.cropXmax - c.cropXmin + 1 == 2301);
    CHECK(c.cropYmax - c.cropYmin + 1 == 759);
    CHECK((c.cropXmax - c.cropXmin + 1) % 2 == 1);
    CHECK((c.cropYmax - c.cropYmin + 1) % 2 == 1);
}

TEST_CASE("the user's 560 iSOAR2 config loads, chunking and all", "[legacy][isoar]") {
    const LegacyReconConfig c = loadLegacyConfig(isoarCfg("isoar2_mount2a_2026-04-21_560.cfg"));

    CHECK(c.wavelengthNm == Approx(605.0f));   // EMISSION; the file is named 560
    CHECK(c.bFastSIM == true);                 // the one algorithmic difference from the 488 file
    CHECK(c.ndirs == 1);
    CHECK(c.nphases == 3);
    CHECK(c.bUint16Output == true);
    CHECK(c.cropXmin == 0);
    CHECK(c.cropXmax == 2300);
    CHECK(c.cropYmin == 10);
    CHECK(c.cropYmax == 760);
    CHECK(c.chunkX == 384);
    CHECK(c.chunkY == 384);
    CHECK(c.chunkZ == 41);
    CHECK(c.chunkOverlap == 10);
    REQUIRE(c.keysPresent.size() == 26);

    CHECK(c.cropXmax - c.cropXmin + 1 == 2301);
    CHECK(c.cropYmax - c.cropYmin + 1 == 751);
    CHECK((c.cropYmax - c.cropYmin + 1) % 2 == 1);
}

TEST_CASE("both of the user's configs convert, and the report names what is not in effect", "[legacy][isoar][convert]") {
    // fromLegacy ends with p.validate(), so parsing the file is only half the
    // story: forcemodamp=0.5 is length 1 where a 3-phase config derives 2
    // orders, and demanding length norders here refused the file after the
    // parser had accepted it.
    for (const char* name : {"isoar2_mount2a_2026-04-21_488.cfg",
                             "isoar2_mount2a_2026-04-21_560.cfg"}) {
        const LegacyReconConfig c = loadLegacyConfig(isoarCfg(name));
        LegacyConversionReport rep;
        const SIMParameters p = fromLegacy(c, &rep);

        CHECK(p.ndirs == 1);
        CHECK(p.nphases == 3);
        CHECK(p.resolvedOrders() == 2);
        CHECK(p.na == Approx(1.35));
        CHECK(p.nimm == Approx(1.405));
        CHECK(p.linespacing_um == Approx(0.504));
        CHECK(p.dx == 0.085);
        CHECK(p.dz == 0.250);
        CHECK(p.zoomfact == Approx(1.0));
        REQUIRE(p.force_mod_amp);
        REQUIRE(p.force_mod_amp->size() == 1);
        // The single value is order 1's floor, and order 0 is left alone.
        CHECK(p.forcedModAmpFloor(0, 1) == 0.5);
        CHECK(p.forcedModAmpFloor(0, 0) == 0.0);

        // What the file asked for and does not get.
        CHECK_FALSE(rep.everythingApplied());
        CHECK(has(rep.dropped, "uint16"));
        CHECK(has(rep.dropped, "cropXmin"));
        CHECK(has(rep.dropped, "cropYmax"));
        CHECK(has(rep.applied, "forcemodamp"));
        CHECK(has(rep.applied, "wavelength"));
        CHECK(has(rep.applied, "ls"));
        // Every key the file set is accounted for exactly once.
        CHECK(rep.applied.size() + rep.notInEffect().size() == c.keysPresent.size());
        // and every non-applied key carries a reason.
        CHECK(rep.notes.size() == rep.notInEffect().size());
        for (const std::string& note : rep.notes) CHECK(note.find(": ") != std::string::npos);
    }

    // Only the 560 file chunks.
    LegacyConversionReport rep488, rep560;
    fromLegacy(loadLegacyConfig(isoarCfg("isoar2_mount2a_2026-04-21_488.cfg")), &rep488);
    fromLegacy(loadLegacyConfig(isoarCfg("isoar2_mount2a_2026-04-21_560.cfg")), &rep560);
    CHECK_FALSE(has(rep488.dropped, "chunkX"));
    CHECK(has(rep560.dropped, "chunkX"));
    CHECK(has(rep560.dropped, "chunkOverlap"));
}

// --------------------------------------------------------------------------
// Accepting a key is not applying it
// --------------------------------------------------------------------------

TEST_CASE("every key the parser accepts has a declared status", "[legacy][report]") {
    // The guard that keeps aliasTable() and statusTable() from drifting: a new
    // alias with no status would otherwise be reported as applied by default,
    // which is the silence this report exists to end.
    const std::vector<std::string> keys = legacyConfigKeys();
    REQUIRE(keys.size() >= 85);
    for (const std::string& key : keys) {
        std::string why;
        const LegacyKeyStatus s = legacyKeyStatus(key, &why);
        if (s == LegacyKeyStatus::Applied)
            CHECK(why.empty());
        else
            CHECK_FALSE(why.empty());          // a dropped key must say why
    }
    // An unrecognized key is dropped with a reason rather than silently fine.
    std::string why;
    CHECK(legacyKeyStatus("not_a_real_key", &why) == LegacyKeyStatus::Dropped);
    CHECK_FALSE(why.empty());
}

TEST_CASE("a key that is parsed and then thrown away is reported, not hidden", "[legacy][report]") {
    // Each of these was read without complaint and then ignored, so the run
    // did something the file did not ask for.
    TempFile tf(".cfg",
                "nphases=5\n"
                "na=1.3\n"
                "gammaApo=0.5\n"       // SIMParameters has no gamma: always 1
                "fitallphases=0\n"     // SIRIUS always uses every fitted phase
                "equalizet=1\n"
                "otfPerAngle=0\n"
                "nzotf=65\n");
    LegacyConversionReport rep;
    fromLegacy(loadLegacyConfig(tf.str()), &rep);

    CHECK(has(rep.applied, "nphases"));
    CHECK(has(rep.applied, "na"));
    for (const char* key : {"gammaApo", "fitallphases", "equalizet", "otfPerAngle", "nzotf"})
        CHECK(has(rep.dropped, key));
    // The notes name where the quantity has to be handled instead.
    bool sawGamma = false;
    for (const std::string& note : rep.notes)
        if (note.rfind("gammaApo:", 0) == 0) sawGamma = true;
    CHECK(sawGamma);
}

TEST_CASE("the report is empty for a config built in code, not parsed", "[legacy][report]") {
    // keysPresent is the parser's record, so there is nothing to audit here --
    // and an empty report must not read as "everything applied".
    LegacyReconConfig c;
    LegacyConversionReport rep;
    fromLegacy(c, &rep);
    CHECK(rep.applied.empty());
    CHECK(rep.notInEffect().empty());
    CHECK(c.keysPresent.empty());
}

TEST_CASE("fromLegacy without a report still converts, and the one-argument call compiles", "[legacy][report]") {
    TempFile tf(".cfg", "nphases=5\nna=1.3\n");
    const SIMParameters p = fromLegacy(loadLegacyConfig(tf.str()));
    CHECK(p.na == Approx(1.3));
}

// --------------------------------------------------------------------------
// searchforvector reaches a field
// --------------------------------------------------------------------------

TEST_CASE("searchforvector reaches SIMParameters", "[legacy][convert]") {
    SECTION("the default searches") {
        TempFile tf(".cfg", "nphases=5\n");
        CHECK(fromLegacy(loadLegacyConfig(tf.str())).search_pattern_vector == true);
    }
    SECTION("searchforvector=0 says the pattern vector is known") {
        TempFile tf(".cfg", "nphases=5\nsearchforvector=0\n");
        LegacyConversionReport rep;
        const SIMParameters p = fromLegacy(loadLegacyConfig(tf.str()), &rep);
        CHECK(p.search_pattern_vector == false);
        // Honest about the state of it: the field is set, the reconstruction
        // does not read it yet, so the key is Pending and not Applied.
        CHECK(has(rep.pending, "searchforvector"));
        CHECK_FALSE(has(rep.applied, "searchforvector"));
        CHECK(has(rep.notInEffect(), "searchforvector"));
    }
    SECTION("searchforvector=1 searches") {
        TempFile tf(".cfg", "nphases=5\nsearchforvector=1\n");
        CHECK(fromLegacy(loadLegacyConfig(tf.str())).search_pattern_vector == true);
    }
}

TEST_CASE("search_pattern_vector survives the TOML round trip", "[legacy][convert][toml]") {
    TempFile toml(".toml");
    SIMParameters in;
    in.search_pattern_vector = false;   // default true
    saveParameters(toml.str(), in);
    CHECK(loadParameters(toml.str()).search_pattern_vector == false);
}

// --------------------------------------------------------------------------
// The spelling drift between cudasirecon builds
// --------------------------------------------------------------------------

TEST_CASE("both spellings of the overlap-filter switch are read", "[legacy]") {
    // 1.1.1 writes `nofilteroverlaps`; 1.2.0 and the user's fork write
    // `nofilterovlps` (cudaSirecon.cpp's own option name). SIRIUS knew only
    // the first, so a file in the fork's vocabulary was refused on it.
    SECTION("1.1.1's spelling") {
        TempFile tf(".cfg", "nofilteroverlaps=1\n");
        CHECK(loadLegacyConfig(tf.str()).bFilteroverlaps == false);
    }
    SECTION("1.2.0's spelling") {
        TempFile tf(".cfg", "nofilterovlps=1\n");
        CHECK(loadLegacyConfig(tf.str()).bFilteroverlaps == false);
    }
}

TEST_CASE("cudasirecon 1.2.0's output and tiling keys are read with its own semantics", "[legacy]") {
    TempFile tf(".cfg",
                "uint16=1\n"
                "uint16offset=100\n"
                "cropXmin=0\ncropXmax=511\n"
                "cropYmin=4\ncropYmax=515\n"
                "cropZmin=2\ncropZmax=42\n"
                "chunkX=256\nchunkY=256\nchunkZ=32\nchunkOverlap=8\n");
    const LegacyReconConfig c = loadLegacyConfig(tf.str());
    CHECK(c.bUint16Output == true);
    CHECK(c.uint16Offset == Approx(100.0f));
    CHECK(c.cropXmin == 0);
    CHECK(c.cropXmax == 511);
    CHECK(c.cropZmin == 2);
    CHECK(c.cropZmax == 42);
    CHECK(c.chunkZ == 32);
    CHECK(c.chunkOverlap == 8);
    // Unset is -1, not 0: 0 is a legal lower bound (the 560 config uses it).
    LegacyReconConfig d;
    CHECK(d.cropXmin == -1);
    CHECK(d.chunkX == 0);     // 0 means "the entire axis", which is the default
}

TEST_CASE("legacyCropBox turns inclusive bounds into a half-open extent, or refuses", "[legacy][crop]") {
    SECTION("all six bounds give a box, sized max - min + 1") {
        TempFile tf(".cfg",
                    "cropXmin=3\ncropXmax=2303\n"
                    "cropYmin=775\ncropYmax=1533\n"
                    "cropZmin=0\ncropZmax=100\n");
        const auto box = legacyCropBox(loadLegacyConfig(tf.str()));
        REQUIRE(box.has_value());
        CHECK(box->x0 == 3);
        CHECK(box->nx == 2301);
        CHECK(box->y0 == 775);
        CHECK(box->ny == 759);
        CHECK(box->z0 == 0);
        CHECK(box->nz == 101);
    }
    SECTION("the user's own configs give no box: they set four bounds, not six") {
        // cudasirecon 1.2.0's own help says the feature "requires all 6
        // crop{X,Y,Z}{min,max}", and both 2026-04-21 configs set only the
        // lateral four. Rather than invent a z range, this answers nullopt.
        for (const char* name : {"isoar2_mount2a_2026-04-21_488.cfg",
                                 "isoar2_mount2a_2026-04-21_560.cfg"})
            CHECK_FALSE(legacyCropBox(loadLegacyConfig(isoarCfg(name))).has_value());
    }
    SECTION("an inverted bound gives no box") {
        TempFile tf(".cfg",
                    "cropXmin=100\ncropXmax=10\n"
                    "cropYmin=0\ncropYmax=10\n"
                    "cropZmin=0\ncropZmax=10\n");
        CHECK_FALSE(legacyCropBox(loadLegacyConfig(tf.str())).has_value());
    }
    SECTION("a single-pixel box is one pixel wide, not zero") {
        TempFile tf(".cfg",
                    "cropXmin=5\ncropXmax=5\ncropYmin=5\ncropYmax=5\ncropZmin=5\ncropZmax=5\n");
        const auto box = legacyCropBox(loadLegacyConfig(tf.str()));
        REQUIRE(box.has_value());
        CHECK(box->nx == 1);
        CHECK(box->ny == 1);
        CHECK(box->nz == 1);
    }
    SECTION("a config with no crop keys at all gives no box") {
        TempFile tf(".cfg", "na=1.3\n");
        CHECK_FALSE(legacyCropBox(loadLegacyConfig(tf.str())).has_value());
    }
}
