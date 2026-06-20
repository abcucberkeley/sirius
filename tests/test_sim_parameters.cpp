#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "sirius/sim_parameters.hpp"
#include "sirius/legacy_config.hpp"

using namespace sirius;
using Catch::Approx;

namespace {

    // RAII temp file: writes `contents` (if any) and removes the file on scope exit.
    struct TempFile {
        std::filesystem::path path;

        explicit TempFile(const std::string& suffix, const std::string& contents = "") {
            static int counter = 0;
            path = std::filesystem::temp_directory_path() /
                   ("sirius_test_" + std::to_string(counter++) + suffix);
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
// SIMParameters::validate
// --------------------------------------------------------------------------

TEST_CASE("default SIMParameters validate cleanly", "[params]") {
    SIMParameters p;
    REQUIRE_NOTHROW(p.validate());
}

TEST_CASE("validate rejects out-of-range fields", "[params]") {
    SECTION("ndirs < 1") {
        SIMParameters p; p.ndirs = 0;
        REQUIRE_THROWS_AS(p.validate(), std::runtime_error);
    }
    SECTION("nphases < 1") {
        SIMParameters p; p.nphases = 0;
        REQUIRE_THROWS_AS(p.validate(), std::runtime_error);
    }
    SECTION("linespacing_um <= 0") {
        SIMParameters p; p.linespacing_um = 0.0;
        REQUIRE_THROWS_AS(p.validate(), std::runtime_error);
    }
    SECTION("na <= 0") {
        SIMParameters p; p.na = 0.0;
        REQUIRE_THROWS_AS(p.validate(), std::runtime_error);
    }
    SECTION("non-positive pixel sizes") {
        SIMParameters p; p.dz = -0.1;
        REQUIRE_THROWS_AS(p.validate(), std::runtime_error);
    }
    SECTION("negative wiener") {
        SIMParameters p; p.wiener = -1.0;
        REQUIRE_THROWS_AS(p.validate(), std::runtime_error);
    }
    SECTION("z_zoom < 1") {
        SIMParameters p; p.z_zoom = 0;
        REQUIRE_THROWS_AS(p.validate(), std::runtime_error);
    }
}

TEST_CASE("validate enforces k0_angles size == ndirs", "[params]") {
    SIMParameters p;
    p.ndirs = 3;
    p.k0_angles = std::vector<double>{0.1, 0.2};  // wrong size
    REQUIRE_THROWS_AS(p.validate(), std::runtime_error);

    p.k0_angles = std::vector<double>{0.1, 0.2, 0.3};  // correct size
    REQUIRE_NOTHROW(p.validate());
}

// --------------------------------------------------------------------------
// TOML save / load
// --------------------------------------------------------------------------

TEST_CASE("TOML round-trip preserves every serialized field", "[params][toml]") {
    SIMParameters in;
    in.ndirs                  = 2;
    in.nphases                = 7;
    in.linespacing_um         = 0.2035;
    in.k0_start_angle         = 1.234;
    in.na                     = 1.42;
    in.nimm                   = 1.515;
    in.wavelength_nm          = 525.0;
    in.k0_angles              = std::vector<double>{0.8043, 1.8555};  // size == ndirs
    in.dx                     = 0.081;
    in.dy                     = 0.082;
    in.dz                     = 0.125;
    in.dz_psf                 = 0.13;
    in.zoomfact               = 3.0;
    in.z_zoom                 = 2;
    in.wiener                 = 0.001;
    in.otfcutoff              = 0.009;
    in.background             = 5.0;
    in.napodize               = 12;
    in.suppression_radius     = 8;
    in.suppress_singularities = false;  // default true
    in.dampen_order0          = true;   // default false
    in.apodize_output         = ApodizationType::None;  // default Triangle
    in.explodefact            = 1.5;
    in.fast_si                = true;   // default false
    in.do_rescale             = false;  // default true
    in.equalizez              = true;   // default false
    in.no_kz0                 = false;  // default true
    in.filter_overlaps        = false;  // default true

    TempFile tf(".toml");
    saveParameters(tf.str(), in);
    SIMParameters out = loadParameters(tf.str());

    REQUIRE(out.ndirs == in.ndirs);
    REQUIRE(out.nphases == in.nphases);
    REQUIRE(out.linespacing_um == Approx(in.linespacing_um));
    REQUIRE(out.k0_start_angle == Approx(in.k0_start_angle));
    REQUIRE(out.na == Approx(in.na));
    REQUIRE(out.nimm == Approx(in.nimm));
    REQUIRE(out.wavelength_nm == Approx(in.wavelength_nm));
    REQUIRE(out.dx == Approx(in.dx));
    REQUIRE(out.dy == Approx(in.dy));
    REQUIRE(out.dz == Approx(in.dz));
    REQUIRE(out.dz_psf == Approx(in.dz_psf));
    REQUIRE(out.zoomfact == Approx(in.zoomfact));
    REQUIRE(out.z_zoom == in.z_zoom);
    REQUIRE(out.wiener == Approx(in.wiener));
    REQUIRE(out.otfcutoff == Approx(in.otfcutoff));
    REQUIRE(out.background == Approx(in.background));
    REQUIRE(out.napodize == in.napodize);
    REQUIRE(out.suppression_radius == in.suppression_radius);
    REQUIRE(out.suppress_singularities == in.suppress_singularities);
    REQUIRE(out.dampen_order0 == in.dampen_order0);
    REQUIRE(out.apodize_output == in.apodize_output);
    REQUIRE(out.explodefact == Approx(in.explodefact));
    REQUIRE(out.fast_si == in.fast_si);
    REQUIRE(out.do_rescale == in.do_rescale);
    REQUIRE(out.equalizez == in.equalizez);
    REQUIRE(out.no_kz0 == in.no_kz0);
    REQUIRE(out.filter_overlaps == in.filter_overlaps);

    REQUIRE(out.k0_angles.has_value());
    REQUIRE(out.k0_angles->size() == in.k0_angles->size());
    REQUIRE((*out.k0_angles)[0] == Approx((*in.k0_angles)[0]));
    REQUIRE((*out.k0_angles)[1] == Approx((*in.k0_angles)[1]));
}

TEST_CASE("loadParameters keeps defaults for absent keys", "[params][toml]") {
    // Only override na; everything else must keep its default.
    TempFile tf(".toml", "[optics]\nna = 1.49\n");
    SIMParameters out = loadParameters(tf.str());

    SIMParameters def;
    REQUIRE(out.na == Approx(1.49));
    REQUIRE(out.nphases == def.nphases);
    REQUIRE(out.linespacing_um == Approx(def.linespacing_um));
    REQUIRE(out.zoomfact == Approx(def.zoomfact));
}

TEST_CASE("loadParameters throws on malformed TOML", "[params][toml]") {
    TempFile tf(".toml", "this is = = not valid toml [[[\n");
    REQUIRE_THROWS_AS(loadParameters(tf.str()), std::runtime_error);
}

TEST_CASE("loadParameters validates after parsing", "[params][toml]") {
    TempFile tf(".toml", "[optics]\nndirs = 0\n");
    REQUIRE_THROWS_AS(loadParameters(tf.str()), std::runtime_error);
}

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

TEST_CASE("fromLegacy rejects invalid apodizeoutput", "[legacy][convert]") {
    LegacyReconConfig c;
    c.apodizeoutput = 5;
    REQUIRE_THROWS_AS(fromLegacy(c), std::runtime_error);
}

TEST_CASE("fromLegacy validates the result", "[legacy][convert]") {
    // k0angles count (2) != ndirs (3) must fail validation inside fromLegacy.
    LegacyReconConfig c;
    c.ndirs = 3;
    c.k0angles = {0.1f, 0.2f};
    REQUIRE_THROWS_AS(fromLegacy(c), std::runtime_error);
}
