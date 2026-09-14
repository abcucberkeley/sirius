// Multi-file datasets: named-group filename patterns, the manifest's JSON /
// TOML round trip and validation, building a manifest from a folder of TIFF
// stacks, opening that folder as one tiled dataset, and stitching its tiles
// back into the scene they were cut from.

// requireOperation returns a reference to a registry-owned object; GCC 13's
// -Wdangling-reference cannot see that and flags every binding of it.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 13
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <sirius/tiff_io.hpp>

#include "core/array_source.hpp"
#include "core/executor.hpp"
#include "core/manifest.hpp"
#include "core/ops/builtin.hpp"
#include "core/pipeline.hpp"

#include "temp_path.hpp"

using namespace sirius;
using namespace sirius::app;
using Catch::Matchers::WithinAbs;

namespace {

    struct Registered {
        Registered() { registerBuiltinOperations(); }
    };
    const Registered kRegistered;

    struct Progress {
        std::vector<double> fractions;
        StepContext ctx;
        Progress() {
            ctx.backend = Backend::Cpu;
            ctx.scratchDir = std::filesystem::temp_directory_path();
            ctx.progress = [this](double f, const std::string&) { fractions.push_back(f); };
        }
    };

    // A scratch folder removed with everything in it on scope exit.
    struct TempFolder {
        std::filesystem::path path;
        std::string str;
        TempFolder() : path(test::uniqueTempPath("app_manifest", "")), str(path.string()) {
            std::filesystem::create_directories(path);
        }
        ~TempFolder() {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }
        TempFolder(const TempFolder&) = delete;
        TempFolder& operator=(const TempFolder&) = delete;
    };

    // One 42 x 42 x 3 scene of Gaussian blobs on a gradient, cut into a 2 x 2
    // grid of 24 x 24 tiles that overlap by 6 voxels (origins 0 and 18). The
    // manifest's nominal origins come from an overlap fraction of 1/3, i.e.
    // 16 voxels: two voxels off, which the registration has to recover.
    constexpr Index kTile = 24, kScene = 42, kPlanes = 3, kCut = 18;
    constexpr Index kChannels = 2, kTimes = 2;
    const Index kTileOrigin[4][2] = {{0, 0}, {kCut, 0}, {0, kCut}, {kCut, kCut}};   // (y, x), manifest tile order
    const char* const kTileNames[4] = {"tile_x0_y0", "tile_x0_y1", "tile_x1_y0", "tile_x1_y1"};

    float sceneValue(Index c, Index t, Index z, Index y, Index x) {
        double v = 10.0 + 0.3 * x + 0.2 * y;
        for (int k = 0; k < 16; ++k) {
            const double cx = 3 + (k * 17) % 37, cy = 3 + (k * 23 + 5) % 37;
            const double amp = 200.0 * (0.6 + 0.4 * std::cos(k + z));
            const double d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            v += amp * std::exp(-d2 / (2.0 * 1.6 * 1.6));
        }
        return static_cast<float>(v * (1.0 + 0.5 * c) + 10.0 * t);
    }

    Buffer<float> tileVolume(int tile, Index c, Index t) {
        Buffer<float> b(Shape{kPlanes, kTile, kTile});
        for (Index z = 0; z < kPlanes; ++z)
            for (Index y = 0; y < kTile; ++y)
                for (Index x = 0; x < kTile; ++x)
                    b.data()[(z * kTile + y) * kTile + x] = sceneValue(c, t, z, kTileOrigin[tile][0] + y, kTileOrigin[tile][1] + x);
        return b;
    }

    std::string tileFileName(int tile, Index c, Index t) {
        const int gx = tile / 2, gy = tile % 2;   // manifest order is x-major: x0_y0, x0_y1, x1_y0, x1_y1
        char buf[64];
        std::snprintf(buf, sizeof buf, "img_c%d_t%03d_x%d_y%d.tif", c == 0 ? 488 : 640, static_cast<int>(t), gx, gy);
        return buf;
    }

    // Writes the tile files plus a decoy that the pattern does not match.
    void writeTileFolder(const std::filesystem::path& folder) {
        for (int tile = 0; tile < 4; ++tile)
            for (Index c = 0; c < kChannels; ++c)
                for (Index t = 0; t < kTimes; ++t)
                    writeTiffStack<float>((folder / tileFileName(tile, c, t)).string(), tileVolume(tile, c, t).view());
        std::ofstream(folder / "readme.tif") << "not an image\n";
        std::ofstream(folder / "notes.txt") << "ignored\n";
    }

    FilenameRule tileRule() {
        FilenameRule rule;
        // both named-group syntaxes in one pattern
        rule.pattern = R"(img_c(?P<channel>\d+)_t(?<t>\d+)_x(?P<x>\d)_y(?P<y>\d)\.tif)";
        rule.positions = FilenameRule::Positions::GridIndex;
        rule.overlapFraction = 1.0 / 3.0;
        rule.voxelUm = {0.1, 0.1, 0.3};
        rule.acquisition = "Tiled test scene";
        return rule;
    }

    void checkVolume(const float* got, const Buffer<float>& want) {
        double maxErr = 0.0;
        for (Index i = 0; i < want.shape().numel(); ++i) maxErr = std::max<double>(maxErr, std::abs(got[i] - want.data()[i]));
        CHECK(maxErr < 1e-3);
    }

} // namespace

// --- patterns --------------------------------------------------------------------

TEST_CASE("plainPattern strips named groups and keeps their order", "[app][manifest]") {
    std::vector<std::string> names;
    CHECK(plainPattern(R"((?P<c>\d+)_(?<t>\d+))", &names) == R"((\d+)_(\d+))");
    CHECK(names == std::vector<std::string>{"c", "t"});

    SECTION("nested groups") {
        CHECK(plainPattern("(?P<a>x(?P<b>y))", &names) == "(x(y))");
        CHECK(names == std::vector<std::string>{"a", "b"});
    }
    SECTION("non-capturing and unnamed groups") {
        CHECK(plainPattern(R"((?:foo)(?P<n>\d))", &names) == R"((?:foo)(\d))");
        CHECK(names == std::vector<std::string>{"n"});
        CHECK(plainPattern(R"((\d)(?P<n>\d)(?=x))", &names) == R"((\d)(\d)(?=x))");
        CHECK(names == std::vector<std::string>{"", "n"});   // the unnamed group keeps its number
    }
    SECTION("escaped parentheses and character classes") {
        CHECK(plainPattern(R"(\((?P<n>\d)\))", &names) == R"(\((\d)\))");
        CHECK(names == std::vector<std::string>{"n"});
        CHECK(plainPattern(R"([(](?P<n>a)[)])", &names) == R"([(](a)[)])");
        CHECK(names == std::vector<std::string>{"n"});
        CHECK(plainPattern(R"([^\]](?<n>b))", &names) == R"([^\]](b))");
        CHECK(names == std::vector<std::string>{"n"});
    }
    SECTION("no names wanted") {
        CHECK(plainPattern("(?P<n>a)", nullptr) == "(a)");
    }
    SECTION("bad names") {
        CHECK_THROWS_AS(plainPattern("(?P<n", &names), std::invalid_argument);
        CHECK_THROWS_AS(plainPattern("(?P<1a>x)", &names), std::invalid_argument);
    }
}

TEST_CASE("AOLLS Scan_Iter names parse camera, tile grid and time", "[app][manifest]") {
    const std::string name =
        "Scan_Iter_0003_0000_0000_0000_CamB_ch0_CAM1_stack0000_488nm_0000000msec_1573519287msecAbs_000x_002y_001z_0003t.tif";
    const std::string pat =
        R"(Scan_Iter_\d+_\d+_\d+_\d+_(?P<channel>Cam[A-Z])_ch\d+_CAM\d+_stack\d+_\d+nm_\d+msec_\d+msecAbs_(?P<x>\d+)x_(?P<y>\d+)y_(?P<z>\d+)z_(?P<t>\d+)t\.tiff?$)";
    const auto matches = matchFilenames({name, "Calib_fish1.h5", "stack_c488_t0_x1_y2.tif"}, pat);
    REQUIRE(matches.size() == 3);
    CHECK(matches[0].matched);
    CHECK(matches[0].groups.at("channel") == "CamB");
    CHECK(matches[0].groups.at("x") == "000");
    CHECK(matches[0].groups.at("y") == "002");
    CHECK(matches[0].groups.at("z") == "001");
    CHECK(matches[0].groups.at("t") == "0003");
    CHECK_FALSE(matches[1].matched);
    CHECK_FALSE(matches[2].matched);
}

TEST_CASE("matchFilenames reports the named groups of every file", "[app][manifest]") {
    const std::vector<std::string> names{"img_c488_t003_x1_y2.tif", "other.tif"};
    const auto matches = matchFilenames(names, R"(img_c(?P<channel>\d+)_t(?P<t>\d+)_x(?P<x>\d+)_y(?P<y>\d+)\.tif)");
    REQUIRE(matches.size() == 2);
    CHECK(matches[0].file == names[0]);
    CHECK(matches[0].matched);
    CHECK(matches[0].groups.at("channel") == "488");
    CHECK(matches[0].groups.at("t") == "003");
    CHECK(matches[0].groups.at("x") == "1");
    CHECK(matches[0].groups.at("y") == "2");
    CHECK_FALSE(matches[1].matched);
    CHECK(matches[1].groups.empty());
    CHECK_THROWS_AS(matchFilenames(names, "("), std::invalid_argument);
}

// --- manifest --------------------------------------------------------------------

TEST_CASE("DatasetManifest round-trips through JSON and TOML", "[app][manifest]") {
    DatasetManifest m;
    m.name = "exp1";
    m.voxelUm = {0.065, 0.065, 0.25};
    m.frameIntervalS = 2.5;
    m.acquisition = "3D-SIM raw";
    m.pattern = R"((?P<c>\d+))";
    m.sim.present = true;
    m.sim.ndirs = 3;
    m.sim.nphases = 5;
    m.sim.fastSi = true;
    ChannelInfo gfp;
    gfp.label = "GFP";
    gfp.wavelengthNm = 488.0;
    gfp.color = colorFromHex("#63e08a");
    gfp.exposure = "8 ms";
    ChannelInfo red;
    red.label = "mCherry";
    red.wavelengthNm = 610.0;
    red.color = {1.f, 0.f, 0.f};
    m.channels = {gfp, red};
    TileInfo a, b;
    a.name = "tile_x0_y0";
    b.name = "tile_x1_y0";
    b.positionUm = {0.0, 0.0, 12.5};
    b.gridIndex = {0, 0, 1};
    m.tiles = {a, b};
    m.files = {{"a_488.tif", "GFP", 0, "tile_x0_y0"}, {"a_610.tif", "mCherry", 0, "tile_x0_y0"}, {"b_488.tif", "GFP", 0, "tile_x1_y0"}, {"b_610_t1.tif", "mCherry", 1, "tile_x1_y0"}};

    CHECK(m.channelIndex("mCherry") == 1);
    CHECK(m.channelIndex("1") == 1);   // an index as text
    CHECK(m.channelIndex("DAPI") == -1);
    CHECK(m.tileIndex("") == 0);
    CHECK(m.tileIndex("tile_x1_y0") == 1);
    CHECK(m.tileIndex("nowhere") == -1);
    CHECK(m.timePoints() == 2);
    REQUIRE(m.file(1, 1, 1));
    CHECK(m.file(1, 1, 1)->path == "b_610_t1.tif");
    CHECK(m.file(0, 1, 1) == nullptr);

    auto check = [&](const DatasetManifest& r) {
        CHECK(r.name == "exp1");
        CHECK_THAT(r.voxelUm[0], WithinAbs(0.065, 1e-12));
        CHECK_THAT(r.voxelUm[2], WithinAbs(0.25, 1e-12));
        CHECK_THAT(r.frameIntervalS, WithinAbs(2.5, 1e-12));
        CHECK(r.acquisition == "3D-SIM raw");
        CHECK(r.pattern == m.pattern);
        CHECK(r.sim.present);
        CHECK(r.sim.ndirs == 3);
        CHECK(r.sim.nphases == 5);
        CHECK(r.sim.fastSi);
        REQUIRE(r.channels.size() == 2);
        CHECK(r.channels[0].label == "GFP");
        CHECK_THAT(r.channels[0].wavelengthNm, WithinAbs(488.0, 1e-9));
        CHECK(r.channels[0].hexColor() == "#63e08a");
        CHECK(r.channels[0].exposure == "8 ms");
        CHECK(r.channels[1].label == "mCherry");
        CHECK(r.channels[1].hexColor() == "#ff0000");
        REQUIRE(r.tiles.size() == 2);
        CHECK(r.tiles[1].name == "tile_x1_y0");
        CHECK_THAT(r.tiles[1].positionUm[2], WithinAbs(12.5, 1e-12));
        CHECK(r.tiles[1].gridIndex == std::array<Index, 3>{0, 0, 1});
        REQUIRE(r.files.size() == 4);
        CHECK(r.files[3].path == "b_610_t1.tif");
        CHECK(r.files[3].channel == "mCherry");
        CHECK(r.files[3].t == 1);
        CHECK(r.files[3].tile == "tile_x1_y0");
    };
    SECTION("json") {
        const nlohmann::json j = m.toJson();
        CHECK(j["channels"][0]["color"] == "#63e08a");
        CHECK(j["tiles"][1]["position_um"][2] == 12.5);
        CHECK(j["tiles"][1]["grid"][2] == 1);
        check(DatasetManifest::fromJson(j));
    }
    SECTION("toml file") {
        const test::TempFile file("app_manifest", ".toml");
        m.save(file.path);
        check(DatasetManifest::load(file.path));
        std::ifstream in(file.path);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        CHECK(text.find("[[channels]]") != std::string::npos);
        CHECK(text.find("[[files]]") != std::string::npos);
    }
    SECTION("toml in a folder, addressed by the folder") {
        const TempFolder folder;
        m.save(folder.path);
        CHECK(std::filesystem::exists(folder.path / DatasetManifest::kFileName));
        check(DatasetManifest::load(folder.path));
    }
    SECTION("a broken toml names the file") {
        const test::TempFile file("app_manifest_bad", ".toml");
        std::ofstream(file.path) << "name = \n";
        CHECK_THROWS_AS(DatasetManifest::load(file.path), std::runtime_error);
    }
}

TEST_CASE("manifestFromFolder builds the manifest from a filename rule", "[app][manifest]") {
    const TempFolder folder;
    writeTileFolder(folder.path);
    FilenameRule rule = tileRule();

    SECTION("grid tiles, channels from the tokens") {
        std::vector<std::string> unmatched;
        const DatasetManifest m = manifestFromFolder(folder.path, rule, &unmatched);
        CHECK(unmatched == std::vector<std::string>{"readme.tif"});
        CHECK(m.name == folder.path.filename().string());
        CHECK(m.pattern == rule.pattern);
        CHECK(m.acquisition == "Tiled test scene");
        CHECK_THAT(m.voxelUm[2], WithinAbs(0.3, 1e-12));
        REQUIRE(m.channels.size() == 2);
        CHECK(m.channels[0].label == "488");
        CHECK_THAT(m.channels[0].wavelengthNm, WithinAbs(488.0, 1e-9));
        CHECK(m.channels[0].hexColor() == "#63e08a");   // the palette colour of 488 nm
        CHECK(m.channels[1].label == "640");
        CHECK_THAT(m.channels[1].wavelengthNm, WithinAbs(640.0, 1e-9));
        REQUIRE(m.tiles.size() == 4);
        for (int i = 0; i < 4; ++i) CHECK(m.tiles[static_cast<std::size_t>(i)].name == kTileNames[i]);
        // 24 voxels * 0.1 um * (1 - 1/3) = 1.6 um between neighbours
        CHECK(m.tiles[0].gridIndex == std::array<Index, 3>{0, 0, 0});
        CHECK(m.tiles[1].gridIndex == std::array<Index, 3>{0, 1, 0});   // tile_x0_y1: row 1
        CHECK(m.tiles[2].gridIndex == std::array<Index, 3>{0, 0, 1});   // tile_x1_y0: col 1
        CHECK_THAT(m.tiles[1].positionUm[1], WithinAbs(1.6, 1e-9));
        CHECK_THAT(m.tiles[1].positionUm[2], WithinAbs(0.0, 1e-9));
        CHECK_THAT(m.tiles[2].positionUm[2], WithinAbs(1.6, 1e-9));
        CHECK_THAT(m.tiles[3].positionUm[1], WithinAbs(1.6, 1e-9));
        CHECK_THAT(m.tiles[3].positionUm[2], WithinAbs(1.6, 1e-9));
        CHECK(m.files.size() == 16);
        CHECK(m.timePoints() == 2);
        REQUIRE(m.file(3, 1, 1));
        CHECK(m.file(3, 1, 1)->path == "img_c640_t001_x1_y1.tif");
        CHECK(m.validate(folder.path).empty());

        // the folder's own manifest reopens to the same thing
        m.save(folder.path);
        const DatasetManifest r = DatasetManifest::load(folder.path);
        CHECK(r.files.size() == 16);
        CHECK(r.tiles.size() == 4);
        CHECK(r.validate(folder.path).empty());
    }
    SECTION("channel names from the rule") {
        ChannelInfo gfp;
        gfp.label = "GFP";
        gfp.wavelengthNm = 488.0;
        rule.channelInfo["488"] = gfp;
        const DatasetManifest m = manifestFromFolder(folder.path, rule);
        REQUIRE(m.channels.size() == 2);
        CHECK(m.channels[0].label == "GFP");
        CHECK(m.channels[1].label == "640");
        CHECK(m.channelIndex("GFP") == 0);
        REQUIRE(m.file(0, 0, 0));
        CHECK(m.file(0, 0, 0)->channel == "GFP");
        CHECK(m.validate(folder.path).empty());
    }
    SECTION("stage positions in micrometres") {
        rule.positions = FilenameRule::Positions::Microns;
        const DatasetManifest m = manifestFromFolder(folder.path, rule);
        REQUIRE(m.tiles.size() == 4);
        CHECK_THAT(m.tiles[2].positionUm[2], WithinAbs(1.0, 1e-12));   // x token "1" is 1 um
        CHECK(m.tiles[2].gridIndex == std::array<Index, 3>{0, 0, 1});
        CHECK(m.tiles[3].gridIndex == std::array<Index, 3>{0, 1, 1});
    }
    SECTION("no positions") {
        rule.positions = FilenameRule::Positions::None;
        const DatasetManifest m = manifestFromFolder(folder.path, rule);
        REQUIRE(m.tiles.size() == 4);
        for (const TileInfo& t : m.tiles) CHECK(t.positionUm == std::array<double, 3>{0, 0, 0});
        CHECK(m.tiles[3].gridIndex == std::array<Index, 3>{0, 0, 3});
    }
    SECTION("a tile group and no time / channel groups") {
        rule.pattern = R"(img_c488_t000_(?P<tile>x\d_y\d)\.tif)";
        const DatasetManifest m = manifestFromFolder(folder.path, rule);
        REQUIRE(m.channels.size() == 1);
        CHECK(m.channels[0].label == "0");
        CHECK(m.timePoints() == 1);
        REQUIRE(m.tiles.size() == 4);
        CHECK(m.tiles[1].name == "x0_y1");
        CHECK(m.tiles[1].gridIndex == std::array<Index, 3>{0, 0, 1});   // no grid tokens: a row
        CHECK(m.files.size() == 4);
        CHECK(m.validate(folder.path).empty());
    }
    SECTION("two files on one (tile, channel, t)") {
        rule.pattern = R"(img_c(?P<channel>\d+)_t(?P<t>\d+)_x(?P<x>\d)\.tif)";   // matches nothing: an empty manifest
        std::vector<std::string> unmatched;
        const DatasetManifest none = manifestFromFolder(folder.path, rule, &unmatched);
        CHECK(none.files.empty());
        CHECK(none.tiles.empty());
        CHECK(unmatched.size() == 17);
        CHECK_FALSE(none.validate(folder.path).empty());
        rule.pattern = R"(img_c(?P<channel>\d+)_t(?P<t>\d+)_x(?P<x>\d)_y\d\.tif)";   // y not captured: two files per slot
        CHECK_THROWS_WITH(manifestFromFolder(folder.path, rule), Catch::Matchers::ContainsSubstring("img_c488_t000_x0_y1.tif"));
    }
    SECTION("a file of another shape is named") {
        Buffer<float> odd(Shape{kPlanes, kTile, kTile + 1});
        std::fill(odd.data(), odd.data() + odd.shape().numel(), 1.0f);
        writeTiffStack<float>((folder.path / "img_c488_t000_x0_y0.tif").string(), odd.view());
        CHECK_THROWS_WITH(manifestFromFolder(folder.path, rule), Catch::Matchers::ContainsSubstring("img_c488_t000_x0_y0.tif"));
    }
    SECTION("validate finds the holes") {
        DatasetManifest m = manifestFromFolder(folder.path, rule);
        REQUIRE(m.validate(folder.path).empty());
        SECTION("missing file") {
            m.files[5].path = "gone.tif";
            const auto problems = m.validate(folder.path);
            REQUIRE_FALSE(problems.empty());
            CHECK(problems[0].find("gone.tif") != std::string::npos);
        }
        SECTION("unknown tile") {
            m.files[2].tile = "nowhere";
            const auto problems = m.validate(folder.path);
            REQUIRE(problems.size() >= 2);   // the unknown tile, and the slot it left empty
            CHECK(problems[0].find("nowhere") != std::string::npos);
        }
        SECTION("unknown channel") {
            m.files[0].channel = "DAPI";
            REQUIRE_FALSE(m.validate(folder.path).empty());
        }
        SECTION("a slot without a file") {
            m.files.pop_back();
            const auto problems = m.validate(folder.path);
            REQUIRE(problems.size() == 1);
            CHECK(problems[0].find("tile_x1_y1") != std::string::npos);
        }
        SECTION("nothing at all") {
            CHECK_FALSE(DatasetManifest{}.validate(folder.path).empty());
        }
    }
}

// --- opening the folder ----------------------------------------------------------

TEST_CASE("A folder with a manifest opens as one tiled dataset", "[app][manifest]") {
    const TempFolder folder;
    writeTileFolder(folder.path);
    manifestFromFolder(folder.path, tileRule()).save(folder.path);
    REQUIRE(isFolderDataset(folder.str));
    CHECK_FALSE(isFolderDataset((folder.path / "img_c488_t000_x0_y0.tif").string()));

    const DatasetMeta probed = probeDataset(folder.str);
    CHECK(probed.dims == Dims5{2, 2, kPlanes, kTile, kTile});
    CHECK(probed.format == "folder");
    CHECK(probed.tiles.size() == 4);
    CHECK(probed.tileIndex == 0);
    CHECK(probed.name == folder.path.filename().string());
    CHECK(probed.sourcePath == folder.str);
    CHECK(probed.sourceType == PixelType::Float32);
    CHECK(probed.bytesOnDisk > 16 * kPlanes * kTile * kTile * sizeof(float));
    REQUIRE(probed.channels.size() == 2);
    CHECK_THAT(probed.channels[1].wavelengthNm, WithinAbs(640.0, 1e-9));
    CHECK_THAT(probed.voxelUm[2], WithinAbs(0.3, 1e-12));
    const auto px = probed.tilePositionsPx();
    REQUIRE(px.size() == 4);
    CHECK_THAT(px[2][2], WithinAbs(16.0, 1e-6));   // 1.6 um / 0.1 um

    OpenResult r = openDataset(folder.str, OpenOptions{});
    REQUIRE(r.source);
    CHECK(r.meta.dims == Dims5{2, 2, kPlanes, kTile, kTile});
    CHECK(r.metadataSummary == "4 tiles · 2 channels · manifest");
    CHECK(r.dimsFromMetadata);
    CHECK(r.source->tileCount() == 4);
    CHECK(r.source->currentTile() == 0);
    CHECK_FALSE(r.source->inMemory());

    std::vector<float> plane(static_cast<std::size_t>(kTile * kTile));
    r.source->readPlane(1, 1, 2, plane.data());
    const Buffer<float> t0c1t1 = tileVolume(0, 1, 1);
    {
        double maxErr = 0.0;
        for (Index i = 0; i < kTile * kTile; ++i)
            maxErr = std::max<double>(maxErr, std::abs(plane[static_cast<std::size_t>(i)] - t0c1t1.data()[2 * kTile * kTile + i]));
        CHECK(maxErr < 1e-3);
    }
    std::vector<float> volume(static_cast<std::size_t>(kPlanes * kTile * kTile));
    r.source->readVolume(0, 1, volume.data());
    checkVolume(volume.data(), tileVolume(0, 0, 1));

    SECTION("other tiles") {
        r.source->readTileVolume(2, 1, 0, volume.data());
        checkVolume(volume.data(), tileVolume(2, 1, 0));
        CHECK(r.source->currentTile() == 0);   // reading another tile does not switch
        r.source->selectTile(3);
        CHECK(r.source->currentTile() == 3);
        CHECK(r.source->meta().tileIndex == 3);
        r.source->readPlane(0, 0, 0, plane.data());
        const Buffer<float> t3 = tileVolume(3, 0, 0);
        double maxErr = 0.0;
        for (Index i = 0; i < kTile * kTile; ++i)
            maxErr = std::max<double>(maxErr, std::abs(plane[static_cast<std::size_t>(i)] - t3.data()[i]));
        CHECK(maxErr < 1e-3);
        CHECK_THROWS_AS(r.source->selectTile(4), std::out_of_range);
        CHECK_THROWS_AS(r.source->readTileVolume(-1, 0, 0, volume.data()), std::out_of_range);
        CHECK_THROWS_AS(r.source->readPlane(2, 0, 0, plane.data()), std::out_of_range);
        CHECK_THROWS_AS(r.source->readPlane(0, 0, kPlanes, plane.data()), std::out_of_range);
    }
    SECTION("the initial tile and a full read") {
        OpenOptions options;
        options.tile = 1;
        options.voxelUm = std::array<double, 3>{0.2, 0.2, 0.5};
        OpenResult r1 = openDataset(folder.str, options);
        CHECK(r1.meta.tileIndex == 1);
        CHECK(r1.source->currentTile() == 1);
        CHECK_THAT(r1.meta.voxelUm[0], WithinAbs(0.2, 1e-12));
        std::shared_ptr<Array5> all = r1.source->readAll();
        REQUIRE(all);
        CHECK(all->dims() == Dims5{2, 2, kPlanes, kTile, kTile});
        checkVolume(all->plane(1, 0, 0), tileVolume(1, 1, 0));
        options.tile = 4;
        CHECK_THROWS(openDataset(folder.str, options));
        options.tile = 0;
        options.readAll = true;
        OpenResult full = openDataset(folder.str, options);
        CHECK(full.source->inMemory());
        CHECK(full.meta.tiles.size() == 4);
    }
    SECTION("an incomplete manifest is refused") {
        DatasetManifest m = DatasetManifest::load(folder.path);
        m.files.pop_back();
        m.save(folder.path);
        CHECK_THROWS_WITH(openDataset(folder.str, OpenOptions{}), Catch::Matchers::ContainsSubstring("tile_x1_y1"));
        CHECK_THROWS(probeDataset(folder.str));
    }
    SECTION("the Load step") {
        const Operation& load = requireOperation("load");
        ParamSet p = load.defaults();
        p.set("path", folder.str);
        p.set("tile", std::int64_t{1});
        CHECK(load.validate(p, DatasetMeta{}).ok());
        CHECK(load.summary(p, DatasetMeta{}).find("tile 2/4 · tile_x0_y1") != std::string::npos);
        const DatasetMeta predicted = load.outputMeta(p, DatasetMeta{});
        CHECK(predicted.tiles.size() == 4);
        CHECK(predicted.tileIndex == 1);
        Progress prog;
        const StepOutput out = load.run(StepInput{}, p, prog.ctx);
        REQUIRE(out.source);
        CHECK(out.source->currentTile() == 1);
        CHECK(out.meta.tileIndex == 1);
        CHECK(out.meta.tiles.size() == 4);
        CHECK(out.note.find("tile 2/4") != std::string::npos);
        p.set("tile", std::int64_t{4});
        CHECK_FALSE(load.validate(p, DatasetMeta{}).ok());
    }
}

TEST_CASE("a manifest saved outside the TIFF folder still opens the files", "[app][manifest]") {
    const TempFolder folder;
    writeTileFolder(folder.path);
    DatasetManifest m = manifestFromFolder(folder.path, tileRule());
    m.filesFolder = folder.path.string();
    const test::TempFile sidecar("app_manifest_sidecar", ".toml");
    m.save(sidecar.path);

    CHECK(isDatasetManifestFile(sidecar.str));
    CHECK_FALSE(isFolderDataset(folder.str));
    const DatasetManifest loaded = DatasetManifest::load(sidecar.path);
    CHECK(loaded.filesFolder == folder.path.string());
    CHECK(loaded.filesRoot(sidecar.path) == folder.path);
    CHECK(loaded.validate(loaded.filesRoot(sidecar.path)).empty());

    const DatasetMeta probed = probeDataset(sidecar.str);
    CHECK(probed.dims == Dims5{2, 2, kPlanes, kTile, kTile});
    CHECK(probed.format == "folder");
    CHECK(probed.sourcePath == sidecar.str);
    CHECK(probed.tiles.size() == 4);

    const OpenResult opened = openDataset(sidecar.str);
    REQUIRE(opened.source);
    CHECK(opened.source->tileCount() == 4);
    CHECK(opened.meta.dims == probed.dims);
}

// --- stitching the tile set ------------------------------------------------------

TEST_CASE("Stitch fuses the tiles of a folder dataset back into the scene", "[app][ops][stitch][manifest]") {
    const TempFolder folder;
    writeTileFolder(folder.path);
    manifestFromFolder(folder.path, tileRule()).save(folder.path);
    OpenResult opened = openDataset(folder.str, OpenOptions{});
    StepInput in;
    in.meta = opened.meta;
    in.source = opened.source;

    const Operation& op = requireOperation("stitch");
    ParamSet p = op.defaults();
    CHECK(op.validate(p, in.meta).ok());
    CHECK(op.summary(p, in.meta).find("4 dataset tiles") != std::string::npos);
    p.set("channel", std::int64_t{1});
    p.set("reference_t", std::int64_t{1});
    p.set("search_radius", std::vector<double>{0, 4, 4});
    p.set("mask_background", false);
    REQUIRE(op.validate(p, in.meta).ok());
    const DatasetMeta predicted = op.outputMeta(p, in.meta);
    CHECK(predicted.dims == Dims5{2, 2, kPlanes, 40, 40});   // the nominal layout: 24 + 16
    CHECK(predicted.tiles.empty());
    CHECK(predicted.channels.size() == 2);

    auto checkMosaic = [&](const StepOutput& r) {
        REQUIRE(r.array);
        const Dims5 d = r.array->dims();
        CHECK(d.c == 2);
        CHECK(d.t == 2);
        CHECK(d.z == kPlanes);
        CHECK(d.y >= kScene - 1);
        CHECK(d.y <= kScene + 1);
        CHECK(d.x >= kScene - 1);
        CHECK(d.x <= kScene + 1);
        CHECK(r.meta.dims == d);
        CHECK(r.meta.tiles.empty());
        CHECK(r.meta.tileIndex == 0);
        // the registration recovers the 2-voxel error of the nominal layout, so
        // the mosaic is the scene again, in every channel and time point
        double sumErr = 0.0, maxErr = 0.0;
        Index n = 0;
        for (Index c = 0; c < 2; ++c)
            for (Index t = 0; t < 2; ++t)
                for (Index z = 0; z < kPlanes; ++z)
                    for (Index y = 0; y < std::min(d.y, kScene); ++y)
                        for (Index x = 0; x < std::min(d.x, kScene); ++x) {
                            const double e = std::abs(r.array->at(c, t, z, y, x) - sceneValue(c, t, z, y, x));
                            sumErr += e;
                            maxErr = std::max(maxErr, e);
                            ++n;
                        }
        CHECK(sumErr / n < 1.0);
        CHECK(maxErr < 10.0);
    };

    Progress prog;
    const StepOutput r = op.run(in, p, prog.ctx);
    checkMosaic(r);
    CHECK(r.diagnostics.kind == DiagnosticsKind::Alignment);
    REQUIRE(r.diagnostics.alignment);
    CHECK(r.diagnostics.alignment->gridRows == 2);
    CHECK(r.diagnostics.alignment->gridCols == 2);
    REQUIRE(r.diagnostics.alignment->tileNames.size() == 4);
    CHECK(r.diagnostics.alignment->tileNames[0] == "tile_x0_y0");
    CHECK(r.diagnostics.alignment->tileNames[1] == "tile_x1_y0");   // row-major over the grid
    CHECK(r.diagnostics.alignment->tileNames[2] == "tile_x0_y1");
    CHECK(r.diagnostics.alignment->highlightedTile == 0);
    CHECK(r.diagnostics.alignment->shiftStats.size() == 4);
    REQUIRE(r.diagnostics.table);
    CHECK(r.diagnostics.table->header.size() == 6);
    CHECK(r.diagnostics.table->rows.size() >= 4);   // the four edges at least
    CHECK(r.diagnostics.table->rows[0][0].rfind("tile_", 0) == 0);
    CHECK_FALSE(r.diagnostics.images.empty());
    CHECK(r.note.find("4 tiles") != std::string::npos);
    CHECK(prog.fractions.back() == 1.0);

    SECTION("a tile read into memory reopens the folder for the other tiles") {
        // what the Load step's "Full load to RAM" hands on: the array and a
        // memory source of the one tile
        StepInput full;
        full.meta = opened.meta;
        full.array = opened.source->readAll();
        full.source = std::make_shared<MemorySource>(full.array, opened.meta);
        const StepOutput r2 = op.run(full, p, prog.ctx);
        checkMosaic(r2);
    }
    SECTION("an input a step computed is refused, not replaced by the files") {
        // a contrast or a flat field between Load and Stitch: the folder
        // reopened for it gave the raw tiles back, the step silently gone
        StepInput computed;
        computed.meta = opened.meta;
        computed.array = opened.source->readAll();
        CHECK_THROWS_WITH(op.run(computed, p, prog.ctx), Catch::Matchers::ContainsSubstring("directly after Load"));
    }
    SECTION("validation") {
        p.set("channel", std::int64_t{2});
        CHECK_FALSE(op.validate(p, in.meta).ok());
        p.set("channel", std::int64_t{0});
        p.set("reference_t", std::int64_t{2});
        CHECK_FALSE(op.validate(p, in.meta).ok());
        p.set("reference_t", std::int64_t{0});
        DatasetMeta single = in.meta;
        single.tiles.resize(1);
        CHECK_FALSE(op.validate(p, single).ok());   // one tile is nothing to stitch
    }
}

TEST_CASE("Stitch of a dataset's tiles runs on Load's tiles, not on what a step made of them", "[app][ops][stitch][manifest]") {
    const TempFolder folder, cache;
    writeTileFolder(folder.path);
    manifestFromFolder(folder.path, tileRule()).save(folder.path);
    StepContext ctx;
    ctx.backend = Backend::Cpu;
    ctx.scratchDir = cache.path;
    auto pipeline = [&](bool contrast, bool fullLoad) {
        Pipeline p;
        ParamSet lp = p.at(0).params;
        lp.set("path", folder.str);
        if (fullLoad) lp.set("read_as", std::string("Full load to RAM"));
        p.setParams(0, lp);
        if (contrast) {
            p.add("contrast");
            ParamSet cp = p.at(1).params;
            cp.set("min", 0.0);
            cp.set("max", 100.0);
            cp.set("gamma", 1.0);
            p.setParams(1, cp);
        }
        p.add("stitch");
        const int s = p.size() - 1;
        ParamSet sp = p.at(s).params;
        sp.set("search_radius", std::vector<double>{0, 4, 4});
        sp.set("mask_background", false);
        p.setParams(s, sp);
        return p;
    };
    Executor direct(cache.path / "direct");
    const std::shared_ptr<const StepOutput> lazy = direct.runAll(pipeline(false, false), ctx);
    REQUIRE(lazy->array);
    CHECK(lazy->array->dims().c == 2);
    Executor full(cache.path / "full");
    const std::shared_ptr<const StepOutput> inMemory = full.runAll(pipeline(false, true), ctx);
    REQUIRE(inMemory->array);
    CHECK(inMemory->array->dims() == lazy->array->dims());
    // Load -> Contrast -> Stitch used to give exactly Load -> Stitch: the
    // folder was reopened and the contrast silently left out
    Executor windowed(cache.path / "contrast");
    CHECK_THROWS_WITH(windowed.runAll(pipeline(true, false), ctx), Catch::Matchers::ContainsSubstring("directly after Load"));
}

TEST_CASE("A folder of TIFFs reads as one stack without a pattern", "[app][manifest]") {
    // The layout that needs no describing: a frame per file, in the order a
    // person reads the names.
    TempFolder dir;
    for (const char* name : {"frame1.tif", "frame2.tif", "frame10.tif", "frame11.tif", "notes.txt", ".hidden.tif"})
        std::ofstream(dir.path / name, std::ios::binary) << "x";

    const std::vector<std::string> names = tiffNamesInOrder(dir.path);
    // "frame2" before "frame10": plain alphabetical order gets this wrong, and
    // a time series read out of order is worse than one that fails to open
    CHECK(names == std::vector<std::string>{"frame1.tif", "frame2.tif", "frame10.tif", "frame11.tif"});

    const DatasetManifest m = manifestOfOneStack(dir.path);
    CHECK(m.files.size() == 4);
    CHECK(m.channels.size() == 1);
    CHECK(m.tiles.size() == 1);
    CHECK(m.timePoints() == 4);
    CHECK(m.acquisition == "One stack per file");
    for (std::size_t i = 0; i < m.files.size(); ++i) {
        CHECK(m.files[i].path == names[i]);
        CHECK(m.files[i].t == static_cast<Index>(i));   // one time point each, in that order
    }

    SECTION("real files: the manifest validates and the folder opens as N time points") {
        TempFolder real;
        const Index frames = 5;
        for (Index t = 0; t < frames; ++t) {
            Buffer<std::uint16_t> stack(Shape{3, 8, 8});
            for (Index i = 0; i < 3 * 8 * 8; ++i) stack.data()[i] = static_cast<std::uint16_t>(t * 100 + i);
            writeTiffStack<std::uint16_t>((real.path / ("f" + std::to_string(t == 0 ? 1 : t * 5) + ".tif")).string(),
                                          stack.view(), TiffCompression::None);
        }
        const DatasetManifest m = manifestOfOneStack(real.path);
        REQUIRE(m.files.size() == static_cast<std::size_t>(frames));
        CHECK(m.validate(real.path).empty());
        m.save(real.path / DatasetManifest::kFileName);
        CHECK(isFolderDataset(real.path.string()));

        const OpenResult opened = openDataset(real.path.string(), {});
        CHECK(opened.meta.dims.t == frames);   // a file each
        CHECK(opened.meta.dims.z == 3);        // the pages of one file
        CHECK(opened.meta.dims.c == 1);
    }

    SECTION("a folder with no TIFFs makes an empty manifest, not a broken one") {
        TempFolder empty;
        std::ofstream(empty.path / "readme.md") << "nothing here";
        CHECK(tiffNamesInOrder(empty.path).empty());
        CHECK(manifestOfOneStack(empty.path).files.empty());
    }
}
