// MRC / DeltaVision reading: the shipped raw.dv and otf.dv against their TIFF
// twins (the same arrays, written by cudasirecon's tools in both containers),
// and files written here for the layouts those two do not show.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "sirius/errors.hpp"
#include "sirius/mrc_io.hpp"
#include "sirius/tiff_io.hpp"

#include "mrc_fixture.hpp"
#include "temp_path.hpp"

using namespace sirius;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinRel;

namespace {
    const std::string kData = SIRIUS_TEST_DATA_DIR;
}

TEST_CASE("inspectMrc reads raw.dv's DeltaVision header", "[mrc]") {
    const MrcInfo i = inspectMrc(kData + "/raw.dv");
    CHECK(i.deltaVision);
    CHECK_FALSE(i.bigEndian);
    CHECK(i.mode == 2);
    CHECK(i.pixelType == PixelType::Float32);
    CHECK_FALSE(i.complex);
    CHECK(i.nx == 64);
    CHECK(i.ny == 64);
    CHECK(i.nz == 135);
    CHECK(i.width == 64);
    CHECK(i.height == 64);
    CHECK(i.sections == 135);
    CHECK(i.extendedHeaderBytes == 0);
    CHECK(i.dataOffset == 1024);
    CHECK(i.bytesOnDisk == 2212864);
    // DeltaVision cell lengths are the pixel sizes in micrometres
    CHECK_THAT(i.voxelUm[0], WithinRel(0.08, 1e-6));
    CHECK_THAT(i.voxelUm[1], WithinRel(0.08, 1e-6));
    CHECK_THAT(i.voxelUm[2], WithinRel(0.125, 1e-6));
    CHECK(i.waves == 1);
    CHECK(i.times == 1);
    CHECK(i.sequence == MrcSequence::ZTW);
    CHECK(i.wavelengthsNm[0] == 528);
    CHECK(i.planes() == 135);
    CHECK(i.sectionOf(0, 0, 17) == 17);
}

namespace {
    // raw.tif is Bio-Formats' export of raw.dv (its ImageJ Info tag says so), and
    // Bio-Formats reverses the rows of a DeltaVision section (MRC's origin is
    // bottom-left); the values are the same to the bit. cudasirecon's config for
    // the TIFF route negates the k0 angles of its .dv-route config for the same
    // reason. So section z of raw.dv is raw.tif's page z with the rows reversed.
    bool sameAsRowReversed(const float* section, const ImageStack<float>& tif, Index z) {
        const Index rows = tif.dimension(1), cols = tif.dimension(2);
        for (Index y = 0; y < rows; ++y)
            if (std::memcmp(section + y * cols, &tif(z, rows - 1 - y, 0), static_cast<std::size_t>(cols) * sizeof(float)) != 0) return false;
        return true;
    }
}

TEST_CASE("raw.dv reads value for value as raw.tif, rows in file order", "[mrc]") {
    MrcFile f(kData + "/raw.dv");
    const Buffer<float> dv = f.readStack<float>();
    REQUIRE(dv.shape() == Shape{135, 64, 64});
    const ImageStack<float> tif = readTiffStack<float>(kData + "/raw.tif");
    REQUIRE(tif.dimension(0) == 135);
    REQUIRE(tif.dimension(1) == 64);
    REQUIRE(tif.dimension(2) == 64);
    const std::size_t plane = 64 * 64;
    for (Index z = 0; z < 135; ++z) {
        INFO("section " << z);
        REQUIRE(sameAsRowReversed(dv.data() + static_cast<std::size_t>(z) * plane, tif, z));
    }
    // and not the same without the reversal: the two files do differ as stored
    CHECK(std::memcmp(dv.data(), tif.data(), plane * sizeof(float)) != 0);

    SECTION("a range of sections, and a conversion") {
        const Buffer<float> some = f.readSections<float>(7, 3);
        REQUIRE(some.shape() == Shape{3, 64, 64});
        for (Index k = 0; k < 3; ++k) CHECK(sameAsRowReversed(some.data() + static_cast<std::size_t>(k) * plane, tif, 7 + k));
        const Buffer<double> last = f.readSections<double>(134, 1);
        REQUIRE(last.shape() == Shape{1, 64, 64});
        CHECK(last.data()[5] == static_cast<double>(tif(134, 63, 5)));
        CHECK(last.data()[plane - 1] == static_cast<double>(tif(134, 0, 63)));
    }
    SECTION("past the last section is out of range, an empty read is empty") {
        CHECK_THROWS_AS(f.readSections<float>(134, 2), std::out_of_range);
        CHECK_THROWS_AS(f.readSections<float>(135, 1), std::out_of_range);
        CHECK(f.readSections<float>(135, 0).shape() == Shape{0, 64, 64});
    }
}

TEST_CASE("otf.dv, complex float32, reads as rows of (re, im) pairs like otf.tif", "[mrc]") {
    const MrcInfo i = inspectMrc(kData + "/otf.dv");
    CHECK(i.deltaVision);
    CHECK(i.mode == 4);
    CHECK(i.complex);
    CHECK(i.pixelType == PixelType::Float32);
    CHECK(i.nx == 65);
    CHECK(i.width == 130);
    CHECK(i.ny == 129);
    CHECK(i.nz == 3);
    CHECK(i.dataOffset == 1024);
    CHECK(i.bytesOnDisk == 202264);
    CHECK(i.minValue == 0.f);
    CHECK(i.maxValue == 1.f);
    const Buffer<float> dv = MrcFile(kData + "/otf.dv").readStack<float>();
    REQUIRE(dv.shape() == Shape{3, 129, 130});
    // the same table layout as otf.tif (orders, kr, 2 * kz: re, im, re, im ...),
    // though not the same OTF: the two files hold different computations
    // (otf.tif's imaginary parts reach 0.13, otf.dv's stay below 1e-6)
    const ImageStack<float> tif = readTiffStack<float>(kData + "/otf.tif");
    REQUIRE(tif.dimension(0) == 3);
    REQUIRE(tif.dimension(1) == 129);
    REQUIRE(tif.dimension(2) == 130);
    // both are normalised to 1 + 0i at the origin of order 0
    CHECK(dv.data()[0] == 1.f);
    CHECK(dv.data()[1] == 0.f);
    CHECK(tif(0, 0, 0) == 1.f);
    CHECK(tif(0, 0, 1) == 0.f);
    // the sections are what the file holds, bit for bit
    std::ifstream in(kData + "/otf.dv", std::ios::binary);
    std::vector<float> raw(3 * 129 * 130);
    in.seekg(1024);
    REQUIRE(in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size() * sizeof(float))));
    CHECK(std::memcmp(dv.data(), raw.data(), raw.size() * sizeof(float)) == 0);
}

namespace {
    // a sample that names its own (w, t, z, y, x); fits int16 for the sizes used
    double stamp(int w, int t, int z, int y, int x) { return w * 10000 + t * 1000 + z * 100 + y * 10 + x; }
}

TEST_CASE("wavelengths and time points map to sections by the DeltaVision sequence", "[mrc]") {
    const int sequence = GENERATE(0, 1, 2);
    INFO("sequence " << sequence);
    test::MrcSpec s;
    s.nx = 8;
    s.ny = 6;
    s.planes = 4;
    s.waves = 2;
    s.times = 3;
    s.sequence = sequence;
    s.mode = 1;
    s.wavelengths = {488, 561, 0, 0, 0};
    s.title = "written by test_mrc_io";
    test::TempFile f("dv_seq", ".dv");
    test::writeMrc(f.str, s, stamp);

    MrcFile file(f.str);
    const MrcInfo& i = file.info();
    CHECK(i.deltaVision);
    CHECK(i.pixelType == PixelType::Int16);
    CHECK(i.sections == 24);
    CHECK(i.waves == 2);
    CHECK(i.times == 3);
    CHECK(i.planes() == 4);
    CHECK(static_cast<int>(i.sequence) == sequence);
    CHECK(i.wavelengthsNm[0] == 488);
    CHECK(i.wavelengthsNm[1] == 561);
    REQUIRE(i.titles.size() == 1);
    CHECK(i.titles[0] == "written by test_mrc_io");
    for (int w = 0; w < 2; ++w)
        for (int t = 0; t < 3; ++t)
            for (int z = 0; z < 4; ++z) {
                const Buffer<float> sec = file.readSections<float>(static_cast<std::size_t>(i.sectionOf(w, t, z)), 1);
                CHECK(sec.data()[0] == static_cast<float>(stamp(w, t, z, 0, 0)));
                CHECK(sec.data()[3 * 8 + 5] == static_cast<float>(stamp(w, t, z, 3, 5)));
            }
    // every section is some (w, t, z): the map is a bijection
    std::vector<bool> seen(24, false);
    for (int w = 0; w < 2; ++w)
        for (int t = 0; t < 3; ++t)
            for (int z = 0; z < 4; ++z) seen[static_cast<std::size_t>(i.sectionOf(w, t, z))] = true;
    CHECK(std::count(seen.begin(), seen.end(), true) == 24);
}

TEST_CASE("a big-endian DeltaVision file reads the same", "[mrc]") {
    test::MrcSpec s;
    s.bigEndian = true;
    s.mode = 2;
    s.pixelX = 0.065f;
    s.pixelY = 0.065f;
    s.pixelZ = 0.2f;
    test::TempFile f("dv_be", ".dv");
    test::writeMrc(f.str, s, [](int, int, int z, int y, int x) { return 0.5 + z * 100 + y * 10 + x; });
    MrcFile file(f.str);
    CHECK(file.info().bigEndian);
    CHECK(file.info().deltaVision);
    CHECK_THAT(file.info().voxelUm[0], WithinRel(0.065, 1e-6));
    CHECK_THAT(file.info().voxelUm[2], WithinRel(0.2, 1e-6));
    const Buffer<float> all = file.readStack<float>();
    REQUIRE(all.shape() == Shape{4, 6, 8});
    CHECK(all.data()[(2 * 6 + 3) * 8 + 7] == 237.5f);
}

TEST_CASE("the integer and complex pixel modes", "[mrc]") {
    SECTION("uint16 (mode 6) converts to float on request") {
        test::MrcSpec s;
        s.mode = 6;
        test::TempFile f("dv_u16", ".dv");
        test::writeMrc(f.str, s, [](int, int, int z, int y, int x) { return 60000 + z * 10 + y + x; });
        MrcFile file(f.str);
        CHECK(file.info().pixelType == PixelType::UInt16);
        const Buffer<std::uint16_t> raw = file.readSections<std::uint16_t>(1, 1);
        CHECK(raw.data()[2 * 8 + 3] == 60015);
        const Buffer<float> asFloat = file.readSections<float>(1, 1);
        CHECK(asFloat.data()[2 * 8 + 3] == 60015.f);
    }
    SECTION("bytes (mode 0) are unsigned in a DeltaVision file, signed in MRC2014") {
        test::MrcSpec s;
        s.mode = 0;
        test::TempFile f("dv_u8", ".dv");
        test::writeMrc(f.str, s, [](int, int, int, int y, int x) { return 200 + y + x; });
        CHECK(inspectMrc(f.str).pixelType == PixelType::UInt8);
        CHECK(MrcFile(f.str).readSections<std::uint8_t>(0, 1).data()[1] == 201);
        s.deltaVision = false;
        test::TempFile g("mrc_i8", ".mrc");
        test::writeMrc(g.str, s, [](int, int, int, int y, int x) { return y + x; });
        CHECK(inspectMrc(g.str).pixelType == PixelType::Int8);
    }
    SECTION("complex float32 (mode 4) is a row of 2 nx values") {
        test::MrcSpec s;
        s.mode = 4;
        test::TempFile f("dv_c", ".dv");
        test::writeMrc(f.str, s, [](int, int, int, int y, int x) { return y * 100 + x; });
        MrcFile file(f.str);
        CHECK(file.info().complex);
        CHECK(file.info().nx == 8);
        CHECK(file.info().width == 16);
        const Buffer<float> sec = file.readSections<float>(0, 1);
        REQUIRE(sec.shape() == Shape{1, 6, 16});
        CHECK(sec.data()[2 * 16 + 15] == 215.f);
    }
}

TEST_CASE("an MRC2014 header: Angstrom cells become micrometres, the sections are one z stack", "[mrc]") {
    test::MrcSpec s;
    s.deltaVision = false;
    s.planes = 5;
    s.pixelX = 0.1f;
    s.pixelY = 0.1f;
    s.pixelZ = 0.3f;
    test::TempFile f("mrc_plain", ".mrc");
    test::writeMrc(f.str, s, [](int, int, int z, int y, int x) { return z + y + x; });
    const MrcInfo i = inspectMrc(f.str);
    CHECK_FALSE(i.deltaVision);
    CHECK_FALSE(i.bigEndian);
    CHECK(i.sections == 5);
    CHECK(i.waves == 1);
    CHECK(i.times == 1);
    CHECK(i.planes() == 5);
    CHECK_THAT(i.voxelUm[0], WithinRel(0.1, 1e-5));
    CHECK_THAT(i.voxelUm[1], WithinRel(0.1, 1e-5));
    CHECK_THAT(i.voxelUm[2], WithinRel(0.3, 1e-5));
    s.bigEndian = true;
    test::TempFile g("mrc_be", ".mrc");
    test::writeMrc(g.str, s, [](int, int, int z, int y, int x) { return z + y + x; });
    CHECK(inspectMrc(g.str).bigEndian);
    CHECK(MrcFile(g.str).readSections<float>(4, 1).data()[5 * 8 + 7] == 16.f);
}

TEST_CASE("an extended header moves the sections", "[mrc]") {
    test::MrcSpec s;
    s.extendedHeaderBytes = 4 * (8 + 32) * 4;   // what OMX writes: 8 ints and 32 floats per section
    test::TempFile f("dv_ext", ".dv");
    test::writeMrc(f.str, s, [](int, int, int z, int y, int x) { return z * 100 + y * 10 + x; });
    const MrcInfo i = inspectMrc(f.str);
    CHECK(i.extendedHeaderBytes == 640);
    CHECK(i.dataOffset == 1024 + 640);
    CHECK(MrcFile(f.str).readSections<float>(3, 1).data()[2 * 8 + 1] == 321.f);
}

TEST_CASE("files that are not MRC stacks are refused with the reason", "[mrc]") {
    CHECK_THROWS_AS(inspectMrc(kData + "/no-such.dv"), IoError);
    // a TIFF: its first bytes are no plausible extents
    CHECK_THROWS_AS(inspectMrc(kData + "/raw.tif"), IoError);
    SECTION("shorter than the header") {
        test::TempFile f("dv_short", ".dv");
        std::ofstream(f.str, std::ios::binary) << "DeltaVision?";
        CHECK_THROWS_WITH(inspectMrc(f.str), ContainsSubstring("1024 bytes"));
    }
    SECTION("truncated after the header") {
        std::ifstream in(kData + "/raw.dv", std::ios::binary);
        std::vector<char> head(100000);
        in.read(head.data(), static_cast<std::streamsize>(head.size()));
        test::TempFile f("dv_trunc", ".dv");
        std::ofstream(f.str, std::ios::binary).write(head.data(), static_cast<std::streamsize>(head.size()));
        CHECK_THROWS_WITH(inspectMrc(f.str), ContainsSubstring("2212864 bytes of header and sections expected, the file has 100000"));
    }
    SECTION("a pixel mode this reader lacks") {
        test::MrcSpec s;
        s.mode = 12;   // MRC2014 float16
        test::TempFile f("dv_mode", ".dv");
        test::writeMrc(f.str, s, [](int, int, int, int, int) { return 0.0; });
        CHECK_THROWS_WITH(inspectMrc(f.str), ContainsSubstring("pixel mode 12 is not read"));
    }
}

TEST_CASE("isMrcName goes by the extension, any case", "[mrc]") {
    CHECK(isMrcName("/data/cells.dv"));
    CHECK(isMrcName("C:/data/CELLS.DV"));
    CHECK(isMrcName("tomo.mrc"));
    CHECK_FALSE(isMrcName("stack.tif"));
    CHECK_FALSE(isMrcName("stack.ome.tif"));
    CHECK_FALSE(isMrcName("dv"));
}
