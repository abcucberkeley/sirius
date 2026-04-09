#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdio>
#include <tiffio.h>

#include "sirius/tiff_io.hpp"

using namespace sirius;

// suppress libtiff's stderr messages — they're noise when we intentionally test error paths
namespace {
    const int silenceTiff = []() {
        TIFFSetErrorHandler(nullptr);
        TIFFSetWarningHandler(nullptr);
        return 0;
    }();

    // Write a tiled TIFF directly via libtiff to exercise the tiled reader code path.
    // The sirius write API only produces scanline TIFFs, so we need raw libtiff here.
    void writeTiledTiffRaw(const std::string& path, const Image<float>& img,
                           uint32_t tileW, uint32_t tileH) {
        TIFF* tif = TIFFOpen(path.c_str(), "w");
        if (!tif) throw std::runtime_error("Failed to create tiled TIFF: " + path);

        const auto rows = static_cast<uint32_t>(img.rows());
        const auto cols = static_cast<uint32_t>(img.cols());

        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,      cols);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH,     rows);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,   32);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT,    SAMPLEFORMAT_IEEEFP);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,     PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG,    PLANARCONFIG_CONTIG);
        TIFFSetField(tif, TIFFTAG_TILEWIDTH,       tileW);
        TIFFSetField(tif, TIFFTAG_TILELENGTH,      tileH);

        std::vector<float> tile(static_cast<size_t>(tileW) * tileH, 0.0f);
        for (uint32_t r = 0; r < rows; r += tileH) {
            for (uint32_t c = 0; c < cols; c += tileW) {
                for (uint32_t tr = 0; tr < tileH; ++tr)
                    for (uint32_t tc = 0; tc < tileW; ++tc) {
                        const uint32_t ir = r + tr, ic = c + tc;
                        tile[tr * tileW + tc] = (ir < rows && ic < cols) ? img(ir, ic) : 0.0f;
                    }
                TIFFWriteTile(tif, tile.data(), c, r, 0, 0);
            }
        }
        TIFFClose(tif);
    }
}

// RAII helper — deletes a file when it goes out of scope
struct TempFile {
    std::string path;
    explicit TempFile(std::string p) : path(std::move(p)) {}
    ~TempFile() { std::remove(path.c_str()); }
};

// -----------------------------------------------------------------------
// ImageStack — memory layout and in-memory operations
// -----------------------------------------------------------------------

TEST_CASE("ImageStack memory layout and slicing", "[ImageStack]") {
    const Eigen::Index depth = 3;
    const Eigen::Index rows = 4;
    const Eigen::Index cols = 5;

    ImageStack<uint16_t> stack(depth, rows, cols);
    for (Eigen::Index z = 0; z < depth; z++)
        for (Eigen::Index r = 0; r < rows; r++)
            for (Eigen::Index c = 0; c < cols; c++)
                stack(z, r, c) = static_cast<uint16_t>(z * 100 + r * 10 + c);

    SECTION("Dimensions are correct") {
        REQUIRE(stack.depth() == depth);
        REQUIRE(stack.rows() == rows);
        REQUIRE(stack.cols() == cols);
        REQUIRE(static_cast<Eigen::Index>(stack.size()) == depth * rows * cols);
        REQUIRE_FALSE(stack.empty());
    }

    SECTION("Slicing returns a view, not a copy") {
        auto slice1 = stack.slice(1);
        REQUIRE(slice1.rows() == rows);
        REQUIRE(slice1.cols() == cols);
        REQUIRE(slice1(0, 0) == 100);
        REQUIRE(slice1(3, 4) == 134);

        // writing through the slice modifies the original stack
        slice1(0, 0) = 999;
        REQUIRE(stack(1, 0, 0) == 999);
    }

    SECTION("Slices from different pages are independent regions") {
        auto s0 = stack.slice(0);
        auto s2 = stack.slice(2);
        s0(0, 0) = 111;
        REQUIRE(stack(0, 0, 0) == 111);
        REQUIRE(stack(2, 0, 0) == 200); // untouched
    }

    SECTION("Move semantics work") {
        ImageStack<uint16_t> moved = std::move(stack);
        REQUIRE(moved.depth() == depth);
        REQUIRE(moved.rows() == rows);
        REQUIRE(moved.cols() == cols);
        REQUIRE(stack.empty()); // source is empty after move
    }

    SECTION("Default constructed stack is empty") {
        ImageStack<float> empty;
        REQUIRE(empty.empty());
        REQUIRE(empty.size() == 0);
        REQUIRE(empty.depth() == 0);
    }
}

// -----------------------------------------------------------------------
// Single image round-trips
// -----------------------------------------------------------------------

TEST_CASE("Single image round-trip — all compression modes", "[tiff][io]") {
    auto compression = GENERATE(
        TiffCompression::None,
        TiffCompression::Lzw,
        TiffCompression::Deflate
    );
    INFO("Compression: " << static_cast<int>(compression));

    TempFile f("test_roundtrip.tiff");
    Image<float> original(32, 32);
    original.setRandom();

    writeTiff(f.path, original, compression);
    auto loaded = readTiff<float>(f.path);

    REQUIRE(loaded.rows() == original.rows());
    REQUIRE(loaded.cols() == original.cols());
    REQUIRE(loaded.isApprox(original));
}

TEST_CASE("Single image round-trip — all supported pixel types", "[tiff][io]") {
    SECTION("uint8") {
        TempFile f("test_uint8.tiff");
        Image<uint8_t> img(16, 16);
        img << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
               255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240,
               100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
               200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215,
               10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
               5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155,
               1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
               2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
               50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
               150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,
               70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
               170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185,
               90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
               190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205,
               110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
               210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225;
        writeTiff(f.path, img);
        REQUIRE(readTiff<uint8_t>(f.path) == img);
    }

    SECTION("uint16") {
        TempFile f("test_uint16.tiff");
        Image<uint16_t> img(8, 8);
        img.setZero();
        img(0, 0) = 0; img(0, 1) = 1000; img(0, 2) = 65535;
        writeTiff(f.path, img);
        REQUIRE(readTiff<uint16_t>(f.path) == img);
    }

    SECTION("double") {
        TempFile f("test_double.tiff");
        Image<double> img(8, 8);
        img.setRandom();
        writeTiff(f.path, img);
        REQUIRE(readTiff<double>(f.path).isApprox(img));
    }
}

TEST_CASE("Reading a TIFF as a different type converts pixels", "[tiff][io][conversion]") {
    TempFile f("test_conversion.tiff");

    // write uint16, read back as float
    Image<uint16_t> original(4, 4);
    original.fill(1000);
    writeTiff(f.path, original);

    auto loaded = readTiff<float>(f.path);
    REQUIRE(loaded.rows() == 4);
    REQUIRE(loaded.cols() == 4);
    // every pixel should be exactly 1000.0f
    REQUIRE(loaded.isApprox(Image<float>::Constant(4, 4, 1000.0f)));
}

TEST_CASE("Image dimensions are preserved exactly", "[tiff][io]") {
    // non-square, non-power-of-two dimensions
    auto [r, c] = GENERATE(table<int, int>({
        {1, 1},
        {1, 100},
        {100, 1},
        {17, 31},
        {256, 256},
        {512, 128}
    }));
    INFO("Dimensions: " << r << "x" << c);

    TempFile f("test_dims.tiff");
    Image<uint16_t> img(r, c);
    img.setRandom();
    writeTiff(f.path, img);

    auto loaded = readTiff<uint16_t>(f.path);
    REQUIRE(loaded.rows() == r);
    REQUIRE(loaded.cols() == c);
    REQUIRE(loaded == img);
}

// -----------------------------------------------------------------------
// Stack round-trips
// -----------------------------------------------------------------------

TEST_CASE("Stack round-trip preserves all pages", "[tiff][io][stack]") {
    TempFile f("test_stack.tiff");

    const Eigen::Index depth = 5, rows = 16, cols = 16;
    ImageStack<uint16_t> original(depth, rows, cols);
    for (Eigen::Index z = 0; z < depth; ++z)
        original.slice(z).fill(static_cast<uint16_t>(z * 1000));

    writeTiffStack(f.path, original);
    auto loaded = readTiffStack<uint16_t>(f.path);

    REQUIRE(loaded.depth() == depth);
    REQUIRE(loaded.rows() == rows);
    REQUIRE(loaded.cols() == cols);

    for (Eigen::Index z = 0; z < depth; ++z)
        REQUIRE(loaded.slice(z).isApprox(original.slice(z)));
}

TEST_CASE("Stack round-trip — compression modes", "[tiff][io][stack]") {
    auto compression = GENERATE(
        TiffCompression::None,
        TiffCompression::Lzw,
        TiffCompression::Deflate
    );
    INFO("Compression: " << static_cast<int>(compression));

    TempFile f("test_stack_compression.tiff");
    ImageStack<float> original(3, 32, 32);
    for (Eigen::Index z = 0; z < 3; ++z)
        original.slice(z).setRandom();

    writeTiffStack(f.path, original, compression);
    auto loaded = readTiffStack<float>(f.path);

    REQUIRE(loaded.depth() == 3);
    for (Eigen::Index z = 0; z < 3; ++z)
        REQUIRE(loaded.slice(z).isApprox(original.slice(z)));
}

TEST_CASE("Stack pages are independent — writing one page does not corrupt others", "[tiff][io][stack]") {
    TempFile f("test_stack_pages.tiff");

    ImageStack<uint16_t> original(4, 8, 8);
    for (Eigen::Index z = 0; z < 4; ++z)
        for (Eigen::Index r = 0; r < 8; ++r)
            for (Eigen::Index c = 0; c < 8; ++c)
                original(z, r, c) = static_cast<uint16_t>(z * 100 + r * 10 + c);

    writeTiffStack(f.path, original);
    auto loaded = readTiffStack<uint16_t>(f.path);

    for (Eigen::Index z = 0; z < 4; ++z)
        for (Eigen::Index r = 0; r < 8; ++r)
            for (Eigen::Index c = 0; c < 8; ++c)
                REQUIRE(loaded(z, r, c) == original(z, r, c));
}

// -----------------------------------------------------------------------
// Missing integer type round-trips (int8, int16, int32, uint32)
// -----------------------------------------------------------------------

TEST_CASE("Single image round-trip — signed and wider integer types", "[tiff][io]") {
    SECTION("int8 — min, zero, max") {
        TempFile f("test_int8.tiff");
        Image<int8_t> img(8, 8);
        img.setZero();
        img(0, 0) = -128; img(0, 1) = 0; img(0, 2) = 127;
        writeTiff(f.path, img);
        REQUIRE(readTiff<int8_t>(f.path) == img);
    }

    SECTION("int16 — min, zero, max") {
        TempFile f("test_int16.tiff");
        Image<int16_t> img(8, 8);
        img.setZero();
        img(0, 0) = -32768; img(0, 1) = 0; img(0, 2) = 32767;
        writeTiff(f.path, img);
        REQUIRE(readTiff<int16_t>(f.path) == img);
    }

    SECTION("uint32 — zero, mid, max") {
        TempFile f("test_uint32.tiff");
        Image<uint32_t> img(8, 8);
        img.setZero();
        img(0, 0) = 0; img(0, 1) = 65536; img(0, 2) = 0xFFFF'FFFFu;
        writeTiff(f.path, img);
        REQUIRE(readTiff<uint32_t>(f.path) == img);
    }

    SECTION("int32 — min, zero, max") {
        TempFile f("test_int32.tiff");
        Image<int32_t> img(8, 8);
        img.setZero();
        img(0, 0) = -2'147'483'647 - 1; img(0, 1) = 0; img(0, 2) = 2'147'483'647;
        writeTiff(f.path, img);
        REQUIRE(readTiff<int32_t>(f.path) == img);
    }
}

TEST_CASE("Stack round-trip — signed and wider integer types", "[tiff][io][stack]") {
    SECTION("int16") {
        TempFile f("test_stack_int16.tiff");
        ImageStack<int16_t> original(3, 8, 8);
        for (Eigen::Index z = 0; z < 3; ++z)
            original.slice(z).fill(static_cast<int16_t>((z - 1) * 1000));
        writeTiffStack(f.path, original);
        auto loaded = readTiffStack<int16_t>(f.path);
        REQUIRE(loaded.depth() == 3);
        for (Eigen::Index z = 0; z < 3; ++z)
            REQUIRE(loaded.slice(z) == original.slice(z));
    }

    SECTION("int32") {
        TempFile f("test_stack_int32.tiff");
        ImageStack<int32_t> original(3, 8, 8);
        for (Eigen::Index z = 0; z < 3; ++z)
            original.slice(z).fill(static_cast<int32_t>(z * 100'000));
        writeTiffStack(f.path, original);
        auto loaded = readTiffStack<int32_t>(f.path);
        REQUIRE(loaded.depth() == 3);
        for (Eigen::Index z = 0; z < 3; ++z)
            REQUIRE(loaded.slice(z) == original.slice(z));
    }
}

// -----------------------------------------------------------------------
// Tiled TIFF reading — exercises the tiled reader code path
// -----------------------------------------------------------------------

TEST_CASE("Tiled TIFF round-trip — tile-aligned dimensions", "[tiff][io][tiled]") {
    // 64x64 image with 16x16 tiles: every tile is fully populated, no edge clamping needed
    TempFile f("test_tiled_aligned.tiff");
    Image<float> original(64, 64);
    original.setRandom();

    writeTiledTiffRaw(f.path, original, 16, 16);
    auto loaded = readTiff<float>(f.path);

    REQUIRE(loaded.rows() == original.rows());
    REQUIRE(loaded.cols() == original.cols());
    REQUIRE(loaded.isApprox(original));
}

TEST_CASE("Tiled TIFF round-trip — non-tile-aligned dimensions (edge clamping)", "[tiff][io][tiled]") {
    // 50x50 with 16x16 tiles: right and bottom edge tiles are partially filled
    auto [rows, cols] = GENERATE(table<int, int>({
        {50,  50},   // partial tiles on both axes
        {1,   100},  // single row
        {100, 1},    // single column
        {17,  31},   // irregular, non-power-of-two
    }));
    INFO("Image " << rows << "x" << cols);

    TempFile f("test_tiled_edge.tiff");
    Image<float> original(rows, cols);
    original.setRandom();

    writeTiledTiffRaw(f.path, original, 16, 16);
    auto loaded = readTiff<float>(f.path);

    REQUIRE(loaded.rows() == rows);
    REQUIRE(loaded.cols() == cols);
    REQUIRE(loaded.isApprox(original));
}

// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------

TEST_CASE("Error handling", "[tiff][io][error_handling]") {
    SECTION("readTiff throws on nonexistent file") {
        REQUIRE_THROWS_AS(readTiff<uint8_t>("does_not_exist_12345.tiff"), std::runtime_error);
    }

    SECTION("readTiffStack throws on nonexistent file") {
        REQUIRE_THROWS_AS(readTiffStack<uint8_t>("does_not_exist_12345.tiff"), std::runtime_error);
    }

    SECTION("writeTiffStack throws on empty stack") {
        ImageStack<float> empty;
        REQUIRE_THROWS_WITH(writeTiffStack("empty.tiff", empty), "Cannot write empty stack");
    }
}
