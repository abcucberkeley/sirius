// The sample layouts, bit depths, codecs and metadata the TIFF reader
// decodes beyond plain one-sample 8/16/32-bit stacks: several samples per
// pixel in both planar configurations, packed 1..31-bit samples, float16,
// 24-bit integers, big-endian files, sparse files, palette images, pages
// split across threads, OME / ImageJ metadata and OME series. Fixtures are
// written here with libtiff from a known pattern; tests/data/tifffile holds
// files written by tifffile (bindings/tests checks every reader feature
// against tifffile itself).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <tiffio.h>

#include "sirius/buffer.hpp"
#include "sirius/tiff_io.hpp"

#include "temp_path.hpp"
#include "tiff_internal.hpp"   // detail::libtiffReadOpens

using namespace sirius;

namespace {

    struct TempFile {
        std::string path;
        explicit TempFile(const char* suffix) : path(test::uniqueTempPath("tiffformats", suffix).string()) {}
        ~TempFile() { std::remove(path.c_str()); }
    };

    struct TiffDeleter {
        void operator()(TIFF* t) const { TIFFClose(t); }
    };
    using TiffPtr = std::unique_ptr<TIFF, TiffDeleter>;

    const int silenceTiff = [] {
        TIFFSetErrorHandler(nullptr);
        TIFFSetWarningHandler(nullptr);
        return 0;
    }();

    // What to write.
    struct Spec {
        uint32_t width = 37, height = 29;
        uint16_t spp = 1, bps = 8, format = SAMPLEFORMAT_UINT, photometric = PHOTOMETRIC_MINISBLACK;
        uint16_t planar = PLANARCONFIG_CONTIG;
        bool tiled = false;
        uint32_t tile = 16;          // tile edge (multiple of 16)
        uint32_t rowsPerStrip = 5;
        uint16_t compression = COMPRESSION_NONE;
        uint16_t predictor = 0;      // 0: no tag
        bool bigEndian = false;
        int pages = 2;
        bool sparse = false;         // leave every other chunk unwritten
    };

    // The value of sample s of pixel (y, x) of page p, before it is
    // reduced to `bps` bits (or used as a float).
    uint64_t rawValue(int p, int s, uint32_t y, uint32_t x) {
        return static_cast<uint64_t>(p) * 131u + static_cast<uint64_t>(s) * 37u + y * 7u + x * 3u + (x * y) % 11u;
    }
    uint64_t intValue(const Spec& sp, int p, int s, uint32_t y, uint32_t x) {
        const uint64_t v = rawValue(p, s, y, x) * (sp.bps > 16 ? 65599u : 1u);
        return sp.bps >= 64 ? v : v & ((uint64_t{1} << sp.bps) - 1);
    }
    // what the reader returns for an integer sample (sign-extended)
    double intExpected(const Spec& sp, int p, int s, uint32_t y, uint32_t x) {
        const uint64_t v = intValue(sp, p, s, y, x);
        if (sp.format == SAMPLEFORMAT_INT && sp.bps < 64 && ((v >> (sp.bps - 1)) & 1))
            return static_cast<double>(static_cast<int64_t>(v | ~((uint64_t{1} << sp.bps) - 1)));
        return static_cast<double>(v);
    }
    double floatValue(int p, int s, uint32_t y, uint32_t x) {
        return (static_cast<double>(rawValue(p, s, y, x)) - 40.0) * 0.25;   // exact in float16 for small values
    }

    uint16_t floatToHalf(float f) {   // exact for the values floatValue makes
        uint32_t b;
        std::memcpy(&b, &f, 4);
        const uint32_t sign = (b >> 16) & 0x8000u;
        const int32_t exp = static_cast<int32_t>((b >> 23) & 0xFF) - 127 + 15;
        const uint32_t mant = b & 0x7FFFFFu;
        if ((b & 0x7FFFFFFFu) == 0) return static_cast<uint16_t>(sign);
        if (exp <= 0) {   // subnormal half
            const uint32_t m = (mant | 0x800000u) >> (1 - exp + 13);
            return static_cast<uint16_t>(sign | m);
        }
        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
    }

    double expected(const Spec& sp, int p, int s, uint32_t y, uint32_t x) {
        if (sp.format == SAMPLEFORMAT_IEEEFP) return floatValue(p, s, y, x);
        return intExpected(sp, p, s, y, x);
    }

    // One chunk row: pixels [x0, x0 + w) of row y, the samples [s0, s0 + ns),
    // packed as the file stores them (host byte order: libtiff swaps).
    void packRow(const Spec& sp, int p, uint32_t y, uint32_t x0, uint32_t w, int s0, int ns, uint8_t* out,
                 std::size_t rowBytes) {
        std::memset(out, 0, rowBytes);
        for (uint32_t i = 0; i < w; ++i) {
            for (int k = 0; k < ns; ++k) {
                const uint32_t x = x0 + i;
                const int s = s0 + k;
                const std::size_t idx = static_cast<std::size_t>(i) * ns + k;
                const bool inside = x < sp.width && y < sp.height;
                if (sp.format == SAMPLEFORMAT_IEEEFP) {
                    const double v = inside ? floatValue(p, s, y, x) : 0.0;
                    if (sp.bps == 16) {
                        const uint16_t h = floatToHalf(static_cast<float>(v));
                        std::memcpy(out + idx * 2, &h, 2);
                    } else if (sp.bps == 32) {
                        const float f = static_cast<float>(v);
                        std::memcpy(out + idx * 4, &f, 4);
                    } else {
                        std::memcpy(out + idx * 8, &v, 8);
                    }
                    continue;
                }
                const uint64_t v = inside ? intValue(sp, p, s, y, x) : 0;
                switch (sp.bps) {
                    case 8: out[idx] = static_cast<uint8_t>(v); break;
                    case 16: {
                        const uint16_t t = static_cast<uint16_t>(v);
                        std::memcpy(out + idx * 2, &t, 2);
                        break;
                    }
                    case 32: {
                        const uint32_t t = static_cast<uint32_t>(v);
                        std::memcpy(out + idx * 4, &t, 4);
                        break;
                    }
                    case 24: {   // host (little-endian) byte order
                        out[idx * 3] = static_cast<uint8_t>(v);
                        out[idx * 3 + 1] = static_cast<uint8_t>(v >> 8);
                        out[idx * 3 + 2] = static_cast<uint8_t>(v >> 16);
                        break;
                    }
                    default: {   // MSB-first bit stream
                        const uint64_t bit0 = idx * sp.bps;
                        for (uint16_t b = 0; b < sp.bps; ++b) {
                            const uint64_t bit = bit0 + b;
                            if ((v >> (sp.bps - 1 - b)) & 1) out[bit >> 3] |= static_cast<uint8_t>(0x80u >> (bit & 7));
                        }
                    }
                }
            }
        }
    }

    void write(const std::string& path, const Spec& sp) {
        const std::string mode = std::string("w") + (sp.bigEndian ? "b" : "l");
        TiffPtr tif(TIFFOpen(path.c_str(), mode.c_str()));
        REQUIRE(tif);
        for (int p = 0; p < sp.pages; ++p) {
            TIFF* t = tif.get();
            TIFFSetField(t, TIFFTAG_IMAGEWIDTH, sp.width);
            TIFFSetField(t, TIFFTAG_IMAGELENGTH, sp.height);
            TIFFSetField(t, TIFFTAG_BITSPERSAMPLE, sp.bps);
            TIFFSetField(t, TIFFTAG_SAMPLESPERPIXEL, sp.spp);
            TIFFSetField(t, TIFFTAG_SAMPLEFORMAT, sp.format);
            TIFFSetField(t, TIFFTAG_PHOTOMETRIC, sp.photometric);
            TIFFSetField(t, TIFFTAG_PLANARCONFIG, sp.planar);
            TIFFSetField(t, TIFFTAG_COMPRESSION, sp.compression);
            if (sp.predictor) TIFFSetField(t, TIFFTAG_PREDICTOR, sp.predictor);
            if (sp.spp == 4) {
                const uint16_t extra = EXTRASAMPLE_UNASSALPHA;
                TIFFSetField(t, TIFFTAG_EXTRASAMPLES, 1, &extra);
            }
            if (sp.photometric == PHOTOMETRIC_PALETTE) {
                const std::size_t n = std::size_t{1} << sp.bps;
                std::vector<uint16_t> r(n), g(n), b(n);
                for (std::size_t i = 0; i < n; ++i) {
                    r[i] = static_cast<uint16_t>(i * 257);
                    g[i] = static_cast<uint16_t>(65535 - i * 257);
                    b[i] = static_cast<uint16_t>(i * 100);
                }
                TIFFSetField(t, TIFFTAG_COLORMAP, r.data(), g.data(), b.data());
            }
            const bool separate = sp.planar == PLANARCONFIG_SEPARATE;
            const int planes = separate ? sp.spp : 1;
            const int perChunk = separate ? 1 : sp.spp;
            if (sp.tiled) {
                TIFFSetField(t, TIFFTAG_TILEWIDTH, sp.tile);
                TIFFSetField(t, TIFFTAG_TILELENGTH, sp.tile);
                const std::size_t rowBytes = (static_cast<std::size_t>(sp.tile) * perChunk * sp.bps + 7) / 8;
                std::vector<uint8_t> buf(rowBytes * sp.tile);
                int k = 0;
                for (int s = 0; s < planes; ++s)
                    for (uint32_t ty = 0; ty < sp.height; ty += sp.tile)
                        for (uint32_t tx = 0; tx < sp.width; tx += sp.tile, ++k) {
                            if (sp.sparse && k % 2 == 1) continue;
                            for (uint32_t r = 0; r < sp.tile; ++r)
                                packRow(sp, p, ty + r, tx, sp.tile, separate ? s : 0, perChunk, buf.data() + r * rowBytes, rowBytes);
                            REQUIRE(TIFFWriteTile(t, buf.data(), tx, ty, 0, static_cast<uint16_t>(s)) >= 0);
                        }
            } else {
                TIFFSetField(t, TIFFTAG_ROWSPERSTRIP, sp.rowsPerStrip);
                const std::size_t rowBytes = (static_cast<std::size_t>(sp.width) * perChunk * sp.bps + 7) / 8;
                std::vector<uint8_t> buf(rowBytes * sp.rowsPerStrip);
                int k = 0;
                for (int s = 0; s < planes; ++s)
                    for (uint32_t y0 = 0; y0 < sp.height; y0 += sp.rowsPerStrip, ++k) {
                        if (sp.sparse && k % 2 == 1) continue;
                        const uint32_t rows = std::min(sp.rowsPerStrip, sp.height - y0);
                        for (uint32_t r = 0; r < rows; ++r)
                            packRow(sp, p, y0 + r, 0, sp.width, separate ? s : 0, perChunk, buf.data() + r * rowBytes, rowBytes);
                        const tstrip_t strip = TIFFComputeStrip(t, y0, static_cast<uint16_t>(s));
                        REQUIRE(TIFFWriteEncodedStrip(t, strip, buf.data(), static_cast<tmsize_t>(rows * rowBytes)) >= 0);
                    }
            }
            REQUIRE(TIFFWriteDirectory(t));
        }
    }

    // Whether chunk (plane s, row/col of the point) was left out of a sparse file.
    bool missing(const Spec& sp, int s, uint32_t y, uint32_t x) {
        if (!sp.sparse) return false;
        const int plane = sp.planar == PLANARCONFIG_SEPARATE ? s : 0;
        int k = 0;
        if (sp.tiled) {
            const uint32_t across = (sp.width + sp.tile - 1) / sp.tile, down = (sp.height + sp.tile - 1) / sp.tile;
            k = static_cast<int>(plane * across * down + (y / sp.tile) * across + x / sp.tile);
        } else {
            const uint32_t down = (sp.height + sp.rowsPerStrip - 1) / sp.rowsPerStrip;
            k = static_cast<int>(plane * down + y / sp.rowsPerStrip);
        }
        return k % 2 == 1;
    }

    // `got` holds pages [0, pages) x samples [s0, s0 + ns) of region r.
    template <typename T>
    void requirePattern(const Buffer<T>& got, const Spec& sp, Region r, int s0, int ns, int firstPage = 0) {
        const Index pages = got.dim(0);
        REQUIRE(got.rank() == (ns == 1 ? 3 : 4));
        if (ns > 1) REQUIRE(got.dim(1) == ns);
        REQUIRE(got.dim(got.rank() - 2) == static_cast<Index>(r.height));
        REQUIRE(got.dim(got.rank() - 1) == static_cast<Index>(r.width));
        const T* d = got.data();
        for (Index p = 0; p < pages; ++p)
            for (int k = 0; k < ns; ++k)
                for (uint32_t y = 0; y < r.height; ++y)
                    for (uint32_t x = 0; x < r.width; ++x) {
                        const uint32_t fy = r.y + y, fx = r.x + x;
                        const int s = s0 + k;
                        double want = expected(sp, static_cast<int>(p) + firstPage, s, fy, fx);
                        if (missing(sp, s, fy, fx)) want = 0.0;
                        const double have = static_cast<double>(d[((p * ns + k) * r.height + y) * r.width + x]);
                        if (have != static_cast<double>(static_cast<T>(want)))
                            FAIL("page " << p << " sample " << s << " (" << fy << "," << fx << "): got " << have
                                         << " expected " << want);
                    }
    }

    template <typename T>
    void checkAllReads(const Spec& sp) {
        TempFile f(".tif");
        write(f.path, sp);
        TiffFile file(f.path);
        const TiffInfo& info = file.info();
        REQUIRE(info.pageCount() == static_cast<std::size_t>(sp.pages));
        REQUIRE(info.samplesPerPixel() == sp.spp);
        REQUIRE(info.page(0).bitsPerSample == sp.bps);
        REQUIRE(info.page(0).planarConfig == sp.planar);
        REQUIRE(info.page(0).decodable());
        REQUIRE(info.bigEndian == sp.bigEndian);
        const Region all{0, 0, sp.width, sp.height};
        // every sample, the whole stack
        requirePattern(file.readStack<T>(), sp, all, 0, sp.spp);
        // a region straddling chunk edges, one page
        const Region part{3, 4, sp.width - 9, sp.height - 7};
        requirePattern(file.readRegion<T>(part), sp, part, 0, sp.spp);
        // one sample, then a sample range
        TiffReadOptions one;
        one.firstSample = static_cast<uint16_t>(sp.spp - 1);
        one.sampleCount = 1;
        requirePattern(file.readPages<T>(1, 1, one), sp, all, sp.spp - 1, 1, 1);
        if (sp.spp >= 3) {
            TiffReadOptions two;
            two.firstSample = 1;
            two.sampleCount = 2;
            requirePattern(file.readRegion<T>(part, 0, two), sp, part, 1, 2);
        }
        // single-threaded decoding gives the same
        TiffReadOptions serial;
        serial.maxThreads = 1;
        requirePattern(file.readStack<T>(serial), sp, all, 0, sp.spp);
    }

} // namespace

TEST_CASE("Several samples per pixel read as channel planes, both planar configurations", "[tiff][samples]") {
    Spec sp;
    sp.spp = GENERATE(as<uint16_t>{}, 3, 4, 2);
    sp.planar = GENERATE(as<uint16_t>{}, PLANARCONFIG_CONTIG, PLANARCONFIG_SEPARATE);
    sp.tiled = GENERATE(false, true);
    sp.photometric = sp.spp >= 3 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK;
    CAPTURE(sp.spp, sp.planar, sp.tiled);
    SECTION("uint8") {
        sp.bps = 8;
        checkAllReads<uint8_t>(sp);
    }
    SECTION("uint16 with LZW and the horizontal predictor") {
        sp.bps = 16;
        sp.compression = COMPRESSION_LZW;
        sp.predictor = PREDICTOR_HORIZONTAL;
        checkAllReads<uint16_t>(sp);
    }
    SECTION("float32 with Deflate and the floating-point predictor, read as double") {
        sp.bps = 32;
        sp.format = SAMPLEFORMAT_IEEEFP;
        sp.compression = COMPRESSION_ADOBE_DEFLATE;
        sp.predictor = PREDICTOR_FLOATINGPOINT;
        checkAllReads<double>(sp);
    }
    SECTION("uint16 with ZSTD and the horizontal predictor") {
        sp.bps = 16;
        sp.compression = COMPRESSION_ZSTD;
        sp.predictor = PREDICTOR_HORIZONTAL;
        checkAllReads<uint16_t>(sp);
    }
    SECTION("float32 with ZSTD and the floating-point predictor") {
        sp.bps = 32;
        sp.format = SAMPLEFORMAT_IEEEFP;
        sp.compression = COMPRESSION_ZSTD;
        sp.predictor = PREDICTOR_FLOATINGPOINT;
        checkAllReads<float>(sp);
    }
}

TEST_CASE("RGB metadata: photometric, extra samples, shapes", "[tiff][samples]") {
    Spec sp;
    sp.spp = 4;
    sp.photometric = PHOTOMETRIC_RGB;
    TempFile f(".tif");
    write(f.path, sp);
    TiffFile file(f.path);
    const TiffImageInfo& p = file.info().page(0);
    CHECK(p.photometric == PHOTOMETRIC_RGB);
    CHECK(p.samplesPerPixel == 4);
    REQUIRE(p.extraSamples.size() == 1);
    CHECK(p.extraSamples[0] == EXTRASAMPLE_UNASSALPHA);
    CHECK(file.readShape(2, sp.height, sp.width) == Shape{2, 4, sp.height, sp.width});
    TiffReadOptions one;
    one.firstSample = 2;
    one.sampleCount = 1;
    CHECK(file.readShape(2, sp.height, sp.width, one) == Shape{2, sp.height, sp.width});
    TiffReadOptions bad;
    bad.firstSample = 4;
    CHECK_THROWS_AS(file.readStack<uint8_t>(bad), std::out_of_range);
    bad.firstSample = 2;
    bad.sampleCount = 3;
    CHECK_THROWS_AS(file.readStack<uint8_t>(bad), std::out_of_range);
    // the Eigen API holds one sample per pixel and says so
    CHECK_THROWS_WITH(readTiffStack<uint8_t>(f.path), Catch::Matchers::ContainsSubstring("samples per pixel"));
    // a destination of the wrong rank is refused
    Buffer<uint8_t> flat(Shape{2, sp.height, sp.width});
    CHECK_THROWS(file.decode<uint8_t>(file.info().pages, Region{}, flat.view()));
}

TEST_CASE("Packed and unusual bit depths are unpacked", "[tiff][bits]") {
    Spec sp;
    sp.bps = GENERATE(as<uint16_t>{}, 1, 2, 4, 6, 10, 12, 14, 24);
    sp.tiled = GENERATE(false, true);
    sp.spp = GENERATE(as<uint16_t>{}, 1, 3);
    sp.photometric = sp.spp == 3 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK;
    CAPTURE(sp.bps, sp.tiled, sp.spp);
    TempFile f(".tif");
    write(f.path, sp);
    TiffFile file(f.path);
    const PixelType want = sp.bps <= 8 ? PixelType::UInt8 : sp.bps <= 16 ? PixelType::UInt16
                                                                         : PixelType::UInt32;
    CHECK(file.info().pixelType() == want);
    const Region all{0, 0, sp.width, sp.height};
    const Region part{5, 2, 17, 13};
    if (sp.bps <= 8) {
        requirePattern(file.readStack<uint8_t>(), sp, all, 0, sp.spp);
        requirePattern(file.readRegion<uint8_t>(part), sp, part, 0, sp.spp);
    } else if (sp.bps <= 16) {
        requirePattern(file.readStack<uint16_t>(), sp, all, 0, sp.spp);
        requirePattern(file.readRegion<uint16_t>(part), sp, part, 0, sp.spp);
    } else {
        requirePattern(file.readStack<uint32_t>(), sp, all, 0, sp.spp);
        requirePattern(file.readRegion<uint32_t>(part), sp, part, 0, sp.spp);
    }
    requirePattern(file.readStack<float>(), sp, all, 0, sp.spp);   // converted on the way
}

TEST_CASE("Signed packed, 24-bit and 8-bit samples are sign-extended", "[tiff][bits]") {
    Spec sp;
    sp.format = SAMPLEFORMAT_INT;
    sp.bps = GENERATE(as<uint16_t>{}, 4, 8, 12, 16, 24, 32);
    CAPTURE(sp.bps);
    TempFile f(".tif");
    write(f.path, sp);
    TiffFile file(f.path);
    const Region all{0, 0, sp.width, sp.height};
    requirePattern(file.readStack<double>(), sp, all, 0, 1);
    requirePattern(file.readStack<int32_t>(), sp, all, 0, 1);
}

TEST_CASE("float16 widens to float32; float64 reads as itself", "[tiff][float]") {
    Spec sp;
    sp.format = SAMPLEFORMAT_IEEEFP;
    sp.bps = GENERATE(as<uint16_t>{}, 16, 64);
    sp.compression = GENERATE(as<uint16_t>{}, COMPRESSION_NONE, COMPRESSION_ADOBE_DEFLATE);
    sp.predictor = sp.compression == COMPRESSION_NONE ? 0 : PREDICTOR_FLOATINGPOINT;
    sp.tiled = GENERATE(false, true);
    CAPTURE(sp.bps, sp.compression, sp.tiled);
    TempFile f(".tif");
    write(f.path, sp);
    TiffFile file(f.path);
    CHECK(file.info().pixelType() == (sp.bps == 16 ? PixelType::Float32 : PixelType::Float64));
    requirePattern(file.readStack<double>(), sp, Region{0, 0, sp.width, sp.height}, 0, 1);
    requirePattern(file.readRegion<float>(Region{1, 1, 20, 20}), sp, Region{1, 1, 20, 20}, 0, 1);
}

TEST_CASE("Big-endian files decode to host order", "[tiff][endian]") {
    Spec sp;
    sp.bigEndian = true;
    sp.spp = GENERATE(as<uint16_t>{}, 1, 3);
    sp.photometric = sp.spp == 3 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK;
    sp.tiled = GENERATE(false, true);
    CAPTURE(sp.spp, sp.tiled);
    SECTION("uint16 LZW + predictor") {
        sp.bps = 16;
        sp.compression = COMPRESSION_LZW;
        sp.predictor = PREDICTOR_HORIZONTAL;
        checkAllReads<uint16_t>(sp);
    }
    SECTION("int32") {
        sp.bps = 32;
        sp.format = SAMPLEFORMAT_INT;
        checkAllReads<int32_t>(sp);
    }
    SECTION("float32") {
        // (big-endian + the floating-point predictor: libtiff 4.7 writes
        // those bytes in an order neither SIRIUS nor tifffile reads back;
        // tests/data/tifffile/be_float32_fppred.tif, written by tifffile,
        // covers that case below)
        sp.bps = 32;
        sp.format = SAMPLEFORMAT_IEEEFP;
        sp.compression = COMPRESSION_ADOBE_DEFLATE;
        checkAllReads<float>(sp);
    }
    SECTION("float16") {
        sp.bps = 16;
        sp.format = SAMPLEFORMAT_IEEEFP;
        checkAllReads<float>(sp);
    }
    SECTION("12-bit packed") {
        sp.bps = 12;
        checkAllReads<uint16_t>(sp);
    }
    SECTION("24-bit") {
        sp.bps = 24;
        checkAllReads<uint32_t>(sp);
    }
}

TEST_CASE("PackBits and every built-in lossless codec decode", "[tiff][codec]") {
    Spec sp;
    sp.bps = 16;
    sp.compression = GENERATE(as<uint16_t>{}, COMPRESSION_PACKBITS, COMPRESSION_LZW, COMPRESSION_DEFLATE,
                              COMPRESSION_ADOBE_DEFLATE, COMPRESSION_ZSTD);
    sp.tiled = GENERATE(false, true);
    sp.spp = GENERATE(as<uint16_t>{}, 1, 3);
    sp.photometric = sp.spp == 3 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK;
    CAPTURE(sp.compression, sp.tiled, sp.spp);
    TempFile f(".tif");
    write(f.path, sp);
    TiffFile file(f.path);
    CHECK(file.info().page(0).compression == sp.compression);
    requirePattern(file.readStack<uint16_t>(), sp, Region{0, 0, sp.width, sp.height}, 0, sp.spp);
}

TEST_CASE("Missing strips and tiles of a sparse file read as zeros", "[tiff][sparse]") {
    Spec sp;
    sp.sparse = true;
    sp.bps = 16;
    sp.tiled = GENERATE(false, true);
    sp.spp = GENERATE(as<uint16_t>{}, 1, 3);
    sp.planar = GENERATE(as<uint16_t>{}, PLANARCONFIG_CONTIG, PLANARCONFIG_SEPARATE);
    sp.photometric = sp.spp == 3 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK;
    CAPTURE(sp.tiled, sp.spp, sp.planar);
    checkAllReads<uint16_t>(sp);
}

TEST_CASE("Palette images read as indices with the colormap in the info", "[tiff][palette]") {
    Spec sp;
    sp.photometric = PHOTOMETRIC_PALETTE;
    sp.bps = GENERATE(as<uint16_t>{}, 4, 8);
    TempFile f(".tif");
    write(f.path, sp);
    TiffFile file(f.path);
    const TiffImageInfo& p = file.info().page(0);
    CHECK(p.photometric == PHOTOMETRIC_PALETTE);
    const std::size_t n = std::size_t{1} << sp.bps;
    REQUIRE(p.colormap.size() == 3 * n);
    CHECK(p.colormap[1] == 257);              // R[1]
    CHECK(p.colormap[n + 1] == 65535 - 257);  // G[1]
    CHECK(p.colormap[2 * n + 2] == 200);      // B[2]
    requirePattern(file.readStack<uint8_t>(), sp, Region{0, 0, sp.width, sp.height}, 0, 1);
}

TEST_CASE("A large single page is split across threads and decodes the same", "[tiff][threads]") {
    Spec sp;
    sp.width = 1500;
    sp.height = 1300;
    sp.pages = 1;
    sp.bps = 16;
    sp.tiled = GENERATE(false, true);
    sp.tile = 128;
    sp.rowsPerStrip = 16;
    sp.compression = COMPRESSION_ADOBE_DEFLATE;
    sp.spp = GENERATE(as<uint16_t>{}, 1, 3);
    sp.photometric = sp.spp == 3 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK;
    CAPTURE(sp.tiled, sp.spp);
    TempFile f(".tif");
    write(f.path, sp);
    TiffFile file(f.path);
    const Region all{0, 0, sp.width, sp.height};
    TiffReadOptions opts;
    std::vector<double> seen;
    opts.progress = [&](double v) { seen.push_back(v); };
    requirePattern(file.readStack<uint16_t>(opts), sp, all, 0, sp.spp);
    REQUIRE_FALSE(seen.empty());
    CHECK(seen.back() == Catch::Approx(1.0));
    requirePattern(file.readRegion<float>(Region{100, 200, 900, 700}), sp, Region{100, 200, 900, 700}, 0, sp.spp);
}

TEST_CASE("IFDs SIRIUS cannot decode are described, not refused, by inspection", "[tiff][unsupported]") {
    TempFile f(".tif");
    {
        TiffPtr tif(TIFFOpen(f.path.c_str(), "w"));
        REQUIRE(tif);
        TIFF* t = tif.get();
        TIFFSetField(t, TIFFTAG_IMAGEWIDTH, 4);
        TIFFSetField(t, TIFFTAG_IMAGELENGTH, 4);
        TIFFSetField(t, TIFFTAG_BITSPERSAMPLE, 64);
        TIFFSetField(t, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(t, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_INT);
        TIFFSetField(t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        std::vector<int64_t> row(4, -1);
        for (uint32_t r = 0; r < 4; ++r) TIFFWriteScanline(t, row.data(), r);
        TIFFWriteDirectory(t);
    }
    TiffFile file(f.path);
    const TiffImageInfo& p = file.info().page(0);
    CHECK_FALSE(p.decodable());
    CHECK_THAT(p.unsupported, Catch::Matchers::ContainsSubstring("64-bit signed"));
    CHECK_THROWS_WITH(file.readStack<double>(), Catch::Matchers::ContainsSubstring("64-bit signed"));
}

// -----------------------------------------------------------------------
// Metadata
// -----------------------------------------------------------------------

TEST_CASE("parseTiffMetadata reads OME images, channels, TiffData and physical sizes", "[tiff][ome]") {
    const std::string xml =
        R"(<?xml version="1.0" encoding="UTF-8"?>)"
        R"(<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" UUID="urn:uuid:aaa">)"
        R"(<!-- a comment with <Image> inside -->)"
        R"(<Image ID="Image:0" Name="first &amp; best"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16")"
        R"( SizeX="64" SizeY="32" SizeZ="3" SizeC="2" SizeT="1" PhysicalSizeX="65" PhysicalSizeXUnit="nm")"
        R"( PhysicalSizeY="0.065" PhysicalSizeZ="0.2" TimeIncrement="500" TimeIncrementUnit="ms">)"
        R"(<Channel ID="Channel:0:0" Name="DAPI" EmissionWavelength="461" Color="-16776961" SamplesPerPixel="1"/>)"
        R"(<Channel ID="Channel:0:1" Name="GFP" EmissionWavelength="0.51" EmissionWavelengthUnit="µm"/>)"
        R"(<TiffData IFD="0" PlaneCount="6"><UUID FileName="a.ome.tif">urn:uuid:aaa</UUID></TiffData>)"
        R"(</Pixels></Image>)"
        R"(<Image ID="Image:1" Name="rgb"><Pixels DimensionOrder="XYCZT" Type="uint8" SizeX="8" SizeY="8" SizeZ="2")"
        R"( SizeC="3" SizeT="1" Interleaved="true"><Channel ID="Channel:1:0" SamplesPerPixel="3"/>)"
        R"(<TiffData IFD="6" PlaneCount="1"/><TiffData IFD="7" FirstZ="1" PlaneCount="1"/></Pixels></Image>)"
        R"(<Image ID="Image:2"><Pixels DimensionOrder="XYCZT" SizeX="8" SizeY="8" SizeZ="1" SizeC="1" SizeT="1">)"
        R"(<TiffData IFD="0"><UUID FileName="other.ome.tif">urn:uuid:bbb</UUID></TiffData></Pixels></Image>)"
        R"(</OME>)";
    const TiffMetadata md = parseTiffMetadata(xml);
    REQUIRE(md.ome);
    CHECK_FALSE(md.imagej);
    CHECK(md.omeUuid == "urn:uuid:aaa");
    REQUIRE(md.omeImages.size() == 3);
    const OmeImage& a = md.omeImages[0];
    CHECK(a.name == "first & best");
    CHECK(a.dimensionOrder == "XYZCT");
    CHECK(a.type == "uint16");
    CHECK((a.sizeX == 64 && a.sizeY == 32 && a.sizeZ == 3 && a.sizeC == 2 && a.sizeT == 1));
    CHECK(a.physicalSizeUm[0] == Catch::Approx(0.065));
    CHECK(a.physicalSizeUm[1] == Catch::Approx(0.065));
    CHECK(a.physicalSizeUm[2] == Catch::Approx(0.2));
    CHECK(a.timeIncrementS == Catch::Approx(0.5));
    REQUIRE(a.channels.size() == 2);
    CHECK(a.channels[0].name == "DAPI");
    CHECK(a.channels[0].emissionNm == Catch::Approx(461));
    CHECK(a.channels[0].hasColor);
    CHECK(a.channels[0].colorRgba == 0xFF0000FFu);   // -16776961: opaque red
    CHECK(a.channels[1].emissionNm == Catch::Approx(510));
    REQUIRE(a.tiffData.size() == 1);
    CHECK(a.tiffData[0].uuid == "urn:uuid:aaa");
    CHECK(a.tiffData[0].fileName == "a.ome.tif");
    CHECK(md.omeImages[1].interleaved);
    CHECK(md.omeImages[1].channels.at(0).samplesPerPixel == 3);
    // the summary is the first image
    CHECK(md.sizeC == 2);
    CHECK(md.sizeZ == 3);
    CHECK(md.dimensionOrder == "XYZCT");
    REQUIRE(md.channels.size() == 2);
    CHECK(md.channels[0].color[0] == Catch::Approx(1.0f));
    CHECK(md.channels[0].color[2] == Catch::Approx(0.0f));
    CHECK_FALSE(md.channels[1].hasColor);
    CHECK(md.frameIntervalS == Catch::Approx(0.5));

    // pages of each image: the third lives in another file
    const auto pages = omeImagePages(md, 8, 3);
    REQUIRE(pages.size() == 3);
    CHECK(pages[0] == std::vector<std::uint32_t>{0, 1, 2, 3, 4, 5});
    CHECK(pages[1] == std::vector<std::uint32_t>{6, 7});
    CHECK(pages[2].empty());
}

TEST_CASE("omeImagePages follows TiffData plane order and images without TiffData follow on", "[tiff][ome]") {
    const std::string xml =
        R"(<OME><Image><Pixels DimensionOrder="XYCZT" SizeZ="2" SizeC="2" SizeT="1">)"
        R"(<TiffData IFD="3" FirstC="1" FirstZ="0" PlaneCount="1"/><TiffData IFD="2" FirstC="0" FirstZ="1"/>)"
        R"(<TiffData IFD="1" FirstC="1" FirstZ="1"/><TiffData IFD="0"/></Pixels></Image>)"
        R"(<Image><Pixels DimensionOrder="XYZCT" SizeZ="3" SizeC="1" SizeT="1"/></Image></OME>)";
    const TiffMetadata md = parseTiffMetadata(xml);
    const auto pages = omeImagePages(md, 7);
    REQUIRE(pages.size() == 2);
    // plane order XYCZT: (c0 z0) (c1 z0) (c0 z1) (c1 z1)
    CHECK(pages[0] == std::vector<std::uint32_t>{0, 3, 2, 1});
    CHECK(pages[1] == std::vector<std::uint32_t>{4, 5, 6});
    // fewer pages than planes: the list stops at the first missing one
    CHECK(omeImagePages(md, 5)[1] == std::vector<std::uint32_t>{4});
}

TEST_CASE("parseTiffMetadata reads ImageJ hyperstack headers", "[tiff][imagej]") {
    const TiffMetadata md = parseTiffMetadata(
        "ImageJ=1.54f\nimages=24\nchannels=2\nslices=4\nframes=3\nhyperstack=true\nmode=composite\n"
        "unit=micron\nspacing=0.25\nfinterval=1.5\nmin=10\nmax=4000\nloop=false\n");
    REQUIRE(md.imagej);
    CHECK_FALSE(md.ome);
    CHECK(md.imageJ.version == "1.54f");
    CHECK(md.imageJ.images == 24);
    CHECK(md.imageJ.channels == 2);
    CHECK(md.imageJ.slices == 4);
    CHECK(md.imageJ.frames == 3);
    CHECK(md.imageJ.hyperstack);
    CHECK(md.imageJ.mode == "composite");
    CHECK(md.imageJ.unitUm == Catch::Approx(1.0));
    CHECK(md.imageJ.spacing == Catch::Approx(0.25));
    CHECK(md.imageJ.hasRange);
    CHECK(md.imageJ.max == Catch::Approx(4000));
    CHECK(md.imageJ.entries.at("loop") == "false");
    CHECK(md.sizeC == 2);
    CHECK(md.sizeZ == 4);
    CHECK(md.sizeT == 3);
    CHECK(md.dimensionOrder == "XYCZT");
    CHECK(md.voxelUm[2] == Catch::Approx(0.25));
    CHECK(md.frameIntervalS == Catch::Approx(1.5));
    CHECK(parseTiffMetadata("ImageJ=1.5\nunit=nm\nspacing=200\n").voxelUm[2] == Catch::Approx(0.2));
    CHECK_FALSE(parseTiffMetadata("a plain description").ome);
    CHECK_FALSE(parseTiffMetadata("").imagej);
}

TEST_CASE("TiffFile::metadata and series read an OME-TIFF's images", "[tiff][ome]") {
    // two images: 3 planes of 20x10, then 2 planes of 8x6 (pages differ in size)
    const std::string xml =
        R"(<OME UUID="urn:uuid:x"><Image ID="Image:0"><Pixels DimensionOrder="XYZCT" Type="uint16" SizeX="20" SizeY="10" SizeZ="3" SizeC="1" SizeT="1">)"
        R"(<Channel ID="Channel:0:0" Name="c0"/><TiffData IFD="0" PlaneCount="3"/></Pixels></Image>)"
        R"(<Image ID="Image:1"><Pixels DimensionOrder="XYZCT" Type="uint16" SizeX="8" SizeY="6" SizeZ="2" SizeC="1" SizeT="1">)"
        R"(<TiffData IFD="3" PlaneCount="2"/></Pixels></Image></OME>)";
    TempFile f(".ome.tif");
    {
        TiffPtr tif(TIFFOpen(f.path.c_str(), "w"));
        REQUIRE(tif);
        for (int p = 0; p < 5; ++p) {
            const uint32_t w = p < 3 ? 20 : 8, h = p < 3 ? 10 : 6;
            TIFF* t = tif.get();
            TIFFSetField(t, TIFFTAG_IMAGEWIDTH, w);
            TIFFSetField(t, TIFFTAG_IMAGELENGTH, h);
            TIFFSetField(t, TIFFTAG_BITSPERSAMPLE, 16);
            TIFFSetField(t, TIFFTAG_SAMPLESPERPIXEL, 1);
            TIFFSetField(t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
            if (p == 0) TIFFSetField(t, TIFFTAG_IMAGEDESCRIPTION, xml.c_str());
            std::vector<uint16_t> row(w);
            for (uint32_t y = 0; y < h; ++y) {
                for (uint32_t x = 0; x < w; ++x) row[x] = static_cast<uint16_t>(p * 1000 + y * 10 + x);
                TIFFWriteScanline(t, row.data(), y);
            }
            TIFFWriteDirectory(t);
        }
    }
    TiffFile file(f.path);
    CHECK_FALSE(file.info().uniformPages());
    const TiffMetadata& md = file.metadata();
    REQUIRE(md.ome);
    CHECK(md.omeImages.size() == 2);
    const auto series = file.series();
    REQUIRE(series.size() == 2);
    CHECK(series[0] == std::vector<std::uint32_t>{0, 1, 2});
    CHECK(series[1] == std::vector<std::uint32_t>{3, 4});
    const Buffer<uint16_t> s1 = file.readSeries<uint16_t>(1);
    REQUIRE(s1.shape() == Shape{2, 6, 8});
    CHECK(s1.data()[0] == 3000);
    CHECK(s1.data()[6 * 8 + 5 * 8 + 7] == 4000 + 57);
    CHECK_THROWS_AS(file.readSeries<uint16_t>(2), std::out_of_range);
}

// -----------------------------------------------------------------------
// Files written by tifffile (tests/data/tifffile/make_fixtures.py)
// -----------------------------------------------------------------------

namespace {
    std::string fixture(const char* name) { return std::string(SIRIUS_TEST_DATA_DIR) + "/tifffile/" + name; }
} // namespace

TEST_CASE("tifffile fixtures: sample layouts, codecs and number formats", "[tiff][tifffile]") {
    Spec sp;   // 29 x 37, 2 pages: the generator's pattern
    SECTION("RGB, contiguous, LZW + horizontal predictor") {
        sp.spp = 3;
        TiffFile f(fixture("rgb_contig_lzw.tif"));
        CHECK(f.info().page(0).photometric == PHOTOMETRIC_RGB);
        CHECK(f.info().page(0).predictor == PREDICTOR_HORIZONTAL);
        requirePattern(f.readStack<uint8_t>(), sp, Region{0, 0, 37, 29}, 0, 3);
    }
    SECTION("RGB, separate planes, Deflate tiles, uint16") {
        sp.spp = 3;
        sp.bps = 16;
        TiffFile f(fixture("rgb_planar_tiled_deflate.tif"));
        CHECK(f.info().page(0).planarConfig == PLANARCONFIG_SEPARATE);
        CHECK(f.info().page(0).layout == TiffLayout::Tiles);
        requirePattern(f.readStack<uint16_t>(), sp, Region{0, 0, 37, 29}, 0, 3);
        TiffReadOptions green;
        green.firstSample = 1;
        green.sampleCount = 1;
        requirePattern(f.readRegion<uint16_t>(Region{5, 3, 20, 20}, 0, green), sp, Region{5, 3, 20, 20}, 1, 1);
    }
    SECTION("big-endian float32 with the floating-point predictor") {
        sp.format = SAMPLEFORMAT_IEEEFP;
        sp.bps = 32;
        TiffFile f(fixture("be_float32_fppred.tif"));
        CHECK(f.info().bigEndian);
        CHECK(f.info().page(0).predictor == PREDICTOR_FLOATINGPOINT);
        requirePattern(f.readStack<float>(), sp, Region{0, 0, 37, 29}, 0, 1);
    }
    SECTION("float16 tiles") {
        sp.format = SAMPLEFORMAT_IEEEFP;
        sp.bps = 16;
        TiffFile f(fixture("float16_tiled.tif"));
        CHECK(f.info().pixelType() == PixelType::Float32);
        requirePattern(f.readStack<float>(), sp, Region{0, 0, 37, 29}, 0, 1);
    }
    SECTION("int16 PackBits") {
        TiffFile f(fixture("int16_packbits.tif"));
        CHECK(f.info().page(0).compression == COMPRESSION_PACKBITS);
        const Buffer<int16_t> a = f.readStack<int16_t>();
        sp.bps = 16;
        for (int p = 0; p < 2; ++p)
            for (uint32_t y = 0; y < 29; ++y)
                for (uint32_t x = 0; x < 37; ++x)
                    REQUIRE(a.data()[(p * 29 + y) * 37 + x] == static_cast<int16_t>(intValue(sp, p, 0, y, x) - 300));
    }
    SECTION("bilevel") {
        sp.bps = 1;
        sp.pages = 1;
        TiffFile f(fixture("bilevel.tif"));
        CHECK(f.info().page(0).bitsPerSample == 1);
        requirePattern(f.readStack<uint8_t>(), sp, Region{0, 0, 37, 29}, 0, 1);
    }
}

namespace {
    // <name>.expected.raw: tifffile + imagecodecs' decode of a lossy fixture,
    // (pages, samples, height, width) little-endian samples of type T.
    template <typename T>
    std::vector<T> expectedPixels(const char* tif) {
        std::string name = fixture(tif);
        name.replace(name.size() - 4, 4, ".expected.raw");
        std::ifstream in(name, std::ios::binary);
        REQUIRE(in);
        const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        REQUIRE(bytes.size() % sizeof(T) == 0);
        std::vector<T> v(bytes.size() / sizeof(T));
        for (std::size_t i = 0; i < v.size(); ++i) {
            uint64_t le = 0;
            for (std::size_t b = 0; b < sizeof(T); ++b)
                le |= uint64_t{static_cast<unsigned char>(bytes[i * sizeof(T) + b])} << (8 * b);
            v[i] = static_cast<T>(le);
        }
        return v;
    }

    template <typename T>
    void requireSame(const Buffer<T>& got, const std::vector<T>& want) {
        REQUIRE(static_cast<std::size_t>(got.size()) == want.size());
        for (std::size_t i = 0; i < want.size(); ++i)
            if (got.data()[i] != want[i])
                FAIL("element " << i << ": got " << +got.data()[i] << " expected " << +want[i]);
    }
} // namespace

TEST_CASE("tifffile fixtures: ZSTD with predictors", "[tiff][tifffile][zstd]") {
    Spec sp;
    SECTION("uint16 tiles, horizontal predictor") {
        sp.bps = 16;
        TiffFile f(fixture("zstd_uint16_tiled_pred.tif"));
        CHECK(f.info().page(0).compression == COMPRESSION_ZSTD);
        CHECK(f.info().page(0).predictor == PREDICTOR_HORIZONTAL);
        requirePattern(f.readStack<uint16_t>(), sp, Region{0, 0, 37, 29}, 0, 1);
        requirePattern(f.readRegion<uint16_t>(Region{5, 3, 20, 20}), sp, Region{5, 3, 20, 20}, 0, 1);
    }
    SECTION("float32 strips, floating-point predictor") {
        sp.format = SAMPLEFORMAT_IEEEFP;
        sp.bps = 32;
        TiffFile f(fixture("zstd_float32_fppred.tif"));
        CHECK(f.info().page(0).predictor == PREDICTOR_FLOATINGPOINT);
        requirePattern(f.readStack<float>(), sp, Region{0, 0, 37, 29}, 0, 1);
    }
}

TEST_CASE("tifffile fixtures: JPEG decodes to what libjpeg-turbo decodes, YCbCr as RGB", "[tiff][tifffile][jpeg]") {
    SECTION("2x2-subsampled YCbCr tiles read as RGB channel planes") {
        TiffFile f(fixture("jpeg_ycbcr_tiled.tif"));
        const TiffImageInfo& p = f.info().page(0);
        CHECK(p.compression == COMPRESSION_JPEG);
        CHECK(p.photometric == PHOTOMETRIC_YCBCR);
        REQUIRE(p.decodable());
        const std::vector<uint8_t> want = expectedPixels<uint8_t>("jpeg_ycbcr_tiled.tif");
        const Buffer<uint8_t> all = f.readStack<uint8_t>();
        REQUIRE(all.shape() == Shape{2, 3, 29, 37});
        requireSame(all, want);
        TiffReadOptions serial;
        serial.maxThreads = 1;
        requireSame(f.readStack<uint8_t>(serial), want);
        // a region and one sample are slices of the same decode
        TiffReadOptions green;
        green.firstSample = 1;
        green.sampleCount = 1;
        const Region r{5, 3, 20, 21};
        const Buffer<uint8_t> part = f.readRegion<uint8_t>(r, 0, green);
        REQUIRE(part.shape() == Shape{2, 21, 20});
        for (std::size_t pg = 0; pg < 2; ++pg)
            for (uint32_t y = 0; y < r.height; ++y)
                for (uint32_t x = 0; x < r.width; ++x)
                    REQUIRE(part.data()[(pg * r.height + y) * r.width + x] ==
                            want[((pg * 3 + 1) * 29 + r.y + y) * 37 + r.x + x]);
    }
    SECTION("greyscale strips") {
        TiffFile f(fixture("jpeg_grey_strips.tif"));
        requireSame(f.readStack<uint8_t>(), expectedPixels<uint8_t>("jpeg_grey_strips.tif"));
    }
    SECTION("12-bit greyscale reads as uint16") {
        TiffFile f(fixture("jpeg12_grey.tif"));
        CHECK(f.info().page(0).bitsPerSample == 12);
        CHECK(f.info().pixelType() == PixelType::UInt16);
        requireSame(f.readStack<uint16_t>(), expectedPixels<uint16_t>("jpeg12_grey.tif"));
    }
}

TEST_CASE("tifffile fixtures: ImageJ hyperstack and OME-TIFF metadata, series and pyramids", "[tiff][tifffile]") {
    SECTION("ImageJ hyperstack") {
        TiffFile f(fixture("imagej_hyperstack.tif"));
        const TiffMetadata& md = f.metadata();
        REQUIRE(md.imagej);
        CHECK(md.sizeT == 3);
        CHECK(md.sizeZ == 2);
        CHECK(md.sizeC == 2);
        CHECK(md.imageJ.images == 12);
        CHECK(md.imageJ.hyperstack);
        CHECK(md.voxelUm[2] == Catch::Approx(0.5));
        CHECK(md.frameIntervalS == Catch::Approx(2.0));
        CHECK(f.info().page(0).xResolution == Catch::Approx(4.0));
        CHECK(f.info().pageCount() == 12);
        Spec sp;
        sp.bps = 16;
        sp.pages = 12;
        requirePattern(f.readStack<uint16_t>(), sp, Region{0, 0, 37, 29}, 0, 1);
    }
    SECTION("OME-TIFF RGB") {
        TiffFile f(fixture("ome_rgb.ome.tif"));
        const TiffMetadata& md = f.metadata();
        REQUIRE(md.ome);
        REQUIRE(md.omeImages.size() == 1);
        CHECK(md.sizeC == 3);
        CHECK(md.sizeZ == 2);
        CHECK(md.voxelUm[0] == Catch::Approx(0.1));
        CHECK(md.voxelUm[2] == Catch::Approx(0.3));
        REQUIRE(md.omeImages[0].channels.size() == 1);
        CHECK(md.omeImages[0].channels[0].samplesPerPixel == 3);
        CHECK(md.omeImages[0].channels[0].name == "RGB");
        CHECK(f.series() == std::vector<std::vector<std::uint32_t>>{{0, 1}});
        Spec sp;
        sp.spp = 3;
        requirePattern(f.readSeries<uint8_t>(0), sp, Region{0, 0, 37, 29}, 0, 3);
    }
    SECTION("OME-TIFF with two series, the first a SubIFD pyramid") {
        TiffFile f(fixture("ome_series_pyramid.ome.tif"));
        const TiffMetadata& md = f.metadata();
        REQUIRE(md.omeImages.size() == 2);
        CHECK(md.omeImages[0].name == "big");
        CHECK(md.omeImages[1].name == "small");
        const auto series = f.series();
        REQUIRE(series.size() == 2);
        CHECK(series[0] == std::vector<std::uint32_t>{0, 1, 2});
        CHECK(series[1] == std::vector<std::uint32_t>{3, 4});
        CHECK(f.seriesLevels(0) == 2);
        CHECK(f.seriesLevels(1) == 1);
        Spec big;
        big.bps = 16;
        big.pages = 3;
        big.width = 48;
        big.height = 64;
        requirePattern(f.readSeries<uint16_t>(0), big, Region{0, 0, 48, 64}, 0, 1);
        const Buffer<uint16_t> half = f.readSeries<uint16_t>(0, 1);
        REQUIRE(half.shape() == Shape{3, 32, 24});
        for (int p = 0; p < 3; ++p)
            for (uint32_t y = 0; y < 32; ++y)
                for (uint32_t x = 0; x < 24; ++x)
                    REQUIRE(half.data()[(p * 32 + y) * 24 + x] == static_cast<uint16_t>(intValue(big, p, 0, 2 * y, 2 * x)));
        const Buffer<uint16_t> small = f.readSeries<uint16_t>(1);
        REQUIRE(small.shape() == Shape{2, 29, 37});
        Spec sp;
        sp.bps = 16;
        CHECK(small.data()[0] == static_cast<uint16_t>(intValue(sp, 0, 0, 0, 0) + 7));
        CHECK_THROWS_AS(f.readSeries<uint16_t>(1, 1), std::out_of_range);
    }
}
