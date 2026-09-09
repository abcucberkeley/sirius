// TiffFile: metadata inspection, region/level reads, pyramids, and the GPU
// (nvTIFF) decode path with its libtiff fallback. GPU cases SKIP without a
// usable CUDA device.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdio>
#include <string>
#include <vector>

#include <tiffio.h>

#include "sirius/buffer.hpp"
#include "sirius/tiff_io.hpp"

#include "temp_path.hpp"

using namespace sirius;

namespace {

    struct TempFile {
        std::string path;
        explicit TempFile(const char* suffix) : path(test::uniqueTempPath("tifffile", suffix).string()) {}
        ~TempFile() { std::remove(path.c_str()); }
    };

    struct TiffDeleter {
        void operator()(TIFF* t) const { TIFFClose(t); }
    };
    using TiffPtr = std::unique_ptr<TIFF, TiffDeleter>;

    Device gpuOrSkip() {
        if (!cudaAvailable()) SKIP("no CUDA device available");
        if (!builtWithNvTiff()) SKIP("built without nvTIFF");
        return Device::cuda(0);
    }

    // Deterministic, non-trivial content: distinct per page, row and column.
    Image<uint16_t> pattern(Eigen::Index rows, Eigen::Index cols, int page = 0) {
        Image<uint16_t> img(rows, cols);
        for (Eigen::Index r = 0; r < rows; ++r)
            for (Eigen::Index c = 0; c < cols; ++c)
                img(r, c) = static_cast<uint16_t>((page * 7919 + r * 131 + c * 17) & 0xFFFF);
        return img;
    }

    Image<uint16_t> downsample2(const Image<uint16_t>& img) {
        Image<uint16_t> out((img.dimension(0) + 1) / 2, (img.dimension(1) + 1) / 2);
        for (Eigen::Index r = 0; r < out.dimension(0); ++r)
            for (Eigen::Index c = 0; c < out.dimension(1); ++c)
                out(r, c) = img(2 * r, 2 * c);
        return out;
    }

    struct PageSpec {
        const Image<uint16_t>* img;
        bool reduced;
        uint16_t nsub;      // number of SubIFDs to reserve (written right after this page)
    };

    void setCommonTags(TIFF* tif, const Image<uint16_t>& img, bool tiled, uint16_t compression, bool reduced) {
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, static_cast<uint32_t>(img.dimension(1)));
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, static_cast<uint32_t>(img.dimension(0)));
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 16);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tif, TIFFTAG_COMPRESSION, compression);
        if (compression != COMPRESSION_NONE) TIFFSetField(tif, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
        TIFFSetField(tif, TIFFTAG_SUBFILETYPE, reduced ? FILETYPE_REDUCEDIMAGE : 0);
        if (tiled) {
            TIFFSetField(tif, TIFFTAG_TILEWIDTH, 16);
            TIFFSetField(tif, TIFFTAG_TILELENGTH, 16);
        } else {
            TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, 8);
        }
    }

    void writePixels(TIFF* tif, const Image<uint16_t>& img, bool tiled) {
        const auto rows = static_cast<uint32_t>(img.dimension(0));
        const auto cols = static_cast<uint32_t>(img.dimension(1));
        if (tiled) {
            std::vector<uint16_t> tile(16 * 16);
            for (uint32_t r = 0; r < rows; r += 16)
                for (uint32_t c = 0; c < cols; c += 16) {
                    for (uint32_t tr = 0; tr < 16; ++tr)
                        for (uint32_t tc = 0; tc < 16; ++tc) {
                            const uint32_t ir = r + tr, ic = c + tc;
                            tile[tr * 16 + tc] = (ir < rows && ic < cols) ? img(ir, ic) : 0;
                        }
                    if (TIFFWriteTile(tif, tile.data(), c, r, 0, 0) < 0) throw std::runtime_error("TIFFWriteTile");
                }
        } else {
            std::vector<uint16_t> row(cols);
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t c = 0; c < cols; ++c) row[c] = img(r, c);
                if (TIFFWriteScanline(tif, row.data(), r) < 0) throw std::runtime_error("TIFFWriteScanline");
            }
        }
    }

    // Writes IFDs in order. A page with nsub > 0 reserves that many SubIFD
    // slots; libtiff links the next nsub directories written as its SubIFDs.
    void writeTiffPages(const std::string& path, const std::vector<PageSpec>& pages, bool tiled,
                        uint16_t compression = COMPRESSION_NONE, bool bigTiff = false) {
        TiffPtr tif(TIFFOpen(path.c_str(), bigTiff ? "w8" : "w"));
        if (!tif) throw std::runtime_error("cannot create " + path);
        for (const PageSpec& p : pages) {
            setCommonTags(tif.get(), *p.img, tiled, compression, p.reduced);
            if (p.nsub > 0) {
                std::vector<uint64_t> zeros(p.nsub, 0);
                TIFFSetField(tif.get(), TIFFTAG_SUBIFD, p.nsub, zeros.data());
            }
            writePixels(tif.get(), *p.img, tiled);
            if (!TIFFWriteDirectory(tif.get())) throw std::runtime_error("TIFFWriteDirectory");
        }
    }

    template <typename T>
    void requireEqual(const Buffer<T>& got, const Image<uint16_t>& expected, Index page = 0) {
        REQUIRE(got.device().isCpu());
        REQUIRE(got.rank() == 3);
        REQUIRE(got.dim(1) == expected.dimension(0));
        REQUIRE(got.dim(2) == expected.dimension(1));
        auto t = asTensor<const T, 3>(got.view());
        for (Eigen::Index r = 0; r < expected.dimension(0); ++r)
            for (Eigen::Index c = 0; c < expected.dimension(1); ++c) {
                if (t(page, r, c) != static_cast<T>(expected(r, c))) {
                    FAIL("mismatch at page " << page << " (" << r << "," << c << "): got " << +t(page, r, c)
                                             << " expected " << expected(r, c));
                }
            }
    }

    Image<uint16_t> crop(const Image<uint16_t>& img, Region r) {
        Image<uint16_t> out(r.height, r.width);
        for (uint32_t y = 0; y < r.height; ++y)
            for (uint32_t x = 0; x < r.width; ++x)
                out(y, x) = img(r.y + y, r.x + x);
        return out;
    }

    const int silenceTiff = [] {
        TIFFSetErrorHandler(nullptr);
        TIFFSetWarningHandler(nullptr);
        return 0;
    }();

} // namespace

// -----------------------------------------------------------------------
// Metadata
// -----------------------------------------------------------------------

TEST_CASE("inspectTiff reports pages layout and codec", "[tifffile][info]") {
    const bool tiled = GENERATE(false, true);
    const uint16_t comp = GENERATE(static_cast<uint16_t>(COMPRESSION_NONE), static_cast<uint16_t>(COMPRESSION_LZW),
                                   static_cast<uint16_t>(COMPRESSION_ADOBE_DEFLATE));
    INFO("tiled=" << tiled << " compression=" << comp);

    TempFile f(".tif");
    const auto p0 = pattern(37, 53, 0), p1 = pattern(37, 53, 1);
    writeTiffPages(f.path, {{&p0, false, 0}, {&p1, false, 0}}, tiled, comp);

    TiffInfo info = inspectTiff(f.path);
    REQUIRE_FALSE(info.bigTiff);
    REQUIRE(info.images.size() == 2);
    REQUIRE(info.pageCount() == 2);
    REQUIRE(info.levelCount() == 1);
    REQUIRE(info.width() == 53);
    REQUIRE(info.height() == 37);
    REQUIRE(info.pixelType() == PixelType::UInt16);
    REQUIRE(info.uniformPages());
    const auto& img = info.page(1);
    REQUIRE(img.compression == comp);
    REQUIRE(img.predictor == (comp == COMPRESSION_NONE ? 1 : 2));
    REQUIRE(img.layout == (tiled ? TiffLayout::Tiles : TiffLayout::Strips));
    if (tiled) {
        REQUIRE(img.tileWidth == 16);
        REQUIRE(img.tileHeight == 16);
    } else {
        REQUIRE(img.rowsPerStrip == 8);
    }
    REQUIRE_FALSE(img.reducedResolution);
    REQUIRE(img.subIfds.empty());
    REQUIRE(info.levels[0].ifds == info.pages);
    REQUIRE_THROWS_AS(info.image(12345), std::out_of_range);
}

TEST_CASE("inspectTiff on BigTIFF and sirius-written stacks", "[tifffile][info]") {
    TempFile f(".tif");
    ImageStack<float> stack(4, 8, 9);
    stack.setZero();
    writeTiffStack(f.path, stack);     // BigTIFF, float32
    TiffInfo info = inspectTiff(f.path);
    REQUIRE(info.bigTiff);
    REQUIRE(info.pageCount() == 4);
    REQUIRE(info.pixelType() == PixelType::Float32);
    REQUIRE(bytesPerPixel(info.pixelType()) == 4);
    REQUIRE(std::string(toString(info.pixelType())) == "float32");
}

TEST_CASE("Region::resolve", "[tifffile][region]") {
    REQUIRE(Region{}.full());
    Region r = Region{}.resolve(100, 50);
    REQUIRE(r.width == 100);
    REQUIRE(r.height == 50);
    Region q = Region{10, 20, 0, 0}.resolve(100, 50);
    REQUIRE(q.width == 90);
    REQUIRE(q.height == 30);
    REQUIRE_THROWS_AS((Region{100, 0, 1, 1}.resolve(100, 50)), std::out_of_range);
    REQUIRE_THROWS_AS((Region{90, 0, 11, 1}.resolve(100, 50)), std::out_of_range);
}

// -----------------------------------------------------------------------
// CPU reads: stack, pages, regions
// -----------------------------------------------------------------------

TEST_CASE("TiffFile reads stacks page ranges and regions on the CPU", "[tifffile][cpu]") {
    const bool tiled = GENERATE(false, true);
    const uint16_t comp = GENERATE(static_cast<uint16_t>(COMPRESSION_NONE), static_cast<uint16_t>(COMPRESSION_LZW));
    INFO("tiled=" << tiled << " compression=" << comp);

    TempFile f(".tif");
    const auto p0 = pattern(45, 70, 0), p1 = pattern(45, 70, 1), p2 = pattern(45, 70, 2);
    writeTiffPages(f.path, {{&p0, false, 0}, {&p1, false, 0}, {&p2, false, 0}}, tiled, comp);

    TiffFile file(f.path);
    REQUIRE(file.path() == f.path);
    REQUIRE(file.info().pageCount() == 3);

    SECTION("readStack") {
        auto stack = file.readStack<uint16_t>();
        REQUIRE(stack.shape() == Shape{3, 45, 70});
        requireEqual(stack, p0, 0);
        requireEqual(stack, p1, 1);
        requireEqual(stack, p2, 2);
    }
    SECTION("readStack with conversion to float") {
        auto stack = file.readStack<float>();
        REQUIRE(stack.shape() == Shape{3, 45, 70});
        requireEqual(stack, p2, 2);
    }
    SECTION("readPages") {
        auto pages = file.readPages<uint16_t>(1, 2);
        REQUIRE(pages.shape() == Shape{2, 45, 70});
        requireEqual(pages, p1, 0);
        requireEqual(pages, p2, 1);
        REQUIRE_THROWS_AS(file.readPages<uint16_t>(2, 2), std::out_of_range);
        REQUIRE_THROWS_AS(file.readPages<uint16_t>(0, 0), std::out_of_range);
    }
    SECTION("readRegion crossing strip and tile boundaries") {
        // Not aligned to the 8-row strips or 16x16 tiles, touches the right/bottom edge cases.
        const Region regions[] = {{5, 3, 30, 20}, {0, 0, 70, 45}, {64, 40, 6, 5}, {17, 9, 1, 1}, {10, 0, 0, 0}};
        for (const Region& reg : regions) {
            INFO("region " << reg.x << "," << reg.y << " " << reg.width << "x" << reg.height);
            const Region r = reg.resolve(70, 45);
            auto out = file.readRegion<uint16_t>(reg);
            REQUIRE(out.shape() == Shape{3, r.height, r.width});
            requireEqual(out, crop(p0, r), 0);
            requireEqual(out, crop(p2, r), 2);
        }
        REQUIRE_THROWS_AS(file.readRegion<uint16_t>(Region{70, 0, 1, 1}), std::out_of_range);
    }
    SECTION("readRegion with conversion") {
        auto out = file.readRegion<double>(Region{3, 5, 20, 11});
        requireEqual(out, crop(p1, Region{3, 5, 20, 11}), 1);
    }
    SECTION("decode into a caller-provided buffer") {
        Buffer<uint16_t> dst(Shape{1, 45, 70});
        file.decode<uint16_t>({file.info().pages[2]}, Region{}, dst.view());
        requireEqual(dst, p2, 0);
        Buffer<uint16_t> wrong(Shape{1, 44, 70});
        REQUIRE_THROWS_AS(file.decode<uint16_t>({file.info().pages[2]}, Region{}, wrong.view()), std::invalid_argument);
        REQUIRE_THROWS_AS(file.decode<uint16_t>({}, Region{}, dst.view()), std::invalid_argument);
    }
    SECTION("pinned host result") {
        TiffReadOptions opts;
        opts.hostMemory = HostMemory::Pinned;
        if (!builtWithCuda()) {
            REQUIRE_THROWS(file.readStack<uint16_t>(opts));
        } else if (cudaAvailable()) {
            auto stack = file.readStack<uint16_t>(opts);
            REQUIRE(stack.pinned());
            requireEqual(stack, p1, 1);
        }
    }
}

TEST_CASE("readTiffAny returns the on-disk pixel type", "[tifffile][cpu]") {
    TempFile f(".tif");
    ImageStack<int32_t> stack(2, 6, 7);
    for (Eigen::Index i = 0; i < stack.size(); ++i) stack.data()[i] = static_cast<int32_t>(i - 20);
    writeTiffStack(f.path, stack);
    AnyBuffer any = readTiffAny(f.path);
    REQUIRE(std::holds_alternative<Buffer<int32_t>>(any));
    const auto& b = std::get<Buffer<int32_t>>(any);
    REQUIRE(b.shape() == Shape{2, 6, 7});
    REQUIRE(b.data()[0] == -20);
    REQUIRE(b.data()[83] == 63);
}

TEST_CASE("Non-uniform pages are rejected by readStack but readable individually", "[tifffile][cpu]") {
    TempFile f(".tif");
    const auto big = pattern(20, 30, 0), small = pattern(10, 15, 1);
    writeTiffPages(f.path, {{&big, false, 0}, {&small, false, 0}}, false);
    TiffFile file(f.path);
    REQUIRE_FALSE(file.info().uniformPages());
    REQUIRE_THROWS_AS(file.readStack<uint16_t>(), std::runtime_error);
    REQUIRE_THROWS_AS(readTiffStack<uint16_t>(f.path), std::runtime_error);
    auto second = file.readPages<uint16_t>(1, 1);
    requireEqual(second, small, 0);
}

// -----------------------------------------------------------------------
// Pyramids
// -----------------------------------------------------------------------

TEST_CASE("SubIFD pyramids expose levels", "[tifffile][pyramid]") {
    const bool tiled = GENERATE(false, true);
    INFO("tiled=" << tiled);
    TempFile f(".tif");
    const auto l0 = pattern(64, 96, 0);
    const auto l1 = downsample2(l0);
    const auto l2 = downsample2(l1);
    writeTiffPages(f.path, {{&l0, false, 2}, {&l1, true, 0}, {&l2, true, 0}}, tiled);

    TiffFile file(f.path);
    const TiffInfo& info = file.info();
    REQUIRE(info.pageCount() == 1);
    REQUIRE(info.images.size() == 3);
    REQUIRE(info.page(0).subIfds.size() == 2);
    REQUIRE(info.levelCount() == 3);
    REQUIRE(info.levels[1].width == 48);
    REQUIRE(info.levels[1].height == 32);
    REQUIRE(info.levels[2].width == 24);
    REQUIRE(info.levels[2].height == 16);
    REQUIRE(info.image(info.levels[1].ifds[0]).reducedResolution);

    requireEqual(file.readLevel<uint16_t>(0), l0);
    requireEqual(file.readLevel<uint16_t>(1), l1);
    requireEqual(file.readLevel<uint16_t>(2), l2);
    REQUIRE_THROWS_AS(file.readLevel<uint16_t>(3), std::out_of_range);

    const Region r{7, 5, 20, 9};
    requireEqual(file.readRegion<uint16_t>(r, 1), crop(l1, r));
    const Region r2{3, 2, 12, 9};   // level 2 is 24x16
    requireEqual(file.readRegion<float>(r2, 2), crop(l2, r2));
}

TEST_CASE("Multi-page SubIFD pyramids group levels across pages", "[tifffile][pyramid]") {
    TempFile f(".tif");
    const auto a0 = pattern(32, 40, 0), b0 = pattern(32, 40, 1);
    const auto a1 = downsample2(a0), b1 = downsample2(b0);
    writeTiffPages(f.path, {{&a0, false, 1}, {&a1, true, 0}, {&b0, false, 1}, {&b1, true, 0}}, true);

    TiffFile file(f.path);
    REQUIRE(file.info().pageCount() == 2);
    REQUIRE(file.info().levelCount() == 2);
    REQUIRE(file.info().levels[1].ifds.size() == 2);

    auto full = file.readStack<uint16_t>();
    requireEqual(full, a0, 0);
    requireEqual(full, b0, 1);
    auto half = file.readLevel<uint16_t>(1);
    REQUIRE(half.shape() == Shape{2, 16, 20});
    requireEqual(half, a1, 0);
    requireEqual(half, b1, 1);
}

TEST_CASE("Flat pyramids (reduced IFDs on the main chain) expose levels", "[tifffile][pyramid]") {
    TempFile f(".tif");
    const auto l0 = pattern(64, 96, 0);
    const auto l1 = downsample2(l0);
    const auto l2 = downsample2(l1);
    writeTiffPages(f.path, {{&l0, false, 0}, {&l1, true, 0}, {&l2, true, 0}}, true);

    TiffFile file(f.path);
    const TiffInfo& info = file.info();
    REQUIRE(info.pageCount() == 1);          // reduced images are not pages
    REQUIRE(info.levelCount() == 3);
    REQUIRE(info.levels[2].width == 24);
    requireEqual(file.readStack<uint16_t>(), l0);
    requireEqual(file.readLevel<uint16_t>(1), l1);
    requireEqual(file.readLevel<uint16_t>(2), l2);
    // The Eigen API sees only the full-resolution page.
    auto eigenStack = readTiffStack<uint16_t>(f.path);
    REQUIRE(eigenStack.dimension(0) == 1);
}

// -----------------------------------------------------------------------
// GPU (nvTIFF) reads
// -----------------------------------------------------------------------

TEST_CASE("TiffFile decodes on the GPU and matches the CPU", "[tifffile][cuda]") {
    const Device gpu = gpuOrSkip();
    const bool tiled = GENERATE(false, true);
    const uint16_t comp = GENERATE(static_cast<uint16_t>(COMPRESSION_NONE), static_cast<uint16_t>(COMPRESSION_LZW),
                                   static_cast<uint16_t>(COMPRESSION_ADOBE_DEFLATE));
    const bool bigTiff = GENERATE(false, true);
    INFO("tiled=" << tiled << " compression=" << comp << " bigtiff=" << bigTiff);

    TempFile f(".tif");
    const auto p0 = pattern(45, 70, 0), p1 = pattern(45, 70, 1), p2 = pattern(45, 70, 2);
    writeTiffPages(f.path, {{&p0, false, 0}, {&p1, false, 0}, {&p2, false, 0}}, tiled, comp, bigTiff);

    TiffFile file(f.path);
    std::string reason;
    const bool gpuOk = file.gpuDecodable(gpu, &reason);
    INFO("gpuDecodable: " << gpuOk << " " << reason);
    if (comp == COMPRESSION_ADOBE_DEFLATE && !gpuOk)
        WARN("Deflate not decodable on the GPU (" << reason << "); exercising the fallback instead");
    else
        REQUIRE(gpuOk);

    TiffReadOptions opts;
    opts.device = gpu;
    Stream stream(gpu);

    SECTION("full stack, native type") {
        auto dev = file.readStack<uint16_t>(opts, stream);
        REQUIRE(dev.device() == gpu);
        REQUIRE(dev.shape() == Shape{3, 45, 70});
        auto host = dev.to(Device::cpu(), stream);
        stream.synchronize();
        requireEqual(host, p0, 0);
        requireEqual(host, p1, 1);
        requireEqual(host, p2, 2);
    }
    SECTION("full stack with on-device conversion to float") {
        auto dev = file.readStack<float>(opts, stream);
        auto host = dev.to(Device::cpu(), stream);
        stream.synchronize();
        requireEqual(host, p1, 1);
    }
    SECTION("region decode on the device") {
        const Region r{5, 3, 30, 20};
        auto dev = file.readRegion<uint16_t>(r, 0, opts, stream);
        auto host = dev.to(Device::cpu(), stream);
        stream.synchronize();
        requireEqual(host, crop(p2, r), 2);
    }
    SECTION("strict mode refuses the CPU fallback only when the GPU cannot decode") {
        opts.allowCpuFallback = false;
        if (gpuOk) {
            REQUIRE_NOTHROW(file.readPages<uint16_t>(0, 1, opts, stream));
        } else {
            REQUIRE_THROWS_WITH(file.readPages<uint16_t>(0, 1, opts, stream),
                                Catch::Matchers::ContainsSubstring("allowCpuFallback"));
        }
    }
}

TEST_CASE("GPU pyramid level reads", "[tifffile][cuda][pyramid]") {
    const Device gpu = gpuOrSkip();
    TempFile f(".tif");
    const auto l0 = pattern(64, 96, 0);
    const auto l1 = downsample2(l0);
    writeTiffPages(f.path, {{&l0, false, 1}, {&l1, true, 0}}, true, COMPRESSION_LZW);

    TiffFile file(f.path);
    TiffReadOptions opts;
    opts.device = gpu;
    auto dev = file.readLevel<uint16_t>(1, opts);
    REQUIRE(dev.shape() == Shape{1, 32, 48});
    auto host = dev.to(Device::cpu());
    Stream::null().synchronize();
    requireEqual(host, l1);
}

TEST_CASE("Files nvTIFF cannot decode fall back to libtiff and still land on the GPU", "[tifffile][cuda]") {
    const Device gpu = gpuOrSkip();
    // sirius writes LZW float with the floating-point predictor, which nvTIFF
    // does not support: the read must succeed through the libtiff fallback.
    TempFile f(".tif");
    ImageStack<float> stack(3, 40, 50);
    for (Eigen::Index i = 0; i < stack.size(); ++i) stack.data()[i] = static_cast<float>(i) * 0.5f;
    writeTiffStack(f.path, stack, TiffCompression::Lzw);

    TiffFile file(f.path);
    REQUIRE(file.info().page(0).predictor == 3);
    std::string reason;
    REQUIRE_FALSE(file.gpuDecodable(gpu, &reason));
    REQUIRE_THAT(reason, Catch::Matchers::ContainsSubstring("NVTIFF_STATUS"));

    TiffReadOptions opts;
    opts.device = gpu;
    auto dev = file.readStack<float>(opts);
    REQUIRE(dev.device() == gpu);
    auto back = toEigen<3>(dev);
    for (Eigen::Index i = 0; i < stack.size(); ++i) REQUIRE(back.data()[i] == stack.data()[i]);

    opts.allowCpuFallback = false;
    REQUIRE_THROWS_AS(file.readStack<float>(opts), std::runtime_error);
}

TEST_CASE("Repository test data decodes identically on CPU and GPU", "[tifffile][cuda][data]") {
    const Device gpu = gpuOrSkip();
    const std::string path = std::string(SIRIUS_TEST_DATA_DIR) + "/raw.tif";
    TiffFile file(path);
    REQUIRE(file.info().pageCount() == 135);
    auto cpu = file.readStack<float>();
    TiffReadOptions opts;
    opts.device = gpu;
    auto dev = file.readStack<float>(opts);
    auto back = dev.to(Device::cpu());
    Stream::null().synchronize();
    REQUIRE(back.shape() == cpu.shape());
    for (Index i = 0; i < cpu.size(); ++i) REQUIRE(back.data()[i] == cpu.data()[i]);
}

TEST_CASE("Writers accept device buffers", "[tifffile][cuda]") {
    const Device gpu = gpuOrSkip();
    TempFile f(".tif");
    auto host = pattern(12, 13, 3);
    Buffer<uint16_t> dev = toDevice(host, gpu);
    writeTiff(f.path, dev);
    auto back = readTiff<uint16_t>(f.path);
    REQUIRE(asMatrix(back) == asMatrix(host));
}

TEST_CASE("TiffInfo::image is indexed and still works on a hand-built info", "[tifffile][info]") {
    TempFile f(".tif");
    std::vector<Image<uint16_t>> imgs;
    for (int i = 0; i < 40; ++i) imgs.push_back(pattern(9, 11, i));
    std::vector<PageSpec> specs;
    for (const auto& img : imgs) specs.push_back({&img, false, 0});
    writeTiffPages(f.path, specs, false);

    TiffInfo info = inspectTiff(f.path);
    REQUIRE(info.imageIndex.size() == info.images.size());
    for (std::size_t i = 0; i < info.images.size(); ++i) {
        REQUIRE(info.imageIndex.at(info.images[i].ifdOffset) == i);
        REQUIRE(&info.image(info.images[i].ifdOffset) == &info.images[i]);
    }
    REQUIRE(info.uniformPages());

    // An info assembled without the index (or with a stale one) falls back
    // to the linear search and reports unknown offsets as before.
    info.imageIndex.clear();
    REQUIRE(&info.image(info.pages[17]) == &info.images[17]);
    info.imageIndex.emplace(info.pages[3], 999);
    REQUIRE(&info.image(info.pages[3]) == &info.images[3]);
    REQUIRE_THROWS_AS(info.image(1), std::out_of_range);
}

TEST_CASE("readPages refuses a page count that would wrap the range", "[tifffile][cpu]") {
    TempFile f(".tif");
    const auto p0 = pattern(8, 8, 0), p1 = pattern(8, 8, 1);
    writeTiffPages(f.path, {{&p0, false, 0}, {&p1, false, 0}}, false, COMPRESSION_NONE);
    TiffFile file(f.path);
    // first + count wrapped around SIZE_MAX and passed the old bounds check
    CHECK_THROWS_AS(file.readPages<uint16_t>(1, std::numeric_limits<std::size_t>::max()), std::out_of_range);
    CHECK_THROWS_AS(file.readPages<uint16_t>(3, 1), std::out_of_range);
    CHECK_THROWS_AS(file.readPages<uint16_t>(0, 0), std::out_of_range);
    CHECK(file.readPages<uint16_t>(1, 1).shape() == Shape{1, 8, 8});
}
