// The workbench's dataset I/O: page-order mapping, OME / ImageJ metadata
// parsing, lazy TIFF sources, zarr sources (when TensorStore is built in)
// and the export writer in every format, scaling and range.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/array.hpp"
#include "core/array_source.hpp"
#include "core/dataset.hpp"
#include "core/export.hpp"
#include "core/labels.hpp"

#include "sirius/tiff_io.hpp"
#include "sirius/zarr_io.hpp"

#include "temp_path.hpp"

using namespace sirius;
using namespace sirius::app;
namespace fs = std::filesystem;
using Catch::Matchers::WithinRel;

namespace {

    struct TempPath {
        fs::path path;
        explicit TempPath(const char* suffix) : path(test::uniqueTempPath("appio", suffix)) {}
        ~TempPath() {
            std::error_code ec;
            fs::remove_all(path, ec);
            fs::remove(fs::path(path.string() + ".json"), ec);
            fs::remove(fs::path(path.string() + ".pipeline.toml"), ec);
        }
        std::string str() const { return path.string(); }
    };

    // Deterministic (c, t, z, y, x) array with a distinct value per voxel.
    Array5 testArray(Dims5 d) {
        Array5 a(d);
        for (Index c = 0; c < d.c; ++c)
            for (Index t = 0; t < d.t; ++t)
                for (Index z = 0; z < d.z; ++z)
                    for (Index y = 0; y < d.y; ++y)
                        for (Index x = 0; x < d.x; ++x)
                            a.at(c, t, z, y, x) = static_cast<float>(((c * 7 + t * 5 + z * 3) * 100 + y * 10 + x) % 997);
        return a;
    }

    DatasetMeta testMeta(const Dims5& d) {
        DatasetMeta m;
        m.name = "sample";
        m.dims = d;
        m.voxelUm = {0.1, 0.1, 0.3};
        m.frameIntervalS = 1.5;
        m.channels = {{"DAPI", 405.0, {1.f, 1.f, 1.f}, {}}, {"GFP", 488.0, {1.f, 1.f, 1.f}, {}}};
        m.channels.resize(static_cast<std::size_t>(d.c));
        m.normalizeChannels();
        return m;
    }

    // Write pages of `a` in page order `order` as a plain TIFF (float32).
    void writePlainTiff(const std::string& path, const Array5& a, const std::string& order, const std::string& description = {},
                        double pixelUm = 0.0) {
        const Dims5& d = a.dims();
        PageOrder po = PageOrder::fromDims(d, order);
        Buffer<float> pages(Shape{d.planes(), d.y, d.x});
        for (Index c = 0; c < d.c; ++c)
            for (Index t = 0; t < d.t; ++t)
                for (Index z = 0; z < d.z; ++z)
                    std::copy_n(a.plane(c, t, z), d.planeSize(), pages.data() + po.planeOf(c, t, z) * d.planeSize());
        TiffWriteOptions o;
        o.description = description;
        o.xPixelUm = o.yPixelUm = pixelUm;
        writeTiffStack<float>(path, pages.view(), o);
    }

    void requireSame(const Array5& a, const Array5& b) {
        REQUIRE(a.dims() == b.dims());
        for (Index i = 0; i < a.numel(); ++i)
            if (a.data()[i] != b.data()[i]) FAIL("mismatch at " << i << ": " << a.data()[i] << " vs " << b.data()[i]);
    }

} // namespace

// --- page order ------------------------------------------------------------------

TEST_CASE("PageOrder maps (c, t, z) to pages with the fastest axis first", "[app][io][pageorder]") {
    PageOrder czt = PageOrder::fromDims(Dims5{2, 3, 4, 1, 1}, "czt");
    CHECK(czt.planeOf(0, 0, 0) == 0);
    CHECK(czt.planeOf(1, 0, 0) == 1);
    CHECK(czt.planeOf(0, 0, 1) == 2);
    CHECK(czt.planeOf(0, 1, 0) == 8);
    CHECK(czt.planeOf(1, 2, 3) == 1 + 3 * 2 + 2 * 8);
    PageOrder ztc = PageOrder::fromDims(Dims5{2, 3, 4, 1, 1}, "ztc");
    CHECK(ztc.planeOf(0, 0, 1) == 1);
    CHECK(ztc.planeOf(0, 1, 0) == 4);
    CHECK(ztc.planeOf(1, 0, 0) == 12);
}

// --- description parsing -----------------------------------------------------------

TEST_CASE("parseTiffDescription reads OME-XML dimensions, physical sizes and channels", "[app][io][ome]") {
    const std::string xml =
        "<?xml version=\"1.0\"?><OME xmlns=\"http://www.openmicroscopy.org/Schemas/OME/2016-06\">"
        "<Image ID=\"Image:0\" Name=\"x\"><Pixels ID=\"Pixels:0\" DimensionOrder=\"XYZCT\" Type=\"uint16\" "
        "SizeX=\"32\" SizeY=\"16\" SizeZ=\"4\" SizeC=\"2\" SizeT=\"3\" PhysicalSizeX=\"32\" PhysicalSizeXUnit=\"nm\" "
        "PhysicalSizeY=\"0.032\" PhysicalSizeYUnit=\"&#181;m\" PhysicalSizeZ=\"0.11\" TimeIncrement=\"1500\" TimeIncrementUnit=\"ms\">"
        "<Channel ID=\"Channel:0:0\" Name=\"&#945;-actinin\" EmissionWavelength=\"488\" Color=\"1677721599\"/>"
        "<Channel ID=\"Channel:0:1\" Name=\"Mito\" EmissionWavelength=\"0.64\" EmissionWavelengthUnit=\"µm\"/>"
        "<TiffData/></Pixels></Image></OME>";
    const ParsedTiffMetadata m = parseTiffDescription(xml);
    REQUIRE(m.ome);
    CHECK_FALSE(m.imagej);
    CHECK(m.c == 2);
    CHECK(m.t == 3);
    CHECK(m.z == 4);
    CHECK(m.dimensionOrder == "XYZCT");
    CHECK_THAT(m.voxelUm[0], WithinRel(0.032, 1e-9));
    CHECK_THAT(m.voxelUm[1], WithinRel(0.032, 1e-9));
    CHECK_THAT(m.voxelUm[2], WithinRel(0.11, 1e-9));
    CHECK_THAT(m.frameIntervalS, WithinRel(1.5, 1e-9));
    REQUIRE(m.channels.size() == 2);
    CHECK(m.channels[0].label == "α-actinin");
    CHECK_THAT(m.channels[0].wavelengthNm, WithinRel(488.0, 1e-9));
    CHECK_THAT(m.channels[1].wavelengthNm, WithinRel(640.0, 1e-9));
    // Color 1677721599 = 0x63FFFFFF -> r 0x63, g 0xff, b 0xff
    CHECK_THAT(m.channels[0].color[0], WithinRel(0x63 / 255.0f, 1e-5f));
    CHECK(m.channels[0].color[1] == 1.f);
}

TEST_CASE("parseTiffDescription reads ImageJ hyperstack metadata", "[app][io][imagej]") {
    const ParsedTiffMetadata m = parseTiffDescription(
        "ImageJ=1.54f\nimages=24\nchannels=2\nslices=4\nframes=3\nhyperstack=true\nunit=micron\nspacing=0.25\nfinterval=2\nloop=false\n");
    REQUIRE(m.imagej);
    CHECK(m.c == 2);
    CHECK(m.z == 4);
    CHECK(m.t == 3);
    CHECK(m.dimensionOrder == "XYCZT");
    CHECK_THAT(m.voxelUm[2], WithinRel(0.25, 1e-9));
    CHECK_THAT(m.frameIntervalS, WithinRel(2.0, 1e-9));
    CHECK(parseTiffDescription("just a comment").ome == false);
}

// --- TIFF sources ---------------------------------------------------------------------

TEST_CASE("openDataset reads a plain TIFF as z pages and honours an explicit page order", "[app][io][tiff]") {
    TempPath f(".tif");
    const Dims5 d{2, 3, 4, 8, 12};
    const Array5 a = testArray(d);
    writePlainTiff(f.str(), a, "czt");

    SECTION("without metadata every page is a z plane") {
        const OpenResult r = openDataset(f.str());
        REQUIRE(r.meta.dims == Dims5{1, 1, 24, 8, 12});
        REQUIRE(r.meta.format == "tiff");
        REQUIRE(r.meta.sourceType == PixelType::Float32);
        REQUIRE_FALSE(r.dimsFromMetadata);
        REQUIRE(r.meta.bytesOnDisk > 0);
        REQUIRE(r.metadataSummary.find("24 pages") != std::string::npos);
        // page 1 is (c1, t0, z0) in czt order
        std::vector<float> plane(static_cast<std::size_t>(d.planeSize()));
        r.source->readPlane(0, 0, 1, plane.data());
        REQUIRE(plane[0] == a.at(1, 0, 0, 0, 0));
    }
    SECTION("an explicit page order recovers the hyperstack") {
        OpenOptions o;
        o.pageOrder = PageOrder::fromDims(d, "czt");
        o.voxelUm = std::array<double, 3>{0.05, 0.05, 0.2};
        const OpenResult r = openDataset(f.str(), o);
        REQUIRE(r.meta.dims == d);
        REQUIRE(r.meta.channels.size() == 2);
        REQUIRE_THAT(r.meta.voxelUm[2], WithinRel(0.2, 1e-9));
        std::vector<float> plane(static_cast<std::size_t>(d.planeSize()));
        r.source->readPlane(1, 2, 3, plane.data());
        REQUIRE(plane[5] == a.at(1, 2, 3, 0, 5));
        // the same plane again comes from the cache
        r.source->readPlane(1, 2, 3, plane.data());
        REQUIRE(plane[5] == a.at(1, 2, 3, 0, 5));
        Buffer<float> vol(Shape{d.z, d.y, d.x});
        r.source->readVolume(1, 1, vol.data());
        REQUIRE(vol.data()[2 * d.planeSize() + 7] == a.at(1, 1, 2, 0, 7));
        double last = 0.0;
        auto all = r.source->readAll([&](double p, const std::string&) { last = p; });
        REQUIRE(last == 1.0);
        requireSame(*all, a);
        REQUIRE_THROWS_AS(r.source->readPlane(2, 0, 0, plane.data()), std::out_of_range);
    }
    SECTION("full load wraps the data in a memory source") {
        OpenOptions o;
        o.pageOrder = PageOrder::fromDims(d, "czt");
        o.readAll = true;
        const OpenResult r = openDataset(f.str(), o);
        REQUIRE(r.source->inMemory());
        requireSame(*r.source->readAll(), a);
        REQUIRE(r.meta.dims == d);
    }
}

TEST_CASE("openDataset uses ImageJ and OME metadata for dimensions and voxel size", "[app][io][tiff]") {
    const Dims5 d{2, 2, 3, 6, 10};
    const Array5 a = testArray(d);
    SECTION("ImageJ") {
        TempPath f(".tif");
        writePlainTiff(f.str(), a, "czt",
                       "ImageJ=1.54f\nimages=12\nchannels=2\nslices=3\nframes=2\nhyperstack=true\nunit=micron\nspacing=0.4\nfinterval=1\n",
                       0.08);
        const OpenResult r = openDataset(f.str());
        REQUIRE(r.dimsFromMetadata);
        REQUIRE(r.meta.dims == d);
        REQUIRE_THAT(r.meta.voxelUm[0], WithinRel(0.08, 1e-4));
        REQUIRE_THAT(r.meta.voxelUm[2], WithinRel(0.4, 1e-9));
        REQUIRE(r.meta.acquisition == "ImageJ hyperstack");
        requireSame(*r.source->readAll(), a);
    }
    SECTION("OME-TIFF with z-fastest order and channel names") {
        TempPath f(".ome.tif");
        DatasetMeta m = testMeta(d);
        writePlainTiff(f.str(), a, "ztc", omeXml(m, d, PixelType::Float32, "x.ome.tif"), 0.1);
        const DatasetMeta probe = probeDataset(f.str());
        REQUIRE(probe.format == "ome-tiff");
        REQUIRE(probe.dims == d);
        const OpenResult r = openDataset(f.str());
        REQUIRE(r.dimsFromMetadata);
        REQUIRE(r.meta.dims == d);
        REQUIRE(r.meta.channels[0].label == "DAPI");
        REQUIRE(r.meta.channels[1].label == "GFP");
        REQUIRE_THAT(r.meta.channels[1].wavelengthNm, WithinRel(488.0, 1e-9));
        REQUIRE_THAT(r.meta.voxelUm[2], WithinRel(0.3, 1e-9));
        REQUIRE_THAT(r.meta.frameIntervalS, WithinRel(1.5, 1e-9));
        REQUIRE_FALSE(r.meta.name.empty());
        requireSame(*r.source->readAll(), a);
    }
    SECTION("metadata that disagrees with the page count falls back to pages as z") {
        TempPath f(".tif");
        writePlainTiff(f.str(), a, "czt", "ImageJ=1.54f\nimages=12\nchannels=5\nslices=5\nframes=1\n");
        const OpenResult r = openDataset(f.str());
        REQUIRE(r.meta.dims == Dims5{1, 1, 12, 6, 10});
        REQUIRE(r.metadataSummary.find("reading pages as z") != std::string::npos);
    }
}

TEST_CASE("openDataset errors and extensions", "[app][io]") {
    REQUIRE_THROWS_AS(openDataset("/nonexistent/file.tif"), std::runtime_error);
    REQUIRE_THROWS_AS(probeDataset("/nonexistent/file.tif"), std::runtime_error);
    const auto ext = readableExtensions();
    REQUIRE(std::find(ext.begin(), ext.end(), ".tif") != ext.end());
    REQUIRE((std::find(ext.begin(), ext.end(), ".zarr") != ext.end()) == sirius::app::zarrSupported());
    REQUIRE(sirius::app::zarrSupported() == sirius::zarrSupported());
}

// --- export --------------------------------------------------------------------------

TEST_CASE("export options: extension, availability, estimate and validation", "[app][io][export]") {
    ExportOptions o;
    o.path = "/tmp/x";
    CHECK(exportExtension(o) == ".ome.tif");
    o.tiff.omeXml = false;
    CHECK(exportExtension(o) == ".tif");
    o.format = ExportFormat::Raw;
    CHECK(exportExtension(o) == ".raw");
    o.format = ExportFormat::Zarr;
    CHECK(exportExtension(o) == ".zarr");
    CHECK(exportFormatAvailable(ExportFormat::Tiff));
    CHECK(exportFormatAvailable(ExportFormat::Zarr) == sirius::app::zarrSupported());

    const Dims5 d{2, 3, 4, 100, 100};
    o.format = ExportFormat::Tiff;
    o.dtype = PixelType::UInt16;
    CHECK(estimateExportBytes(d, o) == 24ull * 100 * 100 * 2);
    o.range.channels = {0};
    o.range.t1 = 1;
    CHECK(estimateExportBytes(d, o) == 4ull * 100 * 100 * 2);
    o.tiff.pyramidLevels = 2;
    CHECK(estimateExportBytes(d, o) == static_cast<std::uint64_t>(4.0 * 100 * 100 * 2 * 1.25));

    CHECK(validateExport(o, d).empty());
    o.path.clear();
    CHECK_FALSE(validateExport(o, d).empty());
    o.path = "/tmp/x";
    o.tiff.tiled = true;
    o.tiff.tileWidth = 8;
    CHECK_THAT(validateExport(o, d), Catch::Matchers::ContainsSubstring("16"));
    o.tiff.tileWidth = 256;
    o.scaling = ExportScaling::FixedRange;
    o.rangeLo = 1.0;
    o.rangeHi = 0.0;
    CHECK_FALSE(validateExport(o, d).empty());
    o.scaling = ExportScaling::Cast;
    o.range.t0 = 5;
    CHECK_THAT(validateExport(o, d), Catch::Matchers::ContainsSubstring("time range"));
    // the inverted range is empty, not a negative count cast to 16 EB beside that message
    CHECK(estimateExportBytes(d, o) == 0);
    o.range.t0 = 0;
    o.range.t1 = -1;
    o.range.z0 = 3;
    o.range.z1 = 1;
    CHECK_THAT(validateExport(o, d), Catch::Matchers::ContainsSubstring("z range"));
    CHECK(estimateExportBytes(d, o) == 0);
    o.includeLabels = true;
    CHECK(estimateExportBytes(d, o) == 0);
}

TEST_CASE("zarr export chunks are given as (c, t, z, y, x) and written as (t, c, z, y, x)", "[app][io][export][zarr]") {
    ZarrExportOptions z;
    z.chunk = {2, 5, 16, 256, 128};   // what the export dialog's "Chunk (c, t, z, y, x)" field says
    CHECK(zarrChunkShape(z) == std::vector<Index>{5, 2, 16, 256, 128});   // was assigned verbatim: t and c swapped
}

TEST_CASE("omeXml describes the array", "[app][io][export][ome]") {
    const Dims5 d{2, 3, 4, 8, 12};
    const std::string xml = omeXml(testMeta(d), d, PixelType::UInt16, "a.ome.tif");
    CHECK_THAT(xml, Catch::Matchers::ContainsSubstring("DimensionOrder=\"XYZTC\""));
    CHECK_THAT(xml, Catch::Matchers::ContainsSubstring("SizeC=\"2\""));
    CHECK_THAT(xml, Catch::Matchers::ContainsSubstring("Type=\"uint16\""));
    CHECK_THAT(xml, Catch::Matchers::ContainsSubstring("PhysicalSizeZ=\"0.3\""));
    CHECK_THAT(xml, Catch::Matchers::ContainsSubstring("Name=\"GFP\""));
    CHECK_THAT(xml, Catch::Matchers::ContainsSubstring("PlaneCount=\"24\""));
    const ParsedTiffMetadata back = parseTiffDescription(xml);
    CHECK(back.c == 2);
    CHECK(back.t == 3);
    CHECK(back.z == 4);
    CHECK(back.channels.size() == 2);
}

TEST_CASE("exportArray writes OME-TIFF that opens with the same data and metadata", "[app][io][export][tiff]") {
    const bool tiled = GENERATE(false, true);
    const int pyramid = GENERATE(1, 3);
    INFO("tiled=" << tiled << " pyramid=" << pyramid);
    TempPath f(".ome.tif");
    const Dims5 d{2, 2, 3, 20, 33};
    const Array5 a = testArray(d);
    const DatasetMeta m = testMeta(d);
    ExportOptions o;
    o.path = f.str();
    o.format = ExportFormat::Tiff;
    o.dtype = PixelType::Float32;
    o.tiff.tiled = tiled;
    o.tiff.tileWidth = o.tiff.tileHeight = 16;
    o.tiff.pyramidLevels = pyramid;
    o.tiff.compression = TiffCompression::Lzw;
    double last = 0.0;
    exportArray(a, m, nullptr, o, [&](double p, const std::string&) { last = p; });
    REQUIRE(last == 1.0);

    const TiffInfo info = inspectTiff(f.str());
    REQUIRE(info.pageCount() == 12);
    REQUIRE(info.levelCount() == static_cast<std::size_t>(pyramid));
    REQUIRE(info.page(0).description.find("<OME") != std::string::npos);
    REQUIRE(info.page(0).resolutionUnit == 3);

    const OpenResult r = openDataset(f.str());
    REQUIRE(r.meta.format == "ome-tiff");
    REQUIRE(r.meta.dims == d);
    REQUIRE(r.meta.channels[1].label == "GFP");
    REQUIRE_THAT(r.meta.voxelUm[0], WithinRel(0.1, 1e-6));
    REQUIRE_THAT(r.meta.voxelUm[2], WithinRel(0.3, 1e-9));
    requireSame(*r.source->readAll(), a);
}

TEST_CASE("exportArray converts pixel types with every scaling rule", "[app][io][export][scaling]") {
    const Dims5 d{1, 1, 2, 4, 5};
    Array5 a = testArray(d);
    a.at(0, 0, 0, 0, 0) = -10.f;   // below zero: clamped by Cast into unsigned types
    a.at(0, 0, 1, 3, 4) = 1000.f;
    const DatasetMeta m = testMeta(d);

    auto exportAs = [&](PixelType type, ExportScaling scaling, double lo = 0.0, double hi = 1.0) {
        TempPath f(".tif");
        ExportOptions o;
        o.path = f.str();
        o.dtype = type;
        o.scaling = scaling;
        o.rangeLo = lo;
        o.rangeHi = hi;
        o.tiff.omeXml = false;
        exportArray(a, m, nullptr, o);
        return readTiffStackAny(f.str());
    };

    SECTION("Cast clamps and rounds into uint8") {
        auto any = exportAs(PixelType::UInt8, ExportScaling::Cast);
        auto& s = std::get<ImageStack<std::uint8_t>>(any);
        REQUIRE(s(0, 0, 0) == 0);
        REQUIRE(s(1, 3, 4) == 255);
        REQUIRE(s(0, 0, 1) == static_cast<std::uint8_t>(std::lround(a.at(0, 0, 0, 0, 1))));
    }
    SECTION("MinMax fills the uint16 range") {
        auto any = exportAs(PixelType::UInt16, ExportScaling::MinMax);
        auto& s = std::get<ImageStack<std::uint16_t>>(any);
        REQUIRE(s(0, 0, 0) == 0);
        REQUIRE(s(1, 3, 4) == 65535);
    }
    SECTION("FixedRange maps the window to int16's positive range") {
        auto any = exportAs(PixelType::Int16, ExportScaling::FixedRange, 0.0, 100.0);
        auto& s = std::get<ImageStack<std::int16_t>>(any);
        REQUIRE(s(0, 0, 0) == 0);              // -10 is below the window: saturates at 0
        REQUIRE(s(1, 3, 4) == 32767);          // 1000 saturates at the top
        const float v = a.at(0, 0, 0, 1, 1);
        if (v <= 100.f) REQUIRE(s(0, 1, 1) == static_cast<std::int16_t>(std::lround(v / 100.0 * 32767.0)));
    }
    SECTION("Percentile normalizes float output to 0..1") {
        auto any = exportAs(PixelType::Float32, ExportScaling::Percentile);
        auto& s = std::get<ImageStack<float>>(any);
        REQUIRE(s(0, 0, 0) <= 0.0f);
        REQUIRE(s(1, 3, 4) >= 1.0f);
    }
    SECTION("double keeps the values under Cast") {
        auto any = exportAs(PixelType::Float64, ExportScaling::Cast);
        auto& s = std::get<ImageStack<double>>(any);
        REQUIRE(s(0, 0, 0) == -10.0);
        REQUIRE(s(1, 3, 4) == 1000.0);
    }
}

TEST_CASE("exportArray honours the range and writes sidecars", "[app][io][export][range]") {
    TempPath f(".tif");
    const Dims5 d{3, 4, 5, 6, 7};
    const Array5 a = testArray(d);
    const DatasetMeta m = testMeta(d);
    LabelVolume labels(d.t, d.z, d.y, d.x);
    labels.plane(2, 2)[0] = 7;   // t = 2, z = 2: inside the exported range
    ExportOptions o;
    o.path = f.str();
    o.tiff.omeXml = true;
    o.range.channels = {2, 0};
    o.range.t0 = 1;
    o.range.t1 = 3;
    o.range.z0 = 2;
    o.range.z1 = 4;
    o.includePipeline = true;
    o.pipelineToml = "# pipeline\nversion = 1\n";
    o.includeLabels = true;
    exportArray(a, m, &labels, o);

    const OpenResult r = openDataset(f.str());
    REQUIRE(r.meta.dims == Dims5{2, 2, 2, 6, 7});
    REQUIRE(r.meta.channels[0].label == m.channels[2].label);
    auto all = r.source->readAll();
    REQUIRE(all->at(0, 0, 0, 0, 0) == a.at(2, 1, 2, 0, 0));
    REQUIRE(all->at(1, 1, 1, 5, 6) == a.at(0, 2, 3, 5, 6));

    const std::string base = f.str().substr(0, f.str().size() - 4);
    REQUIRE(fs::exists(base + ".pipeline.toml"));
    REQUIRE(fs::exists(base + ".labels.tif"));
    const TiffInfo lab = inspectTiff(base + ".labels.tif");
    REQUIRE(lab.pageCount() == 4);   // 2 t x 2 z
    REQUIRE(lab.pixelType() == PixelType::UInt32);
    auto labs = readTiffStack<std::uint32_t>(base + ".labels.tif");
    REQUIRE(labs(2, 0, 0) == 7);     // t = 2 is the second exported time point, z = 2 its first plane
    std::error_code ec;
    fs::remove(base + ".pipeline.toml", ec);
    fs::remove(base + ".labels.tif", ec);
}

TEST_CASE("exportArray writes raw float32 with a JSON sidecar", "[app][io][export][raw]") {
    TempPath f(".raw");
    const Dims5 d{2, 1, 3, 4, 5};
    const Array5 a = testArray(d);
    ExportOptions o;
    o.path = f.str();
    o.format = ExportFormat::Raw;
    exportArray(a, testMeta(d), nullptr, o);
    REQUIRE(fs::file_size(f.path) == a.bytes());
    std::ifstream in(f.path, std::ios::binary);
    std::vector<float> back(static_cast<std::size_t>(a.numel()));
    in.read(reinterpret_cast<char*>(back.data()), static_cast<std::streamsize>(a.bytes()));
    REQUIRE(std::equal(back.begin(), back.end(), a.data()));
    std::ifstream side(f.str() + ".json");
    std::string text((std::istreambuf_iterator<char>(side)), std::istreambuf_iterator<char>());
    REQUIRE(text.find("\"dtype\": \"float32\"") != std::string::npos);
    REQUIRE(text.find("\"order\": \"ctzyx\"") != std::string::npos);
}

TEST_CASE("exportArray can be cancelled", "[app][io][export][cancel]") {
    TempPath f(".tif");
    const Dims5 d{1, 1, 40, 8, 8};
    const Array5 a = testArray(d);
    ExportOptions o;
    o.path = f.str();
    o.dtype = PixelType::UInt16;
    int calls = 0;
    REQUIRE_THROWS_WITH(exportArray(a, testMeta(d), nullptr, o, {}, [&] { return ++calls > 3; }), "cancelled");
}

TEST_CASE("exportArray writes OME-Zarr that opens again", "[app][io][export][zarr]") {
    if (!sirius::app::zarrSupported()) SKIP("built without TensorStore");
    const ExportFormat format = GENERATE(ExportFormat::Zarr, ExportFormat::N5);
    const int version = GENERATE(2, 3);
    if (format == ExportFormat::N5 && version == 2) SKIP("N5 has one version");
    INFO("format=" << static_cast<int>(format) << " version=" << version);
    TempPath f(format == ExportFormat::N5 ? ".n5" : ".zarr");
    const Dims5 d{2, 3, 4, 10, 14};
    const Array5 a = testArray(d);
    const DatasetMeta m = testMeta(d);
    LabelVolume labels(d.t, d.z, d.y, d.x);
    labels.volume(1)[3] = 9;
    ExportOptions o;
    o.path = f.str();
    o.format = format;
    o.dtype = PixelType::UInt16;
    o.scaling = ExportScaling::MinMax;
    o.zarr.zarrVersion = version;
    o.zarr.chunk = {1, 1, 2, 8, 8};
    o.zarr.pyramidLevels = 2;
    o.zarr.codec = "blosc-zstd";
    o.includeLabels = true;
    exportArray(a, m, &labels, o);

    const ZarrArrayInfo info = inspectZarr(f.str());
    REQUIRE(info.isGroup);
    REQUIRE(info.axes == std::vector<std::string>{"t", "c", "z", "y", "x"});
    REQUIRE(info.shape == std::vector<Index>{3, 2, 4, 10, 14});
    REQUIRE(info.multiscalePaths.size() == 2);
    REQUIRE(info.channelNames == std::vector<std::string>{"DAPI", "GFP"});
    REQUIRE_THAT(info.scale[4], WithinRel(0.1, 1e-9));
    REQUIRE_THAT(info.scale[2], WithinRel(0.3, 1e-9));

    const OpenResult r = openDataset(f.str());
    REQUIRE(r.meta.dims == d);
    REQUIRE(r.meta.format == (format == ExportFormat::N5 ? "n5" : "zarr"));
    REQUIRE(r.meta.channels[1].label == "GFP");
    REQUIRE_THAT(r.meta.voxelUm[2], WithinRel(0.3, 1e-9));
    REQUIRE_THAT(r.meta.frameIntervalS, WithinRel(1.5, 1e-9));
    auto all = r.source->readAll();
    // MinMax into uint16: the brightest voxel reads 65535, the dimmest 0
    const auto [lo, hi] = minMax(*all);
    REQUIRE(lo == 0.f);
    REQUIRE(hi == 65535.f);
    // monotone: the ordering of two voxels survives
    REQUIRE((a.at(1, 2, 3, 4, 5) < a.at(0, 0, 0, 0, 1)) == (all->at(1, 2, 3, 4, 5) < all->at(0, 0, 0, 0, 1)));

    const ZarrArrayInfo lab = inspectZarr((f.path / "labels" / "labels").string());
    REQUIRE(lab.shape == std::vector<Index>{3, 4, 10, 14});
    REQUIRE(lab.pixelType == PixelType::UInt32);
    const Buffer<std::uint32_t> lv = readZarr<std::uint32_t>((f.path / "labels" / "labels").string(), {1, 0, 0, 0}, {1, 1, 1, 14});
    REQUIRE(lv.data()[3] == 9);
}

TEST_CASE("openDataset maps zarr axes by name and by position", "[app][io][zarr]") {
    if (!sirius::app::zarrSupported()) SKIP("built without TensorStore");
    SECTION("named (c, z, y, x) without time") {
        TempPath f(".zarr");
        const Dims5 d{2, 1, 3, 6, 9};
        const Array5 a = testArray(d);
        std::vector<float> data(static_cast<std::size_t>(a.numel()));
        std::copy_n(a.data(), a.numel(), data.data());
        ZarrWriteOptions w;
        w.zarrVersion = 2;
        w.axes = {"c", "z", "y", "x"};
        w.scale = {1.0, 0.5, 0.2, 0.2};
        w.chunks = {1, 1, 6, 9};
        writeZarr<float>(f.str(), data.data(), {2, 3, 6, 9}, w);
        const OpenResult r = openDataset(f.str());
        REQUIRE(r.meta.dims == d);
        REQUIRE_THAT(r.meta.voxelUm[2], WithinRel(0.5, 1e-9));
        REQUIRE(r.metadataSummary.find("zarr v2") != std::string::npos);
        requireSame(*r.source->readAll(), a);
        std::vector<float> plane(static_cast<std::size_t>(d.planeSize()));
        r.source->readPlane(1, 0, 2, plane.data());
        REQUIRE(plane[4] == a.at(1, 0, 2, 0, 4));
    }
    SECTION("bare 3D array is (z, y, x)") {
        TempPath f(".zarr");
        std::vector<float> data(3 * 4 * 5);
        for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i);
        ZarrWriteOptions w;
        w.omeNgff = false;
        writeZarr<float>(f.str(), data.data(), {3, 4, 5}, w);
        const DatasetMeta m = probeDataset(f.str());
        REQUIRE(m.dims == Dims5{1, 1, 3, 4, 5});
        const OpenResult r = openDataset(f.str());
        std::vector<float> plane(20);
        r.source->readPlane(0, 0, 2, plane.data());
        REQUIRE(plane[7] == 47.f);
    }
}
