// Rendering a step's output as the viewer draws it, and encoding it
// (app/core/display_model.hpp, image_encode.hpp): the additive channel blend,
// one channel alone, regions and sub-sampling, the label overlay, the
// headless preparation of volumes and projections (lazy, in memory, too
// large, cancelled), and the PNG / JPEG / base64 bytes sirius-cli hands out.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/array.hpp"
#include "core/array_source.hpp"
#include "core/cancel.hpp"
#include "core/dataset.hpp"
#include "core/display_model.hpp"
#include "core/image_encode.hpp"
#include "core/labels.hpp"
#include "core/operation.hpp"
#include "core/workbench.hpp"

#include "temp_path.hpp"

using namespace sirius;
using namespace sirius::app;
using sirius::app::display::DisplayModel;
using sirius::app::display::DisplayWindow;
using sirius::app::display::Image;
using sirius::app::display::RectI;
namespace fs = std::filesystem;

namespace {

    const fs::path kData = SIRIUS_TEST_DATA_DIR;

    // An Image pixel is 0xAABBGGRR: the bytes R, G, B, A in memory.
    int red(std::uint32_t p) { return static_cast<int>(p & 0xff); }
    int green(std::uint32_t p) { return static_cast<int>((p >> 8) & 0xff); }
    int blue(std::uint32_t p) { return static_cast<int>((p >> 16) & 0xff); }
    std::uint32_t pixel(const Image& img, int x, int y) { return img.scanLine(y)[x]; }

    // Windows of 0 .. 255 map an integer value to the same grey level, so the
    // expected pixels below are exact.
    const DisplayWindow kByteWindow{0.0f, 255.0f, 1.0f};

    ChannelInfo channel(const char* label, float r, float g, float b) {
        ChannelInfo ch;
        ch.label = label;
        ch.color = {r, g, b};
        return ch;
    }

    // Two channels in memory, red and green: ch 0 is 50 * x (0 .. 250 across
    // the six columns), ch 1 is 100 everywhere.
    std::shared_ptr<StepOutput> twoChannelOutput() {
        const Dims5 d{2, 1, 2, 4, 6};
        auto a = std::make_shared<Array5>(Array5::zeros(d));
        for (Index z = 0; z < d.z; ++z)
            for (Index y = 0; y < d.y; ++y)
                for (Index x = 0; x < d.x; ++x) {
                    a->at(0, 0, z, y, x) = static_cast<float>(50 * x);
                    a->at(1, 0, z, y, x) = 100.0f;
                }
        auto out = std::make_shared<StepOutput>();
        out->meta.dims = d;
        out->meta.format = "memory";
        out->meta.channels = {channel("red", 1.f, 0.f, 0.f), channel("green", 0.f, 1.f, 0.f)};
        out->array = a;
        return out;
    }

    ViewState visible(std::vector<bool> channels) {
        ViewState vs;
        vs.channelVisible = std::move(channels);
        return vs;
    }

    // A lazy source that makes its planes up (z * 100 + y * 10 + x, or one
    // constant) and counts the planes it is asked for. Its extents cost
    // nothing until a plane is read, so it can pretend to be far larger than
    // memory. `onRead` sees every plane read before it is made.
    class CountingSource final : public ArraySource {
    public:
        explicit CountingSource(Dims5 d) {
            meta_.dims = d;
            meta_.format = "counting";
            meta_.channels.resize(static_cast<std::size_t>(d.c));   // white: the grey level is the pixel
        }
        const DatasetMeta& meta() const noexcept override { return meta_; }
        void readPlane(Index c, Index t, Index z, float* out) const override {
            const Dims5& d = meta_.dims;
            if (c < 0 || c >= d.c || t < 0 || t >= d.t || z < 0 || z >= d.z) throw std::out_of_range("CountingSource: no such plane");
            if (onRead) onRead(c, t, z);
            ++planesRead;
            for (Index y = 0; y < d.y; ++y)
                for (Index x = 0; x < d.x; ++x) out[y * d.x + x] = constant ? *constant : value(z, y, x);
        }
        static float value(Index z, Index y, Index x) { return static_cast<float>(z * 100 + y * 10 + x); }

        mutable std::atomic<int> planesRead{0};
        std::optional<float> constant;
        std::function<void(Index c, Index t, Index z)> onRead;

    private:
        DatasetMeta meta_;
    };

    std::shared_ptr<StepOutput> lazyOutput(const std::shared_ptr<ArraySource>& source) {
        auto out = std::make_shared<StepOutput>();
        out->meta = source->meta();
        out->source = source;
        return out;
    }

    // --- PNG / JPEG structure ------------------------------------------------------------

    std::uint32_t bigEndian32(const std::vector<std::uint8_t>& b, std::size_t at) {
        return (static_cast<std::uint32_t>(b.at(at)) << 24) | (static_cast<std::uint32_t>(b.at(at + 1)) << 16) |
               (static_cast<std::uint32_t>(b.at(at + 2)) << 8) | static_cast<std::uint32_t>(b.at(at + 3));
    }

    int bigEndian16(const std::vector<std::uint8_t>& b, std::size_t at) { return (b.at(at) << 8) | b.at(at + 1); }

    std::uint32_t crc32(const std::uint8_t* p, std::size_t n) {
        std::uint32_t crc = 0xffffffffu;
        for (std::size_t i = 0; i < n; ++i) {
            crc ^= p[i];
            for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
        }
        return crc ^ 0xffffffffu;
    }

    struct Chunk {
        std::string type;
        std::size_t data = 0;   // offset of the chunk's data
        std::uint32_t length = 0;
    };

    // The chunks of a PNG after its signature; each one's CRC is checked.
    std::vector<Chunk> pngChunks(const std::vector<std::uint8_t>& png) {
        std::vector<Chunk> chunks;
        std::size_t at = 8;
        while (at + 12 <= png.size()) {
            Chunk c;
            c.length = bigEndian32(png, at);
            c.type.assign(reinterpret_cast<const char*>(png.data() + at + 4), 4);
            c.data = at + 8;
            REQUIRE(c.data + c.length + 4 <= png.size());
            CHECK(crc32(png.data() + at + 4, c.length + 4) == bigEndian32(png, c.data + c.length));
            chunks.push_back(c);
            at = c.data + c.length + 4;
        }
        CHECK(at == png.size());
        return chunks;
    }

    std::vector<std::uint8_t> gradient(int width, int height, int channels) {
        std::vector<std::uint8_t> px(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(channels));
        for (std::size_t i = 0; i < px.size(); ++i) px[i] = static_cast<std::uint8_t>((i * 7) & 0xff);
        return px;
    }

    std::vector<std::uint8_t> bytesOf(const std::string& s) { return std::vector<std::uint8_t>(s.begin(), s.end()); }

    struct TempDir {
        fs::path path = test::uniqueTempPath("render", "");
        ~TempDir() {
            std::error_code ec;
            fs::remove_all(path, ec);
        }
    };

} // namespace

// --- rendering ------------------------------------------------------------------------------

TEST_CASE("render: renderXY blends the visible channels additively in their colours", "[app][render]") {
    DisplayModel model;
    model.setOutput(twoChannelOutput());
    REQUIRE(model.valid());
    model.setWindow(0, kByteWindow);
    model.setWindow(1, kByteWindow);

    Image img;
    model.renderXY(0, 0, visible({true, true}), 1, img);
    REQUIRE(img.width == 6);
    REQUIRE(img.height == 4);
    for (int x = 0; x < 6; ++x) {
        const std::uint32_t p = pixel(img, x, 2);
        CHECK(red(p) == 50 * x);
        CHECK(green(p) == 100);
        CHECK(blue(p) == 0);
        CHECK((p >> 24) == 0xffu);   // opaque
    }
    // The bytes are what an RGBA encoder takes as they are.
    const std::uint8_t* bytes = img.bytes();
    CHECK(bytes[4 * 5 + 0] == 250);
    CHECK(bytes[4 * 5 + 1] == 100);
    CHECK(bytes[4 * 5 + 2] == 0);
    CHECK(bytes[4 * 5 + 3] == 255);
}

TEST_CASE("render: one channel alone is drawn in its own colour, a white one in grey", "[app][render]") {
    DisplayModel model;
    auto out = twoChannelOutput();
    model.setOutput(out);
    model.setWindow(0, kByteWindow);
    model.setWindow(1, kByteWindow);

    Image img;
    model.renderXY(0, 1, visible({true, false}), 1, img);
    CHECK(red(pixel(img, 3, 0)) == 150);
    CHECK(green(pixel(img, 3, 0)) == 0);
    model.renderXY(0, 1, visible({false, true}), 1, img);
    CHECK(red(pixel(img, 3, 0)) == 0);
    CHECK(green(pixel(img, 3, 0)) == 100);

    // A panel per channel, as the "channels" layout draws them: the same
    // data with white channels comes out grey.
    auto grey = std::make_shared<StepOutput>(*out);
    for (ChannelInfo& ch : grey->meta.channels) ch.color = {1.f, 1.f, 1.f};
    model.setOutput(grey);
    model.setWindow(1, kByteWindow);
    model.renderXY(0, 1, visible({false, true}), 1, img);
    const std::uint32_t p = pixel(img, 3, 0);
    CHECK(red(p) == 100);
    CHECK(green(p) == 100);
    CHECK(blue(p) == 100);

    // Nothing visible: the empty background, not black.
    model.renderXY(0, 1, visible({false, false}), 1, img);
    CHECK(red(pixel(img, 0, 0)) == 0x0a);
    CHECK(green(pixel(img, 0, 0)) == 0x09);
}

TEST_CASE("render: a region and a sub-sampling factor pick the pixels they name", "[app][render]") {
    DisplayModel model;
    model.setOutput(twoChannelOutput());
    model.setWindow(0, kByteWindow);
    const ViewState vs = visible({true, false});

    Image img;
    model.renderXY(0, 0, vs, 1, img, RectI{2, 1, 3, 2});
    REQUIRE(img.width == 3);
    REQUIRE(img.height == 2);
    CHECK(red(pixel(img, 0, 0)) == 100);   // column 2
    CHECK(red(pixel(img, 2, 1)) == 200);   // column 4

    // Factor 2 halves each side, rounding up, and averages 2 x 2 voxels.
    model.renderXY(0, 0, vs, 2, img);
    REQUIRE(img.width == 3);
    REQUIRE(img.height == 2);
    CHECK(red(pixel(img, 1, 0)) == 125);   // columns 2 and 3: (100 + 150) / 2

    // A region reaching past the plane is clamped to it.
    model.renderXY(0, 0, vs, 1, img, RectI{4, 2, 10, 10});
    CHECK(img.width == 2);
    CHECK(img.height == 2);
}

TEST_CASE("render: the label overlay changes the pixels of a label and no others", "[app][render]") {
    auto out = twoChannelOutput();
    auto labels = std::make_shared<LabelVolume>(1, 2, 4, 6);
    for (Index y = 1; y <= 2; ++y)
        for (Index x = 1; x <= 2; ++x) labels->plane(0, 0)[y * 6 + x] = 3;
    out->labels = labels;

    DisplayModel model;
    model.setOutput(out);
    REQUIRE(model.hasLabels());
    model.setWindow(0, kByteWindow);
    model.setWindow(1, kByteWindow);
    ViewState vs = visible({true, true});

    Image plain;
    model.renderXY(0, 0, vs, 1, plain);
    Image drawn = plain;
    model.overlayLabelsXY(0, 0, 1, vs, drawn);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 6; ++x) {
            const bool inside = y >= 1 && y <= 2 && x >= 1 && x <= 2;
            INFO("pixel (" << x << ", " << y << ")");
            CHECK((pixel(drawn, x, y) != pixel(plain, x, y)) == inside);
        }

    // Solo on another label draws nothing of this one.
    vs.selectedLabel = 9;
    vs.soloLabel = true;
    Image solo = plain;
    model.overlayLabelsXY(0, 0, 1, vs, solo);
    CHECK(solo.pixels == plain.pixels);

    // The selected label's outline is white.
    vs.selectedLabel = 3;
    vs.soloLabel = false;
    Image selected = plain;
    model.overlayLabelsXY(0, 0, 1, vs, selected);
    const std::uint32_t p = pixel(selected, 1, 1);
    CHECK(red(p) == 255);
    CHECK(green(p) == 255);
    CHECK(blue(p) == 255);

    // A plane without labels changes nothing.
    Image other = plain;
    model.overlayLabelsXY(0, 1, 1, vs, other);
    CHECK(other.pixels == plain.pixels);
}

// --- projections and the headless preparation --------------------------------------------------

TEST_CASE("render: projectAndRange skips NaN and gives the exact range", "[app][render]") {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    // (z 3, y 1, x 3)
    const std::vector<float> vol{1.0f, nan, -2.0f, 4.0f, nan, 0.5f, 3.0f, nan, -7.0f};
    std::vector<float> mip(3);
    float lo = 0.0f, hi = 0.0f;
    display::projectAndRange(vol.data(), 3, 1, 3, mip.data(), lo, hi);
    CHECK(mip[0] == 4.0f);
    CHECK(mip[1] == -7.0f);   // NaN at every z: the volume's minimum
    CHECK(mip[2] == 0.5f);
    CHECK(lo == -7.0f);
    CHECK(hi == 4.0f);

    SECTION("a constant volume gets a range one wide") {
        const std::vector<float> flat(6, 5.0f);
        std::vector<float> m(3);
        display::projectAndRange(flat.data(), 2, 1, 3, m.data(), lo, hi);
        CHECK(lo == 5.0f);
        CHECK(hi == 6.0f);
        CHECK(m == std::vector<float>(3, 5.0f));
    }
    SECTION("nothing but NaN projects to 0 .. 1") {
        const std::vector<float> empty(4, nan);
        std::vector<float> m(2);
        display::projectAndRange(empty.data(), 2, 1, 2, m.data(), lo, hi);
        CHECK(lo == 0.0f);
        CHECK(hi == 1.0f);
        CHECK(m == std::vector<float>(2, 0.0f));
    }
}

TEST_CASE("render: prepareProjectionSync on a lazy TIFF equals a brute-force MIP", "[app][render]") {
    const OpenResult opened = openDataset((kData / "raw.tif").string());
    REQUIRE(opened.source);
    REQUIRE_FALSE(opened.source->inMemory());
    const Dims5 d = opened.meta.dims;
    REQUIRE(d.z > 1);

    DisplayModel model;
    model.setOutput(lazyOutput(opened.source));
    REQUIRE(model.mipIfReady(0, 0) == nullptr);
    model.prepareProjectionSync(0, 0);
    const float* mip = model.mipIfReady(0, 0);
    REQUIRE(mip != nullptr);
    CHECK(model.volumeIfReady(0, 0) == nullptr);   // streamed: no volume kept

    const Index n = d.y * d.x;
    std::vector<float> vol(static_cast<std::size_t>(d.z * n));
    opened.source->readVolume(0, 0, vol.data());
    std::vector<float> expected(static_cast<std::size_t>(n), -std::numeric_limits<float>::infinity());
    for (Index z = 0; z < d.z; ++z)
        for (Index i = 0; i < n; ++i) {
            const float v = vol[static_cast<std::size_t>(z * n + i)];
            if (!std::isnan(v) && v > expected[static_cast<std::size_t>(i)]) expected[static_cast<std::size_t>(i)] = v;
        }
    Index mismatches = 0;
    for (Index i = 0; i < n; ++i)
        if (mip[i] != expected[static_cast<std::size_t>(i)]) ++mismatches;
    CHECK(mismatches == 0);

    // renderMIP draws it: in the Full window (the projection's exact range)
    // the brightest pixel is white, give or take the float rounding.
    model.setWindowMode(DisplayModel::WindowMode::Full);
    Image img;
    model.renderMIP(0, ViewState{}, 1, img);
    REQUIRE(img.width == static_cast<int>(d.x));
    REQUIRE(img.height == static_cast<int>(d.y));
    int brightest = 0;
    for (std::uint32_t p : img.pixels) brightest = std::max(brightest, red(p));
    CHECK(brightest >= 250);
}

TEST_CASE("render: prepareProjectionSync streams a made-up source once and draws its maximum", "[app][render]") {
    auto source = std::make_shared<CountingSource>(Dims5{1, 1, 3, 4, 6});
    DisplayModel model;
    model.setOutput(lazyOutput(source));
    model.prepareProjectionSync(0, 0);
    CHECK(source->planesRead.load() == 3);
    model.prepareProjectionSync(0, 0);   // already there: nothing is read
    CHECK(source->planesRead.load() == 3);

    model.setWindow(0, kByteWindow);
    Image img;
    model.renderMIP(0, ViewState{}, 1, img);
    REQUIRE(img.width == 6);
    REQUIRE(img.height == 4);
    CHECK(red(pixel(img, 5, 3)) == 235);   // z 2: 200 + 30 + 5
    CHECK(red(pixel(img, 0, 0)) == 200);
}

TEST_CASE("render: prepareVolumeSync reads a lazy volume once and the re-slices draw it", "[app][render]") {
    auto source = std::make_shared<CountingSource>(Dims5{1, 1, 3, 4, 6});
    DisplayModel model;
    model.setOutput(lazyOutput(source));
    CHECK(model.volumeState(0, 0) == DisplayModel::VolumeState::Wanted);
    REQUIRE(model.prepareVolumeSync(0, 0));
    CHECK(source->planesRead.load() == 3);
    CHECK(model.volumeIfReady(0, 0) != nullptr);
    CHECK(model.mipIfReady(0, 0) != nullptr);
    CHECK(model.volumeState(0, 0) == DisplayModel::VolumeState::Ready);
    CHECK_FALSE(model.volumeTooLarge());
    REQUIRE(model.prepareVolumeSync(0, 0));   // ready: nothing is read again
    CHECK(source->planesRead.load() == 3);

    model.setWindow(0, kByteWindow);
    const ViewState vs;
    Image xz;
    model.renderXZ(0, 2, vs, 1, xz);   // rows z, columns x, at y 2
    REQUIRE(xz.width == 6);
    REQUIRE(xz.height == 3);
    CHECK(red(pixel(xz, 4, 1)) == 124);   // z 1, y 2, x 4
    Image yz;
    model.renderYZ(0, 5, vs, 1, yz);   // rows y, columns z, at x 5
    REQUIRE(yz.width == 3);
    REQUIRE(yz.height == 4);
    CHECK(red(pixel(yz, 2, 3)) == 235);   // z 2, y 3, x 5

    // The exact range the loader would give becomes the Full window.
    model.setWindowMode(DisplayModel::WindowMode::Full);
    const DisplayWindow w = model.window(0, 0);
    CHECK(w.lo == 0.0f);
    CHECK(w.hi == 235.0f);
}

TEST_CASE("render: prepareVolumeSync projects an in-memory output without reading", "[app][render]") {
    DisplayModel model;
    model.setOutput(twoChannelOutput());
    REQUIRE(model.prepareVolumeSync(1, 0));
    const float* mip = model.mipIfReady(1, 0);
    REQUIRE(mip != nullptr);
    CHECK(mip[0] == 100.0f);
    CHECK(model.volumeIfReady(1, 0) != nullptr);
}

TEST_CASE("render: prepareProjectionSync projects a volume already in memory without reading", "[app][render]") {
    const Dims5 d{1, 1, 3, 4, 6};
    const Index n = d.y * d.x;

    SECTION("an in-memory output") {
        auto a = std::make_shared<Array5>(Array5::zeros(d));
        for (Index z = 0; z < d.z; ++z)
            for (Index y = 0; y < d.y; ++y)
                for (Index x = 0; x < d.x; ++x) a->at(0, 0, z, y, x) = CountingSource::value(z, y, x);
        auto out = std::make_shared<StepOutput>();
        out->meta.dims = d;
        out->meta.format = "memory";
        out->meta.channels.resize(1);
        out->array = a;

        DisplayModel model;
        model.setOutput(out);
        REQUIRE(model.mipIfReady(0, 0) == nullptr);
        model.prepareProjectionSync(0, 0);
        const float* mip = model.mipIfReady(0, 0);
        REQUIRE(mip != nullptr);
        Index mismatches = 0;
        for (Index i = 0; i < n; ++i)
            if (mip[i] != CountingSource::value(d.z - 1, i / d.x, i % d.x)) ++mismatches;
        CHECK(mismatches == 0);
        model.setWindowMode(DisplayModel::WindowMode::Full);
        const DisplayWindow w = model.window(0, 0);
        CHECK(w.lo == 0.0f);
        CHECK(w.hi == 235.0f);
    }
    SECTION("a lazy output whose volume is cached") {
        auto source = std::make_shared<CountingSource>(d);
        DisplayModel model;
        model.setOutput(lazyOutput(source));
        // Twice what the source would give, installed without a projection:
        // the projection can only come from this volume.
        auto volume = std::make_shared<Buffer<float>>(Shape{d.z, d.y, d.x});
        for (Index z = 0; z < d.z; ++z)
            for (Index y = 0; y < d.y; ++y)
                for (Index x = 0; x < d.x; ++x) volume->data()[z * n + y * d.x + x] = 2.0f * CountingSource::value(z, y, x);
        model.installVolume(0, 0, volume, nullptr, 0.0f, 0.0f);
        REQUIRE(model.volumeIfReady(0, 0) != nullptr);
        REQUIRE(model.mipIfReady(0, 0) == nullptr);

        model.prepareProjectionSync(0, 0);
        CHECK(source->planesRead.load() == 0);
        const float* mip = model.mipIfReady(0, 0);
        REQUIRE(mip != nullptr);
        CHECK(mip[0] == 400.0f);       // z 2, y 0, x 0, doubled
        CHECK(mip[n - 1] == 470.0f);   // z 2, y 3, x 5, doubled
    }
}

TEST_CASE("render: prepareVolumeSync after a projection reads the volume and keeps the projection", "[app][render]") {
    auto source = std::make_shared<CountingSource>(Dims5{1, 1, 3, 4, 6});
    DisplayModel model;
    model.setOutput(lazyOutput(source));
    model.prepareProjectionSync(0, 0);
    const float* mip = model.mipIfReady(0, 0);
    REQUIRE(mip != nullptr);
    REQUIRE(source->planesRead.load() == 3);

    REQUIRE(model.prepareVolumeSync(0, 0));
    CHECK(source->planesRead.load() == 6);
    CHECK(model.volumeIfReady(0, 0) != nullptr);
    CHECK(model.mipIfReady(0, 0) == mip);   // the same buffer: not projected a second time
    CHECK(model.volumeState(0, 0) == DisplayModel::VolumeState::Ready);
    model.setWindowMode(DisplayModel::WindowMode::Full);
    const DisplayWindow w = model.window(0, 0);   // the range the projection found
    CHECK(w.lo == 0.0f);
    CHECK(w.hi == 235.0f);
}

TEST_CASE("render: prepareVolumeSync drops the other time points' volumes before it reads", "[app][render]") {
    auto source = std::make_shared<CountingSource>(Dims5{2, 2, 3, 4, 6});
    DisplayModel model;
    model.setOutput(lazyOutput(source));
    REQUIRE(model.prepareVolumeSync(0, 0));
    REQUIRE(model.prepareVolumeSync(1, 0));

    // Nothing shows t 0 while t 1 is read, so its volumes need not wait.
    bool heldOld = false;
    source->onRead = [&model, &heldOld](Index, Index, Index) {
        if (model.volumeIfReady(0, 0) || model.volumeIfReady(1, 0)) heldOld = true;
    };
    REQUIRE(model.prepareVolumeSync(0, 1));
    source->onRead = {};
    CHECK_FALSE(heldOld);
    CHECK(model.volumeIfReady(0, 1) != nullptr);
    CHECK(model.volumeIfReady(0, 0) == nullptr);
    // The projections are small and stay for play.
    CHECK(model.mipIfReady(0, 0) != nullptr);
    CHECK(model.mipIfReady(1, 0) != nullptr);
}

TEST_CASE("render: a constant volume whose range is empty in float is prepared once", "[app][render]") {
    // 2^25: floats there are 4 apart, so lo + 1 is lo and no range is kept.
    const float big = 33554432.0f;
    std::vector<float> m(1);
    float lo = 0.0f, hi = 0.0f;
    const std::vector<float> flat(2, big);
    display::projectAndRange(flat.data(), 2, 1, 1, m.data(), lo, hi);
    REQUIRE_FALSE(hi > lo);

    auto source = std::make_shared<CountingSource>(Dims5{1, 1, 3, 4, 6});
    source->constant = big;
    DisplayModel model;
    model.setOutput(lazyOutput(source));
    model.prepareProjectionSync(0, 0);
    model.prepareProjectionSync(0, 0);
    CHECK(source->planesRead.load() == 3);
    REQUIRE(model.prepareVolumeSync(0, 0));
    REQUIRE(model.prepareVolumeSync(0, 0));
    CHECK(source->planesRead.load() == 6);
    const float* mip = model.mipIfReady(0, 0);
    REQUIRE(mip != nullptr);
    CHECK(mip[0] == big);
}

TEST_CASE("render: a volume above the cache limit is refused unread and projected plane by plane", "[app][render]") {
    // 4096 planes of 512 x 512 floats: 4 GiB, above the 3 GiB cap.
    auto source = std::make_shared<CountingSource>(Dims5{1, 1, 4096, 512, 512});
    DisplayModel model;
    model.setOutput(lazyOutput(source));
    CHECK_FALSE(model.prepareVolumeSync(0, 0));
    CHECK(model.volumeTooLarge());
    CHECK(source->planesRead.load() == 0);

    // The projection streams any size; a cancel between planes stops it and
    // leaves nothing installed.
    int polls = 0;
    CHECK_THROWS_AS(model.prepareProjectionSync(0, 0, [&polls] { return ++polls > 3; }), CancelledError);
    CHECK(source->planesRead.load() == 3);
    CHECK(model.mipIfReady(0, 0) == nullptr);
}

TEST_CASE("render: a cancelled volume read installs nothing", "[app][render]") {
    auto source = std::make_shared<CountingSource>(Dims5{1, 1, 100, 8, 8});
    DisplayModel model;
    model.setOutput(lazyOutput(source));

    CHECK_THROWS_AS(model.prepareVolumeSync(0, 0, [] { return true; }), CancelledError);
    CHECK(source->planesRead.load() == 0);

    // The first poll comes before the read, the second from the source's
    // progress report partway through it; how many planes lie between two
    // reports is the source's business, so only "some, not all" is checked.
    int polls = 0;
    CHECK_THROWS_AS(model.prepareVolumeSync(0, 0, [&polls] { return ++polls > 1; }), CancelledError);
    CHECK(source->planesRead.load() >= 1);
    CHECK(source->planesRead.load() < 100);
    CHECK(model.volumeIfReady(0, 0) == nullptr);
    CHECK(model.mipIfReady(0, 0) == nullptr);

    // Uncancelled, the same read completes.
    CHECK(model.prepareVolumeSync(0, 0, [] { return false; }));
    CHECK(model.volumeIfReady(0, 0) != nullptr);
}

TEST_CASE("render: preparing a (c, t) outside the output throws", "[app][render]") {
    DisplayModel model;
    CHECK_THROWS_AS(model.prepareProjectionSync(0, 0), std::runtime_error);   // no output
    model.setOutput(twoChannelOutput());
    CHECK_THROWS_AS(model.prepareVolumeSync(2, 0), std::out_of_range);
    CHECK_THROWS_AS(model.prepareProjectionSync(0, 1), std::out_of_range);
    CHECK_THROWS_AS(model.prepareProjectionSync(-1, 0), std::out_of_range);
}

// --- encoding ---------------------------------------------------------------------------------

TEST_CASE("render: encodePng writes a well-formed PNG of the image's size", "[app][render][encode]") {
    const std::array<std::uint8_t, 8> signature{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    struct Case {
        int channels;
        int colourType;   // PNG: 0 grey, 2 RGB, 6 RGBA
    };
    for (const Case c : {Case{1, 0}, Case{3, 2}, Case{4, 6}}) {
        INFO(c.channels << " channels");
        const int w = 37, h = 11;
        const std::vector<std::uint8_t> px = gradient(w, h, c.channels);
        const std::vector<std::uint8_t> png = encodePng(px.data(), w, h, c.channels);
        REQUIRE(png.size() > 8 + 25 + 12);
        CHECK(std::equal(signature.begin(), signature.end(), png.begin()));
        const std::vector<Chunk> chunks = pngChunks(png);
        REQUIRE(chunks.size() >= 3);
        CHECK(chunks.front().type == "IHDR");
        CHECK(chunks.front().length == 13);
        CHECK(bigEndian32(png, chunks.front().data) == static_cast<std::uint32_t>(w));
        CHECK(bigEndian32(png, chunks.front().data + 4) == static_cast<std::uint32_t>(h));
        CHECK(png[chunks.front().data + 8] == 8);   // bits per sample
        CHECK(png[chunks.front().data + 9] == c.colourType);
        CHECK(chunks[1].type == "IDAT");
        CHECK(chunks.back().type == "IEND");
    }

    // A higher level never makes a flat picture larger.
    const std::vector<std::uint8_t> flat(64 * 64 * 3, 42);
    CHECK(encodePng(flat.data(), 64, 64, 3, 9).size() <= encodePng(flat.data(), 64, 64, 3, 1).size());
}

TEST_CASE("render: encodeJpeg writes a baseline JPEG of the image's size", "[app][render][encode]") {
    for (const int channels : {1, 3}) {
        INFO(channels << " channels");
        const int w = 40, h = 24;
        const std::vector<std::uint8_t> px = gradient(w, h, channels);
        const std::vector<std::uint8_t> jpg = encodeJpeg(px.data(), w, h, channels, 90);
        REQUIRE(jpg.size() > 4);
        CHECK(jpg[0] == 0xFF);
        CHECK(jpg[1] == 0xD8);   // SOI
        CHECK(jpg[jpg.size() - 2] == 0xFF);
        CHECK(jpg[jpg.size() - 1] == 0xD9);   // EOI
        // The segments up to SOF0 (FF C0: length, precision, height, width,
        // components); stb writes YCbCr, three components, even for grey.
        std::size_t at = 2;
        while (at + 4 <= jpg.size() && jpg[at] == 0xFF && jpg[at + 1] != 0xC0) at += 2 + static_cast<std::size_t>(bigEndian16(jpg, at + 2));
        REQUIRE(at + 10 <= jpg.size());
        REQUIRE(jpg[at] == 0xFF);
        REQUIRE(jpg[at + 1] == 0xC0);
        CHECK(jpg[at + 4] == 8);
        CHECK(bigEndian16(jpg, at + 5) == h);
        CHECK(bigEndian16(jpg, at + 7) == w);
        CHECK(jpg[at + 9] == 3);
    }
    const std::vector<std::uint8_t> px = gradient(32, 32, 3);
    CHECK(encodeJpeg(px.data(), 32, 32, 3, 30).size() < encodeJpeg(px.data(), 32, 32, 3, 100).size());
}

TEST_CASE("render: the encoders refuse what they cannot write", "[app][render][encode]") {
    const std::vector<std::uint8_t> px = gradient(4, 4, 4);
    CHECK_THROWS_AS(encodePng(nullptr, 4, 4, 3), std::invalid_argument);
    CHECK_THROWS_AS(encodePng(px.data(), 0, 4, 3), std::invalid_argument);
    CHECK_THROWS_AS(encodePng(px.data(), 4, -1, 3), std::invalid_argument);
    CHECK_THROWS_AS(encodePng(px.data(), 4, 4, 2), std::invalid_argument);
    CHECK_THROWS_AS(encodeJpeg(px.data(), 4, 4, 4), std::invalid_argument);
    CHECK_THROWS_AS(encodeJpeg(px.data(), 70000, 1, 1), std::invalid_argument);
    CHECK_THROWS_AS(encodeJpeg(nullptr, 4, 4, 1), std::invalid_argument);
    // 768 MiB of filtered rows: an int, but its zlib stream's capacity could
    // double past INT_MAX. Refused before a pixel is read.
    CHECK_THROWS_AS(encodePng(px.data(), 16384, 16384, 3), std::invalid_argument);
}

TEST_CASE("render: base64Encode gives the RFC 4648 vectors", "[app][render][encode]") {
    CHECK(base64Encode({}).empty());
    CHECK(base64Encode(bytesOf("f")) == "Zg==");
    CHECK(base64Encode(bytesOf("fo")) == "Zm8=");
    CHECK(base64Encode(bytesOf("foo")) == "Zm9v");
    CHECK(base64Encode(bytesOf("foob")) == "Zm9vYg==");
    CHECK(base64Encode(bytesOf("fooba")) == "Zm9vYmE=");
    CHECK(base64Encode(bytesOf("foobar")) == "Zm9vYmFy");
    // the last two letters of the standard alphabet, and a zero byte
    CHECK(base64Encode({0xFB, 0xFF}) == "+/8=");
    CHECK(base64Encode({0x00, 0x00, 0x00}) == "AAAA");
    CHECK(base64Encode({0xFF, 0xFF, 0xFF}) == "////");
}

TEST_CASE("render: writeBinaryFile creates the folders and takes a UTF-8 path", "[app][render][encode]") {
    TempDir dir;
    // "caf\xC3\xA9" is "cafe" with an e acute: UTF-8, whatever the code page.
    const std::string utf8 = dir.path.generic_u8string() + "/nested/caf\xC3\xA9/picture.png";
    const std::vector<std::uint8_t> bytes{0x89, 'P', 'N', 'G', 0x00, 0xFF};
    std::string error;
    REQUIRE(writeBinaryFile(utf8, bytes, &error));
    CHECK(error.empty());
    const fs::path written = fs::u8path(utf8);
    REQUIRE(fs::is_regular_file(written));
    std::ifstream in(written, std::ios::binary);
    const std::vector<std::uint8_t> back((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CHECK(back == bytes);

    // Written again, shorter: replaced, not merged.
    in.close();
    REQUIRE(writeBinaryFile(utf8, {1, 2}, &error));
    CHECK(fs::file_size(written) == 2);

    // A parent that is a file cannot become a folder.
    const std::string blocked = utf8 + "/inside.png";
    CHECK_FALSE(writeBinaryFile(blocked, bytes, &error));
    CHECK_FALSE(error.empty());
    CHECK_FALSE(writeBinaryFile("", bytes, &error));

    // Bytes that are not UTF-8 are a failure to report, never an exception,
    // on every platform: the message may go into JSON, so it must not carry
    // them, even where a later step (here a parent that is a file) fails.
    const std::string base = dir.path.generic_u8string();
    const std::vector<std::string> invalid{
        base + "/bad\xFF\xFE.png",                   // bytes that never occur in UTF-8
        base + "/overlong\xC0\xAF.png",              // an overlong "/"
        base + "/surrogate\xED\xA0\x80.png",         // a UTF-16 surrogate
        base + "/above\xF4\x90\x80\x80.png",         // above U+10FFFF
        base + "/cut\xE2\x82",                       // a sequence cut short
        blocked + "/deeper\xFF.png",                 // and a parent that is a file
    };
    for (const std::string& path : invalid) {
        CAPTURE(path.size());
        bool wrote = true;
        error.clear();
        REQUIRE_NOTHROW(wrote = writeBinaryFile(path, bytes, &error));
        CHECK_FALSE(wrote);
        CHECK_FALSE(error.empty());
        CHECK(std::all_of(error.begin(), error.end(), [](char ch) { return static_cast<unsigned char>(ch) < 0x80; }));
    }
}
