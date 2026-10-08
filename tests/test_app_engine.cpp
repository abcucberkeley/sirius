// SIRIUS's C++ engine (`sirius-cli serve`, core/engine_server.hpp) and what
// it stands on: the build identity (core/build_info), the serializer
// (core/serialize), the wire encoding of arrays (core/array_codec), the
// serving half of the worker protocol (core/rpc_server) and the C++ port of
// the worker's cluster datasets (core/dataset_service).
//
//   * the RemoteWorker client against rpc::Server: the handshake (a wrong
//     token, another protocol version, frames before it), jobs with progress,
//     cancel, busy, errors -- what the scripted stand-in workers of
//     test_app_rpc.cpp answer, now answered by the real C++ server;
//   * the application's RemoteSource / RemoteDatasets, unchanged, against
//     the engine in-process (rpc::loopbackPair) and as a real `sirius-cli
//     serve` process on 127.0.0.1;
//   * parity: the engine's dataset replies equal the Python worker's
//     (app/python/sirius_worker/datasets.py) on the same files. Needs a
//     Python with numpy ($SIRIUS_PYTHON, else one on PATH); TIFF parity also
//     needs the sirius package importable there. Skipped without them.
//
// No network host is contacted: everything is loopback.

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <nlohmann/json.hpp>

#include <sirius/device.hpp>
#include <sirius/tiff_io.hpp>

#include "core/array_codec.hpp"
#include "core/array_source.hpp"
#include "core/build_info.hpp"
#include "core/cancel.hpp"
#include "core/dataset_service.hpp"
#include "core/engine_server.hpp"
#include "core/errors.hpp"
#include "core/host.hpp"
#include "core/local_worker.hpp"
#include "core/manifest.hpp"
#include "core/operation.hpp"
#include "core/ops/builtin.hpp"
#include "core/ops/schema.hpp"
#include "core/process.hpp"
#include "core/remote_source.hpp"
#include "core/rpc.hpp"
#include "core/rpc_server.hpp"
#include "core/serialize.hpp"
#include "core/sha256.hpp"
#include "core/tracks.hpp"
#include "mrc_fixture.hpp"
#include "temp_path.hpp"

using namespace sirius;
using namespace sirius::app;
using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

    template <typename F> bool waitUntil(F done, std::chrono::milliseconds limit = std::chrono::seconds(30)) {
        const auto end = std::chrono::steady_clock::now() + limit;
        while (std::chrono::steady_clock::now() < end) {
            if (done()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return done();
    }

    struct TempDir {
        fs::path path;
        TempDir() : path(sirius::test::uniqueTempPath("engine", "")) { fs::create_directories(path); }
        ~TempDir() {
            std::error_code ec;
            fs::remove_all(path, ec);
        }
        std::string file(const std::string& name) const { return (path / name).generic_string(); }
    };

    // A .npy file of uint16 (c, t, z, y, x) whose voxel i (C order) is i % 997,
    // as tests/test_app_cluster.cpp's stand-in cluster makes one.
    void writeNpy(const std::string& path, const std::vector<Index>& shape) {
        std::string header = "{'descr': '<u2', 'fortran_order': False, 'shape': (";
        for (Index s : shape) header += std::to_string(s) + ", ";
        header += "), }";
        while ((10 + header.size() + 1) % 64 != 0) header += ' ';
        header += '\n';
        std::ofstream out(fs::u8path(path), std::ios::binary);
        out.write("\x93NUMPY\x01\x00", 8);
        const std::uint16_t len = static_cast<std::uint16_t>(header.size());
        out.put(static_cast<char>(len & 0xff));
        out.put(static_cast<char>(len >> 8));
        out << header;
        Index n = 1;
        for (Index s : shape) n *= s;
        for (Index i = 0; i < n; ++i) {
            const std::uint16_t v = static_cast<std::uint16_t>(i % 997);
            out.put(static_cast<char>(v & 0xff));
            out.put(static_cast<char>(v >> 8));
        }
    }

    // An ImageJ hyperstack TIFF of uint16 pages, c fastest, then z, then t,
    // tiled, with a pyramid of `levels` (SubIFDs); page p, pixel (y, x) is
    // (p * 131 + y * 7 + x * 3) % 4093.
    void writeHyperstack(const std::string& path, Index c, Index t, Index z, Index y, Index x, int levels) {
        const Index pages = c * t * z;
        Buffer<std::uint16_t> stack(Shape{pages, y, x});
        for (Index p = 0; p < pages; ++p)
            for (Index r = 0; r < y; ++r)
                for (Index col = 0; col < x; ++col) stack.data()[(p * y + r) * x + col] = static_cast<std::uint16_t>((p * 131 + r * 7 + col * 3) % 4093);
        TiffWriteOptions o;
        o.tiled = true;
        o.tileWidth = 32;
        o.tileHeight = 32;
        o.pyramidLevels = levels;
        o.downsample = 2;
        o.description = "ImageJ=1.53t\nimages=" + std::to_string(pages) + "\nchannels=" + std::to_string(c) + "\nslices=" + std::to_string(z) +
                        "\nframes=" + std::to_string(t) + "\nhyperstack=true\nmode=composite\nspacing=0.3\nunit=micron\n";
        o.xPixelUm = 0.1;
        o.yPixelUm = 0.1;
        writeTiffStack<std::uint16_t>(path, stack.view(), o);
    }

    std::uint16_t hyperstackValue(Index page, Index r, Index col) { return static_cast<std::uint16_t>((page * 131 + r * 7 + col * 3) % 4093); }

    // An engine served in-process: every connect() is a loopback pair whose
    // server end the engine serves on a thread of its own.
    struct LoopbackEngine {
        std::string token;
        EngineServer engine;
        std::mutex m;
        std::vector<std::thread> threads;

        explicit LoopbackEngine(EngineOptions o) : token(o.token), engine(std::move(o)) {}
        ~LoopbackEngine() {
            engine.stop();
            const std::lock_guard<std::mutex> g(m);
            for (std::thread& t : threads) t.join();
        }
        std::unique_ptr<RemoteWorker> connect(const std::string& asToken) {
            auto [client, server] = rpc::loopbackPair();
            {
                const std::lock_guard<std::mutex> g(m);
                threads.emplace_back([this, s = std::move(server)]() mutable { engine.serveConnection(std::move(s)); });
            }
            return std::make_unique<RemoteWorker>(std::move(client), asToken);
        }
        std::unique_ptr<RemoteWorker> connect() { return connect(token); }
    };

    EngineOptions quietEngine(const std::string& token = "s3cret") {
        EngineOptions o;
        o.token = token;
        o.pythonWorker = false;
        o.device = "cpu";
        return o;
    }

    // The array of a dataset_read / dataset_view reply, as the application decodes it.
    std::vector<float> decoded(const WorkerResult& r, std::vector<Index>& shape) {
        REQUIRE(r.tensors.size() == 1);
        return decodeWorkerArray(r.result, r.tensors.front(), shape);
    }

    std::string lowerOf(std::string s) {
        for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        return s;
    }

    std::string testPython() {
        std::string p = host::environment("SIRIUS_PYTHON");
        if (p.empty()) p = host::findPython();
        return p;
    }

    // The flat index of the cluster test's stack.npy voxel (c, t, z, y, x).
    std::uint16_t npyValue(const std::vector<Index>& s, Index c, Index t, Index z, Index y, Index x) {
        return static_cast<std::uint16_t>(((((c * s[1] + t) * s[2] + z) * s[3] + y) * s[4] + x) % 997);
    }

} // namespace

// --- the build -------------------------------------------------------------------------------

TEST_CASE("engine: the build identity names the commit, the op schema and the API", "[app][engine]") {
    const BuildInfo& b = buildInfo();
    CHECK_FALSE(b.version.empty());
    CHECK(b.build.rfind(b.version + "+", 0) == 0);
    CHECK_FALSE(b.commit.empty());
    CHECK(b.opsSchema.size() == 64);
    CHECK(b.api == kEngineApiVersion);
    const BuildInfo back = buildInfoFromJson(toJson(b));
    CHECK(back.build == b.build);
    CHECK(back.commit == b.commit);
    CHECK(back.dirty == b.dirty);
    CHECK(back.opsSchema == b.opsSchema);
    CHECK(back.api == b.api);

    SECTION("the hash is the committed snapshot's, and the snapshot is what the registry exports") {
        // Fails when an operation's parameters changed and op_schema.json was
        // not regenerated (tests/test_app_schema.cpp says how): an engine
        // built from such a tree would claim the old schema.
        registerBuiltinOperations();
        const std::string live = operationSchemas().dump(2) + "\n";
        INFO("regenerate bindings/python/sirius/op_schema.json: SIRIUS_OP_SCHEMA_OUT=... test_app_schema");
        CHECK(crypto::toHex(crypto::sha256(live)) == b.opsSchema);
    }
    SECTION("another commit with the same operations and API may serve") {
        BuildInfo other = b;
        other.commit = "0000000000000000000000000000000000000000";
        other.build = b.version + "+g0000000";
        CHECK(engineMismatch(b, other).empty());
        other.dirty = true;
        CHECK(engineMismatch(b, other).empty());
    }
    SECTION("other operations or another API are refused, with both builds named") {
        BuildInfo other = b;
        other.build = "0.1.0+g1a2b3c4";
        other.opsSchema = std::string(64, 'a');
        const std::string m = engineMismatch(b, other);
        CHECK_THAT(m, Catch::Matchers::ContainsSubstring("0.1.0+g1a2b3c4"));
        CHECK_THAT(m, Catch::Matchers::ContainsSubstring(b.build));
        CHECK_THAT(m, Catch::Matchers::ContainsSubstring("operations differ"));
        other = b;
        other.api = b.api + 1;
        CHECK_THAT(engineMismatch(b, other), Catch::Matchers::ContainsSubstring("engine API"));
        CHECK_FALSE(engineMismatch(b, buildInfoFromJson(json::object())).empty());
    }
}

// --- the serializer ----------------------------------------------------------------------------

TEST_CASE("engine: meta, reports, lineage and diagnostics round-trip through JSON and tensors", "[app][engine]") {
    SECTION("DatasetMeta") {
        DatasetMeta m;
        m.name = "cells";
        m.sourcePath = "cluster://fiona/data/cells.ome.tif";
        m.format = "ome-tiff";
        m.dims = Dims5{2, 3, 4, 50, 60};
        m.sourceType = PixelType::UInt16;
        m.bytesOnDisk = 123456789;
        m.voxelUm = {0.065, 0.066, 0.2};
        m.frameIntervalS = 2.5;
        m.channels = {{"GFP", 510.0, {0.1f, 0.9f, 0.2f}, "8 ms"}, {"RFP", 0.0, {1.f, 0.f, 1.f}, ""}};
        m.acquisition = "lattice";
        m.sim.present = true;
        m.sim.ndirs = 3;
        m.sim.nphases = 5;
        m.sim.fastSi = true;
        m.rgb = false;
        m.lightSheet = true;
        m.sheetAngleDeg = 31.8;
        m.tiles = {{"tile_0", {0, 1, 2}, {0, 0, 0}}, {"tile_1", {0, 1, 102}, {0, 0, 1}}};
        m.tileIndex = 1;
        const DatasetMeta r = datasetMetaFromJson(json::parse(toJson(m).dump()));
        CHECK(r.name == m.name);
        CHECK(r.sourcePath == m.sourcePath);
        CHECK(r.format == m.format);
        CHECK(r.dims.c == 2);
        CHECK(r.dims.x == 60);
        CHECK(r.sourceType == PixelType::UInt16);
        CHECK(r.bytesOnDisk == m.bytesOnDisk);
        CHECK(r.voxelUm == m.voxelUm);
        CHECK(r.frameIntervalS == 2.5);
        REQUIRE(r.channels.size() == 2);
        CHECK(r.channels[0].label == "GFP");
        CHECK(r.channels[0].wavelengthNm == 510.0);
        CHECK(r.channels[0].color == m.channels[0].color);
        CHECK(r.channels[0].exposure == "8 ms");
        CHECK(r.acquisition == "lattice");
        CHECK(r.sim.present);
        CHECK(r.sim.fastSi);
        CHECK(r.lightSheet);
        CHECK(r.sheetAngleDeg == 31.8);
        REQUIRE(r.tiles.size() == 2);
        CHECK(r.tiles[1].name == "tile_1");
        CHECK(r.tiles[1].positionUm[2] == 102.0);
        CHECK(r.tiles[1].gridIndex[2] == 1);
        CHECK(r.tileIndex == 1);
        CHECK_THROWS(datasetMetaFromJson(json{{"dims", {1, 2}}}));
    }
    SECTION("StepReport and lineage") {
        StepReport rep;
        rep.id = 77;
        rep.index = 3;
        rep.state = StepReport::State::Failed;
        rep.seconds = 1.25;
        rep.note = "n";
        rep.error = "out of memory";
        const StepReport r = stepReportFromJson(json::parse(toJson(rep).dump()));
        CHECK(r.id == 77);
        CHECK(r.index == 3);
        CHECK(r.state == StepReport::State::Failed);
        CHECK(r.seconds == 1.25);
        CHECK(r.error == "out of memory");
        CHECK_THROWS(stepReportFromJson(json{{"state", "exploded"}}));
        const Lineage l{{5u, 2u}, {6u, 2u}, {9u, 6u}};
        CHECK(lineageFromJson(json::parse(lineageToJson(l).dump())) == l);
    }
    SECTION("Diagnostics, images as tensors, through a frame") {
        Diagnostics d;
        d.kind = DiagnosticsKind::Deconvolve;
        d.summary = "mean over t";
        d.footer = "Wiener 0.001";
        DiagnosticImage img;
        img.title = "Raw FFT";
        img.meta = "0\xC2\xB0";
        img.rows = 3;
        img.cols = 4;
        img.values = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        img.logScale = true;
        img.marks = {{DiagnosticMark::Kind::Ring, 1.5, 2.0, 3.0, false, "k0"}};
        d.addImage(img);
        img.title = "second";
        img.rows = 1;
        img.cols = 2;
        img.values = {-1.f, 0.5f};
        img.marks.clear();
        d.addImage(img);
        d.tabs = {{"Spectra", {0, 1}}};
        d.table = DiagnosticTable{"Estimated", {"a", "b"}, {{"1", "2"}, {"3", "4"}}, {{1, 0}}};
        d.curves = {{"Convergence", {1, 2, 3}, {0.5, std::nan(""), 0.1}, 2.0, "l", "m", "r", true}};
        DiagnosticHistogram h;
        h.channel = "GFP";
        h.bins = {1, 5, 2};
        h.binLo = 10;
        h.binHi = 900;
        h.gamma = 0.8;
        d.histograms = {h};
        d.facts = {{"Method", "Ray casting"}};
        d.alignment = AlignmentInfo{2, 3, {"a", "b", "c", "d", "e", "f"}, 4, {{"Mean", "2 px"}}};
        d.warnings = {"careful"};

        const EncodedDiagnostics e = encodeDiagnostics(d, "s3.");
        REQUIRE(e.tensors.size() == 2);
        CHECK(e.tensors[0].name == "s3.img0");
        std::vector<rpc::TensorRef> refs;
        for (const rpc::Tensor& t : e.tensors) refs.push_back({t.name, t.dtype, t.shape, t.bytes.data(), t.bytes.size()});
        std::vector<std::byte> frame = rpc::encodeFrame({{"id", 1}, {"type", "result"}, {"result", {{"diagnostics", e.json}}}}, refs);
        const auto msg = rpc::decodeFrame(frame);
        REQUIRE(msg);
        const Diagnostics r = decodeDiagnostics(msg->header["result"]["diagnostics"], msg->tensors);
        CHECK(r.kind == DiagnosticsKind::Deconvolve);
        CHECK(r.summary == d.summary);
        CHECK(r.footer == d.footer);
        REQUIRE(r.images.size() == 2);
        CHECK(r.images[0].values == d.images[0].values);
        CHECK(r.images[0].rows == 3);
        CHECK(r.images[0].logScale);
        CHECK(r.images[0].meta == d.images[0].meta);
        REQUIRE(r.images[0].marks.size() == 1);
        CHECK(r.images[0].marks[0].kind == DiagnosticMark::Kind::Ring);
        CHECK(r.images[0].marks[0].text == "k0");
        CHECK_FALSE(r.images[0].marks[0].accent);
        CHECK(r.images[1].values == d.images[1].values);
        REQUIRE(r.tabs.size() == 1);
        CHECK(r.tabs[0].images == std::vector<int>{0, 1});
        REQUIRE(r.table);
        CHECK(r.table->rows == d.table->rows);
        CHECK(r.table->accentCells == d.table->accentCells);
        REQUIRE(r.curves.size() == 1);
        CHECK(std::isnan(r.curves[0].y[1]));   // NaN travels as null
        CHECK(r.curves[0].y[2] == 0.1);
        CHECK(r.curves[0].stopX == 2.0);
        CHECK(r.curves[0].rightLabel == "r");
        CHECK(r.curves[0].logY);
        REQUIRE(r.histograms.size() == 1);
        CHECK(r.histograms[0].bins == h.bins);
        CHECK(r.histograms[0].gamma == 0.8);
        CHECK(r.facts.size() == 1);
        REQUIRE(r.alignment);
        CHECK(r.alignment->gridCols == 3);
        CHECK(r.alignment->highlightedTile == 4);
        CHECK(r.warnings == d.warnings);
        // an image whose values did not come along is an error, not an empty picture
        CHECK_THROWS_AS(decodeDiagnostics(e.json, {}), ProtocolError);
    }
}

// --- arrays on the wire -------------------------------------------------------------------------------

TEST_CASE("engine: arrays go compressed with shuffled bytes and decode as the application decodes them", "[app][engine]") {
    std::vector<std::uint16_t> ramp(64 * 80);
    for (std::size_t i = 0; i < ramp.size(); ++i) ramp[i] = static_cast<std::uint16_t>(i % 997);
    const auto bytesOf = [](const std::vector<std::uint16_t>& v) {
        std::vector<std::byte> b(v.size() * 2);
        std::memcpy(b.data(), v.data(), b.size());
        return b;
    };
    CHECK(codec::availableEncodings().back() == "zlib");

    const codec::Encoded z = codec::encodeArray("uint16", {64, 80}, bytesOf(ramp), {"zlib"});
    CHECK(z.description["encoding"] == "zlib");
    CHECK(z.description["shuffle"] == true);
    CHECK(z.description["raw_bytes"] == ramp.size() * 2);
    CHECK(z.tensor.dtype == "uint8");
    CHECK(z.tensor.bytes.size() < ramp.size() * 2);
    std::vector<Index> shape;
    const std::vector<float> back = decodeWorkerArray(z.description, z.tensor, shape);
    CHECK(shape == std::vector<Index>{64, 80});
    for (std::size_t i = 0; i < ramp.size(); ++i) REQUIRE(back[i] == static_cast<float>(ramp[i]));

    if (codec::availableEncodings().front() == "zstd") {
        const codec::Encoded zs = codec::encodeArray("uint16", {64, 80}, bytesOf(ramp), {"zstd", "zlib"});
        CHECK(zs.description["encoding"] == "zstd");
        CHECK(zs.description["shuffle"] == true);
        const std::vector<float> zback = decodeWorkerArray(zs.description, zs.tensor, shape);
        for (std::size_t i = 0; i < ramp.size(); ++i) REQUIRE(zback[i] == static_cast<float>(ramp[i]));
        // a frame that states another size than the shape is refused before anything is written
        json lie = zs.description;
        lie["shape"] = {64, 81};
        CHECK_THROWS_AS(decodeWorkerArray(lie, zs.tensor, shape), ProtocolError);
        // only zlib asked for: zlib
        CHECK(codec::encodeArray("uint16", {64, 80}, bytesOf(ramp), {"zlib"}).description["encoding"] == "zlib");
    }
    // nothing accepted, too small to bother, or not smaller: the array itself
    const codec::Encoded raw = codec::encodeArray("uint16", {64, 80}, bytesOf(ramp), {});
    CHECK(raw.description["encoding"] == "raw");
    CHECK(raw.tensor.dtype == "uint16");
    CHECK(raw.tensor.shape == std::vector<Index>{64, 80});
    const codec::Encoded small = codec::encodeArray("uint16", {4, 4}, bytesOf(std::vector<std::uint16_t>(16, 7)), {"zlib"});
    CHECK(small.description["encoding"] == "raw");
    std::vector<std::uint16_t> noise(4096);
    std::uint32_t s = 12345;
    for (auto& v : noise) {
        s = s * 1664525u + 1013904223u;
        v = static_cast<std::uint16_t>(s >> 16);
    }
    const codec::Encoded n = codec::encodeArray("uint16", {4096}, bytesOf(noise), {"zlib"});
    CHECK(n.description["encoding"] == "raw");
    CHECK_THROWS(codec::encodeArray("uint16", {3}, std::vector<std::byte>(5), {}));
}

TEST_CASE("engine: the block mean is datasets.py's reduce_blocks", "[app][engine]") {
    // golden values from app/python/sirius_worker/datasets.py
    const auto array = [](const std::string& dtype, std::vector<Index> shape, const auto& values) {
        HostArray a;
        a.dtype = dtype;
        a.shape = std::move(shape);
        using T = typename std::decay_t<decltype(values)>::value_type;
        a.bytes.resize(values.size() * sizeof(T));
        std::memcpy(a.bytes.data(), values.data(), a.bytes.size());
        return a;
    };
    const auto values = [](const HostArray& a, auto tag) {
        using T = decltype(tag);
        std::vector<double> out(a.elements());
        for (std::size_t i = 0; i < out.size(); ++i) {
            T v;
            std::memcpy(&v, a.bytes.data() + i * sizeof(T), sizeof(T));
            out[i] = static_cast<double>(v);
        }
        return out;
    };
    std::vector<std::uint16_t> u(35);
    for (std::size_t i = 0; i < u.size(); ++i) u[i] = static_cast<std::uint16_t>(i * 977 % 65536);
    const HostArray a = array("uint16", {5, 7}, u);
    HostArray r = reduceBlocks(a, {2, 2});
    CHECK(r.dtype == "uint16");
    CHECK(r.shape == std::vector<Index>{3, 4});
    CHECK(values(r, std::uint16_t{}) == std::vector<double>{3908, 5862, 7816, 9282, 17586, 19540, 21494, 22960, 27844, 29798, 31752, 33218});
    r = reduceBlocks(a, {3, 2});
    CHECK(values(r, std::uint16_t{}) == std::vector<double>{7328, 9282, 11236, 12701, 24425, 26379, 28333, 29798});
    // rounding half to even, negatives too
    r = reduceBlocks(array("int16", {2, 5}, std::vector<std::int16_t>{-3, -2, -1, 0, 1, 2, 3, 4, 5, -6}), {2, 2});
    CHECK(values(r, std::int16_t{}) == std::vector<double>{0, 2, -2});
    r = reduceBlocks(array("uint8", {1, 6}, std::vector<std::uint8_t>{0, 1, 1, 2, 2, 3}), {1, 2});
    CHECK(values(r, std::uint8_t{}) == std::vector<double>{0, 2, 2});
    r = reduceBlocks(array("float32", {2, 3}, std::vector<float>{0.5f, 1.25f, 2.0f, 3.5f, -4.0f, 7.75f}), {2, 2});
    CHECK(r.dtype == "float32");
    CHECK(values(r, float{}) == std::vector<double>{0.3125, 4.875});
    std::vector<std::uint8_t> v(60);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<std::uint8_t>(i * 31 % 251);
    r = reduceBlocks(array("uint8", {3, 4, 5}, v), {2, 3, 2});
    CHECK(r.shape == std::vector<Index>{2, 2, 3});
    CHECK(values(r, std::uint8_t{}) == std::vector<double>{104, 145, 129, 163, 100, 146, 114, 134, 97, 214, 26, 72});
    // factors of 1: the array itself
    CHECK(reduceBlocks(a, {1, 1}).bytes == a.bytes);
}

// --- the server, against the client the application uses ------------------------------------------------

namespace {
    // An rpc::Server with the stand-in worker's methods of test_app_rpc.cpp.
    struct TestServer {
        rpc::Server server;
        std::vector<std::thread> threads;
        std::mutex m;
        std::atomic<int> started{0};

        explicit TestServer(rpc::Server::Options o) : server(std::move(o)) {
            server.setCapabilities([] {
                return json{{"version", "test"}, {"methods", {"run:double", "run:slow", "model_info"}}, {"cuda", false}, {"device", "cpu"}, {"hostname", "loop"}};
            });
            server.handle(
                "run",
                [this](const rpc::Request& req, rpc::CallContext& ctx) {
                    ++started;
                    const std::string kind = req.params.value("kind", "");
                    if (kind == "fail") throw std::runtime_error("kaboom");
                    if (kind == "slow") {
                        while (!ctx.cancelled()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
                        throw CancelledError();
                    }
                    if (req.tensors.size() != 1) throw std::runtime_error("one tensor expected");
                    const rpc::Tensor& in = req.tensors[0];
                    for (int i = 0; i < 3; ++i) ctx.progress((i + 1) / 3.0, "tile " + std::to_string(i), {{"step", i}});
                    rpc::Reply r;
                    r.result = {{"channels", 1}};
                    rpc::Tensor out = in;
                    out.name = "prob";
                    float* p = reinterpret_cast<float*>(out.bytes.data());
                    for (Index i = 0; i < in.numel(); ++i) p[i] *= 2.0f;
                    r.tensors.push_back(std::move(out));
                    return r;
                },
                rpc::Dispatch::Job);
            server.handle("model_info", [](const rpc::Request& req, rpc::CallContext&) {
                rpc::Reply r;
                r.result = {{"spec", req.params.value("spec", "")}, {"peer", req.peer}};
                return r;
            });
        }
        ~TestServer() {
            server.stop();
            const std::lock_guard<std::mutex> g(m);
            for (auto& t : threads) t.join();
        }
        std::unique_ptr<rpc::Transport> raw() {
            auto [client, srv] = rpc::loopbackPair();
            const std::lock_guard<std::mutex> g(m);
            threads.emplace_back([this, s = std::move(srv)]() mutable { server.serveConnection(std::move(s)); });
            return std::move(client);
        }
        std::unique_ptr<RemoteWorker> client(const std::string& token) { return std::make_unique<RemoteWorker>(raw(), token); }
    };

    rpc::Server::Options withToken(const std::string& token, int maxClients = 4) {
        rpc::Server::Options o;
        o.token = token;
        o.maxClients = maxClients;
        o.preauthTimeout = std::chrono::milliseconds(1500);
        return o;
    }

    // One request on a raw transport, its first answer.
    json exchange(rpc::Transport& t, const json& header, std::vector<std::byte>& inbox) {
        t.send(rpc::encodeFrame(header, {}));
        for (int i = 0; i < 400; ++i) {
            if (auto m = rpc::decodeFrame(inbox)) return m->header;
            t.receive(inbox, std::chrono::milliseconds(25));
        }
        FAIL("no answer");
        return {};
    }
} // namespace

TEST_CASE("engine: RemoteWorker talks to the C++ server as to the Python worker", "[app][engine][rpc]") {
    TestServer ts(withToken("tok"));

    SECTION("the handshake, capabilities and a request") {
        auto w = ts.client("tok");
        CHECK(w->capabilities().protocolVersion == rpc::kProtocolVersion);
        CHECK(w->capabilities().version == "test");
        CHECK(w->supports("double"));
        const WorkerResult r = w->call("model_info", {{"spec", "x.pt"}});
        CHECK(r.result["spec"] == "x.pt");
        CHECK(r.result["peer"] == "loopback");
        CHECK(w->call("ping", json::object()).result.contains("time"));
    }
    SECTION("a client that cannot prove the token is refused") {
        CHECK_THROWS_WITH(ts.client("wrong"), Catch::Matchers::ContainsSubstring("authentication failed"));
    }
    SECTION("a job streams progress and returns tensors") {
        auto w = ts.client("tok");
        std::vector<float> in{1.f, 2.f, 3.f, 4.f};
        std::vector<double> fractions;
        const WorkerResult r = w->call("run", {{"kind", "double"}}, {{"input", "float32", {2, 2}, in.data(), in.size() * sizeof(float)}},
                                       [&](double f, const std::string&) { fractions.push_back(f); });
        CHECK(fractions.size() == 3);
        CHECK(r.result["channels"] == 1);
        CHECK(r.result.contains("seconds"));
        REQUIRE(r.tensors.size() == 1);
        CHECK(r.tensors[0].name == "prob");
        CHECK(r.tensors[0].asFloat32()[3] == 8.f);
    }
    SECTION("a failing job reports its message") {
        auto w = ts.client("tok");
        CHECK_THROWS_WITH(w->call("run", {{"kind", "fail"}}), "worker: kaboom");
        CHECK_THROWS_WITH(w->call("nope", json::object()), "worker: unknown method 'nope'");
        CHECK(w->isOpen());   // the connection survives an error
    }
    SECTION("cancel reaches the job; one job at a time across connections") {
        auto a = ts.client("tok");
        auto b = ts.client("tok");
        std::atomic<bool> cancel{false};
        std::thread t([&] {
            try {
                a->call("run", {{"kind", "slow"}}, {}, {}, [&] { return cancel.load(); });
                FAIL("the slow job finished");
            } catch (const CancelledError&) {
            }
        });
        REQUIRE(waitUntil([&] { return ts.started.load() == 1; }));
        CHECK_THROWS_WITH(b->call("run", {{"kind", "slow"}}), Catch::Matchers::ContainsSubstring("busy: request"));
        cancel = true;
        t.join();
        CHECK(a->isOpen());
        std::vector<float> in{1.f};
        CHECK(b->call("run", {{"kind", "double"}}, {{"input", "float32", {1}, in.data(), sizeof(float)}}).tensors.size() == 1);
    }
    SECTION("a connection that goes away cancels its job") {
        auto a = ts.client("tok");
        std::thread t([&] {
            try {
                a->call("run", {{"kind", "slow"}});
            } catch (const std::exception&) {
            }
        });
        REQUIRE(waitUntil([&] { return ts.started.load() == 1; }));
        a->close();
        t.join();
        auto b = ts.client("tok");
        std::vector<float> in{1.f};
        // the slot is free again once the cancelled job has ended
        CHECK(waitUntil([&] {
            try {
                b->call("run", {{"kind", "double"}}, {{"input", "float32", {1}, in.data(), sizeof(float)}});
                return true;
            } catch (const std::exception&) {
                return false;
            }
        }));
    }
    SECTION("frames before the handshake, another version, an oversize frame") {
        std::vector<std::byte> inbox;
        auto t = ts.raw();
        json h = exchange(*t, {{"id", 1}, {"type", "request"}, {"method", "model_info"}, {"params", json::object()}}, inbox);
        CHECK(h["type"] == "error");
        CHECK(h["message"] == "not authenticated: complete the handshake ('hello', then 'auth') first");
        h = exchange(*t, {{"id", 2}, {"type", "request"}, {"method", "auth"}, {"params", {{"client_proof", "00"}}}}, inbox);
        CHECK(h["message"] == "not authenticated: send 'hello' first");

        inbox.clear();
        auto t2 = ts.raw();
        h = exchange(*t2, {{"id", 1}, {"type", "request"}, {"method", "hello"}, {"params", {{"protocol_version", 1}, {"client_nonce", rpc::randomNonce()}}}}, inbox);
        CHECK(h["message"] == "protocol version mismatch: this worker speaks version 2, the client speaks version 1; update the SIRIUS "
                              "application that connects to this worker");

        inbox.clear();
        auto t3 = ts.raw();
        h = exchange(*t3, {{"id", 1}, {"type", "request"}, {"method", "hello"}, {"params", {{"pad", std::string(20000, 'x')}}}}, inbox);
        CHECK(h["id"].is_null());
        CHECK_THAT(h["message"].get<std::string>(), Catch::Matchers::StartsWith("protocol error: header length"));
    }
    SECTION("one client at a time: the next is served once the one before has gone") {
        TestServer one(withToken("tok", 1));
        auto a = one.client("tok");
        std::atomic<bool> connected{false};
        std::unique_ptr<RemoteWorker> b;
        std::thread t([&] {
            b = one.client("tok");
            connected = true;
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        CHECK_FALSE(connected.load());
        a->close();
        t.join();
        CHECK(connected.load());
        CHECK(b->call("ping", json::object()).result.contains("time"));
    }
    SECTION("over TCP, and shutdown") {
        rpc::Listener l("127.0.0.1", 0);
        CHECK(l.port() > 0);
        std::thread srv([&] { ts.server.serve(l); });
        auto w = RemoteWorker::connect("127.0.0.1", l.port(), "tok");
        CHECK(w->call("model_info", {{"spec", "s"}}).result["peer"].get<std::string>().rfind("127.0.0.1:", 0) == 0);
        CHECK_THROWS_WITH(RemoteWorker::connect("127.0.0.1", l.port(), "nope"), Catch::Matchers::ContainsSubstring("authentication failed"));
        w->call("shutdown", json::object());
        srv.join();
        CHECK(ts.server.stopping());
    }
    CHECK(rpc::isLoopbackHost("127.0.0.1"));
    CHECK(rpc::isLoopbackHost("localhost"));
    CHECK(rpc::isLoopbackHost("::1"));
    CHECK_FALSE(rpc::isLoopbackHost("0.0.0.0"));
    CHECK_FALSE(rpc::isLoopbackHost(""));
    CHECK_FALSE(rpc::isLoopbackHost("fiona"));
}

// --- the engine's datasets, through the application's RemoteSource -------------------------------------

TEST_CASE("engine: RemoteSource draws a .npy dataset served by the engine in-process", "[app][engine]") {
    TempDir dir;
    const std::vector<Index> shape{1, 2, 4, 64, 80};
    const std::string path = dir.file("stack.npy");
    writeNpy(path, shape);
    LoopbackEngine le(quietEngine());
    {
        auto caps = le.connect();
        CHECK(caps->capabilities().engine.is_object());
        CHECK(caps->capabilities().engine["build"] == buildInfo().build);
        CHECK(caps->capabilities().engine["python"]["state"] == "disabled");
        CHECK(caps->capabilities().encodings == codec::availableEncodings());
        CHECK_FALSE(caps->capabilities().tiffReader.empty());
        // without a Python worker, what only it serves is refused, and says why
        CHECK_THROWS_WITH(caps->call("list_plugins", json::object()), Catch::Matchers::ContainsSubstring("--no-python-worker"));
    }
    // the cluster test's checks (tests/test_app_cluster.cpp), against the engine
    auto datasets = std::make_shared<RemoteDatasets>("enginehost", [&] { return le.connect(); });
    datasets->install();
    const std::string name = makeClusterPath("enginehost", path);
    const DatasetMeta meta = probeDataset(name);
    CHECK(meta.dims.c == 1);
    CHECK(meta.dims.t == 2);
    CHECK(meta.dims.z == 4);
    CHECK(meta.dims.y == 64);
    CHECK(meta.dims.x == 80);
    CHECK(meta.sourceType == PixelType::UInt16);
    CHECK(meta.format == "cluster npy");
    OpenResult opened = openDataset(name);
    REQUIRE(opened.source);
    ViewProvider* views = opened.source->viewProvider();
    REQUIRE(views);
    auto* remote = dynamic_cast<RemoteSource*>(opened.source.get());
    REQUIRE(remote);
    ViewRequest req;
    req.kind = ViewRequest::Kind::XY;
    req.t = 1;
    req.index = 2;
    req.factor = 2;
    bool exact = true;
    CHECK_FALSE(views->view(req, exact));   // nothing yet: queued
    remote->waitIdle();
    auto tile = views->view(req, exact);
    INFO(views->lastError());
    REQUIRE(tile);
    CHECK(exact);
    CHECK(tile->w == 40);
    CHECK(tile->h == 32);
    const long long base = (1LL * 4 + 2) * 64 * 80;
    const double mean = ((base % 997) + ((base + 1) % 997) + ((base + 80) % 997) + ((base + 81) % 997)) / 4.0;
    CHECK(std::abs(tile->data[0] - static_cast<float>(mean)) <= 0.5f);
    ViewRequest next = req;
    next.index = 3;
    remote->waitIdle();
    views->view(next, exact);
    CHECK(exact);   // prefetched behind the visible plane
    std::vector<float> plane(64 * 80);
    opened.source->readPlane(0, 1, 2, plane.data());
    CHECK(plane[81] == static_cast<float>((base + 81) % 997));
    CHECK(remote->inputReference(0, 1).value("path", std::string()) == path);

    // the other panes: re-slices, the projection, the volume, a window
    for (const auto kind : {ViewRequest::Kind::XZ, ViewRequest::Kind::YZ, ViewRequest::Kind::MIP, ViewRequest::Kind::Volume}) {
        ViewRequest r;
        r.kind = kind;
        r.t = 1;
        r.index = 5;
        r.factor = 1;
        r.maxSide = 16;
        views->view(r, exact);
        remote->waitIdle();
        auto v = views->view(r, exact);
        INFO(views->lastError());
        REQUIRE(v);
        if (kind == ViewRequest::Kind::XZ) {
            CHECK((v->w == 80 && v->h == 4));
            CHECK(v->data[2 * 80 + 7] == npyValue(shape, 0, 1, 2, 5, 7));
        } else if (kind == ViewRequest::Kind::YZ) {
            CHECK((v->w == 4 && v->h == 64));
            CHECK(v->data[9 * 4 + 3] == npyValue(shape, 0, 1, 3, 9, 5));
        } else if (kind == ViewRequest::Kind::MIP) {
            float m = 0;
            for (Index z = 0; z < 4; ++z) m = std::max(m, static_cast<float>(npyValue(shape, 0, 1, z, 10, 11)));
            CHECK(v->data[10 * 80 + 11] == m);
        } else {
            CHECK((v->d == 4 && v->h == 16 && v->w == 16));   // 4 x 64 x 80 to a longest side of 16: factors 1, 4, 5
        }
    }
    CHECK_FALSE(views->window(0, 1, false));
    remote->waitIdle();
    const auto w = views->window(0, 1, true);
    REQUIRE(w);
    CHECK(w->first == 0.f);
    CHECK(w->second == 996.f);
    // a request outside the dataset says where, as the worker says it
    try {
        datasets->call(RemoteDatasets::Lane::Reads, "dataset_read", {{"path", path}, {"c", 3}, {"t", 0}});
        FAIL("read outside the dataset");
    } catch (const std::exception& e) {
        CHECK_THAT(e.what(), Catch::Matchers::ContainsSubstring("DatasetError: c 3, t 0 is outside the dataset (1 channels, 2 time points, 4 planes)"));
    }
    const TransferStats ts = datasets->stats();
    CHECK(ts.requests >= 8);
    CHECK(ts.wireBytes < ts.rawBytes);   // compressed on the wire
    datasets->uninstall();
    opened.source.reset();
}

TEST_CASE("engine: a TIFF hyperstack with a pyramid, read as openDataset shapes it", "[app][engine]") {
    TempDir dir;
    const std::string path = dir.file("hyper.tif");
    writeHyperstack(path, 2, 3, 4, 96, 128, 3);
    // what the application's Load step makes of the file
    const DatasetMeta local = probeDataset(path);
    REQUIRE(local.dims.c == 2);
    REQUIRE(local.dims.t == 3);
    REQUIRE(local.dims.z == 4);
    LoopbackEngine le(quietEngine());
    auto w = le.connect();
    const json info = w->call("dataset_info", {{"path", path}}).result;
    CHECK(info["dims"] == json({2, 3, 4, 96, 128}));
    CHECK(info["dtype"] == "uint16");
    CHECK(info["format"] == "imagej-tiff");
    CHECK(info["dims_from_metadata"] == true);
    CHECK(std::abs(info["voxel_um"][0].get<double>() - 0.1) < 1e-6);
    CHECK(std::abs(info["voxel_um"][2].get<double>() - 0.3) < 1e-6);
    CHECK(info["name"] == "hyper");
    CHECK(info["encodings"] == codec::availableEncodings());

    // the page of (c, t, z): c fastest, then z, then t (ImageJ's order)
    const auto page = [](Index c, Index t, Index z) { return (t * 4 + z) * 2 + c; };
    std::vector<Index> shape;
    std::vector<float> p = decoded(w->call("dataset_read", {{"path", path}, {"c", 1}, {"t", 2}, {"z", 3}, {"accept", {"zlib"}}}), shape);
    CHECK(shape == std::vector<Index>{96, 128});
    CHECK(p[17 * 128 + 33] == hyperstackValue(page(1, 2, 3), 17, 33));
    std::vector<float> vol = decoded(w->call("dataset_read", {{"path", path}, {"c", 1}, {"t", 2}}), shape);
    CHECK(shape == std::vector<Index>{4, 96, 128});
    CHECK(vol[(3 * 96 + 17) * 128 + 33] == hyperstackValue(page(1, 2, 3), 17, 33));
    CHECK(vol[(0 * 96 + 5) * 128 + 6] == hyperstackValue(page(1, 2, 0), 5, 6));

    // a region of a plane at full resolution, read without the rest
    std::vector<float> region =
        decoded(w->call("dataset_view", {{"path", path}, {"kind", "xy"}, {"c", 0}, {"t", 1}, {"index", 2}, {"factor", 1}, {"region", {40, 20, 30, 10}}}), shape);
    CHECK(shape == std::vector<Index>{10, 30});
    CHECK(region[3 * 30 + 4] == hyperstackValue(page(0, 1, 2), 23, 44));
    // reduced by 2 and 4: from the pyramid's levels
    for (int f : {2, 4}) {
        std::vector<float> v = decoded(w->call("dataset_view", {{"path", path}, {"kind", "xy"}, {"c", 0}, {"t", 1}, {"index", 2}, {"factor", f}}), shape);
        CHECK(shape == std::vector<Index>{96 / f, 128 / f});
        // the writer's box filter: within a grey value of the block mean
        double sum = 0;
        for (int r = 0; r < f; ++r)
            for (int c = 0; c < f; ++c) sum += hyperstackValue(page(0, 1, 2), f * 5 + r, f * 7 + c);
        CHECK(std::abs(v[5 * (128 / f) + 7] - sum / (f * f)) <= 1.0);
    }
    // a reduced region not on the level's grid: the full-resolution pixels' block mean
    std::vector<float> odd =
        decoded(w->call("dataset_view", {{"path", path}, {"kind", "xy"}, {"c", 0}, {"t", 0}, {"index", 0}, {"factor", 2}, {"region", {3, 5, 20, 20}}}), shape);
    CHECK(shape == std::vector<Index>{10, 10});
    const double m = (hyperstackValue(0, 5, 3) + hyperstackValue(0, 5, 4) + hyperstackValue(0, 6, 3) + hyperstackValue(0, 6, 4)) / 4.0;
    CHECK(std::abs(odd[0] - m) <= 0.5);
    CHECK_THROWS_WITH(w->call("dataset_view", {{"path", path}, {"kind", "xy"}, {"region", {500, 500, 10, 10}}}),
                      Catch::Matchers::ContainsSubstring("region [500, 500, 10, 10] is outside the 128 x 96 view"));
    CHECK_THROWS_WITH(w->call("dataset_view", {{"path", path}, {"kind", "diagonal"}}), Catch::Matchers::ContainsSubstring("unknown view 'diagonal'"));
    CHECK_THROWS_WITH(w->call("dataset_info", {{"path", dir.file("missing.tif")}}), Catch::Matchers::ContainsSubstring("no such file on"));
    const json stats = w->call("dataset_stats", {{"path", path}, {"c", 0}, {"t", 0}}).result;
    CHECK(stats["min"].get<double>() >= 0.0);
    CHECK(stats["max"].get<double>() <= 4092.0);
    CHECK(stats["lo"].get<double>() < stats["hi"].get<double>());
    // the volume read above is kept for the re-slices
    CHECK(le.engine.datasets().cachedBytes() >= 4u * 96 * 128 * 2);

    // a page order given by the Load step wins over the metadata
    const json given = w->call("dataset_info", {{"path", path}, {"options", {{"page_order", "czt"}, {"c", 1}, {"t", 1}, {"z", 24}}}}).result;
    CHECK(given["dims"] == json({1, 1, 24, 96, 128}));
}

TEST_CASE("engine: a DeltaVision stack, read as openDataset shapes it", "[app][engine][mrc]") {
    const std::string path = std::string(SIRIUS_TEST_DATA_DIR) + "/raw.dv";
    LoopbackEngine le(quietEngine());
    auto w = le.connect();
    const json info = w->call("dataset_info", {{"path", path}}).result;
    CHECK(info["dims"] == json({1, 1, 135, 64, 64}));
    CHECK(info["dtype"] == "float32");
    CHECK(info["format"] == "deltavision");
    CHECK(info["dims_from_metadata"] == true);
    CHECK(info["name"] == "raw");
    CHECK(info["bytes"] == 2212864);
    CHECK(info["rgb"] == false);
    CHECK(std::abs(info["voxel_um"][0].get<double>() - 0.08) < 1e-6);
    CHECK(std::abs(info["voxel_um"][2].get<double>() - 0.125) < 1e-6);
    json channels = json::array();
    channels.push_back({{"name", "528"}, {"wavelength_nm", 528.0}});
    CHECK(info["channels"] == channels);
    // the sections as the library reads them, in file order; raw.tif (Bio-Formats'
    // export of raw.dv) holds the same values with every row reversed
    const ImageStack<float> tif = readTiffStack<float>(std::string(SIRIUS_TEST_DATA_DIR) + "/raw.tif");
    std::vector<Index> shape;
    std::vector<float> p = decoded(w->call("dataset_read", {{"path", path}, {"c", 0}, {"t", 0}, {"z", 7}, {"accept", {"zlib"}}}), shape);
    CHECK(shape == std::vector<Index>{64, 64});
    CHECK(p[17 * 64 + 33] == tif(7, 63 - 17, 33));
    std::vector<float> vol = decoded(w->call("dataset_read", {{"path", path}, {"c", 0}, {"t", 0}}), shape);
    CHECK(shape == std::vector<Index>{135, 64, 64});
    CHECK(vol[(134 * 64 + 5) * 64 + 6] == tif(134, 63 - 5, 6));
    CHECK(vol[(7 * 64 + 17) * 64 + 33] == tif(7, 63 - 17, 33));
    std::vector<float> region =
        decoded(w->call("dataset_view", {{"path", path}, {"kind", "xy"}, {"c", 0}, {"t", 0}, {"index", 2}, {"factor", 1}, {"region", {40, 20, 20, 10}}}), shape);
    CHECK(shape == std::vector<Index>{10, 20});
    CHECK(region[3 * 20 + 4] == tif(2, 63 - 23, 44));
    const json stats = w->call("dataset_stats", {{"path", path}, {"c", 0}, {"t", 0}}).result;
    CHECK(stats["min"].get<double>() >= 1.0e-5);
    CHECK(stats["max"].get<double>() <= 0.009);

    // several wavelengths and time points in the WZT sequence: (c, t, z) picks the right section
    TempDir dir;
    test::MrcSpec s;
    s.waves = 2;
    s.times = 3;
    s.planes = 4;
    s.sequence = 1;
    s.mode = 6;
    s.wavelengths = {488, 561, 0, 0, 0};
    const std::string waves = dir.file("waves.dv");
    const auto stamp = [](int w, int t, int z, int y, int x) { return static_cast<double>(w * 10000 + t * 1000 + z * 100 + y * 10 + x); };
    test::writeMrc(waves, s, stamp);
    const json wi = w->call("dataset_info", {{"path", waves}}).result;
    CHECK(wi["dims"] == json({2, 3, 4, 6, 8}));
    CHECK(wi["dtype"] == "uint16");
    CHECK(wi["dims_from_metadata"] == true);
    json two = json::array();
    two.push_back({{"name", "488"}, {"wavelength_nm", 488.0}});
    two.push_back({{"name", "561"}, {"wavelength_nm", 561.0}});
    CHECK(wi["channels"] == two);
    std::vector<float> one = decoded(w->call("dataset_read", {{"path", waves}, {"c", 1}, {"t", 2}, {"z", 3}}), shape);
    CHECK(shape == std::vector<Index>{6, 8});
    CHECK(one[4 * 8 + 5] == static_cast<float>(stamp(1, 2, 3, 4, 5)));
    std::vector<float> wvol = decoded(w->call("dataset_read", {{"path", waves}, {"c", 0}, {"t", 1}}), shape);
    CHECK(shape == std::vector<Index>{4, 6, 8});
    CHECK(wvol[(2 * 6 + 1) * 8 + 7] == static_cast<float>(stamp(0, 1, 2, 1, 7)));
    CHECK(wvol[5] == static_cast<float>(stamp(0, 1, 0, 0, 5)));
    std::vector<float> mip = decoded(w->call("dataset_view", {{"path", waves}, {"kind", "mip"}, {"c", 1}, {"t", 0}}), shape);
    CHECK(shape == std::vector<Index>{6, 8});
    CHECK(mip[2 * 8 + 3] == static_cast<float>(stamp(1, 0, 3, 2, 3)));
    // a page order given by the Load step wins over the header
    const json given = w->call("dataset_info", {{"path", waves}, {"options", {{"page_order", "czt"}, {"c", 1}, {"t", 1}, {"z", 24}}}}).result;
    CHECK(given["dims"] == json({1, 1, 24, 6, 8}));
    CHECK(given["dims_from_metadata"] == false);
    CHECK_THROWS_WITH(w->call("dataset_info", {{"path", dir.file("missing.dv")}}), Catch::Matchers::ContainsSubstring("no such file on"));
}

TEST_CASE("engine: requests it does not serve go to its Python worker, progress and cancel included", "[app][engine]") {
    // the Python worker is played by a C++ stand-in here; the real one is in the parity case
    TestServer python(withToken("child", 4));
    EngineOptions o = quietEngine();
    o.connectPython = [&](const std::function<bool()>&) { return python.client("child"); };
    LoopbackEngine le(o);
    REQUIRE(waitUntil([&] { return le.engine.pythonStatus()["state"] == "ready"; }));
    auto w = le.connect();
    CHECK(w->capabilities().engine["python"]["state"] == "ready");
    CHECK(w->supports("double"));   // the worker's run kinds, advertised by the engine
    std::vector<float> in{1.f, 2.f};
    std::vector<std::string> messages;
    const WorkerResult r = w->call("run", {{"kind", "double"}}, {{"input", "float32", {2}, in.data(), 2 * sizeof(float)}},
                                   [&](double, const std::string& m) { messages.push_back(m); });
    CHECK(messages == std::vector<std::string>{"tile 0", "tile 1", "tile 2"});
    REQUIRE(r.tensors.size() == 1);
    CHECK(r.tensors[0].asFloat32()[1] == 4.f);
    CHECK_THROWS_WITH(w->call("run", {{"kind", "fail"}}), "worker: kaboom");
    std::atomic<bool> cancel{false};
    std::thread t([&] {
        CHECK_THROWS_AS(w->call("run", {{"kind", "slow"}}, {}, {}, [&] { return cancel.load(); }), CancelledError);
    });
    REQUIRE(waitUntil([&] { return python.started.load() >= 3; }));
    cancel = true;
    t.join();
    // the engine's own methods stay its own: served here, not relayed
    const int relayed = python.started.load();
    CHECK_THROWS_WITH(w->call("dataset_info", {{"path", "/nowhere/x.tif"}}), Catch::Matchers::ContainsSubstring("DatasetError"));
    CHECK(python.started.load() == relayed);
}

// --- a real `sirius-cli serve` --------------------------------------------------------------------------

TEST_CASE("engine: sirius-cli serve announces, takes its token from a file and serves RemoteSource", "[app][engine]") {
    TempDir dir;
    const std::vector<Index> shape{1, 2, 4, 64, 80};
    const std::string data = dir.file("stack.npy");
    writeNpy(data, shape);
    const std::string tokenFile = dir.file("token.engine");
    {
        std::ofstream(fs::u8path(tokenFile)) << "file-token-123\n";
        fs::permissions(fs::u8path(tokenFile), fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace);
    }
    ChildProcess p;
    std::mutex logMutex;
    std::string log;
    p.setErrorHandler([&](const std::string& line) {
        const std::lock_guard<std::mutex> g(logMutex);
        log += line + "\n";
    });
    ChildProcess::Options o;
    o.program = SIRIUS_TEST_CLI;
    o.arguments = {"serve", "--port", "0", "--no-python-worker", "--device", "cpu", "--exit-with-parent"};
    o.environment = {{"SIRIUS_TOKEN_FILE", tokenFile}};
    o.killTree = true;
    REQUIRE(p.start(o));
    std::string line;
    REQUIRE(p.readLine(line, 60000));
    INFO(line << "\n"
              << log);
    CHECK(line.rfind("{\"port\":", 0) == 0);   // what the cluster session looks for in the job's log
    const json a = json::parse(line);
    REQUIRE(a.contains("port"));
    const int port = a["port"].get<int>();
    CHECK(port > 0);
    CHECK(a["host"] == "127.0.0.1");
    CHECK(a["device"] == "cpu");
    CHECK(a["engine"]["build"] == buildInfo().build);
    CHECK(a["engine"]["ops_schema"] == buildInfo().opsSchema);
    CHECK(a["engine"]["api"] == kEngineApiVersion);
    CHECK(waitUntil([&] { return !fs::exists(fs::u8path(tokenFile)); }, std::chrono::seconds(5)));   // read, then deleted

    CHECK_THROWS_WITH(RemoteWorker::connect("127.0.0.1", port, "not-the-token"), Catch::Matchers::ContainsSubstring("authentication failed"));
    auto w = RemoteWorker::connect("127.0.0.1", port, "file-token-123");
    CHECK(w->capabilities().engine["build"] == buildInfo().build);
    CHECK(w->capabilities().maxClients == 8);
    // the hardware fields of the Python worker's hello, from the engine's own CUDA query
    CHECK(w->capabilities().gpus.size() == static_cast<std::size_t>(cudaDeviceCount()));
    CHECK(w->capabilities().cudaUsable == (cudaDeviceCount() > 0));
    CHECK(w->capabilities().cudaReason.empty() == (cudaDeviceCount() > 0));
    CHECK(w->capabilities().cpuThreads >= 1);

    auto datasets = std::make_shared<RemoteDatasets>("localengine", [&] { return RemoteWorker::connect("127.0.0.1", port, "file-token-123"); });
    datasets->install();
    OpenResult opened = openDataset(makeClusterPath("localengine", data));
    REQUIRE(opened.source);
    auto* remote = dynamic_cast<RemoteSource*>(opened.source.get());
    REQUIRE(remote);
    ViewRequest req;
    req.t = 1;
    req.index = 2;
    req.factor = 2;
    bool exact = false;
    remote->view(req, exact);
    remote->waitIdle();
    auto tile = remote->view(req, exact);
    INFO(remote->lastError());
    REQUIRE(tile);
    CHECK(exact);
    CHECK(tile->w == 40);
    std::vector<float> plane(64 * 80);
    opened.source->readPlane(0, 0, 3, plane.data());
    CHECK(plane[200] == npyValue(shape, 0, 0, 3, 2, 40));
    datasets->uninstall();
    opened.source.reset();
    datasets.reset();

    w->call("shutdown", json::object());
    CHECK(p.waitForExit(20000));
    CHECK(p.exitCode() == 0);
    const std::lock_guard<std::mutex> g(logMutex);
    CHECK(log.find("file-token-123") == std::string::npos);
}

TEST_CASE("engine: sirius-cli serve refuses a public address without a token", "[app][engine]") {
    ChildProcess p;
    ChildProcess::Options o;
    o.program = SIRIUS_TEST_CLI;
    o.arguments = {"serve", "--host", "0.0.0.0", "--port", "0", "--no-python-worker"};
    o.unsetEnvironment = {"SIRIUS_TOKEN", "SIRIUS_TOKEN_FILE"};
    o.mergeErrorLines = true;
    o.killTree = true;
    REQUIRE(p.start(o));
    std::string all, line;
    while (p.readLine(line, 20000)) all += line + "\n";
    CHECK(p.waitForExit(20000));
    CHECK(p.exitCode() == 2);
    CHECK(all.find("without a token") != std::string::npos);
    CHECK(all.find("\"port\"") == std::string::npos);
}

// --- parity with the Python worker ------------------------------------------------------------------------

TEST_CASE("engine: dataset replies equal the Python worker's on the same files", "[app][engine][parity]") {
    const std::string python = testPython();
    if (python.empty()) SKIP("no Python for the worker");
    LocalWorker worker;
    worker.setPython(python);
    worker.setScriptDir(SIRIUS_TEST_WORKER_DIR);
    worker.setDevice("cpu");
    std::unique_ptr<RemoteWorker> py;
    try {
        py = worker.connect();
    } catch (const std::exception& e) {
        SKIP(std::string("the Python worker does not start: ") + e.what());
    }
    LoopbackEngine le(quietEngine());
    auto cpp = le.connect();

    TempDir dir;
    std::vector<std::string> files;
    const std::string npy = dir.file("stack.npy");
    writeNpy(npy, {2, 2, 5, 37, 53});
    files.push_back(npy);
    // DeltaVision stacks: the engine (mrc_io.cpp) and the worker (datasets.py,
    // numpy alone) each read them with their own code
    files.push_back(std::string(SIRIUS_TEST_DATA_DIR) + "/raw.dv");
    {
        test::MrcSpec s;
        s.nx = 56;
        s.ny = 40;
        s.waves = 2;
        s.times = 2;
        s.planes = 5;
        s.sequence = 1;   // WZT: the planes of one (c, t) are not consecutive sections
        s.mode = 6;
        s.wavelengths = {488, 561, 0, 0, 0};
        const std::string waves = dir.file("waves.dv");
        test::writeMrc(waves, s, [](int w, int t, int z, int y, int x) { return (w * 2 + t) * 1000 + z * 100 + (y * 7 + x * 3) % 100; });
        files.push_back(waves);
    }
    const bool pythonReadsTiff = !py->capabilities().tiffReader.empty();
    if (pythonReadsTiff) {
        const std::string tif = dir.file("hyper.tif");
        writeHyperstack(tif, 2, 3, 4, 96, 128, 3);
        files.push_back(tif);
        const std::string flat = dir.file("flat.tif");
        writeHyperstack(flat, 1, 1, 6, 45, 70, 1);   // odd sizes, no pyramid
        files.push_back(flat);
    } else {
        WARN("the Python worker's interpreter has no sirius package: TIFF parity is not checked (" + python + ")");
    }

    for (const std::string& path : files) {
        INFO(path);
        const json a = py->call("dataset_info", {{"path", path}}).result, b = cpp->call("dataset_info", {{"path", path}}).result;
        for (const char* key : {"name", "format", "dims", "dtype", "bytes", "frame_interval_s", "channels", "rgb", "dims_from_metadata"}) {
            INFO(key);
            CHECK(a[key] == b[key]);
        }
        for (std::size_t k = 0; k < 3; ++k) CHECK(std::abs(a["voxel_um"][k].get<double>() - b["voxel_um"][k].get<double>()) < 1e-9);
        std::string pa = a["path"].get<std::string>(), pb = b["path"].get<std::string>();
        std::replace(pa.begin(), pa.end(), '\\', '/');
        std::replace(pb.begin(), pb.end(), '\\', '/');
        CHECK(lowerOf(pa) == lowerOf(pb));

        const Index c = 1, t = a["dims"][1].get<Index>() - 1, z = a["dims"][2].get<Index>() / 2;
        std::vector<json> requests;
        const json base = {{"path", path}, {"c", c}, {"t", t}, {"accept", {"zlib"}}};
        json plane = base;
        plane["z"] = z;
        requests.push_back({{"method", "dataset_read"}, {"params", plane}});
        requests.push_back({{"method", "dataset_read"}, {"params", base}});
        for (const char* kind : {"xy", "xz", "yz", "mip"})
            for (int f : {1, 2, 3, 4}) {
                json p = base;
                p["kind"] = kind;
                p["index"] = std::string(kind) == "xy" ? z : 7;
                p["factor"] = f;
                requests.push_back({{"method", "dataset_view"}, {"params", p}});
                p["region"] = {5, 3, 30, 17};
                requests.push_back({{"method", "dataset_view"}, {"params", p}});
                p["region"] = {32, 0, 64, 64};
                requests.push_back({{"method", "dataset_view"}, {"params", p}});
            }
        for (int side : {8, 16, 256}) {
            json p = base;
            p["kind"] = "volume";
            p["max_side"] = side;
            requests.push_back({{"method", "dataset_view"}, {"params", p}});
        }
        for (const json& r : requests) {
            INFO(r.dump());
            const std::string method = r["method"];
            WorkerResult ra, rb;
            std::string ea, eb;
            try {
                ra = py->call(method, r["params"]);
            } catch (const std::exception& e) {
                ea = e.what();
            }
            try {
                rb = cpp->call(method, r["params"]);
            } catch (const std::exception& e) {
                eb = e.what();
            }
            CHECK(ea.empty() == eb.empty());
            if (!ea.empty() || !eb.empty()) {
                INFO(ea << " | " << eb);
                continue;
            }
            CHECK(ra.result["dtype"] == rb.result["dtype"]);
            CHECK(ra.result["shape"] == rb.result["shape"]);
            std::vector<Index> sa, sb;
            const std::vector<float> va = decoded(ra, sa), vb = decoded(rb, sb);
            REQUIRE(sa == sb);
            std::size_t differ = 0;
            for (std::size_t i = 0; i < va.size(); ++i) differ += va[i] != vb[i] ? 1 : 0;
            CHECK(differ == 0);
        }
        for (Index tt = 0; tt < a["dims"][1].get<Index>(); ++tt) {
            const json p = {{"path", path}, {"c", 0}, {"t", tt}};
            const json sa = py->call("dataset_stats", p).result, sb = cpp->call("dataset_stats", p).result;
            for (const char* key : {"lo", "hi", "min", "max"}) {
                INFO(key << " " << sa.dump() << " " << sb.dump());
                CHECK(std::abs(sa[key].get<double>() - sb[key].get<double>()) <= 1e-9 * std::max(1.0, std::abs(sa[key].get<double>())));
            }
        }
    }

    SECTION("the engine relays to the real Python worker as its child") {
        EngineOptions o = quietEngine();
        o.pythonWorker = true;
        o.python = python;
        o.workerDir = SIRIUS_TEST_WORKER_DIR;
        LoopbackEngine withChild(o);
        REQUIRE(waitUntil([&] { return withChild.engine.pythonStatus()["state"] != "starting"; }, std::chrono::seconds(120)));
        INFO(withChild.engine.pythonStatus().dump());
        REQUIRE(withChild.engine.pythonStatus()["state"] == "ready");
        auto w = withChild.connect();
        CHECK(w->capabilities().engine["python"]["state"] == "ready");
        const json plugins = w->call("list_plugins", json::object()).result;
        CHECK(plugins.contains("plugins"));
        CHECK_THROWS_WITH(w->call("no_such_method", json::object()), "worker: unknown method 'no_such_method'");
    }
}

// --- folder datasets on the cluster ----------------------------------------------------------------

namespace {

    // One uint16 stack (z, y, x) whose voxel is (seed * 101 + z * 37 + y * 5 + x * 3) % 3001.
    void writeFolderStack(const fs::path& path, Index z, Index y, Index x, int seed) {
        Buffer<std::uint16_t> s(Shape{z, y, x});
        for (Index k = 0; k < z; ++k)
            for (Index r = 0; r < y; ++r)
                for (Index c = 0; c < x; ++c) s.data()[(k * y + r) * x + c] = static_cast<std::uint16_t>((seed * 101 + k * 37 + r * 5 + c * 3) % 3001);
        writeTiffStack<std::uint16_t>(path.string(), s.view(), TiffWriteOptions{});
    }

    // A folder of stacks with its manifest: two channels, two time points, two tiles.
    fs::path manifestFolder(const fs::path& root) {
        const fs::path f = root / "acq";
        fs::create_directories(f);
        DatasetManifest m;
        m.name = "acq";
        m.voxelUm = {0.2, 0.2, 0.5};
        ChannelInfo a, b;
        a.label = "488";
        a.wavelengthNm = 488.0;
        b.label = "561";
        b.wavelengthNm = 561.0;
        m.channels = {a, b};
        TileInfo t0, t1;
        t0.name = "tile_x0";
        t1.name = "tile_x1";
        t1.positionUm = {0.0, 0.0, 3.6};
        t1.gridIndex = {0, 0, 1};
        m.tiles = {t0, t1};
        int seed = 0;
        for (const TileInfo& tile : m.tiles)
            for (const ChannelInfo& ch : m.channels)
                for (Index t = 0; t < 2; ++t) {
                    const std::string name = tile.name + "_c" + ch.label + "_t" + std::to_string(t) + ".tif";
                    writeFolderStack(f / name, 3, 24, 20, ++seed);
                    ManifestFile file;
                    file.path = name;
                    file.channel = ch.label;
                    file.t = t;
                    file.tile = tile.name;
                    m.files.push_back(file);
                }
        m.save(f / DatasetManifest::kFileName);
        return f;
    }

    // A folder of TIFF stacks and nothing else: f1, f2, f10 (read in that order).
    fs::path plainFolder(const fs::path& root) {
        const fs::path f = root / "frames";
        fs::create_directories(f);
        writeFolderStack(f / "f10.tif", 4, 16, 12, 10);
        writeFolderStack(f / "f1.tif", 4, 16, 12, 1);
        writeFolderStack(f / "f2.tif", 4, 16, 12, 2);
        return f;
    }

    // Every plane of `a` and `b` equal, and their meta as this computer's open says it.
    void sameDataset(ArraySource& local, ArraySource& remote) {
        const DatasetMeta& l = local.meta();
        const DatasetMeta& r = remote.meta();
        CHECK(r.dims.toString() == l.dims.toString());
        CHECK(r.sourceType == l.sourceType);
        CHECK(r.format == "cluster " + l.format);
        for (std::size_t k = 0; k < 3; ++k) CHECK(r.voxelUm[k] == l.voxelUm[k]);
        REQUIRE(r.channels.size() == l.channels.size());
        for (std::size_t k = 0; k < l.channels.size(); ++k) CHECK(r.channels[k].label == l.channels[k].label);
        CHECK(r.tiles.size() == (l.hasTiles() ? l.tiles.size() : 0u));
        CHECK(r.tileIndex == l.tileIndex);
        const Dims5& d = l.dims;
        std::vector<float> a(static_cast<std::size_t>(d.planeSize())), b(a.size());
        for (Index c = 0; c < d.c; ++c)
            for (Index t = 0; t < d.t; ++t)
                for (Index z = 0; z < d.z; ++z) {
                    CAPTURE(c, t, z);
                    local.readPlane(c, t, z, a.data());
                    remote.readPlane(c, t, z, b.data());
                    CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
                }
    }

    OpenOptions lazy(Index tile = 0) {
        OpenOptions o;
        o.readAll = false;
        o.tile = tile;
        return o;
    }

} // namespace

TEST_CASE("engine: folder datasets open on the node as on this computer: the manifest, its tiles, a folder of TIFFs", "[app][engine]") {
    TempDir dir;
    const fs::path withManifest = manifestFolder(dir.path);
    const fs::path frames = plainFolder(dir.path);
    LoopbackEngine le(quietEngine());
    auto datasets = std::make_shared<RemoteDatasets>("enginehost", [&] { return le.connect(); });
    datasets->install();

    SECTION("a folder with its manifest, each tile") {
        for (Index tile = 0; tile < 2; ++tile) {
            CAPTURE(tile);
            OpenResult local = openDataset(withManifest.generic_string(), lazy(tile));
            OpenResult remote = openDataset(makeClusterPath("enginehost", withManifest.generic_string()), lazy(tile));
            REQUIRE(local.source);
            REQUIRE(remote.source);
            CHECK(local.meta.dims.toString() == "c2 t2 z3 y24 x20");
            CHECK(local.meta.format == "folder");
            sameDataset(*local.source, *remote.source);
            CHECK(remote.meta.tiles.size() == 2);
            CHECK(remote.meta.tiles[1].name == "tile_x1");
            CHECK(remote.meta.tiles[1].gridIndex[2] == 1);
        }
        // the manifest file named instead of its folder: the same
        const std::string toml = (withManifest / DatasetManifest::kFileName).generic_string();
        OpenResult byFile = openDataset(makeClusterPath("enginehost", toml), lazy());
        CHECK(byFile.meta.dims.toString() == "c2 t2 z3 y24 x20");
    }
    SECTION("a folder of TIFF stacks without a manifest: one stack per file, in reading order") {
        OpenResult local = openDataset(frames.generic_string(), lazy());
        OpenResult remote = openDataset(makeClusterPath("enginehost", frames.generic_string()), lazy());
        REQUIRE(local.source);
        REQUIRE(remote.source);
        CHECK(local.meta.dims.toString() == "c1 t3 z4 y16 x12");
        sameDataset(*local.source, *remote.source);
        // f1, f2, f10: t = 2 is f10's stack (seed 10)
        std::vector<float> p(16 * 12);
        remote.source->readPlane(0, 2, 1, p.data());
        CHECK(p[5 * 12 + 4] == static_cast<float>((10 * 101 + 1 * 37 + 5 * 5 + 4 * 3) % 3001));
        CHECK_FALSE(fs::exists(frames / DatasetManifest::kFileName));   // nothing written into the data
    }
    SECTION("views and the display window of a folder, through the protocol") {
        auto w = le.connect();
        const json info = w->call("dataset_info", {{"path", withManifest.generic_string()}, {"options", json::object()}}).result;
        CHECK(info["format"] == "folder");
        CHECK(info["dtype"] == "uint16");
        CHECK(info["dims"] == json::array({2, 2, 3, 24, 20}));
        CHECK(info["tiles"].size() == 2);
        const json stats = w->call("dataset_stats", {{"path", withManifest.generic_string()}, {"c", 1}, {"t", 1}}).result;
        CHECK(stats["hi"].get<double>() > stats["lo"].get<double>());
        std::vector<Index> shape;
        const std::vector<float> mip = decoded(w->call("dataset_view", {{"path", withManifest.generic_string()}, {"kind", "mip"}, {"c", 0}, {"t", 0}}), shape);
        CHECK(shape == std::vector<Index>{24, 20});
        CHECK_FALSE(mip.empty());
        // a folder that holds no dataset says so
        fs::create_directories(dir.path / "empty");
        CHECK_THROWS_WITH(w->call("dataset_info", {{"path", (dir.path / "empty").generic_string()}}), Catch::Matchers::ContainsSubstring("no TIFF files"));
    }
    datasets->uninstall();
}

TEST_CASE("engine: sirius-cli serve opens a folder dataset as this computer does", "[app][engine]") {
    TempDir dir;
    const fs::path withManifest = manifestFolder(dir.path);
    const fs::path frames = plainFolder(dir.path);
    const std::string tokenFile = dir.file("token.engine");
    {
        std::ofstream(fs::u8path(tokenFile)) << "folder-token-7\n";
        fs::permissions(fs::u8path(tokenFile), fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace);
    }
    ChildProcess p;
    ChildProcess::Options o;
    o.program = SIRIUS_TEST_CLI;
    o.arguments = {"serve", "--port", "0", "--no-python-worker", "--device", "cpu", "--exit-with-parent"};
    o.environment = {{"SIRIUS_TOKEN_FILE", tokenFile}};
    o.killTree = true;
    REQUIRE(p.start(o));
    std::string line;
    REQUIRE(p.readLine(line, 60000));
    const json a = json::parse(line);
    REQUIRE(a.contains("port"));
    const int port = a["port"].get<int>();
    auto datasets = std::make_shared<RemoteDatasets>("realengine", [&] { return RemoteWorker::connect("127.0.0.1", port, "folder-token-7"); });
    datasets->install();
    for (const fs::path& folder : {withManifest, frames}) {
        CAPTURE(folder.filename().string());
        OpenResult local = openDataset(folder.generic_string(), lazy());
        OpenResult remote = openDataset(makeClusterPath("realengine", folder.generic_string()), lazy());
        REQUIRE(local.source);
        REQUIRE(remote.source);
        sameDataset(*local.source, *remote.source);
    }
    OpenResult tile1 = openDataset(makeClusterPath("realengine", withManifest.generic_string()), lazy(1));
    OpenResult localTile1 = openDataset(withManifest.generic_string(), lazy(1));
    sameDataset(*localTile1.source, *tile1.source);
    datasets->uninstall();
    datasets.reset();
    RemoteWorker::connect("127.0.0.1", port, "folder-token-7")->call("shutdown", json::object());
    CHECK(p.waitForExit(20000));
}
