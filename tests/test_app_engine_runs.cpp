// The application's pipelines on SIRIUS's engine (core/engine_node.hpp,
// core/engine_server.hpp), as the HPC backend runs them: the application's
// Workbench sends the pipeline, the engine runs it with the same Executor and
// keeps the outputs, and the application draws them by their handles
// (NodeOutputSource) without ever holding their volumes.
//
//   * in-process: the engine on rpc::loopbackPair(), the "cluster" dataset a
//     file of this machine named cluster://enginehost/<path>;
//   * parity: the same pipeline on this computer (CPU) and through the engine
//     (CPU) gives bit-identical outputs and the same diagnostics;
//   * the refusals: no engine, another engine, files to upload (never by
//     default, and when agreed, uploaded and run there);
//   * what is computed there and only reported here: previews, validation of
//     the node's files, statistics; whole volumes are never read by accident;
//   * job end: the handles are gone, said in words;
//   * a real `sirius-cli serve` on 127.0.0.1.
//
// No network host is contacted: everything is loopback.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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

#include <sirius/tiff_io.hpp>

#include "core/build_info.hpp"
#include "core/ops/contrast.hpp"
#include "core/engine_node.hpp"
#include "core/engine_server.hpp"
#include "core/errors.hpp"
#include "core/executor.hpp"
#include "core/host.hpp"
#include "core/process.hpp"
#include "core/remote_source.hpp"
#include "core/rpc.hpp"
#include "core/serialize.hpp"
#include "core/statistics.hpp"
#include "core/workbench.hpp"
#include "temp_path.hpp"

using namespace sirius;
using namespace sirius::app;
using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

    struct TempDir {
        fs::path path;
        TempDir() : path(sirius::test::uniqueTempPath("engruns", "")) { fs::create_directories(path); }
        ~TempDir() {
            std::error_code ec;
            fs::remove_all(path, ec);
        }
        std::string file(const std::string& name) const { return (path / name).generic_string(); }
    };

    // An ImageJ stack of uint16 (t, z, y, x), one channel: a smooth blob that
    // moves with t, plus a little structure, so every step changes something.
    void writeStack(const std::string& path, Index t, Index z, Index y, Index x) {
        Buffer<std::uint16_t> stack(Shape{t * z, y, x});
        for (Index ti = 0; ti < t; ++ti)
            for (Index zi = 0; zi < z; ++zi)
                for (Index r = 0; r < y; ++r)
                    for (Index c = 0; c < x; ++c) {
                        const double dx = static_cast<double>(c) - 20.0 - 3.0 * static_cast<double>(ti), dy = static_cast<double>(r) - 30.0,
                                     dz = static_cast<double>(zi) - static_cast<double>(z) / 2.0;
                        const double v = 100.0 + 2000.0 * std::exp(-(dx * dx + dy * dy) / 60.0 - dz * dz / 4.0) + 7.0 * ((r * 3 + c * 5) % 11);
                        stack.data()[((ti * z + zi) * y + r) * x + c] = static_cast<std::uint16_t>(v);
                    }
        TiffWriteOptions o;
        o.description = "ImageJ=1.53t\nimages=" + std::to_string(t * z) + "\nchannels=1\nslices=" + std::to_string(z) + "\nframes=" +
                        std::to_string(t) + "\nhyperstack=true\nspacing=0.3\nunit=micron\n";
        o.xPixelUm = 0.1;
        o.yPixelUm = 0.1;
        writeTiffStack<std::uint16_t>(path, stack.view(), o);
    }

    // A flat-field image: one float32 page, brighter in the middle.
    void writeFlat(const std::string& path, Index y, Index x) {
        Buffer<float> flat(Shape{1, y, x});
        for (Index r = 0; r < y; ++r)
            for (Index c = 0; c < x; ++c) {
                const double u = (static_cast<double>(r) - y / 2.0) / y, v = (static_cast<double>(c) - x / 2.0) / x;
                flat.data()[r * x + c] = static_cast<float>(1.0 - 0.3 * (u * u + v * v));
            }
        writeTiffStack<float>(path, flat.view(), TiffWriteOptions{});
    }

    // An engine served in-process: each connection a loopback pair served on a thread of its own.
    struct LoopbackEngine {
        EngineServer engine;
        std::mutex m;
        std::vector<std::thread> threads;

        explicit LoopbackEngine(EngineOptions o) : engine(std::move(o)) {}
        ~LoopbackEngine() {
            engine.stop();
            const std::lock_guard<std::mutex> g(m);
            for (std::thread& t : threads) t.join();
        }
        std::unique_ptr<RemoteWorker> connect() {
            auto [client, server] = rpc::loopbackPair();
            {
                const std::lock_guard<std::mutex> g(m);
                threads.emplace_back([this, s = std::move(server)]() mutable { engine.serveConnection(std::move(s)); });
            }
            return std::make_unique<RemoteWorker>(std::move(client), "s3cret");
        }
    };

    EngineOptions quietEngine(const std::string& scratch = {}) {
        EngineOptions o;
        o.token = "s3cret";
        o.pythonWorker = false;
        o.device = "cpu";
        o.scratch = scratch;
        return o;
    }

    // The application connected to `engine` (through a box the test can point elsewhere).
    struct Endpoint {
        std::mutex m;
        LoopbackEngine* engine = nullptr;
        std::unique_ptr<RemoteWorker> connect() {
            const std::lock_guard<std::mutex> g(m);
            if (!engine) throw ProtocolError("no engine");
            return engine->connect();
        }
    };

    RemoteConfig engineConfig(const std::shared_ptr<Endpoint>& ep) {
        RemoteConfig rc;
        rc.connect = [ep](const std::function<bool()>&) { return ep->connect(); };
        rc.known = true;
        rc.engine = ep->connect()->capabilities().engine;
        rc.where = "enginehost \xC2\xB7 job 4711";
        return rc;
    }

    // Flat-field -> Deskew -> Deconvolve -> Contrast after Load, every step cached in memory.
    void buildPipeline(Workbench& wb, const std::string& flat) {
        while (wb.pipeline().size() > 1) wb.removeStep(1);
        wb.addStep("flatfield", -1, false);
        wb.setStepParam(1, "flat", flat);
        wb.addStep("deskew", -1, false);
        wb.addStep("decon", -1, false);
        wb.setStepParam(3, "iterations", std::int64_t{3});
        wb.setStepParam(3, "psf_size", std::int64_t{9});
        wb.addStep("contrast", -1, false);
        for (int i = 1; i < wb.pipeline().size(); ++i) wb.setStepCache(i, CachePolicy::Memory);
    }

    std::shared_ptr<RunJob> run(Workbench& wb, int target = -1) {
        std::shared_ptr<RunJob> job = wb.createRun(target);
        if (!job) return nullptr;
        job->execute();
        wb.finishRun(job);
        return job;
    }

    // Polls the workbench until `done` (the engine's answers arrive on a thread of their own).
    template <typename F> bool pollUntil(Workbench& wb, F done, std::chrono::seconds limit = std::chrono::seconds(60)) {
        const auto end = std::chrono::steady_clock::now() + limit;
        while (std::chrono::steady_clock::now() < end) {
            wb.poll();
            if (done()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        wb.poll();
        return done();
    }

    // The whole (c, t) volume of a node output, read from the engine as dataset_read sends it.
    std::vector<float> nodeVolume(RemoteWorker& w, const std::string& handle, Index c, Index t) {
        const WorkerResult r = w.call("dataset_read", {{"path", handle}, {"c", c}, {"t", t}, {"accept", {"zlib"}}});
        REQUIRE(r.tensors.size() == 1);
        std::vector<Index> shape;
        return decodeWorkerArray(r.result, r.tensors.front(), shape);
    }

    bool logHas(const Workbench& wb, const std::string& text) {
        for (const std::string& l : wb.log())
            if (l.find(text) != std::string::npos) return true;
        std::string all;
        for (const std::string& l : wb.log()) all += l + "\n";
        UNSCOPED_INFO("the log:\n"
                      << all);
        return false;
    }

    // Diagnostics as the wire carries them, the images' values included.
    json diagnosticsWire(const Diagnostics& d) {
        EncodedDiagnostics e = encodeDiagnostics(d);
        json j = e.json;
        json values = json::array();
        for (const rpc::Tensor& t : e.tensors) {
            std::vector<float> v(t.bytes.size() / sizeof(float));
            std::memcpy(v.data(), t.bytes.data(), v.size() * sizeof(float));
            values.push_back(v);
        }
        j["values"] = values;
        return j;
    }

} // namespace

TEST_CASE("engine runs: Load(cluster) -> Flat-field -> Deskew -> Deconvolve -> Contrast runs on the node, held there", "[app][engine][hpc]") {
    TempDir dir;
    const std::string data = dir.file("raw.tif"), flat = dir.file("flat.tif");
    writeStack(data, 2, 8, 64, 72);
    writeFlat(flat, 64, 72);
    LoopbackEngine le(quietEngine(dir.file("node-scratch")));
    auto ep = std::make_shared<Endpoint>();
    ep->engine = &le;
    // the cluster's datasets open through the engine, as the cluster session installs them
    auto datasets = std::make_shared<RemoteDatasets>("enginehost", [ep] { return ep->connect(); });
    datasets->install();

    TempDir scratch;
    Workbench wb(scratch.path / "wb");
    wb.openDataset(makeClusterPath("enginehost", data));
    REQUIRE(wb.hasDataset());
    wb.setRemoteConfig(engineConfig(ep));
    wb.setBackend(Backend::Hpc);
    wb.setHpcDevice(HpcDevice::Cpu);
    buildPipeline(wb, makeClusterPath("enginehost", flat));
    // the flat-field image is on the cluster: checked there, by the engine
    CHECK(wb.stepValidation(1).ok());
    REQUIRE(pollUntil(wb, [&] { return wb.stepValidation(1).warnings.empty(); }));
    CHECK(wb.stepValidation(1).ok());

    const std::uint64_t volumes = RemoteDownloads::volumeBytes(), planes = RemoteDownloads::planeBytes();
    std::shared_ptr<RunJob> job = run(wb);
    REQUIRE(job);
    INFO(job->error());
    REQUIRE(job->succeeded());
    CHECK(job->ranOnEngine());
    // nothing came here but the diagnostics: no volume, no plane
    CHECK(RemoteDownloads::volumeBytes() == volumes);
    CHECK(RemoteDownloads::planeBytes() == planes);
    CHECK(logHas(wb, "Run to step 05 on HPC \xC2\xB7 enginehost \xC2\xB7 job 4711 \xC2\xB7 CPU"));
    for (int i = 1; i <= 4; ++i) {
        std::shared_ptr<const StepOutput> out = wb.output(i);
        REQUIRE(out);
        CAPTURE(i);
        CHECK_FALSE(out->array);   // never held here
        auto* node = dynamic_cast<const NodeOutputSource*>(out->source.get());
        REQUIRE(node);
        CHECK(node->session() == le.engine.node().session());
        CHECK(out->where == "enginehost \xC2\xB7 job 4711");
        CHECK(out->ranOnDevice == "CPU");
        CHECK(wb.placementOf(i) == "node CPU");
        CHECK(wb.outputFresh(i));
        CHECK(logHas(wb, "Step " + Step::number(i) + " " + wb.pipeline().at(i).name + " \xC2\xB7"));
    }
    CHECK(logHas(wb, "on enginehost \xC2\xB7 job 4711 (CPU)"));
    CHECK_FALSE(logHas(wb, "on this computer"));
    // the decon's diagnostics came back with the result
    CHECK_FALSE(wb.diagnosticsOf(3).empty());
    CHECK(wb.diagnosticsOf(3).kind == DiagnosticsKind::Deconvolve);

    // the viewer draws any step through views computed on the node
    for (int i = 1; i <= 4; ++i) {
        std::shared_ptr<const StepOutput> out = wb.output(i);
        ViewProvider* views = out->source->viewProvider();
        REQUIRE(views);
        ViewRequest r;
        r.kind = ViewRequest::Kind::XY;
        r.t = 1;
        r.index = out->meta.dims.z / 2;
        r.factor = 2;
        bool exact = false;
        views->view(r, exact);
        static_cast<RemoteSource*>(out->source.get())->waitIdle();
        auto tile = views->view(r, exact);
        INFO(views->lastError());
        REQUIRE(tile);
        CHECK(exact);
        CHECK(tile->w == static_cast<int>((out->meta.dims.x + 1) / 2));
    }
    CHECK(RemoteDownloads::volumeBytes() == volumes);

    // a second run: everything is fresh here and there
    std::shared_ptr<RunJob> again = run(wb);
    REQUIRE(again);
    REQUIRE(again->succeeded());
    for (const StepReport& r : again->reports()) CHECK_FALSE(r.ran());

    // the application drops a step's output (a cache cleared): one round trip,
    // the node still holds it, nothing runs again
    wb.clearCache(4);
    std::shared_ptr<RunJob> third = run(wb);
    REQUIRE(third);
    REQUIRE(third->succeeded());
    for (const StepReport& r : third->reports()) CHECK_FALSE(r.ran());
    REQUIRE(wb.output(4));
    CHECK(dynamic_cast<const NodeOutputSource*>(wb.output(4)->source.get()));

    // a whole volume is never read here by accident; asked for, it is
    std::shared_ptr<const StepOutput> decon = wb.output(3);
    CHECK_THROWS_AS(decon->asInput().materialize(), RemoteDataError);
    {
        RemoteDownloads::Allow allow("the test");
        const ArrayPtr a = decon->asInput().materialize();
        REQUIRE(a);
        CHECK(a->dims().numel() == decon->meta.dims.numel());
    }
    CHECK(RemoteDownloads::volumeBytes() == volumes + static_cast<std::uint64_t>(decon->meta.dims.numel()) * sizeof(float));

    // statistics of a node output are computed on the node
    StatisticsOptions so;
    so.histogramBins = 16;
    const std::vector<ChannelStatistics> stats = channelStatistics(*wb.output(4), so);
    REQUIRE(stats.size() == 1);
    CHECK(stats[0].count == static_cast<std::uint64_t>(wb.output(4)->meta.dims.z * wb.output(4)->meta.dims.planeSize()));
    CHECK(stats[0].min >= 0.0);
    CHECK(stats[0].max <= 1.0);
    CHECK(stats[0].histogram.size() == 16);
    datasets->uninstall();
}

TEST_CASE("engine runs: the node's results equal this computer's, bit for bit, with the same diagnostics", "[app][engine][hpc][parity]") {
    TempDir dir;
    const std::string data = dir.file("raw.tif"), flat = dir.file("flat.tif");
    writeStack(data, 2, 8, 64, 72);
    writeFlat(flat, 64, 72);
    LoopbackEngine le(quietEngine());
    auto ep = std::make_shared<Endpoint>();
    ep->engine = &le;
    auto datasets = std::make_shared<RemoteDatasets>("enginehost", [ep] { return ep->connect(); });
    datasets->install();

    TempDir scratch;
    Workbench node(scratch.path / "node");
    node.openDataset(makeClusterPath("enginehost", data));
    node.setRemoteConfig(engineConfig(ep));
    node.setBackend(Backend::Hpc);
    node.setHpcDevice(HpcDevice::Cpu);
    buildPipeline(node, makeClusterPath("enginehost", flat));
    std::shared_ptr<RunJob> remote = run(node);
    REQUIRE(remote);
    INFO(remote->error());
    REQUIRE(remote->succeeded());

    Workbench here(scratch.path / "here");
    here.openDataset(data);
    here.setBackend(Backend::Cpu);
    buildPipeline(here, flat);
    std::shared_ptr<RunJob> local = run(here);
    REQUIRE(local);
    INFO(local->error());
    REQUIRE(local->succeeded());

    auto w = ep->connect();
    for (int i = 1; i <= 4; ++i) {
        CAPTURE(i);
        std::shared_ptr<const StepOutput> a = here.output(i), b = node.output(i);
        REQUIRE(a);
        REQUIRE(b);
        REQUIRE(a->array);
        CHECK(a->meta.dims.toString() == b->meta.dims.toString());
        auto* handle = dynamic_cast<const NodeOutputSource*>(b->source.get());
        REQUIRE(handle);
        for (Index t = 0; t < a->meta.dims.t; ++t) {
            const std::vector<float> v = nodeVolume(*w, handle->handle(), 0, t);
            const Index n = a->meta.dims.z * a->meta.dims.planeSize();
            REQUIRE(static_cast<Index>(v.size()) == n);
            CHECK(std::memcmp(v.data(), a->array->plane(0, t, 0), static_cast<std::size_t>(n) * sizeof(float)) == 0);
        }
        CHECK(diagnosticsWire(a->diagnostics) == diagnosticsWire(b->diagnostics));
        CHECK(a->ranOnDevice == "CPU");
        CHECK(here.placementOf(i) == "this computer \xC2\xB7 CPU");
    }
    datasets->uninstall();
}

TEST_CASE("engine runs: no engine, another engine, files of this computer: refused in words, never run here", "[app][engine][hpc]") {
    TempDir dir;
    const std::string data = dir.file("raw.tif"), flat = dir.file("flat.tif");
    writeStack(data, 1, 6, 48, 40);
    writeFlat(flat, 48, 40);
    LoopbackEngine le(quietEngine(dir.file("node-scratch")));
    auto ep = std::make_shared<Endpoint>();
    ep->engine = &le;
    TempDir scratch;
    Workbench wb(scratch.path / "wb");
    wb.openDataset(data);   // on this computer
    buildPipeline(wb, flat);
    wb.setBackend(Backend::Hpc);

    // a job that runs the Python worker only: the built-in steps are refused
    RemoteConfig plain = engineConfig(ep);
    plain.engine = nullptr;
    wb.setRemoteConfig(plain);
    CHECK_FALSE(wb.createRun());
    CHECK(wb.lastRunRefusal().kind == RunRefusal::Kind::NoEngine);
    CHECK(wb.lastRunRefusal().step == 1);
    CHECK_THAT(wb.lastRunRefusal().message, Catch::Matchers::ContainsSubstring("runs the Python worker only (no SIRIUS engine)"));
    CHECK_THAT(wb.lastRunRefusal().message, Catch::Matchers::ContainsSubstring("choose CPU/CUDA to run on this computer"));

    // an engine whose operations are another SIRIUS's
    RemoteConfig other = engineConfig(ep);
    other.engine["ops_schema"] = "0000";
    other.engine["build"] = "0.0.9+gdeadbee";
    wb.setRemoteConfig(other);
    CHECK_FALSE(wb.createRun());
    CHECK(wb.lastRunRefusal().kind == RunRefusal::Kind::EngineMismatch);
    CHECK_THAT(wb.lastRunRefusal().message, Catch::Matchers::ContainsSubstring("0.0.9+gdeadbee"));

    // the engine: the dataset and the flat image are on this computer, so the
    // run asks, with the size, and uploads nothing on its own
    wb.setRemoteConfig(engineConfig(ep));
    CHECK_FALSE(wb.createRun());
    const RunRefusal ask = wb.lastRunRefusal();
    CHECK(ask.kind == RunRefusal::Kind::NeedsUpload);
    REQUIRE(ask.uploads.size() == 2);
    CHECK(ask.uploads[0].path == data);
    CHECK(ask.uploads[0].bytes == fs::file_size(data));
    CHECK_THAT(ask.message, Catch::Matchers::ContainsSubstring("The dataset is on this computer: raw.tif ("));
    CHECK(fs::is_empty(le.engine.node().scratch() / "uploads"));
    // agreed: uploaded, then run there
    wb.allowUploads(ask.uploads);
    const std::uint64_t volumes = RemoteDownloads::volumeBytes();
    std::shared_ptr<RunJob> job = run(wb);
    REQUIRE(job);
    INFO(job->error());
    REQUIRE(job->succeeded());
    CHECK(job->ranOnEngine());
    CHECK(RemoteDownloads::volumeBytes() == volumes);
    int uploaded = 0;
    for (const auto& e : fs::recursive_directory_iterator(le.engine.node().scratch() / "uploads"))
        if (e.is_regular_file()) ++uploaded;
    CHECK(uploaded == 2);
    REQUIRE(wb.output(4));
    CHECK(dynamic_cast<const NodeOutputSource*>(wb.output(4)->source.get()));
    // a parameter edit runs again without uploading again (the node keeps the files)
    wb.setStepParam(4, "gamma", 0.8);
    std::shared_ptr<RunJob> second = run(wb);
    REQUIRE(second);
    REQUIRE(second->succeeded());
    CHECK(logHas(wb, "Step 05 Contrast"));

    // the folder of a dataset is not uploaded
    Workbench folder(scratch.path / "folder");
    folder.openDataset(data);
    folder.setRemoteConfig(engineConfig(ep));
    folder.setBackend(Backend::Hpc);
    buildPipeline(folder, makeClusterPath("enginehost", flat));
    folder.setStepParam(0, "path", dir.path.generic_string());
    CHECK_FALSE(folder.createRun());
    CHECK(folder.lastRunRefusal().kind == RunRefusal::Kind::NeedsUpload);
    CHECK_THAT(folder.lastRunRefusal().message, Catch::Matchers::ContainsSubstring("a folder is not uploaded"));
}

TEST_CASE("engine runs: previews and validation come from the node; a missing node file says so", "[app][engine][hpc]") {
    TempDir dir;
    const std::string data = dir.file("raw.tif");
    writeStack(data, 1, 6, 48, 40);
    LoopbackEngine le(quietEngine());
    auto ep = std::make_shared<Endpoint>();
    ep->engine = &le;
    auto datasets = std::make_shared<RemoteDatasets>("enginehost", [ep] { return ep->connect(); });
    datasets->install();
    TempDir scratch;
    Workbench wb(scratch.path / "wb");
    wb.openDataset(makeClusterPath("enginehost", data));
    wb.setRemoteConfig(engineConfig(ep));
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("contrast", -1, false);
    // the contrast histograms of a cluster dataset, computed on the node
    const std::uint64_t planes = RemoteDownloads::planeBytes(), volumes = RemoteDownloads::volumeBytes();
    Diagnostics d = wb.diagnosticsOf(1);
    REQUIRE(pollUntil(wb, [&] { return !wb.diagnosticsOf(1).histograms.empty(); }));
    d = wb.diagnosticsOf(1);
    REQUIRE(d.histograms.size() == 1);
    CHECK(d.histograms[0].binHi > d.histograms[0].binLo);
    // the panel's sliders, the viewer's live window, Auto and Reset: from the node too
    const ParamSet params = wb.pipeline().at(1).params;
    REQUIRE(pollUntil(wb, [&] { return wb.contrastWindowOf(1, params, 0, true).has_value(); }));
    const ContrastWindow w = *wb.contrastWindowOf(1, params, 0, true);
    CHECK(w.dataMax > w.dataMin);
    CHECK(w.hi > w.lo);
    CHECK(w.lo >= w.dataMin);
    CHECK(w.hi <= w.dataMax);
    REQUIRE(pollUntil(wb, [&] { return wb.contrastAutoOf(1, params).has_value(); }));
    const ParamSet automatic = *wb.contrastAutoOf(1, params);
    CHECK(automatic.getDouble("min") == static_cast<double>(w.lo));
    CHECK(automatic.getDouble("max") == static_cast<double>(w.hi));
    REQUIRE(pollUntil(wb, [&] { return wb.contrastResetOf(1, params).has_value(); }));
    const ParamSet reset = *wb.contrastResetOf(1, params);
    CHECK(reset.getDouble("min") == static_cast<double>(w.dataMin));
    CHECK(reset.getDouble("max") == static_cast<double>(w.dataMax));
    // the same as this computer computes from the planes it reads
    {
        const std::shared_ptr<const StepOutput> raw = wb.output(0);
        RemoteDownloads::Allow allow("the test's comparison");
        const ContrastWindow here = contrastWindow(raw->asInput(), params, 0, 8, true);
        CHECK(here.lo == w.lo);
        CHECK(here.hi == w.hi);
        CHECK(here.dataMin == w.dataMin);
        CHECK(here.dataMax == w.dataMax);
    }
    CHECK(RemoteDownloads::volumeBytes() == volumes);
    const std::uint64_t compared = RemoteDownloads::planeBytes();
    // a step added below an input on the cluster takes its defaults: its window is worked out on the node
    wb.addStep("contrast");
    CHECK(wb.pipeline().at(2).params.getDouble("max") <= wb.pipeline().at(2).params.getDouble("min"));
    CHECK(RemoteDownloads::planeBytes() == compared);
    wb.removeStep(2);
    CHECK(planes < compared);   // (the comparison itself read planes here)

    // a flat image that is not on the node: the node says so
    wb.addStep("flatfield", 1, false);
    wb.setStepParam(1, "flat", makeClusterPath("enginehost", dir.file("missing-flat.tif")));
    REQUIRE(pollUntil(wb, [&] { return !wb.stepValidation(1).ok(); }));
    CHECK_THAT(wb.stepValidation(1).firstError(), Catch::Matchers::ContainsSubstring("Flat image not found"));
    // without an engine it cannot be checked at all
    RemoteConfig none;
    wb.setRemoteConfig(none);
    CHECK_THAT(wb.stepValidation(1).firstError(), Catch::Matchers::ContainsSubstring("is on the cluster"));
    datasets->uninstall();
}

TEST_CASE("engine runs: the results are gone when the engine ends, said in words; a new engine holds none of them", "[app][engine][hpc]") {
    TempDir dir;
    const std::string data = dir.file("raw.tif"), flat = dir.file("flat.tif");
    writeStack(data, 1, 6, 48, 40);
    writeFlat(flat, 48, 40);
    auto ep = std::make_shared<Endpoint>();
    auto first = std::make_unique<LoopbackEngine>(quietEngine());
    ep->engine = first.get();
    auto datasets = std::make_shared<RemoteDatasets>("enginehost", [ep] { return ep->connect(); });
    datasets->install();
    TempDir scratch;
    Workbench wb(scratch.path / "wb");
    wb.openDataset(makeClusterPath("enginehost", data));
    wb.setRemoteConfig(engineConfig(ep));
    wb.setBackend(Backend::Hpc);
    buildPipeline(wb, makeClusterPath("enginehost", flat));
    std::shared_ptr<RunJob> job = run(wb);
    REQUIRE(job);
    INFO(job->error());
    REQUIRE(job->succeeded());
    const std::string session = first->engine.node().session();
    CHECK(wb.engineSession() == session);
    std::shared_ptr<const StepOutput> out = wb.output(3);
    auto* node = dynamic_cast<NodeOutputSource*>(const_cast<ArraySource*>(out->source.get()));
    REQUIRE(node);

    // the job ends; another engine (a new job) answers in its place
    {
        const std::lock_guard<std::mutex> g(ep->m);
        ep->engine = nullptr;
    }
    first.reset();
    LoopbackEngine second(quietEngine());
    {
        const std::lock_guard<std::mutex> g(ep->m);
        ep->engine = &second;
    }
    ViewRequest r;
    r.kind = ViewRequest::Kind::XY;
    r.index = 1;
    bool exact = false;
    node->view(r, exact);
    node->waitIdle();
    CHECK_FALSE(node->view(r, exact));
    CHECK_THAT(node->lastError(), Catch::Matchers::ContainsSubstring("earlier SIRIUS engine"));

    // the host learns the job ended: every handle of it goes, said in words
    CHECK(wb.nodeOutputsGone(session, "held by job 4711, which ended (TIMEOUT)") == 4);
    CHECK_FALSE(wb.outputFresh(3));
    REQUIRE(wb.output(3));
    CHECK(wb.output(3)->gone == "held by job 4711, which ended (TIMEOUT)");
    CHECK_FALSE(wb.output(3)->source);
    CHECK_FALSE(wb.output(3)->diagnostics.empty());   // what it said is kept
    CHECK(wb.placementOf(3) == "node CPU \xC2\xB7 gone");
    CHECK(node->lastError() == "held by job 4711, which ended (TIMEOUT)");
    CHECK(logHas(wb, "Step 04 Deconvolve: its result is gone, held by job 4711, which ended (TIMEOUT). Run it again."));
    // Run again: on the new engine
    wb.setRemoteConfig(engineConfig(ep));
    std::shared_ptr<RunJob> again = run(wb);
    REQUIRE(again);
    INFO(again->error());
    REQUIRE(again->succeeded());
    CHECK(wb.engineSession() == second.engine.node().session());
    CHECK(wb.outputFresh(3));
    datasets->uninstall();
}

TEST_CASE("engine runs: a real sirius-cli serve runs the pipeline; its handles end with it", "[app][engine][hpc]") {
    TempDir dir;
    const std::string data = dir.file("raw.tif"), flat = dir.file("flat.tif");
    writeStack(data, 1, 6, 48, 40);
    writeFlat(flat, 48, 40);
    const std::string tokenFile = dir.file("token");
    std::ofstream(fs::u8path(tokenFile)) << "serve-token-9\n";
    ChildProcess p;
    ChildProcess::Options o;
    o.program = SIRIUS_TEST_CLI;
    o.arguments = {"serve", "--port", "0", "--no-python-worker", "--device", "cpu", "--exit-with-parent", "--scratch", dir.file("scratch")};
    o.environment = {{"SIRIUS_TOKEN_FILE", tokenFile}};
    o.killTree = true;
    REQUIRE(p.start(o));
    std::string line;
    REQUIRE(p.readLine(line, 60000));
    const int port = json::parse(line)["port"].get<int>();
    RemoteConfig rc;
    rc.host = "127.0.0.1";
    rc.port = port;
    rc.token = "serve-token-9";
    auto caps = rc.open();
    rc.known = true;
    rc.engine = caps->capabilities().engine;
    REQUIRE(rc.hasEngine());
    auto datasets = std::make_shared<RemoteDatasets>("localengine", [rc] { return rc.open(); });
    datasets->install();
    TempDir scratch;
    Workbench wb(scratch.path / "wb");
    wb.openDataset(makeClusterPath("localengine", data));
    wb.setRemoteConfig(rc);
    wb.setBackend(Backend::Hpc);
    wb.setHpcDevice(HpcDevice::Cpu);
    buildPipeline(wb, makeClusterPath("localengine", flat));
    const std::uint64_t volumes = RemoteDownloads::volumeBytes();
    std::shared_ptr<RunJob> job = run(wb);
    REQUIRE(job);
    INFO(job->error());
    REQUIRE(job->succeeded());
    CHECK(RemoteDownloads::volumeBytes() == volumes);
    std::shared_ptr<const StepOutput> out = wb.output(4);
    auto* node = dynamic_cast<NodeOutputSource*>(const_cast<ArraySource*>(out->source.get()));
    REQUIRE(node);
    const std::optional<std::pair<float, float>> none = node->window(0, 0, true);
    CHECK_FALSE(none);
    node->waitIdle();
    const auto window = node->window(0, 0, true);
    REQUIRE(window);
    CHECK(window->first >= 0.0f);
    CHECK(window->second <= 1.0f);
    CHECK(fs::exists(fs::u8path(dir.file("scratch"))));

    // the engine ends: its outputs cannot be read any more, and say why
    caps->call("shutdown", json::object());
    CHECK(p.waitForExit(30000));
    ViewRequest r;
    r.index = 2;
    bool exact = false;
    node->view(r, exact);
    node->waitIdle();
    CHECK_FALSE(node->lastError().empty());
    CHECK(wb.nodeOutputsGone(node->session(), "held by the engine on 127.0.0.1, which stopped") == 4);
    CHECK_FALSE(wb.outputFresh(4));
    // its scratch went with it
    bool left = false;
    for (const auto& e : fs::directory_iterator(fs::u8path(dir.file("scratch")))) left = left || e.path().filename().string().rfind("sirius-engine-", 0) == 0;
    CHECK_FALSE(left);
    datasets->uninstall();
}

TEST_CASE("engine runs: a run on the node is cancelled from here, and the node serves views meanwhile", "[app][engine][hpc]") {
    TempDir dir;
    const std::string data = dir.file("raw.tif");
    writeStack(data, 3, 12, 96, 96);
    LoopbackEngine le(quietEngine());
    auto ep = std::make_shared<Endpoint>();
    ep->engine = &le;
    auto datasets = std::make_shared<RemoteDatasets>("enginehost", [ep] { return ep->connect(); });
    datasets->install();
    TempDir scratch;
    Workbench wb(scratch.path / "wb");
    wb.openDataset(makeClusterPath("enginehost", data));
    wb.setRemoteConfig(engineConfig(ep));
    wb.setBackend(Backend::Hpc);
    wb.setHpcDevice(HpcDevice::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("decon", -1, false);
    wb.setStepParam(1, "iterations", std::int64_t{500});
    wb.setStepParam(1, "stop_rel_change", 0.0);
    wb.setStepParam(1, "psf_size", std::int64_t{9});
    std::shared_ptr<RunJob> job = wb.createRun();
    REQUIRE(job);
    std::thread runner([job] { job->execute(); });
    // the node reports the step it runs, with its progress
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (job->progress().stepIndex.load() != 1 && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK(job->progress().stepIndex.load() == 1);
    // the dataset on screen is drawn while the node computes
    auto* raw = dynamic_cast<RemoteSource*>(wb.output(0)->source.get());
    REQUIRE(raw);
    ViewRequest r;
    r.index = 5;
    r.factor = 2;
    bool exact = false;
    raw->view(r, exact);
    raw->waitIdle();
    INFO(raw->lastError());
    CHECK(raw->view(r, exact));
    CHECK_FALSE(job->finished());
    job->cancel();
    runner.join();
    wb.finishRun(job);
    CHECK(job->wasCancelled());
    CHECK(logHas(wb, "Run cancelled"));
    // the engine is free again: a short run goes through
    wb.setStepParam(1, "iterations", std::int64_t{2});
    std::shared_ptr<RunJob> again = run(wb);
    REQUIRE(again);
    INFO(again->error());
    CHECK(again->succeeded());
    datasets->uninstall();
}
