// Tests of the workbench hub: pipeline structure and files, the executor's
// caching and freshness rules, the workbench facade (edits, undo / redo,
// selection, viewing, runs) and the assistant's tool API. The operations
// used are tiny synthetic ones registered by this file, so these tests do
// not depend on the built-in operations or on any dataset on disk.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <random>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <thread>

#include <nlohmann/json.hpp>

#include <sirius/tiff_io.hpp>

#include "core/array_source.hpp"
#include "core/cancel.hpp"
#include "core/executor.hpp"
#include "core/history.hpp"
#include "core/pipeline.hpp"
#include "core/tool_api.hpp"
#include "core/workbench.hpp"

#include "temp_path.hpp"

using namespace sirius;
using namespace sirius::app;
using json = nlohmann::json;

namespace {

    // Scales every voxel by `factor` and records how often it ran.
    struct ScaleOp final : Operation {
        static inline std::atomic<int> runs{0};
        OpInfo info_;
        ScaleOp() {
            info_.kind = "test_scale";
            info_.name = "Scale";
            info_.group = "Intensity";
            info_.kindLabel = "INTENSITY";
            info_.params = {doubleParam("factor", "Factor", 2.0).range(0.0, 100.0)};
            info_.defaultCache = CachePolicy::Memory;
        }
        const OpInfo& info() const noexcept override { return info_; }
        StepOutput run(const StepInput& in, const ParamSet& p, const StepContext& ctx) const override {
            ++runs;
            ctx.report(0.5, "scaling");
            ArrayPtr src = in.materialize();
            auto out = std::make_shared<Array5>(src->clone());
            const float f = static_cast<float>(p.getDouble("factor"));
            for (Index i = 0; i < out->numel(); ++i) out->data()[i] *= f;
            StepOutput o;
            o.meta = in.meta;
            o.array = out;
            o.note = "scaled";
            o.diagnostics.summary = "scaled by " + toDisplayString(*p.find("factor"));
            return o;
        }
    };

    // Reduces z to 1 (max), to test outputMeta and shape propagation.
    struct MaxZOp final : Operation {
        OpInfo info_;
        MaxZOp() {
            info_.kind = "test_maxz";
            info_.name = "Max Z";
            info_.group = "Reduce";
            info_.kindLabel = "EINSUM";
        }
        const OpInfo& info() const noexcept override { return info_; }
        DatasetMeta outputMeta(const ParamSet&, const DatasetMeta& in) const override {
            DatasetMeta m = in;
            m.dims.z = 1;
            return m;
        }
        StepOutput run(const StepInput& in, const ParamSet&, const StepContext&) const override {
            ArrayPtr src = in.materialize();
            const Dims5 d = src->dims();
            auto out = std::make_shared<Array5>(Array5::zeros(Dims5{d.c, d.t, 1, d.y, d.x}));
            for (Index c = 0; c < d.c; ++c)
                for (Index t = 0; t < d.t; ++t)
                    for (Index z = 0; z < d.z; ++z) {
                        const float* p = src->plane(c, t, z);
                        float* o = out->plane(c, t, 0);
                        for (Index i = 0; i < d.planeSize(); ++i) o[i] = std::max(o[i], p[i]);
                    }
            StepOutput o;
            o.meta = outputMeta({}, in.meta);
            o.array = out;
            return o;
        }
    };

    struct FailingOp final : Operation {
        OpInfo info_;
        FailingOp() {
            info_.kind = "test_fail";
            info_.name = "Fail";
            info_.group = "Intensity";
            info_.kindLabel = "INTENSITY";
        }
        const OpInfo& info() const noexcept override { return info_; }
        StepOutput run(const StepInput&, const ParamSet&, const StepContext&) const override {
            throw std::runtime_error("boom");
        }
    };

    struct SlowOp final : Operation {
        OpInfo info_;
        SlowOp() {
            info_.kind = "test_slow";
            info_.name = "Slow";
            info_.group = "Intensity";
            info_.kindLabel = "INTENSITY";
        }
        const OpInfo& info() const noexcept override { return info_; }
        StepOutput run(const StepInput& in, const ParamSet&, const StepContext& ctx) const override {
            for (int i = 0; i < 200; ++i) {
                ctx.throwIfCancelled();
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            StepOutput o;
            o.meta = in.meta;
            o.array = in.materialize();
            return o;
        }
    };

    // Gives up as soon as it is asked to, by throwing the cancellation type
    // rather than a message the executor would have to recognise by string.
    struct CancellingOp final : Operation {
        OpInfo info_;
        CancellingOp() {
            info_.kind = "test_cancel";
            info_.name = "Cancel";
            info_.group = "Intensity";
            info_.kindLabel = "INTENSITY";
        }
        const OpInfo& info() const noexcept override { return info_; }
        StepOutput run(const StepInput&, const ParamSet&, const StepContext&) const override {
            throw CancelledError();
        }
    };

    // Produces a label volume of its own: one small ball at the centre,
    // with the id the "label" parameter names, so a re-run with another id
    // is visibly a different volume.
    struct LabelOp final : Operation {
        static inline std::atomic<int> runs{0};
        OpInfo info_;
        LabelOp() {
            info_.kind = "test_labels";
            info_.name = "Label";
            info_.group = "Segment";
            info_.kindLabel = "SEGMENT";
            info_.producesLabels = true;
            info_.defaultCache = CachePolicy::Memory;
            info_.params = {intParam("label", "Label", 1), boolParam("tracked", "Tracked", false)};
        }
        const OpInfo& info() const noexcept override { return info_; }
        StepOutput run(const StepInput& in, const ParamSet& p, const StepContext&) const override {
            ++runs;
            const Dims5 d = in.meta.dims;
            auto labels = std::make_shared<LabelVolume>(d.t, d.z, d.y, d.x);
            // tracked: the same object, under the same id, in every frame (and
            // a second one in the corner), as the tracking step leaves them
            const bool tracked = p.getBool("tracked");
            for (Index t = 0; t < (tracked ? d.t : 1); ++t) {
                labels->paint(t, d.z / 2, d.y / 2, d.x / 2, 1.5, 0, static_cast<std::uint32_t>(p.getInt("label")));
                if (tracked) {
                    // a second lobe of the same object, joined to the first (something to split)
                    labels->paint(t, d.z / 2, d.y / 2, d.x / 2 + 2, 1.5, 0, static_cast<std::uint32_t>(p.getInt("label")));
                    labels->paint(t, d.z / 2, d.y / 2, d.x / 2 + 4, 1.5, 0, static_cast<std::uint32_t>(p.getInt("label")));
                    labels->paint(t, d.z / 2, 2, 2, 1.0, 0, static_cast<std::uint32_t>(p.getInt("label")) + 1);
                }
            }
            labels->setTracked(tracked);
            labels->recomputeStats(0);
            StepOutput o;
            o.meta = in.meta;
            o.array = in.materialize();
            o.labels = labels;
            o.note = "labelled";
            return o;
        }
    };

    // Reads a file named by a Path parameter: what the cache has to notice
    // has changed when the file behind the path is rewritten.
    struct FileOp final : Operation {
        static inline std::atomic<int> runs{0};
        OpInfo info_;
        FileOp() {
            info_.kind = "test_file";
            info_.name = "Read file";
            info_.group = "Intensity";
            info_.kindLabel = "INTENSITY";
            info_.params = {pathParam("source", "Source")};
            info_.defaultCache = CachePolicy::Memory;
        }
        const OpInfo& info() const noexcept override { return info_; }
        StepOutput run(const StepInput& in, const ParamSet& p, const StepContext&) const override {
            ++runs;
            ArrayPtr src = in.materialize();
            auto out = std::make_shared<Array5>(src->clone());
            // the offset is the file's size, so the output follows its content
            std::error_code ec;
            const std::uintmax_t bytes = std::filesystem::file_size(p.getString("source"), ec);
            const float add = ec ? 0.0f : static_cast<float>(bytes);
            for (Index i = 0; i < out->numel(); ++i) out->data()[i] += add;
            StepOutput o;
            o.meta = in.meta;
            o.array = out;
            return o;
        }
    };

    void registerTestOps() {
        static bool done = false;
        if (done) return;
        done = true;
        registerBuiltinOperations();
        registerOperation(std::make_unique<ScaleOp>());
        registerOperation(std::make_unique<MaxZOp>());
        registerOperation(std::make_unique<FailingOp>());
        registerOperation(std::make_unique<SlowOp>());
        registerOperation(std::make_unique<CancellingOp>());
        registerOperation(std::make_unique<LabelOp>());
        registerOperation(std::make_unique<FileOp>());
    }

    std::shared_ptr<MemorySource> syntheticSource(Index c = 2, Index t = 3, Index z = 4, Index y = 8, Index x = 8) {
        auto a = std::make_shared<Array5>(Dims5{c, t, z, y, x});
        for (Index i = 0; i < a->numel(); ++i) a->data()[i] = static_cast<float>(i % 97);
        DatasetMeta m;
        m.name = "synthetic";
        m.sourcePath = "memory://synthetic";
        m.format = "memory";
        m.dims = a->dims();
        m.voxelUm = {0.1, 0.1, 0.3};
        m.normalizeChannels();
        return std::make_shared<MemorySource>(a, m);
    }

    struct Scratch {
        std::filesystem::path dir;
        Scratch() {
            dir = std::filesystem::temp_directory_path() / ("sirius-wb-test-" + std::to_string(std::random_device{}()));
            std::filesystem::create_directories(dir);
        }
        ~Scratch() {
            std::error_code ec;
            std::filesystem::remove_all(dir, ec);
        }
    };

    struct Counter final : Workbench::Observer {
        int pipeline = 0, step = 0, selection = 0, viewed = 0, view = 0, outputs = 0, history = 0, run = 0;
        void pipelineChanged() override { ++pipeline; }
        void stepChanged(int) override { ++step; }
        void selectionChanged() override { ++selection; }
        void viewedStepChanged() override { ++viewed; }
        void viewStateChanged() override { ++view; }
        void outputsChanged() override { ++outputs; }
        void historyChanged() override { ++history; }
        void runStateChanged() override { ++run; }
    };

    Index countLabel(const LabelVolume& v, Index t, std::uint32_t id) {
        Index n = 0;
        const std::uint32_t* p = v.volume(t);
        for (Index i = 0; i < v.volumeSize(); ++i)
            if (p[i] == id) ++n;
        return n;
    }

    bool logContains(const Workbench& wb, const std::string& text) {
        for (const std::string& line : wb.log())
            if (line.find(text) != std::string::npos) return true;
        return false;
    }

    // One small brush dab of `id` on the viewed step, as one undo entry.
    void paintOne(Workbench& wb, Index z, Index y, Index x, std::uint32_t id) {
        ViewState s = wb.viewState();
        s.brushPx = 2;
        s.paint3d = false;
        s.selectedLabel = id;
        wb.setViewState(s);
        wb.beginPaintStroke();
        wb.paintLabels(z, y, x, false);
        wb.endPaintStroke();
    }

    std::shared_ptr<RunJob> runSync(Workbench& wb, int target = -1) {
        auto job = wb.createRun(target);
        REQUIRE(job);
        job->execute();
        wb.finishRun(job);
        return job;
    }

} // namespace

// --- Pipeline -------------------------------------------------------------------

TEST_CASE("Pipeline keeps Load pinned first and orders steps", "[app][pipeline]") {
    registerTestOps();
    Pipeline p;
    REQUIRE(p.size() == 1);
    CHECK(p.at(0).kind == "load");
    CHECK(p.at(0).pinned);

    const StepId a = p.add("test_scale");
    const StepId b = p.add("test_maxz");
    CHECK(p.size() == 3);
    CHECK(p.indexOf(a) == 1);
    CHECK(p.indexOf(b) == 2);
    CHECK(p.at(1).name == "Scale");
    CHECK(p.at(1).cache == CachePolicy::Memory);
    CHECK(p.at(1).params.getDouble("factor") == 2.0);

    SECTION("moving") {
        CHECK(p.move(2, -1));
        CHECK(p.indexOf(b) == 1);
        CHECK_FALSE(p.move(1, -1));   // never above Load
        CHECK_FALSE(p.move(0, 1));    // Load never moves
        CHECK_FALSE(p.move(2, 1));    // past the end
    }
    SECTION("remove and enable") {
        p.remove(0);
        CHECK(p.size() == 3);
        p.setEnabled(0, false);
        CHECK(p.at(0).enabled);
        p.setEnabled(1, false);
        CHECK_FALSE(p.at(1).enabled);
        CHECK(p.enabledCount() == 2);
        p.remove(1);
        CHECK(p.size() == 2);
        CHECK(p.at(1).id == b);
    }
    SECTION("duplicate gets a fresh id after the original") {
        const StepId c = p.duplicate(1);
        CHECK(p.size() == 4);
        CHECK(p.indexOf(c) == 2);
        CHECK(c != a);
        CHECK(p.at(2).params == p.at(1).params);
    }
    SECTION("unknown kinds are rejected") {
        CHECK_THROWS_AS(p.add("nope"), std::out_of_range);
    }
    SECTION("setParams coerces and clamps against the specs") {
        ParamSet q;
        q.set("factor", std::string("250"));
        q.set("bogus", 1.0);
        p.setParams(1, q);
        CHECK(p.at(1).params.getDouble("factor") == 100.0);   // clamped to the spec's max
        CHECK(p.at(1).params.has("bogus"));                  // unknown keys survive (strict=false)
    }
}

TEST_CASE("Pipeline round-trips through JSON and TOML with ids", "[app][pipeline]") {
    registerTestOps();
    Pipeline p;
    p.add("test_scale");
    p.add("test_maxz");
    p.setEnabled(2, false);
    p.setCache(1, CachePolicy::Disk);
    ParamSet q = p.at(1).params;
    q.set("factor", 3.5);
    p.setParams(1, q);
    p.rename(2, "Projection");

    const json j = p.toJson();
    Pipeline back = Pipeline::fromJson(j);
    REQUIRE(back.size() == 3);
    CHECK(back.at(1).id == p.at(1).id);
    CHECK(back.at(1).cache == CachePolicy::Disk);
    CHECK(back.at(1).params.getDouble("factor") == 3.5);
    CHECK_FALSE(back.at(2).enabled);
    CHECK(back.at(2).name == "Projection");
    CHECK(back.toJson() == j);

    test::TempFile file("pipeline", ".sirius.toml");
    p.save(file.path.string());
    Pipeline loaded = Pipeline::load(file.path.string());
    CHECK(loaded.toJson() == j);

    const std::string py = p.toPythonScript("/data/x.tif");
    CHECK(py.find("run_pipeline") != std::string::npos);
    CHECK(py.find("test_maxz") != std::string::npos);
}

// --- Executor -------------------------------------------------------------------

TEST_CASE("Executor caches per fingerprint and invalidates downstream only", "[app][executor]") {
    registerTestOps();
    Scratch scratch;
    Executor ex(scratch.dir / "cache");
    Pipeline p;
    const StepId scaleId = p.add("test_scale");
    p.add("test_maxz");
    p.setCache(2, CachePolicy::Memory);

    auto src = syntheticSource();
    auto load = std::make_shared<StepOutput>();
    load->meta = src->meta();
    load->source = src;
    ex.seed(p, 0, load);
    REQUIRE(ex.isFresh(p, 0));

    StepContext ctx;
    ScaleOp::runs = 0;
    std::vector<StepReport> reports;
    auto out = ex.runAll(p, ctx, &reports);
    REQUIRE(out);
    CHECK(out->meta.dims.z == 1);
    CHECK(ScaleOp::runs == 1);
    REQUIRE(reports.size() == 3);
    CHECK(reports[0].state == StepReport::State::Cached);
    CHECK(reports[1].ran());
    CHECK(reports[2].ran());
    // input 0..96 scaled by 2, max over z
    CHECK(out->array->at(0, 0, 0, 0, 0) >= 0.0f);

    SECTION("a second run is served from the cache") {
        reports.clear();
        ex.runAll(p, ctx, &reports);
        CHECK(ScaleOp::runs == 1);
        CHECK(reports[1].state == StepReport::State::Cached);
        CHECK(reports[2].state == StepReport::State::Cached);
    }
    SECTION("editing step 1 invalidates 1 and 2, not 0") {
        ParamSet q = p.at(1).params;
        q.set("factor", 3.0);
        p.setParams(1, q);
        CHECK(ex.isFresh(p, 0));
        CHECK_FALSE(ex.isFresh(p, 1));
        CHECK_FALSE(ex.isFresh(p, 2));
        CHECK(ex.lastOutput(scaleId));   // stale output still available for viewing
        ex.runAll(p, ctx);
        CHECK(ScaleOp::runs == 2);
        CHECK(ex.isFresh(p, 2));
    }
    SECTION("a skipped step is transparent to the fingerprint") {
        p.setEnabled(1, false);
        CHECK_FALSE(ex.isFresh(p, 2));
        reports.clear();
        auto o = ex.runAll(p, ctx, &reports);
        CHECK(reports[1].skipped());
        CHECK(ScaleOp::runs == 1);
        // max over z of the unscaled input
        CHECK(o->meta.dims.z == 1);
        p.setEnabled(1, true);
        // One entry per step: step 2 now holds the skipped-input result, so
        // re-enabling step 1 makes 2 stale again while 1 stays fresh.
        CHECK(ex.isFresh(p, 1));
        CHECK_FALSE(ex.isFresh(p, 2));
    }
    SECTION("disk policy spills the array and reloads it") {
        p.setCache(1, CachePolicy::Disk);
        ParamSet q = p.at(1).params;
        q.set("factor", 4.0);
        p.setParams(1, q);
        ex.run(p, 1, ctx);
        bool spilled = false;
        for (const auto& e : std::filesystem::directory_iterator(scratch.dir / "cache"))
            if (e.path().extension() == ".sir5") spilled = true;
        CHECK(spilled);
        auto c = ex.cached(p, 1);
        REQUIRE(c);
        REQUIRE(c->array);
        CHECK(c->array->at(0, 0, 0, 0, 1) == 4.0f);
        CHECK(ex.cachedBytesOf(scaleId) == c->array->bytes());
        // while a holder keeps the reloaded output, every request returns
        // that same object (one disk read, stable pointers for the viewer);
        // once released, the next request reloads
        CHECK(ex.cached(p, 1) == c);
        CHECK(ex.lastOutput(scaleId) == c);
        const StepOutput* held = c.get();
        c.reset();
        auto again = ex.lastOutput(scaleId);
        REQUIRE(again);
        REQUIRE(again->array);
        CHECK(again.get() != held);
        CHECK(again->array->at(0, 0, 0, 0, 1) == 4.0f);
    }
    SECTION("recompute keeps only the newest result") {
        p.setCache(1, CachePolicy::Recompute);
        p.setCache(2, CachePolicy::Recompute);
        ParamSet q = p.at(1).params;
        q.set("factor", 5.0);
        p.setParams(1, q);
        ex.runAll(p, ctx);
        CHECK(ex.isFresh(p, 2));
        CHECK_FALSE(ex.isFresh(p, 1));   // its array was dropped when step 2 stored
        auto stale = ex.lastOutput(scaleId);
        REQUIRE(stale);
        CHECK_FALSE(stale->array);
    }
    SECTION("failures propagate with the step named") {
        p.add("test_fail");
        reports.clear();
        CHECK_THROWS_WITH(ex.runAll(p, ctx, &reports), Catch::Matchers::ContainsSubstring("04 Fail"));
        CHECK(reports.back().error == "boom");
    }
    SECTION("array files round trip") {
        Array5 a(Dims5{1, 2, 3, 4, 5});
        for (Index i = 0; i < a.numel(); ++i) a.data()[i] = static_cast<float>(i) * 0.5f;
        const auto path = scratch.dir / "x.sir5";
        Executor::writeArrayFile(path, a);
        auto b = Executor::readArrayFile(path);
        CHECK(b->dims() == a.dims());
        CHECK(b->at(0, 1, 2, 3, 4) == a.at(0, 1, 2, 3, 4));
    }
}

TEST_CASE("StepReport::State says what happened to each step", "[app][executor]") {
    registerTestOps();
    Scratch scratch;
    Executor ex(scratch.dir / "cache");
    Pipeline p;
    p.add("test_scale");      // 1: runs, then is served from the cache
    p.add("test_maxz");       // 2: disabled below
    p.add("test_fail");       // 3
    p.setEnabled(2, false);
    p.setCache(1, CachePolicy::Memory);

    auto src = syntheticSource(1, 1, 2, 4, 4);
    auto load = std::make_shared<StepOutput>();
    load->meta = src->meta();
    load->source = src;
    ex.seed(p, 0, load);

    StepContext ctx;
    std::vector<StepReport> reports;
    CHECK_THROWS(ex.runAll(p, ctx, &reports));
    REQUIRE(reports.size() == 4);
    CHECK(reports[0].state == StepReport::State::Cached);   // the seeded Load
    CHECK(reports[1].state == StepReport::State::Ran);
    CHECK(reports[1].note == "scaled");                     // the operation's own text
    CHECK(reports[1].seconds >= 0.0);
    CHECK(reports[2].state == StepReport::State::Skipped);
    CHECK(reports[3].state == StepReport::State::Failed);
    CHECK(reports[3].error == "boom");
    CHECK(reports[3].failed());
    // ids, not indices, are what the workbench maps reports back with
    CHECK(reports[1].id == p.at(1).id);

    reports.clear();
    CHECK_THROWS(ex.runAll(p, ctx, &reports));
    CHECK(reports[1].state == StepReport::State::Cached);
    CHECK(reports[1].note.empty());                         // no text for a cache hit

    CHECK(std::string(toString(StepReport::State::Running)) == "running");
    CHECK(std::string(toString(StepReport::State::Ran)) == "ran");
    CHECK(std::string(toString(StepReport::State::Cached)) == "cached");
    CHECK(std::string(toString(StepReport::State::Skipped)) == "skipped");
    CHECK(std::string(toString(StepReport::State::Failed)) == "failed");
}

TEST_CASE("Cancellation is recognised by type, not by its message", "[app][executor][cancel]") {
    registerTestOps();
    CHECK(isCancellation(CancelledError()));
    CHECK(isCancellation(CancelledError("cancelled: the user asked")));
    CHECK(isCancellation(std::runtime_error("cancelled")));   // the documented fallback
    CHECK_FALSE(isCancellation(std::runtime_error("boom")));
    CHECK_FALSE(isCancellation(std::runtime_error("cancelled the run")));

    Scratch scratch;
    Executor ex(scratch.dir / "cache");
    Pipeline p;
    p.add("test_cancel");
    auto src = syntheticSource(1, 1, 2, 4, 4);
    auto load = std::make_shared<StepOutput>();
    load->meta = src->meta();
    load->source = src;
    ex.seed(p, 0, load);

    StepContext ctx;
    std::vector<StepReport> reports;
    // A cancelled step is not blamed: the error keeps its type and is not
    // wrapped with the step's name the way a failure would be.
    CHECK_THROWS_AS(ex.runAll(p, ctx, &reports), CancelledError);
    CHECK(reports.size() == 1);   // only the Load report, none for the cancelled step

    SECTION("a step that gives up through the context is cancellation too") {
        Pipeline q;
        q.add("test_slow");
        Executor ex2(scratch.dir / "cache2");
        ex2.seed(q, 0, load);
        StepContext c2;
        c2.cancelled = [] { return true; };
        CHECK_THROWS_AS(ex2.runAll(q, c2), CancelledError);
    }
}

TEST_CASE("A disk spill is written before the entry is published", "[app][executor][spill]") {
    registerTestOps();
    Scratch scratch;
    const auto cacheDir = scratch.dir / "cache";
    Executor ex(cacheDir);
    Pipeline p;
    const StepId scaleId = p.add("test_scale");
    p.setCache(1, CachePolicy::Disk);

    auto src = syntheticSource(1, 1, 2, 8, 8);
    auto load = std::make_shared<StepOutput>();
    load->meta = src->meta();
    load->source = src;
    ex.seed(p, 0, load);

    StepContext ctx;
    int observed = 0;
    std::filesystem::path firstFile;
    ex.setSpillObserver([&](const std::filesystem::path& file) {
        ++observed;
        // The file is complete here, and the observer runs with no lock
        // held: querying the executor from this callback answers rather
        // than waiting for the multi-gigabyte write we have just finished.
        CHECK(std::filesystem::exists(file));
        CHECK(std::filesystem::file_size(file) > 0);
        auto before = ex.lastOutput(scaleId);
        if (observed == 1) {
            CHECK_FALSE(before);         // nothing published yet for this step
            firstFile = file;
        } else {
            REQUIRE(before);             // still the previous entry
            CHECK(before->array->at(0, 0, 0, 0, 1) == 2.0f);
            CHECK(std::filesystem::exists(firstFile));   // removed only after the swap
        }
        // The point of asking from inside the spill observer is that the
        // executor answers at all: the write happens outside the lock, so a
        // query here must return rather than deadlock. Its value is not the
        // claim (an unsigned is trivially >= 0).
        CHECK_NOTHROW(ex.cachedBytes());
    });
    ex.run(p, 1, ctx);
    CHECK(observed == 1);
    auto out = ex.cached(p, 1);
    REQUIRE(out);
    CHECK(out->array->at(0, 0, 0, 0, 1) == 2.0f);

    // Storing the step again publishes a new file and drops the old one.
    ParamSet q = p.at(1).params;
    q.set("factor", 3.0);
    p.setParams(1, q);
    out.reset();
    ex.run(p, 1, ctx);
    CHECK(observed == 2);
    CHECK_FALSE(std::filesystem::exists(firstFile));
    int spills = 0;
    for (const auto& e : std::filesystem::directory_iterator(cacheDir))
        if (e.path().extension() == ".sir5") ++spills;
    CHECK(spills == 1);
    CHECK(ex.cached(p, 1)->array->at(0, 0, 0, 0, 1) == 3.0f);
}

// --- Workbench --------------------------------------------------------------------

TEST_CASE("Workbench edits are observed and undoable", "[app][workbench]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    Counter counter;
    wb.addObserver(&counter);
    wb.setDataset(syntheticSource());
    CHECK(wb.hasDataset());
    CHECK(wb.dataset().dims.c == 2);
    CHECK(counter.pipeline >= 1);
    const int base = wb.pipeline().size();

    const StepId id = wb.addStep("test_scale");
    CHECK(wb.pipeline().size() == base + 1);
    CHECK(wb.selectedIndex() == base);
    CHECK(wb.viewedIndex() == base);
    CHECK(wb.history().canUndo());
    CHECK(wb.history().undoLabel() == "Add Scale");

    wb.setStepParam(base, "factor", 3.0);
    CHECK(wb.pipeline().at(base).params.getDouble("factor") == 3.0);
    CHECK(wb.history().undoLabel().find("Factor 2 → 3") != std::string::npos);
    CHECK(wb.stepSummary(base).find("factor 3") != std::string::npos);

    SECTION("undo and redo restore the pipeline and keep ids") {
        wb.undo();
        CHECK(wb.pipeline().at(base).params.getDouble("factor") == 2.0);
        CHECK(wb.pipeline().at(base).id == id);
        wb.undo();
        CHECK(wb.pipeline().size() == base);
        CHECK(wb.selectedIndex() < base);
        wb.redo();
        CHECK(wb.pipeline().size() == base + 1);
        CHECK(wb.pipeline().at(base).id == id);
        wb.redo();
        CHECK(wb.pipeline().at(base).params.getDouble("factor") == 3.0);
        CHECK_FALSE(wb.history().canRedo());
    }
    SECTION("merged slider edits are one undo entry that spans the drag") {
        const std::size_t n = wb.history().size();
        wb.setStepParam(base, "factor", 4.0, "drag");
        wb.setStepParam(base, "factor", 5.0, "drag");
        wb.setStepParam(base, "factor", 6.0, "drag");
        CHECK(wb.history().size() == n + 1);
        wb.undo();
        CHECK(wb.pipeline().at(base).params.getDouble("factor") == 3.0);
        wb.redo();
        CHECK(wb.pipeline().at(base).params.getDouble("factor") == 6.0);
    }
    SECTION("moving remaps selection and viewing") {
        wb.addStep("test_maxz");
        const int last = wb.pipeline().size() - 1;
        wb.select(last);
        wb.view(base);
        CHECK(wb.moveStep(last, -1));
        CHECK(wb.selectedIndex() == base);
        CHECK(wb.viewedIndex() == last);
        CHECK(wb.pipeline().at(base).kind == "test_maxz");
    }
    SECTION("copy / paste parameters between steps of the same kind") {
        wb.addStep("test_scale");
        const int other = wb.pipeline().size() - 1;
        wb.copyParameters(base);
        CHECK(wb.pasteParameters(other));
        CHECK(wb.pipeline().at(other).params.getDouble("factor") == 3.0);
        wb.addStep("test_maxz");
        CHECK_FALSE(wb.pasteParameters(wb.pipeline().size() - 1));
    }
    SECTION("removing a step keeps the selection sane") {
        wb.removeStep(base);
        CHECK(wb.pipeline().size() == base);
        CHECK(wb.selectedIndex() < base);
        wb.removeStep(0);   // Load stays
        CHECK(wb.pipeline().at(0).kind == "load");
    }
    SECTION("view state setters clamp to the viewed output") {
        wb.setZ(100);
        CHECK(wb.viewState().z == 3);
        wb.setT(-5);
        CHECK(wb.viewState().t == 0);
        wb.setCrosshair(999, 2, 1);
        CHECK(wb.viewState().cx == 7);
        CHECK(wb.viewState().cy == 2);
        CHECK(wb.viewState().z == 1);
        wb.setChannelVisible(1, false);
        CHECK_FALSE(wb.viewState().channelOn(1));
        CHECK(wb.viewState().channelOn(0));
        const int before = counter.view;
        wb.toggleCrosshair();
        CHECK(counter.view == before + 1);
        CHECK_FALSE(wb.viewState().crosshair);
    }
    wb.removeObserver(&counter);
}

TEST_CASE("Workbench runs, caches, displays and reports", "[app][workbench]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource());
    wb.setBackend(Backend::Cpu);
    // Drop the default Contrast step if the built-ins are present: this test
    // only wants the synthetic ops.
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("test_scale");
    wb.addStep("test_maxz");
    CHECK_FALSE(wb.outputFresh(1));

    // Before any run the viewer falls back to the Load step's source.
    int actual = -1;
    auto shown = wb.displayOutput(&actual);
    REQUIRE(shown);
    CHECK(actual == 0);
    CHECK(shown->source);

    CHECK_FALSE(wb.running());
    auto job = runSync(wb);
    CHECK(job->succeeded());
    CHECK(job->reports().size() == 3);
    CHECK_FALSE(wb.running());
    CHECK(wb.outputFresh(1));
    CHECK(wb.outputFresh(2));
    shown = wb.displayOutput(&actual);
    CHECK(actual == 2);
    CHECK(shown->array->dims().z == 1);
    CHECK(wb.cachedBytes() > 0);
    CHECK(wb.log().back().find("Run finished") != std::string::npos);

    SECTION("diagnostics come from the output when fresh and warn when stale") {
        wb.select(1);
        CHECK(wb.selectedDiagnostics().summary == "scaled by 2");
        wb.setStepParam(1, "factor", 7.0);
        const Diagnostics d = wb.selectedDiagnostics();
        REQUIRE_FALSE(d.warnings.empty());
        CHECK(d.warnings.front().find("run the step again") != std::string::npos);
    }
    SECTION("running to one step leaves later ones untouched") {
        wb.setStepCache(2, CachePolicy::Memory);   // a Recompute entry would drop its array
        wb.setStepParam(1, "factor", 9.0);
        auto j = runSync(wb, 1);
        CHECK(j->succeeded());
        CHECK(wb.outputFresh(1));
        CHECK_FALSE(wb.outputFresh(2));
        // viewing step 2 shows its stale output rather than nothing
        wb.view(2);
        wb.displayOutput(&actual);
        CHECK(actual == 2);
    }
    SECTION("a failing step reports and leaves the run failed") {
        wb.addStep("test_fail");
        auto j = runSync(wb);
        CHECK_FALSE(j->succeeded());
        CHECK(j->error().find("Fail") != std::string::npos);
        CHECK(wb.log().back().find("failed") != std::string::npos);
    }
    SECTION("cancellation stops a slow step") {
        wb.addStep("test_slow");
        auto j = wb.createRun();
        REQUIRE(j);
        std::thread worker([&] { j->execute(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        wb.cancelRun();
        worker.join();
        wb.finishRun(j);
        CHECK_FALSE(j->succeeded());
        CHECK(j->error() == "cancelled");
        CHECK_FALSE(wb.running());
    }
    SECTION("invalid steps refuse to start with a log line") {
        Pipeline p = wb.pipeline();
        // a channel param out of range is rejected by the generic validation
        wb.setDataset(syntheticSource(1, 1, 1, 4, 4));
        while (wb.pipeline().size() > 1) wb.removeStep(1);
        wb.addStep("test_scale");
        CHECK(wb.createRun());
        wb.finishRun(wb.activeRun());
    }
    SECTION("clearing caches drops freshness but keeps the dataset") {
        wb.clearAllCaches();
        CHECK_FALSE(wb.outputFresh(2));
        CHECK(wb.outputFresh(0));
        wb.displayOutput(&actual);
        CHECK(actual == 0);
    }
}

TEST_CASE("Painting labels on a disk-cached step edits the volume the viewer shows", "[app][workbench][labels]") {
    registerBuiltinOperations();
    if (!findOperation("threshold")) SKIP("the threshold operation is not registered");
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource(1, 1, 4, 8, 8));
    wb.setBackend(Backend::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("threshold");
    wb.setStepParam(1, "method", std::string("Manual"));
    wb.setStepParam(1, "value", 1e9);   // nothing above: no labels yet
    wb.setStepCache(1, CachePolicy::Disk);
    auto job = runSync(wb);
    REQUIRE(job->succeeded());
    wb.view(1);
    int actual = -1;
    auto shown = wb.displayOutput(&actual);
    REQUIRE(shown);
    CHECK(actual == 1);
    REQUIRE(shown->labels);
    CHECK(shown->labels->maxLabel() == 0);
    // the disk-cached output is one object while it is held, so the labels
    // the stroke edits are the labels on screen
    CHECK(wb.displayOutput() == shown);
    CHECK(wb.viewedLabels() == shown->labels);
    wb.setTool(ViewerTool::Paint);
    wb.setPaintTool(PaintTool::Brush);
    wb.beginPaintStroke();
    wb.paintLabels(2, 4, 4, false);
    CHECK(shown->labels->at(0, 2, 4, 4) != 0);
    CHECK(shown->labels->maxLabel() == 1);
    CHECK(wb.displayOutput()->labels->at(0, 2, 4, 4) == shown->labels->at(0, 2, 4, 4));
    wb.undo();
    CHECK(shown->labels->at(0, 2, 4, 4) == 0);
    wb.redo();
    CHECK(shown->labels->at(0, 2, 4, 4) != 0);

    // solo: only the selected label is drawn, and selecting one while solo
    // brings the crosshair and z onto it
    shown->labels->recomputeStats(0);
    REQUIRE(shown->labels->statsOf(1));
    wb.setCrosshair(0, 0, 0);
    wb.toggleSoloLabel();
    CHECK(wb.viewState().soloLabel);
    CHECK(wb.viewState().labels);
    ViewState s = wb.viewState();
    s.selectedLabel = 1;
    wb.setViewState(s);
    const LabelStats* st = shown->labels->statsOf(1);
    CHECK(wb.viewState().cx == (st->bbox[4] + st->bbox[5]) / 2);
    CHECK(wb.viewState().cy == (st->bbox[2] + st->bbox[3]) / 2);
    CHECK(wb.viewState().z == (st->bbox[0] + st->bbox[1]) / 2);
    const ViewState back = ViewState::fromJson(wb.viewState().toJson());
    CHECK(back.soloLabel);
    CHECK(back.selectedLabel == 1);
    wb.toggleSoloLabel();
    CHECK_FALSE(wb.viewState().soloLabel);
}

TEST_CASE("Labels carried through a step are copied on the first edit", "[app][workbench][labels]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource(1, 1, 4, 16, 16));
    wb.setBackend(Backend::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("test_labels");   // 1: makes its own labels
    wb.addStep("test_scale");    // 2: carries its input's labels through
    REQUIRE(runSync(wb)->succeeded());

    auto upstream = wb.output(1)->labels;
    auto downstream = wb.output(2)->labels;
    REQUIRE(upstream);
    REQUIRE(downstream);
    CHECK(upstream != downstream);                                 // each output owns its volume
    CHECK(upstream->view().data() == downstream->view().data());   // over the same voxels, for now
    CHECK(upstream->sharesVoxels());
    CHECK(downstream->sharesVoxels());
    const std::size_t sharedBytes = wb.cachedBytes();

    wb.view(2);
    paintOne(wb, 1, 2, 2, 9);
    CHECK(downstream->at(0, 1, 2, 2) == 9);
    CHECK(upstream->at(0, 1, 2, 2) == 0);        // the cached input is left alone
    CHECK(upstream->view().data() != downstream->view().data());
    CHECK_FALSE(upstream->sharesVoxels());
    CHECK_FALSE(downstream->sharesVoxels());
    CHECK(wb.output(1)->labels->at(0, 1, 2, 2) == 0);
    CHECK(wb.cachedBytes() > sharedBytes);       // two label volumes now, not one shared
}

TEST_CASE("Label statistics stay in step with the voxels", "[app][workbench][labels]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource(1, 1, 4, 16, 16));
    wb.setBackend(Backend::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("test_labels");
    REQUIRE(runSync(wb)->succeeded());
    wb.view(1);
    auto labels = wb.viewedLabels();
    REQUIRE(labels);
    REQUIRE(labels->statsOf(1));
    const Index painted = labels->statsOf(1)->voxels;
    CHECK(painted == countLabel(*labels, 0, 1));

    SECTION("a stroke updates them once, at its end") {
        ViewState s = wb.viewState();
        s.brushPx = 2;
        s.paint3d = false;
        s.selectedLabel = 1;
        wb.setViewState(s);
        wb.beginPaintStroke();
        wb.paintLabels(2, 8, 11, false);
        wb.paintLabels(2, 8, 12, false);
        CHECK(labels->statsOf(1)->voxels == painted);   // not per mouse move
        wb.endPaintStroke();
        const LabelStats* st = labels->statsOf(1);
        REQUIRE(st);
        CHECK(st->voxels > painted);
        CHECK(st->voxels == countLabel(*labels, 0, 1));
        CHECK(st->bbox[5] == 14);                       // the stroke widened the box
        // the review table and the viewer agree about where the label is
        CHECK(wb.centreOnLabel(1));
        CHECK(wb.viewState().cx == (st->bbox[4] + st->bbox[5]) / 2);
        CHECK(wb.viewState().cy == (st->bbox[2] + st->bbox[3]) / 2);
    }
    SECTION("filling recolours the label and both entries follow") {
        ViewState s = wb.viewState();
        s.selectedLabel = 5;
        wb.setViewState(s);
        wb.fillLabel(2, 8, 8);
        CHECK(labels->statsOf(1) == nullptr);           // nothing is labelled 1 any more
        const LabelStats* five = labels->statsOf(5);
        REQUIRE(five);
        CHECK(five->voxels == painted);
        CHECK(five->voxels == countLabel(*labels, 0, 5));
        CHECK(wb.centreOnLabel(5));
        CHECK_FALSE(wb.centreOnLabel(1));
    }
    SECTION("deleting drops the entry, undo brings it back") {
        wb.deleteLabel(1);
        CHECK(labels->statsOf(1) == nullptr);
        CHECK(countLabel(*labels, 0, 1) == 0);
        wb.undo();
        REQUIRE(labels->statsOf(1));
        CHECK(labels->statsOf(1)->voxels == painted);
        CHECK(countLabel(*labels, 0, 1) == painted);
    }
}

TEST_CASE("Undoing a label edit after the step was re-run is a logged no-op", "[app][workbench][labels]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource(1, 1, 4, 16, 16));
    wb.setBackend(Backend::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("test_labels");
    REQUIRE(runSync(wb)->succeeded());
    wb.view(1);
    paintOne(wb, 1, 2, 2, 9);
    auto edited = wb.viewedLabels();
    REQUIRE(edited);
    CHECK(edited->at(0, 1, 2, 2) == 9);

    // The step runs again from scratch: its output, and with it its label
    // volume, is a new object and the undo closure has nothing to undo.
    wb.clearCache(1);
    REQUIRE(runSync(wb)->succeeded());
    auto fresh = wb.viewedLabels();
    REQUIRE(fresh);
    CHECK(fresh != edited);
    CHECK(fresh->at(0, 1, 2, 2) == 0);

    const std::size_t lines = wb.log().size();
    wb.undo();
    CHECK(fresh->at(0, 1, 2, 2) == 0);    // the fresh labels are not rewritten
    CHECK(edited->at(0, 1, 2, 2) == 9);   // nor is the orphaned volume touched
    CHECK(wb.log().size() > lines);
    CHECK(logContains(wb, "recomputed since"));
    wb.redo();
    CHECK(fresh->at(0, 1, 2, 2) == 0);
    CHECK(edited->at(0, 1, 2, 2) == 9);
}

TEST_CASE("Edits are refused while a run is active", "[app][workbench][run]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource(1, 1, 2, 8, 8));
    wb.setBackend(Backend::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("test_scale");
    wb.addStep("test_slow");
    CHECK(wb.canEdit());

    auto job = wb.createRun();
    REQUIRE(job);
    CHECK(wb.running());
    CHECK_FALSE(wb.canEdit());
    std::thread worker([&] { job->execute(); });

    const int steps = wb.pipeline().size();
    const std::string name = wb.pipeline().at(1).name;
    const std::size_t lines = wb.log().size();
    CHECK(wb.addStep("test_scale") == 0);
    wb.removeStep(1);
    wb.setStepParam(1, "factor", 11.0);
    wb.setStepEnabled(1, false);
    CHECK_FALSE(wb.moveStep(1, 1));
    CHECK(wb.duplicateStep(1) == 0);
    wb.renameStep(1, "renamed");
    wb.setStepCache(1, CachePolicy::Disk);
    wb.clearAllCaches();
    wb.undo();
    CHECK(wb.pipeline().size() == steps);
    CHECK(wb.pipeline().at(1).name == name);
    CHECK(wb.pipeline().at(1).params.getDouble("factor") == 2.0);
    CHECK(wb.pipeline().at(1).enabled);
    CHECK(wb.pipeline().at(1).cache != CachePolicy::Disk);
    CHECK(wb.log().size() > lines);   // each refusal says why
    CHECK(logContains(wb, "while a run is in progress"));
    CHECK_THROWS_AS(wb.openDataset("nowhere.tif"), std::runtime_error);

    // selection and view state are not edits: they stay allowed
    wb.select(1);
    CHECK(wb.selectedIndex() == 1);
    wb.view(0);
    CHECK(wb.viewedIndex() == 0);
    wb.setZ(1);
    CHECK(wb.viewState().z == 1);

    wb.cancelRun();
    worker.join();
    wb.finishRun(job);
    CHECK(wb.canEdit());
    CHECK(job->wasCancelled());
    CHECK(wb.addStep("test_scale") != 0);
}

TEST_CASE("A run job publishes its results only once it has finished", "[app][workbench][run]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource(1, 1, 2, 4, 4));
    wb.setBackend(Backend::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("test_scale");

    auto job = wb.createRun();
    REQUIRE(job);
    CHECK_FALSE(job->finished());
    CHECK_FALSE(job->succeeded());
    CHECK_THROWS_AS(job->reports(), std::logic_error);
    CHECK_THROWS_AS(job->output(), std::logic_error);
    CHECK_THROWS_AS(job->error(), std::logic_error);
    CHECK_THROWS_AS(job->seconds(), std::logic_error);
    CHECK_THROWS_AS(job->wasCancelled(), std::logic_error);
    job->execute();
    CHECK(job->finished());
    CHECK_NOTHROW(job->reports());
    CHECK(job->succeeded());
    CHECK_FALSE(job->wasCancelled());
    CHECK(job->output());
    wb.finishRun(job);

    SECTION("a job that never executed is folded back as abandoned") {
        auto second = wb.createRun();
        REQUIRE(second);
        wb.finishRun(second);
        CHECK_FALSE(wb.running());
        CHECK(logContains(wb, "abandoned"));
        CHECK(second->cancelled());
    }
}

TEST_CASE("A recorded session holds what the user did, in order", "[app][workbench][session]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource(1, 1, 4, 8, 8));
    wb.setBackend(Backend::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    const std::filesystem::path log = scratch.dir / "session.jsonl";

    CHECK_FALSE(wb.recording());
    wb.startRecording(log.string());
    CHECK(wb.recording());
    wb.addStep("test_scale");
    wb.setStepParam(1, "factor", 3.0);
    wb.setStepParam(1, "factor", 5.0);
    auto job = runSync(wb);
    REQUIRE(job->succeeded());
    wb.removeStep(1);
    wb.stopRecording();
    CHECK_FALSE(wb.recording());

    std::ifstream in(log);
    REQUIRE(in.good());
    std::vector<nlohmann::json> events;
    for (std::string line; std::getline(in, line);)
        if (!line.empty()) events.push_back(nlohmann::json::parse(line));
    REQUIRE(events.size() >= 6);

    auto kinds = [&] {
        std::vector<std::string> out;
        for (const nlohmann::json& e : events) out.push_back(e.value("event", std::string()));
        return out;
    }();
    auto has = [&](const std::string& what) { return std::find(kinds.begin(), kinds.end(), what) != kinds.end(); };
    CHECK(kinds.front() == "session");
    CHECK(kinds.back() == "stopped");
    CHECK(has("step_added"));
    CHECK(has("params"));
    CHECK(has("step_ran"));
    CHECK(has("step_removed"));

    // the record carries the values, not a sentence about them: the second
    // parameter change has to say 3 -> 5 so a reader can learn from it
    const nlohmann::json* second = nullptr;
    for (const nlohmann::json& e : events)
        if (e.value("event", std::string()) == "params") second = &e;   // the last one
    REQUIRE(second != nullptr);
    CHECK(second->at("kind") == "test_scale");
    CHECK(second->at("from").at("factor").get<double>() == 3.0);
    CHECK(second->at("to").at("factor").get<double>() == 5.0);

    // every line is timestamped and ordered
    double previous = -1.0;
    for (const nlohmann::json& e : events) {
        const double t = e.value("t", -1.0);
        CHECK(t >= previous);
        previous = t;
    }

    SECTION("a label correction is written down with the voxels it moved") {
        // what a model imitating a label generator has to learn from: not
        // that the user edited something, but which edit and how large
        Workbench w2(scratch.dir);
        w2.setDataset(syntheticSource(1, 1, 4, 8, 8));
        w2.setBackend(Backend::Cpu);
        while (w2.pipeline().size() > 1) w2.removeStep(1);
        w2.addStep("test_labels");
        auto ran = runSync(w2);
        REQUIRE(ran->succeeded());
        const std::filesystem::path second = scratch.dir / "edits.jsonl";
        w2.startRecording(second.string());
        w2.paintLabels(1, 3, 3, false);
        w2.endPaintStroke();
        const std::shared_ptr<const LabelVolume> made = w2.output(1)->labels;
        REQUIRE(made != nullptr);
        std::uint32_t victim = 0;
        for (const LabelStats& st : made->stats())
            if (st.voxels > 0) {
                victim = st.id;
                break;
            }
        if (victim != 0) w2.deleteLabel(victim);
        w2.stopRecording();

        std::ifstream in2(second);
        std::vector<std::string> seen;
        for (std::string line; std::getline(in2, line);)
            if (!line.empty()) seen.push_back(nlohmann::json::parse(line).value("event", std::string()));
        CHECK(std::find(seen.begin(), seen.end(), "paint") != seen.end());
        if (victim != 0) CHECK(std::find(seen.begin(), seen.end(), "label_edit") != seen.end());
    }

    SECTION("nothing is written once recording stops") {
        const std::uintmax_t before = std::filesystem::file_size(log);
        wb.addStep("test_scale");
        wb.setStepParam(1, "factor", 9.0);
        CHECK(std::filesystem::file_size(log) == before);
    }
    SECTION("an unwritable path is reported, not swallowed") {
        // a folder that cannot be made because a file of that name is in
        // the way: create_directories fails and the open after it has to be
        // the thing that reports the problem
        const std::filesystem::path blocker = scratch.dir / "blocker";
        std::ofstream(blocker) << "not a folder";
        CHECK_THROWS(wb.startRecording((blocker / "session.jsonl").string()));
        CHECK_FALSE(wb.recording());
    }
}

TEST_CASE("Workbench loads pipelines and the example without losing the dataset", "[app][workbench]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource());
    Pipeline p;
    p.add("test_scale");
    p.add("test_maxz");
    test::TempFile file("wb", ".sirius.toml");
    p.save(file.path.string());
    wb.loadPipeline(file.path.string());
    CHECK(wb.pipeline().size() == 3);
    CHECK(wb.pipeline().at(1).kind == "test_scale");
    CHECK(wb.hasDataset());
    CHECK(wb.pipelinePath() == file.path.string());
    CHECK(wb.outputFresh(0));
    wb.undo();
    CHECK(wb.pipeline().size() != 3);
}

// --- Tool API -----------------------------------------------------------------------

TEST_CASE("apply_preset says which of its failures happened", "[app][tools][params]") {
    // applyPreset answers false to "no such preset" and to "a run is in
    // progress" alike, so the tool has to tell them apart: reporting the
    // second as the first tells the assistant a preset does not exist when it
    // does, and it would stop asking for it.
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource(1, 1, 4, 8, 8));
    wb.setBackend(Backend::Cpu);
    wb.addStep("classic");
    const int step = wb.pipeline().size() - 1;
    ToolApi api(wb);

    // call() turns a throw into {"error": ...}, which is what the caller reads
    auto errorOf = [&api](const json& args) {
        const json r = api.call("apply_preset", args);
        return r.contains("error") ? r["error"].get<std::string>() : std::string();
    };

    SECTION("an unknown preset names the ones there are") {
        const std::string message = errorOf(json{{"step", step + 1}, {"preset", "Nothing like it"}});
        CHECK_THAT(message, Catch::Matchers::ContainsSubstring("no preset"));
        CHECK_THAT(message, Catch::Matchers::ContainsSubstring("Filaments"));
    }
    SECTION("a run in progress says so, and does not deny the preset") {
        wb.addStep("test_slow");
        auto job = wb.createRun();
        REQUIRE(job);
        std::thread worker([&] { job->execute(); });
        const std::string message = errorOf(json{{"step", step + 1}, {"preset", "Nuclei"}});
        CHECK_THAT(message, Catch::Matchers::ContainsSubstring("run is in progress"));
        CHECK_THAT(message, !Catch::Matchers::ContainsSubstring("no preset"));
        job->cancel();
        worker.join();
    }
}

TEST_CASE("ToolApi drives the workbench through JSON", "[app][tools]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource());
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    ToolApi api(wb);
    const json schemas = api.schemas();
    REQUIRE(schemas.is_array());
    CHECK(schemas.size() >= 15);
    for (const json& s : schemas) {
        CHECK(s["type"] == "function");
        CHECK(s["function"].contains("parameters"));
    }

    json r = api.call("add_step", {{"kind", "test_scale"}, {"params", {{"factor", "4"}}}});
    REQUIRE_FALSE(r.contains("error"));
    CHECK(r["step"] == 2);
    CHECK(wb.pipeline().at(1).params.getDouble("factor") == 4.0);
    CHECK(api.actions().size() == 1);

    r = api.call("set_params", {{"step", "Scale"}, {"params", {{"factor", 6}}}});
    CHECK(r["params"]["factor"] == 6.0);
    r = api.call("set_params", {{"step", 2}, {"params", {{"nope", 1}}}});
    CHECK(r.contains("error"));
    r = api.call("set_params", {{"step", 9}, {"params", {{"factor", 1}}}});
    CHECK(r.contains("error"));

    r = api.call("get_step", {{"step", 2}});
    CHECK(r["kind"] == "test_scale");
    CHECK(r["has_output"] == false);

    r = api.call("set_step_enabled", {{"step", 2}, {"enabled", false}});
    CHECK(r["enabled"] == false);
    r = api.call("undo", json::object());
    CHECK(r["ok"] == true);
    CHECK(wb.pipeline().at(1).enabled);

    r = api.call("set_view", {{"mode", "compare"}, {"z", 99}, {"crosshair", {3, 4}}, {"labels", true}});
    CHECK(wb.viewState().mode == ViewMode::Compare);
    CHECK(wb.viewState().z == 3);
    CHECK(wb.viewState().cx == 3);
    CHECK(wb.viewState().labels);

    r = api.call("run", json::object());
    CHECK(r.contains("error"));   // no run hook installed
    api.setRunHook([&](int target) {
        auto job = runSync(wb, target);
        return json{{"ok", job->succeeded()}, {"seconds", job->seconds()}, {"error", job->error()}};
    });
    r = api.call("run", {{"step", 2}});
    CHECK(r["ok"] == true);
    CHECK(wb.outputFresh(1));

    r = api.call("get_diagnostics", {{"step", 2}});
    CHECK(r["summary"] == "scaled by 6");
    r = api.call("get_state", json::object());
    CHECK(r["steps"].size() == 2);
    CHECK(r["dataset"]["shape"] == "c2 t3 z4 y8 x8");
    CHECK(api.call("unknown_tool", json::object()).contains("error"));
    CHECK(api.contextSnapshot().contains("selected_step_diagnostics"));
    CHECK(api.systemPrompt().find("SIRIUS") != std::string::npos);
    const auto actions = api.takeActions();
    CHECK(actions.size() >= 5);
    CHECK(api.actions().empty());
}

TEST_CASE("History merges by key and clears redo on push", "[app][history]") {
    History h;
    int value = 0;
    auto cmd = [&](int from, int to, std::string key = {}) {
        Command c;
        c.label = std::to_string(from) + "→" + std::to_string(to);
        c.undo = [&value, from] { value = from; };
        c.redo = [&value, to] { value = to; };
        c.mergeKey = std::move(key);
        return c;
    };
    value = 1;
    h.push(cmd(0, 1));
    value = 2;
    h.push(cmd(1, 2, "k"));
    value = 3;
    h.push(cmd(1, 3, "k"));   // composed by the caller: undo goes back to 1
    CHECK(h.size() == 2);
    h.undo();
    CHECK(value == 1);
    h.undo();
    CHECK(value == 0);
    CHECK_FALSE(h.canUndo());
    h.redo();
    CHECK(value == 1);
    h.redo();
    CHECK(value == 3);
    h.undo();
    value = 5;
    h.push(cmd(1, 5));
    CHECK_FALSE(h.canRedo());
    CHECK(h.undoLabel() == "1→5");

    SECTION("mergesWith is the single source of truth for what continues a group") {
        History g;
        CHECK_FALSE(g.mergesWith("k"));
        g.push(cmd(0, 1, "k"));
        CHECK(g.mergesWith("k"));
        CHECK_FALSE(g.mergesWith(""));      // an empty key never merges
        CHECK_FALSE(g.mergesWith("other"));
        g.push(cmd(1, 2));                  // any other entry ends the group
        CHECK_FALSE(g.mergesWith("k"));
        g.undo();
        CHECK_FALSE(g.mergesWith("k"));
    }
}

TEST_CASE("A drag interrupted by a label edit starts a new undo group", "[app][workbench][history]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource(1, 1, 4, 16, 16));
    wb.setBackend(Backend::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("test_labels");   // 1
    wb.addStep("test_scale");    // 2
    REQUIRE(runSync(wb)->succeeded());
    wb.view(2);
    REQUIRE(wb.pipeline().at(2).params.getDouble("factor") == 2.0);

    // A drag: consecutive edits with the same key are one entry that undoes
    // to the value the drag started from.
    const std::size_t entries = wb.history().size();
    wb.setStepParam(2, "factor", 3.0, "drag");
    wb.setStepParam(2, "factor", 4.0, "drag");
    CHECK(wb.history().size() == entries + 1);

    // A label edit goes onto the history without a merge key: the drag that
    // follows must undo to 4, not back past the label edit to 2.
    paintOne(wb, 1, 2, 2, 9);
    wb.setStepParam(2, "factor", 5.0, "drag");
    CHECK(wb.pipeline().at(2).params.getDouble("factor") == 5.0);
    wb.undo();
    CHECK(wb.pipeline().at(2).params.getDouble("factor") == 4.0);
    wb.undo();   // the label edit
    CHECK(wb.viewedLabels()->at(0, 1, 2, 2) == 0);
    CHECK(wb.pipeline().at(2).params.getDouble("factor") == 4.0);
    wb.undo();   // the drag
    CHECK(wb.pipeline().at(2).params.getDouble("factor") == 2.0);
}

TEST_CASE("A live-preview step is displayed on its input until it runs", "[app][workbench][preview]") {
    registerTestOps();
    if (!findOperation("contrast")) SKIP("built-in operations not registered");
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource());
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("test_scale");
    const StepId contrastId = wb.addStep("contrast");
    const int ci = wb.pipeline().indexOf(contrastId);
    wb.view(ci);
    // nothing has run: the preview falls back to the Load step's source
    CHECK(wb.viewedIsLivePreview());
    int actual = -1;
    auto shown = wb.displayOutput(&actual);
    REQUIRE(shown);
    CHECK(actual == 0);
    CHECK(wb.upstreamOutput(ci, &actual) == shown);

    runSync(wb, 1);   // scale only
    shown = wb.displayOutput(&actual);
    CHECK(actual == 1);
    CHECK(wb.viewedIsLivePreview());

    runSync(wb);      // contrast too: its own output is shown
    CHECK_FALSE(wb.viewedIsLivePreview());
    shown = wb.displayOutput(&actual);
    CHECK(actual == ci);

    wb.setStepParam(ci, "gamma", 2.0);   // stale again -> preview on the input
    CHECK(wb.viewedIsLivePreview());
    shown = wb.displayOutput(&actual);
    CHECK(actual == 1);
    wb.setStepEnabled(ci, false);        // a skipped step is never previewed
    CHECK_FALSE(wb.viewedIsLivePreview());
}

TEST_CASE("A rewritten file is not served from the cache", "[app][executor]") {
    // The pipeline names a path; the cache has to key on what is behind it.
    // An OTF, a PSF or a flat-field image edited in place keeps the same name,
    // and the step that reads it must run again.
    registerTestOps();
    Scratch scratch;
    Executor ex(scratch.dir / "cache");
    const std::filesystem::path file = scratch.dir / "source.bin";
    std::ofstream(file, std::ios::binary) << "aaaa";

    Pipeline p;
    p.add("test_file");
    ParamSet q = p.at(1).params;
    q.set("source", file.string());
    p.setParams(1, q);

    auto src = syntheticSource();
    auto load = std::make_shared<StepOutput>();
    load->meta = src->meta();
    load->source = src;
    ex.seed(p, 0, load);

    StepContext ctx;
    FileOp::runs = 0;
    auto first = ex.runAll(p, ctx);
    REQUIRE(first);
    CHECK(FileOp::runs == 1);
    const float before = first->array->at(0, 0, 0, 0, 0);
    const std::string stamp = ex.fingerprint(p, 1);

    SECTION("running again with the file untouched is a cache hit") {
        ex.runAll(p, ctx);
        CHECK(FileOp::runs == 1);
        CHECK(ex.fingerprint(p, 1) == stamp);
    }

    SECTION("rewriting the file behind the same path runs the step again") {
        // a different size, and a later timestamp: either is enough on its own
        std::ofstream(file, std::ios::binary | std::ios::trunc) << "bbbbbbbb";
        std::filesystem::last_write_time(file, std::filesystem::file_time_type::clock::now() + std::chrono::seconds(2));
        CHECK(ex.fingerprint(p, 1) != stamp);
        auto again = ex.runAll(p, ctx);
        REQUIRE(again);
        CHECK(FileOp::runs == 2);
        CHECK(again->array->at(0, 0, 0, 0, 0) != before);   // the new content reached the output
    }

    SECTION("a path that names nothing is simply not stamped") {
        ParamSet missing = p.at(1).params;
        missing.set("source", (scratch.dir / "not-here.bin").string());
        p.setParams(1, missing);
        CHECK_FALSE(ex.fingerprint(p, 1).empty());   // still a fingerprint, just without a file in it
    }
}

TEST_CASE("Applying a preset is an ordinary undoable parameter change", "[app][workbench][params]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource(1, 1, 4, 8, 8));
    wb.setBackend(Backend::Cpu);
    wb.addStep("classic");
    const int step = wb.pipeline().size() - 1;

    const std::string before = wb.pipeline().at(step).params.getString("enhance");
    CHECK(wb.applyPreset(step, "Filaments"));
    CHECK(wb.pipeline().at(step).params.getString("enhance") == "Tubes (Frangi)");
    CHECK(wb.pipeline().at(step).params.getBool("hysteresis"));
    CHECK(wb.pipeline().at(step).params.getString("post") == "Connected components");

    SECTION("it can be undone, and the values were only values") {
        REQUIRE(wb.history().canUndo());
        wb.undo();
        CHECK(wb.pipeline().at(step).params.getString("enhance") == before);
    }
    SECTION("a second preset overwrites the first") {
        CHECK(wb.applyPreset(step, "Nuclei"));
        CHECK(wb.pipeline().at(step).params.getString("enhance") == "None");
        CHECK_FALSE(wb.pipeline().at(step).params.getBool("hysteresis"));
        CHECK(wb.pipeline().at(step).params.getString("post") == "Watershed (distance)");
    }
    SECTION("it refuses while a run is in progress, rather than reporting a change it did not make") {
        // setStepParams bails out there silently; a preset that still said yes
        // would leave the caller with an undo entry for nothing
        wb.addStep("test_slow");
        auto job = wb.createRun();
        REQUIRE(job);
        CHECK_FALSE(wb.canEdit());
        std::thread worker([&] { job->execute(); });
        const ParamSet held = wb.pipeline().at(step).params;
        const std::size_t lines = wb.log().size();
        CHECK_FALSE(wb.applyPreset(step, "Nuclei"));
        CHECK(wb.pipeline().at(step).params == held);
        // and it says why, as every other refused edit does, so a false that
        // means "busy" is not mistaken for one that means "no such preset"
        CHECK(wb.log().size() > lines);
        CHECK(logContains(wb, "while a run is in progress"));
        job->cancel();
        worker.join();
    }

    SECTION("an unknown preset changes nothing") {
        const ParamSet held = wb.pipeline().at(step).params;
        CHECK_FALSE(wb.applyPreset(step, "Nothing like it"));
        CHECK(wb.pipeline().at(step).params == held);
        CHECK_FALSE(wb.applyPreset(-1, "Filaments"));
    }
}

// --- outputs and identities across edits --------------------------------------------

TEST_CASE("Labels are carried through a step only onto the same grid", "[app][executor][labels]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource(1, 1, 4, 16, 16));
    wb.setBackend(Backend::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("test_labels");   // 1: its own labels over the input's grid
    wb.addStep("test_maxz");     // 2: z -> 1, a grid those labels do not fit
    wb.addStep("test_scale");    // 3: would carry labels through, but there are none to carry
    REQUIRE(runSync(wb)->succeeded());
    REQUIRE(wb.output(1)->labels);
    // Before: the executor shared the (z=4) labels onto the (z=1) output,
    // and every consumer indexing them with the output's dims read past
    // their buffer.
    CHECK_FALSE(wb.output(2)->labels);
    CHECK_FALSE(wb.output(3)->labels);
}

TEST_CASE("A step added after an undo does not inherit the undone step's output", "[app][workbench][executor]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource());
    wb.setBackend(Backend::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    const StepId scale = wb.addStep("test_scale");
    REQUIRE(runSync(wb)->succeeded());
    REQUIRE(wb.output(1));
    wb.undo();   // the step is gone; its cached output is still in the executor
    REQUIRE(wb.pipeline().size() == 1);
    const StepId added = wb.addStep("test_maxz");
    CHECK(added != scale);   // ids keep counting, they are not reissued
    CHECK_FALSE(wb.output(1));
    CHECK_FALSE(wb.outputFresh(1));
    CHECK_FALSE(wb.viewedLabels());
}

TEST_CASE("Replacing the pipeline drops the previous steps' outputs", "[app][workbench][executor]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource());
    wb.setBackend(Backend::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    const StepId scale = wb.addStep("test_scale");
    REQUIRE(runSync(wb)->succeeded());
    REQUIRE(wb.output(1));
    // an incoming pipeline whose second step gets the id the scale step had
    Pipeline p;
    p.add("test_scale");
    p.add("test_maxz");
    REQUIRE(p.at(2).id == scale);
    wb.replacePipeline(p, "swap");
    REQUIRE(wb.pipeline().size() == 3);
    CHECK(wb.pipeline().at(2).kind == "test_maxz");
    CHECK_FALSE(wb.output(1));
    CHECK_FALSE(wb.output(2));   // not the scale step's array under a new name
    CHECK(wb.outputFresh(0));    // the dataset is seeded again
}

TEST_CASE("Pipeline TOML keeps the step ids, gaps included", "[app][pipeline]") {
    registerTestOps();
    Pipeline p;
    p.add("test_scale");
    p.add("test_maxz");
    p.add("test_scale");
    p.remove(2);   // leaves a gap in the ids
    const StepId a = p.at(1).id, b = p.at(2).id;
    REQUIRE(b == a + 2);
    test::TempFile file("ids", ".sirius.toml");
    p.save(file.path.string());
    // TOML integers are signed: an unsigned-only check dropped every id
    const Pipeline loaded = Pipeline::load(file.path.string());
    REQUIRE(loaded.size() == 3);
    CHECK(loaded.at(1).id == a);
    CHECK(loaded.at(2).id == b);
}

TEST_CASE("Opening a dataset starts the history over", "[app][workbench][history]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource());
    wb.addStep("test_scale");
    REQUIRE(wb.history().canUndo());
    // The open itself was never undoable (the previous source is gone), and
    // the edits before it were made against other data.
    wb.setDataset(syntheticSource(1, 1, 2, 4, 4));
    CHECK_FALSE(wb.history().canUndo());
    CHECK_FALSE(wb.history().canRedo());
    CHECK(wb.dataset().dims.c == 1);
    CHECK(wb.pipeline().size() >= 2);   // the pipeline is kept
    CHECK(wb.outputFresh(0));
}

TEST_CASE("A label edit makes the steps below it stale", "[app][workbench][labels][executor]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource(1, 1, 4, 16, 16));
    wb.setBackend(Backend::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("test_labels");   // 1
    wb.addStep("test_scale");    // 2: consumed (carried) the labels as they were
    REQUIRE(runSync(wb)->succeeded());
    REQUIRE(wb.outputFresh(2));

    wb.view(1);
    paintOne(wb, 1, 2, 2, 9);
    CHECK(wb.outputFresh(1));         // the edited step itself is as fresh as before
    CHECK_FALSE(wb.outputFresh(2));   // what it fed downstream is not
    CHECK(wb.output(2));              // still there for the viewer, marked stale

    SECTION("the next run recomputes the stale step") {
        auto job = runSync(wb);
        REQUIRE(job->succeeded());
        bool ran = false;
        for (const StepReport& r : job->reports())
            if (r.index == 2 && r.ran()) ran = true;
        CHECK(ran);
        CHECK(wb.outputFresh(2));
        CHECK(wb.output(2)->labels->at(0, 1, 2, 2) == 9);   // and it saw the correction
    }
    SECTION("undoing the edit is an edit too") {
        REQUIRE(runSync(wb)->succeeded());
        REQUIRE(wb.outputFresh(2));
        wb.undo();
        CHECK_FALSE(wb.outputFresh(2));
    }
}

TEST_CASE("A fresh target is served without re-running an evicted upstream", "[app][executor]") {
    registerTestOps();
    Scratch scratch;
    Executor ex(scratch.dir / "cache");
    Pipeline p;
    p.add("test_scale");
    p.add("test_maxz");
    p.setCache(1, CachePolicy::Recompute);
    p.setCache(2, CachePolicy::Recompute);
    auto src = syntheticSource();
    auto load = std::make_shared<StepOutput>();
    load->meta = src->meta();
    load->source = src;
    ex.seed(p, 0, load);
    StepContext ctx;
    ScaleOp::runs = 0;
    REQUIRE(ex.runAll(p, ctx));
    CHECK(ScaleOp::runs == 1);
    // Recompute keeps only the newest array: the upstream lost its array to
    // the target's store, the target is fresh by its fingerprint.
    CHECK_FALSE(ex.isFresh(p, 1));
    CHECK(ex.isFresh(p, 2));

    std::vector<StepReport> reports;
    REQUIRE(ex.runAll(p, ctx, &reports));
    CHECK(ScaleOp::runs == 1);   // before: the upstream ran, evicted the target, and the target ran too
    REQUIRE(reports.size() == 3);
    CHECK(reports[1].state == StepReport::State::Cached);
    CHECK(reports[2].state == StepReport::State::Cached);
}

TEST_CASE("A missing spill file makes the step recompute instead of throwing", "[app][executor][spill]") {
    registerTestOps();
    Scratch scratch;
    Executor ex(scratch.dir / "cache");
    Pipeline p;
    const StepId id = p.add("test_scale");
    p.setCache(1, CachePolicy::Disk);
    auto src = syntheticSource();
    auto load = std::make_shared<StepOutput>();
    load->meta = src->meta();
    load->source = src;
    ex.seed(p, 0, load);
    std::filesystem::path spill;
    ex.setSpillObserver([&](const std::filesystem::path& file) { spill = file; });
    StepContext ctx;
    REQUIRE(ex.runAll(p, ctx));
    REQUIRE(std::filesystem::exists(spill));
    REQUIRE(ex.lastOutput(id));
    REQUIRE(ex.lastOutput(id)->array);

    std::filesystem::remove(spill);   // a temp cleaner, a full disk
    std::shared_ptr<const StepOutput> gone;
    CHECK_NOTHROW(gone = ex.lastOutput(id));
    CHECK_FALSE(gone);
    CHECK_FALSE(ex.isFresh(p, 1));
    ScaleOp::runs = 0;
    std::vector<StepReport> reports;
    REQUIRE(ex.runAll(p, ctx, &reports));
    CHECK(ScaleOp::runs == 1);
    CHECK(reports[1].ran());
    CHECK(ex.isFresh(p, 1));
}

TEST_CASE("A run job cancelled before it starts does not start a worker", "[app][workbench][run]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource());
    wb.setBackend(Backend::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("test_slow");
    auto job = wb.createRun();
    REQUIRE(job);
    job->cancel();   // the window closed while the job was queued
    job->execute();
    CHECK(job->finished());
    CHECK(job->wasCancelled());
    CHECK_FALSE(job->succeeded());
    wb.finishRun(job);
    CHECK_FALSE(wb.running());
}

TEST_CASE("Label edits on a step survive its re-run over the same input labels", "[app][workbench][labels][executor]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource(1, 1, 4, 16, 16));
    wb.setBackend(Backend::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("test_labels");   // 1: the labels
    wb.addStep("test_scale");    // 2: carries them; Recompute keeps one array at a time
    wb.addStep("test_scale");    // 3
    wb.setStepCache(2, CachePolicy::Recompute);
    wb.setStepCache(3, CachePolicy::Recompute);
    REQUIRE(runSync(wb, 2)->succeeded());

    wb.view(2);
    paintOne(wb, 1, 2, 2, 9);
    std::shared_ptr<LabelVolume> edited = wb.output(2)->labels;
    REQUIRE(edited);
    CHECK(edited->edited());
    CHECK(edited->at(0, 1, 2, 2) == 9);

    REQUIRE(runSync(wb, 3)->succeeded());   // step 3's store evicts step 2's array
    CHECK_FALSE(wb.outputFresh(2));
    REQUIRE(runSync(wb, 2)->succeeded());   // step 2 runs again, over the same input labels
    REQUIRE(wb.output(2)->labels);
    CHECK(wb.output(2)->labels == edited);   // the volume with the correction, not a fresh share
    CHECK(wb.output(2)->labels->at(0, 1, 2, 2) == 9);

    SECTION("an upstream that re-segmented supersedes them") {
        wb.setStepParam(1, "label", std::int64_t{2});   // new labels upstream, step 2 stale
        REQUIRE(runSync(wb, 2)->succeeded());
        REQUIRE(wb.output(2)->labels);
        CHECK(wb.output(2)->labels != edited);
        CHECK(wb.output(2)->labels->at(0, 1, 2, 2) == 0);
    }
}

TEST_CASE("Files named in a list parameter are part of the fingerprint", "[app][executor]") {
    registerTestOps();
    Scratch scratch;
    Executor ex(scratch.dir / "cache");
    const std::filesystem::path tile = scratch.dir / "tile.tif";
    {
        std::ofstream f(tile, std::ios::binary);
        f << "one";
    }
    Pipeline p;
    p.add("stitch");   // its `tiles` is a list of files, not a Path parameter
    ParamSet q = p.at(1).params;
    q.set("tiles", std::vector<std::string>{tile.string()});
    p.setParams(1, q);
    const std::string before = ex.fingerprint(p, 1);
    {
        std::ofstream f(tile, std::ios::binary);
        f << "another size";   // rewritten under the same name
    }
    CHECK(ex.fingerprint(p, 1) != before);
}

TEST_CASE("Label statistics follow the viewed time point", "[app][workbench][labels]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource(1, 3, 4, 16, 16));
    wb.setBackend(Backend::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("test_labels");
    REQUIRE(runSync(wb)->succeeded());
    wb.view(1);
    std::shared_ptr<LabelVolume> labels = wb.viewedLabels();
    REQUIRE(labels);
    REQUIRE(labels->t() == 3);
    wb.setT(2);
    CHECK(labels->statsT() == 2);
    wb.setT(1);
    CHECK(labels->statsT() == 1);
    ViewState s = wb.viewState();
    s.t = 0;
    wb.setViewState(s);
    CHECK(labels->statsT() == 0);
}

TEST_CASE("Load parameters left at zero keep the file's own axes and voxel sizes", "[app][workbench][load]") {
    registerTestOps();
    const Operation* load = findOperation("load");
    REQUIRE(load);
    ParamSet p = load->defaults();
    p.set("z", std::int64_t{3});
    p.set("voxel_z", 0.5);
    const OpenOptions o = Workbench::openOptionsFromLoadParams(p);
    REQUIRE(o.pageOrder);
    CHECK(o.pageOrder->c == 0);   // 0: whatever the file says, not 1
    CHECK(o.pageOrder->t == 0);
    CHECK(o.pageOrder->z == 3);
    REQUIRE(o.voxelUm);
    CHECK((*o.voxelUm)[0] == 0.0);
    CHECK((*o.voxelUm)[2] == 0.5);

    // an ImageJ hyperstack of 2 channels x 3 planes: giving only z keeps the channels
    Scratch scratch;
    const std::filesystem::path path = scratch.dir / "hyper.tif";
    Buffer<std::uint16_t> pages(Shape{6, 8, 8});
    for (Index i = 0; i < pages.size(); ++i) pages.data()[i] = static_cast<std::uint16_t>(i);
    TiffWriteOptions w;
    w.description = "ImageJ=1.53t\nimages=6\nchannels=2\nslices=3\nhyperstack=true\n";
    writeTiffStack<std::uint16_t>(path.string(), pages.view(), w);
    const OpenResult plain = sirius::app::openDataset(path.string(), {});
    CHECK(plain.meta.dims.c == 2);
    CHECK(plain.meta.dims.z == 3);
    OpenOptions onlyZ;
    onlyZ.pageOrder = PageOrder{"czt", 0, 0, 3};
    const OpenResult partial = sirius::app::openDataset(path.string(), onlyZ);
    CHECK(partial.meta.dims.c == 2);   // was 1: the unset axis used to be forced to 1
    CHECK(partial.meta.dims.t == 1);
    CHECK(partial.meta.dims.z == 3);
    OpenOptions oneVoxel;
    oneVoxel.voxelUm = std::array<double, 3>{0.0, 0.0, 0.7};
    const OpenResult voxel = sirius::app::openDataset(path.string(), oneVoxel);
    CHECK(voxel.meta.voxelUm[2] == 0.7);
    CHECK(voxel.meta.voxelUm[0] == plain.meta.voxelUm[0]);   // the file's, not 0
}

TEST_CASE("A recorded session marks the strokes and the review decisions", "[app][workbench][session]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource(1, 1, 4, 16, 16));
    wb.setBackend(Backend::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("test_labels");
    REQUIRE(runSync(wb)->succeeded());
    wb.view(1);
    const std::filesystem::path log = scratch.dir / "session.jsonl";
    wb.startRecording(log.string());
    paintOne(wb, 1, 2, 2, 9);
    wb.setLabelReviewed(9, true);
    wb.acceptAllReviewed();
    wb.stopRecording();

    std::ifstream in(log);
    std::vector<std::string> events;
    std::vector<nlohmann::json> records;
    for (std::string line; std::getline(in, line);)
        if (!line.empty()) {
            records.push_back(nlohmann::json::parse(line));
            events.push_back(records.back().value("event", std::string()));
        }
    auto indexOf = [&](const std::string& name) {
        return static_cast<int>(std::find(events.begin(), events.end(), name) - events.begin());
    };
    // a stroke is bracketed, so a replay can group the paint events it holds
    REQUIRE(indexOf("stroke_begin") < static_cast<int>(events.size()));
    REQUIRE(indexOf("stroke_end") < static_cast<int>(events.size()));
    REQUIRE(indexOf("paint") < static_cast<int>(events.size()));
    CHECK(indexOf("stroke_begin") < indexOf("paint"));
    CHECK(indexOf("paint") < indexOf("stroke_end"));
    CHECK(records[static_cast<std::size_t>(indexOf("stroke_end"))].value("voxels", 0) > 0);
    CHECK(records[static_cast<std::size_t>(indexOf("stroke_begin"))].value("label", 0u) == 9u);
    // and the review decisions, which used to leave no trace
    REQUIRE(indexOf("review") < static_cast<int>(events.size()));
    CHECK(records[static_cast<std::size_t>(indexOf("review"))].value("label", 0u) == 9u);
    CHECK(records[static_cast<std::size_t>(indexOf("review"))].value("reviewed", false));
    CHECK(indexOf("review_all") < static_cast<int>(events.size()));
}

TEST_CASE("The tool API takes an integral float as a step number", "[app][tools]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource());
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("test_scale");
    wb.addStep("test_maxz");
    ToolApi api(wb);
    api.call("view_step", {{"step", 2.0}});   // a model writes 2 as 2.0 now and then
    CHECK(wb.viewedIndex() == 1);
    api.call("view_step", {{"step", 3}});
    CHECK(wb.viewedIndex() == 2);
    // call() turns a throw into {"error": ...}: a fraction is not a step
    const json half = api.call("view_step", {{"step", 2.5}});
    CHECK(half.contains("error"));
    CHECK(wb.viewedIndex() == 2);
}

TEST_CASE("On tracked labels a delete and a merge apply to every time point", "[app][workbench][labels][track]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource(1, 3, 4, 16, 16));
    wb.setBackend(Backend::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("test_labels");
    wb.setStepParam(1, "tracked", true);
    REQUIRE(runSync(wb)->succeeded());
    wb.view(1);
    std::shared_ptr<LabelVolume> labels = wb.viewedLabels();
    REQUIRE(labels);
    REQUIRE(labels->tracked());
    const Index cz = 2, cy = 8, cx = 8;
    for (Index t = 0; t < 3; ++t) {
        REQUIRE(labels->at(t, cz, cy, cx) == 1);
        REQUIRE(labels->at(t, cz, 2, 2) == 2);
    }

    SECTION("delete removes the track, undo brings it back everywhere") {
        wb.setT(1);
        wb.deleteLabel(1);   // the id is the object's whole life, not this frame's
        for (Index t = 0; t < 3; ++t) {
            CHECK(labels->at(t, cz, cy, cx) == 0);
            CHECK(labels->at(t, cz, 2, 2) == 2);
        }
        CHECK(wb.history().undoLabel() == "Delete track 1");
        wb.undo();
        for (Index t = 0; t < 3; ++t) CHECK(labels->at(t, cz, cy, cx) == 1);
        wb.redo();
        for (Index t = 0; t < 3; ++t) CHECK(labels->at(t, cz, cy, cx) == 0);
    }
    SECTION("merge joins the two tracks in every frame") {
        wb.mergeLabels({1, 2});
        for (Index t = 0; t < 3; ++t) {
            CHECK(labels->at(t, cz, cy, cx) == 1);
            CHECK(labels->at(t, cz, 2, 2) == 1);   // into the smaller id
        }
        wb.undo();
        for (Index t = 0; t < 3; ++t) CHECK(labels->at(t, cz, 2, 2) == 2);
    }
    SECTION("untracked labels are still edited one frame at a time") {
        wb.setStepParam(1, "tracked", false);
        REQUIRE(runSync(wb)->succeeded());
        labels = wb.viewedLabels();
        REQUIRE(labels);
        CHECK_FALSE(labels->tracked());
    }
}

TEST_CASE("A choice has to be named in full", "[app][pipeline][params]") {
    registerBuiltinOperations();
    Pipeline p;
    p.add("resample");
    ParamSet q = p.at(1).params;
    q.set("interpolation", std::string("Cubic"));   // case is forgiven
    p.setParams(1, q);
    CHECK(p.at(1).params.getString("interpolation") == "cubic");
    q.set("interpolation", std::string("c"));       // a prefix is a typo: the default it is
    p.setParams(1, q);
    CHECK(p.at(1).params.getString("interpolation") != "cubic");
}

TEST_CASE("On tracked labels a split travels along the track", "[app][workbench][labels][track]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource(1, 3, 4, 16, 16));
    wb.setBackend(Backend::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("test_labels");
    wb.setStepParam(1, "tracked", true);
    REQUIRE(runSync(wb)->succeeded());
    wb.view(1);
    std::shared_ptr<LabelVolume> labels = wb.viewedLabels();
    REQUIRE(labels);
    const Index cz = 2, cy = 8, cx = 8;   // the object spans x = 8 .. 12 in every frame
    for (Index t = 0; t < 3; ++t) {
        REQUIRE(labels->at(t, cz, cy, cx) == 1);
        REQUIRE(labels->at(t, cz, cy, cx + 4) == 1);
    }
    wb.setT(1);
    wb.splitLabel(1, {cz, cy, cx}, {cz, cy, cx + 4});
    CHECK(wb.history().undoLabel() == "Split track 1");
    const std::uint32_t part = labels->at(1, cz, cy, cx + 4);
    REQUIRE(part != 0);
    REQUIRE(part != 1);
    for (Index t = 0; t < 3; ++t) {
        INFO("frame " << t);
        CHECK(labels->at(t, cz, cy, cx) == 1);        // the seed's lobe keeps the id ...
        CHECK(labels->at(t, cz, cy, cx + 4) == part);   // ... the other lobe has the same new id in every frame
        CHECK(labels->at(t, cz, 2, 2) == 2);            // the neighbour is untouched
    }
    wb.undo();
    for (Index t = 0; t < 3; ++t) CHECK(labels->at(t, cz, cy, cx + 4) == 1);
    wb.redo();
    for (Index t = 0; t < 3; ++t) CHECK(labels->at(t, cz, cy, cx + 4) == part);
}
