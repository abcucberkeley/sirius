// Tests of the built-in operations (app/core/ops): registry, per-operation
// summaries / validation / output metadata, and run() on small synthetic
// arrays -- plus the SIM reconstruction of the bundled test data through the
// Load and SIM steps, and the Segmentation step against a fake worker
// speaking the RPC protocol over an in-memory transport.

// requireOperation returns a reference to a registry-owned object; GCC 13's
// -Wdangling-reference cannot see that and flags every binding of it.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 13
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <random>
#include <thread>

#include <sirius/tiff_io.hpp>

#include "core/array_source.hpp"
#include "core/cancel.hpp"
#include "core/executor.hpp"
#include "core/ops/builtin.hpp"
#include "core/pipeline.hpp"
#include "core/rpc.hpp"

#include <set>

#include "temp_path.hpp"

using namespace sirius;
using namespace sirius::app;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

    const std::filesystem::path kData = SIRIUS_TEST_DATA_DIR;

    struct Registered {
        Registered() { registerBuiltinOperations(); }
    };
    const Registered kRegistered;

    DatasetMeta metaFor(Dims5 dims, double dx = 0.1, double dz = 0.3) {
        DatasetMeta m;
        m.name = "synthetic";
        m.format = "memory";
        m.dims = dims;
        m.voxelUm = {dx, dx, dz};
        for (Index c = 0; c < dims.c; ++c) {
            ChannelInfo ch;
            ch.label = "ch" + std::to_string(c);
            ch.wavelengthNm = c == 0 ? 488.0 : 640.0;
            m.channels.push_back(ch);
        }
        m.normalizeChannels();
        return m;
    }

    // value = c*1000 + t*100 + z*10 + y + x/100 (distinct, monotone in every axis)
    std::shared_ptr<Array5> rampArray(Dims5 dims) {
        auto a = std::make_shared<Array5>(dims);
        for (Index c = 0; c < dims.c; ++c)
            for (Index t = 0; t < dims.t; ++t)
                for (Index z = 0; z < dims.z; ++z)
                    for (Index y = 0; y < dims.y; ++y)
                        for (Index x = 0; x < dims.x; ++x)
                            a->at(c, t, z, y, x) = static_cast<float>(c * 1000 + t * 100 + z * 10 + y + x / 100.0);
        return a;
    }

    StepInput inputOf(std::shared_ptr<Array5> array, DatasetMeta meta) {
        StepInput in;
        in.meta = std::move(meta);
        in.array = std::move(array);
        return in;
    }

    struct Progress {
        std::vector<double> fractions;
        StepContext ctx;
        Progress() {
            ctx.backend = Backend::Cpu;
            ctx.scratchDir = std::filesystem::temp_directory_path();
            ctx.progress = [this](double f, const std::string&) { fractions.push_back(f); };
        }
    };

    // A blob volume: `n` spheres of radius r on a black ground, plus noise-free background.
    std::shared_ptr<Array5> blobArray(Dims5 dims, int n, double r) {
        auto a = std::make_shared<Array5>(Array5::zeros(dims));
        for (int i = 0; i < n; ++i) {
            const double cz = dims.z / 2.0, cy = (i + 0.5) * dims.y / n, cx = dims.x * 0.5;
            for (Index z = 0; z < dims.z; ++z)
                for (Index y = 0; y < dims.y; ++y)
                    for (Index x = 0; x < dims.x; ++x) {
                        const double d = std::sqrt((z - cz) * (z - cz) + (y - cy) * (y - cy) + (x - cx) * (x - cx));
                        if (d <= r) a->at(0, 0, z, y, x) = 1000.0f;
                    }
        }
        return a;
    }

} // namespace

// --- registry --------------------------------------------------------------

TEST_CASE("the built-in operations are registered with complete metadata", "[app][ops]") {
    const char* kinds[] = {"load", "sim", "decon", "volrec", "einsum", "maxproj", "meant", "contrast", "flatfield",
                           "bleach", "deskew", "croppad", "resample", "merge", "stitch", "register", "seg",
                           "threshold", "cleanup"};
    for (const char* kind : kinds) {
        INFO(kind);
        const Operation* op = findOperation(kind);
        REQUIRE(op != nullptr);
        const OpInfo& info = op->info();
        CHECK(info.kind == kind);
        CHECK_FALSE(info.name.empty());
        CHECK_FALSE(info.group.empty());
        CHECK_FALSE(info.kindLabel.empty());
        CHECK(std::all_of(info.kindLabel.begin(), info.kindLabel.end(),
                          [](unsigned char c) { return !std::islower(c); }));
        // the reduce presets share the einsum page
        const bool preset = info.kind == "maxproj" || info.kind == "meant";
        CHECK(info.helpPage == (preset ? "einsum" : kind));
        const ParamSet defaults = op->defaults();
        CHECK(defaults.size() == info.params.size());
        for (const ParamSpec& s : info.params) CHECK(defaults.has(s.key));
    }
    // other test files register synthetic "test_*" operations in the same process
    std::size_t builtins = 0;
    for (const Operation* op : allOperations())
        if (op->kind().rfind("test_", 0) != 0 && !op->info().plugin) ++builtins;   // nor plugins the worker tests load
    CHECK(builtins == 22);

    SECTION("menu groups follow the design's order and exclude Load") {
        const auto groups = operationGroups();
        REQUIRE(groups.size() >= 6);
        CHECK(groups[0].first == "Reconstruct");
        CHECK(groups[1].first == "Reduce");
        CHECK(groups[2].first == "Intensity");
        CHECK(groups[3].first == "Geometry");
        CHECK(groups[4].first == "Combine");
        CHECK(groups[5].first == "Segment");
        for (const auto& g : groups)
            for (const Operation* op : g.second) CHECK(op->kind() != "load");
        CHECK(groups[0].second.size() == 3);
    }
    SECTION("the example pipeline lists every design step") {
        const Pipeline p = Pipeline::example();
        REQUIRE(p.size() == 9);
        CHECK(p.at(0).kind == "load");
        CHECK(p.at(1).kind == "sim");
        CHECK_FALSE(p.at(2).enabled);   // deskew is skipped
        CHECK(p.at(8).kind == "volrec");
        CHECK(p.at(1).cache == CachePolicy::Disk);
    }
}

// --- load + SIM ---------------------------------------------------------------

TEST_CASE("Load validates its path and describes the raw SIM stack", "[app][ops][load]") {
    const Operation& load = requireOperation("load");
    ParamSet p = load.defaults();
    CHECK_FALSE(load.validate(p, DatasetMeta{}).ok());
    p.set("path", std::string("/nonexistent/file.tif"));
    CHECK_FALSE(load.validate(p, DatasetMeta{}).ok());

    p.set("path", (kData / "raw.tif").string());
    p.set("sim_ndirs", std::int64_t{3});
    p.set("sim_nphases", std::int64_t{5});
    p.set("voxel_x", 0.08);
    p.set("voxel_y", 0.08);
    p.set("voxel_z", 0.125);
    const Validation v = load.validate(p, DatasetMeta{});
    CHECK(v.ok());
    const DatasetMeta meta = load.outputMeta(p, DatasetMeta{});
    CHECK(meta.dims == Dims5{1, 1, 135, 64, 64});
    CHECK(meta.sim.present);
    CHECK(meta.sim.sectionsPerPlane() == 15);
    CHECK_THAT(meta.dx(), WithinRel(0.08, 1e-12));
    CHECK(load.summary(p, DatasetMeta{}).find("15 phase images per plane") != std::string::npos);

    Progress prog;
    const StepOutput out = load.run(StepInput{}, p, prog.ctx);
    REQUIRE(out.source);
    CHECK(out.meta.dims == meta.dims);
    CHECK(out.meta.sim.present);
    CHECK_FALSE(out.array);   // lazy
    Buffer<float> vol = out.asInput().readVolume(0, 0);
    CHECK(vol.shape() == Shape{135, 64, 64});

    SECTION("Full load materializes") {
        p.set("read_as", std::string("Full load to RAM"));
        const StepOutput full = load.run(StepInput{}, p, prog.ctx);
        REQUIRE(full.array);
        CHECK(full.array->dims() == meta.dims);
    }
}

TEST_CASE("SIM reconstructs the bundled stack from a parameter file and reports the fit", "[app][ops][sim]") {
    const Operation& load = requireOperation("load");
    ParamSet lp = load.defaults();
    lp.set("path", (kData / "raw.tif").string());
    lp.set("sim_ndirs", std::int64_t{3});
    lp.set("sim_nphases", std::int64_t{5});
    lp.set("voxel_x", 0.08);
    lp.set("voxel_y", 0.08);
    lp.set("voxel_z", 0.125);
    Progress prog;
    const StepOutput loaded = load.run(StepInput{}, lp, prog.ctx);

    const Operation& sim = requireOperation("sim");
    ParamSet sp = sim.defaults();
    sp.set("mode", std::string("From file"));
    sp.set("params_file", (kData / "config.txt").string());
    sp.set("otf", (kData / "otf.tif").string());
    const Validation v = sim.validate(sp, loaded.meta);
    INFO(v.firstError());
    REQUIRE(v.ok());
    CHECK(sim.summary(sp, loaded.meta).find("3 angles") != std::string::npos);
    const DatasetMeta predicted = sim.outputMeta(sp, loaded.meta);
    CHECK(predicted.dims == Dims5{1, 1, 9, 128, 128});
    CHECK_FALSE(predicted.sim.present);
    CHECK_THAT(predicted.dx(), WithinRel(0.04, 1e-9));

    const StepOutput out = sim.run(loaded.asInput(), sp, prog.ctx);
    REQUIRE(out.array);
    CHECK(out.array->dims() == predicted.dims);
    CHECK(out.meta.dims == predicted.dims);
    CHECK(out.diagnostics.kind == DiagnosticsKind::Sim);
    REQUIRE(out.diagnostics.table);
    CHECK(out.diagnostics.table->rows.size() == 3);
    CHECK(out.diagnostics.table->header.size() == 4);
    REQUIRE_FALSE(out.diagnostics.tabs.empty());
    CHECK(out.diagnostics.tabs.front().name == "Raw spectrum");
    CHECK(out.diagnostics.tabs.front().images.size() == 3);
    CHECK(out.diagnostics.tabs.back().name == "Result spectrum");
    bool bands = false;
    for (const DiagnosticTab& t : out.diagnostics.tabs) bands = bands || t.name == "Separated bands";
    CHECK(bands);   // the stack is small enough for capture
    CHECK(out.diagnostics.footer.find("resolution gain") != std::string::npos);
    CHECK(out.note.find("measured OTF") != std::string::npos);
    CHECK(prog.fractions.back() == 1.0);

    SECTION("a section count that is not angles x phases is rejected") {
        DatasetMeta bad = loaded.meta;
        bad.dims.z = 134;
        CHECK_FALSE(sim.validate(sp, bad).ok());
    }
    SECTION("Manual mode needs one angle per direction, in the degrees the table reports") {
        ParamSet m = sim.defaults();
        m.set("mode", std::string("Manual"));
        CHECK_FALSE(sim.validate(m, loaded.meta).ok());
        m.set("k0_angles", std::vector<double>{46.08, 106.31, -13.68});
        m.set("otf", (kData / "otf.tif").string());
        REQUIRE(sim.validate(m, loaded.meta).ok());
        // What the units have to be right for. Manual mode does not fix the
        // angles: it seeds the k0 fit, which then refines them -- 40, 100, -20
        // converges to the same 46, 106, -14. What the assertion pins is that
        // the seed lands inside the fit's basin, which it only does when the
        // numbers are read as the degrees the form asks for. Read as radians,
        // 46.08 is some 2600 degrees; the old radian values (0.8043 and the
        // rest) seed 0.8 degrees and the table comes out 18, 6, -14. A seed far
        // enough out is not rescued either: 10, 70, -50 reports 18, 67, -38.
        const StepOutput manual = sim.run(loaded.asInput(), m, prog.ctx);
        REQUIRE(manual.diagnostics.table.has_value());
        const std::vector<std::vector<std::string>>& rows = manual.diagnostics.table->rows;
        REQUIRE(rows.size() >= 3);
        CHECK(rows[0][0] == "46°");
        CHECK(rows[1][0] == "106°");
        CHECK(rows[2][0] == "-14°");
    }
    SECTION("the theoretical OTF works without a file") {
        ParamSet e = sim.defaults();
        e.set("linespacing_um", 0.2035);
        e.set("na", 1.42);
        e.set("nimm", 1.515);
        e.set("k0_start_angle", 46.08);   // degrees
        REQUIRE(sim.validate(e, loaded.meta).ok());
        const StepOutput ideal = sim.run(loaded.asInput(), e, prog.ctx);
        CHECK(ideal.array->dims() == predicted.dims);
        CHECK(ideal.note.find("theoretical OTF") != std::string::npos);
    }
}

TEST_CASE("SIM reports a cancelled run as a cancellation, not as a step failure",
          "[app][ops][sim][cancel]") {
    // A SIM reconstruction is the longest thing the application does -- minutes
    // on real data -- so Cancel has to reach into the library, not merely stop
    // between volumes. The step must then surface the abort as a cancellation:
    // the executor recognises it, leaves the step unblamed, and caches nothing.
    const Operation& load = requireOperation("load");
    ParamSet lp = load.defaults();
    lp.set("path", (kData / "raw.tif").string());
    lp.set("sim_ndirs", std::int64_t{3});
    lp.set("sim_nphases", std::int64_t{5});
    lp.set("voxel_x", 0.08);
    lp.set("voxel_y", 0.08);
    lp.set("voxel_z", 0.125);
    Progress prog;
    const StepOutput loaded = load.run(StepInput{}, lp, prog.ctx);

    const Operation& sim = requireOperation("sim");
    ParamSet sp = sim.defaults();
    sp.set("mode", std::string("From file"));
    sp.set("params_file", (kData / "config.txt").string());
    sp.set("otf", (kData / "otf.tif").string());
    REQUIRE(sim.validate(sp, loaded.meta).ok());

    SECTION("run() throws something isCancellation() recognises, mid-reconstruction") {
        Progress p2;
        int polls = 0;
        p2.ctx.cancelled = [&polls] { return ++polls > 2; };
        const auto t0 = std::chrono::steady_clock::now();
        try {
            sim.run(loaded.asInput(), sp, p2.ctx);
            FAIL("the SIM step ran to completion despite the cancel");
        } catch (const std::exception& e) {
            INFO("threw: " << e.what());
            CHECK(isCancellation(e));
        }
        // The library aborted at a stage boundary rather than finishing the
        // volume: the full reconstruction of this stack takes far longer than
        // the handful of stages the predicate allowed.
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        INFO(seconds << " s to the throw, after " << polls << " polls");
        CHECK(polls > 2);
    }

    SECTION("cancelling before any work still reads as a cancellation") {
        Progress p2;
        p2.ctx.cancelled = [] { return true; };
        try {
            sim.run(loaded.asInput(), sp, p2.ctx);
            FAIL("the SIM step ran despite an always-true cancel");
        } catch (const std::exception& e) {
            CHECK(isCancellation(e));
        }
    }

    SECTION("the executor blames nobody and caches nothing for the cancelled step") {
        const std::filesystem::path scratch =
            std::filesystem::temp_directory_path() / ("sirius-sim-cancel-" + std::to_string(std::random_device{}()));
        std::filesystem::create_directories(scratch);
        {
            Executor ex(scratch / "cache");
            Pipeline p;   // a fresh pipeline already holds the Load step at 0
            const StepId simId = p.add("sim");
            p.setParams(0, lp);
            p.setParams(1, sp);
            auto seeded = std::make_shared<StepOutput>(loaded);
            ex.seed(p, 0, seeded);

            StepContext ctx;
            ctx.scratchDir = scratch;
            // Arm only once the SIM step is running, and let a few polls
            // through so the throw comes from inside the reconstruction
            // rather than from the executor's own pre-run check.
            bool inSim = false;
            int polls = 0;
            ctx.cancelled = [&inSim, &polls] { return inSim && ++polls > 3; };
            std::vector<StepReport> reports;
            CHECK_THROWS_AS(ex.runAll(p, ctx, &reports,
                                      [&inSim](const StepReport& r) {
                                          if (r.index == 1 && r.state == StepReport::State::Running) inSim = true;
                                      }),
                            CancelledError);
            for (const StepReport& r : reports) CHECK_FALSE(r.failed());
            // Nothing was published: no cache entry, no spill file left behind.
            CHECK_FALSE(ex.isFresh(p, 1));
            CHECK(ex.cachedBytesOf(simId) == 0);
            CHECK(ex.lastOutput(simId) == nullptr);
            CHECK(polls > 3);   // the abort came from inside the reconstruction
            const std::string spillPrefix = "step-" + std::to_string(simId) + "-";
            std::size_t spills = 0;
            if (std::filesystem::exists(scratch / "cache"))
                for (const auto& e : std::filesystem::directory_iterator(scratch / "cache"))
                    if (e.path().filename().string().rfind(spillPrefix, 0) == 0) ++spills;
            CHECK(spills == 0);   // no half-written cache entry survives the cancel
        }
        std::error_code ec;
        std::filesystem::remove_all(scratch, ec);
    }
}

// --- reductions ---------------------------------------------------------------

TEST_CASE("Einsum reduces the chosen axes and keeps the others in place", "[app][ops][einsum]") {
    const Dims5 dims{2, 3, 4, 5, 6};
    const DatasetMeta meta = metaFor(dims);
    const Operation& op = requireOperation("einsum");
    ParamSet p = op.defaults();
    CHECK(op.summary(p, meta).find("mean over t") != std::string::npos);
    CHECK(op.outputMeta(p, meta).dims == Dims5{2, 1, 4, 5, 6});

    Progress prog;
    const StepOutput out = op.run(inputOf(rampArray(dims), meta), p, prog.ctx);
    REQUIRE(out.array);
    CHECK(out.array->dims() == Dims5{2, 1, 4, 5, 6});
    // mean over t of (t*100 + rest) = rest + 100
    CHECK_THAT(out.array->at(1, 0, 2, 3, 4), WithinAbs(1000 + 100 + 20 + 3 + 0.04, 1e-3));
    CHECK(out.diagnostics.kind == DiagnosticsKind::Generic);
    CHECK_FALSE(out.diagnostics.images.empty());

    SECTION("max over z and c") {
        p.set("keep", std::string("tyx"));
        p.set("reduction", std::string("max"));
        const StepOutput m = op.run(inputOf(rampArray(dims), meta), p, prog.ctx);
        CHECK(m.array->dims() == Dims5{1, 3, 1, 5, 6});
        CHECK_THAT(m.array->at(0, 1, 0, 3, 4), WithinAbs(1000 + 100 + 30 + 3 + 0.04, 1e-3));
        CHECK(m.meta.channels.size() == 1);
    }
    SECTION("identity") {
        p.set("keep", std::string("ctzyx"));
        CHECK(op.summary(p, meta) == "identity — nothing reduced");
        const StepOutput id = op.run(inputOf(rampArray(dims), meta), p, prog.ctx);
        CHECK(id.array->dims() == dims);
    }
    SECTION("presets") {
        const Operation& mp = requireOperation("maxproj");
        CHECK(mp.outputMeta(mp.defaults(), meta).dims == Dims5{2, 3, 1, 5, 6});
        const StepOutput m = mp.run(inputOf(rampArray(dims), meta), mp.defaults(), prog.ctx);
        CHECK_THAT(m.array->at(0, 0, 0, 0, 0), WithinAbs(30.0, 1e-4));
        const Operation& mt = requireOperation("meant");
        CHECK(mt.outputMeta(mt.defaults(), meta).dims == Dims5{2, 1, 4, 5, 6});
        CHECK(mt.validate(mt.defaults(), metaFor(Dims5{1, 1, 4, 5, 6})).warnings.size() == 1);
    }
}

// --- intensity ----------------------------------------------------------------

TEST_CASE("Contrast rescales every channel into 0..1 and reports histograms", "[app][ops][contrast]") {
    const Dims5 dims{2, 2, 3, 8, 8};
    const DatasetMeta meta = metaFor(dims);
    const Operation& op = requireOperation("contrast");
    ParamSet p = op.defaults();
    p.set("lo_percentile", 0.0);
    p.set("hi_percentile", 100.0);
    Progress prog;
    const StepOutput out = op.run(inputOf(rampArray(dims), meta), p, prog.ctx);
    REQUIRE(out.array);
    const auto mm = minMax(*out.array);
    CHECK_THAT(mm.first, WithinAbs(0.0, 1e-6));
    CHECK_THAT(mm.second, WithinAbs(1.0, 1e-6));
    CHECK(out.diagnostics.kind == DiagnosticsKind::Contrast);
    REQUIRE(out.diagnostics.histograms.size() == 2);
    CHECK(out.diagnostics.histograms[0].bins.size() == 30);
    CHECK(out.diagnostics.histograms[1].channel == "ch1");
    // one window for every channel: channel 1 (values ~1000+) sits at the top of it
    CHECK(out.array->at(1, 0, 0, 0, 0) > 0.5f);
    CHECK(out.diagnostics.histograms[0].lo == out.diagnostics.histograms[1].lo);

    SECTION("the live preview needs no run") {
        const Diagnostics d = contrastPreview(inputOf(rampArray(dims), meta), p);
        CHECK(d.kind == DiagnosticsKind::Contrast);
        CHECK(d.histograms.size() == 2);
        CHECK(d.histograms[0].lo <= d.histograms[0].hi);
    }
    SECTION("gamma and a bad window") {
        p.set("gamma", 2.0);
        const StepOutput g = op.run(inputOf(rampArray(dims), meta), p, prog.ctx);
        CHECK(g.array->at(0, 1, 2, 7, 7) <= 1.0f);
        p.set("lo_percentile", 60.0);
        p.set("hi_percentile", 60.0);
        CHECK_FALSE(op.validate(p, meta).ok());
    }
}

TEST_CASE("Flat-field divides by the flat image", "[app][ops][flatfield]") {
    const Dims5 dims{1, 1, 2, 4, 4};
    const DatasetMeta meta = metaFor(dims);
    auto data = std::make_shared<Array5>(Array5::filled(dims, 10.0f));
    Buffer<float> flat(Shape{4, 4});
    for (Index i = 0; i < 16; ++i) flat.data()[i] = i < 8 ? 1.0f : 3.0f;   // mean 2
    const test::TempFile file("app_ops_flat", ".tif");
    writeTiff<float>(file.str, flat.view());

    const Operation& op = requireOperation("flatfield");
    ParamSet p = op.defaults();
    CHECK_FALSE(op.validate(p, meta).ok());
    p.set("flat", file.str);
    REQUIRE(op.validate(p, meta).ok());
    Progress prog;
    const StepOutput out = op.run(inputOf(data, meta), p, prog.ctx);
    REQUIRE(out.array);
    CHECK_THAT(out.array->at(0, 0, 1, 0, 0), WithinRel(20.0, 1e-4));   // 10 / 1 * 2
    CHECK_THAT(out.array->at(0, 0, 1, 3, 3), WithinRel(10.0 / 3.0 * 2.0, 1e-4));
}

TEST_CASE("Bleach correction equalizes frame sums", "[app][ops][bleach]") {
    const Dims5 dims{1, 3, 2, 4, 4};
    const DatasetMeta meta = metaFor(dims);
    auto data = std::make_shared<Array5>(dims);
    for (Index t = 0; t < 3; ++t)
        for (Index i = 0; i < 2 * 16; ++i) data->plane(0, t, 0)[i] = static_cast<float>(t + 1);   // sums 32, 64, 96
    const Operation& op = requireOperation("bleach");
    Progress prog;
    const StepOutput out = op.run(inputOf(data, meta), op.defaults(), prog.ctx);
    REQUIRE(out.array);
    for (Index t = 0; t < 3; ++t) {
        const float* f = out.array->plane(0, t, 0);
        CHECK_THAT(std::accumulate(f, f + 32, 0.0), WithinRel(32.0, 1e-5));
    }
    SECTION("over z, to the mean") {
        ParamSet p = op.defaults();
        p.set("over", std::string("z"));
        p.set("mode", std::string("Match mean"));
        auto d2 = std::make_shared<Array5>(dims);
        for (Index z = 0; z < 2; ++z)
            for (Index i = 0; i < 16; ++i) d2->plane(0, 0, z)[i] = z == 0 ? 1.0f : 3.0f;
        const StepOutput o2 = op.run(inputOf(d2, meta), p, prog.ctx);
        const float* a = o2.array->plane(0, 0, 0);
        const float* b = o2.array->plane(0, 0, 1);
        CHECK_THAT(std::accumulate(a, a + 16, 0.0), WithinRel(std::accumulate(b, b + 16, 0.0), 1e-5));
    }
}

// --- geometry -----------------------------------------------------------------

TEST_CASE("Deskew shears the stack and warns when the data is not light-sheet", "[app][ops][deskew]") {
    const Dims5 dims{1, 1, 6, 8, 10};
    DatasetMeta meta = metaFor(dims, 0.1, 0.4);
    const Operation& op = requireOperation("deskew");
    ParamSet p = op.defaults();
    p.set("rotate_to_coverslip", false);
    CHECK_FALSE(op.validate(p, meta).warnings.empty());
    CHECK(op.summary(p, meta).find("skipped") != std::string::npos);
    meta.lightSheet = true;
    meta.sheetAngleDeg = 31.8;
    CHECK(op.validate(p, meta).ok());
    const DatasetMeta out = op.outputMeta(p, meta);
    CHECK(out.dims.x > dims.x);   // the shear widens x
    CHECK_FALSE(out.lightSheet);
    Progress prog;
    const StepOutput r = op.run(inputOf(rampArray(dims), meta), p, prog.ctx);
    REQUIRE(r.array);
    CHECK(r.array->dims() == out.dims);
    CHECK_FALSE(r.diagnostics.images.empty());
}

TEST_CASE("Crop / pad cuts a box and carries labels", "[app][ops][croppad]") {
    const Dims5 dims{1, 1, 4, 6, 8};
    const DatasetMeta meta = metaFor(dims);
    const Operation& op = requireOperation("croppad");
    ParamSet p = op.defaults();
    p.set("z0", std::int64_t{1});
    p.set("y0", std::int64_t{-1});
    p.set("x0", std::int64_t{2});
    p.set("z", std::int64_t{2});
    p.set("y", std::int64_t{4});
    p.set("x", std::int64_t{0});
    p.set("fill", -1.0);
    CHECK(op.outputMeta(p, meta).dims == Dims5{1, 1, 2, 4, 6});
    StepInput in = inputOf(rampArray(dims), meta);
    auto labels = std::make_shared<LabelVolume>(1, 4, 6, 8);
    labels->volume(0)[(1 * 6 + 0) * 8 + 2] = 7;   // (z1, y0, x2) -> output (0, 1, 0)
    in.labels = labels;
    Progress prog;
    const StepOutput out = op.run(in, p, prog.ctx);
    REQUIRE(out.array);
    CHECK(out.array->dims() == Dims5{1, 1, 2, 4, 6});
    CHECK(out.array->at(0, 0, 0, 0, 0) == -1.0f);                                 // padded row
    CHECK_THAT(out.array->at(0, 0, 0, 1, 0), WithinAbs(10 + 0 + 0.02, 1e-4));   // z1 y0 x2
    REQUIRE(out.labels);
    CHECK(out.labels->at(0, 0, 1, 0) == 7);
    CHECK(out.labels->at(0, 0, 0, 0) == 0);
}

TEST_CASE("Resample changes the voxel size", "[app][ops][resample]") {
    const Dims5 dims{1, 1, 4, 8, 8};
    const DatasetMeta meta = metaFor(dims, 0.1, 0.4);
    const Operation& op = requireOperation("resample");
    ParamSet p = op.defaults();
    p.set("voxel_x", 0.2);
    p.set("voxel_y", 0.2);
    const DatasetMeta out = op.outputMeta(p, meta);
    CHECK(out.dims.x == 4);
    CHECK(out.dims.y == 4);
    CHECK(out.dims.z == 4);
    CHECK_THAT(out.dx(), WithinRel(0.2, 1e-9));
    Progress prog;
    const StepOutput r = op.run(inputOf(rampArray(dims), meta), p, prog.ctx);
    REQUIRE(r.array);
    CHECK(r.array->dims() == out.dims);
}

TEST_CASE("Volume reconstruction resamples to isotropic voxels", "[app][ops][volrec]") {
    const Dims5 dims{1, 1, 4, 8, 8};
    const DatasetMeta meta = metaFor(dims, 0.1, 0.4);
    const Operation& op = requireOperation("volrec");
    ParamSet p = op.defaults();
    const DatasetMeta out = op.outputMeta(p, meta);
    // 4 planes 0.4 um apart span 1.2 um between their centres: 13 planes at 0.1 um
    CHECK(out.dims.z == 13);
    CHECK_THAT(out.dz(), WithinRel(0.1, 1e-9));
    Progress prog;
    const StepOutput r = op.run(inputOf(rampArray(dims), meta), p, prog.ctx);
    REQUIRE(r.array);
    CHECK(r.array->dims() == out.dims);
    CHECK(r.diagnostics.kind == DiagnosticsKind::Volume);
    CHECK(r.diagnostics.curves.size() == 1);
    CHECK_FALSE(r.diagnostics.facts.empty());
    SECTION("keep the grid") {
        p.set("resample", std::string("Keep"));
        CHECK(op.outputMeta(p, meta).dims == dims);
    }
}

// --- combine ------------------------------------------------------------------

TEST_CASE("Merge blends channels into RGB", "[app][ops][merge]") {
    const Dims5 dims{2, 1, 2, 4, 4};
    const DatasetMeta meta = metaFor(dims);
    const Operation& op = requireOperation("merge");
    ParamSet p = op.defaults();
    CHECK(op.summary(p, meta).find("→") != std::string::npos);
    const DatasetMeta out = op.outputMeta(p, meta);
    CHECK(out.rgb);
    CHECK(out.dims.c == 3);
    CHECK(out.channels.size() == 3);
    auto data = std::make_shared<Array5>(Array5::zeros(dims));
    for (Index i = 0; i < 32; ++i) data->plane(0, 0, 0)[i] = 500.0f;   // ch0 = 500 everywhere
    for (Index i = 0; i < 32; ++i) data->plane(1, 0, 0)[i] = (i % 2) ? 800.0f : 0.0f;
    p.set("colors", std::vector<std::string>{"#ff0000", "#0000ff"});
    Progress prog;
    const StepOutput r = op.run(inputOf(data, meta), p, prog.ctx);
    REQUIRE(r.array);
    CHECK(r.array->dims() == out.dims);
    CHECK_THAT(r.array->at(0, 0, 0, 0, 0), WithinAbs(1.0, 1e-5));   // red from ch0 (normalized to its max)
    CHECK_THAT(r.array->at(2, 0, 0, 0, 0), WithinAbs(0.0, 1e-5));
    CHECK_THAT(r.array->at(2, 0, 0, 0, 1), WithinAbs(1.0, 1e-5));   // blue from ch1
    CHECK(r.meta.rgb);
    SECTION("an RGB input is rejected") { CHECK_FALSE(op.validate(p, out).ok()); }
}

TEST_CASE("Register recovers a known translation between channels", "[app][ops][register]") {
    const Dims5 dims{2, 1, 1, 48, 48};
    const DatasetMeta meta = metaFor(dims);
    auto data = std::make_shared<Array5>(Array5::zeros(dims));
    // a few bright blobs in channel 0, the same shifted by (dy 3, dx -4) in channel 1
    const int pts[][2] = {{10, 12}, {30, 20}, {22, 36}, {38, 40}, {14, 30}};
    for (const auto& pt : pts)
        for (Index y = 0; y < 48; ++y)
            for (Index x = 0; x < 48; ++x) {
                const double d0 = std::hypot(y - pt[0], x - pt[1]);
                data->at(0, 0, 0, y, x) += static_cast<float>(100.0 * std::exp(-d0 * d0 / 8.0));
                const double d1 = std::hypot(y - (pt[0] + 3), x - (pt[1] - 4));
                data->at(1, 0, 0, y, x) += static_cast<float>(100.0 * std::exp(-d1 * d1 / 8.0));
            }
    const Operation& op = requireOperation("register");
    ParamSet p = op.defaults();
    p.set("max_shift", std::vector<double>{0.0, 8.0, 8.0});
    REQUIRE(op.validate(p, meta).ok());
    Progress prog;
    const StepOutput r = op.run(inputOf(data, meta), p, prog.ctx);
    REQUIRE(r.array);
    CHECK(r.diagnostics.kind == DiagnosticsKind::Alignment);
    REQUIRE(r.diagnostics.table);
    REQUIRE(r.diagnostics.table->rows.size() == 1);
    // moving[p] matches fixed[p + shift]: shift = (0, -3, +4)
    CHECK_THAT(std::stod(r.diagnostics.table->rows[0][2]), WithinAbs(-3.0, 0.6));
    CHECK_THAT(std::stod(r.diagnostics.table->rows[0][3]), WithinAbs(4.0, 0.6));
    // the aligned channel now peaks where channel 0 does
    CHECK(r.array->at(1, 0, 0, 10, 12) > 80.0f);
    CHECK(r.array->at(0, 0, 0, 10, 12) == data->at(0, 0, 0, 10, 12));
    CHECK_FALSE(r.diagnostics.images.empty());

    SECTION("validation") {
        CHECK_FALSE(op.validate(p, metaFor(Dims5{1, 1, 1, 8, 8})).ok());
        p.set("mode", std::string("Align time points to reference"));
        CHECK_FALSE(op.validate(p, meta).ok());   // single time point
    }
}

TEST_CASE("Stitch fuses two overlapping tile files", "[app][ops][stitch]") {
    // one 24 x 96 synthetic scene, cut into two 24 x 56 tiles that overlap by 16
    Buffer<float> scene(Shape{1, 24, 96});
    for (Index y = 0; y < 24; ++y)
        for (Index x = 0; x < 96; ++x) {
            double v = 10.0;
            for (int k = 0; k < 6; ++k) {
                const double cy = 4 + (k * 7) % 16, cx = 8 + k * 15;
                const double d = std::hypot(y - cy, x - cx);
                v += 200.0 * std::exp(-d * d / 6.0);
            }
            scene.data()[y * 96 + x] = static_cast<float>(v);
        }
    auto cut = [&](Index x0) {
        Buffer<float> t(Shape{1, 24, 56});
        for (Index y = 0; y < 24; ++y)
            for (Index x = 0; x < 56; ++x) t.data()[y * 56 + x] = scene.data()[y * 96 + x0 + x];
        return t;
    };
    const test::TempFile a("app_ops_tile0", ".tif"), b("app_ops_tile1", ".tif");
    writeTiffStack<float>(a.str, cut(0).view());
    writeTiffStack<float>(b.str, cut(40).view());

    const Operation& op = requireOperation("stitch");
    ParamSet p = op.defaults();
    CHECK_FALSE(op.validate(p, DatasetMeta{}).ok());
    p.set("tiles", std::vector<std::string>{a.str, b.str});
    p.set("positions", std::vector<double>{0, 0, 0, 0, 0, 38});   // nominal 2 px off
    p.set("search_radius", std::vector<double>{0, 4, 6});
    p.set("mask_background", false);
    REQUIRE(op.validate(p, DatasetMeta{}).ok());
    const DatasetMeta predicted = op.outputMeta(p, DatasetMeta{});
    CHECK(predicted.dims.y == 24);
    CHECK(predicted.dims.x == 94);
    Progress prog;
    const StepOutput r = op.run(StepInput{}, p, prog.ctx);
    REQUIRE(r.array);
    CHECK(r.array->dims().y == 24);
    CHECK(r.array->dims().x >= 94);
    CHECK(r.array->dims().x <= 98);
    CHECK(r.diagnostics.kind == DiagnosticsKind::Alignment);
    REQUIRE(r.diagnostics.alignment);
    CHECK(r.diagnostics.alignment->tileNames.size() == 2);
    CHECK(r.diagnostics.alignment->gridCols == 2);
    REQUIRE(r.diagnostics.table);
    CHECK(r.diagnostics.table->rows.size() == 1);   // one accepted pair
}

// --- segmentation ---------------------------------------------------------------

TEST_CASE("Threshold labels blobs and Label cleanup drops the small ones", "[app][ops][threshold][cleanup]") {
    const Dims5 dims{1, 1, 9, 40, 20};
    const DatasetMeta meta = metaFor(dims);
    auto data = blobArray(dims, 3, 3.0);
    data->at(0, 0, 4, 1, 1) = 1000.0f;   // a one-voxel speck
    const Operation& op = requireOperation("threshold");
    ParamSet p = op.defaults();
    p.set("method", std::string("Manual"));
    p.set("value", 500.0);
    p.set("min_voxels", std::int64_t{0});
    Progress prog;
    const StepOutput r = op.run(inputOf(data, meta), p, prog.ctx);
    REQUIRE(r.labels);
    CHECK(r.labels->stats().size() == 4);
    CHECK(r.diagnostics.kind == DiagnosticsKind::Segment);
    REQUIRE(r.diagnostics.table);
    CHECK(r.diagnostics.table->rows.size() == 4);
    CHECK(r.diagnostics.table->header.size() == 5);
    CHECK(r.labels->at(0, 4, 1, 1) != 0);
    CHECK(r.labels->at(0, 0, 0, 0) == 0);

    SECTION("Otsu finds the same cut") {
        p.set("method", std::string("Otsu"));
        const StepOutput o = op.run(inputOf(data, meta), p, prog.ctx);
        CHECK(o.labels->stats().size() == 4);
    }
    SECTION("cleanup removes the speck and relabels") {
        const Operation& cleanup = requireOperation("cleanup");
        ParamSet cp = cleanup.defaults();
        cp.set("min_voxels", std::int64_t{10});
        StepInput in = r.asInput();
        const StepOutput c = cleanup.run(in, cp, prog.ctx);
        REQUIRE(c.labels);
        CHECK(c.labels->stats().size() == 3);
        CHECK(c.labels->at(0, 4, 1, 1) == 0);
        CHECK(c.labels->maxLabel() == 3);
        CHECK(r.labels->stats().size() == 4);   // the input labels are untouched
        StepInput none = inputOf(data, meta);
        CHECK_THROWS(cleanup.run(none, cp, prog.ctx));
    }
}

TEST_CASE("Classical segmentation finds blobs with global and local thresholds", "[app][ops][classic]") {
    const Dims5 dims{1, 1, 9, 60, 24};
    const DatasetMeta meta = metaFor(dims);
    auto data = blobArray(dims, 3, 5.0);   // three blobs 10 voxels across
    // a sloped background that a fixed cut would misjudge
    for (Index z = 0; z < dims.z; ++z)
        for (Index y = 0; y < dims.y; ++y)
            for (Index x = 0; x < dims.x; ++x) data->at(0, 0, z, y, x) += 100.0f + 8.0f * static_cast<float>(y);
    const Operation& op = requireOperation("classic");
    ParamSet p = op.defaults();
    p.set("opening", std::int64_t{0});
    p.set("min_voxels", std::int64_t{5});
    p.set("sigma", 0.0);
    Progress prog;
    SECTION("Otsu with the watershed keeps the three blobs apart") {
        const StepOutput r = op.run(inputOf(data, meta), p, prog.ctx);
        REQUIRE(r.labels);
        CHECK(r.labels->stats().size() == 3);
        CHECK(r.diagnostics.kind == DiagnosticsKind::Segment);
        CHECK(r.note.find("labels") != std::string::npos);
        bool threshold = false;
        for (const DiagnosticFact& f : r.diagnostics.facts) threshold = threshold || f.key == "Threshold";
        CHECK(threshold);
    }
    SECTION("local mean follows the background") {
        p.set("method", std::string("Local mean"));
        p.set("window", std::int64_t{15});
        p.set("local_ratio", 1.5);
        p.set("post", std::string("Connected components"));
        const StepOutput r = op.run(inputOf(data, meta), p, prog.ctx);
        REQUIRE(r.labels);
        CHECK(r.labels->stats().size() == 3);
    }
    SECTION("a top-hat removes a wide background bump") {
        for (Index z = 0; z < dims.z; ++z)
            for (Index y = 0; y < dims.y; ++y)
                for (Index x = 0; x < dims.x; ++x) data->at(0, 0, z, y, x) += 900.0f * std::exp(-static_cast<float>((y - 30) * (y - 30)) / 400.0f);
        p.set("method", std::string("Manual"));
        p.set("value", 700.0);
        p.set("post", std::string("Connected components"));
        const StepOutput without = op.run(inputOf(data, meta), p, prog.ctx);
        p.set("tophat", std::int64_t{6});
        const StepOutput with = op.run(inputOf(data, meta), p, prog.ctx);
        REQUIRE(with.labels);
        REQUIRE(without.labels);
        auto voxels = [](const LabelVolume& L) {
            Index n = 0;
            for (const LabelStats& st : L.stats()) n += st.voxels;
            return n;
        };
        CHECK(with.labels->stats().size() == 3);
        // without the top-hat the bump itself is foreground: far more voxels than three blobs
        CHECK(voxels(*without.labels) > 2 * voxels(*with.labels));
    }
    SECTION("opening drops a speck and hole filling closes a hollow blob") {
        data->at(0, 0, 4, 1, 1) = 1000.0f;     // one-voxel speck, far from the blobs
        data->at(0, 0, 4, 10, 12) = 0.0f;      // a hole at the centre of the first blob's middle plane
        p.set("method", std::string("Manual"));
        p.set("value", 700.0);
        p.set("post", std::string("Connected components"));
        p.set("min_voxels", std::int64_t{0});
        p.set("fill_holes", false);
        const StepOutput open = op.run(inputOf(data, meta), p, prog.ctx);
        CHECK(open.labels->stats().size() == 4);   // the speck is a label of its own
        CHECK(open.labels->at(0, 4, 10, 12) == 0);
        p.set("opening", std::int64_t{1});
        p.set("fill_holes", true);
        const StepOutput clean = op.run(inputOf(data, meta), p, prog.ctx);
        CHECK(clean.labels->stats().size() == 3);
        CHECK(clean.labels->at(0, 4, 10, 12) != 0);
    }
}

TEST_CASE("Classical segmentation: enhancement, local thresholds and seeding", "[app][ops][classic]") {
    const Dims5 dims{1, 1, 7, 48, 48};
    const DatasetMeta meta = metaFor(dims);
    const Operation& op = requireOperation("classic");
    Progress prog;

    SECTION("the local contrast cut follows a background the global one cannot") {
        // three blobs on a strong ramp: a single global threshold either keeps
        // the bright end's background or loses the dim end's objects
        auto data = blobArray(dims, 3, 4.0);
        for (Index z = 0; z < dims.z; ++z)
            for (Index y = 0; y < dims.y; ++y)
                for (Index x = 0; x < dims.x; ++x) data->at(0, 0, z, y, x) += 40.0f * static_cast<float>(y);
        ParamSet p = op.defaults();
        p.set("sigma", 0.0);
        p.set("opening", std::int64_t{0});
        p.set("min_voxels", std::int64_t{5});
        p.set("post", std::string("Connected components"));
        p.set("method", std::string("Local contrast"));
        p.set("window", std::int64_t{31});
        p.set("contrast_k", 1.5);
        const StepOutput r = op.run(inputOf(data, meta), p, prog.ctx);
        REQUIRE(r.labels);
        CHECK(r.note.find("SD") != std::string::npos);
        // blobArray puts one blob at each third of y, all at the middle of x
        const Index cx = dims.x / 2, cz = dims.z / 2;
        const std::uint32_t a = r.labels->at(0, cz, 8, cx);
        const std::uint32_t b = r.labels->at(0, cz, 24, cx);
        const std::uint32_t c = r.labels->at(0, cz, 40, cx);
        CHECK(a != 0);
        CHECK(b != 0);
        CHECK(c != 0);
        CHECK(a != b);
        CHECK(b != c);
        CHECK(r.labels->at(0, cz, 8, 2) == 0);    // the ramp itself stays background
        CHECK(r.labels->at(0, cz, 40, 2) == 0);
        // one global cut cannot do that: the dim end's blob sits below the
        // bright end's background
        ParamSet global = p;
        global.set("method", std::string("Otsu"));
        const StepOutput one = op.run(inputOf(data, meta), global, prog.ctx);
        REQUIRE(one.labels);
        const bool dimFound = one.labels->at(0, cz, 8, cx) != 0;
        const bool brightBackground = one.labels->at(0, cz, 40, 2) != 0;
        CHECK((!dimFound || brightBackground));
    }

    SECTION("Multi-Otsu keeps only the brightest class") {
        // background 0, a mid-grey halo and bright cores: the upper of the two
        // cuts must land above the halo
        auto data = std::make_shared<Array5>(Array5::zeros(dims));
        for (Index z = 2; z < 5; ++z)
            for (Index y = 8; y < 40; ++y)
                for (Index x = 8; x < 40; ++x) data->at(0, 0, z, y, x) = 400.0f;   // halo
        for (Index z = 3; z < 4; ++z)
            for (Index y = 20; y < 26; ++y)
                for (Index x = 20; x < 26; ++x) data->at(0, 0, z, y, x) = 4000.0f;  // core
        ParamSet p = op.defaults();
        p.set("sigma", 0.0);
        p.set("opening", std::int64_t{0});
        p.set("min_voxels", std::int64_t{2});
        p.set("post", std::string("Connected components"));
        p.set("method", std::string("Multi-Otsu"));
        const StepOutput r = op.run(inputOf(data, meta), p, prog.ctx);
        REQUIRE(r.labels);
        CHECK(r.labels->stats().size() == 1);
        CHECK(r.labels->at(0, 3, 22, 22) != 0);   // the core is an object
        CHECK(r.labels->at(0, 3, 10, 10) == 0);   // the halo is not
    }

    SECTION("blob enhancement rejects a wide background structure") {
        // one small blob plus a broad bright plateau: the difference of
        // Gaussians answers to the blob and flattens the plateau
        auto data = std::make_shared<Array5>(Array5::zeros(dims));
        for (Index z = 0; z < dims.z; ++z)
            for (Index y = 4; y < 44; ++y)
                for (Index x = 4; x < 24; ++x) data->at(0, 0, z, y, x) = 1500.0f;   // plateau
        for (Index z = 2; z < 5; ++z)
            for (Index y = 32; y < 38; ++y)
                for (Index x = 32; x < 38; ++x) data->at(0, 0, z, y, x) = 3000.0f;  // blob
        ParamSet p = op.defaults();
        p.set("sigma", 0.0);
        p.set("opening", std::int64_t{0});
        p.set("min_voxels", std::int64_t{2});
        p.set("post", std::string("Connected components"));
        p.set("method", std::string("Otsu"));
        p.set("fill_holes", false);   // the band-pass answers at the edges; filling would close them
        const StepOutput plain = op.run(inputOf(data, meta), p, prog.ctx);
        p.set("enhance", std::string("Blobs (DoG)"));
        p.set("enhance_sigma", 1.0);
        const StepOutput r = op.run(inputOf(data, meta), p, prog.ctx);
        REQUIRE(r.labels);
        REQUIRE(plain.labels);
        CHECK(plain.labels->at(0, 3, 20, 14) != 0);   // without it the plateau is an object
        CHECK(r.labels->at(0, 3, 35, 35) != 0);       // the blob survives the band-pass
        CHECK(r.labels->at(0, 3, 20, 14) == 0);       // its flat interior does not
    }

    SECTION("h-maxima seeding does not split one waisted object") {
        // a capsule: two spheres overlapping enough to be one object
        auto data = std::make_shared<Array5>(Array5::zeros(dims));
        auto sphere = [&](double cy, double cx, double r) {
            for (Index z = 0; z < dims.z; ++z)
                for (Index y = 0; y < dims.y; ++y)
                    for (Index x = 0; x < dims.x; ++x) {
                        const double d2 = (z - 3.0) * (z - 3.0) + (y - cy) * (y - cy) + (x - cx) * (x - cx);
                        if (d2 <= r * r) data->at(0, 0, z, y, x) = 3000.0f;
                    }
        };
        sphere(24, 20, 7.0);
        sphere(24, 27, 7.0);
        ParamSet p = op.defaults();
        p.set("sigma", 0.0);
        p.set("opening", std::int64_t{0});
        p.set("min_voxels", std::int64_t{5});
        p.set("method", std::string("Manual"));
        p.set("value", 1500.0);
        p.set("post", std::string("Watershed (distance)"));
        p.set("seeds", std::string("H-maxima"));
        p.set("seed_depth", 2.5);
        const StepOutput whole = op.run(inputOf(data, meta), p, prog.ctx);
        REQUIRE(whole.labels);
        CHECK(whole.labels->stats().size() == 1);
        // a shallow depth lets every bump seed again: strictly more objects,
        // or the section is not testing the setting it names
        p.set("seed_depth", 0.2);
        const StepOutput split = op.run(inputOf(data, meta), p, prog.ctx);
        REQUIRE(split.labels);
        CHECK(split.labels->stats().size() > whole.labels->stats().size());
    }
}

TEST_CASE("The 3D filters ask whether they have been cancelled", "[app][ops][classic]") {
    // The vesselness and the thinning each run the whole volume, several times
    // over, between two of run()'s own checks. Without a poll of their own a
    // cancel is not noticed until they are finished, which on a full-size
    // stack is minutes: the run keeps working after the user stopped it.
    const Dims5 dims{1, 1, 6, 24, 24};
    const DatasetMeta meta = metaFor(dims);
    const Operation& op = requireOperation("classic");
    auto data = blobArray(dims, 2, 3.0);

    auto pollsOf = [&](const ParamSet& p) {
        Progress prog;
        int polls = 0;
        prog.ctx.cancelled = [&polls] {
            ++polls;
            return false;
        };
        op.run(inputOf(data, meta), p, prog.ctx);
        return polls;
    };

    ParamSet base = op.defaults();
    base.set("sigma", 0.0);
    base.set("opening", std::int64_t{0});
    base.set("post", std::string("Connected components"));
    const int plain = pollsOf(base);

    SECTION("the Frangi vesselness polls once per plane per scale") {
        ParamSet p = base;
        p.set("enhance", std::string("Tubes (Frangi)"));
        p.set("enhance_scales", std::int64_t{5});
        CHECK(pollsOf(p) >= plain + 5 * dims.z);
    }
    SECTION("Meijering does too") {
        ParamSet p = base;
        p.set("enhance", std::string("Neurites (Meijering)"));
        p.set("enhance_scales", std::int64_t{5});
        CHECK(pollsOf(p) >= plain + 5 * dims.z);
    }
    SECTION("the thinning polls once per direction of every pass") {
        ParamSet p = base;
        p.set("skeleton", true);
        CHECK(pollsOf(p) >= plain + 6);
    }
    SECTION("and a cancellation raised inside them stops the step") {
        ParamSet p = base;
        p.set("enhance", std::string("Tubes (Frangi)"));
        p.set("enhance_scales", std::int64_t{5});
        Progress prog;
        int polls = 0;
        // true only once the vesselness has started: the plain run never gets
        // this far, so nothing but a poll inside the filter can see it
        prog.ctx.cancelled = [&polls, plain] { return ++polls > plain; };
        CHECK_THROWS(op.run(inputOf(data, meta), p, prog.ctx));
    }
}


namespace {
    // A worker that answers hello / model_info / run(torch_segment) with a
    // probability map thresholding the input at 500 (plus a flat boundary
    // channel), using the public framing API.
    void fakeWorker(std::unique_ptr<rpc::Transport> transport) {
        std::vector<std::byte> inbox;
        for (;;) {
            std::optional<rpc::Message> msg;
            while (!(msg = rpc::decodeFrame(inbox))) {
                try {
                    if (!transport->receive(inbox, std::chrono::milliseconds(2000))) return;
                } catch (const std::exception&) {
                    return;
                }
            }
            const nlohmann::json& h = msg->header;
            const std::string method = h.value("method", "");
            nlohmann::json reply = {{"id", h.value("id", 0)}, {"type", "result"}};
            if (method == "hello") {
                reply["result"] = {{"version", "test"}, {"protocol_version", rpc::kProtocolVersion}, {"methods", {"run:torch_segment", "model_info"}}, {"cuda", false}, {"device", "cpu · fake"}, {"hostname", "fake"}, {"python", "3"}};
                transport->send(rpc::encodeFrame(reply, {}));
            } else if (method == "model_info") {
                reply["result"] = {{"format", "TorchScript"}, {"input_shape", {1, 1, "Z", "Y", "X"}}, {"input_dtype", "float32"}, {"output_shape", {1, 2, "Z", "Y", "X"}}, {"size_bytes", 41 * 1024 * 1024}};
                transport->send(rpc::encodeFrame(reply, {}));
            } else if (method == "run") {
                REQUIRE(msg->tensors.size() == 1);
                const rpc::Tensor& in = msg->tensors.front();
                nlohmann::json prog = {{"id", h.value("id", 0)}, {"type", "progress"}, {"fraction", 0.5}, {"message", "tile 1/2"}};
                transport->send(rpc::encodeFrame(prog, {}));
                const Index n = in.numel();
                std::vector<float> prob(static_cast<std::size_t>(2 * n));
                const float* v = in.asFloat32();
                for (Index i = 0; i < n; ++i) {
                    prob[static_cast<std::size_t>(i)] = v[i] > 500.0f ? 0.9f : 0.05f;
                    prob[static_cast<std::size_t>(n + i)] = 0.0f;
                }
                rpc::TensorRef out;
                out.name = "prob";
                out.dtype = "float32";
                out.shape = {2, in.shape[0], in.shape[1], in.shape[2]};
                out.data = prob.data();
                out.nbytes = prob.size() * sizeof(float);
                reply["result"] = {{"class_names", {"nucleus"}}};
                transport->send(rpc::encodeFrame(reply, {out}));
            } else if (method == "cancel") {
                // nothing running
            } else {
                reply["type"] = "error";
                reply["message"] = "unknown method " + method;
                transport->send(rpc::encodeFrame(reply, {}));
            }
        }
    }
} // namespace

TEST_CASE("Segmentation drives the worker protocol and labels the probabilities", "[app][ops][seg][rpc]") {
    auto pair = rpc::loopbackPair();
    std::thread worker(fakeWorker, std::move(pair.second));
    auto remote = std::make_unique<RemoteWorker>(std::move(pair.first));
    CHECK(remote->supports("torch_segment"));
    CHECK(remote->capabilities().hostname == "fake");

    const nlohmann::json info = torchModelInfo(*remote, "model.pt");
    CHECK(torchModelSummary(info) == "TorchScript · in (1, 1, Z, Y, X) float32 · out (1, 2, Z, Y, X) · 41 MB");

    const Dims5 dims{1, 1, 9, 40, 20};
    const DatasetMeta meta = metaFor(dims);
    auto data = blobArray(dims, 2, 3.0);
    const Operation& op = requireOperation("seg");
    ParamSet p = op.defaults();
    const test::TempFile model("app_ops_model", ".pt");
    { std::ofstream(model.path) << "fake"; }
    CHECK_FALSE(op.validate(p, meta).ok());   // no model
    p.set("model", model.str);
    p.set("post", std::string("Connected components"));
    REQUIRE(op.validate(p, meta).ok());
    CHECK(op.summary(p, meta).find("components") != std::string::npos);

    Progress prog;
    CHECK_THROWS(op.run(inputOf(data, meta), p, prog.ctx));   // no worker attached
    prog.ctx.remote = remote.get();
    const StepOutput r = op.run(inputOf(data, meta), p, prog.ctx);
    REQUIRE(r.labels);
    CHECK(r.labels->stats().size() == 2);
    CHECK(r.labels->stats().front().cls == "nucleus");
    CHECK(r.labels->stats().front().confidence > 0.8);
    CHECK(r.diagnostics.kind == DiagnosticsKind::Segment);
    CHECK(r.note.find("2 labels") != std::string::npos);
    CHECK(std::any_of(prog.fractions.begin(), prog.fractions.end(), [](double f) { return f > 0.0 && f < 1.0; }));
    REQUIRE(r.array);
    CHECK(r.array->dims() == dims);   // the intensities pass through

    SECTION("the watershed post-processing also yields the two blobs") {
        p.set("post", std::string("Watershed on boundary channel"));
        const StepOutput w = op.run(inputOf(data, meta), p, prog.ctx);
        REQUIRE(w.labels);
        CHECK(w.labels->stats().size() >= 2);
    }
    remote->close();
    worker.join();
}

namespace {
    // A worker standing in for a model family (Cellpose / micro-SAM): run
    // replies with instance labels -- voxels above 500 get id 1 in the lower
    // half of z and id 2 above -- plus, optionally, a one-channel
    // probability map; model_info reports the family and its availability.
    void fakeLabelWorker(std::unique_ptr<rpc::Transport> transport, bool withProb) {
        std::vector<std::byte> inbox;
        for (;;) {
            std::optional<rpc::Message> msg;
            while (!(msg = rpc::decodeFrame(inbox))) {
                try {
                    if (!transport->receive(inbox, std::chrono::milliseconds(2000))) return;
                } catch (const std::exception&) {
                    return;
                }
            }
            const nlohmann::json& h = msg->header;
            const std::string method = h.value("method", "");
            nlohmann::json reply = {{"id", h.value("id", 0)}, {"type", "result"}};
            if (method == "hello") {
                reply["result"] = {{"version", "test"}, {"protocol_version", rpc::kProtocolVersion}, {"methods", {"run:torch_segment", "model_info", "hub_search"}}, {"cuda", false}, {"device", "cpu · fake"}, {"hostname", "fake"}, {"python", "3"}};
                transport->send(rpc::encodeFrame(reply, {}));
            } else if (method == "model_info") {
                const std::string spec = h["params"].value("spec", "");
                reply["result"] = {{"format", "cellpose"}, {"model", "cyto3"}, {"available", spec.find("nuclei") == std::string::npos}, {"install_hint", "pip install cellpose"}, {"returns", "labels"}};
                transport->send(rpc::encodeFrame(reply, {}));
            } else if (method == "run") {
                REQUIRE(msg->tensors.size() == 1);
                CHECK(h["params"]["params"].value("model", "") == "cellpose:cyto3");
                const rpc::Tensor& in = msg->tensors.front();
                const Index z = in.shape[0], plane = in.shape[1] * in.shape[2];
                const Index n = in.numel();
                std::vector<std::uint32_t> lab(static_cast<std::size_t>(n));
                std::vector<float> prob(static_cast<std::size_t>(n));
                const float* v = in.asFloat32();
                for (Index i = 0; i < n; ++i) {
                    const bool fg = v[i] > 500.0f;
                    lab[static_cast<std::size_t>(i)] = fg ? (i / plane < z / 2 ? 1u : 2u) : 0u;
                    prob[static_cast<std::size_t>(i)] = fg ? 0.9f : 0.05f;
                }
                rpc::TensorRef labels;
                labels.name = "labels";
                labels.dtype = "uint32";
                labels.shape = {in.shape[0], in.shape[1], in.shape[2]};
                labels.data = lab.data();
                labels.nbytes = lab.size() * sizeof(std::uint32_t);
                rpc::TensorRef p;
                p.name = "prob";
                p.dtype = "float32";
                p.shape = {1, in.shape[0], in.shape[1], in.shape[2]};
                p.data = prob.data();
                p.nbytes = prob.size() * sizeof(float);
                reply["result"] = {{"labels", 2}, {"format", "cellpose"}, {"model", "cellpose:cyto3"}};
                std::vector<rpc::TensorRef> out = {labels};
                if (withProb) out.push_back(p);
                transport->send(rpc::encodeFrame(reply, out));
            } else if (method == "cancel") {
                // nothing running
            } else {
                reply["type"] = "error";
                reply["message"] = "unknown method " + method;
                transport->send(rpc::encodeFrame(reply, {}));
            }
        }
    }
} // namespace

TEST_CASE("Segmentation accepts hub and family model specs without a local file", "[app][ops][seg]") {
    const Dims5 dims{1, 1, 4, 8, 8};
    const DatasetMeta meta = metaFor(dims);
    const Operation& op = requireOperation("seg");
    ParamSet p = op.defaults();
    p.set("model", std::string("/nonexistent/model.pt"));
    CHECK_FALSE(op.validate(p, meta).ok());
    for (const char* spec : {"cellpose:cyto3", "microsam:vit_b_lm", "hf:owner/repo", "hf:owner/repo:weights/model.onnx", "CellPose:nuclei"}) {
        p.set("model", std::string(spec));
        CHECK(op.validate(p, meta).ok());
    }
    p.set("model", std::string("cellpose:cyto3"));
    CHECK(op.summary(p, meta).find("cellpose cyto3") != std::string::npos);
    CHECK(op.summary(p, meta).find("model labels") != std::string::npos);   // no watershed for family models
    p.set("model", std::string("microsam:vit_l_lm"));
    CHECK(op.summary(p, meta).find("micro-SAM vit_l_lm") != std::string::npos);
    p.set("model", std::string("hf:owner/repo:weights/model.onnx"));
    CHECK(op.summary(p, meta).find("hf model.onnx") != std::string::npos);
    CHECK(op.summary(p, meta).find("watershed") != std::string::npos);
    const ParamSpec* modelSpec = nullptr;
    for (const ParamSpec& s : op.info().params)
        if (s.key == "model") modelSpec = &s;
    REQUIRE(modelSpec);
    CHECK(modelSpec->help.find("hf:") != std::string::npos);
    CHECK(modelSpec->help.find("cellpose:") != std::string::npos);
    CHECK(modelSpec->help.find("microsam:") != std::string::npos);

    // family info from the worker: availability and the install hint
    CHECK(torchModelSummary({{"format", "cellpose"}, {"model", "cyto3"}, {"available", true}}) == "cellpose cyto3 · returns labels");
    // an installed package reports its version and whether the weights are on disk
    CHECK(torchModelSummary({{"format", "cellpose"}, {"model", "default"}, {"available", true}, {"version", "4.2.1"}, {"weights_cached", false}}) == "cellpose 4.2.1 default · returns labels · weights download on first run");
    CHECK(torchModelSummary({{"format", "cellpose"}, {"model", "cyto3"}, {"available", true}, {"version", "4.2.1"}, {"weights_cached", true}, {"warning", "cellpose 4.2.1 has no model 'cyto3'"}}) ==
          "cellpose 4.2.1 cyto3 · returns labels · weights cached · cellpose 4.2.1 has no model 'cyto3'");
    CHECK(torchModelSummary({{"format", "micro-sam"}, {"model", "vit_b_lm"}, {"available", false}, {"install_hint", "pip install micro-sam"}}) ==
          "micro-sam vit_b_lm · not installed (Hub… installs it: pip install micro-sam)");
    CHECK(torchModelSummary({{"format", "hf"}, {"repo", "owner/repo"}, {"available", true}, {"cached", false}}) ==
          "hf owner/repo · downloads on first run");
}

TEST_CASE("Segmentation takes instance labels from a family model", "[app][ops][seg][rpc]") {
    const bool withProb = GENERATE(true, false);
    auto pair = rpc::loopbackPair();
    std::thread worker(fakeLabelWorker, std::move(pair.second), withProb);
    auto remote = std::make_unique<RemoteWorker>(std::move(pair.first));
    CHECK(remote->supports("hub_search"));
    CHECK(torchModelSummary(torchModelInfo(*remote, "cellpose:cyto3")) == "cellpose cyto3 · returns labels");
    CHECK(torchModelSummary(torchModelInfo(*remote, "cellpose:nuclei")) == "cellpose cyto3 · not installed (Hub… installs it: pip install cellpose)");

    const Dims5 dims{1, 1, 9, 40, 20};
    const DatasetMeta meta = metaFor(dims);
    auto data = blobArray(dims, 2, 3.0);
    // what the fake worker labels: id 1 below the middle plane, id 2 from it on
    std::vector<std::uint32_t> expected(static_cast<std::size_t>(dims.z * dims.planeSize()));
    bool has1 = false, has2 = false;
    for (Index z = 0; z < dims.z; ++z)
        for (Index y = 0; y < dims.y; ++y)
            for (Index x = 0; x < dims.x; ++x) {
                const bool fg = data->at(0, 0, z, y, x) > 500.0f;
                const std::uint32_t id = fg ? (z < dims.z / 2 ? 1u : 2u) : 0u;
                expected[static_cast<std::size_t>((z * dims.y + y) * dims.x + x)] = id;
                has1 = has1 || id == 1;
                has2 = has2 || id == 2;
            }
    REQUIRE((has1 && has2));

    const Operation& op = requireOperation("seg");
    ParamSet p = op.defaults();
    p.set("model", std::string("cellpose:cyto3"));
    p.set("post", std::string("Watershed on boundary channel"));   // ignored: the labels come from the model
    p.set("threshold", 0.99);                                       // likewise
    REQUIRE(op.validate(p, meta).ok());
    Progress prog;
    prog.ctx.remote = remote.get();
    const StepOutput r = op.run(inputOf(data, meta), p, prog.ctx);
    REQUIRE(r.labels);
    CHECK(r.labels->stats().size() == 2);
    CHECK(r.labels->maxLabel() == 2);
    const std::uint32_t* got = r.labels->volume(0);
    CHECK(std::equal(expected.begin(), expected.end(), got));
    CHECK(r.labels->stats().front().cls == "nucleus");
    if (withProb) CHECK(r.labels->stats().front().confidence > 0.8);
    else CHECK(r.labels->stats().front().confidence == 1.0);   // unknown without a probability map
    CHECK(r.note.find("2 labels") != std::string::npos);
    CHECK(r.note.find("labels from the model") != std::string::npos);
    CHECK(r.diagnostics.kind == DiagnosticsKind::Segment);
    CHECK(r.diagnostics.summary.find("cellpose cyto3") != std::string::npos);

    SECTION("min_voxels still drops small objects") {
        p.set("min_voxels", std::int64_t{1000000});
        const StepOutput s = op.run(inputOf(data, meta), p, prog.ctx);
        REQUIRE(s.labels);
        CHECK(s.labels->stats().empty());
        CHECK(s.note.find("0 labels") != std::string::npos);
    }
    remote->close();
    worker.join();
}

// --- deconvolution ----------------------------------------------------------------

TEST_CASE("Deconvolve runs Richardson-Lucy with a theoretical PSF", "[app][ops][decon]") {
    const Dims5 dims{1, 1, 5, 16, 16};
    const DatasetMeta meta = metaFor(dims, 0.1, 0.3);
    auto data = blobArray(dims, 1, 3.0);
    const Operation& op = requireOperation("decon");
    ParamSet p = op.defaults();
    p.set("iterations", std::int64_t{5});
    p.set("tv_lambda", 0.0);
    p.set("psf_size", std::int64_t{9});
    REQUIRE(op.validate(p, meta).ok());
    CHECK(op.summary(p, meta).find("5 iter") != std::string::npos);
    Progress prog;
    const StepOutput r = op.run(inputOf(data, meta), p, prog.ctx);
    REQUIRE(r.array);
    CHECK(r.array->dims() == dims);
    CHECK(r.diagnostics.kind == DiagnosticsKind::Deconvolve);
    REQUIRE(r.diagnostics.curves.size() == 1);
    CHECK(r.diagnostics.curves[0].y.size() <= 5);
    CHECK_FALSE(r.diagnostics.curves[0].y.empty());
    CHECK_FALSE(r.diagnostics.images.empty());
    SECTION("a missing PSF file is an error") {
        p.set("psf", std::string("/nonexistent/psf.tif"));
        CHECK_FALSE(op.validate(p, meta).ok());
    }
}

TEST_CASE("Contrast window, Auto / Reset helpers and live preview", "[app][ops][contrast]") {
    const Operation& op = requireOperation("contrast");
    CHECK(op.info().livePreview);
    const Dims5 dims{1, 1, 2, 8, 8};
    auto arr = std::make_shared<Array5>(Array5::zeros(dims));
    for (Index i = 0; i < arr->numel(); ++i) arr->data()[i] = static_cast<float>(i) / static_cast<float>(arr->numel() - 1) * 10.0f;   // 0..10
    DatasetMeta meta;
    meta.dims = dims;
    meta.normalizeChannels();
    const StepInput in{meta, arr, nullptr, nullptr};

    ParamSet p = op.defaults();
    // the default (empty) window is automatic
    const ContrastWindow autoW = contrastWindow(in, p, 0, 0, true);
    CHECK(autoW.hi > autoW.lo);
    CHECK(op.summary(p, meta).rfind("auto", 0) == 0);
    p.set("min", 2.0);
    p.set("max", 4.0);
    const ContrastWindow w = contrastWindow(in, p, 0, 0, true);
    CHECK(w.lo == 2.0f);
    CHECK(w.hi == 4.0f);
    CHECK_THAT(w.dataMin, WithinAbs(0.0, 1e-6));
    CHECK_THAT(w.dataMax, WithinAbs(10.0, 1e-6));
    CHECK(op.summary(p, meta).find("window 2 – 4") != std::string::npos);

    Progress prog;
    const StepOutput out = op.run(in, p, prog.ctx);
    REQUIRE(out.array);
    CHECK(out.array->data()[0] == 0.0f);                                  // below min
    CHECK(out.array->data()[out.array->numel() - 1] == 1.0f);             // above max
    const Index mid = out.array->numel() * 3 / 10;                        // value ≈ 3 -> 0.5
    CHECK_THAT(out.array->data()[mid], WithinAbs(0.5, 0.05));

    SECTION("an empty window falls back to automatic instead of failing") {
        p.set("max", 2.0);
        CHECK(op.validate(p, meta).ok());
        const ContrastWindow e = contrastWindow(in, p, 0, 0);
        CHECK(e.hi > e.lo);
    }
    SECTION("Auto takes the percentiles, Reset the full range, a new step starts on Auto") {
        ParamSet a = p;
        a.set("lo_percentile", 5.0);    // 128 samples: the 0.2 / 99.8 defaults are the end points
        a.set("hi_percentile", 95.0);
        a = contrastAutoParams(a, in);
        CHECK(a.getDouble("min") > 0.0);
        CHECK(a.getDouble("max") < 10.0);
        CHECK(a.getDouble("min") < a.getDouble("max"));
        const ParamSet r = contrastResetParams(a, in);
        CHECK_THAT(r.getDouble("min"), WithinAbs(0.0, 1e-6));
        CHECK_THAT(r.getDouble("max"), WithinAbs(10.0, 1e-6));
        CHECK(r.getDouble("gamma") == 1.0);
        const ParamSet initial = op.initialParams(op.defaults(), in);
        CHECK(initial.getDouble("max") > initial.getDouble("min"));
        CHECK(initial.getDouble("max") > 5.0);
    }
}

TEST_CASE("A parameter can say which settings it applies to", "[app][ops][params]") {
    // The rule is a display concern only: the value is still stored and still
    // read, so switching the mode back finds it where it was left.
    ParamSet p;
    p.set("mode", std::string("From file"));
    p.set("seeds", std::string("H-maxima"));
    p.set("hysteresis", true);

    CHECK(doubleParam("plain", "Plain", 0.0).visibleFor(p));
    CHECK(doubleParam("a", "A", 0.0).visibleWhen("mode", {"From file"}).visibleFor(p));
    CHECK_FALSE(doubleParam("b", "B", 0.0).visibleWhen("mode", {"Estimate"}).visibleFor(p));
    CHECK(doubleParam("c", "C", 0.0).visibleWhen("mode", {"Estimate", "From file"}).visibleFor(p));
    CHECK_FALSE(doubleParam("d", "D", 0.0).hiddenWhen("mode", {"From file"}).visibleFor(p));
    CHECK(doubleParam("e", "E", 0.0).hiddenWhen("mode", {"Estimate"}).visibleFor(p));

    SECTION("every rule has to hold") {
        const ParamSpec both = doubleParam("f", "F", 0.0).visibleWhen("mode", {"From file"}).visibleWhen("seeds", {"H-maxima"});
        CHECK(both.visibleFor(p));
        ParamSet other = p;
        other.set("seeds", std::string("Distance maxima"));
        CHECK_FALSE(both.visibleFor(other));
    }

    SECTION("a bool reads as on / off") {
        CHECK(doubleParam("g", "G", 0.0).visibleWhen("hysteresis", {"on"}).visibleFor(p));
        ParamSet off = p;
        off.set("hysteresis", false);
        CHECK_FALSE(doubleParam("g", "G", 0.0).visibleWhen("hysteresis", {"on"}).visibleFor(off));
    }

    SECTION("a rule about a parameter that is not there decides nothing") {
        CHECK(doubleParam("h", "H", 0.0).visibleWhen("no_such_key", {"whatever"}).visibleFor(p));
    }
}

TEST_CASE("The operations hide the fields their mode ignores", "[app][ops][params]") {
    registerBuiltinOperations();
    auto shown = [](const Operation& op, const ParamSet& p) {
        std::set<std::string> out;
        for (const ParamSpec& s : op.info().params)
            if (s.visibleFor(p)) out.insert(s.key);
        return out;
    };

    SECTION("SIM in From file mode offers only what it still reads") {
        const Operation& sim = requireOperation("sim");
        ParamSet fromFile = sim.defaults();
        fromFile.set("mode", std::string("From file"));
        const std::set<std::string> keys = shown(sim, fromFile);
        // buildParameters replaces the whole parameter set from the file, so
        // everything it would have read from the form is ignored
        CHECK(keys.count("params_file") == 1);
        CHECK(keys.count("otf") == 1);
        CHECK(keys.count("dz_psf") == 1);   // applied in every mode
        for (const char* ignored : {"wiener", "angles", "phases", "na", "linespacing_um", "k0_angles", "zoomfact"})
            CHECK(keys.count(ignored) == 0);

        ParamSet estimate = sim.defaults();
        estimate.set("mode", std::string("Estimate"));
        const std::set<std::string> est = shown(sim, estimate);
        CHECK(est.count("wiener") == 1);
        CHECK(est.count("k0_start_angle") == 1);
        CHECK(est.count("params_file") == 0);
        CHECK(est.count("k0_angles") == 0);   // Manual only

        ParamSet manual = sim.defaults();
        manual.set("mode", std::string("Manual"));
        CHECK(shown(sim, manual).count("k0_angles") == 1);
    }

    SECTION("Classical hides the settings of the threshold it is not using") {
        const Operation& classic = requireOperation("classic");
        ParamSet otsu = classic.defaults();
        const std::set<std::string> plain = shown(classic, otsu);
        for (const char* ignored : {"value", "percentile", "window", "contrast_k", "local_ratio"})
            CHECK(plain.count(ignored) == 0);

        ParamSet local = classic.defaults();
        local.set("method", std::string("Local contrast"));
        const std::set<std::string> localKeys = shown(classic, local);
        CHECK(localKeys.count("window") == 1);
        CHECK(localKeys.count("contrast_k") == 1);
        CHECK(localKeys.count("local_ratio") == 0);   // Local mean only

        // the seed settings need both a watershed and that kind of seed
        ParamSet blobs = classic.defaults();
        blobs.set("post", std::string("Watershed (distance)"));
        blobs.set("seeds", std::string("Blob centres (LoG)"));
        CHECK(shown(classic, blobs).count("blob_radius") == 1);
        CHECK(shown(classic, blobs).count("seed_depth") == 0);
        ParamSet components = blobs;
        components.set("post", std::string("Connected components"));
        CHECK(shown(classic, components).count("blob_radius") == 0);
        CHECK(shown(classic, components).count("seeds") == 0);
    }

    SECTION("Track hides the other tracker's settings") {
        const Operation& track = requireOperation("track");
        ParamSet builtin = track.defaults();
        CHECK(shown(track, builtin).count("overlap_weight") == 1);
        CHECK(shown(track, builtin).count("config") == 0);
        ParamSet btrack = track.defaults();
        btrack.set("tracker", std::string("btrack (Bayesian)"));
        CHECK(shown(track, btrack).count("config") == 1);
        CHECK(shown(track, btrack).count("overlap_weight") == 0);
    }

    SECTION("the scikit-image step shows one method's settings at a time") {
        const Operation& sk = requireOperation("skimage_seg");
        ParamSet walker = sk.defaults();
        CHECK(shown(sk, walker).count("beta") == 1);
        CHECK(shown(sk, walker).count("n_segments") == 0);
        ParamSet slic = sk.defaults();
        slic.set("method", std::string("Superpixels (SLIC)"));
        CHECK(shown(sk, slic).count("n_segments") == 1);
        CHECK(shown(sk, slic).count("beta") == 0);
        CHECK(shown(sk, slic).count("compactness") == 1);   // shared with the compact watershed
    }
}

TEST_CASE("Every preset names real parameters and leaves the step runnable", "[app][ops][params]") {
    registerBuiltinOperations();
    int withPresets = 0;
    for (const Operation* op : allOperations()) {
        const std::string kind = op->kind();
        if (op->info().presets.empty()) continue;
        ++withPresets;
        std::set<std::string> names;
        for (const ParamPreset& preset : op->info().presets) {
            INFO(kind << " preset " << preset.name);
            CHECK_FALSE(preset.name.empty());
            CHECK_FALSE(preset.summary.empty());
            CHECK(names.insert(preset.name).second);   // one entry per name
            CHECK_FALSE(preset.values.empty());
            // a preset that names a parameter the operation does not have would
            // silently do nothing
            ParamSet p = op->defaults();
            for (const auto& [key, value] : preset.values) {
                bool known = false;
                for (const ParamSpec& s : op->info().params)
                    if (s.key == key) known = true;
                CHECK(known);
                p.set(key, value);
            }
            // and survive coercion unchanged: a value the spec would clamp or
            // reject is a preset that does not do what it says
            p.coerce(op->info().params);
            for (const auto& [key, value] : preset.values) {
                INFO("after coercion: " << key);
                const ParamValue* got = p.find(key);
                REQUIRE(got != nullptr);
                CHECK(toDisplayString(*got) == toDisplayString(value));
            }
        }
    }
    CHECK(withPresets >= 1);
}

// --- regressions ----------------------------------------------------------------

TEST_CASE("Frangi finds a line on a single plane", "[app][ops][classic]") {
    // One plane: the 3D measure's third eigenvalue is identically zero and it
    // read every line as "no tube"; the 2D measure is what a single plane needs.
    const Dims5 dims{1, 1, 1, 48, 48};
    const DatasetMeta meta = metaFor(dims);
    auto data = std::make_shared<Array5>(Array5::zeros(dims));
    for (Index y = 23; y <= 25; ++y)
        for (Index x = 4; x < 44; ++x) data->at(0, 0, 0, y, x) = 1000.0f;
    const Operation& op = requireOperation("classic");
    ParamSet p = op.defaults();
    p.set("enhance", std::string("Tubes (Frangi)"));
    p.set("enhance_sigma", 1.0);
    p.set("enhance_sigma_max", 3.0);
    p.set("enhance_scales", std::int64_t{3});
    p.set("sigma", 0.0);
    p.set("opening", std::int64_t{0});
    p.set("fill_holes", false);
    p.set("method", std::string("Otsu"));
    p.set("post", std::string("Connected components"));
    p.set("min_voxels", std::int64_t{5});
    Progress prog;
    const StepOutput r = op.run(inputOf(data, meta), p, prog.ctx);
    REQUIRE(r.labels);
    CHECK(r.labels->at(0, 0, 24, 24) != 0);   // the middle of the line
    CHECK(r.labels->at(0, 0, 24, 12) != 0);
    CHECK(r.labels->at(0, 0, 5, 5) == 0);     // the background
    CHECK(r.labels->at(0, 0, 40, 40) == 0);
}

TEST_CASE("Label cleanup keeps ids and confidences when asked", "[app][ops][cleanup]") {
    const Dims5 dims{1, 1, 1, 8, 8};
    const DatasetMeta meta = metaFor(dims);
    auto data = std::make_shared<Array5>(Array5::zeros(dims));
    auto labels = std::make_shared<LabelVolume>(1, 1, 8, 8);
    std::uint32_t* v = labels->volume(0);
    for (Index y = 1; y <= 3; ++y)
        for (Index x = 1; x <= 3; ++x) v[y * 8 + x] = 5;   // nine voxels
    for (Index y = 5; y <= 6; ++y)
        for (Index x = 5; x <= 6; ++x) v[y * 8 + x] = 9;   // four voxels
    v[0 * 8 + 7] = 7;                                       // a speck
    labels->recomputeStats(0);
    for (LabelStats& s : labels->stats())
        if (s.id == 9) s.confidence = 0.3;   // as a segmentation model would have reported it
    const Operation& cleanup = requireOperation("cleanup");
    Progress prog;
    StepInput in = inputOf(data, meta);
    in.labels = labels;

    SECTION("relabel off drops the small one and leaves the other ids alone") {
        ParamSet cp = cleanup.defaults();
        cp.set("min_voxels", std::int64_t{2});
        cp.set("relabel", false);
        cp.set("low_conf", 0.6);
        const StepOutput out = cleanup.run(in, cp, prog.ctx);
        REQUIRE(out.labels);
        CHECK(out.labels->at(0, 0, 2, 2) == 5);
        CHECK(out.labels->at(0, 0, 5, 5) == 9);
        CHECK(out.labels->at(0, 0, 0, 7) == 0);
        CHECK(out.labels->stats().size() == 2);
        // the confidence survived the pass and the flag it earns is set
        const LabelStats* nine = out.labels->statsOf(9);
        REQUIRE(nine);
        CHECK(nine->confidence == 0.3);
        CHECK(std::find(nine->flags.begin(), nine->flags.end(), "low conf") != nine->flags.end());
        const LabelStats* five = out.labels->statsOf(5);
        REQUIRE(five);
        CHECK(five->confidence == 1.0);
    }
    SECTION("relabel on numbers what is left densely") {
        ParamSet cp = cleanup.defaults();
        cp.set("min_voxels", std::int64_t{2});
        cp.set("relabel", true);
        const StepOutput out = cleanup.run(in, cp, prog.ctx);
        REQUIRE(out.labels);
        CHECK(out.labels->at(0, 0, 2, 2) == 1);
        CHECK(out.labels->at(0, 0, 5, 5) == 2);
        CHECK(out.labels->maxLabel() == 2);
    }
    SECTION("recomputeStats without probabilities keeps a known confidence") {
        labels->recomputeStats(0);
        REQUIRE(labels->statsOf(9));
        CHECK(labels->statsOf(9)->confidence == 0.3);
    }
}

TEST_CASE("Flat-field checks the dark image's size like the flat's", "[app][ops][flatfield]") {
    const Dims5 dims{1, 1, 2, 4, 4};
    const DatasetMeta meta = metaFor(dims);
    Buffer<float> flat(Shape{4, 4});
    for (Index i = 0; i < 16; ++i) flat.data()[i] = 2.0f;
    Buffer<float> dark(Shape{2, 2});
    for (Index i = 0; i < 4; ++i) dark.data()[i] = 1.0f;
    const test::TempFile flatFile("app_ops_flat_ok", ".tif"), darkFile("app_ops_dark_small", ".tif");
    writeTiff<float>(flatFile.str, flat.view());
    writeTiff<float>(darkFile.str, dark.view());
    const Operation& op = requireOperation("flatfield");
    ParamSet p = op.defaults();
    p.set("flat", flatFile.str);
    REQUIRE(op.validate(p, meta).ok());
    p.set("dark", darkFile.str);
    const Validation v = op.validate(p, meta);
    CHECK_FALSE(v.ok());
    CHECK(v.firstError().find("dark") != std::string::npos);
    // run() validates first: a 2 x 2 dark under 4 x 4 data would be read past its end
    Progress prog;
    auto data = std::make_shared<Array5>(Array5::filled(dims, 10.0f));
    CHECK_THROWS(op.run(inputOf(data, meta), p, prog.ctx));
}

TEST_CASE("Deskew takes 0 as the dataset's own sheet angle", "[app][ops][deskew]") {
    const Dims5 dims{1, 1, 6, 8, 10};
    DatasetMeta meta = metaFor(dims, 0.1, 0.4);
    meta.lightSheet = true;
    meta.sheetAngleDeg = 31.8;
    const Operation& op = requireOperation("deskew");
    ParamSet zero = op.defaults();
    zero.set("sheet_angle", 0.0);
    zero.coerce(op.info().params);   // a TOML file's 0 arrives through here
    CHECK(zero.getDouble("sheet_angle") == 0.0);   // was clamped to 1 degree
    ParamSet explicitAngle = op.defaults();
    explicitAngle.set("sheet_angle", 31.8);
    CHECK(op.outputMeta(zero, meta).dims == op.outputMeta(explicitAngle, meta).dims);
}

TEST_CASE("Contrast's summary does not promise a per-channel window", "[app][ops][contrast]") {
    const Operation& op = requireOperation("contrast");
    const DatasetMeta meta = metaFor(Dims5{2, 1, 2, 4, 4});
    ParamSet p = op.defaults();
    CHECK(op.summary(p, meta).find("per channel") == std::string::npos);
}

TEST_CASE("Register moves the labels with the time points it aligns", "[app][ops][register][labels]") {
    const Dims5 dims{1, 2, 1, 48, 48};
    const DatasetMeta meta = metaFor(dims);
    auto data = std::make_shared<Array5>(Array5::zeros(dims));
    // the blobs of t = 0 sit at (y + 3, x - 4) in t = 1
    const int pts[][2] = {{10, 12}, {30, 20}, {22, 36}, {38, 40}, {14, 30}};
    for (const auto& pt : pts)
        for (Index y = 0; y < 48; ++y)
            for (Index x = 0; x < 48; ++x) {
                const double d0 = std::hypot(y - pt[0], x - pt[1]);
                data->at(0, 0, 0, y, x) += static_cast<float>(100.0 * std::exp(-d0 * d0 / 8.0));
                const double d1 = std::hypot(y - (pt[0] + 3), x - (pt[1] - 4));
                data->at(0, 1, 0, y, x) += static_cast<float>(100.0 * std::exp(-d1 * d1 / 8.0));
            }
    // a label on the first blob in both frames, where the blob is in each
    auto labels = std::make_shared<LabelVolume>(2, 1, 48, 48);
    for (Index y = 9; y <= 11; ++y)
        for (Index x = 11; x <= 13; ++x) {
            labels->volume(0)[y * 48 + x] = 7;
            labels->volume(1)[(y + 3) * 48 + (x - 4)] = 7;
        }
    const Operation& op = requireOperation("register");
    ParamSet p = op.defaults();
    p.set("mode", std::string("Align time points to reference"));
    p.set("reference_t", std::int64_t{0});
    p.set("max_shift", std::vector<double>{0.0, 8.0, 8.0});
    REQUIRE(op.validate(p, meta).ok());
    Progress prog;
    StepInput in = inputOf(data, meta);
    in.labels = labels;
    const StepOutput r = op.run(in, p, prog.ctx);
    REQUIRE(r.array);
    CHECK(r.array->at(0, 1, 0, 10, 12) > 80.0f);   // the frame moved onto the reference
    REQUIRE(r.labels);
    CHECK(r.labels != labels);                      // a copy: the input's labels are untouched
    CHECK(labels->at(1, 0, 13, 8) == 7);
    CHECK(r.labels->at(1, 0, 10, 12) == 7);         // and so did its label
    CHECK(r.labels->at(1, 0, 13, 8) == 0);
    CHECK(r.labels->at(0, 0, 10, 12) == 7);         // the reference frame stays
}
