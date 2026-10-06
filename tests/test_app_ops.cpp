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
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <thread>

#include <sirius/tiff_io.hpp>

#include "core/array_source.hpp"
#include "core/cancel.hpp"
#include "core/executor.hpp"
#include "core/model_folder.hpp"
#include "core/ops/common.hpp"
#include "core/ops/contrast.hpp"
#include "core/ops/sim_params.hpp"
#include "core/ops/torch_model.hpp"
#include "core/ops/builtin.hpp"
#include "core/pipeline.hpp"
#include "core/rpc.hpp"

#include <set>

#include "sim_synthetic.hpp"
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
                           "foundation", "classic", "cleanup"};
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
    REQUIRE(out.array);   // full load is the default
    Buffer<float> vol = out.asInput().readVolume(0, 0);
    CHECK(vol.shape() == Shape{135, 64, 64});

    SECTION("Lazy keeps the pixels on disk") {
        p.set("read_as", std::string("Lazy (chunk on demand)"));
        const StepOutput lazy = load.run(StepInput{}, p, prog.ctx);
        REQUIRE(lazy.source);
        CHECK_FALSE(lazy.array);
    }

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
    // k0 is in px^-1 of the raw pixel; line spacing is dx / k0, the same
    // window the library locks against the cudasirecon fit.
    for (const std::vector<std::string>& row : out.diagnostics.table->rows) {
        const double k0px = std::stod(row[1]);
        const double spacingUm = 0.08 / k0px;
        CHECK(spacingUm > 0.40);
        CHECK(spacingUm < 0.415);
    }
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

TEST_CASE("SIM From file keeps the file's OTF axial step unless the field is set", "[app][ops][sim]") {
    const test::TempFile cfg("sim_dzpsf", ".txt");
    {
        std::ofstream(cfg.path) << "nphases=5\nndirs=3\nna=1.2\nnimm=1.33\nxyres=0.1\nzres=0.2\nzresPSF=0.5\nls=0.2\n";
    }
    DatasetMeta meta = metaFor(Dims5{1, 1, 15, 8, 8}, 0.1, 0.2);
    meta.sim.present = true;
    meta.sim.ndirs = 3;
    meta.sim.nphases = 5;
    const Operation& sim = requireOperation("sim");
    ParamSet fromFile = sim.defaults();
    fromFile.set("mode", std::string("From file"));
    fromFile.set("params_file", cfg.str);
    SIMParameters kept = simParametersFromStep(fromFile, meta);
    CHECK_THAT(kept.dz_psf, WithinAbs(0.5, 1e-9));
    fromFile.set("dz_psf", 0.3);
    SIMParameters overridden = simParametersFromStep(fromFile, meta);
    CHECK_THAT(overridden.dz_psf, WithinAbs(0.3, 1e-9));

    ParamSet missing = sim.defaults();
    missing.set("mode", std::string("From file"));
    const test::TempFile bare("sim_dzpsf_bare", ".txt");
    std::ofstream(bare.path) << "nphases=5\nndirs=3\nna=1.2\nnimm=1.33\nxyres=0.1\nzres=0.2\nls=0.2\n";
    missing.set("params_file", bare.str);
    SIMParameters fromStack = simParametersFromStep(missing, meta);
    CHECK_THAT(fromStack.dz_psf, WithinAbs(meta.dz(), 1e-9));

    // An inline table and a dotted key are both assignments the loader reads.
    // A scanner that only looks at the first '=' on a line misses them and
    // replaces the file's step with the stack dz.
    const test::TempFile inlined("sim_dzpsf_inline", ".toml");
    std::ofstream(inlined.path) << "pixels = { dx = 0.1, dy = 0.1, dz = 0.2, dz_psf = 0.55 }\n"
                                    "[optics]\nndirs = 3\nnphases = 5\nna = 1.2\nnimm = 1.33\nlinespacing_um = 0.2\n";
    ParamSet inlineFile = sim.defaults();
    inlineFile.set("mode", std::string("From file"));
    inlineFile.set("params_file", inlined.str);
    CHECK_THAT(simParametersFromStep(inlineFile, meta).dz_psf, WithinAbs(0.55, 1e-6));
    const test::TempFile dotted("sim_dzpsf_dotted", ".toml");
    std::ofstream(dotted.path) << "pixels.dx = 0.1\npixels.dy = 0.1\npixels.dz = 0.2\npixels.dz_psf = 0.45\n"
                                   "[optics]\nndirs = 3\nnphases = 5\nna = 1.2\nnimm = 1.33\nlinespacing_um = 0.2\n";
    ParamSet dot = sim.defaults();
    dot.set("mode", std::string("From file"));
    dot.set("params_file", dotted.str);
    CHECK_THAT(simParametersFromStep(dot, meta).dz_psf, WithinAbs(0.45, 1e-6));

    ParamSet estimate = sim.defaults();
    estimate.set("mode", std::string("Estimate"));
    SIMParameters stacked = simParametersFromStep(estimate, meta);
    CHECK_THAT(stacked.dz_psf, WithinAbs(0.2, 1e-9));
    estimate.set("dz_psf", 0.3);
    SIMParameters estimateOverride = simParametersFromStep(estimate, meta);
    CHECK_THAT(estimateOverride.dz_psf, WithinAbs(0.3, 1e-9));
}

TEST_CASE("SIM reconstructs a 2D stack with the step's defaults", "[app][ops][sim][2d]") {
    // The step's defaults skip the kz = 0 plane and damp the zero order, and
    // a 2D stack has no other plane: the result used to be all NaN. The scene
    // is sim_synthetic.hpp's; only the optics are set, every switch keeps its
    // default.
    SIMParameters optics;
    optics.ndirs = 3;
    optics.nphases = 3;
    optics.na = 1.2;
    optics.nimm = 1.33;
    optics.wavelength_nm = 530.0;
    optics.linespacing_um = 0.30;
    optics.k0_start_angle = 0.3;
    optics.dx = 0.08;
    optics.dy = 0.08;
    const Buffer<double> raw = test::syntheticSim2d(optics, 128);

    const Dims5 dims{1, 1, 9, 128, 128};
    DatasetMeta meta = metaFor(dims, 0.08, 0.3);
    meta.sim.present = true;
    meta.sim.ndirs = 3;
    meta.sim.nphases = 3;
    auto array = std::make_shared<Array5>(dims);
    for (Index z = 0; z < dims.z; ++z)
        for (Index y = 0; y < dims.y; ++y)
            for (Index x = 0; x < dims.x; ++x)
                array->at(0, 0, z, y, x) = static_cast<float>(raw.data()[(z * dims.y + y) * dims.x + x]);

    const Operation& sim = requireOperation("sim");
    auto run = [&](const ParamSet& sp) {
        const Validation v = sim.validate(sp, meta);
        INFO(v.firstError());
        REQUIRE(v.ok());
        Progress prog;
        StepOutput out = sim.run(inputOf(array, meta), sp, prog.ctx);
        REQUIRE(out.array);
        CHECK(out.array->dims() == Dims5{1, 1, 1, 256, 256});
        Index bad = 0;
        for (Index i = 0; i < 256 * 256; ++i) bad += std::isfinite(out.array->plane(0, 0, 0)[i]) ? 0 : 1;
        CHECK(bad == 0);
        return out;
    };

    SECTION("Estimate mode finds the simulated pattern") {
        ParamSet sp = sim.defaults();
        CHECK(sp.getBool("no_kz0", false));
        CHECK(sp.getBool("suppress_zero_order", false));
        sp.set("phases", std::int64_t{3});
        sp.set("na", 1.2);
        sp.set("nimm", 1.33);
        sp.set("wavelength_nm", 530.0);
        sp.set("linespacing_um", 0.30);
        sp.set("k0_start_angle", 0.3 * 180.0 / kPi);   // degrees
        const StepOutput out = run(sp);
        REQUIRE(out.diagnostics.table);
        const auto& rows = out.diagnostics.table->rows;
        REQUIRE(rows.size() == 3);
        CHECK(rows[0][0] == "17°");
        CHECK(rows[1][0] == "77°");
        CHECK(rows[2][0] == "137°");
    }
    SECTION("the form refuses what used to crash the run") {
        // one order segfaulted the k0 fit; an NA above the immersion index
        // segfaulted the filter (with a measured OTF, which is not needed to
        // see the form refuse it)
        ParamSet orders = sim.defaults();
        orders.set("phases", std::int64_t{3});
        orders.set("orders", std::int64_t{1});
        CHECK_FALSE(sim.validate(orders, meta).ok());
        ParamSet na = sim.defaults();
        na.set("phases", std::int64_t{3});
        na.set("na", 1.6);
        const Validation v = sim.validate(na, meta);
        REQUIRE_FALSE(v.ok());
        CHECK(v.firstError().find("nimm") != std::string::npos);
    }
    SECTION("From file mode with a TOML file that leaves the orders out") {
        // the form validated, and the run threw "3 phases cannot separate 3 orders"
        const test::TempFile toml("sim2d", ".toml");
        std::ofstream(toml.path) << "[optics]\nndirs = 3\nnphases = 3\nlinespacing_um = 0.30\nk0_start_angle = 0.3\n"
                                    "na = 1.2\nnimm = 1.33\nwavelength_nm = 530.0\n";
        ParamSet sp = sim.defaults();
        sp.set("mode", std::string("From file"));
        sp.set("params_file", toml.str);
        run(sp);
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

TEST_CASE("Infinite voxels leave the contrast window and the Otsu cuts to the finite values", "[app][ops][contrast][classic]") {
    // One +-inf voxel crashed the application as a dataset opened: the
    // histograms spanned [min, inf], and (inf - lo) * (bins / inf) is a NaN
    // bin index that was written outside the counts.
    const float inf = std::numeric_limits<float>::infinity();
    const Dims5 dims{1, 1, 9, 40, 20};
    const DatasetMeta meta = metaFor(dims);
    const auto finite = blobArray(dims, 3, 3.0);   // 0 and 1000
    auto data = std::make_shared<Array5>(finite->clone());
    data->at(0, 0, 4, 1, 1) = inf;
    data->at(0, 0, 0, 39, 19) = -inf;
    Progress prog;

    SECTION("Otsu's cut is the cut of the finite values") {
        std::vector<float> values(finite->data(), finite->data() + finite->numel());
        const float cut = otsuThreshold(values.data(), static_cast<Index>(values.size()));
        CHECK(cut > 0.0f);
        CHECK(cut < 1000.0f);
        values.push_back(inf);
        values.push_back(-inf);
        CHECK(otsuThreshold(values.data(), static_cast<Index>(values.size())) == cut);
        // nothing finite: a cut nothing lies above
        const std::vector<float> none{inf, -inf, std::numeric_limits<float>::quiet_NaN()};
        CHECK(otsuThreshold(none.data(), 3) == inf);
    }
    SECTION("Classic with a plain Otsu cut labels the blobs and the +inf voxel") {
        const Operation& op = requireOperation("classic");
        ParamSet p = op.defaults();
        p.set("sigma", 0.0);
        p.set("opening", std::int64_t{0});
        p.set("fill_holes", false);
        p.set("post", std::string("Connected components"));
        p.set("method", std::string("Otsu"));
        p.set("min_voxels", std::int64_t{0});
        const StepOutput r = op.run(inputOf(data, meta), p, prog.ctx);
        REQUIRE(r.labels);
        CHECK(r.labels->stats().size() == 4);
        CHECK(r.labels->at(0, 4, 1, 1) != 0);
        CHECK(r.labels->at(0, 0, 39, 19) == 0);
    }
    SECTION("Classic Multi-Otsu still separates the core from the halo") {
        const Dims5 d3{1, 1, 8, 48, 48};
        auto three = std::make_shared<Array5>(Array5::zeros(d3));
        for (Index z = 2; z < 5; ++z)
            for (Index y = 8; y < 40; ++y)
                for (Index x = 8; x < 40; ++x) three->at(0, 0, z, y, x) = 400.0f;   // halo
        for (Index y = 20; y < 26; ++y)
            for (Index x = 20; x < 26; ++x) three->at(0, 0, 3, y, x) = 4000.0f;   // core
        three->at(0, 0, 0, 0, 0) = inf;
        three->at(0, 0, 7, 47, 47) = -inf;
        const Operation& op = requireOperation("classic");
        ParamSet p = op.defaults();
        p.set("sigma", 0.0);
        p.set("opening", std::int64_t{0});
        p.set("fill_holes", false);
        p.set("min_voxels", std::int64_t{1});
        p.set("post", std::string("Connected components"));
        p.set("method", std::string("Multi-Otsu"));
        const StepOutput r = op.run(inputOf(three, metaFor(d3)), p, prog.ctx);
        REQUIRE(r.labels);
        CHECK(r.labels->stats().size() == 2);        // the core and the +inf voxel
        CHECK(r.labels->at(0, 3, 22, 22) != 0);
        CHECK(r.labels->at(0, 3, 10, 10) == 0);      // the halo
        CHECK(r.labels->at(0, 0, 0, 0) != 0);
        CHECK(r.labels->at(0, 7, 47, 47) == 0);
    }
    SECTION("Contrast opens on them and maps them to the ends of the window") {
        const Operation& op = requireOperation("contrast");
        const StepInput in = inputOf(data, meta);
        const ParamSet p = op.initialParams(op.defaults(), in);   // what a new step (or a dataset opening) computes
        CHECK(p.getDouble("min", -1.0) == 0.0);
        CHECK(p.getDouble("max", -1.0) == 1000.0);
        const StepOutput out = op.run(in, p, prog.ctx);
        REQUIRE(out.array);
        CHECK(out.array->at(0, 0, 4, 1, 1) == 1.0f);
        CHECK(out.array->at(0, 0, 0, 39, 19) == 0.0f);
        REQUIRE(out.diagnostics.histograms.size() == 1);
        const DiagnosticHistogram& h = out.diagnostics.histograms[0];
        CHECK(h.binLo == 0.0f);
        CHECK(h.binHi == 1000.0f);
        CHECK(std::accumulate(h.bins.begin(), h.bins.end(), 0.0) == static_cast<double>(data->numel() - 2));
        CHECK(contrastPreview(in, p).histograms.size() == 1);
    }
}

TEST_CASE("Otsu and Multi-Otsu break exact ties the way bindings/tests/test_workbench.py expects", "[app][ops][classic]") {
    // A histogram symmetric about its centre scores a split and its mirror
    // image exactly the same; which one wins is decided by the last bit of the
    // between-class variance, so the Python mirror has to evaluate it in the
    // order these loops do. Same data and cuts as
    // TestIntensityHelpers.test_otsu_cuts_break_ties_as_the_application.
    // Values sit at bin centres of [0, bins]: `pairs` holds (bin, count), each
    // mirrored to bins - 1 - bin, and the two ends pin the range.
    const auto symmetric = [](float bins, int ends, std::initializer_list<std::pair<int, int>> pairs) {
        std::vector<float> values(static_cast<std::size_t>(ends), 0.0f);
        values.insert(values.end(), static_cast<std::size_t>(ends), bins);
        for (const auto& [bin, count] : pairs)
            for (int i = 0; i < count; ++i) {
                values.push_back(static_cast<float>(bin) + 0.5f);
                values.push_back(bins - 1.0f - static_cast<float>(bin) + 0.5f);
            }
        return values;
    };
    const auto labelsAbove = [](const char* kind, ParamSet p, const std::vector<float>& values, float cut) {
        const Dims5 dims{1, 1, 1, 1, static_cast<Index>(values.size())};
        auto data = std::make_shared<Array5>(dims);
        std::copy(values.begin(), values.end(), data->data());
        const Operation& op = requireOperation(kind);
        ParamSet params = op.defaults();
        for (const auto& [key, value] : p.items()) params.set(key, value);
        Progress prog;
        const StepOutput r = op.run(inputOf(data, metaFor(dims)), params, prog.ctx);
        REQUIRE(r.labels);
        for (std::size_t i = 0; i < values.size(); ++i) {
            INFO(kind << ": value " << values[i]);
            CHECK((r.labels->at(0, 0, 0, static_cast<Index>(i)) != 0) == (values[i] > cut));
        }
    };
    SECTION("Otsu: 0 + 256 * 37 / 256, not the 139 of the mirror image") {
        ParamSet p;
        p.set("sigma", 0.0);
        p.set("opening", std::int64_t{0});
        p.set("fill_holes", false);
        p.set("post", std::string("Connected components"));
        p.set("method", std::string("Otsu"));
        p.set("min_voxels", std::int64_t{0});
        labelsAbove("classic", p, symmetric(256.0f, 3, {{36, 6}, {117, 17}}), 37.0f);
    }
    SECTION("Multi-Otsu: 0 + 128 * 93 / 128, not the 71 of the mirror image") {
        ParamSet p;
        p.set("sigma", 0.0);
        p.set("opening", std::int64_t{0});
        p.set("fill_holes", false);
        p.set("min_voxels", std::int64_t{1});
        p.set("post", std::string("Connected components"));
        p.set("method", std::string("Multi-Otsu"));
        labelsAbove("classic", p, symmetric(128.0f, 1, {{2, 2}, {36, 2}, {44, 6}, {58, 5}}), 93.0f);
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
    p.set("interpolation", std::string("nearest"));
    p.set("sheet_angle", 31.8);
    p.set("stage_step_um", 0.4);
    CHECK_FALSE(op.validate(p, meta).warnings.empty());
    // The warning says it will shear; the note must describe that shear, not claim it was skipped.
    CHECK(op.summary(p, meta).find("skipped") == std::string::npos);
    CHECK(op.summary(p, meta).find("31.8") != std::string::npos);
    CHECK(op.summary(p, meta).find("shear only") != std::string::npos);
    meta.lightSheet = true;
    meta.sheetAngleDeg = 31.8;
    CHECK(op.validate(p, meta).ok());
    const DatasetMeta out = op.outputMeta(p, meta);
    CHECK(out.dims.x > dims.x);   // the shear widens x
    CHECK_FALSE(out.lightSheet);

    // A marker at x = 0 of each plane lands at x = z * stageStep * cos(angle) / dx.
    auto marked = std::make_shared<Array5>(Array5::zeros(dims));
    for (Index z = 0; z < dims.z; ++z) marked->at(0, 0, z, 0, 0) = 1.0f;
    DatasetMeta plain = metaFor(dims, 0.1, 0.4);   // not marked light-sheet
    Progress prog;
    const StepOutput r = op.run(inputOf(marked, plain), p, prog.ctx);
    REQUIRE(r.array);
    CHECK(r.array->dims() == op.outputMeta(p, plain).dims);
    CHECK(r.note.find("skipped") == std::string::npos);
    const double shear = 0.4 * std::cos(31.8 * 3.14159265358979323846 / 180.0) / 0.1;
    for (Index z = 0; z < dims.z; ++z) {
        const Index expectX = static_cast<Index>(std::lround(z * shear));
        Index argmax = 0;
        float best = r.array->at(0, 0, z, 0, 0);
        for (Index x = 1; x < r.array->dims().x; ++x) {
            const float v = r.array->at(0, 0, z, 0, x);
            if (v > best) {
                best = v;
                argmax = x;
            }
        }
        CHECK(argmax == expectX);
        CHECK(best > 0.5f);
    }
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

TEST_CASE("Crop / pad keeps what was said about the objects it carries", "[app][ops][croppad][labels]") {
    const Dims5 dims{1, 2, 1, 12, 12};
    const DatasetMeta meta = metaFor(dims);
    const Operation& op = requireOperation("croppad");
    ParamSet p = op.defaults();
    p.set("y0", std::int64_t{2});
    p.set("x0", std::int64_t{2});
    p.set("y", std::int64_t{8});
    p.set("x", std::int64_t{8});
    StepInput in = inputOf(rampArray(dims), meta);
    // a track of two frames, as the tracking step leaves it, with a review
    // mark, a model confidence and flag rules of its own
    auto labels = std::make_shared<LabelVolume>(2, 1, 12, 12);
    for (Index t = 0; t < 2; ++t) {
        for (Index y = 4; y < 7; ++y)
            for (Index x = 4 + t; x < 7 + t; ++x) labels->volume(t)[y * 12 + x] = 3;
        labels->recomputeStats(t);
        for (LabelStats& s : labels->stats()) {
            s.cls = "track";
            s.confidence = t == 0 ? 0.4 : 0.9;
        }
    }
    labels->setTracked(true);
    LabelFlagRules rules;
    rules.flagBorder = false;
    rules.lowConfidence = 0.5;
    labels->applyFlags(rules);
    labels->recomputeStats(0);
    labels->stats().front().reviewed = true;
    in.labels = labels;
    Progress prog;
    const StepOutput out = op.run(in, p, prog.ctx);
    REQUIRE(out.labels);
    CHECK(out.labels->tracked());   // a delete on it still takes the whole track
    CHECK(out.labels->statsT() == 0);
    const LabelStats* s = out.labels->statsOf(3);
    REQUIRE(s);
    CHECK(s->cls == "track");
    CHECK(s->reviewed);
    CHECK(s->confidence == 0.4);
    CHECK(s->bbox == std::array<Index, 6>{0, 1, 2, 5, 2, 5});   // measured on the new grid
    CHECK(std::find(s->flags.begin(), s->flags.end(), "low conf") != s->flags.end());   // the rules came along
    CHECK(out.labels->flaggedCount("touching border") == 0);
    CHECK(out.labels->annotationOf(1, 3).confidence == 0.9);   // and the other frame's own
    CHECK(out.labels->annotationOf(1, 3).reviewed);
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

    SECTION("the last plane and column the extent promises are sampled, not filled") {
        // 63 * 0.3 / 0.1 is 189 to within rounding, so 190 planes; the last
        // one's centre, 189 * (0.1 / 0.3), rounds past plane 63. Along x the
        // step is added once per column, and 63 additions of 0.2 pass 63 too.
        const Dims5 d{1, 1, 64, 2, 64};
        const DatasetMeta m = metaFor(d, 0.5, 0.3);
        const auto ones = std::make_shared<Array5>(d);
        std::fill(ones->data(), ones->data() + ones->numel(), 1.0f);
        ParamSet q = op.defaults();
        q.set("voxel_z", 0.1);
        q.set("voxel_x", 0.1);
        for (const char* interp : {"linear", "cubic", "nearest"}) {
            INFO(interp);
            q.set("interpolation", std::string(interp));
            const StepOutput s = op.run(inputOf(ones, m), q, prog.ctx);
            REQUIRE(s.array);
            const Dims5& o = s.array->dims();
            REQUIRE(o.z == 190);
            REQUIRE(o.x == 316);
            CHECK(s.array->at(0, 0, o.z - 1, 0, 0) == 1.0f);
            CHECK(s.array->at(0, 0, 0, 0, o.x - 1) == 1.0f);
            CHECK(s.array->at(0, 0, o.z - 1, 1, o.x - 1) == 1.0f);
        }
    }
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
    // nominal positions are 2 px short of the true offset 40, so a stitch that
    // never moves the tiles is 94 wide and reports Δx = 0. The corrected mosaic
    // is 96 wide and the pair table's Δx (measured minus nominal) is about +2.
    CHECK(r.array->dims().x >= 95);
    CHECK(r.array->dims().x <= 97);
    CHECK(r.diagnostics.kind == DiagnosticsKind::Alignment);
    REQUIRE(r.diagnostics.alignment);
    CHECK(r.diagnostics.alignment->tileNames.size() == 2);
    CHECK(r.diagnostics.alignment->gridCols == 2);
    REQUIRE(r.diagnostics.table);
    CHECK(r.diagnostics.table->rows.size() == 1);   // one accepted pair
    REQUIRE(r.diagnostics.table->rows[0].size() >= 5);
    CHECK_THAT(std::stod(r.diagnostics.table->rows[0][4]), WithinAbs(2.0, 1.0));
}

// --- segmentation ---------------------------------------------------------------

TEST_CASE("A plain classical cut labels blobs and Label cleanup drops the small ones", "[app][ops][classic][cleanup]") {
    const Dims5 dims{1, 1, 9, 40, 20};
    const DatasetMeta meta = metaFor(dims);
    auto data = blobArray(dims, 3, 3.0);
    data->at(0, 0, 4, 1, 1) = 1000.0f;   // a one-voxel speck
    // the classic step reduced to one global cut: no blur, opening or hole
    // fill, connected components (an opening would erase the speck)
    const Operation& op = requireOperation("classic");
    ParamSet p = op.defaults();
    p.set("sigma", 0.0);
    p.set("opening", std::int64_t{0});
    p.set("fill_holes", false);
    p.set("post", std::string("Connected components"));
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

TEST_CASE("Classical labels have an unknown confidence in every frame", "[app][ops][classic]") {
    const Dims5 dims{1, 2, 5, 40, 20};
    const DatasetMeta meta = metaFor(dims);
    auto data = blobArray(Dims5{1, 1, 5, 40, 20}, 3, 3.0);
    auto two = std::make_shared<Array5>(Array5::zeros(dims));
    for (Index t = 0; t < 2; ++t)
        for (Index z = 0; z < dims.z; ++z)
            for (Index y = 0; y < dims.y; ++y)
                for (Index x = 0; x < dims.x; ++x) two->at(0, t, z, y, x) = data->at(0, 0, z, y, x) * (t == 0 ? 0.5f : 1.0f);
    Progress prog;
    for (const double expand : {0.0, 2.0}) {
        INFO("expand " << expand);
        const Operation& op = requireOperation("classic");
        ParamSet p = op.defaults();
        p.set("method", std::string("Manual"));
        p.set("value", 100.0);
        p.set("min_voxels", std::int64_t{0});
        p.set("sigma", 0.0);
        p.set("opening", std::int64_t{0});
        p.set("expand", expand);   // 2 grows into voxels the mask called background
        const StepOutput r = op.run(inputOf(two, meta), p, prog.ctx);
        REQUIRE(r.labels);
        r.labels->recomputeStats(0);   // the viewer on the first frame: the intensities were never probabilities there either
        REQUIRE_FALSE(r.labels->stats().empty());
        for (const LabelStats& s : r.labels->stats()) CHECK(s.confidence == 1.0);
        CHECK(r.labels->flaggedCount("low conf") == 0);
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
    // The "device" of the last run request fakeWorker received.
    std::mutex fakeWorkerMutex;
    std::string fakeWorkerDevice;

    // A worker that answers hello / model_info / run(torch_segment) with a
    // probability map thresholding the input at 500 (plus a flat boundary
    // channel), using the public framing API.
    void fakeWorker(std::unique_ptr<rpc::Transport> transport) {
        std::vector<std::byte> inbox;
        rpc::HandshakeResponder handshake;   // the token-less handshake of a local worker
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
            if (method == "hello" || method == "auth") {
                std::string error;
                const nlohmann::json caps = {{"version", "test"}, {"methods", {"run:torch_segment", "model_info"}}, {"cuda", false}, {"device", "cpu · fake"}, {"hostname", "fake"}, {"python", "3"}};
                if (auto r = handshake.answer(method, h.value("params", nlohmann::json::object()), caps, error)) reply["result"] = *r;
                else reply = {{"id", h.value("id", 0)}, {"type", "error"}, {"message", error}};
                transport->send(rpc::encodeFrame(reply, {}));
            } else if (method == "model_info") {
                reply["result"] = {{"format", "TorchScript"}, {"input_shape", {1, 1, "Z", "Y", "X"}}, {"input_dtype", "float32"}, {"output_shape", {1, 2, "Z", "Y", "X"}}, {"size_bytes", 41 * 1024 * 1024}};
                transport->send(rpc::encodeFrame(reply, {}));
            } else if (method == "run") {
                REQUIRE(msg->tensors.size() == 1);
                {
                    const std::lock_guard<std::mutex> lock(fakeWorkerMutex);
                    fakeWorkerDevice = h.at("params").at("params").value("device", std::string());
                }
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
    SECTION("the request names the GPU the run was given") {
        // "auto" was the device the worker process started on and kept for
        // the session: a GPU chosen in Preferences since never reached it
        prog.ctx.backend = Backend::Cuda;
        prog.ctx.device = Device::cuda(1);
        (void)op.run(inputOf(data, meta), p, prog.ctx);
        const std::lock_guard<std::mutex> lock(fakeWorkerMutex);
        CHECK(fakeWorkerDevice == "cuda:1");
    }
    SECTION("on HPC the request names the session's GPU / CPU choice, run by run") {
        // the switch reaches the worker job with the next step: no new job
        prog.ctx.backend = Backend::Hpc;
        prog.ctx.hpcDevice = HpcDevice::Cpu;
        (void)op.run(inputOf(data, meta), p, prog.ctx);
        {
            const std::lock_guard<std::mutex> lock(fakeWorkerMutex);
            CHECK(fakeWorkerDevice == "cpu");
        }
        prog.ctx.hpcDevice = HpcDevice::Gpu;
        (void)op.run(inputOf(data, meta), p, prog.ctx);
        const std::lock_guard<std::mutex> lock(fakeWorkerMutex);
        CHECK(fakeWorkerDevice == "cuda");
    }
    remote->close();
    worker.join();
}

TEST_CASE("A request to the Python worker names the device of the run", "[app][ops][seg]") {
    StepContext ctx;
    ctx.backend = Backend::Cpu;
    CHECK(workerDevice(ctx) == "cpu");
    ctx.backend = Backend::Cuda;
    ctx.device = Device::cuda(1);
    CHECK(workerDevice(ctx) == "cuda:1");
    ctx.device = Device::cuda(-1);   // every GPU: the launcher's "cuda" too
    CHECK(workerDevice(ctx) == "cuda");
    ctx.backend = Backend::Hpc;   // the session's choice, not the job's own
    CHECK(ctx.hpcDevice == HpcDevice::Gpu);
    CHECK(workerDevice(ctx) == "cuda");
    ctx.hpcDevice = HpcDevice::Cpu;
    CHECK(workerDevice(ctx) == "cpu");
}

TEST_CASE("The HPC device is named in words", "[app][ops][hpc]") {
    CHECK(std::string(toString(HpcDevice::Gpu)) == "GPU");
    CHECK(std::string(toString(HpcDevice::Cpu)) == "CPU");
    CHECK(hpcDeviceFromString("gpu") == HpcDevice::Gpu);
    CHECK(hpcDeviceFromString("CUDA") == HpcDevice::Gpu);
    CHECK(hpcDeviceFromString("Cpu") == HpcDevice::Cpu);
    CHECK_FALSE(hpcDeviceFromString("tpu").has_value());
}

namespace {
    // A worker standing in for a model family (Cellpose / micro-SAM): run
    // replies with instance labels -- voxels above 500 get id 1 in the lower
    // half of z and id 2 above -- plus, optionally, a one-channel
    // probability map; model_info reports the family and its availability.
    void fakeLabelWorker(std::unique_ptr<rpc::Transport> transport, bool withProb) {
        std::vector<std::byte> inbox;
        rpc::HandshakeResponder handshake;   // the token-less handshake of a local worker
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
            if (method == "hello" || method == "auth") {
                std::string error;
                const nlohmann::json caps = {{"version", "test"}, {"methods", {"run:torch_segment", "model_info", "hub_search"}}, {"cuda", false}, {"device", "cpu · fake"}, {"hostname", "fake"}, {"python", "3"}};
                if (auto r = handshake.answer(method, h.value("params", nlohmann::json::object()), caps, error)) reply["result"] = *r;
                else reply = {{"id", h.value("id", 0)}, {"type", "error"}, {"message", error}};
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

// --- prompting --------------------------------------------------------------------

TEST_CASE("Prompts are a parameter that keeps its structure, every prompt in an object", "[app][ops][prompt]") {
    using nlohmann::json;
    ParamSpec spec = promptsParam(kPromptsKey, "Prompts");
    // what a person or an agent gives, in the forms it may come in, and
    // without object ids, as a list written before objects existed
    const json given = json::parse(R"([
        {"x": 12, "y": 40.0, "z": 3},
        {"kind": "point", "x": 1.5, "y": 2, "z": 0, "t": 2, "label": 0},
        {"x": 7, "y": 8, "z": 9, "label": "background"},
        {"kind": "box", "x0": 2, "y0": 3, "z0": 0, "x1": 10, "y1": 12.0, "z1": 4, "t": 1},
        {"kind": "scribble", "points": [[4, 4, 1], [5, 4, 1], [6, 5, 1]]}
    ])");
    const ParamValue v = coerceToSpec(spec, given);
    REQUIRE(std::holds_alternative<ParamJson>(v));
    const json stored = toJson(v);
    REQUIRE(stored.is_array());
    REQUIRE(stored.size() == 5);
    // canonical: every key present, whole voxels as integers, and an object
    // each: an object prompt starts one (1, 2, 3 in list order), a background
    // prompt joins the nearest object on its time point (the scribble, 3), or,
    // on a time point without one, an object of its own (4)
    CHECK(stored[0] == json::parse(R"({"kind": "point", "x": 12, "y": 40, "z": 3, "t": 0, "label": 1, "object": 1})"));
    CHECK(stored[0]["y"].is_number_integer());
    CHECK(stored[1]["x"] == 1.5);
    CHECK(stored[1]["t"] == 2);
    CHECK(stored[1]["label"] == 0);
    CHECK(stored[1]["object"] == 4);
    CHECK(stored[2]["label"] == 0);
    CHECK(stored[2]["object"] == 3);
    CHECK(stored[3] == json::parse(R"({"kind": "box", "x0": 2, "y0": 3, "z0": 0, "x1": 10, "y1": 12, "z1": 4, "t": 1, "object": 2})"));
    CHECK(stored[4] == json::parse(R"({"kind": "scribble", "points": [[4, 4, 1], [5, 4, 1], [6, 5, 1]], "t": 0, "label": 1, "object": 3})"));
    CHECK(toDisplayString(v) == "4 objects: 1 box, 3 points, 1 scribble");
    // its JSON text, the list of texts an older reader made of it, and the
    // canonical list itself are the same value
    CHECK(coerceToSpec(spec, given.dump()) == v);
    json texts = json::array();
    for (const json& e : given) texts.push_back(e.dump());
    CHECK(coerceToSpec(spec, texts) == v);
    CHECK(coerceToSpec(spec, stored) == v);
    // a value read back from JSON stays structured
    CHECK(paramValueFromJson(stored) == v);

    ParamSet p;
    p.set(kPromptsKey, v);
    const std::vector<Prompt> prompts = promptsOf(p);
    REQUIRE(prompts.size() == 5);
    CHECK(prompts[0] == Prompt::point(12, 40, 3, 0, true, 1));
    CHECK(prompts[1] == Prompt::point(1.5, 2, 0, 2, false, 4));
    CHECK(prompts[2] == Prompt::point(7, 8, 9, 0, false, 3));
    CHECK(prompts[3] == Prompt::boxOf({2, 3, 0, 10, 12, 4}, 1, 2));
    CHECK(prompts[4] == Prompt::scribble({{4, 4, 1}, {5, 4, 1}, {6, 5, 1}}, 0, true, 3));
    CHECK(promptsValue(prompts) == v);
    CHECK_FALSE(isPromptStep(p));   // no task
    p.set("task", std::string(kPromptTask));
    CHECK(isPromptStep(p));
    CHECK(promptsOf(ParamSet{}).empty());
    CHECK(toDisplayString(promptsValue({})) == "none");
    // given ids are kept; a new object is numbered after the highest, and a
    // background prompt joins its time point's nearest object
    {
        const json ids = toJson(coerceToSpec(spec, json::parse(R"([{"x": 1, "y": 1, "z": 1, "object": 7}, {"x": 30, "y": 2, "z": 1},
                                                                        {"x": 2, "y": 2, "z": 1, "label": 0}, {"x": 29, "y": 2, "z": 1, "label": 0}])")));
        CHECK(ids[0]["object"] == 7);
        CHECK(ids[1]["object"] == 8);
        CHECK(ids[2]["object"] == 7);
        CHECK(ids[3]["object"] == 8);
        CHECK(promptsValue({Prompt::point(1, 1, 1)}) == coerceToSpec(spec, json::parse(R"([{"x": 1, "y": 1, "z": 1, "object": 1}])")));
    }

    // what cannot be a prompt says which entry and why
    const auto refused = [&](const char* text) {
        try {
            (void)coerceToSpec(spec, json::parse(text));
        } catch (const std::invalid_argument& e) {
            return std::string(e.what());
        }
        return std::string("accepted");
    };
    CHECK(refused(R"([{"x": 1, "y": 2}])").find("prompt 1 (a point) needs a number 'z'") != std::string::npos);
    CHECK(refused(R"([{"x": 1, "y": 2, "z": -1}])").find(">= 0") != std::string::npos);
    CHECK(refused(R"([{"x": 1, "y": 2, "z": 1, "label": 3}])").find("'label'") != std::string::npos);
    CHECK(refused(R"([{"x": 1, "y": 2, "z": 1, "t": 0.5}])").find("'t'") != std::string::npos);
    CHECK(refused(R"([{"kind": "box", "x0": 4, "y0": 0, "z0": 0, "x1": 4, "y1": 2, "z1": 2}])").find("0 <= x0 < x1") != std::string::npos);
    CHECK(refused(R"([{"kind": "box", "x0": 0, "y0": 0, "z0": 0, "x1": 4, "y1": 2, "z1": 2, "label": 0}])").find("always names an object") !=
          std::string::npos);
    CHECK(refused(R"([{"kind": "scribble", "points": []}])").find("needs 'points'") != std::string::npos);
    CHECK(refused(R"([{"kind": "scribble", "points": [[1, 2]]}])").find("not [x, y, z]") != std::string::npos);
    CHECK(refused(R"([{"kind": "lasso", "x": 1, "y": 2, "z": 1}])").find("the kinds are point, box and scribble") != std::string::npos);
    CHECK(refused(R"([{"x": 1, "y": 2, "z": 1}, {"x": 1, "y": 2, "z": 1, "object": 0}])").find("prompt 2: 'object' is the id") != std::string::npos);
    CHECK(refused(R"([{"x": 1, "y": 2, "z": 1, "object": 1.5}])").find("'object'") != std::string::npos);
    CHECK(refused(R"([{"x": 1, "y": 2, "z": 1, "object": "a"}])").find("'object'") != std::string::npos);
    CHECK_THROWS(coerceToSpec(spec, 3));
    // the schema an agent reads
    const json schema = schemaOf(spec);
    CHECK(schema["type"] == "array");
    CHECK(schema["items"]["properties"]["kind"]["enum"] == json::array({"point", "box", "scribble"}));
    CHECK(schema["items"]["properties"]["object"]["type"] == "integer");
    CHECK(schema["items"]["properties"]["object"]["minimum"] == 1);
}

TEST_CASE("Each time point sends its objects, each with all of its prompts", "[app][ops][prompt]") {
    using nlohmann::json;
    const std::vector<Prompt> prompts = {Prompt::point(10, 11, 1, 0, true, 1),
                                         Prompt::point(20, 21, 2, 1, true, 2),
                                         Prompt::scribble({{1, 1, 1}, {2, 1, 1}, {3, 2, 1}}, 0, true, 3),
                                         Prompt::point(30, 31, 3, 0, false, 1),   // a correction of object 1
                                         Prompt::boxOf({0, 0, 0, 4, 4, 2}, 0, 4),
                                         Prompt::point(2, 2, 1, 0, true, 4),        // and a click that grows object 4
                                         Prompt::point(50, 51, 5, 2, false, 5)};   // background only: not sent
    const FramePrompt t0 = framePrompt(prompts, 0);
    CHECK(t0.ids() == std::vector<std::uint32_t>{1, 3, 4});
    REQUIRE(t0.objects.size() == 3);
    CHECK(t0.objects[0].placed == std::vector<std::size_t>{0, 3});
    CHECK(t0.objects[2].placed == std::vector<std::size_t>{4, 5});
    CHECK(t0.backgroundOnly.empty());
    // exactly what the worker is sent: one entry per object, in ascending id,
    // a key only where the object has one
    CHECK(promptObjectsJson(t0) == json::parse(R"([
        {"points": [[10, 11, 1], [30, 31, 3]], "point_labels": [1, 0]},
        {"scribbles": [{"points": [[1, 1, 1], [2, 1, 1], [3, 2, 1]], "label": 1}]},
        {"box": [0, 0, 0, 4, 4, 2], "points": [[2, 2, 1]], "point_labels": [1]}])"));
    const FramePrompt t1 = framePrompt(prompts, 1);
    CHECK(t1.ids() == std::vector<std::uint32_t>{2});
    CHECK(promptObjectsJson(t1) == json::parse(R"([{"points": [[20, 21, 2]], "point_labels": [1]}])"));
    const FramePrompt t2 = framePrompt(prompts, 2);
    CHECK(t2.empty());
    CHECK(t2.backgroundOnly == std::vector<std::uint32_t>{5});
    CHECK(framePrompt(prompts, 3).empty());
    CHECK(framePrompt(prompts, 3).backgroundOnly.empty());

    // the worker's numbering (i + 1 for the i-th object) becomes the objects' own ids
    std::vector<std::uint32_t> masks = {0, 1, 2, 3, 4};
    applyPromptIds(masks.data(), static_cast<Index>(masks.size()), t0);
    CHECK(masks == std::vector<std::uint32_t>{0, 1, 3, 4, 0});

    // a long stroke is sent as a few points along it, both ends included
    std::vector<std::array<double, 3>> stroke;
    for (int i = 0; i < 40; ++i) stroke.push_back({static_cast<double>(i), 5, 2});
    const std::vector<std::array<double, 3>> sent = scribbleSample(stroke);
    REQUIRE(sent.size() == kScribblePointsSent);
    CHECK(sent.front() == stroke.front());
    CHECK(sent.back() == stroke.back());
    CHECK(framePrompt({Prompt::scribble(stroke, 0, true, 1)}, 0).objects[0].scribbles[0].points == sent);

    const DatasetMeta meta = metaFor(Dims5{1, 3, 6, 60, 60});
    Validation v;
    validatePrompts(prompts, meta, v);
    CHECK(v.ok());
    REQUIRE(v.warnings.size() == 1);
    CHECK(v.warnings[0].find("Object 5 has only background prompts on time point 2") != std::string::npos);
    for (const Prompt& outside : {Prompt::point(60, 1, 1, 0, true, 1), Prompt::point(1, 1, 1, 3, true, 1), Prompt::boxOf({50, 50, 0, 61, 60, 6}, 0, 1),
                                  Prompt::scribble({{1, 1, 1}, {1, 1, 6}}, 0, true, 1)}) {
        Validation w;
        validatePrompts({outside}, meta, w);
        REQUIRE_FALSE(w.ok());
        CHECK(w.firstError().find("outside the image") != std::string::npos);
        CHECK(w.firstError().find("of object 1") != std::string::npos);
    }
    {
        // one object is one mask, and takes one box
        std::vector<Prompt> twice = prompts;
        twice.push_back(Prompt::boxOf({5, 5, 0, 9, 9, 2}, 0, 4));
        Validation w;
        validatePrompts(twice, meta, w);
        REQUIRE_FALSE(w.ok());
        CHECK(w.firstError().find("Object 4 has 2 boxes on time point 0") != std::string::npos);
    }
    Validation none;
    validatePrompts({}, meta, none);
    CHECK(none.ok());
    REQUIRE(none.warnings.size() == 1);
    CHECK(none.warnings[0].find("Prompt tool") != std::string::npos);
    CHECK(promptCounts(prompts) == "5 objects \xC2\xB7 1 box \xC2\xB7 3 object points \xC2\xB7 2 background points \xC2\xB7 1 scribble");
}

TEST_CASE("Which object a Prompt tool click belongs to", "[app][ops][prompt]") {
    using Action = PromptClick::Action;
    const std::vector<Prompt> prompts = {Prompt::boxOf({10, 10, 0, 20, 20, 4}, 0, 1), Prompt::point(40, 40, 2, 0, true, 2),
                                         Prompt::point(5, 5, 2, 1, true, 3), Prompt::point(50, 10, 2, 0, false, 4)};
    CHECK(nextPromptObject(prompts) == 5);
    const auto click = [&](std::array<double, 3> at, std::uint32_t under, bool positive, bool forceNew = false, Index t = 0) {
        return promptClickTarget(prompts, t, at, under, positive, forceNew);
    };
    // an object click inside an object's mask grows that object ...
    PromptClick c = click({15, 15, 2}, 1, true);
    CHECK(c.action == Action::AddTo);
    CHECK(c.object == 1);
    // ... and starts a new one with Shift, outside every mask, on the mask of
    // an object of another time point, or of one that is not sent
    for (const PromptClick& n : {click({15, 15, 2}, 1, true, true), click({30, 30, 2}, 0, true), click({5, 5, 2}, 3, true), click({50, 10, 2}, 4, true)}) {
        CHECK(n.action == Action::NewObject);
        CHECK(n.object == 5);
    }
    // a background click corrects the object whose mask is under it ...
    c = click({41, 41, 2}, 2, false);
    CHECK(c.action == Action::AddTo);
    CHECK(c.object == 2);
    // ... else the nearest object of the time point, by distance to its prompts
    c = click({35, 35, 2}, 0, false);
    CHECK(c.action == Action::AddTo);
    CHECK(c.object == 2);
    c = click({21, 15, 2}, 0, false);   // two voxels from the box's last column
    CHECK(c.action == Action::AddTo);
    CHECK(c.object == 1);
    // ... and, with no object on the time point, nothing, saying why
    c = click({5, 5, 2}, 0, false, false, 2);
    CHECK(c.action == Action::None);
    CHECK(c.why.find("place one first") != std::string::npos);
    // on a tie the lower id
    c = promptClickTarget({Prompt::point(0, 0, 0, 0, true, 2), Prompt::point(10, 0, 0, 0, true, 1)}, 0, {5, 0, 0}, 0, false, false);
    CHECK(c.object == 1);
    // a 2-D model counts only the objects on the plane clicked
    const std::vector<Prompt> planes = {Prompt::point(10, 10, 2, 0, true, 1), Prompt::point(30, 30, 5, 0, true, 2)};
    CHECK(promptClickTarget(planes, 0, {29, 29, 2}, 0, false, false, true).object == 1);
    CHECK(promptClickTarget(planes, 0, {29, 29, 2}, 0, false, false, false).object == 2);
    c = promptClickTarget(planes, 0, {29, 29, 7}, 0, false, false, true);
    CHECK(c.action == Action::None);
    CHECK(c.why.find("this plane") != std::string::npos);
    c = promptClickTarget(planes, 0, {30, 30, 2}, 2, true, false, true);
    CHECK(c.action == Action::NewObject);
    CHECK(c.object == 3);

    // removing: a prompt, and with an object's last object prompt its corrections
    const std::vector<Prompt> one = {Prompt::boxOf({10, 10, 0, 20, 20, 4}, 0, 1), Prompt::point(12, 12, 1, 0, false, 1),
                                     Prompt::point(15, 15, 1, 0, true, 1), Prompt::point(40, 40, 2, 0, true, 2)};
    CHECK(removePrompt(one, 2).size() == 3);
    CHECK(removePrompt(one, 1).size() == 3);
    const std::vector<Prompt> last = removePrompt(removePrompt(one, 2), 0);
    REQUIRE(last.size() == 1);
    CHECK(last[0].objectId == 2);
    CHECK(removePromptObject(one, 1) == std::vector<Prompt>{one[3]});
    CHECK(removePrompt(one, 9) == one);
    const std::vector<PromptObject> objects = promptObjects(one);
    REQUIRE(objects.size() == 2);
    CHECK(objects[0].id == 1);
    CHECK(objects[0].prompts == std::vector<std::size_t>{0, 1, 2});
    CHECK(promptObjectText(objects[0]) == "box + 1 point + 1 correction");
    CHECK(promptObjectText(objects[1]) == "1 point");
    CHECK(objects[1].sent());
    CHECK_FALSE(promptObjects({Prompt::point(1, 1, 1, 0, false, 1)})[0].sent());

    // the planes a 2-D model answers an object in, as its worker works them out
    FramePrompt::Object o;
    o.box = std::array<double, 6>{0, 0, 3, 4, 4, 8};
    CHECK(promptPlanes(o) == std::vector<Index>{5});
    o.box = std::array<double, 6>{0, 0, 3, 4, 4, 7};
    CHECK(promptPlanes(o) == std::vector<Index>{4});   // 4.5, rounded half to even as Python does
    o.points = {{1, 1, 2}};
    o.scribbles = {{{{1, 1, 3}, {2, 1, 3}}, 1}};
    CHECK(promptPlanes(o) == std::vector<Index>{2, 3});

    // the mask scores, written by the run and read back by the panel
    const FramePrompt f = framePrompt({Prompt::point(1, 1, 1, 0, true, 3), Prompt::point(5, 5, 1, 0, true, 7)}, 0);
    std::string fact;
    appendPromptScores(fact, f, nlohmann::json::array({0.81, 0.7}), 0, false);
    CHECK(fact == "#3 0.81, #7 0.70");
    Diagnostics d;
    d.facts.push_back({"Mask scores", fact});
    const auto scores = promptScores(d, 0);
    REQUIRE(scores.size() == 2);
    CHECK(scores[0].first == 3);
    CHECK_THAT(scores[0].second, WithinAbs(0.81, 1e-9));
    CHECK(scores[1].first == 7);
    std::string many;
    appendPromptScores(many, f, nlohmann::json::array({0.81, 0.7}), 0, true);
    appendPromptScores(many, framePrompt({Prompt::point(1, 1, 1, 2, true, 3)}, 2), nlohmann::json::array({0.5}), 2, true);
    CHECK(many == "t 0: #3 0.81, #7 0.70 \xC2\xB7 t 2: #3 0.50");
    d.facts.back().value = many;
    REQUIRE(promptScores(d, 2).size() == 1);
    CHECK_THAT(promptScores(d, 2)[0].second, WithinAbs(0.5, 1e-9));
    CHECK(promptScores(d, 1).empty());
    CHECK(promptScores(Diagnostics{}, 0).empty());
}

namespace {
    // What fakePromptWorker was asked: one entry per run, in order.
    struct PromptRequest {
        std::string kind;
        nlohmann::json params;
        std::vector<Index> shape;
    };
    std::mutex promptRequestsMutex;
    std::vector<PromptRequest> promptRequests;

    enum class FakePrompt { Promptable,
                            NotPromptable,
                            Refuses };

    // A worker that answers the joint `objects` form the way the real ones
    // do: one mask per object, numbered i + 1 in the order sent, later masks
    // winning, a score per mask (0.9, 0.8, ...). An object's mask is its box,
    // else a 3 x 3 square on the plane of its first object point (or object
    // scribble's first point); each of its background points then takes its
    // own voxel out of the mask, which is how a correction shows here.
    // model_info lists the tasks a bundle offers (NotPromptable: Segment
    // only) and, for a family spec, whether it is promptable. Refuses answers
    // every run with micro-SAM's refusal of an object across planes.
    void fakePromptWorker(std::unique_ptr<rpc::Transport> transport, FakePrompt mode) {
        const bool promptable = mode != FakePrompt::NotPromptable;
        std::vector<std::byte> inbox;
        rpc::HandshakeResponder handshake;   // the token-less handshake of a local worker
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
            if (method == "hello" || method == "auth") {
                std::string error;
                const nlohmann::json caps = {{"version", "test"}, {"methods", {"run:foundation", "run:torch_segment", "model_info"}}, {"cuda", false}, {"device", "cpu · fake"}, {"hostname", "fake"}, {"python", "3"}};
                if (auto r = handshake.answer(method, h.value("params", nlohmann::json::object()), caps, error)) reply["result"] = *r;
                else reply = {{"id", h.value("id", 0)}, {"type", "error"}, {"message", error}};
                transport->send(rpc::encodeFrame(reply, {}));
            } else if (method == "model_info") {
                const std::string spec = h.at("params").value("spec", std::string());
                if (spec.rfind("microsam:", 0) == 0)
                    reply["result"] = {{"format", "micro-sam"}, {"model", spec.substr(9)}, {"available", true}, {"promptable", promptable}};
                else
                    reply["result"] = {{"format", "latents-model"}, {"name", promptable ? "fake-sam" : "fake-conv"}, {"version", "v1"}, {"tasks", promptable ? nlohmann::json{"segment", "prompt"} : nlohmann::json{"segment"}}, {"promptable", promptable}, {"description", "a fake model"}, {"voxel_um", {0.1, 0.1, 0.5}}};
                transport->send(rpc::encodeFrame(reply, {}));
            } else if (method == "run") {
                REQUIRE(msg->tensors.size() == 1);
                const rpc::Tensor& in = msg->tensors.front();
                const std::string kind = h.at("params").value("kind", std::string());
                const nlohmann::json p = h.at("params").at("params");
                {
                    const std::lock_guard<std::mutex> lock(promptRequestsMutex);
                    promptRequests.push_back({kind, p, in.shape});
                }
                if (mode == FakePrompt::Refuses) {
                    reply["type"] = "error";
                    reply["message"] = "object 0 has prompts on planes [1, 3]; microsam:vit_b_lm is a 2-D model and cannot refine across z. "
                                       "Put the object's corrective prompts in the plane it was opened in, or prompt a model folder on the Foundation step, whose decoder is 3-D.";
                    transport->send(rpc::encodeFrame(reply, {}));
                    continue;
                }
                // (c, t, z, y, x) for the foundation model, (z, y, x) for a family model
                const std::size_t r = in.shape.size();
                const Index z = in.shape[r - 3], y = in.shape[r - 2], x = in.shape[r - 1];
                std::vector<std::uint32_t> labels(static_cast<std::size_t>(z * y * x), 0u);
                std::vector<float> confidence(labels.size(), 0.0f);
                nlohmann::json scores = nlohmann::json::array();
                std::uint32_t id = 0;
                const auto at = [&](Index zz, Index yy, Index xx) { return static_cast<std::size_t>((zz * y + yy) * x + xx); };
                const auto paint = [&](Index z0, Index z1, Index y0, Index y1, Index x0, Index x1) {
                    for (Index zz = std::max<Index>(z0, 0); zz < std::min(z1, z); ++zz)
                        for (Index yy = std::max<Index>(y0, 0); yy < std::min(y1, y); ++yy)
                            for (Index xx = std::max<Index>(x0, 0); xx < std::min(x1, x); ++xx) {
                                labels[at(zz, yy, xx)] = id;
                                confidence[at(zz, yy, xx)] = 0.9f;
                            }
                };
                const auto voxel = [](const nlohmann::json& q, int axis) { return static_cast<Index>(q[axis].get<double>()); };
                for (const nlohmann::json& o : p.value("objects", nlohmann::json::array())) {
                    ++id;
                    const nlohmann::json pts = o.value("points", nlohmann::json::array());
                    const nlohmann::json labs = o.value("point_labels", nlohmann::json::array());
                    if (o.contains("box")) {
                        const nlohmann::json& b = o["box"];
                        paint(voxel(b, 2), voxel(b, 5), voxel(b, 1), voxel(b, 4), voxel(b, 0), voxel(b, 3));
                    } else {
                        std::optional<nlohmann::json> first;
                        for (std::size_t k = 0; k < pts.size() && !first; ++k)
                            if (labs[k] == 1) first = pts[k];
                        for (const nlohmann::json& s : o.value("scribbles", nlohmann::json::array()))
                            if (!first && s.value("label", 1) == 1) first = s.at("points").at(0);
                        if (first) paint(voxel(*first, 2), voxel(*first, 2) + 1, voxel(*first, 1) - 1, voxel(*first, 1) + 2, voxel(*first, 0) - 1, voxel(*first, 0) + 2);
                    }
                    for (std::size_t k = 0; k < pts.size(); ++k)
                        if (labs[k] == 0 && labels[at(voxel(pts[k], 2), voxel(pts[k], 1), voxel(pts[k], 0))] == id)
                            labels[at(voxel(pts[k], 2), voxel(pts[k], 1), voxel(pts[k], 0))] = 0;
                    scores.push_back(1.0 - 0.1 * static_cast<double>(id));
                }
                rpc::TensorRef out;
                out.name = "labels";
                out.dtype = "uint32";
                out.data = labels.data();
                out.nbytes = labels.size() * sizeof(std::uint32_t);
                std::vector<rpc::TensorRef> tensors;
                if (kind == "foundation") {
                    out.shape = {1, z, y, x};
                    rpc::TensorRef conf;
                    conf.name = "confidence";
                    conf.dtype = "float32";
                    conf.shape = {1, z, y, x};
                    conf.data = confidence.data();
                    conf.nbytes = confidence.size() * sizeof(float);
                    tensors = {out, conf};
                    reply["result"] = {{"prompts", id}, {"mask_scores", scores}, {"objects", id}};
                } else {
                    out.shape = {z, y, x};
                    tensors = {out};
                    reply["result"] = {{"task", "prompt"}, {"mask_scores", scores}, {"plane_only", true}};
                }
                transport->send(rpc::encodeFrame(reply, tensors));
            } else if (method == "cancel") {
                // nothing running
            } else {
                reply["type"] = "error";
                reply["message"] = "unknown method " + method;
                transport->send(rpc::encodeFrame(reply, {}));
            }
        }
    }

    // One run of `op` against a fresh fake worker; the error it ended with, if any.
    StepOutput runWithFakeWorker(const Operation& op, const StepInput& input, const ParamSet& p, FakePrompt mode, std::string* error = nullptr) {
        auto pair = rpc::loopbackPair();
        std::thread worker(fakePromptWorker, std::move(pair.second), mode);
        auto remote = std::make_unique<RemoteWorker>(std::move(pair.first));
        Progress prog;
        prog.ctx.remote = remote.get();
        StepOutput out;
        try {
            out = op.run(input, p, prog.ctx);
        } catch (const std::exception& e) {
            if (!error) {
                remote->close();
                worker.join();
                throw;
            }
            *error = e.what();
        }
        remote->close();
        worker.join();
        return out;
    }

    void clearPromptRequests() {
        const std::lock_guard<std::mutex> lock(promptRequestsMutex);
        promptRequests.clear();
    }

    std::string factOf(const Diagnostics& d, const std::string& key) {
        for (const DiagnosticFact& f : d.facts)
            if (f.key == key) return f.value;
        return std::string();
    }
} // namespace

TEST_CASE("The foundation step's Prompt task sends each frame its objects", "[app][ops][prompt][rpc]") {
    using nlohmann::json;
    const Dims5 dims{1, 3, 4, 20, 24};
    const DatasetMeta meta = metaFor(dims);
    auto data = rampArray(dims);
    const Operation& op = requireOperation("foundation");
    ParamSet p = op.defaults();
    p.set("model", std::string("models/fake-sam/v1"));   // the worker's, not on this machine: a warning
    p.set("task", std::string(kPromptTask));
    REQUIRE(isPromptStep(p));
    // t 0: object 1 a click and a correction beside it, object 2 a box,
    // object 3 a click, object 4 only a background click; t 1: nothing;
    // t 2: object 5 a click, object 6 a scribble
    std::vector<Prompt> prompts = {Prompt::point(5, 6, 1, 0, true, 1), Prompt::point(6, 6, 1, 0, false, 1),
                                   Prompt::boxOf({2, 14, 0, 4, 17, 2}, 0, 2), Prompt::point(10, 12, 2, 0, true, 3),
                                   Prompt::point(15, 6, 1, 0, false, 4), Prompt::point(20, 16, 3, 2, true, 5),
                                   Prompt::scribble({{8, 8, 1}, {9, 8, 1}, {10, 8, 1}}, 2, true, 6)};
    p.set(kPromptsKey, promptsValue(prompts));
    const Validation v = op.validate(p, meta);
    CHECK(v.ok());
    CHECK(std::any_of(v.warnings.begin(), v.warnings.end(), [](const std::string& w) { return w.find("Object 4 has only background") != std::string::npos; }));
    CHECK(op.summary(p, meta).find("prompt") != std::string::npos);
    CHECK(op.summary(p, meta).find("6 objects: 1 box, 5 points, 1 scribble") != std::string::npos);

    clearPromptRequests();
    const StepOutput r = runWithFakeWorker(op, inputOf(data, meta), p, FakePrompt::Promptable);

    // two calls, for t 0 and t 2; t 1 asked nothing of the worker, object 4 was not sent
    REQUIRE(promptRequests.size() == 2);
    for (const PromptRequest& q : promptRequests) {
        CHECK(q.kind == "foundation");
        CHECK(q.params["task"] == "prompt");
        CHECK(q.shape == std::vector<Index>{1, 1, 4, 20, 24});   // one channel, one time point
        for (const char* flat : {"points", "point_labels", "boxes", "scribbles"}) CHECK_FALSE(q.params.contains(flat));
    }
    CHECK(promptRequests[0].params["objects"] == json::parse(R"([
        {"points": [[5, 6, 1], [6, 6, 1]], "point_labels": [1, 0]},
        {"box": [2, 14, 0, 4, 17, 2]},
        {"points": [[10, 12, 2]], "point_labels": [1]}])"));
    CHECK(promptRequests[1].params["objects"] == json::parse(R"([
        {"points": [[20, 16, 3]], "point_labels": [1]},
        {"scribbles": [{"points": [[8, 8, 1], [9, 8, 1], [10, 8, 1]], "label": 1}]}])"));

    REQUIRE(r.labels);
    const auto at = [&](const StepOutput& o, Index t, Index z, Index y, Index x) { return o.labels->volume(t)[(z * dims.y + y) * dims.x + x]; };
    // every mask carries its object's id
    CHECK(at(r, 0, 1, 6, 5) == 1);
    CHECK(at(r, 0, 1, 6, 6) == 0);    // the correction took its voxel out of object 1
    CHECK(at(r, 0, 1, 15, 3) == 2);   // the box
    CHECK(at(r, 0, 2, 12, 10) == 3);
    CHECK(at(r, 0, 1, 6, 15) == 0);   // background names no object
    CHECK(at(r, 2, 3, 16, 20) == 5);
    CHECK(at(r, 2, 1, 8, 8) == 6);    // the scribble
    const std::uint32_t* t1 = r.labels->volume(1);
    CHECK(std::all_of(t1, t1 + dims.z * dims.planeSize(), [](std::uint32_t id) { return id == 0; }));
    // the model's score of each object's mask, by the object's id
    CHECK(factOf(r.diagnostics, "Mask scores") == "t 0: #1 0.90, #2 0.80, #3 0.70 \xC2\xB7 t 2: #5 0.90, #6 0.80");
    CHECK(factOf(r.diagnostics, "Prompts") ==
          "6 objects \xC2\xB7 1 box \xC2\xB7 3 object points \xC2\xB7 2 background points \xC2\xB7 1 scribble \xC2\xB7 on 2 of 3 time points");
    REQUIRE(promptScores(r.diagnostics, 2).size() == 2);
    CHECK(promptScores(r.diagnostics, 2)[1].first == 6);

    SECTION("a correction refines its object, and every object keeps its label across re-runs") {
        // object 1 removed, object 3 corrected beside its click: the worker now
        // numbers the box 1 and object 3 2, and they come back as 2 and 3
        std::vector<Prompt> next = removePromptObject(prompts, 1);
        next.push_back(Prompt::point(11, 12, 2, 0, false, 3));
        ParamSet q = p;
        q.set(kPromptsKey, promptsValue(next));
        clearPromptRequests();
        const StepOutput again = runWithFakeWorker(op, inputOf(data, meta), q, FakePrompt::Promptable);
        REQUIRE(promptRequests.size() == 2);
        CHECK(promptRequests[0].params["objects"] == json::parse(R"([
            {"box": [2, 14, 0, 4, 17, 2]},
            {"points": [[10, 12, 2], [11, 12, 2]], "point_labels": [1, 0]}])"));
        CHECK(at(again, 0, 1, 15, 3) == 2);
        CHECK(at(again, 0, 2, 12, 9) == 3);
        CHECK(at(again, 0, 2, 12, 11) == 0);   // corrected away
        CHECK(at(again, 0, 1, 6, 5) == 0);     // object 1 is gone
        CHECK(at(again, 2, 3, 16, 20) == 5);
        CHECK(factOf(again.diagnostics, "Mask scores") == "t 0: #2 0.90, #3 0.80 \xC2\xB7 t 2: #5 0.90, #6 0.80");
    }
    SECTION("without any prompt nothing is asked of the worker, and the labels are empty") {
        ParamSet none = p;
        none.set(kPromptsKey, promptsValue({}));
        CHECK(op.validate(none, meta).ok());   // a warning, not an error
        Progress noWorker;                     // no worker at all
        const StepOutput e = op.run(inputOf(data, meta), none, noWorker.ctx);
        REQUIRE(e.labels);
        CHECK(e.labels->stats().empty());
        CHECK(e.note.find("no prompts") != std::string::npos);
    }
    SECTION("only background prompts: nothing is sent") {
        ParamSet bg = p;
        bg.set(kPromptsKey, promptsValue({Prompt::point(15, 6, 1, 0, false, 4)}));
        CHECK(op.validate(bg, meta).ok());
        Progress noWorker;
        const StepOutput e = op.run(inputOf(data, meta), bg, noWorker.ctx);
        REQUIRE(e.labels);
        CHECK(e.labels->stats().empty());
    }
    SECTION("a prompt outside the image is refused before anything runs") {
        ParamSet out = p;
        out.set(kPromptsKey, promptsValue({Prompt::point(24, 0, 0)}));
        CHECK_FALSE(op.validate(out, meta).ok());
        out.set(kPromptsKey, promptsValue({Prompt::boxOf({0, 0, 0, 4, 4, 5})}));
        CHECK_FALSE(op.validate(out, meta).ok());
        out.set(kPromptsKey, promptsValue({Prompt::point(0, 0, 0, 3)}));
        CHECK_FALSE(op.validate(out, meta).ok());
    }
}

TEST_CASE("A model without a prompt decoder is refused by name", "[app][ops][prompt][rpc]") {
    const Dims5 dims{1, 1, 4, 20, 24};
    const DatasetMeta meta = metaFor(dims);
    auto data = rampArray(dims);
    const Operation& op = requireOperation("foundation");
    ParamSet p = op.defaults();
    p.set("model", std::string("models/fake-conv/v1"));
    p.set("task", std::string(kPromptTask));
    p.set(kPromptsKey, promptsValue({Prompt::point(5, 6, 1)}));
    clearPromptRequests();
    std::string error;
    (void)runWithFakeWorker(op, inputOf(data, meta), p, FakePrompt::NotPromptable, &error);
    CHECK(error.find("cannot be prompted") != std::string::npos);
    CHECK(error.find("fake-conv v1 cannot be prompted: it has no prompt decoder") != std::string::npos);
    CHECK(error.find("it offers segment") != std::string::npos);
    CHECK(promptRequests.empty());   // refused before any frame went out
}

namespace {
    // A models folder of model.json files (what the application reads of a
    // model folder; the worker's tests run whole fake models).
    struct ModelsFolder {
        std::filesystem::path root = test::uniqueTempPath("models", "");
        ModelsFolder() { std::filesystem::create_directories(root); }
        ~ModelsFolder() {
            std::error_code ec;
            std::filesystem::remove_all(root, ec);
        }
        ModelsFolder(const ModelsFolder&) = delete;
        ModelsFolder& operator=(const ModelsFolder&) = delete;

        std::string add(const std::string& name, const std::string& version, std::vector<std::string> tasks, int channels = 1,
                        const std::string& notes = "") {
            const std::filesystem::path folder = root / name / version;
            std::filesystem::create_directories(folder);
            const nlohmann::json man = {{"format", "latents-model/1"}, {"name", name}, {"version", version}, {"tasks", tasks}, {"input", {{"channels", channels}, {"channel_merge", ""}, {"crop", {32, 192, 192}}, {"voxel_um", {0.3, 0.1, 0.1}}}}, {"decode", {{"fg_threshold", 0.5}, {"min_voxels", 0}}}, {"notes", notes}};
            std::ofstream(folder / "model.json") << man.dump(2);
            std::ofstream(folder / "README.md") << "# " << name << "\n\nMembrane-stained cells, from the README.\n";
            std::ofstream(folder / "weights.safetensors", std::ios::binary) << std::string(100, '\0');
            return folder.string();
        }
    };
} // namespace

TEST_CASE("The Foundation step takes a model folder and offers only what model.json lists", "[app][ops][foundation]") {
    const Dims5 dims{2, 1, 4, 20, 24};
    const DatasetMeta meta = metaFor(dims);   // 0.1 x 0.1 x 0.3 um
    const Operation& op = requireOperation("foundation");
    const ParamSpec* model = nullptr;
    const ParamSpec* task = nullptr;
    for (const ParamSpec& s : op.info().params) {
        if (s.key == "model") model = &s;
        if (s.key == "task") task = &s;
    }
    REQUIRE(model);
    CHECK(model->type == ParamType::Path);
    CHECK(model->directory);   // Browse picks a folder, on this computer or on the cluster
    REQUIRE(task);
    CHECK(task->choices == std::vector<std::string>{"Segment objects", kPromptTask});

    ModelsFolder models;
    const std::string sam = models.add("coat-sam-s2", "v1", {"segment", "prompt"}, 1, "Promptable cells.");
    const std::string conv = models.add("coat-conv-r0", "v1", {"segment"});
    ParamSet p = op.defaults();
    p.set("model", sam);
    CHECK(op.validate(p, meta).ok());
    CHECK(op.summary(p, meta).find("coat-sam-s2 v1") != std::string::npos);
    p.set("task", std::string(kPromptTask));
    p.set(kPromptsKey, promptsValue({Prompt::point(5, 6, 1)}));
    CHECK(op.validate(p, meta).ok());
    // model.json itself names its folder too
    p.set("model", (std::filesystem::path(sam) / "model.json").string());
    CHECK(op.validate(p, meta).ok());
    p.set("model", sam);

    SECTION("a model without a prompt decoder offers Segment only") {
        p.set("model", conv);
        const Validation v = op.validate(p, meta);
        REQUIRE_FALSE(v.ok());
        CHECK(v.firstError().find("coat-conv-r0 v1 cannot be prompted") != std::string::npos);
        CHECK(v.firstError().find("it offers segment") != std::string::npos);
        p.set("task", std::string("Segment objects"));
        CHECK(op.validate(p, meta).ok());
    }
    SECTION("the channels the model takes") {
        p.set("channels", std::string("All channels"));   // two channels to a one-channel model
        Validation v = op.validate(p, meta);
        REQUIRE_FALSE(v.ok());
        CHECK(v.firstError().find("takes one channel") != std::string::npos);
        p.set("model", models.add("two", "v1", {"segment"}, 2));
        p.set("task", std::string("Segment objects"));
        CHECK(op.validate(p, meta).ok());
        p.set("channels", std::string("Selected channel"));
        v = op.validate(p, meta);
        REQUIRE_FALSE(v.ok());
        CHECK(v.firstError().find("takes 2 channels") != std::string::npos);
    }
    SECTION("a voxel size far from the model's is a warning") {
        const Validation near = op.validate(p, meta);
        CHECK(std::none_of(near.warnings.begin(), near.warnings.end(), [](const std::string& w) { return w.find("trained at") != std::string::npos; }));
        const Validation far = op.validate(p, metaFor(dims, 0.4, 0.3));
        CHECK(far.ok());
        CHECK(std::any_of(far.warnings.begin(), far.warnings.end(), [](const std::string& w) {
            return w.find("coat-sam-s2 was trained at 0.3 x 0.1 x 0.1 um") != std::string::npos && w.find("far on x, y") != std::string::npos;
        }));
    }
    SECTION("what is not a model folder says what to do") {
        p.set("model", std::string("C:/models/cells.ltb"));
        Validation v = op.validate(p, meta);
        REQUIRE_FALSE(v.ok());
        CHECK(v.firstError().find("Old bundle format") != std::string::npos);
        CHECK(v.firstError().find("scripts/export_model.py") != std::string::npos);
        p.set("model", models.root.string());   // the models folder, not one model
        v = op.validate(p, meta);
        REQUIRE_FALSE(v.ok());
        CHECK(v.firstError().find("no model.json") != std::string::npos);
        std::ofstream(std::filesystem::path(conv) / "model.json") << "{ not json";
        p.set("model", conv);
        v = op.validate(p, meta);
        REQUIRE_FALSE(v.ok());
        CHECK(v.firstError().find("not valid JSON") != std::string::npos);
        p.set("model", std::string());
        CHECK(op.validate(p, meta).firstError().find("Choose a model folder") != std::string::npos);
        // on the cluster: the engine checks it there, with the node's path
        p.set("model", std::string("cluster://hpc/clusterfs/models/coat-sam-s2/v1"));
        CHECK(op.validate(p, meta).ok());
    }
}

TEST_CASE("The Task choice is what the worker's model_info lists", "[app][ops][foundation][rpc]") {
    for (const bool promptable : {true, false}) {
        auto pair = rpc::loopbackPair();
        std::thread worker(fakePromptWorker, std::move(pair.second), promptable ? FakePrompt::Promptable : FakePrompt::NotPromptable);
        RemoteWorker remote(std::move(pair.first));
        const std::string spec = promptable ? "/clusterfs/models/fake-sam/v1" : "/clusterfs/models/fake-conv/v1";
        const nlohmann::json info = remote.call("model_info", {{"path", spec}, {"model", spec}, {"spec", spec}}).result;
        remote.close();
        worker.join();
        std::string error;
        const std::optional<ModelFolderFacts> facts = modelFactsFromJson(info, &error);
        REQUIRE(facts);
        CHECK(facts->name == (promptable ? "fake-sam" : "fake-conv"));
        CHECK(facts->promptable == promptable);
        CHECK(facts->voxelUm == std::vector<double>{0.1, 0.1, 0.5});
        CHECK(modelTaskChoices(facts) ==
              (promptable ? std::vector<std::string>{"Segment objects", kPromptTask} : std::vector<std::string>{"Segment objects"}));
    }
    // nothing known yet: both; a reply that is not a model folder's is refused
    CHECK(modelTaskChoices(std::nullopt).size() == 2);
    std::string error;
    CHECK_FALSE(modelFactsFromJson({{"format", "latents-bundle"}, {"tasks", {"track"}}}, &error));
    CHECK(error.find("not a model folder") != std::string::npos);
    CHECK(modelTaskOfLabel(kPromptTask) == "prompt");
    CHECK(modelTaskOfLabel("Segment objects") == "segment");
}

TEST_CASE("Models are listed by name and version from a models folder", "[app][ops][foundation]") {
    ModelsFolder models;
    models.add("coat-sam-s2", "v1", {"segment", "prompt"}, 1, "Promptable cells.");
    models.add("coat-sam-s2", "v2", {"segment"});
    models.add("alpha", "v1", {"segment"});   // no notes: the README's first paragraph
    std::filesystem::create_directories(models.root / "broken" / "v1");
    std::ofstream(models.root / "broken" / "v1" / "model.json") << "{";
    std::ofstream(models.root / "old.ltb") << "PK";
    std::filesystem::create_directories(models.root / "notes");   // a folder that holds no model
    const std::string missing = (models.root / "missing").string();
    const ModelListing got = listModelFolders({models.root.string(), missing});
    std::vector<std::string> rows;
    for (const ModelFolderFacts& m : got.models) rows.push_back(m.name + " " + m.version + (m.error.empty() ? "" : " !"));
    CHECK(rows == std::vector<std::string>{"old  !", "alpha v1", "broken v1 !", "coat-sam-s2 v1", "coat-sam-s2 v2"});
    CHECK(got.models[0].error.find("Old bundle format") != std::string::npos);
    CHECK(got.models[1].description == "Membrane-stained cells, from the README.");
    CHECK(got.models[3].description == "Promptable cells.");
    CHECK(got.models[3].promptable);
    CHECK(got.models[3].sizeBytes == 100);
    CHECK(got.models[3].title() == "coat-sam-s2 v1 \xC2\xB7 segment, prompt");
    CHECK(got.errors == std::vector<std::string>{"not a folder: " + missing});
    // a model folder named directly is its own listing
    const ModelListing one = listModelFolders({(models.root / "alpha" / "v1").string()});
    REQUIRE(one.models.size() == 1);
    CHECK(one.models[0].name == "alpha");

    // the worker's answer for a folder on the cluster reads the same
    const nlohmann::json reply = {
        {"models",
         {{{"format", "latents-model"}, {"path", "/clusterfs/models/coat-sam-s2/v1"}, {"name", "coat-sam-s2"}, {"version", "v1"}, {"tasks", {"segment", "prompt"}}, {"description", "Promptable cells."}, {"size_bytes", 123}, {"error", ""}},
          {{"path", "/clusterfs/models/broken/v1"}, {"name", "broken"}, {"version", "v1"}, {"tasks", nlohmann::json::array()}, {"error", "not valid JSON"}}}},
        {"errors", {"not a folder: /nope"}}};
    const ModelListing remote = modelListingFromJson(reply);
    REQUIRE(remote.models.size() == 2);
    CHECK(remote.models[0].path == "/clusterfs/models/coat-sam-s2/v1");
    CHECK(remote.models[0].promptable);
    CHECK(remote.models[0].sizeBytes == 123);
    CHECK(remote.models[1].error == "not valid JSON");
    CHECK(remote.errors == std::vector<std::string>{"not a folder: /nope"});
}

TEST_CASE("The segmentation step prompts a micro-SAM model with the same objects, plane by plane", "[app][ops][prompt][rpc]") {
    using nlohmann::json;
    const Dims5 dims{1, 2, 4, 20, 24};
    const DatasetMeta meta = metaFor(dims);
    auto data = rampArray(dims);
    const Operation& op = requireOperation("seg");
    CHECK(op.info().promptPlanar);
    ParamSet p = op.defaults();
    p.set("task", std::string(kPromptTask));
    // t 1: object 1 a box on plane 1 and a correction in it, object 2 a click
    // and a correction beside it on plane 2
    const std::vector<Prompt> prompts = {Prompt::boxOf({2, 2, 1, 8, 8, 2}, 1, 1), Prompt::point(5, 5, 1, 1, false, 1),
                                         Prompt::point(15, 9, 2, 1, true, 2), Prompt::point(16, 9, 2, 1, false, 2)};
    p.set(kPromptsKey, promptsValue(prompts));
    p.set("model", std::string("cellpose:cyto3"));
    Validation v = op.validate(p, meta);
    REQUIRE_FALSE(v.ok());
    CHECK(v.firstError().find("Cellpose cannot be prompted") != std::string::npos);
    p.set("model", std::string("model.pt"));
    v = op.validate(p, meta);
    REQUIRE_FALSE(v.ok());
    CHECK(std::any_of(v.errors.begin(), v.errors.end(), [](const std::string& e) { return e.find("Only micro-SAM") != std::string::npos; }));
    p.set("model", std::string("microsam:vit_b_lm"));
    REQUIRE(op.validate(p, meta).ok());   // boxes and corrections too, inside an object
    CHECK(op.summary(p, meta).find("micro-SAM vit_b_lm") != std::string::npos);
    CHECK(op.summary(p, meta).find("prompt") != std::string::npos);
    {
        // micro-SAM is 2-D: an object whose prompts straddle planes is refused,
        // as the worker would, while it is being placed
        ParamSet across = p;
        std::vector<Prompt> more = prompts;
        more.push_back(Prompt::point(5, 5, 3, 1, false, 1));
        across.set(kPromptsKey, promptsValue(more));
        const Validation b = op.validate(across, meta);
        REQUIRE_FALSE(b.ok());
        CHECK(b.firstError().find("Object 1 on time point 1 has prompts on planes z 1, 3") != std::string::npos);
        CHECK(b.firstError().find("2-D model") != std::string::npos);
    }
    // Segment all objects keeps every parameter of the automatic path
    ParamSet automatic = op.defaults();
    CHECK(automatic.getString("task") == "Segment all objects");
    CHECK_FALSE(isPromptStep(automatic));

    clearPromptRequests();
    const FakePrompt mode = GENERATE(FakePrompt::Promptable, FakePrompt::NotPromptable, FakePrompt::Refuses);
    std::string error;
    const StepOutput r = runWithFakeWorker(op, inputOf(data, meta), p, mode, &error);
    if (mode == FakePrompt::NotPromptable) {
        // the worker's own word (family_info's promptable) is the last one
        CHECK(error.find("cannot be prompted") != std::string::npos);
        CHECK(promptRequests.empty());
        return;
    }
    REQUIRE(promptRequests.size() == 1);   // t 0 has no prompts
    CHECK(promptRequests[0].kind == "torch_segment");
    CHECK(promptRequests[0].shape == std::vector<Index>{4, 20, 24});
    CHECK(promptRequests[0].params["task"] == "prompt");
    for (const char* flat : {"points", "point_labels", "boxes", "scribbles"}) CHECK_FALSE(promptRequests[0].params.contains(flat));
    CHECK(promptRequests[0].params["objects"] == json::parse(R"([
        {"box": [2, 2, 1, 8, 8, 2], "points": [[5, 5, 1]], "point_labels": [0]},
        {"points": [[15, 9, 2], [16, 9, 2]], "point_labels": [1, 0]}])"));
    if (mode == FakePrompt::Refuses) {
        // the worker's refusal reaches the person whole, saying whose it is
        CHECK(error.find("micro-SAM vit_b_lm refused the prompts of time point 1: ") != std::string::npos);
        CHECK(error.find("is a 2-D model and cannot refine across z") != std::string::npos);
        return;
    }
    REQUIRE(error.empty());
    REQUIRE(r.labels);
    const auto at = [&](Index z, Index y, Index x) { return r.labels->volume(1)[(z * dims.y + y) * dims.x + x]; };
    CHECK(at(1, 3, 3) == 1);
    CHECK(at(1, 5, 5) == 0);   // corrected
    CHECK(at(2, 9, 15) == 2);
    CHECK(at(2, 9, 16) == 0);  // corrected
    // micro-SAM's masks are per plane, and the step says so
    REQUIRE_FALSE(r.diagnostics.warnings.empty());
    CHECK(r.diagnostics.warnings.back().find("2-D model") != std::string::npos);
    CHECK(r.note.find("per plane") != std::string::npos);
    CHECK(factOf(r.diagnostics, "Mask scores") == "t 1: #1 0.90, #2 0.80");
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
    CHECK(r.diagnostics.curves[0].y.size() >= 2);
    CHECK(r.diagnostics.curves[0].y.size() <= 5);
    CHECK(r.diagnostics.curves[0].y.back() < r.diagnostics.curves[0].y.front());
    double changed = 0.0;
    float inPeak = 0.0f, outPeak = 0.0f;
    for (Index z = 0; z < dims.z; ++z)
        for (Index y = 0; y < dims.y; ++y)
            for (Index x = 0; x < dims.x; ++x) {
                const float a = data->at(0, 0, z, y, x), b = r.array->at(0, 0, z, y, x);
                changed += std::abs(a - b);
                inPeak = std::max(inPeak, a);
                outPeak = std::max(outPeak, b);
            }
    CHECK(changed > 1.0);          // a delta PSF, or the input returned unchanged, fails
    CHECK(outPeak > inPeak);       // the blob gets sharper
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

TEST_CASE("Frangi and Meijering find a line along x and along z when dz is not dx", "[app][ops][classic]") {
    // dz = 3 dx. A filament along x is a few voxels thick in z; the same
    // filament along z is that many voxels thick in x and y. Voxel curvature
    // without the spacing scale reads the first as a sheet.
    const Dims5 dims{1, 1, 24, 48, 48};
    DatasetMeta meta = metaFor(dims, 0.1, 0.3);
    auto data = std::make_shared<Array5>(Array5::zeros(dims));
    for (Index x = 4; x < 44; ++x)
        for (Index y = 10; y <= 16; ++y)
            for (Index z = 5; z <= 7; ++z) data->at(0, 0, z, y, x) = 1000.0f;   // along x, ~0.7 um across
    for (Index z = 2; z < 22; ++z)
        for (Index y = 30; y <= 36; ++y)
            for (Index x = 30; x <= 36; ++x) data->at(0, 0, z, y, x) = 1000.0f;  // along z
    const Operation& op = requireOperation("classic");
    ParamSet p = op.defaults();
    p.set("enhance_sigma", 1.5);
    p.set("enhance_sigma_max", 3.0);
    p.set("enhance_scales", std::int64_t{3});
    p.set("sigma", 0.0);
    p.set("opening", std::int64_t{0});
    p.set("fill_holes", false);
    p.set("method", std::string("Manual"));
    // Unscaled z curvature (dz = 3 dx) leaves the x-line's vesselness near 0.07,
    // under this cut. The scaled Hessian clears it. Meijering stays above either way;
    // it uses the same derivatives, checked on the x-line at the lower cut.
    p.set("value", 0.05);
    p.set("post", std::string("Connected components"));
    p.set("min_voxels", std::int64_t{8});
    Progress prog;
    for (const char* enhance : {"Tubes (Frangi)", "Neurites (Meijering)"}) {
        p.set("enhance", std::string(enhance));
        p.set("value", std::string(enhance) == "Tubes (Frangi)" ? 0.12 : 0.05);
        const StepOutput r = op.run(inputOf(data, meta), p, prog.ctx);
        REQUIRE(r.labels);
        INFO(enhance);
        CHECK(r.labels->at(0, 6, 13, 24) != 0);   // middle of the line along x
        CHECK(r.labels->at(0, 12, 33, 33) != 0);  // middle of the line along z
        CHECK(r.labels->at(0, 6, 2, 2) == 0);
    }
}

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
        for (LabelStats& s : labels->stats())
            if (s.id == 7) {
                s.cls = "debris";   // the speck that goes, and whose number object 9 had better not inherit its mark with
                s.reviewed = true;
            }
        ParamSet cp = cleanup.defaults();
        cp.set("min_voxels", std::int64_t{2});
        cp.set("relabel", true);
        const StepOutput out = cleanup.run(in, cp, prog.ctx);
        REQUIRE(out.labels);
        CHECK(out.labels->at(0, 0, 2, 2) == 1);
        CHECK(out.labels->at(0, 0, 5, 5) == 2);
        CHECK(out.labels->maxLabel() == 2);
        // the statistics follow the objects to their new numbers, not the numbers
        const LabelStats* was9 = out.labels->statsOf(2);
        REQUIRE(was9);
        CHECK(was9->confidence == 0.3);
        CHECK(was9->cls == "object");
        CHECK_FALSE(was9->reviewed);
        REQUIRE(out.labels->statsOf(1));
        CHECK(out.labels->statsOf(1)->confidence == 1.0);
    }
    SECTION("recomputeStats without probabilities keeps a known confidence") {
        labels->recomputeStats(0);
        REQUIRE(labels->statsOf(9));
        CHECK(labels->statsOf(9)->confidence == 0.3);
    }
}

TEST_CASE("Label cleanup numbers every frame with one map", "[app][ops][cleanup][track]") {
    // tracked labels: track 4 in every frame, track 2 only from t = 1, and a
    // speck of id 3 in t = 0 that the size filter drops
    const Dims5 dims{1, 3, 1, 16, 16};
    const DatasetMeta meta = metaFor(dims);
    auto data = std::make_shared<Array5>(Array5::zeros(dims));
    auto labels = std::make_shared<LabelVolume>(3, 1, 16, 16);
    auto square = [&](Index t, Index y0, Index x0, std::uint32_t id) {
        for (Index y = y0; y < y0 + 3; ++y)
            for (Index x = x0; x < x0 + 3; ++x) labels->volume(t)[y * 16 + x] = id;
    };
    for (Index t = 0; t < 3; ++t) {
        square(t, 10, 10, 4);
        if (t >= 1) square(t, 2, 2, 2);
    }
    labels->volume(0)[15] = 3;
    for (Index t = 0; t < 3; ++t) {
        labels->recomputeStats(t);
        for (LabelStats& s : labels->stats()) s.cls = "track";
    }
    labels->setTracked(true);
    labels->recomputeStats(1);
    for (LabelStats& s : labels->stats())
        if (s.id == 4) s.reviewed = true;
    const Operation& cleanup = requireOperation("cleanup");
    ParamSet cp = cleanup.defaults();
    cp.set("min_voxels", std::int64_t{2});
    Progress prog;
    StepInput in = inputOf(data, meta);
    in.labels = labels;
    const StepOutput out = cleanup.run(in, cp, prog.ctx);
    REQUIRE(out.labels);
    const LabelVolume& c = *out.labels;
    CHECK(c.tracked());
    const std::uint32_t big = c.at(0, 0, 11, 11), late = c.at(1, 0, 3, 3);
    CHECK(big == 2);    // ids 2 and 4 are left: numbered 1 and 2 in every frame
    CHECK(late == 1);
    for (Index t = 0; t < 3; ++t) {
        INFO("frame " << t);
        CHECK(c.at(t, 0, 11, 11) == big);   // was 1 in t = 0 and 2 afterwards
        if (t >= 1) CHECK(c.at(t, 0, 3, 3) == late);
        CHECK(c.annotationOf(t, big).reviewed);   // the mark went with the track
        CHECK(c.annotationOf(t, big).cls == "track");
    }
    CHECK(c.at(0, 0, 0, 15) == 0);
    CHECK(c.maxLabel() == 2);

    SECTION("deleting a track afterwards takes that track and nothing else") {
        auto edited = out.labels->clone();
        for (Index t = 0; t < 3; ++t) edited->remove(t, late);   // what Workbench::deleteLabel does on tracked labels
        for (Index t = 0; t < 3; ++t) {
            INFO("frame " << t);
            CHECK(edited->at(t, 0, 11, 11) == big);
            CHECK(edited->at(t, 0, 3, 3) == 0);
        }
    }
    SECTION("without relabel the ids stay as they were") {
        cp.set("relabel", false);
        const StepOutput kept = cleanup.run(in, cp, prog.ctx);
        for (Index t = 0; t < 3; ++t) CHECK(kept.labels->at(t, 0, 11, 11) == 4);
        CHECK(kept.labels->at(2, 0, 3, 3) == 2);
    }
}

TEST_CASE("Track objects marks its labels tracked only when it gives them track ids", "[app][ops][track]") {
    // two objects that swap nothing but their numbering between the frames
    const Dims5 dims{1, 2, 1, 16, 16};
    const DatasetMeta meta = metaFor(dims);
    auto data = std::make_shared<Array5>(Array5::zeros(dims));
    auto labels = std::make_shared<LabelVolume>(2, 1, 16, 16);
    auto square = [&](Index t, Index y0, Index x0, std::uint32_t id) {
        for (Index y = y0; y < y0 + 3; ++y)
            for (Index x = x0; x < x0 + 3; ++x) labels->volume(t)[y * 16 + x] = id;
    };
    square(0, 2, 2, 1);
    square(0, 10, 10, 2);
    square(1, 2, 3, 2);    // the first object, numbered 2 in this frame
    square(1, 10, 11, 1);
    for (Index t = 0; t < 2; ++t) labels->recomputeStats(t);
    const Operation& op = requireOperation("track");
    Progress prog;
    StepInput in = inputOf(data, meta);
    in.labels = labels;

    ParamSet p = op.defaults();
    const StepOutput relabelled = op.run(in, p, prog.ctx);
    REQUIRE(relabelled.labels);
    CHECK(relabelled.labels->tracked());
    CHECK(relabelled.labels->at(1, 0, 3, 4) == relabelled.labels->at(0, 0, 3, 3));

    p.set("relabel", false);
    const StepOutput asSegmented = op.run(in, p, prog.ctx);
    REQUIRE(asSegmented.labels);
    // id 1 is a different object in each frame: a delete of "track 1" must not
    // take both, which is what the tracked flag would make it do
    CHECK_FALSE(asSegmented.labels->tracked());
    CHECK(asSegmented.labels->at(1, 0, 3, 4) == 2);

    SECTION("only btrack needs the Python worker") {
        CHECK_FALSE(op.needsWorker(op.defaults()));
        ParamSet bayes = op.defaults();
        bayes.set("tracker", std::string("btrack (Bayesian)"));
        CHECK(op.needsWorker(bayes));
        CHECK(op.info().remoteCapable);   // the worker still implements it, for the HPC hints
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
