// The workbench without a window, driven by tool calls (app/core/headless.hpp):
// the tool table sirius-cli serves, on a real Workbench with the bundled test
// data. Every case points SIRIUS_PYTHON_ENV at a directory of its own, so no
// case can read or change the developer's own Python environment, and runs
// with plugins off unless it is about the worker.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <sirius/buffer.hpp>
#include <sirius/tiff_io.hpp>

#include "core/array_source.hpp"
#include "core/headless.hpp"
#include "core/host.hpp"
#include "core/labels.hpp"

#include "temp_path.hpp"

using namespace sirius;
using namespace sirius::app;
using json = nlohmann::json;
using Catch::Matchers::ContainsSubstring;

namespace {

    const std::filesystem::path kData = SIRIUS_TEST_DATA_DIR;
    const std::filesystem::path kExamples = SIRIUS_TEST_EXAMPLES_DIR;

    // Sets an environment variable for the scope and puts back what was there.
    class ScopedEnv {
    public:
        ScopedEnv(const char* name, const std::string& value) : name_(name) {
            if (host::hasEnvironment(name)) old_ = host::environment(name);
            set(value.c_str());
        }
        ~ScopedEnv() { set(old_ ? old_->c_str() : nullptr); }
        ScopedEnv(const ScopedEnv&) = delete;
        ScopedEnv& operator=(const ScopedEnv&) = delete;

    private:
        void set(const char* value) {
#ifdef _WIN32
            _putenv_s(name_, value ? value : "");
#else
            if (value) setenv(name_, value, 1);
            else unsetenv(name_);
#endif
        }
        const char* name_;
        std::optional<std::string> old_;
    };

    // A directory of the case's own, removed at the end.
    struct TempDir {
        std::filesystem::path path;
        explicit TempDir(const char* tag) : path(test::uniqueTempPath(tag, "")) { std::filesystem::create_directories(path); }
        ~TempDir() {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }
        TempDir(const TempDir&) = delete;
        TempDir& operator=(const TempDir&) = delete;
    };

    // Sleeps `ms` in small slices, stopping at a cancel: a run that takes a
    // while without needing any data of its own.
    struct HeadlessSlowOp final : Operation {
        OpInfo info_;
        HeadlessSlowOp() {
            info_.kind = "test_headless_slow";
            info_.name = "Headless slow";
            info_.group = "Intensity";
            info_.kindLabel = "INTENSITY";
            info_.params = {intParam("ms", "Milliseconds", 600).range(0, 60000)};
        }
        const OpInfo& info() const noexcept override { return info_; }
        StepOutput run(const StepInput& in, const ParamSet& p, const StepContext& ctx) const override {
            const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(p.getInt("ms"));
            while (std::chrono::steady_clock::now() < until) {
                ctx.throwIfCancelled();
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            StepOutput o;
            o.meta = in.meta;
            o.array = in.materialize();
            return o;
        }
    };

    void registerSlowOp() {
        static const bool once = [] {
            registerOperation(std::make_unique<HeadlessSlowOp>());
            return true;
        }();
        (void)once;
    }

    // One case's workspace: its scratch, its own Python environment
    // directory, and the workbench.
    struct Fixture {
        TempDir scratch{"headless"};
        TempDir pyenv{"headless_pyenv"};
        ScopedEnv envDir{"SIRIUS_PYTHON_ENV", (pyenv.path / "env").string()};
        std::unique_ptr<HeadlessWorkbench> h;
        agent::CallContext ctx;
        std::atomic<bool> cancel{false};

        explicit Fixture(const std::function<void(HeadlessOptions&)>& configure = {}) {
            registerSlowOp();
            HeadlessOptions o;
            o.scratchDir = scratch.path;
            o.plugins = HeadlessOptions::Plugins::Off;
            o.backend = "cpu";
            if (configure) configure(o);
            h = std::make_unique<HeadlessWorkbench>(std::move(o));
            ctx.cancelled = [this] { return cancel.load(); };
            ctx.progress = [](double, const std::string&) {};
        }

        agent::ToolResult call(const std::string& name, const json& args = json::object()) { return h->call(name, args, ctx); }

        // A call that has to succeed: its value.
        json ok(const std::string& name, const json& args = json::object()) {
            const agent::ToolResult r = call(name, args);
            INFO(name << " " << args.dump() << " -> " << r.error.code << ": " << r.error.message);
            REQUIRE(r.ok);
            return r.value;
        }

        json openRaw() { return ok("open_dataset", {{"path", (kData / "raw.tif").string()}}); }
    };

    bool isPng(const std::vector<std::uint8_t>& b) { return b.size() > 24 && b[0] == 0x89 && b[1] == 'P' && b[2] == 'N' && b[3] == 'G'; }

    std::uint32_t bigEndian(const std::vector<std::uint8_t>& b, std::size_t at) {
        return (std::uint32_t{b[at]} << 24) | (std::uint32_t{b[at + 1]} << 16) | (std::uint32_t{b[at + 2]} << 8) | std::uint32_t{b[at + 3]};
    }

    // A Python for a fake worker: $SIRIUS_PYTHON, else one on this machine.
    std::string anyPython() {
        const std::string named = host::environment("SIRIUS_PYTHON");
        return named.empty() ? host::findPython() : named;
    }

} // namespace

TEST_CASE("headless: the tool table has no view tools and strict schemas", "[app][headless]") {
    Fixture f;
    const std::vector<agent::ToolDescriptor> tools = f.h->tools();
    CHECK(tools.size() == 48);
    std::vector<std::string> names;
    const std::regex name("[A-Za-z0-9_.-]{1,64}");
    for (const agent::ToolDescriptor& t : tools) {
        INFO(t.name);
        names.push_back(t.name);
        CHECK(std::regex_match(t.name, name));
        CHECK_FALSE(t.description.empty());
        CHECK(t.description.size() <= 2048);
        CHECK_FALSE(t.title.empty());
        CHECK(t.inputSchema["type"] == "object");
        CHECK(t.inputSchema["additionalProperties"] == false);
        REQUIRE(t.inputSchema["properties"].is_object());
        CHECK(t.inputSchema["properties"].contains("workspace"));
        CHECK_FALSE(t.inputSchema["properties"].contains("out"));
        CHECK(f.h->hasTool(t.name));
    }
    for (const char* view : {"view_step", "select_step", "set_view", "focus_track"}) {
        CHECK(std::find(names.begin(), names.end(), view) == names.end());
        CHECK_FALSE(f.h->hasTool(view));
    }
    for (const char* wanted : {"open_dataset", "load_pipeline", "render", "statistics", "run", "run_status", "export_result", "setup_worker_env",
                               "list_labels", "paint_label", "fill_label", "merge_labels", "split_label", "delete_label", "clear_labels", "export_labels"})
        CHECK(std::find(names.begin(), names.end(), wanted) != names.end());
    const auto byName = [&](const std::string& n) {
        return *std::find_if(tools.begin(), tools.end(), [&](const agent::ToolDescriptor& t) { return t.name == n; });
    };
    CHECK_FALSE(byName("set_backend").inputSchema["properties"].contains("hpc"));
    CHECK_FALSE(byName("set_backend").inputSchema["properties"].contains("host"));
    CHECK(byName("render").hints.readOnly);
    CHECK(byName("export_result").hints.destructive);
    CHECK(byName("export_labels").hints.destructive);
    CHECK(byName("list_labels").hints.readOnly);
    CHECK_FALSE(byName("paint_label").hints.readOnly);
    CHECK(byName("setup_worker_env").hints.openWorld);
    CHECK(byName("run").hints.openWorld);
    CHECK_FALSE(byName("render").hints.openWorld);
    CHECK(byName("setup_worker_env").meta["anthropic/requiresUserInteraction"] == true);
    CHECK(byName("get_help").meta["anthropic/maxResultSizeChars"] == 200000);
    CHECK(byName("set_params").hints.idempotent);
    // values the tools take in any case are not held to one spelling by an enum
    CHECK_FALSE(byName("set_backend").inputSchema["properties"]["backend"].contains("enum"));
    CHECK_FALSE(byName("render").inputSchema["properties"]["plane"].contains("enum"));
    CHECK_FALSE(byName("export_result").inputSchema["properties"]["format"].contains("enum"));
    CHECK_THAT(byName("render").inputSchema["properties"]["plane"]["description"].get<std::string>(), ContainsSubstring("mip"));

    SECTION("a read-only server lists no tool that writes files or downloads") {
        Fixture ro([](HeadlessOptions& o) { o.readOnly = true; });
        for (const agent::ToolDescriptor& t : ro.h->tools()) CHECK_FALSE(t.hints.destructive);
        CHECK_FALSE(ro.h->hasTool("export_result"));
        CHECK(ro.call("save_pipeline", {{"path", (ro.scratch.path / "p.sirius.toml").string()}}).error.code == "unknown_tool");
        CHECK(ro.h->hasTool("render"));
    }
}

TEST_CASE("headless: the workspace starts with the Load step alone", "[app][headless]") {
    Fixture f;
    const json state = f.ok("get_state");
    CHECK(state["workspace"] == f.h->workspaceId());
    CHECK(std::regex_match(f.h->workspaceId(), std::regex("ws_[0-9a-f]{12}")));
    REQUIRE(state["steps"].size() == 1);
    CHECK(state["steps"][0]["kind"] == "load");
    CHECK_FALSE(state["steps"][0].contains("selected"));
    CHECK_FALSE(state["steps"][0].contains("viewed"));
    CHECK(state["can_undo"] == false);
    CHECK(state["backend"] == "cpu");
    CHECK(state["dataset"].is_null());
    CHECK(state["running"] == false);
}

TEST_CASE("headless: open_dataset describes raw.tif", "[app][headless]") {
    Fixture f;
    const json info = f.openRaw();
    CHECK(info["workspace"] == f.h->workspaceId());
    CHECK(info["dims"]["z"].get<long long>() == 135);
    CHECK(info["dims"]["y"].get<long long>() == 64);
    CHECK(info["dims"]["x"].get<long long>() == 64);
    CHECK(info["dims"]["c"].get<long long>() == 1);
    CHECK(info["shape"].get<std::string>().find("z135") != std::string::npos);
    CHECK(info["dtype"].is_string());
    CHECK(info["voxel_um"].size() == 3);
    CHECK(info["channels"].size() == 1);
    CHECK(info["sim"].is_object());
    CHECK(info["float32_bytes"].get<long long>() == 135LL * 64 * 64 * 4);
    const std::string path = info["path"].get<std::string>();
    CHECK(path.find('\\') == std::string::npos);
    CHECK(std::filesystem::path(path).is_absolute());
    CHECK_THAT(path, ContainsSubstring("raw.tif"));

    SECTION("a missing file is not_found, an unusable option invalid_argument") {
        CHECK(f.call("open_dataset", {{"path", (kData / "no-such.tif").string()}}).error.code == "not_found");
        CHECK(f.call("open_dataset", {{"path", (kData / "raw.tif").string()}, {"page_order", "cxy"}}).error.code == "invalid_argument");
        CHECK(f.call("open_dataset", {{"path", (kData / "raw.tif").string()}, {"voxel_um", {1, 2}}}).error.code == "invalid_argument");
    }
    SECTION("dataset_info probes a file without opening it") {
        const json probed = f.ok("dataset_info", {{"path", (kData / "otf.tif").string()}});
        CHECK(probed["dims"]["x"].get<long long>() > 0);
        CHECK(f.ok("dataset_info")["path"] == path);
    }
    SECTION("an unknown argument is ignored with a warning") {
        const agent::ToolResult r = f.call("get_state", {{"verbose", true}});
        REQUIRE(r.ok);
        REQUIRE(r.warnings.size() == 1);
        CHECK_THAT(r.warnings[0], ContainsSubstring("verbose"));
    }
}

TEST_CASE("headless: render of the Load step writes a PNG under the scratch renders", "[app][headless]") {
    Fixture f;
    f.openRaw();
    const std::filesystem::path renders = f.scratch.path / "renders";
    for (const char* plane : {"xy", "mip", "xz"}) {
        INFO(plane);
        const agent::ToolResult r = f.call("render", {{"plane", plane}});
        INFO(r.error.code << ": " << r.error.message);
        REQUIRE(r.ok);
        REQUIRE(r.images.size() == 1);
        const agent::Attachment& img = r.images[0];
        CHECK(img.mimeType == "image/png");
        REQUIRE(isPng(img.bytes));
        CHECK(img.bytes.size() <= (std::size_t{4} << 20));
        CHECK(bigEndian(img.bytes, 16) == static_cast<std::uint32_t>(img.width));
        CHECK(bigEndian(img.bytes, 20) == static_cast<std::uint32_t>(img.height));
        CHECK(r.value["width"] == img.width);
        CHECK(r.value["height"] == img.height);
        CHECK(r.value["step"] == 1);
        CHECK(r.value["plane"] == plane);
        const std::filesystem::path file = std::filesystem::u8path(r.value["path"].get<std::string>());
        CHECK(std::filesystem::exists(file));
        CHECK(std::filesystem::equivalent(file.parent_path(), renders));
        CHECK(img.path == r.value["path"].get<std::string>());
        if (std::string(plane) == "xz") {
            // 135 planes of 0.2 um over 0.1 um pixels (the file's own voxel size) is taller than 135 rows
            CHECK(img.width == 64);
            CHECK(img.height > 0);
        } else {
            CHECK(img.width == 64);
            CHECK(img.height == 64);
        }
    }

    SECTION("a grid of z planes, a region and a smaller size") {
        const json grid = f.ok("render", {{"z", {0, 10, 20, 30}}, {"max_size", 100}});
        CHECK(grid["width"].get<int>() <= 100);
        CHECK(grid["height"].get<int>() <= 100);
        CHECK(grid["z"].size() == 4);
        const json region = f.ok("render", {{"region", {8, 8, 16, 16}}});
        CHECK(region["width"] == 16);
        CHECK(region["height"] == 16);
    }
    SECTION("the request is checked") {
        CHECK(f.call("render", {{"plane", "xw"}}).error.code == "invalid_argument");
        CHECK(f.call("render", {{"z", 1000}}).error.code == "invalid_argument");
        CHECK(f.call("render", {{"channels", {3}}}).error.code == "invalid_argument");
        CHECK(f.call("render", {{"step", 7}}).error.code == "unknown_step");
    }
    SECTION("probe reads the voxel the render shows") {
        const json p = f.ok("probe", {{"x", 10}, {"y", 12}, {"z", 3}});
        REQUIRE(p["values"].size() == 1);
        CHECK(p["values"][0]["value"].is_number());
        CHECK(p["label"].is_null());
        CHECK(f.call("probe", {{"x", 64}, {"y", 0}}).error.code == "invalid_argument");
    }
}

TEST_CASE("headless: a full-range window is the whole volume's range, not its sampled planes'", "[app][headless]") {
    Fixture f;
    // nine planes: the model samples planes 0, 2, 4, 6 and 8, and the one bright voxel is on plane 3
    const std::filesystem::path stack = f.scratch.path / "bright.tif";
    Buffer<std::uint16_t> voxels(Shape{9, 16, 16});
    for (Index i = 0; i < voxels.size(); ++i) voxels.data()[i] = static_cast<std::uint16_t>(100 + i % 50);
    voxels.data()[3 * 256 + 5 * 16 + 5] = 5000;
    writeTiffStack<std::uint16_t>(stack.string(), voxels.view(), TiffCompression::None);
    f.ok("open_dataset", {{"path", stack.u8string()}});
    const json caption = f.ok("render", {{"z", 4}, {"window", "full"}});
    REQUIRE(caption["channels"].size() == 1);
    CHECK(caption["channels"][0]["window"][0].get<double>() == 100.0);
    CHECK(caption["channels"][0]["window"][1].get<double>() == 5000.0);
    // an explicit window is kept as given
    const json given = f.ok("render", {{"z", 4}, {"window", "full"}, {"windows", {{{"channel", 0}, {"lo", 110}, {"hi", 140}}}}});
    CHECK(given["channels"][0]["window"][1].get<double>() == 140.0);
}

TEST_CASE("headless: statistics of the Load step reports the saturated fraction", "[app][headless]") {
    Fixture f;
    // an integer stack with 10 of its 1024 voxels at the type's maximum
    const std::filesystem::path stack = f.scratch.path / "saturated.tif";
    Buffer<std::uint16_t> voxels(Shape{4, 16, 16});
    for (Index i = 0; i < voxels.size(); ++i) voxels.data()[i] = static_cast<std::uint16_t>(100 + i % 50);
    for (Index i = 0; i < 10; ++i) voxels.data()[i * 37] = 65535;
    writeTiffStack<std::uint16_t>(stack.string(), voxels.view(), TiffCompression::None);
    f.ok("open_dataset", {{"path", stack.u8string()}});
    const agent::ToolResult r = f.call("statistics", {{"histogram_bins", 16}});
    INFO(r.error.code << ": " << r.error.message);
    REQUIRE(r.ok);
    REQUIRE(r.value["channels"].size() == 1);
    const json& c = r.value["channels"][0];
    CHECK(c["count"].get<long long>() == 4 * 16 * 16);
    CHECK(c["max"].get<double>() == 65535.0);
    CHECK(c["min"].get<double>() == 100.0);
    CHECK(c["percentiles"].contains("50"));
    CHECK(c["percentiles"].contains("99.9"));
    REQUIRE(c.contains("saturated_fraction"));
    CHECK(c["saturated_fraction"].get<double>() == 10.0 / 1024.0);
    REQUIRE(c.contains("histogram"));
    long long total = 0;
    for (const json& n : c["histogram"]["counts"]) total += n.get<long long>();
    CHECK(total == 4 * 16 * 16);
    CHECK(r.value["step"] == 1);
    CHECK(r.value["t"] == 0);
    CHECK_FALSE(r.value.contains("labels"));

    SECTION("a float stack has no ceiling to saturate at") {
        f.openRaw();   // float32
        const json floats = f.ok("statistics", {{"t", "all"}});
        CHECK(floats["t"] == "all");
        CHECK(floats["channels"][0]["count"].get<long long>() == 135LL * 64 * 64);
        CHECK_FALSE(floats["channels"][0].contains("saturated_fraction"));
    }
}

TEST_CASE("headless: a contrast step is added, run and rendered", "[app][headless]") {
    Fixture f;
    f.openRaw();
    const agent::ToolResult added = f.call("add_step", {{"kind", "contrast"}});
    REQUIRE(added.ok);
    CHECK(added.value["step"] == 2);
    CHECK_FALSE(added.value.contains("selected"));
    CHECK(added.undoable);
    CHECK_FALSE(added.changes.empty());
    CHECK(f.call("add_step", {{"kind", "no_such_kind"}}).error.code == "unknown_operation");

    const agent::ToolResult ran = f.call("run", {{"wait_s", -1}});
    INFO(ran.error.code << ": " << ran.error.message);
    REQUIRE(ran.ok);
    CHECK(ran.value["status"] == "succeeded");
    CHECK(ran.value["target_step"] == 2);
    CHECK(ran.value["run_id"] == "r1");
    CHECK(ran.value["output"]["step"] == 2);
    CHECK(ran.value["output"]["fresh"] == true);
    CHECK_FALSE(ran.value["steps"].empty());
    CHECK(ran.value["log"].is_array());

    // with no step named, the inspecting tools look at what the run made
    const json caption = f.ok("render");
    CHECK(caption["step"] == 2);
    CHECK(caption["fresh"] == true);
    CHECK(f.ok("get_diagnostics")["step"] == 2);
    CHECK(f.ok("run_status")["status"] == "succeeded");

    SECTION("undo takes the step away again") {
        const agent::ToolResult undone = f.call("undo");
        REQUIRE(undone.ok);
        CHECK(undone.value["ok"] == true);
        CHECK(f.ok("get_state")["steps"].size() == 1);
    }
    SECTION("a step that has not run is not_computed unless run:true") {
        f.ok("set_params", {{"step", 2}, {"params", {{"lo_percentile", 1.0}}}});
        const agent::ToolResult stale = f.call("render", {{"step", 2}});
        REQUIRE(stale.ok);   // the old output, said to be stale
        CHECK(stale.value["fresh"] == false);
        CHECK_FALSE(stale.warnings.empty());
        f.ok("add_step", {{"kind", "contrast"}});
        const agent::ToolResult missing = f.call("render", {{"step", 3}});
        CHECK(missing.error.code == "not_computed");
        CHECK_THAT(missing.error.hint, ContainsSubstring("run:true"));
        const json computed = f.ok("render", {{"step", 3}, {"run", true}});
        CHECK(computed["fresh"] == true);
    }
}

TEST_CASE("headless: add_step takes a preset, parameters and a name, and checks them first", "[app][headless]") {
    Fixture f;
    f.openRaw();
    // Not seeded from the raw data on hand: the window stays automatic.
    const agent::ToolResult contrast = f.call("add_step", {{"kind", "contrast"}, {"params", {{"gamma", 99}}}});
    REQUIRE(contrast.ok);
    CHECK(contrast.value["params"]["min"] == 0.0);
    CHECK(contrast.value["params"]["max"] == 0.0);
    CHECK(contrast.value["params"]["gamma"] == 5.0);
    CHECK_FALSE(contrast.value.contains("number"));
    REQUIRE(contrast.value.contains("clamped"));
    CHECK_THAT(contrast.value["clamped"][0].get<std::string>(), ContainsSubstring("gamma"));
    CHECK_FALSE(contrast.warnings.empty());

    const json seg = f.ok("add_step", {{"kind", "classic"}, {"preset", "Filaments"}, {"name", "Seg"}, {"params", {{"denoise", "Median 3x3"}}}});
    CHECK(seg["name"] == "Seg");
    CHECK(seg["params"]["denoise"] == "Median 3x3");
    CHECK(f.ok("get_step", {{"step", "Seg"}})["step"] == 3);

    // A refused call leaves the pipeline as it was.
    const agent::ToolResult preset = f.call("add_step", {{"kind", "classic"}, {"preset", "No such preset"}});
    CHECK(preset.error.code == "invalid_argument");
    const agent::ToolResult key = f.call("add_step", {{"kind", "contrast"}, {"params", {{"nope", 1}}}});
    CHECK(key.error.code == "invalid_argument");
    CHECK_THAT(key.error.hint, ContainsSubstring("lo_percentile"));
    CHECK(f.ok("get_state")["steps"].size() == 3);
    const agent::ToolResult set = f.call("set_params", {{"step", 2}, {"params", {{"nope", 1}}}});
    CHECK(set.error.code == "invalid_argument");
    CHECK_THAT(set.error.hint, ContainsSubstring("gamma"));
}

TEST_CASE("headless: export_result writes an OME-TIFF that opens with the same dims", "[app][headless]") {
    Fixture f;
    const json info = f.openRaw();
    const std::filesystem::path out = f.scratch.path / "export" / "raw copy.ome.tif";
    std::filesystem::create_directories(out.parent_path());
    const agent::ToolResult r = f.call("export_result", {{"path", out.u8string()}, {"dtype", "uint16"}, {"include_pipeline", true}});
    INFO(r.error.code << ": " << r.error.message);
    REQUIRE(r.ok);
    CHECK(r.value["format"] == "ome-tiff");
    CHECK(r.value["dtype"] == "uint16");
    CHECK(r.value["bytes"].get<long long>() > 0);
    CHECK(r.value["files"].size() == 2);   // the image and its pipeline sidecar
    REQUIRE(std::filesystem::exists(out));
    CHECK(std::filesystem::exists(out.u8string() + ".pipeline.toml"));
    const DatasetMeta written = probeDataset(out.u8string());
    CHECK(written.dims.z == info["dims"]["z"].get<long long>());
    CHECK(written.dims.y == info["dims"]["y"].get<long long>());
    CHECK(written.dims.x == info["dims"]["x"].get<long long>());

    SECTION("a file an earlier export left beside it is not reported as written") {
        const std::filesystem::path again = out.parent_path() / "again.tif";
        std::ofstream(out.parent_path() / "again.labels.tif") << "the labels of an earlier export";
        const json second = f.ok("export_result", {{"path", again.u8string()}, {"dtype", "uint16"}});
        REQUIRE(second["files"].size() == 1);
        CHECK(second["bytes"].get<long long>() == static_cast<long long>(std::filesystem::file_size(again)));
    }
    SECTION("options that do not fit are refused before anything is written") {
        CHECK(f.call("export_result", {{"path", (f.scratch.path / "x.bmp").u8string()}}).error.code == "invalid_argument");
        CHECK(f.call("export_result", {{"path", (f.scratch.path / "x.tif").u8string()}, {"z", {5, 5}}}).error.code == "invalid_argument");
        CHECK(f.call("export_result", {{"path", (f.scratch.path / "x.tif").u8string()}, {"labels_only", true}}).error.code == "invalid_argument");
    }
}

TEST_CASE("headless: labels are imported from a TIFF, edited by the label tools, exported and imported again", "[app][headless]") {
    Fixture f;
    const json info = f.openRaw();
    const Index nz = info["dims"]["z"].get<Index>(), ny = info["dims"]["y"].get<Index>(), nx = info["dims"]["x"].get<Index>();
    REQUIRE(nz >= 8);
    REQUIRE(ny >= 32);
    REQUIRE(nx >= 32);
    // two objects on the dataset's grid: a cube (1) and a bar along x (2)
    Buffer<std::uint32_t> pages(Shape{nz, ny, nx});
    std::fill(pages.data(), pages.data() + pages.size(), 0u);
    auto at = [&](Index z, Index y, Index x) -> std::uint32_t& { return pages.data()[(z * ny + y) * nx + x]; };
    for (Index z = 1; z <= 3; ++z) {
        for (Index y = 2; y <= 6; ++y)
            for (Index x = 2; x <= 6; ++x) at(z, y, x) = 1;
        for (Index y = 10; y <= 12; ++y)
            for (Index x = 2; x <= 21; ++x) at(z, y, x) = 2;
    }
    const std::filesystem::path labelsTif = f.scratch.path / "two objects.tif";
    writeTiffStack<std::uint32_t>(labelsTif.u8string(), pages.view(), TiffCompression::Deflate);

    // before any labels exist the tools say so, by name
    CHECK(f.call("list_labels", {{"step", 1}}).error.code == "no_labels");
    CHECK(f.call("paint_label", {{"x", 1}, {"y", 1}, {"z", 1}}).error.code == "no_labels");

    const json step = f.ok("add_step", {{"kind", "import_labels"}, {"params", {{"path", labelsTif.u8string()}}}, {"name", "Labels"}});
    CHECK(step["step"] == 2);
    const agent::ToolResult ran = f.call("run", {{"wait_s", -1}});
    INFO(ran.error.code << ": " << ran.error.message);
    REQUIRE(ran.ok);
    CHECK(ran.value["output"]["labels"]["count"] == 2);

    // list: largest first, with ids and extents in (x, y, z)
    json listed = f.ok("list_labels");
    REQUIRE(listed["count"] == 2);
    REQUIRE(listed["labels"].size() == 2);
    CHECK(listed["labels"][0]["id"] == 2);   // the bar: 3 * 3 * 20 voxels
    CHECK(listed["labels"][0]["voxels"] == 180);
    CHECK(listed["labels"][1]["id"] == 1);
    CHECK(listed["labels"][1]["voxels"] == 75);
    CHECK(listed["labels"][1]["bbox"]["x0"] == 2);
    CHECK(listed["labels"][1]["bbox"]["x1"] == 7);
    CHECK(listed["labels"][1]["bbox"]["z0"] == 1);
    CHECK(listed["labels"][1]["bbox"]["z1"] == 4);

    // paint a new object where there is nothing: a third label, a ball over three planes
    const json painted = f.ok("paint_label", {{"x", 25}, {"y", 25}, {"z", 2}, {"radius", 4}, {"z_radius", 1}});
    CHECK(painted["label"] == 3);
    CHECK(painted["voxels"].get<Index>() > 0);
    CHECK(painted["labels"] == 3);
    CHECK(f.ok("probe", {{"step", 2}, {"x", 25}, {"y", 25}, {"z", 2}})["label"]["id"] == 3);
    CHECK(f.ok("probe", {{"step", 2}, {"x", 29}, {"y", 25}, {"z", 2}})["label"]["id"] == 3);
    // erase its centre column through the three planes: only that label goes, and what is left is
    // one connected ring (the planes above and below hold only the centre, which goes with it)
    const json erased = f.ok("paint_label", {{"x", 25}, {"y", 25}, {"z", 2}, {"radius", 1}, {"z_radius", 1}, {"erase", true}, {"label", 3}});
    CHECK(erased["voxels"].get<Index>() > 0);
    CHECK(f.ok("probe", {{"step", 2}, {"x", 25}, {"y", 25}, {"z", 2}})["label"].is_null());
    CHECK(f.ok("list_labels")["count"] == 3);
    // fill the rest of label 3 with label 1: 3 is gone
    const json filled = f.ok("fill_label", {{"x", 29}, {"y", 25}, {"z", 2}, {"label", 1}});
    CHECK(filled["label"] == 1);
    CHECK(filled["voxels"].get<Index>() > 0);
    CHECK(f.ok("list_labels")["count"] == 2);
    // split the bar from its two ends: a new label 4
    const json split = f.ok("split_label", {{"label", 2}, {"a", {3, 11, 2}}, {"b", {20, 11, 2}}});
    CHECK(split["created"] == 4);
    CHECK(f.ok("list_labels")["count"] == 3);
    CHECK(f.ok("probe", {{"step", 2}, {"x", 3}, {"y", 11}, {"z", 2}})["label"]["id"] == 2);
    CHECK(f.ok("probe", {{"step", 2}, {"x", 20}, {"y", 11}, {"z", 2}})["label"]["id"] == 4);
    // merge them back: into the smaller id
    const json merged = f.ok("merge_labels", {{"ids", {2, 4}}});
    CHECK(merged["into"] == 2);
    CHECK(merged["voxels"] == 90);   // the voxels renumbered: the half that was label 4
    CHECK(f.ok("list_labels")["count"] == 2);
    CHECK(f.ok("probe", {{"step", 2}, {"x", 20}, {"y", 11}, {"z", 2}})["label"]["id"] == 2);
    // review mark, delete, and the errors for ids that are not there
    CHECK(f.ok("set_label_reviewed", {{"label", 2}})["reviewed"] == true);
    CHECK(f.ok("list_labels", {{"unreviewed", true}})["listed"] == 1);
    CHECK(f.call("delete_label", {{"label", 9}}).error.code == "not_found");
    CHECK(f.call("merge_labels", {{"ids", {7, 8}}}).error.code == "not_found");
    CHECK(f.call("split_label", {{"label", 2}, {"a", {3, 11, 2}}, {"b", {3, 11, 2}}}).error.code == "not_split");
    CHECK(f.call("paint_label", {{"x", nx + 5}, {"y", 1}, {"z", 1}}).error.code == "invalid_argument");
    const json deleted = f.ok("delete_label", {{"label", 1}});
    CHECK(deleted["voxels"].get<Index>() > 75);   // the cube plus what the fill added
    CHECK(f.ok("list_labels")["count"] == 1);

    // save, clear, undo, and read the saved file back as another step
    const std::filesystem::path saved = f.scratch.path / "edited";   // .tif is appended
    const json exported = f.ok("export_labels", {{"path", saved.u8string()}});
    CHECK(exported["path"] == saved.u8string() + ".tif");
    CHECK(exported["pages"] == nz);
    CHECK(exported["labels"] == 1);
    REQUIRE(std::filesystem::exists(saved.u8string() + ".tif"));
    const json cleared = f.ok("clear_labels");
    CHECK(cleared["voxels"] == 180);
    CHECK(f.ok("list_labels")["count"] == 0);
    CHECK(f.ok("undo")["ok"] == true);   // the clear is one undo entry
    CHECK(f.ok("list_labels")["count"] == 1);
    CHECK(f.ok("probe", {{"step", 2}, {"x", 20}, {"y", 11}, {"z", 2}})["label"]["id"] == 2);

    f.ok("add_step", {{"kind", "import_labels"}, {"params", {{"path", saved.u8string() + ".tif"}}}, {"name", "Reloaded"}});
    const agent::ToolResult again = f.call("run", {{"wait_s", -1}});
    INFO(again.error.code << ": " << again.error.message);
    REQUIRE(again.ok);
    const json reloaded = f.ok("list_labels", {{"step", 3}});
    CHECK(reloaded["count"] == 1);
    CHECK(reloaded["labels"][0]["id"] == 2);
    CHECK(reloaded["labels"][0]["voxels"] == 180);
    // the import step refuses a file that is not on the grid, before running
    std::ofstream(f.scratch.path / "not a tif.tif") << "nothing";
    f.ok("set_params", {{"step", 3}, {"params", {{"path", (f.scratch.path / "not a tif.tif").u8string()}}}});
    const json bad = f.ok("get_step", {{"step", 3}});
    CHECK_FALSE(bad["errors"].empty());
}

TEST_CASE("headless: the bundled example pipeline validates with plugins off", "[app][headless]") {
    Fixture f;
    const agent::ToolResult r = f.call("load_pipeline", {{"path", (kExamples / "sim_bundled.sirius.toml").u8string()}});
    INFO(r.error.code << ": " << r.error.message);
    REQUIRE(r.ok);
    CHECK(r.value["workspace"] == f.h->workspaceId());
    CHECK(r.value["steps"].size() == 4);
    CHECK(r.value["missing_kinds"].empty());
    CHECK(r.value["plugins_loaded"] == false);
    REQUIRE(r.value["dataset"].is_object());
    CHECK(r.value["dataset"]["dims"]["z"].get<long long>() == 135);
    CHECK_THAT(r.value["pipeline_path"].get<std::string>(), ContainsSubstring("sim_bundled.sirius.toml"));

    const json v = f.ok("validate");
    INFO(v.dump(2));
    CHECK(v["ok"] == true);
    CHECK(v["has_dataset"] == true);
    CHECK(v["needs_worker"] == false);
    REQUIRE(v["steps"].size() == 4);
    CHECK(v["steps"][1]["kind"] == "sim");
    CHECK(v["steps"][1]["errors"].empty());
    CHECK(v["steps"][3]["enabled"] == false);

    SECTION("save_pipeline writes it back, clear_pipeline empties it") {
        const std::filesystem::path saved = f.scratch.path / "saved.sirius.toml";
        f.ok("save_pipeline", {{"path", saved.u8string()}});
        CHECK(std::filesystem::exists(saved));
        CHECK(f.ok("clear_pipeline")["steps"].size() == 1);
        CHECK(f.ok("get_state")["can_undo"] == true);
    }
}

TEST_CASE("headless: load_pipeline opens a dataset with the pipeline's own Load options", "[app][headless]") {
    Fixture f;
    // a pipeline from another computer: its dataset is not here, its voxel size is its own
    const std::filesystem::path pipeline = f.scratch.path / "elsewhere.sirius.toml";
    std::ofstream(pipeline) << "version = 1\n\n[[steps]]\nkind = \"load\"\nname = \"Load\"\n[steps.params]\n"
                               "path = \"no-such-dataset.tif\"\nread_as = \"Lazy (chunk on demand)\"\n"
                               "voxel_x = 0.5\nvoxel_y = 0.5\nvoxel_z = 2.0\n\n"
                               "[[steps]]\nkind = \"contrast\"\nname = \"Contrast\"\n";
    // and the workspace has another dataset open, with options of its own
    f.ok("open_dataset", {{"path", (kData / "raw.tif").u8string()}, {"voxel_um", {0.25, 0.25, 0.75}}});

    SECTION("with dataset: that one, opened the pipeline's way") {
        const agent::ToolResult r = f.call("load_pipeline", {{"path", pipeline.u8string()}, {"dataset", (kData / "raw.tif").u8string()}});
        INFO(r.error.code << ": " << r.error.message);
        REQUIRE(r.ok);
        CHECK(r.value["steps"].size() == 2);
        REQUIRE(r.value["dataset"].is_object());
        CHECK(r.value["dataset"]["voxel_um"] == json::array({0.5, 0.5, 2.0}));
        CHECK(r.value["steps"][0]["params"]["voxel_z"] == 2.0);
        CHECK_THAT(r.value["steps"][0]["params"]["path"].get<std::string>(), ContainsSubstring("raw.tif"));
        CHECK_FALSE(r.undoable);   // an opened dataset starts a new history
    }
    SECTION("without: the reply says the pipeline's dataset was not opened") {
        const agent::ToolResult r = f.call("load_pipeline", {{"path", pipeline.u8string()}});
        REQUIRE(r.ok);
        bool warned = false;
        for (const std::string& w : r.warnings) warned = warned || (w.find("no-such-dataset.tif") != std::string::npos && w.find("could not be opened") != std::string::npos);
        CHECK(warned);
        CHECK(r.undoable);   // nothing was opened: the load is one change to undo
        REQUIRE(r.value["dataset"].is_object());
        CHECK(r.value["dataset"]["voxel_um"] == json::array({0.25, 0.25, 0.75}));
    }
    SECTION("its own dataset, already open the same way, is not reported missing") {
        const std::filesystem::path own = f.scratch.path / "own.sirius.toml";
        f.ok("save_pipeline", {{"path", own.u8string()}});
        const agent::ToolResult r = f.call("load_pipeline", {{"path", own.u8string()}});
        REQUIRE(r.ok);
        for (const std::string& w : r.warnings) CHECK(w.find("could not be opened") == std::string::npos);
    }
}

TEST_CASE("headless: a run that goes on is polled to its end", "[app][headless]") {
    Fixture f;
    f.openRaw();
    f.ok("add_step", {{"kind", "test_headless_slow"}, {"params", {{"ms", 400}}}});
    const json started = f.ok("run", {{"wait_s", 0}});
    CHECK(started["status"] == "running");
    CHECK(started["run_id"] == "r1");
    CHECK(f.h->status().running);
    CHECK(f.h->status().runId == "r1");

    SECTION("run_status waits for it") {
        const json done = f.ok("run_status", {{"wait_s", -1}});
        CHECK(done["status"] == "succeeded");
        CHECK(done["run_id"] == "r1");
        bool finishedEvent = false;
        for (const json& e : f.h->takeEvents()) finishedEvent = finishedEvent || (e["event"] == "run_finished" && e["run_id"] == "r1");
        CHECK(finishedEvent);
        CHECK_FALSE(f.h->status().running);
    }
    SECTION("a call once it ended is not busy: the run is folded back first") {
        // wait for the job to end without letting the headless workbench look
        const std::shared_ptr<RunJob> job = f.h->workbench().activeRun();
        REQUIRE(job);
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!job->finished() && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        REQUIRE(job->finished());
        CHECK(f.h->workbench().running());   // not folded yet
        const agent::ToolResult edit = f.call("set_params", {{"step", 2}, {"params", {{"ms", 10}}}});
        INFO(edit.error.code << ": " << edit.error.message);
        CHECK(edit.ok);
        CHECK(f.ok("run_status")["status"] == "succeeded");
    }
}

TEST_CASE("headless: the tools that would change the workspace are busy during a run", "[app][headless]") {
    Fixture f;
    f.openRaw();
    f.ok("add_step", {{"kind", "test_headless_slow"}, {"params", {{"ms", 20000}}}});
    CHECK(f.ok("run", {{"wait_s", 0}})["status"] == "running");
    CHECK(f.call("render").error.code == "busy");
    CHECK(f.call("set_params", {{"step", 2}, {"params", {{"ms", 1}}}}).error.code == "busy");
    CHECK(f.call("run").error.code == "busy");
    // the read-only ones answer
    const json state = f.ok("get_state");
    CHECK(state["running"] == true);
    CHECK(state["run"]["status"] == "running");
    CHECK(f.ok("get_step", {{"step", 2}})["kind"] == "test_headless_slow");

    const auto t0 = std::chrono::steady_clock::now();
    const json cancelled = f.ok("cancel_run");
    CHECK(cancelled["cancelled"] == true);
    const json after = f.ok("run_status", {{"wait_s", -1}});
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5));
    CHECK(after["status"] == "cancelled");
    CHECK(f.call("render").ok);
}

TEST_CASE("headless: a call for another workspace is refused", "[app][headless]") {
    Fixture f;
    const agent::ToolResult r = f.call("get_state", {{"workspace", "ws_000000000000"}});
    CHECK(r.error.code == "stale_workspace");
    CHECK(r.error.data["workspace"] == f.h->workspaceId());
    CHECK(f.call("get_state", {{"workspace", f.h->workspaceId()}}).ok);
    // a null workspace is one not given, as clients send optional fields
    CHECK(f.call("get_state", {{"workspace", nullptr}}).ok);
    CHECK(f.call("get_state", {{"workspace", 7}}).error.code == "stale_workspace");
}

TEST_CASE("headless: the hpc backend takes only the endpoint given at start", "[app][headless]") {
    Fixture f;
    const agent::ToolResult r = f.call("set_backend", {{"backend", "hpc"}});
    CHECK(r.error.code == "invalid_argument");
    CHECK_THAT(r.error.hint, ContainsSubstring("--hpc"));
    CHECK(f.ok("set_backend", {{"backend", "CPU"}})["backend"] == "cpu");

    Fixture withHpc([](HeadlessOptions& o) { o.hpc = RemoteConfig{"cluster.example", 7645, "secret"}; });
    CHECK(withHpc.ok("set_backend", {{"backend", "hpc"}})["backend"] == "hpc");
    CHECK(withHpc.ok("get_state")["hpc_configured"] == true);
    CHECK_THROWS_AS(Fixture([](HeadlessOptions& o) { o.backend = "hpc"; }), ToolFailure);

    // where the HPC worker computes: the job's GPU (the default) or its CPU, kept until changed
    CHECK(withHpc.ok("get_state")["hpc_device"] == "gpu");
    const json set = withHpc.ok("set_backend", {{"backend", "hpc"}, {"hpc_device", "CPU"}});
    CHECK(set["hpc_device"] == "cpu");
    CHECK(withHpc.ok("get_state")["hpc_device"] == "cpu");
    CHECK(withHpc.ok("set_backend", {{"backend", "hpc"}})["hpc_device"] == "cpu");
    CHECK(withHpc.call("set_backend", {{"backend", "hpc"}, {"hpc_device", "tpu"}}).error.code == "invalid_argument");
    Fixture cpuJob([](HeadlessOptions& o) {
        o.hpc = RemoteConfig{"cluster.example", 7645, "secret"};
        o.hpcDevice = "cpu";
    });
    CHECK(cpuJob.ok("get_state")["hpc_device"] == "cpu");
    CHECK_THROWS_AS(Fixture([](HeadlessOptions& o) { o.hpcDevice = "tpu"; }), ToolFailure);
}

TEST_CASE("headless: an HPC endpoint that does not answer is a failed run, not a missing Python", "[app][headless]") {
    // nothing listens on port 1 of this computer
    Fixture f([](HeadlessOptions& o) {
        o.backend = "hpc";
        o.hpc = RemoteConfig{"127.0.0.1", 1, "token"};
    });
    f.openRaw();
    f.ok("add_step", {{"kind", "contrast"}});
    const agent::ToolResult r = f.call("run", {{"wait_s", -1}});
    INFO(r.error.code << ": " << r.error.message << " / " << r.error.hint);
    REQUIRE_FALSE(r.ok);
    CHECK(r.error.code == "run_failed");
    CHECK(r.error.data["status"] == "failed");
    CHECK(r.error.data["backend"] == "hpc");
    CHECK_FALSE(r.error.data.contains("worker_error"));
    // what to look at is the endpoint, not a Python environment to download
    CHECK_THAT(r.error.hint, ContainsSubstring("127.0.0.1:1"));
    CHECK_THAT(r.error.hint, !ContainsSubstring("worker setup"));
    CHECK_THAT(r.error.message, ContainsSubstring("127.0.0.1:1"));
    CHECK_THAT(r.error.message, !ContainsSubstring("worker check"));
}

TEST_CASE("headless: setting up the worker environment needs consent and known packages", "[app][headless]") {
    Fixture f;
    const agent::ToolResult refused = f.call("setup_worker_env", {{"confirm", true}});
    CHECK(refused.error.code == "consent_required");
    CHECK_THAT(refused.error.hint, ContainsSubstring("--allow-worker-setup"));

    Fixture allowed([](HeadlessOptions& o) { o.allowWorkerSetup = true; });
    const agent::ToolResult unknown = allowed.call("setup_worker_env", {{"confirm", true}, {"packages", {"requests"}}});
    CHECK(unknown.error.code == "invalid_argument");
    CHECK(unknown.error.data["allowed"].is_array());
    const agent::ToolResult base = allowed.call("setup_worker_env", {{"confirm", true}, {"base_python", "C:/not/a/python.exe"}});
    CHECK(base.error.code == "invalid_argument");
    // nothing was created in the environment directory those calls would set up
    // (SIRIUS_PYTHON_ENV is the second fixture's while it lives), nor in the first's
    CHECK_FALSE(std::filesystem::exists(allowed.pyenv.path / "env"));
    CHECK_FALSE(std::filesystem::exists(f.pyenv.path / "env"));
}

TEST_CASE("headless: worker_status reports the interpreter and SIRIUS's own environment", "[app][headless]") {
    Fixture f([](HeadlessOptions& o) { o.workerDir = SIRIUS_TEST_WORKER_DIR; });
    const json s = f.ok("worker_status");
    CHECK(s["interpreter"]["source"].is_string());
    // the case's own environment directory, which nothing has set up
    CHECK(s["environment"]["state"] == "absent");
    const json required = s["requirements"]["required"];
    CHECK(std::find(required.begin(), required.end(), json("numpy")) != required.end());
    CHECK(s["candidates"].is_array());
    CHECK(s["running"] == false);
    CHECK_THAT(s["worker_dir"].get<std::string>(), ContainsSubstring("app/python"));
    // with plugins off nothing starts the worker to list them
    CHECK(f.call("list_plugins").error.code == "unsupported");
}

TEST_CASE("headless: get_help refuses page names that are paths", "[app][headless]") {
    Fixture f;
    CHECK(f.call("get_help", {{"page", "../../x"}}).error.code == "invalid_argument");
    CHECK(f.call("get_help", {{"kind", "../x"}}).error.code == "invalid_argument");
    CHECK(f.call("get_help", {{"page", "no_such_page_at_all"}}).error.code == "not_found");
    const json load = f.ok("get_help", {{"kind", "load"}});
    CHECK_THAT(load["markdown"].get<std::string>(), ContainsSubstring("#"));
    CHECK(load["exists"] == true);
    CHECK(load["truncated"] == false);
    CHECK_FALSE(f.ok("get_help")["pages"].empty());
    const json op = f.ok("describe_operation", {{"kind", "contrast"}});
    CHECK(op["kind"] == "contrast");
    CHECK_FALSE(op["params"].empty());
    CHECK(f.call("describe_operation", {{"kind", "nope"}}).error.code == "unknown_operation");
    const json ops = f.ok("list_operations");
    CHECK(ops["operations"].size() > 5);
    for (const json& o : ops["operations"]) CHECK(o["kind"] != "load");
}

TEST_CASE("headless: a step that needs the worker fails as no_interpreter without Python", "[app][headless]") {
    TempDir nowhere("headless_nopython");
    Fixture f([&](HeadlessOptions& o) {
        o.python = (nowhere.path / "no-such-python.exe").u8string();
        o.workerDir = SIRIUS_TEST_WORKER_DIR;
    });
    f.openRaw();
    // a foundation model runs only in the worker; a model folder that is not on
    // this machine is a warning, not an error, so the run gets as far as the worker
    f.ok("add_step", {{"kind", "foundation"}, {"params", {{"model", "models/not-here/v1"}}}});
    const agent::ToolResult r = f.call("run", {{"wait_s", -1}});
    INFO(r.error.message);
    REQUIRE_FALSE(r.ok);
    CHECK(r.error.code == "worker_unavailable");
    CHECK(r.error.data["kind"] == "no_interpreter");
    CHECK(r.error.data["status"] == "failed");
    CHECK_FALSE(r.error.hint.empty());
}

TEST_CASE("headless: cancel_run ends a slow worker start", "[app][headless]") {
    const std::string python = anyPython();
    if (python.empty()) SKIP("no Python on this machine for a fake worker");
    // a worker that never prints its port line
    TempDir fake("headless_fakeworker");
    std::filesystem::create_directories(fake.path / "sirius_worker");
    std::ofstream(fake.path / "sirius_worker" / "__init__.py") << "";
    std::ofstream(fake.path / "sirius_worker" / "__main__.py") << "import time\ntime.sleep(60)\n";
    Fixture f([&](HeadlessOptions& o) {
        o.python = python;
        o.workerDir = fake.path.u8string();
    });
    f.openRaw();
    // a foundation model runs only in the worker; a model folder that is not on
    // this machine is a warning, not an error, so the run gets as far as the worker
    f.ok("add_step", {{"kind", "foundation"}, {"params", {{"model", "models/not-here/v1"}}}});
    CHECK(f.ok("run", {{"wait_s", 0.3}})["status"] == "running");
    const auto t0 = std::chrono::steady_clock::now();
    f.ok("cancel_run");
    const json after = f.ok("run_status", {{"wait_s", -1}});
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2));
    CHECK(after["status"] == "cancelled");
}

TEST_CASE("headless: a cancelled call ends the worker start of its plugin load", "[app][headless]") {
    const std::string python = anyPython();
    if (python.empty()) SKIP("no Python on this machine for a fake worker");
    // a worker that never prints its port line
    TempDir fake("headless_fakeplugins");
    std::filesystem::create_directories(fake.path / "sirius_worker");
    std::ofstream(fake.path / "sirius_worker" / "__init__.py") << "";
    std::ofstream(fake.path / "sirius_worker" / "__main__.py") << "import time\ntime.sleep(60)\n";
    Fixture f([&](HeadlessOptions& o) {
        o.python = python;
        o.workerDir = fake.path.u8string();
        o.plugins = HeadlessOptions::Plugins::Auto;
    });
    // the request's own cancel (MCP notifications/cancelled), not cancelActive()
    std::thread canceller([&f] {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        f.cancel = true;
    });
    const auto t0 = std::chrono::steady_clock::now();
    const agent::ToolResult r = f.call("list_plugins");
    canceller.join();
    CHECK(r.error.code == "cancelled");
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5));
    // the next call tries again, and is not refused as cancelled
    f.cancel = false;
    CHECK(f.call("describe_operation", {{"kind", "contrast"}}).ok);
    // describe_operation is not a plugin trigger (D20): a mistyped kind does not wait for Python
    const auto t1 = std::chrono::steady_clock::now();
    CHECK(f.call("describe_operation", {{"kind", "no_such_plugin_kind"}}).error.code == "unknown_operation");
    CHECK(std::chrono::steady_clock::now() - t1 < std::chrono::seconds(2));
}

TEST_CASE("headless: a failed plugin load is not tried again until something changes", "[app][headless]") {
    const std::string python = anyPython();
    if (python.empty()) SKIP("no Python on this machine for a fake worker");
    // a worker that notes each start and then fails
    TempDir fake("headless_failplugins");
    const std::filesystem::path starts = fake.path / "starts.txt";
    std::filesystem::create_directories(fake.path / "sirius_worker");
    std::ofstream(fake.path / "sirius_worker" / "__init__.py") << "";
    std::ofstream(fake.path / "sirius_worker" / "__main__.py")
        << "import sys\nopen(r'" << starts.generic_string() << "', 'a').write('x\\n')\nsys.exit(3)\n";
    Fixture f([&](HeadlessOptions& o) {
        o.python = python;
        o.workerDir = fake.path.u8string();
        o.plugins = HeadlessOptions::Plugins::Auto;
    });
    const auto startCount = [&starts] {
        std::ifstream in(starts);
        std::string line;
        int n = 0;
        while (std::getline(in, line)) ++n;
        return n;
    };
    CHECK(f.call("list_plugins").error.code == "worker_unavailable");
    const int first = startCount();
    CHECK(first >= 1);
    CHECK(f.call("add_step", {{"kind", "no_such_plugin_kind"}}).error.code == "unknown_operation");
    CHECK(f.call("list_operations", {{"include_plugins", true}}).ok);
    CHECK(startCount() == first);
    // a reload asks again, and so does a change of backend
    CHECK(f.call("list_plugins", {{"reload", true}}).error.code == "worker_unavailable");
    const int second = startCount();
    CHECK(second > first);
    f.ok("set_backend", {{"backend", "cpu"}});
    CHECK(f.call("add_step", {{"kind", "no_such_plugin_kind"}}).error.code == "unknown_operation");
    CHECK(startCount() > second);
}

TEST_CASE("headless: log lines reach the sink and the events", "[app][headless]") {
    std::vector<std::pair<std::string, std::string>> sunk;
    Fixture f([&](HeadlessOptions& o) {
        o.logSink = [&](const std::string& source, const std::string& line) { sunk.emplace_back(source, line); };
    });
    f.h->takeEvents();
    f.openRaw();
    bool opened = false;
    for (const auto& [source, line] : sunk) opened = opened || (source == "workbench" && line.find("Opened") != std::string::npos);
    CHECK(opened);
    bool event = false;
    for (const json& e : f.h->takeEvents())
        event = event || (e["event"] == "log" && e["source"] == "workbench" && e["line"].get<std::string>().find("Opened") != std::string::npos);
    CHECK(event);
    CHECK(f.h->takeEvents().empty());
    const json log = f.ok("get_log", {{"lines", 5}});
    CHECK_FALSE(log["lines"].empty());
    CHECK(log["lines"].size() <= 5);
}

TEST_CASE("headless: a diagnostics image is drawn with its marks", "[app][headless]") {
    DiagnosticImage img;
    img.title = "ramp";
    img.rows = 40;
    img.cols = 60;
    for (Index i = 0; i < img.rows * img.cols; ++i) img.values.push_back(static_cast<float>(i % img.cols));
    img.marks.push_back({DiagnosticMark::Kind::Circle, 30.0, 20.0, 6.0, true, "k0"});
    const RenderResult r = renderDiagnosticImage(img, 768);
    REQUIRE(isPng(r.bytes));
    CHECK(r.bytes[25] == 2);   // IHDR colour type: RGB, for the accent
    CHECK(r.width == 60);
    CHECK(r.height == 40);
    CHECK(r.caption["marks"].size() == 1);
    CHECK(r.caption["marks"][0]["kind"] == "circle");
    const RenderResult small = renderDiagnosticImage(img, 20);
    CHECK(small.width <= 20);
    CHECK(small.factor == 3);
    img.values.clear();
    CHECK_THROWS_AS(renderDiagnosticImage(img, 768), ToolFailure);

    SECTION("noise over the byte budget is reduced, then refused") {
        DiagnosticImage noise;
        noise.title = "noise";
        noise.rows = 256;
        noise.cols = 256;
        std::uint32_t state = 12345;
        for (Index i = 0; i < noise.rows * noise.cols; ++i) {
            state = state * 1664525u + 1013904223u;
            noise.values.push_back(static_cast<float>(state >> 8));
        }
        const RenderResult native = renderDiagnosticImage(noise, 1568);
        REQUIRE(isPng(native.bytes));
        CHECK(native.bytes[25] == 0);   // grey: nothing is drawn in colour
        CHECK(native.factor == 1);
        const RenderResult reduced = renderDiagnosticImage(noise, 1568, native.bytes.size() / 2);
        CHECK(reduced.factor > 1);
        CHECK(reduced.bytes.size() <= native.bytes.size() / 2);
        CHECK(reduced.caption["factor"] == reduced.factor);
        try {
            renderDiagnosticImage(noise, 1568, 64);
            FAIL("an image over any size's budget was not refused");
        } catch (const ToolFailure& e) {
            CHECK(e.code() == "too_large");
        }
    }
}

TEST_CASE("headless: the option parsers refuse what they cannot honour", "[app][headless]") {
    CHECK_THROWS_AS(openOptionsFromJson({{"page_order", "czz"}}), ToolFailure);
    CHECK_THROWS_AS(openOptionsFromJson({{"sim", "yes"}}), ToolFailure);
    const OpenOptions o = openOptionsFromJson({{"page_order", "ZCT"}, {"z", 9}, {"c", 1}, {"sim", {{"ndirs", 3}, {"nphases", 5}}}});
    REQUIRE(o.pageOrder);
    CHECK(o.pageOrder->order == "zct");
    CHECK(o.pageOrder->z == 9);
    REQUIRE(o.sim);
    CHECK(o.sim->nphases == 5);
    CHECK_FALSE(o.readAll);

    DatasetMeta meta;
    meta.dims = Dims5{2, 1, 4, 8, 8};
    const ExportOptions e = exportOptionsFromJson({{"path", "out.zarr"}, {"channels", {1}}, {"percentiles", {1, 99}}}, meta);
    CHECK(e.format == ExportFormat::Zarr);
    CHECK(e.scaling == ExportScaling::Percentile);
    CHECK(std::filesystem::path(e.path).is_absolute());
    CHECK_THROWS_AS(exportOptionsFromJson({{"path", "out.zarr"}, {"channels", {2}}}, meta), ToolFailure);
    CHECK_THROWS_AS(exportOptionsFromJson({{"path", "out.tif"}, {"dtype", "float16"}}, meta), ToolFailure);
    CHECK(exportOptionsFromJson({{"path", "out.tif"}}, meta).tiff.omeXml == false);
    CHECK(exportOptionsFromJson({{"path", "out.ome.tif"}}, meta).tiff.omeXml == true);

    // labels_only writes a TIFF, whatever the path's extension says
    StepOutput labelled;
    labelled.meta = meta;
    labelled.array = std::make_shared<Array5>(meta.dims);
    labelled.labels = std::make_shared<LabelVolume>(1, 4, 8, 8);
    CHECK_THROWS_AS(exportStepOutput(std::make_shared<const StepOutput>(labelled), Pipeline{},
                                     exportOptionsFromJson({{"path", "labels.zarr"}}, meta), true, {}, {}),
                    ToolFailure);

    const RenderRequest r = renderRequestFromJson({{"plane", "MIP"}, {"max_size", 5000}, {"format", "jpg"}});
    CHECK(r.plane == "mip");
    CHECK(r.maxSize == 1568);
    CHECK(r.format == "jpeg");
    CHECK_THROWS_AS(renderRequestFromJson({{"plane", "xz"}, {"z", {1, 2}}}), ToolFailure);
    CHECK_THROWS_AS(renderRequestFromJson({{"windows", {{{"channel", 0}, {"lo", 5}, {"hi", 1}}}}}), ToolFailure);
}

TEST_CASE("headless: tools refuse network paths unless the server allows them", "[app][headless][security]") {
    CHECK(isNetworkPath(R"(\\server\share\a.tif)"));
    CHECK(isNetworkPath("//server/share/a.tif"));
    CHECK(isNetworkPath(R"(\\?\UNC\server\share\a.tif)"));
    CHECK(isNetworkPath(R"(\\.\pipe\x)"));
    CHECK(isNetworkPath(R"(/\server\share)"));
    CHECK_FALSE(isNetworkPath(R"(\\?\C:\data\a.tif)"));
    CHECK_FALSE(isNetworkPath(R"(C:\data\a.tif)"));
    CHECK_FALSE(isNetworkPath("/data/a.tif"));
    CHECK_FALSE(isNetworkPath("a.tif"));
    CHECK_FALSE(isNetworkPath(""));

    Fixture f;
    const std::string unc = R"(\\attacker.example\share\a.tif)";
    for (const auto& [tool, args] : std::vector<std::pair<std::string, json>>{
             {"open_dataset", {{"path", unc}}},
             {"dataset_info", {{"path", "//attacker.example/share/a.tif"}}},
             {"load_pipeline", {{"path", unc}}},
             {"save_pipeline", {{"path", unc}}},
             {"export_python", {{"path", R"(\\?\UNC\attacker.example\share\x.py)"}}},
             {"export_result", {{"path", unc}}},
             {"export_training_data", {{"directory", unc}}},
             {"add_step", {{"kind", "flatfield"}, {"params", {{"flat", unc}}}}}}) {
        INFO(tool);
        const agent::ToolResult r = f.call(tool, args);
        CHECK_FALSE(r.ok);
        CHECK(r.error.code == "invalid_argument");
        CHECK_THAT(r.error.message, ContainsSubstring("network path"));
    }
    // a pipeline file that names one is refused before anything is opened
    TempDir dir("headless_uncpipe");
    const std::string pipeFile = (dir.path / "p.sirius.toml").string();
    std::ofstream(pipeFile, std::ios::binary) << "version = 1\n\n[[steps]]\nkind = \"load\"\nname = \"Load\"\nenabled = true\n"
                                                 "cache = \"recompute\"\n[steps.params]\npath = '//attacker.example/share/a.tif'\n";
    const agent::ToolResult loaded = f.call("load_pipeline", {{"path", pipeFile}});
    CHECK_FALSE(loaded.ok);
    CHECK_THAT(loaded.error.message, ContainsSubstring("network path"));

    // --allow-network-paths reaches the shared tools (nothing is opened here:
    // a test never names a host)
    CHECK_FALSE(f.h->toolApi().allowNetworkPaths());
    Fixture allowed([](HeadlessOptions& o) { o.allowNetworkPaths = true; });
    CHECK(allowed.h->toolApi().allowNetworkPaths());
}

TEST_CASE("headless: an exported Python script runs a hostile step name as text", "[app][headless][security]") {
    const std::string python = anyPython();
    if (python.empty()) SKIP("no Python on this machine");
    Fixture f;
    const std::string injection = "x''' + str(print('INJECTED')) + r'''";
    f.ok("add_step", {{"kind", "contrast"}, {"name", injection}});
    const std::string script = f.ok("export_python")["script"].get<std::string>();
    TempDir dir("headless_pyscript");
    const std::filesystem::path scriptFile = dir.path / "exported.py", driver = dir.path / "driver.py", out = dir.path / "out.txt";
    std::ofstream(scriptFile, std::ios::binary) << script;
    // The script's imports are stand-ins, and it runs as a module (not
    // __main__), so only its top level executes: the literals.
    std::ofstream(driver, std::ios::binary)
        << "import json, sys, types\n"
           "for name in ('numpy', 'sirius', 'sirius.workbench'):\n"
           "    sys.modules[name] = types.ModuleType(name)\n"
           "sys.modules['sirius'].workbench = sys.modules['sirius.workbench']\n"
           "sys.modules['sirius.workbench'].run_pipeline = None\n"
           "src = open(sys.argv[1], encoding='utf-8').read()\n"
           "ns = {'__name__': 'exported'}\n"
           "exec(compile(src, sys.argv[1], 'exec'), ns)\n"
           "print('NAME=' + json.dumps(ns['PIPELINE']['steps'][1]['name']))\n";
#ifdef _WIN32
    const std::string cmd = "\"\"" + python + "\" \"" + driver.string() + "\" \"" + scriptFile.string() + "\" > \"" + out.string() + "\" 2>&1\"";
#else
    const std::string cmd = "'" + python + "' '" + driver.string() + "' '" + scriptFile.string() + "' > '" + out.string() + "' 2>&1";
#endif
    const int rc = std::system(cmd.c_str());
    std::ifstream in(out);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    INFO(text);
    CHECK(rc == 0);
    // print() would have put INJECTED on a line of its own
    std::istringstream lines(text);
    for (std::string line; std::getline(lines, line);) CHECK(line.rfind("INJECTED", 0) == std::string::npos);
    CHECK(text.find("NAME=" + json(injection).dump()) != std::string::npos);
}
