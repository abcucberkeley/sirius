// The tool table's busy gate and error kinds (app/core/tool_api.hpp), and
// what the workbench keeps for its host when a run or a plugin load cannot
// get the Python worker, or is refused: RunRefusal, RunJob::workerFailure,
// pluginError and the host's hint (app/core/workbench.hpp).
//
// The operations are this file's own, under kinds of their own: sirius_tests
// links every test file into one binary, and the registry holds one
// operation per kind.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/array_source.hpp"
#include "core/cancel.hpp"
#include "core/operation.hpp"
#include "core/pipeline.hpp"
#include "core/tool_api.hpp"
#include "core/workbench.hpp"
#include "core/worker_error.hpp"

#include "temp_path.hpp"

using namespace sirius;
using namespace sirius::app;
using json = nlohmann::json;
using Catch::Matchers::ContainsSubstring;

namespace {

    // Holds its run until the test lets it go, or the run is cancelled: the
    // tools are called against the workbench while it runs. The deadline
    // only keeps a failed test from hanging.
    struct GateSlowOp final : Operation {
        static inline std::atomic<bool> release{false};
        static inline std::atomic<bool> started{false};
        OpInfo info_;
        GateSlowOp() {
            info_.kind = "test_gate_slow";
            info_.name = "Gate slow";
            info_.group = "Intensity";
            info_.kindLabel = "INTENSITY";
            info_.params = {intParam("ticks", "Ticks", 1).range(0, 100)};
        }
        const OpInfo& info() const noexcept override { return info_; }
        StepOutput run(const StepInput& in, const ParamSet&, const StepContext& ctx) const override {
            started = true;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (!release.load() && std::chrono::steady_clock::now() < deadline) {
                ctx.throwIfCancelled();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            StepOutput o;
            o.meta = in.meta;
            o.array = in.materialize();
            return o;
        }
    };

    // Wants the Python worker, so a run asks the launcher for one, and passes
    // its input on without calling it.
    struct GateWorkerOp final : Operation {
        OpInfo info_;
        GateWorkerOp() {
            info_.kind = "test_gate_worker";
            info_.name = "Gate worker";
            info_.group = "Intensity";
            info_.kindLabel = "INTENSITY";
            info_.remoteCapable = true;
        }
        const OpInfo& info() const noexcept override { return info_; }
        StepOutput run(const StepInput& in, const ParamSet&, const StepContext&) const override {
            StepOutput o;
            o.meta = in.meta;
            o.array = in.materialize();
            return o;
        }
    };

    // Never valid: what createRun refuses as Invalid.
    struct GateInvalidOp final : Operation {
        OpInfo info_;
        GateInvalidOp() {
            info_.kind = "test_gate_invalid";
            info_.name = "Gate invalid";
            info_.group = "Intensity";
            info_.kindLabel = "INTENSITY";
        }
        const OpInfo& info() const noexcept override { return info_; }
        Validation validate(const ParamSet&, const DatasetMeta&) const override {
            Validation v;
            v.errors.push_back("this step never runs");
            return v;
        }
        StepOutput run(const StepInput&, const ParamSet&, const StepContext&) const override {
            throw std::logic_error("an invalid step ran");
        }
    };

    // A plugin's kind may have a '.' in it, which the help tool accepts
    // for a kind that is registered.
    struct GateDottedOp final : Operation {
        OpInfo info_;
        GateDottedOp() {
            info_.kind = "test_gate.dotted";
            info_.name = "Gate dotted";
            info_.group = "Intensity";
            info_.kindLabel = "INTENSITY";
        }
        const OpInfo& info() const noexcept override { return info_; }
        StepOutput run(const StepInput& in, const ParamSet&, const StepContext&) const override {
            StepOutput o;
            o.meta = in.meta;
            o.array = in.materialize();
            return o;
        }
    };

    void registerGateOps() {
        static bool done = false;
        if (done) return;
        done = true;
        registerOperation(std::make_unique<GateSlowOp>());
        registerOperation(std::make_unique<GateWorkerOp>());
        registerOperation(std::make_unique<GateInvalidOp>());
        registerOperation(std::make_unique<GateDottedOp>());
    }

    std::shared_ptr<MemorySource> smallSource() {
        auto a = std::make_shared<Array5>(Dims5{1, 1, 2, 4, 4});
        for (Index i = 0; i < a->numel(); ++i) a->data()[i] = static_cast<float>(i);
        DatasetMeta m;
        m.name = "gate";
        m.sourcePath = "memory://gate";
        m.format = "memory";
        m.dims = a->dims();
        m.voxelUm = {0.1, 0.1, 0.3};
        m.normalizeChannels();
        return std::make_shared<MemorySource>(a, m);
    }

    struct Scratch {
        std::filesystem::path dir = test::uniqueTempPath("tool_gate", "");
        Scratch() { std::filesystem::create_directories(dir); }
        ~Scratch() {
            std::error_code ec;
            std::filesystem::remove_all(dir, ec);
        }
    };

    // A workbench with a dataset, on the CPU, with Load and `kinds` as its
    // steps (the default Contrast step removed).
    struct Bench {
        Scratch scratch;
        Workbench wb{scratch.dir};
        explicit Bench(const std::vector<std::string>& kinds = {}) {
            registerGateOps();
            wb.setDataset(smallSource());
            wb.setBackend(Backend::Cpu);
            while (wb.pipeline().size() > 1) wb.removeStep(1);
            for (const std::string& k : kinds) REQUIRE(wb.addStep(k) != 0);
        }
    };

    // A run executing on a thread of its own. Let go, joined and folded back
    // however the test ends, so a failed check never leaves a thread behind.
    struct BackgroundRun {
        Workbench& wb;
        std::shared_ptr<RunJob> job;
        std::thread thread;
        explicit BackgroundRun(Workbench& w) : wb(w) {
            GateSlowOp::release = false;
            GateSlowOp::started = false;
            job = wb.createRun();
            REQUIRE(job);
            thread = std::thread([j = job] { j->execute(); });
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (!GateSlowOp::started.load() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        void finish() {
            GateSlowOp::release = true;
            if (thread.joinable()) thread.join();
            wb.finishRun(job);
        }
        ~BackgroundRun() {
            if (thread.joinable()) {
                job->cancel();
                finish();
            }
        }
    };

    bool logContains(const Workbench& wb, const std::string& text) {
        for (const std::string& line : wb.log())
            if (line.find(text) != std::string::npos) return true;
        return false;
    }

    std::string kindOf(const json& r) { return r.value("error_kind", std::string()); }

    // What LocalWorker::connect throws when the worker does not come up.
    WorkerStartError startError(WorkerStartError::Kind kind, const std::string& message, const std::string& hint) {
        WorkerStartError e(kind, message);
        e.interpreter = "/opt/python/bin/python3";
        e.source = "discovered";
        e.missing = {"numpy"};
        e.hint = hint;
        return e;
    }

} // namespace

// --- the busy gate ------------------------------------------------------------------

TEST_CASE("tool gate: edits answer busy while a run is active, reads still answer", "[app][tool_gate]") {
    Bench b({"test_gate_slow"});
    ToolApi api(b.wb);
    {
        BackgroundRun run(b.wb);
        REQUIRE(GateSlowOp::started.load());
        REQUIRE(b.wb.running());

        const json setParams = api.call("set_params", {{"step", 2}, {"params", {{"ticks", 7}}}});
        CHECK(kindOf(setParams) == "busy");
        CHECK_THAT(setParams.value("error", std::string()), ContainsSubstring("run is in progress"));
        CHECK(kindOf(api.call("undo", json::object())) == "busy");
        CHECK(kindOf(api.call("remove_step", {{"step", 2}})) == "busy");
        CHECK(kindOf(api.call("add_step", {{"kind", "test_gate_slow"}})) == "busy");
        // the gate comes before the arguments are read: a call that would
        // also be malformed is still told the run is the reason
        CHECK(kindOf(api.call("remove_step", json::object())) == "busy");
        // and nothing was changed
        CHECK(b.wb.pipeline().size() == 2);
        CHECK(b.wb.pipeline().at(1).params.getInt("ticks") == 1);

        const json state = api.call("get_state", json::object());
        CHECK_FALSE(state.contains("error_kind"));
        CHECK(state.value("running", false));
        CHECK_FALSE(api.call("get_log", {{"lines", 5}}).contains("error_kind"));
        CHECK_FALSE(api.call("get_step", {{"step", 2}}).contains("error_kind"));

        run.finish();
        CHECK(run.job->succeeded());
    }
    REQUIRE_FALSE(b.wb.running());
    const json after = api.call("set_params", {{"step", 2}, {"params", {{"ticks", 7}}}});
    CHECK_FALSE(after.contains("error_kind"));
    CHECK(b.wb.pipeline().at(1).params.getInt("ticks") == 7);
    CHECK_FALSE(api.call("undo", json::object()).contains("error_kind"));
    CHECK(b.wb.pipeline().at(1).params.getInt("ticks") == 1);
}

// --- error kinds --------------------------------------------------------------------

TEST_CASE("tool gate: every failure carries an error_kind", "[app][tool_gate]") {
    Bench b({"test_gate_slow"});
    ToolApi api(b.wb);

    SECTION("an unknown tool") {
        const json r = api.call("no_such_tool", json::object());
        CHECK(kindOf(r) == "unknown_tool");
        CHECK_THAT(r.value("error", std::string()), ContainsSubstring("no_such_tool"));
    }
    SECTION("steps that do not exist, and step values that are no step") {
        const json params = {{"ticks", 3}};
        CHECK(kindOf(api.call("set_params", {{"step", 9}, {"params", params}})) == "unknown_step");
        CHECK(kindOf(api.call("set_params", {{"step", 0}, {"params", params}})) == "unknown_step");
        CHECK(kindOf(api.call("set_params", {{"step", 1e30}, {"params", params}})) == "unknown_step");
        CHECK(kindOf(api.call("set_params", {{"step", "no such step"}, {"params", params}})) == "unknown_step");
        const json unknown = api.call("get_step", {{"step", 9}});
        CHECK_THAT(unknown.value("hint", std::string()), ContainsSubstring("get_state"));
        CHECK(kindOf(api.call("set_params", {{"params", params}})) == "invalid_argument");
        CHECK(kindOf(api.call("set_params", {{"step", 2.5}, {"params", params}})) == "invalid_argument");
        CHECK(kindOf(api.call("set_params", {{"step", true}, {"params", params}})) == "invalid_argument");
        // "" is a part of every name, and used to pick the Load step
        CHECK(kindOf(api.call("get_step", {{"step", ""}})) == "invalid_argument");
        // what does resolve still does
        CHECK(api.call("get_step", {{"step", "gate slow"}}).value("step", 0) == 2);
        CHECK(api.call("get_step", {{"step", 2.0}}).value("step", 0) == 2);
    }
    SECTION("arguments the tool cannot use") {
        CHECK(kindOf(api.call("set_params", {{"step", 2}, {"params", {{"nope", 1}}}})) == "invalid_argument");
        CHECK(kindOf(api.call("set_step_enabled", {{"step", 2}, {"enabled", "yes"}})) == "invalid_argument");
        const json kind = api.call("add_step", {{"kind", "no_such_kind"}});
        CHECK(kindOf(kind) == "unknown_operation");
        CHECK_THAT(kind.value("hint", std::string()), ContainsSubstring("list_operations"));
    }
    SECTION("run without a hook is unsupported, not a failed run") {
        const json r = api.call("run", json::object());
        CHECK(kindOf(r) == "unsupported");
        CHECK_FALSE(r.contains("ok"));
    }
    SECTION("export_training_data of a step that has not run") {
        const json r = api.call("export_training_data", {{"step", 2}, {"directory", b.scratch.dir.string()}});
        CHECK(kindOf(r) == "not_computed");
        CHECK(r.value("hint", std::string()) == "run it first");
    }
    SECTION("a ToolFailure keeps its code, hint and data") {
        api.addTool({"gate_fail", "Fails.", json{{"type", "object"}}, [](const json&) -> json {
                         throw ToolFailure("not_computed", "step 02 has not run", "run it first", json{{"step", 2}});
                     }});
        const json r = api.call("gate_fail", json::object());
        CHECK(kindOf(r) == "not_computed");
        CHECK(r.value("error", std::string()) == "step 02 has not run");
        CHECK(r.value("hint", std::string()) == "run it first");
        CHECK(r["data"]["step"] == 2);
        // no hint and no data: neither key
        api.addTool({"gate_bare", "Fails bare.", json{{"type", "object"}}, [](const json&) -> json {
                         throw ToolFailure("too_large", "too large");
                     }});
        const json bare = api.call("gate_bare", json::object());
        CHECK(kindOf(bare) == "too_large");
        CHECK_FALSE(bare.contains("hint"));
        CHECK_FALSE(bare.contains("data"));
    }
    SECTION("a worker that cannot start, a cancellation, anything else") {
        api.addTool({"gate_worker", "Needs the worker.", json{{"type", "object"}}, [](const json&) -> json {
                         throw startError(WorkerStartError::Kind::MissingPackages, "numpy is not installed", "Set it up.");
                     }});
        const json worker = api.call("gate_worker", json::object());
        CHECK(kindOf(worker) == "worker_unavailable");
        CHECK(worker.value("hint", std::string()) == "Set it up.");
        CHECK(worker["data"]["kind"] == "missing_packages");
        CHECK(worker["data"]["missing"] == json::array({"numpy"}));

        api.addTool({"gate_cancel", "Is cancelled.", json{{"type", "object"}}, [](const json&) -> json { throw CancelledError(); }});
        CHECK(kindOf(api.call("gate_cancel", json::object())) == "cancelled");
        // the library's untyped cancellation (app/core/cancel.hpp)
        api.addTool({"gate_cancel_text", "Is cancelled.", json{{"type", "object"}}, [](const json&) -> json {
                         throw std::runtime_error("cancelled");
                     }});
        CHECK(kindOf(api.call("gate_cancel_text", json::object())) == "cancelled");
        api.addTool({"gate_boom", "Fails.", json{{"type", "object"}}, [](const json&) -> json { throw std::runtime_error("boom"); }});
        const json boom = api.call("gate_boom", json::object());
        CHECK(kindOf(boom) == "failed");
        CHECK(boom.value("error", std::string()) == "boom");
        // an index out of range in a tool is the caller's mistake, not a failed tool
        api.addTool({"gate_range", "Indexes.", json{{"type", "object"}}, [](const json&) -> json {
                         return json{{"v", std::vector<int>{}.at(3)}};
                     }});
        CHECK(kindOf(api.call("gate_range", json::object())) == "invalid_argument");
    }
}

TEST_CASE("tool gate: a result with an empty error and no error_kind is a success", "[app][tool_gate]") {
    // The window's run hook answers {"ok", "error": "", "seconds"} for a run
    // that went well: "error" alone never means a failure.
    Bench b({"test_gate_slow"});
    ToolApi api(b.wb);
    api.setRunHook([](int) { return json{{"ok", true}, {"error", ""}, {"seconds", 0.25}}; });
    const json r = api.call("run", json::object());
    CHECK_FALSE(r.contains("error_kind"));
    CHECK(r.value("ok", false));
    CHECK(r.value("error", std::string("x")).empty());
    // the action card says it ran, not that it failed
    const std::vector<ActionRecord> actions = api.takeActions();
    REQUIRE(actions.size() == 1);
    CHECK_THAT(actions[0].text, !ContainsSubstring("failed"));

    // and a run the hook reports as failed is still the hook's value
    api.setRunHook([](int) { return json{{"ok", false}, {"error", "boom"}, {"seconds", 0.1}}; });
    const json failed = api.call("run", {{"step", 2}});
    CHECK_FALSE(failed.contains("error_kind"));
    CHECK(failed.value("error", std::string()) == "boom");
}

TEST_CASE("tool gate: get_help refuses a kind that could leave the help directory", "[app][tool_gate]") {
    Bench b;
    ToolApi api(b.wb);
    int asked = 0;
    api.setHelpHook([&asked](const std::string& kind) {
        ++asked;
        return "# " + kind;
    });
    for (const char* bad : {"../x", "..\\x", "a/b", "C:x", "../../etc/passwd", "test_gate.unknown", ".hidden"}) {
        INFO(bad);
        CHECK(kindOf(api.call("get_help", {{"kind", bad}})) == "invalid_argument");
    }
    CHECK(asked == 0);
    CHECK(api.call("get_help", {{"kind", "contrast"}}).value("markdown", std::string()) == "# contrast");
    CHECK(api.call("get_help", {{"kind", "test_gate-slow_2"}}).value("markdown", std::string()) == "# test_gate-slow_2");
    // a registered kind with a '.', as a plugin's may have
    CHECK(api.call("get_help", {{"kind", "test_gate.dotted"}}).value("markdown", std::string()) == "# test_gate.dotted");
    // no kind: the selected step's
    CHECK_FALSE(api.call("get_help", json::object()).contains("error_kind"));
    CHECK(asked == 4);

    // A pipeline file may name any kind; the stand-in that keeps its step
    // makes it a registered kind, and it is still not handed to the hook.
    b.wb.replacePipeline(Pipeline::fromJson(json::parse(R"({"steps": [{"kind": "load"}, {"kind": "../../test_gate_escape"}]})")),
                         "Gate pipeline");
    REQUIRE(b.wb.pipeline().at(b.wb.selectedIndex()).kind == "../../test_gate_escape");
    const json escaped = api.call("get_help", json::object());
    CHECK(kindOf(escaped) == "not_found");
    CHECK_THAT(escaped.value("hint", std::string()), ContainsSubstring("list_operations"));
    CHECK(asked == 4);
}

// --- the tool table ------------------------------------------------------------------

TEST_CASE("tool gate: addTool replaces a tool in place and removeTool removes one", "[app][tool_gate]") {
    Bench b;
    ToolApi api(b.wb);
    const std::vector<std::string> before = api.toolNames();
    const std::size_t n = before.size();

    api.addTool({"get_log", "Replaced.", json{{"type", "object"}}, [](const json&) { return json{{"replaced", true}}; }, "Log"});
    CHECK(api.toolNames() == before);   // same place, same count
    CHECK(api.call("get_log", json::object()).value("replaced", false));
    REQUIRE(api.findTool("get_log"));
    CHECK(api.findTool("get_log")->description == "Replaced.");
    CHECK(api.findTool("get_log")->title == "Log");

    api.addTool({"gate_extra", "Extra.", json{{"type", "object"}}, [](const json&) { return json{{"extra", 1}}; }});
    REQUIRE(api.tools().size() == n + 1);
    CHECK(api.tools().back().name == "gate_extra");
    CHECK(api.call("gate_extra", json::object()).value("extra", 0) == 1);

    CHECK(api.removeTool("focus_track"));
    CHECK(api.findTool("focus_track") == nullptr);
    CHECK(kindOf(api.call("focus_track", {{"id", 1}})) == "unknown_tool");
    CHECK_FALSE(api.removeTool("focus_track"));
    CHECK(api.tools().size() == n);
}

TEST_CASE("tool gate: a tool may change the tool table while it runs", "[app][tool_gate]") {
    Bench b;
    ToolApi api(b.wb);
    // The tool replaces itself, removes itself and adds another tool from inside its own function; call() runs a copy of
    // the function, so the table growing or shrinking under it does not destroy the function that is running.
    api.addTool({"gate_self", "Rewrites the table.", json{{"type", "object"}}, [&api](const json&) -> json {
                     const std::string before = "still here";
                     api.addTool({"gate_self", "Replaced.", json{{"type", "object"}}, [](const json&) { return json{{"second", true}}; }});
                     for (int i = 0; i < 64; ++i)
                         api.addTool({"gate_fill_" + std::to_string(i), "Fill.", json{{"type", "object"}},
                                      [](const json&) { return json::object(); }});
                     CHECK(api.removeTool("gate_self"));
                     return json{{"first", before}};
                 }});
    const json r = api.call("gate_self", json::object());
    CHECK(r.value("first", std::string()) == "still here");
    CHECK(api.findTool("gate_self") == nullptr);
    CHECK(api.findTool("gate_fill_63") != nullptr);
    CHECK(kindOf(api.call("gate_self", json::object())) == "unknown_tool");
}

TEST_CASE("tool gate: schemas() is what the assistant always had; the hints are set", "[app][tool_gate]") {
    Bench b;
    ToolApi api(b.wb);
    const std::vector<std::string> names = {"get_state", "list_operations", "get_step", "add_step", "remove_step",
                                            "move_step", "set_step_enabled", "set_params", "apply_preset", "set_cache",
                                            "run", "view_step", "select_step", "set_view", "list_tracks", "focus_track",
                                            "get_diagnostics", "get_help", "undo", "redo", "set_backend",
                                            "load_example_pipeline", "export_training_data", "get_log"};
    CHECK(api.toolNames() == names);
    const json schemas = api.schemas();
    REQUIRE(schemas.size() == names.size());
    for (std::size_t i = 0; i < names.size(); ++i) {
        const json& s = schemas[i];
        INFO(names[i]);
        CHECK(s.size() == 2);
        CHECK(s["type"] == "function");
        const json& f = s["function"];
        CHECK(f.size() == 3);   // name, description, parameters: no title, hints or _meta
        CHECK(f["name"] == names[i]);
        CHECK(f["description"].is_string());
        CHECK(f["parameters"]["type"] == "object");
    }

    const std::vector<std::string> busy = {"add_step", "remove_step", "move_step", "set_step_enabled",
                                           "set_params", "apply_preset", "set_cache", "undo", "redo",
                                           "load_example_pipeline", "export_training_data"};
    const std::vector<std::string> readOnly = {"get_state", "list_operations", "get_step", "list_tracks",
                                               "get_diagnostics", "get_help", "get_log"};
    const std::vector<std::string> idempotent = {"set_step_enabled", "set_params", "apply_preset", "set_cache", "set_backend"};
    const std::vector<std::string> big = {"list_operations", "get_help", "get_log"};
    auto in = [](const std::vector<std::string>& list, const std::string& name) {
        return std::find(list.begin(), list.end(), name) != list.end();
    };
    for (const ToolSpec& t : api.tools()) {
        INFO(t.name);
        CHECK_FALSE(t.title.empty());
        CHECK(t.refusedWhileRunning == in(busy, t.name));
        CHECK(t.readOnly == in(readOnly, t.name));
        CHECK(t.idempotent == in(idempotent, t.name));
        CHECK(t.destructive == (t.name == "export_training_data"));
        CHECK(t.openWorld == (t.name == "run"));   // a run may download weights and use the HPC worker
        CHECK(t.meta.is_object());
        CHECK(t.meta.contains("anthropic/maxResultSizeChars") == in(big, t.name));
    }
}

// --- what the workbench keeps for its host ----------------------------------------------

TEST_CASE("tool gate: createRun says why it refused", "[app][tool_gate][workbench]") {
    registerGateOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    CHECK(wb.lastRunRefusal().kind == RunRefusal::Kind::None);

    CHECK_FALSE(wb.createRun());
    CHECK(wb.lastRunRefusal().kind == RunRefusal::Kind::NoDataset);
    CHECK(wb.lastRunRefusal().message == "Open a dataset before running.");
    CHECK(logContains(wb, wb.lastRunRefusal().message));

    wb.setDataset(smallSource());
    wb.setBackend(Backend::Cpu);
    while (wb.pipeline().size() > 1) wb.removeStep(1);

    SECTION("a step that cannot run") {
        wb.addStep("test_gate_invalid");
        CHECK_FALSE(wb.createRun());
        CHECK(wb.lastRunRefusal().kind == RunRefusal::Kind::Invalid);
        CHECK(wb.lastRunRefusal().step == 1);
        CHECK_THAT(wb.lastRunRefusal().message, ContainsSubstring("this step never runs"));
        CHECK(logContains(wb, wb.lastRunRefusal().message));
    }
    SECTION("a step that needs the worker, and no launcher") {
        wb.addStep("test_gate_worker");
        CHECK_FALSE(wb.createRun());
        CHECK(wb.lastRunRefusal().kind == RunRefusal::Kind::NoLauncher);
        // the core names no window: without a host hint, no Preferences
        CHECK(wb.lastRunRefusal().message == "Worker unavailable: no Python worker launcher configured.");
        wb.setWorkerHint("Name an interpreter with --python.");
        CHECK_FALSE(wb.createRun());
        CHECK(wb.lastRunRefusal().message ==
              "Worker unavailable: no Python worker launcher configured. Name an interpreter with --python.");
        CHECK(logContains(wb, wb.lastRunRefusal().message));
        CHECK_FALSE(wb.running());
    }
    SECTION("a run already in progress, then none once a job is made") {
        wb.addStep("test_gate_slow");
        {
            BackgroundRun run(wb);
            CHECK(wb.lastRunRefusal().kind == RunRefusal::Kind::None);
            CHECK_FALSE(wb.createRun());
            CHECK(wb.lastRunRefusal().kind == RunRefusal::Kind::Running);
            CHECK(wb.lastRunRefusal().message == "A run is already in progress.");
            run.finish();
        }
        auto job = wb.createRun();
        REQUIRE(job);
        CHECK(wb.lastRunRefusal().kind == RunRefusal::Kind::None);
        CHECK(wb.lastRunRefusal().message.empty());
        wb.finishRun(job);   // never executed: logged as abandoned
    }
}

TEST_CASE("tool gate: a run keeps the worker start failure, with the host's hint", "[app][tool_gate][workbench]") {
    Bench b({"test_gate_worker"});
    Workbench& wb = b.wb;
    auto runOnce = [&wb] {
        auto job = wb.createRun();
        REQUIRE(job);
        job->execute();
        wb.finishRun(job);
        return job;
    };

    SECTION("a WorkerStartError is kept whole, and its own hint wins") {
        wb.setWorkerHint("The host's generic hint.");
        wb.setLocalWorkerLauncher([]() -> std::unique_ptr<RemoteWorker> {
            throw startError(WorkerStartError::Kind::MissingPackages,
                             "the Python worker cannot start: numpy is not installed in /opt/python/bin/python3 (found on PATH)",
                             "Set up SIRIUS's own Python environment.");
        });
        auto job = runOnce();
        CHECK_FALSE(job->succeeded());
        REQUIRE(job->workerFailure().has_value());
        CHECK(job->workerFailure()->kind == WorkerStartError::Kind::MissingPackages);
        CHECK(job->workerFailure()->missing == std::vector<std::string>{"numpy"});
        CHECK(job->error() ==
              "Worker unavailable: the Python worker cannot start: numpy is not installed in /opt/python/bin/python3 "
              "(found on PATH). Set up SIRIUS's own Python environment.");
        CHECK(logContains(wb, "Run failed: Worker unavailable: the Python worker cannot start"));
    }
    SECTION("a WorkerStartError without a hint takes the host's") {
        wb.setWorkerHint("The host's generic hint.");
        wb.setLocalWorkerLauncher([]() -> std::unique_ptr<RemoteWorker> {
            throw startError(WorkerStartError::Kind::Failed, "it exited with code 1", "");
        });
        auto job = runOnce();
        REQUIRE(job->workerFailure().has_value());
        CHECK(job->error() == "Worker unavailable: it exited with code 1. The host's generic hint.");
    }
    SECTION("any other failure gets the host's hint, and no workerFailure") {
        wb.setWorkerHint("The host's generic hint.");
        wb.setLocalWorkerLauncher([]() -> std::unique_ptr<RemoteWorker> { throw std::runtime_error("the port line never came"); });
        auto job = runOnce();
        CHECK_FALSE(job->workerFailure().has_value());
        CHECK(job->error() == "Worker unavailable: the port line never came. The host's generic hint.");
    }
    SECTION("without a host hint the text names no window") {
        wb.setLocalWorkerLauncher([]() -> std::unique_ptr<RemoteWorker> { return nullptr; });
        auto job = runOnce();
        CHECK(job->error() == "Worker unavailable: the Python worker did not start.");
        CHECK_THAT(job->error(), !ContainsSubstring("Preferences"));
    }
    SECTION("a start that gave up because the run was cancelled is a cancellation") {
        wb.setLocalWorkerLauncher([]() -> std::unique_ptr<RemoteWorker> { throw CancelledError(); });
        auto job = runOnce();
        CHECK(job->wasCancelled());
        CHECK(job->error() == "cancelled");
        CHECK_FALSE(job->workerFailure().has_value());
    }
    SECTION("a job cancelled while the worker failed keeps no workerFailure") {
        std::shared_ptr<RunJob> job;
        wb.setLocalWorkerLauncher([&job]() -> std::unique_ptr<RemoteWorker> {
            job->cancel();
            throw startError(WorkerStartError::Kind::NoInterpreter, "no Python 3 interpreter was found", "Install one.");
        });
        job = wb.createRun();
        REQUIRE(job);
        job->execute();
        wb.finishRun(job);
        CHECK(job->wasCancelled());
        CHECK_FALSE(job->workerFailure().has_value());
    }
}

TEST_CASE("tool gate: loadPlugins keeps why the worker did not start", "[app][tool_gate][workbench]") {
    registerGateOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    CHECK(wb.pluginError().empty());

    CHECK(wb.loadPlugins(false) == 0);
    CHECK(wb.pluginError() == "no Python worker launcher configured");
    CHECK_FALSE(wb.pluginWorkerFailure().has_value());

    wb.setWorkerHint("The host's generic hint.");
    wb.setLocalWorkerLauncher([]() -> std::unique_ptr<RemoteWorker> {
        throw startError(WorkerStartError::Kind::NoInterpreter, "no Python 3 interpreter was found", "Install Python 3.");
    });
    CHECK(wb.loadPlugins(false) == 0);
    CHECK(wb.pluginError() == "no Python 3 interpreter was found");
    REQUIRE(wb.pluginWorkerFailure().has_value());
    CHECK(wb.pluginWorkerFailure()->kind == WorkerStartError::Kind::NoInterpreter);
    CHECK(wb.pluginWorkerFailure()->setupWouldHelp());
    CHECK(logContains(wb, "Plugins unavailable: no Python 3 interpreter was found. Install Python 3."));

    // the next attempt replaces what the last one kept
    wb.setLocalWorkerLauncher([]() -> std::unique_ptr<RemoteWorker> { throw std::runtime_error("connection refused"); });
    CHECK(wb.loadPlugins(true) == 0);
    CHECK(wb.pluginError() == "connection refused");
    CHECK_FALSE(wb.pluginWorkerFailure().has_value());
    CHECK(logContains(wb, "Plugins unavailable: connection refused. The host's generic hint."));

    wb.setLocalWorkerLauncher([]() -> std::unique_ptr<RemoteWorker> { return nullptr; });
    CHECK(wb.loadPlugins(false) == 0);
    CHECK(wb.pluginError() == "the Python worker did not start");

    // Refused during a run, the attempt says so rather than leaving the last
    // one's state (or an empty error that reads as a load that found none).
    Bench b({"test_gate_slow"});
    int started = 0;
    b.wb.setLocalWorkerLauncher([&started]() -> std::unique_ptr<RemoteWorker> {
        ++started;
        throw startError(WorkerStartError::Kind::NoInterpreter, "no Python 3 interpreter was found", "Install Python 3.");
    });
    CHECK(b.wb.loadPlugins(false) == 0);
    REQUIRE(b.wb.pluginWorkerFailure().has_value());
    {
        BackgroundRun run(b.wb);
        REQUIRE(b.wb.running());
        CHECK(b.wb.loadPlugins(false) == 0);
        CHECK(b.wb.pluginError() == "a run is in progress");
        CHECK_FALSE(b.wb.pluginWorkerFailure().has_value());
        CHECK(logContains(b.wb, "Cannot load plugins while a run is in progress"));
        run.finish();
    }
    CHECK(started == 1);
}
