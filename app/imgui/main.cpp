// sirius-imgui: the SIRIUS microscopy workbench (docs/design/README.md) over
// Dear ImGui. The same command line as the Qt application:
//
//   sirius-imgui [--dataset stack.tif] [--pipeline steps.sirius.toml] [--run] [files...]
//
// Everything can also be opened from the File menu; --run runs every
// enabled step as soon as the window is up. Files named without an option
// open as though dropped on the window, which is what a file manager's
// "Open with" passes.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <system_error>
#include <vector>

#include <nlohmann/json.hpp>

#include <sirius/device.hpp>

#include "core/app_paths.hpp"
#include "core/help_pages.hpp"
#include "core/operation.hpp"
#include "core/ops/builtin.hpp"
#include "core/tool_api.hpp"
#include "core/workbench.hpp"
#include "imgui/app.hpp"
#include "imgui/bridge.hpp"
#include "imgui/dialogs/dialogs.hpp"
#include "imgui/platform.hpp"
#include "imgui/secret_store.hpp"
#include "imgui/settings.hpp"
#include "imgui/strings.hpp"
#include "imgui/viewer/viewer.hpp"
#include "imgui/worker_launcher.hpp"

namespace {

    using namespace sirius::app;
    using namespace sirius::app::gui;
    using Clock = std::chrono::steady_clock;

    struct Option {
        const char* name;
        bool takesValue;
        const char* valueName;
        const char* help;
    };

    const Option kOptions[] = {
        {"dataset", true, "path", "Dataset to open (TIFF / OME-TIFF / zarr)"},
        {"pipeline", true, "path", "Pipeline file (.sirius.toml)"},
        {"run", false, "", "Run every enabled step once the window is up"},
        // Developer aids: grab the window to a PNG (after the run when --run is
        // given) and quit; or just quit after a delay, for headless smoke tests.
        {"screenshot", true, "path", "Save a screenshot of the window to <path> and quit"},
        {"quit-after", true, "ms", "Quit after <ms> milliseconds"},
        // Scripting for smoke tests: tool calls through the assistant's typed API
        // ({"name": "set_view", "args": {"mode": "3d"}}) and menu actions by
        // their text ("Assistant"), applied once the run (if any) has finished.
        {"tool", true, "json", "Call a tool of the assistant API (JSON, repeatable)"},
        {"record", true, "path", "Record this session to a JSON-lines file"},
        {"drop", true, "path", "Act as though this path were dropped on the window (repeatable)"},
        {"stroke", true, "spec", "Drag on the XY pane: x0,y0,x1,y1,moves in voxels (repeatable, after the tools)"},
        {"wheel", true, "spec", "Wheel on the XY pane (the step pane in Compare): x,y,steps in voxels (repeatable)"},
        {"action", true, "text", "Trigger a menu action by its text (repeatable)"},
        {"ask", true, "text", "Send a message to the assistant"},
        // A settings file of this run's own. Without it every headless run reads
        // and writes the settings of whoever is logged in.
        {"settings", true, "dir", "Keep settings in <dir> instead of the user's own; 'scratch' uses a new one, removed when the run ends"},
        {"settle", true, "ms", "Milliseconds to wait before the screenshot (default 600)"},
        {"size", true, "WxH", "Window size in design pixels (default 1600x960)"},
        {"list-cuda", false, "", "Print visible CUDA devices and exit"},
        {"help", false, "", "Show this help"},
        {"version", false, "", "Show the version"},
    };

    struct Arguments {
        // in the order they were written: scripted steps run in that order
        std::vector<std::pair<std::string, std::string>> options;
        std::vector<std::string> files;
        std::string error;

        bool has(const std::string& name) const {
            for (const auto& o : options)
                if (o.first == name) return true;
            return false;
        }
        std::string value(const std::string& name, const std::string& def = {}) const {
            for (auto it = options.rbegin(); it != options.rend(); ++it)
                if (it->first == name) return it->second;
            return def;
        }
    };

    Arguments parse(int argc, char** argv) {
        Arguments a;
        bool optionsDone = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (optionsDone || !startsWith(arg, "-") || arg == "-") {
                a.files.push_back(arg);
                continue;
            }
            if (arg == "--") {
                optionsDone = true;
                continue;
            }
            std::string name = arg.substr(startsWith(arg, "--") ? 2 : 1), value;
            bool inlineValue = false;
            const std::size_t eq = name.find('=');
            if (eq != std::string::npos) {
                value = name.substr(eq + 1);
                name.resize(eq);
                inlineValue = true;
            }
            if (name == "h") name = "help";
            if (name == "v") name = "version";
            const Option* option = nullptr;
            for (const Option& o : kOptions)
                if (name == o.name) option = &o;
            if (!option) {
                a.error = "unknown option " + arg;
                return a;
            }
            if (option->takesValue && !inlineValue) {
                if (i + 1 >= argc) {
                    a.error = "option --" + name + " needs a value";
                    return a;
                }
                value = argv[++i];
            }
            a.options.emplace_back(name, value);
        }
        return a;
    }

    void printHelp() {
        std::printf("Usage: sirius-imgui [options] [files...]\nSIRIUS microscopy processing workbench\n\nOptions:\n");
        for (const Option& o : kOptions) {
            std::string left = std::string("  --") + o.name;
            if (o.takesValue) left += std::string(" <") + o.valueName + ">";
            std::printf("%-28s %s\n", left.c_str(), o.help);
        }
        std::printf("\nArguments:\n  files                        Datasets or pipeline files to open, as though dropped on the window\n");
    }

    int toInt(const std::string& s, int def) {
        try {
            return std::stoi(s);
        } catch (const std::exception&) {
            return def;
        }
    }

    double toDouble(const std::string& s) {
        try {
            return std::stod(s);
        } catch (const std::exception&) {
            return 0.0;
        }
    }

} // namespace

int main(int argc, char** argv) {
    platform::attachParentConsole();
    const Arguments args = parse(argc, argv);
    if (!args.error.empty()) {
        std::fprintf(stderr, "sirius-imgui: %s\n", args.error.c_str());
        return 2;
    }
    if (args.has("help")) {
        printHelp();
        return 0;
    }
    if (args.has("version")) {
        std::printf("sirius-imgui %s\n", SIRIUS_VERSION);
        return 0;
    }
    if (args.has("list-cuda")) {
        const int n = sirius::cudaDeviceCount();
        std::printf("cuda devices: %d\n", n);
        for (int i = 0; i < n; ++i) {
            try {
                const auto p = sirius::deviceProperties(sirius::Device::cuda(i));
                std::printf("  %d: %s  %.1f GB  sm_%d%d\n", i, p.name.c_str(),
                            static_cast<double>(p.totalMemoryBytes) / (1024.0 * 1024.0 * 1024.0), p.computeMajor, p.computeMinor);
            } catch (const std::exception& e) {
                std::printf("  %d: (%s)\n", i, e.what());
            }
        }
        return n > 0 ? 0 : 1;
    }

    // the help pages, the worker, the plugins and the fonts are found relative to it
    setApplicationDirectory(platform::executableDirectory());

    const bool filesHavePipeline =
        std::any_of(args.files.begin(), args.files.end(), [](const std::string& f) { return endsWithNoCase(f, ".toml"); });

    // Before anything reads a setting. A scratch directory goes last, when
    // main returns: everything that saves settings on the way out is declared
    // after it.
    struct ScratchDirectory {
        std::string path;
        ~ScratchDirectory() {
            if (path.empty()) return;
            std::error_code ec;
            std::filesystem::remove_all(std::filesystem::u8path(path), ec);
        }
    } scratchSettings;
    if (args.has("settings")) {
        std::string dir = args.value("settings");
        if (dir == "scratch") {
            dir = platform::tempDirectory() + format("/sirius-settings-%d-%lld", platform::processId(),
                                                     static_cast<long long>(Clock::now().time_since_epoch().count() % 1000000));
            scratchSettings.path = dir;
        }
        if (!platform::makePath(dir)) {
            std::fprintf(stderr, "cannot create the settings directory %s\n", dir.c_str());
            return 2;
        }
        settings().setDirectory(dir);
        secrets::setStoreDirectory(dir);
        std::fprintf(stderr, "settings: %s\n", settings().filePath().c_str());
    }

    registerBuiltinOperations();

    // Per-process scratch for the disk cache and worker files.
    const std::string scratch = platform::tempDirectory() + format("/sirius-%d", platform::processId());
    platform::makePath(scratch);

    int rc = 0;
    {
        Workbench workbench{std::filesystem::u8path(scratch)};
        applyStoredPreferences(workbench);
        // Steps that need Python (Torch models) get a worker spawned on demand.
        // Declared before the bridge: the bridge's destructor joins the run
        // thread, which may be inside the launcher starting that worker.
        WorkerLauncher launcher;
        Bridge bridge(workbench);
        launcher.setLogHandler([&bridge, &workbench](const std::string& line) {
            bridge.post([&workbench, line] { workbench.logLine("worker: " + line); });
        });
        workbench.setLocalWorkerLauncher([&launcher] { return launcher.connect(); });
        auto syncWorkerDevice = [&] {
            if (workbench.backend() != Backend::Cuda) launcher.setDevice("cpu");
            else if (workbench.cudaDevice() < 0) launcher.setDevice("cuda");
            else launcher.setDevice(format("cuda:%d", workbench.cudaDevice()));
        };
        syncWorkerDevice();
        bridge.backendChanged.connect(syncWorkerDevice);
        // a gated model's token goes with the request that downloads it
        workbench.setHubTokenProvider([] { return trimmed(secrets::read("hub/token")); });

        ToolApi tools(workbench);
        // get_help answers from the pages the application reads, as it does for the assistant
        tools.setHelpHook([](const std::string& kind) { return loadHelpPage(kind).markdown; });

        const bool scripted = args.has("tool") || args.has("action") || args.has("ask") || args.has("stroke") || args.has("wheel") ||
                              args.has("drop");
        const bool headless = scripted || args.has("screenshot");

        App app(bridge, tools, launcher);
        InitOptions init;
        // Nobody is at the keyboard in any of these modes, so the window must
        // not ask whether to cancel a running job on the way out.
        init.unattended = headless || args.has("quit-after");
        if (args.has("size")) {
            const std::vector<std::string> wh = split(toLower(args.value("size")), 'x');
            if (wh.size() == 2) {
                init.width = std::max(640, toInt(wh[0], init.width));
                init.height = std::max(480, toInt(wh[1], init.height));
            }
        }
        if (!app.init(init)) {
            std::error_code ec;
            std::filesystem::remove_all(std::filesystem::u8path(scratch), ec);
            return 2;
        }

        // Runs block the caller, not the window: a tool's "run" draws frames
        // until the worker thread is done (scripting and the assistant both
        // call tools between frames).
        tools.setRunHook([&app, &bridge](int target) {
            bool done = false, ok = false;
            std::string error;
            const Clock::time_point started = Clock::now();
            const int id = bridge.runFinished.connect([&](bool good, const std::string& err) {
                ok = good;
                error = err;
                done = true;
            });
            if (!bridge.startRun(target)) {
                bridge.runFinished.disconnect(id);
                return nlohmann::json{{"ok", false}, {"error", "the run could not start (see the log)"}};
            }
            app.waitUntil([&] { return done; });
            bridge.runFinished.disconnect(id);
            if (!done) return nlohmann::json{{"ok", false}, {"error", "the window was closed"}};
            const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
            return nlohmann::json{{"ok", ok}, {"error", error}, {"seconds", seconds}};
        });

        // User operations come from the Python worker. A pipeline given on the
        // command line may use them, so load them first in that case; otherwise
        // after the window is up so start-up stays quick.
        if (args.has("pipeline") || filesHavePipeline) {
            workbench.loadPlugins(false);
        } else {
            // off the GUI thread's first frames: the worker takes a moment to start
            const Clock::time_point at = Clock::now() + std::chrono::milliseconds(400);
            std::function<void()> later;
            auto retry = std::make_shared<std::function<void()>>();
            *retry = [&app, &workbench, at, retry] {
                if (Clock::now() < at) {
                    app.defer(*retry);
                    return;
                }
                if (workbench.canEdit()) workbench.loadPlugins(false);
            };
            app.defer(*retry);
        }
        if (args.has("pipeline")) app.openPipelinePath(args.value("pipeline"));
        // recording starts before anything scripted happens, so the run is in it
        if (args.has("record")) {
            try {
                workbench.startRecording(args.value("record"));
            } catch (const std::exception& e) {
                std::fprintf(stderr, "--record: %s\n", e.what());
            }
        }
        if (args.has("dataset")) {
            const std::string dataset = args.value("dataset");
            app.defer([&app, dataset] { app.openDatasetPath(dataset); });
        }
        if (!args.files.empty()) app.defer([&app, files = args.files] { app.dropPaths(files); });

        auto runTool = [&](const std::string& call) {
            const nlohmann::json j = nlohmann::json::parse(call, nullptr, false);
            if (!j.is_object()) {
                std::fprintf(stderr, "--tool: not a JSON object: %s\n", call.c_str());
                return;
            }
            const std::string name = j.value("name", std::string());
            const nlohmann::json toolArgs = j.contains("args") && j["args"].is_object() ? j["args"] : nlohmann::json::object();
            const nlohmann::json r = tools.call(name, toolArgs);
            workbench.logLine("tool " + name + " \xE2\x86\x92 " + r.dump().substr(0, 200));
            std::fprintf(stderr, "tool %s -> %s\n", name.c_str(), r.dump(2).c_str());
            std::fflush(stderr);
            app.settle();
        };
        // Scripted steps run in the order they were written on the command line.
        auto script = [&] {
            app.settle();   // a --dataset still loading
            for (const auto& o : args.options) {
                const std::string& name = o.first;
                const std::string& value = o.second;
                if (name == "tool") {
                    runTool(value);
                } else if (name == "action") {
                    if (!app.triggerAction(value)) std::fprintf(stderr, "no action named %s\n", value.c_str());
                    app.settle();
                } else if (name == "drop") {
                    app.dropPaths({value});
                    app.settle();
                } else if (name == "wheel") {
                    const std::vector<std::string> v = split(value, ',');
                    if (v.size() == 3) app.viewer().syntheticWheel(toDouble(v[0]), toDouble(v[1]), toDouble(v[2]));
                    app.settle();
                } else if (name == "stroke") {
                    const std::vector<std::string> v = split(value, ',');
                    if (v.size() == 5)
                        app.viewer().syntheticStroke(toDouble(v[0]), toDouble(v[1]), toDouble(v[2]), toDouble(v[3]), toInt(v[4], 1));
                    app.settle();
                } else if (name == "ask") {
                    app.askAssistant(value);
                }
            }
        };

        // An interactive --run just starts; a headless one (below) also decides
        // the exit code and when the window is grabbed.
        if (args.has("run") && !headless) app.defer([&app] { app.runAll(); });

        if (args.has("quit-after")) {
            const Clock::time_point at = Clock::now() + std::chrono::milliseconds(toInt(args.value("quit-after"), 0));
            auto tick = std::make_shared<std::function<void()>>();
            *tick = [&app, at, tick] {
                if (Clock::now() >= at) app.quitNow();
                else app.defer(*tick);
            };
            app.defer(*tick);
        }

        if (headless) {
            const std::string shot = args.value("screenshot");
            const int settle = args.has("settle") ? toInt(args.value("settle"), 600) : 600;
            app.defer([&, shot, settle] {
                // What the process exits with: 1 when a headless run failed (or
                // was still going when the deadline struck), 2 when it could
                // not start.
                const auto deadline = Clock::now() + std::chrono::seconds(600);
                auto waitFor = [&](int ms) {
                    const auto until = Clock::now() + std::chrono::milliseconds(ms);
                    app.waitUntil([&] { return Clock::now() >= until; });
                };
                app.settle();   // a --dataset or a dropped file still loading
                if (args.has("run")) {
                    bool done = false, ok = false;
                    const int id = bridge.runFinished.connect([&](bool good, const std::string&) {
                        ok = good;
                        done = true;
                    });
                    app.runAll();
                    if (!bridge.running()) {   // could not start: no dataset, a validation error
                        app.setExitCode(2);
                    } else {
                        app.waitUntil([&] { return done || Clock::now() >= deadline; });   // never hang a headless run
                        if (!done || !ok) app.setExitCode(1);
                    }
                    bridge.runFinished.disconnect(id);
                    waitFor(300);
                } else {
                    waitFor(600);
                }
                if (scripted) script();
                if (!shot.empty()) {
                    // a run, or a dataset still loading: the picture waits for either
                    app.waitUntil([&] { return (!bridge.running() && !bridge.taskRunning()) || Clock::now() >= deadline; });
                    waitFor(settle);
                    if (!app.screenshot(shot)) std::fprintf(stderr, "--screenshot: cannot write %s\n", shot.c_str());
                    if (bridge.running()) {   // a slow step must not hold the exit
                        bridge.cancelRun();
                        if (app.exitCode() == 0) app.setExitCode(1);
                    }
                    app.quitNow();
                }
            });
        }

        rc = app.run();
    }
    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::u8path(scratch), ec);
    return rc;
}
