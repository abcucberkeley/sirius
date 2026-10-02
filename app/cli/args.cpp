#include "cli/args.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>

namespace sirius::cli {

    namespace {

        std::string lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        }

        bool isDigits(const std::string& s) {
            return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
        }

        // Built field by field: an aggregate initialiser that leaves fields to
        // their defaults draws -Wmissing-field-initializers from GCC.
        OptionSpec opt(std::string name, std::string value, std::string type, std::string def, std::string help, bool repeatable = false) {
            OptionSpec o;
            o.name = std::move(name);
            o.value = std::move(value);
            o.type = std::move(type);
            o.def = std::move(def);
            o.help = std::move(help);
            o.repeatable = repeatable;
            return o;
        }

        OptionSpec flag(std::string name, std::string help) { return opt(std::move(name), "", "flag", "", std::move(help)); }

        CommandSpec command(std::string name, std::string synopsis, std::string summary) {
            CommandSpec c;
            c.name = std::move(name);
            c.synopsis = std::move(synopsis);
            c.summary = std::move(summary);
            return c;
        }

        OptionSpec stepOption(const char* what) {
            return opt("step", "S", "string", "",
                       std::string("the step ") + what + ": a number (1 = Load) or a name (default: the step the run ends at, the last enabled one "
                                                         "unless run --to says otherwise; with --no-run, the last step with an output)");
        }

        // --- the option groups ------------------------------------------------

        std::vector<OptionSpec> renderOptions() {
            return {
                stepOption("to draw"),
                opt("plane", "xy|xz|yz|mip", "enum", "xy", "a z plane, a reslice through a row or a column, or the maximum projection along z"),
                opt("z", "N[,N...]", "list", "", "the z plane (default the middle one); several give a grid of planes (xy only)"),
                opt("t", "N", "integer", "0", "the time point"),
                opt("y", "N", "integer", "", "the row an xz reslice goes through (default the middle one)"),
                opt("x", "N", "integer", "", "the column a yz reslice goes through (default the middle one)"),
                opt("channels", "0,1,...", "list", "", "the channels to draw (default all)"),
                opt("layout", "blend|channels", "enum", "blend", "the channels tinted and added, as the viewer draws them, or one grey panel per channel"),
                opt("window", "auto|full|c=lo:hi[:gamma],...", "string", "auto", "the display range: robust percentiles, the full range, or per channel"),
                flag("labels", "draw the labels (the default when the output has labels)"),
                flag("no-labels", "do not draw the labels"),
                opt("label", "ID", "integer", "", "highlight one label"),
                flag("solo", "draw only the label given with --label"),
                opt("label-opacity", "F", "number", "0.45", "the opacity of the label overlay"),
                opt("region", "x,y,w,h", "list", "", "a region of the plane, in voxels"),
                opt("max-size", "N", "integer", "1024", "the longest side in pixels at most (never enlarged past native), at most 1568; 0 = native, still capped"),
                opt("format", "png|jpeg", "enum", "", "the image format (default from the file name; else PNG, or JPEG when a PNG would pass 1 MiB)"),
                flag("no-physical-z", "do not stretch xz and yz by the voxel aspect"),
            };
        }

        std::vector<OptionSpec> statsOptions() {
            return {
                stepOption("to measure"),
                opt("t", "N|all", "string", "0", "the time point, or all"),
                opt("channels", "0,1,...", "list", "", "the channels (default all)"),
                opt("percentiles", "p,p,...", "list", "0.1,1,50,99,99.9", "the percentiles to report"),
                opt("histogram", "N", "integer", "0", "histogram bins (0 = no histogram)"),
                flag("no-labels", "leave out the label statistics"),
            };
        }

        std::vector<OptionSpec> exportOptions() {
            return {
                stepOption("to write"),
                opt("format", "tiff|ome-tiff|zarr|n5|raw", "enum", "", "the file format (default from the file name: .ome.tif, .tif, .zarr, .n5, .raw)"),
                opt("dtype", "T", "string", "float32", "the pixel type: uint8, uint16, float32, ..."),
                opt("scaling", "cast|minmax|fixed|percentile", "enum", "cast", "how values are mapped into the pixel type"),
                opt("range", "lo,hi", "list", "", "the value range for --scaling fixed"),
                opt("percentiles", "lo,hi", "list", "", "the percentiles for --scaling percentile"),
                opt("t", "a:b", "string", "", "time points a up to (not including) b; a: = to the end"),
                opt("z", "a:b", "string", "", "z planes a up to (not including) b; a: = to the end"),
                opt("channels", "0,1,...", "list", "", "the channels (default all)"),
                opt("compression", "none|lzw|deflate", "enum", "", "TIFF compression"),
                opt("level", "N", "integer", "", "the compression level"),
                flag("tiled", "write a tiled TIFF"),
                opt("tile", "W,H", "list", "", "the TIFF tile size"),
                flag("bigtiff", "always write BigTIFF"),
                flag("no-bigtiff", "never write BigTIFF"),
                opt("pyramid", "N", "integer", "", "resolution levels (1 = none)"),
                opt("chunk", "c,t,z,y,x", "list", "", "the zarr / N5 chunk shape"),
                opt("codec", "C", "string", "", "the zarr / N5 codec: blosc-zstd, blosc-lz4, zstd, gzip, none"),
                opt("zarr-version", "2|3", "enum", "", "the zarr format version"),
                flag("labels", "also write the step's labels"),
                flag("labels-only", "write only the labels (uint32 TIFF, deflate)"),
                flag("pipeline-sidecar", "write the pipeline beside the export (<path>.pipeline.toml)"),
            };
        }

        std::vector<OptionSpec> with(std::vector<OptionSpec> base, std::initializer_list<OptionSpec> more) {
            base.insert(base.end(), more.begin(), more.end());
            return base;
        }

        OptionSpec noRun() { return flag("no-run", "use what is computed; do not run the steps it needs"); }

        const OptionSpec* findIn(const std::vector<OptionSpec>& specs, const std::string& name) {
            for (const OptionSpec& s : specs)
                if (s.name == name) return &s;
            return nullptr;
        }

        const ActionSpec* findAction(const CommandSpec& c, const std::string& name) {
            for (const ActionSpec& a : c.actions)
                if (a.flag.name == name) return &a;
            return nullptr;
        }

        std::string valueOf(const std::vector<Option>& options, const std::string& name, const std::string& def) {
            for (auto it = options.rbegin(); it != options.rend(); ++it)
                if (it->name == name) return it->value;
            return def;
        }

        bool hasIn(const std::vector<Option>& options, const std::string& name) {
            return std::any_of(options.begin(), options.end(), [&](const Option& o) { return o.name == name; });
        }

        double parsePositive(const std::string& option, const std::string& value) {
            char* end = nullptr;
            const double v = std::strtod(value.c_str(), &end);
            if (value.empty() || end != value.c_str() + value.size() || !(v > 0.0))
                throw UsageError("--" + option + " expects a number of seconds above 0, not '" + value + "'");
            return v;
        }

        std::string oneOf(const std::string& option, const std::string& value, std::initializer_list<const char*> allowed) {
            const std::string v = lower(value);
            for (const char* a : allowed)
                if (v == a) return v;
            std::string list;
            for (const char* a : allowed) list += (list.empty() ? "" : ", ") + std::string(a);
            throw UsageError("--" + option + " must be one of " + list + ", not '" + value + "'");
        }

        void setGlobal(GlobalOptions& g, const std::string& name, const std::string& value) {
            if (name == "python") g.python = value;
            else if (name == "worker-dir") g.workerDir = value;
            else if (name == "backend") g.backend = oneOf(name, value, {"auto", "cpu", "cuda", "hpc"});
            else if (name == "cuda-device") {
                if (lower(value) == "all") g.cudaDevice = -1;
                else if (isDigits(value) && value.size() < 6) g.cudaDevice = std::atoi(value.c_str());
                else throw UsageError("--cuda-device expects a device number or all, not '" + value + "'");
            } else if (name == "hpc-device") {
                g.hpcDevice = oneOf(name, value, {"gpu", "cpu"});
            } else if (name == "hpc") {
                // host:port; an IPv6 address comes in brackets ([::1]:7645)
                const std::size_t colon = value.rfind(':');
                const std::string port = colon == std::string::npos ? std::string() : value.substr(colon + 1);
                std::string host = colon == std::string::npos ? std::string() : value.substr(0, colon);
                if (host.size() > 2 && host.front() == '[' && host.back() == ']') host = host.substr(1, host.size() - 2);
                if (host.empty() || !isDigits(port) || port.size() > 5 || std::atoi(port.c_str()) < 1 || std::atoi(port.c_str()) > 65535)
                    throw UsageError("--hpc expects host:port, not '" + value + "'");
                g.hpcHost = host;
                g.hpcPort = std::atoi(port.c_str());
            } else if (name == "plugins") g.plugins = oneOf(name, value, {"auto", "on", "off"});
            else if (name == "scratch") g.scratch = value;
            else if (name == "keep-scratch") g.keepScratch = true;
            else if (name == "record") g.record = value;
            else if (name == "timeout") g.timeoutSeconds = parsePositive(name, value);
            else if (name == "progress") g.progress = oneOf(name, value, {"none", "text", "json"});
            else if (name == "pretty") g.pretty = true;
            else if (name == "compact") g.pretty = false;
            else if (name == "quiet") g.quiet = true;
        }

        void setState(StateOptions& s, const std::string& name, const std::string& value) {
            if (name == "dataset") s.dataset = value;
            else if (name == "pipeline") s.pipeline = value;
            else if (name == "steps") s.steps = value;
            else if (name == "set") s.sets.push_back(value);
        }

    } // namespace

    const std::vector<OptionSpec>& globalOptionSpecs() {
        static const std::vector<OptionSpec> specs = {
            opt("python", "exe", "path", "", "the interpreter for the Python worker (default: $SIRIUS_PYTHON, then SIRIUS's own environment, then Python on PATH)"),
            opt("worker-dir", "dir", "path", "", "the directory holding sirius_worker/ (default: the one shipped with sirius-cli)"),
            opt("backend", "auto|cpu|cuda|hpc", "enum", "auto", "the compute backend; auto = CUDA when available, else CPU; hpc needs --hpc"),
            opt("cuda-device", "n|all", "string", "0", "the CUDA device, or all to spread the volumes over every GPU"),
            opt("hpc", "host:port", "string", "", "the HPC worker this process may use, the only one (token from $SIRIUS_HPC_TOKEN)"),
            opt("hpc-device", "gpu|cpu", "enum", "gpu", "where the HPC worker computes: its job's GPU, or its CPU (set_backend switches it later)"),
            opt("plugins", "auto|on|off", "enum", "auto", "user operations from the Python worker: loaded when a step needs them, at start, or never"),
            opt("scratch", "dir", "path", "",
                "the scratch directory (disk cache, rendered images); default a new temporary one. It is removed at exit when "
                "sirius-cli created it"),
            flag("keep-scratch", "keep the scratch directory at exit"),
            opt("record", "file.jsonl", "path", "", "record the session (edits, runs, results) to a JSON-lines file"),
            opt("timeout", "s", "number", "", "one-shot: cancel and exit 124 after s seconds; session: how long end of input waits for a run"),
            opt("progress", "none|text|json", "enum", "", "progress and log lines on stderr (default text on a terminal, else none)"),
            flag("pretty", "indented JSON on stdout (the default on a terminal)"),
            flag("compact", "one line of JSON on stdout (the default otherwise)"),
            flag("quiet", "no log lines on stderr (a failure still prints one summary line)"),
            flag("help", "this help, or a command's (also -h)"),
            flag("version", "the same as the version command"),
        };
        return specs;
    }

    const std::vector<OptionSpec>& stateOptionSpecs() {
        static const std::vector<OptionSpec> specs = {
            opt("dataset", "path", "path", "", "the dataset to open (TIFF, OME-TIFF, zarr, a SIRIUS dataset folder), with the open options"),
            opt("pipeline", "file.sirius.toml", "path", "", "the pipeline to load; it also opens its dataset unless --dataset is given"),
            opt("steps", "json|@file|-", "json", "", "steps to append, a JSON array of {kind, preset?, params?, enabled?, name?}"),
            opt("set", "step.key=value", "string", "",
                "set a parameter; step is a number (1 = Load) or a name, the value JSON or text (repeatable)", true),
        };
        return specs;
    }

    const std::vector<OptionSpec>& openOptionSpecs() {
        static const std::vector<OptionSpec> specs = {
            opt("page-order", "czt", "string", "", "how the pages of a plain TIFF map to channels, time and z, fastest first"),
            opt("page-c", "N", "integer", "", "channels, for a plain TIFF"),
            opt("page-t", "N", "integer", "", "time points, for a plain TIFF"),
            opt("page-z", "N", "integer", "", "z planes, for a plain TIFF (default: from the page count)"),
            opt("voxel", "x,y,z", "list", "", "the voxel size in microns, instead of the file's"),
            opt("sim", "d,p[,fast]", "list", "", "raw SIM data: directions and phases per plane (fast: z before direction)"),
            flag("no-sim", "not raw SIM data, whatever the file says"),
            opt("dataset-tile", "N", "integer", "", "the tile to serve, for a multi-file dataset"),
            flag("full-load", "read the whole dataset into memory now (default: read planes as needed)"),
        };
        return specs;
    }

    const std::vector<CommandSpec>& commandSpecs() {
        static const std::vector<CommandSpec> specs = [] {
            std::vector<CommandSpec> c;
            c.push_back(command("version", "version", "the version, the output schema, the protocols and what this build can do"));
            c.push_back(command("devices", "devices", "the compute backend and the CUDA devices"));
            {
                CommandSpec s = command("info", "info <dataset> [open options] [--open]", "dimensions, pixel type, voxel size, channels (reads no pixels)");
                s.positional = "<dataset>";
                s.minPositionals = s.maxPositionals = 1;
                s.openOptions = true;
                s.options = {flag("open", "open the dataset (lazily) for its metadata summary, not only probe it")};
                c.push_back(std::move(s));
            }
            {
                CommandSpec s = command("ops", "ops [kind...] [--group G] [--detail] [--plugins]", "operations that can be steps, with their parameters");
                s.positional = "[kind...]";
                s.maxPositionals = -1;
                s.options = {opt("group", "G", "string", "", "only the operations of this group"),
                             flag("detail", "the parameters with their schemas; with one kind, its full description"),
                             flag("plugins", "also the user operations of the Python worker (starts it)")};
                c.push_back(std::move(s));
            }
            {
                CommandSpec s = command("help", "help [page] [--markdown]", "help pages (Markdown); --markdown prints the page itself");
                s.positional = "[page]";
                s.maxPositionals = 1;
                s.options = {flag("markdown", "print the page itself instead of JSON")};
                c.push_back(std::move(s));
            }
            {
                CommandSpec s = command("validate", "validate [state options]", "check the pipeline against the dataset without running it");
                s.state = true;
                c.push_back(std::move(s));
            }
            {
                CommandSpec s = command("run",
                                        "run [--to S] [--force] [--render P [render options]]... [--export P [export options]]... "
                                        "[--stats [stats options]] [--save-pipeline T] [--export-python PY]",
                                        "run the pipeline; then --render / --export / --stats");
                s.state = true;
                s.options = {opt("to", "S", "string", "", "run up to this step (default: every enabled step)"),
                             flag("force", "clear every cached output first")};
                s.actions = {
                    {opt("render", "P", "path", "", "then draw an image into P (PNG or JPEG); the render options follow", true),
                     with(renderOptions(), {flag("base64", "also put the image in the result, base64-encoded")})},
                    {opt("export", "P", "path", "", "then write the output to P; the export options follow", true), exportOptions()},
                    {flag("stats", "then measure the output; the stats options follow"), statsOptions()},
                    {opt("save-pipeline", "T", "path", "", "then save the pipeline to T (.sirius.toml)"), {}},
                    {opt("export-python", "PY", "path", "", "then write the pipeline as a Python script"), {}},
                };
                c.push_back(std::move(s));
            }
            {
                CommandSpec s = command("render", "render --out <png> [render options] [--base64] [--no-run]",
                                        "a slice (xy, xz, yz) or a maximum projection of a step, as PNG");
                s.state = true;
                s.options = {opt("out", "P", "path", "", "where to write the image (never over an existing file that is not an image)")};
                for (const OptionSpec& o : renderOptions()) s.options.push_back(o);
                s.options.push_back(flag("base64", "also put the image in the result, base64-encoded"));
                s.options.push_back(noRun());
                c.push_back(std::move(s));
            }
            {
                CommandSpec s = command("stats", "stats [stats options] [--no-run]", "per-channel intensity statistics and label statistics");
                s.state = true;
                s.options = with(statsOptions(), {noRun()});
                c.push_back(std::move(s));
            }
            {
                CommandSpec s = command("diagnostics", "diagnostics [--step S] [--detail] [--images DIR] [--no-run]", "a step's diagnostics (tables, curves, images)");
                s.state = true;
                s.options = {stepOption("to look at"), flag("detail", "the curves with their points and the histograms with their bins"),
                             opt("images", "DIR", "path", "", "also write every diagnostic image into DIR as PNG"), noRun()};
                c.push_back(std::move(s));
            }
            {
                CommandSpec s = command("export", "export --out <path> [export options] [--no-run]", "write a step's output (OME-TIFF, TIFF, zarr, N5, raw)");
                s.state = true;
                s.options = {opt("out", "P", "path", "", "the file (or zarr / N5 folder) to write")};
                for (const OptionSpec& o : exportOptions()) s.options.push_back(o);
                s.options.push_back(noRun());
                c.push_back(std::move(s));
            }
            {
                CommandSpec s = command("export-training",
                                        "export-training --dir D [--step S --sample N --slices --min-voxels N --image-dtype T --image-scaling S] [--no-run]",
                                        "write a step's labels as a training sample");
                s.state = true;
                s.options = {opt("dir", "D", "path", "", "the dataset folder; each call adds one sample"),
                             stepOption("whose labels to write"),
                             opt("sample", "N", "string", "", "the sample folder's name (default the dataset's)"),
                             flag("slices", "also one 8-bit image and one YOLO file per z plane"),
                             opt("min-voxels", "N", "integer", "1", "leave out smaller objects"),
                             opt("image-dtype", "uint8|uint16|float32", "enum", "uint16", "the pixel type of image.tif"),
                             opt("image-scaling", "cast|minmax|percentile", "enum", "percentile", "how the image is mapped into that type"),
                             noRun()};
                c.push_back(std::move(s));
            }
            {
                CommandSpec s = command("export-python", "export-python --out PY", "write the pipeline as a Python script");
                s.state = true;
                s.options = {opt("out", "PY", "path", "", "the script to write")};
                c.push_back(std::move(s));
            }
            {
                CommandSpec s = command("call", "call <tool> [--args JSON|@file|-] [--<param> value]...", "call one tool of the tool API (the tools MCP serves)");
                s.positional = "<tool>";
                s.state = true;
                s.options = {opt("args", "JSON|@file|-", "json", "", "the arguments as one JSON object; --<param> flags override its keys")};
                c.push_back(std::move(s));
            }
            {
                CommandSpec s = command("tools", "tools [--names]", "the tool list, as MCP's tools/list gives it");
                s.options = {flag("names", "only the names")};
                c.push_back(std::move(s));
            }
            c.push_back(command("schema", "schema", "the commands, their options, the exit codes and the envelope, as JSON"));
            {
                CommandSpec s = command("session", "session [--allow-worker-setup] [--allow-network-paths] [state options]", "a live workbench: JSON lines on stdin / stdout");
                s.state = true;
                s.options = {flag("allow-worker-setup", "let the setup_worker_env tool download packages (it still needs confirm:true)"),
                             flag("allow-network-paths", "let tools open network (UNC) paths such as //server/share")};
                c.push_back(std::move(s));
            }
            {
                CommandSpec s = command("mcp", "mcp [--allow-worker-setup] [--read-only] [--allow-network-paths] [state options]", "a Model Context Protocol server on stdio");
                s.state = true;
                s.options = {flag("allow-worker-setup", "let the setup_worker_env tool download packages (the client still asks the user)"),
                             flag("read-only", "leave out the tools that write files or download"),
                             flag("allow-network-paths", "let tools open network (UNC) paths such as //server/share")};
                c.push_back(std::move(s));
            }
            c.push_back(command("worker status", "worker status", "which Python the worker runs, and the state of SIRIUS's own environment"));
            c.push_back(command("worker check", "worker check", "start the Python worker (installs nothing) and say hello"));
            {
                CommandSpec s = command("worker setup",
                                        "worker setup [--yes] [--extras] [--package P]... [--base-python P] [--update|--recreate] [--no-uv] "
                                        "[--index-url U] [--find-links D]... [--no-index] [--dry-run]",
                                        "create or update SIRIUS's own Python environment for the worker (downloads numpy)");
                s.options = {flag("yes", "agree to the download without being asked"),
                             flag("extras", "also scipy and scikit-image (about 57 MB)"),
                             opt("package", "P", "string", "", "another package to install (repeatable)", true),
                             opt("base-python", "P", "path", "", "the Python 3 to build the environment from (default: found automatically)"),
                             flag("update", "install into the existing environment"),
                             flag("recreate", "build the environment anew (the old one stays until the new one works)"),
                             flag("no-uv", "use the standard venv and pip even when uv is installed"),
                             opt("index-url", "U", "string", "", "the package index (default pypi.org)"),
                             opt("find-links", "D", "path", "", "a directory or page of packages (repeatable)", true),
                             flag("no-index", "use no package index (with --find-links)"),
                             flag("dry-run", "print the plan and change nothing")};
                c.push_back(std::move(s));
            }
            {
                CommandSpec s = command("worker remove", "worker remove [--yes]", "delete SIRIUS's own Python environment");
                s.options = {flag("yes", "do not ask")};
                c.push_back(std::move(s));
            }
            return c;
        }();
        return specs;
    }

    const CommandSpec* findCommand(const std::string& name) {
        for (const CommandSpec& c : commandSpecs())
            if (c.name == name) return &c;
        return nullptr;
    }

    bool Action::has(const std::string& option) const { return hasIn(options, option); }

    std::string Action::value(const std::string& option, const std::string& def) const { return valueOf(options, option, def); }

    bool Args::has(const std::string& option) const { return hasIn(options, option); }

    std::string Args::value(const std::string& option, const std::string& def) const { return valueOf(options, option, def); }

    std::vector<std::string> Args::values(const std::string& option) const {
        std::vector<std::string> out;
        for (const Option& o : options)
            if (o.name == option) out.push_back(o.value);
        return out;
    }

    Args parseArgs(const std::vector<std::string>& argv) {
        Args a;
        const CommandSpec* cmd = nullptr;
        bool worker = false;                 // "worker" seen, its subcommand not yet
        bool optionsDone = false;            // after "--"
        std::size_t action = std::string::npos;   // the run action options go to
        std::vector<std::string> stateGiven, openGiven;

        for (std::size_t i = 0; i < argv.size(); ++i) {
            const std::string& arg = argv[i];
            // After `call <tool>` everything belongs to the tool; only a
            // help request is still ours.
            if (cmd && cmd->name == "call" && !a.tool.empty()) {
                if (arg == "--help" || arg == "-h") a.help = true;
                else a.toolArgs.push_back(arg);
                continue;
            }
            if (!optionsDone && arg == "--") {
                optionsDone = true;
                continue;
            }
            const bool isOption = !optionsDone && arg.size() > 1 && arg[0] == '-';
            if (!isOption) {
                if (!cmd && !worker) {
                    if (arg == "worker") {
                        worker = true;
                        a.command = "worker";
                        continue;
                    }
                    cmd = findCommand(arg);
                    if (!cmd) throw UsageError("unknown command '" + arg + "'", "sirius-cli --help lists the commands");
                    a.command = cmd->name;
                    continue;
                }
                if (worker && !cmd) {
                    cmd = findCommand("worker " + arg);
                    if (!cmd) throw UsageError("unknown worker command '" + arg + "'", "the worker commands are status, check, setup and remove");
                    a.command = cmd->name;
                    continue;
                }
                if (cmd && cmd->name == "call" && a.tool.empty()) {
                    a.tool = arg;
                    continue;
                }
                a.positionals.push_back(arg);
                continue;
            }

            std::string name;
            bool inlineGiven = false;
            std::string inlineValue;
            if (arg.compare(0, 2, "--") == 0) {
                name = arg.substr(2);
                const std::size_t eq = name.find('=');
                if (eq != std::string::npos) {
                    inlineValue = name.substr(eq + 1);
                    name.resize(eq);
                    inlineGiven = true;
                }
            } else if (arg == "-h") {
                name = "help";
            } else {
                throw UsageError("unknown option " + arg, "sirius-cli --help lists the options");
            }
            if (name == "help") {
                a.help = true;
                continue;
            }
            if (name == "version" && !cmd) {
                a.version = true;
                continue;
            }
            if (name == "version" && cmd->name != "call") {
                // After the command word it changes nothing; said, like the
                // other options a command has no use for, not refused.
                a.warnings.push_back("--version after the command word is ignored; `sirius-cli version` reports the version");
                continue;
            }
            if (cmd && cmd->name == "call")
                throw UsageError("--" + name + " between `call` and the tool name",
                                 "global and state options go before `call`; the tool's own after its name");

            enum class Where { None,
                               ActionFlag,
                               ActionOption,
                               Own,
                               Global,
                               State,
                               Open };
            Where where = Where::None;
            const OptionSpec* spec = nullptr;
            if (cmd) {
                if (const ActionSpec* as = findAction(*cmd, name)) {
                    spec = &as->flag;
                    where = Where::ActionFlag;
                } else if (action != std::string::npos) {
                    if ((spec = findIn(findAction(*cmd, a.actions[action].name)->options, name))) where = Where::ActionOption;
                }
                if (!spec && (spec = findIn(cmd->options, name))) where = Where::Own;
            }
            if (!spec && (spec = findIn(globalOptionSpecs(), name))) where = Where::Global;
            if (!spec && (spec = findIn(stateOptionSpecs(), name))) where = Where::State;
            if (!spec && (spec = findIn(openOptionSpecs(), name))) where = Where::Open;
            if (!spec) {
                if (!cmd) throw UsageError("unknown option --" + name, "a command's own options come after the command; sirius-cli --help lists the global ones");
                if (action != std::string::npos)
                    throw UsageError("unknown option --" + name + " for --" + a.actions[action].name + " in run", "sirius-cli run --help lists them");
                throw UsageError("unknown option --" + name + " for " + cmd->name, "sirius-cli " + cmd->name + " --help lists them");
            }

            std::string value = "true";
            if (spec->flag()) {
                if (inlineGiven) throw UsageError("--" + name + " takes no value");
            } else if (inlineGiven) {
                value = inlineValue;
            } else {
                if (i + 1 >= argv.size()) throw UsageError("--" + name + " needs a value (" + spec->value + ")");
                value = argv[++i];
            }

            switch (where) {
                case Where::ActionFlag:
                    a.actions.push_back({name, spec->flag() ? std::string() : value, {}});
                    action = a.actions.size() - 1;
                    break;
                case Where::ActionOption: a.actions[action].options.push_back({name, value}); break;
                case Where::Own: a.options.push_back({name, value}); break;
                case Where::Global: setGlobal(a.global, name, value); break;
                case Where::State:
                    setState(a.state, name, value);
                    stateGiven.push_back(name);
                    break;
                case Where::Open:
                    a.state.open.push_back({name, value});
                    openGiven.push_back(name);
                    break;
                case Where::None: break;
            }
        }

        if (a.help) return a;   // whatever else is missing, the help can be shown
        if (worker && !cmd) throw UsageError("worker needs a command: status, check, setup or remove");
        if (cmd && cmd->name == "call" && a.tool.empty()) throw UsageError("call needs a tool name", "sirius-cli tools --names lists them");
        if (cmd) {
            const int n = static_cast<int>(a.positionals.size());
            if (n < cmd->minPositionals) throw UsageError(cmd->name + " needs " + cmd->positional, "sirius-cli " + cmd->name + " --help");
            if (cmd->maxPositionals >= 0 && n > cmd->maxPositionals)
                throw UsageError("unexpected argument '" + a.positionals[static_cast<std::size_t>(cmd->maxPositionals)] + "' for " + cmd->name,
                                 "sirius-cli " + cmd->name + " --help");
            // State and open options that the command has no use for are
            // reported, not refused: a wrapper script may pass them to every
            // command it runs.
            if (!cmd->state) {
                for (const std::string& s : stateGiven) a.warnings.push_back("--" + s + " does not apply to " + cmd->name + "; ignored");
                if (!cmd->openOptions)
                    for (const std::string& s : openGiven) a.warnings.push_back("--" + s + " does not apply to " + cmd->name + "; ignored");
            }
        } else if (!stateGiven.empty() || !openGiven.empty()) {
            throw UsageError("no command given", "sirius-cli --help lists the commands");
        }
        if (a.global.backend == "hpc" && a.global.hpcHost.empty())
            throw UsageError("--backend hpc needs --hpc host:port", "the HPC endpoint is fixed when sirius-cli starts");
        return a;
    }

} // namespace sirius::cli
