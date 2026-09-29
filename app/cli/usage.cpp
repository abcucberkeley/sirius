// sirius-cli's help texts: the top-level --help, each command's --help,
// `schema`, and what the MCP server tells a client about itself. The options
// themselves come from the table in args.cpp, so the texts list exactly what
// the parser accepts.

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cli/args.hpp"
#include "cli/commands.hpp"
#include "core/agent_protocol.hpp"

namespace sirius::cli {

    using json = nlohmann::json;
    namespace app = sirius::app;

    namespace {

        // What each command's --help says beyond its options.
        struct CommandText {
            const char* name;
            const char* description;
            const char* output;     // an example `result`
            const char* exitCodes;
            std::vector<const char*> examples;
        };

        const std::vector<CommandText>& commandTexts() {
            static const std::vector<CommandText> texts = {
                {"version",
                 "Prints the version, the schema of the JSON output, the protocol versions of `session` and `mcp`, what this build can do "
                 "(CUDA, zarr, the export formats, the file types it reads) and where it finds the help pages, the Python worker and "
                 "SIRIUS's own Python environment.",
                 R"({"name":"sirius-cli","version":"0.1.0","schema":"sirius-cli/1",
 "protocols":{"session":"sirius-session/1","mcp":["2026-07-28","2025-11-25","2025-06-18","2025-03-26","2024-11-05"]},
 "features":{"cuda":false,"cuda_devices":0,"zarr":true,"export_formats":["tiff","ome-tiff","zarr","n5","raw"],
             "readable_extensions":["tif","tiff","zarr"]},
 "paths":{"executable_dir":"/opt/sirius/bin","help":"/opt/sirius/share/sirius/help",
          "worker":"/opt/sirius/share/sirius/python","python_env":"/home/me/.local/share/sirius/python-env"}})",
                 "0 ok",
                 {"sirius-cli version"}},
                {"devices",
                 "The compute backend runs would use and the CUDA devices of this machine.",
                 R"({"backend":"CUDA","cuda_available":true,"cuda_device":0,
 "devices":[{"index":0,"name":"NVIDIA RTX 4000 Ada","memory_gb":20.0,"compute":"8.9"}]})",
                 "0 ok, 2 usage",
                 {"sirius-cli devices"}},
                {"info",
                 "Probes a dataset without reading its pixels: dimensions, pixel type, size on disk, voxel size, channels, the SIM layout "
                 "and the tiles. The open options say how the pages of a plain TIFF map to channels, time and z. With --open the "
                 "dataset is opened (lazily), which adds the metadata summary.",
                 R"({"path":"C:/data/raw.tif","name":"raw","format":"tiff","shape":"c1 t1 z120 y256 x256",
 "dims":{"c":1,"t":1,"z":120,"y":256,"x":256},"dtype":"uint16","bytes_on_disk":15728640,"voxel_um":[0.08,0.08,0.125],
 "channels":[{"index":0,"label":"488","wavelength_nm":488,"color":"#00ff00"}],"sim":{"present":false}})",
                 "0 ok, 2 usage, 3 not_found / open_failed",
                 {"sirius-cli info tests/data/raw.tif", "sirius-cli info stack.tif --page-order zct --page-c 2 --voxel 0.1,0.1,0.3"}},
                {"ops",
                 "Lists the operations that can be steps: kind, name, group, the number of parameters, the presets, and whether they "
                 "need the Python worker or a GPU. --detail adds each parameter's key, label, default and schema; with one kind, "
                 "--detail gives that operation's full description. --plugins also loads the user operations of the Python worker.",
                 R"({"operations":[{"kind":"contrast","name":"Contrast","group":"Adjust","params":4,"presets":["Auto"],
                 "produces_labels":false,"needs_labels":false,"needs_worker":false,"plugin":false,"gpu":false}]})",
                 "0 ok, 2 usage, 3 unknown_operation, 4 worker_unavailable (--plugins)",
                 {"sirius-cli ops", "sirius-cli ops sim --detail", "sirius-cli ops --group Segmentation"}},
                {"help",
                 "Without a page, the list of help pages. With one, the page as JSON (its Markdown in `markdown`); with --markdown the "
                 "page itself instead of JSON, and exit 3 when there is no such page. Pages are named after the operation kinds "
                 "(load, sim, contrast, ...).",
                 R"({"page":"load","title":"Load","path":"C:/sirius/help/load.md","exists":true,"markdown":"# Load\n...","truncated":false})",
                 "0 ok, 2 usage, 3 not_found / invalid_argument",
                 {"sirius-cli help", "sirius-cli help sim --markdown"}},
                {"validate",
                 "Checks every step of the pipeline against the dataset without running anything: errors and warnings, the input and "
                 "output shapes, the memory each step needs and whether it needs the Python worker. Exits 3 (validation) when a step "
                 "has errors; error.data then holds the whole report.",
                 R"({"ok":true,"has_dataset":true,"needs_worker":false,
 "steps":[{"step":2,"kind":"contrast","name":"Contrast","enabled":true,"errors":[],"warnings":[],
           "input_shape":"c1 t1 z120 y256 x256","output_shape":"c1 t1 z120 y256 x256","estimated_bytes":31457280}]})",
                 "0 ok, 2 usage, 3 validation / not_found / no_dataset",
                 {"sirius-cli validate --pipeline analysis.sirius.toml", "sirius-cli validate --dataset raw.tif --steps @steps.json"}},
                {"run",
                 "Runs the pipeline (every enabled step, or up to --to), then each action in the order written: --render draws an image "
                 "into a file, --export writes the output, --stats measures it, --save-pipeline and --export-python write the pipeline. "
                 "The options after an action flag belong to that action until the next action flag; --to and --force come before the "
                 "first one. A one-shot command computes from scratch each time: iterative work belongs in `session` or `mcp`.",
                 R"({"run":{"status":"succeeded","run_id":"r1","target_step":3,"seconds":12.4,"backend":"CPU",
        "steps":[{"step":2,"kind":"sim","name":"SIM reconstruction","state":"ran","seconds":9.1}],
        "output":{"step":3,"shape":"c1 t1 z16 y1024 x1024","fresh":true}},
 "renders":[{"step":3,"plane":"mip","width":1024,"height":1024,"format":"png","path":"C:/work/mip.png"}],
 "exports":[{"path":"C:/work/out.ome.tif","format":"tiff","dtype":"float32","bytes":67108864}],
 "statistics":{"step":3,"channels":[{"channel":0,"min":0.0,"max":812.5,"mean":41.2}]}})",
                 "0 ok, 1 run_failed / export_failed / io_error, 2 usage, 3 invalid input, 4 worker_unavailable, 124 timeout, 130 cancelled",
                 {"sirius-cli run --pipeline examples/sim_bundled.sirius.toml --render mip.png --plane mip --stats",
                  "sirius-cli --backend cpu run --dataset raw.tif --steps '[{\"kind\":\"contrast\"}]' --export out.ome.tif --dtype uint16",
                  "sirius-cli run --pipeline p.sirius.toml --render xy.png --z 10 --render xz.png --plane xz --save-pipeline done.sirius.toml"}},
                {"render",
                 "Draws a step's output the way the viewer does, into a PNG or JPEG file: an xy plane, an xz or yz reslice, or the "
                 "maximum projection along z; several z planes give a grid. It runs the steps it needs first unless --no-run. --out "
                 "is never written over an existing file that is not an image.",
                 R"({"step":3,"rendered_step":3,"fresh":true,"plane":"xy","t":0,"z":60,"width":1024,"height":1024,"factor":2,
 "pixel_um":[0.16,0.16],"channels":[{"index":0,"label":"488","color":"#00ff00","window":[112,3890,1.0]}],
 "labels_drawn":false,"format":"png","bytes":482113,"path":"C:/work/xy.png"})",
                 "0 ok, 1 io_error, 2 usage, 3 invalid_argument / not_computed / too_large, 4 worker_unavailable, 124 timeout, 130 cancelled",
                 {"sirius-cli render --dataset stack.tif --plane xz --y 256 --out xz.png",
                  "sirius-cli render --pipeline p.sirius.toml --step 3 --plane mip --window 0=100:4000 --out mip.png",
                  "sirius-cli render --dataset stack.tif --z 10,20,30,40 --layout channels --no-run --out grid.png"}},
                {"stats",
                 "Per-channel statistics of a step's output: minimum, maximum, mean, standard deviation, NaN count and percentiles "
                 "(from at most 4 Mi sampled values), a histogram on request, the fraction of saturated values for the Load step of "
                 "an integer dataset, and for labels their count, sizes, volumes, flags and classes. It runs the steps it needs first "
                 "unless --no-run.",
                 R"({"step":1,"fresh":true,"shape":"c1 t1 z120 y256 x256","sampled":false,
 "channels":[{"channel":0,"label":"488","min":98,"max":4095,"mean":412.5,"std":88.1,"count":7864320,"nan":0,
              "percentiles":{"1":200,"50":400,"99":900},"saturated_fraction":0.0}]})",
                 "0 ok, 2 usage, 3 invalid_argument / not_computed, 4 worker_unavailable, 124 timeout, 130 cancelled",
                 {"sirius-cli stats --dataset raw.tif --no-run", "sirius-cli stats --pipeline p.sirius.toml --t all --histogram 64"}},
                {"diagnostics",
                 "A step's diagnostics, what the application shows under its parameters: summary, facts, table, curves, histograms, "
                 "tabs and warnings. --detail adds the curves' points and the histogram bins; --images DIR writes every diagnostic "
                 "image (spectra, fits) into DIR as PNG. It runs the pipeline up to the step first unless --no-run.",
                 R"({"step":2,"summary":"3 directions, 5 phases","facts":{"Method":"Wiener"},"curves":[],"tabs":["Raw spectrum"],
 "image_files":[{"title":"Raw FFT - phase 1","tab":"Raw spectrum","index":0,"path":"C:/work/diag/step2-raw-spectrum-0.png"}]})",
                 "0 ok, 1 io_error, 2 usage, 3 unknown_step / no_dataset, 4 worker_unavailable, 124 timeout, 130 cancelled",
                 {"sirius-cli diagnostics --pipeline sim.sirius.toml --step 2 --images diag"}},
                {"export",
                 "Writes a step's output: OME-TIFF or TIFF (tiled, compressed, BigTIFF, pyramids), zarr 2 or 3 and N5 (in builds with "
                 "TensorStore), or raw. The format follows the file name unless --format says otherwise; --dtype and --scaling say "
                 "how the values are stored. It runs the steps it needs first unless --no-run.",
                 R"({"path":"C:/work/out.ome.tif","format":"tiff","dtype":"float32","shape":"c1 t1 z16 y1024 x1024",
 "files":["C:/work/out.ome.tif"],"bytes":67108864,"seconds":1.2})",
                 "0 ok, 1 export_failed, 2 usage, 3 invalid_argument / unsupported / not_computed, 4 worker_unavailable, 124 timeout, "
                 "130 cancelled",
                 {"sirius-cli export --pipeline p.sirius.toml --out result.ome.tif --dtype uint16 --scaling minmax",
                  "sirius-cli export --dataset raw.tif --out raw.zarr --chunk 1,1,16,256,256 --codec zstd --no-run"}},
                {"export-training",
                 "Writes a step's labels as one training sample into a dataset folder: instance masks, a semantic mask, bounding boxes "
                 "and the image, and with --slices one 8-bit plane and one YOLO file per z. Every call adds a sample. It runs the "
                 "pipeline up to the step first unless --no-run.",
                 R"({"directory":"C:/train/raw","files":["instances.tif","semantic.tif","boxes.json","image.tif"],"objects":412,
 "slice_objects":0,"classes":1,"frames":1,"bytes":1234567})",
                 "0 ok, 1 failed, 2 usage, 3 invalid_argument / not_computed, 4 worker_unavailable, 124 timeout, 130 cancelled",
                 {"sirius-cli export-training --pipeline segment.sirius.toml --dir training --slices"}},
                {"export-python",
                 "Writes the pipeline as a Python script.",
                 R"({"path":"C:/work/pipeline.py"})",
                 "0 ok, 1 io_error, 2 usage",
                 {"sirius-cli export-python --pipeline p.sirius.toml --out pipeline.py"}},
                {"call",
                 "Calls one tool of the tool API, the same tools `mcp` and `session` serve (`sirius-cli tools` lists them with their "
                 "schemas). The arguments come from --args (a JSON object, @file, or - for stdin) and from flags named after the "
                 "tool's parameters: max_size is --max-size, arrays are comma lists or JSON, objects JSON, booleans --flag and "
                 "--no-flag. Global and state options go before `call`. An image is written into the scratch directory only (keep it "
                 "with --keep-scratch, or use `render --out`).",
                 R"(the tool's own result; an image adds "image":{"path","mime_type","width","height","bytes"})",
                 "0 ok, 2 usage / unknown_tool, and the tool's own error codes",
                 {"sirius-cli --dataset raw.tif call statistics --t all", "sirius-cli --dataset raw.tif --keep-scratch call render --z 10 --max-size 512",
                  "sirius-cli call describe_operation --kind sim", "sirius-cli --dataset raw.tif call add_step --args @step.json"}},
                {"tools",
                 "The tools `mcp` serves, as MCP's tools/list gives them: name, title, description, inputSchema, and the annotations "
                 "readOnlyHint, destructiveHint, idempotentHint and openWorldHint.",
                 R"([{"name":"open_dataset","title":"Open a dataset","description":"...","inputSchema":{"type":"object","properties":{...}},
  "annotations":{"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":false}}])",
                 "0 ok, 2 usage",
                 {"sirius-cli tools --names", "sirius-cli tools"}},
                {"schema",
                 "Every command with its synopsis and options, the global, state and open options, the exit codes, each error code "
                 "with its exit code, and the envelope: for programs that drive sirius-cli.",
                 R"({"commands":[{"name":"render","synopsis":"render --out <png> ...","options":[{"name":"--out","type":"path",...}]}],
 "exit_codes":{"0":"ok",...},"error_codes":{"not_found":3,...},"envelope":{...}})",
                 "0 ok",
                 {"sirius-cli schema"}},
                {"session",
                 "A live workbench on JSON lines (sirius-session/1). Each request {\"id\":1,\"method\":\"<tool or control>\",\"params\":{...}} "
                 "gets one response, {\"id\":1,\"ok\":true,\"result\":{...}} or {\"id\":1,\"ok\":false,\"error\":{...}}; the first line is "
                 "{\"event\":\"ready\",...}. Control methods: tools, status, cancel, subscribe, ping, shutdown. Events: progress, log "
                 "(after subscribe) and run_finished. At the end of input the queued requests are answered and a running run is waited "
                 "for (--timeout bounds the wait), so `sirius-cli session < script.jsonl` works as a batch script.",
                 R"((the protocol) {"event":"ready","protocol":"sirius-session/1","version":"0.1.0","workspace":"ws_1a2b3c4d5e6f","tools":39})",
                 "0 at the end of input or after shutdown, 2 usage, 3 / 4 when the state options fail, 1 fatal",
                 {"sirius-cli session --dataset raw.tif < script.jsonl",
                  "echo '{\"id\":1,\"method\":\"get_state\"}' | sirius-cli session"}},
                {"mcp",
                 "A Model Context Protocol server on stdio (JSON-RPC, one message per line) with one live workspace. It offers tools "
                 "only, and speaks 2025-11-25, 2025-06-18, 2025-03-26 and 2024-11-05 through initialize, and 2026-07-28 per request. "
                 "With --allow-worker-setup the setup_worker_env tool may download packages (the client still asks the user each "
                 "time); --read-only leaves out the tools that write files or download. Logs go to stderr.",
                 R"((the protocol) {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{"tools":{}},
 "serverInfo":{"name":"sirius","title":"SIRIUS microscopy workbench","version":"0.1.0"},"instructions":"..."}})",
                 "0 at the end of input or on SIGTERM, 2 usage, 3 / 4 when the state options fail, 1 fatal",
                 {"claude mcp add --transport stdio --scope user sirius -- /opt/sirius/bin/sirius-cli mcp",
                  "npx @modelcontextprotocol/inspector sirius-cli mcp"}},
                {"worker status",
                 "Which Python the worker would run and why (explicit, environment, managed, discovered, fallback), the state of "
                 "SIRIUS's own environment (absent, incomplete, ready, outdated, broken), uv, the Python interpreters found on this "
                 "machine, the worker's directory and its requirements. It starts nothing.",
                 R"({"interpreter":{"path":"C:/Users/me/AppData/Local/sirius/python-env/Scripts/python.exe","source":"managed"},
 "environment":{"state":"ready","dir":"C:/Users/me/AppData/Local/sirius/python-env"},"uv":{"path":"C:/tools/uv.exe","version":"uv 0.12.18"},
 "candidates":["C:/Python313/python.exe"],"worker_dir":"C:/sirius/python",
 "requirements":{"required":["numpy"],"extras":["scikit-image","scipy"]}})",
                 "0 ok",
                 {"sirius-cli worker status"}},
                {"worker check",
                 "Starts the Python worker the way a run would (it installs nothing), waits for its hello and stops it. Exits 4 "
                 "(worker_unavailable) when it cannot start: error.data says why (missing packages, no interpreter, a broken "
                 "environment) and error.data.fix what to run.",
                 R"({"interpreter":{"path":"/home/me/.local/share/sirius/python-env/bin/python","source":"managed"},
 "capabilities":{"version":"0.1.0","protocol":3,"methods":["plugins"],"cuda":false,"device":"cpu","hostname":"node1",
                 "python":"3.12.3"},"seconds":0.8})",
                 "0 ok, 4 worker_unavailable, 124 timeout, 130 cancelled",
                 {"sirius-cli worker check", "sirius-cli --python /opt/conda/bin/python worker check"}},
                {"worker setup",
                 "Creates SIRIUS's own Python environment for the worker (a venv in your data folder, or in $SIRIUS_PYTHON_ENV) from "
                 "a Python 3.9 or newer found on this machine, and downloads what the worker needs from pypi.org: numpy, and with "
                 "--extras also scipy and scikit-image. It uses uv when uv is installed. Without --yes it asks on a terminal, and "
                 "anywhere else exits 5 (consent_required) with the plan in error.data. --dry-run prints the plan. Ctrl+C cancels "
                 "and leaves the previous environment as it was.",
                 R"({"env_dir":"C:/Users/me/AppData/Local/sirius/python-env",
 "python":"C:/Users/me/AppData/Local/sirius/python-env/Scripts/python.exe","base_python":"C:/Python313/python.exe",
 "python_version":"3.13.7","installer":"uv 0.12.18","mode":"create","packages":{"numpy":"2.5.3","pip":"26.2.1"},
 "extras":false,"seconds":4.1})",
                 "0 ok, 1 failed (offline, no wheel, disk full, ...), 2 usage, 3 busy, 4 python_not_found, 5 consent_required, "
                 "130 cancelled",
                 {"sirius-cli worker setup --yes", "sirius-cli worker setup --dry-run",
                  "sirius-cli worker setup --yes --extras --base-python /usr/bin/python3.12"}},
                {"worker remove",
                 "Deletes SIRIUS's own Python environment. Without --yes it asks on a terminal, and anywhere else exits 5 "
                 "(consent_required). The worker then runs $SIRIUS_PYTHON, or a Python on PATH, until the environment is set up "
                 "again.",
                 R"({"removed":"C:/Users/me/AppData/Local/sirius/python-env"})",
                 "0 ok, 1 failed, 3 busy, 5 consent_required",
                 {"sirius-cli worker remove --yes"}},
            };
            return texts;
        }

        const CommandText* findText(const std::string& name) {
            for (const CommandText& t : commandTexts())
                if (name == t.name) return &t;
            return nullptr;
        }

        // Words into lines of at most `width` characters, each starting with `indent`.
        std::string wrap(const std::string& text, std::size_t width, const std::string& indent) {
            std::istringstream words(text);
            std::string word, line, out;
            while (words >> word) {
                if (!line.empty() && indent.size() + line.size() + 1 + word.size() > width) {
                    out += indent + line + "\n";
                    line.clear();
                }
                line += (line.empty() ? "" : " ") + word;
            }
            if (!line.empty()) out += indent + line + "\n";
            return out;
        }

        std::string optionLeft(const OptionSpec& o) { return "--" + o.name + (o.flag() ? std::string() : " <" + o.value + ">"); }

        std::string optionLines(const std::vector<OptionSpec>& options, const std::string& indent) {
            std::string out;
            for (const OptionSpec& o : options) {
                std::string left = indent + optionLeft(o);
                std::string help = o.help;
                if (!o.def.empty()) help += " (default " + o.def + ")";
                if (left.size() + 2 > 32) {
                    out += left + "\n";
                    left.clear();
                }
                left.resize(32, ' ');
                out += left + help + "\n";
            }
            return out;
        }

        // The options, compactly, as the top-level help lists them: a choice
        // as its alternatives (--backend auto|cpu), anything else as <value>.
        // An option is never split over two lines.
        std::string optionSummary(const std::vector<OptionSpec>& options, const std::string& lead, bool skipHelp) {
            const std::string indent = "  ";
            const std::size_t width = 104;
            std::string out, line = indent + lead;
            for (const OptionSpec& o : options) {
                if (skipHelp && (o.name == "help" || o.name == "version")) continue;
                const std::string item = "--" + o.name + (o.flag() ? "" : o.type == "enum" ? " " + o.value
                                                                                           : " <" + o.value + ">");
                if (line.size() > indent.size() && line.size() + 1 + item.size() > width) {
                    out += line + "\n";
                    line = indent + "  ";
                }
                if (line.size() > indent.size() && line.back() != ' ') line += ' ';
                line += item;
            }
            return out + line + "\n";
        }

        json optionJson(const OptionSpec& o) {
            json j = {{"name", "--" + o.name},
                      {"value", o.flag() ? json(nullptr) : json(o.value)},
                      {"type", o.type},
                      {"default", o.def.empty() ? json(nullptr) : json(o.def)},
                      {"help", o.help}};
            if (o.repeatable) j["repeatable"] = true;
            return j;
        }

        json optionsJson(const std::vector<OptionSpec>& options) {
            json out = json::array();
            for (const OptionSpec& o : options) out.push_back(optionJson(o));
            return out;
        }

    } // namespace

    const std::vector<std::string>& errorCodes() {
        static const std::vector<std::string> codes = {
            "failed", "run_failed", "export_failed", "io_error", "internal",
            "usage", "unknown_tool",
            "not_found", "open_failed", "invalid_argument", "unknown_step", "unknown_operation", "no_dataset", "validation",
            "not_computed", "unsupported", "too_large", "busy", "stale_workspace",
            "worker_unavailable", "python_not_found",
            "consent_required",
            "timeout",
            "cancelled"};
        return codes;
    }

    std::string usageText() {
        std::string s =
            "sirius-cli - the SIRIUS microscopy workbench without a window, for scripts and agents\n"
            "\n"
            "Usage: sirius-cli [global options] <command> [options]\n"
            "\n"
            "Every command prints one JSON document on stdout ({\"ok\": ..., \"result\" | \"error\": ...});\n"
            "progress and log lines go to stderr.\n"
            "\n"
            "Data and operations\n"
            "  info <dataset>          dimensions, pixel type, voxel size, channels (reads no pixels)\n"
            "  ops [kind...]           operations that can be steps, with their parameters\n"
            "  help [page]             help pages (Markdown); --markdown prints the page itself\n"
            "Pipelines\n"
            "  validate                check the pipeline against the dataset without running it\n"
            "  run                     run the pipeline; then --render / --export / --stats\n"
            "  render --out <png>      a slice (xy, xz, yz) or a maximum projection of a step, as PNG\n"
            "  stats                   per-channel intensity statistics and label statistics\n"
            "  diagnostics             a step's diagnostics (tables, curves, images)\n"
            "  export --out <path>     write a step's output (OME-TIFF, TIFF, zarr, N5, raw)\n"
            "  export-training --dir   write a step's labels as a training sample\n"
            "  export-python --out     write the pipeline as a Python script\n"
            "Tools and servers\n"
            "  call <tool>             call one tool of the tool API (the tools MCP serves)\n"
            "  tools | schema          the tool list (MCP format) | commands, options and exit codes\n"
            "  session                 a live workbench: JSON lines on stdin / stdout\n"
            "  mcp                     a Model Context Protocol server on stdio\n"
            "Python worker\n"
            "  worker status | check | setup | remove    SIRIUS's own Python environment for the worker\n"
            "Other\n"
            "  devices | version\n"
            "\n"
            "Global options (before or after the command; before `call`):\n";
        s += optionSummary(globalOptionSpecs(), "", true);
        s += "State options (validate, run, render, stats, diagnostics, export*, call, session, mcp):\n";
        s += optionSummary(stateOptionSpecs(), "", false);
        s += optionSummary(openOptionSpecs(), "with --dataset, and for info: ", false);
        s += "\n"
             "Exit codes: 0 ok, 1 failed, 2 usage, 3 invalid input, 4 Python worker unavailable,\n"
             "  5 consent required, 124 timed out, 130 interrupted or terminated (session and mcp end 0)\n"
             "\n"
             "Examples:\n"
             "  sirius-cli info tests/data/raw.tif\n"
             "  sirius-cli run --pipeline examples/sim_bundled.sirius.toml --render mip.png --plane mip --stats\n"
             "  sirius-cli render --dataset stack.tif --plane xz --y 256 --out xz.png\n"
             "  sirius-cli --dataset raw.tif call statistics --t all\n"
             "  sirius-cli worker setup --yes\n"
             "  claude mcp add sirius -- sirius-cli mcp\n"
             "\n"
             "`sirius-cli <command> --help` describes a command; `sirius-cli schema` gives all of it as JSON.\n";
        return s;
    }

    std::string commandHelp(const std::string& command) {
        if (command == "worker") {
            std::string s = "sirius-cli worker - SIRIUS's own Python environment for the worker\n\nUsage: sirius-cli [global options] worker <command>\n\n";
            for (const CommandSpec& c : commandSpecs()) {
                if (c.name.compare(0, 7, "worker ") != 0) continue;
                std::string left = "  " + c.name.substr(7);
                left.resize(12, ' ');
                s += left + c.summary + "\n";
            }
            s += "\n`sirius-cli worker <command> --help` describes one.\n";
            return s;
        }
        const CommandSpec* c = findCommand(command);
        if (!c) return {};
        const CommandText* t = findText(command);
        std::string s = "sirius-cli " + c->name + " - " + c->summary + "\n\nUsage: sirius-cli [global options] " + c->synopsis + "\n\n";
        if (t) s += wrap(t->description, 100, "") + "\n";
        if (!c->options.empty()) s += "Options:\n" + optionLines(c->options, "  ");
        for (const ActionSpec& a : c->actions) {
            s += "\n" + optionLines({a.flag}, "  ");
            if (!a.options.empty()) s += optionLines(a.options, "      ");
        }
        if (c->state) {
            s += "\nState options (applied in this order: --pipeline, --dataset, --steps, --set):\n" + optionLines(stateOptionSpecs(), "  ");
            s += "Open options (with --dataset):\n" + optionLines(openOptionSpecs(), "  ");
        } else if (c->openOptions) {
            s += "\nOpen options:\n" + optionLines(openOptionSpecs(), "  ");
        }
        s += "\nGlobal options: see sirius-cli --help.\n";
        if (t) {
            s += "\nOutput (\"result\" of the envelope):\n";
            std::istringstream lines(t->output);
            for (std::string line; std::getline(lines, line);) s += "  " + line + "\n";
            s += "\nExit codes: " + std::string(t->exitCodes) + "\n\nExamples:\n";
            for (const char* e : t->examples) s += "  " + std::string(e) + "\n";
        }
        return s;
    }

    json schemaJson() {
        json commands = json::array();
        for (const CommandSpec& c : commandSpecs()) {
            json j = {{"name", c.name},
                      {"synopsis", c.synopsis},
                      {"summary", c.summary},
                      {"positional", c.positional.empty() ? json(nullptr) : json(c.positional)},
                      {"state_options", c.state},
                      {"open_options", c.state || c.openOptions},
                      {"options", optionsJson(c.options)}};
            if (!c.actions.empty()) {
                json actions = json::array();
                for (const ActionSpec& a : c.actions) {
                    json aj = optionJson(a.flag);
                    aj["options"] = optionsJson(a.options);
                    actions.push_back(std::move(aj));
                }
                j["actions"] = std::move(actions);
            }
            if (const CommandText* t = findText(c.name)) j["exit_codes"] = t->exitCodes;
            commands.push_back(std::move(j));
        }
        json errors = json::object();
        for (const std::string& code : errorCodes()) errors[code] = app::agent::exitCodeFor(code);
        return {
            {"schema", kOutputSchema},
            {"usage", "sirius-cli [global options] <command> [options]"},
            {"placement",
             "Global and state options may come before or after the command word, except with call: after `call <tool>` every "
             "--x is a parameter of that tool. After the command word its own options are looked up first. Inside run, the "
             "options after --render P, --export P or --stats belong to that action until the next action flag. Every "
             "JSON-valued option accepts @file or - (stdin)."},
            {"commands", std::move(commands)},
            {"global_options", optionsJson(globalOptionSpecs())},
            {"state_options", optionsJson(stateOptionSpecs())},
            {"open_options", optionsJson(openOptionSpecs())},
            {"exit_codes",
             {{"0", "ok"},
              {"1", "the operation failed"},
              {"2", "usage"},
              {"3", "invalid input, not found, or cannot run as asked"},
              {"4", "the Python worker is unavailable"},
              {"5", "consent required"},
              {"124", "timed out"},
              {"130", "interrupted, terminated (SIGTERM, SIGHUP, the console closed) or cancelled; session and mcp end 0 on termination"}}},
            {"error_codes", std::move(errors)},
            {"envelope",
             {{"success", {{"ok", true}, {"schema", kOutputSchema}, {"command", "<command>"}, {"result", "<the command's result>"}, {"warnings", json::array()}}},
              {"failure",
               {{"ok", false},
                {"schema", kOutputSchema},
                {"command", "<command>"},
                {"error", {{"code", "<error code>"}, {"message", "<what went wrong>"}, {"hint", "<what to do>"}, {"data", "<details or null>"}}},
                {"exit_code", "<the process exit code>"},
                {"warnings", json::array()}}},
              {"notes",
               "stdout carries exactly one JSON document per command (help --markdown prints the page instead). Keys are sorted; "
               "NaN and infinity are written as null, with a warning; paths are absolute, with forward slashes. A failure is "
               "followed by one line on stderr: sirius-cli: <code>: <message>."}}},
        };
    }

    std::string mcpInstructions() {
        return "SIRIUS: a microscopy workbench (3D/4D TIFF, OME-TIFF, zarr; SIM reconstruction, deconvolution, deskew,\n"
               "contrast, stitching, registration, segmentation, tracking). One live workspace per server.\n"
               "Workflow: open_dataset(path) or load_pipeline(path) -> list_operations / describe_operation / get_help\n"
               "-> add_step, set_params, apply_preset (every edit is undoable: undo / redo) -> validate -> run (returns\n"
               "status \"running\" after wait_s; then poll run_status) -> look with render (an image you can see; at most\n"
               "1024 px unless max_size says otherwise), statistics, probe, get_diagnostics -> export_result /\n"
               "save_pipeline. Step 1 is always Load; address steps by number (1 = Load) or name. Relative paths resolve\n"
               "against the server's working directory (get_state.cwd); prefer absolute paths. Steps that need Python\n"
               "(segmentation models, scikit-image, btrack tracking, plugins) use the Python worker: if a tool reports\n"
               "worker_unavailable, ask the user to run `sirius-cli worker setup --yes` (it downloads numpy from\n"
               "pypi.org) instead of working around it.";
    }

} // namespace sirius::cli
