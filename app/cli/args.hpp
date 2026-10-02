#ifndef SIRIUS_CLI_ARGS_HPP
#define SIRIUS_CLI_ARGS_HPP

// sirius-cli's command line: the global and state options, the command, and
// what follows it. One table of options serves the parser, the help texts and
// `sirius-cli schema`, so the three cannot drift apart.
//
// Where options go (app/cli/README.md):
//   - global and state options may come before or after the command word,
//     except with `call`: after `call <tool>` every --x is a parameter of that
//     tool, so global and state options come before `call`;
//   - after the command word, the command's own options are looked up first
//     (`ops --plugins` is the ops flag, `--plugins off ops` the global one);
//   - inside `run`, the options after --render P, --export P or --stats belong
//     to that action until the next action flag.

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::cli {

    // A command line the parser or a command cannot use: exit 2, code "usage".
    class UsageError : public std::runtime_error {
    public:
        explicit UsageError(const std::string& message, std::string hint = {})
            : std::runtime_error(message), hint_(std::move(hint)) {}
        const std::string& hint() const noexcept { return hint_; }

    private:
        std::string hint_;
    };

    // One option as the parser, the help text and `schema` know it.
    struct OptionSpec {
        std::string name;          // without the dashes: "max-size"
        std::string value;         // what the value looks like ("N", "xy|xz|yz|mip"); "" for a flag
        std::string type;          // for schema: flag, string, integer, number, list, path, json, enum
        std::string def;           // the default as text; "" when there is none
        std::string help;
        bool repeatable = false;
        bool flag() const noexcept { return value.empty(); }
    };

    // An action of `run` (--render P, --export P, --stats, ...) and the
    // options that belong to it until the next action flag.
    struct ActionSpec {
        OptionSpec flag;
        std::vector<OptionSpec> options;
    };

    struct CommandSpec {
        std::string name;                   // "render", "worker setup"
        std::string synopsis;               // after "sirius-cli [global options] "
        std::string summary;                // one line
        std::string positional;             // "<dataset>", "[kind...]"; "" when it takes none
        int minPositionals = 0, maxPositionals = 0;   // -1 = any number
        bool state = false;                 // takes the state options (--dataset, --pipeline, ...)
        bool openOptions = false;           // takes the dataset open options on their own (info)
        std::vector<OptionSpec> options;    // its own
        std::vector<ActionSpec> actions;    // run
    };

    const std::vector<OptionSpec>& globalOptionSpecs();
    const std::vector<OptionSpec>& stateOptionSpecs();   // --dataset, --pipeline, --steps, --set
    const std::vector<OptionSpec>& openOptionSpecs();    // --page-order ... --full-load
    const std::vector<CommandSpec>& commandSpecs();
    const CommandSpec* findCommand(const std::string& name);

    // An option as written: a flag has the value "true".
    struct Option {
        std::string name, value;
    };

    struct Action {
        std::string name;                   // "render", "export", "stats", "save-pipeline", "export-python"
        std::string argument;               // the path after --render / --export / ...; "" for --stats
        std::vector<Option> options;
        bool has(const std::string& option) const;
        std::string value(const std::string& option, const std::string& def = {}) const;
    };

    struct GlobalOptions {
        std::string python;                 // --python
        std::string workerDir;              // --worker-dir
        std::string backend = "auto";       // auto | cpu | cuda | hpc
        int cudaDevice = 0;                 // -1 = all (Workbench::kAllCudaDevices)
        std::string hpcDevice = "gpu";      // gpu | cpu: where the HPC worker computes
        std::string hpcHost;                // --hpc host:port, split
        int hpcPort = 0;
        std::string plugins = "auto";       // auto | on | off
        std::string scratch;
        bool keepScratch = false;
        std::string record;
        std::optional<double> timeoutSeconds;
        std::string progress;               // none | text | json; "" = by whether stderr is a terminal
        std::optional<bool> pretty;         // --pretty / --compact; unset = by whether stdout is a terminal
        bool quiet = false;
    };

    struct StateOptions {
        std::string dataset, pipeline, steps;   // steps: as written (JSON, @file or -)
        std::vector<std::string> sets;          // --set <step>.<key>=<value>, in order
        std::vector<Option> open;               // the dataset open options, in order
        bool any() const noexcept { return !dataset.empty() || !pipeline.empty() || !steps.empty() || !sets.empty() || !open.empty(); }
    };

    struct Args {
        GlobalOptions global;
        StateOptions state;
        std::string command;                // "" when none was given; "worker setup" for the worker commands
        std::vector<std::string> positionals;
        std::vector<Option> options;        // the command's own, in order
        std::vector<Action> actions;        // run
        std::string tool;                   // call <tool>
        std::vector<std::string> toolArgs;  // everything after the tool name, as written
        bool help = false;                  // -h / --help
        bool version = false;               // --version before any command
        std::vector<std::string> warnings;  // options that were given but do not apply

        bool has(const std::string& option) const;
        std::string value(const std::string& option, const std::string& def = {}) const;
        std::vector<std::string> values(const std::string& option) const;
    };

    // Throws UsageError. `--help` anywhere (also after `call <tool>`) only
    // sets `help`: the command is still recognised, so its help can be shown.
    Args parseArgs(const std::vector<std::string>& argv);

} // namespace sirius::cli

#endif // SIRIUS_CLI_ARGS_HPP
