#ifndef SIRIUS_APP_OPERATION_HPP
#define SIRIUS_APP_OPERATION_HPP

// A processing operation: a kind ("sim", "einsum", ...), its parameter
// specs, and pure functions from (input meta, params) to a summary, a
// validation and the output meta -- so the ops dock and the viewer toolbar
// can describe a step before it ever runs -- plus run(), which does the work
// on the calling (worker) thread and reports progress through the context.
//
// Operations are registered once at start-up (registerBuiltinOperations) and
// looked up by kind; nothing in the UI knows a kind by name except the few
// bespoke parameter editors.

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include <sirius/device.hpp>

#include "core/array.hpp"
#include "core/dataset.hpp"
#include "core/diagnostics.hpp"
#include "core/labels.hpp"
#include "core/params.hpp"

namespace sirius::app {

    class ArraySource;
    class RemoteWorker;

    enum class Backend { Cuda,
                         Cpu,
                         Hpc };
    const char* toString(Backend b) noexcept;   // "CUDA" "CPU" "HPC"
    std::optional<Backend> backendFromString(const std::string& s) noexcept;

    enum class CachePolicy { Memory,
                             Disk,
                             Recompute };
    const char* toString(CachePolicy c) noexcept;   // "memory" "disk" "recompute"
    std::optional<CachePolicy> cachePolicyFromString(const std::string& s) noexcept;

    // A named set of parameter values for one operation: a starting point for
    // a kind of structure, not a mode. Applying one writes the values into the
    // step and nothing else, so it is an ordinary undoable parameter change
    // and everything stays editable afterwards.
    struct ParamPreset {
        std::string name;            // "Filaments"
        std::string summary;         // what it is for, one line
        std::vector<std::pair<std::string, ParamValue>> values;
    };

    struct OpInfo {
        std::string kind;                 // "sim"
        std::string name;                 // "SIM reconstruction"
        std::string group;                // "Reconstruct" (menu group)
        std::string kindLabel;            // "RECONSTRUCT" (ops row caption)
        std::vector<ParamSpec> params;
        DiagnosticsKind diagnostics = DiagnosticsKind::Generic;
        CachePolicy defaultCache = CachePolicy::Recompute;
        bool separableOverT = false;      // may run one time point at a time
        bool hasGpuPath = false;          // honours Backend::Cuda
        bool remoteCapable = false;       // the Python worker implements it
        bool producesLabels = false;
        bool needsLabels = false;         // consumes the labels of its input
        // The viewer can show this step on its input without running it (a
        // display-level mapping such as Contrast): while the step is not
        // run or stale, the upstream output is displayed through the step's
        // current parameters and updates live as they change.
        bool livePreview = false;
        bool plugin = false;              // a user operation served by the Python worker
        std::string source;               // plugin file
        std::string helpPage;             // markdown file stem under app/help (defaults to kind)
        // Starting points offered by the panel and the apply_preset tool.
        std::vector<ParamPreset> presets;
    };

    struct Validation {
        std::vector<std::string> errors;      // the step cannot run
        std::vector<std::string> warnings;    // it can, but the user should know
        bool ok() const noexcept { return errors.empty(); }
        std::string firstError() const { return errors.empty() ? std::string() : errors.front(); }
    };

    // What a step receives: the upstream output.
    struct StepInput {
        DatasetMeta meta;
        ArrayPtr array;                            // null when only a lazy source exists
        std::shared_ptr<ArraySource> source;       // lazy planes (Load); null downstream
        LabelsPtr labels;

        bool hasArray() const noexcept { return array && !array->empty(); }
        // The array, read from the source when not yet in memory.
        ArrayPtr materialize(const std::function<void(double, const std::string&)>& progress = {}) const;
        // One (c, t) volume: from the array, or read from the source.
        Buffer<float> readVolume(Index c, Index t) const;
    };

    struct StepOutput {
        DatasetMeta meta;
        ArrayPtr array;
        std::shared_ptr<ArraySource> source;
        std::shared_ptr<LabelVolume> labels;      // new labels, or the input's carried through
        Diagnostics diagnostics;
        std::string note;                          // one line for the log ("41 s · plans reused")
        double seconds = 0.0;
        Backend ranOn = Backend::Cpu;

        StepInput asInput() const {
            return StepInput{meta, array, source, labels};
        }
    };

    struct StepContext {
        Backend backend = Backend::Cpu;
        Device device = Device::cpu();             // the CUDA device for Backend::Cuda
        RemoteWorker* remote = nullptr;            // Backend::Hpc
        // Hugging Face access token for a step that fetches a gated model
        // through the worker: sent with that request, never put in the
        // worker's environment (where pip and conda would inherit it).
        std::string hubToken;
        std::function<void(double fraction, const std::string& message)> progress;
        std::function<bool()> cancelled;
        std::filesystem::path scratchDir;          // per-session scratch (disk cache, worker files)

        void report(double fraction, const std::string& message = {}) const {
            if (progress) progress(fraction, message);
        }
        bool isCancelled() const { return cancelled && cancelled(); }
        void throwIfCancelled() const;             // std::runtime_error("cancelled")
    };

    class Operation {
    public:
        virtual ~Operation() = default;
        virtual const OpInfo& info() const noexcept = 0;

        // One line for the ops row ("3 angles · 5 phases · Wiener 0.001").
        virtual std::string summary(const ParamSet& params, const DatasetMeta& input) const;
        virtual Validation validate(const ParamSet& params, const DatasetMeta& input) const;
        // Shape / metadata the step will produce, without running it.
        virtual DatasetMeta outputMeta(const ParamSet& params, const DatasetMeta& input) const;
        // Rough cost, for the ops row and the cache tiles.
        virtual std::size_t estimatedOutputBytes(const ParamSet& params, const DatasetMeta& input) const;

        virtual StepOutput run(const StepInput& input, const ParamSet& params, const StepContext& ctx) const = 0;

        // Cheap diagnostics computed without running the step (the contrast
        // histograms update live while the percentiles are dragged). Default:
        // none; the workbench then shows a generic shape preview.
        virtual std::optional<Diagnostics> preview(const StepInput&, const ParamSet&) const { return std::nullopt; }
        // Parameters a freshly added step should start with, given its input
        // (Contrast takes its window from the data). Default: the defaults.
        virtual ParamSet initialParams(const ParamSet& defaults, const StepInput&) const { return defaults; }

        // Convenience for implementations.
        const std::string& kind() const noexcept { return info().kind; }
        ParamSet defaults() const { return ParamSet(info().params); }
    };

    // --- registry ----------------------------------------------------------
    void registerOperation(std::unique_ptr<Operation> op);   // replaces an existing kind
    const Operation* findOperation(const std::string& kind) noexcept;
    const Operation& requireOperation(const std::string& kind);   // throws std::out_of_range
    std::vector<const Operation*> allOperations();              // in registration order
    // Groups in menu order with their operations (Reconstruct, Reduce, ...).
    std::vector<std::pair<std::string, std::vector<const Operation*>>> operationGroups();
    // Registers every built-in operation (idempotent).
    void registerBuiltinOperations();
    // Every registered operation (built-ins first) with its parameter specs
    // as JSON: {"version", "operations": [{kind, name, group, plugin,
    // produces_labels, needs_labels, params: [{key, label, type, default,
    // choices, min, max, unit, advanced}]}]}. The Python mirror
    // (bindings/python/sirius/workbench.py) is checked against a committed
    // snapshot of it (bindings/python/sirius/op_schema.json), so a parameter
    // renamed here fails that test instead of silently changing a pipeline.
    nlohmann::json operationSchemas();

} // namespace sirius::app

#endif // SIRIUS_APP_OPERATION_HPP
