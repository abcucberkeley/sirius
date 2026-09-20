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
        // The plugin file as it was loaded (its size and modification time):
        // part of the step's fingerprint, so a reloaded edit is not served
        // the result of the code it replaced. Empty for built-ins.
        std::string sourceStamp;
        // A stand-in for a kind nothing loaded provides (registerMissingOperation):
        // it keeps a pipeline's step resolvable, fails validation with the
        // reason, and is left out of allOperations() and so of the add menu,
        // the assistant's tools and the schema.
        bool missing = false;
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
        // CUDA device for Backend::Cuda. index < 0 means every visible GPU:
        // volumes are round-robined across cuda:0..N-1.
        Device device = Device::cpu();
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
        bool allCudaDevices() const noexcept;
        // The device that should run volume (c, t). When allCudaDevices(),
        // this is cuda:((t * nChannels + c) % cudaDeviceCount()).
        Device deviceForVolume(Index c, Index t, Index nChannels) const;
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
        // Whether a run with these parameters needs the Python worker, which
        // a run then starts (or refuses to run without). Default: whether the
        // worker implements the operation at all (OpInfo::remoteCapable); a
        // step with a local method besides a remote one says which is chosen.
        virtual bool needsWorker(const ParamSet&) const { return info().remoteCapable; }

        // Convenience for implementations.
        const std::string& kind() const noexcept { return info().kind; }
        ParamSet defaults() const { return ParamSet(info().params); }
    };

    // --- registry ----------------------------------------------------------
    // Replaces an existing kind. The replaced operation is kept alive: a
    // reference taken before a plugin reload must not dangle.
    void registerOperation(std::unique_ptr<Operation> op);
    const Operation* findOperation(const std::string& kind) noexcept;
    const Operation& requireOperation(const std::string& kind);   // throws std::out_of_range
    // A stand-in for `kind` (OpInfo::missing): a plugin that is not loaded, a
    // kind from a newer SIRIUS. Its validation and run fail with the reason.
    std::unique_ptr<Operation> makeMissingOperation(const std::string& kind);
    // The operation registered for `kind`, registering a stand-in first when
    // there is none, so a pipeline step naming it keeps its place.
    const Operation* registerMissingOperation(const std::string& kind);
    // Every registered operation but the stand-ins, in registration order.
    std::vector<const Operation*> allOperations();
    // Groups in menu order with their operations (Reconstruct, Reduce, ...).
    std::vector<std::pair<std::string, std::vector<const Operation*>>> operationGroups();

} // namespace sirius::app

#endif // SIRIUS_APP_OPERATION_HPP
