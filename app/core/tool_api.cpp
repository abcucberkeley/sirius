#include "core/tool_api.hpp"

#include "core/cancel.hpp"
#include "core/display_model.hpp"    // probe reads a voxel the way the viewer does
#include "core/remote_source.hpp"    // the download gate on an output held on the cluster
#include "core/statistics.hpp"
#include "core/training_export.hpp"

#include <sirius/tiff_io.hpp>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace sirius::app {

    using json = nlohmann::json;

    namespace {
        // What call() answers for a failure (tool_api.hpp).
        json failure(const std::string& code, const std::string& message, const std::string& hint = {},
                     const json& data = nullptr) {
            json r = {{"error", message.empty() ? code : message}, {"error_kind", code}};
            if (!hint.empty()) r["hint"] = hint;
            if (!data.is_null()) r["data"] = data;
            return r;
        }

        // A help page is a file named after the kind, so a kind the caller
        // names must not reach outside the help directory. Letters, digits,
        // '_' and '-' are always fine; a registered operation's kind is too,
        // since a plugin's may have a '.' in it ("user.denoise"), which the
        // worker allows, but never a separator.
        bool helpKindAllowed(const std::string& kind) {
            if (kind.empty()) return false;
            bool plain = true;
            for (const char c : kind) {
                if (c == '/' || c == '\\' || c == ':' || c == '\0') return false;
                if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-') plain = false;
            }
            if (plain) return true;
            return kind.front() != '.' && findOperation(kind) != nullptr;
        }

        // The parameter `key` of `op`, for a value a caller gives. An unknown
        // key fails the call rather than being dropped: the step would run
        // with something other than what was asked, and nothing would say so.
        const ParamSpec& callerSpec(const Operation& op, const std::string& key) {
            std::string known;
            for (const ParamSpec& s : op.info().params) {
                if (s.key == key) return s;
                known += (known.empty() ? "" : ", ") + s.key;
            }
            throw ToolFailure("invalid_argument", "unknown parameter '" + key + "' for " + op.info().kind,
                              "the parameters of " + op.info().kind + " are " + known + " (describe_operation explains each)");
        }

        // A caller's value coerced to its spec. A number outside the spec's
        // range is clamped to it, and a line in `clamped` says so: the step
        // runs with a value the caller did not give.
        // A network path an agent names is refused unless the host allows them.
        void refuseNetworkPath(const std::string& path, const std::string& what, bool allowNetwork) {
            if (allowNetwork || !isNetworkPath(path)) return;
            throw ToolFailure("invalid_argument", what + " '" + path + "' is a network path (\\\\server\\share, //server/share), which tools may not open",
                              "use a local path; a person can still open a network path in the application, or start sirius-cli with "
                              "--allow-network-paths");
        }

        ParamValue callerValue(const ParamSpec& spec, const json& value, json& clamped, bool allowNetwork) {
            if (spec.type == ParamType::Path && value.is_string()) refuseNetworkPath(value.get<std::string>(), "parameter '" + spec.key + "'", allowNetwork);
            const ParamValue v = coerceToSpec(spec, value);
            if (spec.type != ParamType::Double && spec.type != ParamType::Int && spec.type != ParamType::Channel) return v;
            double given = 0.0;
            if (value.is_number()) given = value.get<double>();
            else if (value.is_string()) {
                try {
                    given = std::stod(value.get<std::string>());
                } catch (...) { return v; }
            } else return v;
            if (given >= spec.min && given <= spec.max) return v;
            char range[96];
            std::snprintf(range, sizeof range, "%g to %g", spec.min, spec.max);
            clamped.push_back("parameter '" + spec.key + "': " + value.dump() + " is outside " + range + ", so " +
                              toDisplayString(v) + " is used");
            return v;
        }

        // The hints of each tool, applied after the tools are built so the
        // positional add({...}) calls stay as they are. refusedWhileRunning:
        // an edit the workbench refuses during a run, with only a log line
        // to say so; readOnly: changes nothing; idempotent: a repeat changes
        // nothing more; destructive: writes files; openWorld: reaches beyond
        // this machine (a run may download model weights from Hugging Face
        // and send the data to the HPC worker); bigResult: may answer with a
        // long text (MCP clients cut long results unless told).
        struct ToolTraits {
            const char* name;
            const char* title;
            bool refusedWhileRunning, readOnly, idempotent, destructive, openWorld, bigResult;
        };
        constexpr ToolTraits kToolTraits[] = {
            {"get_state", "Workbench state", false, true, false, false, false, false},
            {"list_operations", "List operations", false, true, false, false, false, true},
            {"get_step", "Step details", false, true, false, false, false, false},
            {"add_step", "Add a step", true, false, false, false, false, false},
            {"remove_step", "Remove a step", true, false, false, false, false, false},
            {"move_step", "Move a step", true, false, false, false, false, false},
            {"set_step_enabled", "Enable or skip a step", true, false, true, false, false, false},
            {"set_params", "Set step parameters", true, false, true, false, false, false},
            {"apply_preset", "Apply a preset", true, false, true, false, false, false},
            {"set_cache", "Set a step's cache policy", true, false, true, false, false, false},
            {"run", "Run the pipeline", false, false, false, false, true, false},
            {"view_step", "View a step", false, false, false, false, false, false},
            {"select_step", "Select a step", false, false, false, false, false, false},
            {"set_view", "Change the viewer", false, false, false, false, false, false},
            {"list_tracks", "List tracks", false, true, false, false, false, false},
            {"focus_track", "Focus a track", false, false, false, false, false, false},
            {"get_diagnostics", "Step diagnostics", false, true, false, false, false, false},
            {"get_help", "Help page", false, true, false, false, false, true},
            {"undo", "Undo", true, false, false, false, false, false},
            {"redo", "Redo", true, false, false, false, false, false},
            {"set_backend", "Compute backend", false, false, true, false, false, false},
            {"load_example_pipeline", "Load the example pipeline", true, false, false, false, false, false},
            {"export_training_data", "Export training data", true, false, false, true, false, false},
            // looking at an output, and writing one out: the three tools a
            // session used to serve alone, so that a run driven through the
            // window can be compared by file and not only by screenshot
            {"probe", "Probe a voxel", true, true, false, false, false, false},
            {"statistics", "Intensity statistics", true, true, false, false, false, false},
            {"export_result", "Export a result", true, false, false, true, false, false},
            // labels by step (core/workbench.hpp "labels by step"): the viewer's paint tools, for scripts
            {"list_labels", "List labels", false, true, false, false, false, true},
            {"paint_label", "Paint a label", true, false, false, false, false, false},
            {"fill_label", "Fill a label", true, false, false, false, false, false},
            {"merge_labels", "Merge labels", true, false, false, false, false, false},
            {"split_label", "Split a label", true, false, false, false, false, false},
            {"delete_label", "Delete a label", true, false, false, false, false, false},
            {"clear_labels", "Clear the labels", true, false, false, false, false, false},
            {"set_label_reviewed", "Mark a label reviewed", true, false, true, false, false, false},
            {"export_labels", "Export labels", true, false, false, true, false, false},
            {"get_log", "Workbench log", false, true, false, false, false, true},
        };
        // MCP's Tool._meta key for how long a result a client should keep
        // (Claude Code otherwise cuts at its default), and the length asked.
        constexpr const char* kMaxResultSizeKey = "anthropic/maxResultSizeChars";
        constexpr int kMaxResultSizeChars = 200000;

        void applyTraits(std::vector<ToolSpec>& tools) {
            for (const ToolTraits& tr : kToolTraits)
                for (ToolSpec& t : tools) {
                    if (t.name != tr.name) continue;
                    t.title = tr.title;
                    t.refusedWhileRunning = tr.refusedWhileRunning;
                    t.readOnly = tr.readOnly;
                    t.idempotent = tr.idempotent;
                    t.destructive = tr.destructive;
                    t.openWorld = tr.openWorld;
                    if (tr.bigResult) t.meta[kMaxResultSizeKey] = kMaxResultSizeChars;
                }
        }

        json stepParam() {
            return {{"type", {"integer", "string"}},
                    {"description", "Step number as shown in the operations list (1 = Load, 2 = second step, ...) or a step name"}};
        }
        // The step probe, statistics and export_result look at. The default is
        // the host's rule, so it is named rather than left to be guessed.
        json inspectStepParam() {
            return {{"type", {"integer", "string"}},
                    {"description", "The step by number (1 = Load) or by name; default the viewed step in the application, "
                                    "the last computed step in a session"}};
        }
        json obj(std::initializer_list<std::pair<const std::string, json>> props, std::vector<std::string> required = {}) {
            json properties = json::object();
            for (const auto& p : props) properties[p.first] = p.second;
            json o = {{"type", "object"}, {"properties", properties}};
            if (!required.empty()) o["required"] = required;
            return o;
        }
        std::string lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        }
        // The tools accept these values in any case, so the choices are named
        // in the description rather than as a JSON Schema enum, which a client
        // that validates arguments would hold to the lowercase spelling.
        json enumDesc(const std::vector<std::string>& values, const std::string& description) {
            std::string names;
            for (const std::string& v : values) names += (names.empty() ? "" : ", ") + v;
            return {{"type", "string"}, {"description", description + ". One of: " + names + " (any case)"}};
        }
        json intList(const char* description) { return {{"type", "array"}, {"items", {{"type", "integer"}}}, {"description", description}}; }
        json numberList(const char* description) { return {{"type", "array"}, {"items", {{"type", "number"}}}, {"description", description}}; }

        // --- arguments of the inspecting tools, as a session reads them -----------
        [[noreturn]] void invalidArg(const std::string& message, const std::string& hint = {}, json data = nullptr) {
            throw ToolFailure("invalid_argument", message, hint, std::move(data));
        }
        bool given(const json& a, const char* key) { return a.contains(key) && !a[key].is_null(); }
        bool boolArg(const json& a, const char* key, bool def) {
            if (!given(a, key)) return def;
            if (!a[key].is_boolean()) invalidArg(std::string("'") + key + "' must be true or false");
            return a[key].get<bool>();
        }
        std::int64_t integerArg(const json& a, const char* key, std::int64_t def, std::int64_t min, std::int64_t max) {
            if (!given(a, key)) return def;
            const json& v = a[key];
            if (!v.is_number() || (!v.is_number_integer() && v.get<double>() != std::floor(v.get<double>())))
                invalidArg(std::string("'") + key + "' must be an integer");
            const std::int64_t i = v.is_number_integer() ? v.get<std::int64_t>() : static_cast<std::int64_t>(v.get<double>());
            if (i < min || i > max)
                invalidArg(std::string("'") + key + "' must be within " + std::to_string(min) + " and " + std::to_string(max) +
                           " here, not " + std::to_string(i));
            return i;
        }
        // "0.1", "50", "99.9": a percentile as a key.
        std::string percentileKey(double p) {
            char buf[32];
            std::snprintf(buf, sizeof buf, "%g", p);
            return buf;
        }
        // Data that stays on the cluster (a cluster dataset, a step's output
        // the engine holds on the node) is here only if it is downloaded
        // first. File > Export result asks the user; a tool call has nobody to
        // ask, so it carries the answer as download:true.
        bool heldRemotely(const StepOutput& out) { return !out.array && out.source && out.source->viewProvider(); }
        std::string downloadQuestion(const StepOutput& out, int index) {
            const double gb = static_cast<double>(std::max<Index>(out.meta.dims.numel(), 0)) * sizeof(float) / 1e9;
            char size[32];
            std::snprintf(size, sizeof size, gb >= 0.1 ? "%.1f GB" : "%.0f MB", gb >= 0.1 ? gb : gb * 1000.0);
            const std::string where = !out.where.empty() ? out.where : std::string("the cluster");
            return "step " + Step::number(index) + "'s data is held on " + where + ", and exporting it here downloads all of it to this computer first: " +
                   size + " (as float32)";
        }
    } // namespace

    bool isNetworkPath(const std::string& path) {
        const auto sep = [](char c) { return c == '/' || c == '\\'; };
        if (path.size() < 2 || !sep(path[0]) || !sep(path[1])) return false;
        // \\?\C:\... and \\.\C:\... are a local drive in the long-path form;
        // \\?\UNC\server\..., \\.\pipe\... and the other devices are not
        if (path.size() >= 6 && (path[2] == '?' || path[2] == '.') && sep(path[3]) && std::isalpha(static_cast<unsigned char>(path[4])) &&
            path[5] == ':')
            return false;
        return true;
    }

    ToolApi::ToolApi(Workbench& wb) : wb_(wb) {
        add({"get_state",
             "Dataset, operations stack (with numbers, kinds, enabled state and parameters), the selected and viewed steps, the viewer state and the backend.",
             obj({}),
             [this](const json&) {
                 json steps = json::array();
                 for (int i = 0; i < wb_.pipeline().size(); ++i) steps.push_back(stepJson(i));
                 json ds;
                 if (wb_.hasDataset()) {
                     const DatasetMeta& m = wb_.dataset();
                     ds = {{"name", m.name}, {"path", m.sourcePath}, {"format", m.format}, {"shape", m.shapeString()}, {"voxel_um", m.voxelUm}, {"acquisition", m.acquisition}, {"dtype", toString(m.sourceType)}};
                     json ch = json::array();
                     for (const ChannelInfo& c : m.channels) ch.push_back({{"label", c.label}, {"wavelength_nm", c.wavelengthNm}, {"color", c.hexColor()}});
                     ds["channels"] = ch;
                     if (m.sim.present) {
                         ds["sim"] = {{"ndirs", m.sim.ndirs}, {"nphases", m.sim.nphases}};
                         if (!m.sim.isShorthand()) ds["sim"]["layout"] = m.sim.storage;
                     }
                 }
                 return json{{"dataset", ds},
                             {"steps", steps},
                             {"selected_step", wb_.selectedIndex() + 1},
                             {"viewed_step", wb_.viewedIndex() + 1},
                             {"view", wb_.viewState().toJson()},
                             {"backend", toString(wb_.backend())},
                             {"hpc_device", lower(toString(wb_.hpcDevice()))},
                             {"running", wb_.running()},
                             {"can_undo", wb_.history().canUndo()},
                             {"undo_label", wb_.history().undoLabel()}};
             }});
        add({"list_operations",
             "Every operation kind that can be added as a step, with its group and parameters.",
             obj({}),
             [](const json&) {
                 json out = json::array();
                 for (const Operation* op : allOperations()) {
                     if (op->kind() == "load") continue;
                     json params = json::array();
                     for (const ParamSpec& s : op->info().params)
                         params.push_back({{"key", s.key}, {"label", s.label}, {"default", toJson(s.defaultValue)}, {"schema", schemaOf(s)}});
                     json presets = json::array();
                     for (const ParamPreset& preset : op->info().presets)
                         presets.push_back({{"name", preset.name}, {"summary", preset.summary}});
                     json entry = {{"kind", op->kind()}, {"name", op->info().name}, {"group", op->info().group}, {"params", params}};
                     if (!presets.empty()) entry["presets"] = std::move(presets);
                     out.push_back(std::move(entry));
                 }
                 return out;
             }});
        add({"get_step",
             "Details of one step: parameters, summary, validation, output shape and diagnostics summary.",
             obj({{"step", stepParam()}}, {"step"}),
             [this](const json& a) {
                 const int i = resolveStep(a);
                 json j = stepJson(i);
                 const Validation v = wb_.stepValidation(i);
                 j["errors"] = v.errors;
                 j["warnings"] = v.warnings;
                 j["output_shape"] = wb_.outputMetaOf(i).shapeString();
                 if (auto out = wb_.output(i)) {
                     // A step cached as recompute keeps only a shell of its
                     // output (diagnostics, note) once the steps below have
                     // read it: there is no data to look at, which is what
                     // has_output says, not that the step did not run.
                     const bool fresh = wb_.outputFresh(i);
                     j["has_output"] = fresh || out->array != nullptr || out->source != nullptr;
                     j["output_fresh"] = fresh;
                     j["diagnostics_summary"] = out->diagnostics.summary;
                     j["note"] = out->note;
                 } else {
                     j["has_output"] = false;
                 }
                 return j;
             }});
        add({"add_step",
             "Append a processing step of the given kind (see list_operations); becomes selected and viewed. An optional "
             "preset is applied first, then the optional parameters; an optional name renames the step.",
             obj({{"kind", {{"type", "string"}}},
                  {"params", {{"type", "object"}, {"description", "parameter key/value pairs to set"}}},
                  {"preset", {{"type", "string"}, {"description", "one of the operation's presets (list_operations gives them)"}}},
                  {"name", {{"type", "string"}, {"description", "the step's name, by which later calls can address it"}}},
                  {"at", {{"type", "integer"}, {"description", "1-based position to insert at (default: end)"}}}},
                 {"kind"}),
             [this](const json& a) {
                 const std::string kind = a.value("kind", "");
                 const Operation* op = findOperation(kind);
                 if (!op || op->info().missing)
                     throw ToolFailure("unknown_operation", "unknown operation kind '" + kind + "'", "list_operations gives the kinds");
                 if (!wb_.canEdit()) throw ToolFailure("busy", "A run is in progress: cancel it or wait before editing the pipeline.");
                 int at = -1;
                 if (a.contains("at") && a["at"].is_number_integer()) at = a["at"].get<int>() - 1;
                 // Everything the caller gave is checked before the step goes
                 // in, so that a refused call leaves the pipeline as it was.
                 const std::string preset = a.contains("preset") && a["preset"].is_string() ? a["preset"].get<std::string>() : std::string();
                 if (!preset.empty()) {
                     std::string known;
                     bool found = false;
                     for (const ParamPreset& p : op->info().presets) {
                         found = found || p.name == preset;
                         known += (known.empty() ? "" : ", ") + p.name;
                     }
                     if (!found)
                         throw std::invalid_argument(known.empty() ? kind + " has no presets"
                                                                   : "no preset '" + preset + "' for " + kind + "; it has " + known);
                 }
                 json clamped = json::array();
                 std::vector<std::pair<std::string, ParamValue>> values;
                 if (a.contains("params") && a["params"].is_object())
                     for (auto it = a["params"].begin(); it != a["params"].end(); ++it) {
                         const ParamSpec& spec = callerSpec(*op, it.key());
                         values.emplace_back(spec.key, callerValue(spec, it.value(), clamped, allowNetworkPaths_));
                     }
                 const std::string name = a.contains("name") && a["name"].is_string() ? a["name"].get<std::string>() : std::string();
                 // Not seeded from the data on hand: that is not yet this
                 // step's input (a SIM step above has not run), and an
                 // automatic window left as it is follows the real input at
                 // run time, in this run and in a saved pipeline.
                 const StepId id = wb_.addStep(kind, at, false);
                 const int i = wb_.pipeline().indexOf(id);
                 if (!preset.empty()) wb_.applyPreset(i, preset);
                 if (!values.empty()) {
                     ParamSet p = wb_.pipeline().at(i).params;
                     for (const auto& [key, value] : values) p.set(key, value);
                     wb_.setStepParams(i, p, "Set parameters of " + wb_.pipeline().at(i).name);
                 }
                 if (!name.empty()) wb_.renameStep(i, name);
                 actions_.push_back({ActionRecord::Kind::Param, "Added step " + Step::number(i) + " · " + wb_.pipeline().at(i).name, "undo", {}, "add_step"});
                 json out = stepJson(i);
                 if (!clamped.empty()) out["clamped"] = clamped;
                 return out;
             }});
        add({"remove_step", "Remove a step (never the Load step).", obj({{"step", stepParam()}}, {"step"}),
             [this](const json& a) {
                 const int i = resolveStep(a);
                 if (i == 0) throw std::invalid_argument("the Load step cannot be removed");
                 const std::string name = wb_.pipeline().at(i).name;
                 wb_.removeStep(i);
                 actions_.push_back({ActionRecord::Kind::Param, "Removed step " + Step::number(i) + " · " + name, "undo", {}, "remove_step"});
                 return json{{"ok", true}};
             }});
        add({"move_step", "Move a step up (delta -1) or down (delta +1) in the stack.",
             obj({{"step", stepParam()}, {"delta", {{"type", "integer"}}}}, {"step", "delta"}),
             [this](const json& a) {
                 const int i = resolveStep(a);
                 const int delta = a.value("delta", 0);
                 if (!wb_.moveStep(i, delta)) throw std::invalid_argument("that move is not possible");
                 actions_.push_back({ActionRecord::Kind::Param, "Moved step " + Step::number(i) + " to " + Step::number(i + delta), "undo", {}, "move_step"});
                 return stepJson(i + delta);
             }});
        add({"set_step_enabled", "Enable (run) or skip a step; a skipped step passes its input through unchanged.",
             obj({{"step", stepParam()}, {"enabled", {{"type", "boolean"}}}}, {"step", "enabled"}),
             [this](const json& a) {
                 const int i = resolveStep(a);
                 const bool on = a.value("enabled", true);
                 wb_.setStepEnabled(i, on);
                 actions_.push_back({ActionRecord::Kind::Param, std::string(on ? "Enabled" : "Skipped") + " step " + Step::number(i) + " · " + wb_.pipeline().at(i).name, "undo", {}, "set_step_enabled"});
                 return stepJson(i);
             }});
        add({"set_params",
             "Set one or more parameters of a step (keys as in get_step / list_operations). Values are validated and coerced; the change is undoable.",
             obj({{"step", stepParam()}, {"params", {{"type", "object"}}}}, {"step", "params"}),
             [this](const json& a) {
                 const int i = resolveStep(a);
                 if (!a.contains("params") || !a["params"].is_object()) throw std::invalid_argument("'params' must be an object");
                 const Step& s = wb_.pipeline().at(i);
                 ParamSet p = s.params;
                 std::string changes;
                 json clamped = json::array();
                 for (auto it = a["params"].begin(); it != a["params"].end(); ++it) {
                     const ParamSpec* spec = &callerSpec(s.op(), it.key());
                     const ParamValue v = callerValue(*spec, it.value(), clamped, allowNetworkPaths_);
                     const ParamValue* old = p.find(spec->key);
                     if (!changes.empty()) changes += ", ";
                     changes += spec->label + " " + (old ? toDisplayString(*old) : "—") + " → " + toDisplayString(v);
                     p.set(spec->key, v);
                 }
                 wb_.setStepParams(i, p, "Step " + Step::number(i) + " · " + changes);
                 actions_.push_back({ActionRecord::Kind::Param, "Step " + Step::number(i) + " · " + changes, "undo", {}, "set_params"});
                 json out = stepJson(i);
                 // A parameter the step's own settings ignore is stored and
                 // will be read again when they change back, but it does
                 // nothing now and the panel does not even show it. Saying so
                 // is the difference between "set" and "had any effect".
                 json ignored = json::array();
                 for (auto it = a["params"].begin(); it != a["params"].end(); ++it)
                     for (const ParamSpec& sp : s.op().info().params)
                         if (sp.key == it.key() && !sp.visibleFor(p)) ignored.push_back(sp.key);
                 if (!ignored.empty())
                     out["ignored"] = {{"keys", ignored},
                                       {"why", "stored, but the step's current settings do not read these; they apply again "
                                               "when the settings that gate them change back"}};
                 if (!clamped.empty()) out["clamped"] = clamped;
                 return out;
             }});
        add({"apply_preset",
             "Fill a step's parameters from one of its operation's presets: a starting point for a kind of structure "
             "(list_operations gives the names). It is an ordinary undoable parameter change, so everything stays editable.",
             obj({{"step", stepParam()}, {"preset", {{"type", "string"}}}}, {"step", "preset"}),
             [this](const json& a) {
                 const int i = resolveStep(a);
                 const std::string name = a.value("preset", std::string());
                 const Step& s = wb_.pipeline().at(i);
                 // told apart here, because applyPreset answers false to both
                 // and "Nuclei does not exist" would be a lie during a run
                 if (!wb_.canEdit())
                     throw ToolFailure("busy", "A run is in progress: cancel it or wait before applying a preset.");
                 if (!wb_.applyPreset(i, name)) {
                     std::string known;
                     for (const ParamPreset& p : s.op().info().presets) known += (known.empty() ? "" : ", ") + p.name;
                     throw std::invalid_argument(known.empty() ? s.kind + " has no presets"
                                                               : "no preset '" + name + "' for " + s.kind + "; it has " + known);
                 }
                 actions_.push_back({ActionRecord::Kind::Param, "Step " + Step::number(i) + " · preset " + name, "undo", {}, "apply_preset"});
                 return stepJson(i);
             }});
        add({"set_cache", "Cache policy of a step's output: memory, disk or recompute.",
             obj({{"step", stepParam()}, {"policy", {{"type", "string"}, {"enum", {"memory", "disk", "recompute"}}}}}, {"step", "policy"}),
             [this](const json& a) {
                 const int i = resolveStep(a);
                 auto pol = cachePolicyFromString(a.value("policy", ""));
                 if (!pol) throw std::invalid_argument("policy must be memory, disk or recompute");
                 wb_.setStepCache(i, *pol);
                 actions_.push_back({ActionRecord::Kind::Param, "Step " + Step::number(i) + " · cache " + toString(*pol), "undo", {}, "set_cache"});
                 return stepJson(i);
             }});
        add({"run", "Run the pipeline up to a step (default: all enabled steps). Blocks until finished; returns timings or the error.",
             obj({{"step", stepParam()}}),
             [this](const json& a) {
                 // A failure, not a value: an {"error"} result would read as
                 // a run that failed, where nothing could run at all.
                 if (!runHook_) throw ToolFailure("unsupported", "running is not available in this context");
                 // refused up front, as the window's Run buttons are: nothing runs on HPC without the engine
                 if (const RunGate gate = wb_.runGate(); !gate.enabled)
                     throw ToolFailure("no_engine", gate.why, "connect to the cluster with SIRIUS's engine (the Cluster button), or set_backend CPU or CUDA");
                 const int target = a.contains("step") ? resolveStep(a) : wb_.pipeline().size() - 1;
                 json r = runHook_(target);
                 std::string text = "Ran to step " + Step::number(target) + " · " + wb_.pipeline().at(target).name;
                 if (r.contains("seconds") && r["seconds"].is_number()) {
                     char buf[32];
                     std::snprintf(buf, sizeof buf, " · %.1f s", r["seconds"].get<double>());
                     text += buf;
                 }
                 if (r.contains("error") && r["error"].is_string() && !r["error"].get<std::string>().empty())
                     text += " · failed: " + r["error"].get<std::string>();
                 actions_.push_back({ActionRecord::Kind::Run, text, "log", {}, "run"});
                 return r;
             }});
        add({"view_step", "Show a step's output in the viewer (the ◉ button).", obj({{"step", stepParam()}}, {"step"}),
             [this](const json& a) {
                 const int i = resolveStep(a);
                 wb_.view(i);
                 actions_.push_back({ActionRecord::Kind::View, "Viewer → step " + Step::number(i) + " · " + wb_.pipeline().at(i).name, "view",
                                     json{{"view_step", i + 1}, {"view", wb_.viewState().toJson()}}, "view_step"});
                 return json{{"viewed_step", i + 1}};
             }});
        add({"select_step", "Select a step: its parameters and diagnostics are shown.", obj({{"step", stepParam()}}, {"step"}),
             [this](const json& a) {
                 const int i = resolveStep(a);
                 wb_.select(i);
                 actions_.push_back({ActionRecord::Kind::View, "Selected step " + Step::number(i) + " · " + wb_.pipeline().at(i).name, "view",
                                     json{{"select_step", i + 1}}, "select_step"});
                 return json{{"selected_step", i + 1}};
             }});
        add({"set_view",
             "Change the viewer: mode (ortho|3d|compare), tool (nav|probe|measure|roi|paint|prompt), prompt_mode (what the Prompt tool places: box|click|scribble), z, t, crosshair [x, y], labels overlay, label (select one and jump to it), solo (draw only the selected label), trajectories and follow_track (tracked labels), channel visibility list, yaw/pitch, diagnostics tab is not part of this.",
             obj({{"mode", {{"type", "string"}, {"enum", {"ortho", "3d", "compare"}}}},
                  {"tool", {{"type", "string"}, {"enum", {"nav", "probe", "measure", "roi", "paint", "prompt"}}}},
                  {"prompt_mode", {{"type", "string"}, {"enum", {"box", "click", "scribble"}}}},
                  {"z", {{"type", "integer"}}},
                  {"t", {{"type", "integer"}}},
                  {"crosshair", {{"type", "array"}, {"items", {{"type", "integer"}}}, {"description", "[x, y] or [x, y, z]"}}},
                  {"labels", {{"type", "boolean"}}},
                  {"label", {{"type", "integer"}, {"description", "label id to select; the view jumps to it"}}},
                  {"solo", {{"type", "boolean"}, {"description", "show only the selected label"}}},
                  {"trajectories", {{"type", "boolean"}, {"description", "tracked labels: draw each track's path over time"}}},
                  {"follow_track", {{"type", "boolean"}, {"description", "tracked labels: keep the selected track under the crosshair as t changes"}}},
                  {"channels", {{"type", "array"}, {"items", {{"type", "boolean"}}}}},
                  {"yaw", {{"type", "number"}}},
                  {"pitch", {{"type", "number"}}}}),
             [this](const json& a) {
                 ViewState s = wb_.viewState();
                 std::string text;
                 auto note = [&](const std::string& t) { text += (text.empty() ? "" : " · ") + t; };
                 if (a.contains("mode")) {
                     auto m = viewModeFromString(a["mode"].get<std::string>());
                     if (!m) throw std::invalid_argument("mode must be ortho, 3d or compare");
                     s.mode = *m;
                     note(std::string("mode ") + toString(*m));
                 }
                 if (a.contains("tool")) {
                     auto t = viewerToolFromString(a["tool"].get<std::string>());
                     if (!t) throw std::invalid_argument("unknown tool");
                     s.tool = *t;
                     note(std::string("tool ") + toString(*t));
                 }
                 if (a.contains("prompt_mode")) {
                     auto m = promptModeFromString(a["prompt_mode"].get<std::string>());
                     if (!m) throw std::invalid_argument("prompt_mode must be box, click or scribble");
                     s.promptMode = *m;
                     note(std::string("prompt ") + toString(*m));
                 }
                 if (a.contains("z")) {
                     s.z = a["z"].get<Index>();
                     note("z " + std::to_string(s.z));
                 }
                 if (a.contains("t")) {
                     s.t = a["t"].get<Index>();
                     note("t " + std::to_string(s.t));
                 }
                 if (a.contains("crosshair") && a["crosshair"].is_array() && a["crosshair"].size() >= 2) {
                     s.cx = a["crosshair"][0].get<Index>();
                     s.cy = a["crosshair"][1].get<Index>();
                     if (a["crosshair"].size() >= 3) s.z = a["crosshair"][2].get<Index>();
                     s.crosshair = true;
                     note("crosshair " + std::to_string(s.cx) + ", " + std::to_string(s.cy));
                 }
                 if (a.contains("labels")) {
                     s.labels = a["labels"].get<bool>();
                     note(s.labels ? "labels on" : "labels off");
                 }
                 if (a.contains("solo")) {
                     s.soloLabel = a["solo"].get<bool>();
                     note(s.soloLabel ? "solo label" : "all labels");
                 }
                 if (a.contains("trajectories")) {
                     s.trajectories = a["trajectories"].get<bool>();
                     note(s.trajectories ? "trajectories on" : "trajectories off");
                 }
                 if (a.contains("follow_track")) {
                     s.followTrack = a["follow_track"].get<bool>();
                     note(s.followTrack ? "following the track" : "not following");
                 }
                 std::uint32_t focus = 0;
                 if (a.contains("label")) {
                     focus = a["label"].get<std::uint32_t>();
                     note("label " + std::to_string(focus));
                 }
                 if (a.contains("channels") && a["channels"].is_array()) {
                     s.channelVisible.clear();
                     for (const json& e : a["channels"]) s.channelVisible.push_back(e.get<bool>());
                     note("channels");
                 }
                 if (a.contains("yaw")) {
                     s.yaw = a["yaw"].get<double>();
                     note("yaw");
                 }
                 if (a.contains("pitch")) {
                     s.pitch = a["pitch"].get<double>();
                     note("pitch");
                 }
                 const DatasetMeta meta = wb_.displayedMeta();
                 s.z = std::clamp<Index>(s.z, 0, std::max<Index>(meta.dims.z - 1, 0));
                 s.t = std::clamp<Index>(s.t, 0, std::max<Index>(meta.dims.t - 1, 0));
                 wb_.setViewState(s);
                 if (focus) wb_.focusLabel(focus);
                 actions_.push_back({ActionRecord::Kind::View, "Viewer → " + (text.empty() ? std::string("unchanged") : text), "view",
                                     json{{"view", wb_.viewState().toJson()}}, "set_view"});
                 return wb_.viewState().toJson();
             }});
        add({"list_tracks",
             "The tracks of the viewed labels, when a tracking step made them: id, first and last frame, frames present, "
             "gaps (frames missing between first and last, where identity may have been lost), um per frame, net "
             "displacement, parent and children. Division counts are the tracker's estimate, not a measurement.",
             obj({{"limit", {{"type", "integer"}, {"description", "rows to return, those with the most gaps first (default 50)"}}}}),
             [this](const json& a) {
                 std::vector<TrackSummary> rows = wb_.viewedTrackSummaries();
                 if (rows.empty()) return json{{"tracks", json::array()}, {"message", "the viewed labels are not tracked"}};
                 const std::size_t total = rows.size();
                 Index gapped = 0;
                 for (const TrackSummary& r : rows) gapped += r.gaps > 0 ? 1 : 0;
                 const Index divisions = countDivisions(rows);
                 std::stable_sort(rows.begin(), rows.end(), [](const TrackSummary& x, const TrackSummary& y) { return x.gaps > y.gaps; });
                 const std::size_t limit = static_cast<std::size_t>(std::max<long long>(1, a.value("limit", 50LL)));
                 if (rows.size() > limit) rows.resize(limit);
                 json list = json::array();
                 for (const TrackSummary& r : rows)
                     list.push_back({{"id", r.id}, {"first", r.first}, {"last", r.last}, {"frames", r.frames}, {"gaps", r.gaps}, {"um_per_frame", r.umPerFrame}, {"net_um", r.netUm}, {"parent", r.parent}, {"children", r.children}});
                 return json{{"total", total}, {"with_gaps", gapped}, {"divisions", divisions}, {"tracks", list}};
             }});
        add({"focus_track",
             "Select a track of the viewed labels and bring it into view: the time point moves to the nearest one the "
             "track exists in, the crosshair onto its centroid.",
             obj({{"id", {{"type", "integer"}}}}, {"id"}),
             [this](const json& a) {
                 const std::uint32_t id = a.at("id").get<std::uint32_t>();
                 if (!wb_.focusTrack(id)) return json{{"ok", false}, {"message", "no track " + std::to_string(id) + " in the viewed labels"}};
                 actions_.push_back({ActionRecord::Kind::View, "Viewer → track " + std::to_string(id), "view",
                                     json{{"view", wb_.viewState().toJson()}}, "focus_track"});
                 return json{{"ok", true}, {"view", wb_.viewState().toJson()}};
             }});
        add({"get_diagnostics",
             "Diagnostics of a step (default: the selected one): summary, table, facts, curves, histograms, warnings.",
             obj({{"step", stepParam()}}),
             [this](const json& a) {
                 const int i = a.contains("step") ? resolveStep(a) : wb_.selectedIndex();
                 const Diagnostics d = wb_.diagnosticsOf(i);
                 json j = {{"step", i + 1}, {"summary", d.summary}, {"footer", d.footer}, {"warnings", d.warnings}};
                 json facts = json::object();
                 for (const DiagnosticFact& f : d.facts) facts[f.key] = f.value;
                 j["facts"] = facts;
                 if (d.table) j["table"] = {{"caption", d.table->caption}, {"header", d.table->header}, {"rows", d.table->rows}};
                 json curves = json::array();
                 for (const DiagnosticCurve& c : d.curves) {
                     json cj = {{"title", c.title}, {"points", c.y.size()}};
                     if (!c.y.empty()) {
                         cj["first"] = c.y.front();
                         cj["last"] = c.y.back();
                     }
                     curves.push_back(cj);
                 }
                 j["curves"] = curves;
                 json hists = json::array();
                 for (const DiagnosticHistogram& h : d.histograms) hists.push_back({{"channel", h.channel}, {"lo", h.lo}, {"hi", h.hi}, {"gamma", h.gamma}});
                 j["histograms"] = hists;
                 json tabs = json::array();
                 for (const DiagnosticTab& t : d.tabs) tabs.push_back(t.name);
                 j["tabs"] = tabs;
                 return j;
             }});
        add({"get_help", "The help page (Markdown) of an operation kind or of the selected step.",
             obj({{"kind", {{"type", "string"}}}}),
             [this](const json& a) {
                 std::string kind = a.value("kind", "");
                 if (!kind.empty() && !helpKindAllowed(kind))
                     throw ToolFailure("invalid_argument", "'" + kind + "' is not an operation kind",
                                       "a kind is letters, digits, '_' and '-', or a registered kind (list_operations gives them)");
                 if (kind.empty()) {
                     // A pipeline file may give a step any kind at all (a
                     // stand-in keeps it), so the selected step's kind is
                     // held to the same rule as a kind the caller names.
                     kind = wb_.pipeline().at(wb_.selectedIndex()).kind;
                     if (!helpKindAllowed(kind))
                         throw ToolFailure("not_found", "the selected step's kind '" + kind + "' has no help page",
                                           "name a kind (list_operations gives them)");
                 }
                 if (!helpHook_) return json{{"kind", kind}, {"markdown", "(help pages are not available in this context)"}};
                 return json{{"kind", kind}, {"markdown", helpHook_(kind)}};
             }});
        add({"undo", "Undo the last change.", obj({}), [this](const json&) {
                 const std::string label = wb_.history().undoLabel();
                 if (label.empty()) return json{{"ok", false}, {"message", "nothing to undo"}};
                 wb_.undo();
                 actions_.push_back({ActionRecord::Kind::Edit, "Undid: " + label, "", {}, "undo"});
                 return json{{"ok", true}, {"undone", label}};
             }});
        add({"redo", "Redo the last undone change.", obj({}), [this](const json&) {
                 const std::string label = wb_.history().redoLabel();
                 if (label.empty()) return json{{"ok", false}, {"message", "nothing to redo"}};
                 wb_.redo();
                 actions_.push_back({ActionRecord::Kind::Edit, "Redid: " + label, "", {}, "redo"});
                 return json{{"ok", true}, {"redone", label}};
             }});
        add({"set_backend",
             "Compute backend for runs: CUDA, CPU or HPC. hpc_device picks where the HPC worker computes, its job's GPU or its "
             "CPU; it is kept until changed and switches without a new job.",
             obj({{"backend", {{"type", "string"}, {"enum", {"CUDA", "CPU", "HPC"}}}},
                  {"hpc_device", {{"type", "string"}, {"enum", {"gpu", "cpu"}}}}},
                 {"backend"}),
             [this](const json& a) {
                 auto b = backendFromString(a.value("backend", ""));
                 if (!b) throw std::invalid_argument("backend must be CUDA, CPU or HPC");
                 std::optional<HpcDevice> d;
                 if (a.contains("hpc_device")) {
                     d = a["hpc_device"].is_string() ? hpcDeviceFromString(a["hpc_device"].get<std::string>()) : std::nullopt;
                     if (!d) throw std::invalid_argument("hpc_device must be gpu or cpu");
                 }
                 wb_.setBackend(*b);
                 if (d) wb_.setHpcDevice(*d);
                 std::string text = std::string("Backend → ") + toString(*b);
                 if (*b == Backend::Hpc) text += std::string(" · ") + toString(wb_.hpcDevice());
                 actions_.push_back({ActionRecord::Kind::Param, text, "", {}, "set_backend"});
                 return json{{"backend", toString(*b)}, {"hpc_device", lower(toString(wb_.hpcDevice()))}};
             }});
        add({"load_example_pipeline", "Replace the stack with the example pipeline (SIM → einsum → contrast → merge → segment → volume).",
             obj({}), [this](const json&) {
                 wb_.loadExamplePipeline();
                 actions_.push_back({ActionRecord::Kind::Param, "Loaded the example pipeline", "undo", {}, "load_example_pipeline"});
                 json steps = json::array();
                 for (int i = 0; i < wb_.pipeline().size(); ++i) steps.push_back(stepJson(i));
                 return json{{"steps", steps}};
             }});
        add({"export_training_data",
             "Write a step's labels as training data: instance masks, a semantic mask, bounding boxes (3D and per plane) and "
             "optionally one 8-bit image plus one YOLO file per plane, into a dataset folder that accumulates one sample per call.",
             obj({{"step", stepParam()},
                  {"directory", {{"type", "string"}, {"description", "Dataset folder; created if missing"}}},
                  {"sample", {{"type", "string"}, {"description", "Sample folder name; a number is appended if it is taken"}}},
                  {"image", {{"type", "boolean"}}},
                  {"instances", {{"type", "boolean"}}},
                  {"semantic", {{"type", "boolean"}}},
                  {"boxes", {{"type", "boolean"}}},
                  {"slices", {{"type", "boolean"}, {"description", "One 8-bit plane and one YOLO file per z, for 2D detectors"}}},
                  {"min_voxels", {{"type", "integer"}, {"description", "Objects smaller than this are left out"}}},
                  {"image_dtype", {{"type", "string"}, {"enum", {"uint8", "uint16", "float32"}}, {"description", "Pixel type of image.tif (default uint16)"}}},
                  {"image_scaling", {{"type", "string"}, {"enum", {"cast", "minmax", "percentile"}}, {"description", "How the image is rescaled into that type (default percentile)"}}}},
                 {"directory"}),
             [this](const json& a) {
                 refuseNetworkPath(a.value("directory", std::string()), "directory", allowNetworkPaths_);
                 // The export materializes the step's input through the same
                 // ArraySource a running job may be reading; keep it under the
                 // run-state rule every other entry point follows.
                 if (!wb_.canEdit())
                     throw ToolFailure("busy", "A run is in progress: cancel it or wait before exporting training data.");
                 const int i = a.contains("step") ? resolveStep(a) : wb_.viewedIndex();
                 std::shared_ptr<const StepOutput> out = wb_.output(i);
                 if (!out) throw ToolFailure("not_computed", "step " + Step::number(i) + " has not been computed yet", "run it first");
                 if (!out->labels || out->labels->empty()) throw std::invalid_argument("step " + Step::number(i) + " produced no labels");
                 TrainingExportOptions o;
                 o.directory = a.value("directory", std::string());
                 o.sample = a.value("sample", wb_.hasDataset() ? wb_.dataset().name : std::string("sample"));
                 o.image = a.value("image", true);
                 o.instances = a.value("instances", true);
                 o.semantic = a.value("semantic", true);
                 o.boxes = a.value("boxes", true);
                 o.slices = a.value("slices", false);
                 o.minVoxels = static_cast<std::uint64_t>(std::max(1, a.value("min_voxels", 1)));
                 const std::string dtype = a.value("image_dtype", std::string("uint16"));
                 o.imageDtype = dtype == "uint8" ? PixelType::UInt8 : dtype == "float32" ? PixelType::Float32
                                                                                         : PixelType::UInt16;
                 const std::string scaling = a.value("image_scaling", std::string("percentile"));
                 o.scaling = scaling == "cast" ? ExportScaling::Cast : scaling == "minmax" ? ExportScaling::MinMax
                                                                                           : ExportScaling::Percentile;
                 o.provenance = {{"step", Step::number(i)},
                                 {"step_name", wb_.pipeline().at(i).name},
                                 {"kind", wb_.pipeline().at(i).kind},
                                 {"dataset", wb_.hasDataset() ? wb_.dataset().sourcePath : std::string()},
                                 {"pipeline", wb_.pipeline().toJson()}};
                 ArrayPtr array = o.image || o.slices ? out->asInput().materialize() : nullptr;
                 const Array5 empty;
                 const TrainingExportResult r = exportTrainingData(array ? *array : empty, out->meta, *out->labels, o);
                 wb_.recordEvent("training_export", {{"directory", r.directory.string()},
                                                     {"objects", r.objects},
                                                     {"classes", r.classes},
                                                     {"frames", r.frames}});
                 actions_.push_back({ActionRecord::Kind::Run,
                                     "Training data → " + r.directory.string() + " · " + std::to_string(r.objects) + " objects",
                                     "log",
                                     {},
                                     "export_training_data"});
                 return json{{"directory", r.directory.string()},
                             {"files", r.files},
                             {"objects", r.objects},
                             {"slice_objects", r.sliceObjects},
                             {"classes", r.classes},
                             {"frames", r.frames},
                             {"bytes", r.bytes}};
             }});
        // --- looking at an output, and writing one out -----------------------------
        // These three were a session's alone until the window could not write
        // a reconstruction out at all, which is why a three-front comparison
        // had to read the window off a screenshot. They live here, so there is
        // one probe, one statistics and one export_result for every front;
        // what only a host knows -- which step, how to get its output,
        // progress, cancellation -- comes from OutputAccess (tool_api.hpp).
        add({"probe", "Read the values of every channel at one voxel of a step's output, and the label there with its statistics.",
             obj({{"step", inspectStepParam()},
                  {"x", {{"type", "integer"}, {"description", "Column"}}},
                  {"y", {{"type", "integer"}, {"description", "Row"}}},
                  {"z", {{"type", "integer"}, {"description", "Plane (default the middle)"}}},
                  {"t", {{"type", "integer"}, {"description", "Time point (default 0)"}}}},
                 {"x", "y"}),
             [this](const json& a) {
                 const int i = inspectStep(a);
                 const std::shared_ptr<const StepOutput> out = outputFor(i, false);
                 const Dims5 d = out->meta.dims;
                 if (!given(a, "x") || !given(a, "y")) invalidArg("probe needs x and y (voxels of the xy plane)");
                 const Index x = static_cast<Index>(integerArg(a, "x", 0, 0, d.x - 1));
                 const Index y = static_cast<Index>(integerArg(a, "y", 0, 0, d.y - 1));
                 const Index z = static_cast<Index>(integerArg(a, "z", d.z / 2, 0, d.z - 1));
                 const Index t = static_cast<Index>(integerArg(a, "t", 0, 0, d.t - 1));
                 display::DisplayModel& model = displayModel();
                 model.setOutput(out);
                 json values = json::array();
                 for (Index c = 0; c < d.c; ++c) {
                     const std::optional<float> v = model.valueAt(c, t, z, y, x);
                     const std::string label = static_cast<std::size_t>(c) < out->meta.channels.size() ? out->meta.channels[static_cast<std::size_t>(c)].label : std::string();
                     values.push_back({{"channel", c}, {"label", label}, {"value", v ? json(*v) : json(nullptr)}});
                 }
                 json label = nullptr;
                 if (const std::shared_ptr<const LabelVolume> labels = out->labels; labels && !labels->empty() && t < labels->t() && z < labels->z() &&
                                                                                    y < labels->y() && x < labels->x()) {
                     const std::uint32_t id = labels->at(t, z, y, x);
                     if (id != 0) {
                         const LabelStats* st = labels->statsT() == t ? labels->statsOf(id) : nullptr;
                         const LabelAnnotation note = labels->annotationOf(t, id);
                         label = {{"id", id},
                                  {"class", st ? st->cls : note.cls},
                                  {"voxels", st ? json(st->voxels) : json(nullptr)},
                                  {"flags", st ? json(st->flags) : json::array()},
                                  {"reviewed", st ? st->reviewed : note.reviewed}};
                     }
                 }
                 return json{{"step", i + 1}, {"fresh", wb_.outputFresh(i)}, {"x", x}, {"y", y}, {"z", z}, {"t", t}, {"values", values}, {"label", label}};
             }});
        add({"statistics",
             "Measure a step's output: per channel the exact minimum, maximum, mean, standard deviation and NaN count, percentiles, "
             "optionally a histogram and (Load step) the saturated fraction; for labels their count, sizes, classes and flags.",
             obj({{"step", inspectStepParam()},
                  {"t", {{"type", {"integer", "string"}}, {"description", "A time point, or \"all\" (default 0)"}}},
                  {"channels", intList("The channels (default all)")},
                  {"percentiles", numberList("Default [0.1, 1, 50, 99, 99.9]")},
                  {"histogram_bins", {{"type", "integer"}, {"description", "Bins of a histogram per channel (default 0 = none)"}}},
                  {"labels", {{"type", "boolean"}, {"description", "Also the label statistics (default true)"}}},
                  {"max_samples", {{"type", "integer"}, {"description", "Values the percentiles and histogram are taken from, at most (default 4194304)"}}},
                  {"run", {{"type", "boolean"}, {"description", "Run the step first when it has no fresh output (default false)"}}}}),
             [this](const json& a) {
                 const int i = inspectStep(a);
                 const std::shared_ptr<const StepOutput> out = outputFor(i, boolArg(a, "run", false));
                 const DatasetMeta& meta = out->meta;
                 StatisticsOptions o;
                 if (given(a, "t") && a["t"].is_string()) {
                     if (lower(a["t"].get<std::string>()) != "all") invalidArg("'t' is a time point or \"all\"");
                     o.t = -1;
                 } else {
                     o.t = static_cast<Index>(integerArg(a, "t", 0, 0, meta.dims.t - 1));
                 }
                 if (given(a, "channels")) {
                     if (!a["channels"].is_array()) invalidArg("'channels' must be a list of channel indices");
                     for (const json& c : a["channels"]) {
                         const json one = {{"channels", c}};
                         o.channels.push_back(static_cast<Index>(integerArg(one, "channels", 0, 0, meta.dims.c - 1)));
                     }
                 }
                 if (given(a, "percentiles")) {
                     if (!a["percentiles"].is_array()) invalidArg("'percentiles' must be a list of numbers within 0..100");
                     o.percentiles.clear();
                     for (const json& pc : a["percentiles"]) {
                         if (!pc.is_number() || pc.get<double>() < 0.0 || pc.get<double>() > 100.0) invalidArg("'percentiles' must be numbers within 0..100");
                         o.percentiles.push_back(pc.get<double>());
                     }
                 }
                 o.histogramBins = static_cast<int>(integerArg(a, "histogram_bins", 0, 0, 4096));
                 o.maxSamples = static_cast<std::uint64_t>(integerArg(a, "max_samples", std::int64_t{1} << 22, 1000, std::int64_t{1} << 32));
                 // what the camera clipped: the Load step of an integer type only
                 if (i == 0) o.saturationLevel = pixelTypeMaximum(meta);
                 const std::vector<ChannelStatistics> stats = channelStatistics(
                     *out, o, [this](double f) { reportProgress(f, "Measuring intensities"); }, cancelledFn());
                 json channels = json::array();
                 bool sampled = false;
                 for (const ChannelStatistics& st : stats) {
                     json percentiles = json::object();
                     for (const auto& [pc, v] : st.percentiles) percentiles[percentileKey(pc)] = v;
                     const std::string label = static_cast<std::size_t>(st.channel) < meta.channels.size() ? meta.channels[static_cast<std::size_t>(st.channel)].label : std::string();
                     json c = {{"channel", st.channel},
                               {"label", label},
                               {"min", st.min},
                               {"max", st.max},
                               {"mean", st.mean},
                               {"std", st.stddev},
                               {"count", st.count},
                               {"nan", st.nanCount},
                               {"percentiles", percentiles}};
                     if (st.saturatedFraction) c["saturated_fraction"] = *st.saturatedFraction;
                     if (!st.histogram.empty()) c["histogram"] = {{"lo", st.histLo}, {"hi", st.histHi}, {"counts", st.histogram}};
                     sampled = sampled || st.sampled;
                     channels.push_back(std::move(c));
                 }
                 json result = {{"step", i + 1},
                                {"fresh", wb_.outputFresh(i)},
                                {"shape", meta.shapeString()},
                                {"t", o.t < 0 ? json("all") : json(o.t)},
                                {"sampled", sampled},
                                {"channels", channels}};
                 if (boolArg(a, "labels", true) && out->labels && !out->labels->empty())
                     result["labels"] = labelStatistics(*out->labels, o.t < 0 ? 0 : o.t, meta.voxelUm);
                 return result;
             }});
        add({"export_result",
             "Write a step's output to a file: OME-TIFF or TIFF (tiles, compression, BigTIFF, pyramid), a zarr or N5 store, or raw, "
             "in any pixel type with a scaling rule and an optional t / z / channel range; the labels and the pipeline beside it "
             "on request. The format follows the extension (.ome.tif, .tif, .zarr, .n5, .raw) unless named. Overwrites.",
             obj({{"path", {{"type", "string"}, {"description", "The file or store to write"}}},
                  {"step", inspectStepParam()},
                  {"format", enumDesc({"ome-tiff", "tiff", "zarr", "n5", "raw"}, "The container (default from the extension)")},
                  {"dtype", enumDesc({"uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"}, "Pixel type (default float32)")},
                  {"scaling", enumDesc({"cast", "minmax", "fixed", "percentile"}, "How values map into the pixel type (default cast)")},
                  {"range", numberList("scaling fixed: [lo, hi]")},
                  {"percentiles", numberList("scaling percentile: [lo, hi]")},
                  {"t", intList("[first, end) of the time points; end -1 = to the last")},
                  {"z", intList("[first, end) of the planes; end -1 = to the last")},
                  {"channels", intList("The channels (default all)")},
                  {"tiff", {{"type", "object"}, {"description", "{tiled, tile:[w, h], compression: none|lzw|deflate, level, bigtiff, ome, pyramid_levels, downsample}"}}},
                  {"zarr", {{"type", "object"}, {"description", "{version: 2|3, chunk:[c, t, z, y, x], codec, level, shard, pyramid_levels, downsample, ome_ngff}"}}},
                  {"include_labels", {{"type", "boolean"}, {"description", "Write the step's labels beside it (default false)"}}},
                  {"include_pipeline", {{"type", "boolean"}, {"description", "Write <path>.pipeline.toml beside it (default false)"}}},
                  {"labels_only", {{"type", "boolean"}, {"description", "Write only the labels, as one 32-bit TIFF (default false)"}}},
                  {"download", {{"type", "boolean"}, {"description", "True once the user agreed to download an output that is held on the cluster; File > Export result asks them instead"}}},
                  {"run", {{"type", "boolean"}, {"description", "Run the step first when it has no fresh output (default false)"}}}},
                 {"path"}),
             [this](const json& a) {
                 // refused before anything runs
                 if (!given(a, "path") || !a["path"].is_string() || a["path"].get<std::string>().empty())
                     invalidArg("export_result needs 'path', the file to write");
                 refuseNetworkPath(a["path"].get<std::string>(), "path", allowNetworkPaths_);
                 json warnings = json::array();
                 const int i = inspectStep(a);
                 const std::shared_ptr<const StepOutput> out = outputFor(i, boolArg(a, "run", false));
                 const ExportOptions o = exportOptionsFromJson(a, out->meta);
                 const bool labelsOnly = boolArg(a, "labels_only", false);
                 if (!labelsOnly) {
                     if (!exportFormatAvailable(o.format))
                         throw ToolFailure("unsupported", "this build cannot write zarr or N5 stores", "export to .ome.tif or .tif instead");
                     if (const std::string why = validateExport(o, out->meta.dims); !why.empty()) invalidArg(why);
                 } else {
                     // The labels are written whole, as they are: say which options that leaves unused.
                     std::string ignored;
                     for (const char* key : {"dtype", "scaling", "range", "percentiles", "t", "z", "channels", "tiff", "zarr", "include_labels", "include_pipeline"})
                         if (given(a, key)) ignored += (ignored.empty() ? "" : ", ") + std::string(key);
                     if (!ignored.empty()) warnings.push_back("labels_only writes every label as uint32 and ignores " + ignored);
                 }
                 // The window asks before it downloads an output the cluster
                 // holds, and never as a side effect of an export (App::
                 // exportResultDialog). A tool call has nobody to ask, so the
                 // answer is an argument, and without it nothing is read.
                 std::optional<RemoteDownloads::Allow> allow;
                 if (!labelsOnly && heldRemotely(*out)) {
                     if (!boolArg(a, "download", false))
                         throw ToolFailure("needs_download", downloadQuestion(*out, i),
                                           "run the step on the HPC backend and export there, or call export_result again with download:true "
                                           "once the user has agreed to the download",
                                           {{"step", i + 1}, {"bytes", static_cast<double>(std::max<Index>(out->meta.dims.numel(), 0)) * sizeof(float)}, {"where", out->where}});
                     allow.emplace("an export the user asked for");
                 }
                 json r = exportStepOutput(out, wb_.pipeline(), o, labelsOnly, progressFn("Exporting"), cancelledFn());
                 for (const json& w : r["warnings"]) warnings.push_back(w);
                 r.erase("warnings");
                 if (!warnings.empty()) r["warnings"] = warnings;
                 r["step"] = i + 1;
                 const std::string path = r["path"].get<std::string>();
                 wb_.logLine("Exported step " + Step::number(i) + " to " + path);
                 wb_.recordEvent("export", {{"step", i + 1}, {"path", path}, {"format", r["format"]}, {"labels_only", labelsOnly}});
                 actions_.push_back({ActionRecord::Kind::Run,
                                     "Exported step " + Step::number(i) + " · " + wb_.pipeline().at(i).name + " → " + path,
                                     "log", {}, "export_result"});
                 return r;
             }});
        // --- labels by step -------------------------------------------------------
        // The viewer's paint tools (brush, fill, merge, split, delete) as tool
        // calls that name the step and the time point, for the assistant, a
        // script (sirius-app --tool) and sirius-cli, where there is no view.
        // Every edit is one undo entry, the same as a stroke in the viewer.
        const auto labelStepParam = [] {
            return json{{"type", {"integer", "string"}},
                        {"description", "The step whose labels to edit, by number (1 = Load) or name; default the viewed step"}};
        };
        const auto labelStep = [this](const json& a) { return a.contains("step") && !a["step"].is_null() ? resolveStep(a) : wb_.viewedIndex(); };
        const auto labelsOf = [this](int i) {
            std::shared_ptr<LabelVolume> labels = wb_.labelsOfStep(i);
            if (!labels || labels->empty())
                throw ToolFailure("no_labels", "step " + Step::number(i) + (i >= 0 && i < wb_.pipeline().size() ? " " + wb_.pipeline().at(i).name : std::string()) + " has no labels",
                                  "run a segmentation step first (run), or add an import_labels step with a label TIFF");
            return labels;
        };
        const auto frameOf = [](const json& a, const LabelVolume& labels) {
            const Index t = a.contains("t") && a["t"].is_number_integer() ? a["t"].get<Index>() : Index{0};
            if (t < 0 || t >= labels.t())
                throw ToolFailure("invalid_argument", "t " + std::to_string(t) + " is past the labels' " + std::to_string(labels.t()) + " time point(s)");
            return t;
        };
        const auto voxelOf = [](const json& a, const char* key, const LabelVolume& labels) {
            // [x, y, z] as the caller gives it, (z, y, x) as the volume takes it
            if (!a.contains(key) || !a[key].is_array() || a[key].size() != 3)
                throw std::invalid_argument(std::string(key) + " must be [x, y, z]");
            std::array<Index, 3> v{a[key][2].get<Index>(), a[key][1].get<Index>(), a[key][0].get<Index>()};
            if (v[0] < 0 || v[0] >= labels.z() || v[1] < 0 || v[1] >= labels.y() || v[2] < 0 || v[2] >= labels.x())
                throw ToolFailure("invalid_argument", std::string(key) + " " + a[key].dump() + " is outside the volume (x " + std::to_string(labels.x()) + ", y " +
                                                          std::to_string(labels.y()) + ", z " + std::to_string(labels.z()) + ")");
            return v;
        };
        // by value: the tool lambdas below copy xyzOf and outlive this constructor's locals
        const auto xyzOf = [voxelOf](const json& a, const LabelVolume& labels) {
            json v = json::array({a.value("x", -1), a.value("y", -1), a.value("z", -1)});
            json wrapped = {{"at", v}};
            return voxelOf(wrapped, "at", labels);
        };
        const auto xyz = [](const char* what) {
            return json{{"type", "array"}, {"items", {{"type", "integer"}}}, {"description", what}};
        };
        const auto busy = [this] {
            if (!wb_.canEdit()) throw ToolFailure("busy", "A run is in progress: cancel it or wait before editing labels.");
        };
        const auto countAfter = [](const LabelVolume& labels, Index t) {
            return labels.statsT() == t ? json(labels.stats().size()) : json(nullptr);
        };
        add({"list_labels",
             "The labels of a step at one time point: id, voxels, class, confidence, flags, reviewed, bounding box and "
             "centre (x, y, z in voxels). Largest first; `flag` keeps only labels carrying that flag (low conf, small, "
             "touching border, merged?), `unreviewed` only those not yet reviewed. The statistics table moves to that frame.",
             obj({{"step", labelStepParam()},
                  {"t", {{"type", "integer"}, {"description", "time point (default 0)"}}},
                  {"limit", {{"type", "integer"}, {"description", "rows at most (default 200)"}}},
                  {"flag", {{"type", "string"}}},
                  {"unreviewed", {{"type", "boolean"}}}}),
             [this, labelStep, labelsOf, frameOf](const json& a) {
                 const int i = labelStep(a);
                 std::shared_ptr<LabelVolume> labels = labelsOf(i);
                 const Index t = frameOf(a, *labels);
                 if (labels->statsT() != t) labels->recomputeStats(t);
                 const int limit = std::max(1, a.value("limit", 200));
                 const std::string flag = a.value("flag", std::string());
                 const bool unreviewed = a.value("unreviewed", false);
                 std::vector<const LabelStats*> rows;
                 for (const LabelStats& s : labels->stats()) {
                     if (!flag.empty() && std::find(s.flags.begin(), s.flags.end(), flag) == s.flags.end()) continue;
                     if (unreviewed && s.reviewed) continue;
                     rows.push_back(&s);
                 }
                 std::stable_sort(rows.begin(), rows.end(), [](const LabelStats* p, const LabelStats* q) { return p->voxels > q->voxels; });
                 json out = json::array();
                 for (const LabelStats* s : rows) {
                     if (static_cast<int>(out.size()) >= limit) break;
                     out.push_back({{"id", s->id},
                                    {"voxels", s->voxels},
                                    {"class", s->cls},
                                    {"confidence", s->confidence},
                                    {"flags", s->flags},
                                    {"reviewed", s->reviewed},
                                    {"bbox", {{"x0", s->bbox[4]}, {"x1", s->bbox[5]}, {"y0", s->bbox[2]}, {"y1", s->bbox[3]}, {"z0", s->bbox[0]}, {"z1", s->bbox[1]}}},
                                    {"centre", {(s->bbox[4] + s->bbox[5]) / 2, (s->bbox[2] + s->bbox[3]) / 2, (s->bbox[0] + s->bbox[1]) / 2}}});
                 }
                 return json{{"step", i + 1},
                             {"t", t},
                             {"count", labels->stats().size()},
                             {"listed", out.size()},
                             {"max_label", labels->maxLabel()},
                             {"tracked", labels->tracked()},
                             {"shape", {{"t", labels->t()}, {"z", labels->z()}, {"y", labels->y()}, {"x", labels->x()}}},
                             {"labels", out}};
             }});
        add({"paint_label",
             "Paint a ball of one label at a voxel: radius in x/y voxels (default 3), z_radius planes above and below "
             "(default 0). label 0 paints a new object. erase paints background instead, over that label only (0: over any). "
             "One undo entry.",
             obj({{"step", labelStepParam()},
                  {"x", {{"type", "integer"}}},
                  {"y", {{"type", "integer"}}},
                  {"z", {{"type", "integer"}}},
                  {"t", {{"type", "integer"}, {"description", "time point (default 0)"}}},
                  {"label", {{"type", "integer"}, {"description", "the label to paint; 0 (default) starts a new object"}}},
                  {"radius", {{"type", "number"}}},
                  {"z_radius", {{"type", "integer"}}},
                  {"erase", {{"type", "boolean"}}}},
                 {"x", "y", "z"}),
             [this, labelStep, labelsOf, frameOf, xyzOf, busy, countAfter](const json& a) {
                 busy();
                 const int i = labelStep(a);
                 std::shared_ptr<LabelVolume> labels = labelsOf(i);
                 const Index t = frameOf(a, *labels);
                 const std::array<Index, 3> v = xyzOf(a, *labels);
                 const bool erase = a.value("erase", false);
                 std::uint32_t painted = 0;
                 const Index n = wb_.paintLabelAt(i, t, v[0], v[1], v[2], a.value("radius", 3.0), a.value("z_radius", 0), a.value("label", 0u), erase, &painted);
                 const std::string text = (erase ? "Erased " : "Painted label " + std::to_string(painted) + ": ") + std::to_string(n) + " voxels" + (erase ? "" : "") + " · step " + Step::number(i);
                 actions_.push_back({ActionRecord::Kind::Edit, text, n ? "undo" : "", {}, "paint_label"});
                 return json{{"step", i + 1}, {"t", t}, {"label", painted}, {"voxels", n}, {"labels", countAfter(*labels, t)}};
             }});
        add({"fill_label",
             "Flood-fill (6-connected) the region under a voxel with one label: a hole, or a whole object re-numbered. "
             "label 0 starts a new object. One undo entry.",
             obj({{"step", labelStepParam()},
                  {"x", {{"type", "integer"}}},
                  {"y", {{"type", "integer"}}},
                  {"z", {{"type", "integer"}}},
                  {"t", {{"type", "integer"}, {"description", "time point (default 0)"}}},
                  {"label", {{"type", "integer"}, {"description", "the label to fill with; 0 (default) starts a new object"}}}},
                 {"x", "y", "z"}),
             [this, labelStep, labelsOf, frameOf, xyzOf, busy, countAfter](const json& a) {
                 busy();
                 const int i = labelStep(a);
                 std::shared_ptr<LabelVolume> labels = labelsOf(i);
                 const Index t = frameOf(a, *labels);
                 const std::array<Index, 3> v = xyzOf(a, *labels);
                 std::uint32_t filled = 0;
                 const Index n = wb_.fillLabelAt(i, t, v[0], v[1], v[2], a.value("label", 0u), &filled);
                 actions_.push_back({ActionRecord::Kind::Edit, "Filled label " + std::to_string(filled) + ": " + std::to_string(n) + " voxels · step " + Step::number(i), n ? "undo" : "", {}, "fill_label"});
                 return json{{"step", i + 1}, {"t", t}, {"label", filled}, {"voxels", n}, {"labels", countAfter(*labels, t)}};
             }});
        add({"merge_labels",
             "Merge two or more labels into the smallest id among them (on tracked labels, in every frame). One undo entry.",
             obj({{"step", labelStepParam()},
                  {"ids", {{"type", "array"}, {"items", {{"type", "integer"}}}, {"description", "two or more label ids"}}},
                  {"t", {{"type", "integer"}, {"description", "time point (default 0)"}}}},
                 {"ids"}),
             [this, labelStep, labelsOf, frameOf, busy, countAfter](const json& a) {
                 busy();
                 const int i = labelStep(a);
                 std::shared_ptr<LabelVolume> labels = labelsOf(i);
                 const Index t = frameOf(a, *labels);
                 std::vector<std::uint32_t> ids;
                 if (a.contains("ids") && a["ids"].is_array())
                     for (const json& e : a["ids"])
                         if (e.is_number_integer() && e.get<long long>() > 0) ids.push_back(e.get<std::uint32_t>());
                 std::sort(ids.begin(), ids.end());
                 ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
                 if (ids.size() < 2) throw std::invalid_argument("ids must name at least two different labels (> 0)");
                 const Index n = wb_.mergeLabelsAt(i, t, ids);
                 if (n == 0) throw ToolFailure("not_found", "none of the labels " + json(ids).dump() + " but one is in frame " + std::to_string(t), "list_labels shows the ids there");
                 actions_.push_back({ActionRecord::Kind::Edit, "Merged " + std::to_string(ids.size()) + " labels into " + std::to_string(ids.front()) + " · step " + Step::number(i), "undo", {}, "merge_labels"});
                 return json{{"step", i + 1}, {"t", t}, {"into", ids.front()}, {"voxels", n}, {"labels", countAfter(*labels, t)}};
             }});
        add({"split_label",
             "Split one label into two by a watershed of its distance transform from two seed voxels a and b ([x, y, z], "
             "each moved onto the label's nearest voxel); the part around b gets a new id (on tracked labels, the split "
             "follows the track). One undo entry.",
             obj({{"step", labelStepParam()},
                  {"label", {{"type", "integer"}}},
                  {"a", xyz("first seed [x, y, z]")},
                  {"b", xyz("second seed [x, y, z]")},
                  {"t", {{"type", "integer"}, {"description", "time point (default 0)"}}}},
                 {"label", "a", "b"}),
             [this, labelStep, labelsOf, frameOf, voxelOf, busy, countAfter](const json& a) {
                 busy();
                 const int i = labelStep(a);
                 std::shared_ptr<LabelVolume> labels = labelsOf(i);
                 const Index t = frameOf(a, *labels);
                 const std::uint32_t label = a.value("label", 0u);
                 if (label == 0) throw std::invalid_argument("label must be > 0");
                 const std::array<Index, 3> sa = voxelOf(a, "a", *labels), sb = voxelOf(a, "b", *labels);
                 std::uint32_t created = 0;
                 const Index n = wb_.splitLabelAt(i, t, label, sa, sb, &created);
                 if (n == 0 || created == 0)
                     throw ToolFailure("not_split", "label " + std::to_string(label) + " was not split (get_log says why: not in this frame, or the two seeds land on one voxel)",
                                       "pick two seeds inside the label, on different sides of the wall");
                 actions_.push_back({ActionRecord::Kind::Edit, "Split label " + std::to_string(label) + " · new label " + std::to_string(created) + " · step " + Step::number(i), "undo", {}, "split_label"});
                 return json{{"step", i + 1}, {"t", t}, {"label", label}, {"created", created}, {"voxels", n}, {"labels", countAfter(*labels, t)}};
             }});
        add({"delete_label",
             "Delete one label (its voxels become background; on tracked labels, in every frame). One undo entry.",
             obj({{"step", labelStepParam()},
                  {"label", {{"type", "integer"}}},
                  {"t", {{"type", "integer"}, {"description", "time point (default 0)"}}}},
                 {"label"}),
             [this, labelStep, labelsOf, frameOf, busy, countAfter](const json& a) {
                 busy();
                 const int i = labelStep(a);
                 std::shared_ptr<LabelVolume> labels = labelsOf(i);
                 const Index t = frameOf(a, *labels);
                 const std::uint32_t label = a.value("label", 0u);
                 if (label == 0) throw std::invalid_argument("label must be > 0");
                 const Index n = wb_.deleteLabelAt(i, t, label);
                 if (n == 0) throw ToolFailure("not_found", "label " + std::to_string(label) + " is not in frame " + std::to_string(t), "list_labels shows the ids there");
                 actions_.push_back({ActionRecord::Kind::Edit, "Deleted label " + std::to_string(label) + " (" + std::to_string(n) + " voxels) · step " + Step::number(i), "undo", {}, "delete_label"});
                 return json{{"step", i + 1}, {"t", t}, {"label", label}, {"voxels", n}, {"labels", countAfter(*labels, t)}};
             }});
        add({"clear_labels",
             "Remove every label of a step's output, at one time point or (default) at all of them: the volume becomes "
             "background, the pixels stay. One undo entry holding every cleared voxel.",
             obj({{"step", labelStepParam()},
                  {"t", {{"type", "integer"}, {"description", "one time point; default every time point"}}}}),
             [this, labelStep, labelsOf, busy](const json& a) {
                 busy();
                 const int i = labelStep(a);
                 std::shared_ptr<LabelVolume> labels = labelsOf(i);
                 const Index t = a.contains("t") && a["t"].is_number_integer() ? a["t"].get<Index>() : Index{-1};
                 if (t >= labels->t()) throw ToolFailure("invalid_argument", "t " + std::to_string(t) + " is past the labels' " + std::to_string(labels->t()) + " time point(s)");
                 const Index n = wb_.clearLabelsAt(i, t);
                 actions_.push_back({ActionRecord::Kind::Edit, "Cleared the labels of step " + Step::number(i) + (t < 0 ? "" : " at t " + std::to_string(t)) + " (" + std::to_string(n) + " voxels)", n ? "undo" : "", {}, "clear_labels"});
                 return json{{"step", i + 1}, {"t", t < 0 ? json("all") : json(t)}, {"voxels", n}, {"labels", 0}};
             }});
        add({"set_label_reviewed",
             "Mark a label reviewed (or not): what the export sidecar and the review queue record.",
             obj({{"step", labelStepParam()},
                  {"label", {{"type", "integer"}}},
                  {"t", {{"type", "integer"}, {"description", "time point (default 0)"}}},
                  {"reviewed", {{"type", "boolean"}, {"description", "default true"}}}},
                 {"label"}),
             [this, labelStep, labelsOf, frameOf, busy](const json& a) {
                 busy();
                 const int i = labelStep(a);
                 std::shared_ptr<LabelVolume> labels = labelsOf(i);
                 const Index t = frameOf(a, *labels);
                 const std::uint32_t label = a.value("label", 0u);
                 if (label == 0) throw std::invalid_argument("label must be > 0");
                 const bool reviewed = a.value("reviewed", true);
                 if (!wb_.setLabelReviewedAt(i, t, label, reviewed))
                     throw ToolFailure("not_found", "label " + std::to_string(label) + " is not in frame " + std::to_string(t), "list_labels shows the ids there");
                 actions_.push_back({ActionRecord::Kind::Edit, std::string(reviewed ? "Reviewed" : "Unreviewed") + " label " + std::to_string(label) + " · step " + Step::number(i), "", {}, "set_label_reviewed"});
                 return json{{"step", i + 1}, {"t", t}, {"label", label}, {"reviewed", reviewed}};
             }});
        add({"export_labels",
             "Write a step's labels as one 32-bit TIFF (one page per plane, t*z pages for a time series, Deflate), the "
             "file an import_labels step reads back and what Segment > Export labels writes. Overwrites.",
             obj({{"step", labelStepParam()},
                  {"path", {{"type", "string"}, {"description", "The .tif to write (.tif is appended when missing)"}}}},
                 {"path"}),
             [this, labelStep, labelsOf](const json& a) {
                 std::string path = a.value("path", std::string());
                 if (path.empty()) throw std::invalid_argument("path is required");
                 refuseNetworkPath(path, "path", allowNetworkPaths_);
                 if (!wb_.canEdit()) throw ToolFailure("busy", "A run is in progress: cancel it or wait before exporting labels.");
                 const int i = labelStep(a);
                 std::shared_ptr<LabelVolume> labels = labelsOf(i);
                 const std::string low = lower(path);
                 if (low.size() < 4 || (low.compare(low.size() - 4, 4, ".tif") != 0 && (low.size() < 5 || low.compare(low.size() - 5, 5, ".tiff") != 0))) path += ".tif";
                 const Index nt = labels->t(), nz = labels->z(), ny = labels->y(), nx = labels->x();
                 Buffer<std::uint32_t> stack(Shape{nt * nz, ny, nx});
                 for (Index t = 0; t < nt; ++t)
                     for (Index z = 0; z < nz; ++z)
                         std::memcpy(stack.data() + (t * nz + z) * ny * nx, std::as_const(*labels).plane(t, z), static_cast<std::size_t>(ny * nx) * sizeof(std::uint32_t));
                 TiffWriteOptions w;
                 w.compression = TiffCompression::Deflate;
                 w.predictor = true;
                 w.description = "SIRIUS labels · order tzyx · t" + std::to_string(nt) + " z" + std::to_string(nz);
                 try {
                     writeTiffStack<std::uint32_t>(path, stack.view(), w);
                 } catch (const std::exception& e) {
                     throw ToolFailure("io_error", std::string("cannot write ") + path + ": " + e.what());
                 }
                 std::error_code ec;
                 const auto bytes = std::filesystem::file_size(std::filesystem::u8path(path), ec);
                 if (labels->statsT() < 0) labels->recomputeStats(0);
                 wb_.logLine("Labels of step " + Step::number(i) + " written to " + path);
                 wb_.recordEvent("export_labels", {{"step", i + 1}, {"path", path}});
                 actions_.push_back({ActionRecord::Kind::Run, "Labels of step " + Step::number(i) + " → " + path, "log", {}, "export_labels"});
                 return json{{"step", i + 1}, {"path", path}, {"pages", nt * nz}, {"shape", {{"t", nt}, {"z", nz}, {"y", ny}, {"x", nx}}}, {"bytes", ec ? json(nullptr) : json(static_cast<long long>(bytes))}, {"labels", labels->stats().size()}, {"max_label", labels->maxLabel()}};
             }});
        add({"get_log", "The most recent lines of the workbench log.",
             obj({{"lines", {{"type", "integer"}}}}),
             [this](const json& a) {
                 const int n = std::clamp(a.value("lines", 30), 1, 500);
                 const auto& log = wb_.log();
                 json out = json::array();
                 for (std::size_t i = log.size() > static_cast<std::size_t>(n) ? log.size() - static_cast<std::size_t>(n) : 0; i < log.size(); ++i) out.push_back(log[i]);
                 return out;
             }});
        applyTraits(tools_);
    }

    ToolFailure::ToolFailure(std::string code, const std::string& message, std::string hint, nlohmann::json data)
        : std::runtime_error(message), code_(std::move(code)), hint_(std::move(hint)), data_(std::move(data)) {}

    const std::string& ToolFailure::code() const noexcept { return code_; }

    const std::string& ToolFailure::hint() const noexcept { return hint_; }

    const nlohmann::json& ToolFailure::data() const noexcept { return data_; }

    void ToolApi::add(ToolSpec t) { tools_.push_back(std::move(t)); }

    void ToolApi::addTool(ToolSpec spec) {
        for (ToolSpec& t : tools_) {
            if (t.name != spec.name) continue;
            t = std::move(spec);
            return;
        }
        tools_.push_back(std::move(spec));
    }

    bool ToolApi::removeTool(const std::string& name) {
        const auto it = std::find_if(tools_.begin(), tools_.end(), [&](const ToolSpec& t) { return t.name == name; });
        if (it == tools_.end()) return false;
        tools_.erase(it);
        return true;
    }

    const ToolSpec* ToolApi::findTool(const std::string& name) const {
        for (const ToolSpec& t : tools_)
            if (t.name == name) return &t;
        return nullptr;
    }

    const std::vector<ToolSpec>& ToolApi::tools() const noexcept { return tools_; }

    void ToolApi::noteAction(ActionRecord r) { actions_.push_back(std::move(r)); }

    json ToolApi::schemas() const {
        json out = json::array();
        for (const ToolSpec& t : tools_)
            out.push_back({{"type", "function"}, {"function", {{"name", t.name}, {"description", t.description}, {"parameters", t.parameters}}}});
        return out;
    }

    std::vector<std::string> ToolApi::toolNames() const {
        std::vector<std::string> names;
        for (const ToolSpec& t : tools_) names.push_back(t.name);
        return names;
    }

    json ToolApi::call(const std::string& name, const json& args) {
        const ToolSpec* spec = findTool(name);
        if (!spec) return failure("unknown_tool", "unknown tool '" + name + "'");
        // The gate comes before the tool reads its arguments: during a run
        // the workbench refuses these edits with a log line and a quiet
        // return, which the tool would have reported as done.
        if (spec->refusedWhileRunning && !wb_.canEdit())
            return failure("busy", "A run is in progress: cancel it or wait for it to finish before calling " + name + ".");
        // A copy: a tool may add or remove tools, which moves the table.
        const std::function<json(const json&)> fn = spec->fn;
        const std::size_t firstRecord = actions_.size();
        const std::uint64_t before = wb_.history().revision();
        json result;
        try {
            result = fn(args.is_object() ? args : json::object());
        } catch (const ToolFailure& e) {
            result = failure(e.code(), e.what(), e.hint(), e.data());
        } catch (const WorkerStartError& e) {
            result = failure("worker_unavailable", e.what(), e.hint, e.toJson());
        } catch (const std::invalid_argument& e) {
            result = failure("invalid_argument", e.what());
        } catch (const std::out_of_range& e) {
            result = failure("invalid_argument", e.what());
        } catch (const json::exception& e) {
            // an argument of another type than the schema says (a string
            // where a number goes), found by the tool's get<>()
            result = failure("invalid_argument", e.what());
        } catch (const std::exception& e) {
            // isCancellation also recognises the library's untyped "cancelled"
            result = failure(isCancellation(e) ? "cancelled" : "failed", e.what());
        } catch (...) {
            result = failure("failed", "unknown error");
        }
        // An "undo" card undoes its own call's change, not whatever is
        // newest by then: it keeps the revisions around the call. A call
        // that pushed nothing (a value set to what it was, an edit
        // refused during a run) has nothing of its own to undo.
        const std::uint64_t after = wb_.history().revision();
        for (std::size_t i = firstRecord; i < actions_.size(); ++i) {
            ActionRecord& r = actions_[i];
            if (r.link != "undo") continue;
            if (after == before) {
                r.link.clear();
            } else {
                r.revBefore = before;
                r.revAfter = after;
            }
        }
        return result;
    }

    int ToolApi::resolveStep(const json& args, const char* key) const { return resolveStepIndex(wb_.pipeline(), args, key); }

    ToolApi::~ToolApi() = default;

    void ToolApi::releaseOutputCaches() noexcept {
        if (model_) model_->setOutput(nullptr);
    }

    display::DisplayModel& ToolApi::displayModel() {
        if (!model_) model_ = std::make_unique<display::DisplayModel>();
        return *model_;
    }

    void ToolApi::reportProgress(double fraction, const std::string& message) const {
        if (output_.progress) output_.progress(fraction, message);
    }

    std::function<void(double, const std::string&)> ToolApi::progressFn(std::string message) const {
        if (!output_.progress) return {};
        return [this, message](double f, const std::string& m) { reportProgress(f, m.empty() ? message : m); };
    }

    std::function<bool()> ToolApi::cancelledFn() const { return output_.cancelled; }

    int ToolApi::inspectStep(const json& args) const {
        if (args.is_object() && args.contains("step") && !args["step"].is_null()) return resolveStep(args);
        const int i = output_.defaultStep ? output_.defaultStep() : wb_.viewedIndex();
        // A host's rule is its own; it still has to name a step that exists.
        if (i < 0 || i >= wb_.pipeline().size())
            throw ToolFailure("unknown_step", "there is no step " + std::to_string(i + 1), "name the step: 1 is Load");
        return i;
    }

    // The window's answer when no host gave one: the output the workbench
    // holds, and the run hook (the Run button's path) when the call says run.
    std::shared_ptr<const StepOutput> ToolApi::outputFor(int index, bool runIfNeeded) {
        if (output_.output) return output_.output(index, runIfNeeded);
        if (!wb_.hasDataset()) throw ToolFailure("no_dataset", "no dataset is open", "open a dataset first");
        const std::string which = "step " + std::to_string(index + 1) + " (" + wb_.pipeline().at(index).name + ")";
        std::shared_ptr<const StepOutput> out = wb_.output(index);
        const bool usable = out && (out->array || out->source);
        if (usable && (!runIfNeeded || wb_.outputFresh(index))) return out;
        if (!runIfNeeded) throw ToolFailure("not_computed", which + " has not been computed", "run it first, or pass run:true");
        if (!runHook_) throw ToolFailure("unsupported", "running is not available in this context");
        const json outcome = runHook_(index);
        if (outcome.is_object() && outcome.contains("error") && outcome["error"].is_string() && !outcome["error"].get<std::string>().empty())
            throw ToolFailure("failed", "the run of " + which + " failed: " + outcome["error"].get<std::string>());
        out = wb_.output(index);
        if (!out || !(out->array || out->source)) throw ToolFailure("not_computed", which + " produced no output", "get_log says what the run did");
        return out;
    }

    int ToolApi::resolveStepIndex(const Pipeline& p, const json& args, const char* key) {
        const std::string there = " (there are " + std::to_string(p.size()) + ", 1 is Load)";
        if (!args.is_object() || !args.contains(key)) throw ToolFailure("invalid_argument", std::string("missing '") + key + "'");
        const json& v = args[key];
        // a model writes 2 as 2.0 now and then: an integral number is a number
        if (v.is_number() && (v.is_number_integer() || v.get<double>() == std::floor(v.get<double>()))) {
            // compared as a double first: a huge number does not fit an int
            const double n = v.get<double>();
            if (!(n >= 1.0 && n <= static_cast<double>(p.size())))
                throw ToolFailure("unknown_step", "no step " + v.dump() + there, "get_state lists the steps");
            return static_cast<int>(n) - 1;
        }
        if (v.is_string()) {
            const std::string s = v.get<std::string>();
            // "" is a part of every name: it would pick the Load step
            if (s.empty()) throw ToolFailure("invalid_argument", std::string("'") + key + "' is empty");
            try {
                const int i = std::stoi(s) - 1;
                if (i >= 0 && i < p.size()) return i;
            } catch (...) {
            }
            const std::string ls = lower(s);
            for (int i = 0; i < p.size(); ++i)
                if (lower(p.at(i).name) == ls || lower(p.at(i).kind) == ls) return i;
            for (int i = 0; i < p.size(); ++i)
                if (lower(p.at(i).name).find(ls) != std::string::npos) return i;
            throw ToolFailure("unknown_step", "no step named '" + s + "'" + there, "get_state lists the steps");
        }
        throw ToolFailure("invalid_argument", std::string("'") + key + "' must be a number or a name");
    }

    json ToolApi::stepJson(int i) const {
        const Step& s = wb_.pipeline().at(i);
        return {{"step", i + 1},
                {"number", Step::number(i)},
                {"kind", s.kind},
                {"name", s.name},
                {"enabled", s.enabled},
                {"pinned", s.pinned},
                {"cache", toString(s.cache)},
                {"params", s.params.toJson()},
                {"summary", wb_.stepSummary(i)},
                {"selected", i == wb_.selectedIndex()},
                {"viewed", i == wb_.viewedIndex()}};
    }

    json ToolApi::contextSnapshot() const {
        json state = const_cast<ToolApi*>(this)->call("get_state", json::object());
        const int sel = wb_.selectedIndex();
        json diag;
        try {
            const Diagnostics d = wb_.selectedDiagnostics();
            diag = {{"summary", d.summary}, {"warnings", d.warnings}, {"footer", d.footer}};
            if (d.table) diag["table"] = {{"header", d.table->header}, {"rows", d.table->rows}};
            json facts = json::object();
            for (const DiagnosticFact& f : d.facts) facts[f.key] = f.value;
            diag["facts"] = facts;
        } catch (const std::exception&) {
        }
        state["selected_step_diagnostics"] = diag;
        state["selected_step_validation"] = {{"errors", wb_.stepValidation(sel).errors}, {"warnings", wb_.stepValidation(sel).warnings}};
        return state;
    }

    std::string ToolApi::systemPrompt() const {
        return "You are the assistant inside SIRIUS, a desktop workbench for microscopy image processing "
               "(structured illumination reconstruction, deconvolution, deskew, reductions, contrast, channel merge, "
               "stitching, registration, segmentation, volume rendering). The user builds an ordered stack of "
               "processing steps; step 1 is always Load. Steps are referred to by their number as shown (1, 2, 3 ...) "
               "or by name. Use the tools to inspect state and diagnostics before answering questions about results, "
               "and to make changes when the user asks for them; every change is undoable and is shown to the user as "
               "an action card, so state what you did briefly rather than repeating parameters. Runs can take from "
               "seconds to minutes; run only when asked or when a change needs a result to be judged. Be concise and "
               "specific: quote numbers from diagnostics (modulation depths, k0, percentiles, label counts). If a "
               "request is ambiguous about which step, ask. Answer in Markdown; write formulas as LaTeX between $...$ "
               "(inline) or $$...$$ (on its own line). Current workbench state follows as JSON.";
    }

    std::vector<ActionRecord> ToolApi::takeActions() {
        std::vector<ActionRecord> out = std::move(actions_);
        actions_.clear();
        return out;
    }

} // namespace sirius::app
