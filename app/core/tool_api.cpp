#include "core/tool_api.hpp"

#include "core/training_export.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace sirius::app {

    using json = nlohmann::json;

    namespace {
        json stepParam() {
            return {{"type", {"integer", "string"}},
                    {"description", "Step number as shown in the operations list (1 = Load, 2 = second step, ...) or a step name"}};
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
    } // namespace

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
                     if (m.sim.present) ds["sim"] = {{"ndirs", m.sim.ndirs}, {"nphases", m.sim.nphases}};
                 }
                 return json{{"dataset", ds},
                             {"steps", steps},
                             {"selected_step", wb_.selectedIndex() + 1},
                             {"viewed_step", wb_.viewedIndex() + 1},
                             {"view", wb_.viewState().toJson()},
                             {"backend", toString(wb_.backend())},
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
                     j["has_output"] = true;
                     j["output_fresh"] = wb_.outputFresh(i);
                     j["diagnostics_summary"] = out->diagnostics.summary;
                     j["note"] = out->note;
                 } else {
                     j["has_output"] = false;
                 }
                 return j;
             }});
        add({"add_step",
             "Append a processing step of the given kind (see list_operations); becomes selected and viewed. Optional parameters are applied.",
             obj({{"kind", {{"type", "string"}}},
                  {"params", {{"type", "object"}, {"description", "parameter key/value pairs to set"}}},
                  {"at", {{"type", "integer"}, {"description", "1-based position to insert at (default: end)"}}}},
                 {"kind"}),
             [this](const json& a) {
                 const std::string kind = a.value("kind", "");
                 if (!findOperation(kind)) throw std::invalid_argument("unknown operation kind '" + kind + "'");
                 if (!wb_.canEdit()) throw std::runtime_error("a run is in progress: cancel it or wait before editing the pipeline");
                 int at = -1;
                 if (a.contains("at") && a["at"].is_number_integer()) at = a["at"].get<int>() - 1;
                 const StepId id = wb_.addStep(kind, at);
                 const int i = wb_.pipeline().indexOf(id);
                 if (a.contains("params") && a["params"].is_object() && !a["params"].empty()) {
                     ParamSet p = wb_.pipeline().at(i).params;
                     for (auto it = a["params"].begin(); it != a["params"].end(); ++it) {
                         bool known = false;
                         for (const ParamSpec& s : wb_.pipeline().at(i).op().info().params)
                             if (s.key == it.key()) {
                                 p.set(s.key, coerceToSpec(s, it.value()));
                                 known = true;
                             }
                         if (!known) throw std::invalid_argument("unknown parameter '" + it.key() + "' for " + kind);
                     }
                     wb_.setStepParams(i, p, "Set parameters of " + wb_.pipeline().at(i).name);
                 }
                 actions_.push_back({ActionRecord::Kind::Param, "Added step " + Step::number(i) + " · " + wb_.pipeline().at(i).name, "undo", {}, "add_step"});
                 return stepJson(i);
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
                 for (auto it = a["params"].begin(); it != a["params"].end(); ++it) {
                     const ParamSpec* spec = nullptr;
                     for (const ParamSpec& sp : s.op().info().params)
                         if (sp.key == it.key()) spec = &sp;
                     if (!spec) throw std::invalid_argument("unknown parameter '" + it.key() + "' for " + s.kind);
                     const ParamValue v = coerceToSpec(*spec, it.value());
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
                     throw std::runtime_error("A run is in progress: cancel it or wait before applying a preset.");
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
                 const int target = a.contains("step") ? resolveStep(a) : wb_.pipeline().size() - 1;
                 if (!runHook_) return json{{"error", "running is not available in this context"}};
                 json r = runHook_(target);
                 std::string text = "Ran to step " + Step::number(target) + " · " + wb_.pipeline().at(target).name;
                 if (r.contains("seconds")) {
                     char buf[32];
                     std::snprintf(buf, sizeof buf, " · %.1f s", r["seconds"].get<double>());
                     text += buf;
                 }
                 if (r.contains("error") && !r["error"].get<std::string>().empty()) text += " · failed: " + r["error"].get<std::string>();
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
             "Change the viewer: mode (ortho|3d|compare), tool (nav|probe|measure|roi|paint), z, t, crosshair [x, y], labels overlay, label (select one and jump to it), solo (draw only the selected label), channel visibility list, yaw/pitch, diagnostics tab is not part of this.",
             obj({{"mode", {{"type", "string"}, {"enum", {"ortho", "3d", "compare"}}}},
                  {"tool", {{"type", "string"}, {"enum", {"nav", "probe", "measure", "roi", "paint"}}}},
                  {"z", {{"type", "integer"}}},
                  {"t", {{"type", "integer"}}},
                  {"crosshair", {{"type", "array"}, {"items", {{"type", "integer"}}}, {"description", "[x, y] or [x, y, z]"}}},
                  {"labels", {{"type", "boolean"}}},
                  {"label", {{"type", "integer"}, {"description", "label id to select; the view jumps to it"}}},
                  {"solo", {{"type", "boolean"}, {"description", "show only the selected label"}}},
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
                 const DatasetMeta meta = wb_.outputMetaOf(wb_.viewedIndex());
                 s.z = std::clamp<Index>(s.z, 0, std::max<Index>(meta.dims.z - 1, 0));
                 s.t = std::clamp<Index>(s.t, 0, std::max<Index>(meta.dims.t - 1, 0));
                 wb_.setViewState(s);
                 if (focus) wb_.focusLabel(focus);
                 actions_.push_back({ActionRecord::Kind::View, "Viewer → " + (text.empty() ? std::string("unchanged") : text), "view",
                                     json{{"view", wb_.viewState().toJson()}}, "set_view"});
                 return wb_.viewState().toJson();
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
                 if (kind.empty()) kind = wb_.pipeline().at(wb_.selectedIndex()).kind;
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
        add({"set_backend", "Compute backend for runs: CUDA, CPU or HPC.",
             obj({{"backend", {{"type", "string"}, {"enum", {"CUDA", "CPU", "HPC"}}}}}, {"backend"}),
             [this](const json& a) {
                 auto b = backendFromString(a.value("backend", ""));
                 if (!b) throw std::invalid_argument("backend must be CUDA, CPU or HPC");
                 wb_.setBackend(*b);
                 actions_.push_back({ActionRecord::Kind::Param, std::string("Backend → ") + toString(*b), "", {}, "set_backend"});
                 return json{{"backend", toString(*b)}};
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
                 // The export materializes the step's input through the same
                 // ArraySource a running job may be reading; keep it under the
                 // run-state rule every other entry point follows.
                 if (!wb_.canEdit())
                     throw std::runtime_error("A run is in progress: cancel it or wait before exporting training data.");
                 const int i = a.contains("step") ? resolveStep(a) : wb_.viewedIndex();
                 std::shared_ptr<const StepOutput> out = wb_.output(i);
                 if (!out) throw std::invalid_argument("step " + Step::number(i) + " has not been computed yet; run it first");
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
        add({"get_log", "The most recent lines of the workbench log.",
             obj({{"lines", {{"type", "integer"}}}}),
             [this](const json& a) {
                 const int n = std::clamp(a.value("lines", 30), 1, 500);
                 const auto& log = wb_.log();
                 json out = json::array();
                 for (std::size_t i = log.size() > static_cast<std::size_t>(n) ? log.size() - static_cast<std::size_t>(n) : 0; i < log.size(); ++i) out.push_back(log[i]);
                 return out;
             }});
    }

    void ToolApi::add(Tool t) { tools_.push_back(std::move(t)); }

    json ToolApi::schemas() const {
        json out = json::array();
        for (const Tool& t : tools_)
            out.push_back({{"type", "function"}, {"function", {{"name", t.name}, {"description", t.description}, {"parameters", t.parameters}}}});
        return out;
    }

    std::vector<std::string> ToolApi::toolNames() const {
        std::vector<std::string> names;
        for (const Tool& t : tools_) names.push_back(t.name);
        return names;
    }

    json ToolApi::call(const std::string& name, const json& args) {
        for (const Tool& t : tools_) {
            if (t.name != name) continue;
            try {
                return t.fn(args.is_object() ? args : json::object());
            } catch (const std::exception& e) {
                return json{{"error", e.what()}};
            }
        }
        return json{{"error", "unknown tool '" + name + "'"}};
    }

    int ToolApi::resolveStep(const json& args, const char* key) const {
        if (!args.contains(key)) throw std::invalid_argument(std::string("missing '") + key + "'");
        const json& v = args[key];
        const Pipeline& p = wb_.pipeline();
        if (v.is_number_integer()) {
            const int i = v.get<int>() - 1;
            if (i < 0 || i >= p.size()) throw std::invalid_argument("no step " + std::to_string(i + 1) + " (there are " + std::to_string(p.size()) + ")");
            return i;
        }
        if (v.is_string()) {
            const std::string s = v.get<std::string>();
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
            throw std::invalid_argument("no step named '" + s + "'");
        }
        throw std::invalid_argument("'step' must be a number or a name");
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
