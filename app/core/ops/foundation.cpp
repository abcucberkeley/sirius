// The Foundation step: a trained model folder, run by the Python worker.
//
// A model is a self-contained folder (core/model_folder.hpp): model.py, the
// model's own API; model.json, what decides a correct answer as plain data
// (the tasks it offers, the input contract, the decode rule); the weights;
// and the model's code under _lib/. The worker imports model.py from the
// folder and calls it -- no latents package anywhere. The Model parameter is
// the folder, on this computer or on the cluster (cluster://host/path, which
// the engine on the node reads and hands its Python worker as a node path).
//
// Segment sends (c, t, z, y, x) in one call and gets one label volume per
// time point. The model normalises its input itself, so the raw intensities
// go; a threshold or minimum size left at zero means model.json's, the values
// the model was scored with.
//
// A model whose tasks include "prompt" also offers Prompt: the person points
// at objects -- a box around one, a click on it, a scribble over it, with the
// viewer's Prompt tool or as an agent sets them -- and gets those objects
// back, in 3-D, one mask per object. Each object goes to the worker with all
// of its prompts together (the joint `objects` form), so a background click
// on an object corrects that object's mask, and its mask comes back labelled
// with the object's id on every re-run. That task sends one time point per
// call, as the decoder takes it, and only the frames someone prompted.
//
// The Task choice is what model.json lists: the step refuses a task the model
// does not offer, by name, before anything is sent.
#include "core/ops/common.hpp"
#include "core/ops/builtin.hpp"
#include "core/model_folder.hpp"
#include "core/rpc.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <system_error>
#include <vector>

namespace sirius::app {

    namespace {

        constexpr const char* kSegment = kModelSegmentLabel;
        constexpr const char* kPrompt = kPromptTask;
        constexpr const char* kOneChannel = "Selected channel";
        constexpr const char* kAllChannels = "All channels";

        bool onCluster(const std::string& path) { return path.rfind("cluster://", 0) == 0; }

        std::string compact(double v) {
            char b[32];
            std::snprintf(b, sizeof b, "%.3g", v);
            return b;
        }

        // "coat-sam-s2 v1": model.json's name when the folder is here to read, else the path's
        std::string modelName(const std::string& model) {
            if (model.empty()) return "no model";
            if (model.rfind("cluster://", 0) != 0 && !isOldBundlePath(model))
                if (const std::optional<ModelFolderFacts> f = readModelFolder(model); f && !f->name.empty())
                    return f->version.empty() ? f->name : f->name + " " + f->version;
            std::filesystem::path p = std::filesystem::u8path(model);
            while (!p.empty() && p.filename().empty()) p = p.parent_path();   // a trailing slash
            const std::string leaf = p.filename().u8string();
            const std::string parent = p.parent_path().filename().u8string();
            // <models>/<name>/<version>: "name version"
            if (!parent.empty() && leaf.size() <= 8 && (leaf[0] == 'v' || leaf[0] == 'V')) return parent + " " + leaf;
            return leaf.empty() ? model : leaf;
        }

        class FoundationOperation final : public Operation {
        public:
            FoundationOperation() {
                info_.kind = "foundation";
                info_.name = "Foundation model";
                info_.group = "Segment";
                info_.kindLabel = "SEGMENT";
                info_.diagnostics = DiagnosticsKind::Segment;
                info_.defaultCache = CachePolicy::Disk;
                info_.separableOverT = false;   // one call for the whole (c, t, z, y, x)
                info_.hasGpuPath = true;
                info_.remoteCapable = true;
                info_.producesLabels = true;
                info_.helpPage = "foundation";
                info_.params = {
                    pathParam("model", "Model")
                        .asDirectory()
                        .withHelp("A model folder: model.py, model.json and its weights, as latents scripts/export_model.py writes "
                                  "them (<models>/<name>/<version>). On this computer or on the cluster; Models\xE2\x80\xA6 lists the "
                                  "models of your models folders"),
                    choiceParam("task", "Task", {kSegment, kPrompt}, kSegment)
                        .withHelp("Segment finds every object; Prompt segments the objects you point at with the viewer's Prompt "
                                  "tool (a box, a click, a scribble). Only what the model offers is shown: Prompt needs a model "
                                  "with a prompt decoder"),
                    promptsParam(kPromptsKey, "Prompts")
                        .visibleWhen("task", {kPrompt})
                        .withHelp("Where the objects are, in voxels of the input: points, boxes and scribbles, each on "
                                  "one time point and belonging to one object. An object's prompts are sent together, so "
                                  "a background point on it corrects its mask, labelled with the object's id. Placed with "
                                  "the viewer's Prompt tool"),
                    choiceParam("channels", "Channels", {kOneChannel, kAllChannels}, kOneChannel)
                        .withHelp("Send one channel, or all of them when the model was trained on several (model.json says how "
                                  "many it takes)"),
                    channelParam("input_channel", "Input channel", 0).visibleWhen("channels", {kOneChannel}),
                    doubleParam("threshold", "Threshold", 0.0)
                        .range(0.0, 1.0, 0.01, 2)
                        .visibleWhen("task", {kSegment})
                        .withHelp("Foreground probability cut. 0 uses the model's own (model.json's decode), the value it was scored with"),
                    intParam("min_voxels", "Min. voxels", 0)
                        .range(0, 1000000000)
                        .withHelp("Drop smaller objects. 0 keeps all on Prompt and uses the model's own on Segment"),
                    doubleParam("label_opacity", "Label opacity", 0.45).range(0.0, 1.0, 0.05, 2),
                    stringParam("class_name", "Class", "object").asAdvanced(),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            std::string summary(const ParamSet& p, const DatasetMeta&) const override {
                const std::string name = modelName(p.getString("model"));
                std::string chans = p.getString("channels", kOneChannel) == kAllChannels ? "all channels" : "";
                const std::string task = modelTaskOfLabel(p.getString("task", kSegment));
                if (task == "prompt") return joinSummary({name, task, toDisplayString(promptsValue(promptsOf(p))), chans});
                return joinSummary({name, task, chans});
            }

            Validation validate(const ParamSet& p, const DatasetMeta& in) const override {
                Validation v = Operation::validate(p, in);
                const std::string model = p.getString("model");
                const bool allChannels = p.getString("channels", kOneChannel) == kAllChannels;
                const std::string task = modelTaskOfLabel(p.getString("task", kSegment));
                std::error_code ec;
                if (model.empty()) {
                    v.errors.push_back("Choose a model folder (Models\xE2\x80\xA6 lists yours).");
                } else if (isOldBundlePath(model)) {
                    v.errors.push_back(oldBundleMessage(model));
                } else if (!onCluster(model) && !std::filesystem::exists(std::filesystem::u8path(model), ec)) {
                    // A warning, not an error: the worker opens the folder, and it
                    // may see one this machine cannot. One that cannot says so.
                    v.warnings.push_back("Model folder not found on this machine: " + model + " (fine when the worker runs where it is)");
                } else if (!onCluster(model)) {
                    // On this machine -- or on the node, where the engine checks
                    // the step with the node's own path: model.json says what the
                    // model offers and takes.
                    std::string why;
                    const std::optional<ModelFolderFacts> facts = readModelFolder(model, &why);
                    if (!facts) {
                        v.errors.push_back(why);
                    } else {
                        checkAgainst(*facts, task, allChannels, in, v);
                    }
                }
                if (in.rgb) v.errors.push_back("The model needs intensity channels, not an RGB merge.");
                if (isPromptStep(p)) validatePrompts(promptsOf(p), in, v);
                return v;
            }

            DatasetMeta outputMeta(const ParamSet&, const DatasetMeta& in) const override { return in; }

            std::size_t estimatedOutputBytes(const ParamSet&, const DatasetMeta& in) const override {
                return in.dims.bytes() +
                       static_cast<std::size_t>(in.dims.t * in.dims.z * in.dims.planeSize()) * sizeof(std::uint32_t);
            }

            StepOutput run(const StepInput& input, const ParamSet& p, const StepContext& ctx) const override {
                const Validation v = validate(p, input.meta);
                if (!v.ok()) throw std::runtime_error(v.firstError());
                if (isPromptStep(p)) return runPrompt(input, p, ctx);
                requireWorker(ctx);

                const DatasetMeta& meta = input.meta;
                const Dims5& d = meta.dims;
                const bool allChannels = p.getString("channels", kOneChannel) == kAllChannels;
                const Index channel = allChannels ? 0 : p.getInt("input_channel", 0);
                const Index nc = allChannels ? d.c : 1;

                StepOutput out;
                out.meta = meta;
                out.array = input.materialize([&](double f, const std::string& m) { ctx.report(0.05 * f, m); });

                // One contiguous (c, t, z, y, x) block. The application stores
                // planes per (c, t), so this is a copy, the same size as the
                // array already in hand.
                const Index volume = d.z * d.planeSize();
                std::vector<float> flat(static_cast<std::size_t>(nc) * d.t * volume);
                for (Index c = 0; c < nc; ++c)
                    for (Index t = 0; t < d.t; ++t) {
                        ctx.throwIfCancelled();
                        const BufferView<const float> vol = out.array->volume(allChannels ? c : channel, t);
                        std::copy_n(vol.data(), volume, flat.data() + (static_cast<std::size_t>(c) * d.t + t) * volume);
                    }

                const nlohmann::json params = {
                    {"model", p.getString("model")},
                    {"task", "segment"},
                    {"threshold", p.getDouble("threshold", 0.0)},
                    {"min_voxels", p.getInt("min_voxels", 0)},
                    {"voxel_um", {meta.voxelUm[0], meta.voxelUm[1], meta.voxelUm[2]}},
                    {"device", workerDevice(ctx)},
                };

                rpc::TensorRef in;
                in.name = "input";
                in.dtype = "float32";
                in.shape = {nc, d.t, d.z, d.y, d.x};
                in.data = flat.data();
                in.nbytes = flat.size() * sizeof(float);

                ctx.report(0.1, "model");
                const auto t0 = std::chrono::steady_clock::now();
                WorkerResult r = ctx.remote->call(
                    "run", {{"kind", "foundation"}, {"params", params}}, {in},
                    [&](double f, const std::string& m) { ctx.report(0.1 + 0.85 * f, m); },
                    [&] { return ctx.isCancelled(); });
                ctx.throwIfCancelled();
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

                const rpc::Tensor* got = nullptr;
                const rpc::Tensor* confidence = nullptr;
                for (const rpc::Tensor& tensor : r.tensors) {
                    if (tensor.name == "labels") got = &tensor;
                    else if (tensor.name == "confidence") confidence = &tensor;
                }
                if (!got) throw std::runtime_error("the worker returned no 'labels' tensor");
                if (got->shape.size() != 4 || got->shape[0] != d.t || got->shape[1] != d.z || got->shape[2] != d.y ||
                    got->shape[3] != d.x)
                    throw std::runtime_error("the worker's labels do not match the volume");
                const bool confMatches = confidence && confidence->shape.size() == 4 && confidence->shape[0] == d.t &&
                                         confidence->shape[1] == d.z && confidence->shape[2] == d.y &&
                                         confidence->shape[3] == d.x;

                auto labels = std::make_shared<LabelVolume>(d.t, d.z, d.y, d.x);
                const std::uint32_t* src = got->asUInt32();
                std::uint32_t total = 0;
                for (Index t = 0; t < d.t; ++t) {
                    ctx.throwIfCancelled();
                    // The labels are used as they come: Min. voxels is the worker's to apply.
                    std::copy_n(src + static_cast<std::size_t>(t) * volume, volume, labels->volume(t));
                    labels->recomputeStats(t, confMatches ? confidence->asFloat32() + static_cast<std::size_t>(t) * volume
                                                          : nullptr);
                    for (const LabelStats& s : labels->stats()) total = std::max(total, s.id);
                }
                const std::string className = p.getString("class_name", "object");
                for (LabelStats& s : labels->stats()) s.cls = className;

                out.labels = labels;
                out.ranOn = ctx.backend;
                out.seconds = seconds;

                Diagnostics diag = labelDiagnostics(*labels, summary(p, meta));
                diag.facts.push_back({"Model", r.result.value("model", modelName(p.getString("model")))});
                diag.facts.push_back({"Task", "segment"});
                diag.facts.push_back({"Threshold", formatNumber(r.result.value("threshold", 0.0), 2)});
                diag.facts.push_back({"Min. voxels", std::to_string(r.result.value("min_voxels", 0LL))});
                diag.facts.push_back({"Channels sent", std::to_string(nc)});
                diag.facts.push_back({"Objects", std::to_string(r.result.value("objects", 0LL))});
                addWorkerWarnings(diag, r.result);
                diag.summary = summary(p, meta) + " · " + std::to_string(total) + " labels";

                char note[240];
                std::snprintf(note, sizeof note, "%.1f s · %s · %s", seconds, diag.summary.c_str(),
                              ctx.remote->capabilities().device.empty() ? "worker"
                                                                        : ctx.remote->capabilities().device.c_str());
                out.note = note;
                out.diagnostics = std::move(diag);
                ctx.report(1.0, "");
                return out;
            }

        private:
            // What model.json says, against what the step asks and what the image is.
            static void checkAgainst(const ModelFolderFacts& f, const std::string& task, bool allChannels, const DatasetMeta& in,
                                     Validation& v) {
                if (!f.offers(task)) {
                    std::string offered;
                    for (const std::string& t : f.tasks) offered += (offered.empty() ? "" : ", ") + t;
                    v.errors.push_back(task == "prompt" ? f.name + " " + f.version + " cannot be prompted: it has no prompt decoder (it offers " +
                                                              offered + "). Choose Segment, or a promptable model."
                                                        : f.name + " " + f.version + " offers " + offered + ", not " + task + ".");
                }
                const Index sent = allChannels ? in.dims.c : 1;
                if (f.channels <= 1 && sent > 1)
                    v.errors.push_back(f.name + " takes one channel: choose Selected channel.");
                else if (f.channels > 1 && sent != f.channels)
                    v.errors.push_back(f.name + " takes " + std::to_string(f.channels) + " channels" +
                                       (f.channelMerge.empty() ? std::string() : " (" + f.channelMerge + ")") + ": choose All channels on an image with " +
                                       std::to_string(f.channels) + ".");
                if (f.voxelUm.size() == 3) {
                    std::string far;
                    for (int a = 0; a < 3; ++a) {
                        const double g = in.voxelUm[static_cast<std::size_t>(a)], m = f.voxelUm[static_cast<std::size_t>(a)];
                        if (g > 0 && m > 0 && (g / m > 1.5 || m / g > 1.5)) far += std::string(far.empty() ? "" : ", ") + "xyz"[a];
                    }
                    if (!far.empty())
                        v.warnings.push_back(f.name + " was trained at " + compact(f.voxelUm[2]) + " x " + compact(f.voxelUm[1]) + " x " +
                                             compact(f.voxelUm[0]) + " um (z, y, x); this image is " + compact(in.voxelUm[2]) + " x " +
                                             compact(in.voxelUm[1]) + " x " + compact(in.voxelUm[0]) +
                                             ", far on " + far + ". Nothing in the model adapts to scale: resample the image first for better objects.");
                }
            }

            static void addWorkerWarnings(Diagnostics& diag, const nlohmann::json& result) {
                const auto it = result.find("warnings");
                if (it == result.end() || !it->is_array()) return;
                for (const nlohmann::json& w : *it)
                    if (w.is_string() && std::find(diag.warnings.begin(), diag.warnings.end(), w.get<std::string>()) == diag.warnings.end())
                        diag.warnings.push_back(w.get<std::string>());
            }

            static void requireWorker(const StepContext& ctx) {
                if (!ctx.remote)
                    throw std::runtime_error("The Foundation step needs the Python worker, which is not available here "
                                             "(see the worker message in the log), or the HPC backend");
                if (!ctx.remote->supports("foundation"))
                    throw std::runtime_error("The connected worker does not run model folders (" + ctx.remote->capabilities().hostname +
                                             "): it is older than this SIRIUS. See Help ▸ Foundation model");
            }

            // Prompt: the objects a person pointed at. One time point per
            // call, because the prompt decoder takes one frame; a frame
            // nobody pointed at is left empty and costs no call at all.
            StepOutput runPrompt(const StepInput& input, const ParamSet& p, const StepContext& ctx) const {
                const DatasetMeta& meta = input.meta;
                const Dims5& d = meta.dims;
                const std::vector<Prompt> prompts = promptsOf(p);
                std::vector<FramePrompt> frames;
                std::size_t prompted = 0;
                for (Index t = 0; t < d.t; ++t) {
                    frames.push_back(framePrompt(prompts, t));
                    if (!frames.back().empty()) ++prompted;
                }
                const std::string model = p.getString("model");
                const bool allChannels = p.getString("channels", kOneChannel) == kAllChannels;
                const Index channel = allChannels ? 0 : p.getInt("input_channel", 0);
                const Index nc = allChannels ? d.c : 1;

                StepOutput out;
                out.meta = meta;
                out.array = input.materialize([&](double f, const std::string& m) { ctx.report(0.05 * f, m); });
                auto labels = std::make_shared<LabelVolume>(d.t, d.z, d.y, d.x);

                if (prompted > 0) {
                    requireWorker(ctx);
                    // A model without a prompt decoder is refused by name before
                    // any frame is sent -- the worker's model.json, which on the
                    // cluster is the only one there is.
                    const nlohmann::json about = ctx.remote->call("model_info", {{"path", model}, {"model", model}, {"spec", model}}).result;
                    if (about.contains("tasks") && about["tasks"].is_array()) {
                        const nlohmann::json& tasks = about["tasks"];
                        if (std::find(tasks.begin(), tasks.end(), "prompt") == tasks.end()) {
                            std::string offered;
                            for (const nlohmann::json& t : tasks)
                                if (t.is_string()) offered += (offered.empty() ? "" : ", ") + t.get<std::string>();
                            const std::string name = about.value("name", modelName(model)) + " " + about.value("version", std::string());
                            throw std::runtime_error(name + " cannot be prompted: it has no prompt decoder (it offers " + offered +
                                                     "). Choose Segment, or a promptable model.");
                        }
                    }
                }

                const Index volume = d.z * d.planeSize();
                std::vector<float> flat(static_cast<std::size_t>(nc) * volume);
                double seconds = 0.0;
                std::size_t done = 0;
                std::string scores;
                nlohmann::json lastResult = nlohmann::json::object();
                Diagnostics diag;
                for (Index t = 0; t < d.t; ++t) {
                    ctx.throwIfCancelled();
                    const FramePrompt& f = frames[static_cast<std::size_t>(t)];
                    if (f.empty()) {
                        labels->recomputeStats(t);
                        continue;
                    }
                    const double base = 0.1 + 0.85 * static_cast<double>(done) / static_cast<double>(prompted);
                    const double span = 0.85 / static_cast<double>(prompted);
                    for (Index c = 0; c < nc; ++c) {
                        const BufferView<const float> vol = out.array->volume(allChannels ? c : channel, t);
                        std::copy_n(vol.data(), volume, flat.data() + static_cast<std::size_t>(c) * volume);
                    }
                    // only `objects`: the i-th object's mask comes back as label i + 1
                    const nlohmann::json params = {
                        {"model", model},
                        {"task", "prompt"},
                        {"objects", promptObjectsJson(f)},
                        {"min_voxels", p.getInt("min_voxels", 0)},
                        {"voxel_um", {meta.voxelUm[0], meta.voxelUm[1], meta.voxelUm[2]}},
                        {"device", workerDevice(ctx)},
                    };
                    rpc::TensorRef in;
                    in.name = "input";
                    in.dtype = "float32";
                    in.shape = {nc, 1, d.z, d.y, d.x};
                    in.data = flat.data();
                    in.nbytes = flat.size() * sizeof(float);
                    const auto t0 = std::chrono::steady_clock::now();
                    WorkerResult r = ctx.remote->call(
                        "run", {{"kind", "foundation"}, {"params", params}}, {in},
                        [&](double fr, const std::string& m) { ctx.report(base + span * fr, m); }, [&] { return ctx.isCancelled(); });
                    ctx.throwIfCancelled();
                    seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    const rpc::Tensor* got = nullptr;
                    const rpc::Tensor* confidence = nullptr;
                    for (const rpc::Tensor& tensor : r.tensors) {
                        if (tensor.name == "labels") got = &tensor;
                        else if (tensor.name == "confidence") confidence = &tensor;
                    }
                    const auto oneFrame = [&](const rpc::Tensor* x) {
                        return x && x->shape.size() == 4 && x->shape[0] == 1 && x->shape[1] == d.z && x->shape[2] == d.y &&
                               x->shape[3] == d.x;
                    };
                    if (!got) throw std::runtime_error("the worker returned no 'labels' tensor");
                    if (!oneFrame(got)) throw std::runtime_error("the worker's labels do not match the volume");
                    std::uint32_t* dst = labels->volume(t);
                    std::copy_n(got->asUInt32(), volume, dst);
                    applyPromptIds(dst, volume, f);
                    labels->recomputeStats(t, oneFrame(confidence) ? confidence->asFloat32() : nullptr);
                    // the model's own score of each object's mask, by the object's id
                    appendPromptScores(scores, f, r.result.value("mask_scores", nlohmann::json::array()), t, d.t > 1);
                    addWorkerWarnings(diag, r.result);
                    lastResult = r.result;
                    ++done;
                }
                const std::string className = p.getString("class_name", "object");
                for (LabelStats& s : labels->stats()) s.cls = className;
                std::uint32_t total = 0;
                for (const LabelStats& s : labels->stats()) total = std::max(total, s.id);

                out.labels = labels;
                out.ranOn = ctx.backend;
                out.seconds = seconds;
                const std::vector<std::string> warnings = std::move(diag.warnings);
                diag = labelDiagnostics(*labels, summary(p, meta));
                diag.warnings = warnings;
                diag.facts.push_back({"Model", lastResult.value("model", modelName(model))});
                diag.facts.push_back({"Task", "prompt"});
                diag.facts.push_back({"Prompts", promptCounts(prompts) + " · on " + std::to_string(prompted) + " of " +
                                                     std::to_string(d.t) + " time points"});
                if (!scores.empty()) diag.facts.push_back({"Mask scores", scores});
                diag.facts.push_back({"Channels sent", std::to_string(nc)});
                diag.summary = summary(p, meta) + " · " + std::to_string(total) + " labels";
                char note[240];
                if (prompted == 0)
                    std::snprintf(note, sizeof note, "no prompts placed · nothing to segment");
                else
                    std::snprintf(note, sizeof note, "%.1f s · %s · %s", seconds, diag.summary.c_str(),
                                  ctx.remote->capabilities().device.empty() ? "worker" : ctx.remote->capabilities().device.c_str());
                out.note = note;
                out.diagnostics = std::move(diag);
                ctx.report(1.0, "");
                return out;
            }

            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeFoundationOperation() { return std::make_unique<FoundationOperation>(); }

} // namespace sirius::app
