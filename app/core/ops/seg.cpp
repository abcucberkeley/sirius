// Segmentation: a model run by the Python worker -- a TorchScript / ONNX file tile-wise,
// (the same worker serves the HPC backend), probabilities turned into
// instance labels natively. The model may also be a spec the worker resolves
// itself -- hf:<repo>[:<file>] downloaded from Hugging Face, or a model
// family (cellpose:<model>, microsam:<model_type>) whose package returns
// instance labels directly; those skip the threshold / watershed stage here.
//
// A micro-SAM model can also be prompted (Task: Prompt objects): the person
// points at objects (a box, clicks, a scribble) and gets those objects back,
// one mask per object, each object's prompts in one predictor call so that a
// background click corrects its mask. This sends the same joint `objects`
// form as the foundation step. micro-SAM is a 2-D model: all of an object's
// prompts must share a plane (refused here before the worker would), and
// its mask lies in that plane; the diagnostics say so rather than leave a
// one-plane object to be taken for a cell.
#include "core/cancel.hpp"
#include "core/errors.hpp"
#include "core/ops/common.hpp"
#include "core/ops/segment_common.hpp"
#include "core/ops/torch_model.hpp"
#include "core/ops/builtin.hpp"
#include "core/remote_source.hpp"
#include "core/rpc.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <vector>

namespace sirius::app {

    namespace {

        constexpr const char* kWatershed = "Watershed on boundary channel";
        constexpr const char* kComponents = "Connected components";
        constexpr const char* kNone = "None (raw probabilities)";
        constexpr const char* kSegmentAll = "Segment all objects";
        constexpr const char* kPrompt = kPromptTask;

        std::string lowered(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        }

        // Specs the worker resolves itself (app/python/sirius_worker/models.py);
        // everything else is a file path on the worker's host.
        bool isHubSpec(const std::string& model) {
            const std::string low = lowered(model);
            return low.rfind("hf:", 0) == 0 || low.rfind("huggingface:", 0) == 0;
        }

        bool isFamilySpec(const std::string& model) {
            const std::string low = lowered(model);
            return low.rfind("cellpose:", 0) == 0 || low.rfind("microsam:", 0) == 0 || low.rfind("micro-sam:", 0) == 0 ||
                   low.rfind("micro_sam:", 0) == 0;
        }

        bool isModelSpec(const std::string& model) { return isHubSpec(model) || isFamilySpec(model); }

        // The family whose models answer a point prompt (the worker's
        // models.family_promptable); cellpose segments whole images only.
        bool isMicroSamSpec(const std::string& model) {
            const std::string low = lowered(model);
            return low.rfind("microsam:", 0) == 0 || low.rfind("micro-sam:", 0) == 0 || low.rfind("micro_sam:", 0) == 0;
        }

        // "cellpose cyto3", "micro-SAM vit_b_lm", "hf model.pt", or the file name.
        std::string modelLabel(const std::string& model) {
            if (model.empty()) return "no model";
            const std::size_t colon = model.find(':');
            if (isFamilySpec(model)) {
                const std::string rest = model.substr(colon + 1);
                return (lowered(model).rfind("cellpose:", 0) == 0 ? "cellpose " : "micro-SAM ") + rest;
            }
            if (isHubSpec(model)) {
                const std::string rest = model.substr(colon + 1);
                const std::size_t sep = rest.rfind(':');
                return "hf " + (sep == std::string::npos ? rest : std::filesystem::path(rest.substr(sep + 1)).filename().string());
            }
            return std::filesystem::path(model).filename().string();
        }

        // Instance labels the model produced itself: copied in, small objects
        // dropped, statistics from the confidence map when the worker sent one.
        std::uint32_t labelsFromModel(const std::uint32_t* in, const float* confidence, Index z, Index y, Index x,
                                      const LabelPostOptions& options, LabelVolume& labels, Index t) {
            const Index n = z * y * x;
            std::uint32_t* out = labels.volume(t);
            std::copy_n(in, n, out);
            const std::uint32_t count = removeSmall(out, n, options.minVoxels);
            labels.recomputeStats(t, confidence);
            for (LabelStats& s : labels.stats()) s.cls = options.className;
            labels.applyFlags(options.flags);
            return count;
        }

        class TorchSegmentationOperation final : public Operation {
        public:
            TorchSegmentationOperation() {
                info_.kind = "seg";
                info_.name = "Segmentation";
                info_.group = "Segment";
                info_.kindLabel = "SEGMENT";
                info_.diagnostics = DiagnosticsKind::Segment;
                info_.defaultCache = CachePolicy::Disk;
                info_.separableOverT = true;
                info_.hasGpuPath = true;
                info_.remoteCapable = true;
                info_.producesLabels = true;
                info_.promptPlanar = true;   // micro-SAM is 2-D: an object's prompts share a plane
                info_.helpPage = "seg";
                info_.params = {
                    pathParam("model", "Model").withFilter("Models (*.pt *.pts *.pth *.onnx);;All files (*)").withHelp("A TorchScript / ONNX file taking (1, 1, Z, Y, X) float32, or a spec the worker resolves: "
                                                                                                                       "hf:<repo>[:<file>] (Hugging Face, cached in $SIRIUS_MODEL_CACHE or ~/.sirius/models), "
                                                                                                                       "cellpose:<model> (default = the installed Cellpose's built-in model, one of its model names, "
                                                                                                                       "or a custom model file) or "
                                                                                                                       "microsam:<model_type> (vit_b_lm, vit_l_lm, vit_t_lm, vit_b_em_organelles, ...). "
                                                                                                                       "Cellpose and micro-SAM return instance labels directly; threshold and post-processing "
                                                                                                                       "then do not apply"),
                    choiceParam("task", "Task", {kSegmentAll, kPrompt}, kSegmentAll)
                        .withHelp("Segment every object, or only the ones you point at with the viewer's Prompt tool. "
                                  "Prompting needs a micro-SAM model (microsam:<type>), whose masks are 2-D: one per object, "
                                  "in the plane of its prompts"),
                    promptsParam(kPromptsKey, "Prompts")
                        .visibleWhen("task", {kPrompt})
                        .withHelp("Where the objects are, in voxels of the input: points (label 1 object, 0 background), "
                                  "boxes and scribbles, each belonging to one object whose prompts all lie on one plane. A "
                                  "background point on an object corrects its mask. Placed with the viewer's Prompt tool"),
                    channelParam("input_channel", "Input channel", 0),
                    doubleListParam("tile", "Tile", {32.0, 256.0, 256.0}).withUnit("px").withHelp("Tile extent (z, y, x); must fit GPU memory").hiddenWhen("task", {kPrompt}),
                    intParam("overlap", "Overlap", 32).range(0, 512).withUnit("px").withHelp("Tile halo; should exceed the model's receptive-field radius").hiddenWhen("task", {kPrompt}),
                    doubleParam("threshold", "Threshold", 0.5).range(0.0, 1.0, 0.01, 2).withHelp("Foreground probability cut").hiddenWhen("task", {kPrompt}),
                    choiceParam("post", "Post-processing", {kWatershed, kComponents, kNone}, kWatershed).hiddenWhen("task", {kPrompt}),
                    intParam("min_voxels", "Min. voxels", 0).range(0, 1000000000).withHelp("Drop smaller objects (0 = keep all)"),
                    doubleParam("label_opacity", "Label opacity", 0.45).range(0.0, 1.0, 0.05, 2),
                    stringParam("class_name", "Class", "nucleus").asAdvanced(),
                    doubleParam("seed_distance", "Seed distance", 5.0).range(1.0, 200.0, 0.5, 1).withUnit("px").withHelp("Minimum distance between watershed seeds").asAdvanced().hiddenWhen("task", {kPrompt}),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            std::string summary(const ParamSet& p, const DatasetMeta&) const override {
                const std::string model = p.getString("model");
                std::string post = p.getString("post", kWatershed);
                post = post.rfind("Watershed", 0) == 0 ? "watershed" : post == kComponents ? "components"
                                                                                           : "probabilities";
                if (isFamilySpec(model)) post = "model labels";
                if (isPromptStep(p)) return joinSummary({modelLabel(model), "prompt", toDisplayString(promptsValue(promptsOf(p)))});
                return joinSummary({modelLabel(model), post});
            }

            Validation validate(const ParamSet& p, const DatasetMeta& in) const override {
                Validation v = Operation::validate(p, in);
                const std::string model = p.getString("model");
                // hub / family specs live on the worker's host (or are downloaded there): no file to check here
                if (model.empty()) v.errors.push_back("Choose a model: a TorchScript / ONNX file, hf:<repo>, cellpose:<model> or microsam:<type>.");
                else if (!isModelSpec(model) && !std::filesystem::exists(model)) v.errors.push_back("Model not found: " + model);
                if (in.rgb) v.errors.push_back("Segmentation needs an intensity channel, not an RGB merge.");
                const std::vector<double> tile = p.getDoubleList("tile");
                if (tile.size() != 3 || std::any_of(tile.begin(), tile.end(), [](double d) { return d < 1; }))
                    v.errors.push_back("Tile must be three positive extents (z, y, x).");
                if (isPromptStep(p)) {
                    // the worker can tell (family_info's promptable) and is asked
                    // at run time too, but a spec that cannot be prompted is
                    // known here already, and the panel should say so at once
                    if (!model.empty() && lowered(model).rfind("cellpose:", 0) == 0)
                        v.errors.push_back("Cellpose cannot be prompted: it segments a whole image and has no prompt interface. "
                                           "Use Task: Segment all objects, or a micro-SAM model (microsam:vit_b_lm, ...).");
                    else if (!model.empty() && !isMicroSamSpec(model))
                        v.errors.push_back("Only micro-SAM models (microsam:<type>) can be prompted in this step; a model folder with a "
                                           "prompt decoder is prompted in the Foundation model step.");
                    const std::vector<Prompt> prompts = promptsOf(p);
                    validatePrompts(prompts, in, v);
                    // what the worker refuses (models.run_microsam_prompt), said
                    // while the prompts are placed rather than when they run
                    for (Index t = 0; t < in.dims.t && v.ok(); ++t)
                        for (const FramePrompt::Object& o : framePrompt(prompts, t).objects) {
                            const std::vector<Index> planes = promptPlanes(o);
                            if (planes.size() < 2) continue;
                            std::string list;
                            for (const Index z : planes) list += (list.empty() ? "" : ", ") + std::to_string(z);
                            v.errors.push_back("Object " + std::to_string(o.id) + (in.dims.t > 1 ? " on time point " + std::to_string(t) : std::string()) +
                                               " has prompts on planes z " + list + ": micro-SAM is a 2-D model and cannot correct across z. "
                                                                                    "Keep an object's corrections on the plane it was started in (a new plane is a new object), "
                                                                                    "or prompt a model folder in the Foundation model step, whose decoder is 3-D.");
                            break;
                        }
                }
                return v;
            }

            DatasetMeta outputMeta(const ParamSet&, const DatasetMeta& in) const override { return in; }

            std::size_t estimatedOutputBytes(const ParamSet&, const DatasetMeta& in) const override {
                return in.dims.bytes() + static_cast<std::size_t>(in.dims.t * in.dims.z * in.dims.planeSize()) * sizeof(std::uint32_t);
            }

            StepOutput run(const StepInput& input, const ParamSet& p, const StepContext& ctx) const override {
                const Validation v = validate(p, input.meta);
                if (!v.ok()) throw std::runtime_error(v.firstError());
                if (isPromptStep(p)) return runPrompt(input, p, ctx);
                requireWorker(ctx);
                const DatasetMeta& meta = input.meta;
                const Dims5& d = meta.dims;
                const Index channel = p.getInt("input_channel", 0);
                const std::vector<double> tile = p.getDoubleList("tile");

                StepOutput out;
                out.meta = meta;
                // A cluster dataset on the HPC backend: the worker reads each
                // volume on the node (input_ref), and the image stays there,
                // shown through the worker (core/remote_source.hpp).
                const auto* remoteInput =
                    !input.array && ctx.backend == Backend::Hpc ? dynamic_cast<const RemoteSource*>(input.source.get()) : nullptr;
                if (remoteInput) out.source = input.source;
                else out.array = input.materialize([&](double f, const std::string& m) { ctx.report(0.05 * f, m); });
                auto labels = std::make_shared<LabelVolume>(d.t, d.z, d.y, d.x);

                LabelPostOptions post;
                post.post = p.getString("post", kWatershed);
                post.threshold = p.getDouble("threshold", 0.5);
                post.minVoxels = p.getInt("min_voxels", 0);
                post.seedMinDistance = p.getDouble("seed_distance", 5.0);
                post.className = p.getString("class_name", "nucleus");
                post.poll = [&ctx] { ctx.throwIfCancelled(); };

                nlohmann::json params = {
                    {"model", p.getString("model")},
                    {"tile", {static_cast<Index>(tile[0]), static_cast<Index>(tile[1]), static_cast<Index>(tile[2])}},
                    {"overlap", p.getInt("overlap", 32)},
                    {"device", workerDevice(ctx)},
                };
                if (!ctx.hubToken.empty()) params["token"] = ctx.hubToken;   // a gated hf: model
                double seconds = 0.0;
                std::uint32_t total = 0;
                std::string classes;
                bool fromModel = false;
                for (Index t = 0; t < d.t; ++t) {
                    ctx.throwIfCancelled();
                    const double base = 0.05 + 0.9 * static_cast<double>(t) / d.t, span = 0.9 / d.t;
                    nlohmann::json request = {{"kind", "torch_segment"}, {"params", params}};
                    std::vector<rpc::TensorRef> ins;
                    if (remoteInput) {
                        request["input_ref"] = remoteInput->inputReference(channel, t);
                    } else {
                        const BufferView<const float> vol = out.array->volume(channel, t);
                        rpc::TensorRef in;
                        in.name = "input";
                        in.dtype = "float32";
                        in.shape = {d.z, d.y, d.x};
                        in.data = vol.data();
                        in.nbytes = vol.bytes();
                        ins.push_back(in);
                    }
                    const auto t0 = std::chrono::steady_clock::now();
                    WorkerResult r = ctx.remote->call(
                        "run", request, ins, [&](double f, const std::string& m) { ctx.report(base + span * 0.8 * f, m); },
                        [&] { return ctx.isCancelled(); });
                    seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    ctx.throwIfCancelled();
                    const rpc::Tensor* prob = nullptr;
                    const rpc::Tensor* modelLabels = nullptr;
                    for (const rpc::Tensor& tensor : r.tensors) {
                        if (tensor.name == "prob") prob = &tensor;
                        else if (tensor.name == "labels") modelLabels = &tensor;
                    }
                    if (!prob && !modelLabels) throw std::runtime_error("the worker returned neither a 'prob' nor a 'labels' tensor");
                    const bool probMatches = prob && prob->shape.size() == 4 && prob->shape[1] == d.z && prob->shape[2] == d.y &&
                                             prob->shape[3] == d.x;
                    if (r.result.contains("class_names") && r.result["class_names"].is_array() && !r.result["class_names"].empty())
                        post.className = r.result["class_names"][0].get<std::string>();
                    if (modelLabels) {
                        // instance labels straight from the model (Cellpose, micro-SAM); a
                        // probability map, when sent, only feeds the per-label confidence
                        if (modelLabels->shape.size() != 3 || modelLabels->shape[0] != d.z || modelLabels->shape[1] != d.y ||
                            modelLabels->shape[2] != d.x)
                            throw std::runtime_error("the worker's labels do not match the volume");
                        ctx.report(base + span * 0.85, "labels");
                        total += labelsFromModel(modelLabels->asUInt32(), probMatches ? prob->asFloat32() : nullptr, d.z, d.y, d.x,
                                                 post, *labels, t);
                        fromModel = true;
                    } else {
                        if (!probMatches) throw std::runtime_error("the worker's probabilities do not match the volume");
                        const float* fg = prob->asFloat32();
                        const Index volSize = d.z * d.planeSize();
                        const float* boundary = prob->shape[0] > 1 ? fg + volSize : nullptr;
                        ctx.report(base + span * 0.85, "labelling");
                        total += labelsFromProbabilities(fg, boundary, d.z, d.y, d.x, post, *labels, t);
                    }
                    if (classes.empty() && r.result.contains("model")) classes = r.result["model"].dump();
                }
                out.labels = labels;
                out.ranOn = ctx.backend;
                out.seconds = seconds;
                char note[200];
                std::snprintf(note, sizeof note, "%.1f s · %u labels%s · %s", seconds, total, fromModel ? " · labels from the model" : "",
                              ctx.remote->capabilities().device.empty() ? "worker" : ctx.remote->capabilities().device.c_str());
                out.note = note;
                out.diagnostics = labelDiagnostics(*labels, summary(p, meta));
                out.diagnostics.summary = summary(p, meta) + " · " + std::to_string(total) + " labels";
                ctx.report(1.0, "");
                return out;
            }

        private:
            static void requireWorker(const StepContext& ctx) {
                if (!ctx.remote)
                    throw std::runtime_error("Segmentation needs the Python worker, which is not available here (see the "
                                             "worker message in the log), or the HPC backend");
                if (!ctx.remote->supports("torch_segment"))
                    throw std::runtime_error("The connected worker does not implement torch_segment (" +
                                             ctx.remote->capabilities().hostname + ")");
            }

            // Prompt: the objects a person pointed at, through micro-SAM's
            // predictor. One call per time point that has prompts; a frame
            // nobody pointed at is left empty without asking the worker.
            StepOutput runPrompt(const StepInput& input, const ParamSet& p, const StepContext& ctx) const {
                const DatasetMeta& meta = input.meta;
                const Dims5& d = meta.dims;
                const std::string model = p.getString("model");
                const Index channel = p.getInt("input_channel", 0);
                const std::vector<Prompt> prompts = promptsOf(p);
                std::vector<FramePrompt> frames;
                std::size_t prompted = 0;
                for (Index t = 0; t < d.t; ++t) {
                    frames.push_back(framePrompt(prompts, t));
                    if (!frames.back().empty()) ++prompted;
                }

                StepOutput out;
                out.meta = meta;
                out.array = input.materialize([&](double f, const std::string& m) { ctx.report(0.05 * f, m); });
                auto labels = std::make_shared<LabelVolume>(d.t, d.z, d.y, d.x);
                if (prompted > 0) {
                    requireWorker(ctx);
                    // The worker says which families answer a prompt
                    // (family_info's promptable); one that says no is refused
                    // with its reason before any frame is sent.
                    const nlohmann::json info = torchModelInfo(*ctx.remote, model);
                    if (info.contains("promptable") && info["promptable"].is_boolean() && !info["promptable"].get<bool>())
                        throw std::runtime_error("The worker reports that " + modelLabel(model) +
                                                 " cannot be prompted. Use Task: Segment all objects, or a micro-SAM model.");
                }

                nlohmann::json params = {{"model", model}, {"task", "prompt"}, {"device", workerDevice(ctx)}};
                if (!ctx.hubToken.empty()) params["token"] = ctx.hubToken;
                const Index volume = d.z * d.planeSize();
                const Index minVoxels = p.getInt("min_voxels", 0);
                double seconds = 0.0;
                std::size_t done = 0;
                bool planeOnly = false;
                std::string scores;
                for (Index t = 0; t < d.t; ++t) {
                    ctx.throwIfCancelled();
                    const FramePrompt& f = frames[static_cast<std::size_t>(t)];
                    if (f.empty()) {
                        labels->recomputeStats(t);
                        continue;
                    }
                    const double base = 0.05 + 0.9 * static_cast<double>(done) / static_cast<double>(prompted);
                    const double span = 0.9 / static_cast<double>(prompted);
                    // only `objects`: the i-th object's mask comes back as label i + 1
                    params["objects"] = promptObjectsJson(f);
                    const BufferView<const float> vol = out.array->volume(channel, t);
                    rpc::TensorRef in;
                    in.name = "input";
                    in.dtype = "float32";
                    in.shape = {d.z, d.y, d.x};
                    in.data = vol.data();
                    in.nbytes = vol.bytes();
                    const auto t0 = std::chrono::steady_clock::now();
                    WorkerResult r;
                    try {
                        r = ctx.remote->call(
                            "run", {{"kind", "torch_segment"}, {"params", params}}, {in},
                            [&](double fr, const std::string& m) { ctx.report(base + span * fr, m); }, [&] { return ctx.isCancelled(); });
                    } catch (const ProtocolError&) {
                        throw;   // the connection, not the prompts
                    } catch (const std::exception& e) {
                        if (isCancellation(e)) throw;
                        // the worker refusing the prompts (an object across
                        // planes, a point outside the volume): say whose
                        throw std::runtime_error(modelLabel(model) + " refused the prompts" +
                                                 (d.t > 1 ? " of time point " + std::to_string(t) : std::string()) + ": " + e.what());
                    }
                    seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    ctx.throwIfCancelled();
                    const rpc::Tensor* got = nullptr;
                    for (const rpc::Tensor& tensor : r.tensors)
                        if (tensor.name == "labels") got = &tensor;
                    if (!got) throw std::runtime_error("the worker returned no 'labels' tensor for the prompt");
                    if (got->shape.size() != 3 || got->shape[0] != d.z || got->shape[1] != d.y || got->shape[2] != d.x)
                        throw std::runtime_error("the worker's labels do not match the volume");
                    std::uint32_t* dst = labels->volume(t);
                    std::copy_n(got->asUInt32(), volume, dst);
                    applyPromptIds(dst, volume, f);
                    // the ids stay the objects', so the scores below name them
                    if (minVoxels > 0) dropSmall(dst, volume, minVoxels);
                    labels->recomputeStats(t);
                    planeOnly = planeOnly || r.result.value("plane_only", false);
                    appendPromptScores(scores, f, r.result.value("mask_scores", nlohmann::json::array()), t, d.t > 1);
                    ++done;
                }
                const std::string className = p.getString("class_name", "nucleus");
                for (LabelStats& s : labels->stats()) s.cls = className;
                std::uint32_t total = 0;
                for (const LabelStats& s : labels->stats()) total = std::max(total, s.id);

                out.labels = labels;
                out.ranOn = ctx.backend;
                out.seconds = seconds;
                Diagnostics diag = labelDiagnostics(*labels, summary(p, meta));
                diag.facts.push_back({"Task", "prompt"});
                diag.facts.push_back({"Prompts", promptCounts(prompts) + " · on " + std::to_string(prompted) + " of " +
                                                     std::to_string(d.t) + " time points"});
                if (!scores.empty()) diag.facts.push_back({"Mask scores", scores});
                if (planeOnly) {
                    // micro-SAM is a 2-D model: the object its prompts name is
                    // the one in their plane, and a cell needs an object per plane
                    diag.facts.push_back({"Masks", "per plane: each covers its object's z plane only"});
                    diag.warnings.push_back(modelLabel(model) +
                                            " is a 2-D model: each mask lies in the plane of its object's prompts. A cell in 3-D needs "
                                            "an object on every plane, or the Foundation model step with a promptable model folder, whose decoder is 3-D.");
                }
                diag.summary = summary(p, meta) + " · " + std::to_string(total) + " labels";
                char note[240];
                if (prompted == 0)
                    std::snprintf(note, sizeof note, "no prompts placed · nothing to segment");
                else
                    std::snprintf(note, sizeof note, "%.1f s · %u labels%s · %s", seconds, total, planeOnly ? " · per plane" : "",
                                  ctx.remote->capabilities().device.empty() ? "worker" : ctx.remote->capabilities().device.c_str());
                out.note = note;
                out.diagnostics = std::move(diag);
                ctx.report(1.0, "");
                return out;
            }

            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeTorchSegmentationOperation() { return std::make_unique<TorchSegmentationOperation>(); }

} // namespace sirius::app
