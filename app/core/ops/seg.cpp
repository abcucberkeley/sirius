// Segmentation: a model run by the Python worker -- a TorchScript / ONNX file tile-wise,
// (the same worker serves the HPC backend), probabilities turned into
// instance labels natively. The model may also be a spec the worker resolves
// itself -- hf:<repo>[:<file>] downloaded from Hugging Face, or a model
// family (cellpose:<model>, microsam:<model_type>) whose package returns
// instance labels directly; those skip the threshold / watershed stage here.
#include "core/ops/builtin.hpp"

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

        std::string shapeText(const nlohmann::json& shape) {
            if (!shape.is_array()) return "?";
            std::string out = "(";
            for (std::size_t i = 0; i < shape.size(); ++i) {
                if (i) out += ", ";
                const nlohmann::json& e = shape[i];
                if (e.is_number()) out += std::to_string(e.get<long long>());
                else if (e.is_string()) out += e.get<std::string>();
                else out += "?";
            }
            return out + ")";
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
                info_.helpPage = "seg";
                info_.params = {
                    pathParam("model", "Model").withFilter("Models (*.pt *.pts *.pth *.onnx);;All files (*)").withHelp("A TorchScript / ONNX file taking (1, 1, Z, Y, X) float32, or a spec the worker resolves: "
                                                                                                                       "hf:<repo>[:<file>] (Hugging Face, cached in $SIRIUS_MODEL_CACHE or ~/.sirius/models), "
                                                                                                                       "cellpose:<model> (default = the installed Cellpose's built-in model, one of its model names, "
                                                                                                                       "or a custom model file) or "
                                                                                                                       "microsam:<model_type> (vit_b_lm, vit_l_lm, vit_t_lm, vit_b_em_organelles, ...). "
                                                                                                                       "Cellpose and micro-SAM return instance labels directly; threshold and post-processing "
                                                                                                                       "then do not apply"),
                    channelParam("input_channel", "Input channel", 0),
                    doubleListParam("tile", "Tile", {32.0, 256.0, 256.0}).withUnit("px").withHelp("Tile extent (z, y, x); must fit GPU memory"),
                    intParam("overlap", "Overlap", 32).range(0, 512).withUnit("px").withHelp("Tile halo; should exceed the model's receptive-field radius"),
                    doubleParam("threshold", "Threshold", 0.5).range(0.0, 1.0, 0.01, 2).withHelp("Foreground probability cut"),
                    choiceParam("post", "Post-processing", {kWatershed, kComponents, kNone}, kWatershed),
                    intParam("min_voxels", "Min. voxels", 0).range(0, 1000000000).withHelp("Drop smaller objects (0 = keep all)"),
                    doubleParam("label_opacity", "Label opacity", 0.45).range(0.0, 1.0, 0.05, 2),
                    stringParam("class_name", "Class", "nucleus").asAdvanced(),
                    doubleParam("seed_distance", "Seed distance", 5.0).range(1.0, 200.0, 0.5, 1).withUnit("px").withHelp("Minimum distance between watershed seeds").asAdvanced(),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            std::string summary(const ParamSet& p, const DatasetMeta&) const override {
                const std::string model = p.getString("model");
                std::string post = p.getString("post", kWatershed);
                post = post.rfind("Watershed", 0) == 0 ? "watershed" : post == kComponents ? "components"
                                                                                           : "probabilities";
                if (isFamilySpec(model)) post = "model labels";
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
                return v;
            }

            DatasetMeta outputMeta(const ParamSet&, const DatasetMeta& in) const override { return in; }

            std::size_t estimatedOutputBytes(const ParamSet&, const DatasetMeta& in) const override {
                return in.dims.bytes() + static_cast<std::size_t>(in.dims.t * in.dims.z * in.dims.planeSize()) * sizeof(std::uint32_t);
            }

            StepOutput run(const StepInput& input, const ParamSet& p, const StepContext& ctx) const override {
                const Validation v = validate(p, input.meta);
                if (!v.ok()) throw std::runtime_error(v.firstError());
                if (!ctx.remote)
                    throw std::runtime_error("Segmentation needs the Python worker: start it from Preferences ▸ Worker "
                                             "or choose the HPC backend");
                if (!ctx.remote->supports("torch_segment"))
                    throw std::runtime_error("The connected worker does not implement torch_segment (" +
                                             ctx.remote->capabilities().hostname + ")");
                const DatasetMeta& meta = input.meta;
                const Dims5& d = meta.dims;
                const Index channel = p.getInt("input_channel", 0);
                const std::vector<double> tile = p.getDoubleList("tile");

                StepOutput out;
                out.meta = meta;
                out.array = input.materialize([&](double f, const std::string& m) { ctx.report(0.05 * f, m); });
                auto labels = std::make_shared<LabelVolume>(d.t, d.z, d.y, d.x);

                LabelPostOptions post;
                post.post = p.getString("post", kWatershed);
                post.threshold = p.getDouble("threshold", 0.5);
                post.minVoxels = p.getInt("min_voxels", 0);
                post.seedMinDistance = p.getDouble("seed_distance", 5.0);
                post.className = p.getString("class_name", "nucleus");

                nlohmann::json params = {
                    {"model", p.getString("model")},
                    {"tile", {static_cast<Index>(tile[0]), static_cast<Index>(tile[1]), static_cast<Index>(tile[2])}},
                    {"overlap", p.getInt("overlap", 32)},
                    {"device", ctx.backend == Backend::Cpu ? "cpu" : "auto"},
                };
                if (!ctx.hubToken.empty()) params["token"] = ctx.hubToken;   // a gated hf: model
                double seconds = 0.0;
                std::uint32_t total = 0;
                std::string classes;
                bool fromModel = false;
                for (Index t = 0; t < d.t; ++t) {
                    ctx.throwIfCancelled();
                    const double base = 0.05 + 0.9 * static_cast<double>(t) / d.t, span = 0.9 / d.t;
                    const BufferView<const float> vol = out.array->volume(channel, t);
                    rpc::TensorRef in;
                    in.name = "input";
                    in.dtype = "float32";
                    in.shape = {d.z, d.y, d.x};
                    in.data = vol.data();
                    in.nbytes = vol.bytes();
                    const auto t0 = std::chrono::steady_clock::now();
                    WorkerResult r = ctx.remote->call(
                        "run", {{"kind", "torch_segment"}, {"params", params}}, {in},
                        [&](double f, const std::string& m) { ctx.report(base + span * 0.8 * f, m); },
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
            OpInfo info_;
        };

    } // namespace

    nlohmann::json torchModelInfo(RemoteWorker& worker, const std::string& modelPath) {
        // "spec" carries hub / family specs; "path" / "model" keep older workers working
        WorkerResult r = worker.call("model_info", {{"path", modelPath}, {"model", modelPath}, {"spec", modelPath}});
        return r.result;
    }

    std::string torchModelSummary(const nlohmann::json& info) {
        if (!info.is_object()) return "no model";
        std::string out = info.value("format", "TorchScript");
        if (info.contains("available") && info["available"].is_boolean()) {
            // a model family (cellpose, micro-sam) or an hf: file not downloaded yet
            if (info.value("model", std::string()).size()) out += " " + info.value("model", std::string());
            if (!info["available"].get<bool>()) {
                const std::string hint = info.value("install_hint", std::string());
                return out + " · not installed (Hub… installs it" + (hint.empty() ? ")" : ": " + hint + ")");
            }
            if (info.contains("cached") && info["cached"].is_boolean() && !info["cached"].get<bool>())
                return out + " " + info.value("repo", std::string()) + " · downloads on first run";
            if (!info.contains("input_shape")) {
                if (info.contains("version") && info["version"].is_string() && !info["version"].get<std::string>().empty())
                    out = info.value("format", std::string()) + " " + info["version"].get<std::string>() + " " + info.value("model", std::string());
                out += " · returns labels";
                if (info.contains("weights_cached") && info["weights_cached"].is_boolean())
                    out += info["weights_cached"].get<bool>() ? " · weights cached" : " · weights download on first run";
                if (info.contains("warning") && info["warning"].is_string()) out += " · " + info["warning"].get<std::string>();
                return out;
            }
        }
        if (info.contains("input_shape")) {
            out += " · in " + shapeText(info["input_shape"]);
            if (info.contains("input_dtype")) out += " " + info["input_dtype"].get<std::string>();
        }
        if (info.contains("output_shape")) out += " · out " + shapeText(info["output_shape"]);
        if (info.contains("size_bytes") && info["size_bytes"].is_number())
            out += " · " + formatBytes(info["size_bytes"].get<std::uint64_t>());
        return out;
    }

    std::unique_ptr<Operation> makeTorchSegmentationOperation() { return std::make_unique<TorchSegmentationOperation>(); }

} // namespace sirius::app
