// The latents foundation model, run by the Python worker.
//
// Every other model operation here sends one 3-D volume of one channel per
// call. This one sends (c, t, z, y, x) in a single call, because the colour and
// time axes are what the model is for: it is the same weights whether the input
// is a plane, a volume, a multi-channel stack or a clip, and taking the time
// axis away would leave it doing the same job as `seg`.
//
// The model file is a bundle (.ltb), not a bare TorchScript graph. Weights
// alone do not reproduce a result: the peak threshold, the minimum separation
// between two objects and the voxel size the distances were calibrated at were
// chosen on held-out data at training time. The bundle carries them, and a
// threshold or separation left at zero here means "use the bundle's", so the
// defaults are the values the model was actually validated with rather than
// whatever this dialog happens to open with.
//
// Tracking is returned the way this application represents a track: one label
// id naming the same object at every time point, with `tracked` set. The
// lineage the model produces rides beside the labels (LabelVolume::lineage)
// for the track review; the diagnostics still give the model's own count.
//
// A bundle whose head has a prompt decoder also offers Prompt: the person
// points at objects -- a box around one, a click on it, a scribble over it,
// with the viewer's Prompt tool or as an agent sets them -- and gets those
// objects back, in 3-D, one mask per object. Each object goes to the worker
// with all of its prompts together (the joint `objects` form), so a
// background click on an object corrects that object's mask, and its mask
// comes back labelled with the object's id on every re-run. That task sends
// one time point per call, as the decoder takes it, and only the frames
// someone prompted.
#include "core/ops/common.hpp"
#include "core/ops/builtin.hpp"
#include "core/rpc.hpp"

#include <nlohmann/json.hpp>
#include "core/tracks.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <system_error>
#include <vector>

namespace sirius::app {

    namespace {

        constexpr const char* kDetect = "Detect centroids";
        constexpr const char* kSegment = "Segment objects";
        constexpr const char* kTrack = "Track over time";
        constexpr const char* kPrompt = kPromptTask;
        constexpr const char* kOneChannel = "Selected channel";
        constexpr const char* kAllChannels = "All channels";

        const char* taskKey(const std::string& label) {
            if (label == kTrack) return "track";
            if (label == kDetect) return "detect";
            if (label == kPrompt) return "prompt";
            return "segment";
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
                info_.separableOverT = false;   // the time axis is an input, not a loop
                info_.hasGpuPath = true;
                info_.remoteCapable = true;
                info_.producesLabels = true;
                info_.helpPage = "foundation";
                info_.params = {
                    pathParam("model", "Model")
                        .withFilter("Model bundles (*.ltb);;All files (*)")
                        .withHelp("A .ltb bundle: the encoder, the task head and the thresholds it was validated "
                                  "with. One bundle serves planes, volumes, multi-channel stacks and clips"),
                    choiceParam("task", "Task", {kSegment, kDetect, kTrack, kPrompt}, kSegment)
                        .withHelp("Segment gives objects with extents; Detect gives one voxel per object, which is "
                                  "faster and is what the model predicts directly; Track follows objects across "
                                  "time and needs more than one time point; Prompt segments the objects you point "
                                  "at with the viewer's Prompt tool (a box, a click, a scribble), and needs a bundle "
                                  "with a prompt decoder"),
                    promptsParam(kPromptsKey, "Prompts")
                        .visibleWhen("task", {kPrompt})
                        .withHelp("Where the objects are, in voxels of the input: points, boxes and scribbles, each on "
                                  "one time point and belonging to one object. An object's prompts are sent together, so "
                                  "a background point on it corrects its mask, labelled with the object's id. Placed with "
                                  "the viewer's Prompt tool"),
                    choiceParam("channels", "Channels", {kOneChannel, kAllChannels}, kOneChannel)
                        .withHelp("The model accepts several channels at once. Send all of them only when the "
                                  "bundle was trained with channel identities, otherwise pick one"),
                    channelParam("input_channel", "Input channel", 0).visibleWhen("channels", {kOneChannel}),
                    doubleParam("threshold", "Threshold", 0.0)
                        .range(0.0, 1.0, 0.01, 2)
                        .hiddenWhen("task", {kPrompt})
                        .withHelp("Peak probability cut. 0 uses the value the bundle was validated at"),
                    doubleParam("min_separation", "Min. separation", 0.0)
                        .range(0.0, 100.0, 0.1, 2)
                        .hiddenWhen("task", {kPrompt})
                        .withUnit("um")
                        .withHelp("Two peaks closer than this are one object. In microns, so it means the same "
                                  "thing on anisotropic data. 0 uses the bundle's value"),
                    // Segment and Prompt only: Detect marks one voxel per
                    // object, and in a tracking run the label id is a track
                    // id (see run()).
                    intParam("min_voxels", "Min. voxels", 0)
                        .range(0, 1000000000)
                        .visibleWhen("task", {kSegment, kPrompt})
                        .withHelp("Drop smaller objects (0 = keep all)"),
                    doubleListParam("tile", "Tile", {0.0, 0.0, 0.0})
                        .withUnit("px")
                        .withHelp("Inference tile (z, y, x); a zero extent uses the bundle's own crop size on that axis")
                        .asAdvanced(),
                    doubleParam("label_opacity", "Label opacity", 0.45).range(0.0, 1.0, 0.05, 2),
                    stringParam("class_name", "Class", "object").asAdvanced(),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            std::string summary(const ParamSet& p, const DatasetMeta&) const override {
                const std::string model = p.getString("model");
                const std::string name = model.empty() ? "no model" : std::filesystem::path(model).stem().string();
                std::string chans = p.getString("channels", kOneChannel) == kAllChannels ? "all channels" : "";
                const std::string task = taskKey(p.getString("task", kSegment));
                if (task == "prompt") return joinSummary({name, task, toDisplayString(promptsValue(promptsOf(p))), chans});
                return joinSummary({name, task, chans});
            }

            Validation validate(const ParamSet& p, const DatasetMeta& in) const override {
                Validation v = Operation::validate(p, in);
                const std::string model = p.getString("model");
                if (model.empty())
                    v.errors.push_back("Choose a model bundle (.ltb).");
                else if (std::error_code ec; !std::filesystem::exists(model, ec))
                    // A warning, not an error: the bundle is opened by the worker,
                    // and on the HPC backend (a bundle picked from a cluster
                    // registry) it lives on a filesystem this machine cannot see.
                    // A worker that cannot find it either says so when it runs.
                    v.warnings.push_back("Model bundle not found on this machine: " + model +
                                         " (fine when the worker runs where the bundle is)");
                if (in.rgb)
                    v.errors.push_back("The model needs intensity channels, not an RGB merge.");
                if (std::string(taskKey(p.getString("task", kSegment))) == "track" && in.dims.t < 2)
                    v.errors.push_back("Tracking needs more than one time point; this dataset has " +
                                       std::to_string(in.dims.t) + ".");
                if (isPromptStep(p)) validatePrompts(promptsOf(p), in, v);
                const std::vector<double> tile = p.getDoubleList("tile");
                if (tile.size() != 3)
                    v.errors.push_back("Tile must be three extents (z, y, x); zero uses the bundle's own.");
                else if (std::any_of(tile.begin(), tile.end(), [](double d) { return d < 0; }))
                    v.errors.push_back("Tile extents cannot be negative.");
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
                // planes per (c, t), so this is a copy; it is the price of
                // letting the model see the axes together, and it is the same
                // size as the array already in hand.
                const Index volume = d.z * d.planeSize();
                std::vector<float> flat(static_cast<std::size_t>(nc) * d.t * volume);
                for (Index c = 0; c < nc; ++c)
                    for (Index t = 0; t < d.t; ++t) {
                        ctx.throwIfCancelled();
                        const BufferView<const float> vol = out.array->volume(allChannels ? c : channel, t);
                        std::copy_n(vol.data(), volume, flat.data() + (static_cast<std::size_t>(c) * d.t + t) * volume);
                    }

                nlohmann::json params = {
                    {"model", p.getString("model")},
                    {"task", taskKey(p.getString("task", kSegment))},
                    {"threshold", p.getDouble("threshold", 0.0)},
                    {"min_separation", p.getDouble("min_separation", 0.0)},
                    {"min_voxels", p.getInt("min_voxels", 0)},
                    {"voxel_um", {meta.voxelUm[0], meta.voxelUm[1], meta.voxelUm[2]}},
                    {"device", workerDevice(ctx)},
                };
                const std::vector<double> tile = p.getDoubleList("tile");
                if (tile.size() == 3 && (tile[0] > 0 || tile[1] > 0 || tile[2] > 0))
                    params["tile"] = {static_cast<Index>(tile[0]), static_cast<Index>(tile[1]), static_cast<Index>(tile[2])};

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
                const std::string task = taskKey(p.getString("task", kSegment));
                std::uint32_t total = 0;
                for (Index t = 0; t < d.t; ++t) {
                    ctx.throwIfCancelled();
                    std::uint32_t* dst = labels->volume(t);
                    // The labels are used as they come. Min. voxels is the
                    // worker's to apply, and it applies it to Segment only: a
                    // detection is one voxel, so a size filter here removed
                    // every object, and in a tracking run the label id is a
                    // track id that dropping an object would punch a hole in.
                    std::copy_n(src + static_cast<std::size_t>(t) * volume, volume, dst);
                    labels->recomputeStats(t, confMatches ? confidence->asFloat32() + static_cast<std::size_t>(t) * volume
                                                          : nullptr);
                    for (const LabelStats& s : labels->stats()) total = std::max(total, s.id);
                }
                const std::string className = task == "track" ? "track" : p.getString("class_name", "object");
                for (LabelStats& s : labels->stats()) s.cls = className;
                if (task == "track") {
                    // the lineage the model reports, kept for the track review
                    // instead of being reduced to a count
                    labels->setTracked(true);
                    labels->setLineage(lineageFromJson(r.result.value("lineage", nlohmann::json::object())));
                    labels->indexTracks();
                }

                out.labels = labels;
                out.ranOn = ctx.backend;
                out.seconds = seconds;

                Diagnostics diag = labelDiagnostics(*labels, summary(p, meta));
                diag.facts.push_back({"Model", r.result.value("model", std::string("?"))});
                diag.facts.push_back({"Task", task});
                diag.facts.push_back({"Threshold", formatNumber(r.result.value("threshold", 0.0), 2)});
                diag.facts.push_back({"Min. separation", formatNumber(r.result.value("min_separation_um", 0.0), 2) + " um"});
                diag.facts.push_back({"Channels sent", std::to_string(nc)});
                if (task == "track") {
                    const long long tracks = r.result.value("tracks", 0LL);
                    diag.facts.push_back({"Tracks", std::to_string(tracks)});
                    // latents recovers divisions after the fact with a geometric
                    // rule that under-calls on real detections (help page)
                    diag.facts.push_back({"Divisions (approx.)", std::to_string(r.result.value("divisions", 0LL))});
                    diag.summary = summary(p, meta) + " · " + std::to_string(tracks) + " tracks";
                } else {
                    diag.facts.push_back({"Objects", std::to_string(r.result.value("objects", 0LL))});
                    diag.summary = summary(p, meta) + " · " + std::to_string(total) + " labels";
                }

                // before diag is moved from: the note used to read "0.8 s ·  · cpu"
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
            static void requireWorker(const StepContext& ctx) {
                if (!ctx.remote)
                    throw std::runtime_error("The foundation model needs the Python worker, which is not available here "
                                             "(see the worker message in the log), or the HPC backend");
                if (!ctx.remote->supports("foundation"))
                    throw std::runtime_error("The connected worker does not implement the foundation model (" +
                                             ctx.remote->capabilities().hostname +
                                             "). It needs the 'latents' package; see Help ▸ Foundation model");
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

                nlohmann::json bundle;
                if (prompted > 0) {
                    requireWorker(ctx);
                    // A bundle without a prompt decoder is refused by name before
                    // any frame is sent; a worker that does not list the tasks
                    // (an older one) answers the run itself.
                    bundle = ctx.remote->call("model_info", {{"path", model}, {"model", model}, {"spec", model}}).result;
                    if (bundle.contains("tasks") && bundle["tasks"].is_array()) {
                        const nlohmann::json& tasks = bundle["tasks"];
                        if (std::find(tasks.begin(), tasks.end(), "prompt") == tasks.end()) {
                            std::string offered;
                            for (const nlohmann::json& t : tasks) offered += (offered.empty() ? "" : ", ") + t.get<std::string>();
                            throw std::runtime_error("This bundle cannot be prompted: its '" + bundle.value("head", std::string("?")) +
                                                     "' head has no prompt decoder (it offers " + offered +
                                                     "). Choose another task, or a bundle trained with a prompt decoder.");
                        }
                    }
                }

                const Index volume = d.z * d.planeSize();
                std::vector<float> flat(static_cast<std::size_t>(nc) * volume);
                double seconds = 0.0;
                std::size_t done = 0;
                std::string scores;
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
                    nlohmann::json params = {
                        {"model", model},
                        {"task", "prompt"},
                        {"objects", promptObjectsJson(f)},
                        {"min_voxels", p.getInt("min_voxels", 0)},
                        {"voxel_um", {meta.voxelUm[0], meta.voxelUm[1], meta.voxelUm[2]}},
                        {"device", workerDevice(ctx)},
                    };
                    const std::vector<double> tile = p.getDoubleList("tile");
                    if (tile.size() == 3 && (tile[0] > 0 || tile[1] > 0 || tile[2] > 0))
                        params["tile"] = {static_cast<Index>(tile[0]), static_cast<Index>(tile[1]), static_cast<Index>(tile[2])};
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
                    ++done;
                }
                const std::string className = p.getString("class_name", "object");
                for (LabelStats& s : labels->stats()) s.cls = className;
                std::uint32_t total = 0;
                for (const LabelStats& s : labels->stats()) total = std::max(total, s.id);

                out.labels = labels;
                out.ranOn = ctx.backend;
                out.seconds = seconds;
                Diagnostics diag = labelDiagnostics(*labels, summary(p, meta));
                diag.facts.push_back({"Model", model});
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
