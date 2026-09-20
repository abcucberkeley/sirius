// scikit-image segmentation: the methods that library has and this one does
// not implement natively.
//
// The Classical segmentation step covers everything that can be written in
// C++ and mirrored exactly in the Python workbench: the filters, the
// thresholds, the watersheds, the morphology. What is left over are a few
// large numerical methods that are not worth reimplementing -- a seeded random
// walker, an edge-driven active contour, two superpixel over-segmentations and
// a compactness-constrained watershed. They run in the Python worker, which
// hands back instance labels the way a Torch model does.
#include "core/ops/common.hpp"
#include "core/ops/builtin.hpp"
#include "core/rpc.hpp"

#include <nlohmann/json.hpp>

#include <chrono>

namespace sirius::app {

    namespace {

        class SkimageSegmentationOperation final : public Operation {
        public:
            SkimageSegmentationOperation() {
                info_.kind = "skimage_seg";
                info_.name = "scikit-image segmentation";
                info_.group = "Segment";
                info_.kindLabel = "SEGMENT";
                info_.diagnostics = DiagnosticsKind::Segment;
                info_.defaultCache = CachePolicy::Memory;
                info_.separableOverT = true;
                info_.producesLabels = true;
                info_.remoteCapable = true;   // it only runs in the worker
                info_.helpPage = "skimage_seg";
                info_.params = {
                    channelParam("channel", "Channel", 0),
                    choiceParam("method", "Method",
                                {"Random walker", "Active contour (geodesic)", "Superpixels (SLIC)",
                                 "Superpixels (Felzenszwalb)", "Watershed (compact)"},
                                "Random walker")
                        .withHelp("Random walker fills outwards from seeds and stops where the image changes, so it copes "
                                  "with a boundary too weak or too broken for a threshold. The geodesic active contour is "
                                  "driven by edges, where the Classical step's Chan-Vese is driven by regions. The "
                                  "superpixels over-segment into pieces to merge or measure. The compact watershed pulls "
                                  "its regions towards round shapes"),
                    doubleParam("threshold", "Threshold", 0.0).range(-1e9, 1e9, 0.01, 4).withHelp("The rough foreground the seeded methods start from; 0 asks the worker for an Otsu cut").visibleWhen("method", {"Random walker", "Active contour (geodesic)", "Watershed (compact)"}),
                    doubleParam("seed_depth", "Seed depth", 2.0).range(0.1, 100.0, 0.5, 1).withUnit("px").withHelp("Random walker and compact watershed: how far a peak of the distance map must stand above "
                                                                                                                   "its surroundings to seed its own object")
                        .visibleWhen("method", {"Random walker", "Watershed (compact)"}),
                    doubleParam("beta", "Diffusion stiffness", 130.0).range(1.0, 10000.0, 5.0, 1).withHelp("Random walker: how strongly an intensity step stops the walk. Higher follows faint "
                                                                                                           "boundaries, lower lets the regions spread past them")
                        .asAdvanced()
                        .visibleWhen("method", {"Random walker"}),
                    doubleParam("tolerance", "Solver tolerance", 0.001).range(1e-8, 1.0, 0.0005, 6).asAdvanced().visibleWhen("method", {"Random walker"}),
                    intParam("iterations", "Contour steps", 30).range(1, 2000).withHelp("Geodesic active contour: how many times the contour is moved").visibleWhen("method", {"Active contour (geodesic)"}),
                    intParam("smoothing", "Contour smoothing", 1).range(0, 5).asAdvanced().visibleWhen("method", {"Active contour (geodesic)"}),
                    doubleParam("balloon", "Balloon", 0.0).range(-3.0, 3.0, 0.5, 2).withHelp("Geodesic active contour: positive inflates the contour, negative deflates it, 0 lets the "
                                                                                             "edges alone decide")
                        .asAdvanced()
                        .visibleWhen("method", {"Active contour (geodesic)"}),
                    doubleParam("alpha", "Edge sharpness", 100.0).range(1.0, 10000.0, 5.0, 1).withHelp("Geodesic active contour: how sharply the edge map falls off at a gradient").asAdvanced().visibleWhen("method", {"Active contour (geodesic)"}),
                    doubleParam("edge_sigma", "Edge σ", 2.0).range(0.0, 50.0, 0.5, 2).withUnit("px").withHelp("Smoothing before the gradient (also Felzenszwalb's own smoothing)").asAdvanced().visibleWhen("method", {"Active contour (geodesic)", "Superpixels (Felzenszwalb)"}),
                    doubleParam("edge_threshold", "Edge threshold", 0.69).range(0.0, 1.0, 0.01, 3).asAdvanced().visibleWhen("method", {"Active contour (geodesic)"}),
                    intParam("n_segments", "Superpixels", 200).range(2, 1000000).withHelp("SLIC: roughly how many pieces to cut the volume into").visibleWhen("method", {"Superpixels (SLIC)"}),
                    doubleParam("compactness", "Compactness", 0.1).range(0.0001, 10000.0, 0.05, 4).withHelp("SLIC and the compact watershed: how much the pieces are pulled towards round shapes "
                                                                                                            "rather than following the intensity")
                        .visibleWhen("method", {"Superpixels (SLIC)", "Watershed (compact)"}),
                    doubleParam("scale", "Merge scale", 100.0).range(0.1, 100000.0, 5.0, 1).withHelp("Felzenszwalb: larger merges more, so the pieces come out bigger").visibleWhen("method", {"Superpixels (Felzenszwalb)"}),
                    intParam("min_size", "Smallest piece", 20).range(1, 1000000).withUnit("px").withHelp("Felzenszwalb: pieces smaller than this are merged into a neighbour").asAdvanced().visibleWhen("method", {"Superpixels (Felzenszwalb)"}),
                    intParam("min_voxels", "Min. voxels", 20).range(0, 1000000000),
                    stringParam("class_name", "Class", "object").asAdvanced(),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            std::string summary(const ParamSet& p, const DatasetMeta& meta) const override {
                const std::string method = p.getString("method", "Random walker");
                std::string detail;
                if (method == "Random walker") detail = "β " + formatNumber(p.getDouble("beta", 130.0), 0);
                else if (method == "Active contour (geodesic)") detail = std::to_string(p.getInt("iterations", 30)) + " steps";
                else if (method == "Superpixels (SLIC)") detail = std::to_string(p.getInt("n_segments", 200)) + " pieces";
                else if (method == "Superpixels (Felzenszwalb)") detail = "scale " + formatNumber(p.getDouble("scale", 100.0), 0);
                else detail = "compactness " + formatNumber(p.getDouble("compactness", 0.1), 3);
                return joinSummary({channelName(meta, p.getInt("channel", 0)), method, detail});
            }

            StepOutput run(const StepInput& input, const ParamSet& p, const StepContext& ctx) const override {
                const Validation v = validate(p, input.meta);
                if (!v.ok()) throw std::runtime_error(v.firstError());
                if (!ctx.remote)
                    throw std::runtime_error("scikit-image segmentation runs in the Python worker: start it in "
                                             "Preferences ▸ Python, or choose Classical segmentation, which runs here");
                if (!ctx.remote->supports("skimage_seg"))
                    throw std::runtime_error("The connected worker does not implement skimage_seg (" +
                                             ctx.remote->capabilities().hostname + "); it is older than this build");

                const DatasetMeta& meta = input.meta;
                const Dims5& d = meta.dims;
                const Index channel = p.getInt("channel", 0);
                const Index volume = d.z * d.y * d.x;
                StepOutput out;
                out.meta = meta;
                out.array = input.materialize([&](double f, const std::string& m) { ctx.report(0.05 * f, m); });
                auto labels = std::make_shared<LabelVolume>(d.t, d.z, d.y, d.x);
                const std::string className = p.getString("class_name", "object");

                nlohmann::json params = {
                    {"method", p.getString("method", "Random walker")},
                    {"seed_depth", p.getDouble("seed_depth", 2.0)},
                    {"beta", p.getDouble("beta", 130.0)},
                    {"tolerance", p.getDouble("tolerance", 0.001)},
                    {"iterations", p.getInt("iterations", 30)},
                    {"smoothing", p.getInt("smoothing", 1)},
                    {"balloon", p.getDouble("balloon", 0.0)},
                    {"alpha", p.getDouble("alpha", 100.0)},
                    {"edge_sigma", p.getDouble("edge_sigma", 2.0)},
                    {"edge_threshold", p.getDouble("edge_threshold", 0.69)},
                    {"n_segments", p.getInt("n_segments", 200)},
                    {"compactness", p.getDouble("compactness", 0.1)},
                    {"scale", p.getDouble("scale", 100.0)},
                    {"min_size", p.getInt("min_size", 20)},
                    {"min_voxels", p.getInt("min_voxels", 20)},
                };
                // 0 means "work the threshold out yourself": the worker cannot
                // tell an explicit zero from an unset one otherwise
                const double cut = p.getDouble("threshold", 0.0);
                if (cut != 0.0) params["threshold"] = cut;

                const auto t0 = std::chrono::steady_clock::now();
                std::uint32_t total = 0;
                std::string note;
                for (Index t = 0; t < d.t; ++t) {
                    ctx.throwIfCancelled();
                    const double base = 0.05 + 0.9 * static_cast<double>(t) / d.t, span = 0.9 / d.t;
                    rpc::TensorRef in;
                    in.name = "input";
                    in.dtype = "float32";
                    in.shape = {d.z, d.y, d.x};
                    in.data = out.array->volume(channel, t).data();
                    in.nbytes = static_cast<std::size_t>(volume) * sizeof(float);
                    WorkerResult r = ctx.remote->call(
                        "run", {{"kind", "skimage_seg"}, {"params", params}}, {in},
                        [&](double f, const std::string& m) { ctx.report(base + span * f, m); },
                        [&] { return ctx.isCancelled(); });
                    ctx.throwIfCancelled();
                    const rpc::Tensor* got = nullptr;
                    for (const rpc::Tensor& tensor : r.tensors)
                        if (tensor.name == "labels") got = &tensor;
                    if (!got) throw std::runtime_error("scikit-image segmentation: the worker returned no 'labels' tensor");
                    if (got->shape.size() != 3 || got->shape[0] != d.z || got->shape[1] != d.y || got->shape[2] != d.x)
                        throw std::runtime_error("scikit-image segmentation: the worker's labels do not match the volume");
                    std::uint32_t* into = labels->volume(t);
                    std::copy_n(got->asUInt32(), volume, into);
                    labels->recomputeStats(t);
                    for (LabelStats& s : labels->stats()) s.cls = className;
                    // each frame is segmented on its own, so the count is the
                    // sum over frames, as the classical step reports it
                    std::uint32_t highest = 0;
                    for (Index i = 0; i < volume; ++i) highest = std::max(highest, into[i]);
                    total += highest;
                    if (t == 0 && r.result.contains("note") && r.result["note"].is_string())
                        note = r.result["note"].get<std::string>();
                }
                labels->resetMaxLabel();
                LabelFlagRules rules;
                labels->applyFlags(rules);
                out.labels = labels;
                out.ranOn = ctx.backend;
                out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                out.note = std::to_string(total) + " labels · worker" + (note.empty() ? "" : " · " + note);
                out.diagnostics = labelDiagnostics(*labels, summary(p, meta) + " · " + std::to_string(total) + " labels");
                if (!note.empty()) out.diagnostics.facts.push_back({"Note", note});
                ctx.report(1.0, "");
                return out;
            }

        private:
            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeSkimageSegmentationOperation() { return std::make_unique<SkimageSegmentationOperation>(); }

} // namespace sirius::app
