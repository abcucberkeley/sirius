// Track objects: follow the labels of a segmentation step through time.
//
// Each frame's objects are matched to the next frame's by optimal assignment
// (core/tracking.hpp), not greedily, so two objects that pass close by keep
// their identities. The cost is the centroid distance in micrometres, gated
// by a maximum step, optionally mixed with how much the two objects overlap.
// Frames where an object was missed are bridged afterwards by a second
// assignment between track ends and track starts. The labels are rewritten so
// one object keeps one id for its whole life, which is what makes the label
// colours, the solo view and the review table follow it.
#include "core/ops/builtin.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <numeric>

#include "core/tracking.hpp"

namespace sirius::app {

    namespace {

        class TrackOperation final : public Operation {
        public:
            TrackOperation() {
                info_.kind = "track";
                info_.name = "Track objects";
                info_.group = "Segment";
                info_.kindLabel = "TRACK";
                info_.diagnostics = DiagnosticsKind::Segment;
                info_.defaultCache = CachePolicy::Memory;
                info_.needsLabels = true;
                info_.producesLabels = true;
                info_.remoteCapable = true;   // the btrack backend runs in the Python worker
                info_.helpPage = "track";
                info_.params = {
                    choiceParam("tracker", "Tracker", {"Built-in (assignment)", "btrack (Bayesian)"},
                                "Built-in (assignment)")
                        .withHelp("Built-in matches each frame to the next by optimal assignment on distance and "
                                  "overlap. btrack adds a motion model, so it holds identities through a crossing, "
                                  "and reconstructs lineages when cells divide; it runs in the Python worker"),
                    doubleParam("max_distance", "Max. step", 10.0).range(0.0, 100000.0, 0.5, 2).withUnit("µm").withHelp("How far an object may move between frames; the gate that keeps distant objects apart"),
                    doubleParam("overlap_weight", "Overlap weight", 0.5).range(0.0, 1.0, 0.05, 2).visibleWhen("tracker", {"Built-in (assignment)"}).withHelp("0 matches on centroid distance alone, 1 on shared voxels alone; in between mixes them"),
                    intParam("max_gap", "Close gaps", 1).range(0, 100).withUnit("frames").visibleWhen("tracker", {"Built-in (assignment)"}).withHelp("Frames an object may be missed for and still continue the same track (0 = never)"),
                    intParam("min_length", "Min. track length", 2).range(1, 1000000).withUnit("frames").withHelp("Tracks seen in fewer frames than this are dropped"),
                    boolParam("relabel", "Relabel by track", true).visibleWhen("tracker", {"Built-in (assignment)"}).withHelp("Give every object of a track the track's id, so one object keeps one colour over time"),
                    stringParam("config", "btrack config", "").visibleWhen("tracker", {"btrack (Bayesian)"}).withHelp("btrack only: a tracker configuration JSON; empty uses btrack's packaged cell "
                                                                                                                      "configuration, which is fetched and cached on first use")
                        .asAdvanced(),
                    boolParam("optimise", "Reconstruct lineages", true).visibleWhen("tracker", {"btrack (Bayesian)"}).withHelp("btrack only: run the global hypothesis optimisation, which is what links a mother "
                                                                                                                               "to its daughters")
                        .asAdvanced(),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            std::string summary(const ParamSet& p, const DatasetMeta& meta) const override {
                const Index gap = p.getInt("max_gap", 1);
                const bool bayes = p.getString("tracker", "Built-in (assignment)").rfind("btrack", 0) == 0;
                if (bayes)
                    return joinSummary({"btrack", "≤ " + formatNumber(p.getDouble("max_distance", 10.0), 1) + " µm",
                                        p.getBool("optimise", true) ? "lineages" : "no lineages",
                                        meta.dims.t > 1 ? std::to_string(meta.dims.t) + " frames" : "one frame only"});
                return joinSummary({"≤ " + formatNumber(p.getDouble("max_distance", 10.0), 1) + " µm",
                                    "overlap " + formatNumber(p.getDouble("overlap_weight", 0.5), 2),
                                    gap > 0 ? "gaps ≤ " + std::to_string(gap) : "no gaps",
                                    meta.dims.t > 1 ? std::to_string(meta.dims.t) + " frames" : "one frame only"});
            }

            Validation validate(const ParamSet& p, const DatasetMeta& in) const override {
                Validation v = Operation::validate(p, in);
                v.warnings.push_back("Needs the labels of a segmentation step upstream.");
                if (in.dims.t < 2)
                    v.warnings.push_back("The dataset has one time point: tracking needs at least two to link anything.");
                if (p.getDouble("max_distance", 10.0) <= 0.0)
                    v.errors.push_back("Max. step must be greater than zero.");
                return v;
            }

            StepOutput run(const StepInput& input, const ParamSet& p, const StepContext& ctx) const override {
                if (!input.labels || input.labels->empty())
                    throw std::runtime_error("Track objects needs labels: add a segmentation step before it");
                const Validation v = validate(p, input.meta);
                if (!v.ok()) throw std::runtime_error(v.firstError());
                const DatasetMeta& meta = input.meta;
                StepOutput out;
                out.meta = meta;
                out.array = input.array;
                out.source = input.source;

                const LabelVolume& in = *input.labels;
                const Index frames = in.t();
                if (p.getString("tracker", "Built-in (assignment)").rfind("btrack", 0) == 0)
                    return runBtrack(input, p, ctx, std::move(out));
                TrackOptions options;
                options.maxDistanceUm = p.getDouble("max_distance", 10.0);
                options.overlapWeight = p.getDouble("overlap_weight", 0.5);
                options.maxGap = p.getInt("max_gap", 1);
                options.minLength = p.getInt("min_length", 2);

                std::vector<std::vector<TrackObject>> byFrame(static_cast<std::size_t>(frames));
                for (Index t = 0; t < frames; ++t) {
                    ctx.throwIfCancelled();
                    ctx.report(0.4 * static_cast<double>(t) / std::max<Index>(1, frames), "objects of frame " + std::to_string(t));
                    byFrame[static_cast<std::size_t>(t)] = objectsOfFrame(in, t);
                }
                std::vector<std::vector<Index>> overlap;
                if (options.overlapWeight > 0.0 && frames > 1) {
                    overlap.resize(static_cast<std::size_t>(frames - 1));
                    for (Index t = 0; t + 1 < frames; ++t) {
                        ctx.throwIfCancelled();
                        ctx.report(0.4 + 0.3 * static_cast<double>(t) / std::max<Index>(1, frames - 1), "overlap");
                        overlap[static_cast<std::size_t>(t)] =
                            overlapBetween(in, t, byFrame[static_cast<std::size_t>(t)], byFrame[static_cast<std::size_t>(t + 1)]);
                    }
                }
                ctx.report(0.75, "linking");
                const TrackResult linked = linkTracks(byFrame, overlap, meta.voxelUm, options);
                ctx.throwIfCancelled();

                // rewrite the labels so a track keeps one id for its whole life
                auto labels = std::make_shared<LabelVolume>(in.t(), in.z(), in.y(), in.x());
                const bool relabel = p.getBool("relabel", true);
                const Index planeVoxels = in.z() * in.y() * in.x();
                std::vector<std::map<std::uint32_t, std::uint32_t>> mapping(static_cast<std::size_t>(frames));
                for (const Track& tr : linked.tracks)
                    for (const auto& [t, label] : tr.points)
                        mapping[static_cast<std::size_t>(t)][label] = relabel ? tr.id : label;
                for (Index t = 0; t < frames; ++t) {
                    ctx.throwIfCancelled();
                    const std::uint32_t* src = in.volume(t);
                    std::uint32_t* dst = labels->volume(t);
                    const auto& m = mapping[static_cast<std::size_t>(t)];
                    for (Index i = 0; i < planeVoxels; ++i) {
                        const auto it = src[i] ? m.find(src[i]) : m.end();
                        dst[i] = it == m.end() ? 0u : it->second;
                    }
                }
                ctx.report(0.95, "statistics");
                for (Index t = 0; t < frames; ++t) labels->recomputeStats(t);
                for (LabelStats& s : labels->stats()) s.cls = "track";

                labels->setTracked(true);   // one id, one object, every frame
                out.labels = labels;
                out.ranOn = Backend::Cpu;
                out.diagnostics = trackDiagnostics(linked, byFrame, meta, summary(p, meta));
                char note[220];
                std::snprintf(note, sizeof note, "%zu tracks · %lld links · %lld gaps closed · CPU", linked.tracks.size(),
                              static_cast<long long>(linked.links), static_cast<long long>(linked.gapsClosed));
                out.note = note;
                ctx.report(1.0, "");
                return out;
            }

        private:
            // btrack in the Python worker: the labels go over as they are and
            // come back renumbered by track, so nothing about btrack leaks
            // into the application.
            StepOutput runBtrack(const StepInput& input, const ParamSet& p, const StepContext& ctx, StepOutput out) const {
                if (!ctx.remote)
                    throw std::runtime_error("btrack tracking needs the Python worker: start it in Preferences ▸ Python");
                const LabelVolume& in = *input.labels;
                const Index frames = in.t(), volume = in.z() * in.y() * in.x();
                std::vector<std::uint32_t> flat(static_cast<std::size_t>(frames * volume));
                for (Index t = 0; t < frames; ++t) std::copy_n(in.volume(t), volume, flat.data() + t * volume);

                nlohmann::json params = {
                    {"max_distance", p.getDouble("max_distance", 10.0)},
                    {"min_length", p.getInt("min_length", 2)},
                    {"config", p.getString("config")},
                    {"optimise", p.getBool("optimise", true)},
                    {"voxel_um", {input.meta.voxelUm[0], input.meta.voxelUm[1], input.meta.voxelUm[2]}},
                };
                rpc::TensorRef marks;
                marks.name = "labels";
                marks.dtype = "uint32";
                marks.shape = {frames, in.z(), in.y(), in.x()};
                marks.data = flat.data();
                marks.nbytes = flat.size() * sizeof(std::uint32_t);
                ctx.report(0.1, "btrack");
                const auto t0 = std::chrono::steady_clock::now();
                WorkerResult r = ctx.remote->call(
                    "run", {{"kind", "btrack"}, {"params", params}}, {marks},
                    [&](double f, const std::string& m) { ctx.report(0.1 + 0.8 * f, m); },
                    [&] { return ctx.isCancelled(); });
                ctx.throwIfCancelled();
                const rpc::Tensor* got = nullptr;
                for (const rpc::Tensor& tensor : r.tensors)
                    if (tensor.name == "labels") got = &tensor;
                if (!got) throw std::runtime_error("btrack tracking: the worker returned no 'labels' tensor");
                if (got->shape.size() != 4 || got->shape[0] != frames || got->shape[1] != in.z() ||
                    got->shape[2] != in.y() || got->shape[3] != in.x())
                    throw std::runtime_error("btrack tracking: the worker's labels do not match the volume");

                auto labels = std::make_shared<LabelVolume>(in.t(), in.z(), in.y(), in.x());
                const std::uint32_t* src = got->asUInt32();
                for (Index t = 0; t < frames; ++t) {
                    std::copy_n(src + t * volume, volume, labels->volume(t));
                    labels->recomputeStats(t);
                }
                for (LabelStats& s : labels->stats()) s.cls = "track";
                labels->setTracked(true);   // one id, one object, every frame
                out.labels = labels;
                out.ranOn = ctx.backend;
                out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

                Diagnostics d;
                d.kind = DiagnosticsKind::Segment;
                const long long tracks = r.result.value("tracks", 0LL);
                d.summary = summary(p, input.meta) + " · " + std::to_string(tracks) + " tracks";
                d.facts.push_back({"Tracker", "btrack"});
                d.facts.push_back({"Tracks", std::to_string(tracks)});
                d.facts.push_back({"Objects", std::to_string(r.result.value("objects", 0LL))});
                d.facts.push_back({"Mean length", formatNumber(r.result.value("mean_length", 0.0), 1) + " / " +
                                                      std::to_string(input.meta.dims.t) + " frames"});
                d.facts.push_back({"Longest", std::to_string(r.result.value("longest", 0LL)) + " frames"});
                d.facts.push_back({"Divisions", std::to_string(r.result.value("divisions", 0LL))});
                out.diagnostics = std::move(d);
                char note[200];
                std::snprintf(note, sizeof note, "btrack · %lld tracks · %lld divisions · %.1f s", tracks,
                              r.result.value("divisions", 0LL), out.seconds);
                out.note = note;
                ctx.report(1.0, "");
                return out;
            }

            // Track table plus the facts that say whether the linking is sane:
            // a mean length near the frame count means objects were followed,
            // a mean near one means the gate is too tight.
            static Diagnostics trackDiagnostics(const TrackResult& linked, const std::vector<std::vector<TrackObject>>& byFrame,
                                                const DatasetMeta& meta, const std::string& summary) {
                Diagnostics d;
                d.kind = DiagnosticsKind::Segment;
                d.summary = summary + " · " + std::to_string(linked.tracks.size()) + " tracks";
                Index objects = 0;
                for (const auto& frame : byFrame) objects += static_cast<Index>(frame.size());
                double meanLength = 0.0;
                std::size_t longest = 0;
                for (const Track& t : linked.tracks) {
                    meanLength += static_cast<double>(t.length()) / std::max<std::size_t>(1, linked.tracks.size());
                    longest = std::max(longest, t.length());
                }
                d.facts.push_back({"Tracks", std::to_string(linked.tracks.size())});
                d.facts.push_back({"Objects", std::to_string(objects)});
                d.facts.push_back({"Mean length", formatNumber(meanLength, 1) + " / " + std::to_string(meta.dims.t) + " frames"});
                d.facts.push_back({"Longest", std::to_string(longest) + " frames"});
                d.facts.push_back({"Gaps closed", std::to_string(linked.gapsClosed)});

                DiagnosticTable table;
                table.caption = "Tracks";
                table.header = {"TRACK", "FRAMES", "FROM", "TO", "PATH (µm)", "SPEED (µm/frame)"};
                auto centroidOf = [&](Index t, std::uint32_t label) {
                    for (const TrackObject& o : byFrame[static_cast<std::size_t>(t)])
                        if (o.label == label) return o.centroid;
                    return std::array<double, 3>{0, 0, 0};
                };
                for (const Track& tr : linked.tracks) {
                    double path = 0.0;
                    for (std::size_t k = 1; k < tr.points.size(); ++k) {
                        const auto a = centroidOf(tr.points[k - 1].first, tr.points[k - 1].second);
                        const auto b = centroidOf(tr.points[k].first, tr.points[k].second);
                        const double dz = (a[0] - b[0]) * meta.voxelUm[2], dy = (a[1] - b[1]) * meta.voxelUm[1],
                                     dx = (a[2] - b[2]) * meta.voxelUm[0];
                        path += std::sqrt(dz * dz + dy * dy + dx * dx);
                    }
                    const double span = static_cast<double>(std::max<std::size_t>(1, tr.points.size() - 1));
                    table.rows.push_back({std::to_string(tr.id), std::to_string(tr.length()), std::to_string(tr.first()),
                                          std::to_string(tr.last()), formatNumber(path, 2), formatNumber(path / span, 2)});
                }
                d.table = std::move(table);
                return d;
            }

            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeTrackOperation() { return std::make_unique<TrackOperation>(); }

} // namespace sirius::app
