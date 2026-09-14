// Threshold: a global cut (manual, Otsu or a percentile) on one channel,
// instances by connected components or a distance watershed.
#include "core/ops/builtin.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include <sirius/image_ops.hpp>

namespace sirius::app {

    float otsuThreshold(const float* v, Index n) {
        // over the finite values: an infinite end has no bins to split
        float mn = std::numeric_limits<float>::infinity(), mx = -mn;
        for (Index i = 0; i < n; ++i) {
            if (!std::isfinite(v[i])) continue;
            mn = std::min(mn, v[i]);
            mx = std::max(mx, v[i]);
        }
        if (!(mx > mn)) return mn;
        constexpr int bins = 256;
        const std::vector<double> h = histogram(v, n, bins, mn, mx);
        double total = 0.0, sumAll = 0.0;
        for (int i = 0; i < bins; ++i) {
            total += h[static_cast<std::size_t>(i)];
            sumAll += i * h[static_cast<std::size_t>(i)];
        }
        double wB = 0.0, sumB = 0.0, best = -1.0;
        int bestBin = 0;
        for (int i = 0; i < bins; ++i) {
            wB += h[static_cast<std::size_t>(i)];
            if (wB == 0.0) continue;
            const double wF = total - wB;
            if (wF == 0.0) break;
            sumB += i * h[static_cast<std::size_t>(i)];
            const double mB = sumB / wB, mF = (sumAll - sumB) / wF;
            const double between = wB * wF * (mB - mF) * (mB - mF);
            if (between > best) {
                best = between;
                bestBin = i;
            }
        }
        return mn + (mx - mn) * static_cast<float>(bestBin + 1) / bins;
    }

    namespace {

        class ThresholdOperation final : public Operation {
        public:
            ThresholdOperation() {
                info_.kind = "threshold";
                info_.name = "Threshold";
                info_.group = "Segment";
                info_.kindLabel = "SEGMENT";
                info_.diagnostics = DiagnosticsKind::Segment;
                info_.defaultCache = CachePolicy::Memory;
                info_.separableOverT = true;
                info_.producesLabels = true;
                info_.helpPage = "threshold";
                info_.params = {
                    channelParam("channel", "Channel", 0),
                    choiceParam("method", "Method", {"Manual", "Otsu", "Percentile"}, "Otsu"),
                    doubleParam("value", "Value", 0.5).range(-1e9, 1e9, 0.01, 4).withHelp("Manual threshold"),
                    doubleParam("percentile", "Percentile", 90.0).range(0.0, 100.0, 0.5, 1).withUnit("%"),
                    choiceParam("post", "Instances", {"Connected components", "Watershed (distance)"}, "Connected components"),
                    intParam("min_voxels", "Min. voxels", 20).range(0, 1000000000),
                    doubleParam("seed_distance", "Seed distance", 5.0).range(1.0, 200.0, 0.5, 1).withUnit("px").asAdvanced(),
                    stringParam("class_name", "Class", "object").asAdvanced(),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            std::string summary(const ParamSet& p, const DatasetMeta& in) const override {
                const std::string method = p.getString("method", "Otsu");
                std::string cut = method;
                if (method == "Manual") cut = "> " + formatNumber(p.getDouble("value", 0.5), 3);
                else if (method == "Percentile") cut = "> p" + formatNumber(p.getDouble("percentile", 90.0), 1);
                return joinSummary({channelName(in, p.getInt("channel")), cut,
                                    p.getString("post").rfind("Watershed", 0) == 0 ? "watershed" : "components"});
            }

            Validation validate(const ParamSet& p, const DatasetMeta& in) const override {
                Validation v = Operation::validate(p, in);
                if (in.rgb) v.errors.push_back("Threshold needs an intensity channel, not an RGB merge.");
                return v;
            }

            std::size_t estimatedOutputBytes(const ParamSet&, const DatasetMeta& in) const override {
                return in.dims.bytes() + static_cast<std::size_t>(in.dims.t * in.dims.z * in.dims.planeSize()) * sizeof(std::uint32_t);
            }

            StepOutput run(const StepInput& input, const ParamSet& p, const StepContext& ctx) const override {
                const Validation v = validate(p, input.meta);
                if (!v.ok()) throw std::runtime_error(v.firstError());
                const DatasetMeta& meta = input.meta;
                const Dims5& d = meta.dims;
                const Index channel = p.getInt("channel", 0);
                StepOutput out;
                out.meta = meta;
                out.array = input.materialize([&](double f, const std::string& m) { ctx.report(0.2 * f, m); });
                auto labels = std::make_shared<LabelVolume>(d.t, d.z, d.y, d.x);
                LabelPostOptions post;
                post.post = p.getString("post", "Connected components").rfind("Watershed", 0) == 0 ? "Watershed (distance)"
                                                                                                   : "Connected components";
                post.minVoxels = p.getInt("min_voxels", 20);
                post.seedMinDistance = p.getDouble("seed_distance", 5.0);
                post.className = p.getString("class_name", "object");
                post.poll = [&ctx] { ctx.throwIfCancelled(); };
                const std::string method = p.getString("method", "Otsu");
                std::uint32_t total = 0;
                std::string cuts;
                for (Index t = 0; t < d.t; ++t) {
                    ctx.throwIfCancelled();
                    ctx.report(0.2 + 0.8 * static_cast<double>(t) / d.t, "t " + std::to_string(t));
                    const BufferView<const float> vol = out.array->volume(channel, t);
                    float cut = static_cast<float>(p.getDouble("value", 0.5));
                    if (method == "Otsu") cut = otsuThreshold(vol.data(), vol.size());
                    else if (method == "Percentile")
                        cut = percentiles(vol.data(), vol.size(), 0.0, p.getDouble("percentile", 90.0)).second;
                    post.threshold = cut;
                    ctx.throwIfCancelled();
                    total += labelsFromProbabilities(vol.data(), nullptr, d.z, d.y, d.x, post, *labels, t);
                    // the intensities are not probabilities: confidence is
                    // unknown, in this frame's table (each frame keeps its own)
                    for (LabelStats& s : labels->stats()) s.confidence = 1.0;
                    labels->applyFlags(post.flags);
                    if (t == 0) cuts = formatNumber(cut, 4);
                }
                out.labels = labels;
                out.ranOn = Backend::Cpu;
                out.note = "threshold " + cuts + " · " + std::to_string(total) + " labels · CPU";
                out.diagnostics = labelDiagnostics(*labels, summary(p, meta) + " · " + std::to_string(total) + " labels");
                ctx.report(1.0, "");
                return out;
            }

        private:
            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeThresholdOperation() { return std::make_unique<ThresholdOperation>(); }

} // namespace sirius::app
