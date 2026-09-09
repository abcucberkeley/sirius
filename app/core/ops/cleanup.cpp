// Label cleanup: drops small and border-touching objects, relabels densely
// and re-flags the rest for review. The intensities pass through.
#include "core/ops/builtin.hpp"

#include <algorithm>
#include <cstdio>
#include <unordered_set>

namespace sirius::app {

    namespace {

        class LabelCleanupOperation final : public Operation {
        public:
            LabelCleanupOperation() {
                info_.kind = "cleanup";
                info_.name = "Label cleanup";
                info_.group = "Segment";
                info_.kindLabel = "SEGMENT";
                info_.diagnostics = DiagnosticsKind::Segment;
                info_.defaultCache = CachePolicy::Memory;
                info_.needsLabels = true;
                info_.producesLabels = true;
                info_.helpPage = "cleanup";
                info_.params = {
                    intParam("min_voxels", "Min. voxels", 50).range(0, 1000000000),
                    boolParam("remove_border", "Remove border-touching", false),
                    boolParam("relabel", "Relabel densely", true),
                    doubleParam("low_conf", "Low-confidence flag", 0.6).range(0.0, 1.0, 0.05, 2),
                    doubleParam("size_outlier_factor", "Size outlier factor", 4.0).range(1.0, 100.0, 0.5, 1).withHelp("Objects larger than this × the median volume are flagged as possible merges"),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            std::string summary(const ParamSet& p, const DatasetMeta&) const override {
                return joinSummary({"min " + std::to_string(p.getInt("min_voxels", 50)) + " voxels",
                                    p.getBool("remove_border") ? "drop border" : "", p.getBool("relabel", true) ? "relabel" : ""});
            }

            Validation validate(const ParamSet& p, const DatasetMeta& in) const override {
                Validation v = Operation::validate(p, in);
                v.warnings.push_back("Needs the labels of a segmentation step upstream.");
                return v;
            }

            StepOutput run(const StepInput& input, const ParamSet& p, const StepContext& ctx) const override {
                if (!input.labels || input.labels->empty())
                    throw std::runtime_error("Label cleanup needs labels: add a segmentation step before it");
                const DatasetMeta& meta = input.meta;
                StepOutput out;
                out.meta = meta;
                out.array = input.array;
                out.source = input.source;
                auto labels = input.labels->clone();
                const Index minVoxels = p.getInt("min_voxels", 50);
                const bool removeBorder = p.getBool("remove_border", false);
                const bool relabel = p.getBool("relabel", true);
                LabelFlagRules rules;
                rules.lowConfidence = p.getDouble("low_conf", 0.6);
                rules.sizeOutlierFactor = p.getDouble("size_outlier_factor", 4.0);
                const Index n = labels->volumeSize();
                for (Index t = 0; t < labels->t(); ++t) {
                    ctx.throwIfCancelled();
                    ctx.report(static_cast<double>(t) / labels->t(), "t " + std::to_string(t));
                    std::uint32_t* vol = labels->volume(t);
                    if (removeBorder) {
                        labels->recomputeStats(t);
                        std::unordered_set<std::uint32_t> drop;
                        for (const LabelStats& s : labels->stats())
                            if (s.touchesBorder) drop.insert(s.id);
                        if (!drop.empty())
                            for (Index i = 0; i < n; ++i)
                                if (vol[i] && drop.count(vol[i])) vol[i] = 0;
                    }
                    ctx.throwIfCancelled();
                    if (minVoxels > 0 || relabel) removeSmall(vol, n, minVoxels);
                    ctx.throwIfCancelled();
                    labels->recomputeStats(t);
                    labels->applyFlags(rules);
                }
                if (relabel) labels->resetMaxLabel();   // ids are dense again
                out.labels = labels;
                out.ranOn = Backend::Cpu;
                out.note = std::to_string(labels->stats().size()) + " labels kept · CPU";
                out.diagnostics = labelDiagnostics(*labels, summary(p, meta) + " · " + std::to_string(labels->stats().size()) + " labels");
                ctx.report(1.0, "");
                return out;
            }

        private:
            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeLabelCleanupOperation() { return std::make_unique<LabelCleanupOperation>(); }

} // namespace sirius::app
