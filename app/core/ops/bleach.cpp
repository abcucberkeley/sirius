// Bleach correction: scale every frame (time point, or plane) of a channel
// so its total intensity matches the first frame or the mean.
#include "core/ops/builtin.hpp"

#include <sirius/image_ops.hpp>

namespace sirius::app {

    namespace {

        constexpr const char* kFirst = "Match first frame";
        constexpr const char* kMean = "Match mean";

        class BleachOperation final : public Operation {
        public:
            BleachOperation() {
                info_.kind = "bleach";
                info_.name = "Bleach correction";
                info_.group = "Intensity";
                info_.kindLabel = "INTENSITY";
                info_.defaultCache = CachePolicy::Recompute;
                info_.helpPage = "bleach";
                info_.params = {
                    choiceParam("mode", "Reference", {kFirst, kMean}, kFirst),
                    choiceParam("over", "Frames along", {"t", "z"}, "t")
                        .withHelp("t equalizes the time series, z the planes of every stack"),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            std::string summary(const ParamSet& params, const DatasetMeta&) const override {
                return joinSummary({params.getString("mode", kFirst) == kMean ? "match mean" : "match first frame",
                                    "over " + params.getString("over", "t")});
            }

            Validation validate(const ParamSet& params, const DatasetMeta& input) const override {
                Validation v = Operation::validate(params, input);
                const std::string over = params.getString("over", "t");
                if (over == "t" && input.dims.t <= 1) v.warnings.push_back("A single time point: nothing to equalize over t.");
                if (over == "z" && input.dims.z <= 1) v.warnings.push_back("A single plane: nothing to equalize over z.");
                return v;
            }

            DatasetMeta outputMeta(const ParamSet&, const DatasetMeta& input) const override {
                DatasetMeta out = input;
                out.sourceType = PixelType::Float32;
                return out;
            }

            StepOutput run(const StepInput& input, const ParamSet& params, const StepContext& ctx) const override {
                const DatasetMeta& meta = input.meta;
                const bool toMean = params.getString("mode", kFirst) == kMean;
                const bool overT = params.getString("over", "t") != "z";
                StepOutput out;
                out.meta = outputMeta(params, meta);
                ArrayPtr in = input.materialize([&](double f, const std::string& m) { ctx.report(0.5 * f, m); });
                auto result = std::make_shared<Array5>(in->clone());
                const Dims5& d = meta.dims;
                for (Index c = 0; c < d.c; ++c) {
                    ctx.throwIfCancelled();
                    ctx.report(0.5 + 0.5 * static_cast<double>(c) / d.c, "channel " + std::to_string(c));
                    if (overT) {
                        // channel c is a contiguous (t, z, y, x) block: frames = t, "plane" = a volume
                        equalizeFrames(result->plane(c, 0, 0), d.t, d.z * d.planeSize(), toMean);
                    } else {
                        for (Index t = 0; t < d.t; ++t) {
                            ctx.throwIfCancelled();
                            equalizeFrames(result->plane(c, t, 0), d.z, d.planeSize(), toMean);
                        }
                    }
                }
                out.array = result;
                // out.labels stays null: the executor carries the input's
                // labels through, and the corrections painted on this step with them
                out.ranOn = Backend::Cpu;
                out.note = summary(params, meta) + " · CPU";
                out.diagnostics = genericDiagnostics(input, out, summary(params, meta));
                ctx.report(1.0, "");
                return out;
            }

        private:
            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeBleachOperation() { return std::make_unique<BleachOperation>(); }

} // namespace sirius::app
