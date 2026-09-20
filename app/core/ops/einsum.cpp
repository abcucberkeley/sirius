// Einsum reduce and its two presets (Max projection, Mean over time): reduce
// the array along any set of axes with sum / mean / max / min, leaving the
// reduced axes at length 1 so the axis semantics never move.
#include "core/ops/common.hpp"
#include "core/ops/builtin.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <sirius/image_ops.hpp>

namespace sirius::app {

    namespace {

        constexpr const char* kAxes = "ctzyx";

        ReduceOp reduceOpOf(const std::string& s) {
            if (s == "sum") return ReduceOp::Sum;
            if (s == "max") return ReduceOp::Max;
            if (s == "min") return ReduceOp::Min;
            return ReduceOp::Mean;
        }

        std::array<bool, 5> reduceMask(const std::string& keep) {
            std::array<bool, 5> m{};
            for (int i = 0; i < 5; ++i) m[static_cast<std::size_t>(i)] = keep.find(kAxes[i]) == std::string::npos;
            return m;
        }

        std::string reducedAxes(const std::array<bool, 5>& m) {
            std::string out;
            for (int i = 0; i < 5; ++i)
                if (m[static_cast<std::size_t>(i)]) {
                    if (!out.empty()) out += ", ";
                    out += kAxes[i];
                }
            return out;
        }

        // Shared implementation: the presets forward to it with fixed params.
        struct ReduceCore {
            static std::string summaryOf(const std::string& keep, const std::string& red, const DatasetMeta& input) {
                const std::array<bool, 5> m = reduceMask(keep);
                if (std::none_of(m.begin(), m.end(), [](bool b) { return b; })) return "identity — nothing reduced";
                Dims5 d = input.dims;
                std::string shape;
                for (int i = 0; i < 5; ++i)
                    if (!m[static_cast<std::size_t>(i)]) {
                        if (!shape.empty()) shape += " ";
                        shape += std::string(1, kAxes[i]) + std::to_string(d[kAxes_[static_cast<std::size_t>(i)]]);
                    }
                if (shape.empty()) shape = "scalar";
                return red + " over " + reducedAxes(m) + " · " + shape;
            }
            static constexpr std::array<Axis, 5> kAxes_{Axis::C, Axis::T, Axis::Z, Axis::Y, Axis::X};

            static DatasetMeta outputMetaOf(const std::string& keep, const DatasetMeta& input) {
                DatasetMeta out = input;
                const std::array<bool, 5> m = reduceMask(keep);
                for (int i = 0; i < 5; ++i)
                    if (m[static_cast<std::size_t>(i)]) out.dims[kAxes_[static_cast<std::size_t>(i)]] = 1;
                if (m[0]) {
                    out.channels.clear();
                    out.rgb = false;
                    out.normalizeChannels();
                }
                if (m[2]) out.sim = SimLayout{};
                out.sourceType = PixelType::Float32;
                return out;
            }

            static StepOutput runOf(const std::string& keep, const std::string& red, const StepInput& input,
                                    const StepContext& ctx) {
                const DatasetMeta& meta = input.meta;
                StepOutput out;
                out.meta = outputMetaOf(keep, meta);
                const std::array<bool, 5> m = reduceMask(keep);
                ctx.report(0.0, "reading");
                ArrayPtr in = input.materialize([&](double f, const std::string& msg) { ctx.report(f * 0.5, msg); });
                ctx.throwIfCancelled();
                auto result = allocateLike(out.meta);
                const Extent5 extent{meta.dims.c, meta.dims.t, meta.dims.z, meta.dims.y, meta.dims.x};
                ctx.report(0.6, red + " over " + reducedAxes(m));
                reduceAxes(in->data(), extent, m, reduceOpOf(red), result->data());
                out.array = result;
                // out.labels stays null: the executor carries the input's
                // labels through when t, z, y and x are all kept (with the
                // corrections painted on this step), and a reduction of any
                // of them leaves them behind
                out.ranOn = Backend::Cpu;
                out.note = summaryOf(keep, red, meta) + " · CPU";
                out.diagnostics = genericDiagnostics(input, out, summaryOf(keep, red, meta));
                ctx.report(1.0, "");
                return out;
            }
        };

        class EinsumOperation final : public Operation {
        public:
            EinsumOperation() {
                info_.kind = "einsum";
                info_.name = "Einsum reduce";
                info_.group = "Reduce";
                info_.kindLabel = "EINSUM";
                info_.defaultCache = CachePolicy::Memory;
                info_.helpPage = "einsum";
                info_.params = {
                    axesParam("keep", "Axes kept", "czyx").withHelp("Axes that survive; the others are reduced"),
                    choiceParam("reduction", "Reduction", {"sum", "mean", "max", "min"}, "mean"),
                };
            }
            const OpInfo& info() const noexcept override { return info_; }
            std::string summary(const ParamSet& p, const DatasetMeta& in) const override {
                return ReduceCore::summaryOf(p.getString("keep", "czyx"), p.getString("reduction", "mean"), in);
            }
            DatasetMeta outputMeta(const ParamSet& p, const DatasetMeta& in) const override {
                return ReduceCore::outputMetaOf(p.getString("keep", "czyx"), in);
            }
            StepOutput run(const StepInput& in, const ParamSet& p, const StepContext& ctx) const override {
                return ReduceCore::runOf(p.getString("keep", "czyx"), p.getString("reduction", "mean"), in, ctx);
            }

        private:
            OpInfo info_;
        };

        class MaxProjectionOperation final : public Operation {
        public:
            MaxProjectionOperation() {
                info_.kind = "maxproj";
                info_.name = "Max projection";
                info_.group = "Reduce";
                info_.kindLabel = "EINSUM";
                info_.defaultCache = CachePolicy::Memory;
                info_.helpPage = "einsum";
                info_.params = {choiceParam("axis", "Axis", {"z", "t", "c"}, "z")};
            }
            const OpInfo& info() const noexcept override { return info_; }
            std::string keep(const ParamSet& p) const {
                std::string k = kAxes;
                k.erase(std::remove(k.begin(), k.end(), p.getString("axis", "z")[0]), k.end());
                return k;
            }
            std::string summary(const ParamSet& p, const DatasetMeta& in) const override {
                return ReduceCore::summaryOf(keep(p), "max", in);
            }
            DatasetMeta outputMeta(const ParamSet& p, const DatasetMeta& in) const override {
                return ReduceCore::outputMetaOf(keep(p), in);
            }
            StepOutput run(const StepInput& in, const ParamSet& p, const StepContext& ctx) const override {
                return ReduceCore::runOf(keep(p), "max", in, ctx);
            }

        private:
            OpInfo info_;
        };

        class MeanOverTimeOperation final : public Operation {
        public:
            MeanOverTimeOperation() {
                info_.kind = "meant";
                info_.name = "Mean over time";
                info_.group = "Reduce";
                info_.kindLabel = "EINSUM";
                info_.defaultCache = CachePolicy::Memory;
                info_.helpPage = "einsum";
                info_.params = {};
            }
            const OpInfo& info() const noexcept override { return info_; }
            std::string summary(const ParamSet&, const DatasetMeta& in) const override {
                return ReduceCore::summaryOf("czyx", "mean", in);
            }
            Validation validate(const ParamSet& p, const DatasetMeta& in) const override {
                Validation v = Operation::validate(p, in);
                if (in.dims.t <= 1) v.warnings.push_back("The input has a single time point; nothing to average.");
                return v;
            }
            DatasetMeta outputMeta(const ParamSet&, const DatasetMeta& in) const override {
                return ReduceCore::outputMetaOf("czyx", in);
            }
            StepOutput run(const StepInput& in, const ParamSet&, const StepContext& ctx) const override {
                return ReduceCore::runOf("czyx", "mean", in, ctx);
            }

        private:
            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeEinsumOperation() { return std::make_unique<EinsumOperation>(); }
    std::unique_ptr<Operation> makeMaxProjectionOperation() { return std::make_unique<MaxProjectionOperation>(); }
    std::unique_ptr<Operation> makeMeanOverTimeOperation() { return std::make_unique<MeanOverTimeOperation>(); }

} // namespace sirius::app
