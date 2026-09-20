// Resample: a new voxel size per axis (0 keeps the axis).
#include "core/ops/common.hpp"
#include "core/ops/builtin.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <sirius/image_ops.hpp>

namespace sirius::app {

    namespace {

        // The step an output axis is sampled with: the voxel ratio, pulled
        // down by the rounding error that would put the last output centre
        // past the last input centre. The extent keeps that centre inside the
        // field (to its 1e-9 tolerance), but reached as resampleAffine reaches
        // it -- (n - 1) * step along z and y, n - 1 additions of the step
        // along x -- it could land a few ulps outside and read as fill: a
        // 64-plane stack at 0.3 um resampled to 0.1 um lost its last plane.
        // Both evaluations are held inside, and a real overshoot (more than
        // rounding) is left alone. workbench.py's _fitted_step is the same.
        double fittedStep(double step, Index samples, Index inputExtent) {
            if (samples <= 1 || inputExtent <= 1) return step;
            const double last = static_cast<double>(inputExtent - 1);
            auto reach = [&](double s) {
                double sum = 0.0;
                for (Index i = 1; i < samples; ++i) sum += s;
                return std::max(sum, static_cast<double>(samples - 1) * s);
            };
            for (int k = 0; k < 64; ++k) {
                const double over = reach(step) - last;
                if (over <= 0.0 || over > 1e-6) break;
                step = std::nextafter(step - over / static_cast<double>(samples - 1), 0.0);
            }
            return step;
        }

        class ResampleOperation final : public Operation {
        public:
            ResampleOperation() {
                info_.kind = "resample";
                info_.name = "Resample";
                info_.group = "Geometry";
                info_.kindLabel = "GEOMETRY";
                info_.defaultCache = CachePolicy::Recompute;
                info_.separableOverT = true;
                info_.helpPage = "resample";
                info_.params = {
                    doubleParam("voxel_x", "Voxel x", 0.0).range(0.0, 1000.0, 0.001, 4).withUnit("µm").withHelp("0 = keep"),
                    doubleParam("voxel_y", "Voxel y", 0.0).range(0.0, 1000.0, 0.001, 4).withUnit("µm").withHelp("0 = keep"),
                    doubleParam("voxel_z", "Voxel z", 0.0).range(0.0, 1000.0, 0.001, 4).withUnit("µm").withHelp("0 = keep"),
                    choiceParam("interpolation", "Interpolation", {"linear", "cubic", "nearest"}, "linear"),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            ResampleGeometry geometry(const ParamSet& p, const DatasetMeta& in) const {
                return resampleGeometry(in.dims.z, in.dims.y, in.dims.x, in.dz(), in.dy(), in.dx(), p.getDouble("voxel_z"),
                                        p.getDouble("voxel_y"), p.getDouble("voxel_x"));
            }

            std::string summary(const ParamSet& p, const DatasetMeta& in) const override {
                const ResampleGeometry g = geometry(p, in);
                char buf[96];
                std::snprintf(buf, sizeof buf, "%.3g × %.3g × %.3g µm", g.outVoxelUm[2], g.outVoxelUm[1], g.outVoxelUm[0]);
                return joinSummary({buf, p.getString("interpolation", "linear")});
            }

            DatasetMeta outputMeta(const ParamSet& p, const DatasetMeta& in) const override {
                DatasetMeta out = in;
                const ResampleGeometry g = geometry(p, in);
                out.dims.z = g.oz;
                out.dims.y = g.oy;
                out.dims.x = g.ox;
                out.voxelUm = {g.outVoxelUm[2], g.outVoxelUm[1], g.outVoxelUm[0]};
                if (g.oz != in.dims.z) out.sim = SimLayout{};
                out.sourceType = PixelType::Float32;
                return out;
            }

            StepOutput run(const StepInput& input, const ParamSet& p, const StepContext& ctx) const override {
                const DatasetMeta& meta = input.meta;
                ResampleGeometry g = geometry(p, meta);
                g.A[0] = fittedStep(g.A[0], g.oz, meta.dims.z);
                g.A[4] = fittedStep(g.A[4], g.oy, meta.dims.y);
                g.A[8] = fittedStep(g.A[8], g.ox, meta.dims.x);
                const std::string interpName = p.getString("interpolation", "linear");
                const Interpolation interp = interpName == "cubic"     ? Interpolation::Cubic
                                             : interpName == "nearest" ? Interpolation::Nearest
                                                                       : Interpolation::Linear;
                StepOutput out;
                out.meta = outputMeta(p, meta);
                auto result = allocateLike(out.meta);
                forEachVolume(meta, ctx, [&](Index c, Index t) {
                    Buffer<float> vol = input.readVolume(c, t);
                    resampleAffine(vol.data(), meta.dims.z, meta.dims.y, meta.dims.x, g.A, g.b,
                                   result->volume(c, t).data(), g.oz, g.oy, g.ox, interp, 0.0f);
                });
                out.array = result;
                out.ranOn = Backend::Cpu;
                out.note = summary(p, meta) + " · CPU";
                out.diagnostics = genericDiagnostics(input, out, summary(p, meta));
                return out;
            }

        private:
            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeResampleOperation() { return std::make_unique<ResampleOperation>(); }

} // namespace sirius::app
