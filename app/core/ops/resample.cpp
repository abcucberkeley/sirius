// Resample: a new voxel size per axis (0 keeps the axis).
#include "core/ops/builtin.hpp"

#include <cstdio>

#include <sirius/image_ops.hpp>

namespace sirius::app {

    namespace {

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
                const ResampleGeometry g = geometry(p, meta);
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
