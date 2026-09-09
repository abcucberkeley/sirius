// Deskew + rotate: shear the light-sheet stack by the stage step and,
// optionally, rotate so z is normal to the coverslip on an isotropic grid.
#include "core/ops/builtin.hpp"

#include <cmath>
#include <cstdio>

#include <sirius/image_ops.hpp>

namespace sirius::app {

    namespace {

        Interpolation interpolationOf(const std::string& s) {
            if (s == "cubic") return Interpolation::Cubic;
            if (s == "nearest") return Interpolation::Nearest;
            return Interpolation::Linear;
        }

        class DeskewOperation final : public Operation {
        public:
            DeskewOperation() {
                info_.kind = "deskew";
                info_.name = "Deskew + rotate";
                info_.group = "Geometry";
                info_.kindLabel = "GEOMETRY";
                info_.defaultCache = CachePolicy::Recompute;
                info_.separableOverT = true;
                info_.helpPage = "deskew";
                info_.params = {
                    doubleParam("sheet_angle", "Sheet angle", 31.8).range(1.0, 89.0, 0.1, 2).withUnit("°").withHelp("Angle between the light sheet and the coverslip (0 = the dataset's)"),
                    doubleParam("stage_step_um", "Stage step", 0.40).range(0.001, 100.0, 0.01, 3).withUnit("µm").withHelp("Stage travel between planes"),
                    boolParam("rotate_to_coverslip", "Rotate to coverslip", true),
                    choiceParam("interpolation", "Interpolation", {"linear", "cubic", "nearest"}, "linear"),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            double angleOf(const ParamSet& params, const DatasetMeta& input) const {
                const double a = params.getDouble("sheet_angle", 31.8);
                return a > 0.0 ? a : input.sheetAngleDeg;
            }

            ResampleGeometry geometry(const ParamSet& params, const DatasetMeta& input) const {
                return deskewGeometry(input.dims.z, input.dims.y, input.dims.x, input.dx(), input.dz(),
                                      angleOf(params, input), params.getDouble("stage_step_um", 0.40),
                                      params.getBool("rotate_to_coverslip", true));
            }

            std::string summary(const ParamSet& params, const DatasetMeta& input) const override {
                if (!input.lightSheet && input.dims.numel() > 1) return "not applicable to this dataset — skipped";
                char a[32], s[32];
                std::snprintf(a, sizeof a, "%.1f°", angleOf(params, input));
                std::snprintf(s, sizeof s, "dz %.2f µm", params.getDouble("stage_step_um", 0.40));
                return joinSummary({a, s, params.getBool("rotate_to_coverslip", true) ? "rotate to coverslip" : "shear only",
                                    params.getString("interpolation", "linear")});
            }

            Validation validate(const ParamSet& params, const DatasetMeta& input) const override {
                Validation v = Operation::validate(params, input);
                if (!input.lightSheet)
                    v.warnings.push_back("The dataset is not marked as light-sheet data (Load ▸ Light-sheet angle); deskew "
                                         "will shear it anyway.");
                if (input.dims.z < 2) v.errors.push_back("Deskew needs a z stack.");
                if (input.sim.present) v.warnings.push_back("The input is a raw SIM stack; reconstruct it first.");
                return v;
            }

            DatasetMeta outputMeta(const ParamSet& params, const DatasetMeta& input) const override {
                DatasetMeta out = input;
                if (input.dims.z < 2) return out;
                const ResampleGeometry g = geometry(params, input);
                out.dims.z = g.oz;
                out.dims.y = g.oy;
                out.dims.x = g.ox;
                out.voxelUm = {g.outVoxelUm[2], g.outVoxelUm[1], g.outVoxelUm[0]};
                out.lightSheet = false;
                out.sheetAngleDeg = 0.0;
                out.sourceType = PixelType::Float32;
                return out;
            }

            StepOutput run(const StepInput& input, const ParamSet& params, const StepContext& ctx) const override {
                const Validation v = validate(params, input.meta);
                if (!v.ok()) throw std::runtime_error(v.firstError());
                const DatasetMeta& meta = input.meta;
                const ResampleGeometry g = geometry(params, meta);
                const Interpolation interp = interpolationOf(params.getString("interpolation", "linear"));
                StepOutput out;
                out.meta = outputMeta(params, meta);
                auto result = allocateLike(out.meta);
                forEachVolume(meta, ctx, [&](Index c, Index t) {
                    Buffer<float> vol = input.readVolume(c, t);
                    resampleAffine(vol.data(), meta.dims.z, meta.dims.y, meta.dims.x, g.A, g.b,
                                   result->volume(c, t).data(), g.oz, g.oy, g.ox, interp, 0.0f);
                });
                out.array = result;
                out.ranOn = Backend::Cpu;
                out.note = summary(params, meta) + " · CPU";
                out.diagnostics = genericDiagnostics(input, out, summary(params, meta));
                // the design's generic panel shows the deskewed stack side-on
                const Dims5& od = out.meta.dims;
                if (od.z > 1) {
                    std::vector<float> xz(static_cast<std::size_t>(od.z * od.x));
                    for (Index z = 0; z < od.z; ++z)
                        std::copy_n(result->plane(0, 0, z) + (od.y / 2) * od.x, od.x, xz.data() + z * od.x);
                    const int img = out.diagnostics.addImage(thumbnail(xz.data(), od.z, od.x, 400, "Output · XZ", out.meta.shapeString()));
                    if (out.diagnostics.tabs.empty()) out.diagnostics.tabs.push_back({"Preview", {}});
                    out.diagnostics.tabs.back().images.push_back(img);
                }
                return out;
            }

        private:
            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeDeskewOperation() { return std::make_unique<DeskewOperation>(); }

} // namespace sirius::app
