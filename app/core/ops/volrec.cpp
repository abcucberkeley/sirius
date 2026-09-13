// Volume reconstruction: resample to isotropic voxels so the 3D viewer can
// ray-cast the stack through a transfer function; the rendering parameters
// (method, step size, opacity knee, isosurface level) travel in the output
// meta's diagnostics facts for the viewer.
#include "core/ops/builtin.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <sirius/image_ops.hpp>

namespace sirius::app {

    namespace {

        constexpr const char* kIsoSmallest = "Isotropic (smallest voxel)";
        constexpr const char* kIsoTarget = "Isotropic (target voxel)";
        constexpr const char* kKeep = "Keep";

        class VolumeOperation final : public Operation {
        public:
            VolumeOperation() {
                info_.kind = "volrec";
                info_.name = "Volume reconstruction";
                info_.group = "Reconstruct";
                info_.kindLabel = "VOLUME";
                info_.diagnostics = DiagnosticsKind::Volume;
                info_.defaultCache = CachePolicy::Memory;
                info_.separableOverT = true;
                info_.helpPage = "volrec";
                info_.params = {
                    choiceParam("method", "Method", {"Ray casting", "Maximum intensity", "Isosurface"}, "Ray casting"),
                    choiceParam("resample", "Resample to", {kIsoSmallest, kIsoTarget, kKeep}, kIsoSmallest),
                    doubleParam("target_voxel_um", "Target voxel", 0.104).range(0.001, 100.0, 0.001, 3).withUnit("µm"),
                    doubleParam("step_size", "Step size", 0.5).range(0.1, 4.0, 0.05, 2).withUnit("voxel").withHelp("Ray sampling distance; smaller is smoother and slower"),
                    doubleParam("opacity_lo", "Opacity ramp start", 0.2).range(0.0, 1.0, 0.01, 2).withHelp("Intensity (0..1 of the range) where the transfer function starts to become opaque"),
                    doubleParam("opacity_hi", "Opacity ramp end", 0.7).range(0.0, 1.0, 0.01, 2),
                    doubleParam("iso_level", "Isosurface level", 0.5).range(0.0, 1.0, 0.01, 2),
                    choiceParam("interpolation", "Interpolation", {"linear", "nearest"}, "linear").asAdvanced(),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            double targetVoxel(const ParamSet& params, const DatasetMeta& input) const {
                const std::string mode = params.getString("resample", kIsoSmallest);
                if (mode == kKeep) return 0.0;
                if (mode == kIsoTarget) return params.getDouble("target_voxel_um", 0.104);
                return std::min({input.dx(), input.dy(), input.dz()});
            }

            std::string summary(const ParamSet& params, const DatasetMeta& input) const override {
                const double v = targetVoxel(params, input);
                char buf[48];
                if (v > 0) std::snprintf(buf, sizeof buf, "isotropic %.3g µm", v);
                else std::snprintf(buf, sizeof buf, "native grid");
                std::string method = params.getString("method", "Ray casting");
                std::transform(method.begin(), method.end(), method.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                return joinSummary({buf, method});
            }

            Validation validate(const ParamSet& params, const DatasetMeta& input) const override {
                Validation v = Operation::validate(params, input);
                if (input.sim.present) v.warnings.push_back("The input is a raw SIM stack; reconstruct it first.");
                const double tv = targetVoxel(params, input);
                if (tv > 0) {
                    const ResampleGeometry g = resampleGeometry(input.dims.z, input.dims.y, input.dims.x, input.dz(),
                                                                input.dy(), input.dx(), tv, tv, tv);
                    if (g.oz * g.oy * g.ox * input.dims.c * input.dims.t > Index{1} << 31)
                        v.warnings.push_back("The isotropic volume exceeds 2 G voxels; consider a larger target voxel.");
                }
                return v;
            }

            DatasetMeta outputMeta(const ParamSet& params, const DatasetMeta& input) const override {
                DatasetMeta out = input;
                const double tv = targetVoxel(params, input);
                if (tv <= 0) return out;
                const ResampleGeometry g = resampleGeometry(input.dims.z, input.dims.y, input.dims.x, input.dz(),
                                                            input.dy(), input.dx(), tv, tv, tv);
                out.dims.z = g.oz;
                out.dims.y = g.oy;
                out.dims.x = g.ox;
                out.voxelUm = {g.outVoxelUm[2], g.outVoxelUm[1], g.outVoxelUm[0]};
                return out;
            }

            StepOutput run(const StepInput& input, const ParamSet& params, const StepContext& ctx) const override {
                const DatasetMeta& meta = input.meta;
                StepOutput out;
                out.meta = outputMeta(params, meta);
                const double tv = targetVoxel(params, meta);
                if (tv <= 0) {
                    out.array = input.materialize([&](double f, const std::string& m) { ctx.report(f, m); });
                    out.labels = input.labels ? input.labels->clone() : nullptr;
                    out.note = "native grid";
                } else {
                    const ResampleGeometry g = resampleGeometry(meta.dims.z, meta.dims.y, meta.dims.x, meta.dz(),
                                                                meta.dy(), meta.dx(), tv, tv, tv);
                    auto result = allocateLike(out.meta);
                    const Interpolation interp =
                        params.getString("interpolation", "linear") == "nearest" ? Interpolation::Nearest : Interpolation::Linear;
                    forEachVolume(meta, ctx, [&](Index c, Index t) {
                        Buffer<float> vol = input.readVolume(c, t);
                        resampleAffine(vol.data(), meta.dims.z, meta.dims.y, meta.dims.x, g.A, g.b,
                                       result->volume(c, t).data(), g.oz, g.oy, g.ox, interp, 0.0f);
                    });
                    out.array = result;
                    char note[96];
                    std::snprintf(note, sizeof note, "resampled to %.3g µm isotropic · %s", tv, "CPU");
                    out.note = note;
                }
                out.ranOn = Backend::Cpu;
                out.diagnostics = diagnostics(out, params);
                return out;
            }

        private:
            Diagnostics diagnostics(const StepOutput& out, const ParamSet& params) const {
                Diagnostics d;
                d.kind = DiagnosticsKind::Volume;
                d.summary = summary(params, out.meta);
                DiagnosticCurve tf;
                tf.title = "Transfer function";
                const double lo = params.getDouble("opacity_lo", 0.2), hi = std::max(params.getDouble("opacity_hi", 0.7), lo + 1e-6);
                for (int i = 0; i <= 40; ++i) {
                    const double x = i / 40.0;
                    double a = (x - lo) / (hi - lo);
                    a = std::clamp(a, 0.0, 1.0);
                    tf.x.push_back(x);
                    tf.y.push_back(a * a * (3 - 2 * a));   // smoothstep knee
                }
                tf.leftLabel = "0";
                tf.midLabel = "opacity vs intensity";
                tf.rightLabel = "1";
                d.curves.push_back(std::move(tf));
                char vox[48];
                std::snprintf(vox, sizeof vox, "%.3g µm", out.meta.dx());
                d.facts.push_back({"Method", params.getString("method", "Ray casting")});
                d.facts.push_back({"Isotropic voxel", vox});
                d.facts.push_back({"Step size", formatNumber(params.getDouble("step_size", 0.5), 2) + " voxel"});
                d.facts.push_back({"Frame time", "—"});
                if (out.array && !out.array->empty()) {
                    const Dims5& od = out.meta.dims;
                    const Index c = od.c / 2;
                    std::vector<float> mip(static_cast<std::size_t>(od.planeSize()), -std::numeric_limits<float>::infinity());
                    for (Index z = 0; z < od.z; ++z) {
                        const float* p = out.array->plane(c, 0, z);
                        for (Index i = 0; i < od.planeSize(); ++i)
                            mip[static_cast<std::size_t>(i)] = std::max(mip[static_cast<std::size_t>(i)], p[i]);
                    }
                    DiagnosticTab tab{"Rendering", {}};
                    tab.images.push_back(d.addImage(thumbnail(mip.data(), od.y, od.x, 400, "Isosurface preview", "MIP · z")));
                    d.tabs.push_back(std::move(tab));
                }
                return d;
            }

            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeVolumeOperation() { return std::make_unique<VolumeOperation>(); }

} // namespace sirius::app
