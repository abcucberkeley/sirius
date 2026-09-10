// Deconvolve: Richardson-Lucy (optionally TV-regularised) per (c, t) volume
// with a measured PSF TIFF or a theoretical Gaussian PSF.
#include "core/ops/builtin.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>

#include <sirius/deconvolution.hpp>
#include <sirius/tiff_io.hpp>

namespace sirius::app {

    namespace {

        class DeconvolveOperation final : public Operation {
        public:
            DeconvolveOperation() {
                info_.kind = "decon";
                info_.name = "Deconvolve";
                info_.group = "Reconstruct";
                info_.kindLabel = "RECONSTRUCT";
                info_.diagnostics = DiagnosticsKind::Deconvolve;
                info_.defaultCache = CachePolicy::Disk;
                info_.separableOverT = true;
                // The library serves a CUDA device from the host (src/deconvolution.cpp):
                // advertising a GPU path would only show "GPU" in the UI for a CPU run.
                info_.hasGpuPath = false;
                info_.helpPage = "decon";
                info_.params = {
                    choiceParam("algorithm", "Algorithm", {"Richardson–Lucy"}, "Richardson–Lucy"),
                    intParam("iterations", "Iterations", 20).range(1, 500),
                    pathParam("psf", "PSF").withFilter("PSF (*.tif *.tiff);;All files (*)").withHelp("Measured bead PSF (z, y, x) at the data's voxel size; empty = theoretical Gaussian"),
                    doubleParam("tv_lambda", "TV regularisation", 0.002).range(0.0, 0.1, 0.0005, 4).withHelp("Total-variation weight; 0 = plain Richardson–Lucy"),
                    doubleParam("stop_rel_change", "Stop below Δ", 1e-4).range(0.0, 1.0, 1e-5, 6).withHelp("Stop when the relative change per iteration drops below this (0 = never)"),
                    doubleParam("na", "NA", 1.4).range(0.1, 2.0, 0.01, 2).withHelp("Theoretical PSF only"),
                    doubleParam("wavelength_nm", "Emission λ", 510.0).range(300.0, 1000.0, 1.0, 0).withUnit("nm"),
                    doubleParam("nimm", "Immersion index", 1.515).range(1.0, 2.0, 0.001, 3),
                    intParam("psf_size", "Theoretical PSF size", 33).range(5, 257).withUnit("px").withHelp("Lateral extent of the theoretical PSF (odd)").asAdvanced(),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            std::string summary(const ParamSet& params, const DatasetMeta&) const override {
                char tv[32];
                std::snprintf(tv, sizeof tv, "TV %g", params.getDouble("tv_lambda"));
                const std::string psf = params.getString("psf");
                return joinSummary({"Richardson–Lucy", std::to_string(params.getInt("iterations")) + " iter",
                                    psf.empty() ? "theoretical PSF" : std::filesystem::path(psf).filename().string(), tv});
            }

            Validation validate(const ParamSet& params, const DatasetMeta& input) const override {
                Validation v = Operation::validate(params, input);
                const std::string psf = params.getString("psf");
                if (!psf.empty() && !std::filesystem::exists(psf)) v.errors.push_back("PSF file not found: " + psf);
                if (input.rgb) v.errors.push_back("Deconvolve needs intensity channels, not an RGB merge.");
                if (input.sim.present) v.warnings.push_back("The input is a raw SIM stack; reconstruct it first.");
                return v;
            }

            StepOutput run(const StepInput& input, const ParamSet& params, const StepContext& ctx) const override {
                const Validation v = validate(params, input.meta);
                if (!v.ok()) throw std::runtime_error(v.firstError());
                const DatasetMeta& meta = input.meta;
                StepOutput out;
                out.meta = meta;
                out.meta.sourceType = PixelType::Float32;
                auto result = allocateLike(meta);

                Buffer<float> psf = loadPsf(params, meta);
                DeconvolutionOptions options;
                options.iterations = static_cast<int>(params.getInt("iterations", 20));
                options.tvLambda = params.getDouble("tv_lambda", 0.0);
                options.stopRelativeChange = params.getDouble("stop_rel_change", 0.0);
                if (ctx.backend == Backend::Cuda && ctx.device.isCuda() && cudaAvailable()) options.device = ctx.device;

                DeconvolutionResult firstResult;
                bool first = true;
                bool gpu = false;
                double seconds = 0.0;
                std::vector<float> inputMid;   // middle plane of the first volume, for the residual panel
                forEachVolume(meta, ctx, [&](Index c, Index t) {
                    Buffer<float> vol = input.readVolume(c, t);
                    if (first) {
                        inputMid.assign(vol.data() + (meta.dims.z / 2) * meta.dims.planeSize(),
                                        vol.data() + (meta.dims.z / 2 + 1) * meta.dims.planeSize());
                    }
                    DeconvolutionOptions o = options;
                    const Index total = meta.dims.c * meta.dims.t;
                    const Index done = t * meta.dims.c + c;
                    // onIteration reports progress only: a false return there is
                    // a successful early stop (stoppedEarly, "converged"), which
                    // is not what a cancel means. Cancellation goes through
                    // `cancelled`, which the library polls once per iteration and
                    // between the transform stages within one, and which aborts
                    // by throwing without writing a partial volume back.
                    o.onIteration = [&, done, total](int iter, double rel) {
                        char msg[64];
                        std::snprintf(msg, sizeof msg, "iteration %d · Δ %.2e", iter, rel);
                        ctx.report((static_cast<double>(done) + static_cast<double>(iter) / options.iterations) /
                                       static_cast<double>(total),
                                   msg);
                        return true;
                    };
                    o.cancelled = [&ctx] { return ctx.isCancelled(); };
                    const auto t0 = std::chrono::steady_clock::now();
                    DeconvolutionResult r = richardsonLucy(vol.view(), psf.view(), o);
                    seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    ctx.throwIfCancelled();
                    gpu = gpu || r.ranOnGpu;
                    copy(vol, result->volume(c, t));
                    if (first) {
                        firstResult = std::move(r);
                        first = false;
                    }
                });
                out.array = result;
                out.ranOn = gpu ? Backend::Cuda : Backend::Cpu;
                out.seconds = seconds;
                char note[128];
                std::snprintf(note, sizeof note, "%.1f s · %d iterations%s · %s", seconds, firstResult.iterations,
                              firstResult.stoppedEarly ? " (converged)" : "", gpu ? "CUDA" : "CPU");
                out.note = note;
                out.diagnostics = diagnostics(meta, psf, firstResult, inputMid, *result, params);
                return out;
            }

        private:
            Buffer<float> loadPsf(const ParamSet& params, const DatasetMeta& meta) const {
                const std::string path = params.getString("psf");
                if (!path.empty()) {
                    TiffFile file(path);
                    TiffReadOptions opts;
                    return file.readStack<float>(opts);
                }
                Index n = params.getInt("psf_size", 33);
                if (n % 2 == 0) ++n;
                n = std::max<Index>(5, n);
                const Index pz = meta.dims.z > 1 ? std::min<Index>(n, meta.dims.z | 1) : 1;
                return gaussianPsf(pz, std::min(n, meta.dims.y | 1), std::min(n, meta.dims.x | 1), meta.dz(), meta.dx(),
                                   params.getDouble("na", 1.4), params.getDouble("wavelength_nm", 510.0),
                                   params.getDouble("nimm", 1.515));
            }

            Diagnostics diagnostics(const DatasetMeta& meta, const Buffer<float>& psf, const DeconvolutionResult& r,
                                    const std::vector<float>& inputMid, const Array5& result,
                                    const ParamSet& params) const {
                Diagnostics d;
                d.kind = DiagnosticsKind::Deconvolve;
                d.summary = summary(params, meta);
                DiagnosticCurve curve;
                curve.title = "Convergence · relative change per iteration";
                for (std::size_t i = 0; i < r.relativeChange.size(); ++i) {
                    curve.x.push_back(static_cast<double>(i + 1));
                    curve.y.push_back(r.relativeChange[i]);
                }
                curve.logY = true;
                curve.leftLabel = "iter 1";
                curve.rightLabel = "iter " + std::to_string(params.getInt("iterations", 20));
                if (!r.relativeChange.empty()) {
                    char mid[64];
                    std::snprintf(mid, sizeof mid, "stop at %d · Δ %.1e", r.iterations, r.relativeChange.back());
                    curve.midLabel = mid;
                    curve.stopX = static_cast<double>(r.iterations);
                }
                d.curves.push_back(std::move(curve));

                DiagnosticTab tab{"Convergence", {}};
                // PSF · XZ: central xz slice
                if (psf.rank() == 3 && psf.dim(0) > 0) {
                    const Index pz = psf.dim(0), py = psf.dim(1), px = psf.dim(2);
                    DiagnosticImage img;
                    img.title = "PSF · XZ";
                    img.rows = pz;
                    img.cols = px;
                    img.values.resize(static_cast<std::size_t>(pz * px));
                    for (Index z = 0; z < pz; ++z)
                        for (Index x = 0; x < px; ++x)
                            img.values[static_cast<std::size_t>(z * px + x)] = psf.data()[(z * py + py / 2) * px + x];
                    tab.images.push_back(d.addImage(std::move(img)));
                }
                if (!inputMid.empty()) {
                    const Dims5& od = result.dims();
                    std::vector<float> resid(inputMid.size());
                    const float* res = result.plane(0, 0, od.z / 2);
                    for (std::size_t i = 0; i < resid.size(); ++i) resid[i] = std::abs(inputMid[i] - res[i]);
                    tab.images.push_back(d.addImage(thumbnail(resid.data(), od.y, od.x, 400,
                                                              "|input − result| · iter " + std::to_string(r.iterations))));
                }
                d.tabs.push_back(std::move(tab));
                d.facts.push_back({"Iterations", std::to_string(r.iterations) + (r.stoppedEarly ? " (converged)" : "")});
                d.facts.push_back({"PSF", params.getString("psf").empty() ? "theoretical Gaussian" : "measured"});
                d.facts.push_back({"TV λ", formatNumber(params.getDouble("tv_lambda"), 4)});
                return d;
            }

            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeDeconvolveOperation() { return std::make_unique<DeconvolveOperation>(); }

} // namespace sirius::app
