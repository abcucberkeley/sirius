// SIM reconstruction: one raw SIM acquisition per (c, t) volume -- the z axis
// holds angles x phases x planes sections -- reconstructed with the library's
// SimReconstructor through the ReconSession (which keeps the FFT plans and
// the OTF between volumes). Diagnostics are the spectra the design's
// "Raw spectrum / Separated bands / Wiener-filtered bands / Result spectrum"
// tabs show, plus the fitted pattern table.
#include "core/ops/builtin.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <vector>

#include <sirius/constants.hpp>
#include <sirius/device.hpp>
#include <sirius/legacy_config.hpp>

#include "core/session.hpp"
#include "core/volume_ops.hpp"

namespace sirius::app {

    namespace {

        constexpr const char* kEstimate = "Estimate";
        constexpr const char* kManual = "Manual";
        constexpr const char* kFromFile = "From file";

        ApodizationType apodizationFromChoice(const std::string& s) {
            if (s == "Cosine") return ApodizationType::Cosine;
            if (s == "None") return ApodizationType::None;
            return ApodizationType::Triangle;
        }

        std::string degrees(double rad) {
            char buf[32];
            std::snprintf(buf, sizeof buf, "%.0f°", rad * 180.0 / kPi);
            return buf;
        }

        // Section index of (direction, z, phase) in the raw stack.
        Index sectionIndex(const SIMParameters& p, Index nz, Index d, Index z, Index phase) {
            if (p.fast_si) return (z * p.ndirs + d) * p.nphases + phase;
            return (d * nz + z) * p.nphases + phase;
        }

        // Pixel of a lateral frequency on a spectrum image whose full-size
        // plane had `fullCols` x `fullRows` pixels of dx x dy: the image is
        // a box-averaged version, so its frequency step is the full plane's.
        struct SpectrumPixels {
            Index rows = 0, cols = 0;
            double dkx = 0.0, dky = 0.0;
            std::array<double, 2> pixel(double kx, double ky) const noexcept {
                return {static_cast<double>(cols / 2) + kx / dkx, static_cast<double>(rows / 2) + ky / dky};
            }
            double radiusPx(double k) const noexcept { return k / dkx; }
        };
        SpectrumPixels pixelsOf(const DiagnosticImage& img, Index fullCols, Index fullRows, double dx, double dy) {
            SpectrumPixels s;
            s.rows = img.rows;
            s.cols = img.cols;
            s.dkx = 1.0 / (static_cast<double>(fullCols) * dx);
            s.dky = 1.0 / (static_cast<double>(fullRows) * dy);
            return s;
        }

        void addK0Marks(DiagnosticImage& img, const SpectrumPixels& px,
                        const std::vector<std::array<double, 2>>& k0, DiagnosticMark::Kind kind, double radiusPx,
                        bool accent, int onlyDirection = -1) {
            for (std::size_t d = 0; d < k0.size(); ++d) {
                if (onlyDirection >= 0 && static_cast<int>(d) != onlyDirection) continue;
                for (double sign : {1.0, -1.0}) {
                    DiagnosticMark m;
                    m.kind = kind;
                    const auto p = px.pixel(sign * k0[d][0], sign * k0[d][1]);
                    m.x = p[0];
                    m.y = p[1];
                    m.radius = radiusPx;
                    m.accent = accent;
                    img.marks.push_back(m);
                }
            }
        }

        // |band| plane of the middle kz of one direction, centered.
        DiagnosticImage bandImage(const SimDiagnostics& d, const Buffer<std::complex<double>>& bands, int dir,
                                  std::string title, std::string meta) {
            Buffer<double> vol = bandMagnitudeVolume(d, bands, dir, 1, BandSide::Plus);
            DiagnosticImage img;
            img.title = std::move(title);
            img.meta = std::move(meta);
            img.logScale = true;
            // keep it thumbnail sized
            const Index rows = vol.dim(1), cols = vol.dim(2);
            const Index f = std::max<Index>(1, (std::max(rows, cols) + 511) / 512);
            const Index r = rows / f, c = cols / f;
            img.rows = r;
            img.cols = c;
            img.values.resize(static_cast<std::size_t>(r * c));
            const double* plane = vol.data() + (vol.dim(0) / 2) * rows * cols;
            for (Index y = 0; y < r; ++y)
                for (Index x = 0; x < c; ++x) {
                    double acc = 0.0;
                    for (Index dy = 0; dy < f; ++dy)
                        for (Index dx = 0; dx < f; ++dx) acc += plane[(y * f + dy) * cols + x * f + dx];
                    img.values[static_cast<std::size_t>(y * c + x)] =
                        static_cast<float>(std::log10(acc / static_cast<double>(f * f) + 1e-12));
                }
            return img;
        }

        DiagnosticImage padSpectrum(const DiagnosticImage& src, Index rows, Index cols, std::string title,
                                    std::string meta) {
            DiagnosticImage out;
            out.title = std::move(title);
            out.meta = std::move(meta);
            out.logScale = src.logScale;
            out.rows = rows;
            out.cols = cols;
            float floor = std::numeric_limits<float>::infinity();
            for (float v : src.values) floor = std::min(floor, v);
            if (!std::isfinite(floor)) floor = 0.0f;
            out.values.assign(static_cast<std::size_t>(rows * cols), floor);
            const Index y0 = rows / 2 - src.rows / 2, x0 = cols / 2 - src.cols / 2;
            for (Index y = 0; y < src.rows; ++y)
                for (Index x = 0; x < src.cols; ++x) {
                    const Index oy = y + y0, ox = x + x0;
                    if (oy < 0 || ox < 0 || oy >= rows || ox >= cols) continue;
                    out.values[static_cast<std::size_t>(oy * cols + ox)] = src.values[static_cast<std::size_t>(y * src.cols + x)];
                }
            return out;
        }

        class SimOperation final : public Operation {
        public:
            SimOperation() {
                info_.kind = "sim";
                info_.name = "SIM reconstruction";
                info_.group = "Reconstruct";
                info_.kindLabel = "RECONSTRUCT";
                info_.diagnostics = DiagnosticsKind::Sim;
                info_.defaultCache = CachePolicy::Disk;
                info_.separableOverT = true;
                info_.hasGpuPath = true;
                info_.helpPage = "sim";
                info_.params = {
                    choiceParam("mode", "Pattern", {kEstimate, kManual, kFromFile}, kEstimate)
                        .withHelp("Estimate fits the pattern vectors from a start angle; Manual starts from the "
                                  "given angles; From file takes every parameter from a TOML / cudasirecon file."),
                    pathParam("params_file", "Parameter file").visibleWhen("mode", {"From file"}).withFilter("Parameters (*.toml *.txt *.cfg);;All files (*)").withHelp("Used by the From file mode"),
                    intParam("angles", "Angles", 3).range(1, 16).hiddenWhen("mode", {"From file"}),
                    intParam("phases", "Phases", 5).range(2, 32).hiddenWhen("mode", {"From file"}),
                    doubleParam("wiener", "Wiener", 0.001).range(1e-5, 1.0, 0.0005, 5).withHelp("Regularisation constant of the generalised Wiener filter").hiddenWhen("mode", {"From file"}),
                    choiceParam("apodization", "Apodization", {"Cosine", "Triangle", "None"}, "Cosine")
                        .withHelp("Window applied to the extended support")
                        .hiddenWhen("mode", {"From file"}),
                    pathParam("otf", "OTF").withFilter("OTF (*.tif *.tiff);;All files (*)").withHelp("Radially averaged OTF TIFF; empty = theoretical OTF from NA / wavelength"),
                    doubleParam("na", "NA", 1.4).range(0.1, 2.0, 0.01, 2).hiddenWhen("mode", {"From file"}),
                    doubleParam("nimm", "Immersion index", 1.515).range(1.0, 2.0, 0.001, 3).hiddenWhen("mode", {"From file"}),
                    doubleParam("wavelength_nm", "Emission λ", 510.0).range(300.0, 1000.0, 1.0, 0).withUnit("nm").hiddenWhen("mode", {"From file"}),
                    doubleParam("linespacing_um", "Line spacing", 0.2).range(0.01, 5.0, 0.001, 4).withUnit("µm").hiddenWhen("mode", {"From file"}),
                    doubleListParam("k0_angles", "Pattern angles", {}).withUnit("°").visibleWhen("mode", {"Manual"}).withHelp("Where the search starts for each direction (Manual mode), in the degrees the fit table reports; "
                                                                                                                              "the fit refines them from there"),
                    doubleParam("k0_start_angle", "Start angle", 0.0).range(-180.0, 180.0, 1.0, 2).withUnit("°").visibleWhen("mode", {"Estimate"}).withHelp("Angle of direction 0 (Estimate mode); the others follow at 180° / angles").asAdvanced(),
                    boolParam("suppress_zero_order", "Suppress zero-order", true)
                        .withHelp("Dampen the order-0 band where the side bands overlap it")
                        .hiddenWhen("mode", {"From file"}),
                    boolParam("bleach_correction", "Bleach correction across phases", true).hiddenWhen("mode", {"From file"}),
                    doubleParam("zoomfact", "Lateral zoom", 2.0).range(1.0, 4.0, 0.5, 1).withHelp("Output grid enlargement in x and y").asAdvanced().hiddenWhen("mode", {"From file"}),
                    intParam("z_zoom", "Axial zoom", 1).range(1, 4).asAdvanced().hiddenWhen("mode", {"From file"}),
                    intParam("orders", "Orders", 0).range(0, 8).withHelp("0 = phases / 2 + 1").asAdvanced().hiddenWhen("mode", {"From file"}),
                    doubleParam("dz_psf", "OTF axial step", 0.0).range(0.0, 10.0, 0.005, 4).withUnit("µm").withHelp("Axial step of the OTF file (0 = the stack's dz)").asAdvanced(),
                    doubleParam("otfcutoff", "OTF cutoff", 0.006).range(0.0, 1.0, 0.001, 4).asAdvanced().hiddenWhen("mode", {"From file"}),
                    doubleParam("background", "Camera background", 0.0).range(0.0, 1e6, 1.0, 1).asAdvanced().hiddenWhen("mode", {"From file"}),
                    choiceParam("apodize_input", "Input apodization", {"Triangle", "Cosine", "None"}, "Triangle").asAdvanced().hiddenWhen("mode", {"From file"}),
                    intParam("napodize", "Input border", 10).range(0, 512).withUnit("px").asAdvanced().hiddenWhen("mode", {"From file"}),
                    intParam("suppression_radius", "Suppression radius", 10).range(0, 512).withUnit("px").asAdvanced().hiddenWhen("mode", {"From file"}),
                    boolParam("suppress_singularities", "Suppress singularities", true).asAdvanced().hiddenWhen("mode", {"From file"}),
                    boolParam("no_kz0", "Skip kz = 0 plane", true).asAdvanced().hiddenWhen("mode", {"From file"}),
                    boolParam("filter_overlaps", "Filter overlaps", true).asAdvanced().hiddenWhen("mode", {"From file"}),
                    doubleParam("explodefact", "Explode factor", 1.0).range(0.5, 4.0, 0.1, 2).asAdvanced().hiddenWhen("mode", {"From file"}),
                    boolParam("equalizez", "Equalize z", false).asAdvanced().hiddenWhen("mode", {"From file"}),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            SIMParameters buildParameters(const ParamSet& params, const DatasetMeta& input) const {
                SIMParameters p;
                const std::string mode = params.getString("mode", kEstimate);
                if (mode == kFromFile) {
                    const std::string file = params.getString("params_file");
                    if (!file.empty()) p = loadParametersAuto(file);
                } else {
                    p.ndirs = static_cast<int>(params.getInt("angles", 3));
                    p.nphases = static_cast<int>(params.getInt("phases", 5));
                    const int orders = static_cast<int>(params.getInt("orders", 0));
                    p.norders = orders > 0 ? orders : p.nphases / 2 + 1;
                    p.wiener = params.getDouble("wiener", 0.001);
                    p.apodize_output = apodizationFromChoice(params.getString("apodization", "Cosine"));
                    p.apodize_input = apodizationFromChoice(params.getString("apodize_input", "Triangle"));
                    p.na = params.getDouble("na", 1.4);
                    p.nimm = params.getDouble("nimm", 1.515);
                    p.wavelength_nm = params.getDouble("wavelength_nm", 510.0);
                    p.linespacing_um = params.getDouble("linespacing_um", 0.2);
                    // the form and the fit table both speak degrees; the
                    // library wants radians, so the conversion happens once, here
                    p.k0_start_angle = params.getDouble("k0_start_angle", 0.0) * kPi / 180.0;
                    if (mode == kManual) {
                        std::vector<double> angles = params.getDoubleList("k0_angles");
                        for (double& a : angles) a *= kPi / 180.0;
                        if (!angles.empty()) p.k0_angles = angles;
                    }
                    p.dampen_order0 = params.getBool("suppress_zero_order", true);
                    p.do_rescale = params.getBool("bleach_correction", true);
                    p.zoomfact = params.getDouble("zoomfact", 2.0);
                    p.z_zoom = static_cast<int>(params.getInt("z_zoom", 1));
                    p.otfcutoff = params.getDouble("otfcutoff", 0.006);
                    p.background = params.getDouble("background", 0.0);
                    p.napodize = static_cast<int>(params.getInt("napodize", 10));
                    p.suppression_radius = static_cast<int>(params.getInt("suppression_radius", 10));
                    p.suppress_singularities = params.getBool("suppress_singularities", true);
                    p.no_kz0 = params.getBool("no_kz0", true);
                    p.filter_overlaps = params.getBool("filter_overlaps", true);
                    p.explodefact = params.getDouble("explodefact", 1.0);
                    p.equalizez = params.getBool("equalizez", false);
                    p.fast_si = input.sim.present && input.sim.fastSi;
                }
                p.dx = input.dx();
                p.dy = input.dy();
                p.dz = input.dz();
                const double dzPsf = params.getDouble("dz_psf", 0.0);
                p.dz_psf = dzPsf > 0.0 ? dzPsf : input.dz();
                return p;
            }

            std::string summary(const ParamSet& params, const DatasetMeta& input) const override {
                SIMParameters p;
                try {
                    p = buildParameters(params, input);
                } catch (const std::exception&) {
                    return "parameter file cannot be read";
                }
                char w[32];
                std::snprintf(w, sizeof w, "Wiener %g", p.wiener);
                return joinSummary({std::to_string(p.ndirs) + " angles", std::to_string(p.nphases) + " phases", w,
                                    params.getString("otf").empty() ? "theoretical OTF" : "measured OTF"});
            }

            Validation validate(const ParamSet& params, const DatasetMeta& input) const override {
                Validation v = Operation::validate(params, input);
                if (!v.ok()) return v;
                SIMParameters p;
                try {
                    p = buildParameters(params, input);
                    p.validate();
                } catch (const std::exception& e) {
                    v.errors.push_back(std::string("Invalid SIM parameters: ") + e.what());
                    return v;
                }
                const std::string mode = params.getString("mode", kEstimate);
                if (mode == kFromFile && params.getString("params_file").empty())
                    v.errors.push_back("From file mode needs a parameter file.");
                if (mode == kManual) {
                    const std::vector<double> angles = params.getDoubleList("k0_angles");
                    if (static_cast<int>(angles.size()) < p.ndirs)
                        v.errors.push_back("Manual mode needs one pattern angle per direction (" +
                                           std::to_string(p.ndirs) + ").");
                }
                const std::string otf = params.getString("otf");
                if (!otf.empty() && !std::filesystem::exists(otf)) v.errors.push_back("OTF file not found: " + otf);
                const Index perPlane = static_cast<Index>(p.ndirs) * p.nphases;
                if (input.dims.z % perPlane != 0)
                    v.errors.push_back(std::to_string(input.dims.z) + " sections is not a multiple of angles × phases = " +
                                       std::to_string(perPlane) + ".");
                if (input.dims.x % 2 != 0 || input.dims.y % 2 != 0 || input.dims.x < 4 || input.dims.y < 4)
                    v.errors.push_back("Image size must be even and at least 4 × 4.");
                if (input.sim.present && (input.sim.ndirs != p.ndirs || input.sim.nphases != p.nphases))
                    v.warnings.push_back("The dataset declares " + std::to_string(input.sim.ndirs) + " angles × " +
                                         std::to_string(input.sim.nphases) + " phases; the step uses " +
                                         std::to_string(p.ndirs) + " × " + std::to_string(p.nphases) + ".");
                if (input.rgb) v.errors.push_back("SIM reconstruction needs raw channels, not an RGB merge.");
                return v;
            }

            DatasetMeta outputMeta(const ParamSet& params, const DatasetMeta& input) const override {
                DatasetMeta out = input;
                SIMParameters p;
                try {
                    p = buildParameters(params, input);
                } catch (const std::exception&) {
                    return out;
                }
                const Index perPlane = std::max<Index>(1, static_cast<Index>(p.ndirs) * p.nphases);
                const Index nz = std::max<Index>(1, input.dims.z / perPlane);
                out.dims.z = nz * std::max(1, p.z_zoom);
                out.dims.y = static_cast<Index>(std::lround(input.dims.y * p.zoomfact));
                out.dims.x = static_cast<Index>(std::lround(input.dims.x * p.zoomfact));
                out.voxelUm[0] = input.dx() / p.zoomfact;
                out.voxelUm[1] = input.dy() / p.zoomfact;
                out.voxelUm[2] = input.dz() / std::max(1, p.z_zoom);
                out.sim = SimLayout{};
                out.acquisition = nz > 1 ? "3D-SIM reconstructed" : "2D-SIM reconstructed";
                out.sourceType = PixelType::Float32;
                return out;
            }

            StepOutput run(const StepInput& input, const ParamSet& params, const StepContext& ctx) const override {
                const Validation v = validate(params, input.meta);
                if (!v.ok()) throw std::runtime_error(v.firstError());
                const SIMParameters p = buildParameters(params, input.meta);
                const Index perPlane = static_cast<Index>(p.ndirs) * p.nphases;
                const Index nz = input.meta.dims.z / perPlane;

                StepOutput out;
                out.meta = outputMeta(params, input.meta);
                auto result = allocateLike(out.meta);

                Device device = Device::cpu();
                if (ctx.backend == Backend::Cuda && ctx.device.isCuda() && cudaAvailable()) device = ctx.device;
                out.ranOn = (device.isCuda() || ctx.allCudaDevices()) ? Backend::Cuda : Backend::Cpu;

                const int nSessions = ctx.allCudaDevices() ? std::max(1, cudaDeviceCount()) : 1;
                std::vector<std::unique_ptr<ReconSession>> sessions;
                std::vector<std::unique_ptr<std::mutex>> sessionLocks;
                sessions.reserve(static_cast<std::size_t>(nSessions));
                for (int i = 0; i < nSessions; ++i) {
                    sessions.push_back(std::make_unique<ReconSession>());
                    sessions.back()->setParameters(p);
                    sessions.back()->setOtfPath(params.getString("otf"));
                    sessionLocks.push_back(std::make_unique<std::mutex>());
                }
                const Index sections = input.meta.dims.z;
                // Capturing the band spectra keeps two complex volumes covering
                // every direction and band -- gigabytes on a full-size stack --
                // so it is only done for small ones. When it is skipped the
                // reason goes in the warnings, which is what the SIM parameter
                // panel renders; a fact would be built and never shown, the
                // panel drawing only images, the fit table and the footer.
                const bool capture = input.meta.dims.y * input.meta.dims.x <= 512 * 512 && sections <= 64 * perPlane;
                std::string captureNote;
                if (!capture) {
                    const int bands = 2 * p.resolvedOrders() - 1;
                    const double bytes = 2.0 * p.ndirs * bands * static_cast<double>(nz) *
                                         static_cast<double>(input.meta.dims.y) * static_cast<double>(input.meta.dims.x) * 16.0;
                    char buf[256];
                    std::snprintf(buf, sizeof buf,
                                  "The separated and Wiener-filtered band spectra are kept only for stacks up to 512 × 512 "
                                  "and 64 z-cycles, so those two tabs are missing here: capturing them would hold about "
                                  "%.1f GB. Crop or bin the stack to see them.",
                                  bytes / (1024.0 * 1024.0 * 1024.0));
                    captureNote = buf;
                }

                double seconds = 0.0;
                bool plansReused = false;
                bool first = true;
                std::mutex firstMu;
                forEachVolumeOnGpus(input.meta, ctx, [&](Index c, Index t, Device volDevice) {
                    Buffer<float> raw = input.readVolume(c, t);
                    Buffer<double> rawD(raw.shape());
                    convert(raw, rawD);
                    const int slot = volDevice.isCuda() ? volDevice.index % nSessions : 0;
                    std::lock_guard<std::mutex> g(*sessionLocks[static_cast<std::size_t>(slot)]);
                    ReconSession& session = *sessions[static_cast<std::size_t>(slot)];
                    session.setRaw(std::move(rawD), input.meta.name);
                    bool captureThis = false;
                    {
                        std::lock_guard<std::mutex> f(firstMu);
                        captureThis = first && capture;
                    }
                    session.setCaptureDiagnostics(captureThis);
                    ReconResult r = session.reconstruct(volDevice.isCuda() ? volDevice : device, PlanRigor::Measure,
                                                        [&ctx] { return ctx.isCancelled(); });
                    ctx.throwIfCancelled();
                    {
                        std::lock_guard<std::mutex> f(firstMu);
                        seconds += r.seconds;
                        plansReused = plansReused || r.plansReused;
                    }
                    if (r.volume.shape() != Shape{out.meta.dims.z, out.meta.dims.y, out.meta.dims.x})
                        throw std::runtime_error("SIM: unexpected output shape " + r.volume.shape().toString());
                    convert(r.volume, result->volume(c, t));
                    ctx.throwIfCancelled();
                    std::lock_guard<std::mutex> f(firstMu);
                    if (first) {
                        out.diagnostics = diagnostics(input, raw, r, p, nz, *result, params, captureNote);
                        first = false;
                    }
                });
                out.array = result;
                const std::string where = ctx.allCudaDevices()
                                              ? ("all " + std::to_string(cudaDeviceCount()) + " GPUs")
                                              : toString(device);
                char note[128];
                std::snprintf(note, sizeof note, "%.1f s · %s · %s · plans %s", seconds, where.c_str(),
                              params.getString("otf").empty() ? "theoretical OTF" : "measured OTF",
                              plansReused ? "reused" : "built");
                out.note = note;
                out.seconds = seconds;
                return out;
            }

        private:
            Diagnostics diagnostics(const StepInput& input, const Buffer<float>& raw, const ReconResult& r,
                                    const SIMParameters& p, Index nz, const Array5& result, const ParamSet& params,
                                    const std::string& captureNote = {}) const {
                Diagnostics d;
                d.kind = DiagnosticsKind::Sim;
                const Index ny = raw.dim(1), nx = raw.dim(2);
                const Index zMid = nz / 2;
                const std::vector<std::array<double, 2>> predicted = predictedK0(p, nz);
                const std::vector<FitRow> rows = summarizeFit(r.fit);
                const double support = otfSupportRadius(p);

                // --- Raw spectrum: one panel per direction (phase 0, middle z)
                DiagnosticTab rawTab{"Raw spectrum", {}};
                for (int dir = 0; dir < p.ndirs; ++dir) {
                    const Index s = sectionIndex(p, nz, dir, zMid, 0);
                    const float* plane = raw.data() + s * ny * nx;
                    const double angle = dir < static_cast<int>(predicted.size())
                                             ? std::atan2(predicted[static_cast<std::size_t>(dir)][1], predicted[static_cast<std::size_t>(dir)][0])
                                             : 0.0;
                    DiagnosticImage img = spectrumImage(plane, ny, nx, "Raw FFT · angle " + std::to_string(dir + 1), degrees(angle));
                    if (img.rows > 0) {
                        const SpectrumPixels px = pixelsOf(img, nx, ny, p.dx, p.dy);
                        addK0Marks(img, px, predicted, DiagnosticMark::Kind::Cross, 0.0, true, dir);
                        DiagnosticMark ring;
                        ring.kind = DiagnosticMark::Kind::Ring;
                        const auto c = px.pixel(0.0, 0.0);
                        ring.x = c[0];
                        ring.y = c[1];
                        ring.radius = px.radiusPx(support);
                        ring.accent = false;
                        img.marks.push_back(ring);
                    }
                    rawTab.images.push_back(d.addImage(std::move(img)));
                }
                d.tabs.push_back(std::move(rawTab));

                // --- Separated bands / Wiener-filtered bands (captured spectra)
                if (r.diagnostics.captured) {
                    DiagnosticTab sepTab{"Separated bands", {}};
                    // Named for what is drawn: one band per direction after the
                    // Wiener filter, at the middle kz. It is not the assembled
                    // Fourier mosaic, and calling it "stitched" said it was.
                    DiagnosticTab filtTab{"Wiener-filtered bands", {}};
                    for (int dir = 0; dir < r.diagnostics.ndirs; ++dir) {
                        const std::string angle = dir < static_cast<int>(rows.size()) ? degrees(rows[static_cast<std::size_t>(dir)].angleDeg * kPi / 180.0) : "";
                        try {
                            DiagnosticImage sep = bandImage(r.diagnostics, r.diagnostics.separated, dir,
                                                            "Order 1 · angle " + std::to_string(dir + 1), angle);
                            const SpectrumPixels px = pixelsOf(sep, nx, ny, p.dx, p.dy);
                            addK0Marks(sep, px, r.fit.k0, DiagnosticMark::Kind::Cross, 0.0, true, dir);
                            sepTab.images.push_back(d.addImage(std::move(sep)));
                            DiagnosticImage filt = bandImage(r.diagnostics, r.diagnostics.filtered, dir,
                                                             "Filtered order 1 · angle " + std::to_string(dir + 1), angle);
                            const SpectrumPixels pxf = pixelsOf(filt, nx, ny, p.dx, p.dy);
                            addK0Marks(filt, pxf, r.fit.k0, DiagnosticMark::Kind::Ring, pxf.radiusPx(support), true, dir);
                            filtTab.images.push_back(d.addImage(std::move(filt)));
                        } catch (const std::exception&) {
                            // a missing band panel is not worth failing the run
                        }
                    }
                    if (!sepTab.images.empty()) d.tabs.push_back(std::move(sepTab));
                    if (!filtTab.images.empty()) d.tabs.push_back(std::move(filtTab));
                }
                if (!captureNote.empty()) d.warnings.push_back(captureNote);

                // --- Result spectrum: widefield vs SIM vs difference
                {
                    std::vector<float> wide(static_cast<std::size_t>(ny * nx), 0.0f);
                    Index n = 0;
                    for (int dir = 0; dir < p.ndirs; ++dir)
                        for (int ph = 0; ph < p.nphases; ++ph, ++n) {
                            const float* plane = raw.data() + sectionIndex(p, nz, dir, zMid, ph) * ny * nx;
                            for (Index i = 0; i < ny * nx; ++i) wide[static_cast<std::size_t>(i)] += plane[i];
                        }
                    for (float& v : wide) v /= static_cast<float>(std::max<Index>(n, 1));
                    const Dims5& od = result.dims();
                    DiagnosticImage sim = spectrumImage(result.plane(0, 0, od.z / 2), od.y, od.x, "SIM result", "");
                    DiagnosticImage wf = spectrumImage(wide.data(), ny, nx, "Widefield", "1.0×");
                    double gain = 1.0;
                    double kmax = 0.0;
                    for (const auto& k : r.fit.k0) kmax = std::max(kmax, std::hypot(k[0], k[1]));
                    if (support > 0.0) gain = (support + (p.resolvedOrders() - 1) * kmax) / support;
                    char g[16];
                    std::snprintf(g, sizeof g, "%.1f×", gain);
                    sim.meta = g;
                    if (sim.rows > 0 && wf.rows > 0) {
                        const SpectrumPixels pxs = pixelsOf(sim, od.x, od.y, p.dx / p.zoomfact, p.dy / p.zoomfact);
                        const SpectrumPixels pxw = pixelsOf(wf, nx, ny, p.dx, p.dy);
                        DiagnosticMark ring;
                        ring.kind = DiagnosticMark::Kind::Ring;
                        ring.accent = true;
                        auto c = pxw.pixel(0.0, 0.0);
                        ring.x = c[0];
                        ring.y = c[1];
                        ring.radius = pxw.radiusPx(support);
                        wf.marks.push_back(ring);
                        c = pxs.pixel(0.0, 0.0);
                        ring.x = c[0];
                        ring.y = c[1];
                        ring.radius = pxs.radiusPx(support * gain);
                        sim.marks.push_back(ring);
                        DiagnosticImage padded = padSpectrum(wf, sim.rows, sim.cols, "Difference", "—");
                        for (std::size_t i = 0; i < padded.values.size(); ++i)
                            padded.values[i] = sim.values[i] - padded.values[i];
                        DiagnosticTab tab{"Result spectrum", {}};
                        tab.images.push_back(d.addImage(std::move(wf)));
                        tab.images.push_back(d.addImage(std::move(sim)));
                        tab.images.push_back(d.addImage(std::move(padded)));
                        d.tabs.push_back(std::move(tab));
                    }
                    // --- table
                    DiagnosticTable table;
                    table.caption = "Estimated parameters";
                    table.header = {"Angle", "k₀ (px⁻¹)", "Phase", "Mod."};
                    for (std::size_t i = 0; i < rows.size(); ++i) {
                        const FitRow& row = rows[i];
                        const double mag = std::hypot(row.kx, row.ky) * p.dx;
                        double phase = 0.0, mod = 0.0;
                        if (i < r.fit.amps.size() && r.fit.amps[i].size() > 1) {
                            phase = std::arg(r.fit.amps[i][1]);
                            mod = std::abs(r.fit.amps[i][1]);
                        }
                        table.rows.push_back({degrees(row.angleDeg * kPi / 180.0), formatNumber(mag, 4),
                                              formatNumber(phase, 2) + " rad", formatNumber(mod, 2)});
                        if (mod < 0.4) {
                            table.accentCells.emplace_back(static_cast<int>(i), 3);
                            char warn[160];
                            std::snprintf(warn, sizeof warn,
                                          "Modulation depth on angle %zu is low (%.2f). Consider re-estimating k₀ or raising the Wiener constant.",
                                          i + 1, mod);
                            d.warnings.push_back(warn);
                        }
                    }
                    d.table = std::move(table);
                    char footer[160];
                    std::snprintf(footer, sizeof footer, "Wiener %g · OTF %s · apodization %s · resolution gain ≈ %.1f×",
                                  p.wiener, params.getString("otf").empty() ? "theoretical" : "measured",
                                  p.apodize_output == ApodizationType::Cosine     ? "cosine"
                                  : p.apodize_output == ApodizationType::Triangle ? "triangle"
                                                                                  : "none",
                                  gain);
                    d.footer = footer;
                    // the empty tabs are in this dock, so the reason belongs
                    // here too: the warning goes to the parameter panel, which
                    // is a scroll away from where the gap is noticed
                    if (!captureNote.empty()) d.footer += " · band spectra not captured (see the step's warning)";
                    d.summary = summary(params, input.meta);
                }
                return d;
            }

            OpInfo info_;
        };

    } // namespace

    SIMParameters simParametersFromStep(const ParamSet& params, const DatasetMeta& input) {
        SimOperation op;
        return op.buildParameters(params, input);
    }

    std::unique_ptr<Operation> makeSimOperation() { return std::make_unique<SimOperation>(); }

} // namespace sirius::app
