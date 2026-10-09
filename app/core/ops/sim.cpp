// SIM reconstruction: one raw SIM acquisition per (c, t) volume -- the z axis
// holds angles x phases x planes sections -- reconstructed with the library's
// SimReconstructor through the ReconSession (which keeps the FFT plans and
// the OTF between volumes). Diagnostics are the spectra the design's
// "Raw spectrum / Separated bands / Wiener-filtered bands / Result spectrum"
// tabs show, plus the fitted pattern table.
#include "core/ops/common.hpp"
#include "core/ops/sim_params.hpp"
#include "core/ops/builtin.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <utility>
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

        // The keys a parameter file assigns -- lower-cased, the last segment
        // of a dotted key -- and whether the file is TOML. From file, a key the
        // file sets is the file's value, and a missing key is not the library
        // default but the stack's: the OTF's axial step, and since 2026-10-08
        // the pixel sizes too (buildParameters says why).
        struct ParameterFileKeys {
            bool toml = false;
            std::set<std::string> keys;
            // Only the key the loader actually reads for a quantity: dz_psf in
            // TOML (including a .toml file with no [table] header), zresPSF in
            // a cudasirecon file.
            bool has(const char* tomlKey, const char* legacyKey) const {
                return keys.count(toml ? tomlKey : legacyKey) > 0;
            }
        };

        ParameterFileKeys parameterFileKeys(const std::string& path) {
            ParameterFileKeys found;
            std::ifstream in(std::filesystem::u8path(path));
            if (!in) return found;
            found.toml = detectParameterFormat(path) == ParameterFormat::Toml;
            std::string all;
            std::string line;
            while (std::getline(in, line)) {
                const auto cut = line.find_first_of("#;");
                if (cut != std::string::npos) line.resize(cut);
                all += line;
                all.push_back('\n');
            }
            // Every assignment, not only the first '=' on the line, so a dotted
            // key and an inline table (pixels.dz_psf, pixels = { dz_psf = … })
            // count.
            for (std::size_t eq = all.find('='); eq != std::string::npos; eq = all.find('=', eq + 1)) {
                std::size_t end = eq;
                while (end > 0 && (all[end - 1] == ' ' || all[end - 1] == '\t')) --end;
                std::size_t begin = end;
                while (begin > 0) {
                    const unsigned char c = static_cast<unsigned char>(all[begin - 1]);
                    if (std::isalnum(c) || all[begin - 1] == '_' || all[begin - 1] == '.' || all[begin - 1] == '"' || all[begin - 1] == '\'') --begin;
                    else break;
                }
                std::string key;
                for (std::size_t i = begin; i < end; ++i) {
                    const char c = all[i];
                    if (c == '"' || c == '\'') continue;
                    key.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
                }
                const auto dot = key.rfind('.');
                if (dot != std::string::npos) key = key.substr(dot + 1);
                if (!key.empty()) found.keys.insert(key);
            }
            return found;
        }

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

        // The layout the raw frames are read through: the dataset's general
        // storage layout when it has one, else the step's angles x phases on
        // z in the order the parameters say -- what every stack was read as
        // before layouts existed, so those runs index exactly as they did.
        SimLayout stackLayout(const SIMParameters& p, const DatasetMeta& input) {
            if (input.sim.present && !input.sim.isShorthand()) return input.sim;
            return SimLayout::shorthand(p.ndirs, p.nphases, p.fast_si);
        }

        // The raw (sections, y, x) stack of real volume (c, t) as the
        // reconstructor takes it: read as stored when the frames already lie
        // on z in an order it knows, otherwise gathered frame by frame
        // through the layout into the order `stack` describes (angle -> z ->
        // phase), one tile at a time for a montage. A file volume is read
        // once however many frames it holds.
        Buffer<float> rawStack(const StepInput& input, const SimFrames& frames, const SimFrames& stack, Index c, Index t) {
            if (frames.libraryOrder()) return input.readVolume(c, t);
            Buffer<float> out(Shape{stack.sections(), frames.tileY, frames.tileX});
            std::map<std::pair<Index, Index>, Buffer<float>> volumes;
            const Index rowStride = input.meta.dims.x, planeStride = input.meta.dims.y * rowStride;
            for (Index a = 0; a < frames.angles; ++a)
                for (Index z = 0; z < frames.nz; ++z)
                    for (Index ph = 0; ph < frames.phases; ++ph) {
                        const SimFrames::Frame fr = frames.frameOf({a, ph, z, c, t});
                        auto it = volumes.find({fr.c, fr.t});
                        if (it == volumes.end()) it = volumes.emplace(std::make_pair(fr.c, fr.t), input.readVolume(fr.c, fr.t)).first;
                        const float* src = it->second.data() + fr.z * planeStride + fr.row * frames.tileY * rowStride + fr.col * frames.tileX;
                        float* dst = out.data() + stack.sectionIndex(a, ph, z) * frames.tileY * frames.tileX;
                        for (Index y = 0; y < frames.tileY; ++y) std::copy_n(src + y * rowStride, frames.tileX, dst + y * frames.tileX);
                    }
            return out;
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
                    pathParam("params_file", "Parameter file").visibleWhen("mode", {"From file"}).withFilter("Parameters (*.toml *.txt *.cfg);;All files (*)").withHelp("Used by the From file mode. The pixel sizes it sets (xyres, zres, zresPSF; pixels.dx, dy, dz, dz_psf) win over the dataset's; the dataset fills in what the file leaves out"),
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
                // a general storage layout says the order itself: the stack
                // is passed through in one of the two orders the reconstructor
                // reads, or gathered into angle -> z -> phase
                if (input.sim.present && !input.sim.isShorthand() && simLayoutProblem(input.sim, input.dims).empty())
                    p.fast_si = bindSimLayout(input.sim, input.dims).libraryOrder().value_or(false);
                // The pixel sizes: the file's where it sets them, the stack's
                // otherwise. A cudasirecon config's xyres / zres are the pixel
                // sizes it reconstructs a TIFF stack with, and the radial step
                // of a measured OTF is derived from xyres (loadOTF: dkr =
                // 1 / (xyres * (nkr - 1) * 2)), so a step driven by such a file
                // must use them whatever the dataset's calibration says -- a
                // plain TIFF has none, and sirius-cli was handed one in z, y, x
                // order on 2026-10-08: dx = 0.125 for a 0.08 um pixel stretched
                // the OTF's radial axis until its support no longer reached
                // the side bands, and every fit with the measured OTF ended in
                // "the overlap of orders 0 and 2 holds no signal" while the
                // theoretical OTF, which carries its own step, was unaffected.
                // Estimate and Manual have no file.
                const ParameterFileKeys file =
                    mode == kFromFile ? parameterFileKeys(params.getString("params_file")) : ParameterFileKeys{};
                if (!file.has("dx", "xyres")) p.dx = input.dx();
                if (!file.has("dy", "xyres")) p.dy = input.dy();
                if (!file.has("dz", "zres")) p.dz = input.dz();
                // 0 means "not set". From file, that keeps the file's OTF step
                // (and the stack's dz only when the file has none). Estimate
                // and Manual have no file, so 0 means the stack's dz.
                const double dzPsf = params.getDouble("dz_psf", 0.0);
                if (dzPsf > 0.0) p.dz_psf = dzPsf;
                else if (!file.has("dz_psf", "zrespsf")) p.dz_psf = input.dz();
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
                const auto differs = [](double a, double b) { return std::abs(a - b) > 1e-9 * std::max(std::abs(a), std::abs(b)); };
                if (mode == kFromFile && (differs(p.dx, input.dx()) || differs(p.dy, input.dy()) || differs(p.dz, input.dz()))) {
                    char buf[256];
                    std::snprintf(buf, sizeof buf,
                                  "The parameter file sets the pixel size %.4g × %.4g × %.4g µm; the dataset says %.4g × %.4g × %.4g µm. "
                                  "The step reconstructs with the file's pixel sizes where it sets them.",
                                  p.dx, p.dy, p.dz, input.dx(), input.dy(), input.dz());
                    v.warnings.push_back(buf);
                }
                if (mode == kManual) {
                    const std::vector<double> angles = params.getDoubleList("k0_angles");
                    if (static_cast<int>(angles.size()) < p.ndirs)
                        v.errors.push_back("Manual mode needs one pattern angle per direction (" +
                                           std::to_string(p.ndirs) + ").");
                }
                const std::string otf = params.getString("otf");
                if (!otf.empty() && !std::filesystem::exists(otf)) v.errors.push_back("OTF file not found: " + otf);
                // the frames through the layout: the extents have to divide the
                // axes (the message quotes the arithmetic), and a layout the
                // dataset states has to hold the angles and phases the step uses
                const SimLayout layout = stackLayout(p, input);
                const std::string problem = simLayoutProblem(layout, input.dims);
                if (!problem.empty()) v.errors.push_back(problem);
                const std::optional<SimFrames> frames = problem.empty() ? std::optional<SimFrames>(bindSimLayout(layout, input.dims)) : std::nullopt;
                const Index ny = frames ? frames->tileY : input.dims.y, nx = frames ? frames->tileX : input.dims.x;
                // the library's condition and the library's wording, so this,
                // ReconSession::validate() and SimReconstructor itself cannot
                // word one condition three ways again
                if (const std::string why = simImageSizeProblem(nx, ny, frames && frames->storage.montage()); !why.empty())
                    v.errors.push_back(why);
                // the library's wording again, because the Python mirror has
                // to refuse the same mismatch in the same words and links the
                // library, not this (the Python front's finding C)
                if (frames && !layout.isShorthand())
                    if (const std::string why = simLayoutCountsProblem(layout.text(), static_cast<int>(frames->angles),
                                                                       static_cast<int>(frames->phases), p.ndirs, p.nphases);
                        !why.empty())
                        v.errors.push_back(why);
                if (input.sim.present && layout.isShorthand())
                    if (const std::string note = simDeclaredCountsNote(static_cast<int>(input.sim.ndirs),
                                                                       static_cast<int>(input.sim.nphases), p.ndirs, p.nphases);
                        !note.empty())
                        v.warnings.push_back(note);
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
                // one result per real (c, t) volume, each tile's size enlarged:
                // through the layout when it binds, else as a z-packed stack would
                const Index perPlane = std::max<Index>(1, static_cast<Index>(p.ndirs) * p.nphases);
                Index nz = std::max<Index>(1, input.dims.z / perPlane), ny = input.dims.y, nx = input.dims.x;
                const SimLayout layout = stackLayout(p, input);
                if (simLayoutProblem(layout, input.dims).empty()) {
                    const SimFrames frames = bindSimLayout(layout, input.dims);
                    nz = std::max<Index>(1, frames.nz);
                    ny = frames.tileY;
                    nx = frames.tileX;
                    out.dims.c = frames.channels;
                    out.dims.t = frames.times;
                    if (out.dims.c != input.dims.c) out.normalizeChannels();
                }
                out.dims.z = nz * std::max(1, p.z_zoom);
                out.dims.y = static_cast<Index>(std::lround(ny * p.zoomfact));
                out.dims.x = static_cast<Index>(std::lround(nx * p.zoomfact));
                // the pixel the reconstruction assumed (From file: the file's),
                // not the dataset's claim
                out.voxelUm[0] = p.dx / p.zoomfact;
                out.voxelUm[1] = p.dy / p.zoomfact;
                out.voxelUm[2] = p.dz / std::max(1, p.z_zoom);
                out.sim = SimLayout{};
                out.acquisition = nz > 1 ? "3D-SIM reconstructed" : "2D-SIM reconstructed";
                out.sourceType = PixelType::Float32;
                return out;
            }

            StepOutput run(const StepInput& input, const ParamSet& params, const StepContext& ctx) const override {
                const Validation v = validate(params, input.meta);
                if (!v.ok()) throw std::runtime_error(v.firstError());
                const SIMParameters p = buildParameters(params, input.meta);
                // validate() passed, so the layout binds; `stack` describes the
                // (sections, y, x) stack handed to the reconstructor: the frames
                // as stored when they are in an order it reads, else the
                // angle -> z -> phase order rawStack gathers them into
                const SimFrames frames = bindSimLayout(stackLayout(p, input.meta), input.meta.dims);
                const SimFrames stack = frames.libraryOrder()
                                            ? frames
                                            : bindSimLayout(SimLayout::shorthand(p.ndirs, p.nphases, false),
                                                            Dims5{1, 1, frames.sections(), frames.tileY, frames.tileX});
                const Index nz = frames.nz;

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
                // Capturing the band spectra keeps two complex volumes covering
                // every direction and band -- gigabytes on a full-size stack --
                // so it is only done for small ones. When it is skipped the
                // reason goes in the warnings, which is what the SIM parameter
                // panel renders; a fact would be built and never shown, the
                // panel drawing only images, the fit table and the footer.
                const bool capture = frames.tileY * frames.tileX <= 512 * 512 && nz <= 64;
                std::string captureNote;
                if (!capture) {
                    const int bands = 2 * p.resolvedOrders() - 1;
                    const double bytes = 2.0 * p.ndirs * bands * static_cast<double>(nz) *
                                         static_cast<double>(frames.tileY) * static_cast<double>(frames.tileX) * 16.0;
                    char buf[256];
                    std::snprintf(buf, sizeof buf,
                                  "The separated and Wiener-filtered band spectra are kept only for stacks up to 512 × 512 "
                                  "and 64 z-cycles, so those two tabs are missing here: capturing them would hold about "
                                  "%.1f GB. Crop or bin the stack to see them.",
                                  bytes / (1024.0 * 1024.0 * 1024.0));
                    captureNote = buf;
                }

                // The diagnostics describe one volume, fixed up front: (c 0,
                // t 0). With "All GPUs" every volume that started before the
                // first one finished captured its band spectra -- gigabytes each,
                // at once -- and the panel showed whichever finished first.
                double seconds = 0.0;
                bool plansReused = false;
                std::mutex firstMu;
                // one reconstruction per real (c, t) volume: the angles on the
                // channel axis, say, are frames of one volume, not channels
                DatasetMeta volumes = input.meta;
                volumes.dims.c = frames.channels;
                volumes.dims.t = frames.times;
                forEachVolumeOnGpus(volumes, ctx, [&](Index c, Index t, Device volDevice) {
                    Buffer<float> raw = rawStack(input, frames, stack, c, t);
                    Buffer<double> rawD(raw.shape());
                    convert(raw, rawD);
                    const int slot = volDevice.isCuda() ? volDevice.index % nSessions : 0;
                    std::lock_guard<std::mutex> g(*sessionLocks[static_cast<std::size_t>(slot)]);
                    ReconSession& session = *sessions[static_cast<std::size_t>(slot)];
                    session.setRaw(std::move(rawD), input.meta.name);
                    const bool diagnosed = c == 0 && t == 0;
                    session.setCaptureDiagnostics(diagnosed && capture);
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
                    if (diagnosed) {
                        std::lock_guard<std::mutex> f(firstMu);
                        out.diagnostics = diagnostics(input, raw, r, p, stack, *result, params, captureNote);
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
            // `stack` indexes `raw`: the (sections, y, x) stack the reconstructor was given.
            Diagnostics diagnostics(const StepInput& input, const Buffer<float>& raw, const ReconResult& r,
                                    const SIMParameters& p, const SimFrames& stack, const Array5& result, const ParamSet& params,
                                    const std::string& captureNote = {}) const {
                Diagnostics d;
                d.kind = DiagnosticsKind::Sim;
                const Index nz = stack.nz;
                const Index ny = raw.dim(1), nx = raw.dim(2);
                const Index zMid = nz / 2;
                const std::vector<std::array<double, 2>> predicted = predictedK0(p, nz);
                const std::vector<FitRow> rows = summarizeFit(r.fit);
                const double support = otfSupportRadius(p);

                // --- Raw spectrum: one panel per direction (phase 0, middle z)
                DiagnosticTab rawTab{"Raw spectrum", {}};
                for (int dir = 0; dir < p.ndirs; ++dir) {
                    const Index s = stack.sectionIndex(dir, 0, zMid);
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
                            const float* plane = raw.data() + stack.sectionIndex(dir, ph, zMid) * ny * nx;
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
