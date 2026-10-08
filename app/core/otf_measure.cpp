#include "core/otf_measure.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>

#include "core/build_info.hpp"
#include "core/cancel.hpp"

namespace sirius::app {

    namespace {

        std::string num(double v, int digits = 6) {
            std::ostringstream os;
            os << std::setprecision(digits) << v;
            return os.str();
        }

        std::string pct(double fraction, int digits = 3) { return num(100.0 * fraction, digits) + "%"; }

        const char* scaleName(sirius::OtfMeasureScale s) noexcept {
            switch (s) {
                case sirius::OtfMeasureScale::Order0Dc: return "order0_dc";
                case sirius::OtfMeasureScale::MakeotfFixOrigin: return "makeotf_fixorigin";
                case sirius::OtfMeasureScale::AsMeasured: return "as_measured";
            }
            return "unknown";
        }

        const char* backgroundName(sirius::BackgroundEstimate b) noexcept {
            switch (b) {
                case sirius::BackgroundEstimate::BorderMean: return "border_mean";
                case sirius::BackgroundEstimate::DarkestFraction: return "darkest_fraction";
            }
            return "unknown";
        }

        const char* packingName(sirius::BeadPhasePacking p) noexcept {
            return p == sirius::BeadPhasePacking::PhaseFastest ? "phase_fastest" : "phase_slowest";
        }

        // "488" / "mito-gfp" reduced to what a file name may hold.
        std::string sanitise(const std::string& s) {
            std::string out;
            for (char c : s) {
                if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '-' || c == '_')
                    out.push_back(c);
                else if (c == ' ' || c == '.' || c == '/' || c == '\\')
                    out.push_back('_');
            }
            return out;
        }

        bool endsWith(const std::string& s, const std::string& tail) {
            return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
        }

        std::string lower(std::string s) {
            for (char& c : s)
                if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            return s;
        }

        // The path the table is written to: the request's, with .tif appended
        // when it names no TIFF suffix, so one place decides it and validate
        // and the run cannot disagree about which file is checked and written.
        std::filesystem::path tablePathOf(const std::string& path) {
            if (path.empty()) return {};
            const std::string low = lower(path);
            if (endsWith(low, ".tif") || endsWith(low, ".tiff")) return std::filesystem::path(path);
            return std::filesystem::path(path + ".tif");
        }

        // Which end of the z axis the phases sit on, as the storage layout
        // states it: the phase factor inside the z axis after the z factor is
        // phase fastest. nullopt when the layout puts no phase on z (the
        // angles on the channel axis, a montage of phases in the plane), where
        // the frames are gathered explicitly and no packing is involved.
        std::optional<sirius::BeadPhasePacking> packingOfStorage(const SimStorage& storage) {
            const std::vector<SimFactor>& z = storage.axes[2];
            int phaseAt = -1, zAt = -1;
            for (std::size_t i = 0; i < z.size(); ++i) {
                if (z[i].axis == SimAxis::Phase) phaseAt = static_cast<int>(i);
                if (z[i].axis == SimAxis::Z) zAt = static_cast<int>(i);
            }
            if (phaseAt < 0 || zAt < 0) return std::nullopt;
            return phaseAt > zAt ? sirius::BeadPhasePacking::PhaseFastest : sirius::BeadPhasePacking::PhaseSlowest;
        }

        // How the sections will be put in order, and the shape the library
        // will see. Computed from metadata alone, so validate and the run ask
        // the same question of the same function and cannot answer it
        // differently.
        struct GatherPlan {
            bool layout = false;
            SimFrames frames;
            int nphases = 1, nz = 1, sections = 0;
            Index ny = 0, nx = 0;
            Index angles = 1, channels = 1, times = 1;
            std::optional<sirius::BeadPhasePacking> layoutPacking;
            std::string layoutProblem;   // non-empty: the dataset's layout does not bind to these dims
        };

        GatherPlan planOf(const DatasetMeta& meta, const Dims5& dims, const OtfMeasureRequest& r) {
            GatherPlan p;
            if (meta.sim.present) {
                p.layoutProblem = simLayoutProblem(meta.sim, dims);
                if (p.layoutProblem.empty()) {
                    p.frames = bindSimLayout(meta.sim, dims);
                    p.layout = true;
                    p.nphases = static_cast<int>(p.frames.phases);
                    p.nz = static_cast<int>(p.frames.nz);
                    p.sections = p.nphases * p.nz;
                    p.ny = p.frames.tileY;
                    p.nx = p.frames.tileX;
                    p.angles = p.frames.angles;
                    p.channels = p.frames.channels;
                    p.times = p.frames.times;
                    p.layoutPacking = packingOfStorage(p.frames.storage);
                    return p;
                }
            }
            p.nphases = std::max(1, r.measure.nphases);
            p.sections = static_cast<int>(dims.z);
            p.nz = p.sections / p.nphases;   // truncating; a stack that does not divide is a problem below
            p.ny = dims.y;
            p.nx = dims.x;
            p.channels = dims.c;
            p.times = dims.t;
            return p;
        }

        // The library's options for this plan: where the layout decided the
        // order, the gather has already produced phase-fastest sections of its
        // phase count, so those two fields are the plan's rather than the
        // request's. Validation refuses a request that disagrees with the
        // layout, so this is never a silent override.
        sirius::OtfMeasureOptions measureOptionsOf(const OtfMeasureRequest& r, const GatherPlan& plan) {
            sirius::OtfMeasureOptions m = r.measure;
            if (plan.layout) {
                m.nphases = plan.nphases;
                m.packing = sirius::BeadPhasePacking::PhaseFastest;
            }
            return m;
        }

        // (sections, ny, nx) of double, as sirius::measureOTF takes it.
        Eigen::Tensor<double, 3, Eigen::RowMajor> gather(const Array5& array, const GatherPlan& plan,
                                                         const OtfMeasureRequest& r) {
            const Index ny = plan.ny, nx = plan.nx, rowStride = array.dims().x;
            Eigen::Tensor<double, 3, Eigen::RowMajor> stack(static_cast<Eigen::Index>(plan.sections),
                                                            static_cast<Eigen::Index>(ny),
                                                            static_cast<Eigen::Index>(nx));
            if (plan.layout) {
                for (int z = 0; z < plan.nz; ++z)
                    for (int ph = 0; ph < plan.nphases; ++ph) {
                        const SimFrames::Frame f =
                            plan.frames.frameOf({r.angle, static_cast<Index>(ph), static_cast<Index>(z), r.channel, r.time});
                        const float* src = array.plane(f.c, f.t, f.z);
                        const Index y0 = f.row * ny, x0 = f.col * nx;   // a montage tile's origin; (0, 0) without one
                        const Eigen::Index sec = static_cast<Eigen::Index>(z) * plan.nphases + ph;
                        for (Index y = 0; y < ny; ++y)
                            for (Index x = 0; x < nx; ++x)
                                stack(sec, static_cast<Eigen::Index>(y), static_cast<Eigen::Index>(x)) =
                                    static_cast<double>(src[(y0 + y) * rowStride + x0 + x]);
                    }
            } else {
                for (int s = 0; s < plan.sections; ++s) {
                    const float* src = array.plane(r.channel, r.time, static_cast<Index>(s));
                    for (Index y = 0; y < ny; ++y)
                        for (Index x = 0; x < nx; ++x)
                            stack(static_cast<Eigen::Index>(s), static_cast<Eigen::Index>(y), static_cast<Eigen::Index>(x)) =
                                static_cast<double>(src[y * rowStride + x]);
                }
            }
            return stack;
        }

        // The field as the detector saw it: the maximum over every section,
        // which is what a z-projected bandpass localises on, thumbnailed, with
        // one mark per candidate. Marks are in the thumbnail's own pixels
        // (diagnostic_cells.cpp scales them by the texture's size), so the
        // scale comes from the image thumbnail() returned rather than from a
        // second copy of its reduction factor.
        DiagnosticImage beadPreview(const Eigen::Tensor<double, 3, Eigen::RowMajor>& stack,
                                    const sirius::OtfMeasureResult& res, Index maxSide) {
            const Index sections = static_cast<Index>(stack.dimension(0));
            const Index ny = static_cast<Index>(stack.dimension(1)), nx = static_cast<Index>(stack.dimension(2));
            std::vector<float> proj(static_cast<std::size_t>(ny * nx), -std::numeric_limits<float>::infinity());
            for (Index s = 0; s < sections; ++s)
                for (Index y = 0; y < ny; ++y)
                    for (Index x = 0; x < nx; ++x) {
                        float& p = proj[static_cast<std::size_t>(y * nx + x)];
                        p = std::max(p, static_cast<float>(stack(static_cast<Eigen::Index>(s), static_cast<Eigen::Index>(y),
                                                                 static_cast<Eigen::Index>(x))));
                    }
            DiagnosticImage img = thumbnail(proj.data(), ny, nx, std::max<Index>(16, maxSide),
                                            "Bead field · maximum over the stack",
                                            std::to_string(nx) + " x " + std::to_string(ny) + " px · " +
                                                std::to_string(res.kept) + " of " + std::to_string(res.found) + " used");
            const double sx = nx > 0 && img.cols > 0 ? static_cast<double>(img.cols) / static_cast<double>(nx) : 1.0;
            const double sy = ny > 0 && img.rows > 0 ? static_cast<double>(img.rows) / static_cast<double>(ny) : 1.0;
            // A field may hold hundreds of candidates and a cell this size can
            // show a few dozen; the table below has every one of them.
            constexpr std::size_t kMaxMarks = 64;
            for (const sirius::BeadFit& b : res.beads) {
                if (img.marks.size() >= kMaxMarks) break;
                DiagnosticMark m;
                m.kind = b.kept ? DiagnosticMark::Kind::Circle : DiagnosticMark::Kind::Ring;
                m.x = b.x * sx;
                m.y = b.y * sy;
                m.radius = (b.sigmaX > 0.0 ? 3.0 * b.sigmaX : 4.0) * sx;
                m.accent = b.kept;
                if (!b.kept) m.text = sirius::beadRejectionName(b.rejection);
                img.marks.push_back(m);
            }
            return img;
        }

        DiagnosticImage orderImage(const sirius::OtfMeasureResult& res, int order) {
            const auto& d = res.otf.data();
            const Eigen::Index nkr = d.dimension(1), nz = d.dimension(2);
            DiagnosticImage img;
            img.title = "Order " + std::to_string(order) + " · |OTF|";
            img.meta = std::to_string(nkr) + " kr x " + std::to_string(nz) + " kz · kz centred · log10";
            img.logScale = true;
            if (order < 0 || order >= static_cast<int>(d.dimension(0)) || nkr <= 0 || nz <= 0) return img;
            img.rows = static_cast<Index>(nkr);
            img.cols = static_cast<Index>(nz);
            img.values.resize(static_cast<std::size_t>(nkr * nz));
            double peak = 0.0;
            for (Eigen::Index r = 0; r < nkr; ++r)
                for (Eigen::Index k = 0; k < nz; ++k) peak = std::max(peak, std::abs(d(order, r, k)));
            const double floorValue = peak > 0.0 ? peak * 1e-6 : 1.0;
            for (Eigen::Index r = 0; r < nkr; ++r)
                for (Eigen::Index c = 0; c < nz; ++c) {
                    // kz is stored DC first; the display puts DC in the middle
                    const Eigen::Index k = (c + nz / 2) % nz;
                    const double v = std::abs(d(order, r, k));
                    img.values[static_cast<std::size_t>(r * nz + c)] = static_cast<float>(std::log10(std::max(v, floorValue)));
                }
            return img;
        }

        DiagnosticCurve radialCurve(const sirius::OtfMeasureResult& res, int order) {
            const auto& d = res.otf.data();
            DiagnosticCurve c;
            c.title = "Order " + std::to_string(order) + " · |OTF| at kz = 0";
            if (order < 0 || order >= static_cast<int>(d.dimension(0))) return c;
            const Eigen::Index nkr = d.dimension(1);
            for (Eigen::Index r = 0; r < nkr; ++r) {
                c.x.push_back(static_cast<double>(r) * res.dkr);
                c.y.push_back(std::abs(d(order, r, 0)));
            }
            c.leftLabel = "0";
            c.midLabel = "kr · 1/um";
            c.rightLabel = num(static_cast<double>(nkr - 1) * res.dkr, 4);
            return c;
        }

        // The scale-free profile: the quantity no normalisation can move.
        DiagnosticCurve bandRatioCurve(const sirius::OtfMeasureResult& res) {
            const auto& d = res.otf.data();
            DiagnosticCurve c;
            c.title = "|order 1| / |order 0| at kz = 0 · the band ratio";
            if (d.dimension(0) < 2) return c;
            const Eigen::Index nkr = d.dimension(1);
            double peak0 = 0.0;
            for (Eigen::Index r = 0; r < nkr; ++r) peak0 = std::max(peak0, std::abs(d(0, r, 0)));
            // kr = 0 is left out for the same reason the result's bandRatio
            // leaves it out: makeotf's modify() has replaced that column.
            for (Eigen::Index r = 1; r < nkr; ++r) {
                const double a0 = std::abs(d(0, r, 0));
                if (peak0 <= 0.0 || a0 < 0.02 * peak0) continue;
                c.x.push_back(static_cast<double>(r) * res.dkr);
                c.y.push_back(std::abs(d(1, r, 0)) / a0);
            }
            c.leftLabel = c.x.empty() ? "0" : num(c.x.front(), 3);
            c.midLabel = "kr · 1/um";
            c.rightLabel = c.x.empty() ? "0" : num(c.x.back(), 3);
            return c;
        }

        DiagnosticTable beadTable(const sirius::OtfMeasureResult& res) {
            DiagnosticTable t;
            t.caption = res.beads.size() > 1 ? "Beads · " + std::to_string(res.kept) + " of " +
                                                   std::to_string(res.found) + " used"
                                             : "Bead centre";
            t.header = {"#", "x", "y", "z", "amplitude", "sigma xy", "sigma z", "residual", "status"};
            int row = 0;
            for (const sirius::BeadFit& b : res.beads) {
                t.rows.push_back({std::to_string(row + 1), num(b.x, 5), num(b.y, 5), num(b.z, 5), num(b.amplitude, 5),
                                  num(b.sigmaX, 4), num(b.sigmaZ, 4), num(b.residual, 3),
                                  sirius::beadRejectionName(b.rejection)});
                if (b.kept) t.accentCells.push_back({row, 8});
                ++row;
            }
            return t;
        }

    } // namespace

    const char* otfSectionOrderName(OtfSectionOrder o) noexcept {
        return o == OtfSectionOrder::Layout ? "layout" : "packing";
    }

    std::string otfMeasureProblems(const std::vector<OtfMeasureProblem>& problems) {
        std::string out;
        for (const OtfMeasureProblem& p : problems) {
            if (!out.empty()) out += "; ";
            out += p.field.empty() ? p.message : p.field + ": " + p.message;
        }
        return out;
    }

    std::vector<OtfMeasureProblem> validateOtfMeasureRequest(const OtfMeasureRequest& request, const DatasetMeta& meta,
                                                             const Dims5& dims) {
        std::vector<OtfMeasureProblem> out;
        const auto add = [&out](std::string field, std::string message) {
            out.push_back({std::move(field), std::move(message)});
        };

        if (dims.z <= 0 || dims.y <= 0 || dims.x <= 0 || dims.c <= 0 || dims.t <= 0) {
            add("dataset", "there is nothing to measure: the array is " + dims.toString());
            return out;
        }

        const GatherPlan plan = planOf(meta, dims, request);
        if (!plan.layoutProblem.empty())
            add("sim_layout", "the dataset's raw-SIM layout (" + meta.sim.text() + ") does not fit " + dims.toString() +
                                  ": " + plan.layoutProblem +
                                  " Correct the layout, or clear it and state the phase count here.");

        if (request.channel < 0 || request.channel >= plan.channels)
            add("channel", "channel " + std::to_string(request.channel) + " is outside the dataset's " +
                               std::to_string(plan.channels));
        if (request.time < 0 || request.time >= plan.times)
            add("time", "time point " + std::to_string(request.time) + " is outside the dataset's " +
                            std::to_string(plan.times));
        if (request.angle < 0 || request.angle >= plan.angles)
            add("angle", "illumination direction " + std::to_string(request.angle) + " is outside the " +
                             std::to_string(plan.angles) + " this acquisition holds");

        if (plan.layout) {
            if (request.measure.nphases != plan.nphases)
                add("nphases", "the acquisition holds " + std::to_string(plan.nphases) + " phases (" + meta.sim.text() +
                                   ") and the request says " + std::to_string(request.measure.nphases) +
                                   "; a stack is measured with the phases it was acquired with");
            if (plan.layoutPacking && *plan.layoutPacking != request.measure.packing)
                add("packing", std::string("the acquisition stores the phases ") +
                                   (*plan.layoutPacking == sirius::BeadPhasePacking::PhaseFastest ? "fastest" : "slowest") +
                                   " on z (" + meta.sim.text() + ") and the request says the other way round; the layout "
                                   "decides how the frames are read, so match it or clear the layout");
        }

        // A radially averaged OTF has ONE radial step for both lateral axes
        // (dkr = 1 / (min(nx, ny) dxy)), so anisotropic pixels put every
        // sample of the table at the wrong frequency -- quietly, which is the
        // failure of findings 9k.48 and exactly what this layer exists to
        // stop. There is no single dxy to pick, so the measurement is refused
        // rather than made on one of the two.
        const double dx = meta.dx(), dy = meta.dy();
        if (dx > 0.0 && dy > 0.0 && std::abs(dx - dy) > 0.01 * std::max(dx, dy))
            add("dxy", "the pixels are not square (dx " + num(dx) + " um, dy " + num(dy) +
                           " um): a radially averaged OTF has one radial step for both lateral axes, so a table "
                           "measured from these would put every sample at the wrong frequency. Resample the stack to "
                           "square pixels first.");

        // The library's own conditions, on the shape the gather will produce,
        // so its wording reaches a dialog instead of arriving later as an
        // exception nobody validated for.
        if (const std::string bad = sirius::validateOtfMeasure(measureOptionsOf(request, plan), plan.sections,
                                                               static_cast<int>(plan.ny), static_cast<int>(plan.nx));
            !bad.empty())
            add("measure", bad);

        if (!request.path.empty()) {
            const std::filesystem::path table = tablePathOf(request.path);
            std::error_code ec;
            if (!request.overwrite && std::filesystem::exists(table, ec))
                add("path", table.string() + " exists; allow overwriting to replace it");
            const std::filesystem::path dir = table.parent_path();
            if (!dir.empty() && !std::filesystem::exists(dir, ec))
                add("path", "the folder " + dir.string() + " does not exist");
        }
        return out;
    }

    OtfMeasureRequest otfMeasureDefaults(const DatasetMeta& meta, const Dims5& dims) {
        OtfMeasureRequest r;
        if (meta.dx() > 0.0) r.measure.dxy = meta.dx();
        if (meta.dz() > 0.0) r.measure.dz = meta.dz();
        const GatherPlan plan = planOf(meta, dims, r);
        if (plan.layout) {
            r.measure.nphases = plan.nphases;
            if (plan.layoutPacking) r.measure.packing = *plan.layoutPacking;
        }
        // An integer sensor's full scale is where a bead clips; a float stack
        // has none to state, and 0 means "do not test for saturation".
        switch (meta.sourceType) {
            case PixelType::UInt8: r.measure.detect.saturationLevel = 255.0; break;
            case PixelType::Int8: r.measure.detect.saturationLevel = 127.0; break;
            case PixelType::UInt16: r.measure.detect.saturationLevel = 65535.0; break;
            case PixelType::Int16: r.measure.detect.saturationLevel = 32767.0; break;
            default: break;
        }
        return r;
    }

    std::string otfMeasureFileName(const DatasetMeta& meta, Index channel) {
        std::string tag;
        if (channel >= 0 && channel < static_cast<Index>(meta.channels.size()))
            tag = sanitise(meta.channels[static_cast<std::size_t>(channel)].shortName());
        return tag.empty() ? "OTF_sirius.tif" : "OTF_" + tag + "_sirius.tif";
    }

    Diagnostics otfMeasureDiagnostics(const sirius::OtfMeasureResult& result, const OtfMeasureRequest& request,
                                      const DiagnosticImage* preview) {
        Diagnostics d;
        // Generic, not Sim: the SIM body expects three spectra per tab and
        // four named tabs. A Generic body draws images 0 and 1, the facts and
        // every curve, which is why the preview and order 0 are first.
        d.kind = DiagnosticsKind::Generic;

        if (preview) d.addImage(*preview);
        const int norders = result.norders;
        for (int o = 0; o < norders; ++o) d.addImage(orderImage(result, o));

        DiagnosticTab first;
        first.name = "Measurement";
        for (int i = 0; i < static_cast<int>(d.images.size()) && i < (preview ? 2 : 1); ++i) first.images.push_back(i);
        d.tabs.push_back(std::move(first));
        if (norders > 1) {
            DiagnosticTab orders;
            orders.name = "Orders";
            for (int o = 0; o < norders; ++o) orders.images.push_back((preview ? 1 : 0) + o);
            d.tabs.push_back(std::move(orders));
        }

        for (int o = 0; o < norders; ++o) d.curves.push_back(radialCurve(result, o));
        if (norders > 1) d.curves.push_back(bandRatioCurve(result));

        d.table = beadTable(result);

        auto fact = [&d](std::string key, std::string value) { d.facts.push_back({std::move(key), std::move(value)}); };
        fact("Table", std::to_string(result.norders) + " orders x " + std::to_string(result.nkr) + " kr x " +
                          std::to_string(result.nzotf) + " kz");
        fact("Sampling", "dkr " + num(result.dkr) + " · dkz " + num(result.dkz) + " 1/um");
        fact("Measured at", "dxy " + num(result.dxy) + " · dz " + num(result.dz) + " um");
        // The side-band division's own constant, and whether anyone stated it.
        if (result.patternPeriodUsedUm > 0.0)
            fact("Line spacing", num(result.patternPeriodUsedUm) + " um" +
                                     (result.patternPeriodStated ? " (stated)"
                                                                 : " (NOT STATED: makeotf's default)"));
        fact("Stack", std::to_string(result.nz) + " z x " + std::to_string(result.ny) + " y x " +
                          std::to_string(result.nx) + " x");
        fact("Beads", std::to_string(result.kept) + " of " + std::to_string(result.found) +
                          (request.measure.field ? " (field)" : " (single bead)"));
        fact("Band ratio", num(result.bandRatio, 5) + " ± " + num(result.bandRatioIqr, 3) + " iqr over " +
                               std::to_string(result.bandRatioSamples) + " samples");
        fact("Depth as stored", num(result.modulationDepth, 5) +
                                    (result.modulationDepthSpread > 0.0
                                         ? " ± " + num(result.modulationDepthSpread, 3) + " across beads"
                                         : std::string()));
        // What was DONE, not what was asked: scaleUsed is the result's own.
        fact("Scale", std::string(scaleName(result.scaleUsed)) + " = " + num(result.scaleDivisor));
        fact("Order 0 DC", num(result.order0Dc) + " · line fit " + num(result.lineFitToOrigin));
        fact("DC of the integral", pct(result.dcFractionOfSignal));
        fact("1 ADU of background", pct(result.scaleSensitivityPerAdu) + " of the scale");
        fact("Background", num(result.backgroundMean) + " ± " + num(result.backgroundSd, 3) + " · border mean " +
                               num(result.backgroundBorderMean) + " · darkest " + num(result.backgroundDarkest));
        fact("Bead peak SNR", num(result.beadPeakSnr, 4));
        fact("Hermitian kz error", num(result.hermitianKzError, 3));

        // The dock shows the first four warnings, so the ones that change what
        // a number means come first.
        if (result.scaleSensitivityPerAdu > 0.05)
            d.warnings.push_back("1 ADU of error in the background moves this table's scale by " +
                                 pct(result.scaleSensitivityPerAdu) + ", because order 0's zero-frequency sample is only " +
                                 pct(result.dcFractionOfSignal) +
                                 " of the stack's integral. Quote the band ratio (" + num(result.bandRatio, 5) +
                                 "), which no divisor can move, rather than the depth as stored (" +
                                 num(result.modulationDepth, 5) + ").");
        if (request.measure.field && result.kept == 1)
            d.warnings.push_back("One bead reached the average, so the field path has reduced to makeotf's "
                                 "single-bead answer; it is the right answer for this stack, not an improvement on it.");
        if (result.hermitianKzError > 1e-9)
            d.warnings.push_back("The table is not Hermitian along kz (" + num(result.hermitianKzError, 3) +
                                 "), which radialft forces by construction: that is a defect in the measurement, "
                                 "not a property of the data.");
        for (const std::string& n : result.notes) d.warnings.push_back(n);

        std::ostringstream summary;
        summary << result.norders << " orders x " << result.nkr << " kr x " << result.nzotf << " kz · band ratio "
                << num(result.bandRatio, 5) << " · " << result.kept << " of " << result.found << " beads · scale "
                << scaleName(result.scaleUsed);
        d.summary = summary.str();
        d.footer = result.summary();
        return d;
    }

    nlohmann::json otfMeasureProvenance(const sirius::OtfMeasureResult& result, const DatasetMeta& meta,
                                        const OtfMeasureRequest& request, const OtfMeasureReport& report) {
        using nlohmann::json;
        const sirius::OtfMeasureOptions& m = request.measure;

        json beads = json::array();
        for (const sirius::BeadFit& b : result.beads)
            beads.push_back({{"x", b.x},
                             {"y", b.y},
                             {"z", b.z},
                             {"amplitude", b.amplitude},
                             {"offset", b.offset},
                             {"sigma_x", b.sigmaX},
                             {"sigma_y", b.sigmaY},
                             {"sigma_z", b.sigmaZ},
                             {"residual", b.residual},
                             {"nearest_neighbour_px", b.nearestNeighbourPx},
                             {"kept", b.kept},
                             {"status", sirius::beadRejectionName(b.rejection)}});
        json rejected = json::object();
        for (int i = 0; i < static_cast<int>(result.rejected.size()); ++i)
            if (result.rejected[static_cast<std::size_t>(i)] > 0)
                rejected[sirius::beadRejectionName(static_cast<sirius::BeadRejection>(i))] =
                    result.rejected[static_cast<std::size_t>(i)];

        json stack = {{"name", meta.name},
                      {"path", meta.sourcePath},
                      {"format", meta.format},
                      {"acquisition", meta.acquisition},
                      {"dims", {{"c", meta.dims.c}, {"t", meta.dims.t}, {"z", meta.dims.z}, {"y", meta.dims.y}, {"x", meta.dims.x}}},
                      {"source_type", sirius::toString(meta.sourceType)},
                      {"voxel_um", {{"x", meta.dx()}, {"y", meta.dy()}, {"z", meta.dz()}}},
                      {"frame_interval_s", meta.frameIntervalS},
                      {"channel", request.channel},
                      {"time", request.time},
                      {"angle", request.angle},
                      {"angles", report.angles},
                      {"sim_layout", meta.sim.present ? json(meta.sim.text()) : json(nullptr)},
                      {"section_order", otfSectionOrderName(report.sectionOrder)},
                      {"sections", report.sections},
                      {"nphases", report.nphases},
                      {"nz", report.nz},
                      {"section_shape", {{"y", report.ny}, {"x", report.nx}}}};
        if (request.channel >= 0 && request.channel < static_cast<Index>(meta.channels.size())) {
            const ChannelInfo& ch = meta.channels[static_cast<std::size_t>(request.channel)];
            stack["channel_label"] = ch.label;
            stack["channel_emission_nm"] = ch.wavelengthNm;
        }

        json options = {{"nphases", report.nphases},
                        {"norders", result.norders},
                        {"packing", packingName(report.sectionOrder == OtfSectionOrder::Layout
                                                    ? sirius::BeadPhasePacking::PhaseFastest
                                                    : m.packing)},
                        {"phases", m.phases},
                        {"dxy_um", m.dxy},
                        {"dz_um", m.dz},
                        {"background", m.background},
                        {"background_estimate", backgroundName(m.backgroundEstimate)},
                        {"background_border_px", m.backgroundBorder},
                        {"darkest_fraction", m.darkestFraction},
                        {"apodize_px", m.apodize},
                        {"bead_diameter_um", m.beadDiameterUm},
                        // What was asked for, then what the division used: 0
                        // asked for means the fallback, and a table measured
                        // on a fallback spacing cannot be told from one
                        // measured on the real thing by looking at it.
                        {"pattern_period_um", m.patternPeriodUm},
                        {"pattern_period_um_used", result.patternPeriodUsedUm},
                        {"pattern_period_stated", result.patternPeriodStated},
                        {"pattern_angle_rad", m.patternAngleRad},
                        {"bead_compensation_pixel_um", m.beadCompensationPixelUm},
                        {"bead_compensation_axial_um", m.beadCompensationAxialUm},
                        {"scale", scaleName(m.scale)},
                        {"line_fit", {m.lineFitFirst, m.lineFitLast}},
                        {"band_ratio_min_order0", m.bandRatioMinOrder0},
                        {"repair_kr0_column", m.repairKr0Column},
                        {"combine_reim", m.combineReIm},
                        {"field", m.field},
                        {"per_bead_normalise", m.perBeadNormalise}};
        if (m.field)
            options["detect"] = {{"dog_small_lateral_um", m.detect.dogSmallLateralUm},
                                 {"dog_small_axial_um", m.detect.dogSmallAxialUm},
                                 {"dog_large_lateral_um", m.detect.dogLargeLateralUm},
                                 {"dog_large_axial_um", m.detect.dogLargeAxialUm},
                                 {"min_separation_lateral_um", m.detect.minSeparationLateralUm},
                                 {"min_separation_axial_um", m.detect.minSeparationAxialUm},
                                 {"min_amplitude", m.detect.minAmplitude},
                                 {"min_amplitude_fraction", m.detect.minAmplitudeFraction},
                                 {"saturation_level", m.detect.saturationLevel},
                                 {"roi_lateral_um", m.detect.roiLateralUm},
                                 {"roi_axial_um", m.detect.roiAxialUm},
                                 {"boundary_margin_lateral_um", m.detect.boundaryMarginLateralUm},
                                 {"boundary_margin_axial_um", m.detect.boundaryMarginAxialUm},
                                 {"sigma_min_lateral_um", m.detect.sigmaMinLateralUm},
                                 {"sigma_max_lateral_um", m.detect.sigmaMaxLateralUm},
                                 {"sigma_min_axial_um", m.detect.sigmaMinAxialUm},
                                 {"sigma_max_axial_um", m.detect.sigmaMaxAxialUm},
                                 {"max_beads", m.detect.maxBeads},
                                 {"max_residual", m.detect.maxResidual}};

        json out = {{"artefact", "otf"},
                    {"produced_by", "sirius measureOtfFromDataset"},
                    {"sirius", toJson(buildInfo())},
                    {"table", request.path.empty() ? json(nullptr) : json(report.tablePath.string())},
                    {"note", request.note},
                    {"stack", stack},
                    {"options", options},
                    {"sampling",
                     {{"dkr_per_um", result.dkr},
                      {"dkz_per_um", result.dkz},
                      {"dxy_um", result.dxy},
                      {"dz_um", result.dz},
                      {"kz_origin", "dc_first"},
                      {"norders", result.norders},
                      {"nkr", result.nkr},
                      {"nzotf", result.nzotf}}},
                    {"scale",
                     {{"used", scaleName(result.scaleUsed)},
                      {"divisor", result.scaleDivisor},
                      {"order0_dc", result.order0Dc},
                      {"line_fit_to_origin", result.lineFitToOrigin},
                      {"dc_fraction_of_signal", result.dcFractionOfSignal},
                      {"sensitivity_per_adu", result.scaleSensitivityPerAdu},
                      {"total_signal", result.totalSignal},
                      {"background_total", result.backgroundTotal}}},
                    {"background",
                     {{"mean", result.backgroundMean},
                      {"sd", result.backgroundSd},
                      {"border_mean", result.backgroundBorderMean},
                      {"darkest", result.backgroundDarkest}}},
                    {"result",
                     {{"modulation_depth", result.modulationDepth},
                      {"modulation_depth_spread", result.modulationDepthSpread},
                      {"band_ratio", result.bandRatio},
                      {"band_ratio_iqr", result.bandRatioIqr},
                      {"band_ratio_samples", result.bandRatioSamples},
                      {"order_dc", result.orderDc},
                      {"hermitian_kz_error", result.hermitianKzError},
                      {"bead_peak_snr", result.beadPeakSnr},
                      {"found", result.found},
                      {"kept", result.kept}}},
                    {"beads", beads},
                    {"rejected", rejected},
                    {"notes", report.notes},
                    {"measurement_notes", result.notes},
                    {"summary", result.summary()}};
        return out;
    }

    OtfMeasureReport measureOtfFromDataset(const Array5& array, const DatasetMeta& meta,
                                           const OtfMeasureRequest& request,
                                           const std::function<void(double, const std::string&)>& progress,
                                           const std::function<bool()>& cancelled) {
        const Dims5 dims = array.dims();
        if (const std::vector<OtfMeasureProblem> problems = validateOtfMeasureRequest(request, meta, dims);
            !problems.empty())
            throw std::invalid_argument(otfMeasureProblems(problems));

        const auto report = [&progress](double f, const std::string& what) {
            if (progress) progress(std::clamp(f, 0.0, 1.0), what);
        };
        const auto checkCancel = [&cancelled] {
            if (cancelled && cancelled()) throw CancelledError();
        };

        const GatherPlan plan = planOf(meta, dims, request);
        const sirius::OtfMeasureOptions m = measureOptionsOf(request, plan);

        OtfMeasureReport out;
        out.sectionOrder = plan.layout ? OtfSectionOrder::Layout : OtfSectionOrder::Packing;
        out.sections = plan.sections;
        out.nphases = m.nphases;
        out.nz = plan.nz;
        out.angles = plan.angles;
        out.ny = plan.ny;
        out.nx = plan.nx;

        // What this layer decided, said once and in both the report and the
        // provenance: a reader of the table should not have to infer it.
        if (plan.layout)
            out.notes.push_back("the frames were gathered through the acquisition's own layout (" + meta.sim.text() +
                                "), direction " + std::to_string(request.angle + 1) + " of " +
                                std::to_string(plan.angles) + ", channel " + std::to_string(request.channel) +
                                ", time point " + std::to_string(request.time) + ", phase fastest");
        else
            out.notes.push_back("the dataset states no raw-SIM layout, so the z axis was read as stored: " +
                                std::to_string(plan.sections) + " sections as " + std::to_string(m.nphases) +
                                " phases x " + std::to_string(plan.nz) + " z, " + packingName(m.packing));
        if (meta.dx() > 0.0 && std::abs(m.dxy - meta.dx()) > 1e-9)
            out.notes.push_back("the measurement used dxy " + num(m.dxy) + " um where the dataset says " +
                                num(meta.dx()) + " um, so dkr is the stated one, not the acquisition's");
        if (meta.dz() > 0.0 && m.dz > 0.0 && std::abs(m.dz - meta.dz()) > 1e-9)
            out.notes.push_back("the measurement used dz " + num(m.dz) + " um where the dataset says " +
                                num(meta.dz()) + " um");
        // The bead diameter and the illumination are stated parameters -- no
        // acquisition metadata this layer sees carries either -- so say which
        // numbers were used, and say it differently when the line spacing was
        // not one of them. A measurement on makeotf's 0.2 um default when the
        // instrument runs 0.504 um (the iSOAR2 configs) scales order 1 by
        // about 1.38 with nothing in the table to show it.
        if (m.patternPeriodUm > 0.0)
            out.notes.push_back("bead diameter " + num(m.beadDiameterUm) + " um with the illumination at " +
                                num(m.patternPeriodUm) + " um / " + num(m.patternAngleRad, 4) +
                                " rad are stated parameters: nothing in the acquisition's metadata says them");
        else if (m.beadDiameterUm > 0.0)
            out.notes.push_back("the illumination line spacing was not stated, so the finite-bead-size "
                                "division of the side bands fell back to makeotf's " +
                                num(sirius::kMakeotfLineSpacingUm) +
                                " um; state the acquisition's own (the iSOAR2 configs say 0.504 um) or "
                                "quote only order 0 and the band ratio");
        if (m.beadCompensationPixelUm > 0.0)
            out.notes.push_back("the finite-bead-size division used " + num(m.beadCompensationPixelUm) +
                                " um rather than the acquisition's pixel, which is what reaching an existing "
                                "makeotf table needs (its own default is 0.106 um)");
        if (m.background < 0.0 && 2 * m.backgroundBorder > std::min(plan.ny, plan.nx) / 2)
            out.notes.push_back("makeotf's background border of " + std::to_string(m.backgroundBorder) + " px covers " +
                                pct(1.0 - static_cast<double>(std::max<Index>(0, plan.nx - 2 * m.backgroundBorder) *
                                                              std::max<Index>(0, plan.ny - 2 * m.backgroundBorder)) /
                                             static_cast<double>(plan.nx * plan.ny)) +
                                " of a " + std::to_string(plan.nx) + " x " + std::to_string(plan.ny) +
                                " section, so it holds the bead's own out-of-focus haze and over-subtracts; "
                                "BackgroundEstimate::DarkestFraction is the alternative");

        report(0.05, "reading the stack");
        const Eigen::Tensor<double, 3, Eigen::RowMajor> stack = gather(array, plan, request);
        checkCancel();

        report(0.2, m.field ? "detecting beads and measuring" : "measuring");
        out.measurement = sirius::measureOTF(stack, m);
        checkCancel();

        if (!request.path.empty()) {
            out.tablePath = tablePathOf(request.path);
            report(0.8, "writing " + out.tablePath.string());
            sirius::OtfWriteOptions w;
            w.sidecar = request.sidecar;
            w.note = request.note;
            out.files = sirius::writeMeasuredOTF(out.tablePath.string(), out.measurement, w);
        }

        out.provenance = otfMeasureProvenance(out.measurement, meta, request, out);
        if (!out.tablePath.empty() && request.provenanceFile) {
            const std::filesystem::path side = out.tablePath.string() + ".json";
            std::ofstream f(side);
            if (!f) throw std::runtime_error("cannot write " + side.string());
            f << out.provenance.dump(2) << "\n";
            if (!f) throw std::runtime_error("writing " + side.string() + " failed");
            out.files.push_back(side.string());
        }
        for (const std::string& file : out.files) {
            std::error_code ec;
            const std::uintmax_t size = std::filesystem::file_size(std::filesystem::path(file), ec);
            if (!ec) out.bytes += size;
        }

        report(0.95, "diagnostics");
        const DiagnosticImage preview = beadPreview(stack, out.measurement, request.previewMaxSide);
        out.diagnostics = otfMeasureDiagnostics(out.measurement, request, &preview);
        for (const std::string& n : out.notes) out.diagnostics.warnings.push_back(n);
        out.summary = out.diagnostics.summary;
        report(1.0, "done");
        return out;
    }


    // --- the JSON face the tool and the bindings share --------------------------

    namespace {

        template <typename T> void readIf(const nlohmann::json& a, const char* key, T& into) {
            const auto it = a.find(key);
            if (it != a.end() && !it->is_null()) into = it->get<T>();
        }

        // The enums, with the accepted spellings in ONE place: a front end
        // that states them in its schema and a front end that does not both
        // get this message for a value neither of them should have sent.
        sirius::BeadPhasePacking packingFromJson(const std::string& v) {
            if (v == "phase_fastest") return sirius::BeadPhasePacking::PhaseFastest;
            if (v == "phase_slowest") return sirius::BeadPhasePacking::PhaseSlowest;
            throw std::invalid_argument("packing must be \"phase_fastest\" or \"phase_slowest\", not \"" + v + "\"");
        }
        sirius::OtfMeasureScale scaleFromJson(const std::string& v) {
            if (v == "order0_dc") return sirius::OtfMeasureScale::Order0Dc;
            if (v == "makeotf_fixorigin") return sirius::OtfMeasureScale::MakeotfFixOrigin;
            if (v == "as_measured") return sirius::OtfMeasureScale::AsMeasured;
            throw std::invalid_argument("scale must be \"order0_dc\", \"makeotf_fixorigin\" or \"as_measured\", not \"" +
                                        v + "\"");
        }
        sirius::BackgroundEstimate backgroundFromJson(const std::string& v) {
            if (v == "border_mean") return sirius::BackgroundEstimate::BorderMean;
            if (v == "darkest_fraction") return sirius::BackgroundEstimate::DarkestFraction;
            throw std::invalid_argument("background_estimate must be \"border_mean\" or \"darkest_fraction\", not \"" +
                                        v + "\"");
        }

        // Every key this face reads, in ONE list beside the reader, so a key
        // it does not read is REFUSED instead of dropped. Dropping is the
        // worse failure of the two: a call carrying "pattern_period" or
        // "patternPeriodUm" measured on the default line spacing and reported
        // success, which is a wrong number with nothing saying so -- the same
        // shape as the defect this list was added with.
        constexpr std::array<const char*, 34> kRequestKeys{
            "channel", "time", "angle", "path", "note", "overwrite", "sidecar", "provenance",
            "preview_max_side", "nphases", "norders", "packing", "phases", "dxy", "dz", "background",
            "background_estimate", "background_border", "darkest_fraction", "apodize", "bead_diameter_um",
            "pattern_period_um", "pattern_angle_rad", "bead_compensation_pixel_um", "bead_compensation_axial_um",
            "scale", "line_fit_first", "line_fit_last", "band_ratio_min_order0", "repair_kr0_column",
            "combine_reim", "field", "per_bead_normalise", "detect"};
        constexpr std::array<const char*, 19> kDetectKeys{
            "dog_small_lateral_um", "dog_small_axial_um", "dog_large_lateral_um", "dog_large_axial_um",
            "min_separation_lateral_um", "min_separation_axial_um", "min_amplitude", "min_amplitude_fraction",
            "saturation_level", "roi_lateral_um", "roi_axial_um", "boundary_margin_lateral_um",
            "boundary_margin_axial_um", "sigma_min_lateral_um", "sigma_max_lateral_um", "sigma_min_axial_um",
            "sigma_max_axial_um", "max_beads", "max_residual"};

        template <std::size_t N>
        void refuseUnknownKeys(const nlohmann::json& object, const std::array<const char*, N>& known,
                               const std::string& where) {
            for (auto it = object.begin(); it != object.end(); ++it) {
                const std::string key = it.key();
                if (std::find_if(known.begin(), known.end(),
                                 [&key](const char* k) { return key == k; }) != known.end())
                    continue;
                std::string accepted;
                for (const char* k : known) accepted += (accepted.empty() ? "" : ", ") + std::string(k);
                throw std::invalid_argument(where + " does not accept \"" + key + "\"; its keys are " + accepted);
            }
        }

    } // namespace

    OtfMeasureRequest otfMeasureRequestFromJson(const nlohmann::json& args, const DatasetMeta& meta, const Dims5& dims) {
        if (!args.is_object() && !args.is_null())
            throw std::invalid_argument("the OTF measurement's arguments must be an object");
        OtfMeasureRequest r = otfMeasureDefaults(meta, dims);
        if (args.is_null()) return r;
        const nlohmann::json& a = args;
        refuseUnknownKeys(a, kRequestKeys, "the OTF measurement");

        readIf(a, "channel", r.channel);
        readIf(a, "time", r.time);
        readIf(a, "angle", r.angle);
        readIf(a, "path", r.path);
        readIf(a, "note", r.note);
        readIf(a, "overwrite", r.overwrite);
        readIf(a, "sidecar", r.sidecar);
        readIf(a, "provenance", r.provenanceFile);
        readIf(a, "preview_max_side", r.previewMaxSide);

        sirius::OtfMeasureOptions& m = r.measure;
        readIf(a, "nphases", m.nphases);
        readIf(a, "norders", m.norders);
        if (const auto it = a.find("packing"); it != a.end() && !it->is_null())
            m.packing = packingFromJson(it->get<std::string>());
        readIf(a, "phases", m.phases);
        readIf(a, "dxy", m.dxy);
        readIf(a, "dz", m.dz);
        readIf(a, "background", m.background);
        if (const auto it = a.find("background_estimate"); it != a.end() && !it->is_null())
            m.backgroundEstimate = backgroundFromJson(it->get<std::string>());
        readIf(a, "background_border", m.backgroundBorder);
        readIf(a, "darkest_fraction", m.darkestFraction);
        readIf(a, "apodize", m.apodize);
        readIf(a, "bead_diameter_um", m.beadDiameterUm);
        readIf(a, "pattern_period_um", m.patternPeriodUm);
        readIf(a, "pattern_angle_rad", m.patternAngleRad);
        readIf(a, "bead_compensation_pixel_um", m.beadCompensationPixelUm);
        readIf(a, "bead_compensation_axial_um", m.beadCompensationAxialUm);
        if (const auto it = a.find("scale"); it != a.end() && !it->is_null())
            m.scale = scaleFromJson(it->get<std::string>());
        readIf(a, "line_fit_first", m.lineFitFirst);
        readIf(a, "line_fit_last", m.lineFitLast);
        readIf(a, "band_ratio_min_order0", m.bandRatioMinOrder0);
        readIf(a, "repair_kr0_column", m.repairKr0Column);
        readIf(a, "combine_reim", m.combineReIm);
        readIf(a, "field", m.field);
        readIf(a, "per_bead_normalise", m.perBeadNormalise);

        // A `detect` that is not an object was dropped without a word, which
        // is the same silence as an unknown key: a caller that sent a list of
        // pairs, or one value, measured with the defaults and was told
        // nothing.
        if (const auto it = a.find("detect"); it != a.end() && !it->is_null()) {
            if (!it->is_object())
                throw std::invalid_argument("detect must be an object of bead-detection options, not " +
                                            std::string(it->type_name()));
            const nlohmann::json& dj = *it;
            refuseUnknownKeys(dj, kDetectKeys, "the OTF measurement's detect");
            sirius::BeadDetectionOptions& d = m.detect;
            readIf(dj, "dog_small_lateral_um", d.dogSmallLateralUm);
            readIf(dj, "dog_small_axial_um", d.dogSmallAxialUm);
            readIf(dj, "dog_large_lateral_um", d.dogLargeLateralUm);
            readIf(dj, "dog_large_axial_um", d.dogLargeAxialUm);
            readIf(dj, "min_separation_lateral_um", d.minSeparationLateralUm);
            readIf(dj, "min_separation_axial_um", d.minSeparationAxialUm);
            readIf(dj, "min_amplitude", d.minAmplitude);
            readIf(dj, "min_amplitude_fraction", d.minAmplitudeFraction);
            readIf(dj, "saturation_level", d.saturationLevel);
            readIf(dj, "roi_lateral_um", d.roiLateralUm);
            readIf(dj, "roi_axial_um", d.roiAxialUm);
            readIf(dj, "boundary_margin_lateral_um", d.boundaryMarginLateralUm);
            readIf(dj, "boundary_margin_axial_um", d.boundaryMarginAxialUm);
            readIf(dj, "sigma_min_lateral_um", d.sigmaMinLateralUm);
            readIf(dj, "sigma_max_lateral_um", d.sigmaMaxLateralUm);
            readIf(dj, "sigma_min_axial_um", d.sigmaMinAxialUm);
            readIf(dj, "sigma_max_axial_um", d.sigmaMaxAxialUm);
            readIf(dj, "max_beads", d.maxBeads);
            readIf(dj, "max_residual", d.maxResidual);
        }
        return r;
    }

    nlohmann::json otfMeasureReportJson(const OtfMeasureReport& report) {
        using nlohmann::json;
        const sirius::OtfMeasureResult& m = report.measurement;
        json rejected = json::object();
        for (int i = 0; i < static_cast<int>(m.rejected.size()); ++i)
            if (m.rejected[static_cast<std::size_t>(i)] > 0)
                rejected[sirius::beadRejectionName(static_cast<sirius::BeadRejection>(i))] =
                    m.rejected[static_cast<std::size_t>(i)];
        return json{{"table", report.tablePath.empty() ? json(nullptr) : json(report.tablePath.string())},
                    {"files", report.files},
                    {"bytes", report.bytes},
                    {"summary", report.summary},
                    {"section_order", otfSectionOrderName(report.sectionOrder)},
                    {"sections", report.sections},
                    {"nphases", report.nphases},
                    {"nz", report.nz},
                    {"angles", report.angles},
                    {"section_shape", {{"y", report.ny}, {"x", report.nx}}},
                    {"otf",
                     {{"norders", m.norders},
                      {"nkr", m.nkr},
                      {"nzotf", m.nzotf},
                      {"dkr_per_um", m.dkr},
                      {"dkz_per_um", m.dkz},
                      {"kz_origin", "dc_first"}}},
                    {"beads", {{"found", m.found}, {"kept", m.kept}, {"rejected", rejected}}},
                    // the two numbers a reader has to keep apart: the band
                    // ratio no divisor can move, and the depth as stored,
                    // which 1 ADU of background can double
                    {"band_ratio", m.bandRatio},
                    {"band_ratio_iqr", m.bandRatioIqr},
                    {"band_ratio_samples", m.bandRatioSamples},
                    {"modulation_depth", m.modulationDepth},
                    {"scale",
                     {{"used", scaleName(m.scaleUsed)},
                      {"divisor", m.scaleDivisor},
                      {"dc_fraction_of_signal", m.dcFractionOfSignal},
                      {"sensitivity_per_adu", m.scaleSensitivityPerAdu}}},
                    {"notes", report.notes},
                    {"warnings", report.diagnostics.warnings},
                    {"provenance", report.provenance}};
    }

} // namespace sirius::app
