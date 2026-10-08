#include "sirius/otf_io.hpp"
#include "sirius/errors.hpp"
#include "sirius/mrc_io.hpp"
#include "sirius/tiff_io.hpp"

#include <toml++/toml.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace sirius {
    namespace {
        using Cplx = std::complex<double>;
        using CTensor = Eigen::Tensor<Cplx, 3, Eigen::RowMajor>;
        using DTensor = Eigen::Tensor<double, 3, Eigen::RowMajor>;
        using Idx = Eigen::Index;

        std::string num(double v, int digits = 6) {
            std::ostringstream os;
            os << std::setprecision(digits) << v;
            return os.str();
        }

        std::string shapeOf(Idx a, Idx b, Idx c) {
            return "(" + std::to_string(a) + ", " + std::to_string(b) + ", " + std::to_string(c) + ")";
        }

        // --- reading the bytes ------------------------------------------------
        struct RawStack {
            DTensor values;                       // (sections, rows, values per row)
            bool mrc = false;
            double cellDkz = 0.0, cellDkr = 0.0;  // > 0 when the container carries them
        };

        RawStack readRawStack(const std::string& filename) {
            RawStack r;
            if (isMrcName(filename)) {
                r.mrc = true;
                // The MRC / DeltaVision OTF containers makeotf writes put the
                // steps in the cell lengths: xlen = dkz (the axis that runs
                // fastest, kz), ylen = dkr. That is how cudasirecon recovers
                // the sampling of a .dv OTF, and it is the only OTF container
                // we read that states its own sampling at all.
                const MrcInfo info = inspectMrc(filename);
                r.cellDkz = static_cast<double>(info.cell[0]);
                r.cellDkr = static_cast<double>(info.cell[1]);
                const Buffer<double> stack = MrcFile(filename).readStack<double>();
                r.values = DTensor(stack.dim(0), stack.dim(1), stack.dim(2));
                if (stack.size() > 0)
                    std::memcpy(r.values.data(), stack.data(),
                                static_cast<std::size_t>(stack.size()) * sizeof(double));
            } else {
                r.values = readTiffStack<double>(filename);
            }
            return r;
        }

        // --- structure --------------------------------------------------------

        // Mean |t(kr, k) - conj(t(kr, -k))| over mean |t|, with the kz axis
        // read as though index j held kz index j - roll. makeotf forces this
        // symmetry exactly (radialft.cpp fills the negative kz half from the
        // positive one), so a real table measures 0 and a misdecoded file
        // measures order 1. Returns 0 when there is nothing to compare
        // (fewer than 3 kz planes).
        double hermitianKzError(const CTensor& t, int roll) {
            const Idx no = t.dimension(0), nkr = t.dimension(1), n = t.dimension(2);
            if (n < 3 || no < 1 || nkr < 1) return 0.0;
            const auto wrap = [n](Idx j) { return ((j % n) + n) % n; };
            double mean = 0.0;
            for (Idx o = 0; o < no; ++o)
                for (Idx ir = 0; ir < nkr; ++ir)
                    for (Idx j = 0; j < n; ++j) mean += std::abs(t(o, ir, j));
            mean /= static_cast<double>(no * nkr * n);
            if (!(mean > 0.0)) return 0.0;
            double err = 0.0;
            Idx count = 0;
            for (Idx o = 0; o < no; ++o)
                for (Idx ir = 0; ir < nkr; ++ir)
                    for (Idx k = 1; k <= n / 2; ++k) {
                        const Cplx a = t(o, ir, wrap(k - roll));
                        const Cplx b = t(o, ir, wrap(n - k - roll));
                        err += std::abs(a - std::conj(b));
                        ++count;
                    }
            if (count == 0) return 0.0;
            return (err / static_cast<double>(count)) / mean;
        }

        // out(o, kr, j) = in(o, kr, j - roll): put the plane that holds kz = 0
        // at index 0, which is where every consumer of the table looks for it
        // (otfInterpolate indexes kz in FFT order).
        CTensor rollKz(const CTensor& t, int roll) {
            const Idx no = t.dimension(0), nkr = t.dimension(1), n = t.dimension(2);
            CTensor out(no, nkr, n);
            for (Idx o = 0; o < no; ++o)
                for (Idx ir = 0; ir < nkr; ++ir)
                    for (Idx j = 0; j < n; ++j) out(o, ir, j) = t(o, ir, (((j - roll) % n) + n) % n);
            return out;
        }

        // --- the normalisation reference --------------------------------------

        // cudasirecon's fixorigin arithmetic (radialft.cpp:1188-1223): the
        // kz-SUMMED real profile, whose sample 0 is the DC alone ("don't want
        // to add up garbages on kz axis"), least-squares fitted over
        // [first, last] and extrapolated to kr = 0. In makeotf this value
        // replaces the kr = 0 sample before rescale() divides by it; here it
        // is the fallback reference for a table whose DC cannot serve.
        double lineFitToOrigin(const CTensor& t, int first, int last, bool* ok) {
            const Idx nkr = t.dimension(1), nz = t.dimension(2);
            const int lo = std::max(1, first);
            const int hi = std::min<int>(last, static_cast<int>(nkr) - 1);
            if (ok) *ok = false;
            if (hi <= lo) return 0.0;
            const double meani = 0.5 * (lo + hi);
            double totsum = 0.0, ysum = 0.0, sqsum = 0.0;
            for (int i = lo; i <= hi; ++i) {
                double s = 0.0;
                for (Idx j = 0; j < nz; ++j) s += t(0, i, j).real();
                totsum += s;
                ysum += s * (static_cast<double>(i) - meani);
                sqsum += (static_cast<double>(i) - meani) * (static_cast<double>(i) - meani);
            }
            if (!(sqsum > 0.0)) return 0.0;
            const double slope = ysum / sqsum;
            const double avg = totsum / static_cast<double>(hi - lo + 1);
            const double v = avg + (0.0 - meani) * slope;
            if (!std::isfinite(v)) return 0.0;
            if (ok) *ok = true;
            return v;
        }

        // --- a plain 2D OTF image --------------------------------------------

        // One real (N, N) page centred on DC (or in FFT order) radially
        // averaged onto N/2 + 1 samples and one kz plane, the same profile in
        // every order. The arithmetic follows
        // latents/scripts/matlab2d_sirius.py's otf_to_sirius with one stated
        // difference: samples whose radius rounds past the last bin are
        // DROPPED, as idealOTF drops them (otf_ideal.cpp: "if (ir >= nkr)
        // continue"), where that script clips them into the last bin. The
        // corners of a square page reach sqrt(2) * N/2 and are not part of
        // the kr axis; folding them in contaminates the outermost sample.
        // That script's OTFpost / OTFedgeF periphery flattening is NOT done
        // here: it is OpenSIM's own preprocessing of its own OTF, and a
        // reader that quietly rewrote measured values would be the very thing
        // this file exists to stop.
        CTensor radialAverage2dImage(const DTensor& page, int norders, const std::string& filename,
                                     std::vector<std::string>& notes) {
            const Idx n = page.dimension(1);
            const Idx nkr = n / 2 + 1;
            Idx imax = 0, jmax = 0;
            double best = -std::numeric_limits<double>::infinity();
            for (Idx i = 0; i < n; ++i)
                for (Idx j = 0; j < n; ++j)
                    if (page(0, i, j) > best) {
                        best = page(0, i, j);
                        imax = i;
                        jmax = j;
                    }
            const Idx c = n / 2;
            const auto nearCentre = [&](Idx v) { return (v > c ? v - c : c - v) <= 2; };
            const auto nearZero = [&](Idx v) { return std::min(v, n - v) <= 2; };
            bool fftOrder = false;
            if (nearCentre(imax) && nearCentre(jmax)) {
                fftOrder = false;
            } else if (nearZero(imax) && nearZero(jmax)) {
                fftOrder = true;
                notes.push_back("the 2D OTF image has its maximum at the corner, so it is read in FFT order (DC at index 0), not centred");
            } else {
                throw IoError(filename + ": read as a plain 2D OTF image " + shapeOf(1, n, n) +
                              ", but its maximum sits at row " + std::to_string(imax) + ", column " +
                              std::to_string(jmax) + ", which is neither the centre (" + std::to_string(c) +
                              ", " + std::to_string(c) + ") nor the DC corner. An OTF image is centred on DC.");
            }
            std::vector<double> sum(static_cast<std::size_t>(nkr), 0.0);
            std::vector<double> count(static_cast<std::size_t>(nkr), 0.0);
            for (Idx i = 0; i < n; ++i)
                for (Idx j = 0; j < n; ++j) {
                    const double dy = fftOrder ? static_cast<double>(std::min(i, n - i)) : static_cast<double>(i - c);
                    const double dx = fftOrder ? static_cast<double>(std::min(j, n - j)) : static_cast<double>(j - c);
                    const auto ir = static_cast<Idx>(std::lround(std::hypot(dx, dy)));
                    if (ir >= nkr) continue;
                    sum[static_cast<std::size_t>(ir)] += page(0, i, j);
                    count[static_cast<std::size_t>(ir)] += 1.0;
                }
            Idx empty = 0;
            std::vector<double> prof(static_cast<std::size_t>(nkr), 0.0);
            for (Idx ir = 0; ir < nkr; ++ir) {
                const double cn = count[static_cast<std::size_t>(ir)];
                if (cn > 0.0) prof[static_cast<std::size_t>(ir)] = sum[static_cast<std::size_t>(ir)] / cn;
                else ++empty;
            }
            if (empty > 0)
                notes.push_back(std::to_string(empty) + " of the " + std::to_string(nkr) +
                                " radial samples had no pixel of the page and are 0");
            notes.push_back("radially averaged a plain 2D OTF image " + shapeOf(1, n, n) + " onto " +
                            std::to_string(nkr) + " radial samples x 1 kz plane, the same profile in each of the " +
                            std::to_string(norders) + " orders (the real part; a 2D OTF image is a magnitude)");
            CTensor out(norders, nkr, 1);
            for (int o = 0; o < norders; ++o)
                for (Idx ir = 0; ir < nkr; ++ir) out(o, ir, 0) = Cplx(prof[static_cast<std::size_t>(ir)], 0.0);
            return out;
        }

        // --- the load ---------------------------------------------------------

        OTFRadiallyAveraged loadImpl(const std::string& filename, const SIMParameters* p,
                                     double dkrExplicit, double dkzExplicit, bool explicitSteps,
                                     const OtfLoadOptions& opts, OtfLoadReport* reportOut) {
            OtfLoadReport rep;
            rep.file = filename;
            rep.normalizationMode = opts.normalization;

            RawStack raw = readRawStack(filename);
            const Idx d0 = raw.values.dimension(0), d1 = raw.values.dimension(1), d2 = raw.values.dimension(2);
            rep.storedShape = {static_cast<int>(d0), static_cast<int>(d1), static_cast<int>(d2)};
            if (raw.values.size() == 0) throw IoError("Radial OTF is empty: " + filename);

            const int expectedOrders = opts.expectedOrders > 0
                                           ? opts.expectedOrders
                                           : (p != nullptr ? p->resolvedOrders() : 0);
            const Idx minKr = std::max(1, opts.minRadialSamples);

            // Which layout? A single square real page is a 2D OTF image: a
            // radial table is square only if nkr == 2 * nzotf, which needs a
            // PSF of 2 * (2 * nzotf - 1) pixels across for nzotf kz planes
            // and does not happen for any (nkr >= minKr) table makeotf
            // writes. Measured shapes: (3, 129, 130), (2, 257, 402),
            // (2, 65, 202), (2, 257, 202 complex), against (1, 512, 512) for
            // OpenSIM's OTF image.
            const bool squarePage = d0 == 1 && d1 == d2 && d1 >= 2 * minKr;
            const bool evenLast = d2 % 2 == 0;

            CTensor table;
            if (squarePage && opts.accept2dImage && !raw.mrc) {
                rep.layout = OtfLayout::Plain2dImage;
                table = radialAverage2dImage(raw.values, std::max(1, expectedOrders), filename, rep.notes);
            } else {
                rep.layout = OtfLayout::CudasireconRadial;
                const std::string got = "got " + shapeOf(d0, d1, d2) +
                                        (raw.mrc ? " (sections, rows, interleaved values per row)"
                                                 : " (pages, rows, columns)");
                const std::string want =
                    "a cudasirecon radially averaged table is (norders, nkr, 2 * nzotf) with real and "
                    "imaginary parts interleaved along the last axis: norders 1.." +
                    std::to_string(opts.maxOrders) + ", nkr >= " + std::to_string(minKr) +
                    ", and an EVEN last axis. A plain 2D OTF image is one square real page (N, N) with N >= " +
                    std::to_string(2 * minKr) + ".";
                if (!evenLast)
                    throw IoError(filename + ": the last axis has an odd length, so it cannot pair into "
                                             "real and imaginary parts -- " + got + "; " + want);
                if (d0 > opts.maxOrders || d1 < minKr || d2 < 2)
                    throw IoError(filename + ": not a shape an OTF is stored in -- " + got + "; " + want);

                // complex_otf = raw[..., 0::2] + i * raw[..., 1::2]
                const Eigen::array<Idx, 3> startReal = {0, 0, 0};
                const Eigen::array<Idx, 3> startImag = {0, 0, 1};
                const Eigen::array<Idx, 3> stop = raw.values.dimensions();
                const Eigen::array<Idx, 3> strides = {1, 1, 2};
                table = raw.values.stridedSlice(startReal, stop, strides).cast<Cplx>() +
                        raw.values.stridedSlice(startImag, stop, strides).cast<Cplx>() * Cplx(0, 1);

                if (expectedOrders > 0 && table.dimension(0) < expectedOrders)
                    throw IoError(filename + ": the OTF holds " + std::to_string(table.dimension(0)) +
                                  " order" + (table.dimension(0) == 1 ? "" : "s") + ", and this configuration resolves " +
                                  std::to_string(expectedOrders) +
                                  (p != nullptr ? " (nphases " + std::to_string(p->nphases) + ", norders " +
                                                      std::to_string(p->norders) + ")"
                                                : "") +
                                  " -- " + got);
            }

            rep.norders = static_cast<int>(table.dimension(0));
            rep.nkr = static_cast<int>(table.dimension(1));
            rep.nzotf = static_cast<int>(table.dimension(2));

            // --- the kz origin, and the structural check that finds it -------
            OtfKzOrigin wanted = opts.kzOrigin;
            const auto sidecarPath = opts.readSidecar ? findOtfSidecar(filename) : std::string();
            OtfSidecar sidecar;
            if (!sidecarPath.empty()) {
                sidecar = readOtfSidecar(sidecarPath);
                rep.sidecarPath = sidecarPath;
                if (sidecar.hasKzOrigin && wanted == OtfKzOrigin::Auto) wanted = sidecar.kzOrigin;
            }
            rep.hermitianErrorAsStored = hermitianKzError(table, 0);
            rep.hermitianErrorApplied = rep.hermitianErrorAsStored;
            if (rep.nzotf >= 3) {
                const double tol = opts.hermitianTolerance;
                const double e0 = rep.hermitianErrorAsStored;
                // Deciding the roll and validating the result are two steps.
                // Auto decides by the symmetry; DcFirst and DcLast are
                // statements the caller (or a sidecar) makes, and a statement
                // can be wrong, so the check below runs on what was read
                // either way.
                if (wanted == OtfKzOrigin::DcLast) {
                    rep.kzRoll = 1;
                } else if (wanted == OtfKzOrigin::DcFirst || !(tol > 0.0)) {
                    rep.kzRoll = 0;
                } else {
                    const double ep = hermitianKzError(table, 1);
                    const double em = hermitianKzError(table, -1);
                    if (e0 <= tol) rep.kzRoll = 0;
                    else if (ep <= tol) rep.kzRoll = 1;
                    else if (em <= tol) rep.kzRoll = -1;
                    else
                        throw IoError(
                            filename + ": the kz axis of this table is not Hermitian, so it is not a radially "
                                       "averaged OTF as makeotf writes one (which forces t(kz) = conj(t(-kz)) "
                                       "exactly). Mean |t(kz) - conj(t(-kz))| / mean |t| is " +
                            num(e0) + " as stored, " + num(ep) + " shifted one plane later and " + num(em) +
                            " one plane earlier, against a tolerance of " + num(tol) + ". Read as " +
                            shapeOf(rep.norders, rep.nkr, rep.nzotf) + " (norders, nkr, nzotf) from " +
                            shapeOf(d0, d1, d2) +
                            ". A plain 2D OTF image, if that is what this is, must be a single square page.");
                }
                if (rep.kzRoll != 0) {
                    table = rollKz(table, rep.kzRoll);
                    rep.kzRolled = true;
                    rep.hermitianErrorApplied = hermitianKzError(table, 0);
                    rep.notes.push_back(
                        "the kz axis was rotated by " + std::to_string(rep.kzRoll) +
                        " plane to put kz = 0 first: Hermitian error " + num(rep.hermitianErrorAsStored) +
                        " as stored, " + num(rep.hermitianErrorApplied) + " after" +
                        (wanted == OtfKzOrigin::DcLast ? " (asked for: DC last)" : " (detected, not assumed)"));
                }
                if (tol > 0.0 && rep.hermitianErrorApplied > tol)
                    throw IoError(filename + ": read with kz " +
                                  (rep.kzRoll == 0 ? std::string("as stored")
                                                   : "rotated by " + std::to_string(rep.kzRoll) + " plane") +
                                  ", as asked for, the table is not Hermitian in kz (mean |t(kz) - conj(t(-kz))| "
                                  "/ mean |t| is " + num(rep.hermitianErrorApplied) + " against a tolerance of " +
                                  num(tol) + "), which a radially averaged OTF always is. Leave the kz origin on "
                                  "Auto to have it detected, or set hermitianTolerance to 0 to read the file "
                                  "anyway.");
            } else if (rep.layout == OtfLayout::CudasireconRadial) {
                rep.notes.push_back("only " + std::to_string(rep.nzotf) +
                                    " kz plane(s): the Hermitian check that validates the kz axis has nothing to compare");
            }

            // --- sampling -----------------------------------------------------
            if (explicitSteps) {
                rep.dkrotf = dkrExplicit;
                rep.dkzotf = dkzExplicit;
                rep.samplingSource = OtfSamplingSource::ExplicitArgument;
            } else if (raw.mrc && raw.cellDkr > 0.0 && raw.cellDkz > 0.0) {
                rep.dkrotf = raw.cellDkr;
                rep.dkzotf = raw.cellDkz;
                rep.samplingSource = OtfSamplingSource::FileCellDimensions;
                rep.notes.push_back("sampling from the container's own cell lengths: dkr " + num(rep.dkrotf) +
                                    ", dkz " + num(rep.dkzotf) + " 1/um, i.e. measured at dx " +
                                    num(1.0 / (rep.dkrotf * static_cast<double>(rep.nkr - 1) * 2.0), 4) +
                                    " um, dz " + num(1.0 / (rep.dkzotf * static_cast<double>(rep.nzotf)), 4) + " um");
            } else if (!sidecarPath.empty() && (sidecar.dkr > 0.0 || sidecar.xyres > 0.0)) {
                const double dkr = sidecar.dkr > 0.0
                                       ? sidecar.dkr
                                       : 1.0 / (sidecar.xyres * static_cast<double>(rep.nkr - 1) * 2.0);
                double dkz = sidecar.dkz;
                if (!(dkz > 0.0) && sidecar.zres > 0.0) dkz = 1.0 / (sidecar.zres * static_cast<double>(rep.nzotf));
                if (!(dkz > 0.0) && p != nullptr && p->dz_psf > 0.0)
                    dkz = 1.0 / (p->dz_psf * static_cast<double>(rep.nzotf));
                if (!(dkz > 0.0)) dkz = 1.0;
                rep.dkrotf = dkr;
                rep.dkzotf = dkz;
                rep.samplingSource = OtfSamplingSource::Sidecar;
                rep.notes.push_back("sampling from the sidecar " + sidecarPath + ": dkr " + num(dkr) + ", dkz " +
                                    num(dkz) + " 1/um");
            } else {
                if (p == nullptr) throw IoError(filename + ": no sampling for this OTF and no parameters to derive it from");
                if (rep.nkr < 2)
                    throw IoError("Radial OTF needs at least 2 radial samples: " + filename);
                if (!(p->dx > 0.0))
                    throw IoError(filename + ": cannot derive the OTF's radial step from a pixel size of " + num(p->dx));
                rep.dkrotf = 1.0 / (p->dx * static_cast<double>(rep.nkr - 1) * 2.0);
                rep.dkzotf = p->dz_psf > 0.0 ? 1.0 / (p->dz_psf * static_cast<double>(rep.nzotf)) : 1.0;
                rep.samplingSource = OtfSamplingSource::DerivedFromParameters;
                rep.notes.push_back(
                    "the file states no sampling and there is no sidecar, so dkr " + num(rep.dkrotf) + " and dkz " +
                    num(rep.dkzotf) + " 1/um were DERIVED from this run's xyres " + num(p->dx) + " and zresPSF " +
                    num(p->dz_psf) + ": they are right only if the OTF was measured at those pixel sizes");
            }
            if (!(rep.dkrotf > 0.0) || !std::isfinite(rep.dkrotf) || !(rep.dkzotf > 0.0) || !std::isfinite(rep.dkzotf))
                throw IoError(filename + ": the OTF's sampling is not usable (dkr " + num(rep.dkrotf) + ", dkz " +
                              num(rep.dkzotf) + " 1/um)");
            if (rep.samplingSource == OtfSamplingSource::FileCellDimensions && p != nullptr && p->dx > 0.0) {
                const double fileDx = 1.0 / (rep.dkrotf * static_cast<double>(rep.nkr - 1) * 2.0);
                const double rel = std::abs(fileDx - p->dx) / p->dx;
                if (rel > 0.001)
                    rep.notes.push_back("the OTF was measured at dx " + num(fileDx, 6) + " um and this run's xyres is " +
                                        num(p->dx, 6) + " um (" + num(100.0 * rel, 3) +
                                        "% apart); the file's own sampling is used, which is what makes the two consistent");
            }

            // --- the scale ----------------------------------------------------
            rep.storedOrder0Dc = table(0, 0, 0);
            bool lineOk = false;
            rep.lineFitValue = lineFitToOrigin(table, opts.lineFitFirst, opts.lineFitLast, &lineOk);
            double divisor = 1.0;
            if (opts.normalization != OtfNormalization::AsStored) {
                const double dc = rep.storedOrder0Dc.real();
                const bool dcUsable = std::isfinite(dc) && dc > 0.0;
                double ref = 1.0;
                if (dcUsable) {
                    ref = dc;
                    rep.reference = OtfNormalizationReference::Order0Dc;
                } else if (lineOk && rep.lineFitValue > 0.0) {
                    ref = rep.lineFitValue;
                    rep.reference = OtfNormalizationReference::Order0LineFit;
                    rep.notes.push_back(
                        "order 0's DC is " + num(dc) +
                        ", which cannot be a scale, so the reference is the fixorigin line fit over kr " +
                        std::to_string(std::max(1, opts.lineFitFirst)) + ".." + std::to_string(opts.lineFitLast) +
                        " extrapolated to kr = 0: " + num(rep.lineFitValue));
                } else {
                    rep.notes.push_back("no normalisation reference could be established (order 0's DC is " + num(dc) +
                                        " and the line fit did not resolve), so the table is used as stored and an "
                                        "absolute otfcutoff is not meaningful against it");
                }
                rep.referenceValue = ref;
                const bool already = std::abs(ref - 1.0) <= opts.normalizedTolerance;
                if (rep.reference != OtfNormalizationReference::None &&
                    (opts.normalization == OtfNormalization::Reference || !already)) {
                    divisor = ref;
                    const double inv = 1.0 / divisor;
                    for (Idx o = 0; o < table.dimension(0); ++o)
                        for (Idx ir = 0; ir < table.dimension(1); ++ir)
                            for (Idx j = 0; j < table.dimension(2); ++j) table(o, ir, j) *= inv;
                    rep.normalizationApplied = true;
                    rep.notes.push_back("every order divided by " + num(divisor) +
                                        " to put order 0's kr = kz = 0 sample at 1; the ratios between orders, which "
                                        "are the modulation depth, are unchanged");
                } else if (already && rep.reference != OtfNormalizationReference::None) {
                    rep.notes.push_back("already at the reference scale (order 0's DC is " + num(ref) +
                                        "), so nothing was rescaled");
                }
            } else {
                rep.referenceValue = 1.0;
                rep.notes.push_back("read as stored: no normalisation, so an absolute otfcutoff means whatever this "
                                    "file's own scale makes it mean");
            }

            // A scale is only as good as the sample it came from. makeotf
            // divides by order 0's kr = kz = 0 sample whether or not that
            // sample is signal: on the user's 488OTF.tif the kz = 0 profile
            // runs 1, 0.964, 0.738, 3.779, 0.809, -1.116, so the file's DC
            // sits among samples that are plainly noise and the whole table's
            // scale rests on it. Say so rather than silently re-deriving it.
            if (rep.reference == OtfNormalizationReference::Order0Dc && rep.nkr > 5) {
                const double dcMag = std::abs(table(0, 0, 0));   // after any division, so this is scale free
                int above = 0, negative = 0;
                const Idx last = std::min<Idx>(rep.nkr - 1, std::max(5, opts.lineFitLast));
                double worst = 0.0;
                for (Idx ir = 1; ir <= last; ++ir) {
                    const double v = table(0, ir, 0).real();
                    if (std::abs(v) > dcMag) ++above;
                    if (v < 0.0) ++negative;
                    worst = std::max(worst, std::abs(v));
                }
                if (above > 0 || negative > 0)
                    rep.notes.push_back(
                        "order 0's low-kr samples at kz = 0 are not a decaying profile (" + std::to_string(above) +
                        " of the first " + std::to_string(last) + " exceed the DC, " + std::to_string(negative) +
                        " are negative, largest magnitude " + num(worst) + " against a DC of " + num(dcMag) +
                        "): this table's scale was set from a sample sitting among noise, and the OTF wants "
                        "re-measuring rather than re-normalising");
            }

            if (reportOut != nullptr) *reportOut = rep;
            return OTFRadiallyAveraged(std::move(table), rep.dkrotf, rep.dkzotf, divisor);
        }
    } // namespace

    std::string OtfLoadReport::summary() const {
        std::ostringstream os;
        os << file << ": "
           << (layout == OtfLayout::Plain2dImage ? "plain 2D OTF image" : "cudasirecon radial table") << " "
           << shapeOf(storedShape[0], storedShape[1], storedShape[2]) << " -> " << norders << " orders x " << nkr
           << " kr x " << nzotf << " kz; dkr " << num(dkrotf) << " dkz " << num(dkzotf) << " 1/um (";
        switch (samplingSource) {
            case OtfSamplingSource::ExplicitArgument: os << "given by the caller"; break;
            case OtfSamplingSource::FileCellDimensions: os << "the file's cell lengths"; break;
            case OtfSamplingSource::Sidecar: os << "sidecar " << sidecarPath; break;
            case OtfSamplingSource::DerivedFromParameters: os << "DERIVED from this run's pixel sizes"; break;
        }
        os << "); kz " << (kzRolled ? "rotated by " + std::to_string(kzRoll) + " plane" : "as stored")
           << " (Hermitian error " << num(hermitianErrorApplied) << "); order 0 DC " << num(storedOrder0Dc.real())
           << ", ";
        if (normalizationApplied) os << "every order divided by " << num(referenceValue);
        else if (normalizationMode == OtfNormalization::AsStored) os << "read as stored";
        else os << "already at the reference scale";
        return os.str();
    }

    std::string findOtfSidecar(const std::string& filename) {
        namespace fs = std::filesystem;
        std::error_code ec;
        const fs::path p(filename);
        const fs::path withSuffix(filename + ".toml");
        if (fs::is_regular_file(withSuffix, ec)) return withSuffix.string();
        fs::path replaced = p;
        replaced.replace_extension(".toml");
        if (replaced != p && fs::is_regular_file(replaced, ec)) return replaced.string();
        return std::string();
    }

    OtfSidecar readOtfSidecar(const std::string& path) {
        OtfSidecar s;
        toml::table doc;
        try {
            doc = toml::parse_file(path);
        } catch (const toml::parse_error& e) {
            throw IoError(path + ": not a readable OTF sidecar (" + std::string(e.description()) + ")");
        }
        const auto grab = [&](auto where) {
            if (auto v = where["dkr"].template value<double>()) s.dkr = *v;
            if (auto v = where["dkz"].template value<double>()) s.dkz = *v;
            if (auto v = where["xyres"].template value<double>()) s.xyres = *v;
            if (auto v = where["zres"].template value<double>()) s.zres = *v;
            if (auto v = where["kz_origin"].template value<std::string>()) {
                s.hasKzOrigin = true;
                if (*v == "dc_first") s.kzOrigin = OtfKzOrigin::DcFirst;
                else if (*v == "dc_last") s.kzOrigin = OtfKzOrigin::DcLast;
                else if (*v == "auto") s.kzOrigin = OtfKzOrigin::Auto;
                else throw IoError(path + ": kz_origin must be \"dc_first\", \"dc_last\" or \"auto\", not \"" + *v + "\"");
            }
        };
        if (auto v = doc["dkr"].value<double>()) s.dkr = *v;
        if (auto v = doc["dkz"].value<double>()) s.dkz = *v;
        if (auto v = doc["xyres"].value<double>()) s.xyres = *v;
        if (auto v = doc["zres"].value<double>()) s.zres = *v;
        if (auto v = doc["kz_origin"].value<std::string>()) {
            s.hasKzOrigin = true;
            if (*v == "dc_first") s.kzOrigin = OtfKzOrigin::DcFirst;
            else if (*v == "dc_last") s.kzOrigin = OtfKzOrigin::DcLast;
            else if (*v == "auto") s.kzOrigin = OtfKzOrigin::Auto;
            else throw IoError(path + ": kz_origin must be \"dc_first\", \"dc_last\" or \"auto\", not \"" + *v + "\"");
        }
        grab(doc["sampling"]);
        grab(doc["psf"]);
        if (s.dkr < 0.0 || s.dkz < 0.0 || s.xyres < 0.0 || s.zres < 0.0)
            throw IoError(path + ": a sidecar's steps and pixel sizes must be positive");
        return s;
    }

    OTFRadiallyAveraged loadOTF(const std::string& filename, double dkrotf, double dkzotf) {
        return loadImpl(filename, nullptr, dkrotf, dkzotf, true, OtfLoadOptions{}, nullptr);
    }

    OTFRadiallyAveraged loadOTF(const std::string& filename, const SIMParameters& p) {
        return loadImpl(filename, &p, 0.0, 0.0, false, OtfLoadOptions{}, nullptr);
    }

    OTFRadiallyAveraged loadOTF(const std::string& filename, const SIMParameters& p, const OtfLoadOptions& opts,
                                OtfLoadReport* report) {
        return loadImpl(filename, &p, 0.0, 0.0, false, opts, report);
    }

    OTFRadiallyAveraged loadOTF(const std::string& filename, double dkrotf, double dkzotf, const OtfLoadOptions& opts,
                                OtfLoadReport* report) {
        return loadImpl(filename, nullptr, dkrotf, dkzotf, true, opts, report);
    }

} // namespace sirius
