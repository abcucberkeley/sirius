#ifndef SIRIUS_OTF_MEASURE_HPP
#define SIRIUS_OTF_MEASURE_HPP

// Measuring an OTF from a bead stack: the calibration artefact sirius/otf_io.hpp
// reads and sirius/otf_ideal.hpp otherwise has to stand in for.
//
// This is not a pipeline operation and deliberately so: no operation in the
// tree writes a file, and a StepOutput is an image on the dataset's own grid,
// where a reciprocal-space (order, kr, kz) table has nowhere to live. The
// pattern copied here is app/core/training_export.hpp -- a pure function, an
// options struct, a validate() that answers before anything runs, and a result
// struct carrying what a user needs to judge the artefact.
//
// --- ONE BEAD: THE makeotf-COMPATIBLE PATH --------------------------------
//
// measureOTF with OtfMeasureOptions::field off is cudasirecon's makeotf
// (extern/cusimfixed/src/otf/radialft.cpp) step for step, in the same order,
// so its output is interchangeable with makeotf's tables:
//
//   1. per-section border-mean background          (estimate_background, border 20)
//   2. lateral edge-blend apodization, 10 px, no axial taper   (apodize)
//   3. the bead's centre from the global maximum of the phase-averaged
//      volume, to sub-pixel by three parabola fits     (determine_center)
//   4. background subtraction, then the analytic cos/sin band separation
//      (makematrix / separate; sirius::separationMatrix is the same matrix
//      up to the 1/nphases this applies, and that factor cancels in step 10)
//   5. ONE batched real-to-complex 3D transform of all the bands
//   6. a phase ramp putting the bead at the origin              (shift_center)
//   7. the finite-bead-size division                     (beadsize_compensate)
//   8. the radial average onto (order, kr, kz) with dkr = 1 / (min(nx,ny) dxy)
//      and Hermitian symmetry forced along kz                      (radialft)
//   9. the kr = 0 column replaced by the kr = 1 column, except order 0's
//      kz = 0 sample                                                 (modify)
//  10. the normalisation -- one divisor for every order, so the ratios
//      between orders, which are the modulation depth, are untouched
//      (fixorigin + rescale; see THE SCALE below)
//  11. the side band's constant phase rotated into the real part (combine_reim)
//
// writeMeasuredOTF then writes it in makeotf's own TIFF layout (norders pages
// of nkr rows by 2 * nzotf columns, kz fastest, real and imaginary
// interleaved) plus the .toml sidecar loadOTF reads, because that layout
// carries no sampling at all and deriving dkr from whatever pixel size a later
// run happens to use is the failure of findings 9k.48.
//
// --- A FIELD OF BEADS -----------------------------------------------------
//
// makeotf handles exactly one bead: determine_center takes the global maximum
// and one phase ramp is applied to the whole volume. Every other bead is then
// an off-origin delta whose own phase ramp survives the radial average as
// interference, and mmmSIM has the same limitation. With field on, this
// follows mcSIM's bead route (extern/mcsim,
// calibration/sim_modulation_depth_from_beads.py): a difference-of-Gaussians
// bandpass, local maxima with a minimum separation, a per-region Gaussian fit,
// then filters on amplitude, width bounds and distance from the boundary. Each
// kept bead is then masked out of the separated bands on the FULL lateral grid
// -- so dkr is the field's, not a small ROI's -- given ITS OWN phase ramp, and
// radially averaged; the per-bead tables are normalised and averaged.
//
// Measured, on five planted beads whose answer is known in closed form: the
// field path reproduces the one-bead table to 4.2e-5 where the single-bead
// path is 0.748 out, and recovers the band ratio as 0.35000 against a planted
// m/2 of 0.35 (tests/test_otf_measure.cpp). On the user's own sparse bead
// field it buys nothing, and for a reason worth knowing rather than a defect:
// that acquisition is a 128 x 128 camera ROI (Left 321, Top 1089, Right 448,
// Bottom 1216) centred on ONE bead, so detection finds one bead above the
// amplitude floor in each channel -- 1 of 10 candidates at 488, 1 of 20 at
// 560 -- and the two paths then agree to 0.06% on the band ratio, which is
// the right answer for a one-bead field rather than an improvement on it.
//
// Two details decide whether this works at all:
//   * the phase ramp has to be removed per bead BEFORE averaging, which is the
//     whole reason a field cannot go through makeotf, and
//   * combine_reim has to run per bead too, because the side band's constant
//     phase is the illumination phase AT THAT BEAD'S POSITION. A field of
//     beads sits at different pattern phases by construction, so averaging the
//     complex side bands first would cancel them against each other.
//
// --- THE SCALE: WHAT IT IS WORTH, AND WHAT IS WORTH READING OFF A TABLE ----
//
// makeotf's rescale() divides every order by order 0's kr = kz = 0 sample.
// That sample is the integral of the background-subtracted band 0 -- the whole
// light the bead delivered -- so it is a difference of two nearly equal large
// numbers: on the user's own sparse bead field (RAW_488_3phase_ols20px_3G,
// 128 x 128 x 101 x 3 phases) it is 7.97e6 against a stack integral of 5.57e8,
// so 98.6% of the signal is the subtracted background. Measured on that stack,
// changing nothing but the background handed to the same code:
//
//   background per section   order 1's kr = kz = 0 sample as stored
//   105                           0.297379
//   107.354 (the border mean)     0.442808
//   109.889                       0.935342
//   112                          12.702596
//
// A 2.5 ADU move -- 2.4% of the level -- doubles the table. This is not a
// defect of this code or of makeotf's: the DC is a physical quantity that a
// window centred on a bright bead cannot measure well, because the PSF's
// out-of-focus wings and the background are the same thing to a border-mean
// estimator. OtfMeasureResult::scaleSensitivityPerAdu puts a number on it for
// the stack in hand, and backgroundBorderMean / backgroundDarkest give the two
// estimates side by side.
//
// IT IS WHY OUR MEASUREMENT AND THE COLLEAGUE'S OTF_488.tif DIFFER, and by how
// much: with makeotf's defaults this code reproduces that file sample for
// sample to 1.9e-7 of peak in BOTH orders after one global factor of 2.112,
// and that factor is the ratio of the two runs' DC, nothing else.
//
// WHAT SURVIVES IT. Every order is divided by the SAME number, so the ratio of
// one order to another at the same (kr, kz) does not move. On those two tables
// order 1 / order 0 at kz = 0 runs 0.59373, 0.63026, 0.64048, 0.63356 from
// kr = 1 and agrees to five decimals -- it has to, since the tables agree
// sample for sample to 1.9e-7 once one scale is taken out -- while "the
// modulation depth", read as order 1's own kr = 0 sample, says 0.442808 in
// ours and 0.209638 in theirs. The same contrast appears between the two
// background estimators on our own run: 0.442808 from the border mean and
// 0.128922 from the darkest tenth, with bandRatio 0.664844 either way.
// So OtfMeasureResult::bandRatio, the median of |order 1| / |order 0| over
// kr >= 1, is the number to quote, and modulationDepth is kept beside it only
// because it is what a reader of the file sees.
//
// WHY NOT NORMALISE SOMEWHERE ELSE. Two candidates were measured and both are
// worse than the DC, which is the answer the design this implements started
// from and the answer it ends at:
//   * makeotf's own -fixorigin (OtfMeasureScale::MakeotfFixOrigin) is
//     background independent -- exactly, to 3e-16, because the background
//     enters only the kx = ky = 0 column, step 9 has replaced that column
//     everywhere but order 0's kz = 0, and the fit is taken over kr >= 1 --
//     but it is NOT an estimate of the DC. Its sum[0] is the DC alone while
//     its sum[i >= 1] are kz SUMS (radialft.cpp: "don't want to add up
//     garbages on kz axis"), so the line it extrapolates lives on the kz-sum
//     scale and the table comes out scaled by roughly nz / (sqrt(2 pi) sigma_z)
//     against the DC convention: 3.07x on the user's stack, and 4.63x on a
//     synthetic Gaussian PSF where the right table is known in closed form
//     (tests/test_otf_measure.cpp). Since loadOTF's otfcutoff is an absolute
//     threshold and the Wiener constant scales as 1/s^2, that is a different
//     filter, not a different label. It is offered because it is makeotf's
//     own repair of the low-kr samples, and it is not the default.
//   * extrapolating order 0's kz = 0 profile to kr = 0 is on the right scale
//     but cannot see the DC: that profile runs 1, 0.3531, 0.2349, 0.1940 on
//     the user's stack, i.e. the DC is a spike narrower than one radial
//     sample, because the DC is the integral of the whole PSF including the
//     haze while kr = 1 already resolves it away. A line through kr = 2..9
//     extrapolates to 0.256 of the DC.
// So the default is OtfMeasureScale::Order0Dc: makeotf's, which is what
// loadOTF, idealOTF and the 0.006 otfcutoff are all calibrated against, with
// the softness measured and reported rather than papered over. One cause of
// that softness IS fixable and the result says so when it bites: makeotf's
// background border is 20 px whatever the section is, which on the user's
// 128 x 128 ROI is 51.7% of the image and holds the bead's own haze, so the
// estimate over-subtracts. BackgroundEstimate::DarkestFraction is the
// alternative, and both numbers are reported whichever was used.

#include <array>
#include <complex>
#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include "sirius/constants.hpp"
#include "sirius/otf.hpp"
#include "sirius/sim_parameters.hpp"

namespace sirius {

    // makeotf's `-ls` default (radialft.cpp: `float linespacing = 0.2f`). It
    // is a DEFAULT and not a property of any instrument: the iSOAR2 configs
    // this project runs say ls=0.504
    // (tests/data/isoar2_mount2a_2026-04-21_488.cfg), where 0.2 inflates
    // order 1 by about 1.38. Named here because the measurement falls back to
    // it when the caller states no line spacing, and a fallback that is not
    // named is how a wrong constant ends up in a result with nothing saying so.
    inline constexpr double kMakeotfLineSpacingUm = 0.2;

    // Where the phases sit along a raw stack's section axis. The iSOAR2
    // RAW_* stacks are PhaseFastest, and their slicelist.sqlite3 says so
    // independently: three rows share each Slice_Index.
    enum class BeadPhasePacking {
        PhaseFastest,   // section = z * nphases + phase  (makeotf's own reading order)
        PhaseSlowest    // section = phase * nz + z
    };

    // What the table is divided by. One divisor for every order in all three
    // cases, so no choice here changes the ratio of one order to another.
    enum class OtfMeasureScale {
        Order0Dc,          // order 0's kr = kz = 0 sample: makeotf's rescale(), the default
        MakeotfFixOrigin,  // makeotf's -fixorigin repair, which also rescales -- see the header
        AsMeasured         // no division at all
    };

    // How the per-section background is estimated when none is stated.
    enum class BackgroundEstimate {
        BorderMean,       // makeotf's: the mean of a border frame of each section
        DarkestFraction   // the mean of each section's darkest fraction, which a
                          // window centred on a bright bead needs because its
                          // border frame still holds the PSF's haze
    };

    // Why a bead candidate was not used.
    enum class BeadRejection {
        None,
        Amplitude,       // below the amplitude floor
        TooClose,        // a brighter candidate is within the minimum separation
        NearBoundary,    // too close to a lateral or axial edge for its ROI
        WidthTooSmall,   // narrower than sigmaMinUm: hot pixel or cosmic ray
        WidthTooLarge,   // wider than sigmaMaxUm: an aggregate or two beads
        FitFailed,       // the Gaussian fit did not converge inside its ROI
        Saturated,       // at or above saturationLevel
        // The last two are not the detector's judgement of a bead but a cap
        // on how many it looks at. They exist so that the inventory adds up:
        // every candidate the detector found lands in exactly one bucket, and
        // `None` means kept and nothing else. Before them, a bead the
        // maxBeads cap left out carried `None` -- the inventory printed
        // "kept" beside a bead that was not kept -- and the candidates past
        // the examination cap were counted in `found` and in no bucket at all.
        OverMaxBeads,    // passed every filter, but maxBeads were already kept
        NotExamined      // above the amplitude floor, past the 4*maxBeads+16 examination cap
    };
    const char* beadRejectionName(BeadRejection r) noexcept;
    // One past the last BeadRejection: the width of OtfMeasureResult::rejected.
    inline constexpr std::size_t kBeadRejectionCount = 10;

    // A candidate from the field path. Positions are in voxels of the input
    // grid, widths in voxels, and `kept` says whether it reached the average.
    struct BeadFit {
        double x = 0.0, y = 0.0, z = 0.0;
        double amplitude = 0.0;         // the fitted peak above its local offset
        double offset = 0.0;            // the fitted local background
        double sigmaX = 0.0, sigmaY = 0.0, sigmaZ = 0.0;
        double residual = 0.0;          // rms of the fit residual over the fitted amplitude
        double nearestNeighbourPx = 0.0;   // 0 when it is the only candidate
        bool kept = false;
        BeadRejection rejection = BeadRejection::None;
    };

    // mcSIM's filter set, in micrometres so the numbers mean the same thing on
    // any pixel size.
    struct BeadDetectionOptions {
        // difference-of-Gaussians bandpass: the small sigma is the bead, the
        // large one the haze to remove
        double dogSmallLateralUm = 0.1, dogSmallAxialUm = 0.1;
        double dogLargeLateralUm = 5.0, dogLargeAxialUm = 5.0;
        // local maxima
        double minSeparationLateralUm = 1.0, minSeparationAxialUm = 0.0;
        // the amplitude floor: the larger of minAmplitude and
        // minAmplitudeFraction x the brightest candidate
        double minAmplitude = 0.0;
        double minAmplitudeFraction = 0.05;
        double saturationLevel = 0.0;      // > 0: candidates at or above it are dropped
        // the per-bead region, and the margin it needs from every edge
        double roiLateralUm = 1.5;         // full width of the fitted / masked box
        double roiAxialUm = 0.0;           // 0: the whole z range
        double boundaryMarginLateralUm = 1.0, boundaryMarginAxialUm = 0.0;
        // width bounds on the fit
        double sigmaMinLateralUm = 0.05, sigmaMaxLateralUm = 0.2;
        double sigmaMinAxialUm = 0.0, sigmaMaxAxialUm = 0.0;   // 0: unbounded
        int maxBeads = 64;                 // after sorting by amplitude
        double maxResidual = 0.0;          // > 0: reject a fit worse than this
    };

    struct OtfMeasureOptions {
        // --- the acquisition
        int nphases = 3;
        int norders = 0;                   // 0 = (nphases + 1) / 2, as makeotf derives it
        BeadPhasePacking packing = BeadPhasePacking::PhaseFastest;
        std::vector<double> phases;        // empty = equally spaced from 0
        double dxy = 0.0;                  // um, required
        double dz = 0.0;                   // um, required

        // --- makeotf's preprocessing
        double background = -1.0;          // >= 0: use this value for every section
        BackgroundEstimate backgroundEstimate = BackgroundEstimate::BorderMean;
        int backgroundBorder = 20;         // px, makeotf's hard-coded border_size
        double darkestFraction = 0.1;      // for BackgroundEstimate::DarkestFraction
        int apodize = 10;                  // px of lateral edge blend; 0 = none

        // --- the finite bead size
        double beadDiameterUm = 0.12;      // 0 = no compensation
        // THE ILLUMINATION LINE SPACING, um. It is one quantity under three
        // names -- makeotf's `-ls`, SIMParameters::linespacing_um and this --
        // and the division uses it exactly as makeotf does: order n is
        // divided by the sphere's transform at |k + n/(norders-1)/period|,
        // which is SIMParameters::patternFundamental's own arithmetic. It is
        // an INSTRUMENT number, not a preference: on the iSOAR2 stacks the
        // spacing is 0.504 um, and measuring them with makeotf's 0.2 um
        // default divides order 1 by 0.687 instead of 0.945 -- order 1 comes
        // out about 1.38x too large, uniformly enough that no shape in the
        // table says anything is wrong.
        //
        // 0 = NOT STATED. Then the division falls back to
        // kMakeotfLineSpacingUm, OtfMeasureResult::patternPeriodUsedUm says
        // what was used, patternPeriodStated says it was a fallback, and
        // notes carries a line to that effect. setIllumination() below is how
        // a caller that has the acquisition's SIM parameters states it
        // instead of a caller guessing.
        double patternPeriodUm = 0.0;
        double patternAngleRad = 1.57;     // makeotf's `-angle`, radians
        // The pixel sizes the DIVISION uses, where they are not the
        // acquisition's. makeotf has one dr for everything, but for a square
        // section dr cancels out of the radial binning entirely (the bin is
        // rint(|k| / dkr) with dkr = 1 / (min(nx,ny) dr) and |k| in units of
        // 1 / (nx dr)), so the only step that reads it is this division -- and
        // makeotf's dr DEFAULTS to 0.106 um, which is nobody's pixel. Reaching
        // an existing table therefore needs the number makeotf used here while
        // dkr still needs the real one. Measured on the user's sparse bead
        // field: 0.106 reproduces the colleague's table to 1.9e-7 of peak,
        // the instrument's own 0.0855263 to 3.5e-3. 0 = use dxy / dz.
        double beadCompensationPixelUm = 0.0;
        double beadCompensationAxialUm = 0.0;

        // --- the table
        OtfMeasureScale scale = OtfMeasureScale::Order0Dc;
        int lineFitFirst = 2, lineFitLast = 9;   // makeotf's -fixorigin window, radial samples
        // the radial samples the band ratio is taken over; the lower bound
        // skips kr = 0, which step 9 has replaced, and the upper one stops
        // where order 0 is noise
        double bandRatioMinOrder0 = 0.02;  // fraction of order 0's peak a sample must clear
        bool repairKr0Column = true;       // makeotf's modify()
        bool combineReIm = true;           // makeotf's combine_reim()

        // --- the field path
        bool field = false;
        bool perBeadNormalise = true;      // each bead's table on its own scale before averaging;
                                           // off makes the average brightness-weighted
        BeadDetectionOptions detect;

        // The illumination the bead stack was taken under, from the SIM
        // parameters a caller that reconstructs this instrument already has
        // (fromLegacy of the cudasirecon config states both numbers: ls and
        // k0angles / k0startangle). ONE direction, because an OTF is measured
        // per direction, and makeotf takes one `-angle`.
        //
        // What it does NOT set: nphases, norders and the pixel sizes. Those
        // come from the stack in front of the measurement, which for a
        // montage or a re-binned acquisition is not what a run's parameters
        // say, and silently taking them from here would put the layout back
        // in two places. The illumination is different: the pattern is a
        // property of the instrument and nothing in a bead stack states it.
        void setIllumination(const SIMParameters& p, int dir = 0) noexcept {
            patternPeriodUm = p.linespacing_um;
            if (p.k0_angles && dir >= 0 && static_cast<std::size_t>(dir) < p.k0_angles->size())
                patternAngleRad = (*p.k0_angles)[static_cast<std::size_t>(dir)];
            else
                patternAngleRad = p.k0_start_angle + (dir > 0 ? dir : 0) * kPi / (p.ndirs > 0 ? p.ndirs : 1);
        }
    };

    struct OtfMeasureResult {
        OTFRadiallyAveraged otf;           // (norders, nkr, nzotf), kz in FFT order, DC first
        int norders = 0, nkr = 0, nzotf = 0;
        int nz = 0, ny = 0, nx = 0;        // the de-interleaved stack it was measured from
        double dkr = 0.0, dkz = 0.0;       // 1/um
        double dxy = 0.0, dz = 0.0;        // um, what the measurement was made at
        // The line spacing the finite-bead-size division actually used, and
        // whether the caller stated it. 0 when nothing was divided out
        // (beadDiameterUm 0, or a single order, where there is no side band
        // to offset). A table whose patternPeriodStated is false was measured
        // on a guess, which is the one thing a reader of an OTF cannot see
        // from the samples.
        double patternPeriodUsedUm = 0.0;
        bool patternPeriodStated = false;

        // --- the inventory. In single-bead mode `beads` holds the one centre
        // makeotf would have found, so the two paths report the same way.
        // `found` is every candidate the detector produced and
        // `rejected` is indexed by BeadRejection, so
        // sum(rejected) == found and rejected[None] == kept, always: the two
        // buckets that make that true are OverMaxBeads and NotExamined.
        // `beads` holds the candidates that were examined one by one, which
        // is fewer than `found` whenever a cap or the amplitude floor cut the
        // list short -- those are counted but not fitted.
        std::vector<BeadFit> beads;
        int found = 0, kept = 0;
        std::array<int, kBeadRejectionCount> rejected{};   // indexed by BeadRejection

        // --- the scale, with both candidates so the choice is visible
        // the raw kr = kz = 0 sample of order 0, averaged over the kept beads
        double order0Dc = 0.0;
        double lineFitToOrigin = 0.0;      // the fixorigin extrapolation, for comparison
        double scaleDivisor = 1.0;         // what every order was divided by
        OtfMeasureScale scaleUsed = OtfMeasureScale::Order0Dc;
        double totalSignal = 0.0;          // sum of the raw stack
        double backgroundTotal = 0.0;      // sum of what was subtracted
        double backgroundMean = 0.0, backgroundSd = 0.0;   // across sections
        double backgroundBorderMean = 0.0; // makeotf's estimator, whichever was used
        double backgroundDarkest = 0.0;    // the darkest-fraction estimator, likewise
        // order0Dc / totalSignal. Small means the scale is a difference of
        // large numbers and the DC is not worth trusting.
        double dcFractionOfSignal = 0.0;
        // The relative change in the table's scale per 1 ADU of error in the
        // background: (voxels per section * sections / nphases) / order0Dc.
        // 0.21 means a 1 ADU error moves every sample but the DC by 21%. It is
        // the whole section's count, which is what the single-bead path
        // integrates; a field path whose masks are smaller than the field is
        // less exposed than this says.
        double scaleSensitivityPerAdu = 0.0;

        // --- what the table says
        std::vector<double> orderDc;       // each order's kr = 0, kz = 0 sample as stored
        // order 1's kr = kz = 0 sample over order 0's, which is what a reader
        // of the file sees and what the background's error lands in
        double modulationDepth = 0.0;
        double modulationDepthSpread = 0.0;// sd across kept beads, field mode only
        // THE SCALE-FREE ONE: the median of |order 1| / |order 0| over the
        // samples where order 0 clears bandRatioMinOrder0 of its peak, with
        // kr = 0 left out. Two runs that disagree 2.11x on modulationDepth
        // agree to 1e-6 on this (see the header).
        double bandRatio = 0.0;
        double bandRatioIqr = 0.0;         // the 25-75 spread of the same samples
        int bandRatioSamples = 0;
        double hermitianKzError = 0.0;     // 0 by construction; a non-zero is a bug
        double beadPeakSnr = 0.0;          // the brightest bead over the background sd

        std::vector<std::string> notes;
        std::string summary() const;
    };

    // Empty when the options fit a stack of this shape, otherwise the problem,
    // so a caller can refuse before reading anything.
    std::string validateOtfMeasure(const OtfMeasureOptions& o, int sections, int ny, int nx);

    // `stack` is (sections, ny, nx) as the file stores it: sections =
    // nz * nphases, de-interleaved per OtfMeasureOptions::packing. Throws
    // std::invalid_argument with validateOtfMeasure's message when the options
    // do not fit, and std::runtime_error when the field path keeps no bead.
    OtfMeasureResult measureOTF(const Eigen::Tensor<double, 3, Eigen::RowMajor>& stack,
                                const OtfMeasureOptions& options);

    struct OtfWriteOptions {
        bool sidecar = true;        // <path>.toml, which loadOTF prefers to any derivation
        std::string note;           // a comment line in the sidecar
    };

    // Writes makeotf's layout: norders pages of nkr rows by 2 * nzotf float32
    // columns, kz fastest, real and imaginary interleaved. Returns the files
    // written, the table first.
    std::vector<std::string> writeMeasuredOTF(const std::string& path, const OtfMeasureResult& result,
                                              const OtfWriteOptions& options = {});

    // The same for any table (an idealOTF, a resampled one), when there is no
    // measurement behind it.
    std::vector<std::string> writeRadialOTF(const std::string& path, const OTFRadiallyAveraged& otf,
                                            const OtfWriteOptions& options = {});

} // namespace sirius

#endif // SIRIUS_OTF_MEASURE_HPP
