#ifndef SIRIUS_OTF_IO_HPP
#define SIRIUS_OTF_IO_HPP

// Measured OTFs: reading the radially averaged table (sirius/otf.hpp) from
// the files SIM codes write, putting it on the scale the reconstruction's
// absolute thresholds are calibrated against, and refusing what it cannot
// read instead of decoding it into something meaningless.
//
// --- THE TWO LAYOUTS -------------------------------------------------------
//
// CudasireconRadial, written by cudasirecon's makeotf (radialft.cpp): norders
// pages of nkr rows by 2*nzotf columns, kz fastest, real and imaginary parts
// interleaved along the columns, kz in FFT order (DC first, negative kz
// wrapping to the top), Hermitian in kz. In a TIFF it carries NO sampling
// metadata at all; in the MRC / DeltaVision container the cell lengths ARE
// the steps (xlen = dkz, ylen = dkr), which is how cudasirecon recovers them.
//
// Plain2dImage: one real (N, N) page centred on DC, which is what OpenSIM,
// SIM4codes and most 2D-SIM codebases distribute. It is radially averaged on
// the way in onto N/2 + 1 samples and one kz plane, and that one profile
// fills every order -- which is what idealOTF does in its 2D case too
// (otf_ideal.cpp: every order page holds the same radial profile unless the
// pattern is three-beam).
//
// Before this reader validated its input, such a page was accepted as a
// radial table: a 512 x 512 OTF image loaded as 1 order x 512 radial samples
// x 256 kz planes, passed the only format check there was (an even last
// axis), and reconstructed silently from a table whose "imaginary parts" were
// every other column of an image. Measured on OpenSIM's own SIMexpt/OTF.tif:
// read that way its Hermitian-in-kz error is 1.4 times its own mean
// magnitude, where every real makeotf table measures exactly 0.
//
// --- THE SCALE, AND WHY IT IS THE READER'S BUSINESS ------------------------
//
// Of the three places the reconstruction consumes the table, the overlap
// whitening (sim_math.hpp, overlap0Value / overlap1Value) is scale
// invariant, but
//   * the otfcutoff gate in those same two functions is an ABSOLUTE
//     threshold on |OTF| (default 0.006), and
//   * the Wiener filter (filterScale) adds a constant to a sum of |OTF|^2,
//     so scaling the table by s scales the effective Wiener constant by
//     1/s^2.
// A table on an arbitrary scale is therefore not a slightly different table,
// it is a different filter. OpenSIM's OTF.tif is uint16 with a DC of 51577
// and a minimum of 2: every sample of it clears an 0.006 cutoff, and its
// Wiener constant lands 51577^2 = 2.7e9 too small. That is the mechanism
// behind the modulation amplitudes of 2.76 / 0.17 / 0.64 measured where one
// truth was 0.85 (findings 9k.51).
//
// THE REFERENCE SCALE is order 0's kr = kz = 0 sample equal to 1, with every
// order divided by that same number so the ratios between orders -- which are
// the modulation depth -- are untouched. It is makeotf's own: rescale() takes
// scalefactor = 1/otf[0].real() from order 0 and multiplies every order by it
// (radialft.cpp:1226-1242; dorescale defaults to 0, so the -rescale flag that
// would normalise each order to its own DC is off). It is also the scale
// idealOTF is already on (otf_ideal.cpp divides by its own DC), so a measured
// and a theoretical table now mean the same thing to one otfcutoff.
//
// WHAT WAS MEASURED, because the design this implements started from another
// invariant -- "order 0's kz = 0 profile extrapolated to kr = 0 by the
// fixorigin line fit equals 1" -- and the files say otherwise:
//
//   file                         order 0 DC   kz-summed   kz=0 profile
//                                             line fit    line fit
//   tests/data/otf.tif            0.99999994    1.0300      0.5173
//   tests/data/otf.dv             1.0           --          --
//   the sparse-field OTF_488.tif  1.0           1.4524      0.2560
//   the user's 488OTF.tif         1.0           2.1153      1.9752
//   the 3D OTF in MRC form        1.0 (rolled)  1.1874      0.8583
//
// Every file is already at DC = 1 and no file is at either line fit. The
// reason is in makeotf: fixorigin is a REPAIR of the samples below kr = kx1,
// not a normalisation reference; it fits the kz-SUMMED real profile rather
// than the kz = 0 one; and it is off unless asked for (radialft.cpp:104-105
// initialise interpkr to 0, 0 and the call at :401 is gated on
// interpkr[0] > 0 -- the usage text's "default is 2 and 9" describes a
// default the code does not set). Dividing a shipped table by its line fit
// would rescale every real file we have, tests/data/otf.tif by 3%, which is
// on its own enough to break the agreement with cudasirecon that stands at
// 1.6e-6 of peak (9k.48, 9k.49).
//
// So the reference is read off the table in this order, and the report says
// which was used: order 0's DC where it is usable (makeotf's own rescale
// reference), and the fixorigin line fit extrapolated to kr = 0 only where
// the DC is not -- zeroed, repaired away, or non-finite -- which is the case
// the line fit was proposed for.
//
// OtfNormalization::Auto, the default, leaves a table that is already at the
// reference exactly as stored, so nothing that loads today changes value.

#include <array>
#include <complex>
#include <string>
#include <vector>

#include "sirius/otf.hpp"
#include "sirius/sim_parameters.hpp"

namespace sirius {

    // Which of the two layouts above a file turned out to hold.
    enum class OtfLayout {
        CudasireconRadial,   // norders x nkr x 2*nzotf, re/im interleaved
        Plain2dImage         // one real (N, N) page, radially averaged on load
    };

    // What the loader is allowed to do to the numbers in the file.
    enum class OtfNormalization {
        AsStored,    // nothing at all: the file's own numbers reach the reconstruction
        Auto,        // normalize only a table that is not already at the reference (default)
        Reference    // divide by the reference whatever the file says it is on
    };

    // Which quantity ended up as the divisor.
    enum class OtfNormalizationReference {
        None,            // nothing was divided (AsStored, or Auto on a table already at 1)
        Order0Dc,        // order 0's (kr=0, kz=0) real part -- makeotf's rescale()
        Order0LineFit    // the fixorigin line fit extrapolated to kr=0, the DC being unusable
    };

    // Where dkr and dkz came from. Only DerivedFromParameters can be wrong
    // without the file knowing: it assumes the caller's pixel sizes are the
    // ones the OTF was measured at.
    enum class OtfSamplingSource {
        ExplicitArgument,      // the caller passed the steps
        FileCellDimensions,    // an MRC / DeltaVision container's own cell lengths
        Sidecar,               // a .toml beside the file
        DerivedFromParameters  // dkr = 1/(dx*(nkr-1)*2), dkz = 1/(dz_psf*nzotf)
    };

    // Where the file puts kz = 0 along its kz axis.
    enum class OtfKzOrigin {
        Auto,      // decide by Hermitian symmetry and report what was decided
        DcFirst,   // the FFT-order convention makeotf writes
        DcLast     // the axis is rotated by one, DC in the last plane
    };

    struct OtfLoadOptions {
        OtfNormalization normalization = OtfNormalization::Auto;

        // makeotf's -fixorigin window, in radial samples, used for the line
        // fit when order 0's DC cannot serve as the reference.
        int lineFitFirst = 2;
        int lineFitLast = 9;

        // |reference - 1| at or below which Auto calls a table normalized
        // already and changes nothing. 1e-5 passes a float32 1.0 (otf.tif's
        // DC is 0.99999994, 6e-8 away) and fails a table on another scale.
        double normalizedTolerance = 1e-5;

        // Orders the reconstruction will ask for. A file with fewer is
        // refused here, where the shape can be quoted, rather than deeper in.
        // 0 means "take it from the SIMParameters overload"; the
        // explicit-steps overload then requires nothing.
        int expectedOrders = 0;

        // Hermitian-in-kz check: mean |t(k) - conj(t(-k))| over mean |t|,
        // which makeotf forces to exactly 0 and a misdecoded file fails by
        // order 1. Refuses above this; 0 switches the check off.
        double hermitianTolerance = 0.2;

        OtfKzOrigin kzOrigin = OtfKzOrigin::Auto;

        bool accept2dImage = true;   // radially average a square real page
        bool readSidecar = true;     // look for <file>.toml / <stem>.toml

        // Shapes below this are refused: a radial table comes from a PSF at
        // least 16 pixels wide (nkr = min(nx,ny)/2 + 1), and a square page
        // that small is not an OTF image either.
        int minRadialSamples = 9;
        int maxOrders = 8;
    };

    // What the loader found and did. Every field is a fact about this load,
    // and `notes` holds the lines a caller should show a user: a derived
    // sampling, a rolled kz axis, a scale that rests on a noise sample.
    struct OtfLoadReport {
        std::string file;
        OtfLayout layout = OtfLayout::CudasireconRadial;
        int norders = 0, nkr = 0, nzotf = 0;
        std::array<int, 3> storedShape{0, 0, 0};   // the file's own dimensions

        double dkrotf = 0.0, dkzotf = 0.0;
        OtfSamplingSource samplingSource = OtfSamplingSource::DerivedFromParameters;
        std::string sidecarPath;          // non-empty when a sidecar was read

        OtfNormalization normalizationMode = OtfNormalization::Auto;
        OtfNormalizationReference reference = OtfNormalizationReference::None;
        double referenceValue = 1.0;      // the divisor, 1.0 when nothing was applied
        bool normalizationApplied = false;
        std::complex<double> storedOrder0Dc{0.0, 0.0};
        double lineFitValue = 0.0;        // the fixorigin extrapolation, for comparison

        bool kzRolled = false;            // the kz axis was rotated to put DC first
        int kzRoll = 0;
        double hermitianErrorAsStored = 0.0;
        double hermitianErrorApplied = 0.0;

        std::vector<std::string> notes;

        // Everything above on one line, for a log or a status bar.
        std::string summary() const;
    };

    // The two long-standing signatures, unchanged in meaning except that both
    // now normalize per OtfLoadOptions's default (Auto), which leaves every
    // table already at the reference exactly as it was.
    OTFRadiallyAveraged loadOTF(const std::string& filename, double dkrotf, double dkzotf);

    // Load a radially averaged OTF, deriving its reciprocal-space sampling
    // from the file dimensions and the acquisition parameters
    // (dkr = 1/(dx*(nkr-1)*2), dkz = 1/(dz_psf*nzotf)), as cudasirecon's
    // determine_otf_dimensions does for otfRA files -- but only when neither
    // the file itself (MRC cell lengths) nor a .toml sidecar says, and the
    // report records which it was.
    OTFRadiallyAveraged loadOTF(const std::string& filename, const SIMParameters& p);

    // The same two with the options and the report exposed.
    OTFRadiallyAveraged loadOTF(const std::string& filename, const SIMParameters& p,
                                const OtfLoadOptions& opts, OtfLoadReport* report);
    OTFRadiallyAveraged loadOTF(const std::string& filename, double dkrotf, double dkzotf,
                                const OtfLoadOptions& opts, OtfLoadReport* report);

    // A sidecar for a file that carries no sampling of its own:
    //
    //     # OTF_488.tif, makeotf on the sparse bead field
    //     [sampling]
    //     dkr = 0.0913461      # 1/um per radial sample
    //     dkz = 0.0990099      # 1/um per kz plane
    //     # or, when the measurement's pixel sizes are what is known:
    //     # [psf]
    //     # xyres = 0.0855263
    //     # zres  = 0.1
    //     # kz_origin = "dc_first"   # or "dc_last" to state a rotated axis
    //
    // loadOTF looks for "<file>.toml" first and then for the file's path with
    // its extension replaced by .toml. This parses one, and is public so a
    // caller can validate a sidecar without loading the table.
    struct OtfSidecar {
        double dkr = 0.0, dkz = 0.0;     // 0 where the file does not say
        double xyres = 0.0, zres = 0.0;  // pixel sizes, if that is the spelling used
        bool hasKzOrigin = false;
        OtfKzOrigin kzOrigin = OtfKzOrigin::Auto;
    };
    OtfSidecar readOtfSidecar(const std::string& path);

    // The sidecar path loadOTF would use for `filename`, or "" if none exists.
    std::string findOtfSidecar(const std::string& filename);

} // namespace sirius

#endif // SIRIUS_OTF_IO_HPP
