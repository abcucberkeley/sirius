#ifndef SIRIUS_LEGACY_CONFIG
#define SIRIUS_LEGACY_CONFIG

#include <optional>
#include <string>
#include <vector>

#include "sirius/sim_parameters.hpp"

namespace sirius {

    // cudasirecon config
    // mainly for converting to SIMParameters
    struct LegacyReconConfig {
        // Geometry / optics
        float              k0startangle = 1.648f;
        float              linespacing  = 0.172f;
        float              na           = 1.2f;
        float              nimm         = 1.33f;
        int                ndirs        = 3;
        int                nphases      = 5;
        int                norders_output = 0;   // 0 -> derive from nphases
        int                norders      = 0;
        int                nbeams       = 0;      // 0 means unset (use nphases)
        std::vector<float> phaseSteps;            // was float* phaseSteps
        std::vector<float> k0angles;
        float              wavelengthNm = 530.0f; // SIRIUS extension (TIFF mode)

        // Pixel sizes (from ImageParams / config). Doubles: they are what a
        // SIM step driven by the file reconstructs with and what a measured
        // OTF's sampling is derived from, and a file's 0.08 is 0.08, not the
        // 0.0799999982 a float makes of it (which then reaches the output
        // voxel and every derived frequency step).
        double dxy   = 0.1;
        double dz    = 0.2;
        double dzPSF = 0.15;

        // I5S / Bessel / deskew
        bool  bTwolens       = false;
        bool  bFastSIM       = false;
        bool  bBessel        = false;
        float BesselNA       = 0.0f;
        float BesselLambdaEx = 0.0f;
        float deskewAngle    = 0.0f;
        int   extraShift     = 0;
        bool  bNoRecon       = false;
        int   cropXYto       = 0;       // legacy: unsigned; 0 means no crop
        bool  bWriteTitle    = false;

        // Algorithm / filtering
        float otfcutoff             = 0.006f;
        float zoomfact              = 2.0f;
        int   z_zoom                = 1;
        int   nzPadTo               = 0;
        float explodefact           = 1.0f;
        bool  bFilteroverlaps       = true;
        int   recalcarrays          = 1;
        int   napodize              = 10;
        int   bSearchforvector      = 1;
        int   bUseTime0k0           = 1;
        // cudasirecon >= 1.1: search for k0 at every time point instead of reusing time 0's.
        // SIRIUS reconstructs one (c, t) volume at a time and fits each, so the flag is
        // accepted for compatibility and changes nothing here.
        int   k0searchAll           = 0;
        int   apodizeoutput         = 2;     // 0-none 1-cosine 2-triangle
        float apoGamma              = 1.0f;
        int   bSuppress_singularities = 1;
        int   suppression_radius    = 10;
        bool  bDampenOrder0         = false;
        int   bFitallphases         = 1;
        int   do_rescale            = 1;
        bool  equalizez             = false;
        bool  equalizet             = false;
        bool  bNoKz0                = true;
        float wiener                = 0.01f;
        float wienerInr             = 0.0f;
        std::vector<float> forceamp;

        // OTF geometry
        int   nxotf = 0, nyotf = 0, nzotf = 0;
        float dkzotf = 0.0f, dkrotf = 0.0f;
        bool  bRadAvgOTF      = false;
        bool  bOneOTFperAngle = false;

        // Drift correction
        int   bFixdrift         = 0;
        float drift_filter_fact = 0.0f;

        // Camera
        float       constbkgd        = 0.0f;
        int         bBgInExtHdr      = 0;
        int         bUsecorr         = 0;
        std::string corrfiles;
        float       readoutNoiseVar  = 0.0f;
        float       electrons_per_bit = 0.0f;

        // Debugging / intermediate output
        int         bMakemodel     = 0;
        int         bSaveSeparated = 0;
        std::string fileSeparated;
        int         bSaveAlignedRaw = 0;
        std::string fileRawAligned;
        int         bSaveOverlaps  = 0;
        std::string fileOverlaps;

        // I/O
        bool        bTIFF = true;
        std::string ifiles;
        std::string ofiles;
        std::string otffiles;

        // ---- cudasirecon 1.2.0 ("2Beam3D") output and tiling keys --------
        // These are in the vocabulary of the build the user runs daily and in
        // neither 1.1.1's nor the fork's, and SIRIUS refused a file outright
        // on the first of them. They are I/O and scheduling quantities, not
        // reconstruction maths: SIRIUS reads them so the file loads, reports
        // them through LegacyConversionReport as not applied, and leaves the
        // acting on them to the caller. The semantics are the ones the
        // binary's own option help states.

        // "Write TIFF output as uint16 instead of float32 (clamps negatives
        // to 0 and values above 65535)" -- an exporter concern.
        bool  bUint16Output = false;
        // "Constant value added to every pixel just before the uint16
        // [0, 65535] clamp ... Only applied when --uint16 is set; default 0."
        float uint16Offset = 0.0f;

        // "Crop bounding box ... (0-indexed, inclusive, TIFF only)", applied
        // to the raw stack at load time (SIM_Reconstructor::cropRawImageToBBox).
        // z is in LOGICAL-z units, i.e. after the phases are de-interleaved.
        // -1 means the key was not in the file. INCLUSIVE: the width is
        // max - min + 1, so the user's two mounts are 2301 x 759 and
        // 2301 x 751 -- odd in both lateral axes.
        int cropXmin = -1, cropXmax = -1;
        int cropYmin = -1, cropYmax = -1;
        int cropZmin = -1, cropZmax = -1;

        // "Chunk size along X/Y in raw input pixels (0 = entire axis)",
        // chunkZ "in logical z-planes", and "Chunk overlap in raw input
        // pixels (must be non-negative and even)". A tiling schedule for a
        // stack too large to reconstruct in one piece, stitched afterwards;
        // it is not a change to the reconstruction of any one tile.
        int chunkX = 0, chunkY = 0, chunkZ = 0;
        int chunkOverlap = 0;

        // Every key the file actually set, in the order the file set them
        // (a key repeated in the file appears once, at its first position).
        // fromLegacy's report is computed against this, so a key that is
        // parsed and then thrown away is visible instead of silent.
        std::vector<std::string> keysPresent;
    };

    // What fromLegacy() did with each key the file set.
    enum class LegacyKeyStatus {
        Applied,   // reaches SIMParameters, and the reconstruction honours it
        Pending,   // reaches SIMParameters; nothing reads it yet
        Dropped    // parsed so the file is accepted, then thrown away
    };

    // The audit of one conversion: accepting a key is not applying it, and
    // before this existed a config saying `gammaApo=0.5` or `fitallphases=0`
    // or `searchforvector=0` was read, stored on LegacyReconConfig and then
    // silently ignored, so the run did something other than the file asked.
    struct LegacyConversionReport {
        std::vector<std::string> applied;   // keys in effect
        std::vector<std::string> pending;   // carried, not yet acted on
        std::vector<std::string> dropped;   // not in effect at all
        // One "key: why" line per pending or dropped key, naming where the
        // quantity has to be handled instead. Fit for a log or a dialog.
        std::vector<std::string> notes;

        // Everything the file asked for that is not in effect.
        std::vector<std::string> notInEffect() const;
        bool everythingApplied() const noexcept { return pending.empty() && dropped.empty(); }
    };

    // The status SIRIUS gives a recognized legacy key, and the reason line
    // that goes with a Pending or Dropped one (empty for Applied). An
    // unrecognized key yields Dropped and a reason saying so.
    LegacyKeyStatus legacyKeyStatus(const std::string& key, std::string* reason = nullptr);

    // Every key loadLegacyConfig accepts, sorted. For the test that keeps the
    // parser's table and the status table from drifting apart.
    std::vector<std::string> legacyConfigKeys();

    // A crop bounding box from a legacy config, as a HALF-OPEN extent, for the
    // caller that applies it. The file's bounds are 0-indexed and inclusive,
    // so each size is max - min + 1; one place does that arithmetic, because
    // an off-by-one here is invisible in the result.
    struct LegacyCropBox {
        int x0 = 0, y0 = 0, z0 = 0;
        int nx = 0, ny = 0, nz = 0;
    };

    // The box the config asks for, or nullopt unless all six bounds are set
    // and each max is >= its min. All six is what cudasirecon 1.2.0 demands of
    // itself ("requires all 6 crop{X,Y,Z}{min,max}"), so a config giving four
    // of them -- which both of the user's 2026-04-21 configs do -- yields
    // nullopt here rather than a box with a guessed z range. z is in logical-z
    // units, i.e. after the phases are de-interleaved.
    std::optional<LegacyCropBox> legacyCropBox(const LegacyReconConfig& c);

    // Parse a legacy flat `key=value` cudasirecon config file. Blank lines and
    // lines beginning with '#' or ';' are ignored. Throws sirius::IoError on
    // a malformed line, a bad value, or an unrecognized key (strict mode).
    LegacyReconConfig loadLegacyConfig(const std::string& path);

    // Convert the legacy config into the modern, lean SIMParameters. Fields the
    // modern container does not model are dropped. The result is validated.
    // Pass `report` to learn which of the keys the file set are actually in
    // effect: `fromLegacy` ends with validate(), so a file can be accepted by
    // the parser and still be refused here, and a key can be accepted and then
    // dropped without a word. The report is filled from c.keysPresent, so it
    // is empty for a LegacyReconConfig built in code rather than parsed.
    SIMParameters fromLegacy(const LegacyReconConfig& c, LegacyConversionReport* report = nullptr);

} // namespace sirius

#endif // SIRIUS_LEGACY_CONFIG
