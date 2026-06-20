#ifndef SIRIUS_LEGACY_CONFIG
#define SIRIUS_LEGACY_CONFIG

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

        // Pixel sizes (from ImageParams / config)
        float dxy   = 0.1f;
        float dz    = 0.2f;
        float dzPSF = 0.15f;

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
    };

    // Parse a legacy flat `key=value` cudasirecon config file. Blank lines and
    // lines beginning with '#' or ';' are ignored. Throws std::runtime_error on
    // a malformed line, a bad value, or an unrecognized key (strict mode).
    LegacyReconConfig loadLegacyConfig(const std::string& path);

    // Convert the legacy config into the modern, lean SIMParameters. Fields the
    // modern container does not model are dropped. The result is validated.
    SIMParameters fromLegacy(const LegacyReconConfig& c);

} // namespace sirius

#endif // SIRIUS_LEGACY_CONFIG
