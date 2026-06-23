#ifndef SIRIUS_SIM_PARAMETERS
#define SIRIUS_SIM_PARAMETERS

#include <optional>
#include <string>
#include <vector>

namespace sirius {
    // Integer values (0/1/2) to match the legacy cudasirecon
    // so they survive round tripping
    enum class ApodizationType {
        None     = 0,
        Cosine   = 1,
        Triangle = 2
    };

    // Parameters consumed by the reconstruction core.
    // cudasirecon configs are loaded via LegacyReconConfig (legacy_config.hpp)
    // and converted via fromLegacy()
    struct SIMParameters {
        // Geometry and optics
        double k0_start_angle = 0.;     // starting illumination angle (rad)
        double linespacing_um = 0.24;   // illumination line spacing (um)
        int ndirs             = 3;      // number of directions (theta)
        int nphases           = 5;      // number of phases (phi)
        int norders           = 3;      // nphases / 2 + 1;
        double na             = 1.0;    // detection numerical aperture
        double nimm           = 1.33;   // immersion refractive index
        double wavelength_nm  = 510.;   // emission wavelength (nm)
        std::optional<std::vector<double>> k0_angles; // null, derive from k0_start_angles

        // Pixel sizes (um) in the sample plane
        double dx     = 0.1;  // transverse pixel size, image column direction
        double dy     = 0.1;  // transverse pixel size, image row direction
        double dz     = 0.2;  // axial pixel size
        double dz_psf = 0.15; // axial step size of the PSF/OTF

        // Output and filtering
        double zoomfact = 2.0; // "Zoom factor" for the output grid transverse dimensions relative to the input data grid. SIM increases resolution.
        int z_zoom = 1; // Zoom factor for the output grid axial dimension.
        double wiener = 0.01;
        double otfcutoff = 0.006;
        double background = 0.0;
        ApodizationType apodize_input = ApodizationType::Triangle;
        int napodize = 10; // Triangle border width (pixels); unused otherwise
        int suppression_radius = 10;
        bool suppress_singularities = true;
        bool dampen_order0 = false;
        ApodizationType apodize_output = ApodizationType::Triangle;
        double explodefact = 1.0;
        bool fast_si = false;
        bool do_rescale = true; // whether to perform bleach correction
        bool equalizez = false; // ref = equalizez ? S[sidx(0, 0, 0)] : S[sidx(0, 0, z)]; where S(d, p, z) is plane sums over (ny, nx)
        bool no_kz0 = true;
        bool filter_overlaps = true;

        // Throws std::runtime_error on invalid parameters
        void validate() const;
    };

    // TOML I/O. loadParameters starts from defaults so a partial file overrides
    // only the keys present, then validate()s before returning.
    SIMParameters loadParameters(const std::string& path);
    void          saveParameters(const std::string& path, const SIMParameters& p);

} // namespace sirius

#endif // SIRIUS_SIM_PARAMETERS