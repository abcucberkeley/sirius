#ifndef SIRIUS_SIM_PARAMETERS
#define SIRIUS_SIM_PARAMETERS

#include <optional>
#include <string>
#include <vector>

namespace sirius {
    // Integer values (0/1/2) to match the legacy cudasirecon
    // so they survive round tripping
    enum class ApodizationType {
        None = 0,
        Cosine = 1,
        Triangle = 2
    };

    // Parameters consumed by the reconstruction core.
    // cudasirecon configs are loaded via LegacyReconConfig (legacy_config.hpp)
    // and converted via fromLegacy()
    struct SIMParameters {
        // Geometry and optics
        double k0_start_angle = 0.;     // starting illumination angle (rad)
        double linespacing_um = 0.24;   // illumination line spacing (um)
        int ndirs = 3;      // number of directions (theta)
        int nphases = 5;      // number of phases (phi)
        int norders = 0;      // orders to separate; 0 derives nphases / 2 + 1 (resolvedOrders)
        double na = 1.0;    // detection numerical aperture
        double nimm = 1.33;   // immersion refractive index (>= na)
        double wavelength_nm = 510.;   // emission wavelength (nm)
        std::optional<std::vector<double>> k0_angles; // null, derive from k0_start_angles
        // Absolute phase of each raw frame, radians, length nphases. Empty: equal steps 2πj/nphases.
        std::optional<std::vector<double>> phase_steps;
        // Modulation amplitudes that replace the fitted ones. Length norders (every direction)
        // or ndirs * norders (per direction, direction-major). Empty: fit them.
        std::optional<std::vector<double>> force_mod_amp;

        // Pixel sizes (um) in the sample plane
        double dx = 0.1;  // transverse pixel size, image column direction
        double dy = 0.1;  // transverse pixel size, image row direction
        double dz = 0.2;  // axial pixel size
        double dz_psf = 0.15; // axial step size of the PSF/OTF

        // Output and filtering
        double zoomfact = 2.0; // "Zoom factor" for the output grid transverse dimensions relative to the input data grid (>= 1). SIM increases resolution.
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

        // The orders the reconstruction separates and assembles: norders, or
        // nphases / 2 + 1 when norders is 0. Everything that consumes the
        // order count goes through this, so a file that leaves norders out
        // means the same thing everywhere.
        int resolvedOrders() const noexcept { return norders > 0 ? norders : nphases / 2 + 1; }

        // A raw SIM stack holds ndirs * nphases frames per plane.
        int sectionsPerPlane() const noexcept { return ndirs * nphases; }

        // Planes (nz) of a raw stack of `sections` frames, or 0 when the count
        // is not a whole number of planes. Everything that asks how deep a raw
        // stack is goes through this: the section-count error, and -- the
        // reason it is here rather than in one front -- whether the
        // theoretical OTF is built in 3D (several planes) or in 2D (one). The
        // GUI, the CLI and the Python mirror each used to do the arithmetic
        // themselves.
        long long planes(long long sections) const noexcept {
            const long long per = sectionsPerPlane();
            return (per > 0 && sections > 0 && sections % per == 0) ? sections / per : 0;
        }

        // The section-count condition, worded once. Empty when `sections` is a
        // whole number of planes; otherwise the sentence every front says.
        //
        // One condition used to read a different way in each front, and three
        // of them for the lateral size below (docs/findings.md 9k.50,
        // finding 5): the library threw "SimReconstructor: 134 sections is
        // not a multiple of ndirs*nphases = 15", ReconSession::validate()
        // returned "134 sections is not a multiple of ndirs * nphases = 15."
        // and the dataset's layout machinery -- which the SIM step and the
        // Python mirror go through -- said "z holds 134 sections, not a
        // multiple of angle 3 × phase 5 = 15.". This is that last wording,
        // the one the layout machinery already produced for the same
        // arithmetic (app/core/dataset.cpp's bindSimLayout through
        // productText), so a front that falls back to this says what the
        // layout would have said. A front decides only whether it throws the
        // string, returns it or collects it into a Validation.
        std::string sectionCountProblem(long long sections) const;

        // Lateral frequency of illumination order 1, in 1/um. The configured
        // line spacing is the finest order, so a 3D pattern's order 1 sits at
        // 1/linespacing/(resolvedOrders()-1). A 2D pattern's spacing is already
        // order 1, and the search does not divide.
        double patternFundamental(bool threeD) const noexcept {
            const double k = 1.0 / linespacing_um;
            if (!threeD) return k;
            const int denom = resolvedOrders() - 1;
            return k / static_cast<double>(denom < 1 ? 1 : denom);
        }

        // Throws std::runtime_error on invalid parameters
        void validate() const;
    };

    // The smallest lateral extent SimReconstructor binds, per axis.
    inline constexpr int kMinSimExtent = 4;

    // The lateral-size condition, worded once, in the same spirit as
    // SIMParameters::sectionCountProblem above: empty when the extents can be
    // reconstructed, otherwise the one sentence every front says.
    //
    // Since 2026-10-08 that condition is the minimum alone. It used to be
    // "even and at least 4", enforced in three places and spelt three ways;
    // the parity half is gone because the index arithmetic is parity-general
    // (src/sim_math.hpp's signedFrequency and mirrorColumns), so an odd stack
    // reconstructs. `montage` words it for one tile of a montaged stack,
    // which is the extent the SIM step actually binds there.
    std::string simImageSizeProblem(long long nx, long long ny, bool montage = false);

    // TOML I/O. loadParameters starts from defaults so a partial file overrides
    // only the keys present, then validate()s before returning.
    SIMParameters loadParameters(const std::string& path);
    void saveParameters(const std::string& path, const SIMParameters& p);

} // namespace sirius

#endif // SIRIUS_SIM_PARAMETERS