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
        // A magnitude FLOOR under the fitted modulation amplitudes, not a
        // replacement for them: this is cudasirecon's `forcemodamp`
        // (cudaSirecon.cpp, "force modamp's amplitude to be a value user
        // provided"), and matching it matters because forcemodamp=0.5 is in
        // the configs the user runs daily. Its rules, all four of which SIRIUS
        // used to get wrong:
        //   * SIDE BANDS ONLY. The loop runs order = 1 .. norders-1; order 0
        //     is the widefield band and is never touched.
        //   * A FLOOR. An order whose fitted magnitude already exceeds its
        //     value is left exactly as fitted.
        //   * THE FITTED PHASE SURVIVES. The complex amplitude is scaled by
        //     floor/|amp|, so only its magnitude changes.
        //   * ONE GATE FOR THE WHOLE FEATURE: the first entry of the list
        //     being <= 0 switches it off, whatever the later entries say.
        // Four lengths are accepted. norders-1 and ndirs*(norders-1) are
        // cudasirecon's own, listing the side bands alone (and norders-1 is
        // what a 3-phase config's single value is); norders and
        // ndirs*norders are SIRIUS's older spelling, whose leading entry is
        // order 0's and is ignored except as the gate. Per direction, the
        // layout is direction-major. Empty: fit everything.
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
        // Fit the pattern vector, or take it as given. False is cudasirecon's
        // `searchforvector=0`: skip the cross-correlation search and the
        // angle/magnitude refinement, keep k0 exactly as k0_angles (or
        // k0_start_angle) and linespacing_um state it, and fit only each
        // order's modulation amplitude and phase against order 0
        // (cudaSirecon.cpp:319, the "assume k0 vector known" branch).
        bool search_pattern_vector = true;

        // The orders the reconstruction separates and assembles: norders, or
        // nphases / 2 + 1 when norders is 0. Everything that consumes the
        // order count goes through this, so a file that leaves norders out
        // means the same thing everywhere.
        int resolvedOrders() const noexcept { return norders > 0 ? norders : nphases / 2 + 1; }

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

        // The magnitude floor forced onto direction `dir`'s order `order`, or
        // 0 when none applies -- and 0 is a no-op, because a magnitude is
        // never below it. One place decides what a force_mod_amp list of each
        // accepted length means, so the reconstruction does not have to, and a
        // dir or order out of range answers 0 rather than reading past the end.
        double forcedModAmpFloor(int dir, int order) const noexcept;

        // Throws std::runtime_error on invalid parameters
        void validate() const;
    };

    // TOML I/O. loadParameters starts from defaults so a partial file overrides
    // only the keys present, then validate()s before returning.
    SIMParameters loadParameters(const std::string& path);
    void saveParameters(const std::string& path, const SIMParameters& p);

} // namespace sirius

#endif // SIRIUS_SIM_PARAMETERS