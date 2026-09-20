#ifndef SIRIUS_OTF_IDEAL_HPP
#define SIRIUS_OTF_IDEAL_HPP

// The theoretical OTF, for when no measured one (sirius/otf_io.hpp) is at hand.

#include "sirius/otf.hpp"
#include "sirius/sim_parameters.hpp"

namespace sirius {

    // Sampling of the simulated PSF behind idealOTF. The lateral grid has
    // lateralSamples^2 points with a pixel of lambda / (8 NA), so its Nyquist
    // frequency is twice the OTF cutoff and the table's radial step is
    // 8 NA / (lambda * lateralSamples). A 3D table has axialSamples planes
    // spaced dzPsf apart (0 selects SIMParameters::dz_psf).
    struct IdealOtfOptions {
        int lateralSamples = 256;
        int axialSamples = 64;
        double dzPsf = 0.0;
    };

    // Theoretical OTF of an aberration-free widefield microscope: circular
    // pupil of radius NA / lambda with the sine-condition apodization, in a
    // medium of index nimm, at the emission wavelength. It is produced in the
    // radially averaged (norders, nkr, nzotf) layout loadOTF reads, so it can
    // stand in for a measured OTF when none is available. `threeD` selects
    // the 3D OTF (nzotf = axialSamples, missing cone included) over the
    // in-focus 2D OTF (nzotf = 1). Order 0 and every order >= 2 are the
    // widefield OTF; for a 3D table order 1 is the mean of the widefield OTF
    // shifted by +-kz of the first illumination order (three-beam
    // interference, excitation wavelength taken as 0.88 x emission like the
    // reconstruction does). Every order is normalized to order 0's DC value.
    // norders follows SIMParameters (norders, or nphases / 2 + 1 when 0).
    OTFRadiallyAveraged idealOTF(const SIMParameters& p, bool threeD, const IdealOtfOptions& opts = {});

} // namespace sirius

#endif // SIRIUS_OTF_IDEAL_HPP
