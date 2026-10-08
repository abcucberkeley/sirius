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

    // Whether idealOTF gives order 1 the axial shift of three-beam
    // interference, in one place so the rule is stated once rather than
    // re-derived at the point of use. Three things must hold: the table is 3D
    // (a single-kz-plane table has nowhere to shift to), more than one order
    // is resolved (there is no order 1 otherwise), and the illumination
    // actually carries an axial component -- which is
    // SIMParameters::illumination_has_axial_component, a stated property of
    // the acquisition and not something guessed from the order count. A
    // two-beam pattern is a pure lateral sinusoid, so its order-1 OTF is the
    // plain widefield OTF and this answers false.
    bool idealOtfShiftsOrderOne(const SIMParameters& p, bool threeD) noexcept;

    // Theoretical OTF of an aberration-free widefield microscope: circular
    // pupil of radius NA / lambda with the sine-condition apodization, in a
    // medium of index nimm, at the emission wavelength. It is produced in the
    // radially averaged (norders, nkr, nzotf) layout loadOTF reads, so it can
    // stand in for a measured OTF when none is available. `threeD` selects
    // the 3D OTF (nzotf = axialSamples, missing cone included) over the
    // in-focus 2D OTF (nzotf = 1). Order 0 and every order >= 2 are the
    // widefield OTF. Order 1 is the widefield OTF too, UNLESS
    // idealOtfShiftsOrderOne(p, threeD) -- when it is, order 1 becomes the
    // mean of the widefield OTF shifted by +-kz of the first illumination
    // order (three-beam interference, excitation wavelength taken as
    // 0.88 x emission like the reconstruction does). Every order is
    // normalized to order 0's DC value. norders follows SIMParameters
    // (norders, or nphases / 2 + 1 when 0).
    OTFRadiallyAveraged idealOTF(const SIMParameters& p, bool threeD, const IdealOtfOptions& opts = {});

} // namespace sirius

#endif // SIRIUS_OTF_IDEAL_HPP
