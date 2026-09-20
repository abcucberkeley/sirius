#ifndef SIRIUS_OTF_IO_HPP
#define SIRIUS_OTF_IO_HPP

// Measured OTFs: reading the radially averaged table (sirius/otf.hpp) from
// the TIFF files cudasirecon's tools write.

#include <string>

#include "sirius/otf.hpp"
#include "sirius/sim_parameters.hpp"

namespace sirius {

    OTFRadiallyAveraged loadOTF(const std::string& filename, double dkrotf, double dkzotf);

    // Load a radially averaged OTF TIFF, deriving its reciprocal-space
    // sampling from the file dimensions and the acquisition parameters
    // (dkr = 1/(dx*(nkr-1)*2), dkz = 1/(dz_psf*nzotf)), as cudasirecon's
    // determine_otf_dimensions does for otfRA files.
    OTFRadiallyAveraged loadOTF(const std::string& filename, const SIMParameters& p);

} // namespace sirius

#endif // SIRIUS_OTF_IO_HPP
