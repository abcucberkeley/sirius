#ifndef SIRIUS_SIM_RECONSTRUCTION_HPP
#define SIRIUS_SIM_RECONSTRUCTION_HPP

#include "sirius/sim_parameters.hpp"
#include "sirius/preprocess.hpp"
#include "sirius/otf.hpp"

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace sirius {
    template <typename Scalar>
    Eigen::Tensor<Scalar, 3, Eigen::RowMajor>
    reconstruct(Eigen::Tensor<Scalar, 5, Eigen::RowMajor>& data,
                const OTFRadiallyAveraged& otf,
                const SIMParameters& p)
    {
        using Index = Eigen::Index;
        const Index ndirs   = data.dimension(0);
        const Index nphases = data.dimension(1);
        const Index nz      = data.dimension(2);
        const Index ny      = data.dimension(3);
        const Index nx      = data.dimension(4);

        // Background subtraction and pre-scale (compensate un-normalied ffts)
        // Reconstruction can be sensitive to background since it could affect various estimates
        // such as
        // - the noise floor for Wiener filtering
        // - the relative scaling of different orders which affects order weighting and phase estimation
        // - wavevector estimation
        // - bleach correction
        Scalar pre_scale = 1.0 / (nx * ny * nz * p.zoomfact * p.zoomfact * p.z_zoom * p.ndirs);
        data = (data - p.background) * pre_scale;

        // Bleach correction
        if (p.do_rescale) bleach_rescale(data, p.equalizez);

        // Real-space apodization to suppress FFT edge-wraparound artifacts.
        // Triangle blends a napodize-wide border; Cosine applies a full sine
        // window; None skips it. (Distinct from p.apodize_output, applied later.)
        switch (p.apodize_input) {
            case ApodizationType::None:                                break;
            case ApodizationType::Cosine:   cosine_apodization(data); break;
            case ApodizationType::Triangle: edge_apodization(data, p.napodize); break;
        }

        // TODO: Separate bands
        // Analytical inverse or solve lsq

        // TODO: k0 correction
        // Whiten each band by each other's OTF

        // TODO: Using k0, compute modulation amplitude

        // TODO: Refine k0 and amplitude estimates

        // TODO: Generalized Wiener filtering

        // TODO: Assembly into the super-resolution volume.

    }
}

#endif // SIRIUS_SIM_RECONSTRUCTION_HPP