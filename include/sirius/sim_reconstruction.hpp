#ifndef SIRIUS_SIM_RECONSTRUCTION_HPP
#define SIRIUS_SIM_RECONSTRUCTION_HPP

#include "sirius/sim_parameters.hpp"
#include "sirius/preprocess.hpp"
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace sirius {
    template <typename Scalar>
    Eigen::Tensor<Scalar, 3, Eigen::RowMajor>
    reconstruct(Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& data,
                Eigen::Tensor<Scalar, 3, Eigen::RowMajor>& otf,
                const SIMParameters& p)
    {
        // Assume ndirs, nphases, nz, ny, nx format
        // TODO: Any other input needs an adapter first
        Eigen::Index ndirs = data.dimension(0);
        Eigen::Index nphases = data.dimension(1);
        Eigen::Index nz = data.dimension(2);
        Eigen::Index ny = data.dimension(3);
        Eigen::Index nx = data.dimension(4);
        Scalar total_n = static_cast<Scalar>(data.size());

        // Assume OTF has the same dimensions as data for now (most general)
        // TODO: Radially averaged OTF ie (m, nr, nz) and other common formats need to be supported

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

        // TODO: edge apodization
        // edge_apodization(data, p.napodize);

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