#ifndef SIRIUS_PREPROCESS_HPP
#define SIRIUS_PREPROCESS_HPP

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace sirius {

    // Photobleaching correction ("rescale")
    void bleach_rescale(Eigen::Tensor<double, 5, Eigen::RowMajor>& data,
                        bool equalizez);

    // Edge apodization: blend opposite edges of each (ny, nx) section over a
    // napodize-wide border so the image is near-periodic, suppressing FFT
    // edge-wraparound artifacts. data: (ndirs, nphases, nz, ny, nx). No-op for
    // napodize <= 0.
    void edge_apodization(Eigen::Tensor<double, 5, Eigen::RowMajor>& data,
                          int napodize);

    // Cosine (sine-window) apodization: multiply each (ny, nx) section by
    // sin(pi*(x+0.5)/nx) * sin(pi*(y+0.5)/ny), tapering all four edges to zero.
    void cosine_apodization(Eigen::Tensor<double, 5, Eigen::RowMajor>& data);

} // namespace sirius

#endif // SIRIUS_PREPROCESS_HPP
