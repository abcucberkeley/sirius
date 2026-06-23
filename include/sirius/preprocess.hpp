#ifndef SIRIUS_PREPROCESS_HPP
#define SIRIUS_PREPROCESS_HPP

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace sirius {

    // Photobleaching correction ("rescale")
    void bleach_rescale(Eigen::Tensor<double, 5, Eigen::RowMajor>& data,
                        bool equalizez);

} // namespace sirius

#endif // SIRIUS_PREPROCESS_HPP
