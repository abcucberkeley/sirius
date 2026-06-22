#ifndef SIRIUS_OTF_HPP
#define SIRIUS_OTF_HPP

#include <complex>
#include <string>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace sirius {
    class OTF {
        // TODO: General OTF class.
        // Hold off on the abstract OTF class for now
    };

    class OTFRadiallyAveraged {
    public:
        OTFRadiallyAveraged() = default;
        OTFRadiallyAveraged(Eigen::Tensor<std::complex<double>, 3, Eigen::RowMajor> data, double dkrotf, double dkzotf)
            : data_(std::move(data)), dkrotf_(dkrotf), dkzotf_(dkzotf) {}

        const Eigen::Tensor<std::complex<double>, 3, Eigen::RowMajor>& data() const { return data_; }
        double dkrotf()  const { return dkrotf_; }
        double dkzotf()  const { return dkzotf_; }

        // Extract one order's (nkr, nzotf) plane as a standalone tensor, ready
        // to pass to resampleOTF. Throws std::out_of_range on an invalid order.
        Eigen::Tensor<std::complex<double>, 2, Eigen::RowMajor> plane(int order) const;

    private:
        // Underlying data in (norders, nkr, nzotf) format
        Eigen::Tensor<std::complex<double>, 3, Eigen::RowMajor> data_;
        double dkrotf_ = 1.0;
        double dkzotf_ = 1.0;
    };

    OTFRadiallyAveraged loadOTF(const std::string& filename, double dkrotf, double dkzotf);

    // Resample a radially averaged OTF (one order) onto a Cartesian Fourier grid
    //
    //   radial_otf : one order, shape (nkr, nzotf), row-major (nzotf contiguous)
    //   returns    : (nz, ny, nx) in FFT layout (DC at index 0; the upper half
    //                of each axis is negative frequency, x fastest)
    // Radial samples outside [0, nkr) contribute zero (grid corners exceed the
    // OTF radius); the kz neighbor wraps, which also covers the nzotf-1 edge.
    Eigen::Tensor<std::complex<double>, 3, Eigen::RowMajor>
    resampleOTF(const Eigen::Tensor<std::complex<double>, 2, Eigen::RowMajor>& radial_otf,
                int nx, int ny, int nz,
                double dkx, double dky, double dkrotf, double kzscale);

} // namespace sirius

#endif // SIRIUS_OTF_HPP
