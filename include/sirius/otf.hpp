#ifndef SIRIUS_OTF_HPP
#define SIRIUS_OTF_HPP

#include <cmath>
#include <complex>
#include <utility>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace sirius {
    class OTFRadiallyAveraged {
    public:
        OTFRadiallyAveraged() = default;
        OTFRadiallyAveraged(Eigen::Tensor<std::complex<double>, 3, Eigen::RowMajor> data, double dkrotf, double dkzotf,
                            double normalizationDivisor = 1.0)
            : data_(std::move(data)), dkrotf_(dkrotf), dkzotf_(dkzotf), normalizationDivisor_(normalizationDivisor) {}

        const Eigen::Tensor<std::complex<double>, 3, Eigen::RowMajor>& data() const { return data_; }
        double dkrotf() const { return dkrotf_; }
        double dkzotf() const { return dkzotf_; }

        // The number every order was divided by on the way in, so a caller
        // can see what was done to the file's numbers; 1.0 means they are the
        // file's own. The scale itself is a property of the data, not of this
        // field -- ask atReferenceScale().
        double normalizationDivisor() const { return normalizationDivisor_; }

        // THE REFERENCE SCALE: order 0's kr = kz = 0 sample is 1. This is the
        // scale an absolute otfcutoff and the Wiener constant are calibrated
        // against (sirius/otf_io.hpp explains why both depend on it), the one
        // makeotf's rescale() leaves its tables on, and the one idealOTF
        // computes on. A caller that has to know whether a table still needs
        // normalizing asks the table, which cannot lie about it, rather than
        // tracking where it came from.
        bool atReferenceScale(double tol = 1e-5) const {
            if (data_.size() == 0) return false;
            const std::complex<double> dc = data_(0, 0, 0);
            return std::abs(dc.real() - 1.0) <= tol && std::abs(dc.imag()) <= tol;
        }

        // Extract one order's (nkr, nzotf) plane as a standalone tensor, ready
        // to pass to resampleOTF. Throws std::out_of_range on an invalid order.
        Eigen::Tensor<std::complex<double>, 2, Eigen::RowMajor> plane(int order) const;

    private:
        // Underlying data in (norders, nkr, nzotf) format
        Eigen::Tensor<std::complex<double>, 3, Eigen::RowMajor> data_;
        double dkrotf_ = 1.0;
        double dkzotf_ = 1.0;
        double normalizationDivisor_ = 1.0;
    };

    // Reading one from a TIFF is sirius/otf_io.hpp (loadOTF), computing the
    // theoretical one is sirius/otf_ideal.hpp (idealOTF): this header is the
    // table and its resampling, which need neither a file format nor an FFT.

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
