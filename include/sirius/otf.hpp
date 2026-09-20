#ifndef SIRIUS_OTF_HPP
#define SIRIUS_OTF_HPP

#include <complex>
#include <string>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include "sirius/sim_parameters.hpp"

namespace sirius {
    class OTFRadiallyAveraged {
    public:
        OTFRadiallyAveraged() = default;
        OTFRadiallyAveraged(Eigen::Tensor<std::complex<double>, 3, Eigen::RowMajor> data, double dkrotf, double dkzotf)
            : data_(std::move(data)), dkrotf_(dkrotf), dkzotf_(dkzotf) {}

        const Eigen::Tensor<std::complex<double>, 3, Eigen::RowMajor>& data() const { return data_; }
        double dkrotf() const { return dkrotf_; }
        double dkzotf() const { return dkzotf_; }

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
