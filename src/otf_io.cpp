#include "sirius/otf_io.hpp"
#include "sirius/errors.hpp"
#include "sirius/tiff_io.hpp"

#include <string>
#include <utility>

namespace sirius {
    namespace {
        // Read a raw OTF TIFF (real/imag interleaved along the last axis)
        // into a complex (norders, nkr, nzotf) tensor.
        Eigen::Tensor<std::complex<double>, 3, Eigen::RowMajor>
        readRadialOTF(const std::string& filename) {
            using Cplx = std::complex<double>;
            using DoubleTensor = Eigen::Tensor<double, 3, Eigen::RowMajor>;

            // Read raw data in any supported format and convert to double (handled by readTiffStack)
            DoubleTensor raw_data = readTiffStack<double>(filename);

            if (raw_data.size() == 0)
                throw IoError("Radial OTF is empty: " + filename);
            if (raw_data.dimension(2) % 2 != 0)
                throw IoError("Radial OTF - incorrect data format");

            // complex_otf = raw_data[..., 0::2] + i*raw_data[..., 1::2]
            Eigen::array<Eigen::Index, 3> start_real = {0, 0, 0};
            Eigen::array<Eigen::Index, 3> start_imag = {0, 0, 1};
            Eigen::array<Eigen::Index, 3> stop = raw_data.dimensions();
            Eigen::array<Eigen::Index, 3> strides = {1, 1, 2};

            return raw_data.stridedSlice(start_real, stop, strides).cast<Cplx>() +
                   raw_data.stridedSlice(start_imag, stop, strides).cast<Cplx>() * Cplx(0, 1);
        }
    } // namespace

    OTFRadiallyAveraged loadOTF(const std::string& filename, double dkrotf, double dkzotf) {
        return OTFRadiallyAveraged(readRadialOTF(filename), dkrotf, dkzotf);
    }

    OTFRadiallyAveraged loadOTF(const std::string& filename, const SIMParameters& p) {
        auto data = readRadialOTF(filename);
        const auto nkr = data.dimension(1);
        const auto nzotf = data.dimension(2);
        if (nkr < 2)
            throw IoError("Radial OTF needs at least 2 radial samples: " + filename);
        const double dkrotf = 1.0 / (p.dx * static_cast<double>(nkr - 1) * 2.0);
        const double dkzotf = 1.0 / (p.dz_psf * static_cast<double>(nzotf));
        return OTFRadiallyAveraged(std::move(data), dkrotf, dkzotf);
    }

} // namespace sirius
