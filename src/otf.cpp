#include "sirius/otf.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace sirius {

    Eigen::Tensor<std::complex<double>, 2, Eigen::RowMajor>
    OTFRadiallyAveraged::plane(int order) const {
        const Eigen::Index norders = data_.dimension(0);
        if (order < 0 || order >= norders)
            throw std::out_of_range(
                "OTFRadiallyAveraged::plane: order " + std::to_string(order) +
                " out of range [0, " + std::to_string(norders) + ")");
        // order is the outermost (contiguous) axis, so this chip is a cheap copy
        return data_.chip(static_cast<Eigen::Index>(order), 0);
    }

    Eigen::Tensor<std::complex<double>, 3, Eigen::RowMajor>
    resampleOTF(const Eigen::Tensor<std::complex<double>, 2, Eigen::RowMajor>& radial_otf,
                int nx, int ny, int nz,
                double dkx, double dky, double dkrotf, double kzscale) {
        using Cplx = std::complex<double>;

        const Eigen::Index nkr = radial_otf.dimension(0);
        const Eigen::Index nzotf = radial_otf.dimension(1);
        const Cplx* otf = radial_otf.data();

        // fold the OTF radial step into the per-axis scales (one mul per voxel)
        const double rxscale = dkx / dkrotf;
        const double ryscale = dky / dkrotf;

        Eigen::Tensor<Cplx, 3, Eigen::RowMajor> out(nz, ny, nx);
        Cplx* dst = out.data();

        // radial fetch: iz already in [0, nzotf); ir is bounds-checked -> zero.
        auto fetch = [&](Eigen::Index ir, Eigen::Index iz) -> Cplx {
            if (ir < 0 || ir >= nkr) return Cplx(0.0, 0.0);
            return otf[ir * nzotf + iz];
        };

#pragma omp parallel for collapse(2) schedule(static)
        for (int iz = 0; iz < nz; ++iz) {
            for (int iy = 0; iy < ny; ++iy) {
                // signed FFT frequency indices (negative freqs in the upper half)
                const int kz = (iz <= nz / 2) ? iz : iz - nz;
                const int ky = (iy <= ny / 2) ? iy : iy - ny;

                // axial index, wrapped circularly into [0, nzotf)
                double kzindex = kz * kzscale;
                if (kzindex < 0) kzindex += nzotf;
                const Eigen::Index izf = static_cast<Eigen::Index>(std::floor(kzindex));
                const double az = kzindex - static_cast<double>(izf);
                Eigen::Index iz0 = izf % nzotf;
                if (iz0 < 0) iz0 += nzotf;
                const Eigen::Index iz1 = (iz0 + 1) % nzotf;

                // ky contribution is constant across the inner loop -> hoist it
                const double kyt = ky * ryscale;
                const double ky2 = kyt * kyt;

                Cplx* row = dst + (static_cast<Eigen::Index>(iz) * ny + iy) * nx;

                for (int ix = 0; ix < nx; ++ix) {
                    const int kx = (ix <= nx / 2) ? ix : ix - nx;
                    const double kxt = kx * rxscale;

                    const double krindex = std::sqrt(kxt * kxt + ky2);
                    const Eigen::Index ir = static_cast<Eigen::Index>(std::floor(krindex));
                    const double ar = krindex - static_cast<double>(ir);

                    const Cplx v00 = fetch(ir, iz0);
                    const Cplx v01 = fetch(ir, iz1);
                    const Cplx v10 = fetch(ir + 1, iz0);
                    const Cplx v11 = fetch(ir + 1, iz1);

                    // bilinear interpolation
                    row[ix] = (1.0 - ar) * ((1.0 - az) * v00 + az * v01) + ar * ((1.0 - az) * v10 + az * v11);
                }
            }
        }
        return out;
    }

} // namespace sirius
