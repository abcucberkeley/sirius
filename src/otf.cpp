#include "sirius/otf.hpp"
#include "sirius/buffer.hpp"
#include "sirius/constants.hpp"
#include "sirius/errors.hpp"
#include "sirius/fft.hpp"
#include "sirius/sim_reconstruction.hpp"
#include "sirius/tiff_io.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

    OTFRadiallyAveraged idealOTF(const SIMParameters& p, bool threeD, const IdealOtfOptions& opts) {
        using Cplx = std::complex<double>;
        p.validate();
        if (p.na >= p.nimm)
            throw std::invalid_argument("idealOTF: NA " + std::to_string(p.na) +
                                        " must be below the immersion index " + std::to_string(p.nimm));
        const int n = opts.lateralSamples;
        if (n < 16 || n % 2 != 0)
            throw std::invalid_argument("idealOTF: lateralSamples must be even and at least 16");
        const int nzotf = threeD ? opts.axialSamples : 1;
        if (nzotf < 1) throw std::invalid_argument("idealOTF: axialSamples must be at least 1");
        const double dzPsf = opts.dzPsf > 0.0 ? opts.dzPsf : p.dz_psf;
        const int norders = p.resolvedOrders();

        const double lambda = p.wavelength_nm * 1e-3;   // um, in vacuum
        const double kn = p.nimm / lambda;              // wavenumber in the medium [1/um]
        const double kPupil = p.na / lambda;            // pupil radius [1/um]
        const double dp = lambda / (8.0 * p.na);        // PSF pixel: grid Nyquist = 4 NA / lambda
        const double dk = 1.0 / (n * dp);               // lateral frequency step of the grid

        // FFT ordering on both axes: index 0 is zero frequency / the PSF origin.
        auto freq = [&](int i) { return static_cast<double>(i < n / 2 ? i : i - n) * dk; };
        const std::size_t nn = static_cast<std::size_t>(n) * n;
        std::vector<double> kz(nn), apod(nn);
        std::vector<char> inside(nn);
        for (int iy = 0; iy < n; ++iy)
            for (int ix = 0; ix < n; ++ix) {
                const std::size_t i = static_cast<std::size_t>(iy) * n + ix;
                const double kx = freq(ix), ky = freq(iy);
                const double kr2 = kx * kx + ky * ky;
                inside[i] = kr2 <= kPupil * kPupil;
                kz[i] = std::sqrt(std::max(0.0, kn * kn - kr2));
                apod[i] = std::sqrt(kz[i] / kn);    // sqrt(cos theta): sine-condition apodization
            }

        // Defocused pupil -> field -> intensity, one plane per z (FFT ordering
        // along z as well, so the 3D transform below yields the periodic kz
        // table the reconstruction interpolates).
        Buffer<Cplx> pupil(Shape{n, n}), field(Shape{n, n});
        Buffer<Cplx> psf(Shape{nzotf, n, n});
        FFT planeFft({n, n}, 1, PlanRigor::Estimate);
        for (int iz = 0; iz < nzotf; ++iz) {
            const double z = static_cast<double>(iz < (nzotf + 1) / 2 ? iz : iz - nzotf) * dzPsf;
            for (std::size_t i = 0; i < nn; ++i)
                pupil.data()[i] = inside[i] ? std::polar(apod[i], 2.0 * kPi * kz[i] * z) : Cplx(0.0, 0.0);
            planeFft.ifft(pupil.data(), field.data());
            Cplx* plane = psf.data() + static_cast<Index>(iz) * static_cast<Index>(nn);
            for (std::size_t i = 0; i < nn; ++i) plane[i] = Cplx(std::norm(field.data()[i]), 0.0);
        }

        // OTF = spectrum of the intensity PSF, then a radial average per kz plane.
        Buffer<Cplx> spec(psf.shape());
        const std::vector<int> dims = threeD ? std::vector<int>{nzotf, n, n} : std::vector<int>{n, n};
        FFT volFft(dims, 1, PlanRigor::Estimate);
        volFft.fft(psf.data(), spec.data());

        const int nkr = n / 2 + 1;
        std::vector<Cplx> radial(static_cast<std::size_t>(nkr) * nzotf, Cplx(0.0, 0.0));
        std::vector<double> count(static_cast<std::size_t>(nkr), 0.0);
        for (int iy = 0; iy < n; ++iy)
            for (int ix = 0; ix < n; ++ix) {
                const double kr = std::hypot(freq(ix), freq(iy));
                const auto ir = static_cast<int>(std::lround(kr / dk));
                if (ir >= nkr) continue;
                count[static_cast<std::size_t>(ir)] += 1.0;
                for (int iz = 0; iz < nzotf; ++iz)
                    radial[static_cast<std::size_t>(ir) * nzotf + iz] +=
                        spec.data()[(static_cast<Index>(iz) * n + iy) * n + ix];
            }
        for (int ir = 0; ir < nkr; ++ir)
            for (int iz = 0; iz < nzotf; ++iz)
                radial[static_cast<std::size_t>(ir) * nzotf + iz] /= std::max(count[static_cast<std::size_t>(ir)], 1.0);
        const double dc = radial[0].real();
        if (!(dc > 0.0)) throw std::runtime_error("idealOTF: degenerate PSF (zero total intensity)");
        for (auto& v : radial) v /= dc;

        Eigen::Tensor<Cplx, 3, Eigen::RowMajor> table(norders, nkr, nzotf);
        for (int order = 0; order < norders; ++order)
            for (int ir = 0; ir < nkr; ++ir)
                for (int iz = 0; iz < nzotf; ++iz)
                    table(order, ir, iz) = radial[static_cast<std::size_t>(ir) * nzotf + iz];

        if (threeD && norders > 1) {
            // First illumination order of the three-beam pattern: lateral
            // frequency k1 (the line spacing refers to the finest, highest
            // order) and the axial frequency of its interference with the
            // central beam.
            const double kex = p.nimm / (0.88 * lambda);
            const double k1 = (1.0 / p.linespacing_um) / static_cast<double>(norders - 1);
            const double kz1 = k1 < kex ? kex - std::sqrt(kex * kex - k1 * k1) : kex;
            const double dkzotf = 1.0 / (nzotf * dzPsf);
            const double shift = kz1 / dkzotf;   // in table planes
            auto sample = [&](int ir, double izf) {
                double pos = std::fmod(izf, static_cast<double>(nzotf));
                if (pos < 0.0) pos += nzotf;
                const int i0 = static_cast<int>(std::floor(pos)) % nzotf;
                const int i1 = (i0 + 1) % nzotf;
                const double a = pos - std::floor(pos);
                return (1.0 - a) * radial[static_cast<std::size_t>(ir) * nzotf + i0] +
                       a * radial[static_cast<std::size_t>(ir) * nzotf + i1];
            };
            for (int ir = 0; ir < nkr; ++ir)
                for (int iz = 0; iz < nzotf; ++iz)
                    table(1, ir, iz) = 0.5 * (sample(ir, iz - shift) + sample(ir, iz + shift));
        }
        return OTFRadiallyAveraged(std::move(table), dk, 1.0 / (nzotf * dzPsf));
    }

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
