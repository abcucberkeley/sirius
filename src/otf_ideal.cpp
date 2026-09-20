#include "sirius/otf_ideal.hpp"
#include "sirius/buffer.hpp"
#include "sirius/constants.hpp"
#include "sirius/fft.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius {

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

} // namespace sirius
