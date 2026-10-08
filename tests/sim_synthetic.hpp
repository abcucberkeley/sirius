#ifndef SIRIUS_TESTS_SIM_SYNTHETIC_HPP
#define SIRIUS_TESTS_SIM_SYNTHETIC_HPP

// A synthetic 2D SIM acquisition for the reconstruction tests: point emitters
// under a sinusoidal pattern per (direction, phase), blurred by the paraxial
// incoherent OTF (2/pi)(acos s - s sqrt(1 - s^2)), s = kr / (2 NA / lambda),
// on a constant background. The pattern follows the parameters exactly --
// period linespacing_um, direction d at k0_start_angle + d * pi / ndirs,
// phase steps 2 pi / nphases -- so the fit has a known answer. The sections
// are ordered (direction, phase) as a raw 2D stack is.
//
// syntheticSim2dRect takes the extents per axis, so a test can ask for an odd
// or a non-square stack. For an odd stack that scene is the ONLY reference
// there is: cudasirecon's own odd path is wrong at the outermost frequency of
// each axis, so there is no odd-size output to match against
// (docs/findings.md 9k.50, and the note in sirius/sim_reconstruction.hpp).

#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <vector>

#include "sirius/buffer.hpp"
#include "sirius/constants.hpp"
#include "sirius/fft.hpp"
#include "sirius/sim_parameters.hpp"

namespace sirius::test {

    inline Buffer<double> syntheticSim2dRect(const SIMParameters& p, int nx, int ny, double modulation = 0.8,
                                             std::vector<double>* objectOut = nullptr) {
        using Cplx = std::complex<double>;
        const std::size_t nn = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);

        // mt19937's sequence is specified, so the scene is the same everywhere
        std::mt19937 rng(1);
        std::vector<double> object(nn, 0.0);
        for (int i = 0; i < 400; ++i) object[rng() % nn] += 1.0 + static_cast<double>(rng() % 1000);
        if (objectOut) *objectOut = object;

        const double kc = 2.0 * p.na / (p.wavelength_nm * 1e-3);
        // FFT ordering per axis, for either parity: index 0 is zero frequency
        auto freq = [](int i, int n, double d) {
            return static_cast<double>(i <= n / 2 ? i : i - n) / (n * d);
        };
        std::vector<double> otf(nn, 0.0);
        for (int iy = 0; iy < ny; ++iy)
            for (int ix = 0; ix < nx; ++ix) {
                const double s = std::hypot(freq(ix, nx, p.dx), freq(iy, ny, p.dy)) / kc;
                if (s < 1.0)
                    otf[static_cast<std::size_t>(iy) * nx + ix] =
                        2.0 / kPi * (std::acos(s) - s * std::sqrt(1.0 - s * s));
            }

        Buffer<double> raw(Shape{static_cast<Index>(p.ndirs) * p.nphases, ny, nx});
        Buffer<Cplx> field(Shape{ny, nx}), spectrum(Shape{ny, nx});
        FFT fft({ny, nx}, 1, PlanRigor::Estimate);
        const double k0 = 1.0 / p.linespacing_um;
        for (int d = 0; d < p.ndirs; ++d) {
            const double angle = p.k0_start_angle + d * kPi / p.ndirs;
            for (int ph = 0; ph < p.nphases; ++ph) {
                const double phase = 2.0 * kPi * ph / p.nphases;
                for (int iy = 0; iy < ny; ++iy)
                    for (int ix = 0; ix < nx; ++ix) {
                        const std::size_t i = static_cast<std::size_t>(iy) * nx + ix;
                        const double x = ix * p.dx, y = iy * p.dy;
                        const double illumination =
                            1.0 + modulation * std::cos(2.0 * kPi * k0 * (std::cos(angle) * x + std::sin(angle) * y) + phase);
                        field.data()[i] = object[i] * illumination;
                    }
                fft.fft(field.data(), spectrum.data());
                for (std::size_t i = 0; i < nn; ++i) spectrum.data()[i] *= otf[i];
                fft.ifft(spectrum.data(), field.data());   // unnormalized
                double* section = raw.data() + (static_cast<Index>(d) * p.nphases + ph) * static_cast<Index>(nn);
                for (std::size_t i = 0; i < nn; ++i) section[i] = field.data()[i].real() / static_cast<double>(nn) + 100.0;
            }
        }
        return raw;
    }

    // The square case, which is what most callers want.
    inline Buffer<double> syntheticSim2d(const SIMParameters& p, int n, double modulation = 0.8,
                                         std::vector<double>* objectOut = nullptr) {
        return syntheticSim2dRect(p, n, n, modulation, objectOut);
    }

} // namespace sirius::test

#endif // SIRIUS_TESTS_SIM_SYNTHETIC_HPP
