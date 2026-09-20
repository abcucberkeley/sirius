#include "sirius/deconvolution.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "sirius/fft_common.hpp"   // nextFastFFTSize
#include "sirius/real_fft.hpp"

// Richardson-Lucy with the total-variation prior of Dey et al. (Microsc. Res.
// Tech. 69, 2006):
//
//   o_{k+1} = o_k * [ (i / (o_k * h)) (x) h ] / (1 - lambda div(grad o_k / |grad o_k|))
//
// (* convolution, (x) correlation). Both products with the PSF are circular
// convolutions on a grid padded to a fast FFT size, with the image extended
// by edge replication into the padding so the boundary sees a smooth
// continuation instead of a wall of zeros. The estimate is kept in double:
// the ratio image amplifies whatever precision the blur lost.
//
// CPU only for now: RealFFT can run on CUDA, but the element-wise updates
// would need kernels of their own, so a CUDA device is served by the host
// path and the result reports ranOnGpu = false.

namespace sirius {

    namespace {
        using Cplx = std::complex<double>;

        struct Grid {
            Index z = 1, y = 1, x = 1;
            Index size() const noexcept { return z * y * x; }
        };

        Grid gridOf(const Shape& s, const char* what) {
            Grid g;
            if (s.rank() == 2) {
                g.y = s[0];
                g.x = s[1];
            } else if (s.rank() == 3) {
                g.z = s[0];
                g.y = s[1];
                g.x = s[2];
            } else {
                throw std::invalid_argument(std::string(what) + ": expected a (rows, cols) or (depth, rows, cols) view, got " +
                                            s.toString());
            }
            if (g.size() <= 0) throw std::invalid_argument(std::string(what) + ": empty view");
            return g;
        }

        // Padded grid: each axis grows by the PSF's half-extent on both sides
        // (clipped to the image extent) so the blur of one edge cannot wrap
        // onto the other, then rounds up to a fast size.
        Grid paddedGrid(const Grid& img, const Grid& psf) {
            auto pad = [](Index n, Index p) { return n == 1 ? Index{1} : nextFastFFTSize(n + std::min(p, n) - 1); };
            return {pad(img.z, psf.z), pad(img.y, psf.y), pad(img.x, psf.x)};
        }

        // Normalized forward-difference gradient direction at (z, y, x) with
        // clamped indices; eps keeps flat regions from dividing by zero.
        inline void unitGradient(const double* o, const Grid& g, Index z, Index y, Index x, double eps,
                                 double& gz, double& gy, double& gx) noexcept {
            const Index i = (z * g.y + y) * g.x + x;
            const double v = o[i];
            gz = z + 1 < g.z ? o[i + g.y * g.x] - v : 0.0;
            gy = y + 1 < g.y ? o[i + g.x] - v : 0.0;
            gx = x + 1 < g.x ? o[i + 1] - v : 0.0;
            const double norm = std::sqrt(gz * gz + gy * gy + gx * gx + eps * eps);
            gz /= norm;
            gy /= norm;
            gx /= norm;
        }

        // Poll the caller's cancel predicate between two stages. Host-side
        // control flow only: it reads nothing the numerics write, so a run
        // that is never cancelled is bit-identical to one with no callback.
        inline void checkCancelled(const DeconvolutionOptions& options) {
            if (options.cancelled && options.cancelled()) throw std::runtime_error("cancelled");
        }
    } // namespace

    DeconvolutionResult richardsonLucy(BufferView<float> image, BufferView<const float> psf,
                                       const DeconvolutionOptions& options) {
        if (!image.device().isCpu() || !psf.device().isCpu())
            throw std::invalid_argument("richardsonLucy: image and PSF must be host views");
        const Grid img = gridOf(image.shape(), "richardsonLucy image");
        Grid ker = gridOf(psf.shape(), "richardsonLucy psf");
        if (options.iterations < 0) throw std::invalid_argument("richardsonLucy: iterations must be >= 0");

        // A 2D image only uses the PSF's central plane; a 2D PSF is one plane.
        const float* psfData = psf.data();
        if (img.z == 1 && ker.z > 1) {
            psfData += (ker.z / 2) * ker.y * ker.x;
            ker.z = 1;
        }

        const Grid pad = paddedGrid(img, ker);
        const Index n = pad.size();
        const Index nc = pad.z * pad.y * (pad.x / 2 + 1);

        // --- transfer function: PSF centred at its middle voxel, wrapped so the
        // centre sits at the origin, cropped to what the padded grid can hold.
        std::vector<double> work(static_cast<std::size_t>(n), 0.0);
        {
            const Index cz = ker.z / 2, cy = ker.y / 2, cx = ker.x / 2;
            double sum = 0.0;
            for (Index z = 0; z < ker.z; ++z) {
                const Index dz = z - cz;
                if (dz < -(pad.z / 2) || dz > (pad.z - 1) / 2) continue;
                for (Index y = 0; y < ker.y; ++y) {
                    const Index dy = y - cy;
                    if (dy < -(pad.y / 2) || dy > (pad.y - 1) / 2) continue;
                    for (Index x = 0; x < ker.x; ++x) {
                        const Index dx = x - cx;
                        if (dx < -(pad.x / 2) || dx > (pad.x - 1) / 2) continue;
                        const double v = psfData[(z * ker.y + y) * ker.x + x];
                        if (!(v > 0.0)) continue;   // negative / NaN taps are not a PSF
                        const Index wz = (dz + pad.z) % pad.z, wy = (dy + pad.y) % pad.y, wx = (dx + pad.x) % pad.x;
                        work[static_cast<std::size_t>((wz * pad.y + wy) * pad.x + wx)] += v;
                        sum += v;
                    }
                }
            }
            if (!(sum > 0.0)) throw std::invalid_argument("richardsonLucy: the PSF has no positive energy");
            for (double& v : work) v /= sum;
        }

        checkCancelled(options);

        std::vector<int> dims;
        if (pad.z > 1) dims.push_back(static_cast<int>(pad.z));
        dims.push_back(static_cast<int>(pad.y));
        dims.push_back(static_cast<int>(pad.x));
        const RealFFT fft(dims, 1, PlanRigor::Estimate);

        std::vector<Cplx> H(static_cast<std::size_t>(nc)), spec(static_cast<std::size_t>(nc));
        fft.rfft(work.data(), H.data());

        // --- image: non-negative, extended by edge replication into the padding
        std::vector<double> data(static_cast<std::size_t>(n)), estimate(static_cast<std::size_t>(n));
        const Index oz = (pad.z - img.z) / 2, oy = (pad.y - img.y) / 2, ox = (pad.x - img.x) / 2;
#pragma omp parallel for collapse(2) schedule(static)
        for (Index z = 0; z < pad.z; ++z)
            for (Index y = 0; y < pad.y; ++y) {
                const Index sz = std::clamp<Index>(z - oz, 0, img.z - 1), sy = std::clamp<Index>(y - oy, 0, img.y - 1);
                const float* row = image.data() + (sz * img.y + sy) * img.x;
                double* dst = data.data() + (z * pad.y + y) * pad.x;
                for (Index x = 0; x < pad.x; ++x) {
                    const float v = row[std::clamp<Index>(x - ox, 0, img.x - 1)];
                    dst[x] = v > 0.0f && std::isfinite(v) ? static_cast<double>(v) : 0.0;
                }
            }
        estimate = data;   // the classic starting point

        // Guards: the ratio's denominator and the TV normalisation are
        // relative to the data scale so they mean the same for counts and for
        // normalized intensities.
        double peak = 0.0;
        for (double v : data) peak = std::max(peak, v);
        const double eps = std::max(peak * 1e-9, 1e-30);
        const double gradEps = std::max(peak * 1e-6, 1e-30);
        const double lambda = std::max(options.tvLambda, 0.0);

        DeconvolutionResult result;
        std::vector<double> next(static_cast<std::size_t>(n));
        for (int iter = 0; iter < options.iterations; ++iter) {
            checkCancelled(options);
            // blur = estimate * h
            fft.rfft(estimate.data(), spec.data());
            for (Index i = 0; i < nc; ++i) spec[static_cast<std::size_t>(i)] *= H[static_cast<std::size_t>(i)];
            fft.irfft(spec.data(), work.data(), true);
            // ratio = data / blur
#pragma omp parallel for schedule(static)
            for (Index i = 0; i < n; ++i) {
                const double b = work[static_cast<std::size_t>(i)];
                work[static_cast<std::size_t>(i)] = data[static_cast<std::size_t>(i)] / (b > eps ? b : eps);
            }
            checkCancelled(options);
            // correction = ratio (x) h
            fft.rfft(work.data(), spec.data());
            for (Index i = 0; i < nc; ++i) spec[static_cast<std::size_t>(i)] *= std::conj(H[static_cast<std::size_t>(i)]);
            fft.irfft(spec.data(), work.data(), true);

            checkCancelled(options);
            // update, measuring the change over the original region only
            double num = 0.0, den = 0.0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+ : num, den)
            for (Index z = 0; z < pad.z; ++z)
                for (Index y = 0; y < pad.y; ++y) {
                    const bool interior = z >= oz && z < oz + img.z && y >= oy && y < oy + img.y;
                    for (Index x = 0; x < pad.x; ++x) {
                        const Index i = (z * pad.y + y) * pad.x + x;
                        const double o = estimate[static_cast<std::size_t>(i)];
                        double factor = work[static_cast<std::size_t>(i)];
                        if (lambda > 0.0) {
                            // div(grad o / |grad o|) with forward gradients and
                            // backward divergence, the discretisation Dey uses
                            double gz, gy, gx, tz, ty, tx;
                            unitGradient(estimate.data(), pad, z, y, x, gradEps, gz, gy, gx);
                            double div = 0.0;
                            if (z > 0) {
                                unitGradient(estimate.data(), pad, z - 1, y, x, gradEps, tz, ty, tx);
                                div += gz - tz;
                            }
                            if (y > 0) {
                                unitGradient(estimate.data(), pad, z, y - 1, x, gradEps, tz, ty, tx);
                                div += gy - ty;
                            }
                            if (x > 0) {
                                unitGradient(estimate.data(), pad, z, y, x - 1, gradEps, tz, ty, tx);
                                div += gx - tx;
                            }
                            // the prior must stay a positive rescaling
                            factor /= std::max(1.0 - lambda * div, 0.1);
                        }
                        const double v = std::max(o * factor, 0.0);
                        next[static_cast<std::size_t>(i)] = v;
                        if (interior && x >= ox && x < ox + img.x) {
                            num += (v - o) * (v - o);
                            den += o * o;
                        }
                    }
                }
            estimate.swap(next);
            const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
            result.relativeChange.push_back(rel);
            result.iterations = iter + 1;
            if (options.onIteration && !options.onIteration(iter + 1, rel)) {
                result.stoppedEarly = true;
                break;
            }
            if (options.stopRelativeChange > 0.0 && rel < options.stopRelativeChange) {
                result.stoppedEarly = true;
                break;
            }
        }

        // A cancelled run leaves `image` untouched: the write-back below is
        // the only place the caller's view is modified, and it is past every
        // check above.
        // --- back to the caller's float view (interior only)
#pragma omp parallel for collapse(2) schedule(static)
        for (Index z = 0; z < img.z; ++z)
            for (Index y = 0; y < img.y; ++y) {
                const double* src = estimate.data() + ((z + oz) * pad.y + (y + oy)) * pad.x + ox;
                float* dst = image.data() + (z * img.y + y) * img.x;
                for (Index x = 0; x < img.x; ++x) dst[x] = static_cast<float>(src[x]);
            }
        result.ranOnGpu = false;
        return result;
    }

    Buffer<float> gaussianPsf(Index pz, Index py, Index px, double dzUm, double dxyUm, double na,
                              double wavelengthNm, double nimm) {
        if (pz < 1 || py < 1 || px < 1) throw std::invalid_argument("gaussianPsf: extents must be >= 1");
        if (dxyUm <= 0.0 || na <= 0.0 || wavelengthNm <= 0.0 || nimm <= 0.0)
            throw std::invalid_argument("gaussianPsf: voxel size, NA, wavelength and index must be positive");
        // Zhang, Zerubia & Olivo-Marin 2007: Gaussian fits to the paraxial
        // widefield PSF, in the same units as the voxel size.
        const double lambda = wavelengthNm * 1e-3;
        const double sxy = 0.21 * lambda / na;
        const double sz = 0.66 * lambda * nimm / (na * na);
        const double dz = dzUm > 0.0 ? dzUm : dxyUm;
        Buffer<float> psf(Shape{pz, py, px});
        const Index cz = pz / 2, cy = py / 2, cx = px / 2;
        double sum = 0.0;
        for (Index z = 0; z < pz; ++z)
            for (Index y = 0; y < py; ++y)
                for (Index x = 0; x < px; ++x) {
                    const double rz = static_cast<double>(z - cz) * dz / sz;
                    const double ry = static_cast<double>(y - cy) * dxyUm / sxy;
                    const double rx = static_cast<double>(x - cx) * dxyUm / sxy;
                    const double v = std::exp(-0.5 * (rz * rz + ry * ry + rx * rx));
                    psf.data()[(z * py + y) * px + x] = static_cast<float>(v);
                    sum += v;
                }
        for (Index i = 0; i < psf.size(); ++i) psf.data()[i] = static_cast<float>(psf.data()[i] / sum);
        return psf;
    }

} // namespace sirius
