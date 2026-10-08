#include "sirius/otf_measure.hpp"

#include "sirius/constants.hpp"
#include "sirius/errors.hpp"
#include "sirius/real_fft.hpp"
#include "sirius/separation.hpp"
#include "sirius/tiff_io.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace sirius {
    namespace {
        using Cplx = std::complex<double>;
        using Idx = Eigen::Index;
        using Volume = std::vector<double>;      // (nz, ny, nx), x fastest

        std::string num(double v, int digits = 6) {
            std::ostringstream os;
            os << std::setprecision(digits) << v;
            return os.str();
        }

        // --- makeotf's preprocessing -------------------------------------------

        // estimate_background: the mean over a border frame of one section.
        // radialft.cpp's test is `y < b || y > ny - b || x < b || x > nx - b`,
        // asymmetric by one row on the high side; kept as it is so the number
        // is makeotf's number.
        double borderMean(const double* sec, int ny, int nx, int b) {
            double sum = 0.0;
            long long n = 0;
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x)
                    if (y < b || y > ny - b || x < b || x > nx - b) {
                        sum += sec[static_cast<std::size_t>(y) * nx + x];
                        ++n;
                    }
            return n > 0 ? sum / static_cast<double>(n) : 0.0;
        }

        // The mean of a section's darkest fraction. A border frame of a window
        // centred on a bright bead still holds the PSF's out-of-focus haze, and
        // on the user's own stack the two estimators differ by enough to move
        // the table's scale by tens of per cent (otf_measure.hpp).
        double darkestMean(const double* sec, std::size_t n, double fraction) {
            const std::size_t k = std::max<std::size_t>(1, static_cast<std::size_t>(n * fraction));
            std::vector<double> v(sec, sec + n);
            std::nth_element(v.begin(), v.begin() + static_cast<long>(k), v.end());
            return std::accumulate(v.begin(), v.begin() + static_cast<long>(k), 0.0) /
                   static_cast<double>(k);
        }

        // apodize: the edge-blend of radialft.cpp, y pass then x pass, exactly
        // in that order because the second reads what the first wrote. It is
        // sum preserving on both axes (what it adds to row l it takes from row
        // ny-1-l), which is why it cannot move the DC of band 0.
        void apodizeSection(double* sec, int ny, int nx, int napodize) {
            if (napodize <= 0) return;
            const int na = std::min(napodize, std::min(ny, nx) / 2);
            std::vector<double> fact(static_cast<std::size_t>(na));
            for (int l = 0; l < na; ++l)
                fact[static_cast<std::size_t>(l)] =
                    1.0 - std::sin(((static_cast<double>(l) + 0.5) / na) * kPi * 0.5);
            const auto at = [&](int y, int x) -> double& { return sec[static_cast<std::size_t>(y) * nx + x]; };
            for (int x = 0; x < nx; ++x) {
                const double diff = (at(ny - 1, x) - at(0, x)) / 2.0;
                for (int l = 0; l < na; ++l) {
                    at(l, x) += diff * fact[static_cast<std::size_t>(l)];
                    at(ny - 1 - l, x) -= diff * fact[static_cast<std::size_t>(l)];
                }
            }
            for (int y = 0; y < ny; ++y) {
                const double diff = (at(y, nx - 1) - at(y, 0)) / 2.0;
                for (int k = 0; k < na; ++k) {
                    at(y, k) += diff * fact[static_cast<std::size_t>(k)];
                    at(y, nx - 1 - k) -= diff * fact[static_cast<std::size_t>(k)];
                }
            }
        }

        // fitparabola: the sub-pixel peak of the parabola through
        // (-1, a1), (0, a2), (1, a3), and 0 when there is no usable peak --
        // radialft.cpp's own guards included.
        double fitParabola(double a1, double a2, double a3) {
            const double slope = 0.5 * (a3 - a1);
            const double curve = (a3 + a1) - 2.0 * a2;
            if (curve == 0.0) return 0.0;
            const double peak = -slope / curve;
            return (peak > 1.5 || peak < -1.5) ? 0.0 : peak;
        }

        // --- separable blurs, for the difference-of-Gaussians bandpass --------

        void blurAxis(Volume& v, int nz, int ny, int nx, int axis, double sigma) {
            if (!(sigma > 0.0)) return;
            const int n = axis == 0 ? nz : (axis == 1 ? ny : nx);
            if (n < 2) return;
            const std::size_t stride =
                axis == 0 ? static_cast<std::size_t>(ny) * nx : (axis == 1 ? static_cast<std::size_t>(nx) : 1);
            const std::size_t outer = static_cast<std::size_t>(nz) * ny * nx / static_cast<std::size_t>(n);
            std::vector<double> line(static_cast<std::size_t>(n)), out(static_cast<std::size_t>(n));

            // An exact kernel while it is small; three box passes, which are
            // O(n) whatever the width, once 3 sigma leaves the axis -- a
            // 5 um haze blur is 58 px on an 0.0855 um pixel and an exact
            // kernel there costs more than the rest of the measurement.
            const bool exact = 3.0 * sigma <= 12.0;
            std::vector<double> kern;
            int radius = 0, boxw = 0;
            if (exact) {
                radius = std::min(static_cast<int>(std::ceil(3.0 * sigma)), n - 1);
                kern.resize(static_cast<std::size_t>(2 * radius + 1));
                double s = 0.0;
                for (int i = -radius; i <= radius; ++i) {
                    const double w = std::exp(-0.5 * (i * i) / (sigma * sigma));
                    kern[static_cast<std::size_t>(i + radius)] = w;
                    s += w;
                }
                for (auto& w : kern) w /= s;
            } else {
                boxw = std::max(1, static_cast<int>(std::round(std::sqrt(12.0 * sigma * sigma / 3.0 + 1.0))));
                if (boxw % 2 == 0) ++boxw;
                boxw = std::min(boxw, 2 * (n - 1) + 1);
            }

            // Walk every line along `axis`. The index arithmetic below turns a
            // flat outer counter into the base offset of one line.
            const std::size_t nOuterFast = axis == 2 ? 1u : static_cast<std::size_t>(nx);
            for (std::size_t o = 0; o < outer; ++o) {
                std::size_t base = 0;
                if (axis == 0) base = o;                                     // (y, x) pair
                else if (axis == 1) base = (o / nOuterFast) * static_cast<std::size_t>(ny) * nx + (o % nOuterFast);
                else base = o * static_cast<std::size_t>(nx);
                for (int i = 0; i < n; ++i) line[static_cast<std::size_t>(i)] = v[base + static_cast<std::size_t>(i) * stride];
                if (exact) {
                    for (int i = 0; i < n; ++i) {
                        double acc = 0.0;
                        for (int k = -radius; k <= radius; ++k) {
                            int j = i + k;
                            if (j < 0) j = 0;
                            if (j >= n) j = n - 1;
                            acc += kern[static_cast<std::size_t>(k + radius)] * line[static_cast<std::size_t>(j)];
                        }
                        out[static_cast<std::size_t>(i)] = acc;
                    }
                } else {
                    const int r = boxw / 2;
                    for (int pass = 0; pass < 3; ++pass) {
                        double acc = 0.0;
                        for (int k = -r; k <= r; ++k) {
                            int j = std::clamp(k, 0, n - 1);
                            acc += line[static_cast<std::size_t>(j)];
                        }
                        for (int i = 0; i < n; ++i) {
                            out[static_cast<std::size_t>(i)] = acc / boxw;
                            const int add = std::clamp(i + r + 1, 0, n - 1);
                            const int sub = std::clamp(i - r, 0, n - 1);
                            acc += line[static_cast<std::size_t>(add)] - line[static_cast<std::size_t>(sub)];
                        }
                        line = out;
                    }
                }
                for (int i = 0; i < n; ++i) v[base + static_cast<std::size_t>(i) * stride] = out[static_cast<std::size_t>(i)];
            }
        }

        Volume blurred(const Volume& v, int nz, int ny, int nx, double sz, double sy, double sx) {
            Volume w = v;
            blurAxis(w, nz, ny, nx, 0, sz);
            blurAxis(w, nz, ny, nx, 1, sy);
            blurAxis(w, nz, ny, nx, 2, sx);
            return w;
        }

        // --- the per-region Gaussian fit --------------------------------------

        struct Roi {
            int z0 = 0, z1 = 0, y0 = 0, y1 = 0, x0 = 0, x1 = 0;   // half open
        };

        // A, x0, y0, z0, sx, sy, sz, b -- eight parameters, Levenberg damped
        // Gauss-Newton, started from the region's background-subtracted
        // moments. Returns false when it leaves the region or stops improving
        // without having converged.
        bool fitGaussian3d(const Volume& vol, int ny, int nx, const Roi& r, BeadFit& fit) {
            const auto at = [&](int z, int y, int x) {
                return vol[(static_cast<std::size_t>(z) * ny + y) * nx + x];
            };
            // the region's own floor: the median of its outer shell
            std::vector<double> shell;
            for (int z = r.z0; z < r.z1; ++z)
                for (int y = r.y0; y < r.y1; ++y)
                    for (int x = r.x0; x < r.x1; ++x)
                        if (y == r.y0 || y == r.y1 - 1 || x == r.x0 || x == r.x1 - 1) shell.push_back(at(z, y, x));
            double floorv = 0.0;
            if (!shell.empty()) {
                std::nth_element(shell.begin(), shell.begin() + static_cast<long>(shell.size() / 2), shell.end());
                floorv = shell[shell.size() / 2];
            }
            double wsum = 0.0, mx = 0.0, my = 0.0, mz = 0.0, peak = -std::numeric_limits<double>::infinity();
            for (int z = r.z0; z < r.z1; ++z)
                for (int y = r.y0; y < r.y1; ++y)
                    for (int x = r.x0; x < r.x1; ++x) {
                        const double w = std::max(0.0, at(z, y, x) - floorv);
                        wsum += w;
                        mx += w * x;
                        my += w * y;
                        mz += w * z;
                        peak = std::max(peak, at(z, y, x) - floorv);
                    }
            if (!(wsum > 0.0) || !(peak > 0.0)) return false;
            mx /= wsum;
            my /= wsum;
            mz /= wsum;
            double vx = 0.0, vy = 0.0, vz = 0.0;
            for (int z = r.z0; z < r.z1; ++z)
                for (int y = r.y0; y < r.y1; ++y)
                    for (int x = r.x0; x < r.x1; ++x) {
                        const double w = std::max(0.0, at(z, y, x) - floorv);
                        vx += w * (x - mx) * (x - mx);
                        vy += w * (y - my) * (y - my);
                        vz += w * (z - mz) * (z - mz);
                    }
            Eigen::Matrix<double, 8, 1> p;
            p << peak, mx, my, mz, std::max(0.5, std::sqrt(vx / wsum)), std::max(0.5, std::sqrt(vy / wsum)),
                std::max(0.5, std::sqrt(vz / wsum)), floorv;

            const auto residualOf = [&](const Eigen::Matrix<double, 8, 1>& q) {
                double ss = 0.0;
                for (int z = r.z0; z < r.z1; ++z)
                    for (int y = r.y0; y < r.y1; ++y)
                        for (int x = r.x0; x < r.x1; ++x) {
                            const double ex = (x - q(1)) / q(4), ey = (y - q(2)) / q(5), ez = (z - q(3)) / q(6);
                            const double g = std::exp(-0.5 * (ex * ex + ey * ey + ez * ez));
                            const double d = q(0) * g + q(7) - at(z, y, x);
                            ss += d * d;
                        }
                return ss;
            };
            double ss = residualOf(p);
            double lambda = 1e-3;
            bool converged = false;
            for (int iter = 0; iter < 60; ++iter) {
                Eigen::Matrix<double, 8, 8> JtJ = Eigen::Matrix<double, 8, 8>::Zero();
                Eigen::Matrix<double, 8, 1> Jtr = Eigen::Matrix<double, 8, 1>::Zero();
                for (int z = r.z0; z < r.z1; ++z)
                    for (int y = r.y0; y < r.y1; ++y)
                        for (int x = r.x0; x < r.x1; ++x) {
                            const double ex = (x - p(1)) / p(4), ey = (y - p(2)) / p(5), ez = (z - p(3)) / p(6);
                            const double g = std::exp(-0.5 * (ex * ex + ey * ey + ez * ez));
                            const double Ag = p(0) * g;
                            Eigen::Matrix<double, 8, 1> J;
                            J << g, Ag * ex / p(4), Ag * ey / p(5), Ag * ez / p(6), Ag * ex * ex / p(4),
                                Ag * ey * ey / p(5), Ag * ez * ez / p(6), 1.0;
                            const double d = Ag + p(7) - at(z, y, x);
                            JtJ.noalias() += J * J.transpose();
                            Jtr.noalias() += J * d;
                        }
                Eigen::Matrix<double, 8, 8> A = JtJ;
                for (int i = 0; i < 8; ++i) A(i, i) *= (1.0 + lambda);
                const Eigen::Matrix<double, 8, 1> step = A.ldlt().solve(-Jtr);
                if (!step.allFinite()) break;
                Eigen::Matrix<double, 8, 1> q = p + step;
                q(4) = std::abs(q(4));
                q(5) = std::abs(q(5));
                q(6) = std::abs(q(6));
                if (!(q(4) > 1e-3) || !(q(5) > 1e-3) || !(q(6) > 1e-3)) break;
                const double ss2 = residualOf(q);
                if (ss2 < ss) {
                    const double rel = (ss - ss2) / std::max(ss, 1e-30);
                    p = q;
                    ss = ss2;
                    lambda = std::max(lambda * 0.3, 1e-9);
                    if (rel < 1e-10) {
                        converged = true;
                        break;
                    }
                } else {
                    lambda *= 8.0;
                    if (lambda > 1e8) {
                        converged = true;
                        break;
                    }
                }
            }
            const long long nvox = static_cast<long long>(r.z1 - r.z0) * (r.y1 - r.y0) * (r.x1 - r.x0);
            fit.amplitude = p(0);
            fit.x = p(1);
            fit.y = p(2);
            fit.z = p(3);
            fit.sigmaX = p(4);
            fit.sigmaY = p(5);
            fit.sigmaZ = p(6);
            fit.offset = p(7);
            fit.residual = p(0) > 0.0 ? std::sqrt(ss / static_cast<double>(nvox)) / p(0) : 1.0;
            const bool inside = p(1) >= r.x0 - 0.5 && p(1) <= r.x1 - 0.5 && p(2) >= r.y0 - 0.5 &&
                                p(2) <= r.y1 - 0.5 && p(3) >= r.z0 - 0.5 && p(3) <= r.z1 - 0.5;
            return converged && inside && p(0) > 0.0;
        }

        // --- the radial average ------------------------------------------------

        // radialft.cpp: bin by rint(|k| / dkr) with dkr = 1 / (min(nx, ny) dxy),
        // drop whatever is beyond the sampling's own lateral limit, average by
        // the count, then force a real kz = 0 and Hermitian symmetry in kz --
        // which is why every table makeotf writes measures exactly 0 on
        // loadOTF's Hermitian check.
        struct RadialPlan {
            int nkr = 0;
            double dkr = 0.0;
            std::vector<int> bin;      // (ny, nx/2+1), -1 outside the limit
        };

        RadialPlan radialPlan(int ny, int nx, double dxy) {
            RadialPlan p;
            const int nr = std::min(nx, ny);
            p.nkr = nr / 2 + 1;
            p.dkr = 1.0 / (nr * dxy);
            const double dkx = 1.0 / (nx * dxy), dky = 1.0 / (ny * dxy), maxK = 0.5 / dxy;
            const int half = nx / 2 + 1;
            p.bin.assign(static_cast<std::size_t>(ny) * half, -1);
            for (int iy = 0; iy < ny; ++iy) {
                const double ky = (iy > ny / 2 ? iy - ny : iy) * dky;
                for (int ix = 0; ix < half; ++ix) {
                    const double kx = ix * dkx;
                    const double rd = std::hypot(kx, ky);
                    if (rd < maxK) {
                        const int b = static_cast<int>(std::lround(rd / p.dkr));
                        if (b < p.nkr) p.bin[static_cast<std::size_t>(iy) * half + ix] = b;
                    }
                }
            }
            return p;
        }

        void radialAverage(const Cplx* band, int nz, int ny, int nx, const RadialPlan& plan, Cplx* out) {
            const int half = nx / 2 + 1;
            std::vector<double> count(static_cast<std::size_t>(plan.nkr) * nz, 0.0);
            std::fill(out, out + static_cast<std::size_t>(plan.nkr) * nz, Cplx(0.0, 0.0));
            for (int iz = 0; iz < nz; ++iz)
                for (int iy = 0; iy < ny; ++iy)
                    for (int ix = 0; ix < half; ++ix) {
                        const int b = plan.bin[static_cast<std::size_t>(iy) * half + ix];
                        if (b < 0) continue;
                        const std::size_t o = static_cast<std::size_t>(b) * nz + iz;
                        out[o] += band[(static_cast<std::size_t>(iz) * ny + iy) * half + ix];
                        count[o] += 1.0;
                    }
            for (std::size_t i = 0; i < static_cast<std::size_t>(plan.nkr) * nz; ++i)
                if (count[i] > 0.0) out[i] /= count[i];
            for (int ir = 0; ir < plan.nkr; ++ir) {
                Cplx* col = out + static_cast<std::size_t>(ir) * nz;
                col[0] = Cplx(col[0].real(), 0.0);
                for (int k = 1; k <= nz / 2; ++k) {
                    const Cplx a = (col[k] + std::conj(col[nz - k])) / 2.0;
                    col[k] = a;
                    col[nz - k] = std::conj(a);
                }
            }
        }

        // modify(): the kr = 0 column is the kx = ky = 0 line of the transform,
        // which holds the section means and nothing about the optics, so it is
        // replaced by kr = 1 -- except order 0's kz = 0, which is the
        // normalisation DC and is left for the scale step to deal with.
        void repairKr0(Cplx* table, int nkr, int nz, int order) {
            if (nkr < 2) return;
            const int from = (order > 0) ? 0 : 1;
            for (int k = from; k < nz; ++k) table[k] = table[static_cast<std::size_t>(nz) + k];
        }

        // fixorigin(): a least-squares line through the kz-SUMMED real profile
        // over [first, last] (sample 0 is the DC alone, "don't want to add up
        // garbages on kz axis"), extrapolated to kr = 0. `repair` replaces the
        // kz = 0 samples below `first` with the line, as makeotf does.
        double lineFitOrigin(Cplx* table, int nkr, int nz, int first, int last, bool repair) {
            const int lo = std::max(1, first);
            const int hi = std::min(last, nkr - 1);
            if (hi <= lo) return std::numeric_limits<double>::quiet_NaN();
            const double meani = 0.5 * (lo + hi);
            std::vector<double> sum(static_cast<std::size_t>(hi + 1), 0.0);
            sum[0] = table[0].real();
            for (int i = 1; i <= hi; ++i)
                for (int k = 0; k < nz; ++k) sum[static_cast<std::size_t>(i)] += table[static_cast<std::size_t>(i) * nz + k].real();
            double totsum = 0.0, ysum = 0.0, sqsum = 0.0;
            for (int i = lo; i <= hi; ++i) {
                totsum += sum[static_cast<std::size_t>(i)];
                ysum += sum[static_cast<std::size_t>(i)] * (i - meani);
                sqsum += (i - meani) * (i - meani);
            }
            const double slope = sqsum > 0.0 ? ysum / sqsum : 0.0;
            const double avg = totsum / (hi - lo + 1);
            const double atZero = avg + (0.0 - meani) * slope;
            if (repair)
                for (int i = 0; i < lo; ++i) {
                    const double lineval = avg + (i - meani) * slope;
                    Cplx& v = table[static_cast<std::size_t>(i) * nz];
                    v = Cplx(v.real() - (sum[static_cast<std::size_t>(i)] - lineval), v.imag());
                }
            return atZero;
        }

        // combine_reim(): the side band carries one constant phase -- the
        // illumination phase at the bead -- and this rotates it into the real
        // part. The quadrant logic is radialft.cpp's, probe sample included.
        void combineReIm(Cplx* re, Cplx* im, int nkr, int nz, int nxForProbe) {
            double rm = 0.0, imm = 0.0;
            const std::size_t n = static_cast<std::size_t>(nkr) * nz;
            for (std::size_t i = 0; i < n; ++i) {
                rm += std::abs(re[i]);
                imm += std::abs(im[i]);
            }
            const std::size_t probe0 = static_cast<std::size_t>(std::min(nxForProbe / 4, nkr - 1)) * nz;
            if (!(rm > 0.0)) {
                // cos(theta) == 0: makeotf would divide by zero here. The
                // rotation is then a quarter turn, whose sign the probe sample
                // gives, and nothing is left unrotated.
                if (!(imm > 0.0)) return;
                const double quarter = im[probe0].real() >= 0.0 ? -0.5 * kPi : 0.5 * kPi;
                for (std::size_t i = 0; i < n; ++i) {
                    re[i] = re[i] * std::cos(quarter) + im[i] * std::sin(-quarter);
                    im[i] = Cplx(0.0, 0.0);
                }
                return;
            }
            double phi = std::atan(imm / rm);
            const std::size_t probe = probe0;
            if (re[probe].real() < 0 && im[probe].real() > 0) phi += kPi;
            else if (re[probe].real() < 0 && im[probe].real() < 0) phi = kPi - phi;
            else if (re[probe].real() > 0 && im[probe].real() > 0) phi = -phi;
            for (std::size_t i = 0; i < n; ++i) {
                re[i] = re[i] * std::cos(phi) + im[i] * std::sin(-phi);
                im[i] = Cplx(0.0, 0.0);
            }
        }

        // sphereFFT / limit_at_origin: Mats's transform of a uniform sphere,
        // which is what a bead of finite diameter divides out of every band.
        double sphereRatio(double k, double radius) {
            const double lim = 4.0 * kPi * radius * radius * radius / 3.0;
            if (k == 0.0) return 1.0;
            const double x = 2.0 * kPi * radius * k;
            return (radius / (kPi * k * k) * (std::sin(x) / x - std::cos(x))) / lim;
        }

        // rescale(): makeotf takes scalefactor = 1 / otf[0].real() from order 0
        // and multiplies every order by it (dorescale, its -rescale flag, which
        // would normalise each order to its own DC and set every modulation
        // depth to 1 by construction, is off by default and is not offered
        // here). `repaired` is order 0's kr = kz = 0 sample after the optional
        // fixorigin repair, `raw` the same sample before it.
        double scaleDivisorOf(double repaired, double raw, OtfMeasureScale scale) {
            const double d = scale == OtfMeasureScale::Order0Dc
                                 ? raw
                                 : (scale == OtfMeasureScale::MakeotfFixOrigin ? repaired : 1.0);
            return (std::isfinite(d) && d != 0.0) ? d : 1.0;
        }

        struct BandSet {
            int nbands = 0, nkr = 0, nz = 0;
            std::vector<Cplx> table;    // (nbands, nkr, nz)
            Cplx* band(int b) { return table.data() + static_cast<std::size_t>(b) * nkr * nz; }
        };
    } // namespace

    const char* beadRejectionName(BeadRejection r) noexcept {
        switch (r) {
            case BeadRejection::None: return "kept";
            case BeadRejection::Amplitude: return "below the amplitude floor";
            case BeadRejection::TooClose: return "within the minimum separation of a brighter bead";
            case BeadRejection::NearBoundary: return "too close to an edge for its region";
            case BeadRejection::WidthTooSmall: return "narrower than the width bound";
            case BeadRejection::WidthTooLarge: return "wider than the width bound";
            case BeadRejection::FitFailed: return "the Gaussian fit did not converge in its region";
            case BeadRejection::Saturated: return "saturated";
            case BeadRejection::OverMaxBeads: return "left out by the maxBeads cap";
            case BeadRejection::NotExamined: return "not examined: past the candidate cap";
        }
        return "unknown";
    }

    std::string validateOtfMeasure(const OtfMeasureOptions& o, int sections, int ny, int nx) {
        if (o.nphases < 1) return "nphases must be at least 1, not " + std::to_string(o.nphases);
        if (sections <= 0 || ny <= 0 || nx <= 0)
            return "the stack is empty (" + std::to_string(sections) + ", " + std::to_string(ny) + ", " +
                   std::to_string(nx) + ")";
        if (sections % o.nphases != 0)
            return "the stack holds " + std::to_string(sections) + " sections, which " +
                   std::to_string(o.nphases) + " phases do not divide: a raw bead stack is nz * nphases";
        const int nz = sections / o.nphases;
        const int maxOrders = (o.nphases + 1) / 2;
        const int norders = o.norders > 0 ? o.norders : maxOrders;
        if (norders > maxOrders)
            return "norders " + std::to_string(norders) + " needs at least " + std::to_string(2 * norders - 1) +
                   " phases, and this stack has " + std::to_string(o.nphases);
        if (!o.phases.empty() && static_cast<int>(o.phases.size()) != o.nphases)
            return "phases holds " + std::to_string(o.phases.size()) + " values for " +
                   std::to_string(o.nphases) + " phases";
        if (!(o.dxy > 0.0)) return "dxy (the lateral pixel size, um) must be given and positive";
        if (!(o.dz > 0.0) && nz > 1) return "dz (the axial step, um) must be given and positive for a 3D stack";
        if (std::min(nx, ny) < 8)
            return "the lateral size is " + std::to_string(nx) + " x " + std::to_string(ny) +
                   ": an OTF needs at least 8 px, since nkr = min(nx, ny) / 2 + 1";
        if (o.backgroundBorder < 0 || 2 * o.backgroundBorder >= std::min(nx, ny))
            return "backgroundBorder " + std::to_string(o.backgroundBorder) +
                   " does not leave an interior in a " + std::to_string(nx) + " x " + std::to_string(ny) +
                   " section";
        if (!(o.darkestFraction > 0.0) || o.darkestFraction > 1.0)
            return "darkestFraction must lie in (0, 1], not " + std::to_string(o.darkestFraction);
        if (o.beadDiameterUm < 0.0) return "beadDiameterUm must not be negative";
        // 0 means "not stated" and falls back to makeotf's own default (the
        // measurement says which was used and that it fell back); a negative
        // spacing is a mistake, not a choice.
        if (o.patternPeriodUm < 0.0)
            return "patternPeriodUm (the illumination line spacing, um) must not be negative; 0 states that "
                   "the acquisition does not say, and the bead-size division then falls back to makeotf's " +
                   num(kMakeotfLineSpacingUm) + " um";
        if (o.scale == OtfMeasureScale::MakeotfFixOrigin) {
            const int nkr = std::min(nx, ny) / 2 + 1;
            if (std::max(1, o.lineFitFirst) + 1 >= std::min(o.lineFitLast, nkr - 1))
                return "the line-fit window (" + std::to_string(o.lineFitFirst) + ", " +
                       std::to_string(o.lineFitLast) + ") is empty for " + std::to_string(nkr) +
                       " radial samples";
        }
        if (o.field && o.detect.maxBeads < 1) return "maxBeads must be at least 1";
        return std::string();
    }

    std::string OtfMeasureResult::summary() const {
        std::ostringstream os;
        os << norders << " orders x " << nkr << " kr x " << nzotf << " kz, dkr " << num(dkr) << " dkz "
           << num(dkz) << " 1/um, measured at dxy " << num(dxy) << " dz " << num(dz) << " um; ";
        os << (beads.size() > 1 || kept > 1 ? "field: " : "one bead: ") << kept << " of " << found
           << " beads used; ";
        os << "scale ";
        switch (scaleUsed) {
            case OtfMeasureScale::Order0Dc: os << "order 0's DC"; break;
            case OtfMeasureScale::MakeotfFixOrigin: os << "makeotf's fixorigin repair"; break;
            case OtfMeasureScale::AsMeasured: os << "as measured"; break;
        }
        os << " = " << num(scaleDivisor) << " (DC " << num(order0Dc) << ", line fit "
           << num(lineFitToOrigin) << ", ratio "
           << (lineFitToOrigin != 0.0 ? num(order0Dc / lineFitToOrigin, 4) : std::string("n/a")) << "); ";
        os << "depth as stored " << num(modulationDepth, 5);
        if (modulationDepthSpread > 0.0) os << " +- " << num(modulationDepthSpread, 3) << " across beads";
        os << ", band ratio " << num(bandRatio, 5) << " (iqr " << num(bandRatioIqr, 3) << " over "
           << bandRatioSamples << " samples); DC is " << num(100.0 * dcFractionOfSignal, 3)
           << "% of the stack's integral, so 1 ADU of background is "
           << num(100.0 * scaleSensitivityPerAdu, 3) << "% of the scale";
        for (const auto& n : notes) os << "; " << n;
        return os.str();
    }

    OtfMeasureResult measureOTF(const Eigen::Tensor<double, 3, Eigen::RowMajor>& stack,
                                const OtfMeasureOptions& o) {
        const int sections = static_cast<int>(stack.dimension(0));
        const int ny = static_cast<int>(stack.dimension(1));
        const int nx = static_cast<int>(stack.dimension(2));
        if (const std::string bad = validateOtfMeasure(o, sections, ny, nx); !bad.empty())
            throw std::invalid_argument("measureOTF: " + bad);

        const int nphases = o.nphases;
        const int nz = sections / nphases;
        const int norders = o.norders > 0 ? o.norders : (nphases + 1) / 2;
        const int nbands = 2 * norders - 1;
        const int half = nx / 2 + 1;
        const std::size_t nvox = static_cast<std::size_t>(nz) * ny * nx;
        const std::size_t nsec = static_cast<std::size_t>(ny) * nx;

        OtfMeasureResult res;
        res.nz = nz;
        res.ny = ny;
        res.nx = nx;
        res.dxy = o.dxy;
        res.dz = o.dz > 0.0 ? o.dz : 1.0;
        res.norders = norders;
        res.scaleUsed = o.scale;

        // --- 1/2: background and apodization, per section, as read ------------
        // The order is radialft.cpp's: the background is estimated from the raw
        // section, the apodization is applied, the centre is found on the
        // apodized but NOT yet background-subtracted volume, and only then is
        // the background taken off.
        std::vector<double> work(static_cast<std::size_t>(nphases) * nvox);
        std::vector<double> bg(static_cast<std::size_t>(sections), 0.0);
        double total = 0.0, borderSum = 0.0, darkSum = 0.0;
        for (int s = 0; s < sections; ++s) {
            const int z = o.packing == BeadPhasePacking::PhaseFastest ? s / nphases : s % nz;
            const int p = o.packing == BeadPhasePacking::PhaseFastest ? s % nphases : s / nz;
            const double* src = stack.data() + static_cast<std::size_t>(s) * nsec;
            double* dst = work.data() + (static_cast<std::size_t>(p) * nz + z) * nsec;
            std::copy(src, src + nsec, dst);
            // both estimators every time: the one not used is a diagnostic, and
            // the gap between them is what the scale's uncertainty is made of
            const double border = borderMean(src, ny, nx, o.backgroundBorder);
            const double dark = darkestMean(src, nsec, o.darkestFraction);
            borderSum += border;
            darkSum += dark;
            bg[static_cast<std::size_t>(s)] =
                o.background >= 0.0
                    ? o.background
                    : (o.backgroundEstimate == BackgroundEstimate::DarkestFraction ? dark : border);
            for (std::size_t i = 0; i < nsec; ++i) total += src[i];
            apodizeSection(dst, ny, nx, o.apodize);
        }
        res.backgroundBorderMean = borderSum / sections;
        res.backgroundDarkest = darkSum / sections;
        {
            // makeotf's border_size is 20 px whatever the section is, so on a
            // small or bead-crowded window the "border" is most of the image
            // and the estimate eats signal. Measured consequence: a constant
            // over-subtraction is a negative pedestal whose transform sits at
            // low kr, which depresses the DC and the first radial samples
            // exactly where the scale is read.
            const int b = o.backgroundBorder;
            const long long inner = static_cast<long long>(std::max(0, ny - 2 * b + 1)) *
                                    std::max(0, nx - 2 * b + 1);
            const double frac = 1.0 - static_cast<double>(inner) / static_cast<double>(nsec);
            if (o.background < 0.0 && o.backgroundEstimate == BackgroundEstimate::BorderMean && frac > 0.5)
                res.notes.emplace_back("the background border of " + std::to_string(b) +
                                       " px covers " + num(100.0 * frac, 3) + "% of a " +
                                       std::to_string(nx) + " x " + std::to_string(ny) +
                                       " section, so the estimate sees the beads: consider "
                                       "BackgroundEstimate::DarkestFraction or a smaller border");
        }
        res.totalSignal = total;
        res.backgroundTotal = std::accumulate(bg.begin(), bg.end(), 0.0) * static_cast<double>(nsec);
        res.backgroundMean = std::accumulate(bg.begin(), bg.end(), 0.0) / static_cast<double>(sections);
        {
            double v = 0.0;
            for (double b : bg) v += (b - res.backgroundMean) * (b - res.backgroundMean);
            res.backgroundSd = std::sqrt(v / static_cast<double>(sections));
        }

        // the phase-averaged volume: makeotf's determine_center input, and the
        // widefield the field path detects on
        Volume widefield(nvox, 0.0);
        for (int p = 0; p < nphases; ++p) {
            const double* src = work.data() + static_cast<std::size_t>(p) * nvox;
            for (std::size_t i = 0; i < nvox; ++i) widefield[i] += src[i] / nphases;
        }

        // --- 4: background off, then the band separation ----------------------
        for (int s = 0; s < sections; ++s) {
            const int z = o.packing == BeadPhasePacking::PhaseFastest ? s / nphases : s % nz;
            const int p = o.packing == BeadPhasePacking::PhaseFastest ? s % nphases : s / nz;
            double* dst = work.data() + (static_cast<std::size_t>(p) * nz + z) * nsec;
            const double b = bg[static_cast<std::size_t>(s)];
            for (std::size_t i = 0; i < nsec; ++i) dst[i] -= b;
        }
        // makeotf's makematrix is sirius::separationMatrix divided by nphases.
        // The factor cancels in the rescale, but dividing here keeps the DC and
        // the band amplitudes this result reports equal to cudasirecon's.
        Eigen::MatrixXd M;
        if (o.phases.empty()) {
            M = separationMatrix(nphases, norders);
        } else {
            const Eigen::VectorXd ph =
                Eigen::Map<const Eigen::VectorXd>(o.phases.data(), static_cast<Idx>(o.phases.size()));
            M = separationMatrix(ph, norders);
        }
        M /= static_cast<double>(nphases);
        std::vector<double> bands(static_cast<std::size_t>(nbands) * nvox, 0.0);
        for (int b = 0; b < nbands; ++b) {
            double* dst = bands.data() + static_cast<std::size_t>(b) * nvox;
            for (int p = 0; p < nphases; ++p) {
                const double c = M(b, p);
                if (c == 0.0) continue;
                const double* src = work.data() + static_cast<std::size_t>(p) * nvox;
                for (std::size_t i = 0; i < nvox; ++i) dst[i] += c * src[i];
            }
        }
        work.clear();
        work.shrink_to_fit();

        // --- the bead inventory ------------------------------------------------
        std::vector<BeadFit> beads;
        // Candidates `found` counts but `beads` does not hold, because the
        // amplitude floor and the examination cap cut the sorted list short
        // before any fit was attempted. They still get a bucket each, so that
        // sum(rejected) == found.
        int extraBelowFloor = 0;
        int extraNotExamined = 0;
        if (!o.field) {
            // determine_center: the global maximum of the phase-averaged volume
            // to sub-pixel by three parabola fits, with the wrap radialft uses.
            std::size_t imax = 0;
            double best = -std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < nvox; ++i)
                if (widefield[i] > best) {
                    best = widefield[i];
                    imax = i;
                }
            const int kx = static_cast<int>(imax % static_cast<std::size_t>(nx));
            const int ky = static_cast<int>((imax / nx) % static_cast<std::size_t>(ny));
            const int kz = static_cast<int>(imax / nsec);
            const auto wf = [&](int z, int y, int x) {
                return widefield[(static_cast<std::size_t>(z) * ny + y) * nx + x];
            };
            BeadFit f;
            f.x = kx + fitParabola(wf(kz, ky, (kx - 1 + nx) % nx), best, wf(kz, ky, (kx + 1) % nx));
            f.y = ky + fitParabola(wf(kz, (ky - 1 + ny) % ny, kx), best, wf(kz, (ky + 1) % ny, kx));
            f.z = nz > 1 ? kz + fitParabola(wf((kz - 1 + nz) % nz, ky, kx), best, wf((kz + 1) % nz, ky, kx))
                         : static_cast<double>(kz);
            f.amplitude = best;
            f.kept = true;
            beads.push_back(f);
            res.found = 1;
            res.notes.emplace_back("one bead, the global maximum, as makeotf does");
        } else {
            const auto& d = o.detect;
            // the difference-of-Gaussians bandpass, on the volume so the axial
            // blur takes the haze out, and then detection on its maximum
            // projection along z: mcSIM localises on a 2D widefield image for
            // the same reason, that a bead is a local maximum in many z planes
            // and a 3D maximum test with no axial separation would return one
            // candidate per plane.
            const double sS = d.dogSmallLateralUm / o.dxy, sSz = d.dogSmallAxialUm / res.dz;
            const double sL = d.dogLargeLateralUm / o.dxy, sLz = d.dogLargeAxialUm / res.dz;
            Volume dog = blurred(widefield, nz, ny, nx, sSz, sS, sS);
            const Volume large = blurred(widefield, nz, ny, nx, sLz, sL, sL);
            for (std::size_t i = 0; i < nvox; ++i) dog[i] -= large[i];
            std::vector<double> proj(nsec, -std::numeric_limits<double>::infinity());
            std::vector<int> argz(nsec, 0);
            for (int z = 0; z < nz; ++z)
                for (std::size_t i = 0; i < nsec; ++i) {
                    const double v = dog[static_cast<std::size_t>(z) * nsec + i];
                    if (v > proj[i]) {
                        proj[i] = v;
                        argz[i] = z;
                    }
                }
            const int sepXY = std::max(1, static_cast<int>(std::round(d.minSeparationLateralUm / o.dxy)));
            const int sepZ = std::max(0, static_cast<int>(std::round(d.minSeparationAxialUm / res.dz)));
            std::vector<BeadFit> cand;
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x) {
                    const double v = proj[static_cast<std::size_t>(y) * nx + x];
                    if (!(v > 0.0)) continue;
                    bool isMax = true;
                    for (int dyi = -sepXY; dyi <= sepXY && isMax; ++dyi)
                        for (int dxi = -sepXY; dxi <= sepXY; ++dxi) {
                            const int yy = y + dyi, xx = x + dxi;
                            if (yy < 0 || yy >= ny || xx < 0 || xx >= nx) continue;
                            if (dyi == 0 && dxi == 0) continue;
                            if (proj[static_cast<std::size_t>(yy) * nx + xx] > v) {
                                isMax = false;
                                break;
                            }
                        }
                    if (!isMax) continue;
                    BeadFit f;
                    f.x = x;
                    f.y = y;
                    f.z = argz[static_cast<std::size_t>(y) * nx + x];
                    f.amplitude = v;
                    cand.push_back(f);
                }
            std::sort(cand.begin(), cand.end(),
                      [](const BeadFit& a, const BeadFit& b) { return a.amplitude > b.amplitude; });
            res.found = static_cast<int>(cand.size());
            const double brightest = cand.empty() ? 0.0 : cand.front().amplitude;
            const double floorAmp = std::max(d.minAmplitude, d.minAmplitudeFraction * brightest);
            // the list is sorted, so everything past the first candidate below
            // the floor is below it too: counted, not fitted
            std::size_t examine = cand.size();
            for (std::size_t i = 0; i < cand.size(); ++i)
                if (cand[i].amplitude < floorAmp) {
                    examine = i;
                    break;
                }
            const std::size_t cap = static_cast<std::size_t>(d.maxBeads) * 4 + 16;
            const std::size_t belowFloor = cand.size() - examine;
            if (examine > cap) {
                extraNotExamined = static_cast<int>(examine - cap);
                res.notes.emplace_back(std::to_string(examine - cap) + " of the " + std::to_string(examine) +
                                       " candidates above the amplitude floor were not examined (the cap is "
                                       "4 x maxBeads + 16)");
                examine = cap;
            }
            const int roiXY = std::max(3, static_cast<int>(std::round(d.roiLateralUm / o.dxy)));
            const int roiZ = d.roiAxialUm > 0.0 ? std::max(3, static_cast<int>(std::round(d.roiAxialUm / res.dz)))
                                                : nz;
            const int marginXY = static_cast<int>(std::round(d.boundaryMarginLateralUm / o.dxy));
            const int marginZ = static_cast<int>(std::round(d.boundaryMarginAxialUm / res.dz));
            std::vector<BeadFit> keptList;
            for (std::size_t ci = 0; ci < examine; ++ci) {
                BeadFit f = cand[ci];
                const int cx = static_cast<int>(std::lround(f.x)), cy = static_cast<int>(std::lround(f.y)),
                          cz = static_cast<int>(std::lround(f.z));
                // The boundary test comes first because it is a fact about the
                // geometry and does not depend on anything the fit says, so a
                // bead at the edge is reported as being at the edge rather
                // than as whatever the apodized edge does to its amplitude.
                if (cx - roiXY / 2 - marginXY < 0 || cx + roiXY / 2 + marginXY >= nx ||
                    cy - roiXY / 2 - marginXY < 0 || cy + roiXY / 2 + marginXY >= ny ||
                    (roiZ < nz && (cz - roiZ / 2 - marginZ < 0 || cz + roiZ / 2 + marginZ >= nz)))
                    f.rejection = BeadRejection::NearBoundary;
                else if (d.saturationLevel > 0.0 &&
                         widefield[(static_cast<std::size_t>(cz) * ny + cy) * nx + cx] >= d.saturationLevel)
                    f.rejection = BeadRejection::Saturated;
                else if (f.amplitude < floorAmp)
                    f.rejection = BeadRejection::Amplitude;
                if (f.rejection == BeadRejection::None)
                    for (const BeadFit& k : keptList) {
                        const double dxp = k.x - f.x, dyp = k.y - f.y, dzp = k.z - f.z;
                        if (std::hypot(dxp, dyp) < d.minSeparationLateralUm / o.dxy &&
                            (sepZ == 0 || std::abs(dzp) < d.minSeparationAxialUm / res.dz)) {
                            f.rejection = BeadRejection::TooClose;
                            break;
                        }
                    }
                if (f.rejection == BeadRejection::None) {
                    Roi r;
                    r.x0 = std::max(0, cx - roiXY / 2);
                    r.x1 = std::min(nx, cx + roiXY / 2 + 1);
                    r.y0 = std::max(0, cy - roiXY / 2);
                    r.y1 = std::min(ny, cy + roiXY / 2 + 1);
                    r.z0 = roiZ >= nz ? 0 : std::max(0, cz - roiZ / 2);
                    r.z1 = roiZ >= nz ? nz : std::min(nz, cz + roiZ / 2 + 1);
                    if (!fitGaussian3d(widefield, ny, nx, r, f)) f.rejection = BeadRejection::FitFailed;
                    else {
                        const double smin = std::min(f.sigmaX, f.sigmaY) * o.dxy;
                        const double smax = std::max(f.sigmaX, f.sigmaY) * o.dxy;
                        const double sz = f.sigmaZ * res.dz;
                        if (d.sigmaMinLateralUm > 0.0 && smin < d.sigmaMinLateralUm)
                            f.rejection = BeadRejection::WidthTooSmall;
                        else if (d.sigmaMaxLateralUm > 0.0 && smax > d.sigmaMaxLateralUm)
                            f.rejection = BeadRejection::WidthTooLarge;
                        else if (d.sigmaMinAxialUm > 0.0 && sz < d.sigmaMinAxialUm)
                            f.rejection = BeadRejection::WidthTooSmall;
                        else if (d.sigmaMaxAxialUm > 0.0 && sz > d.sigmaMaxAxialUm)
                            f.rejection = BeadRejection::WidthTooLarge;
                        else if (d.maxResidual > 0.0 && f.residual > d.maxResidual)
                            f.rejection = BeadRejection::FitFailed;
                    }
                }
                // A bead the maxBeads cap leaves out passed every filter, so
                // it is not None: it carried None before, and the inventory
                // then printed "kept" (beadRejectionName(None)) beside a bead
                // whose `kept` was false. With its own reason, `None` means
                // kept and nothing else.
                if (f.rejection == BeadRejection::None && static_cast<int>(keptList.size()) >= d.maxBeads)
                    f.rejection = BeadRejection::OverMaxBeads;
                f.kept = f.rejection == BeadRejection::None;
                if (f.kept) keptList.push_back(f);
                beads.push_back(f);
            }
            if (belowFloor > 0)
                res.notes.emplace_back(std::to_string(belowFloor) + " of the " + std::to_string(cand.size()) +
                                       " candidates were below the amplitude floor of " + num(floorAmp, 4));
            extraBelowFloor = static_cast<int>(belowFloor);

            // The mask's box reaches half way to the nearest bright object,
            // and that has to mean every candidate above the amplitude floor,
            // not just the kept ones: a bead rejected as saturated or as an
            // aggregate still has to be masked out, or its interference lands
            // in another bead's measurement. The exception is a TooClose
            // candidate, which is a secondary maximum of a bead already in the
            // list (a lattice side lobe) -- cutting the box at one of those
            // would cut the PSF itself.
            for (BeadFit& f : beads) {
                double nearest = std::numeric_limits<double>::infinity();
                for (const BeadFit& g : beads) {
                    if (&g == &f || g.rejection == BeadRejection::TooClose) continue;
                    const double dd = std::hypot(g.x - f.x, g.y - f.y);
                    if (dd > 1e-9) nearest = std::min(nearest, dd);
                }
                f.nearestNeighbourPx = std::isfinite(nearest) ? nearest : 0.0;
            }
        }
        for (const BeadFit& f : beads) {
            ++res.rejected[static_cast<std::size_t>(f.rejection)];
            if (f.kept) ++res.kept;
        }
        res.rejected[static_cast<std::size_t>(BeadRejection::Amplitude)] += extraBelowFloor;
        res.rejected[static_cast<std::size_t>(BeadRejection::NotExamined)] += extraNotExamined;
        // The inventory's arithmetic, which is now an identity rather than
        // something a reader has to reconstruct: every candidate the detector
        // found is in exactly one bucket, and bucket None is exactly the kept
        // ones.
        if (const int capped = res.rejected[static_cast<std::size_t>(BeadRejection::OverMaxBeads)]; capped > 0)
            res.notes.emplace_back(std::to_string(capped) +
                                   " beads passed every filter and were left out by the maxBeads cap of " +
                                   std::to_string(o.detect.maxBeads));
        if (res.kept == 0)
            throw std::runtime_error("measureOTF: no bead survived detection -- " +
                                     std::to_string(res.found) + " candidates found, none kept");
        res.beads = beads;
        {
            // the brightest bead's peak over the spread of the background,
            // taken from the darkest decile of the widefield
            const std::size_t ndark = std::max<std::size_t>(1, nvox / 10);
            std::vector<double> sorted(widefield);
            std::nth_element(sorted.begin(), sorted.begin() + static_cast<long>(ndark), sorted.end());
            const auto mid = sorted.begin() + static_cast<long>(ndark);
            const double mean = std::accumulate(sorted.begin(), mid, 0.0) / static_cast<double>(ndark);
            double var = 0.0;
            for (auto it = sorted.begin(); it != mid; ++it) var += (*it - mean) * (*it - mean);
            const double sd = std::sqrt(var / static_cast<double>(ndark));
            double peak = 0.0;
            for (const BeadFit& f : beads)
                if (f.kept) peak = std::max(peak, f.amplitude);
            // the single-bead path reports the raw maximum, which still has
            // the dark level in it; the field path reports a fitted amplitude,
            // which does not. Taking the dark level off here makes the two
            // comparable.
            if (!o.field) peak -= mean;
            res.beadPeakSnr = sd > 0.0 ? peak / sd : 0.0;
        }

        // --- 5-9: the transform, the ramp, the bead size, the radial average ---
        const RadialPlan plan = radialPlan(ny, nx, o.dxy);
        res.nkr = plan.nkr;
        res.nzotf = nz;
        res.dkr = plan.dkr;
        res.dkz = nz > 1 ? 1.0 / (nz * res.dz) : 1.0;

        RealFFT fft({nz, ny, nx}, nbands, PlanRigor::Estimate, Device::cpu());
        std::vector<Cplx> spec(static_cast<std::size_t>(nbands) * nz * ny * half);
        std::vector<double> masked;

        // the bead-size division's own grid: makeotf's dr, which need not be
        // the acquisition's (otf_measure.hpp says why)
        const double compDxy = o.beadCompensationPixelUm > 0.0 ? o.beadCompensationPixelUm : o.dxy;
        const double compDz = o.beadCompensationAxialUm > 0.0 ? o.beadCompensationAxialUm : res.dz;
        const double dkx = 1.0 / (nx * compDxy), dky = 1.0 / (ny * compDxy);
        const double dkzv = nz > 1 ? 1.0 / (nz * compDz) : 0.0;
        const double radius = 0.5 * o.beadDiameterUm;
        // The side bands' k0 offset into the division is the illumination LINE
        // SPACING -- makeotf's -ls, SIMParameters::linespacing_um. Nothing in
        // a bead stack states it, so it is the caller's to state; 0 means it
        // did not, and then this falls back to makeotf's own default and says
        // so. Getting it wrong is invisible in the table: on the iSOAR2
        // stacks (0.504 um) makeotf's 0.2 um divides order 1 by the sphere's
        // transform at 5.0 instead of 1.98 1/um, which scales order 1 by
        // about 1.38 and leaves its shape alone.
        const bool periodStated = o.patternPeriodUm > 0.0;
        const double periodUm = periodStated ? o.patternPeriodUm : kMakeotfLineSpacingUm;
        const double k0mag = 1.0 / periodUm;
        if (o.beadDiameterUm > 0.0 && norders > 1) {
            res.patternPeriodUsedUm = periodUm;
            res.patternPeriodStated = periodStated;
            if (periodStated)
                res.notes.emplace_back("the finite-bead-size division offset order n by n/" +
                                       std::to_string(norders - 1) + " x " + num(k0mag) +
                                       " 1/um, from the stated line spacing of " + num(periodUm) + " um");
            else
                res.notes.emplace_back(
                    "THE ILLUMINATION LINE SPACING WAS NOT STATED, so the finite-bead-size division used "
                    "makeotf's default of " +
                    num(kMakeotfLineSpacingUm) + " um (order 1 offset by " + num(k0mag) +
                    " 1/um). On an instrument whose spacing is not that, every side band is off by a "
                    "near-uniform factor and the table's shape does not show it: state it with "
                    "OtfMeasureOptions::setIllumination, from the acquisition's own SIM parameters");
        }

        BandSet acc;
        acc.nbands = nbands;
        acc.nkr = plan.nkr;
        acc.nz = nz;
        acc.table.assign(static_cast<std::size_t>(nbands) * plan.nkr * nz, Cplx(0.0, 0.0));
        BandSet one = acc;
        std::vector<double> depths;
        double dcSum = 0.0, lineSum = 0.0, divisorSum = 0.0;
        int used = 0;
        // A single bead is one table, so the two paths coincide for it.
        const bool perBeadScale = !o.field || o.perBeadNormalise;

        for (const BeadFit& f : beads) {
            if (!f.kept) continue;
            const double* source = bands.data();
            if (o.field) {
                // Mask this bead out of the separated bands, on the FULL
                // lateral grid so dkr is the field's. The box is half the way
                // to the nearest kept bead (the whole field when there is only
                // one, which is what makes the single-bead path a special case
                // of this one), with a cosine taper on any side that is an
                // interior cut rather than the volume's own edge.
                masked.assign(static_cast<std::size_t>(nbands) * nvox, 0.0);
                const double halfBox = f.nearestNeighbourPx > 0.0 ? 0.5 * f.nearestNeighbourPx
                                                                  : static_cast<double>(std::max(nx, ny));
                const int hx = std::min(static_cast<int>(std::floor(halfBox)), nx);
                const int hy = std::min(static_cast<int>(std::floor(halfBox)), ny);
                const int x0 = std::max(0, static_cast<int>(std::lround(f.x)) - hx);
                const int x1 = std::min(nx, static_cast<int>(std::lround(f.x)) + hx + 1);
                const int y0 = std::max(0, static_cast<int>(std::lround(f.y)) - hy);
                const int y1 = std::min(ny, static_cast<int>(std::lround(f.y)) + hy + 1);
                const int taper = std::max(0, std::min({(x1 - x0) / 4, (y1 - y0) / 4,
                                                        static_cast<int>(std::lround(0.3 / o.dxy))}));
                std::vector<double> wx(static_cast<std::size_t>(x1 - x0), 1.0);
                std::vector<double> wy(static_cast<std::size_t>(y1 - y0), 1.0);
                for (int i = 0; i < taper; ++i) {
                    const double w = 0.5 * (1.0 - std::cos(kPi * (i + 0.5) / taper));
                    if (x0 > 0) wx[static_cast<std::size_t>(i)] = w;
                    if (x1 < nx) wx[static_cast<std::size_t>(x1 - x0 - 1 - i)] = w;
                    if (y0 > 0) wy[static_cast<std::size_t>(i)] = w;
                    if (y1 < ny) wy[static_cast<std::size_t>(y1 - y0 - 1 - i)] = w;
                }
                for (int b = 0; b < nbands; ++b)
                    for (int z = 0; z < nz; ++z)
                        for (int y = y0; y < y1; ++y) {
                            const std::size_t row = (static_cast<std::size_t>(b) * nz + z) * nsec +
                                                    static_cast<std::size_t>(y) * nx;
                            for (int x = x0; x < x1; ++x)
                                masked[row + static_cast<std::size_t>(x)] =
                                    bands[row + static_cast<std::size_t>(x)] *
                                    wy[static_cast<std::size_t>(y - y0)] * wx[static_cast<std::size_t>(x - x0)];
                        }
                source = masked.data();
            }
            fft.rfft(source, spec.data());

            // shift_center: the ramp that puts THIS bead at the origin.
            for (int b = 0; b < nbands; ++b)
                for (int iz = 0; iz < nz; ++iz) {
                    const int kz = iz > nz / 2 ? iz - nz : iz;
                    const double p1 = 2.0 * kPi * f.z * kz / nz;
                    for (int iy = 0; iy < ny; ++iy) {
                        const int ky = iy > ny / 2 ? iy - ny : iy;
                        const double p2 = 2.0 * kPi * f.y * ky / ny;
                        Cplx* row = spec.data() + ((static_cast<std::size_t>(b) * nz + iz) * ny + iy) * half;
                        for (int ix = 0; ix < half; ++ix) {
                            const double phi = p1 + p2 + 2.0 * kPi * f.x * ix / nx;
                            row[ix] *= Cplx(std::cos(phi), std::sin(phi));
                        }
                    }
                }

            // beadsize_compensate: divide out the transform of a sphere, with
            // each order's own k0 offset.
            if (o.beadDiameterUm > 0.0) {
                for (int order = 0; order < norders; ++order) {
                    const double frac = norders > 1 ? static_cast<double>(order) / (norders - 1) : 0.0;
                    const double k0x = frac * k0mag * std::cos(o.patternAngleRad);
                    const double k0y = frac * k0mag * std::sin(o.patternAngleRad);
                    const int b0 = order == 0 ? 0 : 2 * order - 1;
                    const int b1 = order == 0 ? 0 : 2 * order;
                    for (int iz = 0; iz < nz; ++iz) {
                        const double kz = (iz > nz / 2 ? iz - nz : iz) * dkzv;
                        for (int iy = 0; iy < ny; ++iy) {
                            const double ky = (iy > ny / 2 ? iy - ny : iy) * dky + k0y;
                            for (int ix = 0; ix < half; ++ix) {
                                const double kx = ix * dkx + k0x;
                                const std::size_t off = ((static_cast<std::size_t>(iz) * ny) + iy) * half + ix;
                                const double rho = std::sqrt(kx * kx + ky * ky + kz * kz);
                                const double ratio = (order == 0 && off == 0) ? 1.0 : sphereRatio(rho, radius);
                                if (ratio == 0.0) continue;
                                spec[static_cast<std::size_t>(b0) * nz * ny * half + off] /= ratio;
                                if (b1 != b0)
                                    spec[static_cast<std::size_t>(b1) * nz * ny * half + off] /= ratio;
                            }
                        }
                    }
                }
            }

            for (int b = 0; b < nbands; ++b)
                radialAverage(spec.data() + static_cast<std::size_t>(b) * nz * ny * half, nz, ny, nx, plan,
                              one.band(b));

            // 9: modify(), per band
            if (o.repairKr0Column)
                for (int b = 0; b < nbands; ++b) repairKr0(one.band(b), plan.nkr, nz, (b + 1) / 2);

            // 10 and 11: the scale and then combine_reim, in radialft.cpp's
            // own order (rescale, then combine_reim), with one divisor for
            // every band so no choice here can move a modulation depth.
            //
            // Where it goes is a per-bead decision: a bead's table has to be
            // on its own scale before it is averaged with another's, or the
            // brightest bead decides the answer. With perBeadNormalise off the
            // raw tables are summed and the divisor is taken once at the end,
            // which is the brightness-weighted average and the behaviour
            // makeotf would have on a field it could handle.
            const double dc = one.band(0)[0].real();
            const double lineval = lineFitOrigin(one.band(0), plan.nkr, nz, o.lineFitFirst, o.lineFitLast,
                                                 perBeadScale && o.scale == OtfMeasureScale::MakeotfFixOrigin);
            if (perBeadScale) {
                const double divisor = scaleDivisorOf(one.band(0)[0].real(), dc, o.scale);
                if (divisor != 1.0)
                    for (auto& v : one.table) v /= divisor;
                divisorSum += divisor;
            }
            if (o.combineReIm && norders > 1)
                for (int order = 1; order < norders; ++order)
                    combineReIm(one.band(2 * order - 1), one.band(2 * order), plan.nkr, nz, nx);

            for (std::size_t i = 0; i < acc.table.size(); ++i) acc.table[i] += one.table[i];
            dcSum += dc;
            lineSum += lineval;
            if (norders > 1 && one.band(0)[0].real() != 0.0)
                depths.push_back(one.band(1)[0].real() / one.band(0)[0].real());
            ++used;
        }

        for (auto& v : acc.table) v /= static_cast<double>(used);
        res.order0Dc = dcSum / used;
        res.lineFitToOrigin = lineSum / used;
        res.dcFractionOfSignal = total != 0.0 ? res.order0Dc / total : 0.0;
        res.scaleSensitivityPerAdu =
            res.order0Dc != 0.0
                ? (static_cast<double>(nsec) * sections / nphases) / std::abs(res.order0Dc)
                : 0.0;
        if (perBeadScale) {
            res.scaleDivisor = divisorSum / used;
        } else {
            // the divisor of the averaged table, taken once
            const double dcAvg = acc.band(0)[0].real();
            lineFitOrigin(acc.band(0), plan.nkr, nz, o.lineFitFirst, o.lineFitLast,
                          o.scale == OtfMeasureScale::MakeotfFixOrigin);
            const double divisor = scaleDivisorOf(acc.band(0)[0].real(), dcAvg, o.scale);
            if (divisor != 1.0)
                for (auto& v : acc.table) v /= divisor;
            res.scaleDivisor = divisor;
            res.notes.emplace_back("the per-bead tables were summed unnormalised, so the average is weighted by bead brightness");
        }

        // --- the table, and what it says ---------------------------------------
        Eigen::Tensor<Cplx, 3, Eigen::RowMajor> table(norders, plan.nkr, nz);
        for (int order = 0; order < norders; ++order) {
            const Cplx* src = acc.band(order == 0 ? 0 : 2 * order - 1);
            for (int ir = 0; ir < plan.nkr; ++ir)
                for (int k = 0; k < nz; ++k) table(order, ir, k) = src[static_cast<std::size_t>(ir) * nz + k];
        }
        res.orderDc.resize(static_cast<std::size_t>(norders));
        for (int order = 0; order < norders; ++order) res.orderDc[static_cast<std::size_t>(order)] = table(order, 0, 0).real();
        res.modulationDepth = norders > 1 && res.orderDc[0] != 0.0 ? res.orderDc[1] / res.orderDc[0] : 0.0;
        if (norders > 1) {
            // the scale-free depth: |order 1| / |order 0| wherever order 0 is
            // signal, kr = 0 left out because step 9 replaced it
            double peak0 = 0.0;
            for (int ir = 1; ir < plan.nkr; ++ir)
                for (int k = 0; k < nz; ++k) peak0 = std::max(peak0, std::abs(table(0, ir, k)));
            const double floor0 = o.bandRatioMinOrder0 * peak0;
            std::vector<double> ratios;
            for (int ir = 1; ir < plan.nkr; ++ir)
                for (int k = 0; k < nz; ++k) {
                    const double m0 = std::abs(table(0, ir, k));
                    if (m0 > floor0) ratios.push_back(std::abs(table(1, ir, k)) / m0);
                }
            res.bandRatioSamples = static_cast<int>(ratios.size());
            if (!ratios.empty()) {
                std::sort(ratios.begin(), ratios.end());
                const auto q = [&](double f) {
                    const std::size_t i = std::min(ratios.size() - 1,
                                                   static_cast<std::size_t>(f * (ratios.size() - 1)));
                    return ratios[i];
                };
                res.bandRatio = q(0.5);
                res.bandRatioIqr = q(0.75) - q(0.25);
            }
        }
        if (depths.size() > 1) {
            const double m = std::accumulate(depths.begin(), depths.end(), 0.0) / static_cast<double>(depths.size());
            double v = 0.0;
            for (double d : depths) v += (d - m) * (d - m);
            res.modulationDepthSpread = std::sqrt(v / static_cast<double>(depths.size()));
        }
        {
            double mean = 0.0, err = 0.0;
            long long n = 0, c = 0;
            for (int order = 0; order < norders; ++order)
                for (int ir = 0; ir < plan.nkr; ++ir)
                    for (int k = 0; k < nz; ++k) {
                        mean += std::abs(table(order, ir, k));
                        ++n;
                    }
            mean = n > 0 ? mean / static_cast<double>(n) : 0.0;
            for (int order = 0; order < norders; ++order)
                for (int ir = 0; ir < plan.nkr; ++ir)
                    for (int k = 1; k <= nz / 2; ++k) {
                        err += std::abs(table(order, ir, k) - std::conj(table(order, ir, nz - k)));
                        ++c;
                    }
            res.hermitianKzError = (c > 0 && mean > 0.0) ? (err / static_cast<double>(c)) / mean : 0.0;
        }
        if (res.scaleSensitivityPerAdu > 0.02)
            res.notes.emplace_back(
                "the scale is soft: 1 ADU of background error moves every sample but the DC by " +
                num(100.0 * res.scaleSensitivityPerAdu, 3) + "%, and the two background estimators differ by " +
                num(std::abs(res.backgroundBorderMean - res.backgroundDarkest), 3) +
                " ADU (border mean " + num(res.backgroundBorderMean, 6) + ", darkest " +
                num(res.backgroundDarkest, 6) + "). The band ratio is the number that does not move.");
        res.otf = OTFRadiallyAveraged(std::move(table), res.dkr, res.dkz, res.scaleDivisor);
        return res;
    }

    // --- writing ---------------------------------------------------------------

    namespace {
        std::vector<std::string> writeTable(const std::string& path, const OTFRadiallyAveraged& otf,
                                            const OtfWriteOptions& opts, const std::string& provenance) {
            const auto& d = otf.data();
            const int norders = static_cast<int>(d.dimension(0));
            const int nkr = static_cast<int>(d.dimension(1));
            const int nzotf = static_cast<int>(d.dimension(2));
            if (norders < 1 || nkr < 1 || nzotf < 1) throw IoError(path + ": refusing to write an empty OTF table");
            ImageStack<float> pages(norders, nkr, 2 * nzotf);
            for (int o = 0; o < norders; ++o)
                for (int r = 0; r < nkr; ++r)
                    for (int k = 0; k < nzotf; ++k) {
                        pages(o, r, 2 * k) = static_cast<float>(d(o, r, k).real());
                        pages(o, r, 2 * k + 1) = static_cast<float>(d(o, r, k).imag());
                    }
            writeTiffStack<float>(path, pages);
            std::vector<std::string> written{path};
            if (opts.sidecar) {
                const std::string side = path + ".toml";
                std::ofstream f(side);
                if (!f) throw IoError(side + ": cannot write the OTF sidecar");
                f << "# " << path << "\n";
                if (!opts.note.empty()) f << "# " << opts.note << "\n";
                if (!provenance.empty()) f << "# " << provenance << "\n";
                f << "# makeotf's TIFF layout carries no sampling at all, so these are the\n"
                     "# numbers loadOTF would otherwise have to derive from whatever pixel size\n"
                     "# the run it is loaded into happens to use (findings 9k.48).\n"
                     "[sampling]\n"
                  << std::setprecision(17) << "dkr = " << otf.dkrotf() << "\n"
                  << "dkz = " << otf.dkzotf() << "\n"
                  << "kz_origin = \"dc_first\"\n";
                if (!f) throw IoError(side + ": writing the OTF sidecar failed");
                written.push_back(side);
            }
            return written;
        }
    } // namespace

    std::vector<std::string> writeRadialOTF(const std::string& path, const OTFRadiallyAveraged& otf,
                                            const OtfWriteOptions& options) {
        return writeTable(path, otf, options, std::string());
    }

    std::vector<std::string> writeMeasuredOTF(const std::string& path, const OtfMeasureResult& result,
                                              const OtfWriteOptions& options) {
        return writeTable(path, result.otf, options, "sirius measureOTF: " + result.summary());
    }

} // namespace sirius
