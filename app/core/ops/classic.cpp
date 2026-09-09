// Classical segmentation: the conventional recipe for nuclei and blobs
// without a model -- flatten the background (white top-hat), smooth
// (Gaussian), cut with a global (Otsu, percentile, manual) or local-mean
// threshold, clean the mask (binary opening, hole filling) and split it into
// instances by connected components or a distance watershed. Everything is
// per plane in (y, x) except the instance step, which is 3D.
#include "core/ops/builtin.hpp"

#include <algorithm>
#include <utility>
#include <functional>
#include <limits>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <vector>

#include <sirius/constants.hpp>
#include <sirius/image_ops.hpp>

namespace sirius::app {

    namespace {

        inline Index reflect(Index i, Index n) {
            if (n <= 1) return 0;
            while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - i - 2;
            return i;
        }

        // Separable Gaussian on one (y, x) plane, borders reflected.
        void gaussianPlane(float* plane, Index y, Index x, double sigma, std::vector<float>& tmp) {
            if (sigma <= 0.0) return;
            const Index r = std::max<Index>(1, static_cast<Index>(std::ceil(3.0 * sigma)));
            std::vector<float> k(static_cast<std::size_t>(2 * r + 1));
            double sum = 0.0;
            for (Index i = -r; i <= r; ++i) {
                k[static_cast<std::size_t>(i + r)] = static_cast<float>(std::exp(-0.5 * (i * i) / (sigma * sigma)));
                sum += k[static_cast<std::size_t>(i + r)];
            }
            for (float& v : k) v = static_cast<float>(v / sum);
            tmp.resize(static_cast<std::size_t>(y * x));
            for (Index yy = 0; yy < y; ++yy) {
                const float* src = plane + yy * x;
                float* dst = tmp.data() + yy * x;
                for (Index xx = 0; xx < x; ++xx) {
                    float acc = 0.0f;
                    for (Index i = -r; i <= r; ++i) acc += src[reflect(xx + i, x)] * k[static_cast<std::size_t>(i + r)];
                    dst[xx] = acc;
                }
            }
            for (Index yy = 0; yy < y; ++yy) {
                float* dst = plane + yy * x;
                for (Index xx = 0; xx < x; ++xx) {
                    float acc = 0.0f;
                    for (Index i = -r; i <= r; ++i) acc += tmp[static_cast<std::size_t>(reflect(yy + i, y) * x + xx)] * k[static_cast<std::size_t>(i + r)];
                    dst[xx] = acc;
                }
            }
        }

        // Sliding-window min or max over one line (stride between elements),
        // window 2r+1, borders clamped: monotonic deque, O(n).
        template <typename T, bool Max>
        void slideLine(const T* src, T* dst, Index n, Index stride, Index r) {
            std::deque<Index> q;
            auto better = [](T a, T b) { return Max ? a >= b : a <= b; };
            Index next = 0;
            for (Index i = 0; i < n; ++i) {
                const Index hi = std::min(n - 1, i + r);
                for (; next <= hi; ++next) {
                    while (!q.empty() && better(src[next * stride], src[q.back() * stride])) q.pop_back();
                    q.push_back(next);
                }
                while (!q.empty() && q.front() < i - r) q.pop_front();
                dst[i * stride] = src[q.front() * stride];
            }
        }

        template <typename T, bool Max>
        void boxFilterPlane(T* plane, Index y, Index x, Index r, std::vector<T>& tmp) {
            if (r <= 0) return;
            tmp.resize(static_cast<std::size_t>(y * x));
            for (Index yy = 0; yy < y; ++yy) slideLine<T, Max>(plane + yy * x, tmp.data() + yy * x, x, 1, r);
            for (Index xx = 0; xx < x; ++xx) slideLine<T, Max>(tmp.data() + xx, plane + xx, y, x, r);
        }

        // White top-hat: image minus its opening (erode, then dilate) with a
        // (2r+1)² box: what is smaller than the box survives, the rest is background.
        void topHatPlane(float* plane, Index y, Index x, Index r, std::vector<float>& work, std::vector<float>& tmp) {
            if (r <= 0) return;
            work.assign(plane, plane + y * x);
            boxFilterPlane<float, false>(work.data(), y, x, r, tmp);
            boxFilterPlane<float, true>(work.data(), y, x, r, tmp);
            for (Index i = 0; i < y * x; ++i) plane[i] = std::max(0.0f, plane[i] - work[static_cast<std::size_t>(i)]);
        }

        // Mean over a (2r+1)² window through an integral image, borders clamped.
        void localMeanPlane(const float* plane, Index y, Index x, Index r, float* out, std::vector<double>& integral) {
            const Index W = x + 1;
            integral.assign(static_cast<std::size_t>((y + 1) * W), 0.0);
            for (Index yy = 0; yy < y; ++yy) {
                double row = 0.0;
                for (Index xx = 0; xx < x; ++xx) {
                    row += plane[yy * x + xx];
                    integral[static_cast<std::size_t>((yy + 1) * W + xx + 1)] = integral[static_cast<std::size_t>(yy * W + xx + 1)] + row;
                }
            }
            for (Index yy = 0; yy < y; ++yy) {
                const Index y0 = std::max<Index>(0, yy - r), y1 = std::min(y, yy + r + 1);
                for (Index xx = 0; xx < x; ++xx) {
                    const Index x0 = std::max<Index>(0, xx - r), x1 = std::min(x, xx + r + 1);
                    const double s = integral[static_cast<std::size_t>(y1 * W + x1)] - integral[static_cast<std::size_t>(y0 * W + x1)] -
                                     integral[static_cast<std::size_t>(y1 * W + x0)] + integral[static_cast<std::size_t>(y0 * W + x0)];
                    out[yy * x + xx] = static_cast<float>(s / static_cast<double>((y1 - y0) * (x1 - x0)));
                }
            }
        }

        // Background connected to the plane border stays background; every
        // other 0 (enclosed) becomes foreground.
        void fillHolesPlane(std::uint8_t* mask, Index y, Index x, std::vector<std::uint8_t>& seen, std::vector<Index>& stack) {
            seen.assign(static_cast<std::size_t>(y * x), 0);
            stack.clear();
            auto push = [&](Index i) {
                if (!seen[static_cast<std::size_t>(i)] && mask[i] == 0) {
                    seen[static_cast<std::size_t>(i)] = 1;
                    stack.push_back(i);
                }
            };
            for (Index xx = 0; xx < x; ++xx) {
                push(xx);
                push((y - 1) * x + xx);
            }
            for (Index yy = 0; yy < y; ++yy) {
                push(yy * x);
                push(yy * x + x - 1);
            }
            while (!stack.empty()) {
                const Index i = stack.back();
                stack.pop_back();
                const Index yy = i / x, xx = i % x;
                if (xx > 0) push(i - 1);
                if (xx + 1 < x) push(i + 1);
                if (yy > 0) push(i - x);
                if (yy + 1 < y) push(i + x);
            }
            for (Index i = 0; i < y * x; ++i)
                if (mask[i] == 0 && !seen[static_cast<std::size_t>(i)]) mask[i] = 1;
        }

        // Local mean and standard deviation over a (2r+1)^2 window, both from
        // integral images of the value and its square: O(1) per pixel.
        void localStatsPlane(const float* plane, Index y, Index x, Index r, float* mean, float* stddev,
                             std::vector<double>& sum, std::vector<double>& sumSq) {
            const Index W = x + 1;
            sum.assign(static_cast<std::size_t>((y + 1) * W), 0.0);
            sumSq.assign(static_cast<std::size_t>((y + 1) * W), 0.0);
            for (Index yy = 0; yy < y; ++yy) {
                double row = 0.0, rowSq = 0.0;
                for (Index xx = 0; xx < x; ++xx) {
                    const double v = plane[yy * x + xx];
                    row += v;
                    rowSq += v * v;
                    sum[static_cast<std::size_t>((yy + 1) * W + xx + 1)] = sum[static_cast<std::size_t>(yy * W + xx + 1)] + row;
                    sumSq[static_cast<std::size_t>((yy + 1) * W + xx + 1)] = sumSq[static_cast<std::size_t>(yy * W + xx + 1)] + rowSq;
                }
            }
            auto box = [&](const std::vector<double>& in, Index y0, Index y1, Index x0, Index x1) {
                return in[static_cast<std::size_t>(y1 * W + x1)] - in[static_cast<std::size_t>(y0 * W + x1)] -
                       in[static_cast<std::size_t>(y1 * W + x0)] + in[static_cast<std::size_t>(y0 * W + x0)];
            };
            for (Index yy = 0; yy < y; ++yy) {
                const Index y0 = std::max<Index>(0, yy - r), y1 = std::min(y, yy + r + 1);
                for (Index xx = 0; xx < x; ++xx) {
                    const Index x0 = std::max<Index>(0, xx - r), x1 = std::min(x, xx + r + 1);
                    const double count = static_cast<double>((y1 - y0) * (x1 - x0));
                    const double m = box(sum, y0, y1, x0, x1) / count;
                    const double var = std::max(0.0, box(sumSq, y0, y1, x0, x1) / count - m * m);
                    mean[yy * x + xx] = static_cast<float>(m);
                    stddev[yy * x + xx] = static_cast<float>(std::sqrt(var));
                }
            }
        }

        // Two Otsu thresholds (three classes) by exhaustive search over a
        // 128-bin histogram: the upper cut keeps only the brightest class,
        // which separates objects from a bright halo the single cut merges.
        std::pair<float, float> multiOtsuThresholds(const float* v, Index n) {
            float mn = std::numeric_limits<float>::infinity(), mx = -mn;
            for (Index i = 0; i < n; ++i) {
                if (std::isnan(v[i])) continue;
                mn = std::min(mn, v[i]);
                mx = std::max(mx, v[i]);
            }
            if (!(mx > mn)) return {mn, mn};
            constexpr int bins = 128;
            const std::vector<double> h = histogram(v, n, bins, mn, mx);
            std::array<double, bins + 1> w{}, m{};
            for (int i = 0; i < bins; ++i) {
                w[static_cast<std::size_t>(i + 1)] = w[static_cast<std::size_t>(i)] + h[static_cast<std::size_t>(i)];
                m[static_cast<std::size_t>(i + 1)] = m[static_cast<std::size_t>(i)] + i * h[static_cast<std::size_t>(i)];
            }
            const double total = w[bins], mean = m[bins];
            double best = -1.0;
            int bestA = bins / 3, bestB = 2 * bins / 3;
            if (!(total > 0.0)) return {mn, mx};
            for (int a = 1; a < bins - 1; ++a)
                for (int b = a + 1; b < bins; ++b) {
                    const double w0 = w[static_cast<std::size_t>(a)], w1 = w[static_cast<std::size_t>(b)] - w0, w2 = total - w[static_cast<std::size_t>(b)];
                    if (w0 <= 0.0 || w1 <= 0.0 || w2 <= 0.0) continue;
                    const double m0 = m[static_cast<std::size_t>(a)] / w0;
                    const double m1 = (m[static_cast<std::size_t>(b)] - m[static_cast<std::size_t>(a)]) / w1;
                    const double m2 = (mean - m[static_cast<std::size_t>(b)]) / w2;
                    const double between = w0 * (m0 - mean / total) * (m0 - mean / total) +
                                           w1 * (m1 - mean / total) * (m1 - mean / total) +
                                           w2 * (m2 - mean / total) * (m2 - mean / total);
                    if (between > best) {
                        best = between;
                        bestA = a;
                        bestB = b;
                    }
                }
            const float lo = mn + (mx - mn) * static_cast<float>(bestA + 1) / bins;
            const float hi = mn + (mx - mn) * static_cast<float>(bestB + 1) / bins;
            return {lo, hi};
        }

        // Difference of Gaussians: a band-pass that answers to blobs about
        // `sigma` across and flattens everything larger, so nuclei of one size
        // survive a textured background.
        void dogPlane(const float* src, float* dst, Index y, Index x, double sigma, double ratio, std::vector<float>& a,
                      std::vector<float>& b, std::vector<float>& tmp) {
            const Index n = y * x;
            a.assign(src, src + n);
            b.assign(src, src + n);
            gaussianPlane(a.data(), y, x, sigma, tmp);
            gaussianPlane(b.data(), y, x, sigma * std::max(1.1, ratio), tmp);
            for (Index i = 0; i < n; ++i) dst[i] = std::max(0.0f, a[static_cast<std::size_t>(i)] - b[static_cast<std::size_t>(i)]);
        }

        // Eigenvalues of a symmetric 3x3, smallest absolute first. The
        // analytic (trigonometric) solution: no iteration, no library.
        std::array<double, 3> symmetricEigenvalues(double a11, double a12, double a13, double a22, double a23, double a33) {
            const double p1 = a12 * a12 + a13 * a13 + a23 * a23;
            std::array<double, 3> e{};
            if (p1 <= 1e-30) {
                e = {a11, a22, a33};
            } else {
                const double q = (a11 + a22 + a33) / 3.0;
                const double p2 = (a11 - q) * (a11 - q) + (a22 - q) * (a22 - q) + (a33 - q) * (a33 - q) + 2.0 * p1;
                const double p = std::sqrt(std::max(1e-30, p2 / 6.0));
                const double b11 = (a11 - q) / p, b22 = (a22 - q) / p, b33 = (a33 - q) / p;
                const double b12 = a12 / p, b13 = a13 / p, b23 = a23 / p;
                const double det = b11 * (b22 * b33 - b23 * b23) - b12 * (b12 * b33 - b23 * b13) + b13 * (b12 * b23 - b22 * b13);
                const double r = std::clamp(det / 2.0, -1.0, 1.0);
                const double phi = std::acos(r) / 3.0;
                const double e1 = q + 2.0 * p * std::cos(phi);
                const double e3 = q + 2.0 * p * std::cos(phi + 2.0 * kPi / 3.0);
                e = {e1, 3.0 * q - e1 - e3, e3};
            }
            std::sort(e.begin(), e.end(), [](double a, double b) { return std::abs(a) < std::abs(b); });
            return e;
        }

        // Frangi vesselness in 3D: at each width the Hessian's eigenvalues say
        // whether the neighbourhood looks like a tube (one small eigenvalue
        // along the axis, two large negative ones across it), a sheet or a
        // blob. Filaments that run through z -- microtubules, actin, vessels --
        // are found whatever their direction, which a plane-by-plane filter
        // cannot do: it only sees the slice through them.
        void frangiVolume(const float* src, float* dst, Index z, Index y, Index x, double zAspect, double sigmaMin,
                          double sigmaMax, int scales, std::vector<float>& work, std::vector<float>& tmp,
                          const std::function<void()>& poll) {
            const Index plane = y * x, n = z * plane;
            std::fill(dst, dst + n, 0.0f);
            scales = std::max(1, scales);
            sigmaMin = std::max(0.3, sigmaMin);
            sigmaMax = std::max(sigmaMin, sigmaMax);
            zAspect = std::max(1e-6, zAspect);
            std::vector<float> vesselness(static_cast<std::size_t>(n));
            std::vector<double> sVals(static_cast<std::size_t>(n));
            for (int k = 0; k < scales; ++k) {
                const double sigma = scales == 1 ? sigmaMin
                                                 : sigmaMin * std::pow(sigmaMax / sigmaMin, static_cast<double>(k) / (scales - 1));
                work.assign(src, src + n);
                gaussianVolume(work, z, y, x, sigma, sigma, sigma / zAspect, tmp);
                const double norm = sigma * sigma;
                std::fill(vesselness.begin(), vesselness.end(), 0.0f);
                std::fill(sVals.begin(), sVals.end(), 0.0);
                double maxS = 0.0;
                for (Index iz = 0; iz < z; ++iz) {
                    // a whole volume at every scale: asked once per plane, so
                    // a cancel is noticed within a plane instead of within the
                    // whole filter, which is minutes on a full-size stack
                    if (poll) poll();
                    for (Index iy = 0; iy < y; ++iy)
                        for (Index ix = 0; ix < x; ++ix) {
                            const Index i = (iz * y + iy) * x + ix;
                            auto at = [&](Index jz, Index jy, Index jx) {
                                jz = std::clamp<Index>(jz, 0, z - 1);
                                jy = std::clamp<Index>(jy, 0, y - 1);
                                jx = std::clamp<Index>(jx, 0, x - 1);
                                return static_cast<double>(work[static_cast<std::size_t>((jz * y + jy) * x + jx)]);
                            };
                            const double c = at(iz, iy, ix);
                            const double dxx = norm * (at(iz, iy, ix + 1) + at(iz, iy, ix - 1) - 2.0 * c);
                            const double dyy = norm * (at(iz, iy + 1, ix) + at(iz, iy - 1, ix) - 2.0 * c);
                            const double dzz = z > 1 ? norm * (at(iz + 1, iy, ix) + at(iz - 1, iy, ix) - 2.0 * c) : 0.0;
                            const double dxy = norm * 0.25 * (at(iz, iy + 1, ix + 1) + at(iz, iy - 1, ix - 1) - at(iz, iy + 1, ix - 1) - at(iz, iy - 1, ix + 1));
                            const double dxz = z > 1 ? norm * 0.25 * (at(iz + 1, iy, ix + 1) + at(iz - 1, iy, ix - 1) - at(iz + 1, iy, ix - 1) - at(iz - 1, iy, ix + 1))
                                                     : 0.0;
                            const double dyz = z > 1 ? norm * 0.25 * (at(iz + 1, iy + 1, ix) + at(iz - 1, iy - 1, ix) - at(iz + 1, iy - 1, ix) - at(iz - 1, iy + 1, ix))
                                                     : 0.0;
                            double v, sMag;
                            if (z > 1) {
                                const std::array<double, 3> e = symmetricEigenvalues(dxx, dxy, dxz, dyy, dyz, dzz);
                                const double l1 = e[0], l2 = e[1], l3 = e[2];
                                if (l2 >= 0.0 || l3 >= 0.0) continue;   // a bright tube bends down across its axis
                                const double ra = std::abs(l2) / std::max(std::abs(l3), 1e-12);
                                const double rb = std::abs(l1) / std::max(std::sqrt(std::abs(l2 * l3)), 1e-12);
                                sMag = std::sqrt(l1 * l1 + l2 * l2 + l3 * l3);
                                v = (1.0 - std::exp(-ra * ra / 0.5)) * std::exp(-rb * rb / 0.5);
                            } else {
                                // One plane: the third eigenvalue is identically zero,
                                // which the 3D measure reads as "no tube" (its ra,
                                // |along| / |across|, is ~0 for a line). The 2D Frangi
                                // measure over the two in-plane eigenvalues instead,
                                // ordered by magnitude: a bright line bends down
                                // across its axis and is flat along it.
                                const double tr = dxx + dyy;
                                const double disc = std::sqrt((dxx - dyy) * (dxx - dyy) + 4.0 * dxy * dxy);
                                double l1 = 0.5 * (tr + disc), l2 = 0.5 * (tr - disc);
                                if (std::abs(l1) > std::abs(l2)) std::swap(l1, l2);
                                if (l2 >= 0.0) continue;
                                const double rb = std::abs(l1) / std::max(std::abs(l2), 1e-12);
                                sMag = std::sqrt(l1 * l1 + l2 * l2);
                                v = std::exp(-rb * rb / 0.5);
                            }
                            sVals[static_cast<std::size_t>(i)] = sMag;
                            maxS = std::max(maxS, sMag);
                            vesselness[static_cast<std::size_t>(i)] = static_cast<float>(v);
                        }
                }
                const double c2 = 2.0 * std::max(1e-12, 0.5 * maxS) * std::max(1e-12, 0.5 * maxS);
                for (Index i = 0; i < n; ++i) {
                    if (vesselness[static_cast<std::size_t>(i)] <= 0.0f) continue;
                    const double sMag = sVals[static_cast<std::size_t>(i)];
                    const float v = static_cast<float>(vesselness[static_cast<std::size_t>(i)] * (1.0 - std::exp(-sMag * sMag / c2)));
                    dst[i] = std::max(dst[i], v);
                }
            }
        }

        // Meijering neuriteness: the Hessian eigenvalues are mixed
        // (l_i = (1-a) e_i + a tr H, a = 1/4 in 3D) before the largest by
        // magnitude is taken, which is what makes it answer to a thin line
        // rather than to a tube of some width. On neurites and other very thin
        // filaments it holds together where Frangi, which needs two small
        // eigenvalues and one large, starts to break up.
        void meijeringVolume(const float* src, float* dst, Index z, Index y, Index x, double zAspect, double sigmaMin,
                             double sigmaMax, int scales, std::vector<float>& work, std::vector<float>& tmp,
                             const std::function<void()>& poll) {
            const Index plane = y * x, n = z * plane;
            std::fill(dst, dst + n, 0.0f);
            scales = std::max(1, scales);
            sigmaMin = std::max(0.3, sigmaMin);
            sigmaMax = std::max(sigmaMin, sigmaMax);
            zAspect = std::max(1e-6, zAspect);
            const double alpha = z > 1 ? 0.25 : 1.0 / 3.0;   // 1 / (ndim + 1)
            std::vector<float> response(static_cast<std::size_t>(n));
            for (int k = 0; k < scales; ++k) {
                const double sigma = scales == 1 ? sigmaMin
                                                 : sigmaMin * std::pow(sigmaMax / sigmaMin, static_cast<double>(k) / (scales - 1));
                // negated: the filter is written for dark ridges, and a
                // fluorescence filament is bright
                work.resize(static_cast<std::size_t>(n));
                for (Index i = 0; i < n; ++i) work[static_cast<std::size_t>(i)] = -src[i];
                gaussianVolume(work, z, y, x, sigma, sigma, sigma / zAspect, tmp);
                const double norm = sigma * sigma;
                double maxResponse = 0.0;
                for (Index iz = 0; iz < z; ++iz) {
                    if (poll) poll();
                    for (Index iy = 0; iy < y; ++iy)
                        for (Index ix = 0; ix < x; ++ix) {
                            const Index i = (iz * y + iy) * x + ix;
                            auto at = [&](Index jz, Index jy, Index jx) {
                                jz = std::clamp<Index>(jz, 0, z - 1);
                                jy = std::clamp<Index>(jy, 0, y - 1);
                                jx = std::clamp<Index>(jx, 0, x - 1);
                                return static_cast<double>(work[static_cast<std::size_t>((jz * y + jy) * x + jx)]);
                            };
                            const double c = at(iz, iy, ix);
                            const double dxx = norm * (at(iz, iy, ix + 1) + at(iz, iy, ix - 1) - 2.0 * c);
                            const double dyy = norm * (at(iz, iy + 1, ix) + at(iz, iy - 1, ix) - 2.0 * c);
                            const double dzz = z > 1 ? norm * (at(iz + 1, iy, ix) + at(iz - 1, iy, ix) - 2.0 * c) : 0.0;
                            const double dxy = norm * 0.25 * (at(iz, iy + 1, ix + 1) + at(iz, iy - 1, ix - 1) - at(iz, iy + 1, ix - 1) - at(iz, iy - 1, ix + 1));
                            const double dxz = z > 1 ? norm * 0.25 * (at(iz + 1, iy, ix + 1) + at(iz - 1, iy, ix - 1) - at(iz + 1, iy, ix - 1) - at(iz - 1, iy, ix + 1))
                                                     : 0.0;
                            const double dyz = z > 1 ? norm * 0.25 * (at(iz + 1, iy + 1, ix) + at(iz - 1, iy - 1, ix) - at(iz + 1, iy - 1, ix) - at(iz - 1, iy + 1, ix))
                                                     : 0.0;
                            const std::array<double, 3> e = symmetricEigenvalues(dxx, dxy, dxz, dyy, dyz, dzz);
                            const double trace = e[0] + e[1] + e[2];
                            double picked = 0.0, magnitude = -1.0;
                            for (int q = 0; q < 3; ++q) {
                                if (z == 1 && q == 0) continue;   // the third eigenvalue of a plane is not a measurement
                                const double mixed = (1.0 - alpha) * e[static_cast<std::size_t>(q)] + alpha * trace;
                                if (std::abs(mixed) > magnitude) {
                                    magnitude = std::abs(mixed);
                                    picked = mixed;
                                }
                            }
                            const double v = std::max(0.0, picked);
                            response[static_cast<std::size_t>(i)] = static_cast<float>(v);
                            maxResponse = std::max(maxResponse, v);
                        }
                }
                // each scale is normalised to its own maximum, as the method
                // has it, so one scale cannot drown the others
                if (maxResponse <= 0.0) continue;
                for (Index i = 0; i < n; ++i)
                    dst[i] = std::max(dst[i], static_cast<float>(response[static_cast<std::size_t>(i)] / maxResponse));
            }
        }

        class ClassicalSegmentationOperation final : public Operation {
        public:
            ClassicalSegmentationOperation() {
                info_.kind = "classic";
                info_.name = "Classical segmentation";
                info_.group = "Segment";
                info_.kindLabel = "SEGMENT";
                info_.diagnostics = DiagnosticsKind::Segment;
                info_.defaultCache = CachePolicy::Memory;
                info_.separableOverT = true;
                info_.producesLabels = true;
                info_.helpPage = "classic";
                info_.params = {
                    channelParam("channel", "Channel", 0),
                    choiceParam("denoise", "Denoise", {"None", "Median 3x3", "Anisotropic diffusion"}, "None")
                        .withHelp("Before anything else: a 3x3 median drops shot noise without moving an edge; Perona-Malik "
                                  "diffusion smooths the inside of a region and leaves its boundary alone"),
                    intParam("diffusion_iterations", "Diffusion steps", 5).range(1, 200).withHelp("Anisotropic diffusion: more steps, smoother interiors").asAdvanced().visibleWhen("denoise", {"Anisotropic diffusion"}),
                    doubleParam("diffusion_k", "Diffusion edge", 0.1).range(0.001, 1.0, 0.01, 3).withHelp("Anisotropic diffusion: a gradient this large, as a fraction of the intensity range, counts as an edge and is kept").asAdvanced().visibleWhen("denoise", {"Anisotropic diffusion"}),
                    choiceParam("enhance", "Enhance", {"None", "Blobs (DoG)", "Tubes (Frangi)", "Neurites (Meijering)"}, "None")
                        .withHelp("What to bring out before the threshold: round objects of one size, tubes of a range of "
                                  "widths, or the thinnest lines of all"),
                    doubleParam("enhance_sigma", "Feature σ", 2.0).range(0.3, 100.0, 0.5, 1).withUnit("px").withHelp("Blobs: the radius they respond to. Tubes: the smallest tube width").hiddenWhen("enhance", {"None"}),
                    doubleParam("enhance_sigma_max", "Feature σ max", 6.0).range(0.3, 100.0, 0.5, 1).withUnit("px").withHelp("Tubes: the largest width; the response is the best over the range").asAdvanced().visibleWhen("enhance", {"Tubes (Frangi)", "Neurites (Meijering)"}),
                    intParam("enhance_scales", "Scales", 4).range(1, 16).withHelp("Tubes: how many widths between the two σ").asAdvanced().visibleWhen("enhance", {"Tubes (Frangi)", "Neurites (Meijering)"}),
                    choiceParam("background", "Background", {"Top-hat (box)", "Rolling ball"}, "Top-hat (box)")
                        .withHelp("How the background is estimated before it is subtracted. The box top-hat follows a flat "
                                  "background; the rolling ball follows a curved one, which is what uneven illumination or a "
                                  "thick specimen actually gives"),
                    intParam("tophat", "Background radius", 0).range(0, 2000).withUnit("px").withHelp("Structures larger than this radius are background and are removed (0 = off)"),
                    doubleParam("sigma", "Smoothing σ", 1.0).range(0.0, 50.0, 0.5, 1).withUnit("px").withHelp("Gaussian blur before the threshold (0 = none)"),
                    choiceParam("method", "Threshold",
                                {"Otsu", "Triangle", "Li", "Yen", "Isodata", "Multi-Otsu", "Manual", "Percentile", "Local mean", "Local contrast"}, "Otsu")
                        .withHelp("Otsu splits the histogram in two; Triangle suits the skewed histogram of a mostly empty "
                                  "field, where Otsu cuts too high; Li keeps dim objects; the local rules follow an uneven "
                                  "background"),
                    doubleParam("value", "Value", 0.5).range(-1e9, 1e9, 0.01, 4).withHelp("Manual threshold").visibleWhen("method", {"Manual"}),
                    doubleParam("percentile", "Percentile", 90.0).range(0.0, 100.0, 0.5, 1).withUnit("%").visibleWhen("method", {"Percentile"}),
                    intParam("window", "Local window", 51).range(3, 4001).withUnit("px").withHelp("Local mean: side of the neighbourhood the mean is taken over").visibleWhen("method", {"Local mean", "Local contrast"}),
                    doubleParam("local_ratio", "Local ratio", 1.1).range(0.0, 10.0, 0.05, 2).withHelp("Local mean: foreground where value > ratio × local mean + offset").visibleWhen("method", {"Local mean"}),
                    doubleParam("local_offset", "Local offset", 0.0).range(-1e9, 1e9, 0.01, 4).asAdvanced().visibleWhen("method", {"Local mean", "Local contrast"}),
                    doubleParam("contrast_k", "Contrast k", 1.5).range(0.0, 10.0, 0.1, 2).withHelp("Local contrast: the cut sits k local standard deviations above the local mean, so it "
                                                                                                   "follows both the background level and the local noise")
                        .visibleWhen("method", {"Local contrast"}),
                    boolParam("hysteresis", "Hysteresis", false)
                        .withHelp("Keep everything connected to what is clearly above the cut, down to a lower one. A "
                                  "filament that fades stays whole instead of breaking into pieces"),
                    doubleParam("hysteresis_ratio", "Hysteresis low", 0.5).range(0.0, 1.0, 0.05, 2).withHelp("Where the lower cut sits between the image floor and the threshold: 0.5 is halfway, 1 turns hysteresis off").visibleWhen("hysteresis", {"on"}),
                    choiceParam("refine", "Refine", {"None", "Active contour (Chan-Vese)"}, "None")
                        .withHelp("Morphological Chan-Vese: moves the mask boundary to the best two-region fit of the image. "
                                  "It has no shape assumption, so it suits filaments as well as cells, and it repairs a "
                                  "threshold that leaked or pinched"),
                    intParam("refine_iterations", "Refine steps", 20).range(1, 500).asAdvanced().hiddenWhen("refine", {"None"}),
                    intParam("refine_smoothing", "Refine smoothing", 1).range(0, 5).withHelp("Curvature rounds per step; more gives a smoother contour").asAdvanced().hiddenWhen("refine", {"None"}),
                    intParam("opening", "Opening radius", 1).range(0, 100).withUnit("px").withHelp("Binary opening drops specks and necks thinner than this (0 = off)"),
                    boolParam("fill_holes", "Fill holes", true).withHelp("Enclosed background inside an object becomes object, per plane"),
                    boolParam("fill_holes_3d", "Fill holes (3D)", false)
                        .withHelp("Background the object encloses in three dimensions becomes object. A cavity no single plane "
                                  "closes is invisible to the per-plane fill, which is most of them in a stack"),
                    intParam("hole_max_voxels", "Largest hole", 0).range(0, 1000000000).withHelp("3D fill: leave cavities larger than this open (0 = fill them all), so a real lumen survives").asAdvanced().visibleWhen("fill_holes_3d", {"on"}),
                    choiceParam("post", "Instances", {"Watershed (distance)", "Watershed (gradient)", "Connected components"}, "Watershed (distance)")
                        .withHelp("How the mask becomes objects. The distance watershed splits at the waist between two "
                                  "objects; the gradient watershed splits where the image itself has an edge, which is what "
                                  "separates touching objects that have a visible boundary but no waist"),
                    choiceParam("seeds", "Seeds", {"Distance maxima", "H-maxima", "Blob centres (LoG)"}, "H-maxima")
                        .withHelp("What splits touching objects. The peaks of the distance map; only the peaks that stand "
                                  "clear of their surroundings (fewer false splits); or the centres of the blobs the image "
                                  "itself shows, found over a range of sizes, which is the one that copes with objects of "
                                  "different sizes")
                        .visibleWhen("post", {"Watershed (distance)", "Watershed (gradient)"}),
                    doubleParam("seed_distance", "Seed distance", 8.0).range(1.0, 200.0, 0.5, 1).withUnit("px").withHelp("Distance maxima: minimum distance between seeds").asAdvanced().visibleWhen("post", {"Watershed (distance)", "Watershed (gradient)"}).visibleWhen("seeds", {"Distance maxima"}),
                    doubleParam("seed_depth", "Seed depth", 2.0).range(0.1, 100.0, 0.5, 1).withUnit("px").withHelp("H-maxima: how far a peak must stand above its surroundings to be its own object; "
                                                                                                                   "raise it when one object is split, lower it when two are merged")
                        .visibleWhen("post", {"Watershed (distance)", "Watershed (gradient)"})
                        .visibleWhen("seeds", {"H-maxima"}),
                    doubleParam("blob_radius", "Object radius", 4.0).range(0.5, 500.0, 0.5, 1).withUnit("px").withHelp("Blob centres: the smallest object radius to look for, in x / y pixels").visibleWhen("post", {"Watershed (distance)", "Watershed (gradient)"}).visibleWhen("seeds", {"Blob centres (LoG)"}),
                    doubleParam("blob_radius_max", "Object radius max", 12.0).range(0.5, 500.0, 0.5, 1).withUnit("px").withHelp("Blob centres: the largest radius; the detector answers to every size in between").asAdvanced().visibleWhen("post", {"Watershed (distance)", "Watershed (gradient)"}).visibleWhen("seeds", {"Blob centres (LoG)"}),
                    intParam("blob_scales", "Blob scales", 5).range(1, 24).withHelp("Blob centres: how many sizes are tried between the two radii").asAdvanced().visibleWhen("post", {"Watershed (distance)", "Watershed (gradient)"}).visibleWhen("seeds", {"Blob centres (LoG)"}),
                    intParam("min_voxels", "Min. voxels", 20).range(0, 1000000000),
                    intParam("max_voxels", "Max. voxels", 0).range(0, 1000000000).withHelp("Drop objects larger than this (0 = off): the usual way to remove a merged clump").asAdvanced(),
                    doubleParam("min_fill", "Min. fill", 0.0).range(0.0, 1.0, 0.05, 2).withHelp("Drop objects that fill less than this fraction of their bounding box (0 = off): removes scattered debris").asAdvanced(),
                    doubleParam("max_elongation", "Max. elongation", 0.0).range(0.0, 100.0, 0.5, 1).withHelp("Drop objects whose bounding box is longer than this many times its width (0 = off)").asAdvanced(),
                    doubleParam("expand", "Expand labels", 0.0).range(0.0, 200.0, 0.5, 1).withUnit("px").withHelp("Grow every object outwards into the background by this much, nearest object first. Closes the "
                                                                                                                  "gap an opening or a watershed line left, without letting two objects meet (0 = off)"),
                    boolParam("skeleton", "Centrelines", false)
                        .withHelp("Replace every object with the line down its middle: one voxel thick, the same length and "
                                  "the same topology. The centreline of a filament, and what its length is measured on"),
                    boolParam("drop_border", "Drop border objects", false).withHelp("Objects touching the x / y edge are cut off, so their shape and size are not measurable"),
                    stringParam("class_name", "Class", "object").asAdvanced(),
                };
                // Starting points by what is being segmented, since almost
                // every setting below follows from that. Each is a plain set of
                // values: applying one is an undoable parameter change, and
                // everything stays editable. The numbers come from the
                // measurements in app/help/classic.md.
                info_.presets = {
                    {"Nuclei",
                     "Round, roughly convex objects that touch: a distance watershed on peaks that stand clear",
                     {{"denoise", std::string("None")},
                      {"enhance", std::string("None")},
                      {"background", std::string("Top-hat (box)")},
                      {"tophat", std::int64_t{0}},
                      {"sigma", 1.0},
                      {"method", std::string("Otsu")},
                      {"hysteresis", false},
                      {"refine", std::string("None")},
                      {"opening", std::int64_t{1}},
                      {"fill_holes", true},
                      {"fill_holes_3d", false},
                      {"post", std::string("Watershed (distance)")},
                      {"seeds", std::string("H-maxima")},
                      {"seed_depth", 2.0},
                      {"expand", 0.0},
                      {"skeleton", false},
                      {"min_voxels", std::int64_t{20}}}},
                    {"Cells (touching)",
                     "Objects pressed flat against each other: split on the image's own edges, not on a waist",
                     {{"denoise", std::string("Median 3x3")},
                      {"enhance", std::string("None")},
                      {"background", std::string("Rolling ball")},
                      {"tophat", std::int64_t{25}},
                      {"sigma", 1.0},
                      {"method", std::string("Otsu")},
                      {"hysteresis", false},
                      {"refine", std::string("None")},
                      {"opening", std::int64_t{1}},
                      {"fill_holes", true},
                      {"fill_holes_3d", true},
                      {"post", std::string("Watershed (gradient)")},
                      {"seeds", std::string("H-maxima")},
                      {"seed_depth", 2.0},
                      {"expand", 0.0},
                      {"skeleton", false},
                      {"min_voxels", std::int64_t{50}}}},
                    {"Puncta",
                     "Small round spots of more than one size: found by the image itself, over a range of radii",
                     {{"denoise", std::string("None")},
                      {"enhance", std::string("Blobs (DoG)")},
                      {"enhance_sigma", 2.0},
                      {"background", std::string("Top-hat (box)")},
                      {"tophat", std::int64_t{0}},
                      {"sigma", 0.0},
                      {"method", std::string("Otsu")},
                      {"hysteresis", false},
                      {"refine", std::string("None")},
                      {"opening", std::int64_t{0}},
                      {"fill_holes", false},
                      {"fill_holes_3d", false},
                      {"post", std::string("Watershed (distance)")},
                      {"seeds", std::string("Blob centres (LoG)")},
                      {"blob_radius", 2.0},
                      {"blob_radius_max", 8.0},
                      {"blob_scales", std::int64_t{5}},
                      {"expand", 0.0},
                      {"skeleton", false},
                      {"min_voxels", std::int64_t{5}}}},
                    {"Filaments",
                     "Long thin structures traced whole: Frangi tubes, and hysteresis so a fading filament stays one object",
                     {{"denoise", std::string("None")},
                      {"enhance", std::string("Tubes (Frangi)")},
                      {"enhance_sigma", 0.8},
                      {"enhance_sigma_max", 2.0},
                      {"enhance_scales", std::int64_t{3}},
                      {"background", std::string("Top-hat (box)")},
                      {"tophat", std::int64_t{0}},
                      {"sigma", 0.0},
                      {"method", std::string("Otsu")},
                      {"hysteresis", true},
                      {"hysteresis_ratio", 0.5},
                      {"refine", std::string("None")},
                      {"opening", std::int64_t{0}},
                      {"fill_holes", false},
                      {"fill_holes_3d", false},
                      {"post", std::string("Connected components")},
                      {"expand", 0.0},
                      {"skeleton", false},
                      {"min_voxels", std::int64_t{20}}}},
                    {"Filament network",
                     "A mesh with crossings: Meijering, which answers at a junction where Frangi cuts the network apart",
                     {{"denoise", std::string("None")},
                      {"enhance", std::string("Neurites (Meijering)")},
                      {"enhance_sigma", 0.8},
                      {"enhance_sigma_max", 2.0},
                      {"enhance_scales", std::int64_t{3}},
                      {"background", std::string("Top-hat (box)")},
                      {"tophat", std::int64_t{0}},
                      {"sigma", 0.0},
                      {"method", std::string("Otsu")},
                      {"hysteresis", false},
                      {"refine", std::string("None")},
                      {"opening", std::int64_t{0}},
                      {"fill_holes", false},
                      {"fill_holes_3d", false},
                      {"post", std::string("Connected components")},
                      {"expand", 0.0},
                      {"skeleton", false},
                      {"min_voxels", std::int64_t{20}}}},
                    {"Centrelines",
                     "The network again, thinned to the line down the middle of each filament: what a length is measured on",
                     {{"denoise", std::string("None")},
                      {"enhance", std::string("Neurites (Meijering)")},
                      {"enhance_sigma", 0.8},
                      {"enhance_sigma_max", 2.0},
                      {"enhance_scales", std::int64_t{3}},
                      {"background", std::string("Top-hat (box)")},
                      {"tophat", std::int64_t{0}},
                      {"sigma", 0.0},
                      {"method", std::string("Otsu")},
                      {"hysteresis", false},
                      {"refine", std::string("None")},
                      {"opening", std::int64_t{0}},
                      {"fill_holes", false},
                      {"fill_holes_3d", false},
                      {"post", std::string("Connected components")},
                      {"expand", 0.0},
                      {"skeleton", true},
                      {"min_voxels", std::int64_t{20}}}},
                    {"Faint or noisy",
                     "Weak signal on a busy background: diffusion first, a cut that keeps dim objects, then hysteresis",
                     {{"denoise", std::string("Anisotropic diffusion")},
                      {"diffusion_iterations", std::int64_t{5}},
                      {"diffusion_k", 0.1},
                      {"enhance", std::string("None")},
                      {"background", std::string("Rolling ball")},
                      {"tophat", std::int64_t{25}},
                      {"sigma", 0.0},
                      {"method", std::string("Triangle")},
                      {"hysteresis", true},
                      {"hysteresis_ratio", 0.5},
                      {"refine", std::string("Active contour (Chan-Vese)")},
                      {"refine_iterations", std::int64_t{20}},
                      {"refine_smoothing", std::int64_t{1}},
                      {"opening", std::int64_t{1}},
                      {"fill_holes", true},
                      {"fill_holes_3d", false},
                      {"post", std::string("Watershed (distance)")},
                      {"seeds", std::string("H-maxima")},
                      {"seed_depth", 2.0},
                      {"expand", 0.0},
                      {"skeleton", false},
                      {"min_voxels", std::int64_t{20}}}},
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            std::string summary(const ParamSet& p, const DatasetMeta& meta) const override {
                const std::string method = p.getString("method", "Otsu");
                std::string cut = method;
                if (method == "Manual") cut = "> " + formatNumber(p.getDouble("value", 0.5), 3);
                else if (method == "Percentile") cut = "> p" + formatNumber(p.getDouble("percentile", 90.0), 0);
                else if (method == "Local mean") cut = "local " + std::to_string(p.getInt("window", 51)) + " px";
                else if (method == "Local contrast") cut = "local ± " + std::to_string(p.getInt("window", 51)) + " px";
                const std::string enhance = p.getString("enhance", "None");
                const std::string enhanceText =
                    enhance == "Blobs (DoG)"      ? "blobs σ " + formatNumber(p.getDouble("enhance_sigma", 2.0), 1)
                    : enhance == "Tubes (Frangi)" ? "tubes σ " + formatNumber(p.getDouble("enhance_sigma", 2.0), 1) + "-" +
                                                        formatNumber(p.getDouble("enhance_sigma_max", 6.0), 1)
                    : enhance == "Neurites (Meijering)"
                        ? "neurites σ " + formatNumber(p.getDouble("enhance_sigma", 2.0), 1) + "-" +
                              formatNumber(p.getDouble("enhance_sigma_max", 6.0), 1)
                        : std::string();
                const double sigma = p.getDouble("sigma", 1.0);
                const Index tophat = p.getInt("tophat", 0);
                const std::string denoise = p.getString("denoise", "None");
                if (p.getBool("hysteresis", false) && p.getDouble("hysteresis_ratio", 0.5) < 1.0) cut += " + hysteresis";
                return joinSummary({channelName(meta, p.getInt("channel", 0)),
                                    denoise == "Median 3x3" ? "median" : denoise == "Anisotropic diffusion" ? "diffusion"
                                                                                                            : "",
                                    enhanceText,
                                    tophat > 0 ? (p.getString("background", "Top-hat (box)") == "Rolling ball" ? "ball " : "top-hat ") +
                                                     std::to_string(tophat)
                                               : "",
                                    sigma > 0 ? "σ " + formatNumber(sigma, 1) : "",
                                    cut,
                                    p.getString("refine", "None") != "None" ? "snake" : "",
                                    p.getBool("skeleton", false) ? "centrelines" : "",
                                    p.getString("post", "Watershed (distance)").rfind("Watershed", 0) != 0  ? "components"
                                    : p.getString("post", "Watershed (distance)") == "Watershed (gradient)" ? "watershed edges"
                                    : p.getString("seeds", "H-maxima") == "Blob centres (LoG)"
                                        ? "watershed blobs r " + formatNumber(p.getDouble("blob_radius", 4.0), 1)
                                    : p.getString("seeds", "H-maxima") == "H-maxima" ? "watershed h"
                                                                                     : "watershed d"});
            }

            StepOutput run(const StepInput& input, const ParamSet& p, const StepContext& ctx) const override {
                const Validation v = validate(p, input.meta);
                if (!v.ok()) throw std::runtime_error(v.firstError());
                const DatasetMeta& meta = input.meta;
                const Dims5& d = meta.dims;
                const Index channel = p.getInt("channel", 0);
                const Index n = d.z * d.y * d.x, plane = d.y * d.x;
                StepOutput out;
                out.meta = meta;
                out.array = input.materialize([&](double f, const std::string& m) { ctx.report(0.1 * f, m); });
                auto labels = std::make_shared<LabelVolume>(d.t, d.z, d.y, d.x);

                const Index tophat = p.getInt("tophat", 0);
                const double sigma = p.getDouble("sigma", 1.0);
                const std::string enhance = p.getString("enhance", "None");
                const double enhanceSigma = p.getDouble("enhance_sigma", 2.0);
                const double enhanceSigmaMax = p.getDouble("enhance_sigma_max", 6.0);
                const int enhanceScales = static_cast<int>(p.getInt("enhance_scales", 4));
                const double contrastK = p.getDouble("contrast_k", 1.5);
                const std::string method = p.getString("method", "Otsu");
                const Index window = std::max<Index>(1, p.getInt("window", 51) / 2);
                const double ratio = p.getDouble("local_ratio", 1.1), offset = p.getDouble("local_offset", 0.0);
                const Index opening = p.getInt("opening", 1);
                const bool fillHoles = p.getBool("fill_holes", true);
                const bool fillHolesVolume = p.getBool("fill_holes_3d", false);
                const Index holeMaxVoxels = p.getInt("hole_max_voxels", 0);
                const double expand = p.getDouble("expand", 0.0);
                const bool rollingBall = p.getString("background", "Top-hat (box)") == "Rolling ball";
                const bool skeleton = p.getBool("skeleton", false);
                const std::string denoise = p.getString("denoise", "None");
                const int diffusionIterations = static_cast<int>(p.getInt("diffusion_iterations", 5));
                const double diffusionK = p.getDouble("diffusion_k", 0.1);
                const bool hysteresis = p.getBool("hysteresis", false) && p.getDouble("hysteresis_ratio", 0.5) < 1.0;
                const double hysteresisRatio = p.getDouble("hysteresis_ratio", 0.5);
                const std::string refine = p.getString("refine", "None");
                const int refineIterations = static_cast<int>(p.getInt("refine_iterations", 20));
                const int refineSmoothing = static_cast<int>(p.getInt("refine_smoothing", 1));
                ShapeFilter shape;
                shape.maxVoxels = p.getInt("max_voxels", 0);
                shape.minFill = p.getDouble("min_fill", 0.0);
                shape.maxElongation = p.getDouble("max_elongation", 0.0);
                shape.dropBorder = p.getBool("drop_border", false);
                const bool shapeFilters = shape.maxVoxels > 0 || shape.minFill > 0.0 || shape.maxElongation > 0.0 || shape.dropBorder;

                LabelPostOptions post;
                const std::string postChoice = p.getString("post", "Watershed (distance)");
                const bool gradientWatershed = postChoice == "Watershed (gradient)";
                post.post = postChoice.rfind("Watershed", 0) == 0 ? "Watershed (distance)" : "Connected components";
                post.threshold = 0.5;
                post.minVoxels = p.getInt("min_voxels", 20);
                post.seedMinDistance = p.getDouble("seed_distance", 8.0);
                post.seeds = p.getString("seeds", "H-maxima");
                post.seedDepth = p.getDouble("seed_depth", 2.0);
                const bool blobSeeds = post.seeds == "Blob centres (LoG)" && post.post.rfind("Watershed", 0) == 0;
                // the LoG answers strongest at sigma ~ r / sqrt(3) for a ball of radius r
                const double blobSigma = std::max(0.3, p.getDouble("blob_radius", 4.0) / std::sqrt(3.0));
                const double blobSigmaMax = std::max(blobSigma, p.getDouble("blob_radius_max", 12.0) / std::sqrt(3.0));
                const int blobScales = static_cast<int>(p.getInt("blob_scales", 5));
                const double zAspect = meta.voxelUm[0] > 0.0 ? std::max(1e-6, meta.voxelUm[2] / meta.voxelUm[0]) : 1.0;
                const double zAspectOfMeta = zAspect;
                std::vector<std::uint32_t> seedVolume;
                post.className = p.getString("class_name", "object");

                std::vector<float> work(static_cast<std::size_t>(n)), tmp, scratch, localMean, localStd, enhA, enhB, edges;
                std::vector<std::uint8_t> maskLow, morph;
                std::vector<double> integral, integralSq;
                std::vector<std::uint8_t> mask(static_cast<std::size_t>(n)), maskTmp, seen;
                std::vector<Index> stack;
                std::vector<float> fg(static_cast<std::size_t>(n));
                std::uint32_t total = 0;
                std::string cutText;
                double foregroundFraction = 0.0;
                for (Index t = 0; t < d.t; ++t) {
                    ctx.throwIfCancelled();
                    const double base = 0.1 + 0.9 * static_cast<double>(t) / d.t, span = 0.9 / d.t;
                    const BufferView<const float> vol = out.array->volume(channel, t);
                    std::copy_n(vol.data(), n, work.data());
                    // 1. denoise, then enhance the feature, flatten and smooth.
                    // Tubes are a 3D filter over the whole volume; the rest per plane.
                    if (denoise != "None") {
                        ctx.report(base + span * 0.05, "denoising");
                        for (Index z = 0; z < d.z; ++z) {
                            ctx.throwIfCancelled();
                            float* pl = work.data() + z * plane;
                            if (denoise == "Median 3x3") medianFilterPlane(pl, d.y, d.x, tmp);
                            else anisotropicDiffusionPlane(pl, d.y, d.x, diffusionIterations, diffusionK, tmp);
                        }
                    }
                    if (enhance == "Tubes (Frangi)" || enhance == "Neurites (Meijering)") {
                        const bool tubes = enhance == "Tubes (Frangi)";
                        ctx.report(base + span * 0.1, tubes ? "vesselness" : "neuriteness");
                        enhA.resize(static_cast<std::size_t>(n));
                        const auto poll = [&] { ctx.throwIfCancelled(); };
                        if (tubes)
                            frangiVolume(work.data(), enhA.data(), d.z, d.y, d.x, zAspectOfMeta, enhanceSigma,
                                         std::max(enhanceSigma, enhanceSigmaMax), enhanceScales, scratch, tmp, poll);
                        else
                            meijeringVolume(work.data(), enhA.data(), d.z, d.y, d.x, zAspectOfMeta, enhanceSigma,
                                            std::max(enhanceSigma, enhanceSigmaMax), enhanceScales, scratch, tmp, poll);
                        std::copy_n(enhA.data(), n, work.data());
                    }
                    for (Index z = 0; z < d.z; ++z) {
                        ctx.throwIfCancelled();
                        float* pl = work.data() + z * plane;
                        if (enhance == "Blobs (DoG)") {
                            enhA.resize(static_cast<std::size_t>(plane));
                            dogPlane(pl, enhA.data(), d.y, d.x, enhanceSigma, 1.6, scratch, enhB, tmp);
                            std::copy_n(enhA.data(), plane, pl);
                        }
                        if (rollingBall) rollingBallPlane(pl, d.y, d.x, static_cast<double>(tophat), scratch);
                        else topHatPlane(pl, d.y, d.x, tophat, scratch, tmp);
                        gaussianPlane(pl, d.y, d.x, sigma, tmp);
                        if (z % 8 == 0) ctx.report(base + span * 0.3 * z / std::max<Index>(1, d.z), "filtering");
                    }
                    ctx.throwIfCancelled();
                    // 2. threshold
                    float cut = static_cast<float>(p.getDouble("value", 0.5));
                    if (method == "Otsu") cut = otsuThreshold(work.data(), n);
                    else if (method == "Triangle") cut = triangleThreshold(work.data(), n);
                    else if (method == "Li") cut = liThreshold(work.data(), n);
                    else if (method == "Yen") cut = yenThreshold(work.data(), n);
                    else if (method == "Isodata") cut = isodataThreshold(work.data(), n);
                    else if (method == "Multi-Otsu") cut = multiOtsuThresholds(work.data(), n).second;
                    else if (method == "Percentile") cut = percentiles(work.data(), n, 0.0, p.getDouble("percentile", 90.0)).second;
                    if (method == "Local mean" || method == "Local contrast") {
                        const bool byContrast = method == "Local contrast";
                        localMean.resize(static_cast<std::size_t>(plane));
                        if (byContrast) localStd.resize(static_cast<std::size_t>(plane));
                        if (hysteresis) maskLow.assign(static_cast<std::size_t>(n), 0);
                        for (Index z = 0; z < d.z; ++z) {
                            ctx.throwIfCancelled();
                            const float* pl = work.data() + z * plane;
                            std::uint8_t* m = mask.data() + z * plane;
                            std::uint8_t* lowM = hysteresis ? maskLow.data() + z * plane : nullptr;
                            if (byContrast) {
                                // k standard deviations above the local mean: the cut
                                // rises with the background and with the local noise,
                                // so a flat region does not turn into speckle
                                localStatsPlane(pl, d.y, d.x, window, localMean.data(), localStd.data(), integral, integralSq);
                                for (Index i = 0; i < plane; ++i) {
                                    const double mu = localMean[static_cast<std::size_t>(i)];
                                    const double sd = localStd[static_cast<std::size_t>(i)];
                                    const double T = mu + contrastK * sd + offset;
                                    m[i] = pl[i] > T ? 1 : 0;
                                    if (lowM) lowM[i] = pl[i] > mu + hysteresisRatio * (T - mu) ? 1 : 0;
                                }
                            } else {
                                localMeanPlane(pl, d.y, d.x, window, localMean.data(), integral);
                                for (Index i = 0; i < plane; ++i) {
                                    const double mu = localMean[static_cast<std::size_t>(i)];
                                    const double T = ratio * mu + offset;
                                    m[i] = pl[i] > T ? 1 : 0;
                                    if (lowM) lowM[i] = pl[i] > mu + hysteresisRatio * (T - mu) ? 1 : 0;
                                }
                            }
                        }
                        if (t == 0) cutText = byContrast ? "local mean + " + formatNumber(contrastK, 2) + " SD"
                                                         : "local mean × " + formatNumber(ratio, 2);
                    } else {
                        for (Index i = 0; i < n; ++i) mask[static_cast<std::size_t>(i)] = work[static_cast<std::size_t>(i)] > cut ? 1 : 0;
                        if (t == 0) cutText = "> " + formatNumber(cut, 4);
                        if (hysteresis) {
                            // the lower cut sits between the image floor and the
                            // threshold, so the setting means the same thing
                            // whatever the units and even when values are negative
                            const float floorValue = *std::min_element(work.begin(), work.end());
                            const float lowCut = floorValue + static_cast<float>(hysteresisRatio) * (cut - floorValue);
                            maskLow.assign(static_cast<std::size_t>(n), 0);
                            for (Index i = 0; i < n; ++i) maskLow[static_cast<std::size_t>(i)] = work[static_cast<std::size_t>(i)] > lowCut ? 1 : 0;
                            if (t == 0) cutText += " (down to " + formatNumber(lowCut, 4) + ")";
                        }
                    }
                    if (hysteresis) {
                        ctx.throwIfCancelled();
                        hysteresisMask(mask.data(), maskLow.data(), d.z, d.y, d.x, mask.data());
                    }
                    ctx.report(base + span * 0.5, "mask");
                    // 2b. move the boundary to the best two-region fit of the image
                    if (refine != "None") {
                        ctx.report(base + span * 0.55, "active contour");
                        for (Index z = 0; z < d.z; ++z) {
                            ctx.throwIfCancelled();
                            morphologicalChanVesePlane(work.data() + z * plane, mask.data() + z * plane, d.y, d.x, refineIterations,
                                                       refineSmoothing, morph);
                        }
                    }
                    // 3. clean the mask, per plane
                    for (Index z = 0; z < d.z; ++z) {
                        ctx.throwIfCancelled();
                        std::uint8_t* m = mask.data() + z * plane;
                        if (opening > 0) {
                            boxFilterPlane<std::uint8_t, false>(m, d.y, d.x, opening, maskTmp);
                            boxFilterPlane<std::uint8_t, true>(m, d.y, d.x, opening, maskTmp);
                        }
                        if (fillHoles) fillHolesPlane(m, d.y, d.x, seen, stack);
                    }
                    // the 3D fill sees the whole volume, so it runs once every
                    // plane has been cleaned
                    if (fillHolesVolume) fillHoles3D(mask.data(), d.z, d.y, d.x, holeMaxVoxels);
                    ctx.throwIfCancelled();
                    Index on = 0;
                    for (Index i = 0; i < n; ++i) {
                        fg[static_cast<std::size_t>(i)] = mask[static_cast<std::size_t>(i)] ? 1.0f : 0.0f;
                        on += mask[static_cast<std::size_t>(i)];
                    }
                    foregroundFraction += static_cast<double>(on) / static_cast<double>(std::max<Index>(1, n)) / d.t;
                    // 4. seeds from the image itself, when asked for
                    post.externalSeeds = nullptr;
                    if (blobSeeds) {
                        ctx.report(base + span * 0.6, "blob centres");
                        seedVolume.assign(static_cast<std::size_t>(n), 0u);
                        post.externalSeedCount = logBlobSeeds(work.data(), mask.data(), d.z, d.y, d.x, zAspect, blobSigma,
                                                              blobSigmaMax, blobScales, seedVolume.data());
                        post.externalSeeds = seedVolume.data();
                    }
                    ctx.throwIfCancelled();
                    ctx.report(base + span * 0.8, "instances");
                    // 5. instances. The gradient watershed floods the image's own
                    // edges instead of the distance map, which is what splits
                    // touching objects that have a boundary but no waist.
                    const float* landscape = nullptr;
                    if (gradientWatershed) {
                        edges.resize(static_cast<std::size_t>(n));
                        gradientMagnitude(work.data(), d.z, d.y, d.x, zAspect, edges.data());
                        landscape = edges.data();
                    }
                    std::uint32_t made = labelsFromProbabilities(fg.data(), landscape, d.z, d.y, d.x, post, *labels, t);
                    if (shapeFilters) {
                        made = filterLabelsByShape(labels->volume(t), d.z, d.y, d.x, shape);
                        labels->recomputeStats(t, fg.data());
                        for (LabelStats& st : labels->stats()) st.cls = post.className;
                        labels->applyFlags(post.flags);
                    }
                    // last, so the shape filters measure the objects as they
                    // were segmented and not as they were grown
                    if (expand > 0.0) {
                        expandLabels(labels->volume(t), d.z, d.y, d.x, expand, zAspect);
                        labels->recomputeStats(t, fg.data());
                        for (LabelStats& st : labels->stats()) st.cls = post.className;
                        labels->applyFlags(post.flags);
                    }
                    // centrelines last of all: they are what is left of the
                    // objects, so everything measured on the objects is measured
                    // before they are thinned away
                    if (skeleton) {
                        std::uint32_t* volume = labels->volume(t);
                        std::vector<std::uint8_t> solid(static_cast<std::size_t>(n));
                        for (Index i = 0; i < n; ++i) solid[static_cast<std::size_t>(i)] = volume[i] ? 1 : 0;
                        skeletonize3D(solid.data(), d.z, d.y, d.x, [&] { ctx.throwIfCancelled(); });
                        for (Index i = 0; i < n; ++i)
                            if (!solid[static_cast<std::size_t>(i)]) volume[i] = 0;
                        labels->recomputeStats(t, fg.data());
                        for (LabelStats& st : labels->stats()) st.cls = post.className;
                        labels->applyFlags(post.flags);
                    }
                    total += made;
                }
                for (LabelStats& s : labels->stats()) s.confidence = 1.0;   // intensities, not probabilities
                labels->applyFlags(post.flags);
                out.labels = labels;
                out.ranOn = Backend::Cpu;
                out.note = cutText + " · " + std::to_string(total) + " labels · CPU";
                out.diagnostics = labelDiagnostics(*labels, summary(p, meta) + " · " + std::to_string(total) + " labels");
                out.diagnostics.facts.push_back({"Threshold", cutText});
                out.diagnostics.facts.push_back({"Foreground", formatNumber(100.0 * foregroundFraction, 1) + " %"});
                ctx.report(1.0, "");
                return out;
            }

        private:
            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeClassicalSegmentationOperation() { return std::make_unique<ClassicalSegmentationOperation>(); }

} // namespace sirius::app
