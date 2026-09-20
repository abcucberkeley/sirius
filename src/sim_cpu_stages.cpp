// The data-parallel SIM preprocessing stages on host memory
// (sim_cpu_stages.hpp). In a file of their own: the public preprocessing and
// separation API needs these and nothing else of the reconstruction, which is
// the CPU backend in sim_cpu.cpp.

#include "sim_cpu_stages.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "sirius/constants.hpp"

namespace sirius::simdetail {

    namespace cpu {

        void scaleShift(double* data, IndexT n, double sub, double mul) {
#pragma omp parallel for schedule(static)
            for (IndexT i = 0; i < n; ++i) data[i] = (data[i] - sub) * mul;
        }

        void planeSums(const double* data, IndexT nplanes, IndexT planeElems, double* sums) {
#pragma omp parallel for schedule(static)
            for (IndexT p = 0; p < nplanes; ++p) {
                const double* s = data + p * planeElems;
                double acc = 0.0;
                for (IndexT i = 0; i < planeElems; ++i) acc += s[i];
                sums[p] = acc;
            }
        }

        void scalePlanes(double* data, IndexT nplanes, IndexT planeElems, const double* factors) {
#pragma omp parallel for schedule(static)
            for (IndexT p = 0; p < nplanes; ++p) {
                double* s = data + p * planeElems;
                const double f = factors[p];
                for (IndexT i = 0; i < planeElems; ++i) s[i] *= f;
            }
        }

        void bleachFactors(const double* sums, IndexT ndirs, IndexT nphases, IndexT nz,
                           bool equalizez, double* factors) {
            auto at = [&](IndexT d, IndexT p, IndexT z) { return (d * nphases + p) * nz + z; };
            for (IndexT d = 0; d < ndirs; ++d)
                for (IndexT p = 0; p < nphases; ++p)
                    for (IndexT z = 0; z < nz; ++z) {
                        const double ref = sums[at(0, 0, equalizez ? 0 : z)];
                        const double s = sums[at(d, p, z)];
                        factors[at(d, p, z)] = s != 0.0 ? ref / s : 1.0;
                    }
        }

        void edgeApodize(double* data, IndexT nsec, IndexT ny, IndexT nx, int napodize) {
            if (napodize <= 0 || nsec <= 0) return;
            const IndexT napY = std::min<IndexT>(napodize, ny);
            const IndexT napX = std::min<IndexT>(napodize, nx);

            // The taper depends only on the border index: evaluate the sine
            // once per index instead of once per (section, column, index).
            std::vector<double> fact(static_cast<std::size_t>(napodize));
            for (IndexT i = 0; i < napodize; ++i)
                fact[static_cast<std::size_t>(i)] =
                    1.0 - std::sin((static_cast<double>(i) + 0.5) / napodize * kPi * 0.5);

#pragma omp parallel
            {
                // per-thread scratch: the top/bottom difference of every column,
                // read from the untouched edge rows before any row is modified
                std::vector<double> diff(static_cast<std::size_t>(nx));

#pragma omp for schedule(static)
                for (IndexT s = 0; s < nsec; ++s) {
                    double* img = data + s * ny * nx;

                    // pass 1: blend top/bottom rows. Row-contiguous inner loops
                    // (l outer, k inner) stream through memory; the arithmetic
                    // per element is identical to the column-wise formulation.
                    const double* top = img;
                    const double* bottom = img + (ny - 1) * nx;
                    for (IndexT k = 0; k < nx; ++k) diff[static_cast<std::size_t>(k)] = (bottom[k] - top[k]) * 0.5;
                    for (IndexT l = 0; l < napY; ++l) {
                        const double fl = fact[static_cast<std::size_t>(l)];
                        double* rowLo = img + l * nx;
                        double* rowHi = img + (ny - 1 - l) * nx;
                        for (IndexT k = 0; k < nx; ++k) {
                            const double f = diff[static_cast<std::size_t>(k)] * fl;
                            rowLo[k] += f;
                            rowHi[k] -= f;
                        }
                    }

                    // pass 2: blend left/right columns of every row (sees pass 1's result)
                    for (IndexT l = 0; l < ny; ++l) {
                        double* row = img + l * nx;
                        const double d = (row[nx - 1] - row[0]) * 0.5;
                        for (IndexT k = 0; k < napX; ++k) {
                            const double f = d * fact[static_cast<std::size_t>(k)];
                            row[k] += f;
                            row[nx - 1 - k] -= f;
                        }
                    }
                }
            }
        }

        void cosineApodize(double* data, IndexT nsec, IndexT ny, IndexT nx) {
            // separable window: precompute the two 1D factors once
            std::vector<double> xf(static_cast<std::size_t>(nx)), yf(static_cast<std::size_t>(ny));
            for (IndexT k = 0; k < nx; ++k)
                xf[static_cast<std::size_t>(k)] = std::sin(kPi * (static_cast<double>(k) + 0.5) / nx);
            for (IndexT l = 0; l < ny; ++l)
                yf[static_cast<std::size_t>(l)] = std::sin(kPi * (static_cast<double>(l) + 0.5) / ny);
#pragma omp parallel for schedule(static)
            for (IndexT s = 0; s < nsec; ++s) {
                double* img = data + s * ny * nx;
                for (IndexT l = 0; l < ny; ++l) {
                    const double yl = yf[static_cast<std::size_t>(l)];
                    double* row = img + l * nx;
                    for (IndexT k = 0; k < nx; ++k) row[k] *= xf[static_cast<std::size_t>(k)] * yl;
                }
            }
        }

        void separate(const double* phases, double* bands, const double* mat,
                      int nphases, int nbands, IndexT n) {
            // Blocked over voxels: a block of every phase volume stays in L1
            // while all nbands outputs are formed, so each input element is
            // read from memory once (not once per band) and every inner loop
            // runs over contiguous voxels, which vectorizes. Summation order
            // per voxel (p ascending) matches the naive formulation.
            constexpr IndexT kBlock = 512;
#pragma omp parallel for schedule(static)
            for (IndexT i0 = 0; i0 < n; i0 += kBlock) {
                const IndexT len = std::min(kBlock, n - i0);
                for (int b = 0; b < nbands; ++b) {
                    const double* m = mat + static_cast<IndexT>(b) * nphases;
                    double* out = bands + static_cast<IndexT>(b) * n + i0;
                    const double* src0 = phases + i0;
                    for (IndexT i = 0; i < len; ++i) out[i] = m[0] * src0[i];
                    for (int p = 1; p < nphases; ++p) {
                        const double mp = m[p];
                        const double* src = phases + static_cast<IndexT>(p) * n + i0;
                        for (IndexT i = 0; i < len; ++i) out[i] += mp * src[i];
                    }
                }
            }
        }

    } // namespace cpu

} // namespace sirius::simdetail
