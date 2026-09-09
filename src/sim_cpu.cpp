// CPU (OpenMP) implementation of the SIM reconstruction stages. The per-voxel
// arithmetic lives in sim_math.hpp and is shared with the CUDA backend; the
// preprocessing stages live in the simdetail::cpu free functions below and are
// shared with the public Eigen API (preprocess.cpp, separation.cpp).

#include "sim_internal.hpp"
#include "sim_cpu_stages.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "sirius/constants.hpp"

namespace sirius::simdetail {

    // --- shared CPU stages --------------------------------------------------

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

    // --- backend ------------------------------------------------------------

    namespace {

        class CpuSimBackend final : public SimBackend {
        public:
            void reorderFrames(const double* raw, double* frames,
                               IndexT ndirs, IndexT nphases, IndexT nz,
                               IndexT planeElems, bool fastSi) override {
                const IndexT nplanes = ndirs * nphases * nz;
#pragma omp parallel for schedule(static)
                for (IndexT dst = 0; dst < nplanes; ++dst) {
                    const IndexT z = dst % nz;
                    const IndexT ph = (dst / nz) % nphases;
                    const IndexT d = dst / (nz * nphases);
                    const IndexT src = fastSi ? (z * ndirs + d) * nphases + ph
                                              : (d * nz + z) * nphases + ph;
                    std::memcpy(frames + dst * planeElems, raw + src * planeElems,
                                static_cast<std::size_t>(planeElems) * sizeof(double));
                }
            }

            void scaleShift(double* data, IndexT n, double sub, double mul) override {
                cpu::scaleShift(data, n, sub, mul);
            }

            void planeSums(const double* data, IndexT nplanes, IndexT planeElems,
                           double* hostSums) override {
                cpu::planeSums(data, nplanes, planeElems, hostSums);
            }

            void scalePlanes(double* data, IndexT nplanes, IndexT planeElems,
                             const double* hostFactors) override {
                cpu::scalePlanes(data, nplanes, planeElems, hostFactors);
            }

            void edgeApodize(double* data, IndexT nsec, IndexT ny, IndexT nx,
                             int napodize) override {
                cpu::edgeApodize(data, nsec, ny, nx, napodize);
            }

            void cosineApodize(double* data, IndexT nsec, IndexT ny, IndexT nx) override {
                cpu::cosineApodize(data, nsec, ny, nx);
            }

            void separate(const double* phases, double* bands, const double* hostMat,
                          int nphases, int nbands, IndexT n) override {
                cpu::separate(phases, bands, hostMat, nphases, nbands, n);
            }

            void makeOverlaps(const OverlapCtx& c,
                              const Cd* bandRe1, const Cd* bandIm1,
                              const Cd* bandRe2, const Cd* bandIm2,
                              Cd* ov0, Cd* ov1, int zdistcutoff) override {
                const IndexT nzc = 2 * static_cast<IndexT>(zdistcutoff) + 1;
                const IndexT rows = nzc * c.ny;
#pragma omp parallel for schedule(static)
                for (IndexT r = 0; r < rows; ++r) {
                    const IndexT zs = r / c.ny - zdistcutoff;
                    const IndexT iy = r % c.ny;
                    const IndexT zout = signedToStorage(zs, c.nz);
                    Cd* row0 = ov0 + (zout * c.ny + iy) * c.nx;
                    Cd* row1 = ov1 + (zout * c.ny + iy) * c.nx;
                    for (IndexT ix = 0; ix < c.nx; ++ix) {
                        row0[ix] = overlap0Value(c, bandRe1, bandIm1, zs, iy, ix);
                        row1[ix] = overlap1Value(c, bandRe2, bandIm2, zs, iy, ix);
                    }
                }
            }

            void crossCorrelate(const Cd* ov0, const Cd* ov1, Cd* plane,
                                IndexT nz, IndexT ny, IndexT nx) override {
                const IndexT sec = ny * nx;
#pragma omp parallel for schedule(static)
                for (IndexT i = 0; i < sec; ++i) {
                    Cd acc = cd(0, 0);
                    for (IndexT z = 0; z < nz; ++z)
                        acc = cadd(acc, cmul(ov0[z * sec + i], cconj(ov1[z * sec + i])));
                    plane[i] = acc;
                }
            }

            ModampSums modampReduce(const Cd* ov0, const Cd* ov1,
                                    IndexT nz, IndexT ny, IndexT nx,
                                    double angleX, double angleY) override {
                // Deterministic reduction. `reduction(+:...)` combines the
                // per-thread partials in completion order, so the rounding of
                // this sum -- and with it the |modamp|^2 that the k0 bracket
                // search maximizes -- varies from run to run. Instead reduce
                // fixed-size blocks of the plane and combine them serially in
                // block order: the summation tree then depends only on `sec`,
                // not on the thread count or the schedule, so two
                // reconstructions of the same input agree bit for bit (this
                // matches what the CUDA backend already does with its
                // per-block partials).
                constexpr IndexT kBlock = 1024;
                const IndexT sec = ny * nx;
                const IndexT nblocks = (sec + kBlock - 1) / kBlock;
                struct Partial {
                    double xyRe, xyIm, sumX, sumY;
                };
                std::vector<Partial> partials(static_cast<std::size_t>(nblocks));

#pragma omp parallel for schedule(static)
                for (IndexT b = 0; b < nblocks; ++b) {
                    const IndexT begin = b * kBlock;
                    const IndexT end = std::min(begin + kBlock, sec);
                    double xyRe = 0, xyIm = 0, sumX = 0, sumY = 0;
                    for (IndexT i = begin; i < end; ++i) {
                        const IndexT iy = i / nx;
                        const IndexT ix = i % nx;
                        const double angle = angleX * (static_cast<double>(ix) - 0.5 * static_cast<double>(nx)) +
                                             angleY * (static_cast<double>(iy) - 0.5 * static_cast<double>(ny));
                        const Cd ramp = cd(cos(angle), sin(angle));
                        Cd acc = cd(0, 0);
                        for (IndexT z = 0; z < nz; ++z) {
                            const Cd a = ov0[z * sec + i];
                            const Cd b2 = ov1[z * sec + i];
                            acc = cadd(acc, cmul(cconj(a), b2));
                            sumX += cabs2(a);
                            sumY += cabs2(b2);
                        }
                        const Cd t = cmul(acc, ramp);
                        xyRe += t.re;
                        xyIm += t.im;
                    }
                    partials[static_cast<std::size_t>(b)] = {xyRe, xyIm, sumX, sumY};
                }

                ModampSums s;
                double xyRe = 0, xyIm = 0, sumX = 0, sumY = 0;
                for (const Partial& p : partials) {
                    xyRe += p.xyRe;
                    xyIm += p.xyIm;
                    sumX += p.sumX;
                    sumY += p.sumY;
                }
                s.xy = cd(xyRe, xyIm);
                s.sumX = sumX;
                s.sumY = sumY;
                return s;
            }

            void filterBands(const FilterCtx& c, Cd* bands,
                             const int* hostZd, const Cd* hostConjamp) override {
                const IndexT bandElems = c.nz * c.ny * c.nxh;
                for (int order = 0; order < c.norders; ++order) {
                    const int zdo = hostZd[order];
                    const IndexT nzc = 2 * static_cast<IndexT>(zdo) + 1;
                    const Cd conjamp = hostConjamp[order];
                    Cd* bre = bands + (order == 0 ? 0 : 2 * order - 1) * bandElems;
                    Cd* bim = order == 0 ? nullptr : bands + 2 * order * bandElems;

                    // pass 1: stored (kx >= 0) half, scales the plus component
                    const IndexT rows = nzc * c.ny;
#pragma omp parallel for schedule(static)
                    for (IndexT r = 0; r < rows; ++r) {
                        const IndexT z0 = r / c.ny - zdo;
                        const IndexT y1 = r % c.ny - c.ny / 2;
                        const IndexT zi = signedToStorage(z0, c.nz);
                        const IndexT yi = signedToStorage(y1, c.ny);
                        for (IndexT x1 = 0; x1 <= c.nx / 2; ++x1) {
                            bool inSupport = false;
                            Cd scale = filterScale(c, order, static_cast<double>(x1),
                                                   static_cast<double>(y1),
                                                   static_cast<double>(z0), &inSupport);
                            const IndexT idx = (zi * c.ny + yi) * c.nxh + x1;
                            if (order == 0) {
                                bre[idx] = inSupport ? cmul(bre[idx], scale) : cd(0, 0);
                            } else {
                                filterApplySide(&bre[idx], &bim[idx], cmul(scale, conjamp), inSupport);
                            }
                        }
                    }

                    // pass 2: mirrored (kx < 0) coordinates, scales the minus
                    // component through the values pass 1 wrote
                    if (order != 0) {
#pragma omp parallel for schedule(static)
                        for (IndexT r = 0; r < rows; ++r) {
                            const IndexT z0 = r / c.ny - zdo;
                            const IndexT y1 = r % c.ny - c.ny / 2;
                            const IndexT zi = signedToStorage(-z0, c.nz);
                            const IndexT yi = signedToStorage(-y1, c.ny);
                            for (IndexT x1 = -(c.nx / 2 - 1); x1 < 0; ++x1) {
                                bool inSupport = false;
                                Cd scale = filterScale(c, order, static_cast<double>(x1),
                                                       static_cast<double>(y1),
                                                       static_cast<double>(z0), &inSupport);
                                const IndexT idx = (zi * c.ny + yi) * c.nxh - x1;
                                filterApplySideMirror(&bre[idx], &bim[idx], cmul(scale, conjamp), inSupport);
                            }
                        }
                    }

                    // zero the |kz| planes beyond the axial support (kernel3)
                    if (c.nz - zdo > zdo + 1) {
                        const IndexT plane = c.ny * c.nxh;
#pragma omp parallel for schedule(static)
                        for (IndexT z = zdo + 1; z < c.nz - zdo; ++z) {
                            for (IndexT i = 0; i < plane; ++i) {
                                bre[z * plane + i] = cd(0, 0);
                                if (bim) bim[z * plane + i] = cd(0, 0);
                            }
                        }
                    }
                }
            }

            void moveBand(const MoveCtx& c, const Cd* bandRe, const Cd* bandIm,
                          Cd* big) override {
                const IndexT rows = c.nz * c.ny;
#pragma omp parallel for schedule(static)
                for (IndexT r = 0; r < rows; ++r) {
                    const IndexT zi = r / c.ny;
                    const IndexT yi = r % c.ny;
                    for (IndexT t = 0; t < c.nx; ++t)
                        moveBandElement(c, bandRe, bandIm, big, zi, yi, t);
                }
            }

            void accumulate(double* out, const Cd* big, int order,
                            double angleX, double angleY,
                            IndexT zdim, IndexT ydim, IndexT xdim) override {
                const IndexT rows = zdim * ydim;
#pragma omp parallel for schedule(static)
                for (IndexT r = 0; r < rows; ++r) {
                    const IndexT iy = r % ydim;
                    double* dst = out + r * xdim;
                    const Cd* src = big + r * xdim;
                    for (IndexT ix = 0; ix < xdim; ++ix) {
                        const double angle = angleX * static_cast<double>(ix - xdim / 2) +
                                             angleY * static_cast<double>(iy - ydim / 2);
                        dst[ix] += accumulateValue(src, ix, order, angle);
                    }
                }
            }

            void synchronize() override {}
        };

    } // namespace

    std::unique_ptr<SimBackend> makeCpuSimBackend() {
        return std::make_unique<CpuSimBackend>();
    }

#ifndef SIRIUS_HAS_CUDA
    std::unique_ptr<SimBackend> makeCudaSimBackend(Device device, const Stream&) {
        throw std::runtime_error("SIRIUS was built without CUDA support; cannot reconstruct on " +
                                 toString(device));
    }
#endif

} // namespace sirius::simdetail
