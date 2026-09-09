// CUDA implementation of the SIM reconstruction stages. Per-voxel arithmetic
// is shared with the CPU backend through sim_math.hpp (__host__ __device__),
// so both backends compute identical values; only the FFT library rounding
// differs between them.
//
// Reductions use fixed-size block partials finished on the host, so results
// are deterministic run to run (no atomics).

#include "sim_internal.hpp"
#include "cuda_check.hpp"

#include <cuda_runtime.h>

#include <vector>

namespace sirius::simdetail {

    namespace {

        constexpr int kBlock = 256;
        constexpr unsigned kMaxGrid = 4096;

        unsigned gridFor(IndexT n) {
            const IndexT blocks = (n + kBlock - 1) / kBlock;
            if (blocks <= 0) return 1;
            return static_cast<unsigned>(blocks < kMaxGrid ? blocks : kMaxGrid);
        }

        __global__ void scaleShiftKernel(double* __restrict__ d, IndexT n, double sub, double mul) {
            for (IndexT i = blockIdx.x * (IndexT)blockDim.x + threadIdx.x; i < n;
                 i += (IndexT)gridDim.x * blockDim.x)
                d[i] = (d[i] - sub) * mul;
        }

        __global__ void reorderFramesKernel(const double* __restrict__ raw,
                                            double* __restrict__ frames,
                                            IndexT ndirs, IndexT nphases, IndexT nz,
                                            IndexT planeElems, bool fastSi) {
            const IndexT total = ndirs * nphases * nz * planeElems;
            for (IndexT t = blockIdx.x * (IndexT)blockDim.x + threadIdx.x; t < total;
                 t += (IndexT)gridDim.x * blockDim.x) {
                const IndexT pixel = t % planeElems;
                const IndexT dst = t / planeElems;
                const IndexT z = dst % nz;
                const IndexT ph = (dst / nz) % nphases;
                const IndexT d = dst / (nz * nphases);
                const IndexT src = fastSi ? (z * ndirs + d) * nphases + ph
                                          : (d * nz + z) * nphases + ph;
                frames[t] = raw[src * planeElems + pixel];
            }
        }

        // one block per plane; sequential per-thread accumulation + tree reduce
        __global__ void planeSumsKernel(const double* __restrict__ data, IndexT planeElems,
                                        double* __restrict__ sums) {
            __shared__ double sm[kBlock];
            const double* plane = data + (IndexT)blockIdx.x * planeElems;
            double acc = 0.0;
            for (IndexT i = threadIdx.x; i < planeElems; i += blockDim.x) acc += plane[i];
            sm[threadIdx.x] = acc;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < static_cast<unsigned>(s)) sm[threadIdx.x] += sm[threadIdx.x + s];
                __syncthreads();
            }
            if (threadIdx.x == 0) sums[blockIdx.x] = sm[0];
        }

        __global__ void scalePlanesKernel(double* __restrict__ data, IndexT planeElems,
                                          const double* __restrict__ factors) {
            double* plane = data + (IndexT)blockIdx.x * planeElems;
            const double f = factors[blockIdx.x];
            for (IndexT i = threadIdx.x; i < planeElems; i += blockDim.x) plane[i] *= f;
        }

        __global__ void edgeApodizeXKernel(double* __restrict__ data, IndexT nsec, IndexT ny,
                                           IndexT nx, int napodize, IndexT napY) {
            const double kHalfPi = 0.5 * kPi;
            for (IndexT t = blockIdx.x * (IndexT)blockDim.x + threadIdx.x; t < nsec * nx;
                 t += (IndexT)gridDim.x * blockDim.x) {
                double* img = data + (t / nx) * ny * nx;
                const IndexT k = t % nx;
                const double diff = (img[(ny - 1) * nx + k] - img[k]) * 0.5;
                for (IndexT l = 0; l < napY; ++l) {
                    const double f = diff * (1.0 - sin((static_cast<double>(l) + 0.5) / napodize * kHalfPi));
                    img[l * nx + k] += f;
                    img[(ny - 1 - l) * nx + k] -= f;
                }
            }
        }

        __global__ void edgeApodizeYKernel(double* __restrict__ data, IndexT nsec, IndexT ny,
                                           IndexT nx, int napodize, IndexT napX) {
            const double kHalfPi = 0.5 * kPi;
            for (IndexT t = blockIdx.x * (IndexT)blockDim.x + threadIdx.x; t < nsec * ny;
                 t += (IndexT)gridDim.x * blockDim.x) {
                double* row = data + (t / ny) * ny * nx + (t % ny) * nx;
                const double diff = (row[nx - 1] - row[0]) * 0.5;
                for (IndexT k = 0; k < napX; ++k) {
                    const double f = diff * (1.0 - sin((static_cast<double>(k) + 0.5) / napodize * kHalfPi));
                    row[k] += f;
                    row[nx - 1 - k] -= f;
                }
            }
        }

        __global__ void cosineApodizeKernel(double* __restrict__ data, IndexT nsec, IndexT ny, IndexT nx) {
            for (IndexT t = blockIdx.x * (IndexT)blockDim.x + threadIdx.x; t < nsec * ny * nx;
                 t += (IndexT)gridDim.x * blockDim.x) {
                const IndexT x = t % nx;
                const IndexT y = (t / nx) % ny;
                data[t] *= sin(kPi * (static_cast<double>(x) + 0.5) / nx) *
                           sin(kPi * (static_cast<double>(y) + 0.5) / ny);
            }
        }

        __global__ void separateKernel(const double* __restrict__ phases, double* __restrict__ bands,
                                       const double* __restrict__ mat, int nphases, int nbands, IndexT n) {
            for (IndexT i = blockIdx.x * (IndexT)blockDim.x + threadIdx.x; i < n;
                 i += (IndexT)gridDim.x * blockDim.x) {
                for (int b = 0; b < nbands; ++b) {
                    double acc = 0.0;
                    for (int p = 0; p < nphases; ++p)
                        acc += mat[b * nphases + p] * phases[p * n + i];
                    bands[b * n + i] = acc;
                }
            }
        }

        __global__ void makeOverlapsKernel(OverlapCtx c,
                                           const Cd* __restrict__ re1, const Cd* __restrict__ im1,
                                           const Cd* __restrict__ re2, const Cd* __restrict__ im2,
                                           Cd* __restrict__ ov0, Cd* __restrict__ ov1, int zc) {
            const IndexT nzc = 2 * (IndexT)zc + 1;
            const IndexT total = nzc * c.ny * c.nx;
            for (IndexT t = blockIdx.x * (IndexT)blockDim.x + threadIdx.x; t < total;
                 t += (IndexT)gridDim.x * blockDim.x) {
                const IndexT ix = t % c.nx;
                const IndexT iy = (t / c.nx) % c.ny;
                const IndexT zs = t / (c.nx * c.ny) - zc;
                const IndexT out = (signedToStorage(zs, c.nz) * c.ny + iy) * c.nx + ix;
                ov0[out] = overlap0Value(c, re1, im1, zs, iy, ix);
                ov1[out] = overlap1Value(c, re2, im2, zs, iy, ix);
            }
        }

        __global__ void crossCorrKernel(const Cd* __restrict__ ov0, const Cd* __restrict__ ov1,
                                        Cd* __restrict__ plane, IndexT nz, IndexT sec) {
            for (IndexT i = blockIdx.x * (IndexT)blockDim.x + threadIdx.x; i < sec;
                 i += (IndexT)gridDim.x * blockDim.x) {
                Cd acc = cd(0, 0);
                for (IndexT z = 0; z < nz; ++z)
                    acc = cadd(acc, cmul(ov0[z * sec + i], cconj(ov1[z * sec + i])));
                plane[i] = acc;
            }
        }

        // partials: (gridDim.x, 4) doubles [xyRe, xyIm, sumX, sumY]
        __global__ void modampKernel(const Cd* __restrict__ ov0, const Cd* __restrict__ ov1,
                                     IndexT nz, IndexT ny, IndexT nx,
                                     double angleX, double angleY, double* __restrict__ partials) {
            __shared__ double sm[4][kBlock];
            const IndexT sec = ny * nx;
            double xyRe = 0, xyIm = 0, sumX = 0, sumY = 0;
            for (IndexT i = blockIdx.x * (IndexT)blockDim.x + threadIdx.x; i < sec;
                 i += (IndexT)gridDim.x * blockDim.x) {
                const IndexT iy = i / nx;
                const IndexT ix = i % nx;
                const double angle = angleX * (static_cast<double>(ix) - 0.5 * static_cast<double>(nx)) +
                                     angleY * (static_cast<double>(iy) - 0.5 * static_cast<double>(ny));
                Cd acc = cd(0, 0);
                for (IndexT z = 0; z < nz; ++z) {
                    const Cd a = ov0[z * sec + i];
                    const Cd b = ov1[z * sec + i];
                    acc = cadd(acc, cmul(cconj(a), b));
                    sumX += cabs2(a);
                    sumY += cabs2(b);
                }
                const Cd t = cmul(acc, cd(cos(angle), sin(angle)));
                xyRe += t.re;
                xyIm += t.im;
            }
            sm[0][threadIdx.x] = xyRe;
            sm[1][threadIdx.x] = xyIm;
            sm[2][threadIdx.x] = sumX;
            sm[3][threadIdx.x] = sumY;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < static_cast<unsigned>(s))
                    for (int j = 0; j < 4; ++j) sm[j][threadIdx.x] += sm[j][threadIdx.x + s];
                __syncthreads();
            }
            if (threadIdx.x == 0)
                for (int j = 0; j < 4; ++j) partials[blockIdx.x * 4 + j] = sm[j][0];
        }

        __global__ void filterPass1Kernel(FilterCtx c, int order, int zdo, Cd conjamp,
                                          Cd* __restrict__ bre, Cd* __restrict__ bim) {
            const IndexT nzc = 2 * (IndexT)zdo + 1;
            const IndexT xspan = c.nx / 2 + 1;
            const IndexT total = nzc * c.ny * xspan;
            for (IndexT t = blockIdx.x * (IndexT)blockDim.x + threadIdx.x; t < total;
                 t += (IndexT)gridDim.x * blockDim.x) {
                const IndexT x1 = t % xspan;
                const IndexT y1 = (t / xspan) % c.ny - c.ny / 2;
                const IndexT z0 = t / (xspan * c.ny) - zdo;
                bool inSupport = false;
                Cd scale = filterScale(c, order, static_cast<double>(x1), static_cast<double>(y1),
                                       static_cast<double>(z0), &inSupport);
                const IndexT idx = (signedToStorage(z0, c.nz) * c.ny + signedToStorage(y1, c.ny)) * c.nxh + x1;
                if (order == 0)
                    bre[idx] = inSupport ? cmul(bre[idx], scale) : cd(0, 0);
                else
                    filterApplySide(&bre[idx], &bim[idx], cmul(scale, conjamp), inSupport);
            }
        }

        __global__ void filterPass2Kernel(FilterCtx c, int order, int zdo, Cd conjamp,
                                          Cd* __restrict__ bre, Cd* __restrict__ bim) {
            const IndexT nzc = 2 * (IndexT)zdo + 1;
            const IndexT xspan = c.nx / 2 - 1;   // x1 in [-(nx/2-1), -1]
            const IndexT total = nzc * c.ny * xspan;
            for (IndexT t = blockIdx.x * (IndexT)blockDim.x + threadIdx.x; t < total;
                 t += (IndexT)gridDim.x * blockDim.x) {
                const IndexT x1 = -(c.nx / 2 - 1) + t % xspan;
                const IndexT y1 = (t / xspan) % c.ny - c.ny / 2;
                const IndexT z0 = t / (xspan * c.ny) - zdo;
                bool inSupport = false;
                Cd scale = filterScale(c, order, static_cast<double>(x1), static_cast<double>(y1),
                                       static_cast<double>(z0), &inSupport);
                const IndexT idx = (signedToStorage(-z0, c.nz) * c.ny + signedToStorage(-y1, c.ny)) * c.nxh - x1;
                filterApplySideMirror(&bre[idx], &bim[idx], cmul(scale, conjamp), inSupport);
            }
        }

        __global__ void zeroPlanesKernel(Cd* __restrict__ bre, Cd* __restrict__ bim,
                                         IndexT zlo, IndexT zhi, IndexT planeElems) {
            const IndexT total = (zhi - zlo) * planeElems;
            for (IndexT t = blockIdx.x * (IndexT)blockDim.x + threadIdx.x; t < total;
                 t += (IndexT)gridDim.x * blockDim.x) {
                const IndexT idx = (zlo + t / planeElems) * planeElems + t % planeElems;
                bre[idx] = cd(0, 0);
                if (bim) bim[idx] = cd(0, 0);
            }
        }

        __global__ void moveBandKernel(MoveCtx c, const Cd* __restrict__ bandRe,
                                       const Cd* __restrict__ bandIm, Cd* __restrict__ big) {
            const IndexT total = c.nz * c.ny * c.nx;
            for (IndexT t = blockIdx.x * (IndexT)blockDim.x + threadIdx.x; t < total;
                 t += (IndexT)gridDim.x * blockDim.x) {
                const IndexT x = t % c.nx;
                const IndexT yi = (t / c.nx) % c.ny;
                const IndexT zi = t / (c.nx * c.ny);
                moveBandElement(c, bandRe, bandIm, big, zi, yi, x);
            }
        }

        __global__ void accumulateKernel(double* __restrict__ out, const Cd* __restrict__ big,
                                         int order, double angleX, double angleY,
                                         IndexT zdim, IndexT ydim, IndexT xdim) {
            const IndexT total = zdim * ydim * xdim;
            for (IndexT t = blockIdx.x * (IndexT)blockDim.x + threadIdx.x; t < total;
                 t += (IndexT)gridDim.x * blockDim.x) {
                const IndexT ix = t % xdim;
                const IndexT iy = (t / xdim) % ydim;
                const double angle = angleX * static_cast<double>(ix - xdim / 2) +
                                     angleY * static_cast<double>(iy - ydim / 2);
                out[t] += accumulateValue(big, t, order, angle);
            }
        }

        class CudaSimBackend final : public SimBackend {
        public:
            CudaSimBackend(Device device, const Stream& stream)
                : device_(device), stream_(stream) {
                requireDevice(device);
            }

            void reorderFrames(const double* raw, double* frames,
                               IndexT ndirs, IndexT nphases, IndexT nz,
                               IndexT planeElems, bool fastSi) override {
                cuda::DeviceGuard g(device_.index);
                const IndexT n = ndirs * nphases * nz * planeElems;
                reorderFramesKernel<<<gridFor(n), kBlock, 0, cuda::handle(stream_)>>>(
                    raw, frames, ndirs, nphases, nz, planeElems, fastSi);
                cuda::check(cudaGetLastError(), "reorderFrames kernel");
            }

            void scaleShift(double* data, IndexT n, double sub, double mul) override {
                cuda::DeviceGuard g(device_.index);
                scaleShiftKernel<<<gridFor(n), kBlock, 0, cuda::handle(stream_)>>>(data, n, sub, mul);
                cuda::check(cudaGetLastError(), "scaleShift kernel");
            }

            void planeSums(const double* data, IndexT nplanes, IndexT planeElems,
                           double* hostSums) override {
                cuda::DeviceGuard g(device_.index);
                double* d = deviceScratch(nplanes);
                planeSumsKernel<<<static_cast<unsigned>(nplanes), kBlock, 0, cuda::handle(stream_)>>>(
                    data, planeElems, d);
                cuda::check(cudaGetLastError(), "planeSums kernel");
                cuda::check(cudaMemcpyAsync(hostSums, d, sizeof(double) * nplanes,
                                            cudaMemcpyDeviceToHost, cuda::handle(stream_)),
                            "planeSums copy");
                stream_.synchronize();
            }

            void scalePlanes(double* data, IndexT nplanes, IndexT planeElems,
                             const double* hostFactors) override {
                cuda::DeviceGuard g(device_.index);
                double* d = deviceScratch(nplanes);
                cuda::check(cudaMemcpyAsync(d, hostFactors, sizeof(double) * nplanes,
                                            cudaMemcpyHostToDevice, cuda::handle(stream_)),
                            "scalePlanes upload");
                scalePlanesKernel<<<static_cast<unsigned>(nplanes), kBlock, 0, cuda::handle(stream_)>>>(
                    data, planeElems, d);
                cuda::check(cudaGetLastError(), "scalePlanes kernel");
            }

            void edgeApodize(double* data, IndexT nsec, IndexT ny, IndexT nx,
                             int napodize) override {
                if (napodize <= 0) return;
                cuda::DeviceGuard g(device_.index);
                const IndexT napY = napodize < ny ? napodize : ny;
                const IndexT napX = napodize < nx ? napodize : nx;
                edgeApodizeXKernel<<<gridFor(nsec * nx), kBlock, 0, cuda::handle(stream_)>>>(
                    data, nsec, ny, nx, napodize, napY);
                cuda::check(cudaGetLastError(), "edgeApodize x kernel");
                edgeApodizeYKernel<<<gridFor(nsec * ny), kBlock, 0, cuda::handle(stream_)>>>(
                    data, nsec, ny, nx, napodize, napX);
                cuda::check(cudaGetLastError(), "edgeApodize y kernel");
            }

            void cosineApodize(double* data, IndexT nsec, IndexT ny, IndexT nx) override {
                cuda::DeviceGuard g(device_.index);
                cosineApodizeKernel<<<gridFor(nsec * ny * nx), kBlock, 0, cuda::handle(stream_)>>>(
                    data, nsec, ny, nx);
                cuda::check(cudaGetLastError(), "cosineApodize kernel");
            }

            void separate(const double* phases, double* bands, const double* hostMat,
                          int nphases, int nbands, IndexT n) override {
                cuda::DeviceGuard g(device_.index);
                const IndexT m = static_cast<IndexT>(nphases) * nbands;
                double* d = deviceScratch(m);
                cuda::check(cudaMemcpyAsync(d, hostMat, sizeof(double) * m,
                                            cudaMemcpyHostToDevice, cuda::handle(stream_)),
                            "separate matrix upload");
                separateKernel<<<gridFor(n), kBlock, 0, cuda::handle(stream_)>>>(
                    phases, bands, d, nphases, nbands, n);
                cuda::check(cudaGetLastError(), "separate kernel");
            }

            void makeOverlaps(const OverlapCtx& ctx,
                              const Cd* bandRe1, const Cd* bandIm1,
                              const Cd* bandRe2, const Cd* bandIm2,
                              Cd* ov0, Cd* ov1, int zdistcutoff) override {
                cuda::DeviceGuard g(device_.index);
                const IndexT total = (2 * static_cast<IndexT>(zdistcutoff) + 1) * ctx.ny * ctx.nx;
                makeOverlapsKernel<<<gridFor(total), kBlock, 0, cuda::handle(stream_)>>>(
                    ctx, bandRe1, bandIm1, bandRe2, bandIm2, ov0, ov1, zdistcutoff);
                cuda::check(cudaGetLastError(), "makeOverlaps kernel");
            }

            void crossCorrelate(const Cd* ov0, const Cd* ov1, Cd* plane,
                                IndexT nz, IndexT ny, IndexT nx) override {
                cuda::DeviceGuard g(device_.index);
                crossCorrKernel<<<gridFor(ny * nx), kBlock, 0, cuda::handle(stream_)>>>(
                    ov0, ov1, plane, nz, ny * nx);
                cuda::check(cudaGetLastError(), "crossCorrelate kernel");
            }

            ModampSums modampReduce(const Cd* ov0, const Cd* ov1,
                                    IndexT nz, IndexT ny, IndexT nx,
                                    double angleX, double angleY) override {
                cuda::DeviceGuard g(device_.index);
                const unsigned grid = gridFor(ny * nx);
                double* d = deviceScratch(static_cast<IndexT>(grid) * 4);
                modampKernel<<<grid, kBlock, 0, cuda::handle(stream_)>>>(
                    ov0, ov1, nz, ny, nx, angleX, angleY, d);
                cuda::check(cudaGetLastError(), "modamp kernel");
                hostScratch_.resize(static_cast<std::size_t>(grid) * 4);
                cuda::check(cudaMemcpyAsync(hostScratch_.data(), d, sizeof(double) * grid * 4,
                                            cudaMemcpyDeviceToHost, cuda::handle(stream_)),
                            "modamp copy");
                stream_.synchronize();
                ModampSums s;
                for (unsigned b = 0; b < grid; ++b) {
                    s.xy.re += hostScratch_[b * 4 + 0];
                    s.xy.im += hostScratch_[b * 4 + 1];
                    s.sumX += hostScratch_[b * 4 + 2];
                    s.sumY += hostScratch_[b * 4 + 3];
                }
                return s;
            }

            void filterBands(const FilterCtx& ctx, Cd* bands,
                             const int* hostZd, const Cd* hostConjamp) override {
                cuda::DeviceGuard g(device_.index);
                const IndexT bandElems = ctx.nz * ctx.ny * ctx.nxh;
                for (int order = 0; order < ctx.norders; ++order) {
                    const int zdo = hostZd[order];
                    const IndexT nzc = 2 * static_cast<IndexT>(zdo) + 1;
                    Cd* bre = bands + (order == 0 ? 0 : 2 * order - 1) * bandElems;
                    Cd* bim = order == 0 ? nullptr : bands + 2 * order * bandElems;

                    filterPass1Kernel<<<gridFor(nzc * ctx.ny * (ctx.nx / 2 + 1)), kBlock, 0,
                                        cuda::handle(stream_)>>>(ctx, order, zdo, hostConjamp[order], bre, bim);
                    cuda::check(cudaGetLastError(), "filter pass1 kernel");
                    if (order != 0 && ctx.nx / 2 - 1 > 0) {
                        filterPass2Kernel<<<gridFor(nzc * ctx.ny * (ctx.nx / 2 - 1)), kBlock, 0,
                                            cuda::handle(stream_)>>>(ctx, order, zdo, hostConjamp[order], bre, bim);
                        cuda::check(cudaGetLastError(), "filter pass2 kernel");
                    }
                    if (ctx.nz - zdo > zdo + 1) {
                        const IndexT planeElems = ctx.ny * ctx.nxh;
                        zeroPlanesKernel<<<gridFor((ctx.nz - 2 * zdo - 1) * planeElems), kBlock, 0,
                                           cuda::handle(stream_)>>>(bre, bim, zdo + 1, ctx.nz - zdo, planeElems);
                        cuda::check(cudaGetLastError(), "filter zero-planes kernel");
                    }
                }
            }

            void moveBand(const MoveCtx& ctx, const Cd* bandRe, const Cd* bandIm,
                          Cd* big) override {
                cuda::DeviceGuard g(device_.index);
                moveBandKernel<<<gridFor(ctx.nz * ctx.ny * ctx.nx), kBlock, 0, cuda::handle(stream_)>>>(
                    ctx, bandRe, bandIm, big);
                cuda::check(cudaGetLastError(), "moveBand kernel");
            }

            void accumulate(double* out, const Cd* big, int order,
                            double angleX, double angleY,
                            IndexT zdim, IndexT ydim, IndexT xdim) override {
                cuda::DeviceGuard g(device_.index);
                accumulateKernel<<<gridFor(zdim * ydim * xdim), kBlock, 0, cuda::handle(stream_)>>>(
                    out, big, order, angleX, angleY, zdim, ydim, xdim);
                cuda::check(cudaGetLastError(), "accumulate kernel");
            }

            void synchronize() override { stream_.synchronize(); }

            ~CudaSimBackend() override {
                if (scratch_) {
                    cuda::DeviceGuardNoThrow g(device_.index);
                    (void)cudaFree(scratch_);
                }
            }

        private:
            // small per-call staging area (plane sums, reduction partials,
            // separation matrix); grown once, reused for the whole run
            double* deviceScratch(IndexT n) {
                if (n > scratchElems_) {
                    if (scratch_) (void)cudaFree(scratch_);
                    scratch_ = nullptr;
                    cuda::check(cudaMalloc(&scratch_, sizeof(double) * n), "sim scratch alloc");
                    scratchElems_ = n;
                }
                return scratch_;
            }

            Device device_;
            const Stream& stream_;
            double* scratch_ = nullptr;
            IndexT scratchElems_ = 0;
            std::vector<double> hostScratch_;
        };

    } // namespace

    std::unique_ptr<SimBackend> makeCudaSimBackend(Device device, const Stream& stream) {
        return std::make_unique<CudaSimBackend>(device, stream);
    }

} // namespace sirius::simdetail
