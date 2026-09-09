// Device-agnostic driver of the SIM reconstruction pipeline. All host-side
// control flow lives here (frame ordering, bleach factors, the k0 bracket
// search and parabola fits, Wiener-filter parameters); the data-parallel
// stages run through a SimBackend (sim_cpu.cpp / cuda/sim_kernels.cu) and the
// FFT / RealFFT plans, all of which dispatch on the Device.
//
// The numerics are a faithful port of cudasirecon via the pysirecon reference
// (proto/cusimfixed): several quirks (bracket-search initial state, cutoff
// comparisons, the no-kz0 default) are reproduced deliberately -- see
// proto/cusimfixed/ALGORITHM_MATH.md for the mathematical background.

#include "sirius/sim_reconstruction.hpp"

#include "sirius/constants.hpp"
#include "sirius/fft.hpp"
#include "sirius/real_fft.hpp"
#include "sirius/separation.hpp"

#include "sim_cpu_stages.hpp"
#include "sim_internal.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius {

    namespace {

        using simdetail::Cd;
        using simdetail::IndexT;
        using Cplx = std::complex<double>;

        Cd* asCd(Cplx* p) { return reinterpret_cast<Cd*>(p); }
        const Cd* asCd(const Cplx* p) { return reinterpret_cast<const Cd*>(p); }

        // Sub-pixel peak offset of a parabola through three equally spaced
        // samples (port of fitparabola; offsets beyond 1.5 px are rejected).
        double fitParabola(double a1, double a2, double a3) {
            const double slope = 0.5 * (a3 - a1);
            const double curve = (a3 + a1) - 2.0 * a2;
            if (curve == 0.0) return 0.0;
            const double peak = -slope / curve;
            if (peak > 1.5 || peak < -1.5) return 0.0;
            return peak;
        }

        // Vertex x of the parabola through three arbitrary points (port of
        // fitxyparabola).
        double fitXYParabola(double x1, double y1, double x2, double y2, double x3, double y3) {
            if (x1 == x2 || x2 == x3 || x3 == x1) return 0.0;
            const double xbar1 = 0.5 * (x1 + x2);
            const double xbar2 = 0.5 * (x2 + x3);
            const double slope1 = (y2 - y1) / (x2 - x1);
            const double slope2 = (y3 - y2) / (x3 - x2);
            const double curve = (slope2 - slope1) / (xbar2 - xbar1);
            if (curve == 0.0) return 0.0;
            return xbar2 - slope2 / curve;
        }

    } // namespace

    struct SimReconstructor::Impl {
        SIMParameters p;
        OTFRadiallyAveraged otf;
        Device dev;
        PlanRigor rigor;
        Stream stream;
        std::unique_ptr<simdetail::SimBackend> backend;
        SimFit fit;
        bool capture = false;
        SimDiagnostics diag;
        std::function<bool()> cancelled;   // polled between stages; see the header

        int norders = 0;
        int nbands = 0;
        std::vector<double> sepMatrix;   // row-major (nbands, nphases) separation matrix

        // OTF table (device copy for CUDA; the host tensor itself for CPU)
        Buffer<Cplx> otfDev;
        simdetail::OtfTable otfTable{};

        // ---- state bound to the input shape ----
        Index nx = -1, ny = -1, nz = -1, nxh = 0;
        Index xdim = 0, ydim = 0, zdim = 0;
        double dkx = 0, dky = 0, dkz = 0;
        double rdistcutoff = 0;
        int zdistcutoff = 0;         // axial cutoff used by the overlaps
        double lambdaEm = 0;         // emission wavelength in-sample (um)

        std::optional<RealFFT> bandFft;          // batched (nz, ny, nx) r2c
        std::optional<FFT> volFft;               // (nz, ny, nx) c2c
        std::optional<FFT> planeFft;             // (ny, nx) c2c
        std::optional<FFT> bigFft;               // (zdim, ydim, xdim) c2c

        Buffer<double> frames;       // (ndirs*nphases*nz, ny, nx)
        Buffer<double> realBands;    // (nbands*nz, ny, nx) scratch, one direction
        Buffer<Cplx> bands;          // (ndirs*nbands*nz, ny, nxh)
        Buffer<Cplx> ovF0, ovF1;     // overlap spectra (nz, ny, nx)
        Buffer<Cplx> ov0, ov1;       // overlap real-space volumes
        Buffer<Cplx> planeF, plane;  // findk0 cross-correlation (ny, nx)
        Buffer<Cplx> bigF, big;      // assembly grid (zdim, ydim, xdim)
        Buffer<double> k0Dev, amp2Dev;   // FilterCtx tables on a CUDA device
        std::vector<Cplx> hostPlane;

        Impl(SIMParameters params, OTFRadiallyAveraged otfIn, Device device, PlanRigor planRigor)
            : p(std::move(params)), otf(std::move(otfIn)), dev(device), rigor(planRigor),
              stream(device.isCuda() ? Stream(device) : Stream()) {
            p.validate();
            requireDevice(dev);
            backend = dev.isCuda() ? simdetail::makeCudaSimBackend(dev, stream)
                                   : simdetail::makeCpuSimBackend();

            norders = p.norders > 0 ? p.norders : p.nphases / 2 + 1;
            nbands = 2 * norders - 1;
            if (p.nphases < nbands)
                throw std::invalid_argument("SimReconstructor: " + std::to_string(p.nphases) +
                                            " phases cannot separate " + std::to_string(norders) +
                                            " orders");
            if (otf.data().dimension(0) < norders)
                throw std::invalid_argument("SimReconstructor: OTF has " +
                                            std::to_string(otf.data().dimension(0)) +
                                            " orders, need " + std::to_string(norders));

            // depends only on the parameters: flatten once, reuse for every volume
            const Eigen::MatrixXd sepM = separationMatrix(p.nphases, norders);
            sepMatrix.resize(static_cast<std::size_t>(nbands) * p.nphases);
            for (int b = 0; b < nbands; ++b)
                for (int j = 0; j < p.nphases; ++j)
                    sepMatrix[static_cast<std::size_t>(b) * p.nphases + j] = sepM(b, j);

            if (dev.isCuda()) {
                otfDev = toDevice(otf.data(), dev, stream);
                otfTable.data = asCd(otfDev.data());
            } else {
                otfTable.data = asCd(otf.data().data());
            }
            otfTable.nkr = otf.data().dimension(1);
            otfTable.nzotf = otf.data().dimension(2);
            otfTable.dkrotf = otf.dkrotf();
            // kzscale is bound with the shape (it depends on nz)
        }

        // Poll the caller's cancel predicate at a stage boundary. Nothing
        // here touches the numeric path: it runs on the host between stages,
        // so a run that is never cancelled is bit-identical to one with no
        // callback at all. On CUDA the enqueued work is drained first so no
        // kernel outlives the throw.
        void checkCancelled() {
            if (!cancelled || !cancelled()) return;
            backend->synchronize();
            throw std::runtime_error("cancelled");
        }

        Index bandElems() const { return nz * ny * nxh; }
        Index bandsPerDir() const { return static_cast<Index>(nbands) * bandElems(); }

        Cplx* dirBands(int d) { return bands.data() + d * bandsPerDir(); }

        // Pointers to the stored re/im parts of one order's band.
        Cplx* bandRe(int d, int order) {
            return dirBands(d) + (order == 0 ? 0 : 2 * order - 1) * bandElems();
        }
        Cplx* bandIm(int d, int order) {
            return order == 0 ? nullptr : dirBands(d) + 2 * order * bandElems();
        }

        void bindShape(Index nxIn, Index nyIn, Index nzIn) {
            if (nxIn == nx && nyIn == ny && nzIn == nz) return;
            if (nxIn < 4 || nyIn < 4 || nxIn % 2 != 0 || nyIn % 2 != 0)
                throw std::invalid_argument("SimReconstructor: nx and ny must be even and >= 4, got " +
                                            std::to_string(nxIn) + " x " + std::to_string(nyIn));
            nx = nxIn;
            ny = nyIn;
            nz = nzIn;
            nxh = nx / 2 + 1;

            dkx = 1.0 / (static_cast<double>(nx) * p.dx);
            dky = 1.0 / (static_cast<double>(ny) * p.dy);
            dkz = p.dz > 0 ? 1.0 / (static_cast<double>(nz) * p.dz) : otf.dkzotf();
            otfTable.kzscale = dkz / otf.dkzotf();

            lambdaEm = (p.wavelength_nm / p.nimm) / 1000.0;
            rdistcutoff = std::min(p.na * 2.0 / (p.wavelength_nm * 0.001),
                                   1.0 / (2.0 * std::max(p.dx, p.dy)));

            const double alpha = std::asin(p.na / p.nimm);
            zdistcutoff = static_cast<int>(std::ceil(((1.0 - std::cos(alpha)) / lambdaEm) / dkz));
            // The overlap volumes hold 2 * zdistcutoff + 1 signed planes in
            // nz slots: with an even nz, kz = -nz/2 and kz = +nz/2 would map
            // to the same slot and two threads would write it with different
            // values (the reference code has that race). Keep the range
            // strictly inside the volume.
            if (2 * zdistcutoff + 1 > static_cast<int>(nz))
                zdistcutoff = std::max((static_cast<int>(nz) - 1) / 2, 0);

            xdim = static_cast<Index>(std::lround(p.zoomfact * static_cast<double>(nx)));
            ydim = static_cast<Index>(std::lround(p.zoomfact * static_cast<double>(ny)));
            zdim = static_cast<Index>(p.z_zoom) * nz;

            const auto di = [](Index v) { return static_cast<int>(v); };
            bandFft.emplace(std::vector<int>{di(nz), di(ny), di(nx)}, nbands, rigor, dev);
            volFft.emplace(std::vector<int>{di(nz), di(ny), di(nx)}, 1, rigor, dev);
            planeFft.emplace(std::vector<int>{di(ny), di(nx)}, 1, rigor, dev);
            bigFft.emplace(std::vector<int>{di(zdim), di(ydim), di(xdim)}, 1, rigor, dev);

            const Index nsec = static_cast<Index>(p.ndirs) * p.nphases * nz;
            frames = Buffer<double>(Shape{nsec, ny, nx}, dev, HostMemory::Pageable, stream);
            realBands = Buffer<double>(Shape{static_cast<Index>(nbands) * nz, ny, nx}, dev,
                                       HostMemory::Pageable, stream);
            bands = Buffer<Cplx>(Shape{static_cast<Index>(p.ndirs) * nbands * nz, ny, nxh}, dev,
                                 HostMemory::Pageable, stream);
            ovF0 = Buffer<Cplx>(Shape{nz, ny, nx}, dev, HostMemory::Pageable, stream);
            ovF1 = Buffer<Cplx>(Shape{nz, ny, nx}, dev, HostMemory::Pageable, stream);
            ov0 = Buffer<Cplx>(Shape{nz, ny, nx}, dev, HostMemory::Pageable, stream);
            ov1 = Buffer<Cplx>(Shape{nz, ny, nx}, dev, HostMemory::Pageable, stream);
            planeF = Buffer<Cplx>(Shape{ny, nx}, dev, HostMemory::Pageable, stream);
            plane = Buffer<Cplx>(Shape{ny, nx}, dev, HostMemory::Pageable, stream);
            bigF = Buffer<Cplx>(Shape{zdim, ydim, xdim}, dev, HostMemory::Pageable, stream);
            big = Buffer<Cplx>(Shape{zdim, ydim, xdim}, dev, HostMemory::Pageable, stream);
            if (dev.isCuda()) {
                k0Dev = Buffer<double>(Shape{static_cast<Index>(p.ndirs) * 2}, dev,
                                       HostMemory::Pageable, stream);
                amp2Dev = Buffer<double>(Shape{static_cast<Index>(p.ndirs) * norders}, dev,
                                         HostMemory::Pageable, stream);
            }
            hostPlane.resize(static_cast<std::size_t>(ny * nx));
        }

        // ---- overlaps and the modulation-amplitude machinery -------------

        simdetail::OverlapCtx overlapCtx(int order1, int order2, double k0x, double k0y) const {
            simdetail::OverlapCtx c{};
            c.nx = nx;
            c.ny = ny;
            c.nz = nz;
            c.nxh = nxh;
            c.dkx = dkx;
            c.dky = dky;
            c.rdistcutoff = rdistcutoff;
            c.otfcutoff = p.otfcutoff;
            c.order02factor = nz > 1 ? 5.0 : 1.0;
            c.noKz0 = p.no_kz0 ? 1 : 0;
            c.kx = k0x * (order2 - order1);
            c.ky = k0y * (order2 - order1);
            c.order1 = order1;
            c.order2 = order2;
            c.otf = otfTable;
            return c;
        }

        // Rebuild the two whitened overlap volumes for (order1, order2) at
        // the trial vector k0 and bring them to real space.
        void computeOverlaps(int d, int order1, int order2, double k0x, double k0y) {
            detail::memsetBytes(ovF0.data(), dev, 0, ovF0.bytes(), stream);
            detail::memsetBytes(ovF1.data(), dev, 0, ovF1.bytes(), stream);
            const auto c = overlapCtx(order1, order2, k0x, k0y);
            backend->makeOverlaps(c, asCd(bandRe(d, order1)), asCd(bandIm(d, order1)),
                                  asCd(bandRe(d, order2)), asCd(bandIm(d, order2)),
                                  asCd(ovF0.data()), asCd(ovF1.data()), zdistcutoff);
            volFft->ifft(ovF0.data(), ov0.data(), stream);   // unnormalized, cuFFT convention
            volFft->ifft(ovF1.data(), ov1.data(), stream);
        }

        // Complex modulation amplitude relating the cached overlaps under a
        // shift by (order2-order1)*k0 (port of findrealspacemodamp).
        Cplx modampFromOverlaps(int order1, int order2, double k0x, double k0y) {
            const double kx = k0x * (order2 - order1);
            const double ky = k0y * (order2 - order1);
            const double angleX = 2.0 * kPi * kx * p.dx;
            const double angleY = 2.0 * kPi * ky * p.dy;
            const auto s = backend->modampReduce(asCd(ov0.data()), asCd(ov1.data()),
                                                 nz, ny, nx, angleX, angleY);
            return Cplx(s.xy.re, s.xy.im) / s.sumX;
        }

        struct Modamp {
            double amp2;
            Cplx amp;
        };

        // Port of getmodamp: |modamp|^2 for a trial (angle, magnitude).
        Modamp getModamp(int d, double angle, double mag, int order1, int order2,
                         bool redoArrays, bool& overlapsValid) {
            checkCancelled();
            const double k0x = mag * std::cos(angle);
            const double k0y = mag * std::sin(angle);
            if (redoArrays || !overlapsValid) {
                computeOverlaps(d, order1, order2, k0x, k0y);
                overlapsValid = true;
            }
            const Cplx amp = modampFromOverlaps(order1, order2, k0x, k0y);
            return {std::norm(amp), amp};
        }

        // ---- findk0 -------------------------------------------------------

        std::array<double, 2> findK0(int d, const std::array<double, 2>& guess) {
            int fitorder2 = nz > 1 ? 2 : 1;
            if (fitorder2 >= norders) fitorder2 = norders - 1;

            checkCancelled();
            computeOverlaps(d, 0, fitorder2, guess[0], guess[1]);
            backend->crossCorrelate(asCd(ov0.data()), asCd(ov1.data()), asCd(planeF.data()),
                                    nz, ny, nx);
            planeFft->fft(planeF.data(), plane.data(), stream);
            detail::copyBytes(plane.data(), dev, hostPlane.data(), Device::cpu(),
                              plane.bytes(), stream);
            stream.synchronize();

            // peak of |crosscorr|^2 to sub-pixel precision (port of findpeak)
            const Index n = ny * nx;
            Index best = 0;
            double bestVal = std::norm(hostPlane[0]);
            for (Index i = 1; i < n; ++i) {
                const double v = std::norm(hostPlane[static_cast<std::size_t>(i)]);
                if (v > bestVal) {
                    bestVal = v;
                    best = i;
                }
            }
            const Index xc = best % nx;
            const Index yc = best / nx;
            auto inten = [&](Index iy, Index ix) {
                return std::norm(hostPlane[static_cast<std::size_t>(((iy + ny) % ny) * nx + (ix + nx) % nx)]);
            };
            double peakX = fitParabola(inten(yc, xc - 1), inten(yc, xc), inten(yc, xc + 1)) +
                           static_cast<double>(xc);
            double peakY = fitParabola(inten(yc - 1, xc), inten(yc, xc), inten(yc + 1, xc)) +
                           static_cast<double>(yc);

            // unwrap into the Brillouin zone of the guess
            if (guess[0] / dkx < peakX - static_cast<double>(nx) / 2) peakX -= static_cast<double>(nx);
            if (guess[0] / dkx > peakX + static_cast<double>(nx) / 2) peakX += static_cast<double>(nx);
            if (guess[1] / dky < peakY - static_cast<double>(ny) / 2) peakY -= static_cast<double>(ny);
            if (guess[1] / dky > peakY + static_cast<double>(ny) / 2) peakY += static_cast<double>(ny);

            return {peakX * dkx / fitorder2, peakY * dky / fitorder2};
        }

        // ---- fitk0andmodamps ---------------------------------------------

        // Refine k0 by successive 1D bracket searches over the pattern angle
        // and magnitude (maximizing |modamp|^2), then measure the remaining
        // orders' amplitudes. Faithful port including the reference's
        // bracket-state initialization.
        void fitK0AndModamps(int d, std::array<double, 2>& k0, std::vector<Cplx>& amps) {
            int fitorder2 = nz > 1 ? 2 : 1;
            if (fitorder2 >= norders) fitorder2 = norders - 1;
            const int fitorder1 = 0;

            const double k0mag = std::hypot(k0[0], k0[1]);
            const double k0angle = std::atan2(k0[1], k0[0]);
            const double deltaangle = 0.001;
            const double deltamag = 0.1 / (static_cast<double>(std::max(nx, ny)) * std::max(p.dx, p.dy));

            bool overlapsValid = false;

            // --- angle search ---
            double amp2 = getModamp(d, k0angle, k0mag, fitorder1, fitorder2, true, overlapsValid).amp2;
            double x2 = k0angle;
            double angle = k0angle + deltaangle;
            double x3 = angle;
            double amp3 = getModamp(d, angle, k0mag, fitorder1, fitorder2, false, overlapsValid).amp2;
            double amp1 = amp2;
            double x1 = 0.0;
            if (amp3 > amp2) {
                while (amp3 > amp2) {
                    amp1 = amp2;
                    x1 = x2;
                    amp2 = amp3;
                    x2 = x3;
                    angle += deltaangle;
                    x3 = angle;
                    amp3 = getModamp(d, angle, k0mag, fitorder1, fitorder2, false, overlapsValid).amp2;
                }
            } else {
                angle = k0angle;
                std::swap(amp3, amp2);
                std::swap(x3, x2);
                while (amp3 > amp2) {
                    amp1 = amp2;
                    x1 = x2;
                    amp2 = amp3;
                    x2 = x3;
                    angle -= deltaangle;
                    x3 = angle;
                    amp3 = getModamp(d, angle, k0mag, fitorder1, fitorder2, false, overlapsValid).amp2;
                }
            }
            angle = fitXYParabola(x1, amp1, x2, amp2, x3, amp3);

            // --- magnitude search (reuses the overlaps computed above) ---
            x2 = k0mag;
            amp2 = getModamp(d, angle, k0mag, fitorder1, fitorder2, false, overlapsValid).amp2;
            double mag = k0mag + deltamag;
            x3 = mag;
            amp3 = getModamp(d, angle, mag, fitorder1, fitorder2, false, overlapsValid).amp2;
            if (amp3 > amp2) {
                while (amp3 > amp2) {
                    amp1 = amp2;
                    x1 = x2;
                    amp2 = amp3;
                    x2 = x3;
                    mag += deltamag;
                    x3 = mag;
                    amp3 = getModamp(d, angle, mag, fitorder1, fitorder2, false, overlapsValid).amp2;
                }
            } else {
                mag = k0mag;
                std::swap(amp3, amp2);
                std::swap(x3, x2);
                while (amp3 > amp2) {
                    amp1 = amp2;
                    x1 = x2;
                    amp2 = amp3;
                    x2 = x3;
                    mag -= deltamag;
                    x3 = mag;
                    amp3 = getModamp(d, angle, mag, fitorder1, fitorder2, false, overlapsValid).amp2;
                }
            }
            mag = fitXYParabola(x1, amp1, x2, amp2, x3, amp3);

            const Cplx modamp = getModamp(d, angle, mag, fitorder1, fitorder2, false, overlapsValid).amp;

            k0 = {mag * std::cos(angle), mag * std::sin(angle)};
            amps.assign(static_cast<std::size_t>(norders), Cplx(0, 0));
            amps[0] = Cplx(1, 0);
            amps[static_cast<std::size_t>(fitorder2)] = modamp;

            if (nz == 1) {
                // 2D: fit adjacent order pairs (chained to order 0 by the caller)
                for (int order = 2; order < norders; ++order)
                    amps[static_cast<std::size_t>(order)] =
                        getModamp(d, angle, mag, order - 1, order, true, overlapsValid).amp;
            } else {
                for (int order = 1; order < norders; ++order)
                    if (order != fitorder2)
                        amps[static_cast<std::size_t>(order)] =
                            getModamp(d, angle, mag, 0, order, true, overlapsValid).amp;
            }
        }

        // ---- Wiener filter + assembly ------------------------------------

        // Per-order axial cutoffs (port of the non-Bessel zdistcutoff block).
        std::vector<int> zdistCutoffs() const {
            const double alpha = std::asin(p.na / p.nimm);
            const double lambdaexc = 0.88 * lambdaEm;
            std::vector<int> zd(static_cast<std::size_t>(norders), 0);
            zd[0] = static_cast<int>(std::ceil(((1.0 - std::cos(alpha)) / lambdaEm) / dkz));
            zd[static_cast<std::size_t>(norders - 1)] = static_cast<int>(1.3 * zd[0]);
            for (int order = 1; order < norders - 1; ++order)
                zd[static_cast<std::size_t>(order)] =
                    static_cast<int>((1.0 + lambdaEm / lambdaexc) * zd[0]);
            for (auto& z : zd)
                if (z >= static_cast<int>(nz / 2)) z = std::max(static_cast<int>(nz / 2) - 1, 0);
            return zd;
        }

        void filterAndAssemble(Buffer<double>& out) {
            const auto zd = zdistCutoffs();

            // tables the filter reads for every (direction, order) band
            std::vector<double> k0Host(static_cast<std::size_t>(p.ndirs) * 2);
            std::vector<double> amp2Host(static_cast<std::size_t>(p.ndirs) * norders);
            for (int d = 0; d < p.ndirs; ++d) {
                k0Host[2 * static_cast<std::size_t>(d) + 0] = fit.k0[static_cast<std::size_t>(d)][0];
                k0Host[2 * static_cast<std::size_t>(d) + 1] = fit.k0[static_cast<std::size_t>(d)][1];
                for (int o = 0; o < norders; ++o)
                    amp2Host[static_cast<std::size_t>(d) * norders + o] =
                        std::norm(fit.amps[static_cast<std::size_t>(d)][static_cast<std::size_t>(o)]);
            }
            const double* k0Ptr = k0Host.data();
            const double* amp2Ptr = amp2Host.data();
            if (dev.isCuda()) {
                detail::copyBytes(k0Host.data(), Device::cpu(), k0Dev.data(), dev,
                                  k0Host.size() * sizeof(double), stream);
                detail::copyBytes(amp2Host.data(), Device::cpu(), amp2Dev.data(), dev,
                                  amp2Host.size() * sizeof(double), stream);
                k0Ptr = k0Dev.data();
                amp2Ptr = amp2Dev.data();
            }

            simdetail::FilterCtx fc{};
            fc.nx = nx;
            fc.ny = ny;
            fc.nz = nz;
            fc.nxh = nxh;
            fc.ndirs = p.ndirs;
            fc.norders = norders;
            fc.dkx = dkx;
            fc.dky = dky;
            fc.rdistcutoff = rdistcutoff;
            fc.minDkr = std::min(dkx, dky);
            fc.suppRadius = static_cast<double>(p.suppression_radius) * fc.minDkr;
            const double k0mag0 = std::hypot(fit.k0[0][0], fit.k0[0][1]);
            fc.apocutoff = rdistcutoff + k0mag0 * (norders - 1);
            fc.zapocutoff = static_cast<double>(norders > 1 ? zd[1] : zd[0]);
            fc.wiener2 = p.wiener * p.wiener;
            fc.zd0 = zd[0];
            fc.suppressSingularities = p.suppress_singularities ? 1 : 0;
            fc.dampenOrder0 = p.dampen_order0 ? 1 : 0;
            fc.noKz0 = p.no_kz0 ? 1 : 0;
            fc.filterOverlaps = p.filter_overlaps ? 1 : 0;
            fc.apodizeOutput = static_cast<int>(p.apodize_output);
            fc.k0all = k0Ptr;
            fc.ampmag2 = amp2Ptr;
            fc.otf = otfTable;

            simdetail::MoveCtx mc{};
            mc.nx = nx;
            mc.ny = ny;
            mc.nz = nz;
            mc.nxh = nxh;
            mc.xdim = xdim;
            mc.ydim = ydim;
            mc.zdim = zdim;

            const double fact = p.explodefact / 0.5;

            for (int d = 0; d < p.ndirs; ++d) {
                checkCancelled();
                fc.direction = d;
                std::vector<Cplx> conjamp(static_cast<std::size_t>(norders));
                for (int o = 0; o < norders; ++o)
                    conjamp[static_cast<std::size_t>(o)] =
                        std::conj(fit.amps[static_cast<std::size_t>(d)][static_cast<std::size_t>(o)]);
                backend->filterBands(fc, asCd(dirBands(d)), zd.data(),
                                     asCd(conjamp.data()));

                for (int order = 0; order < norders; ++order) {
                    checkCancelled();
                    detail::memsetBytes(bigF.data(), dev, 0, bigF.bytes(), stream);
                    mc.order = order;
                    backend->moveBand(mc, asCd(bandRe(d, order)), asCd(bandIm(d, order)),
                                      asCd(bigF.data()));
                    bigFft->ifft(bigF.data(), big.data(), stream);   // unnormalized

                    double angleX = 0, angleY = 0;
                    if (order != 0) {
                        angleX = fact * kPi * fit.k0[static_cast<std::size_t>(d)][0] * order *
                                 (p.dx / p.zoomfact);
                        angleY = fact * kPi * fit.k0[static_cast<std::size_t>(d)][1] * order *
                                 (p.dy / p.zoomfact);
                    }
                    backend->accumulate(out.data(), asCd(big.data()), order, angleX, angleY,
                                        zdim, ydim, xdim);
                }
            }
        }

        // Host copy of the whole band storage for the diagnostics.
        void captureBands(Buffer<Cplx>& dst) {
            dst = Buffer<Cplx>(bands.shape(), Device::cpu());
            detail::copyBytes(bands.data(), dev, dst.data(), Device::cpu(), bands.bytes(), stream);
            stream.synchronize();
        }

        // ---- pipeline -----------------------------------------------------

        Buffer<double> run(BufferView<const double> raw) {
            if (raw.rank() != 3)
                throw std::invalid_argument("SimReconstructor: raw stack must be (sections, ny, nx), got rank " +
                                            std::to_string(raw.rank()));
            if (raw.device() != dev)
                throw std::invalid_argument("SimReconstructor: raw stack lives on " + toString(raw.device()) +
                                            " but the reconstructor targets " + toString(dev));
            const Index nsec = raw.dim(0);
            const Index perZ = static_cast<Index>(p.ndirs) * p.nphases;
            if (nsec % perZ != 0)
                throw std::invalid_argument("SimReconstructor: " + std::to_string(nsec) +
                                            " sections is not a multiple of ndirs*nphases = " +
                                            std::to_string(perZ));
            bindShape(raw.dim(2), raw.dim(1), nsec / perZ);

            diag = SimDiagnostics{};
            if (capture) {
                diag.captured = true;
                diag.ndirs = p.ndirs;
                diag.nbands = nbands;
                diag.nx = nx;
                diag.ny = ny;
                diag.nz = nz;
                diag.dkx = dkx;
                diag.dky = dky;
                diag.dkz = dkz;
                diag.rdistcutoff = rdistcutoff;
                diag.zdistcutoff = zdistcutoff;
            }

            checkCancelled();

            // 1. frame ordering: sections -> frames[(d*nphases + p)*nz + z]
            const Index sec = ny * nx;
            backend->reorderFrames(raw.data(), frames.data(), p.ndirs, p.nphases,
                                   nz, sec, p.fast_si);

            // 2. background subtraction + global input scale
            const double inscale = 1.0 / (static_cast<double>(nx) * static_cast<double>(ny) *
                                          static_cast<double>(nz) * p.zoomfact * p.zoomfact *
                                          static_cast<double>(p.z_zoom) * p.ndirs);
            backend->scaleShift(frames.data(), frames.size(), p.background, inscale);

            // 3. bleach correction: scale every phase image so its in-plane
            // sum matches direction-0/phase-0's for that z
            const Index nplanes = static_cast<Index>(p.ndirs) * p.nphases * nz;
            if (p.do_rescale) {
                std::vector<double> sums(static_cast<std::size_t>(nplanes));
                std::vector<double> factors(sums.size());
                backend->planeSums(frames.data(), nplanes, sec, sums.data());
                simdetail::cpu::bleachFactors(sums.data(), p.ndirs, p.nphases, nz, p.equalizez,
                                              factors.data());
                backend->scalePlanes(frames.data(), nplanes, sec, factors.data());
            }

            checkCancelled();

            // 4. input apodization
            switch (p.apodize_input) {
                case ApodizationType::None: break;
                case ApodizationType::Cosine:
                    backend->cosineApodize(frames.data(), nplanes, ny, nx);
                    break;
                case ApodizationType::Triangle:
                    backend->edgeApodize(frames.data(), nplanes, ny, nx, p.napodize);
                    break;
            }

            // 5-6. band separation + 3D real FFT, per direction (batched)
            const Index volN = nz * sec;
            for (int d = 0; d < p.ndirs; ++d) {
                checkCancelled();
                backend->separate(frames.data() + static_cast<Index>(d) * p.nphases * volN,
                                  realBands.data(), sepMatrix.data(), p.nphases, nbands, volN);
                bandFft->rfft(realBands.data(), dirBands(d), stream);
            }
            if (capture) captureBands(diag.separated);

            // 7. pattern vector + modulation amplitudes, per direction
            fit.k0.assign(static_cast<std::size_t>(p.ndirs), {0.0, 0.0});
            fit.amps.assign(static_cast<std::size_t>(p.ndirs),
                            std::vector<Cplx>(static_cast<std::size_t>(norders), Cplx(0, 0)));
            double k0magGuess = 1.0 / p.linespacing_um;
            if (nz > 1) k0magGuess /= (p.nphases / 2 + 1) - 1;
            for (int d = 0; d < p.ndirs; ++d) {
                checkCancelled();
                const double angleGuess =
                    (p.k0_angles && p.k0_angles->size() >= static_cast<std::size_t>(p.ndirs))
                        ? (*p.k0_angles)[static_cast<std::size_t>(d)]
                        : p.k0_start_angle + d * kPi / p.ndirs;
                const std::array<double, 2> guess{k0magGuess * std::cos(angleGuess),
                                                  k0magGuess * std::sin(angleGuess)};
                auto k0 = findK0(d, guess);
                fitK0AndModamps(d, k0, fit.amps[static_cast<std::size_t>(d)]);
                if (nz == 1)
                    for (int order = 2; order < norders; ++order)
                        fit.amps[static_cast<std::size_t>(d)][static_cast<std::size_t>(order)] *=
                            fit.amps[static_cast<std::size_t>(d)][static_cast<std::size_t>(order - 1)];
                fit.k0[static_cast<std::size_t>(d)] = k0;
            }

            // 8-9. generalized Wiener filter + real-space assembly
            Buffer<double> out(Shape{zdim, ydim, xdim}, dev, HostMemory::Pageable, stream);
            detail::memsetBytes(out.data(), dev, 0, out.bytes(), stream);
            filterAndAssemble(out);
            if (capture) captureBands(diag.filtered);   // the filter runs in place on `bands`

            backend->synchronize();
            return out;
        }
    };

    SimReconstructor::SimReconstructor(SIMParameters params, OTFRadiallyAveraged otf,
                                       Device device, PlanRigor rigor)
        : impl_(std::make_unique<Impl>(std::move(params), std::move(otf), device, rigor)) {}

    SimReconstructor::~SimReconstructor() = default;
    SimReconstructor::SimReconstructor(SimReconstructor&&) noexcept = default;
    SimReconstructor& SimReconstructor::operator=(SimReconstructor&&) noexcept = default;

    Device SimReconstructor::device() const noexcept { return impl_->dev; }
    const SimFit& SimReconstructor::lastFit() const noexcept { return impl_->fit; }

    void SimReconstructor::setCancelCallback(std::function<bool()> cancelled) {
        impl_->cancelled = std::move(cancelled);
    }

    void SimReconstructor::setCaptureDiagnostics(bool on) noexcept { impl_->capture = on; }
    bool SimReconstructor::captureDiagnostics() const noexcept { return impl_->capture; }
    const SimDiagnostics& SimReconstructor::lastDiagnostics() const noexcept { return impl_->diag; }
    SimDiagnostics SimReconstructor::takeDiagnostics() noexcept {
        SimDiagnostics out = std::move(impl_->diag);
        impl_->diag = SimDiagnostics{};
        return out;
    }

    Buffer<double> SimReconstructor::reconstruct(BufferView<const double> raw) {
        return impl_->run(raw);
    }

} // namespace sirius
