// Masked FFT translation registration (Padfield 2012), generalized to 3D.
//
// The whole algorithm is six correlations of the two images and their masks,
// combined into the normalized correlation coefficient of the overlapping
// unmasked voxels. Writing them as products of spectra turns the O(N * S)
// sliding-window search (S candidate shifts) into 12 transforms of the padded
// volume.
//
// Performance notes:
//  * The six forward transforms share a size, and so do the six inverse ones,
//    so each is one batched plan (howmany = 6) over a contiguous buffer rather
//    than six separate calls: one plan lookup, one parallel region, and FFTW
//    gets to interleave the batch.
//  * Both are real transforms. The inputs and every result are real, so a
//    complex-to-complex formulation would carry a redundant conjugate half and
//    cost about twice as much in time and memory.
//  * The padded extent is rounded up to a 2/3/5/7-smooth size. The next prime
//    length can be an order of magnitude slower than a nearby smooth one.
//  * Spectrum products are written out with explicit real/imaginary
//    arithmetic. operator* on std::complex calls __muldc3 (the Smith algorithm
//    with an inf/nan fixup) unless -ffast-math is on, which would dominate this
//    loop; the inputs here are finite by construction.
//  * The combine and peak-search passes only touch the cropped correlation
//    map, not the padded volume.

#include "sirius/registration.hpp"

#include "sirius/real_fft.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius {

    namespace {

        using Cplx = std::complex<double>;

        constexpr int kPlanes = 6;   // fixed, rotMoving, fixedMask, rotMovingMask, fixed^2, rotMoving^2

        // Plane indices, both in the padded real buffer before the forward
        // transform and (for the products) after the inverse transform.
        enum Fwd { kFixed = 0,
                   kRotMoving = 1,
                   kFixedMask = 2,
                   kRotMovingMask = 3,
                   kFixedSq = 4,
                   kRotMovingSq = 5 };
        enum Inv { kOverlap = 0,
                   kMaskCorrFixed = 1,
                   kMaskCorrMoving = 2,
                   kCorr = 3,
                   kFixedDenom = 4,
                   kMovingDenom = 5 };

        // The two enums index the same storage; comparing them directly is
        // an -Wenum-compare warning, so spell the intent out.
        constexpr bool sameSlot(Inv i, Fwd f) noexcept { return static_cast<int>(i) == static_cast<int>(f); }

        std::array<Index, 3> extentOf(const Shape& s, const char* what) {
            switch (s.rank()) {
                case 2: return {1, s[0], s[1]};
                case 3: return {s[0], s[1], s[2]};
                default:
                    throw std::invalid_argument(std::string(what) + ": expected a rank-2 (rows, cols) or "
                                                                    "rank-3 (depth, rows, cols) view, got rank " +
                                                std::to_string(s.rank()));
            }
        }

        void requireHost(Device d, const char* what) {
            if (!d.isCpu())
                throw std::invalid_argument(std::string(what) + ": masked registration runs on the host, "
                                                                "but the view lives on " +
                                            toString(d));
        }

        Index product(const std::array<Index, 3>& e) { return e[0] * e[1] * e[2]; }

        // Zero-pad `src` (extent e) into the padded grid at the origin, writing
        // the masked image, the mask itself and the masked squares in one pass
        // over the input. `reverse` produces the 180-degree rotation the
        // correlation needs for the moving image.
        template <typename T, bool Reverse>
        void packPlanes(const T* src, const std::uint8_t* mask, const std::array<Index, 3>& e,
                        const std::array<Index, 3>& pad, double* img, double* msk, double* sq) {
            const Index rows = e[0] * e[1];
#pragma omp parallel for schedule(static)
            for (Index r = 0; r < rows; ++r) {
                const Index z = r / e[1];
                const Index y = r % e[1];
                const Index in = (z * e[1] + y) * e[2];
                const T* s = src + in;
                const std::uint8_t* m = mask ? mask + in : nullptr;
                const Index oz = Reverse ? e[0] - 1 - z : z;
                const Index oy = Reverse ? e[1] - 1 - y : y;
                const Index out = (oz * pad[1] + oy) * pad[2];
                for (Index x = 0; x < e[2]; ++x) {
                    const double w = (m == nullptr || m[x] != 0) ? 1.0 : 0.0;
                    const double v = static_cast<double>(s[x]) * w;
                    const Index ox = out + (Reverse ? e[2] - 1 - x : x);
                    img[ox] = v;
                    msk[ox] = w;
                    sq[ox] = v * v;
                }
            }
        }

        // The six spectrum products of the paper's equations (see the MATLAB
        // reference normxcorr2_masked): every inverse transform below is a
        // correlation of one image or mask with one image or mask.
        //
        // In place. Each output element depends only on the six spectra at the
        // same index, so loading all six into registers before storing lets the
        // products overwrite their own inputs and saves a second buffer of
        // 6 * cplxN complex samples -- a third of the working set.
        void spectrumProducts(Cplx* spec, Index n) {
            Cplx* s0 = spec + kFixed * n;
            Cplx* s1 = spec + kRotMoving * n;
            Cplx* s2 = spec + kFixedMask * n;
            Cplx* s3 = spec + kRotMovingMask * n;
            Cplx* s4 = spec + kFixedSq * n;
            Cplx* s5 = spec + kRotMovingSq * n;

            auto mul = [](const Cplx& a, const Cplx& b) {
                const double ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
                return Cplx(ar * br - ai * bi, ar * bi + ai * br);
            };

            static_assert(sameSlot(kOverlap, kFixed) && sameSlot(kMaskCorrFixed, kRotMoving) &&
                              sameSlot(kMaskCorrMoving, kFixedMask) && sameSlot(kCorr, kRotMovingMask) &&
                              sameSlot(kFixedDenom, kFixedSq) && sameSlot(kMovingDenom, kRotMovingSq),
                          "the in-place product below assumes this slot mapping");

#pragma omp parallel for schedule(static)
            for (Index i = 0; i < n; ++i) {
                const Cplx a0 = s0[i], a1 = s1[i], a2 = s2[i], a3 = s3[i], a4 = s4[i], a5 = s5[i];
                s0[i] = mul(a3, a2);          // kOverlap:        overlapping unmasked voxels
                s1[i] = mul(a3, a0);          // kMaskCorrFixed:  fixed under the moving mask
                s2[i] = mul(a2, a1);          // kMaskCorrMoving: moving under the fixed mask
                s3[i] = mul(a1, a0);          // kCorr:           raw cross-correlation
                s4[i] = mul(a3, a4);          // kFixedDenom:     fixed^2 under the moving mask
                s5[i] = mul(a2, a5);          // kMovingDenom:    moving^2 under the fixed mask
            }
        }

        double parabolaOffset(double left, double centre, double right) {
            const double curve = left - 2.0 * centre + right;
            if (curve == 0.0) return 0.0;
            const double offset = 0.5 * (left - right) / curve;
            // A vertex further than a voxel away means the samples do not
            // describe a peak; trust the integer maximum instead.
            return (offset > 1.0 || offset < -1.0) ? 0.0 : offset;
        }

    } // namespace

    Index nextFastFFTSize(Index n) {
        if (n <= 1) return 1;
        static constexpr Index kRadices[] = {2, 3, 5, 7};
        for (;; ++n) {
            Index m = n;
            for (Index f : kRadices)
                while (m % f == 0) m /= f;
            if (m == 1) return n;
        }
    }

    // --- MaskedCorrelator ---------------------------------------------------

    struct MaskedCorrelator::Impl {
        std::array<Index, 3> fixedExt{}, movingExt{}, corrExt{}, padExt{};
        Index realN = 0;      // voxels per padded transform
        Index cplxN = 0;      // half-complex samples per padded transform
        RealFFT fft;
        Buffer<double> real;     // kPlanes * realN: packed inputs, then the inverse results
        Buffer<Cplx> spec;       // kPlanes * cplxN: forward spectra, then the six products
        Buffer<double> denom;    // cropped denominator scratch
        std::vector<double> rowMax;   // per-row denominator maxima

        static std::vector<int> planDims(const std::array<Index, 3>& pad) {
            std::vector<int> dims;
            // Drop leading singleton axes so a 2D correlation plans a 2D
            // transform instead of a depth-1 3D one.
            const int first = pad[0] > 1 ? 0 : (pad[1] > 1 ? 1 : 2);
            for (int a = first; a < 3; ++a) {
                if (pad[a] > std::numeric_limits<int>::max())
                    throw std::invalid_argument("MaskedCorrelator: padded extent " +
                                                std::to_string(pad[a]) + " exceeds the FFT plan limit");
                dims.push_back(static_cast<int>(pad[a]));
            }
            return dims;
        }

        Impl(const std::array<Index, 3>& fe, const std::array<Index, 3>& me, PlanRigor rigor)
            : fixedExt(fe), movingExt(me),
              corrExt{fe[0] + me[0] - 1, fe[1] + me[1] - 1, fe[2] + me[2] - 1},
              padExt{nextFastFFTSize(corrExt[0]), nextFastFFTSize(corrExt[1]),
                     nextFastFFTSize(corrExt[2])},
              realN(product(padExt)),
              cplxN(padExt[0] * padExt[1] * (padExt[2] / 2 + 1)),
              fft(planDims(padExt), kPlanes, rigor, Device::cpu()),
              real(Shape{kPlanes * realN}),
              spec(Shape{kPlanes * cplxN}),
              denom(Shape{product(corrExt)}),
              rowMax(static_cast<std::size_t>(corrExt[0] * corrExt[1])) {
            // FFTW's batched planner takes the per-transform stride as an int,
            // and the batch is kPlanes transforms of realN samples. Fail here
            // with the size rather than silently overflowing inside the plan.
            if (realN > static_cast<Index>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("MaskedCorrelator: padded transform of " +
                                            std::to_string(realN) +
                                            " voxels exceeds what the FFT planner can address; "
                                            "correlate smaller sub-volumes");
        }

        // Turn the six inverse-transform planes into the correlation map and
        // the overlap count over the cropped region only.
        void combine(double* correlation, double* overlap) {
            const double* rOverlap = real.data() + kOverlap * realN;
            const double* rFixed = real.data() + kMaskCorrFixed * realN;
            const double* rMoving = real.data() + kMaskCorrMoving * realN;
            const double* rCorr = real.data() + kCorr * realN;
            const double* rFDen = real.data() + kFixedDenom * realN;
            const double* rMDen = real.data() + kMovingDenom * realN;
            double* den = denom.data();

            constexpr double kEps = std::numeric_limits<double>::epsilon();
            const Index rows = corrExt[0] * corrExt[1];

            // Per-row maxima combined serially afterwards. `reduction(max:)`
            // would need OpenMP 3.1, which MSVC does not offer by default, and
            // a per-row array keeps the combination order fixed as well.
#pragma omp parallel for schedule(static)
            for (Index r = 0; r < rows; ++r) {
                const Index z = r / corrExt[1];
                const Index y = r % corrExt[1];
                const Index src = (z * padExt[1] + y) * padExt[2];
                const Index dst = r * corrExt[2];
                rowMax[static_cast<std::size_t>(r)] = 0.0;
                for (Index x = 0; x < corrExt[2]; ++x) {
                    // The transform returns the overlap count with rounding
                    // noise; it is an integer by construction.
                    const double count = std::round(rOverlap[src + x]);
                    const double n = std::max(count, kEps);
                    const double f = rFixed[src + x];
                    const double m = rMoving[src + x];
                    const double fDen = std::max(rFDen[src + x] - f * f / n, 0.0);
                    const double mDen = std::max(rMDen[src + x] - m * m / n, 0.0);
                    const double d = std::sqrt(fDen * mDen);
                    correlation[dst + x] = rCorr[src + x] - f * m / n;   // numerator for now
                    den[dst + x] = d;
                    overlap[dst + x] = count;
                    rowMax[static_cast<std::size_t>(r)] =
                        std::max(rowMax[static_cast<std::size_t>(r)], d);
                }
            }

            double maxDen = 0.0;
            for (double v : rowMax) maxDen = std::max(maxDen, v);

            // Both factors of the denominator are non-negative sums, so it can
            // only be zero, never negative: the sole hazard is dividing by
            // (numerically) nothing, which is what this tolerance guards.
            const double tol = 1000.0 * kEps * maxDen;
            const Index n = product(corrExt);
#pragma omp parallel for schedule(static)
            for (Index i = 0; i < n; ++i) {
                const double d = den[i];
                const double c = d > tol ? correlation[i] / d : 0.0;
                correlation[i] = std::clamp(c, -1.0, 1.0);
            }
        }

        template <typename T>
        void correlate(BufferView<const T> fixed, BufferView<const T> moving,
                       BufferView<const std::uint8_t> fixedMask,
                       BufferView<const std::uint8_t> movingMask,
                       BufferView<double> correlation, BufferView<double> overlap) {
            const auto checkImage = [](BufferView<const T> v, const std::array<Index, 3>& want,
                                       const char* what) {
                requireHost(v.device(), what);
                const auto got = extentOf(v.shape(), what);
                if (got != want)
                    throw std::invalid_argument(std::string(what) + ": expected extent " +
                                                std::to_string(want[0]) + "x" + std::to_string(want[1]) +
                                                "x" + std::to_string(want[2]) + ", got " +
                                                std::to_string(got[0]) + "x" + std::to_string(got[1]) +
                                                "x" + std::to_string(got[2]));
            };
            const auto checkMask = [](BufferView<const std::uint8_t> v, const std::array<Index, 3>& want,
                                      const char* what) -> const std::uint8_t* {
                if (v.empty()) return nullptr;   // no mask: every voxel is valid
                requireHost(v.device(), what);
                const auto got = extentOf(v.shape(), what);
                if (got != want)
                    throw std::invalid_argument(std::string(what) + ": mask extent does not match its image");
                return v.data();
            };

            checkImage(fixed, fixedExt, "MaskedCorrelator::correlate (fixed)");
            checkImage(moving, movingExt, "MaskedCorrelator::correlate (moving)");
            const std::uint8_t* fMask = checkMask(fixedMask, fixedExt,
                                                  "MaskedCorrelator::correlate (fixed mask)");
            const std::uint8_t* mMask = checkMask(movingMask, movingExt,
                                                  "MaskedCorrelator::correlate (moving mask)");
            for (auto* out : {&correlation, &overlap}) {
                requireHost(out->device(), "MaskedCorrelator::correlate (output)");
                if (out->size() != product(corrExt))
                    throw std::invalid_argument("MaskedCorrelator::correlate: output must hold " +
                                                std::to_string(product(corrExt)) + " elements, got " +
                                                std::to_string(out->size()));
            }

            // Zero the padding once; the packing below only writes the image
            // extents into the padded grid.
            std::memset(real.data(), 0, real.bytes());
            double* r = real.data();
            packPlanes<T, false>(fixed.data(), fMask, fixedExt, padExt,
                                 r + kFixed * realN, r + kFixedMask * realN, r + kFixedSq * realN);
            packPlanes<T, true>(moving.data(), mMask, movingExt, padExt,
                                r + kRotMoving * realN, r + kRotMovingMask * realN,
                                r + kRotMovingSq * realN);

            fft.rfft(real.data(), spec.data());
            spectrumProducts(spec.data(), cplxN);
            fft.irfft(spec.data(), real.data(), /*normalize=*/true);

            combine(correlation.data(), overlap.data());
        }
    };

    MaskedCorrelator::MaskedCorrelator(Shape fixedShape, Shape movingShape, PlanRigor rigor)
        : impl_(std::make_unique<Impl>(extentOf(fixedShape, "MaskedCorrelator (fixed)"),
                                       extentOf(movingShape, "MaskedCorrelator (moving)"), rigor)) {}

    MaskedCorrelator::~MaskedCorrelator() = default;
    MaskedCorrelator::MaskedCorrelator(MaskedCorrelator&&) noexcept = default;
    MaskedCorrelator& MaskedCorrelator::operator=(MaskedCorrelator&&) noexcept = default;

    std::array<Index, 3> MaskedCorrelator::fixedExtent() const noexcept { return impl_->fixedExt; }
    std::array<Index, 3> MaskedCorrelator::movingExtent() const noexcept { return impl_->movingExt; }
    std::array<Index, 3> MaskedCorrelator::correlationExtent() const noexcept { return impl_->corrExt; }
    std::array<Index, 3> MaskedCorrelator::paddedExtent() const noexcept { return impl_->padExt; }

    std::size_t MaskedCorrelator::workingBytes() const noexcept {
        // Buffer::bytes() is overflow-checked, but these three were allocated
        // through that same check, so their byte counts are known to fit.
        return impl_->real.bytes() + impl_->spec.bytes() + impl_->denom.bytes();
    }

    template <typename T>
    void MaskedCorrelator::correlate(BufferView<const T> fixed, BufferView<const T> moving,
                                     BufferView<const std::uint8_t> fixedMask,
                                     BufferView<const std::uint8_t> movingMask,
                                     BufferView<double> correlation, BufferView<double> overlap) {
        impl_->correlate<T>(fixed, moving, fixedMask, movingMask, correlation, overlap);
    }

    // --- free functions -----------------------------------------------------

    template <typename T>
    MaskedNccResult maskedNormalizedCrossCorrelation(BufferView<const T> fixed, BufferView<const T> moving,
                                                     BufferView<const std::uint8_t> fixedMask,
                                                     BufferView<const std::uint8_t> movingMask,
                                                     const MaskedNccOptions& options) {
        MaskedCorrelator correlator(fixed.shape(), moving.shape(), options.rigor);
        const auto e = correlator.correlationExtent();
        MaskedNccResult out;
        out.correlation = Buffer<double>(Shape{e[0], e[1], e[2]});
        out.overlap = Buffer<double>(Shape{e[0], e[1], e[2]});
        out.movingExtent = correlator.movingExtent();
        correlator.correlate<T>(fixed, moving, fixedMask, movingMask,
                                out.correlation.view(), out.overlap.view());
        return out;
    }

    TranslationResult peakTranslation(const MaskedNccResult& ncc, const MaskedNccOptions& options) {
        const auto e = extentOf(ncc.correlation.shape(), "peakTranslation");
        if (ncc.overlap.shape() != ncc.correlation.shape())
            throw std::invalid_argument("peakTranslation: correlation and overlap shapes differ");
        const double* corr = ncc.correlation.data();
        const double* over = ncc.overlap.data();
        const Index n = product(e);

        double threshold = static_cast<double>(options.requiredOverlapVoxels);
        if (options.requiredOverlapFraction > 0.0) {
            double maxOverlap = 0.0;
            for (Index i = 0; i < n; ++i) maxOverlap = std::max(maxOverlap, over[i]);
            threshold = std::max(threshold, options.requiredOverlapFraction * maxOverlap);
        }
        // A displacement with no overlapping voxels carries no information,
        // whatever the caller asked for.
        threshold = std::max(threshold, 1.0);

        // Deterministic argmax: fixed-size blocks scanned in parallel, then
        // combined in block order with a strict >, so ties always resolve to
        // the lowest index no matter how the blocks are scheduled.
        constexpr Index kBlock = 4096;
        const Index blocks = (n + kBlock - 1) / kBlock;
        struct Best {
            double value;
            Index index;
        };
        std::vector<Best> best(static_cast<std::size_t>(blocks), Best{0.0, -1});

#pragma omp parallel for schedule(static)
        for (Index b = 0; b < blocks; ++b) {
            Best local{0.0, -1};
            const Index end = std::min((b + 1) * kBlock, n);
            for (Index i = b * kBlock; i < end; ++i) {
                if (over[i] < threshold) continue;
                if (local.index >= 0 && !(corr[i] > local.value)) continue;
                const Index z = i / (e[1] * e[2]);
                const Index rem = i % (e[1] * e[2]);
                const std::array<Index, 3> shift =
                    ncc.shiftAt({z, rem / e[2], rem % e[2]});
                bool inRange = true;
                for (int a = 0; a < 3; ++a)
                    if (options.maxShift[a] >= 0 &&
                        std::abs(shift[a] - options.shiftCentre[a]) > options.maxShift[a])
                        inRange = false;
                if (!inRange) continue;
                local = Best{corr[i], i};
            }
            best[static_cast<std::size_t>(b)] = local;
        }

        Best peak{0.0, -1};
        for (const Best& b : best)
            if (b.index >= 0 && (peak.index < 0 || b.value > peak.value)) peak = b;

        TranslationResult out;
        if (peak.index < 0) return out;   // nothing passed the filters

        const Index z = peak.index / (e[1] * e[2]);
        const Index rem = peak.index % (e[1] * e[2]);
        const std::array<Index, 3> at{z, rem / e[2], rem % e[2]};
        out.integerShift = ncc.shiftAt(at);
        out.correlation = corr[peak.index];
        out.overlap = over[peak.index];
        out.valid = true;

        const Index stride[3] = {e[1] * e[2], e[2], 1};
        for (int a = 0; a < 3; ++a) {
            double offset = 0.0;
            if (options.subpixel && at[a] > 0 && at[a] + 1 < e[a])
                offset = parabolaOffset(corr[peak.index - stride[a]], corr[peak.index],
                                        corr[peak.index + stride[a]]);
            out.shift[a] = static_cast<double>(out.integerShift[a]) + offset;
        }
        return out;
    }

    template <typename T>
    TranslationResult registerTranslationMasked(BufferView<const T> fixed, BufferView<const T> moving,
                                                BufferView<const std::uint8_t> fixedMask,
                                                BufferView<const std::uint8_t> movingMask,
                                                const MaskedNccOptions& options) {
        return peakTranslation(
            maskedNormalizedCrossCorrelation<T>(fixed, moving, fixedMask, movingMask, options), options);
    }

    // Explicit instantiations: the pixel types tiles actually arrive in.
#define SIRIUS_INSTANTIATE_REGISTRATION(T)                                                 \
    template void MaskedCorrelator::correlate<T>(BufferView<const T>, BufferView<const T>, \
                                                 BufferView<const std::uint8_t>,           \
                                                 BufferView<const std::uint8_t>,           \
                                                 BufferView<double>, BufferView<double>);  \
    template MaskedNccResult maskedNormalizedCrossCorrelation<T>(                          \
        BufferView<const T>, BufferView<const T>, BufferView<const std::uint8_t>,          \
        BufferView<const std::uint8_t>, const MaskedNccOptions&);                          \
    template TranslationResult registerTranslationMasked<T>(                               \
        BufferView<const T>, BufferView<const T>, BufferView<const std::uint8_t>,          \
        BufferView<const std::uint8_t>, const MaskedNccOptions&);

    SIRIUS_INSTANTIATE_REGISTRATION(double)
    SIRIUS_INSTANTIATE_REGISTRATION(float)
    SIRIUS_INSTANTIATE_REGISTRATION(std::uint16_t)
    SIRIUS_INSTANTIATE_REGISTRATION(std::uint8_t)
#undef SIRIUS_INSTANTIATE_REGISTRATION

} // namespace sirius
