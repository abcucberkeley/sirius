#ifndef SIRIUS_REGISTRATION_HPP
#define SIRIUS_REGISTRATION_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "sirius/buffer.hpp"
#include "sirius/device.hpp"
#include "sirius/fft_common.hpp"

// Masked translation registration in the Fourier domain.
//
// Implements the masked normalized cross-correlation of
//
//   D. Padfield, "Masked Object Registration in the Fourier Domain",
//   IEEE Trans. Image Processing 21(5), 2012
//
// generalized from the paper's 2D derivation to 2D *and* 3D volumes, which is
// what a light-sheet or SIM tile actually is. Both images carry a mask marking
// the voxels that take part in the match (saturated pixels, regions outside the
// specimen, the zero fill left by a deskew, ...); the masking is folded into
// the correlation itself rather than applied afterwards, so the score at every
// candidate displacement is the true normalized correlation of exactly the
// overlapping unmasked voxels.
//
// Everything is computed with 6 forward and 6 inverse real FFTs of the
// zero-padded volume, batched into two planned transforms. The cost is
// therefore O(N log N) in the padded volume no matter how large the search
// range is, which is the reason to prefer this over sliding-window
// correlation.
//
// Layout and conventions:
//  * Views are rank 2 (rows, cols) or rank 3 (depth, rows, cols); a rank-2
//    image is treated as a single-plane volume. Masks are uint8 (0 = ignore,
//    non-zero = use) and either match their image shape or are empty views,
//    which mean "every voxel is valid".
//  * The correlation map has shape fixedShape + movingShape - 1. Index i along
//    an axis encodes the displacement
//        shift = i - (movingExtent - 1),
//    that is, voxel p of `moving` corresponds to voxel p + shift of `fixed`.
//  * All arithmetic is double precision: the algorithm forms differences of
//    large, nearly equal sums and is not usable in single precision.

namespace sirius {

    // nextFastFFTSize(n), the 2/3/5/7-smooth padding size the correlator uses,
    // is declared in sirius/fft_common.hpp (included above).

    struct MaskedNccOptions {
        // Candidate displacements whose masks overlap in fewer than this many
        // voxels are rejected. `requiredOverlapFraction` is relative to the
        // largest overlap present in the map, so it adapts to the tile size;
        // the effective threshold is the larger of the two.
        //
        // Tiny overlaps are the failure mode of masked correlation: two or
        // three voxels correlate perfectly by accident, so the extreme corners
        // of the map -- where the images barely touch -- are covered in
        // coefficients of exactly +-1. The default rejects displacements that
        // use less than a quarter of the best available overlap; set it to 0
        // to search the raw map. When tiles are expected to overlap only
        // slightly, prefer correlating the expected overlap regions (or
        // setting `maxShift`) over lowering this.
        Index requiredOverlapVoxels = 0;
        double requiredOverlapFraction = 0.25;

        // Per-axis bound on |shift - shiftCentre| in voxels; a negative
        // `maxShift` entry means unbounded on that axis. Restricting the
        // search to the displacements a stage can actually produce is both
        // faster and much more robust. `shiftCentre` is the displacement the
        // caller already expects, which is rarely zero once the two images are
        // sub-blocks cut at different offsets.
        std::array<Index, 3> maxShift{-1, -1, -1};
        std::array<Index, 3> shiftCentre{0, 0, 0};

        // Refine the peak with a parabola through its neighbours on each axis.
        bool subpixel = true;

        // FFTW planning rigor. Estimate keeps one-shot calls cheap; Measure
        // pays off when a MaskedCorrelator is reused across many tile pairs.
        PlanRigor rigor = PlanRigor::Estimate;
    };

    struct MaskedNccResult {
        // Correlation coefficients in [-1, 1], shape fixed + moving - 1.
        Buffer<double> correlation;
        // Number of unmasked voxels the two images share at that displacement
        // (whole numbers, kept as double because that is what the transform
        // produces). Zero where the images do not overlap at all.
        Buffer<double> overlap;
        // Extent of the moving image, needed to turn an index into a shift.
        std::array<Index, 3> movingExtent{1, 1, 1};

        // Displacement encoded by a correlation index.
        std::array<Index, 3> shiftAt(std::array<Index, 3> index) const {
            return {index[0] - (movingExtent[0] - 1),
                    index[1] - (movingExtent[1] - 1),
                    index[2] - (movingExtent[2] - 1)};
        }
    };

    struct TranslationResult {
        // Displacement (dz, dy, dx) of `moving` relative to `fixed`: voxel p of
        // the moving image matches voxel p + shift of the fixed image.
        std::array<double, 3> shift{0, 0, 0};
        std::array<Index, 3> integerShift{0, 0, 0};
        double correlation = 0.0;   // masked NCC at the peak, in [-1, 1]
        double overlap = 0.0;       // unmasked voxels contributing to it
        bool valid = false;         // false when no displacement passed the filters
    };

    // Reusable correlator: holds the FFT plans and the padded work buffers for
    // one (fixedShape, movingShape) pair. Stitching registers many tile pairs
    // of identical size, and planning plus allocation dominates the cost of a
    // single small correlation, so keep one of these per shape pair.
    class MaskedCorrelator {
    public:
        MaskedCorrelator(Shape fixedShape, Shape movingShape,
                         PlanRigor rigor = PlanRigor::Estimate);
        ~MaskedCorrelator();
        MaskedCorrelator(MaskedCorrelator&&) noexcept;
        MaskedCorrelator& operator=(MaskedCorrelator&&) noexcept;
        MaskedCorrelator(const MaskedCorrelator&) = delete;
        MaskedCorrelator& operator=(const MaskedCorrelator&) = delete;

        std::array<Index, 3> fixedExtent() const noexcept;
        std::array<Index, 3> movingExtent() const noexcept;
        std::array<Index, 3> correlationExtent() const noexcept;   // fixed + moving - 1
        std::array<Index, 3> paddedExtent() const noexcept;        // transform size
        // Working-set size in bytes, so callers can size their tiles.
        std::size_t workingBytes() const noexcept;

        // Fill `correlation` and `overlap` (both of correlationExtent(), host
        // memory) for the given images. Masks may be empty views. Non-const
        // because the padded work buffers are reused, so one correlator serves
        // one thread at a time.
        template <typename T>
        void correlate(BufferView<const T> fixed, BufferView<const T> moving,
                       BufferView<const std::uint8_t> fixedMask,
                       BufferView<const std::uint8_t> movingMask,
                       BufferView<double> correlation, BufferView<double> overlap);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    // One-shot masked normalized cross-correlation.
    template <typename T>
    MaskedNccResult maskedNormalizedCrossCorrelation(BufferView<const T> fixed,
                                                     BufferView<const T> moving,
                                                     BufferView<const std::uint8_t> fixedMask,
                                                     BufferView<const std::uint8_t> movingMask,
                                                     const MaskedNccOptions& options = {});

    // Peak of a correlation map subject to the overlap and range filters.
    TranslationResult peakTranslation(const MaskedNccResult& ncc, const MaskedNccOptions& options = {});

    // Correlate and return the best displacement.
    template <typename T>
    TranslationResult registerTranslationMasked(BufferView<const T> fixed,
                                                BufferView<const T> moving,
                                                BufferView<const std::uint8_t> fixedMask,
                                                BufferView<const std::uint8_t> movingMask,
                                                const MaskedNccOptions& options = {});

} // namespace sirius

#endif // SIRIUS_REGISTRATION_HPP
