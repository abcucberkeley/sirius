#ifndef SIRIUS_STITCHING_HPP
#define SIRIUS_STITCHING_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sirius/buffer.hpp"
#include "sirius/registration.hpp"

// Mosaic stitching of tiles acquired on a stage.
//
// A specimen that does not fit in one field is imaged as several overlapping
// tiles whose nominal origins come from the stage. Those origins are good to
// within a few percent of the field, not to a voxel, so stitching is three
// steps:
//
//   1. pairwise registration -- for every pair of tiles whose nominal boxes
//      overlap, correlate the overlapping sub-volumes and measure the true
//      relative displacement. Masked correlation (registration.hpp) is what
//      makes this work on real tiles: the overlap strip is usually where the
//      illumination falls off and where a deskew leaves zero fill, and those
//      voxels have to be excluded from the score rather than treated as data.
//   2. global optimization -- the pairwise displacements over-determine the
//      tile origins, so solve for the origins that fit all of them at once
//      instead of chaining tiles and accumulating drift.
//   3. fusion -- resample the tiles onto one canvas, blending the seams.
//
// Everything here works on 2D images and 3D volumes alike; a rank-2 view is a
// single-plane volume, and positions are always (z, y, x) in voxels.

namespace sirius {

    // How overlapping tiles are combined on the canvas.
    enum class BlendMode : std::uint8_t {
        Overwrite,   // later tiles replace earlier ones
        Average,     // unweighted mean of every tile covering the voxel
        Feather,     // distance-to-border weighted mean: hides seams
        Maximum      // brightest tile wins; useful for sparse fluorescence
    };

    struct StitchOptions {
        // --- pairwise registration ---
        // Skip pairs whose nominal boxes share less than this fraction of the
        // smaller tile.
        double minOverlapFraction = 0.02;
        // How far each axis of a tile may move from its nominal position, in
        // voxels. This bounds the correlation search, so keep it near the
        // stage's real repeatability.
        std::array<Index, 3> searchRadius{4, 32, 32};
        // Pair matches scoring below this are dropped from the optimization.
        double minCorrelation = 0.3;
        // Ignore voxels at or below `backgroundLevel` when correlating: the
        // zero fill of a deskewed volume, unilluminated borders, dead space.
        bool maskBackground = false;
        double backgroundLevel = 0.0;
        PlanRigor rigor = PlanRigor::Estimate;

        // --- global optimization ---
        // Weight pulling each tile towards its nominal position, in units of
        // the pair weights (which are the pair correlations). It keeps the fit
        // finite when the overlap graph is disconnected and stops a single bad
        // pair from dragging the whole mosaic; keep it well below 1 so it does
        // not bend the measured geometry.
        double nominalWeight = 1e-3;
        // Tile whose nominal position defines the absolute frame: the solution
        // is translated so this tile lands exactly on its nominal origin.
        // Without it the fit floats to the centroid of the nominal positions,
        // which leaves every tile on a fractional offset. Out of range means
        // "keep the least-squares frame".
        std::size_t anchorTile = 0;

        // --- fusion ---
        BlendMode blend = BlendMode::Feather;
        // Width of the feather ramp in voxels; 0 lets each tile ramp over a
        // quarter of its own smallest extent.
        Index featherWidth = 0;
        // Voxels at or below this are treated as "no data" during fusion, so
        // a tile's zero fill never darkens its neighbour.
        bool skipBackground = false;
        double fusionBackgroundLevel = 0.0;
    };

    // One measured relationship between two tiles.
    struct TileMatch {
        std::size_t fixed = 0;
        std::size_t moving = 0;
        // Origin of `moving` minus origin of `fixed`, in voxels, as measured.
        std::array<double, 3> displacement{0, 0, 0};
        // The same quantity from the nominal positions, for comparison.
        std::array<double, 3> nominalDisplacement{0, 0, 0};
        double correlation = 0.0;
        double overlap = 0.0;      // unmasked voxels behind the score
        bool accepted = false;     // survived minCorrelation and had an overlap
    };

    struct StitchLayout {
        // Refined tile origins in the same (z, y, x) voxel frame as the input
        // positions.
        std::vector<std::array<double, 3>> positions;
        std::vector<TileMatch> matches;
        // Canvas the fused volume covers: origin (may be negative) and extent.
        std::array<Index, 3> canvasOrigin{0, 0, 0};
        std::array<Index, 3> canvasExtent{0, 0, 0};
    };

    // Measure the displacement of `moving` relative to `fixed` given their
    // nominal origins. Only the nominally overlapping sub-volumes take part,
    // grown by options.searchRadius on the fixed side, so the cost depends on
    // the overlap rather than on the tile size.
    template <typename T>
    TileMatch registerTilePair(BufferView<const T> fixed, std::array<double, 3> fixedPosition,
                               BufferView<const T> moving, std::array<double, 3> movingPosition,
                               const StitchOptions& options = {});

    // Origins that best explain every accepted match. Minimizes
    //   sum_pairs w_ij |(p_j - p_i) - d_ij|^2 + nominalWeight * sum_i |p_i - n_i|^2
    // with w_ij the pair correlation -- one sparse Cholesky factorization
    // shared by the three axes. The result is then translated so tile
    // `anchor` sits on its nominal origin, which fixes the gauge the
    // regularizer would otherwise leave at the nominal centroid; an
    // out-of-range anchor keeps the least-squares frame.
    std::vector<std::array<double, 3>> optimizeTilePositions(
        const std::vector<std::array<double, 3>>& nominal, const std::vector<TileMatch>& matches,
        double nominalWeight = 1e-3, std::size_t anchor = 0);

    // Canvas covering every tile at `positions` (positions are rounded to the
    // voxel grid, which is also how fuseTiles places them).
    void tileCanvas(const std::vector<Shape>& tileShapes,
                    const std::vector<std::array<double, 3>>& positions,
                    std::array<Index, 3>& origin, std::array<Index, 3>& extent);

    // Blend the tiles onto one volume. `canvasOrigin`/`canvasExtent` come from
    // tileCanvas (or from a caller-chosen crop). Tiles are placed at the
    // rounded position: sub-voxel placement would need resampling, which this
    // does not do.
    template <typename T>
    Buffer<T> fuseTiles(const std::vector<BufferView<const T>>& tiles,
                        const std::vector<std::array<double, 3>>& positions,
                        std::array<Index, 3> canvasOrigin, std::array<Index, 3> canvasExtent,
                        const StitchOptions& options = {});

    // Register every overlapping pair and solve for the tile origins. The
    // tiles stay where the caller put them; nothing is read or written here.
    template <typename T>
    StitchLayout planStitch(const std::vector<BufferView<const T>>& tiles,
                            const std::vector<std::array<double, 3>>& nominalPositions,
                            const StitchOptions& options = {});

    // stitchTiffTiles (read the tiles from TIFF files, stitch, write the mosaic)
    // is sirius/stitching_tiff.hpp: the registration and the fusion here work on
    // memory and need no file format.

} // namespace sirius

#endif // SIRIUS_STITCHING_HPP
