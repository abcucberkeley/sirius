// Mosaic stitching: pairwise masked registration, a global fit of the tile
// origins, and a blended fusion pass.
//
// Performance notes:
//  * Only the nominally overlapping sub-volumes are correlated, grown by the
//    search radius on the fixed side. A pair of 2048^2 tiles overlapping by
//    10% correlates a 2048 x 240 strip, not the whole tile, which is the
//    difference between milliseconds and seconds per pair.
//  * Pairs are independent, so they run in parallel; FFTW's planner is
//    serialized inside the FFT wrapper and executing distinct plans on
//    distinct buffers is thread-safe.
//  * The global fit is one sparse Cholesky factorization of an
//    (ntiles x ntiles) graph Laplacian, reused for the three axes: the
//    structure is identical, only the right-hand side differs.
//  * Fusion streams each tile once into the canvas accumulators, and the
//    feather weights are separable, so they are three 1D ramps rather than a
//    distance transform.

#include "sirius/stitching.hpp"

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace sirius {

    namespace {

        using Ext = std::array<Index, 3>;
        using Pos = std::array<double, 3>;

        Ext extentOf(const Shape& s, const char* what) {
            switch (s.rank()) {
                case 2: return {1, s[0], s[1]};
                case 3: return {s[0], s[1], s[2]};
                default:
                    throw std::invalid_argument(std::string(what) + ": expected a rank-2 or rank-3 view, "
                                                                    "got rank " +
                                                std::to_string(s.rank()));
            }
        }

        Index product(const Ext& e) { return e[0] * e[1] * e[2]; }

        // Copy a sub-block out of a (z, y, x) volume.
        template <typename T>
        Buffer<T> cropBlock(const T* src, const Ext& srcExtent, const Ext& origin, const Ext& extent) {
            Buffer<T> out(Shape{extent[0], extent[1], extent[2]});
            const Index rows = extent[0] * extent[1];
#pragma omp parallel for schedule(static)
            for (Index r = 0; r < rows; ++r) {
                const Index z = r / extent[1];
                const Index y = r % extent[1];
                const T* in = src + ((origin[0] + z) * srcExtent[1] + origin[1] + y) * srcExtent[2] +
                              origin[2];
                std::copy(in, in + extent[2], out.data() + r * extent[2]);
            }
            return out;
        }

        template <typename T>
        Buffer<std::uint8_t> thresholdMask(const Buffer<T>& block, double level) {
            Buffer<std::uint8_t> mask(block.shape());
            const Index n = block.size();
#pragma omp parallel for schedule(static)
            for (Index i = 0; i < n; ++i)
                mask.data()[i] = static_cast<double>(block.data()[i]) > level ? 1 : 0;
            return mask;
        }

        BufferView<const std::uint8_t> maskViewOf(const Buffer<std::uint8_t>& mask, bool enabled) {
            return enabled ? mask.view() : BufferView<const std::uint8_t>{};
        }

        // Separable feather ramp along one axis: 1 in the middle, tapering to
        // ~0 at the border over `width` voxels. A singleton axis (the z of a
        // 2D tile) contributes no taper.
        std::vector<double> featherRamp(Index extent, Index width) {
            std::vector<double> ramp(static_cast<std::size_t>(extent), 1.0);
            if (extent <= 1 || width <= 0) return ramp;
            const double w = static_cast<double>(std::min(width, (extent + 1) / 2));
            for (Index i = 0; i < extent; ++i) {
                const double d = static_cast<double>(std::min(i, extent - 1 - i)) + 0.5;
                ramp[static_cast<std::size_t>(i)] = std::min(1.0, d / w);
            }
            return ramp;
        }

        template <typename T>
        T saturate(double v) {
            if constexpr (std::is_integral_v<T>) {
                constexpr double lo = static_cast<double>(std::numeric_limits<T>::lowest());
                constexpr double hi = static_cast<double>(std::numeric_limits<T>::max());
                return static_cast<T>(std::clamp(std::round(v), lo, hi));
            } else {
                return static_cast<T>(v);
            }
        }

    } // namespace

    // --- pairwise registration ----------------------------------------------

    template <typename T>
    TileMatch registerTilePair(BufferView<const T> fixed, Pos fixedPosition,
                               BufferView<const T> moving, Pos movingPosition,
                               const StitchOptions& options) {
        TileMatch match;
        for (int a = 0; a < 3; ++a)
            match.nominalDisplacement[a] = movingPosition[a] - fixedPosition[a];

        const Ext fe = extentOf(fixed.shape(), "registerTilePair (fixed)");
        const Ext me = extentOf(moving.shape(), "registerTilePair (moving)");

        // Nominal overlap box, on the voxel grid of the fixed tile.
        Ext fixedOrigin{}, movingOrigin{}, overlapExtent{};
        for (int a = 0; a < 3; ++a) {
            const Index fl = static_cast<Index>(std::llround(fixedPosition[a]));
            const Index ml = static_cast<Index>(std::llround(movingPosition[a]));
            const Index lo = std::max(fl, ml);
            const Index hi = std::min(fl + fe[a], ml + me[a]);
            if (hi <= lo) return match;            // the tiles do not meet
            fixedOrigin[a] = lo - fl;
            movingOrigin[a] = lo - ml;
            overlapExtent[a] = hi - lo;
        }
        const Index smaller = std::min(product(fe), product(me));
        if (static_cast<double>(product(overlapExtent)) < options.minOverlapFraction *
                                                              static_cast<double>(smaller))
            return match;

        // Grow the fixed side by the search radius so the true displacement is
        // inside the correlation map, and remember where the nominal alignment
        // sits inside the grown block.
        Ext grownOrigin{}, grownExtent{}, centre{};
        for (int a = 0; a < 3; ++a) {
            const Index r = std::max<Index>(options.searchRadius[a], 0);
            grownOrigin[a] = std::max<Index>(fixedOrigin[a] - r, 0);
            const Index end = std::min(fixedOrigin[a] + overlapExtent[a] + r, fe[a]);
            grownExtent[a] = end - grownOrigin[a];
            centre[a] = fixedOrigin[a] - grownOrigin[a];
        }

        const Buffer<T> fixedBlock = cropBlock<T>(fixed.data(), fe, grownOrigin, grownExtent);
        const Buffer<T> movingBlock = cropBlock<T>(moving.data(), me, movingOrigin, overlapExtent);
        Buffer<std::uint8_t> fixedMask, movingMask;
        if (options.maskBackground) {
            fixedMask = thresholdMask(fixedBlock, options.backgroundLevel);
            movingMask = thresholdMask(movingBlock, options.backgroundLevel);
        }

        MaskedNccOptions ncc;
        ncc.maxShift = options.searchRadius;
        ncc.shiftCentre = centre;
        ncc.rigor = options.rigor;
        // The whole moving block overlaps the grown fixed block at the nominal
        // displacement, so anything much smaller is a spurious corner peak.
        ncc.requiredOverlapFraction = 0.5;

        const TranslationResult t = registerTranslationMasked<T>(
            fixedBlock.view(), movingBlock.view(), maskViewOf(fixedMask, options.maskBackground),
            maskViewOf(movingMask, options.maskBackground), ncc);
        if (!t.valid) return match;

        for (int a = 0; a < 3; ++a)
            match.displacement[a] = static_cast<double>(grownOrigin[a] - movingOrigin[a]) + t.shift[a];
        match.correlation = t.correlation;
        match.overlap = t.overlap;
        match.accepted = t.correlation >= options.minCorrelation;
        return match;
    }

    // --- global fit ----------------------------------------------------------

    std::vector<Pos> optimizeTilePositions(const std::vector<Pos>& nominal,
                                           const std::vector<TileMatch>& matches,
                                           double nominalWeight, std::size_t anchor) {
        const std::size_t tiles = nominal.size();
        if (tiles == 0) return {};
        if (nominalWeight <= 0.0)
            throw std::invalid_argument("optimizeTilePositions: nominalWeight must be positive, "
                                        "otherwise a tile with no accepted match is unconstrained");
        for (const TileMatch& m : matches)
            if (m.accepted && (m.fixed >= tiles || m.moving >= tiles))
                throw std::out_of_range("optimizeTilePositions: match refers to a tile that does not exist");

        // The anchor is substituted out of the system rather than pulled into
        // place afterwards: a tile with no accepted match then stays on its
        // nominal position instead of riding along with the anchor's
        // correction, and the remaining system stays symmetric positive
        // definite, so one Cholesky factorization serves the three axes.
        const bool anchored = anchor < tiles;
        const auto unknowns = static_cast<Eigen::Index>(anchored ? tiles - 1 : tiles);
        std::vector<Pos> out(tiles);
        if (anchored) out[anchor] = nominal[anchor];
        if (unknowns == 0) return out;

        const auto index = [&](std::size_t t) -> Eigen::Index {
            return static_cast<Eigen::Index>(anchored && t > anchor ? t - 1 : t);
        };

        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(tiles + 4 * matches.size());
        Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(unknowns, 3);
        for (std::size_t t = 0; t < tiles; ++t) {
            if (anchored && t == anchor) continue;
            const auto r = static_cast<int>(index(t));
            triplets.emplace_back(r, r, nominalWeight);
            for (int a = 0; a < 3; ++a) rhs(index(t), a) = nominalWeight * nominal[t][a];
        }

        for (const TileMatch& m : matches) {
            if (!m.accepted || m.fixed == m.moving) continue;
            const double w = std::max(m.correlation, 1e-6);
            const bool fixedFree = !anchored || m.fixed != anchor;
            const bool movingFree = !anchored || m.moving != anchor;
            // Residual (p_moving - p_fixed - d); a term in the anchor is a
            // known constant and moves to the right-hand side.
            if (fixedFree) {
                const auto r = static_cast<int>(index(m.fixed));
                triplets.emplace_back(r, r, w);
                for (int a = 0; a < 3; ++a) rhs(r, a) -= w * m.displacement[a];
                if (!movingFree)
                    for (int a = 0; a < 3; ++a) rhs(r, a) += w * nominal[anchor][a];
            }
            if (movingFree) {
                const auto r = static_cast<int>(index(m.moving));
                triplets.emplace_back(r, r, w);
                for (int a = 0; a < 3; ++a) rhs(r, a) += w * m.displacement[a];
                if (!fixedFree)
                    for (int a = 0; a < 3; ++a) rhs(r, a) += w * nominal[anchor][a];
            }
            if (fixedFree && movingFree) {
                const auto i = static_cast<int>(index(m.fixed));
                const auto j = static_cast<int>(index(m.moving));
                triplets.emplace_back(i, j, -w);
                triplets.emplace_back(j, i, -w);
            }
        }

        Eigen::SparseMatrix<double> normalMatrix(unknowns, unknowns);
        normalMatrix.setFromTriplets(triplets.begin(), triplets.end());
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(normalMatrix);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("optimizeTilePositions: the tile graph could not be factorized");
        const Eigen::MatrixXd solution = solver.solve(rhs);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("optimizeTilePositions: the tile graph could not be solved");

        for (std::size_t t = 0; t < tiles; ++t) {
            if (anchored && t == anchor) continue;
            for (int a = 0; a < 3; ++a) out[t][a] = solution(index(t), a);
        }
        return out;
    }

    // --- canvas and fusion ---------------------------------------------------

    void tileCanvas(const std::vector<Shape>& tileShapes, const std::vector<Pos>& positions,
                    Ext& origin, Ext& extent) {
        if (tileShapes.size() != positions.size())
            throw std::invalid_argument("tileCanvas: one position per tile is required");
        if (tileShapes.empty()) {
            origin = {0, 0, 0};
            extent = {0, 0, 0};
            return;
        }

        Ext lo{}, hi{};
        for (int a = 0; a < 3; ++a) {
            lo[a] = std::numeric_limits<Index>::max();
            hi[a] = std::numeric_limits<Index>::lowest();
        }
        for (std::size_t t = 0; t < tileShapes.size(); ++t) {
            const Ext e = extentOf(tileShapes[t], "tileCanvas");
            for (int a = 0; a < 3; ++a) {
                const Index p = static_cast<Index>(std::llround(positions[t][a]));
                lo[a] = std::min(lo[a], p);
                hi[a] = std::max(hi[a], p + e[a]);
            }
        }
        origin = lo;
        for (int a = 0; a < 3; ++a) extent[a] = hi[a] - lo[a];
    }

    template <typename T>
    Buffer<T> fuseTiles(const std::vector<BufferView<const T>>& tiles, const std::vector<Pos>& positions,
                        Ext canvasOrigin, Ext canvasExtent, const StitchOptions& options) {
        if (tiles.size() != positions.size())
            throw std::invalid_argument("fuseTiles: one position per tile is required");
        for (int a = 0; a < 3; ++a)
            if (canvasExtent[a] <= 0)
                throw std::invalid_argument("fuseTiles: canvas extent must be positive on every axis");

        const Index canvasVoxels = product(canvasExtent);
        std::vector<double> value(static_cast<std::size_t>(canvasVoxels), 0.0);
        std::vector<double> weight(static_cast<std::size_t>(canvasVoxels), 0.0);
        const bool maximum = options.blend == BlendMode::Maximum;

        for (std::size_t t = 0; t < tiles.size(); ++t) {
            const Ext e = extentOf(tiles[t].shape(), "fuseTiles");
            Ext place{};
            for (int a = 0; a < 3; ++a)
                place[a] = static_cast<Index>(std::llround(positions[t][a])) - canvasOrigin[a];

            // Intersection of the tile with the canvas, in tile coordinates.
            Ext begin{}, end{};
            bool empty = false;
            for (int a = 0; a < 3; ++a) {
                begin[a] = std::max<Index>(0, -place[a]);
                end[a] = std::min(e[a], canvasExtent[a] - place[a]);
                if (end[a] <= begin[a]) empty = true;
            }
            if (empty) continue;

            Index width = options.featherWidth;
            if (options.blend == BlendMode::Feather && width <= 0) {
                Index smallest = std::numeric_limits<Index>::max();
                for (int a = 0; a < 3; ++a)
                    if (e[a] > 1) smallest = std::min(smallest, e[a]);
                width = smallest == std::numeric_limits<Index>::max() ? 1 : std::max<Index>(1, smallest / 4);
            }
            const std::vector<double> rz = featherRamp(e[0], width);
            const std::vector<double> ry = featherRamp(e[1], width);
            const std::vector<double> rx = featherRamp(e[2], width);
            const bool feather = options.blend == BlendMode::Feather;

            const T* src = tiles[t].data();
            const Index rows = (end[0] - begin[0]) * (end[1] - begin[1]);
#pragma omp parallel for schedule(static)
            for (Index r = 0; r < rows; ++r) {
                const Index z = begin[0] + r / (end[1] - begin[1]);
                const Index y = begin[1] + r % (end[1] - begin[1]);
                const T* in = src + (z * e[1] + y) * e[2];
                const Index outRow = ((z + place[0]) * canvasExtent[1] + y + place[1]) * canvasExtent[2] +
                                     place[2];
                const double planeWeight = feather ? rz[static_cast<std::size_t>(z)] *
                                                         ry[static_cast<std::size_t>(y)]
                                                   : 1.0;
                for (Index x = begin[2]; x < end[2]; ++x) {
                    const double v = static_cast<double>(in[x]);
                    if (options.skipBackground && v <= options.fusionBackgroundLevel) continue;
                    const auto o = static_cast<std::size_t>(outRow + x);
                    if (maximum) {
                        if (weight[o] == 0.0 || v > value[o]) value[o] = v;
                        weight[o] = 1.0;
                        continue;
                    }
                    if (options.blend == BlendMode::Overwrite) {
                        value[o] = v;
                        weight[o] = 1.0;
                        continue;
                    }
                    const double w = feather ? planeWeight * rx[static_cast<std::size_t>(x)] : 1.0;
                    value[o] += w * v;
                    weight[o] += w;
                }
            }
        }

        Buffer<T> out(Shape{canvasExtent[0], canvasExtent[1], canvasExtent[2]});
        const bool normalize = !maximum && options.blend != BlendMode::Overwrite;
#pragma omp parallel for schedule(static)
        for (Index i = 0; i < canvasVoxels; ++i) {
            const auto u = static_cast<std::size_t>(i);
            const double w = weight[u];
            out.data()[i] = w > 0.0 ? saturate<T>(normalize ? value[u] / w : value[u]) : T{};
        }
        return out;
    }

    // --- planning -------------------------------------------------------------

    template <typename T>
    StitchLayout planStitch(const std::vector<BufferView<const T>>& tiles,
                            const std::vector<Pos>& nominalPositions, const StitchOptions& options) {
        if (tiles.size() != nominalPositions.size())
            throw std::invalid_argument("planStitch: one nominal position per tile is required");

        std::vector<std::pair<std::size_t, std::size_t>> candidates;
        for (std::size_t i = 0; i < tiles.size(); ++i)
            for (std::size_t j = i + 1; j < tiles.size(); ++j)
                candidates.emplace_back(i, j);

        std::vector<TileMatch> measured(candidates.size());
        const auto count = static_cast<Index>(candidates.size());
        // Pairs are independent; each one plans and runs its own transforms.
        // An exception must not leave the parallel region, so carry the first
        // one out and rethrow it afterwards.
        std::exception_ptr failure;
#pragma omp parallel for schedule(dynamic)
        for (Index c = 0; c < count; ++c) {
            const std::size_t i = candidates[static_cast<std::size_t>(c)].first;
            const std::size_t j = candidates[static_cast<std::size_t>(c)].second;
            try {
                TileMatch m = registerTilePair<T>(tiles[i], nominalPositions[i], tiles[j],
                                                  nominalPositions[j], options);
                m.fixed = i;
                m.moving = j;
                measured[static_cast<std::size_t>(c)] = m;
            } catch (...) {
#pragma omp critical(sirius_stitch_failure)
                {
                    if (!failure) failure = std::current_exception();
                }
            }
        }
        if (failure) std::rethrow_exception(failure);

        StitchLayout layout;
        // Keep only the pairs that actually met; a mosaic is mostly
        // non-neighbouring tiles and empty matches would only add noise.
        for (const TileMatch& m : measured)
            if (m.overlap > 0.0) layout.matches.push_back(m);

        layout.positions = optimizeTilePositions(nominalPositions, layout.matches,
                                                 options.nominalWeight, options.anchorTile);

        std::vector<Shape> shapes;
        shapes.reserve(tiles.size());
        for (const auto& t : tiles) shapes.push_back(t.shape());
        tileCanvas(shapes, layout.positions, layout.canvasOrigin, layout.canvasExtent);
        return layout;
    }

#define SIRIUS_INSTANTIATE_STITCHING(T)                                                               \
    template TileMatch registerTilePair<T>(BufferView<const T>, Pos, BufferView<const T>, Pos,        \
                                           const StitchOptions&);                                     \
    template Buffer<T> fuseTiles<T>(const std::vector<BufferView<const T>>&, const std::vector<Pos>&, \
                                    Ext, Ext, const StitchOptions&);                                  \
    template StitchLayout planStitch<T>(const std::vector<BufferView<const T>>&,                      \
                                        const std::vector<Pos>&, const StitchOptions&);

    SIRIUS_INSTANTIATE_STITCHING(double)
    SIRIUS_INSTANTIATE_STITCHING(float)
    SIRIUS_INSTANTIATE_STITCHING(std::uint16_t)
    SIRIUS_INSTANTIATE_STITCHING(std::uint8_t)
#undef SIRIUS_INSTANTIATE_STITCHING

} // namespace sirius
