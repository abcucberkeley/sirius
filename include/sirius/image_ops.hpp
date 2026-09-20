#ifndef SIRIUS_IMAGE_OPS_HPP
#define SIRIUS_IMAGE_OPS_HPP

#include <array>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "sirius/index.hpp"

// Generic array operations on host float volumes: axis reductions, affine
// resampling (deskew, rotation, isotropic resampling), crop / pad, intensity
// windows and box down-sampling. Everything is OpenMP-parallel over planes
// and works on a (c, t, z, y, x) array described by an explicit extent (the
// Buffer's Shape stops at rank 4; the workbench keeps its arrays as
// (planes, y, x) buffers and passes the five extents alongside).

namespace sirius {

    using Extent5 = std::array<Index, 5>;   // c, t, z, y, x

    inline Index numel(const Extent5& e) noexcept { return e[0] * e[1] * e[2] * e[3] * e[4]; }

    enum class ReduceOp : std::uint8_t { Sum,
                                         Mean,
                                         Max,
                                         Min };

    // Reduce `in` over every axis whose `reduce` flag is set (the output keeps
    // those axes with length 1). `out` must hold numel(outExtent) floats where
    // outExtent[i] = reduce[i] ? 1 : in[i]. Max/Min ignore NaN.
    void reduceAxes(const float* in, const Extent5& extent, const std::array<bool, 5>& reduce, ReduceOp op,
                    float* out);
    Extent5 reducedExtent(const Extent5& extent, const std::array<bool, 5>& reduce) noexcept;

    enum class Interpolation : std::uint8_t { Nearest,
                                              Linear,
                                              Cubic };

    // Affine resampling of a (iz, iy, ix) volume onto a (oz, oy, ox) grid:
    // for every output voxel o = (z, y, x) the input coordinate is
    //     p = A * o + b     (A row-major 3x3, coordinates in voxels, z first)
    // sampled with `interp`; samples outside the input read `fill`.
    void resampleAffine(const float* in, Index iz, Index iy, Index ix, const std::array<double, 9>& A,
                        const std::array<double, 3>& b, float* out, Index oz, Index oy, Index ox,
                        Interpolation interp, float fill = 0.0f);

    struct ResampleGeometry {
        Index oz = 0, oy = 0, ox = 0;           // output extent
        std::array<double, 9> A{1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::array<double, 3> b{0, 0, 0};
        std::array<double, 3> outVoxelUm{0, 0, 0};   // z, y, x of the output grid
    };

    // Light-sheet deskew: the stage moves the sample by `stageStepUm` between
    // planes at `angleDeg` to the detection axis, so plane k is shifted by
    // k * stageStep * cos(angle) / dx pixels along x. The output grid holds
    // the sheared stack (fill outside); `rotateToCoverslip` additionally
    // rotates so the output z axis is normal to the coverslip and resamples
    // to an isotropic (dx) grid.
    ResampleGeometry deskewGeometry(Index iz, Index iy, Index ix, double dxUm, double dzUm, double angleDeg,
                                    double stageStepUm, bool rotateToCoverslip);

    // Resample (iz, iy, ix) with voxel (dz, dy, dx) um onto a grid of voxel
    // (tz, ty, tx) um (0 keeps that axis).
    ResampleGeometry resampleGeometry(Index iz, Index iy, Index ix, double dzUm, double dyUm, double dxUm,
                                      double tzUm, double tyUm, double txUm);

    // Copy the (oz, oy, ox) box starting at (z0, y0, x0) of the input
    // (offsets may be negative or exceed the input: those voxels read `fill`).
    void cropPad(const float* in, Index iz, Index iy, Index ix, Index z0, Index y0, Index x0, float* out,
                 Index oz, Index oy, Index ox, float fill = 0.0f);

    // Quantiles [lo, hi] in percent of n values (sub-sampled to maxSamples), NaN ignored.
    std::pair<float, float> percentiles(const float* values, Index n, double loPercent, double hiPercent,
                                        Index maxSamples = Index{1} << 22);

    // v = clamp((v - lo) / (hi - lo), 0, 1) ^ (1 / gamma), in place.
    void rescaleGamma(float* values, Index n, float lo, float hi, float gamma);

    // Box down-sampling by integer factors (edges averaged over the partial box).
    void downsampleBox(const float* in, Index iz, Index iy, Index ix, int fz, int fy, int fx, float* out);
    inline Index downsampledExtent(Index n, int f) noexcept { return (n + f - 1) / f; }

    // Histogram of n values into `bins` equal bins over [lo, hi].
    std::vector<double> histogram(const float* values, Index n, int bins, float lo, float hi);

    // Per-plane total-intensity equalization ("bleach correction") of a
    // (frames, y, x) stack: every frame is scaled so its sum matches the
    // first (or the mean when `toMean`).
    void equalizeFrames(float* stack, Index frames, Index planeSize, bool toMean);

    // Flat-field: v = (v - dark) / max(flat - dark, eps) * mean(flat - dark).
    void flatField(float* values, Index planes, Index planeSize, const float* flat, const float* dark /*nullable*/);

} // namespace sirius

#endif // SIRIUS_IMAGE_OPS_HPP
