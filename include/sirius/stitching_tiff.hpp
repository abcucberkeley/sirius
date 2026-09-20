#ifndef SIRIUS_STITCHING_TIFF_HPP
#define SIRIUS_STITCHING_TIFF_HPP

// Stitching a mosaic whose tiles are TIFF files: the one place where the
// stitcher (sirius/stitching.hpp, which works on memory) meets a file format.

#include <array>
#include <string>
#include <vector>

#include "sirius/buffer.hpp"
#include "sirius/stitching.hpp"
#include "sirius/tiff_io.hpp"

namespace sirius {

    struct StitchTile {
        std::string path;                        // multi-page TIFF, one tile
        std::array<double, 3> position{0, 0, 0}; // nominal origin in voxels
    };

    // Read the tiles, plan the mosaic, fuse it and (when `outputPath` is not
    // empty) write the result as a TIFF. Every tile and the canvas are held in
    // memory at once, so this suits mosaics that fit in RAM; for larger ones
    // drive planStitch/fuseTiles directly over a tile-at-a-time reader.
    template <typename T>
    Buffer<T> stitchTiffTiles(const std::vector<StitchTile>& tiles, const StitchOptions& options,
                              StitchLayout* layout = nullptr, const std::string& outputPath = {},
                              TiffCompression compression = TiffCompression::None);

} // namespace sirius

#endif // SIRIUS_STITCHING_TIFF_HPP
