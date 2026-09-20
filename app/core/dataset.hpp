#ifndef SIRIUS_APP_DATASET_HPP
#define SIRIUS_APP_DATASET_HPP

// Metadata that travels with every array through the pipeline: what the axes
// mean physically (voxel size, frame interval), how the channels are named
// and coloured, where the data came from and, for raw SIM acquisitions, how
// the z axis is really (direction, phase, z) sections.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <sirius/pixel_type.hpp>

#include "core/array.hpp"

namespace sirius::app {

    struct ChannelInfo {
        std::string label;                      // "α-actinin"
        double wavelengthNm = 0.0;              // emission; 0 = unknown
        std::array<float, 3> color{1.f, 1.f, 1.f};   // display colour, linear 0..1
        std::string exposure;                   // free text, e.g. "8 ms"

        // "488" or the label when there is no wavelength.
        std::string shortName() const;
        std::string hexColor() const;           // "#63e08a"
    };

    // Default display colour for an emission wavelength (the design's channel
    // palette: 405 blue, 488 green, 561 magenta, 640 orange; others interpolated).
    std::array<float, 3> colorForWavelength(double nm) noexcept;
    std::array<float, 3> colorFromHex(const std::string& hex);

    // Raw structured-illumination layout of the z axis: the stack holds
    // ndirs * nphases * nz sections in direction -> z -> phase order (or
    // z -> direction -> phase when fastSi).
    struct SimLayout {
        bool present = false;
        int ndirs = 3;
        int nphases = 5;
        bool fastSi = false;

        Index sectionsPerPlane() const noexcept { return static_cast<Index>(ndirs) * nphases; }
    };

    // One tile of a multi-file dataset (a folder described by a manifest):
    // every tile has the same (c, t, z, y, x) shape and a nominal origin.
    struct TileInfo {
        std::string name;                       // "tile_1_2"
        std::array<double, 3> positionUm{0, 0, 0};   // nominal origin, z, y, x in micrometres
        std::array<Index, 3> gridIndex{0, 0, 0};     // z, row, col when the tiles form a grid
    };

    struct DatasetMeta {
        std::string name;                       // display name (file stem)
        std::string sourcePath;                 // file / directory the Load step reads
        std::string format;                     // "tiff", "ome-tiff", "zarr", "n5", "memory"
        Dims5 dims;
        PixelType sourceType = PixelType::Float32;   // dtype on disk
        std::uint64_t bytesOnDisk = 0;
        std::array<double, 3> voxelUm{0.1, 0.1, 0.2};   // x, y, z
        double frameIntervalS = 0.0;            // 0 = unknown
        std::vector<ChannelInfo> channels;      // size == dims.c (or 3 when rgb)
        std::string acquisition;                // "3D-SIM raw · 15 phases", "Lattice light-sheet"
        SimLayout sim;
        bool rgb = false;                       // the c axis holds display R, G, B
        bool lightSheet = false;                // acquired at an angle: deskew applies
        double sheetAngleDeg = 0.0;
        // Multi-file datasets: the tiles the folder holds and the one the
        // array (dims) describes; every tile has the same dims.
        std::vector<TileInfo> tiles;
        Index tileIndex = 0;
        bool hasTiles() const noexcept { return tiles.size() > 1; }
        // Nominal tile origins in voxels of this dataset (from positionUm / voxelUm).
        std::vector<std::array<double, 3>> tilePositionsPx() const;

        double dx() const noexcept { return voxelUm[0]; }
        double dy() const noexcept { return voxelUm[1]; }
        double dz() const noexcept { return voxelUm[2]; }
        // Channel list resized to the array's c, colours assigned by wavelength.
        void normalizeChannels();
        // "c2 t40 z48 y2048 x2048", "rgb z48 …" when rgb
        std::string shapeString() const;
        // "0.032 × 0.032 × 0.110 µm"
        std::string voxelString() const;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_DATASET_HPP
