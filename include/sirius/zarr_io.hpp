#ifndef SIRIUS_ZARR_IO_HPP
#define SIRIUS_ZARR_IO_HPP

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "sirius/buffer.hpp"
#include "sirius/pixel_type.hpp"

// Chunked array stores (zarr v2, zarr v3, N5) through TensorStore. Only
// compiled with SIRIUS_ENABLE_TENSORSTORE; without it every function throws
// std::runtime_error("built without TensorStore") and zarrSupported() is
// false, so callers can offer the feature conditionally.
//
// Conventions: shapes, chunk shapes, origins and axis names are always in C
// order (last axis fastest), the order zarr stores use. N5 lists its
// dimensions the other way round (first axis fastest); the reader and the
// writer reverse them so an N5 dataset written as (t, c, z, y, x) here shows
// up as dimensions [x, y, z, c, t] to BigDataViewer and friends.

namespace sirius {

    bool zarrSupported() noexcept;

    struct ZarrArrayInfo {
        std::string path;                 // store root ("/data/x.zarr") or the array itself
        std::string levelPath;            // dataset opened inside the root ("0"), empty for a bare array
        std::string driver;               // "zarr", "zarr3", "n5"
        std::vector<Index> shape;         // of the opened array, C order
        std::vector<Index> chunks;        // read chunk shape, C order
        PixelType pixelType = PixelType::Float32;
        bool isGroup = false;             // `path` is an OME-NGFF / multiscale group
        std::vector<std::string> axes;    // OME-NGFF axis names when present ("t","c","z","y","x")
        std::vector<std::string> axisTypes;   // "time", "channel", "space"
        std::vector<double> scale;        // OME-NGFF coordinate scale per axis (voxel size) when present
        std::vector<std::string> channelNames;
        std::vector<std::string> channelColors;   // "#rrggbb" from omero metadata when present
        std::vector<std::string> multiscalePaths;   // OME-NGFF pyramid datasets, level 0 first
        std::string codec;                // "blosc(zstd,3)", "gzip(6)", "none", ...
        std::uint64_t bytesOnDisk = 0;

        int rank() const noexcept { return static_cast<int>(shape.size()); }
    };

    // Inspect a zarr / N5 store (an OME-NGFF group resolves to its level-0 array).
    ZarrArrayInfo inspectZarr(const std::string& path);

    // Whether `path` looks like a zarr / N5 store (metadata file present).
    bool isZarrStore(const std::string& path) noexcept;

    // An open array: reads hit TensorStore's chunk cache, so scrubbing
    // through planes of one chunk does not decode the chunk again.
    class ZarrArray {
    public:
        // `levelPath` selects a dataset inside a group ("0"); empty opens the
        // level-0 dataset of a group or the array at `path` itself.
        explicit ZarrArray(const std::string& path, const std::string& levelPath = {});
        ~ZarrArray();
        ZarrArray(ZarrArray&&) noexcept;
        ZarrArray& operator=(ZarrArray&&) noexcept;
        ZarrArray(const ZarrArray&) = delete;
        ZarrArray& operator=(const ZarrArray&) = delete;

        const ZarrArrayInfo& info() const noexcept;

        // Read the hyper-rectangle [origin, origin + shape) into `out`
        // (numel(shape) elements, C order), converting the stored type to T.
        // `shape` entries of 0 extend to the end of that axis.
        template <typename T>
        void read(const std::vector<Index>& origin, const std::vector<Index>& shape, T* out) const;
        // Same, into a new buffer. Buffer shapes stop at rank 4, so a rank-5
        // read comes back as (t * c * z, y, x).
        template <typename T>
        Buffer<T> read(const std::vector<Index>& origin, const std::vector<Index>& shape) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    // One-shot convenience over ZarrArray.
    template <typename T>
    Buffer<T> readZarr(const std::string& path, const std::vector<Index>& origin, const std::vector<Index>& shape,
                       const std::string& levelPath = {});

    struct ZarrWriteOptions {
        int zarrVersion = 3;                            // 2, 3, or 0 for N5
        std::vector<Index> chunks;                      // per axis, C order; empty = TensorStore default
        std::string codec = "blosc-zstd";               // "blosc-zstd", "blosc-lz4", "zstd", "gzip", "none"
        int level = 3;
        bool shard = false;                             // zarr 3: group chunks into shards of shardFactor^n chunks
        int shardFactor = 4;
        std::vector<std::string> axes;                  // OME-NGFF axis names (size == rank), C order
        std::vector<double> scale;                      // voxel size per axis
        std::vector<std::string> channelNames;
        std::vector<std::string> channelColors;         // "#rrggbb" per channel (omero metadata)
        int pyramidLevels = 1;                          // > 1 writes "0", "1", ... with OME-NGFF multiscales
        int downsample = 2;                             // per level, on the y and x axes (and z when downsampleZ)
        bool downsampleZ = false;
        bool omeNgff = true;                            // write a group with multiscales metadata (always when pyramidLevels > 1)
        bool deleteExisting = true;
    };

    // Write a host array (any rank <= 5 given explicitly in `shape`, C order)
    // as a store at `path`; `progress` receives 0..1.
    template <typename T>
    void writeZarr(const std::string& path, const T* data, const std::vector<Index>& shape,
                   const ZarrWriteOptions& options,
                   const std::function<void(double)>& progress = {});

} // namespace sirius

#endif // SIRIUS_ZARR_IO_HPP
