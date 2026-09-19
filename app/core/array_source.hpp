#ifndef SIRIUS_APP_ARRAY_SOURCE_HPP
#define SIRIUS_APP_ARRAY_SOURCE_HPP

// Lazy access to a dataset on disk. The Load step hands a source, not an
// array, down the pipeline: the viewer reads single planes through it, and
// a step that needs the data materializes only the (c, t) volumes it works
// on. Two backends: multi-page TIFF (libtiff / nvTIFF through TiffFile) and
// zarr / N5 through TensorStore when the build has it.

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/array.hpp"
#include "core/dataset.hpp"

namespace sirius::app {

    using ProgressFn = std::function<void(double fraction, const std::string& message)>;

    // How the pages of a plain multi-page TIFF map onto (c, t, z): the
    // fastest varying axis first, as ImageJ's hyperstack order "CZT"
    // (c fastest, then z, then t) or any permutation.
    struct PageOrder {
        std::string order = "czt";     // letters c, z, t; first = fastest
        Index c = 1, t = 1, z = 0;     // z = 0: derive from the page count

        Index planeOf(Index ci, Index ti, Index zi) const noexcept;
        static PageOrder fromDims(const Dims5& d, const std::string& order = "czt");
    };

    class ArraySource {
    public:
        virtual ~ArraySource() = default;
        virtual const DatasetMeta& meta() const noexcept = 0;
        const Dims5& dims() const noexcept { return meta().dims; }
        // One (y, x) plane into `out` (dims().planeSize() floats).
        virtual void readPlane(Index c, Index t, Index z, float* out) const = 0;
        // One (z, y, x) volume; the default loops over readPlane.
        virtual void readVolume(Index c, Index t, float* out, const ProgressFn& progress = {}) const;
        // Everything.
        virtual std::shared_ptr<Array5> readAll(const ProgressFn& progress = {}) const;
        // True when the whole array is already in memory (readAll is free).
        virtual bool inMemory() const noexcept { return false; }
        // Whether the source can feed planes to the GPU decoder (nvTIFF).
        virtual bool gpuDecodable() const noexcept { return false; }

        // --- tiles (multi-file datasets; single-tile sources keep the defaults)
        virtual Index tileCount() const noexcept { return 1; }
        virtual Index currentTile() const noexcept { return 0; }
        // The tile readPlane / readVolume / readAll serve from now on.
        virtual void selectTile(Index /*tile*/) {}
        // One (z, y, x) volume of any tile; the default only knows the current one.
        virtual void readTileVolume(Index tile, Index c, Index t, float* out, const ProgressFn& progress = {}) const;
    };

    // An in-memory array behaving as a source (step outputs, tests).
    class MemorySource final : public ArraySource {
    public:
        MemorySource(ArrayPtr array, DatasetMeta meta);
        const DatasetMeta& meta() const noexcept override { return meta_; }
        void readPlane(Index c, Index t, Index z, float* out) const override;
        void readVolume(Index c, Index t, float* out, const ProgressFn& progress = {}) const override;
        std::shared_ptr<Array5> readAll(const ProgressFn& progress = {}) const override;
        bool inMemory() const noexcept override { return true; }
        Index currentTile() const noexcept override;
        ArrayPtr array() const noexcept { return array_; }

    private:
        ArrayPtr array_;
        DatasetMeta meta_;
    };

    struct OpenOptions {
        // TIFF without OME / ImageJ metadata: how pages map to (c, t, z).
        std::optional<PageOrder> pageOrder;
        // Override the voxel size / channels the file reports.
        std::optional<std::array<double, 3>> voxelUm;
        std::optional<std::vector<ChannelInfo>> channels;
        std::optional<SimLayout> sim;
        bool readAll = false;               // materialize now ("Full load")
        Index tile = 0;                     // multi-file datasets: the tile to serve
        ProgressFn progress;                // 0..1 while readAll decodes; ignored otherwise
    };

    struct OpenResult {
        std::shared_ptr<ArraySource> source;
        DatasetMeta meta;                   // == source->meta()
        // What the file said about itself, for the Open dialog.
        std::string metadataSummary;        // "OME-TIFF · 2 channels · voxel 0.032 µm"
        bool dimsFromMetadata = false;      // c/t/z came from OME/ImageJ/zarr metadata
    };

    // Probe a path without reading pixels: dims (as far as the metadata goes),
    // dtype, size, channels. Throws std::runtime_error when unreadable.
    DatasetMeta probeDataset(const std::string& path);
    // The meta openDataset(path, options) would give, still without reading
    // pixels (`readAll` is ignored). Throws like openDataset.
    DatasetMeta probeDataset(const std::string& path, const OpenOptions& options);
    OpenResult openDataset(const std::string& path, const OpenOptions& options = {});

    // Formats the build can open, as file-dialog filters and extensions.
    std::vector<std::string> readableExtensions();
    // A folder is a dataset when it holds a manifest (core/manifest.hpp).
    bool isFolderDataset(const std::string& path);
    // A TOML file written by DatasetManifest::save (not necessarily named
    // sirius-dataset.toml, and not necessarily sitting in the TIFF folder).
    bool isDatasetManifestFile(const std::string& path);
    // Folder with a sidecar, or a manifest file itself.
    bool isManifestDataset(const std::string& path);
    bool zarrSupported() noexcept;          // built with TensorStore

    // --- metadata helpers (exposed for tests) --------------------------------
    struct ParsedTiffMetadata {
        bool ome = false, imagej = false;
        Index c = 0, t = 0, z = 0;          // 0 = unknown
        std::string dimensionOrder;         // OME DimensionOrder ("XYCZT")
        std::array<double, 3> voxelUm{0, 0, 0};
        double frameIntervalS = 0.0;
        std::vector<ChannelInfo> channels;
    };
    ParsedTiffMetadata parseTiffDescription(const std::string& description);

} // namespace sirius::app

#endif // SIRIUS_APP_ARRAY_SOURCE_HPP
