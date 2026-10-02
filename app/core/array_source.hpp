#ifndef SIRIUS_APP_ARRAY_SOURCE_HPP
#define SIRIUS_APP_ARRAY_SOURCE_HPP

// Lazy access to a dataset on disk. The Load step hands a source, not an
// array, down the pipeline: the viewer reads single planes through it, and
// a step that needs the data materializes only the (c, t) volumes it works
// on. Two backends: multi-page TIFF (libtiff / nvTIFF through TiffFile) and
// zarr / N5 through TensorStore when the build has it.

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/array.hpp"
#include "core/dataset.hpp"

namespace sirius {
    struct TiffInfo;
}

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

    // --- what a pane draws, asked of a source over a slow link -----------------
    //
    // A dataset on a cluster (core/remote_source.hpp) is not read plane by
    // plane at full resolution for display: the worker next to the data
    // computes what a pane shows -- the XY plane, the XZ / YZ re-slice, the
    // z maximum projection, a small volume for the 3-D view -- reduced by the
    // factor the pane draws at, and sends that. The display model asks for it
    // here and never waits: a request answers with what has arrived (or the
    // best stand-in) and queues the rest; revision() moves when more arrives.

    struct ViewRequest {
        enum class Kind { XY,      // (y, x) at z = index
                          XZ,      // rows z, columns x at y = index
                          YZ,      // rows y, columns z at x = index
                          MIP,     // the z maximum projection (y, x)
                          Volume   // the (z, y, x) volume reduced to a longest side of maxSide
        };
        Kind kind = Kind::XY;
        Index c = 0, t = 0, index = 0;
        int factor = 1;                          // view pixels per drawn pixel
        int x = 0, y = 0, w = 0, h = 0;          // the region in the view's columns / rows; w = 0: all of it
        int maxSide = 256;
    };

    struct ViewTile {
        int x = 0, y = 0;                        // the region's origin in view pixels
        int w = 0, h = 0;                        // reduced width and height (Volume: x and y)
        int d = 1;                               // Volume: reduced depth
        int factor = 1;
        std::vector<float> data;                 // h * w (Volume: d * h * w)
    };

    class ViewProvider {
    public:
        virtual ~ViewProvider() = default;
        // What has arrived for `request`, never waiting: `exact` when it is the
        // request's own factor and covers its region; otherwise the best
        // stand-in (another factor of the same plane) or null, and a fetch of
        // the real thing is queued.
        virtual std::shared_ptr<const ViewTile> view(const ViewRequest& request, bool& exact) = 0;
        // A display window for (c, t): the robust percentiles (Auto) or the
        // range (full), once the worker has sent them; queued otherwise.
        virtual std::optional<std::pair<float, float>> window(Index c, Index t, bool fullRange) = 0;
        // Moves whenever a view or a window arrives.
        virtual std::uint64_t revision() const noexcept = 0;
        // Fetches are queued or running.
        virtual bool busy() const = 0;
        // The last fetch that failed, for the viewer's notice; "" when none.
        virtual std::string lastError() const = 0;
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
        // A source that is drawn through display-sized views (a cluster
        // dataset); null for every local one, which the viewer reads directly.
        virtual ViewProvider* viewProvider() const noexcept { return nullptr; }
        // A step's output computed and held by SIRIUS's engine on a cluster
        // node (core/remote_source.hpp, NodeOutputSource): the node's cache
        // evicts it by the step's cache policy as this computer's would, so a
        // Recompute eviction here drops the source too, not only an array.
        virtual bool heldByNodeCache() const noexcept { return false; }

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
        // Why a full load that was asked for did not happen (the dataset is
        // then served lazily); empty otherwise. See fullLoadLimitBytes().
        std::string fullLoadSkipped;
        // What the file said about itself, for the Open dialog.
        std::string metadataSummary;        // "OME-TIFF · 2 channels · voxel 0.032 µm"
        bool dimsFromMetadata = false;      // c/t/z came from OME/ImageJ/zarr metadata
    };

    // Datasets on a cluster are named "cluster://<ssh host>/<absolute path>"
    // and opened through the connected worker. The cluster session installs
    // the opener (core/remote_source.hpp); openDataset and probeDataset hand
    // such a path to it, and throw while none is installed.
    using RemoteDatasetOpener = std::function<OpenResult(const std::string& path, const OpenOptions& options, bool probeOnly)>;
    void setRemoteDatasetOpener(RemoteDatasetOpener opener);
    bool isRemoteDatasetPath(const std::string& path);

    // Probe a path without reading pixels: dims (as far as the metadata goes),
    // dtype, size, channels. Throws std::runtime_error when unreadable.
    DatasetMeta probeDataset(const std::string& path);
    // The meta openDataset(path, options) would give, still without reading
    // pixels (`readAll` is ignored). Throws like openDataset.
    DatasetMeta probeDataset(const std::string& path, const OpenOptions& options);
    OpenResult openDataset(const std::string& path, const OpenOptions& options = {});

    // The largest dataset, decoded to float32, that a "Full load" still reads
    // into RAM: half the machine's physical memory, or $SIRIUS_FULL_LOAD_MAX_BYTES.
    // Full load is the default, and a 26 GB folder became 31 GB of float32 on
    // every open, whatever the machine; past this limit openDataset serves the
    // dataset lazily instead and says so in OpenResult::fullLoadSkipped.
    // 0 = the platform does not say how much memory there is: no limit.
    std::uint64_t fullLoadLimitBytes();

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

    // How openDataset shapes a TIFF, for a reader of its own (the engine's
    // dataset service, core/dataset_service.hpp, which reads pages in the
    // file's pixel type): the meta, the page order, the samples per page,
    // the first page's parsed metadata and the voxel size the file itself
    // states (0 where it says nothing; before defaults and overrides).
    // Throws like openDataset.
    struct TiffDatasetProbe {
        DatasetMeta meta;
        PageOrder order;                         // over pages: c counts page channels
        Index samples = 1;                       // samples (channels) per page
        bool dimsFromMetadata = false;
        ParsedTiffMetadata parsed;
        std::array<double, 3> fileVoxelUm{0, 0, 0};
        std::string summary;
    };
    TiffDatasetProbe probeTiffDataset(const std::string& path, const TiffInfo& info, const OpenOptions* options);

} // namespace sirius::app

#endif // SIRIUS_APP_ARRAY_SOURCE_HPP
