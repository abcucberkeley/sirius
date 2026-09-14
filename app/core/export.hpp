#ifndef SIRIUS_APP_EXPORT_HPP
#define SIRIUS_APP_EXPORT_HPP

// Writing a step's output to disk with full control over the container:
// TIFF (strips or tiles, compression, BigTIFF, OME-XML, resolution pyramid)
// or zarr / N5 through TensorStore (chunks, codec, level, OME-NGFF
// multiscales), any pixel type with an explicit scaling rule, an optional
// t/z/c range and sidecars for the pipeline and the labels.

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <sirius/tiff_io.hpp>

#include "core/array.hpp"
#include "core/dataset.hpp"
#include "core/labels.hpp"

namespace sirius::app {

    enum class ExportFormat { Tiff,
                              Zarr,
                              N5,
                              Raw };
    enum class ExportScaling { Cast,
                               MinMax,
                               FixedRange,
                               Percentile };

    struct ExportRange {
        Index t0 = 0, t1 = -1;              // half open; -1 = to the end
        Index z0 = 0, z1 = -1;
        std::vector<Index> channels;        // empty = all
    };

    struct TiffExportOptions {
        bool tiled = false;
        int tileWidth = 512, tileHeight = 512;
        int rowsPerStrip = 0;               // 0 = libtiff default
        TiffCompression compression = TiffCompression::Deflate;
        int compressionLevel = 6;           // Deflate 1..9
        bool predictor = true;              // horizontal differencing for LZW/Deflate
        bool bigTiff = true;
        bool omeXml = true;                 // ImageDescription with OME-XML
        int pyramidLevels = 1;              // 1 = none; >1 writes reduced-resolution SubIFDs
        int downsample = 2;
    };

    struct ZarrExportOptions {
        int zarrVersion = 3;                // 2 or 3 (ignored for N5)
        std::array<Index, 5> chunk{1, 1, 16, 512, 512};   // (c, t, z, y, x), as the export dialog lists it
        std::string codec = "blosc-zstd";   // "blosc-zstd", "blosc-lz4", "zstd", "gzip", "none"
        int level = 3;
        bool shard = false;                 // zarr 3 sharding (one shard = shardChunks^3 chunks)
        int pyramidLevels = 1;
        int downsample = 2;
        bool omeNgff = true;                // multiscales metadata
    };

    struct ExportOptions {
        std::string path;
        ExportFormat format = ExportFormat::Tiff;
        PixelType dtype = PixelType::Float32;
        ExportScaling scaling = ExportScaling::Cast;
        double rangeLo = 0.0, rangeHi = 1.0;        // FixedRange
        double percentileLo = 0.1, percentileHi = 99.9;   // Percentile
        ExportRange range;
        TiffExportOptions tiff;
        ZarrExportOptions zarr;
        bool includePipeline = false;       // "<path>.pipeline.toml" sidecar
        bool includeLabels = false;         // "<path>.labels.tif" / labels array
        std::string pipelineToml;           // written when includePipeline
    };

    // Rough size of the file(s) the options would produce, before compression.
    std::uint64_t estimateExportBytes(const Dims5& dims, const ExportOptions& o);
    // Extension the format expects (".ome.tif", ".tif", ".zarr", ".n5", ".raw").
    std::string exportExtension(const ExportOptions& o);
    // Formats this build can write (Zarr / N5 need TensorStore).
    bool exportFormatAvailable(ExportFormat f) noexcept;
    // Empty when the options are consistent, otherwise the problem.
    std::string validateExport(const ExportOptions& o, const Dims5& dims);
    // The chunk shape in the order a zarr / N5 export writes its axes,
    // (t, c, z, y, x) as OME-NGFF has them, from ZarrExportOptions::chunk.
    std::vector<Index> zarrChunkShape(const ZarrExportOptions& o);

    void exportArray(const Array5& array, const DatasetMeta& meta, const LabelVolume* labels,
                     const ExportOptions& options, const std::function<void(double, const std::string&)>& progress = {},
                     const std::function<bool()>& cancelled = {});

    // OME-XML for an array (also used by the TIFF writer).
    std::string omeXml(const DatasetMeta& meta, const Dims5& dims, PixelType type, const std::string& fileName);

} // namespace sirius::app

#endif // SIRIUS_APP_EXPORT_HPP
