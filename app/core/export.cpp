#include "core/export.hpp"

#include "core/cancel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

#include <sirius/image_ops.hpp>
#include <sirius/zarr_io.hpp>

namespace sirius::app {

    namespace fs = std::filesystem;
    using json = nlohmann::json;

    // --- small helpers -------------------------------------------------------------

    namespace {

        struct ResolvedRange {
            Index t0 = 0, t1 = 0, z0 = 0, z1 = 0;
            std::vector<Index> channels;
            // an inverted range is empty, not negative (the size estimate is unsigned)
            Index nt() const noexcept { return std::max<Index>(0, t1 - t0); }
            Index nz() const noexcept { return std::max<Index>(0, z1 - z0); }
            Index nc() const noexcept { return static_cast<Index>(channels.size()); }
            Index planes() const noexcept { return nt() * nz() * nc(); }
        };

        ResolvedRange resolveRange(const ExportRange& r, const Dims5& d) {
            ResolvedRange out;
            out.t0 = std::clamp<Index>(r.t0, 0, d.t);
            out.t1 = r.t1 < 0 ? d.t : std::clamp<Index>(r.t1, 0, d.t);
            out.z0 = std::clamp<Index>(r.z0, 0, d.z);
            out.z1 = r.z1 < 0 ? d.z : std::clamp<Index>(r.z1, 0, d.z);
            if (r.channels.empty())
                for (Index c = 0; c < d.c; ++c) out.channels.push_back(c);
            else
                for (Index c : r.channels)
                    if (c >= 0 && c < d.c) out.channels.push_back(c);
            return out;
        }

        bool fullRange(const ResolvedRange& r, const Dims5& d) {
            if (r.t0 != 0 || r.t1 != d.t || r.z0 != 0 || r.z1 != d.z || r.nc() != d.c) return false;
            for (Index c = 0; c < d.c; ++c)
                if (r.channels[static_cast<std::size_t>(c)] != c) return false;
            return true;
        }

        const char* omeTypeName(PixelType t) noexcept {
            switch (t) {
                case PixelType::UInt8: return "uint8";
                case PixelType::Int8: return "int8";
                case PixelType::UInt16: return "uint16";
                case PixelType::Int16: return "int16";
                case PixelType::UInt32: return "uint32";
                case PixelType::Int32: return "int32";
                case PixelType::Float32: return "float";
                case PixelType::Float64: return "double";
            }
            return "float";
        }

        std::string xmlEscape(const std::string& s) {
            std::string out;
            out.reserve(s.size());
            for (char c : s) {
                switch (c) {
                    case '&': out += "&amp;"; break;
                    case '<': out += "&lt;"; break;
                    case '>': out += "&gt;"; break;
                    case '"': out += "&quot;"; break;
                    default: out += c;
                }
            }
            return out;
        }

        std::string fmt(double v) {
            char buf[64];
            std::snprintf(buf, sizeof buf, "%.10g", v);
            return buf;
        }

        // Value range of an integer pixel type (floats: unbounded).
        template <typename T>
        constexpr std::pair<double, double> typeRange() {
            if constexpr (std::is_integral_v<T>)
                return {static_cast<double>(std::numeric_limits<T>::lowest()), static_cast<double>(std::numeric_limits<T>::max())};
            else
                return {-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
        }

        // The affine map v -> v * scale + offset applied before the cast.
        struct Mapping {
            double scale = 1.0;
            double offset = 0.0;
            bool clamp = false;   // integer targets always clamp
            double lo = 0.0, hi = 0.0;
        };

        template <typename T>
        Mapping makeMapping(const ExportOptions& o, double dataLo, double dataHi) {
            Mapping m;
            const auto [tlo, thi] = typeRange<T>();
            m.clamp = std::is_integral_v<T>;
            m.lo = tlo;
            m.hi = thi;
            double lo = 0.0, hi = 1.0;
            switch (o.scaling) {
                case ExportScaling::Cast: return m;
                case ExportScaling::MinMax:
                case ExportScaling::Percentile:
                    lo = dataLo;
                    hi = dataHi;
                    break;
                case ExportScaling::FixedRange:
                    lo = o.rangeLo;
                    hi = o.rangeHi;
                    break;
            }
            if (!(hi > lo)) hi = lo + 1.0;
            // integers: [lo, hi] -> [0, max] and values outside the window
            // saturate at 0 / max (signed types do not go negative);
            // floats: [lo, hi] -> [0, 1]
            const double outHi = std::is_integral_v<T> ? thi : 1.0;
            m.scale = outHi / (hi - lo);
            m.offset = -lo * m.scale;
            if (std::is_integral_v<T>) m.lo = 0.0;
            return m;
        }

        template <typename T>
        inline T mapValue(float v, const Mapping& m) noexcept {
            double d = static_cast<double>(v) * m.scale + m.offset;
            if constexpr (std::is_integral_v<T>) {
                if (std::isnan(d)) return T{};
                d = std::nearbyint(d);
                d = std::min(m.hi, std::max(m.lo, d));
                return static_cast<T>(d);
            } else {
                return static_cast<T>(d);
            }
        }

        // Data range used by MinMax / Percentile over the selected planes.
        std::pair<double, double> selectedRange(const Array5& a, const ResolvedRange& r, const ExportOptions& o) {
            if (o.scaling != ExportScaling::MinMax && o.scaling != ExportScaling::Percentile) return {0.0, 1.0};
            const Dims5& d = a.dims();
            if (o.scaling == ExportScaling::MinMax) {
                float lo = std::numeric_limits<float>::infinity(), hi = -lo;
                for (Index c : r.channels)
                    for (Index t = r.t0; t < r.t1; ++t)
                        for (Index z = r.z0; z < r.z1; ++z) {
                            const float* p = a.plane(c, t, z);
                            for (Index i = 0; i < d.planeSize(); ++i) {
                                const float v = p[i];
                                if (std::isnan(v)) continue;
                                lo = std::min(lo, v);
                                hi = std::max(hi, v);
                            }
                        }
                if (!(lo <= hi)) return {0.0, 1.0};
                return {lo, hi};
            }
            // percentiles over a bounded sample of the selected planes
            std::vector<float> sample;
            const Index planes = r.planes();
            const Index perPlane = std::max<Index>(1, (Index{1} << 22) / std::max<Index>(planes, 1));
            const Index stride = std::max<Index>(1, d.planeSize() / perPlane);
            sample.reserve(static_cast<std::size_t>(planes * (d.planeSize() / stride + 1)));
            for (Index c : r.channels)
                for (Index t = r.t0; t < r.t1; ++t)
                    for (Index z = r.z0; z < r.z1; ++z) {
                        const float* p = a.plane(c, t, z);
                        for (Index i = 0; i < d.planeSize(); i += stride) sample.push_back(p[i]);
                    }
            const auto pr = percentiles(sample.data(), static_cast<Index>(sample.size()), o.percentileLo, o.percentileHi);
            return {pr.first, pr.second};
        }

        // Which plane of the array lands at output plane index `k` for an
        // export whose planes are ordered z fastest, then the middle axis,
        // then the outer axis: "zct" (t outer: OME XYZCT) or "ztc".
        struct PlaneOrder {
            const ResolvedRange& r;
            bool cOuter;   // true: (c, t, z) with z fastest -> OME "XYZTC"; false: (t, c, z) -> "XYZCT"
            Index count() const noexcept { return r.planes(); }
            void at(Index k, Index& c, Index& t, Index& z) const noexcept {
                const Index nz = r.nz(), nt = r.nt(), nc = r.nc();
                z = r.z0 + k % nz;
                k /= nz;
                if (cOuter) {
                    t = r.t0 + k % nt;
                    c = r.channels[static_cast<std::size_t>(k / nt)];
                } else {
                    c = r.channels[static_cast<std::size_t>(k % nc)];
                    t = r.t0 + k / nc;
                }
            }
        };

        // Convert the selected planes into a (planes, y, x) buffer of T.
        template <typename T>
        Buffer<T> convertPlanes(const Array5& a, const PlaneOrder& order, const Mapping& m,
                                const std::function<void(double, const std::string&)>& progress,
                                const std::function<bool()>& cancelled) {
            const Dims5& d = a.dims();
            const Index planes = order.count();
            Buffer<T> out(Shape{planes, d.y, d.x});
            const Index n = d.planeSize();
            for (Index k = 0; k < planes; ++k) {
                if (cancelled && cancelled()) throw CancelledError{};
                Index c, t, z;
                order.at(k, c, t, z);
                const float* src = a.plane(c, t, z);
                T* dst = out.data() + k * n;
                if constexpr (std::is_same_v<T, float>) {
                    if (m.scale == 1.0 && m.offset == 0.0) {
                        std::memcpy(dst, src, static_cast<std::size_t>(n) * sizeof(float));
                        continue;
                    }
                }
                for (Index i = 0; i < n; ++i) dst[i] = mapValue<T>(src[i], m);
                if (progress && (k % 16 == 0)) progress(0.3 * static_cast<double>(k + 1) / static_cast<double>(planes), "converting");
            }
            return out;
        }

        template <typename F>
        void withPixelType(PixelType t, F&& f) {
            switch (t) {
                case PixelType::UInt8: f(std::uint8_t{}); break;
                case PixelType::Int8: f(std::int8_t{}); break;
                case PixelType::UInt16: f(std::uint16_t{}); break;
                case PixelType::Int16: f(std::int16_t{}); break;
                case PixelType::UInt32: f(std::uint32_t{}); break;
                case PixelType::Int32: f(std::int32_t{}); break;
                case PixelType::Float32: f(float{}); break;
                case PixelType::Float64: f(double{}); break;
            }
        }

        std::string stemOf(const std::string& path) {
            fs::path p(path);
            std::string name = p.filename().string();
            // strip ".ome.tif" style double extensions
            for (const char* ext : {".ome.tif", ".ome.tiff", ".tif", ".tiff", ".zarr", ".n5", ".raw"}) {
                const std::string e(ext);
                if (name.size() > e.size() && name.compare(name.size() - e.size(), e.size(), e) == 0) {
                    name.erase(name.size() - e.size());
                    break;
                }
            }
            return (p.parent_path() / name).string();
        }

        json rawSidecar(const DatasetMeta& meta, const Dims5& dims, PixelType type, const std::string& order,
                        const std::array<double, 3>& voxel) {
            json j;
            j["format"] = "sirius-raw";
            j["dtype"] = toString(type);
            j["byte_order"] = "little";
            j["order"] = order;
            j["dims"] = {{"c", dims.c}, {"t", dims.t}, {"z", dims.z}, {"y", dims.y}, {"x", dims.x}};
            j["voxel_um"] = {voxel[0], voxel[1], voxel[2]};
            j["frame_interval_s"] = meta.frameIntervalS;
            json channels = json::array();
            for (const ChannelInfo& c : meta.channels)
                channels.push_back({{"label", c.label}, {"wavelength_nm", c.wavelengthNm}, {"color", c.hexColor()}});
            j["channels"] = channels;
            j["name"] = meta.name;
            return j;
        }

        // Selected channels' metadata, in export order.
        DatasetMeta selectedMeta(const DatasetMeta& meta, const ResolvedRange& r) {
            DatasetMeta m = meta;
            m.dims.c = r.nc();
            m.dims.t = r.nt();
            m.dims.z = r.nz();
            std::vector<ChannelInfo> channels;
            for (Index c : r.channels)
                if (static_cast<std::size_t>(c) < meta.channels.size()) channels.push_back(meta.channels[static_cast<std::size_t>(c)]);
            m.channels = channels;
            m.normalizeChannels();
            return m;
        }

        void writeSidecars(const ExportOptions& o, const std::string& basePath) {
            if (o.includePipeline && !o.pipelineToml.empty()) {
                std::ofstream out(basePath + ".pipeline.toml");
                if (!out) throw std::runtime_error("cannot write " + basePath + ".pipeline.toml");
                out << o.pipelineToml;
            }
        }

    } // namespace

    // --- public --------------------------------------------------------------------------

    std::uint64_t estimateExportBytes(const Dims5& dims, const ExportOptions& o) {
        const ResolvedRange r = resolveRange(o.range, dims);
        const std::uint64_t plane = static_cast<std::uint64_t>(dims.planeSize()) * bytesPerPixel(o.dtype);
        std::uint64_t total = static_cast<std::uint64_t>(r.planes()) * plane;
        int levels = 1;
        double f = 2.0;
        if (o.format == ExportFormat::Tiff) {
            levels = std::max(o.tiff.pyramidLevels, 1);
            f = std::max(o.tiff.downsample, 2);
        } else if (o.format == ExportFormat::Zarr || o.format == ExportFormat::N5) {
            levels = std::max(o.zarr.pyramidLevels, 1);
            f = std::max(o.zarr.downsample, 2);
        }
        double factor = 1.0, level = 1.0;
        for (int k = 1; k < levels; ++k) {
            level /= f * f;
            factor += level;
        }
        total = static_cast<std::uint64_t>(static_cast<double>(total) * factor);
        if (o.includeLabels) total += static_cast<std::uint64_t>(r.nt() * r.nz()) * static_cast<std::uint64_t>(dims.planeSize()) * 4u;
        return total;
    }

    std::vector<Index> zarrChunkShape(const ZarrExportOptions& o) {
        // given (c, t, z, y, x), written (t, c, z, y, x): assigned verbatim, the
        // t and c extents of the chunks were swapped
        return {o.chunk[1], o.chunk[0], o.chunk[2], o.chunk[3], o.chunk[4]};
    }

    std::string exportExtension(const ExportOptions& o) {
        switch (o.format) {
            case ExportFormat::Tiff: return o.tiff.omeXml ? ".ome.tif" : ".tif";
            case ExportFormat::Zarr: return ".zarr";
            case ExportFormat::N5: return ".n5";
            case ExportFormat::Raw: return ".raw";
        }
        return ".tif";
    }

    bool exportFormatAvailable(ExportFormat f) noexcept {
        switch (f) {
            case ExportFormat::Tiff:
            case ExportFormat::Raw: return true;
            case ExportFormat::Zarr:
            case ExportFormat::N5: return zarrSupported();
        }
        return false;
    }

    std::string validateExport(const ExportOptions& o, const Dims5& dims) {
        if (o.path.empty()) return "No destination path.";
        if (!exportFormatAvailable(o.format)) return "This build cannot write zarr / N5 stores (SIRIUS_ENABLE_TENSORSTORE is off).";
        const ResolvedRange r = resolveRange(o.range, dims);
        if (r.nt() <= 0) return "The time range is empty.";
        if (r.nz() <= 0) return "The z range is empty.";
        if (r.nc() <= 0) return "No channel selected.";
        if (o.scaling == ExportScaling::FixedRange && !(o.rangeHi > o.rangeLo)) return "The fixed range needs hi > lo.";
        if (o.scaling == ExportScaling::Percentile && !(o.percentileHi > o.percentileLo)) return "The percentile range needs hi > lo.";
        if (o.format == ExportFormat::Tiff) {
            if (o.tiff.tiled && (o.tiff.tileWidth < 16 || o.tiff.tileHeight < 16)) return "TIFF tiles must be at least 16 x 16.";
            if (o.tiff.pyramidLevels < 1 || o.tiff.pyramidLevels > 16) return "Pyramid levels must be 1..16.";
            if (o.tiff.pyramidLevels > 1 && o.tiff.downsample < 2) return "Pyramid downsample must be >= 2.";
            const std::uint64_t bytes = estimateExportBytes(dims, o);
            if (!o.tiff.bigTiff && bytes > (std::uint64_t{4} << 30)) return "Classic TIFF cannot hold more than 4 GB; enable BigTIFF.";
        }
        if (o.format == ExportFormat::Zarr || o.format == ExportFormat::N5) {
            for (Index c : o.zarr.chunk)
                if (c < 1) return "Chunk extents must be >= 1.";
            if (o.zarr.zarrVersion != 2 && o.zarr.zarrVersion != 3) return "Zarr version must be 2 or 3.";
            if (o.zarr.pyramidLevels < 1 || o.zarr.pyramidLevels > 16) return "Pyramid levels must be 1..16.";
            if (o.zarr.codec != "blosc-zstd" && o.zarr.codec != "blosc-lz4" && o.zarr.codec != "zstd" && o.zarr.codec != "gzip" && o.zarr.codec != "none")
                return "Unknown codec '" + o.zarr.codec + "'.";
        }
        return {};
    }

    std::string omeXml(const DatasetMeta& meta, const Dims5& dims, PixelType type, const std::string& fileName) {
        std::ostringstream x;
        x << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
             "<OME xmlns=\"http://www.openmicroscopy.org/Schemas/OME/2016-06\" "
             "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" "
             "xsi:schemaLocation=\"http://www.openmicroscopy.org/Schemas/OME/2016-06 "
             "http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd\" Creator=\"SIRIUS\">\n";
        x << "  <Image ID=\"Image:0\" Name=\"" << xmlEscape(meta.name.empty() ? fileName : meta.name) << "\">\n";
        if (!meta.acquisition.empty()) x << "    <Description>" << xmlEscape(meta.acquisition) << "</Description>\n";
        // Planes are written z fastest, then t, then c: OME "XYZTC".
        x << "    <Pixels ID=\"Pixels:0\" DimensionOrder=\"XYZTC\" Type=\"" << omeTypeName(type) << "\""
          << " SizeX=\"" << dims.x << "\" SizeY=\"" << dims.y << "\" SizeZ=\"" << dims.z << "\" SizeC=\"" << dims.c
          << "\" SizeT=\"" << dims.t << "\"";
        if (meta.voxelUm[0] > 0) x << " PhysicalSizeX=\"" << fmt(meta.voxelUm[0]) << "\" PhysicalSizeXUnit=\"µm\"";
        if (meta.voxelUm[1] > 0) x << " PhysicalSizeY=\"" << fmt(meta.voxelUm[1]) << "\" PhysicalSizeYUnit=\"µm\"";
        if (meta.voxelUm[2] > 0) x << " PhysicalSizeZ=\"" << fmt(meta.voxelUm[2]) << "\" PhysicalSizeZUnit=\"µm\"";
        if (meta.frameIntervalS > 0) x << " TimeIncrement=\"" << fmt(meta.frameIntervalS) << "\" TimeIncrementUnit=\"s\"";
        x << ">\n";
        for (Index c = 0; c < dims.c; ++c) {
            const ChannelInfo ch = static_cast<std::size_t>(c) < meta.channels.size() ? meta.channels[static_cast<std::size_t>(c)] : ChannelInfo{};
            auto comp = [](float v) { return static_cast<std::uint32_t>(std::lround(std::clamp(v, 0.f, 1.f) * 255.f)); };
            const std::uint32_t rgba = (comp(ch.color[0]) << 24) | (comp(ch.color[1]) << 16) | (comp(ch.color[2]) << 8) | 0xFFu;
            x << "      <Channel ID=\"Channel:0:" << c << "\" Name=\"" << xmlEscape(ch.label.empty() ? "ch " + std::to_string(c) : ch.label)
              << "\" SamplesPerPixel=\"1\" Color=\"" << static_cast<std::int32_t>(rgba) << "\"";
            if (ch.wavelengthNm > 0) x << " EmissionWavelength=\"" << fmt(ch.wavelengthNm) << "\" EmissionWavelengthUnit=\"nm\"";
            x << "/>\n";
        }
        x << "      <TiffData IFD=\"0\" FirstZ=\"0\" FirstC=\"0\" FirstT=\"0\" PlaneCount=\"" << dims.planes() << "\">\n"
          << "        <UUID FileName=\"" << xmlEscape(fileName) << "\">urn:uuid:00000000-0000-0000-0000-000000000000</UUID>\n"
          << "      </TiffData>\n";
        x << "    </Pixels>\n  </Image>\n</OME>\n";
        return x.str();
    }

    void exportArray(const Array5& array, const DatasetMeta& metaIn, const LabelVolume* labels, const ExportOptions& o,
                     const std::function<void(double, const std::string&)>& progress,
                     const std::function<bool()>& cancelled) {
        const Dims5& d = array.dims();
        if (const std::string why = validateExport(o, d); !why.empty()) throw std::runtime_error(why);
        const ResolvedRange r = resolveRange(o.range, d);
        const DatasetMeta meta = selectedMeta(metaIn, r);
        const Dims5 outDims = meta.dims;
        const auto [dataLo, dataHi] = selectedRange(array, r, o);
        const std::string base = stemOf(o.path);
        const std::string fileName = fs::path(o.path).filename().string();
        auto report = [&](double f, const std::string& m) { if (progress) progress(f, m); };
        auto checkCancel = [&] { if (cancelled && cancelled()) throw CancelledError{}; };

        if (o.format == ExportFormat::Tiff || o.format == ExportFormat::Raw) {
            // pages: z fastest, then t, then c (the array's own memory order)
            const PlaneOrder order{r, true};
            withPixelType(o.dtype, [&](auto tag) {
                using T = decltype(tag);
                const Mapping m = makeMapping<T>(o, dataLo, dataHi);
                const bool direct = std::is_same_v<T, float> && o.scaling == ExportScaling::Cast && fullRange(r, d);
                Buffer<T> converted;
                BufferView<const T> view;
                if (direct) {
                    if constexpr (std::is_same_v<T, float>) view = array.stack();
                } else {
                    converted = convertPlanes<T>(array, order, m, progress, cancelled);
                    view = converted.view();
                }
                checkCancel();
                if (o.format == ExportFormat::Tiff) {
                    TiffWriteOptions w;
                    w.compression = o.tiff.compression;
                    w.compressionLevel = o.tiff.compressionLevel;
                    w.predictor = o.tiff.predictor;
                    w.tiled = o.tiff.tiled;
                    w.tileWidth = static_cast<std::uint32_t>(std::max(o.tiff.tileWidth, 16));
                    w.tileHeight = static_cast<std::uint32_t>(std::max(o.tiff.tileHeight, 16));
                    w.rowsPerStrip = static_cast<std::uint32_t>(std::max(o.tiff.rowsPerStrip, 0));
                    w.bigTiff = o.tiff.bigTiff;
                    w.pyramidLevels = o.tiff.pyramidLevels;
                    w.downsample = o.tiff.downsample;
                    w.xPixelUm = meta.voxelUm[0];
                    w.yPixelUm = meta.voxelUm[1];
                    if (o.tiff.omeXml) w.description = omeXml(meta, outDims, o.dtype, fileName);
                    w.progress = [&](double f) { report(0.3 + 0.6 * f, "writing " + fileName); };
                    w.cancelled = cancelled;
                    writeTiffStack<T>(o.path, view, w);
                } else {
                    std::ofstream out(o.path, std::ios::binary);
                    if (!out) throw std::runtime_error("cannot write " + o.path);
                    const std::size_t planeBytes = static_cast<std::size_t>(d.planeSize()) * sizeof(T);
                    for (Index k = 0; k < view.dim(0); ++k) {
                        checkCancel();
                        out.write(reinterpret_cast<const char*>(view.data() + k * d.planeSize()), static_cast<std::streamsize>(planeBytes));
                        if (!out) throw std::runtime_error("write failed: " + o.path);
                        if (k % 16 == 0) report(0.3 + 0.6 * static_cast<double>(k + 1) / static_cast<double>(view.dim(0)), "writing raw");
                    }
                    out.close();
                    std::ofstream side(o.path + ".json");
                    side << rawSidecar(meta, outDims, o.dtype, "ctzyx", meta.voxelUm).dump(2) << "\n";
                }
            });
            if (o.includeLabels && labels && !labels->empty()) {
                checkCancel();
                report(0.92, "writing labels");
                const Index nt = r.nt(), nz = r.nz();
                Buffer<std::uint32_t> lab(Shape{nt * nz, d.y, d.x});
                for (Index t = 0; t < nt; ++t)
                    for (Index z = 0; z < nz; ++z)
                        std::memcpy(lab.data() + (t * nz + z) * d.planeSize(), labels->plane(r.t0 + t, r.z0 + z),
                                    static_cast<std::size_t>(d.planeSize()) * sizeof(std::uint32_t));
                if (o.format == ExportFormat::Tiff) {
                    TiffWriteOptions w;
                    w.compression = TiffCompression::Deflate;
                    w.predictor = true;
                    w.bigTiff = o.tiff.bigTiff;
                    w.description = "SIRIUS labels · order tzyx · t" + std::to_string(nt) + " z" + std::to_string(nz);
                    writeTiffStack<std::uint32_t>(base + ".labels.tif", lab.view(), w);
                } else {
                    std::ofstream out(base + ".labels.raw", std::ios::binary);
                    out.write(reinterpret_cast<const char*>(lab.data()), static_cast<std::streamsize>(lab.bytes()));
                    Dims5 ld = outDims;
                    ld.c = 1;
                    std::ofstream side(base + ".labels.raw.json");
                    side << rawSidecar(meta, ld, PixelType::UInt32, "tzyx", meta.voxelUm).dump(2) << "\n";
                }
            }
        } else {
            // zarr / N5: OME-NGFF axis order (t, c, z, y, x) -> pages z fastest, then c, then t
            const PlaneOrder order{r, false};
            std::vector<std::string> axes{"t", "c", "z", "y", "x"};
            const std::vector<Index> shape{outDims.t, outDims.c, outDims.z, outDims.y, outDims.x};
            ZarrWriteOptions w;
            w.zarrVersion = o.format == ExportFormat::N5 ? 0 : o.zarr.zarrVersion;
            w.chunks = zarrChunkShape(o.zarr);
            w.codec = o.zarr.codec;
            w.level = o.zarr.level;
            w.shard = o.zarr.shard;
            w.axes = axes;
            w.scale = {meta.frameIntervalS > 0 ? meta.frameIntervalS : 1.0, 1.0, meta.voxelUm[2], meta.voxelUm[1], meta.voxelUm[0]};
            for (const ChannelInfo& c : meta.channels) {
                w.channelNames.push_back(c.label);
                w.channelColors.push_back(c.hexColor());
            }
            w.pyramidLevels = o.zarr.pyramidLevels;
            w.downsample = o.zarr.downsample;
            w.omeNgff = o.zarr.omeNgff;
            withPixelType(o.dtype, [&](auto tag) {
                using T = decltype(tag);
                const Mapping m = makeMapping<T>(o, dataLo, dataHi);
                Buffer<T> converted = convertPlanes<T>(array, order, m, progress, cancelled);
                checkCancel();
                writeZarr<T>(o.path, converted.data(), shape, w, [&](double f) { report(0.3 + 0.6 * f, "writing " + fileName); });
            });
            if (o.includeLabels && labels && !labels->empty()) {
                checkCancel();
                report(0.92, "writing labels");
                const Index nt = r.nt(), nz = r.nz();
                std::vector<std::uint32_t> lab(static_cast<std::size_t>(nt * nz * d.planeSize()));
                for (Index t = 0; t < nt; ++t)
                    for (Index z = 0; z < nz; ++z)
                        std::memcpy(lab.data() + (t * nz + z) * d.planeSize(), labels->plane(r.t0 + t, r.z0 + z),
                                    static_cast<std::size_t>(d.planeSize()) * sizeof(std::uint32_t));
                ZarrWriteOptions lw = w;
                lw.axes = {"t", "z", "y", "x"};
                lw.scale = {w.scale[0], w.scale[2], w.scale[3], w.scale[4]};
                lw.chunks = {o.zarr.chunk[1], o.zarr.chunk[2], o.zarr.chunk[3], o.zarr.chunk[4]};   // (t, z, y, x)
                lw.channelNames.clear();
                lw.channelColors.clear();
                lw.pyramidLevels = 1;
                lw.omeNgff = true;
                // OME-NGFF: labels live in <image>/labels/<name>, listed in labels/.zattrs
                const fs::path labelsDir = fs::path(o.path) / "labels";
                std::error_code ec;
                fs::create_directories(labelsDir, ec);
                writeZarr<std::uint32_t>((labelsDir / "labels").string(), lab.data(), {nt, nz, outDims.y, outDims.x}, lw);
                if (lw.zarrVersion == 2) {
                    std::ofstream(labelsDir / ".zgroup") << "{\"zarr_format\": 2}\n";
                    std::ofstream(labelsDir / ".zattrs") << "{\"labels\": [\"labels\"]}\n";
                } else if (lw.zarrVersion == 3) {
                    std::ofstream(labelsDir / "zarr.json") << "{\"zarr_format\": 3, \"node_type\": \"group\", \"attributes\": {\"ome\": {\"version\": \"0.5\", \"labels\": [\"labels\"]}}}\n";
                } else {
                    std::ofstream(labelsDir / "attributes.json") << "{\"labels\": [\"labels\"]}\n";
                }
            }
        }
        writeSidecars(o, o.format == ExportFormat::Tiff || o.format == ExportFormat::Raw ? base : o.path);
        report(1.0, "exported " + fileName);
    }

} // namespace sirius::app
