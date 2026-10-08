#include "core/tool_api.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include <sirius/tiff_io.hpp>

#include "core/cancel.hpp"
#include "core/export.hpp"

// Writing a step's output (export_result) and its labels, the way File >
// Export result and Segment > Export labels do in the application. The two
// halves -- the options a caller's JSON asks for, and the writing itself --
// are declared in core/tool_api.hpp, because every front drives that tool:
// the window through --tool, sirius-cli's session and the MCP server through
// core/headless.hpp.

namespace sirius::app {

    using json = nlohmann::json;

    namespace {
        namespace fs = std::filesystem;

        [[noreturn]] void invalid(const std::string& message, const std::string& hint = {}) {
            throw ToolFailure("invalid_argument", message, hint);
        }

        std::string lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        }

        bool endsWith(const std::string& s, const std::string& suffix) {
            return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
        }

        const json* find(const json& a, const char* key) {
            const auto it = a.find(key);
            return it == a.end() || it->is_null() ? nullptr : &*it;
        }

        std::int64_t integer(const json& v, const std::string& what, std::int64_t min) {
            if (!v.is_number() || (!v.is_number_integer() && v.get<double>() != std::floor(v.get<double>())))
                invalid("'" + what + "' must be an integer");
            const std::int64_t i = v.is_number_integer() ? v.get<std::int64_t>() : static_cast<std::int64_t>(v.get<double>());
            if (i < min) invalid("'" + what + "' must be at least " + std::to_string(min));
            return i;
        }

        bool boolean(const json& v, const std::string& what) {
            if (!v.is_boolean()) invalid("'" + what + "' must be true or false");
            return v.get<bool>();
        }

        std::string text(const json& v, const std::string& what) {
            if (!v.is_string()) invalid("'" + what + "' must be a string");
            return lower(v.get<std::string>());
        }

        // [a, b] of numbers.
        std::pair<double, double> pairOf(const json& v, const std::string& what) {
            if (!v.is_array() || v.size() != 2 || !v[0].is_number() || !v[1].is_number()) invalid("'" + what + "' must be [lo, hi]");
            return {v[0].get<double>(), v[1].get<double>()};
        }

        // [first, end) of an axis; end -1 = to the end.
        std::pair<Index, Index> rangeOf(const json& v, const std::string& what) {
            if (!v.is_array() || v.size() != 2) invalid("'" + what + "' must be [first, end) with end -1 for the last");
            return {static_cast<Index>(integer(v[0], what, 0)), static_cast<Index>(integer(v[1], what, -1))};
        }

        std::optional<PixelType> pixelTypeFromName(const std::string& name) {
            for (PixelType t : {PixelType::UInt8, PixelType::Int8, PixelType::UInt16, PixelType::Int16, PixelType::UInt32, PixelType::Int32,
                                PixelType::Float32, PixelType::Float64})
                if (name == toString(t)) return t;
            return std::nullopt;
        }

        std::string formatName(const ExportOptions& o) {
            switch (o.format) {
                case ExportFormat::Tiff: return o.tiff.omeXml ? "ome-tiff" : "tiff";
                case ExportFormat::Zarr: return "zarr";
                case ExportFormat::N5: return "n5";
                case ExportFormat::Raw: return "raw";
            }
            return "tiff";
        }

        // The path without the extension the writers strip before naming a
        // sidecar ("<base>.labels.tif"); export.cpp's rule.
        std::string baseOf(const std::string& path) {
            std::string out = path;
            const std::string l = lower(path);
            for (const char* ext : {".ome.tif", ".ome.tiff", ".tif", ".tiff", ".zarr", ".n5", ".raw"})
                if (endsWith(l, ext) && l.size() > std::char_traits<char>::length(ext)) {
                    out.resize(out.size() - std::char_traits<char>::length(ext));
                    break;
                }
            return out;
        }

        // A file's size, or everything under a directory (a zarr store).
        std::uint64_t sizeOf(const fs::path& p) {
            std::error_code ec;
            if (fs::is_regular_file(p, ec)) {
                const auto n = fs::file_size(p, ec);
                return ec ? 0 : static_cast<std::uint64_t>(n);
            }
            std::uint64_t total = 0;
            for (fs::recursive_directory_iterator it(p, ec), end; !ec && it != end; it.increment(ec)) {
                std::error_code fec;
                if (it->is_regular_file(fec)) total += static_cast<std::uint64_t>(it->file_size(fec));
            }
            return total;
        }

        std::string reported(const std::string& path) {
            try {
                return fs::u8path(path).lexically_normal().generic_u8string();
            } catch (const std::exception&) {
                return path;
            }
        }
    } // namespace

    ExportOptions exportOptionsFromJson(const json& a, const DatasetMeta& meta) {
        ExportOptions o;
        const json* path = a.is_object() ? find(a, "path") : nullptr;
        if (!path || !path->is_string() || path->get<std::string>().empty()) invalid("export_result needs 'path', the file to write");
        try {
            std::error_code ec;
            const fs::path p = fs::absolute(fs::u8path(path->get<std::string>()), ec);
            o.path = (ec ? fs::u8path(path->get<std::string>()) : p).lexically_normal().u8string();
        } catch (const std::exception&) {
            invalid("'path' is not a usable file name");
        }

        // The container: named, or told by the extension.
        std::string format;
        if (const json* v = find(a, "format")) {
            format = text(*v, "format");
        } else {
            const std::string l = lower(o.path);
            if (endsWith(l, ".ome.tif") || endsWith(l, ".ome.tiff")) format = "ome-tiff";
            else if (endsWith(l, ".tif") || endsWith(l, ".tiff")) format = "tiff";
            else if (endsWith(l, ".zarr")) format = "zarr";
            else if (endsWith(l, ".n5")) format = "n5";
            else if (endsWith(l, ".raw")) format = "raw";
            else invalid("the format cannot be told from '" + o.path + "'", "end the path in .ome.tif, .tif, .zarr, .n5 or .raw, or pass 'format'");
        }
        if (format == "ome-tiff" || format == "tiff") {
            o.format = ExportFormat::Tiff;
            o.tiff.omeXml = format == "ome-tiff";
        } else if (format == "zarr") {
            o.format = ExportFormat::Zarr;
        } else if (format == "n5") {
            o.format = ExportFormat::N5;
        } else if (format == "raw") {
            o.format = ExportFormat::Raw;
        } else {
            invalid("'format' must be tiff, ome-tiff, zarr, n5 or raw");
        }

        if (const json* v = find(a, "dtype")) {
            const auto t = pixelTypeFromName(text(*v, "dtype"));
            if (!t) invalid("'dtype' must be uint8, int8, uint16, int16, uint32, int32, float32 or float64");
            o.dtype = *t;
        }
        const json* range = find(a, "range");
        const json* percentiles = find(a, "percentiles");
        std::string scaling = range ? "fixed" : percentiles ? "percentile"
                                                            : "cast";
        if (const json* v = find(a, "scaling")) scaling = text(*v, "scaling");
        if (scaling == "cast") o.scaling = ExportScaling::Cast;
        else if (scaling == "minmax") o.scaling = ExportScaling::MinMax;
        else if (scaling == "fixed") o.scaling = ExportScaling::FixedRange;
        else if (scaling == "percentile") o.scaling = ExportScaling::Percentile;
        else invalid("'scaling' must be cast, minmax, fixed or percentile");
        if (range) std::tie(o.rangeLo, o.rangeHi) = pairOf(*range, "range");
        if (percentiles) std::tie(o.percentileLo, o.percentileHi) = pairOf(*percentiles, "percentiles");
        if (o.scaling == ExportScaling::FixedRange && !range) invalid("scaling fixed needs 'range': [lo, hi]");

        if (const json* v = find(a, "t")) std::tie(o.range.t0, o.range.t1) = rangeOf(*v, "t");
        if (const json* v = find(a, "z")) std::tie(o.range.z0, o.range.z1) = rangeOf(*v, "z");
        if (const json* v = find(a, "channels")) {
            if (!v->is_array()) invalid("'channels' must be a list of channel indices");
            for (const json& c : *v) {
                const Index ci = static_cast<Index>(integer(c, "channels", 0));
                if (ci >= meta.dims.c) invalid("channel " + std::to_string(ci) + " does not exist (the output has " + std::to_string(meta.dims.c) + ")");
                o.range.channels.push_back(ci);
            }
        }

        if (const json* t = find(a, "tiff")) {
            if (!t->is_object()) invalid("'tiff' must be an object");
            if (const json* v = find(*t, "tiled")) o.tiff.tiled = boolean(*v, "tiff.tiled");
            if (const json* v = find(*t, "tile")) {
                if (!v->is_array() || v->size() != 2) invalid("'tiff.tile' must be [width, height]");
                o.tiff.tileWidth = static_cast<int>(integer((*v)[0], "tiff.tile", 16));
                o.tiff.tileHeight = static_cast<int>(integer((*v)[1], "tiff.tile", 16));
                o.tiff.tiled = true;
            }
            if (const json* v = find(*t, "compression")) {
                const std::string c = text(*v, "tiff.compression");
                if (c == "none") o.tiff.compression = TiffCompression::None;
                else if (c == "lzw") o.tiff.compression = TiffCompression::Lzw;
                else if (c == "deflate" || c == "zip") o.tiff.compression = TiffCompression::Deflate;
                else invalid("'tiff.compression' must be none, lzw or deflate");
            }
            if (const json* v = find(*t, "level")) o.tiff.compressionLevel = static_cast<int>(integer(*v, "tiff.level", 1));
            if (const json* v = find(*t, "bigtiff")) o.tiff.bigTiff = boolean(*v, "tiff.bigtiff");
            if (const json* v = find(*t, "ome")) o.tiff.omeXml = boolean(*v, "tiff.ome");
            if (const json* v = find(*t, "pyramid_levels")) o.tiff.pyramidLevels = static_cast<int>(integer(*v, "tiff.pyramid_levels", 1));
            if (const json* v = find(*t, "downsample")) o.tiff.downsample = static_cast<int>(integer(*v, "tiff.downsample", 2));
        }
        if (const json* z = find(a, "zarr")) {
            if (!z->is_object()) invalid("'zarr' must be an object");
            if (const json* v = find(*z, "version")) o.zarr.zarrVersion = static_cast<int>(integer(*v, "zarr.version", 2));
            if (const json* v = find(*z, "chunk")) {
                if (!v->is_array() || v->size() != 5) invalid("'zarr.chunk' must be [c, t, z, y, x]");
                for (std::size_t i = 0; i < 5; ++i) o.zarr.chunk[i] = static_cast<Index>(integer((*v)[i], "zarr.chunk", 1));
            }
            if (const json* v = find(*z, "codec")) o.zarr.codec = text(*v, "zarr.codec");
            if (const json* v = find(*z, "level")) o.zarr.level = static_cast<int>(integer(*v, "zarr.level", 0));
            if (const json* v = find(*z, "shard")) o.zarr.shard = boolean(*v, "zarr.shard");
            if (const json* v = find(*z, "pyramid_levels")) o.zarr.pyramidLevels = static_cast<int>(integer(*v, "zarr.pyramid_levels", 1));
            if (const json* v = find(*z, "downsample")) o.zarr.downsample = static_cast<int>(integer(*v, "zarr.downsample", 2));
            if (const json* v = find(*z, "ome_ngff")) o.zarr.omeNgff = boolean(*v, "zarr.ome_ngff");
        }
        if (const json* v = find(a, "include_labels")) o.includeLabels = boolean(*v, "include_labels");
        if (const json* v = find(a, "include_pipeline")) o.includePipeline = boolean(*v, "include_pipeline");
        return o;
    }

    json exportStepOutput(std::shared_ptr<const StepOutput> out, const Pipeline& pipeline, const ExportOptions& options, bool labelsOnly,
                          const std::function<void(double, const std::string&)>& progress, const std::function<bool()>& cancelled) {
        const auto started = std::chrono::steady_clock::now();
        if (!out || !(out->array || out->source)) throw ToolFailure("not_computed", "the step has no output to export", "run it first, or pass run:true");
        json warnings = json::array();
        std::vector<std::string> candidates;
        std::string format, dtype, shape;
        const auto fail = [](const std::exception& e) -> ToolFailure {
            if (isCancellation(e)) throw CancelledError();
            return ToolFailure("export_failed", e.what(), "check that the folder exists and can be written, and that the disk has room");
        };

        if (labelsOnly) {
            if (!out->labels || out->labels->empty())
                invalid("the step has no labels to export", "export_result without labels_only writes the pixels");
            // The labels are always one 32-bit TIFF: a .zarr or .raw name would hold a file it does not describe.
            if (options.format != ExportFormat::Tiff)
                invalid("labels_only writes a TIFF: '" + options.path + "' names another format", "end the path in .tif or .ome.tif");
            // A copy, as Segment > Export labels takes the volume it writes.
            const std::shared_ptr<LabelVolume> labels = out->labels->clone();
            // The voxel size goes with them, as ImageJ writes it (the z
            // spacing and the stack's layout in the description, x and y in
            // the resolution tags), so that the file opens at the size of
            // the data it labels instead of a default.
            TiffWriteOptions w;
            w.compression = TiffCompression::Deflate;
            w.xPixelUm = out->meta.voxelUm[0];
            w.yPixelUm = out->meta.voxelUm[1];
            char spacing[32];
            std::snprintf(spacing, sizeof spacing, "%.9g", out->meta.voxelUm[2]);
            w.description = "ImageJ=1.11a\nimages=" + std::to_string(labels->t() * labels->z()) + "\nslices=" + std::to_string(labels->z()) +
                            "\nframes=" + std::to_string(labels->t()) + "\nhyperstack=true\nunit=micron\nspacing=" + spacing + "\n";
            try {
                writeTiffStack<std::uint32_t>(options.path, labels->view().asStack(), w);
            } catch (const std::exception& e) {
                throw fail(e);
            }
            candidates.push_back(options.path);
            format = "tiff";
            dtype = "uint32";
            shape = "c1 t" + std::to_string(labels->t()) + " z" + std::to_string(labels->z()) + " y" + std::to_string(labels->y()) + " x" +
                    std::to_string(labels->x());
        } else {
            ExportOptions o = options;
            // The sidecar is written with Pipeline::save, not Workbench::savePipeline,
            // which would make the export's sidecar the pipeline's own file.
            const std::string sidecar = o.path + ".pipeline.toml";
            const bool writeSidecar = o.includePipeline;
            o.includePipeline = false;
            if (writeSidecar) {
                try {
                    pipeline.save(sidecar);
                    candidates.push_back(sidecar);
                } catch (const std::exception& e) {
                    warnings.push_back(std::string("the pipeline sidecar was not written: ") + e.what());
                }
            }
            std::shared_ptr<const LabelVolume> labels;
            if (o.includeLabels) {
                if (out->labels && !out->labels->empty()) labels = out->labels->clone();
                else warnings.push_back("the step has no labels: none were written");
            }
            try {
                // A lazy source (a Load step's output) is read whole first; its
                // progress callback, called between planes, is where that read stops.
                const auto reading = [&progress, &cancelled](double f, const std::string& m) {
                    if (cancelled && cancelled()) throw CancelledError();
                    if (progress) progress(f, m);
                };
                const ArrayPtr array = out->asInput().materialize(reading);
                if (cancelled && cancelled()) throw CancelledError();
                exportArray(*array, out->meta, labels.get(), o, progress, cancelled);
            } catch (const std::exception& e) {
                throw fail(e);
            }
            // What this export wrote (exportArray's names), not whatever an earlier
            // export to the same base left beside it. The labels of a zarr or N5
            // store live inside the store.
            candidates.push_back(o.path);
            const std::string base = baseOf(o.path);
            if (o.format == ExportFormat::Raw) candidates.push_back(o.path + ".json");
            if (labels && o.format == ExportFormat::Tiff) candidates.push_back(base + ".labels.tif");
            if (labels && o.format == ExportFormat::Raw) {
                candidates.push_back(base + ".labels.raw");
                candidates.push_back(base + ".labels.raw.json");
            }
            format = formatName(o);
            dtype = toString(o.dtype);
            // what the range leaves of the output
            const Dims5& d = out->meta.dims;
            const auto extent = [](Index first, Index end, Index n) { return std::max<Index>(0, (end < 0 ? n : std::min(end, n)) - first); };
            Dims5 e = d;
            e.c = o.range.channels.empty() ? d.c : static_cast<Index>(o.range.channels.size());
            e.t = extent(o.range.t0, o.range.t1, d.t);
            e.z = extent(o.range.z0, o.range.z1, d.z);
            shape = e.toString();
        }

        json files = json::array();
        std::uint64_t bytes = 0;
        for (const std::string& f : candidates) {
            std::error_code ec;
            const fs::path p = fs::u8path(f);
            if (!fs::exists(p, ec)) continue;
            files.push_back(reported(f));
            bytes += sizeOf(p);
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        return {{"path", reported(options.path)},
                {"format", format},
                {"dtype", dtype},
                {"shape", shape},
                {"files", files},
                {"bytes", bytes},
                {"seconds", seconds},
                {"warnings", warnings}};
    }

} // namespace sirius::app
