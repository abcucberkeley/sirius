// Load: the pinned first step. Opens the dataset into RAM by default, or
// lazily (planes on demand), and lets the user override what the file's
// metadata did not say: page order, voxel size, the raw SIM layout, the
// light-sheet angle.
#include "core/ops/builtin.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <map>
#include <mutex>

#include "core/array_source.hpp"
#include "core/manifest.hpp"

namespace sirius::app {

    namespace {

        constexpr const char* kLazy = "Lazy (chunk on demand)";
        constexpr const char* kFull = "Full load to RAM";

        // probeDataset is called from the UI for every repaint of the ops
        // row; cache it per (path, mtime, size) and the options it was probed with.
        struct ProbeCache {
            std::mutex mutex;
            struct Entry {
                std::filesystem::file_time_type mtime;
                std::uintmax_t size = 0;
                DatasetMeta meta;
                std::string error;
            };
            std::map<std::string, Entry> entries;
        };
        ProbeCache& probeCache() {
            static ProbeCache c;
            return c;
        }

        // The options as part of a cache key ("" for none).
        std::string optionsKey(const OpenOptions* o) {
            if (!o) return {};
            const PageOrder po = o->pageOrder.value_or(PageOrder{"-", 0, 0, 0});
            const std::array<double, 3> v = o->voxelUm.value_or(std::array<double, 3>{-1.0, -1.0, -1.0});
            const SimLayout sim = o->sim.value_or(SimLayout{});
            char buf[256];
            std::snprintf(buf, sizeof buf, "|%s %lld %lld %lld|%.17g %.17g %.17g|%d %d %d %d|%lld", po.order.c_str(),
                          static_cast<long long>(po.c), static_cast<long long>(po.t), static_cast<long long>(po.z), v[0], v[1],
                          v[2], sim.present ? 1 : 0, sim.ndirs, sim.nphases, sim.fastSi ? 1 : 0, static_cast<long long>(o->tile));
            return buf;
        }

        // Empty error when the probe succeeded. With `options`, the meta that
        // opening the dataset with them gives (probeDataset(path, options)).
        std::string cachedProbe(const std::string& path, const OpenOptions* options, DatasetMeta& meta) {
            std::error_code ec;
            std::filesystem::path p(path);
            if (path.empty() || !std::filesystem::exists(p, ec)) return "file not found";
            // a multi-file folder changes when its manifest does: a folder's own
            // mtime does not move when a file inside it is rewritten
            if (isFolderDataset(path)) p /= DatasetManifest::kFileName;
            const auto mtime = std::filesystem::last_write_time(p, ec);
            std::uintmax_t size = 0;
            if (std::filesystem::is_regular_file(p, ec)) size = std::filesystem::file_size(p, ec);
            const std::string key = path + optionsKey(options);
            ProbeCache& c = probeCache();
            std::lock_guard<std::mutex> g(c.mutex);
            auto it = c.entries.find(key);
            if (it != c.entries.end() && it->second.mtime == mtime && it->second.size == size) {
                meta = it->second.meta;
                return it->second.error;
            }
            ProbeCache::Entry e;
            e.mtime = mtime;
            e.size = size;
            try {
                e.meta = options ? probeDataset(path, *options) : probeDataset(path);
            } catch (const std::exception& ex) {
                e.error = ex.what();
            }
            // every voxel size dragged through is a key of its own: keep it bounded
            if (c.entries.size() >= 256) c.entries.clear();
            c.entries[key] = e;
            meta = e.meta;
            return e.error;
        }

        // What the parameters add to an opened dataset beyond what the open
        // itself applies (page order, voxel size, SIM layout, tile): the SIM
        // description and the light-sheet angle. Never the dims: those are
        // what the source serves.
        void annotate(const ParamSet& p, DatasetMeta& meta) {
            const Index nd = p.getInt("sim_ndirs"), np = p.getInt("sim_nphases");
            if (nd > 0 && np > 0 && meta.acquisition.empty())
                meta.acquisition = "3D-SIM raw · " + std::to_string(nd * np) + " phase images per plane";
            const double angle = p.getDouble("sheet_angle");
            if (angle > 0.0) {
                meta.lightSheet = true;
                meta.sheetAngleDeg = angle;
                if (meta.acquisition.empty()) meta.acquisition = "Light-sheet";
            }
            meta.normalizeChannels();
        }

        // Tiles a dataset has for the `tile` parameter: one unless a manifest says more.
        Index tileCountOf(const DatasetMeta& meta) { return std::max<Index>(1, static_cast<Index>(meta.tiles.size())); }

        // The pages of a TIFF are mapped onto (c, t, z) by the parameters; a
        // folder's manifest and a zarr store name their axes themselves.
        bool pagedFormat(const DatasetMeta& meta) { return meta.format == "tiff" || meta.format == "ome-tiff"; }

        bool axesGiven(const ParamSet& p) { return p.getInt("c") > 0 || p.getInt("t") > 0 || p.getInt("z") > 0; }

        std::string axesNotAppliedError(const std::string& path) {
            return isManifestDataset(path)
                       ? "A multi-file folder takes its channels, time points and planes from its manifest: set Channels, Time points and Planes to 0."
                       : "A zarr / N5 store names its own axes: set Channels, Time points and Planes to 0.";
        }

        // The dataset as these parameters open it, without reading pixels:
        // what summary, validate and outputMeta describe, so they say what
        // run() will produce. Empty when it opens, the reason otherwise.
        std::string predict(const ParamSet& params, DatasetMeta& meta) {
            const std::string path = params.getString("path");
            DatasetMeta plain;
            std::string err = cachedProbe(path, nullptr, plain);
            if (!err.empty()) return err;
            OpenOptions options = loadOpenOptions(params);
            // an out-of-range tile is validate()'s to report; describe the nearest
            options.tile = std::clamp<Index>(options.tile, 0, tileCountOf(plain) - 1);
            err = cachedProbe(path, &options, meta);
            if (!err.empty()) return err;
            annotate(params, meta);
            return {};
        }

        // "tile 3/9 · tile_x2_y0" for multi-file datasets, empty otherwise.
        std::string tileSummary(const DatasetMeta& meta) {
            if (!meta.hasTiles()) return {};
            const Index tile = std::clamp<Index>(meta.tileIndex, 0, static_cast<Index>(meta.tiles.size()) - 1);
            return "tile " + std::to_string(tile + 1) + "/" + std::to_string(meta.tiles.size()) + " · " +
                   meta.tiles[static_cast<std::size_t>(tile)].name;
        }

        class LoadOperation final : public Operation {
        public:
            LoadOperation() {
                info_.kind = "load";
                info_.name = "Load";
                info_.group = "Input";
                info_.kindLabel = "INPUT";
                info_.defaultCache = CachePolicy::Recompute;
                info_.helpPage = "load";
                info_.params = {
                    pathParam("path", "Source")
                        .withFilter("Images (*.tif *.tiff *.ome.tif *.zarr *.n5);;All files (*)")
                        .withHelp("Multi-page TIFF / OME-TIFF, a zarr / N5 store, or a folder with a sirius-dataset.toml manifest."),
                    choiceParam("read_as", "Read as", {kLazy, kFull}, kFull)
                        .withHelp("Full load reads everything once into RAM (the default); lazy reads planes on demand."),
                    intParam("tile", "Tile", 0).range(0, 1000000).withHelp("Multi-file datasets: the tile to view; Stitch fuses all of them"),
                    stringParam("page_order", "Page order", "czt")
                        .withHelp("Axis order of the pages of a plain TIFF, fastest first (ImageJ: czt).")
                        .asAdvanced(),
                    intParam("c", "Channels", 0).range(0, 1024).withHelp("0 = from the file's metadata").asAdvanced(),
                    intParam("t", "Time points", 0).range(0, 1000000).withHelp("0 = from the file's metadata").asAdvanced(),
                    intParam("z", "Planes", 0).range(0, 1000000).withHelp("0 = from the file's metadata").asAdvanced(),
                    doubleParam("voxel_x", "Voxel x", 0.0).range(0.0, 1000.0, 0.001, 4).withUnit("µm").withHelp("0 = from the file"),
                    doubleParam("voxel_y", "Voxel y", 0.0).range(0.0, 1000.0, 0.001, 4).withUnit("µm").withHelp("0 = from the file"),
                    doubleParam("voxel_z", "Voxel z", 0.0).range(0.0, 1000.0, 0.001, 4).withUnit("µm").withHelp("0 = from the file"),
                    intParam("sim_ndirs", "SIM angles", 0).range(0, 16).withHelp("Raw SIM: pattern directions per plane (0 = not SIM)"),
                    intParam("sim_nphases", "SIM phases", 0).range(0, 32).withHelp("Raw SIM: phase steps per direction"),
                    boolParam("sim_fast", "Fast SI order", false)
                        .withHelp("Sections ordered z → direction → phase instead of direction → z → phase")
                        .asAdvanced(),
                    doubleParam("sheet_angle", "Light-sheet angle", 0.0).range(0.0, 90.0, 0.1, 1).withUnit("°").withHelp("Angle between the light sheet and the coverslip (0 = not light-sheet)"),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            std::string summary(const ParamSet& params, const DatasetMeta&) const override {
                const std::string path = params.getString("path");
                if (path.empty()) return "no dataset";
                DatasetMeta meta;
                const std::string err = predict(params, meta);
                if (!err.empty()) return "cannot open · " + err;
                std::string mode = params.getString("read_as") == kFull ? "full" : "lazy";
                std::string sim;
                if (meta.sim.present)
                    sim = std::to_string(meta.sim.sectionsPerPlane()) + " phase images per plane";
                return joinSummary({mode, meta.format, sim.empty() ? meta.shapeString() : sim, tileSummary(meta)});
            }

            Validation validate(const ParamSet& params, const DatasetMeta&) const override {
                Validation v;
                const std::string path = params.getString("path");
                if (path.empty()) {
                    v.errors.push_back("No dataset: choose a file (File ▸ Open dataset…).");
                    return v;
                }
                DatasetMeta plain;
                const std::string err = cachedProbe(path, nullptr, plain);
                if (!err.empty()) {
                    v.errors.push_back("Cannot open " + path + ": " + err);
                    return v;
                }
                const Index nd = params.getInt("sim_ndirs"), np = params.getInt("sim_nphases");
                if ((nd > 0) != (np > 0)) v.errors.push_back("SIM layout needs both angles and phases.");
                const Index tile = params.getInt("tile");
                if (tile < 0 || tile >= tileCountOf(plain))
                    v.errors.push_back("Tile " + std::to_string(tile) + " is out of range: the dataset has " +
                                       std::to_string(tileCountOf(plain)) + (tileCountOf(plain) == 1 ? " tile." : " tiles."));
                if (axesGiven(params) && !pagedFormat(plain)) {
                    v.errors.push_back(axesNotAppliedError(path));
                    return v;
                }
                DatasetMeta meta;
                const std::string predicted = predict(params, meta);
                if (!predicted.empty()) {
                    v.errors.push_back("Cannot open " + path + ": " + predicted);
                    return v;
                }
                // a layout the pages do not divide into is not applied: the
                // open reads the pages as z instead, and says so here
                const Index c = params.getInt("c"), t = params.getInt("t"), z = params.getInt("z");
                if ((c > 0 && meta.dims.c != c) || (t > 0 && meta.dims.t != t) || (z > 0 && meta.dims.z != z))
                    v.warnings.push_back("Channels, time points and planes do not fit the file's " + std::to_string(meta.dims.planes()) +
                                         " pages: they are read as " + meta.dims.toString() + ".");
                if (meta.sim.present && meta.dims.z % meta.sim.sectionsPerPlane() != 0)
                    v.warnings.push_back(std::to_string(meta.dims.z) + " sections is not a multiple of " +
                                         std::to_string(meta.sim.sectionsPerPlane()) + " (angles × phases).");
                return v;
            }

            DatasetMeta outputMeta(const ParamSet& params, const DatasetMeta& input) const override {
                DatasetMeta meta;
                const std::string err = predict(params, meta);
                if (!err.empty()) return input;
                return meta;
            }

            StepOutput run(const StepInput&, const ParamSet& params, const StepContext& ctx) const override {
                const std::string path = params.getString("path");
                if (path.empty()) throw std::runtime_error("Load: no dataset selected");
                std::error_code ec;
                if (axesGiven(params) && std::filesystem::is_directory(path, ec))
                    throw std::runtime_error("Load: " + axesNotAppliedError(path));
                OpenOptions options = loadOpenOptions(params);
                options.progress = [&](double f, const std::string& m) {
                    ctx.throwIfCancelled();
                    ctx.report(f, m);
                };

                ctx.report(0.0, "opening " + std::filesystem::path(path).filename().string());
                OpenResult opened = openDataset(path, options);
                StepOutput out;
                out.source = opened.source;
                // The dims are the source's. Deriving the axes from the
                // parameters again here, after the open had refused a layout
                // the pages do not divide into, promised planes the source
                // does not have, and every reader of a volume ran past it.
                out.meta = opened.meta;
                annotate(params, out.meta);
                // A full load past the memory limit (fullLoadLimitBytes) was
                // opened lazily instead; reading it all here would undo that.
                const bool inMemory = options.readAll && opened.fullLoadSkipped.empty();
                if (inMemory) {
                    // openDataset already materialized when readAll is set;
                    // this is free on a MemorySource and a fallback otherwise.
                    out.array = opened.source->readAll(options.progress);
                }
                out.note = joinSummary({out.meta.format, out.meta.shapeString(),
                                        inMemory          ? "in memory"
                                        : options.readAll ? "lazy (too large for a full load)"
                                                          : "lazy",
                                        tileSummary(out.meta)});
                out.ranOn = Backend::Cpu;
                ctx.report(1.0, "");
                return out;
            }

        private:
            OpInfo info_;
        };

    } // namespace

    OpenOptions loadOpenOptions(const ParamSet& p) {
        OpenOptions o;
        o.readAll = p.getString("read_as").rfind("Full", 0) == 0;
        const std::string order = p.getString("page_order");
        const Index c = p.getInt("c", 0), t = p.getInt("t", 0), z = p.getInt("z", 0);
        if (c > 0 || t > 0 || z > 0 || (!order.empty() && order != "czt")) {
            PageOrder po;
            po.order = order.empty() ? "czt" : order;
            po.c = std::max<Index>(c, 0);   // 0: the file's own (probeTiff)
            po.t = std::max<Index>(t, 0);
            po.z = std::max<Index>(z, 0);
            o.pageOrder = po;
        }
        // any one of them overrides that axis; 0 keeps the file's
        const double vx = p.getDouble("voxel_x", 0.0), vy = p.getDouble("voxel_y", 0.0), vz = p.getDouble("voxel_z", 0.0);
        if (vx > 0.0 || vy > 0.0 || vz > 0.0) o.voxelUm = std::array<double, 3>{std::max(vx, 0.0), std::max(vy, 0.0), std::max(vz, 0.0)};
        const int ndirs = static_cast<int>(p.getInt("sim_ndirs", 0)), nphases = static_cast<int>(p.getInt("sim_nphases", 0));
        if (ndirs > 0 && nphases > 0) {
            SimLayout sim;
            sim.present = true;
            sim.ndirs = ndirs;
            sim.nphases = nphases;
            sim.fastSi = p.getBool("sim_fast", false);
            o.sim = sim;
        }
        o.tile = std::max<Index>(0, p.getInt("tile", 0));
        return o;
    }

    std::unique_ptr<Operation> makeLoadOperation() { return std::make_unique<LoadOperation>(); }

} // namespace sirius::app
