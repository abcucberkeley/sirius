// Load: the pinned first step. Opens the dataset lazily (planes on demand)
// or reads it fully, and lets the user override what the file's metadata
// did not say: page order, voxel size, the raw SIM layout, the light-sheet
// angle.
#include "core/ops/builtin.hpp"

#include <algorithm>
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
        // row; cache it per (path, mtime, size).
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

        // Empty error when the probe succeeded.
        std::string cachedProbe(const std::string& path, DatasetMeta& meta) {
            std::error_code ec;
            std::filesystem::path p(path);
            if (path.empty() || !std::filesystem::exists(p, ec)) return "file not found";
            // a multi-file folder changes when its manifest does: a folder's own
            // mtime does not move when a file inside it is rewritten
            if (isFolderDataset(path)) p /= DatasetManifest::kFileName;
            const auto mtime = std::filesystem::last_write_time(p, ec);
            std::uintmax_t size = 0;
            if (std::filesystem::is_regular_file(p, ec)) size = std::filesystem::file_size(p, ec);
            ProbeCache& c = probeCache();
            std::lock_guard<std::mutex> g(c.mutex);
            auto it = c.entries.find(path);
            if (it != c.entries.end() && it->second.mtime == mtime && it->second.size == size) {
                meta = it->second.meta;
                return it->second.error;
            }
            ProbeCache::Entry e;
            e.mtime = mtime;
            e.size = size;
            try {
                e.meta = probeDataset(path);
            } catch (const std::exception& ex) {
                e.error = ex.what();
            }
            c.entries[path] = e;
            meta = e.meta;
            return e.error;
        }

        void applyOverrides(const ParamSet& p, DatasetMeta& meta) {
            const double vx = p.getDouble("voxel_x"), vy = p.getDouble("voxel_y"), vz = p.getDouble("voxel_z");
            if (vx > 0) meta.voxelUm[0] = vx;
            if (vy > 0) meta.voxelUm[1] = vy;
            if (vz > 0) meta.voxelUm[2] = vz;
            const Index c = p.getInt("c"), t = p.getInt("t"), z = p.getInt("z");
            if (c > 0 || t > 0 || z > 0) {
                // the page count is fixed: derive the axis that was left at 0
                const Index pages = meta.dims.planes();
                Dims5 d = meta.dims;
                d.c = c > 0 ? c : 1;
                d.t = t > 0 ? t : 1;
                d.z = z > 0 ? z : std::max<Index>(1, pages / (d.c * d.t));
                if (d.planes() == pages) meta.dims = d;
            }
            const Index nd = p.getInt("sim_ndirs"), np = p.getInt("sim_nphases");
            if (nd > 0 && np > 0) {
                meta.sim.present = true;
                meta.sim.ndirs = static_cast<int>(nd);
                meta.sim.nphases = static_cast<int>(np);
                meta.sim.fastSi = p.getBool("sim_fast");
                if (meta.acquisition.empty())
                    meta.acquisition = "3D-SIM raw · " + std::to_string(nd * np) + " phase images per plane";
            }
            const double angle = p.getDouble("sheet_angle");
            if (angle > 0.0) {
                meta.lightSheet = true;
                meta.sheetAngleDeg = angle;
                if (meta.acquisition.empty()) meta.acquisition = "Light-sheet";
            }
            const Index tile = p.getInt("tile");
            if (tile >= 0 && tile < static_cast<Index>(meta.tiles.size())) meta.tileIndex = tile;
            meta.normalizeChannels();
        }

        // Tiles a dataset has for the `tile` parameter: one unless a manifest says more.
        Index tileCountOf(const DatasetMeta& meta) { return std::max<Index>(1, static_cast<Index>(meta.tiles.size())); }

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
                    choiceParam("read_as", "Read as", {kLazy, kFull}, kLazy)
                        .withHelp("Lazy reads planes on demand; full load reads everything once."),
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
                const std::string err = cachedProbe(path, meta);
                if (!err.empty()) return "cannot open · " + err;
                applyOverrides(params, meta);
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
                DatasetMeta meta;
                const std::string err = cachedProbe(path, meta);
                if (!err.empty()) {
                    v.errors.push_back("Cannot open " + path + ": " + err);
                    return v;
                }
                const Index nd = params.getInt("sim_ndirs"), np = params.getInt("sim_nphases");
                if ((nd > 0) != (np > 0)) v.errors.push_back("SIM layout needs both angles and phases.");
                const Index tile = params.getInt("tile");
                if (tile < 0 || tile >= tileCountOf(meta))
                    v.errors.push_back("Tile " + std::to_string(tile) + " is out of range: the dataset has " +
                                       std::to_string(tileCountOf(meta)) + (tileCountOf(meta) == 1 ? " tile." : " tiles."));
                applyOverrides(params, meta);
                if (meta.sim.present && meta.dims.z % meta.sim.sectionsPerPlane() != 0)
                    v.warnings.push_back(std::to_string(meta.dims.z) + " sections is not a multiple of " +
                                         std::to_string(meta.sim.sectionsPerPlane()) + " (angles × phases).");
                return v;
            }

            DatasetMeta outputMeta(const ParamSet& params, const DatasetMeta& input) const override {
                DatasetMeta meta;
                const std::string err = cachedProbe(params.getString("path"), meta);
                if (!err.empty()) return input;
                applyOverrides(params, meta);
                return meta;
            }

            StepOutput run(const StepInput&, const ParamSet& params, const StepContext& ctx) const override {
                const std::string path = params.getString("path");
                if (path.empty()) throw std::runtime_error("Load: no dataset selected");
                OpenOptions options;
                const Index c = params.getInt("c"), t = params.getInt("t"), z = params.getInt("z");
                const std::string order = params.getString("page_order", "czt");
                if (c > 0 || t > 0 || z > 0 || order != "czt") {
                    PageOrder po;
                    po.order = order.empty() ? "czt" : order;
                    po.c = std::max<Index>(1, c);
                    po.t = std::max<Index>(1, t);
                    po.z = std::max<Index>(0, z);
                    options.pageOrder = po;
                }
                const double vx = params.getDouble("voxel_x"), vy = params.getDouble("voxel_y"), vz = params.getDouble("voxel_z");
                if (vx > 0 || vy > 0 || vz > 0) {
                    DatasetMeta probe;
                    cachedProbe(path, probe);
                    options.voxelUm = {vx > 0 ? vx : probe.voxelUm[0], vy > 0 ? vy : probe.voxelUm[1],
                                       vz > 0 ? vz : probe.voxelUm[2]};
                }
                const Index nd = params.getInt("sim_ndirs"), np = params.getInt("sim_nphases");
                if (nd > 0 && np > 0) {
                    SimLayout sim;
                    sim.present = true;
                    sim.ndirs = static_cast<int>(nd);
                    sim.nphases = static_cast<int>(np);
                    sim.fastSi = params.getBool("sim_fast");
                    options.sim = sim;
                }
                options.readAll = params.getString("read_as") == kFull;
                options.tile = std::max<Index>(0, params.getInt("tile"));

                ctx.report(0.0, "opening " + std::filesystem::path(path).filename().string());
                OpenResult opened = openDataset(path, options);
                StepOutput out;
                out.source = opened.source;
                out.meta = opened.meta;
                applyOverrides(params, out.meta);
                if (options.readAll) {
                    out.array = opened.source->readAll([&](double f, const std::string& m) { ctx.report(f, m); });
                }
                out.note = joinSummary({out.meta.format, out.meta.shapeString(), options.readAll ? "in memory" : "lazy", tileSummary(out.meta)});
                out.ranOn = Backend::Cpu;
                ctx.report(1.0, "");
                return out;
            }

        private:
            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeLoadOperation() { return std::make_unique<LoadOperation>(); }

} // namespace sirius::app
