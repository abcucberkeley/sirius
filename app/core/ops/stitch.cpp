// Stitch tiles: masked pairwise registration of every overlapping tile pair,
// a global fit of the tile origins and a blended fusion (sirius/stitching.hpp).
// Two sources of tiles:
//   - a list of tile TIFFs with nominal stage positions (the `tiles` param);
//   - with `tiles` empty, the tiles of the input dataset itself, a multi-file
//     folder whose manifest gives every tile's nominal origin. Registration
//     runs once, on one channel at one time point; the layout it finds is
//     then applied to every (c, t).
// Memory: every tile of one (c, t) plus the fused canvas are held at once, so
// this suits mosaics whose one-channel volume fits in RAM.
#include "core/ops/builtin.hpp"

#include <limits>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <set>

#include <sirius/stitching.hpp>

#include "core/array_source.hpp"

namespace sirius::app {

    namespace {

        BlendMode blendOf(const std::string& s) {
            if (s == "Average") return BlendMode::Average;
            if (s == "Maximum") return BlendMode::Maximum;
            if (s == "Overwrite") return BlendMode::Overwrite;
            return BlendMode::Feather;
        }

        // Canvas of the nominal layout: the tile extent plus the span of the origins.
        Dims5 nominalCanvas(const std::vector<std::array<double, 3>>& positions, Index z, Index y, Index x) {
            Dims5 d{1, 1, z, y, x};
            if (positions.empty()) return d;
            double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
            for (const auto& q : positions)
                for (std::size_t k = 0; k < 3; ++k) {
                    lo[k] = std::min(lo[k], q[k]);
                    hi[k] = std::max(hi[k], q[k]);
                }
            d.z = static_cast<Index>(std::lround(hi[0] - lo[0])) + z;
            d.y = static_cast<Index>(std::lround(hi[1] - lo[1])) + y;
            d.x = static_cast<Index>(std::lround(hi[2] - lo[2])) + x;
            return d;
        }

        // The tile map of a multi-file dataset: rows / columns from the
        // manifest's grid indices, names row-major over that grid. Tiles that
        // do not form a grid (no indices, or two tiles on one cell) go in a row.
        void tileGrid(const std::vector<TileInfo>& tiles, AlignmentInfo& a) {
            // rows / columns from the indices' own origin: a folder numbered
            // from 1 is not a grid with an empty first row and column
            Index row0 = std::numeric_limits<Index>::max(), col0 = row0;
            for (const TileInfo& t : tiles)
                if (t.gridIndex[1] >= 0 && t.gridIndex[2] >= 0) {
                    row0 = std::min(row0, t.gridIndex[1]);
                    col0 = std::min(col0, t.gridIndex[2]);
                }
            if (row0 == std::numeric_limits<Index>::max()) row0 = col0 = 0;
            Index rows = 0, cols = 0;
            for (const TileInfo& t : tiles) {
                rows = std::max(rows, t.gridIndex[1] - row0 + 1);
                cols = std::max(cols, t.gridIndex[2] - col0 + 1);
            }
            const std::size_t n = tiles.size();
            bool grid = rows > 0 && cols > 0 && static_cast<std::size_t>(rows * cols) >= n && static_cast<std::size_t>(rows * cols) <= 4 * n + 4;
            std::vector<std::string> names;
            if (grid) {
                names.assign(static_cast<std::size_t>(rows * cols), std::string());
                for (const TileInfo& t : tiles) {
                    if (t.gridIndex[1] < 0 || t.gridIndex[2] < 0) {
                        grid = false;
                        break;
                    }
                    std::string& cell = names[static_cast<std::size_t>((t.gridIndex[1] - row0) * cols + (t.gridIndex[2] - col0))];
                    if (!cell.empty()) {
                        grid = false;
                        break;
                    }   // two tiles on one cell (a z stack of tiles)
                    cell = t.name;
                }
            }
            if (!grid) {
                rows = 1;
                cols = static_cast<Index>(n);
                names.clear();
                for (const TileInfo& t : tiles) names.push_back(t.name);
            }
            a.gridRows = rows;
            a.gridCols = cols;
            a.tileNames = std::move(names);
        }

        // Where tile `index` sits in the row-major tile map.
        int cellOf(const AlignmentInfo& a, const std::string& name) {
            for (std::size_t i = 0; i < a.tileNames.size(); ++i)
                if (a.tileNames[i] == name) return static_cast<int>(i);
            return -1;
        }

        struct ShiftStats {
            int accepted = 0;
            double meanShift = 0.0, maxShift = 0.0, meanNcc = 0.0;
        };

        // Residual of every accepted pair: measured minus nominal displacement.
        ShiftStats shiftStats(const StitchLayout& layout) {
            ShiftStats s;
            double sum = 0.0, ncc = 0.0;
            for (const TileMatch& m : layout.matches) {
                if (!m.accepted) continue;
                double dd = 0.0;
                for (std::size_t k = 0; k < 3; ++k) {
                    const double e = m.displacement[k] - m.nominalDisplacement[k];
                    dd += e * e;
                }
                dd = std::sqrt(dd);
                sum += dd;
                s.maxShift = std::max(s.maxShift, dd);
                ncc += m.correlation;
                ++s.accepted;
            }
            if (s.accepted > 0) {
                s.meanShift = sum / s.accepted;
                s.meanNcc = ncc / s.accepted;
            }
            return s;
        }

        class StitchOperation final : public Operation {
        public:
            StitchOperation() {
                info_.kind = "stitch";
                info_.name = "Stitch tiles";
                info_.group = "Combine";
                info_.kindLabel = "COMBINE";
                info_.diagnostics = DiagnosticsKind::Alignment;
                info_.defaultCache = CachePolicy::Disk;
                info_.helpPage = "stitch";
                ParamSpec tiles;
                tiles.key = "tiles";
                tiles.label = "Tiles";
                tiles.type = ParamType::StringList;
                tiles.defaultValue = std::vector<std::string>{};
                tiles.help = "One multi-page TIFF per tile; empty = the tiles of the input dataset (multi-file folder)";
                info_.params = {
                    tiles,
                    channelParam("channel", "Registration channel", 0)
                        .withHelp("Dataset tiles: the channel the tiles are registered on; the layout applies to every channel"),
                    intParam("reference_t", "Registration t", 0).range(0, 1000000).withHelp("Dataset tiles: the time point the tiles are registered at"),
                    doubleListParam("positions", "Positions", {})
                        .withUnit("voxels")
                        .withHelp("Tile files: nominal origins, z y x per tile (flattened); empty = a grid from the overlap"),
                    intParam("grid_cols", "Grid columns", 0).range(0, 1000).withHelp("Tile files: 0 = square grid (positions empty)"),
                    doubleParam("overlap_fraction", "Overlap", 0.10).range(0.0, 0.9, 0.01, 2).withHelp("Tile files: nominal overlap between neighbours (positions empty)"),
                    doubleListParam("search_radius", "Search radius", {4.0, 32.0, 32.0}).withUnit("voxels").withHelp("How far a tile may move from its nominal origin (z, y, x)"),
                    doubleParam("min_correlation", "Min. correlation", 0.3).range(0.0, 1.0, 0.05, 2),
                    choiceParam("blend", "Blend", {"Feather", "Average", "Maximum", "Overwrite"}, "Feather"),
                    boolParam("mask_background", "Mask background", true)
                        .withHelp("Ignore voxels at or below the background level when correlating"),
                    doubleParam("background_level", "Background level", 0.0).range(-1e9, 1e9, 1.0, 2),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            // Tile-set mode: no tile files given and the input dataset has tiles.
            static bool tileSetMode(const ParamSet& p, const DatasetMeta& in) {
                return p.getStringList("tiles").empty() && in.hasTiles();
            }

            std::string summary(const ParamSet& p, const DatasetMeta& in) const override {
                std::string blend = p.getString("blend", "Feather");
                std::transform(blend.begin(), blend.end(), blend.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (tileSetMode(p, in))
                    return joinSummary({std::to_string(in.tiles.size()) + " dataset tiles", "on " + channelName(in, p.getInt("channel")), blend});
                const std::vector<std::string> tiles = p.getStringList("tiles");
                if (tiles.empty()) return "no tiles";
                char ov[32];
                std::snprintf(ov, sizeof ov, "overlap %.0f %%", 100.0 * p.getDouble("overlap_fraction", 0.1));
                return joinSummary({std::to_string(tiles.size()) + " tiles", ov, blend});
            }

            Validation validate(const ParamSet& p, const DatasetMeta& in) const override {
                Validation v;
                if (tileSetMode(p, in)) {
                    const Index c = p.getInt("channel"), t = p.getInt("reference_t");
                    if (c < 0 || c >= in.dims.c)
                        v.errors.push_back("Registration channel " + std::to_string(c) + " is out of range: the dataset has " +
                                           std::to_string(in.dims.c) + " channels.");
                    if (t < 0 || t >= in.dims.t)
                        v.errors.push_back("Registration t " + std::to_string(t) + " is out of range: the dataset has " +
                                           std::to_string(in.dims.t) + " time points.");
                    return v;
                }
                const std::vector<std::string> tiles = p.getStringList("tiles");
                if (tiles.size() < 2)
                    v.errors.push_back(tiles.empty() ? "Stitching needs tile files, or a multi-file dataset with more than one tile."
                                                     : "Stitching needs at least two tile files.");
                for (const std::string& t : tiles)
                    if (!std::filesystem::exists(t)) v.errors.push_back("Tile not found: " + t);
                const std::vector<double> pos = p.getDoubleList("positions");
                if (!pos.empty() && pos.size() != 3 * tiles.size())
                    v.errors.push_back("Positions must hold z y x for every tile (" + std::to_string(3 * tiles.size()) +
                                       " numbers, got " + std::to_string(pos.size()) + ").");
                return v;
            }

            // Nominal positions: given, or a grid from the overlap and the first tile's size.
            std::vector<std::array<double, 3>> positionsOf(const ParamSet& p, const std::vector<std::string>& tiles,
                                                           Index& rows, Index& cols) const {
                std::vector<std::array<double, 3>> out;
                const std::vector<double> pos = p.getDoubleList("positions");
                const std::size_t n = tiles.size();
                if (pos.size() == 3 * n) {
                    for (std::size_t i = 0; i < n; ++i) out.push_back({pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]});
                    std::set<long long> ys, xs;
                    for (const auto& q : out) {
                        ys.insert(std::llround(q[1]));
                        xs.insert(std::llround(q[2]));
                    }
                    rows = static_cast<Index>(ys.size());
                    cols = static_cast<Index>(xs.size());
                    return out;
                }
                const TiffInfo info = inspectTiff(tiles.front());
                cols = p.getInt("grid_cols", 0);
                if (cols <= 0) cols = static_cast<Index>(std::ceil(std::sqrt(static_cast<double>(n))));
                rows = static_cast<Index>((n + static_cast<std::size_t>(cols) - 1) / static_cast<std::size_t>(cols));
                const double ov = p.getDouble("overlap_fraction", 0.1);
                const double sx = info.width() * (1.0 - ov), sy = info.height() * (1.0 - ov);
                for (std::size_t i = 0; i < n; ++i) {
                    const Index r = static_cast<Index>(i) / cols, c = static_cast<Index>(i) % cols;
                    out.push_back({0.0, r * sy, c * sx});
                }
                return out;
            }

            DatasetMeta outputMeta(const ParamSet& p, const DatasetMeta& in) const override {
                DatasetMeta out = in;
                out.acquisition = "Stitched mosaic";
                out.sourceType = PixelType::Float32;
                out.tiles.clear();
                out.tileIndex = 0;
                if (tileSetMode(p, in)) {
                    // the mosaic keeps every channel and time point of the tiles
                    const Dims5 canvas = nominalCanvas(in.tilePositionsPx(), in.dims.z, in.dims.y, in.dims.x);
                    out.dims = Dims5{in.dims.c, in.dims.t, canvas.z, canvas.y, canvas.x};
                    out.normalizeChannels();
                    return out;
                }
                out.dims = Dims5{1, 1, 1, 1, 1};
                out.channels.clear();
                out.rgb = false;
                out.sim = SimLayout{};
                const std::vector<std::string> tiles = p.getStringList("tiles");
                try {
                    if (!tiles.empty()) {
                        Index rows = 0, cols = 0;
                        const auto pos = positionsOf(p, tiles, rows, cols);
                        const TiffInfo info = inspectTiff(tiles.front());
                        out.dims = nominalCanvas(pos, static_cast<Index>(info.pageCount()), info.height(), info.width());
                    }
                } catch (const std::exception&) {
                    // unreadable tiles: validate() reports it
                }
                out.normalizeChannels();
                return out;
            }

            StepOutput run(const StepInput& input, const ParamSet& p, const StepContext& ctx) const override {
                const Validation v = validate(p, input.meta);
                if (!v.ok()) throw std::runtime_error(v.firstError());
                return tileSetMode(p, input.meta) ? runTileSet(input, p, ctx) : runFiles(input, p, ctx);
            }

        private:
            StitchOptions optionsOf(const ParamSet& p) const {
                StitchOptions options;
                const std::vector<double> sr = p.getDoubleList("search_radius");
                if (sr.size() == 3)
                    options.searchRadius = {static_cast<Index>(sr[0]), static_cast<Index>(sr[1]), static_cast<Index>(sr[2])};
                options.minCorrelation = p.getDouble("min_correlation", 0.3);
                options.blend = blendOf(p.getString("blend", "Feather"));
                options.maskBackground = p.getBool("mask_background", true);
                options.backgroundLevel = p.getDouble("background_level", 0.0);
                options.skipBackground = options.maskBackground;
                options.fusionBackgroundLevel = options.backgroundLevel;
                options.minOverlapFraction = 0.02;
                options.anchorTile = 0;
                return options;
            }

            // --- tile files -------------------------------------------------------

            StepOutput runFiles(const StepInput& input, const ParamSet& p, const StepContext& ctx) const {
                const std::vector<std::string> paths = p.getStringList("tiles");
                Index rows = 0, cols = 0;
                const auto positions = positionsOf(p, paths, rows, cols);
                std::vector<StitchTile> tiles;
                for (std::size_t i = 0; i < paths.size(); ++i) tiles.push_back({paths[i], positions[i]});
                const StitchOptions options = optionsOf(p);
                ctx.report(0.05, "registering " + std::to_string(tiles.size()) + " tiles");
                StitchLayout layout;
                Buffer<float> mosaic = stitchTiffTiles<float>(tiles, options, &layout);
                ctx.throwIfCancelled();
                ctx.report(0.9, "fused");

                StepOutput out;
                out.meta = outputMeta(p, input.meta);
                const Shape s = mosaic.shape().asStack();
                out.meta.dims = Dims5{1, 1, s[0], s[1], s[2]};
                mosaic.reshape(s);
                out.array = std::make_shared<Array5>(Array5::fromBuffer(std::move(mosaic), out.meta.dims));
                out.ranOn = Backend::Cpu;

                AlignmentInfo a;
                a.gridRows = rows;
                a.gridCols = cols;
                for (std::size_t i = 0; i < tiles.size(); ++i) a.tileNames.push_back("t" + std::to_string(i + 1));
                a.highlightedTile = 0;
                const ShiftStats stats = shiftStats(layout);
                out.diagnostics = diagnose(summary(p, input.meta), std::move(a), layout, stats, *out.array, 0, 0);
                out.note = std::to_string(tiles.size()) + " tiles · " + std::to_string(stats.accepted) + " pairs · CPU";
                ctx.report(1.0, "");
                return out;
            }

            // --- the input dataset's tiles ----------------------------------------

            // A source that can serve every tile: the lazy folder source the Load
            // step made, or the folder reopened when the input was materialized
            // (a memory source holds one tile) or comes from a later step.
            static std::shared_ptr<ArraySource> tileSource(const StepInput& input) {
                const Index n = static_cast<Index>(input.meta.tiles.size());
                if (input.source && input.source->tileCount() == n) return input.source;
                if (isFolderDataset(input.meta.sourcePath)) return openDataset(input.meta.sourcePath, OpenOptions{}).source;
                throw std::runtime_error("Stitch: the tiles of " + input.meta.name + " are not readable here (" +
                                         (input.meta.sourcePath.empty() ? std::string("no source folder") : input.meta.sourcePath) + ")");
            }

            StepOutput runTileSet(const StepInput& input, const ParamSet& p, const StepContext& ctx) const {
                const DatasetMeta& meta = input.meta;
                const Dims5 d = meta.dims;
                const std::size_t n = meta.tiles.size();
                const Index channel = p.getInt("channel"), refT = p.getInt("reference_t");
                const std::shared_ptr<ArraySource> source = tileSource(input);
                if (source->meta().dims != d)
                    throw std::runtime_error("Stitch: the tile source is " + source->meta().dims.toString() + ", the input " + d.toString());
                const StitchOptions options = optionsOf(p);
                const std::vector<std::array<double, 3>> nominal = meta.tilePositionsPx();

                // every tile of one (c, t): the working set of both registration and fusion
                std::vector<Buffer<float>> volumes;
                volumes.reserve(n);
                for (std::size_t i = 0; i < n; ++i) volumes.emplace_back(Shape{d.z, d.y, d.x});
                std::vector<BufferView<const float>> views;
                for (const Buffer<float>& b : volumes) views.push_back(b.view());
                auto load = [&](Index c, Index t, double from, double to) {
                    for (std::size_t i = 0; i < n; ++i) {
                        ctx.throwIfCancelled();
                        ctx.report(from + (to - from) * static_cast<double>(i) / static_cast<double>(n),
                                   "reading " + meta.tiles[i].name + " · c" + std::to_string(c + 1) + " t" + std::to_string(t + 1));
                        source->readTileVolume(static_cast<Index>(i), c, t, volumes[i].data());
                    }
                };

                load(channel, refT, 0.0, 0.1);
                ctx.report(0.1, "registering " + std::to_string(n) + " tiles on " + channelName(meta, channel));
                const StitchLayout layout = planStitch<float>(views, nominal, options);
                ctx.throwIfCancelled();

                StepOutput out;
                out.meta = outputMeta(p, meta);
                const std::array<Index, 3>& ext = layout.canvasExtent;
                out.meta.dims = Dims5{d.c, d.t, ext[0], ext[1], ext[2]};
                auto mosaic = std::make_shared<Array5>(out.meta.dims);
                const std::size_t canvasValues = static_cast<std::size_t>(ext[0] * ext[1] * ext[2]);

                // fuse every (c, t) with the one layout; the registration volumes
                // are still loaded, so their (c, t) goes first without a re-read
                Index loadedC = channel, loadedT = refT;
                const double total = static_cast<double>(std::max<Index>(1, d.c * d.t));
                Index done = 0;
                auto fuse = [&](Index c, Index t) {
                    const double from = 0.2 + 0.8 * static_cast<double>(done) / total, to = 0.2 + 0.8 * static_cast<double>(done + 1) / total;
                    if (c != loadedC || t != loadedT) {
                        load(c, t, from, from + 0.5 * (to - from));
                        loadedC = c;
                        loadedT = t;
                    }
                    ctx.report(from + 0.5 * (to - from), "fusing c" + std::to_string(c + 1) + " t" + std::to_string(t + 1));
                    Buffer<float> fused = fuseTiles<float>(views, layout.positions, layout.canvasOrigin, layout.canvasExtent, options);
                    if (static_cast<std::size_t>(fused.shape().numel()) != canvasValues)
                        throw std::runtime_error("Stitch: fused " + std::to_string(fused.shape().numel()) + " values for a canvas of " + std::to_string(canvasValues));
                    std::memcpy(mosaic->plane(c, t, 0), fused.data(), canvasValues * sizeof(float));
                    ++done;
                    ctx.throwIfCancelled();
                };
                fuse(channel, refT);
                for (Index c = 0; c < d.c; ++c)
                    for (Index t = 0; t < d.t; ++t)
                        if (c != channel || t != refT) fuse(c, t);
                out.array = mosaic;
                out.ranOn = Backend::Cpu;

                AlignmentInfo a;
                tileGrid(meta.tiles, a);
                a.highlightedTile = cellOf(a, meta.tiles[options.anchorTile < n ? options.anchorTile : 0].name);
                const ShiftStats stats = shiftStats(layout);
                out.diagnostics = diagnose(summary(p, meta), std::move(a), layout, stats, *out.array, channel, refT,
                                           [&](std::size_t i) { return meta.tiles[i].name; });
                out.note = std::to_string(n) + " tiles · " + std::to_string(stats.accepted) + " pairs · c" + std::to_string(d.c) +
                           " t" + std::to_string(d.t) + " · CPU";
                ctx.report(1.0, "");
                return out;
            }

            // --- diagnostics shared by both modes -----------------------------------

            Diagnostics diagnose(std::string summaryText, AlignmentInfo a, const StitchLayout& layout, const ShiftStats& stats,
                                 const Array5& mosaic, Index c, Index t,
                                 const std::function<std::string(std::size_t)>& nameOf = {}) const {
                auto name = [&](std::size_t i) { return nameOf ? nameOf(i) : "t" + std::to_string(i + 1); };
                Diagnostics d;
                d.kind = DiagnosticsKind::Alignment;
                d.summary = std::move(summaryText);
                a.shiftStats.push_back({"Pairs", std::to_string(stats.accepted) + " / " + std::to_string(layout.matches.size())});
                a.shiftStats.push_back({"Mean |Δ|", stats.accepted ? formatNumber(stats.meanShift, 1) + " px" : "—"});
                a.shiftStats.push_back({"Max |Δ|", stats.accepted ? formatNumber(stats.maxShift, 1) + " px" : "—"});
                a.shiftStats.push_back({"NCC", stats.accepted ? formatNumber(stats.meanNcc, 2) : "—"});
                d.alignment = std::move(a);
                d.facts = d.alignment->shiftStats;
                DiagnosticTab tab{"Alignment", {}};
                const Dims5& od = mosaic.dims();
                tab.images.push_back(d.addImage(thumbnail(mosaic.plane(c, t, od.z / 2), od.y, od.x, 512, "Mosaic", od.toString())));
                d.tabs.push_back(std::move(tab));
                DiagnosticTable table;
                table.caption = "Pairwise shifts · measured minus nominal";
                table.header = {"Fixed", "Moving", "Δz", "Δy", "Δx", "NCC"};
                for (const TileMatch& m : layout.matches) {
                    if (!m.accepted) continue;
                    table.rows.push_back({name(m.fixed), name(m.moving),
                                          formatNumber(m.displacement[0] - m.nominalDisplacement[0], 1),
                                          formatNumber(m.displacement[1] - m.nominalDisplacement[1], 1),
                                          formatNumber(m.displacement[2] - m.nominalDisplacement[2], 1),
                                          formatNumber(m.correlation, 2)});
                }
                d.table = std::move(table);
                return d;
            }

            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeStitchOperation() { return std::make_unique<StitchOperation>(); }

} // namespace sirius::app
