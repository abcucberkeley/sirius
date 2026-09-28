// File ▸ Export result…: the design's 640 px dialog. Format list on the
// left (TIFF variants, zarr, N5, raw) with a note and a size estimate;
// on the right the source step, the t / z / c range, the pixel type and
// scaling rule, the container's own knobs (compression, tiles, pyramid,
// chunks, codec, sharding), the destination and the sidecar options.

#include "imgui/dialogs/dialogs.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>

#include "core/export.hpp"
#include "imgui/dialogs/export_dialog_support.hpp"
#include "imgui/platform.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui {

    namespace {

        using namespace dialog_support;
        using theme::px;

        struct FormatRow {
            const char* name;
            const char* note;
            ExportFormat format;
            bool tiled, pyramid;   // presets applied when the row is chosen
            int pyramidLevels;
        };

        const std::vector<FormatRow>& formatRows() {
            static const std::vector<FormatRow> rows = {
                {"OME-TIFF", "Single file · broad compatibility", ExportFormat::Tiff, false, false, 1},
                {"Tiled TIFF", "512² tiles · random access", ExportFormat::Tiff, true, false, 1},
                {"Pyramidal OME-TIFF", "Multi-resolution · viewers & QuPath", ExportFormat::Tiff, true, true, 5},
                {"OME-Zarr", "Chunked · cloud & Dask friendly", ExportFormat::Zarr, true, true, 5},
                {"N5", "BigDataViewer · Fiji", ExportFormat::N5, true, true, 5},
                {"Raw float32", "No metadata · fastest", ExportFormat::Raw, false, false, 1},
            };
            return rows;
        }

        const std::vector<PixelType>& pixelTypes() {
            static const std::vector<PixelType> types = {PixelType::UInt8, PixelType::Int8, PixelType::UInt16, PixelType::Int16,
                                                         PixelType::UInt32, PixelType::Int32, PixelType::Float32, PixelType::Float64};
            return types;
        }

        const std::vector<std::string>& codecs() {
            static const std::vector<std::string> names = {"blosc-zstd", "blosc-lz4", "zstd", "gzip", "none"};
            return names;
        }

        // Strict: the whole of the trimmed text has to be a number.
        bool parseInt(const std::string& text, long long& value) {
            const std::string t = trimmed(text);
            if (t.empty()) return false;
            char* end = nullptr;
            value = std::strtoll(t.c_str(), &end, 10);
            return end != nullptr && *end == '\0';
        }

        void rowTooltip(const std::string& text) {
            ImGui::PushStyleColor(ImGuiCol_Border, theme::kText);
            ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, theme::crispPen(theme::kBorder));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, px(8, 4));
            if (ImGui::BeginTooltip()) {
                widgets::text(text, 12);
                ImGui::EndTooltip();
            }
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor();
        }

        class ExportDialog : public Dialog {
        public:
            ExportDialog(App& app, std::function<void(int, const ExportOptions&)> accepted) : accepted_(std::move(accepted)) {
                const Workbench& wb = app.wb();
                step_ = std::max(0, wb.viewedIndex());

                // defaults from the dataset
                const DatasetMeta& ds = wb.dataset();
                std::string base = ds.name;
                if (base.empty()) base = "result";
                std::string dir = ds.sourcePath.empty() ? std::string() : parentPath(ds.sourcePath);
                if (dir.empty() || dir == ".") dir = platform::homeDirectory();
#ifdef _WIN32
                dir = replaceAll(dir, "\\", "/");   // forward slashes throughout
#endif
                destination_ = dir + "/" + base + "_processed";
                selectFormat(0);
            }

            std::string title() const override { return "Export result"; }
            // 640 px is the design's width; the height follows the options the
            // chosen container has.
            ImVec2 size() const override { return ImVec2(640, 0); }

            void draw(App& app) override {
                const Workbench& wb = app.wb();
                const Pipeline& p = wb.pipeline();
                step_ = std::clamp(step_, 0, std::max(0, p.size() - 1));
                DatasetMeta meta = wb.outputMetaOf(step_);
                const std::shared_ptr<const StepOutput> out = wb.output(step_);
                if (out) meta = out->meta;
                followStep(meta.dims);
                labelsAvailable_ = out && out->labels;

                widgets::vspace(6);
                ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, px(9, 0));
                if (ImGui::BeginTable("##exportBody", 2, ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_NoSavedSettings)) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    drawFormats(meta.dims);
                    ImGui::TableNextColumn();
                    drawOptions(app, meta.dims, out != nullptr);
                    ImGui::EndTable();
                }
                ImGui::PopStyleVar();

                widgets::vspace(6);
                const ExportOptions o = options();
                const std::string problem = validateExport(o, meta.dims);
                const bool enabled = problem.empty() && !trimmed(destination_).empty();
                switch (actionRow(std::string("Export ") + formatRows()[static_cast<std::size_t>(format_)].name, enabled)) {
                    case Action::Cancel: close(); break;
                    case Action::Accept:
                        if (accepted_) accepted_(step_, o);
                        close();
                        break;
                    case Action::None: break;
                }
            }

        private:
            void selectFormat(int i) {
                if (i < 0 || i >= static_cast<int>(formatRows().size())) return;
                const FormatRow& f = formatRows()[static_cast<std::size_t>(i)];
                if (!exportFormatAvailable(f.format)) return;
                format_ = i;
                tiled_ = f.tiled;
                pyramid_ = f.pyramid ? f.pyramidLevels : 1;
                downsample_ = 2;
                omeXml_ = std::string(f.name).find("OME") != std::string::npos;
                // keep the destination's stem, swap the extension
                std::string dest = destination_;
                for (const char* ext : {".ome.tif", ".ome.tiff", ".tif", ".tiff", ".zarr", ".n5", ".raw"})
                    if (endsWithNoCase(dest, ext)) {
                        dest.resize(dest.size() - std::string(ext).size());
                        break;
                    }
                destination_ = dest + exportExtension(options());
            }

            // The t / z range against the extents of the chosen step.
            void followStep(const Dims5& dims) {
                const std::int64_t t = static_cast<std::int64_t>(dims.t), z = static_cast<std::int64_t>(dims.z);
                // the whole of the chosen step until the user narrows it; a range
                // the user set is kept (clamped to the new step) when the step changes
                if (!tEdited_) t0_ = 0;
                if (!tEdited_ || t1_ == 0) t1_ = t;
                if (!zEdited_) z0_ = 0;
                if (!zEdited_ || z1_ == 0) z1_ = z;
                t0_ = std::clamp<std::int64_t>(t0_, 0, std::max<std::int64_t>(t - 1, 0));
                t1_ = std::clamp<std::int64_t>(t1_, 0, t);
                z0_ = std::clamp<std::int64_t>(z0_, 0, std::max<std::int64_t>(z - 1, 0));
                z1_ = std::clamp<std::int64_t>(z1_, 0, z);
            }

            void drawFormats(const Dims5& dims) {
                {
                    const Spacing s(8, 6);
                    widgets::caption("Format");
                }
                const Spacing s(8, 2);
                ImDrawList* dl = ImGui::GetWindowDrawList();
                for (int i = 0; i < static_cast<int>(formatRows().size()); ++i) {
                    const FormatRow& f = formatRows()[static_cast<std::size_t>(i)];
                    const bool available = exportFormatAvailable(f.format);
                    const bool selected = i == format_;
                    ExportOptions probe = options();
                    probe.format = f.format;
                    probe.tiff.tiled = f.tiled;
                    probe.tiff.pyramidLevels = f.pyramid ? f.pyramidLevels : 1;
                    probe.zarr.pyramidLevels = f.pyramid ? f.pyramidLevels : 1;
                    const std::string size = bytesText(estimateExportBytes(dims, probe));

                    widgets::RowOpts ro;
                    ro.selected = selected;
                    ro.hoverable = false;
                    ro.topRule = 0.0f;
                    const widgets::Row row = widgets::beginRow(f.name, 50, ro);
                    const float opacity = available ? 1.0f : 0.45f;
                    const float sizeW = theme::textSize(size, 11).x;
                    const float textW = (row.max.x - row.min.x) - px(10) * 3 - sizeW;
                    widgets::drawText(dl, ImVec2(row.min.x + px(10), row.min.y + px(8)), widgets::elideText(f.name, textW, 13, theme::Weight::ExtraBold),
                                      13, theme::withAlpha(theme::kText, opacity), theme::Weight::ExtraBold);
                    widgets::drawText(dl, ImVec2(row.min.x + px(10), row.min.y + px(27)),
                                      widgets::elideText(available ? f.note : "not available in this build", textW, 11), 11,
                                      theme::withAlpha(theme::kNeutral600, opacity));
                    widgets::drawText(dl, ImVec2(row.max.x - px(10) - sizeW, row.min.y + px(9)), size, 11,
                                      theme::withAlpha(theme::kNeutral600, opacity));
                    if (selected) widgets::crispRect(dl, row.min, row.max, theme::kAccent, theme::kBorder);
                    if (available && row.hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                    if (!available && row.hovered) rowTooltip("Rebuild with SIRIUS_ENABLE_TENSORSTORE=ON for zarr / N5");
                    if (row.clicked) selectFormat(i);
                    widgets::endRow(row);
                }
            }

            void drawOptions(App& app, const Dims5& dims, bool computed) {
                const Workbench& wb = app.wb();
                const Pipeline& p = wb.pipeline();
                const Spacing spacing(8, 12);
                const ExportOptions o = options();
                const bool tiff = o.format == ExportFormat::Tiff;
                const bool zarr = o.format == ExportFormat::Zarr || o.format == ExportFormat::N5;

                {
                    std::vector<std::string> steps;
                    for (int s = 0; s < p.size(); ++s) {
                        std::string label = Step::number(s) + " " + p.at(s).name;
                        if (!wb.output(s)) label += "  (not computed)";
                        else if (!wb.outputFresh(s)) label += "  (out of date)";
                        steps.push_back(std::move(label));
                    }
                    const Field f("From step");
                    widgets::combo("##step", &step_, steps);
                }
                // The file gets the step's last output, while the sidecar records the
                // pipeline as it is now: after a parameter edit or an undo the two do
                // not belong together, and nothing said so.
                if (computed && !wb.outputFresh(step_))
                    note("The parameters changed since step " + Step::number(step_) +
                             " was computed: the export writes that earlier "
                             "result, and a pipeline sidecar would record the current parameters, which did not "
                             "produce it. Run the step again for a matching pair.",
                         theme::kAccentText);

                {
                    const Field f("Range");
                    const Spacing tight(6, 4);
                    const float w = columnWidth(4, 6);
                    const std::int64_t t = static_cast<std::int64_t>(dims.t), z = static_cast<std::int64_t>(dims.z);
                    if (prefixedInt("##t0", "t", &t0_, 0, std::max<std::int64_t>(t - 1, 0), w)) tEdited_ = true;
                    ImGui::SameLine();
                    if (prefixedInt("##t1", "to", &t1_, 0, t, w)) tEdited_ = true;
                    ImGui::SameLine();
                    if (prefixedInt("##z0", "z", &z0_, 0, std::max<std::int64_t>(z - 1, 0), w)) zEdited_ = true;
                    ImGui::SameLine();
                    if (prefixedInt("##z1", "to", &z1_, 0, z, w)) zEdited_ = true;
                    widgets::FieldOpts c;
                    c.hint = "all channels (or 0, 2)";
                    widgets::inputText("##channels", &channels_, c);
                }

                {
                    const float w = columnWidth(2, 10);
                    widgets::FieldOpts fo;
                    fo.width = design(w);
                    {
                        std::vector<std::string> names;
                        for (PixelType t : pixelTypes()) names.emplace_back(toString(t));
                        const Field f("Dtype");
                        widgets::combo("##dtype", &dtype_, names, fo);
                    }
                    ImGui::SameLine(0.0f, px(10));
                    {
                        const Field f("Scaling");
                        widgets::combo("##scaling", &scaling_,
                                       {"cast (no rescale)", "rescale min – max", "rescale fixed range", "rescale percentiles"}, fo);
                    }
                    if (o.scaling == ExportScaling::FixedRange || o.scaling == ExportScaling::Percentile) {
                        {
                            const Field f("Low");
                            widgets::inputDouble("##rangeLo", &rangeLo_, -1e12, 1e12, 1.0, 3, fo);
                        }
                        ImGui::SameLine(0.0f, px(10));
                        const Field f("High");
                        widgets::inputDouble("##rangeHi", &rangeHi_, -1e12, 1e12, 1.0, 3, fo);
                    }
                }

                if (tiff) {
                    const Spacing box(8, 8);
                    const float w = columnWidth(2, 10);
                    widgets::FieldOpts fo;
                    fo.width = design(w);
                    {
                        const Field f("Compression");
                        widgets::combo("##compression", &compression_, {"none", "LZW", "Deflate (zlib)"}, fo);
                    }
                    ImGui::SameLine(0.0f, px(10));
                    {
                        const Field f(" ");
                        prefixedInt("##level", "level", &level_, 1, 9, w, 1, o.tiff.compression == TiffCompression::Deflate);
                    }
                    {
                        const Spacing row(6, 8);
                        checkboxOnFieldLine("Tiled", &tiled_);
                        ImGui::SameLine();
                        const float cross = theme::textSize(std::string("×"), 12).x;
                        const float each = std::max(px(60), std::floor((ImGui::GetContentRegionAvail().x - cross - 2 * px(6)) * 0.5f));
                        widgets::FieldOpts tile;
                        tile.width = design(each);
                        tile.enabled = tiled_;
                        spinInt("##tileW", &tileW_, 16, 8192, 16, tile);
                        ImGui::SameLine();
                        const ImVec2 at = ImGui::GetCursorScreenPos();
                        widgets::drawTextIn(ImGui::GetWindowDrawList(), at, ImVec2(at.x + cross, at.y + px(theme::kInputH)), "×", 12,
                                            theme::kNeutral600);
                        ImGui::Dummy(ImVec2(cross, px(theme::kInputH)));
                        ImGui::SameLine();
                        spinInt("##tileH", &tileH_, 16, 8192, 16, tile);
                    }
                    widgets::checkbox("Predictor", &predictor_, o.tiff.compression != TiffCompression::None);
                    ImGui::SameLine(0.0f, px(16));
                    widgets::checkbox("BigTIFF", &bigTiff_);
                    ImGui::SameLine(0.0f, px(16));
                    widgets::checkbox("OME-XML", &omeXml_);
                }

                // pyramid box (TIFF + zarr)
                if (tiff || zarr) {
                    const float w = columnWidth(2, 10);
                    {
                        // one level is no pyramid, and the label says so
                        const Field f(pyramid_ <= 1 ? "Pyramid levels (none)" : "Pyramid levels");
                        widgets::FieldOpts fo;
                        fo.width = design(w);
                        widgets::inputInt("##pyramid", &pyramid_, 1, 12, 1, fo);
                    }
                    ImGui::SameLine(0.0f, px(10));
                    const Field f("Downsample");
                    prefixedInt("##downsample", "÷", &downsample_, 2, 8, w);
                }

                if (zarr) {
                    const Spacing box(8, 8);
                    const float w = columnWidth(2, 10);
                    widgets::FieldOpts fo;
                    fo.width = design(w);
                    {
                        const Field f("Store");
                        widgets::FieldOpts store = fo;
                        store.enabled = o.format == ExportFormat::Zarr;
                        widgets::combo("##zarrVersion", &zarrVersion_, {"zarr v3", "zarr v2"}, store);
                    }
                    ImGui::SameLine(0.0f, px(10));
                    {
                        const Field f("Chunk (c, t, z, y, x)");
                        widgets::inputText("##chunk", &chunk_, fo);
                        widgets::tooltip("Chunk shape over c, t, z, y, x");
                    }
                    {
                        const Field f("Codec");
                        widgets::combo("##codec", &codec_, codecs(), fo);
                    }
                    ImGui::SameLine(0.0f, px(10));
                    {
                        const Field f(" ");
                        prefixedInt("##zarrLevel", "level", &zarrLevel_, 0, 22, w);
                    }
                    widgets::checkbox("Shard chunks (zarr v3)", &shard_, o.format == ExportFormat::Zarr && o.zarr.zarrVersion == 3);
                    widgets::checkbox("OME-NGFF multiscales metadata", &ngff_);
                }

                {
                    const Field f("Destination");
                    const Spacing row(6, 8);
                    widgets::ButtonOpts b;
                    b.small = true;
                    const float browseW = theme::snap(theme::textSize(std::string("Browse"), 12, theme::Weight::SemiBold).x + 2 * px(10) +
                                                      2 * px(theme::kBorder));
                    widgets::FieldOpts fo;
                    fo.width = design(std::max(px(60), ImGui::GetContentRegionAvail().x - browseW - px(6)));
                    widgets::inputText("##destination", &destination_, fo);
                    ImGui::SameLine();
                    // the small button on the middle of the field's line
                    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + std::floor((px(theme::kInputH) - px(14 + 2 * 4 + 2 * theme::kBorder)) * 0.5f));
                    if (widgets::button("Browse", b)) browse();
                }
                {
                    const Spacing checks(8, 8);
                    widgets::checkbox("Include pipeline sidecar (.pipeline.toml)", &sidecarPipeline_);
                    widgets::checkbox("Include labels sidecar", &sidecarLabels_, labelsAvailable_);
                }
                const std::string problem = validateExport(o, dims);
                if (!problem.empty()) note(problem, theme::kAccentText);
            }

            void browse() {
                const ExportOptions o = options();
                const bool dir = o.format == ExportFormat::Zarr || o.format == ExportFormat::N5;
                const std::string current = trimmed(destination_);
                const std::string start = current.empty() ? std::string() : parentPath(current);
                const std::string chosen = dir ? platform::saveFileDialog("Export store", start, fileName(current), {{"Stores", "zarr,n5"}})
                                               : platform::saveFileDialog("Export file", start, fileName(current));
                if (!chosen.empty()) destination_ = chosen;
            }

            ExportOptions options() const {
                ExportOptions o;
                const FormatRow& f = formatRows()[static_cast<std::size_t>(format_)];
                o.format = f.format;
                o.path = trimmed(destination_);
                o.dtype = pixelTypes()[static_cast<std::size_t>(std::clamp(dtype_, 0, static_cast<int>(pixelTypes().size()) - 1))];
                static const ExportScaling scalings[] = {ExportScaling::Cast, ExportScaling::MinMax, ExportScaling::FixedRange,
                                                         ExportScaling::Percentile};
                o.scaling = scalings[std::clamp(scaling_, 0, 3)];
                if (o.scaling == ExportScaling::FixedRange) {
                    o.rangeLo = rangeLo_;
                    o.rangeHi = rangeHi_;
                } else if (o.scaling == ExportScaling::Percentile) {
                    o.percentileLo = rangeLo_;
                    o.percentileHi = rangeHi_;
                }
                o.range.t0 = static_cast<Index>(t0_);
                o.range.t1 = t1_ > 0 ? static_cast<Index>(t1_) : -1;
                o.range.z0 = static_cast<Index>(z0_);
                o.range.z1 = z1_ > 0 ? static_cast<Index>(z1_) : -1;
                for (const std::string& part : split(channels_, ',', true)) {
                    long long c = 0;
                    if (parseInt(part, c)) o.range.channels.push_back(static_cast<Index>(c));
                }
                o.tiff.tiled = tiled_;
                o.tiff.tileWidth = static_cast<int>(tileW_);
                o.tiff.tileHeight = static_cast<int>(tileH_);
                static const TiffCompression compressions[] = {TiffCompression::None, TiffCompression::Lzw, TiffCompression::Deflate};
                o.tiff.compression = compressions[std::clamp(compression_, 0, 2)];
                o.tiff.compressionLevel = static_cast<int>(level_);
                o.tiff.predictor = predictor_;
                o.tiff.bigTiff = bigTiff_;
                o.tiff.omeXml = omeXml_;
                o.tiff.pyramidLevels = static_cast<int>(pyramid_);
                o.tiff.downsample = static_cast<int>(downsample_);
                o.zarr.zarrVersion = zarrVersion_ == 1 ? 2 : 3;
                {
                    const std::vector<std::string> parts = split(chunk_, ',', true);
                    for (std::size_t i = 0; i < 5 && i < parts.size(); ++i) {
                        long long v = 0;
                        if (parseInt(parts[i], v) && v > 0) o.zarr.chunk[i] = static_cast<Index>(v);
                    }
                }
                o.zarr.codec = codecs()[static_cast<std::size_t>(std::clamp(codec_, 0, static_cast<int>(codecs().size()) - 1))];
                o.zarr.level = static_cast<int>(zarrLevel_);
                o.zarr.shard = shard_;
                o.zarr.pyramidLevels = static_cast<int>(pyramid_);
                o.zarr.downsample = static_cast<int>(downsample_);
                o.zarr.omeNgff = ngff_;
                o.includePipeline = sidecarPipeline_;
                o.includeLabels = sidecarLabels_ && labelsAvailable_;
                return o;
            }

            std::function<void(int, const ExportOptions&)> accepted_;
            int format_ = 0;
            int step_ = 0;
            std::int64_t t0_ = 0, t1_ = 0, z0_ = 0, z1_ = 0;
            // Whether the user set the t / z range. Until then it follows the
            // chosen step's extents; it used to keep the first step's, so a
            // switch from a 14-plane step to a 135-plane one exported 0..14.
            bool tEdited_ = false, zEdited_ = false;
            std::string channels_;
            int dtype_ = 6;   // float32
            int scaling_ = 0;
            double rangeLo_ = 0.1, rangeHi_ = 99.9;
            int compression_ = 2;   // Deflate
            std::int64_t level_ = 6;
            bool predictor_ = false;
            bool tiled_ = false;
            std::int64_t tileW_ = 512, tileH_ = 512;
            bool bigTiff_ = true;
            bool omeXml_ = true;
            std::int64_t pyramid_ = 1;
            std::int64_t downsample_ = 2;
            int zarrVersion_ = 0;   // zarr v3
            std::string chunk_ = "1, 1, 16, 512, 512";
            int codec_ = 0;
            std::int64_t zarrLevel_ = 3;
            bool shard_ = false;
            bool ngff_ = true;
            std::string destination_;
            bool sidecarPipeline_ = true;
            bool sidecarLabels_ = false;
            bool labelsAvailable_ = false;
        };

    } // namespace

    std::shared_ptr<Dialog> makeExportDialog(App& app, std::function<void(int stepIndex, const ExportOptions& options)> accepted) {
        return std::make_shared<ExportDialog>(app, std::move(accepted));
    }

} // namespace sirius::app::gui
