// Import labels: a label volume read from a TIFF becomes the labels of the
// input, which passes through unchanged. The file is what Export labels /
// export_labels / export_result's labels sidecar write -- one page per plane,
// t*z pages for a time series, integer pixels, 0 = background -- so a label
// map saved from one session (or made elsewhere) can be loaded back, reviewed
// and edited in the viewer like a segmentation's own, and saved again.
//
// The step is a step so that the pipeline records where the labels came from
// and a saved pipeline reloads them; every edit made on top of it is an
// ordinary undoable label edit on this step's output.
#include "core/ops/common.hpp"
#include "core/ops/builtin.hpp"

#include <sirius/tiff_io.hpp>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace sirius::app {

    namespace {

        std::string fileName(const std::string& path) {
            std::error_code ec;
            const std::filesystem::path p = std::filesystem::u8path(path);
            const std::string leaf = p.filename().u8string();
            return leaf.empty() ? path : leaf;
        }

        class ImportLabelsOperation final : public Operation {
        public:
            ImportLabelsOperation() {
                info_.kind = "import_labels";
                info_.name = "Import labels";
                info_.group = "Segment";
                info_.kindLabel = "SEGMENT";
                info_.diagnostics = DiagnosticsKind::Segment;
                info_.defaultCache = CachePolicy::Memory;
                info_.producesLabels = true;
                info_.helpPage = "import_labels";
                info_.params = {
                    pathParam("path", "Labels file")
                        .withFilter("Label TIFF (*.tif *.tiff);;All files (*)")
                        .withHelp("A TIFF of integer labels, 0 = background: one page per plane of the input, t × z pages for a "
                                  "time series (what Export labels writes)."),
                    intParam("min_voxels", "Min. voxels", 0).range(0, 1000000000).withHelp("Drop labels smaller than this; 0 keeps every one."),
                    boolParam("relabel", "Relabel densely", false).withHelp("Number the labels 1 … n in the order of their ids."),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            std::string summary(const ParamSet& p, const DatasetMeta&) const override {
                const std::string path = p.getString("path");
                return joinSummary({path.empty() ? "no file" : fileName(path), p.getInt("min_voxels", 0) > 0 ? "min " + std::to_string(p.getInt("min_voxels")) + " voxels" : "",
                                    p.getBool("relabel") ? "relabel" : ""});
            }

            Validation validate(const ParamSet& p, const DatasetMeta& in) const override {
                Validation v = Operation::validate(p, in);
                const std::string path = p.getString("path");
                if (path.empty()) {
                    v.errors.push_back("Name the label TIFF to import.");
                    return v;
                }
                std::error_code ec;
                if (!std::filesystem::is_regular_file(std::filesystem::u8path(path), ec)) {
                    v.errors.push_back("The labels file " + path + " is not there.");
                    return v;
                }
                // The page count against the input: told before the run, as
                // every other mismatch a step can see from the metadata.
                try {
                    const TiffStackShape shape = inspectTiffShape(path);
                    const Index pages = static_cast<Index>(shape.pages);
                    const Index want = in.dims.t * in.dims.z;
                    if (pages != want && pages != in.dims.z)
                        v.errors.push_back("The labels file has " + std::to_string(pages) + " page(s); the input has " + std::to_string(in.dims.z) +
                                           " plane(s)" + (in.dims.t > 1 ? " × " + std::to_string(in.dims.t) + " time points = " + std::to_string(want) + " pages" : "") + ".");
                    else if (static_cast<Index>(shape.width) != in.dims.x || static_cast<Index>(shape.height) != in.dims.y)
                        v.errors.push_back("The labels file is " + std::to_string(shape.width) + " × " + std::to_string(shape.height) + " pixels; the input is " +
                                           std::to_string(in.dims.x) + " × " + std::to_string(in.dims.y) + ".");
                    else if (pages == in.dims.z && in.dims.t > 1)
                        v.warnings.push_back("The labels file holds one time point; it is used for every one of the " + std::to_string(in.dims.t) + ".");
                    if (shape.pixelType == PixelType::Float32 || shape.pixelType == PixelType::Float64)
                        v.errors.push_back("The labels file has floating-point pixels; labels are integers.");
                    if (shape.samplesPerPixel != 1) v.errors.push_back("The labels file has " + std::to_string(shape.samplesPerPixel) + " samples per pixel; labels have one.");
                } catch (const std::exception& e) {
                    v.errors.push_back(std::string("The labels file does not read as a TIFF: ") + e.what());
                }
                return v;
            }

            DatasetMeta outputMeta(const ParamSet&, const DatasetMeta& input) const override { return input; }

            StepOutput run(const StepInput& input, const ParamSet& p, const StepContext& ctx) const override {
                const DatasetMeta& meta = input.meta;
                const Validation v = validate(p, meta);
                if (!v.ok()) throw std::runtime_error(v.firstError());
                const std::string path = p.getString("path");
                ctx.report(0.05, "reading " + fileName(path));
                TiffFile file(path);
                TiffReadOptions opts;
                // Every integer type is read as uint32: a label is an id, not a measurement.
                const Buffer<std::uint32_t> pages = file.readStack<std::uint32_t>(opts);
                ctx.throwIfCancelled();
                const Index nt = meta.dims.t, nz = meta.dims.z, ny = meta.dims.y, nx = meta.dims.x;
                const Index got = static_cast<Index>(pages.shape()[0]);
                const bool perFrame = got == nt * nz;
                if (!perFrame && got != nz) throw std::runtime_error("the labels file has " + std::to_string(got) + " pages, the input " + std::to_string(nt * nz));
                if (static_cast<Index>(pages.shape()[1]) != ny || static_cast<Index>(pages.shape()[2]) != nx)
                    throw std::runtime_error("the labels file's planes are " + std::to_string(pages.shape()[2]) + " × " + std::to_string(pages.shape()[1]) +
                                             ", the input's " + std::to_string(nx) + " × " + std::to_string(ny));
                auto labels = std::make_shared<LabelVolume>(nt, nz, ny, nx);
                const Index minVoxels = p.getInt("min_voxels", 0);
                for (Index t = 0; t < nt; ++t) {
                    ctx.throwIfCancelled();
                    ctx.report(0.2 + 0.6 * static_cast<double>(t) / nt, "t " + std::to_string(t));
                    std::uint32_t* vol = labels->volume(t);
                    const std::uint32_t* src = pages.data() + (perFrame ? t : 0) * nz * ny * nx;
                    std::memcpy(vol, src, static_cast<std::size_t>(nz * ny * nx) * sizeof(std::uint32_t));
                    if (minVoxels > 0) dropSmall(vol, labels->volumeSize(), minVoxels);
                }
                labels->resetMaxLabel();
                if (p.getBool("relabel", false)) labels->relabelDensely();
                for (Index t = 0; t < nt; ++t) {
                    ctx.throwIfCancelled();
                    ctx.report(0.8 + 0.2 * static_cast<double>(t) / nt, "statistics");
                    labels->recomputeStats(t);
                    labels->applyFlags(LabelFlagRules{});
                }
                StepOutput out;
                out.meta = meta;
                out.array = input.array;
                out.source = input.source;
                out.labels = labels;
                out.ranOn = Backend::Cpu;
                out.note = std::to_string(labels->stats().size()) + " labels from " + fileName(path) + " · CPU";
                out.diagnostics = labelDiagnostics(*labels, summary(p, meta) + " · " + std::to_string(labels->stats().size()) + " labels");
                ctx.report(1.0, "");
                return out;
            }

        private:
            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeImportLabelsOperation() { return std::make_unique<ImportLabelsOperation>(); }

} // namespace sirius::app
