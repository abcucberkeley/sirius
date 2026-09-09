// Flat-field: (I - dark) / (flat - dark), normalised so the mean stays put.
// One flat (and dark) image, or one page per channel when the file has as
// many pages as the input has channels.
#include "core/ops/builtin.hpp"

#include <cmath>
#include <filesystem>

#include <sirius/image_ops.hpp>
#include <sirius/tiff_io.hpp>

namespace sirius::app {

    namespace {

        class FlatFieldOperation final : public Operation {
        public:
            FlatFieldOperation() {
                info_.kind = "flatfield";
                info_.name = "Flat-field";
                info_.group = "Intensity";
                info_.kindLabel = "INTENSITY";
                info_.defaultCache = CachePolicy::Recompute;
                info_.separableOverT = true;
                info_.helpPage = "flatfield";
                info_.params = {
                    pathParam("flat", "Flat image").withFilter("TIFF (*.tif *.tiff);;All files (*)").withHelp("Illumination profile; one page, or one page per channel"),
                    pathParam("dark", "Dark image").withFilter("TIFF (*.tif *.tiff);;All files (*)").withHelp("Camera offset (optional)"),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            std::string summary(const ParamSet& params, const DatasetMeta&) const override {
                const std::string flat = params.getString("flat"), dark = params.getString("dark");
                if (flat.empty()) return "no flat image";
                return joinSummary({std::filesystem::path(flat).filename().string(),
                                    dark.empty() ? "no dark" : "dark " + std::filesystem::path(dark).filename().string()});
            }

            Validation validate(const ParamSet& params, const DatasetMeta& input) const override {
                Validation v = Operation::validate(params, input);
                const std::string flat = params.getString("flat"), dark = params.getString("dark");
                if (flat.empty()) v.errors.push_back("Choose a flat image.");
                else if (!std::filesystem::exists(flat)) v.errors.push_back("Flat image not found: " + flat);
                else {
                    try {
                        const TiffInfo info = inspectTiff(flat);
                        if (static_cast<Index>(info.width()) != input.dims.x || static_cast<Index>(info.height()) != input.dims.y)
                            v.errors.push_back("The flat image is " + std::to_string(info.width()) + " × " +
                                               std::to_string(info.height()) + ", the data " + std::to_string(input.dims.x) +
                                               " × " + std::to_string(input.dims.y) + ".");
                    } catch (const std::exception& e) {
                        v.errors.push_back(std::string("Cannot read the flat image: ") + e.what());
                    }
                }
                if (!dark.empty() && !std::filesystem::exists(dark)) v.errors.push_back("Dark image not found: " + dark);
                return v;
            }

            DatasetMeta outputMeta(const ParamSet&, const DatasetMeta& input) const override {
                DatasetMeta out = input;
                out.sourceType = PixelType::Float32;
                return out;
            }

            StepOutput run(const StepInput& input, const ParamSet& params, const StepContext& ctx) const override {
                const Validation v = validate(params, input.meta);
                if (!v.ok()) throw std::runtime_error(v.firstError());
                const DatasetMeta& meta = input.meta;
                TiffReadOptions opts;
                Buffer<float> flat = TiffFile(params.getString("flat")).readStack<float>(opts);
                Buffer<float> dark;
                if (!params.getString("dark").empty()) dark = TiffFile(params.getString("dark")).readStack<float>(opts);
                const Index planeSize = meta.dims.planeSize();
                auto page = [&](const Buffer<float>& b, Index c) -> const float* {
                    if (b.empty()) return nullptr;
                    const Index pages = b.dim(0);
                    return b.data() + (pages == meta.dims.c ? c : 0) * planeSize;
                };
                StepOutput out;
                out.meta = outputMeta(params, meta);
                auto result = allocateLike(meta);
                forEachVolume(meta, ctx, [&](Index c, Index t) {
                    Buffer<float> vol = input.readVolume(c, t);
                    flatField(vol.data(), meta.dims.z, planeSize, page(flat, c), page(dark, c));
                    copy(vol, result->volume(c, t));
                });
                out.array = result;
                out.labels = input.labels ? input.labels->clone() : nullptr;
                out.ranOn = Backend::Cpu;
                out.note = summary(params, meta) + " · CPU";
                out.diagnostics = genericDiagnostics(input, out, summary(params, meta));
                return out;
            }

        private:
            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeFlatFieldOperation() { return std::make_unique<FlatFieldOperation>(); }

} // namespace sirius::app
