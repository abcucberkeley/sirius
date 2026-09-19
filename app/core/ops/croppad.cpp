// Crop / pad: a (z, y, x) box, possibly extending past the input (fill).
#include "core/ops/builtin.hpp"

#include <algorithm>
#include <cstdio>

#include <sirius/image_ops.hpp>

namespace sirius::app {

    namespace {

        class CropPadOperation final : public Operation {
        public:
            CropPadOperation() {
                info_.kind = "croppad";
                info_.name = "Crop / pad";
                info_.group = "Geometry";
                info_.kindLabel = "GEOMETRY";
                info_.defaultCache = CachePolicy::Recompute;
                info_.separableOverT = true;
                info_.helpPage = "croppad";
                info_.params = {
                    intParam("z0", "Origin z", 0).range(-100000, 100000).withUnit("px").withHelp("May be negative (padding)"),
                    intParam("y0", "Origin y", 0).range(-100000, 100000).withUnit("px"),
                    intParam("x0", "Origin x", 0).range(-100000, 100000).withUnit("px"),
                    intParam("z", "Size z", 0).range(0, 100000).withUnit("px").withHelp("0 = to the edge"),
                    intParam("y", "Size y", 0).range(0, 100000).withUnit("px").withHelp("0 = to the edge"),
                    intParam("x", "Size x", 0).range(0, 100000).withUnit("px").withHelp("0 = to the edge"),
                    doubleParam("fill", "Fill value", 0.0).range(-1e9, 1e9, 1.0, 2),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            struct Box {
                Index z0, y0, x0, z, y, x;
            };
            Box boxOf(const ParamSet& p, const DatasetMeta& in) const {
                Box b{p.getInt("z0"), p.getInt("y0"), p.getInt("x0"), p.getInt("z"), p.getInt("y"), p.getInt("x")};
                if (b.z <= 0) b.z = std::max<Index>(1, in.dims.z - b.z0);
                if (b.y <= 0) b.y = std::max<Index>(1, in.dims.y - b.y0);
                if (b.x <= 0) b.x = std::max<Index>(1, in.dims.x - b.x0);
                return b;
            }

            std::string summary(const ParamSet& p, const DatasetMeta& in) const override {
                const Box b = boxOf(p, in);
                char buf[128];
                std::snprintf(buf, sizeof buf, "z %lld+%lld · y %lld+%lld · x %lld+%lld", static_cast<long long>(b.z0),
                              static_cast<long long>(b.z), static_cast<long long>(b.y0), static_cast<long long>(b.y),
                              static_cast<long long>(b.x0), static_cast<long long>(b.x));
                return buf;
            }

            Validation validate(const ParamSet& p, const DatasetMeta& in) const override {
                Validation v = Operation::validate(p, in);
                const Box b = boxOf(p, in);
                if (b.z0 >= in.dims.z || b.y0 >= in.dims.y || b.x0 >= in.dims.x || b.z0 + b.z <= 0 || b.y0 + b.y <= 0 ||
                    b.x0 + b.x <= 0)
                    v.warnings.push_back("The box does not overlap the data: the output is all fill.");
                return v;
            }

            DatasetMeta outputMeta(const ParamSet& p, const DatasetMeta& in) const override {
                DatasetMeta out = in;
                const Box b = boxOf(p, in);
                out.dims.z = b.z;
                out.dims.y = b.y;
                out.dims.x = b.x;
                if (b.z != in.dims.z) out.sim = SimLayout{};
                out.sourceType = PixelType::Float32;
                return out;
            }

            StepOutput run(const StepInput& input, const ParamSet& p, const StepContext& ctx) const override {
                const DatasetMeta& meta = input.meta;
                const Box b = boxOf(p, meta);
                const float fill = static_cast<float>(p.getDouble("fill", 0.0));
                StepOutput out;
                out.meta = outputMeta(p, meta);
                auto result = allocateLike(out.meta);
                forEachVolume(meta, ctx, [&](Index c, Index t) {
                    Buffer<float> vol = input.readVolume(c, t);
                    cropPad(vol.data(), meta.dims.z, meta.dims.y, meta.dims.x, b.z0, b.y0, b.x0,
                            result->volume(c, t).data(), b.z, b.y, b.x, fill);
                });
                out.array = result;
                if (input.labels && !input.labels->empty()) {
                    auto labels = std::make_shared<LabelVolume>(meta.dims.t, b.z, b.y, b.x);
                    // the same objects on a smaller grid: their classes,
                    // confidences, review marks, flag rules and track ids go along
                    labels->copyAnnotationsFrom(*input.labels);
                    for (Index t = 0; t < meta.dims.t; ++t) {
                        const std::uint32_t* src = input.labels->volume(t);
                        std::uint32_t* dst = labels->volume(t);
                        for (Index z = 0; z < b.z; ++z)
                            for (Index y = 0; y < b.y; ++y)
                                for (Index x = 0; x < b.x; ++x) {
                                    const Index sz = z + b.z0, sy = y + b.y0, sx = x + b.x0;
                                    dst[(z * b.y + y) * b.x + x] =
                                        (sz >= 0 && sy >= 0 && sx >= 0 && sz < meta.dims.z && sy < meta.dims.y && sx < meta.dims.x)
                                            ? src[(sz * meta.dims.y + sy) * meta.dims.x + sx]
                                            : 0u;
                                }
                        labels->recomputeStats(t);   // the boxes and border flags of the new extent
                    }
                    // the table ends on the frame the input's was on
                    if (input.labels->statsT() >= 0 && input.labels->statsT() < meta.dims.t - 1)
                        labels->recomputeStats(input.labels->statsT());
                    // cropping keeps the ids, so tracks stay tracks (a track
                    // wholly outside the box is simply gone from the index)
                    if (input.labels->tracked()) {
                        labels->setTracked(true);
                        labels->setLineage(input.labels->lineage());
                        labels->indexTracks();
                    }
                    out.labels = labels;
                }
                out.ranOn = Backend::Cpu;
                out.note = summary(p, meta) + " · CPU";
                out.diagnostics = genericDiagnostics(input, out, summary(p, meta));
                return out;
            }

        private:
            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeCropPadOperation() { return std::make_unique<CropPadOperation>(); }

} // namespace sirius::app
