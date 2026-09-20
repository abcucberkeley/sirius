// Merge channels: map every channel to a display colour and blend them into
// one RGB image (additive, screen or max) in 0..1.
#include "core/ops/common.hpp"
#include "core/ops/builtin.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <sirius/image_ops.hpp>

namespace sirius::app {

    namespace {

        std::string colorName(const std::array<float, 3>& c) {
            struct Named {
                const char* name;
                std::array<float, 3> rgb;
            };
            static const Named names[] = {
                {"green", {0x63 / 255.f, 0xe0 / 255.f, 0x8a / 255.f}},
                {"orange", {0xff / 255.f, 0x7a / 255.f, 0x5c / 255.f}},
                {"magenta", {0xe8 / 255.f, 0x71 / 255.f, 0xd9 / 255.f}},
                {"blue", {0x7c / 255.f, 0x9c / 255.f, 0xff / 255.f}},
                {"red", {1.f, 0.f, 0.f}},
                {"green", {0.f, 1.f, 0.f}},
                {"blue", {0.f, 0.f, 1.f}},
                {"cyan", {0.f, 1.f, 1.f}},
                {"yellow", {1.f, 1.f, 0.f}},
                {"magenta", {1.f, 0.f, 1.f}},
                {"white", {1.f, 1.f, 1.f}},
                {"gray", {0.5f, 0.5f, 0.5f}},
            };
            const Named* best = nullptr;
            float bestD = 1e9f;
            for (const Named& n : names) {
                float d = 0.f;
                for (int k = 0; k < 3; ++k) d += (n.rgb[static_cast<std::size_t>(k)] - c[static_cast<std::size_t>(k)]) * (n.rgb[static_cast<std::size_t>(k)] - c[static_cast<std::size_t>(k)]);
                if (d < bestD) {
                    bestD = d;
                    best = &n;
                }
            }
            return best ? best->name : "custom";
        }

        class MergeOperation final : public Operation {
        public:
            MergeOperation() {
                info_.kind = "merge";
                info_.name = "Merge channels";
                info_.group = "Combine";
                info_.kindLabel = "COMBINE";
                info_.defaultCache = CachePolicy::Recompute;
                info_.helpPage = "merge";
                ParamSpec colors;
                colors.key = "colors";
                colors.label = "Colours";
                colors.type = ParamType::StringList;
                colors.defaultValue = std::vector<std::string>{};
                colors.help = "One #rrggbb per channel; empty = the channels' own colours";
                info_.params = {
                    choiceParam("blend", "Blend", {"Additive", "Screen", "Max"}, "Additive")
                        .withHelp("Additive is physically faithful; screen avoids clipping; max keeps the brightest channel"),
                    colors,
                    doubleListParam("weights", "Weights", {}).withHelp("Per-channel gain; empty = 1"),
                    doubleParam("normalize_percentile", "Normalize at percentile", 99.9).range(50.0, 100.0, 0.1, 2).withUnit("%").withHelp("Each channel is scaled so this percentile maps to 1 (data already in 0..1 is left alone)").asAdvanced(),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            std::vector<std::array<float, 3>> colorsOf(const ParamSet& p, const DatasetMeta& in) const {
                std::vector<std::array<float, 3>> out;
                const std::vector<std::string> hex = p.getStringList("colors");
                for (Index c = 0; c < in.dims.c; ++c) {
                    std::array<float, 3> col{1.f, 1.f, 1.f};
                    if (static_cast<std::size_t>(c) < hex.size()) {
                        try {
                            col = colorFromHex(hex[static_cast<std::size_t>(c)]);
                        } catch (const std::exception&) {}
                    } else if (static_cast<std::size_t>(c) < in.channels.size()) {
                        col = in.channels[static_cast<std::size_t>(c)].color;
                    }
                    out.push_back(col);
                }
                return out;
            }

            std::string summary(const ParamSet& p, const DatasetMeta& in) const override {
                if (in.rgb) return "already RGB";
                const auto colors = colorsOf(p, in);
                std::string out;
                for (Index c = 0; c < in.dims.c && c < 4; ++c) {
                    if (!out.empty()) out += " · ";
                    const ChannelInfo* ch = static_cast<std::size_t>(c) < in.channels.size() ? &in.channels[static_cast<std::size_t>(c)] : nullptr;
                    out += (ch ? ch->shortName() : "ch " + std::to_string(c)) + " → " + colorName(colors[static_cast<std::size_t>(c)]);
                }
                std::string blend = p.getString("blend", "Additive");
                std::transform(blend.begin(), blend.end(), blend.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
                return joinSummary({out, blend});
            }

            Validation validate(const ParamSet& p, const DatasetMeta& in) const override {
                Validation v = Operation::validate(p, in);
                if (in.rgb) v.errors.push_back("The input is already an RGB merge.");
                if (in.sim.present) v.warnings.push_back("The input is a raw SIM stack; reconstruct it first.");
                return v;
            }

            DatasetMeta outputMeta(const ParamSet&, const DatasetMeta& in) const override {
                DatasetMeta out = in;
                out.rgb = true;
                out.dims.c = 3;
                out.normalizeChannels();
                out.sim = SimLayout{};
                out.sourceType = PixelType::Float32;
                return out;
            }

            StepOutput run(const StepInput& input, const ParamSet& p, const StepContext& ctx) const override {
                const Validation v = validate(p, input.meta);
                if (!v.ok()) throw std::runtime_error(v.firstError());
                const DatasetMeta& meta = input.meta;
                StepOutput out;
                out.meta = outputMeta(p, meta);
                ArrayPtr in = input.materialize([&](double f, const std::string& m) { ctx.report(0.3 * f, m); });
                const Dims5& d = meta.dims;
                const auto colors = colorsOf(p, meta);
                const std::vector<double> weights = p.getDoubleList("weights");
                const std::string blend = p.getString("blend", "Additive");
                const double pct = p.getDouble("normalize_percentile", 99.9);

                // per-channel scale so `pct` maps to 1 (unless already 0..1)
                std::vector<float> scale(static_cast<std::size_t>(d.c), 1.0f);
                const Index channelSize = d.t * d.z * d.planeSize();
                for (Index c = 0; c < d.c; ++c) {
                    ctx.throwIfCancelled();
                    const float* ch = in->plane(c, 0, 0);
                    float mn = std::numeric_limits<float>::infinity(), mx = -mn;
                    for (Index i = 0; i < channelSize; ++i) {
                        const float x = ch[i];
                        if (std::isnan(x)) continue;
                        mn = std::min(mn, x);
                        mx = std::max(mx, x);
                    }
                    if (mn >= 0.0f && mx <= 1.0f) continue;
                    const float hi = percentiles(ch, channelSize, 0.0, pct).second;
                    scale[static_cast<std::size_t>(c)] = hi > 0.0f ? 1.0f / hi : 1.0f;
                }
                auto result = std::make_shared<Array5>(Array5::zeros(out.meta.dims));
                const Index planes = d.t * d.z;
                for (Index t = 0; t < d.t; ++t)
                    for (Index z = 0; z < d.z; ++z) {
                        ctx.throwIfCancelled();
                        ctx.report(0.3 + 0.7 * static_cast<double>(t * d.z + z) / planes, "");
                        float* r = result->plane(0, t, z);
                        float* g = result->plane(1, t, z);
                        float* b = result->plane(2, t, z);
                        for (Index c = 0; c < d.c; ++c) {
                            const float* src = in->plane(c, t, z);
                            const float w = static_cast<std::size_t>(c) < weights.size() ? static_cast<float>(weights[static_cast<std::size_t>(c)]) : 1.0f;
                            const float s = scale[static_cast<std::size_t>(c)] * w;
                            const std::array<float, 3>& col = colors[static_cast<std::size_t>(c)];
                            for (Index i = 0; i < d.planeSize(); ++i) {
                                const float v0 = std::clamp(src[i] * s, 0.0f, 1.0f);
                                const float cr = col[0] * v0, cg = col[1] * v0, cb = col[2] * v0;
                                if (blend == "Screen") {
                                    r[i] = 1.0f - (1.0f - r[i]) * (1.0f - cr);
                                    g[i] = 1.0f - (1.0f - g[i]) * (1.0f - cg);
                                    b[i] = 1.0f - (1.0f - b[i]) * (1.0f - cb);
                                } else if (blend == "Max") {
                                    r[i] = std::max(r[i], cr);
                                    g[i] = std::max(g[i], cg);
                                    b[i] = std::max(b[i], cb);
                                } else {
                                    r[i] = std::min(1.0f, r[i] + cr);
                                    g[i] = std::min(1.0f, g[i] + cg);
                                    b[i] = std::min(1.0f, b[i] + cb);
                                }
                            }
                        }
                    }
                out.array = result;
                // out.labels stays null: the executor carries the input's
                // labels through (the grid is the same, only c changes), and
                // the corrections painted on this step with them
                out.ranOn = Backend::Cpu;
                out.note = summary(p, meta) + " · CPU";
                out.diagnostics = genericDiagnostics(input, out, summary(p, meta));
                ctx.report(1.0, "");
                return out;
            }

        private:
            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeMergeOperation() { return std::make_unique<MergeOperation>(); }

} // namespace sirius::app
