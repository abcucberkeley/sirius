// Contrast: linear rescale between two percentiles of the intensity
// histogram, then a gamma, per channel. The histograms it shows are also
// available as a live preview before the step runs.
#include "core/ops/builtin.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

#include <sirius/image_ops.hpp>

#include "core/array_source.hpp"

namespace sirius::app {

    namespace {

        struct ChannelWindow {
            float lo = 0.0f, hi = 1.0f;
            float dataMin = 0.0f, dataMax = 1.0f;
            DiagnosticHistogram histogram;
        };

        // Histogram + percentile window of channel `c` from at most
        // `maxPlanes` planes spread over t and z (all of them when 0).
        ChannelWindow windowOf(const StepInput& input, Index c, double loPct, double hiPct, double gamma,
                               Index maxPlanes) {
            const Dims5& d = input.meta.dims;
            const Index planes = d.t * d.z;
            const Index step = maxPlanes > 0 ? std::max<Index>(1, (planes + maxPlanes - 1) / maxPlanes) : 1;
            std::vector<float> samples;
            samples.reserve(static_cast<std::size_t>(((planes + step - 1) / step) * d.planeSize()));
            std::vector<float> plane(static_cast<std::size_t>(d.planeSize()));
            for (Index k = 0; k < planes; k += step) {
                const Index t = k / d.z, z = k % d.z;
                const float* src = nullptr;
                if (input.hasArray()) src = input.array->plane(c, t, z);
                else if (input.source) {
                    input.source->readPlane(c, t, z, plane.data());
                    src = plane.data();
                }
                if (!src) break;
                samples.insert(samples.end(), src, src + d.planeSize());
            }
            ChannelWindow w;
            const Index n = static_cast<Index>(samples.size());
            if (n == 0) return w;
            const auto pct = percentiles(samples.data(), n, loPct, hiPct);
            w.lo = pct.first;
            w.hi = pct.second;
            float mn = std::numeric_limits<float>::infinity(), mx = -mn;
            for (float v : samples) {
                if (std::isnan(v)) continue;
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            if (!(mn <= mx)) {
                mn = 0.0f;
                mx = 1.0f;
            }
            if (mx <= mn) mx = mn + 1.0f;
            w.dataMin = mn;
            w.dataMax = mx;
            w.histogram.bins = histogram(samples.data(), n, 30, mn, mx);
            w.histogram.binLo = mn;
            w.histogram.binHi = mx;
            w.histogram.lo = w.lo;
            w.histogram.hi = w.hi;
            w.histogram.gamma = gamma;
            if (c >= 0 && static_cast<std::size_t>(c) < input.meta.channels.size()) {
                w.histogram.channel = input.meta.channels[static_cast<std::size_t>(c)].label;
                w.histogram.color = input.meta.channels[static_cast<std::size_t>(c)].color;
            } else {
                w.histogram.channel = "ch " + std::to_string(c);
            }
            return w;
        }

        Diagnostics contrastDiagnostics(const StepInput& input, const ParamSet& params, Index maxPlanes,
                                        const std::string& summary) {
            Diagnostics d;
            d.kind = DiagnosticsKind::Contrast;
            d.summary = summary;
            const double lo = params.getDouble("lo_percentile", 0.2), hi = params.getDouble("hi_percentile", 99.8);
            const double gamma = params.getDouble("gamma", 1.0);
            const bool automatic = !(params.getDouble("max", 0.0) > params.getDouble("min", 0.0));
            ContrastWindow eff;
            if (automatic) eff = contrastWindow(input, params, 0, maxPlanes);
            for (Index c = 0; c < input.meta.dims.c; ++c) {
                ChannelWindow w = windowOf(input, c, lo, hi, gamma, maxPlanes);
                w.histogram.lo = automatic ? eff.lo : static_cast<float>(params.getDouble("min", w.dataMin));
                w.histogram.hi = automatic ? eff.hi : static_cast<float>(params.getDouble("max", w.dataMax));
                d.histograms.push_back(w.histogram);
            }
            return d;
        }

        class ContrastOperation final : public Operation {
        public:
            ContrastOperation() {
                info_.kind = "contrast";
                info_.name = "Contrast";
                info_.group = "Intensity";
                info_.kindLabel = "INTENSITY";
                info_.diagnostics = DiagnosticsKind::Contrast;
                info_.defaultCache = CachePolicy::Recompute;
                info_.separableOverT = false;   // the window spans every time point
                info_.helpPage = "contrast";
                info_.livePreview = true;
                info_.params = {
                    doubleParam("min", "Min", 0.0).withHelp("Values at or below map to 0"),
                    doubleParam("max", "Max", 0.0).withHelp("Values at or above map to 1; max <= min means automatic (the percentiles)"),
                    doubleParam("gamma", "Gamma", 1.0).range(0.1, 5.0, 0.05, 2),
                    doubleParam("lo_percentile", "Auto low percentile", 0.2).range(0.0, 50.0, 0.1, 2).withUnit("%").withHelp("Auto sets Min to this percentile of the input").asAdvanced(),
                    doubleParam("hi_percentile", "Auto high percentile", 99.8).range(50.0, 100.0, 0.1, 2).withUnit("%").withHelp("Auto sets Max to this percentile of the input").asAdvanced(),
                    boolParam("bake", "Bake into data", true)
                        .withHelp("The step rewrites intensities into 0..1; kept for future display-only use")
                        .asAdvanced(),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            // A new step starts with the auto window of the data it sees.
            ParamSet initialParams(const ParamSet& defaults, const StepInput& input) const override {
                return contrastAutoParams(defaults, input);
            }

            // Histograms update live while the window is dragged.
            std::optional<Diagnostics> preview(const StepInput& input, const ParamSet& params) const override {
                if (!input.hasArray() && !input.source) return std::nullopt;
                return contrastPreview(input, params);
            }

            std::string summary(const ParamSet& params, const DatasetMeta&) const override {
                char buf[64];
                if (params.getDouble("max", 0.0) <= params.getDouble("min", 0.0))
                    std::snprintf(buf, sizeof buf, "auto %.1f – %.1f %%", params.getDouble("lo_percentile", 0.2),
                                  params.getDouble("hi_percentile", 99.8));
                else
                    std::snprintf(buf, sizeof buf, "window %.4g – %.4g", params.getDouble("min", 0.0), params.getDouble("max", 0.0));
                const double gamma = params.getDouble("gamma", 1.0);
                char g[32];
                std::snprintf(g, sizeof g, "γ %.2g", gamma);
                return joinSummary({buf, params.getBool("per_channel", true) ? "per channel" : "global",
                                    gamma != 1.0 ? g : ""});
            }

            Validation validate(const ParamSet& params, const DatasetMeta& input) const override {
                Validation v = Operation::validate(params, input);
                if (params.getDouble("lo_percentile") >= params.getDouble("hi_percentile"))
                    v.errors.push_back("The auto low percentile must be below the high one.");
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
                StepOutput out;
                out.meta = outputMeta(params, meta);
                ctx.report(0.0, "reading");
                ArrayPtr in = input.materialize([&](double f, const std::string& m) { ctx.report(0.4 * f, m); });
                auto result = std::make_shared<Array5>(in->clone());
                const float gamma = static_cast<float>(params.getDouble("gamma", 1.0));
                const ContrastWindow cw = contrastWindow(StepInput{meta, in, nullptr, nullptr}, params, 0, 0);
                const float mn = cw.lo, mx = cw.hi;
                ctx.report(0.5, "rescaling");
                rescaleGamma(result->data(), result->numel(), mn, mx, gamma);
                char nb[64];
                std::snprintf(nb, sizeof nb, "%g – %g", mn, mx);
                std::string note = nb;
                out.array = result;
                out.labels = input.labels ? input.labels->clone() : nullptr;
                out.ranOn = Backend::Cpu;
                out.note = note + " · CPU";
                out.diagnostics = contrastDiagnostics(StepInput{meta, in, nullptr, nullptr}, params, 0,
                                                      summary(params, meta));
                ctx.report(1.0, "");
                return out;
            }

        private:
            OpInfo info_;
        };

    } // namespace

    ContrastWindow contrastWindow(const StepInput& input, const ParamSet& paramsIn, Index c, Index maxPlanes,
                                  bool wantRange) {
        ParamSet params = paramsIn;
        if (const Operation* op = findOperation("contrast")) params.applyDefaults(op->info().params);
        ContrastWindow out;
        out.gamma = static_cast<float>(params.getDouble("gamma", 1.0));
        out.lo = static_cast<float>(params.getDouble("min", 0.0));
        out.hi = static_cast<float>(params.getDouble("max", 0.0));
        const bool automatic = !(out.hi > out.lo);
        if (!wantRange && !automatic) return out;
        if (automatic) {
            // one window for every channel: the extreme percentiles across them
            const ParamSet a = contrastAutoParams(params, input);
            out.lo = static_cast<float>(a.getDouble("min", 0.0));
            out.hi = static_cast<float>(a.getDouble("max", 1.0));
            if (!wantRange) return out;
        }
        const ChannelWindow w = windowOf(input, c, params.getDouble("lo_percentile", 0.2),
                                         params.getDouble("hi_percentile", 99.8), out.gamma, maxPlanes);
        out.dataMin = w.dataMin;
        out.dataMax = w.dataMax;
        return out;
    }

    ParamSet contrastAutoParams(const ParamSet& current, const StepInput& input) {
        ParamSet p = current;
        if (const Operation* op = findOperation("contrast")) p.applyDefaults(op->info().params);
        // percentiles over every channel's samples, one window for all
        float lo = std::numeric_limits<float>::infinity(), hi = -lo;
        for (Index c = 0; c < input.meta.dims.c; ++c) {
            const ChannelWindow w = windowOf(input, c, p.getDouble("lo_percentile", 0.2), p.getDouble("hi_percentile", 99.8),
                                             p.getDouble("gamma", 1.0), 8);
            lo = std::min(lo, w.lo);
            hi = std::max(hi, w.hi);
        }
        if (!(lo < hi)) {
            lo = 0.0f;
            hi = 1.0f;
        }
        p.set("min", static_cast<double>(lo));
        p.set("max", static_cast<double>(hi));
        return p;
    }

    ParamSet contrastResetParams(const ParamSet& current, const StepInput& input) {
        ParamSet p = current;
        float mn = std::numeric_limits<float>::infinity(), mx = -mn;
        for (Index c = 0; c < input.meta.dims.c; ++c) {
            const ContrastWindow w = contrastWindow(input, current, c, 8, true);
            mn = std::min(mn, w.dataMin);
            mx = std::max(mx, w.dataMax);
        }
        if (!(mn < mx)) {
            mn = 0.0f;
            mx = 1.0f;
        }
        p.set("min", static_cast<double>(mn));
        p.set("max", static_cast<double>(mx));
        p.set("gamma", 1.0);
        return p;
    }

    Diagnostics contrastPreview(const StepInput& input, const ParamSet& params) {
        // 8 planes per channel keep this under ~100 ms on 2048² planes
        ParamSet p = params;
        if (const Operation* op = findOperation("contrast")) p.applyDefaults(op->info().params);
        return contrastDiagnostics(input, p, 8, findOperation("contrast") ? findOperation("contrast")->summary(p, input.meta) : "");
    }

    std::unique_ptr<Operation> makeContrastOperation() { return std::make_unique<ContrastOperation>(); }

} // namespace sirius::app
