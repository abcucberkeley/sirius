// Helpers shared by the operation implementations.
#include "core/ops/common.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <exception>
#include <mutex>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/array_source.hpp"

#include <sirius/image_ops.hpp>   // histogram

namespace sirius::app {

    void forEachVolume(const DatasetMeta& meta, const StepContext& ctx,
                       const std::function<void(Index c, Index t)>& fn) {
        const Index total = std::max<Index>(1, meta.dims.c * meta.dims.t);
        Index done = 0;
        for (Index t = 0; t < meta.dims.t; ++t)
            for (Index c = 0; c < meta.dims.c; ++c) {
                ctx.throwIfCancelled();
                char msg[64];
                if (total > 1)
                    std::snprintf(msg, sizeof msg, "c %lld · t %lld", static_cast<long long>(c),
                                  static_cast<long long>(t));
                else
                    msg[0] = '\0';
                ctx.report(static_cast<double>(done) / static_cast<double>(total), msg);
                fn(c, t);
                ++done;
            }
        ctx.report(1.0, "");
    }

    void forEachVolumeOnGpus(const DatasetMeta& meta, const StepContext& ctx,
                             const std::function<void(Index c, Index t, Device device)>& fn) {
        const Index C = std::max<Index>(1, meta.dims.c);
        const Index T = std::max<Index>(1, meta.dims.t);
        const Index total = C * T;
        const int nDev = cudaDeviceCount();
        const bool parallel = ctx.allCudaDevices() && nDev > 1 && total > 1;
        if (!parallel) {
            forEachVolume(meta, ctx, [&](Index c, Index t) { fn(c, t, ctx.deviceForVolume(c, t, C)); });
            return;
        }

        std::atomic<Index> done{0};
        std::mutex progressMu;
        std::exception_ptr ep;
        std::mutex epMu;
#ifdef _OPENMP
#pragma omp parallel for num_threads(nDev) schedule(dynamic)
#endif
        for (Index i = 0; i < total; ++i) {
            if (ctx.isCancelled()) continue;
            {
                // one volume failed: the rest of a long movie is not worth
                // running only to throw its exception away at the end
                std::lock_guard<std::mutex> g(epMu);
                if (ep) continue;
            }
            const Index t = i / C, c = i % C;
            // Everything that can throw stays inside the try (the progress
            // report included): an exception leaving an OpenMP loop body ends
            // the process.
            try {
                fn(c, t, Device::cuda(static_cast<int>(i % nDev)));
                const Index n = done.fetch_add(1) + 1;
                std::lock_guard<std::mutex> g(progressMu);
                char msg[64];
                std::snprintf(msg, sizeof msg, "c %lld · t %lld", static_cast<long long>(c), static_cast<long long>(t));
                ctx.report(static_cast<double>(n) / static_cast<double>(total), msg);
            } catch (...) {
                std::lock_guard<std::mutex> g(epMu);
                if (!ep) ep = std::current_exception();
            }
        }
        ctx.throwIfCancelled();
        if (ep) std::rethrow_exception(ep);
        ctx.report(1.0, "");
    }

    std::string joinSummary(std::initializer_list<std::string> parts) {
        std::string out;
        for (const std::string& p : parts) {
            if (p.empty()) continue;
            if (!out.empty()) out += " · ";
            out += p;
        }
        return out;
    }

    std::string channelName(const DatasetMeta& meta, Index c) {
        if (c < 0 || static_cast<std::size_t>(c) >= meta.channels.size()) return "ch " + std::to_string(c);
        const ChannelInfo& ch = meta.channels[static_cast<std::size_t>(c)];
        std::string out;
        if (ch.wavelengthNm > 0.0) out = std::to_string(static_cast<int>(std::lround(ch.wavelengthNm)));
        if (!ch.label.empty()) out += (out.empty() ? "" : " ") + ch.label;
        return out.empty() ? "ch " + std::to_string(c) : out;
    }

    std::string formatBytes(std::uint64_t bytes) {
        char buf[32];
        const double gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
        if (gb >= 1.0) std::snprintf(buf, sizeof buf, "%.1f GB", gb);
        else if (bytes >= 1024ull * 1024ull)
            std::snprintf(buf, sizeof buf, "%.0f MB", static_cast<double>(bytes) / (1024.0 * 1024.0));
        else std::snprintf(buf, sizeof buf, "%.0f kB", static_cast<double>(bytes) / 1024.0);
        return buf;
    }

    std::string formatNumber(double v, int decimals) {
        char buf[48];
        std::snprintf(buf, sizeof buf, "%.*f", decimals, v);
        return buf;
    }

    std::string estimatedTime(std::uint64_t bytes, double bytesPerSecond) {
        const double s = static_cast<double>(bytes) / std::max(bytesPerSecond, 1.0);
        char buf[32];
        if (s < 1.0) std::snprintf(buf, sizeof buf, "< 1 s");
        else if (s < 90.0) std::snprintf(buf, sizeof buf, "~%.0f s", s);
        else std::snprintf(buf, sizeof buf, "~%.1f min", s / 60.0);
        return buf;
    }

    Diagnostics genericDiagnostics(const StepInput& input, const StepOutput& output, const std::string& summary,
                                   double bytesPerSecond) {
        Diagnostics d;
        d.kind = DiagnosticsKind::Generic;
        d.summary = summary;
        const Dims5& id = input.meta.dims;
        if (input.hasArray() || input.source) {
            try {
                const Index c = 0, t = 0, z = id.z / 2;
                std::vector<float> plane(static_cast<std::size_t>(id.planeSize()));
                if (input.hasArray()) std::copy_n(input.array->plane(c, t, z), id.planeSize(), plane.data());
                else input.source->readPlane(c, t, z, plane.data());
                d.tabs.push_back({"Preview", {}});
                d.tabs.back().images.push_back(
                    d.addImage(thumbnail(plane.data(), id.y, id.x, 400, "Input", input.meta.shapeString())));
            } catch (const std::exception&) {
                // a preview is never worth failing a step for
            }
        }
        if (output.array && !output.array->empty()) {
            const Dims5& od = output.meta.dims;
            const int img = d.addImage(thumbnail(output.array->plane(0, 0, od.z / 2), od.y, od.x, 400,
                                                 "Output · live", output.meta.shapeString()));
            if (d.tabs.empty()) d.tabs.push_back({"Preview", {}});
            d.tabs.back().images.push_back(img);
        }
        d.facts.push_back({"Est. time", estimatedTime(output.meta.dims.bytes(), bytesPerSecond)});
        d.facts.push_back({"Peak memory", formatBytes(input.meta.dims.bytes() + output.meta.dims.bytes())});
        return d;
    }

    std::shared_ptr<Array5> allocateLike(const DatasetMeta& meta) { return std::make_shared<Array5>(meta.dims); }

    std::size_t FramePrompt::objects() const noexcept {
        return static_cast<std::size_t>(std::count_if(ids.begin(), ids.end(), [](std::uint32_t id) { return id != 0; }));
    }

    std::vector<std::array<double, 3>> scribbleSample(const std::vector<std::array<double, 3>>& stroke, std::size_t atMost) {
        if (stroke.size() <= atMost || atMost < 2) return stroke;
        std::vector<std::array<double, 3>> out;
        const double step = static_cast<double>(stroke.size() - 1) / static_cast<double>(atMost - 1);
        for (std::size_t k = 0; k < atMost; ++k)
            out.push_back(stroke[std::min(stroke.size() - 1, static_cast<std::size_t>(std::lround(static_cast<double>(k) * step)))]);
        return out;
    }

    FramePrompt framePrompt(const std::vector<Prompt>& prompts, Index t) {
        FramePrompt f;
        std::uint32_t next = 1;
        const auto mask = [&](std::size_t i, bool object) {
            f.ids.push_back(object ? next++ : 0u);
            f.placed.push_back(i);
        };
        for (const bool object : {false, true})
            for (std::size_t i = 0; i < prompts.size(); ++i) {
                const Prompt& p = prompts[i];
                if (p.t != t || p.kind != Prompt::Kind::Point || p.object != object) continue;
                f.points.push_back(p.at);
                f.pointLabels.push_back(object ? 1 : 0);
                mask(i, object);
            }
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            const Prompt& p = prompts[i];
            if (p.t != t || p.kind != Prompt::Kind::Box) continue;
            f.boxes.push_back(p.box);
            mask(i, true);
        }
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            const Prompt& p = prompts[i];
            if (p.t != t || p.kind != Prompt::Kind::Scribble) continue;
            f.scribbles.push_back({scribbleSample(p.stroke), p.object ? 1 : 0});
            mask(i, p.object);
        }
        return f;
    }

    void applyPromptIds(std::uint32_t* labels, Index n, const FramePrompt& frame) {
        const std::size_t sent = frame.ids.size();
        for (Index i = 0; i < n; ++i) {
            const std::uint32_t id = labels[i];
            if (id == 0) continue;
            labels[i] = id <= sent ? frame.ids[id - 1] : 0u;
        }
    }

    void validatePrompts(const std::vector<Prompt>& prompts, const DatasetMeta& in, Validation& v) {
        const Dims5& d = in.dims;
        if (prompts.empty()) {
            v.warnings.push_back("No prompts yet: choose the viewer's Prompt tool and drag a box around an object, or click it.");
            return;
        }
        const auto inside = [&](const std::array<double, 3>& q) {
            return q[0] < static_cast<double>(d.x) && q[1] < static_cast<double>(d.y) && q[2] < static_cast<double>(d.z);
        };
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            const Prompt& p = prompts[i];
            bool ok = p.t < d.t;
            const char* what = "point";
            switch (p.kind) {
                case Prompt::Kind::Point: ok = ok && inside(p.at); break;
                case Prompt::Kind::Box:
                    what = "box";
                    ok = ok && p.box[3] <= static_cast<double>(d.x) && p.box[4] <= static_cast<double>(d.y) && p.box[5] <= static_cast<double>(d.z);
                    break;
                case Prompt::Kind::Scribble:
                    what = "scribble";
                    ok = ok && std::all_of(p.stroke.begin(), p.stroke.end(), inside);
                    break;
            }
            if (ok) continue;
            char text[240];
            std::snprintf(text, sizeof text, "Prompt %zu (a %s on time point %lld) lies outside the image (%lld x %lld x %lld voxels, %lld time points).",
                          i + 1, what, static_cast<long long>(p.t), static_cast<long long>(d.x), static_cast<long long>(d.y),
                          static_cast<long long>(d.z), static_cast<long long>(d.t));
            v.errors.push_back(text);
            return;
        }
        if (std::none_of(prompts.begin(), prompts.end(), [](const Prompt& p) { return p.object; }))
            v.warnings.push_back("Only background prompts: background names no object, so there is nothing to segment.");
    }

    std::string promptCounts(const std::vector<Prompt>& prompts) {
        std::size_t boxes = 0, objects = 0, background = 0, scribbles = 0;
        for (const Prompt& p : prompts) {
            if (p.kind == Prompt::Kind::Box) ++boxes;
            else if (p.kind == Prompt::Kind::Scribble) ++scribbles;
            else if (p.object) ++objects;
            else ++background;
        }
        std::string out;
        const auto part = [&out](std::size_t n, const char* one, const char* many) {
            if (n == 0) return;
            out += (out.empty() ? "" : " \xC2\xB7 ") + std::to_string(n) + " " + (n == 1 ? one : many);
        };
        part(boxes, "box", "boxes");
        part(objects, "object point", "object points");
        part(background, "background point", "background points");
        part(scribbles, "scribble", "scribbles");
        return out.empty() ? std::string("none") : out;
    }

    Diagnostics labelDiagnostics(const LabelVolume& labels, const std::string& summary) {
        Diagnostics d;
        d.kind = DiagnosticsKind::Segment;
        d.summary = summary;
        DiagnosticTable table;
        table.caption = "Labels";
        table.header = {"ID", "Class", "Voxels", "Conf.", "Flag"};
        int row = 0;
        Index lowConf = 0, border = 0, size = 0;
        for (const LabelStats& s : labels.stats()) {
            char id[16];
            std::snprintf(id, sizeof id, "%04u", s.id);
            table.rows.push_back({id, s.cls, std::to_string(s.voxels), formatNumber(s.confidence, 2), s.flagText()});
            if (s.confidence < 0.6) table.accentCells.emplace_back(row, 3);
            for (const std::string& f : s.flags) {
                if (f == "low conf") ++lowConf;
                else if (f == "touching border") ++border;
                else ++size;
            }
            ++row;
        }
        d.table = std::move(table);
        d.facts.push_back({"Labels", std::to_string(labels.stats().size())});
        d.facts.push_back({"Low confidence (< 0.6)", std::to_string(lowConf)});
        d.facts.push_back({"Touching border", std::to_string(border)});
        d.facts.push_back({"Size outliers", std::to_string(size)});
        d.facts.push_back({"Reviewed", std::to_string(labels.reviewedCount()) + " / " +
                                           std::to_string(labels.stats().size())});
        return d;
    }

    float otsuThreshold(const float* v, Index n) {
        // over the finite values: an infinite end has no bins to split
        float mn = std::numeric_limits<float>::infinity(), mx = -mn;
        for (Index i = 0; i < n; ++i) {
            if (!std::isfinite(v[i])) continue;
            mn = std::min(mn, v[i]);
            mx = std::max(mx, v[i]);
        }
        if (!(mx > mn)) return mn;
        constexpr int bins = 256;
        const std::vector<double> h = histogram(v, n, bins, mn, mx);
        double total = 0.0, sumAll = 0.0;
        for (int i = 0; i < bins; ++i) {
            total += h[static_cast<std::size_t>(i)];
            sumAll += i * h[static_cast<std::size_t>(i)];
        }
        double wB = 0.0, sumB = 0.0, best = -1.0;
        int bestBin = 0;
        for (int i = 0; i < bins; ++i) {
            wB += h[static_cast<std::size_t>(i)];
            if (wB == 0.0) continue;
            const double wF = total - wB;
            if (wF == 0.0) break;
            sumB += i * h[static_cast<std::size_t>(i)];
            const double mB = sumB / wB, mF = (sumAll - sumB) / wF;
            const double between = wB * wF * (mB - mF) * (mB - mF);
            if (between > best) {
                best = between;
                bestBin = i;
            }
        }
        return mn + (mx - mn) * static_cast<float>(bestBin + 1) / bins;
    }

} // namespace sirius::app
