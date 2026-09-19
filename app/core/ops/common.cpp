// Helpers shared by the operation implementations.
#include "core/ops/builtin.hpp"

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

} // namespace sirius::app
