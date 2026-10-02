// Helpers shared by the operation implementations.
#include "core/ops/common.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <mutex>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/array_source.hpp"

#include <nlohmann/json.hpp>

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

    std::vector<std::uint32_t> FramePrompt::ids() const {
        std::vector<std::uint32_t> out;
        for (const Object& o : objects) out.push_back(o.id);
        return out;
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
        std::vector<FramePrompt::Object> all;   // every object on t, background-only ones too
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            const Prompt& p = prompts[i];
            if (p.t != t) continue;
            auto it = std::find_if(all.begin(), all.end(), [&](const FramePrompt::Object& o) { return o.id == p.objectId; });
            if (it == all.end()) {
                all.emplace_back();
                all.back().id = p.objectId;
                it = all.end() - 1;
            }
            FramePrompt::Object& o = *it;
            o.placed.push_back(i);
            switch (p.kind) {
                case Prompt::Kind::Point:
                    o.points.push_back(p.at);
                    o.pointLabels.push_back(p.positive ? 1 : 0);
                    break;
                case Prompt::Kind::Box:
                    if (!o.box) o.box = p.box;
                    ++o.boxes;
                    break;
                case Prompt::Kind::Scribble: o.scribbles.push_back({scribbleSample(p.stroke), p.positive ? 1 : 0}); break;
            }
        }
        std::sort(all.begin(), all.end(), [](const FramePrompt::Object& a, const FramePrompt::Object& b) { return a.id < b.id; });
        for (FramePrompt::Object& o : all) {
            const bool named = o.box || std::find(o.pointLabels.begin(), o.pointLabels.end(), 1) != o.pointLabels.end() ||
                               std::any_of(o.scribbles.begin(), o.scribbles.end(), [](const FramePrompt::Stroke& s) { return s.label == 1; });
            if (named) f.objects.push_back(std::move(o));
            else f.backgroundOnly.push_back(o.id);
        }
        return f;
    }

    nlohmann::json promptObjectsJson(const FramePrompt& frame) {
        nlohmann::json list = nlohmann::json::array();
        for (const FramePrompt::Object& o : frame.objects) {
            nlohmann::json e = nlohmann::json::object();
            if (o.box) e["box"] = *o.box;
            if (!o.points.empty()) {
                e["points"] = o.points;
                e["point_labels"] = o.pointLabels;
            }
            if (!o.scribbles.empty()) {
                nlohmann::json strokes = nlohmann::json::array();
                for (const FramePrompt::Stroke& s : o.scribbles) strokes.push_back({{"points", s.points}, {"label", s.label}});
                e["scribbles"] = std::move(strokes);
            }
            list.push_back(std::move(e));
        }
        return list;
    }

    void applyPromptIds(std::uint32_t* labels, Index n, const FramePrompt& frame) {
        const std::size_t sent = frame.objects.size();
        for (Index i = 0; i < n; ++i) {
            const std::uint32_t id = labels[i];
            if (id == 0) continue;
            labels[i] = id <= sent ? frame.objects[id - 1].id : 0u;
        }
    }

    std::vector<Index> promptPlanes(const FramePrompt::Object& o) {
        // Python's round(), as the worker's: half to even (the default
        // floating-point rounding mode of nearbyint)
        std::vector<Index> planes;
        const auto add = [&planes](double z) {
            const Index p = static_cast<Index>(std::nearbyint(z));
            if (std::find(planes.begin(), planes.end(), p) == planes.end()) planes.push_back(p);
        };
        for (const std::array<double, 3>& q : o.points) add(q[2]);
        for (const FramePrompt::Stroke& s : o.scribbles)
            for (const std::array<double, 3>& q : s.points) add(q[2]);
        if (planes.empty() && o.box) add(((*o.box)[2] + (*o.box)[5] - 1.0) / 2.0);
        std::sort(planes.begin(), planes.end());
        return planes;
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
            char text[260];
            std::snprintf(text, sizeof text,
                          "Prompt %zu (a %s of object %u on time point %lld) lies outside the image (%lld x %lld x %lld voxels, %lld time points).",
                          i + 1, what, p.objectId, static_cast<long long>(p.t), static_cast<long long>(d.x), static_cast<long long>(d.y),
                          static_cast<long long>(d.z), static_cast<long long>(d.t));
            v.errors.push_back(text);
            return;
        }
        std::vector<Index> times;
        for (const Prompt& p : prompts)
            if (std::find(times.begin(), times.end(), p.t) == times.end()) times.push_back(p.t);
        std::sort(times.begin(), times.end());
        for (const Index t : times) {
            const FramePrompt f = framePrompt(prompts, t);
            for (const FramePrompt::Object& o : f.objects) {
                if (o.boxes <= 1) continue;
                v.errors.push_back("Object " + std::to_string(o.id) + " has " + std::to_string(o.boxes) + " boxes on time point " + std::to_string(t) +
                                   ": one object is one mask and takes one box. Draw the second box as a new object, or remove one.");
                return;
            }
            for (const std::uint32_t id : f.backgroundOnly)
                v.warnings.push_back("Object " + std::to_string(id) + " has only background prompts on time point " + std::to_string(t) +
                                     ", so it is not sent: give it an object point, a box or a scribble, or remove it.");
        }
        if (std::none_of(prompts.begin(), prompts.end(), [](const Prompt& p) { return p.positive; }))
            v.warnings.push_back("Only background prompts: background names no object, so there is nothing to segment.");
    }

    std::string promptCounts(const std::vector<Prompt>& prompts) {
        std::size_t boxes = 0, objects = 0, background = 0, scribbles = 0;
        for (const Prompt& p : prompts) {
            if (p.kind == Prompt::Kind::Box) ++boxes;
            else if (p.kind == Prompt::Kind::Scribble) ++scribbles;
            else if (p.positive) ++objects;
            else ++background;
        }
        std::string out;
        const auto part = [&out](std::size_t n, const char* one, const char* many) {
            if (n == 0) return;
            out += (out.empty() ? "" : " \xC2\xB7 ") + std::to_string(n) + " " + (n == 1 ? one : many);
        };
        part(promptObjects(prompts).size(), "object", "objects");
        part(boxes, "box", "boxes");
        part(objects, "object point", "object points");
        part(background, "background point", "background points");
        part(scribbles, "scribble", "scribbles");
        return out.empty() ? std::string("none") : out;
    }

    std::vector<PromptObject> promptObjects(const std::vector<Prompt>& prompts) {
        std::vector<PromptObject> out;
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            const Prompt& p = prompts[i];
            auto it = std::find_if(out.begin(), out.end(), [&](const PromptObject& o) { return o.id == p.objectId; });
            if (it == out.end()) {
                out.emplace_back();
                out.back().id = p.objectId;
                it = out.end() - 1;
            }
            it->prompts.push_back(i);
            if (std::find(it->times.begin(), it->times.end(), p.t) == it->times.end()) it->times.push_back(p.t);
            if (!p.positive) ++it->corrections;
            else if (p.kind == Prompt::Kind::Box) ++it->boxes;
            else if (p.kind == Prompt::Kind::Scribble) ++it->scribbles;
            else ++it->points;
        }
        for (PromptObject& o : out) std::sort(o.times.begin(), o.times.end());
        std::sort(out.begin(), out.end(), [](const PromptObject& a, const PromptObject& b) { return a.id < b.id; });
        return out;
    }

    std::string promptObjectText(const PromptObject& o) {
        std::string out;
        // "box", "scribble" when there is one (an object rarely has two),
        // counted otherwise: "box + 2 points + 1 correction"
        const auto part = [&out](std::size_t n, const char* one, const char* many, bool countOne) {
            if (n == 0) return;
            out += (out.empty() ? "" : " + ") + (n == 1 && !countOne ? std::string() : std::to_string(n) + " ") + (n == 1 ? one : many);
        };
        part(o.boxes, "box", "boxes", false);
        part(o.points, "point", "points", true);
        part(o.scribbles, "scribble", "scribbles", false);
        part(o.corrections, "correction", "corrections", true);
        return out.empty() ? std::string("no prompts") : out;
    }

    std::uint32_t nextPromptObject(const std::vector<Prompt>& prompts) {
        std::uint32_t top = 0;
        for (const Prompt& p : prompts) top = std::max(top, p.objectId);
        return top + 1;
    }

    PromptClick promptClickTarget(const std::vector<Prompt>& prompts, Index t, const std::array<double, 3>& at, std::uint32_t maskUnder,
                                  bool positive, bool forceNew, bool planar) {
        const FramePrompt f = framePrompt(prompts, t);
        const Index plane = static_cast<Index>(std::floor(at[2]));
        const auto counts = [&](const FramePrompt::Object& o) {
            if (!planar) return true;
            const std::vector<Index> planes = promptPlanes(o);
            return planes.size() == 1 && planes[0] == plane;
        };
        const auto under = [&]() -> const FramePrompt::Object* {
            if (maskUnder == 0) return nullptr;
            for (const FramePrompt::Object& o : f.objects)
                if (o.id == maskUnder && counts(o)) return &o;
            return nullptr;
        };
        PromptClick c;
        if (positive) {
            if (const FramePrompt::Object* o = forceNew ? nullptr : under()) {
                c.action = PromptClick::Action::AddTo;
                c.object = o->id;
            } else {
                c.action = PromptClick::Action::NewObject;
                c.object = nextPromptObject(prompts);
            }
            return c;
        }
        if (const FramePrompt::Object* o = under()) {
            c.action = PromptClick::Action::AddTo;
            c.object = o->id;
            return c;
        }
        double bestD = std::numeric_limits<double>::infinity();
        for (const FramePrompt::Object& o : f.objects) {
            if (!counts(o)) continue;
            double d = std::numeric_limits<double>::infinity();
            for (const std::size_t k : o.placed) d = std::min(d, promptDistance(prompts[k], at));
            if (d < bestD) {   // ascending ids: the lower one wins a tie
                bestD = d;
                c.object = o.id;
            }
        }
        if (c.object != 0) {
            c.action = PromptClick::Action::AddTo;
            return c;
        }
        c.why = planar && !f.objects.empty()
                    ? "a background click corrects an object on its own plane (a 2-D model): place an object on this plane first"
                    : "a background click corrects an object: place one first (a box, a click or a scribble)";
        return c;
    }

    std::vector<Prompt> removePrompt(const std::vector<Prompt>& prompts, std::size_t k) {
        if (k >= prompts.size()) return prompts;
        const Prompt gone = prompts[k];
        std::vector<Prompt> out = prompts;
        out.erase(out.begin() + static_cast<std::ptrdiff_t>(k));
        const auto same = [&](const Prompt& p) { return p.objectId == gone.objectId && p.t == gone.t; };
        if (gone.positive && std::none_of(out.begin(), out.end(), [&](const Prompt& p) { return same(p) && p.positive; }))
            out.erase(std::remove_if(out.begin(), out.end(), same), out.end());
        return out;
    }

    std::vector<Prompt> removePromptObject(const std::vector<Prompt>& prompts, std::uint32_t id) {
        std::vector<Prompt> out;
        for (const Prompt& p : prompts)
            if (p.objectId != id) out.push_back(p);
        return out;
    }

    void appendPromptScores(std::string& fact, const FramePrompt& frame, const nlohmann::json& maskScores, Index t, bool manyTimes) {
        std::string line;
        for (std::size_t i = 0; i < frame.objects.size(); ++i) {
            if (!maskScores.is_array() || i >= maskScores.size() || !maskScores[i].is_number()) continue;
            line += (line.empty() ? "" : ", ") + std::string("#") + std::to_string(frame.objects[i].id) + " " + formatNumber(maskScores[i].get<double>(), 2);
        }
        if (line.empty()) return;
        fact += (fact.empty() ? "" : " \xC2\xB7 ") + (manyTimes ? "t " + std::to_string(t) + ": " : std::string()) + line;
    }

    std::vector<std::pair<std::uint32_t, double>> promptScores(const Diagnostics& diagnostics, Index t) {
        std::vector<std::pair<std::uint32_t, double>> out;
        const DiagnosticFact* fact = nullptr;
        for (const DiagnosticFact& f : diagnostics.facts)
            if (f.key == "Mask scores") fact = &f;
        if (!fact) return out;
        const std::string sep = " \xC2\xB7 ";
        std::size_t at = 0;
        while (at <= fact->value.size()) {
            std::size_t end = fact->value.find(sep, at);
            if (end == std::string::npos) end = fact->value.size();
            std::string part = fact->value.substr(at, end - at);
            at = end + sep.size();
            Index partT = 0;
            if (part.rfind("t ", 0) == 0) {
                const std::size_t colon = part.find(": ");
                if (colon == std::string::npos) continue;
                partT = std::strtoll(part.c_str() + 2, nullptr, 10);
                part = part.substr(colon + 2);
            }
            if (partT != t) continue;
            std::size_t q = 0;
            while ((q = part.find('#', q)) != std::string::npos) {
                char* next = nullptr;
                const unsigned long id = std::strtoul(part.c_str() + q + 1, &next, 10);
                const double score = std::strtod(next, nullptr);
                out.emplace_back(static_cast<std::uint32_t>(id), score);
                ++q;
            }
        }
        return out;
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
