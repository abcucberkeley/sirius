#include "core/tracks.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace sirius::app {

    // --- TrackIndex -----------------------------------------------------------------

    TrackIndex::TrackIndex(const LabelVolume& labels)
        : frames_(static_cast<std::size_t>(std::max<Index>(0, labels.t()))), y_(labels.y()), x_(labels.x()) {
        if (labels.empty()) return;
        const Index n = labels.t();
#pragma omp parallel for schedule(dynamic)
        for (Index t = 0; t < n; ++t) countFrame(labels, t, frames_[static_cast<std::size_t>(t)]);
    }

    bool TrackIndex::empty() const noexcept {
        return std::all_of(frames_.begin(), frames_.end(), [](const Frame& f) { return f.empty(); });
    }

    void TrackIndex::countFrame(const LabelVolume& labels, Index t, Frame& out) {
        out.clear();
        const Index z = labels.z(), y = labels.y(), x = labels.x();
        const std::uint32_t* v = labels.volume(t);
        for (Index iz = 0; iz < z; ++iz)
            for (Index iy = 0; iy < y; ++iy) {
                const std::uint32_t* row = v + (iz * y + iy) * x;
                // objects come in runs along a row: look the id up once per run
                std::uint32_t current = 0;
                Sum* s = nullptr;
                for (Index ix = 0; ix < x; ++ix) {
                    const std::uint32_t id = row[ix];
                    if (!id) continue;
                    if (id != current) {
                        current = id;
                        s = &out[id];
                    }
                    s->z += static_cast<double>(iz);
                    s->y += static_cast<double>(iy);
                    s->x += static_cast<double>(ix);
                    ++s->n;
                }
            }
    }

    void TrackIndex::rescanFrame(const LabelVolume& labels, Index t) {
        if (t < 0 || t >= labels.t()) throw std::out_of_range("TrackIndex::rescanFrame: t out of range");
        if (labels.t() != frames() || labels.y() != y_ || labels.x() != x_)
            throw std::invalid_argument("TrackIndex::rescanFrame: the labels are not the ones indexed");
        countFrame(labels, t, frames_[static_cast<std::size_t>(t)]);
    }

    void TrackIndex::move(Frame& frame, Index linear, std::uint32_t from, std::uint32_t to) {
        if (from == to) return;
        const double iz = static_cast<double>(linear / (y_ * x_));
        const double iy = static_cast<double>((linear / x_) % y_);
        const double ix = static_cast<double>(linear % x_);
        if (from) {
            const auto it = frame.find(from);
            if (it != frame.end()) {
                Sum& s = it->second;
                s.z -= iz;
                s.y -= iy;
                s.x -= ix;
                if (--s.n <= 0) frame.erase(it);
            }
        }
        if (to) {
            Sum& s = frame[to];
            s.z += iz;
            s.y += iy;
            s.x += ix;
            ++s.n;
        }
    }

    void TrackIndex::apply(const LabelDiff& diff, bool forward) {
        if (diff.empty()) return;
        if (diff.t < 0 || diff.t >= frames()) throw std::out_of_range("TrackIndex::apply: t out of range");
        if (diff.before.size() != diff.indices.size() || diff.after.size() != diff.indices.size())
            throw std::invalid_argument("TrackIndex::apply: malformed diff");
        Frame& frame = frames_[static_cast<std::size_t>(diff.t)];
        const std::size_t count = diff.indices.size();
        for (std::size_t step = 0; step < count; ++step) {
            const std::size_t k = forward ? step : count - 1 - step;
            move(frame, diff.indices[k], forward ? diff.before[k] : diff.after[k], forward ? diff.after[k] : diff.before[k]);
        }
    }

    TrackPoint TrackIndex::pointOf(Index t, const Sum& s) {
        TrackPoint p;
        p.t = t;
        p.voxels = s.n;
        const double n = static_cast<double>(std::max<Index>(1, s.n));
        p.centroid = {s.z / n, s.y / n, s.x / n};
        return p;
    }

    std::vector<std::uint32_t> TrackIndex::ids() const {
        std::vector<std::uint32_t> out;
        for (const Frame& f : frames_)
            for (const auto& [id, s] : f) out.push_back(id);
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    std::vector<TrackPoint> TrackIndex::points(std::uint32_t id) const {
        std::vector<TrackPoint> out;
        for (Index t = 0; t < frames(); ++t) {
            const Frame& f = frames_[static_cast<std::size_t>(t)];
            const auto it = f.find(id);
            if (it != f.end()) out.push_back(pointOf(t, it->second));
        }
        return out;
    }

    std::optional<TrackPoint> TrackIndex::pointAt(std::uint32_t id, Index t) const {
        if (t < 0 || t >= frames()) return std::nullopt;
        const Frame& f = frames_[static_cast<std::size_t>(t)];
        const auto it = f.find(id);
        if (it == f.end()) return std::nullopt;
        return pointOf(t, it->second);
    }

    std::optional<TrackPoint> TrackIndex::nearestPoint(std::uint32_t id, Index t) const {
        const Index n = frames();
        if (n == 0) return std::nullopt;
        t = std::clamp<Index>(t, 0, n - 1);
        for (Index d = 0; d < n; ++d) {
            if (auto p = pointAt(id, t - d)) return p;
            if (auto p = pointAt(id, t + d)) return p;
        }
        return std::nullopt;
    }

    void TrackIndex::forEachPoint(const std::function<void(std::uint32_t, const TrackPoint&)>& fn) const {
        for (Index t = 0; t < frames(); ++t)
            for (const auto& [id, sum] : frames_[static_cast<std::size_t>(t)]) fn(id, pointOf(t, sum));
    }

    // --- summaries ------------------------------------------------------------------

    std::vector<TrackSummary> summarizeTracks(const TrackIndex& index, const Lineage& lineage,
                                              const std::array<double, 3>& voxelUm) {
        std::map<std::uint32_t, TrackSummary> rows;
        std::map<std::uint32_t, TrackPoint> previous;   // last point seen, per id
        std::map<std::uint32_t, TrackPoint> firstPoint;
        const auto stepUm = [&](const TrackPoint& a, const TrackPoint& b) {
            double sum = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                const double d = (b.centroid[k] - a.centroid[k]) * voxelUm[k];
                sum += d * d;
            }
            return std::sqrt(sum);
        };

        index.forEachPoint([&](std::uint32_t id, const TrackPoint& p) {
            TrackSummary& row = rows[id];
            if (row.frames == 0) {
                row.id = id;
                row.first = p.t;
                firstPoint[id] = p;
            } else {
                const TrackPoint& before = previous[id];
                row.pathUm += stepUm(before, p);
                row.gaps += p.t - before.t - 1;
            }
            row.last = p.t;
            ++row.frames;
            row.meanVoxels += static_cast<double>(p.voxels);
            previous[id] = p;
        });

        for (auto& [id, row] : rows) {
            row.meanVoxels /= static_cast<double>(std::max<Index>(1, row.frames));
            row.netUm = stepUm(firstPoint[id], previous[id]);
            row.umPerFrame = row.last > row.first ? row.pathUm / static_cast<double>(row.last - row.first) : 0.0;
        }
        for (const auto& [child, parent] : lineage) {
            if (child == parent) continue;
            const auto c = rows.find(child);
            const auto p = rows.find(parent);
            if (c == rows.end() || p == rows.end()) continue;
            c->second.parent = parent;
            p->second.children.push_back(child);   // lineage is ordered by child, so ascending
        }
        for (auto& [id, row] : rows) {
            Index after = 0;
            for (std::uint32_t child : row.children) {
                const Index born = rows.at(child).first;
                if (row.first < born && born <= row.last) ++row.divisions;   // beside a mother still there
                else if (born > row.last) ++after;
            }
            if (after >= 2) ++row.divisions;
        }

        std::vector<TrackSummary> out;
        out.reserve(rows.size());
        for (auto& [id, row] : rows) out.push_back(std::move(row));
        return out;
    }

    Lineage lineageFromJson(const nlohmann::json& j) {
        Lineage out;
        if (!j.is_object()) return out;
        const auto id = [](const std::string& text, std::uint32_t& value) {
            if (text.empty() || text.size() > 10 || !std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; }))
                return false;
            const unsigned long long v = std::stoull(text);
            if (v == 0 || v > std::numeric_limits<std::uint32_t>::max()) return false;
            value = static_cast<std::uint32_t>(v);
            return true;
        };
        for (auto it = j.begin(); it != j.end(); ++it) {
            std::uint32_t child = 0, parent = 0;
            if (!id(it.key(), child)) continue;
            const nlohmann::json& v = it.value();
            if (v.is_number_unsigned() || (v.is_number_integer() && v.get<long long>() > 0)) {
                const unsigned long long p = v.get<unsigned long long>();
                if (p == 0 || p > std::numeric_limits<std::uint32_t>::max()) continue;
                parent = static_cast<std::uint32_t>(p);
            } else if (!(v.is_string() && id(v.get<std::string>(), parent))) {
                continue;
            }
            if (child != parent) out[child] = parent;
        }
        return out;
    }

    Index countDivisions(const std::vector<TrackSummary>& tracks) {
        Index n = 0;
        for (const TrackSummary& s : tracks) n += s.divisions;
        return n;
    }

} // namespace sirius::app
