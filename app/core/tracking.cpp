#include "core/tracking.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace sirius::app {

    namespace {
        // A finite stand-in for kNoAssignment inside the solver: large enough
        // that a forbidden pair is never preferred, small enough that the
        // potentials stay well away from overflow.
        constexpr double kBig = 1e12;
    } // namespace

    std::vector<int> solveAssignment(const std::vector<double>& cost, int rows, int cols, const std::function<void()>& poll) {
        if (rows < 0 || cols < 0) throw std::invalid_argument("solveAssignment: negative extent");
        if (static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) != cost.size())
            throw std::invalid_argument("solveAssignment: cost size does not match rows x cols");
        std::vector<int> assignment(static_cast<std::size_t>(std::max(rows, 0)), -1);
        if (rows == 0 || cols == 0) return assignment;

        // The classic O(n^3) formulation wants rows <= cols; transpose when it
        // does not hold and map the answer back.
        const bool flip = rows > cols;
        const int n = flip ? cols : rows, m = flip ? rows : cols;
        auto at = [&](int i, int j) {
            const double c = flip ? cost[static_cast<std::size_t>(j) * cols + i] : cost[static_cast<std::size_t>(i) * cols + j];
            return c == kNoAssignment || !(c < kBig) ? kBig : c;
        };

        // 1-based potentials and column assignment, shortest augmenting path.
        std::vector<double> u(static_cast<std::size_t>(n) + 1, 0.0), v(static_cast<std::size_t>(m) + 1, 0.0);
        std::vector<int> p(static_cast<std::size_t>(m) + 1, 0), way(static_cast<std::size_t>(m) + 1, 0);
        for (int i = 1; i <= n; ++i) {
            if (poll) poll();
            p[0] = i;
            int j0 = 0;
            std::vector<double> minv(static_cast<std::size_t>(m) + 1, std::numeric_limits<double>::max());
            std::vector<char> used(static_cast<std::size_t>(m) + 1, 0);
            do {
                used[static_cast<std::size_t>(j0)] = 1;
                const int i0 = p[static_cast<std::size_t>(j0)];
                double delta = std::numeric_limits<double>::max();
                int j1 = 0;
                for (int j = 1; j <= m; ++j) {
                    if (used[static_cast<std::size_t>(j)]) continue;
                    const double cur = at(i0 - 1, j - 1) - u[static_cast<std::size_t>(i0)] - v[static_cast<std::size_t>(j)];
                    if (cur < minv[static_cast<std::size_t>(j)]) {
                        minv[static_cast<std::size_t>(j)] = cur;
                        way[static_cast<std::size_t>(j)] = j0;
                    }
                    if (minv[static_cast<std::size_t>(j)] < delta) {
                        delta = minv[static_cast<std::size_t>(j)];
                        j1 = j;
                    }
                }
                for (int j = 0; j <= m; ++j) {
                    if (used[static_cast<std::size_t>(j)]) {
                        u[static_cast<std::size_t>(p[static_cast<std::size_t>(j)])] += delta;
                        v[static_cast<std::size_t>(j)] -= delta;
                    } else {
                        minv[static_cast<std::size_t>(j)] -= delta;
                    }
                }
                j0 = j1;
            } while (p[static_cast<std::size_t>(j0)] != 0);
            do {
                const int j1 = way[static_cast<std::size_t>(j0)];
                p[static_cast<std::size_t>(j0)] = p[static_cast<std::size_t>(j1)];
                j0 = j1;
            } while (j0);
        }

        for (int j = 1; j <= m; ++j) {
            const int i = p[static_cast<std::size_t>(j)];
            if (i <= 0) continue;
            const int row = flip ? j - 1 : i - 1, col = flip ? i - 1 : j - 1;
            // a pair that only the padding made attractive is not a match
            const double original = cost[static_cast<std::size_t>(row) * cols + col];
            if (original == kNoAssignment || !(original < kBig)) continue;
            assignment[static_cast<std::size_t>(row)] = col;
        }
        return assignment;
    }

    std::vector<TrackObject> objectsOfFrame(const LabelVolume& labels, Index t) {
        std::vector<TrackObject> out;
        if (t < 0 || t >= labels.t()) return out;
        const Index z = labels.z(), y = labels.y(), x = labels.x();
        const std::uint32_t* v = labels.volume(t);
        std::map<std::uint32_t, std::array<double, 4>> sums;   // z, y, x, count
        for (Index iz = 0; iz < z; ++iz)
            for (Index iy = 0; iy < y; ++iy) {
                const std::uint32_t* row = v + (iz * y + iy) * x;
                for (Index ix = 0; ix < x; ++ix) {
                    const std::uint32_t id = row[ix];
                    if (!id) continue;
                    std::array<double, 4>& s = sums[id];
                    s[0] += static_cast<double>(iz);
                    s[1] += static_cast<double>(iy);
                    s[2] += static_cast<double>(ix);
                    s[3] += 1.0;
                }
            }
        out.reserve(sums.size());
        for (const auto& [id, s] : sums) {
            TrackObject o;
            o.label = id;
            o.voxels = static_cast<Index>(s[3]);
            const double n = std::max(1.0, s[3]);
            o.centroid = {s[0] / n, s[1] / n, s[2] / n};
            out.push_back(o);
        }
        return out;
    }

    std::vector<Index> overlapBetween(const LabelVolume& labels, Index t, const std::vector<TrackObject>& a,
                                      const std::vector<TrackObject>& b) {
        std::vector<Index> counts(a.size() * b.size(), 0);
        if (a.empty() || b.empty() || t < 0 || t + 1 >= labels.t()) return counts;
        std::map<std::uint32_t, std::size_t> rowOf, colOf;
        for (std::size_t i = 0; i < a.size(); ++i) rowOf[a[i].label] = i;
        for (std::size_t j = 0; j < b.size(); ++j) colOf[b[j].label] = j;
        const Index n = labels.z() * labels.y() * labels.x();
        const std::uint32_t* p = labels.volume(t);
        const std::uint32_t* q = labels.volume(t + 1);
        for (Index i = 0; i < n; ++i) {
            if (!p[i] || !q[i]) continue;
            const auto r = rowOf.find(p[i]);
            const auto c = colOf.find(q[i]);
            if (r == rowOf.end() || c == colOf.end()) continue;
            ++counts[r->second * b.size() + c->second];
        }
        return counts;
    }

    namespace {
        double distanceUm(const TrackObject& a, const TrackObject& b, const std::array<double, 3>& voxelUm) {
            // voxelUm is (x, y, z); the centroid is (z, y, x)
            const double dz = (a.centroid[0] - b.centroid[0]) * voxelUm[2];
            const double dy = (a.centroid[1] - b.centroid[1]) * voxelUm[1];
            const double dx = (a.centroid[2] - b.centroid[2]) * voxelUm[0];
            return std::sqrt(dz * dz + dy * dy + dx * dx);
        }
    } // namespace

    TrackResult linkTracks(const std::vector<std::vector<TrackObject>>& byFrame,
                           const std::vector<std::vector<Index>>& overlap, const std::array<double, 3>& voxelUm,
                           const TrackOptions& options) {
        TrackResult result;
        const std::size_t frames = byFrame.size();
        if (frames == 0) return result;
        const double maxD = std::max(1e-9, options.maxDistanceUm);
        const double w = std::clamp(options.overlapWeight, 0.0, 1.0);

        // track index each object belongs to, per frame
        std::vector<std::vector<int>> trackOf(frames);
        for (std::size_t t = 0; t < frames; ++t) trackOf[t].assign(byFrame[t].size(), -1);
        std::vector<Track> tracks;

        auto startTrack = [&](std::size_t t, std::size_t i) {
            Track tr;
            tr.points.push_back({static_cast<Index>(t), byFrame[t][i].label});
            tracks.push_back(std::move(tr));
            trackOf[t][i] = static_cast<int>(tracks.size()) - 1;
            return trackOf[t][i];
        };

        const std::function<void()>& poll = options.poll;
        for (std::size_t t = 0; t + 1 < frames; ++t) {
            if (poll) poll();
            const std::vector<TrackObject>& a = byFrame[t];
            const std::vector<TrackObject>& b = byFrame[t + 1];
            if (a.empty() || b.empty()) continue;
            const std::vector<Index>* ov = t < overlap.size() && overlap[t].size() == a.size() * b.size() ? &overlap[t] : nullptr;
            std::vector<double> cost(a.size() * b.size(), kNoAssignment);
            for (std::size_t i = 0; i < a.size(); ++i)
                for (std::size_t j = 0; j < b.size(); ++j) {
                    const double d = distanceUm(a[i], b[j], voxelUm);
                    if (d > maxD) continue;
                    double c = d / maxD;
                    if (ov && w > 0.0) {
                        const double shared = static_cast<double>((*ov)[i * b.size() + j]);
                        const double denom = static_cast<double>(a[i].voxels + b[j].voxels) - shared;
                        const double iou = denom > 0.0 ? shared / denom : 0.0;
                        c = (1.0 - w) * c + w * (1.0 - iou);
                    }
                    cost[i * b.size() + j] = c;
                }
            const std::vector<int> match = solveAssignment(cost, static_cast<int>(a.size()), static_cast<int>(b.size()), poll);
            for (std::size_t i = 0; i < a.size(); ++i) {
                if (trackOf[t][i] < 0) startTrack(t, i);
                const int j = match[i];
                if (j < 0) continue;
                trackOf[t + 1][static_cast<std::size_t>(j)] = trackOf[t][i];
                tracks[static_cast<std::size_t>(trackOf[t][i])].points.push_back(
                    {static_cast<Index>(t + 1), b[static_cast<std::size_t>(j)].label});
                ++result.links;
            }
        }
        // anything still unattached starts its own track, in frame order
        for (std::size_t t = 0; t < frames; ++t)
            for (std::size_t i = 0; i < byFrame[t].size(); ++i)
                if (trackOf[t][i] < 0) startTrack(t, i);

        // Gap closing: a second assignment, this time between track ends and
        // track starts a few frames later, for objects a frame or two missed.
        if (options.maxGap > 0 && tracks.size() > 1) {
            auto objectAt = [&](std::size_t track, bool end) {
                const auto& pt = end ? tracks[track].points.back() : tracks[track].points.front();
                const std::vector<TrackObject>& frame = byFrame[static_cast<std::size_t>(pt.first)];
                for (const TrackObject& o : frame)
                    if (o.label == pt.second) return o;
                return TrackObject{};
            };
            std::vector<std::size_t> ends, starts;
            for (std::size_t k = 0; k < tracks.size(); ++k) {
                if (tracks[k].points.empty()) continue;
                ends.push_back(k);
                starts.push_back(k);
            }
            std::vector<double> cost(ends.size() * starts.size(), kNoAssignment);
            for (std::size_t r = 0; r < ends.size(); ++r) {
                if (poll) poll();
                for (std::size_t c = 0; c < starts.size(); ++c) {
                    const std::size_t from = ends[r], to = starts[c];
                    if (from == to) continue;
                    const Index gap = tracks[to].first() - tracks[from].last();
                    if (gap < 1 || gap > options.maxGap + 1) continue;
                    const double d = distanceUm(objectAt(from, true), objectAt(to, false), voxelUm);
                    // a longer gap must be a better match to be worth closing
                    if (d > maxD * static_cast<double>(gap)) continue;
                    cost[r * starts.size() + c] = d / maxD + 0.25 * static_cast<double>(gap - 1);
                }
            }
            const std::vector<int> match = solveAssignment(cost, static_cast<int>(ends.size()), static_cast<int>(starts.size()), poll);
            // Applied in order. A track that was merged away has had its
            // points moved into another one, so a later link out of it
            // continues from where they went: an object missed twice is one
            // track, not the first link and then nothing. Only the target of
            // a link is consumed; following the chain is what closes the rest.
            std::vector<std::size_t> mergedInto(tracks.size());
            for (std::size_t k = 0; k < tracks.size(); ++k) mergedInto[k] = k;
            std::vector<char> consumed(tracks.size(), 0);
            auto rootOf = [&](std::size_t k) {
                while (mergedInto[k] != k) k = mergedInto[k];
                return k;
            };
            for (std::size_t r = 0; r < ends.size(); ++r) {
                const int c = match[r];
                if (c < 0) continue;
                const std::size_t to = starts[static_cast<std::size_t>(c)];
                const std::size_t from = rootOf(ends[r]);
                if (from == to || consumed[to] || tracks[to].points.empty() || tracks[from].points.empty()) continue;
                if (tracks[to].first() <= tracks[from].last()) continue;
                tracks[from].points.insert(tracks[from].points.end(), tracks[to].points.begin(), tracks[to].points.end());
                tracks[to].points.clear();
                consumed[to] = 1;
                mergedInto[to] = from;
                ++result.gapsClosed;
            }
        }

        // drop the short ones, number what is left in order of appearance
        std::vector<Track> kept;
        for (Track& tr : tracks) {
            if (tr.points.empty() || static_cast<Index>(tr.points.size()) < std::max<Index>(1, options.minLength)) continue;
            std::sort(tr.points.begin(), tr.points.end());
            kept.push_back(std::move(tr));
        }
        std::stable_sort(kept.begin(), kept.end(), [](const Track& a, const Track& b) {
            return a.first() != b.first() ? a.first() < b.first() : a.points.front().second < b.points.front().second;
        });
        for (std::size_t k = 0; k < kept.size(); ++k) kept[k].id = static_cast<std::uint32_t>(k + 1);
        result.tracks = std::move(kept);
        return result;
    }

} // namespace sirius::app
