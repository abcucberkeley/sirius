#include "core/labels.hpp"

#include <algorithm>
#include <array>
#include <deque>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace sirius::app {

    namespace {
        constexpr float kInf = std::numeric_limits<float>::infinity();

        // The design's label palette, cycled by id.
        constexpr std::array<std::array<float, 3>, 7> kPalette{{
            {0xff / 255.f, 0xb3 / 255.f, 0x47 / 255.f},
            {0x7c / 255.f, 0x9c / 255.f, 0xff / 255.f},
            {0xe8 / 255.f, 0x71 / 255.f, 0xd9 / 255.f},
            {0x63 / 255.f, 0xe0 / 255.f, 0x8a / 255.f},
            {0xff / 255.f, 0x5c / 255.f, 0x7a / 255.f},
            {0x6e / 255.f, 0xe7 / 255.f, 0xf2 / 255.f},
            {0xf2 / 255.f, 0xe3 / 255.f, 0x5c / 255.f},
        }};

        void requireExtent(Index z, Index y, Index x, const char* what) {
            if (z < 1 || y < 1 || x < 1)
                throw std::invalid_argument(std::string(what) + ": extents must be >= 1, got " + std::to_string(z) +
                                            " x " + std::to_string(y) + " x " + std::to_string(x));
        }

        // Union-find over provisional component labels (path halving).
        struct UnionFind {
            std::vector<std::uint32_t> parent;
            std::uint32_t find(std::uint32_t a) noexcept {
                while (parent[a] != a) {
                    parent[a] = parent[parent[a]];
                    a = parent[a];
                }
                return a;
            }
            void unite(std::uint32_t a, std::uint32_t b) noexcept {
                a = find(a);
                b = find(b);
                if (a == b) return;
                if (a < b) parent[b] = a;
                else parent[a] = b;
            }
            std::uint32_t make() {
                parent.push_back(static_cast<std::uint32_t>(parent.size()));
                return static_cast<std::uint32_t>(parent.size() - 1);
            }
        };

        // 1D squared Euclidean distance transform (Felzenszwalb & Huttenlocher
        // 2012) of f (INF where no source yet) along a strided line.
        void edt1d(const float* f, Index n, Index stride, float* out, std::vector<Index>& v, std::vector<double>& z,
                   std::vector<double>& g) {
            v.resize(static_cast<std::size_t>(n));
            z.resize(static_cast<std::size_t>(n) + 1);
            g.resize(static_cast<std::size_t>(n));
            for (Index i = 0; i < n; ++i) g[static_cast<std::size_t>(i)] = f[i * stride];
            Index k = 0;
            v[0] = 0;
            z[0] = -std::numeric_limits<double>::infinity();
            z[1] = std::numeric_limits<double>::infinity();
            for (Index q = 1; q < n; ++q) {
                const double gq = g[static_cast<std::size_t>(q)];
                if (std::isinf(gq)) continue;   // no parabola for an unreachable sample
                double s;
                for (;;) {
                    const Index p = v[static_cast<std::size_t>(k)];
                    const double gp = g[static_cast<std::size_t>(p)];
                    if (std::isinf(gp)) {
                        // the running lower envelope holds only infinities: restart from q
                        s = -std::numeric_limits<double>::infinity();
                        k = -1;
                        break;
                    }
                    s = ((gq + static_cast<double>(q) * q) - (gp + static_cast<double>(p) * p)) /
                        (2.0 * static_cast<double>(q - p));
                    if (s > z[static_cast<std::size_t>(k)]) break;
                    --k;
                    if (k < 0) {
                        s = -std::numeric_limits<double>::infinity();
                        break;
                    }
                }
                ++k;
                v[static_cast<std::size_t>(k)] = q;
                z[static_cast<std::size_t>(k)] = s;
                z[static_cast<std::size_t>(k) + 1] = std::numeric_limits<double>::infinity();
            }
            k = 0;
            for (Index q = 0; q < n; ++q) {
                while (z[static_cast<std::size_t>(k) + 1] < static_cast<double>(q)) ++k;
                const Index p = v[static_cast<std::size_t>(k)];
                const double gp = g[static_cast<std::size_t>(p)];
                const double d = static_cast<double>(q - p);
                out[q * stride] = std::isinf(gp) ? kInf : static_cast<float>(d * d + gp);
            }
        }

        struct Voxel {
            Index z, y, x;
        };
    } // namespace

    // --- LabelStats ---------------------------------------------------------------

    std::string LabelStats::flagText() const { return flags.empty() ? std::string() : flags.front(); }

    // --- LabelVolume --------------------------------------------------------------

    LabelVolume::LabelVolume() : data_(std::make_shared<Buffer<std::uint32_t>>()) {}

    LabelVolume::LabelVolume(Index t, Index z, Index y, Index x) : t_(t), z_(z), y_(y), x_(x) {
        if (t < 1) throw std::invalid_argument("LabelVolume: t must be >= 1");
        requireExtent(z, y, x, "LabelVolume");
        data_ = std::make_shared<Buffer<std::uint32_t>>(Shape{t, z, y, x});
        std::fill(data_->data(), data_->data() + data_->size(), 0u);
    }

    void LabelVolume::detach() {
        // Copy-on-write: the voxels are shared with another volume (share())
        // until the first write, which lands in a private copy.
        if (data_.use_count() > 1) data_ = std::make_shared<Buffer<std::uint32_t>>(data_->clone());
    }

    std::uint32_t* LabelVolume::volume(Index t) {
        detach();
        return data_->data() + t * volumeSize();
    }
    const std::uint32_t* LabelVolume::volume(Index t) const noexcept { return data_->data() + t * volumeSize(); }
    std::uint32_t* LabelVolume::plane(Index t, Index z) { return volume(t) + z * y_ * x_; }
    const std::uint32_t* LabelVolume::plane(Index t, Index z) const noexcept { return volume(t) + z * y_ * x_; }
    std::uint32_t LabelVolume::at(Index t, Index z, Index y, Index x) const noexcept { return plane(t, z)[y * x_ + x]; }

    std::uint32_t LabelVolume::maxLabel() const noexcept { return maxLabel_; }

    void LabelVolume::resetMaxLabel() noexcept {
        std::uint32_t m = 0;
        const std::uint32_t* v = data_->data();
        const Index n = data_->size();
        for (Index i = 0; i < n; ++i) m = std::max(m, v[i]);
        maxLabel_ = m;
    }

    void LabelVolume::recomputeStats(Index t, const float* probabilities) {
        if (t < 0 || t >= t_) throw std::out_of_range("LabelVolume::recomputeStats: t out of range");
        // annotations survive a recompute: keep class / reviewed of known ids
        std::unordered_map<std::uint32_t, LabelStats> previous;
        for (const LabelStats& s : stats_) previous.emplace(s.id, s);

        const std::uint32_t* v = static_cast<const LabelVolume&>(*this).volume(t);   // read only: no detach
        const Index n = volumeSize();
        std::uint32_t maxId = 0;
        for (Index i = 0; i < n; ++i) maxId = std::max(maxId, v[i]);
        maxLabel_ = std::max(maxLabel_, maxId);

        struct Acc {
            Index voxels = 0;
            double prob = 0.0;
            Index z0 = std::numeric_limits<Index>::max(), z1 = -1, y0 = std::numeric_limits<Index>::max(), y1 = -1,
                  x0 = std::numeric_limits<Index>::max(), x1 = -1;
        };
        std::vector<Acc> acc(static_cast<std::size_t>(maxId) + 1);
        for (Index z = 0; z < z_; ++z)
            for (Index y = 0; y < y_; ++y) {
                const std::uint32_t* row = v + (z * y_ + y) * x_;
                const float* prow = probabilities ? probabilities + (z * y_ + y) * x_ : nullptr;
                for (Index x = 0; x < x_; ++x) {
                    const std::uint32_t id = row[x];
                    if (!id) continue;
                    Acc& a = acc[id];
                    ++a.voxels;
                    if (prow) a.prob += prow[x];
                    a.z0 = std::min(a.z0, z);
                    a.z1 = std::max(a.z1, z);
                    a.y0 = std::min(a.y0, y);
                    a.y1 = std::max(a.y1, y);
                    a.x0 = std::min(a.x0, x);
                    a.x1 = std::max(a.x1, x);
                }
            }

        stats_.clear();
        for (std::uint32_t id = 1; id <= maxId; ++id) {
            const Acc& a = acc[id];
            if (!a.voxels) continue;
            LabelStats s;
            auto it = previous.find(id);
            if (it != previous.end()) {
                s.cls = it->second.cls;
                s.reviewed = it->second.reviewed;
                // The mean probability came with the segmentation; a later
                // pass without one (cleanup, crop, tracking) keeps it, as
                // updateStats does, rather than promoting every label to 1.
                s.confidence = it->second.confidence;
            }
            s.id = id;
            s.voxels = a.voxels;
            if (probabilities) s.confidence = a.prob / static_cast<double>(a.voxels);
            s.bbox = {a.z0, a.z1 + 1, a.y0, a.y1 + 1, a.x0, a.x1 + 1};
            // a single plane has no z border to touch
            s.touchesBorder = a.y0 == 0 || a.y1 == y_ - 1 || a.x0 == 0 || a.x1 == x_ - 1 ||
                              (z_ > 1 && (a.z0 == 0 || a.z1 == z_ - 1));
            stats_.push_back(std::move(s));
        }
        statsT_ = t;
        // the flags describe the new statistics with the rules last given
        if (flagRules_) applyFlags(*flagRules_);
    }

    void LabelVolume::updateStats(const LabelDiff& diff) {
        if (diff.empty()) return;
        if (diff.t < 0 || diff.t >= t_) throw std::out_of_range("LabelVolume::updateStats: t out of range");
        if (statsT_ != diff.t) {
            recomputeStats(diff.t);
            return;
        }
        // The labels the diff touched and the box it spans: each touched
        // label is rescanned within its old bounding box joined with that
        // box, which is where every voxel it has now must lie.
        std::vector<std::uint32_t> touched;
        const Index plane = y_ * x_, n = volumeSize();
        std::array<Index, 6> box{z_, -1, y_, -1, x_, -1};
        for (std::size_t k = 0; k < diff.indices.size(); ++k) {
            const Index i = diff.indices[k];
            if (i < 0 || i >= n) throw std::out_of_range("LabelVolume::updateStats: index outside the volume");
            const Index z = i / plane, y = (i / x_) % y_, x = i % x_;
            box[0] = std::min(box[0], z);
            box[1] = std::max(box[1], z);
            box[2] = std::min(box[2], y);
            box[3] = std::max(box[3], y);
            box[4] = std::min(box[4], x);
            box[5] = std::max(box[5], x);
            if (diff.before[k]) touched.push_back(diff.before[k]);
            if (diff.after[k]) touched.push_back(diff.after[k]);
        }
        std::sort(touched.begin(), touched.end());
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
        const std::uint32_t* v = static_cast<const LabelVolume&>(*this).volume(diff.t);
        for (std::uint32_t id : touched) {
            LabelStats* known = mutableStatsOf(id);
            Index z0 = box[0], z1 = box[1], y0 = box[2], y1 = box[3], x0 = box[4], x1 = box[5];
            if (known) {
                z0 = std::min(z0, known->bbox[0]);
                z1 = std::max(z1, known->bbox[1] - 1);
                y0 = std::min(y0, known->bbox[2]);
                y1 = std::max(y1, known->bbox[3] - 1);
                x0 = std::min(x0, known->bbox[4]);
                x1 = std::max(x1, known->bbox[5] - 1);
            }
            Index count = 0;
            Index bz0 = z_, bz1 = -1, by0 = y_, by1 = -1, bx0 = x_, bx1 = -1;
            for (Index z = z0; z <= z1; ++z)
                for (Index y = y0; y <= y1; ++y) {
                    const std::uint32_t* row = v + (z * y_ + y) * x_;
                    for (Index x = x0; x <= x1; ++x) {
                        if (row[x] != id) continue;
                        ++count;
                        bz0 = std::min(bz0, z);
                        bz1 = std::max(bz1, z);
                        by0 = std::min(by0, y);
                        by1 = std::max(by1, y);
                        bx0 = std::min(bx0, x);
                        bx1 = std::max(bx1, x);
                    }
                }
            if (count == 0) {
                stats_.erase(std::remove_if(stats_.begin(), stats_.end(), [id](const LabelStats& s) { return s.id == id; }),
                             stats_.end());
                continue;
            }
            if (!known) {
                LabelStats s;
                s.id = id;
                // keep the table ordered by id, as recomputeStats leaves it
                auto at = std::find_if(stats_.begin(), stats_.end(), [id](const LabelStats& o) { return o.id > id; });
                known = &*stats_.insert(at, std::move(s));
            }
            known->voxels = count;
            known->bbox = {bz0, bz1 + 1, by0, by1 + 1, bx0, bx1 + 1};
            known->touchesBorder = by0 == 0 || by1 == y_ - 1 || bx0 == 0 || bx1 == x_ - 1 ||
                                   (z_ > 1 && (bz0 == 0 || bz1 == z_ - 1));
        }
        if (flagRules_) applyFlags(*flagRules_);
    }

    const LabelStats* LabelVolume::statsOf(std::uint32_t id) const noexcept {
        for (const LabelStats& s : stats_)
            if (s.id == id) return &s;
        return nullptr;
    }

    LabelStats* LabelVolume::mutableStatsOf(std::uint32_t id) noexcept {
        for (LabelStats& s : stats_)
            if (s.id == id) return &s;
        return nullptr;
    }

    void LabelVolume::applyFlags(const LabelFlagRules& rules) {
        flagRules_ = rules;
        if (stats_.empty()) return;
        std::vector<Index> sizes;
        sizes.reserve(stats_.size());
        for (const LabelStats& s : stats_) sizes.push_back(s.voxels);
        std::nth_element(sizes.begin(), sizes.begin() + static_cast<std::ptrdiff_t>(sizes.size() / 2), sizes.end());
        const Index median = sizes[sizes.size() / 2];
        const Index minVoxels = rules.minVoxels > 0 ? rules.minVoxels : std::max<Index>(1, median / 8);
        for (LabelStats& s : stats_) {
            s.flags.clear();
            if (s.confidence < rules.lowConfidence) s.flags.push_back("low conf");
            if (s.voxels < minVoxels) s.flags.push_back("small");
            if (rules.flagBorder && s.touchesBorder) s.flags.push_back("touching border");
            if (rules.sizeOutlierFactor > 0.0 && static_cast<double>(s.voxels) > rules.sizeOutlierFactor * static_cast<double>(median))
                s.flags.push_back("merged?");
        }
    }

    Index LabelVolume::reviewedCount() const noexcept {
        return static_cast<Index>(std::count_if(stats_.begin(), stats_.end(), [](const LabelStats& s) { return s.reviewed; }));
    }

    Index LabelVolume::flaggedCount(const std::string& flag) const noexcept {
        Index n = 0;
        for (const LabelStats& s : stats_)
            if (std::find(s.flags.begin(), s.flags.end(), flag) != s.flags.end()) ++n;
        return n;
    }

    // --- edits ----------------------------------------------------------------------

    LabelDiff LabelVolume::paint(Index t, Index cz, Index cy, Index cx, double radius, Index zRadius,
                                 std::uint32_t label, std::uint32_t onlyLabel) {
        LabelDiff diff;
        diff.t = t;
        if (t < 0 || t >= t_) throw std::out_of_range("LabelVolume::paint: t out of range");
        edited_ = true;
        std::uint32_t* v = volume(t);
        const double r = std::max(radius, 0.0);
        const Index ri = static_cast<Index>(std::ceil(r));
        zRadius = std::max<Index>(zRadius, 0);
        const double rz = static_cast<double>(zRadius);
        for (Index z = std::max<Index>(cz - zRadius, 0); z <= std::min<Index>(cz + zRadius, z_ - 1); ++z) {
            const double fz = rz > 0.0 ? static_cast<double>(z - cz) / rz : 0.0;
            for (Index y = std::max<Index>(cy - ri, 0); y <= std::min<Index>(cy + ri, y_ - 1); ++y)
                for (Index x = std::max<Index>(cx - ri, 0); x <= std::min<Index>(cx + ri, x_ - 1); ++x) {
                    const double dy = static_cast<double>(y - cy), dx = static_cast<double>(x - cx);
                    // an ellipsoid: (dx² + dy²) / r² + (dz / zRadius)² <= 1, a single
                    // voxel when the radius is below half a voxel
                    const double lateral = r >= 0.5 ? (dx * dx + dy * dy) / (r * r) : (dx == 0.0 && dy == 0.0 ? 0.0 : 2.0);
                    if (lateral + fz * fz > 1.0 + 1e-9) continue;
                    const Index i = (z * y_ + y) * x_ + x;
                    const std::uint32_t cur = v[i];
                    if (cur == label) continue;
                    if (onlyLabel && cur != onlyLabel) continue;
                    diff.indices.push_back(i);
                    diff.before.push_back(cur);
                    diff.after.push_back(label);
                    v[i] = label;
                }
        }
        maxLabel_ = std::max(maxLabel_, label);
        return diff;
    }

    LabelDiff LabelVolume::fill(Index t, Index z, Index y, Index x, std::uint32_t label) {
        LabelDiff diff;
        diff.t = t;
        if (t < 0 || t >= t_) throw std::out_of_range("LabelVolume::fill: t out of range");
        if (z < 0 || z >= z_ || y < 0 || y >= y_ || x < 0 || x >= x_)
            throw std::out_of_range("LabelVolume::fill: seed outside the volume");
        edited_ = true;
        std::uint32_t* v = volume(t);
        const Index seed = (z * y_ + y) * x_ + x;
        const std::uint32_t from = v[seed];
        if (from == label) return diff;
        // the changed value doubles as the visited mark
        std::vector<Index> stack{seed};
        v[seed] = label;
        diff.indices.push_back(seed);
        const Index plane = y_ * x_;
        while (!stack.empty()) {
            const Index i = stack.back();
            stack.pop_back();
            const Index iz = i / plane, iy = (i / x_) % y_, ix = i % x_;
            const Index nb[6] = {iz > 0 ? i - plane : -1, iz + 1 < z_ ? i + plane : -1,
                                 iy > 0 ? i - x_ : -1, iy + 1 < y_ ? i + x_ : -1,
                                 ix > 0 ? i - 1 : -1, ix + 1 < x_ ? i + 1 : -1};
            for (Index j : nb) {
                if (j < 0 || v[j] != from) continue;
                v[j] = label;
                diff.indices.push_back(j);
                stack.push_back(j);
            }
        }
        diff.before.assign(diff.indices.size(), from);
        diff.after.assign(diff.indices.size(), label);
        maxLabel_ = std::max(maxLabel_, label);
        return diff;
    }

    LabelDiff LabelVolume::merge(Index t, const std::vector<std::uint32_t>& ids) {
        LabelDiff diff;
        diff.t = t;
        if (t < 0 || t >= t_) throw std::out_of_range("LabelVolume::merge: t out of range");
        std::vector<std::uint32_t> sources;
        for (std::uint32_t id : ids)
            if (id) sources.push_back(id);
        if (sources.size() < 2) return diff;
        std::sort(sources.begin(), sources.end());
        sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
        const std::uint32_t target = sources.front();
        edited_ = true;
        std::uint32_t* v = volume(t);
        const Index n = volumeSize();
        for (Index i = 0; i < n; ++i) {
            const std::uint32_t cur = v[i];
            if (cur == target || cur == 0) continue;
            if (!std::binary_search(sources.begin(), sources.end(), cur)) continue;
            diff.indices.push_back(i);
            diff.before.push_back(cur);
            diff.after.push_back(target);
            v[i] = target;
        }
        return diff;
    }

    LabelDiff LabelVolume::remove(Index t, std::uint32_t id) {
        LabelDiff diff;
        diff.t = t;
        if (t < 0 || t >= t_) throw std::out_of_range("LabelVolume::remove: t out of range");
        if (!id) return diff;
        edited_ = true;
        std::uint32_t* v = volume(t);
        const Index n = volumeSize();
        for (Index i = 0; i < n; ++i) {
            if (v[i] != id) continue;
            diff.indices.push_back(i);
            diff.before.push_back(id);
            diff.after.push_back(0);
            v[i] = 0;
        }
        return diff;
    }

    LabelDiff LabelVolume::split(Index t, std::uint32_t id, std::array<Index, 3> seedA, std::array<Index, 3> seedB) {
        LabelDiff diff;
        diff.t = t;
        if (t < 0 || t >= t_) throw std::out_of_range("LabelVolume::split: t out of range");
        if (!id) throw std::invalid_argument("LabelVolume::split: cannot split the background");
        edited_ = true;
        std::uint32_t* v = volume(t);
        auto inside = [&](const std::array<Index, 3>& s) {
            return s[0] >= 0 && s[0] < z_ && s[1] >= 0 && s[1] < y_ && s[2] >= 0 && s[2] < x_ &&
                   v[(s[0] * y_ + s[1]) * x_ + s[2]] == id;
        };
        if (!inside(seedA) || !inside(seedB))
            throw std::invalid_argument("LabelVolume::split: both seeds must lie inside label " + std::to_string(id));

        // bounding box of the label, padded by one background voxel so the
        // distance transform sees the object's boundary on every side
        Index z0 = z_, z1 = -1, y0 = y_, y1 = -1, x0 = x_, x1 = -1;
        for (Index z = 0; z < z_; ++z)
            for (Index y = 0; y < y_; ++y) {
                const std::uint32_t* row = v + (z * y_ + y) * x_;
                for (Index x = 0; x < x_; ++x)
                    if (row[x] == id) {
                        z0 = std::min(z0, z);
                        z1 = std::max(z1, z);
                        y0 = std::min(y0, y);
                        y1 = std::max(y1, y);
                        x0 = std::min(x0, x);
                        x1 = std::max(x1, x);
                    }
            }
        if (z1 < 0) return diff;
        const Index bz = z1 - z0 + 3, by = y1 - y0 + 3, bx = x1 - x0 + 3;   // one voxel of padding each side
        const Index oz = z0 - 1, oy = y0 - 1, ox = x0 - 1;
        const Index bn = bz * by * bx;
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(bn), 0);
        for (Index z = z0; z <= z1; ++z)
            for (Index y = y0; y <= y1; ++y)
                for (Index x = x0; x <= x1; ++x)
                    if (v[(z * y_ + y) * x_ + x] == id)
                        mask[static_cast<std::size_t>(((z - oz) * by + (y - oy)) * bx + (x - ox))] = 1;

        std::vector<float> dist(static_cast<std::size_t>(bn));
        distanceTransform(mask.data(), bz, by, bx, dist.data());
        // ridges of the watershed are the thin necks: flood from deep inside
        for (float& d : dist) d = -d;

        const std::uint32_t newId = maxLabel_ + 1;
        std::vector<std::uint32_t> labels(static_cast<std::size_t>(bn), 0);
        labels[static_cast<std::size_t>(((seedA[0] - oz) * by + (seedA[1] - oy)) * bx + (seedA[2] - ox))] = id;
        labels[static_cast<std::size_t>(((seedB[0] - oz) * by + (seedB[1] - oy)) * bx + (seedB[2] - ox))] = newId;
        watershed(dist.data(), mask.data(), bz, by, bx, labels.data());

        for (Index z = z0; z <= z1; ++z)
            for (Index y = y0; y <= y1; ++y)
                for (Index x = x0; x <= x1; ++x) {
                    const std::uint32_t l = labels[static_cast<std::size_t>(((z - oz) * by + (y - oy)) * bx + (x - ox))];
                    if (l != newId) continue;
                    const Index i = (z * y_ + y) * x_ + x;
                    diff.indices.push_back(i);
                    diff.before.push_back(id);
                    diff.after.push_back(newId);
                    v[i] = newId;
                }
        if (!diff.empty()) maxLabel_ = newId;
        return diff;
    }

    void LabelVolume::apply(const LabelDiff& diff, bool forward) {
        if (diff.t < 0 || diff.t >= t_) throw std::out_of_range("LabelVolume::apply: t out of range");
        if (diff.before.size() != diff.indices.size() || diff.after.size() != diff.indices.size())
            throw std::invalid_argument("LabelVolume::apply: malformed diff");
        edited_ = true;
        std::uint32_t* v = volume(diff.t);
        const std::vector<std::uint32_t>& values = forward ? diff.after : diff.before;
        const Index n = volumeSize();
        // A stroke's diff is the concatenation of every mouse move, so one
        // voxel may appear twice; replaying it backwards in reverse order
        // restores the value it had before the first touch.
        const std::size_t count = diff.indices.size();
        for (std::size_t step = 0; step < count; ++step) {
            const std::size_t k = forward ? step : count - 1 - step;
            const Index i = diff.indices[k];
            if (i < 0 || i >= n) throw std::out_of_range("LabelVolume::apply: index outside the volume");
            v[i] = values[k];
            maxLabel_ = std::max(maxLabel_, values[k]);
        }
    }

    std::shared_ptr<LabelVolume> LabelVolume::clone() const {
        auto c = share();
        c->detach();
        return c;
    }

    std::shared_ptr<LabelVolume> LabelVolume::share() const {
        auto c = std::make_shared<LabelVolume>();
        c->t_ = t_;
        c->z_ = z_;
        c->y_ = y_;
        c->x_ = x_;
        c->data_ = data_;
        c->stats_ = stats_;
        c->statsT_ = statsT_;
        c->flagRules_ = flagRules_;
        c->maxLabel_ = maxLabel_;
        c->tracked_ = tracked_;
        return c;
    }

    // --- algorithms -----------------------------------------------------------------

    std::uint32_t connectedComponents(const std::uint8_t* mask, Index z, Index y, Index x, std::uint32_t* out) {
        requireExtent(z, y, x, "connectedComponents");
        // Two passes: a raster scan joins each foreground voxel to its already
        // visited neighbours (-z, -y, -x) through a union-find of provisional
        // labels, then the roots are renumbered densely. Sequential, but one
        // cache-friendly sweep, which is what makes 512³ affordable.
        UnionFind uf;
        uf.make();   // 0 = background
        const Index plane = y * x;
        for (Index iz = 0; iz < z; ++iz)
            for (Index iy = 0; iy < y; ++iy) {
                const Index base = (iz * y + iy) * x;
                for (Index ix = 0; ix < x; ++ix) {
                    const Index i = base + ix;
                    if (!mask[i]) {
                        out[i] = 0;
                        continue;
                    }
                    std::uint32_t label = 0;
                    const std::uint32_t nz = iz > 0 ? out[i - plane] : 0;
                    const std::uint32_t ny = iy > 0 ? out[i - x] : 0;
                    const std::uint32_t nx = ix > 0 ? out[i - 1] : 0;
                    if (nx) label = nx;
                    if (ny) {
                        if (label) uf.unite(label, ny);
                        else label = ny;
                    }
                    if (nz) {
                        if (label) uf.unite(label, nz);
                        else label = nz;
                    }
                    if (!label) label = uf.make();
                    out[i] = label;
                }
            }
        // dense renumbering of the roots
        std::vector<std::uint32_t> dense(uf.parent.size(), 0);
        std::uint32_t count = 0;
        for (std::uint32_t l = 1; l < uf.parent.size(); ++l) {
            const std::uint32_t r = uf.find(l);
            if (r == l) dense[l] = ++count;
        }
        const Index n = z * plane;
        for (Index i = 0; i < n; ++i)
            if (out[i]) out[i] = dense[uf.find(out[i])];
        return count;
    }

    void watershed(const float* landscape, const std::uint8_t* mask, Index z, Index y, Index x,
                   std::uint32_t* labels) {
        requireExtent(z, y, x, "watershed");
        // Meyer's flooding: seeds enter a priority queue keyed by height, the
        // lowest voxel expands into its unlabelled masked neighbours, which
        // enter the queue at their own height. Insertion order breaks ties so
        // plateaus grow evenly from every side.
        using Item = std::tuple<float, std::uint64_t, Index>;
        std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
        std::uint64_t seq = 0;
        const Index plane = y * x, n = z * plane;
        for (Index i = 0; i < n; ++i) {
            if (!mask[i]) {
                labels[i] = 0;
                continue;
            }
            if (labels[i]) queue.emplace(landscape[i], seq++, i);
        }
        while (!queue.empty()) {
            const Index i = std::get<2>(queue.top());
            queue.pop();
            const std::uint32_t label = labels[i];
            const Index iz = i / plane, iy = (i / x) % y, ix = i % x;
            const Index nb[6] = {iz > 0 ? i - plane : -1, iz + 1 < z ? i + plane : -1,
                                 iy > 0 ? i - x : -1, iy + 1 < y ? i + x : -1,
                                 ix > 0 ? i - 1 : -1, ix + 1 < x ? i + 1 : -1};
            for (Index j : nb) {
                if (j < 0 || !mask[j] || labels[j]) continue;
                labels[j] = label;
                queue.emplace(landscape[j], seq++, j);
            }
        }
    }

    void distanceTransform(const std::uint8_t* mask, Index z, Index y, Index x, float* out) {
        requireExtent(z, y, x, "distanceTransform");
        const Index plane = y * x, n = z * plane;
        for (Index i = 0; i < n; ++i) out[i] = mask[i] ? kInf : 0.0f;
        // separable exact squared EDT: one 1D pass per axis, each in place
#pragma omp parallel
        {
            std::vector<Index> v;
            std::vector<double> zz, g;
#pragma omp for collapse(2) schedule(static)
            for (Index iz = 0; iz < z; ++iz)
                for (Index iy = 0; iy < y; ++iy) {
                    float* line = out + (iz * y + iy) * x;
                    edt1d(line, x, 1, line, v, zz, g);
                }
#pragma omp for collapse(2) schedule(static)
            for (Index iz = 0; iz < z; ++iz)
                for (Index ix = 0; ix < x; ++ix) {
                    float* line = out + iz * plane + ix;
                    edt1d(line, y, x, line, v, zz, g);
                }
            if (z > 1) {
#pragma omp for collapse(2) schedule(static)
                for (Index iy = 0; iy < y; ++iy)
                    for (Index ix = 0; ix < x; ++ix) {
                        float* line = out + iy * x + ix;
                        edt1d(line, z, plane, line, v, zz, g);
                    }
            }
        }
        // no background anywhere: every voxel is as far as the volume is wide
        const float far = static_cast<float>(std::max({z, y, x}));
#pragma omp parallel for schedule(static)
        for (Index i = 0; i < n; ++i) out[i] = std::isinf(out[i]) ? far : std::sqrt(out[i]);
    }

    // Vincent's hybrid reconstruction: a raster and an anti-raster sweep, then
    // a queue for the voxels the sweeps left unfinished. Linear in practice,
    // unlike iterating geodesic dilations to stability.
    void reconstructByDilation(float* marker, const float* mask, Index z, Index y, Index x) {
        requireExtent(z, y, x, "reconstructByDilation");
        const Index plane = y * x, n = z * plane;
        for (Index i = 0; i < n; ++i) marker[i] = std::min(marker[i], mask[i]);
        const Index step[3] = {plane, x, 1};
        auto sweep = [&](bool forward) {
            for (Index k = 0; k < n; ++k) {
                const Index i = forward ? k : n - 1 - k;
                const Index iz = i / plane, iy = (i % plane) / x, ix = i % x;
                const Index at[3] = {iz, iy, ix};
                const Index extent[3] = {z, y, x};
                float v = marker[i];
                for (int a = 0; a < 3; ++a) {
                    const Index j = forward ? at[a] - 1 : at[a] + 1;
                    if (j < 0 || j >= extent[a]) continue;
                    v = std::max(v, marker[forward ? i - step[a] : i + step[a]]);
                }
                marker[i] = std::min(v, mask[i]);
            }
        };
        sweep(true);
        sweep(false);
        // queue the voxels whose backward neighbours can still grow
        std::deque<Index> queue;
        for (Index i = 0; i < n; ++i) {
            const Index iz = i / plane, iy = (i % plane) / x, ix = i % x;
            const Index at[3] = {iz, iy, ix};
            const Index extent[3] = {z, y, x};
            for (int a = 0; a < 3; ++a) {
                if (at[a] + 1 >= extent[a]) continue;
                const Index j = i + step[a];
                if (marker[j] < marker[i] && marker[j] < mask[j]) {
                    queue.push_back(i);
                    break;
                }
            }
        }
        while (!queue.empty()) {
            const Index i = queue.front();
            queue.pop_front();
            const Index iz = i / plane, iy = (i % plane) / x, ix = i % x;
            const Index at[3] = {iz, iy, ix};
            const Index extent[3] = {z, y, x};
            for (int a = 0; a < 3; ++a)
                for (int d = -1; d <= 1; d += 2) {
                    const Index j2 = at[a] + d;
                    if (j2 < 0 || j2 >= extent[a]) continue;
                    const Index j = i + d * step[a];
                    if (marker[j] < marker[i] && mask[j] != marker[j]) {
                        marker[j] = std::min(marker[i], mask[j]);
                        queue.push_back(j);
                    }
                }
        }
    }

    std::uint32_t hMaximaSeeds(const float* values, const std::uint8_t* mask, Index z, Index y, Index x, double h,
                               std::uint32_t* out) {
        requireExtent(z, y, x, "hMaximaSeeds");
        const Index plane = y * x, n = z * plane;
        std::fill(out, out + n, 0u);
        const float depth = static_cast<float>(std::max(h, 1e-6));
        // the h-maxima transform: reconstruct (values - h) under values, so
        // every maximum shallower than h is filled in and only the deep ones
        // remain as plateaus
        std::vector<float> g(static_cast<std::size_t>(n));
        for (Index i = 0; i < n; ++i) g[static_cast<std::size_t>(i)] = mask[i] ? values[i] - depth : 0.0f;
        std::vector<float> under(static_cast<std::size_t>(n));
        for (Index i = 0; i < n; ++i) under[static_cast<std::size_t>(i)] = mask[i] ? values[i] : 0.0f;
        reconstructByDilation(g.data(), under.data(), z, y, x);
        // regional maxima of g: plateaus of equal value with no higher neighbour
        const Index step[3] = {plane, x, 1};
        std::vector<std::uint8_t> seen(static_cast<std::size_t>(n), 0);
        std::vector<Index> plateau, stack;
        std::uint32_t count = 0;
        for (Index start = 0; start < n; ++start) {
            if (seen[static_cast<std::size_t>(start)] || !mask[start] || !(g[static_cast<std::size_t>(start)] > 0.0f)) continue;
            const float level = g[static_cast<std::size_t>(start)];
            plateau.clear();
            stack.assign(1, start);
            seen[static_cast<std::size_t>(start)] = 1;
            bool maximal = true;
            while (!stack.empty()) {
                const Index i = stack.back();
                stack.pop_back();
                plateau.push_back(i);
                const Index at[3] = {i / plane, (i % plane) / x, i % x};
                const Index extent[3] = {z, y, x};
                for (int a = 0; a < 3; ++a)
                    for (int d = -1; d <= 1; d += 2) {
                        const Index j2 = at[a] + d;
                        if (j2 < 0 || j2 >= extent[a]) continue;
                        const Index j = i + d * step[a];
                        if (!mask[j]) continue;
                        const float gv = g[static_cast<std::size_t>(j)];
                        if (gv > level) maximal = false;
                        else if (gv == level && !seen[static_cast<std::size_t>(j)]) {
                            seen[static_cast<std::size_t>(j)] = 1;
                            stack.push_back(j);
                        }
                    }
            }
            if (!maximal) continue;
            ++count;
            for (Index i : plateau) out[i] = count;
        }
        return count;
    }

    void gaussianVolume(std::vector<float>& v, Index z, Index y, Index x, double sx, double sy, double sz,
                        std::vector<float>& tmp) {
        const Index plane = y * x, n = z * plane;
        tmp.resize(static_cast<std::size_t>(n));
        auto axis = [&](double sigma, Index stride, Index count, Index outer) {
            if (sigma <= 1e-6 || count < 2) return;
            const Index r = std::max<Index>(1, static_cast<Index>(std::ceil(3.0 * sigma)));
            std::vector<float> k(static_cast<std::size_t>(2 * r + 1));
            double sum = 0.0;
            for (Index i = -r; i <= r; ++i) {
                k[static_cast<std::size_t>(i + r)] = static_cast<float>(std::exp(-0.5 * (i * i) / (sigma * sigma)));
                sum += k[static_cast<std::size_t>(i + r)];
            }
            for (float& e : k) e = static_cast<float>(e / sum);
            std::copy_n(v.data(), n, tmp.data());
            for (Index o = 0; o < outer; ++o) {
                // `o` walks every line along this axis; base is its first element
                const Index base = stride == 1 ? o * count : (stride == x ? (o / x) * plane + (o % x) : o);
                for (Index c = 0; c < count; ++c) {
                    float acc = 0.0f;
                    for (Index i = -r; i <= r; ++i) {
                        Index j = c + i;
                        // mirror until inside (scipy's "mirror"), which one
                        // reflection followed by a clamp is not when the
                        // kernel is wider than the axis
                        if (count == 1) j = 0;
                        else
                            while (j < 0 || j >= count) j = j < 0 ? -j : 2 * count - j - 2;
                        acc += tmp[static_cast<std::size_t>(base + j * stride)] * k[static_cast<std::size_t>(i + r)];
                    }
                    v[static_cast<std::size_t>(base + c * stride)] = acc;
                }
            }
        };
        axis(sx, 1, x, z * y);           // rows
        axis(sy, x, y, z * x);           // columns
        axis(sz, plane, z, plane);       // planes
    }

    std::uint32_t logBlobSeeds(const float* values, const std::uint8_t* mask, Index z, Index y, Index x, double zAspect,
                               double sigmaMin, double sigmaMax, int scales, std::uint32_t* out) {
        requireExtent(z, y, x, "logBlobSeeds");
        const Index plane = y * x, n = z * plane;
        std::fill(out, out + n, 0u);
        scales = std::max(1, scales);
        sigmaMin = std::max(0.3, sigmaMin);
        sigmaMax = std::max(sigmaMin, sigmaMax);
        zAspect = std::max(1e-6, zAspect);

        std::vector<float> best(static_cast<std::size_t>(n), 0.0f), bestScale(static_cast<std::size_t>(n), 0.0f);
        std::vector<float> blur(static_cast<std::size_t>(n)), tmp;
        for (int k = 0; k < scales; ++k) {
            const double sigma = scales == 1 ? sigmaMin
                                             : sigmaMin * std::pow(sigmaMax / sigmaMin, static_cast<double>(k) / (scales - 1));
            std::copy_n(values, n, blur.data());
            gaussianVolume(blur, z, y, x, sigma, sigma, sigma / zAspect, tmp);
            const double norm = sigma * sigma;
            for (Index iz = 0; iz < z; ++iz)
                for (Index iy = 0; iy < y; ++iy)
                    for (Index ix = 0; ix < x; ++ix) {
                        const Index i = (iz * y + iy) * x + ix;
                        if (!mask[i]) continue;
                        const float c = blur[static_cast<std::size_t>(i)];
                        auto tap = [&](Index j) { return blur[static_cast<std::size_t>(j)]; };
                        const float lx = tap(ix > 0 ? i - 1 : i) + tap(ix + 1 < x ? i + 1 : i) - 2.0f * c;
                        const float ly = tap(iy > 0 ? i - x : i) + tap(iy + 1 < y ? i + x : i) - 2.0f * c;
                        const float lz = z > 1 ? tap(iz > 0 ? i - plane : i) + tap(iz + 1 < z ? i + plane : i) - 2.0f * c : 0.0f;
                        // bright blob: the Laplacian dips, so negate it
                        const float response = static_cast<float>(-norm * (lx + ly + lz));
                        if (response > best[static_cast<std::size_t>(i)]) {
                            best[static_cast<std::size_t>(i)] = response;
                            bestScale[static_cast<std::size_t>(i)] = static_cast<float>(sigma);
                        }
                    }
        }
        // peaks of the response, strongest first, each suppressing its own width
        std::vector<Index> candidates;
        for (Index iz = 0; iz < z; ++iz)
            for (Index iy = 0; iy < y; ++iy)
                for (Index ix = 0; ix < x; ++ix) {
                    const Index i = (iz * y + iy) * x + ix;
                    if (!mask[i] || !(best[static_cast<std::size_t>(i)] > 0.0f)) continue;
                    bool maximal = true;
                    for (Index dz = -1; dz <= 1 && maximal; ++dz)
                        for (Index dy = -1; dy <= 1 && maximal; ++dy)
                            for (Index dx = -1; dx <= 1; ++dx) {
                                const Index jz = iz + dz, jy = iy + dy, jx = ix + dx;
                                if (jz < 0 || jz >= z || jy < 0 || jy >= y || jx < 0 || jx >= x) continue;
                                if (best[static_cast<std::size_t>((jz * y + jy) * x + jx)] > best[static_cast<std::size_t>(i)]) {
                                    maximal = false;
                                    break;
                                }
                            }
                    if (maximal) candidates.push_back(i);
                }
        std::stable_sort(candidates.begin(), candidates.end(), [&](Index a, Index b) {
            return best[static_cast<std::size_t>(a)] > best[static_cast<std::size_t>(b)];
        });
        std::vector<std::array<double, 4>> accepted;   // z, y, x, radius^2
        std::uint32_t count = 0;
        for (Index i : candidates) {
            const double iz = static_cast<double>(i / plane), iy = static_cast<double>((i % plane) / x),
                         ix = static_cast<double>(i % x);
            // the scale that answered corresponds to an object of radius sigma * sqrt(3),
            // and that is the distance over which this peak owns the image
            const double radius = std::max(1.0, std::sqrt(3.0) * static_cast<double>(bestScale[static_cast<std::size_t>(i)]));
            bool keep = true;
            for (const auto& a : accepted) {
                const double dz = (iz - a[0]) * zAspect, dy = iy - a[1], dx = ix - a[2];
                if (dz * dz + dy * dy + dx * dx < std::max(radius * radius, a[3])) {
                    keep = false;
                    break;
                }
            }
            if (!keep) continue;
            accepted.push_back({iz, iy, ix, radius * radius});
            out[i] = 1u;   // accepted; numbered below
        }
        // Number the seeds by position, not by how strong the response was.
        // The strength decides which peaks survive, but it can be all but tied
        // between two of them, and then the ids would depend on the last bit
        // of a Gaussian -- which differs between implementations and machines.
        for (Index i = 0; i < n; ++i)
            if (out[i]) out[i] = ++count;
        return count;
    }

    std::uint32_t distanceSeeds(const std::uint8_t* mask, Index z, Index y, Index x, double minDistance,
                                std::uint32_t* out) {
        requireExtent(z, y, x, "distanceSeeds");
        const Index plane = y * x, n = z * plane;
        std::vector<float> dist(static_cast<std::size_t>(n));
        distanceTransform(mask, z, y, x, dist.data());
        std::fill(out, out + n, 0u);
        // candidates: masked voxels at least as far from the background as
        // every 26-neighbour
        std::vector<Index> candidates;
        for (Index iz = 0; iz < z; ++iz)
            for (Index iy = 0; iy < y; ++iy)
                for (Index ix = 0; ix < x; ++ix) {
                    const Index i = (iz * y + iy) * x + ix;
                    const float d = dist[static_cast<std::size_t>(i)];
                    if (!mask[i] || !(d > 0.0f)) continue;
                    bool maximal = true;
                    for (Index dz = -1; dz <= 1 && maximal; ++dz)
                        for (Index dy = -1; dy <= 1 && maximal; ++dy)
                            for (Index dx = -1; dx <= 1; ++dx) {
                                const Index jz = iz + dz, jy = iy + dy, jx = ix + dx;
                                if (jz < 0 || jz >= z || jy < 0 || jy >= y || jx < 0 || jx >= x) continue;
                                if (dist[static_cast<std::size_t>((jz * y + jy) * x + jx)] > d) {
                                    maximal = false;
                                    break;
                                }
                            }
                    if (maximal) candidates.push_back(i);
                }
        // deepest first; a candidate closer than minDistance to an accepted
        // seed is the same object's plateau, not a new one
        std::stable_sort(candidates.begin(), candidates.end(),
                         [&](Index a, Index b) { return dist[static_cast<std::size_t>(a)] > dist[static_cast<std::size_t>(b)]; });
        const double minD2 = std::max(minDistance, 1.0) * std::max(minDistance, 1.0);
        std::vector<Voxel> accepted;
        for (Index i : candidates) {
            const Voxel c{i / plane, (i / x) % y, i % x};
            bool ok = true;
            for (const Voxel& a : accepted) {
                const double dz = static_cast<double>(a.z - c.z), dy = static_cast<double>(a.y - c.y), dx = static_cast<double>(a.x - c.x);
                if (dz * dz + dy * dy + dx * dx < minD2) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            accepted.push_back(c);
            out[i] = static_cast<std::uint32_t>(accepted.size());
        }
        return static_cast<std::uint32_t>(accepted.size());
    }

    namespace {
        // Voxel counts per id. Dense ids get a table indexed by id; ids far
        // beyond the voxel count (a plugin's, a corrupt file's) get a map,
        // so a single id near 2^32 does not size a 16 GB table.
        struct IdCounts {
            std::vector<Index> table;
            std::unordered_map<std::uint32_t, Index> sparse;
            bool dense = true;
            Index of(std::uint32_t id) const {
                if (dense) return id < table.size() ? table[id] : 0;
                const auto it = sparse.find(id);
                return it == sparse.end() ? 0 : it->second;
            }
            template <typename F>
            void forEachId(F&& f) const {   // ascending id order, ids present only
                if (dense) {
                    for (std::uint32_t id = 1; id < table.size(); ++id)
                        if (table[id] > 0) f(id, table[id]);
                } else {
                    std::vector<std::uint32_t> ids;
                    ids.reserve(sparse.size());
                    for (const auto& [id, c] : sparse)
                        if (id) ids.push_back(id);
                    std::sort(ids.begin(), ids.end());
                    for (std::uint32_t id : ids) f(id, sparse.at(id));
                }
            }
        };

        IdCounts countIds(const std::uint32_t* labels, Index n) {
            IdCounts c;
            std::uint32_t maxId = 0;
            for (Index i = 0; i < n; ++i) maxId = std::max(maxId, labels[i]);
            c.dense = static_cast<Index>(maxId) <= 4 * n + 1024;
            if (c.dense) {
                c.table.assign(static_cast<std::size_t>(maxId) + 1, 0);
                for (Index i = 0; i < n; ++i) ++c.table[labels[i]];
            } else {
                for (Index i = 0; i < n; ++i) ++c.sparse[labels[i]];
            }
            return c;
        }
    } // namespace

    std::uint32_t removeSmall(std::uint32_t* labels, Index n, Index minVoxels) {
        const IdCounts counts = countIds(labels, n);
        std::unordered_map<std::uint32_t, std::uint32_t> remap;
        std::uint32_t next = 0;
        counts.forEachId([&](std::uint32_t id, Index c) {
            if (c >= minVoxels) remap[id] = ++next;
        });
        for (Index i = 0; i < n; ++i) {
            if (!labels[i]) continue;
            const auto it = remap.find(labels[i]);
            labels[i] = it == remap.end() ? 0u : it->second;
        }
        return next;
    }

    std::uint32_t dropSmall(std::uint32_t* labels, Index n, Index minVoxels) {
        const IdCounts counts = countIds(labels, n);
        std::uint32_t kept = 0;
        counts.forEachId([&](std::uint32_t, Index c) {
            if (c >= minVoxels) ++kept;
        });
        for (Index i = 0; i < n; ++i)
            if (labels[i] && counts.of(labels[i]) < minVoxels) labels[i] = 0;
        return kept;
    }

    // --- filters and thresholds -------------------------------------------

    void medianFilterPlane(float* plane, Index y, Index x, std::vector<float>& tmp) {
        if (y < 3 || x < 3) return;
        tmp.assign(plane, plane + static_cast<std::size_t>(y * x));
        std::array<float, 9> win{};
        for (Index r = 0; r < y; ++r)
            for (Index c = 0; c < x; ++c) {
                int k = 0;
                for (Index dr = -1; dr <= 1; ++dr)
                    for (Index dc = -1; dc <= 1; ++dc) {
                        const Index rr = std::clamp<Index>(r + dr, 0, y - 1), cc = std::clamp<Index>(c + dc, 0, x - 1);
                        win[static_cast<std::size_t>(k++)] = tmp[static_cast<std::size_t>(rr * x + cc)];
                    }
                std::nth_element(win.begin(), win.begin() + 4, win.end());
                plane[r * x + c] = win[4];
            }
    }

    void anisotropicDiffusionPlane(float* plane, Index y, Index x, int iterations, double k, std::vector<float>& tmp) {
        if (iterations <= 0 || y < 3 || x < 3) return;
        const Index n = y * x;
        float lo = std::numeric_limits<float>::max(), hi = std::numeric_limits<float>::lowest();
        for (Index i = 0; i < n; ++i) {
            lo = std::min(lo, plane[i]);
            hi = std::max(hi, plane[i]);
        }
        // k is a fraction of the range, so the same setting works whatever the
        // units are; a flat plane has nothing to diffuse
        const double kk = std::max(1e-12, k) * std::max(1e-12, static_cast<double>(hi - lo));
        const double inv = 1.0 / (kk * kk);
        constexpr double kLambda = 0.25;   // stability limit for four neighbours
        for (int it = 0; it < iterations; ++it) {
            tmp.assign(plane, plane + static_cast<std::size_t>(n));
            for (Index r = 0; r < y; ++r)
                for (Index c = 0; c < x; ++c) {
                    const float centre = tmp[static_cast<std::size_t>(r * x + c)];
                    double sum = 0.0;
                    const Index rr[4] = {r > 0 ? r - 1 : 0, r + 1 < y ? r + 1 : y - 1, r, r};
                    const Index cc[4] = {c, c, c > 0 ? c - 1 : 0, c + 1 < x ? c + 1 : x - 1};
                    for (int nb = 0; nb < 4; ++nb) {
                        const double g = static_cast<double>(tmp[static_cast<std::size_t>(rr[nb] * x + cc[nb])]) - centre;
                        sum += g * std::exp(-g * g * inv);   // Perona-Malik conductance
                    }
                    plane[r * x + c] = static_cast<float>(centre + kLambda * sum);
                }
        }
    }

    namespace {
        // 256 bin histogram over the data range; the thresholds below all work
        // on it, and all return a value in the data's own units.
        struct Histogram {
            std::array<Index, 256> bins{};
            float lo = 0.0f, hi = 0.0f;
            bool degenerate = true;

            float valueOf(std::size_t bin) const {
                return lo + static_cast<float>((static_cast<double>(bin) + 0.5) / 256.0 * (static_cast<double>(hi) - lo));
            }
        };

        Histogram histogramOf(const float* values, Index n) {
            Histogram h;
            if (n <= 0) return h;
            h.lo = std::numeric_limits<float>::max();
            h.hi = std::numeric_limits<float>::lowest();
            for (Index i = 0; i < n; ++i) {
                if (!std::isfinite(values[i])) continue;
                h.lo = std::min(h.lo, values[i]);
                h.hi = std::max(h.hi, values[i]);
            }
            if (!(h.hi > h.lo)) return h;
            h.degenerate = false;
            // 256 equal bins over [lo, hi] (hi lands in the last one), so a
            // bin's centre really is (b + 0.5) / 256 of the range, as
            // valueOf() reports it
            const double scale = 256.0 / (static_cast<double>(h.hi) - h.lo);
            for (Index i = 0; i < n; ++i) {
                if (!std::isfinite(values[i])) continue;
                const int b = static_cast<int>((static_cast<double>(values[i]) - h.lo) * scale);
                ++h.bins[static_cast<std::size_t>(std::clamp(b, 0, 255))];
            }
            return h;
        }
    } // namespace

    float triangleThreshold(const float* values, Index n) {
        const Histogram h = histogramOf(values, n);
        if (h.degenerate) return h.lo;
        // peak, and the far end of the histogram: the last non-empty bin
        std::size_t peak = 0;
        Index peakCount = 0;
        for (std::size_t b = 0; b < h.bins.size(); ++b)
            if (h.bins[b] > peakCount) {
                peakCount = h.bins[b];
                peak = b;
            }
        std::size_t first = 0, last = h.bins.size() - 1;
        while (first < h.bins.size() && h.bins[first] == 0) ++first;
        while (last > 0 && h.bins[last] == 0) --last;
        // the long tail is the side of the peak with more room; the method
        // measures the drop from the peak to the end of that tail
        const bool tailRight = last - peak >= peak - first;
        const std::size_t end = tailRight ? last : first;
        if (end == peak) return h.valueOf(peak);
        const double dx = static_cast<double>(end) - static_cast<double>(peak);
        const double dy = -static_cast<double>(peakCount);
        const double norm = std::sqrt(dx * dx + dy * dy);
        double best = -1.0;
        std::size_t bestBin = peak;
        const std::size_t from = std::min(peak, end), to = std::max(peak, end);
        for (std::size_t b = from; b <= to; ++b) {
            // distance from the bin to the line peak -> end
            const double px = static_cast<double>(b) - static_cast<double>(peak);
            const double py = static_cast<double>(h.bins[b]) - static_cast<double>(peakCount);
            const double d = std::fabs(px * dy - py * dx) / norm;
            if (d > best) {
                best = d;
                bestBin = b;
            }
        }
        return h.valueOf(bestBin);
    }

    float liThreshold(const float* values, Index n) {
        const Histogram h = histogramOf(values, n);
        if (h.degenerate) return h.lo;
        // Li & Tam's fixed point: the cut is the point where the two class
        // means agree with the cross entropy, found by iterating on it
        double total = 0.0, weighted = 0.0;
        for (std::size_t b = 0; b < h.bins.size(); ++b) {
            total += static_cast<double>(h.bins[b]);
            weighted += static_cast<double>(b) * static_cast<double>(h.bins[b]);
        }
        if (total <= 0.0) return h.lo;
        double t = weighted / total;   // start at the mean
        for (int it = 0; it < 100; ++it) {
            double sumLo = 0.0, countLo = 0.0, sumHi = 0.0, countHi = 0.0;
            for (std::size_t b = 0; b < h.bins.size(); ++b) {
                const double c = static_cast<double>(h.bins[b]);
                if (static_cast<double>(b) <= t) {
                    sumLo += static_cast<double>(b) * c;
                    countLo += c;
                } else {
                    sumHi += static_cast<double>(b) * c;
                    countHi += c;
                }
            }
            const double meanLo = countLo > 0.0 ? sumLo / countLo : 0.0;
            const double meanHi = countHi > 0.0 ? sumHi / countHi : 0.0;
            // both means have to be positive for the logarithms to exist; the
            // histogram starts at bin 0, so shift by one
            const double a = meanLo + 1.0, b2 = meanHi + 1.0;
            const double next = (b2 - a) / (std::log(b2) - std::log(a));
            if (!std::isfinite(next)) break;
            if (std::fabs(next - t) < 0.5) {
                t = next;
                break;
            }
            t = next;
        }
        return h.valueOf(static_cast<std::size_t>(std::clamp(t, 0.0, 255.0)));
    }

    float yenThreshold(const float* values, Index n) {
        const Histogram h = histogramOf(values, n);
        if (h.degenerate) return h.lo;
        double total = 0.0;
        for (Index c : h.bins) total += static_cast<double>(c);
        if (total <= 0.0) return h.lo;
        // Yen's criterion on the probability mass function: the cut where the
        // two classes are most uniform relative to how much of the image each
        // holds. P1 is the cumulative mass, P1sq / P2sq the cumulative squared
        // mass from each end.
        std::array<double, 256> p1{}, p1sq{}, p2sq{};
        double c1 = 0.0, c1sq = 0.0;
        for (std::size_t b = 0; b < h.bins.size(); ++b) {
            const double pm = static_cast<double>(h.bins[b]) / total;
            c1 += pm;
            c1sq += pm * pm;
            p1[b] = c1;
            p1sq[b] = c1sq;
        }
        double c2sq = 0.0;
        for (std::size_t k = h.bins.size(); k-- > 0;) {
            const double pm = static_cast<double>(h.bins[k]) / total;
            c2sq += pm * pm;
            p2sq[k] = c2sq;
        }
        double best = -std::numeric_limits<double>::infinity();
        std::size_t bestBin = 0;
        for (std::size_t b = 0; b + 1 < h.bins.size(); ++b) {
            const double denom = p1sq[b] * p2sq[b + 1];
            const double mass = p1[b] * (1.0 - p1[b]);
            if (denom <= 0.0 || mass <= 0.0) continue;
            const double crit = std::log(mass * mass / denom);
            if (crit > best) {
                best = crit;
                bestBin = b;
            }
        }
        return h.valueOf(bestBin);
    }

    float isodataThreshold(const float* values, Index n) {
        const Histogram h = histogramOf(values, n);
        if (h.degenerate) return h.lo;
        double total = 0.0, weighted = 0.0;
        for (std::size_t b = 0; b < h.bins.size(); ++b) {
            total += static_cast<double>(h.bins[b]);
            weighted += static_cast<double>(b) * static_cast<double>(h.bins[b]);
        }
        if (total <= 0.0) return h.lo;
        double t = weighted / total;
        for (int it = 0; it < 100; ++it) {
            double sumLo = 0.0, countLo = 0.0, sumHi = 0.0, countHi = 0.0;
            for (std::size_t b = 0; b < h.bins.size(); ++b) {
                const double c = static_cast<double>(h.bins[b]);
                if (static_cast<double>(b) <= t) {
                    sumLo += static_cast<double>(b) * c;
                    countLo += c;
                } else {
                    sumHi += static_cast<double>(b) * c;
                    countHi += c;
                }
            }
            // one side empty: the cut has walked off the histogram, keep the last
            if (countLo <= 0.0 || countHi <= 0.0) break;
            const double next = 0.5 * (sumLo / countLo + sumHi / countHi);
            if (std::fabs(next - t) < 0.5) {
                t = next;
                break;
            }
            t = next;
        }
        return h.valueOf(static_cast<std::size_t>(std::clamp(t, 0.0, 255.0)));
    }

    namespace {
        // The half-widths and heights of the ball, and how far the image is
        // shrunk before it is rolled: shared by the plane below and by nothing
        // else, but kept together so the two agree.
        struct Ball {
            int shrink = 1;
            int halfWidth = 1;
            std::vector<double> height;   // (2 hw + 1)^2, negative where the ball does not reach

            double at(int dy, int dx) const {
                return height[static_cast<std::size_t>((dy + halfWidth) * (2 * halfWidth + 1) + (dx + halfWidth))];
            }
        };

        Ball ballFor(double radius) {
            Ball b;
            b.shrink = std::clamp(static_cast<int>(std::lround(radius / 10.0)), 1, 8);
            const double scaled = std::max(1.0, radius / b.shrink);
            b.halfWidth = std::max(1, static_cast<int>(std::floor(scaled)));
            const int side = 2 * b.halfWidth + 1;
            b.height.assign(static_cast<std::size_t>(side) * side, -1.0);
            for (int dy = -b.halfWidth; dy <= b.halfWidth; ++dy)
                for (int dx = -b.halfWidth; dx <= b.halfWidth; ++dx) {
                    const double r2 = scaled * scaled - static_cast<double>(dy) * dy - static_cast<double>(dx) * dx;
                    if (r2 >= 0.0) b.height[static_cast<std::size_t>((dy + b.halfWidth) * side + (dx + b.halfWidth))] = std::sqrt(r2);
                }
            return b;
        }
    } // namespace

    void rollingBallPlane(float* plane, Index y, Index x, double radius, std::vector<float>& scratch) {
        if (radius <= 0.0 || y <= 0 || x <= 0) return;
        const Ball ball = ballFor(radius);
        const Index s = ball.shrink;
        const Index sy = (y + s - 1) / s, sx = (x + s - 1) / s;

        // shrink by taking the darkest pixel of each block: the ball rolls
        // under the surface, so it is the low side that decides where it fits
        std::vector<double> small(static_cast<std::size_t>(sy * sx), 0.0);
        for (Index r = 0; r < sy; ++r)
            for (Index c = 0; c < sx; ++c) {
                double lowest = std::numeric_limits<double>::max();
                for (Index dr = 0; dr < s; ++dr)
                    for (Index dc = 0; dc < s; ++dc) {
                        const Index yy = std::min(r * s + dr, y - 1), xx = std::min(c * s + dc, x - 1);
                        lowest = std::min(lowest, static_cast<double>(plane[yy * x + xx]));
                    }
                small[static_cast<std::size_t>(r * sx + c)] = lowest;
            }

        // where the ball's centre can sit (an erosion by the ball), then the
        // surface its top traces from there (a dilation): a grey opening
        std::vector<double> centre(static_cast<std::size_t>(sy * sx), 0.0);
        for (Index r = 0; r < sy; ++r)
            for (Index c = 0; c < sx; ++c) {
                double highest = std::numeric_limits<double>::max();
                for (int dy = -ball.halfWidth; dy <= ball.halfWidth; ++dy)
                    for (int dx = -ball.halfWidth; dx <= ball.halfWidth; ++dx) {
                        const double h = ball.at(dy, dx);
                        if (h < 0.0) continue;
                        const Index yy = std::clamp<Index>(r + dy, 0, sy - 1), xx = std::clamp<Index>(c + dx, 0, sx - 1);
                        highest = std::min(highest, small[static_cast<std::size_t>(yy * sx + xx)] - h);
                    }
                centre[static_cast<std::size_t>(r * sx + c)] = highest;
            }
        std::vector<double> background(static_cast<std::size_t>(sy * sx), 0.0);
        for (Index r = 0; r < sy; ++r)
            for (Index c = 0; c < sx; ++c) {
                double top = std::numeric_limits<double>::lowest();
                for (int dy = -ball.halfWidth; dy <= ball.halfWidth; ++dy)
                    for (int dx = -ball.halfWidth; dx <= ball.halfWidth; ++dx) {
                        const double h = ball.at(dy, dx);
                        if (h < 0.0) continue;
                        const Index yy = std::clamp<Index>(r - dy, 0, sy - 1), xx = std::clamp<Index>(c - dx, 0, sx - 1);
                        top = std::max(top, centre[static_cast<std::size_t>(yy * sx + xx)] + h);
                    }
                background[static_cast<std::size_t>(r * sx + c)] = top;
            }

        // back to full size: the shrunk sample at (r, c) stands for the centre
        // of its block, and the background between them is bilinear
        scratch.assign(static_cast<std::size_t>(y * x), 0.0f);
        const double half = 0.5 * (static_cast<double>(s) - 1.0);
        for (Index r = 0; r < y; ++r) {
            const double fy = s == 1 ? static_cast<double>(r) : (static_cast<double>(r) - half) / static_cast<double>(s);
            const Index r0 = std::clamp<Index>(static_cast<Index>(std::floor(fy)), 0, sy - 1);
            const Index r1 = std::clamp<Index>(r0 + 1, 0, sy - 1);
            const double wy = std::clamp(fy - static_cast<double>(r0), 0.0, 1.0);
            for (Index c = 0; c < x; ++c) {
                const double fx = s == 1 ? static_cast<double>(c) : (static_cast<double>(c) - half) / static_cast<double>(s);
                const Index c0 = std::clamp<Index>(static_cast<Index>(std::floor(fx)), 0, sx - 1);
                const Index c1 = std::clamp<Index>(c0 + 1, 0, sx - 1);
                const double wx = std::clamp(fx - static_cast<double>(c0), 0.0, 1.0);
                const double a = background[static_cast<std::size_t>(r0 * sx + c0)];
                const double b = background[static_cast<std::size_t>(r0 * sx + c1)];
                const double cc = background[static_cast<std::size_t>(r1 * sx + c0)];
                const double d = background[static_cast<std::size_t>(r1 * sx + c1)];
                const double value = (a * (1.0 - wx) + b * wx) * (1.0 - wy) + (cc * (1.0 - wx) + d * wx) * wy;
                scratch[static_cast<std::size_t>(r * x + c)] = static_cast<float>(value);
            }
        }
        for (Index i = 0; i < y * x; ++i) plane[i] = std::max(0.0f, plane[i] - scratch[static_cast<std::size_t>(i)]);
    }

    Index fillHoles3D(std::uint8_t* mask, Index z, Index y, Index x, Index maxVoxels) {
        const Index n = z * y * x;
        if (n <= 0) return 0;
        // flood the background inwards from the border; what it never reaches
        // is enclosed
        std::vector<std::uint8_t> outside(static_cast<std::size_t>(n), 0);
        std::vector<Index> stack;
        auto push = [&](Index i) {
            if (mask[i] == 0 && !outside[static_cast<std::size_t>(i)]) {
                outside[static_cast<std::size_t>(i)] = 1;
                stack.push_back(i);
            }
        };
        for (Index k = 0; k < z; ++k)
            for (Index r = 0; r < y; ++r)
                for (Index c = 0; c < x; ++c)
                    if (k == 0 || k == z - 1 || r == 0 || r == y - 1 || c == 0 || c == x - 1) push((k * y + r) * x + c);
        while (!stack.empty()) {
            const Index cur = stack.back();
            stack.pop_back();
            const Index cz = cur / (y * x), rest = cur % (y * x), cy = rest / x, cx = rest % x;
            const Index nz[6] = {cz - 1, cz + 1, cz, cz, cz, cz};
            const Index ny[6] = {cy, cy, cy - 1, cy + 1, cy, cy};
            const Index nx[6] = {cx, cx, cx, cx, cx - 1, cx + 1};
            for (int q = 0; q < 6; ++q) {
                if (nz[q] < 0 || nz[q] >= z || ny[q] < 0 || ny[q] >= y || nx[q] < 0 || nx[q] >= x) continue;
                push((nz[q] * y + ny[q]) * x + nx[q]);
            }
        }
        // the enclosed background, one cavity at a time, so a size limit can
        // keep a large lumen open while small holes are closed
        std::vector<std::uint8_t> seen(static_cast<std::size_t>(n), 0);
        std::vector<Index> cavity;
        Index filled = 0;
        for (Index i = 0; i < n; ++i) {
            if (mask[i] != 0 || outside[static_cast<std::size_t>(i)] || seen[static_cast<std::size_t>(i)]) continue;
            cavity.clear();
            seen[static_cast<std::size_t>(i)] = 1;
            stack.push_back(i);
            while (!stack.empty()) {
                const Index cur = stack.back();
                stack.pop_back();
                cavity.push_back(cur);
                const Index cz = cur / (y * x), rest = cur % (y * x), cy = rest / x, cx = rest % x;
                const Index nz[6] = {cz - 1, cz + 1, cz, cz, cz, cz};
                const Index ny[6] = {cy, cy, cy - 1, cy + 1, cy, cy};
                const Index nx[6] = {cx, cx, cx, cx, cx - 1, cx + 1};
                for (int q = 0; q < 6; ++q) {
                    if (nz[q] < 0 || nz[q] >= z || ny[q] < 0 || ny[q] >= y || nx[q] < 0 || nx[q] >= x) continue;
                    const Index j = (nz[q] * y + ny[q]) * x + nx[q];
                    if (mask[j] == 0 && !outside[static_cast<std::size_t>(j)] && !seen[static_cast<std::size_t>(j)]) {
                        seen[static_cast<std::size_t>(j)] = 1;
                        stack.push_back(j);
                    }
                }
            }
            if (maxVoxels > 0 && static_cast<Index>(cavity.size()) > maxVoxels) continue;
            for (Index j : cavity) mask[j] = 1;
            filled += static_cast<Index>(cavity.size());
        }
        return filled;
    }

    namespace {
        // The 26 neighbourhood of a voxel, packed into a 3x3x3 block with the
        // centre at (1, 1, 1), and the connectivity tests the thinning needs.
        using Block = std::array<std::uint8_t, 27>;
        constexpr int blockIndex(int dz, int dy, int dx) { return ((dz + 1) * 3 + (dy + 1)) * 3 + (dx + 1); }

        // One 26-connected component of the object in the neighbourhood?
        bool objectStaysConnected(const Block& b) {
            Block seen{};
            int start = -1;
            int total = 0;
            for (int i = 0; i < 27; ++i)
                if (i != blockIndex(0, 0, 0) && b[static_cast<std::size_t>(i)]) {
                    ++total;
                    if (start < 0) start = i;
                }
            if (total == 0) return false;   // an isolated voxel is not simple: deleting it removes a component
            std::vector<int> stack{start};
            seen[static_cast<std::size_t>(start)] = 1;
            int reached = 1;
            while (!stack.empty()) {
                const int cur = stack.back();
                stack.pop_back();
                const int cz = cur / 9, cy = (cur / 3) % 3, cx = cur % 3;
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int nz = cz + dz, ny = cy + dy, nx = cx + dx;
                            if (nz < 0 || nz > 2 || ny < 0 || ny > 2 || nx < 0 || nx > 2) continue;
                            const int j = (nz * 3 + ny) * 3 + nx;
                            if (j == blockIndex(0, 0, 0) || !b[static_cast<std::size_t>(j)] || seen[static_cast<std::size_t>(j)]) continue;
                            seen[static_cast<std::size_t>(j)] = 1;
                            ++reached;
                            stack.push_back(j);
                        }
            }
            return reached == total;
        }

        // One 6-connected background component in the 18 neighbourhood that
        // touches the centre? (The eight corners are left out: they cannot
        // reach the centre 6-connectedly.)
        bool backgroundStaysConnected(const Block& b) {
            auto inEighteen = [](int i) {
                const int dz = i / 9 - 1, dy = (i / 3) % 3 - 1, dx = i % 3 - 1;
                return std::abs(dz) + std::abs(dy) + std::abs(dx) <= 2;
            };
            // the six face neighbours are the only way out of the centre
            std::vector<int> starts;
            for (const int face : {blockIndex(-1, 0, 0), blockIndex(1, 0, 0), blockIndex(0, -1, 0), blockIndex(0, 1, 0),
                                   blockIndex(0, 0, -1), blockIndex(0, 0, 1)})
                if (!b[static_cast<std::size_t>(face)]) starts.push_back(face);
            if (starts.empty()) return false;   // the centre is buried: deleting it opens a cavity
            Block seen{};
            std::vector<int> stack{starts.front()};
            seen[static_cast<std::size_t>(starts.front())] = 1;
            while (!stack.empty()) {
                const int cur = stack.back();
                stack.pop_back();
                const int cz = cur / 9, cy = (cur / 3) % 3, cx = cur % 3;
                const int nz[6] = {cz - 1, cz + 1, cz, cz, cz, cz};
                const int ny[6] = {cy, cy, cy - 1, cy + 1, cy, cy};
                const int nx[6] = {cx, cx, cx, cx, cx - 1, cx + 1};
                for (int q = 0; q < 6; ++q) {
                    if (nz[q] < 0 || nz[q] > 2 || ny[q] < 0 || ny[q] > 2 || nx[q] < 0 || nx[q] > 2) continue;
                    const int j = (nz[q] * 3 + ny[q]) * 3 + nx[q];
                    if (j == blockIndex(0, 0, 0) || !inEighteen(j) || b[static_cast<std::size_t>(j)] || seen[static_cast<std::size_t>(j)]) continue;
                    seen[static_cast<std::size_t>(j)] = 1;
                    stack.push_back(j);
                }
            }
            for (const int face : starts)
                if (!seen[static_cast<std::size_t>(face)]) return false;   // two ways out that do not meet
            return true;
        }
    } // namespace

    Index skeletonize3D(std::uint8_t* mask, Index z, Index y, Index x, const std::function<void()>& poll) {
        const Index n = z * y * x;
        if (n <= 0) return 0;
        // the six directions, taken in turn: deleting a whole border at once
        // would eat a thin structure from both sides
        const int dirZ[6] = {-1, 1, 0, 0, 0, 0};
        const int dirY[6] = {0, 0, -1, 1, 0, 0};
        const int dirX[6] = {0, 0, 0, 0, -1, 1};
        std::vector<Index> candidates;
        bool changed = true;
        for (int pass = 0; pass < 1000 && changed; ++pass) {
            changed = false;
            for (int dir = 0; dir < 6; ++dir) {
                if (poll) poll();
                candidates.clear();
                for (Index k = 0; k < z; ++k)
                    for (Index r = 0; r < y; ++r)
                        for (Index c = 0; c < x; ++c) {
                            const Index i = (k * y + r) * x + c;
                            if (!mask[i]) continue;
                            // on this border: the neighbour in the direction is background
                            const Index nz = k + dirZ[dir], ny = r + dirY[dir], nx = c + dirX[dir];
                            const bool open = nz < 0 || nz >= z || ny < 0 || ny >= y || nx < 0 || nx >= x ||
                                              !mask[(nz * y + ny) * x + nx];
                            if (!open) continue;
                            Block b{};
                            int neighbours = 0;
                            for (int dz = -1; dz <= 1; ++dz)
                                for (int dy = -1; dy <= 1; ++dy)
                                    for (int dx = -1; dx <= 1; ++dx) {
                                        const Index jz = k + dz, jy = r + dy, jx = c + dx;
                                        const bool on = jz >= 0 && jz < z && jy >= 0 && jy < y && jx >= 0 && jx < x &&
                                                        mask[(jz * y + jy) * x + jx] != 0;
                                        b[static_cast<std::size_t>(blockIndex(dz, dy, dx))] = on ? 1 : 0;
                                        if (on && !(dz == 0 && dy == 0 && dx == 0)) ++neighbours;
                                    }
                            if (neighbours <= 1) continue;   // an end point: the line stops here
                            if (!objectStaysConnected(b) || !backgroundStaysConnected(b)) continue;
                            candidates.push_back(i);
                        }
                // deleted one at a time, rechecking: two voxels that are each
                // simple on their own need not both be
                for (Index i : candidates) {
                    const Index k = i / (y * x), rest = i % (y * x), r = rest / x, c = rest % x;
                    Block b{};
                    int neighbours = 0;
                    for (int dz = -1; dz <= 1; ++dz)
                        for (int dy = -1; dy <= 1; ++dy)
                            for (int dx = -1; dx <= 1; ++dx) {
                                const Index jz = k + dz, jy = r + dy, jx = c + dx;
                                const bool on = jz >= 0 && jz < z && jy >= 0 && jy < y && jx >= 0 && jx < x &&
                                                mask[(jz * y + jy) * x + jx] != 0;
                                b[static_cast<std::size_t>(blockIndex(dz, dy, dx))] = on ? 1 : 0;
                                if (on && !(dz == 0 && dy == 0 && dx == 0)) ++neighbours;
                            }
                    if (neighbours <= 1) continue;
                    if (!objectStaysConnected(b) || !backgroundStaysConnected(b)) continue;
                    mask[i] = 0;
                    changed = true;
                }
            }
        }
        Index remaining = 0;
        for (Index i = 0; i < n; ++i) remaining += mask[i] ? 1 : 0;
        return remaining;
    }

    Index expandLabels(std::uint32_t* labels, Index z, Index y, Index x, double distance, double zAspect) {
        const Index n = z * y * x;
        if (n <= 0 || distance <= 0.0) return 0;
        zAspect = std::max(1e-6, zAspect);
        // Dijkstra from every labelled voxel at once. A voxel reached from two
        // labels at the same distance stays background: growing it either way
        // would join two objects that the segmentation kept apart.
        struct Node {
            double d;
            Index i;
            bool operator>(const Node& o) const { return d > o.d; }
        };
        std::priority_queue<Node, std::vector<Node>, std::greater<Node>> queue;
        std::vector<double> best(static_cast<std::size_t>(n), std::numeric_limits<double>::infinity());
        std::vector<std::uint32_t> from(static_cast<std::size_t>(n), 0u);
        std::vector<std::uint8_t> tied(static_cast<std::size_t>(n), 0);
        for (Index i = 0; i < n; ++i)
            if (labels[i] != 0) {
                best[static_cast<std::size_t>(i)] = 0.0;
                from[static_cast<std::size_t>(i)] = labels[i];
                queue.push({0.0, i});
            }
        const double stepZ = zAspect;   // the planes are that much further apart than the pixels
        while (!queue.empty()) {
            const Node cur = queue.top();
            queue.pop();
            if (cur.d > best[static_cast<std::size_t>(cur.i)]) continue;
            if (cur.d >= distance) continue;
            const Index cz = cur.i / (y * x), rest = cur.i % (y * x), cy = rest / x, cx = rest % x;
            const Index nz[6] = {cz - 1, cz + 1, cz, cz, cz, cz};
            const Index ny[6] = {cy, cy, cy - 1, cy + 1, cy, cy};
            const Index nx[6] = {cx, cx, cx, cx, cx - 1, cx + 1};
            const double step[6] = {stepZ, stepZ, 1.0, 1.0, 1.0, 1.0};
            for (int q = 0; q < 6; ++q) {
                if (nz[q] < 0 || nz[q] >= z || ny[q] < 0 || ny[q] >= y || nx[q] < 0 || nx[q] >= x) continue;
                const Index j = (nz[q] * y + ny[q]) * x + nx[q];
                if (labels[j] != 0) continue;   // never overwrite a segmented voxel
                const double d = cur.d + step[q];
                if (d > distance) continue;
                double& bj = best[static_cast<std::size_t>(j)];
                if (d < bj - 1e-9) {
                    bj = d;
                    from[static_cast<std::size_t>(j)] = from[static_cast<std::size_t>(cur.i)];
                    tied[static_cast<std::size_t>(j)] = 0;
                    queue.push({d, j});
                } else if (std::fabs(d - bj) <= 1e-9 && from[static_cast<std::size_t>(cur.i)] != from[static_cast<std::size_t>(j)]) {
                    tied[static_cast<std::size_t>(j)] = 1;
                }
            }
        }
        Index claimed = 0;
        for (Index i = 0; i < n; ++i)
            if (labels[i] == 0 && from[static_cast<std::size_t>(i)] != 0 && !tied[static_cast<std::size_t>(i)]) {
                labels[i] = from[static_cast<std::size_t>(i)];
                ++claimed;
            }
        return claimed;
    }

    Index hysteresisMask(const std::uint8_t* high, const std::uint8_t* low, Index z, Index y, Index x, std::uint8_t* out) {
        const Index n = z * y * x;
        std::vector<std::uint8_t> keep(static_cast<std::size_t>(n), 0);
        std::vector<Index> stack;
        for (Index i = 0; i < n; ++i)
            if (high[i] && low[i] && !keep[static_cast<std::size_t>(i)]) {
                keep[static_cast<std::size_t>(i)] = 1;
                stack.push_back(i);
                while (!stack.empty()) {
                    const Index cur = stack.back();
                    stack.pop_back();
                    const Index cz = cur / (y * x), rest = cur % (y * x), cy = rest / x, cx = rest % x;
                    const Index nz[6] = {cz - 1, cz + 1, cz, cz, cz, cz};
                    const Index ny[6] = {cy, cy, cy - 1, cy + 1, cy, cy};
                    const Index nx[6] = {cx, cx, cx, cx, cx - 1, cx + 1};
                    for (int k = 0; k < 6; ++k) {
                        if (nz[k] < 0 || nz[k] >= z || ny[k] < 0 || ny[k] >= y || nx[k] < 0 || nx[k] >= x) continue;
                        const Index j = (nz[k] * y + ny[k]) * x + nx[k];
                        if (low[j] && !keep[static_cast<std::size_t>(j)]) {
                            keep[static_cast<std::size_t>(j)] = 1;
                            stack.push_back(j);
                        }
                    }
                }
            }
        Index kept = 0;
        for (Index i = 0; i < n; ++i) {
            out[i] = keep[static_cast<std::size_t>(i)];
            kept += out[i];
        }
        return kept;
    }

    void gradientMagnitude(const float* values, Index z, Index y, Index x, double zAspect, float* out) {
        const double invZ = zAspect > 0.0 ? 1.0 / zAspect : 1.0;
        for (Index k = 0; k < z; ++k)
            for (Index r = 0; r < y; ++r)
                for (Index c = 0; c < x; ++c) {
                    const Index i = (k * y + r) * x + c;
                    const Index kp = std::min(k + 1, z - 1), km = std::max<Index>(k - 1, 0);
                    const Index rp = std::min(r + 1, y - 1), rm = std::max<Index>(r - 1, 0);
                    const Index cp = std::min(c + 1, x - 1), cm = std::max<Index>(c - 1, 0);
                    const double gx = 0.5 * (static_cast<double>(values[(k * y + r) * x + cp]) - values[(k * y + r) * x + cm]);
                    const double gy = 0.5 * (static_cast<double>(values[(k * y + rp) * x + c]) - values[(k * y + rm) * x + c]);
                    const double gz = 0.5 * (static_cast<double>(values[(kp * y + r) * x + c]) - values[(km * y + r) * x + c]) * invZ;
                    out[i] = static_cast<float>(std::sqrt(gx * gx + gy * gy + gz * gz));
                }
    }

    namespace {
        // The four line elements of the morphological curvature operator: the
        // horizontal, the vertical and the two diagonals.
        constexpr int kLineDy[4][3] = {{0, 0, 0}, {-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
        constexpr int kLineDx[4][3] = {{-1, 0, 1}, {0, 0, 0}, {-1, 0, 1}, {1, 0, -1}};

        std::uint8_t lineValue(const std::uint8_t* m, Index y, Index x, Index r, Index c, int line, bool maximum) {
            std::uint8_t acc = maximum ? 0 : 1;
            for (int k = 0; k < 3; ++k) {
                const Index rr = std::clamp<Index>(r + kLineDy[line][k], 0, y - 1);
                const Index cc = std::clamp<Index>(c + kLineDx[line][k], 0, x - 1);
                const std::uint8_t v = m[rr * x + cc];
                acc = maximum ? std::max(acc, v) : std::min(acc, v);
            }
            return acc;
        }

        // sup over the lines of the inf along each (SI), or the other way (IS)
        void supInf(std::uint8_t* mask, Index y, Index x, std::vector<std::uint8_t>& tmp, bool supOfInf) {
            tmp.assign(mask, mask + static_cast<std::size_t>(y * x));
            for (Index r = 0; r < y; ++r)
                for (Index c = 0; c < x; ++c) {
                    std::uint8_t acc = supOfInf ? 0 : 1;
                    for (int line = 0; line < 4; ++line) {
                        const std::uint8_t v = lineValue(tmp.data(), y, x, r, c, line, !supOfInf);
                        acc = supOfInf ? std::max(acc, v) : std::min(acc, v);
                    }
                    mask[r * x + c] = acc;
                }
        }
    } // namespace

    void morphologicalChanVesePlane(const float* image, std::uint8_t* mask, Index y, Index x, int iterations, int smoothing,
                                    std::vector<std::uint8_t>& tmp) {
        const Index n = y * x;
        if (iterations <= 0 || n <= 0) return;
        for (int it = 0; it < iterations; ++it) {
            double sumIn = 0.0, countIn = 0.0, sumOut = 0.0, countOut = 0.0;
            for (Index i = 0; i < n; ++i) {
                if (mask[i]) {
                    sumIn += image[i];
                    countIn += 1.0;
                } else {
                    sumOut += image[i];
                    countOut += 1.0;
                }
            }
            if (countIn == 0.0 || countOut == 0.0) return;   // the contour has vanished or filled the plane
            const double c1 = sumIn / countIn, c0 = sumOut / countOut;
            // the region force only acts where the contour is: |grad u| is
            // zero everywhere else, which is what keeps this a narrow band
            tmp.assign(mask, mask + static_cast<std::size_t>(n));
            for (Index r = 0; r < y; ++r)
                for (Index c = 0; c < x; ++c) {
                    const Index i = r * x + c;
                    const Index rp = std::min(r + 1, y - 1), rm = std::max<Index>(r - 1, 0);
                    const Index cp = std::min(c + 1, x - 1), cm = std::max<Index>(c - 1, 0);
                    const int gy = static_cast<int>(tmp[static_cast<std::size_t>(rp * x + c)]) - tmp[static_cast<std::size_t>(rm * x + c)];
                    const int gx = static_cast<int>(tmp[static_cast<std::size_t>(r * x + cp)]) - tmp[static_cast<std::size_t>(r * x + cm)];
                    if (gy == 0 && gx == 0) continue;
                    const double v = image[i];
                    const double aux = (v - c1) * (v - c1) - (v - c0) * (v - c0);
                    if (aux < 0.0) mask[i] = 1;        // closer to the inside
                    else if (aux > 0.0) mask[i] = 0;   // closer to the outside
                }
            for (int k = 0; k < smoothing; ++k) {
                // alternating, as the paper has it: SI then IS, then IS then SI
                supInf(mask, y, x, tmp, (it + k) % 2 == 0);
                supInf(mask, y, x, tmp, (it + k) % 2 != 0);
            }
        }
    }

    std::uint32_t filterLabelsByShape(std::uint32_t* labels, Index z, Index y, Index x, const ShapeFilter& filter) {
        const Index n = z * y * x;
        std::uint32_t maxId = 0;
        for (Index i = 0; i < n; ++i) maxId = std::max(maxId, labels[i]);
        if (maxId == 0) return 0;
        struct Acc {
            Index z0 = std::numeric_limits<Index>::max(), z1 = 0, y0 = std::numeric_limits<Index>::max(), y1 = 0;
            Index x0 = std::numeric_limits<Index>::max(), x1 = 0;
            Index voxels = 0;
        };
        std::vector<Acc> acc(static_cast<std::size_t>(maxId) + 1);
        for (Index k = 0; k < z; ++k)
            for (Index r = 0; r < y; ++r)
                for (Index c = 0; c < x; ++c) {
                    const std::uint32_t id = labels[(k * y + r) * x + c];
                    if (id == 0) continue;
                    Acc& a = acc[id];
                    a.z0 = std::min(a.z0, k);
                    a.z1 = std::max(a.z1, k + 1);
                    a.y0 = std::min(a.y0, r);
                    a.y1 = std::max(a.y1, r + 1);
                    a.x0 = std::min(a.x0, c);
                    a.x1 = std::max(a.x1, c + 1);
                    ++a.voxels;
                }
        std::vector<std::uint32_t> remap(acc.size(), 0);
        std::uint32_t next = 0;
        for (std::uint32_t id = 1; id <= maxId; ++id) {
            const Acc& a = acc[id];
            if (a.voxels == 0) continue;
            if (filter.minVoxels > 0 && a.voxels < filter.minVoxels) continue;
            if (filter.maxVoxels > 0 && a.voxels > filter.maxVoxels) continue;
            if (filter.dropBorder && (a.y0 == 0 || a.y1 == y || a.x0 == 0 || a.x1 == x)) continue;
            const Index dz = a.z1 - a.z0, dy = a.y1 - a.y0, dx = a.x1 - a.x0;
            if (filter.minFill > 0.0) {
                const double box = static_cast<double>(dz) * static_cast<double>(dy) * static_cast<double>(dx);
                if (box > 0.0 && static_cast<double>(a.voxels) / box < filter.minFill) continue;
            }
            if (filter.maxElongation > 0.0) {
                // in plane only: a stack of few, thick planes is not elongated
                // in z in any sense the caller means by the word
                const Index longSide = std::max(dy, dx), shortSide = std::max<Index>(1, std::min(dy, dx));
                if (static_cast<double>(longSide) / static_cast<double>(shortSide) > filter.maxElongation) continue;
            }
            remap[id] = ++next;
        }
        for (Index i = 0; i < n; ++i) labels[i] = remap[labels[i]];
        return next;
    }

    std::array<float, 3> labelColor(std::uint32_t id) noexcept {
        if (!id) return {0.f, 0.f, 0.f};
        return kPalette[(id - 1) % kPalette.size()];
    }

} // namespace sirius::app
