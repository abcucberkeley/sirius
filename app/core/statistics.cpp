#include "core/statistics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include <sirius/pixel_type.hpp>

#include "core/array_source.hpp"
#include "core/cancel.hpp"
#include "core/tracks.hpp"

namespace sirius::app {

    namespace {

        constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
        constexpr double kInf = std::numeric_limits<double>::infinity();
        // Enough to draw any histogram an agent can read; a bin count typed
        // with a few zeros too many should not allocate gigabytes.
        constexpr int kMaxHistogramBins = 1 << 16;
        // Up to this many percentiles a selection each is cheaper than
        // sorting the samples once; beyond it the sort wins.
        constexpr std::size_t kSelectedPercentiles = 8;
        // The most samples a channel keeps, whatever is asked: 256 MB of
        // floats, one channel at a time. More buys percentiles no one could
        // tell from these, and a request is not bounded by the memory of the
        // machine it runs on.
        constexpr std::uint64_t kMaxSamples = std::uint64_t{1} << 26;

        std::string number(double v) {
            std::ostringstream s;
            s << v;
            return s.str();
        }

        // Moments of a run of values, merged pairwise (Chan, Golub and
        // LeVeque). Each plane's are taken around its own mean, so a large
        // offset -- a camera baseline of 100 under a signal of 2 -- costs no
        // precision, as a sum of squares over the whole channel would.
        struct Moments {
            std::uint64_t n = 0;
            double mean = 0.0, m2 = 0.0;

            void merge(const Moments& b) {
                if (b.n == 0) return;
                if (n == 0) {
                    *this = b;
                    return;
                }
                const double na = static_cast<double>(n), nb = static_cast<double>(b.n), total = na + nb;
                const double delta = b.mean - mean;
                mean += delta * nb / total;
                m2 += b.m2 + delta * delta * na * nb / total;
                n += b.n;
            }
        };

        // One channel while its planes stream past.
        struct Accumulator {
            Moments moments;
            double min = kInf, max = -kInf;
            std::uint64_t nan = 0, saturated = 0;
            std::vector<float> samples;
            Index next = 0;   // the next position to sample, counted over the channel's values
        };

        // The stride that takes at most maxSamples of n values. A stride that
        // shares a factor with the row length takes every sample from the same
        // few columns -- 1024 on planes 1024 wide reads column 0 of every row,
        // the image's edge -- so it is lengthened until the two are coprime
        // and the samples walk across the columns from one row to the next.
        Index sampleStride(Index n, std::uint64_t maxSamples, Index rowLength) {
            const std::uint64_t most = std::clamp<std::uint64_t>(maxSamples, 1, kMaxSamples);
            if (n <= 0 || static_cast<std::uint64_t>(n) <= most) return 1;
            Index stride = static_cast<Index>((static_cast<std::uint64_t>(n) + most - 1) / most);
            while (rowLength > 1 && std::gcd(stride, rowLength) != 1) ++stride;
            return stride;
        }

        // `base` is the position of v[0] among the channel's values, which is
        // what keeps the stride running on across planes.
        void addPlane(Accumulator& acc, const float* v, Index n, Index base, Index stride, double saturation) {
            Moments m;
            double sum = 0.0, lo = kInf, hi = -kInf;
            std::uint64_t nan = 0, saturated = 0;
            for (Index i = 0; i < n; ++i) {
                const double x = v[i];
                if (std::isnan(x)) {
                    ++nan;
                    continue;
                }
                ++m.n;
                sum += x;
                lo = std::min(lo, x);
                hi = std::max(hi, x);
                if (saturation > 0.0 && x >= saturation) ++saturated;
            }
            if (m.n > 0) {
                m.mean = sum / static_cast<double>(m.n);
                for (Index i = 0; i < n; ++i) {
                    const double x = v[i];
                    if (std::isnan(x)) continue;
                    const double d = x - m.mean;
                    m.m2 += d * d;
                }
            }
            acc.moments.merge(m);
            acc.min = std::min(acc.min, lo);
            acc.max = std::max(acc.max, hi);
            acc.nan += nan;
            acc.saturated += saturated;

            Index i = acc.next - base;
            for (; i < n; i += stride)
                if (!std::isnan(v[i])) acc.samples.push_back(v[i]);
            acc.next = base + i;
        }

        // numpy's _lerp between finite neighbours: a + frac (b - a) below the
        // midpoint and b - (1 - frac) (b - a) from it on, so each half
        // rounds toward its nearer neighbour and the result matches numpy to
        // the last bit. Beside an infinity that form turns into inf - inf,
        // NaN, where any weight on an infinity is that infinity: an equal
        // pair is itself, and an infinity on one side wins. Only -inf beside
        // +inf has no value.
        double interpolate(double a, double b, double frac) {
            if (a == b) return a;
            if (std::isfinite(a) && std::isfinite(b)) {
                const double diff = b - a;
                return frac >= 0.5 ? b - diff * (1.0 - frac) : a + diff * frac;
            }
            return (1.0 - frac) * a + frac * b;
        }

        // Linear interpolation between the two nearest order statistics,
        // numpy's default, in the order the percentiles were asked for.
        std::vector<double> percentilesOf(std::vector<float>& s, const std::vector<double>& ps) {
            std::vector<double> out(ps.size(), kNaN);
            if (s.empty()) return out;
            const std::size_t n = s.size();
            std::vector<std::size_t> order(ps.size());
            std::iota(order.begin(), order.end(), std::size_t{0});
            std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return ps[a] < ps[b]; });
            const bool sorted = ps.size() > kSelectedPercentiles;
            if (sorted) std::sort(s.begin(), s.end());
            // Ascending ranks: after a selection at `from`, nothing before it
            // is larger than anything after, so the next one only partitions
            // the tail.
            std::size_t from = 0;
            for (std::size_t k : order) {
                const double pos = ps[k] / 100.0 * static_cast<double>(n - 1);
                const std::size_t lo = std::min(n - 1, static_cast<std::size_t>(std::floor(pos)));
                const double frac = pos - static_cast<double>(lo);
                if (!sorted) {
                    std::nth_element(s.begin() + static_cast<std::ptrdiff_t>(from), s.begin() + static_cast<std::ptrdiff_t>(lo),
                                     s.end());
                    from = lo;
                }
                const double a = s[lo];
                double b = a;
                if (frac > 0.0 && lo + 1 < n)
                    b = sorted ? s[lo + 1] : *std::min_element(s.begin() + static_cast<std::ptrdiff_t>(lo + 1), s.end());
                out[k] = interpolate(a, b, frac);
            }
            return out;
        }

        void fillHistogram(ChannelStatistics& st, const std::vector<float>& samples, int bins) {
            st.histogram.assign(static_cast<std::size_t>(bins), 0);
            double lo = st.min, hi = st.max;
            if (!std::isfinite(lo) || !std::isfinite(hi)) {
                // An infinity (a division by zero upstream) would make every
                // bin infinitely wide: span the finite samples instead, and
                // let the infinities land in the end bins.
                lo = kInf;
                hi = -kInf;
                for (float v : samples)
                    if (std::isfinite(v)) {
                        lo = std::min(lo, static_cast<double>(v));
                        hi = std::max(hi, static_cast<double>(v));
                    }
                if (lo > hi) lo = hi = 0.0;
            }
            st.histLo = lo;
            st.histHi = hi;
            const double width = hi - lo;
            const double last = static_cast<double>(bins - 1);
            for (float v : samples) {
                // a constant channel has zero width: everything is one bin
                const double f = width > 0.0 ? (static_cast<double>(v) - lo) / width * static_cast<double>(bins) : 0.0;
                const std::size_t b = static_cast<std::size_t>(std::clamp(std::floor(f), 0.0, last));
                ++st.histogram[b];
            }
        }

    } // namespace

    std::vector<ChannelStatistics> channelStatistics(const StepOutput& out, const StatisticsOptions& o,
                                                     const std::function<void(double)>& progress,
                                                     const std::function<bool()>& cancelled) {
        // The array when there is one: planes are read in place. A lazy
        // source (Load before a full load) is read a plane at a time.
        const Array5* array = out.array && !out.array->empty() ? out.array.get() : nullptr;
        if (!array && !out.source) throw std::runtime_error("the output holds no data");
        const Dims5 dims = array ? array->dims() : out.source->dims();

        std::vector<Index> channels = o.channels;
        if (channels.empty()) {
            channels.resize(static_cast<std::size_t>(dims.c));
            std::iota(channels.begin(), channels.end(), Index{0});
        }
        for (Index c : channels)
            if (c < 0 || c >= dims.c)
                throw std::invalid_argument("channel " + std::to_string(c) + " is outside [0, " + std::to_string(dims.c) + ")");
        if (o.t < -1 || o.t >= dims.t)
            throw std::invalid_argument("t " + std::to_string(o.t) + " is outside [0, " + std::to_string(dims.t) +
                                        ") (or -1 for every time point)");
        for (double p : o.percentiles)
            if (!(p >= 0.0 && p <= 100.0)) throw std::invalid_argument("percentile " + number(p) + " is outside [0, 100]");
        if (o.histogramBins < 0 || o.histogramBins > kMaxHistogramBins)
            throw std::invalid_argument("histogram bins " + std::to_string(o.histogramBins) + " is outside [0, " +
                                        std::to_string(kMaxHistogramBins) + "]");

        const Index t0 = o.t < 0 ? 0 : o.t, t1 = o.t < 0 ? dims.t : o.t + 1;
        const Index planeSize = dims.planeSize();
        const Index values = (t1 - t0) * dims.z * planeSize;   // per channel, NaN included
        const Index stride = sampleStride(values, o.maxSamples, dims.x);
        const Index planesTotal = static_cast<Index>(channels.size()) * (t1 - t0) * dims.z;
        std::vector<float> plane(array ? 0 : static_cast<std::size_t>(planeSize));

        std::vector<ChannelStatistics> result;
        result.reserve(channels.size());
        Index done = 0;
        for (Index c : channels) {
            Accumulator acc;
            // Every position the stride visits, at most kMaxSamples, reserved
            // at once: growing as the samples come would hold the old buffer
            // and the new one together, and over-allocate on the last step.
            acc.samples.reserve(static_cast<std::size_t>((values + stride - 1) / stride));
            Index base = 0;
            for (Index t = t0; t < t1; ++t)
                for (Index z = 0; z < dims.z; ++z) {
                    if (cancelled && cancelled()) throw CancelledError();
                    const float* v = nullptr;
                    if (array) {
                        v = array->plane(c, t, z);
                    } else {
                        out.source->readPlane(c, t, z, plane.data());
                        v = plane.data();
                    }
                    addPlane(acc, v, planeSize, base, stride, o.saturationLevel);
                    base += planeSize;
                    ++done;
                    if (progress && planesTotal > 0) progress(static_cast<double>(done) / static_cast<double>(planesTotal));
                }

            ChannelStatistics st;
            st.channel = c;
            st.count = acc.moments.n;
            st.nanCount = acc.nan;
            st.sampled = stride > 1;
            if (st.count > 0) {
                st.min = acc.min;
                st.max = acc.max;
                st.mean = acc.moments.mean;
                st.stddev = std::sqrt(acc.moments.m2 / static_cast<double>(st.count));
            } else {
                st.min = st.max = st.mean = st.stddev = kNaN;
            }
            if (o.saturationLevel > 0.0)
                st.saturatedFraction = st.count > 0 ? static_cast<double>(acc.saturated) / static_cast<double>(st.count) : kNaN;
            const std::vector<double> p = percentilesOf(acc.samples, o.percentiles);
            st.percentiles.reserve(p.size());
            for (std::size_t k = 0; k < p.size(); ++k) st.percentiles.emplace_back(o.percentiles[k], p[k]);
            if (o.histogramBins > 0) fillHistogram(st, acc.samples, o.histogramBins);
            result.push_back(std::move(st));
        }
        return result;
    }

    double pixelTypeMaximum(const DatasetMeta& meta) {
        switch (meta.sourceType) {
            case PixelType::UInt8: return std::numeric_limits<std::uint8_t>::max();
            case PixelType::Int8: return std::numeric_limits<std::int8_t>::max();
            case PixelType::UInt16: return std::numeric_limits<std::uint16_t>::max();
            case PixelType::Int16: return std::numeric_limits<std::int16_t>::max();
            case PixelType::UInt32: return std::numeric_limits<std::uint32_t>::max();
            case PixelType::Int32: return std::numeric_limits<std::int32_t>::max();
            case PixelType::Float32:
            case PixelType::Float64: return 0.0;   // no ceiling a detector runs into
        }
        return 0.0;
    }

    nlohmann::json labelStatistics(const LabelVolume& labels, Index t, const std::array<double, 3>& voxelUm) {
        using nlohmann::json;
        if (!labels.empty() && (t < 0 || t >= labels.t()))
            throw std::invalid_argument("t " + std::to_string(t) + " is outside the labels' [0, " + std::to_string(labels.t()) + ")");

        // The table of frame t: the volume's own when it describes t, which
        // is what the reviewer has edited; else one recomputed on a volume
        // that shares the voxels -- share() copies the table and the
        // annotations, not the voxels, and a recompute only reads them -- so
        // asking about another frame leaves the volume's table where it was.
        static const std::vector<LabelStats> kNone;
        std::shared_ptr<LabelVolume> other;
        const std::vector<LabelStats>* rows = labels.empty() ? &kNone : &labels.stats();
        if (!labels.empty() && labels.statsT() != t) {
            other = labels.share();
            other->recomputeStats(t);
            rows = &other->stats();
        }

        std::vector<Index> sizes;
        sizes.reserve(rows->size());
        std::map<std::string, Index> classes, flags;
        Index reviewed = 0;
        for (const LabelStats& s : *rows) {
            sizes.push_back(s.voxels);
            ++classes[s.cls];
            for (const std::string& f : s.flags) ++flags[f];
            if (s.reviewed) ++reviewed;
        }
        std::sort(sizes.begin(), sizes.end());
        double lo = 0.0, median = 0.0, mean = 0.0, hi = 0.0;
        if (!sizes.empty()) {
            const std::size_t n = sizes.size();
            lo = static_cast<double>(sizes.front());
            hi = static_cast<double>(sizes.back());
            median = (n % 2 != 0) ? static_cast<double>(sizes[n / 2])
                                  : 0.5 * (static_cast<double>(sizes[n / 2 - 1]) + static_cast<double>(sizes[n / 2]));
            double total = 0.0;
            for (Index s : sizes) total += static_cast<double>(s);
            mean = total / static_cast<double>(n);
        }
        // x * y * z: the order of the axes does not matter to a volume
        const double voxel = voxelUm[0] * voxelUm[1] * voxelUm[2];

        json j = {{"count", sizes.size()},
                  {"t", t},
                  // min and max are counts, so integers; the median of an even
                  // count and the mean are not
                  {"voxels", {{"min", static_cast<std::int64_t>(lo)}, {"median", median}, {"mean", mean}, {"max", static_cast<std::int64_t>(hi)}}},
                  {"volume_um3", {{"min", lo * voxel}, {"median", median * voxel}, {"mean", mean * voxel}, {"max", hi * voxel}}},
                  {"flags", flags},
                  {"classes", classes},
                  {"reviewed", reviewed},
                  {"tracked", labels.tracked()}};
        if (labels.tracked() && !labels.empty()) {
            // A tracking step leaves the index built; a volume that lost it
            // to a raw write is indexed here, once, without keeping it.
            std::shared_ptr<const TrackIndex> index = labels.tracks();
            if (!index) index = std::make_shared<const TrackIndex>(labels.frames());
            const std::vector<TrackSummary> tracks =
                summarizeTracks(*index, labels.lineage(), {voxelUm[2], voxelUm[1], voxelUm[0]});   // (z, y, x)
            Index gapped = 0;
            for (const TrackSummary& r : tracks) gapped += r.gaps > 0 ? 1 : 0;
            j["tracks"] = {{"total", tracks.size()}, {"with_gaps", gapped}, {"divisions", countDivisions(tracks)}};
        }
        return j;
    }

} // namespace sirius::app
