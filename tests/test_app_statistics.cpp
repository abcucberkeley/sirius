// Intensity and label statistics of a step's output (app/core/statistics.hpp):
// the exact moments and counts, the percentiles and histogram drawn from a
// strided subsample (including a stride that would alias with the rows), NaN
// and infinity handling, the saturated fraction against the source's pixel
// type, the lazy-source path agreeing with the in-memory one, and the label
// summary of any frame without disturbing the volume's own table.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/array_source.hpp"
#include "core/cancel.hpp"
#include "core/labels.hpp"
#include "core/statistics.hpp"

using namespace sirius;
using namespace sirius::app;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

    using Fill = std::function<float(Index c, Index t, Index z, Index y, Index x)>;

    // A step output of `d` filled by `f`: in memory, or behind a MemorySource
    // with no array, which is what a lazily loaded dataset looks like.
    StepOutput outputOf(const Dims5& d, const Fill& f, bool lazy = false, PixelType type = PixelType::Float32) {
        auto a = std::make_shared<Array5>(d);
        for (Index c = 0; c < d.c; ++c)
            for (Index t = 0; t < d.t; ++t)
                for (Index z = 0; z < d.z; ++z)
                    for (Index y = 0; y < d.y; ++y)
                        for (Index x = 0; x < d.x; ++x) a->at(c, t, z, y, x) = f(c, t, z, y, x);
        StepOutput out;
        out.meta.name = "test";
        out.meta.format = "memory";
        out.meta.dims = d;
        out.meta.sourceType = type;
        out.meta.normalizeChannels();
        if (lazy) out.source = std::make_shared<MemorySource>(a, out.meta);
        else out.array = a;
        return out;
    }

    // The position of (t, z, y, x) among one channel's values, from 1.
    Fill ramp(const Dims5& d) {
        return [d](Index, Index t, Index z, Index y, Index x) {
            return static_cast<float>(((t * d.z + z) * d.y + y) * d.x + x + 1);
        };
    }

    double percentile(const ChannelStatistics& s, double p) {
        for (const auto& [q, v] : s.percentiles)
            if (q == p) return v;
        FAIL("percentile " << p << " was not reported");
        return 0.0;
    }

    std::uint64_t histogramTotal(const ChannelStatistics& s) {
        std::uint64_t n = 0;
        for (std::uint64_t b : s.histogram) n += b;
        return n;
    }

    // A cube of `id` with its low corner at (z, y, x) and `side` voxels.
    void cube(LabelVolume& labels, Index t, std::uint32_t id, Index z, Index y, Index x, Index side) {
        std::uint32_t* v = labels.volume(t);
        for (Index iz = z; iz < z + side; ++iz)
            for (Index iy = y; iy < y + side; ++iy)
                for (Index ix = x; ix < x + side; ++ix) v[(iz * labels.y() + iy) * labels.x() + ix] = id;
    }

    // Three frames: ids 1 (8 voxels, frames 0 and 2 but not 1), 2 (8 voxels,
    // every frame), 3 (27 voxels on the border, frame 0) and 4 (1 voxel,
    // frame 1). The table describes frame 0, where id 2 is a reviewed nucleus.
    LabelVolume sampleLabels() {
        LabelVolume labels(3, 4, 10, 10);
        cube(labels, 0, 1, 1, 3, 3, 2);
        cube(labels, 0, 2, 1, 5, 5, 2);
        cube(labels, 0, 3, 0, 0, 0, 3);
        cube(labels, 1, 2, 1, 5, 5, 2);
        cube(labels, 1, 4, 2, 8, 8, 1);
        cube(labels, 2, 1, 1, 2, 6, 2);
        cube(labels, 2, 2, 1, 5, 5, 2);
        labels.recomputeStats(0);
        labels.applyFlags(LabelFlagRules{});
        for (LabelStats& s : labels.stats())
            if (s.id == 2) {
                s.cls = "nucleus";
                s.reviewed = true;
            }
        return labels;
    }

} // namespace

TEST_CASE("statistics: exact min, max, mean and std of a known array", "[app][statistics]") {
    const Dims5 d{2, 2, 3, 4, 5};   // 60 values per channel and time point
    const Fill f = [&](Index c, Index t, Index z, Index y, Index x) {
        const float k = ramp(d)(c, t, z, y, x);   // 1..120 over both time points
        return c == 0 ? k : 100.0f + 2.0f * k;
    };
    const StepOutput out = outputOf(d, f);

    SECTION("every channel over every time point") {
        StatisticsOptions o;
        o.t = -1;
        o.percentiles = {0.0, 25.0, 50.0, 100.0};
        const std::vector<ChannelStatistics> s = channelStatistics(out, o);
        REQUIRE(s.size() == 2);
        CHECK(s[0].channel == 0);
        CHECK(s[0].count == 120);
        CHECK(s[0].nanCount == 0);
        CHECK_FALSE(s[0].sampled);
        CHECK(s[0].min == 1.0);
        CHECK(s[0].max == 120.0);
        CHECK_THAT(s[0].mean, WithinAbs(60.5, 1e-12));
        // the population standard deviation of 1..n is sqrt((n^2 - 1) / 12)
        CHECK_THAT(s[0].stddev, WithinAbs(std::sqrt((120.0 * 120.0 - 1.0) / 12.0), 1e-9));
        // linear interpolation between order statistics, as numpy
        CHECK(percentile(s[0], 0.0) == 1.0);
        CHECK_THAT(percentile(s[0], 25.0), WithinAbs(30.75, 1e-12));
        CHECK_THAT(percentile(s[0], 50.0), WithinAbs(60.5, 1e-12));
        CHECK(percentile(s[0], 100.0) == 120.0);
        CHECK_FALSE(s[0].saturatedFraction);
        CHECK(s[0].histogram.empty());

        CHECK(s[1].channel == 1);
        CHECK(s[1].min == 102.0);
        CHECK(s[1].max == 340.0);
        CHECK_THAT(s[1].mean, WithinAbs(221.0, 1e-12));
        CHECK_THAT(s[1].stddev, WithinAbs(2.0 * std::sqrt((120.0 * 120.0 - 1.0) / 12.0), 1e-9));
    }
    SECTION("one channel at one time point") {
        StatisticsOptions o;
        o.channels = {1};
        o.t = 1;
        const std::vector<ChannelStatistics> s = channelStatistics(out, o);
        REQUIRE(s.size() == 1);
        CHECK(s[0].channel == 1);
        CHECK(s[0].count == 60);
        CHECK(s[0].min == 222.0);   // 100 + 2 * 61
        CHECK(s[0].max == 340.0);
        CHECK_THAT(s[0].mean, WithinAbs(281.0, 1e-12));
        CHECK_THAT(s[0].stddev, WithinAbs(2.0 * std::sqrt((60.0 * 60.0 - 1.0) / 12.0), 1e-9));
    }
    SECTION("the percentiles come back in the order they were asked for, by selection or by sort") {
        StatisticsOptions o;
        o.channels = {0};
        o.t = -1;
        o.percentiles = {99.0, 1.0, 50.0};
        std::vector<ChannelStatistics> s = channelStatistics(out, o);
        REQUIRE(s[0].percentiles.size() == 3);
        CHECK(s[0].percentiles[0].first == 99.0);
        CHECK(s[0].percentiles[1].first == 1.0);
        CHECK(s[0].percentiles[2].first == 50.0);
        // on 1..120 the interpolated percentile p is 1 + p / 100 * 119
        for (const auto& [p, v] : s[0].percentiles) CHECK_THAT(v, WithinAbs(1.0 + p * 1.19, 1e-9));

        o.percentiles.clear();   // more than the selection handles: one sort
        for (int p = 100; p >= 0; p -= 5) o.percentiles.push_back(p);
        s = channelStatistics(out, o);
        REQUIRE(s[0].percentiles.size() == o.percentiles.size());
        for (std::size_t k = 0; k < o.percentiles.size(); ++k) {
            CHECK(s[0].percentiles[k].first == o.percentiles[k]);
            CHECK_THAT(s[0].percentiles[k].second, WithinAbs(1.0 + o.percentiles[k] * 1.19, 1e-9));
        }
    }
    SECTION("a lazy source gives what the array gives") {
        const StepOutput lazy = outputOf(d, f, true);
        REQUIRE_FALSE(lazy.array);
        StatisticsOptions o;
        o.t = -1;
        o.histogramBins = 7;
        o.saturationLevel = 300.0;
        const std::vector<ChannelStatistics> a = channelStatistics(out, o), b = channelStatistics(lazy, o);
        REQUIRE(a.size() == b.size());
        for (std::size_t c = 0; c < a.size(); ++c) {
            CHECK(a[c].min == b[c].min);
            CHECK(a[c].max == b[c].max);
            CHECK(a[c].mean == b[c].mean);
            CHECK(a[c].stddev == b[c].stddev);
            CHECK(a[c].count == b[c].count);
            CHECK(a[c].percentiles == b[c].percentiles);
            CHECK(a[c].histogram == b[c].histogram);
            CHECK(a[c].saturatedFraction == b[c].saturatedFraction);
        }
    }
}

TEST_CASE("statistics: the moments keep their precision under a large offset", "[app][statistics]") {
    // A camera baseline far above the signal: a sum of squares over the
    // channel would lose the 0.5 to cancellation, the merged moments do not.
    const Dims5 d{1, 1, 16, 64, 64};
    const StepOutput out = outputOf(d, [](Index, Index, Index z, Index y, Index x) {
        return 1.0e6f + static_cast<float>((z + y + x) % 2);
    });
    const std::vector<ChannelStatistics> s = channelStatistics(out, StatisticsOptions{});
    REQUIRE(s.size() == 1);
    CHECK_THAT(s[0].mean, WithinAbs(1.0e6 + 0.5, 1e-9));
    CHECK_THAT(s[0].stddev, WithinAbs(0.5, 1e-9));
}

TEST_CASE("statistics: percentiles of a sampled array stay close to the exact ones", "[app][statistics]") {
    SECTION("uniform noise") {
        const Dims5 d{1, 1, 8, 128, 128};
        // mt19937's output is the same on every platform, which the
        // standard distributions' are not
        std::mt19937 rng(20260929u);
        std::vector<float> values(static_cast<std::size_t>(d.numel()));
        for (float& v : values) v = static_cast<float>(rng() >> 8) / 16777216.0f;
        const StepOutput out = outputOf(d, [&](Index, Index, Index z, Index y, Index x) {
            return values[static_cast<std::size_t>((z * d.y + y) * d.x + x)];
        });
        StatisticsOptions o;
        o.maxSamples = 4096;
        o.histogramBins = 20;
        const std::vector<ChannelStatistics> s = channelStatistics(out, o);
        REQUIRE(s.size() == 1);
        CHECK(s[0].sampled);
        // the moments and the extremes stay exact
        CHECK(s[0].count == values.size());
        CHECK(s[0].min == *std::min_element(values.begin(), values.end()));
        CHECK(s[0].max == *std::max_element(values.begin(), values.end()));
        double sum = 0.0;
        for (float v : values) sum += v;
        CHECK_THAT(s[0].mean, WithinAbs(sum / static_cast<double>(values.size()), 1e-9));

        std::vector<float> sorted = values;
        std::sort(sorted.begin(), sorted.end());
        const auto exact = [&](double p) {
            return static_cast<double>(sorted[static_cast<std::size_t>(p / 100.0 * static_cast<double>(sorted.size() - 1))]);
        };
        CHECK_THAT(percentile(s[0], 1.0), WithinAbs(exact(1.0), 0.01));
        CHECK_THAT(percentile(s[0], 50.0), WithinAbs(exact(50.0), 0.03));
        CHECK_THAT(percentile(s[0], 99.0), WithinAbs(exact(99.0), 0.01));
        const std::uint64_t n = histogramTotal(s[0]);
        CHECK(n > 0);
        CHECK(n <= o.maxSamples);
    }
    SECTION("a stride that divides the row length does not read the same column every row") {
        // A ramp across x only. 524288 values into 2048 samples is a stride
        // of 256, the row length: every sample would be column 0.
        const Dims5 d{1, 1, 8, 256, 256};
        const StepOutput out = outputOf(d, [](Index, Index, Index, Index, Index x) { return static_cast<float>(x) / 255.0f; });
        StatisticsOptions o;
        o.maxSamples = 2048;
        o.histogramBins = 8;
        const std::vector<ChannelStatistics> s = channelStatistics(out, o);
        REQUIRE(s.size() == 1);
        CHECK(s[0].sampled);
        CHECK_THAT(percentile(s[0], 1.0), WithinAbs(0.01, 0.01));
        CHECK_THAT(percentile(s[0], 50.0), WithinAbs(0.5, 0.01));
        CHECK_THAT(percentile(s[0], 99.0), WithinAbs(0.99, 0.01));
        CHECK(histogramTotal(s[0]) <= o.maxSamples);
        for (std::uint64_t b : s[0].histogram) CHECK(b > 0);   // every column range is represented
    }
}

TEST_CASE("statistics: NaN is counted apart and left out of everything else", "[app][statistics]") {
    const Dims5 d{2, 1, 2, 3, 4};   // 24 values per channel
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const StepOutput out = outputOf(d, [&](Index c, Index, Index z, Index y, Index x) {
        const Index i = (z * d.y + y) * d.x + x;
        if (c == 1) return nan;   // nothing to measure
        return i % 5 == 0 ? nan : static_cast<float>(i);
    });
    StatisticsOptions o;
    o.percentiles = {0.0, 100.0};
    o.histogramBins = 4;
    o.saturationLevel = 10.0;
    const std::vector<ChannelStatistics> s = channelStatistics(out, o);
    REQUIRE(s.size() == 2);

    CHECK(s[0].count == 19);
    CHECK(s[0].nanCount == 5);   // 0, 5, 10, 15, 20
    CHECK(s[0].min == 1.0);
    CHECK(s[0].max == 23.0);
    CHECK_THAT(s[0].mean, WithinAbs((276.0 - 50.0) / 19.0, 1e-12));
    CHECK(percentile(s[0], 0.0) == 1.0);
    CHECK(percentile(s[0], 100.0) == 23.0);
    CHECK(histogramTotal(s[0]) == 19);
    // 11..23 but for 15 and 20, out of the 19 that are not NaN
    REQUIRE(s[0].saturatedFraction);
    CHECK_THAT(*s[0].saturatedFraction, WithinAbs(11.0 / 19.0, 1e-12));

    // an all-NaN channel reports no number rather than a made-up 0
    CHECK(s[1].count == 0);
    CHECK(s[1].nanCount == 24);
    CHECK(std::isnan(s[1].min));
    CHECK(std::isnan(s[1].max));
    CHECK(std::isnan(s[1].mean));
    CHECK(std::isnan(s[1].stddev));
    CHECK(std::isnan(percentile(s[1], 0.0)));
    REQUIRE(s[1].saturatedFraction);
    CHECK(std::isnan(*s[1].saturatedFraction));
    CHECK(s[1].histogram.size() == 4);
    CHECK(histogramTotal(s[1]) == 0);
}

TEST_CASE("statistics: a percentile beside an infinity is that infinity, not NaN", "[app][statistics]") {
    const double inf = std::numeric_limits<double>::infinity();
    const float finf = std::numeric_limits<float>::infinity();
    SECTION("infinities at both ends") {
        // sorted: -inf, -inf, 2, 3, 4, 5, 6, 7, +inf, +inf; position p / 100 * 9
        const StepOutput out = outputOf(Dims5{1, 1, 1, 1, 10}, [&](Index, Index, Index, Index, Index x) {
            if (x < 2) return -finf;
            return x > 7 ? finf : static_cast<float>(x);
        });
        StatisticsOptions o;
        o.percentiles = {0.0, 5.0, 15.0, 50.0, 85.0, 95.0, 100.0};
        std::vector<ChannelStatistics> s = channelStatistics(out, o);
        CHECK(percentile(s[0], 0.0) == -inf);    // -inf itself
        CHECK(percentile(s[0], 5.0) == -inf);    // between -inf and -inf
        CHECK(percentile(s[0], 15.0) == -inf);   // between -inf and 2
        CHECK(percentile(s[0], 50.0) == 4.5);
        CHECK(percentile(s[0], 85.0) == inf);    // between 7 and +inf
        CHECK(percentile(s[0], 95.0) == inf);
        CHECK(percentile(s[0], 100.0) == inf);

        // the sort path interpolates the same way
        for (int p = 0; p <= 100; p += 10) o.percentiles.push_back(p);
        s = channelStatistics(out, o);
        CHECK(percentile(s[0], 15.0) == -inf);
        CHECK(percentile(s[0], 85.0) == inf);
    }
    SECTION("between -inf and +inf there is no value") {
        const StepOutput out = outputOf(Dims5{1, 1, 1, 1, 2}, [&](Index, Index, Index, Index, Index x) {
            return x == 0 ? -finf : finf;
        });
        StatisticsOptions o;
        o.percentiles = {0.0, 50.0, 100.0};
        const std::vector<ChannelStatistics> s = channelStatistics(out, o);
        CHECK(percentile(s[0], 0.0) == -inf);
        CHECK(std::isnan(percentile(s[0], 50.0)));
        CHECK(percentile(s[0], 100.0) == inf);
    }
}

TEST_CASE("statistics: the histogram spans min to max and sums to the count", "[app][statistics]") {
    SECTION("a ramp") {
        const Dims5 d{1, 1, 2, 8, 8};
        const StepOutput out = outputOf(d, [&](Index, Index, Index z, Index y, Index x) {
            return static_cast<float>((z * d.y + y) * d.x + x);   // 0..127
        });
        StatisticsOptions o;
        o.histogramBins = 10;
        const std::vector<ChannelStatistics> s = channelStatistics(out, o);
        REQUIRE(s[0].histogram.size() == 10);
        CHECK(histogramTotal(s[0]) == s[0].count);
        CHECK(s[0].count == 128);
        CHECK(s[0].histLo == 0.0);
        CHECK(s[0].histHi == 127.0);
        CHECK(s[0].histogram.front() == 13);   // 0..12, bins 12.7 wide
        CHECK(s[0].histogram.back() == 13);    // 115..127, the maximum included
    }
    SECTION("a constant channel is one bin") {
        const StepOutput out = outputOf(Dims5{1, 1, 1, 4, 4}, [](Index, Index, Index, Index, Index) { return 3.0f; });
        StatisticsOptions o;
        o.histogramBins = 5;
        const std::vector<ChannelStatistics> s = channelStatistics(out, o);
        CHECK(s[0].histLo == 3.0);
        CHECK(s[0].histHi == 3.0);
        CHECK(s[0].histogram.front() == 16);
        CHECK(histogramTotal(s[0]) == 16);
        CHECK(s[0].stddev == 0.0);
    }
    SECTION("an infinity lands in an end bin of the finite range") {
        const float inf = std::numeric_limits<float>::infinity();
        const StepOutput out = outputOf(Dims5{1, 1, 1, 1, 10}, [&](Index, Index, Index, Index, Index x) {
            return x == 9 ? inf : static_cast<float>(x);   // 0..8, then +inf
        });
        StatisticsOptions o;
        o.histogramBins = 4;
        const std::vector<ChannelStatistics> s = channelStatistics(out, o);
        CHECK(std::isinf(s[0].max));
        CHECK(s[0].histLo == 0.0);
        CHECK(s[0].histHi == 8.0);
        CHECK(histogramTotal(s[0]) == 10);
        CHECK(s[0].histogram.back() == 4);   // 6, 7, 8 and the infinity
    }
}

TEST_CASE("statistics: the saturated fraction against the source's pixel type", "[app][statistics]") {
    DatasetMeta meta;
    meta.sourceType = PixelType::UInt8;
    CHECK(pixelTypeMaximum(meta) == 255.0);
    meta.sourceType = PixelType::Int8;
    CHECK(pixelTypeMaximum(meta) == 127.0);
    meta.sourceType = PixelType::UInt16;
    CHECK(pixelTypeMaximum(meta) == 65535.0);
    meta.sourceType = PixelType::Int16;
    CHECK(pixelTypeMaximum(meta) == 32767.0);
    meta.sourceType = PixelType::UInt32;
    CHECK(pixelTypeMaximum(meta) == 4294967295.0);
    meta.sourceType = PixelType::Int32;
    CHECK(pixelTypeMaximum(meta) == 2147483647.0);
    meta.sourceType = PixelType::Float32;
    CHECK(pixelTypeMaximum(meta) == 0.0);
    meta.sourceType = PixelType::Float64;
    CHECK(pixelTypeMaximum(meta) == 0.0);

    // 7 of 100 values clipped at the top of an 8-bit camera
    const Dims5 d{1, 1, 1, 10, 10};
    const Fill clipped = [](Index, Index, Index, Index y, Index x) {
        return y * 10 + x < 7 ? 255.0f : static_cast<float>((y * 10 + x) % 200);
    };
    const StepOutput out = outputOf(d, clipped, false, PixelType::UInt8);
    StatisticsOptions o;
    o.saturationLevel = pixelTypeMaximum(out.meta);
    std::vector<ChannelStatistics> s = channelStatistics(out, o);
    REQUIRE(s[0].saturatedFraction);
    CHECK_THAT(*s[0].saturatedFraction, WithinAbs(0.07, 1e-12));

    o.saturationLevel = 0.0;   // a float type: nothing to saturate at
    s = channelStatistics(out, o);
    CHECK_FALSE(s[0].saturatedFraction);
}

TEST_CASE("statistics: bad options, progress and cancellation", "[app][statistics]") {
    const Dims5 d{2, 3, 4, 5, 6};
    const StepOutput out = outputOf(d, ramp(d));

    SECTION("options outside the output are refused") {
        StatisticsOptions o;
        o.channels = {2};
        CHECK_THROWS_AS(channelStatistics(out, o), std::invalid_argument);
        o.channels = {-1};
        CHECK_THROWS_AS(channelStatistics(out, o), std::invalid_argument);
        o = {};
        o.t = 3;
        CHECK_THROWS_AS(channelStatistics(out, o), std::invalid_argument);
        o.t = -2;
        CHECK_THROWS_AS(channelStatistics(out, o), std::invalid_argument);
        o = {};
        o.percentiles = {50.0, 101.0};
        CHECK_THROWS_AS(channelStatistics(out, o), std::invalid_argument);
        o.percentiles = {-0.5};
        CHECK_THROWS_AS(channelStatistics(out, o), std::invalid_argument);
        o.percentiles = {std::numeric_limits<double>::quiet_NaN()};
        CHECK_THROWS_AS(channelStatistics(out, o), std::invalid_argument);
        o = {};
        o.histogramBins = -1;
        CHECK_THROWS_AS(channelStatistics(out, o), std::invalid_argument);
        o.histogramBins = 70000;
        CHECK_THROWS_AS(channelStatistics(out, o), std::invalid_argument);
        CHECK_THROWS_AS(channelStatistics(StepOutput{}, StatisticsOptions{}), std::runtime_error);
    }
    SECTION("any maxSamples is taken: 0 as 1, the largest held to the cap") {
        StatisticsOptions o;
        o.channels = {0};
        o.percentiles = {0.0, 100.0};
        o.maxSamples = std::numeric_limits<std::uint64_t>::max();
        std::vector<ChannelStatistics> s = channelStatistics(out, o);
        CHECK_FALSE(s[0].sampled);
        CHECK(percentile(s[0], 0.0) == 1.0);
        CHECK(percentile(s[0], 100.0) == 120.0);
        o.maxSamples = 0;
        o.histogramBins = 3;
        s = channelStatistics(out, o);
        CHECK(s[0].sampled);
        CHECK(histogramTotal(s[0]) == 1);
        CHECK(s[0].count == 120);
    }
    SECTION("progress climbs to 1, a plane at a time") {
        StatisticsOptions o;
        o.t = -1;
        std::vector<double> seen;
        channelStatistics(out, o, [&](double f) { seen.push_back(f); });
        REQUIRE(seen.size() == static_cast<std::size_t>(d.c * d.t * d.z));
        CHECK(std::is_sorted(seen.begin(), seen.end()));
        CHECK(std::adjacent_find(seen.begin(), seen.end()) == seen.end());
        CHECK(seen.back() == 1.0);
    }
    SECTION("cancellation stops between planes") {
        int asked = 0;
        CHECK_THROWS_AS(channelStatistics(out, StatisticsOptions{}, {}, [&] { return ++asked > 2; }), CancelledError);
        CHECK(asked == 3);
    }
}

TEST_CASE("statistics: labelStatistics summarizes one frame of a label volume", "[app][statistics][labels]") {
    const std::array<double, 3> voxelUm{0.1, 0.2, 0.5};   // x, y, z: 0.01 um3 a voxel

    SECTION("the frame the table describes") {
        const LabelVolume labels = sampleLabels();
        const nlohmann::json j = labelStatistics(labels, 0, voxelUm);
        CHECK(j["count"] == 3);
        CHECK(j["t"] == 0);
        CHECK(j["voxels"]["min"] == 8.0);
        CHECK(j["voxels"]["median"] == 8.0);
        CHECK_THAT(j["voxels"]["mean"].get<double>(), WithinAbs(43.0 / 3.0, 1e-12));
        CHECK(j["voxels"]["max"] == 27.0);
        CHECK_THAT(j["volume_um3"]["min"].get<double>(), WithinRel(0.08, 1e-12));
        CHECK_THAT(j["volume_um3"]["max"].get<double>(), WithinRel(0.27, 1e-12));
        CHECK(j["flags"] == nlohmann::json{{"touching border", 1}});
        CHECK(j["classes"] == nlohmann::json{{"nucleus", 1}, {"object", 2}});
        CHECK(j["reviewed"] == 1);
        CHECK(j["tracked"] == false);
        CHECK_FALSE(j.contains("tracks"));
    }
    SECTION("another frame, without moving the volume's table") {
        const LabelVolume labels = sampleLabels();
        const nlohmann::json j = labelStatistics(labels, 1, voxelUm);
        CHECK(j["count"] == 2);   // ids 2 and 4
        CHECK(j["t"] == 1);
        CHECK(j["voxels"]["min"] == 1.0);
        CHECK(j["voxels"]["median"] == 4.5);
        CHECK(j["voxels"]["max"] == 8.0);
        // id 2 of another frame is another object when the labels are not tracked
        CHECK(j["classes"] == nlohmann::json{{"object", 2}});
        CHECK(j["reviewed"] == 0);

        CHECK(labels.statsT() == 0);
        CHECK(labels.stats().size() == 3);
        CHECK_FALSE(labels.sharesVoxels());
    }
    SECTION("tracked labels add the tracks, and class and review follow the track") {
        LabelVolume labels = sampleLabels();
        labels.setTracked(true);
        labels.setLineage({{4, 2}});   // 4 divided from 2, which lives on
        labels.indexTracks();
        nlohmann::json j = labelStatistics(labels, 1, voxelUm);
        CHECK(j["tracked"] == true);
        CHECK(j["classes"] == nlohmann::json{{"nucleus", 1}, {"object", 1}});
        CHECK(j["reviewed"] == 1);
        // 1 misses frame 1: one track with a gap
        CHECK(j["tracks"] == nlohmann::json{{"total", 4}, {"with_gaps", 1}, {"divisions", 1}});

        // a raw write drops the index; the summary builds one and keeps nothing
        labels.volume(2);
        REQUIRE_FALSE(labels.tracks());
        j = labelStatistics(labels, 0, voxelUm);
        CHECK(j["tracks"] == nlohmann::json{{"total", 4}, {"with_gaps", 1}, {"divisions", 1}});
        CHECK_FALSE(labels.tracks());
    }
    SECTION("an empty volume and a frame outside it") {
        const nlohmann::json j = labelStatistics(LabelVolume{}, 0, voxelUm);
        CHECK(j["count"] == 0);
        CHECK(j["voxels"]["max"] == 0.0);
        CHECK(j["classes"].empty());

        const LabelVolume labels = sampleLabels();
        CHECK_THROWS_AS(labelStatistics(labels, 3, voxelUm), std::invalid_argument);
        CHECK_THROWS_AS(labelStatistics(labels, -1, voxelUm), std::invalid_argument);
    }
}
