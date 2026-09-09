// Tracking: the optimal assignment solver, linking objects across frames
// (including the crossing case that defeats nearest-neighbour matching), gap
// closing, and the scale-space blob seeding that the classical segmentation
// uses for objects of different sizes.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "core/labels.hpp"
#include "core/tracking.hpp"

using namespace sirius;
using namespace sirius::app;

namespace {
    double totalCost(const std::vector<double>& cost, int cols, const std::vector<int>& assignment) {
        double sum = 0.0;
        for (std::size_t r = 0; r < assignment.size(); ++r)
            if (assignment[r] >= 0) sum += cost[r * static_cast<std::size_t>(cols) + static_cast<std::size_t>(assignment[r])];
        return sum;
    }

    TrackObject object(std::uint32_t label, double z, double y, double x, Index voxels = 8) {
        TrackObject o;
        o.label = label;
        o.centroid = {z, y, x};
        o.voxels = voxels;
        return o;
    }
} // namespace

TEST_CASE("solveAssignment finds the cheapest matching, not the greedy one", "[app][tracking]") {
    SECTION("greedy would take the tempting first row and pay for it later") {
        // row 0 prefers column 0 by a hair, but row 1 can only use column 0
        const std::vector<double> cost{1.0, 2.0, 1.5, 100.0};
        const std::vector<int> got = solveAssignment(cost, 2, 2);
        CHECK(got[0] == 1);
        CHECK(got[1] == 0);
        CHECK(totalCost(cost, 2, got) == 3.5);   // greedy would pay 1.0 + 100.0
    }
    SECTION("more rows than columns leaves the dearest rows unmatched") {
        const std::vector<double> cost{1.0, 5.0, 9.0};   // 3 rows, 1 column
        const std::vector<int> got = solveAssignment(cost, 3, 1);
        CHECK(got.size() == 3);
        CHECK(got[0] == 0);
        CHECK(got[1] == -1);
        CHECK(got[2] == -1);
    }
    SECTION("more columns than rows matches every row") {
        const std::vector<double> cost{4.0, 1.0, 7.0};   // 1 row, 3 columns
        const std::vector<int> got = solveAssignment(cost, 1, 3);
        CHECK(got[0] == 1);
    }
    SECTION("forbidden pairs are never returned") {
        const std::vector<double> cost{kNoAssignment, 2.0, 3.0, kNoAssignment};
        const std::vector<int> got = solveAssignment(cost, 2, 2);
        CHECK(got[0] == 1);
        CHECK(got[1] == 0);
        // a row with nothing feasible stays unmatched
        const std::vector<double> blocked{kNoAssignment, kNoAssignment, 1.0, 2.0};
        const std::vector<int> half = solveAssignment(blocked, 2, 2);
        CHECK(half[0] == -1);
        CHECK(half[1] >= 0);
    }
    SECTION("degenerate shapes are answered, not crashed into") {
        CHECK(solveAssignment({}, 0, 0).empty());
        CHECK(solveAssignment({}, 0, 3).empty());
        CHECK(solveAssignment({}, 3, 0).size() == 3);
        CHECK_THROWS(solveAssignment({1.0}, 2, 2));
    }
}

TEST_CASE("linkTracks follows objects and keeps identities when paths cross", "[app][tracking]") {
    const std::array<double, 3> voxel{1.0, 1.0, 1.0};   // 1 µm voxels: distance == voxels
    TrackOptions options;
    options.maxDistanceUm = 5.0;
    options.overlapWeight = 0.0;
    options.maxGap = 0;
    options.minLength = 1;

    SECTION("a cheap pair does not steal the object another one needs") {
        // taking the nearest pair first (B to the object at x = 5, cost 1)
        // strands A with a cost of 7; matching the frame as a whole pays 6
        std::vector<std::vector<TrackObject>> frames{
            {object(1, 0, 0, 0), object(2, 0, 0, 6)},
            {object(1, 0, 0, 5), object(2, 0, 0, 7)},
        };
        options.maxDistanceUm = 8.0;
        const TrackResult r = linkTracks(frames, {}, voxel, options);
        REQUIRE(r.tracks.size() == 2);
        for (const Track& t : r.tracks) CHECK(t.length() == 2);
        // each track keeps its own label: 1 -> 1 and 2 -> 2
        for (const Track& t : r.tracks) CHECK(t.points.front().second == t.points.back().second);
    }

    SECTION("an object that moves further than the gate starts a new track") {
        std::vector<std::vector<TrackObject>> frames{{object(1, 0, 0, 0)}, {object(1, 0, 0, 20)}};
        const TrackResult r = linkTracks(frames, {}, voxel, options);
        CHECK(r.tracks.size() == 2);
        CHECK(r.links == 0);
    }

    SECTION("a missed frame is bridged only when gap closing is on") {
        std::vector<std::vector<TrackObject>> frames{{object(1, 0, 0, 0)}, {}, {object(1, 0, 0, 2)}};
        TrackResult without = linkTracks(frames, {}, voxel, options);
        CHECK(without.tracks.size() == 2);
        CHECK(without.gapsClosed == 0);

        options.maxGap = 1;
        TrackResult with = linkTracks(frames, {}, voxel, options);
        CHECK(with.tracks.size() == 1);
        CHECK(with.gapsClosed == 1);
        CHECK(with.tracks[0].length() == 2);
        CHECK(with.tracks[0].first() == 0);
        CHECK(with.tracks[0].last() == 2);
    }

    SECTION("short tracks are dropped and the rest numbered in order of appearance") {
        options.minLength = 2;
        std::vector<std::vector<TrackObject>> frames{
            {object(1, 0, 0, 0)},
            {object(1, 0, 0, 1), object(9, 0, 0, 40)},   // 9 appears once, far away
        };
        const TrackResult r = linkTracks(frames, {}, voxel, options);
        REQUIRE(r.tracks.size() == 1);
        CHECK(r.tracks[0].id == 1);
        CHECK(r.tracks[0].length() == 2);
    }

    SECTION("overlap breaks a tie that distance alone cannot") {
        // both candidates sit the same distance away; only the shared voxels
        // say which one is the same object
        std::vector<std::vector<TrackObject>> frames{{object(1, 0, 0, 4, 10)},
                                                     {object(1, 0, 0, 2, 10), object(2, 0, 0, 6, 10)}};
        std::vector<std::vector<Index>> overlap{{0, 8}};   // row 0: no overlap with 1, strong with 2
        options.overlapWeight = 0.9;
        const TrackResult r = linkTracks(frames, overlap, voxel, options);
        REQUIRE(r.links == 1);
        const Track& first = r.tracks.front();
        REQUIRE(first.length() == 2);
        CHECK(first.points.back().second == 2);
    }
}

TEST_CASE("objectsOfFrame and overlapBetween read a label volume", "[app][tracking]") {
    LabelVolume labels(2, 1, 4, 4);
    std::uint32_t* a = labels.volume(0);
    std::uint32_t* b = labels.volume(1);
    a[0 * 4 + 0] = 1;
    a[0 * 4 + 1] = 1;
    a[3 * 4 + 3] = 2;
    b[0 * 4 + 1] = 5;   // overlaps label 1 in one voxel
    b[0 * 4 + 2] = 5;

    const std::vector<TrackObject> first = objectsOfFrame(labels, 0);
    REQUIRE(first.size() == 2);
    CHECK(first[0].label == 1);
    CHECK(first[0].voxels == 2);
    CHECK(first[0].centroid[2] == 0.5);   // x centroid of columns 0 and 1
    CHECK(first[1].label == 2);

    const std::vector<TrackObject> second = objectsOfFrame(labels, 1);
    REQUIRE(second.size() == 1);
    const std::vector<Index> counts = overlapBetween(labels, 0, first, second);
    REQUIRE(counts.size() == 2);
    CHECK(counts[0] == 1);   // label 1 shares one voxel with label 5
    CHECK(counts[1] == 0);
}

TEST_CASE("logBlobSeeds puts one seed in each blob whatever its size", "[app][tracking][labels]") {
    // two balls of very different radii, far apart: the distance map would
    // need one setting for each, the scale-space detector finds both at once
    const Index z = 13, y = 21, x = 45;
    const Index n = z * y * x;
    std::vector<float> values(static_cast<std::size_t>(n), 0.0f);
    std::vector<std::uint8_t> mask(static_cast<std::size_t>(n), 0);
    auto ball = [&](double cz, double cy, double cx, double r) {
        for (Index iz = 0; iz < z; ++iz)
            for (Index iy = 0; iy < y; ++iy)
                for (Index ix = 0; ix < x; ++ix) {
                    const double d2 = (iz - cz) * (iz - cz) + (iy - cy) * (iy - cy) + (ix - cx) * (ix - cx);
                    if (d2 <= r * r) {
                        const std::size_t i = static_cast<std::size_t>((iz * y + iy) * x + ix);
                        values[i] = 1000.0f;
                        mask[i] = 1;
                    }
                }
    };
    ball(6, 10, 9, 3.0);
    ball(6, 10, 32, 7.0);
    std::vector<std::uint32_t> seeds(static_cast<std::size_t>(n), 0u);
    const std::uint32_t count = logBlobSeeds(values.data(), mask.data(), z, y, x, 1.0, 1.5, 5.0, 5, seeds.data());
    CHECK(count == 2);
    // one seed inside each ball, and none outside the mask
    Index inSmall = 0, inLarge = 0, outside = 0;
    for (Index i = 0; i < n; ++i) {
        if (!seeds[static_cast<std::size_t>(i)]) continue;
        if (!mask[static_cast<std::size_t>(i)]) ++outside;
        const Index ix = i % x;
        if (ix < 20) ++inSmall;
        else ++inLarge;
    }
    CHECK(outside == 0);
    CHECK(inSmall == 1);
    CHECK(inLarge == 1);
}

TEST_CASE("Gap closing follows a chain of missed frames, not just the first link",
          "[app][tracking]") {
    // One stationary object detected in frames 0, 2 and 4 and missed in 1 and
    // 3. Both gaps are within the gate, so this is one object seen five frames
    // running, not two tracks. The end-to-start assignment finds both links;
    // applying them has to follow the first merge, or the second link is
    // dropped because its "from" track was the one just consumed.
    const std::array<double, 3> voxel{1.0, 1.0, 1.0};
    TrackOptions options;
    options.maxDistanceUm = 10.0;
    options.overlapWeight = 0.0;
    options.minLength = 1;
    options.maxGap = 1;   // a gap of one missed frame, twice over
    const std::vector<std::vector<TrackObject>> frames{
        {object(1, 0, 0, 0)}, {}, {object(1, 0, 0, 0)}, {}, {object(1, 0, 0, 0)}};

    const TrackResult r = linkTracks(frames, {}, voxel, options);
    REQUIRE(r.tracks.size() == 1);
    CHECK(r.gapsClosed == 2);
    CHECK(r.tracks[0].length() == 3);
    CHECK(r.tracks[0].first() == 0);
    CHECK(r.tracks[0].last() == 4);

    SECTION("a chain of three gaps closes as one track too") {
        const std::vector<std::vector<TrackObject>> six{{object(1, 0, 0, 0)}, {}, {object(1, 0, 0, 0)}, {}, {object(1, 0, 0, 0)}, {}, {object(1, 0, 0, 0)}};
        const TrackResult chain = linkTracks(six, {}, voxel, options);
        REQUIRE(chain.tracks.size() == 1);
        CHECK(chain.gapsClosed == 3);
        CHECK(chain.tracks[0].length() == 4);
    }
}
