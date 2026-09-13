// Track review: the per-frame centroid index of a tracked label volume, kept
// current from edit diffs (checked against a rescan after every kind of edit,
// including a stroke that touches a voxel twice and its undo), and the track
// summaries the table shows -- extent, gaps, path length in microns on
// anisotropic voxels, and lineage that survives edits without inventing
// relationships for ids that are gone.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/labels.hpp"
#include "core/tracks.hpp"

using namespace sirius;
using namespace sirius::app;
using Catch::Matchers::WithinAbs;

namespace {

    // A cube of `id` with its low corner at (z, y, x) and `side` voxels.
    void cube(LabelVolume& labels, Index t, std::uint32_t id, Index z, Index y, Index x, Index side) {
        std::uint32_t* v = labels.volume(t);
        for (Index iz = z; iz < z + side; ++iz)
            for (Index iy = y; iy < y + side; ++iy)
                for (Index ix = x; ix < x + side; ++ix) v[(iz * labels.y() + iy) * labels.x() + ix] = id;
    }

    // The same index, id for id and frame for frame.
    void requireSame(const TrackIndex& got, const TrackIndex& want) {
        REQUIRE(got.frames() == want.frames());
        REQUIRE(got.ids() == want.ids());
        for (std::uint32_t id : want.ids()) {
            const std::vector<TrackPoint> a = got.points(id), b = want.points(id);
            REQUIRE(a.size() == b.size());
            for (std::size_t k = 0; k < a.size(); ++k) {
                CHECK(a[k].t == b[k].t);
                CHECK(a[k].voxels == b[k].voxels);
                for (std::size_t d = 0; d < 3; ++d) CHECK_THAT(a[k].centroid[d], WithinAbs(b[k].centroid[d], 1e-9));
            }
        }
    }

    const std::array<double, 3> kIsotropic{1.0, 1.0, 1.0};

} // namespace

TEST_CASE("TrackIndex follows each id through time", "[app][tracks]") {
    LabelVolume labels(4, 4, 16, 16);
    // track 1 moves 2 voxels along x per frame; track 2 is missed in frame 2
    for (Index t = 0; t < 4; ++t) cube(labels, t, 1, 0, 0, 2 * t, 2);
    cube(labels, 0, 2, 2, 10, 10, 2);
    cube(labels, 1, 2, 2, 10, 10, 2);
    cube(labels, 3, 2, 2, 12, 10, 2);
    labels.setTracked(true);

    const TrackIndex index(labels);
    CHECK(index.frames() == 4);
    CHECK_FALSE(index.empty());
    CHECK(index.ids() == std::vector<std::uint32_t>{1, 2});

    const std::vector<TrackPoint> one = index.points(1);
    REQUIRE(one.size() == 4);
    for (Index t = 0; t < 4; ++t) {
        CHECK(one[static_cast<std::size_t>(t)].t == t);
        CHECK(one[static_cast<std::size_t>(t)].voxels == 8);
        CHECK_THAT(one[static_cast<std::size_t>(t)].centroid[0], WithinAbs(0.5, 1e-12));
        CHECK_THAT(one[static_cast<std::size_t>(t)].centroid[2], WithinAbs(2.0 * static_cast<double>(t) + 0.5, 1e-12));
    }

    CHECK(index.points(2).size() == 3);
    CHECK_FALSE(index.pointAt(2, 2).has_value());
    CHECK_FALSE(index.pointAt(2, 99).has_value());
    CHECK(index.points(7).empty());

    SECTION("nearestPoint sends the cursor to the closest frame the track is in") {
        REQUIRE(index.nearestPoint(2, 2).has_value());
        CHECK(index.nearestPoint(2, 2)->t == 1);    // tie between 1 and 3: the earlier
        CHECK(index.nearestPoint(2, 3)->t == 3);
        CHECK(index.nearestPoint(2, -5)->t == 0);   // clamped into the clip
        CHECK_FALSE(index.nearestPoint(7, 0).has_value());
    }

    SECTION("an empty volume indexes to nothing") {
        const TrackIndex none(LabelVolume{});
        CHECK(none.frames() == 0);
        CHECK(none.empty());
        CHECK(none.ids().empty());
        CHECK_FALSE(none.nearestPoint(1, 0).has_value());
    }
}

TEST_CASE("TrackIndex stays equal to a rescan through every edit and its undo", "[app][tracks]") {
    LabelVolume labels(3, 6, 20, 20);
    for (Index t = 0; t < 3; ++t) {
        cube(labels, t, 1, 1, 2, 2 + t, 4);
        cube(labels, t, 2, 1, 10, 10, 5);
        cube(labels, t, 3, 3, 14, 2, 3);
    }
    labels.setTracked(true);
    TrackIndex index(labels);
    std::vector<LabelDiff> done;
    const auto edit = [&](LabelDiff diff) {
        index.apply(diff);
        requireSame(index, TrackIndex(labels));
        done.push_back(std::move(diff));
    };

    SECTION("paint, erase, fill, merge, split and remove") {
        edit(labels.paint(1, 2, 4, 4, 3.0, 1, 4));               // a new id over existing ones
        edit(labels.paint(1, 2, 12, 12, 2.0, 0, 0));             // erase inside track 2
        edit(labels.paint(0, 2, 4, 4, 2.0, 1, 1, 2));            // only over track 2's voxels: a no-op here
        edit(labels.fill(1, 0, 0, 0, 7));                         // the background around everything
        edit(labels.merge(2, {2, 3}));
        edit(labels.split(0, 2, {2, 11, 11}, {4, 13, 13}));
        edit(labels.remove(2, 1));

        // undo everything, newest first, the way the history does
        for (auto it = done.rbegin(); it != done.rend(); ++it) {
            labels.apply(*it, false);
            index.apply(*it, false);
            requireSame(index, TrackIndex(labels));
        }
        // and redo
        for (const LabelDiff& d : done) {
            labels.apply(d, true);
            index.apply(d, true);
        }
        requireSame(index, TrackIndex(labels));
    }

    SECTION("a stroke that crosses its own path is counted once") {
        // the workbench concatenates each mouse move's diff into one
        LabelDiff stroke;
        // painting and erasing over the same voxels, so indices repeat
        const std::array<std::pair<Index, std::uint32_t>, 4> moves{{{4, 9}, {5, 0}, {6, 9}, {4, 9}}};
        for (const auto& [x, label] : moves) {
            const LabelDiff move = labels.paint(1, 2, 4, x, 2.5, 1, label);
            stroke.t = move.t;
            stroke.indices.insert(stroke.indices.end(), move.indices.begin(), move.indices.end());
            stroke.before.insert(stroke.before.end(), move.before.begin(), move.before.end());
            stroke.after.insert(stroke.after.end(), move.after.begin(), move.after.end());
        }
        std::vector<Index> sorted = stroke.indices;
        std::sort(sorted.begin(), sorted.end());
        REQUIRE(std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end());
        index.apply(stroke);
        requireSame(index, TrackIndex(labels));
        labels.apply(stroke, false);
        index.apply(stroke, false);
        requireSame(index, TrackIndex(labels));
        CHECK(index.points(9).empty());
    }

    SECTION("rescanFrame recounts writes that bypass the edits") {
        cube(labels, 2, 5, 0, 0, 15, 3);
        index.rescanFrame(labels, 2);
        requireSame(index, TrackIndex(labels));
        CHECK_THROWS(index.rescanFrame(labels, 3));
        CHECK_THROWS(index.rescanFrame(LabelVolume(3, 6, 20, 21), 0));
    }

    SECTION("malformed diffs are refused") {
        LabelDiff bad;
        bad.t = 0;
        bad.indices = {0, 1};
        bad.before = {0};
        bad.after = {1, 1};
        CHECK_THROWS(index.apply(bad));
        bad.before = {0, 0};
        bad.t = 3;
        CHECK_THROWS(index.apply(bad));
    }
}

TEST_CASE("summarizeTracks reports extent, gaps and motion in microns", "[app][tracks]") {
    LabelVolume labels(5, 8, 16, 16);
    // track 1: one voxel step along z per frame, present in 0, 1, 3 (gap at 2)
    cube(labels, 0, 1, 0, 0, 0, 2);
    cube(labels, 1, 1, 1, 0, 0, 2);
    cube(labels, 3, 1, 3, 0, 0, 2);
    // track 2: one voxel step along x per frame, frames 1..4
    for (Index t = 1; t < 5; ++t) cube(labels, t, 2, 4, 8, t, 2);
    // track 3: a single frame
    cube(labels, 4, 3, 5, 12, 12, 3);

    const TrackIndex index(labels);
    const std::array<double, 3> voxelUm{0.75, 0.15, 0.15};   // anisotropic, a real case
    const std::vector<TrackSummary> rows = summarizeTracks(index, {}, voxelUm);
    REQUIRE(rows.size() == 3);

    const TrackSummary& a = rows[0];
    CHECK(a.id == 1);
    CHECK(a.first == 0);
    CHECK(a.last == 3);
    CHECK(a.frames == 3);
    CHECK(a.span() == 4);
    CHECK(a.gaps == 1);
    CHECK_THAT(a.pathUm, WithinAbs(3.0 * 0.75, 1e-9));   // 1 + 2 z steps
    CHECK_THAT(a.netUm, WithinAbs(3.0 * 0.75, 1e-9));
    CHECK_THAT(a.umPerFrame, WithinAbs(0.75, 1e-9));
    CHECK_THAT(a.meanVoxels, WithinAbs(8.0, 1e-12));

    const TrackSummary& b = rows[1];
    CHECK(b.first == 1);
    CHECK(b.last == 4);
    CHECK(b.gaps == 0);
    // the same voxel step is five times shorter in plane than along z
    CHECK_THAT(b.pathUm, WithinAbs(3.0 * 0.15, 1e-9));
    CHECK_THAT(b.umPerFrame, WithinAbs(0.15, 1e-9));

    const TrackSummary& c = rows[2];
    CHECK(c.frames == 1);
    CHECK(c.pathUm == 0.0);
    CHECK(c.umPerFrame == 0.0);
    CHECK(c.meanVoxels == 27.0);

    CHECK(summarizeTracks(TrackIndex(LabelVolume{}), {}, kIsotropic).empty());
}

TEST_CASE("summarizeTracks attaches lineage only between tracks that exist", "[app][tracks]") {
    LabelVolume labels(3, 2, 16, 16);
    cube(labels, 0, 1, 0, 6, 6, 2);    // mother
    cube(labels, 1, 1, 0, 6, 6, 2);
    cube(labels, 2, 2, 0, 2, 2, 2);    // daughters
    cube(labels, 2, 3, 0, 10, 10, 2);
    cube(labels, 2, 4, 0, 13, 0, 2);   // unrelated
    const TrackIndex index(labels);

    SECTION("a division: one parent, two children") {
        const std::vector<TrackSummary> rows = summarizeTracks(index, {{2, 1}, {3, 1}}, kIsotropic);
        REQUIRE(rows.size() == 4);
        CHECK(rows[0].children == std::vector<std::uint32_t>{2, 3});
        CHECK(rows[0].divides());
        CHECK(rows[0].parent == 0);
        CHECK(rows[1].parent == 1);
        CHECK(rows[2].parent == 1);
        CHECK(rows[3].parent == 0);
        CHECK(rows[3].children.empty());
        CHECK(countDivisions(rows) == 1);
    }

    SECTION("ids that are gone, self-parents and cycles describe nothing and hang nothing") {
        const Lineage lineage{{2, 1}, {3, 99}, {4, 4}, {98, 1}, {1, 2}};
        const std::vector<TrackSummary> rows = summarizeTracks(index, lineage, kIsotropic);
        REQUIRE(rows.size() == 4);
        CHECK(rows[0].children == std::vector<std::uint32_t>{2});   // 98 is not in the labels
        CHECK_FALSE(rows[0].divides());
        CHECK(rows[0].parent == 2);                                 // the 1 <-> 2 cycle is reported as given
        CHECK(rows[2].parent == 0);                                 // parent 99 is not in the labels
        CHECK(rows[3].parent == 0);                                 // its own parent
        CHECK(countDivisions(rows) == 0);
    }

    SECTION("deleting a daughter leaves the mother with one child, not a division") {
        LabelVolume edited(labels);
        TrackIndex live(edited);
        live.apply(edited.remove(2, 3));
        const std::vector<TrackSummary> rows = summarizeTracks(live, {{2, 1}, {3, 1}}, kIsotropic);
        REQUIRE(rows.size() == 3);
        CHECK(rows[0].children == std::vector<std::uint32_t>{2});
        CHECK(countDivisions(rows) == 0);
    }
}
