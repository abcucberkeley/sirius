// Label volumes of the workbench: connected components and watershed on
// synthetic objects, the editing operations and their reversible diffs,
// statistics and review flags.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/labels.hpp"
#include "core/ops/builtin.hpp"

using namespace sirius;
using namespace sirius::app;
using Catch::Matchers::WithinAbs;

namespace {

    struct Mask {
        Index z, y, x;
        std::vector<std::uint8_t> v;
        Mask(Index nz, Index ny, Index nx) : z(nz), y(ny), x(nx), v(static_cast<std::size_t>(nz * ny * nx), 0) {}
        Index at(Index iz, Index iy, Index ix) const { return (iz * y + iy) * x + ix; }
        void sphere(Index cz, Index cy, Index cx, double r) {
            for (Index iz = 0; iz < z; ++iz)
                for (Index iy = 0; iy < y; ++iy)
                    for (Index ix = 0; ix < x; ++ix) {
                        const double d2 = static_cast<double>((iz - cz) * (iz - cz) + (iy - cy) * (iy - cy) + (ix - cx) * (ix - cx));
                        if (d2 <= r * r) v[static_cast<std::size_t>(at(iz, iy, ix))] = 1;
                    }
        }
    };

    std::set<std::uint32_t> distinct(const std::vector<std::uint32_t>& labels) {
        std::set<std::uint32_t> s(labels.begin(), labels.end());
        s.erase(0);
        return s;
    }

} // namespace

TEST_CASE("connectedComponents labels 6-connected blobs densely", "[app][labels]") {
    Mask m(8, 16, 16);
    m.sphere(3, 4, 4, 2.0);
    m.sphere(3, 11, 11, 2.5);
    // two voxels touching only diagonally are separate components in 6-connectivity
    m.v[static_cast<std::size_t>(m.at(7, 0, 0))] = 1;
    m.v[static_cast<std::size_t>(m.at(7, 1, 1))] = 1;
    std::vector<std::uint32_t> out(m.v.size());
    const std::uint32_t n = connectedComponents(m.v.data(), m.z, m.y, m.x, out.data());
    CHECK(n == 4);
    CHECK(distinct(out) == std::set<std::uint32_t>{1, 2, 3, 4});
    // background stays 0, every masked voxel is labelled, one label per blob
    for (std::size_t i = 0; i < out.size(); ++i) CHECK((out[i] != 0) == (m.v[i] != 0));
    CHECK(out[static_cast<std::size_t>(m.at(3, 4, 4))] == out[static_cast<std::size_t>(m.at(4, 4, 5))]);
    CHECK(out[static_cast<std::size_t>(m.at(3, 4, 4))] != out[static_cast<std::size_t>(m.at(3, 11, 11))]);
    CHECK(out[static_cast<std::size_t>(m.at(7, 0, 0))] != out[static_cast<std::size_t>(m.at(7, 1, 1))]);

    SECTION("a U shape closed late in the raster scan is still one component") {
        Mask u(1, 5, 5);
        for (Index iy = 0; iy < 5; ++iy) {
            u.v[static_cast<std::size_t>(u.at(0, iy, 0))] = 1;
            u.v[static_cast<std::size_t>(u.at(0, iy, 4))] = 1;
        }
        for (Index ix = 0; ix < 5; ++ix) u.v[static_cast<std::size_t>(u.at(0, 4, ix))] = 1;
        std::vector<std::uint32_t> lab(u.v.size());
        CHECK(connectedComponents(u.v.data(), 1, 5, 5, lab.data()) == 1);
    }
    SECTION("removeSmall drops the specks and renumbers") {
        std::vector<std::uint32_t> lab = out;
        const std::uint32_t kept = removeSmall(lab.data(), static_cast<Index>(lab.size()), 10);
        CHECK(kept == 2);
        CHECK(distinct(lab) == std::set<std::uint32_t>{1, 2});
        CHECK(lab[static_cast<std::size_t>(m.at(7, 0, 0))] == 0);
    }
}

TEST_CASE("distanceTransform is the exact Euclidean distance to the background", "[app][labels]") {
    Mask m(5, 9, 9);
    m.sphere(2, 4, 4, 3.0);
    std::vector<float> d(m.v.size());
    distanceTransform(m.v.data(), m.z, m.y, m.x, d.data());
    // brute force reference
    for (Index iz = 0; iz < m.z; ++iz)
        for (Index iy = 0; iy < m.y; ++iy)
            for (Index ix = 0; ix < m.x; ++ix) {
                double best = 1e300;
                for (Index jz = 0; jz < m.z; ++jz)
                    for (Index jy = 0; jy < m.y; ++jy)
                        for (Index jx = 0; jx < m.x; ++jx)
                            if (!m.v[static_cast<std::size_t>(m.at(jz, jy, jx))])
                                best = std::min(best, std::sqrt(static_cast<double>((iz - jz) * (iz - jz) + (iy - jy) * (iy - jy) + (ix - jx) * (ix - jx))));
                const float got = d[static_cast<std::size_t>(m.at(iz, iy, ix))];
                if (!m.v[static_cast<std::size_t>(m.at(iz, iy, ix))]) CHECK(got == 0.0f);
                else CHECK_THAT(got, WithinAbs(best, 1e-4));
            }
    SECTION("a mask without background reads the volume's extent") {
        std::vector<std::uint8_t> full(8, 1);
        std::vector<float> dd(8);
        distanceTransform(full.data(), 2, 2, 2, dd.data());
        for (float v : dd) CHECK(v == 2.0f);
    }
}

TEST_CASE("watershed splits two touching spheres from distance seeds", "[app][labels]") {
    Mask m(9, 15, 25);
    m.sphere(4, 7, 7, 4.0);
    m.sphere(4, 7, 15, 4.0);   // centres 8 apart, radii 4: they touch in a neck
    std::vector<std::uint32_t> cc(m.v.size());
    REQUIRE(connectedComponents(m.v.data(), m.z, m.y, m.x, cc.data()) == 1);

    std::vector<std::uint32_t> seeds(m.v.size());
    const std::uint32_t nSeeds = distanceSeeds(m.v.data(), m.z, m.y, m.x, 5.0, seeds.data());
    REQUIRE(nSeeds == 2);
    std::vector<float> dist(m.v.size());
    distanceTransform(m.v.data(), m.z, m.y, m.x, dist.data());
    for (float& v : dist) v = -v;   // ridges high, basins low

    std::vector<std::uint32_t> labels = seeds;
    watershed(dist.data(), m.v.data(), m.z, m.y, m.x, labels.data());
    CHECK(distinct(labels) == std::set<std::uint32_t>{1, 2});
    Index n1 = 0, n2 = 0;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        CHECK((labels[i] != 0) == (m.v[i] != 0));
        if (labels[i] == 1) ++n1;
        if (labels[i] == 2) ++n2;
    }
    // the two halves are the same size and each centre keeps its own label
    CHECK(std::abs(n1 - n2) <= 12);
    CHECK(labels[static_cast<std::size_t>(m.at(4, 7, 7))] != labels[static_cast<std::size_t>(m.at(4, 7, 15))]);
    CHECK(labels[static_cast<std::size_t>(m.at(4, 7, 7))] == labels[static_cast<std::size_t>(m.at(4, 7, 5))]);
    CHECK(labels[static_cast<std::size_t>(m.at(4, 7, 15))] == labels[static_cast<std::size_t>(m.at(4, 7, 17))]);
}

TEST_CASE("reconstructByDilation grows the marker up to the mask", "[app][labels]") {
    // one ridge the marker reaches and one it does not: reconstruction fills
    // the first to the mask's height and leaves the second alone
    const Index z = 1, y = 1, x = 9;
    std::vector<float> mask{1.f, 5.f, 1.f, 0.f, 1.f, 7.f, 1.f, 0.f, 1.f};
    std::vector<float> marker{0.f, 3.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    reconstructByDilation(marker.data(), mask.data(), z, y, x);
    CHECK(marker[1] == 3.0f);            // capped by the marker, not the mask
    CHECK(marker[0] == 1.0f);            // spread sideways, capped by the mask
    CHECK(marker[2] == 1.0f);
    CHECK(marker[3] == 0.0f);            // the zero valley blocks it
    CHECK(marker[5] == 0.0f);            // the far ridge was never seeded
    // a marker equal to the mask is a fixed point
    std::vector<float> same = mask;
    reconstructByDilation(same.data(), mask.data(), z, y, x);
    CHECK(same == mask);
}

TEST_CASE("hMaximaSeeds keeps deep peaks and swallows shallow ones", "[app][labels]") {
    // a ridge with a deep peak (10) and a shallow shoulder (6) on one object
    const Index z = 1, y = 1, x = 11;
    const std::vector<float> values{0.f, 4.f, 10.f, 4.f, 5.f, 6.f, 5.f, 2.f, 9.f, 3.f, 0.f};
    const std::vector<std::uint8_t> mask(static_cast<std::size_t>(x), 1);
    std::vector<std::uint32_t> out(static_cast<std::size_t>(x));

    // a small depth keeps every local maximum apart
    const std::uint32_t shallow = hMaximaSeeds(values.data(), mask.data(), z, y, x, 0.5, out.data());
    CHECK(shallow == 3);

    // deeper than the shoulder stands above its surroundings: it merges away
    const std::uint32_t deep = hMaximaSeeds(values.data(), mask.data(), z, y, x, 2.0, out.data());
    CHECK(deep == 2);
    CHECK(out[2] != 0);                  // the 10 peak survives
    CHECK(out[8] != 0);                  // so does the 9
    CHECK(out[5] == 0);                  // the shoulder does not

    // outside the mask nothing is seeded
    std::vector<std::uint8_t> none(static_cast<std::size_t>(x), 0);
    CHECK(hMaximaSeeds(values.data(), none.data(), z, y, x, 1.0, out.data()) == 0);
}

TEST_CASE("h-maxima seeds keep a lumpy object whole where distance maxima split it", "[app][labels]") {
    // a capsule: two spheres so close they are one object, with a waist too
    // shallow to be a real boundary
    const Index z = 9, y = 17, x = 29;
    const Index n = z * y * x;
    std::vector<std::uint8_t> mask(static_cast<std::size_t>(n), 0);
    auto sphere = [&](double cz, double cy, double cx, double r) {
        for (Index iz = 0; iz < z; ++iz)
            for (Index iy = 0; iy < y; ++iy)
                for (Index ix = 0; ix < x; ++ix) {
                    const double d2 = (iz - cz) * (iz - cz) + (iy - cy) * (iy - cy) + (ix - cx) * (ix - cx);
                    if (d2 <= r * r) mask[static_cast<std::size_t>((iz * y + iy) * x + ix)] = 1;
                }
    };
    sphere(4, 8, 11, 6.0);
    sphere(4, 8, 17, 6.0);   // centres 6 apart, radii 6: a barely waisted capsule
    std::vector<float> dist(static_cast<std::size_t>(n));
    distanceTransform(mask.data(), z, y, x, dist.data());
    std::vector<std::uint32_t> seeds(static_cast<std::size_t>(n));
    const std::uint32_t byDistance = distanceSeeds(mask.data(), z, y, x, 3.0, seeds.data());
    const std::uint32_t byDepth = hMaximaSeeds(dist.data(), mask.data(), z, y, x, 2.0, seeds.data());
    CHECK(byDistance >= 2);   // every peak of the distance map becomes a seed
    CHECK(byDepth == 1);      // the waist is too shallow to be its own object
}

TEST_CASE("LabelVolume edits produce reversible diffs", "[app][labels]") {
    LabelVolume vol(2, 6, 12, 12);
    CHECK(vol.maxLabel() == 0);
    CHECK(vol.volumeSize() == 6 * 12 * 12);

    const LabelDiff brush = vol.paint(1, 3, 6, 6, 2.0, 1, 7);
    CHECK_FALSE(brush.empty());
    CHECK(vol.maxLabel() == 7);
    CHECK(vol.at(1, 3, 6, 6) == 7);
    CHECK(vol.at(1, 3, 6, 8) == 7);   // on the ring
    CHECK(vol.at(1, 3, 6, 9) == 0);   // outside the radius
    CHECK(vol.at(1, 4, 6, 6) == 7);   // one plane up, on the ellipsoid's axis
    CHECK(vol.at(1, 4, 6, 8) == 0);   // one plane up, ring shrinks
    CHECK(vol.at(1, 5, 6, 6) == 0);   // beyond zRadius
    CHECK(vol.at(0, 3, 6, 6) == 0);   // other time point untouched
    for (std::size_t k = 0; k < brush.indices.size(); ++k) {
        CHECK(brush.before[k] == 0);
        CHECK(brush.after[k] == 7);
    }

    SECTION("undo through apply(forward = false) restores every voxel") {
        vol.apply(brush, false);
        for (Index i = 0; i < vol.volumeSize(); ++i) CHECK(vol.volume(1)[i] == 0);
        vol.apply(brush, true);
        CHECK(vol.at(1, 3, 6, 6) == 7);
    }
    SECTION("painting the same label again changes nothing; onlyLabel restricts") {
        CHECK(vol.paint(1, 3, 6, 6, 2.0, 1, 7).empty());
        const LabelDiff restricted = vol.paint(1, 3, 6, 6, 4.0, 0, 9, 7);
        for (std::uint32_t b : restricted.before) CHECK(b == 7);
        CHECK(vol.at(1, 3, 6, 6) == 9);
        CHECK(vol.at(1, 3, 6, 9) == 0);   // was background, left alone
    }
    SECTION("erase is painting 0") {
        const LabelDiff erase = vol.paint(1, 3, 6, 6, 1.0, 0, 0);
        CHECK(vol.at(1, 3, 6, 6) == 0);
        CHECK(vol.at(1, 3, 6, 8) == 7);
        vol.apply(erase, false);
        CHECK(vol.at(1, 3, 6, 6) == 7);
    }
    SECTION("fill recolours the connected region under the seed") {
        vol.paint(1, 0, 1, 1, 1.0, 0, 3);   // a separate small object
        const LabelDiff filled = vol.fill(1, 3, 6, 6, 4);
        CHECK(filled.indices.size() == brush.indices.size());
        CHECK(vol.at(1, 3, 6, 6) == 4);
        CHECK(vol.at(1, 0, 1, 1) == 3);
        CHECK(vol.fill(1, 3, 6, 6, 4).empty());
        CHECK_THROWS_AS(vol.fill(1, 99, 0, 0, 4), std::out_of_range);
    }
    SECTION("merge keeps the smallest id, remove clears") {
        vol.paint(1, 0, 1, 1, 1.0, 0, 3);
        vol.paint(1, 0, 10, 10, 1.0, 0, 12);
        const LabelDiff merged = vol.merge(1, {12, 7, 3});
        CHECK(vol.at(1, 3, 6, 6) == 3);
        CHECK(vol.at(1, 0, 10, 10) == 3);
        CHECK(vol.at(1, 0, 1, 1) == 3);
        for (std::uint32_t a : merged.after) CHECK(a == 3);
        vol.apply(merged, false);
        CHECK(vol.at(1, 3, 6, 6) == 7);
        const LabelDiff removed = vol.remove(1, 7);
        CHECK(removed.indices.size() == brush.indices.size());
        CHECK(vol.at(1, 3, 6, 6) == 0);
        CHECK(vol.remove(1, 0).empty());
    }
    SECTION("split separates a dumbbell at its neck") {
        LabelVolume w(1, 9, 15, 25);
        for (Index z = 0; z < 9; ++z)
            for (Index y = 0; y < 15; ++y)
                for (Index x = 0; x < 25; ++x) {
                    const bool a = (z - 4) * (z - 4) + (y - 7) * (y - 7) + (x - 7) * (x - 7) <= 16;
                    const bool b = (z - 4) * (z - 4) + (y - 7) * (y - 7) + (x - 15) * (x - 15) <= 16;
                    if (a || b) w.volume(0)[(z * 15 + y) * 25 + x] = 5;
                }
        w.recomputeStats(0);
        REQUIRE(w.maxLabel() == 5);
        const LabelDiff split = w.split(0, 5, {4, 7, 7}, {4, 7, 15});
        CHECK_FALSE(split.empty());
        CHECK(w.maxLabel() == 6);
        CHECK(w.at(0, 4, 7, 7) == 5);
        CHECK(w.at(0, 4, 7, 15) == 6);
        CHECK(w.at(0, 4, 7, 17) == 6);
        CHECK(w.at(0, 4, 7, 5) == 5);
        for (std::uint32_t b : split.before) CHECK(b == 5);
        w.apply(split, false);
        CHECK(w.at(0, 4, 7, 15) == 5);
        CHECK_THROWS_AS(w.split(0, 5, {0, 0, 0}, {4, 7, 15}), std::invalid_argument);
    }
    SECTION("clone is independent") {
        auto c = vol.clone();
        c->paint(1, 3, 6, 6, 3.0, 2, 0);
        CHECK(vol.at(1, 3, 6, 6) == 7);
        CHECK(c->at(1, 3, 6, 6) == 0);
        CHECK(c->t() == 2);
    }
}

TEST_CASE("LabelVolume statistics and review flags", "[app][labels]") {
    LabelVolume vol(1, 5, 20, 20);
    vol.paint(0, 2, 10, 10, 3.0, 1, 1);   // a healthy object
    vol.paint(0, 2, 3, 3, 1.0, 0, 2);     // small
    vol.paint(0, 2, 10, 0, 1.5, 0, 3);    // touches the x border
    vol.paint(0, 0, 16, 16, 0.0, 0, 4);   // single voxel on the z border
    std::vector<float> prob(static_cast<std::size_t>(vol.volumeSize()), 0.9f);
    // low confidence under object 2
    for (Index i = 0; i < vol.volumeSize(); ++i)
        if (vol.volume(0)[i] == 2) prob[static_cast<std::size_t>(i)] = 0.3f;

    vol.recomputeStats(0, prob.data());
    REQUIRE(vol.stats().size() == 4);
    const LabelStats* one = vol.statsOf(1);
    REQUIRE(one);
    CHECK(one->voxels > 30);
    CHECK_THAT(one->confidence, WithinAbs(0.9, 1e-5));
    CHECK(one->bbox == std::array<Index, 6>{1, 4, 7, 14, 7, 14});
    CHECK_FALSE(one->touchesBorder);
    const LabelStats* two = vol.statsOf(2);
    REQUIRE(two);
    CHECK_THAT(two->confidence, WithinAbs(0.3, 1e-5));
    CHECK(vol.statsOf(3)->touchesBorder);
    CHECK(vol.statsOf(4)->touchesBorder);
    CHECK(vol.statsOf(4)->voxels == 1);
    CHECK(vol.statsOf(99) == nullptr);

    LabelFlagRules rules;
    rules.minVoxels = 3;
    vol.applyFlags(rules);
    CHECK(vol.statsOf(2)->flagText() == "low conf");
    CHECK(vol.flaggedCount("low conf") == 1);
    CHECK(vol.flaggedCount("touching border") == 2);
    CHECK(vol.flaggedCount("small") == 1);   // object 4
    // object 1 dwarfs the median: the only thing wrong with it is its size
    CHECK(vol.statsOf(1)->flags == std::vector<std::string>{"merged?"});
    CHECK(vol.flaggedCount("merged?") == 1);

    SECTION("review marks and classes survive a recompute") {
        vol.stats()[0].reviewed = true;
        vol.stats()[0].cls = "nucleus";
        CHECK(vol.reviewedCount() == 1);
        const double confidence = vol.statsOf(1)->confidence;
        REQUIRE(confidence < 1.0);   // it came from the probabilities above
        vol.recomputeStats(0);
        CHECK(vol.statsOf(1)->reviewed);
        CHECK(vol.statsOf(1)->cls == "nucleus");
        // no probabilities this time: the known label keeps the confidence
        // the segmentation gave it (cleanup, crop and tracking recompute
        // without probabilities, and used to report every label as 1.0)
        CHECK(vol.statsOf(1)->confidence == confidence);
        CHECK(vol.flaggedCount("low conf") == 1);
    }
    SECTION("distance seeds mark one seed per object") {
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(vol.volumeSize()));
        for (Index i = 0; i < vol.volumeSize(); ++i) mask[static_cast<std::size_t>(i)] = vol.volume(0)[i] ? 1 : 0;
        std::vector<std::uint32_t> seeds(mask.size());
        // the rim of a flat disc is a plateau of distance 1, so the minimum
        // seed distance has to exceed the disc radius to get one seed per object
        CHECK(distanceSeeds(mask.data(), 5, 20, 20, 4.0, seeds.data()) == 4);
        CHECK(distanceSeeds(mask.data(), 5, 20, 20, 3.0, seeds.data()) > 4);
    }
}

TEST_CASE("share() gives a second volume that copies the voxels on its first write", "[app][labels]") {
    LabelVolume a(1, 4, 8, 8);
    a.paint(0, 2, 4, 4, 1.5, 0, 3);
    a.recomputeStats(0);
    REQUIRE(a.statsOf(3));
    const Index voxels = a.statsOf(3)->voxels;

    auto b = a.share();
    CHECK(b->view().data() == a.view().data());   // the same voxels
    CHECK(a.sharesVoxels());
    CHECK(b->sharesVoxels());
    CHECK(b->maxLabel() == a.maxLabel());
    REQUIRE(b->statsOf(3));
    CHECK(b->statsOf(3)->voxels == voxels);       // its own copy of the statistics
    CHECK(b->statsT() == a.statsT());

    // reading does not detach
    CHECK(b->at(0, 2, 4, 4) == 3);
    CHECK(b->view().data() == a.view().data());

    // the first write does
    const LabelDiff diff = b->paint(0, 0, 0, 0, 0.0, 0, 7);
    CHECK_FALSE(diff.empty());
    CHECK(b->view().data() != a.view().data());
    CHECK_FALSE(a.sharesVoxels());
    CHECK_FALSE(b->sharesVoxels());
    CHECK(a.at(0, 0, 0, 0) == 0);
    CHECK(b->at(0, 0, 0, 0) == 7);
    b->stats()[0].reviewed = true;
    CHECK_FALSE(a.statsOf(3)->reviewed);

    SECTION("clone detaches straight away") {
        auto c = a.clone();
        CHECK(c->view().data() != a.view().data());
        CHECK_FALSE(a.sharesVoxels());
        CHECK(c->at(0, 2, 4, 4) == 3);
    }
    SECTION("a share of a share is independent of both") {
        auto c = a.share();
        auto d = a.share();
        c->paint(0, 0, 1, 1, 0.0, 0, 9);
        CHECK(a.at(0, 0, 1, 1) == 0);
        CHECK(d->at(0, 0, 1, 1) == 0);
        CHECK(d->view().data() == a.view().data());
    }
}

TEST_CASE("updateStats brings the table up to date after an edit", "[app][labels]") {
    LabelVolume vol(1, 5, 20, 20);
    vol.paint(0, 2, 10, 10, 3.0, 1, 1);
    vol.paint(0, 2, 3, 3, 1.0, 0, 2);
    LabelFlagRules rules;
    rules.minVoxels = 3;
    vol.recomputeStats(0);
    vol.applyFlags(rules);
    CHECK(vol.statsT() == 0);
    const Index one = vol.statsOf(1)->voxels;
    vol.stats()[0].cls = "nucleus";
    vol.stats()[0].reviewed = true;

    auto countOf = [&vol](std::uint32_t id) {
        Index n = 0;
        for (Index i = 0; i < vol.volumeSize(); ++i)
            if (vol.volume(0)[i] == id) ++n;
        return n;
    };
    auto agrees = [&](std::uint32_t id) {
        const LabelStats* s = vol.statsOf(id);
        const Index n = countOf(id);
        if (n == 0) return s == nullptr;
        return s && s->voxels == n;
    };

    SECTION("growing a label widens its box and its count") {
        const LabelDiff d = vol.paint(0, 2, 10, 16, 1.0, 0, 1);
        vol.updateStats(d);
        CHECK(agrees(1));
        CHECK(vol.statsOf(1)->voxels == one + static_cast<Index>(d.indices.size()));
        CHECK(vol.statsOf(1)->bbox[5] == 18);
        CHECK(vol.statsOf(1)->cls == "nucleus");    // the class and the review mark survive
        CHECK(vol.statsOf(1)->reviewed);
        CHECK(agrees(2));
    }
    SECTION("a new label gets an entry, in id order") {
        const LabelDiff d = vol.paint(0, 4, 18, 18, 0.0, 0, 40);
        vol.updateStats(d);
        REQUIRE(vol.statsOf(40));
        CHECK(vol.statsOf(40)->voxels == 1);
        CHECK(vol.statsOf(40)->touchesBorder);
        CHECK(vol.statsOf(40)->confidence == 1.0);
        CHECK(vol.stats().back().id == 40);
        CHECK(vol.stats().front().id == 1);
    }
    SECTION("removing a label drops its entry") {
        const LabelDiff d = vol.remove(0, 2);
        vol.updateStats(d);
        CHECK(vol.statsOf(2) == nullptr);
        CHECK(agrees(1));
        vol.apply(d, false);
        vol.updateStats(d);
        REQUIRE(vol.statsOf(2));
        CHECK(agrees(2));
    }
    SECTION("merging folds one entry into the other and widens its box") {
        const LabelDiff m = vol.merge(0, {1, 2});
        vol.updateStats(m);
        CHECK(vol.statsOf(2) == nullptr);
        CHECK(agrees(1));
        CHECK(vol.statsOf(1)->bbox[4] == 2);   // the box reaches object 2 now
        CHECK(vol.statsOf(1)->bbox[2] == 2);
    }
    SECTION("the flags of the last applyFlags are refreshed") {
        const LabelDiff d = vol.paint(0, 2, 3, 3, 1.0, 0, 0);   // erase object 2 entirely
        vol.updateStats(d);
        CHECK(vol.statsOf(2) == nullptr);
        CHECK(vol.flaggedCount("small") == 0);
    }
    SECTION("an empty diff and one for another time point are handled") {
        LabelVolume two(2, 3, 8, 8);
        two.paint(1, 1, 4, 4, 1.0, 0, 6);
        two.recomputeStats(0);
        CHECK(two.statsT() == 0);
        CHECK(two.stats().empty());
        const LabelDiff d = two.paint(1, 1, 4, 6, 1.0, 0, 6);
        two.updateStats(d);              // falls back to a full recompute of t = 1
        CHECK(two.statsT() == 1);
        REQUIRE(two.statsOf(6));
        CHECK(two.statsOf(6)->voxels > 1);
        two.updateStats(LabelDiff{});    // nothing to do, and no throw
        CHECK(two.statsT() == 1);
    }
}

TEST_CASE("labelColor cycles the palette and keeps the background black", "[app][labels]") {
    CHECK(labelColor(0) == std::array<float, 3>{0.f, 0.f, 0.f});
    CHECK(labelColor(1) == labelColor(8));
    CHECK(labelColor(1) != labelColor(2));
    CHECK_THAT(labelColor(2)[2], WithinAbs(1.0, 1e-6));   // #7c9cff
}

// --- the classical segmentation's newer building blocks --------------------

TEST_CASE("A median filter removes shot noise and leaves the edge alone", "[app][labels][classic]") {
    // a step edge down the middle, with one hot pixel on each side
    const Index y = 7, x = 8;
    std::vector<float> plane(static_cast<std::size_t>(y * x), 0.0f);
    for (Index r = 0; r < y; ++r)
        for (Index c = 4; c < x; ++c) plane[static_cast<std::size_t>(r * x + c)] = 1.0f;
    plane[static_cast<std::size_t>(3 * x + 1)] = 9.0f;   // hot pixel in the dark half
    plane[static_cast<std::size_t>(3 * x + 6)] = -9.0f;  // cold pixel in the bright half

    std::vector<float> tmp;
    medianFilterPlane(plane.data(), y, x, tmp);
    CHECK_THAT(plane[static_cast<std::size_t>(3 * x + 1)], WithinAbs(0.0, 1e-6));
    CHECK_THAT(plane[static_cast<std::size_t>(3 * x + 6)], WithinAbs(1.0, 1e-6));
    // the edge did not move: column 3 is still dark, column 4 still bright
    CHECK_THAT(plane[static_cast<std::size_t>(1 * x + 3)], WithinAbs(0.0, 1e-6));
    CHECK_THAT(plane[static_cast<std::size_t>(1 * x + 4)], WithinAbs(1.0, 1e-6));

    SECTION("a plane too small to hold the window is left alone") {
        std::vector<float> tiny{1.0f, 5.0f};
        medianFilterPlane(tiny.data(), 1, 2, tmp);
        CHECK_THAT(tiny[1], WithinAbs(5.0, 1e-6));
    }
}

TEST_CASE("Anisotropic diffusion flattens the inside and keeps the boundary", "[app][labels][classic]") {
    const Index y = 9, x = 10;
    std::vector<float> plane(static_cast<std::size_t>(y * x), 0.0f);
    for (Index r = 0; r < y; ++r)
        for (Index c = 5; c < x; ++c) plane[static_cast<std::size_t>(r * x + c)] = 1.0f;
    // texture inside each half that diffusion should even out
    for (Index r = 0; r < y; ++r) {
        plane[static_cast<std::size_t>(r * x + 2)] += (r % 2 == 0 ? 0.05f : -0.05f);
        plane[static_cast<std::size_t>(r * x + 7)] += (r % 2 == 0 ? -0.05f : 0.05f);
    }
    const float stepBefore = plane[static_cast<std::size_t>(4 * x + 5)] - plane[static_cast<std::size_t>(4 * x + 4)];
    auto roughness = [&](const std::vector<float>& v, Index column) {
        double sum = 0.0;
        for (Index r = 1; r < y; ++r)
            sum += std::fabs(v[static_cast<std::size_t>(r * x + column)] - v[static_cast<std::size_t>((r - 1) * x + column)]);
        return sum;
    };
    const double roughBefore = roughness(plane, 2);

    std::vector<float> tmp;
    anisotropicDiffusionPlane(plane.data(), y, x, 12, 0.05, tmp);
    CHECK(roughness(plane, 2) < 0.4 * roughBefore);
    const float stepAfter = plane[static_cast<std::size_t>(4 * x + 5)] - plane[static_cast<std::size_t>(4 * x + 4)];
    CHECK(stepAfter > 0.8f * stepBefore);   // a Gaussian would have halved it

    SECTION("no iterations is no change") {
        std::vector<float> flat(static_cast<std::size_t>(y * x), 0.25f);
        anisotropicDiffusionPlane(flat.data(), y, x, 0, 0.1, tmp);
        CHECK_THAT(flat[10], WithinAbs(0.25, 1e-9));
        // a plane with nothing in it must not divide by its own zero range
        anisotropicDiffusionPlane(flat.data(), y, x, 3, 0.1, tmp);
        CHECK(std::isfinite(flat[10]));
    }
}

TEST_CASE("Triangle and Li cut a skewed histogram lower than Otsu", "[app][labels][classic]") {
    // the shape of a fluorescence field: a tall background peak at 0.05 and a
    // thin tail of signal running up from it with no gap in between. Otsu wants
    // two classes of comparable weight, so it cuts well into the tail and
    // throws most of the signal away; the triangle sits at the foot of the
    // peak and Li lands between the two.
    std::vector<float> values;
    for (int i = 0; i <= 20; ++i)
        for (int k = 0, count = 40 * (11 - std::abs(i - 10)); k < count; ++k) values.push_back(0.005f * static_cast<float>(i));
    const float backgroundPeak = 0.05f, backgroundTop = 0.10f;
    for (int i = 0; i < 60; ++i) values.push_back(0.12f + 0.014f * static_cast<float>(i));
    const float signalTop = 0.12f + 0.014f * 59.0f;
    const Index n = static_cast<Index>(values.size());

    const float triangle = triangleThreshold(values.data(), n);
    const float li = liThreshold(values.data(), n);
    const float otsu = otsuThreshold(values.data(), n);
    CHECK(triangle > backgroundPeak);
    CHECK(triangle < li);
    CHECK(li < otsu);
    CHECK(otsu < signalTop);
    // what the ordering costs and buys: the triangle keeps every signal voxel
    // and lets some background in, Otsu is the other way round
    auto kept = [&](float cut, float from, float to) {
        Index count = 0;
        for (float v : values)
            if (v > cut && v >= from && v <= to) ++count;
        return count;
    };
    CHECK(kept(triangle, 0.12f, 1.0f) == 60);
    CHECK(kept(otsu, 0.12f, 1.0f) < 60);
    CHECK(kept(otsu, 0.0f, backgroundTop) == 0);
    CHECK(kept(triangle, 0.0f, backgroundTop) > 0);

    SECTION("a flat image has no threshold to find") {
        std::vector<float> flat(64, 0.4f);
        CHECK_THAT(triangleThreshold(flat.data(), 64), WithinAbs(0.4, 1e-6));
        CHECK_THAT(liThreshold(flat.data(), 64), WithinAbs(0.4, 1e-6));
        CHECK_THAT(triangleThreshold(nullptr, 0), WithinAbs(0.0, 1e-9));
        CHECK_THAT(liThreshold(nullptr, 0), WithinAbs(0.0, 1e-9));
    }
}

TEST_CASE("Hysteresis keeps what is attached to something certain", "[app][labels][classic]") {
    // a bright core with a dim tail, and a dim speck on its own
    const Index z = 1, y = 5, x = 9;
    const Index n = z * y * x;
    std::vector<std::uint8_t> high(static_cast<std::size_t>(n), 0), low(static_cast<std::size_t>(n), 0), out(static_cast<std::size_t>(n), 0);
    auto at = [&](Index r, Index c) { return static_cast<std::size_t>(r * x + c); };
    for (Index c = 1; c <= 2; ++c) high[at(2, c)] = 1;             // the certain core
    for (Index c = 1; c <= 5; ++c) low[at(2, c)] = 1;              // core plus its fading tail
    low[at(0, 8)] = 1;                                             // a speck with no core

    const Index kept = hysteresisMask(high.data(), low.data(), z, y, x, out.data());
    CHECK(kept == 5);
    for (Index c = 1; c <= 5; ++c) CHECK(out[at(2, c)] == 1);
    CHECK(out[at(0, 8)] == 0);

    SECTION("nothing certain keeps nothing") {
        std::fill(high.begin(), high.end(), std::uint8_t{0});
        CHECK(hysteresisMask(high.data(), low.data(), z, y, x, out.data()) == 0);
    }
}

TEST_CASE("The gradient magnitude peaks on the edge, not in the object", "[app][labels][classic]") {
    const Index z = 3, y = 5, x = 7;
    std::vector<float> values(static_cast<std::size_t>(z * y * x), 0.0f);
    auto at = [&](Index k, Index r, Index c) { return static_cast<std::size_t>((k * y + r) * x + c); };
    for (Index k = 0; k < z; ++k)
        for (Index r = 0; r < y; ++r)
            for (Index c = 4; c < x; ++c) values[at(k, r, c)] = 1.0f;
    std::vector<float> g(values.size(), 0.0f);
    gradientMagnitude(values.data(), z, y, x, 1.0, g.data());
    CHECK_THAT(g[at(1, 2, 3)], WithinAbs(0.5, 1e-6));   // one voxel before the step
    CHECK_THAT(g[at(1, 2, 4)], WithinAbs(0.5, 1e-6));   // one voxel after it
    CHECK_THAT(g[at(1, 2, 1)], WithinAbs(0.0, 1e-6));   // flat inside
    CHECK_THAT(g[at(1, 2, 6)], WithinAbs(0.0, 1e-6));

    SECTION("z is measured in the same physical units as x and y") {
        std::vector<float> ramp(static_cast<std::size_t>(z * y * x), 0.0f);
        for (Index k = 0; k < z; ++k)
            for (Index r = 0; r < y; ++r)
                for (Index c = 0; c < x; ++c) ramp[at(k, r, c)] = static_cast<float>(k);
        gradientMagnitude(ramp.data(), z, y, x, 2.0, g.data());
        CHECK_THAT(g[at(1, 2, 3)], WithinAbs(0.5, 1e-6));   // one per plane, planes twice as far apart
    }
}

TEST_CASE("The active contour moves the mask onto the object", "[app][labels][classic]") {
    // a bright square on a dark ground
    const Index y = 16, x = 16;
    std::vector<float> image(static_cast<std::size_t>(y * x), 0.0f);
    auto at = [&](Index r, Index c) { return static_cast<std::size_t>(r * x + c); };
    for (Index r = 4; r < 12; ++r)
        for (Index c = 4; c < 12; ++c) image[at(r, c)] = 1.0f;
    auto agreement = [&](const std::vector<std::uint8_t>& m) {
        Index inside = 0, outside = 0;
        for (Index r = 0; r < y; ++r)
            for (Index c = 0; c < x; ++c)
                if (m[at(r, c)]) ((r >= 4 && r < 12 && c >= 4 && c < 12) ? inside : outside) += 1;
        return std::pair<Index, Index>{inside, outside};
    };
    std::vector<std::uint8_t> tmp;

    SECTION("a mask that fell short of the object grows out to its edge") {
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(y * x), 0);
        for (Index r = 6; r < 10; ++r)
            for (Index c = 6; c < 10; ++c) mask[at(r, c)] = 1;
        // no curvature term here: the region force alone has to do the work
        morphologicalChanVesePlane(image.data(), mask.data(), y, x, 30, 0, tmp);
        const auto [inside, outside] = agreement(mask);
        CHECK(inside == 64);   // the whole square
        CHECK(outside == 0);
    }

    SECTION("a mask that spilled over the object contracts onto it") {
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(y * x), 0);
        for (Index r = 2; r < 14; ++r)
            for (Index c = 2; c < 14; ++c) mask[at(r, c)] = 1;
        morphologicalChanVesePlane(image.data(), mask.data(), y, x, 30, 0, tmp);
        const auto [inside, outside] = agreement(mask);
        CHECK(inside == 64);
        CHECK(outside == 0);
    }

    SECTION("the curvature term removes a thin arm the threshold left behind") {
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(y * x), 0);
        for (Index r = 4; r < 12; ++r)
            for (Index c = 4; c < 12; ++c) mask[at(r, c)] = 1;
        for (Index c = 12; c < 15; ++c) mask[at(8, c)] = 1;   // one voxel wide, over dark ground
        morphologicalChanVesePlane(image.data(), mask.data(), y, x, 6, 1, tmp);
        for (Index c = 12; c < 15; ++c) CHECK(mask[at(8, c)] == 0);
        const auto [inside, outside] = agreement(mask);
        CHECK(inside > 50);
        CHECK(outside == 0);
    }

    SECTION("a mask that covers everything is left alone") {
        std::vector<std::uint8_t> all(static_cast<std::size_t>(y * x), 1);
        morphologicalChanVesePlane(image.data(), all.data(), y, x, 5, 1, tmp);
        CHECK(std::all_of(all.begin(), all.end(), [](std::uint8_t v) { return v == 1; }));
    }
}

TEST_CASE("Shape filters drop what is not the object being looked for", "[app][labels][classic]") {
    const Index z = 1, y = 12, x = 12;
    std::vector<std::uint32_t> labels(static_cast<std::size_t>(z * y * x), 0u);
    auto at = [&](Index r, Index c) { return static_cast<std::size_t>(r * x + c); };
    // 1: a compact 3x3 block well inside the plane
    for (Index r = 4; r < 7; ++r)
        for (Index c = 4; c < 7; ++c) labels[at(r, c)] = 1;
    // 2: a long thin line, also inside
    for (Index c = 1; c < 11; ++c) labels[at(9, c)] = 2;
    // 3: a block against the left edge
    for (Index r = 1; r < 3; ++r)
        for (Index c = 0; c < 2; ++c) labels[at(r, c)] = 3;

    auto surviving = [&](const ShapeFilter& f) {
        std::vector<std::uint32_t> copy = labels;
        const std::uint32_t count = filterLabelsByShape(copy.data(), z, y, x, f);
        std::set<std::uint32_t> ids;
        for (std::uint32_t v : copy)
            if (v) ids.insert(v);
        CHECK(ids.size() == count);   // relabelled densely
        return count;
    };

    ShapeFilter none;
    CHECK(surviving(none) == 3);

    ShapeFilter border;
    border.dropBorder = true;
    CHECK(surviving(border) == 2);   // the one against the edge goes

    ShapeFilter thin;
    thin.maxElongation = 3.0;
    CHECK(surviving(thin) == 2);     // the 10 x 1 line goes

    ShapeFilter big;
    big.maxVoxels = 10;
    CHECK(surviving(big) == 3);      // the largest object here is the 10 voxel line
    big.maxVoxels = 9;
    CHECK(surviving(big) == 2);      // which this drops
    big.maxVoxels = 5;
    CHECK(surviving(big) == 1);      // only the 4 voxel corner block is left

    ShapeFilter fill;
    fill.minFill = 0.9;
    CHECK(surviving(fill) == 3);     // all three fill their own boxes

    SECTION("a volume with no labels filters to nothing") {
        std::vector<std::uint32_t> empty(static_cast<std::size_t>(z * y * x), 0u);
        CHECK(filterLabelsByShape(empty.data(), z, y, x, border) == 0);
    }
}

TEST_CASE("Yen and Isodata place the cut where their definitions say", "[app][labels][classic]") {
    // two clean modes of very different weight: 4000 voxels of background
    // around 0.1 and 800 of signal around 0.7
    std::vector<float> values;
    for (int i = 0; i < 4000; ++i) values.push_back(0.08f + 0.001f * static_cast<float>(i % 41));
    for (int i = 0; i < 800; ++i) values.push_back(0.66f + 0.0002f * static_cast<float>(i % 401));

    const Index n = static_cast<Index>(values.size());
    const float yen = yenThreshold(values.data(), n);
    const float isodata = isodataThreshold(values.data(), n);
    // both land in the empty band between the two modes, or at its very edge:
    // Yen settles at the top of the background, isodata halfway across
    CHECK(yen > 0.11f);
    CHECK(yen < 0.66f);
    CHECK(isodata > 0.12f);
    CHECK(isodata < 0.66f);
    CHECK(yen < isodata);
    // isodata is the midpoint of the two class means, so it does not care that
    // there is five times as much background as signal
    CHECK_THAT(isodata, WithinAbs(0.5 * (0.10 + 0.70), 0.05));

    SECTION("a flat image has no threshold to find") {
        std::vector<float> flat(64, 0.3f);
        CHECK_THAT(yenThreshold(flat.data(), 64), WithinAbs(0.3, 1e-6));
        CHECK_THAT(isodataThreshold(flat.data(), 64), WithinAbs(0.3, 1e-6));
        CHECK_THAT(yenThreshold(nullptr, 0), WithinAbs(0.0, 1e-9));
        CHECK_THAT(isodataThreshold(nullptr, 0), WithinAbs(0.0, 1e-9));
    }
}

TEST_CASE("The 3D hole fill closes what no single plane encloses", "[app][labels][classic]") {
    // a solid 5x5x5 block of foreground with an empty voxel at its centre. In
    // every plane through that voxel the background around it is enclosed only
    // by way of the planes above and below, so a per-plane fill cannot see it.
    const Index z = 7, y = 7, x = 7;
    std::vector<std::uint8_t> mask(static_cast<std::size_t>(z * y * x), 0);
    auto at = [&](Index k, Index r, Index c) { return static_cast<std::size_t>((k * y + r) * x + c); };
    for (Index k = 1; k <= 5; ++k)
        for (Index r = 1; r <= 5; ++r)
            for (Index c = 1; c <= 5; ++c) mask[at(k, r, c)] = 1;
    mask[at(3, 3, 3)] = 0;   // the cavity

    std::vector<std::uint8_t> copy = mask;
    CHECK(fillHoles3D(copy.data(), z, y, x, 0) == 1);
    CHECK(copy[at(3, 3, 3)] == 1);
    CHECK(copy[at(0, 0, 0)] == 0);   // the outside is still outside

    SECTION("a size limit leaves a real lumen open") {
        std::vector<std::uint8_t> bigger = mask;
        bigger[at(3, 3, 4)] = 0;   // a two voxel cavity, still inside the block
        std::vector<std::uint8_t> limited = bigger;
        CHECK(fillHoles3D(limited.data(), z, y, x, 1) == 0);
        CHECK(limited[at(3, 3, 3)] == 0);
        std::vector<std::uint8_t> unlimited = bigger;
        CHECK(fillHoles3D(unlimited.data(), z, y, x, 0) == 2);
        CHECK(unlimited[at(3, 3, 3)] == 1);
        CHECK(unlimited[at(3, 3, 4)] == 1);
    }

    SECTION("a cavity open to the border is background, not a hole") {
        std::vector<std::uint8_t> open = mask;
        open[at(3, 3, 4)] = 0;
        open[at(3, 3, 5)] = 0;   // a channel through the wall to the outside
        CHECK(fillHoles3D(open.data(), z, y, x, 0) == 0);
        CHECK(open[at(3, 3, 3)] == 0);
    }

    SECTION("a volume with nothing in it fills nothing") {
        std::vector<std::uint8_t> empty(static_cast<std::size_t>(z * y * x), 0);
        CHECK(fillHoles3D(empty.data(), z, y, x, 0) == 0);
    }
}

TEST_CASE("Expanding labels closes the gap without joining two objects", "[app][labels][classic]") {
    // two single-voxel labels five apart in x, in one plane
    const Index z = 1, y = 7, x = 11;
    std::vector<std::uint32_t> labels(static_cast<std::size_t>(z * y * x), 0u);
    auto at = [&](Index r, Index c) { return static_cast<std::size_t>(r * x + c); };
    labels[at(3, 2)] = 1;
    labels[at(3, 8)] = 2;

    std::vector<std::uint32_t> grown = labels;
    const Index claimed = expandLabels(grown.data(), z, y, x, 2.0, 1.0);
    CHECK(claimed > 0);
    CHECK(grown[at(3, 2)] == 1);   // the seeds keep their own ids
    CHECK(grown[at(3, 8)] == 2);
    CHECK(grown[at(3, 3)] == 1);   // one step out
    CHECK(grown[at(3, 4)] == 1);   // two steps out, the limit
    CHECK(grown[at(3, 7)] == 2);
    CHECK(grown[at(3, 6)] == 2);
    // the midpoint is the same distance from both, so it stays background and
    // the two objects do not fuse
    CHECK(grown[at(3, 5)] == 0);
    // and nothing grew further than it was told to: two steps in y is still
    // the object, three is not
    CHECK(grown[at(1, 2)] == 1);
    CHECK(grown[at(0, 2)] == 0);

    SECTION("a distance of zero changes nothing") {
        std::vector<std::uint32_t> same = labels;
        CHECK(expandLabels(same.data(), z, y, x, 0.0, 1.0) == 0);
        CHECK(same == labels);
    }

    SECTION("z costs more when the planes are further apart") {
        std::vector<std::uint32_t> tall(static_cast<std::size_t>(5 * y * x), 0u);
        tall[static_cast<std::size_t>((2 * y + 3) * x + 5)] = 1;
        // planes three times as far apart as pixels: one step in z is three
        std::vector<std::uint32_t> near = tall;
        expandLabels(near.data(), 5, y, x, 2.0, 3.0);
        CHECK(near[static_cast<std::size_t>((1 * y + 3) * x + 5)] == 0u);   // z is out of reach
        CHECK(near[static_cast<std::size_t>((2 * y + 3) * x + 6)] == 1u);   // x is not
        std::vector<std::uint32_t> far = tall;
        expandLabels(far.data(), 5, y, x, 4.0, 3.0);
        CHECK(far[static_cast<std::size_t>((1 * y + 3) * x + 5)] == 1u);
    }
}

TEST_CASE("Thinning leaves a one voxel centreline with the same topology", "[app][labels][classic]") {
    auto count = [](const std::vector<std::uint8_t>& v) {
        return static_cast<Index>(std::count_if(v.begin(), v.end(), [](std::uint8_t m) { return m != 0; }));
    };

    SECTION("a thick bar becomes a line down its middle") {
        const Index z = 5, y = 7, x = 15;
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(z * y * x), 0);
        auto at = [&](Index k, Index r, Index c) { return static_cast<std::size_t>((k * y + r) * x + c); };
        for (Index k = 1; k <= 3; ++k)
            for (Index r = 2; r <= 4; ++r)
                for (Index c = 2; c <= 12; ++c) mask[at(k, r, c)] = 1;
        const Index before = count(mask);
        const Index after = skeletonize3D(mask.data(), z, y, x);
        CHECK(after < before / 4);
        CHECK(after >= 9);   // the bar is 11 long: the line keeps its length
        // what is left runs along the middle of the bar, not along its face
        Index onAxis = 0;
        for (Index c = 0; c < x; ++c)
            if (mask[at(2, 3, c)]) ++onAxis;
        CHECK(onAxis >= after - 2);
    }

    SECTION("a loop stays a loop") {
        // a square annulus three voxels thick, in one plane. Its hole is a
        // tunnel, not a cavity -- the background reaches it from above -- so
        // what has to survive the thinning is the cycle itself.
        const Index z = 3, y = 15, x = 15;
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(z * y * x), 0);
        auto at = [&](Index k, Index r, Index c) { return static_cast<std::size_t>((k * y + r) * x + c); };
        for (Index r = 2; r <= 12; ++r)
            for (Index c = 2; c <= 12; ++c)
                if (!(r >= 5 && r <= 9 && c >= 5 && c <= 9)) mask[at(1, r, c)] = 1;
        const Index before = count(mask);
        const Index after = skeletonize3D(mask.data(), z, y, x);
        CHECK(after < before / 2);

        // one 26-connected piece: a skeleton steps diagonally, which the
        // 6-connected connectedComponents() would count as several
        auto neighbours = [&](Index i) {
            const Index k = i / (y * x), rest = i % (y * x), r = rest / x, c = rest % x;
            std::vector<Index> out;
            for (Index dz = -1; dz <= 1; ++dz)
                for (Index dy = -1; dy <= 1; ++dy)
                    for (Index dx = -1; dx <= 1; ++dx) {
                        if (dz == 0 && dy == 0 && dx == 0) continue;
                        const Index jz = k + dz, jy = r + dy, jx = c + dx;
                        if (jz < 0 || jz >= z || jy < 0 || jy >= y || jx < 0 || jx >= x) continue;
                        const Index j = (jz * y + jy) * x + jx;
                        if (mask[static_cast<std::size_t>(j)]) out.push_back(j);
                    }
            return out;
        };
        Index first = -1;
        for (Index i = 0; i < z * y * x && first < 0; ++i)
            if (mask[static_cast<std::size_t>(i)]) first = i;
        REQUIRE(first >= 0);
        std::vector<std::uint8_t> seen(static_cast<std::size_t>(z * y * x), 0);
        std::vector<Index> stack{first};
        seen[static_cast<std::size_t>(first)] = 1;
        Index reached = 1;
        while (!stack.empty()) {
            const Index cur = stack.back();
            stack.pop_back();
            for (Index j : neighbours(cur))
                if (!seen[static_cast<std::size_t>(j)]) {
                    seen[static_cast<std::size_t>(j)] = 1;
                    ++reached;
                    stack.push_back(j);
                }
        }
        CHECK(reached == after);   // one piece

        // and it is still a cycle rather than a cut line. Thinning can leave a
        // single spur where a corner forces a voxel to become an end point
        // before its turn comes -- scikit-image's ordering avoids it on this
        // shape and ours does not -- so one loose end is allowed, two are not.
        Index ends = 0;
        for (Index i = 0; i < z * y * x; ++i)
            if (mask[static_cast<std::size_t>(i)] && neighbours(i).size() < 2) ++ends;
        CHECK(ends <= 1);
    }

    SECTION("an isolated voxel and an empty volume are left alone") {
        const Index z = 3, y = 3, x = 3;
        std::vector<std::uint8_t> one(static_cast<std::size_t>(z * y * x), 0);
        one[static_cast<std::size_t>((1 * y + 1) * x + 1)] = 1;
        CHECK(skeletonize3D(one.data(), z, y, x) == 1);
        std::vector<std::uint8_t> none(static_cast<std::size_t>(z * y * x), 0);
        CHECK(skeletonize3D(none.data(), z, y, x) == 0);
    }
}

TEST_CASE("dropSmall drops without renumbering, removeSmall renumbers", "[app][labels]") {
    std::vector<std::uint32_t> v{0, 5, 5, 5, 9, 9, 0, 7, 0, 5};
    std::vector<std::uint32_t> w = v;
    CHECK(dropSmall(v.data(), static_cast<Index>(v.size()), 2) == 2);
    CHECK(v == std::vector<std::uint32_t>{0, 5, 5, 5, 9, 9, 0, 0, 0, 5});
    CHECK(removeSmall(w.data(), static_cast<Index>(w.size()), 2) == 2);
    CHECK(w == std::vector<std::uint32_t>{0, 1, 1, 1, 2, 2, 0, 0, 0, 1});
    CHECK(dropSmall(v.data(), static_cast<Index>(v.size()), 0) == 2);   // nothing to drop
    CHECK(v == std::vector<std::uint32_t>{0, 5, 5, 5, 9, 9, 0, 0, 0, 5});
}

TEST_CASE("The 256-bin histogram puts a bin's centre where valueOf says", "[app][labels][threshold]") {
    // two values only: the cut of every histogram threshold lies between them
    // and, with equal bins over [lo, hi], nearer neither end than the binning
    // bias put it before (255 bins of width but 256 in the centre formula)
    std::vector<float> values(1000, 0.0f);
    for (std::size_t i = 0; i < values.size(); i += 2) values[i] = 1.0f;
    const float t = isodataThreshold(values.data(), static_cast<Index>(values.size()));
    CHECK(t > 0.3f);
    CHECK(t < 0.7f);
}

TEST_CASE("A LabelVolume remembers that it was edited", "[app][labels]") {
    LabelVolume v(1, 2, 8, 8);
    CHECK_FALSE(v.edited());
    v.paint(0, 0, 4, 4, 1.0, 0, 3);
    CHECK(v.edited());
    CHECK_FALSE(v.share()->edited());   // a share over the same voxels has not been edited on its own
    CHECK_FALSE(v.clone()->edited());
    LabelVolume w(1, 2, 8, 8);
    LabelDiff d;
    d.t = 0;
    d.indices = {0};
    d.before = {0};
    d.after = {5};
    w.apply(d, true);   // an undo / redo counts as an edit too
    CHECK(w.edited());
    LabelVolume untouched(1, 2, 8, 8);
    untouched.recomputeStats(0);
    (void)untouched.volume(0);   // a mutable accessor alone is not an edit
    CHECK_FALSE(untouched.edited());
}
