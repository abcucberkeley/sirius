// Mosaic stitching: pairwise registration of overlapping tiles, the global
// fit of the tile origins, and blended fusion. The scenes here are cut from a
// known volume, so the correct answer is exactly the cut offset.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "sirius/stitching.hpp"
#include "stitching_scene.hpp"

using namespace sirius;
using namespace sirius::test::stitching;
using Catch::Matchers::WithinAbs;

TEST_CASE("a tile pair registers back to its true offset", "[stitching]") {
    const Scene scene = makeScene({3, 90, 120}, 5);
    const Tile a = cut(scene, {0, 0, 0}, {2, 60, 60});
    const Tile b = cut(scene, {1, 6, 40}, {2, 60, 50});

    // The stage reported the tile 7 columns and 3 rows off where it is.
    const Pos nominalB{1.0, 3.0, 47.0};
    StitchOptions options;
    options.searchRadius = {1, 12, 12};

    const TileMatch m = registerTilePair<float>(a.view(), {0.0, 0.0, 0.0}, b.view(), nominalB, options);
    REQUIRE(m.accepted);
    CHECK(m.correlation > 0.99);
    CHECK_THAT(m.displacement[0], WithinAbs(1.0, 0.05));
    CHECK_THAT(m.displacement[1], WithinAbs(6.0, 0.05));
    CHECK_THAT(m.displacement[2], WithinAbs(40.0, 0.05));
    CHECK_THAT(m.nominalDisplacement[2], WithinAbs(47.0, 1e-12));
}

TEST_CASE("registration reports no match for tiles that do not meet", "[stitching]") {
    const Scene scene = makeScene({1, 60, 60}, 6);
    const Tile a = cut(scene, {0, 0, 0}, {1, 20, 20});
    const Tile b = cut(scene, {0, 30, 30}, {1, 20, 20});

    const TileMatch m = registerTilePair<float>(a.view(), {0, 0, 0}, b.view(), {0, 30, 30});
    CHECK_FALSE(m.accepted);
    CHECK(m.overlap == 0.0);
}

TEST_CASE("the global fit turns pairwise displacements into consistent origins", "[stitching]") {
    // Three tiles in a row. The measured displacements are exact; the nominal
    // positions carry stage error. With pair weights well above the nominal
    // weight, the relative geometry must follow the measurements.
    const std::vector<Pos> truth{{0, 0, 0}, {0, 0, 100}, {0, 0, 200}};
    const std::vector<Pos> nominal{{0, 0, 0}, {0, 0, 93}, {0, 0, 211}};

    std::vector<TileMatch> matches(2);
    matches[0] = {0, 1, {0, 0, 100}, {0, 0, 93}, 0.95, 1000.0, true};
    matches[1] = {1, 2, {0, 0, 100}, {0, 0, 118}, 0.9, 1000.0, true};

    const auto fit = optimizeTilePositions(nominal, matches);
    REQUIRE(fit.size() == 3);
    // Anchored on tile 0, so the absolute positions are the true ones.
    for (std::size_t i = 0; i < 3; ++i)
        CHECK_THAT(fit[i][2], WithinAbs(truth[i][2], 0.2));

    // Without an anchor the fit floats to the centroid of the nominal
    // positions, but the geometry it measures is the same.
    const auto floating = optimizeTilePositions(nominal, matches, 1e-3, std::size_t(-1));
    for (std::size_t i = 0; i + 1 < 3; ++i)
        CHECK_THAT(floating[i + 1][2] - floating[i][2], WithinAbs(100.0, 0.2));

    // A rejected match carries no weight at all, so its tile stays nominal.
    std::vector<TileMatch> rejected = matches;
    rejected[1].accepted = false;
    const auto partial = optimizeTilePositions(nominal, rejected);
    CHECK_THAT(partial[2][2], WithinAbs(nominal[2][2], 1e-9));

    CHECK_THROWS_AS(optimizeTilePositions(nominal, matches, 0.0), std::invalid_argument);
}

TEST_CASE("the canvas covers every placed tile", "[stitching]") {
    const std::vector<Shape> shapes{Shape{2, 10, 10}, Shape{2, 10, 10}};
    const std::vector<Pos> positions{{0, -4, 3}, {1, 6, 20}};
    Ext origin{}, extent{};
    tileCanvas(shapes, positions, origin, extent);
    CHECK(origin == Ext{0, -4, 3});
    CHECK(extent == Ext{3, 20, 27});
}

TEST_CASE("fusion reproduces the scene under every blend mode", "[stitching]") {
    const Scene scene = makeScene({2, 40, 80}, 9);
    const Tile a = cut(scene, {0, 0, 0}, {2, 40, 50});
    const Tile b = cut(scene, {0, 0, 30}, {2, 40, 50});
    const std::vector<BufferView<const float>> tiles{a.view(), b.view()};
    const std::vector<Pos> positions{{0, 0, 0}, {0, 0, 30}};

    Ext origin{}, extent{};
    tileCanvas({Shape{2, 40, 50}, Shape{2, 40, 50}}, positions, origin, extent);
    REQUIRE(extent == Ext{2, 40, 80});

    for (BlendMode mode : {BlendMode::Overwrite, BlendMode::Average, BlendMode::Feather,
                           BlendMode::Maximum}) {
        StitchOptions options;
        options.blend = mode;
        const Buffer<float> fused = fuseTiles<float>(tiles, positions, origin, extent, options);
        REQUIRE(fused.shape() == Shape{2, 40, 80});
        double worst = 0.0;
        for (Index z = 0; z < 2; ++z)
            for (Index y = 0; y < 40; ++y)
                for (Index x = 0; x < 80; ++x)
                    worst = std::max(worst,
                                     std::abs(static_cast<double>(
                                                  fused.data()[(z * 40 + y) * 80 + x]) -
                                              scene.value(z, y, x)));
        // Both tiles carry identical values in the overlap, so any weighting
        // of them is the scene again.
        INFO("blend mode " << static_cast<int>(mode));
        CHECK(worst < 1e-5);
    }
}

TEST_CASE("fusion leaves uncovered voxels at zero and can skip background", "[stitching]") {
    Tile t;
    t.extent = {1, 4, 4};
    t.data.assign(16, 5.0f);
    t.data[0] = 0.0f;   // a "no data" voxel in the corner

    StitchOptions options;
    options.blend = BlendMode::Average;
    options.skipBackground = true;
    options.fusionBackgroundLevel = 0.0;

    const Buffer<float> fused = fuseTiles<float>({t.view()}, {{0, 0, 0}}, {0, 0, 0}, {1, 6, 6}, options);
    REQUIRE(fused.shape() == Shape{1, 6, 6});
    CHECK(fused.data()[0] == 0.0f);            // skipped, stays empty
    CHECK(fused.data()[1] == 5.0f);
    CHECK(fused.data()[5] == 0.0f);            // outside the tile
    CHECK(fused.data()[35] == 0.0f);
}

TEST_CASE("a 2x2 mosaic is planned and fused back into the scene", "[stitching]") {
    const Scene scene = makeScene({2, 96, 120}, 17);
    const Ext tileExtent{2, 56, 70};
    const std::vector<Ext> origins{{0, 0, 0}, {0, 0, 50}, {0, 40, 0}, {0, 40, 50}};

    std::vector<Tile> tiles;
    std::vector<Pos> nominal;
    std::vector<Pos> truth;
    // Stage error of a few voxels per tile, different for each.
    const std::vector<Pos> error{{0, 0, 0}, {0, -4, 6}, {0, 5, -3}, {0, -6, -5}};
    for (std::size_t i = 0; i < origins.size(); ++i) {
        tiles.push_back(cut(scene, origins[i], tileExtent));
        truth.push_back({static_cast<double>(origins[i][0]), static_cast<double>(origins[i][1]),
                         static_cast<double>(origins[i][2])});
        nominal.push_back({truth[i][0] + error[i][0], truth[i][1] + error[i][1],
                           truth[i][2] + error[i][2]});
    }

    std::vector<BufferView<const float>> views;
    for (const Tile& t : tiles) views.push_back(t.view());

    StitchOptions options;
    options.searchRadius = {1, 12, 12};
    // The diagonal pairs nominally share only ~2% of a tile, and the stage
    // error shrinks that further; they still register once the search radius
    // grows the fixed block.
    options.minOverlapFraction = 0.01;

    const StitchLayout layout = planStitch<float>(views, nominal, options);
    REQUIRE(layout.positions.size() == 4);
    // Every pair of these four tiles overlaps, so all six should be measured.
    CHECK(layout.matches.size() == 6);
    for (const TileMatch& m : layout.matches) {
        INFO("pair " << m.fixed << " -> " << m.moving);
        CHECK(m.accepted);
        CHECK(m.correlation > 0.9);
    }

    for (std::size_t i = 0; i < 4; ++i) {
        INFO("tile " << i);
        for (int a = 1; a < 3; ++a)
            CHECK_THAT(layout.positions[i][a], WithinAbs(truth[i][a], 0.4));
    }
    CHECK(layout.canvasOrigin == Ext{0, 0, 0});
    CHECK(layout.canvasExtent == Ext{2, 96, 120});

    const Buffer<float> fused = fuseTiles<float>(views, layout.positions, layout.canvasOrigin,
                                                 layout.canvasExtent, options);
    CHECK(fused.shape() == Shape{2, layout.canvasExtent[1], layout.canvasExtent[2]});
}
