// stitchTiffTiles: the stitcher driven from TIFF files and back to one.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include "sirius/stitching_tiff.hpp"
#include "stitching_scene.hpp"
#include "temp_path.hpp"

using namespace sirius;
using namespace sirius::test::stitching;
using Catch::Matchers::WithinAbs;

TEST_CASE("TIFF tiles stitch end to end", "[stitching]") {
    const Scene scene = makeScene({3, 40, 96}, 31);
    const Tile left = cut(scene, {0, 0, 0}, {3, 40, 60});
    const Tile right = cut(scene, {0, 0, 36}, {3, 40, 60});

    sirius::test::TempFile leftFile("stitch_left", ".tif");
    sirius::test::TempFile rightFile("stitch_right", ".tif");
    sirius::test::TempFile outFile("stitch_out", ".tif");
    writeTiffStack<float>(leftFile.str, left.view());
    writeTiffStack<float>(rightFile.str, right.view());

    StitchOptions options;
    options.searchRadius = {1, 8, 10};
    options.blend = BlendMode::Feather;

    StitchLayout layout;
    const std::vector<StitchTile> inputs{{leftFile.str, {0, 0, 0}}, {rightFile.str, {0, 2, 41}}};
    const Buffer<float> fused = stitchTiffTiles<float>(inputs, options, &layout, outFile.str);

    REQUIRE(layout.positions.size() == 2);
    CHECK_THAT(layout.positions[0][2], WithinAbs(0.0, 1e-9));
    CHECK_THAT(layout.positions[1][1], WithinAbs(0.0, 0.4));
    CHECK_THAT(layout.positions[1][2], WithinAbs(36.0, 0.4));
    REQUIRE(fused.shape() == Shape{3, 40, 96});

    double worst = 0.0;
    for (Index z = 0; z < 3; ++z)
        for (Index y = 0; y < 40; ++y)
            for (Index x = 0; x < 96; ++x)
                worst = std::max(worst, std::abs(static_cast<double>(fused.data()[(z * 40 + y) * 96 + x]) -
                                                 scene.value(z, y, x)));
    INFO("largest deviation from the source scene: " << worst);
    CHECK(worst < 1e-5);

    const auto reread = readTiffStack<float>(outFile.str);
    CHECK(reread.dimension(0) == 3);
    CHECK(reread.dimension(2) == 96);
}
