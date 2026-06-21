#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include "sirius/tiff_io.hpp"

using namespace sirius;
using namespace std::filesystem;

TEST_CASE("Reconstruction", "[reconstruction]") {
    const path data_dir = SIRIUS_TEST_DATA_DIR;
    INFO("Data directory: " << data_dir);
    const path image_path = data_dir / "raw.tif";
    const path otf_path = data_dir / "otf.tif";
    const path expected_path = data_dir / "raw_proc.tif";

    const auto image = readTiffStack<float>(image_path.string());
    const auto otf = readTiffStack<float>(otf_path.string());
    const auto expected = readTiffStack<float>(expected_path.string());

    // TODO: after implementing reconstruction, uncomment this

    // const auto actual = reconstruct(image, otf);
    // REQUIRE(actual.dimension(0) == expected.dimension(0));
    // REQUIRE(actual.dimension(1) == expected.dimension(1));
    // REQUIRE(actual.dimension(2) == expected.dimension(2));
    // for (Eigen::Index z = 0; z < expected.dimension(0); ++z) {
    //     REQUIRE(slice(actual, z).isApprox(slice(expected, z), 1e-4f));
    // }
    REQUIRE(expected.dimension(0) == 9);
    REQUIRE(expected.dimension(1) == 128);
    REQUIRE(expected.dimension(2) == 128);
}