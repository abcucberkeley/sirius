#ifndef SIRIUS_TESTS_STITCHING_SCENE_HPP
#define SIRIUS_TESTS_STITCHING_SCENE_HPP

// The scenes of the stitching tests: tiles cut from a known volume, so the
// correct answer is exactly the cut offset. Shared by test_stitching.cpp
// (memory only) and test_stitching_tiff.cpp (the same through TIFF files).

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "sirius/buffer.hpp"

namespace sirius::test::stitching {

    using Ext = std::array<Index, 3>;
    using Pos = std::array<double, 3>;

    struct Scene {
        Ext extent{1, 1, 1};
        std::vector<float> data;

        Index at(Index z, Index y, Index x) const { return (z * extent[1] + y) * extent[2] + x; }
        float value(Index z, Index y, Index x) const {
            return data[static_cast<std::size_t>(at(z, y, x))];
        }
    };

    // Smooth-but-textured content: correlation of pure white noise is fine but
    // this is closer to an image, and every tile sees the same field.
    inline Scene makeScene(Ext extent, unsigned seed) {
        Scene s;
        s.extent = extent;
        s.data.resize(static_cast<std::size_t>(extent[0] * extent[1] * extent[2]));
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> phase(0.0, 6.283185307179586);
        std::uniform_real_distribution<double> noise(-0.05, 0.05);
        const double p1 = phase(rng), p2 = phase(rng), p3 = phase(rng);
        for (Index z = 0; z < extent[0]; ++z)
            for (Index y = 0; y < extent[1]; ++y)
                for (Index x = 0; x < extent[2]; ++x) {
                    const double v = std::sin(0.31 * x + p1) * std::cos(0.23 * y + p2) +
                                     0.6 * std::sin(0.11 * (x + y) + p3) +
                                     0.4 * std::cos(0.47 * z + 0.19 * x) + noise(rng);
                    s.data[static_cast<std::size_t>(s.at(z, y, x))] = static_cast<float>(v + 2.0);
                }
        return s;
    }

    struct Tile {
        Ext extent{};
        std::vector<float> data;
        BufferView<const float> view() const {
            return {data.data(), Shape{extent[0], extent[1], extent[2]}, Device::cpu()};
        }
    };

    inline Tile cut(const Scene& scene, Ext origin, Ext extent) {
        Tile t;
        t.extent = extent;
        t.data.resize(static_cast<std::size_t>(extent[0] * extent[1] * extent[2]));
        for (Index z = 0; z < extent[0]; ++z)
            for (Index y = 0; y < extent[1]; ++y)
                for (Index x = 0; x < extent[2]; ++x)
                    t.data[static_cast<std::size_t>((z * extent[1] + y) * extent[2] + x)] =
                        scene.value(z + origin[0], y + origin[1], x + origin[2]);
        return t;
    }

} // namespace sirius::test::stitching

#endif // SIRIUS_TESTS_STITCHING_SCENE_HPP
