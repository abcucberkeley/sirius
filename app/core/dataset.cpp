#include "core/dataset.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace sirius::app {

    std::string ChannelInfo::shortName() const {
        if (wavelengthNm > 0.0) return std::to_string(static_cast<int>(std::lround(wavelengthNm)));
        return label.empty() ? std::string("ch") : label.substr(0, 4);
    }

    std::string ChannelInfo::hexColor() const {
        char buf[8];
        auto ch = [](float v) { return static_cast<int>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); };
        std::snprintf(buf, sizeof buf, "#%02x%02x%02x", ch(color[0]), ch(color[1]), ch(color[2]));
        return buf;
    }

    std::array<float, 3> colorFromHex(const std::string& hex) {
        std::string h = hex;
        if (!h.empty() && h[0] == '#') h.erase(0, 1);
        if (h.size() != 6) throw std::invalid_argument("colorFromHex: expected #rrggbb, got " + hex);
        auto part = [&](std::size_t i) { return static_cast<float>(std::stoi(h.substr(i, 2), nullptr, 16)) / 255.0f; };
        return {part(0), part(2), part(4)};
    }

    std::array<float, 3> colorForWavelength(double nm) noexcept {
        // The design's channel palette; anything else is interpolated
        // between its neighbours so unusual lines still get a sensible hue.
        struct Stop {
            double nm;
            std::array<float, 3> c;
        };
        static const Stop stops[] = {
            {405.0, {0x7c / 255.f, 0x9c / 255.f, 0xff / 255.f}},
            {488.0, {0x63 / 255.f, 0xe0 / 255.f, 0x8a / 255.f}},
            {561.0, {0xe8 / 255.f, 0x71 / 255.f, 0xd9 / 255.f}},
            {640.0, {0xff / 255.f, 0x7a / 255.f, 0x5c / 255.f}},
        };
        if (nm <= 0.0) return {1.f, 1.f, 1.f};
        for (const Stop& s : stops)
            if (std::abs(s.nm - nm) < 25.0) return s.c;
        if (nm <= stops[0].nm) return stops[0].c;
        if (nm >= stops[3].nm) return stops[3].c;
        for (int i = 0; i < 3; ++i) {
            if (nm >= stops[i].nm && nm <= stops[i + 1].nm) {
                const float f = static_cast<float>((nm - stops[i].nm) / (stops[i + 1].nm - stops[i].nm));
                std::array<float, 3> c{};
                for (int k = 0; k < 3; ++k) c[static_cast<std::size_t>(k)] = stops[i].c[static_cast<std::size_t>(k)] * (1 - f) + stops[i + 1].c[static_cast<std::size_t>(k)] * f;
                return c;
            }
        }
        return {1.f, 1.f, 1.f};
    }

    void DatasetMeta::normalizeChannels() {
        const std::size_t n = static_cast<std::size_t>(std::max<Index>(dims.c, 1));
        if (rgb) {
            channels = {{"R", 0.0, {1.f, 0.f, 0.f}, {}}, {"G", 0.0, {0.f, 1.f, 0.f}, {}}, {"B", 0.0, {0.f, 0.f, 1.f}, {}}};
            return;
        }
        channels.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            ChannelInfo& ch = channels[i];
            if (ch.label.empty()) ch.label = "ch " + std::to_string(i);
            const bool defaultColor = ch.color[0] == 1.f && ch.color[1] == 1.f && ch.color[2] == 1.f;
            if (defaultColor) {
                if (ch.wavelengthNm > 0.0) ch.color = colorForWavelength(ch.wavelengthNm);
                else if (n > 1) {
                    static const double fallback[] = {488.0, 561.0, 405.0, 640.0};
                    ch.color = colorForWavelength(fallback[i % 4]);
                }
            }
        }
    }

    std::vector<std::array<double, 3>> DatasetMeta::tilePositionsPx() const {
        std::vector<std::array<double, 3>> out;
        out.reserve(tiles.size());
        // a lateral voxel size the file did not give borrows the other one
        // (pixels are square far more often than they are unknown); without
        // a z size the tiles stack in one plane
        const double vx = voxelUm[0] > 0 ? voxelUm[0] : voxelUm[1];
        const double vy = voxelUm[1] > 0 ? voxelUm[1] : voxelUm[0];
        const double vz = voxelUm[2];
        for (const TileInfo& t : tiles)
            out.push_back({vz > 0 ? t.positionUm[0] / vz : 0.0, vy > 0 ? t.positionUm[1] / vy : 0.0, vx > 0 ? t.positionUm[2] / vx : 0.0});
        return out;
    }

    std::string DatasetMeta::shapeString() const {
        if (rgb)
            return "rgb t" + std::to_string(dims.t) + " z" + std::to_string(dims.z) + " y" + std::to_string(dims.y) +
                   " x" + std::to_string(dims.x);
        return dims.toString();
    }

    std::string DatasetMeta::voxelString() const {
        char buf[96];
        std::snprintf(buf, sizeof buf, "%.3g × %.3g × %.3g µm", voxelUm[0], voxelUm[1], voxelUm[2]);
        return buf;
    }

} // namespace sirius::app
