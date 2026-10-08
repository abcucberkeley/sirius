#ifndef SIRIUS_TESTS_MRC_FIXTURE_HPP
#define SIRIUS_TESTS_MRC_FIXTURE_HPP

// A DeltaVision (.dv) or MRC2014 stack written for a test: the 1024-byte
// header with the fields the reader uses, then the sections in the order the
// header's sequence says. The shipped raw.dv / otf.dv cover the real thing;
// this covers the layouts they do not (several wavelengths and time points,
// the three sequences, the integer and complex modes, big-endian files).

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::test {

    struct MrcSpec {
        std::uint32_t nx = 8, ny = 6, planes = 4;
        int waves = 1, times = 1;
        int sequence = 0;                 // 0 ZTW, 1 WZT, 2 ZWT (DeltaVision only)
        int mode = 2;                     // 0 bytes, 1 int16, 2 float32, 3 / 4 complex, 6 uint16, 7 int32
        bool bigEndian = false;
        bool deltaVision = true;          // false: an MRC2014 header (Angstrom cells, "MAP ", a machine stamp)
        float pixelX = 0.1f, pixelY = 0.1f, pixelZ = 0.3f;   // micrometres
        std::array<int, 5> wavelengths{0, 0, 0, 0, 0};
        std::uint32_t extendedHeaderBytes = 0;
        std::string title;
    };

    // value(w, t, z, y, x): the sample written; complex modes take x over 2 * nx
    // columns (real, imaginary, real, ...).
    using MrcValue = std::function<double(int w, int t, int z, int y, int x)>;

    inline void writeMrc(const std::string& path, const MrcSpec& s, const MrcValue& value) {
        std::vector<unsigned char> bytes(1024 + s.extendedHeaderBytes, 0);
        const auto put = [&](std::size_t at, const void* v, std::size_t n) {
            unsigned char tmp[8];
            std::memcpy(tmp, v, n);
            if (s.bigEndian) std::reverse(tmp, tmp + n);
            std::memcpy(bytes.data() + at, tmp, n);
        };
        const auto i32 = [&](std::size_t at, std::int32_t v) { put(at, &v, 4); };
        const auto i16 = [&](std::size_t at, std::int16_t v) { put(at, &v, 2); };
        const auto f32 = [&](std::size_t at, float v) { put(at, &v, 4); };
        const std::uint32_t sections = s.planes * static_cast<std::uint32_t>(s.waves) * static_cast<std::uint32_t>(s.times);
        i32(0, static_cast<std::int32_t>(s.nx));
        i32(4, static_cast<std::int32_t>(s.ny));
        i32(8, static_cast<std::int32_t>(sections));
        i32(12, s.mode);
        i32(64, 1);
        i32(68, 2);
        i32(72, 3);
        i32(92, static_cast<std::int32_t>(s.extendedHeaderBytes));
        if (s.deltaVision) {
            i32(28, 1);
            i32(32, 1);
            i32(36, 1);
            f32(40, s.pixelX);
            f32(44, s.pixelY);
            f32(48, s.pixelZ);
            i16(96, static_cast<std::int16_t>(0xC0A0));
            i16(128, 8);    // ints and floats per section in the extended header, as OMX writes them
            i16(130, 32);
            i16(180, static_cast<std::int16_t>(s.times));
            i16(182, static_cast<std::int16_t>(s.sequence));
            i16(196, static_cast<std::int16_t>(s.waves));
            for (std::size_t k = 0; k < 5; ++k) i16(198 + 2 * k, static_cast<std::int16_t>(s.wavelengths[k]));
        } else {
            i32(28, static_cast<std::int32_t>(s.nx));
            i32(32, static_cast<std::int32_t>(s.ny));
            i32(36, static_cast<std::int32_t>(sections));
            f32(40, s.pixelX * 1e4f * static_cast<float>(s.nx));   // Angstrom per cell
            f32(44, s.pixelY * 1e4f * static_cast<float>(s.ny));
            f32(48, s.pixelZ * 1e4f * static_cast<float>(sections));
            std::memcpy(bytes.data() + 208, "MAP ", 4);
            const unsigned char stamp[4] = {static_cast<unsigned char>(s.bigEndian ? 0x11 : 0x44), static_cast<unsigned char>(s.bigEndian ? 0x11 : 0x44), 0, 0};
            std::memcpy(bytes.data() + 212, stamp, 4);
        }
        if (!s.title.empty()) {
            i32(220, 1);
            std::string t = s.title.substr(0, 80);
            t.resize(80, ' ');
            std::memcpy(bytes.data() + 224, t.data(), 80);
        }

        const bool complex = s.mode == 3 || s.mode == 4;
        const std::uint32_t width = complex ? 2 * s.nx : s.nx;
        const std::size_t item = s.mode == 0 ? 1 : (s.mode == 1 || s.mode == 3 || s.mode == 6) ? 2 : 4;
        for (std::uint32_t k = 0; k < sections; ++k) {
            // (w, t, z) of section k by the sequence
            int w = 0, t = 0, z = 0;
            const int nz = static_cast<int>(s.planes), nw = s.waves, nt = s.times;
            const int kk = static_cast<int>(k);
            if (!s.deltaVision || s.sequence == 0) {        // ZTW
                z = kk % nz;
                t = (kk / nz) % nt;
                w = kk / (nz * nt);
            } else if (s.sequence == 1) {                   // WZT
                w = kk % nw;
                z = (kk / nw) % nz;
                t = kk / (nw * nz);
            } else {                                        // ZWT
                z = kk % nz;
                w = (kk / nz) % nw;
                t = kk / (nz * nw);
            }
            for (std::uint32_t y = 0; y < s.ny; ++y)
                for (std::uint32_t x = 0; x < width; ++x) {
                    const double v = value(w, t, z, static_cast<int>(y), static_cast<int>(x));
                    unsigned char raw[4];
                    switch (s.mode) {
                        case 0: raw[0] = static_cast<unsigned char>(static_cast<std::uint8_t>(v)); break;
                        case 1:
                        case 3: {
                            const std::int16_t i = static_cast<std::int16_t>(v);
                            std::memcpy(raw, &i, 2);
                            break;
                        }
                        case 6: {
                            const std::uint16_t u = static_cast<std::uint16_t>(v);
                            std::memcpy(raw, &u, 2);
                            break;
                        }
                        case 7: {
                            const std::int32_t i = static_cast<std::int32_t>(v);
                            std::memcpy(raw, &i, 4);
                            break;
                        }
                        default: {
                            const float f = static_cast<float>(v);
                            std::memcpy(raw, &f, 4);
                            break;
                        }
                    }
                    if (s.bigEndian) std::reverse(raw, raw + item);
                    bytes.insert(bytes.end(), raw, raw + item);
                }
        }
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("cannot write " + path);
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }

} // namespace sirius::test

#endif // SIRIUS_TESTS_MRC_FIXTURE_HPP
