#ifndef SIRIUS_APP_IMAGE_ENCODE_HPP
#define SIRIUS_APP_IMAGE_ENCODE_HPP

// Pictures as bytes: PNG and JPEG in memory (the pinned stb_image_write,
// compiled privately into this unit), base64 for inlining them in JSON, and a
// binary file write that takes a UTF-8 path. What sirius-cli sends an agent
// or writes for a script; nothing here knows about the viewer.
//
// The encoders throw std::invalid_argument for null pixels, an empty image,
// an unsupported channel count or an image too large for the format, and
// std::runtime_error when stb gives up. writeBinaryFile reports every failure,
// a path that is not UTF-8 included, as false and a message instead.

#include <cstdint>
#include <string>
#include <vector>

namespace sirius::app {
    // 8-bit pixels, rows top to bottom, tightly packed; channels 1 (grey), 3 (RGB) or 4 (RGBA).
    std::vector<std::uint8_t> encodePng(const std::uint8_t* pixels, int width, int height, int channels, int compressionLevel = 6);
    std::vector<std::uint8_t> encodeJpeg(const std::uint8_t* pixels, int width, int height, int channels, int quality = 90);   // 1 or 3 channels
    std::string base64Encode(const std::vector<std::uint8_t>& bytes);
    bool writeBinaryFile(const std::string& utf8Path, const std::vector<std::uint8_t>& bytes, std::string* error = nullptr);  // creates parents
} // namespace sirius::app

#endif // SIRIUS_APP_IMAGE_ENCODE_HPP
