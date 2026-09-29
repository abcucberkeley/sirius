#include "core/image_encode.hpp"

#include <algorithm>
#include <climits>
#include <exception>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <utility>

// stb_image_write, private to this file. STB_IMAGE_WRITE_STATIC gives every
// function and setting internal linkage, so this copy never meets the one the
// GUI compiles in imgui/gl.cpp when both end up in sirius-app, and
// STBI_WRITE_NO_STDIO leaves out the file writers: the bytes go to memory,
// and writeBinaryFile() writes them with a UTF-8 path. The warning block is
// gl.cpp's, since the core units build with /W4 or -Wall -Wextra -Wpedantic,
// and with /WX or -Werror under SIRIUS_WARNINGS_AS_ERRORS (cmake/Warnings.cmake).
#if defined(_MSC_VER)
#pragma warning(push, 0)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#define STBI_WRITE_NO_STDIO
#include <stb_image_write.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace sirius::app {

    namespace {
        // stb's PNG compression level is a setting of this file, not an
        // argument, so two encodes at different levels must not overlap.
        std::mutex& pngMutex() {
            static std::mutex m;
            return m;
        }

        // Where stb's write callback puts the bytes. An exception must not
        // unwind through stb: the PNG writer frees its encoded buffer only
        // after the callback returns, so a failed append (out of memory)
        // would leak all of it. The first failure is kept, the rest of the
        // output is ignored, and the encoder rethrows once stb is done.
        struct Sink {
            std::vector<std::uint8_t> bytes;
            std::exception_ptr error;
        };

        void appendBytes(void* context, void* data, int size) {
            auto* sink = static_cast<Sink*>(context);
            if (sink->error) return;
            try {
                const auto* p = static_cast<const std::uint8_t*>(data);
                sink->bytes.insert(sink->bytes.end(), p, p + size);
            } catch (...) {
                sink->error = std::current_exception();
            }
        }

        std::vector<std::uint8_t> take(Sink& sink, int ok, const char* who) {
            if (sink.error) std::rethrow_exception(sink.error);
            if (!ok || sink.bytes.empty()) throw std::runtime_error(std::string(who) + ": the image could not be encoded");
            return std::move(sink.bytes);
        }

        void checkPixels(const char* who, const std::uint8_t* pixels, int width, int height) {
            if (!pixels) throw std::invalid_argument(std::string(who) + ": no pixels");
            if (width <= 0 || height <= 0)
                throw std::invalid_argument(std::string(who) + ": an empty image (" + std::to_string(width) + " x " + std::to_string(height) + ")");
        }
    } // namespace

    std::vector<std::uint8_t> encodePng(const std::uint8_t* pixels, int width, int height, int channels, int compressionLevel) {
        checkPixels("encodePng", pixels, width, height);
        if (channels != 1 && channels != 3 && channels != 4)
            throw std::invalid_argument("encodePng: " + std::to_string(channels) + " channels (1, 3 or 4 are written)");
        // stb counts in int. The filtered rows, (width * channels + 1) *
        // height bytes, are one int; the zlib stream they compress to grows
        // by doubling an int capacity (2 * m + 1), and its fixed Huffman
        // codes can spend 9 bits on a byte, so incompressible rows come out
        // up to an eighth larger. A quarter of INT_MAX keeps the doubled
        // capacity of that stream inside an int; far below it, the picture is
        // no longer something to look at.
        const long long filtered = (static_cast<long long>(width) * channels + 1) * static_cast<long long>(height);
        if (filtered > INT_MAX / 4)
            throw std::invalid_argument("encodePng: " + std::to_string(width) + " x " + std::to_string(height) + " is too large to encode");
        Sink sink;
        int ok = 0;
        {
            const std::lock_guard<std::mutex> lock(pngMutex());
            stbi_write_png_compression_level = std::clamp(compressionLevel, 0, 9);
            ok = stbi_write_png_to_func(appendBytes, &sink, width, height, channels, pixels, width * channels);
        }
        return take(sink, ok, "encodePng");
    }

    std::vector<std::uint8_t> encodeJpeg(const std::uint8_t* pixels, int width, int height, int channels, int quality) {
        checkPixels("encodeJpeg", pixels, width, height);
        if (channels != 1 && channels != 3)
            throw std::invalid_argument("encodeJpeg: " + std::to_string(channels) + " channels (1 or 3 are written)");
        // The frame header holds each dimension in 16 bits.
        if (width > 65535 || height > 65535)
            throw std::invalid_argument("encodeJpeg: " + std::to_string(width) + " x " + std::to_string(height) + " is too large for JPEG");
        Sink sink;
        const int ok = stbi_write_jpg_to_func(appendBytes, &sink, width, height, channels, pixels, std::clamp(quality, 1, 100));
        return take(sink, ok, "encodeJpeg");
    }

    std::string base64Encode(const std::vector<std::uint8_t>& bytes) {
        static const char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out;
        out.reserve((bytes.size() + 2) / 3 * 4);
        std::size_t i = 0;
        for (; i + 3 <= bytes.size(); i += 3) {
            const std::uint32_t v = (static_cast<std::uint32_t>(bytes[i]) << 16) | (static_cast<std::uint32_t>(bytes[i + 1]) << 8) | bytes[i + 2];
            out += kAlphabet[(v >> 18) & 63];
            out += kAlphabet[(v >> 12) & 63];
            out += kAlphabet[(v >> 6) & 63];
            out += kAlphabet[v & 63];
        }
        const std::size_t rest = bytes.size() - i;
        if (rest > 0) {
            std::uint32_t v = static_cast<std::uint32_t>(bytes[i]) << 16;
            if (rest == 2) v |= static_cast<std::uint32_t>(bytes[i + 1]) << 8;
            out += kAlphabet[(v >> 18) & 63];
            out += kAlphabet[(v >> 12) & 63];
            out += rest == 2 ? kAlphabet[(v >> 6) & 63] : '=';
            out += '=';
        }
        return out;
    }

    namespace {
        // Whether the bytes are well-formed UTF-8 (RFC 3629): no overlong
        // forms, no surrogates, nothing above U+10FFFF.
        bool isValidUtf8(const std::string& s) {
            const auto* p = reinterpret_cast<const unsigned char*>(s.data());
            const std::size_t n = s.size();
            std::size_t i = 0;
            while (i < n) {
                const unsigned char b = p[i];
                std::size_t len = 0;
                unsigned char lo = 0x80, hi = 0xBF;   // the range of the second byte
                if (b < 0x80) {
                    ++i;
                    continue;
                } else if (b >= 0xC2 && b <= 0xDF) {
                    len = 2;
                } else if (b >= 0xE0 && b <= 0xEF) {
                    len = 3;
                    if (b == 0xE0) lo = 0xA0;
                    if (b == 0xED) hi = 0x9F;
                } else if (b >= 0xF0 && b <= 0xF4) {
                    len = 4;
                    if (b == 0xF0) lo = 0x90;
                    if (b == 0xF4) hi = 0x8F;
                } else {
                    return false;
                }
                if (n - i < len) return false;
                if (p[i + 1] < lo || p[i + 1] > hi) return false;
                for (std::size_t k = 2; k < len; ++k)
                    if (p[i + k] < 0x80 || p[i + k] > 0xBF) return false;
                i += len;
            }
            return true;
        }
    } // namespace

    bool writeBinaryFile(const std::string& utf8Path, const std::vector<std::uint8_t>& bytes, std::string* error) {
        namespace fs = std::filesystem;
        auto fail = [error](const std::string& why) {
            if (error) *error = why;
            return false;
        };
        if (utf8Path.empty()) return fail("no path to write to");
        // The path goes into the error messages below, and those may end up
        // in JSON, which takes nothing but UTF-8 (nlohmann's dump() throws on
        // anything else). Windows would refuse such a path in u8path anyway,
        // but POSIX takes any bytes, so the check is made here on every
        // platform, and the message leaves the bytes out.
        if (!isValidUtf8(utf8Path)) return fail("cannot write to a path that is not valid UTF-8");
        // u8path: a narrow path is in the ANSI code page on Windows, and the
        // path an agent or a script hands over is UTF-8. It may still throw
        // (on a path the platform cannot represent); that is reported like
        // the other failures rather than thrown.
        fs::path path;
        try {
            path = fs::u8path(utf8Path);
        } catch (const std::exception&) {
            return fail("cannot write to that path");
        }
        std::error_code ec;
        if (path.has_parent_path()) {
            fs::create_directories(path.parent_path(), ec);
            if (ec) return fail("cannot create the folder of " + utf8Path + ": " + ec.message());
        }
        {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out) return fail("cannot write " + utf8Path);
            out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            out.close();
            if (out) return true;
        }
        // A half-written picture is worse than none: whoever reads it next
        // takes it for the whole one.
        fs::remove(path, ec);
        return fail("cannot write " + utf8Path + " (is the disk full?)");
    }

} // namespace sirius::app
