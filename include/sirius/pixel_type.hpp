#ifndef SIRIUS_PIXEL_TYPE_HPP
#define SIRIUS_PIXEL_TYPE_HPP

// The pixel types a file on disk can hold, and the two facts about them that
// everything else asks for. Header-only, and on its own: the zarr reader, the
// exporters and the workbench's dataset model name a PixelType without having
// anything to do with TIFF, which is where this enum used to live.

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace sirius {

    enum class PixelType : std::uint8_t { UInt8,
                                          Int8,
                                          UInt16,
                                          Int16,
                                          UInt32,
                                          Int32,
                                          Float32,
                                          Float64 };

    constexpr std::size_t bytesPerPixel(PixelType t) noexcept {
        switch (t) {
            case PixelType::UInt8:
            case PixelType::Int8: return 1;
            case PixelType::UInt16:
            case PixelType::Int16: return 2;
            case PixelType::UInt32:
            case PixelType::Int32:
            case PixelType::Float32: return 4;
            case PixelType::Float64: return 8;
        }
        return 0;
    }

    constexpr const char* toString(PixelType t) noexcept {
        switch (t) {
            case PixelType::UInt8: return "uint8";
            case PixelType::Int8: return "int8";
            case PixelType::UInt16: return "uint16";
            case PixelType::Int16: return "int16";
            case PixelType::UInt32: return "uint32";
            case PixelType::Int32: return "int32";
            case PixelType::Float32: return "float32";
            case PixelType::Float64: return "float64";
        }
        return "unknown";
    }

    template <typename T> constexpr PixelType pixelTypeOf() {
        if constexpr (std::is_same_v<T, std::uint8_t>) return PixelType::UInt8;
        else if constexpr (std::is_same_v<T, std::int8_t>) return PixelType::Int8;
        else if constexpr (std::is_same_v<T, std::uint16_t>) return PixelType::UInt16;
        else if constexpr (std::is_same_v<T, std::int16_t>) return PixelType::Int16;
        else if constexpr (std::is_same_v<T, std::uint32_t>) return PixelType::UInt32;
        else if constexpr (std::is_same_v<T, std::int32_t>) return PixelType::Int32;
        else if constexpr (std::is_same_v<T, float>) return PixelType::Float32;
        else if constexpr (std::is_same_v<T, double>) return PixelType::Float64;
        else static_assert(sizeof(T) == 0, "unsupported pixel type");
    }

} // namespace sirius

#endif // SIRIUS_PIXEL_TYPE_HPP
