#include "sirius/tiff_io.hpp"
#include "sirius/errors.hpp"
#include "downsample.hpp"
#include "tiff_internal.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>
#include <tiffio.h>

namespace sirius {

    // anon namespace so stuff isnt seen outside the translation unit
    namespace {

        // libtiff is a c library so need to handle raw pointers
        // by using a custom deleter + unique pointer
        struct TiffDeleter {
            void operator()(TIFF* tif) const { TIFFClose(tif); }
        };
        using TiffPtr = std::unique_ptr<TIFF, TiffDeleter>;

        // Per-handle warning filter (libtiff >= 4.5). Microscopy TIFFs routinely
        // carry private tags libtiff does not know (ImageJ 50838/50839, OME, ...);
        // LabVIEW / ScanImage stacks also write ImageDescription with an embedded
        // NUL and unsorted IFD tags. Those warnings fire once per page and would
        // drown a multi-file open. Anything else falls through to libtiff's
        // global warning handler, so real warnings stay visible.
        int warningFilter(TIFF*, void*, const char*, const char* fmt, va_list) {
            if (fmt && std::strstr(fmt, "Unknown field with tag")) return 1;
            if (fmt && std::strstr(fmt, "contains null byte in value")) return 1;
            if (fmt && std::strstr(fmt, "tags are not sorted")) return 1;
            return 0;
        }

        // Per-handle error filter: a codec libtiff was built without is
        // reported by inspection (TiffImageInfo::unsupported) and by the read
        // that needs it, in SIRIUS's words; libtiff's line on stderr, printed
        // for every page of every inspection, adds nothing.
        int errorFilter(TIFF*, void*, const char*, const char* fmt, va_list) {
            if (fmt && std::strstr(fmt, "compression support is not configured")) return 1;
            return 0;
        }

        struct OpenOptionsDeleter {
            void operator()(TIFFOpenOptions* o) const { TIFFOpenOptionsFree(o); }
        };

        std::atomic<std::size_t> g_readOpens{0};

        // The file is closed when TiffPtr goes out of scope, normally or via exception.
        TiffPtr openTiff(const std::string& path, const char* mode) {
            if (mode[0] == 'r') g_readOpens.fetch_add(1, std::memory_order_relaxed);
            std::unique_ptr<TIFFOpenOptions, OpenOptionsDeleter> opts(TIFFOpenOptionsAlloc());
            if (opts) {
                TIFFOpenOptionsSetWarningHandlerExtR(opts.get(), warningFilter, nullptr);
                TIFFOpenOptionsSetErrorHandlerExtR(opts.get(), errorFilter, nullptr);
            }
            TiffPtr tif(TIFFOpenExt(path.c_str(), mode, opts.get()));
            if (!tif) throw IoError("Failed to open TIFF: " + path);
            return tif;
        }

        const char* compressionName(uint16_t c) {
            switch (c) {
                case COMPRESSION_NONE: return "None";
                case COMPRESSION_CCITTRLE: return "CCITT RLE";
                case COMPRESSION_CCITTFAX3: return "CCITT Group 3";
                case COMPRESSION_CCITTFAX4: return "CCITT Group 4";
                case COMPRESSION_LZW: return "LZW";
                case COMPRESSION_OJPEG: return "old-style JPEG";
                case COMPRESSION_JPEG: return "JPEG";
                case COMPRESSION_ADOBE_DEFLATE: return "Adobe Deflate";
                case COMPRESSION_DEFLATE: return "Deflate";
                case COMPRESSION_PACKBITS: return "PackBits";
                case 34712: return "JPEG 2000";
                case 34887: return "LERC";
                case 34925: return "LZMA";
                case 50000: return "ZSTD";
                case 50001: return "WebP";
                case 50002:
                case 52546: return "JPEG XL";
                default: return "an unknown codec";
            }
        }

        // The decoded type of one sample, or "" in `why` when there is none:
        // 1..8-bit unsigned -> UInt8, 9..16 -> UInt16, 17..32 -> UInt32; signed
        // the same with sign extension; float16 widens to Float32.
        PixelType sampleTypeFrom(uint16_t bps, uint16_t fmt, std::string& why) {
            switch (fmt) {
                case SAMPLEFORMAT_IEEEFP:
                    if (bps == 16 || bps == 32) return PixelType::Float32;
                    if (bps == 64) return PixelType::Float64;
                    why = std::to_string(bps) + "-bit floating-point samples are not supported";
                    return PixelType::Float32;
                case SAMPLEFORMAT_INT:
                    if (bps >= 1 && bps <= 8) return PixelType::Int8;
                    if (bps >= 9 && bps <= 16) return PixelType::Int16;
                    if (bps >= 17 && bps <= 32) return PixelType::Int32;
                    why = std::to_string(bps) + "-bit signed integer samples are not supported (no int64 pixel type)";
                    return PixelType::Int32;
                case SAMPLEFORMAT_COMPLEXINT:
                case SAMPLEFORMAT_COMPLEXIEEEFP:
                    why = "complex samples are not supported";
                    return PixelType::Float32;
                default:   // SAMPLEFORMAT_UINT, SAMPLEFORMAT_VOID and the (common) unspecified case
                    if (bps >= 1 && bps <= 8) return PixelType::UInt8;
                    if (bps >= 9 && bps <= 16) return PixelType::UInt16;
                    if (bps >= 17 && bps <= 32) return PixelType::UInt32;
                    why = std::to_string(bps) + "-bit unsigned integer samples are not supported (no uint64 pixel type)";
                    return PixelType::UInt32;
            }
        }

        // Metadata of the directory `tif` currently points at. Throws only
        // when the directory is not an image at all (no width / height /
        // bits); one SIRIUS cannot decode is described in `unsupported`.
        TiffImageInfo readImageInfo(TIFF* tif) {
            TiffImageInfo info;
            info.ifdOffset = TIFFCurrentDirOffset(tif);

            uint32_t subfileType = 0;
            if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &info.width))
                throw IoError("TIFF missing required tag: IMAGEWIDTH");
            if (!TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &info.height))
                throw IoError("TIFF missing required tag: IMAGELENGTH");
            if (!TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &info.bitsPerSample))
                throw IoError("TIFF missing required tag: BITSPERSAMPLE");
            TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &info.samplesPerPixel);
            TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &info.sampleFormat);
            TIFFGetFieldDefaulted(tif, TIFFTAG_COMPRESSION, &info.compression);
            // The predictor tag only exists for codecs that register it
            // (LZW/Deflate/...); for others libtiff reports nothing.
            if (!TIFFGetField(tif, TIFFTAG_PREDICTOR, &info.predictor) || info.predictor == 0)
                info.predictor = 1;
            TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &info.planarConfig);
            TIFFGetFieldDefaulted(tif, TIFFTAG_SUBFILETYPE, &subfileType);
            TIFFGetFieldDefaulted(tif, TIFFTAG_ORIENTATION, &info.orientation);
            if (!TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &info.photometric))
                info.photometric = info.samplesPerPixel >= 3 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK;
            if (info.samplesPerPixel == 0) info.samplesPerPixel = 1;
            if (info.planarConfig != PLANARCONFIG_SEPARATE) info.planarConfig = PLANARCONFIG_CONTIG;

            uint16_t extraCount = 0;
            uint16_t* extra = nullptr;
            if (TIFFGetField(tif, TIFFTAG_EXTRASAMPLES, &extraCount, &extra) && extra)
                info.extraSamples.assign(extra, extra + extraCount);
            if (info.photometric == PHOTOMETRIC_PALETTE && info.bitsPerSample <= 16) {
                uint16_t *r = nullptr, *g = nullptr, *b = nullptr;
                if (TIFFGetField(tif, TIFFTAG_COLORMAP, &r, &g, &b) && r && g && b) {
                    const std::size_t n = std::size_t{1} << info.bitsPerSample;
                    info.colormap.reserve(3 * n);
                    info.colormap.insert(info.colormap.end(), r, r + n);
                    info.colormap.insert(info.colormap.end(), g, g + n);
                    info.colormap.insert(info.colormap.end(), b, b + n);
                }
            }

            info.pixelType = sampleTypeFrom(info.bitsPerSample, info.sampleFormat, info.unsupported);
            if (info.unsupported.empty() && !TIFFIsCODECConfigured(info.compression))
                info.unsupported = std::string("compression ") + std::to_string(info.compression) + " (" +
                                   compressionName(info.compression) + ") is not built into this SIRIUS's libtiff";
            if (info.unsupported.empty() && info.photometric == PHOTOMETRIC_YCBCR &&
                info.compression != COMPRESSION_JPEG) {
                uint16_t sx = 1, sy = 1;
                TIFFGetFieldDefaulted(tif, TIFFTAG_YCBCRSUBSAMPLING, &sx, &sy);
                if (sx != 1 || sy != 1) info.unsupported = "subsampled YCbCr samples are not supported";
            }

            if (TIFFIsTiled(tif)) {
                info.layout = TiffLayout::Tiles;
                if (!TIFFGetField(tif, TIFFTAG_TILEWIDTH, &info.tileWidth) || info.tileWidth == 0)
                    throw IoError("TIFF missing or invalid TILEWIDTH");
                if (!TIFFGetField(tif, TIFFTAG_TILELENGTH, &info.tileHeight) || info.tileHeight == 0)
                    throw IoError("TIFF missing or invalid TILELENGTH");
            } else {
                info.layout = TiffLayout::Strips;
                TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &info.rowsPerStrip);
                if (info.rowsPerStrip == 0 || info.rowsPerStrip > info.height)
                    info.rowsPerStrip = info.height;
            }
            info.reducedResolution = (subfileType & FILETYPE_REDUCEDIMAGE) != 0;

            uint16_t subCount = 0;
            uint64_t* subOffsets = nullptr;
            if (TIFFGetField(tif, TIFFTAG_SUBIFD, &subCount, &subOffsets) && subOffsets)
                info.subIfds.assign(subOffsets, subOffsets + subCount);

            // Metadata the workbench reads: OME-XML / ImageJ descriptions and
            // the pixel size. Cheap when absent (TIFFGetField returns 0).
            const char* description = nullptr;
            if (TIFFGetField(tif, TIFFTAG_IMAGEDESCRIPTION, &description) && description)
                info.description = description;
            float xres = 0.0f, yres = 0.0f;
            if (TIFFGetField(tif, TIFFTAG_XRESOLUTION, &xres)) info.xResolution = xres;
            if (TIFFGetField(tif, TIFFTAG_YRESOLUTION, &yres)) info.yResolution = yres;
            uint16_t unit = RESUNIT_INCH;
            TIFFGetFieldDefaulted(tif, TIFFTAG_RESOLUTIONUNIT, &unit);
            info.resolutionUnit = unit;
            return info;
        }

        // map types to TIFF tags
        template <typename T>
        constexpr uint16_t sampleFormat() {
            if constexpr (std::is_floating_point_v<T>) return SAMPLEFORMAT_IEEEFP;
            else if constexpr (std::is_unsigned_v<T>) return SAMPLEFORMAT_UINT;
            else return SAMPLEFORMAT_INT;
        }

        uint16_t mapCompression(TiffCompression comp) {
            switch (comp) {
                case TiffCompression::Lzw: return COMPRESSION_LZW;
                case TiffCompression::Deflate: return COMPRESSION_ADOBE_DEFLATE;
                default: return COMPRESSION_NONE;
            }
        }

        // ------------------------------------------------------------------
        // Decoding. libtiff turns a strip or tile (a "chunk") into raw rows:
        // codec and predictor undone, multi-byte samples in host byte order,
        // samples of a pixel side by side (contiguous planar configuration) or
        // one sample per chunk (separate planes). Everything after that --
        // unpacking 1..32-bit samples, widening float16, splitting samples
        // into planes, cropping to the region and converting to the caller's
        // type -- happens here, row by row, straight into the destination.
        // The libtiff-facing code is untemplated; the per-row kernels are
        // chosen once per read.
        // ------------------------------------------------------------------

        [[noreturn]] void throwReadError(const char* what, uint32_t x, uint32_t y, uint16_t sample) {
            throw IoError(std::string("Failed to read TIFF ") + what + " at (" + std::to_string(x) + "," +
                          std::to_string(y) + ")" + (sample ? " of sample " + std::to_string(sample) : std::string()));
        }

        // IEEE half -> float, exactly (subnormals, inf and NaN included).
        float halfToFloat(uint16_t h) {
            const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
            uint32_t exp = (h >> 10) & 0x1Fu;
            uint32_t mant = h & 0x3FFu;
            uint32_t bits = 0;
            if (exp == 0x1F) {
                bits = sign | 0x7F800000u | (mant << 13);
            } else if (exp != 0) {
                bits = sign | ((exp + 112u) << 23) | (mant << 13);
            } else if (mant != 0) {   // subnormal: normalize
                exp = 113;
                while ((mant & 0x400u) == 0) {
                    mant <<= 1;
                    --exp;
                }
                bits = sign | (exp << 23) | ((mant & 0x3FFu) << 13);
            } else {
                bits = sign;
            }
            float f;
            std::memcpy(&f, &bits, sizeof f);
            return f;
        }

        bool hostLittleEndian() noexcept {
            const uint16_t one = 1;
            uint8_t first = 0;
            std::memcpy(&first, &one, 1);
            return first == 1;
        }

        // How the samples of one chunk row are stored.
        enum class SampleCoding : std::uint8_t {
            Aligned,   // 8/16/32/64-bit samples, already in the decoded pixel type
            Half,      // 16-bit float
            Int24,     // 24-bit integers, host byte order (libtiff swaps them)
            Packed     // any other width: an MSB-first bit stream, rows padded to a byte
        };

        SampleCoding codingOf(const TiffImageInfo& g) {
            if (g.sampleFormat == SAMPLEFORMAT_IEEEFP && g.bitsPerSample == 16) return SampleCoding::Half;
            if (g.bitsPerSample == 8 || g.bitsPerSample == 16 || g.bitsPerSample == 32 || g.bitsPerSample == 64)
                return SampleCoding::Aligned;
            if (g.bitsPerSample == 24) return SampleCoding::Int24;
            return SampleCoding::Packed;
        }

        // Sample `s` of pixels [px, px + count) of a raw row with `stride`
        // samples per pixel, as the decoded type N (written to `out`).
        template <typename N>
        void extractRow(const uint8_t* raw, SampleCoding coding, uint16_t bps, bool isSigned, uint32_t px,
                        uint32_t count, uint16_t stride, uint16_t s, N* out) {
            switch (coding) {
                case SampleCoding::Aligned: {
                    const N* src = reinterpret_cast<const N*>(raw) + static_cast<std::size_t>(px) * stride + s;
                    if (stride == 1) {
                        std::memcpy(out, src, static_cast<std::size_t>(count) * sizeof(N));
                    } else {
                        // unaligned rows are possible (odd tile widths of 3-sample 8-bit data
                        // never are, but stay safe): copy byte-wise per sample
                        for (uint32_t i = 0; i < count; ++i) std::memcpy(out + i, src + static_cast<std::size_t>(i) * stride, sizeof(N));
                    }
                    return;
                }
                case SampleCoding::Half:
                    if constexpr (std::is_same_v<N, float>) {
                        const uint8_t* src = raw + (static_cast<std::size_t>(px) * stride + s) * 2;
                        for (uint32_t i = 0; i < count; ++i) {
                            uint16_t h;
                            std::memcpy(&h, src + static_cast<std::size_t>(i) * stride * 2, 2);
                            out[i] = halfToFloat(h);
                        }
                    }
                    return;
                case SampleCoding::Int24:
                    if constexpr (sizeof(N) == 4 && std::is_integral_v<N>) {
                        const uint8_t* src = raw + (static_cast<std::size_t>(px) * stride + s) * 3;
                        for (uint32_t i = 0; i < count; ++i) {
                            const uint8_t* b = src + static_cast<std::size_t>(i) * stride * 3;
                            uint32_t v = 0;
                            if (hostLittleEndian())
                                v = uint32_t{b[0]} | (uint32_t{b[1]} << 8) | (uint32_t{b[2]} << 16);
                            else
                                v = (uint32_t{b[0]} << 16) | (uint32_t{b[1]} << 8) | uint32_t{b[2]};
                            if (isSigned && (v & 0x800000u)) v |= 0xFF000000u;
                            out[i] = static_cast<N>(v);
                        }
                    }
                    return;
                case SampleCoding::Packed: {
                    if constexpr (std::is_integral_v<N>) {
                        const uint64_t mask = (uint64_t{1} << bps) - 1;
                        uint64_t bit = (static_cast<uint64_t>(px) * stride + s) * bps;
                        const uint64_t step = static_cast<uint64_t>(stride) * bps;
                        for (uint32_t i = 0; i < count; ++i, bit += step) {
                            // up to 32 bits starting anywhere in a byte span 5 bytes
                            const uint8_t* b = raw + (bit >> 3);
                            const unsigned shift = static_cast<unsigned>(bit & 7);
                            const unsigned nbytes = (shift + bps + 7) / 8;
                            uint64_t acc = 0;
                            for (unsigned k = 0; k < nbytes; ++k) acc = (acc << 8) | b[k];
                            uint64_t v = (acc >> (nbytes * 8 - shift - bps)) & mask;
                            if (isSigned && (v >> (bps - 1)) & 1) v |= ~mask;   // sign-extend
                            out[i] = static_cast<N>(v);   // modular: keeps the sign bits
                        }
                    }
                    return;
                }
            }
        }

        // extractRow for a native type known at run time.
        using ExtractFn = void (*)(const uint8_t*, SampleCoding, uint16_t, bool, uint32_t, uint32_t, uint16_t, uint16_t,
                                   void*);
        template <typename N>
        void extractRowErased(const uint8_t* raw, SampleCoding coding, uint16_t bps, bool isSigned, uint32_t px,
                              uint32_t count, uint16_t stride, uint16_t s, void* out) {
            extractRow<N>(raw, coding, bps, isSigned, px, count, stride, s, static_cast<N*>(out));
        }

        // Elementwise conversion of one row segment (the same scalar
        // conversion as sirius::convert).
        using ConvertFn = void (*)(const void*, void*, std::size_t);
        template <typename From, typename To>
        void convertRowErased(const void* src, void* dst, std::size_t n) {
            const From* s = static_cast<const From*>(src);
            To* d = static_cast<To*>(dst);
            for (std::size_t i = 0; i < n; ++i) d[i] = detail::convertScalar<To>(s[i]);
        }

        template <typename F>
        void withType(PixelType t, F&& f) {
            switch (t) {
                case PixelType::UInt8: f(std::uint8_t{}); return;
                case PixelType::Int8: f(std::int8_t{}); return;
                case PixelType::UInt16: f(std::uint16_t{}); return;
                case PixelType::Int16: f(std::int16_t{}); return;
                case PixelType::UInt32: f(std::uint32_t{}); return;
                case PixelType::Int32: f(std::int32_t{}); return;
                case PixelType::Float32: f(float{}); return;
                case PixelType::Float64: f(double{}); return;
            }
        }

        ExtractFn extractFor(PixelType native) {
            ExtractFn fn = nullptr;
            withType(native, [&](auto tag) { fn = &extractRowErased<decltype(tag)>; });
            return fn;
        }

        ConvertFn convertFor(PixelType from, PixelType to) {
            ConvertFn fn = nullptr;
            withType(from, [&](auto f) {
                withType(to, [&](auto t) { fn = &convertRowErased<decltype(f), decltype(t)>; });
            });
            return fn;
        }

        // The strips / tiles of one IFD, as the region read needs them.
        struct ChunkGrid {
            bool tiled = false;
            uint32_t chunkW = 0, chunkH = 0;     // tile size, or (width, rows per strip)
            uint32_t across = 1, down = 1;       // chunks per row / column of one plane
            uint16_t chunkSamples = 1;           // samples per pixel inside a chunk
            uint16_t planes = 1;                 // separate planes: samplesPerPixel, else 1
            std::size_t rowBytes = 0;            // one decoded chunk row
            // the region in chunk coordinates
            uint32_t cx0 = 0, cx1 = 0, cy0 = 0, cy1 = 0;
            uint16_t s0 = 0, s1 = 1;             // planes to decode (separate planes only)

            std::size_t count() const noexcept {
                return static_cast<std::size_t>(cx1 - cx0) * (cy1 - cy0) * (s1 - s0);
            }
        };

        ChunkGrid gridOf(const TiffImageInfo& g, const Region& r, uint16_t firstSample, uint16_t sampleCount) {
            ChunkGrid c;
            c.tiled = g.layout == TiffLayout::Tiles;
            c.chunkW = c.tiled ? g.tileWidth : g.width;
            c.chunkH = c.tiled ? g.tileHeight : std::max<uint32_t>(1, std::min(g.rowsPerStrip, g.height));
            c.across = c.tiled ? (g.width + c.chunkW - 1) / c.chunkW : 1;
            c.down = (g.height + c.chunkH - 1) / c.chunkH;
            const bool separate = g.planarConfig == PLANARCONFIG_SEPARATE && g.samplesPerPixel > 1;
            c.chunkSamples = separate ? 1 : g.samplesPerPixel;
            c.planes = separate ? g.samplesPerPixel : 1;
            c.rowBytes = (static_cast<std::size_t>(c.chunkW) * c.chunkSamples * g.bitsPerSample + 7) / 8;
            c.cx0 = r.x / c.chunkW;
            c.cx1 = (r.x + r.width + c.chunkW - 1) / c.chunkW;
            c.cy0 = r.y / c.chunkH;
            c.cy1 = (r.y + r.height + c.chunkH - 1) / c.chunkH;
            if (separate) {
                c.s0 = firstSample;
                c.s1 = static_cast<uint16_t>(firstSample + sampleCount);
            }
            return c;
        }

        // Decoded bytes of one page of a region read: every chunk it touches,
        // whole. A measure of the work, not of the output.
        std::size_t decodedBytes(const ChunkGrid& c) {
            return c.count() * c.rowBytes * c.chunkH;
        }

        // One read, as every decoding thread sees it.
        struct ReadPlan {
            const std::vector<const TiffImageInfo*>* images = nullptr;   // per page
            const std::vector<std::uint64_t>* ifds = nullptr;
            Region region;
            uint16_t firstSample = 0, sampleCount = 1;
            PixelType nativeType = PixelType::UInt8;
            PixelType dstType = PixelType::UInt8;
            std::uint8_t* dst = nullptr;
            std::size_t dstBpp = 1;
            ExtractFn extract = nullptr;
            ConvertFn convert = nullptr;   // null: dst is the native type

            uint8_t* dstRow(std::size_t page, uint16_t sample, uint32_t y, uint32_t x) const {
                const std::size_t plane = page * sampleCount + (sample - firstSample);
                return dst + ((plane * region.height + (y - region.y)) * region.width + (x - region.x)) * dstBpp;
            }
        };

        // A thread's libtiff handle, the directory it is on, and its buffers.
        struct ChunkReader {
            const std::string* path = nullptr;
            TiffPtr tif;
            std::uint64_t dir = ~std::uint64_t{0};
            std::vector<uint8_t> chunk;
            std::vector<uint8_t> row;   // native row segment, conversion path only

            TIFF* at(std::uint64_t ifd) {
                if (!tif) tif = openTiff(*path, "r");
                if (dir != ifd) {
                    dir = ~std::uint64_t{0};
                    if (!TIFFSetSubDirectory(tif.get(), ifd))
                        throw IoError("Failed to seek to TIFF directory at offset " + std::to_string(ifd));
                    dir = ifd;
                }
                return tif.get();
            }
        };

        // Chunks [k0, k1) of page `page`'s region, in the order of the grid
        // (plane, chunk row, chunk column).
        void decodeChunks(ChunkReader& rd, const ReadPlan& plan, std::size_t page, std::size_t k0, std::size_t k1) {
            const TiffImageInfo& g = *(*plan.images)[page];
            const Region& r = plan.region;
            TIFF* tif = rd.at((*plan.ifds)[page]);
            const ChunkGrid c = gridOf(g, r, plan.firstSample, plan.sampleCount);
            const SampleCoding coding = codingOf(g);
            const bool isSigned = g.sampleFormat == SAMPLEFORMAT_INT;
            const std::size_t nativeBpp = bytesPerPixel(plan.nativeType);
            const uint32_t gx = c.cx1 - c.cx0, gy = c.cy1 - c.cy0;
            const tmsize_t chunkBytes = c.tiled ? TIFFTileSize(tif) : TIFFStripSize(tif);
            if (chunkBytes <= 0) throw IoError(std::string("TIFF reports an invalid ") + (c.tiled ? "tile" : "strip") + " size");
            const uint32_t chunksPerPlane = c.across * c.down;

            for (std::size_t k = k0; k < k1; ++k) {
                const uint16_t plane = static_cast<uint16_t>(c.s0 + k / (static_cast<std::size_t>(gx) * gy));
                const std::size_t inPlane = k % (static_cast<std::size_t>(gx) * gy);
                const uint32_t cy = c.cy0 + static_cast<uint32_t>(inPlane / gx);
                const uint32_t cx = c.cx0 + static_cast<uint32_t>(inPlane % gx);
                const uint32_t x0 = cx * c.chunkW, y0 = cy * c.chunkH;
                const uint32_t ix0 = std::max(x0, r.x), ix1 = std::min({x0 + c.chunkW, r.x + r.width, g.width});
                const uint32_t iy0 = std::max(y0, r.y), iy1 = std::min({y0 + c.chunkH, r.y + r.height, g.height});
                const uint32_t chunkIndex = static_cast<uint32_t>(plane) * chunksPerPlane + cy * c.across + cx;
                // the samples this chunk holds that the read wants
                const uint16_t sFirst = c.planes > 1 ? plane : plan.firstSample;
                const uint16_t sEnd = c.planes > 1 ? static_cast<uint16_t>(plane + 1)
                                                   : static_cast<uint16_t>(plan.firstSample + plan.sampleCount);
                const std::size_t n = ix1 - ix0;

                // Sparse files: a chunk never written (offset or byte count 0)
                // reads as zeros, as tifffile and GDAL read it.
                if (TIFFGetStrileByteCount(tif, chunkIndex) == 0 || TIFFGetStrileOffset(tif, chunkIndex) == 0) {
                    for (uint16_t s = sFirst; s < sEnd; ++s)
                        for (uint32_t y = iy0; y < iy1; ++y)
                            std::memset(plan.dstRow(page, s, y, ix0), 0, n * plan.dstBpp);
                    continue;
                }

                // Strips covering the full width of a one-sample-per-chunk
                // read decode straight into the destination when no sample
                // needs unpacking or converting.
                const bool direct = !c.tiled && c.chunkSamples == 1 && coding == SampleCoding::Aligned && !plan.convert &&
                                    r.x == 0 && r.width == g.width && y0 >= r.y;
                if (direct) {
                    const tmsize_t bytes = static_cast<tmsize_t>(iy1 - y0) * static_cast<tmsize_t>(c.rowBytes);
                    if (TIFFReadEncodedStrip(tif, chunkIndex, plan.dstRow(page, sFirst, y0, 0), bytes) < 0)
                        throwReadError("strip", 0, y0, plane);
                    continue;
                }

                rd.chunk.resize(static_cast<std::size_t>(chunkBytes));
                if (c.tiled) {
                    if (TIFFReadEncodedTile(tif, chunkIndex, rd.chunk.data(), chunkBytes) < 0)
                        throwReadError("tile", x0, y0, plane);
                } else {
                    // decode only through the last row needed
                    const tmsize_t bytes = static_cast<tmsize_t>(iy1 - y0) * static_cast<tmsize_t>(c.rowBytes);
                    if (TIFFReadEncodedStrip(tif, chunkIndex, rd.chunk.data(), bytes) < 0)
                        throwReadError("strip", 0, y0, plane);
                }
                if (plan.convert) rd.row.resize(n * nativeBpp);
                for (uint16_t s = sFirst; s < sEnd; ++s) {
                    const uint16_t sInChunk = c.planes > 1 ? 0 : s;
                    for (uint32_t y = iy0; y < iy1; ++y) {
                        const uint8_t* raw = rd.chunk.data() + static_cast<std::size_t>(y - y0) * c.rowBytes;
                        uint8_t* out = plan.dstRow(page, s, y, ix0);
                        if (plan.convert) {
                            plan.extract(raw, coding, g.bitsPerSample, isSigned, ix0 - x0, static_cast<uint32_t>(n),
                                         c.chunkSamples, sInChunk, rd.row.data());
                            plan.convert(rd.row.data(), out, n);
                        } else {
                            plan.extract(raw, coding, g.bitsPerSample, isSigned, ix0 - x0, static_cast<uint32_t>(n),
                                         c.chunkSamples, sInChunk, out);
                        }
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Writing. One code path serves every writer: a page is described
        // by its pixels and the options, written as strips or tiles, and
        // optionally followed by its reduced-resolution SubIFDs (a pyramid).
        // ------------------------------------------------------------------

        struct PageTags {
            bool page = false;          // FILETYPE_PAGE (multi-page stacks)
            bool reduced = false;       // FILETYPE_REDUCEDIMAGE (pyramid level)
            uint16_t subIfds = 0;       // SubIFD slots to reserve after this directory
            const std::string* description = nullptr;
        };

        template <typename T>
        void setPageTags(TIFF* tif, uint32_t height, uint32_t width, const TiffWriteOptions& o, const PageTags& tags) {
            TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
            TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
            TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, static_cast<uint16_t>(sizeof(T) * 8));
            TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, static_cast<uint16_t>(1));
            TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, sampleFormat<T>());
            TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
            TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

            const uint16_t comp = mapCompression(o.compression);
            TIFFSetField(tif, TIFFTAG_COMPRESSION, comp);
            if (comp == COMPRESSION_LZW || comp == COMPRESSION_ADOBE_DEFLATE) {
                if (o.predictor) {
                    if constexpr (std::is_integral_v<T>) TIFFSetField(tif, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
                    else TIFFSetField(tif, TIFFTAG_PREDICTOR, PREDICTOR_FLOATINGPOINT);
                }
                if (comp == COMPRESSION_ADOBE_DEFLATE)
                    TIFFSetField(tif, TIFFTAG_ZIPQUALITY, std::clamp(o.compressionLevel, 1, 9));
            }

            if (o.tiled) {
                // libtiff requires tile edges that are multiples of 16
                const uint32_t tw = std::max<uint32_t>(16, (o.tileWidth + 15) / 16 * 16);
                const uint32_t th = std::max<uint32_t>(16, (o.tileHeight + 15) / 16 * 16);
                TIFFSetField(tif, TIFFTAG_TILEWIDTH, tw);
                TIFFSetField(tif, TIFFTAG_TILELENGTH, th);
            } else {
                const uint32_t rps = o.rowsPerStrip > 0 ? std::min(o.rowsPerStrip, height) : TIFFDefaultStripSize(tif, 0);
                TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, rps);
            }

            uint32_t subfile = 0;
            if (tags.page) subfile |= FILETYPE_PAGE;
            if (tags.reduced) subfile |= FILETYPE_REDUCEDIMAGE;
            TIFFSetField(tif, TIFFTAG_SUBFILETYPE, subfile);

            if (tags.description && !tags.description->empty())
                TIFFSetField(tif, TIFFTAG_IMAGEDESCRIPTION, tags.description->c_str());
            if (o.xPixelUm > 0.0 && o.yPixelUm > 0.0) {
                // pixels per centimetre: 1 cm = 1e4 um
                TIFFSetField(tif, TIFFTAG_RESOLUTIONUNIT, RESUNIT_CENTIMETER);
                TIFFSetField(tif, TIFFTAG_XRESOLUTION, static_cast<float>(1e4 / o.xPixelUm));
                TIFFSetField(tif, TIFFTAG_YRESOLUTION, static_cast<float>(1e4 / o.yPixelUm));
            }
            if (tags.subIfds > 0) {
                // Reserving slots makes libtiff link the next `subIfds`
                // directories written as this page's SubIFDs (pyramid levels)
                // instead of appending them to the main chain.
                std::vector<uint64_t> zeros(tags.subIfds, 0);
                TIFFSetField(tif, TIFFTAG_SUBIFD, tags.subIfds, zeros.data());
            }
        }

        // libtiff may modify the buffers it encodes (byte swapping), so the
        // caller's const pixels always go through `scratch`.
        template <typename T>
        void writePixels(TIFF* tif, const T* src, uint32_t height, uint32_t width, std::vector<T>& scratch) {
            if (TIFFIsTiled(tif)) {
                uint32_t tw = 0, th = 0;
                TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tw);
                TIFFGetField(tif, TIFFTAG_TILELENGTH, &th);
                scratch.resize(static_cast<std::size_t>(tw) * th);
                for (uint32_t ty = 0; ty < height; ty += th)
                    for (uint32_t tx = 0; tx < width; tx += tw) {
                        const uint32_t rows = std::min(th, height - ty), cols = std::min(tw, width - tx);
                        if (rows < th || cols < tw) std::fill(scratch.begin(), scratch.end(), T{});
                        for (uint32_t r = 0; r < rows; ++r)
                            std::memcpy(scratch.data() + static_cast<std::size_t>(r) * tw,
                                        src + static_cast<std::size_t>(ty + r) * width + tx, cols * sizeof(T));
                        if (TIFFWriteTile(tif, scratch.data(), tx, ty, 0, 0) < 0)
                            throw IoError("Failed to write TIFF tile at (" + std::to_string(tx) + "," +
                                          std::to_string(ty) + ")");
                    }
            } else {
                uint32_t rps = 0;
                TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rps);
                if (rps == 0 || rps > height) rps = height;
                scratch.resize(static_cast<std::size_t>(rps) * width);
                for (uint32_t y = 0; y < height; y += rps) {
                    const uint32_t rows = std::min(rps, height - y);
                    const std::size_t bytes = static_cast<std::size_t>(rows) * width * sizeof(T);
                    std::memcpy(scratch.data(), src + static_cast<std::size_t>(y) * width, bytes);
                    const tstrip_t strip = TIFFComputeStrip(tif, y, 0);
                    if (TIFFWriteEncodedStrip(tif, strip, scratch.data(), static_cast<tmsize_t>(bytes)) < 0)
                        throw IoError("Failed to write TIFF strip at row " + std::to_string(y));
                }
            }
        }

        // Box down-sampling of a (rows, cols) plane by `f` in both axes;
        // partial boxes at the far edges average what is inside them.
        template <typename T>
        void downsamplePlane(const T* src, uint32_t rows, uint32_t cols, int f, std::vector<T>& dst,
                             uint32_t& outRows, uint32_t& outCols) {
            outRows = static_cast<uint32_t>(detail::downsampledExtent(rows, f));
            outCols = static_cast<uint32_t>(detail::downsampledExtent(cols, f));
            dst.resize(static_cast<std::size_t>(outRows) * outCols);
            detail::downsampleBoxMean<T>(src, {Index{rows}, Index{cols}}, {f, f}, dst.data());
        }

        template <typename T>
        void writePages(const std::string& path, const T* data, Index pages, Index rows, Index cols,
                        const TiffWriteOptions& o, bool pageFlag) {
            if (pages <= 0 || rows <= 0 || cols <= 0) throw std::runtime_error("Cannot write empty stack");
            const int levels = std::max(o.pyramidLevels, 1);
            const int f = std::max(o.downsample, 2);
            auto tif = openTiff(path, o.bigTiff ? "w8" : "w");
            std::vector<T> scratch;
            std::vector<T> level, nextLevel;
            const Index stride = rows * cols;
            for (Index z = 0; z < pages; ++z) {
                if (o.cancelled && o.cancelled()) {
                    tif.reset();
                    std::remove(path.c_str());
                    throw std::runtime_error("cancelled");
                }
                PageTags tags;
                tags.page = pageFlag;
                tags.subIfds = static_cast<uint16_t>(levels - 1);
                if (z == 0) tags.description = &o.description;
                setPageTags<T>(tif.get(), static_cast<uint32_t>(rows), static_cast<uint32_t>(cols), o, tags);
                writePixels<T>(tif.get(), data + z * stride, static_cast<uint32_t>(rows), static_cast<uint32_t>(cols), scratch);
                if (!TIFFWriteDirectory(tif.get()))
                    throw IoError("Failed to finalize TIFF directory for page " + std::to_string(z));

                // reduced-resolution levels, each from the previous one
                const T* srcLevel = data + z * stride;
                uint32_t lr = static_cast<uint32_t>(rows), lc = static_cast<uint32_t>(cols);
                for (int k = 1; k < levels; ++k) {
                    uint32_t nr = 0, nc = 0;
                    downsamplePlane<T>(srcLevel, lr, lc, f, nextLevel, nr, nc);
                    std::swap(level, nextLevel);
                    srcLevel = level.data();
                    lr = nr;
                    lc = nc;
                    PageTags ltags;
                    ltags.reduced = true;
                    setPageTags<T>(tif.get(), lr, lc, o, ltags);
                    writePixels<T>(tif.get(), srcLevel, lr, lc, scratch);
                    if (!TIFFWriteDirectory(tif.get()))
                        throw IoError("Failed to finalize TIFF pyramid level " + std::to_string(k) +
                                      " of page " + std::to_string(z));
                }
                if (o.progress) o.progress(static_cast<double>(z + 1) / static_cast<double>(pages));
            }
        }

        // Host copy of a view that may live on a device (writers are host-only).
        template <typename T>
        Buffer<T> onHost(BufferView<const T> v) {
            Buffer<T> h(v.shape(), Device::cpu());
            copy(v, h);            // synchronous: pageable destination
            return h;
        }

        Shape stackShape(const TiffInfo& info) {
            return Shape{static_cast<Index>(info.pageCount()), static_cast<Index>(info.height()),
                         static_cast<Index>(info.width())};
        }

        Shape levelShape(const TiffLevel& level) {
            return Shape{static_cast<Index>(level.ifds.size()), static_cast<Index>(level.height),
                         static_cast<Index>(level.width)};
        }

    } // anonymous namespace

    // --- metadata ------------------------------------------------------------

    const TiffImageInfo& TiffInfo::image(std::uint64_t ifdOffset) const {
        const auto it = imageIndex.find(ifdOffset);
        if (it != imageIndex.end() && it->second < images.size() && images[it->second].ifdOffset == ifdOffset)
            return images[it->second];
        for (const auto& i : images)
            if (i.ifdOffset == ifdOffset) return i;
        throw std::out_of_range("TIFF has no image directory at offset " + std::to_string(ifdOffset));
    }

    bool TiffInfo::uniformPages() const noexcept {
        if (pages.empty()) return false;
        const auto& p0 = page(0);
        for (std::size_t i = 1; i < pages.size(); ++i) {
            const auto& p = page(i);
            if (p.width != p0.width || p.height != p0.height || p.pixelType != p0.pixelType ||
                p.samplesPerPixel != p0.samplesPerPixel)
                return false;
        }
        return true;
    }

    Region Region::resolve(std::uint32_t imageWidth, std::uint32_t imageHeight) const {
        if (x >= imageWidth || y >= imageHeight)
            throw std::out_of_range("Region origin (" + std::to_string(x) + "," + std::to_string(y) +
                                    ") lies outside a " + std::to_string(imageWidth) + "x" +
                                    std::to_string(imageHeight) + " image");
        Region r = *this;
        if (r.width == 0) r.width = imageWidth - x;
        if (r.height == 0) r.height = imageHeight - y;
        if (static_cast<std::uint64_t>(x) + r.width > imageWidth ||
            static_cast<std::uint64_t>(y) + r.height > imageHeight)
            throw std::out_of_range("Region " + std::to_string(r.width) + "x" + std::to_string(r.height) +
                                    " at (" + std::to_string(x) + "," + std::to_string(y) + ") exceeds a " +
                                    std::to_string(imageWidth) + "x" + std::to_string(imageHeight) + " image");
        return r;
    }

    TiffInfo inspectTiff(const std::string& path) {
        auto tif = openTiff(path, "r");
        TiffInfo info;
        info.bigTiff = TIFFIsBigTIFF(tif.get()) != 0;
        info.bigEndian = TIFFIsBigEndian(tif.get()) != 0;

        // Walk the main IFD chain sequentially: directories are a linked list,
        // so this is the only O(n) way to see them all. Offsets are cached so
        // later decodes seek in O(1) with TIFFSetSubDirectory.
        do {
            info.images.push_back(readImageInfo(tif.get()));
        } while (TIFFReadDirectory(tif.get()));
        const std::size_t chainCount = info.images.size();

        // SubIFDs (pyramid levels hanging off a page) are not on the chain.
        for (std::size_t i = 0; i < chainCount; ++i) {
            for (std::uint64_t off : info.images[i].subIfds) {
                if (!TIFFSetSubDirectory(tif.get(), off))
                    throw IoError("Failed to read SubIFD at offset " + std::to_string(off) + " in " + path);
                info.images.push_back(readImageInfo(tif.get()));
            }
        }

        // `images` is complete: index it before the level discovery below and
        // every later image() lookup.
        info.imageIndex.reserve(info.images.size());
        for (std::size_t i = 0; i < info.images.size(); ++i)
            info.imageIndex.emplace(info.images[i].ifdOffset, i);

        for (std::size_t i = 0; i < chainCount; ++i)
            if (!info.images[i].reducedResolution) info.pages.push_back(info.images[i].ifdOffset);
        const bool chainIsPages = info.pages.empty();
        if (chainIsPages)   // every IFD flagged reduced: treat the chain as pages anyway
            for (std::size_t i = 0; i < chainCount; ++i) info.pages.push_back(info.images[i].ifdOffset);

        // Level 0: the full-resolution pages.
        {
            TiffLevel l0;
            l0.width = info.page(0).width;
            l0.height = info.page(0).height;
            l0.ifds = info.pages;
            info.levels.push_back(std::move(l0));
        }

        // Levels from SubIFDs: the k-th SubIFD of every page forms level k+1,
        // provided every page has one and they agree in size.
        for (std::size_t k = 0;; ++k) {
            TiffLevel level;
            bool complete = true;
            for (std::uint64_t pageOff : info.pages) {
                const auto& p = info.image(pageOff);
                if (p.subIfds.size() <= k) {
                    complete = false;
                    break;
                }
                const auto& sub = info.image(p.subIfds[k]);
                if (level.ifds.empty()) {
                    level.width = sub.width;
                    level.height = sub.height;
                } else if (sub.width != level.width || sub.height != level.height) {
                    complete = false;
                    break;
                }
                level.ifds.push_back(sub.ifdOffset);
            }
            if (!complete || level.ifds.empty()) break;
            info.levels.push_back(std::move(level));
        }

        // Levels from reduced-resolution IFDs on the main chain (GDAL/Aperio
        // style): the reduced IFDs of one size form a level, in chain order,
        // one per page -- whether each page is followed by its own reductions
        // (page 0, its 1/2, its 1/4, page 1, its 1/2, ...) or the reductions
        // come level by level after the pages. Grouping only consecutive IFDs
        // split the first layout into a level per IFD. A chain that is all
        // reduced IFDs already serves as the pages and forms no levels.
        const std::size_t firstChainLevel = info.levels.size();
        for (std::size_t i = 0; i < chainCount && !chainIsPages; ++i) {
            const auto& img = info.images[i];
            if (!img.reducedResolution) continue;
            const auto level = std::find_if(info.levels.begin() + static_cast<std::ptrdiff_t>(firstChainLevel),
                                            info.levels.end(), [&](const TiffLevel& l) {
                                                return l.width == img.width && l.height == img.height &&
                                                       l.ifds.size() < info.pages.size();
                                            });
            if (level != info.levels.end()) {
                level->ifds.push_back(img.ifdOffset);
            } else {
                TiffLevel added;
                added.width = img.width;
                added.height = img.height;
                added.ifds.push_back(img.ifdOffset);
                info.levels.push_back(std::move(added));
            }
        }
        return info;
    }

    TiffStackShape inspectTiffShape(const std::string& path) {
        auto tif = openTiff(path, "r");
        const TiffImageInfo first = readImageInfo(tif.get());
        TiffStackShape s;
        s.width = first.width;
        s.height = first.height;
        s.pixelType = first.pixelType;
        s.samplesPerPixel = first.samplesPerPixel;
        // Directory count only (next-IFD links), not every tag of every page.
        const tdir_t n = TIFFNumberOfDirectories(tif.get());
        s.pages = n > 0 ? static_cast<std::size_t>(n) : 1;
        return s;
    }

    // --- type-erased conversion ------------------------------------------------

    namespace detail {

        void convertPixels(const void* src, PixelType srcType, void* dst, PixelType dstType, Index n,
                           Device device, const Stream& stream) {
            const Shape shape{n};
            auto toAll = [&](auto fromTag) {
                using From = decltype(fromTag);
                BufferView<const From> s(static_cast<const From*>(src), shape, device);
                switch (dstType) {
                    case PixelType::UInt8: convert<From, std::uint8_t>(s, BufferView<std::uint8_t>(static_cast<std::uint8_t*>(dst), shape, device), stream); break;
                    case PixelType::Int8: convert<From, std::int8_t>(s, BufferView<std::int8_t>(static_cast<std::int8_t*>(dst), shape, device), stream); break;
                    case PixelType::UInt16: convert<From, std::uint16_t>(s, BufferView<std::uint16_t>(static_cast<std::uint16_t*>(dst), shape, device), stream); break;
                    case PixelType::Int16: convert<From, std::int16_t>(s, BufferView<std::int16_t>(static_cast<std::int16_t*>(dst), shape, device), stream); break;
                    case PixelType::UInt32: convert<From, std::uint32_t>(s, BufferView<std::uint32_t>(static_cast<std::uint32_t*>(dst), shape, device), stream); break;
                    case PixelType::Int32: convert<From, std::int32_t>(s, BufferView<std::int32_t>(static_cast<std::int32_t*>(dst), shape, device), stream); break;
                    case PixelType::Float32: convert<From, float>(s, BufferView<float>(static_cast<float*>(dst), shape, device), stream); break;
                    case PixelType::Float64: convert<From, double>(s, BufferView<double>(static_cast<double*>(dst), shape, device), stream); break;
                }
            };
            switch (srcType) {
                case PixelType::UInt8: toAll(std::uint8_t{}); break;
                case PixelType::Int8: toAll(std::int8_t{}); break;
                case PixelType::UInt16: toAll(std::uint16_t{}); break;
                case PixelType::Int16: toAll(std::int16_t{}); break;
                case PixelType::UInt32: toAll(std::uint32_t{}); break;
                case PixelType::Int32: toAll(std::int32_t{}); break;
                case PixelType::Float32: toAll(float{}); break;
                case PixelType::Float64: toAll(double{}); break;
            }
        }

        std::size_t libtiffReadOpens() noexcept { return g_readOpens.load(std::memory_order_relaxed); }

        // Work is handed out in units: a page, or -- when there are fewer
        // pages than threads and the pages are big (one large tiled image,
        // a few RGB planes) -- a run of a page's strips / tiles. Each thread
        // opens the file on its first unit and keeps its handle, hopping
        // between directories with TIFFSetSubDirectory only when a unit is on
        // another page: libtiff handles are not thread-safe, but one handle
        // reads any directory without reopening the file.
        void decodeWithLibtiff(const std::string& path, const DecodeJob& job, void* dstHost) {
            const auto& ifds = *job.ifds;
            const Region r = job.region;
            const std::size_t n = ifds.size();
            if (job.images.size() != n) throw std::logic_error("decodeWithLibtiff: one TiffImageInfo per IFD expected");

            ReadPlan plan;
            plan.images = &job.images;
            plan.ifds = &ifds;
            plan.region = r;
            plan.firstSample = job.firstSample;
            plan.sampleCount = job.sampleCount;
            plan.nativeType = job.geometry->pixelType;
            plan.dstType = job.dstType;
            plan.dst = static_cast<std::uint8_t*>(dstHost);
            plan.dstBpp = bytesPerPixel(job.dstType);
            plan.extract = extractFor(plan.nativeType);
            if (plan.nativeType != job.dstType) plan.convert = convertFor(plan.nativeType, job.dstType);

            // Units: (page, first chunk, end chunk).
            struct Unit {
                std::size_t page, k0, k1;
            };
            std::vector<std::size_t> chunks(n), work(n);
            std::size_t totalWork = 0;
            for (std::size_t p = 0; p < n; ++p) {
                const ChunkGrid c = gridOf(*job.images[p], r, job.firstSample, job.sampleCount);
                chunks[p] = c.count();
                work[p] = decodedBytes(c);
                totalWork += work[p];
            }

            // Every thread of the team used to open the file -- and parse its
            // first directory, which can carry megabytes of ImageJ / OME
            // metadata -- before the loop: 32 opens and ~10 ms for a one-page
            // read on 32 cores. A thread now opens the file only when it is
            // handed work, and the team is sized by the work: about one thread
            // per MiB decoded, no more threads than units.
            constexpr std::size_t kBytesPerThread = std::size_t{1} << 20;
            constexpr std::size_t kBytesPerSplit = std::size_t{2} << 20;   // smallest piece of a split page
            int threads = 1;
            if (job.maxThreads != 1) {
                threads = static_cast<int>(std::clamp<std::size_t>(totalWork / kBytesPerThread, 1,
                                                                   static_cast<std::size_t>(omp_get_max_threads())));
                if (job.maxThreads > 1) threads = std::min(threads, job.maxThreads);
            }
            std::vector<Unit> units;
            units.reserve(n);
            const bool split = threads > 1 && n < static_cast<std::size_t>(threads);
            for (std::size_t p = 0; p < n; ++p) {
                std::size_t parts = 1;
                if (split) {
                    const std::size_t wanted = (static_cast<std::size_t>(threads) * 4 + n - 1) / n;
                    parts = std::clamp<std::size_t>(work[p] / kBytesPerSplit, 1, std::min(wanted, std::max<std::size_t>(chunks[p], 1)));
                }
                for (std::size_t i = 0; i < parts; ++i)
                    units.push_back(Unit{p, chunks[p] * i / parts, chunks[p] * (i + 1) / parts});
            }
            threads = static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(threads), units.size()));
            const auto nUnits = static_cast<std::ptrdiff_t>(units.size());

            std::exception_ptr ex;
            std::atomic<bool> failed{false};
            std::atomic<std::ptrdiff_t> done{0};
            std::mutex progressMu;
#pragma omp parallel num_threads(threads) if (threads > 1)
            {
                ChunkReader rd;
                rd.path = &path;
                // A unit at a time: with the team sized to the work every unit
                // is worth handing out (chunks of 4 left most of a 3-page team idle).
#pragma omp for schedule(dynamic, 1)
                for (std::ptrdiff_t u = 0; u < nUnits; ++u) {
                    if (failed.load(std::memory_order_relaxed)) continue;
                    try {
                        const Unit& unit = units[static_cast<std::size_t>(u)];
                        decodeChunks(rd, plan, unit.page, unit.k0, unit.k1);
                        const auto nDone = done.fetch_add(1, std::memory_order_relaxed) + 1;
                        if (job.progress) {
                            const std::ptrdiff_t step = std::max<std::ptrdiff_t>(1, nUnits / 50);
                            if (nDone == nUnits || nDone % step == 0) {
                                std::lock_guard<std::mutex> lock(progressMu);
                                job.progress(static_cast<double>(nDone) / static_cast<double>(nUnits));
                            }
                        }
                    } catch (...) {
#pragma omp critical
                        {
                            if (!ex) ex = std::current_exception();
                        }
                        failed.store(true, std::memory_order_relaxed);
                    }
                }
            }
            if (ex) std::rethrow_exception(ex);
        }

#ifndef SIRIUS_HAS_NVTIFF
        bool decodeWithNvTiff(TiffFile::Impl&, const DecodeJob&, void*, Device, const Stream&, std::string& reason) {
            reason = "SIRIUS was built without nvTIFF (SIRIUS_ENABLE_NVTIFF=OFF)";
            return false;
        }
        bool nvTiffSupports(TiffFile::Impl&, const DecodeJob&, Device, std::string& reason) {
            reason = "SIRIUS was built without nvTIFF (SIRIUS_ENABLE_NVTIFF=OFF)";
            return false;
        }
#endif

    } // namespace detail

    // --- TiffFile ----------------------------------------------------------------

    namespace {

        // nvTIFF decodes one sample per pixel of 8, 16, 32 or 64 bits; other
        // layouts (RGB, packed 12-bit, float16, ...) decode with libtiff.
        bool gpuEligible(const detail::DecodeJob& job, std::string& reason) {
            for (const TiffImageInfo* g : job.images) {
                if (g->samplesPerPixel != 1) {
                    reason = std::to_string(g->samplesPerPixel) +
                             " samples per pixel are decoded on the CPU (the GPU path reads one-sample images)";
                    return false;
                }
                if (codingOf(*g) != SampleCoding::Aligned) {
                    reason = std::to_string(g->bitsPerSample) + "-bit " +
                             (g->sampleFormat == SAMPLEFORMAT_IEEEFP ? "float" : "integer") +
                             " samples are decoded on the CPU";
                    return false;
                }
            }
            return true;
        }

        // Untyped core of every read: CPU decode in place, or GPU decode via
        // nvTIFF with a libtiff+upload fallback.
        void decodeInto(TiffFile::Impl& impl, const detail::DecodeJob& job, void* dst, Device device,
                        const TiffReadOptions& opts, const Stream& stream) {
            if (device.isCpu()) {
                detail::decodeWithLibtiff(impl.path, job, dst);
                return;
            }
            requireDevice(device);
            std::string reason;
            if (gpuEligible(job, reason) && detail::decodeWithNvTiff(impl, job, dst, device, stream, reason)) {
                if (job.progress) job.progress(1.0);
                return;
            }
            if (!opts.allowCpuFallback)
                throw std::runtime_error("GPU decode of " + impl.path + " is not possible: " + reason +
                                         " (TiffReadOptions::allowCpuFallback is off)");

            // Fallback: libtiff into pinned staging, uploaded chunk by chunk so
            // arbitrarily large stacks never need a stack-sized host buffer.
            const auto& ifds = *job.ifds;
            const std::size_t n = ifds.size();
            const std::size_t pageBytes = static_cast<std::size_t>(job.region.width) * job.region.height *
                                          job.sampleCount * bytesPerPixel(job.dstType);
            constexpr std::size_t kChunkBytes = std::size_t{512} << 20;
            const std::size_t chunk = std::min(n, std::max<std::size_t>(1, kChunkBytes / std::max<std::size_t>(pageBytes, 1)));
            Buffer<std::uint8_t> staging(Shape{static_cast<Index>(chunk * pageBytes)}, Device::cpu(),
                                         HostMemory::Pinned);
            for (std::size_t first = 0; first < n; first += chunk) {
                const std::size_t count = std::min(chunk, n - first);
                const auto b = static_cast<std::ptrdiff_t>(first), e = static_cast<std::ptrdiff_t>(first + count);
                const std::vector<std::uint64_t> part(ifds.begin() + b, ifds.begin() + e);
                detail::DecodeJob sub = job;
                sub.ifds = &part;
                sub.images.assign(job.images.begin() + b, job.images.begin() + e);
                sub.progress = {};
                if (job.progress) {
                    sub.progress = [&job, first, count, n](double f) {
                        job.progress((static_cast<double>(first) + f * static_cast<double>(count)) /
                                     static_cast<double>(n));
                    };
                }
                detail::decodeWithLibtiff(impl.path, sub, staging.data());
                detail::copyBytes(staging.data(), Device::cpu(), static_cast<std::uint8_t*>(dst) + first * pageBytes,
                                  device, count * pageBytes, stream);
                stream.synchronize();   // staging is reused by the next chunk
            }
        }

        const TiffLevel& levelAt(const TiffInfo& info, std::size_t level) {
            if (level >= info.levels.size())
                throw std::out_of_range("TIFF has " + std::to_string(info.levels.size()) +
                                        " pyramid level(s); level " + std::to_string(level) + " requested");
            return info.levels[level];
        }

        // The samples a read takes: [first, first + count).
        std::pair<std::uint16_t, std::uint16_t> samplesOf(std::uint16_t spp, const TiffReadOptions& opts) {
            if (opts.firstSample >= spp)
                throw std::out_of_range("Sample " + std::to_string(opts.firstSample) + " requested from a TIFF with " +
                                        std::to_string(spp) + " sample(s) per pixel");
            const std::uint16_t left = static_cast<std::uint16_t>(spp - opts.firstSample);
            if (opts.sampleCount > left)
                throw std::out_of_range("Samples [" + std::to_string(opts.firstSample) + ", " +
                                        std::to_string(opts.firstSample + opts.sampleCount) +
                                        ") requested from a TIFF with " + std::to_string(spp) +
                                        " sample(s) per pixel");
            return {opts.firstSample, opts.sampleCount == 0 ? left : opts.sampleCount};
        }

        Shape shapeOf(std::size_t pages, std::uint16_t samples, std::uint32_t height, std::uint32_t width) {
            if (samples == 1) return Shape{static_cast<Index>(pages), static_cast<Index>(height), static_cast<Index>(width)};
            return Shape{static_cast<Index>(pages), static_cast<Index>(samples), static_cast<Index>(height),
                         static_cast<Index>(width)};
        }

    } // namespace

    TiffFile::TiffFile(std::string path) : impl_(std::make_unique<Impl>()) {
        impl_->path = std::move(path);
        impl_->info = inspectTiff(impl_->path);
    }

    TiffFile::~TiffFile() = default;
    TiffFile::TiffFile(TiffFile&&) noexcept = default;
    TiffFile& TiffFile::operator=(TiffFile&&) noexcept = default;

    const std::string& TiffFile::path() const noexcept { return impl_->path; }
    const TiffInfo& TiffFile::info() const noexcept { return impl_->info; }

    const TiffMetadata& TiffFile::metadata() const {
        std::call_once(impl_->metadataOnce, [this] {
            if (!impl_->info.pages.empty()) impl_->metadata = parseTiffMetadata(impl_->info.page(0).description);
        });
        return impl_->metadata;
    }

    std::vector<std::vector<std::uint32_t>> TiffFile::series() const {
        const TiffInfo& info = impl_->info;
        const TiffMetadata& md = metadata();
        if (md.ome && !md.omeImages.empty()) {
            auto s = omeImagePages(md, info.pageCount(), info.pageCount() ? info.samplesPerPixel() : 1);
            // keep the images that have pages; a file whose OME-XML maps none
            // of its pages (another file's companion) reads as one series
            s.erase(std::remove_if(s.begin(), s.end(), [](const auto& v) { return v.empty(); }), s.end());
            if (!s.empty()) return s;
        }
        std::vector<std::uint32_t> all(info.pageCount());
        for (std::size_t i = 0; i < all.size(); ++i) all[i] = static_cast<std::uint32_t>(i);
        return {std::move(all)};
    }

    Shape TiffFile::readShape(std::size_t pages, std::uint32_t height, std::uint32_t width,
                              const TiffReadOptions& opts) const {
        const auto [first, count] = samplesOf(impl_->info.pageCount() ? impl_->info.samplesPerPixel() : 1, opts);
        (void)first;
        return shapeOf(pages, count, height, width);
    }

    bool TiffFile::gpuDecodable(Device device, std::string* reason) const {
        std::string why;
        bool ok = false;
        if (!device.isCuda()) {
            why = "not a CUDA device";
        } else if (!builtWithNvTiff()) {
            why = "SIRIUS was built without nvTIFF";
        } else if (device.index < 0 || device.index >= cudaDeviceCount()) {
            why = "CUDA device " + toString(device) + " is not available";
        } else {
            const TiffInfo& info = impl_->info;
            detail::DecodeJob job;
            job.ifds = &info.pages;
            job.geometry = &info.page(0);
            for (std::uint64_t off : info.pages) job.images.push_back(&info.image(off));
            job.region = Region{}.resolve(info.width(), info.height());
            job.dstType = info.pixelType();
            try {
                ok = gpuEligible(job, why) && detail::nvTiffSupports(*impl_, job, device, why);
            } catch (const std::exception& e) {
                why = e.what();
            }
        }
        if (reason) *reason = why;
        return ok;
    }

    template <typename T>
    void TiffFile::decode(const std::vector<std::uint64_t>& ifds, Region region, BufferView<T> dst,
                          const TiffReadOptions& opts, const Stream& stream) const {
        if (ifds.empty()) throw std::invalid_argument("TiffFile::decode: no image directories given");
        const TiffInfo& info = impl_->info;
        const TiffImageInfo& g = info.image(ifds[0]);
        detail::DecodeJob job;
        job.images.reserve(ifds.size());
        for (std::uint64_t off : ifds) {
            const auto& i = info.image(off);
            if (i.width != g.width || i.height != g.height || i.pixelType != g.pixelType ||
                i.samplesPerPixel != g.samplesPerPixel)
                throw IoError("TIFF image at offset " + std::to_string(off) + " (" + std::to_string(i.width) + "x" +
                              std::to_string(i.height) + " " + toString(i.pixelType) + " x" +
                              std::to_string(i.samplesPerPixel) + ") does not match the first one (" +
                              std::to_string(g.width) + "x" + std::to_string(g.height) + " " + toString(g.pixelType) +
                              " x" + std::to_string(g.samplesPerPixel) + ")");
            if (!i.decodable())
                throw IoError("Cannot decode the TIFF image at offset " + std::to_string(off) + " of " + impl_->path +
                              ": " + i.unsupported);
            job.images.push_back(&i);
        }
        const auto [firstSample, sampleCount] = samplesOf(g.samplesPerPixel, opts);
        const Region r = region.resolve(g.width, g.height);
        const Shape expected = shapeOf(ifds.size(), sampleCount, r.height, r.width);
        if (dst.shape() != expected) detail::throwShapeMismatch("TiffFile::decode destination", dst.shape(), expected);

        job.ifds = &ifds;
        job.geometry = &g;
        job.region = r;
        job.firstSample = firstSample;
        job.sampleCount = sampleCount;
        job.dstType = pixelTypeOf<T>();
        job.maxThreads = opts.maxThreads;
        job.progress = opts.progress;
        decodeInto(*impl_, job, dst.data(), dst.device(), opts, stream);
    }

    template <typename T>
    Buffer<T> TiffFile::readStack(const TiffReadOptions& opts, const Stream& stream) const {
        if (!impl_->info.uniformPages())
            throw IoError("TIFF pages differ in size, pixel type or samples per pixel; read them individually: " +
                          impl_->path);
        Buffer<T> out(readShape(impl_->info.pageCount(), impl_->info.height(), impl_->info.width(), opts), opts.device,
                      opts.hostMemory, stream);
        decode<T>(impl_->info.pages, Region{}, out.view(), opts, stream);
        return out;
    }

    template <typename T>
    Buffer<T> TiffFile::readPages(std::size_t first, std::size_t count, const TiffReadOptions& opts,
                                  const Stream& stream) const {
        const auto& pages = impl_->info.pages;
        // `count > pages.size() - first` rather than `first + count > size`:
        // the sum can wrap for a count near SIZE_MAX and pass the check
        if (count == 0 || first > pages.size() || count > pages.size() - first)
            throw std::out_of_range("Pages [" + std::to_string(first) + ", " + std::to_string(first + count) +
                                    ") requested from a TIFF with " + std::to_string(pages.size()) + " page(s)");
        const std::vector<std::uint64_t> ifds(pages.begin() + static_cast<std::ptrdiff_t>(first),
                                              pages.begin() + static_cast<std::ptrdiff_t>(first + count));
        const auto& g = impl_->info.image(ifds[0]);
        Buffer<T> out(readShape(count, g.height, g.width, opts), opts.device, opts.hostMemory, stream);
        decode<T>(ifds, Region{}, out.view(), opts, stream);
        return out;
    }

    template <typename T>
    Buffer<T> TiffFile::readLevel(std::size_t level, const TiffReadOptions& opts, const Stream& stream) const {
        const TiffLevel& l = levelAt(impl_->info, level);
        Buffer<T> out(readShape(l.ifds.size(), l.height, l.width, opts), opts.device, opts.hostMemory, stream);
        decode<T>(l.ifds, Region{}, out.view(), opts, stream);
        return out;
    }

    std::vector<std::uint64_t> TiffFile::seriesIfds(std::size_t index, std::size_t level) const {
        const auto all = series();
        if (index >= all.size())
            throw std::out_of_range("TIFF has " + std::to_string(all.size()) + " series; series " +
                                    std::to_string(index) + " requested");
        const TiffInfo& info = impl_->info;
        const auto& pages = all[index];
        std::vector<std::uint64_t> ifds;
        ifds.reserve(pages.size());
        if (level == 0) {
            for (std::uint32_t p : pages) ifds.push_back(info.pages.at(p));
            return ifds;
        }
        // SubIFD pyramids hang off each page: the series has the levels its
        // own pages have, whatever other series hold (OME-TIFF writes a
        // pyramid per image). Flat pyramids come from TiffInfo::levels.
        bool sub = true;
        for (std::uint32_t p : pages) {
            const auto& subs = info.image(info.pages.at(p)).subIfds;
            if (subs.size() < level) {
                sub = false;
                break;
            }
            ifds.push_back(subs[level - 1]);
        }
        if (sub) return ifds;
        ifds.clear();
        if (level < info.levels.size()) {
            const TiffLevel& l = info.levels[level];
            for (std::uint32_t p : pages) {
                if (p >= l.ifds.size()) break;
                ifds.push_back(l.ifds[p]);
            }
            if (ifds.size() == pages.size()) return ifds;
        }
        throw std::out_of_range("Series " + std::to_string(index) + " has " + std::to_string(seriesLevels(index)) +
                                " pyramid level(s); level " + std::to_string(level) + " requested");
    }

    std::size_t TiffFile::seriesLevels(std::size_t index) const {
        const auto all = series();
        if (index >= all.size()) return 0;
        const TiffInfo& info = impl_->info;
        std::size_t sub = ~std::size_t{0};
        for (std::uint32_t p : all[index]) sub = std::min(sub, info.image(info.pages.at(p)).subIfds.size());
        if (sub != ~std::size_t{0} && sub > 0) return 1 + sub;
        std::size_t levels = 1;
        for (std::size_t k = 1; k < info.levels.size(); ++k) {
            bool all_ = true;
            for (std::uint32_t p : all[index]) all_ = all_ && p < info.levels[k].ifds.size();
            if (!all_) break;
            levels = k + 1;
        }
        return levels;
    }

    template <typename T>
    Buffer<T> TiffFile::readSeries(std::size_t index, std::size_t level, const TiffReadOptions& opts,
                                   const Stream& stream) const {
        const std::vector<std::uint64_t> ifds = seriesIfds(index, level);
        if (ifds.empty()) throw std::out_of_range("Series " + std::to_string(index) + " has no pages");
        const auto& g = impl_->info.image(ifds[0]);
        Buffer<T> out(readShape(ifds.size(), g.height, g.width, opts), opts.device, opts.hostMemory, stream);
        decode<T>(ifds, Region{}, out.view(), opts, stream);
        return out;
    }

    template <typename T>
    Buffer<T> TiffFile::readRegion(Region region, std::size_t level, const TiffReadOptions& opts,
                                   const Stream& stream) const {
        const TiffLevel& l = levelAt(impl_->info, level);
        const Region r = region.resolve(l.width, l.height);
        Buffer<T> out(readShape(l.ifds.size(), r.height, r.width, opts), opts.device, opts.hostMemory, stream);
        decode<T>(l.ifds, r, out.view(), opts, stream);
        return out;
    }

    AnyBuffer readTiffAny(const std::string& path, const TiffReadOptions& opts, const Stream& stream) {
        TiffFile file(path);
        switch (file.info().pixelType()) {
            case PixelType::UInt8: return file.readStack<std::uint8_t>(opts, stream);
            case PixelType::Int8: return file.readStack<std::int8_t>(opts, stream);
            case PixelType::UInt16: return file.readStack<std::uint16_t>(opts, stream);
            case PixelType::Int16: return file.readStack<std::int16_t>(opts, stream);
            case PixelType::UInt32: return file.readStack<std::uint32_t>(opts, stream);
            case PixelType::Int32: return file.readStack<std::int32_t>(opts, stream);
            case PixelType::Float32: return file.readStack<float>(opts, stream);
            case PixelType::Float64: return file.readStack<double>(opts, stream);
        }
        throw IoError("Unsupported TIFF format");
    }

    // --- Eigen convenience API -------------------------------------------------

    namespace {
        // The Eigen tensors hold one sample per pixel.
        void requireOneSample(const TiffInfo& info, const std::string& path) {
            if (info.pageCount() && info.samplesPerPixel() != 1)
                throw IoError(path + " has " + std::to_string(info.samplesPerPixel()) +
                              " samples per pixel; the Eigen API reads one-sample images (use TiffFile, which "
                              "returns {pages, samples, height, width}, or TiffReadOptions::firstSample)");
        }
    } // namespace

    template <typename T>
    Image<T> readTiff(const std::string& path) {
        TiffFile file(path);
        requireOneSample(file.info(), path);
        const auto& p = file.info().page(0);
        Image<T> image(p.height, p.width);
        file.decode<T>({p.ifdOffset}, Region{}, toView(image).asStack());
        return image;
    }

    template <typename T>
    ImageStack<T> readTiffStack(const std::string& path) {
        TiffFile file(path);
        const TiffInfo& info = file.info();
        if (!info.uniformPages())
            throw IoError("TIFF pages differ in size or pixel type: " + path);
        requireOneSample(info, path);
        ImageStack<T> stack(static_cast<Eigen::Index>(info.pageCount()), info.height(), info.width());
        file.decode<T>(info.pages, Region{}, toView(stack));
        return stack;
    }

    AnyImageStack readTiffStackAny(const std::string& path) {
        TiffFile file(path);
        const TiffInfo& info = file.info();
        if (!info.uniformPages())
            throw IoError("TIFF pages differ in size or pixel type: " + path);
        requireOneSample(info, path);
        auto read = [&](auto tag) -> AnyImageStack {
            using T = decltype(tag);
            ImageStack<T> stack(static_cast<Eigen::Index>(info.pageCount()), info.height(), info.width());
            file.decode<T>(info.pages, Region{}, toView(stack));
            return stack;
        };
        switch (info.pixelType()) {
            case PixelType::UInt8: return read(std::uint8_t{});
            case PixelType::Int8: return read(std::int8_t{});
            case PixelType::UInt16: return read(std::uint16_t{});
            case PixelType::Int16: return read(std::int16_t{});
            case PixelType::UInt32: return read(std::uint32_t{});
            case PixelType::Int32: return read(std::int32_t{});
            case PixelType::Float32: return read(float{});
            case PixelType::Float64: return read(double{});
        }
        throw IoError("Unsupported TIFF format");
    }

    template <typename T>
    void writeTiffStack(const std::string& path, BufferView<const T> stack, const TiffWriteOptions& options) {
        if (stack.rank() != 2 && stack.rank() != 3)
            throw std::invalid_argument("writeTiffStack expects a (pages, rows, cols) or (rows, cols) view, got " +
                                        stack.shape().toString());
        if (stack.size() == 0) throw std::runtime_error("Cannot write empty stack");
        if (!stack.device().isCpu()) {
            writeTiffStack<T>(path, onHost(stack).view(), options);
            return;
        }
        const BufferView<const T> s = stack.rank() == 3 ? stack : stack.asStack();
        writePages<T>(path, s.data(), s.dim(0), s.dim(1), s.dim(2), options, stack.rank() == 3);
    }

    namespace {
        // The original writers: compressed data always carried a predictor
        // (the floating-point one for float data), classic TIFF for single
        // images, BigTIFF for stacks.
        TiffWriteOptions legacyOptions(TiffCompression comp, bool bigTiff) {
            TiffWriteOptions o;
            o.compression = comp;
            o.predictor = true;
            o.bigTiff = bigTiff;
            return o;
        }
    } // namespace

    template <typename T>
    void writeTiff(const std::string& path, BufferView<const T> image, TiffCompression comp) {
        if (image.rank() != 2)
            throw std::invalid_argument("writeTiff expects a rank-2 (rows, cols) view, got " + image.shape().toString());
        if (!image.device().isCpu()) {
            writeTiff<T>(path, onHost(image).view(), comp);
            return;
        }
        writePages<T>(path, image.data(), 1, image.dim(0), image.dim(1), legacyOptions(comp, false), false);
    }

    template <typename T>
    void writeTiff(const std::string& path, const Image<T>& image, TiffCompression comp) {
        writeTiff<T>(path, toConstView(image), comp);
    }

    template <typename T>
    void writeTiffStack(const std::string& path, BufferView<const T> stack, TiffCompression comp) {
        if (stack.rank() != 3)
            throw std::invalid_argument("writeTiffStack expects a rank-3 (pages, rows, cols) view, got " + stack.shape().toString());
        if (stack.size() == 0)
            throw std::runtime_error("Cannot write empty stack");
        // BigTIFF ("w8") lifts the 4 GiB offset limit. For small stacks this
        // is mild overhead; for large ones it is the only option that works.
        writeTiffStack<T>(path, stack, legacyOptions(comp, true));
    }

    template <typename T>
    void writeTiffStack(const std::string& path, const ImageStack<T>& stack, TiffCompression comp) {
        writeTiffStack<T>(path, toConstView(stack), comp);
    }

    // Explicit instantiations for every supported pixel type.
#define SIRIUS_TIFF_INSTANTIATE(T)                                                                                     \
    template void TiffFile::decode<T>(const std::vector<std::uint64_t>&, Region, BufferView<T>,                        \
                                      const TiffReadOptions&, const Stream&) const;                                    \
    template Buffer<T> TiffFile::readStack<T>(const TiffReadOptions&, const Stream&) const;                            \
    template Buffer<T> TiffFile::readPages<T>(std::size_t, std::size_t, const TiffReadOptions&, const Stream&) const;  \
    template Buffer<T> TiffFile::readLevel<T>(std::size_t, const TiffReadOptions&, const Stream&) const;               \
    template Buffer<T> TiffFile::readSeries<T>(std::size_t, std::size_t, const TiffReadOptions&, const Stream&) const; \
    template Buffer<T> TiffFile::readRegion<T>(Region, std::size_t, const TiffReadOptions&, const Stream&) const;      \
    template Image<T> readTiff<T>(const std::string&);                                                                 \
    template ImageStack<T> readTiffStack<T>(const std::string&);                                                       \
    template void writeTiff<T>(const std::string&, BufferView<const T>, TiffCompression);                              \
    template void writeTiff<T>(const std::string&, const Image<T>&, TiffCompression);                                  \
    template void writeTiffStack<T>(const std::string&, BufferView<const T>, TiffCompression);                         \
    template void writeTiffStack<T>(const std::string&, BufferView<const T>, const TiffWriteOptions&);                 \
    template void writeTiffStack<T>(const std::string&, const ImageStack<T>&, TiffCompression);

    SIRIUS_TIFF_INSTANTIATE(std::uint8_t)
    SIRIUS_TIFF_INSTANTIATE(std::int8_t)
    SIRIUS_TIFF_INSTANTIATE(std::uint16_t)
    SIRIUS_TIFF_INSTANTIATE(std::int16_t)
    SIRIUS_TIFF_INSTANTIATE(std::uint32_t)
    SIRIUS_TIFF_INSTANTIATE(std::int32_t)
    SIRIUS_TIFF_INSTANTIATE(float)
    SIRIUS_TIFF_INSTANTIATE(double)
#undef SIRIUS_TIFF_INSTANTIATE

} // namespace sirius
