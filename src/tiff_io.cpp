#include "sirius/tiff_io.hpp"
#include <cstring>
#include <type_traits>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <atomic>
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

        // openTiff will now return unique_ptr
        // aaand tiff file will be closed when TiffPtr goes out of scope whether normall or via exception
        // better safe than sorry
        TiffPtr openTiff(const std::string& path, const char* mode) {
            TiffPtr tif(TIFFOpen(path.c_str(), mode));
            if (!tif) throw std::runtime_error("Failed to open TIFF: " + path);
            return tif;
        }

        struct TiffPageInfo {
            uint32_t width;
            uint32_t height;
            uint16_t bps; // bits per sample
            uint16_t spp; // sample per pixel (channels)
            uint16_t fmt; // format
        };

        TiffPageInfo getPageInfo(TIFF* tif) {
            TiffPageInfo info{};
            if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH,      &info.width))
                throw std::runtime_error("TIFF missing required tag: IMAGEWIDTH");
            if (!TIFFGetField(tif, TIFFTAG_IMAGELENGTH,     &info.height))
                throw std::runtime_error("TIFF missing required tag: IMAGELENGTH");
            if (!TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE,   &info.bps))
                throw std::runtime_error("TIFF missing required tag: BITSPERSAMPLE");
            if (!TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &info.spp))
                throw std::runtime_error("TIFF missing required tag: SAMPLESPERPIXEL");
            TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &info.fmt);

            if (info.spp != 1)
                throw std::runtime_error("Only single-channel (grayscale) TIFFs are supported.");

            // validate input formats
            if (info.fmt == SAMPLEFORMAT_IEEEFP && info.bps != 32 && info.bps != 64)
                throw std::runtime_error("Unsupported float bit depth: " + std::to_string(info.bps));
            if (info.fmt != SAMPLEFORMAT_IEEEFP && info.bps != 8 && info.bps != 16 && info.bps != 32)
                throw std::runtime_error("Unsupported integer bit depth: " + std::to_string(info.bps));

            return info;
        }

        // map types to TIFF tags
        template <typename T>
        constexpr uint16_t sampleFormat() {
            if constexpr (std::is_floating_point_v<T>) return SAMPLEFORMAT_IEEEFP;
            else if constexpr (std::is_unsigned_v<T>)  return SAMPLEFORMAT_UINT;
            else return SAMPLEFORMAT_INT;
        }

        uint16_t mapCompression(TiffCompression comp) {
            switch(comp) {
                case TiffCompression::Lzw:     return COMPRESSION_LZW;
                case TiffCompression::Deflate: return COMPRESSION_ADOBE_DEFLATE;
                default:                       return COMPRESSION_NONE;
            }
        }

        // TIFF stores pixels as raw bytes so need to reinterpret them with the correct type
        // using "safe type punning"
        // why not use this instead?: float v = *(float *) src + i*4
        //      because some cpus will crash
        // why not use this instead?: *reinterpret_cast<const float*> (src + i*4)
        //      because of Strict Aliasing rule
        // but isn't memcpy slow?
        //      Apparently, compilers recognize this pattern while taking alignment into account.
        // See eg: https://developer.arm.com/documentation/100748/0624/Writing-Optimized-Code/C-and-C---aliasing
        template <typename T>
        void convertScanline(const uint8_t* src, T* dst, uint32_t width,
                            uint16_t bps, uint16_t fmt) {
            switch (fmt) {
                case SAMPLEFORMAT_IEEEFP:
                    switch (bps) {
                        case 32: for (uint32_t i = 0; i < width; ++i) { float  v; std::memcpy(&v, src+i*4, 4); dst[i] = static_cast<T>(v); } break;
                        case 64: for (uint32_t i = 0; i < width; ++i) { double v; std::memcpy(&v, src+i*8, 8); dst[i] = static_cast<T>(v); } break;
                    } break;
                case SAMPLEFORMAT_INT:
                    switch (bps) {
                        case  8: for (uint32_t i = 0; i < width; ++i) { int8_t  v; std::memcpy(&v, src+i,   1); dst[i] = static_cast<T>(v); } break;
                        case 16: for (uint32_t i = 0; i < width; ++i) { int16_t v; std::memcpy(&v, src+i*2, 2); dst[i] = static_cast<T>(v); } break;
                        case 32: for (uint32_t i = 0; i < width; ++i) { int32_t v; std::memcpy(&v, src+i*4, 4); dst[i] = static_cast<T>(v); } break;
                    } break;
                default: // SAMPLEFORMAT_UINT
                    switch (bps) {
                        case  8: for (uint32_t i = 0; i < width; ++i) { dst[i] = static_cast<T>(src[i]); } break;
                        case 16: for (uint32_t i = 0; i < width; ++i) { uint16_t v; std::memcpy(&v, src+i*2, 2); dst[i] = static_cast<T>(v); } break;
                        case 32: for (uint32_t i = 0; i < width; ++i) { uint32_t v; std::memcpy(&v, src+i*4, 4); dst[i] = static_cast<T>(v); } break;
                    } break;
            }
        }

        // check if Eigen tensor type T is an exact match
        // to circumvent the slow "pixel by pixel" conversion process
        template <typename T>
        constexpr bool isExactMatch(uint16_t bps, uint16_t fmt) {
            // Check Floating Point matches
            if (std::is_same_v<T, float>)    return fmt == SAMPLEFORMAT_IEEEFP && bps == 32;
            if (std::is_same_v<T, double>)   return fmt == SAMPLEFORMAT_IEEEFP && bps == 64;
            
            // Check Unsigned Integer matches
            if (std::is_same_v<T, uint8_t>)  return fmt == SAMPLEFORMAT_UINT && bps == 8;
            if (std::is_same_v<T, uint16_t>) return fmt == SAMPLEFORMAT_UINT && bps == 16;
            if (std::is_same_v<T, uint32_t>) return fmt == SAMPLEFORMAT_UINT && bps == 32;
            
            // Check Signed Integer matches
            if (std::is_same_v<T, int8_t>)   return fmt == SAMPLEFORMAT_INT && bps == 8;
            if (std::is_same_v<T, int16_t>)  return fmt == SAMPLEFORMAT_INT && bps == 16;
            if (std::is_same_v<T, int32_t>)  return fmt == SAMPLEFORMAT_INT && bps == 32;

            return false;
        }

        // Reads a strip-organized TIFF page into dst.
        //
        // TIFFReadEncodedStrip reduces API call overhead from O(height) to
        // O(nStrips) per page and enables one large memcpy per strip instead
        // of many small ones. On the fast path (T matches the on-disk type
        // exactly) each strip is decoded directly into dst with no intermediate
        // buffer.
        template <typename T>
        void readScanlinePage(TIFF* tif, T* dst, const TiffPageInfo& info) {
            uint32_t rowsPerStrip = 0;
            TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);

            const tstrip_t nStrips = TIFFNumberOfStrips(tif);
            if (nStrips == 0)
                throw std::runtime_error("TIFF reports zero strips");

            const tmsize_t maxStripBytes = TIFFStripSize(tif);
            if (maxStripBytes <= 0)
                throw std::runtime_error("TIFF reports invalid strip size");

            const bool useFastPath = isExactMatch<T>(info.bps, info.fmt);
            const size_t bytesPerPixel = info.bps / 8;

            // Conversion path only: one strip-sized buffer, allocated once.
            std::vector<uint8_t> buf;
            if (!useFastPath)
                buf.resize(static_cast<size_t>(maxStripBytes));

            for (tstrip_t s = 0; s < nStrips; ++s) {
                const uint32_t startRow = s * rowsPerStrip;
                const uint32_t validRows = std::min(rowsPerStrip, info.height - startRow);
                // Exact decoded byte count for this strip — handles the partial
                // last strip without relying on codec-specific padding behavior.
                const tmsize_t stripDataBytes =
                    static_cast<tmsize_t>(validRows) * info.width *
                    static_cast<tmsize_t>(bytesPerPixel);

                if (useFastPath) {
                    // Decode directly into the caller's buffer — no intermediate copy.
                    T* stripDst = dst + static_cast<size_t>(startRow) * info.width;
                    if (TIFFReadEncodedStrip(tif, s, stripDst, stripDataBytes) < 0)
                        throw std::runtime_error("Failed to read strip " + std::to_string(s));
                } else {
                    if (TIFFReadEncodedStrip(tif, s, buf.data(), maxStripBytes) < 0)
                        throw std::runtime_error("Failed to read strip " + std::to_string(s));
                    for (uint32_t r = 0; r < validRows; ++r) {
                        const uint8_t* srcRow = buf.data() +
                            static_cast<size_t>(r) * info.width * bytesPerPixel;
                        T* dstRow = dst + static_cast<size_t>(startRow + r) * info.width;
                        convertScanline<T>(srcRow, dstRow, info.width, info.bps, info.fmt);
                    }
                }
            }
        }

        // Tiled layout: stored in blocks (e.g. 256x256).
        // Using TIFFReadScanline on tiled TIFFs forces libtiff to decompress an entire
        // tile for every row — O(height * tileH) decompressions instead of O(numTiles).
        // TIFFReadTile decompresses each tile exactly once.
        template <typename T>
        void readTiledPage(TIFF* tif, T* dst, const TiffPageInfo& info) {
            uint32_t tileW = 0, tileH = 0;
            if (!TIFFGetField(tif, TIFFTAG_TILEWIDTH,  &tileW) || tileW == 0)
                throw std::runtime_error("TIFF missing or invalid TILEWIDTH");
            if (!TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileH) || tileH == 0)
                throw std::runtime_error("TIFF missing or invalid TILELENGTH");

            const tmsize_t tileBytes = TIFFTileSize(tif);
            if (tileBytes <= 0)
                throw std::runtime_error("TIFF reports invalid tile size");

            // allocate once outside the loop
            std::vector<uint8_t> tileBuf(static_cast<size_t>(tileBytes));
            const size_t bytesPerPixel = info.bps / 8;

            // use fast path if there is exact match between eigen type and the tiff type
            const bool useFastPath = isExactMatch<T>(info.bps, info.fmt);

            // Loop counters stay uint32_t to match TIFFReadTile's x/y params
            // and the TIFF spec (image dims are uint32_t). Any multiplication
            // that could overflow uint32_t is promoted to size_t at the site.
            for (uint32_t tileRow = 0; tileRow < info.height; tileRow += tileH) {
                for (uint32_t tileCol = 0; tileCol < info.width; tileCol += tileW) {
                    if (TIFFReadTile(tif, tileBuf.data(), tileCol, tileRow, 0, 0) < 0)
                        throw std::runtime_error("Failed to read tile at (" +
                            std::to_string(tileCol) + "," + std::to_string(tileRow) + ")");

                    // edge tiles are padded to full tile size — clamp to actual image bounds
                    const uint32_t validH = std::min(tileH, info.height - tileRow);
                    const uint32_t validW = std::min(tileW, info.width  - tileCol);

                    for (uint32_t r = 0; r < validH; ++r) {
                        // promote to size_t before multiplying to avoid overflow on large tiles
                        const uint8_t* srcRow = tileBuf.data() +
                            static_cast<size_t>(r) * tileW * bytesPerPixel;
                        T* dstRow = dst +
                            (static_cast<size_t>(tileRow) + r) * info.width + tileCol;

                        if (useFastPath) {
                            std::memcpy(dstRow, srcRow, static_cast<size_t>(validW) * sizeof(T));
                        } else {
                            convertScanline<T>(srcRow, dstRow, validW, info.bps, info.fmt);
                        }
                    }
                }
            }
        }

        template <typename T>
        void writePageFrom(TIFF* tif, const T* src, uint32_t height, uint32_t width,
                        bool multiPage, TiffCompression comp) {
            TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
            TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
            TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, static_cast<uint16_t>(sizeof(T) * 8));
            TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, static_cast<uint16_t>(1));
            TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, sampleFormat<T>());
            TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
            TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
            
            // Setup Compression
            uint16_t tiffComp = mapCompression(comp);
            TIFFSetField(tif, TIFFTAG_COMPRESSION, tiffComp);
            
            // Set predictor for better compression ratios on compressed data
            if (tiffComp == COMPRESSION_LZW || tiffComp == COMPRESSION_DEFLATE) {
                if constexpr (std::is_integral_v<T>) {
                    TIFFSetField(tif, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
                } else {
                    TIFFSetField(tif, TIFFTAG_PREDICTOR, PREDICTOR_FLOATINGPOINT);
                }
            }
            
            TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, 0));
            
            if (multiPage)
                TIFFSetField(tif, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);

            // TIFFWriteScanline is documented as allowed to modify its input
            // buffer (e.g. in-place byte-swap when writing non-native endian).
            // Copy each row into a scratch buffer so the caller's const data
            // is never touched — also lets us drop the const_cast UB.
            std::vector<T> rowBuf(width);
            const size_t rowBytes = static_cast<size_t>(width) * sizeof(T);
            for (uint32_t row = 0; row < height; ++row) {
                std::memcpy(rowBuf.data(),
                            src + static_cast<size_t>(row) * width,
                            rowBytes);
                if (TIFFWriteScanline(tif, rowBuf.data(), row) < 0)
                    throw std::runtime_error("Failed to write scanline " + std::to_string(row));
            }
        }

    } // anonymous namespace

    // --- Single image ---

    template <typename T>
    Image<T> readTiff(const std::string& path) {
        auto tif = openTiff(path, "r");
        auto info = getPageInfo(tif.get());
        Image<T> image(info.height, info.width);
        // check layout once — a file is either tiled or not
        if (TIFFIsTiled(tif.get()))
            readTiledPage<T>(tif.get(), image.data(), info);
        else
            readScanlinePage<T>(tif.get(), image.data(), info);
        return image;
    }

    template <typename T>
    void writeTiff(const std::string& path, const Image<T>& image, TiffCompression comp) {
        auto tif = openTiff(path, "w");
        writePageFrom<T>(tif.get(), image.data(),
                        static_cast<uint32_t>(image.dimension(0)),
                        static_cast<uint32_t>(image.dimension(1)), false, comp);
    }

    // --- Image stack ---

    template <typename T>
    ImageStack<T> readTiffStack(const std::string& path) {
        Eigen::Index pageCount = 0;
        Eigen::Index rows = 0;
        Eigen::Index cols = 0;

        // Pass 1: walk the directory chain sequentially to validate geometry
        // and cache each page's on-disk offset. TIFF directories are a linked
        // list, so repeatedly calling TIFFSetDirectory(z) would be O(n^2).
        // TIFFSetSubDirectory(offset) is O(1). Validation here keeps the
        // parallel read path free of per-page tag reads.
        TiffPageInfo pageInfo{};
        bool isTiled = false;
        std::vector<uint64_t> offset;
        {
            auto tif = openTiff(path, "r");
            pageInfo = getPageInfo(tif.get());
            isTiled  = static_cast<bool>(TIFFIsTiled(tif.get()));
            rows = static_cast<Eigen::Index>(pageInfo.height);
            cols = static_cast<Eigen::Index>(pageInfo.width);

            do {
                offset.push_back(TIFFCurrentDirOffset(tif.get()));
                if (pageCount > 0) {
                    auto pi = getPageInfo(tif.get());
                    if (static_cast<Eigen::Index>(pi.height) != rows ||
                        static_cast<Eigen::Index>(pi.width)  != cols)
                        throw std::runtime_error(
                            "TIFF stack page " + std::to_string(pageCount) +
                            " dimensions (" + std::to_string(pi.width) + "x" +
                            std::to_string(pi.height) + ") do not match page 0 (" +
                            std::to_string(cols) + "x" + std::to_string(rows) + ")");
                    if (static_cast<bool>(TIFFIsTiled(tif.get())) != isTiled)
                        throw std::runtime_error(
                            "TIFF stack page " + std::to_string(pageCount) +
                            " has mixed tiled/scanline layout");
                }
                ++pageCount;
            } while (TIFFReadDirectory(tif.get()));
        }

        // Allocate the contiguous memory block once
        ImageStack<T> stack(pageCount, rows, cols);
        const Eigen::Index stride = rows * cols;

        std::exception_ptr ex;
        std::atomic<bool> failed{false};

        // Each thread opens its own handle once and reuses it across every page
        // it processes. libtiff handles are not thread-safe (cannot be shared
        // across threads), but a single handle can navigate between directories
        // via TIFFSetSubDirectory without reopening the file.
        #pragma omp parallel
        {
            TiffPtr localTif;
            bool openOk = false;
            try {
                localTif = openTiff(path, "r");
                openOk = true;
            } catch (...) {
                #pragma omp critical
                { if (!ex) ex = std::current_exception(); }
                failed.store(true, std::memory_order_relaxed);
            }

            #pragma omp for schedule(dynamic, 4)
            for (Eigen::Index z = 0; z < pageCount; ++z) {
                if (failed.load(std::memory_order_relaxed) || !openOk) continue;
                try {
                    if (!TIFFSetSubDirectory(localTif.get(), offset[z]))
                        throw std::runtime_error(
                            "Failed to seek to TIFF directory " + std::to_string(z));

                    T* dst = stack.data() + z * stride;
                    if (isTiled)
                        readTiledPage<T>(localTif.get(), dst, pageInfo);
                    else
                        readScanlinePage<T>(localTif.get(), dst, pageInfo);
                } catch (...) {
                    #pragma omp critical
                    { if (!ex) ex = std::current_exception(); }
                    failed.store(true, std::memory_order_relaxed);
                }
            }
        }
        if (ex) std::rethrow_exception(ex);

        return stack;
    }

    template <typename T>
    void writeTiffStack(const std::string& path, const ImageStack<T>& stack, TiffCompression comp) {
        if (stack.size() == 0)
            throw std::runtime_error("Cannot write empty stack");

        // BigTIFF ("w8") lifts the 4 GiB offset limit. For small stacks this
        // is mild overhead; for large ones it is the only option that works.
        auto tif = openTiff(path, "w8");

        const Eigen::Index pages  = stack.dimension(0);
        const Eigen::Index rows   = stack.dimension(1);
        const Eigen::Index cols   = stack.dimension(2);
        const Eigen::Index stride = rows * cols; // 64-bit: safe for huge pages

        const auto height = static_cast<uint32_t>(rows);
        const auto width  = static_cast<uint32_t>(cols);

        for (Eigen::Index z = 0; z < pages; ++z) {
            writePageFrom<T>(tif.get(), stack.data() + z * stride,
                             height, width, true, comp);
            if (!TIFFWriteDirectory(tif.get()))
                throw std::runtime_error(
                    "Failed to finalize TIFF directory for page " + std::to_string(z));
        }
    }

    AnyImageStack readTiffStackAny(const std::string& path) {
        auto tif = openTiff(path, "r");
        auto info = getPageInfo(tif.get());

        switch(info.fmt) {
            case SAMPLEFORMAT_INT:
                switch(info.bps) {
                    case 8 : return readTiffStack<int8_t>(path);
                    case 16: return readTiffStack<int16_t>(path);
                    case 32: return readTiffStack<int32_t>(path);
                    default: break;
                }
                break;
            case SAMPLEFORMAT_UINT:
                switch(info.bps) {
                    case 8 : return readTiffStack<uint8_t>(path);
                    case 16: return readTiffStack<uint16_t>(path);
                    case 32: return readTiffStack<uint32_t>(path);
                    default: break;
                }
                break;
            case SAMPLEFORMAT_IEEEFP:
                switch(info.bps) {
                    case 32: return readTiffStack<float>(path);
                    case 64: return readTiffStack<double>(path);
                    default: break;
                }
                break;
            default:
                break;
        }

        throw std::runtime_error("Unsupported TIFF format");
    }

    // Explicit instantiations
    template Image<uint8_t> readTiff(const std::string&);
    template Image<int8_t> readTiff(const std::string&);
    template Image<uint16_t> readTiff(const std::string&);
    template Image<int16_t> readTiff(const std::string&);
    template Image<uint32_t> readTiff(const std::string&);
    template Image<int32_t> readTiff(const std::string&);
    template Image<float> readTiff(const std::string&);
    template Image<double> readTiff(const std::string&);

    template ImageStack<uint8_t> readTiffStack(const std::string&);
    template ImageStack<int8_t> readTiffStack(const std::string&);
    template ImageStack<uint16_t> readTiffStack(const std::string&);
    template ImageStack<int16_t> readTiffStack(const std::string&);
    template ImageStack<uint32_t> readTiffStack(const std::string&);
    template ImageStack<int32_t> readTiffStack(const std::string&);
    template ImageStack<float> readTiffStack(const std::string&);
    template ImageStack<double> readTiffStack(const std::string&);

    template void writeTiff(const std::string&, const Image<uint8_t>&, TiffCompression);
    template void writeTiff(const std::string&, const Image<int8_t>&, TiffCompression);
    template void writeTiff(const std::string&, const Image<uint16_t>&, TiffCompression);
    template void writeTiff(const std::string&, const Image<int16_t>&, TiffCompression);
    template void writeTiff(const std::string&, const Image<uint32_t>&, TiffCompression);
    template void writeTiff(const std::string&, const Image<int32_t>&, TiffCompression);
    template void writeTiff(const std::string&, const Image<float>&, TiffCompression);
    template void writeTiff(const std::string&, const Image<double>&, TiffCompression);

    template void writeTiffStack(const std::string&, const ImageStack<uint8_t>&, TiffCompression);
    template void writeTiffStack(const std::string&, const ImageStack<int8_t>&, TiffCompression);
    template void writeTiffStack(const std::string&, const ImageStack<uint16_t>&, TiffCompression);
    template void writeTiffStack(const std::string&, const ImageStack<int16_t>&, TiffCompression);
    template void writeTiffStack(const std::string&, const ImageStack<uint32_t>&, TiffCompression);
    template void writeTiffStack(const std::string&, const ImageStack<int32_t>&, TiffCompression);
    template void writeTiffStack(const std::string&, const ImageStack<float>&, TiffCompression);
    template void writeTiffStack(const std::string&, const ImageStack<double>&, TiffCompression);

} // namespace sirius