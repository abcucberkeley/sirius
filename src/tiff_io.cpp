#include "sirius/tiff_io.hpp"
#include <tiffio.h>
#include <cstring>
#include <type_traits>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <atomic>

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
            TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &info.width);
            TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &info.height);
            TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &info.bps);
            TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &info.spp);
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
            for (uint32_t i = 0; i < width; ++i) {
                if (fmt == SAMPLEFORMAT_IEEEFP) {
                    if      (bps == 32) { float v;  std::memcpy(&v, src + i*4, 4); dst[i] = static_cast<T>(v); }
                    else if (bps == 64) { double v; std::memcpy(&v, src + i*8, 8); dst[i] = static_cast<T>(v); }
                } else if (fmt == SAMPLEFORMAT_INT) {
                    if      (bps == 8)  { int8_t  v; std::memcpy(&v, src + i,   1); dst[i] = static_cast<T>(v); }
                    else if (bps == 16) { int16_t v; std::memcpy(&v, src + i*2, 2); dst[i] = static_cast<T>(v); }
                    else if (bps == 32) { int32_t v; std::memcpy(&v, src + i*4, 4); dst[i] = static_cast<T>(v); }
                } else { // SAMPLEFORMAT_UINT
                    if      (bps == 8)  { dst[i] = static_cast<T>(src[i]); }
                    else if (bps == 16) { uint16_t v; std::memcpy(&v, src + i*2, 2); dst[i] = static_cast<T>(v); }
                    else if (bps == 32) { uint32_t v; std::memcpy(&v, src + i*4, 4); dst[i] = static_cast<T>(v); }
                }
            }
        }

        // check if Eigen matrix type T is an exact match
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

        // Scanline/strip layout: stored row-by-row, TIFFReadScanline handles both fine
        template <typename T>
        void readScanlinePage(TIFF* tif, T* dst, const TiffPageInfo& info) {
            std::vector<uint8_t> buf(static_cast<size_t>(TIFFScanlineSize(tif)));

            // if tiff format is an exact match to the Eigen matrix, we will just memcpy
            bool useFastPath = isExactMatch<T>(info.bps, info.fmt);

            // scanline is a row
            for (uint32_t row = 0; row < info.height; ++row) {

                // read row into buffer
                if (TIFFReadScanline(tif, buf.data(), row) < 0)
                    throw std::runtime_error("Failed to read scanline " + std::to_string(row));

                // location in dst
                T* rowDst = dst + row * info.width;

                if (useFastPath) {
                    std::memcpy(rowDst, buf.data(), info.width * sizeof(T));
                } else {
                    // convert to appropriate format and copy to the correct location on the Eigen dst
                    convertScanline<T>(buf.data(), rowDst, info.width, info.bps, info.fmt);
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
            TIFFGetField(tif, TIFFTAG_TILEWIDTH,  &tileW);
            TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileH);

            // allocate once outside the loop
            std::vector<uint8_t> tileBuf(static_cast<size_t>(TIFFTileSize(tif)));
            const uint32_t bytesPerPixel = info.bps / 8;

            // use fast path if there is exact match between eigen type and the tiff type
            bool useFastPath = isExactMatch<T>(info.bps, info.fmt);

            for (uint32_t tileRow = 0; tileRow < info.height; tileRow += tileH) {
                for (uint32_t tileCol = 0; tileCol < info.width; tileCol += tileW) {
                    if (TIFFReadTile(tif, tileBuf.data(), tileCol, tileRow, 0, 0) < 0)
                        throw std::runtime_error("Failed to read tile at (" +
                            std::to_string(tileCol) + "," + std::to_string(tileRow) + ")");

                    // edge tiles are padded to full tile size — clamp to actual image bounds
                    uint32_t validH = std::min(tileH, info.height - tileRow);
                    uint32_t validW = std::min(tileW, info.width  - tileCol);

                    for (uint32_t r = 0; r < validH; ++r) {
                        const uint8_t* srcRow = tileBuf.data() + r * tileW * bytesPerPixel;
                        T* dstRow = dst + (tileRow + r) * info.width + tileCol;

                        if (useFastPath) {
                            std::memcpy(dstRow, srcRow, validW * sizeof(T));
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

            for (uint32_t row = 0; row < height; ++row) {
                if (TIFFWriteScanline(tif, const_cast<void*>(static_cast<const void*>(src + row * width)), row) < 0)
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
                        static_cast<uint32_t>(image.rows()),
                        static_cast<uint32_t>(image.cols()), false, comp);
    }

    // --- Image stack ---

    template <typename T>
    ImageStack<T> readTiffStack(const std::string& path) {
        Eigen::Index pageCount = 0;
        Eigen::Index rows = 0;
        Eigen::Index cols = 0;

        // Pass 1: Light-weight sequential read to grab dimensions and count pages
        bool tiled = false; // check if tiled
        std::vector<uint64_t> offset;
        {
            auto tif = openTiff(path, "r");
            auto info = getPageInfo(tif.get());
            rows = static_cast<Eigen::Index>(info.height);
            cols = static_cast<Eigen::Index>(info.width);
            tiled = TIFFIsTiled(tif.get());

            do {
                // total page count needed for allocation size
                ++pageCount; 
                // offset needed since pages are like linked lists
                // so we should avoid the n^2 penalty of setting set directory in a parallel loop later on
                offset.push_back(TIFFCurrentDirOffset(tif.get()));
            } while (TIFFReadDirectory(tif.get()));
        }

        if (pageCount > 65535) {
            // Note: Technically BigTIFF allows more, but standard TIFF caps at 65535.
            throw std::runtime_error("TIFF stack exceeds expected maximum directory count.");
        }

        // Allocate the contiguous memory block once
        ImageStack<T> stack(pageCount, rows, cols);

        // Parallel thread pool opening individual file handles
        std::exception_ptr ex;
        std::atomic<bool> failed{false};

        #pragma omp parallel for schedule(dynamic)
        for (Eigen::Index z = 0; z < pageCount; ++z) {
            if (failed.load()) continue; 
            try {
                auto localTif = openTiff(path, "r");
                TIFFSetSubDirectory(localTif.get(), offset[z]);
                auto info = getPageInfo(localTif.get());
                T* dst = stack.data() + z * stack.stride();
                if (tiled) readTiledPage<T>(localTif.get(), dst, info);
                else       readScanlinePage<T>(localTif.get(), dst, info);
            } catch (...) {
                #pragma omp critical
                if (!ex) ex = std::current_exception();
                failed.store(true); 
            }
        }
        if (ex) std::rethrow_exception(ex);

        return stack;
    }

    template <typename T>
    void writeTiffStack(const std::string& path, const ImageStack<T>& stack, TiffCompression comp) {
        if (stack.empty())
            throw std::runtime_error("Cannot write empty stack");

        // use bigtiff always
        auto tif = openTiff(path, "w8");
        auto height = static_cast<uint32_t>(stack.rows());
        auto width = static_cast<uint32_t>(stack.cols());

        for (Eigen::Index z = 0; z < stack.depth(); ++z) {
            writePageFrom<T>(tif.get(), stack.data() + z * stack.stride(),
                            height, width, true, comp);
            TIFFWriteDirectory(tif.get());
        }
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