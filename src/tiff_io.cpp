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
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <tiffio.h>

namespace sirius {

    // --- pixel types --------------------------------------------------------

    std::size_t bytesPerPixel(PixelType t) noexcept {
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

    const char* toString(PixelType t) noexcept {
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
        // the resulting "Unknown field" warnings are expected and would otherwise
        // be printed once per page per reader thread. Anything else falls through
        // to libtiff's global warning handler, so real warnings stay visible.
        int warningFilter(TIFF*, void*, const char*, const char* fmt, va_list) {
            if (fmt && std::strstr(fmt, "Unknown field with tag")) return 1;   // handled
            return 0;                                                          // let the global handler print it
        }

        struct OpenOptionsDeleter {
            void operator()(TIFFOpenOptions* o) const { TIFFOpenOptionsFree(o); }
        };

        // The file is closed when TiffPtr goes out of scope, normally or via exception.
        TiffPtr openTiff(const std::string& path, const char* mode) {
            std::unique_ptr<TIFFOpenOptions, OpenOptionsDeleter> opts(TIFFOpenOptionsAlloc());
            if (opts) TIFFOpenOptionsSetWarningHandlerExtR(opts.get(), warningFilter, nullptr);
            TiffPtr tif(TIFFOpenExt(path.c_str(), mode, opts.get()));
            if (!tif) throw IoError("Failed to open TIFF: " + path);
            return tif;
        }

        PixelType pixelTypeFrom(uint16_t bps, uint16_t fmt) {
            switch (fmt) {
                case SAMPLEFORMAT_IEEEFP:
                    if (bps == 32) return PixelType::Float32;
                    if (bps == 64) return PixelType::Float64;
                    throw IoError("Unsupported float bit depth: " + std::to_string(bps));
                case SAMPLEFORMAT_INT:
                    if (bps == 8) return PixelType::Int8;
                    if (bps == 16) return PixelType::Int16;
                    if (bps == 32) return PixelType::Int32;
                    throw IoError("Unsupported integer bit depth: " + std::to_string(bps));
                default: // SAMPLEFORMAT_UINT and the (common) unspecified case
                    if (bps == 8) return PixelType::UInt8;
                    if (bps == 16) return PixelType::UInt16;
                    if (bps == 32) return PixelType::UInt32;
                    throw IoError("Unsupported integer bit depth: " + std::to_string(bps));
            }
        }

        // Metadata of the directory `tif` currently points at.
        TiffImageInfo readImageInfo(TIFF* tif) {
            TiffImageInfo info;
            info.ifdOffset = TIFFCurrentDirOffset(tif);

            uint16_t bps = 0, fmt = SAMPLEFORMAT_UINT, planar = PLANARCONFIG_CONTIG;
            uint32_t subfileType = 0;
            if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &info.width))
                throw IoError("TIFF missing required tag: IMAGEWIDTH");
            if (!TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &info.height))
                throw IoError("TIFF missing required tag: IMAGELENGTH");
            if (!TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps))
                throw IoError("TIFF missing required tag: BITSPERSAMPLE");
            TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &info.samplesPerPixel);
            TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &fmt);
            TIFFGetFieldDefaulted(tif, TIFFTAG_COMPRESSION, &info.compression);
            // The predictor tag only exists for codecs that register it
            // (LZW/Deflate/...); for others libtiff reports nothing.
            if (!TIFFGetField(tif, TIFFTAG_PREDICTOR, &info.predictor) || info.predictor == 0)
                info.predictor = 1;
            TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &planar);
            TIFFGetFieldDefaulted(tif, TIFFTAG_SUBFILETYPE, &subfileType);

            if (info.samplesPerPixel != 1)
                throw IoError("Only single-channel (grayscale) TIFFs are supported.");
            if (planar != PLANARCONFIG_CONTIG)
                throw IoError("Only contiguous (chunky) planar configuration is supported.");
            info.pixelType = pixelTypeFrom(bps, fmt);

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
        // Raw region decoding (native pixel type). The libtiff-facing code is
        // deliberately untemplated: one instantiation decodes every pixel type
        // as bytes, and conversion -- when the caller wants another type --
        // runs afterwards on the dense native page (see decodeWithLibtiff).
        // ------------------------------------------------------------------

        [[noreturn]] void throwReadError(const char* what, uint32_t x, uint32_t y) {
            throw IoError(std::string("Failed to read TIFF ") + what + " at (" +
                          std::to_string(x) + "," + std::to_string(y) + ")");
        }

        // Strips are full-width, so a strip whose wanted rows begin at its own
        // first row decodes straight into dst (no bounce buffer). Only strips
        // that start above the region, or when the region is narrower than
        // the image, go through `scratch`.
        void readStripsRegion(TIFF* tif, const TiffImageInfo& g, const Region& r, uint8_t* dst,
                              std::vector<uint8_t>& scratch) {
            const std::size_t bpp = bytesPerPixel(g.pixelType);
            const std::size_t dstPitch = static_cast<std::size_t>(r.width) * bpp;
            const std::size_t srcPitch = static_cast<std::size_t>(g.width) * bpp;

            uint32_t rowsPerStrip = 0;
            TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip);
            if (rowsPerStrip == 0 || rowsPerStrip > g.height) rowsPerStrip = g.height;
            const tmsize_t stripSize = TIFFStripSize(tif);
            if (stripSize <= 0) throw IoError("TIFF reports invalid strip size");

            const uint32_t yEnd = r.y + r.height;
            const bool fullWidth = (r.x == 0 && r.width == g.width);

            for (uint32_t stripRow0 = (r.y / rowsPerStrip) * rowsPerStrip; stripRow0 < yEnd; stripRow0 += rowsPerStrip) {
                const tstrip_t strip = TIFFComputeStrip(tif, stripRow0, 0);
                const uint32_t stripRows = std::min(rowsPerStrip, g.height - stripRow0);
                const uint32_t y0 = std::max(stripRow0, r.y);
                const uint32_t y1 = std::min(stripRow0 + stripRows, yEnd);

                if (fullWidth && y0 == stripRow0) {
                    uint8_t* out = dst + static_cast<std::size_t>(y0 - r.y) * dstPitch;
                    const tmsize_t bytes = static_cast<tmsize_t>(y1 - y0) * static_cast<tmsize_t>(srcPitch);
                    if (TIFFReadEncodedStrip(tif, strip, out, bytes) < 0) throwReadError("strip", 0, stripRow0);
                } else {
                    scratch.resize(static_cast<std::size_t>(stripSize));
                    // Decode only through the last row we need.
                    const tmsize_t bytes = static_cast<tmsize_t>(y1 - stripRow0) * static_cast<tmsize_t>(srcPitch);
                    if (TIFFReadEncodedStrip(tif, strip, scratch.data(), bytes) < 0) throwReadError("strip", 0, stripRow0);
                    for (uint32_t y = y0; y < y1; ++y)
                        std::memcpy(dst + static_cast<std::size_t>(y - r.y) * dstPitch,
                                    scratch.data() + static_cast<std::size_t>(y - stripRow0) * srcPitch + r.x * bpp,
                                    dstPitch);
                }
            }
        }

        // Tiled layout: every tile intersecting the region is decoded exactly
        // once and its intersecting rows are copied out.
        void readTilesRegion(TIFF* tif, const TiffImageInfo& g, const Region& r, uint8_t* dst,
                             std::vector<uint8_t>& scratch) {
            uint32_t tileW = 0, tileH = 0;
            if (!TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileW) || tileW == 0)
                throw IoError("TIFF missing or invalid TILEWIDTH");
            if (!TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileH) || tileH == 0)
                throw IoError("TIFF missing or invalid TILELENGTH");
            const tmsize_t tileSize = TIFFTileSize(tif);
            if (tileSize <= 0) throw IoError("TIFF reports invalid tile size");
            scratch.resize(static_cast<std::size_t>(tileSize));

            const std::size_t bpp = bytesPerPixel(g.pixelType);
            const std::size_t dstPitch = static_cast<std::size_t>(r.width) * bpp;
            const std::size_t tilePitch = static_cast<std::size_t>(tileW) * bpp;
            const uint32_t xEnd = r.x + r.width;
            const uint32_t yEnd = r.y + r.height;

            for (uint32_t ty = (r.y / tileH) * tileH; ty < yEnd; ty += tileH) {
                for (uint32_t tx = (r.x / tileW) * tileW; tx < xEnd; tx += tileW) {
                    if (TIFFReadTile(tif, scratch.data(), tx, ty, 0, 0) < 0) throwReadError("tile", tx, ty);
                    const uint32_t y0 = std::max(ty, r.y), y1 = std::min(ty + tileH, yEnd);
                    const uint32_t x0 = std::max(tx, r.x), x1 = std::min(tx + tileW, xEnd);
                    const std::size_t rowBytes = static_cast<std::size_t>(x1 - x0) * bpp;
                    for (uint32_t y = y0; y < y1; ++y)
                        std::memcpy(dst + static_cast<std::size_t>(y - r.y) * dstPitch + (x0 - r.x) * bpp,
                                    scratch.data() + static_cast<std::size_t>(y - ty) * tilePitch + (x0 - tx) * bpp,
                                    rowBytes);
                }
            }
        }

        void readRegionRaw(TIFF* tif, const TiffImageInfo& g, const Region& r, uint8_t* dst,
                           std::vector<uint8_t>& scratch) {
            if (TIFFIsTiled(tif)) readTilesRegion(tif, g, r, dst, scratch);
            else readStripsRegion(tif, g, r, dst, scratch);
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
            outRows = static_cast<uint32_t>(downsampledExtent(rows, f));
            outCols = static_cast<uint32_t>(downsampledExtent(cols, f));
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
            if (p.width != p0.width || p.height != p0.height || p.pixelType != p0.pixelType) return false;
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
        if (info.pages.empty())   // every IFD flagged reduced: treat the chain as pages anyway
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
        // style): consecutive reduced IFDs of one size form a level.
        for (std::size_t i = 0; i < chainCount; ++i) {
            const auto& img = info.images[i];
            if (!img.reducedResolution) continue;
            TiffLevel* last = info.levels.size() > 1 ? &info.levels.back() : nullptr;
            const bool sameAsLast = last && last->width == img.width && last->height == img.height &&
                                    info.image(last->ifds.back()).reducedResolution &&
                                    info.image(last->ifds.back()).subIfds.empty() &&
                                    std::find(info.pages.begin(), info.pages.end(), last->ifds.back()) == info.pages.end();
            if (sameAsLast && last->ifds.size() < info.pages.size()) {
                last->ifds.push_back(img.ifdOffset);
            } else {
                TiffLevel level;
                level.width = img.width;
                level.height = img.height;
                level.ifds.push_back(img.ifdOffset);
                info.levels.push_back(std::move(level));
            }
        }
        return info;
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

        // Parallel over pages. Each thread opens its own handle once and
        // reuses it for every page it processes: libtiff handles are not
        // thread-safe, but one handle can hop between directories with
        // TIFFSetSubDirectory without reopening the file.
        void decodeWithLibtiff(const std::string& path, const DecodeJob& job, void* dstHost) {
            const auto& ifds = *job.ifds;
            const TiffImageInfo& g = *job.geometry;
            const Region r = job.region;
            const auto n = static_cast<std::ptrdiff_t>(ifds.size());
            const std::size_t pixels = static_cast<std::size_t>(r.width) * r.height;
            const std::size_t nativePageBytes = pixels * bytesPerPixel(g.pixelType);
            const std::size_t dstPageBytes = pixels * bytesPerPixel(job.dstType);
            const bool needConvert = g.pixelType != job.dstType;
            auto* dst = static_cast<std::uint8_t*>(dstHost);

            std::exception_ptr ex;
            std::atomic<bool> failed{false};

#pragma omp parallel
            {
                TiffPtr localTif;
                bool openOk = false;
                try {
                    localTif = openTiff(path, "r");
                    openOk = true;
                } catch (...) {
#pragma omp critical
                    {
                        if (!ex) ex = std::current_exception();
                    }
                    failed.store(true, std::memory_order_relaxed);
                }

                std::vector<std::uint8_t> scratch;      // one strip / tile
                std::vector<std::uint8_t> nativePage;   // conversion path only

#pragma omp for schedule(dynamic, 4)
                for (std::ptrdiff_t z = 0; z < n; ++z) {
                    if (failed.load(std::memory_order_relaxed) || !openOk) continue;
                    try {
                        if (!TIFFSetSubDirectory(localTif.get(), ifds[static_cast<std::size_t>(z)]))
                            throw IoError("Failed to seek to TIFF directory at offset " +
                                          std::to_string(ifds[static_cast<std::size_t>(z)]));
                        std::uint8_t* out = dst + static_cast<std::size_t>(z) * dstPageBytes;
                        if (needConvert) {
                            nativePage.resize(nativePageBytes);
                            readRegionRaw(localTif.get(), g, r, nativePage.data(), scratch);
                            convertPixels(nativePage.data(), g.pixelType, out, job.dstType,
                                          static_cast<Index>(pixels), Device::cpu(), Stream::null());
                        } else {
                            readRegionRaw(localTif.get(), g, r, out, scratch);
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
            if (detail::decodeWithNvTiff(impl, job, dst, device, stream, reason)) return;
            if (!opts.allowCpuFallback)
                throw std::runtime_error("GPU decode of " + impl.path + " is not possible: " + reason +
                                         " (TiffReadOptions::allowCpuFallback is off)");

            // Fallback: libtiff into pinned staging, uploaded chunk by chunk so
            // arbitrarily large stacks never need a stack-sized host buffer.
            const auto& ifds = *job.ifds;
            const std::size_t n = ifds.size();
            const std::size_t pageBytes = static_cast<std::size_t>(job.region.width) * job.region.height *
                                          bytesPerPixel(job.dstType);
            constexpr std::size_t kChunkBytes = std::size_t{512} << 20;
            const std::size_t chunk = std::min(n, std::max<std::size_t>(1, kChunkBytes / std::max<std::size_t>(pageBytes, 1)));
            Buffer<std::uint8_t> staging(Shape{static_cast<Index>(chunk * pageBytes)}, Device::cpu(),
                                         HostMemory::Pinned);
            for (std::size_t first = 0; first < n; first += chunk) {
                const std::size_t count = std::min(chunk, n - first);
                const std::vector<std::uint64_t> part(ifds.begin() + static_cast<std::ptrdiff_t>(first),
                                                      ifds.begin() + static_cast<std::ptrdiff_t>(first + count));
                detail::DecodeJob sub = job;
                sub.ifds = &part;
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
            job.region = Region{}.resolve(info.width(), info.height());
            job.dstType = info.pixelType();
            try {
                ok = detail::nvTiffSupports(*impl_, job, device, why);
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
        for (std::uint64_t off : ifds) {
            const auto& i = info.image(off);
            if (i.width != g.width || i.height != g.height || i.pixelType != g.pixelType)
                throw IoError("TIFF image at offset " + std::to_string(off) + " (" +
                              std::to_string(i.width) + "x" + std::to_string(i.height) + " " +
                              toString(i.pixelType) + ") does not match the first one (" +
                              std::to_string(g.width) + "x" + std::to_string(g.height) + " " +
                              toString(g.pixelType) + ")");
        }
        const Region r = region.resolve(g.width, g.height);
        const Shape expected{static_cast<Index>(ifds.size()), static_cast<Index>(r.height), static_cast<Index>(r.width)};
        if (dst.shape() != expected) detail::throwShapeMismatch("TiffFile::decode destination", dst.shape(), expected);

        detail::DecodeJob job;
        job.ifds = &ifds;
        job.geometry = &g;
        job.region = r;
        job.dstType = pixelTypeOf<T>();
        decodeInto(*impl_, job, dst.data(), dst.device(), opts, stream);
    }

    template <typename T>
    Buffer<T> TiffFile::readStack(const TiffReadOptions& opts, const Stream& stream) const {
        if (!impl_->info.uniformPages())
            throw IoError("TIFF pages differ in size or pixel type; read them individually: " + impl_->path);
        Buffer<T> out(stackShape(impl_->info), opts.device, opts.hostMemory, stream);
        decode<T>(impl_->info.pages, Region{}, out.view(), opts, stream);
        return out;
    }

    template <typename T>
    Buffer<T> TiffFile::readPages(std::size_t first, std::size_t count, const TiffReadOptions& opts,
                                  const Stream& stream) const {
        const auto& pages = impl_->info.pages;
        if (first + count > pages.size() || count == 0)
            throw std::out_of_range("Pages [" + std::to_string(first) + ", " + std::to_string(first + count) +
                                    ") requested from a TIFF with " + std::to_string(pages.size()) + " page(s)");
        const std::vector<std::uint64_t> ifds(pages.begin() + static_cast<std::ptrdiff_t>(first),
                                              pages.begin() + static_cast<std::ptrdiff_t>(first + count));
        const auto& g = impl_->info.image(ifds[0]);
        Buffer<T> out(Shape{static_cast<Index>(count), static_cast<Index>(g.height), static_cast<Index>(g.width)},
                      opts.device, opts.hostMemory, stream);
        decode<T>(ifds, Region{}, out.view(), opts, stream);
        return out;
    }

    template <typename T>
    Buffer<T> TiffFile::readLevel(std::size_t level, const TiffReadOptions& opts, const Stream& stream) const {
        const TiffLevel& l = levelAt(impl_->info, level);
        Buffer<T> out(levelShape(l), opts.device, opts.hostMemory, stream);
        decode<T>(l.ifds, Region{}, out.view(), opts, stream);
        return out;
    }

    template <typename T>
    Buffer<T> TiffFile::readRegion(Region region, std::size_t level, const TiffReadOptions& opts,
                                   const Stream& stream) const {
        const TiffLevel& l = levelAt(impl_->info, level);
        const Region r = region.resolve(l.width, l.height);
        Buffer<T> out(Shape{static_cast<Index>(l.ifds.size()), static_cast<Index>(r.height), static_cast<Index>(r.width)},
                      opts.device, opts.hostMemory, stream);
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

    template <typename T>
    Image<T> readTiff(const std::string& path) {
        TiffFile file(path);
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
        ImageStack<T> stack(static_cast<Eigen::Index>(info.pageCount()), info.height(), info.width());
        file.decode<T>(info.pages, Region{}, toView(stack));
        return stack;
    }

    AnyImageStack readTiffStackAny(const std::string& path) {
        TiffFile file(path);
        const TiffInfo& info = file.info();
        if (!info.uniformPages())
            throw IoError("TIFF pages differ in size or pixel type: " + path);
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
#define SIRIUS_TIFF_INSTANTIATE(T)                                                                                    \
    template void TiffFile::decode<T>(const std::vector<std::uint64_t>&, Region, BufferView<T>,                       \
                                      const TiffReadOptions&, const Stream&) const;                                   \
    template Buffer<T> TiffFile::readStack<T>(const TiffReadOptions&, const Stream&) const;                           \
    template Buffer<T> TiffFile::readPages<T>(std::size_t, std::size_t, const TiffReadOptions&, const Stream&) const; \
    template Buffer<T> TiffFile::readLevel<T>(std::size_t, const TiffReadOptions&, const Stream&) const;              \
    template Buffer<T> TiffFile::readRegion<T>(Region, std::size_t, const TiffReadOptions&, const Stream&) const;     \
    template Image<T> readTiff<T>(const std::string&);                                                                \
    template ImageStack<T> readTiffStack<T>(const std::string&);                                                      \
    template void writeTiff<T>(const std::string&, BufferView<const T>, TiffCompression);                             \
    template void writeTiff<T>(const std::string&, const Image<T>&, TiffCompression);                                 \
    template void writeTiffStack<T>(const std::string&, BufferView<const T>, TiffCompression);                        \
    template void writeTiffStack<T>(const std::string&, BufferView<const T>, const TiffWriteOptions&);                \
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
