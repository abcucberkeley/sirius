#ifndef SIRIUS_TIFF_IO_HPP
#define SIRIUS_TIFF_IO_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include "sirius/buffer.hpp"
#include "sirius/device.hpp"
#include "sirius/pixel_type.hpp"
#include "sirius/tiff_metadata.hpp"

// TIFF reading and writing for scientific stacks.
//
// What is read: strips and tiles, classic TIFF and BigTIFF, either byte
// order, multi-page stacks, SubIFD and flat pyramids, any number of samples
// per pixel (RGB, RGBA, multi-sample) in either planar configuration, 1..32
// bit unsigned integers (1, 2, 4, 12, 14 ... bit samples unpacked into the
// next byte size), 8/16/24/32-bit signed integers, 16/32/64-bit floats
// (float16 widened to float32), the codecs libtiff was built with (None,
// LZW, Deflate / Adobe Deflate, PackBits, CCITT for bilevel; ZSTD, JPEG and
// LZMA only when libtiff has them) with the horizontal and floating-point
// predictors, sparse files (missing strips / tiles read as zeros), palette
// images (the indices; TiffImageInfo::colormap holds the palette), and the
// OME-XML / ImageJ metadata (tiff_metadata.hpp).
//
// Samples are channels: a read of more than one sample returns
// {pages, samples, height, width} -- each sample its own plane, whatever the
// planar configuration on disk -- and a read of one sample {pages, height,
// width}. TiffReadOptions::firstSample / sampleCount pick the samples.
//
// Two layers:
//  * TiffFile        opens a file once, exposes its metadata (pages, pyramid
//                    levels, layout, codec) and decodes pages / levels /
//                    regions into a Buffer on any Device. On CUDA devices the
//                    decode runs on the GPU through nvTIFF (strips and tiles,
//                    None/LZW/Deflate, BigTIFF) and lands directly in device
//                    memory; files nvTIFF cannot handle fall back to libtiff
//                    plus an upload unless TiffReadOptions says otherwise.
//  * free functions  the original Eigen-tensor API (readTiff, readTiffStack,
//                    writeTiff, ...), kept for convenience and implemented on
//                    top of TiffFile.

namespace sirius {

    // Row major so the inner most dim is contig maching tiff scan layyout
    template <typename T>
    using Image = Eigen::Tensor<T, 2, Eigen::RowMajor>;

    template <typename T>
    using ImageStack = Eigen::Tensor<T, 3, Eigen::RowMajor>;

    // Row-major matrix type returned by asMatrix/slice views. Exposes the
    // Eigen matrix API (isApprox, operator==, comma-init, block ops, etc.)
    template <typename T>
    using ImageMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // Need to be able to dispatch the correct reader based on tiff data
    using AnyImageStack = std::variant<ImageStack<uint8_t>, ImageStack<int8_t>, ImageStack<uint16_t>, ImageStack<int16_t>,
                                       ImageStack<uint32_t>, ImageStack<int32_t>, ImageStack<float>, ImageStack<double>>;

    // Same set of pixel types as owning device-agnostic buffers.
    using AnyBuffer = std::variant<Buffer<uint8_t>, Buffer<int8_t>, Buffer<uint16_t>, Buffer<int16_t>,
                                   Buffer<uint32_t>, Buffer<int32_t>, Buffer<float>, Buffer<double>>;

    // Compression options for writing
    enum class TiffCompression {
        None,
        Lzw,
        Deflate // Often referred to as ZIP
    };

    // Zero-copy 2D dense-matrix view over an Image<T>.
    template <typename T>
    inline Eigen::Map<ImageMatrix<T>> asMatrix(Image<T>& img) {
        return {img.data(), img.dimension(0), img.dimension(1)};
    }
    template <typename T>
    inline Eigen::Map<const ImageMatrix<T>> asMatrix(const Image<T>& img) {
        return {img.data(), img.dimension(0), img.dimension(1)};
    }

    // Zero-copy 2D dense-matrix view over page z of an ImageStack<T>.
    template <typename T>
    inline Eigen::Map<ImageMatrix<T>> slice(ImageStack<T>& stack, Eigen::Index z) {
        const auto rows = stack.dimension(1);
        const auto cols = stack.dimension(2);
        return {stack.data() + z * rows * cols, rows, cols};
    }
    template <typename T>
    inline Eigen::Map<const ImageMatrix<T>> slice(const ImageStack<T>& stack, Eigen::Index z) {
        const auto rows = stack.dimension(1);
        const auto cols = stack.dimension(2);
        return {stack.data() + z * rows * cols, rows, cols};
    }

    // --- metadata ---------------------------------------------------------

    enum class TiffLayout : std::uint8_t { Strips,
                                           Tiles };

    // Metadata of one image file directory (IFD).
    struct TiffImageInfo {
        std::uint64_t ifdOffset = 0;
        std::string description;             // ImageDescription tag (OME-XML, ImageJ metadata)
        double xResolution = 0.0;            // XResolution / YResolution tags (pixels per resolutionUnit), 0 = absent
        double yResolution = 0.0;
        std::uint16_t resolutionUnit = 2;    // 1 none, 2 inch, 3 centimetre
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        PixelType pixelType = PixelType::UInt8;   // of one decoded sample (12-bit -> UInt16, float16 -> Float32)
        std::uint16_t samplesPerPixel = 1;
        std::uint16_t bitsPerSample = 8;
        std::uint16_t sampleFormat = 1;      // 1 unsigned, 2 signed, 3 IEEE float (4 "void" reads as unsigned)
        std::uint16_t photometric = 1;       // 0 min-is-white, 1 min-is-black, 2 RGB, 3 palette, 5 CMYK, 6 YCbCr ...
        std::uint16_t planarConfig = 1;      // 1 contiguous (RGBRGB...), 2 separate planes (RRR...GGG...)
        std::uint16_t orientation = 1;       // Orientation tag; 1 = rows top to bottom, columns left to right
        std::vector<std::uint16_t> extraSamples;   // ExtraSamples: 0 unspecified, 1 associated alpha, 2 unassociated alpha
        std::vector<std::uint16_t> colormap;       // palette images: 3 * 2^bitsPerSample entries, all R then G then B
        std::uint16_t compression = 1;       // raw Compression tag: 1 none, 5 LZW, 8/32946 Deflate, 7 JPEG ...
        std::uint16_t predictor = 1;         // 1 none, 2 horizontal differencing, 3 floating point
        TiffLayout layout = TiffLayout::Strips;
        std::uint32_t tileWidth = 0;         // tiles only
        std::uint32_t tileHeight = 0;
        std::uint32_t rowsPerStrip = 0;      // strips only
        bool reducedResolution = false;      // NewSubfileType bit 0: a pyramid level, not a page
        std::vector<std::uint64_t> subIfds;  // SubIFD tag (330): reduced-resolution children
        // Why this IFD cannot be decoded (a codec libtiff was built without,
        // a 64-bit integer or complex sample, ...); "" when it can. Inspection
        // never fails on such an IFD -- a file may hold one next to readable
        // pages -- a read of it throws with this text.
        std::string unsupported;

        bool decodable() const noexcept { return unsupported.empty(); }
    };

    // One resolution level of a (possibly multi-page) pyramid: the IFDs that
    // hold every page at this resolution, in page order.
    struct TiffLevel {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::vector<std::uint64_t> ifds;
    };

    struct TiffInfo {
        bool bigTiff = false;
        bool bigEndian = false;              // "MM" byte order (decoded to the host's)
        std::vector<TiffImageInfo> images;   // every IFD found: main chain first, then SubIFDs
        std::vector<std::uint64_t> pages;    // full-resolution pages (the "stack"), in file order
        // levels[0] is the full-resolution stack (== pages). levels[k] is the
        // k-th reduction of every page, discovered from SubIFDs or from
        // reduced-resolution IFDs in the main chain.
        std::vector<TiffLevel> levels;
        // ifdOffset -> index into `images`, filled by inspectTiff so image()
        // is O(1) on stacks with thousands of pages (the per-IFD validation
        // of every read goes through it). image() falls back to a linear
        // search for offsets missing here, e.g. in an info assembled by hand.
        std::unordered_map<std::uint64_t, std::size_t> imageIndex;

        const TiffImageInfo& image(std::uint64_t ifdOffset) const;   // throws if unknown
        const TiffImageInfo& page(std::size_t i) const { return image(pages.at(i)); }
        std::size_t pageCount() const noexcept { return pages.size(); }
        std::size_t levelCount() const noexcept { return levels.size(); }
        PixelType pixelType() const { return page(0).pixelType; }
        std::uint32_t width() const { return page(0).width; }
        std::uint32_t height() const { return page(0).height; }
        std::uint16_t samplesPerPixel() const { return page(0).samplesPerPixel; }
        // All pages share width, height, pixel type and samples per pixel
        // (required for readStack).
        bool uniformPages() const noexcept;
    };

    TiffInfo inspectTiff(const std::string& path);

    // Width, height and page count without walking every IFD's tags. Used when
    // describing a folder of stacks: a full inspectTiff of every file would
    // open hundreds of multi-page TIFFs just to confirm they match.
    struct TiffStackShape {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::size_t pages = 0;
        PixelType pixelType = PixelType::UInt8;
        std::uint16_t samplesPerPixel = 1;
    };
    TiffStackShape inspectTiffShape(const std::string& path);

    // Rectangle inside an image, in pixels. width/height of 0 extend to the edge.
    struct Region {
        std::uint32_t x = 0;
        std::uint32_t y = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;

        bool full() const noexcept { return x == 0 && y == 0 && width == 0 && height == 0; }
        // Concrete extents for an image of the given size; throws if out of bounds.
        Region resolve(std::uint32_t imageWidth, std::uint32_t imageHeight) const;
    };

    struct TiffReadOptions {
        Device device = Device::cpu();
        HostMemory hostMemory = HostMemory::Pageable;  // for results on the CPU
        // CUDA: when nvTIFF cannot decode the file (unsupported codec/predictor,
        // nvCOMP missing for Deflate, ...) decode with libtiff and upload
        // instead of throwing.
        bool allowCpuFallback = true;
        // libtiff page loop: 0 = size the OpenMP team from the work (about
        // one thread per MiB), 1 = one handle, pages in file order. Folder
        // datasets use 1 so many files stream in parallel without seeking
        // around each stack.
        int maxThreads = 0;
        // Samples (channels) to read: [firstSample, firstSample + sampleCount);
        // sampleCount 0 = every sample from firstSample on. One sample reads
        // as {pages, height, width}, several as {pages, samples, height, width}.
        // With separate planes on disk only the chosen planes are decoded.
        std::uint16_t firstSample = 0;
        std::uint16_t sampleCount = 0;
        // 0..1 over pages. Called from the decode threads; keep it cheap.
        std::function<void(double)> progress;
    };

    // --- TiffFile ---------------------------------------------------------
    // Read-only handle. Reading functions return once the data is fully
    // decoded (they synchronize `stream` before returning), so the result can
    // be consumed on the host or enqueued on any stream immediately.
    class TiffFile {
    public:
        explicit TiffFile(std::string path);
        ~TiffFile();
        TiffFile(TiffFile&&) noexcept;
        TiffFile& operator=(TiffFile&&) noexcept;
        TiffFile(const TiffFile&) = delete;
        TiffFile& operator=(const TiffFile&) = delete;

        const std::string& path() const noexcept;
        const TiffInfo& info() const noexcept;
        // OME-XML / ImageJ metadata of the first page, parsed on first use.
        const TiffMetadata& metadata() const;
        // Full-resolution page indices of every series: one per OME <Image>
        // (omeImagePages), else one series of every page.
        std::vector<std::vector<std::uint32_t>> series() const;

        // The shape a read of `pages` pages of `height` x `width` returns
        // with `opts` (samples): {pages, h, w} or {pages, samples, h, w}.
        Shape readShape(std::size_t pages, std::uint32_t height, std::uint32_t width,
                        const TiffReadOptions& opts = {}) const;

        // Full-resolution stack: every page, shape {pages, height, width}
        // (or {pages, samples, height, width}; see readShape).
        template <typename T>
        Buffer<T> readStack(const TiffReadOptions& opts = {}, const Stream& stream = Stream::null()) const;

        // Pages [first, first + count).
        template <typename T>
        Buffer<T> readPages(std::size_t first, std::size_t count, const TiffReadOptions& opts = {},
                            const Stream& stream = Stream::null()) const;

        // Every page at pyramid level `level` (0 = full resolution).
        template <typename T>
        Buffer<T> readLevel(std::size_t level, const TiffReadOptions& opts = {},
                            const Stream& stream = Stream::null()) const;

        // Pyramid levels of series `index`: its pages' SubIFDs, else the
        // flat levels of TiffInfo::levels that hold every one of its pages.
        std::size_t seriesLevels(std::size_t index) const;
        // The IFDs of series `index` at pyramid level `level`, in plane order.
        std::vector<std::uint64_t> seriesIfds(std::size_t index, std::size_t level = 0) const;

        // Series `index` (series()) at pyramid level `level`.
        template <typename T>
        Buffer<T> readSeries(std::size_t index, std::size_t level = 0, const TiffReadOptions& opts = {},
                             const Stream& stream = Stream::null()) const;

        // `region` of every page at `level`; shape {pages, region.height, region.width}.
        template <typename T>
        Buffer<T> readRegion(Region region, std::size_t level = 0, const TiffReadOptions& opts = {},
                             const Stream& stream = Stream::null()) const;

        // Lowest level: decode `region` of each IFD (all must share size,
        // pixel type and samples per pixel) into dst of shape readShape(ifds.size(),
        // region.height, region.width, opts) on dst.device(). Pixels are
        // converted to T when the file type differs.
        template <typename T>
        void decode(const std::vector<std::uint64_t>& ifds, Region region, BufferView<T> dst,
                    const TiffReadOptions& opts = {}, const Stream& stream = Stream::null()) const;

        // Whether the GPU path (nvTIFF) can decode this file as it is. False
        // on CPU-only builds, without a GPU, or for unsupported codecs.
        bool gpuDecodable(Device device = Device::cuda(), std::string* reason = nullptr) const;

        struct Impl;   // public for the backend translation units, not for users

    private:
        std::unique_ptr<Impl> impl_;
    };

    // Decode any file into the buffer type matching its on-disk pixel type.
    AnyBuffer readTiffAny(const std::string& path, const TiffReadOptions& opts = {},
                          const Stream& stream = Stream::null());

    // --- writing with full control over the container ---------------------
    struct TiffWriteOptions {
        TiffCompression compression = TiffCompression::None;
        int compressionLevel = 6;            // Deflate 1..9
        // Horizontal differencing (integer types) / floating-point predictor
        // (float types) for LZW and Deflate. Off by default: nvTIFF cannot
        // decode the floating-point predictor.
        bool predictor = false;
        bool tiled = false;
        std::uint32_t tileWidth = 256;
        std::uint32_t tileHeight = 256;
        std::uint32_t rowsPerStrip = 0;      // strips: 0 = libtiff default
        bool bigTiff = true;
        std::string description;             // ImageDescription of the first page (OME-XML, ImageJ)
        // > 1 writes reduced-resolution SubIFDs (NewSubfileType 1) under every
        // page, each level box down-sampled by `downsample` in x and y.
        int pyramidLevels = 1;
        int downsample = 2;
        double xPixelUm = 0.0;               // > 0: written as XResolution / YResolution in cm
        double yPixelUm = 0.0;
        std::function<void(double)> progress;   // 0..1 over pages
        std::function<bool()> cancelled;        // checked between pages (the file is removed when cancelled)
    };

    // (pages, rows, cols) host or device view, any pixel type.
    template <typename T>
    void writeTiffStack(const std::string& path, BufferView<const T> stack, const TiffWriteOptions& options);

    // --- Eigen convenience API --------------------------------------------

    template <typename T>
    Image<T> readTiff(const std::string& path);

    template <typename T>
    ImageStack<T> readTiffStack(const std::string& path);

    // Call the correct function based on tiff data type
    // Usage:
    //     std::visit([](auto& img) {}, readTiffStackAny("file.tiff"));
    // Note: it is more efficient to call the correct function
    // if the underlying data is already known or data will be recast downstream
    AnyImageStack readTiffStackAny(const std::string& path);

    // Writers accept host or device views: rank 2 (rows, cols) writes a single
    // image, rank 3 (pages, rows, cols) a multi-page BigTIFF.
    template <typename T>
    void writeTiff(const std::string& path, BufferView<const T> image,
                   TiffCompression comp = TiffCompression::None);

    template <typename T>
    void writeTiff(const std::string& path, const Image<T>& image,
                   TiffCompression comp = TiffCompression::None);

    template <typename T>
    void writeTiffStack(const std::string& path, BufferView<const T> stack,
                        TiffCompression comp = TiffCompression::None);

    template <typename T>
    void writeTiffStack(const std::string& path, const ImageStack<T>& stack,
                        TiffCompression comp = TiffCompression::None);

    template <typename T>
    void writeTiff(const std::string& path, const Buffer<T>& buffer,
                   TiffCompression comp = TiffCompression::None) {
        if (buffer.rank() == 2) writeTiff<T>(path, buffer.view(), comp);
        else writeTiffStack<T>(path, buffer.view().asStack(), comp);
    }

} // namespace sirius

#endif // SIRIUS_TIFF_IO_HPP
