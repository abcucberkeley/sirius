#ifndef SIRIUS_TIFF_INTERNAL_HPP
#define SIRIUS_TIFF_INTERNAL_HPP

// Internal glue between the libtiff (CPU) and nvTIFF (GPU) TIFF backends.

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "sirius/tiff_io.hpp"

namespace sirius {

    namespace detail {
        struct NvTiffSession;   // defined in tiff_nvtiff.cpp
    }

    struct TiffFile::Impl {
        std::string path;
        TiffInfo info;
        // nvTIFF parser + decoders for this file, created on first GPU decode
        // and reused afterwards (parsing 10k IFDs per read would dominate).
        std::shared_ptr<detail::NvTiffSession> nv;
        std::mutex nvMutex;
    };

    namespace detail {

        // A validated decode request: `ifds` share the geometry described by
        // `geometry`, `region` is resolved (non-zero extents inside the image).
        struct DecodeJob {
            const std::vector<std::uint64_t>* ifds = nullptr;
            const TiffImageInfo* geometry = nullptr;
            Region region;
            PixelType dstType = PixelType::UInt8;
        };

        // libtiff: decode into dense host memory {ifds, region.height, region.width}
        // of dstType, converting pixels when the on-disk type differs.
        void decodeWithLibtiff(const std::string& path, const DecodeJob& job, void* dstHost);

        // libtiff handles opened for reading so far in this process (inspection
        // and decoding). For tests: a read opens the file once per thread that
        // decodes a page, not once per thread of the OpenMP team.
        std::size_t libtiffReadOpens() noexcept;

        // nvTIFF (only linked with SIRIUS_HAS_NVTIFF): decode into device memory
        // of the same layout. Returns false with `reason` set, and dst untouched,
        // when nvTIFF cannot decode this file; throws on hard errors.
        bool decodeWithNvTiff(TiffFile::Impl& impl, const DecodeJob& job, void* dstDevice, Device device,
                              const Stream& stream, std::string& reason);
        bool nvTiffSupports(TiffFile::Impl& impl, const DecodeJob& job, Device device, std::string& reason);

        // Type-erased elementwise conversion between pixel types (same device).
        void convertPixels(const void* src, PixelType srcType, void* dst, PixelType dstType, Index n,
                           Device device, const Stream& stream);

    } // namespace detail

} // namespace sirius

#endif // SIRIUS_TIFF_INTERNAL_HPP
