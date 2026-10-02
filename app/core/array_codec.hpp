#ifndef SIRIUS_APP_ARRAY_CODEC_HPP
#define SIRIUS_APP_ARRAY_CODEC_HPP

// Arrays on the wire, as dataset_read / dataset_view replies carry them: the
// serving half of what sirius_worker/datasets.py does (encode) and
// remote_source.cpp reads (decodeWorkerArray). The client names the encodings
// it accepts; the best one this build has is used when it makes the array
// smaller, with the bytes of a multi-byte sample regrouped by significance
// first ("shuffle"), which is what makes 16-bit microscopy compress. The reply
// describes it:
//
//   {"encoding": "zlib" | "zstd" | "raw", "shuffle": bool, "dtype", "shape", "raw_bytes"}
//
// and its one tensor is the array itself (raw) or the compressed bytes (uint8).
//
// Encodings: zstd (level 3, as the worker's zstandard) when the build has its
// zstd library (SIRIUS_APP_HAVE_ZSTD, cmake/Dependencies.cmake), and zlib
// (level 1) always; the application decodes both (decompress below).

#include <cstddef>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/rpc.hpp"

namespace sirius::app::codec {

    // What this build can send, best first.
    std::vector<std::string> availableEncodings();

    // Arrays smaller than this go raw (datasets.py: a.nbytes < 4096).
    inline constexpr std::size_t kMinCompressBytes = 4096;

    struct Encoded {
        nlohmann::json description;
        rpc::Tensor tensor;
    };

    // `bytes` hold the elements of `shape` in C order as `dtype` ("uint16",
    // "float32", ...: rpc's names), little endian. The first of
    // availableEncodings() that `accept` names is used, when the result is
    // smaller than the array; otherwise the array goes raw. `name` is the
    // tensor's.
    Encoded encodeArray(const std::string& dtype, const std::vector<Index>& shape, std::vector<std::byte> bytes,
                        const std::vector<std::string>& accept, const std::string& name = "data");

    // The bytes of one element of `dtype`; throws std::invalid_argument for a
    // name the protocol does not have.
    std::size_t dtypeSize(const std::string& dtype);

    // `in` decompressed ("zlib" or "zstd"), which must come to exactly
    // `expected` bytes: a stream that would inflate past it (a decompression
    // bomb) is refused, not followed. Throws ProtocolError.
    std::vector<std::byte> decompress(const std::string& encoding, const std::vector<std::byte>& in, std::size_t expected);

} // namespace sirius::app::codec

#endif // SIRIUS_APP_ARRAY_CODEC_HPP
