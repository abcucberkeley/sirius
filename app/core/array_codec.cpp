#include "core/array_codec.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

#include <zlib.h>

#ifdef SIRIUS_APP_HAVE_ZSTD
#include <zstd.h>
#endif

#include "core/errors.hpp"

namespace sirius::app::codec {

    using json = nlohmann::json;

    std::vector<std::string> availableEncodings() {
#ifdef SIRIUS_APP_HAVE_ZSTD
        return {"zstd", "zlib"};
#else
        return {"zlib"};
#endif
    }

    std::size_t dtypeSize(const std::string& dtype) {
        if (dtype == "uint8" || dtype == "int8" || dtype == "bool") return 1;
        if (dtype == "uint16" || dtype == "int16") return 2;
        if (dtype == "uint32" || dtype == "int32" || dtype == "float32") return 4;
        if (dtype == "uint64" || dtype == "int64" || dtype == "float64") return 8;
        throw std::invalid_argument("unsupported dtype '" + dtype + "'");
    }

    namespace {
        // zlib.compress(data, 1): one zlib stream (header and checksum), at
        // level 1, as the Python worker sends.
        bool deflateLevel1(const std::vector<std::byte>& in, std::vector<std::byte>& out) {
            z_stream zs{};
            if (deflateInit(&zs, 1) != Z_OK) return false;
            out.resize(static_cast<std::size_t>(deflateBound(&zs, static_cast<uLong>(std::min<std::size_t>(in.size(), std::numeric_limits<uLong>::max())))));
            if (out.size() < in.size() / 2) out.resize(in.size() + in.size() / 100 + 64);
            std::size_t inPos = 0, outPos = 0;
            constexpr std::size_t kChunk = std::size_t{1} << 30;   // zlib counts in uInt
            int rc = Z_OK;
            while (rc != Z_STREAM_END) {
                if (outPos == out.size()) out.resize(out.size() + out.size() / 2 + 64);
                const std::size_t inLeft = in.size() - inPos;
                zs.next_in = reinterpret_cast<Bytef*>(const_cast<std::byte*>(in.data() + inPos));
                zs.avail_in = static_cast<uInt>(std::min(inLeft, kChunk));
                zs.next_out = reinterpret_cast<Bytef*>(out.data() + outPos);
                zs.avail_out = static_cast<uInt>(std::min(out.size() - outPos, kChunk));
                const uInt availIn = zs.avail_in, availOut = zs.avail_out;
                rc = deflate(&zs, inLeft <= kChunk ? Z_FINISH : Z_NO_FLUSH);
                inPos += availIn - zs.avail_in;
                outPos += availOut - zs.avail_out;
                if (rc != Z_OK && rc != Z_STREAM_END && rc != Z_BUF_ERROR) {
                    deflateEnd(&zs);
                    return false;
                }
            }
            deflateEnd(&zs);
            out.resize(outPos);
            return true;
        }
#ifdef SIRIUS_APP_HAVE_ZSTD
        // zstandard.ZstdCompressor(level=3).compress(): one frame that states its size
        bool zstdLevel3(const std::vector<std::byte>& in, std::vector<std::byte>& out) {
            out.resize(ZSTD_compressBound(in.size()));
            const std::size_t n = ZSTD_compress(out.data(), out.size(), in.data(), in.size(), 3);
            if (ZSTD_isError(n)) return false;
            out.resize(n);
            return true;
        }
#endif

        std::vector<std::byte> inflateExactly(const std::vector<std::byte>& in, std::size_t expected) {
            std::vector<std::byte> out(expected);
            z_stream zs{};
            if (inflateInit(&zs) != Z_OK) throw ProtocolError("worker: zlib cannot start");
            std::size_t inPos = 0, outPos = 0;
            int rc = Z_OK;
            constexpr std::size_t kChunk = std::size_t{1} << 30;   // zlib counts in uInt
            while (rc != Z_STREAM_END) {
                const std::size_t inLeft = in.size() - inPos, outLeft = out.size() - outPos;
                zs.next_in = reinterpret_cast<Bytef*>(const_cast<std::byte*>(in.data() + inPos));
                zs.avail_in = static_cast<uInt>(std::min(inLeft, kChunk));
                zs.next_out = reinterpret_cast<Bytef*>(out.data() + outPos);
                zs.avail_out = static_cast<uInt>(std::min(outLeft, kChunk));
                const uInt availIn = zs.avail_in, availOut = zs.avail_out;
                rc = inflate(&zs, Z_NO_FLUSH);
                inPos += availIn - zs.avail_in;
                outPos += availOut - zs.avail_out;
                if (rc == Z_STREAM_END) break;
                if (rc != Z_OK || (availIn == zs.avail_in && availOut == zs.avail_out)) {
                    inflateEnd(&zs);
                    throw ProtocolError("worker: a compressed array does not decompress");
                }
            }
            inflateEnd(&zs);
            if (outPos != expected) throw ProtocolError("worker: a compressed array has the wrong size");
            return out;
        }
    } // namespace

    std::vector<std::byte> decompress(const std::string& encoding, const std::vector<std::byte>& in, std::size_t expected) {
        if (encoding == "zlib") return inflateExactly(in, expected);
#ifdef SIRIUS_APP_HAVE_ZSTD
        if (encoding == "zstd") {
            const unsigned long long stated = ZSTD_getFrameContentSize(in.data(), in.size());
            if (stated == ZSTD_CONTENTSIZE_ERROR) throw ProtocolError("worker: a zstd array is not a zstd frame");
            if (stated != ZSTD_CONTENTSIZE_UNKNOWN && stated != expected)
                throw ProtocolError("worker: a zstd array of " + std::to_string(stated) + " bytes does not match its shape (" + std::to_string(expected) + " bytes)");
            std::vector<std::byte> out(expected);
            const std::size_t n = ZSTD_decompress(out.data(), out.size(), in.data(), in.size());
            if (ZSTD_isError(n)) throw ProtocolError(std::string("worker: a zstd array does not decompress: ") + ZSTD_getErrorName(n));
            if (n != expected) throw ProtocolError("worker: a compressed array has the wrong size");
            return out;
        }
#endif
        throw ProtocolError("worker: an array encoded as '" + encoding + "', which this application does not read");
    }

    Encoded encodeArray(const std::string& dtype, const std::vector<Index>& shape, std::vector<std::byte> bytes,
                        const std::vector<std::string>& accept, const std::string& name) {
        const std::size_t item = dtypeSize(dtype);
        std::size_t n = 1;
        for (Index s : shape) n *= static_cast<std::size_t>(std::max<Index>(s, 0));
        if (n * item != bytes.size())
            throw std::invalid_argument("encodeArray: " + std::to_string(bytes.size()) + " bytes do not hold " + std::to_string(n) + " " + dtype);
        Encoded e;
        e.description = {{"dtype", dtype}, {"shape", shape}, {"raw_bytes", bytes.size()}, {"encoding", "raw"}, {"shuffle", false}};
        std::string wanted;
        for (const std::string& have : availableEncodings())
            if (std::find(accept.begin(), accept.end(), have) != accept.end()) {
                wanted = have;
                break;
            }
        const auto raw = [&] {
            e.tensor.name = name;
            e.tensor.dtype = dtype;
            e.tensor.shape = shape;
            e.tensor.bytes = std::move(bytes);
            return std::move(e);
        };
        if (wanted.empty() || bytes.size() < kMinCompressBytes) return raw();
        const bool shuffle = item > 1;
        std::vector<std::byte> grouped;
        const std::vector<std::byte>* src = &bytes;
        if (shuffle) {
            // byte b of element i goes to b * n + i: (n, item) transposed
            grouped.resize(bytes.size());
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t b = 0; b < item; ++b) grouped[b * n + i] = bytes[i * item + b];
            src = &grouped;
        }
        std::vector<std::byte> packed;
        bool ok = false;
        if (wanted == "zlib") ok = deflateLevel1(*src, packed);
#ifdef SIRIUS_APP_HAVE_ZSTD
        if (wanted == "zstd") ok = zstdLevel3(*src, packed);
#endif
        if (!ok || packed.size() >= bytes.size()) return raw();
        e.description["encoding"] = wanted;
        e.description["shuffle"] = shuffle;
        e.tensor.name = name;
        e.tensor.dtype = "uint8";
        e.tensor.shape = {static_cast<Index>(packed.size())};
        e.tensor.bytes = std::move(packed);
        return e;
    }

} // namespace sirius::app::codec
