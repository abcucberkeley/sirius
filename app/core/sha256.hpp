#ifndef SIRIUS_APP_SHA256_HPP
#define SIRIUS_APP_SHA256_HPP

// SHA-256 (FIPS 180-4) and HMAC-SHA256 (RFC 2104), for the worker
// handshake's challenge-response (core/rpc.hpp): small, dependency-free and
// tested against the published vectors (tests/test_app_rpc.cpp). Not for
// bulk data: it is a straightforward implementation, not a fast one.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace sirius::app::crypto {

    using Digest = std::array<std::uint8_t, 32>;

    class Sha256 {
    public:
        Sha256();
        void update(const void* data, std::size_t n);
        void update(std::string_view s) { update(s.data(), s.size()); }
        Digest finish();   // the object must not be used afterwards

    private:
        void block(const std::uint8_t* p);
        std::array<std::uint32_t, 8> h_{};
        std::array<std::uint8_t, 64> buf_{};
        std::size_t used_ = 0;
        std::uint64_t bits_ = 0;
    };

    Digest sha256(std::string_view data);
    Digest hmacSha256(std::string_view key, std::string_view message);

    // Lower-case hex of a digest (or any bytes).
    std::string toHex(const std::uint8_t* p, std::size_t n);
    inline std::string toHex(const Digest& d) { return toHex(d.data(), d.size()); }

    // Equality that takes as long whatever the first difference is.
    bool constantTimeEqual(std::string_view a, std::string_view b);

} // namespace sirius::app::crypto

#endif // SIRIUS_APP_SHA256_HPP
