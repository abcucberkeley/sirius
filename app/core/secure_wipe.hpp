#ifndef SIRIUS_APP_SECURE_WIPE_HPP
#define SIRIUS_APP_SECURE_WIPE_HPP

// Overwrites memory that held a secret (a password, a token) with zeros in a
// way the optimiser may not drop: SecureZeroMemory on Windows, explicit_bzero
// where libc has it, a volatile loop elsewhere. std::string's small-string
// buffer keeps its bytes after a move or a clear(), so a string is wiped over
// its whole capacity before it is cleared.

#include <cstddef>
#include <string>

namespace sirius::app {

    void secureWipe(void* p, std::size_t n) noexcept;

    // The whole buffer, up to its capacity (the small-string bytes
    // included), then empty.
    inline void secureWipe(std::string& s) noexcept {
        s.resize(s.capacity());   // within the capacity: no reallocation
        secureWipe(s.data(), s.size());
        s.clear();
    }

} // namespace sirius::app

#endif // SIRIUS_APP_SECURE_WIPE_HPP
