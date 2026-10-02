#include "core/secure_wipe.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__GLIBC__) || defined(__OpenBSD__) || defined(__FreeBSD__)
#include <string.h>
#define SIRIUS_HAVE_EXPLICIT_BZERO 1
#endif

namespace sirius::app {

    void secureWipe(void* p, std::size_t n) noexcept {
        if (!p || n == 0) return;
#ifdef _WIN32
        SecureZeroMemory(p, n);
#elif defined(SIRIUS_HAVE_EXPLICIT_BZERO)
        explicit_bzero(p, n);
#else
        volatile unsigned char* v = static_cast<volatile unsigned char*>(p);
        while (n--) *v++ = 0;
#endif
    }

} // namespace sirius::app
