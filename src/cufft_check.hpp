#ifndef SIRIUS_CUFFT_CHECK_HPP
#define SIRIUS_CUFFT_CHECK_HPP

// Internal: cuFFT status -> CudaError, for the two cuFFT backends
// (fft_cufft.cpp, real_fft_cufft.cpp).

#include <cufft.h>

#include <string>

#include "sirius/device.hpp"   // CudaError

namespace sirius::detail {

    inline const char* cufftErrorName(cufftResult r) {
        switch (r) {
            case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
            case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
            case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
            case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
            case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
            case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
            case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
            case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
            case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
            case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
            case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
            case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
            case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
            default: return "CUFFT_UNKNOWN";
        }
    }

    inline void cufftCheck(cufftResult r, const char* what) {
        if (r != CUFFT_SUCCESS) throw CudaError(std::string(what) + ": " + cufftErrorName(r));
    }

} // namespace sirius::detail

#endif // SIRIUS_CUFFT_CHECK_HPP
