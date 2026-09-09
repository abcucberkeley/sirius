#ifndef SIRIUS_CUDA_CHECK_HPP
#define SIRIUS_CUDA_CHECK_HPP

// Internal helpers for translation units that talk to the CUDA runtime.
// Only included from sources compiled when SIRIUS_HAS_CUDA is defined.

#include <cuda_runtime.h>

#include <string>

#include "sirius/device.hpp"

namespace sirius::cuda {

    // Records that this process has run something on device `index`
    // (device.cpp). Stream::null().synchronize() waits only on such devices.
    void markDeviceUsed(int index) noexcept;

    inline void check(cudaError_t err, const char* what) {
        if (err != cudaSuccess) {
            // Clear the sticky error so later calls see a clean state.
            (void)cudaGetLastError();
            throw CudaError(std::string(what) + ": " + cudaGetErrorName(err) + " (" +
                            cudaGetErrorString(err) + ")");
        }
    }

    // The same for destructors, which must not throw: during process
    // teardown (a static, a Python module global at interpreter exit) the
    // runtime answers cudaErrorCudartUnloading, and a throwing destructor is
    // std::terminate. A failed switch is ignored; the release that follows
    // then fails quietly too.
    class DeviceGuardNoThrow {
    public:
        explicit DeviceGuardNoThrow(int index) noexcept {
            if (cudaGetDevice(&previous_) != cudaSuccess) {
                (void)cudaGetLastError();
                return;
            }
            if (previous_ != index && cudaSetDevice(index) == cudaSuccess) changed_ = true;
            else (void)cudaGetLastError();
        }
        ~DeviceGuardNoThrow() {
            if (changed_) (void)cudaSetDevice(previous_);
        }
        DeviceGuardNoThrow(const DeviceGuardNoThrow&) = delete;
        DeviceGuardNoThrow& operator=(const DeviceGuardNoThrow&) = delete;

    private:
        int previous_ = 0;
        bool changed_ = false;
    };

    // RAII cudaSetDevice; restores the previous current device on scope exit
    // so library calls never leak a device change into caller code.
    class DeviceGuard {
    public:
        explicit DeviceGuard(int index) {
            markDeviceUsed(index);
            check(cudaGetDevice(&previous_), "cudaGetDevice");
            if (previous_ != index) {
                check(cudaSetDevice(index), "cudaSetDevice");
                changed_ = true;
            }
        }
        ~DeviceGuard() {
            if (changed_) (void)cudaSetDevice(previous_);
        }
        DeviceGuard(const DeviceGuard&) = delete;
        DeviceGuard& operator=(const DeviceGuard&) = delete;

    private:
        int previous_ = 0;
        bool changed_ = false;
    };

    inline cudaStream_t handle(const Stream& s) noexcept {
        return static_cast<cudaStream_t>(s.handle());
    }

} // namespace sirius::cuda

#endif // SIRIUS_CUDA_CHECK_HPP
