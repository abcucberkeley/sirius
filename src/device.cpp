#include "sirius/device.hpp"

#include <algorithm>
#include <atomic>
#include <utility>

#ifdef SIRIUS_HAS_CUDA
#include "cuda_check.hpp"
#endif

namespace sirius {

#ifdef SIRIUS_HAS_CUDA
    namespace cuda {
        namespace {
            // Devices the library has run anything on (set by DeviceGuard, which
            // every CUDA code path goes through). The null stream's synchronize
            // waits on these and never creates a context on an untouched GPU.
            constexpr int kMaxTrackedDevices = 64;
            std::atomic<bool> g_deviceUsed[kMaxTrackedDevices];

            bool deviceUsed(int index) noexcept {
                return index >= 0 && index < kMaxTrackedDevices &&
                       g_deviceUsed[index].load(std::memory_order_relaxed);
            }
        } // namespace

        void markDeviceUsed(int index) noexcept {
            if (index >= 0 && index < kMaxTrackedDevices)
                g_deviceUsed[index].store(true, std::memory_order_relaxed);
        }
    } // namespace cuda
#endif

    std::string toString(Device d) {
        if (d.isCpu()) return "cpu";
        return "cuda:" + std::to_string(d.index);
    }

    bool builtWithCuda() noexcept {
#ifdef SIRIUS_HAS_CUDA
        return true;
#else
        return false;
#endif
    }

    bool builtWithNvTiff() noexcept {
#ifdef SIRIUS_HAS_NVTIFF
        return true;
#else
        return false;
#endif
    }

    int cudaDeviceCount() noexcept {
#ifdef SIRIUS_HAS_CUDA
        // Cached: the count never changes during a process, and the first
        // call is the expensive one (driver load / context probing).
        static const int count = [] {
            int n = 0;
            const cudaError_t err = cudaGetDeviceCount(&n);
            if (err != cudaSuccess) {
                (void)cudaGetLastError();   // e.g. no driver: clear and report 0 devices
                return 0;
            }
            return n;
        }();
        return count;
#else
        return 0;
#endif
    }

    bool cudaAvailable() noexcept { return cudaDeviceCount() > 0; }

    void requireDevice(Device d) {
        if (d.isCpu()) return;
        if (!builtWithCuda())
            throw std::runtime_error("SIRIUS was built without CUDA support (SIRIUS_ENABLE_CUDA=OFF); "
                                     "cannot use " +
                                     toString(d));
        if (d.index < 0 || d.index >= cudaDeviceCount())
            throw std::runtime_error("CUDA device " + toString(d) + " does not exist (" +
                                     std::to_string(cudaDeviceCount()) + " device(s) visible)");
    }

    DeviceProperties deviceProperties(Device d) {
        requireDevice(d);
        if (d.isCpu()) throw std::runtime_error("deviceProperties: not a CUDA device");
#ifdef SIRIUS_HAS_CUDA
        cudaDeviceProp p{};
        cuda::check(cudaGetDeviceProperties(&p, d.index), "cudaGetDeviceProperties");
        DeviceProperties out;
        out.name = p.name;
        out.computeMajor = p.major;
        out.computeMinor = p.minor;
        out.multiprocessorCount = p.multiProcessorCount;
        out.totalMemoryBytes = p.totalGlobalMem;
        return out;
#else
        return {};
#endif
    }

    void synchronizeDevice(Device d) {
        requireDevice(d);
#ifdef SIRIUS_HAS_CUDA
        if (d.isCuda()) {
            cuda::DeviceGuard g(d.index);
            cuda::check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        }
#endif
    }

    // --- Stream ---------------------------------------------------------

    Stream::Stream() noexcept = default;

    Stream::Stream(Device device) : device_(device) {
        requireDevice(device);
#ifdef SIRIUS_HAS_CUDA
        if (device.isCuda()) {
            cuda::DeviceGuard g(device.index);
            cudaStream_t s = nullptr;
            // Blocking (default-flag) stream: keeps the legacy NULL stream
            // ordered relative to us, which Buffer deallocation depends on.
            cuda::check(cudaStreamCreate(&s), "cudaStreamCreate");
            handle_ = s;
        }
#endif
    }

    // Like every other CUDA call in the library, stream and event teardown and
    // synchronization run with the owning device current: on a multi-GPU host
    // the call would otherwise go to (and create a context on) whichever
    // device the caller happens to have selected.

    Stream::~Stream() {
#ifdef SIRIUS_HAS_CUDA
        if (handle_) {
            cuda::DeviceGuardNoThrow g(device_.index);
            (void)cudaStreamDestroy(static_cast<cudaStream_t>(handle_));
        }
#endif
    }

    Stream::Stream(Stream&& other) noexcept
        : device_(other.device_), handle_(std::exchange(other.handle_, nullptr)) {}

    Stream& Stream::operator=(Stream&& other) noexcept {
        if (this != &other) {
#ifdef SIRIUS_HAS_CUDA
            if (handle_) {
                cuda::DeviceGuardNoThrow g(device_.index);
                (void)cudaStreamDestroy(static_cast<cudaStream_t>(handle_));
            }
#endif
            device_ = other.device_;
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    void Stream::synchronize() const {
#ifdef SIRIUS_HAS_CUDA
        if (handle_) {
            cuda::DeviceGuard g(device_.index);
            cuda::check(cudaStreamSynchronize(static_cast<cudaStream_t>(handle_)), "cudaStreamSynchronize");
        } else if (device_.isCuda()) {
            cuda::DeviceGuard g(device_.index);
            cuda::check(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize(null)");
        } else {
            // Stream::null() or a CPU stream: operations it was passed with
            // CUDA memory ran on the legacy default stream of that memory's
            // device. Wait on every device this process has touched (none in
            // a CPU-only run, so this stays free of CUDA calls there).
            const int n = std::min(cudaDeviceCount(), cuda::kMaxTrackedDevices);
            for (int i = 0; i < n; ++i) {
                if (!cuda::deviceUsed(i)) continue;
                cuda::DeviceGuard g(i);
                cuda::check(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize(null)");
            }
        }
#endif
    }

    const Stream& Stream::null() noexcept {
        static const Stream s;
        return s;
    }

    // --- Event ----------------------------------------------------------

    Event::~Event() {
#ifdef SIRIUS_HAS_CUDA
        if (handle_) {
            cuda::DeviceGuardNoThrow g(deviceIndex_);
            (void)cudaEventDestroy(static_cast<cudaEvent_t>(handle_));
        }
#endif
    }

    Event::Event(Event&& other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)), deviceIndex_(other.deviceIndex_) {}

    Event& Event::operator=(Event&& other) noexcept {
        if (this != &other) {
#ifdef SIRIUS_HAS_CUDA
            if (handle_) {
                cuda::DeviceGuardNoThrow g(deviceIndex_);
                (void)cudaEventDestroy(static_cast<cudaEvent_t>(handle_));
            }
#endif
            handle_ = std::exchange(other.handle_, nullptr);
            deviceIndex_ = other.deviceIndex_;
        }
        return *this;
    }

    void Event::record(const Stream& stream) {
        if (!stream.device().isCuda()) return;
#ifdef SIRIUS_HAS_CUDA
        cuda::DeviceGuard g(stream.device().index);
        if (!handle_) {
            cudaEvent_t e = nullptr;
            cuda::check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "cudaEventCreate");
            handle_ = e;
            deviceIndex_ = stream.device().index;
        }
        cuda::check(cudaEventRecord(static_cast<cudaEvent_t>(handle_), cuda::handle(stream)), "cudaEventRecord");
#endif
    }

    void Event::wait(const Stream& stream) const {
        if (!handle_ || !stream.device().isCuda()) return;
#ifdef SIRIUS_HAS_CUDA
        cuda::DeviceGuard g(stream.device().index);
        cuda::check(cudaStreamWaitEvent(cuda::handle(stream), static_cast<cudaEvent_t>(handle_), 0),
                    "cudaStreamWaitEvent");
#endif
    }

    void Event::synchronize() const {
        if (!handle_) return;
#ifdef SIRIUS_HAS_CUDA
        cuda::DeviceGuard g(deviceIndex_);
        cuda::check(cudaEventSynchronize(static_cast<cudaEvent_t>(handle_)), "cudaEventSynchronize");
#endif
    }

    bool Event::ready() const {
        if (!handle_) return true;
#ifdef SIRIUS_HAS_CUDA
        cuda::DeviceGuard g(deviceIndex_);
        const cudaError_t e = cudaEventQuery(static_cast<cudaEvent_t>(handle_));
        if (e == cudaSuccess) return true;
        if (e == cudaErrorNotReady) return false;
        cuda::check(e, "cudaEventQuery");
#endif
        return true;
    }

} // namespace sirius
