#ifndef SIRIUS_DEVICE_HPP
#define SIRIUS_DEVICE_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "sirius/errors.hpp"

// Execution/memory placement primitives shared by every SIRIUS algorithm.
//
// The public headers are free of CUDA types and of build-configuration macros:
// a CPU-only build of libsirius and a CUDA build expose the same API, and code
// asking for Device::cuda() on a CPU-only build fails at run time with a clear
// error instead of failing to compile. Query builtWithCuda()/cudaAvailable()
// to pick a device.

namespace sirius {

    enum class DeviceType : std::uint8_t { Cpu = 0, Cuda = 1 };

    // A place where memory lives and kernels run. `index` is the CUDA device
    // ordinal (ignored for the CPU). With MPI, each rank typically binds to
    // Device::cuda(localRank % cudaDeviceCount()).
    struct Device {
        DeviceType type = DeviceType::Cpu;
        int index = 0;

        static constexpr Device cpu() noexcept { return Device{DeviceType::Cpu, 0}; }
        static constexpr Device cuda(int index = 0) noexcept { return Device{DeviceType::Cuda, index}; }

        constexpr bool isCpu() const noexcept { return type == DeviceType::Cpu; }
        constexpr bool isCuda() const noexcept { return type == DeviceType::Cuda; }

        friend constexpr bool operator==(Device a, Device b) noexcept {
            return a.type == b.type && (a.isCpu() || a.index == b.index);
        }
        friend constexpr bool operator!=(Device a, Device b) noexcept { return !(a == b); }
    };

    std::string toString(Device d);

    // Thrown for any failing CUDA runtime / library call. The message carries
    // the CUDA error string and the call that failed.
    class CudaError : public SiriusError {
    public:
        using SiriusError::SiriusError;
    };

    // --- capability queries ---------------------------------------------
    bool builtWithCuda() noexcept;      // compiled with SIRIUS_ENABLE_CUDA
    bool builtWithNvTiff() noexcept;    // compiled with nvTIFF support
    int  cudaDeviceCount() noexcept;    // 0 without CUDA, without a driver, or without GPUs
    bool cudaAvailable() noexcept;      // cudaDeviceCount() > 0

    // --- what a front may ask for ----------------------------------------
    //
    // One rule for every front. The window, the command line and the Python
    // bindings each used to decide for themselves what "auto" means and what
    // to do with a request they cannot honour, and they did not agree: the
    // window and sirius-cli asked cudaAvailable() while the Python mirror
    // asked `torch.cuda.is_available()` -- so in the worker image, which has
    // numpy and the compiled extension but no torch, every Python
    // reconstruction ran on the CPU with nothing saying so, and an explicit
    // `device="cuda"` was quietly downgraded instead of refused
    // (docs/findings.md 9k.50 finding 3, and the Python front's finding A).
    // torch's view of the GPUs is not SIRIUS's: SIRIUS's own kernels are what
    // a SIM reconstruction runs on.

    // "cpu", "cuda", "cuda:N" -- and "gpu" for "cuda:0", which the
    // application's backend names already accept (backendFromString) -- as a
    // Device; nullopt for anything else. Case-insensitive.
    std::optional<Device> deviceFromString(const std::string& spec);

    // "" when a request for `spec` can be honoured by THIS build on THIS
    // computer; otherwise the one sentence every front says about it. Three
    // conditions, one sentence each, in one place: the spec is not a device,
    // the build has no CUDA, the computer has no such CUDA device.
    // requireDevice throws exactly this text, so there is one wording and not
    // one per front. The Device overload is for a front that has already
    // parsed one (Workbench's backend plus its GPU index); the string
    // overload also reports a spec that is not a device at all.
    std::string deviceRequestProblem(Device d);
    std::string deviceRequestProblem(const std::string& spec);

    // The rule for a request that names no device: the GPU when there is one,
    // else the CPU. Workbench's constructor and sirius-cli's --backend auto
    // have always done this; `resolve_device` in the Python mirror now does
    // the same instead of asking torch.
    Device defaultDevice() noexcept;

    struct DeviceProperties {
        std::string name;
        int computeMajor = 0;
        int computeMinor = 0;
        int multiprocessorCount = 0;
        std::size_t totalMemoryBytes = 0;
    };
    DeviceProperties deviceProperties(Device d);   // throws for a CPU or an invalid ordinal

    // Throws std::runtime_error/CudaError unless `d` can be used right now.
    void requireDevice(Device d);

    // --- streams ---------------------------------------------------------
    // Ordering handle for asynchronous work. On the CPU every operation is
    // synchronous and a Stream is a no-op. On CUDA it wraps a cudaStream_t
    // created with default (blocking) flags, so the legacy NULL stream --
    // which Stream::null() denotes -- stays correctly ordered with respect to
    // it; buffer deallocation relies on that.
    //
    // Every SIRIUS operation that takes a Stream is enqueued on it and returns
    // without waiting, unless documented otherwise. Call synchronize() (or
    // record an Event) before touching results from the host.
    class Stream {
    public:
        Stream() noexcept;                       // CPU / legacy default stream
        explicit Stream(Device device);          // creates a stream on a CUDA device
        ~Stream();

        Stream(Stream&& other) noexcept;
        Stream& operator=(Stream&& other) noexcept;
        Stream(const Stream&) = delete;
        Stream& operator=(const Stream&) = delete;

        Device device() const noexcept { return device_; }
        // Underlying cudaStream_t (nullptr for the CPU or the legacy default stream).
        void* handle() const noexcept { return handle_; }
        bool isNull() const noexcept { return handle_ == nullptr; }

        // Block the host until everything enqueued on this stream has run.
        // For a CPU stream -- Stream::null() included -- work handed to a
        // CUDA device went to that device's legacy default stream, so this
        // waits on the legacy stream of every device the process has used.
        void synchronize() const;

        // Shared "no particular stream" object: CPU device, null handle. On a
        // CUDA device it denotes the legacy default stream.
        static const Stream& null() noexcept;

    private:
        Device device_ = Device::cpu();
        void* handle_ = nullptr;
    };

    // --- events ----------------------------------------------------------
    // Cross-stream dependency marker (cudaEvent_t with timing disabled). A
    // default-constructed Event is lazily created on first record().
    class Event {
    public:
        Event() noexcept = default;
        ~Event();
        Event(Event&& other) noexcept;
        Event& operator=(Event&& other) noexcept;
        Event(const Event&) = delete;
        Event& operator=(const Event&) = delete;

        void record(const Stream& stream);            // no-op on CPU streams
        void wait(const Stream& stream) const;        // make `stream` wait for this event
        void synchronize() const;                     // block the host until recorded work is done
        bool ready() const;                           // completed (true when never recorded)

    private:
        void* handle_ = nullptr;
        int deviceIndex_ = -1;
    };

    // Synchronize an entire device (all streams). Rarely needed; prefer streams.
    void synchronizeDevice(Device d);

} // namespace sirius

#endif // SIRIUS_DEVICE_HPP
