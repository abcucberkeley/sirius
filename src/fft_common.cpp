// What the complex and the real transform share and neither owns: FFTW's
// planner lock, its thread count, the rigor flags and the aligned allocator
// (fft_common.hpp, fftw_internal.hpp). On its own so that RealFFT -- all the
// registration and the deconvolution need -- does not link the complex FFT,
// or the buffer layer under it, to get at a mutex.

#include "sirius/fft_common.hpp"
#include "fftw_internal.hpp"

#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>

#include <fftw3.h>

namespace sirius {
    namespace {
        // FFTW's planner modifies global state — must be serialized across all instances
        std::mutex s_planner_mutex;
        int s_fftw_thread_count = 1;
        bool s_fftw_threads_initialized = false;
    } // namespace

    namespace detail {
        std::mutex& fftwPlannerMutex() {
            return s_planner_mutex;
        }

        // map plan rigor to fftw flags
        unsigned int toFFTWFlag(PlanRigor r) {
            switch (r) {
                case PlanRigor::Estimate: return FFTW_ESTIMATE;
                case PlanRigor::Measure: return FFTW_MEASURE;
                case PlanRigor::Patient: return FFTW_PATIENT;
                case PlanRigor::Exhaustive: return FFTW_EXHAUSTIVE;
            }
            throw std::invalid_argument("Unknown PlanRigor value");
        }

        int checkedProduct(const std::vector<int>& dims, const char* what) {
            long long total = 1;
            for (int d : dims) {
                if (d <= 0)
                    throw std::invalid_argument(std::string(what) + " dimensions must be positive");
                if (total > std::numeric_limits<int>::max() / d)
                    throw std::overflow_error(std::string(what) + " dimensions overflow int");
                total *= d;
            }
            return static_cast<int>(total);
        }

        int checkedMultiply(int a, int b, const char* what) {
            if (a < 0 || b < 0)
                throw std::invalid_argument(std::string(what) + " size must not be negative");
            if (b != 0 && a > std::numeric_limits<int>::max() / b)
                throw std::overflow_error(std::string(what) + " size overflows int");
            return a * b;
        }

        // Caller must hold fftwPlannerMutex().
        void ensureDoubleThreadsInitializedLocked() {
            if (!s_fftw_threads_initialized) {
                if (fftw_init_threads() == 0)
                    throw std::runtime_error("FFTW failed to initialize double-precision threading");
                s_fftw_threads_initialized = true;
            }
            fftw_plan_with_nthreads(s_fftw_thread_count);
        }

        void* checkedFftwMalloc(std::size_t bytes) {
            if (bytes == 0) return nullptr;
            void* p = fftw_malloc(bytes);
            if (!p) throw std::bad_alloc();
            return p;
        }
    } // namespace detail

    void setFFTWThreadCount(int nthreads) {
        if (nthreads < 1)
            throw std::invalid_argument("FFTW thread count must be >= 1");

        std::lock_guard<std::mutex> lock(s_planner_mutex);
        s_fftw_thread_count = nthreads;
        detail::ensureDoubleThreadsInitializedLocked();
    }

    int getFFTWThreadCount() {
        std::lock_guard<std::mutex> lock(s_planner_mutex);
        return s_fftw_thread_count;
    }

    void* fftwAlignedMalloc(std::size_t bytes) {
        return detail::checkedFftwMalloc(bytes);
    }

    void fftwAlignedFree(void* p) noexcept {
        fftw_free(p);
    }

} // namespace sirius
