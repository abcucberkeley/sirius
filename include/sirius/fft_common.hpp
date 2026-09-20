#ifndef SIRIUS_FFT_COMMON_HPP
#define SIRIUS_FFT_COMMON_HPP

#include <cstddef>

namespace sirius {

    // Planning rigor controls the time FFTW spends searching for an optimal plan.
    // Higher rigor = better runtime FFT performance, but longer one-time planning cost.
    // Use Estimate for exploratory work; Measure or Patient for production runs on
    // fixed-size transforms that execute many times.
    enum class PlanRigor {
        Estimate,   // No measurement. Fast planning, suboptimal execution.
        Measure,    // Measure a few strategies. Good balance (seconds of planning).
        Patient,    // Measure many strategies. Better plan, slower to create.
        Exhaustive, // Try everything. Rarely worth it over Patient.
    };

    // FFTW planner/execution threading. Process-wide and shared by all transforms.
    // The default is 1 thread to avoid oversubscription surprises; applications can
    // raise this before planning.
    void setFFTWThreadCount(int nthreads);
    int getFFTWThreadCount();

    // Smallest n' >= n that factors into 2, 3, 5 and 7 -- the radices FFTW and
    // cuFFT have hand-written codelets for. Padding a transform up to such a
    // size is normally far cheaper than running the next prime length. Here,
    // inline, rather than in the registration unit where it started: the
    // deconvolution pads its transforms with it too, and that was its only
    // reason to link registration. (std::ptrdiff_t is sirius::Index.)
    inline std::ptrdiff_t nextFastFFTSize(std::ptrdiff_t n) {
        if (n <= 1) return 1;
        constexpr std::ptrdiff_t kRadices[] = {2, 3, 5, 7};
        for (;; ++n) {
            std::ptrdiff_t m = n;
            for (std::ptrdiff_t f : kRadices)
                while (m % f == 0) m /= f;
            if (m == 1) return n;
        }
    }

    // Allocate/free buffers with FFTW's alignment. Useful for Python-owned output
    // arrays so execute_safe() can avoid allocation+copy fallbacks.
    void* fftwAlignedMalloc(std::size_t bytes);
    void fftwAlignedFree(void* p) noexcept;

} // namespace sirius

#endif // SIRIUS_FFT_COMMON_HPP
