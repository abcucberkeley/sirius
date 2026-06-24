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
    int  getFFTWThreadCount();

    // Allocate/free buffers with FFTW's alignment. Useful for Python-owned output
    // arrays so execute_safe() can avoid allocation+copy fallbacks.
    void* fftwAlignedMalloc(std::size_t bytes);
    void  fftwAlignedFree(void* p) noexcept;

} // namespace sirius

#endif // SIRIUS_FFT_COMMON_HPP
