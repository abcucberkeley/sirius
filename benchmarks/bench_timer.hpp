#ifndef SIRIUS_BENCH_TIMER_HPP
#define SIRIUS_BENCH_TIMER_HPP

#include <chrono>
#include <cstdint>
#include <cstdio>

// Tiny shared timing utility for the standalone TIFF read benchmarks.
//
// Both bench executables emit a single machine-parseable line that
// bindings/benchmarks/bench_tiff.py consumes:
//
//     <name>\t<seconds>\t<bytes>
//
// where <seconds> is the minimum wall-clock time of a full stack read and
// <bytes> is the decoded stack size, so the Python driver can compute GB/s
// and cross-reader speedups without re-reading the file.

namespace bench {

    // Run fn once to warm caches (cold disk read -> OS page cache, allocator
    // first-touch), then return the minimum wall-clock seconds over `repeats`
    // timed runs. Minimum filters OS scheduling noise better than the mean.
    template <typename Fn>
    double time_min(Fn&& fn, int repeats) {
        fn(); // warm-up: the cold read; subsequent reads are page-cache warm
        double best = -1.0;
        for (int i = 0; i < repeats; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            fn();
            const auto t1 = std::chrono::steady_clock::now();
            const double s = std::chrono::duration<double>(t1 - t0).count();
            if (best < 0.0 || s < best) {
                best = s;
            }
        }
        return best;
    }

    inline void report(const char* name, double seconds, std::uint64_t bytes) {
        std::printf("%s\t%.9f\t%llu\n", name, seconds,
                    static_cast<unsigned long long>(bytes));
        std::fflush(stdout);
    }

} // namespace bench

#endif // SIRIUS_BENCH_TIMER_HPP
