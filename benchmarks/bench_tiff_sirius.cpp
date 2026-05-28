// Standalone benchmark: time sirius::readTiffStack<uint16_t> on a BigTIFF.
//
// Built inside SIRIUS's CMake (gated by SIRIUS_ENABLE_BENCHMARKS) and linked
// against the sirius static lib. Kept separate from the cpp-tiff bench so the
// two libtiff builds never share a CMake target. See benchmarks/CMakeLists.txt.
//
//     usage: bench_tiff_sirius <path> [repeats]
//
// Emits one "<name>\t<seconds>\t<bytes>" line for bench_tiff.py to merge.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include <sirius/tiff_io.hpp>

#include "bench_timer.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path> [repeats]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int repeats = (argc > 2) ? std::atoi(argv[2]) : 3;

    try {
        std::uint64_t bytes = 0;
        const double s = bench::time_min([&] {
            auto stack = sirius::readTiffStack<std::uint16_t>(path);
            bytes = static_cast<std::uint64_t>(stack.size()) * sizeof(std::uint16_t);
        }, repeats);

        bench::report("sirius", s, bytes);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sirius read failed: %s\n", e.what());
        return 1;
    }
    return 0;
}
