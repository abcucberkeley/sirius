// Standalone benchmark: time cpp-tiff's parallel reader on a BigTIFF.
//
// NOT built by SIRIUS's CMake. benchmarks/setup_cpptiff.sh clones cpp-tiff
// fresh, builds libcppTiff.so (which statically embeds its own libtiff), then
// compiles this file directly against the clone's headers:
//
//     g++ -O3 -fopenmp bench_tiff_cpptiff.cpp \
//         -I<clone>/src -L<build> -lcppTiff -Wl,-rpath,<build> \
//         -o bench_tiff_cpptiff
//
//     usage: bench_tiff_cpptiff <path> [repeats]
//
// Uses readTiffParallelWrapperNoXYFlip (flipXY=0) to match sirius, which does
// no XY transpose. Emits one "<name>\t<seconds>\t<bytes>" line.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "parallelreadtiff.h"
#include "helperfunctions.h"

#include "bench_timer.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path> [repeats]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int repeats = (argc > 2) ? std::atoi(argv[2]) : 3;

    // File geometry is fixed, so resolve bytes once, outside the timed region.
    // getImageSize returns a malloc'd [y, x, z]; getDataType returns bits.
    std::uint64_t* dims = getImageSize(path.c_str());
    const std::uint64_t bits = getDataType(path.c_str());
    const std::uint64_t bytes = dims[0] * dims[1] * dims[2] * (bits / 8);
    std::free(dims);

    const double s = bench::time_min([&] {
        void* buf = readTiffParallelWrapperNoXYFlip(path.c_str());
        if (buf == nullptr) {
            std::fprintf(stderr, "cpptiff read returned NULL for %s\n", path.c_str());
            std::exit(1);
        }
        std::free(buf);
    }, repeats);

    bench::report("cpptiff", s, bytes);
    return 0;
}
