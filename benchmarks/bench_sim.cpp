// End-to-end in-memory SIM reconstruction benchmark. I/O, input conversion,
// device upload, FFT planning, and first-use allocator costs are deliberately
// outside the timed region: production callers reuse SimReconstructor for a
// time series, and this measures the steady-state algorithm they pay per volume.
//
// usage: bench_sim <data-dir> [repeats] [cpu|cuda|cuda:N]

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

#include <sirius/buffer.hpp>
#include <sirius/legacy_config.hpp>
#include <sirius/otf_io.hpp>
#include <sirius/sim_reconstruction.hpp>
#include <sirius/tiff_io.hpp>

#include "bench_timer.hpp"

namespace {
    sirius::Device parseDevice(const std::string& text) {
        if (text == "cpu") return sirius::Device::cpu();
        if (text == "cuda") return sirius::Device::cuda();
        if (text.rfind("cuda:", 0) == 0)
            return sirius::Device::cuda(std::atoi(text.c_str() + 5));
        throw std::invalid_argument("device must be cpu, cuda, or cuda:N");
    }
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <data-dir> [repeats] [cpu|cuda|cuda:N]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const int repeats = argc > 2 ? std::max(1, std::atoi(argv[2])) : 5;
    const sirius::Device device = parseDevice(argc > 3 ? argv[3] : "cpu");

    try {
        sirius::requireDevice(device);
        auto params = sirius::fromLegacy(sirius::loadLegacyConfig(dir + "/config.txt"));
        auto otf = sirius::loadOTF(dir + "/otf.tif", params);
        auto rawHost = sirius::readTiffStack<double>(dir + "/raw.tif");
        auto raw = sirius::toDevice(rawHost, device);
        sirius::Stream::null().synchronize();

        sirius::SimReconstructor recon(params, std::move(otf), device,
                                       sirius::PlanRigor::Measure);
        std::uint64_t bytes = 0;
        const double seconds = bench::time_min([&] {
            auto output = recon.reconstruct(raw);
            bytes = static_cast<std::uint64_t>(output.bytes());
        },
                                               repeats);
        bench::report(device.isCuda() ? "sim-cuda" : "sim-cpu", seconds, bytes);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "SIM benchmark failed: %s\n", e.what());
        return 1;
    }
    return 0;
}
