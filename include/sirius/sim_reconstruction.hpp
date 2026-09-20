#ifndef SIRIUS_SIM_RECONSTRUCTION_HPP
#define SIRIUS_SIM_RECONSTRUCTION_HPP

#include <array>
#include <complex>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "sirius/buffer.hpp"
#include "sirius/device.hpp"
#include "sirius/fft_common.hpp"
#include "sirius/otf.hpp"
#include "sirius/sim_parameters.hpp"

namespace sirius {

    // Per-direction results of the pattern-vector search and modulation
    // amplitude fit of the last reconstruct() call.
    struct SimFit {
        std::vector<std::array<double, 2>> k0;                // (ndirs) fitted {kx, ky}, 1/um
        std::vector<std::vector<std::complex<double>>> amps;  // (ndirs, norders); amps[d][0] == 1
    };

    // Intermediate spectra of a reconstruct() call, captured only when
    // SimReconstructor::setCaptureDiagnostics(true). Both volumes are host
    // copies of the band storage: (ndirs * nbands * nz, ny, nx / 2 + 1)
    // half spectra of the real band volumes in r2c layout and FFT ordering
    // (band b of direction d starts at plane (d * nbands + b) * nz). Band 0
    // is order 0; bands 2o - 1 and 2o are the cosine and sine parts of order
    // o, so the +o side band is re + i * im and the -o side band re - i * im,
    // with the kx < 0 half following from conjugate symmetry.
    struct SimDiagnostics {
        bool captured = false;
        int ndirs = 0;
        int nbands = 0;                      // 2 * norders - 1
        Index nx = 0, ny = 0, nz = 0;        // data grid
        double dkx = 0.0, dky = 0.0, dkz = 0.0;   // its frequency steps [1/um]
        double rdistcutoff = 0.0;            // lateral OTF support radius the filter used [1/um]
        int zdistcutoff = 0;                 // axial support used by the overlaps [planes]
        Buffer<std::complex<double>> separated;   // after band separation and the band FFT
        Buffer<std::complex<double>> filtered;    // after the generalized Wiener filter
    };

    // 3-beam structured illumination reconstruction (Gustafsson et al. 2008),
    // matching the cudasirecon algorithm: preprocessing, band separation,
    // pattern-vector / modulation-amplitude fitting, generalized Wiener
    // filtering and real-space assembly.
    //
    // The same object API runs on the CPU (FFTW + OpenMP) or on a CUDA device
    // (cuFFT + kernels); pass Device::cuda(n) and device-resident input to
    // run on the GPU -- everything else is identical. FFT plans and work
    // buffers are created lazily for the input shape and reused across calls
    // (e.g. over a time series), so construct once and reconstruct many.
    class SimReconstructor {
    public:
        // The OTF must be radially averaged with at least norders orders
        // (loadOTF in sirius/otf_io.hpp, idealOTF in sirius/otf_ideal.hpp).
        // PlanRigor affects only the FFTW backend.
        SimReconstructor(SIMParameters params, OTFRadiallyAveraged otf,
                         Device device = Device::cpu(), PlanRigor rigor = PlanRigor::Measure);
        ~SimReconstructor();

        SimReconstructor(const SimReconstructor&) = delete;
        SimReconstructor& operator=(const SimReconstructor&) = delete;
        SimReconstructor(SimReconstructor&&) noexcept;
        SimReconstructor& operator=(SimReconstructor&&) noexcept;

        Device device() const noexcept;

        // raw: (ndirs*nphases*nz, ny, nx) camera frames on device(), in the
        // standard direction->z->phase section order (fast_si selects the
        // z->direction->phase order instead). nx and ny must be even.
        // Returns the (z_zoom*nz, zoomfact*ny, zoomfact*nx) super-resolution
        // volume on device(); all enqueued work has completed on return.
        Buffer<double> reconstruct(BufferView<const double> raw);

        // Convenience for Buffers and host Eigen tensors.
        template <typename Src>
        Buffer<double> reconstruct(const Src& src) {
            return reconstruct(BufferView<const double>(toConstView(src)));
        }

        // Cooperative cancellation. `cancelled` is polled at the stage
        // boundaries of reconstruct() -- once per direction during band
        // separation and the pattern fit, once per bracket-search trial, and
        // once per (direction, order) during filtering and assembly -- and a
        // true return aborts the call by throwing
        // std::runtime_error("cancelled"), the same contract as
        // TiffWriteOptions::cancelled. The predicate runs on the calling
        // thread between stages, never inside a kernel or an FFT, so it costs
        // nothing measurable and cannot change a single output bit: a run
        // that is never cancelled is bit-identical to one with no callback.
        //
        // Worst-case latency is one stage: the largest is the assembly grid's
        // (z_zoom*nz, zoomfact*ny, zoomfact*nx) inverse FFT, run once per
        // (direction, order). After a cancelled call the reconstructor stays
        // usable -- plans and buffers are intact and the next reconstruct()
        // overwrites everything -- but lastFit() and lastDiagnostics() then
        // describe a partial run and must not be read.
        //
        // An empty callback (the default) disables the checks entirely.
        void setCancelCallback(std::function<bool()> cancelled);

        // Fit diagnostics of the last reconstruct() call.
        const SimFit& lastFit() const noexcept;

        // Capture the intermediate spectra of reconstruct() (see
        // SimDiagnostics). Off by default: it keeps two host copies of every
        // band volume and, on CUDA, transfers them mid-pipeline.
        void setCaptureDiagnostics(bool on) noexcept;
        bool captureDiagnostics() const noexcept;
        // Diagnostics of the last reconstruct(); `captured` is false unless
        // capture was on for that call. takeDiagnostics() moves them out.
        const SimDiagnostics& lastDiagnostics() const noexcept;
        SimDiagnostics takeDiagnostics() noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    // loadOTF(filename, SIMParameters) is declared in sirius/otf_io.hpp, beside
    // the code that implements it. It was declared here, so the OTF code had
    // to include this header and the two units formed a cycle.

} // namespace sirius

#endif // SIRIUS_SIM_RECONSTRUCTION_HPP
