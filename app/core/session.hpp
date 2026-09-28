#ifndef SIRIUS_APP_SESSION_HPP
#define SIRIUS_APP_SESSION_HPP

// Application model of one reconstruction session: the raw stack, the OTF,
// the parameters and the reconstructor built from them. GUI-free so it can be
// unit-tested and driven from a worker thread; the GUI owns one instance and
// serializes access to it (edit on the GUI thread, reconstruct on the worker,
// never both at once).

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <sirius/buffer.hpp>
#include <sirius/device.hpp>
#include <sirius/fft_common.hpp>
#include <sirius/otf.hpp>
#include <sirius/sim_parameters.hpp>
#include <sirius/sim_reconstruction.hpp>
#include <sirius/tiff_io.hpp>

namespace sirius::app {

    // --- parameter files -------------------------------------------------

    enum class ParameterFormat { Toml,
                                 Legacy };

    // .toml files are TOML; otherwise the first significant line decides: a
    // `[table]` header means TOML, anything else the flat cudasirecon
    // `key=value` format. Throws std::runtime_error if the file cannot be read.
    ParameterFormat detectParameterFormat(const std::string& path);

    // Load either format (see detectParameterFormat). The detected format is
    // reported through `format` when non-null.
    SIMParameters loadParametersAuto(const std::string& path, ParameterFormat* format = nullptr);

    // --- fit summary -----------------------------------------------------

    // One direction of a SimFit in the units a user reads: line spacing (um),
    // pattern angle (degrees) and modulation depths |amp| per order.
    struct FitRow {
        int direction = 0;
        double kx = 0.0, ky = 0.0;      // 1/um
        double spacingUm = 0.0;         // 1/|k0|
        double angleDeg = 0.0;          // atan2(ky, kx)
        std::vector<double> ampMagnitude;
    };
    std::vector<FitRow> summarizeFit(const SimFit& fit);

    // --- session ---------------------------------------------------------

    struct ReconResult {
        Buffer<double> volume;    // host copy, (z_zoom*nz, zoomfact*ny, zoomfact*nx)
        SimFit fit;
        SimDiagnostics diagnostics;   // captured only when the session asks for them
        Device device;
        double seconds = 0.0;     // wall time of SimReconstructor::reconstruct
        bool plansReused = false; // false when the reconstructor had to be rebuilt
        bool idealOtf = false;    // the OTF was computed from NA, not loaded from a file
    };

    class ReconSession {
    public:
        ReconSession();
        ~ReconSession();
        ReconSession(ReconSession&&) noexcept;
        ReconSession& operator=(ReconSession&&) noexcept;
        ReconSession(const ReconSession&) = delete;
        ReconSession& operator=(const ReconSession&) = delete;

        // Raw camera stack (sections, ny, nx), decoded to double on the host.
        void loadRaw(const std::string& path);
        void setRaw(Buffer<double> stack, std::string label = {});
        bool hasRaw() const noexcept;
        const Buffer<double>& raw() const noexcept;
        const std::string& rawPath() const noexcept;

        // Radially averaged OTF TIFF; validated against the parameters when a
        // reconstructor is built. An empty path (the default) means no
        // measured OTF: the theoretical OTF of an aberration-free objective
        // with the parameters' NA, immersion index and wavelength is used
        // instead (sirius::idealOTF), in 3D when the stack has several planes.
        void setOtfPath(std::string path);
        const std::string& otfPath() const noexcept;
        bool usesIdealOtf() const noexcept;

        // The OTF a reconstruction would use right now (loaded or ideal),
        // cached until the parameters, OTF path or stack dimensionality
        // change. Throws when the file cannot be read.
        std::shared_ptr<const OTFRadiallyAveraged> otf() const;

        // Capture intermediate spectra (SimDiagnostics) during reconstruct().
        // Costs memory and time, so it is off by default.
        void setCaptureDiagnostics(bool on) noexcept;
        bool captureDiagnostics() const noexcept;

        void setParameters(const SIMParameters& p);
        const SIMParameters& parameters() const noexcept;

        // Empty when a reconstruction can start, otherwise the reason it cannot
        // (missing inputs, section count not a multiple of ndirs*nphases, odd
        // nx/ny, ...). Cheap: no I/O.
        std::string validate() const;

        // Section count implied by the parameters for the loaded stack (nz), or
        // 0 when the stack does not match.
        Index inferredNz() const noexcept;

        // Run the reconstruction on `device`. The SimReconstructor, and on a
        // CUDA device the device copy of the raw stack, are kept between calls
        // and rebuilt only when the parameters, OTF or device changed, so
        // repeated runs pay for the FFT planning once. Throws on failure.
        //
        // `cancelled` is handed to SimReconstructor::setCancelCallback for
        // this call: it is polled at the reconstruction's stage boundaries
        // and a true return aborts by throwing (the message "cancelled",
        // which isCancellation() recognises). The session keeps no partial
        // result -- nothing is written until reconstruct() returns.
        ReconResult reconstruct(Device device, PlanRigor rigor = PlanRigor::Measure,
                                std::function<bool()> cancelled = {});

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_SESSION_HPP
