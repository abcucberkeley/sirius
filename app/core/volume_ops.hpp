#ifndef SIRIUS_APP_VOLUME_OPS_HPP
#define SIRIUS_APP_VOLUME_OPS_HPP

// GUI-free helpers behind the stack viewer: cropping, orthogonal re-slicing,
// centered magnitude spectra, expansion of the reconstruction's half-spectrum
// bands, an OTF rendered onto the data grid, and the frequency-space overlay
// geometry. Everything works on (depth, rows, cols) host volumes so it can be
// unit-tested without a display.

#include <array>
#include <complex>
#include <memory>
#include <vector>

#include <sirius/buffer.hpp>
#include <sirius/otf.hpp>
#include <sirius/sim_parameters.hpp>
#include <sirius/sim_reconstruction.hpp>

namespace sirius::app {

    // Sub-volume [z0, z1) x [y0, y1) x [x0, x1) of a (nz, ny, nx) host volume.
    // Throws std::out_of_range for an empty or out-of-bounds box.
    Buffer<double> cropVolume(BufferView<const double> v, Index z0, Index z1, Index y0, Index y1,
                              Index x0, Index x1);

    // Orthogonal re-slices of a (nz, ny, nx) host volume, in the layouts the
    // orthoviews display: XZ at row y writes (nz, nx) values (z down, x
    // across), YZ at column x writes (ny, nz) values (y down, z across).
    void sliceXZ(BufferView<const double> v, Index y, double* out);
    void sliceYZ(BufferView<const double> v, Index x, double* out);

    // Centered magnitude spectrum |F| of a (rows, cols) real plane, with the
    // zero frequency at (rows / 2, cols / 2) as numpy's fftshift puts it.
    // Keeps the FFT plan for the last plane size, so scrubbing through a
    // stack re-plans nothing.
    class PlaneSpectrum {
    public:
        PlaneSpectrum();
        ~PlaneSpectrum();
        PlaneSpectrum(PlaneSpectrum&&) noexcept;
        PlaneSpectrum& operator=(PlaneSpectrum&&) noexcept;

        void magnitude(const double* plane, Index rows, Index cols, double* out);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    // Which complex band to form from the reconstruction's stored cosine (re)
    // and sine (im) parts of an order: re + i im is the +order side band,
    // re - i im the -order one; order 0 has only re.
    enum class BandSide { Plus,
                          Minus,
                          ReOnly };

    // Centered |band| of plane z from the half spectra of SimDiagnostics:
    // `re` and `im` point at one band's (nz, ny, nx / 2 + 1) storage (`im`
    // null for BandSide::ReOnly). The kx < 0 half follows from conjugate
    // symmetry, as the reconstruction's own assembly step does. Writes
    // (ny, nx) values.
    void bandPlaneMagnitude(const std::complex<double>* re, const std::complex<double>* im,
                            Index nz, Index ny, Index nx, Index z, BandSide side, double* out);

    // Band `band` (0 .. nbands - 1) of direction `dir` as a centered (nz, ny, nx)
    // magnitude volume, from either capture of the diagnostics.
    Buffer<double> bandMagnitudeVolume(const SimDiagnostics& d, const Buffer<std::complex<double>>& bands,
                                       int dir, int band, BandSide side);

    // |OTF| of one order resampled onto the data grid (nz, ny, nx) with the
    // sampling SimReconstructor uses (dk = 1 / (n * pixel)), centered so the
    // zero frequency sits at (nz / 2, ny / 2, nx / 2).
    Buffer<double> otfDisplayVolume(const OTFRadiallyAveraged& otf, int order, const SIMParameters& p,
                                    Index nx, Index ny, Index nz);

    // --- frequency-space overlays ----------------------------------------

    // Pattern vectors the reconstruction starts its search from, derived from
    // the parameters exactly as SimReconstructor does (line spacing, start
    // angle / explicit angles, the 3D order-1 convention). One (kx, ky) [1/um]
    // per direction.
    std::vector<std::array<double, 2>> predictedK0(const SIMParameters& p, Index nz);

    // Geometry of a centered (rows, cols) spectrum whose pixel steps are
    // dkx / dky [1/um]: converts frequencies to display pixels.
    struct SpectrumGeometry {
        Index rows = 0, cols = 0;
        double dkx = 0.0, dky = 0.0;

        // Pixel of frequency (kx, ky): the zero frequency is at (cols / 2, rows / 2).
        std::array<double, 2> pixelOf(double kx, double ky) const noexcept {
            return {static_cast<double>(cols / 2) + kx / dkx, static_cast<double>(rows / 2) + ky / dky};
        }
        // Radius in pixels of a lateral frequency r (x and y steps may differ).
        std::array<double, 2> radiusPixels(double r) const noexcept { return {r / dkx, r / dky}; }
    };

    // Lateral OTF support radius 2 NA / lambda_emission [1/um].
    double otfSupportRadius(const SIMParameters& p) noexcept;

} // namespace sirius::app

#endif // SIRIUS_APP_VOLUME_OPS_HPP
