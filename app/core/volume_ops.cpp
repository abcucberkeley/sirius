#include "core/volume_ops.hpp"

#include <sirius/constants.hpp>
#include <sirius/fft.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

namespace sirius::app {

    namespace {
        using Cplx = std::complex<double>;

        void requireVolume(BufferView<const double> v, const char* what) {
            if (v.rank() != 3 || !v.device().isCpu())
                throw std::invalid_argument(std::string(what) + ": expects a (depth, rows, cols) host volume, got " +
                                            v.shape().toString() + " on " + toString(v.device()));
        }

        // out[(r + rows/2) % rows][(c + cols/2) % cols] = |in[r][c]|
        void centeredMagnitude(const Cplx* in, Index rows, Index cols, double* out) {
            for (Index r = 0; r < rows; ++r) {
                const Index ro = (r + rows / 2) % rows;
                for (Index c = 0; c < cols; ++c)
                    out[ro * cols + (c + cols / 2) % cols] = std::abs(in[r * cols + c]);
            }
        }
    } // namespace

    Buffer<double> cropVolume(BufferView<const double> v, Index z0, Index z1, Index y0, Index y1,
                              Index x0, Index x1) {
        requireVolume(v, "cropVolume");
        if (z0 < 0 || y0 < 0 || x0 < 0 || z1 > v.dim(0) || y1 > v.dim(1) || x1 > v.dim(2) ||
            z1 <= z0 || y1 <= y0 || x1 <= x0)
            throw std::out_of_range("cropVolume: box [" + std::to_string(z0) + "," + std::to_string(z1) + ") x [" +
                                    std::to_string(y0) + "," + std::to_string(y1) + ") x [" + std::to_string(x0) +
                                    "," + std::to_string(x1) + ") is empty or exceeds " + v.shape().toString());
        Buffer<double> out(Shape{z1 - z0, y1 - y0, x1 - x0});
        const Index ny = v.dim(1), nx = v.dim(2), w = x1 - x0;
        for (Index z = z0; z < z1; ++z)
            for (Index y = y0; y < y1; ++y)
                std::copy_n(v.data() + (z * ny + y) * nx + x0, w,
                            out.data() + ((z - z0) * (y1 - y0) + (y - y0)) * w);
        return out;
    }

    void sliceXZ(BufferView<const double> v, Index y, double* out) {
        requireVolume(v, "sliceXZ");
        if (y < 0 || y >= v.dim(1)) throw std::out_of_range("sliceXZ: row " + std::to_string(y) + " out of range");
        const Index ny = v.dim(1), nx = v.dim(2);
        for (Index z = 0; z < v.dim(0); ++z)
            std::copy_n(v.data() + (z * ny + y) * nx, nx, out + z * nx);
    }

    void sliceYZ(BufferView<const double> v, Index x, double* out) {
        requireVolume(v, "sliceYZ");
        if (x < 0 || x >= v.dim(2)) throw std::out_of_range("sliceYZ: column " + std::to_string(x) + " out of range");
        const Index nz = v.dim(0), ny = v.dim(1), nx = v.dim(2);
        for (Index z = 0; z < nz; ++z)
            for (Index y = 0; y < ny; ++y)
                out[y * nz + z] = v.data()[(z * ny + y) * nx + x];
    }

    // --- PlaneSpectrum ------------------------------------------------------

    struct PlaneSpectrum::Impl {
        std::optional<FFT> fft;
        Index rows = 0, cols = 0;
        std::vector<Cplx> in, out;
    };

    PlaneSpectrum::PlaneSpectrum() : impl_(std::make_unique<Impl>()) {}
    PlaneSpectrum::~PlaneSpectrum() = default;
    PlaneSpectrum::PlaneSpectrum(PlaneSpectrum&&) noexcept = default;
    PlaneSpectrum& PlaneSpectrum::operator=(PlaneSpectrum&&) noexcept = default;

    void PlaneSpectrum::magnitude(const double* plane, Index rows, Index cols, double* out) {
        if (rows <= 0 || cols <= 0) throw std::invalid_argument("PlaneSpectrum: empty plane");
        Impl& s = *impl_;
        if (!s.fft || s.rows != rows || s.cols != cols) {
            s.fft.emplace(std::vector<int>{static_cast<int>(rows), static_cast<int>(cols)}, 1, PlanRigor::Estimate);
            s.rows = rows;
            s.cols = cols;
            s.in.resize(static_cast<std::size_t>(rows * cols));
            s.out.resize(s.in.size());
        }
        const Index n = rows * cols;
        for (Index i = 0; i < n; ++i) s.in[static_cast<std::size_t>(i)] = Cplx(plane[i], 0.0);
        s.fft->fft(s.in.data(), s.out.data());
        centeredMagnitude(s.out.data(), rows, cols, out);
    }

    // --- bands --------------------------------------------------------------

    void bandPlaneMagnitude(const Cplx* re, const Cplx* im, Index nz, Index ny, Index nx, Index z,
                            BandSide side, double* out) {
        if (nz <= 0 || ny <= 0 || nx <= 0 || z < 0 || z >= nz)
            throw std::out_of_range("bandPlaneMagnitude: bad plane index or shape");
        if (side != BandSide::ReOnly && !im)
            throw std::invalid_argument("bandPlaneMagnitude: a side band needs the sine part");
        const Index nxh = nx / 2 + 1;
        auto at = [&](const Cplx* h, Index kz, Index ky, Index kx) { return h[(kz * ny + ky) * nxh + kx]; };
        for (Index iy = 0; iy < ny; ++iy)
            for (Index ix = 0; ix < nx; ++ix) {
                Cplx r, i(0.0, 0.0);
                if (ix < nxh) {
                    r = at(re, z, iy, ix);
                    if (im) i = at(im, z, iy, ix);
                } else {   // negative kx: conjugate of the mirrored sample (-kz, -ky, -kx)
                    const Index kx = nx - ix, ky = (ny - iy) % ny, kz = (nz - z) % nz;
                    r = std::conj(at(re, kz, ky, kx));
                    if (im) i = std::conj(at(im, kz, ky, kx));
                }
                Cplx v = r;
                if (side == BandSide::Plus) v = r + Cplx(0.0, 1.0) * i;
                else if (side == BandSide::Minus) v = r - Cplx(0.0, 1.0) * i;
                out[((iy + ny / 2) % ny) * nx + (ix + nx / 2) % nx] = std::abs(v);
            }
    }

    Buffer<double> bandMagnitudeVolume(const SimDiagnostics& d, const Buffer<Cplx>& bands, int dir, int band,
                                       BandSide side) {
        if (!d.captured || bands.empty()) throw std::invalid_argument("bandMagnitudeVolume: nothing captured");
        if (dir < 0 || dir >= d.ndirs || band < 0 || band >= d.nbands)
            throw std::out_of_range("bandMagnitudeVolume: direction " + std::to_string(dir) + " / band " +
                                    std::to_string(band) + " out of range");
        const Index planeElems = d.ny * (d.nx / 2 + 1);
        const Index bandElems = d.nz * planeElems;
        auto bandPtr = [&](int b) { return bands.data() + (static_cast<Index>(dir) * d.nbands + b) * bandElems; };
        const int order = band == 0 ? 0 : (band + 1) / 2;
        const Cplx* re = bandPtr(order == 0 ? 0 : 2 * order - 1);
        const Cplx* im = order == 0 ? nullptr : bandPtr(2 * order);
        if (order == 0) side = BandSide::ReOnly;

        Buffer<double> out(Shape{d.nz, d.ny, d.nx});
        for (Index z = 0; z < d.nz; ++z)
            bandPlaneMagnitude(re, im, d.nz, d.ny, d.nx, z, side,
                               out.data() + ((z + d.nz / 2) % d.nz) * d.ny * d.nx);   // center kz too
        return out;
    }

    // --- OTF ----------------------------------------------------------------

    Buffer<double> otfDisplayVolume(const OTFRadiallyAveraged& otf, int order, const SIMParameters& p,
                                    Index nx, Index ny, Index nz) {
        if (nx < 2 || ny < 2 || nz < 1) throw std::invalid_argument("otfDisplayVolume: grid too small");
        const double dkx = 1.0 / (static_cast<double>(nx) * p.dx);
        const double dky = 1.0 / (static_cast<double>(ny) * p.dy);
        const double dkz = p.dz > 0 ? 1.0 / (static_cast<double>(nz) * p.dz) : otf.dkzotf();
        const double kzscale = nz > 1 ? dkz / otf.dkzotf() : 0.0;
        const auto grid = resampleOTF(otf.plane(order), static_cast<int>(nx), static_cast<int>(ny),
                                      static_cast<int>(nz), dkx, dky, otf.dkrotf(), kzscale);
        Buffer<double> out(Shape{nz, ny, nx});
        for (Index z = 0; z < nz; ++z)
            centeredMagnitude(grid.data() + z * ny * nx, ny, nx,
                              out.data() + ((z + nz / 2) % nz) * ny * nx);
        return out;
    }

    // --- overlays -----------------------------------------------------------

    std::vector<std::array<double, 2>> predictedK0(const SIMParameters& p, Index nz) {
        double mag = 1.0 / p.linespacing_um;
        // A 3D stack's line spacing is that of the highest order, nphases / 2
        // times order 1's. The overlay draws parameters the form has not
        // validated yet, so a single phase must not divide by zero.
        const int highest = p.nphases / 2;
        if (nz > 1 && highest > 0) mag /= static_cast<double>(highest);
        std::vector<std::array<double, 2>> k0(static_cast<std::size_t>(std::max(p.ndirs, 0)));
        for (int d = 0; d < p.ndirs; ++d) {
            const double angle = (p.k0_angles && p.k0_angles->size() >= static_cast<std::size_t>(p.ndirs))
                                     ? (*p.k0_angles)[static_cast<std::size_t>(d)]
                                     : p.k0_start_angle + d * kPi / p.ndirs;
            k0[static_cast<std::size_t>(d)] = {mag * std::cos(angle), mag * std::sin(angle)};
        }
        return k0;
    }

    double otfSupportRadius(const SIMParameters& p) noexcept {
        return 2.0 * p.na / (p.wavelength_nm * 1e-3);
    }

} // namespace sirius::app
