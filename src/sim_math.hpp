#ifndef SIRIUS_SIM_MATH_HPP
#define SIRIUS_SIM_MATH_HPP

// Per-voxel arithmetic of the SIM reconstruction, shared by the CPU (OpenMP)
// and CUDA backends. Everything here is a pure function of scalars and raw
// pointers, compiled by both the host compiler and nvcc, so the two backends
// run identical arithmetic and can be compared bit-for-bit (up to FFT library
// rounding).
//
// The algorithm follows Gustafsson 3-beam SIM as implemented by cudasirecon;
// the numerical conventions (index wrapping, cutoff comparisons, damping
// factors) are ported from the reference implementation in
// proto/cusimfixed/python/pysirecon and must not be "cleaned up" -- several
// asymmetries (e.g. a strict '<' in one overlap kernel vs '<=' in the other)
// are needed to reproduce the reference output exactly.

#include <math.h>

#include "sirius/constants.hpp"

#if defined(__CUDACC__)
#define SIRIUS_HD __host__ __device__ inline
#else
#define SIRIUS_HD inline
#endif

namespace sirius::simdetail {

    using IndexT = long long;

    // POD complex, layout-compatible with std::complex<double> (which is
    // guaranteed to be {real, imag} doubles), usable in device code.
    struct Cd {
        double re;
        double im;
    };

    SIRIUS_HD Cd cd(double re, double im) { return Cd{re, im}; }
    SIRIUS_HD Cd cadd(Cd a, Cd b) { return Cd{a.re + b.re, a.im + b.im}; }
    SIRIUS_HD Cd csub(Cd a, Cd b) { return Cd{a.re - b.re, a.im - b.im}; }
    SIRIUS_HD Cd cmul(Cd a, Cd b) { return Cd{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }
    SIRIUS_HD Cd cscale(Cd a, double s) { return Cd{a.re * s, a.im * s}; }
    SIRIUS_HD Cd cconj(Cd a) { return Cd{a.re, -a.im}; }
    SIRIUS_HD double cabs2(Cd a) { return a.re * a.re + a.im * a.im; }
    SIRIUS_HD double cabs(Cd a) { return sqrt(cabs2(a)); }
    // a + i*b (combine the separated "re"/"im" band values into bandplus)
    SIRIUS_HD Cd cplusi(Cd a, Cd b) { return Cd{a.re - b.im, a.im + b.re}; }
    // a - i*b (bandminus)
    SIRIUS_HD Cd cminusi(Cd a, Cd b) { return Cd{a.re + b.im, a.im - b.re}; }
    // -i * a / 2 == a * (0, -0.5)
    SIRIUS_HD Cd cmulNegHalfI(Cd a) { return Cd{0.5 * a.im, -0.5 * a.re}; }

    // storage index of a signed FFT frequency: negative frequencies live in
    // the upper half of the axis
    SIRIUS_HD IndexT signedToStorage(IndexT s, IndexT n) { return s < 0 ? s + n : s; }
    // storage index of the mirrored (negated) frequency: (n - i) % n
    SIRIUS_HD IndexT mirrorIndex(IndexT i, IndexT n) { return i == 0 ? 0 : n - i; }

    SIRIUS_HD double clampd(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
    SIRIUS_HD IndexT clampi(IndexT v, IndexT lo, IndexT hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    // ---------------------------------------------------------------------
    // Radially averaged OTF: bilinear interpolation at an arbitrary lateral
    // frequency (kx, ky) [1/um] and *data* axial index kz (signed, in data-z
    // pixels; kzscale = dkz_data / dkz_otf maps it onto the OTF's kz axis).
    // Port of pysirecon otf.interpolate / dev_otfinterpolate: the kz axis is
    // in FFT order and wraps; the kr axis is clamped (callers gate on
    // rdistcutoff, so clamped samples are never actually used).
    // ---------------------------------------------------------------------
    struct OtfTable {
        const Cd* data;   // (norders, nkr, nzotf), nzotf contiguous
        IndexT nkr;
        IndexT nzotf;
        double dkrotf;
        double kzscale;
    };

    SIRIUS_HD Cd otfInterpolate(const OtfTable& t, int order, double kx, double ky, double kz) {
        const Cd* tbl = t.data + static_cast<IndexT>(order) * t.nkr * t.nzotf;

        const double krindex = sqrt(kx * kx + ky * ky) / t.dkrotf;
        double kzindex = kz * t.kzscale;
        if (kzindex < 0) kzindex += static_cast<double>(t.nzotf);

        const IndexT irf = static_cast<IndexT>(floor(krindex));
        const IndexT izf = static_cast<IndexT>(floor(kzindex));
        const double ar = krindex - static_cast<double>(irf);
        const double az = kzindex - static_cast<double>(izf);

        const IndexT iz1w = (izf == t.nzotf - 1) ? 0 : izf + 1;   // kz wraps at the top
        const IndexT ir0 = clampi(irf, 0, t.nkr - 1);
        const IndexT ir1 = clampi(irf + 1, 0, t.nkr - 1);
        const IndexT iz0 = clampi(izf, 0, t.nzotf - 1);
        const IndexT iz1 = clampi(iz1w, 0, t.nzotf - 1);

        const Cd v00 = tbl[ir0 * t.nzotf + iz0];
        const Cd v01 = tbl[ir0 * t.nzotf + iz1];
        const Cd v10 = tbl[ir1 * t.nzotf + iz0];
        const Cd v11 = tbl[ir1 * t.nzotf + iz1];

        return cadd(cscale(cadd(cscale(v00, 1.0 - az), cscale(v01, az)), 1.0 - ar),
                    cscale(cadd(cscale(v10, 1.0 - az), cscale(v11, az)), ar));
    }

    // Notch dampening values near a band center (port of dev_suppress).
    // x is the radial distance in units of the smaller lateral frequency step.
    SIRIUS_HD double suppressSingularity(double x) {
        const double x3 = x * x * x;
        const double x6 = x3 * x3;
        return 1.0 / (1.0 + 20000.0 / (x6 + 20.0));
    }

    // Smooth high-pass for the widefield order (port of dev_order0damping).
    SIRIUS_HD double order0Damping(double radius, double zindex, double rlimit, double zlimit) {
        const double rfrac = radius / rlimit;
        const double zfrac = fabs(zindex / zlimit);
        return rfrac * rfrac + zfrac * zfrac * zfrac;
    }

    // ---------------------------------------------------------------------
    // Band storage. Bands are half-complex 3D spectra (nz, ny, nxh = nx/2+1)
    // of the separated orders: band 0 is the widefield order; bands 2o-1 and
    // 2o are the cosine ("re") / sine ("im") parts of order o. All bands of
    // one direction are contiguous: bands + b * nz*ny*nxh.
    // ---------------------------------------------------------------------

    // Value of a stored band at a *full-grid* frequency column ix in [0, nx):
    // columns beyond nx/2 are fetched from the mirrored half via Hermitian
    // symmetry (the conj/iin/jin logic of the reference kernels).
    // zs is the signed z frequency; iy the storage row.
    SIRIUS_HD Cd gatherFullGrid(const Cd* band, IndexT nz, IndexT ny, IndexT nxh,
                                IndexT zs, IndexT iy, IndexT ix, IndexT nx) {
        const bool conjFlag = ix > nx / 2;
        const IndexT jin = conjFlag ? nx - ix : ix;
        const IndexT iin = conjFlag ? mirrorIndex(iy, ny) : iy;
        const IndexT zin = signedToStorage(conjFlag ? -zs : zs, nz);
        const Cd v = band[(zin * ny + iin) * nxh + jin];
        return conjFlag ? cconj(v) : v;
    }

    // ---------------------------------------------------------------------
    // makeoverlaps: geometry + per-voxel values (ports of makeOverlaps0Kernel
    // and makeOverlaps1Kernel). The overlap volumes are full complex grids
    // (nz, ny, nx); only |zs| <= zdistcutoff planes are written, the rest is 0.
    // ---------------------------------------------------------------------
    struct OverlapCtx {
        IndexT nx, ny, nz, nxh;
        double dkx, dky;
        double rdistcutoff;
        double otfcutoff;
        double order02factor;   // relaxes otfcutoff for the bright order-0 OTF
        int noKz0;              // block the kz==0 plane
        double kx, ky;          // (order2 - order1) * k0 [1/um]
        int order1, order2;
        OtfTable otf;
    };

    // Signed lateral frequency of a full-grid index.
    SIRIUS_HD double signedFreq(IndexT i, IndexT n, double dk) {
        return static_cast<double>(i > n / 2 ? i - n : i) * dk;
    }

    // overlap0: band order1 at its native location, whitened by order2's
    // shifted OTF. Returns 0 outside the mutual support.
    SIRIUS_HD Cd overlap0Value(const OverlapCtx& c, const Cd* bandRe, const Cd* bandIm,
                               IndexT zs, IndexT iy, IndexT ix) {
        const double x1f = signedFreq(ix, c.nx, c.dkx);
        const double y1f = signedFreq(iy, c.ny, c.dky);

        const double rdist1 = sqrt(x1f * x1f + y1f * y1f);
        const double x12 = x1f - c.kx, y12 = y1f - c.ky;
        const double x21 = x1f + c.kx, y21 = y1f + c.ky;
        const double rdist12 = sqrt(x12 * x12 + y12 * y12);
        const double rdist21 = sqrt(x21 * x21 + y21 * y21);

        const bool overlap12 = (rdist12 <= c.rdistcutoff) || (rdist21 <= c.rdistcutoff);
        if (!(rdist1 <= c.rdistcutoff && overlap12 && rdist12 <= c.rdistcutoff)) return cd(0, 0);
        if (c.noKz0 && zs == 0) return cd(0, 0);

        const Cd otf1 = otfInterpolate(c.otf, c.order1, x1f, y1f, static_cast<double>(zs));
        const Cd otf12 = otfInterpolate(c.otf, c.order2, x12, y12, static_cast<double>(zs));
        if (!(cabs(otf1) > c.otfcutoff && cabs(otf12) * c.order02factor > c.otfcutoff)) return cd(0, 0);

        double root = sqrt(cabs2(otf1) + cabs2(otf12));
        if (root == 0.0) root = 1.0;
        const Cd fact = cscale(otf12, 1.0 / root);

        const Cd vre = cmul(gatherFullGrid(bandRe, c.nz, c.ny, c.nxh, zs, iy, ix, c.nx), fact);
        if (c.order1 == 0) return vre;
        const Cd vim = cmul(gatherFullGrid(bandIm, c.nz, c.ny, c.nxh, zs, iy, ix, c.nx), fact);
        return cplusi(vre, vim);
    }

    // overlap1: band order2 at its native location, whitened by order1's
    // back-shifted OTF. Note the strict '<' on rdist1 (reference asymmetry).
    SIRIUS_HD Cd overlap1Value(const OverlapCtx& c, const Cd* bandRe, const Cd* bandIm,
                               IndexT zs, IndexT iy, IndexT ix) {
        const double x1f = signedFreq(ix, c.nx, c.dkx);
        const double y1f = signedFreq(iy, c.ny, c.dky);

        const double rdist1 = sqrt(x1f * x1f + y1f * y1f);
        const double x12 = x1f - c.kx, y12 = y1f - c.ky;
        const double x21 = x1f + c.kx, y21 = y1f + c.ky;
        const double rdist12 = sqrt(x12 * x12 + y12 * y12);
        const double rdist21 = sqrt(x21 * x21 + y21 * y21);

        const bool overlap12 = (rdist12 <= c.rdistcutoff) || (rdist21 <= c.rdistcutoff);
        if (!(rdist1 < c.rdistcutoff && overlap12 && rdist21 <= c.rdistcutoff)) return cd(0, 0);
        if (c.noKz0 && zs == 0) return cd(0, 0);

        const Cd otf2 = otfInterpolate(c.otf, c.order2, x1f, y1f, static_cast<double>(zs));
        const Cd otf21 = otfInterpolate(c.otf, c.order1, x21, y21, static_cast<double>(zs));
        if (!(cabs(otf2) * c.order02factor > c.otfcutoff && cabs(otf21) > c.otfcutoff)) return cd(0, 0);

        double root = sqrt(cabs2(otf2) + cabs2(otf21));
        if (root == 0.0) root = 1.0;
        const Cd fact = cscale(otf21, 1.0 / root);

        const Cd vre = cmul(gatherFullGrid(bandRe, c.nz, c.ny, c.nxh, zs, iy, ix, c.nx), fact);
        const Cd vim = cmul(gatherFullGrid(bandIm, c.nz, c.ny, c.nxh, zs, iy, ix, c.nx), fact);
        return cplusi(vre, vim);
    }

    // ---------------------------------------------------------------------
    // Generalized Wiener filter (port of filterbands_kernel1). filterScale
    // is one "thread" of the kernel: the filter value for `order` of
    // `direction` at the *band-local* signed frequency (x1, y1, z0) [pixels].
    // The caller multiplies by conjamp for the side bands and zeroes samples
    // outside the lateral support.
    // ---------------------------------------------------------------------
    struct FilterCtx {
        IndexT nx, ny, nz, nxh;
        int direction;
        int ndirs, norders;
        double dkx, dky;
        double rdistcutoff;
        double suppRadius;      // suppression_radius * min(dkx, dky) [1/um]
        double minDkr;          // min(dkx, dky)
        double apocutoff;       // lateral apodization support radius
        double zapocutoff;      // axial apodization support (data-z pixels)
        double wiener2;         // wiener constant squared
        int zd0;                // order-0 axial cutoff (order0Damping zlimit)
        int suppressSingularities;
        int dampenOrder0;
        int noKz0;
        int filterOverlaps;
        int apodizeOutput;      // 0 none, 1 cosine, 2 triangle
        const double* k0all;    // (ndirs, 2) pattern vectors [1/um]
        const double* ampmag2;  // (ndirs, norders) |modamp|^2
        OtfTable otf;
    };

    SIRIUS_HD double bandWeightDamping(const FilterCtx& c, int order, double rdist, double z0) {
        double damp = 1.0;
        if (c.suppressSingularities && order != 0) {
            if (rdist <= c.suppRadius) damp = suppressSingularity(rdist / c.minDkr);
        } else if (!c.dampenOrder0 && c.suppressSingularities && order == 0) {
            if (rdist <= c.suppRadius) damp = suppressSingularity(rdist / c.minDkr);
        } else if (c.dampenOrder0 && order == 0) {
            damp = order0Damping(rdist, z0, c.rdistcutoff, static_cast<double>(c.zd0));
        }
        return damp;
    }

    SIRIUS_HD Cd filterScale(const FilterCtx& c, int order,
                             double x1, double y1, double z0, bool* inSupport) {
        const double x1f = x1 * c.dkx;
        const double y1f = y1 * c.dky;
        const double kx = order * c.k0all[2 * c.direction + 0];
        const double ky = order * c.k0all[2 * c.direction + 1];

        const double rdist1 = sqrt(x1f * x1f + y1f * y1f);
        *inSupport = rdist1 <= c.rdistcutoff;
        if (!*inSupport) return cd(0, 0);

        const double xabs = x1f + kx;
        const double yabs = y1f + ky;
        const double rdistabs = sqrt(xabs * xabs + yabs * yabs);

        const Cd otf1 = otfInterpolate(c.otf, order, x1f, y1f, z0);

        // self weight with the band-center / order-0 damping
        double dampfact = bandWeightDamping(c, order, rdist1, z0);
        if (order == 0 && c.noKz0 && z0 == 0.0) dampfact = 0.0;

        double weight = cabs2(otf1);
        if (order != 0) weight *= c.ampmag2[c.direction * c.norders + order];
        weight *= dampfact;
        double sumweight = weight;

        // every other (direction, order) band overlapping this sample
        for (int dir2 = 0; dir2 < c.ndirs; ++dir2) {
            for (int order2 = -(c.norders - 1); order2 < c.norders; ++order2) {
                if (dir2 == c.direction && order2 == order) continue;
                if (!c.filterOverlaps && !(order2 == 0 && order == 0)) continue;
                const int ao = order2 < 0 ? -order2 : order2;
                const double x2 = xabs - order2 * c.k0all[2 * dir2 + 0];
                const double y2 = yabs - order2 * c.k0all[2 * dir2 + 1];
                const double rdist2 = sqrt(x2 * x2 + y2 * y2);
                if (!(rdist2 < c.rdistcutoff)) continue;

                const Cd otf2 = otfInterpolate(c.otf, ao, x2, y2, z0);
                double w = cabs2(otf2);   // noise variance factors == 1
                if (order2 != 0) w *= c.ampmag2[dir2 * c.norders + ao];
                w *= bandWeightDamping(c, order2, rdist2, z0);
                if (c.noKz0 && order2 == 0 && z0 == 0.0) w = 0.0;
                sumweight += w;
            }
        }
        sumweight += c.wiener2;

        Cd scale = cscale(cconj(otf1), dampfact / sumweight);

        // output apodization on the *absolute* (assembled) frequency
        const double zdistabs = fabs(z0);
        double rho = c.zapocutoff > 0
                         ? sqrt((rdistabs / c.apocutoff) * (rdistabs / c.apocutoff) +
                                (zdistabs / c.zapocutoff) * (zdistabs / c.zapocutoff))
                         : rdistabs / c.apocutoff;
        if (rho > 1.0) rho = 1.0;
        double apofact = 1.0;
        if (c.apodizeOutput == 1) apofact = cos(0.5 * kPi * rho);
        else if (c.apodizeOutput == 2) apofact = 1.0 - rho;
        return cscale(scale, apofact);
    }

    // Filter application, first pass: the stored (kx >= 0) half; scales the
    // plus component in place. `scale` already includes conjamp.
    SIRIUS_HD void filterApplySide(Cd* re, Cd* im, Cd scale, bool inSupport) {
        if (!inSupport) {
            *re = cd(0, 0);
            *im = cd(0, 0);
            return;
        }
        const Cd bandplus = cmul(cplusi(*re, *im), scale);
        const Cd bandminus = cminusi(*re, *im);
        *re = cscale(cadd(bandplus, bandminus), 0.5);
        *im = cmulNegHalfI(csub(bandplus, bandminus));
    }

    // Second pass: mirrored (kx < 0) coordinates; conjugates the values the
    // first pass wrote, scales the minus component, conjugates back.
    // Out-of-support samples keep the first pass's values.
    SIRIUS_HD void filterApplySideMirror(Cd* re, Cd* im, Cd scale, bool inSupport) {
        if (!inSupport) return;
        const Cd cre = cconj(*re);
        const Cd cim = cconj(*im);
        const Cd bandplus = cmul(cplusi(cre, cim), scale);
        const Cd bandminus = cminusi(cre, cim);
        *re = cconj(cscale(cadd(bandplus, bandminus), 0.5));
        *im = cconj(cmulNegHalfI(csub(bandplus, bandminus)));
    }

    // ---------------------------------------------------------------------
    // Assembly (ports of move_kernel and write_outbuffer kernels).
    // ---------------------------------------------------------------------

    // One element of the zero-padded big Fourier grid: gathers the band value
    // for output column t in [0, nx) of the (zi, yi) plane of the *small*
    // grid, and its destination indices in the big grid. Returns false for
    // the never-written column (none here: t enumerates xs_pos ++ xs_neg).
    struct MoveCtx {
        IndexT nx, ny, nz, nxh;
        IndexT xdim, ydim, zdim;
        int order;            // 0: band_im unused
    };

    SIRIUS_HD void moveBandElement(const MoveCtx& c, const Cd* bandRe, const Cd* bandIm,
                                   Cd* big, IndexT zi, IndexT yi, IndexT t) {
        // signed frequencies enumerated exactly as move_kernel does
        const IndexT ySigned = yi - (c.ny / 2 - 1);
        const IndexT zSigned = c.nz > 1 ? zi - (c.nz / 2 - 1) : 0;
        const IndexT xSigned = t <= c.nx / 2 ? t : t - c.nx;   // [-(nx/2-1) .. nx/2]

        const IndexT yout = signedToStorage(ySigned, c.ydim);
        const IndexT zout = signedToStorage(zSigned, c.zdim);
        const IndexT xout = signedToStorage(xSigned, c.xdim);

        const bool conjFlag = xSigned < 0;
        const IndexT xin = conjFlag ? -xSigned : xSigned;
        const IndexT yin = signedToStorage(conjFlag ? -ySigned : ySigned, c.ny);
        const IndexT zin = signedToStorage(conjFlag ? -zSigned : zSigned, c.nz);

        const IndexT src = (zin * c.ny + yin) * c.nxh + xin;
        Cd v;
        if (c.order == 0) {
            v = bandRe[src];
            if (conjFlag) v = cconj(v);
        } else {
            Cd re = bandRe[src];
            Cd im = bandIm[src];
            if (conjFlag) {
                re = cconj(re);
                im = cconj(im);
            }
            v = cplusi(re, im);
        }
        big[(zout * c.ydim + yout) * c.xdim + xout] = v;
    }

    // Real-space accumulation of one inverse-transformed big-grid band:
    // order 0 adds its real part; side bands are shifted to +-order*k0 via a
    // cos/sin carrier and the conjugate pair sums to twice the real part.
    SIRIUS_HD double accumulateValue(const Cd* big, IndexT idx, int order,
                                     double angle) {
        if (order == 0) return big[idx].re;
        return 2.0 * (big[idx].re * cos(angle) - big[idx].im * sin(angle));
    }

} // namespace sirius::simdetail

#endif // SIRIUS_SIM_MATH_HPP
