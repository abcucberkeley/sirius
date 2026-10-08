// The index arithmetic of the band grids (src/sim_math.hpp), where the parity
// of an extent enters: signedFrequency, r2cColumns, mirrorColumns, and
// moveBandElement, which is the only kernel whose indices depend on all
// three.
//
// Odd lateral sizes are still refused by every guard, so these cases are
// what makes the parity-general form safe: for an EVEN extent they assert
// that the index set is exactly the one the expression it replaced produced
// (`i - (n/2 - 1)` for y and z, `-(nx/2 - 1)` for the filter's mirror pass),
// and for an ODD extent that it is the complete set, which that expression
// was not.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <vector>

#include "sim_math.hpp"

using namespace sirius::simdetail;

namespace {

    // --- the two expressions this change replaced -----------------------------

    // moveBandElement's y and z centring until 2026-10-08, and what
    // cudasirecon's move_kernel still does on all three axes
    // (gpuFunctionsImpl.cu:1872-1875).
    IndexT oldCentred(IndexT i, IndexT n) { return n > 1 ? i - (n / 2 - 1) : 0; }

    // the whole of moveBandElement with that centring, so the grid the new
    // form writes can be compared with the grid the old one wrote
    void moveBandElementOld(const MoveCtx& c, const Cd* bandRe, const Cd* bandIm,
                            Cd* big, IndexT zi, IndexT yi, IndexT t) {
        const IndexT ySigned = yi - (c.ny / 2 - 1);
        const IndexT zSigned = c.nz > 1 ? zi - (c.nz / 2 - 1) : 0;
        const IndexT xSigned = t <= c.nx / 2 ? t : t - c.nx;

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

    // --- fixtures -------------------------------------------------------------

    constexpr double kSentinel = -7.5e300;   // never produced by the band values below

    MoveCtx makeCtx(IndexT nx, IndexT ny, IndexT nz, IndexT zoom, int order) {
        MoveCtx c{};
        c.nx = nx;
        c.ny = ny;
        c.nz = nz;
        c.nxh = nx / 2 + 1;
        c.xdim = zoom * nx;
        c.ydim = zoom * ny;
        c.zdim = nz > 1 ? zoom * nz : 1;
        c.order = order;
        return c;
    }

    // distinct, exactly representable, and no two of them equal under
    // conjugation or under cplusi, so a swapped index cannot pass unnoticed
    std::vector<Cd> bandValues(std::size_t n, double base) {
        std::vector<Cd> v(n);
        for (std::size_t i = 0; i < n; ++i)
            v[i] = cd(base + static_cast<double>(i), base / 8.0 + 0.5 * static_cast<double>(i));
        return v;
    }

    // Every element of the big grid moveBandElement should write, derived
    // from the frequencies rather than from the storage indices: an n-point
    // axis carries kz, ky, kx in [-((n-1)/2) .. n/2], which is n values for
    // either parity (the Nyquist of an even axis counted positive), and the
    // value at each is the band's, read out of the stored kx >= 0 half with
    // Hermitian symmetry.
    std::vector<Cd> referenceBig(const MoveCtx& c, const std::vector<Cd>& bandRe,
                                 const std::vector<Cd>& bandIm) {
        std::vector<Cd> big(static_cast<std::size_t>(c.zdim * c.ydim * c.xdim), cd(kSentinel, kSentinel));
        const IndexT kzLo = c.nz > 1 ? -((c.nz - 1) / 2) : 0, kzHi = c.nz > 1 ? c.nz / 2 : 0;
        for (IndexT kz = kzLo; kz <= kzHi; ++kz)
            for (IndexT ky = -((c.ny - 1) / 2); ky <= c.ny / 2; ++ky)
                for (IndexT kx = -((c.nx - 1) / 2); kx <= c.nx / 2; ++kx) {
                    const bool conjFlag = kx < 0;
                    const IndexT sx = conjFlag ? -kx : kx;
                    const IndexT sy = ((conjFlag ? -ky : ky) + c.ny) % c.ny;
                    const IndexT sz = ((conjFlag ? -kz : kz) + c.nz) % c.nz;
                    const std::size_t src = static_cast<std::size_t>((sz * c.ny + sy) * c.nxh + sx);
                    Cd v;
                    if (c.order == 0) {
                        v = bandRe[src];
                        if (conjFlag) v = cconj(v);
                    } else {
                        Cd re = bandRe[src], im = bandIm[src];
                        if (conjFlag) {
                            re = cconj(re);
                            im = cconj(im);
                        }
                        v = cplusi(re, im);
                    }
                    const IndexT ox = (kx + c.xdim) % c.xdim;
                    const IndexT oy = (ky + c.ydim) % c.ydim;
                    const IndexT oz = (kz + c.zdim) % c.zdim;
                    big[static_cast<std::size_t>((oz * c.ydim + oy) * c.xdim + ox)] = v;
                }
        return big;
    }

    // moveBandElement over the whole small grid, as both backends drive it
    std::vector<Cd> runMove(const MoveCtx& c, const std::vector<Cd>& bandRe,
                            const std::vector<Cd>& bandIm, bool oldForm) {
        std::vector<Cd> big(static_cast<std::size_t>(c.zdim * c.ydim * c.xdim), cd(kSentinel, kSentinel));
        for (IndexT zi = 0; zi < c.nz; ++zi)
            for (IndexT yi = 0; yi < c.ny; ++yi)
                for (IndexT t = 0; t < c.nx; ++t) {
                    if (oldForm)
                        moveBandElementOld(c, bandRe.data(), bandIm.data(), big.data(), zi, yi, t);
                    else
                        moveBandElement(c, bandRe.data(), bandIm.data(), big.data(), zi, yi, t);
                }
        return big;
    }

    std::size_t written(const std::vector<Cd>& big) {
        return static_cast<std::size_t>(
            std::count_if(big.begin(), big.end(), [](Cd v) { return v.re != kSentinel; }));
    }

    std::vector<IndexT> sortedFrequencies(IndexT n, bool oldForm) {
        std::vector<IndexT> f;
        for (IndexT i = 0; i < n; ++i) f.push_back(oldForm ? oldCentred(i, n) : signedFrequency(i, n));
        std::sort(f.begin(), f.end());
        return f;
    }

    // every frequency of an n-point axis, Nyquist positive
    std::vector<IndexT> canonical(IndexT n) {
        std::vector<IndexT> f;
        for (IndexT k = -((n - 1) / 2); k <= n / 2; ++k) f.push_back(k);
        return f;
    }

} // namespace

TEST_CASE("signedFrequency enumerates an even axis's own set and the whole of an odd one",
          "[sim_math][parity]") {
    SECTION("even: the same set as the expression it replaced") {
        for (IndexT n : {2, 4, 6, 8, 64, 128, 482}) {
            INFO("n = " << n);
            CHECK(sortedFrequencies(n, /*oldForm=*/false) == sortedFrequencies(n, /*oldForm=*/true));
            CHECK(sortedFrequencies(n, false) == canonical(n));
            CHECK(canonical(n).front() == -(n / 2 - 1));
            CHECK(canonical(n).back() == n / 2);
            CHECK(static_cast<IndexT>(canonical(n).size()) == n);
        }
    }
    SECTION("odd: the complete set, which the old expression was not") {
        for (IndexT n : {3, 5, 9, 63, 101}) {
            INFO("n = " << n);
            CHECK(sortedFrequencies(n, false) == canonical(n));
            CHECK(static_cast<IndexT>(canonical(n).size()) == n);
            CHECK(sortedFrequencies(n, true) != canonical(n));
            // the old form omitted -(n/2) and produced (n+1)/2, which an
            // n-point axis does not have; on the r2c axis x, |(n+1)/2| is
            // also one column past the last stored one
            CHECK(sortedFrequencies(n, true).front() == -(n / 2 - 1));
            CHECK(sortedFrequencies(n, true).back() == n / 2 + 1);
            // and (n+1)/2 == n/2 + 1 is r2cColumns(n): one past the last
            // stored column, whose index is r2cColumns(n) - 1
            CHECK(n / 2 + 1 == r2cColumns(n));
        }
    }
    SECTION("a single plane needs no special case") {
        CHECK(signedFrequency(0, 1) == 0);
        CHECK(canonical(1) == std::vector<IndexT>{0});
    }
}

TEST_CASE("r2cColumns and mirrorColumns split the stored half into every frequency once",
          "[sim_math][parity]") {
    auto visited = [](IndexT n) {
        std::vector<IndexT> f;
        for (IndexT x1 = 0; x1 < r2cColumns(n); ++x1) f.push_back(x1);          // pass 1
        for (IndexT x1 = -mirrorColumns(n); x1 < 0; ++x1) f.push_back(x1);      // pass 2
        std::sort(f.begin(), f.end());
        return f;
    };
    SECTION("even: mirrorColumns is the old -(nx/2 - 1) bound exactly") {
        for (IndexT n : {4, 6, 8, 64, 128, 562}) {
            INFO("n = " << n);
            CHECK(r2cColumns(n) == n / 2 + 1);
            CHECK(mirrorColumns(n) == n / 2 - 1);
            const std::vector<IndexT> v = visited(n);
            CHECK(v == canonical(n));
            // the Nyquist column is its own mirror and is visited once, by pass 1
            CHECK(std::count(v.begin(), v.end(), n / 2) == 1);
            CHECK(std::count(v.begin(), v.end(), -(n / 2)) == 0);
        }
        CHECK(mirrorColumns(2) == 0);   // nothing to mirror, as before
    }
    SECTION("odd: one column more, the one the old bound left out") {
        for (IndexT n : {3, 5, 9, 63}) {
            INFO("n = " << n);
            CHECK(r2cColumns(n) == n / 2 + 1);
            CHECK(mirrorColumns(n) == n / 2);
            CHECK(mirrorColumns(n) == (n / 2 - 1) + 1);
            const std::vector<IndexT> v = visited(n);
            CHECK(v == canonical(n));
            // the last stored column's negative is a frequency of its own
            CHECK(std::count(v.begin(), v.end(), -(n / 2)) == 1);
        }
    }
}

TEST_CASE("moveBandElement writes the same big grid on even axes as the expression it replaced",
          "[sim_math][parity]") {
    struct Case {
        IndexT nx, ny, nz, zoom;
    };
    for (Case g : {Case{8, 6, 4, 2}, Case{6, 8, 2, 2}, Case{8, 8, 1, 2}, Case{8, 6, 4, 1}}) {
        for (int order : {0, 1}) {
            INFO("nx " << g.nx << " ny " << g.ny << " nz " << g.nz << " zoom " << g.zoom
                       << " order " << order);
            const MoveCtx c = makeCtx(g.nx, g.ny, g.nz, g.zoom, order);
            const std::size_t bandElems = static_cast<std::size_t>(c.nz * c.ny * c.nxh);
            const std::vector<Cd> bre = bandValues(bandElems, 1.0);
            const std::vector<Cd> bim = bandValues(bandElems, 1000.0);

            const std::vector<Cd> now = runMove(c, bre, bim, /*oldForm=*/false);
            const std::vector<Cd> before = runMove(c, bre, bim, /*oldForm=*/true);
            const std::vector<Cd> want = referenceBig(c, bre, bim);

            REQUIRE(now.size() == before.size());
            for (std::size_t i = 0; i < now.size(); ++i) {
                REQUIRE(now[i].re == before[i].re);
                REQUIRE(now[i].im == before[i].im);
                REQUIRE(now[i].re == want[i].re);
                REQUIRE(now[i].im == want[i].im);
            }
            // nothing was lost to a destination collision
            CHECK(written(now) == static_cast<std::size_t>(c.nz * c.ny * c.nx));
            CHECK(written(before) == written(now));
        }
    }
}

TEST_CASE("moveBandElement covers an odd axis exactly once, where the old expression aliased",
          "[sim_math][parity]") {
    struct Case {
        IndexT nx, ny, nz, zoom;
    };
    for (Case g : {Case{9, 7, 5, 2}, Case{7, 9, 1, 2}, Case{9, 8, 4, 2}}) {
        for (int order : {0, 1}) {
            INFO("nx " << g.nx << " ny " << g.ny << " nz " << g.nz << " order " << order);
            const MoveCtx c = makeCtx(g.nx, g.ny, g.nz, g.zoom, order);
            const std::size_t bandElems = static_cast<std::size_t>(c.nz * c.ny * c.nxh);
            const std::vector<Cd> bre = bandValues(bandElems, 1.0);
            const std::vector<Cd> bim = bandValues(bandElems, 1000.0);

            const std::vector<Cd> now = runMove(c, bre, bim, /*oldForm=*/false);
            const std::vector<Cd> want = referenceBig(c, bre, bim);
            for (std::size_t i = 0; i < now.size(); ++i) {
                REQUIRE(now[i].re == want[i].re);
                REQUIRE(now[i].im == want[i].im);
            }
            CHECK(written(now) == static_cast<std::size_t>(c.nz * c.ny * c.nx));

            // and every read stayed inside its own r2c row, which is what the
            // old expression (and cudasirecon's move_kernel) did not do on an
            // odd x: its largest |kx| is (nx+1)/2, one column past the last
            // stored one. Index arithmetic only -- the read itself is out of
            // bounds, so it is not performed here.
            for (IndexT t = 0; t < c.nx; ++t) {
                const IndexT kx = signedFrequency(t, c.nx);
                CHECK((kx < 0 ? -kx : kx) < c.nxh);
            }
            const IndexT oldMaxCol = c.nx - 1 - (c.nx / 2 - 1);
            CHECK(oldMaxCol == c.nxh);   // exactly one past the end of the row
        }
    }
}
