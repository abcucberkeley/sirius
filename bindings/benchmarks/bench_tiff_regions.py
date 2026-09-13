#!/usr/bin/env python3
"""Benchmark the access patterns a training pipeline actually uses, against
tifffile, on real acquisition files rather than round-trips of our own writes.

The existing bench_tiff.py measures whole-stack decode against cpp-tiff. That is
not what a data loader does: it reads a *window* out of a large stack, many
times, from many processes at once. This measures the three patterns that cost
real time:

    metadata   shape/dtype/page count without decoding pixels  (survey, staging)
    pages      a contiguous run of z-planes                    (block staging)
    region     a sub-rectangle of a run of z-planes            (crop sampling)

and reports sirius against tifffile for each, plus a correctness check that the
two agree bit for bit.

    python bench_tiff_regions.py --files '/path/*.tif' --repeat 3
    python bench_tiff_regions.py --files ... --device cuda     # nvTIFF path
"""
from __future__ import annotations

import argparse
import glob
import statistics
import time

import numpy as np


def timeit(fn, repeat: int, warmup: int = 1):
    """Minimum of `repeat` runs, after warmup. Minimum, not mean: we want the
    cost without scheduler and page-cache noise, and the floor is the honest
    comparison when another job is hammering the same filesystem."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), statistics.median(ts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", required=True, help="glob of TIFFs to benchmark")
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--pages", type=int, default=32, help="z-planes per read (a staged block is 32)")
    p.add_argument("--region", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--no-verify", action="store_true")
    a = p.parse_args()

    import sirius
    import tifffile

    paths = sorted(glob.glob(a.files))[: a.limit]
    if not paths:
        raise SystemExit(f"no files matched {a.files}")
    print(f"sirius {getattr(sirius, '__version__', '?')}  tifffile {tifffile.__version__}  "
          f"device={a.device}  {len(paths)} files, {a.repeat} repeats (minimum reported)")
    print(f"{'file':34s} {'pages':>6s} {'shape':>18s} {'pattern':>9s} "
          f"{'sirius ms':>10s} {'tifffile ms':>12s} {'speedup':>8s} {'match':>6s}")

    totals = {}
    for path in paths:
        name = path.rsplit("/", 1)[-1][:34]
        f = sirius.TiffFile(path)
        info = f.info
        n_pages = info.page_count
        h, w = info.height, info.width
        first = max(0, n_pages // 2 - a.pages // 2)
        count = min(a.pages, n_pages)
        rh, rw = min(a.region[0], h), min(a.region[1], w)
        y0, x0 = (h - rh) // 2, (w - rw) // 2

        cases = {
            "metadata": (lambda: sirius.inspect_tiff(path),
                         lambda: tifffile.TiffFile(path).series[0].shape),
            "pages": (lambda: f.read_pages(first, count, device=a.device),
                      lambda: tifffile.imread(path, key=range(first, first + count))),
            "region": (lambda: f.read_region(x0, y0, rw, rh, level=0, device=a.device),
                       # tifffile has no region read: the honest comparison is
                       # decode the pages then slice, which is what we do today
                       lambda: tifffile.imread(path, key=range(first, first + count))[:, y0:y0 + rh, x0:x0 + rw]),
        }
        for pattern, (sfn, tfn) in cases.items():
            try:
                s_min, _ = timeit(sfn, a.repeat)
                t_min, _ = timeit(tfn, a.repeat)
            except Exception as exc:  # noqa: BLE001 -- a failure is a result
                print(f"{name:34s} {n_pages:6d} {str((n_pages, h, w)):>18s} {pattern:>9s} "
                      f"   ERROR {type(exc).__name__}: {str(exc)[:40]}")
                continue
            match = "-"
            if not a.no_verify and pattern in ("pages", "region"):
                sa = np.asarray(sfn()); ta = np.asarray(tfn())
                if sa.shape == ta.shape:
                    match = "yes" if np.array_equal(sa, ta) else "NO"
                else:
                    match = f"{sa.shape}!={ta.shape}"
            print(f"{name:34s} {n_pages:6d} {str((n_pages, h, w)):>18s} {pattern:>9s} "
                  f"{1e3 * s_min:10.1f} {1e3 * t_min:12.1f} {t_min / max(s_min, 1e-9):7.2f}x {match:>6s}")
            totals.setdefault(pattern, []).append(t_min / max(s_min, 1e-9))

    print("\nmedian speedup over tifffile, by pattern:")
    for pattern, v in totals.items():
        print(f"  {pattern:9s} {statistics.median(v):.2f}x   (n={len(v)})")


if __name__ == "__main__":
    main()
