"""Read speed of SIRIUS's TIFF reader against tifffile on representative files.

Each case is written once with tifffile (excluded from timing), then read
with both readers: one warm-up read (the file then sits in the OS page
cache), and the minimum of --repeats timed reads. Both return the whole file
as one array in their native layout (SIRIUS: (pages, samples, y, x) for RGB,
tifffile: (pages, y, x, samples)); every result is checked equal before it
is timed. tifffile decodes compressed files with a thread pool (its default
maxworkers); SIRIUS with OpenMP over pages, and over the strips / tiles of a
page when there are fewer pages than threads. The ZSTD and JPEG cases need
imagecodecs (tifffile's codecs) to write and read the reference.

Usage:
    python bindings/benchmarks/bench_tiff_vs_tifffile.py [--scale 0.25] [--repeats 3] [--dir D]

--scale shrinks every case (page count / image edge), for a quick run.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

import numpy as np
import tifffile

import sirius


def _cases(scale: float):
    s = max(scale, 1e-3)
    rng = np.random.default_rng(0)

    def smooth(shape, dtype):
        # microscopy-like: a smooth field plus Poisson-ish noise, so codecs compress realistically
        y = np.linspace(0, 6, shape[-2], dtype=np.float32)[:, None]
        x = np.linspace(0, 6, shape[-1], dtype=np.float32)[None, :]
        base = (np.sin(y) * np.cos(x) + 1.0) * 400.0
        out = np.empty(shape, dtype)
        for i in np.ndindex(shape[:-2]):
            out[i] = (base + rng.normal(0, 20, shape[-2:])).clip(0, np.iinfo(dtype).max).astype(dtype)
        return out

    pages = max(int(128 * s), 2)
    yield ("uint16 stack, uncompressed, 128 x 1024^2", smooth((pages, 1024, 1024), np.uint16), {"photometric": "minisblack"})
    yield (
        "uint16 stack, Deflate + predictor, 64 x 1024^2",
        smooth((max(int(64 * s), 2), 1024, 1024), np.uint16),
        {"photometric": "minisblack", "compression": "zlib", "predictor": True},
    )
    yield (
        "uint16 stack, ZSTD + predictor, 64 x 1024^2",
        smooth((max(int(64 * s), 2), 1024, 1024), np.uint16),
        {"photometric": "minisblack", "compression": "zstd", "predictor": True},
    )
    edge = max(int(8192 * s**0.5) // 256 * 256, 512)
    yield (
        f"uint16 single page {edge}^2, Deflate + predictor, 256^2 tiles",
        smooth((edge, edge), np.uint16),
        {"photometric": "minisblack", "compression": "zlib", "predictor": True, "tile": (256, 256)},
    )
    rgb = smooth((max(int(16 * s), 2), 3, 2048, 2048), np.uint16)
    rgb = np.moveaxis((rgb >> 2).astype(np.uint8), 1, -1).copy()
    yield ("RGB uint8, LZW, 16 x 2048^2", rgb, {"photometric": "rgb", "compression": "lzw"})
    yield ("RGB uint8, uncompressed, 16 x 2048^2", rgb, {"photometric": "rgb"})
    # JPEG: tifffile stores RGB as 2x2-subsampled YCbCr; both read it back as RGB
    yield ("RGB uint8, JPEG (YCbCr 2x2), 16 x 2048^2", rgb, {"photometric": "rgb", "compression": "jpeg"})


def _best(fn, repeats: int):
    fn()  # warm-up: page cache, thread pools
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
        del out
    return min(times)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dir", default=None, help="where to write the files (default: a temporary folder)")
    args = ap.parse_args()

    tmp = None if args.dir else tempfile.TemporaryDirectory()
    folder = args.dir or tmp.name
    print(f"sirius {getattr(sirius, '__version__', '?')}, tifffile {tifffile.__version__}, {os.cpu_count()} logical CPUs")
    print(f"{'case':58s} {'MB':>7s} {'tifffile s':>11s} {'sirius s':>9s} {'speed-up':>9s}")
    try:
        for i, (name, data, kw) in enumerate(_cases(args.scale)):
            path = os.path.join(folder, f"bench{i}.tif")
            tifffile.imwrite(path, data, **kw)
            del data
            ref = tifffile.imread(path)
            got = np.asarray(sirius.read_tiff(path))
            if ref.ndim == got.ndim and got.ndim >= 3 and ref.shape[-1] == got.shape[1] and ref.shape != got.shape:
                got_cmp = np.moveaxis(got, 1, -1)  # RGB: (pages, s, y, x) -> (pages, y, x, s)
            else:
                got_cmp = got
            if not np.array_equal(got_cmp.reshape(ref.shape), ref):
                raise SystemExit(f"{name}: SIRIUS and tifffile disagree")
            mb = ref.nbytes / 1e6
            del ref, got, got_cmp
            t_tf = _best(lambda p=path: tifffile.imread(p), args.repeats)
            t_sr = _best(lambda p=path: sirius.read_tiff(p), args.repeats)
            print(f"{name:58s} {mb:7.0f} {t_tf:11.3f} {t_sr:9.3f} {t_tf / t_sr:8.2f}x", flush=True)
            os.remove(path)
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()
