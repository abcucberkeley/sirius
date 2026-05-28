"""Benchmark the SIRIUS TIFF reader against cpp-tiff, at both binding levels.

Compares read throughput of four readers on one synthesized BigTIFF stack:

    sirius   python   sirius.read_tiff(path)            -> ndarray (z, y, x)
    cpp-tiff python   cpptiff.read_tiff(path)           -> ndarray (z, y, x)
    sirius   cpp      bench_tiff_sirius   (shells out)
    cpp-tiff cpp      bench_tiff_cpptiff  (shells out)

Both libraries decode the whole stack with OpenMP and perform no XY transpose
(the C++ bench calls cpp-tiff's readTiffParallelWrapperNoXYFlip -- the exact
wrapper cpp-tiff's own Python binding wraps -- so cpp vs python is apples to
apples: the python row just adds the binding's marshalling + a cheap view
transpose on top of the identical read).

Methodology
-----------
* Dataset generation is excluded from timing: a random BigTIFF is written once
  with tifffile (page by page, RAM stays flat), then read repeatedly.
* Same timing harness as bench_fft.py: one warm-up read, then the minimum of
  --repeats timed reads (min filters OS scheduling noise).
* The C++ benches do their own warm-up + min internally and print the result.

Caveats
-------
* Warm-cache metric. The first (warm-up) read is cold off disk; the dataset
  then lives in the OS page cache, so timed reads measure parallel decode +
  allocation, not disk bandwidth. This is the interesting CPU-bound comparison.
  Dropping caches between reads (a true cold metric) is left as a follow-up.
* Both backends use OpenMP. Benches run sequentially (one reader at a time) so
  the two OpenMP runtimes never oversubscribe. OMP_NUM_THREADS is honored.

Usage
-----
    # tiny correctness check across all available readers
    python bindings/benchmarks/bench_tiff.py --shape 8 64 64 --verify

    # full case: ~18 GB, write to gitignored ./.bench_tmp/, delete after
    python bindings/benchmarks/bench_tiff.py --shape 10000 1800 512 --repeats 3 \
        --cpp-sirius  build/linux-gcc-release/benchmarks/bench_tiff_sirius \
        --cpp-cpptiff .bench_tmp/cpptiff/bench_tiff_cpptiff

    # just write the dataset (CI two-step / debugging); keeps the file
    python bindings/benchmarks/bench_tiff.py --shape 8 64 64 --path /tmp/sm.tif --generate-only
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Timing core (reused from bench_fft.py; local fallback keeps this script      #
# usable for --generate-only / C++-only runs even if sirius/bench_fft can't    #
# be imported).                                                                #
# --------------------------------------------------------------------------- #

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from bench_fft import _autorange, bench, fmt_time  # type: ignore
except Exception:
    def _autorange(fn: Callable[[], None], target_sec: float = 0.1) -> int:
        number = 1
        while True:
            t0 = time.perf_counter()
            for _ in range(number):
                fn()
            if time.perf_counter() - t0 >= target_sec:
                return number
            number *= 10

    def bench(fn: Callable[[], None], *, repeats: int = 5) -> float:
        fn()  # warm-up
        number = _autorange(fn)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(number):
                fn()
            times.append((time.perf_counter() - t0) / number)
        return min(times)

    def fmt_time(t: float) -> str:
        for scale, unit in ((1e-9, "ns"), (1e-6, "us"), (1e-3, "ms"), (1.0, "s")):
            if t < scale * 1000:
                return f"{t / scale:7.2f} {unit}"
        return f"{t:7.2f}  s"


REPO_ROOT = Path(__file__).resolve().parents[2]
DTYPES = ("uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64")


def _warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #

def generate_tiff(path: Path, shape: tuple[int, int, int], dtype: np.dtype,
                  *, seed: int, compression: str) -> None:
    """Write a random (z, y, x) BigTIFF page by page so RAM stays flat."""
    import tifffile

    z, y, x = shape
    comp = None if compression in (None, "none", "None") else compression
    contiguous = comp is None  # contiguous strips are incompatible with compression
    rng = np.random.default_rng(seed)
    is_float = np.issubdtype(dtype, np.floating)
    info = None if is_float else np.iinfo(dtype)

    with tifffile.TiffWriter(str(path), bigtiff=True) as tif:
        for _ in range(z):
            if is_float:
                page = rng.standard_normal((y, x)).astype(dtype, copy=False)
            else:
                page = rng.integers(info.min, info.max, size=(y, x),
                                    dtype=dtype, endpoint=True)
            tif.write(page, photometric="minisblack",
                      compression=comp, contiguous=contiguous)


def file_geometry(path: Path) -> tuple[tuple[int, ...], np.dtype]:
    """Read (shape, dtype) from TIFF metadata without loading pixel data."""
    import tifffile

    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0]
        return tuple(int(s) for s in series.shape), np.dtype(series.dtype)


# --------------------------------------------------------------------------- #
# Readers                                                                      #
# --------------------------------------------------------------------------- #

def discover_python_readers() -> list[tuple[str, Callable[[str], np.ndarray]]]:
    readers: list[tuple[str, Callable[[str], np.ndarray]]] = []
    try:
        import sirius
        readers.append(("sirius", sirius.read_tiff))
    except Exception as e:  # noqa: BLE001 - report and continue
        _warn(f"sirius python reader unavailable: {e}")
    try:
        import cpptiff
        readers.append(("cpp-tiff", cpptiff.read_tiff))
    except Exception as e:  # noqa: BLE001
        _warn(f"cpp-tiff python reader unavailable (pip install cpp-tiff): {e}")
    return readers


def _first_executable(*patterns: str) -> Optional[Path]:
    for pat in patterns:
        for p in sorted(REPO_ROOT.glob(pat)):
            if p.is_file() and os.access(p, os.X_OK):
                return p
    return None


def discover_cpp_benches(cpp_sirius: Optional[str],
                         cpp_cpptiff: Optional[str]) -> list[tuple[str, Path]]:
    benches: list[tuple[str, Path]] = []
    sb = Path(cpp_sirius) if cpp_sirius else _first_executable(
        "build/*/benchmarks/bench_tiff_sirius", "build/benchmarks/bench_tiff_sirius")
    if sb and Path(sb).is_file():
        benches.append(("sirius", Path(sb)))
    elif cpp_sirius:
        _warn(f"--cpp-sirius not found: {cpp_sirius}")

    cb = Path(cpp_cpptiff) if cpp_cpptiff else _first_executable(
        ".bench_tmp/cpptiff/bench_tiff_cpptiff", ".bench_tmp/**/bench_tiff_cpptiff")
    if cb and Path(cb).is_file():
        benches.append(("cpp-tiff", Path(cb)))
    elif cpp_cpptiff:
        _warn(f"--cpp-cpptiff not found: {cpp_cpptiff}")
    return benches


def run_cpp_bench(binary: Path, file: Path, repeats: int) -> tuple[float, int]:
    """Invoke a C++ bench; parse its trailing '<name>\\t<seconds>\\t<bytes>'."""
    proc = subprocess.run([str(binary), str(file), str(repeats)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{binary} exited {proc.returncode}: {proc.stderr.strip()}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"{binary} produced no output")
    _name, secs, nbytes = lines[-1].split("\t")
    return float(secs), int(nbytes)


# --------------------------------------------------------------------------- #
# Results / presentation                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class Result:
    reader: str   # "sirius" | "cpp-tiff"
    lang: str     # "python" | "cpp"
    seconds: float
    nbytes: int

    @property
    def gbps(self) -> float:
        return self.nbytes / self.seconds / 1e9


def print_table(results: list[Result]) -> None:
    # Per-language baseline: speedup is measured against cpp-tiff in the same lang.
    baseline = {r.lang: r.seconds for r in results if r.reader == "cpp-tiff"}
    order = {("python", "sirius"): 0, ("python", "cpp-tiff"): 1,
             ("cpp", "sirius"): 2, ("cpp", "cpp-tiff"): 3}
    results = sorted(results, key=lambda r: order.get((r.lang, r.reader), 99))

    header = (f"{'reader':<9} {'lang':<7} {'min time':>11} "
              f"{'GB/s':>8} {'vs cpp-tiff':>12}")
    print(header)
    print("-" * len(header))
    for r in results:
        base = baseline.get(r.lang)
        speed = f"{base / r.seconds:.2f}x" if base else "-"
        print(f"{r.reader:<9} {r.lang:<7} {fmt_time(r.seconds):>11} "
              f"{r.gbps:>8.2f} {speed:>12}")


# --------------------------------------------------------------------------- #
# Verify                                                                       #
# --------------------------------------------------------------------------- #

def verify(path: Path, readers: list[tuple[str, Callable[[str], np.ndarray]]]) -> None:
    if len(readers) < 2:
        _warn("verify needs >=2 python readers; skipping cross-check")
        return
    arrays = {name: fn(str(path)) for name, fn in readers}
    names = list(arrays)
    ref_name = names[0]
    ref = arrays[ref_name]
    for name in names[1:]:
        a = arrays[name]
        if a.shape != ref.shape:
            raise AssertionError(
                f"shape mismatch: {name}={a.shape} vs {ref_name}={ref.shape}")
        if not np.array_equal(a, ref):
            raise AssertionError(f"data mismatch: {name} != {ref_name}")
    print(f"verify OK: {names} agree (shape={ref.shape}, dtype={ref.dtype})")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shape", type=int, nargs=3, metavar=("Z", "Y", "X"),
                   default=[10000, 1800, 512], help="stack shape (default: 10000 1800 512)")
    p.add_argument("--dtype", choices=DTYPES, default="uint16",
                   help="pixel dtype (default: uint16)")
    p.add_argument("--compression", default="none",
                   help="tifffile compression for the dataset (default: none)")
    p.add_argument("--repeats", type=int, default=3,
                   help="timed reads per reader; best is reported (default: 3)")
    p.add_argument("--seed", type=int, default=0xC0FFEE, help="RNG seed for the dataset")
    p.add_argument("--path", type=str, default=None,
                   help="explicit dataset path (used + kept; generated if absent)")
    p.add_argument("--keep", action="store_true",
                   help="keep the auto-generated temp dataset instead of deleting it")
    p.add_argument("--generate", action="store_true",
                   help="force (re)generation even if --path already exists")
    p.add_argument("--generate-only", action="store_true",
                   help="write the dataset and exit (no reading/timing)")
    p.add_argument("--verify", action="store_true",
                   help="read once with every python reader and assert arrays equal")
    p.add_argument("--cpp-sirius", default=None, help="path to bench_tiff_sirius binary")
    p.add_argument("--cpp-cpptiff", default=None, help="path to bench_tiff_cpptiff binary")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    shape = tuple(args.shape)
    dtype = np.dtype(args.dtype)

    # Resolve dataset path + lifecycle.
    if args.path:
        path = Path(args.path)
        keep = True  # never auto-delete a user-specified file
        need_generate = args.generate or args.generate_only or not path.exists()
    else:
        bench_tmp = REPO_ROOT / ".bench_tmp"
        bench_tmp.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix="bench_", suffix=".tif", dir=str(bench_tmp))
        os.close(fd)
        path = Path(tmp)
        keep = args.keep
        need_generate = True  # fresh temp file

    try:
        if need_generate:
            nbytes_expected = math.prod(shape) * dtype.itemsize
            print(f"generating {path} "
                  f"({'x'.join(map(str, shape))} {dtype}, "
                  f"{nbytes_expected / 1e9:.2f} GB) ...", flush=True)
            t0 = time.perf_counter()
            generate_tiff(path, shape, dtype, seed=args.seed,
                          compression=args.compression)
            print(f"  wrote in {time.perf_counter() - t0:.1f}s (excluded from timing)")
            geom_shape, geom_dtype = shape, dtype
        else:
            geom_shape, geom_dtype = file_geometry(path)

        if args.generate_only:
            print(f"dataset ready: {path}")
            return

        nbytes = math.prod(geom_shape) * geom_dtype.itemsize

        py_readers = discover_python_readers()
        cpp_benches = discover_cpp_benches(args.cpp_sirius, args.cpp_cpptiff)

        if args.verify:
            verify(path, py_readers)

        print()
        print(f"shape={'x'.join(map(str, geom_shape))} dtype={geom_dtype} | "
              f"warm-cache reads, repeats={args.repeats}, "
              f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<all>')} | "
              f"GB = 1e9 bytes")
        print()

        results: list[Result] = []

        # Python readers: time in-process with the shared harness.
        for name, fn in py_readers:
            print(f"{name:<9} python ...", end=" ", flush=True)
            secs = bench(lambda fn=fn: fn(str(path)), repeats=args.repeats)
            print(f"{fmt_time(secs)}")
            results.append(Result(name, "python", secs, nbytes))

        # C++ benches: shell out; each does its own warm-up + min.
        for name, binary in cpp_benches:
            print(f"{name:<9} cpp    ... ({binary})", flush=True)
            try:
                secs, cbytes = run_cpp_bench(binary, path, args.repeats)
            except Exception as e:  # noqa: BLE001
                _warn(f"{name} cpp bench failed: {e}")
                continue
            results.append(Result(name, "cpp", secs, cbytes))

        if not results:
            _warn("no readers available; nothing to report")
            return

        print()
        print_table(results)
    finally:
        if not keep and path.exists():
            path.unlink()
            print(f"\nremoved temp dataset {path}")


if __name__ == "__main__":
    main()
