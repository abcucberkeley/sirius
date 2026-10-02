"""Datasets read where they live: the HPC backend's cluster files.

The application opens a dataset on the cluster (a path the worker's node can
read) without copying it home. Everything it shows is computed here, next to
the data, and sent at the resolution it is drawn at:

    meta     dims (c, t, z, y, x), dtype, voxel size, channel names
    plane    one (y, x) plane at full resolution          (a step's input, a hover)
    volume   one (z, y, x) volume at full resolution       (a step that needs it)
    view     what a pane draws: the XY plane at z, the XZ / YZ re-slice at y / x
             or the z maximum projection, of a region, reduced by an integer
             factor (the mean of each factor x factor block, in the source
             dtype), or the whole volume reduced to a longest side (the 3-D view)
    stats    a display window: robust percentiles of a few planes, and the range

Readers: TIFF, OME-TIFF and ImageJ hyperstacks through SIRIUS's own TIFF
reader (the ``sirius`` package built into the worker's Python: ``pip install
<checkout>``; without it a TIFF request fails and says so), and ``.npy``
arrays. A TIFF is shaped exactly as the application shapes it (page order,
OME / ImageJ dimensions and voxel size: ``sirius.workbench.tiff_dims`` /
``tiff_voxel``, the port of array_source.cpp's probeTiff, on the metadata
SIRIUS's C++ reader parses; samples per pixel are channels, so an RGB TIFF
has three, channel c being sample c % samples of the page channel
c // samples). What is read is
what is needed: a z-stack is one page range, a zoomed-in XY plane only its
visible region, a reduced XY plane a pyramid level when the file has one;
on a CUDA device with nvTIFF the pages are decoded on the GPU. The (c, t)
volumes read most recently are kept in memory, up to
``$SIRIUS_WORKER_VIEW_CACHE_MB`` (default 4096), so re-slicing and scrubbing
read the file once.

Arrays go back raw or compressed (``encode``): the client names the encodings
it accepts, the worker picks the best it has (zstd when ``zstandard`` is
installed, zlib always), with the bytes of a multi-byte sample regrouped by
significance first ("shuffle"), which is what makes 16-bit microscopy
compress.
"""

from __future__ import annotations

import collections
import os
import socket
import threading
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Dataset",
    "DatasetError",
    "encode",
    "forget_all",
    "open_dataset",
    "read_ref",
    "reduce_blocks",
]


class DatasetError(ValueError):
    """A dataset that cannot be opened or read as asked; the message is the user's."""


# --- page order (the application's PageOrder, app/core/array_source.cpp) --------------------

def _plane_of(order: str, c: int, t: int, z: int, nc: int, nt: int, nz: int) -> int:
    """Page index of (c, t, z) for pages ordered `order`, fastest axis first."""
    page, stride = 0, 1
    for a in order:
        if a == "c":
            page += c * stride
            stride *= max(nc, 1)
        elif a == "t":
            page += t * stride
            stride *= max(nt, 1)
        elif a == "z":
            page += z * stride
            stride *= max(nz, 1)
    return page


def _options_key(options: Optional[Dict[str, Any]]) -> Tuple:
    o = options or {}
    order = o.get("page_order")
    return (None if order is None else str(order), int(o.get("c", 0) or 0), int(o.get("t", 0) or 0),
            int(o.get("z", 0) or 0))


# --- the TIFF reader: the sirius package ------------------------------------------------------

TIFF_NEEDS_SIRIUS = ("reading TIFF on the cluster needs the sirius package in the worker's Python "
                     "(pip install <checkout> in the worker's venv); see app/python/slurm/README.md")


def _sirius() -> Any:
    """The sirius extension (SIRIUS's TIFF reader), or None when it does not
    import. A directory named sirius on the path imports as an empty
    namespace package, hence the attributes."""
    try:
        import sirius  # type: ignore

        sirius.inspect_tiff  # noqa: B018
        sirius.TiffFile  # noqa: B018
        sirius.Device  # noqa: B018
        return sirius
    except Exception:  # noqa: BLE001 - absent, or built for another Python / without its libraries
        return None


def _sirius_version(ext: Any) -> str:
    v = str(getattr(ext, "__version__", "") or "")
    if not v:
        try:
            from importlib import metadata

            v = metadata.version("sirius")
        except Exception:  # noqa: BLE001 - a build tree on PYTHONPATH has no distribution
            v = "unknown"
    return v


def _gpu(ext: Any, device: Optional[str]) -> Any:
    """The CUDA device a decode named `device` ("cuda", "cuda:1", "auto"
    (the GPU when there is one), "cpu") runs on with nvTIFF, or None for the
    CPU decoder: a CPU device, a build without nvTIFF, or no such GPU."""
    text = str(device or "auto").strip().lower()
    if text != "auto" and not text.startswith("cuda"):
        return None
    try:
        if not (ext.built_with_nvtiff() and ext.cuda_available()):
            return None
        index = int(text.split(":", 1)[1]) if text.startswith("cuda:") else 0
        if not 0 <= index < int(ext.cuda_device_count()):
            return None
        return ext.Device.cuda(index)
    except Exception:  # noqa: BLE001 - an odd device name or a driver that will not say
        return None


def tiff_reader(device: Optional[str] = "auto") -> Dict[str, Any]:
    """hello's "tiff_reader": {"sirius": the package's version, or None when
    TIFF cannot be read here; "nvtiff": whether a decode on `device` runs on
    the GPU}."""
    ext = _sirius()
    if ext is None:
        return {"sirius": None, "nvtiff": False}
    return {"sirius": _sirius_version(ext), "nvtiff": _gpu(ext, device) is not None}


def _host(a: Any) -> np.ndarray:
    """A read's pixels on the host: numpy from a CPU decode; a sirius.Buffer
    (GPU memory) from an nvTIFF decode, copied home."""
    if isinstance(a, np.ndarray):
        return a
    to_numpy = getattr(a, "numpy", None)
    return np.asarray(to_numpy() if callable(to_numpy) else a)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# --- the dataset -----------------------------------------------------------------------------------

class Dataset:
    """One opened dataset; planes and volumes in the file's own dtype.

    The reads take the decode `device` ("cpu", "cuda", "cuda:N" or "auto"):
    a TIFF is decoded with nvTIFF on that GPU when it is one the sirius
    package can decode on, else on the CPU; what comes back is on the host."""

    def __init__(self, path: str, options: Optional[Dict[str, Any]] = None) -> None:
        self.path = os.path.expanduser(path)
        self.options = dict(options or {})
        self.key = (os.path.abspath(self.path), _options_key(options))
        self._tf: Any = None        # sirius.TiffFile of a TIFF
        self._ext: Any = None       # the sirius module that opened it
        self._array = None          # (c, t, z, y, x) view (memmap) of a .npy
        self._order = "czt"         # a TIFF's page order, fastest axis first
        self._samples = 1           # a TIFF's samples (channels) per page
        self._page_c = 1            # a TIFF's channels in pages (c // samples)
        self.rgb = False
        self._levels: List[Tuple[int, int, int]] = []   # a TIFF's pyramid: (width, height, pages) per level
        self._gpu_ok: Dict[str, bool] = {}              # device -> nvTIFF can decode this file there
        self._paged_regions = True  # the extension reads a region of chosen pages (read_region first / count)
        if not os.path.exists(self.path):
            raise DatasetError(f"{path}: no such file on {socket.gethostname()}")
        if os.path.isdir(self.path):
            raise DatasetError(f"{path} is a folder: open a TIFF or .npy file inside it (folder datasets and zarr stores "
                               "are not read on the cluster yet)")
        lower = self.path.lower()
        self.bytes_on_disk = os.path.getsize(self.path)
        self.voxel = [0.0, 0.0, 0.0]
        self.channels: List[Dict[str, Any]] = []
        self.frame_interval = 0.0
        if lower.endswith(".npy"):
            self._open_npy()
        elif lower.endswith((".tif", ".tiff", ".btf", ".tf8")):
            self._open_tiff()
        else:
            raise DatasetError(f"{os.path.basename(path)}: only TIFF / OME-TIFF and .npy are read on the cluster")

    # --- opening ---------------------------------------------------------------------------------

    def _open_npy(self) -> None:
        a = np.load(self.path, mmap_mode="r")
        if a.ndim == 2:
            a = a[None, None, None]
        elif a.ndim == 3:
            a = a[None, None]
        elif a.ndim == 4:
            a = a[:, None]
        elif a.ndim != 5:
            raise DatasetError(f"{os.path.basename(self.path)}: a {a.ndim}-D array; 2 to 5 dimensions (c, t, z, y, x) are read")
        self._array = a
        self.dtype = a.dtype
        self.c, self.t, self.z, self.y, self.x = (int(v) for v in a.shape)
        self.format = "npy"
        self.dims_from_metadata = a.ndim > 3

    def _open_tiff(self) -> None:
        ext = _sirius()
        if ext is None:
            raise DatasetError(TIFF_NEEDS_SIRIUS)
        from .steps import workbench

        wb = workbench()
        if not hasattr(wb, "tiff_dims"):
            raise DatasetError("the sirius package in the worker's Python is older than this worker: reinstall it "
                               "from the checkout (pip install <checkout> in the worker's venv)")
        name = os.path.basename(self.path)
        try:
            tf = ext.TiffFile(self.path)
            info = tf.info
            uniform = bool(info.uniform_pages)
        except Exception as e:  # noqa: BLE001 - libtiff's own words: not a TIFF, a codec it lacks ...
            raise DatasetError(f"{name}: {e}") from e
        if not uniform:
            raise DatasetError(f"{name}: the pages differ in size or pixel type")
        page0 = info.page(0)
        if not getattr(page0, "decodable", True):
            raise DatasetError(f"{name}: {page0.unsupported}")
        spp = max(int(getattr(page0, "samples_per_pixel", 1) or 1), 1)
        self._ext, self._tf = ext, tf
        pages = int(info.page_count)
        self.y, self.x = int(info.height), int(info.width)
        self.dtype = np.dtype(info.dtype)
        self._levels = [(int(lv.width), int(lv.height), len(lv.ifds)) for lv in info.levels]
        tags = wb.tiff_tags(page0) or {"description": "", "xres": 0.0, "yres": 0.0, "res_unit": 2}
        # the C++ reader's OME / ImageJ parser (the application's), or the
        # Python port of it in an older extension
        md = wb.tiff_metadata(tags["description"]) if hasattr(wb, "tiff_metadata") \
            else wb._parse_tiff_description(tags["description"])
        page_md = md
        if spp > 1 and md["ome"] and md["c"] >= spp and md["c"] % spp == 0:
            page_md = dict(md, c=md["c"] // spp)   # OME's SizeC counts samples
        # the application sends a page order exactly when it has one
        # (remote_source.cpp remoteOptionsJson), and then it wins over the
        # metadata; its channel count is the dataset's (samples included)
        o = self.options
        given = o.get("page_order") is not None
        oc = o.get("c")
        if spp > 1 and oc and int(oc) >= spp and int(oc) % spp == 0:
            oc = int(oc) // spp
        page_c, self.t, self.z, self._order, from_meta = wb.tiff_dims(
            pages, page_md, given, str(o.get("page_order") or "czt"), oc, o.get("t"), o.get("z"))
        self._samples, self._page_c = spp, page_c
        self.c = page_c * spp
        self.rgb = spp == 3 and page_c == 1 and int(getattr(page0, "photometric", 1)) == 2
        self.dims_from_metadata = bool(from_meta)
        self.format = "ome-tiff" if md["ome"] else ("imagej-tiff" if md["imagej"] else "tiff")
        self.voxel = wb.tiff_voxel(md, tags["xres"], tags["yres"], tags["res_unit"])
        self.frame_interval = max(float(md["frame_interval_s"] or 0.0), 0.0)
        if md["ome"] and spp == 1:
            for ch in md["channels"]:
                entry: Dict[str, Any] = {"name": str(ch.get("label", ""))}
                if float(ch.get("wavelength_nm", 0.0) or 0.0) > 0.0:
                    entry["wavelength_nm"] = float(ch["wavelength_nm"])
                self.channels.append(entry)

    def close(self) -> None:
        """Lets the file go."""
        self._array, self._tf = None, None

    # --- facts -------------------------------------------------------------------------------------

    def meta(self) -> Dict[str, Any]:
        name = os.path.basename(self.path)
        for ext in (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".npy"):
            if name.lower().endswith(ext):
                name = name[: -len(ext)]
                break
        return {
            "name": name,
            "path": os.path.abspath(self.path),
            "format": self.format,
            "dims": [self.c, self.t, self.z, self.y, self.x],
            "dtype": str(np.dtype(self.dtype).name),
            "bytes": int(self.bytes_on_disk),
            "voxel_um": [float(v) for v in self.voxel],
            "frame_interval_s": float(self.frame_interval),
            "channels": self.channels,
            "rgb": bool(self.rgb),
            "dims_from_metadata": bool(self.dims_from_metadata),
        }

    def _check(self, c: int, t: int, z: Optional[int] = None) -> None:
        if not (0 <= c < self.c and 0 <= t < self.t) or (z is not None and not 0 <= z < self.z):
            where = f"c {c}, t {t}" + ("" if z is None else f", z {z}")
            raise DatasetError(f"{where} is outside the dataset ({self.c} channels, {self.t} time points, {self.z} planes)")

    # --- reading a TIFF (sirius.TiffFile) --------------------------------------------------------

    def _page_of(self, c: int, t: int, z: int) -> int:
        return _plane_of(self._order, c // self._samples, t, z, self._page_c, self.t, self.z)

    def _sample_args(self, c: int) -> Dict[str, int]:
        """A read's keywords for channel c: one sample of a multi-sample page
        (none for one-sample files, which any extension reads)."""
        return {"first_sample": c % self._samples, "samples": 1} if self._samples > 1 else {}

    def _device(self, device: Optional[str]) -> Any:
        """sirius.Device of a decode: the GPU when nvTIFF decodes this file
        there (TiffFile.gpu_decodable), else the CPU."""
        ext = self._ext
        gpu = _gpu(ext, device)
        if gpu is None:
            return ext.Device.cpu()
        key = str(device or "auto").strip().lower()
        ok = self._gpu_ok.get(key)
        if ok is None:
            try:
                ok = bool(self._tf.gpu_decodable(gpu)[0])
            except Exception:  # noqa: BLE001 - say no and decode on the CPU
                ok = False
            self._gpu_ok[key] = ok
        return gpu if ok else ext.Device.cpu()

    def _pages(self, first: int, count: int, device: Optional[str], c: int = 0) -> np.ndarray:
        """Pages [first, first + count) of channel c's sample as (count, y, x)."""
        out = self._tf.read_pages(int(first), int(count), device=self._device(device), allow_cpu_fallback=True,
                                  **self._sample_args(c))
        return _host(out).reshape(int(count), self.y, self.x)

    def _region(self, page: int, x: int, y: int, w: int, h: int, level: int,
                device: Optional[str], c: int = 0) -> Optional[np.ndarray]:
        """(h, w) at (x, y) of one page (channel c's sample) at pyramid
        `level`, decoding only the tiles / strips it covers; None when this
        extension cannot read a region of one page there (built before
        read_region took first / count)."""
        if self._paged_regions:
            try:
                out = self._tf.read_region(int(x), int(y), int(w), int(h), level=int(level), device=self._device(device),
                                           allow_cpu_fallback=True, first=int(page), count=1, **self._sample_args(c))
                return _host(out).reshape(int(h), int(w))
            except TypeError:
                self._paged_regions = False   # an older extension: whole pages, cropped here
        if level != 0:
            return None
        return self._pages(page, 1, device, c)[0, y:y + h, x:x + w]

    def _level_for(self, factor: int) -> Optional[Tuple[int, int]]:
        """(level, scale) of the most reduced pyramid level whose integer
        scale divides `factor` and that holds every page, or None."""
        best = None
        pages = self._levels[0][2] if self._levels else 0
        for k, (w, h, n) in enumerate(self._levels[1:], 1):
            if w <= 0 or h <= 0 or n != pages:
                continue
            s = int(round(self.x / w))
            if s < 2 or factor % s or w not in (self.x // s, _ceil_div(self.x, s)) \
                    or h not in (self.y // s, _ceil_div(self.y, s)):
                continue
            if best is None or s > best[1]:
                best = (k, s)
        return best

    def _tiff_xy(self, c: int, t: int, z: int, factor: int, box: Tuple[int, int, int, int],
                 device: Optional[str]) -> np.ndarray:
        """The XY view of a TIFF plane: the region `box` (x0, y0, x1, y1)
        reduced by `factor`, from a pyramid level when one fits (the
        writer's reduction, not this worker's block mean), else from the
        region's own pixels at full resolution."""
        x0, y0, x1, y1 = box
        page = self._page_of(c, t, z)
        lv = self._level_for(factor) if factor > 1 else None
        if lv is not None and x0 % lv[1] == 0 and y0 % lv[1] == 0:
            k, s = lv
            w, h, _ = self._levels[k]
            lx0, ly0 = x0 // s, y0 // s
            lx1, ly1 = min(_ceil_div(x1, s), w), min(_ceil_div(y1, s), h)
            part = self._region(page, lx0, ly0, lx1 - lx0, ly1 - ly0, k, device, c) if lx1 > lx0 and ly1 > ly0 else None
            if part is not None:
                out = reduce_blocks(part, (factor // s, factor // s))
                if out.shape == (_ceil_div(y1 - y0, factor), _ceil_div(x1 - x0, factor)):
                    return out
        if (x0, y0, x1, y1) == (0, 0, self.x, self.y):
            full = self._pages(page, 1, device, c)[0]
        else:
            full = self._region(page, x0, y0, x1 - x0, y1 - y0, 0, device, c)
        return reduce_blocks(full, (factor, factor))

    # --- reading (the file's dtype) ------------------------------------------------------------------

    def plane(self, c: int, t: int, z: int, device: Optional[str] = "auto") -> np.ndarray:
        self._check(c, t, z)
        if self._array is not None:
            return np.asarray(self._array[c, t, z])
        cached = _VOLUMES.get((self.key, c, t))
        if cached is not None:
            return cached[z]
        return self._pages(self._page_of(c, t, z), 1, device, c)[0]

    def volume(self, c: int, t: int, device: Optional[str] = "auto") -> np.ndarray:
        self._check(c, t)
        key = (self.key, c, t)
        cached = _VOLUMES.get(key)
        if cached is not None:
            return cached
        if self._array is not None:
            vol = np.ascontiguousarray(self._array[c, t])
        else:
            vol = self._tiff_volume(c, t, device)
        _VOLUMES.put(key, vol)
        return vol

    def _tiff_volume(self, c: int, t: int, device: Optional[str]) -> np.ndarray:
        """The z planes of (c, t): one page range when z is the fastest axis
        (or the only one); when the pages of other axes sit between them,
        one range of at most 4x the volume (and 1 GiB) sliced every
        stride-th page, else page by page."""
        first = self._page_of(c, t, 0)
        stride = self._page_of(c, t, 1) - first if self.z > 1 else 1
        if stride == 1:
            return np.ascontiguousarray(self._pages(first, self.z, device, c))
        span = (self.z - 1) * stride + 1
        plane_bytes = self.y * self.x * np.dtype(self.dtype).itemsize
        if stride <= 4 and span * plane_bytes <= (1 << 30):
            return np.ascontiguousarray(self._pages(first, span, device, c)[::stride])
        vol = np.empty((self.z, self.y, self.x), dtype=self.dtype)
        for z in range(self.z):
            vol[z] = self._pages(first + z * stride, 1, device, c)[0]
        return vol

    # --- what a pane draws -------------------------------------------------------------------------

    def view(self, kind: str, c: int, t: int, index: int = 0, factor: int = 1,
             region: Optional[Sequence[int]] = None, max_side: int = 256,
             device: Optional[str] = "auto") -> np.ndarray:
        """kind xy: the (y, x) plane at z = index; xz: rows z, columns x at y = index;
        yz: rows y, columns z at x = index; mip: the z maximum projection (y, x);
        volume: the (z, y, x) volume reduced to a longest side of `max_side`.
        `region` is (x, y, w, h) in the view's own columns and rows."""
        factor = max(int(factor), 1)
        kind = kind.lower()
        if kind == "volume":
            vol = self.volume(c, t, device)
            f = tuple(max(1, -(-n // max(int(max_side), 1))) for n in vol.shape)
            return reduce_blocks(vol, f)
        if kind == "xy":
            self._check(c, t, index)
            cached = _VOLUMES.get((self.key, c, t))
            if cached is None and self._tf is not None:
                # a TIFF plane not in memory: read only what the pane shows
                return self._tiff_xy(c, t, index, factor, _box(region, self.x, self.y), device)
            full = cached[index] if cached is not None else self.plane(c, t, index, device)
        elif kind == "xz":
            self._check(c, t)
            full = self.volume(c, t, device)[:, min(max(int(index), 0), self.y - 1), :]
        elif kind == "yz":
            self._check(c, t)
            full = self.volume(c, t, device)[:, :, min(max(int(index), 0), self.x - 1)].T
        elif kind == "mip":
            self._check(c, t)
            key = (self.key, c, t, "mip")
            full = _VOLUMES.get(key)
            if full is None:
                full = np.max(self.volume(c, t, device), axis=0)
                _VOLUMES.put(key, full)
        else:
            raise DatasetError(f"unknown view '{kind}': xy, xz, yz, mip or volume")
        rows, cols = full.shape
        x0, y0, x1, y1 = _box(region, cols, rows)
        return reduce_blocks(full[y0:y1, x0:x1], (factor, factor))

    def stats(self, c: int, t: int, device: Optional[str] = "auto") -> Dict[str, float]:
        """A display window from a few planes spread over z: the 0.1 / 99.9
        percentiles (what the viewer's Auto window is) and the range."""
        self._check(c, t)
        n = min(self.z, 5)
        zs = [0] if n == 1 else [k * (self.z - 1) // (n - 1) for k in range(n)]
        samples = []
        for z in zs:
            p = self.plane(c, t, z, device).ravel()
            stride = max(1, p.size // (1 << 16))
            samples.append(p[::stride].astype(np.float32))
        s = np.concatenate(samples) if samples else np.zeros(1, np.float32)
        s = s[np.isfinite(s)]
        if s.size == 0:
            return {"lo": 0.0, "hi": 1.0, "min": 0.0, "max": 1.0}
        lo, hi = (float(v) for v in np.percentile(s, [0.1, 99.9]))
        mn, mx = float(s.min()), float(s.max())
        if hi <= lo:
            lo, hi = mn, (mx if mx > mn else mn + 1.0)
        return {"lo": lo, "hi": hi, "min": mn, "max": mx}


def _box(region: Optional[Sequence[int]], cols: int, rows: int) -> Tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of `region` (x, y, w, h) clipped to a cols x rows
    view; the whole view without one."""
    if region is None or len(region) != 4:
        return 0, 0, cols, rows
    x0, y0, w, h = (int(v) for v in region)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x0 + max(w, 0), cols), min(y0 + max(h, 0), rows)
    if x1 <= x0 or y1 <= y0:
        raise DatasetError(f"region {list(region)} is outside the {cols} x {rows} view")
    return x0, y0, x1, y1


def reduce_blocks(a: np.ndarray, factors: Sequence[int]) -> np.ndarray:
    """The mean of every block of `factors` (one per axis; a partial block at
    the end of an axis is the mean of what it has), in a's dtype: integers are
    rounded. A factor of 1 everywhere returns `a` itself (contiguous)."""
    factors = [max(int(f), 1) for f in factors]
    if all(f == 1 for f in factors):
        return np.ascontiguousarray(a)
    work = np.float64 if a.dtype.itemsize >= 4 and a.dtype.kind == "f" else np.float32
    acc = a
    counts = None
    # the largest factor first, summed straight from the source dtype: the
    # first pass shrinks the array most and no float copy of the whole is made
    for axis in sorted(range(len(factors)), key=lambda k: -factors[k]):
        f = factors[axis]
        if f == 1:
            continue
        n = acc.shape[axis]
        starts = np.arange(0, n, f)
        whole = n - n % f
        if whole:
            # whole blocks: a reshaped sum, many times faster than reduceat on a middle axis
            head = np.take(acc, np.arange(whole), axis=axis) if whole < n else acc
            shape = acc.shape[:axis] + (whole // f, f) + acc.shape[axis + 1:]
            summed = head.reshape(shape).sum(axis=axis + 1, dtype=work)
            if whole < n:
                tail = np.take(acc, np.arange(whole, n), axis=axis).sum(axis=axis, keepdims=True, dtype=work)
                summed = np.concatenate([summed, tail], axis=axis)
            acc = summed
        else:
            acc = np.add.reduceat(acc, starts, axis=axis, dtype=work)
        sizes = np.minimum(starts + f, n) - starts
        shape = [1] * acc.ndim
        shape[axis] = len(sizes)
        sizes = sizes.reshape(shape).astype(acc.dtype)
        counts = sizes if counts is None else counts * sizes
    mean = acc / counts
    if a.dtype.kind in "iub":
        info = np.iinfo(a.dtype) if a.dtype.kind != "b" else None
        mean = np.rint(mean)
        if info is not None:
            mean = np.clip(mean, info.min, info.max)
        return np.ascontiguousarray(mean.astype(a.dtype))
    return np.ascontiguousarray(mean.astype(np.float32 if a.dtype.itemsize <= 4 else a.dtype))


# --- the cache ---------------------------------------------------------------------------------

class _ByteLru:
    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.used = 0
        self._items: collections.OrderedDict[Any, np.ndarray] = collections.OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            v = self._items.get(key)
            if v is not None:
                self._items.move_to_end(key)
            return v

    def put(self, key, value: np.ndarray) -> None:
        if value.nbytes > self.budget:
            return
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self.used -= old.nbytes
            self._items[key] = value
            self.used += value.nbytes
            while self.used > self.budget and self._items:
                _, gone = self._items.popitem(last=False)
                self.used -= gone.nbytes

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self.used = 0


def _cache_budget() -> int:
    try:
        return max(0, int(os.environ.get("SIRIUS_WORKER_VIEW_CACHE_MB", "4096"))) << 20
    except ValueError:
        return 4096 << 20


_VOLUMES = _ByteLru(_cache_budget())
_OPEN: collections.OrderedDict[Tuple, Tuple[float, Dataset]] = collections.OrderedDict()
_OPEN_LOCK = threading.Lock()


def open_dataset(path: str, options: Optional[Dict[str, Any]] = None) -> Dataset:
    """The dataset at `path` (~ expanded), opened once and kept while the file is unchanged."""
    if not path:
        raise DatasetError("no path")
    full = os.path.abspath(os.path.expanduser(path))
    key = (full, _options_key(options))
    try:
        stamp = os.path.getmtime(full)
    except OSError as e:
        raise DatasetError(f"{path}: {e.strerror or e}") from e
    with _OPEN_LOCK:
        hit = _OPEN.get(key)
        if hit is not None and hit[0] == stamp:
            _OPEN.move_to_end(key)
            return hit[1]
    ds = Dataset(full, options)
    with _OPEN_LOCK:
        _OPEN[key] = (stamp, ds)
        while len(_OPEN) > 8:
            _OPEN.popitem(last=False)[1][1].close()
    return ds


def forget_all() -> None:
    """Closes every dataset opened and drops the cached volumes."""
    _VOLUMES.clear()
    with _OPEN_LOCK:
        opened = [ds for _, ds in _OPEN.values()]
        _OPEN.clear()
    for ds in opened:
        ds.close()


def _indices(value: Any, extent: int, axis: str) -> List[int]:
    """The channel or time indices a request names: all of them (None), one,
    or a list -- each within the dataset and none twice, so a request cannot
    size the output by repeating an index (a list of a billion zeros used to
    be one np.empty of a billion volumes)."""
    if value is None:
        return list(range(extent))
    raw = [value] if isinstance(value, int) else value
    if not isinstance(raw, (list, tuple)) or len(raw) > max(extent, 1):
        raise DatasetError(f"{axis}: expected an index or a list of at most {extent} indices")
    out: List[int] = []
    for v in raw:
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < extent:
            raise DatasetError(f"{axis} index {v!r} is outside the dataset (0..{extent - 1})")
        if v in out:
            raise DatasetError(f"{axis} index {v} is named twice")
        out.append(v)
    return out


def read_ref(ref: Dict[str, Any], device: Optional[str] = "auto") -> np.ndarray:
    """A step's input named by reference instead of sent (the HPC backend
    with a cluster dataset): {path, options, c, t} -> the (z, y, x) volume as
    float32; with c / t lists (or absent: all), layout "ctzyx" -> (c, t, z, y, x).
    A TIFF is decoded on `device` (see Dataset)."""
    ds = open_dataset(str(ref.get("path", "")), ref.get("options") or {})
    layout = str(ref.get("layout", "zyx"))
    if layout == "zyx":
        return np.ascontiguousarray(ds.volume(int(ref.get("c", 0)), int(ref.get("t", 0)), device), dtype=np.float32)
    cs = _indices(ref.get("c"), ds.c, "c")
    ts = _indices(ref.get("t"), ds.t, "t")
    out = np.empty((len(cs), len(ts), ds.z, ds.y, ds.x), dtype=np.float32)
    for i, c in enumerate(cs):
        for j, t in enumerate(ts):
            out[i, j] = ds.volume(c, t, device)
    return out


# --- the wire --------------------------------------------------------------------------------

def _zstd():
    try:
        import zstandard  # type: ignore

        return zstandard
    except ImportError:
        return None


def available_encodings() -> List[str]:
    """What this worker can send, best first."""
    out = []
    if _zstd() is not None:
        out.append("zstd")
    out.append("zlib")
    return out


def encode(a: np.ndarray, accept: Sequence[str]) -> Tuple[Dict[str, Any], np.ndarray]:
    """(description, tensor) for `a`: the array itself when the client
    accepts no encoding this worker has, or (when that is smaller) its bytes
    shuffled and compressed as a uint8 tensor. The description says how:
    {"encoding": "zstd" | "zlib" | "raw", "shuffle": bool, "dtype", "shape", "raw_bytes"}."""
    a = np.ascontiguousarray(a)
    if a.dtype.byteorder == ">":
        a = a.astype(a.dtype.newbyteorder("<"))
    desc: Dict[str, Any] = {"dtype": str(a.dtype.name), "shape": [int(v) for v in a.shape], "raw_bytes": int(a.nbytes),
                            "encoding": "raw", "shuffle": False}
    wanted = [e for e in available_encodings() if e in set(accept or [])]
    if not wanted or a.nbytes < 4096:
        return desc, a
    raw = a.view(np.uint8)
    shuffle = a.dtype.itemsize > 1
    if shuffle:
        raw = np.ascontiguousarray(raw.reshape(-1, a.dtype.itemsize).T)
    data = raw.tobytes()
    if wanted[0] == "zstd":
        packed = _zstd().ZstdCompressor(level=3).compress(data)
    else:
        packed = zlib.compress(data, 1)
    if len(packed) >= a.nbytes:
        return desc, a
    desc.update({"encoding": wanted[0], "shuffle": shuffle})
    return desc, np.frombuffer(packed, dtype=np.uint8)


def decode(desc: Dict[str, Any], tensor: np.ndarray) -> np.ndarray:
    """encode()'s inverse (tests; the application's is app/core/remote_source.cpp)."""
    if desc.get("encoding", "raw") == "raw":
        return tensor
    data = bytes(tensor)
    dtype = np.dtype(desc["dtype"])
    # What the shape says the bytes are, and not one byte more: a stream that
    # inflates past it (a decompression bomb) is refused, not followed.
    expected = dtype.itemsize
    for n in desc["shape"]:
        expected *= int(n)
    if int(min(desc["shape"], default=0)) < 0 or expected > (64 << 30):
        raise DatasetError(f"an array of {expected} bytes is not plausible")
    if desc["encoding"] == "zstd":
        data = _zstd().ZstdDecompressor().decompress(data, max_output_size=max(expected, 1))
    else:
        inflater = zlib.decompressobj()
        data = inflater.decompress(data, max(expected, 1))   # 0 would mean "no limit"
        if inflater.unconsumed_tail or not inflater.eof:
            raise DatasetError("a compressed array inflates past its shape")
    if len(data) != expected:
        raise DatasetError(f"a compressed array of {len(data)} bytes does not match its shape ({expected} bytes)")
    raw = np.frombuffer(data, dtype=np.uint8)
    if desc.get("shuffle"):
        raw = np.ascontiguousarray(raw.reshape(dtype.itemsize, -1).T)
    return raw.view(dtype).reshape(desc["shape"])
