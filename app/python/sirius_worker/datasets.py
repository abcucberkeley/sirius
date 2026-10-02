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

Readers: TIFF, OME-TIFF and ImageJ hyperstacks through ``tifffile`` (an
optional package of the worker: ``pip install tifffile``), and ``.npy``
arrays. The (c, t) volumes read most recently are kept in memory, up to
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
import re
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
    return (str(o.get("page_order", "") or ""), int(o.get("c", 0) or 0), int(o.get("t", 0) or 0), int(o.get("z", 0) or 0))


# --- metadata ----------------------------------------------------------------------------------

_OME_SIZE = re.compile(r'PhysicalSize([XYZ])="([0-9.eE+-]+)"')
_OME_CHANNEL = re.compile(r"<(?:\w+:)?Channel\b([^>]*)>")
_ATTR = re.compile(r'(\w+)="([^"]*)"')


def _ome_voxel_and_channels(xml: str) -> Tuple[List[float], List[Dict[str, Any]]]:
    voxel = [0.0, 0.0, 0.0]
    for axis, value in _OME_SIZE.findall(xml or ""):
        try:
            voxel["XYZ".index(axis)] = float(value)
        except ValueError:
            pass
    channels = []
    for attrs in _OME_CHANNEL.findall(xml or ""):
        a = dict(_ATTR.findall(attrs))
        ch: Dict[str, Any] = {"name": a.get("Name", "")}
        try:
            if a.get("EmissionWavelength"):
                ch["wavelength_nm"] = float(a["EmissionWavelength"])
        except ValueError:
            pass
        channels.append(ch)
    return voxel, channels


def _imagej_voxel(tf: Any, ij: Dict[str, Any]) -> List[float]:
    voxel = [0.0, 0.0, 0.0]
    try:
        page = tf.pages[0]
        xr = page.tags.get("XResolution")
        yr = page.tags.get("YResolution")
        if xr is not None and xr.value[0]:
            voxel[0] = float(xr.value[1]) / float(xr.value[0])
        if yr is not None and yr.value[0]:
            voxel[1] = float(yr.value[1]) / float(yr.value[0])
    except Exception:  # noqa: BLE001 - a missing or odd tag only loses the voxel size
        pass
    try:
        voxel[2] = float(ij.get("spacing", 0.0) or 0.0)
    except (TypeError, ValueError):
        pass
    return voxel


# --- the dataset -----------------------------------------------------------------------------------

class Dataset:
    """One opened dataset; planes and volumes in the file's own dtype."""

    def __init__(self, path: str, options: Optional[Dict[str, Any]] = None) -> None:
        self.path = os.path.expanduser(path)
        self.options = dict(options or {})
        self.key = (os.path.abspath(self.path), _options_key(options))
        self._lock = threading.Lock()
        self._tf = None
        self._pages = None          # sequence of pages, or None for an in-memory / memmapped array
        self._array = None          # (c, t, z, y, x) view (memmap or ndarray) when one exists
        self._rgb = False
        self._axes_map: Optional[Tuple[str, Tuple[int, ...]]] = None
        self._plain = True
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
        try:
            import tifffile  # type: ignore
        except ImportError as e:
            raise DatasetError("reading TIFF on the cluster needs tifffile in the worker's Python: "
                               "pip install tifffile") from e
        tf = tifffile.TiffFile(self.path)
        self._tf = tf
        series = tf.series[0]
        axes = str(series.axes)
        shape = tuple(int(v) for v in series.shape)
        self.dtype = np.dtype(series.dtype)
        self.format = "ome-tiff" if tf.is_ome else ("imagej-tiff" if tf.is_imagej else "tiff")
        if tf.is_ome:
            self.voxel, self.channels = _ome_voxel_and_channels(tf.ome_metadata or "")
        elif tf.is_imagej:
            ij = tf.imagej_metadata or {}
            self.voxel = _imagej_voxel(tf, ij)
            try:
                self.frame_interval = float(ij.get("finterval", 0.0) or 0.0)
            except (TypeError, ValueError):
                self.frame_interval = 0.0
        self._samples_last = True
        if "S" in axes:
            # samples (RGB, or planar channels): the sample axis is the channel axis
            i = axes.index("S")
            self._rgb = True
            self._samples_last = i == len(axes) - 1
            self._samples = shape[i]
            axes, shape = axes[:i] + axes[i + 1:], shape[:i] + shape[i + 1:]
        if not axes.endswith("YX"):
            raise DatasetError(f"{os.path.basename(self.path)}: axes {series.axes} do not end in YX")
        self.y, self.x = shape[-2], shape[-1]
        outer_axes, outer_shape = axes[:-2], shape[:-2]
        # the planes, in the order the file stores them
        contiguous = series.dataoffset is not None and not self._rgb
        if contiguous:
            n = int(np.prod(outer_shape)) if outer_shape else 1
            mm = np.memmap(self.path, dtype=series.dtype.newbyteorder(tf.byteorder), mode="r",
                           offset=series.dataoffset, shape=(n, self.y, self.x))
            self._pages = mm
        else:
            self._pages = series.pages
        explicit = set(outer_axes) & set("CTZ")
        o = self.options
        if self._rgb:
            # one page per plane, its samples the channels
            self._plain = False
            pages = int(np.prod(outer_shape)) if outer_shape else 1
            self.c, self.t, self.z = int(self._samples), 1, pages
            self._axes_map = ("rgb", outer_shape)
            self.dims_from_metadata = True
        elif explicit and len(explicit) == len(outer_axes) and not any(o.get(k) for k in ("c", "t", "z")) \
                and not o.get("page_order"):
            # every outer axis is C, T or Z (OME / ImageJ hyperstacks): the file says
            self._plain = False
            sizes = dict(zip(outer_axes, outer_shape))
            self.c, self.t, self.z = sizes.get("C", 1), sizes.get("T", 1), sizes.get("Z", 1)
            self._axes_map = (outer_axes, outer_shape)
            self.dims_from_metadata = True
        else:
            # plain pages, mapped by the page order the application sends
            pages = int(np.prod(outer_shape)) if outer_shape else 1
            order = str(o.get("page_order", "") or "czt")
            c, t = max(int(o.get("c", 0) or 0), 1), max(int(o.get("t", 0) or 0), 1)
            z = int(o.get("z", 0) or 0)
            if z <= 0:
                z = pages // (c * t) if pages % (c * t) == 0 else 0
            if z <= 0 or c * t * z > pages:
                c, t, z = 1, 1, pages   # a layout the pages do not divide into: read them as z
            self.c, self.t, self.z = c, t, z
            self._order = order
            self._plain = True
            self.dims_from_metadata = False

    def close(self) -> None:
        """Lets the file go (Windows will not delete a file a memmap or a TiffFile holds)."""
        pages, self._pages, self._array = self._pages, None, None
        mm = getattr(pages, "_mmap", None)
        if mm is not None:
            try:
                mm.close()
            except (BufferError, ValueError):
                pass
        if self._tf is not None:
            self._tf.close()
            self._tf = None

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
            "rgb": bool(self._rgb),
            "dims_from_metadata": bool(self.dims_from_metadata),
        }

    def _check(self, c: int, t: int, z: Optional[int] = None) -> None:
        if not (0 <= c < self.c and 0 <= t < self.t) or (z is not None and not 0 <= z < self.z):
            where = f"c {c}, t {t}" + ("" if z is None else f", z {z}")
            raise DatasetError(f"{where} is outside the dataset ({self.c} channels, {self.t} time points, {self.z} planes)")

    # --- reading (the file's dtype) ------------------------------------------------------------------

    def _page(self, index: int) -> np.ndarray:
        pages = self._pages
        if isinstance(pages, np.ndarray):
            return np.asarray(pages[index])
        with self._lock:   # a TiffFile handle is not safe across threads
            return np.asarray(pages[index].asarray())

    def plane(self, c: int, t: int, z: int) -> np.ndarray:
        self._check(c, t, z)
        if self._array is not None:
            return np.asarray(self._array[c, t, z])
        if self._axes_map is not None and self._axes_map[0] == "rgb":
            page = self._page(z)
            return np.asarray(page[..., c] if self._samples_last else page[c])
        if not self._plain:
            axes, shape = self._axes_map
            index = {"C": c, "T": t, "Z": z}
            flat = 0
            for a, n in zip(axes, shape):
                flat = flat * n + index.get(a, 0)
            return self._page(flat)
        return self._page(_plane_of(self._order, c, t, z, self.c, self.t, self.z))

    def volume(self, c: int, t: int) -> np.ndarray:
        self._check(c, t)
        key = (self.key, c, t)
        cached = _VOLUMES.get(key)
        if cached is not None:
            return cached
        if self._array is not None:
            vol = np.ascontiguousarray(self._array[c, t])
        else:
            vol = np.empty((self.z, self.y, self.x), dtype=self.dtype)
            for z in range(self.z):
                vol[z] = self.plane(c, t, z)
        _VOLUMES.put(key, vol)
        return vol

    # --- what a pane draws -------------------------------------------------------------------------

    def view(self, kind: str, c: int, t: int, index: int = 0, factor: int = 1,
             region: Optional[Sequence[int]] = None, max_side: int = 256) -> np.ndarray:
        """kind xy: the (y, x) plane at z = index; xz: rows z, columns x at y = index;
        yz: rows y, columns z at x = index; mip: the z maximum projection (y, x);
        volume: the (z, y, x) volume reduced to a longest side of `max_side`.
        `region` is (x, y, w, h) in the view's own columns and rows."""
        factor = max(int(factor), 1)
        kind = kind.lower()
        if kind == "volume":
            vol = self.volume(c, t)
            f = tuple(max(1, -(-n // max(int(max_side), 1))) for n in vol.shape)
            return reduce_blocks(vol, f)
        if kind == "xy":
            self._check(c, t, index)
            cached = _VOLUMES.get((self.key, c, t))
            full = cached[index] if cached is not None else self.plane(c, t, index)
        elif kind == "xz":
            self._check(c, t)
            full = self.volume(c, t)[:, min(max(int(index), 0), self.y - 1), :]
        elif kind == "yz":
            self._check(c, t)
            full = self.volume(c, t)[:, :, min(max(int(index), 0), self.x - 1)].T
        elif kind == "mip":
            self._check(c, t)
            key = (self.key, c, t, "mip")
            full = _VOLUMES.get(key)
            if full is None:
                full = np.max(self.volume(c, t), axis=0)
                _VOLUMES.put(key, full)
        else:
            raise DatasetError(f"unknown view '{kind}': xy, xz, yz, mip or volume")
        rows, cols = full.shape
        if region is not None and len(region) == 4:
            x0, y0, w, h = (int(v) for v in region)
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x0 + max(w, 0), cols), min(y0 + max(h, 0), rows)
            if x1 <= x0 or y1 <= y0:
                raise DatasetError(f"region {list(region)} is outside the {cols} x {rows} view")
            full = full[y0:y1, x0:x1]
        return reduce_blocks(full, (factor, factor))

    def stats(self, c: int, t: int) -> Dict[str, float]:
        """A display window from a few planes spread over z: the 0.1 / 99.9
        percentiles (what the viewer's Auto window is) and the range."""
        self._check(c, t)
        n = min(self.z, 5)
        zs = [0] if n == 1 else [k * (self.z - 1) // (n - 1) for k in range(n)]
        samples = []
        for z in zs:
            p = self.plane(c, t, z).ravel()
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


def read_ref(ref: Dict[str, Any]) -> np.ndarray:
    """A step's input named by reference instead of sent (the HPC backend
    with a cluster dataset): {path, options, c, t} -> the (z, y, x) volume as
    float32; with c / t lists (or absent: all), layout "ctzyx" -> (c, t, z, y, x)."""
    ds = open_dataset(str(ref.get("path", "")), ref.get("options") or {})
    layout = str(ref.get("layout", "zyx"))
    if layout == "zyx":
        return np.ascontiguousarray(ds.volume(int(ref.get("c", 0)), int(ref.get("t", 0))), dtype=np.float32)
    cs = ref.get("c")
    ts = ref.get("t")
    cs = list(range(ds.c)) if cs is None else ([int(cs)] if isinstance(cs, int) else [int(v) for v in cs])
    ts = list(range(ds.t)) if ts is None else ([int(ts)] if isinstance(ts, int) else [int(v) for v in ts])
    out = np.empty((len(cs), len(ts), ds.z, ds.y, ds.x), dtype=np.float32)
    for i, c in enumerate(cs):
        for j, t in enumerate(ts):
            out[i, j] = ds.volume(c, t)
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
    if desc["encoding"] == "zstd":
        data = _zstd().ZstdDecompressor().decompress(data, max_output_size=int(desc["raw_bytes"]))
    else:
        data = zlib.decompress(data)
    dtype = np.dtype(desc["dtype"])
    raw = np.frombuffer(data, dtype=np.uint8)
    if desc.get("shuffle"):
        raw = np.ascontiguousarray(raw.reshape(dtype.itemsize, -1).T)
    return raw.view(dtype).reshape(desc["shape"])
