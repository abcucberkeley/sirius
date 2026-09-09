"""Pipeline steps of the SIRIUS workbench, in numpy.

This module is the single implementation behind two entry points:

* ``run_pipeline(dataset_path, pipeline)`` -- what the desktop app's
  "Export pipeline as Python script" calls: it loads the dataset and runs
  the saved pipeline (the app's ``.sirius.toml`` converted to JSON) step by
  step, returning the final ``(c, t, z, y, x)`` float32 array and its
  metadata.
* the compute worker (``app/python/sirius_worker``) that the app's HPC
  backend talks to: every ``run`` request maps onto :func:`run_step` here.
  The worker imports ``sirius.workbench`` when the ``sirius`` package is
  installed and otherwise loads this file straight from the source tree, so
  there is exactly one copy of the step code.

Arrays are always ``(c, t, z, y, x)`` float32 (x fastest), labels are
``(t, z, y, x)`` uint32 with 0 = background. Only numpy is required;
``scipy`` (resampling, connected components, distance transforms),
``torch`` (segmentation models), ``tifffile`` / ``zarr`` (loaders) and the
``sirius`` extension (SIM reconstruction, GPU TIFF decode) are used when
importable and reported as missing otherwise.

Parameter keys are the application's: every step declares a :class:`StepSpec`
with exactly the keys, defaults and choices of the C++ operation's parameter
table (``app/core/ops/*.cpp``), and ``bindings/tests/test_workbench_schema.py``
checks those declarations against a snapshot of the C++ tables
(``op_schema.json``). Older key spellings are accepted through each spec's
aliases; a key a step does not understand raises an
:class:`UnknownParameterWarning` instead of being ignored.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "AXES",
    "Cancelled",
    "NotAvailable",
    "StepResult",
    "StepSpec",
    "UnknownParameterWarning",
    "load_dataset",
    "load_model",
    "model_cache_dir",
    "model_info",
    "resolve_device",
    "resolve_model_spec",
    "run_pipeline",
    "run_step",
    "step_kinds",
    "step_spec",
    "tiled_inference",
]

AXES = "ctzyx"

ProgressFn = Optional[Callable[[float, str], None]]
CancelFn = Optional[Callable[[], bool]]


class NotAvailable(NotImplementedError):
    """A step cannot run here: a dependency is missing or the kind is not
    implemented in Python. The message names the step and what is missing."""


class Cancelled(RuntimeError):
    """Raised inside a step when the caller's cancel callback fired."""


class UnknownParameterWarning(UserWarning):
    """A step received parameter keys it does not understand; they are
    ignored, but never silently (see :func:`_prepare_params`)."""


# --------------------------------------------------------------------------
# parameter helpers
# --------------------------------------------------------------------------


def _get(params: Dict[str, Any], keys: Any, default: Any = None) -> Any:
    """The first non-None value of `keys` (one key or a sequence) in `params`."""
    if isinstance(keys, str):
        keys = (keys,)
    for k in keys:
        if k in params and params[k] is not None:
            return params[k]
    return default


def _float(params, keys, default: float) -> float:
    v = _get(params, keys, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _int(params, keys, default: int) -> int:
    v = _get(params, keys, default)
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return int(default)


def _bool(params, keys, default: bool) -> bool:
    v = _get(params, keys, default)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "on", "yes")
    return bool(v)


def _str(params, keys, default: str = "") -> str:
    v = _get(params, keys, default)
    return "" if v is None else str(v)


def _list(params, keys, n: int, default: Sequence[float]) -> List[float]:
    return _as_list(_get(params, keys, None), n, default)


def _as_list(v: Any, n: int, default: Sequence[float]) -> List[float]:
    """`v` (a list, a "z, y, x" string or one number) as n floats, padded from `default`."""
    if v is None:
        return [float(d) for d in default]
    if isinstance(v, str):
        parts = [p for p in v.replace("×", ",").replace("x", ",").replace(";", ",").split(",") if p.strip()]
        try:
            v = [float(p) for p in parts]
        except ValueError:
            return [float(d) for d in default]
    if isinstance(v, (int, float)):
        v = [float(v)] * n
    v = [float(x) for x in v]
    if len(v) < n:
        v = v + [float(d) for d in default[len(v):]]
    return v[:n]


def _floats(v: Any) -> List[float]:
    """A list of floats from a list, a comma-separated string or one number (None = empty)."""
    if v is None or v == "":
        return []
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, str):
        v = [p for p in v.replace(";", ",").split(",") if p.strip()]
    return [float(x) for x in v]


def _pop_first(p: Dict[str, Any], keys: Sequence[str]) -> Any:
    """Removes every key of `keys` from `p`; returns the first non-None value."""
    found = None
    for k in keys:
        v = p.pop(k, None)
        if found is None and v is not None:
            found = v
    return found


def _voxel_um(meta: Optional[Dict[str, Any]]) -> Tuple[float, float, float]:
    """(x, y, z) voxel size of the metadata (the loader's defaults when absent)."""
    v = list((meta or {}).get("voxel_um") or []) + [0.1, 0.1, 0.2]
    return float(v[0]), float(v[1]), float(v[2])


def _choice(value: Any, options: Sequence[str], default: str) -> str:
    """Case-insensitive match of `value` against `options`, accepting prefixes
    ("cos" -> "Cosine")."""
    if value is None:
        return default
    s = str(value).strip().lower()
    if not s:
        return default
    for o in options:
        if o.lower() == s:
            return o
    for o in options:
        if o.lower().startswith(s) or s.startswith(o.lower()):
            return o
    return default


@dataclass(frozen=True)
class StepSpec:
    """What a step consumes. `defaults` holds the application's canonical
    parameter keys with the defaults of the C++ operation's parameter table
    (kept identical: the schema test compares them), `choices` the options of
    its choice parameters. `aliases` map older / worker spellings onto a
    canonical key (used only when the canonical key is absent), `translate`
    rewrites older key layouts in place before the check, and `extra` lists
    Python-only keys. Any other key raises an UnknownParameterWarning."""

    kind: str
    defaults: Dict[str, Any]
    choices: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    aliases: Dict[str, str] = field(default_factory=dict)
    extra: Tuple[str, ...] = ()
    translate: Optional[Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Any]] = None
    needs_labels: bool = False

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(self.defaults)

    def known(self) -> set:
        return set(self.defaults) | set(self.aliases) | set(self.extra)


def _prepare_params(spec: StepSpec, params: Optional[Dict[str, Any]],
                    meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The step's parameters with aliases renamed onto the canonical keys
    (canonical values win), older layouts translated, unknown keys warned
    about and missing keys filled from the defaults."""
    p = dict(params or {})
    for alias, key in spec.aliases.items():
        if alias in p:
            v = p.pop(alias)
            if p.get(key) is None:
                p[key] = v
    if spec.translate is not None:
        spec.translate(p, meta)
    unknown = sorted(k for k in p if k not in spec.known())
    if unknown:
        warnings.warn(f"step '{spec.kind}': unknown parameter(s) {', '.join(unknown)} ignored; "
                      f"it takes {', '.join(spec.keys) or 'no parameters'}", UnknownParameterWarning, stacklevel=3)
    for k, d in spec.defaults.items():
        if p.get(k) is None:
            p[k] = list(d) if isinstance(d, list) else d
    return p


def _progress(progress: ProgressFn, fraction: float, message: str = "") -> None:
    if progress is not None:
        progress(max(0.0, min(1.0, float(fraction))), message)


def _check_cancel(cancelled: CancelFn) -> None:
    if cancelled is not None and cancelled():
        raise Cancelled("cancelled")


def _as5(a: np.ndarray) -> np.ndarray:
    """Any rank <= 5 array as (c, t, z, y, x) float32 (missing leading axes are 1)."""
    a = np.asarray(a)
    if a.ndim > 5:
        raise ValueError(f"expected at most 5 dimensions, got shape {a.shape}")
    while a.ndim < 5:
        a = a[np.newaxis]
    return np.ascontiguousarray(a, dtype=np.float32)


def _dims(a: np.ndarray) -> Dict[str, int]:
    return dict(zip(AXES, (int(n) for n in a.shape)))


def _channel_index(params, keys: Any, meta: Dict[str, Any], c: int, default: int = 0) -> int:
    """A channel given as an index or as a label / wavelength string."""
    v = _get(params, keys, default)
    if isinstance(v, str):
        s = v.strip().lower()
        for i, ch in enumerate(meta.get("channels", [])[:c]):
            label = str(ch.get("label", "")).lower()
            nm = ch.get("wavelength_nm", 0)
            if s == label or (nm and s.startswith(str(int(nm)))) or (s and label.startswith(s)):
                return i
        try:
            v = int(float(s))
        except ValueError:
            v = default
    idx = int(v)
    if idx < 0 or idx >= c:
        raise ValueError(f"channel {idx} does not exist (input has {c})")
    return idx


# --------------------------------------------------------------------------
# metadata and loading
# --------------------------------------------------------------------------


def _default_meta(a: np.ndarray, source: str = "", fmt: str = "memory") -> Dict[str, Any]:
    c = int(a.shape[0])
    return {
        "name": os.path.splitext(os.path.basename(source.rstrip("/\\")))[0] if source else "array",
        "source": source,
        "format": fmt,
        "dims": _dims(a),
        "voxel_um": [0.1, 0.1, 0.2],  # x, y, z
        "frame_interval_s": 0.0,
        "channels": [{"label": f"ch {i}", "wavelength_nm": 0.0, "color": "#ffffff"} for i in range(c)],
        "rgb": False,
        "sim": {"present": False, "ndirs": 3, "nphases": 5, "fast_si": False},
    }


def _reorder_to_ctzyx(a: np.ndarray, axes: str) -> np.ndarray:
    """Permute an array whose axes are named by `axes` (letters of "ctzyx";
    other letters are folded into t) into (c, t, z, y, x)."""
    axes = axes.lower()
    a = np.asarray(a)
    if len(axes) != a.ndim:
        raise ValueError(f"axes '{axes}' do not match shape {a.shape}")
    known = [ax if ax in AXES else "t" for ax in axes]
    order = [[i for i, k in enumerate(known) if k == ax] for ax in AXES]
    perm = [i for idx in order for i in idx]
    out = np.transpose(a, perm)
    shape = []
    for idx in order:
        n = 1
        for i in idx:
            n *= a.shape[i]
        shape.append(n)
    return out.reshape(shape)


def _tiff_metadata(path: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """(axes string of the stored series, metadata dict) via tifffile, or (None, {})."""
    try:
        import tifffile  # type: ignore
    except ImportError:
        return None, {}
    info: Dict[str, Any] = {}
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0] if tf.series else None
        axes = series.axes if series is not None else None
        if tf.is_ome:
            info["format"] = "ome-tiff"
            try:
                info.update(_parse_ome_xml(tf.ome_metadata or ""))
            except Exception:  # noqa: BLE001 - metadata is best effort
                pass
        elif tf.is_imagej:
            info["format"] = "tiff"
            ij = tf.imagej_metadata or {}
            voxel = [0.0, 0.0, 0.0]
            if "spacing" in ij:
                voxel[2] = float(ij["spacing"])
            if "finterval" in ij:
                info["frame_interval_s"] = float(ij["finterval"])
            page = tf.pages[0]
            xres = page.tags.get("XResolution")
            yres = page.tags.get("YResolution")
            unit = str(ij.get("unit") or "").lower()
            if xres is not None and yres is not None and unit in ("micron", "um", "µm", "μm"):
                if xres.value[0] and yres.value[0]:
                    voxel[0] = xres.value[1] / xres.value[0]
                    voxel[1] = yres.value[1] / yres.value[0]
            if any(voxel):
                info["voxel_um"] = voxel
        else:
            info["format"] = "tiff"
        info["dtype"] = str(tf.pages[0].dtype)
        info["bytes_on_disk"] = os.path.getsize(path)
    return axes, info


def _parse_ome_xml(xml: str) -> Dict[str, Any]:
    import xml.etree.ElementTree as ET

    out: Dict[str, Any] = {}
    if not xml.strip():
        return out
    root = ET.fromstring(xml)
    ns = root.tag[: root.tag.index("}") + 1] if root.tag.startswith("{") else ""
    pixels = root.find(f".//{ns}Pixels")
    if pixels is None:
        return out
    g = pixels.get
    voxel = [float(g("PhysicalSizeX") or 0), float(g("PhysicalSizeY") or 0), float(g("PhysicalSizeZ") or 0)]
    if any(voxel):
        out["voxel_um"] = voxel
    if g("TimeIncrement"):
        out["frame_interval_s"] = float(g("TimeIncrement"))
    channels = []
    for ch in pixels.findall(f"{ns}Channel"):
        nm = ch.get("EmissionWavelength") or ch.get("ExcitationWavelength") or 0
        entry: Dict[str, Any] = {"label": ch.get("Name") or "", "wavelength_nm": float(nm)}
        color = ch.get("Color")
        if color:
            try:
                rgba = int(color) & 0xFFFFFFFF
                entry["color"] = f"#{(rgba >> 24) & 255:02x}{(rgba >> 16) & 255:02x}{(rgba >> 8) & 255:02x}"
            except ValueError:
                pass
        channels.append(entry)
    if channels:
        out["channels"] = channels
    return out


def load_dataset(path: str, page_order: str = "czt", c: Optional[int] = None, t: Optional[int] = None,
                 z: Optional[int] = None, progress: ProgressFn = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a TIFF / OME-TIFF or a zarr / N5 store as (c, t, z, y, x) float32.

    Plain multi-page TIFFs without dimension metadata are reshaped with
    `page_order` (fastest axis first, ImageJ's "czt") and the explicit
    counts; an unspecified count is derived from the page count.
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    _progress(progress, 0.0, f"reading {os.path.basename(path)}")
    lower = path.lower().rstrip("/\\")
    if os.path.isdir(path) or lower.endswith((".zarr", ".n5")):
        a, meta = _load_zarr(path)
    else:
        a, meta = _load_tiff(path, page_order, c, t, z)
    _progress(progress, 1.0, "loaded")
    return a, meta


def _load_zarr(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        import zarr  # type: ignore
    except ImportError as e:
        raise NotAvailable("loading zarr stores needs the 'zarr' package") from e
    node = zarr.open(path, mode="r")
    axes_names: Optional[List[str]] = None
    scale: Optional[List[float]] = None
    channels: List[Dict[str, Any]] = []
    arr = node
    attrs = dict(getattr(node, "attrs", {}) or {})
    if not hasattr(node, "shape"):  # a group: OME-NGFF multiscales or the first array
        ms = attrs.get("multiscales") or (attrs.get("ome") or {}).get("multiscales")
        if ms:
            m0 = ms[0]
            axes_names = [ax["name"] if isinstance(ax, dict) else str(ax) for ax in m0.get("axes", [])]
            ds0 = m0["datasets"][0]
            arr = node[ds0["path"]]
            for tr in ds0.get("coordinateTransformations", []):
                if tr.get("type") == "scale":
                    scale = [float(s) for s in tr["scale"]]
        else:
            keys = sorted(node.keys())
            if not keys:
                raise ValueError(f"{path}: empty group")
            arr = node[keys[0]]
        omero = attrs.get("omero") or (attrs.get("ome") or {}).get("omero")
        if omero:
            for ch in omero.get("channels", []):
                channels.append({"label": ch.get("label", ""),
                                 "wavelength_nm": float(ch.get("emission_wavelength") or 0),
                                 "color": "#" + ch["color"] if ch.get("color") else "#ffffff"})
    data = np.asarray(arr[...])
    if axes_names is None or len(axes_names) != data.ndim:
        if data.ndim > 5:
            raise ValueError(f"{path}: cannot map {data.ndim} axes onto (c, t, z, y, x)")
        axes_names = list(AXES[5 - data.ndim:])
    a = _as5(_reorder_to_ctzyx(data, "".join(ax[0] for ax in axes_names)))
    meta = _default_meta(a, path, "n5" if path.lower().rstrip("/\\").endswith(".n5") else "zarr")
    if scale is not None and len(scale) == len(axes_names):
        v = meta["voxel_um"]
        for ax, s in zip(axes_names, scale):
            if ax[0] == "x":
                v[0] = s
            elif ax[0] == "y":
                v[1] = s
            elif ax[0] == "z":
                v[2] = s
    if channels:
        meta["channels"] = (channels + meta["channels"])[: a.shape[0]]
    meta["dtype"] = str(data.dtype)
    meta["dims_from_metadata"] = True
    return a, meta


def _read_tiff_pages(path: str) -> Tuple[np.ndarray, Optional[str]]:
    """(array, axes) -- the sirius extension for plain stacks, tifffile otherwise."""
    axes, _ = _tiff_metadata(path)
    if axes is None or len(axes) <= 3:
        try:
            import sirius  # type: ignore

            return np.asarray(sirius.read_tiff(path, dtype=np.float32)), None
        except Exception:  # noqa: BLE001 - fall back to tifffile
            pass
    try:
        import tifffile  # type: ignore
    except ImportError as e:
        raise NotAvailable("loading TIFF needs the 'sirius' extension or 'tifffile'") from e
    return np.asarray(tifffile.imread(path)), axes


def _load_tiff(path: str, page_order: str, c: Optional[int], t: Optional[int], z: Optional[int]):
    _, info = _tiff_metadata(path)
    data, axes = _read_tiff_pages(path)
    dims_from_meta = False
    if axes is not None and len(axes) == data.ndim and data.ndim > 3:
        norm = axes.upper().replace("Q", "T").replace("S", "C").replace("I", "T")
        a = _as5(_reorder_to_ctzyx(data, norm))
        dims_from_meta = True
    else:
        pages = data.reshape((-1,) + data.shape[-2:])
        n = pages.shape[0]
        counts = {"c": int(c or 0), "t": int(t or 0), "z": int(z or 0)}
        unknown = [k for k, v in counts.items() if v <= 0]
        known = 1
        for v in counts.values():
            if v > 0:
                known *= v
        if len(unknown) > 1:
            for k in unknown:
                counts[k] = 1
            counts["z" if "z" in unknown else unknown[0]] = max(1, n // known)
        elif unknown:
            counts[unknown[0]] = max(1, n // known)
        if counts["c"] * counts["t"] * counts["z"] != n:
            raise ValueError(f"{n} pages do not factor into c{counts['c']} t{counts['t']} z{counts['z']}")
        order = page_order.lower()
        slowest_first = "".join(reversed(order))
        shape = tuple(counts[ax] for ax in slowest_first)
        a = _as5(_reorder_to_ctzyx(pages.reshape(shape + pages.shape[1:]), slowest_first + "yx"))
    meta = _default_meta(a, path, info.get("format", "tiff"))
    if info.get("voxel_um") and any(info["voxel_um"]):
        v = meta["voxel_um"]
        for i in range(3):
            if info["voxel_um"][i]:
                v[i] = float(info["voxel_um"][i])
    if info.get("frame_interval_s"):
        meta["frame_interval_s"] = float(info["frame_interval_s"])
    for i, ch in enumerate(info.get("channels", [])[: a.shape[0]]):
        meta["channels"][i].update({k: v for k, v in ch.items() if v not in ("", None)})
    meta["dtype"] = info.get("dtype", str(data.dtype))
    meta["bytes_on_disk"] = info.get("bytes_on_disk", os.path.getsize(path))
    meta["dims_from_metadata"] = dims_from_meta
    return a, meta


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------


@dataclass
class StepResult:
    array: np.ndarray
    meta: Dict[str, Any]
    labels: Optional[np.ndarray] = None       # (t, z, y, x) uint32
    prob: Optional[np.ndarray] = None         # (C, z, y, x) float32 of the last time point
    info: Dict[str, Any] = field(default_factory=dict)


_STEPS: Dict[str, Callable[..., StepResult]] = {}
_SPECS: Dict[str, StepSpec] = {}


def _step(spec: StepSpec) -> Callable[[Callable[..., StepResult]], Callable[..., StepResult]]:
    """Registers a step function under its spec's kind (dispatch and the
    schema drift test read ``_SPECS``)."""

    def wrap(fn: Callable[..., StepResult]) -> Callable[..., StepResult]:
        _SPECS[spec.kind] = spec
        _STEPS[spec.kind] = fn
        return fn

    return wrap


def _no_sim() -> Dict[str, Any]:
    return {"present": False, "ndirs": 3, "nphases": 5, "fast_si": False}


# --- intensity helpers (mirrors of sirius/image_ops.cpp) ---------------------


def _percentiles(values: np.ndarray, lo_pct: float, hi_pct: float, max_samples: int = 1 << 22) -> Tuple[float, float]:
    """``sirius::percentiles``: order statistics (no interpolation) of a
    fixed-stride sub-sample with NaNs dropped; flat quantiles fall back to the
    full range."""
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    n = v.size
    if n == 0:
        return 0.0, 0.0
    lo_pct = min(max(float(lo_pct), 0.0), 100.0)
    hi_pct = min(max(float(hi_pct), lo_pct), 100.0)
    stride = max(1, -(-n // max(int(max_samples), 1)))
    s = v[::stride]
    s = s[~np.isnan(s)]
    if s.size == 0:
        return 0.0, 0.0

    def rank(pct: float) -> int:  # llround for a non-negative argument
        return int(math.floor(pct / 100.0 * (s.size - 1) + 0.5))

    klo, khi = rank(lo_pct), rank(hi_pct)
    part = np.partition(s, [klo, khi])
    lo, hi = float(part[klo]), float(part[khi])
    if hi > lo:
        return lo, hi
    finite = v[~np.isnan(v)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(finite.min()), float(finite.max())


def _histogram(values: np.ndarray, bins: int, lo: float, hi: float) -> np.ndarray:
    """``sirius::histogram``: counts of `bins` equal bins over [lo, hi]; NaN and
    values outside the range are not counted, hi lands in the last bin."""
    counts = np.zeros(max(bins, 1), dtype=np.float64)
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0 or not hi > lo:
        return counts
    scale = bins / (float(hi) - float(lo))
    x = x[(x >= lo) & (x <= hi)]
    b = ((x - lo) * scale).astype(np.int64)
    b[b >= bins] = bins - 1
    return np.bincount(b, minlength=bins).astype(np.float64)


def _otsu_threshold(values: np.ndarray) -> float:
    """``sirius::app::otsuThreshold`` (threshold.cpp): Otsu's cut on a 256-bin
    histogram between the data's min and max, returned as the upper edge of
    the best bin."""
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("inf")
    mn, mx = float(v.min()), float(v.max())
    if not mx > mn:
        return mn
    bins = 256
    h = _histogram(v, bins, mn, mx)
    idx = np.arange(bins, dtype=np.float64)
    total = h.sum()
    sum_all = (idx * h).sum()
    w_b = np.cumsum(h)
    sum_b = np.cumsum(idx * h)
    w_f = total - w_b
    valid = (w_b > 0) & (w_f > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
    between[~valid] = -np.inf
    best = int(np.argmax(between))   # the first maximum, as the C++ strict '>' keeps
    return mn + (mx - mn) * float(best + 1) / bins


def _rescale_gamma(a: np.ndarray, lo: float, hi: float, gamma: float) -> np.ndarray:
    """``sirius::rescaleGamma``: clamp((v - lo) / (hi - lo), 0, 1) ^ (1 / gamma);
    an empty window maps v > hi to 1 and the rest to 0 (NaN stays NaN)."""
    span = float(hi) - float(lo)
    inv_gamma = 1.0 / gamma if gamma > 0.0 else 1.0
    out = np.empty_like(a, dtype=np.float32)
    for c in range(a.shape[0]):   # one (t, z, y, x) block at a time bounds the float64 scratch
        v = a[c]
        if not span > 0.0:
            r = np.where(v > hi, np.float32(1.0), np.float32(0.0))
            r[np.isnan(v)] = np.nan
            out[c] = r
            continue
        t = np.clip((v.astype(np.float64) - lo) / span, 0.0, 1.0)
        out[c] = t if inv_gamma == 1.0 else np.power(t, inv_gamma)
    return out


# --- reductions ---------------------------------------------------------------


def _kept_axes(params: Dict[str, Any]) -> str:
    v = params.get("keep")
    if isinstance(v, dict):  # {"c": true, "t": false, ...}
        return "".join(ax for ax in AXES if v.get(ax, True))
    if isinstance(v, (list, tuple)):
        return "".join(str(x)[0].lower() for x in v)
    if v is None:
        return AXES
    s = str(v).lower().replace("->", " ").split()
    s = s[-1] if s else ""
    return "".join(ax for ax in AXES if ax in s)


def _reduce(a: np.ndarray, reduce_axes: Sequence[int], op: str) -> np.ndarray:
    if not reduce_axes:
        return a
    axes = tuple(sorted(reduce_axes))
    if op == "sum":
        return a.sum(axis=axes, keepdims=True, dtype=np.float64).astype(np.float32)
    if op == "mean":
        return a.mean(axis=axes, keepdims=True, dtype=np.float64).astype(np.float32)
    if op == "max":
        return np.nanmax(a, axis=axes, keepdims=True)
    if op == "min":
        return np.nanmin(a, axis=axes, keepdims=True)
    raise ValueError(f"unknown reduction '{op}'")


def _einsum_legacy(p: Dict[str, Any], meta: Optional[Dict[str, Any]]) -> None:
    # older exports named the reduced axes instead of the kept ones
    reduced = _pop_first(p, ("reduce", "reduced"))
    if reduced is not None and p.get("keep") is None:
        if isinstance(reduced, (list, tuple)):
            reduced = "".join(str(x)[0].lower() for x in reduced)
        p["keep"] = "".join(ax for ax in AXES if ax not in str(reduced).lower())


_EINSUM = StepSpec("einsum", {"keep": "czyx", "reduction": "mean"},
                   choices={"reduction": ("sum", "mean", "max", "min")},
                   aliases={"axes": "keep", "kept": "keep", "op": "reduction", "red": "reduction"},
                   translate=_einsum_legacy)


@_step(_EINSUM)
def step_einsum(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any]) -> StepResult:
    """keep: the surviving axes ("czyx"); reduction: sum | mean | max | min.
    Reduced axes stay with length 1."""
    kept = _kept_axes(params)
    op = _choice(params.get("reduction"), _EINSUM.choices["reduction"], "mean")
    reduce_axes = [i for i, ax in enumerate(AXES) if ax not in kept]
    out = _reduce(a, reduce_axes, op)
    meta = dict(meta, dims=_dims(out))
    if "c" not in kept:
        meta["channels"] = [{"label": f"{op} over c", "wavelength_nm": 0.0, "color": "#ffffff"}]
        meta["rgb"] = False
    if "z" not in kept:
        meta["sim"] = _no_sim()
    return StepResult(np.ascontiguousarray(out, dtype=np.float32), meta,
                      info={"expression": f"{AXES} -> {kept or '·'}", "reduction": op})


_MAXPROJ = StepSpec("maxproj", {"axis": "z"}, choices={"axis": ("z", "t", "c")})


@_step(_MAXPROJ)
def step_maxproj(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any]) -> StepResult:
    """axis: z (default) | t | c."""
    axis = _choice(params.get("axis"), _MAXPROJ.choices["axis"], "z")
    return step_einsum(a, {"keep": AXES.replace(axis, ""), "reduction": "max"}, meta)


@_step(StepSpec("meant", {}))
def step_meant(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any]) -> StepResult:
    return step_einsum(a, {"keep": "czyx", "reduction": "mean"}, meta)


# --- intensity ----------------------------------------------------------------


def _contrast_legacy(p: Dict[str, Any], meta: Optional[Dict[str, Any]]) -> None:
    if _pop_first(p, ("per_channel", "perChannel")):
        warnings.warn("step 'contrast': per_channel is not a parameter of the application's Contrast step; "
                      "one window (the extreme percentiles over every channel) applies to all channels",
                      UnknownParameterWarning, stacklevel=5)


_CONTRAST = StepSpec(
    "contrast",
    {"min": 0.0, "max": 0.0, "gamma": 1.0, "lo_percentile": 0.2, "hi_percentile": 99.8, "bake": True},
    aliases={"low": "lo_percentile", "lo": "lo_percentile", "low_percentile": "lo_percentile",
             "p_low": "lo_percentile", "percentile_low": "lo_percentile",
             "high": "hi_percentile", "hi": "hi_percentile", "high_percentile": "hi_percentile",
             "p_high": "hi_percentile", "percentile_high": "hi_percentile"},
    translate=_contrast_legacy)


def _contrast_auto_window(a: np.ndarray, lo_pct: float, hi_pct: float, max_planes: int = 8) -> Tuple[float, float]:
    """``contrastAutoParams``: the extreme lo / hi percentiles over every
    channel, each estimated from at most `max_planes` planes spread over (t, z)."""
    c, t, z = a.shape[:3]
    planes = t * z
    step = max(1, -(-planes // max_planes)) if max_planes > 0 else 1
    lo, hi = float("inf"), float("-inf")
    for ch in range(c):
        samples = np.concatenate([a[ch, k // z, k % z].reshape(-1) for k in range(0, planes, step)])
        plo, phi = _percentiles(samples, lo_pct, hi_pct)
        lo, hi = min(lo, plo), max(hi, phi)
    if not lo < hi:
        lo, hi = 0.0, 1.0
    return lo, hi


@_step(_CONTRAST)
def step_contrast(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any]) -> StepResult:
    """min / max: the window mapped to 0..1 (max <= min = automatic: the
    lo_percentile / hi_percentile of the input, one window for every channel);
    gamma. `bake` is accepted (the data is always rewritten)."""
    lo_pct = _float(params, "lo_percentile", 0.2)
    hi_pct = _float(params, "hi_percentile", 99.8)
    if lo_pct >= hi_pct:
        raise ValueError("contrast: the auto low percentile must be below the high one")
    gamma = _float(params, "gamma", 1.0)
    lo, hi = _float(params, "min", 0.0), _float(params, "max", 0.0)
    automatic = not hi > lo
    if automatic:
        lo, hi = _contrast_auto_window(a, lo_pct, hi_pct)
    out = _rescale_gamma(a, lo, hi, gamma)
    return StepResult(out, dict(meta), info={"window": [lo, hi], "automatic": automatic, "gamma": gamma})


def _read_pages(path: str) -> np.ndarray:
    """A TIFF as (pages, y, x) float32 (flat / dark images)."""
    a, _ = load_dataset(path)
    return a.reshape((-1,) + a.shape[-2:])


_FLATFIELD = StepSpec("flatfield", {"flat": "", "dark": ""},
                      aliases={"flat_field": "flat", "flat_path": "flat", "dark_path": "dark"})


@_step(_FLATFIELD)
def step_flatfield(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any]) -> StepResult:
    """flat: TIFF of the illumination profile (one page, or one page per
    channel); dark: optional camera offset. v = (v - dark) * mean(flat - dark) / (flat - dark)."""
    flat_path = _str(params, "flat")
    if not flat_path:
        raise ValueError("flat-field: choose a flat image ('flat')")
    flat = _read_pages(flat_path)
    dark_path = _str(params, "dark")
    dark = _read_pages(dark_path) if dark_path else None
    c = a.shape[0]
    if flat.shape[-2:] != a.shape[-2:]:
        raise ValueError(f"the flat image is {flat.shape[-1]} × {flat.shape[-2]}, the data {a.shape[-1]} × {a.shape[-2]}")
    if dark is not None and dark.shape[-2:] != a.shape[-2:]:
        raise ValueError(f"the dark image is {dark.shape[-1]} × {dark.shape[-2]}, the data {a.shape[-1]} × {a.shape[-2]}")
    out = np.empty_like(a, dtype=np.float32)
    for ch in range(c):
        f = flat[ch if flat.shape[0] == c else 0]
        d = dark[ch if dark.shape[0] == c else 0] if dark is not None else None
        gain = f - (d if d is not None else 0.0)
        mean = float(gain.mean(dtype=np.float64))
        if not mean > 0.0:
            raise ValueError("flat-field: the flat image must be brighter than the dark image")
        gain = (mean / np.maximum(gain, np.float32(1e-6 * mean))).astype(np.float32)
        out[ch] = (a[ch] - (d if d is not None else 0.0)) * gain
    return StepResult(out, dict(meta))


def _bleach_legacy(p: Dict[str, Any], meta: Optional[Dict[str, Any]]) -> None:
    to_mean = _pop_first(p, ("to_mean", "toMean", "reference_mean"))
    if to_mean is not None and p.get("mode") is None:
        p["mode"] = "Match mean" if _bool({"v": to_mean}, "v", False) else "Match first frame"


_BLEACH = StepSpec("bleach", {"mode": "Match first frame", "over": "t"},
                   choices={"mode": ("Match first frame", "Match mean"), "over": ("t", "z")},
                   translate=_bleach_legacy)


def _equalize_frames(stack: np.ndarray, to_mean: bool) -> Tuple[np.ndarray, np.ndarray]:
    """``sirius::equalizeFrames``: scale every frame (axis 0) so its sum
    matches the first frame's (or the mean); empty frames are left alone."""
    sums = stack.reshape(stack.shape[0], -1).sum(axis=1, dtype=np.float64)
    target = float(sums.mean()) if to_mean else float(sums[0])
    ok = (sums != 0.0) & np.isfinite(sums)
    scale = np.ones_like(sums, dtype=np.float32)
    scale[ok] = (target / sums[ok]).astype(np.float32)
    return stack * scale.reshape((-1,) + (1,) * (stack.ndim - 1)), scale


@_step(_BLEACH)
def step_bleach(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any]) -> StepResult:
    """mode: Match first frame | Match mean; over: t (the time series of every
    channel) | z (the planes of every stack)."""
    to_mean = _choice(params.get("mode"), _BLEACH.choices["mode"], "Match first frame") == "Match mean"
    over = _choice(params.get("over"), _BLEACH.choices["over"], "t")
    out = np.empty_like(a, dtype=np.float32)
    scales: List[Any] = []
    for c in range(a.shape[0]):
        if over == "t":
            out[c], s = _equalize_frames(a[c], to_mean)
            scales.append(s.tolist())
        else:
            per_t = []
            for t in range(a.shape[1]):
                out[c, t], s = _equalize_frames(a[c, t], to_mean)
                per_t.append(s.tolist())
            scales.append(per_t)
    return StepResult(out, dict(meta), info={"scales": scales, "mode": "mean" if to_mean else "first", "over": over})


# --- geometry -----------------------------------------------------------------


def _croppad_legacy(p: Dict[str, Any], meta: Optional[Dict[str, Any]]) -> None:
    # older exports: origin [z, y, x] and size [z, y, x] lists
    origin = _pop_first(p, ("origin", "offset", "start"))
    size = _pop_first(p, ("size", "extent", "shape"))
    if origin is not None:
        for k, v in zip(("z0", "y0", "x0"), _as_list(origin, 3, [0, 0, 0])):
            p.setdefault(k, int(v))
    if size is not None:
        for k, v in zip(("z", "y", "x"), _as_list(size, 3, [0, 0, 0])):
            p.setdefault(k, int(v))


_CROPPAD = StepSpec("croppad", {"z0": 0, "y0": 0, "x0": 0, "z": 0, "y": 0, "x": 0, "fill": 0.0},
                    aliases={"pad_value": "fill"}, translate=_croppad_legacy, needs_labels=True)


@_step(_CROPPAD)
def step_croppad(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any],
                 labels: Optional[np.ndarray] = None) -> StepResult:
    """z0, y0, x0: origin (negative = padding); z, y, x: size (0 = to the
    edge); fill. Labels are cropped along (padding is background)."""
    nz, ny, nx = a.shape[2:]
    origin = [_int(params, k, 0) for k in ("z0", "y0", "x0")]
    size = [_int(params, k, 0) for k in ("z", "y", "x")]
    ext = [size[i] if size[i] > 0 else max(1, (nz, ny, nx)[i] - origin[i]) for i in range(3)]
    fill = _float(params, "fill", 0.0)

    def box(src: np.ndarray, dst: np.ndarray) -> None:
        s, d = [], []
        for i, n in enumerate((nz, ny, nx)):
            s0, s1 = max(origin[i], 0), min(origin[i] + ext[i], n)
            if s1 <= s0:
                return   # no overlap: the output is all fill
            s.append(slice(s0, s1))
            d.append(slice(s0 - origin[i], s1 - origin[i]))
        lead = (slice(None),) * (src.ndim - 3)
        dst[lead + tuple(d)] = src[lead + tuple(s)]

    out = np.full(a.shape[:2] + tuple(ext), np.float32(fill), dtype=np.float32)
    box(a, out)
    out_labels = None
    if labels is not None and labels.size:
        out_labels = np.zeros((labels.shape[0],) + tuple(ext), dtype=np.uint32)
        box(labels, out_labels)
    m = dict(meta, dims=_dims(out))
    if ext[0] != nz:
        m["sim"] = _no_sim()
    return StepResult(out, m, labels=out_labels, info={"origin": origin, "size": ext})


def _resample_legacy(p: Dict[str, Any], meta: Optional[Dict[str, Any]]) -> None:
    # older exports: voxel [z, y, x] (or one isotropic size) or factor [z, y, x]
    voxel = _pop_first(p, ("voxel", "voxel_um", "target", "isotropic"))
    factor = _pop_first(p, ("factor", "factors", "zoom"))
    if voxel is not None:
        target = [float(voxel)] * 3 if isinstance(voxel, (int, float)) else _as_list(voxel, 3, [0, 0, 0])
        for k, v in zip(("voxel_z", "voxel_y", "voxel_x"), target):
            p.setdefault(k, v)
    elif factor is not None:
        vx, vy, vz = _voxel_um(meta)
        f = _as_list(factor, 3, [1, 1, 1])
        for k, cur, fi in zip(("voxel_z", "voxel_y", "voxel_x"), (vz, vy, vx), f):
            p.setdefault(k, cur / fi if fi > 0 else 0.0)


_RESAMPLE = StepSpec("resample", {"voxel_x": 0.0, "voxel_y": 0.0, "voxel_z": 0.0, "interpolation": "linear"},
                     choices={"interpolation": ("linear", "cubic", "nearest")},
                     aliases={"interp": "interpolation"}, translate=_resample_legacy)


def _resample_extent(n: int, d: float, t: float) -> int:
    """``resampleGeometry``: the output extent keeps the physical field."""
    return 1 if n == 1 else int(math.floor((n - 1) * d / t + 1e-9)) + 1


def _axis_taps(n_in: int, n_out: int, ratio: float, interp: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """``axisTaps`` of image_ops.cpp for every output index of one axis:
    (indices, weights) pairs; positions outside the input weigh 0 (fill)."""
    p = np.arange(n_out, dtype=np.float64) * ratio
    if n_in == 1:
        ok = (p >= -0.5) & (p <= 0.5)
        return [(np.zeros(n_out, dtype=np.int64), ok.astype(np.float64))]
    if interp == "nearest":
        ok = (p >= -0.5) & (p < n_in - 0.5)
        i = np.clip(np.floor(p + 0.5), 0, n_in - 1).astype(np.int64)
        return [(i, ok.astype(np.float64))]
    ok = (p >= 0.0) & (p <= n_in - 1)
    fl = np.floor(p)
    i0 = fl.astype(np.int64)
    f = p - fl
    if interp == "linear":
        return [(np.clip(i0, 0, n_in - 1), (1.0 - f) * ok), (np.clip(i0 + 1, 0, n_in - 1), f * ok)]
    f2, f3 = f * f, f * f * f   # Catmull-Rom, clamped taps at the edges
    w = [0.5 * (-f3 + 2.0 * f2 - f), 0.5 * (3.0 * f3 - 5.0 * f2 + 2.0), 0.5 * (-3.0 * f3 + 4.0 * f2 + f), 0.5 * (f3 - f2)]
    return [(np.clip(i0 - 1 + k, 0, n_in - 1), w[k] * ok) for k in range(4)]


def _resample_volume(v: np.ndarray, extents: Sequence[int], ratios: Sequence[float], interp: str) -> np.ndarray:
    """Separable resampling of a (z, y, x) volume onto `extents` with the
    taps of the application's `resampleAffine` (axis-aligned scaling)."""
    out = v
    for axis in (2, 1, 0):
        n_in, n_out = out.shape[axis], int(extents[axis])
        if n_in == n_out and abs(ratios[axis] - 1.0) < 1e-12:
            continue
        acc = None
        for idx, w in _axis_taps(n_in, n_out, ratios[axis], interp):
            shape = [1, 1, 1]
            shape[axis] = n_out
            term = np.take(out, idx, axis=axis) * w.astype(np.float32).reshape(shape)
            acc = term if acc is None else acc + term
        out = acc
    return np.ascontiguousarray(out, dtype=np.float32)


@_step(_RESAMPLE)
def step_resample(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any]) -> StepResult:
    """voxel_x / voxel_y / voxel_z: the output voxel size in µm (0 keeps the
    axis); interpolation: linear | cubic | nearest."""
    vx, vy, vz = _voxel_um(meta)
    if not (vx > 0 and vy > 0 and vz > 0):
        raise ValueError(f"resample: the input voxel size must be positive (voxel_um = {[vx, vy, vz]})")
    interp = _choice(params.get("interpolation"), _RESAMPLE.choices["interpolation"], "linear")
    target = [_float(params, "voxel_z", 0.0), _float(params, "voxel_y", 0.0), _float(params, "voxel_x", 0.0)]
    cur = [vz, vy, vx]
    tgt = [target[i] if target[i] > 0 else cur[i] for i in range(3)]
    extents = [_resample_extent(a.shape[2 + i], cur[i], tgt[i]) for i in range(3)]
    ratios = [tgt[i] / cur[i] for i in range(3)]
    voxel_um = [tgt[2], tgt[1], tgt[0]]
    if tuple(extents) == a.shape[2:] and all(abs(r - 1.0) < 1e-12 for r in ratios):
        return StepResult(a, dict(meta, voxel_um=voxel_um), info={"voxel_um": voxel_um})
    out = np.empty(a.shape[:2] + tuple(extents), dtype=np.float32)
    for c in range(a.shape[0]):
        for t in range(a.shape[1]):
            out[c, t] = _resample_volume(a[c, t], extents, ratios, interp)
    m = dict(meta, dims=_dims(out), voxel_um=voxel_um)
    if extents[0] != a.shape[2]:
        m["sim"] = _no_sim()
    return StepResult(out, m, info={"voxel_um": voxel_um, "interpolation": interp})


# --- combine ------------------------------------------------------------------


def _hex_to_rgb(s: str) -> Tuple[float, float, float]:
    s = str(s).strip().lstrip("#")
    if len(s) != 6:
        return 1.0, 1.0, 1.0
    try:
        return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return 1.0, 1.0, 1.0


_MERGE = StepSpec("merge", {"blend": "Additive", "colors": [], "weights": [], "normalize_percentile": 99.9},
                  choices={"blend": ("Additive", "Screen", "Max")}, aliases={"colours": "colors"})


@_step(_MERGE)
def step_merge(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any]) -> StepResult:
    """blend: Additive | Screen | Max; colors: "#rrggbb" per channel (empty =
    the channels' own colours); weights: per-channel gain (empty = 1);
    normalize_percentile: each channel is scaled so this percentile maps to 1
    unless it already lies in 0..1."""
    if meta.get("rgb"):
        raise ValueError("merge: the input is already an RGB merge")
    blend = _choice(params.get("blend"), _MERGE.choices["blend"], "Additive")
    c = a.shape[0]
    colors = params.get("colors")
    if isinstance(colors, str):
        colors = [x.strip() for x in colors.split(",") if x.strip()]
    colors = list(colors or [])
    channel_colors = [ch.get("color", "#ffffff") for ch in meta.get("channels", [])]
    colors = [colors[i] if i < len(colors) else (channel_colors[i] if i < len(channel_colors) else "#ffffff")
              for i in range(c)]
    rgbs = [_hex_to_rgb(col) for col in colors]
    weights = _floats(params.get("weights"))
    pct = _float(params, "normalize_percentile", 99.9)
    scales = []
    for i in range(c):
        ch = a[i]
        finite = ch[~np.isnan(ch)]
        mn, mx = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        if mn >= 0.0 and mx <= 1.0:
            scales.append(1.0)
            continue
        hi = _percentiles(ch, 0.0, pct)[1]
        scales.append(1.0 / hi if hi > 0.0 else 1.0)
    out = np.zeros((3,) + a.shape[1:], dtype=np.float32)
    for i in range(c):
        w = float(weights[i]) if i < len(weights) else 1.0
        v0 = np.clip(a[i] * np.float32(scales[i] * w), 0.0, 1.0)
        for k in range(3):
            if rgbs[i][k] == 0.0:
                continue
            contribution = v0 * np.float32(rgbs[i][k])
            if blend == "Screen":
                out[k] = 1.0 - (1.0 - out[k]) * (1.0 - contribution)
            elif blend == "Max":
                np.maximum(out[k], contribution, out=out[k])
            else:
                out[k] = np.minimum(1.0, out[k] + contribution)
    m = dict(meta, dims=_dims(out), rgb=True,
             channels=[{"label": n, "wavelength_nm": 0.0, "color": col}
                       for n, col in (("R", "#ff0000"), ("G", "#00ff00"), ("B", "#0000ff"))],
             sim=_no_sim())
    return StepResult(out, m, info={"blend": blend, "colors": colors, "scales": scales})


# --- labels (mirrors of app/core/labels.cpp and segment_common.cpp) -----------


def _ndimage():
    try:
        from scipy import ndimage  # type: ignore
    except ImportError as e:
        raise NotAvailable("label post-processing needs 'scipy' (pip install scipy)") from e
    return ndimage


def _label_components(mask: np.ndarray) -> np.ndarray:
    """6-connected components, labelled 1..n in raster order."""
    labels, _ = _ndimage().label(mask)
    return labels.astype(np.uint32, copy=False)


def _distance_transform(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance of every foreground voxel to the nearest background
    voxel; without any background every voxel is as far as the volume is wide."""
    mask = np.asarray(mask, dtype=bool)
    if mask.all():
        return np.full(mask.shape, float(max(mask.shape)), dtype=np.float32)
    return _ndimage().distance_transform_edt(mask).astype(np.float32)


def _distance_seeds(mask: np.ndarray, min_distance: float) -> Tuple[np.ndarray, int]:
    """``distanceSeeds``: local maxima (26-neighbourhood) of the distance
    transform, accepted deepest first when at least `min_distance` from every
    accepted seed. Returns (seed volume, count)."""
    ndimage = _ndimage()
    mask = np.asarray(mask, dtype=bool)
    dist = _distance_transform(mask)
    neighbourhood = ndimage.maximum_filter(dist, size=3, mode="constant", cval=-np.inf)
    candidates = np.flatnonzero(mask & (dist > 0.0) & (dist >= neighbourhood))
    seeds = np.zeros(mask.shape, dtype=np.uint32)
    if candidates.size == 0:
        return seeds, 0
    d = dist.reshape(-1)[candidates]
    order = np.argsort(-d, kind="stable")
    candidates = candidates[order]
    coords = np.stack(np.unravel_index(candidates, mask.shape), axis=1).astype(np.float64)
    min_d2 = max(float(min_distance), 1.0) ** 2
    alive = np.ones(candidates.size, dtype=bool)
    flat = seeds.reshape(-1)
    n = 0
    for k in range(candidates.size):
        if not alive[k]:
            continue
        n += 1
        flat[candidates[k]] = n
        alive &= ((coords - coords[k]) ** 2).sum(axis=1) >= min_d2
    return seeds, n


def _watershed(landscape: np.ndarray, mask: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Marker-based flooding of `landscape` (higher = ridge) from `seeds`
    inside `mask`, 6-connected -- scikit-image's watershed, which is the same
    priority flood as the application's."""
    try:
        from skimage.segmentation import watershed  # type: ignore
    except ImportError as e:
        raise NotAvailable("watershed post-processing needs 'scikit-image' (pip install scikit-image); "
                           "choose post = Connected components to run without it") from e
    return watershed(landscape, markers=seeds, mask=mask, connectivity=1).astype(np.uint32)


def _remove_small(labels: np.ndarray, min_voxels: int, relabel: bool = True) -> np.ndarray:
    """``removeSmall`` / ``dropSmall``: drop labels with fewer than `min_voxels`
    voxels and, with `relabel`, number the rest 1..n densely in id order."""
    counts = np.bincount(labels.reshape(-1))
    keep = (counts >= max(int(min_voxels), 0)) & (counts > 0)
    keep[0] = False
    remap = np.zeros(counts.size, dtype=np.uint32)
    if relabel:
        remap[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.uint32)
    else:
        remap[keep] = np.nonzero(keep)[0].astype(np.uint32)
    return remap[labels]


def _labels_from_probabilities(fg: np.ndarray, boundary: Optional[np.ndarray], threshold: float, post: str,
                               min_voxels: int, seed_distance: float, seeds: str = "Distance maxima",
                               seed_depth: float = 2.0, external: Optional[np.ndarray] = None) -> np.ndarray:
    """``labelsFromProbabilities``: fg > threshold -> instances by connected
    components, a seeded watershed (on the boundary map, or the negated
    distance transform) or none (one semantic label), then small-object removal."""
    mask = fg > threshold
    if post.startswith("None"):
        labels = mask.astype(np.uint32)
    elif post.startswith("Watershed"):
        distance = _distance_transform(mask)
        if external is not None:
            marks = np.where(mask, external, 0).astype(np.uint32)
            n = int(marks.max())
        elif seeds == "H-maxima":
            marks, n = _h_maxima_seeds(distance, mask, seed_depth)
        else:
            marks, n = _distance_seeds(mask, seed_distance)
        if n == 0:
            labels = _label_components(mask)
        else:
            landscape = boundary if boundary is not None else -distance
            labels = _watershed(landscape, mask, marks)
    else:
        labels = _label_components(mask)
    return _remove_small(labels, min_voxels)


def _border_labels(vol: np.ndarray) -> np.ndarray:
    """Ids whose bounding box touches the volume border (z only counts when
    the volume has more than one plane)."""
    faces = [vol[:, 0, :], vol[:, -1, :], vol[:, :, 0], vol[:, :, -1]]
    if vol.shape[0] > 1:
        faces += [vol[0], vol[-1]]
    ids = np.unique(np.concatenate([f.reshape(-1) for f in faces]))
    return ids[ids > 0]


def _label_flags(vol: np.ndarray, low_conf: float, size_outlier_factor: float,
                 confidence: Optional[Dict[int, float]] = None) -> Dict[str, List[int]]:
    """``LabelVolume::applyFlags`` on one volume: small (< median / 8),
    touching border, merged? (> factor x median), low conf (when known)."""
    counts = np.bincount(vol.reshape(-1))
    ids = np.flatnonzero(counts[1:] > 0) + 1
    flags: Dict[str, List[int]] = {"low conf": [], "small": [], "touching border": [], "merged?": []}
    if ids.size == 0:
        return flags
    sizes = counts[ids]
    median = int(np.sort(sizes)[sizes.size // 2])
    min_voxels = max(1, median // 8)
    border = set(int(i) for i in _border_labels(vol))
    for i, n in zip(ids.tolist(), sizes.tolist()):
        if confidence is not None and confidence.get(i, 1.0) < low_conf:
            flags["low conf"].append(i)
        if n < min_voxels:
            flags["small"].append(i)
        if i in border:
            flags["touching border"].append(i)
        if size_outlier_factor > 0.0 and n > size_outlier_factor * median:
            flags["merged?"].append(i)
    return flags


def _threshold_legacy(p: Dict[str, Any], meta: Optional[Dict[str, Any]]) -> None:
    # Python-only pipelines named the manual cut `threshold` and had no method
    manual = _pop_first(p, ("threshold",))
    if manual is not None:
        p.setdefault("value", manual)
        if p.get("method") is None:
            p["method"] = "Manual"
    if p.get("method") is None and p.get("percentile") is not None:
        p["method"] = "Percentile"


_THRESHOLD = StepSpec(
    "threshold",
    {"channel": 0, "method": "Otsu", "value": 0.5, "percentile": 90.0, "post": "Connected components",
     "min_voxels": 20, "seed_distance": 5.0, "class_name": "object"},
    choices={"method": ("Manual", "Otsu", "Percentile"), "post": ("Connected components", "Watershed (distance)")},
    aliases={"input_channel": "channel", "minVoxels": "min_voxels", "min_size": "min_voxels"},
    translate=_threshold_legacy)


def _multi_otsu_upper(values: np.ndarray) -> float:
    """``multiOtsuThresholds`` (classic.cpp): the upper of two Otsu cuts over a
    128-bin histogram, which keeps only the brightest of three classes."""
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return 0.0
    lo, hi = float(v.min()), float(v.max())
    if not hi > lo:
        return lo
    bins = 128
    counts, _ = np.histogram(v, bins=bins, range=(lo, hi))
    w = np.concatenate([[0.0], np.cumsum(counts.astype(np.float64))])
    m = np.concatenate([[0.0], np.cumsum(np.arange(bins) * counts.astype(np.float64))])
    total, mean = w[bins], m[bins]
    if not total > 0.0:
        return hi
    best, best_b = -1.0, 2 * bins // 3
    grand = mean / total
    for a_ in range(1, bins - 1):
        w0 = w[a_]
        if w0 <= 0.0:
            continue
        m0 = m[a_] / w0
        w1 = w[a_ + 1:bins] - w0
        w2 = total - w[a_ + 1:bins]
        ok = (w1 > 0.0) & (w2 > 0.0)
        if not ok.any():
            continue
        m1 = np.where(ok, (m[a_ + 1:bins] - m[a_]) / np.where(ok, w1, 1.0), 0.0)
        m2 = np.where(ok, (mean - m[a_ + 1:bins]) / np.where(ok, w2, 1.0), 0.0)
        between = w0 * (m0 - grand) ** 2 + w1 * (m1 - grand) ** 2 + w2 * (m2 - grand) ** 2
        between = np.where(ok, between, -1.0)
        k = int(np.argmax(between))
        if between[k] > best:
            best, best_b = float(between[k]), a_ + 1 + k
    return lo + (hi - lo) * (best_b + 1) / bins


def _global_cut(v: np.ndarray, method: str, params: Dict[str, Any]) -> float:
    if method == "Otsu":
        return _otsu_threshold(v)
    if method == "Triangle":
        return _triangle_threshold(v)
    if method == "Li":
        return _li_threshold(v)
    if method == "Yen":
        return _yen_threshold(v)
    if method == "Isodata":
        return _isodata_threshold(v)
    if method == "Multi-Otsu":
        return _multi_otsu_upper(v)
    if method == "Percentile":
        return _percentiles(v, 0.0, _float(params, "percentile", 90.0))[1]
    return _float(params, "value", 0.5)


@_step(_THRESHOLD)
def step_threshold(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any]) -> StepResult:
    """channel; method: Manual (value) | Otsu | Percentile (percentile); post:
    Connected components | Watershed (distance) (seed_distance); min_voxels;
    class_name."""
    if meta.get("rgb"):
        raise ValueError("threshold needs an intensity channel, not an RGB merge")
    c = _channel_index(params, "channel", meta, a.shape[0])
    method = _choice(params.get("method"), _THRESHOLD.choices["method"], "Otsu")
    post = _choice(params.get("post"), _THRESHOLD.choices["post"], "Connected components")
    min_voxels = _int(params, "min_voxels", 20)
    seed_distance = _float(params, "seed_distance", 5.0)
    labels = np.zeros((a.shape[1],) + a.shape[2:], dtype=np.uint32)
    cuts = []
    total = 0
    for t in range(a.shape[1]):
        v = a[c, t]
        cut = _global_cut(v, method, params)
        labels[t] = _labels_from_probabilities(v, None, cut, post, min_voxels, seed_distance)
        total += int(labels[t].max())
        cuts.append(cut)
    return StepResult(a, dict(meta), labels=labels,
                      info={"thresholds": cuts, "method": method, "channel": c, "labels": total,
                            "class_name": _str(params, "class_name", "object")})


def _local_mean_plane(pl: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)² window clamped to the plane (integral image)."""
    y, x = pl.shape
    integral = np.zeros((y + 1, x + 1), dtype=np.float64)
    integral[1:, 1:] = pl.astype(np.float64).cumsum(axis=0).cumsum(axis=1)
    yy, xx = np.arange(y), np.arange(x)
    y0, y1 = np.maximum(yy - r, 0), np.minimum(yy + r + 1, y)
    x0, x1 = np.maximum(xx - r, 0), np.minimum(xx + r + 1, x)
    s = (integral[y1[:, None], x1[None, :]] - integral[y0[:, None], x1[None, :]]
         - integral[y1[:, None], x0[None, :]] + integral[y0[:, None], x0[None, :]])
    count = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return (s / count).astype(np.float32)


def _local_stats_plane(pl: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray]:
    """Local mean and standard deviation over a (2r+1)² window clamped to the
    plane -- ``localStatsPlane`` in classic.cpp."""
    mean = _local_mean_plane(pl, r)
    mean_sq = _local_mean_plane(np.asarray(pl, dtype=np.float64) ** 2, r)
    return mean, np.sqrt(np.maximum(0.0, mean_sq - mean ** 2))


def _dog_plane(pl: np.ndarray, sigma: float, ratio: float = 1.6) -> np.ndarray:
    """``dogPlane``: difference of two Gaussians, clamped at zero -- a band-pass
    that answers to blobs about `sigma` across."""
    ndimage = _ndimage()
    small = ndimage.gaussian_filter(pl.astype(np.float32), sigma=sigma, mode="mirror",
                                    truncate=math.ceil(3.0 * sigma) / sigma if sigma > 0 else 4.0)
    big_sigma = sigma * max(1.1, ratio)
    big = ndimage.gaussian_filter(pl.astype(np.float32), sigma=big_sigma, mode="mirror",
                                  truncate=math.ceil(3.0 * big_sigma) / big_sigma if big_sigma > 0 else 4.0)
    return np.maximum(0.0, small - big).astype(np.float32)


def _gaussian_volume(v: np.ndarray, sx: float, sy: float, sz: float) -> np.ndarray:
    """``gaussianVolume``: separable 3D Gaussian, mirrored borders, truncated
    at three sigma -- the same taps the application uses."""
    ndimage = _ndimage()
    out = np.asarray(v, dtype=np.float32)
    for axis, sigma in ((2, sx), (1, sy), (0, sz)):
        if sigma <= 1e-6 or out.shape[axis] < 2:
            continue
        out = ndimage.gaussian_filter1d(out, sigma=sigma, axis=axis, mode="mirror",
                                        truncate=math.ceil(3.0 * sigma) / sigma).astype(np.float32)
    return out


def _symmetric_eigenvalues(a11, a12, a13, a22, a23, a33):
    """``symmetricEigenvalues``: the analytic (trigonometric) eigenvalues of a
    symmetric 3x3, smallest absolute value first."""
    p1 = a12 ** 2 + a13 ** 2 + a23 ** 2
    q = (a11 + a22 + a33) / 3.0
    p2 = (a11 - q) ** 2 + (a22 - q) ** 2 + (a33 - q) ** 2 + 2.0 * p1
    p = np.sqrt(np.maximum(1e-30, p2 / 6.0))
    b11, b22, b33 = (a11 - q) / p, (a22 - q) / p, (a33 - q) / p
    b12, b13, b23 = a12 / p, a13 / p, a23 / p
    det = b11 * (b22 * b33 - b23 * b23) - b12 * (b12 * b33 - b23 * b13) + b13 * (b12 * b23 - b22 * b13)
    phi = np.arccos(np.clip(det / 2.0, -1.0, 1.0)) / 3.0
    e1 = q + 2.0 * p * np.cos(phi)
    e3 = q + 2.0 * p * np.cos(phi + 2.0 * math.pi / 3.0)
    e2 = 3.0 * q - e1 - e3
    diag = p1 <= 1e-30
    e1 = np.where(diag, a11, e1)
    e2 = np.where(diag, a22, e2)
    e3 = np.where(diag, a33, e3)
    stack = np.stack([e1, e2, e3], axis=0)
    order = np.argsort(np.abs(stack), axis=0, kind="stable")
    return tuple(np.take_along_axis(stack, order[k:k + 1], axis=0)[0] for k in range(3))


def _frangi_volume(vol: np.ndarray, z_aspect: float, sigma_min: float, sigma_max: float, scales: int) -> np.ndarray:
    """``frangiVolume``: Frangi vesselness in 3D, the best response over
    `scales` widths -- filaments whatever direction they run in."""
    z, y, x = vol.shape
    out = np.zeros(vol.shape, dtype=np.float32)
    scales = max(1, int(scales))
    sigma_min = max(0.3, sigma_min)
    sigma_max = max(sigma_min, sigma_max)
    z_aspect = max(1e-6, z_aspect)
    zi, yi, xi = (np.clip(np.arange(n) + d, 0, n - 1) for n, d in ((z, 0), (y, 0), (x, 0)))
    zp, zm = np.clip(np.arange(z) + 1, 0, z - 1), np.clip(np.arange(z) - 1, 0, z - 1)
    yp, ym = np.clip(np.arange(y) + 1, 0, y - 1), np.clip(np.arange(y) - 1, 0, y - 1)
    xp, xm = np.clip(np.arange(x) + 1, 0, x - 1), np.clip(np.arange(x) - 1, 0, x - 1)
    for k in range(scales):
        sigma = sigma_min if scales == 1 else sigma_min * (sigma_max / sigma_min) ** (k / (scales - 1))
        w = _gaussian_volume(vol, sigma, sigma, sigma / z_aspect).astype(np.float64)
        norm = sigma * sigma
        c = w
        dxx = norm * (w[:, :, xp] + w[:, :, xm] - 2.0 * c)
        dyy = norm * (w[:, yp, :] + w[:, ym, :] - 2.0 * c)
        dzz = norm * (w[zp, :, :] + w[zm, :, :] - 2.0 * c) if z > 1 else np.zeros_like(w)
        dxy = norm * 0.25 * (w[:, yp][:, :, xp] + w[:, ym][:, :, xm] - w[:, yp][:, :, xm] - w[:, ym][:, :, xp])
        if z > 1:
            dxz = norm * 0.25 * (w[zp][:, :, xp] + w[zm][:, :, xm] - w[zp][:, :, xm] - w[zm][:, :, xp])
            dyz = norm * 0.25 * (w[zp][:, yp, :] + w[zm][:, ym, :] - w[zp][:, ym, :] - w[zm][:, yp, :])
        else:
            dxz = np.zeros_like(w)
            dyz = np.zeros_like(w)
        if z > 1:
            l1, l2, l3 = _symmetric_eigenvalues(dxx, dxy, dxz, dyy, dyz, dzz)
            bright = (l2 < 0.0) & (l3 < 0.0)
            ra = np.abs(l2) / np.maximum(np.abs(l3), 1e-12)
            rb = np.abs(l1) / np.maximum(np.sqrt(np.abs(l2 * l3)), 1e-12)
            s_mag = np.sqrt(l1 ** 2 + l2 ** 2 + l3 ** 2)
            shape = 1.0 - np.exp(-ra ** 2 / 0.5)
        else:
            # one plane: the third eigenvalue is identically zero, which the
            # 3D measure reads as "no tube"; the 2D Frangi measure over the
            # two in-plane eigenvalues, ordered by magnitude (classic.cpp)
            tr = dxx + dyy
            disc = np.sqrt((dxx - dyy) ** 2 + 4.0 * dxy ** 2)
            l1, l2 = 0.5 * (tr + disc), 0.5 * (tr - disc)
            swap = np.abs(l1) > np.abs(l2)
            l1, l2 = np.where(swap, l2, l1), np.where(swap, l1, l2)
            bright = l2 < 0.0
            rb = np.abs(l1) / np.maximum(np.abs(l2), 1e-12)
            s_mag = np.sqrt(l1 ** 2 + l2 ** 2)
            shape = 1.0
        max_s = float(s_mag[bright].max()) if bright.any() else 0.0
        c2 = 2.0 * max(1e-12, 0.5 * max_s) ** 2
        v = np.where(bright, shape * np.exp(-rb ** 2 / 0.5), 0.0).astype(np.float32)
        v = np.where(v > 0.0, (v * (1.0 - np.exp(-s_mag ** 2 / c2))).astype(np.float32), 0.0).astype(np.float32)
        out = np.maximum(out, v)
    return out


def _meijering_volume(vol: np.ndarray, z_aspect: float, sigma_min: float, sigma_max: float, scales: int) -> np.ndarray:
    """``meijeringVolume``: Meijering neuriteness. The Hessian eigenvalues are
    mixed (l_i = (1 - a) e_i + a tr H, a = 1 / (ndim + 1)) before the largest by
    magnitude is taken, and each scale is normalised to its own maximum. The
    image is negated first: the filter is written for dark ridges."""
    z, y, x = vol.shape
    out = np.zeros(vol.shape, dtype=np.float32)
    scales = max(1, int(scales))
    sigma_min = max(0.3, sigma_min)
    sigma_max = max(sigma_min, sigma_max)
    z_aspect = max(1e-6, z_aspect)
    alpha = 0.25 if z > 1 else 1.0 / 3.0
    zp, zm = np.clip(np.arange(z) + 1, 0, z - 1), np.clip(np.arange(z) - 1, 0, z - 1)
    yp, ym = np.clip(np.arange(y) + 1, 0, y - 1), np.clip(np.arange(y) - 1, 0, y - 1)
    xp, xm = np.clip(np.arange(x) + 1, 0, x - 1), np.clip(np.arange(x) - 1, 0, x - 1)
    negated = (-vol).astype(np.float32)
    for k in range(scales):
        sigma = sigma_min if scales == 1 else sigma_min * (sigma_max / sigma_min) ** (k / (scales - 1))
        w = _gaussian_volume(negated, sigma, sigma, sigma / z_aspect).astype(np.float64)
        norm = sigma * sigma
        c = w
        dxx = norm * (w[:, :, xp] + w[:, :, xm] - 2.0 * c)
        dyy = norm * (w[:, yp, :] + w[:, ym, :] - 2.0 * c)
        dzz = norm * (w[zp, :, :] + w[zm, :, :] - 2.0 * c) if z > 1 else np.zeros_like(w)
        dxy = norm * 0.25 * (w[:, yp][:, :, xp] + w[:, ym][:, :, xm] - w[:, yp][:, :, xm] - w[:, ym][:, :, xp])
        if z > 1:
            dxz = norm * 0.25 * (w[zp][:, :, xp] + w[zm][:, :, xm] - w[zp][:, :, xm] - w[zm][:, :, xp])
            dyz = norm * 0.25 * (w[zp][:, yp, :] + w[zm][:, ym, :] - w[zp][:, ym, :] - w[zm][:, yp, :])
        else:
            dxz = np.zeros_like(w)
            dyz = np.zeros_like(w)
        e = _symmetric_eigenvalues(dxx, dxy, dxz, dyy, dyz, dzz)
        trace = e[0] + e[1] + e[2]
        # a single plane has no third eigenvalue worth the name: the C++ skips it
        candidates = [(1.0 - alpha) * ev + alpha * trace for ev in (e if z > 1 else e[1:])]
        stacked = np.stack(candidates)
        picked = np.take_along_axis(stacked, np.abs(stacked).argmax(0)[None], 0).squeeze(0)
        response = np.maximum(picked, 0.0)
        peak = float(response.max())
        if peak <= 0.0:
            continue
        out = np.maximum(out, (response / peak).astype(np.float32))
    return out


def _log_blob_seeds(values: np.ndarray, mask: np.ndarray, z_aspect: float, sigma_min: float, sigma_max: float,
                    scales: int) -> Tuple[np.ndarray, int]:
    """``logBlobSeeds``: the strongest scale-normalised Laplacian-of-Gaussian
    response over a range of widths peaks once per round object whatever its
    size; peaks are taken strongest first and suppress their own radius."""
    ndimage = _ndimage()
    z, y, x = values.shape
    mask = np.asarray(mask, dtype=bool)
    scales = max(1, int(scales))
    sigma_min = max(0.3, sigma_min)
    sigma_max = max(sigma_min, sigma_max)
    z_aspect = max(1e-6, z_aspect)
    best = np.zeros(values.shape, dtype=np.float32)
    best_scale = np.zeros(values.shape, dtype=np.float32)
    zp, zm = np.clip(np.arange(z) + 1, 0, z - 1), np.clip(np.arange(z) - 1, 0, z - 1)
    yp, ym = np.clip(np.arange(y) + 1, 0, y - 1), np.clip(np.arange(y) - 1, 0, y - 1)
    xp, xm = np.clip(np.arange(x) + 1, 0, x - 1), np.clip(np.arange(x) - 1, 0, x - 1)
    for k in range(scales):
        sigma = sigma_min if scales == 1 else sigma_min * (sigma_max / sigma_min) ** (k / (scales - 1))
        blur = _gaussian_volume(values, sigma, sigma, sigma / z_aspect)
        # float32 through the Laplacian, as the C++ does: computing the taps in
        # float64 reorders equal-looking peaks and renumbers the seeds
        two = np.float32(2.0)
        lx = (blur[:, :, xp] + blur[:, :, xm] - two * blur).astype(np.float32)
        ly = (blur[:, yp, :] + blur[:, ym, :] - two * blur).astype(np.float32)
        lz = ((blur[zp, :, :] + blur[zm, :, :] - two * blur).astype(np.float32) if z > 1
              else np.zeros_like(blur, dtype=np.float32))
        total = (lx + ly + lz).astype(np.float32)
        response = (-(sigma * sigma) * total.astype(np.float64)).astype(np.float32)
        response = np.where(mask, response, 0.0).astype(np.float32)
        take = response > best
        best = np.where(take, response, best).astype(np.float32)
        best_scale = np.where(take, np.float32(sigma), best_scale).astype(np.float32)
    # peaks: no stronger response anywhere in the 26-neighbourhood
    peak = ndimage.maximum_filter(best, size=3, mode="nearest")
    candidates = np.flatnonzero(mask.reshape(-1) & (best.reshape(-1) > 0.0) & (best.reshape(-1) >= peak.reshape(-1)))
    seeds = np.zeros(values.shape, dtype=np.uint32)
    if candidates.size == 0:
        return seeds, 0
    order = np.argsort(-best.reshape(-1)[candidates], kind="stable")
    candidates = candidates[order]
    coords = np.stack(np.unravel_index(candidates, values.shape), axis=1).astype(np.float64)
    coords[:, 0] *= z_aspect
    radii = np.maximum(1.0, math.sqrt(3.0) * best_scale.reshape(-1)[candidates].astype(np.float64))
    flat = seeds.reshape(-1)
    accepted: List[Tuple[np.ndarray, float]] = []
    for k in range(candidates.size):
        r2 = radii[k] ** 2
        if any(float(((coords[k] - c) ** 2).sum()) < max(r2, ar2) for c, ar2 in accepted):
            continue
        accepted.append((coords[k], r2))
        flat[candidates[k]] = 1
    # numbered by position, as the C++ does: the response decides which peaks
    # survive, but two near-tied peaks must not renumber the result
    taken = np.flatnonzero(flat)
    flat[taken] = np.arange(1, taken.size + 1, dtype=np.uint32)
    return seeds, int(taken.size)


def _frangi_plane(pl: np.ndarray, sigma_min: float, sigma_max: float, steps: int) -> np.ndarray:
    """``frangiPlane``: Frangi vesselness in the plane, the best response over
    `steps` widths between the two sigmas."""
    ndimage = _ndimage()
    out = np.zeros(pl.shape, dtype=np.float32)
    steps = max(1, int(steps))
    for k in range(steps):
        sigma = sigma_min if steps == 1 else sigma_min + (sigma_max - sigma_min) * k / (steps - 1)
        trunc = math.ceil(3.0 * sigma) / sigma if sigma > 0 else 4.0
        w = ndimage.gaussian_filter(pl.astype(np.float64), sigma=sigma, mode="mirror", truncate=trunc)
        norm = sigma * sigma
        # clamped neighbours, exactly as the C++ indexes them: a wrapped shift
        # would differ along every border
        yy, xx = np.arange(w.shape[0]), np.arange(w.shape[1])
        ym, yp = np.maximum(yy - 1, 0), np.minimum(yy + 1, w.shape[0] - 1)
        xm, xp = np.maximum(xx - 1, 0), np.minimum(xx + 1, w.shape[1] - 1)
        dxx = norm * (w[:, xp] + w[:, xm] - 2.0 * w)
        dyy = norm * (w[yp, :] + w[ym, :] - 2.0 * w)
        dxy = norm * 0.25 * (w[np.ix_(yp, xp)] + w[np.ix_(ym, xm)] - w[np.ix_(yp, xm)] - w[np.ix_(ym, xp)])
        t = np.sqrt((dxx - dyy) ** 2 + 4.0 * dxy ** 2)
        l1, l2 = 0.5 * (dxx + dyy + t), 0.5 * (dxx + dyy - t)
        swap = np.abs(l1) > np.abs(l2)
        l1, l2 = np.where(swap, l2, l1), np.where(swap, l1, l2)
        bright = l2 < 0.0
        rb = np.abs(l1) / np.maximum(np.abs(l2), 1e-12)
        s_mag = np.sqrt(l1 ** 2 + l2 ** 2)
        max_s = float(s_mag[bright].max()) if bright.any() else 0.0
        c2 = 2.0 * max(1e-12, 0.5 * max_s) ** 2
        v = np.where(bright, np.exp(-rb ** 2 / 0.5) * (1.0 - np.exp(-s_mag ** 2 / c2)), 0.0)
        out = np.maximum(out, v.astype(np.float32))
    return out


def _h_maxima_seeds(values: np.ndarray, mask: np.ndarray, h: float) -> Tuple[np.ndarray, int]:
    """``hMaximaSeeds``: the regional maxima of the h-maxima transform, so a
    peak must stand `h` above its surroundings to seed its own object."""
    ndimage = _ndimage()
    mask = np.asarray(mask, dtype=bool)
    values = np.asarray(values, dtype=np.float32)
    under = np.where(mask, values, 0.0).astype(np.float32)
    marker = np.where(mask, values - max(float(h), 1e-6), 0.0).astype(np.float32)
    g = _reconstruct_by_dilation(marker, under)
    # Regional maxima of g inside the mask: a whole plateau of equal value is
    # maximal only when nothing next to it is higher, so non-maximality has to
    # spread across the plateau. Testing each voxel on its own would keep the
    # inside of a plateau whose rim touches something higher.
    structure = ndimage.generate_binary_structure(g.ndim, 1)
    inside = mask & (g > 0.0)
    masked = np.where(mask, g, -np.inf)
    nonmax = inside & (ndimage.grey_dilation(masked, footprint=structure, mode="constant", cval=-np.inf) > g)
    while True:
        reach = ndimage.grey_dilation(np.where(nonmax, g, -np.inf), footprint=structure, mode="constant", cval=-np.inf)
        grown = inside & ~nonmax & (reach == g)
        if not grown.any():
            break
        nonmax |= grown
    seeds, count = ndimage.label(inside & ~nonmax, structure=structure)
    return seeds.astype(np.uint32), int(count)


def _reconstruct_by_dilation(marker: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``reconstructByDilation``: geodesic dilation of `marker` under `mask` to
    stability, 6-connected."""
    ndimage = _ndimage()
    structure = ndimage.generate_binary_structure(marker.ndim, 1)
    out = np.minimum(marker, mask).astype(np.float32)
    while True:
        grown = np.minimum(ndimage.grey_dilation(out, footprint=structure, mode="nearest"), mask)
        if np.array_equal(grown, out):
            return out
        out = grown


def _filter_plane(pl: np.ndarray, tophat: int, sigma: float) -> np.ndarray:
    """White top-hat with a (2r+1)² box (borders clamped), then a Gaussian
    truncated at ceil(3 sigma) with mirrored borders -- classic.cpp's steps."""
    ndimage = _ndimage()
    out = pl
    if tophat > 0:
        size = (2 * tophat + 1, 2 * tophat + 1)
        opening = ndimage.grey_dilation(ndimage.grey_erosion(out, size=size, mode="nearest"), size=size, mode="nearest")
        out = np.maximum(0.0, out - opening).astype(np.float32)
    if sigma > 0.0:
        r = max(1, int(math.ceil(3.0 * sigma)))
        out = ndimage.gaussian_filter(out.astype(np.float32), sigma, mode="mirror", truncate=r / sigma)
    return np.asarray(out, dtype=np.float32)


def _median_plane(pl: np.ndarray) -> np.ndarray:
    """``medianFilterPlane``: 3x3 median with clamped borders."""
    if pl.shape[0] < 3 or pl.shape[1] < 3:
        return np.asarray(pl, dtype=np.float32)
    return np.asarray(_ndimage().median_filter(pl, size=3, mode="nearest"), dtype=np.float32)


def _anisotropic_diffusion_plane(pl: np.ndarray, iterations: int, k: float) -> np.ndarray:
    """``anisotropicDiffusionPlane``: Perona-Malik with the exponential
    conductance, four clamped neighbours and lambda 0.25. ``k`` is a fraction of
    the plane's intensity range, measured once before the first step."""
    out = np.asarray(pl, dtype=np.float32)
    if iterations <= 0 or out.shape[0] < 3 or out.shape[1] < 3:
        return out
    lo, hi = float(out.min()), float(out.max())
    kk = max(1e-12, k) * max(1e-12, hi - lo)
    inv = 1.0 / (kk * kk)
    for _ in range(iterations):
        cur = out.astype(np.float64)
        total = np.zeros_like(cur)
        for shifted in (np.vstack([cur[:1], cur[:-1]]), np.vstack([cur[1:], cur[-1:]]),
                        np.hstack([cur[:, :1], cur[:, :-1]]), np.hstack([cur[:, 1:], cur[:, -1:]])):
            g = shifted - cur
            total += g * np.exp(-g * g * inv)
        out = (cur + 0.25 * total).astype(np.float32)
    return out


def _histogram_256(v: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """The 256 bin histogram ``triangleThreshold`` and ``liThreshold`` share."""
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return np.zeros(256, dtype=np.int64), 0.0, 0.0
    lo, hi = float(finite.min()), float(finite.max())
    if not hi > lo:
        return np.zeros(256, dtype=np.int64), lo, lo
    # 256 equal bins over [lo, hi], hi in the last one (labels.cpp histogramOf)
    idx = np.clip(((finite.astype(np.float64) - lo) * (256.0 / (hi - lo))).astype(np.int64), 0, 255)
    return np.bincount(idx, minlength=256).astype(np.int64), lo, hi


def _bin_value(bin_index: int, lo: float, hi: float) -> float:
    return float(np.float32(lo + np.float32((bin_index + 0.5) / 256.0 * (hi - lo))))


def _triangle_threshold(v: np.ndarray) -> float:
    """``triangleThreshold``: the bin furthest from the line joining the
    histogram's peak to the far end of its longer tail."""
    bins, lo, hi = _histogram_256(v)
    if not hi > lo:
        return lo
    peak = int(np.argmax(bins))
    peak_count = int(bins[peak])
    nonzero = np.nonzero(bins)[0]
    first, last = int(nonzero[0]), int(nonzero[-1])
    tail_right = (last - peak) >= (peak - first)
    end = last if tail_right else first
    if end == peak:
        return _bin_value(peak, lo, hi)
    dx = float(end - peak)
    dy = -float(peak_count)
    norm = math.sqrt(dx * dx + dy * dy)
    span = range(min(peak, end), max(peak, end) + 1)
    best, best_bin = -1.0, peak
    for b in span:
        px = float(b - peak)
        py = float(int(bins[b]) - peak_count)
        d = abs(px * dy - py * dx) / norm
        if d > best:
            best, best_bin = d, b
    return _bin_value(best_bin, lo, hi)


def _li_threshold(v: np.ndarray) -> float:
    """``liThreshold``: Li & Tam's minimum cross-entropy fixed point."""
    bins, lo, hi = _histogram_256(v)
    if not hi > lo:
        return lo
    levels = np.arange(256, dtype=np.float64)
    total = float(bins.sum())
    if total <= 0.0:
        return lo
    t = float((levels * bins).sum() / total)
    for _ in range(100):
        below = levels <= t
        count_lo, count_hi = float(bins[below].sum()), float(bins[~below].sum())
        mean_lo = float((levels[below] * bins[below]).sum() / count_lo) if count_lo > 0.0 else 0.0
        mean_hi = float((levels[~below] * bins[~below]).sum() / count_hi) if count_hi > 0.0 else 0.0
        a, b = mean_lo + 1.0, mean_hi + 1.0
        denominator = math.log(b) - math.log(a)
        if denominator == 0.0:
            break
        nxt = (b - a) / denominator
        if not math.isfinite(nxt):
            break
        if abs(nxt - t) < 0.5:
            t = nxt
            break
        t = nxt
    return _bin_value(int(min(max(t, 0.0), 255.0)), lo, hi)


def _yen_threshold(v: np.ndarray) -> float:
    """``yenThreshold``: Yen's entropic correlation criterion on the histogram."""
    bins, lo, hi = _histogram_256(v)
    total = float(bins.sum())
    if not hi > lo or total <= 0.0:
        return lo
    pmf = bins.astype(np.float64) / total
    p1 = np.cumsum(pmf)
    p1_sq = np.cumsum(pmf**2)
    p2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
    best, best_bin = -math.inf, 0
    for b in range(255):
        denominator = p1_sq[b] * p2_sq[b + 1]
        mass = p1[b] * (1.0 - p1[b])
        if denominator <= 0.0 or mass <= 0.0:
            continue
        crit = math.log(mass * mass / denominator)
        if crit > best:
            best, best_bin = crit, b
    return _bin_value(best_bin, lo, hi)


def _isodata_threshold(v: np.ndarray) -> float:
    """``isodataThreshold``: the Ridler-Calvard iteration from the image mean."""
    bins, lo, hi = _histogram_256(v)
    if not hi > lo:
        return lo
    levels = np.arange(256, dtype=np.float64)
    total = float(bins.sum())
    if total <= 0.0:
        return lo
    t = float((levels * bins).sum() / total)
    for _ in range(100):
        below = levels <= t
        count_lo, count_hi = float(bins[below].sum()), float(bins[~below].sum())
        if count_lo <= 0.0 or count_hi <= 0.0:
            break
        mean_lo = float((levels[below] * bins[below]).sum() / count_lo)
        mean_hi = float((levels[~below] * bins[~below]).sum() / count_hi)
        nxt = 0.5 * (mean_lo + mean_hi)
        if abs(nxt - t) < 0.5:
            t = nxt
            break
        t = nxt
    return _bin_value(int(min(max(t, 0.0), 255.0)), lo, hi)


def _fill_holes_3d(mask: np.ndarray, max_voxels: int) -> np.ndarray:
    """``fillHoles3D``: background the foreground encloses in 3D, cavity by
    cavity so a size limit can leave a real lumen open."""
    ndimage = _ndimage()
    structure = ndimage.generate_binary_structure(3, 1)
    background = mask == 0
    labelled, count = ndimage.label(background, structure=structure)
    if count == 0:
        return mask
    border = np.unique(np.concatenate([
        labelled[0].ravel(), labelled[-1].ravel(),
        labelled[:, 0].ravel(), labelled[:, -1].ravel(),
        labelled[:, :, 0].ravel(), labelled[:, :, -1].ravel()]))
    outside = np.zeros(count + 1, dtype=bool)
    outside[border[border > 0]] = True
    sizes = np.bincount(labelled.ravel(), minlength=count + 1)
    fill = ~outside
    fill[0] = False
    if max_voxels > 0:
        fill &= sizes <= max_voxels
    out = mask.copy()
    out[fill[labelled]] = 1
    return out


def _rolling_ball_plane(pl: np.ndarray, radius: float) -> np.ndarray:
    """``rollingBallPlane``: a grey opening with a hemispherical structuring
    element, run on a decimated copy and interpolated back up, subtracted."""
    out = np.asarray(pl, dtype=np.float32)
    y, x = out.shape
    if radius <= 0.0 or y <= 0 or x <= 0:
        return out
    # half away from zero like C++'s lround (round() is half to even: 2.5 -> 2)
    shrink = int(min(8, max(1, math.floor(radius / 10.0 + 0.5))))
    scaled = max(1.0, radius / shrink)
    half = max(1, int(math.floor(scaled)))
    side = 2 * half + 1
    dy, dx = np.mgrid[-half:half + 1, -half:half + 1]
    r2 = scaled * scaled - dy.astype(np.float64) ** 2 - dx.astype(np.float64) ** 2
    ball = np.where(r2 >= 0.0, np.sqrt(np.maximum(r2, 0.0)), -1.0)

    sy, sx = (y + shrink - 1) // shrink, (x + shrink - 1) // shrink
    small = np.empty((sy, sx), dtype=np.float64)
    big = out.astype(np.float64)
    for r in range(sy):
        for c in range(sx):
            rows = np.minimum(np.arange(r * shrink, r * shrink + shrink), y - 1)
            cols = np.minimum(np.arange(c * shrink, c * shrink + shrink), x - 1)
            small[r, c] = big[np.ix_(rows, cols)].min()

    def sweep(source: np.ndarray, sign: int) -> np.ndarray:
        best = np.full(source.shape, np.inf if sign < 0 else -np.inf)
        rows = np.arange(sy)
        cols = np.arange(sx)
        for i in range(side):
            for j in range(side):
                h = ball[i, j]
                if h < 0.0:
                    continue
                off_y, off_x = i - half, j - half
                rr = np.clip(rows + sign * off_y, 0, sy - 1)
                cc = np.clip(cols + sign * off_x, 0, sx - 1)
                shifted = source[np.ix_(rr, cc)]
                best = np.minimum(best, shifted - h) if sign < 0 else np.maximum(best, shifted + h)
        return best

    centre = sweep(small, -1)
    background = sweep(centre, 1)

    offset = 0.5 * (shrink - 1.0)
    fy = np.arange(y, dtype=np.float64) if shrink == 1 else (np.arange(y, dtype=np.float64) - offset) / shrink
    fx = np.arange(x, dtype=np.float64) if shrink == 1 else (np.arange(x, dtype=np.float64) - offset) / shrink
    r0 = np.clip(np.floor(fy).astype(np.int64), 0, sy - 1)
    c0 = np.clip(np.floor(fx).astype(np.int64), 0, sx - 1)
    r1, c1 = np.clip(r0 + 1, 0, sy - 1), np.clip(c0 + 1, 0, sx - 1)
    wy = np.clip(fy - r0, 0.0, 1.0)[:, None]
    wx = np.clip(fx - c0, 0.0, 1.0)[None, :]
    a = background[np.ix_(r0, c0)]
    b = background[np.ix_(r0, c1)]
    cc2 = background[np.ix_(r1, c0)]
    d = background[np.ix_(r1, c1)]
    full = (a * (1.0 - wx) + b * wx) * (1.0 - wy) + (cc2 * (1.0 - wx) + d * wx) * wy
    return np.maximum(0.0, out - full.astype(np.float32)).astype(np.float32)


def _skeletonize_3d(mask: np.ndarray) -> np.ndarray:
    """``skeletonize3D``: topological thinning by deleting simple points, one
    of the six border directions at a time, keeping end points so a curve
    thins to a line instead of eroding away."""
    m = (np.asarray(mask) != 0).astype(np.uint8).copy()
    z, y, x = m.shape
    if m.size == 0:
        return m

    def block(k, r, c):
        b = np.zeros((3, 3, 3), dtype=np.uint8)
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    jz, jy, jx = k + dz, r + dy, c + dx
                    if 0 <= jz < z and 0 <= jy < y and 0 <= jx < x:
                        b[dz + 1, dy + 1, dx + 1] = m[jz, jy, jx]
        return b

    def object_connected(b):
        on = [(i, j, k2) for i in range(3) for j in range(3) for k2 in range(3)
              if b[i, j, k2] and not (i == 1 and j == 1 and k2 == 1)]
        if not on:
            return False
        seen = {on[0]}
        stack = [on[0]]
        while stack:
            cz, cy, cx = stack.pop()
            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        n = (cz + dz, cy + dy, cx + dx)
                        if not all(0 <= v < 3 for v in n) or n == (1, 1, 1) or n in seen:
                            continue
                        if b[n]:
                            seen.add(n)
                            stack.append(n)
        return len(seen) == len(on)

    def background_connected(b):
        faces = [(0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)]
        starts = [f for f in faces if not b[f]]
        if not starts:
            return False
        seen = {starts[0]}
        stack = [starts[0]]
        while stack:
            cz, cy, cx = stack.pop()
            for dz, dy, dx in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
                n = (cz + dz, cy + dy, cx + dx)
                if not all(0 <= v < 3 for v in n) or n == (1, 1, 1) or n in seen:
                    continue
                if abs(n[0] - 1) + abs(n[1] - 1) + abs(n[2] - 1) > 2 or b[n]:
                    continue
                seen.add(n)
                stack.append(n)
        return all(f in seen for f in starts)

    def removable(k, r, c):
        b = block(k, r, c)
        if int(b.sum()) - 1 <= 1:
            return False   # an end point: the line stops here
        return object_connected(b) and background_connected(b)

    directions = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
    changed = True
    passes = 0
    while changed and passes < 1000:
        changed = False
        passes += 1
        for dz, dy, dx in directions:
            candidates = []
            for k in range(z):
                for r in range(y):
                    for c in range(x):
                        if not m[k, r, c]:
                            continue
                        jz, jy, jx = k + dz, r + dy, c + dx
                        outside = not (0 <= jz < z and 0 <= jy < y and 0 <= jx < x)
                        if not (outside or not m[jz, jy, jx]):
                            continue
                        if removable(k, r, c):
                            candidates.append((k, r, c))
            for k, r, c in candidates:
                if removable(k, r, c):
                    m[k, r, c] = 0
                    changed = True
    return m


def _expand_labels(labels: np.ndarray, distance: float, z_aspect: float) -> np.ndarray:
    """``expandLabels``: a Dijkstra from every labelled voxel at once over the
    6-neighbourhood (a z step costs z_aspect, the planes being that much
    further apart than the pixels). A voxel the same distance
    from two labels stays background so the two cannot fuse."""
    import heapq
    if distance <= 0.0 or labels.size == 0:
        return labels
    z_aspect = max(1e-6, z_aspect)
    step_z = z_aspect
    shape = labels.shape
    best = np.full(shape, np.inf)
    came_from = np.zeros(shape, dtype=np.uint32)
    tied = np.zeros(shape, dtype=bool)
    queue = []
    for idx in zip(*np.nonzero(labels)):
        best[idx] = 0.0
        came_from[idx] = labels[idx]
        heapq.heappush(queue, (0.0, idx))
    offsets = (((-1, 0, 0), step_z), ((1, 0, 0), step_z), ((0, -1, 0), 1.0),
               ((0, 1, 0), 1.0), ((0, 0, -1), 1.0), ((0, 0, 1), 1.0))
    while queue:
        d, idx = heapq.heappop(queue)
        if d > best[idx] or d >= distance:
            continue
        for (dz, dy, dx), step in offsets:
            j = (idx[0] + dz, idx[1] + dy, idx[2] + dx)
            if not (0 <= j[0] < shape[0] and 0 <= j[1] < shape[1] and 0 <= j[2] < shape[2]):
                continue
            if labels[j] != 0:
                continue
            nd = d + step
            if nd > distance:
                continue
            if nd < best[j] - 1e-9:
                best[j] = nd
                came_from[j] = came_from[idx]
                tied[j] = False
                heapq.heappush(queue, (nd, j))
            elif abs(nd - best[j]) <= 1e-9 and came_from[idx] != came_from[j]:
                tied[j] = True
    grown = labels.copy()
    claim = (labels == 0) & (came_from != 0) & ~tied
    grown[claim] = came_from[claim]
    return grown


def _hysteresis_mask(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """``hysteresisMask``: every 6-connected component of ``low`` that holds at
    least one voxel of ``high``."""
    ndimage = _ndimage()
    labelled, count = ndimage.label(low.astype(bool), structure=ndimage.generate_binary_structure(low.ndim, 1))
    if count == 0:
        return np.zeros_like(low, dtype=np.uint8)
    keep = np.zeros(count + 1, dtype=bool)
    keep[np.unique(labelled[(high > 0) & (low > 0)])] = True
    keep[0] = False
    return keep[labelled].astype(np.uint8)


def _gradient_magnitude(v: np.ndarray, z_aspect: float) -> np.ndarray:
    """``gradientMagnitude``: central differences with clamped borders, z
    divided by the voxel aspect so the gradient is physical."""
    inv_z = 1.0 / z_aspect if z_aspect > 0.0 else 1.0
    d = v.astype(np.float64)
    gz = 0.5 * (np.concatenate([d[1:], d[-1:]]) - np.concatenate([d[:1], d[:-1]])) * inv_z
    gy = 0.5 * (np.concatenate([d[:, 1:], d[:, -1:]], axis=1) - np.concatenate([d[:, :1], d[:, :-1]], axis=1))
    gx = 0.5 * (np.concatenate([d[:, :, 1:], d[:, :, -1:]], axis=2) - np.concatenate([d[:, :, :1], d[:, :, :-1]], axis=2))
    return np.sqrt(gx * gx + gy * gy + gz * gz).astype(np.float32)


_LINE_DY = ((0, 0, 0), (-1, 0, 1), (-1, 0, 1), (-1, 0, 1))
_LINE_DX = ((-1, 0, 1), (0, 0, 0), (-1, 0, 1), (1, 0, -1))


def _line_shift(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = m
    if dy < 0:
        out = np.vstack([out[:1]] * (-dy) + [out[:dy]])
    elif dy > 0:
        out = np.vstack([out[dy:]] + [out[-1:]] * dy)
    if dx < 0:
        out = np.hstack([out[:, :1]] * (-dx) + [out[:, :dx]])
    elif dx > 0:
        out = np.hstack([out[:, dx:]] + [out[:, -1:]] * dx)
    return out


def _sup_inf(m: np.ndarray, sup_of_inf: bool) -> np.ndarray:
    """One half of the morphological curvature operator: the sup over the four
    line elements of the inf along each, or the other way round."""
    per_line = []
    for line in range(4):
        along = [_line_shift(m, _LINE_DY[line][k], _LINE_DX[line][k]) for k in range(3)]
        stacked = np.stack(along)
        per_line.append(stacked.min(axis=0) if sup_of_inf else stacked.max(axis=0))
    stacked = np.stack(per_line)
    return (stacked.max(axis=0) if sup_of_inf else stacked.min(axis=0)).astype(np.uint8)


def _morphological_chan_vese_plane(image: np.ndarray, mask: np.ndarray, iterations: int, smoothing: int) -> np.ndarray:
    """``morphologicalChanVesePlane``: the region force of Chan-Vese applied
    only on the contour, then the morphological curvature operator."""
    m = np.asarray(mask, dtype=np.uint8).copy()
    if iterations <= 0 or m.size == 0:
        return m
    values = image.astype(np.float64)
    for it in range(iterations):
        inside = m > 0
        if not inside.any() or inside.all():
            return m
        c1 = float(values[inside].mean())
        c0 = float(values[~inside].mean())
        cur = m.astype(np.int32)
        gy = np.vstack([cur[1:], cur[-1:]]) - np.vstack([cur[:1], cur[:-1]])
        gx = np.hstack([cur[:, 1:], cur[:, -1:]]) - np.hstack([cur[:, :1], cur[:, :-1]])
        on_contour = (gy != 0) | (gx != 0)
        aux = (values - c1) ** 2 - (values - c0) ** 2
        m[on_contour & (aux < 0.0)] = 1
        m[on_contour & (aux > 0.0)] = 0
        for k in range(smoothing):
            first = (it + k) % 2 == 0
            m = _sup_inf(m, first)
            m = _sup_inf(m, not first)
    return m


def _filter_labels_by_shape(labels: np.ndarray, max_voxels: int, min_fill: float, max_elongation: float,
                            drop_border: bool) -> np.ndarray:
    """``filterLabelsByShape``: bounding-box measures, then a dense relabel."""
    max_id = int(labels.max())
    if max_id == 0:
        return labels
    ndimage = _ndimage()
    boxes = ndimage.find_objects(labels, max_label=max_id)
    counts = np.bincount(labels.ravel(), minlength=max_id + 1)
    remap = np.zeros(max_id + 1, dtype=np.uint32)
    nxt = 0
    nz, ny, nx = labels.shape
    for label_id in range(1, max_id + 1):
        box = boxes[label_id - 1]
        voxels = int(counts[label_id])
        if box is None or voxels == 0:
            continue
        if max_voxels > 0 and voxels > max_voxels:
            continue
        if drop_border and (box[1].start == 0 or box[1].stop == ny or box[2].start == 0 or box[2].stop == nx):
            continue
        dz = box[0].stop - box[0].start
        dy = box[1].stop - box[1].start
        dx = box[2].stop - box[2].start
        if min_fill > 0.0:
            volume = float(dz * dy * dx)
            if volume > 0.0 and voxels / volume < min_fill:
                continue
        if max_elongation > 0.0 and max(dy, dx) / max(1, min(dy, dx)) > max_elongation:
            continue
        nxt += 1
        remap[label_id] = nxt
    return remap[labels]


def _clean_plane(m: np.ndarray, opening: int, fill_holes: bool) -> np.ndarray:
    ndimage = _ndimage()
    if opening > 0:
        size = (2 * opening + 1, 2 * opening + 1)
        m = ndimage.grey_dilation(ndimage.grey_erosion(m, size=size, mode="nearest"), size=size, mode="nearest")
    if fill_holes:
        m = ndimage.binary_fill_holes(m).astype(np.uint8)
    return m


_CLASSIC = StepSpec(
    "classic",
    {"channel": 0, "denoise": "None", "diffusion_iterations": 5, "diffusion_k": 0.1,
     "enhance": "None", "enhance_sigma": 2.0, "enhance_sigma_max": 6.0, "enhance_scales": 4,
     "background": "Top-hat (box)", "tophat": 0, "sigma": 1.0, "method": "Otsu", "value": 0.5, "percentile": 90.0, "window": 51,
     "local_ratio": 1.1, "local_offset": 0.0, "contrast_k": 1.5,
     "hysteresis": False, "hysteresis_ratio": 0.5,
     "refine": "None", "refine_iterations": 20, "refine_smoothing": 1,
     "opening": 1, "fill_holes": True, "fill_holes_3d": False, "hole_max_voxels": 0,
     "post": "Watershed (distance)", "seeds": "H-maxima", "seed_distance": 8.0, "seed_depth": 2.0,
     "blob_radius": 4.0, "blob_radius_max": 12.0, "blob_scales": 5, "min_voxels": 20,
     "max_voxels": 0, "min_fill": 0.0, "max_elongation": 0.0, "expand": 0.0, "skeleton": False, "drop_border": False,
     "class_name": "object"},
    choices={"method": ("Otsu", "Triangle", "Li", "Yen", "Isodata", "Multi-Otsu", "Manual", "Percentile",
                        "Local mean", "Local contrast"),
             "denoise": ("None", "Median 3x3", "Anisotropic diffusion"),
             "enhance": ("None", "Blobs (DoG)", "Tubes (Frangi)", "Neurites (Meijering)"),
             "refine": ("None", "Active contour (Chan-Vese)"),
             "background": ("Top-hat (box)", "Rolling ball"),
             "seeds": ("Distance maxima", "H-maxima", "Blob centres (LoG)"),
             "post": ("Watershed (distance)", "Watershed (gradient)", "Connected components")},
    aliases={"input_channel": "channel", "minVoxels": "min_voxels"})


@_step(_CLASSIC)
def step_classic(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any]) -> StepResult:
    """Classical segmentation (classic.cpp): per plane an optional enhancement
    (enhance, enhance_sigma, enhance_sigma_max, enhance_scales), a white top-hat
    (tophat radius), Gaussian (sigma), a global (Otsu | Multi-Otsu | Manual |
    Percentile) or local (Local mean: window, local_ratio, local_offset; Local
    contrast: window, contrast_k) threshold, binary opening and hole filling;
    then 3D instances (post, seeds, seed_distance, seed_depth, min_voxels).
    Also denoise (Median 3x3 | Anisotropic diffusion: diffusion_iterations,
    diffusion_k) before the enhancement, the Triangle and Li thresholds,
    hysteresis (hysteresis_ratio), an active contour refinement (refine,
    refine_iterations, refine_smoothing), the gradient watershed, the shape
    filters max_voxels, min_fill, max_elongation and drop_border, the Yen and
    Isodata thresholds, the Meijering neurite enhancement, a 3D hole fill
    (fill_holes_3d, hole_max_voxels), label expansion (expand), the rolling
    ball background (background) and centrelines (skeleton)."""
    c = _channel_index(params, "channel", meta, a.shape[0])
    denoise = _choice(params.get("denoise"), _CLASSIC.choices["denoise"], "None")
    diffusion_iterations = _int(params, "diffusion_iterations", 5)
    diffusion_k = _float(params, "diffusion_k", 0.1)
    enhance = _choice(params.get("enhance"), _CLASSIC.choices["enhance"], "None")
    enhance_sigma = _float(params, "enhance_sigma", 2.0)
    enhance_sigma_max = _float(params, "enhance_sigma_max", 6.0)
    enhance_scales = _int(params, "enhance_scales", 4)
    background = _choice(params.get("background"), _CLASSIC.choices["background"], "Top-hat (box)")
    tophat = _int(params, "tophat", 0)
    sigma = _float(params, "sigma", 1.0)
    method = _choice(params.get("method"), _CLASSIC.choices["method"], "Otsu")
    window = max(1, _int(params, "window", 51) // 2)
    ratio, offset = _float(params, "local_ratio", 1.1), _float(params, "local_offset", 0.0)
    contrast_k = _float(params, "contrast_k", 1.5)
    hysteresis_ratio = _float(params, "hysteresis_ratio", 0.5)
    hysteresis = _bool(params, "hysteresis", False) and hysteresis_ratio < 1.0
    refine = _choice(params.get("refine"), _CLASSIC.choices["refine"], "None")
    refine_iterations = _int(params, "refine_iterations", 20)
    refine_smoothing = _int(params, "refine_smoothing", 1)
    opening = _int(params, "opening", 1)
    fill_holes = _bool(params, "fill_holes", True)
    fill_holes_3d = _bool(params, "fill_holes_3d", False)
    hole_max_voxels = _int(params, "hole_max_voxels", 0)
    expand = _float(params, "expand", 0.0)
    post = _choice(params.get("post"), _CLASSIC.choices["post"], "Watershed (distance)")
    min_voxels = _int(params, "min_voxels", 20)
    max_voxels = _int(params, "max_voxels", 0)
    min_fill = _float(params, "min_fill", 0.0)
    max_elongation = _float(params, "max_elongation", 0.0)
    drop_border = _bool(params, "drop_border", False)
    skeleton = _bool(params, "skeleton", False)
    shape_filters = max_voxels > 0 or min_fill > 0.0 or max_elongation > 0.0 or drop_border
    seeds = _choice(params.get("seeds"), _CLASSIC.choices["seeds"], "H-maxima")
    seed_distance = _float(params, "seed_distance", 8.0)
    seed_depth = _float(params, "seed_depth", 2.0)
    blob_sigma = max(0.3, _float(params, "blob_radius", 4.0) / math.sqrt(3.0))
    blob_sigma_max = max(blob_sigma, _float(params, "blob_radius_max", 12.0) / math.sqrt(3.0))
    blob_scales = _int(params, "blob_scales", 5)
    voxel = _voxel_um(meta)
    z_aspect = max(1e-6, voxel[2] / voxel[0]) if voxel[0] > 0 else 1.0
    labels = np.zeros((a.shape[1],) + a.shape[2:], dtype=np.uint32)
    cuts: List[Any] = []
    total = 0
    foreground = 0.0
    for t in range(a.shape[1]):
        volume = a[c, t]
        if denoise == "Median 3x3":
            volume = np.stack([_median_plane(pl) for pl in volume])
        elif denoise == "Anisotropic diffusion":
            volume = np.stack([_anisotropic_diffusion_plane(pl, diffusion_iterations, diffusion_k) for pl in volume])
        if enhance == "Tubes (Frangi)":
            # tubes are a 3D filter: a filament running through z is invisible
            # to a plane-by-plane one
            volume = _frangi_volume(volume, z_aspect, enhance_sigma, max(enhance_sigma, enhance_sigma_max), enhance_scales)
        elif enhance == "Neurites (Meijering)":
            volume = _meijering_volume(volume, z_aspect, enhance_sigma, max(enhance_sigma, enhance_sigma_max), enhance_scales)
        planes = []
        for z in range(a.shape[2]):
            pl = volume[z]
            if enhance == "Blobs (DoG)":
                pl = _dog_plane(pl, enhance_sigma)
            if background == "Rolling ball" and tophat > 0:
                pl = _rolling_ball_plane(pl, float(tophat))
                planes.append(_filter_plane(pl, 0, sigma))
            else:
                planes.append(_filter_plane(pl, tophat, sigma))
        work = np.stack(planes)
        low = None
        if method in ("Local mean", "Local contrast"):
            by_contrast = method == "Local contrast"
            rows, low_rows = [], []
            for pl in work:
                if by_contrast:
                    mean, sd = _local_stats_plane(pl, window)
                    cut_plane = mean + contrast_k * sd + offset
                else:
                    mean = _local_mean_plane(pl, window)
                    cut_plane = ratio * mean + offset
                rows.append(pl > cut_plane)
                if hysteresis:
                    low_rows.append(pl > mean + hysteresis_ratio * (cut_plane - mean))
            mask = np.stack(rows).astype(np.uint8)
            if hysteresis:
                low = np.stack(low_rows).astype(np.uint8)
            cuts.append(f"local mean + {contrast_k:g} SD" if by_contrast else f"local mean × {ratio:g}")
        else:
            cut = _global_cut(work, method, params)
            mask = (work > cut).astype(np.uint8)
            if hysteresis:
                floor_value = np.float32(work.min())
                low_cut = floor_value + np.float32(hysteresis_ratio) * (np.float32(cut) - floor_value)
                low = (work > low_cut).astype(np.uint8)
            cuts.append(cut)
        if hysteresis and low is not None:
            mask = _hysteresis_mask(mask, low)
        if refine != "None":
            mask = np.stack([_morphological_chan_vese_plane(work[z], mask[z], refine_iterations, refine_smoothing)
                             for z in range(mask.shape[0])])
        mask = np.stack([_clean_plane(m, opening, fill_holes) for m in mask])
        if fill_holes_3d:
            mask = _fill_holes_3d(mask, hole_max_voxels)
        foreground += float(mask.mean()) / a.shape[1]
        external = None
        if seeds == "Blob centres (LoG)" and post.startswith("Watershed"):
            external, _ = _log_blob_seeds(work, mask.astype(bool), z_aspect, blob_sigma, blob_sigma_max, blob_scales)
        boundary = _gradient_magnitude(work, z_aspect) if post == "Watershed (gradient)" else None
        labels[t] = _labels_from_probabilities(mask.astype(np.float32), boundary, 0.5, post, min_voxels, seed_distance,
                                               seeds, seed_depth, external)
        if shape_filters:
            labels[t] = _filter_labels_by_shape(labels[t], max_voxels, min_fill, max_elongation, drop_border)
        # counted here, as the C++ counts: growing the labels adds none, and
        # thinning them to centrelines must not look like losing one
        total += int(labels[t].max())
        if expand > 0.0:
            labels[t] = _expand_labels(labels[t], expand, z_aspect)
        if skeleton:
            labels[t] = np.where(_skeletonize_3d(labels[t] > 0) > 0, labels[t], 0)
    return StepResult(a, dict(meta), labels=labels,
                      info={"thresholds": cuts, "method": method, "channel": c, "labels": total,
                            "foreground_fraction": foreground, "class_name": _str(params, "class_name", "object")})


_TRACK = StepSpec(
    "track",
    {"tracker": "Built-in (assignment)", "max_distance": 10.0, "overlap_weight": 0.5, "max_gap": 1,
     "min_length": 2, "relabel": True, "config": "", "optimise": True},
    choices={"tracker": ("Built-in (assignment)", "btrack (Bayesian)")},
    needs_labels=True)


def _track_objects(vol: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``objectsOfFrame``: the labels present in one frame, their centroids
    (z, y, x in voxels) and their voxel counts, in ascending label order."""
    ids = np.unique(vol)
    ids = ids[ids > 0]
    if ids.size == 0:
        return ids.astype(np.uint32), np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    ndimage = _ndimage()
    centres = np.asarray(ndimage.center_of_mass(np.ones_like(vol, dtype=np.float32), labels=vol, index=ids), dtype=np.float64)
    counts = np.asarray(ndimage.sum_labels(np.ones_like(vol, dtype=np.float32), labels=vol, index=ids), dtype=np.int64)
    return ids.astype(np.uint32), centres.reshape(-1, 3), counts


def _track_overlap(a_vol: np.ndarray, b_vol: np.ndarray, a_ids: np.ndarray, b_ids: np.ndarray) -> np.ndarray:
    """``overlapBetween``: voxels shared by each (label of t, label of t+1)."""
    counts = np.zeros((a_ids.size, b_ids.size), dtype=np.int64)
    both = (a_vol > 0) & (b_vol > 0)
    if not both.any():
        return counts
    rows = np.searchsorted(a_ids, a_vol[both])
    cols = np.searchsorted(b_ids, b_vol[both])
    np.add.at(counts, (rows, cols), 1)
    return counts


def _solve_assignment(cost: np.ndarray) -> List[int]:
    """``solveAssignment``: minimum-cost matching, with np.inf forbidding a
    pair. Returns one column per row, or -1 where the row stays unmatched."""
    rows, cols = cost.shape
    out = [-1] * rows
    if rows == 0 or cols == 0:
        return out
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
    except ImportError as e:
        raise NotAvailable("tracking needs 'scipy' (pip install scipy)") from e
    big = 1e12
    padded = np.where(np.isfinite(cost), cost, big)
    padded = np.minimum(padded, big)
    r, c = linear_sum_assignment(padded)
    for i, j in zip(r.tolist(), c.tolist()):
        if np.isfinite(cost[i, j]) and cost[i, j] < big:
            out[i] = j
    return out


@_step(_TRACK)
def step_track(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any],
               labels: Optional[np.ndarray] = None) -> StepResult:
    """Track objects (track.cpp) over the labels of a segmentation step
    upstream: frame-to-frame optimal assignment on centroid distance
    (max_distance, in micrometres) mixed with shared voxels (overlap_weight),
    then gap closing (max_gap) and min_length; relabel gives every object of a
    track the track's id. The intensities pass through."""
    if labels is None or not labels.size:
        raise ValueError("Track objects needs labels: add a segmentation step before it")
    tracker = _choice(params.get("tracker"), _TRACK.choices["tracker"], "Built-in (assignment)")
    if tracker.startswith("btrack"):
        # the same backend the application drives through the worker
        try:
            from sirius_worker import tracking as backends  # type: ignore
        except ImportError as e:
            raise NotAvailable("btrack tracking runs in the Python worker (sirius_worker); "
                               "use the built-in tracker, or run this where the worker package is importable") from e
        volume = labels if labels.ndim == 4 else labels[:, np.newaxis]
        out, info = backends.run_btrack(volume, _voxel_um(meta), dict(params))
        return StepResult(a, dict(meta), labels=out.reshape(labels.shape).astype(np.uint32), info=info)
    max_distance = max(1e-9, _float(params, "max_distance", 10.0))
    weight = min(1.0, max(0.0, _float(params, "overlap_weight", 0.5)))
    max_gap = _int(params, "max_gap", 1)
    min_length = max(1, _int(params, "min_length", 2))
    relabel = _bool(params, "relabel", True)
    voxel = _voxel_um(meta)
    scale = np.array([voxel[2], voxel[1], voxel[0]], dtype=np.float64)   # centroid is (z, y, x)

    frames = int(labels.shape[0])
    ids: List[np.ndarray] = []
    centres: List[np.ndarray] = []
    counts: List[np.ndarray] = []
    for t in range(frames):
        i, c, n = _track_objects(labels[t])
        ids.append(i)
        centres.append(c)
        counts.append(n)

    track_of: List[np.ndarray] = [np.full(i.size, -1, dtype=np.int64) for i in ids]
    tracks: List[List[Tuple[int, int]]] = []   # (frame, label)
    links = 0

    def start(t: int, k: int) -> int:
        tracks.append([(t, int(ids[t][k]))])
        track_of[t][k] = len(tracks) - 1
        return len(tracks) - 1

    for t in range(frames - 1):
        if ids[t].size == 0 or ids[t + 1].size == 0:
            continue
        delta = (centres[t][:, None, :] - centres[t + 1][None, :, :]) * scale
        dist = np.sqrt((delta ** 2).sum(axis=2))
        cost = np.where(dist > max_distance, np.inf, dist / max_distance)
        if weight > 0.0:
            shared = _track_overlap(labels[t], labels[t + 1], ids[t], ids[t + 1]).astype(np.float64)
            denom = counts[t][:, None] + counts[t + 1][None, :] - shared
            iou = np.where(denom > 0, shared / np.where(denom > 0, denom, 1.0), 0.0)
            cost = np.where(np.isfinite(cost), (1.0 - weight) * cost + weight * (1.0 - iou), np.inf)
        match = _solve_assignment(cost)
        for i in range(ids[t].size):
            if track_of[t][i] < 0:
                start(t, i)
            j = match[i]
            if j < 0:
                continue
            track_of[t + 1][j] = track_of[t][i]
            tracks[int(track_of[t][i])].append((t + 1, int(ids[t + 1][j])))
            links += 1
    for t in range(frames):
        for i in range(ids[t].size):
            if track_of[t][i] < 0:
                start(t, i)

    gaps = 0
    if max_gap > 0 and len(tracks) > 1:
        def centre_of(track: int, end: bool) -> np.ndarray:
            t, label = tracks[track][-1] if end else tracks[track][0]
            k = int(np.searchsorted(ids[t], label))
            return centres[t][k]

        live = [k for k in range(len(tracks)) if tracks[k]]
        cost = np.full((len(live), len(live)), np.inf)
        for r, frm in enumerate(live):
            for c, to in enumerate(live):
                if frm == to:
                    continue
                gap = tracks[to][0][0] - tracks[frm][-1][0]
                if gap < 1 or gap > max_gap + 1:
                    continue
                d = float(np.sqrt((((centre_of(frm, True) - centre_of(to, False)) * scale) ** 2).sum()))
                if d > max_distance * gap:
                    continue
                cost[r, c] = d / max_distance + 0.25 * (gap - 1)
        match = _solve_assignment(cost)
        # Applied in order. A track that was merged away has had its points
        # moved into another one, so a later link out of it continues from
        # where they went (tracking.cpp does the same): an object missed twice
        # is one track, not the first link and then nothing. Only the target
        # of a link is consumed.
        merged_into = {k: k for k in live}

        def root_of(k: int) -> int:
            while merged_into[k] != k:
                k = merged_into[k]
            return k

        consumed = set()
        for r, c in enumerate(match):
            if c < 0:
                continue
            to = live[c]
            frm = root_of(live[r])
            if frm == to or to in consumed or not tracks[to] or not tracks[frm]:
                continue
            if tracks[to][0][0] <= tracks[frm][-1][0]:
                continue
            tracks[frm].extend(tracks[to])
            tracks[to] = []
            consumed.add(to)
            merged_into[to] = frm
            gaps += 1

    kept = [sorted(t) for t in tracks if len(t) >= min_length]
    kept.sort(key=lambda t: (t[0][0], t[0][1]))
    out = np.zeros_like(labels, dtype=np.uint32)
    for n, track in enumerate(kept, start=1):
        for t, label in track:
            out[t][labels[t] == label] = n if relabel else label
    lengths = [len(t) for t in kept]
    return StepResult(a, dict(meta), labels=out,
                      info={"tracks": len(kept), "links": links, "gaps_closed": gaps,
                            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
                            "longest": int(max(lengths)) if lengths else 0})


_CLEANUP = StepSpec(
    "cleanup",
    {"min_voxels": 50, "remove_border": False, "relabel": True, "low_conf": 0.6, "size_outlier_factor": 4.0},
    aliases={"minVoxels": "min_voxels"}, needs_labels=True)


@_step(_CLEANUP)
def step_cleanup(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any],
                 labels: Optional[np.ndarray] = None) -> StepResult:
    """Label cleanup (cleanup.cpp) on the labels of the segmentation step
    upstream: remove_border, min_voxels, relabel (densely); low_conf and
    size_outlier_factor only set review flags (reported in info["flags"]).
    The intensities pass through."""
    if labels is None or not labels.size:
        raise ValueError("Label cleanup needs labels: add a segmentation step before it")
    min_voxels = _int(params, "min_voxels", 50)
    remove_border = _bool(params, "remove_border", False)
    relabel = _bool(params, "relabel", True)
    low_conf = _float(params, "low_conf", 0.6)
    outlier = _float(params, "size_outlier_factor", 4.0)
    out = np.array(labels, dtype=np.uint32, copy=True)
    flags: Dict[str, List[int]] = {}
    for t in range(out.shape[0]):
        vol = out[t]
        if remove_border:
            drop = _border_labels(vol)
            if drop.size:
                vol[np.isin(vol, drop)] = 0
        if min_voxels > 0 or relabel:
            vol = _remove_small(vol, min_voxels, relabel)
        out[t] = vol
        flags = _label_flags(vol, low_conf, outlier)
    kept = int(np.count_nonzero(np.bincount(out.reshape(-1))[1:]))
    return StepResult(a, dict(meta), labels=out, info={"labels": kept, "flags": flags})


# --- SIM ----------------------------------------------------------------------

_sim_cache: Dict[str, Any] = {}


def _sim_legacy(p: Dict[str, Any], meta: Optional[Dict[str, Any]]) -> None:
    # a parameter file without a mode means "From file" (the older Python
    # export loaded it first); the application ignores it in the other modes
    if p.get("mode") is None and p.get("params_file"):
        p["mode"] = "From file"


_SIM = StepSpec(
    "sim",
    {"mode": "Estimate", "params_file": "", "angles": 3, "phases": 5, "wiener": 0.001, "apodization": "Cosine",
     "otf": "", "na": 1.4, "nimm": 1.515, "wavelength_nm": 510.0, "linespacing_um": 0.2, "k0_angles": [],
     "k0_start_angle": 0.0, "suppress_zero_order": True, "bleach_correction": True,
     "zoomfact": 2.0, "z_zoom": 1, "orders": 0, "dz_psf": 0.0, "otfcutoff": 0.006, "background": 0.0,
     "apodize_input": "Triangle", "napodize": 10, "suppression_radius": 10, "suppress_singularities": True,
     "no_kz0": True, "filter_overlaps": True, "explodefact": 1.0, "equalizez": False},
    choices={"mode": ("Estimate", "Manual", "From file"), "apodization": ("Cosine", "Triangle", "None"),
             "apodize_input": ("Triangle", "Cosine", "None")},
    aliases={"ndirs": "angles", "directions": "angles", "nphases": "phases", "norders": "orders",
             "wavelength": "wavelength_nm", "linespacing": "linespacing_um", "line_spacing": "linespacing_um",
             "k0angle": "k0_start_angle", "k0_angle": "k0_start_angle", "k0angles": "k0_angles", "zoom": "zoomfact",
             "do_rescale": "bleach_correction", "bleach": "bleach_correction",
             "dampen_order0": "suppress_zero_order", "suppress_zero": "suppress_zero_order",
             "apodize_output": "apodization", "otf_cutoff": "otfcutoff", "immersion": "nimm",
             "otf_path": "otf", "otf_file": "otf", "config": "params_file", "parameter_file": "params_file"},
    extra=("dx", "dy", "dz", "fast_si"),   # SIMParameters fields the metadata normally provides
    translate=_sim_legacy)


def _sirius_ext():
    try:
        import sirius  # type: ignore
    except ImportError as e:
        raise NotAvailable("SIM reconstruction needs the 'sirius' extension") from e
    if not hasattr(sirius, "SimReconstructor"):
        raise NotAvailable("SIM reconstruction needs the 'sirius' extension (the installed package has no SimReconstructor)")
    return sirius


def _sim_parameters(params: Dict[str, Any], meta: Dict[str, Any]):
    """SIMParameters as sim.cpp's buildParameters assembles them."""
    sirius = _sirius_ext()
    apod = {"None": sirius.ApodizationType.None_, "Cosine": sirius.ApodizationType.Cosine,
            "Triangle": sirius.ApodizationType.Triangle}
    mode = _choice(params.get("mode"), _SIM.choices["mode"], "Estimate")
    if mode == "From file":
        cfg = _str(params, "params_file")
        if not cfg:
            raise ValueError("SIM: From file mode needs a parameter file ('params_file')")
        try:
            p = sirius.load_parameters(cfg) if cfg.lower().endswith(".toml") else sirius.load_legacy_parameters(cfg)
        except Exception:  # noqa: BLE001 - try the other format
            p = sirius.load_legacy_parameters(cfg)
    else:
        p = sirius.SIMParameters()
        p.ndirs = _int(params, "angles", 3)
        p.nphases = _int(params, "phases", 5)
        orders = _int(params, "orders", 0)
        p.norders = orders if orders > 0 else p.nphases // 2 + 1
        p.wiener = _float(params, "wiener", 0.001)
        p.apodize_output = apod[_choice(params.get("apodization"), list(apod), "Cosine")]
        p.apodize_input = apod[_choice(params.get("apodize_input"), list(apod), "Triangle")]
        p.na = _float(params, "na", 1.4)
        p.nimm = _float(params, "nimm", 1.515)
        p.wavelength_nm = _float(params, "wavelength_nm", 510.0)
        p.linespacing_um = _float(params, "linespacing_um", 0.2)
        # degrees in the parameters, as the fit table reports them; radians in
        # the library, converted once here as the application does
        p.k0_start_angle = _float(params, "k0_start_angle", 0.0) * math.pi / 180.0
        if mode == "Manual":
            angles = [a * math.pi / 180.0 for a in _floats(params.get("k0_angles"))]
            if angles:
                p.k0_angles = angles
        p.dampen_order0 = _bool(params, "suppress_zero_order", True)
        p.do_rescale = _bool(params, "bleach_correction", True)
        p.zoomfact = _float(params, "zoomfact", 2.0)
        p.z_zoom = _int(params, "z_zoom", 1)
        p.otfcutoff = _float(params, "otfcutoff", 0.006)
        p.background = _float(params, "background", 0.0)
        p.napodize = _int(params, "napodize", 10)
        p.suppression_radius = _int(params, "suppression_radius", 10)
        p.suppress_singularities = _bool(params, "suppress_singularities", True)
        p.no_kz0 = _bool(params, "no_kz0", True)
        p.filter_overlaps = _bool(params, "filter_overlaps", True)
        p.explodefact = _float(params, "explodefact", 1.0)
        p.equalizez = _bool(params, "equalizez", False)
        sim = meta.get("sim") or {}
        p.fast_si = bool(sim.get("present") and sim.get("fast_si"))
    vx, vy, vz = _voxel_um(meta)
    p.dx, p.dy, p.dz = float(vx), float(vy), float(vz)
    for k in ("dx", "dy", "dz"):
        if params.get(k) is not None:
            setattr(p, k, _float(params, k, getattr(p, k)))
    if params.get("fast_si") is not None:
        p.fast_si = _bool(params, "fast_si", p.fast_si)
    dz_psf = _float(params, "dz_psf", 0.0)
    p.dz_psf = dz_psf if dz_psf > 0.0 else p.dz
    p.validate()
    return p


def _params_key(p) -> Dict[str, str]:
    out = {}
    for k in dir(p):
        if k.startswith("_") or k == "validate":
            continue
        v = getattr(p, k)
        if not callable(v):
            out[k] = str(v)
    return out


def _to_numpy(result) -> np.ndarray:
    """numpy from what SimReconstructor.reconstruct returned (numpy on the CPU,
    a DLPack sirius.Buffer on CUDA)."""
    if isinstance(result, np.ndarray):
        return result
    if hasattr(result, "numpy"):          # sirius.Buffer copies device memory to the host
        return np.asarray(result.numpy())
    try:
        import torch  # type: ignore

        return torch.from_dlpack(result).cpu().numpy()
    except ImportError as e:
        raise NotAvailable("reading a CUDA sirius.Buffer back needs 'torch' (DLPack)") from e


@_step(_SIM)
def step_sim(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any], progress: ProgressFn = None,
             cancelled: CancelFn = None, device: str = "cpu") -> StepResult:
    """Structured illumination reconstruction of a raw stack whose z axis holds
    angles * phases * nz sections, with the application's SIM step parameters
    (mode: Estimate | Manual | From file (params_file); angles, phases, wiener,
    apodization, na, nimm, wavelength_nm, linespacing_um, k0_start_angle /
    k0_angles, ...). Needs a measured `otf` file: the theoretical OTF exists
    only in the application."""
    sirius = _sirius_ext()
    p = _sim_parameters(params, meta)
    otf = _str(params, "otf")
    if not otf:
        raise NotAvailable("SIM reconstruction in Python needs a measured OTF file ('otf'); "
                           "the theoretical OTF is only available in the application")
    if not os.path.exists(otf):
        raise FileNotFoundError(f"OTF file not found: {otf}")
    sections = p.ndirs * p.nphases
    if a.shape[2] % sections:
        raise ValueError(f"{a.shape[2]} sections is not a multiple of angles × phases = {sections}")
    device = resolve_device(device)
    use_cuda = device.startswith("cuda") and sirius.cuda_available()
    dev = sirius.Device.cuda(int(device.split(":")[1]) if ":" in device else 0) if use_cuda else sirius.Device.cpu()
    key = json.dumps({"otf": os.path.abspath(otf), "dev": str(dev), "p": _params_key(p)}, sort_keys=True)
    recon = _sim_cache.get(key)
    if recon is None:
        _sim_cache.clear()
        recon = sirius.SimReconstructor(p, otf, dev)
        _sim_cache[key] = recon
    out = None
    fits = []
    n = a.shape[0] * a.shape[1]
    k = 0
    for c in range(a.shape[0]):
        for t in range(a.shape[1]):
            _check_cancel(cancelled)
            _progress(progress, k / max(n, 1), f"reconstructing c{c} t{t}")
            raw = np.ascontiguousarray(a[c, t], dtype=np.float64)
            src = sirius.to_device(raw, dev) if use_cuda else raw
            res = _to_numpy(recon.reconstruct(src)).astype(np.float32, copy=False)
            if out is None:
                out = np.empty(a.shape[:2] + res.shape, dtype=np.float32)
            out[c, t] = res
            fit = recon.last_fit
            fits.append({"k0": [[float(v[0]), float(v[1])] for v in fit.k0],
                         "amps": [[[float(x.real), float(x.imag)] for x in row] for row in fit.amps]})
            k += 1
    assert out is not None
    vx, vy, vz = _voxel_um(meta)
    m = dict(meta, dims=_dims(out), voxel_um=[vx / p.zoomfact, vy / p.zoomfact, vz / max(p.z_zoom, 1)],
             sim={"present": False, "ndirs": p.ndirs, "nphases": p.nphases, "fast_si": p.fast_si})
    _progress(progress, 1.0, "done")
    return StepResult(out, m, info={"fits": fits, "wiener": p.wiener, "ndirs": p.ndirs, "nphases": p.nphases,
                                    "device": str(dev)})


# --- Torch segmentation ------------------------------------------------------


def _torch():
    try:
        import torch  # type: ignore
    except ImportError as e:
        raise NotAvailable("Torch models need the 'torch' package") from e
    return torch


def resolve_device(device: str = "auto") -> str:
    """'auto' -> 'cuda' when torch sees a GPU, else 'cpu'."""
    device = (device or "auto").lower()
    if device == "auto":
        try:
            return "cuda" if _torch().cuda.is_available() else "cpu"
        except NotAvailable:
            return "cpu"
    return device


_model_cache: Dict[Tuple[str, str], Any] = {}

# Model specs beyond a file path (the worker's ``sirius_worker.models`` has
# the full hub / family machinery; the layout of the download cache is shared):
#   hf:<owner>/<repo>[:<file>]   a Hugging Face file, downloaded once into
#                                $SIRIUS_MODEL_CACHE or ~/.sirius/models/hf/<owner>--<repo>/
#   cellpose:<model>, microsam:<model_type>   package families returning labels
_MODEL_EXTENSIONS = (".pt", ".pts", ".pth", ".onnx")
_FAMILY_PREFIXES = ("cellpose:", "microsam:", "micro-sam:", "micro_sam:")


def model_cache_dir() -> str:
    env = os.environ.get("SIRIUS_MODEL_CACHE", "").strip()
    return os.path.expanduser(env) if env else os.path.join(os.path.expanduser("~"), ".sirius", "models")


def _is_hf_spec(spec: str) -> bool:
    return spec.lower().startswith(("hf:", "huggingface:"))


def _is_family_spec(spec: str) -> bool:
    return spec.lower().startswith(_FAMILY_PREFIXES)


def _family_of(spec: str) -> Tuple[str, str]:
    """('cellpose' | 'microsam', model name) of a family spec."""
    prefix, _, name = spec.partition(":")
    family = "cellpose" if prefix.lower() == "cellpose" else "microsam"
    return family, name.strip()


def resolve_model_spec(spec: str, progress: ProgressFn = None) -> str:
    """A local file path for a path or ``hf:`` spec (downloading the file into
    the model cache when needed). Family specs are returned unchanged."""
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("no model given")
    if _is_family_spec(spec) or not _is_hf_spec(spec):
        return spec
    body = spec.split(":", 1)[1].lstrip("/")
    repo, _, filename = body.partition(":")
    repo = repo.strip().strip("/")
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise ValueError(f"model spec '{spec}': expected hf:<owner>/<repo>[:<filename>]")
    filename = filename.strip()
    target = os.path.join(model_cache_dir(), "hf", repo.replace("/", "--"))
    if filename and os.path.isfile(os.path.join(target, filename)):
        return os.path.join(target, filename)
    try:
        from huggingface_hub import HfApi, hf_hub_download  # type: ignore
    except ImportError as e:
        raise NotAvailable("hf: model specs need the 'huggingface_hub' package (pip install huggingface_hub)") from e
    if not filename:
        names = [s.rfilename for s in (HfApi().model_info(repo).siblings or [])
                 if s.rfilename.lower().endswith(_MODEL_EXTENSIONS)]
        if len(names) != 1:
            raise ValueError(f"hf:{repo} holds {len(names)} model files ({', '.join(names[:10]) or 'none'}); "
                             f"choose one as hf:{repo}:<filename>")
        filename = names[0]
    _progress(progress, 0.0, f"downloading {filename}")
    os.makedirs(target, exist_ok=True)
    path = hf_hub_download(repo_id=repo, filename=filename, local_dir=target)
    _progress(progress, 1.0, filename)
    return os.path.abspath(path)


def _family_info(spec: str) -> Dict[str, Any]:
    family, name = _family_of(spec)
    module = "cellpose" if family == "cellpose" else "micro_sam"
    hint = "pip install cellpose" if family == "cellpose" else "conda install -c conda-forge micro_sam"
    try:
        __import__(module)
        available = True
    except ImportError:
        available = False
    return {"path": spec, "spec": spec, "format": "cellpose" if family == "cellpose" else "micro-sam",
            "family": family, "model": name, "available": available, "install_hint": "" if available else hint,
            "returns": "labels", "dtype": "float32", "input_shape": [1, 1, -1, -1, -1],
            "output_shape": ["labels", -1, -1, -1]}


class _OnnxModel:
    """Callable wrapper so ONNX sessions look like a TorchScript module taking numpy."""

    def __init__(self, session):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.input_shape = session.get_inputs()[0].shape
        self.output_shape = session.get_outputs()[0].shape

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: np.asarray(x, dtype=np.float32)})[0]


def load_model(path: str, device: str = "auto", progress: ProgressFn = None):
    """Load a TorchScript (.pt / .pth / .ts) or ONNX (.onnx, via onnxruntime)
    model once per (path, device). ``path`` may be an ``hf:`` spec (downloaded
    first); ``cellpose:`` / ``microsam:`` specs are not files and run through
    ``sirius_worker.models.run_family`` instead."""
    if _is_family_spec(path):
        family, _ = _family_of(path)
        raise NotAvailable(f"'{path}' is a {family} model family spec: it returns labels through the worker "
                           "(sirius_worker.models.run_family), not a loadable tensor model")
    device = resolve_device(device)
    key = (path.strip(), device)
    m = _model_cache.get(key)
    if m is not None:
        return m
    path = resolve_model_spec(path, progress)
    key = (os.path.abspath(path), device)
    m = _model_cache.get(key)
    if m is not None:
        return m
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".onnx"):
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as e:
            raise NotAvailable("ONNX models need 'onnxruntime'") from e
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda")
                     else ["CPUExecutionProvider"])
        m = _OnnxModel(ort.InferenceSession(path, providers=providers))
    else:
        torch = _torch()
        try:
            m = torch.jit.load(path, map_location=device)
        except Exception as e:  # noqa: BLE001 - say what the file is instead of PytorchStreamReader details
            raise ValueError(_not_torchscript_message(path, e)) from e
        m.eval()
    _model_cache[key] = m
    return m


def _not_torchscript_message(path: str, error: BaseException) -> str:
    """A checkpoint / state dict / safetensors file is not a runnable model."""
    name = os.path.basename(path)
    low = name.lower()
    detail = str(error).splitlines()[0][:120] if str(error).strip() else error.__class__.__name__
    if low.endswith(".safetensors"):
        what = "a safetensors weights file"
    elif "constants.pkl" in detail or low.endswith((".bin", ".ckpt")) or "state_dict" in low:
        what = "a checkpoint / state dict (weights only)"
    elif "central directory" in detail or "zip archive" in detail:
        what = "not a TorchScript archive"
    else:
        what = "not loadable as TorchScript"
    return (f"{name} is {what}: SIRIUS runs TorchScript or ONNX files that carry the model's code. "
            f"Export the model with torch.jit.trace / torch.jit.script (or to ONNX), or pick a model family "
            f"(cellpose:default, microsam:vit_b_lm) from the hub. ({detail})")


def model_info(path: str, device: str = "cpu") -> Dict[str, Any]:
    """format, input_shape, output_shape, dtype, size_bytes, channels_out of a
    model file. TorchScript shapes are probed with a (1, 1, 16, 32, 32) tensor.
    Family specs report {format, available, install_hint} without loading."""
    if _is_family_spec(path):
        return _family_info(path)
    spec = path.strip()
    m = load_model(spec, device)
    path = resolve_model_spec(spec)
    info: Dict[str, Any] = {"path": path, "spec": spec, "size_bytes": os.path.getsize(path), "dtype": "float32"}
    if isinstance(m, _OnnxModel):
        info["format"] = "ONNX"
        info["input_shape"] = [int(s) if isinstance(s, int) else -1 for s in m.input_shape]
        info["output_shape"] = [int(s) if isinstance(s, int) else -1 for s in m.output_shape]
        out = info["output_shape"]
        info["channels_out"] = out[1] if len(out) > 1 and out[1] > 0 else -1
        return info
    torch = _torch()
    info["format"] = "TorchScript"
    dev = resolve_device(device)
    probe = torch.zeros((1, 1, 16, 32, 32), dtype=torch.float32, device=dev)
    info["input_shape"] = [1, 1, -1, -1, -1]
    with torch.no_grad():
        try:
            y = m(probe)
        except Exception as e:  # noqa: BLE001 - report instead of failing model_info
            info["output_shape"] = [1, -1, -1, -1, -1]
            info["channels_out"] = -1
            info["probe_error"] = str(e).splitlines()[0][:200]
            return info
    y = y[0] if isinstance(y, (tuple, list)) else y
    info["output_shape"] = [1, int(y.shape[1]), -1, -1, -1] if y.dim() == 5 else [int(s) for s in y.shape]
    info["channels_out"] = int(y.shape[1]) if y.dim() >= 2 else 1
    return info


def _blend_window(shape: Sequence[int], overlap: Sequence[int]) -> np.ndarray:
    """Separable raised-cosine window: 1 in the tile core, tapering over each
    overlap band, so overlapping predictions cross-fade without seams."""
    w = np.ones(tuple(shape), dtype=np.float32)
    for ax, (n, o) in enumerate(zip(shape, overlap)):
        o = int(min(max(o, 0), n // 2))
        prof = np.ones(n, dtype=np.float32)
        if o > 0:
            ramp = (0.5 - 0.5 * np.cos(np.pi * (np.arange(o) + 0.5) / o)).astype(np.float32)
            prof[:o] = ramp
            prof[n - o:] = ramp[::-1]
        prof = np.maximum(prof, 1e-3)
        view = [1] * len(shape)
        view[ax] = n
        w *= prof.reshape(view)
    return w


def _activation(out: np.ndarray, activation: str) -> np.ndarray:
    activation = (activation or "auto").lower()
    if activation == "auto":
        if out.size == 0 or (out.min() >= 0.0 and out.max() <= 1.0):
            return out
        activation = "softmax" if out.shape[0] > 1 else "sigmoid"
    if activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-out))
    if activation == "softmax":
        e = np.exp(out - out.max(axis=0, keepdims=True))
        return e / e.sum(axis=0, keepdims=True)
    return out


def tiled_inference(volume: np.ndarray, model, tile: Sequence[int] = (32, 256, 256),
                    overlap: Sequence[int] = (4, 32, 32), device: str = "auto", pad_to: int = 1,
                    activation: str = "auto", normalize: bool = True, progress: ProgressFn = None,
                    cancelled: CancelFn = None) -> np.ndarray:
    """Run a (1, 1, z, y, x) -> (1, C, z, y, x) model over `volume` (z, y, x)
    tile by tile with overlap blending. Returns (C, z, y, x) float32 probabilities."""
    volume = np.ascontiguousarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"tiled_inference expects (z, y, x), got {volume.shape}")
    if normalize:
        lo, hi = _percentiles(volume, 1.0, 99.9)
        if hi <= lo:
            hi = lo + 1.0
        volume = np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    shape = volume.shape
    tile = [int(min(max(int(t), 1), n)) for t, n in zip(tile, shape)]
    overlap = [int(min(max(int(o), 0), t // 2)) for o, t in zip(overlap, tile)]
    starts = []
    for n, t, o in zip(shape, tile, overlap):
        if t >= n:
            starts.append([0])
            continue
        step = max(t - o, 1)
        pos = list(range(0, n - t, step)) + [n - t]
        starts.append(sorted(set(pos)))
    tiles = [(z0, y0, x0) for z0 in starts[0] for y0 in starts[1] for x0 in starts[2]]
    window = _blend_window(tile, overlap)
    is_onnx = isinstance(model, _OnnxModel)
    torch = None if is_onnx else _torch()
    dev = resolve_device(device)
    acc: Optional[np.ndarray] = None
    weight = np.zeros(shape, dtype=np.float32)
    for i, (z0, y0, x0) in enumerate(tiles):
        _check_cancel(cancelled)
        _progress(progress, i / len(tiles), f"tile {i + 1} / {len(tiles)}")
        patch = volume[z0:z0 + tile[0], y0:y0 + tile[1], x0:x0 + tile[2]]
        patch_in = patch
        if pad_to > 1:
            pads = [(0, (-n) % pad_to) for n in patch.shape]
            if any(p[1] for p in pads):
                mode = "reflect" if all(n > 1 for n in patch.shape) else "edge"
                patch_in = np.pad(patch, pads, mode=mode)
        x = patch_in[np.newaxis, np.newaxis]
        if is_onnx:
            y = np.asarray(model(x))
        else:
            with torch.no_grad():  # type: ignore[union-attr]
                yt = model(torch.from_numpy(np.ascontiguousarray(x)).to(dev))  # type: ignore[union-attr]
                yt = yt[0] if isinstance(yt, (tuple, list)) else yt
                y = yt.float().cpu().numpy()
        if y.ndim == 4:
            y = y[np.newaxis]
        if y.ndim != 5:
            raise ValueError(f"model returned shape {y.shape}, expected (1, C, z, y, x)")
        y = y[0][:, : patch.shape[0], : patch.shape[1], : patch.shape[2]]
        if acc is None:
            acc = np.zeros((y.shape[0],) + shape, dtype=np.float32)
        w = window[: patch.shape[0], : patch.shape[1], : patch.shape[2]]
        sl = (slice(z0, z0 + patch.shape[0]), slice(y0, y0 + patch.shape[1]), slice(x0, x0 + patch.shape[2]))
        acc[(slice(None),) + sl] += y * w
        weight[sl] += w
    assert acc is not None
    acc /= np.maximum(weight, 1e-6)
    _progress(progress, 1.0, "tiles done")
    return _activation(acc, activation).astype(np.float32, copy=False)


_SEG = StepSpec(
    "seg",
    {"model": "", "input_channel": 0, "tile": [32.0, 256.0, 256.0], "overlap": 32, "threshold": 0.5,
     "post": "Watershed on boundary channel", "min_voxels": 0, "label_opacity": 0.45, "class_name": "nucleus",
     "seed_distance": 5.0},
    choices={"post": ("Watershed on boundary channel", "Connected components", "None (raw probabilities)")},
    aliases={"channel": "input_channel", "model_path": "model", "torch_model": "model", "tile_size": "tile",
             "tau": "threshold", "post_processing": "post", "postprocess": "post", "minVoxels": "min_voxels"},
    # Python-only: how the model runs (the worker's torch_segment takes the
    # same), which probability channels are foreground / boundary, and the
    # options of the cellpose: / microsam: model families
    extra=("normalize", "activation", "pad_to", "fg_channel", "boundary_channel", "diameter", "do_3d",
           "anisotropy", "flow_threshold", "cellprob_threshold", "stitch_threshold", "mode", "amg", "checkpoint"))


@_step(_SEG)
def step_seg(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any], progress: ProgressFn = None,
             cancelled: CancelFn = None, device: str = "auto") -> StepResult:
    """Torch segmentation: model (a file or hub / family spec), input_channel,
    tile [z, y, x], overlap, threshold, post (Watershed on boundary channel |
    Connected components | None (raw probabilities)), min_voxels, seed_distance,
    class_name; label_opacity is display-only. Probability channel 0 is the
    foreground and channel 1 (when present) the boundary, as in the
    application; fg_channel / boundary_channel override that here."""
    path = _str(params, "model")
    if not path:
        raise ValueError("segmentation: no model given")
    c = _channel_index(params, "input_channel", meta, a.shape[0])
    tile = [int(v) for v in _as_list(params.get("tile"), 3, [32, 256, 256])]
    ov = params.get("overlap", 32)
    if isinstance(ov, (list, tuple, str)):
        overlap = [int(v) for v in _as_list(ov, 3, [4, 32, 32])]
    else:
        overlap = [max(1, int(ov) // 8), int(ov), int(ov)]
    threshold = _float(params, "threshold", 0.5)
    post = _choice(params.get("post"), _SEG.choices["post"], "Watershed on boundary channel")
    min_voxels = _int(params, "min_voxels", 0)
    seed_distance = _float(params, "seed_distance", 5.0)
    labels = np.zeros((a.shape[1],) + a.shape[2:], dtype=np.uint32)
    prob_last = None
    nt = a.shape[1]
    if _is_family_spec(path):
        # cellpose: / microsam: models return instance labels themselves
        try:
            from sirius_worker import models as hub  # type: ignore
        except ImportError as e:
            raise NotAvailable(f"'{path}' needs the sirius_worker package (app/python) on the Python path") from e
        for t in range(nt):
            lab, prob = hub.run_family(path, a[c, t], params, device,
                                       progress=lambda f, m, _t=t: _progress(progress, (_t + f) / nt, m),
                                       cancelled=cancelled)
            labels[t] = _remove_small(np.asarray(lab, dtype=np.uint32), min_voxels)
            prob_last = prob
        return StepResult(a, dict(meta), labels=labels, prob=prob_last,
                          info={"model": path, "channels_out": int(prob_last.shape[0]) if prob_last is not None else 0,
                                "labels": int(labels.max()), "post": "model labels"})
    model = load_model(path, device, progress)
    for t in range(nt):
        prob = tiled_inference(a[c, t], model, tile, overlap, device, _int(params, "pad_to", 1),
                               _str(params, "activation", "auto"), _bool(params, "normalize", True),
                               progress=lambda f, m, _t=t: _progress(progress, (_t + f) / nt, m),
                               cancelled=cancelled)
        fg = min(max(_int(params, "fg_channel", 0), 0), prob.shape[0] - 1)
        bd = _int(params, "boundary_channel", 1 if prob.shape[0] > 1 else -1)
        boundary = prob[bd] if 0 <= bd < prob.shape[0] and bd != fg else None
        labels[t] = _labels_from_probabilities(prob[fg], boundary, threshold, post, min_voxels, seed_distance)
        prob_last = prob
    return StepResult(a, dict(meta), labels=labels, prob=prob_last,
                      info={"model": path, "channels_out": int(prob_last.shape[0]) if prob_last is not None else 0,
                            "labels": int(labels.max()), "post": post})


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------

# The Load step's parameters (run_pipeline reads them; the step itself is the
# dataset loader, not a run_step kind).
_LOAD = StepSpec(
    "load",
    {"path": "", "read_as": "Lazy (chunk on demand)", "tile": 0, "page_order": "czt", "c": 0, "t": 0, "z": 0,
     "voxel_x": 0.0, "voxel_y": 0.0, "voxel_z": 0.0, "sim_ndirs": 0, "sim_nphases": 0, "sim_fast": False,
     "sheet_angle": 0.0},
    choices={"read_as": ("Lazy (chunk on demand)", "Full load to RAM")},
    aliases={"pageOrder": "page_order", "channels": "c", "timepoints": "t", "planes": "z",
             "ndirs": "sim_ndirs", "nphases": "sim_nphases", "fast_si": "sim_fast"},
    translate=lambda p, meta: [p.setdefault(k, v) for k, v in
                               zip(("voxel_x", "voxel_y", "voxel_z"),
                                   _as_list(_pop_first(p, ("voxel", "voxel_um")), 3, [0, 0, 0]))]
    if any(k in p for k in ("voxel", "voxel_um")) else None)
_SPECS["load"] = _LOAD

_KIND_ALIASES = {
    "max_projection": "maxproj", "max": "maxproj", "mean_over_time": "meant", "mean_t": "meant",
    "crop": "croppad", "crop_pad": "croppad", "flat_field": "flatfield", "torch": "seg",
    "torch_segment": "seg", "segmentation": "seg", "sim_reconstruction": "sim",
    "label_cleanup": "cleanup", "classical": "classic", "classical_segmentation": "classic",
}
# Kinds only the application implements: run_step raises NotAvailable.
_UNSUPPORTED = {
    "decon": "Richardson-Lucy deconvolution", "deskew": "deskew + rotate",
    "volrec": "volume reconstruction (a display-level rendering; it also resamples the grid)",
    "stitch": "tile stitching", "register": "registration",
    "skimage_seg": "the scikit-image methods (they run in the worker, not here)",
}
# Steps run_pipeline skips: Load is the dataset loader itself.
_PASSTHROUGH = {"load"}


def step_kinds() -> List[str]:
    """Kinds run_step implements here (canonical names)."""
    return sorted(_STEPS)


def step_spec(kind: str) -> StepSpec:
    """The parameter declaration (canonical keys, defaults, choices, aliases)
    of a kind; KeyError for kinds without one."""
    return _SPECS[_KIND_ALIASES.get(kind, kind)]


def run_step(kind: str, params: Dict[str, Any], array: np.ndarray, meta: Optional[Dict[str, Any]] = None,
             labels: Optional[np.ndarray] = None, progress: ProgressFn = None, cancelled: CancelFn = None,
             device: str = "auto") -> StepResult:
    """Run one step on a (c, t, z, y, x) array. Raises NotAvailable for kinds
    only the C++ application implements. Parameter keys are the
    application's; unknown keys raise an UnknownParameterWarning."""
    a = _as5(array)
    meta = dict(meta) if meta else _default_meta(a)
    meta["dims"] = _dims(a)
    k = _KIND_ALIASES.get(kind, kind)
    fn = _STEPS.get(k)
    if fn is None:
        what = _UNSUPPORTED.get(k, k)
        raise NotAvailable(f"step '{kind}' ({what}) is not implemented in Python; run it in the SIRIUS application")
    spec = _SPECS[k]
    p = _prepare_params(spec, params, meta)
    kwargs: Dict[str, Any] = {}
    if k in ("sim", "seg"):
        kwargs.update(progress=progress, cancelled=cancelled, device=device)
    if spec.needs_labels:
        kwargs["labels"] = labels
    res = fn(a, p, meta, **kwargs)
    if res.labels is None and labels is not None and res.array.shape[1:] == a.shape[1:]:
        res.labels = labels
    res.meta["dims"] = _dims(res.array)
    return res


def run_pipeline(dataset_path: str, pipeline: Any, progress: ProgressFn = None, device: str = "auto",
                 strict: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load `dataset_path` and run the pipeline (the app's JSON: {"steps": [...]}
    or a list of steps, each {"kind", "params", "enabled"}). Returns the final
    (c, t, z, y, x) float32 array and its metadata (with "labels" when a
    segmentation step produced them). Steps the Python side cannot run raise
    NotAvailable, or are skipped with their kinds in meta["skipped"] when
    strict is False.
    """
    if isinstance(pipeline, str):
        pipeline = json.loads(pipeline)
    steps = pipeline.get("steps", pipeline) if isinstance(pipeline, dict) else pipeline
    load_params: Dict[str, Any] = {}
    for s in steps:
        if s.get("kind") == "load":
            load_params = s.get("params", {}) or {}
            break
    lp = _prepare_params(_LOAD, load_params, None)
    order = _str(lp, "page_order", "czt") or "czt"
    counts = [_int(lp, k, 0) or None for k in ("c", "t", "z")]
    array, meta = load_dataset(dataset_path, order, counts[0], counts[1], counts[2],
                               progress=lambda f, m: _progress(progress, 0.0, m))
    voxel = [_float(lp, k, 0.0) for k in ("voxel_x", "voxel_y", "voxel_z")]
    meta["voxel_um"] = [v if v > 0 else cur for v, cur in zip(voxel, meta["voxel_um"])]
    if _int(lp, "sim_ndirs", 0) > 0 and _int(lp, "sim_nphases", 0) > 0:
        meta["sim"] = {"present": True, "ndirs": _int(lp, "sim_ndirs", 0), "nphases": _int(lp, "sim_nphases", 0),
                       "fast_si": _bool(lp, "sim_fast", False)}
    labels: Optional[np.ndarray] = None
    skipped: List[str] = []
    n = max(len(steps), 1)
    for i, s in enumerate(steps):
        kind = s.get("kind", "")
        name = s.get("name") or kind
        if kind == "load" or not s.get("enabled", True) or kind in _PASSTHROUGH:
            continue
        _progress(progress, i / n, name)
        try:
            res = run_step(kind, s.get("params", {}) or {}, array, meta, labels,
                           progress=lambda f, m, _i=i, _name=name: _progress(progress, (_i + f) / n, f"{_name}: {m}"),
                           cancelled=None, device=device)
        except NotAvailable:
            if strict:
                raise
            skipped.append(kind)
            continue
        array, meta = res.array, res.meta
        if res.labels is not None:
            labels = res.labels
    if labels is not None:
        meta["labels"] = labels
    if skipped:
        meta["skipped"] = skipped
    _progress(progress, 1.0, "done")
    return array, meta
