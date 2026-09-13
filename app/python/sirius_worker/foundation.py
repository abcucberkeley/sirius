"""The latents foundation model, run for the application.

Everything else the worker serves takes one 3-D volume of one channel and gives
back either probabilities or labels. This model is 5-D: it accepts a colour
channel and a time axis together, and it is the same weights whether the input
is a plane, a volume or a clip. The narrower contracts cannot express that, so
it gets its own kind rather than another branch inside `torch_segment`.

The model arrives as a **bundle** (a single .ltb file) rather than a bare
TorchScript or ONNX graph. The reason is that the weights alone are not enough
to reproduce a result: the peak threshold, the minimum separation between two
objects, the intensity normalization and the voxel size the distances were
calibrated at were all chosen on held-out data at training time, and a guessed
peak threshold is the difference between an F1 of 0.9 and an F1 of 0.03. The
bundle carries them beside the weights, and the parameters shown in the
application default to what the bundle says.

Returned to the application, per timepoint or for the whole clip:

    labels      (t, z, y, x) uint32, 0 = background; for a tracking run one id
                names the same object at every timepoint, which is how this
                application represents a track
    confidence  (t, z, y, x) float32, the model's centroid probability; the
                application folds it into per-label confidence
    lineage     JSON, {child track id: parent track id}; divisions only

Requires the `latents` package. Point SIRIUS_LATENTS_PATH at a checkout if it
is not installed.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

_BUNDLES: Dict[Tuple[str, float, str], Any] = {}


def _import_latents():
    """Import `latents.deploy`, with a message that says what to do if absent.

    A worker that cannot find the package is the most likely failure in a fresh
    install, and "ModuleNotFoundError: latents" on its own tells a microscopist
    nothing actionable."""
    extra = os.environ.get("SIRIUS_LATENTS_PATH", "")
    if extra and extra not in sys.path:
        sys.path.insert(0, extra)
    try:
        from latents import deploy                       # noqa: PLC0415
    except ImportError as exc:                           # pragma: no cover - install-dependent
        raise RuntimeError(
            "The foundation model needs the 'latents' package, which is not importable. "
            "Install it (pip install latents) or set SIRIUS_LATENTS_PATH to a checkout. "
            f"({exc})") from exc
    return deploy


def load_bundle(path: str, device: str = "auto"):
    """Load and cache a bundle. Re-reading a 500 MB file per timepoint would
    dominate the run, and the application calls once per timepoint."""
    deploy = _import_latents()
    if not path:
        raise ValueError("no model given: choose a .ltb bundle")
    if not os.path.exists(path):
        raise FileNotFoundError(f"model bundle not found: {path}")
    key = (os.path.abspath(path), os.path.getmtime(path), device)
    got = _BUNDLES.get(key)
    if got is None:
        dev = None if device in ("auto", "") else device
        got = deploy.Bundle.load(path, device=dev)
        _BUNDLES.clear()                                  # one model resident at a time
        _BUNDLES[key] = got
    return got


def model_info(path: str) -> Dict[str, Any]:
    """What the bundle says about itself, for the model dialog and for the
    application's parameter defaults."""
    m = load_bundle(path, "cpu")
    man = m.m
    return {
        "format": "latents-bundle",
        "name": man.name,
        "task": man.task,
        "head": man.head,
        "encoder": {k: man.encoder.get(k) for k in ("dim", "depth", "heads", "patch", "arch", "objective")},
        "patch": list(man.patch),
        "crop": list(man.crop),
        "voxel_um": list(man.voxel_size),
        "peak_threshold": man.peak_threshold,
        "min_separation_um": man.min_separation_um,
        "link_max_dist_um": man.link_max_dist_um,
        "channels": man.channels,
        "notes": man.notes,
        "tasks": ["detect", "segment", "track"],
    }


def _as_ctzyx(a: np.ndarray) -> np.ndarray:
    """The application always sends (c, t, z, y, x); tests may send less."""
    a = np.asarray(a, dtype=np.float32)
    while a.ndim < 5:
        a = a[np.newaxis]
    if a.ndim != 5:
        raise ValueError(f"expected (c, t, z, y, x), got shape {a.shape}")
    return a


def _labels_from_points(shape, points, probability, threshold: float, min_voxels: int):
    """Centroids to regions, one timepoint.

    A detection model says where objects are, not how far they extend. Watershed
    seeded on the peaks and bounded by the probability map turns one into the
    other with no extra training, so a detection bundle is still usable where
    the application wants regions rather than points."""
    deploy = _import_latents()
    lab = deploy.watershed_from_heatmap(probability, points, threshold=threshold, min_size=min_voxels)
    return lab.reshape(shape).astype(np.uint32)


def run(volume: np.ndarray, params: Dict[str, Any], device: str = "auto",
        progress: Optional[Callable[[float, str], None]] = None,
        cancelled: Optional[Callable[[], bool]] = None):
    """(c, t, z, y, x) float32 -> ((t, z, y, x) uint32 labels, info, extras).

    `params`:
        model            path to a .ltb bundle
        task             detect | segment | track
        threshold        peak probability; <= 0 means use the bundle's
        min_separation   microns between two objects; <= 0 means the bundle's
        min_voxels       drop smaller objects
        voxel_um         (z, y, x) of THIS image; the bundle's value is what the
                         thresholds were tuned at, not an assumption about the
                         input, so the caller's wins when given
        tile             (z, y, x) inference tile; empty means the bundle's
    """
    def report(f: float, msg: str = "") -> None:
        if progress:
            progress(max(0.0, min(1.0, float(f))), msg)

    def check() -> None:
        if cancelled and cancelled():
            raise KeyboardInterrupt("cancelled")

    a = _as_ctzyx(volume)
    C, T, Z, Y, X = a.shape
    m = load_bundle(str(params.get("model") or ""), device)
    man = m.m
    task = str(params.get("task") or man.task or "detect").lower()
    if task not in ("detect", "segment", "track"):
        raise ValueError(f"unknown task '{task}'; expected detect, segment or track")

    thr = float(params.get("threshold", 0) or 0)
    thr = man.peak_threshold if thr <= 0 else thr
    sep = float(params.get("min_separation", 0) or 0)
    sep = man.min_separation_um if sep <= 0 else sep
    min_voxels = int(params.get("min_voxels", 0) or 0)
    voxel = params.get("voxel_um") or man.voxel_size
    voxel = tuple(float(v) for v in voxel)[:3]
    tile = params.get("tile")
    if tile:
        m.m.crop = tuple(int(v) for v in tile)[:3]

    multi = C > 1
    labels = np.zeros((T, Z, Y, X), np.uint32)
    conf = np.zeros((T, Z, Y, X), np.float32)
    info: Dict[str, Any] = {"task": task, "threshold": thr, "min_separation_um": sep,
                            "voxel_um": list(voxel), "channels": int(C), "frames": int(T),
                            "model": man.name, "device": str(m.device)}
    extras: Dict[str, Any] = {}

    if task == "track":
        # The whole clip in one call: this is the only path that uses the time
        # axis, and splitting it per frame would defeat the point of the model.
        report(0.05, "tracking")
        clip = a if multi else a[0]
        res = m.track(clip, channels=multi, voxel_size=voxel)
        check()
        hm = m.heatmap(clip, time=True, channels=multi)
        for t, (pts, ids) in enumerate(zip(res["points"], res["ids"])):
            report(0.5 + 0.45 * t / max(T, 1), f"labelling frame {t + 1}/{T}")
            conf[t] = hm[t]
            if len(pts):
                # One track id names the same object at every timepoint: that
                # is how this application stores a track, so the ids go
                # straight into the label volume rather than alongside it.
                lab = _labels_from_points((Z, Y, X), pts, hm[t], thr, min_voxels)
                remap = np.zeros(int(lab.max()) + 1, np.uint32)
                for i, tid in enumerate(np.asarray(ids, np.uint32), start=1):
                    if i < remap.size:
                        remap[i] = tid
                labels[t] = remap[lab]
            check()
        parents = {int(k): int(v) for k, v in res["parents"].items() if v}
        info["tracks"] = int(max((int(i.max()) for i in res["ids"] if len(i)), default=0))
        info["divisions"] = len(parents) // 2
        info["links"] = int(sum(len(i) for i in res["ids"]))
        extras["lineage"] = parents
        extras["confidence"] = conf
        report(1.0, "")
        return labels, info, extras

    total = 0
    for t in range(T):
        check()
        report(0.05 + 0.9 * t / max(T, 1), f"frame {t + 1}/{T}")
        frame = a[:, t] if multi else a[0, t]
        hm = m.heatmap(frame, channels=multi)
        conf[t] = hm
        if task == "detect":
            from latents.downstream.track import peaks_from_heatmap
            pts = peaks_from_heatmap(hm, threshold=thr, voxel_size=voxel, min_sep_um=sep)
            # A detection is a point; the application's unit is a region, so
            # every point becomes a small ball of one voxel and grows no
            # further. Use task "segment" for extents.
            lab = np.zeros((Z, Y, X), np.uint32)
            for i, pnt in enumerate(np.round(pts).astype(int), start=1):
                pnt = np.clip(pnt, 0, np.array([Z, Y, X]) - 1)
                lab[tuple(pnt)] = i
            labels[t] = lab
            total += len(pts)
        else:
            lab = m.segment(frame, min_size=min_voxels, channels=multi)
            labels[t] = lab.astype(np.uint32)
            total += int(lab.max())
    info["objects"] = int(total)
    extras["confidence"] = conf
    report(1.0, "")
    return labels, info, extras
