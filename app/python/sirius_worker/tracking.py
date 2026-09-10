"""Tracking backends the worker can offer beyond the application's own.

The application always has its optimal-assignment tracker in C++. This module
adds btrack (Bayesian multi-object tracking, MIT), which is worth reaching for
when objects cross, because it carries a motion model, and when cells divide,
because it reconstructs lineages.

btrack ships its tracking core as a compiled C++ library inside the wheel and
solves the global hypothesis optimisation in Python with cvxopt / GLPK, so the
whole of it is a `pip install btrack` away and none of it has to be built here.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

INSTALL_HINT = "pip install btrack"


class NotAvailable(RuntimeError):
    """btrack is missing, or its compiled core will not load."""


def available() -> Tuple[bool, str]:
    """(usable, why not). Importing is not enough: the wheel's libtracker.so is
    built against a newer libstdc++ than some conda environments carry, and
    that only shows up when the library is actually loaded."""
    import importlib.util

    if importlib.util.find_spec("btrack") is None:
        return False, INSTALL_HINT
    try:
        from btrack import libwrapper  # type: ignore

        libwrapper.get_library()
    except Exception as e:  # noqa: BLE001 - any loader failure means unusable
        # btrack re-raises a bare Exception, so the reason is on the cause
        text = str(e)
        cause: Optional[BaseException] = e
        while not text and cause is not None:
            cause = cause.__cause__ or cause.__context__
            text = str(cause) if cause is not None else ""
        if "GLIBCXX" in text or "libstdc++" in text:
            return False, ("btrack's compiled core needs a newer libstdc++ than this Python has; "
                           "conda install -c conda-forge libstdcxx-ng, or run the worker on a system Python")
        return False, f"btrack will not load: {text.splitlines()[0][:160]}" if text else "btrack will not load"
    return True, ""


def _config(path: str):
    """btrack's tracker configuration: the file the caller named, else the
    packaged cell configuration (downloaded and cached on first use)."""
    if path:
        if not os.path.exists(path):
            raise NotAvailable(f"btrack configuration not found: {path}")
        return path
    try:
        from btrack import datasets  # type: ignore

        return datasets.cell_config()
    except Exception as e:  # noqa: BLE001 - offline, or the download failed
        raise NotAvailable("btrack needs a tracker configuration: give one in the step's Config parameter "
                           "(btrack's cell_config.json), or connect once so it can be fetched and cached") from e


def _frame_centroids(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The ids and centroids (z, y, x, in voxels) of one frame's objects.

    btrack reports a track's position as the centroid of the object it came
    from, so this is what turns a track point back into the label to repaint.
    Reading the label *under* the centroid instead loses anything the centroid
    misses -- a ring, a C, a bent filament -- because for those the centroid
    lies on the background."""
    flat = frame.reshape(-1)
    idx = np.flatnonzero(flat)
    if idx.size == 0:
        return np.zeros(0, dtype=np.uint32), np.zeros((0, 3), dtype=np.float64)
    # counted by the rank of the id, so the arrays are sized by the ids
    # present and not by the largest one (a stray id near 2^32 is a 32 GB
    # bincount otherwise)
    ids, inverse = np.unique(flat[idx], return_inverse=True)
    coords = np.stack(np.unravel_index(idx, frame.shape), axis=1).astype(np.float64)
    counts = np.bincount(inverse, minlength=ids.size)
    sums = np.stack([np.bincount(inverse, weights=coords[:, k], minlength=ids.size) for k in range(3)], axis=1)
    return ids.astype(np.uint32), sums / counts[:, None]


def run_btrack(labels: np.ndarray, voxel_um: Tuple[float, float, float], params: Dict[str, Any],
               progress=None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Track the objects of a (t, z, y, x) label volume with btrack.

    Returns the volume relabelled by track id (0 where an object was dropped)
    and a summary. Coordinates are voxels, as btrack takes them from the
    segmentation; the caller's max_distance is in micrometres, so it is
    converted with the x voxel size."""
    ok, why = available()
    if not ok:
        raise NotAvailable(why)
    import btrack  # type: ignore

    labels = np.ascontiguousarray(labels, dtype=np.uint32)
    if labels.ndim != 4:
        raise ValueError(f"btrack tracking wants a (t, z, y, x) label volume, got shape {labels.shape}")
    t_, z_, y_, x_ = labels.shape
    if progress:
        progress(0.05, "objects")
    # Objects in micrometres, so the search radius means the same thing on
    # every axis (in voxels, a light-sheet stack's z gate was dx / dz times
    # too generous); the built-in tracker gates in micrometres too. An older
    # btrack without the `scale` argument tracks in voxels, gated by dx.
    dx = float(voxel_um[0]) if voxel_um and voxel_um[0] > 0 else 1.0
    dy = float(voxel_um[1]) if voxel_um and len(voxel_um) > 1 and voxel_um[1] > 0 else dx
    dz = float(voxel_um[2]) if voxel_um and len(voxel_um) > 2 and voxel_um[2] > 0 else dx
    scale = (dz, dy, dx) if z_ > 1 else (dy, dx)
    try:
        objects = btrack.utils.segmentation_to_objects(labels, scale=scale)
        physical = True
    except TypeError:
        objects = btrack.utils.segmentation_to_objects(labels)
        physical = False
    if not objects:
        return np.zeros_like(labels), {"tracks": 0, "objects": 0}

    max_distance = float(params.get("max_distance", 10.0))
    config = _config(str(params.get("config", "") or ""))
    if progress:
        progress(0.2, "btrack")
    with btrack.BayesianTracker() as tracker:
        tracker.configure(config)
        if physical:
            tracker.max_search_radius = max(1e-6, max_distance)
            tracker.volume = ((0, x_ * dx), (0, y_ * dy), (0, z_ * dz)) if z_ > 1 else ((0, x_ * dx), (0, y_ * dy))
        else:
            tracker.max_search_radius = max(1.0, max_distance / dx)
            tracker.volume = ((0, x_), (0, y_), (0, z_)) if z_ > 1 else ((0, x_), (0, y_))
        tracker.track(step_size=100)
        if bool(params.get("optimise", True)):
            # the hypothesis optimisation is what reconstructs lineages
            tracker.optimize()
        tracks = tracker.tracks
    if progress:
        progress(0.8, "relabelling")

    min_length = max(1, int(params.get("min_length", 2)))
    out = np.zeros_like(labels)
    centroids = [_frame_centroids(labels[t]) for t in range(t_)]
    kept = 0
    lengths = []
    parents = set()
    for track in tracks:
        frames = list(track.t)
        if len(frames) < min_length:
            continue
        kept += 1
        lengths.append(len(frames))
        parent = getattr(track, "parent", None)
        if parent is not None and parent != track.ID:
            # one division makes two daughters, and each of them names the
            # mother: it is the mothers that have to be counted
            parents.add(parent)
        # btrack keeps the source object's centroid, so the object to repaint
        # is the one whose centroid the track point is standing on. A point
        # btrack interpolated across a gap stands on no object at all; fall
        # back to the label under it, which is how those used to be found.
        for t, zc, yc, xc in zip(frames, track.z, track.y, track.x):
            if not (0 <= t < t_):
                continue
            if physical:   # back to voxels, where the centroids live
                zc, yc, xc = zc / dz, yc / dy, xc / dx
            ids_t, centres_t = centroids[t]
            src = 0
            if ids_t.size:
                want = np.array([zc if z_ > 1 else 0.0, yc, xc], dtype=np.float64)
                near = int(np.argmin(((centres_t - want) ** 2).sum(axis=1)))
                if ((centres_t[near] - want) ** 2).sum() <= 1.0:
                    src = int(ids_t[near])
            if not src:
                zi = min(max(int(round(zc)) if z_ > 1 else 0, 0), z_ - 1)
                yi = min(max(int(round(yc)), 0), y_ - 1)
                xi = min(max(int(round(xc)), 0), x_ - 1)
                src = int(labels[t, zi, yi, xi])
            if src:
                out[t][labels[t] == src] = kept
    if progress:
        progress(1.0, f"{kept} tracks")
    return out, {"tracks": kept, "objects": len(objects), "divisions": len(parents),
                 "mean_length": float(np.mean(lengths)) if lengths else 0.0,
                 "longest": int(max(lengths)) if lengths else 0}
