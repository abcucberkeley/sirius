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
    confidence  (t, z, y, x) float32, the model's centroid probability (the
                foreground probability for a three-class head); the
                application folds it into per-label confidence
    lineage     JSON, {child track id: parent track id}; divisions only

Requires the `latents` package. Point SIRIUS_LATENTS_PATH at a checkout if it
is not installed.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import threading
import zipfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_BUNDLES: Dict[Tuple[str, float, str], Any] = {}
# model_info runs on the connection thread while a job may be loading a bundle
_BUNDLES_LOCK = threading.Lock()


class Cancelled(RuntimeError):
    """The run was cancelled. An Exception on purpose: the server answers the
    request for any Exception, while a BaseException such as KeyboardInterrupt
    escapes its handler and ends the job thread without a reply, which left the
    application waiting out its grace period and then dropping the connection."""


def _import_latents():
    """Import `latents.deploy`, with a message that says what to do if absent.

    A worker that cannot find the package is the most likely failure in a fresh
    install, and "ModuleNotFoundError: latents" on its own tells a microscopist
    nothing actionable."""
    extra = os.environ.get("SIRIUS_LATENTS_PATH", "")
    if extra and extra not in sys.path:
        sys.path.insert(0, extra)
    try:
        from latents import deploy
    except ImportError as exc:                           # pragma: no cover - install-dependent
        raise RuntimeError(
            "The foundation model needs the 'latents' package, which is not importable. "
            "Install it (pip install latents) or set SIRIUS_LATENTS_PATH to a checkout. "
            f"({exc})") from exc
    return deploy


def _bundle_file(path: str) -> Tuple[str, float]:
    if not path:
        raise ValueError("no model given: choose a .ltb bundle")
    if not os.path.exists(path):
        raise FileNotFoundError(f"model bundle not found: {path}")
    return os.path.abspath(path), os.path.getmtime(path)


def load_bundle(path: str, device: str = "auto"):
    """Load and cache a bundle. Re-reading a 500 MB file per timepoint would
    dominate the run, and the application calls once per timepoint.

    The cached object is shared by every later run and by model_info, so
    nothing may write to it; a per-run setting goes on a copy (see `run`)."""
    deploy = _import_latents()
    key = (*_bundle_file(path), device)
    with _BUNDLES_LOCK:
        got = _BUNDLES.get(key)
    if got is None:
        dev = None if device in ("auto", "") else device
        got = deploy.Bundle.load(path, device=dev)
        with _BUNDLES_LOCK:
            _BUNDLES.clear()                              # one model resident at a time
            _BUNDLES[key] = got
    return got


def _manifest(path: str):
    """The bundle's manifest, without loading the model for inference.

    model_info used to load the bundle on the CPU, and since one model stays
    resident at a time that evicted the one a run had just put on the GPU: the
    model dialog and a run, alternating, reloaded it every time. A bundle
    already loaded on any device answers from memory; otherwise only the
    checkpoint is opened (memory-mapped, the weights are not read) and nothing
    is cached or evicted."""
    deploy = _import_latents()
    file_key = _bundle_file(path)
    with _BUNDLES_LOCK:
        for key, bundle in _BUNDLES.items():
            if key[:2] == file_key:
                return bundle.m
    # Ask latents for the manifest rather than torch.load-ing the file here.
    # Since latents bundle format 2 (2026-09-13) an .ltb is a plain zip with
    # manifest.json at its root and the weights beside it, which torch.load
    # rejects outright ("file in archive is not in a subdirectory"); the
    # previous code retried the same call without mmap and raised, so
    # model_info failed on every current bundle while run() -- which goes
    # through Bundle.load -- kept working. Bundle.manifest_of reads the JSON
    # member without touching a weight and still understands a version-1 file.
    manifest_of = getattr(getattr(deploy, "Bundle", None), "manifest_of", None)
    if manifest_of is not None:
        return manifest_of(str(path))
    import torch                                          # a latents older than format 2

    try:
        ck = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):                    # an older torch, or a file without the zip format
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
    want = getattr(deploy, "BUNDLE_VERSION", 1)
    if int(ck.get("bundle_version", 0)) != want:
        raise ValueError(f"{path}: bundle version {ck.get('bundle_version')}, expected {want}")
    return deploy.Manifest.from_dict(ck["manifest"])


def _tasks(man) -> list:
    # A three-class head predicts regions, not centroids: there is nothing to
    # detect, and nothing to link.
    return ["segment"] if man.head == "threeclass" else ["detect", "segment", "track"]


def model_info(path: str) -> Dict[str, Any]:
    """What the bundle says about itself, for the model dialog and for the
    application's parameter defaults.

    `voxel_um` is (x, y, z), the order the application uses for a voxel size
    everywhere; the manifest stores latents' (z, y, x). `patch` and `crop` stay
    (z, y, x), the order of the step's Tile."""
    man = _manifest(path)
    return {
        "format": "latents-bundle",
        "name": man.name,
        "task": man.task,
        "head": man.head,
        "encoder": {k: man.encoder.get(k) for k in ("dim", "depth", "heads", "patch", "arch", "objective")},
        "patch": list(man.patch),
        "crop": list(man.crop),
        "voxel_um": [float(v) for v in man.voxel_size][::-1],
        "peak_threshold": man.peak_threshold,
        "min_separation_um": man.min_separation_um,
        "link_max_dist_um": man.link_max_dist_um,
        "channels": man.channels,
        "notes": man.notes,
        "tasks": _tasks(man),
    }


def manifest_of(path: str) -> Dict[str, Any]:
    """A bundle's manifest without its weights.

    `model_info` answers the same questions by loading the bundle, which for a
    directory listing would mean reading every file on disk, and the worker
    keeps one bundle resident so listing would also evict whatever is loaded.
    A bundle is a zip, so the manifest can be read on its own. Returns {} when
    it cannot be, which is not an error here: the listing still names the file
    and the application can ask `model_info` about the one the user picks.
    """
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            want = [n for n in names if os.path.basename(n).lower() in ("manifest.json", "meta.json")]
            # else the shallowest .json in the archive
            if not want:
                want = sorted((n for n in names if n.lower().endswith(".json")), key=lambda n: (n.count("/"), n))
            for name in want:
                try:
                    got = json.loads(z.read(name))
                except (ValueError, OSError):
                    continue
                # Some other .json in the archive is not a manifest, and showing
                # its numbers as calibration would be worse than showing none:
                # a bundle listed with a voxel size it was not trained at reads
                # as fact. Require something only a manifest has.
                if isinstance(got, dict) and any(k in got for k in ("task", "peak_threshold", "voxel_size", "patch")):
                    return got
    except (zipfile.BadZipFile, OSError):
        return {}
    return {}


# A manifest is written elsewhere, by a version of latents this worker does not
# choose. So every field is read defensively: a bundle whose manifest says
# something unexpected is listed with that field blank, rather than taking the
# whole directory listing down and leaving the user with a dialog that says the
# registry is unreadable.
def _text(man: Dict[str, Any], key: str) -> str:
    got = man.get(key)
    return got if isinstance(got, str) else ""


def _number(man: Dict[str, Any], key: str) -> Optional[float]:
    got = man.get(key)
    return float(got) if isinstance(got, (int, float)) and not isinstance(got, bool) else None


def _numbers(man: Dict[str, Any], key: str) -> List[float]:
    got = man.get(key)
    if not isinstance(got, (list, tuple)):
        return []
    return [float(v) for v in got if isinstance(v, (int, float)) and not isinstance(v, bool)]


def list_bundles(directory: str) -> List[Dict[str, Any]]:
    """Every .ltb in `directory`, with what its manifest says about it.

    The application shows this as the list a user picks a model from, so a
    bundle whose manifest cannot be read is still listed, with `manifest`
    empty: a file that is there and unreadable is something the user needs to
    see, not something to hide. Sorted by name so the list does not reorder
    itself between calls. Not recursive: a registry is a directory of bundles,
    and walking a filesystem the worker shares with a cluster is not something
    to do behind a dialog opening.
    """
    if not directory:
        raise ValueError("no registry directory given")
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"not a directory: {directory}")
    out: List[Dict[str, Any]] = []
    with os.scandir(directory) as entries:
        for e in sorted(entries, key=lambda e: e.name.lower()):
            if not e.is_file() or not e.name.lower().endswith(".ltb"):
                continue
            try:
                stat = e.stat()
                size, mtime = int(stat.st_size), float(stat.st_mtime)
            except OSError:
                size, mtime = 0, 0.0
            man = manifest_of(e.path)
            out.append({
                "path": os.path.abspath(e.path),
                "file": e.name,
                "name": _text(man, "name") or os.path.splitext(e.name)[0],
                "task": _text(man, "task"),
                "encoder": man.get("encoder") if isinstance(man.get("encoder"), dict) else {},
                "patch": _numbers(man, "patch"),
                "crop": _numbers(man, "crop"),
                "voxel_um": _numbers(man, "voxel_size") or _numbers(man, "voxel_um"),
                "peak_threshold": _number(man, "peak_threshold"),
                "min_separation_um": _number(man, "min_separation_um"),
                "channels": man.get("channels") if isinstance(man.get("channels"), list) else None,
                "notes": _text(man, "notes"),
                "size_bytes": size,
                "mtime": mtime,
                "manifest": bool(man),
            })
    return out


def _as_ctzyx(a: np.ndarray) -> np.ndarray:
    """The application always sends (c, t, z, y, x); tests may send less."""
    a = np.asarray(a, dtype=np.float32)
    while a.ndim < 5:
        a = a[np.newaxis]
    if a.ndim != 5:
        raise ValueError(f"expected (c, t, z, y, x), got shape {a.shape}")
    return a


def _per_axis(given: Any, fallback: Sequence[float]) -> Tuple[float, ...]:
    """Three values, each one missing or <= 0 replaced by the fallback's."""
    try:
        vals = [float(v) for v in (given if given is not None else [])][:3]
    except (TypeError, ValueError):
        vals = []
    vals += [0.0] * (3 - len(vals))
    return tuple(v if np.isfinite(v) and v > 0 else float(f) for v, f in zip(vals, list(fallback)[-3:]))


def voxel_zyx(voxel_um: Any, bundle_zyx: Sequence[float]) -> Tuple[float, ...]:
    """The application's (x, y, z) voxel size in latents' (z, y, x) order.

    Every distance in latents is (z, y, x): the peak gate divides the separation
    by it per axis of the heatmap, and the linker scales (z, y, x) coordinate
    differences by it. Passed on as (x, y, z), a 0.75 um z step became 0.15 um,
    so two nuclei 3 um apart in depth were gated into one, and a nucleus moving
    1.5 um a frame in x started a new track every frame. An axis missing or
    <= 0 uses the bundle's calibration."""
    return _per_axis(voxel_um, list(bundle_zyx)[::-1])[::-1]


def _dense(lab: np.ndarray) -> np.ndarray:
    """Labels renumbered 1..n in the order of their old ids; 0 stays background."""
    lab = np.asarray(lab)
    if lab.size == 0 or int(lab.max()) == 0:
        return lab.astype(np.uint32)
    present = np.bincount(lab.ravel().astype(np.int64)) > 0
    present[0] = False
    remap = np.zeros(present.size, np.uint32)
    remap[present] = np.arange(1, int(present.sum()) + 1, dtype=np.uint32)
    return remap[lab]


def _foreground(logits: np.ndarray) -> np.ndarray:
    """(3, z, y, x) background / interior / boundary logits -> P(not background)."""
    l = np.asarray(logits, np.float64)
    e = np.exp(l - l.max(axis=0, keepdims=True))
    return (1.0 - e[0] / e.sum(axis=0)).astype(np.float32)


def lineage(parents: Dict[Any, Any]) -> Dict[int, int]:
    """latents' parent map -> {daughter track id: mother track id}.

    `track_points` records a parent of 0 for every track that starts without
    one, and the first time a mother divides it also gives the daughter that
    keeps the mother's id the mother as parent, i.e. itself (a `setdefault`).
    Neither is a division, and counting half the entries, as this used to,
    reported a track that divides twice as one division. Each division starts
    exactly one new track whose parent is another track: those entries are the
    divisions. The count is still only as good as latents' geometric rule."""
    return {int(k): int(v) for k, v in parents.items() if int(v) and int(v) != int(k)}


def run(volume: np.ndarray, params: Dict[str, Any], device: str = "auto",
        progress: Optional[Callable[[float, str], None]] = None,
        cancelled: Optional[Callable[[], bool]] = None):
    """(c, t, z, y, x) float32 -> ((t, z, y, x) uint32 labels, info, extras).

    `params`:
        model            path to a .ltb bundle
        task             detect | segment | track
        threshold        peak probability; <= 0 means use the bundle's
        min_separation   microns between two objects; <= 0 means the bundle's
        min_voxels       Segment only: drop smaller objects. Detect marks one
                         voxel per object, and a tracking run keeps every
                         object because there the label id is a track id
        voxel_um         (x, y, z) of THIS image, the application's order; the
                         bundle's value is what the thresholds were tuned at,
                         not an assumption about the input, so the caller's
                         wins on every axis it gives as > 0
        tile             (z, y, x) inference tile; an extent <= 0, or no tile,
                         uses the bundle's crop on that axis

    All of these apply to this call only: the cached bundle is never written
    to. The model runs once per frame (once for the whole clip when tracking),
    and the peaks, the regions and the tracks all come from that one heatmap
    with the values reported in `info`.
    """
    def report(f: float, msg: str = "") -> None:
        if progress:
            progress(max(0.0, min(1.0, float(f))), msg)

    def check() -> None:
        if cancelled and cancelled():
            raise Cancelled("cancelled")

    a = _as_ctzyx(volume)
    n_c, n_t, n_z, n_y, n_x = a.shape
    m = load_bundle(str(params.get("model") or ""), device)
    man = m.m
    task = str(params.get("task") or man.task or "detect").lower()
    if task not in ("detect", "segment", "track"):
        raise ValueError(f"unknown task '{task}'; expected detect, segment or track")
    if task not in _tasks(man):
        raise ValueError(f"this bundle's '{man.head}' head predicts regions, not centroids, so it cannot {task}; "
                         "choose the Segment task")
    deploy = _import_latents()
    from latents.downstream.track import peaks_from_heatmap, track_points

    thr = float(params.get("threshold", 0) or 0)
    thr = float(man.peak_threshold) if thr <= 0 else thr
    sep = float(params.get("min_separation", 0) or 0)
    sep = float(man.min_separation_um) if sep <= 0 else sep
    min_voxels = max(0, int(params.get("min_voxels", 0) or 0))
    voxel = voxel_zyx(params.get("voxel_um"), man.voxel_size)
    crop = tuple(int(v) for v in _per_axis(params.get("tile"), man.crop))
    if crop != tuple(int(v) for v in man.crop):
        # Bundle.heatmap reads its tile from the manifest. Copies of both, so
        # neither a later run without a tile nor model_info sees this one.
        man = copy.copy(man)
        man.crop = crop
        m = copy.copy(m)
        m.m = man

    multi = n_c > 1
    labels = np.zeros((n_t, n_z, n_y, n_x), np.uint32)
    conf = np.zeros((n_t, n_z, n_y, n_x), np.float32)
    info: Dict[str, Any] = {"task": task, "threshold": thr, "min_separation_um": sep,
                            "voxel_um": list(voxel[::-1]), "tile": list(crop), "channels": int(n_c),
                            "frames": int(n_t), "model": man.name, "device": str(m.device)}
    extras: Dict[str, Any] = {}

    def peaks(hm: np.ndarray) -> np.ndarray:
        return peaks_from_heatmap(hm, threshold=thr, voxel_size=voxel, min_sep_um=sep)

    if task == "track":
        # The whole clip in one call: this is the only path that uses the time
        # axis, and splitting it per frame would defeat the point of the model.
        # Bundle.track would run it again and use the bundle's threshold and
        # separation instead of these, so the linking is done here.
        report(0.05, "tracking")
        clip = a if multi else a[0]
        hm = np.asarray(m.heatmap(clip, time=True, channels=multi), np.float32)
        check()
        pts = []
        for t in range(n_t):
            pts.append(peaks(hm[t]))
            check()
        ids, parents = track_points(pts, max_dist=man.link_max_dist_um, voxel_size=voxel,
                                    division_dist=man.division_dist_um)
        for t in range(n_t):
            report(0.5 + 0.45 * t / max(n_t, 1), f"labelling frame {t + 1}/{n_t}")
            conf[t] = hm[t]
            if len(pts[t]):
                # One track id names the same object at every timepoint: that
                # is how this application stores a track, so the ids go
                # straight into the label volume rather than alongside it.
                # min_size 0: dropping an object whose region is small in one
                # frame would punch a hole in its track.
                lab = deploy.watershed_from_heatmap(hm[t], pts[t], threshold=thr, min_size=0)
                remap = np.zeros(max(int(lab.max()), len(pts[t])) + 1, np.uint32)
                remap[1:len(pts[t]) + 1] = np.asarray(ids[t], np.uint32)
                labels[t] = remap[lab]
            check()
        kids = lineage(parents)
        info["tracks"] = int(max((int(np.max(i)) for i in ids if len(i)), default=0))
        info["divisions"] = len(kids)
        info["links"] = int(sum(len(i) for i in ids))
        extras["lineage"] = kids
        extras["confidence"] = conf
        report(1.0, "")
        return labels, info, extras

    total = 0
    for t in range(n_t):
        check()
        report(0.05 + 0.9 * t / max(n_t, 1), f"frame {t + 1}/{n_t}")
        frame = a[:, t] if multi else a[0, t]
        if man.head == "threeclass":
            # Bundle.heatmap cannot run this head (it writes three channels
            # into a one-channel buffer and raises); Bundle.segment's own
            # route for it is the class logits, and so is this one.
            from latents.downstream.instance import instances_from_three_class

            m._check_channels(n_c)
            logits = np.asarray(m._class_logits(frame, channels=multi), np.float32)
            conf[t] = _foreground(logits)
            lab = _dense(instances_from_three_class(logits, min_size=min_voxels))
        else:
            hm = np.asarray(m.heatmap(frame, channels=multi), np.float32)
            conf[t] = hm
            pts = peaks(hm)
            if task == "detect":
                # A detection is a point; the application's unit is a region,
                # so every point becomes one voxel and grows no further, and
                # min_voxels does not apply. Use task "segment" for extents.
                lab = np.zeros((n_z, n_y, n_x), np.uint32)
                for i, pnt in enumerate(np.round(pts).astype(int), start=1):
                    pnt = np.clip(pnt, 0, np.array([n_z, n_y, n_x]) - 1)
                    lab[tuple(pnt)] = i
            else:
                # Bundle.segment would run the model twice more and use the
                # bundle's threshold, separation and voxel size.
                lab = _dense(deploy.watershed_from_heatmap(hm, pts, threshold=thr, min_size=min_voxels))
        labels[t] = lab
        total += int(lab.max())
    info["objects"] = int(total)
    extras["confidence"] = conf
    report(1.0, "")
    return labels, info, extras
