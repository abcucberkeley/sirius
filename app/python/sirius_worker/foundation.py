"""Trained models that arrive as a folder, run for the application's Foundation step.

A model is a self-contained folder, written by latents' scripts/export_model.py:

    <models>/<name>/<version>/
        model.py              the model's own API: load(folder, device) -> Model
        model.json            format "latents-model/1": tasks, the input contract,
                              the decode rule, the prompt form, provenance
        weights.safetensors   the tensors
        README.md             what it segments, its limits
        _lib/                 the model's code, which imports only torch, numpy
                              and each other

Nothing here imports latents. The worker imports the folder's model.py (a
module name of its own, the folder on sys.path only while it is imported) and
calls it: Segment -> Model.segment, Prompt -> Model.prompt with the joint
`objects` form. The model normalises its own input (model.json's
`input.normalisation`), so the raw intensities go to it unchanged; what this
module does with the contract is check it -- the channel count, the voxel size
the model was trained at -- and say so when the image does not fit.

The old single-file bundles (.ltb) needed the latents package to load. They are
refused with what to do instead: re-export the run as a folder.

Returned to the application, per time point:

    labels      (t, z, y, x) uint32, 0 = background. Segment: one id per
                object, numbered densely. Prompt: the i-th object's mask is
                label i + 1, so the application can give it the object's id
    confidence  Prompt only: (t, z, y, x) float32, each mask's own score
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

FORMAT = "latents-model"
FORMAT_MAJOR = 1
OLD_BUNDLE = ("old bundle format: {path} is a .ltb bundle, which needed the latents package. Models are "
              "self-contained folders now: re-export it with latents scripts/export_model.py "
              "(<models>/<name>/<version>/ with model.py, model.json and weights.safetensors) and choose that folder")
# A voxel size this many times larger or smaller than the model's, on any axis, is said in the run's
# warnings: nothing in the model adapts to scale.
VOXEL_TOLERANCE = 1.5

# The model resident in this worker: (folder, stamp, device) -> _Loaded. One at a time, since a model
# can be gigabytes of GPU memory; model_info and the listing never load one.
_MODELS: Dict[Tuple[str, str, str], _Loaded] = {}
_LOCK = threading.RLock()


class Cancelled(RuntimeError):
    """The run was cancelled. An Exception on purpose: the server answers the
    request for any Exception, while a BaseException such as KeyboardInterrupt
    escapes its handler and ends the job thread without a reply."""


class ModelError(ValueError):
    """A model folder that cannot be used, said so the person knows what to do."""


# --- the folder ------------------------------------------------------------------------------------

def is_old_bundle(path: str) -> bool:
    return str(path or "").lower().endswith(".ltb")


def model_folder(path: str) -> str:
    """The absolute model folder `path` names: the folder, or its model.json. Raises with a sentence."""
    if not path:
        raise ModelError("no model given: choose a model folder (one holding model.py and model.json)")
    if is_old_bundle(path):
        raise ModelError(OLD_BUNDLE.format(path=path))
    p = os.path.abspath(os.path.expanduser(str(path)))
    if os.path.basename(p).lower() == "model.json" and os.path.isfile(p):
        p = os.path.dirname(p)
    if not os.path.exists(p):
        raise FileNotFoundError(f"model folder not found: {path}")
    if not os.path.isdir(p):
        raise ModelError(f"{path} is a file, not a model folder: choose the folder that holds model.py and model.json")
    if not os.path.isfile(os.path.join(p, "model.json")):
        raise ModelError(f"{path} is not a model folder: it has no model.json. A model folder holds model.py, "
                         "model.json and weights.safetensors (latents scripts/export_model.py writes one); "
                         "a folder of models is chosen through Models…")
    return p


def is_model_folder(path: str) -> bool:
    """A folder whose model.json says it is a latents-model; never raises."""
    try:
        read_manifest(model_folder(path))
        return True
    except (OSError, ValueError):
        return False


def read_manifest(folder: str) -> Dict[str, Any]:
    """model.json, checked for what this worker relies on."""
    path = os.path.join(folder, "model.json")
    try:
        with open(path, encoding="utf-8") as f:
            man = json.load(f)
    except OSError as e:
        raise ModelError(f"cannot read {path}: {e}") from e
    except ValueError as e:
        raise ModelError(f"{path} is not valid JSON ({e}); the export did not finish, or the file was edited") from e
    if not isinstance(man, dict):
        raise ModelError(f"{path} does not describe a model")
    fmt = str(man.get("format") or "")
    if not fmt.startswith(FORMAT + "/"):
        raise ModelError(f"{path} is format '{fmt or '?'}', not {FORMAT}/{FORMAT_MAJOR}: not a model this worker runs")
    try:
        major = int(fmt.split("/", 1)[1].split(".")[0])
    except ValueError:
        major = -1
    if major != FORMAT_MAJOR:
        raise ModelError(f"{path} is {fmt}; this worker reads {FORMAT}/{FORMAT_MAJOR}. "
                         "Update SIRIUS, or export the model with the matching latents")
    tasks = man.get("tasks")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(t, str) for t in tasks):
        raise ModelError(f"{path} lists no tasks")
    return man


def _stamp(folder: str) -> str:
    """What changes when the model is re-exported in place."""
    parts = []
    for name in ("model.json", "model.py", "weights.safetensors"):
        try:
            st = os.stat(os.path.join(folder, name))
            parts.append(f"{name}:{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            parts.append(f"{name}:-")
    return "|".join(parts)


def _readme_line(folder: str) -> str:
    """The README's first paragraph that is not a heading, as one line."""
    try:
        with open(os.path.join(folder, "README.md"), encoding="utf-8") as f:
            text = f.read(16384)
    except OSError:
        return ""
    para: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#") or s.startswith("```"):
            if para:
                break
            continue
        if not s:
            if para:
                break
            continue
        para.append(s)
    return " ".join(para)


def _numbers(v: Any) -> List[float]:
    if not isinstance(v, (list, tuple)):
        return []
    return [float(x) for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]


def _dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _summary(folder: str, man: Dict[str, Any]) -> Dict[str, Any]:
    """What the application shows about a model, from model.json and README.md alone."""
    inp = _dict(man.get("input"))
    voxel_zyx = _numbers(inp.get("voxel_um"))
    tasks = [str(t) for t in man.get("tasks") or []]
    notes = str(man.get("notes") or "").strip()
    head = _dict(man.get("head"))
    try:
        channels = int(inp.get("channels") or 1)
    except (TypeError, ValueError):
        channels = 1
    return {
        "format": FORMAT,
        "format_version": str(man.get("format")),
        "path": folder,
        "name": str(man.get("name") or os.path.basename(os.path.dirname(folder)) or os.path.basename(folder)),
        "version": str(man.get("version") or os.path.basename(folder)),
        "tasks": tasks,
        "promptable": "prompt" in tasks,
        "description": notes or _readme_line(folder),
        "head": str(head.get("kind") or ""),
        # the application's order for a voxel size is (x, y, z); model.json's is (z, y, x)
        "voxel_um": voxel_zyx[::-1],
        "crop": _numbers(inp.get("crop")),
        "patch": _numbers(inp.get("patch")),
        "channels": channels,
        "channel_merge": str(inp.get("channel_merge") or ""),
        "input": inp,
        "decode": _dict(man.get("decode")),
        "prompt": man.get("prompt") if isinstance(man.get("prompt"), dict) else None,
        "needs": list(_dict(man.get("decode")).get("needs") or []),
        "provenance": {k: _dict(man.get("provenance")).get(k) for k in ("latents_commit", "latents_dirty", "run")},
        "created": str(man.get("created") or ""),
    }


def model_info(path: str) -> Dict[str, Any]:
    """What a model folder says about itself, without loading it: the model dialog, the Task choice
    (only what `tasks` lists) and the step's checks read this."""
    folder = model_folder(path)
    info = _summary(folder, read_manifest(folder))
    info["loaded"] = any(k[0] == folder for k in list(_MODELS))   # no lock: a load in progress holds it
    return info


def list_models(dirs: Sequence[str]) -> Dict[str, Any]:
    """Every model under each of `dirs`: <dir>/<name>/<version>/model.json, or <dir>/<name>/model.json,
    or `dir` itself being a model folder. Two levels, no deeper: a models folder on a cluster is
    not something to walk behind a dialog opening.

    A folder whose model.json does not read is listed with `error` set (a broken export is
    something to see, not to hide), as is an old .ltb bundle beside the models. A `dir` that is not
    there is reported in `errors`, and the others are still listed."""
    models: List[Dict[str, Any]] = []
    errors: List[str] = []
    seen = set()

    def add(folder: str) -> None:
        folder = os.path.abspath(folder)
        if folder in seen:
            return
        seen.add(folder)
        try:
            entry = _summary(folder, read_manifest(folder))
            entry["error"] = ""
        except (OSError, ValueError) as e:
            rel = folder.replace("\\", "/").rstrip("/").split("/")
            entry = {"path": folder, "name": rel[-2] if len(rel) > 1 else rel[-1], "version": rel[-1],
                     "tasks": [], "promptable": False, "description": "", "error": str(e)}
        try:
            entry["size_bytes"] = int(os.path.getsize(os.path.join(folder, "weights.safetensors")))
        except OSError:
            entry["size_bytes"] = -1
        try:
            entry["mtime"] = float(os.path.getmtime(os.path.join(folder, "model.json")))
        except OSError:
            entry["mtime"] = 0.0
        models.append(entry)

    def subdirs(d: str) -> List[os.DirEntry]:
        try:
            with os.scandir(d) as it:
                return sorted((e for e in it if e.is_dir() and not e.name.startswith((".", "_"))),
                              key=lambda e: e.name.lower())
        except OSError:
            return []

    for d in dirs:
        d = str(d or "").strip()
        if not d:
            continue
        d = os.path.expanduser(d)
        if not os.path.isdir(d):
            errors.append(f"not a folder: {d}")
            continue
        if os.path.isfile(os.path.join(d, "model.json")):
            add(d)
            continue
        try:
            with os.scandir(d) as it:
                for e in sorted(it, key=lambda e: e.name.lower()):
                    if e.is_file() and is_old_bundle(e.name):
                        models.append({"path": os.path.abspath(e.path), "name": os.path.splitext(e.name)[0],
                                       "version": "", "tasks": [], "promptable": False, "description": "",
                                       "size_bytes": -1, "mtime": 0.0,
                                       "error": OLD_BUNDLE.format(path=e.name)})
        except OSError as e:
            errors.append(f"cannot read {d}: {e}")
            continue
        for name in subdirs(d):
            if os.path.isfile(os.path.join(name.path, "model.json")):
                add(name.path)
                continue
            for version in subdirs(name.path):
                if os.path.isfile(os.path.join(version.path, "model.json")):
                    add(version.path)
    return {"models": models, "errors": errors}


# --- loading -----------------------------------------------------------------------------------------

def _lib_modules() -> Dict[str, Any]:
    return {k: m for k, m in sys.modules.items() if k == "_lib" or k.startswith("_lib.")}


def _purge(names: Sequence[str]) -> None:
    for k in list(sys.modules):
        if k in names or k == "_lib" or k.startswith("_lib."):
            sys.modules.pop(k, None)


class _Loaded:
    """One model folder, imported and loaded: its model.py module, the Model it returned, and its
    `_lib` modules. Every exported model has a `_lib` package of its own and imports it by that name,
    so they are put back in sys.modules for each call -- a lazy import inside the model then finds
    this model's code, never another's."""

    def __init__(self, folder: str, module: Any, model: Any, lib: Dict[str, Any], manifest: Dict[str, Any]):
        self.folder = folder
        self.module = module
        self.model = model
        self.lib = lib
        self.manifest = manifest
        self.calls = threading.Lock()

    def call(self, method: str, *args, **kwargs):
        with self.calls:
            for k in [k for k in sys.modules if (k == "_lib" or k.startswith("_lib.")) and k not in self.lib]:
                sys.modules.pop(k, None)
            sys.modules.update(self.lib)
            try:
                return getattr(self.model, method)(*args, **kwargs)
            finally:
                self.lib.update(_lib_modules())   # what it imported lazily is its own too


def _missing_package(folder: str, man: Dict[str, Any], e: ModuleNotFoundError) -> ModelError:
    needs = ", ".join(_dict(man.get("decode")).get("needs") or [])
    what = e.name or str(e)
    hint = {"torch": "torch (the model's network)", "safetensors": "safetensors (the weights file)",
            "scipy": "scipy (the instance decode)", "skimage": "scikit-image (the instance decode)"}.get(
        (what or "").split(".")[0], what)
    return ModelError(f"{man.get('name', folder)}: the model needs {hint}, which this worker's Python does not have. "
                      "Install it in the worker environment (Preferences ▸ Python), or run on the cluster, whose "
                      "worker image has it" + (f". The model's decode needs {needs}" if needs else ""))


def load_model(path: str, device: str = "auto") -> _Loaded:
    """Import the folder's model.py and call its load(). Cached per (folder, files' stamp, device):
    the application calls once per time point, and a re-export in place is picked up. One model is
    resident at a time; loading another unloads it, `_lib` modules included."""
    folder = model_folder(path)
    man = read_manifest(folder)
    dev = str(device or "auto")
    key = (folder, _stamp(folder), dev)
    with _LOCK:
        got = _MODELS.get(key)
        if got is not None:
            return got
        for old in _MODELS.values():
            _purge([old.module.__name__])
        _MODELS.clear()
        _purge([])
        entry = os.path.join(folder, "model.py")
        if not os.path.isfile(entry):
            raise ModelError(f"{folder} has model.json but no model.py: the export is incomplete")
        name = "_sirius_model_" + hashlib.sha1(("\n".join(key)).encode("utf-8")).hexdigest()[:16]
        saved_path = list(sys.path)
        sys.path.insert(0, folder)
        try:
            spec = importlib.util.spec_from_file_location(name, entry)
            if spec is None or spec.loader is None:
                raise ModelError(f"{entry} cannot be imported")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loader = getattr(module, "load", None)
            if not callable(loader):
                raise ModelError(f"{entry} has no load(folder, device): not a model this worker can call")
            model = loader(folder, None if dev in ("auto", "") else dev)
            lib = _lib_modules()
        except ModelError:
            _purge([name])
            raise
        except ModuleNotFoundError as e:
            _purge([name])
            if e.name and (e.name == "_lib" or e.name.startswith("_lib.")):
                raise ModelError(f"{folder}: model.py imports {e.name}, which the folder does not have; "
                                 "the export is incomplete") from e
            raise _missing_package(folder, man, e) from e
        except Exception as e:  # noqa: BLE001 - the model's own failure, said with its folder
            _purge([name])
            raise ModelError(f"{man.get('name', '?')} {man.get('version', '')} in {folder} did not load: "
                             f"{type(e).__name__}: {e}") from e
        finally:
            sys.path[:] = saved_path
        for want in ("segment", "info") + (("prompt",) if "prompt" in man["tasks"] else ()):
            if not callable(getattr(model, want, None)):
                _purge([name])
                raise ModelError(f"{entry}: load() returned an object without {want}(); not the model API "
                                 "(segment, prompt, info, tasks, logits)")
        got = _Loaded(folder, module, model, lib, man)
        _MODELS[key] = got
        return got


def unload() -> None:
    """Forget the resident model (tests; a worker asked to free memory)."""
    with _LOCK:
        for old in _MODELS.values():
            _purge([old.module.__name__])
        _MODELS.clear()


# --- running ----------------------------------------------------------------------------------------

def _as_ctzyx(a: np.ndarray) -> np.ndarray:
    """The application always sends (c, t, z, y, x); tests may send less."""
    a = np.asarray(a, dtype=np.float32)
    while a.ndim < 5:
        a = a[np.newaxis]
    if a.ndim != 5:
        raise ValueError(f"expected (c, t, z, y, x), got shape {a.shape}")
    return a


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


def voxel_warning(given_xyz: Any, info: Dict[str, Any]) -> str:
    """A sentence when the image's voxel size is far from the model's on some axis, else ""."""
    model_xyz = list(info.get("voxel_um") or [])
    try:
        given = [float(v) for v in (given_xyz or [])][:3]
    except (TypeError, ValueError):
        return ""
    if len(given) != 3 or len(model_xyz) != 3:
        return ""
    far = []
    for axis, g, m in zip("xyz", given, model_xyz):
        if g > 0 and m > 0 and np.isfinite(g) and (g / m > VOXEL_TOLERANCE or m / g > VOXEL_TOLERANCE):
            far.append(axis)
    if not far:
        return ""
    fmt = lambda v: " x ".join(f"{x:.3g}" for x in v[::-1])  # noqa: E731 - (z, y, x), as model.json says it
    return (f"the image's voxel size ({fmt(given)} um, z x y x x) is far from the {fmt(model_xyz)} um "
            f"{info.get('name', 'the model')} was trained at, on {', '.join(far)}; nothing in the model adapts to "
            "scale, so expect worse objects (resample the image to the model's voxel size first)")


def run(volume: np.ndarray, params: Dict[str, Any], device: str = "auto",
        progress: Optional[Callable[[float, str], None]] = None,
        cancelled: Optional[Callable[[], bool]] = None):
    """(c, t, z, y, x) float32 -> ((t, z, y, x) uint32 labels, info, extras).

    `params`:
        model        a model folder (or its model.json)
        task         segment | prompt; what model.json's `tasks` lists
        objects      Prompt: [{"box": [x0, y0, z0, x1, y1, z1], "points": [[x, y, z], ...],
                     "point_labels": [1|0, ...], "scribbles": [{"points": [...], "label": 1|0}]}, ...]
                     in VOXELS of this image, the application's (x, y, z) order. One entry is ONE
                     object and ONE mask holding all of its prompts, so a correction refines it; the
                     i-th object's mask is label i + 1. (Bare points / boxes / scribbles are accepted
                     too, one mask each, before the objects.)
        snap_z       Prompt: move a lone object click along z to the model's own distance peak
                     (default true)
        threshold    Segment: the foreground threshold; <= 0 uses model.json's decode.fg_threshold
        min_voxels   drop smaller objects; <= 0 uses model.json's decode.min_voxels (Segment)
        voxel_um     (x, y, z) of THIS image: compared with the model's, and said when far
        tile         ignored: the model's window is its own crop (said in `warnings`)

    The model normalises its input itself (model.json's input.normalisation): raw intensities go
    to it unchanged.
    """
    def report(f: float, msg: str = "") -> None:
        if progress:
            progress(max(0.0, min(1.0, float(f))), msg)

    def check() -> None:
        if cancelled and cancelled():
            raise Cancelled("cancelled")

    a = _as_ctzyx(volume)
    n_c, n_t, n_z, n_y, n_x = a.shape
    path = str(params.get("model") or "")
    folder = model_folder(path)
    man = read_manifest(folder)
    facts = _summary(folder, man)
    name = f"{facts['name']} {facts['version']}".strip()
    tasks = facts["tasks"]
    task = str(params.get("task") or tasks[0]).lower()
    if task not in ("segment", "prompt", "detect", "track"):
        raise ValueError(f"unknown task '{task}'; a model folder offers {', '.join(tasks)}")
    if task not in tasks:
        if task == "prompt":
            raise ModelError(f"{name} cannot be prompted: it has no prompt decoder (it offers "
                             f"{', '.join(tasks)}). Choose Segment, or a promptable model")
        raise ModelError(f"{name} offers {', '.join(tasks)}; it cannot {task}")

    # the input contract: channels are checked here, the normalisation is the model's own
    want = int(facts["channels"])
    if want <= 1 and n_c > 1:
        raise ModelError(f"{name} takes one channel and was sent {n_c}: choose 'Selected channel'")
    if want > 1 and n_c != want:
        merge = f" ({facts['channel_merge']})" if facts["channel_merge"] else ""
        raise ModelError(f"{name} takes {want} channels{merge} and was sent {n_c}: choose 'All channels' on an "
                         f"image with {want}")
    multi = want > 1
    warnings: List[str] = []
    w = voxel_warning(params.get("voxel_um"), facts)
    if w:
        warnings.append(w)
    tile = params.get("tile")
    if isinstance(tile, (list, tuple)) and any(float(v or 0) > 0 for v in tile):
        warnings.append(f"Tile is ignored: {name} answers in its own window, {facts['crop']} (z, y, x)")

    check()
    report(0.02, f"loading {name}")
    m = load_model(folder, device)
    check()
    decode = facts["decode"]
    labels = np.zeros((n_t, n_z, n_y, n_x), np.uint32)
    info: Dict[str, Any] = {"task": task, "model": name, "path": folder, "channels": int(n_c), "frames": int(n_t),
                            "voxel_um": list(params.get("voxel_um") or []), "model_voxel_um": facts["voxel_um"],
                            "normalisation": facts["input"].get("normalisation"), "device": str(device)}
    extras: Dict[str, Any] = {}

    if task == "prompt":
        if n_t != 1:
            raise ValueError(f"the Prompt task takes one time point, this image has {n_t}; prompt a single time point")
        from . import models as model_hub

        pr = model_hub.app_prompts_to_zyx(params, (n_z, n_y, n_x))
        zyx, plab, bz, sz, oz = pr["points"], pr["point_labels"], pr["boxes"], pr["scribbles"], pr["objects"]
        report(0.1, f"{pr['count']} prompt(s)")
        vol = a[:, 0] if multi else a[0, 0]
        snap = bool(params.get("snap_z", True))
        got = m.call("prompt", vol, objects=oz or None, points=zyx if len(zyx) else None,
                     point_labels=plab if len(zyx) else None, boxes=bz if len(bz) else None,
                     scribbles=sz or None, channels=multi, snap_z=snap)
        check()
        if not isinstance(got, tuple) or len(got) < 2:
            raise ModelError(f"{name}: prompt() returned {type(got).__name__}, not (masks, scores, info)")
        masks, scores = np.asarray(got[0]), np.asarray(got[1], np.float32).reshape(-1)
        nfo = got[2] if len(got) > 2 and isinstance(got[2], dict) else {}
        if masks.ndim != 4 or masks.shape[1:] != (n_z, n_y, n_x):
            raise ModelError(f"{name}: prompt() returned masks of shape {masks.shape}, not (n, {n_z}, {n_y}, {n_x})")
        min_voxels = max(0, int(params.get("min_voxels", 0) or 0))
        conf = np.zeros((1, n_z, n_y, n_x), np.float32)
        # Later prompts win where two masks overlap, which is what a person adding a point expects.
        for i, mk in enumerate(masks.astype(bool)):
            report(0.2 + 0.7 * (i + 1) / max(len(masks), 1), f"mask {i + 1}/{len(masks)}")
            if min_voxels and int(mk.sum()) < min_voxels:
                continue
            labels[0][mk] = i + 1
            conf[0][mk] = float(scores[i]) if i < len(scores) else 0.0
        info["snap_z"] = snap
        info["prompts"] = int(pr["count"])
        info["prompt_kinds"] = {"points": int(len(zyx)), "boxes": int(len(bz)), "scribbles": int(len(sz)),
                                "objects": int(len(oz))}
        info["mask_scores"] = [round(float(v), 4) for v in scores]
        for k in ("clipped", "window", "windows", "snapped"):
            if k in nfo:
                info[k] = nfo[k]
        if nfo.get("clipped"):
            warnings.append(f"a prompt did not fit {name}'s window and was clipped (masks {nfo['clipped']}); "
                            "for an object deeper than the window a click beats a box")
        info["objects"] = int(labels.max())
        extras["confidence"] = conf
    else:
        given = float(params.get("threshold", 0) or 0)
        thr = given if given > 0 else float(decode.get("fg_threshold", 0.5) or 0.5)
        mv = int(params.get("min_voxels", 0) or 0)
        mv = mv if mv > 0 else int(decode.get("min_voxels", 0) or 0)
        total = 0
        for t in range(n_t):
            check()
            report(0.05 + 0.9 * t / max(n_t, 1), f"time point {t + 1}/{n_t}")
            vol = a[:, t] if multi else a[0, t]
            # both always given: the model keeps an override on itself, and a cached model must
            # answer the next run with that run's values, not this one's
            lab = np.asarray(m.call("segment", vol, channels=multi, threshold=thr, min_voxels=mv))
            if lab.shape != (n_z, n_y, n_x):
                raise ModelError(f"{name}: segment() returned shape {lab.shape}, not ({n_z}, {n_y}, {n_x})")
            labels[t] = _dense(lab)
            total += int(labels[t].max())
        info["threshold"] = thr
        info["min_voxels"] = mv
        info["decode"] = {k: decode.get(k) for k in ("kind", "fg_threshold", "seed_hmax", "seed_hrel", "seed_sigma")
                          if k in decode}
        info["objects"] = int(total)
    if warnings:
        info["warnings"] = warnings
    report(1.0, "done")
    return labels, info, extras
