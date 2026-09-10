"""Segmentation models: where they come from and how they run.

A model *spec* is the string the application's Torch segmentation step holds:

    /path/to/model.pt            TorchScript (.pt / .pts / .pth) or ONNX (.onnx) file
    hf:<repo_id>[:<filename>]    a file from a Hugging Face repository, downloaded once
                                 into the cache ($SIRIUS_MODEL_CACHE or ~/.sirius/models)
    cellpose:<model>             the `cellpose` package: one of its model names (cpsam on
                                 Cellpose 4, cyto3 / nuclei on Cellpose 3), or a path / hf:
                                 spec of a custom model file. The model has to be named.
    microsam:<model_type>        the `micro_sam` package's automatic instance
                                 segmentation: vit_b_lm, vit_l_lm, vit_t_lm, vit_b_em_organelles, ...

File models return per-tile probabilities (the application turns them into
labels); the package families return instance labels directly.  The optional
packages are imported lazily so the worker starts without them and reports a
`pip install` hint instead of failing.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

ProgressFn = Optional[Callable[[float, str], None]]
CancelFn = Optional[Callable[[], bool]]

MODEL_EXTENSIONS = (".pt", ".pts", ".pth", ".onnx")
FAMILIES = ("file", "hf", "cellpose", "microsam")

# `install` runs pip / conda inside the worker's environment on a client's
# request. Off unless the worker was started with --allow-install (the local
# launcher passes it; a shared cluster node must not): see app/python/SECURITY.md.
ALLOW_INSTALL = False

CELLPOSE_MODELS = ("cyto3", "nuclei", "cyto2", "cyto", "livecell", "tissuenet", "yeast_PhC", "yeast_BF",
                   "bact_phase", "bact_fluor", "deepbacs", "cyto2_cp3")
MICROSAM_MODELS = ("vit_b_lm", "vit_l_lm", "vit_t_lm", "vit_b_em_organelles", "vit_l_em_organelles",
                   "vit_t_em_organelles", "vit_b", "vit_l", "vit_h")

INSTALL_HINTS = {
    "btrack": "pip install btrack",
    "cellpose": "pip install cellpose",
    "microsam": "conda install -c conda-forge micro_sam   (or: pip install micro-sam)",
    "hf": "pip install huggingface_hub",
    "onnx": "pip install onnxruntime",
}

# pip names, and the conda-forge name where the authors recommend conda
PACKAGES = {
    "cellpose": {"pip": ["cellpose"], "conda": None},
    "microsam": {"pip": ["micro-sam"], "conda": ["micro_sam"]},
    "hf": {"pip": ["huggingface_hub"], "conda": None},
    "onnx": {"pip": ["onnxruntime"], "conda": None},
    "btrack": {"pip": ["btrack"], "conda": None},
}


class ModelError(Exception):
    pass


class NotAvailable(ModelError):
    """An optional package (cellpose, micro_sam, huggingface_hub) is missing."""


# --- specs -----------------------------------------------------------------------------


class ModelSpec:
    """Parsed model spec."""

    def __init__(self, family: str, name: str = "", filename: str = "", raw: str = "") -> None:
        self.family = family          # "file" | "hf" | "cellpose" | "microsam"
        self.name = name              # path, repo id, or model name
        self.filename = filename      # hf: file inside the repo (may be empty)
        self.raw = raw or self.text()

    def text(self) -> str:
        if self.family == "file":
            return self.name
        if self.family == "hf":
            return f"hf:{self.name}:{self.filename}" if self.filename else f"hf:{self.name}"
        return f"{self.family}:{self.name}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ModelSpec({self.text()!r})"


def parse_spec(spec: str) -> ModelSpec:
    s = (spec or "").strip()
    if not s:
        raise ModelError("no model given")
    low = s.lower()
    if low.startswith("hf:") or low.startswith("huggingface:"):
        body = s.split(":", 1)[1]
        if body.startswith("//"):
            body = body[2:]
        repo, _, filename = body.partition(":")
        repo = repo.strip().strip("/")
        if repo.count("/") != 1 or not all(repo.split("/")):
            raise ModelError(f"hf spec '{s}': expected hf:<owner>/<repo>[:<filename>]")
        return ModelSpec("hf", repo, filename.strip(), s)
    if low.startswith("cellpose:"):
        name = s.split(":", 1)[1].strip()
        if not name:
            raise ModelError("cellpose spec needs a model name, e.g. cellpose:cpsam; "
                             "the installed version's names are in the hub dialog")
        return ModelSpec("cellpose", name, "", s)
    if low.startswith("microsam:") or low.startswith("micro-sam:") or low.startswith("micro_sam:"):
        name = s.split(":", 1)[1].strip()
        if not name:
            raise ModelError("microsam spec needs a model type, e.g. microsam:vit_b_lm")
        return ModelSpec("microsam", name, "", s)
    return ModelSpec("file", s, "", s)


def is_family_spec(spec: str) -> bool:
    try:
        return parse_spec(spec).family in ("cellpose", "microsam")
    except ModelError:
        return False


def is_file_spec(spec: str) -> bool:
    try:
        return parse_spec(spec).family == "file"
    except ModelError:
        return False


# --- cache -----------------------------------------------------------------------------


def cache_dir() -> Path:
    env = os.environ.get("SIRIUS_MODEL_CACHE", "").strip()
    return Path(env).expanduser() if env else Path.home() / ".sirius" / "models"


def repo_dir(repo: str) -> Path:
    # one flat directory per repository; "/" is not a legal file-name character
    return cache_dir() / "hf" / repo.replace("/", "--")


def cached_path(repo: str, filename: str) -> Optional[str]:
    """Local path of an already-downloaded repository file, else None."""
    if not filename:
        return None
    root = repo_dir(repo).resolve()
    p = (repo_dir(repo) / filename).resolve()
    # a file name with ".." or a drive in it must not name a file outside
    # the repository's cache directory
    if p != root and root not in p.parents:
        return None
    return str(p) if p.is_file() else None


def list_cached_models() -> List[Dict[str, Any]]:
    """Model files in the cache: [{spec, path, bytes, repo, file}]."""
    out: List[Dict[str, Any]] = []
    root = cache_dir()
    hf = root / "hf"
    if hf.is_dir():
        for rdir in sorted(hf.iterdir()):
            if not rdir.is_dir():
                continue
            repo = rdir.name.replace("--", "/", 1)
            for f in sorted(rdir.rglob("*")):
                if f.is_file() and f.suffix.lower() in MODEL_EXTENSIONS and not f.name.startswith("."):
                    rel = f.relative_to(rdir).as_posix()
                    out.append({"spec": f"hf:{repo}:{rel}", "path": str(f), "bytes": f.stat().st_size,
                                "repo": repo, "file": rel})
    local = root / "local"
    if local.is_dir():
        for f in sorted(local.iterdir()):
            if f.is_file() and f.suffix.lower() in MODEL_EXTENSIONS:
                out.append({"spec": str(f), "path": str(f), "bytes": f.stat().st_size, "repo": "", "file": f.name})
    return out


def delete_cached_model(path: str) -> Dict[str, Any]:
    """Remove one model from the cache and report what that freed.

    Only paths inside the cache are touched -- a spec can name a file anywhere
    on the machine, and a model chosen from outside the cache is the user's
    file, not ours to delete. An empty repository directory goes with the last
    file in it, so the Local list does not fill up with empty rows.
    """
    if not path:
        raise ModelError("no model given to delete")
    root = cache_dir().resolve()
    target = Path(path).expanduser()
    try:
        resolved = target.resolve()
    except OSError as e:
        raise ModelError(f"cannot resolve {path}: {e}") from e
    if not resolved.is_relative_to(root):
        raise ModelError(f"{path} is not in the model cache ({root}); delete it yourself if you meant to")
    if resolved == root:
        # is_relative_to() is true for the root itself, and a client naming
        # the cache directory would otherwise rmtree every cached model.
        raise ModelError(f"{path} is the model cache itself, not a model in it")
    if not resolved.exists():
        raise ModelError(f"{path} is already gone")

    freed = 0
    if resolved.is_dir():
        for f in resolved.rglob("*"):
            if f.is_file():
                freed += f.stat().st_size
        shutil.rmtree(resolved)
    else:
        freed = resolved.stat().st_size
        resolved.unlink()

    # prune the directories the file left behind, but never the cache itself
    removed_dirs = []
    parent = resolved.parent
    while parent != root and parent.is_relative_to(root):
        if any(parent.iterdir()):
            break
        parent.rmdir()
        removed_dirs.append(str(parent))
        parent = parent.parent
    return {"path": str(resolved), "bytes": freed, "removed_directories": removed_dirs}


# --- Hugging Face ------------------------------------------------------------------------


# The access token of the request being served (set_hub_token), handed to each
# huggingface_hub call as its `token=` argument. It is never written to the
# environment: HF_TOKEN would outlive the request and reach every subprocess
# the worker starts (pip, conda). None lets huggingface_hub fall back to its
# own sources (HF_TOKEN in the worker's environment, `huggingface-cli login`).
# Per thread: the connection thread serves the synchronous methods while a
# job thread may be inside a download, and the two must not swap each
# other's token (the server sets the job thread's copy when the job starts).
_request = threading.local()


def _token_arg(token: Optional[str]) -> Optional[str]:
    """`token` when given, else the request's token; "" means "none given"."""
    token = (token if token is not None else current_request_token()) or ""
    return token.strip() or None


def current_request_token() -> Optional[str]:
    """The token set_hub_token gave this thread, or None."""
    return getattr(_request, "token", None)


def _hf_api(token: Optional[str] = None):
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError as e:
        raise NotAvailable(f"Hugging Face access needs the 'huggingface_hub' package ({INSTALL_HINTS['hf']})") from e
    return HfApi(token=_token_arg(token))


def _hub_error(e: BaseException, repo: str) -> ModelError:
    """huggingface_hub failures as one sentence that says what to do."""
    name = e.__class__.__name__
    text = str(e).strip()
    if name == "GatedRepoError" or "gated" in text.lower():
        return ModelError(
            f"{repo} is a gated repository: sign in at https://huggingface.co/{repo}, accept its terms, "
            "then paste an access token (Hugging Face settings > Access Tokens) into the hub's Token field "
            "or Preferences > Compute, or run `huggingface-cli login` for the worker's Python.")
    if name == "RepositoryNotFoundError":
        return ModelError(f"{repo} was not found on Hugging Face (a private repository needs your access token).")
    if name == "EntryNotFoundError":
        return ModelError(f"{repo}: no such file in the repository ({text.splitlines()[0] if text else name}).")
    if name in ("LocalEntryNotFoundError", "OfflineModeIsEnabled") or "Connection" in name:
        return ModelError(f"Hugging Face is unreachable from the worker ({name}).")
    return ModelError(f"{repo}: {text.splitlines()[0] if text else name}")


# fields list_models omits unless asked for; gated repositories need a token
_SEARCH_EXPAND = ["gated", "private", "downloads", "likes", "tags", "pipeline_tag", "lastModified", "library_name"]


def hub_search(query: str, limit: int = 25, filter_tag: str = "", token: Optional[str] = None) -> List[Dict[str, Any]]:
    api = _hf_api()
    kwargs: Dict[str, Any] = {"search": query or None, "limit": max(1, int(limit)), "sort": "downloads",
                              "direction": -1}
    if filter_tag:
        kwargs["filter"] = filter_tag
    try:
        found = list(api.list_models(expand=_SEARCH_EXPAND, **kwargs))
    except (TypeError, ValueError):   # older huggingface_hub without expand
        found = list(api.list_models(**kwargs))
    except Exception as e:  # noqa: BLE001
        raise _hub_error(e, query or "search") from e
    out = []
    for m in found:
        gated = getattr(m, "gated", None)
        out.append({
            "id": m.id,
            "downloads": int(getattr(m, "downloads", 0) or 0),
            "likes": int(getattr(m, "likes", 0) or 0),
            "tags": list(getattr(m, "tags", []) or [])[:8],
            "last_modified": str(getattr(m, "last_modified", "") or ""),
            "pipeline_tag": str(getattr(m, "pipeline_tag", "") or ""),
            "library": str(getattr(m, "library_name", "") or ""),
            "gated": str(gated) if gated else False,
            "private": bool(getattr(m, "private", False)),
        })
    return out


def hub_files(repo: str, token: Optional[str] = None) -> List[Dict[str, Any]]:
    api = _hf_api()
    try:
        info = api.model_info(repo, files_metadata=True)
    except Exception as e:  # noqa: BLE001
        raise _hub_error(e, repo) from e
    files = []
    for s in getattr(info, "siblings", None) or []:
        name = getattr(s, "rfilename", "")
        size = getattr(s, "size", None)
        files.append({"name": name, "size": int(size) if size is not None else -1,
                      "model": name.lower().endswith(MODEL_EXTENSIONS)})
    files.sort(key=lambda f: (not f["model"], f["name"]))
    return files


def pick_model_file(repo: str, token: Optional[str] = None) -> str:
    """The single model file of a repository, or an error naming the candidates."""
    candidates = [f["name"] for f in hub_files(repo, token) if f["model"]]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ModelError(f"hf:{repo} holds no TorchScript / ONNX file; give hf:{repo}:<filename>")
    raise ModelError(f"hf:{repo} holds several model files: " + ", ".join(candidates[:10]) +
                     f"; choose one as hf:{repo}:<filename>")


def _progress_tqdm(progress: ProgressFn):
    """A tqdm subclass whose updates call `progress(fraction, message)`; None
    when tqdm is not importable (then only start/end are reported)."""
    if progress is None:
        return None
    try:
        from tqdm.auto import tqdm  # type: ignore
    except ImportError:
        return None

    class _Tqdm(tqdm):  # type: ignore[misc]
        def update(self, n=1):
            super().update(n)
            total = self.total or 0
            if total > 0:
                progress(min(1.0, self.n / total), f"{self.n / 2**20:.0f} / {total / 2**20:.0f} MB")

    return _Tqdm


def hub_download(repo: str, filename: str = "", progress: ProgressFn = None, token: Optional[str] = None) -> str:
    """Download one repository file into the cache; returns the local path.
    Files already present are not fetched again."""
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as e:
        raise NotAvailable(f"Hugging Face downloads need 'huggingface_hub' ({INSTALL_HINTS['hf']})") from e
    if not filename:
        filename = pick_model_file(repo, token)
    have = cached_path(repo, filename)
    if have:
        if progress:
            progress(1.0, "cached")
        return have
    target = repo_dir(repo)
    target.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(0.0, f"downloading {filename}")
    kwargs: Dict[str, Any] = {"repo_id": repo, "filename": filename, "local_dir": str(target),
                              "token": _token_arg(token)}
    tqdm_class = _progress_tqdm(progress)
    if tqdm_class is not None:
        kwargs["tqdm_class"] = tqdm_class
    try:
        try:
            path = hf_hub_download(**kwargs)
        except TypeError:   # older huggingface_hub without tqdm_class
            kwargs.pop("tqdm_class", None)
            path = hf_hub_download(**kwargs)
    except (ModelError, RuntimeError):
        raise
    except Exception as e:  # noqa: BLE001
        raise _hub_error(e, repo) from e
    if progress:
        progress(1.0, filename)
    return str(Path(path).resolve())


def set_hub_token(token: str) -> None:
    """The access token for gated / private repositories that the hub calls of
    the current request use (the server calls this before each of them; an
    empty token means "the client sent none"). Kept out of os.environ."""
    _request.token = (token or "").strip() or None


def resolve(spec: str, progress: ProgressFn = None) -> Tuple[ModelSpec, str]:
    """Spec -> (parsed spec, local file path) for file and hf models; family
    specs resolve to (spec, model name)."""
    ms = parse_spec(spec)
    if ms.family == "file":
        if not os.path.exists(ms.name):
            raise FileNotFoundError(ms.name)
        return ms, ms.name
    if ms.family == "hf":
        return ms, hub_download(ms.name, ms.filename, progress)
    return ms, ms.name


# --- families ----------------------------------------------------------------------------


def family_available(family: str) -> Tuple[bool, str]:
    """(importable, install hint)."""
    if family == "cellpose":
        try:
            import cellpose  # type: ignore  # noqa: F401
            return True, ""
        except ImportError:
            return False, INSTALL_HINTS["cellpose"]
    if family == "microsam":
        try:
            import micro_sam  # type: ignore  # noqa: F401
            return True, ""
        except ImportError:
            return False, INSTALL_HINTS["microsam"]
    if family == "btrack":
        from . import tracking as tracking_backends

        return tracking_backends.available()
    if family == "hf":
        try:
            import huggingface_hub  # type: ignore  # noqa: F401
            return True, ""
        except ImportError:
            return False, INSTALL_HINTS["hf"]
    return True, ""


def _conda_executable() -> Optional[str]:
    """The conda that owns this interpreter's environment, or None."""
    if not os.path.isdir(os.path.join(sys.prefix, "conda-meta")):
        return None
    for candidate in (os.environ.get("CONDA_EXE", ""), shutil.which("conda"), shutil.which("mamba")):
        if candidate and os.path.exists(candidate):
            return candidate
    # a conda env under <base>/envs/<name>, or the base itself
    for base in (Path(sys.prefix).parent.parent, Path(sys.prefix)):
        for exe in ("conda", "mamba"):
            c = base / "bin" / exe
            if c.exists():
                return str(c)
    return None


def _installed_by(distribution: str) -> str:
    """"pip", "conda", ... from the distribution's INSTALLER record, or ""."""
    try:
        from importlib import metadata
        text = metadata.distribution(distribution).read_text("INSTALLER") or ""
        return text.strip().lower()
    except Exception:  # noqa: BLE001
        return ""


def install_plan(family: str) -> Dict[str, Any]:
    """How `install` would add a family's package to this interpreter:
    {family, installer: pip|conda, command: [...], display, packages, python, note}.
    conda is used for packages whose authors recommend it, unless torch here
    came from pip: conda would then add its own torch build next to it.
    Advisory only -- the dialog shows the command; `install` runs it, and
    refuses without --allow-install."""
    if family not in PACKAGES:
        raise ModelError(f"nothing to install for '{family}'")
    spec = PACKAGES[family]
    conda = _conda_executable() if spec["conda"] else None
    if conda and _installed_by("torch") == "pip":
        conda = None
        pip_note = "pip, because torch here was installed by pip (conda would add its own torch build)"
    else:
        pip_note = "pip (no conda found for this interpreter; pip pulls a large dependency set for micro-sam)"
    if conda:
        return {"family": family, "installer": "conda", "packages": list(spec["conda"]), "python": sys.executable,
                "command": [conda, "install", "-y", "-p", sys.prefix, "-c", "conda-forge", *spec["conda"]],
                "display": "conda install -c conda-forge " + " ".join(spec["conda"]),
                "note": "conda-forge, as the package's authors recommend; conda may also swap in its own torch build"}
    return {"family": family, "installer": "pip", "packages": list(spec["pip"]), "python": sys.executable,
            "command": [sys.executable, "-m", "pip", "install", *spec["pip"]],
            "display": "pip install " + " ".join(spec["pip"]),
            "note": "into the worker's Python" if not spec["conda"] else pip_note}


def install(family: str, progress: ProgressFn = None, cancelled: CancelFn = None, dry_run: bool = False) -> Dict[str, Any]:
    """Run the family's install command, streaming its output lines through
    `progress`; returns {ok, returncode, available, command, tail}. `dry_run`
    adds pip's --dry-run (conda: --dry-run) so nothing is changed.

    A real install is refused unless the worker was started with
    --allow-install: it is code execution in the worker's environment, which a
    shared node must not hand to whoever holds the token. `dry_run` is always
    allowed -- it resolves against the index and writes nothing, and the model
    hub dialog uses it to preview the command (app/python/SECURITY.md)."""
    plan = install_plan(family)
    if not (ALLOW_INSTALL or dry_run):
        raise ModelError(f"this worker does not install packages; start it with --allow-install, or run "
                         f"'{plan['display']}' in {sys.executable} yourself")
    command = list(plan["command"])
    if dry_run:
        command.append("--dry-run")
    if progress:
        progress(0.0, "$ " + " ".join(command))
    env = dict(os.environ)
    # The tokens the worker holds are for the hub and for its own clients;
    # an installer (and every setup.py it runs) has no business seeing them.
    for secret in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "SIRIUS_TOKEN"):
        env.pop(secret, None)
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env=env, errors="replace")
    tail: List[str] = []
    fraction = 0.02
    milestones = (("Collecting", 0.1), ("Downloading", 0.3), ("Installing collected", 0.8), ("Successfully", 1.0),
                  ("Solving environment", 0.1), ("Downloading and Extracting", 0.4), ("Executing transaction", 0.8),
                  ("Preparing transaction", 0.7))
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        tail.append(line)
        if len(tail) > 60:
            del tail[0]
        for key, f in milestones:
            if key in line:
                fraction = max(fraction, f)
        if progress:
            progress(min(fraction, 0.98), line[-300:])
        if cancelled and cancelled():
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise RuntimeError("cancelled")
    rc = proc.wait()
    importlib.invalidate_caches()
    available = family_available(family)[0] if not dry_run else False
    if progress:
        progress(1.0, "done" if rc == 0 else f"exit code {rc}")
    return {"ok": rc == 0, "returncode": rc, "available": available, "command": command, "installer": plan["installer"],
            "tail": tail[-25:], "family": family}


def _cellpose_models_module():
    from cellpose import models  # type: ignore
    return models


def cellpose_model_names() -> List[str]:
    """The installed Cellpose's built-in model names (empty when not installed)."""
    try:
        models = _cellpose_models_module()
    except ImportError:
        return []
    names = list(getattr(models, "MODEL_NAMES", []) or [])
    return names or list(CELLPOSE_MODELS)


def _cellpose_is_v4(models) -> bool:
    return any(str(n).startswith("cpsam") for n in getattr(models, "MODEL_NAMES", []) or [])


def microsam_model_names() -> List[str]:
    try:
        from micro_sam import util  # type: ignore
        registry = util.models().registry
        names = [k for k in registry if not k.endswith("_decoder")]
        return names or list(MICROSAM_MODELS)
    except Exception:  # noqa: BLE001
        return []


def _module_version(name: str) -> str:
    try:
        mod = importlib.import_module(name)
    except Exception:  # noqa: BLE001
        return ""
    return str(getattr(mod, "__version__", "") or getattr(mod, "version", "") or "")


def weights_cached(spec: str) -> Optional[bool]:
    """Whether a family model's weights are already on this machine; None
    when it cannot be told (package missing)."""
    ms = parse_spec(spec)
    try:
        if ms.family == "cellpose":
            models = _cellpose_models_module()
            name = ms.name
            if os.path.exists(name):
                return True
            root = Path(str(getattr(models, "MODEL_DIR", Path.home() / ".cellpose" / "models")))
            return any((root / candidate).exists() for candidate in (name, f"{name}torch_0", f"{name}_0"))
        if ms.family == "microsam":
            from micro_sam import util  # type: ignore
            root = Path(util.get_cache_directory()) / "models"
            return (root / ms.name).exists()
        if ms.family == "hf":
            return cached_path(ms.name, ms.filename) is not None
    except Exception:  # noqa: BLE001
        return None
    return None


def prepare(spec: str, progress: ProgressFn = None, cancelled: CancelFn = None) -> Dict[str, Any]:
    """Fetch a model's weights now rather than on the first run: family
    models through their packages, hf: files through the cache."""
    ms = parse_spec(spec)
    if progress:
        progress(0.02, f"fetching {ms.text()}")
    if ms.family == "hf":
        path = hub_download(ms.name, ms.filename, progress)
        return {"spec": ms.text(), "path": path, "cached": True}
    if ms.family == "cellpose":
        try:
            models = _cellpose_models_module()
        except ImportError as e:
            raise NotAvailable(f"{ms.text()} needs the 'cellpose' package ({INSTALL_HINTS['cellpose']})") from e
        model = _cellpose_model(models, ms.name, gpu=False, progress=progress)
        path = str(getattr(model, "pretrained_model", "") or "")
        del model
        _check(cancelled)
        if progress:
            progress(1.0, "weights ready")
        return {"spec": ms.text(), "path": path, "cached": True}
    if ms.family == "microsam":
        try:
            from micro_sam import util  # type: ignore
            from micro_sam.automatic_segmentation import get_predictor_and_segmenter  # type: ignore
        except ImportError as e:
            raise NotAvailable(f"{ms.text()} needs the 'micro_sam' package ({INSTALL_HINTS['microsam']})") from e
        predictor, segmenter = get_predictor_and_segmenter(model_type=ms.name, device="cpu", amg=False, is_tiled=False)
        del predictor, segmenter
        _check(cancelled)
        if progress:
            progress(1.0, "weights ready")
        return {"spec": ms.text(), "path": str(Path(util.get_cache_directory()) / "models" / ms.name), "cached": True}
    raise ModelError(f"'{spec}' names a local file; nothing to fetch")


def family_info(spec: str) -> Dict[str, Any]:
    """What model_info reports for cellpose:/microsam: specs."""
    ms = parse_spec(spec)
    available, hint = family_available(ms.family)
    info: Dict[str, Any] = {
        "spec": ms.text(), "family": ms.family, "model": ms.name,
        "format": {"cellpose": "cellpose", "microsam": "micro-sam"}.get(ms.family, ms.family),
        "available": available, "install_hint": hint, "returns": "labels",
        "dtype": "float32", "input_shape": [1, 1, -1, -1, -1], "output_shape": ["labels", -1, -1, -1],
        "python": sys.executable,
    }
    try:
        plan = install_plan(ms.family)
        info["install"] = {"installer": plan["installer"], "command": " ".join(plan["command"]), "display": plan["display"],
                           "note": plan["note"]}
    except ModelError:
        pass
    if ms.family == "cellpose":
        names = cellpose_model_names() if available else list(CELLPOSE_MODELS)
        info["known_models"] = names
        if available:
            info["version"] = _module_version("cellpose")
            if ms.name and ms.name not in names and not os.path.exists(ms.name) \
                    and not ms.name.lower().startswith("hf:"):
                info["warning"] = f"cellpose {info['version']} has no model '{ms.name}'; it offers " + ", ".join(names)
    elif ms.family == "microsam":
        info["known_models"] = (microsam_model_names() if available else []) or list(MICROSAM_MODELS)
        if available:
            info["version"] = _module_version("micro_sam")
    if available:
        cached = weights_cached(spec)
        if cached is not None:
            info["weights_cached"] = cached
    return info


def _normalize(volume: np.ndarray) -> np.ndarray:
    v = np.asarray(volume, dtype=np.float32)
    lo, hi = np.percentile(v, (1.0, 99.9))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _check(cancelled: CancelFn) -> None:
    if cancelled and cancelled():
        raise RuntimeError("cancelled")


def _cellpose_model(models, model_name: str, gpu: bool, progress: ProgressFn = None):
    """A CellposeModel for a name, path or hf: spec across Cellpose 3 and 4
    (4 keeps only its own built-in models and would silently fall back)."""
    name = model_name
    if not name:
        raise ModelError("cellpose spec needs a model name, e.g. cellpose:cpsam")
    if name.lower().startswith("hf:") or os.path.exists(name):
        _, path = resolve(name, progress)
        return models.CellposeModel(gpu=gpu, pretrained_model=path)
    names = list(getattr(models, "MODEL_NAMES", []) or [])
    if names and name not in names:
        raise ModelError(f"cellpose {_module_version('cellpose')} has no model '{name}'; it offers " + ", ".join(names) +
                         "")
    if _cellpose_is_v4(models):
        return models.CellposeModel(gpu=gpu, pretrained_model=name)
    return models.CellposeModel(gpu=gpu, model_type=name)


def run_cellpose(volume: np.ndarray, model_name: str, params: Dict[str, Any], device: str = "auto",
                 progress: ProgressFn = None, cancelled: CancelFn = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Instance labels (uint32 (z, y, x)) from the Cellpose package, plus the
    cell probability as one channel when the model reports flows."""
    try:
        from cellpose import models  # type: ignore
    except ImportError as e:
        raise NotAvailable(f"cellpose:{model_name} needs the 'cellpose' package ({INSTALL_HINTS['cellpose']})") from e
    gpu = device != "cpu"
    model = _cellpose_model(models, model_name, gpu=gpu, progress=progress)
    z = volume.shape[0]
    diameter = params.get("diameter")
    diameter = float(diameter) if diameter not in (None, "", 0, "0") else None
    do_3d = bool(params.get("do_3d", z > 1 and str(params.get("mode", "3D")) == "3D"))
    kwargs: Dict[str, Any] = {"diameter": diameter, "do_3D": bool(do_3d and z > 1)}
    if not _cellpose_is_v4(models):
        kwargs["channels"] = [0, 0]   # grayscale; Cellpose 4 infers it
    if z > 1:
        kwargs["z_axis"] = 0          # (z, y, x) stacks, single channel
    if params.get("anisotropy") not in (None, "", 0):
        kwargs["anisotropy"] = float(params["anisotropy"])
    if not kwargs["do_3D"] and z > 1:
        kwargs["stitch_threshold"] = float(params.get("stitch_threshold", 0.5))
    for key in ("flow_threshold", "cellprob_threshold"):
        if params.get(key) not in (None, ""):
            kwargs[key] = float(params[key])
    if progress:
        progress(0.05, f"cellpose {model_name}")

        class _Bar:
            # Cellpose drives a Qt-style bar (setValue 0..100) through its stages
            def setValue(self, v):  # noqa: N802 - Cellpose's expectation
                progress(0.05 + 0.85 * min(max(float(v), 0.0), 100.0) / 100.0, f"cellpose {model_name}")

        kwargs["progress"] = _Bar()
    data = _normalize(volume) if params.get("normalize", True) else np.asarray(volume, dtype=np.float32)
    result = model.eval(data if z > 1 else data[0], **kwargs)
    _check(cancelled)
    masks = result[0]
    flows = result[1] if len(result) > 1 else None
    labels = np.asarray(masks, dtype=np.uint32)
    if labels.ndim == 2:
        labels = labels[np.newaxis]
    prob = None
    try:
        cellprob = flows[2] if flows is not None else None
        if cellprob is not None:
            p = 1.0 / (1.0 + np.exp(-np.asarray(cellprob, dtype=np.float32)))
            if p.ndim == 2:
                p = p[np.newaxis]
            if p.shape == labels.shape:
                prob = np.ascontiguousarray(p[np.newaxis], dtype=np.float32)   # (1, z, y, x)
    except Exception:  # noqa: BLE001 - the probability map is optional
        prob = None
    if progress:
        progress(1.0, f"{int(labels.max()) if labels.size else 0} labels")
    return np.ascontiguousarray(labels), prob


def run_microsam(volume: np.ndarray, model_type: str, params: Dict[str, Any], device: str = "auto",
                 progress: ProgressFn = None, cancelled: CancelFn = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Instance labels from micro-SAM's automatic instance segmentation, plane
    by plane (or through its 3D z-linking for stacks in 3D mode)."""
    try:
        from micro_sam.automatic_segmentation import (  # type: ignore
            automatic_instance_segmentation,
            get_predictor_and_segmenter,
        )
    except ImportError as e:
        raise NotAvailable(f"microsam:{model_type} needs the 'micro_sam' package ({INSTALL_HINTS['microsam']})") from e
    z = volume.shape[0]
    checkpoint = params.get("checkpoint") or None
    if checkpoint and str(checkpoint).lower().startswith("hf:"):
        _, checkpoint = resolve(str(checkpoint), progress)
    if progress:
        progress(0.05, f"micro-sam {model_type}")
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type, checkpoint=checkpoint, device=None if device == "auto" else device,
        amg=bool(params.get("amg", False)), is_tiled=False)
    data = _normalize(volume) if params.get("normalize", True) else np.asarray(volume, dtype=np.float32)
    data8 = (np.clip(data, 0.0, 1.0) * 255.0).astype(np.uint8)
    if z > 1 and str(params.get("mode", "3D")) == "3D":
        labels = np.asarray(automatic_instance_segmentation(predictor=predictor, segmenter=segmenter,
                                                            input_path=data8, ndim=3, verbose=False), dtype=np.uint32)
    else:
        planes = []
        for k in range(z):
            _check(cancelled)
            lab = automatic_instance_segmentation(predictor=predictor, segmenter=segmenter, input_path=data8[k],
                                                  ndim=2, verbose=False)
            planes.append(np.asarray(lab, dtype=np.uint32))
            if progress:
                progress(0.05 + 0.9 * (k + 1) / z, f"plane {k + 1} / {z}")
        labels = np.stack(planes)
    if progress:
        progress(1.0, f"{int(labels.max()) if labels.size else 0} labels")
    return np.ascontiguousarray(labels), None


def run_family(spec: str, volume: np.ndarray, params: Dict[str, Any], device: str = "auto",
               progress: ProgressFn = None, cancelled: CancelFn = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Dispatch cellpose:/microsam: specs; returns (labels uint32 (z, y, x), prob (1, z, y, x) or None)."""
    ms = parse_spec(spec)
    volume = np.asarray(volume, dtype=np.float32)
    while volume.ndim > 3 and volume.shape[0] == 1:
        volume = volume[0]
    if volume.ndim == 2:
        volume = volume[np.newaxis]
    if volume.ndim != 3:
        raise ModelError(f"family models take a (z, y, x) volume, got shape {volume.shape}")
    if ms.family == "cellpose":
        return run_cellpose(volume, ms.name, params, device, progress, cancelled)
    if ms.family == "microsam":
        return run_microsam(volume, ms.name, params, device, progress, cancelled)
    raise ModelError(f"'{spec}' is not a model family spec")
