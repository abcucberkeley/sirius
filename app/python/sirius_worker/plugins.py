"""User-provided operations.

A plugin is one Python file in a plugin directory that defines

    STEP = {"kind": "dog_filter", "name": "Difference of Gaussians", "group": "Intensity",
            "params": [{"key": "sigma", "label": "σ", "type": "double", "default": 1.0,
                        "min": 0.1, "max": 50, "help": "..."}],
            "separable_over_t": True,      # optional: run one time point at a time
            "produces_labels": False,      # optional: returns instance labels
            "needs_labels": False,         # optional: receives the input's labels
            "help": "markdown"}            # optional: run.__doc__ otherwise

    def run(data, params, meta, ctx):
        # data: float32 (c, t, z, y, x); params: {key: value}; meta: dataset metadata
        # ctx.progress(fraction, message=""), ctx.cancelled(), ctx.labels (uint32 (t, z, y, x) or None)
        return output                          # (c, t, z, y, x) float32 (lower ranks are expanded)
        # or (output, labels) / (output, diagnostics) / (output, labels, diagnostics)
        # or {"output": ..., "labels": ..., "diagnostics": {...}, "meta": {...}}

Diagnostics is a dict with any of "summary" (str), "facts" ({name: value}),
"warnings" ([str]), "table" ({"header": [...], "rows": [[...]]}) and "images"
([{"title": str, "meta": str, "data": 2-D array, "log": bool}]).

Parameter types: double, int, bool, choice (with "choices"), path, string,
channel (index into the input's channels), axes ("ctzyx" subset),
double_list, string_list.

Directories searched, in order: $SIRIUS_PLUGIN_DIRS (os.pathsep separated),
~/.sirius/plugins, and the "plugins" directory beside the application
(app/plugins in a checkout).

A plugin runs with every right of the user, so on POSIX a directory or a file
that another user could have put there -- one owned by someone else (root
excepted), or writable by its group (unless that is the user's own private
group) or by everybody -- is skipped, and the log says why.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import re
import stat
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("sirius_worker")

PARAM_TYPES = ("double", "int", "bool", "choice", "path", "string", "channel", "axes", "double_list", "string_list")
_KIND_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


class PluginError(Exception):
    pass


class Plugin:
    def __init__(self, file: Path, spec: Dict[str, Any], run, error: str = "") -> None:
        self.file = file
        self.spec = spec
        self.run = run
        self.error = error

    @property
    def kind(self) -> str:
        return str(self.spec.get("kind", self.file.stem))

    def describe(self) -> Dict[str, Any]:
        d = dict(self.spec)
        d["file"] = str(self.file)
        if self.error:
            d["error"] = self.error
        return d


def plugin_dirs(extra: Optional[List[str]] = None) -> List[Path]:
    dirs: List[Path] = []
    for raw in (extra or []):
        if raw:
            dirs.append(Path(raw).expanduser())
    env = os.environ.get("SIRIUS_PLUGIN_DIRS", "")
    for raw in env.split(os.pathsep):
        if raw.strip():
            dirs.append(Path(raw.strip()).expanduser())
    dirs.append(Path.home() / ".sirius" / "plugins")
    # <exe>/plugins next to the app, app/plugins in a checkout: two levels above this package
    dirs.append(Path(__file__).resolve().parents[2] / "plugins")
    out: List[Path] = []
    for d in dirs:
        if d not in out:
            out.append(d)
    return out


def _private_group(st: os.stat_result) -> bool:
    """The user's own group in the user-private-group scheme (umask 002): the
    user's primary group, named after the user, with no other members."""
    try:
        import grp
        import pwd

        group = grp.getgrgid(st.st_gid)
        return (st.st_gid == os.getgid() and not group.gr_mem
                and group.gr_name == pwd.getpwuid(os.getuid()).pw_name)
    except (ImportError, KeyError, AttributeError, OSError):
        return False


def unsafe_reason(path: Path, st: Optional[os.stat_result] = None, uid: Optional[int] = None) -> str:
    """Why `path` must not be imported from ("" when it may): on POSIX, owned
    by another user than this one or root, or writable by others. Windows
    keeps a user's folders private by ACL and is not checked (the `st` and
    `uid` arguments let a test give the facts on any platform)."""
    if st is None:
        if os.name != "posix":
            return ""
        try:
            st = os.stat(path)
        except OSError:
            return ""
    if uid is None:
        uid = os.getuid() if hasattr(os, "getuid") else st.st_uid
    if st.st_uid not in (uid, 0):
        return f"{path} belongs to another user (uid {st.st_uid})"
    if st.st_mode & stat.S_IWOTH:
        return f"{path} is writable by every user"
    if st.st_mode & stat.S_IWGRP and not _private_group(st):
        return f"{path} is writable by its group"
    return ""


def _normalize_param(p: Dict[str, Any], where: str) -> Dict[str, Any]:
    if not isinstance(p, dict) or "key" not in p:
        raise PluginError(f"{where}: every parameter needs a 'key'")
    key = str(p["key"])
    ptype = str(p.get("type", "double")).lower()
    if ptype == "float":
        ptype = "double"
    if ptype not in PARAM_TYPES:
        raise PluginError(f"{where}: parameter '{key}' has unknown type '{ptype}' (one of {', '.join(PARAM_TYPES)})")
    q: Dict[str, Any] = {
        "key": key,
        "label": str(p.get("label", key.replace("_", " ").capitalize())),
        "type": ptype,
        "help": str(p.get("help", "")),
        "unit": str(p.get("unit", "")),
        "advanced": bool(p.get("advanced", False)),
    }
    default = p.get("default")
    if ptype == "double":
        q["default"] = float(default if default is not None else 0.0)
    elif ptype in ("int", "channel"):
        q["default"] = int(default if default is not None else 0)
    elif ptype == "bool":
        q["default"] = bool(default)
    elif ptype == "choice":
        choices = [str(c) for c in p.get("choices", [])]
        if not choices:
            raise PluginError(f"{where}: choice parameter '{key}' needs 'choices'")
        q["choices"] = choices
        q["default"] = str(default) if default is not None else choices[0]
        if q["default"] not in choices:
            raise PluginError(f"{where}: default of '{key}' is not one of its choices")
    elif ptype in ("path", "string", "axes"):
        q["default"] = str(default if default is not None else "")
    elif ptype == "double_list":
        q["default"] = [float(v) for v in (default or [])]
    elif ptype == "string_list":
        q["default"] = [str(v) for v in (default or [])]
    for k in ("min", "max", "step"):
        if p.get(k) is not None:
            q[k] = float(p[k])
    if p.get("decimals") is not None:
        q["decimals"] = int(p["decimals"])
    if p.get("filter"):
        q["filter"] = str(p["filter"])
    return q


def validate_spec(spec: Any, run, file: Path) -> Dict[str, Any]:
    where = file.name
    if not isinstance(spec, dict):
        raise PluginError(f"{where}: STEP must be a dict")
    kind = str(spec.get("kind", file.stem))
    if not _KIND_RE.match(kind):
        raise PluginError(f"{where}: kind '{kind}' must be an identifier (letters, digits, _ . -)")
    if not callable(run):
        raise PluginError(f"{where}: needs a run(data, params, meta, ctx) function")
    params = spec.get("params", [])
    if not isinstance(params, (list, tuple)):
        raise PluginError(f"{where}: 'params' must be a list")
    help_text = spec.get("help")
    if not help_text:
        help_text = inspect.cleandoc(getattr(run, "__doc__", None) or "")
    out = {
        "kind": kind,
        "name": str(spec.get("name", kind)),
        "group": str(spec.get("group", "Plugins")),
        "params": [_normalize_param(p, where) for p in params],
        "separable_over_t": bool(spec.get("separable_over_t", False)),
        "produces_labels": bool(spec.get("produces_labels", False)),
        "needs_labels": bool(spec.get("needs_labels", False)),
        "help": str(help_text),
        "cache": str(spec.get("cache", "memory")),
        "version": str(spec.get("version", "")),
    }
    return out


def load_file(file: Path) -> Plugin:
    """Import one plugin; failures become a Plugin whose error is set so the
    application can list it with the reason."""
    name = "sirius_plugin_" + re.sub(r"[^A-Za-z0-9_]", "_", file.stem) + f"_{abs(hash(str(file))) & 0xffff:x}"
    try:
        spec = importlib.util.spec_from_file_location(name, str(file))
        if spec is None or spec.loader is None:
            raise PluginError(f"{file.name}: cannot import")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        step = getattr(module, "STEP", None)
        run = getattr(module, "run", None)
        if step is None:
            raise PluginError(f"{file.name}: no STEP dict")
        normalized = validate_spec(step, run, file)
        return Plugin(file, normalized, run)
    except Exception as e:  # noqa: BLE001 - reported, not fatal
        sys.modules.pop(name, None)
        return Plugin(file, {"kind": file.stem, "name": file.stem, "params": []}, None,
                      error=f"{e.__class__.__name__}: {e}\n{traceback.format_exc(limit=3)}".strip())


def load_all(extra_dirs: Optional[List[str]] = None) -> Tuple[List[Plugin], List[str]]:
    plugins: List[Plugin] = []
    seen: Dict[str, Path] = {}
    dirs = plugin_dirs(extra_dirs)
    for d in dirs:
        if not d.is_dir():
            continue
        why = unsafe_reason(d)
        if why:
            log.warning("plugins in %s are not loaded: %s (chmod go-w, or move them)", d, why)
            continue
        for file in sorted(d.glob("*.py")):
            if file.name.startswith("_"):
                continue
            why = unsafe_reason(file)
            if why:
                log.warning("plugin %s is not loaded: %s", file, why)
                plugins.append(Plugin(file, {"kind": file.stem, "name": file.stem, "params": []}, None,
                                      error=f"not loaded: {why}"))
                continue
            plugin = load_file(file)
            if not plugin.error and plugin.kind in seen:
                plugin.error = f"kind '{plugin.kind}' is already provided by {seen[plugin.kind]}"
                plugin.run = None
            elif not plugin.error:
                seen[plugin.kind] = file
            plugins.append(plugin)
    return plugins, [str(d) for d in dirs]


# --- running -----------------------------------------------------------------------


def _expand(a: np.ndarray, rank: int, what: str) -> np.ndarray:
    if a.ndim > rank:
        raise PluginError(f"{what} has {a.ndim} dimensions, at most {rank} expected")
    while a.ndim < rank:
        a = a[np.newaxis, ...]
    return np.ascontiguousarray(a)


def run_plugin(plugin: Plugin, data: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any],
               labels: Optional[np.ndarray], progress=None, cancelled=None):
    """Returns (output float32 (c, t, z, y, x), labels uint32 (t, z, y, x) or None,
    diagnostics dict, meta overrides dict)."""
    if plugin.run is None:
        raise PluginError(plugin.error or f"plugin '{plugin.kind}' did not load")
    ctx = SimpleNamespace(
        progress=lambda f, m="": progress(float(f), str(m)) if progress else None,
        cancelled=lambda: bool(cancelled()) if cancelled else False,
        labels=labels,
        meta=meta,
        log=lambda *a: print(f"[plugin {plugin.kind}]", *a, file=sys.stderr),
    )
    # defaults for parameters the caller left out
    full = {p["key"]: p["default"] for p in plugin.spec.get("params", [])}
    full.update(params or {})
    result = plugin.run(np.ascontiguousarray(data, dtype=np.float32), full, meta, ctx)
    out = None
    out_labels = None
    diagnostics: Dict[str, Any] = {}
    meta_out: Dict[str, Any] = {}
    if isinstance(result, dict):
        out = result.get("output")
        out_labels = result.get("labels")
        diagnostics = dict(result.get("diagnostics") or {})
        meta_out = dict(result.get("meta") or {})
    elif isinstance(result, tuple):
        for item in result:
            if isinstance(item, np.ndarray) and out is None:
                out = item
            elif isinstance(item, np.ndarray) and out_labels is None:
                out_labels = item
            elif isinstance(item, dict):
                diagnostics = dict(item)
    else:
        out = result
    if out is None:
        raise PluginError(f"plugin '{plugin.kind}' returned no output array")
    out = _expand(np.asarray(out, dtype=np.float32), 5, "output")
    if out_labels is not None:
        out_labels = _expand(np.asarray(out_labels, dtype=np.uint32), 4, "labels")
    return out, out_labels, diagnostics, meta_out
