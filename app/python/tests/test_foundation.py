"""The Foundation step's models: self-contained folders, run without the latents package.

A model is a folder (latents scripts/export_model.py writes it): model.py with
load(folder, device) -> Model, model.json ("latents-model/1"), weights and
_lib/ (the model's own code, imported by that name). These tests build a FAKE
model folder with the same API and a trivial numpy network -- a threshold and a
connected-component decode -- so they need neither torch nor latents. What is
checked is the contract the application depends on: how a folder is found,
imported (its _lib kept apart from another model's), cached, told what to do
(Segment, Prompt with the joint objects form), and how a broken folder or an
old .ltb bundle is said.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sirius_worker import foundation, protocol  # noqa: E402

try:
    import scipy.ndimage  # noqa: F401 - the fake model's decode

    HAVE_SCIPY = True
except ImportError:                                        # pragma: no cover - environment dependent
    HAVE_SCIPY = False

try:
    import safetensors.numpy  # noqa: F401

    HAVE_SAFETENSORS = True
except ImportError:                                        # pragma: no cover - environment dependent
    HAVE_SAFETENSORS = False


# model.py of the fake model: the exported API (load, Model.info/tasks/segment/prompt/logits) on a
# network that is a threshold. It imports its code as `_lib`, as an exported model does, and once
# lazily inside a method, which is what goes wrong when two models' _lib packages are confused.
MODEL_PY = '''
"""A fake exported model: the latents-model API on a threshold network (tests only)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _lib.net import KIND, normalise          # noqa: E402

LOADS = []


class Model:
    def __init__(self, folder, device=None):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / "model.json").read_text())
        self.device = device or "cpu"
        self.kind = KIND
        self.scale = 1.0
        w = self.folder / "weights.safetensors"
        if w.exists():
            from safetensors.numpy import load_file
            self.scale = float(load_file(str(w))["head.scale"][0])
        d = self.manifest["decode"]
        self.fg_threshold = float(d["fg_threshold"])
        self.min_voxels = int(d["min_voxels"])
        LOADS.append(self.device)

    def info(self):
        return self.manifest

    def tasks(self):
        return list(self.manifest["tasks"])

    def _fg(self, volume, channels):
        v = np.asarray(volume, np.float32)
        if channels:
            v = v.mean(axis=0)
        n = self.manifest["input"]["normalisation"]
        return normalise(v, n["percentiles"], n["clip"]) * self.scale

    def logits(self, volume, channels=False):
        fg = self._fg(volume, channels)
        return np.stack([fg, fg])

    def segment(self, volume, channels=False, threshold=None, min_voxels=None):
        if threshold is not None:
            self.fg_threshold = float(threshold)
        if min_voxels is not None:
            self.min_voxels = int(min_voxels)
        from _lib.decode import components          # lazy, as the real decode is
        lab = components(self._fg(volume, channels) > self.fg_threshold, self.min_voxels)
        return np.asarray(lab, np.uint32)

    def prompt(self, volume, objects=None, points=None, point_labels=None, boxes=None, scribbles=None,
               channels=False, snap_z=True, origin=(0, 0, 0)):
        if "prompt" not in self.manifest["tasks"]:
            raise ValueError("no prompt decoder")
        from _lib.decode import components
        fg = self._fg(volume, channels) > self.fg_threshold
        lab = components(fg, 0)
        masks, scores = [], []
        for ob in list(objects or []):
            m = np.zeros(fg.shape, bool)
            if ob.get("box") is not None:
                z0, y0, x0, z1, y1, x1 = [int(v) for v in ob["box"]]
                m[z0:z1, y0:y1, x0:x1] = fg[z0:z1, y0:y1, x0:x1]
            pts = np.asarray(ob.get("points", np.zeros((0, 3))), np.float32).reshape(-1, 3)
            labs = np.asarray(ob.get("point_labels", np.ones(len(pts))), np.int64).reshape(-1)
            for p, l in zip(pts.astype(int), labs):
                if l == 1 and lab[tuple(p)]:
                    m |= lab == lab[tuple(p)]
            for sc in ob.get("scribbles", []) or []:
                for p in np.asarray(sc["points"]).astype(int):
                    if int(sc.get("label", 1)) == 1 and lab[tuple(p)]:
                        m |= lab == lab[tuple(p)]
            for p, l in zip(pts.astype(int), labs):
                if l == 0:
                    z, y, x = p
                    m[max(z - 1, 0):z + 2, max(y - 2, 0):y + 3, max(x - 2, 0):x + 3] = False
            masks.append(m)
            scores.append(0.9 - 0.1 * len(masks))
        n = len(masks)
        return (np.stack(masks) if n else np.zeros((0,) + fg.shape, bool), np.asarray(scores, np.float32),
                {"clipped": [], "window": list(self.manifest["input"]["crop"]), "kind": self.kind})


def load(folder=None, device=None):
    return Model(folder or HERE, device)
'''

LIB_NET = '''
import numpy as np

KIND = "{kind}"


def normalise(v, percentiles, clip):
    lo, hi = np.percentile(v, percentiles)
    return np.clip((v - lo) / max(hi - lo, 1e-6), clip[0], clip[1])
'''

LIB_DECODE = '''
import numpy as np
from scipy import ndimage as ndi

from .net import KIND  # noqa: F401 - relative, as the vendored code imports


def components(mask, min_voxels):
    lab, n = ndi.label(mask)
    if min_voxels:
        sizes = np.bincount(lab.ravel())
        small = np.nonzero(sizes < min_voxels)[0]
        lab[np.isin(lab, small[small > 0])] = 0
    return lab
'''


def manifest(name="fake-sam", version="v1", tasks=("segment", "prompt"), channels=1, voxel_zyx=(0.5, 0.1, 0.1),
             fg_threshold=0.5, min_voxels=0, notes="A fake model for tests."):
    return {
        "format": "latents-model/1", "name": name, "version": version, "created": "2026-10-03T00:00:00Z",
        "tasks": list(tasks),
        "encoder": {"dim": 8, "patch": [2, 4, 4]},
        "head": {"kind": "sam" if "prompt" in tasks else "conv", "class": "Fake", "args": {}},
        "input": {"axes": "(c, t, z, y, x), any leading axis optional", "dtype": "float32", "channels": channels,
                  "channel_merge": "", "patch": [2, 4, 4], "crop": [8, 32, 32],
                  "normalisation": {"kind": "robust percentile, per volume", "percentiles": [0.5, 99.8],
                                    "clip": [-0.5, 2.0]},
                  "voxel_um": list(voxel_zyx)},
        "output": {"channels": 2, "0": "foreground logit", "1": "distance"},
        "decode": {"kind": "distance-watershed", "fg_threshold": fg_threshold, "seed_hmax": 0.1, "seed_hrel": 0.0,
                   "seed_sigma": 2.0, "min_voxels": min_voxels, "needs": ["scipy.ndimage"]},
        "prompt": ({"form": "objects", "pad_width": 2, "snap_z": "on"} if "prompt" in tasks else None),
        "provenance": {"latents_commit": "c037177", "latents_dirty": False, "run": "/runs/fake"},
        "notes": notes,
    }


def make_model(root, name="fake-sam", version="v1", kind="a", weights_scale=None, **kw) -> str:
    """<root>/<name>/<version>/ with model.py, model.json, README.md and _lib/."""
    folder = Path(root) / name / version
    (folder / "_lib").mkdir(parents=True, exist_ok=True)
    (folder / "model.py").write_text(MODEL_PY)
    (folder / "model.json").write_text(json.dumps(manifest(name=name, version=version, **kw), indent=2))
    (folder / "README.md").write_text(f"# {name} ({version})\n\nSegments bright blobs; a fake for tests.\n\n## Use\n")
    (folder / "_lib" / "__init__.py").write_text("")
    (folder / "_lib" / "net.py").write_text(LIB_NET.format(kind=kind))
    (folder / "_lib" / "decode.py").write_text(LIB_DECODE)
    if weights_scale is not None:
        from safetensors.numpy import save_file

        save_file({"head.scale": np.asarray([weights_scale], np.float32)}, str(folder / "weights.safetensors"))
    return str(folder)


def blobs(c=1, t=1, z=8, y=32, x=32):
    """Three bright balls on a dark background, (c, t, z, y, x)."""
    v = np.zeros((c, t, z, y, x), np.float32)
    zz, yy, xx = np.ogrid[:z, :y, :x]
    for cz, cy, cx in ((4, 8, 8), (4, 22, 10), (4, 12, 24)):
        v += (((zz - cz) ** 2) / 4.0 + (yy - cy) ** 2 + (xx - cx) ** 2 <= 9).astype(np.float32) * 100.0
    return v + 5.0


class Base(unittest.TestCase):
    def setUp(self):
        foundation.unload()
        self.root = tempfile.mkdtemp(prefix="sirius-models-")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.addCleanup(foundation.unload)
        self.path_before = list(sys.path)

    def tearDown(self):
        # nothing of a model's folder is left on sys.path
        self.assertEqual(sys.path, self.path_before)


class Folder(Base):
    def test_model_info_reads_model_json_and_readme(self):
        folder = make_model(self.root)
        info = foundation.model_info(folder)
        self.assertEqual(info["format"], "latents-model")
        self.assertEqual((info["name"], info["version"]), ("fake-sam", "v1"))
        self.assertEqual(info["tasks"], ["segment", "prompt"])
        self.assertTrue(info["promptable"])
        self.assertEqual(info["description"], "A fake model for tests.")
        self.assertEqual(info["voxel_um"], [0.1, 0.1, 0.5])            # the application's (x, y, z)
        self.assertEqual(info["crop"], [8.0, 32.0, 32.0])
        self.assertEqual(info["input"]["normalisation"]["percentiles"], [0.5, 99.8])
        self.assertEqual(info["channels"], 1)
        self.assertFalse(info["loaded"])
        # model.json itself names the folder too, and model_info imports nothing
        self.assertEqual(foundation.model_info(os.path.join(folder, "model.json"))["path"], folder)
        self.assertFalse(any(k == "_lib" or k.startswith("_lib.") for k in sys.modules))

    def test_description_falls_back_to_the_readme(self):
        folder = make_model(self.root, notes="", tasks=("segment",))
        info = foundation.model_info(folder)
        self.assertEqual(info["description"], "Segments bright blobs; a fake for tests.")
        self.assertFalse(info["promptable"])

    def test_a_broken_folder_says_what_is_wrong(self):
        with self.assertRaisesRegex(FileNotFoundError, "model folder not found"):
            foundation.model_info(os.path.join(self.root, "nope"))
        empty = os.path.join(self.root, "empty")
        os.makedirs(empty)
        with self.assertRaisesRegex(foundation.ModelError, "no model.json"):
            foundation.model_info(empty)
        folder = make_model(self.root)
        Path(folder, "model.json").write_text("{ not json")
        with self.assertRaisesRegex(foundation.ModelError, "not valid JSON"):
            foundation.model_info(folder)
        Path(folder, "model.json").write_text(json.dumps({"format": "something-else/1", "tasks": ["segment"]}))
        with self.assertRaisesRegex(foundation.ModelError, "not latents-model"):
            foundation.model_info(folder)
        Path(folder, "model.json").write_text(json.dumps({**manifest(), "format": "latents-model/2"}))
        with self.assertRaisesRegex(foundation.ModelError, "this worker reads latents-model/1"):
            foundation.model_info(folder)

    def test_an_old_bundle_says_to_re_export(self):
        old = os.path.join(self.root, "cells.ltb")
        Path(old).write_bytes(b"PK")
        for call in (lambda: foundation.model_info(old),
                     lambda: foundation.run(blobs(), {"model": old, "task": "segment"}, "cpu")):
            with self.assertRaises(foundation.ModelError) as e:
                call()
            self.assertIn("old bundle format", str(e.exception))
            self.assertIn("scripts/export_model.py", str(e.exception))

    def test_nothing_imports_latents(self):
        self.assertNotIn("latents", sys.modules)
        src = Path(foundation.__file__).read_text(encoding="utf-8")
        self.assertNotIn("SIRIUS_LATENTS_PATH", src)
        self.assertNotIn("import latents", src)


@unittest.skipUnless(HAVE_SCIPY, "the fake model's decode needs scipy")
class Running(Base):
    def test_segment_returns_dense_labels_per_time_point(self):
        folder = make_model(self.root)
        labels, info, extras = foundation.run(blobs(t=2), {"model": folder, "task": "segment"}, "cpu")
        self.assertEqual(labels.shape, (2, 8, 32, 32))
        self.assertEqual(labels.dtype, np.uint32)
        self.assertEqual(int(labels[0].max()), 3)
        self.assertEqual(sorted(np.unique(labels[1]).tolist()), [0, 1, 2, 3])
        self.assertEqual(info["objects"], 6)
        self.assertEqual(info["threshold"], 0.5)                     # model.json's, the call gave none
        self.assertEqual(info["model"], "fake-sam v1")

    def test_a_threshold_and_min_voxels_apply_to_their_run_only(self):
        folder = make_model(self.root)
        big = foundation.run(blobs(), {"model": folder, "task": "segment", "min_voxels": 10 ** 6}, "cpu")[0]
        self.assertEqual(int(big.max()), 0)
        again = foundation.run(blobs(), {"model": folder, "task": "segment"}, "cpu")
        self.assertEqual(int(again[0].max()), 3)                     # not the last run's min_voxels
        self.assertEqual(again[1]["min_voxels"], 0)

    def test_a_model_is_loaded_once_per_folder_and_device(self):
        folder = make_model(self.root)
        foundation.run(blobs(), {"model": folder, "task": "segment"}, "cpu")
        first = foundation.load_model(folder, "cpu")
        foundation.run(blobs(), {"model": folder, "task": "segment"}, "cpu")
        self.assertIs(foundation.load_model(folder, "cpu"), first)
        self.assertEqual(first.module.LOADS, ["cpu"])
        self.assertTrue(foundation.model_info(folder)["loaded"])
        # another device is another model; one model is resident at a time
        other = foundation.load_model(folder, "cuda:1")
        self.assertIsNot(other, first)
        self.assertEqual(other.model.device, "cuda:1")
        self.assertEqual(len(foundation._MODELS), 1)

    def test_a_re_export_in_place_is_picked_up(self):
        folder = make_model(self.root)
        first = foundation.load_model(folder, "cpu")
        time.sleep(0.02)
        man = json.loads(Path(folder, "model.json").read_text())
        man["decode"]["fg_threshold"] = 0.25
        Path(folder, "model.json").write_text(json.dumps(man))
        os.utime(os.path.join(folder, "model.json"), ns=(time.time_ns() + 10 ** 9,) * 2)
        second = foundation.load_model(folder, "cpu")
        self.assertIsNot(second, first)
        self.assertEqual(second.model.fg_threshold, 0.25)

    def test_two_models_keep_their_own_lib(self):
        a = make_model(self.root, name="model-a", kind="a")
        b = make_model(self.root, name="model-b", kind="b")
        la = foundation.load_model(a, "cpu")
        self.assertEqual(la.model.kind, "a")
        lb = foundation.load_model(b, "cpu")
        self.assertEqual(lb.model.kind, "b")
        self.assertEqual(sys.modules["_lib.net"].KIND, "b")
        # model-a again: its own _lib, imported anew, not model-b's left in sys.modules
        la2 = foundation.load_model(a, "cpu")
        self.assertEqual(la2.model.kind, "a")
        self.assertEqual(la2.call("prompt", blobs()[0, 0], objects=[{"points": [[4, 8, 8]]}])[2]["kind"], "a")

    def test_prompt_returns_one_mask_per_object_labelled_by_position(self):
        folder = make_model(self.root)
        objects = [{"points": [[8, 8, 4]]},                       # the application's (x, y, z)
                   {"box": [20, 8, 2, 30, 17, 7]},                 # around the blob at (x 24, y 12)
                   {"scribbles": [{"points": [[10, 22, 4], [11, 22, 4]], "label": 1}]}]
        labels, info, extras = foundation.run(blobs(), {"model": folder, "task": "prompt", "objects": objects}, "cpu")
        self.assertEqual(labels.shape, (1, 8, 32, 32))
        self.assertEqual(labels[0, 4, 8, 8], 1)
        self.assertEqual(labels[0, 4, 12, 24], 2)
        self.assertEqual(labels[0, 4, 22, 10], 3)
        self.assertEqual(info["mask_scores"], [0.8, 0.7, 0.6])
        self.assertEqual(info["prompt_kinds"]["objects"], 3)
        self.assertEqual(extras["confidence"].shape, (1, 8, 32, 32))
        self.assertAlmostEqual(float(extras["confidence"][0, 4, 8, 8]), 0.8, places=5)
        self.assertEqual(info["window"], [8, 32, 32])

    def test_a_background_click_corrects_its_objects_mask(self):
        folder = make_model(self.root)
        one = {"points": [[8, 8, 4]]}
        before = foundation.run(blobs(), {"model": folder, "task": "prompt", "objects": [one]}, "cpu")[0]
        fixed = {"points": [[8, 8, 4], [8, 10, 4]], "point_labels": [1, 0]}
        after = foundation.run(blobs(), {"model": folder, "task": "prompt", "objects": [fixed]}, "cpu")[0]
        self.assertEqual(int(after.max()), 1)                         # still one object, one mask
        self.assertLess(int((after == 1).sum()), int((before == 1).sum()))

    def test_a_segment_only_model_cannot_be_prompted(self):
        folder = make_model(self.root, name="fake-conv", tasks=("segment",))
        with self.assertRaisesRegex(foundation.ModelError, "cannot be prompted.*offers segment"):
            foundation.run(blobs(), {"model": folder, "task": "prompt", "objects": [{"points": [[8, 8, 4]]}]}, "cpu")
        with self.assertRaisesRegex(foundation.ModelError, "cannot track"):
            foundation.run(blobs(t=2), {"model": folder, "task": "track"}, "cpu")

    def test_the_channel_contract_is_checked(self):
        folder = make_model(self.root)
        with self.assertRaisesRegex(foundation.ModelError, "takes one channel and was sent 2"):
            foundation.run(blobs(c=2), {"model": folder, "task": "segment"}, "cpu")
        two = make_model(self.root, name="two", channels=2)
        with self.assertRaisesRegex(foundation.ModelError, "takes 2 channels"):
            foundation.run(blobs(c=3), {"model": two, "task": "segment"}, "cpu")
        labels, info, _ = foundation.run(blobs(c=2), {"model": two, "task": "segment"}, "cpu")
        self.assertEqual(int(labels.max()), 3)

    def test_a_far_voxel_size_is_warned_about(self):
        folder = make_model(self.root)
        _, near, _ = foundation.run(blobs(), {"model": folder, "task": "segment", "voxel_um": [0.11, 0.1, 0.6]}, "cpu")
        self.assertNotIn("warnings", near)
        _, far, _ = foundation.run(blobs(), {"model": folder, "task": "segment", "voxel_um": [0.4, 0.4, 0.5]}, "cpu")
        self.assertEqual(len(far["warnings"]), 1)
        self.assertIn("trained at", far["warnings"][0])
        self.assertIn("x, y", far["warnings"][0])
        _, tiled, _ = foundation.run(blobs(), {"model": folder, "task": "segment", "tile": [4, 16, 16]}, "cpu")
        self.assertIn("Tile is ignored", tiled["warnings"][0])

    def test_a_model_py_that_fails_names_the_folder(self):
        folder = make_model(self.root)
        Path(folder, "model.py").write_text("raise RuntimeError('weights do not match the code')\n")
        with self.assertRaises(foundation.ModelError) as e:
            foundation.load_model(folder, "cpu")
        self.assertIn(folder, str(e.exception))
        self.assertIn("weights do not match the code", str(e.exception))
        Path(folder, "model.py").write_text("import torch_that_is_not_there\n")
        with self.assertRaisesRegex(foundation.ModelError, "needs torch_that_is_not_there"):
            foundation.load_model(folder, "cpu")
        Path(folder, "model.py").write_text("x = 1\n")
        with self.assertRaisesRegex(foundation.ModelError, r"no load\(folder, device\)"):
            foundation.load_model(folder, "cpu")
        os.remove(os.path.join(folder, "model.py"))
        with self.assertRaisesRegex(foundation.ModelError, "no model.py"):
            foundation.load_model(folder, "cpu")
        self.assertFalse(any(k.startswith("_sirius_model_") for k in sys.modules))

    def test_cancel_raises_an_exception_not_a_keyboard_interrupt(self):
        folder = make_model(self.root)
        self.assertTrue(issubclass(foundation.Cancelled, Exception))
        with self.assertRaises(foundation.Cancelled):
            foundation.run(blobs(), {"model": folder, "task": "segment"}, "cpu", cancelled=lambda: True)

    @unittest.skipUnless(HAVE_SAFETENSORS, "safetensors is not installed in this Python: the weights file "
                                           "cannot be written, so the weights path is not tested here")
    def test_the_weights_file_is_the_models_own(self):
        folder = make_model(self.root, weights_scale=0.0)            # a network that sees nothing
        labels = foundation.run(blobs(), {"model": folder, "task": "segment"}, "cpu")[0]
        self.assertEqual(int(labels.max()), 0)


class Listing(Base):
    def test_models_are_listed_by_name_and_version(self):
        make_model(self.root, name="coat-sam-s2", version="v1")
        make_model(self.root, name="coat-sam-s2", version="v2", tasks=("segment",))
        make_model(self.root, name="alpha", version="v1")
        broken = Path(self.root, "broken", "v1")
        broken.mkdir(parents=True)
        (broken / "model.json").write_text("{")
        Path(self.root, "old.ltb").write_bytes(b"PK")
        Path(self.root, "not-a-model").mkdir()
        got = foundation.list_models([self.root, os.path.join(self.root, "missing")])
        rows = [(m["name"], m["version"], m["tasks"], bool(m["error"])) for m in got["models"]]
        self.assertEqual(rows, [("old", "", [], True),
                                ("alpha", "v1", ["segment", "prompt"], False),
                                ("broken", "v1", [], True),
                                ("coat-sam-s2", "v1", ["segment", "prompt"], False),
                                ("coat-sam-s2", "v2", ["segment"], False)])
        self.assertIn("old bundle format", got["models"][0]["error"])
        self.assertEqual(got["models"][1]["description"], "A fake model for tests.")
        self.assertEqual(got["errors"], [f"not a folder: {os.path.join(self.root, 'missing')}"])
        # a model folder named directly is its own listing
        one = foundation.list_models([os.path.join(self.root, "alpha", "v1")])
        self.assertEqual([m["name"] for m in one["models"]], ["alpha"])

    def test_listing_loads_nothing(self):
        make_model(self.root)
        foundation.list_models([self.root])
        self.assertEqual(foundation._MODELS, {})
        self.assertFalse(any(k.startswith("_sirius_model_") for k in sys.modules))


@unittest.skipUnless(HAVE_SCIPY, "the fake model's decode needs scipy")
class OverTheSocket(Base):
    """What the application sends: model_info, list_bundles and run kind foundation."""

    def call(self, method, params, tensors=None):
        from sirius_worker.server import WorkerServer

        server = WorkerServer("127.0.0.1", 0, "t", "cpu")
        port = server.bind()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        sock = socket.create_connection(("127.0.0.1", port), timeout=20)
        try:
            self.assertEqual(protocol.client_handshake(sock, "t", first_id=100)["type"], "result")
            protocol.write_frame(sock, {"id": 2, "type": "request", "method": method, "params": params}, tensors or {})
            while True:
                header, got = protocol.read_frame(sock)
                if header.get("id") == 2 and header["type"] != "progress":
                    return header, got
        finally:
            sock.close()
            server.stop()
            thread.join(timeout=5)

    def test_model_info_of_a_folder(self):
        folder = make_model(self.root)
        header, _ = self.call("model_info", {"path": folder, "model": folder, "spec": folder})
        self.assertEqual(header["type"], "result", header)
        self.assertEqual(header["result"]["tasks"], ["segment", "prompt"])
        self.assertTrue(header["result"]["promptable"])

    def test_model_info_of_a_folder_that_is_not_there(self):
        missing = os.path.join(self.root, "coat-sam-s2", "v9")
        header, _ = self.call("model_info", {"path": missing, "model": missing, "spec": missing})
        self.assertEqual(header["type"], "error", header)
        self.assertIn("model folder not found", header["message"])

    def test_list_bundles_lists_the_model_folders(self):
        make_model(self.root)
        header, _ = self.call("list_bundles", {"dirs": [self.root]})
        self.assertEqual(header["type"], "result", header)
        self.assertEqual([m["name"] for m in header["result"]["models"]], ["fake-sam"])

    def test_a_prompt_run(self):
        folder = make_model(self.root)
        header, got = self.call("run", {"kind": "foundation",
                                        "params": {"model": folder, "task": "prompt", "voxel_um": [0.1, 0.1, 0.5],
                                                   "objects": [{"points": [[8, 8, 4]], "point_labels": [1]}]}},
                                {"input": blobs()})
        self.assertEqual(header["type"], "result", header)
        self.assertEqual(got["labels"].shape, (1, 8, 32, 32))
        self.assertEqual(int(got["labels"][0, 4, 8, 8]), 1)
        self.assertEqual(header["result"]["mask_scores"], [0.8])


class WorkbenchStep(Base):
    """bindings' run_step("foundation", ...) goes through the same folder loader."""

    @staticmethod
    def workbench():
        """This checkout's bindings/python/sirius/workbench.py, whatever `sirius` the interpreter has
        installed (as bindings/tests/test_workbench_schema.py)."""
        import importlib.util

        here = Path(__file__).resolve().parents[3] / "bindings" / "python" / "sirius" / "workbench.py"
        try:
            import sirius.workbench as wb  # type: ignore

            if Path(wb.__file__).resolve() == here:
                return wb
        except Exception:  # noqa: BLE001
            pass
        name = "sirius_workbench_under_test"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, here)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        return sys.modules[name]

    @unittest.skipUnless(HAVE_SCIPY, "the fake model's decode needs scipy")
    def test_run_step_segments_and_prompts_with_a_folder(self):
        wb = self.workbench()
        folder = make_model(self.root)
        seen = []
        res = wb.run_step("foundation", {"model": folder, "task": "Segment objects"}, blobs(),
                          {"voxel_um": [0.1, 0.1, 0.5]}, progress=lambda f, m: seen.append(f), device="cpu")
        self.assertEqual(int(np.asarray(res.labels).max()), 3)
        self.assertTrue(seen)
        prompts = [{"kind": "point", "x": 8, "y": 8, "z": 4, "object": 7}]
        res = wb.run_step("foundation", {"model": folder, "task": "Prompt objects", "prompts": prompts}, blobs(),
                          {"voxel_um": [0.1, 0.1, 0.5]}, device="cpu")
        self.assertEqual(int(np.asarray(res.labels)[0, 4, 8, 8]), 7)   # labelled with the object's id
        with self.assertRaises(wb.Cancelled):
            wb.run_step("foundation", {"model": folder, "task": "Segment objects"}, blobs(), None,
                        cancelled=lambda: True)


if __name__ == "__main__":
    unittest.main()
