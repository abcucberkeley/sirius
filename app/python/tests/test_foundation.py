"""The foundation-model kind: the contract the application depends on.

Not a quality test. The weights here are untrained, so nothing is checked about
where the objects land. What is checked is the shape of the exchange, because
that is what breaks silently: the application sends (c, t, z, y, x) and reads
back labels of exactly (t, z, y, x) uint32 plus a confidence map of the same
shape, a tracking run must give one label id per object for its whole life
rather than a fresh id each frame, and a missing package must say what to
install rather than raise ModuleNotFoundError from four frames down.

`Foundation` runs against the real package (SIRIUS_LATENTS_PATH). The geometry
the worker derives from a heatmap -- voxel order, thresholds, tiles, sizes,
tracks -- is checked in `WithScriptedHeatmap` on a stand-in for the package
whose heatmap the test decides, which needs no latents, torch or weights.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
import tempfile
import threading
import time
import types
import unittest
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sirius_worker import foundation, protocol  # noqa: E402

_path = list(sys.path)
try:
    foundation._import_latents()
    HAVE, WHY = True, ""
except Exception as exc:                                   # noqa: BLE001
    HAVE, WHY = False, str(exc)
finally:
    # _import_latents puts SIRIUS_LATENTS_PATH first on sys.path, and a latents
    # checkout has a top-level `tests` package of its own: left there while the
    # suite is being discovered, it hides this directory's from test_models.
    # The package stays importable from sys.modules; runs put the path back.
    sys.path[:] = _path

try:
    from scipy import ndimage as ndi
    from scipy.optimize import linear_sum_assignment
    from skimage.segmentation import watershed

    HAVE_SCIPY = True
except ImportError:                                        # pragma: no cover - environment dependent
    HAVE_SCIPY = False


def make_bundle(path: str, five_d: bool = False, head: str = "detection") -> None:
    """An untrained bundle of the smallest shape the model supports."""
    from latents.deploy import Bundle, Manifest
    from latents.downstream.instance import ThreeClassHead
    from latents.downstream.track_train import DetectionHead
    from latents.model import ChannelTimeMAE, build_model

    patch = (4, 16, 16)
    if five_d:
        cfg = {"seq": True, "patch": patch, "dim": 192, "depth": 2, "heads": 3,
               "dec_dim": 128, "dec_depth": 1}
        enc = ChannelTimeMAE(patch=patch, dim=192, depth=2, heads=3, dec_dim=128, dec_depth=1)
    else:
        cfg = {"patch": patch, "in_channels": 1, "dim": 192, "depth": 2, "heads": 3,
               "dec_dim": 128, "dec_depth": 1}
        enc = build_model(cfg)
    if head == "threeclass":
        head_args = dict(dim=192, patch=patch)
        module = ThreeClassHead(**head_args)
    else:
        head_args = dict(dim=192, patch=patch, flow=False)
        module = DetectionHead(**head_args)
    man = Manifest(task="detect", name="test", encoder=cfg, head=head, head_args=head_args,
                   patch=patch, crop=(8, 64, 64), voxel_size=(0.5, 0.15, 0.15),
                   peak_threshold=0.5, min_separation_um=1.0)
    Bundle.save(path, man, enc, module)


def blobs(c=1, t=1, z=8, y=64, x=64):
    """A few bright balls, so the model has something with structure to look at."""
    v = np.zeros((c, t, z, y, x), np.float32)
    zz, yy, xx = np.ogrid[:z, :y, :x]
    for cz, cy, cx in ((4, 16, 16), (4, 44, 20), (4, 24, 46)):
        v += np.exp(-((zz - cz) ** 2 * 4.0 + (yy - cy) ** 2 + (xx - cx) ** 2) / 18.0)
    return v + 0.01 * np.random.default_rng(0).random(v.shape).astype(np.float32)


def load_workbench_under_test():
    """This checkout's bindings/python/sirius/workbench.py, whatever `sirius`
    the interpreter has installed (as bindings/tests/test_workbench_schema.py)."""
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


@unittest.skipUnless(HAVE, f"latents not importable: {WHY}")
class Foundation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.path = os.path.join(cls.dir, "m.ltb")
        make_bundle(cls.path)
        cls.path5 = os.path.join(cls.dir, "m5.ltb")
        make_bundle(cls.path5, five_d=True)

    def test_model_info_reads_the_manifest(self):
        foundation._BUNDLES.clear()
        info = foundation.model_info(self.path)
        self.assertEqual(foundation._BUNDLES, {})              # read, not loaded: nothing cached or evicted
        self.assertEqual(info["format"], "latents-bundle")
        # (x, y, z) like every voxel size in the application; the manifest's
        # own (0.5, 0.15, 0.15) is (z, y, x)
        self.assertEqual(info["voxel_um"], [0.15, 0.15, 0.5])
        self.assertEqual(info["crop"], [8, 64, 64])
        self.assertEqual(info["peak_threshold"], 0.5)
        self.assertIn("track", info["tasks"])

    def test_segment_returns_labels_of_the_input_shape(self):
        v = blobs(t=2)
        labels, info, extras = foundation.run(v, {"model": self.path, "task": "segment"}, "cpu")
        self.assertEqual(labels.shape, v.shape[1:])
        self.assertEqual(labels.dtype, np.uint32)
        self.assertEqual(extras["confidence"].shape, v.shape[1:])
        self.assertEqual(info["frames"], 2)

    def test_detect_marks_one_voxel_per_object(self):
        v = blobs()
        labels, info, _ = foundation.run(v, {"model": self.path, "task": "detect"}, "cpu")
        self.assertEqual(labels.shape, v.shape[1:])
        self.assertEqual(int((labels > 0).sum()), info["objects"])

    def test_track_keeps_one_id_per_object(self):
        v = blobs(t=3)
        labels, info, extras = foundation.run(v, {"model": self.path, "task": "track"}, "cpu")
        self.assertEqual(labels.shape, v.shape[1:])
        self.assertIn("tracks", info)
        self.assertIn("divisions", info)
        # Ids must come from one numbering across the whole clip, not restart
        # per frame: that is what makes the label volume a set of tracks.
        biggest = int(labels.max())
        self.assertLessEqual(biggest, info["tracks"] if info["tracks"] else biggest)

    def test_colour_needs_the_five_d_model(self):
        v = blobs(c=2)
        with self.assertRaises(ValueError) as cm:
            foundation.run(v, {"model": self.path, "task": "segment"}, "cpu")
        self.assertIn("channel", str(cm.exception))
        labels, _, _ = foundation.run(v, {"model": self.path5, "task": "segment"}, "cpu")
        self.assertEqual(labels.shape, v.shape[1:])

    def test_caller_thresholds_override_the_bundle(self):
        v = blobs()
        _, info, _ = foundation.run(v, {"model": self.path, "task": "detect"}, "cpu")
        self.assertEqual(info["threshold"], 0.5)             # 0 means "the bundle's"
        _, info, _ = foundation.run(v, {"model": self.path, "task": "detect", "threshold": 0.8}, "cpu")
        self.assertEqual(info["threshold"], 0.8)

    def test_a_tile_does_not_stick_to_the_cached_bundle(self):
        v = blobs()
        _, info, _ = foundation.run(v, {"model": self.path, "task": "detect", "tile": [8, 32, 32]}, "cpu")
        self.assertEqual(info["tile"], [8, 32, 32])
        _, info, _ = foundation.run(v, {"model": self.path, "task": "detect"}, "cpu")
        self.assertEqual(info["tile"], [8, 64, 64])
        self.assertEqual(foundation.model_info(self.path)["crop"], [8, 64, 64])

    def test_a_three_class_bundle_segments_and_refuses_the_rest(self):
        # Bundle.heatmap writes this head's three channels into a one-channel
        # buffer and raises, so every task used to fail on such a bundle.
        path = os.path.join(self.dir, "tc.ltb")
        make_bundle(path, head="threeclass")
        v = blobs(t=2)
        labels, info, extras = foundation.run(v, {"model": path, "task": "segment"}, "cpu")
        self.assertEqual(labels.shape, v.shape[1:])
        conf = extras["confidence"]
        self.assertEqual(conf.shape, v.shape[1:])
        self.assertTrue(np.all((conf >= 0) & (conf <= 1)))
        self.assertEqual(info["objects"], sum(int(f.max()) for f in labels))
        for task in ("detect", "track"):
            with self.assertRaises(ValueError) as cm:
                foundation.run(v, {"model": path, "task": task}, "cpu")
            self.assertIn("Segment", str(cm.exception))
        self.assertEqual(foundation.model_info(path)["tasks"], ["segment"])

    def test_divisions_are_counted_from_latents_own_parent_map(self):
        from latents.downstream.track import track_points

        # One mother dividing at frame 1 and her continuing daughter dividing
        # again at frame 3, nothing else in the field.
        frames = [np.array([[0, 50, 50]], np.float32),
                  np.array([[0, 50, 44], [0, 50, 56]], np.float32),
                  np.array([[0, 50, 44], [0, 50, 56]], np.float32),
                  np.array([[0, 44, 44], [0, 56, 44], [0, 50, 56]], np.float32)]
        _, parents = track_points(frames, max_dist=10.0, voxel_size=(1.0, 1.0, 1.0), division_dist=15.0)
        self.assertEqual(len(foundation.lineage(parents)), 2)
        self.assertEqual(set(foundation.lineage(parents).values()), {1})

    def test_an_unknown_task_is_refused(self):
        with self.assertRaises(ValueError):
            foundation.run(blobs(), {"model": self.path, "task": "cluster"}, "cpu")

    def test_a_missing_bundle_says_so(self):
        with self.assertRaises(FileNotFoundError):
            foundation.run(blobs(), {"model": os.path.join(self.dir, "nope.ltb")}, "cpu")


# --- a scripted stand-in for the package ---------------------------------------


def blob(shape, centre, sigma, peak):
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    return peak * np.exp(-sum(((g - c) / s) ** 2 for g, c, s in zip(grids, centre, sigma)) / 2)


def clip(t, z, y, x, c=1):
    """An input whose first voxel of frame t is t, so the stand-in's per-frame
    heatmap() knows which scripted frame it was handed."""
    v = np.zeros((c, t, z, y, x), np.float32)
    v[:, :, 0, 0, 0] = np.arange(t, dtype=np.float32)
    return v


class Script:
    """What the stand-in model predicts and what it was asked, reset per test."""
    heatmaps = None          # (t, z, y, x)
    logits = None            # (t, 3, z, y, x), three-class head only
    parents = None           # a scripted track_points parent map
    head = "detection"
    delay = 0.0
    loads: list = []
    calls: list = []


@dataclass
class StubManifest:
    task: str = "detect"
    name: str = "stub"
    encoder: dict = field(default_factory=dict)
    head: str = "detection"
    head_args: dict = field(default_factory=dict)
    patch: tuple = (4, 16, 16)
    crop: tuple = (32, 192, 192)
    voxel_size: tuple = (1.0, 1.0, 1.0)       # (z, y, x), as in latents
    peak_threshold: float = 0.5
    min_separation_um: float = 1.0
    link_max_dist_um: float = 4.0
    division_dist_um: float = 4.0
    channels: int = 1
    notes: str = ""


class StubBundle:
    def __init__(self, device):
        self.m = StubManifest(head=Script.head)
        self.device = device

    @staticmethod
    def load(path, device=None):
        Script.loads.append(device)
        return StubBundle(device or "cpu")

    def heatmap(self, volume, *, time=False, channels=False, batch=4):
        Script.calls.append(("heatmap", tuple(self.m.crop), bool(time)))
        if Script.delay:
            import time as clock

            clock.sleep(Script.delay)
        if time:
            return Script.heatmaps
        return Script.heatmaps[int(round(float(np.asarray(volume).reshape(-1)[0])))]

    # Bundle.detect / segment / track as latents' deploy.py has them: they read
    # the threshold, the separation and the voxel size from the manifest. The
    # worker must not call them (the caller's values would not reach the
    # model), but a regression that did should fail on its result here rather
    # than on a missing method.
    def detect(self, volume, *, threshold=None, voxel_size=None, **kw):
        return stub_peaks(self.heatmap(volume, **kw), self.m.peak_threshold if threshold is None else threshold,
                          voxel_size=tuple(voxel_size or self.m.voxel_size), min_sep_um=self.m.min_separation_um)

    def segment(self, volume, *, min_size=20, **kw):
        return stub_watershed(self.heatmap(volume, **kw), self.detect(volume, **kw), self.m.peak_threshold, min_size)

    def track(self, clip, *, channels=False, voxel_size=None, **kw):
        hm = self.heatmap(clip, time=True, channels=channels)
        vs = tuple(voxel_size or self.m.voxel_size)
        pts = [stub_peaks(h, self.m.peak_threshold, voxel_size=vs, min_sep_um=self.m.min_separation_um) for h in hm]
        ids, parents = stub_track_points(pts, self.m.link_max_dist_um, vs, self.m.division_dist_um)
        return {"points": pts, "ids": ids, "parents": parents}

    def _check_channels(self, c):
        pass

    def _class_logits(self, volume, *, time=False, channels=False, batch=4):
        Script.calls.append(("logits", tuple(self.m.crop), False))
        return Script.logits[int(round(float(np.asarray(volume).reshape(-1)[0])))]


def stub_peaks(hm, threshold=0.3, min_distance=3, voxel_size=None, min_sep_um=0.0):
    # latents.downstream.track.peaks_from_heatmap: a per-axis gate in microns
    vs = np.asarray(voxel_size, np.float32)[-hm.ndim:]
    rad = np.maximum(np.round(min_sep_um / vs).astype(int), 1)
    peak = hm >= ndi.maximum_filter(hm, size=tuple(2 * int(r) + 1 for r in rad), mode="nearest")
    return np.argwhere(peak & (hm >= threshold)).astype(np.float32)


def stub_track_points(frames, max_dist=15.0, voxel_size=(1.0, 1.0, 1.0), division_dist=None):
    # latents' one-to-one linker in (z, y, x) microns, without the division pass
    vs = np.asarray(voxel_size, np.float32)
    ids, parents, nxt = [], {}, 1
    for t, pts in enumerate(frames):
        cur = np.zeros(len(pts), np.int64)
        if t and len(pts) and len(frames[t - 1]):
            d = np.linalg.norm((frames[t - 1][:, None] - pts[None]) * vs, axis=-1)
            cost = np.where(d <= max_dist, d, 1e6)
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e6:
                    cur[j] = ids[t - 1][i]
        for j in range(len(pts)):
            if not cur[j]:
                cur[j] = nxt
                if t:
                    parents[nxt] = 0
                nxt += 1
        ids.append(cur)
    return ids, (Script.parents if Script.parents is not None else parents)


def stub_watershed(hm, peaks, threshold, min_size=20):
    # latents.deploy.watershed_from_heatmap: region i grows from peaks[i - 1],
    # regions under min_size are zeroed and not renumbered
    Script.calls.append(("watershed", float(threshold), int(min_size)))
    markers = np.zeros(hm.shape, np.int32)
    for i, p in enumerate(np.round(peaks).astype(int), start=1):
        markers[tuple(np.clip(p, 0, np.array(hm.shape) - 1))] = i
    lab = watershed(-hm, markers, mask=hm >= threshold).astype(np.int32)
    if min_size > 0:
        small = np.flatnonzero(np.bincount(lab.ravel()) < min_size)
        lab[np.isin(lab, small)] = 0
    return lab


def stub_instances(logits, min_size=20):
    lab, _ = ndi.label(np.asarray(logits).argmax(0) > 0)
    small = np.flatnonzero(np.bincount(lab.ravel()) < min_size)
    lab[np.isin(lab, small[small > 0])] = 0
    return lab.astype(np.int32)


def stub_modules():
    deploy = types.ModuleType("latents.deploy")
    deploy.Bundle, deploy.Manifest, deploy.BUNDLE_VERSION = StubBundle, StubManifest, 1
    deploy.watershed_from_heatmap = stub_watershed
    track = types.ModuleType("latents.downstream.track")
    track.peaks_from_heatmap, track.track_points = stub_peaks, stub_track_points
    instance = types.ModuleType("latents.downstream.instance")
    instance.instances_from_three_class = stub_instances
    downstream = types.ModuleType("latents.downstream")
    downstream.track, downstream.instance = track, instance
    latents = types.ModuleType("latents")
    latents.deploy, latents.downstream = deploy, downstream
    return {"latents": latents, "latents.deploy": deploy, "latents.downstream": downstream,
            "latents.downstream.track": track, "latents.downstream.instance": instance}


@unittest.skipUnless(HAVE_SCIPY, "scipy and scikit-image are needed for the scripted model")
class WithScriptedHeatmap(unittest.TestCase):
    Z, Y, X = 16, 48, 48

    def setUp(self):
        # Only these names are swapped and put back: mock.patch.dict would also
        # drop every module first imported during the test, and a nanobind
        # extension (sirius._sirius_ext) aborts the process when imported twice.
        stubs = stub_modules()
        self.saved = {name: sys.modules.get(name) for name in stubs}
        sys.modules.update(stubs)
        foundation._BUNDLES.clear()
        Script.heatmaps = Script.logits = Script.parents = None
        Script.head, Script.delay = "detection", 0.0
        Script.loads, Script.calls = [], []
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "stub.ltb")
        Path(self.path).write_text("stub")

    def tearDown(self):
        foundation._BUNDLES.clear()
        for name, module in self.saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def run_model(self, task, t=1, **params):
        return foundation.run(clip(t, self.Z, self.Y, self.X), {"model": self.path, "task": task, **params}, "cpu")

    def ids(self, labels):
        return [sorted(set(np.unique(f).tolist()) - {0}) for f in labels]

    def test_the_voxel_size_reaches_latents_as_z_y_x(self):
        # two nuclei 4 planes = 3 um apart in depth, on 0.15 x 0.15 x 0.75 um
        # data: distinct at a 1 um separation, one object if z is read as 0.15
        Script.heatmaps = np.maximum(blob((self.Z, self.Y, self.X), (5, 24, 24), (1, 3, 3), 0.9),
                                     blob((self.Z, self.Y, self.X), (9, 24, 24), (1, 3, 3), 0.8))[None]
        _, info, _ = self.run_model("detect", voxel_um=[0.15, 0.15, 0.75])
        self.assertEqual(info["objects"], 2)
        self.assertEqual(info["voxel_um"], [0.15, 0.15, 0.75])       # reported back in the application's order
        # an axis not given, or <= 0, is the bundle's calibration
        self.assertEqual(foundation.voxel_zyx([0.2, 0.3, 0], (0.75, 0.15, 0.15)), (0.75, 0.3, 0.2))
        self.assertEqual(foundation.voxel_zyx(None, (0.75, 0.15, 0.15)), (0.75, 0.15, 0.15))

    def test_a_track_is_linked_in_microns_along_the_right_axes(self):
        # one nucleus moving 10 px = 1.5 um a frame in x; the link gate is 4 um
        Script.heatmaps = np.stack([blob((self.Z, self.Y, self.X), (8, 24, 10 + 10 * t), (1.5, 2, 2), 0.9)
                                    for t in range(3)])
        labels, info, _ = self.run_model("track", t=3, voxel_um=[0.15, 0.15, 0.75])
        self.assertEqual(self.ids(labels), [[1], [1], [1]])
        self.assertEqual(info["tracks"], 1)

    def test_threshold_and_separation_reach_every_task(self):
        # a bright and a dim object; the bundle's threshold (0.5) sees one
        two = np.maximum(blob((self.Z, self.Y, self.X), (8, 12, 12), (1.5, 3, 3), 0.9),
                         blob((self.Z, self.Y, self.X), (8, 36, 36), (1.5, 3, 3), 0.4))
        Script.heatmaps = two[None]
        vox = [0.15, 0.15, 0.75]
        for task in ("detect", "segment"):
            labels, info, _ = self.run_model(task, threshold=0.3, voxel_um=vox)
            self.assertEqual(len(self.ids(labels)[0]), 2, task)
            self.assertEqual(info["objects"], 2, task)
            labels, _, _ = self.run_model(task, threshold=0.3, min_separation=20.0, voxel_um=vox)
            self.assertEqual(len(self.ids(labels)[0]), 1, task)
        Script.heatmaps = np.stack([two, two])
        _, info, _ = self.run_model("track", t=2, threshold=0.3, voxel_um=vox)
        self.assertEqual(info["tracks"], 2)
        _, info, _ = self.run_model("track", t=2, threshold=0.3, min_separation=20.0, voxel_um=vox)
        self.assertEqual(info["tracks"], 1)
        self.assertTrue(all(c[1] == 0.3 for c in Script.calls if c[0] == "watershed"))

    def test_the_model_runs_once_per_frame_or_once_per_clip(self):
        Script.heatmaps = np.stack([blob((self.Z, self.Y, self.X), (8, 24, 24), (1.5, 3, 3), 0.9)] * 3)
        for task, expected in (("segment", 3), ("detect", 3), ("track", 1)):
            Script.calls = []
            self.run_model(task, t=3)
            self.assertEqual(sum(1 for c in Script.calls if c[0] == "heatmap"), expected, task)

    def test_a_tile_applies_to_its_own_run_only(self):
        Script.heatmaps = blob((self.Z, self.Y, self.X), (8, 24, 24), (1.5, 3, 3), 0.9)[None]
        _, info, _ = self.run_model("detect", tile=[4, 16, 16])
        self.run_model("detect")                                     # the app sends no tile for all zeros
        self.run_model("detect", tile=[0, 16, 0])                    # zero is the bundle's, per axis
        self.assertEqual([c[1] for c in Script.calls if c[0] == "heatmap"],
                         [(4, 16, 16), (32, 192, 192), (32, 16, 192)])
        self.assertEqual(info["tile"], [4, 16, 16])
        self.assertEqual(foundation.model_info(self.path)["crop"], [32, 192, 192])

    def test_model_info_does_not_evict_the_bundle_a_run_loaded(self):
        Script.heatmaps = blob((self.Z, self.Y, self.X), (8, 24, 24), (1.5, 3, 3), 0.9)[None]
        foundation.run(clip(1, self.Z, self.Y, self.X), {"model": self.path, "task": "detect"}, "cuda:0")
        info = foundation.model_info(self.path)
        self.assertEqual(info["voxel_um"], [1.0, 1.0, 1.0])
        foundation.run(clip(1, self.Z, self.Y, self.X), {"model": self.path, "task": "detect"}, "cuda:0")
        self.assertEqual(Script.loads, ["cuda:0"])

    def test_detect_ignores_min_voxels(self):
        Script.heatmaps = np.maximum(blob((self.Z, self.Y, self.X), (8, 12, 12), (1.5, 3, 3), 0.9),
                                     blob((self.Z, self.Y, self.X), (8, 36, 36), (1.5, 3, 3), 0.8))[None]
        labels, info, _ = self.run_model("detect", min_voxels=50)
        self.assertEqual(int((labels > 0).sum()), 2)
        self.assertEqual(info["objects"], 2)

    def test_segment_drops_small_objects_and_numbers_the_rest_densely(self):
        shape = (self.Z, self.Y, self.X)
        Script.heatmaps = np.maximum.reduce([blob(shape, (8, 8, 24), (2, 4, 4), 0.9),
                                             blob(shape, (8, 24, 24), (0.6, 0.8, 0.8), 0.9),
                                             blob(shape, (8, 40, 24), (2, 4, 4), 0.9)])[None]
        labels, info, _ = self.run_model("segment", min_voxels=30)
        self.assertEqual(self.ids(labels), [[1, 2]])
        self.assertEqual(labels[0, 8, 24, 24], 0)
        self.assertEqual(info["objects"], 2)

    def test_a_tracking_run_keeps_small_objects(self):
        # the small object is under Min. voxels in frame 1 only; dropping it
        # there would leave a hole in its track
        shape = (self.Z, self.Y, self.X)
        big = blob(shape, (8, 36, 36), (2.0, 5, 5), 0.9)
        Script.heatmaps = np.stack([np.maximum(blob(shape, (8, 12, 12), (1.0, 1.5, 1.5), 0.9), big),
                                    np.maximum(blob(shape, (8, 12, 13), (0.6, 0.8, 0.8), 0.9), big)])
        labels, _, _ = self.run_model("track", t=2, min_voxels=10)
        self.assertEqual(self.ids(labels), [[1, 2], [1, 2]])
        self.assertTrue(all(c[2] == 0 for c in Script.calls if c[0] == "watershed"))

    def test_divisions_count_each_daughter_track(self):
        # latents' shape for a mother (1) dividing twice: daughters 2 and 3,
        # the continuing daughter recorded as her own parent, 4 a fresh start
        self.assertEqual(foundation.lineage({1: 1, 2: 1, 3: 1, 4: 0}), {2: 1, 3: 1})
        Script.heatmaps = np.stack([blob((self.Z, self.Y, self.X), (8, 24, 24), (1.5, 3, 3), 0.9)] * 2)
        Script.parents = {1: 1, 2: 1, 3: 1, 4: 0}
        _, info, extras = self.run_model("track", t=2)
        self.assertEqual(info["divisions"], 2)
        self.assertEqual(extras["lineage"], {2: 1, 3: 1})

    def test_a_three_class_head_segments_from_its_class_logits(self):
        Script.head = "threeclass"
        logits = np.zeros((1, 3, self.Z, self.Y, self.X), np.float32)
        logits[0, 0] = 4.0
        logits[0, 1, 4:12, 10:20, 10:20] = 8.0
        Script.logits = logits
        labels, info, extras = self.run_model("segment")
        self.assertEqual(self.ids(labels), [[1]])
        self.assertGreater(float(extras["confidence"][0, 8, 15, 15]), 0.95)
        self.assertLess(float(extras["confidence"][0, 0, 0, 0]), 0.05)
        with self.assertRaises(ValueError):
            self.run_model("detect")
        self.assertEqual(foundation.model_info(self.path)["tasks"], ["segment"])

    def test_cancel_raises_an_exception_not_a_keyboard_interrupt(self):
        Script.heatmaps = blob((self.Z, self.Y, self.X), (8, 24, 24), (1.5, 3, 3), 0.9)[None]
        self.assertTrue(issubclass(foundation.Cancelled, Exception))
        with self.assertRaises(foundation.Cancelled):
            foundation.run(clip(1, self.Z, self.Y, self.X), {"model": self.path}, "cpu", cancelled=lambda: True)

    def test_a_cancelled_run_is_answered_over_the_socket(self):
        # KeyboardInterrupt escaped the server's handler: no reply at all, and
        # the application waited out its grace period, then dropped the link
        from sirius_worker.server import WorkerServer

        Script.heatmaps = np.zeros((6, 2, 8, 8), np.float32)
        Script.delay = 0.3
        server = WorkerServer("127.0.0.1", 0, "t", "cpu")
        port = server.bind()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            protocol.write_frame(sock, {"id": 1, "type": "request", "method": "hello",
                                        "params": {"token": "t", "protocol_version": protocol.PROTOCOL_VERSION}})
            self.assertEqual(protocol.read_frame(sock)[0]["type"], "result")
            vol = clip(6, 2, 8, 8)
            protocol.write_frame(sock, {"id": 2, "type": "request", "method": "run",
                                        "params": {"kind": "foundation", "params": {"model": self.path, "task": "segment"}}},
                                 {"input": vol})
            header, _ = protocol.read_frame(sock)
            self.assertEqual(header["type"], "progress")
            protocol.write_frame(sock, {"id": 3, "type": "request", "method": "cancel", "params": {"id": 2}})
            t0, answer = time.time(), None
            while answer is None:
                header, _ = protocol.read_frame(sock)
                if header.get("id") == 2 and header["type"] != "progress":
                    answer = header
            self.assertEqual(answer["type"], "error")
            self.assertEqual(answer["message"], "cancelled")
            self.assertLess(time.time() - t0, 5.0)
        finally:
            sock.close()
            server.stop()
            thread.join(timeout=5)

    def test_run_step_hands_the_step_its_device_progress_and_cancel(self):
        wb = load_workbench_under_test()
        Script.heatmaps = blob((self.Z, self.Y, self.X), (8, 24, 24), (1.5, 3, 3), 0.9)[None]
        seen = []
        res = wb.run_step("foundation", {"model": self.path, "task": "Detect centroids"},
                          clip(1, self.Z, self.Y, self.X), {"voxel_um": [0.15, 0.15, 0.75]},
                          progress=lambda f, m: seen.append(f), device="cuda:3")
        self.assertEqual(Script.loads, ["cuda:3"])
        self.assertTrue(seen)
        self.assertEqual(res.info["voxel_um"], [0.15, 0.15, 0.75])
        with self.assertRaises(wb.Cancelled):
            wb.run_step("foundation", {"model": self.path, "task": "Detect centroids"},
                        clip(1, self.Z, self.Y, self.X), None, cancelled=lambda: True)


if __name__ == "__main__":
    unittest.main()
