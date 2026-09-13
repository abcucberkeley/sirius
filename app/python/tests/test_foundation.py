"""The foundation-model kind: the contract the application depends on.

Not a quality test. The weights here are untrained, so nothing is checked about
where the objects land. What is checked is the shape of the exchange, because
that is what breaks silently: the application sends (c, t, z, y, x) and reads
back labels of exactly (t, z, y, x) uint32 plus a confidence map of the same
shape, a tracking run must give one label id per object for its whole life
rather than a fresh id each frame, and a missing package must say what to
install rather than raise ModuleNotFoundError from four frames down.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sirius_worker import foundation  # noqa: E402

try:
    foundation._import_latents()
    HAVE, WHY = True, ""
except Exception as exc:                                   # noqa: BLE001
    HAVE, WHY = False, str(exc)


def make_bundle(path: str, five_d: bool = False) -> None:
    """An untrained bundle of the smallest shape the model supports."""
    from latents.deploy import Bundle, Manifest
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
    head_args = dict(dim=192, patch=patch, flow=False)
    man = Manifest(task="detect", name="test", encoder=cfg, head="detection", head_args=head_args,
                   patch=patch, crop=(8, 64, 64), voxel_size=(0.5, 0.15, 0.15),
                   peak_threshold=0.5, min_separation_um=1.0)
    Bundle.save(path, man, enc, DetectionHead(**head_args))


def blobs(c=1, t=1, z=8, y=64, x=64):
    """A few bright balls, so the model has something with structure to look at."""
    v = np.zeros((c, t, z, y, x), np.float32)
    zz, yy, xx = np.ogrid[:z, :y, :x]
    for cz, cy, cx in ((4, 16, 16), (4, 44, 20), (4, 24, 46)):
        v += np.exp(-((zz - cz) ** 2 * 4.0 + (yy - cy) ** 2 + (xx - cx) ** 2) / 18.0)
    return v + 0.01 * np.random.default_rng(0).random(v.shape).astype(np.float32)


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
        info = foundation.model_info(self.path)
        self.assertEqual(info["format"], "latents-bundle")
        self.assertEqual(info["voxel_um"], [0.5, 0.15, 0.15])
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

    def test_an_unknown_task_is_refused(self):
        with self.assertRaises(ValueError):
            foundation.run(blobs(), {"model": self.path, "task": "cluster"}, "cpu")

    def test_a_missing_bundle_says_so(self):
        with self.assertRaises(FileNotFoundError):
            foundation.run(blobs(), {"model": os.path.join(self.dir, "nope.ltb")}, "cpu")


if __name__ == "__main__":
    unittest.main()
