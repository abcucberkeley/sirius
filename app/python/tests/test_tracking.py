"""The worker's tracking backends: btrack when it is installed and its
compiled core loads, and a clear refusal when it is not."""

from __future__ import annotations

import json
import os
import sys
import types
import unittest
import unittest.mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sirius_worker import tracking  # noqa: E402

AVAILABLE, WHY = tracking.available()


def moving_labels(t=5, z=4, y=40, x=40):
    """Three objects crossing the field at constant velocity."""
    out = np.zeros((t, z, y, x), np.uint32)
    for frame in range(t):
        for k, (y0, x0, vy, vx) in enumerate([(8, 8, 2, 1), (20, 30, -1, 2), (30, 10, 0, -2)], start=1):
            yc, xc = int(y0 + vy * frame), int(x0 + vx * frame)
            out[frame, 1:3, yc - 2:yc + 3, xc - 2:xc + 3] = k
    return out


class TestAvailability(unittest.TestCase):
    def test_reports_why_it_cannot_run(self):
        ok, why = tracking.available()
        self.assertIsInstance(ok, bool)
        if ok:
            self.assertEqual(why, "")
        else:
            # the reason has to be actionable, not just "no"
            self.assertTrue(why)
            self.assertTrue(any(w in why for w in ("pip install", "libstdc++", "will not load")), why)

    @unittest.skipIf(AVAILABLE, "btrack loads here")
    def test_refuses_clearly_when_unavailable(self):
        with self.assertRaises(tracking.NotAvailable) as cm:
            tracking.run_btrack(moving_labels(), (1.0, 1.0, 1.0), {})
        self.assertTrue(str(cm.exception))


@unittest.skipUnless(AVAILABLE, f"btrack unavailable: {WHY}")
class TestBtrack(unittest.TestCase):
    def test_follows_three_objects(self):
        labels = moving_labels()
        out, info = tracking.run_btrack(labels, (1.0, 1.0, 1.0), {"max_distance": 8.0, "min_length": 2})
        self.assertEqual(out.shape, labels.shape)
        self.assertEqual(info["objects"], 15)
        self.assertEqual(info["tracks"], 3)
        self.assertEqual(info["longest"], 5)
        # every object voxel keeps a track id, and one object keeps one id
        for t in range(labels.shape[0]):
            for k in (1, 2, 3):
                ids = set(np.unique(out[t][labels[t] == k]).tolist())
                self.assertEqual(len(ids), 1, f"frame {t} object {k} split across {ids}")

    def test_short_tracks_are_dropped(self):
        labels = moving_labels()
        labels[1:, ...][labels[1:, ...] == 3] = 0   # object 3 exists in one frame only
        _, info = tracking.run_btrack(labels, (1.0, 1.0, 1.0), {"max_distance": 8.0, "min_length": 3})
        self.assertEqual(info["tracks"], 2)

    def test_a_missing_configuration_is_named(self):
        with self.assertRaises(tracking.NotAvailable) as cm:
            tracking.run_btrack(moving_labels(), (1.0, 1.0, 1.0), {"config": "/no/such/config.json"})
        self.assertIn("config", str(cm.exception).lower())

    def test_a_2d_time_series_is_tracked(self):
        # (t, 1, y, x): every 2-D series reaches run_btrack like this, and a
        # (y, x) scale for it made btrack raise before tracking anything
        labels = moving_labels(z=4)[:, 1:2]
        out, info = tracking.run_btrack(labels, (0.5, 0.5, 2.0), {"max_distance": 4.0, "min_length": 2})
        self.assertEqual(info["tracks"], 3)
        self.assertEqual(int((out > 0).sum()), int((labels > 0).sum()))


class _StubTrack:
    """One btrack tracklet: the frames it was seen in and where it was."""

    def __init__(self, ident, t, z, y, x, parent=None):
        self.ID, self.t, self.z, self.y, self.x, self.parent = ident, t, z, y, x, parent


class _StubTracker:
    """Enough of BayesianTracker to reach run_btrack's own relabelling. Like
    btrack, it tracks only the objects it was given: without append() its
    queue is empty and it returns no tracks."""

    planned: list = []   # what tracking the appended objects "finds"

    def __init__(self):
        self.objects = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def configure(self, config):
        pass

    def append(self, objects):
        self.objects = list(objects)

    def track(self, **kwargs):
        pass

    def optimize(self):
        pass

    @property
    def tracks(self):
        return list(_StubTracker.planned) if self.objects else []


def _stub_segmentation_to_objects(labels, scale=None):
    """btrack's check: one scale per spatial axis of a frame."""
    if scale is not None and len(scale) != labels.ndim - 1:
        raise ValueError(f"Scale dimensions do not match segmentation: {scale}.")
    return ["object"] * 3


class TestRelabelling(unittest.TestCase):
    """What run_btrack does with the tracks it is handed, without btrack: the
    relabelling is ours, and it is where a track can quietly lose its object."""

    def setUp(self):
        stub = types.ModuleType("btrack")
        stub.utils = types.SimpleNamespace(segmentation_to_objects=_stub_segmentation_to_objects)
        stub.BayesianTracker = _StubTracker
        stub.datasets = types.SimpleNamespace(cell_config=lambda: "config")
        stub.libwrapper = types.SimpleNamespace(get_library=lambda: None)
        modules = {"btrack": stub, "btrack.utils": stub.utils, "btrack.libwrapper": stub.libwrapper}
        patch = unittest.mock.patch.dict(sys.modules, modules)
        patch.start()
        self.addCleanup(patch.stop)
        available = unittest.mock.patch.object(tracking, "available", lambda: (True, ""))
        available.start()
        self.addCleanup(available.stop)

    def test_a_ring_keeps_its_track_although_its_centroid_is_on_the_background(self):
        # a ring, a C, a bent filament: the centroid btrack reports is outside
        # the object, so the label under it is background and the track would
        # repaint nothing at all
        labels = np.zeros((3, 1, 21, 21), np.uint32)
        yy, xx = np.ogrid[:21, :21]
        radius = (yy - 10) ** 2 + (xx - 10) ** 2
        labels[:, 0][:, (radius <= 64) & (radius >= 25)] = 1
        self.assertEqual(int(labels[0, 0, 10, 10]), 0, "the centroid is meant to be on the background")
        _StubTracker.planned = [_StubTrack(1, [0, 1, 2], [0, 0, 0], [10.0] * 3, [10.0] * 3)]

        out, info = tracking.run_btrack(labels, (1.0, 1.0, 1.0), {"min_length": 2})
        self.assertEqual(info["tracks"], 1)
        for t in range(3):
            self.assertEqual(int((out[t] > 0).sum()), int((labels[t] > 0).sum()), f"frame {t} lost its object")
            self.assertEqual(set(np.unique(out[t][labels[t] == 1]).tolist()), {1})

    def test_the_objects_are_handed_to_the_tracker(self):
        labels = np.zeros((2, 1, 10, 10), np.uint32)
        labels[:, 0, 2:5, 2:5] = 1
        _StubTracker.planned = [_StubTrack(1, [0, 1], [0, 0], [3.0, 3.0], [3.0, 3.0])]
        out, info = tracking.run_btrack(labels, (1.0, 1.0, 1.0), {"min_length": 2})
        self.assertEqual(info["tracks"], 1)
        self.assertEqual(int((out > 0).sum()), int((labels > 0).sum()))

    def test_a_2d_series_is_scaled_on_every_axis_of_a_frame(self):
        # (t, 1, y, x) frames are 3-D to btrack; a two-number scale is refused
        labels = np.zeros((2, 1, 10, 10), np.uint32)
        labels[:, 0, 2:5, 2:5] = 1
        _StubTracker.planned = [_StubTrack(1, [0, 1], [0.0, 0.0], [1.5, 1.5], [1.5, 1.5])]
        out, info = tracking.run_btrack(labels, (0.5, 0.5, 2.0), {"min_length": 2})
        self.assertEqual(info["tracks"], 1)
        self.assertEqual(int((out > 0).sum()), int((labels > 0).sum()))

    def test_one_division_is_one_division(self):
        # btrack gives every daughter the mother's id as its parent, so a
        # division is seen twice if the daughters are what gets counted
        labels = np.zeros((2, 1, 10, 10), np.uint32)
        labels[:, 0, 2:5, 2:5] = 1
        labels[:, 0, 6:9, 6:9] = 2
        _StubTracker.planned = [_StubTrack(1, [0, 1], [0, 0], [3.0, 3.0], [3.0, 3.0], parent=1),
                                _StubTrack(2, [0, 1], [0, 0], [7.0, 7.0], [7.0, 7.0], parent=1)]
        _, info = tracking.run_btrack(labels, (1.0, 1.0, 1.0), {"min_length": 2})
        self.assertEqual(info["tracks"], 2)
        self.assertEqual(info["divisions"], 1)

    def test_lineage_is_reported_in_the_ids_the_labels_carry(self):
        # btrack ids are not label ids: tracks come back renumbered 1..n in
        # the order they are kept, and a dropped track takes its links along
        labels = np.zeros((3, 1, 12, 12), np.uint32)
        labels[:, 0, 1:4, 1:4] = 1
        labels[:, 0, 5:8, 5:8] = 2
        labels[:, 0, 9:12, 9:12] = 3
        _StubTracker.planned = [_StubTrack(40, [0, 1, 2], [0] * 3, [2.0] * 3, [2.0] * 3, parent=40),
                               _StubTrack(41, [1, 2], [0] * 2, [6.0] * 2, [6.0] * 2, parent=40),
                               _StubTrack(42, [2], [0], [10.0], [10.0], parent=40),        # too short
                               _StubTrack(43, [1, 2], [0] * 2, [10.0] * 2, [10.0] * 2, parent=99)]
        out, info = tracking.run_btrack(labels, (1.0, 1.0, 1.0), {"min_length": 2})
        self.assertEqual(info["tracks"], 3)
        self.assertEqual(info["lineage"], {"2": 1})
        self.assertEqual(set(np.unique(out).tolist()), {0, 1, 2, 3})
        json.dumps(info)   # it travels in the result header


if __name__ == "__main__":
    unittest.main()
