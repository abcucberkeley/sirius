"""The numpy mirror of the operations (bindings/python/sirius/workbench.py),
loaded the way the worker loads it: behaviours that must agree with the C++
operations, on inputs the parity fixtures do not cover."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sirius_worker.steps import workbench  # noqa: E402

try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

wb = workbench()


def _meta(t=1, z=1, y=8, x=8, c=1):
    return {"dims": [c, t, z, y, x], "voxel_um": [0.1, 0.1, 0.3]}


@unittest.skipUnless(HAVE_SCIPY, "scipy is not installed")
class TestTracking(unittest.TestCase):
    def test_gap_closing_follows_a_chain_of_merges(self):
        # one object present at t = 0, 2, 4 and absent in between: with a gap
        # of one frame allowed, that is one track of three points -- the
        # application follows the merge chain (tracking.cpp), and so must the
        # mirror, which used to close the first gap and then drop the rest
        t, z, y, x = 5, 1, 16, 16
        labels = np.zeros((t, z, y, x), dtype=np.uint32)
        for frame in (0, 2, 4):
            labels[frame, 0, 6:10, 6:10] = 1
        a = np.zeros((1, t, z, y, x), dtype=np.float32)
        params = {"tracker": "Built-in (assignment)", "max_distance": 5.0, "max_gap": 1, "min_length": 2,
                  "overlap_weight": 0.0, "relabel": True}
        result = wb.run_step("track", params, a, _meta(t=t, z=z, y=y, x=x), labels=labels)
        out = result.labels
        self.assertIsNotNone(out)
        ids = [int(np.unique(out[frame][out[frame] > 0])[0]) for frame in (0, 2, 4)]
        self.assertEqual(ids, [ids[0]] * 3, "one object, one track id in every frame it appears")
        self.assertEqual(int(out.max()), 1)


class TestCleanup(unittest.TestCase):
    def test_relabel_off_keeps_the_ids(self):
        labels = np.zeros((1, 1, 8, 8), dtype=np.uint32)
        labels[0, 0, 1:4, 1:4] = 5
        labels[0, 0, 5:7, 5:7] = 9
        labels[0, 0, 0, 7] = 7   # a speck
        a = np.zeros((1, 1, 1, 8, 8), dtype=np.float32)
        kept = wb.run_step("cleanup", {"min_voxels": 2, "relabel": False}, a, _meta(), labels=labels).labels
        self.assertEqual(int(kept[0, 0, 2, 2]), 5)
        self.assertEqual(int(kept[0, 0, 5, 5]), 9)
        self.assertEqual(int(kept[0, 0, 0, 7]), 0)
        dense = wb.run_step("cleanup", {"min_voxels": 2, "relabel": True}, a, _meta(), labels=labels).labels
        self.assertEqual(sorted(int(v) for v in np.unique(dense)), [0, 1, 2])


@unittest.skipUnless(HAVE_SCIPY, "scipy is not installed")
class TestFrangi(unittest.TestCase):
    def test_a_line_on_a_single_plane_is_found(self):
        plane = np.zeros((1, 1, 1, 48, 48), dtype=np.float32)
        plane[0, 0, 0, 23:26, 4:44] = 1000.0
        params = {"enhance": "Tubes (Frangi)", "enhance_sigma": 1.0, "enhance_sigma_max": 3.0, "enhance_scales": 3,
                  "sigma": 0.0, "opening": 0, "fill_holes": False, "method": "Otsu",
                  "post": "Connected components", "min_voxels": 5}
        labels = wb.run_step("classic", params, plane, _meta(y=48, x=48)).labels
        self.assertIsNotNone(labels)
        self.assertNotEqual(int(labels[0, 0, 24, 24]), 0)
        self.assertNotEqual(int(labels[0, 0, 24, 12]), 0)
        self.assertEqual(int(labels[0, 0, 5, 5]), 0)


if __name__ == "__main__":
    unittest.main()


class TestSparseIds(unittest.TestCase):
    def test_remove_small_copes_with_an_id_near_2_32(self):
        labels = np.zeros((1, 1, 4, 8), dtype=np.uint32)
        labels[0, 0, 0, 0:3] = 4_000_000_000   # a bincount by id would be 16 GB
        labels[0, 0, 1, 0:2] = 7
        labels[0, 0, 2, 0] = 3
        dense = wb._remove_small(labels, 2, True)
        self.assertEqual(sorted(int(v) for v in np.unique(dense)), [0, 1, 2])
        self.assertEqual(int(dense[0, 0, 1, 0]), 1)   # numbered in id order: 7 first, then the huge one
        self.assertEqual(int(dense[0, 0, 0, 0]), 2)
        kept = wb._remove_small(labels, 2, False)
        self.assertEqual(int(kept[0, 0, 0, 0]), 4_000_000_000)
        self.assertEqual(int(kept[0, 0, 1, 0]), 7)
        self.assertEqual(int(kept[0, 0, 2, 0]), 0)
        flags = wb._label_flags(labels[0], 0.6, 4.0)
        self.assertIn(4_000_000_000, flags["touching border"])


class TestResampleLegacyKeys(unittest.TestCase):
    def test_voxel_um_is_xyz_like_the_metadata(self):
        a = np.zeros((1, 1, 4, 16, 16), dtype=np.float32)
        meta = {"dims": [1, 1, 4, 16, 16], "voxel_um": [0.1, 0.1, 0.4]}
        # the metadata's key, in the metadata's order: x, y, z
        out = wb.run_step("resample", {"voxel_um": [0.2, 0.2, 0.8], "interpolation": "nearest"}, a, meta)
        self.assertEqual([round(v, 6) for v in out.meta["voxel_um"]], [0.2, 0.2, 0.8])
        self.assertEqual(tuple(out.array.shape), (1, 1, 2, 8, 8))


class TestSparseIdsElsewhere(unittest.TestCase):
    def test_shape_filter_and_cleanup_cope_with_an_id_near_2_32(self):
        labels = np.zeros((1, 2, 8, 8), dtype=np.uint32)
        labels[0, :, 1:4, 1:4] = 4_000_000_000
        labels[0, :, 5:7, 5:7] = 7
        labels[0, 0, 0, 7] = 3
        a = np.zeros((1, 1, 2, 8, 8), dtype=np.float32)
        result = wb.run_step("cleanup", {"min_voxels": 2, "relabel": True}, a, _meta(z=2), labels=labels)
        self.assertEqual(result.info["labels"], 2)
        filtered = wb._filter_labels_by_shape(labels[0], 0, 0.0, 0.0, False)
        self.assertEqual(sorted(int(v) for v in np.unique(filtered)), [0, 1, 2, 3])

    def test_cleanup_flags_cover_every_time_point(self):
        labels = np.zeros((2, 1, 16, 16), dtype=np.uint32)
        labels[0, 0, 4:8, 4:8] = 1       # an interior object in frame 0 ...
        labels[1, 0, 0:4, 0:4] = 1       # ... touching the border in frame 1
        labels[:, 0, 12:14, 12:14] = 2
        a = np.zeros((1, 2, 1, 16, 16), dtype=np.float32)
        result = wb.run_step("cleanup", {"min_voxels": 1, "relabel": False}, a, _meta(t=2, y=16, x=16), labels=labels)
        self.assertIn(1, result.info["flags"]["touching border"])   # frame 1's verdict is not lost to frame 0's
