"""The worker's scikit-image segmentation methods: every method returns dense
instance labels of the right shape, the seeded ones say so when nothing seeded,
and a bad method name is refused rather than guessed at."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sirius_worker import skimage_seg  # noqa: E402

AVAILABLE, WHY = skimage_seg.available()


def two_balls(z=8, y=32, x=32):
    """Two well separated bright balls on a dark ground, with a little noise."""
    volume = np.zeros((z, y, x), dtype=np.float32)
    zz, yy, xx = np.ogrid[:z, :y, :x]
    for cz, cy, cx in ((4, 10, 10), (4, 22, 22)):
        r2 = (zz - cz) ** 2 * 4.0 + (yy - cy) ** 2 + (xx - cx) ** 2
        volume[r2 <= 36] = 1.0
    volume += np.random.default_rng(0).normal(0.0, 0.02, volume.shape).astype(np.float32)
    return volume


@unittest.skipUnless(AVAILABLE, WHY)
class TestMethods(unittest.TestCase):
    def setUp(self):
        self.volume = two_balls()

    def test_every_method_returns_labels_of_the_right_shape(self):
        for method in skimage_seg.METHODS:
            params = {"method": method, "seed_depth": 1.0, "n_segments": 20, "iterations": 5,
                      "min_voxels": 5, "scale": 50.0, "min_size": 5}
            labels, info = skimage_seg.run(self.volume, params)
            self.assertEqual(labels.shape, self.volume.shape, method)
            self.assertEqual(labels.dtype, np.uint32, method)
            self.assertEqual(info["method"], method)
            self.assertEqual(int(info["labels"]), int(labels.max()), method)
            # dense: 1..n with nothing skipped
            present = np.unique(labels)
            present = present[present > 0]
            self.assertEqual(list(present), list(range(1, int(labels.max()) + 1)), method)

    def test_the_random_walker_separates_two_objects(self):
        labels, info = skimage_seg.run(self.volume, {"method": "Random walker", "seed_depth": 1.0, "min_voxels": 20})
        self.assertGreaterEqual(int(info["seeds"]), 2)
        self.assertGreaterEqual(int(labels.max()), 2)
        # the two centres belong to different objects
        self.assertNotEqual(int(labels[4, 10, 10]), int(labels[4, 22, 22]))
        self.assertNotEqual(int(labels[4, 10, 10]), 0)

    def test_a_seeded_method_says_when_nothing_seeded_it(self):
        flat = np.zeros((4, 8, 8), dtype=np.float32)
        labels, info = skimage_seg.run(flat, {"method": "Random walker"})
        self.assertEqual(int(labels.max()), 0)
        self.assertIn("note", info)
        self.assertEqual(int(info["seeds"]), 0)

    def test_felzenszwalb_says_it_works_a_plane_at_a_time(self):
        labels, info = skimage_seg.run(self.volume, {"method": "Superpixels (Felzenszwalb)", "scale": 50.0,
                                                     "min_size": 5, "min_voxels": 1})
        self.assertIn("2D method", info["note"])
        # no piece spans two planes
        for value in np.unique(labels):
            if value == 0:
                continue
            planes = np.unique(np.nonzero(labels == value)[0])
            self.assertEqual(planes.size, 1, f"label {value} spans {planes}")

    def test_small_objects_are_dropped(self):
        loose = {"method": "Superpixels (SLIC)", "n_segments": 30, "min_voxels": 1}
        strict = {**loose, "min_voxels": 400}
        many, _ = skimage_seg.run(self.volume, loose)
        few, _ = skimage_seg.run(self.volume, strict)
        self.assertGreater(int(many.max()), int(few.max()))

    def test_a_single_plane_goes_through_the_active_contour(self):
        # a 2D dataset reaches the worker as one plane; the edge map and the
        # contour both differentiate along every axis, so a volume one voxel
        # deep has to be handed over as the 2D image it is
        plane = two_balls(1, 32, 32)
        labels, info = skimage_seg.run(plane, {"method": "Active contour (geodesic)", "iterations": 5,
                                               "min_voxels": 5})
        self.assertEqual(labels.shape, plane.shape)
        self.assertGreaterEqual(int(labels.max()), 1)
        self.assertEqual(int(info["labels"]), int(labels.max()))

    def test_an_all_foreground_mask_still_yields_its_objects(self):
        # with no background there is no background marker, and the random
        # walker renumbers the markers it is given: object 1 must not be
        # mistaken for the background and dropped
        labels, info = skimage_seg.run(self.volume, {"method": "Random walker", "threshold": -1.0,
                                                     "seed_depth": 1.0, "min_voxels": 1})
        self.assertGreaterEqual(int(info["seeds"]), 1)
        self.assertEqual(int(labels.max()), int(info["seeds"]))

    def test_a_non_finite_voxel_does_not_flatten_the_volume(self):
        # one inf makes the intensity span infinite and one NaN is refused
        # outright by SLIC, so both have to be dealt with before normalising
        for bad in (np.inf, np.nan):
            spoiled = self.volume.copy()
            spoiled[0, 0, 0] = bad
            for method in ("Superpixels (SLIC)", "Random walker"):
                labels, _ = skimage_seg.run(spoiled, {"method": method, "n_segments": 20, "seed_depth": 1.0,
                                                      "min_voxels": 1})
                clean, _ = skimage_seg.run(self.volume, {"method": method, "n_segments": 20, "seed_depth": 1.0,
                                                         "min_voxels": 1})
                self.assertEqual(labels.shape, spoiled.shape)
                self.assertGreaterEqual(int(labels.max()), 1, f"{method} with {bad}")
                self.assertAlmostEqual(int(labels.max()), int(clean.max()), delta=2,
                                       msg=f"{method}: one {bad} changed the segmentation")

    def test_a_manual_threshold_is_used_instead_of_otsu(self):
        # nothing is above 5.0, so the mask is empty and nothing can seed
        labels, info = skimage_seg.run(self.volume, {"method": "Random walker", "threshold": 5.0})
        self.assertEqual(int(labels.max()), 0)
        self.assertEqual(int(info["seeds"]), 0)


class TestGuards(unittest.TestCase):
    """These hold whether or not scikit-image is installed: the caller's own
    mistakes are reported before the machine's missing package."""

    def test_an_unknown_method_is_refused(self):
        with self.assertRaises(skimage_seg.SkimageError) as caught:
            skimage_seg.run(np.zeros((2, 4, 4), np.float32), {"method": "Magic"})
        self.assertIn("unknown method", str(caught.exception))

    def test_a_volume_has_to_be_three_dimensional(self):
        with self.assertRaises(skimage_seg.SkimageError) as caught:
            skimage_seg.run(np.zeros((4, 4), np.float32), {"method": "Superpixels (SLIC)"})
        self.assertIn("(z, y, x)", str(caught.exception))

    def test_availability_reports_a_reason_when_it_is_missing(self):
        ok, why = skimage_seg.available()
        self.assertEqual(ok, why == "")
        if not ok:
            self.assertIn("scikit-image", why)


class TestCleanSparseIds(unittest.TestCase):
    def test_clean_copes_with_an_id_near_2_32(self):
        from sirius_worker import skimage_seg

        labels = np.zeros((4, 8, 8), dtype=np.int64)
        labels[:, 1:4, 1:4] = 4_000_000_000
        labels[:, 5:7, 5:7] = 7
        labels[0, 0, 7] = 3
        out = skimage_seg._clean(labels, 2)
        self.assertEqual(sorted(int(v) for v in np.unique(out)), [0, 1, 2])
        self.assertEqual(int(out[0, 5, 5]), 1)   # in id order: 7 before the huge one
        self.assertEqual(int(out[0, 1, 1]), 2)
