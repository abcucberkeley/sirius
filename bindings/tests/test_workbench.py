"""Tests of sirius.workbench: loading a synthetic TIFF and running a pipeline
of numpy steps (the code path of "Export pipeline as Python script"), and the
individual steps against the semantics of their C++ counterparts in
app/core/ops (the parameter keys are the application's; see
test_workbench_schema.py for the key / default / choice drift check)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

try:
    import scipy  # noqa: F401
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

try:
    import skimage  # noqa: F401
    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False

try:
    import tifffile  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    tifffile = None


def _load_workbench():
    """sirius.workbench of this tree. The installed package may be another
    checkout (an editable install elsewhere), so load the file beside this test
    when the import does not resolve to it."""
    here = Path(__file__).resolve().parents[1] / "python" / "sirius" / "workbench.py"
    try:
        import sirius.workbench as wb  # type: ignore

        if Path(wb.__file__).resolve() == here:
            return wb
    except Exception:  # noqa: BLE001
        pass
    spec = importlib.util.spec_from_file_location("sirius_workbench_under_test", here)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sirius_workbench_under_test"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


wb = _load_workbench()


def _no_sim_meta():
    return {"present": False, "ndirs": 3, "nphases": 5, "fast_si": False}


@unittest.skipIf(tifffile is None, "tifffile not installed")
class TestRunPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(3)
        # (t, z, c, y, x) ImageJ hyperstack: 2 channels, 3 time points, 4 planes
        cls.data = (rng.random((3, 4, 2, 16, 24)) * 1000).astype(np.uint16)
        cls.path = os.path.join(cls.tmp.name, "stack.tif")
        tifffile.imwrite(cls.path, cls.data, imagej=True, resolution=(1 / 0.1, 1 / 0.1),
                         metadata={"axes": "TZCYX", "spacing": 0.3, "unit": "um"})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_load_dataset_reads_hyperstack_dims_and_voxel(self):
        a, meta = wb.load_dataset(self.path)
        self.assertEqual(a.shape, (2, 3, 4, 16, 24))
        self.assertEqual(a.dtype, np.float32)
        self.assertEqual(meta["dims"], {"c": 2, "t": 3, "z": 4, "y": 16, "x": 24})
        self.assertAlmostEqual(meta["voxel_um"][2], 0.3, places=6)
        self.assertAlmostEqual(meta["voxel_um"][0], 0.1, places=6)
        np.testing.assert_array_equal(a[1, 2, 3], self.data[2, 3, 1].astype(np.float32))

    def test_pipeline_einsum_contrast_merge(self):
        # the application's keys, as "Export pipeline as Python script" writes them
        pipeline = {
            "version": 1,
            "steps": [
                {"kind": "load", "name": "Load", "enabled": True, "params": {}},
                {"kind": "einsum", "name": "Einsum reduce", "enabled": True,
                 "params": {"keep": "czyx", "reduction": "mean"}},
                {"kind": "contrast", "name": "Contrast", "enabled": True,
                 "params": {"min": 0.0, "max": 0.0, "gamma": 1.0, "lo_percentile": 1.0, "hi_percentile": 99.0,
                            "bake": True}},
                {"kind": "deskew", "name": "Deskew + rotate", "enabled": False, "params": {}},
                {"kind": "merge", "name": "Merge channels", "enabled": True,
                 "params": {"blend": "Additive", "colors": ["#ff0000", "#00ff00"], "weights": [],
                            "normalize_percentile": 99.9}},
            ],
        }
        messages = []
        with warnings.catch_warnings():
            warnings.simplefilter("error", wb.UnknownParameterWarning)
            out, meta = wb.run_pipeline(self.path, pipeline, progress=lambda f, m: messages.append((f, m)))
        self.assertEqual(out.shape, (3, 1, 4, 16, 24))
        self.assertTrue(meta["rgb"])
        self.assertEqual(meta["dims"]["c"], 3)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)
        # one window for both channels: the extreme percentiles over them
        expected = self.data.astype(np.float32).mean(axis=0)  # (z, c, y, x)
        chans = np.transpose(expected, (1, 0, 2, 3))
        lo = min(wb._percentiles(chans[c], 1.0, 99.0)[0] for c in range(2))
        hi = max(wb._percentiles(chans[c], 1.0, 99.0)[1] for c in range(2))
        red = np.clip((chans[0] - lo) / (hi - lo), 0, 1)
        np.testing.assert_allclose(out[0, 0], red, atol=1e-6)
        self.assertEqual(float(out[2].max()), 0.0)
        self.assertEqual(messages[-1][0], 1.0)

    def test_older_keys_still_run_without_warnings(self):
        pipeline = [{"kind": "load", "params": {}},
                    {"kind": "einsum", "params": {"axes": "czyx", "reduction": "mean"}},
                    {"kind": "contrast", "params": {"low": 1.0, "high": 99.0}}]
        with warnings.catch_warnings():
            warnings.simplefilter("error", wb.UnknownParameterWarning)
            out, _ = wb.run_pipeline(self.path, pipeline)
        self.assertEqual(out.shape, (2, 1, 4, 16, 24))

    def test_unsupported_step_raises_or_is_skipped(self):
        pipeline = [{"kind": "load", "params": {}}, {"kind": "decon", "enabled": True, "params": {}}]
        with self.assertRaises(NotImplementedError) as cm:
            wb.run_pipeline(self.path, pipeline)
        self.assertIn("decon", str(cm.exception))
        out, meta = wb.run_pipeline(self.path, pipeline, strict=False)
        self.assertEqual(meta["skipped"], ["decon"])
        self.assertEqual(out.shape, (2, 3, 4, 16, 24))

    def test_json_string_pipeline_and_crop(self):
        pipeline = json.dumps({"steps": [{"kind": "croppad", "params": {"z0": 1, "y0": 2, "x0": -2, "z": 2, "y": 8,
                                                                        "x": 10, "fill": -1}}]})
        out, meta = wb.run_pipeline(self.path, pipeline)
        self.assertEqual(out.shape, (2, 3, 2, 8, 10))
        self.assertEqual(float(out[0, 0, 0, 0, 0]), -1.0)  # padded column
        np.testing.assert_array_equal(out[0, 0, 0, 0, 2:], self.data[0, 1, 0, 2, 0:8].astype(np.float32))
        # the older origin / size lists mean the same
        legacy = [{"kind": "croppad", "params": {"origin": [1, 2, -2], "size": [2, 8, 10], "fill": -1}}]
        out2, _ = wb.run_pipeline(self.path, legacy)
        np.testing.assert_array_equal(out2, out)

    @unittest.skipUnless(_HAVE_SCIPY, "labelling needs scipy")
    def test_labels_do_not_outlive_a_step_that_changes_the_grid(self):
        # threshold -> resample: the labels cover the old grid, so the
        # application drops them (executor.cpp, labelsFit); a crop after the
        # resample used to cut the stale 64 x 64 labels as though they fitted
        img = np.zeros((4, 64, 64), np.float32)
        img[:, 10:20, 10:20] = 100.0
        img[:, 40:50, 40:55] = 100.0
        path = os.path.join(self.tmp.name, "grid.tif")
        tifffile.imwrite(path, img)
        steps = [{"kind": "load", "params": {}},
                 {"kind": "threshold", "params": {"method": "Manual", "value": 50.0, "min_voxels": 1}},
                 {"kind": "contrast", "params": {}}]
        _, meta = wb.run_pipeline(path, {"steps": steps})
        self.assertEqual(meta["labels"].shape, (1, 4, 64, 64))   # the grid is unchanged: carried through
        steps[2] = {"kind": "resample", "params": {"voxel_x": 0.2, "voxel_y": 0.2}}
        out, meta = wb.run_pipeline(path, {"steps": steps})
        self.assertEqual(out.shape, (1, 1, 4, 32, 32))
        self.assertNotIn("labels", meta)
        steps.append({"kind": "croppad", "params": {"x0": 2}})
        out, meta = wb.run_pipeline(path, {"steps": steps})
        self.assertEqual(out.shape, (1, 1, 4, 32, 30))
        self.assertNotIn("labels", meta)
        steps[2] = {"kind": "maxproj", "params": {"axis": "z"}}
        _, meta = wb.run_pipeline(path, {"steps": steps[:3]})
        self.assertNotIn("labels", meta)

    def test_load_step_voxel_overrides(self):
        pipeline = [{"kind": "load", "params": {"voxel_x": 0.05, "voxel_y": 0.0, "voxel_z": 0.5}}]
        _, meta = wb.run_pipeline(self.path, pipeline)
        self.assertEqual([round(v, 6) for v in meta["voxel_um"]], [0.05, 0.1, 0.5])


def _sirius_extension():
    try:
        import sirius  # type: ignore

        return sirius if hasattr(sirius, "SimReconstructor") else None
    except Exception:  # noqa: BLE001
        return None


@unittest.skipIf(_sirius_extension() is None, "sirius extension not importable")
class TestSimStep(unittest.TestCase):
    DATA = Path(__file__).resolve().parents[2] / "tests" / "data"

    def test_sim_step_reproduces_the_reference_reconstruction(self):
        sirius = _sirius_extension()
        raw = sirius.read_tiff(str(self.DATA / "raw.tif"), dtype=np.float32)
        expected = sirius.read_tiff(str(self.DATA / "raw_proc.tif"), dtype=np.float32)
        params = {"mode": "From file", "params_file": str(self.DATA / "config.txt"), "otf": str(self.DATA / "otf.tif")}
        r = wb.run_step("sim", params, raw, {"voxel_um": [0.08, 0.08, 0.125]}, device="cpu")
        self.assertEqual(r.array.shape, (1, 1) + expected.shape)
        rel = np.max(np.abs(r.array[0, 0] - expected)) / np.max(np.abs(expected))
        self.assertLess(rel, 1e-5)
        self.assertEqual(len(r.info["fits"]), 1)
        self.assertEqual(len(r.info["fits"][0]["k0"]), 3)
        self.assertAlmostEqual(r.meta["voxel_um"][0], 0.04, places=6)
        self.assertEqual(r.meta["dims"]["z"], 9)
        # the same parameters again reuse the reconstructor
        r2 = wb.run_step("sim", params, raw, {"voxel_um": [0.08, 0.08, 0.125]}, device="cpu")
        np.testing.assert_array_equal(r2.array, r.array)

    def test_sim_step_needs_an_otf_file(self):
        raw = np.zeros((15, 8, 8), np.float32)
        with self.assertRaises(wb.NotAvailable):
            wb.run_step("sim", {"angles": 3, "phases": 5}, raw)


class TestParameters(unittest.TestCase):
    def test_unknown_keys_warn_and_are_ignored(self):
        a = np.ones((1, 2, 2, 4, 4), np.float32)
        with self.assertWarns(wb.UnknownParameterWarning) as cm:
            r = wb.run_step("bleach", {"mode": "Match mean", "to_the_moon": 1}, a)
        self.assertIn("bleach", str(cm.warning))
        self.assertIn("to_the_moon", str(cm.warning))
        self.assertEqual(r.array.shape, a.shape)

    def test_canonical_key_wins_over_an_alias(self):
        a = np.zeros((1, 1, 1, 6, 6), np.float32)
        a[0, 0, 0, 1:3, 1:3] = 5.0
        p = wb._prepare_params(wb.step_spec("threshold"), {"threshold": 9.0, "value": 1.0, "method": "Manual"}, None)
        self.assertEqual(p["value"], 1.0)
        self.assertNotIn("threshold", p)

    def test_defaults_are_filled(self):
        p = wb._prepare_params(wb.step_spec("threshold"), {}, None)
        self.assertEqual(p["method"], "Otsu")
        self.assertEqual(p["min_voxels"], 20)
        self.assertEqual(p["post"], "Connected components")

    def test_kind_aliases_resolve_to_implemented_kinds(self):
        for alias, kind in wb._KIND_ALIASES.items():
            self.assertIn(kind, wb.step_kinds(), alias)
        self.assertIs(wb.step_spec("label_cleanup"), wb.step_spec("cleanup"))
        self.assertIs(wb.step_spec("classical"), wb.step_spec("classic"))


class TestIntensityHelpers(unittest.TestCase):
    def test_percentiles_are_order_statistics_with_flat_fallback(self):
        v = np.arange(101, dtype=np.float32)
        self.assertEqual(wb._percentiles(v, 10.0, 90.0), (10.0, 90.0))
        self.assertEqual(wb._percentiles(v, 0.0, 100.0), (0.0, 100.0))
        # a flat quantile pair (mostly zeros) falls back to the full range
        z = np.zeros(1000, np.float32)
        z[:3] = 7.0
        self.assertEqual(wb._percentiles(z, 0.2, 99.8), (0.0, 7.0))
        self.assertEqual(wb._percentiles(np.array([np.nan], np.float32), 0, 100), (0.0, 0.0))

    def test_otsu_separates_two_modes(self):
        rng = np.random.default_rng(0)
        v = np.concatenate([rng.normal(10, 1, 5000), rng.normal(50, 1, 5000)]).astype(np.float32)
        cut = wb._otsu_threshold(v)
        # the between-class variance is flat across the empty bins between the
        # modes and the C++ loop keeps the first maximum ('>'), so the cut sits
        # at the top of the lower mode -- but it does separate the two
        self.assertTrue((v[:5000] < cut).all())
        self.assertTrue((v[5000:] > cut).all())
        # bin edge semantics: mn + (mx - mn) * (best + 1) / 256
        self.assertEqual(wb._otsu_threshold(np.array([3.0, 3.0], np.float32)), 3.0)
        two = np.array([0.0, 1.0], np.float32)
        self.assertAlmostEqual(wb._otsu_threshold(two), 1.0 / 256, places=7)

    def test_rescale_gamma(self):
        a = np.array([[[[[-1.0, 0.0, 0.5, 1.0, 2.0]]]]], np.float32)
        out = wb._rescale_gamma(a, 0.0, 1.0, 1.0)
        np.testing.assert_allclose(out[0, 0, 0, 0], [0, 0, 0.5, 1, 1])
        out = wb._rescale_gamma(a, 0.0, 1.0, 2.0)
        np.testing.assert_allclose(out[0, 0, 0, 0, 2], 0.5 ** 0.5, rtol=1e-6)
        empty = wb._rescale_gamma(a, 1.0, 1.0, 1.0)
        np.testing.assert_allclose(empty[0, 0, 0, 0], [0, 0, 0, 0, 1])


class TestSteps(unittest.TestCase):
    def test_reductions_keep_axes_with_length_one(self):
        a = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.float32).reshape(2, 3, 4, 5, 6)
        r = wb.run_step("einsum", {"keep": "ctyx", "reduction": "max"}, a)
        self.assertEqual(r.array.shape, (2, 3, 1, 5, 6))
        np.testing.assert_array_equal(r.array[:, :, 0], a.max(axis=2))
        r = wb.run_step("maxproj", {"axis": "z"}, a)
        self.assertEqual(r.array.shape, (2, 3, 1, 5, 6))
        r = wb.run_step("meant", {}, a)
        self.assertEqual(r.array.shape, (2, 1, 4, 5, 6))
        np.testing.assert_allclose(r.array[:, 0], a.mean(axis=1), rtol=1e-6)
        r = wb.run_step("einsum", {"keep": "tzyx", "reduction": "sum"}, a, {"channels": [{"label": "a"}, {"label": "b"}]})
        self.assertEqual(len(r.meta["channels"]), 1)

    def test_contrast_manual_window_and_auto_window(self):
        a = np.linspace(0, 100, 2 * 1 * 2 * 8 * 8, dtype=np.float32).reshape(2, 1, 2, 8, 8)
        r = wb.run_step("contrast", {"min": 25.0, "max": 75.0, "gamma": 1.0}, a)
        self.assertFalse(r.info["automatic"])
        np.testing.assert_allclose(r.array, np.clip((a - 25) / 50, 0, 1), atol=1e-6)
        # max <= min: the lo / hi percentiles over both channels
        r = wb.run_step("contrast", {"min": 0.0, "max": 0.0, "lo_percentile": 0.0, "hi_percentile": 100.0}, a)
        self.assertTrue(r.info["automatic"])
        self.assertEqual(r.info["window"], [0.0, 100.0])
        np.testing.assert_allclose(r.array, a / 100.0, atol=1e-6)
        with self.assertRaises(ValueError):
            wb.run_step("contrast", {"lo_percentile": 60.0, "hi_percentile": 50.0}, a)

    def test_merge_weights_and_normalization(self):
        a = np.zeros((2, 1, 1, 4, 4), np.float32)
        a[0] = 200.0   # scaled so its 99.9th percentile maps to 1
        a[1] = 0.25    # already in 0..1: left alone
        r = wb.run_step("merge", {"blend": "Additive", "colors": ["#ff0000", "#0000ff"], "weights": [0.5, 2.0]}, a)
        self.assertEqual(r.array.shape, (3, 1, 1, 4, 4))
        np.testing.assert_allclose(r.array[0], 0.5, atol=1e-6)
        np.testing.assert_allclose(r.array[2], 0.5, atol=1e-6)
        self.assertEqual(float(r.array[1].max()), 0.0)
        r = wb.run_step("merge", {"blend": "Max", "colors": ["#ffffff", "#ffffff"]}, a)
        np.testing.assert_allclose(r.array[1], 1.0, atol=1e-6)
        with self.assertRaises(ValueError):
            wb.run_step("merge", {}, r.array, r.meta)

    @unittest.skipUnless(_HAVE_SCIPY, "connected components need scipy")
    def test_threshold_methods_and_min_voxels(self):
        a = np.zeros((1, 1, 4, 10, 10), np.float32)
        a[0, 0, :, 1:3, 1:3] = 5.0
        a[0, 0, :, 6:9, 6:9] = 7.0
        r = wb.run_step("threshold", {"channel": 0, "method": "Manual", "value": 1.0, "min_voxels": 0}, a)
        self.assertIsNotNone(r.labels)
        self.assertEqual(r.labels.shape, (1, 4, 10, 10))
        self.assertEqual(int(r.labels.max()), 2)
        self.assertEqual(r.info["thresholds"], [1.0])
        r2 = wb.run_step("threshold", {"channel": 0, "method": "Manual", "value": 1.0, "min_voxels": 20}, a)
        self.assertEqual(int(r2.labels.max()), 1)
        # the application's default cut is Otsu, which separates 0 from the objects
        r3 = wb.run_step("threshold", {"channel": 0, "min_voxels": 0}, a)
        self.assertEqual(r3.info["method"], "Otsu")
        self.assertEqual(int(r3.labels.max()), 2)
        # percentiles are order statistics: the 90th of 400 voxels is 5.0, so
        # only the 7.0 block is strictly above it (the 99th would be 7.0, which
        # cuts everything away -- as it does in the application)
        r4 = wb.run_step("threshold", {"channel": 0, "method": "Percentile", "percentile": 90.0, "min_voxels": 0}, a)
        self.assertEqual(int(r4.labels.max()), 1)
        self.assertEqual(r4.info["thresholds"], [5.0])
        # the older Python-only spelling: threshold = a manual value
        r5 = wb.run_step("threshold", {"channel": 0, "threshold": 1.0, "min_voxels": 0}, a)
        np.testing.assert_array_equal(r5.labels, r.labels)

    @unittest.skipUnless(_HAVE_SCIPY, "label post-processing needs scipy")
    def test_remove_small_relabels_densely(self):
        lab = np.array([[[0, 3, 3, 0, 7, 0, 9, 9, 9]]], np.uint32)
        np.testing.assert_array_equal(wb._remove_small(lab, 0), [[[0, 1, 1, 0, 2, 0, 3, 3, 3]]])
        np.testing.assert_array_equal(wb._remove_small(lab, 2), [[[0, 1, 1, 0, 0, 0, 2, 2, 2]]])

    @unittest.skipUnless(_HAVE_SCIPY, "label post-processing needs scipy")
    def test_distance_seeds_pick_one_seed_per_blob(self):
        mask = np.zeros((1, 20, 40), bool)
        mask[0, 5:15, 5:15] = True
        mask[0, 5:15, 25:35] = True
        seeds, n = wb._distance_seeds(mask, 5.0)
        self.assertEqual(n, 2)
        self.assertEqual(int(seeds.max()), 2)
        self.assertTrue(mask[seeds > 0].all())

    @unittest.skipUnless(_HAVE_SCIPY and _HAVE_SKIMAGE, "watershed needs scipy and scikit-image")
    def test_watershed_splits_touching_blobs(self):
        # two overlapping disks: the distance transform has one maximum in each
        # and a saddle at the waist, so distanceSeeds accepts exactly two seeds
        y, x = np.mgrid[0:20, 0:40]
        blobs = ((y - 10) ** 2 + (x - 13) ** 2 <= 49) | ((y - 10) ** 2 + (x - 25) ** 2 <= 49)
        a = np.zeros((1, 1, 1, 20, 40), np.float32)
        a[0, 0, 0] = blobs
        p = {"method": "Manual", "value": 0.5, "seed_distance": 5.0, "min_voxels": 0}
        r = wb.run_step("threshold", dict(p, post="Watershed (distance)"), a)
        self.assertEqual(int(r.labels.max()), 2)
        # they touch, so connected components sees one object
        r = wb.run_step("threshold", dict(p, post="Connected components"), a)
        self.assertEqual(int(r.labels.max()), 1)

    @unittest.skipUnless(_HAVE_SCIPY, "label post-processing needs scipy")
    def test_watershed_without_skimage_is_reported(self):
        if _HAVE_SKIMAGE:
            self.skipTest("scikit-image is installed")
        a = np.zeros((1, 1, 1, 8, 8), np.float32)
        a[0, 0, 0, 2:6, 2:6] = 1.0
        with self.assertRaises(wb.NotAvailable):
            wb.run_step("threshold", {"method": "Manual", "value": 0.5, "post": "Watershed (distance)"}, a)

    @unittest.skipUnless(_HAVE_SCIPY, "classical segmentation needs scipy")
    def test_classic_segmentation_finds_blobs(self):
        rng = np.random.default_rng(1)
        a = (rng.random((1, 1, 3, 40, 40)) * 0.1).astype(np.float32)
        a[0, 0, :, 5:15, 5:15] += 1.0
        a[0, 0, :, 22:34, 20:34] += 1.0
        p = {"channel": 0, "tophat": 0, "sigma": 1.0, "method": "Otsu", "opening": 1, "fill_holes": True,
             "post": "Connected components", "min_voxels": 20}
        r = wb.run_step("classic", p, a)
        self.assertEqual(int(r.labels.max()), 2)
        self.assertEqual(r.info["labels"], 2)
        self.assertGreater(r.info["foreground_fraction"], 0.1)
        r = wb.run_step("classic", dict(p, method="Local mean", window=15, local_ratio=1.1), a)
        self.assertGreaterEqual(int(r.labels.max()), 2)
        r = wb.run_step("classic", dict(p, method="Manual", value=0.5, tophat=8), a)
        self.assertEqual(int(r.labels.max()), 2)
        # the white top-hat keeps only what is smaller than its box: a radius
        # below the blobs' own size removes them, as it does in classic.cpp
        r = wb.run_step("classic", dict(p, method="Manual", value=0.5, tophat=3), a)
        self.assertEqual(int(r.labels.max()), 0)

    def test_local_mean_plane_clamps_the_window(self):
        pl = np.arange(16, dtype=np.float32).reshape(4, 4)
        m = wb._local_mean_plane(pl, 1)
        self.assertAlmostEqual(float(m[0, 0]), float(pl[:2, :2].mean()), places=5)
        self.assertAlmostEqual(float(m[2, 2]), float(pl[1:4, 1:4].mean()), places=5)

    @unittest.skipUnless(_HAVE_SCIPY, "label post-processing needs scipy")
    def test_cleanup_drops_small_and_border_labels(self):
        a = np.zeros((1, 1, 1, 10, 10), np.float32)
        labels = np.zeros((1, 1, 10, 10), np.uint32)
        labels[0, 0, 0:4, 0:4] = 4      # touches the border, 16 voxels
        labels[0, 0, 5:9, 5:9] = 7      # interior, 16 voxels
        labels[0, 0, 5, 1] = 9          # an interior speck, below the median / 8 flag
        r = wb.run_step("cleanup", {"min_voxels": 2, "remove_border": False, "relabel": True}, a, labels=labels)
        self.assertEqual(sorted(np.unique(r.labels).tolist()), [0, 1, 2])
        self.assertEqual(r.info["labels"], 2)
        r = wb.run_step("cleanup", {"min_voxels": 2, "remove_border": True, "relabel": True}, a, labels=labels)
        self.assertEqual(sorted(np.unique(r.labels).tolist()), [0, 1])
        self.assertEqual(int((r.labels == 1).sum()), 16)
        r = wb.run_step("cleanup", {"min_voxels": 0, "remove_border": False, "relabel": False}, a, labels=labels)
        np.testing.assert_array_equal(r.labels, labels)
        self.assertIn(9, r.info["flags"]["small"])
        self.assertIn(4, r.info["flags"]["touching border"])
        with self.assertRaises(ValueError):
            wb.run_step("cleanup", {}, a)
        # the label_cleanup alias names the same step
        r2 = wb.run_step("label_cleanup", {"min_voxels": 2}, a, labels=labels)
        self.assertEqual(r2.info["labels"], 2)

    def test_resample_keeps_the_physical_field(self):
        a = np.ones((1, 2, 4, 8, 8), np.float32)
        meta = {"voxel_um": [0.1, 0.1, 0.4]}
        r = wb.run_step("resample", {"voxel_x": 0.2, "voxel_y": 0.2, "voxel_z": 0.2}, a, meta)
        # (n - 1) * d / t + 1 samples: z 3 * 0.4 / 0.2 + 1 = 7, y 7 * 0.1 / 0.2 + 1 = 4
        self.assertEqual(r.array.shape, (1, 2, 7, 4, 4))
        self.assertEqual([round(v, 6) for v in r.meta["voxel_um"]], [0.2, 0.2, 0.2])
        np.testing.assert_allclose(r.array, 1.0, atol=1e-6)
        # 0 keeps an axis; the older list spelling means the same
        r2 = wb.run_step("resample", {"voxel_x": 0.0, "voxel_y": 0.0, "voxel_z": 0.2}, a, meta)
        self.assertEqual(r2.array.shape, (1, 2, 7, 8, 8))
        r3 = wb.run_step("resample", {"voxel": [0.2, 0, 0]}, a, meta)
        np.testing.assert_array_equal(r3.array, r2.array)
        # linear interpolation of a ramp along z
        ramp = np.arange(4, dtype=np.float32).reshape(1, 1, 4, 1, 1) * np.ones((1, 1, 4, 3, 3), np.float32)
        r4 = wb.run_step("resample", {"voxel_z": 0.2}, ramp, meta)
        np.testing.assert_allclose(r4.array[0, 0, :, 0, 0], np.arange(7) * 0.5, atol=1e-6)
        r5 = wb.run_step("resample", {"voxel_z": 0.2, "interpolation": "nearest"}, ramp, meta)
        np.testing.assert_allclose(r5.array[0, 0, :, 0, 0], [0, 1, 1, 2, 2, 3, 3], atol=1e-6)
        r6 = wb.run_step("resample", {"voxel_z": 0.2, "interpolation": "cubic"}, ramp, meta)
        self.assertEqual(r6.array.shape, (1, 1, 7, 3, 3))

    def test_bleach_mode_and_over(self):
        a = np.ones((1, 2, 4, 8, 8), np.float32)
        a[0, 1] *= 0.5
        r = wb.run_step("bleach", {"mode": "Match first frame", "over": "t"}, a)
        np.testing.assert_allclose(r.array[0, 1], 1.0)
        r = wb.run_step("bleach", {"mode": "Match mean", "over": "t"}, a)
        np.testing.assert_allclose(r.array[0, 0], 0.75)
        np.testing.assert_allclose(r.array[0, 1], 0.75)
        # over z: the planes of every stack match their first plane
        b = np.ones((1, 1, 3, 4, 4), np.float32)
        b[0, 0, 1] *= 2.0
        b[0, 0, 2] *= 0.0   # an empty plane cannot be scaled
        r = wb.run_step("bleach", {"mode": "Match first frame", "over": "z"}, b)
        np.testing.assert_allclose(r.array[0, 0, 1], 1.0)
        np.testing.assert_allclose(r.array[0, 0, 2], 0.0)
        # the older to_mean flag
        r = wb.run_step("bleach", {"to_mean": True}, a)
        np.testing.assert_allclose(r.array[0, 0], 0.75)

    def test_croppad_crops_labels_and_fills_outside(self):
        a = np.arange(2 * 3 * 4, dtype=np.float32).reshape(1, 1, 2, 3, 4)
        labels = np.arange(2 * 3 * 4, dtype=np.uint32).reshape(1, 2, 3, 4)
        r = wb.run_step("croppad", {"z0": 0, "y0": 1, "x0": 2, "z": 0, "y": 0, "x": 0}, a, labels=labels)
        self.assertEqual(r.array.shape, (1, 1, 2, 2, 2))
        np.testing.assert_array_equal(r.array[0, 0], a[0, 0, :, 1:, 2:])
        np.testing.assert_array_equal(r.labels[0], labels[0, :, 1:, 2:])
        r = wb.run_step("croppad", {"z0": 5, "y0": 0, "x0": 0, "z": 2, "y": 0, "x": 0, "fill": 3.0}, a)
        self.assertEqual(r.array.shape, (1, 1, 2, 3, 4))
        np.testing.assert_allclose(r.array, 3.0)   # no overlap: all fill

    @unittest.skipIf(tifffile is None, "tifffile not installed")
    def test_flatfield(self):
        with tempfile.TemporaryDirectory() as d:
            flat = np.full((4, 4), 2.0, np.float32)
            flat[:, :2] = 4.0
            dark = np.ones((4, 4), np.float32)
            fp, dp = os.path.join(d, "flat.tif"), os.path.join(d, "dark.tif")
            tifffile.imwrite(fp, flat)
            tifffile.imwrite(dp, dark)
            a = np.full((1, 1, 2, 4, 4), 5.0, np.float32)
            r = wb.run_step("flatfield", {"flat": fp, "dark": dp}, a)
            gain = flat - dark            # 3 | 1, mean 2
            expected = (5.0 - 1.0) * 2.0 / gain
            np.testing.assert_allclose(r.array[0, 0, 1], expected, rtol=1e-6)
            with self.assertRaises(ValueError):
                wb.run_step("flatfield", {}, a)

    @unittest.skipUnless(_HAVE_SCIPY, "connected components need scipy")
    def test_channel_by_name(self):
        a = np.zeros((2, 1, 2, 4, 4), np.float32)
        a[1] = 3.0
        meta = {"channels": [{"label": "DAPI", "wavelength_nm": 405}, {"label": "GFP", "wavelength_nm": 488}]}
        r = wb.run_step("threshold", {"channel": "488", "method": "Manual", "value": 1.0, "min_voxels": 0}, a, meta)
        self.assertEqual(r.info["channel"], 1)
        self.assertEqual(int(r.labels.max()), 1)


if __name__ == "__main__":
    unittest.main()
