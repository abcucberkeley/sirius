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
    import tifffile  # type: ignore  # writes the test files only; the loader reads with sirius
except ImportError:  # pragma: no cover - environment dependent
    tifffile = None


def _tiff_reader():
    """The compiled sirius package, which reads every TIFF the workbench
    loads (there is no other reader), or None."""
    try:
        import sirius  # type: ignore

        sirius.inspect_tiff  # noqa: B018 - the extension, not a namespace package
        return sirius
    except Exception:  # noqa: BLE001
        return None


_NO_TIFF_READER = "reading TIFF needs the compiled sirius package (build the Python bindings)"


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


# The classic segmentation step reduced to one global cut and its instances:
# no top-hat, no blur, no opening, no hole filling, connected components.
_PLAIN_CUT = {"tophat": 0, "sigma": 0.0, "opening": 0, "fill_holes": False, "post": "Connected components"}


@unittest.skipIf(tifffile is None, "tifffile not installed")
@unittest.skipIf(_tiff_reader() is None, _NO_TIFF_READER)
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
        # segment -> resample: the labels cover the old grid, so the
        # application drops them (executor.cpp, labelsFit); a crop after the
        # resample used to cut the stale 64 x 64 labels as though they fitted
        img = np.zeros((4, 64, 64), np.float32)
        img[:, 10:20, 10:20] = 100.0
        img[:, 40:50, 40:55] = 100.0
        path = os.path.join(self.tmp.name, "grid.tif")
        tifffile.imwrite(path, img, photometric="minisblack")   # 4 leading planes are not RGBA
        steps = [{"kind": "load", "params": {}},
                 {"kind": "classic", "params": dict(_PLAIN_CUT, method="Manual", value=50.0, min_voxels=1)},
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

    def test_a_removed_step_fails_by_name_or_is_dropped_with_a_warning(self):
        for kind in ("threshold", "skimage_seg"):
            steps = [{"kind": "load", "params": {}}, {"kind": "contrast", "params": {}},
                     {"kind": kind, "params": {"method": "Otsu"}}]
            with self.assertRaises(wb.RemovedStep) as caught:
                wb.run_pipeline(self.path, {"steps": steps})
            self.assertIn(f"step 03 '{kind}' was removed from SIRIUS (2026-10)", str(caught.exception))
            self.assertIn("classic segmentation step", str(caught.exception))
            # a disabled one too: the application refuses the file either way
            steps[2]["enabled"] = False
            with self.assertRaises(wb.RemovedStep):
                wb.run_pipeline(self.path, {"steps": steps})
            with self.assertWarns(UserWarning) as warned:
                out, meta = wb.run_pipeline(self.path, {"steps": steps}, strict=False)
            self.assertIn("was removed from SIRIUS", str(warned.warning))
            self.assertEqual(meta["skipped"], [kind])
            self.assertNotIn("labels", meta)
            with self.assertRaises(wb.RemovedStep):
                wb.run_step(kind, {}, np.zeros((1, 1, 1, 4, 4), np.float32))
            with self.assertRaises(KeyError):
                wb.step_spec(kind)
            self.assertNotIn(kind, wb.step_kinds())

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


@unittest.skipIf(tifffile is None, "tifffile not installed")
@unittest.skipIf(_tiff_reader() is None, _NO_TIFF_READER)
class TestTiffLoader(unittest.TestCase):
    """Files as ImageJ and tifffile's OME writer make them, loaded as the
    application loads them. The expected values were read with the
    application's Load step (bindings/tests/test_parity.py compares the files
    the C++ fixture writer makes; these cover what that writer cannot
    produce: ImageJ's "none" resolution unit, tifffile's OME-XML)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _write(self, name, data, **kw):
        path = os.path.join(self.tmp.name, name)
        tifffile.imwrite(path, data, **kw)
        return path

    def test_imagej_files_with_three_axes_or_fewer_keep_their_axes(self):
        cyx = np.arange(2 * 16 * 12, dtype=np.uint16).reshape(2, 16, 12)
        a, meta = wb.load_dataset(self._write("cyx.tif", cyx, imagej=True, metadata={"axes": "CYX"}))
        self.assertEqual(a.shape, (2, 1, 1, 16, 12))   # was (1, 1, 2, ...): every page a z plane
        np.testing.assert_array_equal(a[1, 0, 0], cyx[1])
        self.assertTrue(meta["dims_from_metadata"])
        self.assertEqual([ch["color"] for ch in meta["channels"]], ["#63e08a", "#e871d9"])   # the palette, not white
        a, _ = wb.load_dataset(self._write("tyx.tif", cyx[:, :, :].repeat(2, axis=0)[:3], imagej=True,
                                           metadata={"axes": "TYX", "finterval": 2.0}))
        self.assertEqual(a.shape, (1, 3, 1, 16, 12))

    def test_imagej_length_units(self):
        zyx = np.zeros((5, 16, 12), np.uint16)
        # ImageJ writes the resolution unit "none" and the unit in its description
        _, meta = wb.load_dataset(self._write("nm.tif", zyx, imagej=True, resolution=(1 / 65, 1 / 65),
                                              metadata={"axes": "ZYX", "spacing": 300, "unit": "nm"}))
        self.assertEqual(meta["voxel_um"], [0.06499999993946404, 0.06499999993946404, 0.3])   # was 65 x 65 x 300
        _, meta = wb.load_dataset(self._write("mm.tif", zyx[:4], imagej=True, resolution=(5000.0, 5000.0),
                                              metadata={"axes": "ZYX", "spacing": 0.001, "unit": "mm"}))
        self.assertEqual(meta["voxel_um"], [0.2, 0.2, 1.0])
        # a resolution tag in centimetres wins over the description's unit;
        # without a spacing, z is twice x
        _, meta = wb.load_dataset(self._write("cm.tif", zyx[:4].astype(np.float32), imagej=True,
                                              resolution=(1e4 / 0.2, 1e4 / 0.2), resolutionunit="CENTIMETER",
                                              metadata={"axes": "ZYX", "unit": "nm"}))
        self.assertEqual(meta["voxel_um"], [0.2, 0.2, 0.4])

    def test_ome_units_order_and_channels(self):
        data = np.arange(2 * 3 * 4 * 16 * 12, dtype=np.uint16).reshape(2, 3, 4, 16, 12)   # t, c, z, y, x
        path = self._write("tczyx.ome.tif", data, metadata={
            "axes": "TCZYX", "PhysicalSizeX": 0.1, "PhysicalSizeY": 0.12, "TimeIncrement": 1500, "TimeIncrementUnit": "ms",
            "Channel": {"Name": ["DAPI", "GFP", "RFP"], "Color": [-16776961, 16711935, -1]}})
        a, meta = wb.load_dataset(path)
        self.assertEqual(a.shape, (3, 2, 4, 16, 12))
        np.testing.assert_array_equal(a[2, 1, 3], data[1, 2, 3])
        self.assertEqual(meta["name"], "tczyx")
        self.assertEqual(meta["voxel_um"], [0.1, 0.12, 0.2])
        self.assertEqual(meta["frame_interval_s"], 1.5)
        self.assertEqual([(ch["label"], ch["color"]) for ch in meta["channels"]],
                         [("DAPI", "#ff0000"), ("GFP", "#00ff00"), ("RFP", "#7c9cff")])   # white -> the palette
        path = self._write("czyx.ome.tif", data[0, :, :2], metadata={
            "axes": "CZYX", "PhysicalSizeX": 65, "PhysicalSizeXUnit": "nm", "PhysicalSizeY": 65, "PhysicalSizeYUnit": "nm",
            "Channel": {"EmissionWavelength": [450.0, 0.52, 640.0], "EmissionWavelengthUnit": ["nm", "\u00b5m", "nm"]}})
        _, meta = wb.load_dataset(path)
        self.assertEqual(meta["voxel_um"], [0.065, 0.065, 0.13])
        self.assertEqual([(ch["wavelength_nm"], ch["color"]) for ch in meta["channels"]],
                         [(450.0, "#6ec1c0"), (520.0, "#9dafad"), (640.0, "#ff7a5c")])

    def test_explicit_counts_follow_the_application(self):
        pages = np.arange(12 * 4 * 3, dtype=np.float32).reshape(12, 4, 3)
        path = self._write("plain.tif", pages, photometric="minisblack", metadata=None)
        self.assertEqual(wb.load_dataset(path, c=3)[0].shape, (3, 1, 4, 4, 3))
        # 4 planes do not divide 12 pages without a channel count: pages as z
        # (this read c3 z4 before)
        self.assertEqual(wb.load_dataset(path, z=4)[0].shape, (1, 1, 12, 4, 3))
        a, _ = wb.load_dataset(path, "zct", c=2)
        self.assertEqual(a.shape, (2, 1, 6, 4, 3))
        np.testing.assert_array_equal(a[1, 0, 0], pages[6])   # z fastest: channel 1 starts at page 6
        # an ImageJ file with an explicit count keeps the file's other axes
        ij = self._write("ij.tif", pages.reshape(2, 3, 2, 4, 3), imagej=True, metadata={"axes": "TZCYX"})
        self.assertEqual(wb.load_dataset(ij, c=3)[0].shape, (3, 2, 2, 4, 3))

    def test_colour_tiffs_open_with_their_samples_as_channels_like_the_application(self):
        rgb = np.arange(2 * 8 * 6 * 3, dtype=np.uint8).reshape(2, 8, 6, 3)
        a, meta = wb.load_dataset(self._write("rgb.tif", rgb, photometric="rgb"))
        self.assertEqual(a.shape, (3, 1, 2, 8, 6))
        np.testing.assert_array_equal(a[2, 0, 1], rgb[1, :, :, 2])
        self.assertTrue(meta["rgb"])
        self.assertEqual([ch["label"] for ch in meta["channels"]], ["R", "G", "B"])
        # separate planes read the same
        planar = self._write("planar.tif", np.moveaxis(rgb, -1, 1), photometric="rgb", planarconfig="separate")
        np.testing.assert_array_equal(wb.load_dataset(planar)[0], a)
        # an OME-TIFF's SizeC counts the samples: 2 z of one RGB channel
        ome = self._write("rgb.ome.tif", rgb, photometric="rgb", metadata={"axes": "ZYXS"})
        b, meta = wb.load_dataset(ome)
        self.assertEqual(b.shape, (3, 1, 2, 8, 6))
        np.testing.assert_array_equal(b, a)
        self.assertTrue(meta["dims_from_metadata"])
        # RGBA: four channels, not an RGB merge
        rgba = self._write("rgba.tif", np.concatenate([rgb, rgb[..., :1]], axis=-1), photometric="rgb",
                           extrasamples=["unassalpha"])
        c, meta = wb.load_dataset(rgba)
        self.assertEqual(c.shape, (4, 1, 2, 8, 6))
        self.assertFalse(meta["rgb"])

    def test_tiff_metadata_is_the_extension_parser_and_matches_the_python_port(self):
        xml = ('<OME><Image><Pixels DimensionOrder="XYZCT" SizeX="4" SizeY="4" SizeZ="3" SizeC="2" SizeT="5" '
               'PhysicalSizeX="65" PhysicalSizeXUnit="nm" PhysicalSizeY="0.065" PhysicalSizeZ="0.3" '
               'TimeIncrement="250" TimeIncrementUnit="ms"><Channel Name="a" EmissionWavelength="520" '
               'Color="-16776961"/><Channel Name="b"/></Pixels></Image></OME>')
        ij = "ImageJ=1.54f\nimages=30\nchannels=2\nslices=3\nframes=5\nunit=nm\nspacing=300\nfinterval=0.5\n"
        for text in (xml, ij, "", "nothing"):
            self.assertEqual(wb.tiff_metadata(text), wb._parse_tiff_description(text))


try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TestModelCache(unittest.TestCase):
    """load_model serves a model from memory only while its file is the one
    it read: a re-export under the same name is another model."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "model.pt")

    def _export(self, path, k, mtime_ns):
        class Scale(torch.nn.Module):
            def __init__(self, k):
                super().__init__()
                self.k = k

            def forward(self, x):
                return x * self.k

        torch.jit.script(Scale(k)).save(path)
        os.utime(path, ns=(mtime_ns, mtime_ns))   # distinct stamps however coarse the file system clock

    def _value(self, model):
        with torch.no_grad():
            return float(model(torch.ones(1))[0])

    def test_a_model_rewritten_in_place_is_read_again(self):
        self._export(self.path, 1.0, 1_000_000_000)
        first = wb.load_model(self.path, "cpu")
        self.assertEqual(self._value(first), 1.0)
        self.assertIs(wb.load_model(self.path, "cpu"), first)   # unchanged: served from memory
        self._export(self.path, 2.0, 2_000_000_000)
        second = wb.load_model(self.path, "cpu")
        self.assertEqual(self._value(second), 2.0)
        self.assertIsNot(second, first)

    def test_the_cache_is_bounded(self):
        paths = [os.path.join(self.tmp.name, f"m{i}.pt") for i in range(wb._MODEL_CACHE_SIZE + 2)]
        for i, p in enumerate(paths):
            self._export(p, float(i), 1_000_000_000 + i)
            wb.load_model(p, "cpu")
        self.assertLessEqual(len(wb._model_cache), wb._MODEL_CACHE_SIZE)
        self.assertNotIn((os.path.abspath(paths[0]), "cpu"), wb._model_cache)   # the least recently used went
        self.assertIn((os.path.abspath(paths[-1]), "cpu"), wb._model_cache)


@unittest.skipIf(torch is None, "torch not installed")
class TestTiledInference(unittest.TestCase):
    def test_a_constant_model_stays_constant_up_to_the_volume_corners(self):
        # the blend window tapered the tile faces on the volume's border too,
        # so a 3-D corner covered by one tile had a weight of ~4e-8, divided
        # by a floor of 1e-6: with the application's overlap (z 4, y / x 32)
        # 0.9 came out as 0.034 there
        class Constant(torch.nn.Module):
            def forward(self, x):
                return torch.ones_like(x) * 0.9

        model = torch.jit.script(Constant())
        volume = np.random.default_rng(0).random((10, 100, 100), dtype=np.float32)
        prob = wb.tiled_inference(volume, model, (8, 64, 64), (4, 32, 32), "cpu")
        self.assertEqual(prob.shape, (1, 10, 100, 100))
        np.testing.assert_allclose(prob, 0.9, atol=1e-5)
        # one tile bigger than the volume on every axis: no neighbour anywhere
        small = wb.tiled_inference(volume[:4, :20, :20], model, (8, 64, 64), (4, 32, 32), "cpu")
        np.testing.assert_allclose(small, 0.9, atol=1e-5)

    def test_overlapping_tiles_still_cross_fade(self):
        # an identity model through overlapping tiles reproduces the input
        class Identity(torch.nn.Module):
            def forward(self, x):
                return x

        model = torch.jit.script(Identity())
        volume = np.random.default_rng(1).random((12, 50, 50), dtype=np.float32)
        prob = wb.tiled_inference(volume, model, (6, 24, 24), (2, 6, 6), "cpu", normalize=False)
        np.testing.assert_allclose(prob[0], volume, atol=1e-5)


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

    def test_sim_from_file_reports_the_toml_error(self):
        # a TOML file that fails to load was retried as a legacy config, which
        # hid the real error behind "Unknown legacy config key"
        meta = {"voxel_um": [0.08, 0.08, 0.125]}
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.toml"
            bad.write_text("[optics]\nndirs = 0\n")
            with self.assertRaises(Exception) as cm:
                wb._sim_parameters({"mode": "From file", "params_file": str(bad)}, meta)
            self.assertIn("ndirs", str(cm.exception))
            self.assertNotIn("legacy", str(cm.exception))
            # a TOML file without the extension is read as TOML, as the application does
            plain = Path(d) / "sim2d.cfg"
            plain.write_text("# 2D SIM\n[optics]\nndirs = 3\nnphases = 3\n")
            p = wb._sim_parameters({"mode": "From file", "params_file": str(plain)}, meta)
            self.assertEqual(p.nphases, 3)
            self.assertEqual(p.norders, 0)   # derived: 2 orders for 3 phases
        # and a legacy config still loads as one
        p = wb._sim_parameters({"mode": "From file", "params_file": str(self.DATA / "config.txt")}, meta)
        self.assertEqual(p.nphases, 5)

    def test_sim_step_reconstructs_with_the_theoretical_otf(self):
        # An empty `otf` means the theoretical OTF here as it does in the GUI
        # and the CLI, and the choice is the library's one selectOTF
        # (sirius/otf_select.hpp). Until 2026-10-08 this raised NotAvailable,
        # which is what made every default-OTF pipeline the application
        # exports with export_python die on its own run_pipeline
        # (docs/findings.md 9k.50, finding 4).
        #
        # Two arms, one for each C++ case that covers the ideal OTF, asserting
        # what each of those asserts:
        #   * the step's own defaults, as test_app_ops.cpp's "the theoretical
        #     OTF works without a file" has it -- the output grid and that it
        #     ran, and no pattern (see the note below);
        #   * the reference config's parameters, as test_reconstruction.cpp's
        #     "Reconstruction with the ideal OTF resembles the reference" has
        #     it -- there the pattern IS pinned and the result has to correlate
        #     above 0.8 with cudasirecon's measured-OTF reconstruction.
        sirius = _sirius_extension()
        raw = sirius.read_tiff(str(self.DATA / "raw.tif"), dtype=np.float32)
        self.assertEqual(raw.shape, (135, 64, 64))   # 3 angles x 5 phases x 9 z
        meta = {"voxel_um": [0.08, 0.08, 0.125]}

        estimate = {"na": 1.42, "nimm": 1.515, "linespacing_um": 0.2035, "k0_start_angle": 46.08}
        r = wb.run_step("sim", estimate, raw, meta, device="cpu")
        self.assertEqual(r.array.shape, (1, 1, 9, 128, 128))   # zoomfact 2, z_zoom 1
        self.assertEqual(int(np.count_nonzero(~np.isfinite(r.array))), 0)
        self.assertGreater(float(np.max(np.abs(r.array))), 0.0)
        self.assertAlmostEqual(r.meta["voxel_um"][0], 0.04, places=9)
        self.assertEqual(len(r.info["fits"]), 1)
        self.assertEqual(len(r.info["fits"][0]["k0"]), 3)
        # No band on the Estimate-mode pattern, and not for want of trying.
        # Measured on this stack (fiona job 4247382): through the theoretical
        # OTF, directions 0 and 1 fit 0.4076 um -- the configured
        # 1 / 0.2035 / 2 = 0.407 -- and direction 2 lands at 0.5375 um
        # (1.861 /um, angle 172.9 deg); through otf.tif, from the same seeds,
        # all three fit 0.406 - 0.408. So the Estimate seed is not enough for
        # direction 2 of this stack through the theoretical OTF, and a band
        # here would be a band around a fit that is wrong. It is not something
        # this commit changed: the OTF the C++ builds is the same table as
        # before, value for value (tests/test_otf_select.cpp), and job 4247383
        # gets the same 0.5375 out of the unpatched build.

        ref = {"mode": "From file", "params_file": str(self.DATA / "config.txt")}
        q = wb.run_step("sim", ref, raw, meta, device="cpu")
        self.assertEqual(q.array.shape, (1, 1, 9, 128, 128))
        self.assertEqual(int(np.count_nonzero(~np.isfinite(q.array))), 0)
        for kx, ky in q.info["fits"][0]["k0"]:
            # 0.4075, 0.4075, 0.4076 measured; the band is half a per cent, so
            # a plan-rigor difference cannot move it but a collapsed fit
            # (k0 = 0, "spacing=inf um", 9k.49) or a wrong minimum must
            self.assertAlmostEqual(1.0 / float(np.hypot(kx, ky)), 0.407, delta=0.002)
        expected = sirius.read_tiff(str(self.DATA / "raw_proc.tif"), dtype=np.float32)
        corr = float(np.corrcoef(q.array[0, 0].ravel(), expected.ravel())[0, 1])
        self.assertGreater(corr, 0.8)   # 0.8169 measured; the C++ case's own threshold

    def test_the_theoretical_otf_follows_the_stacks_planes(self):
        # The theoretical OTF is built in 3D for a stack of several planes and
        # in 2D (one kz sample, no missing cone) for a single plane, and the
        # application decides from the stack: session.cpp's threeD(), which is
        # SIMParameters::planes(sections) > 1. The mirror asks the same
        # parameters the same question rather than doing the arithmetic itself.
        sirius = _sirius_extension()
        meta = {"voxel_um": [0.08, 0.08, 0.125]}
        params = {"na": 1.42, "nimm": 1.515, "linespacing_um": 0.2035, "k0_start_angle": 46.08}
        p = wb._sim_parameters(params, meta)
        self.assertEqual(p.sections_per_plane(), 15)      # 3 angles x 5 phases
        self.assertEqual(p.planes(135), 9)                # tests/data/raw.tif: 3D
        self.assertEqual(p.planes(15), 1)                 # one plane: 2D
        self.assertEqual(p.planes(134), 0)                # not a whole number of planes
        # both theoretical tables are reachable with no OTF file at all
        for three_d in (True, False):
            sirius.SimReconstructor(p, "", sirius.Device.cpu(), sirius.PlanRigor.Estimate,
                                    three_d=three_d)
        # and three_d is in the reconstructor cache key, as threeD is in the
        # C++ one (session.cpp's Impl::Cache): without it a 2D stack following
        # a 3D one would be reconstructed with the 3D stack's OTF
        raw = sirius.read_tiff(str(self.DATA / "raw.tif"), dtype=np.float32)
        wb.run_step("sim", params, raw, meta, device="cpu")
        self.assertEqual([json.loads(k)["three_d"] for k in wb._sim_cache], [True])
        # a section count that is not a whole number of planes still names the
        # arithmetic, in the application's words
        with self.assertRaises(ValueError) as cm:
            wb.run_step("sim", params, np.zeros((134, 8, 8), np.float32), meta, device="cpu")
        self.assertIn("not a multiple of angle 3 \u00d7 phase 5 = 15", str(cm.exception))
        # a named OTF file that is not there is still an error, not a silent
        # fall back to the theoretical OTF
        with self.assertRaises(FileNotFoundError):
            wb.run_step("sim", dict(params, otf=str(self.DATA / "no-such-otf.tif")),
                        np.zeros((15, 8, 8), np.float32), meta, device="cpu")

    def test_odd_lateral_sizes_reconstruct_and_the_shape_messages_are_the_fronts(self):
        # Until 2026-10-08 three C++ guards refused an odd nx or ny and worded
        # that one condition three ways, with nothing at all in Python
        # (docs/findings.md 9k.50, finding 5). The guards are gone and both
        # sentences now come from the library, so the message Python raises is
        # the message the GUI shows and the CLI returns -- there is no Python
        # copy of the wording to drift. This is the mirror's half of
        # tests/test_sim_parameters.cpp's "the shape conditions are one
        # wording" and tests/test_reconstruction.cpp's refusal case; the
        # known-answer odd reconstruction is the C++ one
        # (tests/test_reconstruction.cpp, a synthetic pattern on 127 x 97),
        # since there is no odd-size reference output anywhere.
        sirius = _sirius_extension()
        meta = {"voxel_um": [0.08, 0.08, 0.125]}
        params = {"mode": "From file", "params_file": str(self.DATA / "config.txt"),
                  "otf": str(self.DATA / "otf.tif")}
        # an odd, square-but-odd lateral extent of the bundled acquisition --
        # a shape case, not a measurement: 63 x 63 of its 64 x 64 frames
        raw = sirius.read_tiff(str(self.DATA / "raw.tif"), dtype=np.float32)
        self.assertEqual(raw.shape[-3:], (135, 64, 64))
        odd = np.ascontiguousarray(raw[..., :63, :63])
        r = wb.run_step("sim", params, odd, meta, device="cpu")
        self.assertEqual(r.array.shape, (1, 1, 9, 126, 126))   # zoomfact 2 of 63 x 63
        self.assertEqual(int(np.count_nonzero(~np.isfinite(r.array))), 0)
        self.assertEqual(len(r.info["fits"][0]["k0"]), 3)

        # The one wording, straight from the library, reached through the
        # PACKAGE and not the extension module -- sirius/__init__.py has to
        # re-export it or `sirius.sim_image_size_problem` is an AttributeError
        # for every caller, which is how job 4247406 found it missing.
        self.assertIn("sim_image_size_problem", sirius.__all__)
        self.assertEqual(sirius.sim_image_size_problem(281, 241), "")   # the isoar stack
        self.assertEqual(sirius.sim_image_size_problem(5, 7), "")
        self.assertEqual(sirius.sim_image_size_problem(4, 4), "")
        self.assertEqual(sirius.sim_image_size_problem(3, 8),
                         "Image size must be at least 4 \u00d7 4, got 3 \u00d7 8.")
        self.assertEqual(sirius.sim_image_size_problem(3, 8, True),
                         "Each tile must be at least 4 \u00d7 4, got 3 \u00d7 8.")
        # and the mirror raises exactly it, as a ValueError, for a stack the
        # reconstructor cannot bind
        with self.assertRaises(ValueError) as cm:
            wb.run_step("sim", params, np.zeros((15, 3, 8), np.float32), meta, device="cpu")
        self.assertEqual(str(cm.exception), sirius.sim_image_size_problem(8, 3))
        # the section-count sentence likewise comes from the parameters
        p = wb._sim_parameters(params, meta)
        self.assertEqual(p.section_count_problem(135), "")
        self.assertEqual(p.section_count_problem(134),
                         "z holds 134 sections, not a multiple of angle 3 \u00d7 phase 5 = 15.")
        with self.assertRaises(ValueError) as cm:
            wb.run_step("sim", params, np.zeros((134, 8, 8), np.float32), meta, device="cpu")
        self.assertEqual(str(cm.exception), p.section_count_problem(134))

    def test_an_exported_default_sim_pipeline_runs_through_run_pipeline(self):
        # The application's export_python writes a script whose whole body is
        # run_pipeline(DATASET, PIPELINE) (app/core/pipeline.cpp's
        # toPythonScript), so a pipeline the application can export has to be
        # one this module can run. PIPELINE here is what sirius-cli's
        # export_python actually wrote for [Load, SIM] on tests/data/raw.tif
        # with the OTF field left empty (job 4247377 on dev f8e8211), with
        # only the dataset path substituted. Run then, it died at step 2 with
        # NotAvailable: "SIM reconstruction in Python needs a measured OTF
        # file ('otf')".
        #
        # The backend is pinned because the three fronts default to three
        # different ones (9k.50, finding 3): run_pipeline's own default is
        # "auto", which is CUDA wherever torch sees a device.
        path = str(self.DATA / "raw.tif")
        pipeline = {
            "version": 1,
            "steps": [
                {"cache": "recompute", "enabled": True, "id": 1, "kind": "load", "name": "Load",
                 "params": {"c": 0, "page_order": "czt", "path": path,
                            "read_as": "Lazy (chunk on demand)", "sheet_angle": 0.0,
                            "sim_fast": False, "sim_layout": "", "sim_ndirs": 0, "sim_nphases": 0,
                            "t": 0, "tile": 0, "voxel_x": 0.08, "voxel_y": 0.08, "voxel_z": 0.125,
                            "z": 0}},
                {"cache": "disk", "enabled": True, "id": 3, "kind": "sim", "name": "SIM",
                 "params": {"angles": 3, "apodization": "Cosine", "apodize_input": "Triangle",
                            "background": 0.0, "bleach_correction": True, "dz_psf": 0.0,
                            "equalizez": False, "explodefact": 1.0, "filter_overlaps": True,
                            "k0_angles": [], "k0_start_angle": 46.08, "linespacing_um": 0.2035,
                            "mode": "Estimate", "na": 1.42, "napodize": 10, "nimm": 1.515,
                            "no_kz0": True, "orders": 0, "otf": "", "otfcutoff": 0.006,
                            "params_file": "", "phases": 5, "suppress_singularities": True,
                            "suppress_zero_order": True, "suppression_radius": 10,
                            "wavelength_nm": 510.0, "wiener": 0.001, "z_zoom": 1,
                            "zoomfact": 2.0}},
            ],
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error", wb.UnknownParameterWarning)
            out, meta = wb.run_pipeline(path, pipeline, device="cpu")
        self.assertEqual(out.shape, (1, 1, 9, 128, 128))
        self.assertNotIn("skipped", meta)
        self.assertAlmostEqual(meta["voxel_um"][0], 0.04, places=9)
        self.assertAlmostEqual(meta["voxel_um"][2], 0.125, places=9)

    def test_sim_from_file_reconstructs_with_the_files_pixel_sizes(self):
        # tests/test_app_ops.cpp "SIM From file reconstructs with the file's
        # pixel sizes": a cudasirecon config's xyres / zres are what it
        # reconstructs a TIFF stack with, and a measured OTF's radial step is
        # derived from xyres, so the dataset's calibration (none for a plain
        # TIFF, or mistaken: on 2026-10-08 sirius-cli was handed raw.tif's
        # voxel in z, y, x order) must not stretch the OTF. The file's pixel
        # sizes win where it sets them, the stack's fill in the rest. The
        # worker runs every node-side SIM step through this module, so the
        # fix in sim.cpp alone left the node route reconstructing with
        # dx = 0.125 and "the overlap of orders 0 and 2 holds no signal".
        sirius = _sirius_extension()
        wrong = {"voxel_um": [0.125, 0.08, 0.08]}
        params = {"mode": "From file", "params_file": str(self.DATA / "config.txt"), "otf": str(self.DATA / "otf.tif")}
        p = wb._sim_parameters(params, wrong)   # xyres=0.08 zres=0.125 zresPSF=0.125
        self.assertAlmostEqual(p.dx, 0.08, places=9)
        self.assertAlmostEqual(p.dy, 0.08, places=9)
        self.assertAlmostEqual(p.dz, 0.125, places=9)
        self.assertAlmostEqual(p.dz_psf, 0.125, places=9)
        with tempfile.TemporaryDirectory() as d:
            # a file without pixel sizes takes the stack's
            bare = Path(d) / "bare.txt"
            bare.write_text("nphases=5\nndirs=3\nna=1.42\nnimm=1.515\nls=0.2035\n")
            q = wb._sim_parameters({"mode": "From file", "params_file": str(bare)}, wrong)
            self.assertAlmostEqual(q.dx, 0.125, places=9)
            self.assertAlmostEqual(q.dy, 0.08, places=9)
            self.assertAlmostEqual(q.dz, 0.08, places=9)
            self.assertAlmostEqual(q.dz_psf, 0.08, places=9)
            # a TOML file's pixels table is the same contract
            toml = Path(d) / "pixels.toml"
            toml.write_text("pixels = { dx = 0.07, dy = 0.07, dz = 0.15 }\n"
                            "[optics]\nndirs = 3\nnphases = 5\nna = 1.42\nnimm = 1.515\nlinespacing_um = 0.2035\n")
            q = wb._sim_parameters({"mode": "From file", "params_file": str(toml)}, wrong)
            self.assertAlmostEqual(q.dx, 0.07, places=9)
            self.assertAlmostEqual(q.dy, 0.07, places=9)
            self.assertAlmostEqual(q.dz, 0.15, places=9)
            self.assertAlmostEqual(q.dz_psf, 0.08, places=9)   # not in the file: the stack's dz
        # Estimate mode has no file: the stack's
        q = wb._sim_parameters({"mode": "Estimate"}, wrong)
        self.assertAlmostEqual(q.dx, 0.125, places=9)
        self.assertAlmostEqual(q.dy, 0.08, places=9)
        self.assertAlmostEqual(q.dz, 0.08, places=9)
        # and the reconstruction is cudasirecon's own (raw_proc.tif), as it
        # is from a correctly opened dataset
        raw = sirius.read_tiff(str(self.DATA / "raw.tif"), dtype=np.float32)
        expected = sirius.read_tiff(str(self.DATA / "raw_proc.tif"), dtype=np.float32)
        r = wb.run_step("sim", params, raw, wrong, device="cpu")
        self.assertEqual(r.array.shape, (1, 1) + expected.shape)
        rel = np.max(np.abs(r.array[0, 0] - expected)) / np.max(np.abs(expected))
        self.assertLess(rel, 1e-3)
        # the output voxel is the reconstruction's pixel, not the dataset's claim
        self.assertAlmostEqual(r.meta["voxel_um"][0], 0.04, places=9)
        self.assertAlmostEqual(r.meta["voxel_um"][1], 0.04, places=9)
        self.assertAlmostEqual(r.meta["voxel_um"][2], 0.125, places=9)
        right = wb.run_step("sim", params, raw, {"voxel_um": [0.08, 0.08, 0.125]}, device="cpu")
        np.testing.assert_array_equal(right.array, r.array)

    def test_sim_from_file_keeps_the_files_otf_axial_step_unless_the_field_is_set(self):
        # tests/test_app_ops.cpp "SIM From file keeps the file's OTF axial
        # step unless the field is set": 0 in dz_psf means "not set"
        meta = {"voxel_um": [0.1, 0.1, 0.2]}

        def dz_psf(path, **extra):
            return wb._sim_parameters(dict({"mode": "From file", "params_file": str(path)}, **extra), meta).dz_psf

        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "dzpsf.txt"
            cfg.write_text("nphases=5\nndirs=3\nna=1.2\nnimm=1.33\nxyres=0.1\nzres=0.2\nzresPSF=0.5\nls=0.2\n")
            self.assertAlmostEqual(dz_psf(cfg), 0.5, places=9)
            self.assertAlmostEqual(dz_psf(cfg, dz_psf=0.3), 0.3, places=9)
            bare = Path(d) / "bare.txt"
            bare.write_text("nphases=5\nndirs=3\nna=1.2\nnimm=1.33\nxyres=0.1\nzres=0.2\nls=0.2\n")
            self.assertAlmostEqual(dz_psf(bare), 0.2, places=9)   # not in the file: the stack's dz
            # an inline table and a dotted key are both assignments the
            # loader reads; a scanner that only looks at the first '=' on a
            # line misses them and replaces the file's step with the stack dz
            inlined = Path(d) / "inline.toml"
            inlined.write_text("pixels = { dx = 0.1, dy = 0.1, dz = 0.2, dz_psf = 0.55 }\n"
                               "[optics]\nndirs = 3\nnphases = 5\nna = 1.2\nnimm = 1.33\nlinespacing_um = 0.2\n")
            self.assertAlmostEqual(dz_psf(inlined), 0.55, places=6)
            dotted = Path(d) / "dotted.toml"
            dotted.write_text("pixels.dx = 0.1\npixels.dy = 0.1\npixels.dz = 0.2\npixels.dz_psf = 0.45\n"
                              "[optics]\nndirs = 3\nnphases = 5\nna = 1.2\nnimm = 1.33\nlinespacing_um = 0.2\n")
            self.assertAlmostEqual(dz_psf(dotted), 0.45, places=6)
        self.assertAlmostEqual(wb._sim_parameters({"mode": "Estimate"}, meta).dz_psf, 0.2, places=9)
        self.assertAlmostEqual(wb._sim_parameters({"mode": "Estimate", "dz_psf": 0.3}, meta).dz_psf, 0.3, places=9)


class TestParameters(unittest.TestCase):
    def test_unknown_keys_warn_and_are_ignored(self):
        a = np.ones((1, 2, 2, 4, 4), np.float32)
        with self.assertWarns(wb.UnknownParameterWarning) as cm:
            r = wb.run_step("bleach", {"mode": "Match mean", "to_the_moon": 1}, a)
        self.assertIn("bleach", str(cm.warning))
        self.assertIn("to_the_moon", str(cm.warning))
        self.assertEqual(r.array.shape, a.shape)

    def test_canonical_key_wins_over_an_alias(self):
        p = wb._prepare_params(wb.step_spec("classic"), {"minVoxels": 9, "min_voxels": 1, "method": "Manual"}, None)
        self.assertEqual(p["min_voxels"], 1)
        self.assertNotIn("minVoxels", p)

    def test_defaults_are_filled(self):
        p = wb._prepare_params(wb.step_spec("classic"), {}, None)
        self.assertEqual(p["method"], "Otsu")
        self.assertEqual(p["min_voxels"], 20)
        self.assertEqual(p["post"], "Watershed (distance)")

    def test_numbers_are_parsed_as_the_application_loads_them(self):
        # coerceToSpec: integers round half away from zero (llround, where
        # Python's round() goes to even) and every number is clamped to the
        # parameter's range
        self.assertEqual([wb._int({"v": v}, "v", 0) for v in (2.5, 3.5, -2.5, "0.5", 1.49)], [3, 4, -3, 1, 1])
        p = wb._prepare_params(wb.step_spec("classic"),
                               {"window": 1, "min_voxels": -4, "sigma": 75.0, "opening": 2.5}, None)
        self.assertEqual((p["window"], p["min_voxels"], p["sigma"], p["opening"]), (3, 0, 50.0, 3))
        p = wb._prepare_params(wb.step_spec("croppad"), {"z0": -1e9, "x": "12.5"}, None)
        self.assertEqual((p["z0"], p["x"]), (-100000, 13))

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

    def test_histograms_and_otsu_cuts_leave_out_infinities(self):
        # an infinite voxel made the C++ histogram write out of bounds (the
        # application crashed opening the dataset) and these raise ValueError;
        # both now work on the finite values, as tests/test_app_ops.cpp checks
        inf = np.float32(np.inf)
        np.testing.assert_array_equal(wb._histogram(np.array([-inf, 0, 0.5, 1, inf], np.float32), 4, 0.0, 1.0),
                                      [1, 0, 1, 1])
        for lo, hi in ((0.0, np.inf), (-np.inf, 1.0), (-np.inf, np.inf)):
            np.testing.assert_array_equal(wb._histogram(np.array([0.5, inf], np.float32), 30, lo, hi), np.zeros(30))
        rng = np.random.default_rng(3)
        v = np.concatenate([rng.normal(10, 1, 500), rng.normal(50, 1, 500), rng.normal(90, 1, 100)]).astype(np.float32)
        poked = np.concatenate([v, [inf, -inf, np.nan]]).astype(np.float32)
        self.assertEqual(wb._otsu_threshold(poked), wb._otsu_threshold(v))
        self.assertEqual(wb._multi_otsu_upper(poked), wb._multi_otsu_upper(v))
        self.assertTrue(10 < wb._otsu_threshold(v) < 90)
        # nothing finite: a cut nothing lies above, as the C++ returns
        none = np.array([inf, -inf, np.nan], np.float32)
        self.assertEqual(wb._otsu_threshold(none), np.inf)
        self.assertEqual(wb._multi_otsu_upper(none), np.inf)

    def test_otsu_cuts_break_ties_as_the_application(self):
        # Symmetric about its centre, each histogram scores a split and its
        # mirror image exactly the same; the float rounding of the between-class
        # variance picks one, so the mirror must evaluate it in the C++ order.
        # The same data and cuts as the "Otsu and Multi-Otsu break exact ties"
        # case in tests/test_app_ops.cpp (the old `** 2` gave 139 and 71).
        def symmetric(bins, ends, pairs):
            values = [0.0] * ends + [float(bins)] * ends
            for b, count in pairs:
                values += [b + 0.5, bins - 1 - b + 0.5] * count
            return np.array(values, np.float32)

        self.assertEqual(wb._otsu_threshold(symmetric(256, 3, ((36, 6), (117, 17)))), 37.0)
        self.assertEqual(wb._multi_otsu_upper(symmetric(128, 1, ((2, 2), (36, 2), (44, 6), (58, 5)))), 93.0)

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

    def test_max_and_min_keep_infinities_and_nan_only_runs(self):
        # reduceAxes semantics (tests/test_image_ops.cpp): NaN is skipped, a
        # real +-inf is kept, and a run with nothing but NaN stays NaN
        a = np.array([np.inf, 1, 2, np.nan, np.nan, np.nan, -np.inf, -np.inf, 0, 5, -np.inf, np.nan],
                     np.float32).reshape(2, 1, 1, 2, 3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # numpy's "All-NaN slice"
            mx = wb.run_step("einsum", {"keep": "cy", "reduction": "max"}, a).array.reshape(-1)
            mn = wb.run_step("einsum", {"keep": "cy", "reduction": "min"}, a).array.reshape(-1)
        np.testing.assert_array_equal(mx, [np.inf, np.nan, 0, 5])
        np.testing.assert_array_equal(mn, [1, np.nan, -np.inf, -np.inf])

    @unittest.skipUnless(_HAVE_SCIPY, "connected components need scipy")
    def test_otsu_cut_runs_on_an_infinite_voxel(self):
        a = np.zeros((1, 1, 1, 8, 8), np.float32)
        a[0, 0, 0, 2:5, 2:5] = 1.0
        a[0, 0, 0, 7, 7] = np.inf
        a[0, 0, 0, 0, 7] = -np.inf
        r = wb.run_step("classic", dict(_PLAIN_CUT, method="Otsu", min_voxels=0), a)
        self.assertEqual(int(r.labels.max()), 2)   # the square and the +inf voxel
        self.assertNotEqual(int(r.labels[0, 0, 7, 7]), 0)
        self.assertEqual(int(r.labels[0, 0, 0, 7]), 0)

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

    def test_merge_gives_uncoloured_channels_the_palette(self):
        # the application colours a multi-channel dataset without colours
        # 488 / 561 nm (green, magenta); the mirror used white for both, so an
        # exported Merge came out grey
        a = np.zeros((2, 1, 1, 4, 4), np.float32)
        a[0] = 0.5
        a[1] = 0.25
        r = wb.run_step("merge", {}, a)
        f32 = np.float32
        green = [f32(v) / f32(255) for v in (0x63, 0xE0, 0x8A)]
        magenta = [f32(v) / f32(255) for v in (0xE8, 0x71, 0xD9)]
        expected = [min(f32(1), green[k] * f32(0.5) + magenta[k] * f32(0.25)) for k in range(3)]
        self.assertEqual(r.array[:, 0, 0, 0, 0].tolist(), [float(v) for v in expected])
        # a single channel stays white, as there
        r = wb.run_step("merge", {}, a[:1])
        self.assertEqual(r.array[:, 0, 0, 0, 0].tolist(), [0.5, 0.5, 0.5])

    def test_merge_treats_nan_as_the_application_does(self):
        a = np.full((2, 1, 1, 1, 3), 0.5, np.float32)
        a[0, 0, 0, 0, 1] = np.nan
        meta = {"channels": [{"color": "#ff0000"}, {"color": "#00ff00"}]}
        # additive: std::min(1, r + NaN) is 1 -- the voxel turns white
        r = wb.run_step("merge", {"blend": "Additive"}, a, meta)
        self.assertEqual(r.array[:, 0, 0, 0, 1].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(r.array[:, 0, 0, 0, 0].tolist(), [0.5, 0.5, 0.0])
        # max: std::max(r, NaN) keeps r
        r = wb.run_step("merge", {"blend": "Max"}, a, meta)
        self.assertEqual(r.array[:, 0, 0, 0, 1].tolist(), [0.0, 0.5, 0.0])
        # screen passes the NaN on
        r = wb.run_step("merge", {"blend": "Screen"}, a, meta)
        self.assertTrue(np.isnan(r.array[:, 0, 0, 0, 1]).all())

    @unittest.skipUnless(_HAVE_SCIPY, "connected components need scipy")
    def test_global_cuts_and_min_voxels(self):
        a = np.zeros((1, 1, 4, 10, 10), np.float32)
        a[0, 0, :, 1:3, 1:3] = 5.0
        a[0, 0, :, 6:9, 6:9] = 7.0
        r = wb.run_step("classic", dict(_PLAIN_CUT, channel=0, method="Manual", value=1.0, min_voxels=0), a)
        self.assertIsNotNone(r.labels)
        self.assertEqual(r.labels.shape, (1, 4, 10, 10))
        self.assertEqual(int(r.labels.max()), 2)
        self.assertEqual(r.info["thresholds"], [1.0])
        r2 = wb.run_step("classic", dict(_PLAIN_CUT, channel=0, method="Manual", value=1.0, min_voxels=20), a)
        self.assertEqual(int(r2.labels.max()), 1)
        # the application's default cut is Otsu, which separates 0 from the objects
        r3 = wb.run_step("classic", dict(_PLAIN_CUT, channel=0, min_voxels=0), a)
        self.assertEqual(r3.info["method"], "Otsu")
        self.assertEqual(int(r3.labels.max()), 2)
        # percentiles are order statistics: the 90th of 400 voxels is 5.0, so
        # only the 7.0 block is strictly above it (the 99th would be 7.0, which
        # cuts everything away -- as it does in the application)
        r4 = wb.run_step("classic", dict(_PLAIN_CUT, channel=0, method="Percentile", percentile=90.0, min_voxels=0), a)
        self.assertEqual(int(r4.labels.max()), 1)
        self.assertEqual(r4.info["thresholds"], [5.0])

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

    @unittest.skipUnless(_HAVE_SCIPY, "watershed seeds need scipy")
    def test_watershed_splits_touching_blobs(self):
        # two overlapping disks: the distance transform has one maximum in each
        # and a saddle at the waist, so distanceSeeds accepts exactly two seeds
        y, x = np.mgrid[0:20, 0:40]
        blobs = ((y - 10) ** 2 + (x - 13) ** 2 <= 49) | ((y - 10) ** 2 + (x - 25) ** 2 <= 49)
        a = np.zeros((1, 1, 1, 20, 40), np.float32)
        a[0, 0, 0] = blobs
        p = dict(_PLAIN_CUT, method="Manual", value=0.5, seeds="Distance maxima", seed_distance=5.0, min_voxels=0)
        r = wb.run_step("classic", dict(p, post="Watershed (distance)"), a)
        self.assertEqual(int(r.labels.max()), 2)
        # they touch, so connected components sees one object
        r = wb.run_step("classic", dict(p, post="Connected components"), a)
        self.assertEqual(int(r.labels.max()), 1)

    def test_watershed_floods_in_the_application_order(self):
        # Equal heights leave the queue in the order they entered it (seeds in
        # raster order, neighbours -z, +z, -y, +y, -x, +x), as in labels.cpp:
        # seed 2 enters first and takes the low row before seed 1's turn at
        # the tied voxels below it. test_app_labels.cpp pins the C++ to the
        # same answer; scikit-image's flood gave the bottom row to seed 1.
        land = np.array([[[1, 0, 0], [1, 1, 1]]], np.float32)
        seeds = np.array([[[2, 0, 0], [1, 0, 0]]], np.uint32)
        out = wb._watershed(land, np.ones((1, 2, 3), bool), seeds)
        self.assertEqual(out.tolist(), [[[2, 2, 2], [1, 2, 2]]])
        # nothing leaves the mask, and a seed outside it is no seed
        mask = np.array([[[True, True, False], [False, True, True]]])
        out = wb._watershed(land, mask, seeds)
        self.assertEqual(out.tolist(), [[[2, 2, 0], [0, 2, 2]]])

    def test_expand_labels_passes_ties_on_in_the_application_order(self):
        # the case test_app_labels.cpp pins the C++ to (its heap used to give
        # (1, 0) .. (3, 1) another answer): (1, 1) is tied and stays
        # background, and what it passes on is the label that reached it first
        lab = np.array([[[0, 1, 0], [0, 0, 2], [0, 0, 0], [0, 0, 0]]], np.uint32)
        grown = wb._expand_labels(lab, 4.0, 3.0)
        self.assertEqual(grown.tolist(), [[[1, 1, 0], [1, 0, 2], [1, 0, 2], [1, 0, 2]]])

    @unittest.skipUnless(_HAVE_SCIPY, "label post-processing needs scipy")
    def test_watershed_keeps_a_component_no_seed_reached(self):
        # two 7 x 7 squares two pixels apart: with seed_distance 10 only the
        # first gets a seed, and the second used to vanish from the labels
        a = np.zeros((1, 1, 1, 12, 20), np.float32)
        a[0, 0, 0, 2:9, 2:9] = 1.0
        a[0, 0, 0, 2:9, 11:18] = 1.0
        p = dict(_PLAIN_CUT, method="Manual", value=0.5, post="Watershed (distance)", seeds="Distance maxima",
                 seed_distance=10.0, min_voxels=0)
        r = wb.run_step("classic", p, a)
        self.assertEqual(int(r.labels.max()), 2)
        self.assertEqual(int(np.count_nonzero(r.labels)), 98)
        self.assertEqual(len(np.unique(r.labels[0, 0, 2:9, 11:18])), 1)

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

    @unittest.skipUnless(_HAVE_SCIPY, "label post-processing needs scipy")
    @unittest.skipIf(tifffile is None, "tifffile writes the fixture")
    def test_import_labels_reads_a_label_tiff_onto_the_input(self):
        # what export_labels writes: one uint32 page per plane, t * z pages for a series
        a = np.zeros((1, 2, 3, 8, 8), np.float32)
        labels = np.zeros((2, 3, 8, 8), np.uint32)
        labels[:, :, 1:4, 1:4] = 4
        labels[:, 1:, 5:7, 5:7] = 7
        labels[1, 2, 0, 7] = 9      # a one-voxel speck in the last frame
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "labels.tif")
            tifffile.imwrite(path, labels.reshape(6, 8, 8), photometric="minisblack")   # 3 or 4 leading planes are not RGB(A)
            r = wb.run_step("import_labels", {"path": path}, a)
            np.testing.assert_array_equal(r.labels, labels)
            np.testing.assert_array_equal(r.array, a)
            self.assertEqual(r.info["labels"], 3)
            r = wb.run_step("import_labels", {"path": path, "min_voxels": 2, "relabel": True}, a)
            self.assertEqual(sorted(np.unique(r.labels).tolist()), [0, 1, 2])
            # one time point's planes serve every time point
            tifffile.imwrite(os.path.join(d, "one.tif"), labels[0], photometric="minisblack")
            r = wb.run_step("import_labels", {"path": os.path.join(d, "one.tif")}, a)
            np.testing.assert_array_equal(r.labels[1], labels[0])
            # the refusals: no file, the wrong grid, floating-point pixels
            with self.assertRaises(ValueError):
                wb.run_step("import_labels", {}, a)
            tifffile.imwrite(os.path.join(d, "short.tif"), labels.reshape(6, 8, 8)[:4], photometric="minisblack")
            with self.assertRaises(ValueError):
                wb.run_step("import_labels", {"path": os.path.join(d, "short.tif")}, a)
            tifffile.imwrite(os.path.join(d, "float.tif"), labels.reshape(6, 8, 8).astype(np.float32), photometric="minisblack")
            with self.assertRaises(ValueError):
                wb.run_step("import_labels", {"path": os.path.join(d, "float.tif")}, a)

    @unittest.skipUnless(_HAVE_SCIPY, "label post-processing needs scipy")
    def test_cleanup_numbers_every_frame_with_one_map(self):
        # track 4 in every frame, track 2 from t = 1, a speck of 3 in t = 0
        a = np.zeros((1, 3, 1, 16, 16), np.float32)
        labels = np.zeros((3, 1, 16, 16), np.uint32)
        labels[:, 0, 10:13, 10:13] = 4
        labels[1:, 0, 2:5, 2:5] = 2
        labels[0, 0, 0, 15] = 3
        r = wb.run_step("cleanup", {"min_voxels": 2, "relabel": True}, a, labels=labels)
        # one id per object in every frame, as cleanup.cpp: a numbering per
        # frame made the track 1 at t = 0 and 2 afterwards
        self.assertEqual(r.labels[:, 0, 11, 11].tolist(), [2, 2, 2])
        self.assertEqual(r.labels[1:, 0, 3, 3].tolist(), [1, 1])
        self.assertEqual(int(r.labels[0, 0, 0, 15]), 0)
        self.assertEqual(r.info["labels"], 2)

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
        # the last plane / column the extent promises is sampled: 189 * (0.1 / 0.3)
        # rounds past plane 63, and was read as fill
        ones = np.ones((1, 1, 64, 2, 64), np.float32)
        for interp in ("linear", "cubic", "nearest"):
            r7 = wb.run_step("resample", {"voxel_z": 0.1, "voxel_x": 0.1, "interpolation": interp}, ones,
                             {"voxel_um": [0.5, 0.5, 0.3]})
            self.assertEqual(r7.array.shape, (1, 1, 190, 2, 316))
            self.assertAlmostEqual(float(r7.array.min()), 1.0, places=6, msg=interp)   # not 0: filled

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
    @unittest.skipIf(_tiff_reader() is None, _NO_TIFF_READER)
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
        r = wb.run_step("classic", dict(_PLAIN_CUT, channel="488", method="Manual", value=1.0, min_voxels=0), a, meta)
        self.assertEqual(r.info["channel"], 1)
        self.assertEqual(int(r.labels.max()), 1)


if __name__ == "__main__":
    unittest.main()
