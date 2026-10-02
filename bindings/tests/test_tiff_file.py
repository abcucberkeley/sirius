"""Tests for sirius.TiffFile / inspect_tiff and device-aware read_tiff."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

import sirius


def _write(path, arr, **kw):
    """Write with tifffile when available (tiles, pyramids), else sirius.write_tiff."""
    try:
        import tifffile
    except ImportError as e:
        if kw:
            raise unittest.SkipTest("tifffile is required for this test") from e
        sirius.write_tiff(path, arr)
        return
    try:
        tifffile.imwrite(path, arr, photometric="minisblack", **kw)
    except KeyError as exc:
        # tifffile delegates most codecs to imagecodecs and raises KeyError
        # naming the missing package. The reader under test is sirius's, so
        # skip rather than fail when the fixture cannot be produced.
        if "imagecodecs" not in str(exc):
            raise
        raise unittest.SkipTest(f"imagecodecs is required to write {kw}") from exc


class TestInspect(unittest.TestCase):
    def test_info_fields(self):
        arr = np.arange(3 * 20 * 30, dtype=np.uint16).reshape(3, 20, 30)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.tif")
            sirius.write_tiff(path, arr)
            info = sirius.inspect_tiff(path)
            self.assertTrue(info.big_tiff)
            self.assertEqual(info.page_count, 3)
            self.assertEqual(info.level_count, 1)
            self.assertEqual(info.shape, (3, 20, 30))
            self.assertEqual(info.dtype, np.dtype(np.uint16))
            self.assertEqual(info.pixel_type, sirius.PixelType.UInt16)
            self.assertTrue(info.uniform_pages)
            img = info.page(2)
            self.assertEqual((img.width, img.height), (30, 20))
            self.assertEqual(img.layout, sirius.TiffLayout.Strips)
            self.assertEqual(img.compression, 1)
            self.assertEqual(img.predictor, 1)
            self.assertFalse(img.reduced_resolution)
            self.assertEqual(info.levels[0].ifds, info.pages)
            self.assertIn("TiffImageInfo", repr(img))
            f = sirius.TiffFile(path)
            self.assertEqual(f.path, path)
            self.assertEqual(f.info.page_count, 3)

    def test_missing_file_raises(self):
        with self.assertRaises(RuntimeError):
            sirius.inspect_tiff("/no/such/file.tif")


class TestReads(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.arr = self.rng.integers(0, 65535, size=(4, 37, 53), dtype=np.uint16)

    def test_read_stack_native_and_converted(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.tif")
            sirius.write_tiff(path, self.arr)
            f = sirius.TiffFile(path)
            a = f.read_stack()
            self.assertIsInstance(a, np.ndarray)
            self.assertEqual(a.dtype, np.uint16)
            np.testing.assert_array_equal(a, self.arr)
            b = f.read_stack(dtype=np.float32)
            self.assertEqual(b.dtype, np.float32)
            np.testing.assert_array_equal(b, self.arr.astype(np.float32))
            c = f.read_stack(dtype="float64")
            self.assertEqual(c.dtype, np.float64)
            with self.assertRaises(ValueError):
                f.read_stack(dtype=np.complex64)

    def test_read_pages_and_region(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.tif")
            sirius.write_tiff(path, self.arr)
            f = sirius.TiffFile(path)
            np.testing.assert_array_equal(f.read_pages(1, 2), self.arr[1:3])
            np.testing.assert_array_equal(f.read_region(5, 3, 20, 10), self.arr[:, 3:13, 5:25])
            np.testing.assert_array_equal(f.read_region(10, 0), self.arr[:, :, 10:])
            with self.assertRaises(IndexError):
                f.read_region(53, 0, 1, 1)
            with self.assertRaises(IndexError):
                f.read_pages(3, 2)

    def test_region_of_chosen_pages_and_the_tags(self):
        # What the cluster worker reads a zoomed-in plane with: the region of
        # one page, not of every page; and the tags it shapes a dataset by.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.tif")
            sirius.write_tiff(path, self.arr)
            f = sirius.TiffFile(path)
            np.testing.assert_array_equal(f.read_region(5, 3, 20, 10, first=1, count=1), self.arr[1:2, 3:13, 5:25])
            np.testing.assert_array_equal(f.read_region(5, 3, 20, 10, first=1), self.arr[1:, 3:13, 5:25])
            for first, count in ((4, 1), (1, 4), (0, 2**63)):
                with self.subTest(first=first, count=count), self.assertRaises(IndexError):
                    f.read_region(0, 0, 1, 1, first=first, count=count)
            page = f.info.page(0)
            self.assertIsInstance(page.description, str)
            self.assertGreaterEqual(page.x_resolution, 0.0)
            self.assertGreaterEqual(page.y_resolution, 0.0)
            self.assertIn(page.resolution_unit, (1, 2, 3))

    def test_tiled_and_compressed(self):
        for kw in ({"tile": (16, 16)}, {"compression": "lzw"}, {"tile": (16, 16), "compression": "zlib"}):
            with self.subTest(kw=kw), tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "t.tif")
                _write(path, self.arr, **kw)
                f = sirius.TiffFile(path)
                if "tile" in kw:
                    self.assertEqual(f.info.page(0).layout, sirius.TiffLayout.Tiles)
                    self.assertEqual(f.info.page(0).tile_width, 16)
                np.testing.assert_array_equal(f.read_stack(), self.arr)
                np.testing.assert_array_equal(f.read_region(7, 9, 30, 20), self.arr[:, 9:29, 7:37])

    def test_pyramid_levels(self):
        try:
            import tifffile
        except ImportError:
            self.skipTest("tifffile is required")
        base = self.rng.integers(0, 65535, size=(128, 160), dtype=np.uint16)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.tif")
            with tifffile.TiffWriter(path) as tw:
                tw.write(base, subifds=2, tile=(32, 32), photometric="minisblack")
                tw.write(base[::2, ::2], subfiletype=1, tile=(32, 32), photometric="minisblack")
                tw.write(base[::4, ::4], subfiletype=1, tile=(32, 32), photometric="minisblack")
            f = sirius.TiffFile(path)
            self.assertEqual(f.info.page_count, 1)
            self.assertEqual(f.info.level_count, 3)
            self.assertEqual((f.info.levels[1].width, f.info.levels[1].height), (80, 64))
            self.assertEqual(len(f.info.page(0).sub_ifds), 2)
            np.testing.assert_array_equal(f.read_level(0)[0], base)
            np.testing.assert_array_equal(f.read_level(1)[0], base[::2, ::2])
            np.testing.assert_array_equal(f.read_level(2)[0], base[::4, ::4])
            np.testing.assert_array_equal(f.read_region(3, 5, 20, 10, level=1)[0], base[::2, ::2][5:15, 3:23])
            with self.assertRaises(IndexError):
                f.read_level(3)

    def test_read_tiff_keeps_numpy_contract(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.tif")
            sirius.write_tiff(path, self.arr)
            a = sirius.read_tiff(path)
            self.assertIsInstance(a, np.ndarray)
            np.testing.assert_array_equal(a, self.arr)
            b = sirius.read_tiff(path, dtype=np.float32)
            self.assertEqual(b.dtype, np.float32)


class TestGpuReads(unittest.TestCase):
    def setUp(self):
        if not sirius.cuda_available():
            self.skipTest("no CUDA device available")
        if not sirius.built_with_nvtiff():
            self.skipTest("built without nvTIFF")
        self.gpu = sirius.Device.cuda(0)
        self.arr = np.random.default_rng(2).integers(0, 65535, size=(5, 45, 70), dtype=np.uint16)

    def test_gpu_decode_matches_cpu(self):
        for kw in ({}, {"tile": (16, 16)}, {"compression": "lzw"}, {"compression": "zlib"}):
            with self.subTest(kw=kw), tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "g.tif")
                _write(path, self.arr, **kw)
                f = sirius.TiffFile(path)
                ok, reason = f.gpu_decodable(self.gpu)
                if kw.get("compression") == "zlib" and not ok:
                    print(f"note: deflate not GPU-decodable here ({reason}); fallback path exercised")
                else:
                    self.assertTrue(ok, reason)
                buf = f.read_stack(device=self.gpu)
                self.assertIsInstance(buf, sirius.Buffer)
                self.assertEqual(buf.device, self.gpu)
                self.assertEqual(buf.shape, (5, 45, 70))
                np.testing.assert_array_equal(buf.numpy(), self.arr)

                reg = f.read_region(5, 3, 30, 20, device="cuda", dtype=np.float32)
                self.assertEqual(reg.dtype, np.dtype(np.float32))
                np.testing.assert_array_equal(reg.numpy(), self.arr[:, 3:23, 5:35].astype(np.float32))

    def test_read_tiff_device_argument(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "g.tif")
            sirius.write_tiff(path, self.arr)
            buf = sirius.read_tiff(path, device="cuda")
            self.assertIsInstance(buf, sirius.Buffer)
            np.testing.assert_array_equal(buf.numpy(), self.arr)

    def test_fallback_for_unsupported_codec(self):
        # sirius writes float LZW with the floating-point predictor, which
        # nvTIFF rejects: the read must still land on the GPU via libtiff.
        arr = np.random.default_rng(3).standard_normal((3, 32, 40)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.tif")
            sirius.write_tiff(path, arr, comp=sirius.TiffCompression.Lzw)
            f = sirius.TiffFile(path)
            ok, reason = f.gpu_decodable(self.gpu)
            self.assertFalse(ok)
            self.assertIn("NVTIFF_STATUS", reason)
            buf = f.read_stack(device=self.gpu)
            np.testing.assert_array_equal(buf.numpy(), arr)
            with self.assertRaises(RuntimeError):
                f.read_stack(device=self.gpu, allow_cpu_fallback=False)

    def test_torch_consumes_decoded_stack(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        if not torch.cuda.is_available():
            self.skipTest("torch has no CUDA")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "g.tif")
            sirius.write_tiff(path, self.arr)
            buf = sirius.TiffFile(path).read_stack(device="cuda", dtype=np.float32)
            t = torch.from_dlpack(buf)
            self.assertEqual(t.device.type, "cuda")
            # sum in float64 on the GPU: exact for these integer-valued pixels
            self.assertEqual(float(t.to(torch.float64).sum().item()), float(self.arr.astype(np.float64).sum()))


if __name__ == "__main__":
    unittest.main()
