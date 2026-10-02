"""SIRIUS's TIFF reader against tifffile, the reference.

tifffile (with imagecodecs for its codecs) writes every file here and reads
it back; SIRIUS's C++ reader must return the same pixels -- page by page,
sample by sample -- and the same metadata. tifffile is a test oracle only:
neither the library, the application nor the worker reads with it.

Layout convention: SIRIUS returns samples as channel planes, (pages, samples,
height, width), whatever the planar configuration on disk; tifffile returns a
contiguous page as (height, width, samples) and a separate-planes page as
(samples, height, width). `_as_planes` turns tifffile's into SIRIUS's.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

import sirius

try:
    import tifffile  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    tifffile = None

try:
    import imagecodecs  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover - environment dependent
    imagecodecs = None


def _as_planes(page) -> np.ndarray:
    """A tifffile page as SIRIUS reads one page: (h, w) or (s, h, w)."""
    a = page.asarray()
    if int(page.samplesperpixel) > 1 and int(page.planarconfig) == 1:   # contiguous: (h, w, s)
        a = np.moveaxis(a, -1, 0)
    return a


def _rng_array(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return rng.integers(0, 2, size=shape).astype(bool)
    if dtype.kind == "f":
        return (rng.standard_normal(shape) * 100).astype(dtype)
    info = np.iinfo(dtype)
    return rng.integers(info.min, int(info.max) + 1, size=shape, dtype=np.int64).astype(dtype)


@unittest.skipIf(tifffile is None, "tifffile is the reference and is not installed")
class _Parity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def write(self, name, data, **kw):
        p = self.path(name)
        try:
            tifffile.imwrite(p, data, **kw)
        except (KeyError, ValueError, NotImplementedError) as e:
            if "imagecodecs" in str(e) or imagecodecs is None:
                self.skipTest(f"tifffile cannot write {kw} here: {e}")
            raise
        return p

    def assert_same_pages(self, path, exact_dtype=True):
        """Every main-chain page, every sample, plus a region and one sample."""
        f = sirius.TiffFile(path)
        with tifffile.TiffFile(path) as t:
            pages = [p for p in t.pages if not int(getattr(p, "subfiletype", 0)) & 1]
            self.assertEqual(f.info.page_count, len(pages))
            for k, page in enumerate(pages):
                ref = _as_planes(page)
                got = np.asarray(f.read_pages(k, 1))[0]
                if ref.dtype == np.bool_:
                    ref = ref.astype(np.uint8)
                if ref.dtype == np.float16:
                    ref = ref.astype(np.float32)   # SIRIUS widens float16
                if exact_dtype:
                    self.assertEqual(got.dtype, ref.dtype, f"page {k} of {os.path.basename(path)}")
                self.assertEqual(got.shape, ref.shape, f"page {k}")
                np.testing.assert_array_equal(got, ref, f"page {k} of {os.path.basename(path)}")
            # a region straddling strips / tiles, every page
            info = f.info
            h, w = info.height, info.width
            x, y, rw, rh = w // 5, h // 3, max(1, w // 2), max(1, h // 2)
            got = np.asarray(f.read_region(x, y, rw, rh))
            for k, page in enumerate(pages):
                ref = _as_planes(page)[..., y:y + rh, x:x + rw]
                np.testing.assert_array_equal(got[k].astype(np.float64), ref.astype(np.float64), f"region of page {k}")
            # one sample of a multi-sample file
            spp = int(info.samples_per_pixel)
            if spp > 1:
                last = np.asarray(f.read_stack(first_sample=spp - 1, samples=1))
                for k, page in enumerate(pages):
                    np.testing.assert_array_equal(last[k], _as_planes(page)[spp - 1])
        return f


class TestPixelTypesAndCodecs(_Parity):
    def test_every_pixel_type_uncompressed(self):
        for dtype in (np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.float32, np.float64,
                      np.float16, np.bool_):
            with self.subTest(dtype=np.dtype(dtype).name):
                data = _rng_array((3, 21, 34), dtype, seed=1)
                kw = {} if dtype == np.bool_ else {"photometric": "minisblack"}
                self.assert_same_pages(self.write(f"{np.dtype(dtype).name}.tif", data, **kw))

    def test_codecs_and_predictors(self):
        cases = [
            (np.uint16, "lzw", None), (np.uint16, "lzw", True), (np.uint16, "zlib", None), (np.uint16, "zlib", True),
            (np.uint16, "adobe_deflate", True), (np.uint8, "packbits", None), (np.int16, "packbits", None),
            (np.float32, "zlib", 3), (np.float32, "lzw", 3), (np.float64, "zlib", 3), (np.float16, "zlib", 3),
            (np.int32, "zlib", True), (np.uint32, "lzw", True),
        ]
        for dtype, comp, pred in cases:
            for tiled in (False, True):
                with self.subTest(dtype=np.dtype(dtype).name, compression=comp, predictor=pred, tiled=tiled):
                    data = _rng_array((2, 45, 70), dtype, seed=2)
                    if np.dtype(dtype).kind in "ui":
                        data = (np.cumsum(data.astype(np.int64), axis=-1) % 200).astype(dtype)   # compressible
                    kw = {"compression": comp, "photometric": "minisblack"}
                    if pred is not None:
                        kw["predictor"] = pred
                    if tiled:
                        kw["tile"] = (16, 32)
                    else:
                        kw["rowsperstrip"] = 7
                    self.assert_same_pages(self.write(f"{comp}-{pred}-{tiled}.tif", data, **kw))

    def test_byte_orders_and_bigtiff(self):
        for byteorder in ("<", ">"):
            for bigtiff in (False, True):
                for dtype, comp, pred in ((np.uint16, None, None), (np.int32, "zlib", True), (np.float32, "zlib", 3),
                                          (np.float64, None, None), (np.float16, None, None)):
                    with self.subTest(byteorder=byteorder, bigtiff=bigtiff, dtype=np.dtype(dtype).name, comp=comp):
                        data = _rng_array((2, 19, 23), dtype, seed=3)
                        kw = {"byteorder": byteorder, "bigtiff": bigtiff, "photometric": "minisblack"}
                        if comp:
                            kw.update(compression=comp, predictor=pred)
                        p = self.write(f"bo{byteorder == '>'}-{bigtiff}-{np.dtype(dtype).name}.tif", data, **kw)
                        f = self.assert_same_pages(p)
                        self.assertEqual(f.info.big_endian, byteorder == ">")
                        self.assertEqual(f.info.big_tiff, bigtiff)

    def test_packed_bit_depths(self):
        for bits in (2, 4, 10, 12, 14):
            for comp in (None,):   # tifffile writes packed samples uncompressed only
                with self.subTest(bits=bits, compression=comp):
                    dtype = np.uint8 if bits <= 8 else np.uint16
                    data = (_rng_array((2, 17, 29), dtype, seed=bits) & ((1 << bits) - 1)).astype(dtype)
                    p = self.write(f"bits{bits}-{comp}.tif", data, bitspersample=bits, compression=comp,
                                   photometric="minisblack")
                    f = self.assert_same_pages(p)
                    self.assertEqual(f.info.page(0).bits_per_sample, bits)

    def test_zstd_with_predictors(self):
        cases = [
            (np.uint8, None), (np.uint8, True), (np.uint16, None), (np.uint16, True), (np.int16, True),
            (np.int32, True), (np.uint32, None), (np.float32, None), (np.float32, 3), (np.float64, 3),
            (np.float16, 3),
        ]
        for dtype, pred in cases:
            for tiled in (False, True):
                with self.subTest(dtype=np.dtype(dtype).name, predictor=pred, tiled=tiled):
                    data = _rng_array((2, 45, 70), dtype, seed=11)
                    if np.dtype(dtype).kind in "ui":
                        data = (np.cumsum(data.astype(np.int64), axis=-1) % 200).astype(dtype)
                    kw = {"compression": "zstd", "photometric": "minisblack"}
                    if pred is not None:
                        kw["predictor"] = pred
                    kw.update({"tile": (16, 32)} if tiled else {"rowsperstrip": 7})
                    f = self.assert_same_pages(self.write(f"zstd-{np.dtype(dtype).name}-{pred}-{tiled}.tif", data, **kw))
                    self.assertEqual(f.info.page(0).compression, 50000)

    def test_codecs_this_build_lacks_are_reported_not_misread(self):
        data = _rng_array((1, 16, 16), np.uint8, seed=4)
        written = 0
        for comp in ("lzma", "webp", "jpeg2000", "jpegxl", "lerc"):
            with self.subTest(compression=comp):
                try:
                    p = self.path(f"{comp}.tif")
                    tifffile.imwrite(p, data, compression=comp, photometric="minisblack")
                except Exception:  # noqa: BLE001 - a codec tifffile cannot write here either
                    continue
                written += 1
                info = sirius.inspect_tiff(p)
                page = info.page(0)
                if page.decodable:
                    self.assert_same_pages(p)   # a libtiff built with the codec reads it
                else:
                    self.assertIn("compression", page.unsupported)
                    with self.assertRaises(Exception) as cm:
                        sirius.read_tiff(p)
                    self.assertIn("compression", str(cm.exception))
        if not written:
            self.skipTest("tifffile could write none of these codecs (imagecodecs missing)")


def _smooth(shape, dtype, seed):
    """Image-like data: JPEG's output depends on content, so give it some."""
    rng = np.random.default_rng(seed)
    return (np.cumsum(rng.integers(0, 9, size=shape), axis=-2) % 256).astype(dtype)


class TestJpeg(_Parity):
    """JPEG is lossy: the reference is what libjpeg-turbo decodes, through
    imagecodecs for tifffile and through libtiff for SIRIUS -- the same
    pixels, bit for bit. YCbCr comes back as RGB channels, as tifffile
    returns it."""

    def test_greyscale_strips_and_tiles(self):
        data = _smooth((2, 45, 70), np.uint8, seed=12)
        for tiled in (False, True):
            with self.subTest(tiled=tiled):
                kw = {"tile": (16, 32)} if tiled else {"rowsperstrip": 16}
                f = self.assert_same_pages(self.write(f"jpeg-grey-{tiled}.tif", data, compression="jpeg",
                                                      photometric="minisblack", **kw))
                self.assertEqual(f.info.page(0).compression, 7)

    def test_ycbcr_every_subsampling_as_rgb(self):
        data = _smooth((2, 45, 70, 3), np.uint8, seed=13)
        for ss in ((1, 1), (2, 1), (2, 2), (4, 1)):
            for tiled in (False, True):
                with self.subTest(subsampling=ss, tiled=tiled):
                    kw = {"tile": (32, 32)} if tiled else {"rowsperstrip": 16}
                    p = self.write(f"jpeg-ycbcr-{ss[0]}{ss[1]}-{tiled}.tif", data, compression="jpeg",
                                   photometric="rgb", subsampling=ss, **kw)
                    f = self.assert_same_pages(p)
                    page = f.info.page(0)
                    self.assertEqual(page.photometric, 6)   # YCbCr on disk ...
                    self.assertEqual(f.info.shape, (2, 3, 45, 70))   # ... RGB channels read
                    with tifffile.TiffFile(p) as t:
                        ref = np.moveaxis(t.asarray(), -1, 1)
                    np.testing.assert_array_equal(np.asarray(f.read_stack()), ref)

    def test_rgb_stored_as_rgb(self):
        data = _smooth((2, 33, 40, 3), np.uint8, seed=14)
        p = self.write("jpeg-rgb.tif", data, compression="jpeg", photometric="rgb",
                       compressionargs={"outcolorspace": "rgb"})
        f = self.assert_same_pages(p)
        self.assertEqual(f.info.page(0).photometric, 2)

    def test_twelve_bit(self):
        # an odd width too: libtiff 4.7 drops the last sample of such rows
        # (cmake/patches/fix_libtiff_jpeg12_odd.cmake)
        for width in (70, 69):
            with self.subTest(width=width):
                data = (_smooth((2, 45, width), np.uint16, seed=15) * 16).astype(np.uint16)
                p = self.write(f"jpeg12-{width}.tif", data, compression="jpeg", photometric="minisblack",
                               bitspersample=12)
                f = self.assert_same_pages(p)
                self.assertEqual(f.info.page(0).bits_per_sample, 12)
                self.assertEqual(np.asarray(f.read_stack()).dtype, np.uint16)


class TestSamples(_Parity):
    def test_rgb_and_rgba_both_planar_configurations(self):
        for spp, photometric, extra in ((3, "rgb", None), (4, "rgb", ["unassalpha"]), (2, "minisblack", ["unspecified"])):
            for planar in ("contig", "separate"):
                for tiled in (False, True):
                    for dtype, comp in ((np.uint8, None), (np.uint16, "lzw"), (np.float32, "zlib")):
                        with self.subTest(spp=spp, planar=planar, tiled=tiled, dtype=np.dtype(dtype).name):
                            shape = (3, 33, 41, spp) if planar == "contig" else (3, spp, 33, 41)
                            data = _rng_array(shape, dtype, seed=spp)
                            kw = {"photometric": photometric, "planarconfig": planar, "compression": comp}
                            if extra:
                                kw["extrasamples"] = extra
                            if tiled:
                                kw["tile"] = (16, 16)
                            p = self.write(f"s{spp}-{planar}-{tiled}-{np.dtype(dtype).name}.tif", data, **kw)
                            f = self.assert_same_pages(p)
                            self.assertEqual(f.info.shape, (3, spp, 33, 41))
                            self.assertEqual(f.info.page(0).planar_config, 1 if planar == "contig" else 2)
                            full = np.asarray(f.read_stack())
                            ref = data if planar == "separate" else np.moveaxis(data, -1, 1)
                            np.testing.assert_array_equal(full, ref)
                            np.testing.assert_array_equal(np.asarray(f.read_stack(first_sample=1, samples=1)), ref[:, 1])
                            np.testing.assert_array_equal(np.asarray(f.read_stack(dtype=np.float64)),
                                                          ref.astype(np.float64))

    def test_palette_image_indices_and_colormap(self):
        data = _rng_array((2, 20, 30), np.uint8, seed=5)
        cmap = np.stack([np.arange(256) * 257, 65535 - np.arange(256) * 257, np.arange(256) * 100]).astype(np.uint16)
        p = self.write("palette.tif", data, photometric="palette", colormap=cmap)
        f = self.assert_same_pages(p)
        with tifffile.TiffFile(p) as t:
            np.testing.assert_array_equal(f.info.page(0).colormap, t.pages[0].colormap)
        self.assertEqual(f.info.page(0).photometric, 3)


class TestSparseFiles(_Parity):
    def test_missing_tiles_and_strips_read_as_zeros(self):
        for tile in ((16, 16), None):
            with self.subTest(tile=tile):
                p = self.path(f"sparse-{tile is not None}.tif")
                with tifffile.TiffWriter(p) as tw:
                    tw.write(shape=(2, 40, 50), dtype=np.uint16, tile=tile, photometric="minisblack")
                self.assert_same_pages(p)
                self.assertFalse(np.asarray(sirius.read_tiff(p)).any())


class TestMetadataSeriesAndPyramids(_Parity):
    def test_imagej_hyperstack(self):
        data = _rng_array((3, 4, 2, 20, 30), np.uint16, seed=6)   # t z c y x
        p = self.write("ij.tif", data, imagej=True, resolution=(1 / 0.08, 1 / 0.08),
                       metadata={"axes": "TZCYX", "spacing": 0.4, "unit": "um", "finterval": 3.0})
        f = self.assert_same_pages(p)
        md = f.metadata
        with tifffile.TiffFile(p) as t:
            ij = t.imagej_metadata
        self.assertTrue(md.imagej)
        self.assertEqual((md.size_t, md.size_z, md.size_c), (ij["frames"], ij["slices"], ij["channels"]))
        self.assertEqual(md.image_j.images, ij["images"])
        self.assertAlmostEqual(md.image_j.spacing, ij["spacing"])
        self.assertAlmostEqual(md.frame_interval_s, ij["finterval"])
        self.assertEqual(md.image_j.unit, ij["unit"])
        self.assertAlmostEqual(md.voxel_um[2], 0.4)
        np.testing.assert_array_equal(np.asarray(f.read_stack()).reshape(data.shape), data)

    def test_ome_series_metadata_and_pyramids(self):
        big = _rng_array((2, 3, 64, 80), np.uint16, seed=7)   # c z y x
        small = _rng_array((4, 30, 20), np.uint8, seed=8)
        p = self.path("multi.ome.tif")
        with tifffile.TiffWriter(p, ome=True) as tw:
            tw.write(big, subifds=2, tile=(16, 16), photometric="minisblack",
                     metadata={"axes": "CZYX", "Name": "big", "PhysicalSizeX": 0.11, "PhysicalSizeY": 0.11,
                               "PhysicalSizeZ": 0.5, "Channel": {"Name": ["DAPI", "GFP"]}})
            tw.write(big[..., ::2, ::2], subfiletype=1, tile=(16, 16), photometric="minisblack")
            tw.write(big[..., ::4, ::4], subfiletype=1, tile=(16, 16), photometric="minisblack")
            tw.write(small, photometric="minisblack", metadata={"axes": "TYX", "Name": "small"})
        f = sirius.TiffFile(p)
        md = f.metadata
        with tifffile.TiffFile(p) as t:
            self.assertEqual(len(f.series()), len(t.series))
            for i, s in enumerate(t.series):
                levels = s.levels
                self.assertEqual(f.series_levels(i), len(levels), f"series {i}")
                for k, lv in enumerate(levels):
                    ref = lv.asarray()
                    got = np.asarray(f.read_series(i, k))
                    np.testing.assert_array_equal(got.reshape(ref.shape), ref, f"series {i} level {k}")
            self.assertEqual([im.name for im in md.ome_images], ["big", "small"])
            img = md.ome_images[0]
            self.assertEqual((img.size_c, img.size_z, img.size_t, img.size_y, img.size_x), (2, 3, 1, 64, 80))
            self.assertTrue(img.dimension_order.startswith("XY"), img.dimension_order)
            self.assertAlmostEqual(img.physical_size_um[0], 0.11)
            self.assertAlmostEqual(img.physical_size_um[2], 0.5)
            self.assertEqual([c.name for c in img.channels], ["DAPI", "GFP"])
            self.assertEqual(md.ome_images[1].size_t, 4)

    def test_ome_rgb(self):
        data = _rng_array((2, 3, 25, 31, 3), np.uint8, seed=9)   # t z y x s
        p = self.write("rgb.ome.tif", data, photometric="rgb", metadata={"axes": "TZYXS"})
        f = self.assert_same_pages(p)
        md = f.metadata
        self.assertEqual((md.size_c, md.size_z, md.size_t), (3, 3, 2))
        self.assertEqual(md.ome_images[0].channels[0].samples_per_pixel, 3)
        with tifffile.TiffFile(p) as t:
            ref = t.series[0].asarray()
        got = np.asarray(f.read_series(0))   # (pages, s, y, x)
        np.testing.assert_array_equal(np.moveaxis(got, 1, -1).reshape(ref.shape), ref)

    def test_metadata_parser_against_tifffile_ome(self):
        data = _rng_array((2, 3, 4, 10, 12), np.uint16, seed=10)
        p = self.write("o.ome.tif", data, photometric="minisblack",
                       metadata={"axes": "TCZYX", "PhysicalSizeX": 0.2, "PhysicalSizeXUnit": "nm",
                                 "TimeIncrement": 1.5, "TimeIncrementUnit": "s"})
        with tifffile.TiffFile(p) as t:
            desc = t.pages[0].description
            series = t.series[0]
        md = sirius.parse_tiff_metadata(desc)
        self.assertTrue(md.ome)
        self.assertEqual((md.size_t, md.size_c, md.size_z), (2, 3, 4))
        self.assertAlmostEqual(md.voxel_um[0], 0.2e-3)
        self.assertAlmostEqual(md.frame_interval_s, 1.5)
        pages = sirius.ome_image_pages(md, len(series.pages))
        self.assertEqual(pages[0], list(range(24)))


if __name__ == "__main__":
    unittest.main()
