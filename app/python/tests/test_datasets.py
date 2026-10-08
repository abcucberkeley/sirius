"""Tests of the cluster dataset reader (sirius_worker.datasets) and of the
worker serving it: the page order the application uses, the block reduction
a pane is drawn from, the compressed wire form, dataset_* over a real socket,
a step reading its input by reference, and several clients at once
(--max-clients) with a run's cancel reaching only its own connection.

    python -m unittest discover -s app/python/tests
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # app/python

from sirius_worker import datasets, protocol  # noqa: E402
from sirius_worker.server import WorkerServer  # noqa: E402

# The real sirius extension, when it is built for this Python (before any test
# replaces sys.modules["sirius"] with a fake).
REAL_SIRIUS = datasets._sirius()

try:
    import tifffile  # type: ignore  # only to write an ImageJ hyperstack for a test; the worker never reads with it

    HAVE_TIFFFILE = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_TIFFFILE = False


class _Client:
    def __init__(self, port: int, token: str = ""):
        import socket

        self.sock = socket.create_connection(("127.0.0.1", port), timeout=30)
        self.next_id = 1
        self.token = token

    def call(self, method, params=None, tensors=None):
        rid = self.next_id
        self.next_id += 1
        protocol.write_frame(self.sock, {"id": rid, "type": "request", "method": method, "params": params or {}}, tensors)
        while True:
            header, out = protocol.read_frame(self.sock)
            if header.get("type") == "progress":
                continue
            return header, out

    def hello(self):
        header = protocol.client_handshake(self.sock, self.token, first_id=self.next_id)
        self.next_id += 2
        return header

    def close(self):
        self.sock.close()


class TestReduction(unittest.TestCase):
    def test_blocks_are_means_with_partial_edges(self):
        a = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
        r = datasets.reduce_blocks(a, (2, 3))
        self.assertEqual(r.shape, (3, 3))
        self.assertAlmostEqual(float(r[0, 0]), float(a[0:2, 0:3].mean()))
        self.assertAlmostEqual(float(r[2, 2]), float(a[4:5, 6:7].mean()))   # a 1 x 1 corner block

    def test_integers_keep_their_dtype_rounded(self):
        a = np.array([[1, 2], [2, 2]], dtype=np.uint16)
        r = datasets.reduce_blocks(a, (2, 2))
        self.assertEqual(r.dtype, np.uint16)
        self.assertEqual(int(r[0, 0]), 2)   # 1.75 rounds to 2

    def test_factor_one_is_the_array(self):
        a = np.ones((3, 4), np.uint8)
        np.testing.assert_array_equal(datasets.reduce_blocks(a, (1, 1)), a)


class TestEncoding(unittest.TestCase):
    def test_round_trip_shuffled_zlib(self):
        rng = np.random.default_rng(1)
        a = (rng.normal(1000, 20, size=(64, 128))).astype(np.uint16)
        desc, t = datasets.encode(a, ["zlib"])
        self.assertEqual(desc["encoding"], "zlib")
        self.assertTrue(desc["shuffle"])
        self.assertEqual(t.dtype, np.uint8)
        self.assertLess(t.nbytes, a.nbytes)
        np.testing.assert_array_equal(datasets.decode(desc, t), a)

    def test_no_accepted_encoding_sends_the_array(self):
        a = np.zeros((64, 64), np.float32)
        desc, t = datasets.encode(a, [])
        self.assertEqual(desc["encoding"], "raw")
        self.assertIs(t.dtype, a.dtype)
        desc, t = datasets.encode(a, ["lz4"])   # one this worker does not have
        self.assertEqual(desc["encoding"], "raw")

    def test_incompressible_data_goes_raw(self):
        a = np.random.default_rng(2).integers(0, 255, size=(128, 128), dtype=np.uint8)
        desc, _ = datasets.encode(a, ["zlib"])
        self.assertEqual(desc["encoding"], "raw")

    def test_a_stream_that_inflates_past_its_shape_is_refused(self):
        import zlib

        bomb = np.frombuffer(zlib.compress(bytes(1 << 20), 9), dtype=np.uint8)   # 1 MiB of zeros, ~1 KiB packed
        desc = {"encoding": "zlib", "shuffle": False, "dtype": "uint8", "shape": [16]}
        with self.assertRaises(datasets.DatasetError):
            datasets.decode(desc, bomb)
        whole = datasets.decode({**desc, "shape": [1024, 1024]}, bomb)
        self.assertEqual(whole.shape, (1024, 1024))


class _Files(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        datasets._VOLUMES.clear()

    def tearDown(self):
        datasets.forget_all()
        self.tmp.cleanup()

    def path(self, name):
        return os.path.join(self.tmp.name, name)


class TestNpy(_Files):
    def test_a_5d_array_reads_by_c_t_z(self):
        a = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.uint16).reshape(2, 3, 4, 5, 6)
        np.save(self.path("a.npy"), a)
        ds = datasets.open_dataset(self.path("a.npy"))
        self.assertEqual(ds.meta()["dims"], [2, 3, 4, 5, 6])
        self.assertEqual(ds.meta()["dtype"], "uint16")
        np.testing.assert_array_equal(ds.plane(1, 2, 3), a[1, 2, 3])
        np.testing.assert_array_equal(ds.volume(1, 0), a[1, 0])

    def test_views_are_the_reduced_slices(self):
        a = np.random.default_rng(3).integers(0, 4000, size=(1, 1, 8, 10, 12)).astype(np.uint16)
        np.save(self.path("v.npy"), a)
        ds = datasets.open_dataset(self.path("v.npy"))
        vol = a[0, 0]
        np.testing.assert_array_equal(ds.view("xy", 0, 0, 3, 2), datasets.reduce_blocks(vol[3], (2, 2)))
        np.testing.assert_array_equal(ds.view("xz", 0, 0, 4, 1), vol[:, 4, :])
        np.testing.assert_array_equal(ds.view("yz", 0, 0, 5, 1), vol[:, :, 5].T)
        np.testing.assert_array_equal(ds.view("mip", 0, 0, 0, 1), vol.max(axis=0))
        # a region (x, y, w, h) of the plane at full resolution
        np.testing.assert_array_equal(ds.view("xy", 0, 0, 2, 1, [4, 2, 5, 3]), vol[2, 2:5, 4:9])
        small = ds.view("volume", 0, 0, max_side=4)
        self.assertLessEqual(max(small.shape), 4)

    def test_a_missing_file_says_so(self):
        with self.assertRaises(datasets.DatasetError) as e:
            datasets.open_dataset(self.path("nope.tif"))
        self.assertIn("nope.tif", str(e.exception))

    def test_read_ref_gives_float32_volumes(self):
        a = np.arange(2 * 1 * 3 * 4 * 5, dtype=np.uint8).reshape(2, 1, 3, 4, 5)
        np.save(self.path("r.npy"), a)
        v = datasets.read_ref({"path": self.path("r.npy"), "c": 1, "t": 0})
        self.assertEqual(v.dtype, np.float32)
        np.testing.assert_array_equal(v, a[1, 0].astype(np.float32))
        all5 = datasets.read_ref({"path": self.path("r.npy"), "layout": "ctzyx"})
        self.assertEqual(all5.shape, (2, 1, 3, 4, 5))

    def test_read_ref_indices_are_bounded_by_the_dataset(self):
        # a request names channels and times: each must exist and come once,
        # so the request cannot size the output (a billion zeros used to be a
        # billion volumes)
        a = np.zeros((2, 1, 3, 4, 5), dtype=np.uint8)
        np.save(self.path("b.npy"), a)
        ref = {"path": self.path("b.npy"), "layout": "ctzyx"}
        self.assertEqual(datasets.read_ref({**ref, "c": [1, 0], "t": 0}).shape, (2, 1, 3, 4, 5))
        for c in ([0] * 1000, [0, 0], [2], [-1], ["0"], [True], "01"):
            with self.subTest(c=c):
                with self.assertRaises(datasets.DatasetError):
                    datasets.read_ref({**ref, "c": c})


# --- MRC / DeltaVision: the header with struct, the sections a memmap ---------------------------

# the SIM test case's raw.dv / otf.dv, the same arrays as raw.tif / otf.tif beside them
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "tests", "data")


def write_mrc(path, nx=8, ny=6, planes=4, waves=1, times=1, sequence=0, mode=2, big=False,
              wavelengths=(), voxel=(0.1, 0.1, 0.3), deltavision=True, value=None):
    """A DeltaVision (or MRC2014) stack: the header fields the reader uses, then the
    sections in the sequence's order; value(w, t, z, yy, xx) gives each section."""
    import struct

    e = ">" if big else "<"
    hdr = bytearray(1024)
    sections = planes * waves * times
    struct.pack_into(e + "4i", hdr, 0, nx, ny, sections, mode)
    struct.pack_into(e + "3i", hdr, 64, 1, 2, 3)
    if deltavision:
        struct.pack_into(e + "3i", hdr, 28, 1, 1, 1)
        struct.pack_into(e + "3f", hdr, 40, *voxel)
        struct.pack_into(e + "h", hdr, 96, -16224)
        struct.pack_into(e + "2h", hdr, 128, 8, 32)
        struct.pack_into(e + "h", hdr, 180, times)
        struct.pack_into(e + "h", hdr, 182, sequence)
        struct.pack_into(e + "h", hdr, 196, waves)
        struct.pack_into(e + "5h", hdr, 198, *(list(wavelengths) + [0] * 5)[:5])
    else:
        struct.pack_into(e + "3i", hdr, 28, nx, ny, sections)
        struct.pack_into(e + "3f", hdr, 40, voxel[0] * 1e4 * nx, voxel[1] * 1e4 * ny, voxel[2] * 1e4 * sections)
        hdr[208:212] = b"MAP "
        hdr[212:216] = bytes([0x11, 0x11, 0, 0]) if big else bytes([0x44, 0x44, 0, 0])
    code = {0: "u1", 1: "i2", 2: "f4", 3: "i2", 4: "f4", 6: "u2", 7: "i4"}[mode]
    width = 2 * nx if mode in (3, 4) else nx
    data = np.zeros((sections, ny, width), dtype=np.dtype(e + code))
    yy, xx = np.mgrid[0:ny, 0:width]
    for k in range(sections):
        if not deltavision or sequence == 0:          # ZTW
            z, t, w = k % planes, (k // planes) % times, k // (planes * times)
        elif sequence == 1:                           # WZT
            w, z, t = k % waves, (k // waves) % planes, k // (waves * planes)
        else:                                         # ZWT
            z, w, t = k % planes, (k // planes) % waves, k // (planes * waves)
        data[k] = (value or stamp)(w, t, z, yy, xx)
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(data.tobytes())


def stamp(w, t, z, yy, xx):
    """A sample that names its own (w, t, z, y, x); fits int16 for the sizes used."""
    return w * 10000 + t * 1000 + z * 100 + yy * 10 + xx


class TestMrc(_Files):
    @unittest.skipUnless(os.path.exists(os.path.join(DATA, "raw.dv")), "tests/data/raw.dv is not beside this checkout")
    def test_the_shipped_raw_dv_opens_as_the_application_does(self):
        path = os.path.join(DATA, "raw.dv")
        ds = datasets.open_dataset(path)
        m = ds.meta()
        self.assertEqual(m["name"], "raw")
        self.assertEqual(m["format"], "deltavision")
        self.assertEqual(m["dims"], [1, 1, 135, 64, 64])
        self.assertEqual(m["dtype"], "float32")
        self.assertEqual(m["bytes"], 2212864)
        self.assertTrue(m["dims_from_metadata"])
        self.assertFalse(m["rgb"])
        np.testing.assert_allclose(m["voxel_um"], [0.08, 0.08, 0.125], rtol=1e-6)
        self.assertEqual(m["channels"], [{"name": "528", "wavelength_nm": 528.0}])
        h = datasets.mrc_header(path)
        self.assertEqual((h["nx"], h["ny"], h["sections"], h["mode"], h["data_offset"]), (64, 64, 135, 2, 1024))
        self.assertTrue(h["deltavision"])
        self.assertFalse(h["big_endian"])
        # the sections, in file order, are the (z, y, x) volume
        raw = np.fromfile(path, dtype="<f4", offset=1024).reshape(135, 64, 64)
        vol = ds.volume(0, 0)
        self.assertEqual(vol.dtype.name, "float32")
        np.testing.assert_array_equal(vol, raw)
        np.testing.assert_array_equal(ds.plane(0, 0, 70), raw[70])
        np.testing.assert_array_equal(ds.view("xy", 0, 0, 7, 1, [4, 2, 5, 3]), raw[7, 2:5, 4:9])
        if HAVE_TIFFFILE:
            # raw.tif is Bio-Formats' export of raw.dv, which reverses the rows of a
            # DeltaVision section (MRC's origin is bottom-left); the values are the same
            np.testing.assert_array_equal(vol, tifffile.imread(os.path.join(DATA, "raw.tif"))[:, ::-1, :])
        # the wire form is the same array
        desc, tensor = datasets.encode(ds.plane(0, 0, 3), ["zlib"])
        self.assertEqual(desc["dtype"], "float32")
        self.assertEqual(desc["shape"], [64, 64])

    @unittest.skipUnless(os.path.exists(os.path.join(DATA, "otf.dv")), "tests/data/otf.dv is not beside this checkout")
    def test_otf_dv_reads_its_complex_pairs_as_columns(self):
        path = os.path.join(DATA, "otf.dv")
        h = datasets.mrc_header(path)
        self.assertEqual((h["mode"], h["nx"], h["width"], h["ny"], h["sections"]), (4, 65, 130, 129, 3))
        self.assertTrue(h["complex"])
        ds = datasets.open_dataset(path)
        self.assertEqual(ds.meta()["dims"], [1, 1, 3, 129, 130])
        self.assertEqual(ds.meta()["dtype"], "float32")
        vol = ds.volume(0, 0)
        np.testing.assert_array_equal(vol, np.fromfile(path, dtype="<f4", offset=1024).reshape(3, 129, 130))
        self.assertEqual((float(vol[0, 0, 0]), float(vol[0, 0, 1])), (1.0, 0.0))   # 1 + 0i at the origin of order 0

    def test_wavelengths_and_time_points_follow_the_sequence(self):
        yy, xx = np.mgrid[0:6, 0:8]
        for sequence in (0, 1, 2):
            with self.subTest(sequence=sequence):
                path = self.path(f"seq{sequence}.dv")
                write_mrc(path, waves=2, times=3, planes=4, sequence=sequence, mode=1, wavelengths=(488, 561))
                ds = datasets.open_dataset(path)
                m = ds.meta()
                self.assertEqual(m["dims"], [2, 3, 4, 6, 8])
                self.assertEqual(m["dtype"], "int16")
                self.assertEqual(m["format"], "deltavision")
                self.assertTrue(m["dims_from_metadata"])
                self.assertEqual(m["channels"], [{"name": "488", "wavelength_nm": 488.0}, {"name": "561", "wavelength_nm": 561.0}])
                np.testing.assert_allclose(m["voxel_um"], [0.1, 0.1, 0.3], rtol=1e-6)
                for w in range(2):
                    for t in range(3):
                        for z in range(4):
                            np.testing.assert_array_equal(ds.plane(w, t, z), stamp(w, t, z, yy, xx))
                vol = ds.volume(1, 2)
                self.assertEqual(vol.shape, (4, 6, 8))
                np.testing.assert_array_equal(vol[3], stamp(1, 2, 3, yy, xx))
                np.testing.assert_array_equal(ds.view("mip", 1, 0, 0, 1), stamp(1, 0, 3, yy, xx))

    def test_big_endian_uint16_reads_and_ships_little_endian(self):
        path = self.path("be.dv")
        write_mrc(path, mode=6, big=True, value=lambda w, t, z, yy, xx: 60000 + z * 10 + yy + xx)
        ds = datasets.open_dataset(path)
        self.assertEqual(ds.meta()["dtype"], "uint16")
        self.assertEqual(int(ds.plane(0, 0, 2)[3, 4]), 60027)
        desc, tensor = datasets.encode(ds.volume(0, 0), ["zlib"])
        self.assertEqual(desc["dtype"], "uint16")
        self.assertNotEqual(tensor.dtype.byteorder, ">")

    def test_a_plain_mrc_file_is_one_z_stack(self):
        path = self.path("tomo.mrc")
        write_mrc(path, planes=5, deltavision=False, value=lambda w, t, z, yy, xx: z + yy + xx)
        ds = datasets.open_dataset(path)
        m = ds.meta()
        self.assertEqual(m["name"], "tomo")
        self.assertEqual(m["format"], "mrc")
        self.assertEqual(m["dims"], [1, 1, 5, 6, 8])
        self.assertFalse(m["dims_from_metadata"])
        self.assertEqual(m["channels"], [])
        np.testing.assert_allclose(m["voxel_um"], [0.1, 0.1, 0.3], rtol=1e-5)
        self.assertEqual(float(ds.plane(0, 0, 4)[5, 7]), 16.0)

    def test_a_page_order_from_the_application_wins_over_the_header(self):
        path = self.path("order.dv")
        write_mrc(path, waves=2, times=3, planes=4, sequence=0, mode=1)
        # c and t given with the default order: the header's sequence still maps the sections
        ds = datasets.open_dataset(path, {"page_order": "czt", "c": 2, "t": 3})
        self.assertEqual(ds.meta()["dims"], [2, 3, 4, 6, 8])
        self.assertFalse(ds.meta()["dims_from_metadata"])
        yy, xx = np.mgrid[0:6, 0:8]
        np.testing.assert_array_equal(ds.plane(1, 2, 3), stamp(1, 2, 3, yy, xx))
        # a layout the sections do not divide into: the sections are read as z
        odd = datasets.open_dataset(path, {"page_order": "czt", "c": 5})
        self.assertEqual(odd.meta()["dims"], [1, 1, 24, 6, 8])

    def test_files_that_are_not_stacks_say_why(self):
        short = self.path("short.dv")
        with open(short, "wb") as f:
            f.write(b"DeltaVision?")
        with self.assertRaises(datasets.DatasetError) as e:
            datasets.open_dataset(short)
        self.assertIn("1024 bytes", str(e.exception))
        truncated = self.path("trunc.dv")
        write_mrc(truncated, planes=4)
        with open(truncated, "r+b") as f:
            f.truncate(1024 + 100)
        with self.assertRaises(datasets.DatasetError) as e:
            datasets.open_dataset(truncated)
        self.assertIn("expected, the file has 1124", str(e.exception))
        unknown = self.path("mode.dv")
        write_mrc(unknown, mode=2)
        with open(unknown, "r+b") as f:
            f.seek(12)
            f.write((12).to_bytes(4, "little", signed=True))
        with self.assertRaises(datasets.DatasetError) as e:
            datasets.open_dataset(unknown)
        self.assertIn("pixel mode 12 is not read", str(e.exception))


# --- TIFF: SIRIUS's own reader (the sirius package) ------------------------------------------
#
# The worker reads TIFF only through the sirius extension. Where it is not
# built for this Python, a fake module with the same API surface
# (inspect_tiff, TiffFile with read_pages / read_region / gpu_decodable,
# Device, built_with_nvtiff, cuda_available) stands in, so the layout, the
# region and pyramid arithmetic, the device choice and the error without the
# package are covered; TestTiffRealSirius runs the same reads through the
# real extension when it imports.


class _FakeDevice:
    def __init__(self, kind: str = "cpu", index: int = 0):
        self.kind, self.index = kind, index

    @staticmethod
    def cpu():
        return _FakeDevice("cpu")

    @staticmethod
    def cuda(index: int = 0):
        return _FakeDevice("cuda", index)

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    def __repr__(self) -> str:
        return self.kind if self.kind == "cpu" else f"cuda:{self.index}"


_FAKE_CPU = _FakeDevice.cpu()


class _FakeBuffer:
    """What a CUDA read returns: pixels in device memory, .numpy() copies them home."""

    def __init__(self, a: np.ndarray, device: _FakeDevice):
        self._a, self.device = a, device

    def numpy(self) -> np.ndarray:
        return self._a.copy()


class _FakeObject:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _fake_sirius(nvtiff: bool = False, cuda: bool = False, paged_regions: bool = True, tags: bool = True):
    """A stand-in `sirius` module. Files are registered with add(path, levels,
    ...), levels[0] the (pages, y, x) stack -- (pages, samples, y, x) with
    several samples per pixel -- and levels[k] its k-th reduction; every read
    is logged in .calls as (what, level, first, count, x, y, w, h, device)."""
    m = types.ModuleType("sirius")
    m.__version__ = "0.1.0-fake"
    m.files = {}
    m.calls = []
    m.Device = _FakeDevice
    m.built_with_nvtiff = lambda: nvtiff
    m.cuda_available = lambda: cuda
    m.cuda_device_count = lambda: 1 if cuda else 0

    def add(path, levels, description="", xres=0.0, yres=0.0, unit=2, gpu_ok=True, error=None, photometric=1):
        with open(path, "wb") as f:
            f.write(b"II*\0")   # the worker only needs the file to exist
        m.files[os.path.abspath(path)] = {"levels": [np.asarray(lv) for lv in levels], "description": description,
                                          "xres": xres, "yres": yres, "unit": unit, "gpu_ok": gpu_ok, "error": error,
                                          "photometric": photometric}

    m.add = add

    def info_of(spec):
        a = spec["levels"][0]
        samples = a.shape[1] if a.ndim == 4 else 1
        page = {"width": a.shape[-1], "height": a.shape[-2], "dtype": a.dtype, "samples_per_pixel": samples,
                "photometric": spec["photometric"], "decodable": True, "unsupported": ""}
        if tags:
            page.update(description=spec["description"], x_resolution=spec["xres"], y_resolution=spec["yres"],
                        resolution_unit=spec["unit"])
        levels = [_FakeObject(width=lv.shape[-1], height=lv.shape[-2], ifds=list(range(lv.shape[0])))
                  for lv in spec["levels"]]
        return _FakeObject(page_count=a.shape[0], height=a.shape[-2], width=a.shape[-1], dtype=a.dtype,
                           uniform_pages=True, levels=levels, page=lambda i: _FakeObject(**page))

    class TiffFile:
        def __init__(self, path):
            spec = m.files.get(os.path.abspath(path))
            if spec is None:
                raise RuntimeError(f"Cannot open TIFF: {path}")
            if spec["error"]:
                raise RuntimeError(spec["error"])
            self._s = spec
            self.info = info_of(spec)

        def gpu_decodable(self, device=None):
            return (bool(self._s["gpu_ok"]), "" if self._s["gpu_ok"] else "compression 7 (JPEG) is not decoded by nvTIFF")

        @staticmethod
        def _out(a, device):
            a = np.ascontiguousarray(a)
            return _FakeBuffer(a, device) if device.is_cuda else a.copy()

        @staticmethod
        def _samples(a, first_sample, samples):
            """The chosen samples of (pages, s, y, x), as the extension returns them."""
            if a.ndim != 4:
                return a
            n = samples or a.shape[1] - first_sample
            out = a[:, first_sample:first_sample + n]
            return out[:, 0] if n == 1 else out

        def read_pages(self, first, count, dtype=None, device=_FAKE_CPU, allow_cpu_fallback=True, pinned=False,
                       stream=None, first_sample=0, samples=0):
            a = self._s["levels"][0]
            if count == 0 or first + count > a.shape[0]:
                raise IndexError(f"Pages [{first}, {first + count}) of {a.shape[0]}")
            m.calls.append(("pages", 0, first, count, 0, 0, a.shape[-1], a.shape[-2], device.kind))
            return self._out(self._samples(a[first:first + count], first_sample, samples), device)

        def _region(self, x, y, width, height, level, device, first, count, what, first_sample=0, samples=0):
            a = self._s["levels"][level]
            count = count or a.shape[0] - first
            w = width or a.shape[-1] - x
            h = height or a.shape[-2] - y
            if x + w > a.shape[-1] or y + h > a.shape[-2] or first + count > a.shape[0]:
                raise ValueError("region out of bounds")
            m.calls.append((what, level, first, count, x, y, w, h, device.kind))
            return self._out(self._samples(a[first:first + count, ..., y:y + h, x:x + w], first_sample, samples), device)

        if paged_regions:
            def read_region(self, x, y, width=0, height=0, level=0, dtype=None, device=_FAKE_CPU,
                            allow_cpu_fallback=True, pinned=False, stream=None, first=0, count=0, first_sample=0,
                            samples=0):
                return self._region(x, y, width, height, level, device, first, count, "region", first_sample, samples)
        else:
            # an extension built before read_region took first / count: every page
            def read_region(self, x, y, width=0, height=0, level=0, dtype=None, device=_FAKE_CPU,
                            allow_cpu_fallback=True, pinned=False, stream=None):
                return self._region(x, y, width, height, level, device, 0, 0, "region-all")

    m.TiffFile = TiffFile
    m.inspect_tiff = lambda path: TiffFile(path).info
    return m


class _WithSirius(_Files):
    """Runs with sys.modules["sirius"] replaced (a fake, or None: not importable)."""

    fake_options: dict = {}

    def install(self, module):
        sys.modules["sirius"] = module
        return module

    def setUp(self):
        super().setUp()
        self._saved = sys.modules.get("sirius", _MISSING)
        self.fake = self.install(_fake_sirius(**self.fake_options))

    def tearDown(self):
        super().tearDown()
        if self._saved is _MISSING:
            sys.modules.pop("sirius", None)
        else:
            sys.modules["sirius"] = self._saved

    def reads(self, what=None):
        return [c for c in self.fake.calls if what is None or c[0] == what]


_MISSING = object()

IMAGEJ = "ImageJ=1.54f\nimages=12\nchannels=2\nslices=3\nframes=2\nhyperstack=true\nmode=composite\nunit=micron\n" \
         "spacing=0.5\nfinterval=2.0\nloop=false\n"
OME = ('<?xml version="1.0" encoding="UTF-8"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
       '<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16" SizeX="5" SizeY="4" SizeZ="3" '
       'SizeC="2" SizeT="1" PhysicalSizeX="0.2" PhysicalSizeY="0.2" PhysicalSizeZ="1.0">'
       '<Channel ID="Channel:0:0" Name="DAPI" EmissionWavelength="460" SamplesPerPixel="1"/>'
       '<Channel ID="Channel:0:1" Name="GFP" SamplesPerPixel="1"/>'
       '<TiffData/></Pixels></Image></OME>')


class TestTiff(_WithSirius):
    def test_plain_pages_follow_the_page_order(self):
        # 2 channels x 3 planes, ImageJ's czt: channel fastest
        pages = np.arange(6 * 4 * 5, dtype=np.uint16).reshape(6, 4, 5)
        self.fake.add(self.path("p.tif"), [pages])
        ds = datasets.open_dataset(self.path("p.tif"), {"page_order": "czt", "c": 2})
        self.assertEqual(ds.meta()["dims"], [2, 1, 3, 4, 5])
        self.assertEqual(ds.meta()["format"], "tiff")
        self.assertFalse(ds.meta()["dims_from_metadata"])
        np.testing.assert_array_equal(ds.plane(1, 0, 2), pages[1 + 2 * 2])
        # without options the pages are z
        plain = datasets.open_dataset(self.path("p.tif"))
        self.assertEqual(plain.meta()["dims"], [1, 1, 6, 4, 5])
        # counts the pages do not divide into: pages as z, as the application reads them
        odd = datasets.open_dataset(self.path("p.tif"), {"page_order": "czt", "c": 4})
        self.assertEqual(odd.meta()["dims"], [1, 1, 6, 4, 5])

    def test_an_imagej_hyperstack_says_its_axes(self):
        data = np.arange(2 * 3 * 2 * 4 * 5, dtype=np.uint16).reshape(2, 3, 2, 4, 5)   # t z c y x
        self.fake.add(self.path("h.tif"), [data.reshape(12, 4, 5)], IMAGEJ, xres=10.0, yres=10.0, unit=1)
        ds = datasets.open_dataset(self.path("h.tif"))
        m = ds.meta()
        self.assertEqual(m["dims"], [2, 2, 3, 4, 5])
        self.assertEqual(m["format"], "imagej-tiff")
        self.assertTrue(m["dims_from_metadata"])
        self.assertAlmostEqual(m["voxel_um"][2], 0.5)
        self.assertAlmostEqual(m["voxel_um"][0], 0.1, places=6)
        self.assertAlmostEqual(m["frame_interval_s"], 2.0)
        np.testing.assert_array_equal(ds.plane(1, 0, 2), data[0, 2, 1])
        self.fake.calls.clear()
        np.testing.assert_array_equal(ds.volume(0, 1), data[1, :, 0])
        # the z planes of a channel are every 2nd page: one range of 5 pages, not 3 reads
        self.assertEqual([c[2:4] for c in self.reads()], [(6, 5)])

    def test_an_ome_tiff_gives_channels_voxels_and_page_order(self):
        data = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)   # c z y x (XYZCT: z fastest)
        self.fake.add(self.path("o.ome.tif"), [data.reshape(6, 4, 5)], OME)
        ds = datasets.open_dataset(self.path("o.ome.tif"))
        m = ds.meta()
        self.assertEqual(m["name"], "o")
        self.assertEqual(m["format"], "ome-tiff")
        self.assertEqual(m["dims"], [2, 1, 3, 4, 5])
        np.testing.assert_allclose(m["voxel_um"], [0.2, 0.2, 1.0])
        self.assertEqual([c["name"] for c in m["channels"]], ["DAPI", "GFP"])
        self.assertAlmostEqual(m["channels"][0]["wavelength_nm"], 460.0, places=6)
        self.assertNotIn("wavelength_nm", m["channels"][1])
        self.fake.calls.clear()
        np.testing.assert_array_equal(ds.volume(1, 0), data[1])
        self.assertEqual([c[:4] for c in self.reads()], [("pages", 0, 3, 3)])   # one page range
        # an explicit page order wins over the metadata, as in the application
        ds2 = datasets.open_dataset(self.path("o.ome.tif"), {"page_order": "zct", "c": 0, "t": 0, "z": 6})
        self.assertEqual(ds2.meta()["dims"], [1, 1, 6, 4, 5])
        self.assertFalse(ds2.meta()["dims_from_metadata"])

    def test_a_zoomed_in_view_reads_only_its_region(self):
        stack = np.random.default_rng(5).integers(0, 4000, size=(3, 64, 80)).astype(np.uint16)
        self.fake.add(self.path("big.tif"), [stack])
        ds = datasets.open_dataset(self.path("big.tif"))
        got = ds.view("xy", 0, 0, 1, 1, [10, 20, 16, 8])
        np.testing.assert_array_equal(got, stack[1, 20:28, 10:26])
        self.assertEqual(self.reads(), [("region", 0, 1, 1, 10, 20, 16, 8, "cpu")])
        # reduced: the region's pixels, then the block means
        np.testing.assert_array_equal(ds.view("xy", 0, 0, 2, 2, [6, 2, 20, 30]),
                                      datasets.reduce_blocks(stack[2, 2:32, 6:26], (2, 2)))
        # a region past the edge is clipped; one wholly outside is refused
        np.testing.assert_array_equal(ds.view("xy", 0, 0, 0, 1, [70, 60, 50, 50]), stack[0, 60:64, 70:80])
        with self.assertRaises(datasets.DatasetError):
            ds.view("xy", 0, 0, 0, 1, [100, 0, 4, 4])
        # the whole plane is one page
        self.fake.calls.clear()
        np.testing.assert_array_equal(ds.view("xy", 0, 0, 0, 4), datasets.reduce_blocks(stack[0], (4, 4)))
        self.assertEqual([c[:4] for c in self.reads()], [("pages", 0, 0, 1)])
        # once the volume is in memory, a view is cut from it
        ds.volume(0, 0)
        self.fake.calls.clear()
        np.testing.assert_array_equal(ds.view("xy", 0, 0, 1, 1, [10, 20, 16, 8]), stack[1, 20:28, 10:26])
        self.assertEqual(self.reads(), [])

    def test_a_reduced_view_comes_from_a_pyramid_level(self):
        rng = np.random.default_rng(6)
        full = rng.integers(0, 4000, size=(2, 64, 80)).astype(np.uint16)
        half = rng.integers(0, 4000, size=(2, 32, 40)).astype(np.uint16)       # level 1: the writer's own reduction
        quarter = rng.integers(0, 4000, size=(2, 16, 20)).astype(np.uint16)    # level 2
        self.fake.add(self.path("pyr.tif"), [full, half, quarter])
        ds = datasets.open_dataset(self.path("pyr.tif"))
        self.assertEqual(ds.meta()["dims"], [1, 1, 2, 64, 80])
        # factor 8: level 2 (scale 4), reduced by 2 more
        np.testing.assert_array_equal(ds.view("xy", 0, 0, 1, 8), datasets.reduce_blocks(quarter[1], (2, 2)))
        self.assertEqual(self.reads(), [("region", 2, 1, 1, 0, 0, 20, 16, "cpu")])
        # factor 2 with a region on the level's grid: level 1's pixels themselves
        self.fake.calls.clear()
        np.testing.assert_array_equal(ds.view("xy", 0, 0, 0, 2, [8, 4, 32, 16]), half[0, 2:10, 4:20])
        self.assertEqual(self.reads(), [("region", 1, 0, 1, 4, 2, 16, 8, "cpu")])
        # off the grid, or a factor no level divides: full resolution
        self.fake.calls.clear()
        np.testing.assert_array_equal(ds.view("xy", 0, 0, 0, 2, [1, 0, 32, 16]),
                                      datasets.reduce_blocks(full[0, 0:16, 1:33], (2, 2)))
        np.testing.assert_array_equal(ds.view("xy", 0, 0, 0, 3), datasets.reduce_blocks(full[0], (3, 3)))
        self.assertTrue(all(c[1] == 0 for c in self.reads()))

    def test_an_extension_without_tags_reads_pages_as_z(self):
        self.install(_fake_sirius(tags=False))
        sys.modules["sirius"].add(self.path("n.tif"), [np.zeros((12, 4, 5), np.uint8)], IMAGEJ)
        m = datasets.open_dataset(self.path("n.tif")).meta()
        self.assertEqual(m["dims"], [1, 1, 12, 4, 5])
        self.assertEqual(m["voxel_um"], [0.0, 0.0, 0.0])

    def test_an_older_extension_crops_whole_pages(self):
        fake = self.install(_fake_sirius(paged_regions=False))
        full = np.random.default_rng(7).integers(0, 255, size=(3, 32, 40)).astype(np.uint8)
        fake.add(self.path("old.tif"), [full, full[:, ::2, ::2].copy()])
        ds = datasets.open_dataset(self.path("old.tif"))
        np.testing.assert_array_equal(ds.view("xy", 0, 0, 2, 1, [4, 6, 10, 12]), full[2, 6:18, 4:14])
        # a pyramid level of one page cannot be read there: full resolution, reduced here
        np.testing.assert_array_equal(ds.view("xy", 0, 0, 1, 2), datasets.reduce_blocks(full[1], (2, 2)))
        self.assertEqual({c[0] for c in fake.calls}, {"pages"})

    def test_a_file_the_reader_refuses_says_why(self):
        self.fake.add(self.path("zstd.tif"), [np.zeros((1, 2, 2), np.uint8)],
                      error="compression 50000 (ZSTD) is not built into this SIRIUS's libtiff")
        with self.assertRaises(datasets.DatasetError) as e:
            datasets.open_dataset(self.path("zstd.tif"))
        self.assertIn("zstd.tif", str(e.exception))
        self.assertIn("ZSTD", str(e.exception))

    def test_an_rgb_tiff_opens_with_three_channels(self):
        # (pages, samples, y, x): 2 z planes of RGB
        rgb = np.arange(2 * 3 * 6 * 8, dtype=np.uint8).reshape(2, 3, 6, 8)
        self.fake.add(self.path("rgb.tif"), [rgb], photometric=2)
        ds = datasets.open_dataset(self.path("rgb.tif"))
        meta = ds.meta()
        self.assertEqual(meta["dims"], [3, 1, 2, 6, 8])
        self.assertTrue(meta["rgb"])
        for c in range(3):
            np.testing.assert_array_equal(ds.plane(c, 0, 1), rgb[1, c])
            np.testing.assert_array_equal(ds.volume(c, 0), rgb[:, c])
        np.testing.assert_array_equal(ds.view("xy", 2, 0, 0, 1, [2, 1, 4, 3]), rgb[0, 2, 1:4, 2:6])
        # an OME-TIFF's SizeC counts the samples; the application's page order counts channels
        ome = ('<OME><Image><Pixels DimensionOrder="XYCZT" SizeX="8" SizeY="6" SizeZ="2" SizeC="3" SizeT="1">'
               '<Channel SamplesPerPixel="3"/></Pixels></Image></OME>')
        self.fake.add(self.path("rgb.ome.tif"), [rgb], description=ome, photometric=2)
        meta = datasets.open_dataset(self.path("rgb.ome.tif")).meta()
        self.assertEqual(meta["dims"], [3, 1, 2, 6, 8])
        self.assertTrue(meta["dims_from_metadata"])
        given = datasets.open_dataset(self.path("rgb.tif"), {"page_order": "czt", "c": 3, "t": 1, "z": 2}).meta()
        self.assertEqual(given["dims"], [3, 1, 2, 6, 8])
        # four samples are four channels, not an RGB merge
        self.fake.add(self.path("rgba.tif"), [np.zeros((1, 4, 2, 2), np.uint8)], photometric=2)
        meta = datasets.open_dataset(self.path("rgba.tif")).meta()
        self.assertEqual(meta["dims"][0], 4)
        self.assertFalse(meta["rgb"])


class TestTiffDevice(_WithSirius):
    fake_options = {"nvtiff": True, "cuda": True}

    def setUp(self):
        super().setUp()
        self.stack = np.arange(4 * 6 * 7, dtype=np.uint16).reshape(4, 6, 7)
        self.fake.add(self.path("g.tif"), [self.stack])

    def device_of_reads(self):
        return {c[-1] for c in self.fake.calls}

    def test_a_cuda_device_decodes_with_nvtiff(self):
        ds = datasets.open_dataset(self.path("g.tif"))
        for asked, ran in (("cuda", "cuda"), ("cuda:0", "cuda"), ("auto", "cuda"), (None, "cuda"), ("cpu", "cpu"),
                           ("cuda:3", "cpu")):   # a GPU this node does not have: the CPU decoder
            with self.subTest(device=asked):
                self.fake.calls.clear()
                got = ds.plane(0, 0, 2, asked)
                self.assertIsInstance(got, np.ndarray)   # home from GPU memory
                np.testing.assert_array_equal(got, self.stack[2])
                self.assertEqual(self.device_of_reads(), {ran})
        self.fake.calls.clear()
        np.testing.assert_array_equal(ds.view("xy", 0, 0, 1, 1, [1, 1, 3, 2], device="cuda"), self.stack[1, 1:3, 1:4])
        self.assertEqual(self.device_of_reads(), {"cuda"})
        self.fake.calls.clear()
        v = datasets.read_ref({"path": self.path("g.tif"), "c": 0, "t": 0}, "cuda")
        self.assertEqual(v.dtype, np.float32)
        np.testing.assert_array_equal(v, self.stack.astype(np.float32))
        self.assertEqual(self.device_of_reads(), {"cuda"})

    def test_a_file_nvtiff_cannot_decode_goes_to_the_cpu(self):
        self.fake.add(self.path("jpeg.tif"), [self.stack], gpu_ok=False)
        ds = datasets.open_dataset(self.path("jpeg.tif"))
        np.testing.assert_array_equal(ds.plane(0, 0, 1, "cuda"), self.stack[1])
        self.assertEqual(self.device_of_reads(), {"cpu"})

    def test_without_nvtiff_or_a_gpu_the_cpu_decodes(self):
        for opts in ({"nvtiff": False, "cuda": True}, {"nvtiff": True, "cuda": False}):
            with self.subTest(**opts):
                datasets.forget_all()
                fake = self.install(_fake_sirius(**opts))
                fake.add(self.path("g.tif"), [self.stack])
                ds = datasets.open_dataset(self.path("g.tif"))
                np.testing.assert_array_equal(ds.plane(0, 0, 3, "cuda"), self.stack[3])
                self.assertEqual({c[-1] for c in fake.calls}, {"cpu"})
                self.assertEqual(datasets.tiff_reader("cuda"), {"sirius": "0.1.0-fake", "nvtiff": False})

    def test_tiff_reader_says_what_decodes(self):
        self.assertEqual(datasets.tiff_reader("cuda"), {"sirius": "0.1.0-fake", "nvtiff": True})
        self.assertEqual(datasets.tiff_reader("auto"), {"sirius": "0.1.0-fake", "nvtiff": True})
        self.assertEqual(datasets.tiff_reader("cpu"), {"sirius": "0.1.0-fake", "nvtiff": False})


class TestWithoutSirius(_WithSirius):
    def setUp(self):
        super().setUp()
        self.install(None)   # `import sirius` raises ImportError

    def test_tiff_needs_the_sirius_package_and_says_so(self):
        with open(self.path("s.tif"), "wb") as f:
            f.write(b"II*\0")
        with self.assertRaises(datasets.DatasetError) as e:
            datasets.open_dataset(self.path("s.tif"))
        self.assertEqual(str(e.exception), datasets.TIFF_NEEDS_SIRIUS)
        self.assertIn("pip install <checkout>", str(e.exception))
        self.assertIn("app/python/slurm/README.md", str(e.exception))
        self.assertEqual(datasets.tiff_reader("cuda"), {"sirius": None, "nvtiff": False})

    def test_npy_needs_nothing(self):
        np.save(self.path("a.npy"), np.ones((3, 4), np.float32))
        self.assertEqual(datasets.open_dataset(self.path("a.npy")).meta()["dims"], [1, 1, 1, 3, 4])


@unittest.skipUnless(REAL_SIRIUS is not None, "the sirius extension is not importable in this Python")
class TestTiffRealSirius(_Files):
    """The same reads through the real extension (CI installs it; locally:
    pip install -e . or PYTHONPATH at a build's bindings)."""

    def setUp(self):
        super().setUp()
        self._saved = sys.modules.get("sirius", _MISSING)
        sys.modules["sirius"] = REAL_SIRIUS

    def tearDown(self):
        super().tearDown()
        if self._saved is _MISSING:
            sys.modules.pop("sirius", None)
        else:
            sys.modules["sirius"] = self._saved

    def test_pages_regions_and_views(self):
        stack = np.random.default_rng(8).integers(0, 4000, size=(6, 40, 50)).astype(np.uint16)
        REAL_SIRIUS.write_tiff(self.path("r.tif"), stack)
        ds = datasets.open_dataset(self.path("r.tif"), {"page_order": "czt", "c": 2})
        self.assertEqual(ds.meta()["dims"], [2, 1, 3, 40, 50])
        self.assertEqual(ds.meta()["dtype"], "uint16")
        np.testing.assert_array_equal(ds.plane(1, 0, 2, "cpu"), stack[5])
        np.testing.assert_array_equal(ds.view("xy", 0, 0, 1, 1, [7, 9, 20, 11], device="cpu"), stack[2, 9:20, 7:27])
        np.testing.assert_array_equal(ds.view("xy", 1, 0, 0, 3, device="cpu"), datasets.reduce_blocks(stack[1], (3, 3)))
        np.testing.assert_array_equal(ds.volume(0, 0, "cpu"), stack[0::2])
        np.testing.assert_array_equal(ds.volume(1, 0, "auto"), stack[1::2])   # the GPU when this build has nvTIFF
        reader = datasets.tiff_reader("cpu")
        self.assertTrue(reader["sirius"])
        self.assertFalse(reader["nvtiff"])

    @unittest.skipUnless(HAVE_TIFFFILE, "writing an ImageJ hyperstack for the test needs tifffile")
    def test_an_imagej_hyperstack_says_its_axes(self):
        if not hasattr(REAL_SIRIUS.TiffImageInfo, "description"):
            self.skipTest("this build of the extension does not hand over the ImageDescription yet")
        data = np.arange(2 * 3 * 2 * 4 * 5, dtype=np.uint16).reshape(2, 3, 2, 4, 5)   # t z c y x
        tifffile.imwrite(self.path("h.tif"), data, imagej=True, metadata={"axes": "TZCYX", "spacing": 0.5},
                         resolution=(1 / 0.1, 1 / 0.1))
        ds = datasets.open_dataset(self.path("h.tif"))
        m = ds.meta()
        self.assertEqual(m["dims"], [2, 2, 3, 4, 5])
        self.assertTrue(m["dims_from_metadata"])
        self.assertAlmostEqual(m["voxel_um"][2], 0.5)
        self.assertAlmostEqual(m["voxel_um"][0], 0.1, places=4)
        np.testing.assert_array_equal(ds.plane(1, 0, 2), data[0, 2, 1])
        np.testing.assert_array_equal(ds.volume(0, 1), data[1, :, 0])

    @unittest.skipUnless(HAVE_TIFFFILE, "writing an RGB TIFF for the test needs tifffile")
    def test_an_rgb_tiff_opens_with_three_channels(self):
        if not hasattr(REAL_SIRIUS, "parse_tiff_metadata"):
            self.skipTest("this build of the extension reads one sample per pixel")
        rgb = np.random.default_rng(3).integers(0, 255, size=(3, 30, 20, 3)).astype(np.uint8)   # z y x s
        for planar in ("contig", "separate"):
            data = rgb if planar == "contig" else np.moveaxis(rgb, -1, 1)
            tifffile.imwrite(self.path(f"{planar}.tif"), data, photometric="rgb", planarconfig=planar,
                             tile=(16, 16) if planar == "separate" else None)
            ds = datasets.open_dataset(self.path(f"{planar}.tif"))
            m = ds.meta()
            self.assertEqual(m["dims"], [3, 1, 3, 30, 20])
            self.assertTrue(m["rgb"])
            for c in range(3):
                np.testing.assert_array_equal(ds.plane(c, 0, 2, "cpu"), rgb[2, :, :, c])
                np.testing.assert_array_equal(ds.volume(c, 0, "cpu"), rgb[:, :, :, c])
            np.testing.assert_array_equal(ds.view("xy", 1, 0, 1, 1, [3, 4, 10, 12], device="cpu"),
                                          rgb[1, 4:16, 3:13, 1])


class TestOverTheSocket(_Files):
    token = "tok"

    def setUp(self):
        super().setUp()
        self.server = WorkerServer("127.0.0.1", 0, self.token, "cpu", max_clients=4)
        self.port = self.server.bind()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.stop()
        self.thread.join(timeout=10)
        super().tearDown()

    def client(self):
        c = _Client(self.port, self.token)
        header = c.hello()
        self.assertEqual(header["type"], "result")
        c.caps = header["result"]
        return c

    def test_hello_names_the_dataset_methods_and_encodings(self):
        c = self.client()
        try:
            caps = c.caps
            for m in ("dataset_info", "dataset_read", "dataset_view", "dataset_stats"):
                self.assertIn(m, caps["methods"])
            self.assertIn("zlib", caps["encodings"])
            self.assertEqual(caps["max_clients"], 4)
            # who reads TIFF here; tifffile is not it
            self.assertEqual(set(caps["tiff_reader"]), {"sirius", "nvtiff"})
            self.assertEqual(caps["tiff_reader"], datasets.tiff_reader("cpu"))
            self.assertNotIn("tifffile", caps)
        finally:
            c.close()

    def test_dataset_methods(self):
        a = (np.random.default_rng(4).normal(500, 30, size=(1, 2, 6, 32, 40))).astype(np.uint16)
        np.save(self.path("d.npy"), a)
        c = self.client()
        try:
            h, _ = c.call("dataset_info", {"path": self.path("d.npy")})
            self.assertEqual(h["result"]["dims"], [1, 2, 6, 32, 40])
            h, t = c.call("dataset_view", {"path": self.path("d.npy"), "kind": "xy", "c": 0, "t": 1, "index": 2,
                                           "factor": 4, "accept": ["zlib"]})
            self.assertEqual(h["type"], "result", h)
            got = datasets.decode(h["result"], t["data"])
            np.testing.assert_array_equal(got, datasets.reduce_blocks(a[0, 1, 2], (4, 4)))
            h, t = c.call("dataset_read", {"path": self.path("d.npy"), "c": 0, "t": 0, "z": 5})
            self.assertEqual(h["result"]["encoding"], "raw")
            np.testing.assert_array_equal(t["data"], a[0, 0, 5])
            h, _ = c.call("dataset_stats", {"path": self.path("d.npy"), "c": 0, "t": 0})
            self.assertLess(h["result"]["lo"], h["result"]["hi"])
            h, _ = c.call("dataset_info", {"path": self.path("missing.tif")})
            self.assertEqual(h["type"], "error")
            self.assertIn("missing.tif", h["message"])
        finally:
            c.close()

    def test_a_tiff_over_the_socket(self):
        saved = sys.modules.get("sirius", _MISSING)
        fake = sys.modules["sirius"] = _fake_sirius(nvtiff=True, cuda=True)
        try:
            stack = np.arange(3 * 16 * 20, dtype=np.uint16).reshape(3, 16, 20)
            fake.add(self.path("s.tif"), [stack])
            c = self.client()
            try:
                h, _ = c.call("dataset_info", {"path": self.path("s.tif")})
                self.assertEqual(h["result"]["dims"], [1, 1, 3, 16, 20])
                h, t = c.call("dataset_view", {"path": self.path("s.tif"), "kind": "xy", "c": 0, "t": 0, "index": 2,
                                               "region": [4, 2, 8, 6], "device": "cuda"})
                self.assertEqual(h["type"], "result", h)
                np.testing.assert_array_equal(t["data"], stack[2, 2:8, 4:12])
                self.assertEqual(fake.calls[-1], ("region", 0, 2, 1, 4, 2, 8, 6, "cuda"))
                # the request names the CPU: decoded there
                h, t = c.call("dataset_read", {"path": self.path("s.tif"), "c": 0, "t": 0, "z": 1, "device": "cpu"})
                np.testing.assert_array_equal(t["data"], stack[1])
                self.assertEqual(fake.calls[-1][-1], "cpu")
                # none named: this worker's --device (cpu here)
                c.call("dataset_read", {"path": self.path("s.tif"), "c": 0, "t": 0, "z": 0})
                self.assertEqual(fake.calls[-1][-1], "cpu")
            finally:
                c.close()
            sys.modules["sirius"] = None
            datasets.forget_all()
            c = self.client()
            try:
                h, _ = c.call("dataset_info", {"path": self.path("s.tif")})
                self.assertEqual(h["type"], "error")
                self.assertIn("needs the sirius package", h["message"])
            finally:
                c.close()
        finally:
            if saved is _MISSING:
                sys.modules.pop("sirius", None)
            else:
                sys.modules["sirius"] = saved

    def test_a_run_reads_its_input_by_reference(self):
        a = np.arange(1 * 1 * 2 * 3 * 4, dtype=np.uint16).reshape(1, 1, 2, 3, 4)
        np.save(self.path("in.npy"), a)
        c = self.client()
        try:
            h, out = c.call("run", {"kind": "einsum", "params": {"expr": "ctzyx->ctzyx"},
                                    "input_ref": {"path": self.path("in.npy"), "layout": "ctzyx"}})
            if h["type"] == "error" and "einsum" in h["message"] and "unknown" in h["message"]:
                self.skipTest("the step library has no einsum here")
            self.assertEqual(h["type"], "result", h)
            np.testing.assert_array_equal(out["output"], a.astype(np.float32))
        finally:
            c.close()

    def test_clients_are_served_at_once_and_a_disconnect_cancels_only_its_own_job(self):
        np.save(self.path("x.npy"), np.ones((1, 1, 1, 4, 4), np.uint8))
        started = threading.Event()
        release = threading.Event()

        def slow(progress, cancel):
            started.set()
            while not cancel.is_set() and not release.is_set():
                time.sleep(0.01)
            return {}, None

        a = self.client()
        b = self.client()   # a second hello is answered while a is connected
        try:
            # a job owned by connection a's owner token cannot be cancelled by b
            owner_a = object()
            sent = []
            self.server._start_job(1, "slow", lambda h, t=None: sent.append(h), slow, owner_a)
            self.assertTrue(started.wait(5))
            self.server._cancel(None, object())   # another connection
            job = self.server._current_job()
            self.assertIsNotNone(job)
            self.assertFalse(job["cancel"].is_set())
            # b reads a dataset while the job runs
            h, _ = b.call("dataset_info", {"path": self.path("x.npy")})
            self.assertEqual(h["type"], "result")
            self.server._cancel(None, owner_a)
            self.assertTrue(job["cancel"].is_set())
            job["thread"].join(timeout=5)
        finally:
            release.set()
            a.close()
            b.close()

    def test_one_client_too_many_is_told_so(self):
        clients = [self.client() for _ in range(4)]
        try:
            # told once it has authenticated: an anonymous peer takes no slot
            extra = _Client(self.port, self.token)
            header = extra.hello()
            self.assertEqual(header["type"], "error")
            self.assertIn("busy", header["message"])
            extra.close()
        finally:
            for c in clients:
                c.close()


if __name__ == "__main__":
    unittest.main()
