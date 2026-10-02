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
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # app/python

from sirius_worker import datasets, protocol  # noqa: E402
from sirius_worker.server import WorkerServer  # noqa: E402

try:
    import tifffile  # type: ignore

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


@unittest.skipUnless(HAVE_TIFFFILE, "tifffile not importable")
class TestTiff(_Files):
    def test_plain_pages_follow_the_page_order(self):
        # 2 channels x 3 planes, ImageJ's czt: channel fastest
        pages = np.arange(6 * 4 * 5, dtype=np.uint16).reshape(6, 4, 5)
        tifffile.imwrite(self.path("p.tif"), pages, photometric="minisblack")
        ds = datasets.open_dataset(self.path("p.tif"), {"page_order": "czt", "c": 2})
        self.assertEqual(ds.meta()["dims"], [2, 1, 3, 4, 5])
        np.testing.assert_array_equal(ds.plane(1, 0, 2), pages[1 + 2 * 2])
        # without options the pages are z
        plain = datasets.open_dataset(self.path("p.tif"))
        self.assertEqual(plain.meta()["dims"], [1, 1, 6, 4, 5])

    def test_an_imagej_hyperstack_says_its_axes(self):
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

    def test_compressed_pages_are_read_page_by_page(self):
        data = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)
        tifffile.imwrite(self.path("z.tif"), data, compression="zlib", photometric="minisblack")
        ds = datasets.open_dataset(self.path("z.tif"))
        np.testing.assert_array_equal(ds.plane(0, 0, 2), data[2])

    def test_rgb_samples_are_the_channels(self):
        rgb = np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3)
        tifffile.imwrite(self.path("rgb.tif"), rgb, photometric="rgb")
        ds = datasets.open_dataset(self.path("rgb.tif"))
        self.assertEqual(ds.meta()["dims"][:3], [3, 1, 2])
        self.assertTrue(ds.meta()["rgb"])
        np.testing.assert_array_equal(ds.plane(2, 0, 1), rgb[1, :, :, 2])


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
