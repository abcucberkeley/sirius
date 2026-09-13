"""The bundle registry: what the application lists models from.

No `latents` here on purpose. Listing a directory must not need the package or
the weights: the worker keeps one bundle resident, and loading every file to
draw a list would evict it and read gigabytes to fill a dialog. So these tests
build zip files that look like bundles and check the listing on its own terms.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import unittest
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sirius_worker import foundation, protocol  # noqa: E402

MANIFEST = {
    "name": "nuclei-5d",
    "task": "track",
    "patch": [4, 16, 16],
    "crop": [8, 64, 64],
    "voxel_size": [0.75, 0.15, 0.15],
    "peak_threshold": 0.42,
    "min_separation_um": 1.5,
    "channels": ["dapi"],
    "notes": "validated on held-out embryos",
}


def write_bundle(directory: str, name: str, manifest: dict | None = MANIFEST) -> str:
    path = os.path.join(directory, name)
    with zipfile.ZipFile(path, "w") as z:
        if manifest is not None:
            z.writestr("manifest.json", json.dumps(manifest))
        z.writestr("weights.bin", b"\x00" * 16)
    return path


class ListBundles(unittest.TestCase):
    def test_lists_ltb_files_with_their_manifest(self):
        with tempfile.TemporaryDirectory() as box:
            write_bundle(box, "nuclei.ltb")
            got = foundation.list_bundles(box)
            self.assertEqual(len(got), 1)
            one = got[0]
            self.assertEqual(one["file"], "nuclei.ltb")
            self.assertEqual(one["name"], "nuclei-5d")
            self.assertEqual(one["task"], "track")
            self.assertEqual(one["voxel_um"], [0.75, 0.15, 0.15])
            self.assertAlmostEqual(one["peak_threshold"], 0.42)
            self.assertAlmostEqual(one["min_separation_um"], 1.5)
            self.assertTrue(one["manifest"])
            self.assertTrue(os.path.isabs(one["path"]))
            self.assertGreater(one["size_bytes"], 0)

    def test_ignores_everything_that_is_not_a_bundle(self):
        with tempfile.TemporaryDirectory() as box:
            write_bundle(box, "a.ltb")
            open(os.path.join(box, "notes.txt"), "w").close()
            open(os.path.join(box, "model.pt"), "w").close()
            os.mkdir(os.path.join(box, "sub.ltb"))            # a directory, not a bundle
            got = foundation.list_bundles(box)
            self.assertEqual([b["file"] for b in got], ["a.ltb"])

    def test_a_bundle_whose_manifest_cannot_be_read_is_still_listed(self):
        # A file that is there and unreadable is something the user has to see:
        # hiding it would look like the bundle is missing.
        with tempfile.TemporaryDirectory() as box:
            with open(os.path.join(box, "broken.ltb"), "wb") as f:
                f.write(b"not a zip at all")
            write_bundle(box, "nameless.ltb", manifest=None)
            got = foundation.list_bundles(box)
            self.assertEqual([b["file"] for b in got], ["broken.ltb", "nameless.ltb"])
            self.assertFalse(any(b["manifest"] for b in got))
            # the file name stands in for the name the manifest would have given
            self.assertEqual(got[0]["name"], "broken")
            self.assertEqual(got[1]["name"], "nameless")
            self.assertEqual(got[0]["task"], "")

    def test_sorted_by_name_so_the_list_does_not_reshuffle(self):
        with tempfile.TemporaryDirectory() as box:
            for name in ("Zeta.ltb", "alpha.ltb", "Beta.ltb"):
                write_bundle(box, name)
            self.assertEqual([b["file"] for b in foundation.list_bundles(box)],
                             ["alpha.ltb", "Beta.ltb", "Zeta.ltb"])

    def test_a_missing_directory_says_so(self):
        with self.assertRaises(NotADirectoryError):
            foundation.list_bundles(os.path.join(tempfile.gettempdir(), "sirius-no-such-registry"))
        with self.assertRaises(ValueError):
            foundation.list_bundles("")

    def test_an_empty_registry_is_empty_not_an_error(self):
        with tempfile.TemporaryDirectory() as box:
            self.assertEqual(foundation.list_bundles(box), [])

    def test_a_json_that_is_not_a_manifest_is_not_read_as_one(self):
        # Numbers from some other .json shown as calibration would read as fact.
        with tempfile.TemporaryDirectory() as box:
            path = os.path.join(box, "config.ltb")
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("config.json", json.dumps({"lr": 0.001, "name": "run-7"}))
                z.writestr("weights.bin", b"\x00" * 16)
            self.assertEqual(foundation.manifest_of(path), {})
            one = foundation.list_bundles(box)[0]
            self.assertFalse(one["manifest"])
            self.assertEqual(one["name"], "config")        # the file name, not the config's
            self.assertEqual(one["voxel_um"], [])

    def test_manifest_of_reads_neither_weights_nor_latents(self):
        with tempfile.TemporaryDirectory() as box:
            path = write_bundle(box, "one.ltb")
            self.assertEqual(foundation.manifest_of(path)["name"], "nuclei-5d")
            self.assertEqual(foundation.manifest_of(os.path.join(box, "missing.ltb")), {})


class ListBundlesOverTheSocket(unittest.TestCase):
    """The method as the application actually reaches it.

    The application never calls list_bundles in-process: it asks the worker,
    because on a cluster the worker is what can see the bundles. So the thing
    worth testing is the dispatch and the shape of the reply, not the function
    on its own.
    """

    token = "s3cret"

    @classmethod
    def setUpClass(cls):
        from sirius_worker.server import WorkerServer

        cls.box = tempfile.mkdtemp(prefix="sirius-registry-")
        write_bundle(cls.box, "nuclei.ltb")
        cls.server = WorkerServer("127.0.0.1", 0, cls.token, "cpu")
        cls.port = cls.server.bind()
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls.thread.join(timeout=5)
        shutil.rmtree(cls.box, ignore_errors=True)

    def connect(self):
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=30)
        protocol.write_frame(sock, {"id": 1, "type": "request", "method": "hello",
                                    "params": {"token": self.token, "protocol_version": protocol.PROTOCOL_VERSION}})
        header, _ = protocol.read_frame(sock)
        self.assertEqual(header["type"], "result", header)
        return sock, header

    def call(self, sock, rid, method, params):
        protocol.write_frame(sock, {"id": rid, "type": "request", "method": method, "params": params})
        while True:
            header, _ = protocol.read_frame(sock)
            if header.get("type") != "progress":
                return header

    def test_the_worker_answers_with_the_registry(self):
        sock, _ = self.connect()
        try:
            got = self.call(sock, 2, "list_bundles", {"dir": self.box})
            self.assertEqual(got["type"], "result", got)
            self.assertEqual(got["result"]["dir"], self.box)
            bundles = got["result"]["bundles"]
            self.assertEqual(len(bundles), 1)
            self.assertEqual(bundles[0]["name"], "nuclei-5d")
            self.assertEqual(bundles[0]["task"], "track")
        finally:
            sock.close()

    def test_a_directory_that_is_not_there_is_an_error_not_a_crash(self):
        sock, _ = self.connect()
        try:
            got = self.call(sock, 2, "list_bundles", {"dir": os.path.join(self.box, "nope")})
            self.assertEqual(got["type"], "error", got)
            self.assertIn("nope", got["message"])
            # the connection survives it: the dialog can correct the path and retry
            again = self.call(sock, 3, "list_bundles", {"dir": self.box})
            self.assertEqual(again["type"], "result", again)
        finally:
            sock.close()

    def test_hello_advertises_the_method(self):
        # the application checks capabilities before offering the registry
        sock, hello = self.connect()
        try:
            self.assertIn("list_bundles", hello["result"]["methods"])
        finally:
            sock.close()


if __name__ == "__main__":
    unittest.main()
