"""The models registry: what the application's Models… chooser lists.

The application never lists a models folder itself: it asks the worker
(list_bundles, the name the engine on a node relays it by), because on a
cluster the worker is what sees the folder. Listing reads model.json and
README.md only -- no model is imported, nothing resident is evicted. The
folders here are model.json files alone; test_foundation.py runs whole models.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sirius_worker import foundation, protocol  # noqa: E402


def write_model(root: str, name: str, version: str, tasks=("segment",), notes: str = "") -> str:
    folder = os.path.join(root, name, version)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "model.json"), "w") as f:
        json.dump({"format": "latents-model/1", "name": name, "version": version, "tasks": list(tasks),
                   "input": {"channels": 1, "crop": [32, 192, 192], "voxel_um": [0.5, 0.1, 0.1]},
                   "decode": {"fg_threshold": 0.5, "min_voxels": 0}, "notes": notes}, f)
    with open(os.path.join(folder, "weights.safetensors"), "wb") as f:
        f.write(b"\x00" * 64)
    return folder


class ListModels(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="sirius-models-")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_each_version_is_one_row(self):
        write_model(self.root, "coat-sam-s2", "v1", ("segment", "prompt"), "Membrane cells, promptable.")
        write_model(self.root, "coat-conv-r0", "v1")
        got = foundation.list_models([self.root])["models"]
        self.assertEqual([(m["name"], m["version"]) for m in got], [("coat-conv-r0", "v1"), ("coat-sam-s2", "v1")])
        sam = got[1]
        self.assertTrue(sam["promptable"])
        self.assertEqual(sam["description"], "Membrane cells, promptable.")
        self.assertEqual(sam["voxel_um"], [0.1, 0.1, 0.5])
        self.assertEqual(sam["size_bytes"], 64)
        self.assertTrue(os.path.isabs(sam["path"]))
        self.assertEqual(sam["error"], "")

    def test_several_folders_and_one_that_is_not_there(self):
        other = tempfile.mkdtemp(prefix="sirius-models-b-")
        self.addCleanup(shutil.rmtree, other, ignore_errors=True)
        write_model(self.root, "a", "v1")
        write_model(other, "b", "v2")
        got = foundation.list_models([self.root, other, os.path.join(other, "nope")])
        self.assertEqual([m["name"] for m in got["models"]], ["a", "b"])
        self.assertEqual(len(got["errors"]), 1)
        self.assertIn("nope", got["errors"][0])

    def test_an_empty_folder_is_empty_not_an_error(self):
        self.assertEqual(foundation.list_models([self.root]), {"models": [], "errors": []})


class ListModelsOverTheSocket(unittest.TestCase):
    token = "s3cret"

    @classmethod
    def setUpClass(cls):
        from sirius_worker.server import WorkerServer

        cls.box = tempfile.mkdtemp(prefix="sirius-registry-")
        write_model(cls.box, "coat-sam-s2", "v1", ("segment", "prompt"))
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
        header = protocol.client_handshake(sock, self.token, first_id=100)
        self.assertEqual(header["type"], "result", header)
        return sock, header

    def call(self, sock, rid, method, params):
        protocol.write_frame(sock, {"id": rid, "type": "request", "method": method, "params": params})
        while True:
            header, _ = protocol.read_frame(sock)
            if header.get("type") != "progress":
                return header

    def test_the_worker_answers_with_the_models(self):
        sock, _ = self.connect()
        try:
            got = self.call(sock, 2, "list_bundles", {"dirs": [self.box]})
            self.assertEqual(got["type"], "result", got)
            self.assertEqual(got["result"]["dirs"], [self.box])
            models = got["result"]["models"]
            self.assertEqual([(m["name"], m["tasks"]) for m in models], [("coat-sam-s2", ["segment", "prompt"])])
            # `dir` alone, as an older application sends it
            again = self.call(sock, 3, "list_bundles", {"dir": self.box})
            self.assertEqual(len(again["result"]["models"]), 1)
        finally:
            sock.close()

    def test_a_folder_that_is_not_there_is_reported_not_a_crash(self):
        sock, _ = self.connect()
        try:
            got = self.call(sock, 2, "list_bundles", {"dirs": [os.path.join(self.box, "nope")]})
            self.assertEqual(got["type"], "result", got)
            self.assertEqual(got["result"]["models"], [])
            self.assertIn("nope", got["result"]["errors"][0])
        finally:
            sock.close()

    def test_hello_advertises_the_method(self):
        sock, hello = self.connect()
        try:
            self.assertIn("list_bundles", hello["result"]["methods"])
        finally:
            sock.close()


if __name__ == "__main__":
    unittest.main()
