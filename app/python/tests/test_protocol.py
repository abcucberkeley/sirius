"""Tests of the SIRIUS compute worker: frame codec, the hostile-frame and
authentication rules of app/python/SECURITY.md, and hello / model_info / run /
cancel over a real socket. Torch cases build a tiny scripted model on the fly
and are skipped when torch is not importable.

    python -m unittest discover -s app/python/tests
"""

from __future__ import annotations

import json
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # app/python

from sirius_worker import protocol  # noqa: E402
from sirius_worker.server import WorkerServer  # noqa: E402
from sirius_worker.steps import workbench  # noqa: E402

try:
    import torch  # type: ignore

    HAVE_TORCH = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_TORCH = False


class TestFraming(unittest.TestCase):
    def test_round_trip_preserves_header_and_tensors(self):
        a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        b = np.array([[1, 2], [3, 4]], dtype=np.uint32)
        frame = protocol.encode_frame({"id": 7, "type": "request", "method": "run", "params": {"kind": "x"}},
                                      {"input": a, "labels": b})
        buf = bytearray(frame)
        decoded = protocol.decode_frame(buf)
        self.assertIsNotNone(decoded)
        header, tensors = decoded
        self.assertEqual(len(buf), 0)
        self.assertEqual(header["id"], 7)
        self.assertEqual(header["params"], {"kind": "x"})
        self.assertEqual([t["name"] for t in header["tensors"]], ["input", "labels"])
        self.assertEqual(header["tensors"][0]["dtype"], "float32")
        self.assertEqual(header["tensors"][1]["offset"], a.nbytes)
        np.testing.assert_array_equal(tensors["input"], a)
        np.testing.assert_array_equal(tensors["labels"], b)
        self.assertEqual(tensors["labels"].dtype, np.uint32)

    def test_layout_is_little_endian_length_prefixed(self):
        frame = protocol.encode_frame({"id": 1, "type": "request", "method": "ping"})
        hlen = int.from_bytes(frame[:4], "little")
        header = json.loads(frame[4:4 + hlen])
        plen = int.from_bytes(frame[4 + hlen:12 + hlen], "little")
        self.assertEqual(header["method"], "ping")
        self.assertEqual(plen, 0)
        self.assertEqual(len(frame), 12 + hlen)

    def test_incremental_reader_handles_split_and_joined_frames(self):
        f1 = protocol.encode_frame({"id": 1, "type": "request", "method": "ping"})
        f2 = protocol.encode_frame({"id": 2, "type": "request", "method": "ping"}, {"x": np.ones(5, np.float64)})
        data = f1 + f2
        reader = protocol.FrameReader()
        frames = []
        for i in range(0, len(data), 7):
            frames.extend(reader.feed(data[i:i + 7]))
        self.assertEqual([h["id"] for h, _ in frames], [1, 2])
        np.testing.assert_array_equal(frames[1][1]["x"], np.ones(5))
        self.assertEqual(reader.pending, 0)

    def test_malformed_frames_raise(self):
        bad = bytearray(b"\xff\xff\xff\xff" + b"\x00" * 16)
        with self.assertRaises(protocol.ProtocolError):
            protocol.decode_frame(bad)
        frame = bytearray(protocol.encode_frame({"id": 1, "type": "request"}, {"x": np.zeros(3, np.float32)}))
        frame[4:4 + int.from_bytes(frame[:4], "little")] = frame[4:4 + int.from_bytes(frame[:4], "little")].replace(
            b'"shape":[3]', b'"shape":[4]')
        with self.assertRaises(protocol.ProtocolError):
            protocol.decode_frame(frame)


def raw_frame(header: dict, payload: bytes = b"", header_len: int = -1, payload_len: int = -1) -> bytearray:
    """A frame built by hand, so a test can announce lengths that do not match
    what it actually sends."""
    hb = json.dumps(header).encode("utf-8")
    return bytearray(struct.pack("<I", len(hb) if header_len < 0 else header_len) + hb +
                     struct.pack("<Q", len(payload) if payload_len < 0 else payload_len) + payload)


class TestHostileFrames(unittest.TestCase):
    """Every length in a frame comes from the peer. None of them may size an
    allocation or index the payload before it has been checked -- the same
    cases the C++ decoder is tested against in tests/test_app_rpc.cpp."""

    def test_caps_match_the_cpp_decoder(self):
        self.assertEqual(protocol.MAX_HEADER, 64 << 20)
        self.assertEqual(protocol.MAX_PAYLOAD, 32 << 30)
        self.assertLessEqual(protocol.MAX_PREAUTH_FRAME, 64 << 10)

    def test_an_oversize_header_is_refused(self):
        frame = raw_frame({"id": 1}, header_len=protocol.MAX_HEADER + 1)
        with self.assertRaises(protocol.ProtocolError) as e:
            protocol.decode_frame(frame)
        self.assertIn("header length", str(e.exception))

    def test_an_oversize_payload_is_refused_before_it_is_waited_for(self):
        frame = raw_frame({"id": 1}, payload_len=protocol.MAX_PAYLOAD + 1)
        with self.assertRaises(protocol.ProtocolError) as e:
            protocol.decode_frame(frame)
        self.assertIn("payload length", str(e.exception))
        # ... and a payload length that would wrap a sum on the C++ side
        with self.assertRaises(protocol.ProtocolError):
            protocol.decode_frame(raw_frame({"id": 1}, payload_len=(1 << 64) - 17))

    def test_a_tensor_reaching_past_the_payload_is_refused(self):
        header = {"id": 1, "tensors": [{"name": "x", "dtype": "float32", "shape": [4], "offset": 8, "nbytes": 16}]}
        with self.assertRaises(protocol.ProtocolError) as e:
            protocol.decode_frame(raw_frame(header, b"\x00" * 16))
        self.assertIn("do not fit the payload", str(e.exception))

    def test_a_tensor_offset_that_would_wrap_is_refused(self):
        header = {"id": 1, "tensors": [{"name": "x", "dtype": "float32", "shape": [1],
                                        "offset": (1 << 64) - 3, "nbytes": 4}]}
        with self.assertRaises(protocol.ProtocolError):
            protocol.decode_frame(raw_frame(header, b"\x00" * 4))
        header["tensors"][0].update({"offset": -8, "nbytes": 4})
        with self.assertRaises(protocol.ProtocolError):
            protocol.decode_frame(raw_frame(header, b"\x00" * 4))

    def test_an_absurd_shape_product_is_refused_before_it_sizes_anything(self):
        header = {"id": 1, "tensors": [{"name": "x", "dtype": "float32",
                                        "shape": [1 << 20, 1 << 20, 1 << 20, 1 << 20], "offset": 0, "nbytes": 4}]}
        with self.assertRaises(protocol.ProtocolError) as e:
            protocol.decode_frame(raw_frame(header, b"\x00" * 4))
        self.assertIn("shape", str(e.exception))


class _Client:
    """Minimal blocking client used by the socket tests."""

    def __init__(self, port: int, token: str = ""):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=30)
        self.next_id = 1
        self.token = token

    def request(self, method, params=None, tensors=None, rid=None):
        rid = rid or self.next_id
        self.next_id += 1
        protocol.write_frame(self.sock, {"id": rid, "type": "request", "method": method, "params": params or {}},
                             tensors)
        return rid

    def read(self):
        return protocol.read_frame(self.sock)

    def call(self, method, params=None, tensors=None):
        """Send a request and collect (progress frames, final frame, tensors)."""
        rid = self.request(method, params, tensors)
        progress = []
        while True:
            header, tensors_out = self.read()
            assert header.get("id") == rid, header
            if header["type"] == "progress":
                progress.append(header)
                continue
            return progress, header, tensors_out

    def hello(self, version=protocol.PROTOCOL_VERSION):
        _, header, _ = self.call("hello", {"token": self.token, "protocol_version": version})
        return header

    def close(self):
        self.sock.close()


class ServerTestCase(unittest.TestCase):
    token = "s3cret"

    @classmethod
    def setUpClass(cls):
        cls.server = WorkerServer("127.0.0.1", 0, cls.token, "cpu")
        cls.port = cls.server.bind()
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls.thread.join(timeout=5)


class TestListenerRules(unittest.TestCase):
    """Reaching the port is the whole of the authorisation model, so a port the
    network can reach must at least require the token (SECURITY.md)."""

    def test_a_public_bind_without_a_token_refuses_to_start(self):
        for host in ("0.0.0.0", "", "192.0.2.7"):
            with self.assertRaises(ValueError) as e:
                WorkerServer(host, 0, "", "cpu").bind()
            self.assertIn("--token", str(e.exception))

    def test_a_public_bind_with_a_token_is_allowed(self):
        server = WorkerServer("0.0.0.0", 0, "t", "cpu")
        try:
            self.assertGreater(server.bind(), 0)
        finally:
            server.close()

    def test_loopback_without_a_token_is_allowed_but_warns(self):
        server = WorkerServer("127.0.0.1", 0, "", "cpu")
        try:
            with self.assertLogs("sirius_worker", level="WARNING") as logs:
                self.assertGreater(server.bind(), 0)
            self.assertTrue(any("no token" in line for line in logs.output), logs.output)
        finally:
            server.close()


class TestServer(ServerTestCase):
    def test_hello_reports_capabilities_and_rejects_bad_token(self):
        c = _Client(self.port, self.token)
        try:
            header = c.hello()
            self.assertEqual(header["type"], "result")
            caps = header["result"]
            self.assertEqual(caps["protocol_version"], protocol.PROTOCOL_VERSION)
            self.assertIn("run:torch_segment", caps["methods"])
            self.assertIn("run:einsum", caps["methods"])
            self.assertIn("model_info", caps["methods"])
            self.assertIn("hostname", caps)
            self.assertIsInstance(caps["cuda"], bool)
        finally:
            c.close()
        bad = _Client(self.port, "wrong")
        try:
            header = bad.hello()
            self.assertEqual(header["type"], "error")
            self.assertIn("token", header["message"])
        finally:
            bad.close()

    def test_a_client_speaking_another_protocol_version_is_refused(self):
        for version, fix in ((protocol.PROTOCOL_VERSION + 6, "update sirius_worker"),
                             (None, "update the SIRIUS application")):
            c = _Client(self.port, self.token)
            try:
                if version is None:   # a client predating the handshake sends no field at all
                    _, header, _ = c.call("hello", {"token": self.token})
                else:
                    header = c.hello(version)
                self.assertEqual(header["type"], "error", header)
                self.assertIn(f"version {protocol.PROTOCOL_VERSION}", header["message"])
                self.assertIn(f"version {version if version is not None else 0}", header["message"])
                self.assertIn(fix, header["message"])
            finally:
                c.close()

    def test_an_oversize_frame_before_hello_is_refused_without_reading_it(self):
        # 1 MiB is far below MAX_HEADER but far above what a hello may cost, so
        # only the pre-authentication cap can refuse it -- and it does so on the
        # length prefix alone, before the announced bytes are read.
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=30)
        try:
            sock.sendall(struct.pack("<I", 1 << 20))
            header, _ = protocol.read_frame(sock)
            self.assertEqual(header["type"], "error")
            self.assertIn(str(protocol.MAX_PREAUTH_FRAME), header["message"])
        finally:
            sock.close()

    def test_an_oversize_payload_before_hello_is_refused(self):
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=30)
        try:
            head = json.dumps({"id": 1, "type": "request", "method": "hello",
                               "params": {"token": self.token}}).encode("utf-8")
            sock.sendall(struct.pack("<I", len(head)) + head + struct.pack("<Q", 1 << 30))
            header, _ = protocol.read_frame(sock)
            self.assertEqual(header["type"], "error")
            self.assertIn("payload length", header["message"])
        finally:
            sock.close()

    def test_install_is_refused_without_allow_install(self):
        c = _Client(self.port, self.token)
        try:
            c.hello()
            _, header, _ = c.call("install", {"family": "cellpose"})
            self.assertEqual(header["type"], "error", header)
            self.assertIn("--allow-install", header["message"])
        finally:
            c.close()

    def test_requests_before_hello_are_refused(self):
        c = _Client(self.port, self.token)
        try:
            _, header, _ = c.call("ping")
            self.assertEqual(header["type"], "error")
            self.assertIn("hello", header["message"])
        finally:
            c.close()

    def test_numpy_kind_runs_over_the_socket(self):
        c = _Client(self.port, self.token)
        try:
            c.hello()
            rng = np.random.default_rng(1)
            arr = rng.random((2, 3, 4, 8, 8), dtype=np.float32)
            progress, header, tensors = c.call("run", {"kind": "einsum", "params": {"axes": "czyx", "reduction": "mean"}},
                                               {"input": arr})
            self.assertEqual(header["type"], "result", header)
            out = tensors["output"]
            self.assertEqual(out.shape, (2, 1, 4, 8, 8))
            np.testing.assert_allclose(out, arr.mean(axis=1, keepdims=True), rtol=1e-5)
            self.assertEqual(header["result"]["meta"]["dims"]["t"], 1)
            self.assertGreaterEqual(header["result"]["seconds"], 0.0)
        finally:
            c.close()

    def test_unknown_kind_lists_supported_ones(self):
        c = _Client(self.port, self.token)
        try:
            c.hello()
            _, header, _ = c.call("run", {"kind": "teleport", "params": {}}, {"input": np.zeros((1, 1, 2, 2, 2), np.float32)})
            self.assertEqual(header["type"], "error")
            self.assertIn("einsum", header["message"])
            self.assertIn("torch_segment", header["message"])
        finally:
            c.close()

    def test_model_info_reports_missing_file(self):
        c = _Client(self.port, self.token)
        try:
            c.hello()
            _, header, _ = c.call("model_info", {"path": "/nonexistent/model.pt"})
            self.assertEqual(header["type"], "error")
        finally:
            c.close()


class TestOneJobAtATime(unittest.TestCase):
    def test_a_second_job_while_one_is_in_flight_is_refused_as_busy(self):
        server = WorkerServer("127.0.0.1", 0, "t", "cpu")
        release = threading.Event()
        lock = threading.Lock()
        sent = []

        def send(header, tensors=None):
            with lock:
                sent.append(header)

        def slow(progress, cancel):
            release.wait(30)
            return {}, None

        server._start_job(1, "slow", send, slow)
        try:
            server._start_job(2, "slow", send, slow)   # the reader thread stays live and answers straight away
            with lock:
                frames = list(sent)
            self.assertEqual(len(frames), 1, frames)
            self.assertEqual(frames[0]["id"], 2)
            self.assertEqual(frames[0]["type"], "error")
            self.assertIn("busy", frames[0]["message"])
            self.assertIn("1", frames[0]["message"])
        finally:
            release.set()
        job = server._current_job()
        if job is not None:
            job["thread"].join(timeout=30)


@unittest.skipUnless(HAVE_TORCH, "torch not importable")
class TestTorchOverSocket(ServerTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model_path = os.path.join(cls.tmp.name, "blob.pt")

        class Blob(torch.nn.Module):
            """Two 'probability' channels: foreground = intensity, boundary = 1 - intensity."""

            def forward(self, x):
                fg = torch.clamp(x, 0.0, 1.0)
                return torch.cat([fg, 1.0 - fg], dim=1)

        torch.jit.script(Blob()).save(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.tmp.cleanup()

    def test_model_info(self):
        c = _Client(self.port, self.token)
        try:
            c.hello()
            _, header, _ = c.call("model_info", {"path": self.model_path})
            self.assertEqual(header["type"], "result", header)
            info = header["result"]
            self.assertEqual(info["format"], "TorchScript")
            self.assertEqual(info["channels_out"], 2)
            self.assertEqual(info["input_shape"][:2], [1, 1])
            self.assertGreater(info["size_bytes"], 0)
        finally:
            c.close()

    def test_torch_segment_tiles_blend_and_stream_progress(self):
        c = _Client(self.port, self.token)
        try:
            c.hello()
            z, y, x = 6, 40, 52
            vol = np.zeros((z, y, x), np.float32)
            vol[2:5, 10:30, 12:40] = 1000.0
            progress, header, tensors = c.call(
                "run", {"kind": "torch_segment", "params": {"model": self.model_path, "tile": [4, 16, 16], "overlap": [1, 4, 4]}},
                {"input": vol})
            self.assertEqual(header["type"], "result", header)
            prob = tensors["prob"]
            self.assertEqual(prob.shape, (2, z, y, x))
            self.assertEqual(header["result"]["channels"], 2)
            self.assertGreater(len(progress), 1)
            self.assertTrue(all(0.0 <= p["fraction"] <= 1.0 for p in progress))
            # blended tiles reproduce the identity model without seams
            np.testing.assert_allclose(prob[0], (vol > 0).astype(np.float32), atol=1e-4)
            np.testing.assert_allclose(prob[1], 1.0 - (vol > 0).astype(np.float32), atol=1e-4)
        finally:
            c.close()

    def test_seg_step_returns_labels(self):
        c = _Client(self.port, self.token)
        try:
            c.hello()
            vol = np.zeros((1, 1, 6, 40, 52), np.float32)
            vol[0, 0, 1:5, 5:15, 5:15] = 100.0
            vol[0, 0, 1:5, 25:35, 30:45] = 100.0
            _, header, tensors = c.call(
                "run", {"kind": "seg", "params": {"model": self.model_path, "tile": [8, 32, 32], "overlap": 4,
                                                   "post": "Connected components", "fg_channel": 0}},
                {"input": vol})
            self.assertEqual(header["type"], "result", header)
            labels = tensors["labels"]
            self.assertEqual(labels.shape, (1, 6, 40, 52))
            self.assertEqual(labels.dtype, np.uint32)
            self.assertEqual(int(labels.max()), 2)
            self.assertEqual(header["result"]["info"]["labels"], 2)
        finally:
            c.close()

    def test_cancel_stops_a_run(self):
        c = _Client(self.port, self.token)
        try:
            c.hello()
            vol = np.random.default_rng(0).random((16, 256, 256), dtype=np.float32)
            rid = c.request("run", {"kind": "torch_segment", "params": {"model": self.model_path, "tile": [2, 32, 32],
                                                                          "overlap": [0, 8, 8]}}, {"input": vol})
            # wait for the first progress frame, then cancel
            header, _ = c.read()
            self.assertEqual(header["type"], "progress")
            cancel_id = c.request("cancel", {"id": rid})
            seen = {}
            deadline = time.time() + 30
            while len(seen) < 2 and time.time() < deadline:
                header, _ = c.read()
                if header["type"] == "progress":
                    continue
                seen[header["id"]] = header
            self.assertEqual(seen[cancel_id]["type"], "result")
            self.assertEqual(seen[rid]["type"], "error")
            self.assertEqual(seen[rid]["message"], "cancelled")
            # the worker is usable afterwards
            _, header, _ = c.call("ping")
            self.assertEqual(header["type"], "result")
        finally:
            c.close()


class TestStepLibraryLocation(unittest.TestCase):
    def test_workbench_is_found(self):
        wb = workbench()
        self.assertTrue(callable(wb.run_step))
        self.assertIn("einsum", wb.step_kinds())


class TestCommandLine(unittest.TestCase):
    def test_module_announces_its_port_and_serves(self):
        import subprocess

        env = dict(os.environ, PYTHONPATH=os.path.dirname(HERE) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        proc = subprocess.Popen([sys.executable, "-m", "sirius_worker", "--port", "0", "--token", "t", "--device", "cpu",
                                 "--log-level", "WARNING"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                text=True)
        try:
            line = proc.stdout.readline()
            if not line.strip():
                proc.kill()
                self.fail("worker printed nothing on stdout; stderr: " + proc.stderr.read())
            announce = json.loads(line)
            self.assertEqual(announce["pid"], proc.pid)
            self.assertGreater(announce["port"], 0)
            c = _Client(announce["port"], "t")
            try:
                caps = c.hello()["result"]
                self.assertIn("run:einsum", caps["methods"])
                _, header, _ = c.call("shutdown")
                self.assertEqual(header["type"], "result")
            finally:
                c.close()
            self.assertEqual(proc.wait(timeout=15), 0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    unittest.main()


class TestConnectionLifecycle(unittest.TestCase):
    """The connection loop stays answerable: stop() reaches an idle
    connection, and a peer that never says hello does not hold the port."""

    def _serve(self, server):
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return thread

    def test_stop_takes_effect_while_a_client_is_connected_and_idle(self):
        server = WorkerServer("127.0.0.1", 0, "t", "cpu")
        port = server.bind()
        thread = self._serve(server)
        client = _Client(port, "t")
        self.assertEqual(client.hello()["type"], "result")
        # before: the loop sat in recv and only noticed the flag on the next frame
        server.stop()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        client.close()

    def test_a_peer_that_never_says_hello_is_dropped_and_the_next_one_served(self):
        server = WorkerServer("127.0.0.1", 0, "t", "cpu")
        server.HELLO_TIMEOUT = 0.5
        port = server.bind()
        thread = self._serve(server)
        try:
            silent = socket.create_connection(("127.0.0.1", port), timeout=10)
            self.assertEqual(silent.recv(1), b"")   # closed on us, nothing sent
            silent.close()
            client = _Client(port, "t")
            self.assertEqual(client.hello()["type"], "result")
            client.close()
        finally:
            server.stop()
            thread.join(timeout=5)

    def test_hello_params_of_the_wrong_type_are_an_error_not_a_crash(self):
        server = WorkerServer("127.0.0.1", 0, "t", "cpu")
        port = server.bind()
        thread = self._serve(server)
        try:
            client = _Client(port, "t")
            protocol.write_frame(client.sock, {"id": 1, "type": "request", "method": "hello", "params": [1, 2]})
            header, _ = client.read()
            self.assertEqual(header["type"], "error")   # a failed hello, which ends the connection
            client.close()
            again = _Client(port, "t")                   # and the server is still there for the next one
            self.assertEqual(again.hello()["type"], "result")
            again.close()
        finally:
            server.stop()
            thread.join(timeout=5)


class TestJsonScrubbing(unittest.TestCase):
    def test_nan_inside_an_array_is_scrubbed_like_a_bare_float(self):
        from sirius_worker.server import _jsonable
        table = {"rows": [np.array([np.nan, 1.5]), (float("inf"), 2)], "n": np.int64(3)}
        out = _jsonable(table)
        self.assertEqual(out, {"rows": [[None, 1.5], [None, 2]], "n": 3})
        json.dumps(out, allow_nan=False)   # what encode_frame does
