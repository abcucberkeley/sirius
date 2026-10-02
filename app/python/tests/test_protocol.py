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
import unittest.mock
import warnings

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
        """The handshake (protocol.client_handshake): the last reply header."""
        header = protocol.client_handshake(self.sock, self.token, version, first_id=self.next_id)
        self.next_id += 2
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
        # A client with the wrong token stops at the worker's proof, having
        # sent nothing but its nonce.
        bad = _Client(self.port, "wrong")
        try:
            with self.assertRaises(protocol.ProtocolError) as e:
                bad.hello()
            self.assertIn("could not prove", str(e.exception))
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


class TestRequestDevice(unittest.TestCase):
    """A run goes where its request says: seg.cpp sends "device": "cpu" for
    the CPU backend and "auto" otherwise, and the step reports where it ran
    from the request, so a worker started on CUDA must honour "cpu"."""

    def test_the_request_names_the_device_and_auto_is_the_workers_own(self):
        server = WorkerServer("127.0.0.1", 0, "t", "cuda")
        server._cuda = True   # a GPU worker, whatever this machine has
        self.assertEqual(server.request_device("cpu"), "cpu")
        self.assertEqual(server.request_device("CPU"), "cpu")
        self.assertEqual(server.request_device("auto"), "cuda")
        self.assertEqual(server.request_device(None), "cuda")
        self.assertEqual(server.request_device("cuda:1"), "cuda:1")

    def test_a_gpu_asked_of_a_worker_without_one_is_refused_clearly(self):
        # The HPC backend sends the session's GPU / CPU choice with every
        # step; a job started without GPUs must say so, not run on the CPU.
        from sirius_worker import server as server_module  # noqa: PLC0415

        server = WorkerServer("127.0.0.1", 0, "t", "cpu")
        server._cuda = False
        self.assertEqual(server.request_device("cpu"), "cpu")
        with unittest.mock.patch.dict(os.environ, {"SLURM_JOB_ID": "4242"}):
            for asked in ("cuda", "CUDA", "cuda:0"):
                with self.assertRaises(ValueError, msg=asked) as e:
                    server.request_device(asked)
                self.assertEqual(str(e.exception), "this worker job has no GPU; choose CPU or reconnect with GPUs >= 1")
        env = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as e:
                server.request_device("cuda")
            self.assertEqual(str(e.exception), server_module.NO_GPU_HERE)

    def test_each_run_is_refused_or_served_by_its_own_device(self):
        # One connection, the device switched between requests: a GPU run is
        # refused at once (before the cluster input is read), the next CPU run
        # of the same connection is served -- no new job, no restart.
        server = WorkerServer("127.0.0.1", 0, "t", "cpu")
        server._cuda = False
        port = server.bind()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            c = _Client(port, "t")
            c.hello()
            ref = {"path": "/nowhere/missing.tif", "options": {}, "c": 0, "t": 0, "layout": "zyx"}
            with unittest.mock.patch.dict(os.environ, {"SLURM_JOB_ID": "4242"}):
                _, header, _ = c.call("run", {"kind": "einsum", "input_ref": ref,
                                              "params": {"device": "cuda", "keep": "ctyx", "reduction": "max"}})
            self.assertEqual(header["type"], "error", header)
            self.assertIn("this worker job has no GPU; choose CPU or reconnect with GPUs >= 1", header["message"])
            vol = np.ones((1, 1, 2, 3, 4), dtype=np.float32)
            _, header, _ = c.call("run", {"kind": "einsum", "params": {"device": "cpu", "keep": "ctyx", "reduction": "max"}},
                                  {"input": vol})
            self.assertEqual(header["type"], "result", header)
            self.assertEqual(header["result"]["device"], "cpu")
            c.close()
        finally:
            server.stop()
            thread.join(timeout=5)

    def test_the_command_line_takes_the_gpu_the_application_names(self):
        # The application starts its worker with --device cuda:N once a GPU is
        # chosen; argparse's choices took only auto / cuda / cpu, so the worker
        # exited with a usage error and every Python step on the CUDA backend failed.
        import argparse  # noqa: PLC0415

        from sirius_worker.__main__ import _device  # noqa: PLC0415

        for text, expected in (("auto", "auto"), ("cpu", "cpu"), ("CUDA", "cuda"), ("cuda:0", "cuda:0"), ("cuda:3", "cuda:3")):
            self.assertEqual(_device(text), expected)
        for bad in ("gpu", "cuda:", "cuda:x", "cuda:-1", "cpu:0", ""):
            with self.assertRaises(argparse.ArgumentTypeError, msg=bad):
                _device(bad)

    @unittest.skipUnless(HAVE_TORCH, "torch not importable")
    def test_a_cpu_request_runs_on_the_cpu_of_a_cuda_worker(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "where.pt")

        class Where(torch.nn.Module):
            """1 where the input tensor is on CUDA, 0 on the CPU."""

            def forward(self, x):
                return torch.ones_like(x) * float(x.is_cuda)

        torch.jit.script(Where()).save(path)
        # started for CUDA: without a GPU a run that ignored the request's
        # device fails, with one it computes on the GPU; either way not "cpu"
        server = WorkerServer("127.0.0.1", 0, "t", "cuda")
        port = server.bind()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            c = _Client(port, "t")
            c.hello()
            vol = np.random.default_rng(0).random((4, 16, 16), dtype=np.float32)
            _, header, tensors = c.call("run", {"kind": "torch_segment", "params": {
                "model": path, "tile": [4, 16, 16], "overlap": 0, "device": "cpu"}}, {"input": vol})
            c.close()
            self.assertEqual(header["type"], "result", header)
            self.assertEqual(header["result"]["device"], "cpu")
            self.assertEqual(float(tensors["prob"].max()), 0.0)
        finally:
            server.stop()
            thread.join(timeout=5)


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
        server.PREAUTH_TIMEOUT = 0.5
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

    def test_a_partial_frame_before_hello_does_not_lock_the_next_client_out(self):
        # one byte of a header length, then silence: the frame read used to
        # block with no deadline, so nobody else was ever served
        server = WorkerServer("127.0.0.1", 0, "t", "cpu")
        server.PREAUTH_TIMEOUT = 1.0
        port = server.bind()
        thread = self._serve(server)
        try:
            stalled = socket.create_connection(("127.0.0.1", port), timeout=10)
            stalled.sendall(b"\x05")
            time.sleep(0.3)
            client = _Client(port, "t")
            client.sock.settimeout(10)
            t0 = time.monotonic()
            self.assertEqual(client.hello()["type"], "result")
            self.assertLess(time.monotonic() - t0, 5.0)
            self.assertEqual(stalled.recv(1), b"")   # the stalled peer was dropped
            stalled.close()
            client.close()
        finally:
            server.stop()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())

    def test_a_hello_dripped_a_byte_at_a_time_is_cut_off_at_the_deadline(self):
        # every byte arrives well within the poll interval, so only a
        # deadline on the whole frame stops it; a served hello is the failure
        server = WorkerServer("127.0.0.1", 0, "t", "cpu")
        server.PREAUTH_TIMEOUT = 0.5
        port = server.bind()
        thread = self._serve(server)
        try:
            frame = protocol.encode_frame({"id": 1, "type": "request", "method": "hello",
                                           "params": {"client_nonce": "ab" * 16,
                                                      "protocol_version": protocol.PROTOCOL_VERSION}})
            self.assertGreater(len(frame) * 0.03, 2 * server.PREAUTH_TIMEOUT)
            drip = socket.create_connection(("127.0.0.1", port), timeout=10)
            closed = False
            for byte in frame:
                try:
                    drip.sendall(bytes([byte]))
                except OSError:
                    closed = True
                    break
                time.sleep(0.03)
            if not closed:
                drip.settimeout(10)
                try:
                    header, _ = protocol.read_frame(drip)
                    closed = header.get("type") != "result"
                except (ConnectionError, OSError):
                    closed = True
            self.assertTrue(closed, "a hello that took longer than PREAUTH_TIMEOUT was served")
            drip.close()
        finally:
            server.stop()
            thread.join(timeout=5)

    def test_stop_takes_effect_while_an_authenticated_frame_is_half_sent(self):
        server = WorkerServer("127.0.0.1", 0, "t", "cpu")
        port = server.bind()
        thread = self._serve(server)
        client = _Client(port, "t")
        self.assertEqual(client.hello()["type"], "result")
        frame = protocol.encode_frame({"id": 2, "type": "request", "method": "ping", "params": {}})
        client.sock.sendall(frame[:6])   # the length and two bytes of the header, then nothing
        time.sleep(0.3)
        server.stop()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive(), "stop() did not reach a reader waiting inside a frame")
        client.close()

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

    def test_a_numpy_nan_scalar_is_scrubbed(self):
        # a plugin's np.nanmean of an all-NaN column is a numpy float64 NaN;
        # it made encode_frame raise and the whole reply was lost
        from sirius_worker.server import _jsonable
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(np.array([np.nan, np.nan]))
        out = _jsonable({"mean": mean, "peak": np.float32(np.inf), "count": np.int32(2)})
        self.assertEqual(out, {"mean": None, "peak": None, "count": 2})
        protocol.encode_frame({"id": 1, "type": "result", "result": out})


class TestDescriptorNumbers(unittest.TestCase):
    """Every number in a tensor descriptor must be a non-negative JSON
    integer. json.loads makes 1e309 an infinite float and accepts Infinity
    and NaN; int() of the first raised OverflowError, which nothing caught,
    and the worker process ended on an unauthenticated frame."""

    @staticmethod
    def _frame(tensors_json: str, payload: bytes = b"\0" * 4) -> bytes:
        header = ('{"id":1,"type":"request","method":"run","tensors":' + tensors_json + '}').encode("utf-8")
        return protocol.HEADER_LEN.pack(len(header)) + header + protocol.PAYLOAD_LEN.pack(len(payload)) + payload

    def test_non_integer_descriptor_numbers_are_protocol_errors(self):
        cases = ['"shape":[1e309],"offset":0,"nbytes":4', '"shape":[Infinity],"offset":0,"nbytes":4',
                 '"shape":[1.5],"offset":0,"nbytes":4', '"shape":[-1],"offset":0,"nbytes":4',
                 '"shape":["2"],"offset":0,"nbytes":4', '"shape":[true],"offset":0,"nbytes":4',
                 '"shape":1,"offset":0,"nbytes":4', '"shape":[1],"offset":1e309,"nbytes":4',
                 '"shape":[1],"offset":0.5,"nbytes":4', '"shape":[1],"offset":0,"nbytes":NaN',
                 '"shape":[1],"offset":0,"nbytes":4.0']
        for fields in cases:
            with self.subTest(fields=fields):
                reader = protocol.FrameReader()
                with self.assertRaises(protocol.ProtocolError):
                    reader.feed(self._frame('[{"name":"a","dtype":"float32",' + fields + '}]'))
        reader = protocol.FrameReader()
        frames = reader.feed(self._frame('[{"name":"a","dtype":"float32","shape":[1],"offset":0,"nbytes":4}]'))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][1]["a"].shape, (1,))

    def test_a_hostile_descriptor_does_not_end_the_worker(self):
        server = WorkerServer("127.0.0.1", 0, "t", "cpu")
        port = server.bind()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            hostile = socket.create_connection(("127.0.0.1", port), timeout=10)
            hostile.sendall(self._frame('[{"name":"a","dtype":"float32","shape":[1e309],"offset":0,"nbytes":4}]'))
            header, _ = protocol.read_frame(hostile)
            self.assertEqual(header["type"], "error")
            hostile.close()
            self.assertTrue(thread.is_alive())
            client = _Client(port, "t")   # the next client is served as before
            self.assertEqual(client.hello()["type"], "result")
            client.close()
        finally:
            server.stop()
            thread.join(timeout=5)


class TestReloadWhileBusy(unittest.TestCase):
    def test_reloading_plugins_is_refused_while_a_job_runs(self):
        server = WorkerServer("127.0.0.1", 0, "t", "cpu")
        release = threading.Event()

        def send(header, tensors=None):
            pass

        def slow(progress, cancel):
            release.wait(30)
            return {}, None

        server._start_job(1, "slow", send, slow)
        try:
            # re-importing a plugin file under a step that may be executing it
            with self.assertRaises(RuntimeError) as caught:
                server.plugin_list(reload=True)
            self.assertIn("busy", str(caught.exception))
            server.plugin_list(reload=False)   # listing is fine
        finally:
            release.set()
        job = server._current_job()
        if job is not None:
            job["thread"].join(timeout=30)
        server.plugin_list(reload=True)   # and afterwards a reload is


class TestJobSlot(unittest.TestCase):
    """One job at a time, also when two requests race for the slot."""

    def test_two_requests_racing_for_the_slot_start_one_job(self):
        server = WorkerServer("127.0.0.1", 0, "t", "cpu")
        release = threading.Event()
        started = []
        replies = []
        replies_lock = threading.Lock()

        def send(header, tensors=None):
            with replies_lock:
                replies.append(header)

        def work(progress, cancel):
            started.append(threading.current_thread().name)
            release.wait(5)
            return {}, {}

        # widen the window between claiming the slot and starting the thread,
        # which is where the second request used to slip in
        real_start = threading.Thread.start

        def slow_start(thread):
            time.sleep(0.05)
            real_start(thread)

        errors = []

        def request(rid):
            try:
                server._start_job(rid, "race", send, work)
            except Exception as e:  # noqa: BLE001 - the failure under test
                errors.append(e)

        with unittest.mock.patch.object(threading.Thread, "start", slow_start):
            racers = [threading.Thread(target=request, args=(rid,)) for rid in (1, 2)]
            for r in racers:
                real_start(r)
            for r in racers:
                r.join(5)
        time.sleep(0.1)
        self.assertEqual(errors, [])
        self.assertEqual(len(started), 1, "exactly one job runs")
        busy = [h for h in replies if h.get("type") == "error" and "busy" in h.get("message", "")]
        self.assertEqual(len(busy), 1, "the other request is told the worker is busy")
        release.set()
        for _ in range(50):
            if server._current_job() is None:
                break
            time.sleep(0.02)
        self.assertIsNone(server._current_job(), "the slot is free again once the job ends")


class TestDescriptorTiling(unittest.TestCase):
    """Tensor descriptors tile the payload in order, at most MAX_TENSORS of
    them: a peer cannot have one byte decoded twice, or describe more tensors
    than any request carries."""

    @staticmethod
    def two(offset_a, offset_b, name_b="b"):
        return {"id": 1, "tensors": [
            {"name": "a", "dtype": "float32", "shape": [1], "offset": offset_a, "nbytes": 4},
            {"name": name_b, "dtype": "float32", "shape": [1], "offset": offset_b, "nbytes": 4}]}

    def test_in_order_and_apart_is_accepted(self):
        header, tensors = protocol.decode_frame(raw_frame(self.two(0, 4), b"\x00" * 8))
        self.assertEqual(sorted(tensors), ["a", "b"])

    def test_overlapping_or_backwards_descriptors_are_refused(self):
        for a, b in ((0, 0), (0, 2), (4, 0)):
            with self.subTest(offsets=(a, b)):
                with self.assertRaises(protocol.ProtocolError) as e:
                    protocol.decode_frame(raw_frame(self.two(a, b), b"\x00" * 8))
                self.assertIn("overlaps", str(e.exception))

    def test_a_name_described_twice_is_refused(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.decode_frame(raw_frame(self.two(0, 4, name_b="a"), b"\x00" * 8))

    def test_too_many_descriptors_are_refused(self):
        header = {"id": 1, "tensors": [{"name": f"t{i}", "dtype": "uint8", "shape": [0], "offset": 0, "nbytes": 0}
                                       for i in range(protocol.MAX_TENSORS + 1)]}
        with self.assertRaises(protocol.ProtocolError) as e:
            protocol.decode_frame(raw_frame(header))
        self.assertIn("tensors", str(e.exception))


class TestHandshake(unittest.TestCase):
    """Protocol version 2: the token never crosses the wire, the worker proves
    it first, and anonymous peers can hold neither a client slot nor the
    listener (SECURITY.md)."""

    def _serve(self, server):
        port = server.bind()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return port, thread

    def _stop(self, server, thread):
        server.stop()
        thread.join(timeout=5)

    def test_proofs_match_the_cpp_definition(self):
        import hashlib
        import hmac

        proof = protocol.handshake_proof("tok", "worker", "a" * 32, "b" * 32)
        self.assertEqual(proof, hmac.new(b"tok", ("sirius-worker-auth/2|worker|" + "a" * 32 + "|" + "b" * 32).encode(),
                                         hashlib.sha256).hexdigest())
        self.assertNotEqual(proof, protocol.handshake_proof("tok", "client", "a" * 32, "b" * 32))
        self.assertTrue(protocol.valid_nonce("0f" * 16))
        for bad in ("0F" * 16, "0f" * 15, "zz" * 16, 7, None, "0f" * 65):
            self.assertFalse(protocol.valid_nonce(bad), bad)

    def test_the_hello_reply_proves_the_token_and_reveals_nothing_else(self):
        server = WorkerServer("127.0.0.1", 0, "s3cret", "cpu")
        port, thread = self._serve(server)
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=10)
            nonce = "12" * 16
            protocol.write_frame(sock, {"id": 1, "type": "request", "method": "hello",
                                        "params": {"protocol_version": protocol.PROTOCOL_VERSION,
                                                   "client_nonce": nonce}})
            header, _ = protocol.read_frame(sock)
            self.assertEqual(header["type"], "result", header)
            result = header["result"]
            self.assertNotIn("methods", result)   # the capabilities wait for the client's proof
            self.assertNotIn("s3cret", json.dumps(header))
            self.assertEqual(result["server_proof"],
                             protocol.handshake_proof("s3cret", "worker", nonce, result["server_nonce"]))
            # nothing is served on the worker's proof alone
            protocol.write_frame(sock, {"id": 2, "type": "request", "method": "ping", "params": {}})
            header, _ = protocol.read_frame(sock)
            self.assertEqual(header["type"], "error")
            self.assertIn("handshake", header["message"])
            sock.close()
        finally:
            self._stop(server, thread)

    def test_a_wrong_client_proof_is_refused_and_ends_the_connection(self):
        server = WorkerServer("127.0.0.1", 0, "s3cret", "cpu")
        port, thread = self._serve(server)
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=10)
            protocol.write_frame(sock, {"id": 1, "type": "request", "method": "hello",
                                        "params": {"protocol_version": protocol.PROTOCOL_VERSION,
                                                   "client_nonce": "34" * 16}})
            header, _ = protocol.read_frame(sock)
            # the worker's own proof sent back as the client's: a reflection
            protocol.write_frame(sock, {"id": 2, "type": "request", "method": "auth",
                                        "params": {"client_proof": header["result"]["server_proof"]}})
            header, _ = protocol.read_frame(sock)
            self.assertEqual(header["type"], "error")
            self.assertIn("authentication failed", header["message"])
            self.assertEqual(sock.recv(1), b"")
            sock.close()
        finally:
            self._stop(server, thread)

    def test_a_version_1_hello_with_the_token_is_refused_by_version(self):
        server = WorkerServer("127.0.0.1", 0, "s3cret", "cpu")
        port, thread = self._serve(server)
        try:
            c = _Client(port, "s3cret")
            _, header, _ = c.call("hello", {"token": "s3cret", "protocol_version": 1})
            self.assertEqual(header["type"], "error")
            self.assertIn("update the SIRIUS application", header["message"])
            c.close()
        finally:
            self._stop(server, thread)

    def test_a_silent_peer_does_not_hold_a_single_client_worker(self):
        server = WorkerServer("127.0.0.1", 0, "t", "cpu")   # one client at a time
        port, thread = self._serve(server)
        try:
            silent = socket.create_connection(("127.0.0.1", port), timeout=10)
            t0 = time.monotonic()
            client = _Client(port, "t")
            self.assertEqual(client.hello()["type"], "result")
            self.assertLess(time.monotonic() - t0, server.PREAUTH_TIMEOUT)   # served before the silent one is dropped
            client.close()
            silent.close()
        finally:
            self._stop(server, thread)

    def test_the_next_client_of_a_single_client_worker_waits_for_the_first(self):
        server = WorkerServer("127.0.0.1", 0, "t", "cpu")
        port, thread = self._serve(server)
        try:
            first = _Client(port, "t")
            self.assertEqual(first.hello()["type"], "result")
            answered = []

            def second():
                c = _Client(port, "t")
                answered.append(c.hello())
                c.close()

            waiter = threading.Thread(target=second, daemon=True)
            waiter.start()
            time.sleep(1.0)
            self.assertEqual(answered, [], "the second client was served while the first was connected")
            first.close()
            waiter.join(timeout=10)
            self.assertEqual(answered[0]["type"], "result")
        finally:
            self._stop(server, thread)

    def test_peers_in_their_handshake_take_no_client_slot_and_are_capped(self):
        server = WorkerServer("127.0.0.1", 0, "t", "cpu", max_clients=2)
        port, thread = self._serve(server)
        silent = []
        try:
            for _ in range(server.MAX_PREAUTH):
                silent.append(socket.create_connection(("127.0.0.1", port), timeout=10))
            time.sleep(0.3)
            # one more anonymous peer is closed at once
            extra = socket.create_connection(("127.0.0.1", port), timeout=10)
            extra.settimeout(3)
            self.assertEqual(extra.recv(1), b"")
            extra.close()
            for s_ in silent:
                s_.close()
            time.sleep(server.IDLE_POLL * 3)
            # both client slots are still free for clients that authenticate
            a, b = _Client(port, "t"), _Client(port, "t")
            self.assertEqual(a.hello()["type"], "result")
            self.assertEqual(b.hello()["type"], "result")
            c = _Client(port, "t")
            header = c.hello()
            self.assertEqual(header["type"], "error")
            self.assertIn("busy", header["message"])
            for x in (a, b, c):
                x.close()
        finally:
            for s_ in silent:
                s_.close()
            self._stop(server, thread)

    def test_an_idle_authenticated_connection_is_closed(self):
        server = WorkerServer("127.0.0.1", 0, "t", "cpu", idle_timeout=0.6)
        port, thread = self._serve(server)
        try:
            c = _Client(port, "t")
            self.assertEqual(c.hello()["type"], "result")
            c.sock.settimeout(10)
            self.assertEqual(c.sock.recv(1), b"")   # closed by the worker
            c.close()
            again = _Client(port, "t")                # and its slot is free
            self.assertEqual(again.hello()["type"], "result")
            again.close()
        finally:
            self._stop(server, thread)

    def test_a_public_worker_refuses_plugin_folders_a_client_names(self):
        server = WorkerServer("127.0.0.1", 0, "t", "cpu")
        port = server.bind()
        server.host = "192.0.2.7"   # as if bound to a routable address; the socket stays on loopback
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            c = _Client(port, "t")
            self.assertEqual(c.hello()["type"], "result")
            _, header, _ = c.call("reload_plugins", {"dirs": [tempfile.gettempdir()]})
            self.assertEqual(header["type"], "error")
            self.assertIn("folders it was started with", header["message"])
            c.close()
        finally:
            self._stop(server, thread)


class TestTokenSources(unittest.TestCase):
    """The token reaches the worker without being in its command line or, on
    a cluster, in the job's environment (__main__.py)."""

    def setUp(self):
        from sirius_worker import __main__ as cli

        self.cli = cli
        self.saved = {k: os.environ.get(k) for k in ("SIRIUS_TOKEN", "SIRIUS_TOKEN_FILE")}

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def args(self, **kw):
        import argparse

        return argparse.Namespace(**{"token": None, "token_file": None, **kw})

    def write_token(self, text="tok-from-file"):
        fd, path = tempfile.mkstemp(prefix="sirius-token-")
        os.write(fd, (text + "\n").encode())
        os.close(fd)
        if os.name == "posix":
            os.chmod(path, 0o600)
        return path

    def test_a_token_file_is_read_deleted_and_both_variables_dropped(self):
        path = self.write_token()
        os.environ["SIRIUS_TOKEN_FILE"] = path
        os.environ["SIRIUS_TOKEN"] = "from-env"
        self.assertEqual(self.cli._token(self.args()), "tok-from-file")
        self.assertFalse(os.path.exists(path))
        self.assertNotIn("SIRIUS_TOKEN", os.environ)
        self.assertNotIn("SIRIUS_TOKEN_FILE", os.environ)

    def test_the_environment_variable_still_works(self):
        os.environ.pop("SIRIUS_TOKEN_FILE", None)
        os.environ["SIRIUS_TOKEN"] = "from-env"
        self.assertEqual(self.cli._token(self.args()), "from-env")
        self.assertNotIn("SIRIUS_TOKEN", os.environ)

    def test_a_token_on_the_command_line_warns(self):
        with self.assertLogs("sirius_worker", level="WARNING") as logs:
            self.assertEqual(self.cli._token(self.args(token="argv")), "argv")
        self.assertTrue(any("visible to every user" in line for line in logs.output), logs.output)

    @unittest.skipUnless(os.name == "posix", "file modes are POSIX")
    def test_a_token_file_others_can_read_is_refused(self):
        path = self.write_token()
        os.chmod(path, 0o644)
        try:
            with self.assertRaises(ValueError) as e:
                self.cli.read_token_file(path)
            self.assertIn("chmod 600", str(e.exception))
        finally:
            if os.path.exists(path):
                os.remove(path)



def _fake_nvidia_smi(directory: str, lines) -> None:
    """An nvidia-smi on `directory` that prints `lines` (its --query-gpu CSV)."""
    if os.name == "nt":
        with open(os.path.join(directory, "nvidia-smi.bat"), "w", encoding="ascii") as f:
            f.write("@echo off\r\n" + "".join(f"echo {line}\r\n" for line in lines))
    else:
        path = os.path.join(directory, "nvidia-smi")
        with open(path, "w", encoding="ascii") as f:
            f.write("#!/bin/sh\n" + "".join(f"echo '{line}'\n" for line in lines))
        os.chmod(path, 0o755)


class TestGpuHardware(unittest.TestCase):
    """A job holds a GPU its environment cannot compute on (a venv without
    torch or the sirius package): hello still names the GPU, from nvidia-smi,
    and says why it is not usable -- rather than reporting a CPU-only job."""

    A100 = "0, NVIDIA A100-SXM4-80GB, 81920, GPU-aaaa-1111"
    V100 = "1, Tesla V100-SXM2-32GB, 32768, GPU-bbbb-2222"

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil  # noqa: PLC0415

        shutil.rmtree(self.dir, ignore_errors=True)

    def env(self, **extra):
        env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_JOB_ID")}
        env["PATH"] = self.dir   # nothing but the fake: the machine's own nvidia-smi stays out
        env.update(extra)
        return env

    def caps(self, **extra):
        from sirius_worker import server as server_module  # noqa: PLC0415

        server = WorkerServer("127.0.0.1", 0, "t", "auto")
        with unittest.mock.patch.dict(os.environ, self.env(**extra), clear=True), \
                unittest.mock.patch.object(server_module, "_cuda_device", return_value=None):
            return server.capabilities()

    def test_nvidia_smi_names_the_gpu_a_worker_without_cuda_cannot_use(self):
        from sirius_worker import server as server_module  # noqa: PLC0415

        _fake_nvidia_smi(self.dir, [self.A100])
        with unittest.mock.patch.dict(sys.modules, {"torch": None, "sirius": None}):
            caps = self.caps(SLURM_JOB_ID="4238488", CUDA_VISIBLE_DEVICES="0")
        self.assertEqual(caps["gpus"], [{"name": "NVIDIA A100-SXM4-80GB", "memory_mb": 81920}])
        self.assertIs(caps["cuda_usable"], False)
        self.assertIs(caps["cuda"], False)
        self.assertEqual(caps["cuda_reason"], server_module.NO_CUDA_LIBRARY)
        self.assertIn("no CUDA library in the worker's environment", caps["cuda_reason"])
        self.assertEqual(caps["cpu_threads"], os.cpu_count() or 1)
        self.assertTrue(caps["device"].startswith("cpu"), caps["device"])

    def test_without_nvidia_smi_there_is_no_gpu_and_the_reason_says_so(self):
        caps = self.caps(SLURM_JOB_ID="4238488")
        self.assertEqual(caps["gpus"], [])
        self.assertIs(caps["cuda_usable"], False)
        self.assertIn("this worker job has no GPU", caps["cuda_reason"])
        caps = self.caps()
        self.assertIn("no NVIDIA GPU on this machine", caps["cuda_reason"])

    def test_a_worker_that_computes_on_the_gpu_gives_no_reason(self):
        from sirius_worker import server as server_module  # noqa: PLC0415

        _fake_nvidia_smi(self.dir, [self.A100])
        server = WorkerServer("127.0.0.1", 0, "t", "auto")
        with unittest.mock.patch.dict(os.environ, self.env(), clear=True), \
                unittest.mock.patch.object(server_module, "_cuda_device", return_value="cuda:0 · A100 · 80 GB"):
            caps = server.capabilities()
        self.assertIs(caps["cuda_usable"], True)
        self.assertEqual(caps["cuda_reason"], "")
        self.assertEqual(len(caps["gpus"]), 1)

    def test_the_jobs_own_gpus_from_cuda_visible_devices_or_slurm(self):
        from sirius_worker.server import detect_gpus  # noqa: PLC0415

        _fake_nvidia_smi(self.dir, [self.A100, self.V100])
        names = lambda env: [g["name"] for g in detect_gpus(env)]  # noqa: E731
        self.assertEqual(names(self.env()), ["NVIDIA A100-SXM4-80GB", "Tesla V100-SXM2-32GB"])
        self.assertEqual(names(self.env(CUDA_VISIBLE_DEVICES="1")), ["Tesla V100-SXM2-32GB"])
        self.assertEqual(names(self.env(CUDA_VISIBLE_DEVICES="GPU-aaaa")), ["NVIDIA A100-SXM4-80GB"])
        self.assertEqual(names(self.env(SLURM_JOB_GPUS="1")), ["Tesla V100-SXM2-32GB"])
        self.assertEqual(names(self.env(CUDA_VISIBLE_DEVICES="")), [])
        # a node confining the job's devices lists its one GPU as index 0 while
        # Slurm names the physical one: both are the same GPU
        _fake_nvidia_smi(self.dir, [self.A100])
        self.assertEqual(names(self.env(SLURM_JOB_GPUS="3")), ["NVIDIA A100-SXM4-80GB"])
        # a Slurm job given no GPU on a node that does not confine devices:
        # the GPUs nvidia-smi lists are other jobs'
        self.assertEqual(names(self.env(SLURM_JOB_ID="7")), [])

    def test_hello_carries_the_new_fields_over_the_socket(self):
        _fake_nvidia_smi(self.dir, [self.A100])
        server = WorkerServer("127.0.0.1", 0, "t", "cpu")
        with unittest.mock.patch.dict(os.environ, self.env(), clear=True):
            server.gpus()   # asked once, here, with the fake on PATH
        port = server.bind()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            c = _Client(port, "t")
            caps = c.hello()["result"]
            c.close()
        finally:
            server.stop()
            thread.join(timeout=5)
        self.assertEqual(caps["gpus"], [{"name": "NVIDIA A100-SXM4-80GB", "memory_mb": 81920}])
        self.assertIsInstance(caps["cuda_usable"], bool)
        self.assertIsInstance(caps["cuda_reason"], str)
        self.assertGreaterEqual(caps["cpu_threads"], 1)
