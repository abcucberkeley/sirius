"""Plugin loading and execution, in-process and over the socket."""

import os
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sirius_worker import plugins  # noqa: E402
from sirius_worker.protocol import PROTOCOL_VERSION, encode_frame, read_frame  # noqa: E402
from sirius_worker.server import WorkerServer  # noqa: E402

GOOD = '''
STEP = {"kind": "scale", "name": "Scale", "group": "Intensity",
        "params": [{"key": "factor", "type": "double", "default": 2.0, "min": 0, "max": 10},
                   {"key": "mode", "type": "choice", "choices": ["a", "b"], "default": "b"}],
        "separable_over_t": True}

def run(data, params, meta, ctx):
    """# Scale

    Multiplies by *factor*.
    """
    ctx.progress(0.5, "half")
    out = data * params["factor"]
    return out, {"summary": "scaled", "facts": {"factor": str(params["factor"])},
                 "images": [{"title": "mid", "data": out[0, 0, 0]}]}
'''
BAD = "STEP = {'kind': 'bad', 'params': [{'key': 'x', 'type': 'nope'}]}\ndef run(d, p, m, c): return d\n"
LABELS = '''
STEP = {"kind": "blobs", "produces_labels": True}
def run(data, params, meta, ctx):
    labels = (data[0] > 0.5).astype("uint32")
    return data, labels
'''


class PluginTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / "scale.py").write_text(GOOD)
        (d / "bad.py").write_text(BAD)
        (d / "blobs.py").write_text(LABELS)
        (d / "_private.py").write_text("raise RuntimeError('never imported')\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_plugin_others_could_write_is_not_loaded(self):
        def facts(mode, uid=1000):
            return os.stat_result((mode, 0, 0, 1, uid, 1000, 0, 0, 0, 0))

        path = Path(self.tmp.name) / "scale.py"
        self.assertEqual(plugins.unsafe_reason(path, facts(0o100644), uid=1000), "")
        self.assertEqual(plugins.unsafe_reason(path, facts(0o100644, uid=0), uid=1000), "")   # root's: an install
        self.assertIn("every user", plugins.unsafe_reason(path, facts(0o100666), uid=1000))
        self.assertIn("another user", plugins.unsafe_reason(path, facts(0o100644, uid=1001), uid=1000))
        self.assertIn("another user", plugins.unsafe_reason(path, facts(0o40755, uid=1001), uid=1000))
        if os.name == "posix":
            os.chmod(self.tmp.name, 0o777)
            try:
                with self.assertLogs("sirius_worker", "WARNING") as logged:
                    found, _ = plugins.load_all([self.tmp.name])
                self.assertNotIn("scale", {p.kind for p in found if p.file.parent == Path(self.tmp.name)})
                self.assertIn("writable by every user", "\n".join(logged.output))
            finally:
                os.chmod(self.tmp.name, 0o700)

    def test_load_and_validate(self):
        found, dirs = plugins.load_all([self.tmp.name])
        by_kind = {p.kind: p for p in found}
        self.assertIn("scale", by_kind)
        self.assertIn("bad", by_kind)
        self.assertNotIn("_private", by_kind)
        self.assertTrue(by_kind["bad"].error)
        self.assertIn("nope", by_kind["bad"].error)
        spec = by_kind["scale"].spec
        self.assertEqual(spec["params"][0]["max"], 10.0)
        self.assertEqual(spec["params"][1]["default"], "b")
        self.assertTrue(spec["help"].startswith("# Scale"))
        self.assertTrue(spec["separable_over_t"])
        self.assertEqual(dirs[0], self.tmp.name)

    def test_run_in_process(self):
        found, _ = plugins.load_all([self.tmp.name])
        scale = next(p for p in found if p.kind == "scale")
        data = np.ones((1, 1, 2, 3, 3), dtype=np.float32)
        msgs = []
        out, labels, diag, meta = plugins.run_plugin(scale, data, {"factor": 3}, {}, None,
                                                     progress=lambda f, m: msgs.append(m))
        self.assertEqual(out.shape, data.shape)
        self.assertEqual(out[0, 0, 1, 1, 1], 3.0)
        self.assertIsNone(labels)
        self.assertEqual(diag["summary"], "scaled")
        self.assertEqual(msgs, ["half"])
        blobs = next(p for p in found if p.kind == "blobs")
        out, labels, _, _ = plugins.run_plugin(blobs, data, {}, {}, None)
        self.assertEqual(labels.shape, (1, 2, 3, 3))
        self.assertEqual(labels.dtype, np.uint32)

    def test_over_the_socket(self):
        os.environ["SIRIUS_PLUGIN_DIRS"] = self.tmp.name
        try:
            server = WorkerServer("127.0.0.1", 0, token="t", device="cpu")
            port = server.bind()
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            with socket.create_connection(("127.0.0.1", port)) as conn:
                def call(rid, method, params, tensors=None):
                    conn.sendall(encode_frame({"id": rid, "type": "request", "method": method, "params": params}, tensors))
                    while True:
                        header, tens = read_frame(conn)
                        if header.get("type") == "progress":
                            continue
                        return header, tens
                from sirius_worker.protocol import client_handshake

                h = client_handshake(conn, "t", PROTOCOL_VERSION, first_id=100)
                self.assertIn("run:plugin", h["result"]["methods"])
                h, _ = call(2, "list_plugins", {})
                kinds = {p["kind"]: p for p in h["result"]["plugins"]}
                self.assertIn("scale", kinds)
                self.assertIn("error", kinds["bad"])
                data = np.full((1, 2, 2, 3, 3), 2.0, dtype=np.float32)
                h, t = call(3, "run", {"kind": "plugin", "plugin": "scale", "params": {"factor": 4}}, {"input": data})
                self.assertEqual(h["type"], "result", h)
                self.assertEqual(t["output"].shape, data.shape)
                self.assertEqual(float(t["output"][0, 1, 1, 1, 1]), 8.0)
                self.assertEqual(h["result"]["diagnostics"]["images"][0]["tensor"], "image0")
                self.assertEqual(t["image0"].shape, (3, 3))
                h, _ = call(4, "run", {"kind": "plugin", "plugin": "nope"}, {"input": data})
                self.assertEqual(h["type"], "error")
                call(5, "shutdown", {})
            thread.join(timeout=5)
        finally:
            os.environ.pop("SIRIUS_PLUGIN_DIRS", None)


if __name__ == "__main__":
    unittest.main()
