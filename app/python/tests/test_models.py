"""Tests of the worker's model hub: spec parsing, the download cache layout,
model families reporting their availability, cache deletion, and the hub_* RPC methods.

Hub calls go to a tiny public repository (hf-internal-testing/tiny-random-bert)
and are skipped when huggingface_hub is missing or the Hub is unreachable;
Cellpose / micro-SAM runs are skipped when those packages are not installed.

    python -m unittest app/python/tests/test_models.py
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # app/python
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "bindings", "python"))

from sirius_worker import models  # noqa: E402
from tests.test_protocol import ServerTestCase, _Client  # noqa: E402

TINY_REPO = "hf-internal-testing/tiny-random-bert"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))


def workbench():
    """This checkout's bindings/python/sirius/workbench.py, loaded from the
    file so the test covers it even when another sirius package is installed."""
    import importlib.util

    existing = sys.modules.get("sirius_workbench_checkout")
    if existing is not None:
        return existing
    path = os.path.join(ROOT, "bindings", "python", "sirius", "workbench.py")
    spec = importlib.util.spec_from_file_location("sirius_workbench_checkout", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sirius_workbench_checkout"] = module
    spec.loader.exec_module(module)
    return module

try:
    import huggingface_hub  # type: ignore  # noqa: F401

    HAVE_HF = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_HF = False


def _online() -> bool:
    if os.environ.get("HF_HUB_OFFLINE") or os.environ.get("SIRIUS_OFFLINE"):
        return False
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
        return True
    except OSError:
        return False


ONLINE = HAVE_HF and _online()


class _CacheCase(unittest.TestCase):
    """Points $SIRIUS_MODEL_CACHE at a fresh directory for the test."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = os.environ.get("SIRIUS_MODEL_CACHE")
        os.environ["SIRIUS_MODEL_CACHE"] = self.tmp.name

    def tearDown(self):
        if self._old is None:
            os.environ.pop("SIRIUS_MODEL_CACHE", None)
        else:
            os.environ["SIRIUS_MODEL_CACHE"] = self._old
        self.tmp.cleanup()


class TestSpecs(unittest.TestCase):
    def test_local_path(self):
        s = models.parse_spec("/data/models/unet.pt")
        self.assertEqual((s.family, s.name), ("file", "/data/models/unet.pt"))
        self.assertTrue(models.is_file_spec("C:/x/y.onnx"))
        self.assertFalse(models.is_family_spec("model.pt"))

    def test_hugging_face(self):
        s = models.parse_spec("hf:someone/unet3d")
        self.assertEqual((s.family, s.name, s.filename), ("hf", "someone/unet3d", ""))
        s = models.parse_spec("hf:someone/unet3d:weights/model.pt")
        self.assertEqual(s.filename, "weights/model.pt")
        self.assertEqual(s.text(), "hf:someone/unet3d:weights/model.pt")
        self.assertEqual(models.parse_spec("HF://someone/unet3d").name, "someone/unet3d")
        for bad in ("hf:", "hf:noslash", "hf:a/b/c"):
            with self.assertRaises(models.ModelError):
                models.parse_spec(bad)

    def test_families(self):
        s = models.parse_spec("cellpose:cyto3")
        self.assertEqual((s.family, s.name), ("cellpose", "cyto3"))
        self.assertEqual(models.parse_spec("cellpose:/models/custom_cp").name, "/models/custom_cp")
        s = models.parse_spec("microsam:vit_b_lm")
        self.assertEqual((s.family, s.name), ("microsam", "vit_b_lm"))
        self.assertEqual(models.parse_spec("micro-sam:vit_l_lm").family, "microsam")
        self.assertTrue(models.is_family_spec("cellpose:nuclei"))
        with self.assertRaises(models.ModelError):
            models.parse_spec("cellpose:")   # no default: name the model
        with self.assertRaises(models.ModelError):
            models.parse_spec("microsam:")
        with self.assertRaises(models.ModelError):
            models.parse_spec("")


class TestCache(_CacheCase):
    def test_layout(self):
        self.assertEqual(str(models.cache_dir()), self.tmp.name)
        self.assertEqual(models.repo_dir("owner/repo"), models.cache_dir() / "hf" / "owner--repo")
        self.assertIsNone(models.cached_path("owner/repo", "model.pt"))
        self.assertEqual(models.list_cached_models(), [])
        d = models.repo_dir("owner/repo")
        d.mkdir(parents=True)
        (d / "model.pt").write_bytes(b"\x00" * 10)
        (d / "README.md").write_text("not a model")
        self.assertEqual(models.cached_path("owner/repo", "model.pt"), str(d / "model.pt"))
        listed = models.list_cached_models()
        self.assertEqual([m["spec"] for m in listed], ["hf:owner/repo:model.pt"])
        self.assertEqual(listed[0]["bytes"], 10)
        self.assertEqual(listed[0]["repo"], "owner/repo")

    def test_deleting_a_cached_model_frees_it_and_prunes_the_directory(self):
        d = models.repo_dir("owner/repo")
        d.mkdir(parents=True)
        (d / "model.pt").write_bytes(b"\x00" * 32)
        (d / "other.pt").write_bytes(b"\x00" * 8)

        result = models.delete_cached_model(str(d / "model.pt"))
        self.assertEqual(result["bytes"], 32)
        self.assertFalse((d / "model.pt").exists())
        # the repository still holds a model, so its directory stays
        self.assertTrue(d.is_dir())
        self.assertEqual(result["removed_directories"], [])
        self.assertEqual([m["file"] for m in models.list_cached_models()], ["other.pt"])

        # the last file takes the empty directories with it, but not the cache
        result = models.delete_cached_model(str(d / "other.pt"))
        self.assertFalse(d.exists())
        self.assertIn(str(d), result["removed_directories"])
        self.assertTrue(models.cache_dir().is_dir())
        self.assertEqual(models.list_cached_models(), [])

    def test_a_whole_repository_can_go_at_once(self):
        d = models.repo_dir("owner/repo")
        (d / "nested").mkdir(parents=True)
        (d / "nested" / "a.pt").write_bytes(b"\x00" * 4)
        (d / "b.onnx").write_bytes(b"\x00" * 6)
        result = models.delete_cached_model(str(d))
        self.assertEqual(result["bytes"], 10)
        self.assertFalse(d.exists())
        self.assertEqual(models.list_cached_models(), [])

    def test_delete_refuses_anything_outside_the_cache(self):
        with tempfile.TemporaryDirectory() as outside:
            mine = os.path.join(outside, "my_own_model.pt")
            with open(mine, "wb") as f:
                f.write(b"\x00")
            with self.assertRaises(models.ModelError) as caught:
                models.delete_cached_model(mine)
            self.assertIn("not in the model cache", str(caught.exception))
            self.assertTrue(os.path.exists(mine))   # a file of the user's own is left alone
        with self.assertRaises(models.ModelError):
            models.delete_cached_model("")
        with self.assertRaises(models.ModelError):
            models.delete_cached_model(str(models.cache_dir() / "hf" / "nothing--here" / "gone.pt"))

    def test_delete_refuses_the_cache_directory_itself(self):
        # is_relative_to() is true for the root, so without an explicit guard a
        # client naming the cache directory would remove every cached model.
        keep = models.cache_dir() / "hf" / "owner--repo"
        keep.mkdir(parents=True)
        (keep / "model.pt").write_bytes(b"\x00")
        for root in (str(models.cache_dir()), str(models.cache_dir()) + os.sep):
            with self.assertRaises(models.ModelError) as caught:
                models.delete_cached_model(root)
            self.assertIn("cache itself", str(caught.exception))
        self.assertTrue((keep / "model.pt").exists())

    def test_default_cache_is_under_home(self):
        os.environ.pop("SIRIUS_MODEL_CACHE")
        self.assertEqual(models.cache_dir(), models.Path.home() / ".sirius" / "models")

    def test_workbench_shares_the_layout(self):
        wb = workbench()
        self.assertEqual(wb.model_cache_dir(), self.tmp.name)
        d = models.repo_dir("owner/repo")
        d.mkdir(parents=True)
        (d / "m.pt").write_bytes(b"\x00")
        # a cached file resolves without touching the network
        self.assertEqual(wb.resolve_model_spec("hf:owner/repo:m.pt"), str(d / "m.pt"))
        self.assertEqual(wb.resolve_model_spec("/plain/path.pt"), "/plain/path.pt")
        self.assertEqual(wb.resolve_model_spec("cellpose:cyto3"), "cellpose:cyto3")

    def test_resolve_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            models.resolve("/nonexistent/model.pt")


class TestFamilies(unittest.TestCase):
    def _importable(self, module: str) -> bool:
        try:
            __import__(module)
            return True
        except ImportError:
            return False

    def test_family_info_reports_availability_and_install_hint(self):
        for spec, module, fmt in (("cellpose:cyto3", "cellpose", "cellpose"), ("microsam:vit_b_lm", "micro_sam", "micro-sam")):
            info = models.family_info(spec)
            self.assertEqual(info["format"], fmt)
            self.assertEqual(info["returns"], "labels")
            self.assertEqual(info["available"], self._importable(module))
            if not info["available"]:
                self.assertIn("install", info["install_hint"])
            else:
                self.assertEqual(info["install_hint"], "")
        cp = models.family_info("cellpose:cyto3")
        self.assertTrue(cp["known_models"])
        if not cp["available"]:
            self.assertIn("cyto3", cp["known_models"])
        self.assertIn("vit_b_lm", models.family_info("microsam:vit_b_lm")["known_models"])

    def test_family_info_carries_the_install_plan(self):
        for spec, family in (("cellpose:cpsam", "cellpose"), ("microsam:vit_b_lm", "microsam")):
            info = models.family_info(spec)
            self.assertEqual(info["install"]["installer"], models.install_plan(family)["installer"])
            self.assertIn(info["install"]["installer"], ("pip", "conda"))
            self.assertTrue(info["install"]["display"].startswith(("pip install", "conda install")))
            self.assertEqual(info["python"], sys.executable)
        plan = models.install_plan("hf")
        self.assertEqual(plan["installer"], "pip")
        self.assertEqual(plan["command"][:4], [sys.executable, "-m", "pip", "install"])
        self.assertIn("huggingface_hub", plan["command"])
        with self.assertRaises(models.ModelError):
            models.install_plan("file")

    def test_a_cellpose_spec_must_name_a_model(self):
        # there is no default: the caller says which model, so a result can be
        # traced back to one
        with self.assertRaises(models.ModelError):
            models.parse_spec("cellpose:")
        self.assertEqual(models.parse_spec("cellpose:cpsam").name, "cpsam")
        info = models.family_info("cellpose:cpsam")
        self.assertEqual(info["model"], "cpsam")
        if info["available"]:
            self.assertIn("weights_cached", info)
            names = info["known_models"]
            if "cyto3" not in names:
                self.assertIn("cyto3", models.family_info("cellpose:cyto3")["warning"])
                with self.assertRaises(models.ModelError):
                    models.run_family("cellpose:cyto3", np.zeros((2, 8, 8), np.float32), {}, "cpu")

    def test_prepare_rejects_local_files_and_missing_packages(self):
        with self.assertRaises(models.ModelError):
            models.prepare("/some/model.pt")
        if not self._importable("micro_sam"):
            with self.assertRaises(models.NotAvailable) as cm:
                models.prepare("microsam:vit_b_lm")
            self.assertIn("micro_sam", str(cm.exception))

    def test_hub_errors_say_what_to_do(self):
        class GatedRepoError(Exception):
            pass

        class RepositoryNotFoundError(Exception):
            pass

        gated = models._hub_error(GatedRepoError("403 Client Error"), "facebook/sam3")
        self.assertIsInstance(gated, models.ModelError)
        self.assertIn("gated", str(gated))
        self.assertIn("https://huggingface.co/facebook/sam3", str(gated))
        self.assertIn("token", str(gated))
        missing = models._hub_error(RepositoryNotFoundError("404"), "owner/none")
        self.assertIn("not found", str(missing))
        other = models._hub_error(ValueError("boom\nsecond line"), "owner/x")
        self.assertEqual(str(other), "owner/x: boom")

    @unittest.skipUnless(HAVE_HF, "huggingface_hub missing (the dry run would need PyPI)")
    def test_install_dry_run_streams_output(self):
        lines = []
        r = models.install("hf", progress=lambda f, m: lines.append(m), dry_run=True)
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["returncode"], 0)
        self.assertIn("--dry-run", r["command"])
        self.assertTrue(lines and lines[0].startswith("$ "), lines[:2])
        self.assertTrue(any("huggingface" in line.lower() for line in lines), lines)

    def test_missing_package_raises_not_available(self):
        vol = np.zeros((2, 8, 8), np.float32)
        if not self._importable("cellpose"):
            with self.assertRaises(models.NotAvailable) as cm:
                models.run_family("cellpose:cyto3", vol, {}, "cpu")
            self.assertIn("pip install cellpose", str(cm.exception))
        if not self._importable("micro_sam"):
            with self.assertRaises(models.NotAvailable) as cm:
                models.run_family("microsam:vit_b_lm", vol, {}, "cpu")
            self.assertIn("micro_sam", str(cm.exception))
        with self.assertRaises(models.ModelError):
            models.run_family("/not/a/family.pt", vol, {}, "cpu")

    def test_workbench_family_specs(self):
        wb = workbench()
        info = wb.model_info("cellpose:nuclei")
        self.assertEqual(info["format"], "cellpose")
        self.assertEqual(info["available"], self._importable("cellpose"))
        with self.assertRaises(wb.NotAvailable):
            wb.load_model("microsam:vit_b_lm", "cpu")


class TestHubMethods(ServerTestCase, _CacheCase):
    def setUp(self):
        _CacheCase.setUp(self)

    def tearDown(self):
        _CacheCase.tearDown(self)

    def test_capabilities_list_hub_methods(self):
        c = _Client(self.port, self.token)
        try:
            caps = c.hello()["result"]
            for m in ("hub_search", "hub_files", "hub_download", "models_list", "model_info", "install", "model_prepare"):
                self.assertIn(m, caps["methods"])
        finally:
            c.close()

    @unittest.skipUnless(HAVE_HF, "huggingface_hub missing (the dry run would need PyPI)")
    def test_install_job_streams_progress(self):
        c = _Client(self.port, self.token)
        try:
            c.hello()
            progress, header, _ = c.call("install", {"family": "hf", "dry_run": True})
            self.assertEqual(header["type"], "result", header)
            self.assertTrue(header["result"]["ok"])
            self.assertIn("--dry-run", header["result"]["command"])
            self.assertTrue(progress, "install should stream its output lines")
            _, header, _ = c.call("install", {"family": "file"})
            self.assertEqual(header["type"], "error", header)
        finally:
            c.close()

    def test_model_prepare_reports_missing_package(self):
        try:
            import micro_sam  # type: ignore  # noqa: F401
            self.skipTest("micro_sam is installed")
        except ImportError:
            pass
        c = _Client(self.port, self.token)
        try:
            c.hello()
            _, header, _ = c.call("model_prepare", {"spec": "microsam:vit_b_lm"})
            self.assertEqual(header["type"], "error", header)
            self.assertIn("micro_sam", header["message"])
        finally:
            c.close()

    def test_models_delete_removes_the_file_and_guards_the_rest_of_the_disk(self):
        d = models.repo_dir("owner/repo")
        d.mkdir(parents=True)
        (d / "net.onnx").write_bytes(b"\x00" * 5)
        c = _Client(self.port, self.token)
        try:
            c.hello()
            _, header, _ = c.call("models_delete", {"path": str(d / "net.onnx")})
            self.assertEqual(header["type"], "result", header)
            self.assertEqual(header["result"]["bytes"], 5)
            self.assertFalse((d / "net.onnx").exists())
            _, header, _ = c.call("models_list")
            self.assertEqual(header["result"]["models"], [])
            # a path outside the cache is an error, not a deletion
            with tempfile.TemporaryDirectory() as outside:
                mine = os.path.join(outside, "mine.pt")
                with open(mine, "wb") as f:
                    f.write(b"\x00")
                _, header, _ = c.call("models_delete", {"path": mine})
                self.assertEqual(header["type"], "error", header)
                self.assertTrue(os.path.exists(mine))
        finally:
            c.close()

    def test_models_list_and_family_model_info(self):
        d = models.repo_dir("owner/repo")
        d.mkdir(parents=True)
        (d / "net.onnx").write_bytes(b"\x00" * 3)
        c = _Client(self.port, self.token)
        try:
            c.hello()
            _, header, _ = c.call("models_list")
            self.assertEqual(header["type"], "result", header)
            self.assertEqual(header["result"]["cache"], self.tmp.name)
            self.assertEqual([m["spec"] for m in header["result"]["models"]], ["hf:owner/repo:net.onnx"])
            _, header, _ = c.call("model_info", {"spec": "cellpose:cyto3"})
            self.assertEqual(header["type"], "result", header)
            self.assertEqual(header["result"]["format"], "cellpose")
            self.assertIn("available", header["result"])
            # an hf: file that is not cached yet is described, not downloaded
            _, header, _ = c.call("model_info", {"path": "hf:owner/other:big.pt"})
            self.assertEqual(header["type"], "result", header)
            self.assertEqual(header["result"]["format"], "hf")
            self.assertFalse(header["result"]["cached"])
        finally:
            c.close()

    def test_family_run_without_package_reports_install_hint(self):
        try:
            import cellpose  # type: ignore  # noqa: F401
            self.skipTest("cellpose is installed")
        except ImportError:
            pass
        c = _Client(self.port, self.token)
        try:
            c.hello()
            _, header, _ = c.call("run", {"kind": "torch_segment", "params": {"model": "cellpose:cyto3"}},
                                  {"input": np.zeros((2, 8, 8), np.float32)})
            self.assertEqual(header["type"], "error", header)
            self.assertIn("pip install cellpose", header["message"])
        finally:
            c.close()

    @unittest.skipUnless(ONLINE, "huggingface_hub missing or the Hub is unreachable")
    def test_hub_search_files_download(self):
        c = _Client(self.port, self.token)
        try:
            c.hello()
            _, header, _ = c.call("hub_search", {"query": "tiny-random-bert", "limit": 5})
            self.assertEqual(header["type"], "result", header)
            found = header["result"]["models"]
            self.assertTrue(found)
            self.assertTrue(all({"id", "downloads", "likes", "tags"} <= set(m) for m in found))
            _, header, _ = c.call("hub_files", {"repo": TINY_REPO})
            self.assertEqual(header["type"], "result", header)
            names = [f["name"] for f in header["result"]["files"]]
            self.assertIn("config.json", names)
            progress, header, _ = c.call("hub_download", {"repo": TINY_REPO, "file": "config.json"})
            self.assertEqual(header["type"], "result", header)
            path = header["result"]["path"]
            self.assertTrue(os.path.isfile(path))
            self.assertTrue(path.startswith(self.tmp.name), path)
            self.assertEqual(header["result"]["bytes"], os.path.getsize(path))
            self.assertTrue(progress, "download should stream progress frames")
            self.assertEqual(models.cached_path(TINY_REPO, "config.json"), path)
            # second download is served from the cache
            progress, header, _ = c.call("hub_download", {"repo": TINY_REPO, "file": "config.json"})
            self.assertEqual(header["result"]["path"], path)
        finally:
            c.close()

    @unittest.skipUnless(ONLINE, "huggingface_hub missing or the Hub is unreachable")
    def test_spec_without_filename_picks_the_single_model_file(self):
        # the tiny BERT repo carries exactly one .onnx next to its safetensors / .bin weights
        self.assertEqual(models.pick_model_file(TINY_REPO), "onnx/model.onnx")

    def test_ambiguous_or_empty_repos_are_reported(self):
        original = models.hub_files
        try:
            models.hub_files = lambda repo, token=None: [{"name": "a.pt", "size": 1, "model": True},
                                                        {"name": "b.onnx", "size": 1, "model": True}]
            with self.assertRaises(models.ModelError) as cm:
                models.pick_model_file("owner/two")
            self.assertIn("a.pt", str(cm.exception))
            self.assertIn("b.onnx", str(cm.exception))
            models.hub_files = lambda repo, token=None: [{"name": "README.md", "size": 1, "model": False}]
            with self.assertRaises(models.ModelError) as cm:
                models.pick_model_file("owner/none")
            self.assertIn("owner/none", str(cm.exception))
        finally:
            models.hub_files = original


if __name__ == "__main__":
    unittest.main()


class TestCachePathJail(unittest.TestCase):
    """A repository file name must not name a file outside the repository's
    cache directory (".." or an absolute path on Windows)."""

    def test_cached_path_stays_inside_the_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("SIRIUS_MODEL_CACHE")
            os.environ["SIRIUS_MODEL_CACHE"] = tmp
            try:
                repo = models.repo_dir("owner/repo")
                repo.mkdir(parents=True)
                (repo / "model.pt").write_bytes(b"x")
                outside = os.path.join(tmp, "secret.pt")
                with open(outside, "wb") as f:
                    f.write(b"y")
                self.assertEqual(models.cached_path("owner/repo", "model.pt"), str((repo / "model.pt").resolve()))
                self.assertIsNone(models.cached_path("owner/repo", "../secret.pt"))
                self.assertIsNone(models.cached_path("owner/repo", "../../secret.pt"))
                self.assertIsNone(models.cached_path("owner/repo", outside))
                self.assertIsNone(models.cached_path("owner/repo", "missing.pt"))
            finally:
                if old is None:
                    os.environ.pop("SIRIUS_MODEL_CACHE", None)
                else:
                    os.environ["SIRIUS_MODEL_CACHE"] = old


class TestRequestTokenIsPerThread(unittest.TestCase):
    def test_a_token_set_on_one_thread_is_not_seen_by_another(self):
        import threading

        models.set_hub_token("connection-token")
        seen = {}

        def job():
            seen["before"] = models.current_request_token()
            models.set_hub_token("job-token")
            seen["after"] = models.current_request_token()

        t = threading.Thread(target=job)
        t.start()
        t.join()
        self.assertIsNone(seen["before"])
        self.assertEqual(seen["after"], "job-token")
        self.assertEqual(models.current_request_token(), "connection-token")
        models.set_hub_token("")
        self.assertIsNone(models.current_request_token())
