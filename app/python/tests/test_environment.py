"""Tests of the worker's start-up checks (sirius_worker.__main__): the --check
report, exit code 3 with the missing_packages line when a required package is
absent, and the requirement files agreeing with REQUIRED and OPTIONAL, which
SIRIUS's own Python environment is set up from (app/core/python_env.cpp).

None of them needs numpy, and none imports it: they pass in an interpreter
with numpy and in one without.

    python -m unittest discover -s app/python/tests
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import unittest
import unittest.mock

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(HERE)  # app/python
sys.path.insert(0, PYTHON_DIR)

import sirius_worker  # noqa: E402
from sirius_worker import __main__ as worker_main  # noqa: E402


def run_python(*args: str) -> subprocess.CompletedProcess:
    """This interpreter in app/python, as the application starts the worker:
    sirius_worker is found in the working directory."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, *args], cwd=PYTHON_DIR, env=env, capture_output=True, text=True,
                          encoding="utf-8", timeout=120, check=False)


def json_lines(text: str) -> list:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def requirement_names(path: str) -> list:
    """The distribution names a requirements file lists (comments and
    version specifiers dropped)."""
    names = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                names.append(re.split(r"[<>=!~;\[ ]", line, maxsplit=1)[0].lower())
    return names


class TestCheck(unittest.TestCase):
    def test_check_prints_one_json_line_and_exits_0(self):
        result = run_python("-m", "sirius_worker", "--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = json_lines(result.stdout)
        self.assertEqual(len(lines), 1, result.stdout)
        report = lines[0]
        for key, kind in (("python", str), ("executable", str), ("base_executable", str), ("version", str),
                          ("venv", bool), ("externally_managed", bool), ("pip", bool), ("ensurepip", bool),
                          ("free_threaded", bool), ("bits", int), ("missing", list), ("optional", dict),
                          ("packages", dict)):
            self.assertIsInstance(report.get(key), kind, key)
        self.assertRegex(report["version"], r"^\d+\.\d+\.\d+$")
        self.assertIn(report["bits"], (32, 64))
        self.assertEqual(sorted(report["optional"]), sorted(sirius_worker.OPTIONAL.values()))
        numpy_present = importlib.util.find_spec("numpy") is not None
        self.assertEqual(report["missing"], [] if numpy_present else ["numpy"])
        self.assertEqual("numpy" in report["packages"], numpy_present)

    def test_check_imports_none_of_the_packages(self):
        code = ("import json, sys\n"
                "from sirius_worker.__main__ import main\n"
                "rc = main(['--check'])\n"
                "heavy = sorted(m for m in ('numpy', 'scipy', 'skimage', 'torch', 'sirius_worker.models') if m in sys.modules)\n"
                "print(json.dumps({'rc': rc, 'imported': heavy}))\n")
        result = run_python("-c", code)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = json_lines(result.stdout)
        self.assertEqual(len(lines), 2, result.stdout)
        self.assertEqual(lines[1], {"rc": 0, "imported": []})


class TestMissingPackages(unittest.TestCase):
    def test_main_returns_3_with_the_missing_packages_line(self):
        # _missing_required is replaced in the child, so this runs the same
        # with numpy installed; nothing past the check may be imported.
        code = ("import json, sys\n"
                "import sirius_worker.__main__ as m\n"
                "m._missing_required = lambda: ['numpy']\n"
                "rc = m.main(['--port', '0'])\n"
                "print(json.dumps({'rc': rc, 'models': 'sirius_worker.models' in sys.modules}))\n")
        result = run_python("-c", code)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = json_lines(result.stdout)
        self.assertEqual(len(lines), 2, result.stdout)
        line, outcome = lines
        self.assertEqual(outcome, {"rc": 3, "models": False})
        self.assertEqual(line["error"], "missing_packages")
        self.assertEqual(line["missing"], ["numpy"])
        self.assertTrue(line["python"])
        self.assertRegex(line["version"], r"^\d+\.\d+\.\d+$")
        self.assertIn("missing packages: numpy (not installed in", result.stderr)

    def test_exit_code_of_the_process_is_3(self):
        code = ("import sys\n"
                "import sirius_worker.__main__ as m\n"
                "m._missing_required = lambda: ['numpy']\n"
                "sys.exit(m.main([]))\n")
        result = run_python("-c", code)
        self.assertEqual(result.returncode, worker_main.EXIT_MISSING_PACKAGES)
        self.assertEqual(worker_main.EXIT_MISSING_PACKAGES, 3)

    def test_missing_required_asks_find_spec_for_each_required_module(self):
        with unittest.mock.patch("importlib.util.find_spec", return_value=None) as find_spec:
            self.assertEqual(worker_main._missing_required(), list(sirius_worker.REQUIRED))
        self.assertEqual(sorted(c.args[0] for c in find_spec.call_args_list), sorted(sirius_worker.REQUIRED))
        expected = [] if importlib.util.find_spec("numpy") is not None else ["numpy"]
        self.assertEqual(worker_main._missing_required(), expected)


class TestRequirementFiles(unittest.TestCase):
    def test_requirements_txt_lists_exactly_the_required_distributions(self):
        names = requirement_names(os.path.join(PYTHON_DIR, "requirements.txt"))
        self.assertEqual(sorted(names), sorted(sirius_worker.REQUIRED.values()))

    def test_extra_requirements_are_optional_distributions(self):
        names = requirement_names(os.path.join(PYTHON_DIR, "requirements-extra.txt"))
        self.assertTrue(names)
        self.assertLessEqual(set(names), set(sirius_worker.OPTIONAL.values()))

    def test_required_and_optional_do_not_overlap(self):
        self.assertFalse(set(sirius_worker.REQUIRED) & set(sirius_worker.OPTIONAL))
        self.assertFalse(set(sirius_worker.REQUIRED.values()) & set(sirius_worker.OPTIONAL.values()))


if __name__ == "__main__":
    unittest.main()
