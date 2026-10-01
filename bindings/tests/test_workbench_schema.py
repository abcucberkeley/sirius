"""Drift test between the application's operations and their Python mirror.

``bindings/python/sirius/op_schema.json`` is a snapshot of the C++ parameter
tables (``OpInfo.params`` of every built-in operation), written by
``tests/test_app_schema.cpp`` (``SIRIUS_OP_SCHEMA_OUT=... sirius_tests
"[schema]"``). Every kind in it must be implemented by ``sirius.workbench``
with a StepSpec that declares exactly the C++ keys, defaults and choices, or
be listed as unsupported / pass-through -- so a parameter renamed in
app/core/ops fails here instead of silently changing what an exported
pipeline computes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE.parents[0] / "python" / "sirius" / "op_schema.json"


def _load_workbench():
    here = HERE.parents[0] / "python" / "sirius" / "workbench.py"
    try:
        import sirius.workbench as wb  # type: ignore

        if Path(wb.__file__).resolve() == here:
            return wb
    except Exception:  # noqa: BLE001
        pass
    name = "sirius_workbench_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, here)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


wb = _load_workbench()


def _operations():
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    return {op["kind"]: op for op in schema["operations"] if not op.get("plugin")}


def _same_default(python_value, cxx_value) -> bool:
    if isinstance(cxx_value, bool) or isinstance(python_value, bool):
        return bool(python_value) == bool(cxx_value)
    if isinstance(cxx_value, (int, float)) and isinstance(python_value, (int, float)):
        return abs(float(python_value) - float(cxx_value)) <= 1e-12 * max(1.0, abs(float(cxx_value)))
    if isinstance(cxx_value, list) and isinstance(python_value, (list, tuple)):
        return len(cxx_value) == len(python_value) and all(_same_default(p, c) for p, c in zip(python_value, cxx_value))
    return python_value == cxx_value


@unittest.skipUnless(SCHEMA_PATH.is_file(), f"{SCHEMA_PATH} missing: run sirius_tests \"[schema]\" with SIRIUS_OP_SCHEMA_OUT")
class TestOperationSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ops = _operations()

    def test_snapshot_lists_the_built_in_kinds(self):
        for kind in ("einsum", "maxproj", "meant", "contrast", "flatfield", "bleach", "croppad", "resample", "merge",
                     "threshold", "classic", "cleanup", "seg", "sim", "load"):
            self.assertIn(kind, self.ops)

    def test_every_kind_is_implemented_unsupported_or_passthrough(self):
        for kind in self.ops:
            handled = kind in wb._STEPS or kind in wb._UNSUPPORTED or kind in wb._PASSTHROUGH
            self.assertTrue(handled, f"kind '{kind}' is neither implemented, unsupported nor pass-through in workbench.py")
            if kind in wb._STEPS:
                self.assertNotIn(kind, wb._UNSUPPORTED, kind)
        for kind in list(wb._STEPS) + list(wb._SPECS):
            self.assertIn(kind, self.ops, f"workbench.py declares '{kind}', which the application does not register")
        for kind in wb._UNSUPPORTED:
            self.assertIn(kind, self.ops, f"_UNSUPPORTED names '{kind}', which the application does not register")

    def test_keys_match(self):
        for kind, spec in wb._SPECS.items():
            cxx = [p["key"] for p in self.ops[kind]["params"]]
            self.assertEqual(list(spec.keys), cxx, f"'{kind}': Python keys {list(spec.keys)} vs C++ keys {cxx}")

    def test_defaults_match(self):
        for kind, spec in wb._SPECS.items():
            for p in self.ops[kind]["params"]:
                self.assertTrue(_same_default(spec.defaults[p["key"]], p["default"]),
                                f"'{kind}.{p['key']}': Python default {spec.defaults[p['key']]!r} vs C++ {p['default']!r}")

    def test_choices_match(self):
        for kind, spec in wb._SPECS.items():
            choice_keys = {p["key"] for p in self.ops[kind]["params"] if p["type"] == "choice"}
            self.assertEqual(set(spec.choices), choice_keys, f"'{kind}': choice parameters differ")
            for p in self.ops[kind]["params"]:
                if p["type"] == "choice":
                    self.assertEqual(list(spec.choices[p["key"]]), p["choices"], f"'{kind}.{p['key']}' choices")

    def test_types_are_consistent_with_the_defaults(self):
        expect = {"bool": bool, "int": int, "channel": int, "double": (int, float), "string": str, "path": str,
                  "choice": str, "axes": str, "double_list": list, "string_list": list, "prompts": list}
        for kind, spec in wb._SPECS.items():
            for p in self.ops[kind]["params"]:
                d = spec.defaults[p["key"]]
                self.assertIsInstance(d, expect[p["type"]], f"'{kind}.{p['key']}' is {p['type']} but the Python default is {d!r}")
                if p["type"] in ("int", "channel"):
                    self.assertNotIsInstance(d, bool)

    def test_aliases_and_extras_do_not_collide_with_canonical_keys(self):
        for kind, spec in wb._SPECS.items():
            for alias, target in spec.aliases.items():
                self.assertNotIn(alias, spec.defaults, f"'{kind}': alias '{alias}' is a canonical key")
                self.assertIn(target, spec.defaults, f"'{kind}': alias '{alias}' points at unknown key '{target}'")
            for extra in spec.extra:
                self.assertNotIn(extra, spec.defaults, f"'{kind}': extra '{extra}' is a canonical key")
                self.assertNotIn(extra, spec.aliases, f"'{kind}': extra '{extra}' is also an alias")

    def test_canonical_parameters_run_without_warnings(self):
        """The defaults of every implemented kind are accepted silently (the
        keys a saved pipeline carries are exactly these)."""
        a = np.zeros((1, 1, 2, 4, 4), np.float32)
        for kind, spec in wb._SPECS.items():
            params = {p["key"]: p["default"] for p in self.ops[kind]["params"]}
            with warnings.catch_warnings():
                warnings.simplefilter("error", wb.UnknownParameterWarning)
                wb._prepare_params(spec, params, wb._default_meta(a))

    def test_unknown_key_warns(self):
        a = np.zeros((1, 1, 2, 4, 4), np.float32)
        with self.assertWarns(wb.UnknownParameterWarning):
            wb.run_step("meant", {"bogus": 1}, a)


if __name__ == "__main__":
    unittest.main()
