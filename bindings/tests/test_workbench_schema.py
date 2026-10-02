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
                     "classic", "cleanup", "seg", "sim", "load"):
            self.assertIn(kind, self.ops)

    def test_removed_kinds_are_gone_from_both_sides(self):
        for kind in wb._REMOVED:
            self.assertNotIn(kind, self.ops, f"'{kind}' is listed as removed but the application still registers it")
            self.assertNotIn(kind, wb._STEPS)
            self.assertNotIn(kind, wb._UNSUPPORTED)

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


class TestPromptObjects(unittest.TestCase):
    """The Prompt step's request, as the application builds it
    (tests/test_app_ops.cpp "Each time point sends its objects ...")."""

    def test_each_time_point_sends_its_objects(self):
        prompts = [{"x": 10, "y": 11, "z": 1, "object": 1},
                   {"x": 20, "y": 21, "z": 2, "t": 1, "object": 2},
                   {"kind": "scribble", "points": [[1, 1, 1], [2, 1, 1], [3, 2, 1]], "object": 3},
                   {"x": 30, "y": 31, "z": 3, "label": 0, "object": 1},
                   {"kind": "box", "x0": 0, "y0": 0, "z0": 0, "x1": 4, "y1": 4, "z1": 2, "object": 4},
                   {"x": 2, "y": 2, "z": 1, "object": 4},
                   {"x": 50, "y": 51, "z": 5, "t": 2, "label": 0, "object": 5}]
        objects, ids = wb._frame_objects(prompts, 0)
        self.assertEqual(ids, [1, 3, 4])
        self.assertEqual(objects, [
            {"points": [[10.0, 11.0, 1.0], [30.0, 31.0, 3.0]], "point_labels": [1, 0]},
            {"scribbles": [{"points": [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [3.0, 2.0, 1.0]], "label": 1}]},
            {"box": [0.0, 0.0, 0.0, 4.0, 4.0, 2.0], "points": [[2.0, 2.0, 1.0]], "point_labels": [1]}])
        self.assertEqual(wb._frame_objects(prompts, 1)[1], [2])
        self.assertEqual(wb._frame_objects(prompts, 2), ([], []))   # background only: not sent
        lab = wb._renumber_prompt_masks(np.array([0, 1, 2, 3, 4]), ids)
        self.assertEqual(lab.tolist(), [0, 1, 3, 4, 0])

    def test_a_list_without_objects_gets_them_as_the_application_gives_them(self):
        # tests/test_app_pipeline.cpp "A pipeline written before prompt objects still loads"
        old = [{"kind": "box", "x0": 1, "y0": 2, "z0": 0, "x1": 6, "y1": 7, "z1": 3, "t": 0},
               {"kind": "point", "x": 30, "y": 30, "z": 1, "t": 0, "label": 1},
               {"kind": "point", "x": 28, "y": 31, "z": 1, "t": 0, "label": 0},
               {"kind": "point", "x": 7, "y": 3, "z": 1, "t": 0, "label": 0},
               {"kind": "point", "x": 9, "y": 9, "z": 1, "t": 1, "label": 0},
               {"kind": "scribble", "points": [[2, 2, 1], [3, 2, 1]], "t": 1, "label": 1}]
        self.assertEqual([i for _, i in wb._prompt_objects(old)], [1, 2, 2, 1, 3, 3])
        stray = [{"x": 4, "y": 4, "z": 1, "label": 0}, {"x": 5, "y": 4, "z": 1, "label": 0}, {"x": 9, "y": 9, "z": 0, "t": 1}]
        self.assertEqual([i for _, i in wb._prompt_objects(stray)], [2, 2, 1])


if __name__ == "__main__":
    unittest.main()
