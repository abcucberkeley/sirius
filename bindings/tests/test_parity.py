"""Numeric parity between the C++ operations and their Python mirror.

``bindings/python/sirius/workbench.py`` reimplements a dozen of the
application's steps for the exported scripts and the HPC backend, and
``test_workbench_schema.py`` guarantees the two sides agree on the *parameter
keys*. This module checks that they agree on the *numbers*, which is the
failure that would silently corrupt someone's results.

The fixtures come from the C++ side, so there is no second copy of the input
to drift:

    SIRIUS_PARITY_OUT=<dir> sirius_tests "[parity]"
    SIRIUS_PARITY_DIR=<dir> python -m unittest bindings.tests.test_parity

Without a fixture directory (the variable, or a ``parity`` directory under a
sibling ``build/*``) every case skips: the C++ tests have to run first.

Tolerances are per step and stated in ``TOLERANCES`` below with the reason
each one is not zero. Where a step is exact it is asserted exactly.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load_workbench():
    """The workbench module of this source tree (never an installed copy)."""
    here = REPO / "bindings" / "python" / "sirius" / "workbench.py"
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


try:
    wb = _load_workbench()
except Exception as exc:  # noqa: BLE001
    wb = None
    _LOAD_ERROR = exc
else:
    _LOAD_ERROR = None


def _fixture_dir():
    """The fixture directory: SIRIUS_PARITY_DIR when set, else the most
    recently written build/*/parity. Newest rather than alphabetically first,
    because a stale directory from an earlier build lacks the cases newer
    steps add, and the coverage check would then fail for the wrong reason."""
    env = os.environ.get("SIRIUS_PARITY_DIR")
    if env:
        d = Path(env)
        return d if (d / "cases.json").is_file() else None
    found = [d for d in list((REPO / "build").glob("*/parity")) + list((REPO / "build").glob("*/tests/parity"))
             if (d / "cases.json").is_file()]
    if not found:
        return None
    return max(found, key=lambda d: (d / "cases.json").stat().st_mtime)


FIXTURES = _fixture_dir()


# One ulp of a float32 in [0.5, 1): the granularity of the outputs, and the
# unit every tolerance below is quoted in.
ULP = 2.0 ** -23

# Per-step absolute / relative tolerance, and why it is what it is. The
# comparison is |a - b| <= atol + rtol * |b| on every voxel, as numpy's. The
# measured error on the committed fixture is quoted with each one: none of
# these is a bound that was loosened until the test passed.
TOLERANCES = {
    # Copies and selections: the same float32 values move, nothing is
    # computed, so these must agree bit for bit. Measured: 0.
    "croppad": (0.0, 0.0),
    "threshold": (0.0, 0.0),   # the step returns its input; the labels are the result (compared exactly below)
    # Classical segmentation also returns its input untouched -- the labels
    # carry the result and are compared exactly below, which is what makes
    # this the strict test of the filters, the thresholds and the seeding.
    "classic": (0.0, 0.0),
    # Both sides accumulate in float64 but in different orders -- the C++
    # folds (c, t, z) outermost and sums a plane sequentially, numpy sums
    # pairwise over the reduced axes -- so the float32 result may round
    # differently in the last bit. Measured on the fixture: 0.
    "einsum": (ULP, 1e-6),
    "maxproj": (ULP, 1e-6),
    "meant": (ULP, 1e-6),
    # equalizeFrames sums a frame sequentially in float64, numpy pairwise;
    # the scale is rounded to float32 on both sides before it multiplies.
    # Measured: 0.
    "bleach": (ULP, 1e-6),
    # One real (and harmless) divergence: the application carries the contrast
    # window and the gamma as float32 -- ContrastWindow::lo / ::hi and
    # rescaleGamma's `gamma` are floats -- and only then widens them to
    # float64 for (v - lo) / span and 1 / gamma. The Python mirror keeps them
    # float64 throughout, so a window or a gamma that is not exactly
    # representable in float32 (0.25 .. 0.8, gamma 0.45) shifts the result by
    # the last float32 bit. The automatic window comes from float32 samples,
    # is therefore exact in both, and matches bit for bit. Measured: 5.96e-08
    # absolute, i.e. one ulp -- 3.5e-05 relative, but only on voxels a few
    # ulp away from zero, which is why this bound is absolute.
    "contrast": (ULP, 1e-6),
    # The other real difference. resampleAffine gathers every output voxel
    # from a 3D neighbourhood in one pass; the Python mirror applies the same
    # taps separably, one axis at a time, with a float32 array between axes.
    # The taps are identical -- nearest is exact -- but linear and cubic
    # round twice more. Measured: 1.19e-07, two ulp. Matching bit for bit
    # would mean giving the mirror the C++ loop order, five times slower in
    # numpy for no numerical gain.
    "resample": (2.0 * ULP, 1e-6),
    # The RGB blend is float32 on both sides in the same order: exact.
    "merge": (0.0, 0.0),
}


def _read_f32(path: Path, shape):
    a = np.fromfile(path, dtype="<f4")
    return a.reshape(shape)


def _read_u32(path: Path, shape):
    a = np.fromfile(path, dtype="<u4")
    return a.reshape(shape)


def _worst(actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float):
    """(index, actual, expected, absolute error) of the voxel that misses the
    tolerance by the most, or None when every voxel is inside it."""
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    err = np.abs(a - b)
    allowed = atol + rtol * np.abs(b)
    over = np.where(both_nan, -1.0, err - allowed)
    k = int(np.argmax(over))
    if over.flat[k] <= 0.0:
        return None
    idx = np.unravel_index(k, a.shape)
    return idx, float(a.flat[k]), float(b.flat[k]), float(err.flat[k])


@unittest.skipIf(wb is None, f"sirius.workbench did not import: {_LOAD_ERROR}")
@unittest.skipIf(FIXTURES is None,
                 "no parity fixtures: run SIRIUS_PARITY_OUT=<dir> sirius_tests \"[parity]\" first "
                 "and point SIRIUS_PARITY_DIR at <dir>")
class TestParity(unittest.TestCase):
    """Every case in cases.json, as one subTest each."""

    @classmethod
    def setUpClass(cls):
        with open(FIXTURES / "input.json", encoding="utf-8") as f:
            spec = json.load(f)
        cls.input = _read_f32(FIXTURES / "input.f32", tuple(spec["dims"]))
        # the same array as 16-bit camera counts, for the cases that name it
        cls.inputs = {"input": cls.input}
        if (FIXTURES / "input16.f32").is_file():
            cls.inputs["input16"] = _read_f32(FIXTURES / "input16.f32", tuple(spec["dims"]))
        cls.meta = {"voxel_um": list(spec["voxel_um"]), "dims": {}}
        with open(FIXTURES / "cases.json", encoding="utf-8") as f:
            cls.cases = json.load(f)["cases"]

    def _run(self, case):
        kind = case["kind"]
        meta = dict(self.meta)
        with warnings.catch_warnings():
            warnings.simplefilter("error", wb.UnknownParameterWarning)
            try:
                return wb.run_step(kind, dict(case["params"]), self.inputs[case.get("input", "input")], meta)
            except wb.NotAvailable as e:   # scipy / scikit-image missing
                self.skipTest(f"{kind}: {e}")

    def _check_case(self, case):
        kind, name = case["kind"], case["name"]
        atol, rtol = TOLERANCES[kind]
        expected = _read_f32(FIXTURES / f"{name}.f32", tuple(case["dims"]))
        got = self._run(case).array
        self.assertEqual(got.shape, expected.shape,
                         f"parity[{name}] ({kind}): shape {got.shape} from Python, {expected.shape} from C++")
        worst = _worst(got, expected, atol, rtol)
        if worst is not None:
            idx, py, cxx, err = worst
            self.fail(f"parity[{name}] ({kind}) exceeds atol={atol:.3e} rtol={rtol:.3e} at voxel "
                      f"(c,t,z,y,x)={tuple(int(i) for i in idx)}: Python {py!r} vs C++ {cxx!r} "
                      f"(|delta| = {err:.3e}, allowed {atol + rtol * abs(cxx):.3e}, "
                      f"{err / ULP:.1f} float32 ulp)")

    def _check_labels(self, case):
        name = case["name"]
        expected = _read_u32(FIXTURES / f"{name}.u32", tuple(case["labels_dims"]))
        got = self._run(case).labels
        self.assertIsNotNone(got, f"parity[{name}]: the Python step produced no labels, the C++ one did")
        self.assertEqual(got.shape, expected.shape,
                         f"parity[{name}] labels: shape {got.shape} from Python, {expected.shape} from C++")
        # Label ids, not just the foreground mask: both sides number connected
        # components densely in raster order of first appearance, so an equal
        # partition must come out equally numbered.
        bad = np.argwhere(got != expected)
        self.assertEqual(
            len(bad), 0,
            f"parity[{name}] labels differ at {len(bad)} voxels, first (t,z,y,x)={tuple(bad[0]) if len(bad) else ()}: "
            f"Python {got[tuple(bad[0])] if len(bad) else 0} vs C++ {expected[tuple(bad[0])] if len(bad) else 0}; "
            f"foreground masks {'agree' if np.array_equal(got > 0, expected > 0) else 'DIFFER'}, "
            f"counts Python {int(got.max())} vs C++ {int(expected.max())}")

    def test_every_case(self):
        """Every fixture case, so one run names every step that moved."""
        for case in self.cases:
            with self.subTest(case=case["name"], kind=case["kind"]):
                self._check_case(case)
                if case.get("labels"):
                    self._check_labels(case)

    def test_every_kind_is_covered(self):
        """The fixture list must exercise every kind it claims to."""
        kinds = {c["kind"] for c in self.cases}
        self.assertEqual(kinds, set(TOLERANCES),
                         "TOLERANCES and the fixture kinds have drifted apart")


@unittest.skipIf(wb is None, f"sirius.workbench did not import: {_LOAD_ERROR}")
@unittest.skipIf(FIXTURES is None or not (FIXTURES / "loader.json").is_file(),
                 "no loader fixtures: run SIRIUS_PARITY_OUT=<dir> sirius_tests \"[parity]\" first")
class TestLoaderParity(unittest.TestCase):
    """The TIFFs the fixture writer made, opened by the application's Load
    step and by run_pipeline's loader: the same array, dimensions, voxel size,
    frame interval and channels, exactly. The dimensions are where the two
    used to part: ImageJ / OME files with three axes or fewer, counts that do
    not divide the pages, length units."""

    def test_every_file(self):
        with open(FIXTURES / "loader.json", encoding="utf-8") as f:
            cases = json.load(f)["cases"]
        self.assertTrue(cases)
        for case in cases:
            with self.subTest(case=case["name"]):
                pipeline = [{"kind": "load", "params": dict(case["params"])}]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", wb.UnknownParameterWarning)
                        got, meta = wb.run_pipeline(str(FIXTURES / case["file"]), pipeline)
                except wb.NotAvailable as e:   # neither tifffile nor the extension
                    self.skipTest(str(e))
                name = case["name"]
                self.assertEqual(list(got.shape), case["dims"], f"load[{name}]: dimensions")
                expected = _read_f32(FIXTURES / f"load_{name}.f32", tuple(case["dims"]))
                self.assertTrue(np.array_equal(got, expected), f"load[{name}]: the planes differ")
                self.assertEqual(meta["voxel_um"], case["voxel_um"], f"load[{name}]: voxel size")
                self.assertEqual(meta["frame_interval_s"], case["frame_interval_s"], f"load[{name}]: frame interval")
                self.assertEqual(meta["format"], case["format"])
                self.assertEqual([[ch["label"], ch["wavelength_nm"], ch["color"]] for ch in meta["channels"]],
                                 [[ch["label"], ch["wavelength_nm"], ch["color"]] for ch in case["channels"]],
                                 f"load[{name}]: channels")


if __name__ == "__main__":
    unittest.main()
