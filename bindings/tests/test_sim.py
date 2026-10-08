"""Python API tests for CPU/GPU SIM reconstruction."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import sirius

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


class TestSIMParameters(unittest.TestCase):
    def test_defaults_validate_and_fields_are_mutable(self):
        params = sirius.SIMParameters()
        params.wiener = 0.002
        params.k0_angles = [0.1, 1.2, 2.3]
        params.validate()
        self.assertEqual(params.wiener, 0.002)
        self.assertEqual(params.k0_angles, [0.1, 1.2, 2.3])

    def test_orders_are_derived_by_default(self):
        # norders used to default to 3, so 3 phases (2D SIM) failed to separate
        params = sirius.SIMParameters()
        self.assertEqual(params.norders, 0)
        params.nphases = 3
        params.validate()
        params.norders = 1
        with self.assertRaises(RuntimeError):
            params.validate()

    def test_legacy_config_maps_reference_dataset(self):
        params = sirius.load_legacy_parameters(str(DATA / "config.txt"))
        self.assertEqual(params.ndirs, 3)
        self.assertEqual(params.nphases, 5)
        self.assertAlmostEqual(params.dx, 0.08, places=6)
        self.assertTrue(params.dampen_order0)


class TestSIMReconstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = sirius.load_legacy_parameters(str(DATA / "config.txt"))
        cls.raw = sirius.read_tiff(str(DATA / "raw.tif"), dtype=np.float64)
        cls.expected = sirius.read_tiff(str(DATA / "raw_proc.tif"), dtype=np.float64)

    def test_cpu_reconstructs_reference_and_reuses_instance(self):
        recon = sirius.SimReconstructor(
            self.params, str(DATA / "otf.tif"), rigor=sirius.PlanRigor.Estimate
        )
        actual = recon.reconstruct(self.raw)
        self.assertEqual(actual.shape, self.expected.shape)
        rel = np.max(np.abs(actual - self.expected)) / np.max(np.abs(self.expected))
        self.assertLess(rel, 2e-6)
        self.assertEqual(len(recon.last_fit.k0), self.params.ndirs)
        again = recon.reconstruct(self.raw)
        np.testing.assert_array_equal(again, actual)

    def test_no_otf_path_is_the_theoretical_otf(self):
        # The GUI and the CLI leave the OTF field empty to mean "theoretical
        # OTF"; the binding has to mean the same thing, through the same
        # library function (sirius::selectOTF). Before 2026-10-08 the binding
        # called loadOTF unconditionally, so an empty path was a failed file
        # open and the Python front could not reconstruct without a measured
        # OTF at all (docs/findings.md 9k.50, finding 4).
        params = sirius.load_legacy_parameters(str(DATA / "config.txt"))
        self.assertEqual(params.sections_per_plane(), 15)
        self.assertEqual(params.planes(self.raw.shape[0]), 9)   # several planes: the 3D OTF
        recon = sirius.SimReconstructor(params, "", rigor=sirius.PlanRigor.Estimate,
                                        three_d=params.planes(self.raw.shape[0]) > 1)
        actual = recon.reconstruct(self.raw)
        self.assertEqual(actual.shape, self.expected.shape)
        self.assertEqual(int(np.count_nonzero(~np.isfinite(actual))), 0)
        self.assertGreater(float(np.max(np.abs(actual))), 0.0)
        self.assertEqual(len(recon.last_fit.k0), params.ndirs)
        # the default is the theoretical OTF too: otf_path is optional
        again = sirius.SimReconstructor(params, rigor=sirius.PlanRigor.Estimate)
        peak = float(np.max(np.abs(actual)))
        np.testing.assert_allclose(again.reconstruct(self.raw), actual, rtol=0, atol=1e-9 * peak)
        # (the 2D table itself -- one kz sample instead of 64 -- is checked in
        # tests/test_otf_select.cpp; reconstructing a nine-plane stack with it
        # is a configuration no front can ask for, so it is not asserted here)

    @unittest.skipUnless(sirius.cuda_available(), "no CUDA device available")
    def test_gpu_buffer_input_and_output(self):
        device = sirius.Device.cuda()
        raw = sirius.to_device(self.raw, device)
        recon = sirius.SimReconstructor(
            self.params, str(DATA / "otf.tif"), device=device,
            rigor=sirius.PlanRigor.Estimate,
        )
        output = recon.reconstruct(raw)
        self.assertIsInstance(output, sirius.Buffer)
        actual = output.numpy()
        rel = np.max(np.abs(actual - self.expected)) / np.max(np.abs(self.expected))
        self.assertLess(rel, 2e-6)


if __name__ == "__main__":
    unittest.main()
