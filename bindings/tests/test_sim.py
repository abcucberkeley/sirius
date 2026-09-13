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
