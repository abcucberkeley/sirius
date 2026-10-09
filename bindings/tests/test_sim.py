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
        # the theoretical OTF is the default too: otf_path is optional
        again = sirius.SimReconstructor(params, rigor=sirius.PlanRigor.Estimate,
                                        three_d=params.planes(self.raw.shape[0]) > 1)
        peak = float(np.max(np.abs(actual)))
        np.testing.assert_allclose(again.reconstruct(self.raw), actual, rtol=0, atol=1e-9 * peak)
        # (the 2D table itself -- one kz sample instead of 64 -- is checked in
        # tests/test_otf_select.cpp; reconstructing a nine-plane stack with it
        # is a configuration no front can ask for, so it is not asserted here)

    def test_three_d_has_to_be_given_because_the_data_decides_it(self):
        """three_d defaulted to True, so a ONE-PLANE stack built with the
        Python default got the 3D theoretical OTF -- missing cone, order 1
        shifted by the illumination's kz -- which no front would choose for
        it: the library rule is that a stack is 3D when it has more than one
        plane (SIMParameters::planes(sections) > 1, session.cpp's threeD()).
        The constructor is handed parameters, not a stack, so it cannot derive
        the value; required where it means something is the only answer that
        is not a guess (the Python front's finding B).
        """
        params = sirius.load_legacy_parameters(str(DATA / "config.txt"))
        with self.assertRaises(ValueError) as cm:
            sirius.SimReconstructor(params, "", rigor=sirius.PlanRigor.Estimate)
        self.assertIn("three_d has to be given when no OTF file is named", str(cm.exception))
        self.assertIn("parameters.planes(sections) > 1", str(cm.exception))
        # named explicitly: both tables are reachable, either way round
        for three_d in (True, False):
            sirius.SimReconstructor(params, "", rigor=sirius.PlanRigor.Estimate, three_d=three_d)
        # and it is ignored, so it may be left out, when a file is named --
        # which is what the library does with it (selectOTF)
        sirius.SimReconstructor(params, str(DATA / "otf.tif"), rigor=sirius.PlanRigor.Estimate)

    def test_a_declared_layout_and_the_steps_counts_have_to_agree(self):
        """The sentence the SIM operation refuses the pipeline with, in the
        library so that the window, a session and Python say it identically
        (the Python front's finding C). An agreeing layout says nothing."""
        self.assertEqual(sirius.sim_layout_counts_problem("z=[angle 3, z, phase 5]", 3, 5, 3, 5), "")
        why = sirius.sim_layout_counts_problem("z=[angle 3, z, phase 5]", 3, 5, 5, 3)
        self.assertIn("The dataset's layout z=[angle 3, z, phase 5] holds 3 angles \u00d7 5 phases", why)
        self.assertIn("the step uses 5 \u00d7 3", why)
        self.assertIn("They have to agree for the frames to be gathered.", why)
        # the weaker case: no layout, so the step's counts are used and the
        # run goes ahead with a warning
        self.assertEqual(sirius.sim_declared_counts_note(3, 5, 3, 5), "")
        note = sirius.sim_declared_counts_note(3, 5, 5, 3)
        self.assertIn("The dataset declares 3 angles \u00d7 5 phases", note)
        self.assertNotIn("have to agree", note)

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
