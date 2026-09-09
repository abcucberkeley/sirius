"""Tests for the planned-FFT bindings (sirius.FFT, sirius.PlanRigor)."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
from _helpers import silenced_stderr

import sirius


def _random_complex(shape, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex128
    )


class TestPlanRigorEnum(unittest.TestCase):
    def test_all_members_exposed(self):
        for name in ("Estimate", "Measure", "Patient", "Exhaustive"):
            self.assertTrue(hasattr(sirius.PlanRigor, name), name)

    def test_distinct_values(self):
        values = {
            sirius.PlanRigor.Estimate,
            sirius.PlanRigor.Measure,
            sirius.PlanRigor.Patient,
            sirius.PlanRigor.Exhaustive,
        }
        self.assertEqual(len(values), 4)


class TestFFTConstruction(unittest.TestCase):
    def test_default_args(self):
        # howmany=1, rigor=Measure
        f = sirius.FFT([8])
        x = _random_complex(8)
        y = f.fft(x)
        self.assertEqual(y.shape, (8,))
        self.assertEqual(y.dtype, np.complex128)

    def test_one_dim(self):
        sirius.FFT([16], 1, sirius.PlanRigor.Estimate)

    def test_two_dim(self):
        sirius.FFT([8, 8])

    def test_three_dim(self):
        sirius.FFT([4, 4, 4])

    def test_each_rigor(self):
        # Estimate is the cheapest level; using it everywhere here keeps planning fast.
        for rigor in (
            sirius.PlanRigor.Estimate,
            sirius.PlanRigor.Measure,
            sirius.PlanRigor.Patient,
            sirius.PlanRigor.Exhaustive,
        ):
            with self.subTest(rigor=rigor):
                f = sirius.FFT([8], 1, rigor)
                y = f.fft(np.zeros(8, dtype=np.complex128))
                self.assertEqual(y.shape, (8,))


class TestFFTRoundTrip(unittest.TestCase):
    """ifft(fft(x), normalize=True) should return x within float tolerance."""

    def _check_round_trip(self, dims):
        f = sirius.FFT(dims, 1, sirius.PlanRigor.Estimate)
        x = _random_complex(dims, seed=hash(tuple(dims)) & 0xFFFF)
        y = f.fft(x)
        x_back = f.ifft(y, normalize=True)
        np.testing.assert_allclose(x_back, x, atol=1e-10)

    def test_one_dim(self):
        for n in (1, 2, 7, 16, 257):
            with self.subTest(n=n):
                self._check_round_trip([n])

    def test_two_dim(self):
        for shape in [(4, 4), (8, 16), (15, 17)]:
            with self.subTest(shape=shape):
                self._check_round_trip(list(shape))

    def test_three_dim(self):
        for shape in [(2, 4, 4), (3, 5, 7)]:
            with self.subTest(shape=shape):
                self._check_round_trip(list(shape))


class TestFFTAgainstNumpy(unittest.TestCase):
    """Sanity-check forward output against numpy's reference implementation."""

    def test_forward_1d_matches_numpy(self):
        n = 32
        f = sirius.FFT([n], 1, sirius.PlanRigor.Estimate)
        x = _random_complex(n, seed=1)
        np.testing.assert_allclose(f.fft(x), np.fft.fft(x), atol=1e-9)

    def test_forward_2d_matches_numpy(self):
        shape = (8, 16)
        f = sirius.FFT(list(shape), 1, sirius.PlanRigor.Estimate)
        x = _random_complex(shape, seed=2)
        np.testing.assert_allclose(f.fft(x), np.fft.fft2(x), atol=1e-9)

    def test_forward_3d_matches_numpy(self):
        shape = (4, 4, 8)
        f = sirius.FFT(list(shape), 1, sirius.PlanRigor.Estimate)
        x = _random_complex(shape, seed=3)
        np.testing.assert_allclose(f.fft(x), np.fft.fftn(x), atol=1e-9)


class TestFFTNormalization(unittest.TestCase):
    """Without normalize, ifft(fft(x)) returns N*x (FFTW convention)."""

    def test_unnormalized_inverse_scales_by_n(self):
        n = 8
        f = sirius.FFT([n], 1, sirius.PlanRigor.Estimate)
        x = _random_complex(n, seed=5)
        x_back = f.ifft(f.fft(x), normalize=False)
        np.testing.assert_allclose(x_back, n * x, atol=1e-10)

    def test_normalize_default_is_false(self):
        n = 4
        f = sirius.FFT([n], 1, sirius.PlanRigor.Estimate)
        x = _random_complex(n, seed=6)
        # Default of normalize is False -> N*x, not x.
        x_back_default = f.ifft(f.fft(x))
        np.testing.assert_allclose(x_back_default, n * x, atol=1e-10)


class TestFFTBatched(unittest.TestCase):
    """`howmany>1` runs the planned transform over contiguous chunks."""

    def test_batched_unit_impulses(self):
        # Three independent 4-point transforms of unit deltas, packed flat.
        f = sirius.FFT([4], howmany=3, rigor=sirius.PlanRigor.Estimate)
        x = np.zeros(12, dtype=np.complex128)
        x[0] = 1
        x[4] = 1
        x[8] = 1
        y = f.fft(x)
        self.assertEqual(y.shape, (12,))
        # FFT of a delta is the all-ones vector; expect [1]*12.
        np.testing.assert_allclose(y, np.ones(12, dtype=np.complex128), atol=1e-12)

    def test_batched_round_trip(self):
        dims = [8]
        howmany = 4
        f = sirius.FFT(dims, howmany, sirius.PlanRigor.Estimate)
        x = _random_complex(dims[0] * howmany, seed=7)
        x_back = f.ifft(f.fft(x), normalize=True)
        np.testing.assert_allclose(x_back, x, atol=1e-10)

    def test_batched_2d_round_trip(self):
        # Plan for 4x4 transforms, batched 3 wide.
        f = sirius.FFT([4, 4], 3, sirius.PlanRigor.Estimate)
        x = _random_complex(3 * 4 * 4, seed=8)  # flat, total = dims_product * howmany
        x_back = f.ifft(f.fft(x), normalize=True)
        np.testing.assert_allclose(x_back, x, atol=1e-10)


class TestFFTIntoBuffer(unittest.TestCase):
    """The two-arg fft/ifft variants write into a preallocated output."""

    def test_fft_into_writes_buffer(self):
        n = 8
        f = sirius.FFT([n], 1, sirius.PlanRigor.Estimate)
        x = _random_complex(n, seed=9)
        out = np.empty_like(x)
        out.fill(np.nan + 1j * np.nan)
        f.fft(x, out)
        np.testing.assert_allclose(out, np.fft.fft(x), atol=1e-9)

    def test_ifft_into_normalizes(self):
        n = 8
        f = sirius.FFT([n], 1, sirius.PlanRigor.Estimate)
        x = _random_complex(n, seed=10)
        y = f.fft(x)
        out = np.empty_like(x)
        f.ifft(y, out, normalize=True)
        np.testing.assert_allclose(out, x, atol=1e-10)

    def test_fft_into_accepts_reshape(self):
        # PyFFT only checks total element count, not exact shape: a (4,4)
        # plan should accept a flat 16-element output buffer too.
        f = sirius.FFT([4, 4], 1, sirius.PlanRigor.Estimate)
        x = _random_complex((4, 4), seed=11)
        out_flat = np.empty(16, dtype=np.complex128)
        f.fft(x, out_flat)
        np.testing.assert_allclose(out_flat.reshape(4, 4), np.fft.fft2(x), atol=1e-9)


class TestFFTInPlace(unittest.TestCase):
    """out=in transforms in place (a dedicated in-place FFTW plan on the CPU)."""

    def test_in_place_matches_out_of_place(self):
        for dims, howmany in (([64], 1), ([15, 17], 1), ([4, 6, 10], 1), ([9, 12], 3)):
            with self.subTest(dims=dims, howmany=howmany):
                f = sirius.FFT(dims, howmany, sirius.PlanRigor.Measure)
                x = _random_complex(f.shape, seed=len(dims))
                reference = f.fft(x)
                work = x.copy()
                f.fft(work, work)
                np.testing.assert_allclose(work, reference, atol=1e-9)
                f.ifft(work, work, normalize=True)
                np.testing.assert_allclose(work, x, atol=1e-10)


class TestFFTOnCuda(unittest.TestCase):
    """The same class plans on a CUDA device; arrays arrive through DLPack."""

    def setUp(self):
        if not sirius.cuda_available():
            self.skipTest("no CUDA device available")
        self.gpu = sirius.Device.cuda(0)

    def test_device_argument(self):
        self.assertTrue(sirius.FFT([8]).device.is_cpu)
        f = sirius.FFT([8, 8], device="cuda")
        self.assertEqual(f.device, self.gpu)
        self.assertEqual(f.dims, [8, 8])
        self.assertEqual(f.shape, (8, 8))

    def test_buffer_round_trip_matches_numpy(self):
        x = _random_complex((8, 16), seed=21)
        f = sirius.FFT([8, 16], device=self.gpu)
        xin = sirius.to_device(x, self.gpu)
        self.assertIsInstance(xin, sirius.Buffer)
        self.assertEqual(xin.dtype, np.dtype(np.complex128))
        y = f.fft(xin)
        self.assertIsInstance(y, sirius.Buffer)
        self.assertEqual(y.device, self.gpu)
        self.assertEqual(y.shape, (8, 16))
        np.testing.assert_allclose(y.numpy(), np.fft.fft2(x), atol=1e-9)
        back = f.ifft(y, normalize=True)
        np.testing.assert_allclose(back.numpy(), x, atol=1e-10)

    def test_into_and_in_place(self):
        x = _random_complex(32, seed=22)
        f = sirius.FFT([32], device=self.gpu)
        buf = sirius.to_device(x, self.gpu)
        out = sirius.to_device(np.zeros(32, dtype=np.complex128), self.gpu)
        f.fft(buf, out)
        np.testing.assert_allclose(out.numpy(), np.fft.fft(x), atol=1e-9)
        f.fft(buf, buf)
        np.testing.assert_allclose(buf.numpy(), np.fft.fft(x), atol=1e-9)

    def test_host_array_rejected_by_cuda_plan(self):
        f = sirius.FFT([8], device=self.gpu)
        with self.assertRaises(ValueError):
            f.fft(np.zeros(8, dtype=np.complex128))
        g = sirius.FFT([8])
        with self.assertRaises(ValueError):
            g.fft(sirius.to_device(np.zeros(8, dtype=np.complex128), self.gpu))

    def test_torch_tensors_via_dlpack(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        if not torch.cuda.is_available():
            self.skipTest("torch has no CUDA")
        x = _random_complex((4, 8), seed=23)
        t = torch.from_numpy(x).cuda()
        f = sirius.FFT([4, 8], device="cuda")
        y = f.fft(t)
        self.assertIsInstance(y, sirius.Buffer)
        np.testing.assert_allclose(y.numpy(), np.fft.fft2(x), atol=1e-9)
        out = torch.empty_like(t)
        f.fft(t, out)
        torch.cuda.synchronize()
        np.testing.assert_allclose(out.cpu().numpy(), np.fft.fft2(x), atol=1e-9)
        np.testing.assert_allclose(torch.from_dlpack(y).cpu().numpy(), np.fft.fft2(x), atol=1e-9)


class TestFFTValidation(unittest.TestCase):
    """The Python wrapper rejects buffers whose total size doesn't match the plan."""

    def test_wrong_size_input_raises(self):
        f = sirius.FFT([8], 1, sirius.PlanRigor.Estimate)
        with self.assertRaises((ValueError, RuntimeError, TypeError)):
            f.fft(np.zeros(7, dtype=np.complex128))

    def test_wrong_size_output_raises(self):
        f = sirius.FFT([8], 1, sirius.PlanRigor.Estimate)
        x = np.zeros(8, dtype=np.complex128)
        out_too_small = np.empty(7, dtype=np.complex128)
        with self.assertRaises((ValueError, RuntimeError, TypeError)):
            f.fft(x, out_too_small)

    def test_batched_size_validated(self):
        # howmany=2, dims=[4]: total must be 8. A single 4-vector should fail.
        f = sirius.FFT([4], 2, sirius.PlanRigor.Estimate)
        with self.assertRaises((ValueError, RuntimeError, TypeError)):
            f.fft(np.zeros(4, dtype=np.complex128))


class TestFFTWisdom(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "wisdom.bin")
            # Plan something so FFTW has wisdom worth saving.
            sirius.FFT([8], 1, sirius.PlanRigor.Measure)
            sirius.FFT.save_wisdom(path)
            self.assertTrue(os.path.exists(path))
            sirius.FFT.load_wisdom(path)  # should not raise

    def test_load_missing_file_is_silent(self):
        with tempfile.TemporaryDirectory() as d:
            with silenced_stderr():
                sirius.FFT.load_wisdom(os.path.join(d, "does_not_exist"))


if __name__ == "__main__":
    unittest.main()


class TestFFTOutputArgument(unittest.TestCase):
    """`out` is written in place, so it must be the caller's array: an array
    nanobind would have to convert (Fortran order, another dtype) is refused
    rather than transformed into a temporary that is thrown away."""

    def test_out_must_be_c_contiguous_complex128(self):
        f = sirius.FFT([4, 4], 1, sirius.PlanRigor.Estimate)
        x = _random_complex([4, 4], seed=7)
        fortran = np.empty((4, 4), dtype=np.complex128, order="F")
        with self.assertRaises(TypeError):
            f.fft(x, fortran)
        single = np.empty((4, 4), dtype=np.complex64)
        with self.assertRaises(TypeError):
            f.ifft(x, single)
        ok = np.empty((4, 4), dtype=np.complex128)
        f.fft(x, ok)
        np.testing.assert_allclose(ok, np.fft.fft2(x), atol=1e-10)
