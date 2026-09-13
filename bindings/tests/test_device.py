"""Tests for device queries, Device/Stream objects and sirius.Buffer."""

from __future__ import annotations

import unittest

import numpy as np

import sirius


def _gpu_or_skip(test: unittest.TestCase) -> sirius.Device:
    if not sirius.cuda_available():
        test.skipTest("no CUDA device available")
    return sirius.Device.cuda(0)


class TestDevice(unittest.TestCase):
    def test_constructors_and_strings(self):
        self.assertTrue(sirius.Device.cpu().is_cpu)
        self.assertTrue(sirius.Device.cuda(2).is_cuda)
        self.assertEqual(sirius.Device.cuda(2).index, 2)
        self.assertEqual(str(sirius.Device.cpu()), "cpu")
        self.assertEqual(str(sirius.Device.cuda(1)), "cuda:1")
        self.assertEqual(sirius.Device("cuda:3"), sirius.Device.cuda(3))
        self.assertEqual(sirius.Device("cuda"), sirius.Device.cuda(0))
        self.assertEqual(sirius.Device("cpu"), sirius.Device.cpu())
        self.assertNotEqual(sirius.Device.cpu(), sirius.Device.cuda(0))
        self.assertEqual(len({sirius.Device.cpu(), sirius.Device("cpu")}), 1)
        with self.assertRaises(ValueError):
            sirius.Device("tpu")

    def test_queries_are_consistent(self):
        self.assertIsInstance(sirius.built_with_cuda(), bool)
        self.assertIsInstance(sirius.built_with_nvtiff(), bool)
        self.assertGreaterEqual(sirius.cuda_device_count(), 0)
        self.assertEqual(sirius.cuda_available(), sirius.cuda_device_count() > 0)
        if not sirius.built_with_cuda():
            self.assertFalse(sirius.cuda_available())
            self.assertFalse(sirius.built_with_nvtiff())

    def test_cpu_stream_is_noop(self):
        s = sirius.Stream()
        self.assertTrue(s.device.is_cpu)
        s.synchronize()

    def test_cuda_unavailable_raises(self):
        if sirius.cuda_available():
            self.skipTest("CUDA available")
        with self.assertRaises(RuntimeError):
            sirius.Stream(sirius.Device.cuda(0))
        with self.assertRaises(RuntimeError):
            sirius.to_device(np.zeros((2, 2), dtype=np.float32), "cuda")

    def test_device_properties(self):
        gpu = _gpu_or_skip(self)
        p = sirius.device_properties(gpu)
        self.assertTrue(p.name)
        self.assertGreater(p.total_memory_bytes, 0)
        self.assertGreaterEqual(p.compute_major, 5)
        self.assertIn("DeviceProperties", repr(p))
        with self.assertRaises(RuntimeError):
            sirius.device_properties(sirius.Device.cpu())


class TestBuffer(unittest.TestCase):
    def test_complex_dtypes(self):
        a = (np.arange(6, dtype=np.float64) + 1j).reshape(2, 3)
        b = sirius.to_device(a, "cpu")
        self.assertEqual(b.dtype, np.complex128)
        np.testing.assert_array_equal(a, b)
        c = sirius.to_device(a.astype(np.complex64), "cpu")
        self.assertEqual(c.dtype, np.complex64)
        if sirius.cuda_available():
            d = sirius.to_device(a, "cuda")
            self.assertEqual(d.dtype, np.dtype(np.complex128))
            self.assertIn("complex128", repr(d))
            np.testing.assert_array_equal(d.numpy(), a)
            np.testing.assert_array_equal(np.from_dlpack(d.to("cpu")) if hasattr(np, "from_dlpack") else d.numpy(), a)

    def test_to_device_rejects_a_0d_array(self):
        # a rank-0 Shape is the empty shape: np.array(3.0) came back as a 0-d
        # array over no memory (reading 5e-324 or anything else)
        with self.assertRaisesRegex(ValueError, "0-d"):
            sirius.to_device(np.array(3.0), "cpu")
        np.testing.assert_array_equal(sirius.to_device(np.array(3.0).reshape(1), "cpu"), [3.0])

    def test_to_device_takes_read_only_arrays(self):
        a = np.broadcast_to(np.float32(2.0), (3, 4))   # read-only, and not C-contiguous
        b = sirius.to_device(a, "cpu")
        self.assertEqual(b.dtype, np.float32)
        np.testing.assert_array_equal(b, a)
        ro = np.arange(6, dtype=np.int16).reshape(2, 3)
        ro.flags.writeable = False
        np.testing.assert_array_equal(sirius.to_device(ro, "cpu"), ro)

    def test_to_device_cpu_returns_numpy_copy(self):
        a = np.arange(12, dtype=np.uint16).reshape(3, 4)
        b = sirius.to_device(a, "cpu")
        self.assertIsInstance(b, np.ndarray)
        np.testing.assert_array_equal(a, b)
        b[0, 0] = 99
        self.assertEqual(a[0, 0], 0)  # a copy, not a view

    def test_round_trip_all_dtypes(self):
        gpu = _gpu_or_skip(self)
        for dtype in (np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.float32, np.float64):
            with self.subTest(dtype=dtype.__name__):
                a = np.arange(60, dtype=dtype).reshape(3, 4, 5)
                buf = sirius.to_device(a, gpu)
                self.assertIsInstance(buf, sirius.Buffer)
                self.assertEqual(buf.shape, (3, 4, 5))
                self.assertEqual(buf.ndim, 3)
                self.assertEqual(buf.dtype, np.dtype(dtype))
                self.assertEqual(buf.device, gpu)
                self.assertEqual(buf.nbytes, a.nbytes)
                self.assertEqual(buf.size, 60)
                self.assertEqual(len(buf), 3)
                self.assertIn("cuda:0", repr(buf))
                np.testing.assert_array_equal(buf.numpy(), a)
                np.testing.assert_array_equal(np.asarray(buf), a)

    def test_to_moves_between_devices(self):
        gpu = _gpu_or_skip(self)
        a = np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32)
        dev = sirius.to_device(a, gpu)
        dev2 = dev.to(gpu)
        self.assertEqual(dev2.device, gpu)
        np.testing.assert_array_equal(dev2.numpy(), a)
        host = dev.to("cpu")
        self.assertTrue(host.device.is_cpu)
        np.testing.assert_array_equal(host.numpy(), a)

    def test_dlpack_device(self):
        gpu = _gpu_or_skip(self)
        buf = sirius.to_device(np.ones((4,), dtype=np.float32), gpu)
        kind, index = buf.__dlpack_device__()
        self.assertEqual(int(kind), 2)  # DLDeviceType.kDLCUDA
        self.assertEqual(index, 0)

    def test_torch_adopts_gpu_memory_via_dlpack(self):
        gpu = _gpu_or_skip(self)
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        if not torch.cuda.is_available():
            self.skipTest("torch has no CUDA")
        a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        buf = sirius.to_device(a, gpu)
        t = torch.from_dlpack(buf)
        self.assertEqual(t.device.type, "cuda")
        self.assertEqual(tuple(t.shape), (2, 3, 4))
        self.assertEqual(t.dtype, torch.float32)
        np.testing.assert_array_equal(t.cpu().numpy(), a)
        # zero-copy: writes through torch are visible in the buffer
        t[0, 0, 0] = -5.0
        torch.cuda.synchronize()
        self.assertEqual(buf.numpy()[0, 0, 0], -5.0)


if __name__ == "__main__":
    unittest.main()
