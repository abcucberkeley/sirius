#include "py_common.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <sirius/fft.hpp>

#include <complex>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace sirius;

namespace {

    using Cplx = std::complex<double>;

    // Writable C-contiguous complex128 array on any device: numpy on the CPU,
    // anything exporting DLPack (torch, cupy, sirius.Buffer) on CUDA. The
    // rank is not pinned here -- the plan accepts whichever shape has the
    // element count it was planned for (flat, {rows, cols}, batched, ...).
    using CArray = nb::ndarray<Cplx, nb::c_contig>;

    Device deviceOf(const CArray& a) {
        if (a.device_type() == nb::device::cpu::value) return Device::cpu();
        if (a.device_type() == nb::device::cuda::value) return Device::cuda(a.device_id());
        throw std::invalid_argument("FFT: array lives on an unsupported device type " +
                                    std::to_string(a.device_type()));
    }

    Shape shapeOf(const CArray& a) {
        std::vector<Index> dims(a.ndim());
        for (std::size_t i = 0; i < a.ndim(); ++i) dims[i] = static_cast<Index>(a.shape(i));
        return Shape(dims.begin(), dims.end());
    }

    BufferView<Cplx> viewOf(CArray& a) { return {a.data(), shapeOf(a), deviceOf(a)}; }

    // Fresh numpy array with the same shape as `like`. Heap-owned by a
    // capsule that frees it on Python GC.
    nb::ndarray<nb::numpy, Cplx> emptyLike(const CArray& like) {
        const std::size_t ndim = like.ndim();
        std::vector<std::size_t> shape(ndim);
        std::size_t total = 1;
        for (std::size_t i = 0; i < ndim; ++i) {
            shape[i] = like.shape(i);
            total *= shape[i];
        }
        auto* data = std::make_unique<Cplx[]>(total).release();
        nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<Cplx*>(p); });
        return nb::ndarray<nb::numpy, Cplx>(data, ndim, shape.data(), owner);
    }

    // Python-facing wrapper around the planned transform. Size and device
    // validation is sirius::FFT's; this layer only adapts array types.
    class PyFFT {
    public:
        PyFFT(std::vector<int> dims, int howmany, PlanRigor rigor, Device device)
            : fft_(std::move(dims), howmany, rigor, device) {}

        Device device() const noexcept { return fft_.device(); }
        const std::vector<int>& dims() const noexcept { return fft_.dims(); }
        int howmany() const noexcept { return fft_.howmany(); }
        nb::tuple shape() const {
            const Shape s = fft_.shape();
            nb::list out;
            for (int i = 0; i < s.rank(); ++i) out.append(s[i]);
            return nb::tuple(out);
        }

        nb::object fft(CArray in) const { return run(in, true, false); }
        void fft_into(CArray in, CArray out) const { execute(viewOf(in), viewOf(out), true, false); }
        nb::object ifft(CArray in, bool normalize) const { return run(in, false, normalize); }
        void ifft_into(CArray in, CArray out, bool normalize) const {
            execute(viewOf(in), viewOf(out), false, normalize);
        }

    private:
        void execute(BufferView<const Cplx> in, BufferView<Cplx> out, bool forward, bool normalize) const {
            nb::gil_scoped_release release;
            if (forward) fft_.fft(in, out);
            else fft_.ifft(in, out, normalize);
            // Device work runs on the legacy default stream; make sure it is
            // complete before Python -- or a framework stream we know nothing
            // about -- reads the result.
            if (fft_.device().isCuda()) synchronizeDevice(fft_.device());
        }

        nb::object run(CArray& in, bool forward, bool normalize) const {
            BufferView<Cplx> vin = viewOf(in);
            if (fft_.device().isCpu()) {
                auto out = emptyLike(in);
                execute(vin, BufferView<Cplx>(out.data(), vin.shape(), Device::cpu()), forward, normalize);
                return nb::cast(std::move(out));
            }
            Buffer<Cplx> out(vin.shape(), fft_.device());
            execute(vin, out.view(), forward, normalize);
            return sirius_py::toPython(sirius_py::AnyPyBuffer(std::move(out)));
        }

        FFT fft_;
    };

} // anonymous namespace

void bind_fft(nb::module_& m) {
    nb::enum_<PlanRigor>(m, "PlanRigor",
                         "Planning rigor for FFTW. Trades one-time planning cost for "
                         "runtime speed (no effect on cuFFT plans).")
        .value("Estimate", PlanRigor::Estimate)
        .value("Measure", PlanRigor::Measure)
        .value("Patient", PlanRigor::Patient)
        .value("Exhaustive", PlanRigor::Exhaustive)
        .export_values();

    nb::class_<PyFFT>(m, "FFT",
                      "Planned complex128 FFT: 1D, 2D and 3D transforms, batched via `howmany`, on the "
                      "CPU (FFTW) or a CUDA device (cuFFT). Arrays must be C-contiguous complex128 on the "
                      "plan's device: numpy for 'cpu'; sirius.Buffer, torch or cupy (anything exporting "
                      "DLPack) for 'cuda'. Both `fft` and `ifft` come in allocating and in-place variants:\n\n"
                      "    f = FFT([8, 8])\n"
                      "    y = f.fft(x)              # new array, shape matches x\n"
                      "    f.fft(x, out=y)           # no allocation; out=x transforms in place\n"
                      "    x_back = f.ifft(y, normalize=True)\n\n"
                      "    g = FFT([8, 8], device='cuda')\n"
                      "    gy = g.fft(torch_tensor)  # sirius.Buffer on the GPU (torch.from_dlpack adopts it)\n")
        .def(nb::init<std::vector<int>, int, PlanRigor, Device>(),
             nb::arg("dims"),
             nb::arg("howmany") = 1,
             nb::arg("rigor") = PlanRigor::Measure,
             nb::arg("device") = Device::cpu(),
             "Plan an FFT for `dims` ([n], [rows, cols], or [depth, rows, cols]) "
             "and `howmany` batched transforms on `device` ('cpu', 'cuda', 'cuda:N' or a "
             "sirius.Device). On the CPU construction runs FFTW's planner, which can take a "
             "moment at higher rigor levels.")
        .def_prop_ro("device", &PyFFT::device)
        .def_prop_ro("dims", &PyFFT::dims)
        .def_prop_ro("howmany", &PyFFT::howmany)
        .def_prop_ro("shape", &PyFFT::shape, "Natural array shape: (howmany, *dims), howmany omitted when 1.")

        .def("fft", &PyFFT::fft,
             nb::arg("in"),
             "Forward transform. Returns a new array with the same shape as `in`: numpy on the "
             "CPU, a sirius.Buffer on CUDA.")
        .def("fft", &PyFFT::fft_into,
             nb::arg("in"), nb::arg("out"),
             "Forward transform into a preallocated output array. `out` must be complex128, "
             "C-contiguous, on the plan's device, and have the same total element count as `in`; "
             "`out` may be `in` itself.")

        .def("ifft", &PyFFT::ifft,
             nb::arg("in"), nb::arg("normalize") = false,
             "Inverse transform. When `normalize=True`, the result is divided by the "
             "product of `dims`, so `ifft(fft(x), normalize=True)` recovers `x`.")
        .def("ifft", &PyFFT::ifft_into,
             nb::arg("in"), nb::arg("out"), nb::arg("normalize") = false,
             "Inverse transform into a preallocated output array of matching size "
             "(`out` may be `in` itself).")

        .def_static("load_wisdom", &FFT::loadWisdom, nb::arg("path"),
                    "Import FFTW wisdom from a file. A missing file is silently ignored.")
        .def_static("save_wisdom", &FFT::saveWisdom, nb::arg("path"),
                    "Export accumulated FFTW wisdom to a file.");
}
