#include "py_common.hpp"

#include <nanobind/stl/string.h>

#include <optional>
#include <sstream>
#include <variant>

using namespace sirius;

namespace sirius_py {

    std::optional<PixelType> pixelTypeFromDtype(nb::handle dtype) {
        if (dtype.is_none()) return std::nullopt;
        nb::object np = nb::module_::import_("numpy");
        const std::string name = nb::cast<std::string>(np.attr("dtype")(dtype).attr("name"));
        if (name == "uint8") return PixelType::UInt8;
        if (name == "int8") return PixelType::Int8;
        if (name == "uint16") return PixelType::UInt16;
        if (name == "int16") return PixelType::Int16;
        if (name == "uint32") return PixelType::UInt32;
        if (name == "int32") return PixelType::Int32;
        if (name == "float32") return PixelType::Float32;
        if (name == "float64") return PixelType::Float64;
        throw std::invalid_argument("unsupported dtype '" + name +
                                    "' (supported: u/int 8/16/32, float32, float64)");
    }

    nb::object dtypeObject(const char* name) {
        return nb::module_::import_("numpy").attr("dtype")(name);
    }

    nb::object dtypeObject(PixelType t) { return dtypeObject(toString(t)); }

    namespace {
        template <typename T>
        std::vector<std::size_t> shapeVector(const Buffer<T>& b) {
            std::vector<std::size_t> s(static_cast<std::size_t>(b.rank()));
            for (int i = 0; i < b.rank(); ++i) s[static_cast<std::size_t>(i)] = static_cast<std::size_t>(b.dim(i));
            return s;
        }

        // Hand an owning host buffer to numpy without copying: a capsule
        // deletes the Buffer when the array is garbage collected.
        template <typename T>
        nb::object hostBufferToNumpy(Buffer<T>&& b) {
            auto* owner = new Buffer<T>(std::move(b));
            nb::capsule cap(owner, [](void* p) noexcept { delete static_cast<Buffer<T>*>(p); });
            const auto shape = shapeVector(*owner);
            return nb::cast(nb::ndarray<nb::numpy, T>(owner->data(), shape.size(), shape.data(), cap));
        }
    } // namespace

    const char* PyBuffer::dtypeName() const {
        return std::visit([](const auto& b) { return dtypeNameOf<typename std::decay_t<decltype(b)>::value_type>(); },
                          buffer_);
    }
    nb::object PyBuffer::dtype() const { return dtypeObject(dtypeName()); }
    Shape PyBuffer::shape() const {
        return std::visit([](const auto& b) { return b.shape(); }, buffer_);
    }
    Device PyBuffer::device() const {
        return std::visit([](const auto& b) { return b.device(); }, buffer_);
    }
    std::size_t PyBuffer::nbytes() const {
        return std::visit([](const auto& b) { return b.bytes(); }, buffer_);
    }
    std::size_t PyBuffer::size() const {
        return std::visit([](const auto& b) { return static_cast<std::size_t>(b.size()); }, buffer_);
    }
    bool PyBuffer::pinned() const {
        return std::visit([](const auto& b) { return b.pinned(); }, buffer_);
    }

    PyBuffer PyBuffer::to(Device device) const {
        return PyBuffer(std::visit([&](const auto& b) -> AnyPyBuffer {
            auto out = b.to(device);
            Stream::null().synchronize();
            return out;
        },
                                   buffer_));
    }

    nb::object PyBuffer::numpy() const {
        return std::visit([](const auto& b) {
            auto host = b.to(Device::cpu());
            Stream::null().synchronize();
            return hostBufferToNumpy(std::move(host));
        },
                          buffer_);
    }

    nb::object PyBuffer::ndarray(nb::handle self) const {
        return std::visit([&](const auto& b) {
            using T = typename std::decay_t<decltype(b)>::value_type;
            const auto shape = shapeVector(b);
            const int32_t devType = b.device().isCuda() ? nb::device::cuda::value : nb::device::cpu::value;
            return nb::cast(nb::ndarray<>(const_cast<T*>(b.data()), shape.size(), shape.data(), self, nullptr,
                                          nb::dtype<T>(), devType, b.device().index));
        },
                          buffer_);
    }

    nb::object toPython(AnyPyBuffer&& buffer) {
        const Device dev = std::visit([](const auto& b) { return b.device(); }, buffer);
        if (dev.isCpu())
            return std::visit([](auto&& b) { return hostBufferToNumpy(std::move(b)); }, std::move(buffer));
        return nb::cast(PyBuffer(std::move(buffer)));
    }

} // namespace sirius_py

using sirius_py::AnyPyBuffer;
using sirius_py::PyBuffer;

void bind_buffer(nb::module_& m) {
    nb::class_<PyBuffer>(m, "Buffer",
                         "Owning, contiguous, row-major array on a device. Host results are returned as "
                         "numpy arrays; this type wraps device memory. It implements the DLPack protocol, so "
                         "`torch.from_dlpack(buf)` / `cupy.from_dlpack(buf)` adopt the GPU memory without a copy. "
                         "Element types: u/int 8/16/32, float32, float64, complex64, complex128.")
        .def_prop_ro("shape", [](const PyBuffer& b) {
            const Shape s = b.shape();
            nb::list out;
            for (int i = 0; i < s.rank(); ++i) out.append(s[i]);
            return nb::tuple(out);
        })
        .def_prop_ro("ndim", [](const PyBuffer& b) { return b.shape().rank(); })
        .def_prop_ro("dtype", &PyBuffer::dtype)
        .def_prop_ro("device", &PyBuffer::device)
        .def_prop_ro("nbytes", &PyBuffer::nbytes)
        .def_prop_ro("size", &PyBuffer::size)
        .def_prop_ro("pinned", &PyBuffer::pinned)
        .def("to", &PyBuffer::to, nb::arg("device"), "Copy to another device (returns a new Buffer).")
        .def("numpy", &PyBuffer::numpy, "Copy to the host as a numpy array.")
        .def("__array__", [](nb::handle self, nb::args, nb::kwargs) {
            return nb::cast<PyBuffer&>(self).numpy();
        })
        // DLPack producer protocol. nanobind turns a framework-less nb::ndarray
        // into a "dltensor" capsule whose deleter holds a reference to `self`,
        // so the consumer (torch/cupy/numpy) keeps this buffer alive. Version
        // negotiation kwargs (max_version, dl_device, copy, stream) are accepted
        // and ignored: the unversioned capsule is what every consumer accepts.
        .def("__dlpack__", [](nb::handle self, nb::args, nb::kwargs) {
            return nb::cast<PyBuffer&>(self).ndarray(self);
        })
        .def("__dlpack_device__", [](const PyBuffer& b) {
            // DLDeviceType: kDLCPU = 1, kDLCUDA = 2
            return nb::make_tuple(b.device().isCuda() ? 2 : 1, b.device().index);
        })
        .def("__len__", [](const PyBuffer& b) { return b.shape().rank() ? static_cast<std::size_t>(b.shape()[0]) : 0u; })
        .def("__repr__", [](const PyBuffer& b) {
            std::ostringstream os;
            os << "Buffer(shape=" << b.shape().toString() << ", dtype=" << b.dtypeName()
               << ", device=" << toString(b.device()) << ")";
            return os.str();
        });

    // nb::ro: only read, so a read-only array (a memory map, a broadcast) is as good as any.
    m.def("to_device", [](nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> array, Device device) {
              // Host array -> owning buffer on `device` (a copy). Supports every element type.
              // A Shape of rank 0 is the empty shape (numel() == 0), not a scalar: a 0-d
              // array would come back as a 0-d array over no memory at all.
              if (array.ndim() == 0)
                  throw std::invalid_argument("to_device: a 0-d array has no shape a Buffer can hold; "
                                              "reshape it to (1,) first");
              std::vector<Index> shape(array.ndim());
              for (std::size_t i = 0; i < array.ndim(); ++i) shape[i] = static_cast<Index>(array.shape(i));
              const Shape s(shape.begin(), shape.end());
              auto upload = [&](auto tag) -> AnyPyBuffer {
                  using T = decltype(tag);
                  BufferView<const T> src(static_cast<const T*>(array.data()), s, Device::cpu());
                  auto out = toDevice(src, device);
                  Stream::null().synchronize();
                  return out;
              };
              const auto dt = array.dtype();
              using nb::dlpack::dtype_code;
              if (dt.code == static_cast<uint8_t>(dtype_code::UInt)) {
                  if (dt.bits == 8)  return sirius_py::toPython(upload(std::uint8_t{}));
                  if (dt.bits == 16) return sirius_py::toPython(upload(std::uint16_t{}));
                  if (dt.bits == 32) return sirius_py::toPython(upload(std::uint32_t{}));
              } else if (dt.code == static_cast<uint8_t>(dtype_code::Int)) {
                  if (dt.bits == 8)  return sirius_py::toPython(upload(std::int8_t{}));
                  if (dt.bits == 16) return sirius_py::toPython(upload(std::int16_t{}));
                  if (dt.bits == 32) return sirius_py::toPython(upload(std::int32_t{}));
              } else if (dt.code == static_cast<uint8_t>(dtype_code::Float)) {
                  if (dt.bits == 32) return sirius_py::toPython(upload(float{}));
                  if (dt.bits == 64) return sirius_py::toPython(upload(double{}));
              } else if (dt.code == static_cast<uint8_t>(dtype_code::Complex)) {
                  if (dt.bits == 64)  return sirius_py::toPython(upload(std::complex<float>{}));
                  if (dt.bits == 128) return sirius_py::toPython(upload(std::complex<double>{}));
              }
              throw std::invalid_argument("to_device: unsupported dtype (supported: u/int 8/16/32, float32, "
                                          "float64, complex64, complex128)"); }, nb::arg("array"), nb::arg("device"), "Copy a C-contiguous host array to `device`. Returns a numpy array for 'cpu' and a Buffer for CUDA.");
}
