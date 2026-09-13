#include "py_common.hpp"

#include <nanobind/eigen/tensor.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <sirius/tiff_io.hpp>

#include <optional>
#include <sstream>

using namespace sirius;
using sirius_py::pixelTypeFromDtype;
using sirius_py::toPython;
using sirius_py::withPixelType;

namespace {
    TiffReadOptions makeOptions(Device device, bool allowCpuFallback, bool pinned) {
        TiffReadOptions o;
        o.device = device;
        o.allowCpuFallback = allowCpuFallback;
        o.hostMemory = pinned ? HostMemory::Pinned : HostMemory::Pageable;
        return o;
    }

    const Stream& streamOrNull(const Stream* s) { return s ? *s : Stream::null(); }

    // Legacy Eigen writers: one instantiation per pixel type and rank.
    template <typename T>
    void writeImage(const std::string& path, const Image<T>& image, TiffCompression comp) {
        writeTiff<T>(path, image, comp);
    }
    template <typename T>
    void writeStack(const std::string& path, const ImageStack<T>& stack, TiffCompression comp) {
        writeTiffStack<T>(path, stack, comp);
    }

    template <typename T>
    void defWriters(nb::module_& m) {
        m.def("write_tiff", &writeImage<T>, nb::arg("path"), nb::arg("image"), nb::arg("comp") = TiffCompression::None);
        m.def("write_tiff", &writeStack<T>, nb::arg("path"), nb::arg("image"), nb::arg("comp") = TiffCompression::None);
    }
} // anonymous namespace

void bind_tiff_io(nb::module_& m) {
    nb::enum_<TiffCompression>(m, "TiffCompression",
                               "Write tiff compression options.")
        .value("NoCompression", TiffCompression::None)
        .value("Lzw", TiffCompression::Lzw)
        .value("Deflate", TiffCompression::Deflate);

    nb::enum_<PixelType>(m, "PixelType")
        .value("UInt8", PixelType::UInt8)
        .value("Int8", PixelType::Int8)
        .value("UInt16", PixelType::UInt16)
        .value("Int16", PixelType::Int16)
        .value("UInt32", PixelType::UInt32)
        .value("Int32", PixelType::Int32)
        .value("Float32", PixelType::Float32)
        .value("Float64", PixelType::Float64)
        .def_prop_ro("dtype", [](PixelType t) { return sirius_py::dtypeObject(t); });

    nb::enum_<TiffLayout>(m, "TiffLayout")
        .value("Strips", TiffLayout::Strips)
        .value("Tiles", TiffLayout::Tiles);

    nb::class_<TiffImageInfo>(m, "TiffImageInfo", "Metadata of one image file directory (IFD).")
        .def_ro("ifd_offset", &TiffImageInfo::ifdOffset)
        .def_ro("width", &TiffImageInfo::width)
        .def_ro("height", &TiffImageInfo::height)
        .def_ro("pixel_type", &TiffImageInfo::pixelType)
        .def_prop_ro("dtype", [](const TiffImageInfo& i) { return sirius_py::dtypeObject(i.pixelType); })
        .def_ro("samples_per_pixel", &TiffImageInfo::samplesPerPixel)
        .def_ro("compression", &TiffImageInfo::compression)
        .def_ro("predictor", &TiffImageInfo::predictor)
        .def_ro("layout", &TiffImageInfo::layout)
        .def_ro("tile_width", &TiffImageInfo::tileWidth)
        .def_ro("tile_height", &TiffImageInfo::tileHeight)
        .def_ro("rows_per_strip", &TiffImageInfo::rowsPerStrip)
        .def_ro("reduced_resolution", &TiffImageInfo::reducedResolution)
        .def_ro("sub_ifds", &TiffImageInfo::subIfds)
        .def("__repr__", [](const TiffImageInfo& i) {
            std::ostringstream os;
            os << "TiffImageInfo(" << i.width << "x" << i.height << " " << toString(i.pixelType)
               << ", " << (i.layout == TiffLayout::Tiles ? "tiles" : "strips") << ", compression=" << i.compression
               << (i.reducedResolution ? ", reduced" : "") << ", ifd=" << i.ifdOffset << ")";
            return os.str();
        });

    nb::class_<TiffLevel>(m, "TiffLevel", "One pyramid level: same-resolution IFDs, one per page.")
        .def_ro("width", &TiffLevel::width)
        .def_ro("height", &TiffLevel::height)
        .def_ro("ifds", &TiffLevel::ifds)
        .def("__repr__", [](const TiffLevel& l) {
            return "TiffLevel(" + std::to_string(l.width) + "x" + std::to_string(l.height) + ", pages=" +
                   std::to_string(l.ifds.size()) + ")";
        });

    nb::class_<TiffInfo>(m, "TiffInfo")
        .def_ro("big_tiff", &TiffInfo::bigTiff)
        .def_ro("images", &TiffInfo::images)
        .def_ro("pages", &TiffInfo::pages)
        .def_ro("levels", &TiffInfo::levels)
        .def_prop_ro("page_count", &TiffInfo::pageCount)
        .def_prop_ro("level_count", &TiffInfo::levelCount)
        .def_prop_ro("width", &TiffInfo::width)
        .def_prop_ro("height", &TiffInfo::height)
        .def_prop_ro("pixel_type", &TiffInfo::pixelType)
        .def_prop_ro("dtype", [](const TiffInfo& i) { return sirius_py::dtypeObject(i.pixelType()); })
        .def_prop_ro("shape", [](const TiffInfo& i) { return nb::make_tuple(i.pageCount(), i.height(), i.width()); })
        .def_prop_ro("uniform_pages", &TiffInfo::uniformPages)
        .def("image", &TiffInfo::image, nb::arg("ifd_offset"))
        .def("page", &TiffInfo::page, nb::arg("index"));

    m.def("inspect_tiff", &inspectTiff, nb::arg("path"), "Read TIFF metadata (pages, pyramid levels, layout, codec).");

    nb::class_<TiffFile>(m, "TiffFile",
                         "Open a TIFF once and decode pages / pyramid levels / regions onto any device.\n\n"
                         "    f = TiffFile('stack.tif')\n"
                         "    f.info.shape                      # (pages, height, width)\n"
                         "    a = f.read_stack()                # numpy (z, y, x), native dtype\n"
                         "    g = f.read_stack(device='cuda')   # sirius.Buffer in GPU memory (nvTIFF decode)\n"
                         "    t = torch.from_dlpack(g)          # zero-copy adoption\n"
                         "    r = f.read_region(x, y, w, h, level=1, dtype='float32')\n")
        .def(nb::init<std::string>(), nb::arg("path"))
        .def_prop_ro("path", &TiffFile::path)
        .def_prop_ro("info", &TiffFile::info)
        .def("gpu_decodable", [](const TiffFile& f, Device device) {
                 std::string reason;
                 const bool ok = f.gpuDecodable(device, &reason);
                 return nb::make_tuple(ok, reason); }, nb::arg("device") = Device::cuda(0), "(ok, reason): whether nvTIFF can decode this file on `device` without the CPU fallback.")
        .def("read_stack", [](const TiffFile& f, nb::handle dtype, Device device, bool allowCpuFallback, bool pinned, const Stream* stream) {
                 const PixelType t = pixelTypeFromDtype(dtype).value_or(f.info().pixelType());
                 const auto opts = makeOptions(device, allowCpuFallback, pinned);
                 return toPython(withPixelType(t, [&](auto tag) -> AnyBuffer {
                     return f.readStack<decltype(tag)>(opts, streamOrNull(stream));
                 })); }, nb::arg("dtype") = nb::none(), nb::arg("device") = Device::cpu(), nb::arg("allow_cpu_fallback") = true, nb::arg("pinned") = false, nb::arg("stream") = nb::none(), "All full-resolution pages as (pages, height, width).")
        .def("read_pages", [](const TiffFile& f, std::size_t first, std::size_t count, nb::handle dtype, Device device, bool allowCpuFallback, bool pinned, const Stream* stream) {
                 const PixelType t = pixelTypeFromDtype(dtype).value_or(f.info().pixelType());
                 const auto opts = makeOptions(device, allowCpuFallback, pinned);
                 return toPython(withPixelType(t, [&](auto tag) -> AnyBuffer {
                     return f.readPages<decltype(tag)>(first, count, opts, streamOrNull(stream));
                 })); }, nb::arg("first"), nb::arg("count"), nb::arg("dtype") = nb::none(), nb::arg("device") = Device::cpu(), nb::arg("allow_cpu_fallback") = true, nb::arg("pinned") = false, nb::arg("stream") = nb::none(), "Pages [first, first + count).")
        .def("read_level", [](const TiffFile& f, std::size_t level, nb::handle dtype, Device device, bool allowCpuFallback, bool pinned, const Stream* stream) {
                 const PixelType t = pixelTypeFromDtype(dtype).value_or(f.info().pixelType());
                 const auto opts = makeOptions(device, allowCpuFallback, pinned);
                 return toPython(withPixelType(t, [&](auto tag) -> AnyBuffer {
                     return f.readLevel<decltype(tag)>(level, opts, streamOrNull(stream));
                 })); }, nb::arg("level"), nb::arg("dtype") = nb::none(), nb::arg("device") = Device::cpu(), nb::arg("allow_cpu_fallback") = true, nb::arg("pinned") = false, nb::arg("stream") = nb::none(), "Every page at pyramid level `level` (0 = full resolution).")
        .def("read_region", [](const TiffFile& f, std::uint32_t x, std::uint32_t y, std::uint32_t width, std::uint32_t height, std::size_t level, nb::handle dtype, Device device, bool allowCpuFallback, bool pinned, const Stream* stream) {
                 const PixelType t = pixelTypeFromDtype(dtype).value_or(f.info().pixelType());
                 const auto opts = makeOptions(device, allowCpuFallback, pinned);
                 const Region r{x, y, width, height};
                 return toPython(withPixelType(t, [&](auto tag) -> AnyBuffer {
                     return f.readRegion<decltype(tag)>(r, level, opts, streamOrNull(stream));
                 })); }, nb::arg("x"), nb::arg("y"), nb::arg("width") = 0, nb::arg("height") = 0, nb::arg("level") = 0, nb::arg("dtype") = nb::none(), nb::arg("device") = Device::cpu(), nb::arg("allow_cpu_fallback") = true, nb::arg("pinned") = false, nb::arg("stream") = nb::none(), "Rectangle (x, y, width, height) of every page at `level`; width/height 0 extend to the edge.");

    m.def("read_tiff", [](const std::string& path, nb::handle dtype, Device device, bool allowCpuFallback) {
              TiffFile f(path);
              const PixelType t = pixelTypeFromDtype(dtype).value_or(f.info().pixelType());
              const auto opts = makeOptions(device, allowCpuFallback, false);
              return toPython(withPixelType(t, [&](auto tag) -> AnyBuffer {
                  return f.readStack<decltype(tag)>(opts);
              })); }, nb::arg("path"), nb::arg("dtype") = nb::none(), nb::arg("device") = Device::cpu(), nb::arg("allow_cpu_fallback") = true, "Read a whole TIFF stack as (pages, height, width). Returns numpy on the CPU and a "
                                                                                                                                                                                                                                         "sirius.Buffer (DLPack) on CUDA devices.");

    defWriters<int8_t>(m);
    defWriters<uint8_t>(m);
    defWriters<int16_t>(m);
    defWriters<uint16_t>(m);
    defWriters<int32_t>(m);
    defWriters<uint32_t>(m);
    defWriters<float>(m);
    defWriters<double>(m);
}
