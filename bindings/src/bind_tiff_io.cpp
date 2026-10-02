#include "py_common.hpp"

#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <sirius/tiff_io.hpp>

#include <optional>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

using namespace sirius;
using sirius_py::pixelTypeFromDtype;
using sirius_py::toPython;
using sirius_py::withPixelType;

// Appended to every read's docstring by string literal concatenation
// (nanobind keeps the docstring pointer, so it must be a literal).
#define SIRIUS_SAMPLES_DOC                                                                              \
    " first_sample / samples pick the samples (channels) of a multi-sample image: one sample reads as " \
    "(pages, height, width), several as (pages, samples, height, width); samples=0 reads every sample " \
    "from first_sample on."

namespace {
    TiffReadOptions makeOptions(Device device, bool allowCpuFallback, bool pinned, std::uint16_t firstSample = 0,
                                std::uint16_t samples = 0) {
        TiffReadOptions o;
        o.device = device;
        o.allowCpuFallback = allowCpuFallback;
        o.hostMemory = pinned ? HostMemory::Pinned : HostMemory::Pageable;
        o.firstSample = firstSample;
        o.sampleCount = samples;
        return o;
    }

    // (pages, h, w) or (pages, samples, h, w), as a read returns it.
    nb::tuple shapeTuple(const TiffInfo& i) {
        if (i.pageCount() && i.samplesPerPixel() > 1)
            return nb::make_tuple(i.pageCount(), i.samplesPerPixel(), i.height(), i.width());
        return nb::make_tuple(i.pageCount(), i.height(), i.width());
    }


    const Stream& streamOrNull(const Stream* s) { return s ? *s : Stream::null(); }

    // write_tiff's image: any host array. nanobind copies a non-contiguous one
    // (a transpose, a stepped slice) to C order; the dtype is checked below.
    // It used to be one Eigen::Tensor overload per dtype and rank, int8
    // first, and the caster converts: an array no overload matched exactly
    // -- int64, numpy's default integer, or any non-contiguous view -- was
    // cast to int8 and written with wraparound, without a word.
    using WriteArray = nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>;

    // numpy's name of a DLPack dtype, for the error message.
    std::string dtypeName(nb::dlpack::dtype dt) {
        using nb::dlpack::dtype_code;
        switch (static_cast<dtype_code>(dt.code)) {
            case dtype_code::Int: return "int" + std::to_string(dt.bits);
            case dtype_code::UInt: return "uint" + std::to_string(dt.bits);
            case dtype_code::Float: return "float" + std::to_string(dt.bits);
            case dtype_code::Complex: return "complex" + std::to_string(dt.bits);
            case dtype_code::Bool: return "bool";
            default: return "an unsupported dtype";
        }
    }

    std::optional<PixelType> writablePixelType(nb::dlpack::dtype dt) {
        if (dt == nb::dtype<std::uint8_t>()) return PixelType::UInt8;
        if (dt == nb::dtype<std::int8_t>()) return PixelType::Int8;
        if (dt == nb::dtype<std::uint16_t>()) return PixelType::UInt16;
        if (dt == nb::dtype<std::int16_t>()) return PixelType::Int16;
        if (dt == nb::dtype<std::uint32_t>()) return PixelType::UInt32;
        if (dt == nb::dtype<std::int32_t>()) return PixelType::Int32;
        if (dt == nb::dtype<float>()) return PixelType::Float32;
        if (dt == nb::dtype<double>()) return PixelType::Float64;
        return std::nullopt;
    }

    void writeArray(const std::string& path, const WriteArray& image, TiffCompression comp) {
        if (image.ndim() != 2 && image.ndim() != 3)
            throw nb::type_error(("write_tiff: image must be 2-D (rows, cols) or 3-D (pages, rows, cols), got " +
                                  std::to_string(image.ndim()) + "-D")
                                     .c_str());
        const std::optional<PixelType> t = writablePixelType(image.dtype());
        if (!t)
            throw nb::type_error(("write_tiff: cannot write " + dtypeName(image.dtype()) +
                                  " (supported: u/int 8/16/32, float32, float64); convert first, "
                                  "e.g. image.astype(np.uint16)")
                                     .c_str());
        std::vector<Index> dims(image.ndim());
        for (std::size_t i = 0; i < image.ndim(); ++i) dims[i] = static_cast<Index>(image.shape(i));
        const Shape shape(dims.begin(), dims.end());
        withPixelType(*t, [&](auto tag) {
            using T = decltype(tag);
            const BufferView<const T> view(static_cast<const T*>(image.data()), shape, Device::cpu());
            nb::gil_scoped_release release;
            if (shape.rank() == 2) writeTiff<T>(path, view, comp);
            else writeTiffStack<T>(path, view, comp);
        });
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
        .def_ro("description", &TiffImageInfo::description, "ImageDescription tag (OME-XML, ImageJ metadata); \"\" when absent.")
        .def_ro("x_resolution", &TiffImageInfo::xResolution, "XResolution tag (pixels per resolution_unit); 0 when absent.")
        .def_ro("y_resolution", &TiffImageInfo::yResolution, "YResolution tag (pixels per resolution_unit); 0 when absent.")
        .def_ro("resolution_unit", &TiffImageInfo::resolutionUnit, "ResolutionUnit tag: 1 none, 2 inch, 3 centimetre.")
        .def_ro("width", &TiffImageInfo::width)
        .def_ro("height", &TiffImageInfo::height)
        .def_ro("pixel_type", &TiffImageInfo::pixelType)
        .def_prop_ro("dtype", [](const TiffImageInfo& i) { return sirius_py::dtypeObject(i.pixelType); })
        .def_ro("samples_per_pixel", &TiffImageInfo::samplesPerPixel)
        .def_ro("bits_per_sample", &TiffImageInfo::bitsPerSample)
        .def_ro("sample_format", &TiffImageInfo::sampleFormat, "1 unsigned, 2 signed, 3 IEEE float.")
        .def_ro("photometric", &TiffImageInfo::photometric,
                "0 min-is-white, 1 min-is-black, 2 RGB, 3 palette, 5 CMYK, 6 YCbCr ...")
        .def_ro("planar_config", &TiffImageInfo::planarConfig, "1 contiguous (RGBRGB...), 2 separate planes.")
        .def_ro("orientation", &TiffImageInfo::orientation)
        .def_ro("extra_samples", &TiffImageInfo::extraSamples)
        .def_prop_ro("colormap", [](const TiffImageInfo& i) -> nb::object {
                if (i.colormap.empty()) return nb::none();
                const std::size_t n = i.colormap.size() / 3;
                auto* data = new std::uint16_t[i.colormap.size()];
                std::copy(i.colormap.begin(), i.colormap.end(), data);
                nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<std::uint16_t*>(p); });
                return nb::cast(nb::ndarray<nb::numpy, std::uint16_t, nb::ndim<2>>(data, {3, n}, owner)); }, "Palette images: the (3, 2**bits_per_sample) uint16 colormap (R, G, B rows); None otherwise.")
        .def_ro("unsupported", &TiffImageInfo::unsupported, "Why this IFD cannot be decoded; \"\" when it can.")
        .def_prop_ro("decodable", &TiffImageInfo::decodable)
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
               << (i.samplesPerPixel > 1 ? " x" + std::to_string(i.samplesPerPixel) + " samples" : std::string())
               << ", " << (i.layout == TiffLayout::Tiles ? "tiles" : "strips") << ", compression=" << i.compression
               << (i.reducedResolution ? ", reduced" : "") << ", ifd=" << i.ifdOffset << ")";
            return os.str(); });

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
        .def_ro("big_endian", &TiffInfo::bigEndian)
        .def_ro("images", &TiffInfo::images)
        .def_ro("pages", &TiffInfo::pages)
        .def_ro("levels", &TiffInfo::levels)
        .def_prop_ro("page_count", &TiffInfo::pageCount)
        .def_prop_ro("level_count", &TiffInfo::levelCount)
        .def_prop_ro("width", &TiffInfo::width)
        .def_prop_ro("height", &TiffInfo::height)
        .def_prop_ro("pixel_type", &TiffInfo::pixelType)
        .def_prop_ro("dtype", [](const TiffInfo& i) { return sirius_py::dtypeObject(i.pixelType()); })
        .def_prop_ro("samples_per_pixel", &TiffInfo::samplesPerPixel)
        .def_prop_ro("shape", &shapeTuple, "(pages, height, width), or (pages, samples, height, width) with several samples per pixel.")
        .def_prop_ro("uniform_pages", &TiffInfo::uniformPages)
        .def("image", &TiffInfo::image, nb::arg("ifd_offset"))
        .def("page", &TiffInfo::page, nb::arg("index"));

    m.def("inspect_tiff", &inspectTiff, nb::arg("path"), "Read TIFF metadata (pages, pyramid levels, layout, codec).");

    // --- OME-XML / ImageJ metadata -------------------------------------------------------
    nb::class_<OmeChannel>(m, "OmeChannel", "One OME <Channel>.")
        .def_ro("id", &OmeChannel::id)
        .def_ro("name", &OmeChannel::name)
        .def_ro("samples_per_pixel", &OmeChannel::samplesPerPixel)
        .def_ro("emission_nm", &OmeChannel::emissionNm, "0 when not given.")
        .def_ro("excitation_nm", &OmeChannel::excitationNm, "0 when not given.")
        .def_ro("has_color", &OmeChannel::hasColor)
        .def_ro("color_rgba", &OmeChannel::colorRgba, "OME Color as unsigned RGBA (r << 24 | g << 16 | b << 8 | a).")
        .def_ro("fluor", &OmeChannel::fluor)
        .def("__repr__", [](const OmeChannel& c) { return "OmeChannel(" + c.name + ")"; });

    nb::class_<OmeTiffData>(m, "OmeTiffData", "One OME <TiffData> block: which IFDs hold which planes.")
        .def_ro("ifd", &OmeTiffData::ifd)
        .def_ro("first_c", &OmeTiffData::firstC)
        .def_ro("first_z", &OmeTiffData::firstZ)
        .def_ro("first_t", &OmeTiffData::firstT)
        .def_ro("plane_count", &OmeTiffData::planeCount)
        .def_ro("has_ifd", &OmeTiffData::hasIfd)
        .def_ro("has_plane_count", &OmeTiffData::hasPlaneCount)
        .def_ro("uuid", &OmeTiffData::uuid)
        .def_ro("file_name", &OmeTiffData::fileName);

    nb::class_<OmeImage>(m, "OmeImage", "One OME <Image> (a series).")
        .def_ro("id", &OmeImage::id)
        .def_ro("name", &OmeImage::name)
        .def_ro("dimension_order", &OmeImage::dimensionOrder)
        .def_ro("type", &OmeImage::type)
        .def_ro("size_x", &OmeImage::sizeX)
        .def_ro("size_y", &OmeImage::sizeY)
        .def_ro("size_z", &OmeImage::sizeZ)
        .def_ro("size_c", &OmeImage::sizeC)
        .def_ro("size_t", &OmeImage::sizeT)
        .def_ro("physical_size_um", &OmeImage::physicalSizeUm, "(x, y, z) in micrometres; 0 when not given.")
        .def_ro("time_increment_s", &OmeImage::timeIncrementS)
        .def_ro("interleaved", &OmeImage::interleaved)
        .def_ro("channels", &OmeImage::channels)
        .def_ro("tiff_data", &OmeImage::tiffData)
        .def("__repr__", [](const OmeImage& i) {
            std::ostringstream os;
            os << "OmeImage(" << i.name << ", " << i.dimensionOrder << " c" << i.sizeC << " z" << i.sizeZ << " t"
               << i.sizeT << " y" << i.sizeY << " x" << i.sizeX << ")";
            return os.str();
        });

    nb::class_<ImageJMetadata>(m, "ImageJMetadata", "ImageJ's ImageDescription header.")
        .def_ro("version", &ImageJMetadata::version)
        .def_ro("images", &ImageJMetadata::images)
        .def_ro("channels", &ImageJMetadata::channels)
        .def_ro("slices", &ImageJMetadata::slices)
        .def_ro("frames", &ImageJMetadata::frames)
        .def_ro("hyperstack", &ImageJMetadata::hyperstack)
        .def_ro("mode", &ImageJMetadata::mode)
        .def_ro("unit", &ImageJMetadata::unit)
        .def_ro("unit_um", &ImageJMetadata::unitUm, "The unit in micrometres; 0 when it is not a length.")
        .def_ro("spacing", &ImageJMetadata::spacing)
        .def_ro("frame_interval", &ImageJMetadata::frameInterval)
        .def_ro("min", &ImageJMetadata::min)
        .def_ro("max", &ImageJMetadata::max)
        .def_ro("has_range", &ImageJMetadata::hasRange)
        .def_ro("entries", &ImageJMetadata::entries, "Every key=value line.");

    nb::class_<TiffChannel>(m, "TiffChannel")
        .def_ro("name", &TiffChannel::name)
        .def_ro("emission_nm", &TiffChannel::emissionNm)
        .def_ro("has_color", &TiffChannel::hasColor)
        .def_ro("color", &TiffChannel::color, "(r, g, b), linear 0..1.");

    nb::class_<TiffMetadata>(m, "TiffMetadata",
                             "OME-XML or ImageJ metadata of a TIFF's first page. size_c / size_z / size_t, "
                             "dimension_order, voxel_um, frame_interval_s and channels describe the first "
                             "image in one vocabulary; ome_images / image_j hold everything parsed.")
        .def_ro("ome", &TiffMetadata::ome)
        .def_ro("imagej", &TiffMetadata::imagej)
        .def_ro("ome_images", &TiffMetadata::omeImages)
        .def_ro("ome_uuid", &TiffMetadata::omeUuid)
        .def_ro("image_j", &TiffMetadata::imageJ)
        .def_ro("size_c", &TiffMetadata::sizeC)
        .def_ro("size_z", &TiffMetadata::sizeZ)
        .def_ro("size_t", &TiffMetadata::sizeT)
        .def_ro("dimension_order", &TiffMetadata::dimensionOrder)
        .def_ro("voxel_um", &TiffMetadata::voxelUm, "(x, y, z) in micrometres; 0 when not given (ImageJ: z only).")
        .def_ro("frame_interval_s", &TiffMetadata::frameIntervalS)
        .def_ro("channels", &TiffMetadata::channels)
        .def("__repr__", [](const TiffMetadata& md) {
            std::ostringstream os;
            os << "TiffMetadata(" << (md.ome ? "OME" : md.imagej ? "ImageJ"
                                                                 : "none");
            if (md.ome) os << ", " << md.omeImages.size() << " image(s)";
            os << ", c" << md.sizeC << " z" << md.sizeZ << " t" << md.sizeT << ")";
            return os.str();
        });

    m.def("parse_tiff_metadata", &parseTiffMetadata, nb::arg("description"),
          "Parse an ImageDescription: OME-XML or ImageJ's header (TiffMetadata).");
    m.def("ome_image_pages", &omeImagePages, nb::arg("metadata"), nb::arg("page_count"),
          nb::arg("samples_per_pixel") = 1,
          "Page indices of every OME image, in each image's plane order (TiffData / IFD mapping).");

    nb::class_<TiffFile>(m, "TiffFile",
                         "Open a TIFF once and decode pages / pyramid levels / regions / series onto any device.\n\n"
                         "    f = TiffFile('stack.tif')\n"
                         "    f.info.shape                      # (pages, height, width) or (pages, samples, height, width)\n"
                         "    a = f.read_stack()                # numpy, native dtype; RGB as (pages, 3, y, x)\n"
                         "    g = f.read_stack(device='cuda')   # sirius.Buffer in GPU memory (nvTIFF decode)\n"
                         "    t = torch.from_dlpack(g)          # zero-copy adoption\n"
                         "    r = f.read_region(x, y, w, h, level=1, dtype='float32')\n"
                         "    f.metadata                        # OME-XML / ImageJ metadata (TiffMetadata)\n"
                         "    s = f.read_series(1)              # the second OME image\n")
        .def(nb::init<std::string>(), nb::arg("path"))
        .def_prop_ro("path", &TiffFile::path)
        .def_prop_ro("info", &TiffFile::info)
        .def_prop_ro("metadata", &TiffFile::metadata, nb::rv_policy::reference_internal)
        .def("series", &TiffFile::series,
             "Page indices of every series: one per OME image, else one series of every page.")
        .def("series_levels", &TiffFile::seriesLevels, nb::arg("series") = 0,
             "Pyramid levels of a series (its pages' SubIFDs, or the file's flat levels).")
        .def("series_ifds", &TiffFile::seriesIfds, nb::arg("series") = 0, nb::arg("level") = 0,
             "IFD offsets of a series at a pyramid level, in plane order.")
        .def("gpu_decodable", [](const TiffFile& f, Device device) {
                 std::string reason;
                 const bool ok = f.gpuDecodable(device, &reason);
                 return nb::make_tuple(ok, reason); }, nb::arg("device") = Device::cuda(0), "(ok, reason): whether nvTIFF can decode this file on `device` without the CPU fallback.")
        .def("read_stack", [](const TiffFile& f, nb::handle dtype, Device device, bool allowCpuFallback, bool pinned, const Stream* stream, std::uint16_t firstSample, std::uint16_t samples) {
                 const PixelType t = pixelTypeFromDtype(dtype).value_or(f.info().pixelType());
                 const auto opts = makeOptions(device, allowCpuFallback, pinned, firstSample, samples);
                 return toPython(withPixelType(t, [&](auto tag) -> AnyBuffer {
                     return f.readStack<decltype(tag)>(opts, streamOrNull(stream));
                 })); }, nb::arg("dtype") = nb::none(), nb::arg("device") = Device::cpu(), nb::arg("allow_cpu_fallback") = true, nb::arg("pinned") = false, nb::arg("stream") = nb::none(), nb::arg("first_sample") = 0, nb::arg("samples") = 0, "All full-resolution pages as (pages, height, width)." SIRIUS_SAMPLES_DOC)
        .def("read_pages", [](const TiffFile& f, std::size_t first, std::size_t count, nb::handle dtype, Device device, bool allowCpuFallback, bool pinned, const Stream* stream, std::uint16_t firstSample, std::uint16_t samples) {
                 const PixelType t = pixelTypeFromDtype(dtype).value_or(f.info().pixelType());
                 const auto opts = makeOptions(device, allowCpuFallback, pinned, firstSample, samples);
                 return toPython(withPixelType(t, [&](auto tag) -> AnyBuffer {
                     return f.readPages<decltype(tag)>(first, count, opts, streamOrNull(stream));
                 })); }, nb::arg("first"), nb::arg("count"), nb::arg("dtype") = nb::none(), nb::arg("device") = Device::cpu(), nb::arg("allow_cpu_fallback") = true, nb::arg("pinned") = false, nb::arg("stream") = nb::none(), nb::arg("first_sample") = 0, nb::arg("samples") = 0, "Pages [first, first + count)." SIRIUS_SAMPLES_DOC)
        .def("read_level", [](const TiffFile& f, std::size_t level, nb::handle dtype, Device device, bool allowCpuFallback, bool pinned, const Stream* stream, std::uint16_t firstSample, std::uint16_t samples) {
                 const PixelType t = pixelTypeFromDtype(dtype).value_or(f.info().pixelType());
                 const auto opts = makeOptions(device, allowCpuFallback, pinned, firstSample, samples);
                 return toPython(withPixelType(t, [&](auto tag) -> AnyBuffer {
                     return f.readLevel<decltype(tag)>(level, opts, streamOrNull(stream));
                 })); }, nb::arg("level"), nb::arg("dtype") = nb::none(), nb::arg("device") = Device::cpu(), nb::arg("allow_cpu_fallback") = true, nb::arg("pinned") = false, nb::arg("stream") = nb::none(), nb::arg("first_sample") = 0, nb::arg("samples") = 0, "Every page at pyramid level `level` (0 = full resolution)." SIRIUS_SAMPLES_DOC)
        .def("read_series", [](const TiffFile& f, std::size_t series, std::size_t level, nb::handle dtype, Device device, bool allowCpuFallback, bool pinned, const Stream* stream, std::uint16_t firstSample, std::uint16_t samples) {
                 const PixelType t = pixelTypeFromDtype(dtype).value_or(f.info().pixelType());
                 const auto opts = makeOptions(device, allowCpuFallback, pinned, firstSample, samples);
                 return toPython(withPixelType(t, [&](auto tag) -> AnyBuffer {
                     return f.readSeries<decltype(tag)>(series, level, opts, streamOrNull(stream));
                 })); }, nb::arg("series") = 0, nb::arg("level") = 0, nb::arg("dtype") = nb::none(), nb::arg("device") = Device::cpu(), nb::arg("allow_cpu_fallback") = true, nb::arg("pinned") = false, nb::arg("stream") = nb::none(), nb::arg("first_sample") = 0, nb::arg("samples") = 0, "The pages of series `series` (see series()) at pyramid level `level`, in plane order." SIRIUS_SAMPLES_DOC)
        .def("read_region", [](const TiffFile& f, std::uint32_t x, std::uint32_t y, std::uint32_t width, std::uint32_t height, std::size_t level, nb::handle dtype, Device device, bool allowCpuFallback, bool pinned, const Stream* stream, std::size_t first, std::size_t count, std::uint16_t firstSample, std::uint16_t samples) {
                 const PixelType t = pixelTypeFromDtype(dtype).value_or(f.info().pixelType());
                 const auto opts = makeOptions(device, allowCpuFallback, pinned, firstSample, samples);
                 const auto& levels = f.info().levels;
                 if (level >= levels.size())
                     throw std::out_of_range("Level " + std::to_string(level) + " requested from a TIFF with " +
                                             std::to_string(levels.size()) + " level(s)");
                 const TiffLevel& l = levels[level];
                 const std::size_t n = l.ifds.size();
                 // count 0: every page from `first` on. `count > n - first`
                 // rather than `first + count > n`: the sum can wrap.
                 if (count == 0 && first < n) count = n - first;
                 if (count == 0 || first >= n || count > n - first)
                     throw std::out_of_range("Pages [" + std::to_string(first) + ", +" + std::to_string(count) +
                                             ") requested from a TIFF level with " + std::to_string(n) + " page(s)");
                 const Region r = Region{x, y, width, height}.resolve(l.width, l.height);
                 const std::vector<std::uint64_t> ifds(l.ifds.begin() + static_cast<std::ptrdiff_t>(first),
                                                       l.ifds.begin() + static_cast<std::ptrdiff_t>(first + count));
                 return toPython(withPixelType(t, [&](auto tag) -> AnyBuffer {
                     using T = decltype(tag);
                     Buffer<T> out(f.readShape(count, r.height, r.width, opts), opts.device, opts.hostMemory, streamOrNull(stream));
                     f.decode<T>(ifds, r, out.view(), opts, streamOrNull(stream));
                     return AnyBuffer{std::move(out)};
                 })); }, nb::arg("x"), nb::arg("y"), nb::arg("width") = 0, nb::arg("height") = 0, nb::arg("level") = 0, nb::arg("dtype") = nb::none(), nb::arg("device") = Device::cpu(), nb::arg("allow_cpu_fallback") = true, nb::arg("pinned") = false, nb::arg("stream") = nb::none(), nb::arg("first") = 0, nb::arg("count") = 0, nb::arg("first_sample") = 0, nb::arg("samples") = 0, "Rectangle (x, y, width, height) of pages [first, first + count) at `level` (count 0: to the last page); width/height 0 extend to the edge." SIRIUS_SAMPLES_DOC);

    m.def("read_tiff", [](const std::string& path, nb::handle dtype, Device device, bool allowCpuFallback, std::uint16_t firstSample, std::uint16_t samples) {
              TiffFile f(path);
              const PixelType t = pixelTypeFromDtype(dtype).value_or(f.info().pixelType());
              const auto opts = makeOptions(device, allowCpuFallback, false, firstSample, samples);
              return toPython(withPixelType(t, [&](auto tag) -> AnyBuffer {
                  return f.readStack<decltype(tag)>(opts);
              })); }, nb::arg("path"), nb::arg("dtype") = nb::none(), nb::arg("device") = Device::cpu(), nb::arg("allow_cpu_fallback") = true, nb::arg("first_sample") = 0, nb::arg("samples") = 0, "Read a whole TIFF stack as (pages, height, width), or (pages, samples, height, width) for a "
                                                                                                                                                                                                                                                                                                                                                "multi-sample (RGB) image. Returns numpy on the CPU and a sirius.Buffer (DLPack) on CUDA devices." SIRIUS_SAMPLES_DOC);

    m.def("write_tiff", &writeArray, nb::arg("path"), nb::arg("image"), nb::arg("comp") = TiffCompression::None,
          "Write a 2-D (rows, cols) image or a 3-D (pages, rows, cols) stack in its own dtype: "
          "u/int 8/16/32, float32 or float64. Any other dtype (int64, bool, float16, ...) raises "
          "TypeError rather than being narrowed; a non-contiguous array is copied first.");
}
