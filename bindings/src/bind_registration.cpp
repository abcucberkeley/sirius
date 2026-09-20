// Python bindings for masked FFT registration and mosaic stitching.

#include "py_common.hpp"

#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <sirius/registration.hpp>
#include <sirius/stitching.hpp>
#include <sirius/stitching_tiff.hpp>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace sirius;

namespace {

    using HostArray = nb::ndarray<nb::c_contig, nb::device::cpu>;
    using MaskArray = nb::ndarray<const std::uint8_t, nb::c_contig, nb::device::cpu>;

    Shape shapeOf(const HostArray& a, const char* what) {
        if (a.ndim() < 2 || a.ndim() > 3)
            throw std::invalid_argument(std::string(what) + " must be a 2D or 3D array, got " +
                                        std::to_string(a.ndim()) + " dimensions");
        std::vector<Index> dims(a.ndim());
        for (std::size_t i = 0; i < a.ndim(); ++i) dims[i] = static_cast<Index>(a.shape(i));
        return Shape(dims.begin(), dims.end());
    }

    // The pixel types registration is instantiated for.
    PixelType registrationDtype(const HostArray& a, const char* what) {
        const nb::dlpack::dtype dt = a.dtype();
        if (dt == nb::dtype<double>()) return PixelType::Float64;
        if (dt == nb::dtype<float>()) return PixelType::Float32;
        if (dt == nb::dtype<std::uint16_t>()) return PixelType::UInt16;
        if (dt == nb::dtype<std::uint8_t>()) return PixelType::UInt8;
        throw std::invalid_argument(std::string(what) + " must be float64, float32, uint16 or uint8");
    }

    // Call f(T{}) for one of the four supported types.
    template <typename F>
    decltype(auto) withRegistrationType(PixelType t, F&& f) {
        switch (t) {
            case PixelType::Float64: return f(double{});
            case PixelType::Float32: return f(float{});
            case PixelType::UInt16: return f(std::uint16_t{});
            case PixelType::UInt8: return f(std::uint8_t{});
            default: break;
        }
        throw std::invalid_argument("unsupported pixel type for registration");
    }

    template <typename T>
    BufferView<const T> viewOf(const HostArray& a, const char* what) {
        return {static_cast<const T*>(a.data()), shapeOf(a, what), Device::cpu()};
    }

    BufferView<const std::uint8_t> maskView(const std::optional<MaskArray>& mask, const Shape& image,
                                            const char* what) {
        if (!mask) return {};
        std::vector<Index> dims(mask->ndim());
        for (std::size_t i = 0; i < mask->ndim(); ++i) dims[i] = static_cast<Index>(mask->shape(i));
        Shape s(dims.begin(), dims.end());
        if (s != image)
            throw std::invalid_argument(std::string(what) + " must have the same shape as its image");
        return {mask->data(), s, Device::cpu()};
    }

    nb::object numpyFrom(Buffer<double>&& b) {
        return sirius_py::toPython(AnyBuffer(std::move(b)));
    }

} // namespace

void bind_registration(nb::module_& m) {
    m.def("next_fast_fft_size", &nextFastFFTSize, nb::arg("n"),
          "Smallest 2/3/5/7-smooth size at least n, the padding masked correlation uses.");

    nb::class_<MaskedNccOptions>(m, "MaskedNccOptions",
                                 "Filters and planning options for masked normalized cross-correlation.")
        .def(nb::init<>())
        .def_rw("required_overlap_voxels", &MaskedNccOptions::requiredOverlapVoxels)
        .def_rw("required_overlap_fraction", &MaskedNccOptions::requiredOverlapFraction)
        .def_rw("max_shift", &MaskedNccOptions::maxShift)
        .def_rw("shift_centre", &MaskedNccOptions::shiftCentre)
        .def_rw("subpixel", &MaskedNccOptions::subpixel)
        .def_rw("rigor", &MaskedNccOptions::rigor);

    nb::class_<TranslationResult>(m, "TranslationResult",
                                  "Displacement of the moving image relative to the fixed one, in (z, y, x) voxels.")
        .def_ro("shift", &TranslationResult::shift)
        .def_ro("integer_shift", &TranslationResult::integerShift)
        .def_ro("correlation", &TranslationResult::correlation)
        .def_ro("overlap", &TranslationResult::overlap)
        .def_ro("valid", &TranslationResult::valid)
        .def("__repr__", [](const TranslationResult& r) {
            return std::string("TranslationResult(shift=(") + std::to_string(r.shift[0]) + ", " +
                   std::to_string(r.shift[1]) + ", " + std::to_string(r.shift[2]) +
                   "), correlation=" + std::to_string(r.correlation) +
                   ", valid=" + (r.valid ? "True" : "False") + ")";
        });

    m.def("masked_ncc", [](HostArray fixed, HostArray moving, std::optional<MaskArray> fixedMask, std::optional<MaskArray> movingMask, const MaskedNccOptions& options) {
              const PixelType t = registrationDtype(fixed, "fixed");
              if (registrationDtype(moving, "moving") != t)
                  throw std::invalid_argument("fixed and moving must have the same dtype");
              MaskedNccResult result;
              withRegistrationType(t, [&](auto tag) {
                  using T = decltype(tag);
                  const auto f = viewOf<T>(fixed, "fixed");
                  const auto g = viewOf<T>(moving, "moving");
                  const auto fm = maskView(fixedMask, f.shape(), "fixed_mask");
                  const auto gm = maskView(movingMask, g.shape(), "moving_mask");
                  nb::gil_scoped_release release;
                  result = maskedNormalizedCrossCorrelation<T>(f, g, fm, gm, options);
              });
              return nb::make_tuple(numpyFrom(std::move(result.correlation)),
                                    numpyFrom(std::move(result.overlap))); }, nb::arg("fixed"), nb::arg("moving"), nb::arg("fixed_mask") = nb::none(), nb::arg("moving_mask") = nb::none(), nb::arg("options") = MaskedNccOptions{}, "Masked normalized cross-correlation (Padfield 2012).\n\n"
                                                                                                                                                                                                                                                                                                                                    "Returns (correlation, overlap), both of shape fixed.shape + moving.shape - 1.\n"
                                                                                                                                                                                                                                                                                                                                    "Index i along an axis is the displacement i - (moving.shape - 1).");

    m.def("register_translation_masked", [](HostArray fixed, HostArray moving, std::optional<MaskArray> fixedMask, std::optional<MaskArray> movingMask, const MaskedNccOptions& options) {
              const PixelType t = registrationDtype(fixed, "fixed");
              if (registrationDtype(moving, "moving") != t)
                  throw std::invalid_argument("fixed and moving must have the same dtype");
              TranslationResult out;
              withRegistrationType(t, [&](auto tag) {
                  using T = decltype(tag);
                  const auto f = viewOf<T>(fixed, "fixed");
                  const auto g = viewOf<T>(moving, "moving");
                  const auto fm = maskView(fixedMask, f.shape(), "fixed_mask");
                  const auto gm = maskView(movingMask, g.shape(), "moving_mask");
                  nb::gil_scoped_release release;
                  out = registerTranslationMasked<T>(f, g, fm, gm, options);
              });
              return out; }, nb::arg("fixed"), nb::arg("moving"), nb::arg("fixed_mask") = nb::none(), nb::arg("moving_mask") = nb::none(), nb::arg("options") = MaskedNccOptions{}, "Displacement of `moving` relative to `fixed`: moving[p] matches fixed[p + shift].");

    // --- stitching ---------------------------------------------------------

    nb::enum_<BlendMode>(m, "BlendMode")
        .value("Overwrite", BlendMode::Overwrite)
        .value("Average", BlendMode::Average)
        .value("Feather", BlendMode::Feather)
        .value("Maximum", BlendMode::Maximum);

    nb::class_<StitchOptions>(m, "StitchOptions", "Mosaic stitching parameters.")
        .def(nb::init<>())
        .def_rw("min_overlap_fraction", &StitchOptions::minOverlapFraction)
        .def_rw("search_radius", &StitchOptions::searchRadius)
        .def_rw("min_correlation", &StitchOptions::minCorrelation)
        .def_rw("mask_background", &StitchOptions::maskBackground)
        .def_rw("background_level", &StitchOptions::backgroundLevel)
        .def_rw("rigor", &StitchOptions::rigor)
        .def_rw("nominal_weight", &StitchOptions::nominalWeight)
        .def_rw("anchor_tile", &StitchOptions::anchorTile)
        .def_rw("blend", &StitchOptions::blend)
        .def_rw("feather_width", &StitchOptions::featherWidth)
        .def_rw("skip_background", &StitchOptions::skipBackground)
        .def_rw("fusion_background_level", &StitchOptions::fusionBackgroundLevel);

    nb::class_<TileMatch>(m, "TileMatch", "One measured relationship between two tiles.")
        .def(nb::init<>())
        .def_rw("fixed", &TileMatch::fixed)
        .def_rw("moving", &TileMatch::moving)
        .def_rw("displacement", &TileMatch::displacement)
        .def_rw("nominal_displacement", &TileMatch::nominalDisplacement)
        .def_rw("correlation", &TileMatch::correlation)
        .def_rw("overlap", &TileMatch::overlap)
        .def_rw("accepted", &TileMatch::accepted)
        .def("__repr__", [](const TileMatch& t) {
            return std::string("TileMatch(") + std::to_string(t.fixed) + " -> " +
                   std::to_string(t.moving) + ", correlation=" + std::to_string(t.correlation) +
                   ", accepted=" + (t.accepted ? "True" : "False") + ")";
        });

    nb::class_<StitchLayout>(m, "StitchLayout", "Refined tile origins and the canvas they cover.")
        .def_ro("positions", &StitchLayout::positions)
        .def_ro("matches", &StitchLayout::matches)
        .def_ro("canvas_origin", &StitchLayout::canvasOrigin)
        .def_ro("canvas_extent", &StitchLayout::canvasExtent);

    m.def("register_tile_pair", [](HostArray fixed, std::array<double, 3> fixedPosition, HostArray moving, std::array<double, 3> movingPosition, const StitchOptions& options) {
              const PixelType t = registrationDtype(fixed, "fixed");
              if (registrationDtype(moving, "moving") != t)
                  throw std::invalid_argument("fixed and moving must have the same dtype");
              TileMatch out;
              withRegistrationType(t, [&](auto tag) {
                  using T = decltype(tag);
                  const auto f = viewOf<T>(fixed, "fixed");
                  const auto g = viewOf<T>(moving, "moving");
                  nb::gil_scoped_release release;
                  out = registerTilePair<T>(f, fixedPosition, g, movingPosition, options);
              });
              return out; }, nb::arg("fixed"), nb::arg("fixed_position"), nb::arg("moving"), nb::arg("moving_position"), nb::arg("options") = StitchOptions{}, "Measure the displacement between two tiles placed at their nominal origins.");

    m.def("optimize_tile_positions", &optimizeTilePositions, nb::arg("nominal"), nb::arg("matches"),
          nb::arg("nominal_weight") = 1e-3, nb::arg("anchor") = std::size_t(0),
          "Tile origins that best explain every accepted match.");

    m.def("plan_stitch", [](nb::sequence tiles, std::vector<std::array<double, 3>> positions, const StitchOptions& options) {
              std::vector<HostArray> arrays;
              for (nb::handle h : tiles) arrays.push_back(nb::cast<HostArray>(h));
              if (arrays.empty()) throw std::invalid_argument("plan_stitch: no tiles given");
              const PixelType t = registrationDtype(arrays.front(), "tile 0");
              StitchLayout layout;
              withRegistrationType(t, [&](auto tag) {
                  using T = decltype(tag);
                  std::vector<BufferView<const T>> views;
                  views.reserve(arrays.size());
                  for (const HostArray& a : arrays) {
                      if (registrationDtype(a, "tile") != t)
                          throw std::invalid_argument("plan_stitch: every tile must share a dtype");
                      views.push_back(viewOf<T>(a, "tile"));
                  }
                  nb::gil_scoped_release release;
                  layout = planStitch<T>(views, positions, options);
              });
              return layout; }, nb::arg("tiles"), nb::arg("positions"), nb::arg("options") = StitchOptions{}, "Register every overlapping pair and solve for the tile origins.");

    m.def("fuse_tiles", [](nb::sequence tiles, std::vector<std::array<double, 3>> positions, std::array<Index, 3> canvasOrigin, std::array<Index, 3> canvasExtent, const StitchOptions& options) {
              std::vector<HostArray> arrays;
              for (nb::handle h : tiles) arrays.push_back(nb::cast<HostArray>(h));
              if (arrays.empty()) throw std::invalid_argument("fuse_tiles: no tiles given");
              const PixelType t = registrationDtype(arrays.front(), "tile 0");
              nb::object out;
              withRegistrationType(t, [&](auto tag) {
                  using T = decltype(tag);
                  std::vector<BufferView<const T>> views;
                  views.reserve(arrays.size());
                  for (const HostArray& a : arrays) {
                      if (registrationDtype(a, "tile") != t)
                          throw std::invalid_argument("fuse_tiles: every tile must share a dtype");
                      views.push_back(viewOf<T>(a, "tile"));
                  }
                  Buffer<T> fused;
                  {
                      nb::gil_scoped_release release;
                      fused = fuseTiles<T>(views, positions, canvasOrigin, canvasExtent, options);
                  }
                  out = sirius_py::toPython(AnyBuffer(std::move(fused)));
              });
              return out; }, nb::arg("tiles"), nb::arg("positions"), nb::arg("canvas_origin"), nb::arg("canvas_extent"), nb::arg("options") = StitchOptions{}, "Blend tiles onto one canvas at the given (rounded) origins.");

    m.def("stitch_tiff_tiles", [](const std::vector<std::string>& paths, const std::vector<std::array<double, 3>>& positions, const StitchOptions& options, const std::string& outputPath, nb::handle dtype, TiffCompression compression) {
              if (paths.size() != positions.size())
                  throw std::invalid_argument("stitch_tiff_tiles: one position per tile is required");
              if (paths.empty()) throw std::invalid_argument("stitch_tiff_tiles: no tiles given");
              std::vector<StitchTile> tiles(paths.size());
              for (std::size_t i = 0; i < paths.size(); ++i)
                  tiles[i] = StitchTile{paths[i], positions[i]};

              // Default to the file's own pixel type; the caller can widen it.
              PixelType t = inspectTiff(paths.front()).pixelType();
              if (!dtype.is_none()) {
                  const auto requested = sirius_py::pixelTypeFromDtype(dtype);
                  if (!requested) throw std::invalid_argument("stitch_tiff_tiles: unsupported dtype");
                  t = *requested;
              }
              if (t != PixelType::Float64 && t != PixelType::Float32 && t != PixelType::UInt16 &&
                  t != PixelType::UInt8)
                  t = PixelType::Float32;

              StitchLayout layout;
              nb::object fused;
              withRegistrationType(t, [&](auto tag) {
                  using T = decltype(tag);
                  Buffer<T> result;
                  {
                      nb::gil_scoped_release release;
                      result = stitchTiffTiles<T>(tiles, options, &layout, outputPath, compression);
                  }
                  fused = sirius_py::toPython(AnyBuffer(std::move(result)));
              });
              return nb::make_tuple(fused, layout); }, nb::arg("paths"), nb::arg("positions"), nb::arg("options") = StitchOptions{}, nb::arg("output_path") = std::string{}, nb::arg("dtype") = nb::none(), nb::arg("compression") = TiffCompression::None, "Read TIFF tiles, stitch them and optionally write the mosaic.\n\n"
                                                                                                                                                                                                                                                                                                                                                                                                                                                    "Returns (mosaic, layout). Every tile and the canvas are held in memory.");
}
