#include "py_common.hpp"

#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <sirius/legacy_config.hpp>
#include <sirius/otf_select.hpp>
#include <sirius/sim_reconstruction.hpp>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace nb = nanobind;
using namespace sirius;
using sirius_py::PyBuffer;

namespace {
    using DoubleArray = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;

    Shape arrayShape(const DoubleArray& a) {
        std::vector<Index> dims(a.ndim());
        for (std::size_t i = 0; i < a.ndim(); ++i)
            dims[i] = static_cast<Index>(a.shape(i));
        return Shape(dims.begin(), dims.end());
    }

    // `threeD` reaches the theoretical OTF only, and the DATA decides it:
    // SIMParameters::planes(sections) > 1, which is session.cpp's threeD().
    // The constructor is handed parameters, not a stack, so it cannot derive
    // the value -- and a default of either parity is a guess that silently
    // builds the wrong OTF (a 2D stack got the 3D theoretical OTF, missing
    // cone and all, that no front would have chosen for it: the Python
    // front's finding B). So it is REQUIRED exactly where it means something
    // -- no OTF file -- and ignored, as the library ignores it, when a file
    // is named.
    bool resolveThreeD(const std::string& otfPath, std::optional<bool> threeD) {
        if (threeD) return *threeD;
        if (!otfPath.empty()) return false;   // a file's table; the flag is not read
        throw std::invalid_argument(
            "SimReconstructor: three_d has to be given when no OTF file is named, because the theoretical OTF is "
            "built in 3D for a stack of several planes and in 2D for one, and parameters alone do not say which "
            "this is. Pass three_d=parameters.planes(sections) > 1, where sections is the raw stack's z extent "
            "(sirius.workbench.step_sim does exactly that).");
    }

    class PySimReconstructor {
    public:
        // An empty otfPath is the theoretical OTF, exactly as it is for the
        // GUI and the CLI: the choice is the library's one selectOTF, not a
        // second one written here (app/core/session.cpp makes it the same
        // way).
        PySimReconstructor(SIMParameters params, const std::string& otfPath,
                           Device device, PlanRigor rigor, std::optional<bool> threeD)
            : impl_(params, selectOTF(otfPath, params, resolveThreeD(otfPath, threeD)), device, rigor) {}

        Device device() const noexcept { return impl_.device(); }

        nb::object reconstructArray(DoubleArray raw) {
            BufferView<const double> view(raw.data(), arrayShape(raw), Device::cpu());
            Buffer<double> result;
            {
                nb::gil_scoped_release release;
                result = impl_.reconstruct(view);
            }
            return sirius_py::toPython(AnyBuffer(std::move(result)));
        }

        nb::object reconstructBuffer(const PyBuffer& raw) {
            const auto* input = std::get_if<Buffer<double>>(&raw.any());
            if (!input)
                throw std::invalid_argument("SIM input Buffer must have dtype float64");
            Buffer<double> result;
            {
                nb::gil_scoped_release release;
                result = impl_.reconstruct(input->view());
            }
            return sirius_py::toPython(AnyBuffer(std::move(result)));
        }

        const SimFit& lastFit() const noexcept { return impl_.lastFit(); }

    private:
        SimReconstructor impl_;
    };
} // namespace

void bind_sim(nb::module_& m) {
    nb::enum_<ApodizationType>(m, "ApodizationType")
        .value("None_", ApodizationType::None)
        .value("Cosine", ApodizationType::Cosine)
        .value("Triangle", ApodizationType::Triangle);

    nb::class_<SIMParameters>(m, "SIMParameters",
                              "Parameters for 3-beam structured-illumination reconstruction.")
        .def(nb::init<>())
        .def_rw("k0_start_angle", &SIMParameters::k0_start_angle)
        .def_rw("linespacing_um", &SIMParameters::linespacing_um)
        .def_rw("ndirs", &SIMParameters::ndirs)
        .def_rw("nphases", &SIMParameters::nphases)
        .def_rw("norders", &SIMParameters::norders, "Orders to separate; 0 (the default) derives nphases // 2 + 1.")
        .def_rw("na", &SIMParameters::na)
        .def_rw("nimm", &SIMParameters::nimm)
        .def_rw("wavelength_nm", &SIMParameters::wavelength_nm)
        .def_rw("k0_angles", &SIMParameters::k0_angles)
        .def_rw("dx", &SIMParameters::dx)
        .def_rw("dy", &SIMParameters::dy)
        .def_rw("dz", &SIMParameters::dz)
        .def_rw("dz_psf", &SIMParameters::dz_psf)
        .def_rw("zoomfact", &SIMParameters::zoomfact)
        .def_rw("z_zoom", &SIMParameters::z_zoom)
        .def_rw("wiener", &SIMParameters::wiener)
        .def_rw("otfcutoff", &SIMParameters::otfcutoff)
        .def_rw("background", &SIMParameters::background)
        .def_rw("apodize_input", &SIMParameters::apodize_input)
        .def_rw("napodize", &SIMParameters::napodize)
        .def_rw("suppression_radius", &SIMParameters::suppression_radius)
        .def_rw("suppress_singularities", &SIMParameters::suppress_singularities)
        .def_rw("dampen_order0", &SIMParameters::dampen_order0)
        .def_rw("apodize_output", &SIMParameters::apodize_output)
        .def_rw("explodefact", &SIMParameters::explodefact)
        .def_rw("fast_si", &SIMParameters::fast_si)
        .def_rw("do_rescale", &SIMParameters::do_rescale)
        .def_rw("equalizez", &SIMParameters::equalizez)
        .def_rw("no_kz0", &SIMParameters::no_kz0)
        .def_rw("filter_overlaps", &SIMParameters::filter_overlaps)
        .def("validate", &SIMParameters::validate)
        .def("sections_per_plane", &SIMParameters::sectionsPerPlane,
             "Frames a raw SIM stack holds per plane: ndirs * nphases.")
        .def("planes", &SIMParameters::planes, nb::arg("sections"),
             "Planes (nz) of a raw stack of `sections` frames, or 0 when the count is not a whole "
             "number of planes. More than one plane is a 3D stack, which is what the theoretical "
             "OTF's `three_d` follows.")
        .def("resolved_orders", &SIMParameters::resolvedOrders,
             "Orders the reconstruction separates: norders, or nphases // 2 + 1 when norders is 0.")
        .def("section_count_problem", &SIMParameters::sectionCountProblem, nb::arg("sections"),
             "Empty when `sections` is a whole number of planes, otherwise the one sentence every "
             "front says about it -- so the Python mirror does not keep its own copy of the wording.");

    nb::class_<SimFit>(m, "SimFit")
        .def_ro("k0", &SimFit::k0)
        .def_ro("amps", &SimFit::amps);

    m.def("sim_image_size_problem", &simImageSizeProblem, nb::arg("nx"), nb::arg("ny"),
          nb::arg("montage") = false,
          "Empty when a raw SIM stack's lateral extents can be reconstructed, otherwise the one "
          "sentence every front says about them. Either parity reconstructs; only the minimum is "
          "refused. SimReconstructor raises this same text as a ValueError.");

    m.def("sim_layout_counts_problem", &simLayoutCountsProblem, nb::arg("layout_text"), nb::arg("layout_angles"),
          nb::arg("layout_phases"), nb::arg("step_angles"), nb::arg("step_phases"),
          "Empty when a dataset's declared storage layout holds the angles and phases the step uses, otherwise the "
          "one sentence every front refuses the pipeline with. The SIM operation's validate() and "
          "sirius.workbench.step_sim raise this same text, so a mismatch reads the same in the window, in a session "
          "and in Python.");
    m.def("sim_declared_counts_note", &simDeclaredCountsNote, nb::arg("dataset_angles"), nb::arg("dataset_phases"),
          nb::arg("step_angles"), nb::arg("step_phases"),
          "Empty when a dataset's declared angle and phase counts are the step's, otherwise the one sentence every "
          "front WARNS with -- the weaker case, where the dataset says what it holds but not how it is stored, so "
          "the step's own counts are used and the run goes ahead.");

    m.def("load_parameters", &loadParameters, nb::arg("path"));
    m.def("save_parameters", &saveParameters, nb::arg("path"), nb::arg("parameters"));
    m.def("load_legacy_parameters", [](const std::string& path) { return fromLegacy(loadLegacyConfig(path)); }, nb::arg("path"));

    nb::class_<PySimReconstructor>(m, "SimReconstructor",
                                   "Reusable CPU/GPU SIM reconstructor. FFT plans and work buffers are retained "
                                   "between calls; construct once for a time series.\n\n"
                                   "otf_path empty (the default) is the theoretical OTF of an aberration-free "
                                   "objective with the parameters' NA, immersion index and emission wavelength -- "
                                   "what the GUI and the CLI use when their OTF field is empty. three_d then picks "
                                   "the 3D OTF (missing cone, order 1 shifted by the illumination's kz) over the "
                                   "in-focus 2D one, and has to be given, because the DATA decides it and the "
                                   "parameters do not say: parameters.planes(sections) > 1, with sections the raw "
                                   "stack's z extent. It is ignored, and so may be left out, when an OTF file is "
                                   "named.")
        .def(nb::init<SIMParameters, const std::string&, Device, PlanRigor, std::optional<bool>>(),
             nb::arg("parameters"), nb::arg("otf_path") = std::string(),
             nb::arg("device") = Device::cpu(),
             nb::arg("rigor") = PlanRigor::Measure,
             nb::arg("three_d") = nb::none())
        .def_prop_ro("device", &PySimReconstructor::device)
        .def_prop_ro("last_fit", &PySimReconstructor::lastFit,
                     nb::rv_policy::reference_internal)
        .def("reconstruct", &PySimReconstructor::reconstructArray, nb::arg("raw"),
             "Reconstruct a C-contiguous float64 NumPy stack on the CPU.")
        .def("reconstruct", &PySimReconstructor::reconstructBuffer, nb::arg("raw"),
             "Reconstruct a float64 sirius.Buffer on the reconstructor's device.");
}
