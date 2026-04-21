#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

#include <sirius/fft.hpp>

namespace nb = nanobind;
using namespace sirius;

using CdArray = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<1>, nb::c_contig>;

void bind_fft(nb::module_& m) {
    // PlanRigor enum
    nb::enum_<PlanRigor>(m, "PlanRigor")
        .value("Estimate", PlanRigor::Estimate)
        .value("Measure", PlanRigor::Measure)
        .value("Patient", PlanRigor::Patient)
        .value("Exhaustive", PlanRigor::Exhaustive)
        .export_values();

    // TODO: Update bindings to use the new fft class

    // // FFT1D class 
    // // in Python's idiomatic return style callers just do: 
    // //     out = fft.forward(in) # out is allocated then returned
    // // otherwise data can be preallocated
    // //    out = np.empty(n, dtype=np.complex128)
    // //    fft.forward(in_arr, out)  # no allocation per call
    // nb::class_<FFT1D>(m, "FFT1D")
    //     .def(nb::init<Eigen::Index, PlanRigor, bool>(),
    //          nb::arg("n"),
    //          nb::arg("plan_rigor") = PlanRigor::Measure,
    //          nb::arg("normalize") = false)
    //     .def("forward", [](const FFT1D& self, CdArray in) {
    //         Eigen::Map<const Eigen::VectorXcd> in_map(in.data(), in.shape(0));
    //         Eigen::VectorXcd out(in.shape(0));
    //         self.forward(in_map, out);
    //         return out;
    //     }, nb::arg("in"))
    //     .def("forward", [](const FFT1D& self, CdArray in, CdArray out) {
    //         Eigen::Map<const Eigen::VectorXcd> in_map(in.data(), in.shape(0));
    //         Eigen::Map<Eigen::VectorXcd>      out_map(out.data(), out.shape(0));
    //         self.forward(in_map, out_map);
    //     }, nb::arg("in"), nb::arg("out"))
    //     .def("inverse", [](const FFT1D& self, CdArray in) {
    //         Eigen::Map<const Eigen::VectorXcd> in_map(in.data(), in.shape(0));
    //         Eigen::VectorXcd out(in.shape(0));
    //         self.inverse(in_map, out);
    //         return out;
    //     }, nb::arg("in"))
    //     .def("inverse", [](const FFT1D& self, CdArray in, CdArray out) {
    //         Eigen::Map<const Eigen::VectorXcd> in_map(in.data(), in.shape(0));
    //         Eigen::Map<Eigen::VectorXcd>      out_map(out.data(), out.shape(0));
    //         self.inverse(in_map, out_map);
    //     }, nb::arg("in"), nb::arg("out"))
    //     .def_static("load_wisdom", &FFT1D::loadWisdom, nb::arg("path"))
    //     .def_static("save_wisdom", &FFT1D::saveWisdom, nb::arg("path"));
}