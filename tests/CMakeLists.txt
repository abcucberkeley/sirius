# One test executable per unit (cmake/Units.cmake): test_<unit> links the unit
# under test, the units it depends on, and nothing else -- so `buffer` is
# tested without an FFT in the binary, `registration` without a TIFF reader,
# and a unit that reaches for something it did not declare fails to link here
# first. Every TEST_CASE is a CTest test named "sirius::<case>", as it was when
# this was one binary.
#
#     cmake --build <build> --target test_registration
#     ctest --test-dir <build> -L registration        # labels are lib.<unit> / core.<unit>; -L is a regex
#
# GPU-only cases SKIP at run time when no CUDA device is usable, so the same
# binaries pass on CPU-only CI; the [cuda] tag selects them.

include(Catch)
include(Units)

# sirius_lib_test(<unit> <sources>... [UNITS <more units>...] [LINK ...] [INTERNAL])
# The test of a library unit: the unit itself plus whatever else its test
# sources include (a Buffer to hold the data, a TIFF to read a fixture).
function(sirius_lib_test unit)
    cmake_parse_arguments(PARSE_ARGV 1 T "INTERNAL" "" "UNITS;LINK")
    set(internal "")
    if(T_INTERNAL)
        set(internal INTERNAL)
    endif()
    sirius_unit_test(${unit}
        SOURCES ${T_UNPARSED_ARGUMENTS}
        UNITS lib/${unit} ${T_UNITS}
        LINK ${T_LINK}
        LABEL lib.${unit}
        ${internal})
endfunction()

# --- the library ---------------------------------------------------------------

sirius_lib_test(tensor_util     test_tensor_util.cpp)
sirius_lib_test(fft_util        test_fft_util.cpp)
sirius_lib_test(buffer          test_buffer.cpp)        # and `device`, which has no life apart from it
sirius_lib_test(fft             test_fft1d.cpp test_fft2d.cpp test_fft3d.cpp test_fft_inplace.cpp test_fft_cuda.cpp)
sirius_lib_test(real_fft        test_real_fft.cpp test_real_fft_cuda.cpp UNITS lib/buffer)
# test_tiff_file reads tags back with libtiff and includes tiff_internal.hpp
sirius_lib_test(tiff_io         test_tiff_io.cpp test_tiff_file.cpp LINK TIFF::TIFF INTERNAL)
sirius_lib_test(zarr_io         test_zarr_io.cpp)
# downsample.hpp, the internal header image_ops shares with the writers, is tested here too
sirius_lib_test(image_ops       test_image_ops.cpp INTERNAL)
sirius_lib_test(registration    test_registration.cpp)
sirius_lib_test(deconvolution   test_deconvolution.cpp)
sirius_lib_test(stitching       test_stitching.cpp)
sirius_lib_test(stitching_tiff  test_stitching_tiff.cpp)
sirius_lib_test(sim_parameters  test_sim_parameters.cpp)
sirius_lib_test(legacy_config   test_legacy_config.cpp)
# the library-internal OpenMP kernels shared by the public preprocessing API
# and the reconstruction pipeline
sirius_lib_test(sim_stages      test_sim_cpu_stages.cpp INTERNAL)
sirius_lib_test(preprocess      test_preprocess.cpp)
sirius_lib_test(separation      test_separation.cpp)
sirius_lib_test(otf             test_otf.cpp)
sirius_lib_test(otf_io          test_otf_io.cpp)
sirius_lib_test(otf_ideal       test_otf_ideal.cpp)
sirius_lib_test(sim_reconstruction test_reconstruction.cpp
    UNITS lib/otf_io lib/otf_ideal lib/legacy_config lib/tiff_io)

# --- the workbench core ----------------------------------------------------------
# The GUI's Qt-free model is tested the same way when the app is enabled; the
# Widgets layer itself stays untested here (tools/gui_tests.py drives it).
if(SIRIUS_ENABLE_APP)
    include(app_tests.cmake)
endif()

# --- all of them in one binary ------------------------------------------------------
# Not registered with CTest (the cases above already are): it is for running
# across units by tag -- `sirius_tests "[cuda]"`, `sirius_tests "[parity]"` --
# and it links the *archives*, which shows that `sirius` and `sirius_app_core`
# still hold everything the units do.
get_property(_sirius_test_objects GLOBAL PROPERTY SIRIUS_TEST_OBJECTS)
add_executable(sirius_tests)
target_link_libraries(sirius_tests PRIVATE ${_sirius_test_objects} sirius Catch2::Catch2WithMain)
if(SIRIUS_ENABLE_APP)
    target_link_libraries(sirius_tests PRIVATE sirius::app_core)
endif()
sirius_finish_test_executable(sirius_tests)

# The workbench as `cmake --install` lays it out, run from there (Linux only;
# see app/CMakeLists.txt). With $SIRIUS_PYTHON set when the test runs it also
# starts the installed worker.
if(SIRIUS_ENABLE_APP AND UNIX AND NOT APPLE)
    add_test(NAME sirius::app.install
        COMMAND ${CMAKE_COMMAND}
            -DBUILD_DIR=${PROJECT_BINARY_DIR}
            -DPREFIX=${CMAKE_CURRENT_BINARY_DIR}/install-check
            -DCONFIG=$<CONFIG>
            -DRAW=${PROJECT_SOURCE_DIR}/tests/data/raw.tif
            -P ${PROJECT_SOURCE_DIR}/app/linux/check_install.cmake)
    set_tests_properties(sirius::app.install PROPERTIES TIMEOUT 300 LABELS app)
endif()
