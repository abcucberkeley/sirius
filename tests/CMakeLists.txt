add_executable(sirius_tests)

target_sources(sirius_tests
    PRIVATE
        test_tiff_io.cpp
        test_tiff_file.cpp
        test_buffer.cpp
        test_tensor_util.cpp
        test_fft_util.cpp
        test_fft1d.cpp
        test_fft2d.cpp
        test_fft3d.cpp
        test_fft_inplace.cpp
        test_real_fft.cpp
        test_fft_cuda.cpp
        test_real_fft_cuda.cpp
        test_sim_parameters.cpp
        test_separation.cpp
        test_reconstruction.cpp
        test_otf.cpp
        test_preprocess.cpp
        test_sim_cpu_stages.cpp
        test_registration.cpp
        test_stitching.cpp
        test_image_ops.cpp
        test_deconvolution.cpp
        test_zarr_io.cpp
)

target_link_libraries(sirius_tests
    PRIVATE
        sirius
        Catch2::Catch2WithMain
        TIFF::TIFF
        # Tests that include a library-internal header directly (downsample.hpp,
        # sim_cpu_stages.hpp) compile its OpenMP pragmas themselves, so the test
        # binary needs the same flag the library is built with -- otherwise the
        # pragmas are silently ignored and the tested code is not the shipped one.
        OpenMP::OpenMP_CXX
)

# test_sim_cpu_stages exercises the library-internal OpenMP kernels shared by
# the public preprocessing API and the reconstruction pipeline
target_include_directories(sirius_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_compile_definitions(sirius_tests
    PRIVATE
        SIRIUS_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data"
)

# The GUI's Qt-free model (session, display mapping) is tested with the same
# binary when the app is enabled; the Widgets layer itself stays untested here.
if(SIRIUS_ENABLE_APP)
    target_sources(sirius_tests PRIVATE
        test_app_core.cpp
        test_app_labels.cpp
        test_app_tracking.cpp
        test_app_training.cpp
        test_app_pipeline.cpp
        test_app_ops.cpp
        test_app_io.cpp
        test_app_rpc.cpp
        test_app_help.cpp
        test_app_manifest.cpp
        test_app_schema.cpp
        test_app_parity.cpp
    )
    target_link_libraries(sirius_tests PRIVATE sirius::app_core)
endif()

sirius_copy_runtime_dlls(sirius_tests)

include(Catch)
# GPU-only cases SKIP at run time when no CUDA device is usable, so the same
# binary passes on CPU-only CI. `ctest -L cuda` / the [cuda] tag select them.
catch_discover_tests(sirius_tests TEST_PREFIX "sirius::")

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

if(SIRIUS_ENABLE_WARNINGS)
    target_set_warnings(sirius_tests)
endif()

if(SIRIUS_ENABLE_SANITIZERS)
    target_enable_sanitizers(sirius_tests)
endif()
