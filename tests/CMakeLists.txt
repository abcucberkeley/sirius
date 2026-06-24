add_executable(sirius_tests)

target_sources(sirius_tests
    PRIVATE
        test_tiff_io.cpp
        test_tensor_util.cpp
        test_fft_util.cpp
        test_fft1d.cpp
        test_fft2d.cpp
        test_fft3d.cpp
        test_real_fft.cpp
        test_sim_parameters.cpp
        test_reconstruction.cpp
        test_otf.cpp
        test_preprocess.cpp
)

target_link_libraries(sirius_tests
    PRIVATE
        sirius
        Catch2::Catch2WithMain
        TIFF::TIFF
)

target_compile_definitions(sirius_tests
    PRIVATE
        SIRIUS_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data"
)

include(Catch)
catch_discover_tests(sirius_tests TEST_PREFIX "sirius::")

if(SIRIUS_ENABLE_WARNINGS)
    target_set_warnings(sirius_tests)
endif()

if(SIRIUS_ENABLE_SANITIZERS)
    target_enable_sanitizers(sirius_tests)
endif()