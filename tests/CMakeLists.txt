add_executable(sirius_tests)

target_sources(sirius_tests
    PRIVATE
        test_tiff_io.cpp
        test_fft1d.cpp
        test_fft2d.cpp
        test_fft3d.cpp
)

target_link_libraries(sirius_tests
    PRIVATE
        sirius
        Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(sirius_tests DISCOVERY_MODE PRE_TEST)