add_executable(sirius_tests)

target_sources(sirius_tests
    PRIVATE
        test_tiff_io.cpp
)

target_link_libraries(sirius_tests
    PRIVATE
        sirius
        Catch2::Catch2WithMain
        TIFF::TIFF
)

include(Catch)
catch_discover_tests(sirius_tests DISCOVERY_MODE PRE_TEST)