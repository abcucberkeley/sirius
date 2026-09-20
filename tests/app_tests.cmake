# Tests of the workbench core (app/core), one executable per unit like the
# library's. Included from tests/CMakeLists.txt when SIRIUS_ENABLE_APP is on.
# INTERIM: one binary against the archive until app/core is split into units.
add_library(test_app_core_objects OBJECT
    test_app_core.cpp
    test_app_labels.cpp
    test_app_lru.cpp
    test_app_tracking.cpp
    test_app_tracks.cpp
    test_app_training.cpp
    test_app_pipeline.cpp
    test_app_ops.cpp
    test_app_io.cpp
    test_app_rpc.cpp
    test_app_help.cpp
    test_app_manifest.cpp
    test_app_schema.cpp
    test_app_parity.cpp)
target_link_libraries(test_app_core_objects PUBLIC sirius::app_core sirius Catch2::Catch2 TIFF::TIFF)
target_compile_definitions(test_app_core_objects PRIVATE SIRIUS_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data")
set_property(GLOBAL APPEND PROPERTY SIRIUS_TEST_OBJECTS test_app_core_objects)
add_executable(test_app_core)
target_link_libraries(test_app_core PRIVATE test_app_core_objects Catch2::Catch2WithMain)
sirius_finish_test_executable(test_app_core)
catch_discover_tests(test_app_core TEST_PREFIX "sirius::")
