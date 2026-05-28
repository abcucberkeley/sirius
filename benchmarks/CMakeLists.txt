# C++ TIFF read benchmarks. Gated by SIRIUS_ENABLE_BENCHMARKS at the top level.
#
# Only the sirius bench is built here. The cpp-tiff bench is built standalone by
# setup_cpptiff.sh, because cpp-tiff bundles its own static libtiff under a
# target also named `tiff` -- adding it to this build would collide with the
# `tiff` target SIRIUS pulls in via FetchContent and risk duplicate-libtiff ODR.

add_executable(bench_tiff_sirius bench_tiff_sirius.cpp)

target_include_directories(bench_tiff_sirius
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(bench_tiff_sirius
    PRIVATE
        sirius
        OpenMP::OpenMP_CXX
)

if(SIRIUS_ENABLE_WARNINGS)
    target_set_warnings(bench_tiff_sirius)
endif()

if(SIRIUS_ENABLE_SANITIZERS)
    target_enable_sanitizers(bench_tiff_sirius)
endif()
