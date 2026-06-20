cmake_minimum_required(VERSION 3.25)

project(SIRIUS
    VERSION 0.1.0
    DESCRIPTION "Structured Illumination Reconstruction and Image Utility Suite"
    LANGUAGES C CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ProjectOptions)
include(SIMD)
include(Warnings)
include(Sanitizers)
include(StaticAnalysis)
include(Dependencies)

add_subdirectory(src)

if(SIRIUS_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SIRIUS_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(SIRIUS_ENABLE_PYTHON_BINDINGS)
    add_subdirectory(bindings)
endif()