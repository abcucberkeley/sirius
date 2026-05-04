cmake_minimum_required(VERSION 3.24)

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

project(SIRIUS
    VERSION 0.1.0
    DESCRIPTION "Structured Illumination Reconstruction and Image Utility Suite"
    LANGUAGES C CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ProjectOptions)
include(Warnings)
include(Sanitizers)
include(StaticAnalysis)
include(Dependencies)

add_subdirectory(src)

if(SIRIUS_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SIRIUS_ENABLE_PYTHON_BINDINGS)
    add_subdirectory(bindings)
endif()