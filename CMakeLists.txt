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
include(RuntimeDeps)
include(Dependencies)

add_subdirectory(src)

# before tests: the app's GUI-free core library is unit-tested there. The core
# and sirius-cli need no window, so SIRIUS_ENABLE_CLI builds them without the GUI.
if(SIRIUS_ENABLE_APP OR SIRIUS_ENABLE_CLI)
    add_subdirectory(app)
endif()

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

# Last: the export set describes targets defined above (and the vendored
# dependencies they link).
if(SIRIUS_ENABLE_INSTALL)
    include(Install)
endif()
