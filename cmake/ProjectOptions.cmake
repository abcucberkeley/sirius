option(SIRIUS_ENABLE_MPI "Enable MPI" OFF)
option(SIRIUS_ENABLE_CUDA "Enable CUDA (device buffers, cuFFT, nvTIFF)" OFF)
option(SIRIUS_ENABLE_PYTHON_BINDINGS "Enable nanobind python bindings" OFF)
# The workbench application (app/): Dear ImGui over GLFW and OpenGL 3.3, with
# ImPlot and friends, all fetched and built in-tree (cmake/Dependencies.cmake).
option(SIRIUS_ENABLE_APP "Build the sirius-app GUI (fetches GLFW, Dear ImGui, ImPlot, libcurl)" OFF)
# sirius-cli (app/cli): the workbench core without a window, for scripts and agents
# (command line, JSON-lines session, MCP server). Always built with SIRIUS_ENABLE_APP;
# ON builds it on its own, e.g. on a cluster node without X11 / Wayland.
option(SIRIUS_ENABLE_CLI "Build sirius-cli (always built when SIRIUS_ENABLE_APP is ON)" OFF)
include(CMakeDependentOption)

# nvTIFF decodes TIFF strips/tiles straight into device memory. It is an NVIDIA
# redistributable (no source), fetched from developer.download.nvidia.com by
# cmake/Dependencies.cmake, or taken from SIRIUS_NVTIFF_ROOT when set (e.g. a
# cluster module). Deflate/ZIP decoding additionally needs nvCOMP at runtime.
cmake_dependent_option(SIRIUS_ENABLE_NVTIFF "Enable GPU TIFF decoding via nvTIFF" ON
                       "SIRIUS_ENABLE_CUDA" OFF)
cmake_dependent_option(SIRIUS_ENABLE_NVCOMP "Fetch nvCOMP so nvTIFF can decode Deflate/ZIP TIFFs on the GPU" ON
                       "SIRIUS_ENABLE_NVTIFF" OFF)
set(SIRIUS_NVTIFF_ROOT "" CACHE PATH "Existing nvTIFF install (include/ and lib/) to use instead of downloading")
set(SIRIUS_NVCOMP_ROOT "" CACHE PATH "Existing nvCOMP install (include/ and lib/) to use instead of downloading")

# TensorStore gives the library (and the workbench's Load / Export) zarr v2,
# zarr v3 and N5 stores. It is a Bazel project built through its CMake bridge
# (bazel_to_cmake), which fetches ~40 dependencies and needs NASM and a
# python3 at configure time; the first build takes several minutes and about
# 1.5 GB. Off by default so the plain library/CI builds stay light; the
# *-app-* presets turn it on. See cmake/Dependencies.cmake for the wiring.
option(SIRIUS_ENABLE_TENSORSTORE "Enable zarr / N5 stores through TensorStore (long first build, needs nasm)" OFF)
set(SIRIUS_TENSORSTORE_VERSION "0.1.78" CACHE STRING "TensorStore release to fetch")

# scikit-build-core always builds the Python extension
if(SKBUILD)
    set(SIRIUS_ENABLE_PYTHON_BINDINGS ON CACHE BOOL "" FORCE)
endif()
option(SIRIUS_ENABLE_SSE2   "Enable SSE2 instruction set"    OFF)
option(SIRIUS_ENABLE_AVX    "Enable AVX instruction set"     OFF)
option(SIRIUS_ENABLE_AVX2   "Enable AVX2 + FMA instruction sets" OFF)
option(SIRIUS_ENABLE_AVX512 "Enable AVX-512F + FMA instruction sets" OFF)

# Install / export rules for `find_package(SIRIUS CONFIG)` (cmake/Install.cmake).
# The install tree ships the fetched static dependencies alongside the library;
# nvTIFF, nvCOMP and TensorStore are prebuilt redistributables / a Bazel build
# that SIRIUS does not install, so those builds cannot produce a usable package.
cmake_dependent_option(SIRIUS_ENABLE_INSTALL "Generate install and export rules for the sirius library" ON
                       "PROJECT_IS_TOP_LEVEL;NOT SIRIUS_ENABLE_NVTIFF;NOT SIRIUS_ENABLE_NVCOMP;NOT SIRIUS_ENABLE_TENSORSTORE" OFF)

# Development related options
option(SIRIUS_ENABLE_TESTS "Enable tests" OFF)
option(SIRIUS_ENABLE_BENCHMARKS "Build C++ benchmarks" OFF)
option(SIRIUS_ENABLE_WARNINGS "Enable extra warnings" OFF)
# Warnings become errors on SIRIUS's own targets (the library, the app core;
# never the FetchContent dependencies). The Linux GCC CI job turns it on.
option(SIRIUS_WARNINGS_AS_ERRORS "Treat warnings as errors on the sirius targets (needs SIRIUS_ENABLE_WARNINGS)" OFF)
option(SIRIUS_ENABLE_SANITIZERS "Enable sanitizers (Debug, non-MSVC)" OFF)
option(SIRIUS_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
option(SIRIUS_ENABLE_CPPCHECK "Enable cppcheck" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(SIRIUS_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    # Distro-packaged nvcc (/usr/bin/nvcc) is frequently older than the host
    # compiler supports. Prefer the toolkit a module or the CUDA installer put
    # in CUDA_HOME / CUDA_PATH / /usr/local/cuda, unless the user pinned one.
    if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
        find_program(_sirius_nvcc NAMES nvcc
            HINTS ENV CUDA_HOME ENV CUDA_PATH ENV CUDA_ROOT /usr/local/cuda
            PATH_SUFFIXES bin
            NO_DEFAULT_PATH)
        if(_sirius_nvcc)
            set(CMAKE_CUDA_COMPILER "${_sirius_nvcc}" CACHE FILEPATH "CUDA compiler" FORCE)
        endif()
    endif()
    # "native" = the GPUs in this machine (dev builds). Release presets pass an
    # explicit list so the binary runs on the cluster's cards too.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
endif()

# Symlink the build-dir compile_commands.json for IDE integration
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Generate compile_commands.json" FORCE)
    # Silently ignore failures (e.g. Windows without Developer Mode enabled)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            "${CMAKE_BINARY_DIR}/compile_commands.json"
            "${CMAKE_SOURCE_DIR}/compile_commands.json"
        RESULT_VARIABLE _symlink_result
        ERROR_QUIET
        OUTPUT_QUIET
    )
endif()
