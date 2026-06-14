include(FetchContent)

# Static deps must be PIC-compatible when linked into the Python extension (.so)
if(SIRIUS_ENABLE_PYTHON_BINDINGS)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Eigen3
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
set(EIGEN_BUILD_DOC     OFF)
set(EIGEN_BUILD_TESTING OFF)
FetchContent_MakeAvailable(Eigen3)

# zlib — provides the DEFLATE/ZIP codec for libtiff. Without it, libtiff's
# internal find_package(ZLIB) fails and ZIP_SUPPORT is left undefined, so
# writing a TIFF with TiffCompression::Deflate fails at encode time.
# OVERRIDE_FIND_PACKAGE redirects that find_package(ZLIB) to this fetched copy.
FetchContent_Declare(
    ZLIB
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG        v1.3.1
    GIT_SHALLOW    TRUE
    OVERRIDE_FIND_PACKAGE
)
block()
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)  # zlib targets an old CMake floor
    set(ZLIB_BUILD_EXAMPLES OFF)
    FetchContent_MakeAvailable(ZLIB)
    # zlib's CMake exports zlibstatic/zlib but not the canonical ZLIB::ZLIB
    # target libtiff links against. Create it from the static lib (PIC is on
    # for the Python extension) and ensure its headers are on the usage
    # interface: zlib.h lives in the source tree, generated zconf.h in the
    # build tree.
    if(NOT TARGET ZLIB::ZLIB)
        target_include_directories(zlibstatic PUBLIC
            $<BUILD_INTERFACE:${zlib_SOURCE_DIR}>
            $<BUILD_INTERFACE:${zlib_BINARY_DIR}>)
        add_library(ZLIB::ZLIB ALIAS zlibstatic)
    endif()
endblock()

# libtiff
FetchContent_Declare(
    libtiff
    GIT_REPOSITORY https://gitlab.com/libtiff/libtiff.git
    GIT_TAG        v4.7.0
    GIT_SHALLOW    TRUE
)
# compatibility with cmake < 3.5 has been removed from CMake
block()
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    set(tiff-tools   OFF)
    set(tiff-tests   OFF)
    set(tiff-contrib OFF)
    set(tiff-docs    OFF)
    FetchContent_MakeAvailable(libtiff)
    if(NOT TARGET TIFF::TIFF)
        add_library(TIFF::TIFF ALIAS tiff)
    endif()
endblock()

# FFTW3
FetchContent_Declare(
    fftw3
    URL https://www.fftw.org/fftw-3.3.10.tar.gz
)
# fftw using offensive global names
block()
    # FFTW 3.3.10 declares cmake_minimum_required(VERSION 3.0); CMake 4.x
    # removed support for <3.5, so spoof a 3.5 floor for this subtree only.
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    # FFTW3 uses cmake_minimum_required(3.0), so CMP0077 defaults OLD and option()
    # ignores normal variables; NEW makes it honor our BUILD_SHARED_LIBS=OFF below.
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_TESTS OFF) # Build tests
    set(ENABLE_OPENMP  ON) # Use OpenMP for multithreading
    set(ENABLE_THREADS OFF) # Use pthread for multithreading
    set(ENABLE_FLOAT OFF) # single-precision (unused; sirius uses double fftw_* API)
    set(ENABLE_LONG_DOUBLE OFF) # long-double precision
    set(ENABLE_QUAD_PRECISION OFF) # quadruple-precision
    set(ENABLE_SSE OFF)
    set(ENABLE_SSE2  ${SIRIUS_ENABLE_SSE2})
    set(ENABLE_AVX   ${SIRIUS_ENABLE_AVX})
    set(ENABLE_AVX2  ${SIRIUS_ENABLE_AVX2})
    set(ENABLE_AVX512 ${SIRIUS_ENABLE_AVX512})
    FetchContent_MakeAvailable(fftw3)
    target_include_directories(fftw3 PUBLIC $<BUILD_INTERFACE:${fftw3_SOURCE_DIR}/api>)
    add_library(FFTW3::fftw3 ALIAS fftw3)
endblock()

# OpenMP (provided by the host compiler)
find_package(OpenMP REQUIRED)

if(SIRIUS_ENABLE_TESTS)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.7.1
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(Catch2)
endif()

if(SIRIUS_ENABLE_PYTHON_BINDINGS)
    find_package(Python 3.9
        REQUIRED COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG        a835245fa0c8f6c8d06a25713562100464e95039
        # Fix an upstream MSVC build error in the Eigen::Tensor caster
        # (std::array<long, N> vs Eigen::Index). See the patch script for details.
        PATCH_COMMAND  ${CMAKE_COMMAND} -P
                       ${CMAKE_CURRENT_LIST_DIR}/patches/fix_nanobind_tensor.cmake
    )
    FetchContent_MakeAvailable(nanobind)
endif()

if(SIRIUS_ENABLE_MPI)
    find_package(MPI REQUIRED)
endif()

if(SIRIUS_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()