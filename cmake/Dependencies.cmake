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

# libtiff
FetchContent_Declare(
    libtiff
    GIT_REPOSITORY https://gitlab.com/libtiff/libtiff.git
    GIT_TAG        v4.7.0
    GIT_SHALLOW    TRUE
)
set(tiff-tools   OFF)
set(tiff-tests   OFF)
set(tiff-contrib OFF)
set(tiff-docs    OFF)
FetchContent_MakeAvailable(libtiff)
if(NOT TARGET TIFF::TIFF)
    add_library(TIFF::TIFF ALIAS tiff)
endif()

# FFTW3
FetchContent_Declare(
    fftw3
    URL https://www.fftw.org/fftw-3.3.10.tar.gz
)
# fftw using offensive global names
block()
    # FFTW3 uses cmake_minimum_required(3.0), so CMP0077 defaults OLD and option()
    # ignores normal variables; NEW makes it honor our BUILD_SHARED_LIBS=OFF below.
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_TESTS OFF) # Build tests
    set(ENABLE_OPENMP  ON) # Use OpenMP for multithreading
    set(ENABLE_THREADS OFF) # Use pthread for multithreading
    set(ENABLE_FLOAT OFF) # single-precision
    set(ENABLE_LONG_DOUBLE OFF) # long-double precision
    set(ENABLE_QUAD_PRECISION OFF) # quadruple-precision
    set(ENABLE_SSE OFF) # Compile with SSE instruction set support
    set(ENABLE_SSE2 OFF) # Compile with SSE2 instruction set support
    set(ENABLE_AVX OFF) # Compile with AVX instruction set support
    set(ENABLE_AVX2 OFF) # Compile with AVX2 instruction set support
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