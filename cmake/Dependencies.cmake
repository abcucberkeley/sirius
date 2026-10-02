include(FetchContent)

# Every dependency is pinned to an immutable revision: git dependencies to the
# commit a release tag pointed at when it was recorded (the tag is kept as a
# comment; `git ls-remote <repo> refs/tags/<tag>^{}` resolves a new one) and
# tarballs to their SHA-256. A shallow clone only holds the branch tips and the
# tagged commits, so GIT_SHALLOW works for a tagged commit alone: a commit no
# tag points at is fetched as GitHub's tarball of that commit
# (https://github.com/<owner>/<repo>/archive/<sha>.tar.gz) with its SHA-256, or
# cloned in full (nanobind).

# Static deps must be PIC-compatible when linked into the Python extension (.so)
if(SIRIUS_ENABLE_PYTHON_BINDINGS)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Eigen3 (header-only). Its CMake project is deliberately NOT added:
# Eigen 3.4's CMakeLists probes OpenGL, Python and -- through the legacy
# FindCUDA module in unsupported/test -- the system CUDA libraries, which
# pre-seeds CUDA_cufft_LIBRARY & co. in the cache with whatever distro toolkit
# lives in /usr/lib before our FindCUDAToolkit runs. SOURCE_SUBDIR points at a
# directory without a CMakeLists.txt, so FetchContent only downloads the
# sources and we describe the target ourselves.
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3147391d946bb4b6c68edd901f2add6ac1f31f8c   # 3.4.0
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  cmake-not-used
)
FetchContent_MakeAvailable(Eigen3)
add_library(sirius_eigen INTERFACE)
# SYSTEM: excluded from warnings and MSVC /analyze, like the other deps.
target_include_directories(sirius_eigen SYSTEM INTERFACE
    $<BUILD_INTERFACE:${eigen3_SOURCE_DIR}>)
add_library(Eigen3::Eigen ALIAS sirius_eigen)

# zlib — provides the DEFLATE/ZIP codec for libtiff. Without it, libtiff's
# internal find_package(ZLIB) fails and ZIP_SUPPORT is left undefined, so
# writing a TIFF with TiffCompression::Deflate fails at encode time.
# OVERRIDE_FIND_PACKAGE redirects that find_package(ZLIB) to this fetched copy.
FetchContent_Declare(
    ZLIB
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG        51b7f2abdade71cd9bb0e7a373ef2610ec6f9daf   # v1.3.1
    GIT_SHALLOW    TRUE
    OVERRIDE_FIND_PACKAGE
)
block()
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)  # zlib targets an old CMake floor
    set(ZLIB_BUILD_EXAMPLES OFF)
    # zlib bakes the install prefix into cache variables at configure time, so
    # its install rules ignore `cmake --install --prefix` and try to write to
    # C:/Program Files. SIRIUS installs the archive itself (cmake/Install.cmake).
    set(SKIP_INSTALL_ALL ON)
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

# --- libtiff's codecs beyond zlib ---------------------------------------------
# libdeflate (faster Deflate decoding; libtiff still needs zlib beside it),
# zstd (ZSTD-compressed TIFF) and libjpeg-turbo (JPEG / YCbCr TIFF, 8- and
# 12-bit). All three are built here, static, and handed to libtiff through
# the find_package() redirects directory (written below, before libtiff is
# configured): libtiff's find_package(Deflate / ZSTD / JPEG) and TensorStore's
# find_package(Zstd / JPEG) then resolve to these targets instead of
# searching the system, so every build -- the app, sirius-cli, the Python
# wheel, a cluster image -- has the same codecs and needs no system package.

# libdeflate
FetchContent_Declare(
    libdeflate
    GIT_REPOSITORY https://github.com/ebiggers/libdeflate.git
    GIT_TAG        92e6a0db9fa848d742f9eb286c92afc60f2c3dda   # v1.26
    GIT_SHALLOW    TRUE
    SYSTEM
)
block()
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)   # its option() calls honour the variables set here
    set(LIBDEFLATE_BUILD_STATIC_LIB ON)
    set(LIBDEFLATE_BUILD_SHARED_LIB OFF)
    set(LIBDEFLATE_BUILD_GZIP OFF)
    set(LIBDEFLATE_BUILD_TESTS OFF)
    set(LIBDEFLATE_INSTALL OFF)   # SIRIUS installs the archive itself (cmake/Install.cmake)
    FetchContent_MakeAvailable(libdeflate)
endblock()
add_library(Deflate::Deflate ALIAS libdeflate_static)

# zstd. One copy for the whole build: TensorStore's bridge (zarr / N5 blosc
# and zstd codecs) is pointed at this one too (TENSORSTORE_USE_SYSTEM_ZSTD
# below), so a TensorStore build never links two.
FetchContent_Declare(
    zstd
    GIT_REPOSITORY https://github.com/facebook/zstd.git
    GIT_TAG        f8745da6ff1ad1e7bab384bd1f9d742439278e99   # v1.5.7
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  build/cmake
    SYSTEM
)
block()
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(ZSTD_BUILD_STATIC ON)
    set(ZSTD_BUILD_SHARED OFF)
    set(ZSTD_BUILD_PROGRAMS OFF)
    set(ZSTD_BUILD_TESTS OFF)
    set(ZSTD_BUILD_CONTRIB OFF)
    set(ZSTD_LEGACY_SUPPORT OFF)       # pre-1.0 frames: no TIFF writer makes them
    set(ZSTD_MULTITHREAD_SUPPORT OFF)  # libtiff and TensorStore call it from their own threads
    # zstd has no switch for its install rules; send them into a vendor
    # subdirectory like FFTW's (the archive SIRIUS's consumers link is
    # installed by cmake/Install.cmake).
    set(CMAKE_INSTALL_INCLUDEDIR share/sirius-vendor/zstd/include)
    set(CMAKE_INSTALL_LIBDIR     share/sirius-vendor/zstd/lib)
    set(CMAKE_INSTALL_BINDIR     share/sirius-vendor/zstd/bin)
    FetchContent_MakeAvailable(zstd)
endblock()
add_library(ZSTD::ZSTD ALIAS libzstd_static)   # the name libtiff links
add_library(Zstd::Zstd ALIAS libzstd_static)   # the name TensorStore's bridge links

# libjpeg-turbo. Its CMake project refuses to be added with add_subdirectory()
# (upstream policy), so it is built as an ExternalProject into a private
# prefix at build time and described here as an INTERFACE target that carries
# the archive, its headers and the build-order dependency. The sub-build uses
# the same generator and compiler; a multi-config generator builds the
# configuration being built (Debug archives get a "d" postfix so Debug and
# Release coexist in the prefix). The SIMD code needs NASM on x86 (found on
# PATH or given as CMAKE_ASM_NASM_COMPILER); without it libjpeg-turbo builds
# its portable C code, which decodes to the same pixels.
include(ExternalProject)
set(SIRIUS_JPEG_PREFIX "${CMAKE_BINARY_DIR}/_deps/libjpeg-turbo-install")
set(SIRIUS_JPEG_SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/libjpeg-turbo-src")
if(MSVC OR CMAKE_C_SIMULATE_ID STREQUAL "MSVC")
    set(_sirius_jpeg_base jpeg-static)   # libjpeg-turbo's archive name for MSVC-like compilers
else()
    set(_sirius_jpeg_base jpeg)
endif()
set(SIRIUS_JPEG_LIB_NAME
    "${CMAKE_STATIC_LIBRARY_PREFIX}${_sirius_jpeg_base}${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(SIRIUS_JPEG_LIB_NAME_DEBUG
    "${CMAKE_STATIC_LIBRARY_PREFIX}${_sirius_jpeg_base}d${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(SIRIUS_JPEG_LIB
    "${SIRIUS_JPEG_PREFIX}/lib/$<IF:$<CONFIG:Debug>,${SIRIUS_JPEG_LIB_NAME_DEBUG},${SIRIUS_JPEG_LIB_NAME}>")
block()
    get_property(_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    set(_cache_args
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_DEBUG_POSTFIX:STRING=d)
    if(_multi)
        set(_byproducts "${SIRIUS_JPEG_PREFIX}/lib/${SIRIUS_JPEG_LIB_NAME}"
                        "${SIRIUS_JPEG_PREFIX}/lib/${SIRIUS_JPEG_LIB_NAME_DEBUG}")
    else()
        set(_type "${CMAKE_BUILD_TYPE}")
        if(NOT _type)
            set(_type Release)
        endif()
        list(APPEND _cache_args -DCMAKE_BUILD_TYPE:STRING=${_type})
        string(TOUPPER "${_type}" _type_uc)
        if(_type_uc STREQUAL "DEBUG")
            set(_byproducts "${SIRIUS_JPEG_PREFIX}/lib/${SIRIUS_JPEG_LIB_NAME_DEBUG}")
        else()
            set(_byproducts "${SIRIUS_JPEG_PREFIX}/lib/${SIRIUS_JPEG_LIB_NAME}")
        endif()
    endif()
    if(NOT CMAKE_GENERATOR MATCHES "Visual Studio")
        list(APPEND _cache_args -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER})
    endif()
    foreach(_var CMAKE_TOOLCHAIN_FILE CMAKE_OSX_ARCHITECTURES CMAKE_OSX_DEPLOYMENT_TARGET CMAKE_OSX_SYSROOT
                 CMAKE_C_COMPILER_LAUNCHER)
        if(DEFINED ${_var} AND NOT "${${_var}}" STREQUAL "")
            list(APPEND _cache_args "-D${_var}:STRING=${${_var}}")
        endif()
    endforeach()
    list(LENGTH CMAKE_OSX_ARCHITECTURES _archs)
    if(_archs GREATER 1)
        list(APPEND _cache_args -DWITH_SIMD:BOOL=OFF)   # SIMD code is per architecture
    else()
        if(DEFINED CMAKE_ASM_NASM_COMPILER)
            set(SIRIUS_JPEG_NASM "${CMAKE_ASM_NASM_COMPILER}")
        else()
            find_program(SIRIUS_JPEG_NASM NAMES nasm yasm)
        endif()
        if(SIRIUS_JPEG_NASM)
            list(APPEND _cache_args -DCMAKE_ASM_NASM_COMPILER:FILEPATH=${SIRIUS_JPEG_NASM})
        endif()
    endif()
    ExternalProject_Add(sirius_libjpeg_turbo_build
        URL         https://github.com/libjpeg-turbo/libjpeg-turbo/archive/9217719d3a58633923b096af4c1d50d304768a64.tar.gz   # 3.1.4.1
        URL_HASH    SHA256=7cf1ad7af15e3c4bf0694334f9682d862f81d8cacb93f8daf37fd9e5adbe5984
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        PREFIX      "${CMAKE_BINARY_DIR}/_deps/libjpeg-turbo"
        SOURCE_DIR  "${SIRIUS_JPEG_SOURCE_DIR}"
        BINARY_DIR  "${CMAKE_BINARY_DIR}/_deps/libjpeg-turbo-build"
        INSTALL_DIR "${SIRIUS_JPEG_PREFIX}"
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DCMAKE_INSTALL_LIBDIR=lib
            -DCMAKE_INSTALL_INCLUDEDIR=include
            -DENABLE_SHARED=OFF
            -DENABLE_STATIC=ON
            -DWITH_TURBOJPEG=OFF
            -DWITH_TOOLS=OFF
            -DWITH_TESTS=OFF
            -DWITH_CRT_DLL=ON          # MSVC: the DLL run-time, as the rest of SIRIUS
        CMAKE_CACHE_ARGS ${_cache_args}
        BUILD_BYPRODUCTS ${_byproducts}
        LOG_CONFIGURE TRUE
        LOG_BUILD TRUE
        LOG_INSTALL TRUE
        LOG_OUTPUT_ON_FAILURE TRUE)
endblock()
file(MAKE_DIRECTORY "${SIRIUS_JPEG_PREFIX}/include")   # an include directory must exist at generate time
add_library(sirius_jpeg INTERFACE)
add_dependencies(sirius_jpeg sirius_libjpeg_turbo_build)
target_include_directories(sirius_jpeg SYSTEM INTERFACE $<BUILD_INTERFACE:${SIRIUS_JPEG_PREFIX}/include>)
target_link_libraries(sirius_jpeg INTERFACE $<BUILD_INTERFACE:${SIRIUS_JPEG_LIB}>)
add_library(JPEG::JPEG ALIAS sirius_jpeg)   # the name libtiff and TensorStore's bridge link

# The redirects: find_package(<name>) anywhere in the build -- libtiff's
# Deflate / ZSTD / JPEG, TensorStore's Zstd / JPEG -- finds the targets above.
# (File names are lower-case: one file answers both ZSTD and Zstd.)
file(WRITE "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/deflate-config.cmake"
     "# libdeflate is built in-tree by SIRIUS (cmake/Dependencies.cmake); Deflate::Deflate exists.\n"
     "set(Deflate_FOUND TRUE)\n"
     "set(Deflate_VERSION_STRING 1.26)\n"
     "set(Deflate_LIBRARY Deflate::Deflate)\n"
     "set(Deflate_LIBRARIES Deflate::Deflate)\n"
     "set(Deflate_INCLUDE_DIRS \"\")\n")
file(WRITE "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/zstd-config.cmake"
     "# zstd is built in-tree by SIRIUS (cmake/Dependencies.cmake); ZSTD::ZSTD and Zstd::Zstd exist.\n"
     "set(ZSTD_FOUND TRUE)\n"
     "set(Zstd_FOUND TRUE)\n"
     "set(ZSTD_VERSION 1.5.7)\n"
     "set(ZSTD_LIBRARIES ZSTD::ZSTD)\n"
     "set(ZSTD_INCLUDE_DIRS \"\")\n"
     "set(ZSTD_HAVE_DECOMPRESS_STREAM ON)   # libtiff would otherwise link-test a library not built yet\n")
file(WRITE "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/jpeg-config.cmake"
     "# libjpeg-turbo is built by SIRIUS (cmake/Dependencies.cmake); JPEG::JPEG exists.\n"
     "set(JPEG_FOUND TRUE)\n"
     "set(JPEG_VERSION 62)\n"
     "set(JPEG_LIBRARIES JPEG::JPEG)\n"
     "set(JPEG_LIBRARY JPEG::JPEG)\n"
     "set(JPEG_INCLUDE_DIRS \"${SIRIUS_JPEG_PREFIX}/include\")\n"
     "set(JPEG_INCLUDE_DIR \"${SIRIUS_JPEG_PREFIX}/include\")\n")

# libtiff
FetchContent_Declare(
    libtiff
    GIT_REPOSITORY https://gitlab.com/libtiff/libtiff.git
    GIT_TAG        9dff73bebc5661f2dace6f16e14cf9e857172f4e   # v4.7.0
    GIT_SHALLOW    TRUE
    # 12-bit JPEG rows with an odd sample count lose their last sample
    # upstream; see the patch script.
    PATCH_COMMAND  ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_LIST_DIR}/patches/fix_libtiff_jpeg12_odd.cmake
)
# compatibility with cmake < 3.5 has been removed from CMake
block()
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    set(tiff-install OFF)   # SIRIUS installs the archive itself (cmake/Install.cmake)
    set(tiff-tools   OFF)
    set(tiff-tests   OFF)
    set(tiff-contrib OFF)
    set(tiff-docs    OFF)
    # The codecs, forced: a cache written before these libraries were
    # fetched holds them OFF, and option() never revisits a cached value.
    # Codecs SIRIUS does not build (LZMA, WebP, JBIG, LERC) stay off rather
    # than following whatever the build machine happens to have installed.
    set(zlib       ON  CACHE BOOL "libtiff: zlib (Deflate)" FORCE)
    set(libdeflate ON  CACHE BOOL "libtiff: libdeflate (fast Deflate)" FORCE)
    set(zstd       ON  CACHE BOOL "libtiff: zstd" FORCE)
    set(jpeg       ON  CACHE BOOL "libtiff: libjpeg-turbo (JPEG)" FORCE)
    set(old-jpeg   ON  CACHE BOOL "libtiff: old-style JPEG (read-only)" FORCE)
    set(lzma       OFF CACHE BOOL "libtiff: LZMA (not built by SIRIUS)" FORCE)
    set(webp       OFF CACHE BOOL "libtiff: WebP (not built by SIRIUS)" FORCE)
    set(jbig       OFF CACHE BOOL "libtiff: JBIG (not built by SIRIUS)" FORCE)
    set(lerc       OFF CACHE BOOL "libtiff: LERC (not built by SIRIUS)" FORCE)
    # libjpeg-turbo >= 3 decodes 8- and 12-bit JPEG in one library; libtiff
    # would find that out by linking a test program against an archive that
    # is only built later.
    set(HAVE_JPEGTURBO_DUAL_MODE_8_12 1 CACHE INTERNAL "libjpeg-turbo 3 has the jpeg12_* API")
    FetchContent_MakeAvailable(libtiff)
    if(NOT TARGET TIFF::TIFF)
        add_library(TIFF::TIFF ALIAS tiff)
    endif()
endblock()
add_dependencies(tiff sirius_libjpeg_turbo_build)
FetchContent_GetProperties(libtiff SOURCE_DIR _sirius_tiff_src)
foreach(_codec ZIP_SUPPORT LIBDEFLATE_SUPPORT ZSTD_SUPPORT JPEG_SUPPORT)
    # set in libtiff's directory scope
    get_directory_property(_on DIRECTORY "${_sirius_tiff_src}" DEFINITION ${_codec})
    if(NOT _on)
        message(FATAL_ERROR "libtiff was configured without ${_codec}; see the codec block in cmake/Dependencies.cmake")
    endif()
endforeach()

# FFTW3
FetchContent_Declare(
    fftw3
    URL      https://www.fftw.org/fftw-3.3.10.tar.gz
    URL_HASH SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467
)
# fftw using offensive global names
block()
    # FFTW 3.3.10 has no switch to turn its own install rules off, and they
    # would drop fftw3.h, a second copy of the archive, a pkg-config file and an
    # FFTW3Config.cmake at the top of SIRIUS's install prefix. Redirect them
    # into a vendor subdirectory (the copy SIRIUS's consumers link is installed
    # by cmake/Install.cmake); these are normal variables, so they shadow
    # GNUInstallDirs' cache entries for this subtree only.
    set(CMAKE_INSTALL_INCLUDEDIR share/sirius-vendor/fftw3/include)
    set(CMAKE_INSTALL_LIBDIR     share/sirius-vendor/fftw3/lib)
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
    # if omp is enabled, create fftw3_omp alias for the target
    if(TARGET fftw3_omp)
        add_library(FFTW3::fftw3_omp ALIAS fftw3_omp)
    endif()
endblock()

# Canonical FFTW link targets: the core double-precision lib, plus the OpenMP
# threading lib when FFTW was built with ENABLE_OPENMP. Consumers link
# ${SIRIUS_FFTW_TARGETS} rather than repeating this conditional.
set(SIRIUS_FFTW_TARGETS FFTW3::fftw3)
if(TARGET FFTW3::fftw3_omp)
    list(APPEND SIRIUS_FFTW_TARGETS FFTW3::fftw3_omp)
endif()

# OpenMP (provided by the host compiler)
find_package(OpenMP REQUIRED)

# Threads: the app core's child processes and worker connections, sirius-cli
# and the tests all use Threads::Threads, so it is found once, here, for every
# directory.
find_package(Threads REQUIRED)

# toml++ : TOML parser/serializer
FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        30172438cee64926dc41fdd9c11fb3ba5b2ba9de   # v3.4.0
    GIT_SHALLOW    TRUE
    SYSTEM          # mark its headers as system -> excluded from warnings/analyze
)
FetchContent_MakeAvailable(tomlplusplus)

# nlohmann/json: pipeline files, the assistant tool API and the worker
# protocol (header-only; TensorStore uses the same library internally).
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    SYSTEM
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

# TensorStore (zarr / N5). Its CMake bridge normally fetches private copies of
# zlib, zstd, libjpeg-turbo, libtiff and nlohmann/json under the same target
# names we already define (ZLIB::ZLIB, Zstd::Zstd, JPEG::JPEG, TIFF::TIFF,
# nlohmann_json::nlohmann_json), and two zlibs or zstds in one static link
# would clash anyway. So those five are declared "system" packages for
# TensorStore and find_package() is redirected to the targets built above;
# everything else (abseil, blosc, riegeli, ...) is fetched and built by the
# bridge, out of sight.
#
# The system's libcurl for sirius-app (Linux distributions ship it with their
# TLS library and its CA store) is looked for first: the bridge registers its
# own bundled curl, built without a CA store, as the answer to every later
# find_package(CURL). When there is a system libcurl, TensorStore shares it.
if(SIRIUS_ENABLE_APP AND NOT WIN32)
    if(SIRIUS_ENABLE_TENSORSTORE)
        # FindCURL's own search, here and in the bridge, not a libcurl's
        # CURLConfig.cmake: one built with CMake loads the system OpenSSL
        # (find_dependency) as OpenSSL::SSL and OpenSSL::Crypto, the names the
        # bridge gives its BoringSSL (third_party/boringssl/workspace.bzl).
        set(CURL_NO_CURL_CMAKE ON)
    endif()
    find_package(CURL QUIET)
endif()
if(SIRIUS_ENABLE_TENSORSTORE)
    # The bridge enables the ASM_NASM language (libjpeg-turbo / BoringSSL).
    # Look where package managers put nasm when it is not on PATH; conda-forge's
    # `conda install nasm` is the no-sudo route on a shared machine.
    if(NOT DEFINED CMAKE_ASM_NASM_COMPILER AND NOT DEFINED ENV{ASM_NASM})
        find_program(SIRIUS_NASM NAMES nasm
            HINTS ENV CONDA_PREFIX "$ENV{HOME}/miniconda3" "$ENV{HOME}/anaconda3" "$ENV{HOME}/mambaforge"
                  "$ENV{HOME}/miniforge3" /opt/conda /usr/local
            PATH_SUFFIXES bin)
        if(SIRIUS_NASM)
            set(CMAKE_ASM_NASM_COMPILER "${SIRIUS_NASM}" CACHE FILEPATH "NASM assembler for TensorStore's dependencies" FORCE)
        else()
            message(FATAL_ERROR "SIRIUS_ENABLE_TENSORSTORE needs the NASM assembler (apt install nasm, "
                                "conda install -c conda-forge nasm, or set CMAKE_ASM_NASM_COMPILER).")
        endif()
    endif()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)   # bazel_to_cmake

    set(TENSORSTORE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(TENSORSTORE_USE_SYSTEM_ZLIB ON CACHE BOOL "" FORCE)
    set(TENSORSTORE_USE_SYSTEM_TIFF ON CACHE BOOL "" FORCE)
    set(TENSORSTORE_USE_SYSTEM_NLOHMANN_JSON ON CACHE BOOL "" FORCE)
    # zstd and libjpeg-turbo too: built above for libtiff, and a second copy
    # from the bridge would put every ZSTD_* / jpeg_* symbol in the link twice
    # (and a second JPEG::JPEG target in the project).
    set(TENSORSTORE_USE_SYSTEM_ZSTD ON CACHE BOOL "" FORCE)
    set(TENSORSTORE_USE_SYSTEM_JPEG ON CACHE BOOL "" FORCE)
    if(CURL_FOUND)
        set(TENSORSTORE_USE_SYSTEM_CURL ON CACHE BOOL "" FORCE)
    endif()
    # find_package(TIFF) / find_package(nlohmann_json) inside the bridge must
    # resolve to our targets: drop config files into the redirects directory
    # CMake consults before any module or installed package (zlib already has
    # one from OVERRIDE_FIND_PACKAGE above).
    foreach(_pkg TIFF nlohmann_json)
        string(TOLOWER "${_pkg}" _lc)
        file(WRITE "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/${_lc}-config.cmake"
             "# ${_pkg} is built in-tree by SIRIUS (cmake/Dependencies.cmake); the targets already exist.\n"
             "set(${_pkg}_FOUND TRUE)\n")
        file(WRITE "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/${_pkg}Config.cmake"
             "include(\"\${CMAKE_CURRENT_LIST_DIR}/${_lc}-config.cmake\")\n")
    endforeach()
    set(TIFF_LIBRARIES TIFF::TIFF)
    set(TIFF_INCLUDE_DIRS "")

    FetchContent_Declare(
        tensorstore
        URL      https://github.com/google/tensorstore/archive/refs/tags/v${SIRIUS_TENSORSTORE_VERSION}.tar.gz
        URL_HASH SHA256=f59667a32357b8cc0c752429927ad97654f6a67c7d3d62b9efea902c6798d473
        SYSTEM
    )
    FetchContent_MakeAvailable(tensorstore)
    # The drivers the library uses. all_drivers would also pull gcs/s3/http.
    add_library(sirius_tensorstore INTERFACE)
    target_link_libraries(sirius_tensorstore INTERFACE
        tensorstore::tensorstore
        tensorstore::cast
        tensorstore::index_space_dim_expression
        tensorstore::driver_cast
        tensorstore::driver_zarr
        tensorstore::driver_zarr3
        tensorstore::driver_n5
        tensorstore::kvstore_file)
    add_library(sirius::tensorstore ALIAS sirius_tensorstore)
    message(STATUS "TensorStore ${SIRIUS_TENSORSTORE_VERSION} (zarr v2/v3, N5) enabled")
endif()

if(SIRIUS_ENABLE_TESTS)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        fa43b77429ba76c462b1898d6cd2f2d7a9416b14   # v3.7.1
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

# stb (a commit of master, as a tarball): PNG in and out for the GUI's icons
# and screenshots, and PNG / JPEG out for the app core's renders, which
# sirius-cli hands to scripts and agents -- so it is fetched for either
# executable. Header-only; the one source file of each that holds an
# implementation defines it there.
if(SIRIUS_ENABLE_APP OR SIRIUS_ENABLE_CLI)
    FetchContent_Declare(
        stb
        URL      https://github.com/nothings/stb/archive/2c980bb59875b0d32144a71867fbdebb2f77cd20.tar.gz
        URL_HASH SHA256=9a955b1b49a4410088a2e0ee2a9c057c3c907d0c1d75454144cb980aca0ba515
    )
    FetchContent_MakeAvailable(stb)
    add_library(sirius_stb INTERFACE)
    # SYSTEM: excluded from warnings and MSVC /analyze, like the other deps.
    target_include_directories(sirius_stb SYSTEM INTERFACE ${stb_SOURCE_DIR})
endif()

# The application (app/imgui). Everything it needs is small enough to fetch
# and build in-tree, pinned like the rest: GLFW for the window and the
# OpenGL context, Dear ImGui (docking branch: dockable, floatable panels) with
# its GLFW and OpenGL 3 backends, ImPlot for the diagnostics charts, a text
# editor widget for the plugin files, native file dialogs, and libcurl for
# the assistant and the model hub (stb is fetched above). Dear ImGui, ImPlot
# and the editor ship no CMake project, so their targets are described here.
if(SIRIUS_ENABLE_APP)
    find_package(OpenGL REQUIRED)

    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        a74efa0d5628b74adc0426af4c5710e287fa7c2c   # 3.4
        GIT_SHALLOW    TRUE
        SYSTEM
    )
    block()
        set(BUILD_SHARED_LIBS OFF)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(glfw)
    endblock()
    FetchContent_GetProperties(glfw SOURCE_DIR glfw_SOURCE_DIR)   # set inside the block above

    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        b48d1afbe8ee8b238e2961dc363a949dd7304e23   # v1.92.9b-docking
        GIT_SHALLOW    TRUE
    )
    # These two have no release tags: commits of master, as tarballs.
    FetchContent_Declare(
        implot
        URL      https://github.com/epezent/implot/archive/09e2ba71766e25d88053a2173936c9d1043bae42.tar.gz   # 1.92-compatible
        URL_HASH SHA256=17dd3b860237cb95c0d1c0c6accaf92ad4ea4be61add84eca85e3dc077963e12
    )
    FetchContent_Declare(
        imgui_text_editor
        URL      https://github.com/goossens/ImGuiColorTextEdit/archive/133614b0d5e1008527a26f46a93fdb1d751ca115.tar.gz
        URL_HASH SHA256=69e419617763720da3b9619b4d3090f5efc7982e08281a2ecce42f76ddd4bc70
        SOURCE_SUBDIR cmake-not-used
    )
    FetchContent_MakeAvailable(imgui implot imgui_text_editor)

    add_library(sirius_imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
        ${implot_SOURCE_DIR}/implot.cpp
        ${implot_SOURCE_DIR}/implot_items.cpp
        ${imgui_text_editor_SOURCE_DIR}/TextEditor.cpp)
    # SYSTEM: excluded from warnings and MSVC /analyze, like the other deps.
    target_include_directories(sirius_imgui SYSTEM PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
        ${imgui_SOURCE_DIR}/misc/cpp
        ${implot_SOURCE_DIR}
        ${imgui_text_editor_SOURCE_DIR}
        ${stb_SOURCE_DIR}
        # glad's single-header OpenGL 3.3 loader, as GLFW's own examples use it
        ${glfw_SOURCE_DIR}/deps)
    target_compile_features(sirius_imgui PUBLIC cxx_std_17)
    # 32-bit ImWchar: with the default 16 bits the text editor decodes every
    # character outside the Basic Multilingual Plane (emoji, CJK extension B)
    # as U+FFFD and writes that back on save, and typed ones are dropped.
    # PUBLIC, since every file that includes imgui.h must agree on the type.
    target_compile_definitions(sirius_imgui PUBLIC IMGUI_USE_WCHAR32)
    if(MSVC)
        # third-party sources: not ours to analyse (cmake/StaticAnalysis.cmake)
        target_compile_options(sirius_imgui PRIVATE /analyze- /w)
    endif()
    target_link_libraries(sirius_imgui PUBLIC glfw OpenGL::GL)
    add_library(sirius::imgui ALIAS sirius_imgui)

    FetchContent_Declare(
        nfd
        GIT_REPOSITORY https://github.com/btzy/nativefiledialog-extended.git
        GIT_TAG        86d5f2005fe1c00747348a12070fec493ea2407e   # v1.2.1
        GIT_SHALLOW    TRUE
        SYSTEM
    )
    block()
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)   # its option() calls honour the variables set here
        set(BUILD_SHARED_LIBS OFF)
        set(NFD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(NFD_INSTALL OFF CACHE BOOL "" FORCE)
        # Linux: GTK 3's dialog where its development package is installed,
        # since it works wherever the window does (ssh -X, VNC, a bare window
        # manager). Otherwise xdg-desktop-portal over D-Bus, which needs no GTK
        # to build but a running portal with a FileChooser backend to open.
        if(UNIX AND NOT APPLE)
            find_package(PkgConfig QUIET)
            if(PkgConfig_FOUND)
                pkg_check_modules(SIRIUS_GTK3 QUIET gtk+-3.0)
            endif()
            if(SIRIUS_GTK3_FOUND)
                set(NFD_PORTAL OFF CACHE BOOL "" FORCE)
            else()
                set(NFD_PORTAL ON CACHE BOOL "" FORCE)
            endif()
        endif()
        FetchContent_MakeAvailable(nfd)
    endblock()

    # libcurl: the system's where there is one (looked for above, before
    # TensorStore); built in-tree otherwise, against the platform's TLS
    # (Schannel on Windows), with everything but HTTP(S) turned off.
    if(NOT CURL_FOUND)
        FetchContent_Declare(
            curl
            GIT_REPOSITORY https://github.com/curl/curl.git
            GIT_TAG        8c908d2d0a6d32abdedda2c52e90bd56ec76c24d   # curl-8_19_0
            GIT_SHALLOW    TRUE
            SYSTEM
        )
        block()
            set(BUILD_SHARED_LIBS OFF)
            set(BUILD_CURL_EXE OFF CACHE BOOL "" FORCE)
            set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
            set(BUILD_TESTING OFF)
            set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
            set(BUILD_LIBCURL_DOCS OFF CACHE BOOL "" FORCE)
            set(BUILD_MISC_DOCS OFF CACHE BOOL "" FORCE)
            set(ENABLE_CURL_MANUAL OFF CACHE BOOL "" FORCE)
            set(CURL_DISABLE_INSTALL ON CACHE BOOL "" FORCE)
            set(HTTP_ONLY ON CACHE BOOL "" FORCE)
            set(CURL_USE_LIBPSL OFF CACHE BOOL "" FORCE)
            set(CURL_USE_LIBSSH2 OFF CACHE BOOL "" FORCE)
            set(USE_LIBIDN2 OFF CACHE BOOL "" FORCE)
            set(USE_NGHTTP2 OFF CACHE BOOL "" FORCE)
            set(CURL_BROTLI OFF CACHE BOOL "" FORCE)
            set(CURL_ZSTD OFF CACHE BOOL "" FORCE)
            set(CURL_ZLIB OFF CACHE BOOL "" FORCE)
            if(WIN32)
                set(CURL_USE_SCHANNEL ON CACHE BOOL "" FORCE)
                set(CURL_USE_OPENSSL OFF CACHE BOOL "" FORCE)
            endif()
            FetchContent_MakeAvailable(curl)
        endblock()
    endif()
    message(STATUS "Dear ImGui (docking), ImPlot, GLFW and libcurl for sirius-app")
endif()

if(SIRIUS_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "SIRIUS_ENABLE_CUDA=ON but no CUDA compiler was found. "
                            "Set CUDACXX or CMAKE_CUDA_COMPILER to nvcc.")
    endif()
    enable_language(CUDA)
    # Pin the toolkit to the one nvcc came from. Without this FindCUDAToolkit
    # can pick libraries of a second, distro-packaged toolkit that sits in the
    # default linker paths (/usr/lib/x86_64-linux-gnu on Ubuntu), and the
    # binary ends up with cuFFT/cudart from a different CUDA major than the
    # compiler that built the kernels.
    if(NOT DEFINED CUDAToolkit_ROOT)
        get_filename_component(_sirius_cuda_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
        get_filename_component(CUDAToolkit_ROOT "${_sirius_cuda_bin}/.." ABSOLUTE)
    endif()
    find_package(CUDAToolkit 12.0 REQUIRED)
    message(STATUS "CUDA toolkit ${CUDAToolkit_VERSION} (nvcc ${CMAKE_CUDA_COMPILER}), "
                   "architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    # Every CUDA library we link must come from that same toolkit.
    foreach(_lib cufft cudart_static)
        file(REAL_PATH "${CUDA_${_lib}_LIBRARY}" _sirius_lib_real)
        file(REAL_PATH "${CUDAToolkit_LIBRARY_ROOT}" _sirius_root_real)
        string(FIND "${_sirius_lib_real}" "${_sirius_root_real}/" _sirius_pos)
        if(NOT _sirius_pos EQUAL 0)
            message(FATAL_ERROR "CUDA::${_lib} resolved to ${CUDA_${_lib}_LIBRARY}, outside the toolkit "
                                "${CUDAToolkit_LIBRARY_ROOT} that provides nvcc. Clear CUDA_*_LIBRARY cache "
                                "entries (cmake -U 'CUDA_*_LIBRARY') or set CUDAToolkit_ROOT.")
        endif()
    endforeach()
    if(SIRIUS_ENABLE_NVTIFF)
        include(NvidiaRedist)
    endif()
endif()