# Install and export rules: make a built SIRIUS consumable from another project
# with
#     find_package(SIRIUS CONFIG REQUIRED)
#     target_link_libraries(myapp PRIVATE sirius::sirius)
#
# Why this file looks the way it does
# -----------------------------------
# `sirius` is a STATIC library, so every library it links -- PRIVATE ones
# included -- still has to be on the link line of whoever links sirius, and
# every target named in its usage interface has to exist again in the
# consumer's project. Almost all of those come from FetchContent
# (cmake/Dependencies.cmake): Eigen, zlib, libdeflate, zstd, libjpeg-turbo,
# libtiff, FFTW, toml++, nlohmann/json. They are built here, pinned to exact revisions, and are not
# packages a consumer could find_package() -- requiring a downstream user to
# install the same six projects, at the same versions, with the same options
# would defeat the point of the fetch.
#
# So the install is *self-contained*: the static archives SIRIUS was linked
# against are installed next to libsirius and exported as part of SIRIUSTargets,
# and Eigen -- the one fetched dependency that leaks into the public headers
# (<sirius/buffer.hpp> includes <unsupported/Eigen/CXX11/Tensor>) -- is bundled
# as headers under include/sirius-vendor/eigen3. toml++ and nlohmann/json are
# header-only and used only inside .cpp files, so they disappear at the library
# boundary and are neither installed nor exported.
#
# What a consumer must still supply is exactly what SIRIUS itself took from the
# system rather than from FetchContent: OpenMP, and for a CUDA build the CUDA
# toolkit. SIRIUSConfig.cmake find_dependency()s those.

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Where the package config lands; also the anchor the config file resolves the
# rest of the prefix from.
set(SIRIUS_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/SIRIUS")
# The vendored third-party archives go in a subdirectory of libdir so they can
# never be mistaken for (or collide with) a system libtiff/zlib/fftw3.
set(SIRIUS_INSTALL_VENDOR_LIBDIR "${CMAKE_INSTALL_LIBDIR}/sirius")
# Bundled Eigen: a sibling of include/sirius, never inside it, so that
# `#include <sirius/...>` and `#include <Eigen/...>` stay separate roots.
set(SIRIUS_INSTALL_VENDOR_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/sirius-vendor/eigen3")

# ---------------------------------------------------------------------------
# The public dependency: Eigen (header-only, in the public API)
# ---------------------------------------------------------------------------
# sirius_eigen is declared in cmake/Dependencies.cmake with only a
# BUILD_INTERFACE include directory; give it the installed one as well so the
# exported sirius::sirius keeps compiling in a consumer's tree.
target_include_directories(sirius_eigen SYSTEM INTERFACE
    $<INSTALL_INTERFACE:${SIRIUS_INSTALL_VENDOR_INCLUDEDIR}>)

if(NOT eigen3_SOURCE_DIR)
    message(FATAL_ERROR "SIRIUS_ENABLE_INSTALL needs the fetched Eigen sources to bundle; "
                        "eigen3_SOURCE_DIR is empty (see cmake/Dependencies.cmake).")
endif()
install(DIRECTORY "${eigen3_SOURCE_DIR}/Eigen"
        DESTINATION "${SIRIUS_INSTALL_VENDOR_INCLUDEDIR}"
        COMPONENT sirius_Development
        PATTERN "CMakeLists.txt" EXCLUDE)
install(DIRECTORY "${eigen3_SOURCE_DIR}/unsupported/Eigen"
        DESTINATION "${SIRIUS_INSTALL_VENDOR_INCLUDEDIR}/unsupported"
        COMPONENT sirius_Development
        PATTERN "CMakeLists.txt" EXCLUDE)
install(FILES "${eigen3_SOURCE_DIR}/COPYING.MPL2" "${eigen3_SOURCE_DIR}/README.md"
        DESTINATION "${CMAKE_INSTALL_DOCDIR}/vendor/eigen3"
        COMPONENT sirius_Development)

# ---------------------------------------------------------------------------
# The private dependencies: the static archives sirius was linked against
# ---------------------------------------------------------------------------
# They carry no headers into the install tree -- a consumer never includes
# tiffio.h or fftw3.h through a SIRIUS header -- only the archives themselves,
# because a static sirius.lib cannot resolve their symbols on its own.
set(SIRIUS_VENDORED_LIB_TARGETS tiff zlibstatic libdeflate_static libzstd_static sirius_jpeg fftw3)
if(TARGET fftw3_omp)
    list(APPEND SIRIUS_VENDORED_LIB_TARGETS fftw3_omp)
endif()

# Archives only: no SIRIUS header includes libdeflate.h.
set_target_properties(libdeflate_static PROPERTIES PUBLIC_HEADER "")

# libjpeg-turbo is an ExternalProject (cmake/Dependencies.cmake), so its
# archive is not a target install(TARGETS) knows: sirius_jpeg, the INTERFACE
# target that carries it, links the build-tree archive; install the archive of
# the configuration being installed under one name and point the installed
# sirius_jpeg at it.
install(FILES "${SIRIUS_JPEG_LIB}"
        DESTINATION "${SIRIUS_INSTALL_VENDOR_LIBDIR}"
        RENAME "${SIRIUS_JPEG_LIB_NAME}"
        COMPONENT sirius_Development)
target_link_libraries(sirius_jpeg INTERFACE
    "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${SIRIUS_INSTALL_VENDOR_LIBDIR}/${SIRIUS_JPEG_LIB_NAME}>")

# zlib's CMakeLists puts bare absolute build- and source-tree paths on
# zlibstatic's usage interface (target_include_directories(... PUBLIC
# ${CMAKE_CURRENT_BINARY_DIR} ...)), and install(EXPORT) rightly refuses to
# write those into a package. Nothing outside the build needs them -- no SIRIUS
# header includes zlib.h, tiffio.h or fftw3.h -- so re-wrap every bare path as
# BUILD_INTERFACE; the build itself is unaffected.
foreach(_dep IN LISTS SIRIUS_VENDORED_LIB_TARGETS)
    get_target_property(_dirs ${_dep} INTERFACE_INCLUDE_DIRECTORIES)
    if(_dirs)
        set(_wrapped "")
        foreach(_dir IN LISTS _dirs)
            if(_dir MATCHES "^\$<")
                list(APPEND _wrapped "${_dir}")
            else()
                list(APPEND _wrapped "$<BUILD_INTERFACE:${_dir}>")
            endif()
        endforeach()
        set_target_properties(${_dep} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_wrapped}")
    endif()
endforeach()

# ---------------------------------------------------------------------------
# The export set
# ---------------------------------------------------------------------------
# sirius_simd (compile flags) and sirius_eigen are INTERFACE targets with no
# artifacts, but they are named in sirius's usage interface, so CMake requires
# them in an export set too.
install(TARGETS sirius sirius_simd sirius_eigen
        EXPORT SIRIUSTargets
        FILE_SET HEADERS DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
            COMPONENT sirius_Development
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT sirius_Development
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT sirius_Runtime
            NAMELINK_COMPONENT sirius_Development
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT sirius_Runtime)

install(TARGETS ${SIRIUS_VENDORED_LIB_TARGETS}
        EXPORT SIRIUSTargets
        ARCHIVE DESTINATION "${SIRIUS_INSTALL_VENDOR_LIBDIR}" COMPONENT sirius_Development
        LIBRARY DESTINATION "${SIRIUS_INSTALL_VENDOR_LIBDIR}" COMPONENT sirius_Runtime
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT sirius_Runtime)

install(EXPORT SIRIUSTargets
        FILE SIRIUSTargets.cmake
        NAMESPACE sirius::
        DESTINATION "${SIRIUS_INSTALL_CMAKEDIR}"
        COMPONENT sirius_Development)

# ---------------------------------------------------------------------------
# The package config
# ---------------------------------------------------------------------------
# Recorded in the config so a consumer can query how this copy was built
# (`if(SIRIUS_WITH_CUDA)`) and so the config only re-finds what was actually used.
# Keep the versions in step with the pins in cmake/Dependencies.cmake.
set(SIRIUS_VENDORED_EIGEN_VERSION "3.4.0")
set(SIRIUS_VENDORED_SUMMARY
    "Eigen 3.4.0 (headers), libtiff 4.7.0, zlib 1.3.1, libdeflate 1.26, zstd 1.5.7, libjpeg-turbo 3.1.4.1, FFTW 3.3.10")

configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/SIRIUSConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SIRIUSConfig.cmake"
    INSTALL_DESTINATION "${SIRIUS_INSTALL_CMAKEDIR}"
    PATH_VARS SIRIUS_INSTALL_VENDOR_INCLUDEDIR SIRIUS_INSTALL_VENDOR_LIBDIR)

# 0.x: every minor release may break the API, so require an exact minor match.
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SIRIUSConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMinorVersion)

install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/SIRIUSConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/SIRIUSConfigVersion.cmake"
        DESTINATION "${SIRIUS_INSTALL_CMAKEDIR}"
        COMPONENT sirius_Development)

# The licences of the vendored archives (each asks binary redistributions
# to carry its notice).
FetchContent_GetProperties(libtiff SOURCE_DIR _sirius_lic_tiff)
FetchContent_GetProperties(zlib SOURCE_DIR _sirius_lic_zlib)
FetchContent_GetProperties(libdeflate SOURCE_DIR _sirius_lic_deflate)
FetchContent_GetProperties(zstd SOURCE_DIR _sirius_lic_zstd)
FetchContent_GetProperties(fftw3 SOURCE_DIR _sirius_lic_fftw3)
foreach(_lic IN ITEMS "libtiff|${_sirius_lic_tiff}/LICENSE.md"
                      "zlib|${_sirius_lic_zlib}/LICENSE"
                      "libdeflate|${_sirius_lic_deflate}/COPYING"
                      "zstd|${_sirius_lic_zstd}/LICENSE"
                      "libjpeg-turbo|${SIRIUS_JPEG_SOURCE_DIR}/LICENSE.md"
                      "libjpeg-turbo|${SIRIUS_JPEG_SOURCE_DIR}/README.ijg"
                      "fftw3|${_sirius_lic_fftw3}/COPYING")
    string(REPLACE "|" ";" _lic "${_lic}")
    list(GET _lic 0 _name)
    list(GET _lic 1 _file)
    install(FILES "${_file}" DESTINATION "${CMAKE_INSTALL_DOCDIR}/vendor/${_name}" COMPONENT sirius_Development)
endforeach()

install(FILES "${PROJECT_SOURCE_DIR}/LICENSE"
        DESTINATION "${CMAKE_INSTALL_DOCDIR}"
        COMPONENT sirius_Development)
