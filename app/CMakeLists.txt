# The SIRIUS workbench (docs/design/README.md). Gated by SIRIUS_ENABLE_APP at
# the top level; Qt is located in cmake/Dependencies.cmake as sirius::qt.
#
#   sirius_app_core  static, Qt-free: array model, operations, pipeline,
#                    executor, workbench, tool API, worker protocol, export.
#                    Built as a graph of units (cmake/Units.cmake), each with
#                    a test executable of its own in tests/app_tests.cmake.
#   sirius-app       the Qt Widgets executable (app/qt/**), globbed
#                    (CONFIGURE_DEPENDS) so adding a panel is adding a file.

include(Units)

# --- core -------------------------------------------------------------------
#
# One unit per header: its sources, and the units it is allowed to use. They
# are declared in dependency order, which is what keeps the graph acyclic, and
# `sirius_app_core` at the bottom is still the single archive the application,
# the tests and the install link.

# Where an installed tree keeps help/, python/ and plugins/, relative to the
# executable's directory (core/app_paths.cpp): computed from the full install
# directories so a --prefix given at install time moves both together.
include(GNUInstallDirs)
set(SIRIUS_APP_INSTALL_DATADIR "${CMAKE_INSTALL_DATADIR}/sirius")
file(RELATIVE_PATH _sirius_app_datadir_from_bindir
    "${CMAKE_INSTALL_FULL_BINDIR}" "${CMAKE_INSTALL_FULL_DATADIR}/sirius")

# sirius_core_unit(<name> [HEADERS ...] [SOURCES ...] [DEPENDS ...] [LINK ...] [PUBLIC_LINK ...] [DEFINES ...])
# HEADERS and SOURCES are relative to app/core; DEPENDS names core units, or
# library ones as lib/<unit>. PUBLIC_LINK is for a library the unit's *headers*
# include -- nlohmann/json, whose json_fwd.hpp several of them declare
# parameters with -- and LINK for one only its sources use.
function(sirius_core_unit name)
    cmake_parse_arguments(PARSE_ARGV 1 C "" "" "HEADERS;SOURCES;DEPENDS;LINK;PUBLIC_LINK;DEFINES")
    list(TRANSFORM C_HEADERS PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/core/)
    list(TRANSFORM C_SOURCES PREPEND core/)
    sirius_add_unit(core ${name}
        HEADERS ${C_HEADERS}
        SOURCES ${C_SOURCES}
        DEPENDS ${C_DEPENDS}
        LINK ${C_LINK}
        PUBLIC_LINK ${C_PUBLIC_LINK}
        DEFINES ${C_DEFINES}
        INCLUDE_PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    if(C_SOURCES)
        # Library-wide, as when this was one target: OpenMP (Eigen reads
        # _OPENMP, so a unit without a pragma still has to agree).
        target_link_libraries(sirius_core_${name} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()

# op(<name> <source> [DEPENDS ...]): one built-in operation, which always has
# the registry's declaration of its factory and the shared helpers.
function(op name source)
    cmake_parse_arguments(PARSE_ARGV 2 O "" "" "DEPENDS")
    sirius_core_unit(ops_${name} SOURCES ${source} DEPENDS ops_factories ops_common ${O_DEPENDS})
    set(SIRIUS_APP_OPS ${SIRIUS_APP_OPS} ops_${name} PARENT_SCOPE)
endfunction()

# --- vocabulary --------------------------------------------------------------

sirius_core_unit(cancel            HEADERS cancel.hpp)
sirius_core_unit(errors            HEADERS errors.hpp            DEPENDS lib/errors)
sirius_core_unit(byte_budget_lru   HEADERS byte_budget_lru.hpp)
sirius_core_unit(label_frames      HEADERS label_frames.hpp      DEPENDS lib/index)

sirius_core_unit(app_paths
    HEADERS app_paths.hpp
    SOURCES app_paths.cpp
    DEFINES SIRIUS_APP_DATADIR_FROM_BINDIR="${_sirius_app_datadir_from_bindir}")

sirius_core_unit(history      HEADERS history.hpp      SOURCES history.cpp)
sirius_core_unit(session_log  HEADERS session_log.hpp  SOURCES session_log.cpp  PUBLIC_LINK nlohmann_json::nlohmann_json)
sirius_core_unit(params       HEADERS params.hpp       SOURCES params.cpp       PUBLIC_LINK nlohmann_json::nlohmann_json)
# The checkout's help/ and python/ are where a binary built from this tree
# looks first (app_paths.hpp), so the two units that own those directories are
# told where the tree is, as app_paths is told where an install puts them.
sirius_core_unit(help_pages
    HEADERS help_pages.hpp
    SOURCES help_pages.cpp
    DEPENDS app_paths
    DEFINES SIRIUS_APP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# --- data model ----------------------------------------------------------------

sirius_core_unit(array            HEADERS array.hpp            SOURCES array.cpp            DEPENDS lib/buffer lib/checked_math)
sirius_core_unit(dataset          HEADERS dataset.hpp          SOURCES dataset.cpp          DEPENDS array lib/pixel_type)
sirius_core_unit(display_mapping  HEADERS display_mapping.hpp  SOURCES display_mapping.cpp  DEPENDS lib/buffer)
sirius_core_unit(diagnostics      HEADERS diagnostics.hpp      SOURCES diagnostics.cpp      DEPENDS lib/buffer lib/fft)

# The index is below the volume that owns one: it reads frames and diffs
# (label_frames.hpp), not LabelVolume, so the two are not a cycle.
sirius_core_unit(tracks    HEADERS tracks.hpp    SOURCES tracks.cpp    DEPENDS label_frames PUBLIC_LINK nlohmann_json::nlohmann_json)
sirius_core_unit(labels    HEADERS labels.hpp    SOURCES labels.cpp    DEPENDS array label_frames tracks lib/buffer)
sirius_core_unit(tracking  HEADERS tracking.hpp  SOURCES tracking.cpp  DEPENDS labels)

sirius_core_unit(manifest
    HEADERS manifest.hpp
    SOURCES manifest.cpp
    DEPENDS dataset lib/tiff_io
    PUBLIC_LINK nlohmann_json::nlohmann_json   # manifest.hpp declares with json_fwd
    LINK tomlplusplus::tomlplusplus)

sirius_core_unit(array_source
    HEADERS array_source.hpp
    SOURCES array_source.cpp
    DEPENDS array dataset manifest lib/device lib/tiff_io lib/zarr_io)

sirius_core_unit(export
    HEADERS export.hpp
    SOURCES export.cpp
    DEPENDS array dataset labels cancel lib/tiff_io lib/image_ops lib/zarr_io
    LINK nlohmann_json::nlohmann_json)

sirius_core_unit(training_export
    HEADERS training_export.hpp
    SOURCES training_export.cpp
    DEPENDS array dataset export labels cancel lib/tiff_io
    PUBLIC_LINK nlohmann_json::nlohmann_json)

# --- the Python worker ------------------------------------------------------------

sirius_core_unit(rpc
    HEADERS rpc.hpp
    SOURCES rpc.cpp
    DEPENDS app_paths cancel errors lib/buffer lib/checked_math
    DEFINES SIRIUS_APP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    PUBLIC_LINK nlohmann_json::nlohmann_json)
if(WIN32)
    target_link_libraries(sirius_core_rpc PRIVATE ws2_32)
endif()

# --- steps and the pipeline ---------------------------------------------------------

sirius_core_unit(operation
    HEADERS operation.hpp
    SOURCES operation.cpp
    DEPENDS array dataset diagnostics labels params array_source cancel lib/device
    PUBLIC_LINK nlohmann_json::nlohmann_json)

sirius_core_unit(pipeline
    HEADERS pipeline.hpp
    SOURCES pipeline.cpp
    DEPENDS operation params
    PUBLIC_LINK nlohmann_json::nlohmann_json
    LINK tomlplusplus::tomlplusplus)

sirius_core_unit(executor
    HEADERS executor.hpp
    SOURCES executor.cpp
    DEPENDS operation pipeline array_source cancel
    LINK nlohmann_json::nlohmann_json)

# --- SIM reconstruction, as the workbench drives it -------------------------------------

sirius_core_unit(session
    HEADERS session.hpp
    SOURCES session.cpp
    DEPENDS lib/sim_reconstruction lib/otf lib/otf_io lib/otf_ideal lib/legacy_config lib/tiff_io
            lib/sim_parameters lib/buffer lib/device lib/fft_common lib/constants)

sirius_core_unit(volume_ops
    HEADERS volume_ops.hpp
    SOURCES volume_ops.cpp
    DEPENDS lib/buffer lib/otf lib/sim_parameters lib/sim_reconstruction lib/fft lib/constants)

# --- the operations --------------------------------------------------------------------
# Every operation is a unit of its own, so what it needs is what it links: the
# stitch step has the mosaic code in it and the deskew step does not. They are
# leaves -- nothing includes an operation -- and ops/builtin.hpp, which only
# declares their factories, is the header-only unit they share.

sirius_core_unit(ops_factories  HEADERS ops/builtin.hpp  DEPENDS operation)

sirius_core_unit(ops_common
    HEADERS ops/common.hpp
    SOURCES ops/common.cpp
    DEPENDS operation array_source lib/image_ops)

sirius_core_unit(ops_contrast_api
    HEADERS ops/contrast.hpp
    DEPENDS operation)

sirius_core_unit(ops_load_api
    HEADERS ops/load.hpp
    DEPENDS array_source params)

sirius_core_unit(ops_sim_params
    HEADERS ops/sim_params.hpp
    DEPENDS dataset params lib/sim_parameters)

sirius_core_unit(ops_segment_common
    HEADERS ops/segment_common.hpp
    SOURCES ops/segment_common.cpp
    DEPENDS labels)

sirius_core_unit(ops_torch_model
    HEADERS ops/torch_model.hpp
    SOURCES ops/torch_model.cpp
    DEPENDS rpc ops_common
    PUBLIC_LINK nlohmann_json::nlohmann_json)

# op(<name> <source> [DEPENDS ...]): one built-in operation.
op(load        ops/load.cpp        DEPENDS ops_load_api array_source manifest)
op(sim         ops/sim.cpp         DEPENDS ops_sim_params session volume_ops lib/legacy_config lib/device lib/constants)
op(decon       ops/decon.cpp       DEPENDS lib/deconvolution lib/tiff_io)
op(volrec      ops/volrec.cpp      DEPENDS lib/image_ops)
op(einsum      ops/einsum.cpp      DEPENDS lib/image_ops)
op(contrast    ops/contrast.cpp    DEPENDS ops_contrast_api array_source lib/image_ops)
op(flatfield   ops/flatfield.cpp   DEPENDS lib/image_ops lib/tiff_io)
op(bleach      ops/bleach.cpp      DEPENDS lib/image_ops)
op(deskew      ops/deskew.cpp      DEPENDS lib/image_ops)
op(croppad     ops/croppad.cpp     DEPENDS lib/image_ops)
op(resample    ops/resample.cpp    DEPENDS lib/image_ops)
op(merge       ops/merge.cpp       DEPENDS lib/image_ops)
op(stitch      ops/stitch.cpp      DEPENDS array_source lib/stitching lib/stitching_tiff)
op(register    ops/register.cpp    DEPENDS lib/image_ops lib/registration)
op(seg         ops/seg.cpp         DEPENDS ops_segment_common ops_torch_model rpc)
op(foundation  ops/foundation.cpp  DEPENDS tracks rpc)
op(threshold   ops/threshold.cpp   DEPENDS ops_segment_common lib/image_ops)
op(classic     ops/classic.cpp     DEPENDS ops_segment_common lib/image_ops lib/constants)
op(skimage_seg ops/skimage_seg.cpp DEPENDS rpc)
op(cleanup     ops/cleanup.cpp     DEPENDS labels)
op(track       ops/track.cpp       DEPENDS tracking tracks rpc)

# The list of them, and registerBuiltinOperations() that walks it: the one
# place that depends on every operation.
sirius_core_unit(ops_registry
    SOURCES ops/builtin.cpp ops/builtin_list.cpp
    DEPENDS ops_factories ${SIRIUS_APP_OPS})

# The parameter specs of every built-in as JSON: above the registry, since
# naming every one of them is what it is for.
sirius_core_unit(ops_schema
    HEADERS ops/schema.hpp
    SOURCES ops/schema.cpp
    DEPENDS ops_registry ops_factories
    PUBLIC_LINK nlohmann_json::nlohmann_json)

# A user operation the Python worker serves, and the plugin files behind it.
sirius_core_unit(plugin
    HEADERS ops/plugin.hpp
    SOURCES ops/plugin.cpp
    DEPENDS operation rpc executor help_pages array_source ops_common
    PUBLIC_LINK nlohmann_json::nlohmann_json)

# --- the workbench -----------------------------------------------------------------

sirius_core_unit(workbench
    HEADERS workbench.hpp
    SOURCES workbench.cpp
    DEPENDS array_source dataset executor history labels operation pipeline rpc session_log tracks
            cancel ops_registry ops_factories ops_load_api plugin
    PUBLIC_LINK nlohmann_json::nlohmann_json)

sirius_core_unit(tool_api
    HEADERS tool_api.hpp
    SOURCES tool_api.cpp
    DEPENDS workbench training_export
    PUBLIC_LINK nlohmann_json::nlohmann_json)

# --- the archive -------------------------------------------------------------

add_library(sirius_app_core STATIC)
add_library(sirius::app_core ALIAS sirius_app_core)
sirius_archive_units(sirius_app_core core)
set_target_properties(sirius_app_core PROPERTIES LINKER_LANGUAGE CXX)   # objects, no sources to tell by
target_include_directories(sirius_app_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# The units compile; the archive is what consumers link, so the libraries the
# objects need are named once more here, for them.
target_link_libraries(sirius_app_core
    PUBLIC sirius nlohmann_json::nlohmann_json
    PRIVATE OpenMP::OpenMP_CXX TIFF::TIFF tomlplusplus::tomlplusplus)
if(WIN32)
    target_link_libraries(sirius_app_core PRIVATE ws2_32)
endif()
if(SIRIUS_ENABLE_SANITIZERS)
    target_enable_sanitizers(sirius_app_core)
endif()

# --- GUI --------------------------------------------------------------------
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

file(GLOB_RECURSE SIRIUS_APP_QT_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/qt/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/qt/*.hpp)
add_executable(sirius-app WIN32 MACOSX_BUNDLE
    ${SIRIUS_APP_QT_SOURCES}
    qt/resources/resources.qrc)
target_include_directories(sirius-app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sirius-app PRIVATE sirius::app_core sirius::qt)
# The process code page is UTF-8 (see the manifest): the core opens files by
# their UTF-8 bytes through the narrow Win32 APIs. The linker merges .manifest
# sources into the executable's manifest.
if(MSVC)
    target_sources(sirius-app PRIVATE ${CMAKE_SOURCE_DIR}/cmake/utf8-codepage.manifest)
endif()
# crypt32: DPAPI (CryptProtectData) behind qt/secret_store.cpp
if(WIN32)
    target_link_libraries(sirius-app PRIVATE crypt32)
endif()
if(SIRIUS_ENABLE_WARNINGS)
    target_set_warnings(sirius-app)
endif()
if(SIRIUS_ENABLE_SANITIZERS)
    target_enable_sanitizers(sirius-app)
endif()

# Help pages and the Python worker live next to the executable so users can
# edit the pages and launch the worker from the build tree.
add_custom_command(TARGET sirius-app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/help $<TARGET_FILE_DIR:sirius-app>/help
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/python $<TARGET_FILE_DIR:sirius-app>/python
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/plugins $<TARGET_FILE_DIR:sirius-app>/plugins
    COMMENT "Copying help pages, the Python worker and the example plugins next to sirius-app"
    VERBATIM)

# Runtime DLLs on Windows: the NVIDIA libraries (as for the tests and
# benchmarks) and the Qt libraries/plugins via windeployqt, so the executable
# starts from the build tree without PATH changes. SIRIUS_APP_DEPLOY_QT opts
# out of the Qt part (see ProjectOptions.cmake).
sirius_copy_runtime_dlls(sirius-app)
if(SIRIUS_APP_DEPLOY_QT)
    get_target_property(_qmake_location Qt${QT_VERSION_MAJOR}::qmake IMPORTED_LOCATION)
    get_filename_component(_qt_bin_dir "${_qmake_location}" DIRECTORY)
    find_program(SIRIUS_WINDEPLOYQT windeployqt HINTS "${_qt_bin_dir}")
    if(SIRIUS_WINDEPLOYQT)
        get_target_property(_qt_core_debug Qt${QT_VERSION_MAJOR}::Core IMPORTED_LOCATION_DEBUG)
        if(_qt_core_debug)
            set(_deploy_flavor "$<IF:$<CONFIG:Debug>,--debug,--release>")
        else()
            set(_deploy_flavor "--release")
        endif()
        add_custom_command(TARGET sirius-app POST_BUILD
            COMMAND "${SIRIUS_WINDEPLOYQT}" ${_deploy_flavor} --no-translations --no-compiler-runtime
                    "$<TARGET_FILE:sirius-app>"
            COMMENT "Deploying Qt runtime next to sirius-app"
            VERBATIM)
    else()
        message(WARNING "windeployqt not found; sirius-app needs the Qt bin directory on PATH to run")
    endif()
endif()

# --- install (Linux) --------------------------------------------------------
#
#   cmake --install <build> --component app --prefix <prefix>
#
#   bin/sirius-app
#   share/sirius/{help,python,plugins}     what the build copies beside the executable
#   share/sirius/python/workbench.py       the step library (and its op_schema.json) the
#                                          worker loads when no sirius package is importable
#   share/applications/sirius-app.desktop  the menu entry and "Open with" for TIFFs
#   share/icons/hicolor/*/apps/sirius-app.{png,svg}
#   lib/sirius/                            nvTIFF / nvCOMP of a CUDA build, found via $ORIGIN
#
# tests/CMakeLists.txt installs this into the build tree and runs the result
# (app/linux/check_install.cmake). Windows deploys from the build tree
# (windeployqt above); a macOS bundle would need a layout of its own.
if(UNIX AND NOT APPLE)
    set(_app_rpath "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/sirius")
    if(SIRIUS_ENABLE_CUDA)
        list(APPEND _app_rpath "${CUDAToolkit_LIBRARY_DIR}")
    endif()
    # a Qt outside the linker's own directories (SIRIUS_QT_DIR, a Qt installer kit)
    get_target_property(_qt_core_location Qt${QT_VERSION_MAJOR}::Core LOCATION)
    get_filename_component(_qt_lib_dir "${_qt_core_location}" DIRECTORY)
    if(NOT _qt_lib_dir IN_LIST CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES)
        list(APPEND _app_rpath "${_qt_lib_dir}")
    endif()
    set_target_properties(sirius-app PROPERTIES INSTALL_RPATH "${_app_rpath}")

    install(TARGETS sirius-app RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT app)
    install(DIRECTORY help python plugins
        DESTINATION ${SIRIUS_APP_INSTALL_DATADIR} COMPONENT app
        PATTERN __pycache__ EXCLUDE
        PATTERN "python/tests" EXCLUDE)
    install(FILES
        ${PROJECT_SOURCE_DIR}/bindings/python/sirius/workbench.py
        ${PROJECT_SOURCE_DIR}/bindings/python/sirius/op_schema.json
        DESTINATION ${SIRIUS_APP_INSTALL_DATADIR}/python COMPONENT app)
    install(FILES linux/sirius-app.desktop DESTINATION ${CMAKE_INSTALL_DATADIR}/applications COMPONENT app)
    install(FILES qt/resources/icons/app/sirius-app.svg
        DESTINATION ${CMAKE_INSTALL_DATADIR}/icons/hicolor/scalable/apps COMPONENT app)
    foreach(_px 16 24 32 48 64 128 256)
        install(FILES qt/resources/icons/app/sirius-app-${_px}.png
            DESTINATION ${CMAKE_INSTALL_DATADIR}/icons/hicolor/${_px}x${_px}/apps
            RENAME sirius-app.png COMPONENT app)
    endforeach()
    # The redistributables ship "libnvtiff.so.0 -> libnvtiff.so.0.8.0.82"
    # symlinks; install the real file under the soname the executable needs
    # (as bindings/CMakeLists.txt does for the wheel).
    foreach(_lib IN LISTS SIRIUS_NVTIFF_RUNTIME_LIBS SIRIUS_NVCOMP_RUNTIME_LIBS)
        get_filename_component(_soname "${_lib}" NAME)
        file(REAL_PATH "${_lib}" _real)
        install(FILES "${_real}" DESTINATION ${CMAKE_INSTALL_LIBDIR}/sirius COMPONENT app RENAME "${_soname}")
    endforeach()
endif()
