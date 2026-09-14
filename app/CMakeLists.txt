# The SIRIUS workbench (docs/design/README.md). Gated by SIRIUS_ENABLE_APP at
# the top level; Qt is located in cmake/Dependencies.cmake as sirius::qt.
#
#   sirius_app_core  static, Qt-free: array model, operations, pipeline,
#                    executor, workbench, tool API, worker protocol, export.
#                    Unit-tested by tests/test_app_*.cpp without a display.
#   sirius-app       the Qt Widgets executable (app/qt/**).
#
# Sources are globbed (CONFIGURE_DEPENDS) so adding an operation or a panel
# is adding a file.

# --- core -------------------------------------------------------------------
file(GLOB_RECURSE SIRIUS_APP_CORE_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp)
add_library(sirius_app_core STATIC ${SIRIUS_APP_CORE_SOURCES})
add_library(sirius::app_core ALIAS sirius_app_core)
target_include_directories(sirius_app_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sirius_app_core
    PUBLIC sirius nlohmann_json::nlohmann_json
    PRIVATE OpenMP::OpenMP_CXX TIFF::TIFF tomlplusplus::tomlplusplus)
# Where an installed tree keeps help/, python/ and plugins/, relative to the
# executable's directory (core/app_paths.cpp): computed from the full install
# directories so a --prefix given at install time moves both together.
include(GNUInstallDirs)
set(SIRIUS_APP_INSTALL_DATADIR "${CMAKE_INSTALL_DATADIR}/sirius")
file(RELATIVE_PATH _sirius_app_datadir_from_bindir
    "${CMAKE_INSTALL_FULL_BINDIR}" "${CMAKE_INSTALL_FULL_DATADIR}/sirius")
target_compile_definitions(sirius_app_core PRIVATE
    SIRIUS_APP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    SIRIUS_APP_DATADIR_FROM_BINDIR="${_sirius_app_datadir_from_bindir}")
if(WIN32)
    target_link_libraries(sirius_app_core PRIVATE ws2_32)
endif()
if(SIRIUS_ENABLE_WARNINGS)
    target_set_warnings(sirius_app_core)
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
    target_sources(sirius-app PRIVATE qt/resources/sirius-app.manifest)
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
