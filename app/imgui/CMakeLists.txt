# sirius-app: the workbench of docs/design over Dear ImGui (docking), GLFW and
# OpenGL 3.3, with ImPlot for the diagnostics charts, on the GUI-free core
# (sirius::app_core). Sources are globbed (CONFIGURE_DEPENDS) so adding a panel
# is adding a file.

file(GLOB_RECURSE SIRIUS_APP_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_executable(sirius-app ${SIRIUS_APP_SOURCES})
# The executable sits in <build>/app (<build>/app/<config> for a multi-config
# generator), where the scripts, CI and the install check look for it.
set_target_properties(sirius-app PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/app)
# "core/..." and "imgui/..." both resolve from app/.
target_include_directories(sirius-app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(sirius-app PRIVATE sirius::app_core sirius::imgui nfd::nfd)
if(TARGET CURL::libcurl)
    target_link_libraries(sirius-app PRIVATE CURL::libcurl)
else()
    target_link_libraries(sirius-app PRIVATE libcurl)
endif()
target_compile_definitions(sirius-app PRIVATE
    SIRIUS_APP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/.."
    SIRIUS_VERSION="${PROJECT_VERSION}"
    GLFW_INCLUDE_NONE)
if(MSVC)
    # The process code page is UTF-8 (see the manifest): the core opens files
    # by their UTF-8 bytes through the narrow Win32 APIs. The linker merges
    # .manifest sources into the executable's manifest.
    target_sources(sirius-app PRIVATE ${CMAKE_SOURCE_DIR}/cmake/utf8-codepage.manifest)
    target_compile_options(sirius-app PRIVATE /utf-8 /bigobj)
    target_compile_definitions(sirius-app PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    # A GUI program: no console window behind it when started from Explorer;
    # main() stays the entry point, and a console it was started from is
    # re-attached at start-up so --help and the scripting output still print.
    target_link_options(sirius-app PRIVATE /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup)
endif()
if(WIN32)
    # crypt32: DPAPI (CryptProtectData) behind imgui/secret_store.cpp
    target_link_libraries(sirius-app PRIVATE crypt32 shell32)
endif()
find_package(Threads REQUIRED)
target_link_libraries(sirius-app PRIVATE Threads::Threads)
if(SIRIUS_ENABLE_WARNINGS)
    target_set_warnings(sirius-app)
endif()
if(SIRIUS_ENABLE_SANITIZERS)
    target_enable_sanitizers(sirius-app)
endif()

# Help pages, the Python worker, the example plugins, the fonts and the window
# icons live next to the executable so users can edit the pages and launch the
# worker from the build tree.
add_custom_command(TARGET sirius-app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/../help $<TARGET_FILE_DIR:sirius-app>/help
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/../python $<TARGET_FILE_DIR:sirius-app>/python
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/../plugins $<TARGET_FILE_DIR:sirius-app>/plugins
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/../resources/fonts $<TARGET_FILE_DIR:sirius-app>/fonts
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/../resources/icons $<TARGET_FILE_DIR:sirius-app>/icons
    COMMENT "Copying help pages, the Python worker, the example plugins, fonts and icons next to sirius-app"
    VERBATIM)

# Runtime DLLs on Windows: the NVIDIA libraries (as for the tests and
# benchmarks), so the executable starts from the build tree without PATH
# changes. Everything else is linked statically.
sirius_copy_runtime_dlls(sirius-app)

# --- install (Linux) --------------------------------------------------------
#
#   cmake --install <build> --component app --prefix <prefix>
#
#   bin/sirius-app
#   share/sirius/{help,python,plugins,fonts,icons}  what the build copies beside the executable
#   share/sirius/python/workbench.py       the step library (and its op_schema.json) the
#                                          worker loads when no sirius package is importable
#   share/applications/sirius-app.desktop  the menu entry and "Open with" for TIFFs
#   share/icons/hicolor/*/apps/sirius-app.{png,svg}
#   lib/sirius/                            nvTIFF / nvCOMP of a CUDA build, found via $ORIGIN
#
# tests/CMakeLists.txt installs this into the build tree and runs the result
# (app/linux/check_install.cmake). Windows runs from the build tree; a macOS
# bundle would need a layout of its own.
if(UNIX AND NOT APPLE)
    include(GNUInstallDirs)
    set(_app_rpath "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/sirius")
    if(SIRIUS_ENABLE_CUDA)
        list(APPEND _app_rpath "${CUDAToolkit_LIBRARY_DIR}")
    endif()
    set_target_properties(sirius-app PROPERTIES INSTALL_RPATH "${_app_rpath}")

    set(_app_dir ${CMAKE_CURRENT_SOURCE_DIR}/..)
    install(TARGETS sirius-app RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT app)
    install(DIRECTORY ${_app_dir}/help ${_app_dir}/python ${_app_dir}/plugins
        DESTINATION ${SIRIUS_APP_INSTALL_DATADIR} COMPONENT app
        PATTERN __pycache__ EXCLUDE
        PATTERN "python/tests" EXCLUDE)
    install(DIRECTORY ${_app_dir}/resources/fonts ${_app_dir}/resources/icons
        DESTINATION ${SIRIUS_APP_INSTALL_DATADIR} COMPONENT app
        PATTERN README.md EXCLUDE)
    install(FILES
        ${PROJECT_SOURCE_DIR}/bindings/python/sirius/workbench.py
        ${PROJECT_SOURCE_DIR}/bindings/python/sirius/op_schema.json
        DESTINATION ${SIRIUS_APP_INSTALL_DATADIR}/python COMPONENT app)
    install(FILES ${_app_dir}/linux/sirius-app.desktop DESTINATION ${CMAKE_INSTALL_DATADIR}/applications COMPONENT app)
    install(FILES ${_app_dir}/resources/icons/sirius-app.svg
        DESTINATION ${CMAKE_INSTALL_DATADIR}/icons/hicolor/scalable/apps COMPONENT app)
    foreach(_px 16 24 32 48 64 128 256)
        install(FILES ${_app_dir}/resources/icons/sirius-app-${_px}.png
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
