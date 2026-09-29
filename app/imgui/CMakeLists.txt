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
if(GLFW_BUILD_X11)
    # imgui/native_dialog.cpp hands the X11 window to the file dialogs as their
    # parent; a GLFW built for Wayland alone has no X11 functions to call.
    target_compile_definitions(sirius-app PRIVATE SIRIUS_GLFW_X11)
endif()
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
target_link_libraries(sirius-app PRIVATE Threads::Threads)   # found in cmake/Dependencies.cmake
if(SIRIUS_ENABLE_WARNINGS)
    target_set_warnings(sirius-app)
endif()
if(SIRIUS_ENABLE_SANITIZERS)
    target_enable_sanitizers(sirius-app)
endif()

# The fonts and the window icons live next to the executable, as the help
# pages, the Python worker and the example plugins do: those, and the NVIDIA
# runtime DLLs on Windows, are copied by sirius_app_runtime_files
# (app/CMakeLists.txt), which sirius-cli shares.
add_custom_command(TARGET sirius-app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/../resources/fonts $<TARGET_FILE_DIR:sirius-app>/fonts
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/../resources/icons $<TARGET_FILE_DIR:sirius-app>/icons
    COMMENT "Copying fonts and icons next to sirius-app"
    VERBATIM)
add_dependencies(sirius-app sirius_app_runtime_files)

# --- install (Linux) --------------------------------------------------------
#
#   cmake --install <build> --component app --prefix <prefix>
#
#   bin/sirius-app
#   share/sirius/{fonts,icons}             what the build copies beside the executable
#   share/applications/sirius-app.desktop  the menu entry and "Open with" for TIFFs
#   share/icons/hicolor/*/apps/sirius-app.{png,svg}
#
# The help pages, the worker, the plugins and the nv libraries both
# executables share are installed by app/CMakeLists.txt. tests/CMakeLists.txt
# installs this into the build tree and runs the result
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
    install(DIRECTORY ${_app_dir}/resources/fonts ${_app_dir}/resources/icons
        DESTINATION ${SIRIUS_APP_INSTALL_DATADIR} COMPONENT app
        PATTERN README.md EXCLUDE)
    install(FILES ${_app_dir}/linux/sirius-app.desktop DESTINATION ${CMAKE_INSTALL_DATADIR}/applications COMPONENT app)
    install(FILES ${_app_dir}/resources/icons/sirius-app.svg
        DESTINATION ${CMAKE_INSTALL_DATADIR}/icons/hicolor/scalable/apps COMPONENT app)
    foreach(_px 16 24 32 48 64 128 256)
        install(FILES ${_app_dir}/resources/icons/sirius-app-${_px}.png
            DESTINATION ${CMAKE_INSTALL_DATADIR}/icons/hicolor/${_px}x${_px}/apps
            RENAME sirius-app.png COMPONENT app)
    endforeach()
endif()
