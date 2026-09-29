# sirius-cli: the workbench core (sirius::app_core) without a window, for
# scripts and agents -- one-shot commands that print JSON, a JSON-lines
# session and an MCP server on stdio (app/cli/README.md). No ImGui, GLFW,
# OpenGL, nfd or curl, so it also builds on a cluster node without X11 /
# Wayland (SIRIUS_ENABLE_CLI).

add_executable(sirius-cli main.cpp args.cpp commands.cpp stdio.cpp usage.cpp)
# Beside sirius-app in <build>/app (<build>/app/<config> for a multi-config
# generator): both find the help pages, the worker and the plugins there.
set_target_properties(sirius-cli PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/app)
# "core/..." and "cli/..." both resolve from app/.
target_include_directories(sirius-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(sirius-cli PRIVATE sirius::app_core Threads::Threads)
target_compile_definitions(sirius-cli PRIVATE SIRIUS_VERSION="${PROJECT_VERSION}")
# The help pages, the worker, the plugins (and the nv DLLs on Windows).
add_dependencies(sirius-cli sirius_app_runtime_files)
if(MSVC)
    # The process code page is UTF-8 (see the manifest), so argv and the
    # paths the core opens are UTF-8, as in sirius-app. The console subsystem
    # (the default) is kept: stdin and stdout are the protocol.
    target_sources(sirius-cli PRIVATE ${CMAKE_SOURCE_DIR}/cmake/utf8-codepage.manifest)
    target_compile_options(sirius-cli PRIVATE /utf-8 /bigobj)
    target_compile_definitions(sirius-cli PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()
if(SIRIUS_ENABLE_WARNINGS)
    target_set_warnings(sirius-cli)
endif()
if(SIRIUS_ENABLE_SANITIZERS)
    target_enable_sanitizers(sirius-cli)
endif()

# --- install (Linux) --------------------------------------------------------
#
#   bin/sirius-cli    beside bin/sirius-app; the shared data is installed by
#                     app/CMakeLists.txt
if(UNIX AND NOT APPLE)
    include(GNUInstallDirs)
    set(_cli_rpath "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/sirius")
    if(SIRIUS_ENABLE_CUDA)
        list(APPEND _cli_rpath "${CUDAToolkit_LIBRARY_DIR}")
    endif()
    set_target_properties(sirius-cli PROPERTIES INSTALL_RPATH "${_cli_rpath}")
    install(TARGETS sirius-cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT app)
endif()
