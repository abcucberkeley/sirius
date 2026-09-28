# sirius-imgui: the workbench of docs/design over Dear ImGui (docking), GLFW
# and OpenGL 3.3, with ImPlot for the diagnostics charts. The same Qt-free
# core as sirius-app (sirius::app_core); everything in this directory is the
# GUI layer only. Sources are globbed (CONFIGURE_DEPENDS) so adding a panel
# is adding a file, as for the Qt application.

file(GLOB_RECURSE SIRIUS_IMGUI_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_executable(sirius-imgui ${SIRIUS_IMGUI_SOURCES})
# "core/..." and "imgui/..." both resolve from app/.
target_include_directories(sirius-imgui PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(sirius-imgui PRIVATE sirius::app_core sirius::imgui nfd::nfd)
if(TARGET CURL::libcurl)
    target_link_libraries(sirius-imgui PRIVATE CURL::libcurl)
else()
    target_link_libraries(sirius-imgui PRIVATE libcurl)
endif()
target_compile_definitions(sirius-imgui PRIVATE
    SIRIUS_APP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/.."
    SIRIUS_VERSION="${PROJECT_VERSION}"
    GLFW_INCLUDE_NONE)
if(MSVC)
    # The process code page is UTF-8 (see the manifest): the core opens files
    # by their UTF-8 bytes through the narrow Win32 APIs.
    target_sources(sirius-imgui PRIVATE ${CMAKE_SOURCE_DIR}/cmake/utf8-codepage.manifest)
    target_compile_options(sirius-imgui PRIVATE /utf-8 /bigobj)
    target_compile_definitions(sirius-imgui PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    # A GUI program: no console window behind it when started from Explorer;
    # main() stays the entry point, and a console it was started from is
    # re-attached at start-up so --help and the scripting output still print.
    target_link_options(sirius-imgui PRIVATE /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup)
endif()
if(WIN32)
    # crypt32: DPAPI (CryptProtectData) behind imgui/secret_store.cpp
    target_link_libraries(sirius-imgui PRIVATE crypt32 shell32)
endif()
find_package(Threads REQUIRED)
target_link_libraries(sirius-imgui PRIVATE Threads::Threads)
if(SIRIUS_ENABLE_WARNINGS)
    target_set_warnings(sirius-imgui)
endif()
if(SIRIUS_ENABLE_SANITIZERS)
    target_enable_sanitizers(sirius-imgui)
endif()

# Help pages, the Python worker, the example plugins and the fonts live next
# to the executable so users can edit the pages and launch the worker from
# the build tree.
add_custom_command(TARGET sirius-imgui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/../help $<TARGET_FILE_DIR:sirius-imgui>/help
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/../python $<TARGET_FILE_DIR:sirius-imgui>/python
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/../plugins $<TARGET_FILE_DIR:sirius-imgui>/plugins
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/../qt/resources/fonts $<TARGET_FILE_DIR:sirius-imgui>/fonts
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/../qt/resources/icons/app $<TARGET_FILE_DIR:sirius-imgui>/icons
    COMMENT "Copying help pages, the Python worker, the example plugins and the fonts next to sirius-imgui"
    VERBATIM)

sirius_copy_runtime_dlls(sirius-imgui)

if(UNIX AND NOT APPLE)
    include(GNUInstallDirs)
    install(TARGETS sirius-imgui RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT imgui-app)
    install(DIRECTORY ../qt/resources/fonts
        DESTINATION ${SIRIUS_APP_INSTALL_DATADIR} COMPONENT imgui-app)
    if(NOT SIRIUS_ENABLE_APP)
        install(DIRECTORY ../help ../python ../plugins
            DESTINATION ${SIRIUS_APP_INSTALL_DATADIR} COMPONENT imgui-app
            PATTERN __pycache__ EXCLUDE
            PATTERN "python/tests" EXCLUDE)
        install(FILES
            ${PROJECT_SOURCE_DIR}/bindings/python/sirius/workbench.py
            ${PROJECT_SOURCE_DIR}/bindings/python/sirius/op_schema.json
            DESTINATION ${SIRIUS_APP_INSTALL_DATADIR}/python COMPONENT imgui-app)
    endif()
endif()
