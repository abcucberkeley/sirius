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
target_compile_definitions(sirius_app_core PRIVATE
    SIRIUS_APP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
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
