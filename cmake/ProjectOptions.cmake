option(SIRIUS_ENABLE_MPI "Enable MPI" OFF)
option(SIRIUS_ENABLE_CUDA "Enable CUDA" OFF)
option(SIRIUS_ENABLE_PYTHON_BINDINGS "Enable nanobind python bindings" OFF)

# Development related options
option(SIRIUS_ENABLE_TESTS "Enable tests" OFF)
option(SIRIUS_ENABLE_WARNINGS "Enable extra warnings" OFF)
option(SIRIUS_ENABLE_SANITIZERS "Enable sanitizers (Debug, non-MSVC)" OFF)
option(SIRIUS_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
option(SIRIUS_ENABLE_CPPCHECK "Enable cppcheck" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Symlink the build-dir compile_commands.json for IDE integration
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Generate compile_commands.json" FORCE)
    file(CREATE_LINK
        "${CMAKE_BINARY_DIR}/compile_commands.json"
        "${CMAKE_SOURCE_DIR}/compile_commands.json"
        SYMBOLIC)
endif()