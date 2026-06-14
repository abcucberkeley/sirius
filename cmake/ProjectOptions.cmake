option(SIRIUS_ENABLE_MPI "Enable MPI" OFF)
option(SIRIUS_ENABLE_CUDA "Enable CUDA" OFF)
option(SIRIUS_ENABLE_PYTHON_BINDINGS "Enable nanobind python bindings" OFF)

# scikit-build-core always builds the Python extension
if(SKBUILD)
    set(SIRIUS_ENABLE_PYTHON_BINDINGS ON CACHE BOOL "" FORCE)
endif()
option(SIRIUS_ENABLE_SSE2   "Enable SSE2 instruction set"    OFF)
option(SIRIUS_ENABLE_AVX    "Enable AVX instruction set"     OFF)
option(SIRIUS_ENABLE_AVX2   "Enable AVX2 + FMA instruction sets" OFF)
option(SIRIUS_ENABLE_AVX512 "Enable AVX-512F + FMA instruction sets" OFF)

# Development related options
option(SIRIUS_ENABLE_TESTS "Enable tests" OFF)
option(SIRIUS_ENABLE_BENCHMARKS "Build C++ benchmarks" OFF)
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
    # Silently ignore failures (e.g. Windows without Developer Mode enabled)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            "${CMAKE_BINARY_DIR}/compile_commands.json"
            "${CMAKE_SOURCE_DIR}/compile_commands.json"
        RESULT_VARIABLE _symlink_result
        ERROR_QUIET
        OUTPUT_QUIET
    )
endif()