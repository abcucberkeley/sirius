option(SIRIUS_ENABLE_MPI "Enable MPI" OFF)
option(SIRIUS_ENABLE_CUDA "Enable CUDA" OFF)

# Development related options
option(SIRIUS_ENABLE_TESTS "Enable tests" OFF)
option(SIRIUS_ENABLE_WARNINGS "Enable extra warnings" OFF)
option(SIRIUS_ENABLE_SANITIZERS "Enable sanitizers (Debug, non-MSVC)" OFF)
option(SIRIUS_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
option(SIRIUS_ENABLE_CPPCHECK "Enable cppcheck" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)