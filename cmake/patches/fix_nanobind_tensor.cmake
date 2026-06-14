# Patch for an upstream nanobind bug in the Eigen::Tensor type caster.
#
# nanobind/eigen/tensor.h declares the tensor shape as `std::array<long, N>`.
# On MSVC `long` is 32-bit, but Eigen::Tensor's index type is Eigen::Index
# (std::ptrdiff_t == __int64 on Win64), so `value.resize(out_dims)` finds no
# viable overload and the extension fails to compile. The bug is still present
# on nanobind master, so this cannot be resolved by bumping the pinned commit.
#
# Replacing `long` with `Eigen::Index` is correct on every platform:
# Eigen::Index is 64-bit on both Win64 and LP64 Linux, matching the resize
# overload that takes `std::array<Eigen::Index, N>`.
#
# Run via FetchContent's PATCH_COMMAND, whose working directory is the
# populated nanobind source tree. The string replace is a no-op once applied,
# so re-running the patch is safe.

set(_tensor_header "include/nanobind/eigen/tensor.h")

if(NOT EXISTS "${_tensor_header}")
    message(FATAL_ERROR "fix_nanobind_tensor: ${_tensor_header} not found "
                        "(working dir: ${CMAKE_CURRENT_BINARY_DIR}). "
                        "Has the nanobind layout changed?")
endif()

file(READ "${_tensor_header}" _contents)
string(REPLACE
    "std::array<long, NumIndices>"
    "std::array<Eigen::Index, NumIndices>"
    _contents "${_contents}")
file(WRITE "${_tensor_header}" "${_contents}")

message(STATUS "fix_nanobind_tensor: patched ${_tensor_header} "
               "(std::array<long, N> -> std::array<Eigen::Index, N>)")
