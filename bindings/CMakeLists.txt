set(SIRIUS_BINDING_SOURCES
    src/module.cpp
    src/bind_fft.cpp
    src/bind_tiff_io.cpp
)

# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  _sirius_ext

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
  ${SIRIUS_BINDING_SOURCES}
)

target_link_libraries(_sirius_ext PRIVATE sirius sirius_simd)
target_include_directories(_sirius_ext PRIVATE ${PROJECT_SOURCE_DIR}/include)

if(SIRIUS_ENABLE_CUDA)
    target_compile_definitions(_sirius_ext PRIVATE SIRIUS_HAS_CUDA_BINDINGS)
    target_sources(_sirius_ext PRIVATE src/bind_cuda_batch.cpp)
endif()

if(SIRIUS_ENABLE_MPI)
    target_sources(_sirius_ext PRIVATE src/bind_mpi_distributor.cpp)
endif()

# The pure-Python sources are shipped via `wheel.packages` in pyproject.toml,
# which is also what makes `pip install -e .` editable for Python files.
# CMake only owns the compiled extension and any runtime DLLs.
install(TARGETS _sirius_ext LIBRARY DESTINATION sirius COMPONENT python)
install(FILES $<TARGET_RUNTIME_DLLS:_sirius_ext> DESTINATION sirius COMPONENT python)