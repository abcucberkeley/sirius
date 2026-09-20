# The library, unit by unit (cmake/Units.cmake): every block below is one
# header, its sources and the units it may include or call -- declared in
# dependency order, so a unit can only name units above it and the graph stays
# acyclic. tests/CMakeLists.txt gives each unit a test executable that links
# the unit's closure and nothing else; docs/architecture.md draws the graph.
#
# `sirius`, at the bottom, is what it always was: one static archive of all of
# it, the target the bindings, the application, the benchmarks and the
# installed package link.

include(Units)

# sirius_lib_unit(<name> [HEADERS ...] [SOURCES ...] [CUDA_SOURCES ...] [DEPENDS ...] [LINK ...] [PUBLIC_LINK ...])
# HEADERS are relative to include/ (public) or src/ (internal, the ones with no
# sirius/ prefix). CUDA_SOURCES (and CUDA_LINK) count only in a CUDA build.
function(sirius_lib_unit name)
    cmake_parse_arguments(PARSE_ARGV 1 L "" "" "HEADERS;SOURCES;CUDA_SOURCES;DEPENDS;LINK;CUDA_LINK;PUBLIC_LINK")
    set(headers "")
    foreach(h IN LISTS L_HEADERS)
        if(h MATCHES "^sirius/")
            list(APPEND headers ${PROJECT_SOURCE_DIR}/include/${h})
        else()
            list(APPEND headers ${CMAKE_CURRENT_SOURCE_DIR}/${h})
        endif()
    endforeach()
    if(SIRIUS_ENABLE_CUDA)
        list(APPEND L_SOURCES ${L_CUDA_SOURCES})
        list(APPEND L_LINK ${L_CUDA_LINK})
    endif()
    sirius_add_unit(lib ${name}
        HEADERS ${headers}
        SOURCES ${L_SOURCES}
        DEPENDS ${L_DEPENDS}
        LINK ${L_LINK}
        PUBLIC_LINK ${L_PUBLIC_LINK}
        INCLUDE_PUBLIC ${PROJECT_SOURCE_DIR}/include)
    if(L_SOURCES)
        set(tgt sirius_lib_${name})
        # internal headers (src/) are for the library's own sources -- and for
        # the tests that ask for them (INTERNAL in sirius_unit_test)
        target_include_directories(${tgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        # Library-wide, as they were when this was one target: OpenMP (Eigen
        # reads _OPENMP too, so a unit without a pragma still has to agree),
        # and the feature macros. The public headers never depend on these
        # macros (see device.hpp); they only select implementation paths.
        target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(${tgt} PRIVATE ${SIRIUS_LIB_DEFINITIONS})
    endif()
endfunction()

set(SIRIUS_LIB_DEFINITIONS "")
if(SIRIUS_ENABLE_TENSORSTORE)
    list(APPEND SIRIUS_LIB_DEFINITIONS SIRIUS_HAS_TENSORSTORE=1)
endif()
if(SIRIUS_ENABLE_CUDA)
    list(APPEND SIRIUS_LIB_DEFINITIONS SIRIUS_HAS_CUDA=1)
    if(SIRIUS_ENABLE_NVTIFF)
        list(APPEND SIRIUS_LIB_DEFINITIONS SIRIUS_HAS_NVTIFF=1)
    endif()
endif()

# --- vocabulary: header-only, no dependencies -------------------------------

sirius_lib_unit(errors        HEADERS sirius/errors.hpp)
sirius_lib_unit(index         HEADERS sirius/index.hpp)
sirius_lib_unit(checked_math  HEADERS sirius/checked_math.hpp)
sirius_lib_unit(constants     HEADERS sirius/constants.hpp)
sirius_lib_unit(pixel_type    HEADERS sirius/pixel_type.hpp)
sirius_lib_unit(sim_math      HEADERS sim_math.hpp                       DEPENDS constants)
sirius_lib_unit(tensor_util   HEADERS sirius/tensor_util.hpp             PUBLIC_LINK Eigen3::Eigen)
sirius_lib_unit(fft_util      HEADERS sirius/fft_util.hpp                DEPENDS tensor_util)
# compiles OpenMP pragmas into whoever includes it
sirius_lib_unit(downsample    HEADERS downsample.hpp                     DEPENDS index checked_math)

# --- memory ------------------------------------------------------------------

sirius_lib_unit(device
    HEADERS sirius/device.hpp cuda_check.hpp
    SOURCES device.cpp
    DEPENDS errors
    CUDA_LINK CUDA::cudart_static)

sirius_lib_unit(buffer
    HEADERS sirius/buffer.hpp cuda/kernels.hpp
    SOURCES buffer.cpp
    CUDA_SOURCES cuda/kernels.cu
    DEPENDS device index checked_math errors
    PUBLIC_LINK Eigen3::Eigen
    CUDA_LINK CUDA::cudart_static)

# --- transforms ---------------------------------------------------------------

# what the two transforms share: the FFTW planner lock and thread count, the
# rigor flags, the aligned allocator, the GPU normalization kernels
sirius_lib_unit(fft_common
    HEADERS sirius/fft_common.hpp fftw_internal.hpp cufft_check.hpp cuda/fft_kernels.hpp
    SOURCES fft_common.cpp
    CUDA_SOURCES cuda/fft_kernels.cu
    LINK ${SIRIUS_FFTW_TARGETS}
    CUDA_LINK CUDA::cudart_static)

sirius_lib_unit(fft
    HEADERS sirius/fft.hpp fft_backend.hpp
    SOURCES fft.cpp
    CUDA_SOURCES fft_cufft.cpp
    DEPENDS buffer device fft_common tensor_util
    LINK ${SIRIUS_FFTW_TARGETS}
    CUDA_LINK CUDA::cudart_static CUDA::cufft)

sirius_lib_unit(real_fft
    HEADERS sirius/real_fft.hpp real_fft_backend.hpp
    SOURCES real_fft.cpp
    CUDA_SOURCES real_fft_cufft.cpp
    DEPENDS device fft_common tensor_util
    LINK ${SIRIUS_FFTW_TARGETS}
    CUDA_LINK CUDA::cudart_static CUDA::cufft)

# --- files -------------------------------------------------------------------

sirius_lib_unit(tiff_io
    HEADERS sirius/tiff_io.hpp tiff_internal.hpp
    SOURCES tiff_io.cpp
    DEPENDS buffer device pixel_type downsample errors
    LINK TIFF::TIFF)
if(SIRIUS_ENABLE_CUDA AND SIRIUS_ENABLE_NVTIFF)
    target_sources(sirius_lib_tiff_io PRIVATE tiff_nvtiff.cpp)
    target_link_libraries(sirius_lib_tiff_io PRIVATE CUDA::cudart_static nvtiff::nvtiff)
    if(SIRIUS_ENABLE_NVCOMP)
        target_link_libraries(sirius_lib_tiff_io PRIVATE sirius_nvcomp_runtime)
    endif()
endif()

sirius_lib_unit(zarr_io
    HEADERS sirius/zarr_io.hpp
    SOURCES zarr_io.cpp
    DEPENDS buffer pixel_type downsample errors
    # Header-only and used only inside the .cpp: compiled into the archive, it
    # contributes nothing to a consumer's link line.
    LINK nlohmann_json::nlohmann_json)   # zarr / N5 metadata
if(SIRIUS_ENABLE_TENSORSTORE)
    target_link_libraries(sirius_lib_zarr_io PRIVATE sirius::tensorstore)
endif()

# --- image processing ---------------------------------------------------------

sirius_lib_unit(image_ops
    HEADERS sirius/image_ops.hpp
    SOURCES image_ops.cpp
    DEPENDS index downsample constants checked_math)

sirius_lib_unit(registration
    HEADERS sirius/registration.hpp
    SOURCES registration.cpp
    DEPENDS buffer device real_fft fft_common)

sirius_lib_unit(deconvolution
    HEADERS sirius/deconvolution.hpp
    SOURCES deconvolution.cpp
    DEPENDS buffer device real_fft fft_common)

sirius_lib_unit(stitching
    HEADERS sirius/stitching.hpp
    SOURCES stitching.cpp
    DEPENDS buffer registration)

sirius_lib_unit(stitching_tiff
    HEADERS sirius/stitching_tiff.hpp
    SOURCES stitching_tiff.cpp
    DEPENDS stitching tiff_io buffer)

# --- structured illumination ----------------------------------------------------

sirius_lib_unit(sim_parameters
    HEADERS sirius/sim_parameters.hpp
    SOURCES sim_parameters.cpp
    DEPENDS errors
    LINK tomlplusplus::tomlplusplus)

sirius_lib_unit(legacy_config
    HEADERS sirius/legacy_config.hpp
    SOURCES legacy_config.cpp
    DEPENDS sim_parameters errors)

# the OpenMP preprocessing stages the public API and the reconstruction share
sirius_lib_unit(sim_stages
    HEADERS sim_cpu_stages.hpp
    SOURCES sim_cpu_stages.cpp
    DEPENDS sim_math constants)

sirius_lib_unit(preprocess
    HEADERS sirius/preprocess.hpp
    SOURCES preprocess.cpp
    DEPENDS sim_stages
    PUBLIC_LINK Eigen3::Eigen)

sirius_lib_unit(separation
    HEADERS sirius/separation.hpp
    SOURCES separation.cpp
    DEPENDS sim_stages constants
    PUBLIC_LINK Eigen3::Eigen)

sirius_lib_unit(otf
    HEADERS sirius/otf.hpp
    SOURCES otf.cpp
    PUBLIC_LINK Eigen3::Eigen)

sirius_lib_unit(otf_io
    HEADERS sirius/otf_io.hpp
    SOURCES otf_io.cpp
    DEPENDS otf sim_parameters tiff_io errors)

sirius_lib_unit(otf_ideal
    HEADERS sirius/otf_ideal.hpp
    SOURCES otf_ideal.cpp
    DEPENDS otf sim_parameters fft buffer constants)

# the per-device stage implementations behind the reconstruction driver
sirius_lib_unit(sim_backend
    HEADERS sim_internal.hpp
    SOURCES sim_cpu.cpp
    CUDA_SOURCES cuda/sim_kernels.cu
    DEPENDS sim_stages sim_math device
    CUDA_LINK CUDA::cudart_static)

sirius_lib_unit(sim_reconstruction
    HEADERS sirius/sim_reconstruction.hpp
    SOURCES sim_reconstruction.cpp
    DEPENDS sim_backend sim_stages separation otf sim_parameters fft real_fft fft_common buffer device constants
            errors)

# --- the archive -----------------------------------------------------------------

add_library(sirius STATIC)
sirius_archive_units(sirius lib)
set_target_properties(sirius PROPERTIES LINKER_LANGUAGE CXX)   # it has objects, no sources to tell by

if(SIRIUS_ENABLE_PYTHON_BINDINGS)
    set_target_properties(sirius PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(sirius
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# the installed headers: every unit's public ones
target_sources(sirius
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS
            ${PROJECT_SOURCE_DIR}/include
        FILES
            ${PROJECT_SOURCE_DIR}/include/sirius/errors.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/index.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/device.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/checked_math.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/buffer.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/pixel_type.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/tiff_io.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/tensor_util.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/fft.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/fft_common.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/real_fft.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/fft_util.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/sim_parameters.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/legacy_config.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/otf.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/otf_io.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/otf_ideal.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/preprocess.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/separation.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/sim_reconstruction.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/registration.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/stitching.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/stitching_tiff.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/constants.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/image_ops.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/deconvolution.hpp
            ${PROJECT_SOURCE_DIR}/include/sirius/zarr_io.hpp
)

# The units compile; the archive is what consumers link, so the libraries the
# objects need are named once more here, for them.
target_link_libraries(sirius
    PUBLIC
        Eigen3::Eigen
        # Public, like Eigen: every translation unit that instantiates Eigen
        # must agree on the instruction set, because it decides
        # EIGEN_MAX_ALIGN_BYTES. A consumer built without -mavx2 allocates
        # 16-byte-aligned matrices that the library's AVX code then reads with
        # aligned loads (segfault in separationMatrix and the stitching global
        # fit on the fiona-avx2-* presets), and the linker may pick either
        # copy of an inline Eigen function.
        sirius_simd
    PRIVATE
        TIFF::TIFF
        OpenMP::OpenMP_CXX
        ${SIRIUS_FFTW_TARGETS}
)
# toml++ and nlohmann_json are header-only and used only inside the .cpp files
# of their units: nothing of them reaches a consumer's link line, so they are
# not named here and the install tree does not have to ship them
# (cmake/Install.cmake).

if(SIRIUS_ENABLE_TENSORSTORE)
    target_link_libraries(sirius PRIVATE sirius::tensorstore)
endif()

if(SIRIUS_ENABLE_CUDA)
    target_link_libraries(sirius PRIVATE CUDA::cudart_static CUDA::cufft)
    if(SIRIUS_ENABLE_NVTIFF)
        target_link_libraries(sirius PRIVATE nvtiff::nvtiff)
        if(SIRIUS_ENABLE_NVCOMP)
            target_link_libraries(sirius PRIVATE sirius_nvcomp_runtime)
        endif()
    endif()
endif()

if(SIRIUS_ENABLE_SANITIZERS)
    target_enable_sanitizers(sirius)
endif()
