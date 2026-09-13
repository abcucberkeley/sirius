include(CheckCXXCompilerFlag)

add_library(sirius_simd INTERFACE)

# Host-compiler flags only: nvcc does not understand -m<isa> (it would need
# -Xcompiler), and CUDA sources never contain the SIMD-sensitive FFTW/Eigen code.
function(_sirius_enable_simd_flag option flag)
    if(NOT ${option})
        return()
    endif()
    # One cache variable per flag: check_cxx_compiler_flag caches its result,
    # so a shared name would answer every later flag with the first's result.
    string(MAKE_C_IDENTIFIER "SIRIUS_CXX_SUPPORTS${flag}" _var)
    check_cxx_compiler_flag("${flag}" ${_var})
    if(${_var})
        target_compile_options(sirius_simd INTERFACE
            $<$<COMPILE_LANGUAGE:C,CXX>:${flag}>)
    else()
        message(WARNING "${option} requested but compiler does not support ${flag}")
    endif()
endfunction()

# AVX2 implies FMA; check both together since they are always paired in practice.
# AVX512 implies AVX2, so stack the flags.
_sirius_enable_simd_flag(SIRIUS_ENABLE_SSE2   "-msse2")
_sirius_enable_simd_flag(SIRIUS_ENABLE_AVX    "-mavx")
if(SIRIUS_ENABLE_AVX2)
    _sirius_enable_simd_flag(SIRIUS_ENABLE_AVX2 "-mavx2")
    _sirius_enable_simd_flag(SIRIUS_ENABLE_AVX2 "-mfma")
endif()
if(SIRIUS_ENABLE_AVX512)
    _sirius_enable_simd_flag(SIRIUS_ENABLE_AVX512 "-mavx512f")
    _sirius_enable_simd_flag(SIRIUS_ENABLE_AVX512 "-mfma")
endif()
