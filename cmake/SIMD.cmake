include(CheckCXXCompilerFlag)

add_library(sirius_simd INTERFACE)

function(_sirius_enable_simd_flag option flag)
    if(NOT ${option})
        return()
    endif()
    check_cxx_compiler_flag("${flag}" _flag_supported)
    if(_flag_supported)
        target_compile_options(sirius_simd INTERFACE "${flag}")
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
