include(CheckCXXCompilerFlag)

add_library(sirius_simd INTERFACE)

# Host-compiler flags only: nvcc does not understand -m<isa> (it would need
# -Xcompiler), and CUDA sources never contain the SIMD-sensitive FFTW/Eigen code.
#
# MSVC spells these differently and coarsely: one /arch: for the whole
# instruction set, no separate FMA switch (/arch:AVX2 brings FMA with it), and
# x64 has SSE2 unconditionally, so there is nothing to ask for below AVX.
# Without this mapping SIRIUS_ENABLE_AVX2=ON on Windows produced a warning and
# no flag, and the build quietly stayed at baseline x64.
function(_sirius_simd_flag_for out_var gcc_flag)
    set(${out_var} "" PARENT_SCOPE)
    if(NOT MSVC)
        set(${out_var} "${gcc_flag}" PARENT_SCOPE)
        return()
    endif()
    if("${gcc_flag}" STREQUAL "-msse2")
        return()                            # implied by x64; /arch:SSE2 is x86-only
    elseif("${gcc_flag}" STREQUAL "-mfma")
        return()                            # /arch:AVX2 already implies FMA
    elseif("${gcc_flag}" STREQUAL "-mavx")
        set(${out_var} "/arch:AVX" PARENT_SCOPE)
    elseif("${gcc_flag}" STREQUAL "-mavx2")
        set(${out_var} "/arch:AVX2" PARENT_SCOPE)
    elseif("${gcc_flag}" STREQUAL "-mavx512f")
        set(${out_var} "/arch:AVX512" PARENT_SCOPE)
    endif()
endfunction()

function(_sirius_enable_simd_flag option gcc_flag)
    if(NOT ${option})
        return()
    endif()
    _sirius_simd_flag_for(flag "${gcc_flag}")
    if("${flag}" STREQUAL "")
        return()                            # nothing this compiler needs asking for
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
