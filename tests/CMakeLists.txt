# One test executable per unit (cmake/Units.cmake): test_<unit> links the unit
# under test, the units it depends on, and nothing else -- so `buffer` is
# tested without an FFT in the binary, `registration` without a TIFF reader,
# and a unit that reaches for something it did not declare fails to link here
# first. Every TEST_CASE is a CTest test named "sirius::<case>", as it was when
# this was one binary.
#
#     cmake --build <build> --target test_registration
#     ctest --test-dir <build> -L registration        # labels are lib.<unit> / core.<unit>; -L is a regex
#
# GPU-only cases SKIP at run time when no CUDA device is usable, so the same
# binaries pass on CPU-only CI; the [cuda] tag selects them.

include(Catch)
include(Units)

# sirius_lib_test(<unit> <sources>... [UNITS <more units>...] [LINK ...] [INTERNAL])
# The test of a library unit: the unit itself plus whatever else its test
# sources include (a Buffer to hold the data, a TIFF to read a fixture).
function(sirius_lib_test unit)
    cmake_parse_arguments(PARSE_ARGV 1 T "INTERNAL" "" "UNITS;LINK")
    set(internal "")
    if(T_INTERNAL)
        set(internal INTERNAL)
    endif()
    sirius_unit_test(${unit}
        SOURCES ${T_UNPARSED_ARGUMENTS}
        UNITS lib/${unit} ${T_UNITS}
        LINK ${T_LINK}
        LABEL lib.${unit}
        ${internal})
endfunction()

# --- the library ---------------------------------------------------------------

sirius_lib_test(tensor_util     test_tensor_util.cpp)
sirius_lib_test(fft_util        test_fft_util.cpp)
sirius_lib_test(buffer          test_buffer.cpp)        # and `device`, which has no life apart from it
sirius_lib_test(fft             test_fft1d.cpp test_fft2d.cpp test_fft3d.cpp test_fft_inplace.cpp test_fft_cuda.cpp)
sirius_lib_test(real_fft        test_real_fft.cpp test_real_fft_cuda.cpp UNITS lib/buffer)
# test_tiff_file / test_tiff_formats write and read back tags with libtiff and
# include tiff_internal.hpp
sirius_lib_test(tiff_io         test_tiff_io.cpp test_tiff_file.cpp test_tiff_formats.cpp LINK TIFF::TIFF INTERNAL)
sirius_lib_test(zarr_io         test_zarr_io.cpp)
# the .dv fixtures are checked value for value against their TIFF twins
sirius_lib_test(mrc_io          test_mrc_io.cpp UNITS lib/tiff_io)
# downsample.hpp, the internal header image_ops shares with the writers, is tested here too
sirius_lib_test(image_ops       test_image_ops.cpp INTERNAL)
sirius_lib_test(registration    test_registration.cpp)
sirius_lib_test(deconvolution   test_deconvolution.cpp)
sirius_lib_test(stitching       test_stitching.cpp)
sirius_lib_test(stitching_tiff  test_stitching_tiff.cpp)
sirius_lib_test(sim_parameters  test_sim_parameters.cpp)
sirius_lib_test(legacy_config   test_legacy_config.cpp)
# the library-internal OpenMP kernels shared by the public preprocessing API
# and the reconstruction pipeline
sirius_lib_test(sim_stages      test_sim_cpu_stages.cpp INTERNAL)
sirius_lib_test(preprocess      test_preprocess.cpp)
sirius_lib_test(separation      test_separation.cpp)
sirius_lib_test(otf             test_otf.cpp)
sirius_lib_test(otf_io          test_otf_io.cpp)
sirius_lib_test(otf_ideal       test_otf_ideal.cpp)
sirius_lib_test(sim_reconstruction test_reconstruction.cpp
    UNITS lib/otf_io lib/otf_ideal lib/legacy_config lib/tiff_io)

# --- the workbench core ----------------------------------------------------------
# The GUI's core is tested the same way when the application or sirius-cli is
# enabled; the GUI layer itself is exercised by CI's headless run of the bundled
# pipeline.
if(SIRIUS_ENABLE_APP OR SIRIUS_ENABLE_CLI)
    # What the child-process, Python-environment and worker tests start: a
    # stdlib-only program whose argv picks what it does (echo stdin, print,
    # sleep, exit with a code, print a variable, start a grandchild), so they
    # need no shell and no CMake newer than the minimum on any platform.
    add_executable(sirius_test_child tools/sirius_test_child.cpp)
    if(SIRIUS_ENABLE_WARNINGS)
        target_set_warnings(sirius_test_child)
    endif()
    sirius_finish_test_executable(sirius_test_child)
    include(app_tests.cmake)
endif()

# --- all of them in one binary ------------------------------------------------------
# Not registered with CTest (the cases above already are): it is for running
# across units by tag -- `sirius_tests "[cuda]"`, `sirius_tests "[parity]"` --
# and it links the *archives*, which shows that `sirius` and `sirius_app_core`
# still hold everything the units do.
get_property(_sirius_test_objects GLOBAL PROPERTY SIRIUS_TEST_OBJECTS)
add_executable(sirius_tests)
target_link_libraries(sirius_tests PRIVATE ${_sirius_test_objects} sirius Catch2::Catch2WithMain)
if(SIRIUS_ENABLE_APP OR SIRIUS_ENABLE_CLI)
    target_link_libraries(sirius_tests PRIVATE sirius::app_core)
    add_dependencies(sirius_tests sirius_test_child)
endif()
sirius_finish_test_executable(sirius_tests)

# The workbench as `cmake --install` lays it out, run from there (Linux only;
# see app/CMakeLists.txt): sirius-cli always, the GUI when it is built. With
# $SIRIUS_PYTHON set when the test runs it also starts the installed worker.
if((SIRIUS_ENABLE_APP OR SIRIUS_ENABLE_CLI) AND UNIX AND NOT APPLE)
    if(SIRIUS_ENABLE_APP)
        set(_has_app ON)
    else()
        set(_has_app OFF)
    endif()
    add_test(NAME sirius::app.install
        COMMAND ${CMAKE_COMMAND}
            -DBUILD_DIR=${PROJECT_BINARY_DIR}
            -DPREFIX=${CMAKE_CURRENT_BINARY_DIR}/install-check
            -DCONFIG=$<CONFIG>
            -DRAW=${PROJECT_SOURCE_DIR}/tests/data/raw.tif
            -DHAS_APP=${_has_app}
            -P ${PROJECT_SOURCE_DIR}/app/linux/check_install.cmake)
    set_tests_properties(sirius::app.install PROPERTIES TIMEOUT 300 LABELS app)
endif()

# --- sirius-cli, end to end ------------------------------------------------------
# The executable as a script or an agent drives it: one-shot commands, the
# JSON-lines session and the MCP server, over pipes (tests/cli/check_cli.cmake
# reads what comes back line by line; tests/cli/agent_client.py talks to it
# interactively). Each mode works in a directory of its own, and
# SIRIUS_PYTHON_ENV points into it, so no test ever reads or changes the
# user's own Python environment. The bundled SIM run is labelled cli.slow:
# unmeasured in an MSVC Debug build, it is left to the Linux jobs there
# (ctest -LE cli.slow). A test that cannot run here prints "SKIPPED:".
if(TARGET sirius-cli)
    function(sirius_cli_test mode)
        cmake_parse_arguments(PARSE_ARGV 1 C "" "TIMEOUT" "LABELS")
        set(_work ${CMAKE_CURRENT_BINARY_DIR}/cli-work/${mode})
        add_test(NAME sirius::cli.${mode}
            COMMAND ${CMAKE_COMMAND}
                -DCLI=$<TARGET_FILE:sirius-cli>
                -DSRC=${PROJECT_SOURCE_DIR}
                -DWORK=${_work}
                -DMODE=${mode}
                -P ${PROJECT_SOURCE_DIR}/tests/cli/check_cli.cmake)
        if(NOT C_LABELS)
            set(C_LABELS cli)
        endif()
        if(NOT C_TIMEOUT)
            set(C_TIMEOUT 300)
        endif()
        set_tests_properties(sirius::cli.${mode} PROPERTIES
            LABELS "${C_LABELS}"
            TIMEOUT ${C_TIMEOUT}
            SKIP_REGULAR_EXPRESSION "SKIPPED:"
            ENVIRONMENT "SIRIUS_PYTHON_ENV=${_work}/pyenv")
    endfunction()

    sirius_cli_test(oneshot)
    sirius_cli_test(run_contrast)
    sirius_cli_test(run_bundled LABELS cli cli.slow TIMEOUT 900)
    sirius_cli_test(session)
    sirius_cli_test(mcp_legacy)
    sirius_cli_test(mcp_modern)
    sirius_cli_test(worker_check)

    # The agent's side of the protocols, interactive over pipes (stdlib only).
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        set(_work ${CMAKE_CURRENT_BINARY_DIR}/cli-work/agent_client)
        add_test(NAME sirius::cli.agent_client
            COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/cli/agent_client.py
                --cli $<TARGET_FILE:sirius-cli> --data ${PROJECT_SOURCE_DIR})
        set_tests_properties(sirius::cli.agent_client PROPERTIES
            LABELS cli
            TIMEOUT 600
            SKIP_REGULAR_EXPRESSION "SKIPPED:"
            ENVIRONMENT "SIRIUS_PYTHON_ENV=${_work}/pyenv")
    endif()
endif()
