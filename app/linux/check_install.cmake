# Installs the app component into a scratch prefix and runs what was installed.
#
#   cmake -DBUILD_DIR=<build> -DPREFIX=<scratch> [-DCONFIG=<cfg>] -DRAW=<raw.tif>
#         [-DPYTHON=<interpreter with numpy>] -P check_install.cmake
#
# PYTHON defaults to $SIRIUS_PYTHON; without either the worker is not started.
#
# Registered as a test by tests/CMakeLists.txt. The binary is built with its
# source tree compiled in as a fallback, so an installed copy on this machine
# would still find the checkout's help pages and worker if its own were
# missing: the checks below plant a marker in the installed help and look for
# it, and start the worker from the installed directory, so the install is what
# is tested.

cmake_minimum_required(VERSION 3.25)

foreach(_var BUILD_DIR PREFIX RAW)
    if(NOT DEFINED ${_var})
        message(FATAL_ERROR "check_install.cmake: -D${_var}=... is required")
    endif()
endforeach()

function(fail)
    string(JOIN "" _msg ${ARGN})
    message(FATAL_ERROR "install check: ${_msg}")
endfunction()

if(NOT PYTHON AND DEFINED ENV{SIRIUS_PYTHON})
    set(PYTHON "$ENV{SIRIUS_PYTHON}")
endif()

file(REMOVE_RECURSE "${PREFIX}")
set(_config_args)
if(CONFIG)
    set(_config_args --config "${CONFIG}")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --component app --prefix "${PREFIX}" ${_config_args}
    RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
    fail("cmake --install failed (${_rc}):\n${_out}${_err}")
endif()

# --- the layout ---------------------------------------------------------------
set(_data "${PREFIX}/share/sirius")
foreach(_file
        bin/sirius-app
        share/sirius/help/load.md
        share/sirius/python/sirius_worker/__main__.py
        share/sirius/python/slurm/sirius_worker.sbatch
        share/sirius/python/workbench.py
        share/sirius/python/op_schema.json
        share/applications/sirius-app.desktop
        share/icons/hicolor/scalable/apps/sirius-app.svg
        share/icons/hicolor/48x48/apps/sirius-app.png)
    if(NOT EXISTS "${PREFIX}/${_file}")
        fail("${_file} was not installed")
    endif()
endforeach()
if(NOT IS_DIRECTORY "${_data}/plugins")
    fail("share/sirius/plugins was not installed")
endif()
if(EXISTS "${_data}/python/tests")
    fail("the worker's test suite was installed with it")
endif()
file(GLOB_RECURSE _pycache LIST_DIRECTORIES true "${_data}/*")
list(FILTER _pycache INCLUDE REGEX "/__pycache__$|\\.pyc$")
if(_pycache)
    fail("bytecode caches were installed: ${_pycache}")
endif()

find_program(_validate desktop-file-validate)
if(_validate)
    execute_process(COMMAND "${_validate}" "${PREFIX}/share/applications/sirius-app.desktop"
                    RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0 OR NOT "${_out}${_err}" STREQUAL "")
        fail("desktop-file-validate: ${_out}${_err}")
    endif()
endif()

# every shared library resolves without the build tree's RUNPATH
find_program(_ldd ldd)
if(_ldd)
    execute_process(COMMAND "${_ldd}" "${PREFIX}/bin/sirius-app" RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0 OR _out MATCHES "not found")
        fail("the installed sirius-app does not load:\n${_out}${_err}")
    endif()
    # the scratch prefix may itself sit in the build tree (the test puts it there)
    string(REPLACE "${PREFIX}" "<prefix>" _outside_prefix "${_out}")
    if(_outside_prefix MATCHES "${BUILD_DIR}")
        fail("the installed sirius-app still loads a library from the build tree:\n${_out}")
    endif()
endif()

# --- the installed application reads its own help pages ------------------------
set(_marker "installed-tree-marker-7f3a")
file(APPEND "${_data}/help/load.md" "\n<!-- ${_marker} -->\n")
set(_home "${PREFIX}/check-home")
file(MAKE_DIRECTORY "${_home}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env --unset=SIRIUS_HELP_DIR --unset=SIRIUS_WORKER_DIR
            QT_QPA_PLATFORM=offscreen QT_FORCE_STDERR_LOGGING=1
            "HOME=${_home}" "XDG_CONFIG_HOME=${_home}/.config" "SIRIUS_PYTHON=${PYTHON}"
            "${PREFIX}/bin/sirius-app" --dataset "${RAW}"
            --tool "{\"name\":\"get_help\",\"args\":{\"kind\":\"load\"}}" --quit-after 5000
    WORKING_DIRECTORY "${_home}"
    RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err TIMEOUT 120)
if(NOT _rc EQUAL 0)
    fail("the installed sirius-app exited with ${_rc}:\n${_out}${_err}")
endif()
if(NOT "${_out}${_err}" MATCHES "${_marker}")
    fail("get_help did not return the installed page (the checkout's instead?):\n${_out}${_err}")
endif()

# --- the installed worker starts ----------------------------------------------
# With stdin at end-of-file, --exit-with-parent stops it right after it listens.
if(PYTHON)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env --unset=SIRIUS_WORKBENCH_PY
                "${PYTHON}" -m sirius_worker --port 0 --exit-with-parent
        WORKING_DIRECTORY "${_data}/python"
        INPUT_FILE /dev/null
        RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err TIMEOUT 120)
    if(NOT _rc EQUAL 0 OR NOT "${_out}${_err}" MATCHES "listening on")
        fail("the installed worker did not start (${_rc}):\n${_out}${_err}")
    endif()
endif()

message(STATUS "install check passed: ${PREFIX}")
