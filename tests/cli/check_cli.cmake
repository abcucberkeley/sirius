# sirius-cli end to end, as a script or an agent drives it: one mode per CTest
# test (tests/CMakeLists.txt).
#
#   cmake -DCLI=<sirius-cli> -DSRC=<source dir> -DWORK=<scratch dir> -DMODE=<mode> -P check_cli.cmake
#
#   oneshot       one command per process: every stdout is one JSON document, the
#                 error codes and exit codes, --out, the worker commands, and the
#                 scratch directory is gone when the process has exited
#   run_contrast  run a contrast step on the test stack, then render, export, stats
#   run_bundled   the same with examples/sim_bundled.sirius.toml (SIM; slow)
#   session       tests/cli/session_basic.jsonl through `sirius-cli session`
#   mcp_legacy    tests/cli/mcp_legacy.jsonl through `sirius-cli mcp` (initialize)
#   mcp_modern    tests/cli/mcp_modern.jsonl (2026-07-28 request metadata), and
#                 the server's own scratch directory is gone after the end of input
#                 (both MCP transcripts must stay quick: see the MCP section)
#   worker_check  `sirius-cli worker check`; skipped unless $SIRIUS_PYTHON is set
#
# CLI may also be a list, a wrapper followed by the executable. The transcripts
# name the source and the scratch directory as @SRC@ and @WORK@, which are
# substituted into a copy in WORK. What comes back is never turned into a CMake
# list, since JSON holds ';' and brackets: stdout is walked line by line, each
# line is parsed with string(JSON), and the checks here stay shallow. The deep
# protocol checks, the ones that must react to what comes back, live in
# tests/cli/agent_client.py.
#
# SIRIUS_PYTHON_ENV always points into WORK, so no mode reads or changes the
# user's own Python environment, and stdin is always a file, so no command ever
# waits for a terminal. A mode that cannot run here prints "SKIPPED:", which the
# test registration treats as a skip.

cmake_minimum_required(VERSION 3.25)

foreach(_var CLI SRC WORK MODE)
    if(NOT DEFINED ${_var} OR "${${_var}}" STREQUAL "")
        message(FATAL_ERROR "check_cli.cmake: -D${_var}=... is required")
    endif()
endforeach()

set(_modes oneshot run_contrast run_bundled session mcp_legacy mcp_modern worker_check)
if(NOT MODE IN_LIST _modes)
    message(FATAL_ERROR "check_cli.cmake: unknown MODE '${MODE}' (one of: ${_modes})")
endif()

# The commands run in WORK (or SRC), not in the caller's directory, so a relative
# path given on the command line is made absolute against the directory it was
# typed in, which a script run with -P sees as CMAKE_CURRENT_SOURCE_DIR. An
# element of CLI that names no file there is left alone: it may be a program
# found on the PATH, such as a wrapper.
get_filename_component(SRC "${SRC}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
get_filename_component(WORK "${WORK}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(_cli)
foreach(_part IN LISTS CLI)
    get_filename_component(_abs "${_part}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(EXISTS "${_abs}" AND NOT IS_DIRECTORY "${_abs}")
        set(_part "${_abs}")
    endif()
    list(APPEND _cli "${_part}")
endforeach()
set(CLI "${_cli}")

set(RAW "${SRC}/tests/data/raw.tif")
set(BUNDLED "${SRC}/examples/sim_bundled.sirius.toml")

# --- helpers ------------------------------------------------------------------
# Messages take one argument: JSON in a message keeps its ';' only that way.
function(fail msg)
    message(FATAL_ERROR "cli check (${MODE}): ${msg}")
endfunction()

function(expect what actual expected)
    if(NOT "${actual}" STREQUAL "${expected}")
        fail("${what}: expected '${expected}', got '${actual}'")
    endif()
endfunction()

# The last 3000 characters of <text>: enough to see what went wrong without
# flooding the test log with a base64 image.
function(clip var text)
    set(_n 3000)
    string(LENGTH "${text}" _len)
    if(_len GREATER _n)
        math(EXPR _from "${_len} - ${_n}")
        string(SUBSTRING "${text}" ${_from} -1 text)
        set(text "...${text}")
    endif()
    set(${var} "${text}" PARENT_SCOPE)
endfunction()

# Text that holds paths, compared the way the file system does on Windows: case
# and slashes aside.
function(normal_text var text)
    string(REPLACE "\\" "/" text "${text}")
    if(CMAKE_HOST_WIN32)
        string(TOLOWER "${text}" text)
    endif()
    set(${var} "${text}" PARENT_SCOPE)
endfunction()

# A path that exists is also resolved through symbolic links first: the CLI
# resolves a relative path against its working directory, which the system
# reports as the physical path, so a checkout or a build tree reached through a
# link would otherwise differ from what CMake was given.
function(normal_path var path)
    string(REPLACE "\\" "/" path "${path}")
    if(EXISTS "${path}")
        file(REAL_PATH "${path}" path)
    endif()
    normal_text(path "${path}")
    set(${var} "${path}" PARENT_SCOPE)
endfunction()

function(expect_path what actual expected)
    normal_path(_a "${actual}")
    normal_path(_e "${expected}")
    if(NOT _a STREQUAL _e)
        fail("${what}: expected ${expected}, got ${actual}")
    endif()
endfunction()

function(expect_path_under what path dir)
    normal_path(_p "${path}")
    normal_path(_d "${dir}")
    string(FIND "${_p}" "${_d}/" _at)
    if(NOT _at EQUAL 0)
        fail("${what}: ${path} is not under ${dir}")
    endif()
endfunction()

# The value at <path...> in <json>: a string, a number, ON / OFF, or JSON text.
function(jget var json)
    string(JSON _v ERROR_VARIABLE _e GET "${json}" ${ARGN})
    if(_e)
        string(JOIN "." _path ${ARGN})
        clip(_j "${json}")
        fail("no ${_path} in ${_j}")
    endif()
    set(${var} "${_v}" PARENT_SCOPE)
endfunction()

# The JSON type at <path...> (NULL, NUMBER, STRING, BOOLEAN, ARRAY, OBJECT), or MISSING.
function(jtype var json)
    string(JSON _t ERROR_VARIABLE _e TYPE "${json}" ${ARGN})
    if(_e)
        set(_t MISSING)
    endif()
    set(${var} "${_t}" PARENT_SCOPE)
endfunction()

function(jlength var json)
    string(JSON _n ERROR_VARIABLE _e LENGTH "${json}" ${ARGN})
    if(_e)
        string(JOIN "." _path ${ARGN})
        clip(_j "${json}")
        fail("${_path} is not an array or an object in ${_j}")
    endif()
    set(${var} "${_n}" PARENT_SCOPE)
endfunction()

# jexpect(<json> <expected> <path...>): the value at <path> is <expected>. Booleans read ON / OFF.
function(jexpect json expected)
    jget(_v "${json}" ${ARGN})
    if(NOT "${_v}" STREQUAL "${expected}")
        string(JOIN "." _path ${ARGN})
        clip(_j "${json}")
        fail("${_path} is '${_v}', expected '${expected}' in ${_j}")
    endif()
endfunction()

function(jexpect_type json expected)
    jtype(_t "${json}" ${ARGN})
    if(NOT _t STREQUAL expected)
        string(JOIN "." _path ${ARGN})
        clip(_j "${json}")
        fail("${_path} is ${_t}, expected ${expected} in ${_j}")
    endif()
endfunction()

# jexpect_positive(<json> <path...>): a number greater than zero.
function(jexpect_positive json)
    jexpect_type("${json}" NUMBER ${ARGN})
    jget(_v "${json}" ${ARGN})
    if(NOT _v GREATER 0)
        string(JOIN "." _path ${ARGN})
        fail("${_path} is ${_v}, expected a positive number")
    endif()
endfunction()

# jexpect_match(<json> <regex> <path...>): the string at <path> matches.
function(jexpect_match json regex)
    jexpect_type("${json}" STRING ${ARGN})
    jget(_v "${json}" ${ARGN})
    if(NOT _v MATCHES "${regex}")
        string(JOIN "." _path ${ARGN})
        fail("${_path} is '${_v}', which does not match ${regex}")
    endif()
endfunction()

# jarray_has(<json> <member> <value> <path...>): the array at <path> holds an
# object whose <member> is <value> (or, with <member> "-", the value itself).
# Sets FOUND_INDEX in the caller.
function(jarray_has json member value)
    jexpect_type("${json}" ARRAY ${ARGN})
    jlength(_n "${json}" ${ARGN})
    set(_found -1)
    if(_n GREATER 0)
        math(EXPR _last "${_n} - 1")
        foreach(_i RANGE ${_last})
            if(member STREQUAL "-")
                string(JSON _v ERROR_VARIABLE _e GET "${json}" ${ARGN} ${_i})
            else()
                string(JSON _v ERROR_VARIABLE _e GET "${json}" ${ARGN} ${_i} ${member})
            endif()
            if(NOT _e AND "${_v}" STREQUAL "${value}")
                set(_found ${_i})
                break()
            endif()
        endforeach()
    endif()
    if(_found EQUAL -1)
        string(JOIN "." _path ${ARGN})
        clip(_j "${json}")
        fail("${_path} has no entry with ${member} = ${value} in ${_j}")
    endif()
    set(FOUND_INDEX ${_found} PARENT_SCOPE)
endfunction()

function(expect_png file)
    if(NOT EXISTS "${file}")
        fail("${file} was not written")
    endif()
    file(READ "${file}" _hex LIMIT 8 HEX)
    if(NOT _hex STREQUAL "89504e470d0a1a0a")
        fail("${file} is not a PNG (it starts with ${_hex})")
    endif()
endfunction()

# The width and height in a PNG's IHDR chunk.
function(png_size file wvar hvar)
    expect_png("${file}")
    file(READ "${file}" _hex OFFSET 16 LIMIT 8 HEX)
    string(SUBSTRING "${_hex}" 0 8 _w)
    string(SUBSTRING "${_hex}" 8 8 _h)
    math(EXPR _w "0x${_w}")
    math(EXPR _h "0x${_h}")
    set(${wvar} ${_w} PARENT_SCOPE)
    set(${hvar} ${_h} PARENT_SCOPE)
endfunction()

function(expect_tiff file)
    if(NOT EXISTS "${file}" OR IS_DIRECTORY "${file}")
        fail("${file} was not written")
    endif()
    file(READ "${file}" _hex LIMIT 4 HEX)
    if(NOT _hex MATCHES "^(49492a00|4d4d002a|49492b00|4d4d002b)$")
        fail("${file} is not a TIFF (it starts with ${_hex})")
    endif()
endfunction()

# cli(<name> [INPUT <file>] [TIMEOUT <s>] [WORKDIR <dir>] ARGS <arg>...)
# Runs sirius-cli; sets <name>_rc, <name>_out, <name>_err and <name>_cmd in the caller.
function(cli name)
    cmake_parse_arguments(PARSE_ARGV 1 C "" "INPUT;TIMEOUT;WORKDIR" "ARGS")
    if(NOT C_INPUT)
        set(C_INPUT "${WORK}/stdin-empty.txt")
    endif()
    if(NOT C_TIMEOUT)
        set(C_TIMEOUT 120)
    endif()
    if(NOT C_WORKDIR)
        set(C_WORKDIR "${WORK}")
    endif()
    string(JOIN " " _cmd ${C_ARGS})
    execute_process(
        COMMAND ${CLI} ${C_ARGS}
        WORKING_DIRECTORY "${C_WORKDIR}"
        INPUT_FILE "${C_INPUT}"
        RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err
        TIMEOUT ${C_TIMEOUT})
    message(STATUS "sirius-cli ${_cmd} -> ${_rc}")
    set(${name}_rc "${_rc}" PARENT_SCOPE)
    set(${name}_out "${_out}" PARENT_SCOPE)
    set(${name}_err "${_err}" PARENT_SCOPE)
    set(${name}_cmd "sirius-cli ${_cmd}" PARENT_SCOPE)
endfunction()

# What a failed check prints about the command it ran.
function(ran var name)
    clip(_out "${${name}_out}")
    clip(_err "${${name}_err}")
    set(${var} "${${name}_cmd}\nexit: ${${name}_rc}\nstdout:\n${_out}\nstderr:\n${_err}" PARENT_SCOPE)
endfunction()

function(expect_exit name code)
    if(NOT "${${name}_rc}" STREQUAL "${code}")
        ran(_ran ${name})
        fail("expected exit ${code}\n${_ran}")
    endif()
endfunction()

# The one-shot envelope, {"ok", "schema", "command", "result" | "error",
# "warnings"}: stdout is exactly one JSON object on one line (stdout is not a
# terminal, so the output is compact). Sets <name>_doc in the caller.
function(envelope name)
    set(_out "${${name}_out}")
    string(LENGTH "${_out}" _len)
    string(FIND "${_out}" "\n" _first)
    math(EXPR _last "${_len} - 1")
    if(_len EQUAL 0 OR NOT _first EQUAL _last)
        ran(_ran ${name})
        fail("stdout is not one line ending in a newline\n${_ran}")
    endif()
    string(SUBSTRING "${_out}" 0 ${_last} _doc)
    string(JSON _type ERROR_VARIABLE _e TYPE "${_doc}")
    if(_e OR NOT _type STREQUAL "OBJECT")
        ran(_ran ${name})
        fail("stdout is not a JSON object (${_e})\n${_ran}")
    endif()
    jexpect("${_doc}" "sirius-cli/1" schema)
    jtype(_command "${_doc}" command)                # null when no command word was recognised
    if(NOT _command MATCHES "^(STRING|NULL)$")
        fail("the envelope's command is ${_command}: ${_doc}")
    endif()
    jexpect_type("${_doc}" ARRAY warnings)
    set(${name}_doc "${_doc}" PARENT_SCOPE)
endfunction()

# expect_ok(<name>): exit 0 and {"ok":true, "result":...}. Sets <name>_doc and <name>_result.
function(expect_ok name)
    expect_exit(${name} 0)
    envelope(${name})
    jexpect("${${name}_doc}" ON ok)
    jget(_result "${${name}_doc}" result)
    set(${name}_doc "${${name}_doc}" PARENT_SCOPE)
    set(${name}_result "${_result}" PARENT_SCOPE)
endfunction()

# expect_error(<name> <exit> <code>): the failure envelope, its exit_code, and
# the one summary line on stderr. Sets <name>_doc in the caller.
function(expect_error name exit code)
    expect_exit(${name} ${exit})
    envelope(${name})
    set(_doc "${${name}_doc}")
    jexpect("${_doc}" OFF ok)
    jexpect("${_doc}" "${code}" error code)
    jexpect_type("${_doc}" STRING error message)
    jexpect("${_doc}" "${exit}" exit_code)
    if(NOT "${${name}_err}" MATCHES "sirius-cli: ${code}: ")
        ran(_ran ${name})
        fail("no 'sirius-cli: ${code}: ...' summary line on stderr\n${_ran}")
    endif()
    set(${name}_doc "${_doc}" PARENT_SCOPE)
endfunction()

# A transcript from tests/cli with @SRC@ and @WORK@ filled in, LF line endings
# whatever the checkout's are.
function(transcript var name)
    set(_copy "${WORK}/${name}.jsonl")
    configure_file("${SRC}/tests/cli/${name}.jsonl" "${_copy}" @ONLY NEWLINE_STYLE UNIX)
    set(${var} "${_copy}" PARENT_SCOPE)
endfunction()

# Splits a protocol stream into LINE_0 ... LINE_<n-1> and LINE_COUNT in the
# caller, each checked to be one JSON object.
function(split_lines name)
    set(_text "${${name}_out}")
    set(_n 0)
    while(NOT _text STREQUAL "")
        string(FIND "${_text}" "\n" _at)
        if(_at EQUAL -1)
            ran(_ran ${name})
            fail("the last line on stdout has no newline\n${_ran}")
        endif()
        string(SUBSTRING "${_text}" 0 ${_at} _line)
        math(EXPR _next "${_at} + 1")
        string(SUBSTRING "${_text}" ${_next} -1 _text)
        string(JSON _type ERROR_VARIABLE _e TYPE "${_line}")
        if(_e OR NOT _type STREQUAL "OBJECT")
            ran(_ran ${name})
            fail("stdout line ${_n} is not a JSON object: ${_line}\n${_ran}")
        endif()
        set(LINE_${_n} "${_line}" PARENT_SCOPE)
        math(EXPR _n "${_n} + 1")
    endwhile()
    set(LINE_COUNT ${_n} PARENT_SCOPE)
endfunction()

# response(<var> <id> [STRING]): the one line whose "id" is <id> (a number, or
# a string with STRING) and that is not an event. Sets <var> and <var>_at (its
# line number) in the caller.
function(response var id)
    set(_want NUMBER)
    if(ARGN STREQUAL "STRING")
        set(_want STRING)
    endif()
    set(_found -1)
    if(LINE_COUNT GREATER 0)
        math(EXPR _last "${LINE_COUNT} - 1")
        foreach(_i RANGE ${_last})
            jtype(_t "${LINE_${_i}}" id)
            jtype(_event "${LINE_${_i}}" event)
            if(_t STREQUAL _want AND _event STREQUAL "MISSING")
                jget(_v "${LINE_${_i}}" id)
                if(_v STREQUAL id)
                    if(NOT _found EQUAL -1)
                        fail("request ${id} was answered twice:\n${LINE_${_found}}\n${LINE_${_i}}")
                    endif()
                    set(_found ${_i})
                endif()
            endif()
        endforeach()
    endif()
    if(_found EQUAL -1)
        fail("request ${id} was not answered")
    endif()
    set(${var} "${LINE_${_found}}" PARENT_SCOPE)
    set(${var}_at ${_found} PARENT_SCOPE)
endfunction()

# The line numbers of the responses without an id (or with a null one): the
# answers to lines that could not be read as requests. Sets <var> and <var>_count.
function(idless_responses var)
    set(_at)
    if(LINE_COUNT GREATER 0)
        math(EXPR _last "${LINE_COUNT} - 1")
        foreach(_i RANGE ${_last})
            jtype(_t "${LINE_${_i}}" id)
            jtype(_event "${LINE_${_i}}" event)
            if((_t STREQUAL "MISSING" OR _t STREQUAL "NULL") AND _event STREQUAL "MISSING")
                list(APPEND _at ${_i})
            endif()
        endforeach()
    endif()
    list(LENGTH _at _n)
    set(${var} "${_at}" PARENT_SCOPE)
    set(${var}_count ${_n} PARENT_SCOPE)
endfunction()

# --- the working directory ------------------------------------------------------
# WORK is this mode's own directory. It is emptied first, but only when it is
# new, empty or one this script made, so a mistyped -DWORK deletes nothing else
# (a file of that name included).
set(_stamp "${WORK}/.sirius-cli-check")
if(EXISTS "${WORK}" AND NOT IS_DIRECTORY "${WORK}")
    fail("${WORK} is a file, not a directory; refusing to replace it")
endif()
if(IS_DIRECTORY "${WORK}" AND NOT EXISTS "${_stamp}")
    file(GLOB _existing LIST_DIRECTORIES true "${WORK}/*")
    if(_existing)
        fail("${WORK} is not empty and was not made by this script; refusing to clear it")
    endif()
endif()
file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")
file(WRITE "${_stamp}" "sirius-cli end-to-end scratch; tests/cli/check_cli.cmake empties it on every run\n")
file(WRITE "${WORK}/stdin-empty.txt" "")
set(ENV{SIRIUS_PYTHON_ENV} "${WORK}/pyenv")

# --- oneshot ------------------------------------------------------------------
function(mode_oneshot)
    # version: the envelope and what the executable speaks
    cli(v ARGS version)
    expect_ok(v)
    jexpect("${v_doc}" version command)
    jexpect("${v_result}" sirius-cli name)
    jexpect("${v_result}" sirius-cli/1 schema)
    jexpect_type("${v_result}" STRING version)
    jexpect("${v_result}" sirius-session/1 protocols session)
    jarray_has("${v_result}" - 2026-07-28 protocols mcp)
    jarray_has("${v_result}" - 2025-11-25 protocols mcp)
    cli(v ARGS --version)
    expect_ok(v)
    # --pretty indents the same document over several lines
    cli(v ARGS --pretty version)
    expect_exit(v 0)
    string(JSON _type ERROR_VARIABLE _e TYPE "${v_out}")
    string(REGEX MATCHALL "\n" _newlines "${v_out}")
    list(LENGTH _newlines _newlines)
    if(_e OR NOT _type STREQUAL "OBJECT" OR _newlines LESS 3)
        ran(_ran v)
        fail("--pretty version did not print an indented JSON object\n${_ran}")
    endif()

    # tools: the MCP tool list; tools --names: just the names
    cli(t ARGS tools)
    expect_ok(t)
    jarray_has("${t_result}" name open_dataset)
    jexpect("${t_result}" object ${FOUND_INDEX} inputSchema type)
    jarray_has("${t_result}" name render)
    jtype(_out "${t_result}" ${FOUND_INDEX} inputSchema properties out)
    if(NOT _out STREQUAL "MISSING")
        fail("render has an 'out' argument: images go to the scratch directory only")
    endif()
    cli(t ARGS tools --names)
    expect_ok(t)
    jarray_has("${t_result}" - open_dataset)

    # schema: commands, exit codes and which error code exits how
    cli(s ARGS schema)
    expect_ok(s)
    jexpect_type("${s_result}" ARRAY commands)
    jexpect_type("${s_result}" OBJECT exit_codes)
    jlength(_n "${s_result}" exit_codes)
    if(_n LESS 8)
        fail("schema lists ${_n} exit codes, expected the 8 of the envelope")
    endif()
    jexpect("${s_result}" 2 error_codes usage)
    jexpect("${s_result}" 3 error_codes not_found)
    jexpect("${s_result}" 4 error_codes worker_unavailable)
    jexpect("${s_result}" 5 error_codes consent_required)
    jexpect("${s_result}" 124 error_codes timeout)
    jexpect("${s_result}" 130 error_codes cancelled)

    # info: a relative path resolves against the working directory and comes back absolute
    cli(i WORKDIR "${SRC}" ARGS info tests/data/raw.tif)
    expect_ok(i)
    jexpect_positive("${i_result}" dims z)
    jexpect_positive("${i_result}" dims x)
    jexpect_type("${i_result}" STRING dtype)
    jget(_path "${i_result}" path)
    expect_path("info path" "${_path}" "${RAW}")
    if(_path MATCHES "\\\\")
        fail("info reports the path with backslashes: ${_path}")
    endif()

    # ops and one operation in detail
    cli(o ARGS ops)
    expect_ok(o)
    jarray_has("${o_result}" kind contrast operations)
    jarray_has("${o_result}" kind sim operations)
    cli(o ARGS ops sim --detail)
    expect_ok(o)
    jexpect("${o_result}" sim kind)
    jexpect_type("${o_result}" ARRAY params)
    jlength(_n "${o_result}" params)
    if(_n EQUAL 0)
        fail("ops sim --detail lists no parameters")
    endif()

    # help: the page list, and one page as Markdown (not JSON)
    cli(h ARGS help)
    expect_ok(h)
    jarray_has("${h_result}" page load pages)
    cli(h ARGS help load --markdown)
    expect_exit(h 0)
    if(NOT h_out MATCHES "#")
        ran(_ran h)
        fail("help load --markdown printed no Markdown heading\n${_ran}")
    endif()

    # devices: CPU always works
    cli(d ARGS devices)
    expect_ok(d)
    jexpect_type("${d_result}" BOOLEAN cuda_available)
    jexpect_type("${d_result}" ARRAY devices)

    # validate: the bundled pipeline is valid; a step that cannot run is exit 3
    cli(va ARGS validate --pipeline "${BUNDLED}")
    expect_ok(va)
    jexpect("${va_result}" ON ok)
    cli(va ARGS --dataset "${RAW}" --steps "[{\"kind\":\"sim\",\"params\":{\"mode\":\"Manual\"}}]" validate)
    expect_error(va 3 validation)

    # render --out writes the PNG where it is told...
    cli(r ARGS render --dataset "${RAW}" --no-run --out "${WORK}/xy.png")
    expect_ok(r)
    jget(_path "${r_result}" path)
    expect_path("render path" "${_path}" "${WORK}/xy.png")
    png_size("${WORK}/xy.png" _w _h)
    jexpect("${r_result}" ${_w} width)
    jexpect("${r_result}" ${_h} height)
    # ... but never over a file that is not an image
    file(WRITE "${WORK}/note.txt" "keep me\n")
    cli(r ARGS render --dataset "${RAW}" --no-run --out "${WORK}/note.txt")
    expect_error(r 3 invalid_argument)
    file(READ "${WORK}/note.txt" _note)
    if(NOT _note STREQUAL "keep me\n")
        fail("render --out overwrote ${WORK}/note.txt")
    endif()

    # stats of the Load step, without running anything
    cli(st ARGS stats --dataset "${RAW}" --no-run)
    expect_ok(st)
    jexpect_type("${st_result}" ARRAY channels)
    jexpect_type("${st_result}" NUMBER channels 0 min)
    jexpect_type("${st_result}" NUMBER channels 0 max)
    jexpect_type("${st_result}" NUMBER channels 0 mean)

    # call: state options before `call`; a new workspace is Load only
    cli(c ARGS --dataset "${RAW}" call get_state)
    expect_ok(c)
    jexpect_match("${c_result}" "^ws_[0-9a-f]+$" workspace)
    jexpect_positive("${c_result}" dataset dims z)
    jlength(_n "${c_result}" steps)
    expect("steps of a new workspace" "${_n}" 1)
    jexpect("${c_result}" OFF running)
    # a one-shot command ends like the servers: the scratch directory it made
    # in the temporary folder for itself is gone once it has exited
    jexpect_match("${c_result}" "." scratch)
    jget(_scratch "${c_result}" scratch)
    if(EXISTS "${_scratch}")
        fail("call get_state left its scratch directory behind: ${_scratch}")
    endif()

    # errors: their codes and exit codes
    cli(e ARGS info "${WORK}/missing.tif")
    expect_error(e 3 not_found)
    cli(e ARGS frobnicate)
    expect_error(e 2 usage)
    cli(e ARGS call get_help --page ../x)
    expect_error(e 3 invalid_argument)
    cli(e ARGS call no_such_tool)
    expect_error(e 2 unknown_tool)

    # the worker commands, against the empty environment in WORK: nothing is
    # downloaded without consent, and stdin is not a terminal. The CLI makes
    # SIRIUS_PYTHON_ENV absolute without resolving links, and the directory does
    # not exist, so it is looked for as given.
    normal_text(_pyenv "${WORK}/pyenv")
    cli(w ARGS worker setup)
    if(w_rc STREQUAL "4")
        expect_error(w 4 python_not_found)
    else()
        expect_error(w 5 consent_required)
        jexpect_type("${w_doc}" OBJECT error data)
        normal_text(_text "${w_out}")
        string(FIND "${_text}" "${_pyenv}" _at)
        if(_at EQUAL -1)
            fail("the setup plan does not name SIRIUS_PYTHON_ENV (${WORK}/pyenv):\n${w_out}")
        endif()
    endif()
    cli(w ARGS worker setup --dry-run)
    if(w_rc STREQUAL "4")
        expect_error(w 4 python_not_found)
    else()
        expect_ok(w)
        normal_text(_text "${w_out}")
        string(FIND "${_text}" "${_pyenv}" _at)
        if(_at EQUAL -1)
            fail("the dry run does not plan for SIRIUS_PYTHON_ENV (${WORK}/pyenv):\n${w_out}")
        endif()
    endif()
    cli(w ARGS worker status)
    expect_ok(w)
    jexpect_type("${w_result}" OBJECT interpreter)
    jexpect_type("${w_result}" OBJECT environment)
    jget(_env "${w_result}" environment)
    normal_text(_text "${_env}")
    string(FIND "${_text}" "${_pyenv}" _at)
    if(_at EQUAL -1)
        fail("worker status does not report SIRIUS_PYTHON_ENV (${WORK}/pyenv):\n${w_out}")
    endif()
    if(EXISTS "${WORK}/pyenv")
        fail("a worker command created ${WORK}/pyenv without consent")
    endif()
endfunction()

# --- run_contrast / run_bundled ---------------------------------------------------
# check_run(<timeout> <state options>...): run, then render the projection,
# export OME-TIFF and take statistics, in one process. Sets RUN_RESULT in the caller.
function(check_run timeout)
    cli(run TIMEOUT ${timeout} ARGS --backend cpu --plugins off run ${ARGN}
        --render "${WORK}/mip.png" --plane mip --export "${WORK}/out.ome.tif" --stats)
    expect_ok(run)
    jexpect("${run_doc}" run command)
    jexpect("${run_result}" succeeded run status)
    jexpect_type("${run_result}" ARRAY run steps)
    jlength(_n "${run_result}" renders)
    expect("renders" "${_n}" 1)
    jget(_path "${run_result}" renders 0 path)
    expect_path("render path" "${_path}" "${WORK}/mip.png")
    png_size("${WORK}/mip.png" _w _h)
    jexpect("${run_result}" ${_w} renders 0 width)
    jexpect("${run_result}" ${_h} renders 0 height)
    jlength(_n "${run_result}" exports)
    expect("exports" "${_n}" 1)
    expect_tiff("${WORK}/out.ome.tif")
    jexpect_type("${run_result}" ARRAY statistics channels)
    set(RUN_RESULT "${run_result}" PARENT_SCOPE)
endfunction()

function(mode_run_contrast)
    check_run(240 --dataset "${RAW}" --steps "[{\"kind\":\"contrast\"}]")
    jarray_has("${RUN_RESULT}" kind contrast run steps)
    jexpect("${RUN_RESULT}" ran run steps ${FOUND_INDEX} state)
endfunction()

function(mode_run_bundled)
    check_run(840 --pipeline "${BUNDLED}")
    jarray_has("${RUN_RESULT}" kind sim run steps)
    jexpect("${RUN_RESULT}" ran run steps ${FOUND_INDEX} state)
    # the reconstruction doubles x and y and folds the 15 phases into each plane
    jexpect("${RUN_RESULT}" 9 run output dims z)
    jexpect("${RUN_RESULT}" 128 run output dims x)
endfunction()

# --- session ------------------------------------------------------------------
function(mode_session)
    transcript(_script session_basic)
    file(MAKE_DIRECTORY "${WORK}/scratch")
    cli(s INPUT "${_script}" TIMEOUT 240
        ARGS --backend cpu --plugins off --scratch "${WORK}/scratch" --keep-scratch session)
    expect_exit(s 0)
    split_lines(s)

    # the ready event comes first
    if(LINE_COUNT EQUAL 0)
        ran(_ran s)
        fail("the session printed nothing\n${_ran}")
    endif()
    set(_ready "${LINE_0}")
    jtype(_t "${_ready}" event)
    if(NOT _t STREQUAL "STRING")
        fail("the first line is not the ready event: ${_ready}")
    endif()
    jexpect("${_ready}" ready event)
    jexpect("${_ready}" sirius-session/1 protocol)
    jexpect_match("${_ready}" "^ws_[0-9a-f]+$" workspace)
    jexpect_positive("${_ready}" tools)
    jget(_ws "${_ready}" workspace)
    jget(_tools "${_ready}" tools)

    # every request is answered once, and everything the main thread answers
    # (all but ping, status and cancel, which the reader answers at once) in the
    # order it was sent, the unknown method and the control methods included
    set(_ok 1 2 3 4 5 6 7 8 9 14 15 16)
    set(_failed 10 11 12)
    foreach(_id IN LISTS _ok)
        response(_r ${_id})
        jexpect("${_r}" ON ok)
    endforeach()
    foreach(_id IN LISTS _failed)
        response(_r ${_id})
        jexpect("${_r}" OFF ok)
    endforeach()
    set(_previous -1)
    foreach(_id 3 4 5 6 7 8 9 10 11 12 s-13 14 15 16)
        if(_id MATCHES "^[0-9]+$")
            response(_r ${_id})
        else()
            response(_r ${_id} STRING)
        endif()
        if(NOT _r_at GREATER _previous)
            fail("request ${_id} was answered out of order")
        endif()
        set(_previous ${_r_at})
    endforeach()

    response(_r 1)                                   # ping
    response(_r 2)                                   # status (with a tolerated "jsonrpc")
    jexpect_type("${_r}" BOOLEAN result running)
    jexpect("${_r}" "${_ws}" result workspace)
    response(_r 3)                                   # open_dataset
    jexpect("${_r}" "${_ws}" result workspace)
    jexpect_positive("${_r}" result dims z)
    response(_r 4)                                   # get_state: Load only
    jexpect("${_r}" "${_ws}" result workspace)
    jlength(_n "${_r}" result steps)
    expect("steps" "${_n}" 1)
    jget(_scratch "${_r}" result scratch)
    expect_path("get_state scratch" "${_scratch}" "${WORK}/scratch")
    response(_r 5)                                   # add_step: an undoable change
    jexpect("${_r}" ON undoable)
    jexpect_type("${_r}" ARRAY changes)
    jlength(_n "${_r}" changes)
    if(_n EQUAL 0)
        fail("add_step reported no change: ${_r}")
    endif()
    response(_r 6)                                   # undo
    jexpect_type("${_r}" BOOLEAN undoable)
    jexpect_match("${_r}" "." result undone)
    response(_r 7)                                   # render: a file in the scratch directory
    jexpect("${_r}" image/png result image mime_type)
    jget(_png "${_r}" result image path)
    expect_path_under("render" "${_png}" "${WORK}/scratch/renders")
    png_size("${_png}" _w _h)
    jexpect("${_r}" ${_w} result image width)
    jexpect("${_r}" ${_h} result image height)
    response(_r 8)                                   # render inline
    jexpect_match("${_r}" "^iVBORw0KGgo" result image base64)
    response(_r 9)                                   # statistics
    jexpect_type("${_r}" ARRAY result channels)
    response(_r 10)                                  # a step that does not exist
    jexpect("${_r}" unknown_step error code)
    response(_r 11)
    jexpect("${_r}" unknown_method error code)
    response(_r 12)
    jexpect("${_r}" stale_workspace error code)
    response(_r s-13 STRING)                         # tools: the ids keep their JSON type
    jexpect("${_r}" ON ok)
    jlength(_n "${_r}" result tools)
    expect("tools in the ready event" "${_tools}" "${_n}")
    response(_r 15)                                  # load_pipeline
    jlength(_n "${_r}" result steps)
    expect("bundled pipeline steps" "${_n}" 4)
    response(_r 16)                                  # run, not waited for
    jexpect("${_r}" running result status)
    jget(_run "${_r}" result run_id)
    set(_run_at ${_r_at})

    # the lines that were not requests: one parse error, one missing id
    idless_responses(_idless)
    expect("responses without an id" "${_idless_count}" 2)
    set(_codes)
    foreach(_i IN LISTS _idless)
        jexpect("${LINE_${_i}}" OFF ok)
        jget(_code "${LINE_${_i}}" error code)
        list(APPEND _codes ${_code})
    endforeach()
    list(SORT _codes)
    expect("protocol errors" "${_codes}" "invalid_request;parse_error")

    # end of input waited for the run and reported it; no log events, as none
    # were subscribed to; nothing else on stdout: 16 requests with an id and the
    # two lines above
    set(_finished -1)
    set(_responses 0)
    math(EXPR _last "${LINE_COUNT} - 1")
    foreach(_i RANGE 1 ${_last})
        jtype(_t "${LINE_${_i}}" event)
        if(_t STREQUAL "MISSING")
            math(EXPR _responses "${_responses} + 1")
            continue()
        endif()
        jget(_event "${LINE_${_i}}" event)
        if(_event STREQUAL "run_finished")
            if(NOT _finished EQUAL -1)
                fail("two run_finished events")
            endif()
            set(_finished ${_i})
            jexpect("${LINE_${_i}}" "${_run}" run_id)
            jexpect("${LINE_${_i}}" succeeded result status)
        elseif(NOT _event STREQUAL "progress")
            fail("unexpected event: ${LINE_${_i}}")
        endif()
    endforeach()
    expect("responses on stdout" "${_responses}" 18)
    if(_finished EQUAL -1)
        fail("end of input did not wait for run ${_run}: no run_finished event")
    endif()
    if(NOT _finished GREATER _run_at)
        fail("run_finished came before the run's own response")
    endif()
endfunction()

# --- MCP ------------------------------------------------------------------------
# The MCP transcripts must stay quick. stdin is a file, so the end of input
# comes at once, and the server answers what was sent before it for a fixed
# grace only (10 s, kEofGrace in app/core/agent_mcp.cpp; --timeout sets it for
# the session alone). A request still queued after that is dropped without an
# answer, which would show here as "request N was not answered". So no run in
# mcp_legacy.jsonl or mcp_modern.jsonl (a JSON-lines file has no room for this
# note): anything slow belongs in tests/cli/agent_client.py, which waits for
# each answer before it closes stdin.

# Every line is JSON-RPC 2.0, and every response is to a request of the transcript.
function(check_jsonrpc expected_lines)
    if(NOT LINE_COUNT EQUAL expected_lines)
        set(_all "")
        math(EXPR _last "${LINE_COUNT} - 1")
        if(LINE_COUNT GREATER 0)
            foreach(_i RANGE ${_last})
                string(APPEND _all "${LINE_${_i}}\n")
            endforeach()
        endif()
        clip(_all "${_all}")
        fail("expected ${expected_lines} lines on stdout, got ${LINE_COUNT} (a notification answered?):\n${_all}")
    endif()
    math(EXPR _last "${LINE_COUNT} - 1")
    foreach(_i RANGE ${_last})
        jexpect("${LINE_${_i}}" 2.0 jsonrpc)
    endforeach()
endfunction()

function(mode_mcp_legacy)
    transcript(_script mcp_legacy)                   # quick requests only: see above
    file(MAKE_DIRECTORY "${WORK}/scratch")
    cli(m INPUT "${_script}" TIMEOUT 240
        ARGS --backend cpu --plugins off --scratch "${WORK}/scratch" --keep-scratch mcp)
    expect_exit(m 0)
    split_lines(m)
    # ids 1-7, "s-8" and 11; one answer each to the batch and the broken line;
    # nothing for the notifications or the batch's own request
    check_jsonrpc(11)

    response(_r 1)                                   # initialize
    jexpect("${_r}" 2025-11-25 result protocolVersion)
    jexpect_type("${_r}" OBJECT result capabilities tools)
    jexpect("${_r}" sirius result serverInfo name)
    jexpect_type("${_r}" STRING result serverInfo title)
    jexpect_type("${_r}" STRING result serverInfo version)
    jexpect_match("${_r}" "worker setup" result instructions)

    response(_r 2)                                   # tools/list
    jarray_has("${_r}" name open_dataset result tools)
    set(_open ${FOUND_INDEX})
    jexpect_type("${_r}" OBJECT result tools ${_open} annotations)
    jexpect_type("${_r}" BOOLEAN result tools ${_open} annotations readOnlyHint)
    jexpect("${_r}" object result tools ${_open} inputSchema type)
    jexpect("${_r}" OFF result tools ${_open} inputSchema additionalProperties)
    jexpect_type("${_r}" OBJECT result tools ${_open} inputSchema properties path)
    jexpect_type("${_r}" STRING result tools ${_open} title)
    jtype(_t "${_r}" result nextCursor)
    if(NOT _t STREQUAL "MISSING")
        fail("tools/list has a nextCursor; every tool fits one page")
    endif()

    response(_r 3)                                   # open_dataset
    jexpect("${_r}" OFF result isError)
    jexpect_positive("${_r}" result structuredContent dims z)
    jexpect("${_r}" text result content 0 type)

    response(_r 4)                                   # render: a caption and the image itself
    jexpect("${_r}" OFF result isError)
    jlength(_n "${_r}" result content)
    expect("render content items" "${_n}" 2)
    jexpect("${_r}" text result content 0 type)
    jexpect("${_r}" image result content 1 type)
    jexpect("${_r}" image/png result content 1 mimeType)
    jexpect_match("${_r}" "^iVBORw0KGgo" result content 1 data)
    jget(_caption "${_r}" result content 0 text)
    jget(_png "${_caption}" path)
    expect_path_under("render" "${_png}" "${WORK}/scratch/renders")
    png_size("${_png}" _w _h)
    jexpect("${_caption}" ${_w} width)
    jexpect("${_caption}" ${_h} height)

    response(_r 5)                                   # get_step 99: a tool error the model can fix
    jexpect("${_r}" ON result isError)
    jexpect("${_r}" unknown_step result structuredContent error code)
    jexpect_match("${_r}" "^unknown_step: " result content 0 text)

    response(_r 6)                                   # an unknown tool: a protocol error
    jexpect("${_r}" -32602 error code)
    jtype(_t "${_r}" result)
    if(NOT _t STREQUAL "MISSING")
        fail("an error response has a result: ${_r}")
    endif()

    response(_r 7)                                   # ping
    jexpect_type("${_r}" OBJECT result)
    jlength(_n "${_r}" result)
    expect("ping result members" "${_n}" 0)

    response(_r s-8 STRING)                          # unsupported method; a string id stays one
    jexpect("${_r}" -32601 error code)

    response(_r 11)                                  # arguments that are not an object
    jexpect("${_r}" -32602 error code)

    idless_responses(_idless)                        # the batch and the broken line
    expect("responses without an id" "${_idless_count}" 2)
    set(_codes)
    foreach(_i IN LISTS _idless)
        jget(_code "${LINE_${_i}}" error code)
        list(APPEND _codes ${_code})
    endforeach()
    list(SORT _codes)
    expect("id-less protocol errors" "${_codes}" "-32600;-32700")
endfunction()

function(mode_mcp_modern)
    transcript(_script mcp_modern)                   # quick requests only: see above
    cli(m INPUT "${_script}" TIMEOUT 240 ARGS --backend cpu --plugins off mcp)
    expect_exit(m 0)
    split_lines(m)
    check_jsonrpc(7)
    set(_server "io.modelcontextprotocol/serverInfo")

    response(_r 1)                                   # server/discover
    jexpect("${_r}" complete result resultType)
    jarray_has("${_r}" - 2026-07-28 result supportedVersions)
    jarray_has("${_r}" - 2025-11-25 result supportedVersions)
    jexpect_positive("${_r}" result ttlMs)
    jexpect_type("${_r}" OBJECT result capabilities tools)
    jexpect_type("${_r}" STRING result instructions)
    jexpect("${_r}" sirius result _meta ${_server} name)

    response(_r 2)                                   # tools/list
    jexpect("${_r}" complete result resultType)
    jexpect_positive("${_r}" result ttlMs)
    jarray_has("${_r}" name open_dataset result tools)

    response(_r 3)                                   # tools/call with request metadata
    jexpect("${_r}" complete result resultType)
    jexpect("${_r}" sirius result _meta ${_server} name)
    jexpect("${_r}" OFF result isError)
    jexpect_match("${_r}" "^ws_[0-9a-f]+$" result structuredContent workspace)
    # the server made its own scratch directory; the end of input removed it
    jexpect_match("${_r}" "." result structuredContent scratch)
    jget(_scratch "${_r}" result structuredContent scratch)
    if(EXISTS "${_scratch}")
        fail("the end of input left the scratch directory behind: ${_scratch}")
    endif()

    response(_r 4)                                   # a version the server does not speak
    jexpect("${_r}" -32022 error code)
    jexpect("${_r}" 1900-01-01 error data requested)
    jarray_has("${_r}" - 2026-07-28 error data supported)

    response(_r 5)                                   # no metadata and no initialize
    jexpect("${_r}" -32602 error code)

    response(_r 6)                                   # subscriptions/listen is not offered
    jexpect("${_r}" -32601 error code)

    response(_r 7)                                   # ping
    jexpect_type("${_r}" OBJECT result)
endfunction()

# --- worker_check ---------------------------------------------------------------
function(mode_worker_check)
    if("$ENV{SIRIUS_PYTHON}" STREQUAL "")
        message("SKIPPED: set SIRIUS_PYTHON to an interpreter that has numpy to start the worker")
        return()
    endif()
    cli(w ARGS worker check)
    expect_ok(w)
    jexpect_match("${w_result}" "." interpreter path)
    jexpect_match("${w_result}" "." capabilities version)
    jexpect_type("${w_result}" NUMBER seconds)
    if(EXISTS "${WORK}/pyenv")
        fail("worker check created ${WORK}/pyenv")
    endif()
endfunction()

# --- dispatch -------------------------------------------------------------------
cmake_language(CALL mode_${MODE})
message(STATUS "cli check passed: ${MODE}")
