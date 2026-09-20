# Units: the library and the application core as a graph of small targets.
#
# A *unit* is one header (or a few that belong together), its sources and its
# tests -- `buffer`, `tiff_io`, `registration`, the workbench's `params`,
# `labels`, `executor` ... Each unit is a CMake target of its own that names
# the units it depends on, and each has a test executable that links the unit
# and its dependencies and nothing else:
#
#     sirius_lib_unit(buffer
#         HEADERS  sirius/buffer.hpp
#         SOURCES  buffer.cpp
#         DEPENDS  device checked_math errors)
#
#     sirius_unit_test(buffer SOURCES test_buffer.cpp UNITS lib/buffer)
#
#     cmake --build <build> --target test_buffer && ctest --test-dir <build> -R ...
#
# So a unit builds and is tested without the rest of the tree existing, the
# dependencies of every unit are written down where the build enforces them (a
# symbol from a unit that was not declared does not link), and the graph can
# only be what these files say it is. tools/check_units.py holds the #include
# lines to the same graph.
#
# What consumers see does not change. A unit with sources is an OBJECT library;
# `sirius` and `sirius_app_core` are still one static archive each, made of
# their units' objects, so the installed package, the bindings, the application
# and the benchmarks link exactly what they linked before.
#
# Naming: the unit `buffer` of group `lib` is the target sirius_lib_buffer
# (alias sirius::lib::buffer); in DEPENDS / UNITS it is `buffer` from inside
# its own group and `lib/buffer` from anywhere else.

include_guard(GLOBAL)

# The target of a unit named in DEPENDS / UNITS, seen from `group`.
function(_sirius_unit_target out group spec)
    if(spec MATCHES "^([^/]+)/(.+)$")
        set(${out} "sirius_${CMAKE_MATCH_1}_${CMAKE_MATCH_2}" PARENT_SCOPE)
    else()
        set(${out} "sirius_${group}_${spec}" PARENT_SCOPE)
    endif()
endfunction()

# sirius_add_unit(<group> <name>
#     [HEADERS <files>...]        headers of the unit, relative to the group's header root (for IDEs and the docs)
#     [SOURCES <files>...]        sources, relative to the calling CMakeLists; none = a header-only unit
#     [DEPENDS <units>...]        units whose headers this one includes or whose symbols it links
#     [LINK <targets>...]         external libraries the sources use (PRIVATE: not part of the unit's interface)
#     [PUBLIC_LINK <targets>...]  external libraries the unit's *headers* include (Eigen)
#     [DEFINES <defs>...]         PRIVATE compile definitions
#     [INCLUDE_PUBLIC <dirs>...]  [INCLUDE_PRIVATE <dirs>...])
function(sirius_add_unit group name)
    cmake_parse_arguments(PARSE_ARGV 2 U "" ""
        "HEADERS;SOURCES;DEPENDS;LINK;PUBLIC_LINK;DEFINES;INCLUDE_PUBLIC;INCLUDE_PRIVATE")
    if(U_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "sirius_add_unit(${group} ${name}): unknown arguments ${U_UNPARSED_ARGUMENTS}")
    endif()
    set(tgt sirius_${group}_${name})

    if(U_SOURCES)
        add_library(${tgt} OBJECT ${U_SOURCES} ${U_HEADERS})
        set(iface PUBLIC)
        set_target_properties(${tgt} PROPERTIES SIRIUS_UNIT_HAS_OBJECTS TRUE)
        if(SIRIUS_ENABLE_PYTHON_BINDINGS)
            set_target_properties(${tgt} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        endif()
        target_include_directories(${tgt} PRIVATE ${U_INCLUDE_PRIVATE})
        target_compile_definitions(${tgt} PRIVATE ${U_DEFINES})
        target_link_libraries(${tgt} PRIVATE ${U_LINK})
        if(SIRIUS_ENABLE_WARNINGS)
            target_set_warnings(${tgt})
        endif()
        if(SIRIUS_ENABLE_SANITIZERS)
            target_enable_sanitizers(${tgt})
        endif()
    else()
        add_library(${tgt} INTERFACE ${U_HEADERS})
        set(iface INTERFACE)
        set_target_properties(${tgt} PROPERTIES SIRIUS_UNIT_HAS_OBJECTS FALSE)
        if(U_LINK OR U_DEFINES OR U_INCLUDE_PRIVATE)
            message(FATAL_ERROR "sirius_add_unit(${group} ${name}): LINK / DEFINES / INCLUDE_PRIVATE need SOURCES")
        endif()
    endif()
    add_library(sirius::${group}::${name} ALIAS ${tgt})

    target_include_directories(${tgt} ${iface} ${U_INCLUDE_PUBLIC})
    # Every translation unit that instantiates Eigen has to agree on the
    # instruction set (it decides EIGEN_MAX_ALIGN_BYTES), so the flags are part
    # of every unit's interface, as they are of the archives'.
    target_link_libraries(${tgt} ${iface} sirius_simd ${U_PUBLIC_LINK})

    set(deps "")
    foreach(spec IN LISTS U_DEPENDS)
        _sirius_unit_target(dep ${group} ${spec})
        if(NOT TARGET ${dep})
            message(FATAL_ERROR "sirius_add_unit(${group} ${name}): unit '${spec}' (${dep}) is not defined yet; "
                                "units are declared in dependency order, which is also what keeps the graph acyclic")
        endif()
        list(APPEND deps ${dep})
    endforeach()
    # Usage requirements (include directories, definitions, the libraries a
    # dependency's headers need) come through the link interface; object
    # files do not, so whoever links units asks for their closure
    # (sirius_unit_closure).
    target_link_libraries(${tgt} ${iface} ${deps})
    set_target_properties(${tgt} PROPERTIES SIRIUS_UNIT_DEPENDS "${deps}" SIRIUS_UNIT_NAME "${group}/${name}")
    set_property(GLOBAL APPEND PROPERTY SIRIUS_UNITS_${group} ${tgt})
endfunction()

# The given unit targets and everything they depend on, dependencies last.
function(sirius_unit_closure out)
    set(todo ${ARGN})
    set(seen "")
    while(todo)
        list(POP_FRONT todo u)
        if(u IN_LIST seen)
            continue()
        endif()
        if(NOT TARGET ${u})
            message(FATAL_ERROR "sirius_unit_closure: ${u} is not a unit target")
        endif()
        list(APPEND seen ${u})
        get_target_property(deps ${u} SIRIUS_UNIT_DEPENDS)
        if(deps)
            list(APPEND todo ${deps})
        endif()
    endwhile()
    set(${out} ${seen} PARENT_SCOPE)
endfunction()

# The object files of every unit of `group` into the archive `tgt`.
function(sirius_archive_units tgt group)
    get_property(units GLOBAL PROPERTY SIRIUS_UNITS_${group})
    foreach(u IN LISTS units)
        get_target_property(has ${u} SIRIUS_UNIT_HAS_OBJECTS)
        if(has)
            target_sources(${tgt} PRIVATE $<TARGET_OBJECTS:${u}>)
        endif()
    endforeach()
endfunction()

# sirius_unit_test(<name>
#     SOURCES <files>...     Catch2 test sources
#     UNITS <units>...       `lib/buffer`, `app/labels` ...: the units under test; their closure is linked
#     [LINK <targets>...]    other libraries the test itself uses (TIFF::TIFF for a test that reads tags)
#     [INTERNAL]             the test includes a library-internal header (src/): gets that include
#                            directory and OpenMP, so the pragmas it compiles are the shipped ones
#     [LABEL <label>])       CTest label of its cases (`ctest -L`); one word -- a list would not survive
#                            the command line Catch's discovery passes the properties through
#
# Makes the executable test_<name>, registered with CTest under the same
# "sirius::<test case>" names as ever. The sources are compiled once, as the
# object library test_<name>_objects, which the all-in-one `sirius_tests`
# binary links as well.
function(sirius_unit_test name)
    cmake_parse_arguments(PARSE_ARGV 1 T "INTERNAL" "LABEL" "SOURCES;UNITS;LINK")
    if(T_UNPARSED_ARGUMENTS OR NOT T_SOURCES OR NOT T_UNITS)
        message(FATAL_ERROR "sirius_unit_test(${name}): needs SOURCES and UNITS (got '${T_UNPARSED_ARGUMENTS}')")
    endif()
    set(requested "")
    foreach(spec IN LISTS T_UNITS)
        _sirius_unit_target(u "" ${spec})
        list(APPEND requested ${u})
    endforeach()
    sirius_unit_closure(units ${requested})

    add_library(test_${name}_objects OBJECT ${T_SOURCES})
    target_link_libraries(test_${name}_objects PUBLIC ${units} Catch2::Catch2 ${T_LINK})
    target_include_directories(test_${name}_objects PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(test_${name}_objects PRIVATE SIRIUS_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data")
    if(T_INTERNAL)
        target_include_directories(test_${name}_objects PRIVATE ${PROJECT_SOURCE_DIR}/src)
        target_link_libraries(test_${name}_objects PUBLIC OpenMP::OpenMP_CXX)
    endif()
    if(SIRIUS_ENABLE_WARNINGS)
        target_set_warnings(test_${name}_objects)
    endif()
    if(SIRIUS_ENABLE_SANITIZERS)
        target_enable_sanitizers(test_${name}_objects)
    endif()
    set_property(GLOBAL APPEND PROPERTY SIRIUS_TEST_OBJECTS test_${name}_objects)

    add_executable(test_${name})
    # The units are named again here: an OBJECT library's object files reach
    # only what links it directly.
    target_link_libraries(test_${name} PRIVATE test_${name}_objects ${units} Catch2::Catch2WithMain)
    sirius_finish_test_executable(test_${name})
    if(T_LABEL)
        catch_discover_tests(test_${name} TEST_PREFIX "sirius::" PROPERTIES LABELS ${T_LABEL})
    else()
        catch_discover_tests(test_${name} TEST_PREFIX "sirius::")
    endif()
endfunction()

# What every test executable needs, whatever it is made of.
function(sirius_finish_test_executable tgt)
    # The same UTF-8 code page the application gets. Without it a test binary
    # dies before main, in a modal Visual C++ runtime dialog with nothing on
    # stdout, whenever TEMP holds a character the ANSI code page cannot
    # represent -- which is every developer whose Windows account name has an
    # accent in it.
    if(MSVC)
        target_sources(${tgt} PRIVATE ${PROJECT_SOURCE_DIR}/cmake/utf8-codepage.manifest)
    endif()
    # The runtime DLLs have to be beside the binary before Catch's test
    # discovery runs it. The test executables share a directory, so they are
    # copied once, by one target all of them depend on, rather than by every
    # executable's POST_BUILD step racing the others for the same files.
    if(WIN32 AND SIRIUS_NVTIFF_RUNTIME_LIBS)
        if(NOT TARGET sirius_test_runtime_dlls)
            set(copies "")
            foreach(dll IN LISTS SIRIUS_NVTIFF_RUNTIME_LIBS SIRIUS_NVCOMP_RUNTIME_LIBS)
                list(APPEND copies COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:${tgt}>")
            endforeach()
            add_custom_target(sirius_test_runtime_dlls ${copies} VERBATIM)
        endif()
        add_dependencies(${tgt} sirius_test_runtime_dlls)
    endif()
    if(SIRIUS_ENABLE_SANITIZERS)
        target_enable_sanitizers(${tgt})
    endif()
endfunction()
