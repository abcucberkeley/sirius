# Tests of the workbench core (app/core), one executable per unit like the
# library's (tests/CMakeLists.txt). Included when SIRIUS_ENABLE_APP or
# SIRIUS_ENABLE_CLI is on.
#
# app_test(<name> <sources>... UNITS <units>...) is sirius_unit_test with the
# core/ group spelled out; the units named are the ones the test drives, and
# their closure is what the binary links.
function(app_test name)
    cmake_parse_arguments(PARSE_ARGV 1 T "" "" "UNITS;LINK")
    list(TRANSFORM T_UNITS PREPEND core/)
    sirius_unit_test(app_${name} SOURCES ${T_UNPARSED_ARGUMENTS} UNITS ${T_UNITS} LINK ${T_LINK}
                     LABEL app.${name})
    # std::getenv, as the core units use it (app/CMakeLists.txt).
    if(MSVC)
        target_compile_definitions(test_app_${name}_objects PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
endfunction()

app_test(lru       test_app_lru.cpp       UNITS byte_budget_lru)
app_test(help      test_app_help.cpp      UNITS help_pages app_paths)
app_test(display   test_app_display.cpp   UNITS display_mapping array)
app_test(labels    test_app_labels.cpp    UNITS labels ops_common ops_segment_common)
app_test(tracking  test_app_tracking.cpp  UNITS tracking labels)
app_test(training  test_app_training.cpp  UNITS training_export LINK TIFF::TIFF)
app_test(io        test_app_io.cpp        UNITS array_source export labels dataset)
app_test(manifest  test_app_manifest.cpp  UNITS manifest array_source executor pipeline ops_registry)
app_test(session   test_app_session.cpp   UNITS session volume_ops)
app_test(rpc       test_app_rpc.cpp       UNITS rpc app_paths cancel errors ops_registry plugin workbench
                                                help_pages array_source pipeline)
app_test(schema    test_app_schema.cpp    UNITS ops_schema)
app_test(parity    test_app_parity.cpp    UNITS operation ops_registry)
# The workbench and everything under it: the pipeline, the executor, the
# history, the tool API and the plugin loader.
app_test(pipeline  test_app_pipeline.cpp  UNITS workbench tool_api plugin history manifest ops_schema)
app_test(tracks    test_app_tracks.cpp    UNITS tracks labels workbench tool_api array_source ops_registry)
# One binary for every built-in: the operations are leaves -- nothing includes
# one -- so what a per-operation binary would prove, the unit targets already
# do, and several cases here are regressions that cross two steps.
app_test(ops       test_app_ops.cpp       UNITS ops_registry ops_common ops_contrast_api ops_torch_model
                                                executor pipeline rpc array_source
                   LINK TIFF::TIFF)

# --- what sirius-cli stands on -------------------------------------------------
# The operating system without a window, child processes, SIRIUS's own Python
# environment and the worker launcher; the tool table's busy gate; rendering,
# statistics, the session and MCP protocols, and the window-less workbench.
app_test(host           test_app_host.cpp           UNITS host)
app_test(process        test_app_process.cpp        UNITS process host)
app_test(python_env     test_app_python_env.cpp     UNITS python_env host process)
app_test(local_worker   test_app_local_worker.cpp   UNITS local_worker worker_error python_env rpc)
app_test(tool_gate      test_app_tool_gate.cpp      UNITS tool_api workbench ops_registry)
app_test(render         test_app_render.cpp         UNITS display_model image_encode workbench ops_registry array_source)
app_test(statistics     test_app_statistics.cpp     UNITS statistics array_source labels)
app_test(agent_protocol test_app_agent_protocol.cpp UNITS agent_protocol)
app_test(headless       test_app_headless.cpp       UNITS headless ops_registry LINK TIFF::TIFF)
# What the tests start: the child helper (tests/tools/sirius_test_child.cpp),
# the worker in the checkout and the examples. The objects are shared with
# sirius_tests, which depends on the helper too (tests/CMakeLists.txt).
foreach(_t process python_env local_worker)
    target_compile_definitions(test_app_${_t}_objects PRIVATE SIRIUS_TEST_CHILD="$<TARGET_FILE:sirius_test_child>")
    add_dependencies(test_app_${_t} sirius_test_child)
endforeach()
target_compile_definitions(test_app_python_env_objects   PRIVATE SIRIUS_TEST_WORKER_DIR="${PROJECT_SOURCE_DIR}/app/python")
target_compile_definitions(test_app_local_worker_objects PRIVATE SIRIUS_TEST_WORKER_DIR="${PROJECT_SOURCE_DIR}/app/python"
                                                                 SIRIUS_TEST_FAKE_WORKER_DIR="${PROJECT_SOURCE_DIR}/tests/data/fake_worker_missing")
target_compile_definitions(test_app_headless_objects     PRIVATE SIRIUS_TEST_EXAMPLES_DIR="${PROJECT_SOURCE_DIR}/examples"
                                                                 SIRIUS_TEST_WORKER_DIR="${PROJECT_SOURCE_DIR}/app/python")
