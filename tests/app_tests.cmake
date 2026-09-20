# Tests of the workbench core (app/core), one executable per unit like the
# library's (tests/CMakeLists.txt). Included when SIRIUS_ENABLE_APP is on.
#
# app_test(<name> <sources>... UNITS <units>...) is sirius_unit_test with the
# core/ group spelled out; the units named are the ones the test drives, and
# their closure is what the binary links.
function(app_test name)
    cmake_parse_arguments(PARSE_ARGV 1 T "" "" "UNITS;LINK")
    list(TRANSFORM T_UNITS PREPEND core/)
    sirius_unit_test(app_${name} SOURCES ${T_UNPARSED_ARGUMENTS} UNITS ${T_UNITS} LINK ${T_LINK}
                     LABEL app.${name})
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
