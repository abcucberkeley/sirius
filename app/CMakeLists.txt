# The SIRIUS workbench (docs/design/README.md). Gated by SIRIUS_ENABLE_APP or
# SIRIUS_ENABLE_CLI at the top level; Dear ImGui, GLFW, ImPlot and friends are
# fetched in cmake/Dependencies.cmake as sirius::imgui, for the GUI only.
#
#   sirius_app_core  static, GUI-free: array model, operations, pipeline,
#                    executor, workbench, tool API, worker protocol, export,
#                    and what sirius-cli drives (rendering, statistics, the
#                    session and MCP protocols). Built as a graph of units
#                    (cmake/Units.cmake), each with a test executable of its
#                    own in tests/app_tests.cmake.
#   sirius-app       the GUI (app/imgui/**), globbed (CONFIGURE_DEPENDS) so
#                    adding a panel is adding a file; SIRIUS_ENABLE_APP only.
#   sirius-cli       the core without a window, for scripts and agents
#                    (app/cli); built whenever this directory is.

include(Units)

# --- core -------------------------------------------------------------------
#
# One unit per header: its sources, and the units it is allowed to use. They
# are declared in dependency order, which is what keeps the graph acyclic, and
# `sirius_app_core` at the bottom is still the single archive the application,
# the tests and the install link.

# Where an installed tree keeps help/, python/ and plugins/, relative to the
# executable's directory (core/app_paths.cpp): computed from the full install
# directories so a --prefix given at install time moves both together.
include(GNUInstallDirs)
set(SIRIUS_APP_INSTALL_DATADIR "${CMAKE_INSTALL_DATADIR}/sirius")
file(RELATIVE_PATH _sirius_app_datadir_from_bindir
    "${CMAKE_INSTALL_FULL_BINDIR}" "${CMAKE_INSTALL_FULL_DATADIR}/sirius")

# sirius_core_unit(<name> [HEADERS ...] [SOURCES ...] [DEPENDS ...] [LINK ...] [PUBLIC_LINK ...] [DEFINES ...])
# HEADERS and SOURCES are relative to app/core; DEPENDS names core units, or
# library ones as lib/<unit>. PUBLIC_LINK is for a library the unit's *headers*
# include -- nlohmann/json, whose json_fwd.hpp several of them declare
# parameters with -- and LINK for one only its sources use.
function(sirius_core_unit name)
    cmake_parse_arguments(PARSE_ARGV 1 C "" "" "HEADERS;SOURCES;DEPENDS;LINK;PUBLIC_LINK;DEFINES")
    list(TRANSFORM C_HEADERS PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/core/)
    list(TRANSFORM C_SOURCES PREPEND core/)
    sirius_add_unit(core ${name}
        HEADERS ${C_HEADERS}
        SOURCES ${C_SOURCES}
        DEPENDS ${C_DEPENDS}
        LINK ${C_LINK}
        PUBLIC_LINK ${C_PUBLIC_LINK}
        DEFINES ${C_DEFINES}
        INCLUDE_PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    if(C_SOURCES)
        # Library-wide, as when this was one target: OpenMP (Eigen reads
        # _OPENMP, so a unit without a pragma still has to agree).
        target_link_libraries(sirius_core_${name} PRIVATE OpenMP::OpenMP_CXX)
        # getenv and strerror are the portable spellings the core uses on
        # purpose; MSVC's _s replacements exist only on Windows.
        if(MSVC)
            target_compile_definitions(sirius_core_${name} PRIVATE _CRT_SECURE_NO_WARNINGS)
        endif()
    endif()
endfunction()

# op(<name> <source> [DEPENDS ...]): one built-in operation, which always has
# the registry's declaration of its factory and the shared helpers.
function(op name source)
    cmake_parse_arguments(PARSE_ARGV 2 O "" "" "DEPENDS")
    sirius_core_unit(ops_${name} SOURCES ${source} DEPENDS ops_factories ops_common ${O_DEPENDS})
    set(SIRIUS_APP_OPS ${SIRIUS_APP_OPS} ops_${name} PARENT_SCOPE)
endfunction()

# --- vocabulary --------------------------------------------------------------

sirius_core_unit(cancel            HEADERS cancel.hpp)
# Wiping secrets from memory, and the SHA-256 / HMAC of the worker handshake.
sirius_core_unit(secure_wipe       HEADERS secure_wipe.hpp   SOURCES secure_wipe.cpp)
sirius_core_unit(sha256            HEADERS sha256.hpp        SOURCES sha256.cpp        DEPENDS secure_wipe)
sirius_core_unit(errors            HEADERS errors.hpp            DEPENDS lib/errors)
sirius_core_unit(byte_budget_lru   HEADERS byte_budget_lru.hpp)
sirius_core_unit(label_frames      HEADERS label_frames.hpp      DEPENDS lib/index)

sirius_core_unit(app_paths
    HEADERS app_paths.hpp
    SOURCES app_paths.cpp
    DEFINES SIRIUS_APP_DATADIR_FROM_BINDIR="${_sirius_app_datadir_from_bindir}")

# What the application asks of the operating system without a window (the
# user's directories, the environment, PATH, whole files), and child processes
# with their standard streams on pipes.
sirius_core_unit(host              HEADERS host.hpp          SOURCES host.cpp)
sirius_core_unit(process           HEADERS process.hpp       SOURCES process.cpp       LINK Threads::Threads)

sirius_core_unit(history      HEADERS history.hpp      SOURCES history.cpp)
sirius_core_unit(session_log  HEADERS session_log.hpp  SOURCES session_log.cpp  PUBLIC_LINK nlohmann_json::nlohmann_json)
sirius_core_unit(params       HEADERS params.hpp       SOURCES params.cpp       PUBLIC_LINK nlohmann_json::nlohmann_json)
# The checkout's help/ and python/ are where a binary built from this tree
# looks first (app_paths.hpp), so the two units that own those directories are
# told where the tree is, as app_paths is told where an install puts them.
sirius_core_unit(help_pages
    HEADERS help_pages.hpp
    SOURCES help_pages.cpp
    DEPENDS app_paths
    DEFINES SIRIUS_APP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# --- data model ----------------------------------------------------------------

sirius_core_unit(array            HEADERS array.hpp            SOURCES array.cpp            DEPENDS lib/buffer lib/checked_math)
sirius_core_unit(dataset          HEADERS dataset.hpp          SOURCES dataset.cpp          DEPENDS array lib/pixel_type)
sirius_core_unit(display_mapping  HEADERS display_mapping.hpp  SOURCES display_mapping.cpp  DEPENDS lib/buffer)
sirius_core_unit(diagnostics      HEADERS diagnostics.hpp      SOURCES diagnostics.cpp      DEPENDS lib/buffer lib/fft)

# The index is below the volume that owns one: it reads frames and diffs
# (label_frames.hpp), not LabelVolume, so the two are not a cycle.
sirius_core_unit(tracks    HEADERS tracks.hpp    SOURCES tracks.cpp    DEPENDS label_frames PUBLIC_LINK nlohmann_json::nlohmann_json)
sirius_core_unit(labels    HEADERS labels.hpp    SOURCES labels.cpp    DEPENDS array label_frames tracks lib/buffer)
sirius_core_unit(tracking  HEADERS tracking.hpp  SOURCES tracking.cpp  DEPENDS labels)

sirius_core_unit(manifest
    HEADERS manifest.hpp
    SOURCES manifest.cpp
    DEPENDS dataset lib/tiff_io
    PUBLIC_LINK nlohmann_json::nlohmann_json   # manifest.hpp declares with json_fwd
    LINK tomlplusplus::tomlplusplus)

sirius_core_unit(array_source
    HEADERS array_source.hpp
    SOURCES array_source.cpp
    DEPENDS array dataset manifest lib/device lib/tiff_io lib/zarr_io)

sirius_core_unit(export
    HEADERS export.hpp
    SOURCES export.cpp
    DEPENDS array dataset labels cancel lib/tiff_io lib/image_ops lib/zarr_io
    LINK nlohmann_json::nlohmann_json)

sirius_core_unit(training_export
    HEADERS training_export.hpp
    SOURCES training_export.cpp
    DEPENDS array dataset export labels cancel lib/tiff_io
    PUBLIC_LINK nlohmann_json::nlohmann_json)

# --- the Python worker ------------------------------------------------------------

sirius_core_unit(rpc
    HEADERS rpc.hpp
    SOURCES rpc.cpp
    DEPENDS app_paths cancel errors secure_wipe sha256 lib/buffer lib/checked_math
    DEFINES SIRIUS_APP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    PUBLIC_LINK nlohmann_json::nlohmann_json)
if(WIN32)
    target_link_libraries(sirius_core_rpc PRIVATE ws2_32)
endif()

# Why the worker did not come up, SIRIUS's own Python environment for it, and
# the launcher that starts it as a child process.
sirius_core_unit(worker_error      HEADERS worker_error.hpp  SOURCES worker_error.cpp  PUBLIC_LINK nlohmann_json::nlohmann_json)
sirius_core_unit(python_env        HEADERS python_env.hpp    SOURCES python_env.cpp
                                   DEPENDS host process app_paths PUBLIC_LINK nlohmann_json::nlohmann_json)
sirius_core_unit(local_worker      HEADERS local_worker.hpp  SOURCES local_worker.cpp
                                   DEPENDS rpc process python_env worker_error host cancel app_paths LINK Threads::Threads)

# The cluster: one SSH session (its command channel, its SOCKS proxy and the
# askpass relay of its prompts), the Slurm job of the worker it submits and
# watches, and the datasets that stay on the cluster, read by that worker.
sirius_core_unit(remote_host       HEADERS remote_host.hpp   SOURCES remote_host.cpp
                                   DEPENDS process host secure_wipe LINK nlohmann_json::nlohmann_json Threads::Threads)

# --- the engine: SIRIUS's C++ side of the worker protocol (`sirius-cli serve`) -------------
#
# Which build this is (git commit, operation schema hash, engine API), written
# into a header by cmake/BuildInfo.cmake at every build: the engine reports it
# in its hello, and the application refuses an engine whose operations differ.
# SIRIUS_BUILD_COMMIT (+ SIRIUS_BUILD_DIRTY) name the commit of a tree without
# .git, such as a container build from an archive.
find_package(Git QUIET)
set(SIRIUS_BUILD_COMMIT "" CACHE STRING "The git commit this tree is, when it has no .git (container builds)")
set(SIRIUS_BUILD_DIRTY OFF CACHE BOOL "With SIRIUS_BUILD_COMMIT: the tree has changes over that commit")
set(_sirius_build_info_dir ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(_sirius_build_info_header ${_sirius_build_info_dir}/sirius_build_info_generated.hpp)
set(_sirius_build_info_command ${CMAKE_COMMAND}
    -DSOURCE_DIR=${PROJECT_SOURCE_DIR} -DOUTPUT=${_sirius_build_info_header} -DVERSION=${PROJECT_VERSION}
    -DSCHEMA_FILE=${PROJECT_SOURCE_DIR}/bindings/python/sirius/op_schema.json
    -DGIT_EXECUTABLE=${GIT_EXECUTABLE} -DCOMMIT=${SIRIUS_BUILD_COMMIT} -DDIRTY=${SIRIUS_BUILD_DIRTY}
    -P ${PROJECT_SOURCE_DIR}/cmake/BuildInfo.cmake)
file(MAKE_DIRECTORY ${_sirius_build_info_dir})
execute_process(COMMAND ${_sirius_build_info_command})   # so the header exists from the start
add_custom_target(sirius_build_info_header
    COMMAND ${_sirius_build_info_command}
    BYPRODUCTS ${_sirius_build_info_header}
    COMMENT "The build's identity (git commit, operation schema hash)"
    VERBATIM)
sirius_core_unit(build_info        HEADERS build_info.hpp    SOURCES build_info.cpp
                                   PUBLIC_LINK nlohmann_json::nlohmann_json)
target_include_directories(sirius_core_build_info PRIVATE ${_sirius_build_info_dir})
add_dependencies(sirius_core_build_info sirius_build_info_header)
# The application's settings file: TOML (sirius-app.toml) <-> the "group/name"
# keys the settings are held by in memory, and the store that reads, merges
# and writes it (sirius-app's Settings; sirius-cli does not use it).
sirius_core_unit(settings_toml     HEADERS settings_toml.hpp SOURCES settings_toml.cpp
                                   PUBLIC_LINK nlohmann_json::nlohmann_json LINK tomlplusplus::tomlplusplus)
sirius_core_unit(settings_store    HEADERS settings_store.hpp SOURCES settings_store.cpp
                                   DEPENDS host settings_toml PUBLIC_LINK nlohmann_json::nlohmann_json)
# The Slurm session (after build_info: an engine of other operations is refused
# at its hello), and the cluster profiles as the settings file keeps them.
sirius_core_unit(cluster           HEADERS cluster.hpp cluster_profiles.hpp cluster_wizard.hpp
                                   SOURCES cluster.cpp cluster_profiles.cpp cluster_wizard.cpp
                                   DEPENDS remote_host rpc cancel errors build_info settings_toml
                                   PUBLIC_LINK nlohmann_json::nlohmann_json LINK Threads::Threads)
# The worker's launch script (app/python/slurm/sirius_worker.sbatch) compiled
# into the session: it is sent to the cluster with every worker start, so the
# worker never runs whatever older copy the cluster's checkout holds.
set(_sirius_worker_script ${CMAKE_CURRENT_SOURCE_DIR}/python/slurm/sirius_worker.sbatch)
set(_sirius_worker_script_header ${_sirius_build_info_dir}/sirius_worker_script_generated.hpp)
set(_sirius_worker_script_command ${CMAKE_COMMAND} -DINPUT=${_sirius_worker_script} -DOUTPUT=${_sirius_worker_script_header}
    -DNAME=kSiriusWorkerScript -P ${PROJECT_SOURCE_DIR}/cmake/EmbedText.cmake)
execute_process(COMMAND ${_sirius_worker_script_command})   # so the header exists from the start
add_custom_command(OUTPUT ${_sirius_worker_script_header}
    COMMAND ${_sirius_worker_script_command}
    DEPENDS ${_sirius_worker_script} ${PROJECT_SOURCE_DIR}/cmake/EmbedText.cmake
    COMMENT "The worker's launch script, into the application"
    VERBATIM)
add_custom_target(sirius_worker_script_header DEPENDS ${_sirius_worker_script_header})
target_include_directories(sirius_core_cluster PRIVATE ${_sirius_build_info_dir})
add_dependencies(sirius_core_cluster sirius_worker_script_header)
# Arrays on the wire: zstd (the build's own, cmake/Dependencies.cmake) or zlib,
# over shuffled bytes, as the client accepts.
sirius_core_unit(array_codec       HEADERS array_codec.hpp   SOURCES array_codec.cpp
                                   DEPENDS rpc errors PUBLIC_LINK nlohmann_json::nlohmann_json LINK ZLIB::ZLIB)
if(TARGET ZSTD::ZSTD)
    target_link_libraries(sirius_core_array_codec PRIVATE ZSTD::ZSTD)
    target_compile_definitions(sirius_core_array_codec PRIVATE SIRIUS_APP_HAVE_ZSTD)
endif()
# The serving half of the protocol: handshake, slots, jobs, progress, cancel.
sirius_core_unit(rpc_server        HEADERS rpc_server.hpp    SOURCES rpc_server.cpp
                                   DEPENDS rpc cancel errors PUBLIC_LINK nlohmann_json::nlohmann_json LINK Threads::Threads)
# dataset_info / _read / _view / _stats in C++ (sirius_worker/datasets.py's replies).
sirius_core_unit(dataset_service   HEADERS dataset_service.hpp SOURCES dataset_service.cpp
                                   DEPENDS rpc_server array_codec array_source manifest build_info host lib/buffer lib/device lib/tiff_io
                                   PUBLIC_LINK nlohmann_json::nlohmann_json LINK Threads::Threads)
# The datasets that stay on the cluster, read by the worker or the engine there.
sirius_core_unit(remote_source     HEADERS remote_source.hpp SOURCES remote_source.cpp
                                   DEPENDS array_source array_codec dataset rpc errors lib/checked_math
                                   PUBLIC_LINK nlohmann_json::nlohmann_json LINK Threads::Threads)
# A folder of TIFF stacks on the cluster as one dataset: listed over the
# session, its manifest built as on this computer and kept in ~/.sirius/manifests there.
sirius_core_unit(cluster_folder    HEADERS cluster_folder.hpp SOURCES cluster_folder.cpp
                                   DEPENDS cluster remote_host remote_source array_source manifest
                                   LINK nlohmann_json::nlohmann_json)
if(WIN32)
    target_link_libraries(sirius_core_remote_host PRIVATE ws2_32)
endif()

# --- steps and the pipeline ---------------------------------------------------------

sirius_core_unit(operation
    HEADERS operation.hpp
    SOURCES operation.cpp
    DEPENDS array dataset diagnostics labels params array_source cancel lib/device
    PUBLIC_LINK nlohmann_json::nlohmann_json)

sirius_core_unit(pipeline
    HEADERS pipeline.hpp
    SOURCES pipeline.cpp
    DEPENDS operation params
    PUBLIC_LINK nlohmann_json::nlohmann_json
    LINK tomlplusplus::tomlplusplus)

sirius_core_unit(executor
    HEADERS executor.hpp
    SOURCES executor.cpp
    DEPENDS operation pipeline array_source cancel lib/device
    LINK nlohmann_json::nlohmann_json)

# A step's result besides its array, for the wire to the engine: meta, reports,
# lineage, diagnostics (JSON + tensors).
sirius_core_unit(serialize
    HEADERS serialize.hpp
    SOURCES serialize.cpp
    DEPENDS dataset diagnostics executor label_frames rpc errors
    PUBLIC_LINK nlohmann_json::nlohmann_json)

# --- SIM reconstruction, as the workbench drives it -------------------------------------

sirius_core_unit(session
    HEADERS session.hpp
    SOURCES session.cpp
    DEPENDS lib/sim_reconstruction lib/otf lib/otf_io lib/otf_ideal lib/legacy_config lib/tiff_io
            lib/sim_parameters lib/buffer lib/device lib/fft_common lib/constants)

sirius_core_unit(volume_ops
    HEADERS volume_ops.hpp
    SOURCES volume_ops.cpp
    DEPENDS lib/buffer lib/otf lib/sim_parameters lib/sim_reconstruction lib/fft lib/constants)

# --- the operations --------------------------------------------------------------------
# Every operation is a unit of its own, so what it needs is what it links: the
# stitch step has the mosaic code in it and the deskew step does not. They are
# leaves -- nothing includes an operation -- and ops/builtin.hpp, which only
# declares their factories, is the header-only unit they share.

sirius_core_unit(ops_factories  HEADERS ops/builtin.hpp  DEPENDS operation)

sirius_core_unit(ops_common
    HEADERS ops/common.hpp
    SOURCES ops/common.cpp
    DEPENDS operation array_source lib/image_ops)

sirius_core_unit(ops_contrast_api
    HEADERS ops/contrast.hpp
    DEPENDS operation)

sirius_core_unit(ops_load_api
    HEADERS ops/load.hpp
    DEPENDS array_source params)

sirius_core_unit(ops_sim_params
    HEADERS ops/sim_params.hpp
    DEPENDS dataset params lib/sim_parameters)

sirius_core_unit(ops_segment_common
    HEADERS ops/segment_common.hpp
    SOURCES ops/segment_common.cpp
    DEPENDS labels)

sirius_core_unit(ops_torch_model
    HEADERS ops/torch_model.hpp
    SOURCES ops/torch_model.cpp
    DEPENDS rpc ops_common
    PUBLIC_LINK nlohmann_json::nlohmann_json)

# A trained model folder (model.json, latents-model/1): what the Foundation
# step and the parameter panel read of one, without the worker.
sirius_core_unit(model_folder
    HEADERS model_folder.hpp
    SOURCES model_folder.cpp
    DEPENDS host params
    PUBLIC_LINK nlohmann_json::nlohmann_json)

# op(<name> <source> [DEPENDS ...]): one built-in operation.
op(load        ops/load.cpp        DEPENDS ops_load_api array_source manifest)
op(sim         ops/sim.cpp         DEPENDS ops_sim_params session volume_ops lib/legacy_config lib/device lib/constants)
op(decon       ops/decon.cpp       DEPENDS lib/deconvolution lib/tiff_io)
op(volrec      ops/volrec.cpp      DEPENDS lib/image_ops)
op(einsum      ops/einsum.cpp      DEPENDS lib/image_ops)
op(contrast    ops/contrast.cpp    DEPENDS ops_contrast_api array_source lib/image_ops)
op(flatfield   ops/flatfield.cpp   DEPENDS lib/image_ops lib/tiff_io)
op(bleach      ops/bleach.cpp      DEPENDS lib/image_ops)
op(deskew      ops/deskew.cpp      DEPENDS lib/image_ops)
op(croppad     ops/croppad.cpp     DEPENDS lib/image_ops)
op(resample    ops/resample.cpp    DEPENDS lib/image_ops)
op(merge       ops/merge.cpp       DEPENDS lib/image_ops)
op(stitch      ops/stitch.cpp      DEPENDS array_source lib/stitching lib/stitching_tiff)
op(register    ops/register.cpp    DEPENDS lib/image_ops lib/registration)
op(seg         ops/seg.cpp         DEPENDS ops_segment_common ops_torch_model rpc remote_source cancel errors)
op(foundation  ops/foundation.cpp  DEPENDS tracks rpc model_folder)
op(classic     ops/classic.cpp     DEPENDS ops_segment_common lib/image_ops lib/constants)
op(cleanup     ops/cleanup.cpp     DEPENDS labels)
op(import_labels ops/import_labels.cpp DEPENDS labels lib/tiff_io)
op(track       ops/track.cpp       DEPENDS tracking tracks rpc)

# The list of them, and registerBuiltinOperations() that walks it: the one
# place that depends on every operation.
sirius_core_unit(ops_registry
    SOURCES ops/builtin.cpp ops/builtin_list.cpp
    DEPENDS ops_factories ${SIRIUS_APP_OPS})

# The parameter specs of every built-in as JSON: above the registry, since
# naming every one of them is what it is for.
sirius_core_unit(ops_schema
    HEADERS ops/schema.hpp
    SOURCES ops/schema.cpp
    DEPENDS ops_registry ops_factories
    PUBLIC_LINK nlohmann_json::nlohmann_json)

# A user operation the Python worker serves, and the plugin files behind it.
sirius_core_unit(plugin
    HEADERS ops/plugin.hpp
    SOURCES ops/plugin.cpp
    DEPENDS operation rpc executor help_pages array_source ops_common
    PUBLIC_LINK nlohmann_json::nlohmann_json)

# --- the workbench -----------------------------------------------------------------

sirius_core_unit(workbench
    HEADERS workbench.hpp
    SOURCES workbench.cpp
    DEPENDS array_source dataset executor history labels operation pipeline rpc session_log tracks
            cancel ops_registry ops_factories ops_load_api ops_contrast_api plugin worker_error remote_source serialize build_info sha256
    PUBLIC_LINK nlohmann_json::nlohmann_json)

sirius_core_unit(tool_api
    HEADERS tool_api.hpp
    SOURCES tool_api.cpp
    DEPENDS workbench training_export worker_error cancel lib/tiff_io
    PUBLIC_LINK nlohmann_json::nlohmann_json)

# --- headless: what sirius-cli drives --------------------------------------------
# The viewer's display model (so a render is what the viewer draws), image
# encoding, statistics, the JSON-lines session and MCP protocols, and the
# window-less workbench that serves one tool table to all of them.

sirius_core_unit(image_encode      HEADERS image_encode.hpp  SOURCES image_encode.cpp  LINK sirius_stb)
sirius_core_unit(display_model     HEADERS display_model.hpp SOURCES display_model.cpp
                                   DEPENDS byte_budget_lru operation workbench array_source cancel)
sirius_core_unit(statistics        HEADERS statistics.hpp    SOURCES statistics.cpp
                                   DEPENDS operation array_source labels tracks cancel dataset remote_source lib/pixel_type
                                   PUBLIC_LINK nlohmann_json::nlohmann_json)
sirius_core_unit(agent_protocol    HEADERS agent_protocol.hpp agent_server.hpp
                                   SOURCES agent_protocol.cpp agent_session.cpp agent_mcp.cpp
                                   PUBLIC_LINK nlohmann_json::nlohmann_json LINK Threads::Threads)
sirius_core_unit(headless          HEADERS headless.hpp
                                   SOURCES headless.cpp headless_tools.cpp headless_render.cpp headless_export.cpp
                                   DEPENDS agent_protocol workbench tool_api display_model statistics image_encode export
                                           training_export help_pages ops_schema ops_registry ops_factories plugin
                                           local_worker python_env worker_error host array_source pipeline dataset params
                                           labels tracks diagnostics display_mapping cancel app_paths rpc executor
                                           lib/tiff_io lib/device
                                   PUBLIC_LINK nlohmann_json::nlohmann_json LINK Threads::Threads)

# --- the engine's node side: the pipelines it runs and the outputs it keeps ---------------
# (core/engine_node.hpp; above the operations and the executor, below engine_server)
sirius_core_unit(engine_node
    HEADERS engine_node.hpp
    SOURCES engine_node.cpp
    DEPENDS dataset_service rpc_server rpc executor pipeline operation ops_registry ops_factories serialize statistics
            remote_source sha256 host cancel array_source lib/device
    PUBLIC_LINK nlohmann_json::nlohmann_json)
# `sirius-cli serve`: the datasets in C++, the pipelines run on the node with
# their outputs kept there (engine_node), the rest relayed to a Python worker child.
sirius_core_unit(engine_server     HEADERS engine_server.hpp SOURCES engine_server.cpp
                                   DEPENDS dataset_service rpc_server rpc array_codec build_info local_worker host cancel errors
                                           engine_node lib/device
                                   PUBLIC_LINK nlohmann_json::nlohmann_json LINK Threads::Threads)

# --- the archive -------------------------------------------------------------

add_library(sirius_app_core STATIC)
add_library(sirius::app_core ALIAS sirius_app_core)
sirius_archive_units(sirius_app_core core)
set_target_properties(sirius_app_core PROPERTIES LINKER_LANGUAGE CXX)   # objects, no sources to tell by
target_include_directories(sirius_app_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# The units compile; the archive is what consumers link, so the libraries the
# objects need are named once more here, for them.
target_link_libraries(sirius_app_core
    PUBLIC sirius nlohmann_json::nlohmann_json Threads::Threads
    PRIVATE OpenMP::OpenMP_CXX TIFF::TIFF tomlplusplus::tomlplusplus ZLIB::ZLIB)
if(TARGET ZSTD::ZSTD)
    target_link_libraries(sirius_app_core PRIVATE ZSTD::ZSTD)   # the arrays on the wire (array_codec)
endif()
if(WIN32)
    target_link_libraries(sirius_app_core PRIVATE ws2_32)
endif()
if(SIRIUS_ENABLE_SANITIZERS)
    target_enable_sanitizers(sirius_app_core)
endif()

# --- what both executables find beside them ----------------------------------------
# Help pages, the Python worker and the example plugins sit next to sirius-app
# and sirius-cli, so users can edit the pages and start the worker from the
# build tree. One target copies them for both executables: two POST_BUILD steps
# of a parallel build would copy the same files at the same time. On Windows it
# also brings the NVIDIA runtime DLLs (as for the tests), so either executable
# starts from the build tree without PATH changes; everything else is linked
# statically.
get_property(_sirius_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_sirius_multi_config)
    set(SIRIUS_APP_RUNTIME_DIR "${PROJECT_BINARY_DIR}/app/$<CONFIG>")
else()
    set(SIRIUS_APP_RUNTIME_DIR "${PROJECT_BINARY_DIR}/app")
endif()
set(_runtime_copies
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/help    ${SIRIUS_APP_RUNTIME_DIR}/help
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python  ${SIRIUS_APP_RUNTIME_DIR}/python
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/plugins ${SIRIUS_APP_RUNTIME_DIR}/plugins)
if(WIN32)
    foreach(_dll IN LISTS SIRIUS_NVTIFF_RUNTIME_LIBS SIRIUS_NVCOMP_RUNTIME_LIBS)
        list(APPEND _runtime_copies COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" ${SIRIUS_APP_RUNTIME_DIR})
    endforeach()
endif()
add_custom_target(sirius_app_runtime_files ${_runtime_copies}
    COMMENT "Copying help pages, the Python worker, the example plugins (and nv DLLs) beside the executables"
    VERBATIM)

# --- install (Linux): what both executables share -----------------------------------
#
#   cmake --install <build> --component app --prefix <prefix>
#
#   share/sirius/{help,python,plugins}     what the build copies beside the executables
#   share/sirius/python/workbench.py       the step library (and its op_schema.json) the
#                                          worker loads when no sirius package is importable
#   lib/sirius/                            nvTIFF / nvCOMP of a CUDA build, found via $ORIGIN
#
# The executables' own rules (bin/, and the GUI's fonts, icons and desktop
# file) are in imgui/ and cli/. tests/CMakeLists.txt installs this into the
# build tree and runs the result (app/linux/check_install.cmake). Windows runs
# from the build tree; a macOS bundle would need a layout of its own.
if(UNIX AND NOT APPLE)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/help ${CMAKE_CURRENT_SOURCE_DIR}/python
                      ${CMAKE_CURRENT_SOURCE_DIR}/plugins
        DESTINATION ${SIRIUS_APP_INSTALL_DATADIR} COMPONENT app
        PATTERN __pycache__ EXCLUDE
        PATTERN "python/tests" EXCLUDE)
    install(FILES
        ${PROJECT_SOURCE_DIR}/bindings/python/sirius/workbench.py
        ${PROJECT_SOURCE_DIR}/bindings/python/sirius/op_schema.json
        DESTINATION ${SIRIUS_APP_INSTALL_DATADIR}/python COMPONENT app)
    # The redistributables ship "libnvtiff.so.0 -> libnvtiff.so.0.8.0.82"
    # symlinks; install the real file under the soname the executables need
    # (as bindings/CMakeLists.txt does for the wheel). A CLI-only CUDA build
    # needs them as much as the GUI does.
    foreach(_lib IN LISTS SIRIUS_NVTIFF_RUNTIME_LIBS SIRIUS_NVCOMP_RUNTIME_LIBS)
        get_filename_component(_soname "${_lib}" NAME)
        file(REAL_PATH "${_lib}" _real)
        install(FILES "${_real}" DESTINATION ${CMAKE_INSTALL_LIBDIR}/sirius COMPONENT app RENAME "${_soname}")
    endforeach()
endif()

# --- the executables ------------------------------------------------------------------
# The GUI: Dear ImGui, GLFW, OpenGL 3.3 and ImPlot over the core
# (app/imgui/README.md); its build and install rules are in
# app/imgui/CMakeLists.txt. sirius-cli (app/cli) needs none of them.
if(SIRIUS_ENABLE_APP)
    add_subdirectory(imgui)
endif()
add_subdirectory(cli)
