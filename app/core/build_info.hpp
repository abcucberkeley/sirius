#ifndef SIRIUS_APP_BUILD_INFO_HPP
#define SIRIUS_APP_BUILD_INFO_HPP

// Which SIRIUS this is: the version, the git commit the build was made from
// (and whether the tree had uncommitted changes), the hash of the operation
// schema (every operation's parameters) and the engine's method-set version.
// Generated at build time (cmake/BuildInfo.cmake) and compiled into the core,
// so sirius-app, sirius-cli and the engine on a cluster node (`sirius-cli
// serve`, core/engine_server.hpp) all carry it; the engine reports it in its
// hello ("engine") and the container image as /opt/sirius/BUILD.json.
//
// Whether an engine may serve this application is engineMismatch(): the
// operation set and the engine API must be the same; the commit may differ
// (the user's choice: a dev build talks to an image built a commit earlier as
// long as no operation changed).
//
// WHAT NAMES THE OPERATION SET, and why it is not the schema hash alone.
// `ops_schema` is SHA-256 of the whole committed schema snapshot, labels,
// help strings, units and ranges included, so it moves when nothing about
// what an operation MEANS has moved: 3968e78 changed one parameter's help
// text and every engine built before it stopped serving, and 5d2a8c9 deleted
// the Contrast step's `bake` parameter, which no code had ever read, and did
// the same. A compatibility key that a help string can break is not a
// compatibility key. So the key is kOpsGeneration, a number bumped by hand
// when the operation set changes in a way that makes a pipeline mean
// something else or be refused (an operation added or removed, a parameter
// renamed, retyped, or given a new meaning) and NOT for wording, ordering,
// ranges, or the removal of a parameter nothing reads. The hash stays: it is
// the build's identity, it is what BUILD.json and `version` have always
// carried, and it is the only thing an engine built before the generation
// existed reports -- those are matched against kAcceptedOlderOpsSchemas.

#include <string>

#include <nlohmann/json.hpp>

namespace sirius::app {

    // The engine's method set (pipeline_run, dataset_*, ...): bumped whenever
    // a method is added, removed or changes meaning. 1 = P1 (datasets served
    // in C++, the rest relayed to the Python worker); 2 = P2 (pipeline_run and
    // the outputs kept on the node, step_preview, step_validate, output_stats,
    // put_file, stat_file, outputs_release, cache_status).
    inline constexpr int kEngineApiVersion = 2;

    // The operation set's version: the compatibility key between an
    // application and a cluster worker's engine. BUMP IT when the operation
    // set changes incompatibly -- an operation added or removed, a parameter
    // renamed or retyped, a parameter's meaning changed -- and leave it alone
    // for help text, labels, ranges, ordering, and parameters nothing reads.
    // 1 = the set at 5d2a8c9 (the first build to carry a generation).
    // Regenerating bindings/python/sirius/op_schema.json is the moment to
    // decide: tests/test_app_engine.cpp ("the hash is the committed
    // snapshot's") fails until the snapshot is regenerated, and this comment
    // is what that failure should send the author to.
    inline constexpr int kOpsGeneration = 1;

    // Engines built before kOpsGeneration existed report only an ops_schema
    // hash. These are the hashes whose operation sets this build can serve
    // anyway, each with the reason -- an entry is a statement that the
    // difference cannot change what a pipeline means, not a convenience.
    struct AcceptedOpsSchema {
        const char* hash;
        const char* why;
    };
    inline constexpr AcceptedOpsSchema kAcceptedOlderOpsSchemas[] = {
        {"74a6ee51680740973c34356aeee322c206a076ec421c8000d22c1bc4f58dcc67",
         "differs from this build's snapshot only by the Contrast step's `bake` parameter, which no code read (5d2a8c9)"},
    };

    struct BuildInfo {
        std::string version;     // "0.1.0"
        std::string build;       // "0.1.0+g7cc582c", "0.1.0+g7cc582c.dirty"
        std::string commit;      // the full hash; "unknown" without git
        bool dirty = false;
        std::string opsSchema;   // SHA-256 hex of the operation schema snapshot
        int api = kEngineApiVersion;
        // 0 = the build reports none (it predates kOpsGeneration): older than
        // this one, and judged by its opsSchema.
        int opsGeneration = kOpsGeneration;
    };

    // This build's.
    const BuildInfo& buildInfo();

    // {"build", "version", "commit", "dirty", "ops_schema", "ops_generation",
    // "api"}: the object BUILD.json holds and the engine's hello carries (with
    // more fields).
    nlohmann::json toJson(const BuildInfo& info);
    // The fields toJson writes; missing ones stay empty, and both versions
    // stay 0 -- api 0 and ops_generation 0 mean "this build does not say".
    BuildInfo buildInfoFromJson(const nlohmann::json& j);

    // "" when an engine built as `engine` may serve an application built as
    // `app`: the same operation generation (or, for an engine that reports
    // none, an accepted schema) and the same engine API. Otherwise the
    // refusal, which names both builds, SAYS WHICH SIDE IS OLDER, and gives
    // the fix for that side.
    std::string engineMismatch(const BuildInfo& app, const BuildInfo& engine);

    // The kAcceptedOlderOpsSchemas entry for `hash`, or null. This build's
    // own hash is never one of them (tests/test_app_engine.cpp pins that).
    const AcceptedOpsSchema* acceptedOlderOpsSchema(const std::string& hash);

} // namespace sirius::app

#endif // SIRIUS_APP_BUILD_INFO_HPP
