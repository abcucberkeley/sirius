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
// Whether an engine may serve this application is engineMismatch(): the op
// schema and the engine API must be the same; the commit may differ (the
// user's choice: a dev build talks to an image built a commit earlier as long
// as no operation changed).

#include <string>

#include <nlohmann/json.hpp>

namespace sirius::app {

    // The engine's method set (pipeline_run, dataset_*, ...): bumped whenever
    // a method is added, removed or changes meaning. 1 = P1 (datasets served
    // in C++, the rest relayed to the Python worker).
    inline constexpr int kEngineApiVersion = 1;

    struct BuildInfo {
        std::string version;     // "0.1.0"
        std::string build;       // "0.1.0+g7cc582c", "0.1.0+g7cc582c.dirty"
        std::string commit;      // the full hash; "unknown" without git
        bool dirty = false;
        std::string opsSchema;   // SHA-256 hex of the operation schema snapshot
        int api = kEngineApiVersion;
    };

    // This build's.
    const BuildInfo& buildInfo();

    // {"build", "version", "commit", "dirty", "ops_schema", "api"}: the object
    // BUILD.json holds and the engine's hello carries (with more fields).
    nlohmann::json toJson(const BuildInfo& info);
    // The fields toJson writes; missing ones stay empty (api 0).
    BuildInfo buildInfoFromJson(const nlohmann::json& j);

    // "" when an engine built as `engine` may serve an application built as
    // `app`: the same operation schema and engine API. Otherwise the refusal,
    // naming both builds and the fix.
    std::string engineMismatch(const BuildInfo& app, const BuildInfo& engine);

} // namespace sirius::app

#endif // SIRIUS_APP_BUILD_INFO_HPP
