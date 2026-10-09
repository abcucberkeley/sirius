#include "core/build_info.hpp"

#include <cstdlib>

#include "sirius_build_info_generated.hpp"

namespace sirius::app {

    using json = nlohmann::json;

    const BuildInfo& buildInfo() {
        static const BuildInfo info = [] {
            BuildInfo b;
            b.version = SIRIUS_BUILD_VERSION;
            b.build = SIRIUS_BUILD_ID;
            b.commit = SIRIUS_BUILD_COMMIT;
            b.dirty = static_cast<bool>(SIRIUS_BUILD_DIRTY);
            b.opsSchema = SIRIUS_BUILD_OPS_SCHEMA;
            b.api = kEngineApiVersion;
            b.opsGeneration = kOpsGeneration;
            return b;
        }();
        return info;
    }

    json toJson(const BuildInfo& info) {
        return {{"build", info.build},         {"version", info.version},  {"commit", info.commit}, {"dirty", info.dirty},
                {"ops_schema", info.opsSchema}, {"ops_generation", info.opsGeneration}, {"api", info.api}};
    }

    BuildInfo buildInfoFromJson(const json& j) {
        BuildInfo b;
        b.api = 0;
        // 0 means "this build reports no generation", which is how an engine
        // built before kOpsGeneration existed is recognised: not a default.
        b.opsGeneration = 0;
        if (!j.is_object()) return b;
        const auto text = [&](const char* key) {
            const auto it = j.find(key);
            return it != j.end() && it->is_string() ? it->get<std::string>() : std::string();
        };
        b.build = text("build");
        b.version = text("version");
        b.commit = text("commit");
        // sirius-cli calls it ops_schema; a BUILD.json may carry it as schema_hash too (or only)
        b.opsSchema = text("ops_schema");
        if (b.opsSchema.empty()) b.opsSchema = text("schema_hash");
        if (auto it = j.find("dirty"); it != j.end() && it->is_boolean()) b.dirty = it->get<bool>();
        if (auto it = j.find("ops_generation"); it != j.end()) {
            if (it->is_number_integer()) b.opsGeneration = it->get<int>();
            else if (it->is_string()) b.opsGeneration = std::atoi(it->get<std::string>().c_str());
        }
        for (const char* key : {"api", "engine_api"})
            if (auto it = j.find(key); it != j.end() && b.api == 0) {
                if (it->is_number_integer()) b.api = it->get<int>();
                else if (it->is_string()) b.api = std::atoi(it->get<std::string>().c_str());
            }
        return b;
    }

    const AcceptedOpsSchema* acceptedOlderOpsSchema(const std::string& hash) {
        if (hash.empty() || hash == "unknown") return nullptr;
        for (const AcceptedOpsSchema& a : kAcceptedOlderOpsSchemas)
            if (hash == a.hash) return &a;
        return nullptr;
    }

    namespace {

        std::string shortHash(const std::string& h) { return h.size() > 12 ? h.substr(0, 12) : h; }

        // What to do, said to whichever side is behind. `engineOlder` false
        // means this application is the old one, and rebuilding the engine
        // would not help.
        std::string fixFor(bool engineOlder) {
            return engineOlder ? " The engine is the older side: rebuild the engine image (or the engine build folder) from this "
                                 "commit, or point the profile at an engine built from it."
                               : " This application is the older side: update it to the engine's commit, or point the profile at an "
                                 "engine built from this application's commit.";
        }

    } // namespace

    std::string engineMismatch(const BuildInfo& app, const BuildInfo& engine) {
        const std::string names = "The cluster engine is SIRIUS " + (engine.build.empty() ? std::string("of an unknown build") : engine.build) +
                                  ", this application is " + app.build;
        // The method set first: a side that cannot be spoken to at all.
        if (engine.api != app.api)
            return names + ": they speak different engine method sets (engine API " + std::to_string(engine.api) + ", this one " +
                   std::to_string(app.api) + ")." + fixFor(engine.api < app.api);
        // The operation set. Both sides name their generation: the comparison
        // is then exact, and so is which of them is behind.
        if (engine.opsGeneration > 0 && app.opsGeneration > 0) {
            if (engine.opsGeneration == app.opsGeneration) return {};
            return names + ": their operation sets differ (engine operation set " + std::to_string(engine.opsGeneration) + ", this one " +
                   std::to_string(app.opsGeneration) + "), so a pipeline would not mean the same on both." + fixFor(engine.opsGeneration < app.opsGeneration);
        }
        // An engine that reports no generation was built before the
        // generation existed, so it IS the older side: its schema hash is
        // this build's, one this build declares it can serve, or nothing.
        // `<= 0`, not `== 0`: a number that is not a generation at all -- a
        // negative one in a hand-written BUILD.json -- used to fall past this
        // to the last return and be refused with the sentence meant for an
        // application that is itself behind. Either way it is refused; this is
        // the wording. (It was the one line of the previous stage that no
        // build had compiled, so it was recorded in that stage's hand-over
        // rather than shipped unbuilt; this stage's build compiled it, and
        // tests/test_app_engine.cpp drives a negative generation.)
        if (engine.opsGeneration <= 0) {
            if (engine.opsSchema.empty() || engine.opsSchema == "unknown")
                return names + ": the engine reports neither an operation set version nor an operation schema, so there is nothing to "
                               "compare and a pipeline could mean anything on it." +
                       fixFor(true);
            if (app.opsSchema != "unknown" && engine.opsSchema == app.opsSchema) return {};
            // Declared compatible: the difference cannot change what a
            // pipeline means (kAcceptedOlderOpsSchemas says why, per hash).
            if (acceptedOlderOpsSchema(engine.opsSchema)) return {};
            return names + ": the engine predates the operation set version and its operation schema (" + shortHash(engine.opsSchema) +
                   ") is not one this application accepts (" + shortHash(app.opsSchema) + ")." + fixFor(true);
        }
        // This application reports no generation and the engine does: this
        // application is the older side. (A build of this file always reports
        // one, so this is only reachable through a hand-made BuildInfo.)
        return names + ": the engine names an operation set version (" + std::to_string(engine.opsGeneration) +
               ") and this application does not." + fixFor(false);
    }

} // namespace sirius::app
