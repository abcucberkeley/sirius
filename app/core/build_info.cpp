#include "core/build_info.hpp"

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
            return b;
        }();
        return info;
    }

    json toJson(const BuildInfo& info) {
        return {{"build", info.build}, {"version", info.version}, {"commit", info.commit}, {"dirty", info.dirty}, {"ops_schema", info.opsSchema}, {"api", info.api}};
    }

    BuildInfo buildInfoFromJson(const json& j) {
        BuildInfo b;
        b.api = 0;
        if (!j.is_object()) return b;
        const auto text = [&](const char* key) {
            const auto it = j.find(key);
            return it != j.end() && it->is_string() ? it->get<std::string>() : std::string();
        };
        b.build = text("build");
        b.version = text("version");
        b.commit = text("commit");
        b.opsSchema = text("ops_schema");
        if (auto it = j.find("dirty"); it != j.end() && it->is_boolean()) b.dirty = it->get<bool>();
        if (auto it = j.find("api"); it != j.end() && it->is_number_integer()) b.api = it->get<int>();
        return b;
    }

    std::string engineMismatch(const BuildInfo& app, const BuildInfo& engine) {
        const std::string names = "The cluster engine is SIRIUS " + (engine.build.empty() ? std::string("of an unknown build") : engine.build) +
                                  ", this application is " + app.build;
        if (engine.api != app.api)
            return names + ": they speak different engine method sets (engine API " + std::to_string(engine.api) + " here " +
                   std::to_string(app.api) + "). Rebuild the engine image from this commit, or point the profile at a matching engine.";
        if (engine.opsSchema.empty() || engine.opsSchema == "unknown" || app.opsSchema == "unknown" || engine.opsSchema != app.opsSchema)
            return names + ": their operations differ (another operation schema), so a pipeline would not mean the same on "
                           "both. Rebuild the engine image from this commit, or point the profile at a matching engine.";
        return {};
    }

} // namespace sirius::app
