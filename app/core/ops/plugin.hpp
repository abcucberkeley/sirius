#ifndef SIRIUS_APP_OPS_PLUGIN_HPP
#define SIRIUS_APP_OPS_PLUGIN_HPP

// User operations: Python files the worker loads (app/python/sirius_worker/
// plugins.py documents the file format). The worker describes each one as a
// JSON spec; PluginOperation turns the spec into an Operation whose run()
// ships the array to the worker and takes the result back, so a plugin gets
// the parameter form, the ops menu, undo, the assistant's tools and a help
// page without any code in the application.

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/operation.hpp"
#include "core/rpc.hpp"

namespace sirius::app {

    // Operation from a worker spec ({"kind", "name", "group", "params": [...],
    // "separable_over_t", "produces_labels", "needs_labels", "help", "file"}).
    // Throws std::invalid_argument for a spec that cannot be used.
    std::unique_ptr<Operation> makePluginOperation(const nlohmann::json& spec);

    struct PluginLoadResult {
        std::vector<std::string> kinds;      // registered (or re-registered) operation kinds
        // Kinds registered before that no listed file provides any more: now
        // stand-ins (OpInfo::missing) that keep the steps naming them.
        std::vector<std::string> removed;
        std::vector<std::string> errors;     // "file: reason" for plugins that did not load
        std::vector<std::string> dirs;       // directories the worker searched
        struct Entry {
            std::string kind, name, file, error;
        };
        std::vector<Entry> entries;          // every plugin file the worker saw
    };
    // The per-user plugin directory (~/.sirius/plugins), created when `create`.
    std::string userPluginDirectory(bool create = false);
    // Asks the worker for its plugins (re-importing them when `reload`) and
    // registers every valid one; a kind that collides with a built-in
    // operation is reported as an error, not registered. A kind registered
    // by an earlier call whose file the worker no longer lists is replaced by
    // a stand-in (PluginLoadResult::removed).
    PluginLoadResult registerPluginOperations(RemoteWorker& worker, bool reload);
    // Kinds registered so far by registerPluginOperations.
    std::vector<std::string> pluginKinds();

} // namespace sirius::app

#endif // SIRIUS_APP_OPS_PLUGIN_HPP
