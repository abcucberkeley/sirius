#ifndef SIRIUS_APP_TOOL_API_HPP
#define SIRIUS_APP_TOOL_API_HPP

// The typed tool API the assistant (and scripts) drive the workbench with.
// Every tool is a JSON function with a JSON-schema description in the
// OpenAI / Ollama "tools" format; every mutating call goes through the
// workbench, so it is undoable and shows up as an action card.

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/workbench.hpp"
#include "core/worker_error.hpp"   // call() maps WorkerStartError

namespace sirius::app {

    struct ActionRecord {
        enum class Kind { Param,
                          Run,
                          View,
                          Edit,
                          Info };
        Kind kind = Kind::Info;
        std::string text;                 // "Step 02 · Wiener 0.001 → 0.002"
        std::string link;                 // "undo", "view", "log", ""
        nlohmann::json viewState;         // for "view": what to restore
        std::string toolName;
        // For "undo": the history's revision before and after the call that
        // made the change, so the link undoes that change and nothing newer.
        // A call that pushed nothing gets no "undo" link.
        std::uint64_t revBefore = 0, revAfter = 0;
    };

    // What a tool throws to fail with an error code of its own ("unknown_step",
    // "not_computed", ...), a hint at the next step and data for the caller;
    // call() turns it into {"error", "error_kind", "hint", "data"}.
    class ToolFailure : public std::runtime_error {
    public:
        ToolFailure(std::string code, const std::string& message, std::string hint = {}, nlohmann::json data = nullptr);
        const std::string& code() const noexcept;
        const std::string& hint() const noexcept;
        const nlohmann::json& data() const noexcept;

    private:
        std::string code_, hint_;
        nlohmann::json data_;
    };

    // Field order matters: the existing tools are built with positional initialisers
    // add({"name", "description", parameters, fn}) (C++17: no designated initialisers).
    // Every field after fn has a default member initialiser, so that such a
    // brace list leaves none of them out as far as GCC's and Clang's
    // -Wmissing-field-initializers (-Wextra) are concerned.
    struct ToolSpec {
        std::string name, description;
        nlohmann::json parameters;                                   // JSON schema, type object
        std::function<nlohmann::json(const nlohmann::json&)> fn;
        std::string title{};
        bool refusedWhileRunning = false;                            // call() answers "busy" during a run
        bool readOnly = false, destructive = false, idempotent = false, openWorld = false;
        nlohmann::json meta = nlohmann::json::object();              // MCP Tool._meta
    };

    // True for a path that names another machine or a device rather than a
    // local file: \\server\share, //server/share, \\?\UNC\server\share,
    // \\.\pipe\x (the \\?\C:\ long-path form of a local drive is local).
    // Opening one makes Windows connect to that server with the user's
    // credentials, so the tools an agent drives refuse it unless the host
    // allows network paths.
    bool isNetworkPath(const std::string& path);

    class ToolApi {
    public:
        explicit ToolApi(Workbench& wb);

        // OpenAI-format tool list: [{"type":"function","function":{name, description, parameters}}]
        nlohmann::json schemas() const;
        std::vector<std::string> toolNames() const;
        // Runs a tool; errors come back as a value rather than thrown.
        // call(): a failure is {"error": msg, "error_kind": code[, "hint": ..][, "data": ..]}; a result
        // without "error_kind" is a success even when it has an "error" key (the run hook's "error":"").
        // codes: unknown_tool, busy, invalid_argument, unknown_step, cancelled, worker_unavailable,
        //        unsupported, failed, or a ToolFailure's own.
        // A tool flagged refusedWhileRunning answers "busy" during a run
        // without being called: the workbench would refuse its edit anyway,
        // but only with a log line, and the tool would report success.
        nlohmann::json call(const std::string& name, const nlohmann::json& args);

        // Compact description of the current state for the system prompt:
        // dataset, ops stack, selected step's params, diagnostics summary.
        nlohmann::json contextSnapshot() const;
        std::string systemPrompt() const;

        // Runs are asynchronous in the app: the hook starts one and returns
        // its JSON outcome once finished (the application blocks the assistant
        // loop, not the GUI). Without a hook, the run tool fails as "unsupported".
        void setRunHook(std::function<nlohmann::json(int targetIndex)> hook) { runHook_ = std::move(hook); }
        // Help page lookup (markdown text for a kind).
        void setHelpHook(std::function<std::string(const std::string& kind)> hook) { helpHook_ = std::move(hook); }

        const std::vector<ActionRecord>& actions() const noexcept { return actions_; }
        std::vector<ActionRecord> takeActions();

        // The tool table, for a host that adds tools of its own or replaces
        // some (sirius-cli). Every tool has a title and its hints; schemas()
        // shows only the name, the description and the parameters.
        void addTool(ToolSpec spec);                                 // replaces a tool of the same name, else appends
        bool removeTool(const std::string& name);
        const ToolSpec* findTool(const std::string& name) const;
        const std::vector<ToolSpec>& tools() const noexcept;
        // args[key] as a 0-based step index: a number (1 = Load), or a step's
        // name or kind. Throws ToolFailure: "unknown_step" for a step there
        // is not, "invalid_argument" for a value missing or of another type.
        static int resolveStepIndex(const Pipeline& p, const nlohmann::json& args, const char* key = "step");
        nlohmann::json stepJson(int index) const;
        void noteAction(ActionRecord r);
        // Whether a Path parameter (add_step, set_params) or a directory a
        // tool writes to may be a network path (isNetworkPath); off by default.
        void setAllowNetworkPaths(bool on) noexcept { allowNetworkPaths_ = on; }
        bool allowNetworkPaths() const noexcept { return allowNetworkPaths_; }

    private:
        void add(ToolSpec t);
        int resolveStep(const nlohmann::json& args, const char* key = "step") const;   // throws with a message

        Workbench& wb_;
        std::vector<ToolSpec> tools_;
        std::vector<ActionRecord> actions_;
        std::function<nlohmann::json(int)> runHook_;
        std::function<std::string(const std::string&)> helpHook_;
        bool allowNetworkPaths_ = false;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_TOOL_API_HPP
