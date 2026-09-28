#ifndef SIRIUS_APP_TOOL_API_HPP
#define SIRIUS_APP_TOOL_API_HPP

// The typed tool API the assistant (and scripts) drive the workbench with.
// Every tool is a JSON function with a JSON-schema description in the
// OpenAI / Ollama "tools" format; every mutating call goes through the
// workbench, so it is undoable and shows up as an action card.

#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/workbench.hpp"

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
    };

    class ToolApi {
    public:
        explicit ToolApi(Workbench& wb);

        // OpenAI-format tool list: [{"type":"function","function":{name, description, parameters}}]
        nlohmann::json schemas() const;
        std::vector<std::string> toolNames() const;
        // Runs a tool; errors come back as {"error": "..."} rather than throwing.
        nlohmann::json call(const std::string& name, const nlohmann::json& args);

        // Compact description of the current state for the system prompt:
        // dataset, ops stack, selected step's params, diagnostics summary.
        nlohmann::json contextSnapshot() const;
        std::string systemPrompt() const;

        // Runs are asynchronous in the app: the hook starts one and returns
        // its JSON outcome once finished (the application blocks the assistant
        // loop, not the GUI). Without a hook, run tools report "unavailable".
        void setRunHook(std::function<nlohmann::json(int targetIndex)> hook) { runHook_ = std::move(hook); }
        // Help page lookup (markdown text for a kind).
        void setHelpHook(std::function<std::string(const std::string& kind)> hook) { helpHook_ = std::move(hook); }

        const std::vector<ActionRecord>& actions() const noexcept { return actions_; }
        std::vector<ActionRecord> takeActions();

    private:
        struct Tool {
            std::string name, description;
            nlohmann::json parameters;
            std::function<nlohmann::json(const nlohmann::json&)> fn;
        };
        void add(Tool t);
        int resolveStep(const nlohmann::json& args, const char* key = "step") const;   // throws with a message
        nlohmann::json stepJson(int index) const;

        Workbench& wb_;
        std::vector<Tool> tools_;
        std::vector<ActionRecord> actions_;
        std::function<nlohmann::json(int)> runHook_;
        std::function<std::string(const std::string&)> helpHook_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_TOOL_API_HPP
