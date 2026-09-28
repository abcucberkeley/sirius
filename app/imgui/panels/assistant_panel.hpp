#ifndef SIRIUS_IMGUI_ASSISTANT_PANEL_HPP
#define SIRIUS_IMGUI_ASSISTANT_PANEL_HPP

// Assistant dock (330 px): transcript with user bubbles, assistant text and
// action cards, busy indicator, suggestion chips, input and the
// "Ask before acting" toggle. Talks to an OpenAI-compatible chat endpoint
// (Ollama, OpenRouter) with the ToolApi's tools; every tool call is applied
// through the workbench and shown as a card.

#include <memory>
#include <string>

namespace sirius::app::gui {

    class App;

    struct AssistantSettings {
        std::string provider = "ollama";     // "ollama" | "openrouter" | "custom"
        std::string baseUrl = "http://localhost:11434/v1";
        std::string model;
        // OpenRouter / custom: the key in the secret store, else the one in
        // $OPENROUTER_API_KEY (OpenRouter) or $SIRIUS_LLM_API_KEY, in which
        // case apiKeyVariable names the variable.
        std::string apiKey;
        std::string apiKeyVariable;
        bool askBeforeActing = false;

        static AssistantSettings load();     // the settings and the secret store
        // Everything but the key. It runs on every model pick and toggle,
        // and writing the key there would put a key from the environment
        // into the secret store; storeApiKey() writes what the user typed.
        void save() const;
        // The key a request carries: none for Ollama, which takes none, so
        // a key kept for OpenRouter never travels to an Ollama server.
        std::string requestKey() const;
        static bool storeApiKey(const std::string& key);   // empty removes it; false when the store refused
        // The environment's key for `provider`, and which variable held it.
        static std::string environmentKey(const std::string& provider, std::string* variable = nullptr);
    };

    class AssistantPanel {
    public:
        explicit AssistantPanel(App& app);
        ~AssistantPanel();
        AssistantPanel(const AssistantPanel&) = delete;
        AssistantPanel& operator=(const AssistantPanel&) = delete;

        void draw();
        void setSettings(const AssistantSettings& s);
        AssistantSettings settings() const;
        void focusInput();
        // Submit a message as if typed (scripting, tests).
        void ask(const std::string& text);
        // True while a request or a tool loop is in flight.
        bool busy() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_ASSISTANT_PANEL_HPP
