#ifndef SIRIUS_IMGUI_LLM_CLIENT_HPP
#define SIRIUS_IMGUI_LLM_CLIENT_HPP

// Client for OpenAI-compatible chat completion endpoints (Ollama's /v1,
// OpenRouter, anything else speaking the same JSON) with tool calling and
// server-sent-event streaming. One request at a time; the caller (the
// assistant panel) runs the tool loop on top of onFinished. Over
// http::Fetch: every callback arrives on the GUI thread, between two frames.

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "imgui/http.hpp"

namespace sirius::app::gui {

    class LlmClient {
    public:
        struct Request {
            std::string baseUrl;          // "http://localhost:11434/v1"
            std::string model;
            std::string apiKey;           // Bearer token when non-empty
            nlohmann::json messages = nlohmann::json::array();   // OpenAI chat messages
            nlohmann::json tools = nlohmann::json::array();      // OpenAI tool schemas (may be empty)
            bool stream = true;
            double temperature = 0.2;
        };
        using ListCallback = std::function<void(std::vector<std::string> names, std::string error)>;

        LlmClient();
        ~LlmClient();
        LlmClient(const LlmClient&) = delete;
        LlmClient& operator=(const LlmClient&) = delete;

        void send(const Request& request);
        // Nothing of the request in flight is reported after this returns.
        void abort();
        bool busy() const noexcept { return active_; }

        // GET {baseUrl}/models; `done(ids, error)` on the GUI thread.
        void fetchModels(const std::string& baseUrl, const std::string& apiKey, ListCallback done);
        // Ollama only: the models it holds in memory right now (GET /api/ps
        // on the server behind `baseUrl`), so the panel can say up front
        // that the first answer will wait for a load. Other servers answer
        // with an error, which the caller treats as "unknown".
        void fetchLoadedModels(const std::string& baseUrl, ListCallback done);

        // A model name as typed into an editable dropdown, against the names
        // the server lists: the listed name it matches ignoring case, or the
        // one it is a prefix of when that is unique ("gemma" -> "gemma4:31b"),
        // else the text as typed -- a name the server does not list yet may
        // still be right, and the answer will say if it is not.
        static std::string resolveModel(const std::string& typed, const std::vector<std::string>& listed);

        // Parses one SSE "data:" payload (or a whole non-streaming body) into
        // the accumulator; exposed for tests.
        struct ToolCall {
            std::string id, name, arguments;
        };
        struct Accumulator {
            std::string content;
            int reasoningChars = 0;                  // hidden reasoning seen so far
            std::map<int, ToolCall> toolCalls;       // by index
            std::string finishReason;
            void mergeDelta(const nlohmann::json& delta);
            void mergeMessage(const nlohmann::json& message);
            nlohmann::json toMessage() const;        // {"role":"assistant","content":...,"tool_calls":[...]}
        };
        static std::string errorMessageOf(const std::string& body, const std::string& fallback);

        // --- events, on the GUI thread ---------------------------------------
        std::function<void(const std::string& text)> onDelta;               // streamed content fragment
        std::function<void(const nlohmann::json& message)> onFinished;      // complete assistant message
        std::function<void(const std::string& error)> onFailed;
        // A "thinking" model streams its reasoning before any content (the
        // `reasoning` field of a delta): how many characters of it so far,
        // so the panel can show that something is happening.
        std::function<void(int reasoningChars)> onThinking;

    private:
        void start(bool stream);
        void onData(const std::string& chunk);
        void onReplyFinished(const http::Response& response);
        // False once the line was an error event and onFailed has been
        // called: the reply is over, and nothing may follow it -- above all
        // not onFinished, which cleared the error and added an empty turn.
        bool consumeSseLine(const std::string& line);
        // A request of its own for each list, so that one asked for while
        // another is on its way does not take its place (and its answer).
        http::Fetch& lookup();

        http::Fetch chat_;
        std::vector<std::unique_ptr<http::Fetch>> lookups_;
        bool active_ = false;
        Request request_;
        bool streaming_ = true;
        bool sawData_ = false;
        bool done_ = false;
        std::string buffer_;               // the line being received
        std::string raw_;                  // the body until it proved to be a stream
        Accumulator acc_;
    };

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_LLM_CLIENT_HPP
