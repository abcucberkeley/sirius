#include "imgui/panels/llm_client.hpp"

#include <algorithm>
#include <utility>

#include "imgui/strings.hpp"

namespace sirius::app::gui {

    namespace {

        using json = nlohmann::json;

        std::string joinUrl(const std::string& base, const std::string& path) {
            std::string b = trimmed(base);
            while (!b.empty() && b.back() == '/') b.pop_back();
            return b + path;
        }

        // Never throws: what does not parse is "discarded", which is no object.
        json parse(const std::string& text) { return json::parse(text, nullptr, false); }

        // Text from a server may hold bytes that are not UTF-8; they are
        // replaced rather than thrown over.
        std::string dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

        const json& member(const json& j, const char* key) {
            static const json none;
            if (!j.is_object()) return none;
            const auto it = j.find(key);
            return it == j.end() ? none : *it;
        }

        std::string text(const json& j, const char* key) {
            const json& v = member(j, key);
            return v.is_string() ? v.get<std::string>() : std::string();
        }

        // The array under `key`, or an empty one.
        const json& list(const json& j, const char* key) {
            static const json empty = json::array();
            const json& v = member(j, key);
            return v.is_array() ? v : empty;
        }

        // The arguments of a call as the text the API carries them in: some
        // servers send the object itself.
        std::string argumentsText(const json& a) {
            if (a.is_string()) return a.get<std::string>();
            return a.is_object() ? dump(a) : std::string("{}");
        }

        int characters(const std::string& utf8) {
            int n = 0;
            for (const char c : utf8)
                if ((static_cast<unsigned char>(c) & 0xC0) != 0x80) ++n;
            return n;
        }

        bool contains(const std::vector<std::string>& names, const std::string& name) {
            return std::find(names.begin(), names.end(), name) != names.end();
        }

        // Before a request has shown to be a stream, its body is kept as it
        // came: it may be a JSON error, or a whole answer from a server that
        // ignores stream=true.
        constexpr std::size_t kRawLimit = 1u << 20;

    } // namespace

    // --- accumulator -----------------------------------------------------------

    void LlmClient::Accumulator::mergeDelta(const json& delta) {
        if (member(delta, "content").is_string()) content += member(delta, "content").get<std::string>();
        // OpenAI-style "reasoning" / "reasoning_content": not shown, but
        // counted, so a long think is not a dead panel
        for (const char* key : {"reasoning", "reasoning_content"})
            if (member(delta, key).is_string()) reasoningChars += characters(member(delta, key).get<std::string>());
        const json& calls = list(delta, "tool_calls");
        for (std::size_t i = 0; i < calls.size(); ++i) {
            const json& c = calls[i];
            const json& at = member(c, "index");
            const int index = at.is_number() ? static_cast<int>(at.get<double>()) : static_cast<int>(i);
            ToolCall& tc = toolCalls[index];
            if (!text(c, "id").empty()) tc.id = text(c, "id");
            const json& fn = member(c, "function");
            // A name is never streamed in pieces, and some servers repeat it
            // in every delta: appending made "set_viewset_view".
            if (!text(fn, "name").empty()) tc.name = text(fn, "name");
            if (fn.is_object() && fn.contains("arguments")) tc.arguments += argumentsText(fn["arguments"]);
        }
    }

    void LlmClient::Accumulator::mergeMessage(const json& message) {
        if (member(message, "content").is_string()) content += member(message, "content").get<std::string>();
        const json& calls = list(message, "tool_calls");
        for (std::size_t i = 0; i < calls.size(); ++i) {
            const json& call = calls[i];
            ToolCall tc;
            tc.id = text(call, "id");
            const json& fn = member(call, "function");
            tc.name = text(fn, "name");
            tc.arguments = argumentsText(member(fn, "arguments"));
            toolCalls[static_cast<int>(toolCalls.size())] = tc;
        }
    }

    json LlmClient::Accumulator::toMessage() const {
        json m = {{"role", "assistant"}, {"content", content}};
        if (!toolCalls.empty()) {
            json calls = json::array();
            int n = 0;
            for (const auto& [index, tc] : toolCalls) {
                (void)index;
                if (tc.name.empty()) continue;
                json call;
                call["id"] = tc.id.empty() ? "call_" + std::to_string(n) : tc.id;
                call["type"] = "function";
                call["function"] = {{"name", tc.name}, {"arguments", tc.arguments.empty() ? std::string("{}") : tc.arguments}};
                calls.push_back(std::move(call));
                ++n;
            }
            if (!calls.empty()) m["tool_calls"] = std::move(calls);
        }
        return m;
    }

    std::string LlmClient::errorMessageOf(const std::string& body, const std::string& fallback) {
        const json doc = parse(body);
        if (doc.is_object()) {
            const json& err = member(doc, "error");
            if (err.is_object() && !text(err, "message").empty()) return text(err, "message");
            if (err.is_string() && !err.get<std::string>().empty()) return err.get<std::string>();
        }
        const std::string shown = trimmed(body);
        if (!shown.empty() && shown.size() < 400) return fallback + ": " + shown;
        return fallback;
    }

    // --- client ------------------------------------------------------------------

    LlmClient::LlmClient() = default;

    LlmClient::~LlmClient() {
        abort();
        // the lists still on their way: their callbacks are never called
        lookups_.clear();
    }

    void LlmClient::abort() {
        if (!active_) return;
        active_ = false;
        chat_.cancel();
    }

    void LlmClient::send(const Request& request) {
        abort();
        request_ = request;
        start(request.stream);
    }

    void LlmClient::start(bool stream) {
        streaming_ = stream;
        sawData_ = false;
        done_ = false;
        buffer_.clear();
        raw_.clear();
        acc_ = Accumulator{};

        json body;
        body["model"] = request_.model;
        body["messages"] = request_.messages;
        body["temperature"] = request_.temperature;
        body["stream"] = stream;
        if (request_.tools.is_array() && !request_.tools.empty()) {
            body["tools"] = request_.tools;
            body["tool_choice"] = "auto";
        }
        http::Request req;
        req.method = "POST";
        req.url = joinUrl(request_.baseUrl, "/chat/completions");
        req.headers.emplace_back("Content-Type", "application/json");
        req.headers.emplace_back("Accept", stream ? "text/event-stream" : "application/json");
        req.headers.emplace_back("HTTP-Referer", "https://github.com/abcucberkeley/sirius");
        req.headers.emplace_back("X-Title", "SIRIUS workbench");
        req.bearer = request_.apiKey;
        // ten minutes without a byte, as the Qt client's transfer timeout: a
        // model that loads first is silent for a minute or two
        req.stallSeconds = 10 * 60;
        req.body = dump(body);

        http::Fetch::Handlers handlers;
        if (stream) handlers.onData = [this](const std::string& chunk) { onData(chunk); };
        handlers.done = [this](const http::Response& response) { onReplyFinished(response); };
        active_ = true;
        chat_.start(req, std::move(handlers));
    }

    bool LlmClient::consumeSseLine(const std::string& rawLine) {
        std::string line = trimmed(rawLine);
        if (line.empty() || line[0] == ':') return true;
        if (!startsWith(line, "data:")) return true;
        line = trimmed(line.substr(5));
        if (line == "[DONE]") {
            done_ = true;
            return true;
        }
        const json obj = parse(line);
        if (!obj.is_object()) return true;
        if (obj.contains("error")) {
            const std::string msg = errorMessageOf(line, "server error");
            abort();
            if (onFailed) onFailed(msg);
            return false;
        }
        const json& choices = list(obj, "choices");
        if (choices.empty()) return true;
        const json& choice = choices[0];
        sawData_ = true;
        const std::size_t before = acc_.content.size();
        const int reasoningBefore = acc_.reasoningChars;
        acc_.mergeDelta(member(choice, "delta"));
        const std::string finish = text(choice, "finish_reason");
        if (!finish.empty()) acc_.finishReason = finish;
        // the callbacks last: what they start (a new request) must find this one done with
        const bool more = acc_.content.size() > before;
        const bool thought = acc_.reasoningChars > reasoningBefore;
        const std::string fragment = more ? acc_.content.substr(before) : std::string();
        const int reasoning = acc_.reasoningChars;
        if (more && onDelta) onDelta(fragment);
        if (thought && onThinking) onThinking(reasoning);
        return true;
    }

    void LlmClient::onData(const std::string& chunk) {
        if (!active_ || !streaming_) return;
        if (!sawData_ && raw_.size() < kRawLimit) raw_ += chunk;
        buffer_ += chunk;
        std::size_t nl;
        while ((nl = buffer_.find('\n')) != std::string::npos) {
            const std::string line = buffer_.substr(0, nl);
            buffer_.erase(0, nl + 1);
            if (!consumeSseLine(line)) return;   // onFailed: the reply was aborted
        }
    }

    void LlmClient::onReplyFinished(const http::Response& response) {
        if (!active_) return;
        active_ = false;
        const long status = response.status;
        const bool failed = !response.ok();

        if (streaming_) {
            // the body may be JSON (an error, or a server ignoring stream=true)
            const std::string whole = trimmed(raw_);
            if (!sawData_ && !whole.empty() && whole[0] == '{') {
                const json doc = parse(whole);
                if (doc.is_object() && doc.contains("choices")) {
                    const json& choices = list(doc, "choices");
                    acc_.mergeMessage(choices.empty() ? json() : member(choices[0], "message"));
                    const json message = acc_.toMessage();
                    if (onFinished) onFinished(message);
                    return;
                }
            }
            // The last event may be an error with no newline after it, so it
            // is only parsed here: onFailed is then the reply's last word.
            const std::string rest = buffer_;
            buffer_.clear();
            active_ = true;   // consumeSseLine aborts the request it belongs to
            for (const std::string& line : split(rest, '\n'))
                if (!consumeSseLine(line)) return;
            active_ = false;
            if (failed && !sawData_) {
                // servers that reject streaming answer 4xx: retry once without it
                if (status >= 400 && status < 500 && status != 401 && status != 403 && status != 429 && request_.stream) {
                    request_.stream = false;
                    start(false);
                    return;
                }
                if (onFailed) onFailed(errorMessageOf(raw_, response.message()));
                return;
            }
            const json message = acc_.toMessage();
            if (onFinished) onFinished(message);
            return;
        }

        if (failed) {
            if (onFailed) onFailed(errorMessageOf(response.body, response.message()));
            return;
        }
        const json doc = parse(response.body);
        if (!doc.is_object()) {
            if (onFailed) onFailed("unexpected reply from the model server");
            return;
        }
        if (doc.contains("error")) {
            if (onFailed) onFailed(errorMessageOf(response.body, "server error"));
            return;
        }
        const json& choices = list(doc, "choices");
        if (choices.empty()) {
            if (onFailed) onFailed("the model returned no choices");
            return;
        }
        acc_.mergeMessage(member(choices[0], "message"));
        const json message = acc_.toMessage();
        const std::string content = acc_.content;
        if (!content.empty() && onDelta) onDelta(content);
        if (onFinished) onFinished(message);
    }

    std::string LlmClient::resolveModel(const std::string& typed, const std::vector<std::string>& listed) {
        const std::string t = trimmed(typed);
        if (t.empty() || contains(listed, t)) return t;
        const std::string lower = toLower(t);
        for (const std::string& name : listed)
            if (toLower(name) == lower) return name;
        std::string unique;
        for (const std::string& name : listed) {
            if (!startsWith(toLower(name), lower)) continue;
            if (!unique.empty()) return t;   // several: the user has to say which
            unique = name;
        }
        return unique.empty() ? t : unique;
    }

    http::Fetch& LlmClient::lookup() {
        lookups_.erase(std::remove_if(lookups_.begin(), lookups_.end(), [](const std::unique_ptr<http::Fetch>& f) { return !f->busy(); }),
                       lookups_.end());
        lookups_.push_back(std::make_unique<http::Fetch>());
        return *lookups_.back();
    }

    void LlmClient::fetchLoadedModels(const std::string& baseUrl, ListCallback done) {
        // the OpenAI-compatible base ends in /v1; Ollama's own API sits beside it
        std::string root = trimmed(baseUrl);
        while (!root.empty() && root.back() == '/') root.pop_back();
        if (endsWith(root, "/v1")) root.erase(root.size() - 3);
        http::Request req;
        req.url = root + "/api/ps";
        req.connectTimeoutSeconds = 3;
        req.timeoutSeconds = 3;
        http::Fetch::Handlers handlers;
        handlers.done = [done = std::move(done)](const http::Response& response) {
            if (!done) return;
            if (!response.ok()) {
                done({}, response.message());
                return;
            }
            const json doc = parse(response.body);
            if (!doc.is_object() || !doc.contains("models")) {
                done({}, "not an Ollama server");
                return;
            }
            std::vector<std::string> names;
            for (const json& m : list(doc, "models"))
                for (const char* key : {"name", "model"}) {
                    const std::string n = text(m, key);
                    if (!n.empty() && !contains(names, n)) names.push_back(n);
                }
            done(std::move(names), {});
        };
        lookup().start(req, std::move(handlers));
    }

    void LlmClient::fetchModels(const std::string& baseUrl, const std::string& apiKey, ListCallback done) {
        http::Request req;
        req.url = joinUrl(baseUrl, "/models");
        req.bearer = apiKey;
        req.connectTimeoutSeconds = 5;
        req.timeoutSeconds = 5;
        http::Fetch::Handlers handlers;
        handlers.done = [done = std::move(done)](const http::Response& response) {
            if (!done) return;
            if (!response.ok()) {
                done({}, response.message());
                return;
            }
            std::vector<std::string> ids;
            // kept whole while the loop runs: list() returns a reference into it
            const json doc = parse(response.body);
            for (const json& v : list(doc, "data")) {
                const std::string id = text(v, "id");
                if (!id.empty()) ids.push_back(id);
            }
            done(std::move(ids), {});
        };
        lookup().start(req, std::move(handlers));
    }

} // namespace sirius::app::gui
