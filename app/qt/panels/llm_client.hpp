#ifndef SIRIUS_APP_LLM_CLIENT_HPP
#define SIRIUS_APP_LLM_CLIENT_HPP

// Client for OpenAI-compatible chat completion endpoints (Ollama's /v1,
// OpenRouter, anything else speaking the same JSON) with tool calling and
// server-sent-event streaming. One request at a time; the caller (the
// assistant panel) runs the tool loop on top of finished().

#include <functional>
#include <map>

#include <QByteArray>
#include <QJsonArray>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QObject>
#include <QString>
#include <QStringList>

class QNetworkReply;

namespace sirius::app {

    class LlmClient : public QObject {
        Q_OBJECT
    public:
        struct Request {
            QString baseUrl;          // "http://localhost:11434/v1"
            QString model;
            QString apiKey;           // Bearer token when non-empty
            QJsonArray messages;      // OpenAI chat messages
            QJsonArray tools;         // OpenAI tool schemas (may be empty)
            bool stream = true;
            double temperature = 0.2;
        };

        explicit LlmClient(QObject* parent = nullptr);
        ~LlmClient() override;

        void send(const Request& request);
        void abort();
        bool busy() const noexcept { return reply_ != nullptr; }

        // GET {baseUrl}/models; `done(ids, error)` on the GUI thread.
        // Ollama only: the models it holds in memory right now (GET /api/ps
        // on the server behind `baseUrl`), so the panel can say up front
        // that the first answer will wait for a load. Other servers answer
        // with an error, which the caller treats as "unknown".
        void fetchLoadedModels(const QString& baseUrl, std::function<void(QStringList, QString)> done);
        void fetchModels(const QString& baseUrl, const QString& apiKey,
                         std::function<void(QStringList ids, QString error)> done);

        // Parses one SSE "data:" payload (or a whole non-streaming body) into
        // the accumulator; exposed for tests.
        struct ToolCall {
            QString id, name, arguments;
        };
        struct Accumulator {
            QString content;
            int reasoningChars = 0;                      // hidden reasoning seen so far
            std::map<int, ToolCall> toolCalls;   // by index
            QString finishReason;
            void mergeDelta(const QJsonObject& delta);
            void mergeMessage(const QJsonObject& message);
            QJsonObject toMessage() const;       // {"role":"assistant","content":...,"tool_calls":[...]}
        };
        static QString errorMessageOf(const QByteArray& body, const QString& fallback);

    signals:
        void delta(const QString& text);              // streamed content fragment
        void finished(const QJsonObject& message);    // complete assistant message
        void failed(const QString& error);
        // A "thinking" model streams its reasoning before any content (the
        // `reasoning` field of a delta): how many characters of it so far,
        // so the panel can show that something is happening.
        void thinking(int reasoningChars);

    private:
        void start(bool stream);
        void onReadyRead();
        void onReplyFinished();
        void consumeSseLine(const QByteArray& line);

        QNetworkAccessManager nam_;
        QNetworkReply* reply_ = nullptr;
        Request request_;
        bool streaming_ = true;
        bool sawData_ = false;
        bool done_ = false;
        QByteArray buffer_;
        Accumulator acc_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_LLM_CLIENT_HPP
