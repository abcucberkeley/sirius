#include "qt/panels/llm_client.hpp"

#include <QJsonDocument>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QUrl>

namespace sirius::app {

    namespace {
        QString joinUrl(const QString& base, const QString& path) {
            QString b = base.trimmed();
            while (b.endsWith('/')) b.chop(1);
            return b + path;
        }
    } // namespace

    // --- accumulator -----------------------------------------------------------

    void LlmClient::Accumulator::mergeDelta(const QJsonObject& delta) {
        if (delta.contains(QStringLiteral("content")) && delta[QStringLiteral("content")].isString())
            content += delta[QStringLiteral("content")].toString();
        // OpenAI-style "reasoning" / "reasoning_content": not shown, but
        // counted, so a long think is not a dead panel
        for (const char* key : {"reasoning", "reasoning_content"})
            if (delta.contains(QLatin1String(key)) && delta[QLatin1String(key)].isString())
                reasoningChars += static_cast<int>(delta[QLatin1String(key)].toString().size());
        const QJsonArray calls = delta[QStringLiteral("tool_calls")].toArray();
        for (int i = 0; i < calls.size(); ++i) {
            const QJsonObject c = calls[i].toObject();
            const int index = c.contains(QStringLiteral("index")) ? c[QStringLiteral("index")].toInt() : i;
            ToolCall& tc = toolCalls[index];
            if (c.contains(QStringLiteral("id")) && !c[QStringLiteral("id")].toString().isEmpty()) tc.id = c[QStringLiteral("id")].toString();
            const QJsonObject fn = c[QStringLiteral("function")].toObject();
            // A name is never streamed in pieces, and some servers repeat it
            // in every delta: appending made "set_viewset_view".
            if (fn.contains(QStringLiteral("name")) && !fn[QStringLiteral("name")].toString().isEmpty()) tc.name = fn[QStringLiteral("name")].toString();
            if (fn.contains(QStringLiteral("arguments"))) {
                const QJsonValue a = fn[QStringLiteral("arguments")];
                tc.arguments += a.isString() ? a.toString() : QString::fromUtf8(QJsonDocument(a.toObject()).toJson(QJsonDocument::Compact));
            }
        }
    }

    void LlmClient::Accumulator::mergeMessage(const QJsonObject& message) {
        const QJsonValue c = message[QStringLiteral("content")];
        if (c.isString()) content += c.toString();
        const QJsonArray calls = message[QStringLiteral("tool_calls")].toArray();
        for (int i = 0; i < calls.size(); ++i) {
            const QJsonObject call = calls[i].toObject();
            ToolCall tc;
            tc.id = call[QStringLiteral("id")].toString();
            const QJsonObject fn = call[QStringLiteral("function")].toObject();
            tc.name = fn[QStringLiteral("name")].toString();
            const QJsonValue a = fn[QStringLiteral("arguments")];
            tc.arguments = a.isString() ? a.toString() : QString::fromUtf8(QJsonDocument(a.toObject()).toJson(QJsonDocument::Compact));
            toolCalls[static_cast<int>(toolCalls.size())] = tc;
        }
    }

    QJsonObject LlmClient::Accumulator::toMessage() const {
        QJsonObject m;
        m[QStringLiteral("role")] = QStringLiteral("assistant");
        m[QStringLiteral("content")] = content;
        if (!toolCalls.empty()) {
            QJsonArray calls;
            int n = 0;
            for (const auto& [index, tc] : toolCalls) {
                if (tc.name.isEmpty()) continue;
                QJsonObject fn;
                fn[QStringLiteral("name")] = tc.name;
                fn[QStringLiteral("arguments")] = tc.arguments.isEmpty() ? QStringLiteral("{}") : tc.arguments;
                QJsonObject call;
                call[QStringLiteral("id")] = tc.id.isEmpty() ? QStringLiteral("call_%1").arg(n) : tc.id;
                call[QStringLiteral("type")] = QStringLiteral("function");
                call[QStringLiteral("function")] = fn;
                calls.append(call);
                ++n;
            }
            if (!calls.isEmpty()) m[QStringLiteral("tool_calls")] = calls;
        }
        return m;
    }

    QString LlmClient::errorMessageOf(const QByteArray& body, const QString& fallback) {
        const QJsonDocument doc = QJsonDocument::fromJson(body);
        if (doc.isObject()) {
            const QJsonValue err = doc.object()[QStringLiteral("error")];
            if (err.isObject()) {
                const QString msg = err.toObject()[QStringLiteral("message")].toString();
                if (!msg.isEmpty()) return msg;
            }
            if (err.isString() && !err.toString().isEmpty()) return err.toString();
        }
        const QString text = QString::fromUtf8(body).trimmed();
        if (!text.isEmpty() && text.size() < 400) return fallback + ": " + text;
        return fallback;
    }

    // --- client ------------------------------------------------------------------

    LlmClient::LlmClient(QObject* parent) : QObject(parent) {}

    LlmClient::~LlmClient() { abort(); }

    void LlmClient::abort() {
        if (!reply_) return;
        QNetworkReply* r = reply_;
        reply_ = nullptr;
        r->disconnect(this);
        r->abort();
        r->deleteLater();
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
        acc_ = Accumulator{};

        QJsonObject body;
        body[QStringLiteral("model")] = request_.model;
        body[QStringLiteral("messages")] = request_.messages;
        body[QStringLiteral("temperature")] = request_.temperature;
        body[QStringLiteral("stream")] = stream;
        if (!request_.tools.isEmpty()) {
            body[QStringLiteral("tools")] = request_.tools;
            body[QStringLiteral("tool_choice")] = QStringLiteral("auto");
        }
        QNetworkRequest req(QUrl(joinUrl(request_.baseUrl, QStringLiteral("/chat/completions"))));
        req.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
        req.setRawHeader("Accept", stream ? "text/event-stream" : "application/json");
        if (!request_.apiKey.isEmpty()) req.setRawHeader("Authorization", ("Bearer " + request_.apiKey).toUtf8());
        req.setRawHeader("HTTP-Referer", "https://github.com/abcucberkeley/sirius");
        req.setRawHeader("X-Title", "SIRIUS workbench");
        req.setTransferTimeout(10 * 60 * 1000);
        reply_ = nam_.post(req, QJsonDocument(body).toJson(QJsonDocument::Compact));
        connect(reply_, &QNetworkReply::readyRead, this, &LlmClient::onReadyRead);
        connect(reply_, &QNetworkReply::finished, this, &LlmClient::onReplyFinished);
    }

    void LlmClient::consumeSseLine(const QByteArray& rawLine) {
        QByteArray line = rawLine.trimmed();
        if (line.isEmpty() || line.startsWith(':')) return;
        if (!line.startsWith("data:")) return;
        line = line.mid(5).trimmed();
        if (line == "[DONE]") {
            done_ = true;
            return;
        }
        const QJsonDocument doc = QJsonDocument::fromJson(line);
        if (!doc.isObject()) return;
        const QJsonObject obj = doc.object();
        if (obj.contains(QStringLiteral("error"))) {
            const QString msg = errorMessageOf(line, QStringLiteral("server error"));
            abort();
            emit failed(msg);
            return;
        }
        const QJsonArray choices = obj[QStringLiteral("choices")].toArray();
        if (choices.isEmpty()) return;
        const QJsonObject choice = choices[0].toObject();
        sawData_ = true;
        const QJsonObject delta = choice[QStringLiteral("delta")].toObject();
        const QString before = acc_.content;
        const int reasoningBefore = acc_.reasoningChars;
        acc_.mergeDelta(delta);
        if (acc_.content.size() > before.size()) emit this->delta(acc_.content.mid(before.size()));
        if (acc_.reasoningChars > reasoningBefore) emit thinking(acc_.reasoningChars);
        const QJsonValue finish = choice[QStringLiteral("finish_reason")];
        if (finish.isString() && !finish.toString().isEmpty()) acc_.finishReason = finish.toString();
    }

    void LlmClient::onReadyRead() {
        if (!reply_ || !streaming_) return;
        buffer_ += reply_->readAll();
        int nl;
        while ((nl = buffer_.indexOf('\n')) >= 0) {
            const QByteArray line = buffer_.left(nl);
            buffer_.remove(0, nl + 1);
            consumeSseLine(line);
            if (!reply_) return;   // failed() aborted the reply
        }
    }

    void LlmClient::onReplyFinished() {
        if (!reply_) return;
        QNetworkReply* r = reply_;
        reply_ = nullptr;
        r->deleteLater();
        const int status = r->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        const QByteArray rest = r->readAll();

        if (streaming_) {
            buffer_ += rest;
            // the body may be JSON (an error, or a server ignoring stream=true)
            const QByteArray trimmed = buffer_.trimmed();
            if (!sawData_ && trimmed.startsWith('{')) {
                const QJsonDocument doc = QJsonDocument::fromJson(trimmed);
                if (doc.isObject() && doc.object().contains(QStringLiteral("choices"))) {
                    acc_.mergeMessage(doc.object()[QStringLiteral("choices")].toArray().at(0).toObject()[QStringLiteral("message")].toObject());
                    emit finished(acc_.toMessage());
                    return;
                }
            }
            for (const QByteArray& line : buffer_.split('\n')) consumeSseLine(line);
            if (r->error() != QNetworkReply::NoError && !sawData_) {
                // servers that reject streaming answer 4xx: retry once without it
                if (status >= 400 && status < 500 && status != 401 && status != 403 && status != 429 && request_.stream) {
                    request_.stream = false;
                    start(false);
                    return;
                }
                emit failed(errorMessageOf(rest, r->errorString()));
                return;
            }
            if (!sawData_ && r->error() != QNetworkReply::NoError) {
                emit failed(errorMessageOf(rest, r->errorString()));
                return;
            }
            emit finished(acc_.toMessage());
            return;
        }

        if (r->error() != QNetworkReply::NoError) {
            emit failed(errorMessageOf(rest, r->errorString()));
            return;
        }
        const QJsonDocument doc = QJsonDocument::fromJson(rest);
        if (!doc.isObject()) {
            emit failed(QStringLiteral("unexpected reply from the model server"));
            return;
        }
        const QJsonObject obj = doc.object();
        if (obj.contains(QStringLiteral("error"))) {
            emit failed(errorMessageOf(rest, QStringLiteral("server error")));
            return;
        }
        const QJsonArray choices = obj[QStringLiteral("choices")].toArray();
        if (choices.isEmpty()) {
            emit failed(QStringLiteral("the model returned no choices"));
            return;
        }
        acc_.mergeMessage(choices[0].toObject()[QStringLiteral("message")].toObject());
        if (!acc_.content.isEmpty()) emit delta(acc_.content);
        emit finished(acc_.toMessage());
    }

    void LlmClient::fetchLoadedModels(const QString& baseUrl, std::function<void(QStringList, QString)> done) {
        // the OpenAI-compatible base ends in /v1; Ollama's own API sits beside it
        QString root = baseUrl.trimmed();
        while (root.endsWith(QLatin1Char('/'))) root.chop(1);
        if (root.endsWith(QLatin1String("/v1"))) root.chop(3);
        QNetworkRequest req(QUrl(root + QStringLiteral("/api/ps")));
        req.setTransferTimeout(3000);
        QNetworkReply* reply = nam_.get(req);
        connect(reply, &QNetworkReply::finished, this, [reply, done = std::move(done)] {
            reply->deleteLater();
            if (reply->error() != QNetworkReply::NoError) {
                done({}, reply->errorString());
                return;
            }
            const QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());
            if (!doc.isObject() || !doc.object().contains(QStringLiteral("models"))) {
                done({}, QStringLiteral("not an Ollama server"));
                return;
            }
            QStringList names;
            for (const QJsonValue& v : doc.object()[QStringLiteral("models")].toArray()) {
                const QJsonObject m = v.toObject();
                for (const char* key : {"name", "model"}) {
                    const QString n = m[QLatin1String(key)].toString();
                    if (!n.isEmpty() && !names.contains(n)) names << n;
                }
            }
            done(names, {});
        });
    }

    void LlmClient::fetchModels(const QString& baseUrl, const QString& apiKey,
                                std::function<void(QStringList, QString)> done) {
        QNetworkRequest req(QUrl(joinUrl(baseUrl, QStringLiteral("/models"))));
        if (!apiKey.isEmpty()) req.setRawHeader("Authorization", ("Bearer " + apiKey).toUtf8());
        req.setTransferTimeout(5000);
        QNetworkReply* reply = nam_.get(req);
        connect(reply, &QNetworkReply::finished, this, [reply, done = std::move(done)] {
            reply->deleteLater();
            if (reply->error() != QNetworkReply::NoError) {
                done({}, reply->errorString());
                return;
            }
            const QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());
            QStringList ids;
            for (const QJsonValue& v : doc.object()[QStringLiteral("data")].toArray()) {
                const QString id = v.toObject()[QStringLiteral("id")].toString();
                if (!id.isEmpty()) ids << id;
            }
            done(ids, {});
        });
    }

} // namespace sirius::app
