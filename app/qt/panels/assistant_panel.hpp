#ifndef SIRIUS_APP_ASSISTANT_PANEL_HPP
#define SIRIUS_APP_ASSISTANT_PANEL_HPP

// Assistant dock (330 px): transcript with user bubbles, assistant text and
// action cards, busy indicator, suggestion chips, input and the
// "Ask before acting" toggle. Talks to an OpenAI-compatible chat endpoint
// (Ollama, OpenRouter) with the ToolApi's tools; every tool call is applied
// through the workbench and shown as a card.

#include <QWidget>

#include "qt/workbench_bridge.hpp"

namespace sirius::app {

    struct AssistantSettings {
        QString provider = QStringLiteral("ollama");     // "ollama" | "openrouter" | "custom"
        QString baseUrl = QStringLiteral("http://localhost:11434/v1");
        QString model;
        QString apiKey;                                  // OpenRouter / custom
        bool askBeforeActing = false;
        static AssistantSettings load();                 // QSettings
        void save() const;
    };

    class AssistantPanel : public QWidget {
        Q_OBJECT
    public:
        explicit AssistantPanel(WorkbenchBridge& bridge, QWidget* parent = nullptr);
        ~AssistantPanel() override;

        void setSettings(const AssistantSettings& s);
        AssistantSettings settings() const;
        void focusInput();
        // Submit a message as if typed (scripting, tests).
        void ask(const QString& text);

    signals:
        void closeRequested();

    protected:
        void resizeEvent(QResizeEvent* event) override;
        bool eventFilter(QObject* watched, QEvent* event) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_ASSISTANT_PANEL_HPP
