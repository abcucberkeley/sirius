#include "qt/panels/assistant_panel.hpp"

#include <deque>
#include <functional>
#include <memory>

#include <QEventLoop>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QIcon>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QResizeEvent>
#include <QScrollArea>
#include <QScrollBar>
#include <QSettings>
#include <QTimer>
#include <QToolTip>
#include <QVBoxLayout>

#include "core/help_pages.hpp"
#include "core/tool_api.hpp"
#include "qt/panels/llm_client.hpp"
#include "qt/qt_strings.hpp"
#include "qt/shortcuts.hpp"
#include "qt/secret_store.hpp"
#include "qt/theme.hpp"
#include "qt/widgets/controls.hpp"

namespace sirius::app {

    // --- settings ------------------------------------------------------------------

    AssistantSettings AssistantSettings::load() {
        QSettings s;
        s.beginGroup(QStringLiteral("assistant"));
        AssistantSettings a;
        a.provider = s.value(QStringLiteral("provider"), a.provider).toString();
        a.baseUrl = s.value(QStringLiteral("baseUrl"), a.baseUrl).toString();
        a.model = s.value(QStringLiteral("model"), a.model).toString();
        a.apiKey = secrets::read(QStringLiteral("assistant/apiKey"));
        a.askBeforeActing = s.value(QStringLiteral("askBeforeActing"), a.askBeforeActing).toBool();
        if (a.apiKey.isEmpty()) {
            if (a.provider == QLatin1String("openrouter")) a.apiKey = qEnvironmentVariable("OPENROUTER_API_KEY");
            if (a.apiKey.isEmpty()) a.apiKey = qEnvironmentVariable("SIRIUS_LLM_API_KEY");
        }
        return a;
    }

    void AssistantSettings::save() const {
        QSettings s;
        s.beginGroup(QStringLiteral("assistant"));
        s.setValue(QStringLiteral("provider"), provider);
        s.setValue(QStringLiteral("baseUrl"), baseUrl);
        s.setValue(QStringLiteral("model"), model);
        secrets::write(QStringLiteral("assistant/apiKey"), apiKey);
        s.setValue(QStringLiteral("askBeforeActing"), askBeforeActing);
    }

    namespace {

        QString nlohmannToQ(const nlohmann::json& j) { return fromStd(j.dump()); }

        QJsonValue toQJson(const nlohmann::json& j) {
            const std::string dumped = j.dump();
            const QJsonDocument doc = QJsonDocument::fromJson(QByteArray(dumped.data(), static_cast<qsizetype>(dumped.size())));
            if (doc.isArray()) return doc.array();
            if (doc.isObject()) return doc.object();
            return QJsonValue();
        }

        bool mutatingTool(const QString& name) {
            static const char* readOnly[] = {"get_", "list_", "read_", "describe", "help", "explain", "context", "find_"};
            for (const char* prefix : readOnly)
                if (name.startsWith(QLatin1String(prefix))) return false;
            return true;
        }

        QLabel* mutedLabel(const QString& text, QWidget* parent, int px = theme::kSmallPx) {
            auto* l = new QLabel(text, parent);
            l->setFont(theme::font(px));
            QPalette pal = l->palette();
            pal.setColor(QPalette::WindowText, theme::kNeutral600);
            l->setPalette(pal);
            return l;
        }

        // One action card: glyph | text | link.
        class ActionCard : public QFrame {
        public:
            ActionCard(const ActionRecord& rec, std::function<void(const QString&)> onLink, QWidget* parent)
                : QFrame(parent) {
                setFrameShape(QFrame::NoFrame);
                widgets::setWidgetClass(this, "card");
                auto* g = new QGridLayout(this);
                g->setContentsMargins(10, 7, 10, 7);
                g->setHorizontalSpacing(8);
                g->setColumnMinimumWidth(0, 18);
                g->setColumnStretch(1, 1);
                widgets::Icon icon = widgets::Icon::Info;
                bool accent = false;
                switch (rec.kind) {
                    case ActionRecord::Kind::Param: icon = widgets::Icon::Pencil; break;
                    case ActionRecord::Kind::Run: icon = widgets::Icon::Play; break;
                    case ActionRecord::Kind::View:
                        icon = widgets::Icon::Eye;
                        accent = true;
                        break;
                    case ActionRecord::Kind::Edit: icon = widgets::Icon::Pencil; break;
                    case ActionRecord::Kind::Info: break;
                }
                auto* gl = new QLabel(this);
                gl->setFixedSize(14, 14);
                gl->setPixmap(widgets::iconPixmap(icon, 14, accent ? theme::kAccent : theme::kText));
                g->addWidget(gl, 0, 0);
                auto* text = new QLabel(fromStd(rec.text), this);
                text->setFont(theme::font(12));
                text->setToolTip(fromStd(rec.text));
                text->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
                g->addWidget(text, 0, 1);
                if (!rec.link.empty()) {
                    auto* link = new QLabel(QStringLiteral("<a href=\"%1\" style=\"color:%2; text-decoration:none\">%1</a>")
                                                .arg(fromStd(rec.link), theme::hex(theme::kAccent)),
                                            this);
                    link->setFont(theme::font(12));
                    link->setCursor(Qt::PointingHandCursor);
                    // Tab reaches the link and Enter follows it.
                    link->setTextInteractionFlags(Qt::LinksAccessibleByMouse | Qt::LinksAccessibleByKeyboard);
                    link->setFocusPolicy(Qt::StrongFocus);
                    link->setAccessibleName(fromStd(rec.link));
                    QObject::connect(link, &QLabel::linkActivated, this, [onLink](const QString& href) { onLink(href); });
                    g->addWidget(link, 0, 2);
                }
            }
        };

        class Chip : public QPushButton {
        public:
            Chip(const QString& text, QWidget* parent) : QPushButton(text, parent) {
                setCursor(Qt::PointingHandCursor);
                // Reachable with Tab; the style sheet draws the focus ring.
                setFocusPolicy(Qt::StrongFocus);
                setAccessibleName(text);
                // The face also has to be on the widget: the size hint is
                // measured from it, not from the style sheet's font-size.
                setFont(theme::font(theme::kSmallPx));
                widgets::setButtonClass(this, "chip");
            }
        };

    } // namespace

    // --- panel -----------------------------------------------------------------------

    struct AssistantPanel::Impl {
        AssistantPanel* self = nullptr;
        WorkbenchBridge& bridge;
        AssistantSettings settings;
        ToolApi api;
        LlmClient client;

        QLabel* context = nullptr;
        QScrollArea* scroll = nullptr;
        QWidget* transcript = nullptr;
        QVBoxLayout* messages = nullptr;
        QWidget* busyRow = nullptr;
        QLabel* busyText = nullptr;
        QLineEdit* input = nullptr;
        QPushButton* send = nullptr;
        QLabel* askToggle = nullptr;
        QLabel* streamingLabel = nullptr;      // the assistant text being streamed
        QString streamingSource;               // its Markdown so far (the label holds HTML)
        static QString renderMarkdown(const QString& markdown) {
            return fromStd(helpMarkdownToHtml(toStd(markdown), std::string()));
        }
        QWidget* streamingBlock = nullptr;

        QJsonArray history;                    // API messages after the system prompt
        struct PendingCall {
            QString id, name, arguments;
        };
        std::deque<PendingCall> pending;
        bool busy = false;

        explicit Impl(AssistantPanel* s, WorkbenchBridge& b) : self(s), bridge(b), api(b.wb()) {}

        // --- transcript --------------------------------------------------------
        void scrollToBottom() {
            QTimer::singleShot(0, self, [this] { scroll->verticalScrollBar()->setValue(scroll->verticalScrollBar()->maximum()); });
        }

        void addUserBubble(const QString& text) {
            auto* row = new QWidget(transcript);
            auto* h = new QHBoxLayout(row);
            h->setContentsMargins(0, 0, 0, 0);
            auto* bubble = new QLabel(text, row);
            bubble->setWordWrap(true);
            bubble->setFont(theme::font(theme::kBodyPx));
            bubble->setTextInteractionFlags(Qt::TextSelectableByMouse);
            widgets::setWidgetClass(bubble, "bubble");
            bubble->setMaximumWidth(static_cast<int>(scroll->viewport()->width() * 0.88));
            h->addStretch(1);
            h->addWidget(bubble, 0, Qt::AlignRight);
            messages->insertWidget(messages->count() - 1, row);
            scrollToBottom();
        }

        // A block: assistant text + cards below it.
        QWidget* addAssistantBlock(const QString& text) {
            auto* block = new QWidget(transcript);
            auto* v = new QVBoxLayout(block);
            v->setContentsMargins(0, 0, 0, 0);
            v->setSpacing(8);
            auto* label = new QLabel(block);
            label->setWordWrap(true);
            label->setFont(theme::font(theme::kBodyPx));
            // Markdown with $..$ / \(..\) math through the help renderer:
            // Qt's own Markdown has no math, and this keeps chat and help pages alike.
            label->setTextFormat(Qt::RichText);
            label->setOpenExternalLinks(true);
            label->setTextInteractionFlags(Qt::TextSelectableByMouse | Qt::LinksAccessibleByMouse);
            label->setText(renderMarkdown(text));
            label->setVisible(!text.isEmpty());
            label->setMaximumWidth(static_cast<int>(scroll->viewport()->width() * 0.94));
            label->setProperty("assistantText", true);
            v->addWidget(label);
            messages->insertWidget(messages->count() - 1, block);
            streamingBlock = block;
            streamingLabel = label;
            scrollToBottom();
            return block;
        }

        void addCards(QWidget* block, const std::vector<ActionRecord>& records) {
            if (!block) block = addAssistantBlock({});
            auto* v = static_cast<QVBoxLayout*>(block->layout());
            for (const ActionRecord& rec : records) {
                const nlohmann::json state = rec.viewState;
                auto* card = new ActionCard(rec, [this, state](const QString& link) { onCardLink(link, state); }, block);
                card->setMaximumWidth(static_cast<int>(scroll->viewport()->width() * 0.94));
                v->addWidget(card);
            }
            scrollToBottom();
        }

        void onCardLink(const QString& link, const nlohmann::json& state) {
            Workbench& wb = bridge.wb();
            if (link == QLatin1String("undo")) {
                wb.undo();
            } else if (link == QLatin1String("view")) {
                if (state.is_object()) {
                    if (state.contains("select") && state["select"].is_number_integer()) wb.select(state["select"].get<int>());
                    if (state.contains("view") && state["view"].is_number_integer()) wb.view(state["view"].get<int>());
                    wb.setViewState(ViewState::fromJson(state, wb.viewState()));
                }
            } else if (link == QLatin1String("log")) {
                const auto& log = wb.log();
                QString tail;
                const std::size_t start = log.size() > 12 ? log.size() - 12 : 0;
                for (std::size_t i = start; i < log.size(); ++i) tail += fromStd(log[i]) + "\n";
                QToolTip::showText(QCursor::pos(), tail.trimmed().isEmpty() ? QStringLiteral("(log is empty)") : tail.trimmed(), self);
            }
        }

        void setBusy(bool on, const QString& text = {}) {
            busy = on;
            busyRow->setVisible(on);
            busyText->setText(text.isEmpty() ? QStringLiteral("Thinking…") : text);
            send->setEnabled(!on);
            input->setEnabled(!on);
            if (on) scrollToBottom();
        }

        void showError(const QString& error) {
            addAssistantBlock(QStringLiteral("⚠ %1").arg(error));
            setBusy(false);
        }

        // --- conversation ------------------------------------------------------
        QJsonArray systemMessages() {
            QJsonObject sys;
            sys[QStringLiteral("role")] = QStringLiteral("system");
            QString prompt = fromStd(api.systemPrompt());
            prompt += QStringLiteral("\n\nCurrent workbench state (JSON):\n") + nlohmannToQ(api.contextSnapshot());
            sys[QStringLiteral("content")] = prompt;
            return {sys};
        }

        void submit(const QString& text) {
            const QString t = text.trimmed();
            if (t.isEmpty() || busy) return;
            input->clear();
            addUserBubble(t);
            QJsonObject m;
            m[QStringLiteral("role")] = QStringLiteral("user");
            m[QStringLiteral("content")] = t;
            history.append(m);
            ensureModelThen([this] { step(); });
        }

        void ensureModelThen(std::function<void()> next) {
            if (!settings.model.isEmpty()) {
                next();
                return;
            }
            setBusy(true, QStringLiteral("Looking up models…"));
            client.fetchModels(settings.baseUrl, settings.apiKey, [this, next](QStringList ids, QString error) {
                if (ids.isEmpty()) {
                    showError(error.isEmpty() ? QStringLiteral("No model configured and the server lists none. Set one in Preferences ▸ Assistant.")
                                              : QStringLiteral("Cannot reach the model server at %1 (%2). Configure the assistant in Preferences.")
                                                    .arg(settings.baseUrl, error));
                    return;
                }
                settings.model = ids.first();
                settings.save();
                next();
            });
        }

        void step() {
            setBusy(true);
            LlmClient::Request r;
            r.baseUrl = settings.baseUrl;
            r.model = settings.model;
            r.apiKey = settings.apiKey;
            QJsonArray msgs = systemMessages();
            for (const QJsonValue& v : history) msgs.append(v);
            r.messages = msgs;
            r.tools = toQJson(api.schemas()).toArray();
            streamingBlock = nullptr;
            streamingLabel = nullptr;
            streamingSource.clear();
            client.send(r);
        }

        void onDelta(const QString& text) {
            if (!streamingLabel) {
                addAssistantBlock({});
                streamingSource.clear();
            }
            streamingSource += text;
            streamingLabel->setText(renderMarkdown(streamingSource));
            streamingLabel->setVisible(true);
            scrollToBottom();
        }

        void onFinished(const QJsonObject& message) {
            history.append(message);
            const QString content = message[QStringLiteral("content")].toString();
            if (streamingLabel) {
                streamingLabel->setText(renderMarkdown(content));
                streamingLabel->setVisible(!content.isEmpty());
            } else if (!content.isEmpty()) {
                addAssistantBlock(content);
            }
            pending.clear();
            for (const QJsonValue& v : message[QStringLiteral("tool_calls")].toArray()) {
                const QJsonObject call = v.toObject();
                const QJsonObject fn = call[QStringLiteral("function")].toObject();
                pending.push_back({call[QStringLiteral("id")].toString(), fn[QStringLiteral("name")].toString(),
                                   fn[QStringLiteral("arguments")].toString()});
            }
            if (pending.empty()) {
                setBusy(false);
                return;
            }
            if (!streamingBlock) addAssistantBlock({});
            processNextCall();
        }

        void processNextCall() {
            if (pending.empty()) {
                step();   // let the model see the results
                return;
            }
            const PendingCall call = pending.front();
            if (settings.askBeforeActing && mutatingTool(call.name)) {
                askConfirmation(call);
                return;
            }
            pending.pop_front();
            executeCall(call);
            processNextCall();
        }

        void askConfirmation(const PendingCall& call) {
            setBusy(true, QStringLiteral("Waiting for your confirmation…"));
            auto* row = new QFrame(transcript);
            row->setFrameShape(QFrame::NoFrame);
            widgets::setWidgetClass(row, "card-accent");
            auto* h = new QHBoxLayout(row);
            h->setContentsMargins(10, 7, 10, 7);
            h->setSpacing(8);
            auto* text = new QLabel(QStringLiteral("Apply %1 %2?").arg(call.name, call.arguments.left(80)), row);
            text->setFont(theme::font(12));
            text->setWordWrap(true);
            h->addWidget(text, 1);
            auto* apply = new Chip(QStringLiteral("Apply"), row);
            auto* skip = new Chip(QStringLiteral("Skip"), row);
            h->addWidget(apply);
            h->addWidget(skip);
            static_cast<QVBoxLayout*>(streamingBlock->layout())->addWidget(row);
            scrollToBottom();
            auto finish = [this, row, call](bool doIt) {
                row->deleteLater();
                if (!pending.empty()) pending.pop_front();
                if (doIt) {
                    executeCall(call);
                } else {
                    QJsonObject tool;
                    tool[QStringLiteral("role")] = QStringLiteral("tool");
                    tool[QStringLiteral("tool_call_id")] = call.id;
                    tool[QStringLiteral("content")] = QStringLiteral("{\"error\":\"the user declined this action\"}");
                    history.append(tool);
                }
                setBusy(true);
                processNextCall();
            };
            QObject::connect(apply, &QPushButton::clicked, self, [finish] { finish(true); });
            QObject::connect(skip, &QPushButton::clicked, self, [finish] { finish(false); });
        }

        void executeCall(const PendingCall& call) {
            setBusy(true, QStringLiteral("Running %1…").arg(call.name));
            nlohmann::json args = nlohmann::json::parse(toStd(call.arguments), nullptr, false);
            if (args.is_discarded() || !args.is_object()) args = nlohmann::json::object();
            nlohmann::json result;
            try {
                result = api.call(toStd(call.name), args);
            } catch (const std::exception& e) {
                result = {{"error", e.what()}};
            }
            const std::vector<ActionRecord> actions = api.takeActions();
            if (!actions.empty()) addCards(streamingBlock, actions);
            QJsonObject tool;
            tool[QStringLiteral("role")] = QStringLiteral("tool");
            tool[QStringLiteral("tool_call_id")] = call.id;
            tool[QStringLiteral("name")] = call.name;
            QString content = nlohmannToQ(result);
            if (content.size() > 12000) content = content.left(12000) + QStringLiteral("…(truncated)");
            tool[QStringLiteral("content")] = content;
            history.append(tool);
        }

        void updateContextLine() {
            const Workbench& wb = bridge.wb();
            const int sel = wb.selectedIndex();
            // With no step there is nothing to see: the line said "sees step
            // 01" over an empty pipeline.
            const int steps = wb.pipeline().size();
            if (sel < 0 || sel >= steps)
                context->setText(steps == 0 ? QStringLiteral("no steps yet") : QStringLiteral("sees the ops stack"));
            else
                context->setText(QStringLiteral("sees step %1, diagnostics, ops stack").arg(fromStd(Step::number(sel))));
        }

        void updateAskToggle() {
            askToggle->setText(QStringLiteral("<a href=\"toggle\" style=\"color:%1; text-decoration:none\">%2</a>")
                                   .arg(theme::hex(theme::kAccent), settings.askBeforeActing ? QStringLiteral("Ask before acting ✓")
                                                                                             : QStringLiteral("Ask before acting")));
        }
    };

    AssistantPanel::AssistantPanel(WorkbenchBridge& bridge, QWidget* parent)
        : QWidget(parent), impl_(std::make_unique<Impl>(this, bridge)) {
        Impl& d = *impl_;
        d.settings = AssistantSettings::load();
        setAutoFillBackground(true);
        QPalette pal = palette();
        pal.setColor(QPalette::Window, theme::kBg);
        setPalette(pal);
        setMinimumWidth(theme::kAssistantW);

        auto* v = new QVBoxLayout(this);
        v->setContentsMargins(0, 0, 0, 0);
        v->setSpacing(0);

        // header
        auto* header = new QWidget(this);
        header->setFixedHeight(40);
        auto* h = new QHBoxLayout(header);
        h->setContentsMargins(14, 0, 14, 0);
        h->setSpacing(10);
        auto* star = new QLabel(header);
        star->setFixedSize(14, 14);
        star->setPixmap(widgets::iconPixmap(widgets::Icon::Sparkle, 14, theme::kAccent));
        h->addWidget(star);
        auto* title = new QLabel(QStringLiteral("Assistant"), header);
        title->setFont(theme::heading(13));
        h->addWidget(title);
        d.context = mutedLabel({}, header);
        d.context->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
        h->addWidget(d.context, 1);
        auto* close = new widgets::GlyphButton(widgets::Icon::Close, 18, header);
        close->setBorderless(true);
        close->setIconPx(11);
        close->setToolTip(QStringLiteral("Close the assistant"));
        connect(close, &QAbstractButton::clicked, this, [this] { emit closeRequested(); });
        h->addWidget(close);
        v->addWidget(header);
        auto* rule = new QFrame(this);
        rule->setFixedHeight(theme::kRule);
        rule->setAutoFillBackground(true);
        QPalette rp = rule->palette();
        rp.setColor(QPalette::Window, theme::kDivider);
        rule->setPalette(rp);
        v->addWidget(rule);

        // transcript
        d.scroll = new QScrollArea(this);
        d.scroll->setWidgetResizable(true);
        d.scroll->setFrameShape(QFrame::NoFrame);
        d.scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        d.transcript = new QWidget(d.scroll);
        d.transcript->setAutoFillBackground(true);
        d.transcript->setPalette(pal);
        d.messages = new QVBoxLayout(d.transcript);
        d.messages->setContentsMargins(14, 14, 14, 14);
        d.messages->setSpacing(14);
        d.busyRow = new QWidget(d.transcript);
        auto* bh = new QHBoxLayout(d.busyRow);
        bh->setContentsMargins(0, 0, 0, 0);
        bh->setSpacing(8);
        auto* square = new QWidget(d.busyRow);
        square->setFixedSize(8, 8);
        square->setAutoFillBackground(true);
        QPalette qp = square->palette();
        qp.setColor(QPalette::Window, theme::kAccent);
        square->setPalette(qp);
        bh->addWidget(square);
        d.busyText = mutedLabel(QStringLiteral("Thinking…"), d.busyRow, 12);
        bh->addWidget(d.busyText, 1);
        d.busyRow->hide();
        d.messages->addWidget(d.busyRow);
        d.messages->addStretch(1);
        d.scroll->setWidget(d.transcript);
        v->addWidget(d.scroll, 1);

        // footer
        auto* footerRule = new QFrame(this);
        footerRule->setFixedHeight(theme::kRule);
        footerRule->setAutoFillBackground(true);
        footerRule->setPalette(rp);
        v->addWidget(footerRule);
        auto* footer = new QWidget(this);
        auto* f = new QVBoxLayout(footer);
        f->setContentsMargins(14, 12, 14, 12);
        f->setSpacing(10);
        auto* chips = new QWidget(footer);
        auto* chipsLayout = new QHBoxLayout(chips);
        chipsLayout->setContentsMargins(0, 0, 0, 0);
        chipsLayout->setSpacing(6);
        auto* chipsGrid = new QGridLayout;
        chipsGrid->setContentsMargins(0, 0, 0, 0);
        chipsGrid->setSpacing(6);
        const QStringList suggestions{QStringLiteral("Explain the Wiener parameter"), QStringLiteral("Why is step 06 skipped?"),
                                      QStringLiteral("Max-project over Z and show me"), QStringLiteral("Flag low-confidence labels")};
        for (int i = 0; i < suggestions.size(); ++i) {
            auto* chip = new Chip(suggestions[i], chips);
            connect(chip, &QPushButton::clicked, this, [this, text = suggestions[i]] {
                impl_->input->setText(text);
                impl_->input->setFocus();
            });
            chipsGrid->addWidget(chip, i / 2, i % 2, Qt::AlignLeft);
        }
        chipsLayout->addLayout(chipsGrid);
        chipsLayout->addStretch(1);
        f->addWidget(chips);
        auto* inputRow = new QHBoxLayout;
        inputRow->setSpacing(6);
        d.input = new QLineEdit(footer);
        d.input->setPlaceholderText(QStringLiteral("Ask, or tell it what to do…"));
        d.input->setMinimumHeight(34);
        d.input->setFont(theme::font(theme::kBodyPx));
        connect(d.input, &QLineEdit::returnPressed, this, [this] { impl_->submit(impl_->input->text()); });
        inputRow->addWidget(d.input, 1);
        d.send = new QPushButton(footer);
        d.send->setIcon(QIcon(widgets::iconPixmap(widgets::Icon::Enter, 14, theme::kBg)));
        d.send->setToolTip(withShortcut(QStringLiteral("Send"), keys::send()));
        d.send->setAccessibleName(QStringLiteral("Send"));
        d.send->setFixedSize(34, 34);
        d.send->setCursor(Qt::PointingHandCursor);
        widgets::setButtonClass(d.send, "primary icon");
        connect(d.send, &QPushButton::clicked, this, [this] { impl_->submit(impl_->input->text()); });
        inputRow->addWidget(d.send);
        f->addLayout(inputRow);
        auto* noteRow = new QHBoxLayout;
        noteRow->addWidget(mutedLabel(QStringLiteral("Changes are applied as undoable steps"), footer));
        noteRow->addStretch(1);
        d.askToggle = new QLabel(footer);
        d.askToggle->setFont(theme::font(theme::kSmallPx));
        d.askToggle->setCursor(Qt::PointingHandCursor);
        connect(d.askToggle, &QLabel::linkActivated, this, [this](const QString&) {
            impl_->settings.askBeforeActing = !impl_->settings.askBeforeActing;
            impl_->settings.save();
            impl_->updateAskToggle();
        });
        noteRow->addWidget(d.askToggle);
        f->addLayout(noteRow);
        v->addWidget(footer);

        // model client
        connect(&d.client, &LlmClient::delta, this, [this](const QString& t) { impl_->onDelta(t); });
        connect(&d.client, &LlmClient::finished, this, [this](const QJsonObject& m) { impl_->onFinished(m); });
        connect(&d.client, &LlmClient::failed, this, [this](const QString& e) {
            impl_->pending.clear();
            impl_->showError(QStringLiteral("The model server answered: %1 (provider %2, %3)")
                                 .arg(e, impl_->settings.provider, impl_->settings.baseUrl));
        });

        // runs requested by tools block the tool loop, not the GUI
        d.api.setRunHook([this](int target) -> nlohmann::json {
            WorkbenchBridge& b = impl_->bridge;
            if (!b.startRun(target)) return {{"ok", false}, {"error", "the run could not start (see the log)"}};
            QEventLoop loop;
            bool ok = false;
            QString error;
            const QMetaObject::Connection c = connect(&b, &WorkbenchBridge::runFinished, &loop, [&](bool k, const QString& e) {
                ok = k;
                error = e;
                loop.quit();
            });
            impl_->setBusy(true, QStringLiteral("Running the pipeline…"));
            loop.exec();
            disconnect(c);
            nlohmann::json out = {{"ok", ok}};
            if (!error.isEmpty()) out["error"] = toStd(error);
            if (auto job = b.wb().activeRun(); job) out["seconds"] = job->seconds();
            return out;
        });
        d.api.setHelpHook([](const std::string& kind) { return loadHelpPage(kind).markdown; });

        connect(&bridge, &WorkbenchBridge::selectionChanged, this, [this] { impl_->updateContextLine(); });
        connect(&bridge, &WorkbenchBridge::pipelineChanged, this, [this] { impl_->updateContextLine(); });
        d.updateContextLine();
        d.updateAskToggle();
        d.addAssistantBlock(QStringLiteral("Ask about a step, or tell me what to do: I can edit parameters, run steps and change the view. "
                                           "Every change lands in the undo stack."));
    }

    AssistantPanel::~AssistantPanel() = default;

    void AssistantPanel::setSettings(const AssistantSettings& s) {
        impl_->settings = s;
        impl_->settings.save();
        impl_->updateAskToggle();
    }

    AssistantSettings AssistantPanel::settings() const { return impl_->settings; }

    void AssistantPanel::ask(const QString& text) { impl_->submit(text); }

    void AssistantPanel::focusInput() { impl_->input->setFocus(); }

    void AssistantPanel::resizeEvent(QResizeEvent* event) {
        QWidget::resizeEvent(event);
        const int w = impl_->scroll->viewport()->width();
        for (QLabel* l : impl_->transcript->findChildren<QLabel*>()) {
            if (l->property("class").toString() == QLatin1String("bubble")) l->setMaximumWidth(static_cast<int>(w * 0.88));
            else if (l->property("assistantText").toBool()) l->setMaximumWidth(static_cast<int>(w * 0.94));
        }
    }

} // namespace sirius::app
