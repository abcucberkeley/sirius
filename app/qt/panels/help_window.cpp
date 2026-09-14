#include "qt/panels/help_window.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

#include <QDesktopServices>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileInfo>
#include <QFileSystemWatcher>
#include <QHBoxLayout>
#include <QLabel>
#include <QMimeData>
#include <QMouseEvent>
#include <QPainter>
#include <QTextBrowser>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

#include "core/help_pages.hpp"
#include "qt/qt_strings.hpp"
#include "qt/theme.hpp"
#include "qt/widgets/controls.hpp"

namespace sirius::app {

    namespace fs = std::filesystem;

    namespace {

        QString esc(const std::string& s) { return fromStd(s).toHtmlEscaped(); }

        // Markdown of the page after the parts the structured layout renders
        // itself (front matter, intro, first display formula), split into the
        // sections before and after "## Parameters" (which, like "## Note",
        // is rendered from the parsed page), so further sections keep their
        // place.
        struct Remainder {
            std::string before, after;
        };
        Remainder remainderMarkdown(const HelpPage& page) {
            std::istringstream in(page.markdown);
            std::string line;
            Remainder out;
            bool inFront = false, frontDone = false, introDone = page.intro.empty(), texDone = page.tex.empty();
            bool inTex = false, inIntro = false, afterParams = false, skipping = false;
            std::size_t lineNo = 0;
            while (std::getline(in, line)) {
                std::string trimmed;
                {
                    const auto a = line.find_first_not_of(" \t\r");
                    trimmed = a == std::string::npos ? std::string() : line.substr(a, line.find_last_not_of(" \t\r") - a + 1);
                }
                ++lineNo;
                if (!frontDone) {
                    if (lineNo == 1 && trimmed == "---") {
                        inFront = true;
                        continue;
                    }
                    if (inFront) {
                        if (trimmed == "---") {
                            inFront = false;
                            frontDone = true;
                        }
                        continue;
                    }
                    frontDone = true;
                }
                if (inTex) {
                    if (trimmed.find("$$") != std::string::npos) {
                        inTex = false;
                        texDone = true;
                    }
                    continue;
                }
                if (!texDone && trimmed.rfind("$$", 0) == 0) {
                    if (trimmed.find("$$", 2) != std::string::npos) texDone = true;
                    else inTex = true;
                    continue;
                }
                if (inIntro) {
                    if (trimmed.empty()) {
                        inIntro = false;
                        introDone = true;
                    }
                    continue;
                }
                if (!introDone && !trimmed.empty() && trimmed[0] != '#' && trimmed[0] != '|') {
                    inIntro = true;
                    continue;
                }
                if (!trimmed.empty() && trimmed[0] == '#') {
                    std::size_t level = 0;
                    while (level < trimmed.size() && trimmed[level] == '#') ++level;
                    std::string name = trimmed.substr(level);
                    const auto b = name.find_first_not_of(" \t");
                    name = b == std::string::npos ? std::string() : name.substr(b);
                    for (char& c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                    skipping = name == "parameters" || name == "note";
                    if (name == "parameters") afterParams = true;
                }
                if (skipping) continue;
                (afterParams ? out.after : out.before) += line + "\n";
            }
            return out;
        }

        // The whole page as rich text for QTextBrowser.
        QString pageHtml(const HelpPage& page, bool hasFigure) {
            const QString text = theme::hex(theme::kText), n600 = theme::hex(theme::kNeutral600),
                          n800 = theme::hex(theme::kNeutral800), accent = theme::hex(theme::kAccent),
                          divider = theme::hex(theme::kDivider), surface = theme::hex(theme::kSurface),
                          n400 = theme::hex(theme::kNeutral400);
            const std::string baseDir = page.path.empty() ? std::string() : fs::path(page.path).parent_path().string();
            QString html;
            html += QStringLiteral("<body style=\"color:%1; font-size:13px\">").arg(text);
            html += QStringLiteral("<h3 style=\"font-size:24px; font-weight:800; margin:0 0 10px 0\">%1</h3>").arg(esc(page.title));
            if (!page.intro.empty())
                html += QStringLiteral("<p style=\"margin:0 0 14px 0\">%1</p>")
                            .arg(fromStd(helpMarkdownToHtml(page.intro, baseDir)));
            if (!page.tex.empty())
                html += QStringLiteral("<table width=\"100%\" cellspacing=\"0\" cellpadding=\"12\" style=\"background:%1; border:1px solid %2; margin-bottom:14px\">"
                                       "<tr><td>%3</td></tr></table>")
                            .arg(surface, divider, fromStd(latexToHtml(page.tex, true)));
            // figure slot
            if (hasFigure) {
                html += QStringLiteral("<table width=\"100%\" cellspacing=\"0\" cellpadding=\"6\" style=\"margin-bottom:14px\"><tr><td align=\"center\">"
                                       "<img src=\"%1\"><br><span style=\"color:%2; font-size:11px\">%3</span></td></tr></table>")
                            .arg(QUrl::fromLocalFile(fromStd(page.figurePath)).toString(QUrl::FullyEncoded),
                                 n600, esc(page.figure));
            } else {
                html += QStringLiteral("<table width=\"100%\" border=\"1\" cellspacing=\"0\" cellpadding=\"10\" bordercolor=\"%1\" style=\"border-style:dashed; border-color:%1; border-collapse:collapse; margin-bottom:14px\">"
                                       "<tr><td align=\"center\" height=\"150\" style=\"color:%2; font-size:12px\">"
                                       "<b style=\"font-weight:800\">Figure</b><br>%3<br><span style=\"color:%4\">Drop image · PNG, SVG, PDF</span></td></tr></table>")
                            .arg(n400, n600, esc(page.figure), accent);
            }
            // sections that precede the parameter table
            {
                const Remainder rest = remainderMarkdown(page);
                if (rest.before.find_first_not_of(" \t\r\n") != std::string::npos)
                    html += QStringLiteral("<div style=\"margin-bottom:12px\">%1</div>")
                                .arg(fromStd(helpMarkdownToHtml(rest.before, baseDir)));
            }
            // parameters
            if (!page.params.empty()) {
                html += QStringLiteral("<p style=\"font-size:10px; letter-spacing:1px; color:%1; margin:0; padding-bottom:6px; border-bottom:2px solid %2\">PARAMETERS</p>")
                            .arg(n600, divider);
                html += QStringLiteral("<table width=\"100%\" cellspacing=\"0\" cellpadding=\"0\">");
                for (const HelpParam& p : page.params) {
                    html += QStringLiteral("<tr><td width=\"120\" valign=\"top\" style=\"padding:10px 12px 10px 0; border-bottom:1px solid %1\">"
                                           "<b style=\"font-size:12px; font-weight:800\">%2</b><br><span style=\"color:%3; font-size:11px\">%4</span></td>"
                                           "<td valign=\"top\" style=\"padding:10px 0; border-bottom:1px solid %1\">%5%6</td></tr>")
                                .arg(divider, esc(p.name), n600, esc(p.range),
                                     fromStd(helpMarkdownToHtml(p.body, baseDir)),
                                     p.tex.empty() ? QString()
                                                   : QStringLiteral("<p style=\"margin:4px 0 0 0; color:%1\">%2</p>")
                                                         .arg(n800, fromStd(latexToHtml(p.tex, p.tex.find("\\frac") != std::string::npos))));
                }
                html += QStringLiteral("</table>");
            }
            // any further sections (manual pages) in their source order
            {
                const Remainder rest = remainderMarkdown(page);
                if (rest.after.find_first_not_of(" \t\r\n") != std::string::npos)
                    html += QStringLiteral("<div style=\"margin-top:12px\">%1</div>")
                                .arg(fromStd(helpMarkdownToHtml(rest.after, baseDir)));
            }
            if (!page.note.empty())
                html += QStringLiteral("<p style=\"font-size:11px; color:%1; margin-top:14px\">%2</p>")
                            .arg(n600, fromStd(helpMarkdownToHtml(page.note, baseDir)));
            html += QStringLiteral("</body>");
            return html;
        }

    } // namespace

    // --- HelpView ---------------------------------------------------------------

    HelpView::HelpView(QWidget* parent) : QTextBrowser(parent) {
        setOpenExternalLinks(true);
        setFrameShape(QFrame::NoFrame);
        setFont(theme::font(theme::kBodyPx));
        document()->setDocumentMargin(22);
        QPalette pal = palette();
        pal.setColor(QPalette::Base, theme::kBg);
        pal.setColor(QPalette::Text, theme::kText);
        setPalette(pal);
    }

    void HelpView::setPage(const HelpPage& page) {
        page_ = page;
        const bool hasFigure = !page.figurePath.empty() && fs::exists(page.figurePath);
        if (!page.path.empty()) setSearchPaths({fromStd(fs::path(page.path).parent_path().string())});
        setHtml(pageHtml(page, hasFigure));
    }

    // --- HelpWindow ---------------------------------------------------------------

    struct HelpWindow::Impl {
        WorkbenchBridge& bridge;
        std::string kind;
        HelpPage page;
        QWidget* header = nullptr;
        QLabel* caption = nullptr;
        QLabel* edit = nullptr;
        widgets::GlyphButton* close = nullptr;
        HelpView* view = nullptr;
        QFileSystemWatcher watcher;
        QTimer reloadTimer;
        QPoint dragOffset;
        bool dragging = false;
        bool followSelection = true;

        explicit Impl(WorkbenchBridge& b) : bridge(b) {}

        void load(const std::string& k) {
            kind = k;
            page = loadHelpPage(k);
            caption->setText(QStringLiteral("HELP · %1").arg(captionCase(fromStd(page.title))));
            view->setPage(page);
            const QStringList watched = watcher.files();
            if (!watched.isEmpty()) watcher.removePaths(watched);
            if (!page.path.empty() && fs::exists(page.path)) watcher.addPath(fromStd(page.path));
        }

        std::string selectedKind() const {
            const Workbench& wb = bridge.wb();
            const int sel = wb.selectedIndex();
            if (sel < 0 || sel >= wb.pipeline().size()) return "manual";
            return wb.pipeline().at(sel).kind;
        }
    };

    HelpWindow::HelpWindow(WorkbenchBridge& bridge, QWidget* parent)
        : QWidget(parent, Qt::Tool | Qt::CustomizeWindowHint | Qt::FramelessWindowHint), impl_(std::make_unique<Impl>(bridge)) {
        Impl& d = *impl_;
        setWindowTitle(QStringLiteral("Help"));
        setAcceptDrops(true);
        setAutoFillBackground(true);
        QPalette pal = palette();
        pal.setColor(QPalette::Window, theme::kBg);
        setPalette(pal);
        resize(520, 700);
        setMinimumWidth(360);
        setMaximumHeight(760);

        auto* v = new QVBoxLayout(this);
        v->setContentsMargins(2, 2, 2, 2);   // the ink border painted in paintEvent
        v->setSpacing(0);
        d.header = new QWidget(this);
        d.header->setFixedHeight(36);
        d.header->setCursor(Qt::SizeAllCursor);
        auto* h = new QHBoxLayout(d.header);
        h->setContentsMargins(14, 0, 14, 0);
        h->setSpacing(12);
        d.caption = new QLabel(QStringLiteral("HELP"), d.header);
        d.caption->setFont(theme::caption());
        QPalette cp = d.caption->palette();
        cp.setColor(QPalette::WindowText, theme::kNeutral600);
        d.caption->setPalette(cp);
        h->addWidget(d.caption);
        h->addStretch(1);
        d.edit = new QLabel(QStringLiteral("<a href=\"edit\" style=\"color:%1; text-decoration:none\">Edit page</a>").arg(theme::hex(theme::kAccent)), d.header);
        d.edit->setFont(theme::font(theme::kSmallPx));
        d.edit->setCursor(Qt::PointingHandCursor);
        d.edit->setToolTip(QStringLiteral("Open the Markdown page in your editor; the window reloads when the file changes"));
        connect(d.edit, &QLabel::linkActivated, this, [this](const QString&) {
            Impl& d = *impl_;
            if (d.page.path.empty()) return;
            if (!fs::exists(d.page.path)) {
                fs::create_directories(fs::path(d.page.path).parent_path());
                std::ofstream(d.page.path) << d.page.markdown;
                d.watcher.addPath(fromStd(d.page.path));
            }
            QDesktopServices::openUrl(QUrl::fromLocalFile(fromStd(d.page.path)));
        });
        h->addWidget(d.edit);
        d.close = new widgets::GlyphButton(widgets::Icon::Close, 18, d.header);
        d.close->setBorderless(true);
        d.close->setIconPx(11);
        d.close->setToolTip(QStringLiteral("Close"));
        connect(d.close, &QAbstractButton::clicked, this, [this] { hide(); });
        h->addWidget(d.close);
        v->addWidget(d.header, 0);
        auto* rule = new QWidget(this);
        rule->setFixedHeight(theme::kRule);
        rule->setAutoFillBackground(true);
        QPalette rp = rule->palette();
        rp.setColor(QPalette::Window, theme::kDivider);
        rule->setPalette(rp);
        v->addWidget(rule, 0);
        d.view = new HelpView(this);
        v->addWidget(d.view, 1);

        d.reloadTimer.setSingleShot(true);
        d.reloadTimer.setInterval(200);
        connect(&d.reloadTimer, &QTimer::timeout, this, [this] { impl_->load(impl_->kind); });
        connect(&d.watcher, &QFileSystemWatcher::fileChanged, this, [this](const QString&) { impl_->reloadTimer.start(); });
        connect(&bridge, &WorkbenchBridge::selectionChanged, this, [this] {
            if (isVisible() && impl_->followSelection) impl_->load(impl_->selectedKind());
        });
        d.header->installEventFilter(this);
        d.load("manual");
    }

    HelpWindow::~HelpWindow() = default;

    void HelpWindow::showKind(const std::string& kind) {
        impl_->followSelection = true;
        impl_->load(kind);
        show();
        raise();
    }

    void HelpWindow::showManual() {
        impl_->followSelection = false;
        impl_->load("manual");
        show();
        raise();
    }

    void HelpWindow::showShortcuts() {
        impl_->followSelection = false;
        impl_->load("shortcuts");
        show();
        raise();
    }

    std::string HelpWindow::currentKind() const { return impl_->kind; }

    void HelpWindow::showEvent(QShowEvent* event) {
        QWidget::showEvent(event);
        emit visibilityChanged(true);
    }

    void HelpWindow::hideEvent(QHideEvent* event) {
        QWidget::hideEvent(event);
        emit visibilityChanged(false);
    }

    bool HelpWindow::eventFilter(QObject* watched, QEvent* event) {
        Impl& d = *impl_;
        if (watched == d.header) {
            if (event->type() == QEvent::MouseButtonPress) {
                auto* me = static_cast<QMouseEvent*>(event);
                if (me->button() == Qt::LeftButton) {
                    d.dragging = true;
                    d.dragOffset = me->globalPosition().toPoint() - frameGeometry().topLeft();
                    return true;
                }
            } else if (event->type() == QEvent::MouseMove && d.dragging) {
                auto* me = static_cast<QMouseEvent*>(event);
                move(me->globalPosition().toPoint() - d.dragOffset);
                return true;
            } else if (event->type() == QEvent::MouseButtonRelease) {
                d.dragging = false;
            }
        }
        return QWidget::eventFilter(watched, event);
    }

    void HelpWindow::paintEvent(QPaintEvent* event) {
        QWidget::paintEvent(event);
        QPainter p(this);
        p.setPen(QPen(theme::kText, 2));
        p.drawRect(rect().adjusted(1, 1, -1, -1));
    }

    void HelpWindow::dragEnterEvent(QDragEnterEvent* event) {
        if (!event->mimeData()->hasUrls()) return;
        for (const QUrl& url : event->mimeData()->urls()) {
            const QString ext = QFileInfo(url.toLocalFile()).suffix().toLower();
            if (ext == QLatin1String("png") || ext == QLatin1String("svg") || ext == QLatin1String("jpg") ||
                ext == QLatin1String("jpeg") || ext == QLatin1String("pdf")) {
                event->acceptProposedAction();
                return;
            }
        }
    }

    void HelpWindow::dropEvent(QDropEvent* event) {
        Impl& d = *impl_;
        if (!event->mimeData()->hasUrls() || d.page.path.empty()) return;
        for (const QUrl& url : event->mimeData()->urls()) {
            const QString local = url.toLocalFile();
            const QString ext = QFileInfo(local).suffix().toLower();
            if (ext != QLatin1String("png") && ext != QLatin1String("svg") && ext != QLatin1String("jpg") &&
                ext != QLatin1String("jpeg") && ext != QLatin1String("pdf"))
                continue;
            // copy next to the page as <kind>-figure.<ext> and reference it from the front matter
            const fs::path pagePath(d.page.path);
            const fs::path target = pagePath.parent_path() / (d.kind + "-figure." + toStd(ext));
            std::error_code ec;
            fs::create_directories(pagePath.parent_path(), ec);
            fs::copy_file(toStd(local), target, fs::copy_options::overwrite_existing, ec);
            if (ec) {
                d.bridge.wb().logLine("Help: could not copy the figure: " + ec.message());
                return;
            }
            std::string md = d.page.markdown;
            const std::string key = "figure_path:";
            const std::size_t at = md.find(key);
            const std::string value = target.filename().string();
            if (at != std::string::npos) {
                const std::size_t eol = md.find('\n', at);
                md.replace(at, (eol == std::string::npos ? md.size() : eol) - at, key + " " + value);
            } else if (md.rfind("---\n", 0) == 0) {
                md.insert(4, key + " " + value + "\n");
            } else {
                md = "---\n" + key + " " + value + "\n---\n\n" + md;
            }
            std::ofstream(pagePath) << md;
            d.load(d.kind);
            event->acceptProposedAction();
            return;
        }
    }

} // namespace sirius::app
