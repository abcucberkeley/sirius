#include "qt/theme.hpp"

#include <QtGlobal>
#include <QApplication>
#include <QEvent>
#include <QFocusEvent>
#include <QFontDatabase>
#include <QLineEdit>
#include <QPalette>
#include <QPlainTextEdit>
#include <QStringList>
#include <QStyle>
#include <QTextEdit>

namespace sirius::app::theme {

    namespace {
        bool fontsLoaded = false;

        void loadFonts() {
            if (fontsLoaded) return;
            fontsLoaded = true;
            for (const char* face : {"Archivo-Regular", "Archivo-SemiBold", "Archivo-Bold", "Archivo-ExtraBold"})
                QFontDatabase::addApplicationFont(QStringLiteral(":/fonts/fonts/%1.ttf").arg(QLatin1String(face)));
        }

        QString family() {
            loadFonts();
            if (QFontDatabase::hasFamily(kFontFamily)) return kFontFamily;
            return QStringLiteral("sans-serif");
        }

        // Archivo is Latin only, and operation labels carry Greek ("Emission
        // λ") and maths. Qt falls through this list per character, so the
        // rest of the text still comes out in Archivo.
        QStringList families() {
            QStringList list{family()};
            for (const QString& f : {QStringLiteral("Segoe UI"), QStringLiteral("Noto Sans"),
                                     QStringLiteral("DejaVu Sans"), QStringLiteral("sans-serif")})
                if (!list.contains(f)) list << f;
            return list;
        }
    } // namespace

    QFont font(int px, int weight) {
        QFont f(family());
        f.setFamilies(families());
        f.setPixelSize(px);
        f.setWeight(static_cast<QFont::Weight>(weight));
        f.setStyleStrategy(QFont::PreferAntialias);
        return f;
    }

    QFont heading(int px) { return font(px, QFont::ExtraBold); }

    QFont caption() {
        QFont f = font(kCaptionPx);
        f.setCapitalization(QFont::AllUppercase);
        f.setLetterSpacing(QFont::PercentageSpacing, 110.0);
        return f;
    }

    QFont mono(int px) {
        QFont f(QStringLiteral("monospace"));
        f.setStyleHint(QFont::Monospace);
        f.setPixelSize(px);
        return f;
    }

    QFont tabular(QFont f) {
        // Fixed-advance figures, so columns of numbers line up. QFont's font
        // feature API is Qt 6.7 and newer; on an older Qt (the Linux CI image
        // ships 6.4) the text keeps the font's default figures rather than
        // failing to build.
#if QT_VERSION >= QT_VERSION_CHECK(6, 7, 0)
        f.setFeature(QFont::Tag("tnum"), 1);
#endif
        return f;
    }

    QString hex(const QColor& c) { return c.name(QColor::HexRgb); }

    namespace {
        // docs/design/README.md wants the 2 px accent ring on keyboard focus
        // only; QSS ":focus" also fires on a mouse click. This stamps
        // "focusVisible" for the keyboard reasons (and for text entry, which
        // needs to show where typing lands however focus arrived) and the
        // stylesheet keys on the property instead.
        class FocusVisibleFilter : public QObject {
        public:
            using QObject::QObject;

        protected:
            bool eventFilter(QObject* target, QEvent* e) override {
                if (e->type() != QEvent::FocusIn && e->type() != QEvent::FocusOut) return false;
                auto* w = qobject_cast<QWidget*>(target);
                if (!w) return false;
                bool visible = false;
                if (e->type() == QEvent::FocusIn) {
                    switch (static_cast<QFocusEvent*>(e)->reason()) {
                        case Qt::TabFocusReason:
                        case Qt::BacktabFocusReason:
                        case Qt::ShortcutFocusReason: visible = true; break;
                        default:
                            visible = qobject_cast<QLineEdit*>(w) || qobject_cast<QPlainTextEdit*>(w) ||
                                      qobject_cast<QTextEdit*>(w) || w->inherits("QAbstractSpinBox");
                            break;
                    }
                }
                if (w->property("focusVisible").toBool() == visible) return false;
                w->setProperty("focusVisible", visible);
                w->style()->unpolish(w);
                w->style()->polish(w);
                return false;
            }
        };
    } // namespace

    void installFocusVisibleFilter(QApplication& app) {
        static FocusVisibleFilter* filter = nullptr;
        if (filter) return;
        filter = new FocusVisibleFilter(&app);
        app.installEventFilter(filter);
    }

    QString styleSheet() {
        const QString bg = hex(kBg), surface = hex(kSurface), text = hex(kText), divider = hex(kDivider),
                      accent = hex(kAccent), accent600 = hex(kAccent600), n200 = hex(kNeutral200),
                      n300 = hex(kNeutral300), n400 = hex(kNeutral400), n500 = hex(kNeutral500),
                      n600 = hex(kNeutral600), n700 = hex(kNeutral700);
        QString qss;
        // Every rule below is derived from docs/design/README.md "Design tokens":
        // 0 radius, 1.5 px control borders, 2 px region rules, flush-left
        // button labels, accent focus ring, 45 % disabled opacity emulated
        // with lightened colours (QSS has no opacity).
        // No font-family rule: QSS takes one family name, which would drop
        // the fallbacks theme::font() sets (and with them every character
        // Archivo does not carry, such as the Greek in "Emission λ").
        // QApplication::setFont in applyTheme carries the face instead.
        qss += QStringLiteral(
                   "* { font-size: 13px; color: %1; }\n"
                   "QMainWindow, QDialog, QDockWidget, QWidget#Panel { background: %2; }\n"
                   "QMainWindow::separator { background: %3; width: 2px; height: 2px; }\n"
                   "QMainWindow::separator:hover { background: %4; }\n"
                   "QToolTip { background: %2; color: %1; border: 1.5px solid %1; padding: 4px 8px; font-size: 12px; }\n")
                   .arg(text, bg, divider, accent);

        // menus
        qss += QStringLiteral(
                   "QMenuBar { background: %2; font-size: 13px; padding: 0; spacing: 2px; border-bottom: 2px solid %6; min-height: 36px; }\n"
                   "QMenuBar::item { background: transparent; padding: 0 9px; }\n"
                   "QMenuBar::item:selected { background: %1; color: %2; }\n"
                   "QMenuBar::item:pressed { background: %1; color: %2; }\n"
                   "QMenu { background: %2; border: 2px solid %1; padding: 4px 0; font-size: 12px; min-width: 240px; }\n"
                   "QMenu::item { padding: 6px 12px 6px 28px; background: transparent; }\n"
                   "QMenu::item:selected { background: %3; color: %4; }\n"
                   "QMenu::item:disabled { color: %5; }\n"
                   "QMenu::separator { height: 1px; background: %6; margin: 4px 0; }\n"
                   "QMenu::indicator { width: 10px; height: 10px; left: 10px; }\n"
                   "QMenu::indicator:checked, QMenu::indicator:exclusive:checked { image: none; background: %4; }\n"
                   "QMenu::indicator:unchecked, QMenu::indicator:exclusive:unchecked { image: none; background: transparent; }\n"
                   "QMenu::right-arrow { width: 8px; height: 8px; }\n")
                   .arg(text, bg, n200, accent, n500, divider);

        // docks
        qss += QStringLiteral(
                   "QDockWidget { titlebar-close-icon: none; titlebar-normal-icon: none; border: none; }\n"
                   "QDockWidget::title { background: %1; padding: 0; margin: 0; height: 0; }\n"
                   "QDockWidget::close-button, QDockWidget::float-button { width: 0; height: 0; }\n")
                   .arg(bg);

        // buttons
        qss += QStringLiteral(
                   "QPushButton { background: transparent; border: 1.5px solid %1; border-radius: 0; padding: 7px 12px;"
                   "  text-align: left; font-weight: 600; font-size: 13px; min-height: 18px; }\n"
                   "QPushButton:hover { border-color: %2; color: %2; }\n"
                   "QPushButton:pressed { background: %3; }\n"
                   "QPushButton:disabled { color: %4; border-color: %5; }\n"
                   "QPushButton[focusVisible=\"true\"] { border: 2px solid %2; }\n"
                   "QPushButton[class~=\"primary\"] { background: %2; color: %6; border-color: %2; font-weight: 800; }\n"
                   "QPushButton[class~=\"primary\"]:hover { background: %7; border-color: %7; color: %6; }\n"
                   "QPushButton[class~=\"primary\"]:pressed { background: %8; }\n"
                   "QPushButton[class~=\"primary\"]:disabled { background: %5; border-color: %5; color: %6; }\n"
                   "QPushButton[class~=\"secondary\"] { border-color: %1; color: %1; }\n"
                   "QPushButton[class~=\"ghost\"] { border-color: transparent; padding: 7px 8px; }\n"
                   "QPushButton[class~=\"ghost\"]:hover { color: %2; }\n"
                   "QPushButton[class~=\"link\"] { border: none; padding: 0; color: %2; font-weight: 400; font-size: 11px; }\n"
                   "QPushButton[class~=\"link\"]:hover { color: %7; }\n"
                   "QPushButton[class~=\"small\"] { padding: 4px 10px; font-size: 12px; min-height: 14px; }\n"
                   "QToolButton { background: transparent; border: 1.5px solid %9; border-radius: 0; padding: 2px; }\n"
                   "QToolButton:hover { border-color: %2; }\n"
                   "QToolButton:checked { background: %2; color: %6; border-color: %2; }\n")
                   .arg(text, accent, n200, n500, n300, bg, accent600, hex(kAccent700), divider);

        // check boxes / radios
        qss += QStringLiteral(
                   "QCheckBox { spacing: 8px; font-size: 12px; background: transparent; }\n"
                   "QCheckBox::indicator { width: 14px; height: 14px; border: 1.5px solid %1; background: transparent; }\n"
                   "QCheckBox::indicator:checked { background: %2; }\n"
                   "QCheckBox::indicator:hover { border-color: %2; }\n"
                   "QCheckBox::indicator:disabled { border-color: %3; }\n"
                   "QCheckBox::indicator:checked:disabled { background: %3; }\n"
                   "QRadioButton { spacing: 8px; font-size: 12px; }\n"
                   "QRadioButton::indicator { width: 12px; height: 12px; border: 1.5px solid %1; border-radius: 6px; }\n"
                   "QRadioButton::indicator:checked { background: %2; }\n")
                   .arg(n700, accent, n400);

        // inputs
        qss += QStringLiteral(
                   "QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {"
                   "  background: %1; border: 1.5px solid %2; border-radius: 0; padding: 0 8px; min-height: 29px;"
                   "  selection-background-color: %3; selection-color: %1; }\n"
                   "QPlainTextEdit, QTextEdit { padding: 6px 8px; }\n"
                   "QLineEdit[focusVisible=\"true\"], QSpinBox[focusVisible=\"true\"], QDoubleSpinBox[focusVisible=\"true\"],"
                   " QComboBox[focusVisible=\"true\"], QPlainTextEdit[focusVisible=\"true\"], QTextEdit[focusVisible=\"true\"]"
                   " { border: 2px solid %3; }\n"
                   "QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled { color: %4; border-color: %5; }\n"
                   "QLineEdit[readOnly=\"true\"] { background: %6; }\n"
                   "QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {"
                   "  width: 16px; border: none; background: transparent; }\n"
                   "QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { width: 7px; height: 7px; }\n"
                   "QComboBox::drop-down { border: none; width: 22px; }\n"
                   "QComboBox QAbstractItemView { background: %1; border: 2px solid %7; selection-background-color: %8;"
                   "  selection-color: %3; outline: none; padding: 2px 0; }\n"
                   "QComboBox QAbstractItemView::item { min-height: 24px; padding: 2px 8px; }\n")
                   .arg(bg, divider, accent, n500, n300, surface, text, n200);

        // sliders, scroll bars, progress
        qss += QStringLiteral(
                   "QSlider { min-height: 18px; background: transparent; }\n"
                   "QSlider::groove:horizontal { height: 2px; background: %1; margin: 0; }\n"
                   "QSlider::sub-page:horizontal { background: %2; }\n"
                   "QSlider::handle:horizontal { width: 10px; height: 14px; margin: -6px 0; background: %2; border: none; }\n"
                   "QSlider::handle:horizontal:hover { background: %3; }\n"
                   "QSlider:disabled::sub-page:horizontal, QSlider::handle:horizontal:disabled { background: %4; }\n"
                   "QScrollBar:vertical { background: transparent; width: 8px; margin: 0; }\n"
                   "QScrollBar::handle:vertical { background: %4; min-height: 24px; }\n"
                   "QScrollBar::handle:vertical:hover { background: %5; }\n"
                   "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }\n"
                   "QScrollBar:horizontal { background: transparent; height: 8px; margin: 0; }\n"
                   "QScrollBar::handle:horizontal { background: %4; min-width: 24px; }\n"
                   "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }\n"
                   "QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }\n"
                   "QProgressBar { background: %1; border: none; height: 4px; max-height: 4px; }\n"
                   "QProgressBar::chunk { background: %2; }\n"
                   "QScrollArea { border: none; background: transparent; }\n"
                   "QScrollArea > QWidget > QWidget { background: transparent; }\n")
                   .arg(n300, accent, accent600, n400, n500);

        // tables, tabs, status bar, splitters, group boxes
        qss += QStringLiteral(
                   "QTableView, QTableWidget, QTreeView, QListView { background: transparent; border: none; gridline-color: %1;"
                   "  selection-background-color: %2; selection-color: %3; alternate-background-color: transparent; outline: none; }\n"
                   "QTableView::item, QListView::item { padding: 4px 6px; border-bottom: 1px solid %1; }\n"
                   "QHeaderView { background: transparent; }\n"
                   "QHeaderView::section { background: transparent; border: none; border-bottom: 2px solid %1;"
                   "  padding: 4px 6px; font-size: 10px; font-weight: 400; color: %4; text-transform: uppercase; }\n"
                   "QTabWidget::pane { border: none; border-top: 2px solid %1; }\n"
                   "QTabBar { background: transparent; }\n"
                   "QTabBar::tab { background: transparent; border: none; border-bottom: 2px solid transparent;"
                   "  padding: 4px 10px; font-size: 12px; margin-right: 2px; min-width: 54px; }\n"
                   "QTabBar::tab:selected { border-bottom-color: %5; font-weight: 800; }\n"
                   "QTabBar::tab:hover { color: %5; }\n"
                   "QStatusBar { background: %6; border-top: 2px solid %1; font-size: 11px; color: %7; min-height: 26px; max-height: 26px; }\n"
                   "QStatusBar::item { border: none; }\n"
                   "QStatusBar QLabel { font-size: 11px; color: %7; }\n"
                   "QSplitter::handle { background: %1; }\n"
                   "QGroupBox { border: none; margin-top: 0; }\n"
                   "QLabel { background: transparent; }\n"
                   "QLabel[class~=\"muted\"] { color: %7; }\n"
                   "QLabel[class~=\"small\"] { font-size: 11px; color: %7; }\n"
                   "QLabel[class~=\"error\"] { color: %5; font-size: 11px; }\n"
                   "QFrame[class~=\"floating\"] { background: %6; border: 2px solid %3; }\n"
                   "QDialog QLabel[class~=\"title\"] { font-size: 20px; font-weight: 800; }\n")
                   .arg(divider, surface, text, n600, accent, bg, n600);
        // Component classes the panels set with widgets::setWidgetClass, so no
        // widget needs a style sheet of its own: theme.cpp stays the only place
        // that writes QSS.
        qss += QStringLiteral(
                   "QPushButton[class~=\"icon\"] { text-align: center; padding: 0; }\n"
                   "QPushButton[class~=\"tiny\"] { padding: 5px 9px; font-size: 11px; min-height: 12px; }\n"
                   "QPushButton[class~=\"chip\"] { border: 1.5px solid %1; padding: 3px 8px; font-size: 11px;"
                   "  font-weight: 400; min-height: 0; }\n"
                   "QPushButton[class~=\"chip\"]:hover { border-color: %2; color: %2; }\n"
                   "QFrame[class~=\"card\"] { border: 1.5px solid %1; }\n"
                   "QFrame[class~=\"card-accent\"] { border: 1.5px solid %2; }\n"
                   "QFrame[class~=\"card\"] QLabel, QFrame[class~=\"card-accent\"] QLabel"
                   "  { border: none; }\n"
                   "QLabel[class~=\"bubble\"] { background: %3; color: %4; padding: 8px 12px; }\n"
                   "QLabel[class~=\"mono\"] { font-family: monospace; font-size: 11px; color: %5; }\n")
                   .arg(divider, accent, text, bg, n700);
        qss += QStringLiteral(
                   "QTextBrowser { background: %1; border: none; padding: 0; }\n"
                   "QTableView[class~=\"dense\"]::item { padding: 2px 6px; }\n"
                   "QPlainTextEdit[class~=\"code\"] { font-family: monospace; font-size: 13px; background: %1;"
                   "  color: %2; border: none; padding: 0; selection-background-color: %3; selection-color: %1; }\n"
                   "QPlainTextEdit[class~=\"log\"] { font-family: monospace; font-size: 12px; color: %4;"
                   "  background: %5; border: 1px solid %6; }\n"
                   "QPlainTextEdit[class~=\"banner\"] { font-family: monospace; font-size: 11px; color: %7;"
                   "  background: transparent; border: none; padding: 0; min-height: 0; }\n"
                   "QLabel[class~=\"onDark\"] { background: %8; color: %9; }\n"
            // Panes sitting on the viewer ground are parted by the design's
            // 2 px dark gaps, not by the divider every other splitter uses.
                   "QSplitter[class~=\"onDark\"]::handle { background: %10; }\n")
                   .arg(bg, text, accent, n700, surface, n300, hex(kAccent700), hex(kViewerGround), hex(kViewerText))
                   .arg(hex(kNeutral900));
        // Text at 11 px has to reach 4.5:1 (WCAG AA) and the accent does not
        // (3.8:1 on the background): the rules above set it on links and
        // error lines, these later rules -- same specificity, so they win --
        // put the darkened accent there instead. Fills keep kAccent.
        qss += QStringLiteral(
                   "QLabel[class~=\"error\"] { color: %1; }\n"
                   "QPushButton[class~=\"link\"] { color: %1; }\n"
                   "QPushButton[class~=\"link\"]:hover { color: %2; }\n")
                   .arg(hex(kAccentText), hex(kAccent700));
        return qss;
    }

    void applyTheme(QApplication& app) {
        loadFonts();
        app.setStyle(QStringLiteral("Fusion"));
        QPalette p = app.palette();
        p.setColor(QPalette::Window, kBg);
        p.setColor(QPalette::WindowText, kText);
        p.setColor(QPalette::Base, kBg);
        p.setColor(QPalette::AlternateBase, kSurface);
        p.setColor(QPalette::Text, kText);
        p.setColor(QPalette::Button, kBg);
        p.setColor(QPalette::ButtonText, kText);
        p.setColor(QPalette::Highlight, kAccent);
        p.setColor(QPalette::HighlightedText, kBg);
        p.setColor(QPalette::ToolTipBase, kBg);
        p.setColor(QPalette::ToolTipText, kText);
        p.setColor(QPalette::PlaceholderText, kNeutral500);
        p.setColor(QPalette::Link, kAccent);
        p.setColor(QPalette::Disabled, QPalette::Text, kNeutral500);
        p.setColor(QPalette::Disabled, QPalette::WindowText, kNeutral500);
        p.setColor(QPalette::Disabled, QPalette::ButtonText, kNeutral500);
        app.setPalette(p);
        app.setFont(font(kBodyPx));
        app.setStyleSheet(styleSheet());
        installFocusVisibleFilter(app);
    }

} // namespace sirius::app::theme
