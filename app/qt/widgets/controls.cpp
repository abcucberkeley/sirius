#include "qt/widgets/controls.hpp"

#include <algorithm>
#include <cmath>

#include <QEnterEvent>
#include <QFocusEvent>
#include <QKeyEvent>
#include <QEvent>
#include <QFontMetrics>
#include <QGraphicsDropShadowEffect>
#include <QHelpEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <QResizeEvent>
#include <QStyle>
#include <QToolTip>

#include "core/ops/builtin.hpp"
#include "qt/qt_strings.hpp"
#include "qt/theme.hpp"

namespace sirius::app::widgets {

    // --- SegmentedControl -----------------------------------------------------

    SegmentedControl::SegmentedControl(QWidget* parent) : QWidget(parent) {
        setMouseTracking(true);
        setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        setFont(theme::font(12));
        // a radio group in behaviour: tab in, arrows change the choice
        setFocusPolicy(Qt::StrongFocus);
        setAccessibleDescription(QStringLiteral("Left and right arrow keys change the selection."));
    }

    SegmentedControl::SegmentedControl(const QStringList& options, QWidget* parent) : SegmentedControl(parent) {
        setOptions(options);
    }

    void SegmentedControl::setOptions(const QStringList& options) {
        options_ = options;
        if (accessibleName().isEmpty() && !options.isEmpty())
            setAccessibleName(options.join(QStringLiteral(" or ")));
        enabled_.clear();
        tips_.clear();
        for (int i = 0; i < options_.size(); ++i) {
            enabled_.push_back(true);
            tips_.push_back(QString());
        }
        current_ = options_.isEmpty() ? -1 : std::min(std::max(current_, 0), static_cast<int>(options_.size()) - 1);
        updateGeometry();
        update();
    }

    void SegmentedControl::setCurrentIndex(int index) {
        if (index < 0 || index >= options_.size() || current_ == index) return;
        current_ = index;
        update();
    }

    QString SegmentedControl::currentText() const {
        return current_ >= 0 && current_ < options_.size() ? options_[current_] : QString();
    }

    void SegmentedControl::setCurrentText(const QString& text) {
        const int i = static_cast<int>(options_.indexOf(text));
        if (i >= 0) setCurrentIndex(i);
    }

    void SegmentedControl::setTileMode(bool tiles) {
        tiles_ = tiles;
        setFont(theme::font(tiles ? 13 : 12));
        setSizePolicy(tiles ? QSizePolicy::Expanding : QSizePolicy::Preferred, QSizePolicy::Fixed);
        updateGeometry();
        update();
    }

    void SegmentedControl::setOptionEnabled(int index, bool enabled) {
        if (index < 0 || index >= enabled_.size()) return;
        enabled_[index] = enabled;
        update();
    }

    void SegmentedControl::setOptionToolTip(int index, const QString& tip) {
        if (index < 0 || index >= tips_.size()) return;
        tips_[index] = tip;
    }

    // Elide rather than clip. A tile can end up narrower than its text -- a
    // dock dragged in, or a layout restored from before the bundled font
    // loaded -- and "Recomput" with the R cut off reads as a rendering fault,
    // where "Recom..." reads as a tile too small to say it, which the tooltip
    // then says in full.
    QString SegmentedControl::textFor(int index) const {
        const QFontMetrics fm(font());
        const int room = rectOf(index).width() - 4;
        if (fm.horizontalAdvance(options_[index]) <= room) return options_[index];
        return fm.elidedText(options_[index], Qt::ElideRight, room);
    }

    int SegmentedControl::optionWidth(int index) const {
        if (tiles_) {
            const int n = std::max(1, static_cast<int>(options_.size()));
            const int gaps = 2 * (n - 1);
            const int base = (width() - gaps) / n;
            return index == n - 1 ? width() - gaps - base * (n - 1) : base;
        }
        const QFontMetrics fm(font());
        return fm.horizontalAdvance(options_[index]) + 22;
    }

    QRect SegmentedControl::rectOf(int index) const {
        int x = 0;
        for (int i = 0; i < index; ++i) x += optionWidth(i) + 2;
        return QRect(x, 0, optionWidth(index), height());
    }

    QSize SegmentedControl::sizeHint() const {
        const int h = tiles_ ? 36 : 26;
        if (tiles_) return {static_cast<int>(options_.size()) * 60, h};
        int w = 0;
        const QFontMetrics fm(font());
        for (int i = 0; i < options_.size(); ++i) w += fm.horizontalAdvance(options_[i]) + 22 + (i ? 2 : 0);
        return {w, h};
    }

    QSize SegmentedControl::minimumSizeHint() const {
        // Segmented mode never shrinks below its text: the design's
        // "Ortho | 3D | Compare" must not read "Orth" when the toolbar is
        // tight -- whatever sits beside it gives way instead.
        if (!tiles_) return sizeHint();
        // Tiles divide the width equally, so the widest option decides: 3 x 40
        // ignored the text and let "Memory | Disk | Recompute" lose glyphs off
        // both ends of the last tile in a narrowed dock. Padding is tighter
        // than the 60 px of sizeHint(): this is the floor, not the preference.
        const QFontMetrics fm(font());
        int widest = 0;
        for (const QString& option : options_) widest = std::max(widest, fm.horizontalAdvance(option));
        return {static_cast<int>(options_.size()) * std::max(40, widest + 12), 36};
    }

    // The bundled face arrives after the widget is built: re-measure.
    void SegmentedControl::changeEvent(QEvent* e) {
        if (e->type() == QEvent::FontChange || e->type() == QEvent::StyleChange) {
            updateGeometry();
            update();
        }
        QWidget::changeEvent(e);
    }

    void SegmentedControl::keyPressEvent(QKeyEvent* e) {
        int dir = 0;
        int from = current_;
        switch (e->key()) {
            case Qt::Key_Left:
            case Qt::Key_Up: dir = -1; break;
            case Qt::Key_Right:
            case Qt::Key_Down: dir = 1; break;
            case Qt::Key_Home:
                dir = 1;
                from = -1;
                break;
            case Qt::Key_End:
                dir = -1;
                from = static_cast<int>(options_.size());
                break;
            default: QWidget::keyPressEvent(e); return;
        }
        for (int i = from + dir; i >= 0 && i < options_.size(); i += dir) {
            if (!enabled_[i]) continue;
            if (i != current_) {
                current_ = i;
                update();
                emit changed(i);
            }
            break;
        }
        e->accept();
    }

    void SegmentedControl::focusInEvent(QFocusEvent* e) {
        QWidget::focusInEvent(e);
        update();
    }

    void SegmentedControl::focusOutEvent(QFocusEvent* e) {
        QWidget::focusOutEvent(e);
        update();
    }

    int SegmentedControl::indexAt(const QPoint& p) const {
        for (int i = 0; i < options_.size(); ++i)
            if (rectOf(i).contains(p)) return i;
        return -1;
    }

    void SegmentedControl::paintEvent(QPaintEvent*) {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing, false);
        painter.setFont(font());
        const qreal dpr = devicePixelRatioF();
        const qreal pen1 = crispPen(1.5, dpr);
        const QColor selFill = accent_ ? theme::kAccent : theme::kText;
        const bool focused = property("focusVisible").toBool();
        for (int i = 0; i < options_.size(); ++i) {
            const QRect r = rectOf(i);
            const bool sel = i == current_;
            const bool en = enabled_[i] && isEnabled();
            painter.setOpacity(en ? 1.0 : 0.45);
            painter.setPen(QPen(sel ? selFill : (hover_ == i && en ? theme::kAccent : theme::kDivider), pen1));
            painter.setBrush(sel ? QBrush(selFill) : Qt::NoBrush);
            painter.drawRect(crispRect(QRectF(r), 1.5, dpr));
            painter.setPen(sel ? theme::kBg : theme::kText);
            painter.drawText(r, Qt::AlignCenter, textFor(i));
            if (focused && sel) {
                painter.setPen(QPen(theme::kAccent, crispPen(2.0, dpr)));
                painter.setBrush(Qt::NoBrush);
                painter.drawRect(crispRect(QRectF(r), 2.0, dpr));
            }
        }
    }

    void SegmentedControl::mousePressEvent(QMouseEvent* e) {
        if (e->button() != Qt::LeftButton) return;
        const int i = indexAt(e->pos());
        if (i < 0 || !enabled_[i]) return;
        if (i != current_) {
            current_ = i;
            update();
            emit changed(i);
        }
    }

    void SegmentedControl::mouseMoveEvent(QMouseEvent* e) {
        const int i = indexAt(e->pos());
        if (i != hover_) {
            hover_ = i;
            update();
        }
    }

    void SegmentedControl::leaveEvent(QEvent*) {
        hover_ = -1;
        update();
    }

    bool SegmentedControl::event(QEvent* e) {
        if (e->type() == QEvent::ToolTip) {
            auto* he = static_cast<QHelpEvent*>(e);
            const int i = indexAt(he->pos());
            QString tip = i >= 0 ? tips_[i] : QString();
            // an elided tile says what it could not draw, whether or not the
            // option was given a tooltip of its own
            if (i >= 0 && textFor(i) != options_[i])
                tip = tip.isEmpty() ? options_[i] : options_[i] + QStringLiteral(" · ") + tip;
            if (!tip.isEmpty()) QToolTip::showText(he->globalPos(), tip, this);
            else QToolTip::hideText();
            return true;
        }
        return QWidget::event(e);
    }

    // --- GlyphButton ------------------------------------------------------------

    GlyphButton::GlyphButton(const QString& glyph, int size, QWidget* parent)
        : GlyphButton(glyph, QSize(size, size), parent) {}

    GlyphButton::GlyphButton(const QString& glyph, QSize size, QWidget* parent)
        : QAbstractButton(parent), glyph_(glyph), w_(size.width()), h_(size.height()), idle_(theme::kText),
          border_(theme::kDivider) {
        setCursor(Qt::PointingHandCursor);
        // reachable from the keyboard: space (QAbstractButton) or enter
        setFocusPolicy(Qt::StrongFocus);
        setFixedSize(w_, h_);
    }

    GlyphButton::GlyphButton(Icon icon, int size, QWidget* parent) : GlyphButton(icon, QSize(size, size), parent) {}

    GlyphButton::GlyphButton(Icon icon, QSize size, QWidget* parent) : GlyphButton(QString(), size, parent) {
        icon_ = icon;
    }

    void GlyphButton::setGlyph(const QString& glyph) {
        glyph_ = glyph;
        icon_ = Icon::None;
        update();
    }

    void GlyphButton::setSymbol(Icon icon) {
        if (icon_ == icon) return;
        icon_ = icon;
        glyph_.clear();
        update();
    }

    void GlyphButton::setSize(int w, int h) {
        w_ = w;
        h_ = h;
        setFixedSize(w, h);
        updateGeometry();
    }

    void GlyphButton::setActive(bool on) {
        if (active_ == on) return;
        active_ = on;
        update();
    }

    void GlyphButton::paintEvent(QPaintEvent*) {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing, false);
        painter.setOpacity(isEnabled() ? 1.0 : 0.35);
        const bool on = active_ || isChecked();
        const bool live = hover_ && isEnabled();
        const qreal dpr = devicePixelRatioF();
        QColor border = on || live ? theme::kAccent : border_;
        if (onDark_ && !on && !live) border = QColor(theme::kViewerText.red(), theme::kViewerText.green(),
                                                     theme::kViewerText.blue(), 128);
        else if (dimmed_ && !on && !live) border = theme::kNeutral400;
        if (borderless_ && !on) border = Qt::transparent;
        QPen pen(border, crispPen(1.5, dpr));
        if (dashed_ && !on) pen.setStyle(Qt::DashLine);
        painter.setPen(pen);
        painter.setBrush(on ? QBrush(theme::kAccent) : Qt::NoBrush);
        painter.drawRect(crispRect(QRectF(rect()), 1.5, dpr));
        if (property("focusVisible").toBool()) {
            painter.setPen(QPen(theme::kAccent, crispPen(2.0, dpr)));
            painter.setBrush(Qt::NoBrush);
            painter.drawRect(crispRect(QRectF(rect()), 2.0, dpr));
        }
        QColor fg = idle_;
        if (on) fg = theme::kBg;
        else if (hover_ && borderless_) fg = theme::kAccent;
        else if (onDark_) fg = theme::kViewerText;
        else if (dimmed_) fg = theme::kNeutral600;
        if (icon_ != Icon::None) {
            const double side = iconPx_ > 0 ? iconPx_ : std::max(10.0, std::min(w_, h_) * 0.68);
            const double stroke = side >= 20.0 ? 1.8 : (side >= 14.0 ? 1.5 : 1.25);
            const QPointF c = QRectF(rect()).center();
            drawIcon(painter, QRectF(c.x() - side / 2.0, c.y() - side / 2.0, side, side), icon_, fg, stroke);
            return;
        }
        painter.setPen(fg);
        painter.setFont(theme::font(glyphPx_));
        painter.drawText(rect(), Qt::AlignCenter, glyph_);
    }

    void GlyphButton::keyPressEvent(QKeyEvent* e) {
        // QAbstractButton answers to space; the design's controls also answer
        // to enter, which is what a keyboard user reaches for first.
        if (e->key() == Qt::Key_Return || e->key() == Qt::Key_Enter) {
            e->accept();
            click();
            return;
        }
        QAbstractButton::keyPressEvent(e);
    }

    void GlyphButton::enterEvent(QEnterEvent* e) {
        hover_ = true;
        update();
        QAbstractButton::enterEvent(e);
    }

    void GlyphButton::leaveEvent(QEvent* e) {
        hover_ = false;
        update();
        QAbstractButton::leaveEvent(e);
    }

    // --- ElidedLabel ---------------------------------------------------------------

    ElidedLabel::ElidedLabel(QWidget* parent, Qt::TextElideMode mode) : QLabel(parent), mode_(mode) {
        setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    }

    void ElidedLabel::setFullText(const QString& text) {
        full_ = text;
        setToolTip(text);
        relayout();
    }

    void ElidedLabel::resizeEvent(QResizeEvent* e) {
        QLabel::resizeEvent(e);
        relayout();
    }

    void ElidedLabel::relayout() { QLabel::setText(fontMetrics().elidedText(full_, mode_, std::max(20, width() - 2))); }

    // --- CaptionLabel ------------------------------------------------------------

    CaptionLabel::CaptionLabel(const QString& text, QWidget* parent) : QLabel(parent) {
        setFont(theme::caption());
        setAccent(false);
        setText(text);
    }

    void CaptionLabel::setAccent(bool on) { setColor(on ? theme::kAccent : theme::kNeutral600); }

    void CaptionLabel::setColor(const QColor& c) {
        QPalette p = palette();
        p.setColor(QPalette::WindowText, c);
        setPalette(p);
    }

    void CaptionLabel::setText(const QString& text) { QLabel::setText(captionCase(text)); }

    // --- Rule ------------------------------------------------------------------------

    Rule::Rule(int px, Qt::Orientation orientation, QWidget* parent)
        : QFrame(parent), px_(px), orientation_(orientation), color_(theme::kDivider) {
        setFrameShape(QFrame::NoFrame);
        if (orientation == Qt::Horizontal) {
            setFixedHeight(px);
            setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        } else {
            setFixedWidth(px);
            setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
        }
    }

    void Rule::setColor(const QColor& c) {
        color_ = c;
        update();
    }

    void Rule::paintEvent(QPaintEvent*) {
        QPainter painter(this);
        painter.fillRect(rect(), color_);
    }

    // --- ClickRow ------------------------------------------------------------------

    ClickRow::ClickRow(QWidget* parent) : QWidget(parent) {
        setAttribute(Qt::WA_Hover, true);
        setCursor(Qt::PointingHandCursor);
        setFocusPolicy(Qt::StrongFocus);
        setAccessibleDescription(QStringLiteral("Space or enter selects this row."));
    }

    void ClickRow::setSelected(bool on) {
        if (selected_ == on) return;
        selected_ = on;
        update();
    }

    void ClickRow::paintEvent(QPaintEvent*) {
        QPainter painter(this);
        if (selected_) painter.fillRect(rect(), theme::kSurface);
        else if (hover_ && hoverable_) painter.fillRect(rect(), theme::kNeutral200);
        else if (fill_.isValid()) painter.fillRect(rect(), fill_);
        if (topRule_ > 0) painter.fillRect(QRect(0, 0, width(), topRule_), theme::kDivider);
        if (edge_ && selected_) painter.fillRect(QRect(0, 0, 3, height()), theme::kAccent);
        if (property("focusVisible").toBool()) {
            const qreal dpr = devicePixelRatioF();
            painter.setRenderHint(QPainter::Antialiasing, false);
            painter.setPen(QPen(theme::kAccent, crispPen(2.0, dpr)));
            painter.setBrush(Qt::NoBrush);
            painter.drawRect(crispRect(QRectF(rect()), 2.0, dpr));
        }
    }

    void ClickRow::keyPressEvent(QKeyEvent* e) {
        if (e->key() == Qt::Key_Space || e->key() == Qt::Key_Return || e->key() == Qt::Key_Enter) {
            e->accept();
            emit clicked();
            return;
        }
        QWidget::keyPressEvent(e);
    }

    void ClickRow::focusInEvent(QFocusEvent* e) {
        QWidget::focusInEvent(e);
        update();
    }

    void ClickRow::focusOutEvent(QFocusEvent* e) {
        QWidget::focusOutEvent(e);
        update();
    }

    void ClickRow::mousePressEvent(QMouseEvent* e) {
        if (e->button() == Qt::LeftButton) pressed_ = true;
    }

    void ClickRow::mouseReleaseEvent(QMouseEvent* e) {
        if (e->button() == Qt::LeftButton && pressed_ && rect().contains(e->pos())) emit clicked();
        pressed_ = false;
    }

    void ClickRow::enterEvent(QEnterEvent*) {
        hover_ = true;
        update();
    }

    void ClickRow::leaveEvent(QEvent*) {
        hover_ = false;
        pressed_ = false;
        update();
    }

    // --- helpers ----------------------------------------------------------------------

    qreal crispPen(qreal logicalPx, qreal dpr) {
        if (dpr <= 0.0) dpr = 1.0;
        return std::max(1.0, std::round(logicalPx * dpr)) / dpr;
    }

    QRectF crispRect(const QRectF& outer, qreal logicalPx, qreal dpr) {
        if (dpr <= 0.0) dpr = 1.0;
        const qreal device = std::max(1.0, std::round(logicalPx * dpr));
        const qreal half = device / 2.0;
        const qreal x0 = std::round(outer.left() * dpr) + half;
        const qreal y0 = std::round(outer.top() * dpr) + half;
        const qreal x1 = std::max(x0, std::round(outer.right() * dpr) - half);
        const qreal y1 = std::max(y0, std::round(outer.bottom() * dpr) - half);
        return QRectF(QPointF(x0 / dpr, y0 / dpr), QPointF(x1 / dpr, y1 / dpr));
    }

    qreal crispLine(qreal coordinate, qreal logicalPx, qreal dpr) {
        if (dpr <= 0.0) dpr = 1.0;
        const qreal device = std::max(1.0, std::round(logicalPx * dpr));
        return (std::round(coordinate * dpr) + device / 2.0) / dpr;
    }

    QLabel* label(const QString& text, int px, const QColor& color, int weight, QWidget* parent) {
        auto* l = new QLabel(text, parent);
        l->setFont(theme::font(px, weight < 0 ? QFont::Normal : weight));
        QPalette p = l->palette();
        p.setColor(QPalette::WindowText, color);
        l->setPalette(p);
        return l;
    }

    QLabel* heading(const QString& text, int px, QWidget* parent) {
        auto* l = new QLabel(text, parent);
        l->setFont(theme::heading(px));
        return l;
    }

    void setWidgetClass(QWidget* w, const char* cls) {
        w->setProperty("class", QString::fromLatin1(cls));
        w->style()->unpolish(w);
        w->style()->polish(w);
    }

    QWidget* colorChip(const QColor& color, int w, int h, QWidget* parent) {
        auto* chip = new QWidget(parent);
        chip->setFixedSize(w, h);
        chip->setAutoFillBackground(true);
        setChipColor(chip, color);
        return chip;
    }

    void setChipColor(QWidget* chip, const QColor& color) {
        QPalette p = chip->palette();
        p.setColor(QPalette::Window, color);
        chip->setPalette(p);
        chip->update();
    }

    QString elide(const QWidget* w, const QString& s, int width) {
        return QFontMetrics(w->font()).elidedText(s, Qt::ElideRight, width);
    }

    void useTabularNumbers(QWidget* w) { w->setFont(theme::tabular(w->font())); }

    QString bytesText(quint64 bytes) { return fromStd(formatBytes(bytes)); }

    void applyShadow(QWidget* w, bool large) {
        // docs/design/README.md: shadow-lg 0 12px 32px rgba(45,43,43,.22),
        // shadow-md 0 3px 10px rgba(45,43,43,.16). QSS has no box-shadow.
        auto* shadow = qobject_cast<QGraphicsDropShadowEffect*>(w->graphicsEffect());
        if (!shadow) {
            shadow = new QGraphicsDropShadowEffect(w);
            w->setGraphicsEffect(shadow);
        }
        QColor ink = theme::kNeutral900;
        ink.setAlphaF(large ? 0.22f : 0.16f);
        shadow->setColor(ink);
        shadow->setBlurRadius(large ? 32.0 : 10.0);
        shadow->setOffset(0.0, large ? 12.0 : 3.0);
    }

    void clearShadow(QWidget* w) { w->setGraphicsEffect(nullptr); }

} // namespace sirius::app::widgets
