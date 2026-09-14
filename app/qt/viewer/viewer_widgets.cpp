#include "qt/viewer/viewer_widgets.hpp"

#include <algorithm>
#include <cmath>

#include <QFocusEvent>
#include <QFontMetrics>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QPen>

#include "qt/qt_strings.hpp"
#include "qt/theme.hpp"
#include "qt/widgets/controls.hpp"

namespace sirius::app {

    namespace {
        QPen rule(const QColor& c, double w = 1.5) {
            QPen p(c, w);
            p.setJoinStyle(Qt::MiterJoin);
            return p;
        }

        // The design's focus ring: 2 px accent, on whole device pixels.
        void drawFocusRing(QPainter& p, const QWidget& w) {
            if (!w.property("focusVisible").toBool()) return;
            const qreal dpr = w.devicePixelRatioF();
            p.save();
            p.setOpacity(1.0);
            p.setRenderHint(QPainter::Antialiasing, false);
            p.setPen(QPen(theme::kAccent, widgets::crispPen(2.0, dpr)));
            p.setBrush(Qt::NoBrush);
            p.drawRect(widgets::crispRect(QRectF(w.rect()), 2.0, dpr));
            p.restore();
        }

        // A rectangle filled on whole device pixels (the range slider's
        // track and handles, which sit at fractional coordinates).
        void fillCrisp(QPainter& p, const QRectF& r, const QColor& c, qreal dpr) {
            if (dpr <= 0.0) dpr = 1.0;
            const qreal x0 = std::round(r.left() * dpr) / dpr, x1 = std::round(r.right() * dpr) / dpr;
            const qreal y0 = std::round(r.top() * dpr) / dpr, y1 = std::round(r.bottom() * dpr) / dpr;
            p.fillRect(QRectF(QPointF(x0, y0), QPointF(std::max(x1, x0 + 1.0 / dpr), std::max(y1, y0 + 1.0 / dpr))), c);
        }
    } // namespace

    void drawOverlayText(QPainter& p, const QPointF& topLeft, const QString& text, bool bold, double opacity, int px) {
        p.save();
        p.setOpacity(opacity);
        p.setFont(theme::font(px, bold ? QFont::ExtraBold : QFont::Normal));
        p.setPen(theme::kViewerText);
        const QFontMetrics fm(p.font());
        p.drawText(QPointF(topLeft.x(), topLeft.y() + fm.ascent()), text);
        p.restore();
    }

    // --- TokenCheck -----------------------------------------------------------------

    TokenCheck::TokenCheck(const QString& label, QWidget* parent) : QAbstractButton(parent) {
        setText(label);
        setCheckable(true);
        setCursor(Qt::PointingHandCursor);
        setFocusPolicy(Qt::StrongFocus);
        setAccessibleName(label);
        setAccessibleDescription(QStringLiteral("Space or enter turns %1 on and off.").arg(label));
    }

    void TokenCheck::setCaption(const QString& caption) {
        if (caption_ == caption) return;
        caption_ = caption;
        updateGeometry();
        update();
    }

    QSize TokenCheck::sizeHint() const {
        const QFontMetrics fm(theme::font(12));
        int w = 14 + 8 + fm.horizontalAdvance(text());
        if (!caption_.isEmpty()) {
            QFont cf = theme::font(10);
            cf.setLetterSpacing(QFont::PercentageSpacing, 106);
            w += 6 + QFontMetrics(cf).horizontalAdvance(captionCase(caption_)) + 8;
        }
        return {w, 22};
    }

    void TokenCheck::paintEvent(QPaintEvent*) {
        QPainter p(this);
        if (!isEnabled()) p.setOpacity(0.45);
        const qreal dpr = devicePixelRatioF();
        const QRectF box = widgets::crispRect(QRectF(0, (height() - 14) / 2.0, 14, 14), 1.5, dpr);
        if (isChecked()) p.fillRect(box, theme::kAccent);
        p.setPen(rule(theme::kNeutral700, widgets::crispPen(1.5, dpr)));
        p.drawRect(box);
        p.setFont(theme::font(12));
        p.setPen(theme::kText);
        const QFontMetrics fm(p.font());
        int x = 22;
        p.drawText(QPointF(x, (height() + fm.ascent() - fm.descent()) / 2.0), text());
        x += fm.horizontalAdvance(text());
        if (!caption_.isEmpty()) {
            QFont cf = theme::font(10);
            cf.setLetterSpacing(QFont::PercentageSpacing, 106);
            p.setFont(cf);
            p.setPen(theme::kNeutral500);
            const QFontMetrics cfm(cf);
            p.drawText(QPointF(x + 6, (height() + cfm.ascent() - cfm.descent()) / 2.0), captionCase(caption_));
        }
        drawFocusRing(p, *this);
    }

    void TokenCheck::keyPressEvent(QKeyEvent* e) {
        if (e->key() == Qt::Key_Return || e->key() == Qt::Key_Enter) {
            e->accept();
            click();
            return;
        }
        QAbstractButton::keyPressEvent(e);
    }

    void TokenCheck::focusInEvent(QFocusEvent* e) {
        QAbstractButton::focusInEvent(e);
        update();
    }

    void TokenCheck::focusOutEvent(QFocusEvent* e) {
        QAbstractButton::focusOutEvent(e);
        update();
    }

    // --- ChannelSwatch ----------------------------------------------------------------

    ChannelSwatch::ChannelSwatch(const QString& label, const QColor& color, QWidget* parent)
        : QAbstractButton(parent), label_(label), color_(color) {
        setCheckable(true);
        setChecked(true);
        setCursor(Qt::PointingHandCursor);
        setFocusPolicy(Qt::StrongFocus);
        setFixedSize(22, 22);
        setToolTip(label);
        setAccessibleName(QStringLiteral("Channel %1").arg(label));
        setAccessibleDescription(QStringLiteral("Space or enter shows and hides this channel."));
    }

    void ChannelSwatch::paintEvent(QPaintEvent*) {
        QPainter p(this);
        const qreal dpr = devicePixelRatioF();
        const QRectF r = widgets::crispRect(QRectF(rect()), 1.5, dpr);
        if (isChecked()) p.fillRect(r, color_);
        p.setPen(rule(color_, widgets::crispPen(1.5, dpr)));
        p.drawRect(r);
        p.setFont(theme::font(10));
        p.setPen(isChecked() ? theme::kViewerGround : color_);
        p.drawText(rect(), Qt::AlignCenter, label_);
        drawFocusRing(p, *this);
    }

    void ChannelSwatch::keyPressEvent(QKeyEvent* e) {
        if (e->key() == Qt::Key_Return || e->key() == Qt::Key_Enter) {
            e->accept();
            click();
            return;
        }
        QAbstractButton::keyPressEvent(e);
    }

    void ChannelSwatch::focusInEvent(QFocusEvent* e) {
        QAbstractButton::focusInEvent(e);
        update();
    }

    void ChannelSwatch::focusOutEvent(QFocusEvent* e) {
        QAbstractButton::focusOutEvent(e);
        update();
    }

    // --- RangeSlider ---------------------------------------------------------------------

    RangeSlider::RangeSlider(QWidget* parent) : QWidget(parent) {
        setFixedSize(120, 14);
        setCursor(Qt::SizeHorCursor);
        setFocusPolicy(Qt::StrongFocus);
        setAccessibleName(QStringLiteral("Clip range"));
        describe();
    }

    void RangeSlider::describe() {
        setAccessibleDescription(QStringLiteral("From %1 % to %2 %. Left and right arrows move the %3 handle, "
                                                "up and down switch handles.")
                                     .arg(std::lround(lo_ * 100.0))
                                     .arg(std::lround(hi_ * 100.0))
                                     .arg(handle_ == 1 ? QStringLiteral("lower") : QStringLiteral("upper")));
    }

    void RangeSlider::setRange(double lo, double hi) {
        lo = std::clamp(lo, 0.0, 1.0);
        hi = std::clamp(hi, lo, 1.0);
        if (lo == lo_ && hi == hi_) return;
        lo_ = lo;
        hi_ = hi;
        describe();
        update();
    }

    double RangeSlider::fromX(double x) const { return std::clamp((x - 2.0) / (width() - 4.0), 0.0, 1.0); }

    void RangeSlider::paintEvent(QPaintEvent*) {
        QPainter p(this);
        const qreal dpr = devicePixelRatioF();
        const double y = height() / 2.0;
        fillCrisp(p, QRectF(2, y - 1, width() - 4, 2), QColor(243, 242, 242, 90), dpr);
        const double x0 = 2 + lo_ * (width() - 4), x1 = 2 + hi_ * (width() - 4);
        fillCrisp(p, QRectF(x0, y - 1, std::max(1.0, x1 - x0), 2), theme::kAccent, dpr);
        // handles: small squares the design would draw in accent
        fillCrisp(p, QRectF(x0 - 2, y - 4, 4, 8), theme::kAccent, dpr);
        fillCrisp(p, QRectF(x1 - 2, y - 4, 4, 8), theme::kAccent, dpr);
        // which handle the arrow keys move
        if (property("focusVisible").toBool()) {
            const double hx = handle_ == 1 ? x0 : x1;
            p.setRenderHint(QPainter::Antialiasing, false);
            p.setPen(QPen(theme::kViewerText, widgets::crispPen(1.0, dpr)));
            p.setBrush(Qt::NoBrush);
            p.drawRect(widgets::crispRect(QRectF(hx - 3.5, y - 5.5, 7, 11), 1.0, dpr));
        }
        drawFocusRing(p, *this);
    }

    void RangeSlider::keyPressEvent(QKeyEvent* e) {
        const double step = e->modifiers().testFlag(Qt::ShiftModifier) ? 0.005 : 0.02;
        double lo = lo_, hi = hi_;
        switch (e->key()) {
            case Qt::Key_Left:
                if (handle_ == 1) lo = std::max(0.0, lo_ - step);
                else hi = std::max(lo_, hi_ - step);
                break;
            case Qt::Key_Right:
                if (handle_ == 1) lo = std::min(hi_, lo_ + step);
                else hi = std::min(1.0, hi_ + step);
                break;
            case Qt::Key_Home:
                if (handle_ == 1) lo = 0.0;
                else hi = lo_;
                break;
            case Qt::Key_End:
                if (handle_ == 1) lo = hi_;
                else hi = 1.0;
                break;
            case Qt::Key_Up:
            case Qt::Key_Down:
            case Qt::Key_Space:
                handle_ = handle_ == 1 ? 2 : 1;
                describe();
                update();
                e->accept();
                return;
            default: QWidget::keyPressEvent(e); return;
        }
        e->accept();
        if (lo == lo_ && hi == hi_) return;
        lo_ = lo;
        hi_ = hi;
        describe();
        update();
        emit rangeChanged(lo_, hi_);
    }

    void RangeSlider::focusInEvent(QFocusEvent* e) {
        QWidget::focusInEvent(e);
        update();
    }

    void RangeSlider::focusOutEvent(QFocusEvent* e) {
        QWidget::focusOutEvent(e);
        update();
    }

    void RangeSlider::mousePressEvent(QMouseEvent* e) {
        const double v = fromX(e->position().x());
        drag_ = std::abs(v - lo_) <= std::abs(v - hi_) ? 1 : 2;
        handle_ = drag_;
        mouseMoveEvent(e);
    }

    void RangeSlider::mouseMoveEvent(QMouseEvent* e) {
        if (!drag_) return;
        const double v = fromX(e->position().x());
        double lo = lo_, hi = hi_;
        if (drag_ == 1) lo = std::min(v, hi_);
        else hi = std::max(v, lo_);
        if (lo != lo_ || hi != hi_) {
            lo_ = lo;
            hi_ = hi;
            describe();
            update();
            emit rangeChanged(lo_, hi_);
        }
    }

} // namespace sirius::app
