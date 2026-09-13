#include "qt/viewer/slice_pane.hpp"

#include <algorithm>
#include <cmath>

#include <QFocusEvent>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QElapsedTimer>
#include <QPainter>
#include <QPen>
#include <QResizeEvent>
#include <QWheelEvent>

#include "qt/theme.hpp"
#include "qt/trace.hpp"
#include "qt/widgets/controls.hpp"
#include "qt/viewer/viewer_constants.hpp"
#include "qt/viewer/viewer_widgets.hpp"

namespace sirius::app {

    SlicePane::SlicePane(Kind kind, QWidget* parent) : QWidget(parent), kind_(kind) {
        setMouseTracking(true);
        setMinimumSize(40, 40);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        setAutoFillBackground(false);
        setAttribute(Qt::WA_OpaquePaintEvent);
        setCursor(Qt::CrossCursor);
        // reachable from the keyboard: tab to a pane, then walk the crosshair
        setFocusPolicy(Qt::StrongFocus);
        static const struct { Kind kind; const char* h; const char* v; const char* d; const char* name; } axes[] = {
            {Kind::XY, "x", "y", "z", "XY view"},   {Kind::YZ, "z", "y", "x", "YZ view"},
            {Kind::XZ, "x", "z", "y", "XZ view"},   {Kind::MIP, "x", "y", "z", "Z maximum projection"},
            {Kind::Compare, "x", "y", "z", "Compare view"}};
        for (const auto& a : axes)
            if (a.kind == kind_) {
                setAxisNames(QString::fromLatin1(a.h), QString::fromLatin1(a.v), QString::fromLatin1(a.d));
                setAccessibleName(QString::fromLatin1(a.name));
                break;
            }
    }

    void SlicePane::setAxisNames(const QString& horizontal, const QString& vertical, const QString& depth) {
        axisH_ = horizontal;
        axisV_ = vertical;
        axisD_ = depth;
        setAccessibleDescription(QStringLiteral("Arrow keys move the crosshair along %1 and %2, page up and page down step %3; "
                                                "shift moves ten voxels at a time.")
                                     .arg(axisH_, axisV_, axisD_));
    }

    void SlicePane::setContent(const QImage& img, int factor, Index cols, Index rows, const QPoint& origin) {
        image_ = img;
        factor_ = std::max(factor, 1);
        cols_ = cols;
        rows_ = rows;
        origin_ = origin;
        update();
    }

    void SlicePane::clearContent() {
        image_ = QImage();
        cols_ = rows_ = 0;
        update();
    }

    void SlicePane::setView(const View& v) {
        view_ = v;
        update();
    }

    SlicePane::View SlicePane::fitView(double ax, double ay) const {
        View v;
        if (cols_ <= 0 || rows_ <= 0) return v;
        const double ex = static_cast<double>(cols_) * ax, ey = static_cast<double>(rows_) * ay;
        const double z = std::max(1e-6, std::min(width() / ex, height() / ey));
        v.zx = z * ax;
        v.zy = z * ay;
        v.ox = (width() - cols_ * v.zx) / 2.0;
        v.oy = (height() - rows_ * v.zy) / 2.0;
        return v;
    }

    QPointF SlicePane::toVoxel(const QPointF& s) const {
        return {(s.x() - view_.ox) / view_.zx, (s.y() - view_.oy) / view_.zy};
    }

    QPointF SlicePane::toScreen(const QPointF& v) const {
        return {view_.ox + v.x() * view_.zx, view_.oy + v.y() * view_.zy};
    }

    bool SlicePane::inside(const QPointF& v) const {
        return v.x() >= 0.0 && v.y() >= 0.0 && v.x() < static_cast<double>(cols_) && v.y() < static_cast<double>(rows_);
    }

    void SlicePane::setTitle(const QString& t) { title_ = t; update(); }
    void SlicePane::setHint(const QString& h) { hint_ = h; update(); }
    void SlicePane::setScaleBar(double um) { umPerVoxel_ = um; update(); }

    void SlicePane::setCrosshair(const QPointF& voxel, bool visible, bool locked) {
        cross_ = voxel;
        crossVisible_ = visible;
        crossLocked_ = locked;
        update();
    }

    void SlicePane::setBrushCursor(bool on, double radiusVoxels) {
        brush_ = on;
        brushRadius_ = radiusVoxels;
        update();
    }

    void SlicePane::setAnnotations(const QVector<Annotation>& annotations) {
        annotations_ = annotations;
        update();
    }

    void SlicePane::setTracks(std::shared_ptr<const QVector<TrackPath>> paths, const TrackPaintOptions& options) {
        const bool same = paths == tracks_ && options.t == trackOptions_.t && options.selected == trackOptions_.selected &&
                          options.tail == trackOptions_.tail && options.future == trackOptions_.future;
        tracks_ = std::move(paths);
        trackOptions_ = options;
        if (!same) update();
    }

    void SlicePane::setMessage(const QString& text) { message_ = text; update(); }

    // --- painting ----------------------------------------------------------------

    void SlicePane::paintEvent(QPaintEvent*) {
        const bool trace = ScopedTrace::enabled();
        QElapsedTimer clock;
        if (trace) clock.start();
        QPainter p(this);
        p.fillRect(rect(), theme::kViewerGround);

        if (!image_.isNull() && cols_ > 0 && rows_ > 0) {
            p.save();
            QTransform t;
            t.translate(view_.ox + origin_.x() * view_.zx, view_.oy + origin_.y() * view_.zy);
            t.scale(view_.zx * factor_, view_.zy * factor_);
            p.setTransform(t);
            // nearest neighbour when magnifying keeps voxels crisp; smooth
            // when shrinking (in device pixels) avoids aliasing on large frames
            p.setRenderHint(QPainter::SmoothPixmapTransform, smooth_ || view_.zx * factor_ * devicePixelRatioF() < 1.0);
            p.drawImage(QPointF(0, 0), image_);
            p.restore();
        }

        p.setRenderHint(QPainter::Antialiasing, true);

        if (tracks_ && hasContent())
            paintTrackPaths(p, *tracks_, trackOptions_, [this](const QPointF& v) { return toScreen(v); }, QRectF(rect()));

        // annotations (ROI boxes dashed, measurements in accent) sit under the crosshair
        if (hasContent()) {
            for (const Annotation& a : annotations_) {
                p.setOpacity(a.pending ? 0.6 : 1.0);
                if (a.kind == Annotation::Kind::Roi) {
                    if (a.rect.isNull()) continue;
                    const QRectF r(toScreen(a.rect.topLeft()), toScreen(a.rect.bottomRight()));
                    p.setPen(QPen(theme::kViewerText, 1.0, Qt::DashLine));
                    p.setBrush(Qt::NoBrush);
                    p.drawRect(r);
                    if (!a.text.isEmpty()) drawOverlayText(p, r.topLeft() + QPointF(4, -16), a.text, true);
                } else {
                    if (a.points.isEmpty()) continue;
                    p.setPen(QPen(theme::kAccent, 1.5));
                    for (int i = 0; i + 1 < a.points.size(); ++i) p.drawLine(toScreen(a.points[i]), toScreen(a.points[i + 1]));
                    for (const QPointF& v : a.points) {
                        const QPointF sp = toScreen(v);
                        p.drawLine(sp + QPointF(-4, 0), sp + QPointF(4, 0));
                        p.drawLine(sp + QPointF(0, -4), sp + QPointF(0, 4));
                    }
                    if (!a.text.isEmpty()) drawOverlayText(p, toScreen(a.points.last()) + QPointF(8, -16), a.text, true);
                }
            }
            p.setOpacity(1.0);
        }

        if (crossVisible_ && hasContent()) {
            const QPointF c = toScreen(cross_ + QPointF(0.5, 0.5));
            QPen pen(theme::kAccent, 1.0);
            if (crossLocked_) {
                pen.setStyle(Qt::DashLine);
                p.setOpacity(0.45);
            }
            const qreal dpr = devicePixelRatioF();
            pen.setWidthF(widgets::crispPen(1.0, dpr));
            p.setPen(pen);
            p.setRenderHint(QPainter::Antialiasing, false);
            const double x = widgets::crispLine(c.x(), 1.0, dpr), y = widgets::crispLine(c.y(), 1.0, dpr);
            p.drawLine(QPointF(x, 0), QPointF(x, height()));
            p.drawLine(QPointF(0, y), QPointF(width(), y));
            p.setOpacity(1.0);
            p.setRenderHint(QPainter::Antialiasing, true);
        }

        if (brush_ && mouseIn_ && hasContent()) {
            const double r = std::max(1.0, brushRadius_ * view_.zx);
            p.setPen(QPen(theme::kViewerText, 1.5));
            p.setBrush(Qt::NoBrush);
            p.drawEllipse(mouse_, r, r * (view_.zy / std::max(1e-9, view_.zx)));
        }

        // corner label, scale bar, hint
        if (!title_.isEmpty()) {
            // bold first token, the rest at 70 %
            const int cut = title_.indexOf(QLatin1String("  "));
            QFont bold = theme::font(11, QFont::ExtraBold);
            bold.setLetterSpacing(QFont::PercentageSpacing, 108);
            const QFontMetrics fm(bold);
            const QString head = cut < 0 ? title_ : title_.left(cut);
            drawOverlayText(p, QPointF(viewer::kOverlayInset, viewer::kOverlayTop), head, true);
            if (cut >= 0)
                drawOverlayText(p, QPointF(viewer::kOverlayInset + fm.horizontalAdvance(head) + viewer::kOverlayGap,
                                           viewer::kOverlayTop),
                                title_.mid(cut + 2), false, 0.7);
        }
        if (umPerVoxel_ > 0.0 && hasContent()) {
            // largest of the design's steps that stays under 140 px
            static const double steps[] = {50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05};
            double um = steps[sizeof steps / sizeof steps[0] - 1];
            for (double s : steps) {
                if (s / umPerVoxel_ * view_.zx <= viewer::kScaleBarMaxPx) {
                    um = s;
                    break;
                }
            }
            const double px = um / umPerVoxel_ * view_.zx;
            QString label = um >= 1.0 ? QString::number(um) + QStringLiteral(" µm") : QString::number(um * 1000.0) + QStringLiteral(" nm");
            const QFontMetrics fm(theme::tabular(theme::font(11)));
            const int lw = fm.horizontalAdvance(label);
            const double x1 = width() - viewer::kOverlayInset - lw - 6.0;
            const qreal dpr = devicePixelRatioF();
            const double barY = std::round((height() - viewer::kOverlayBottom - 8.0) * dpr) / dpr;
            p.fillRect(QRectF(std::round((x1 - px) * dpr) / dpr, barY, std::round(px * dpr) / dpr,
                              widgets::crispPen(2.0, dpr)),
                       theme::kViewerText);
            drawOverlayText(p, QPointF(width() - viewer::kOverlayInset - lw, height() - viewer::kOverlayBottom - fm.height()),
                            label);
        }
        if (!hint_.isEmpty()) {
            const QFontMetrics fm(theme::font(11));
            drawOverlayText(p, QPointF(viewer::kOverlayInset, height() - viewer::kOverlayBottom - fm.height()), hint_, false, 0.75);
        }
        if (!message_.isEmpty()) {
            p.setFont(theme::font(12));
            p.setPen(QColor(243, 242, 242, 180));
            p.drawText(rect().adjusted(12, 12, -12, -12), Qt::AlignCenter | Qt::TextWordWrap, message_);
        }
        // the design's 2 px accent focus ring, keyboard focus only
        if (property("focusVisible").toBool()) {
            const qreal dpr = devicePixelRatioF();
            const qreal pen = widgets::crispPen(2.0, dpr);
            p.setRenderHint(QPainter::Antialiasing, false);
            p.setPen(QPen(theme::kAccent, pen));
            p.setBrush(Qt::NoBrush);
            p.drawRect(widgets::crispRect(QRectF(rect()), 2.0, dpr));
        }
        if (trace) qInfo("pane %s paint %lld us (%dx%d, image %dx%d)", objectName().isEmpty() ? "?" : qPrintable(objectName()), clock.nsecsElapsed() / 1000, width(), height(), image_.width(), image_.height());
    }

    // --- mouse ----------------------------------------------------------------------

    void SlicePane::mousePressEvent(QMouseEvent* e) {
        mouse_ = e->position();
        if (e->button() == Qt::RightButton) {
            emit contextMenuRequested(e->position().toPoint(), toVoxel(mouse_));
            return;
        }
        button_ = e->button();
        pressPos_ = lastDrag_ = mouse_;
        moved_ = false;
        emit pressed(toVoxel(mouse_), e->button(), e->modifiers());
    }

    void SlicePane::mouseMoveEvent(QMouseEvent* e) {
        mouse_ = e->position();
        mouseIn_ = true;
        if (button_ != Qt::NoButton && (e->buttons() & button_)) {
            const QPointF delta = mouse_ - lastDrag_;
            lastDrag_ = mouse_;
            if ((mouse_ - pressPos_).manhattanLength() > 2) moved_ = true;
            emit dragged(toVoxel(mouse_), delta, button_, e->modifiers());
        }
        emit hovered(toVoxel(mouse_));
        if (brush_) update();
    }

    void SlicePane::mouseReleaseEvent(QMouseEvent* e) {
        mouse_ = e->position();
        const Qt::MouseButton b = button_;
        button_ = Qt::NoButton;
        emit released(toVoxel(mouse_), b == Qt::NoButton ? e->button() : b, e->modifiers(), moved_);
    }

    void SlicePane::mouseDoubleClickEvent(QMouseEvent* e) {
        if (e->button() == Qt::LeftButton) emit doubleClicked(toVoxel(e->position()), e->modifiers());
    }

    void SlicePane::wheelEvent(QWheelEvent* e) {
        const double steps = e->angleDelta().y() / 120.0;
        if (steps == 0.0) return;
        emit wheeled(e->position(), steps, e->modifiers());
        e->accept();
    }

    void SlicePane::leaveEvent(QEvent*) {
        mouseIn_ = false;
        emit exited();
        if (brush_) update();
    }

    void SlicePane::resizeEvent(QResizeEvent*) { emit resized(); }

    void SlicePane::keyPressEvent(QKeyEvent* e) {
        const int step = e->modifiers().testFlag(Qt::ShiftModifier) ? 10 : 1;
        int dc = 0, dr = 0, dd = 0;
        switch (e->key()) {
            case Qt::Key_Left: dc = -step; break;
            case Qt::Key_Right: dc = step; break;
            case Qt::Key_Up: dr = -step; break;
            case Qt::Key_Down: dr = step; break;
            case Qt::Key_PageUp: dd = -step; break;
            case Qt::Key_PageDown: dd = step; break;
            default: QWidget::keyPressEvent(e); return;
        }
        emit keyNavigated(dc, dr, dd);
        e->accept();
    }

    void SlicePane::focusInEvent(QFocusEvent* e) {
        QWidget::focusInEvent(e);
        update();
    }

    void SlicePane::focusOutEvent(QFocusEvent* e) {
        QWidget::focusOutEvent(e);
        update();
    }

} // namespace sirius::app
