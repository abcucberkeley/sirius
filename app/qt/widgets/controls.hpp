#ifndef SIRIUS_APP_WIDGETS_CONTROLS_HPP
#define SIRIUS_APP_WIDGETS_CONTROLS_HPP

// Small custom controls the design uses everywhere and stock Qt does not
// have in this shape: a segmented control / tile row (outlined options,
// selected = ink fill or accent fill), a square icon button (the view /
// help / dock chrome and the tool strip), caption labels (10 px uppercase,
// 0.1 em tracking), rules between regions and a clickable row with hover /
// selected states. Everything paints from theme:: tokens and the icon table
// in widgets/icons.hpp; nothing here reads QSS.

#include <QAbstractButton>
#include <QColor>
#include <QFrame>
#include <QLabel>
#include <QList>
#include <QSize>
#include <QString>
#include <QStringList>
#include <QWidget>

#include "qt/widgets/icons.hpp"

class QEnterEvent;
class QFocusEvent;
class QKeyEvent;

namespace sirius::app::widgets {

    // "Ortho | 3D | Compare", "CUDA | CPU | HPC", "sum | mean | max | min".
    class SegmentedControl : public QWidget {
        Q_OBJECT
    public:
        explicit SegmentedControl(QWidget* parent = nullptr);
        SegmentedControl(const QStringList& options, QWidget* parent = nullptr);

        void setOptions(const QStringList& options);
        QStringList options() const { return options_; }
        int currentIndex() const { return current_; }
        void setCurrentIndex(int index);          // does not emit changed()
        QString currentText() const;
        void setCurrentText(const QString& text);
        // Tiles: equal-width, 36 px high, 13 px text (Backend / Cache rows).
        // Segmented: compact, 26 px, 12 px text, hugs its content.
        void setTileMode(bool tiles);
        // Selected option drawn with the accent instead of ink (the "active" semantic).
        void setAccentSelection(bool accent) {
            accent_ = accent;
            update();
        }
        void setOptionEnabled(int index, bool enabled);
        void setOptionToolTip(int index, const QString& tip);
        QSize sizeHint() const override;
        QSize minimumSizeHint() const override;

    signals:
        void changed(int index);                  // user input only

    protected:
        void paintEvent(QPaintEvent*) override;
        void mousePressEvent(QMouseEvent* e) override;
        void mouseMoveEvent(QMouseEvent* e) override;
        void leaveEvent(QEvent*) override;
        void keyPressEvent(QKeyEvent* e) override;
        void focusInEvent(QFocusEvent* e) override;
        void focusOutEvent(QFocusEvent* e) override;
        void changeEvent(QEvent* e) override;
        bool event(QEvent* e) override;

    private:
        int indexAt(const QPoint& p) const;
        QRect rectOf(int index) const;
        int optionWidth(int index) const;
        QString textFor(int index) const;      // the option, elided if its tile is too narrow

        QStringList options_;
        QList<bool> enabled_;
        QStringList tips_;
        int current_ = 0;
        int hover_ = -1;
        bool tiles_ = false;
        bool accent_ = false;
    };

    // Square button carrying one painted icon (or, where the design really
    // does want a letter, a short text glyph): 1.5 px border, accent fill
    // when active. The single button of the whole application -- the tool
    // strip, the ◉ view buttons, the dock chrome, the transport, the 3D
    // presets -- so `onDark` also covers the ones sitting on the viewer
    // ground.
    class GlyphButton : public QAbstractButton {
        Q_OBJECT
    public:
        explicit GlyphButton(Icon icon, int size = 24, QWidget* parent = nullptr);
        GlyphButton(Icon icon, QSize size, QWidget* parent = nullptr);
        explicit GlyphButton(const QString& glyph, int size = 24, QWidget* parent = nullptr);
        GlyphButton(const QString& glyph, QSize size, QWidget* parent = nullptr);

        void setGlyph(const QString& glyph);
        void setSymbol(Icon icon);
        Icon symbol() const { return icon_; }
        void setSize(int w, int h);
        // Active = accent fill + paper icon (independent of checked so a
        // non-checkable button can still show a state).
        void setActive(bool on);
        bool isActive() const { return active_; }
        // Painted size of a text glyph inside the button.
        void setGlyphPx(int px) {
            glyphPx_ = px;
            update();
        }
        // Side of the icon's box; 0 (the default) scales it to the button.
        void setIconPx(int px) {
            iconPx_ = px;
            update();
        }
        // No border when idle (the reorder chevrons).
        void setBorderless(bool on) {
            borderless_ = on;
            update();
        }
        // Dashed border (the "+" add square).
        void setDashed(bool on) {
            dashed_ = on;
            update();
        }
        // Idle icon colour (default: text).
        void setIdleColor(const QColor& c) {
            idle_ = c;
            update();
        }
        // Idle border colour (default: divider).
        void setBorderColor(const QColor& c) {
            border_ = c;
            update();
        }
        // Sitting on the viewer ground: paper icon, translucent idle border.
        void setOnDark(bool on) {
            onDark_ = on;
            update();
        }
        // Half-strength idle colours (a control that is available but not
        // the one in play).
        void setDimmed(bool on) {
            dimmed_ = on;
            update();
        }
        QSize sizeHint() const override { return {w_, h_}; }
        QSize minimumSizeHint() const override { return {w_, h_}; }

    protected:
        void paintEvent(QPaintEvent*) override;
        void keyPressEvent(QKeyEvent* e) override;
        void enterEvent(QEnterEvent* e) override;
        void leaveEvent(QEvent* e) override;

    private:
        QString glyph_;
        Icon icon_ = Icon::None;
        int w_, h_;
        int glyphPx_ = 11;
        int iconPx_ = 0;
        bool active_ = false;
        bool hover_ = false;
        bool borderless_ = false;
        bool dashed_ = false;
        bool onDark_ = false;
        bool dimmed_ = false;
        QColor idle_;
        QColor border_;
    };

    // A label that keeps its full text and cuts it to whatever width it is
    // given (a QLabel does not elide by itself, it clips). Its minimum width
    // is zero, so it never widens the row it sits in -- which is what lets
    // the 290 px Operations dock hold long operation names.
    class ElidedLabel : public QLabel {
        Q_OBJECT
    public:
        explicit ElidedLabel(QWidget* parent = nullptr, Qt::TextElideMode mode = Qt::ElideRight);
        void setFullText(const QString& text);
        QString fullText() const { return full_; }

    protected:
        void resizeEvent(QResizeEvent* e) override;

    private:
        void relayout();
        QString full_;
        Qt::TextElideMode mode_;
    };

    // 10 px uppercase, 0.1 em tracking, neutral-600 (or accent).
    class CaptionLabel : public QLabel {
        Q_OBJECT
    public:
        explicit CaptionLabel(const QString& text = {}, QWidget* parent = nullptr);
        void setAccent(bool on);
        void setColor(const QColor& c);
        void setText(const QString& text);   // upper-cases
    };

    // 2 px (region) or 1 px (row) rule.
    class Rule : public QFrame {
        Q_OBJECT
    public:
        explicit Rule(int px = 2, Qt::Orientation orientation = Qt::Horizontal, QWidget* parent = nullptr);
        void setColor(const QColor& c);

    protected:
        void paintEvent(QPaintEvent*) override;

    private:
        int px_;
        Qt::Orientation orientation_;
        QColor color_;
    };

    // A row that emits clicked() anywhere on it and paints its own hover /
    // selected / accent-edge states; children are laid out by the caller.
    class ClickRow : public QWidget {
        Q_OBJECT
    public:
        explicit ClickRow(QWidget* parent = nullptr);
        void setSelected(bool on);
        bool isSelected() const { return selected_; }
        void setHoverable(bool on) { hoverable_ = on; }
        void setTopRule(int px) {
            topRule_ = px;
            update();
        }
        void setEdge(bool on) {
            edge_ = on;
            update();
        }   // 3 px accent left edge when selected
        void setFill(const QColor& c) {
            fill_ = c;
            update();
        }   // idle background

    signals:
        void clicked();

    protected:
        void paintEvent(QPaintEvent*) override;
        void mousePressEvent(QMouseEvent* e) override;
        void mouseReleaseEvent(QMouseEvent* e) override;
        void enterEvent(QEnterEvent*) override;
        void leaveEvent(QEvent*) override;
        void keyPressEvent(QKeyEvent* e) override;
        void focusInEvent(QFocusEvent* e) override;
        void focusOutEvent(QFocusEvent* e) override;

    private:
        bool selected_ = false;
        bool hover_ = false;
        bool hoverable_ = true;
        bool pressed_ = false;
        bool edge_ = false;
        int topRule_ = 1;
        QColor fill_;
    };

    // --- helpers ------------------------------------------------------------
    // Crisp strokes at fractional device pixel ratios (1.25x, 1.5x, 2x).
    // A 1.5 px pen on a rect inset by 0.75 only lands on whole device pixels
    // at integer scale factors; these round the pen to a whole number of
    // device pixels and put the stroke where its edges fall on device pixel
    // boundaries, so a border is one solid line and not two grey ones.
    qreal crispPen(qreal logicalPx, qreal dpr);
    QRectF crispRect(const QRectF& outer, qreal logicalPx, qreal dpr);
    // The same for a single line: snaps a coordinate to the stroke centre.
    qreal crispLine(qreal coordinate, qreal logicalPx, qreal dpr);

    // QLabel with the body font at `px` and an optional colour / weight.
    QLabel* label(const QString& text, int px, const QColor& color, int weight = -1, QWidget* parent = nullptr);
    // Heading (800) label.
    QLabel* heading(const QString& text, int px, QWidget* parent = nullptr);
    // Sets the "class" dynamic property that theme.cpp's QSS styles
    // ("primary", "secondary", "ghost", "link", "chip", "card", "code" …)
    // and repolishes the widget so the new rules take effect.
    void setWidgetClass(QWidget* w, const char* cls);
    inline void setButtonClass(QWidget* button, const char* cls) { setWidgetClass(button, cls); }
    // Small colour chip (w × h, filled with `color`, no border).
    QWidget* colorChip(const QColor& color, int w = 10, int h = 10, QWidget* parent = nullptr);
    void setChipColor(QWidget* chip, const QColor& color);
    // Elide a string to a width with the widget's font.
    QString elide(const QWidget* w, const QString& s, int width);
    // Tabular figures on a widget's font, so columns of numbers line up.
    void useTabularNumbers(QWidget* w);
    // "12.8 GB", "412 MB", "48 kB" -- core's formatBytes, so every byte
    // readout in the application rounds the same way.
    QString bytesText(quint64 bytes);
    // shadow-lg / shadow-md of the design tokens, for floating panels and
    // the pop-over menus (QSS cannot express a box shadow).
    void applyShadow(QWidget* w, bool large);
    void clearShadow(QWidget* w);

} // namespace sirius::app::widgets

#endif // SIRIUS_APP_WIDGETS_CONTROLS_HPP
