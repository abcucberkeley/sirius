#include "qt/panels/diagnostic_cells.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <QFontMetrics>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QVBoxLayout>

#include "qt/qt_strings.hpp"
#include "qt/theme.hpp"
#include "qt/widgets/controls.hpp"

namespace sirius::app {

    namespace {
        QLabel* captionLabel(const QString& text, QWidget* parent) {
            auto* l = new QLabel(captionCase(text), parent);
            l->setFont(theme::caption());
            QPalette p = l->palette();
            p.setColor(QPalette::WindowText, theme::kNeutral600);
            l->setPalette(p);
            return l;
        }

        void paintPlaceholder(QPainter& p, const QRect& r, const QString& text) {
            QColor c = theme::kViewerText;
            c.setAlphaF(0.55);
            p.setPen(c);
            p.setFont(theme::font(theme::kSmallPx));
            p.drawText(r.adjusted(10, 8, -10, -8), Qt::AlignCenter | Qt::TextWordWrap, text);
        }

        QString formatNumber(double v) {
            if (!std::isfinite(v)) return QStringLiteral("–");
            if (std::abs(v) >= 1000.0 || v == std::floor(v)) return QString::number(std::llround(v));
            return QString::number(v, 'g', 4);
        }
    } // namespace

    // --- image rendering -------------------------------------------------------

    QImage renderDiagnosticImage(const DiagnosticImage& image) {
        if (image.rows <= 0 || image.cols <= 0 || image.values.size() < static_cast<std::size_t>(image.rows * image.cols))
            return {};
        const Index n = image.rows * image.cols;
        // robust window from a bounded sample
        std::vector<float> sample;
        const Index stride = std::max<Index>(1, n / 65536);
        sample.reserve(static_cast<std::size_t>(n / stride + 1));
        for (Index i = 0; i < n; i += stride) {
            const float v = image.values[static_cast<std::size_t>(i)];
            if (std::isfinite(v)) sample.push_back(v);
        }
        float lo = 0.0f, hi = 1.0f;
        if (!sample.empty()) {
            auto rank = [&](double frac) {
                return static_cast<std::ptrdiff_t>(std::llround(frac * static_cast<double>(sample.size() - 1)));
            };
            const std::ptrdiff_t kLo = rank(0.005), kHi = rank(0.998);
            std::nth_element(sample.begin(), sample.begin() + kLo, sample.end());
            lo = sample[static_cast<std::size_t>(kLo)];
            std::nth_element(sample.begin() + kLo, sample.begin() + kHi, sample.end());
            hi = sample[static_cast<std::size_t>(kHi)];
            if (!(hi > lo)) {
                const auto [mn, mx] = std::minmax_element(sample.begin(), sample.end());
                lo = *mn;
                hi = *mx;
            }
            if (!(hi > lo)) hi = lo + 1.0f;
        }
        QImage out(static_cast<int>(image.cols), static_cast<int>(image.rows), QImage::Format_Grayscale8);
        const float scale = 255.0f / (hi - lo);
        for (Index y = 0; y < image.rows; ++y) {
            uchar* row = out.scanLine(static_cast<int>(y));
            const float* src = image.values.data() + y * image.cols;
            for (Index x = 0; x < image.cols; ++x) {
                const float t = std::min(255.0f, std::max(0.0f, (src[x] - lo) * scale));
                row[x] = static_cast<uchar>(std::isfinite(t) ? t + 0.5f : 0.0f);
            }
        }
        return out;
    }

    // --- DiagnosticCell --------------------------------------------------------

    DiagnosticCell::DiagnosticCell(const QString& title, const QString& meta, QWidget* content, QWidget* parent)
        : QWidget(parent), content_(content) {
        setAutoFillBackground(true);
        QPalette pal = palette();
        pal.setColor(QPalette::Window, theme::kBg);
        setPalette(pal);
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(0);
        auto* captionRow = new QWidget(this);
        auto* cap = new QHBoxLayout(captionRow);
        cap->setContentsMargins(12, 6, 12, 4);
        cap->setSpacing(8);
        title_ = captionLabel(title, captionRow);
        meta_ = captionLabel(meta, captionRow);
        meta_->setFont(theme::font(theme::kCaptionPx));
        cap->addWidget(title_);
        cap->addStretch(1);
        cap->addWidget(meta_);
        layout->addWidget(captionRow, 0);
        if (content_) {
            content_->setParent(this);
            layout->addWidget(content_, 1);
        }
        setMinimumWidth(80);
    }

    void DiagnosticCell::setCaption(const QString& title, const QString& meta) {
        title_->setText(captionCase(title));
        meta_->setText(meta);
    }

    // --- ImageCellView ---------------------------------------------------------

    ImageCellView::ImageCellView(QWidget* parent) : QWidget(parent) {
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        setMinimumSize(60, 40);
    }

    void ImageCellView::setImage(const DiagnosticImage& image) {
        image_ = renderDiagnosticImage(image);
        marks_ = image.marks;
        placeholder_.clear();
        update();
    }

    void ImageCellView::clear(const QString& placeholder) {
        image_ = QImage();
        marks_.clear();
        placeholder_ = placeholder;
        update();
    }

    void ImageCellView::paintEvent(QPaintEvent*) {
        QPainter p(this);
        p.fillRect(rect(), theme::kViewerGround);
        if (image_.isNull()) {
            if (!placeholder_.isEmpty()) paintPlaceholder(p, rect(), placeholder_);
            return;
        }
        const double sx = static_cast<double>(width()) / image_.width();
        const double sy = static_cast<double>(height()) / image_.height();
        const double s = std::min(sx, sy);
        const double w = image_.width() * s, h = image_.height() * s;
        const QRectF target((width() - w) / 2.0, (height() - h) / 2.0, w, h);
        p.setRenderHint(QPainter::SmoothPixmapTransform, s < 1.0);
        p.drawImage(target, image_);
        if (marks_.empty()) return;
        p.setRenderHint(QPainter::Antialiasing, true);
        p.setFont(theme::font(theme::kCaptionPx));
        for (const DiagnosticMark& m : marks_) {
            const QColor c = m.accent ? theme::kAccent : theme::kViewerText;
            const QPointF at(target.left() + m.x * s, target.top() + m.y * s);
            const double r = std::max(2.0, m.radius * s);
            p.setPen(QPen(c, 1.2));
            switch (m.kind) {
                case DiagnosticMark::Kind::Cross:
                    p.drawLine(at + QPointF(-5, 0), at + QPointF(5, 0));
                    p.drawLine(at + QPointF(0, -5), at + QPointF(0, 5));
                    break;
                case DiagnosticMark::Kind::Circle: {
                    QColor fill = c;
                    fill.setAlphaF(0.35);
                    p.setBrush(fill);
                    p.drawEllipse(at, r, r);
                    p.setBrush(Qt::NoBrush);
                    break;
                }
                case DiagnosticMark::Kind::Ring:
                    p.setBrush(Qt::NoBrush);
                    p.drawEllipse(at, r, r);
                    break;
            }
            if (!m.text.empty()) p.drawText(at + QPointF(7, -4), fromStd(m.text));
        }
    }

    // --- CurveView ---------------------------------------------------------------

    CurveView::CurveView(QWidget* parent) : QWidget(parent) {
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        setMinimumSize(80, 50);
    }

    void CurveView::setCurve(const DiagnosticCurve& curve) {
        curve_ = curve;
        update();
    }

    void CurveView::clear() {
        curve_.reset();
        update();
    }

    void CurveView::paintEvent(QPaintEvent*) {
        QPainter p(this);
        p.fillRect(rect(), theme::kBg);
        const int labelH = 16;
        const QRect plot = rect().adjusted(14, 4, -14, -(labelH + 6));
        p.setPen(QPen(theme::kText, 2));
        p.drawLine(plot.bottomLeft(), plot.bottomRight());
        if (!curve_ || curve_->y.empty()) {
            QColor c = theme::kNeutral600;
            p.setPen(c);
            p.setFont(theme::font(theme::kSmallPx));
            p.drawText(plot, Qt::AlignCenter, QStringLiteral("Run the step to record the curve"));
            return;
        }
        const DiagnosticCurve& c = *curve_;
        const std::size_t n = c.y.size();
        double xmin = 0.0, xmax = static_cast<double>(n - 1);
        if (c.x.size() == n && n > 1) {
            xmin = *std::min_element(c.x.begin(), c.x.end());
            xmax = *std::max_element(c.x.begin(), c.x.end());
        }
        if (xmax <= xmin) xmax = xmin + 1.0;
        auto yval = [&](double y) { return c.logY ? std::log10(std::max(y, 1e-12)) : y; };
        double ymin = std::numeric_limits<double>::infinity(), ymax = -ymin;
        for (double y : c.y) {
            const double v = yval(y);
            if (!std::isfinite(v)) continue;
            ymin = std::min(ymin, v);
            ymax = std::max(ymax, v);
        }
        if (!std::isfinite(ymin)) {
            ymin = 0.0;
            ymax = 1.0;
        }
        if (!c.logY) ymin = std::min(ymin, 0.0);
        if (ymax <= ymin) ymax = ymin + 1.0;
        auto toX = [&](double x) { return plot.left() + (x - xmin) / (xmax - xmin) * plot.width(); };
        auto toY = [&](double y) { return plot.bottom() - (yval(y) - ymin) / (ymax - ymin) * (plot.height() - 2); };
        QPainterPath path;
        for (std::size_t i = 0; i < n; ++i) {
            const double x = c.x.size() == n ? c.x[i] : static_cast<double>(i);
            const QPointF pt(toX(x), toY(c.y[i]));
            if (i == 0) path.moveTo(pt);
            else path.lineTo(pt);
        }
        p.setRenderHint(QPainter::Antialiasing, true);
        p.setPen(QPen(theme::kAccent, 2));
        p.drawPath(path);
        if (c.stopX) {
            QPen dashed(theme::kText, 1, Qt::CustomDashLine);
            dashed.setDashPattern({3, 3});
            p.setPen(dashed);
            const double x = toX(*c.stopX);
            p.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()));
        }
        p.setPen(theme::kNeutral600);
        p.setFont(theme::font(theme::kSmallPx));
        const QRect labels(plot.left(), plot.bottom() + 4, plot.width(), labelH);
        if (!c.leftLabel.empty()) p.drawText(labels, Qt::AlignLeft | Qt::AlignVCenter, fromStd(c.leftLabel));
        if (!c.midLabel.empty()) p.drawText(labels, Qt::AlignHCenter | Qt::AlignVCenter, fromStd(c.midLabel));
        if (!c.rightLabel.empty()) p.drawText(labels, Qt::AlignRight | Qt::AlignVCenter, fromStd(c.rightLabel));
    }

    // --- HistogramView -----------------------------------------------------------

    HistogramView::HistogramView(QWidget* parent) : QWidget(parent) {
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        setMinimumSize(80, 50);
    }

    void HistogramView::setHistogram(const DiagnosticHistogram& h) {
        hist_ = h;
        update();
    }

    void HistogramView::clear() {
        hist_.reset();
        update();
    }

    void HistogramView::paintEvent(QPaintEvent*) {
        QPainter p(this);
        p.fillRect(rect(), theme::kBg);
        const QRect inner = rect().adjusted(14, 4, -14, -8);
        const int headerH = 18;
        if (!hist_) {
            p.setPen(theme::kNeutral600);
            p.setFont(theme::font(theme::kSmallPx));
            p.drawText(inner, Qt::AlignCenter, QStringLiteral("No intensity data yet"));
            return;
        }
        const DiagnosticHistogram& h = *hist_;
        // header: chip, label, window
        const QColor chip(static_cast<int>(h.color[0] * 255), static_cast<int>(h.color[1] * 255), static_cast<int>(h.color[2] * 255));
        p.fillRect(QRect(inner.left(), inner.top() + 4, 10, 10), chip);
        p.setFont(theme::heading(theme::kSmallPx));
        p.setPen(theme::kText);
        const QString label = fromStd(h.channel);
        p.drawText(QRect(inner.left() + 18, inner.top(), inner.width() - 18, headerH), Qt::AlignLeft | Qt::AlignVCenter, label);
        const int labelW = QFontMetrics(theme::heading(theme::kSmallPx)).horizontalAdvance(label);
        p.setFont(theme::font(theme::kSmallPx));
        p.setPen(theme::kNeutral600);
        p.drawText(QRect(inner.left() + 18 + labelW + 8, inner.top(), inner.width(), headerH), Qt::AlignLeft | Qt::AlignVCenter,
                   QStringLiteral("%1 – %2 · γ %3").arg(formatNumber(h.lo), formatNumber(h.hi), QString::number(h.gamma, 'g', 3)));
        // bars
        const QRect bars(inner.left(), inner.top() + headerH + 6, inner.width(), inner.height() - headerH - 6);
        p.setPen(QPen(theme::kDivider, 2));
        p.drawLine(bars.bottomLeft(), bars.bottomRight());
        if (h.bins.empty()) return;
        const double maxBin = std::max(1e-12, *std::max_element(h.bins.begin(), h.bins.end()));
        const int n = static_cast<int>(h.bins.size());
        const double gap = 2.0;
        const double bw = (bars.width() - gap * (n - 1)) / n;
        const double binW = (h.binHi - h.binLo) / n;
        for (int i = 0; i < n; ++i) {
            const double centre = h.binLo + (i + 0.5) * binW;
            const bool tail = centre < h.lo || centre > h.hi;
            const double frac = h.bins[static_cast<std::size_t>(i)] / maxBin;
            const double hh = std::max(1.0, frac * (bars.height() - 2));
            const QRectF bar(bars.left() + i * (bw + gap), bars.bottom() - 1 - hh, bw, hh);
            p.fillRect(bar, tail ? theme::kNeutral400 : theme::kText);
        }
    }

    // --- FactsView -----------------------------------------------------------------

    FactsView::FactsView(QWidget* parent) : QWidget(parent) {
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        setMinimumSize(80, 40);
    }

    void FactsView::setFacts(const std::vector<DiagnosticFact>& facts, const QString& lead, const QString& trailer) {
        facts_ = facts;
        lead_ = lead;
        trailer_ = trailer;
        updateGeometry();
        update();
    }

    QSize FactsView::sizeHint() const {
        int h = 12 + static_cast<int>(facts_.size()) * 24;
        if (!lead_.isEmpty()) h += 40;
        if (!trailer_.isEmpty()) h += 20;
        return {220, h};
    }

    void FactsView::paintEvent(QPaintEvent*) {
        QPainter p(this);
        p.fillRect(rect(), theme::kBg);
        const int left = 14, right = width() - 14;
        int y = 4;
        if (!lead_.isEmpty()) {
            p.setPen(theme::kText);
            p.setFont(theme::font(theme::kBodyPx));
            const QRect box(left, y, right - left, 1000);
            const QRect used = p.boundingRect(box, Qt::TextWordWrap, lead_);
            p.drawText(box, Qt::TextWordWrap, lead_);
            y += used.height() + 8;
        }
        p.setFont(theme::font(12));
        for (const DiagnosticFact& f : facts_) {
            const QRect row(left, y, right - left, 24);
            p.setPen(theme::kText);
            p.drawText(row, Qt::AlignLeft | Qt::AlignVCenter, fromStd(f.key));
            p.drawText(row, Qt::AlignRight | Qt::AlignVCenter, fromStd(f.value));
            p.setPen(QPen(theme::kDivider, 1));
            p.drawLine(left, y + 23, right, y + 23);
            y += 24;
        }
        if (!trailer_.isEmpty()) {
            p.setPen(theme::kNeutral600);
            p.setFont(theme::font(theme::kSmallPx));
            p.drawText(QRect(left, y + 6, right - left, 1000), Qt::TextWordWrap, trailer_);
        }
    }

    // --- TileMapView ---------------------------------------------------------------

    TileMapView::TileMapView(QWidget* parent) : QWidget(parent) {
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        setMinimumSize(80, 50);
    }

    void TileMapView::setAlignment(const AlignmentInfo& info) {
        info_ = info;
        update();
    }

    void TileMapView::paintEvent(QPaintEvent*) {
        QPainter p(this);
        p.fillRect(rect(), theme::kBg);
        const Index rows = std::max<Index>(info_.gridRows, 1), cols = std::max<Index>(info_.gridCols, 1);
        const QRect area = rect().adjusted(12, 8, -12, -8);
        const double gap = 4.0;
        const double tw = (area.width() - gap * (cols - 1)) / cols;
        const double th = (area.height() - gap * (rows - 1)) / rows;
        p.setFont(theme::font(theme::kSmallPx));
        for (Index r = 0; r < rows; ++r)
            for (Index c = 0; c < cols; ++c) {
                const int i = static_cast<int>(r * cols + c);
                const QRectF tile(area.left() + c * (tw + gap), area.top() + r * (th + gap), tw, th);
                const bool hi = i == info_.highlightedTile;
                p.fillRect(tile, hi ? theme::kAccent : theme::kBg);
                p.setPen(QPen(theme::kText, 1.5));
                p.drawRect(tile.adjusted(0.75, 0.75, -0.75, -0.75));
                p.setPen(hi ? theme::kBg : theme::kText);
                const QString name = i < static_cast<int>(info_.tileNames.size())
                                         ? fromStd(info_.tileNames[static_cast<std::size_t>(i)])
                                         : QStringLiteral("t%1").arg(i + 1);
                p.drawText(tile, Qt::AlignCenter, name);
            }
    }

    // --- DiagnosticTableView ---------------------------------------------------

    DiagnosticTableView::DiagnosticTableView(QWidget* parent) : QTableWidget(parent) {
        setFrameShape(QFrame::NoFrame);
        setShowGrid(false);
        setSelectionMode(QAbstractItemView::NoSelection);
        setEditTriggers(QAbstractItemView::NoEditTriggers);
        setFocusPolicy(Qt::NoFocus);
        verticalHeader()->setVisible(false);
        horizontalHeader()->setStretchLastSection(true);
        horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
        horizontalHeader()->setHighlightSections(false);
        horizontalHeader()->setDefaultAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        verticalHeader()->setDefaultSectionSize(22);
        setFont(theme::tabular(theme::font(theme::kSmallPx)));
        widgets::setWidgetClass(this, "dense");
    }

    void DiagnosticTableView::setTable(const DiagnosticTable& table) {
        clear();
        setColumnCount(static_cast<int>(table.header.size()));
        QStringList header;
        for (const std::string& h : table.header) header << captionCase(fromStd(h));
        setHorizontalHeaderLabels(header);
        setRowCount(static_cast<int>(table.rows.size()));
        QFont bold = theme::heading(theme::kSmallPx);
        for (int r = 0; r < rowCount(); ++r) {
            const auto& row = table.rows[static_cast<std::size_t>(r)];
            for (int c = 0; c < columnCount() && c < static_cast<int>(row.size()); ++c) {
                auto* item = new QTableWidgetItem(fromStd(row[static_cast<std::size_t>(c)]));
                const bool accent = std::find(table.accentCells.begin(), table.accentCells.end(), std::make_pair(r, c)) !=
                                    table.accentCells.end();
                if (accent) {
                    item->setForeground(theme::kAccentText);
                    item->setFont(bold);
                }
                setItem(r, c, item);
            }
        }
    }

    // --- DiagnosticsBody -------------------------------------------------------------

    DiagnosticsBody::DiagnosticsBody(QWidget* parent) : QWidget(parent) {
        setAutoFillBackground(true);
        QPalette pal = palette();
        pal.setColor(QPalette::Window, theme::kDivider);
        setPalette(pal);
        grid_ = new QGridLayout(this);
        grid_->setContentsMargins(0, 0, 0, 0);
        grid_->setSpacing(2);
    }

    void DiagnosticsBody::clearGrid() {
        while (QLayoutItem* item = grid_->takeAt(0)) {
            if (QWidget* w = item->widget()) w->deleteLater();
            delete item;
        }
        for (int c = 0; c < 8; ++c) grid_->setColumnStretch(c, 0);
    }

    DiagnosticCell* DiagnosticsBody::addCell(const QString& title, const QString& meta, QWidget* content, int column,
                                             int stretch, int fixedWidth) {
        auto* cell = new DiagnosticCell(title, meta, content, this);
        if (fixedWidth > 0) cell->setFixedWidth(fixedWidth);
        grid_->addWidget(cell, 0, column);
        grid_->setColumnStretch(column, stretch);
        return cell;
    }

    QStringList DiagnosticsBody::tabNames(const Diagnostics& d, DiagnosticsKind kind) {
        QStringList names;
        for (const DiagnosticTab& t : d.tabs) names << fromStd(t.name);
        if (!names.isEmpty()) return names;
        switch (kind) {
            case DiagnosticsKind::Sim:
                return {QStringLiteral("Raw spectrum"), QStringLiteral("Separated bands"),
                        QStringLiteral("Wiener-filtered bands"), QStringLiteral("Result spectrum")};
            case DiagnosticsKind::Deconvolve: return {QStringLiteral("Convergence")};
            case DiagnosticsKind::Contrast: return {QStringLiteral("Histograms")};
            case DiagnosticsKind::Segment: return {QStringLiteral("Cleanup")};
            case DiagnosticsKind::Volume: return {QStringLiteral("Rendering")};
            case DiagnosticsKind::Alignment: return {QStringLiteral("Alignment")};
            case DiagnosticsKind::Generic: break;
        }
        return {QStringLiteral("Preview")};
    }

    void DiagnosticsBody::setDiagnostics(const Diagnostics& d, DiagnosticsKind kind, int tab, const Context& ctx) {
        clearGrid();
        auto imageOr = [&](std::size_t index, const QString& placeholder) {
            auto* view = new ImageCellView;
            if (index < d.images.size()) view->setImage(d.images[index]);
            else view->clear(placeholder);
            return view;
        };
        auto imageTitle = [&](std::size_t index, const QString& fallback) {
            return index < d.images.size() ? fromStd(d.images[index].title) : fallback;
        };
        auto imageMeta = [&](std::size_t index) {
            return index < d.images.size() ? fromStd(d.images[index].meta) : QString();
        };
        // images of the active tab (all images without tabs)
        std::vector<std::size_t> tabImages;
        if (!d.tabs.empty()) {
            const std::size_t t = static_cast<std::size_t>(std::clamp(tab, 0, static_cast<int>(d.tabs.size()) - 1));
            for (int i : d.tabs[t].images)
                if (i >= 0 && static_cast<std::size_t>(i) < d.images.size()) tabImages.push_back(static_cast<std::size_t>(i));
        } else {
            for (std::size_t i = 0; i < d.images.size(); ++i) tabImages.push_back(i);
        }
        int col = 0;
        switch (kind) {
            case DiagnosticsKind::Sim: {
                static const char* kPlaceholders[4][3] = {
                    {"Raw FFT · phase 1", "Raw FFT · phase 2", "Raw FFT · phase 3"},
                    {"Order 1 · angle 1", "Order 1 · angle 2", "Order 1 · angle 3"},
                    {"Filtered order 1 · angle 1", "Filtered order 1 · angle 2", "Filtered order 1 · angle 3"},
                    {"Widefield", "SIM result", "Difference"},
                };
                const int t = std::clamp(tab, 0, 3);
                for (int i = 0; i < 3; ++i) {
                    auto* view = new ImageCellView;
                    QString title = QString::fromUtf8(kPlaceholders[t][i]), meta;
                    if (static_cast<std::size_t>(i) < tabImages.size()) {
                        const DiagnosticImage& img = d.images[tabImages[static_cast<std::size_t>(i)]];
                        view->setImage(img);
                        title = fromStd(img.title);
                        meta = fromStd(img.meta);
                    } else {
                        view->clear(QStringLiteral("Run the step to see the spectrum"));
                    }
                    addCell(title, meta, view, col++, 1);
                }
                auto* column = new QWidget;
                auto* v = new QVBoxLayout(column);
                v->setContentsMargins(0, 0, 0, 0);
                v->setSpacing(6);
                if (d.table) {
                    auto* table = new DiagnosticTableView;
                    table->setTable(*d.table);
                    v->addWidget(table, 1);
                } else {
                    auto* empty = new QLabel(QStringLiteral("Run the step to fit the pattern vectors and modulation depths."));
                    empty->setWordWrap(true);
                    empty->setFont(theme::font(theme::kSmallPx));
                    empty->setContentsMargins(12, 0, 12, 0);
                    v->addWidget(empty, 1, Qt::AlignTop);
                }
                if (!d.footer.empty()) {
                    auto* footer = new QLabel(fromStd(d.footer));
                    footer->setWordWrap(true);
                    footer->setFont(theme::font(theme::kSmallPx));
                    QPalette fp = footer->palette();
                    fp.setColor(QPalette::WindowText, theme::kNeutral600);
                    footer->setPalette(fp);
                    footer->setContentsMargins(12, 0, 12, 8);
                    v->addWidget(footer, 0);
                }
                addCell(d.table ? fromStd(d.table->caption) : QStringLiteral("Estimated parameters"), {}, column,
                        col++, 0, 300);
                break;
            }
            case DiagnosticsKind::Deconvolve: {
                auto* curve = new CurveView;
                QString title = QStringLiteral("Convergence · relative change per iteration");
                if (!d.curves.empty()) {
                    curve->setCurve(d.curves.front());
                    if (!d.curves.front().title.empty()) title = fromStd(d.curves.front().title);
                }
                addCell(title, {}, curve, col++, 1);
                addCell(imageTitle(0, QStringLiteral("PSF · XZ")), imageMeta(0), imageOr(0, QStringLiteral("PSF")), col++, 0, 260);
                addCell(imageTitle(1, QStringLiteral("Residual")), imageMeta(1), imageOr(1, QStringLiteral("Residual after the run")),
                        col++, 0, 260);
                break;
            }
            case DiagnosticsKind::Contrast: {
                if (d.histograms.empty()) {
                    auto* h = new HistogramView;
                    h->clear();
                    addCell(QStringLiteral("Histograms"), {}, h, col++, 1);
                } else {
                    for (const DiagnosticHistogram& h : d.histograms) {
                        auto* view = new HistogramView;
                        view->setHistogram(h);
                        addCell(fromStd(h.channel), {}, view, col++, 1);
                    }
                }
                break;
            }
            case DiagnosticsKind::Alignment: {
                addCell(imageTitle(0, QStringLiteral("Checkerboard · fixed ⇄ moving")), imageMeta(0),
                        imageOr(0, QStringLiteral("Run the step to compare fixed and moving")), col++, 1);
                auto* tiles = new TileMapView;
                QString tileTitle = QStringLiteral("Tile layout");
                if (d.alignment) {
                    tiles->setAlignment(*d.alignment);
                } else {
                    AlignmentInfo none;
                    none.gridRows = 1;
                    none.gridCols = 1;
                    none.tileNames = {"–"};
                    tiles->setAlignment(none);
                }
                addCell(tileTitle, {}, tiles, col++, 1);
                auto* stats = new FactsView;
                stats->setFacts(d.alignment ? d.alignment->shiftStats : d.facts, {},
                                d.alignment ? QString() : QStringLiteral("Pairwise shifts appear after the run."));
                addCell(QStringLiteral("Pairwise shifts"), {}, stats, col++, 0, 300);
                break;
            }
            case DiagnosticsKind::Volume: {
                auto* curve = new CurveView;
                if (!d.curves.empty()) curve->setCurve(d.curves.front());
                addCell(QStringLiteral("Transfer function"), {}, curve, col++, 1);
                auto* facts = new FactsView;
                facts->setFacts(d.facts, {}, {});
                addCell(QStringLiteral("Reconstruction"), {}, facts, col++, 1);
                addCell(imageTitle(0, QStringLiteral("Isosurface preview")), imageMeta(0),
                        imageOr(0, QStringLiteral("Preview after the run")), col++, 1);
                break;
            }
            case DiagnosticsKind::Segment:
            case DiagnosticsKind::Generic: {
                const std::size_t in = tabImages.size() > 0 ? tabImages[0] : d.images.size();
                const std::size_t out = tabImages.size() > 1 ? tabImages[1] : d.images.size();
                addCell(imageTitle(in, QStringLiteral("Input")), in < d.images.size() ? imageMeta(in) : ctx.inputShape,
                        imageOr(in, QStringLiteral("Input preview")), col++, 1);
                addCell(imageTitle(out, QStringLiteral("Output · live")), out < d.images.size() ? imageMeta(out) : ctx.outputShape,
                        imageOr(out, QStringLiteral("Output preview after the run")), col++, 1);
                auto* facts = new FactsView;
                facts->setFacts(d.facts, d.summary.empty() ? ctx.stepSummary : fromStd(d.summary), ctx.estimate);
                addCell(QStringLiteral("Step summary"), {}, facts, col++, 1);
                break;
            }
        }
        // extra curves / histograms of a generic diagnostics land in more cells
        if (kind == DiagnosticsKind::Generic) {
            for (std::size_t i = 0; i < d.curves.size(); ++i) {
                auto* curve = new CurveView;
                curve->setCurve(d.curves[i]);
                addCell(fromStd(d.curves[i].title), {}, curve, col++, 1);
            }
            for (const DiagnosticHistogram& h : d.histograms) {
                auto* view = new HistogramView;
                view->setHistogram(h);
                addCell(fromStd(h.channel), {}, view, col++, 1);
            }
        }
    }

} // namespace sirius::app
