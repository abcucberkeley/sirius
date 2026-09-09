#include "qt/panels/diagnostics_panel.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <array>
#include <cmath>

#include <QAbstractButton>
#include <QApplication>
#include <QEvent>
#include <QFontMetrics>
#include <QFrame>
#include <QGridLayout>
#include <QCheckBox>
#include <QDockWidget>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QPainter>
#include <QPushButton>
#include <QSlider>
#include <QStackedWidget>
#include <QAbstractTableModel>
#include <QIcon>
#include <QMouseEvent>
#include <QPixmap>
#include <QStyledItemDelegate>
#include <QTableView>
#include <QTableWidget>
#include <QResizeEvent>
#include <QMenu>
#include <QTimer>
#include <QVBoxLayout>

#include "core/ops/builtin.hpp"
#include "qt/panels/diagnostic_cells.hpp"
#include "qt/qt_strings.hpp"
#include "qt/theme.hpp"
#include "qt/trace.hpp"
#include "qt/widgets/controls.hpp"

namespace sirius::app {

    using widgets::GlyphButton;
    using widgets::Icon;

    namespace {

        // A checkable icon square with a tool tip -- the header chrome and
        // the cleanup tool grid.
        GlyphButton* iconButton(Icon icon, const QString& tip, QSize size, QWidget* parent) {
            auto* b = new GlyphButton(icon, size, parent);
            b->setToolTip(tip);
            b->setAccessibleName(tip);
            b->setCheckable(true);
            return b;
        }

        // The design's 2 px accent focus ring, on the widgets that paint
        // themselves; theme.cpp's filter stamps "focusVisible" for keyboard
        // focus only.
        void drawFocusRing(QPainter& p, const QRectF& box) {
            p.setPen(QPen(theme::kAccent, 2));
            p.setBrush(Qt::NoBrush);
            p.drawRect(box);
        }

        bool focusRingVisible(const QWidget* w) {
            return w->hasFocus() && w->property("focusVisible").toBool();
        }

        // Tab of the diagnostics header: 12 px text, 2 px accent underline + 800 when active.
        class TabButton : public QAbstractButton {
        public:
            explicit TabButton(const QString& text, QWidget* parent = nullptr) : QAbstractButton(parent) {
                setText(text);
                setCheckable(true);
                setCursor(Qt::PointingHandCursor);
                setFocusPolicy(Qt::StrongFocus);
                setAccessibleName(text);
                setAccessibleDescription(QStringLiteral("Diagnostics tab"));
                const int w = QFontMetrics(theme::heading(12)).horizontalAdvance(text) + 20;
                setFixedSize(w, 26);
            }

        protected:
            void paintEvent(QPaintEvent*) override {
                QPainter p(this);
                const bool on = isChecked();
                p.setFont(on ? theme::heading(12) : theme::font(12));
                p.setPen(underMouse() && !on ? theme::kAccent : theme::kText);
                p.drawText(rect().adjusted(0, 0, 0, -2), Qt::AlignCenter, text());
                if (on) p.fillRect(QRect(0, height() - 2, width(), 2), theme::kAccent);
                if (focusRingVisible(this)) drawFocusRing(p, QRectF(rect()).adjusted(1.0, 1.0, -1.0, -1.0));
            }
            void keyPressEvent(QKeyEvent* e) override {
                if (e->key() == Qt::Key_Return || e->key() == Qt::Key_Enter) {
                    click();
                    return;
                }
                QAbstractButton::keyPressEvent(e);
            }
            void enterEvent(QEnterEvent*) override { update(); }
            void leaveEvent(QEvent*) override { update(); }
        };

        QLabel* caption(const QString& text, QWidget* parent) {
            auto* l = new QLabel(text, parent);
            l->setFont(theme::caption());
            QPalette pal = l->palette();
            pal.setColor(QPalette::WindowText, theme::kNeutral600);
            l->setPalette(pal);
            return l;
        }

        QString groupThousands(Index n) {
            QString s = QString::number(n);
            for (int i = s.size() - 3; i > 0; i -= 3) s.insert(i, QChar(0x2009));   // thin space
            return s;
        }

        QPushButton* styledButton(const QString& text, bool secondary, QWidget* parent) {
            auto* b = new QPushButton(text, parent);
            b->setCursor(Qt::PointingHandCursor);
            // Keyboard-reachable, with the style sheet's focus ring.
            b->setFocusPolicy(Qt::StrongFocus);
            b->setAccessibleName(text);
            widgets::setButtonClass(b, secondary ? "secondary tiny" : "ghost tiny");
            return b;
        }

        struct SegTool {
            PaintTool tool;
            Icon icon;
            const char* name;
        };
        constexpr std::array<SegTool, 8> kSegTools{{
            {PaintTool::Brush, Icon::Brush, "Brush"},
            {PaintTool::Erase, Icon::Erase, "Erase"},
            {PaintTool::Fill, Icon::Fill, "Fill region"},
            {PaintTool::Pick, Icon::Pick, "Pick label"},
            {PaintTool::Merge, Icon::Merge, "Merge labels"},
            {PaintTool::Split, Icon::Split, "Split (watershed seed)"},
            {PaintTool::Delete, Icon::Trash, "Delete label"},
            {PaintTool::Lasso, Icon::Lasso, "Lasso"},
        }};

        // --- label table: a model over the label statistics ------------------------
        // Tens of thousands of labels are common; a QTableWidget with items and
        // a link widget per row took 20 s to rebuild after every stroke. The
        // model answers per cell from the stats vector, so a refresh is a reset.
        class LabelTableModel : public QAbstractTableModel {
        public:
            explicit LabelTableModel(QObject* parent = nullptr) : QAbstractTableModel(parent) {}

            void setLabels(std::shared_ptr<LabelVolume> labels) {
                beginResetModel();
                labels_ = std::move(labels);
                endResetModel();
            }
            const std::vector<LabelStats>& stats() const {
                static const std::vector<LabelStats> none;
                return labels_ ? labels_->stats() : none;
            }
            std::uint32_t idAt(int row) const {
                const auto& st = stats();
                return row >= 0 && row < static_cast<int>(st.size()) ? st[static_cast<std::size_t>(row)].id : 0u;
            }
            int rowOf(std::uint32_t id) const {
                const auto& st = stats();
                for (std::size_t i = 0; i < st.size(); ++i)
                    if (st[i].id == id) return static_cast<int>(i);
                return -1;
            }

            int rowCount(const QModelIndex& parent = QModelIndex()) const override {
                return parent.isValid() ? 0 : static_cast<int>(stats().size());
            }
            int columnCount(const QModelIndex& parent = QModelIndex()) const override { return parent.isValid() ? 0 : 6; }

            QVariant headerData(int section, Qt::Orientation o, int role) const override {
                if (o != Qt::Horizontal || role != Qt::DisplayRole) return {};
                static const char* names[] = {"ID", "CLASS", "VOXELS", "CONF.", "FLAG", ""};
                return section >= 0 && section < 6 ? QString::fromUtf8(names[section]) : QString();
            }

            QVariant data(const QModelIndex& index, int role) const override {
                const auto& st = stats();
                if (!index.isValid() || index.row() < 0 || index.row() >= static_cast<int>(st.size())) return {};
                const LabelStats& s = st[static_cast<std::size_t>(index.row())];
                switch (role) {
                    case Qt::DisplayRole:
                        switch (index.column()) {
                            case 0: return QStringLiteral("%1").arg(s.id, 4, 10, QChar('0'));
                            case 1: return fromStd(s.cls);
                            case 2: return groupThousands(s.voxels);
                            case 3: return QString::number(s.confidence, 'f', 2);
                            case 4: return fromStd(s.flagText());
                            default: return {};
                        }
                    case Qt::DecorationRole:
                        return index.column() == 0 ? QVariant(chip(s.id)) : QVariant();
                    case Qt::ForegroundRole:
                        if ((index.column() == 3 && s.confidence < 0.6) || (index.column() == 4 && !s.flags.empty())) return QBrush(theme::kAccentText);
                        return {};
                    case Qt::FontRole:
                        if (index.column() == 4 && !s.flags.empty()) return theme::heading(theme::kSmallPx);
                        return {};
                    case Qt::UserRole:
                        return static_cast<uint>(s.id);
                    default:
                        return {};
                }
            }

        private:
            // one chip per palette colour, shared by every label of that colour
            const QPixmap& chip(std::uint32_t id) const {
                const auto c = labelColor(id);
                const QRgb key = QColor::fromRgbF(c[0], c[1], c[2]).rgb();
                auto it = chips_.find(key);
                if (it == chips_.end()) {
                    // Allocated at the screen's ratio, like widgets::iconPixmap:
                    // a 10 x 10 device pixmap is blurry at 1.25x / 1.5x / 2x.
                    const qreal dpr = qApp ? qApp->devicePixelRatio() : 1.0;
                    QPixmap pm(qRound(10 * dpr), qRound(10 * dpr));
                    pm.setDevicePixelRatio(dpr);
                    pm.fill(QColor(key));
                    it = chips_.emplace(key, pm).first;
                }
                return it->second;
            }

            std::shared_ptr<LabelVolume> labels_;
            mutable std::map<QRgb, QPixmap> chips_;
        };

        // The last column: "merge · split ·" and a bin, painted in accent; a
        // click on a word (or the bin) calls back with the link and the row's label.
        class LabelActionDelegate : public QStyledItemDelegate {
        public:
            explicit LabelActionDelegate(QObject* parent = nullptr) : QStyledItemDelegate(parent) {}
            std::function<void(const QString&, std::uint32_t)> onAction;

            void paint(QPainter* p, const QStyleOptionViewItem& option, const QModelIndex& index) const override {
                // Background and selection only: the model has no display text
                // or decoration for this column, so the base class draws just
                // those. It must be given the real index -- a debug Qt asserts
                // on an invalid one (QStyledItemDelegate::paint) and aborts the
                // application the moment the label table has a row to draw.
                QStyledItemDelegate::paint(p, option, index);
                p->save();
                p->setFont(theme::font(theme::kSmallPx));
                p->setPen(theme::kAccentText);
                const QRect r = option.rect.adjusted(6, 0, -6, 0);
                p->drawText(r, Qt::AlignLeft | Qt::AlignVCenter, text());
                const int x = r.left() + QFontMetrics(theme::font(theme::kSmallPx)).horizontalAdvance(text());
                widgets::drawIcon(*p, QRectF(x, r.center().y() - 5.5, 11, 11), Icon::Trash, theme::kAccentText, 1.25);
                p->restore();
            }
            QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex&) const override {
                return {QFontMetrics(theme::font(theme::kSmallPx)).horizontalAdvance(text()) + 23, option.rect.height()};
            }
            bool editorEvent(QEvent* e, QAbstractItemModel* model, const QStyleOptionViewItem& option, const QModelIndex& index) override {
                if (e->type() != QEvent::MouseButtonRelease || !onAction) return false;
                auto* me = static_cast<QMouseEvent*>(e);
                if (me->button() != Qt::LeftButton) return false;
                const QFontMetrics fm(theme::font(theme::kSmallPx));
                const int x = static_cast<int>(me->position().x()) - option.rect.x() - 6;
                const int wMerge = fm.horizontalAdvance(QStringLiteral("merge")), wSep = fm.horizontalAdvance(QStringLiteral(" · ")),
                          wSplit = fm.horizontalAdvance(QStringLiteral("split"));
                QString link;
                if (x >= 0 && x <= wMerge) link = QStringLiteral("merge");
                else if (x >= wMerge + wSep && x <= wMerge + wSep + wSplit) link = QStringLiteral("split");
                else if (x > wMerge + 2 * wSep + wSplit - 2) link = QStringLiteral("delete");
                if (link.isEmpty()) return false;
                onAction(link, model->data(index, Qt::UserRole).toUInt());
                return true;
            }

        private:
            static QString text() { return QStringLiteral("merge · split · "); }
        };

        class LabelTableView : public QTableView {
        public:
            explicit LabelTableView(QWidget* parent = nullptr) : QTableView(parent) {
                setFrameShape(QFrame::NoFrame);
                setShowGrid(false);
                setEditTriggers(QAbstractItemView::NoEditTriggers);
                setFocusPolicy(Qt::StrongFocus);
                setAccessibleName(QStringLiteral("Labels"));
                setAccessibleDescription(QStringLiteral("One row per label: id, class, voxels, confidence, flag"));
                verticalHeader()->setVisible(false);
                verticalHeader()->setDefaultSectionSize(22);
                horizontalHeader()->setStretchLastSection(true);
                horizontalHeader()->setHighlightSections(false);
                horizontalHeader()->setDefaultAlignment(Qt::AlignLeft | Qt::AlignVCenter);
                setFont(theme::tabular(theme::font(theme::kSmallPx)));
                widgets::setWidgetClass(this, "dense");
            }
        };

        // --- segmentation cleanup ---------------------------------------------------

        class SegmentCleanupView : public QWidget {
        public:
            explicit SegmentCleanupView(WorkbenchBridge& bridge, QWidget* parent = nullptr)
                : QWidget(parent), bridge_(bridge) {
                setAutoFillBackground(true);
                QPalette pal = palette();
                pal.setColor(QPalette::Window, theme::kDivider);
                setPalette(pal);
                auto* row = new QHBoxLayout(this);
                row->setContentsMargins(0, 0, 0, 0);
                row->setSpacing(2);
                row->addWidget(buildTools(), 0);
                row->addWidget(buildTable(), 1);
                row->addWidget(buildQueue(), 0);
            }

            // After a view-state change: the selection and the tool state,
            // without resetting the table (which drops a multi-row selection
            // and the scroll position on every pan).
            void refreshView() {
                viewOnly_ = true;
                refresh();
                viewOnly_ = false;
            }

            void refresh() {
                const ViewState& vs = bridge_.wb().viewState();
                // Label edits are refused while a run is active, painting
                // included: the tools say so instead of doing nothing.
                const bool editable = bridge_.wb().canEdit();
                const QString frozen = QStringLiteral("Not while a run is in progress — cancel it (Esc) or wait");
                for (GlyphButton* b : tools_) b->setEnabled(editable);
                brush_->setEnabled(editable);
                paint3d_->setEnabled(editable);
                for (QPushButton* b : {next_, undo_, accept_})
                    if (b) {
                        b->setEnabled(editable);
                        b->setToolTip(editable ? QString() : frozen);
                    }
                updating_ = true;
                for (std::size_t i = 0; i < tools_.size(); ++i) tools_[i]->setChecked(kSegTools[i].tool == vs.paintTool);
                brush_->setValue(vs.brushPx);
                brushLabel_->setText(QStringLiteral("%1 px").arg(vs.brushPx));
                toolName_->setText(QString::fromUtf8(kSegTools[toolIndex(vs.paintTool)].name));
                paint3d_->setChecked(vs.paint3d);
                paint3d_->setText(QStringLiteral("Paint in 3D (±%1 z)").arg(std::max(1, static_cast<int>(std::lround(vs.brushPx / 6.0)))));
                updating_ = false;
                refreshTable();
            }

        private:
            static std::size_t toolIndex(PaintTool t) {
                for (std::size_t i = 0; i < kSegTools.size(); ++i)
                    if (kSegTools[i].tool == t) return i;
                return 0;
            }

            QWidget* cell(QWidget* content = nullptr) {
                auto* w = content ? content : new QWidget(this);
                w->setAutoFillBackground(true);
                QPalette pal = w->palette();
                pal.setColor(QPalette::Window, theme::kBg);
                w->setPalette(pal);
                return w;
            }

            QWidget* buildTools() {
                QWidget* w = cell();
                // 230 px was narrower than the caption, which then clipped to
                // "CLEANUP TOOLS · PAINT IN V": the cell takes whichever is
                // wider.
                const QString cap = QStringLiteral("CLEANUP TOOLS · PAINT IN VIEWER");
                w->setFixedWidth(std::max(230, QFontMetrics(theme::caption()).horizontalAdvance(cap) + 30));
                auto* v = new QVBoxLayout(w);
                v->setContentsMargins(14, 10, 14, 10);
                v->setSpacing(8);
                v->addWidget(caption(cap, w));
                auto* gridHost = new QWidget(w);
                auto* grid = new QGridLayout(gridHost);
                grid->setContentsMargins(0, 0, 0, 0);
                grid->setSpacing(2);
                for (std::size_t i = 0; i < kSegTools.size(); ++i) {
                    auto* b = iconButton(kSegTools[i].icon, QString::fromUtf8(kSegTools[i].name), QSize(48, 34), gridHost);
                    b->setIconPx(16);
                    b->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
                    b->setMinimumWidth(30);
                    b->setMaximumWidth(1000);
                    grid->addWidget(b, static_cast<int>(i / 4), static_cast<int>(i % 4));
                    QObject::connect(b, &QAbstractButton::clicked, this, [this, i] {
                        if (updating_) return;
                        bridge_.wb().setPaintTool(kSegTools[i].tool);
                        bridge_.wb().setTool(ViewerTool::Paint);
                    });
                    tools_.push_back(b);
                }
                v->addWidget(gridHost);
                auto* nameRow = new QHBoxLayout;
                toolName_ = new QLabel(w);
                toolName_->setFont(theme::font(theme::kSmallPx));
                QPalette np = toolName_->palette();
                np.setColor(QPalette::WindowText, theme::kNeutral600);
                toolName_->setPalette(np);
                brushLabel_ = new QLabel(w);
                brushLabel_->setFont(theme::font(theme::kSmallPx));
                nameRow->addWidget(toolName_);
                nameRow->addStretch(1);
                nameRow->addWidget(brushLabel_);
                v->addLayout(nameRow);
                brush_ = new QSlider(Qt::Horizontal, w);
                brush_->setRange(2, 60);
                QObject::connect(brush_, &QSlider::valueChanged, this, [this](int value) {
                    if (updating_) return;
                    ViewState vs = bridge_.wb().viewState();
                    vs.brushPx = value;
                    bridge_.wb().setViewState(vs);
                });
                v->addWidget(brush_);
                paint3d_ = new QCheckBox(QStringLiteral("Paint in 3D"), w);
                paint3d_->setFont(theme::font(theme::kSmallPx));
                QObject::connect(paint3d_, &QCheckBox::toggled, this, [this](bool on) {
                    if (updating_) return;
                    ViewState vs = bridge_.wb().viewState();
                    vs.paint3d = on;
                    bridge_.wb().setViewState(vs);
                });
                v->addWidget(paint3d_);
                v->addStretch(1);
                return w;
            }

            QWidget* buildTable() {
                table_ = new LabelTableView(this);
                cell(table_);
                model_ = new LabelTableModel(table_);
                table_->setModel(model_);
                table_->setSelectionMode(QAbstractItemView::ExtendedSelection);
                table_->setSelectionBehavior(QAbstractItemView::SelectRows);
                auto* actions = new LabelActionDelegate(table_);
                actions->onAction = [this](const QString& link, std::uint32_t id) { act(link, id); };
                table_->setItemDelegateForColumn(5, actions);
                // fixed widths: ResizeToContents would measure every row
                const QFontMetrics fm(theme::font(theme::kSmallPx));
                table_->horizontalHeader()->setSectionResizeMode(QHeaderView::Interactive);
                table_->setColumnWidth(0, fm.horizontalAdvance(QStringLiteral("00000")) + 46);
                table_->setColumnWidth(1, fm.horizontalAdvance(QStringLiteral("nucleus")) + 28);
                table_->setColumnWidth(2, fm.horizontalAdvance(QStringLiteral("1 000 000")) + 20);
                table_->setColumnWidth(3, fm.horizontalAdvance(QStringLiteral("0.00")) + 30);
                table_->setColumnWidth(4, fm.horizontalAdvance(QStringLiteral("touching border")) + 32);
                table_->horizontalHeader()->setStretchLastSection(true);
                QObject::connect(table_->selectionModel(), &QItemSelectionModel::selectionChanged, this, [this] {
                    if (updating_) return;
                    const auto rows = table_->selectionModel()->selectedRows();
                    if (rows.isEmpty()) return;
                    const std::uint32_t id = model_->idAt(rows.first().row());
                    ViewState vs = bridge_.wb().viewState();
                    if (vs.selectedLabel == id) return;
                    vs.selectedLabel = id;
                    bridge_.wb().setViewState(vs);
                });
                return table_;
            }

            QWidget* buildQueue() {
                QWidget* w = cell();
                w->setFixedWidth(280);
                auto* v = new QVBoxLayout(w);
                v->setContentsMargins(14, 10, 14, 10);
                v->setSpacing(6);
                v->addWidget(caption(QStringLiteral("REVIEW QUEUE"), w));
                queue_ = new FactsView(w);
                v->addWidget(queue_, 1);
                // Two rows: the three labels never fitted across 280 px, and
                // "Accept all reviewed" was the one that lost its ending.
                auto* buttons = new QGridLayout;
                buttons->setHorizontalSpacing(6);
                buttons->setVerticalSpacing(6);
                next_ = styledButton(QStringLiteral("Next flagged →"), true, w);
                QObject::connect(next_, &QPushButton::clicked, this, [this] { bridge_.wb().nextFlaggedLabel(true); });
                undo_ = styledButton(QStringLiteral("Undo"), false, w);
                QObject::connect(undo_, &QPushButton::clicked, this, [this] { bridge_.wb().undo(); });
                accept_ = styledButton(QStringLiteral("Accept all reviewed"), false, w);
                QObject::connect(accept_, &QPushButton::clicked, this, [this] { bridge_.wb().acceptAllReviewed(); });
                buttons->addWidget(next_, 0, 0);
                buttons->addWidget(undo_, 0, 1);
                buttons->addWidget(accept_, 1, 0, 1, 2, Qt::AlignLeft);
                buttons->setColumnStretch(2, 1);
                v->addLayout(buttons);
                return w;
            }

            void refreshTable() {
                updating_ = true;
                std::shared_ptr<LabelVolume> labels = bridge_.wb().viewedLabels();
                const ViewState& vs = bridge_.wb().viewState();
                std::vector<DiagnosticFact> facts;
                if (!labels || labels->stats().empty()) {
                    if (model_->rowCount() > 0 || labels != shownLabels_) model_->setLabels(nullptr);
                    shownLabels_ = nullptr;
                    facts.push_back({"Labels", "none yet"});
                    queue_->setFacts(facts, {}, QStringLiteral("Run the segmentation step to fill the review queue."));
                    updating_ = false;
                    return;
                }
                // the model reads the stats in place; a reset is all a change
                // needs -- and a view-state change (pan, wheel, crosshair)
                // needs none, only the selection synced below
                if (!viewOnly_ || labels != shownLabels_) model_->setLabels(labels);
                shownLabels_ = labels;
                const int row = vs.selectedLabel ? model_->rowOf(vs.selectedLabel) : -1;
                if (row >= 0) {
                    table_->selectRow(row);
                    table_->scrollTo(model_->index(row, 0), QAbstractItemView::EnsureVisible);
                } else {
                    table_->clearSelection();
                }
                // review queue
                facts.push_back({"Low confidence (< 0.6)", std::to_string(labels->flaggedCount("low conf"))});
                facts.push_back({"Touching border", std::to_string(labels->flaggedCount("touching border"))});
                facts.push_back({"Size outliers", std::to_string(labels->flaggedCount("small") + labels->flaggedCount("merged?"))});
                facts.push_back({"Reviewed", std::to_string(labels->reviewedCount()) + " / " + std::to_string(labels->stats().size())});
                queue_->setFacts(facts);
                updating_ = false;
            }

            void act(const QString& link, std::uint32_t id) {
                Workbench& wb = bridge_.wb();
                if (!wb.canEdit()) {
                    wb.logLine("Label edits are refused while a run is in progress.");
                    return;
                }
                if (link == QLatin1String("delete")) {
                    wb.deleteLabel(id);
                } else if (link == QLatin1String("merge")) {
                    std::vector<std::uint32_t> ids{id};
                    for (const QModelIndex& row : table_->selectionModel()->selectedRows()) {
                        const std::uint32_t other = model_->idAt(row.row());
                        if (other != id) ids.push_back(other);
                    }
                    if (ids.size() < 2 && wb.viewState().selectedLabel != 0 && wb.viewState().selectedLabel != id)
                        ids.push_back(wb.viewState().selectedLabel);
                    if (ids.size() >= 2) wb.mergeLabels(ids);
                    else bridge_.wb().logLine("Merge: select another label row first.");
                } else if (link == QLatin1String("split")) {
                    std::shared_ptr<LabelVolume> labels = wb.viewedLabels();
                    const LabelStats* s = labels ? labels->statsOf(id) : nullptr;
                    if (!s) return;
                    // two seeds at a quarter and three quarters of the longest bbox axis
                    const std::array<Index, 3> extent{s->bbox[1] - s->bbox[0], s->bbox[3] - s->bbox[2], s->bbox[5] - s->bbox[4]};
                    const int axis = static_cast<int>(std::max_element(extent.begin(), extent.end()) - extent.begin());
                    std::array<Index, 3> centre{(s->bbox[0] + s->bbox[1]) / 2, (s->bbox[2] + s->bbox[3]) / 2, (s->bbox[4] + s->bbox[5]) / 2};
                    std::array<Index, 3> a = centre, b = centre;
                    const std::size_t ax = static_cast<std::size_t>(axis);
                    a[ax] = s->bbox[2 * ax] + extent[ax] / 4;
                    b[ax] = s->bbox[2 * ax] + (3 * extent[ax]) / 4;
                    wb.splitLabel(id, a, b);
                }
            }

            WorkbenchBridge& bridge_;
            std::vector<GlyphButton*> tools_;
            QLabel* toolName_ = nullptr;
            QLabel* brushLabel_ = nullptr;
            QSlider* brush_ = nullptr;
            QCheckBox* paint3d_ = nullptr;
            LabelTableView* table_ = nullptr;
            LabelTableModel* model_ = nullptr;
            std::shared_ptr<LabelVolume> shownLabels_;
            bool viewOnly_ = false;   // refreshing for a view-state change: the labels did not change
            FactsView* queue_ = nullptr;
            QPushButton* next_ = nullptr;
            QPushButton* undo_ = nullptr;
            QPushButton* accept_ = nullptr;
            bool updating_ = false;
        };

    } // namespace

    // --- DiagnosticsPanel -----------------------------------------------------------

    struct DiagnosticsPanel::Impl {
        DiagnosticsPanel* self = nullptr;
        WorkbenchBridge& bridge;
        QDockWidget* dock = nullptr;
        bool collapsed = false;
        bool maximized = false;
        int tab = 0;
        int expandedHeight = theme::kDiagnosticsH;

        QWidget* header = nullptr;
        QLabel* toggle = nullptr;
        QLabel* captionLabel = nullptr;
        QWidget* tabRow = nullptr;
        QHBoxLayout* tabLayout = nullptr;
        std::vector<TabButton*> tabs;
        QLabel* hint = nullptr;
        QWidget* leftBox = nullptr;
        GlyphButton* moreBtn = nullptr;
        QString hintText;
        void applyHint() {
            if (hint) hint->setText(widgets::elide(hint, hintText, std::max(20, hint->width())));
        }
        // The header clips rather than widening the window (its size policy
        // is Ignored), so it decides itself what to show when squeezed: tabs
        // that do not fit are dropped from the right, the hint goes first.
        void layoutHeader() {
            if (!header || !tabRow || !leftBox) return;
            // the panel's width is current inside its resizeEvent; the
            // header's is only updated by the layout afterwards
            const int width = self->width();
            constexpr int margins = 28, spacing = 16, buttons = 3 * 24 + 2 * 2;
            const int avail = width - margins - leftBox->sizeHint().width() - spacing - buttons - spacing;
            // Show the tabs in order while they fit, always including the
            // active one; the rest go behind the "…" button.
            int total = 0;
            for (std::size_t i = 0; i < tabs.size(); ++i) total += tabs[i]->minimumWidth() + (i ? 2 : 0);
            const bool overflow = total > avail;
            const int room = overflow ? avail - 26 - 2 : avail;   // the "…" button's share
            int used = 0;
            bool any = false, activeShown = false;
            const std::size_t active = static_cast<std::size_t>(std::max(tab, 0));
            for (std::size_t i = 0; i < tabs.size(); ++i) {
                const int w = tabs[i]->minimumWidth() + (any ? 2 : 0);   // fixed-size buttons: sizeHint() is unset
                const bool isActive = i == active;
                bool fits = !collapsed && tabs.size() > 1 && used + w <= room;
                if (fits && isActive) activeShown = true;
                // reserve the active tab's width so it is never the one dropped
                if (fits && !isActive && !activeShown && active < tabs.size() && i < active &&
                    used + w + tabs[active]->minimumWidth() + 2 > room)
                    fits = false;
                tabs[i]->setVisible(fits);
                if (fits) {
                    used += w;
                    any = true;
                }
            }
            tabRow->setVisible(any);
            bool hidden = false;
            for (TabButton* t : tabs) hidden = hidden || (!t->isVisible() && !collapsed && tabs.size() > 1);
            moreBtn->setVisible(hidden);
            const int rest = avail - used - (any ? spacing : 0) - (hidden ? 28 : 0);
            hint->setVisible(rest >= 80);
            applyHint();
        }
        GlyphButton* dockBtn = nullptr;
        GlyphButton* floatBtn = nullptr;
        GlyphButton* maxBtn = nullptr;
        QStackedWidget* stack = nullptr;
        DiagnosticsBody* body = nullptr;
        SegmentCleanupView* segment = nullptr;
        QTimer refreshTimer;
        QFrame* rule = nullptr;

        explicit Impl(WorkbenchBridge& b) : bridge(b) {}

        void buildHeader() {
            header = new QWidget(self);
            header->setFixedHeight(theme::kDiagnosticsHeaderH);
            auto* h = new QHBoxLayout(header);
            h->setContentsMargins(14, 0, 14, 0);
            h->setSpacing(16);
            auto* left = new QWidget(header);
            leftBox = left;
            left->setCursor(Qt::PointingHandCursor);
            auto* lh = new QHBoxLayout(left);
            lh->setContentsMargins(0, 0, 0, 0);
            lh->setSpacing(8);
            toggle = new QLabel(left);
            toggle->setFixedSize(12, 12);
            toggle->setPixmap(widgets::iconPixmap(Icon::ChevronDown, 12, theme::kAccent));
            captionLabel = caption(QStringLiteral("DIAGNOSTICS"), left);
            lh->addWidget(toggle);
            lh->addWidget(captionLabel);
            left->installEventFilter(self);
            h->addWidget(left);
            tabRow = new QWidget(header);
            tabLayout = new QHBoxLayout(tabRow);
            tabLayout->setContentsMargins(0, 0, 0, 0);
            tabLayout->setSpacing(2);
            h->addWidget(tabRow);
            // Tabs that do not fit stay reachable through this menu.
            moreBtn = iconButton(Icon::More, QStringLiteral("More tabs"), QSize(24, 22), header);
            moreBtn->hide();
            QObject::connect(moreBtn, &QAbstractButton::clicked, self, [this] {
                QMenu menu(self);
                for (std::size_t i = 0; i < tabs.size(); ++i) {
                    if (tabs[i]->isVisible()) continue;
                    QAction* a = menu.addAction(tabs[i]->text());
                    QObject::connect(a, &QAction::triggered, self, [this, i] { self->setTab(static_cast<int>(i)); });
                }
                menu.exec(moreBtn->mapToGlobal(QPoint(0, moreBtn->height())));
            });
            h->addWidget(moreBtn);
            // The header never dictates the dock's minimum width: a long tab
            // row or hint clips instead of widening the whole window.
            header->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
            hint = new QLabel(header);
            hint->setFont(theme::font(theme::kSmallPx));
            QPalette hp = hint->palette();
            hp.setColor(QPalette::WindowText, theme::kNeutral600);
            hint->setPalette(hp);
            hint->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
            hint->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
            h->addWidget(hint, 1);
            auto* modes = new QHBoxLayout;
            modes->setSpacing(2);
            dockBtn = iconButton(Icon::Dock, QStringLiteral("Dock to bottom"), QSize(24, 22), header);
            floatBtn = iconButton(Icon::Float, QStringLiteral("Undock as floating window"), QSize(24, 22), header);
            maxBtn = iconButton(Icon::Maximize, QStringLiteral("Maximize over viewer"), QSize(24, 22), header);
            for (GlyphButton* b : {dockBtn, floatBtn, maxBtn}) {
                b->setDimmed(true);
                modes->addWidget(b);
            }
            h->addLayout(modes);
            QObject::connect(dockBtn, &QAbstractButton::clicked, self, [this] {
                self->setMaximized(false);
                if (dock) dock->setFloating(false);
                self->setCollapsed(false);
                updateModes();
            });
            QObject::connect(floatBtn, &QAbstractButton::clicked, self, [this] {
                self->setMaximized(false);
                self->setCollapsed(false);
                if (dock) dock->setFloating(true);
                updateModes();
            });
            QObject::connect(maxBtn, &QAbstractButton::clicked, self, [this] {
                self->setCollapsed(false);
                if (dock && dock->isFloating()) dock->setFloating(false);
                self->setMaximized(!maximized);
            });
        }

        void updateModes() {
            const bool floating = dock && dock->isFloating();
            dockBtn->setChecked(!floating && !maximized);
            floatBtn->setChecked(floating);
            maxBtn->setChecked(maximized);
            self->update();
        }

        void rebuildTabs(const QStringList& names) {
            // Detach the old buttons now: deleteLater() alone leaves them in
            // the layout until the event loop runs, and several refreshes
            // before the first paint would stack every generation of tabs
            // into the header's minimum width (which the window then adopts).
            for (TabButton* t : tabs) {
                tabLayout->removeWidget(t);
                t->hide();
                t->deleteLater();
            }
            tabs.clear();
            for (int i = 0; i < names.size(); ++i) {
                auto* t = new TabButton(names[i], tabRow);
                t->setChecked(i == tab);
                QObject::connect(t, &QAbstractButton::clicked, self, [this, i] { self->setTab(i); });
                tabLayout->addWidget(t);
                tabs.push_back(t);
            }
            layoutHeader();
        }

        void refresh() {
            const ScopedTrace trace("diagnostics refresh");
            const Workbench& wb = bridge.wb();
            const int sel = wb.selectedIndex();
            const Pipeline& p = wb.pipeline();
            if (sel < 0 || sel >= p.size()) {
                captionLabel->setText(QStringLiteral("DIAGNOSTICS"));
                rebuildTabs({});
                body->setDiagnostics(Diagnostics{}, DiagnosticsKind::Generic, 0, {});
                stack->setCurrentWidget(body);
                return;
            }
            const Step& step = p.at(sel);
            captionLabel->setText(QStringLiteral("DIAGNOSTICS · %1").arg(fromStd(step.name).toUpper()));
            const Operation* op = findOperation(step.kind);
            const DiagnosticsKind kind = op ? op->info().diagnostics : DiagnosticsKind::Generic;
            Diagnostics d;
            try {
                d = wb.selectedDiagnostics();
            } catch (const std::exception&) {
            }
            const QStringList names = DiagnosticsBody::tabNames(d, kind);
            tab = std::clamp(tab, 0, std::max(0, static_cast<int>(names.size()) - 1));
            rebuildTabs(names);
            hintText = collapsed ? QStringLiteral("Click to expand") : QStringLiteral("Updates live as parameters change");
            applyHint();
            if (collapsed) return;
            if (kind == DiagnosticsKind::Segment) {
                segment->refresh();
                stack->setCurrentWidget(segment);
                return;
            }
            DiagnosticsBody::Context ctx;
            try {
                ctx.stepSummary = fromStd(wb.stepSummary(sel));
                ctx.inputShape = fromStd(wb.inputMetaOf(sel).shapeString());
                ctx.outputShape = fromStd(wb.outputMetaOf(sel).shapeString());
                ctx.estimate = QStringLiteral("≈ %1 output · cache %2")
                                   .arg(widgets::bytesText(wb.estimatedBytesOf(sel)), QString::fromUtf8(toString(step.cache)));
            } catch (const std::exception& e) {
                ctx.stepSummary = QString::fromUtf8(e.what());
            }
            body->setDiagnostics(d, kind, tab, ctx);
            stack->setCurrentWidget(body);
        }

        void scheduleRefresh() { refreshTimer.start(); }
    };

    DiagnosticsPanel::DiagnosticsPanel(WorkbenchBridge& bridge, QWidget* parent)
        : QWidget(parent), impl_(std::make_unique<Impl>(bridge)) {
        Impl& d = *impl_;
        d.self = this;
        setAutoFillBackground(true);
        QPalette pal = palette();
        pal.setColor(QPalette::Window, theme::kBg);
        setPalette(pal);
        auto* v = new QVBoxLayout(this);
        v->setContentsMargins(0, 0, 0, 0);
        v->setSpacing(0);
        d.buildHeader();
        v->addWidget(d.header, 0);
        d.rule = new QFrame(this);
        d.rule->setFixedHeight(theme::kHairline);
        d.rule->setAutoFillBackground(true);
        QPalette rp = d.rule->palette();
        rp.setColor(QPalette::Window, theme::kDivider);
        d.rule->setPalette(rp);
        v->addWidget(d.rule, 0);
        d.stack = new QStackedWidget(this);
        d.body = new DiagnosticsBody(d.stack);
        d.segment = new SegmentCleanupView(bridge, d.stack);
        d.stack->addWidget(d.body);
        d.stack->addWidget(d.segment);
        v->addWidget(d.stack, 1);
        setMinimumHeight(theme::kDiagnosticsHeaderH);

        d.refreshTimer.setSingleShot(true);
        d.refreshTimer.setInterval(150);
        connect(&d.refreshTimer, &QTimer::timeout, this, [this] { impl_->refresh(); });
        connect(&bridge, &WorkbenchBridge::selectionChanged, this, [this] { impl_->tab = 0; impl_->refresh(); });
        connect(&bridge, &WorkbenchBridge::outputsChanged, this, [this] { impl_->refresh(); });
        connect(&bridge, &WorkbenchBridge::pipelineChanged, this, [this] { impl_->scheduleRefresh(); });
        connect(&bridge, &WorkbenchBridge::stepChanged, this, [this](int) { impl_->scheduleRefresh(); });
        connect(&bridge, &WorkbenchBridge::datasetChanged, this, [this] { impl_->scheduleRefresh(); });
        connect(&bridge, &WorkbenchBridge::labelsChanged, this, [this](quint64) { impl_->scheduleRefresh(); });
        connect(&bridge, &WorkbenchBridge::viewStateChanged, this, [this] {
            if (impl_->stack->currentWidget() == impl_->segment) impl_->segment->refreshView();
        });
        connect(&bridge, &WorkbenchBridge::runFinished, this, [this](bool, const QString&) { impl_->refresh(); });
        // What the cleanup tools may do depends on the run state.
        connect(&bridge, &WorkbenchBridge::runStateChanged, this, [this] { impl_->refresh(); });
        d.updateModes();
        d.refresh();
    }

    DiagnosticsPanel::~DiagnosticsPanel() = default;

    void DiagnosticsPanel::setDock(QDockWidget* dock) {
        impl_->dock = dock;
        if (dock) {
            connect(dock, &QDockWidget::topLevelChanged, this, [this](bool) {
                impl_->updateModes();
                update();
            });
        }
        impl_->updateModes();
    }

    bool DiagnosticsPanel::isCollapsed() const { return impl_->collapsed; }

    void DiagnosticsPanel::setCollapsed(bool collapsed) {
        Impl& d = *impl_;
        if (d.collapsed == collapsed) return;
        d.collapsed = collapsed;
        d.toggle->setPixmap(widgets::iconPixmap(collapsed ? Icon::ChevronRight : Icon::ChevronDown, 12, theme::kAccent));
        if (collapsed) {
            d.expandedHeight = std::max(height(), theme::kDiagnosticsHeaderH + 60);
            d.stack->hide();
            d.rule->hide();
            d.layoutHeader();
            setMaximumHeight(theme::kDiagnosticsHeaderH);
            setMinimumHeight(theme::kDiagnosticsHeaderH);
        } else {
            setMaximumHeight(QWIDGETSIZE_MAX);
            setMinimumHeight(theme::kDiagnosticsHeaderH + 40);
            d.stack->show();
            d.rule->show();
            resize(width(), d.expandedHeight);
        }
        d.refresh();
        emit collapsedChanged(collapsed);
    }

    void DiagnosticsPanel::setMaximized(bool on) {
        Impl& d = *impl_;
        if (d.maximized == on) return;
        d.maximized = on;
        d.updateModes();
        emit maximizedChanged(on);
    }

    bool DiagnosticsPanel::isMaximized() const { return impl_->maximized; }

    void DiagnosticsPanel::setTab(int index) {
        Impl& d = *impl_;
        d.tab = std::max(0, index);
        for (std::size_t i = 0; i < d.tabs.size(); ++i) d.tabs[i]->setChecked(static_cast<int>(i) == d.tab);
        d.refresh();
    }

    int DiagnosticsPanel::tabCount() const { return static_cast<int>(impl_->tabs.size()); }

    bool DiagnosticsPanel::eventFilter(QObject* watched, QEvent* event) {
        if (event->type() == QEvent::MouseButtonRelease && watched == impl_->toggle->parentWidget()) {
            setCollapsed(!impl_->collapsed);
            return true;
        }
        return QWidget::eventFilter(watched, event);
    }

    void DiagnosticsPanel::resizeEvent(QResizeEvent* event) {
        QWidget::resizeEvent(event);
        impl_->layoutHeader();
    }

    void DiagnosticsPanel::paintEvent(QPaintEvent* event) {
        QWidget::paintEvent(event);
        QPainter p(this);
        // 2 px rule on top (the region divider); an ink border when floating
        p.fillRect(QRect(0, 0, width(), theme::kRule), theme::kDivider);
        if (impl_->dock && impl_->dock->isFloating()) {
            p.setPen(QPen(theme::kText, 2));
            p.drawRect(rect().adjusted(1, 1, -1, -1));
        }
    }

} // namespace sirius::app
