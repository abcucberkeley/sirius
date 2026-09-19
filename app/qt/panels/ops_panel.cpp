#include "qt/panels/ops_panel.hpp"

#include <QApplication>
#include <QBoxLayout>
#include <QCheckBox>
#include <QEvent>
#include <QSettings>
#include <QFileInfo>
#include <QFrame>
#include <QGridLayout>
#include <algorithm>
#include <functional>

#include <QPen>
#include <QEnterEvent>
#include <QLabel>
#include <QShowEvent>
#include <QResizeEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QPushButton>
#include <QScrollArea>
#include <QStyle>
#include <QVector>

#include "qt/qt_strings.hpp"
#include "qt/shortcuts.hpp"
#include "core/help_pages.hpp"
#include "qt/theme.hpp"
#include "qt/widgets/controls.hpp"

namespace sirius::app {

    using widgets::CaptionLabel;
    using widgets::ClickRow;
    using widgets::GlyphButton;
    using widgets::Icon;
    using widgets::Rule;

    namespace {

        // The design's 2 px accent focus ring on the widgets that paint
        // themselves, inset so that even a 14 px box can show one. theme.cpp's
        // filter stamps "focusVisible" for keyboard focus only, so a mouse
        // click never draws a ring.
        void drawFocusRing(QPainter& p, const QRectF& box) {
            p.setPen(QPen(theme::kAccent, 2));
            p.setBrush(Qt::NoBrush);
            p.drawRect(box);
        }

        bool focusRingVisible(const QWidget* w) {
            return w->hasFocus() && w->property("focusVisible").toBool();
        }

        // A 14 × 14 enable box that swallows the click so the row does not select.
        class EnableBox : public QAbstractButton {
        public:
            explicit EnableBox(QWidget* parent = nullptr) : QAbstractButton(parent) {
                setCheckable(true);
                setFixedSize(14, 14);
                setCursor(Qt::PointingHandCursor);
                setToolTip(withShortcut(QStringLiteral("Enable / skip this step"), keys::enableStep()));
                setAccessibleName(QStringLiteral("Enabled"));
                setAccessibleDescription(QStringLiteral("A skipped step passes its input on unchanged"));
                setFocusPolicy(Qt::StrongFocus);
            }

        protected:
            void paintEvent(QPaintEvent*) override {
                QPainter p(this);
                p.setPen(QPen(isEnabled() ? theme::kNeutral700 : theme::kNeutral400, 1.5));
                p.setBrush(isChecked() ? QBrush(isEnabled() ? theme::kAccent : theme::kNeutral400) : Qt::NoBrush);
                p.drawRect(QRectF(rect()).adjusted(0.75, 0.75, -0.75, -0.75));
                if (focusRingVisible(this)) drawFocusRing(p, QRectF(rect()).adjusted(1.0, 1.0, -1.0, -1.0));
            }
            void keyPressEvent(QKeyEvent* e) override {
                if (e->key() == Qt::Key_Return || e->key() == Qt::Key_Enter) {
                    click();
                    return;
                }
                QAbstractButton::keyPressEvent(e);   // Space is the button's own
            }
            void mouseReleaseEvent(QMouseEvent* e) override {
                QAbstractButton::mouseReleaseEvent(e);
                e->accept();
            }
        };

        // The cache glyph of the row and of the legend: M and D really are
        // letters in the design, Recompute is the circular arrow.
        struct CacheLook {
            QString glyph;
            Icon icon;
            QColor color;
            QString title;
        };

        CacheLook cacheLook(CachePolicy c) {
            switch (c) {
                case CachePolicy::Memory:
                    return {QStringLiteral("M"), Icon::None, theme::kAccentText, QStringLiteral("Cached in GPU/RAM")};
                case CachePolicy::Disk:
                    return {QStringLiteral("D"), Icon::None, theme::kText, QStringLiteral("Cached on disk (zarr scratch)")};
                case CachePolicy::Recompute: break;
            }
            return {QString(), Icon::Recompute, theme::kNeutral500, QStringLiteral("Recomputed on demand")};
        }

        // A label-sized painted icon, for the places the design puts an icon
        // in running text (the cache column, the legend).
        QLabel* iconLabel(Icon icon, int px, const QColor& color, QWidget* parent) {
            auto* l = new QLabel(parent);
            l->setFixedSize(px, px);
            l->setPixmap(widgets::iconPixmap(icon, px, color));
            return l;
        }

        // One step row: grid 22 | 1fr | auto.
        class StepRow : public ClickRow {
        public:
            StepRow(WorkbenchBridge& bridge, QWidget* parent) : ClickRow(parent), bridge_(bridge) {
                setEdge(true);
                auto* grid = new QGridLayout(this);
                grid->setContentsMargins(10, 9, 14, 9);
                grid->setHorizontalSpacing(10);
                grid->setVerticalSpacing(0);
                grid->setColumnMinimumWidth(0, 22);
                grid->setColumnStretch(1, 1);

                auto* leftCell = new QWidget(this);
                auto* leftLayout = new QHBoxLayout(leftCell);
                leftLayout->setContentsMargins(0, 0, 0, 0);
                leftLayout->setAlignment(Qt::AlignCenter);
                enable_ = new EnableBox(leftCell);
                pin_ = iconLabel(Icon::Pin, 12, theme::kNeutral500, leftCell);
                pin_->setToolTip(QStringLiteral("Always first, always enabled"));
                pin_->setAlignment(Qt::AlignCenter);
                leftLayout->addWidget(enable_);
                leftLayout->addWidget(pin_);
                grid->addWidget(leftCell, 0, 0, 2, 1, Qt::AlignCenter);

                body_ = new QWidget(this);
                auto* bodyLayout = new QVBoxLayout(body_);
                bodyLayout->setContentsMargins(0, 0, 0, 0);
                bodyLayout->setSpacing(0);
                auto* titleRow = new QHBoxLayout();
                titleRow->setContentsMargins(0, 0, 0, 0);
                titleRow->setSpacing(8);
                name_ = new widgets::ElidedLabel(body_);
                name_->setFont(theme::heading(13));
                kind_ = new QLabel(body_);
                QFont kf = theme::font(10);
                kf.setLetterSpacing(QFont::PercentageSpacing, 106.0);
                kind_->setFont(kf);
                QPalette kp = kind_->palette();
                kp.setColor(QPalette::WindowText, theme::kNeutral600);
                kind_->setPalette(kp);
                titleRow->addWidget(name_, 1);
                titleRow->addWidget(kind_, 0, Qt::AlignBaseline);
                bodyLayout->addLayout(titleRow);
                auto* sumRow = new QHBoxLayout();
                sumRow->setContentsMargins(0, 0, 0, 0);
                sumRow->setSpacing(6);
                cache_ = widgets::label(QString(), 11, theme::kText, QFont::ExtraBold, body_);
                cache_->setFixedWidth(12);
                cache_->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
                summary_ = new widgets::ElidedLabel(body_);
                summary_->setFont(theme::font(11));
                QPalette sp0 = summary_->palette();
                sp0.setColor(QPalette::WindowText, theme::kNeutral600);
                summary_->setPalette(sp0);
                sumRow->addWidget(cache_);
                sumRow->addWidget(summary_, 1);
                bodyLayout->addLayout(sumRow);
                grid->addWidget(body_, 0, 1, 2, 1);

                auto* right = new QWidget(this);
                auto* rightLayout = new QHBoxLayout(right);
                rightLayout->setContentsMargins(0, 0, 0, 0);
                rightLayout->setSpacing(4);
                // The chevrons stack into one 14 px column: at the design's
                // 290 px dock every pixel of the row belongs to the name.
                auto* reorder = new QWidget(right);
                auto* reorderLayout = new QVBoxLayout(reorder);
                reorderLayout->setContentsMargins(0, 0, 0, 0);
                reorderLayout->setSpacing(0);
                up_ = new GlyphButton(Icon::ChevronUp, QSize(14, 11), reorder);
                up_->setBorderless(true);
                up_->setIconPx(11);
                up_->setIdleColor(theme::kNeutral500);
                up_->setToolTip(withShortcut(QStringLiteral("Move up"), keys::moveUp()));
                up_->setAccessibleName(QStringLiteral("Move step up"));
                down_ = new GlyphButton(Icon::ChevronDown, QSize(14, 11), reorder);
                down_->setBorderless(true);
                down_->setIconPx(11);
                down_->setIdleColor(theme::kNeutral500);
                down_->setToolTip(withShortcut(QStringLiteral("Move down"), keys::moveDown()));
                down_->setAccessibleName(QStringLiteral("Move step down"));
                reorderLayout->addWidget(up_);
                reorderLayout->addWidget(down_);
                reorder_ = reorder;
                view_ = new GlyphButton(Icon::Eye, 20, right);
                view_->setIconPx(14);
                view_->setIdleColor(theme::kNeutral400);
                view_->setToolTip(QStringLiteral("Show this step's output in the viewer"));
                view_->setAccessibleName(QStringLiteral("Show in the viewer"));
                remove_ = new GlyphButton(Icon::Trash, 20, right);
                remove_->setIconPx(14);
                remove_->setIdleColor(theme::kNeutral400);
                remove_->setBorderColor(theme::kNeutral400);
                remove_->setToolTip(withShortcut(QStringLiteral("Remove this step"), keys::removeStep()));
                remove_->setAccessibleName(QStringLiteral("Remove step"));
                rightLayout->addWidget(reorder);
                rightLayout->addWidget(view_);
                rightLayout->addWidget(remove_);
                grid->addWidget(right, 0, 2, 2, 1, Qt::AlignVCenter);

                connect(this, &ClickRow::clicked, this, [this] { bridge_.wb().select(index_); });
                connect(enable_, &QAbstractButton::clicked, this, [this](bool on) {
                    bridge_.wb().setStepEnabled(index_, on);
                });
                connect(up_, &QAbstractButton::clicked, this, [this] { bridge_.wb().moveStep(index_, -1); });
                connect(down_, &QAbstractButton::clicked, this, [this] { bridge_.wb().moveStep(index_, +1); });
                connect(view_, &QAbstractButton::clicked, this, [this] { bridge_.wb().view(index_); });
                connect(remove_, &QAbstractButton::clicked, this, [this] {
                    if (onRemove) onRemove(index_);
                });
            }

            // The window handles removal (cache warning, undo hint).
            std::function<void(int)> onRemove;

            void refresh(int index) {
                index_ = index;
                const Workbench& wb = bridge_.wb();
                const Step& step = wb.pipeline().at(index);
                const bool selected = wb.selectedIndex() == index;
                const bool viewed = wb.viewedIndex() == index;
                setSelected(selected);
                enable_->setVisible(!step.pinned);
                pin_->setVisible(step.pinned);
                remove_->setVisible(!step.pinned);
                enable_->setChecked(step.enabled);
                name_->setFullText(fromStd(step.name));
                // The kind label keeps its natural width (it is a short, fixed
                // vocabulary); the name is what gives way in a narrow dock.
                kind_->setText(captionCase(fromStd(step.op().info().kindLabel)));
                cacheLook_ = cacheLook(step.cache);
                cache_->setToolTip(cacheLook_.title);
                QString sum = fromStd(wb.stepSummary(index));
                const Validation v = wb.stepValidation(index);
                if (!v.ok()) sum = fromStd(v.firstError());
                else if (!step.enabled) sum += QStringLiteral(" — skipped");
                summary_->setFullText(sum);
                QPalette sp = summary_->palette();
                sp.setColor(QPalette::WindowText, v.ok() ? theme::kNeutral600 : theme::kAccentText);
                summary_->setPalette(sp);
                body_->setEnabled(true);
                setBodyOpacity(step.enabled ? 1.0 : 0.45);
                reorder_->setVisible(!step.pinned);
                // While a run is active the workbench refuses every pipeline
                // edit, so the row's controls say so instead of doing nothing.
                const bool editable = wb.canEdit() && !bridge_.taskRunning();
                enable_->setEnabled(editable);
                remove_->setEnabled(editable);
                up_->setEnabled(editable && index > 1);
                down_->setEnabled(editable && index < wb.pipeline().size() - 1);
                if (!editable) {
                    const QString frozen = QStringLiteral(" — not while a run or load is in progress");
                    for (QWidget* w : {static_cast<QWidget*>(enable_), static_cast<QWidget*>(up_),
                                       static_cast<QWidget*>(down_), static_cast<QWidget*>(remove_)})
                        w->setToolTip(w->toolTip().section(QStringLiteral(" — "), 0, 0) + frozen);
                } else {
                    enable_->setToolTip(withShortcut(QStringLiteral("Enable / skip this step"), keys::enableStep()));
                    up_->setToolTip(withShortcut(QStringLiteral("Move up"), keys::moveUp()));
                    down_->setToolTip(withShortcut(QStringLiteral("Move down"), keys::moveDown()));
                    remove_->setToolTip(withShortcut(QStringLiteral("Remove this step"), keys::removeStep()));
                }
                view_->setActive(viewed);
                setAccessibleName(QStringLiteral("Step %1 %2").arg(fromStd(Step::number(index)), fromStd(step.name)));
                setAccessibleDescription(QStringLiteral("%1 · %2%3")
                                             .arg(fromStd(step.op().info().kindLabel), summary_->fullText(),
                                                  viewed ? QStringLiteral(" · shown in the viewer") : QString()));
            }

        private:
            void setBodyOpacity(double op) {
                // QSS has no opacity: fade the labels' colours instead.
                auto fade = [op](QLabel* l, const QColor& base) {
                    QPalette p = l->palette();
                    QColor c = base;
                    c.setAlphaF(static_cast<float>(op));
                    p.setColor(QPalette::WindowText, c);
                    l->setPalette(p);
                };
                fade(name_, theme::kText);
                fade(kind_, theme::kNeutral600);
                if (op < 1.0) fade(summary_, theme::kNeutral600);
                // The cache column is a letter for Memory / Disk and a painted
                // arrow for Recompute, so the fade has to reach both.
                QColor cacheColor = cacheLook_.color;
                cacheColor.setAlphaF(static_cast<float>(op));
                if (cacheLook_.icon == Icon::None) {
                    cache_->setText(cacheLook_.glyph);
                    QPalette cp = cache_->palette();
                    cp.setColor(QPalette::WindowText, cacheColor);
                    cache_->setPalette(cp);
                } else {
                    cache_->setPixmap(widgets::iconPixmap(cacheLook_.icon, 11, cacheColor));
                }
            }

            WorkbenchBridge& bridge_;
            int index_ = 0;
            EnableBox* enable_ = nullptr;
            QLabel* pin_ = nullptr;
            QWidget* body_ = nullptr;
            widgets::ElidedLabel* name_ = nullptr;
            QLabel* kind_ = nullptr;
            QLabel* cache_ = nullptr;
            CacheLook cacheLook_;
            widgets::ElidedLabel* summary_ = nullptr;
            QWidget* reorder_ = nullptr;
            GlyphButton* up_ = nullptr;
            GlyphButton* down_ = nullptr;
            GlyphButton* view_ = nullptr;
            GlyphButton* remove_ = nullptr;
        };

        // One sentence about an operation, from the first paragraph of its help
        // page, with the markdown and inline maths stripped.
        QString operationBlurb(const std::string& kind) {
            std::string intro;
            try {
                intro = loadHelpPage(kind).intro;
            } catch (const std::exception&) {
                return QString();
            }
            std::string out;
            bool math = false;
            for (char c : intro) {
                if (c == '$') {
                    math = !math;
                    continue;
                }
                if (math || c == '*' || c == '`' || c == '_' || c == '\n' || c == '\r') {
                    if (c == '\n' || c == '\r') out += ' ';
                    continue;
                }
                out += c;
            }
            // the first sentence, or a trimmed line
            const std::size_t stop = out.find(". ");
            if (stop != std::string::npos && stop > 30) out = out.substr(0, stop + 1);
            if (out.size() > 190) out = out.substr(0, 187) + "…";
            return fromStd(out).simplified();
        }

        // Grouped dropdown: 82 px group caption column | items; with
        // descriptions (a setting) every item carries its one-line blurb.
        class AddMenu : public QFrame {
        public:
            AddMenu(WorkbenchBridge& bridge, QWidget* parent) : QFrame(parent), bridge_(bridge) {
                setObjectName(QStringLiteral("AddMenu"));
                setAutoFillBackground(true);
                QPalette p = palette();
                p.setColor(QPalette::Window, theme::kBg);
                setPalette(p);
                layout_ = new QVBoxLayout(this);
                layout_->setContentsMargins(2, 2, 2, 2);
                layout_->setSpacing(0);
                details_ = QSettings().value(QStringLiteral("ops/addMenuDetails"), false).toBool();
                rebuild();
            }

            // Plugins register operations after start-up, and the HPC notes
            // depend on the backend: rebuild the list the next time the menu
            // opens.
            void markStale() { stale_ = true; }

            // "Manage user operations…" link (in the User group, or under the
            // last group while there are no user operations yet).
            std::function<void()> onManagePlugins;

        protected:
            void showEvent(QShowEvent* e) override {
                if (stale_) {
                    rebuild();
                    stale_ = false;
                }
                QFrame::showEvent(e);
            }

        private:
            void rebuild() {
                QVBoxLayout* layout = layout_;
                while (QLayoutItem* item = layout->takeAt(0)) {
                    if (item->widget()) item->widget()->deleteLater();
                    delete item;
                }
                auto manageLink = [this](QWidget* parent) {
                    auto* link = new ClickRow(parent);
                    link->setTopRule(0);
                    auto* ll = new QHBoxLayout(link);
                    ll->setContentsMargins(10, 6, 10, 6);
                    ll->addWidget(widgets::label(QStringLiteral("Manage user operations…"), 11, theme::kAccentText, -1, link));
                    link->setAccessibleName(QStringLiteral("Manage user operations"));
                    link->setToolTip(QStringLiteral("Browse, edit and create the Python files that define user operations"));
                    connect(link, &ClickRow::clicked, this, [this] {
                        hide();
                        if (onManagePlugins) onManagePlugins();
                    });
                    return link;
                };
                // header: what this is, and the descriptions toggle
                {
                    auto* head = new QWidget(this);
                    auto* hl = new QHBoxLayout(head);
                    hl->setContentsMargins(10, 6, 10, 6);
                    hl->addWidget(new CaptionLabel(QStringLiteral("Add a step"), head), 1);
                    auto* toggle = new ClickRow(head);
                    toggle->setTopRule(0);
                    auto* tl = new QHBoxLayout(toggle);
                    tl->setContentsMargins(6, 2, 6, 2);
                    tl->addWidget(widgets::label(details_ ? QStringLiteral("Hide descriptions") : QStringLiteral("Show descriptions"), 11,
                                                 theme::kAccentText, -1, toggle));
                    toggle->setAccessibleName(QStringLiteral("Show descriptions"));
                    toggle->setToolTip(QStringLiteral("Show a sentence about every operation (kept in the settings)"));
                    connect(toggle, &ClickRow::clicked, this, [this] {
                        details_ = !details_;
                        QSettings().setValue(QStringLiteral("ops/addMenuDetails"), details_);
                        rebuild();
                    });
                    hl->addWidget(toggle);
                    layout->addWidget(head);
                    layout->addWidget(new Rule(1, Qt::Horizontal, this));
                }
                bool linked = false;
                for (const auto& [group, ops] : operationGroups()) {
                    auto* row = new QWidget(this);
                    auto* grid = new QGridLayout(row);
                    grid->setContentsMargins(0, 0, 0, 0);
                    grid->setSpacing(0);
                    // 72 rather than the design's 82: the dock is 290 px wide
                    // and "Volume reconstruction" has to fit beside the caption.
                    grid->setColumnMinimumWidth(0, 72);
                    auto* cap = new CaptionLabel(fromStd(group), row);
                    cap->setContentsMargins(10, 8, 10, 8);
                    cap->setAlignment(Qt::AlignTop | Qt::AlignLeft);
                    grid->addWidget(cap, 0, 0, Qt::AlignTop);
                    auto* items = new QWidget(row);
                    auto* itemsLayout = new QVBoxLayout(items);
                    itemsLayout->setContentsMargins(0, 0, 0, 0);
                    itemsLayout->setSpacing(0);
                    const bool hpc = bridge_.wb().backend() == Backend::Hpc;
                    for (const Operation* op : ops) {
                        auto* item = new ClickRow(items);
                        item->setTopRule(0);
                        auto* il = new QVBoxLayout(item);
                        il->setContentsMargins(10, details_ ? 7 : 6, 10, details_ ? 7 : 6);
                        il->setSpacing(2);
                        auto* name = new widgets::ElidedLabel(item);
                        name->setFont(theme::heading(12));
                        name->setFullText(fromStd(op->info().name));
                        il->addWidget(name);
                        item->setAccessibleName(fromStd(op->info().name));
                        // Only the operations the Python worker implements
                        // (OpInfo::remoteCapable) actually go to the cluster;
                        // the rest are C++ and run on this machine whatever
                        // the backend says. Say so where the step is chosen.
                        if (hpc && !op->info().remoteCapable) {
                            auto* note = new widgets::ElidedLabel(item);
                            note->setFont(theme::font(11));
                            QPalette np = note->palette();
                            np.setColor(QPalette::WindowText, theme::kNeutral600);
                            note->setPalette(np);
                            note->setFullText(QStringLiteral("local only"));
                            note->setToolTip(QStringLiteral("%1 has no HPC implementation: it runs on this machine even "
                                                            "with the HPC backend selected.")
                                                 .arg(fromStd(op->info().name)));
                            il->addWidget(note);
                            item->setAccessibleDescription(QStringLiteral("No HPC implementation: runs on this machine"));
                        }
                        if (details_) {
                            QString blurb = operationBlurb(op->kind());
                            if (op->info().plugin && !op->info().source.empty())
                                blurb = (blurb.isEmpty() ? QString() : blurb + QStringLiteral(" · ")) +
                                        QStringLiteral("user operation · ") + QFileInfo(fromStd(op->info().source)).fileName();
                            if (!blurb.isEmpty()) {
                                auto* text = widgets::label(blurb, 11, theme::kNeutral600, -1, item);
                                text->setWordWrap(true);
                                il->addWidget(text);
                            }
                        }
                        const std::string kind = op->kind();
                        connect(item, &ClickRow::clicked, this, [this, kind] { add(kind); });
                        itemsLayout->addWidget(item);
                    }
                    if (group == "User") {
                        itemsLayout->addWidget(manageLink(items));
                        linked = true;
                    }
                    grid->addWidget(items, 0, 1);
                    layout->addWidget(row);
                    layout->addWidget(new Rule(1, Qt::Horizontal, this));
                }
                if (!linked) {
                    layout->addWidget(manageLink(this));
                    layout->addWidget(new Rule(1, Qt::Horizontal, this));
                }
                auto* example = new ClickRow(this);
                example->setTopRule(0);
                auto* el = new QHBoxLayout(example);
                el->setContentsMargins(10, 8, 10, 8);
                auto* text = widgets::label(
                    QStringLiteral("Load example pipeline (SIM → einsum → contrast → merge → segment → volume)"), 11,
                    theme::kAccentText, -1, example);
                example->setAccessibleName(QStringLiteral("Load example pipeline"));
                text->setWordWrap(true);
                el->addWidget(text);
                connect(example, &ClickRow::clicked, this, [this] {
                    bridge_.wb().loadExamplePipeline();
                    hide();
                });
                layout->addWidget(example);
            }

        protected:
            void paintEvent(QPaintEvent* e) override {
                QFrame::paintEvent(e);
                QPainter p(this);
                p.setPen(QPen(theme::kText, 2));
                p.drawRect(rect().adjusted(1, 1, -1, -1));
            }

        private:
            QVBoxLayout* layout_ = nullptr;
            bool stale_ = false;
            bool details_ = false;

            void add(const std::string& kind) {
                bridge_.wb().addStep(kind);
                hide();
            }
            WorkbenchBridge& bridge_;
        };

    } // namespace

    struct OpsPanel::Impl {
        WorkbenchBridge& bridge;
        QLabel* count = nullptr;
        QWidget* rowsHost = nullptr;
        QVBoxLayout* rowsLayout = nullptr;
        QVector<StepRow*> rows;
        ClickRow* addRow = nullptr;
        GlyphButton* addGlyph = nullptr;
        QLabel* addTitle = nullptr;
        QLabel* addHint = nullptr;
        AddMenu* addMenu = nullptr;
        QPushButton* runAll = nullptr;
        QPushButton* exportBtn = nullptr;

        explicit Impl(WorkbenchBridge& b) : bridge(b) {}
    };

    OpsPanel::OpsPanel(WorkbenchBridge& bridge, QWidget* parent) : QWidget(parent), impl_(std::make_unique<Impl>(bridge)) {
        setObjectName(QStringLiteral("Panel"));
        // The design's 290 px is the width this asks for (sizeHint), not the
        // width it insists on: rows elide, so the dock can be dragged much
        // narrower when the viewer needs the room.
        setMinimumWidth(170);
        setAccessibleName(QStringLiteral("Operations"));
        auto* root = new QVBoxLayout(this);
        root->setContentsMargins(0, 0, 0, 0);
        root->setSpacing(0);

        // header
        auto* header = new QWidget(this);
        auto* hl = new QHBoxLayout(header);
        hl->setContentsMargins(14, 12, 14, 8);
        hl->addWidget(new CaptionLabel(QStringLiteral("Operations · any order"), header));
        hl->addStretch(1);
        impl_->count = widgets::label(QStringLiteral("0 steps"), 11, theme::kNeutral600, -1, header);
        hl->addWidget(impl_->count);
        root->addWidget(header);

        // scrolling rows + add + legend
        auto* scroll = new QScrollArea(this);
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);
        scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        auto* content = new QWidget(scroll);
        auto* cl = new QVBoxLayout(content);
        cl->setContentsMargins(0, 0, 0, 0);
        cl->setSpacing(0);

        impl_->rowsHost = new QWidget(content);
        impl_->rowsLayout = new QVBoxLayout(impl_->rowsHost);
        impl_->rowsLayout->setContentsMargins(0, 0, 0, 0);
        impl_->rowsLayout->setSpacing(0);
        cl->addWidget(impl_->rowsHost);

        impl_->addRow = new ClickRow(content);
        impl_->addRow->setAccessibleName(QStringLiteral("Add a processing step"));
        {
            auto* grid = new QGridLayout(impl_->addRow);
            grid->setContentsMargins(10, 12, 14, 12);
            grid->setHorizontalSpacing(10);
            grid->setColumnMinimumWidth(0, 22);
            grid->setColumnStretch(1, 1);
            impl_->addGlyph = new GlyphButton(Icon::Plus, 16, impl_->addRow);
            impl_->addGlyph->setDashed(true);
            impl_->addGlyph->setIconPx(10);
            impl_->addGlyph->setIdleColor(theme::kNeutral600);
            impl_->addGlyph->setAttribute(Qt::WA_TransparentForMouseEvents, true);
            impl_->addGlyph->setFocusPolicy(Qt::NoFocus);   // the row is the control, this is its "+"
            grid->addWidget(impl_->addGlyph, 0, 0, 2, 1, Qt::AlignCenter);
            impl_->addTitle = widgets::heading(QStringLiteral("Add a processing step"), 13, impl_->addRow);
            QPalette tp = impl_->addTitle->palette();
            tp.setColor(QPalette::WindowText, theme::kNeutral600);
            impl_->addTitle->setPalette(tp);
            impl_->addHint = widgets::label(QString(), 11, theme::kNeutral600, -1, impl_->addRow);
            grid->addWidget(impl_->addTitle, 0, 1);
            grid->addWidget(impl_->addHint, 1, 1);
        }
        connect(impl_->addRow, &ClickRow::clicked, this, [this] {
            impl_->addMenu->setVisible(!impl_->addMenu->isVisible());
            impl_->addRow->setSelected(impl_->addMenu->isVisible());
        });
        cl->addWidget(impl_->addRow);

        impl_->addMenu = new AddMenu(bridge, content);
        impl_->addMenu->hide();
        widgets::applyShadow(impl_->addMenu, false);
        impl_->addMenu->onManagePlugins = [this] {
            impl_->addRow->setSelected(false);
            emit managePluginsRequested();
        };
        connect(&bridge, &WorkbenchBridge::operationsChanged, this, [this] { impl_->addMenu->markStale(); });
        // The menu marks the operations that have no HPC implementation, so
        // it has to be rebuilt when the backend changes.
        connect(&bridge, &WorkbenchBridge::backendChanged, this, [this] { impl_->addMenu->markStale(); });
        auto* menuHost = new QWidget(content);
        auto* ml = new QHBoxLayout(menuHost);
        ml->setContentsMargins(8, 0, 8, 14);   // room under the menu for its shadow
        ml->addWidget(impl_->addMenu);
        cl->addWidget(menuHost);

        // legend
        cl->addWidget(new Rule(2, Qt::Horizontal, content));
        auto* legend = new QWidget(content);
        auto* lg = new QGridLayout(legend);
        lg->setContentsMargins(14, 12, 14, 12);
        lg->setHorizontalSpacing(8);
        lg->setVerticalSpacing(8);
        lg->setColumnMinimumWidth(0, 36);
        lg->setColumnStretch(1, 1);
        auto addLegend = [&](int row, QWidget* icon, const QString& text) {
            lg->addWidget(icon, row, 0, Qt::AlignCenter | Qt::AlignTop);
            auto* t = widgets::label(text, 12, theme::kNeutral700, -1, legend);
            t->setWordWrap(true);
            lg->addWidget(t, row, 1);
        };
        auto* onBox = new EnableBox(legend);
        onBox->setChecked(true);
        onBox->setAttribute(Qt::WA_TransparentForMouseEvents, true);
        onBox->setFocusPolicy(Qt::NoFocus);
        addLegend(0, onBox, QStringLiteral("Enabled — runs and passes its output on"));
        auto* offBox = new EnableBox(legend);
        offBox->setAttribute(Qt::WA_TransparentForMouseEvents, true);
        offBox->setFocusPolicy(Qt::NoFocus);
        addLegend(1, offBox, QStringLiteral("Skipped — data passes through unchanged"));
        auto* eye = new GlyphButton(Icon::Eye, 20, legend);
        eye->setActive(true);
        eye->setIconPx(14);
        eye->setAttribute(Qt::WA_TransparentForMouseEvents, true);
        eye->setFocusPolicy(Qt::NoFocus);   // the legend only explains the icons
        addLegend(2, eye, QStringLiteral("Shown in the viewer (click any step's eye)"));
        auto* arrows = new QWidget(legend);
        auto* al = new QHBoxLayout(arrows);
        al->setContentsMargins(0, 0, 0, 0);
        al->setSpacing(0);
        auto* pair = new QWidget(arrows);
        auto* pl = new QVBoxLayout(pair);
        pl->setContentsMargins(0, 0, 0, 0);
        pl->setSpacing(0);
        pl->addWidget(iconLabel(Icon::ChevronUp, 11, theme::kNeutral500, pair));
        pl->addWidget(iconLabel(Icon::ChevronDown, 11, theme::kNeutral500, pair));
        al->addWidget(pair);
        addLegend(3, arrows, QStringLiteral("Reorder — steps run top to bottom"));
        auto* cacheIcons = new QWidget(legend);
        auto* cil = new QHBoxLayout(cacheIcons);
        cil->setContentsMargins(0, 0, 0, 0);
        cil->setSpacing(3);
        cil->addWidget(widgets::label(QStringLiteral("M"), 11, theme::kAccentText, QFont::ExtraBold, cacheIcons));
        cil->addWidget(widgets::label(QStringLiteral("D"), 11, theme::kText, QFont::ExtraBold, cacheIcons));
        cil->addWidget(iconLabel(Icon::Recompute, 11, theme::kNeutral500, cacheIcons));
        addLegend(4, cacheIcons, QStringLiteral("Output cached in memory · on disk · recomputed"));
        cl->addWidget(legend);
        cl->addStretch(1);
        scroll->setWidget(content);
        root->addWidget(scroll, 1);

        // footer
        root->addWidget(new Rule(2, Qt::Horizontal, this));
        auto* footer = new QWidget(this);
        auto* fl = new QVBoxLayout(footer);
        fl->setContentsMargins(14, 12, 14, 12);
        fl->setSpacing(8);
        impl_->runAll = new QPushButton(QStringLiteral("Run all enabled"), footer);
        widgets::setButtonClass(impl_->runAll, "primary");
        impl_->runAll->setToolTip(withShortcut(QStringLiteral("Run every enabled step top to bottom"), keys::runAll()));
        impl_->exportBtn = new QPushButton(QStringLiteral("Export result…"), footer);
        widgets::setButtonClass(impl_->exportBtn, "secondary");
        fl->addWidget(impl_->runAll);
        fl->addWidget(impl_->exportBtn);
        root->addWidget(footer);

        connect(impl_->runAll, &QPushButton::clicked, this, [this] { impl_->bridge.startRun(-1); });
        connect(impl_->exportBtn, &QPushButton::clicked, this, &OpsPanel::exportRequested);

        auto refreshAll = [this] { refresh(); };
        connect(&bridge, &WorkbenchBridge::pipelineChanged, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::datasetChanged, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::selectionChanged, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::viewedStepChanged, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::outputsChanged, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::stepChanged, this, [this](int index) {
            if (index >= 0 && index < impl_->rows.size()) impl_->rows[index]->refresh(index);
        });
        connect(&bridge, &WorkbenchBridge::runStarted, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::runFinished, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::runStateChanged, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::taskStarted, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::taskFinished, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::runProgress, this, [this](double f, int, const QString&) {
            if (impl_->bridge.running())
                impl_->runAll->setText(QStringLiteral("Running · %1 %").arg(static_cast<int>(f * 100.0 + 0.5)));
        });
        refresh();
    }

    OpsPanel::~OpsPanel() = default;

    void OpsPanel::openAddMenu() {
        impl_->addMenu->show();
        impl_->addRow->setSelected(true);
    }

    void OpsPanel::refresh() {
        const Workbench& wb = impl_->bridge.wb();
        const Pipeline& p = wb.pipeline();
        const int n = p.size();
        while (impl_->rows.size() < n) {
            auto* row = new StepRow(impl_->bridge, impl_->rowsHost);
            row->onRemove = [this](int index) { emit removeStepRequested(index); };
            impl_->rowsLayout->addWidget(row);
            impl_->rows.push_back(row);
        }
        while (impl_->rows.size() > n) {
            StepRow* row = impl_->rows.takeLast();
            impl_->rowsLayout->removeWidget(row);
            row->deleteLater();
        }
        for (int i = 0; i < n; ++i) impl_->rows[i]->refresh(i);
        impl_->count->setText(QStringLiteral("%1 %2").arg(n).arg(n == 1 ? QStringLiteral("step") : QStringLiteral("steps")));
        impl_->addHint->setText(n < 2 ? QStringLiteral("Reconstruct, reduce, adjust, combine, segment…")
                                      : QStringLiteral("Runs after step %1").arg(fromStd(Step::number(n - 1))));
        const bool running = impl_->bridge.running();
        const bool busy = running || impl_->bridge.taskRunning();
        impl_->runAll->setEnabled(!busy && wb.hasDataset());
        if (!busy) impl_->runAll->setText(QStringLiteral("Run all enabled"));
        impl_->exportBtn->setEnabled(wb.hasDataset() && !busy);
        // Adding a step is an edit like any other: refused while a run holds
        // the pipeline, so the row (and the menu it opens) go with it.
        const bool editable = wb.canEdit() && !impl_->bridge.taskRunning();
        impl_->addRow->setEnabled(editable);
        impl_->addTitle->setEnabled(editable);
        impl_->addHint->setEnabled(editable);
        impl_->addRow->setToolTip(editable ? QString()
                                           : QStringLiteral("Not while a run or load is in progress — cancel it (Esc) or wait"));
        if (!editable && impl_->addMenu->isVisible()) {
            impl_->addMenu->hide();
            impl_->addRow->setSelected(false);
        }
    }

    QSize OpsPanel::sizeHint() const { return {theme::kOpsDockW, QWidget::sizeHint().height()}; }

} // namespace sirius::app
