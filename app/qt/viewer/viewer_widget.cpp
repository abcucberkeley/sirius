#include "qt/viewer/viewer_widget.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <set>
#include <tuple>

#include <QComboBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QGuiApplication>
#include <QLabel>
#include <QMenu>
#include <QPushButton>
#include <QCoreApplication>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QElapsedTimer>
#include <QPainter>
#include <QSettings>
#include <QShortcut>
#include <QShowEvent>
#include <QSplitter>
#include <QSplitterHandle>
#include <QStackedWidget>
#include <QTextDocument>
#include <QTimer>
#include <QVBoxLayout>

#include <sirius/constants.hpp>

#include "core/array_source.hpp"
#include "core/ops/builtin.hpp"
#include "qt/qt_strings.hpp"
#include "qt/theme.hpp"
#include "qt/trace.hpp"
#include "qt/widgets/controls.hpp"
#include "qt/viewer/dims_strip.hpp"
#include "qt/viewer/display_model.hpp"
#include "qt/viewer/slice_pane.hpp"
#include "qt/viewer/viewer_constants.hpp"
#include "qt/viewer/viewer_loader.hpp"
#include "qt/viewer/viewer_widgets.hpp"
#include "qt/viewer/volume_view.hpp"

namespace sirius::app {

    using widgets::GlyphButton;
    using widgets::Icon;
    using widgets::SegmentedControl;

    namespace {
        using viewer::kButtonZoomFactor;
        using viewer::kMaxZoom;
        using viewer::kMinZoom;
        using viewer::kPlayIntervalMs;
        using viewer::kWheelZoomBase;

        // The ortho splitters' saved balance, beside the window layout.
        const QString kOrthoRowsKey = QStringLiteral("viewer/orthoRows");
        const QString kOrthoColsKey = QStringLiteral("viewer/orthoCols");

        // "(V)", "(Shift+A)": the shortcut a tooltip advertises, in the
        // platform's own notation -- never hard-coded Mac glyphs.
        QString shortcutSuffix(const QKeySequence& keys) {
            const QString text = keys.toString(QKeySequence::NativeText);
            return text.isEmpty() ? QString() : QStringLiteral(" (") + text + QLatin1Char(')');
        }

        // A splitter whose 2 px handles keep the viewer ground of the design
        // ("2 px gaps on neutral-900") instead of the divider grey the
        // application stylesheet gives every other splitter.
        class GroundSplitter : public QSplitter {
        public:
            GroundSplitter(Qt::Orientation o, QWidget* parent) : QSplitter(o, parent) {
                setHandleWidth(viewer::kPaneGap);
                setChildrenCollapsible(false);
                setOpaqueResize(true);
            }

        protected:
            QSplitterHandle* createHandle() override {
                class Handle : public QSplitterHandle {
                public:
                    using QSplitterHandle::QSplitterHandle;

                protected:
                    void paintEvent(QPaintEvent*) override {
                        QPainter p(this);
                        p.fillRect(rect(), theme::kNeutral900);
                    }
                };
                auto* h = new Handle(orientation(), this);
                h->setAccessibleName(orientation() == Qt::Horizontal ? QStringLiteral("Pane width")
                                                                     : QStringLiteral("Pane height"));
                return h;
            }
        };

        // The zoom readout of the tool strip: 10 px text running upwards.
        class VerticalLabel : public QWidget {
        public:
            explicit VerticalLabel(QWidget* parent) : QWidget(parent) { setFixedWidth(14); }
            void setText(const QString& t) {
                text_ = t;
                update();
            }
            QSize sizeHint() const override { return {14, 60}; }

        protected:
            void paintEvent(QPaintEvent*) override {
                QPainter p(this);
                p.setFont(theme::tabular(theme::font(10)));
                p.setPen(theme::kNeutral600);
                p.translate(width(), height());
                p.rotate(-90);
                p.drawText(QRect(0, 0, height(), width()), Qt::AlignVCenter | Qt::AlignLeft, text_);
            }

        private:
            QString text_;
        };

        QLabel* richLabel(QWidget* parent) {
            auto* l = new QLabel(parent);
            l->setTextFormat(Qt::RichText);
            l->setFont(theme::font(12));
            return l;
        }

        QString num2(int index) {
            return fromStd(Step::number(index));
        }

        Index clampIndex(double v, Index n) {
            return std::clamp<Index>(static_cast<Index>(std::floor(v)), 0, std::max<Index>(n - 1, 0));
        }
    } // namespace

    // -------------------------------------------------------------------------
    // Impl
    // -------------------------------------------------------------------------

    struct ViewerWidget::Impl {
        ViewerWidget* q;
        WorkbenchBridge& bridge;
        Workbench& wb;

        // chrome
        SegmentedControl* modeSeg = nullptr;
        QLabel* viewing = nullptr;
        QString viewingHtml;
        TokenCheck* labelsCheck = nullptr;
        TokenCheck* soloCheck = nullptr;     // only the selected label
        TokenCheck* crossCheck = nullptr;
        TokenCheck* boxCheck = nullptr;
        QWidget* swatchHost = nullptr;
        QHBoxLayout* swatchLayout = nullptr;
        std::vector<ChannelSwatch*> swatches;
        QString swatchSignature;
        // multi-file datasets: which tile the pipeline reads (Load ▸ tile)
        QWidget* tileHost = nullptr;
        QComboBox* tileCombo = nullptr;
        QString tileSignature;

        // tool strip
        std::array<GlyphButton*, 5> tools{};
        VerticalLabel* zoomReadout = nullptr;

        // views
        QStackedWidget* stack = nullptr;
        QWidget* orthoPage = nullptr;
        // The ortho grid is two horizontal splitters (XY | YZ and XZ | MIP)
        // inside a vertical one, their columns kept in step, so the design's
        // 220 / 170 targets are a starting balance and not a cage.
        QSplitter* orthoRows = nullptr;
        QSplitter* orthoTop = nullptr;
        QSplitter* orthoBottom = nullptr;
        bool syncingSplit = false;
        bool orthoSized = false;
        QTimer saveSplitTimer;
        SlicePane* xy = nullptr;
        SlicePane* yz = nullptr;
        SlicePane* xz = nullptr;
        SlicePane* mip = nullptr;
        VolumeView* volume = nullptr;      // null when the platform cannot host a QOpenGLWidget
        QWidget* volumePage = nullptr;     // the VolumeView, or a notice
        QWidget* comparePage = nullptr;
        SlicePane* cmpLeft = nullptr;
        SlicePane* cmpRight = nullptr;
        DimsStrip* dims = nullptr;

        // rendering state
        DisplayModel model, rawModel;
        // Volumes, projections, exact ranges and the 3D bricks are produced
        // here, never on the GUI thread.
        ViewerLoader loader;
        quint64 volumeKey = 0;          // the reduction the 3D view is waiting for
        QString sliceNotice;            // "Loading…" for the panes that need a volume
        // (output, c, t) reads that threw: not asked for again until the
        // displayed output changes.
        std::set<std::tuple<const StepOutput*, Index, Index>> failedVolumes;
        int displayIndex = -1;
        QImage xyImg, xzImg, yzImg, mipImg, cmpLeftImg, cmpRightImg;
        int xyFactor = 1, mipFactor = 1, cmpFactor = 1, cmpLeftFactor = 1;
        // Rendered voxel regions of the XY-like panes: the visible part plus
        // a margin. Panning out of them, or a factor change, re-renders.
        QRect xyRegion, cmpRegion, cmpLeftRegion;
        // The voxels a pane shows now (no margin), for containment checks.
        static QRect visibleVoxels(const SlicePane* pane, Index cols, Index rows) {
            const QPointF a = pane->toVoxel(QPointF(0, 0));
            const QPointF b = pane->toVoxel(QPointF(pane->width(), pane->height()));
            const int x0 = std::clamp(static_cast<int>(std::floor(std::min(a.x(), b.x()))), 0, static_cast<int>(std::max<Index>(cols - 1, 0)));
            const int y0 = std::clamp(static_cast<int>(std::floor(std::min(a.y(), b.y()))), 0, static_cast<int>(std::max<Index>(rows - 1, 0)));
            const int x1 = std::clamp(static_cast<int>(std::ceil(std::max(a.x(), b.x()))), x0 + 1, static_cast<int>(cols));
            const int y1 = std::clamp(static_cast<int>(std::ceil(std::max(a.y(), b.y()))), y0 + 1, static_cast<int>(rows));
            return QRect(x0, y0, x1 - x0, y1 - y0);
        }
        // What to render: the visible voxels grown by a quarter of their size
        // on every side, aligned to the factor; the whole plane when that is
        // about as big anyway.
        static QRect renderRegion(const SlicePane* pane, int factor, Index cols, Index rows) {
            if (cols <= 0 || rows <= 0) return QRect();
            const QRect vis = visibleVoxels(pane, cols, rows);
            const int mx = std::max(factor, vis.width() / 4), my = std::max(factor, vis.height() / 4);
            int x0 = std::max(0, vis.x() - mx), y0 = std::max(0, vis.y() - my);
            int x1 = std::min(static_cast<int>(cols), vis.x() + vis.width() + mx);
            int y1 = std::min(static_cast<int>(rows), vis.y() + vis.height() + my);
            x0 = (x0 / factor) * factor;
            y0 = (y0 / factor) * factor;
            x1 = std::min(static_cast<int>(cols), ((x1 + factor - 1) / factor) * factor);
            y1 = std::min(static_cast<int>(rows), ((y1 + factor - 1) / factor) * factor);
            const QRect r(x0, y0, x1 - x0, y1 - y0);
            const QRect whole(0, 0, static_cast<int>(cols), static_cast<int>(rows));
            return static_cast<double>(r.width()) * r.height() >= 0.7 * static_cast<double>(cols) * rows ? whole : r;
        }
        struct Dirty {
            bool xy = true, xz = true, yz = true, mip = true, cmp = true, vol = true;
        } dirty;
        bool updateQueued = false;
        ViewState prev;
        bool havePrev = false;
        bool rebuildQueued = false;

        // interaction: annotations live on the plane (t, z) they were drawn
        // on; a measurement or ROI in progress is shown lighter until committed.
        struct Annotation {
            SlicePane::Annotation::Kind kind = SlicePane::Annotation::Kind::Measure;
            QVector<QPointF> points;
            QRectF rect;
            Index t = 0, z = 0;
        };
        std::vector<Annotation> annotations;
        QVector<QPointF> measure;   // pending measurement
        QRectF roi;                 // pending box
        QPointF roiStart;
        QString measureText(const QVector<QPointF>& points) const;
        QString roiText(const QRectF& r) const;
        void pushAnnotations();     // to the XY pane, filtered by the current plane
        void commitMeasure();
        void clearAnnotations();
        void showXYContextMenu(const QPoint& screen);
        std::array<Index, 3> splitA{};
        bool splitPending = false;
        std::uint32_t mergeFirst = 0;
        QPointF lastPaint;
        bool painting = false;
        quint64 labelsVersion = 0;   // bumps on every label edit: the 3D label texture follows
        QTimer playTimer;
        QString cursorText = QStringLiteral("cursor —");
        QString zoomText = QStringLiteral("100 %");

        Impl(ViewerWidget* q, WorkbenchBridge& b) : q(q), bridge(b), wb(b.wb()) {}

        // --- helpers ---------------------------------------------------------
        const ViewState& vs() const { return wb.viewState(); }
        Index nx() const { return model.dims().x; }
        Index ny() const { return model.dims().y; }
        Index nz() const { return model.dims().z; }
        Index nt() const { return model.dims().t; }
        double zAspect() const {
            if (!vs().physicalZ) return 1.0;   // voxel grid: one row per plane
            const auto& v = model.meta().voxelUm;
            return v[0] > 0.0 && v[2] > 0.0 ? v[2] / v[0] : 1.0;
        }
        Index curZ() const { return std::clamp<Index>(vs().z, 0, std::max<Index>(nz() - 1, 0)); }
        Index curT() const { return std::clamp<Index>(vs().t, 0, std::max<Index>(nt() - 1, 0)); }
        Index curX() const { return std::clamp<Index>(vs().cx, 0, std::max<Index>(nx() - 1, 0)); }
        Index curY() const { return std::clamp<Index>(vs().cy, 0, std::max<Index>(ny() - 1, 0)); }
        bool probe() const { return vs().tool == ViewerTool::Probe; }

        bool paintAvailable() const {
            if (model.hasLabels()) return true;
            for (const Step& s : wb.pipeline().steps()) {
                if (!s.enabled) continue;
                if (const Operation* op = findOperation(s.kind))
                    if (op->info().producesLabels || op->info().needsLabels) return true;
            }
            return false;
        }

        void build();
        void buildOrthoPage(QWidget* parent);
        void connectSignals();
        void restoreOrthoSizes();
        void saveOrthoSizes();
        void rebuildOutput();          // display output changed
        void applyLivePreview();       // window / gamma of a previewed step
        bool previewing = false;
        void refreshChrome();
        void refreshSwatches();
        void refreshTiles();
        void refreshDims();
        void refreshHints();
        void applyViewStateDiff(const ViewState& s);
        void scheduleUpdate();
        void applyDirty();
        void layoutPanes();
        void renderVolume();
        // Asks the loader for whatever (c, t) volumes the visible channels
        // still need; the aggregate state of those channels.
        DisplayModel::VolumeState ensureVolumes(DisplayModel& m, Index t);
        void onVolumeReady(const ViewerLoader::Volume& v);
        void onReductionReady(const ViewerLoader::Reduction& r);
        bool canPaint() const { return wb.canEdit(); }
        // Compare's own plane when View ▸ Sync Z / T is off.
        Index compareZ() const;
        Index compareT() const;
        Index cmpZ = 0, cmpT = 0;      // the raw pane's plane while unsynced
        bool cmpPinned = false;        // set when the sync is switched off
        void setZoomPan(double zoom, double panX, double panY);
        void zoomAround(double factor, const QPointF& xyScreen);
        void fit();
        void setCursorFor(ViewerTool t);

        // tools
        void onXYPressed(const QPointF& v, Qt::MouseButton b, Qt::KeyboardModifiers m);
        void onXYDragged(const QPointF& v, const QPointF& delta, Qt::MouseButton b, Qt::KeyboardModifiers m);
        void onXYReleased(const QPointF& v, Qt::MouseButton b, Qt::KeyboardModifiers m, bool moved);
        void paintAt(const QPointF& v, bool erase);
        std::uint32_t labelAt(Index z, Index y, Index x) const;
        void hover(SlicePane::Kind kind, const QPointF& v);
    };

    // --- building ----------------------------------------------------------------

    // "Grid 1fr 220px / 1fr 170px, 2 px gaps on neutral-900" -- as splitters,
    // so the user can rebalance XY against YZ / XZ and the design values are
    // where the balance starts. The two column splitters are kept in step so
    // XY stays above XZ and YZ above MIP.
    void ViewerWidget::Impl::buildOrthoPage(QWidget* parent) {
        orthoPage = new QWidget(parent);
        auto* ol = new QVBoxLayout(orthoPage);
        ol->setContentsMargins(0, 0, 0, 0);
        ol->setSpacing(0);
        orthoRows = new GroundSplitter(Qt::Vertical, orthoPage);
        orthoTop = new GroundSplitter(Qt::Horizontal, orthoRows);
        orthoBottom = new GroundSplitter(Qt::Horizontal, orthoRows);
        xy = new SlicePane(SlicePane::Kind::XY, orthoTop);
        yz = new SlicePane(SlicePane::Kind::YZ, orthoTop);
        xz = new SlicePane(SlicePane::Kind::XZ, orthoBottom);
        mip = new SlicePane(SlicePane::Kind::MIP, orthoBottom);
        xy->setObjectName(QStringLiteral("xyPane"));
        yz->setObjectName(QStringLiteral("yzPane"));
        xz->setObjectName(QStringLiteral("xzPane"));
        mip->setObjectName(QStringLiteral("mipPane"));
        xy->setMinimumSize(viewer::kMainPaneMin, viewer::kMainPaneMin);
        yz->setMinimumWidth(viewer::kSidePaneMin);
        xz->setMinimumHeight(viewer::kSidePaneMin);
        mip->setMinimumSize(viewer::kSidePaneMin, viewer::kSidePaneMin);
        orthoTop->addWidget(xy);
        orthoTop->addWidget(yz);
        orthoBottom->addWidget(xz);
        orthoBottom->addWidget(mip);
        orthoRows->addWidget(orthoTop);
        orthoRows->addWidget(orthoBottom);
        for (QSplitter* sp : {orthoTop, orthoBottom, orthoRows}) {
            sp->setStretchFactor(0, 1);
            sp->setStretchFactor(1, 0);
        }
        ol->addWidget(orthoRows);

        // one column balance for both rows
        auto mirror = [this](QSplitter* from, QSplitter* to) {
            if (syncingSplit) return;
            syncingSplit = true;
            to->setSizes(from->sizes());
            syncingSplit = false;
            saveSplitTimer.start();
        };
        QObject::connect(orthoTop, &QSplitter::splitterMoved, q, [this, mirror](int, int) { mirror(orthoTop, orthoBottom); });
        QObject::connect(orthoBottom, &QSplitter::splitterMoved, q, [this, mirror](int, int) { mirror(orthoBottom, orthoTop); });
        QObject::connect(orthoRows, &QSplitter::splitterMoved, q, [this](int, int) { saveSplitTimer.start(); });
        saveSplitTimer.setSingleShot(true);
        saveSplitTimer.setInterval(400);
        QObject::connect(&saveSplitTimer, &QTimer::timeout, q, [this] { saveOrthoSizes(); });
    }

    void ViewerWidget::Impl::restoreOrthoSizes() {
        if (orthoSized || !orthoRows) return;
        orthoSized = true;
        QSettings settings;
        const QByteArray rows = settings.value(kOrthoRowsKey).toByteArray();
        const QByteArray cols = settings.value(kOrthoColsKey).toByteArray();
        syncingSplit = true;
        const bool haveRows = !rows.isEmpty() && orthoRows->restoreState(rows);
        const bool haveCols = !cols.isEmpty() && orthoTop->restoreState(cols);
        if (haveCols) orthoBottom->setSizes(orthoTop->sizes());
        syncingSplit = false;
        if (haveRows && haveCols) return;
        // the design's targets for the 1600 x 960 window, as the balance to start from
        const int h = orthoRows->height(), w = orthoTop->width();
        syncingSplit = true;
        if (!haveRows && h > viewer::kXzHeight + viewer::kMainPaneMin)
            orthoRows->setSizes({h - viewer::kXzHeight - viewer::kPaneGap, viewer::kXzHeight});
        if (!haveCols && w > viewer::kYzWidth + viewer::kMainPaneMin) {
            const QList<int> sizes{w - viewer::kYzWidth - viewer::kPaneGap, viewer::kYzWidth};
            orthoTop->setSizes(sizes);
            orthoBottom->setSizes(sizes);
        }
        syncingSplit = false;
    }

    void ViewerWidget::Impl::saveOrthoSizes() {
        if (!orthoRows || !orthoSized) return;
        QSettings settings;
        settings.setValue(kOrthoRowsKey, orthoRows->saveState());
        settings.setValue(kOrthoColsKey, orthoTop->saveState());
    }

    void ViewerWidget::Impl::build() {
        auto* root = new QVBoxLayout(q);
        root->setContentsMargins(0, 0, 0, 0);
        root->setSpacing(0);

        // toolbar
        auto* bar = new QWidget(q);
        bar->setFixedHeight(theme::kViewerToolbarH);
        bar->setObjectName(QStringLiteral("viewerToolbar"));
        auto* bl = new QHBoxLayout(bar);
        bl->setContentsMargins(14, 0, 14, 0);
        bl->setSpacing(14);
        modeSeg = new SegmentedControl({QStringLiteral("Ortho"), QStringLiteral("3D"), QStringLiteral("Compare")}, bar);
        bl->addWidget(modeSeg);
        modeSeg->setAccessibleName(QStringLiteral("View mode"));
        viewing = richLabel(bar);
        // the one widget that yields when the toolbar is tight, so
        // "Ortho | 3D | Compare" is never clipped to "Orth"
        viewing->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
        viewing->setMinimumWidth(0);
        bl->addWidget(viewing, 1);
        labelsCheck = new TokenCheck(QStringLiteral("Labels"), bar);
        soloCheck = new TokenCheck(QStringLiteral("Solo"), bar);
        soloCheck->setToolTip(QStringLiteral("Show only the selected label, in the slices and in 3D; selecting a label jumps to it") +
                              shortcutSuffix(QKeySequence(Qt::Key_O)));
        crossCheck = new TokenCheck(QStringLiteral("Crosshair"), bar);
        boxCheck = new TokenCheck(QStringLiteral("Bounding box"), bar);
        bl->addWidget(labelsCheck);
        bl->addWidget(soloCheck);
        bl->addWidget(crossCheck);
        bl->addWidget(boxCheck);
        swatchHost = new QWidget(bar);
        swatchLayout = new QHBoxLayout(swatchHost);
        swatchLayout->setContentsMargins(0, 0, 0, 0);
        swatchLayout->setSpacing(4);
        bl->addWidget(swatchHost);
        // tile chooser: only for multi-file datasets with more than one tile
        tileHost = new QWidget(bar);
        auto* tl = new QHBoxLayout(tileHost);
        tl->setContentsMargins(6, 0, 0, 0);
        tl->setSpacing(4);
        tl->addWidget(new widgets::CaptionLabel(QStringLiteral("Tile"), tileHost));
        tileCombo = new QComboBox(tileHost);
        tileCombo->setAccessibleName(QStringLiteral("Tile"));
        tileCombo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
        tileCombo->setToolTip(QStringLiteral("Which tile of the multi-file dataset the pipeline reads (Load ▸ tile); the viewed step is re-run on it"));
        tl->addWidget(tileCombo);
        tileHost->hide();
        bl->addWidget(tileHost);
        QObject::connect(tileCombo, qOverload<int>(&QComboBox::currentIndexChanged), q, [this](int i) {
            if (i < 0) return;
            try {
                wb.setStepParam(0, "tile", static_cast<std::int64_t>(i));
            } catch (const std::exception& e) {
                wb.logLine(std::string("Tile: ") + e.what());
                return;
            }
            // switching tiles is navigation: show the new tile without a manual run
            if (!bridge.running()) bridge.startRun(wb.viewedIndex());
        });
        // display contrast: the auto percentile window or the full data range
        auto* autoBtn = new QPushButton(QStringLiteral("Auto"), bar);
        auto* resetBtn = new QPushButton(QStringLiteral("Reset"), bar);
        for (QPushButton* b : {autoBtn, resetBtn}) {
            widgets::setButtonClass(b, "ghost small");
            b->setCursor(Qt::PointingHandCursor);
        }
        autoBtn->setToolTip(QStringLiteral("Auto contrast (display): window on the 0.1–99.9 percentiles") +
                            shortcutSuffix(QKeySequence(Qt::SHIFT | Qt::Key_A)));
        resetBtn->setToolTip(QStringLiteral("Reset the display window to the full data range") +
                             shortcutSuffix(QKeySequence(Qt::SHIFT | Qt::Key_R)));
        QObject::connect(autoBtn, &QPushButton::clicked, q, [this] { q->autoContrast(); });
        QObject::connect(resetBtn, &QPushButton::clicked, q, [this] { q->resetContrast(); });
        bl->addWidget(autoBtn);
        bl->addWidget(resetBtn);
        root->addWidget(bar);
        root->addWidget(new widgets::Rule(theme::kRule, Qt::Horizontal, q));

        // canvas area: tool strip + views on the neutral-900 ground
        auto* canvas = new QWidget(q);
        canvas->setAutoFillBackground(true);
        {
            QPalette pal = canvas->palette();
            pal.setColor(QPalette::Window, theme::kNeutral900);
            canvas->setPalette(pal);
        }
        auto* cl = new QHBoxLayout(canvas);
        cl->setContentsMargins(2, 2, 2, 2);
        cl->setSpacing(2);

        auto* strip = new QWidget(canvas);
        strip->setFixedWidth(theme::kToolStripW);
        strip->setAutoFillBackground(true);
        {
            QPalette pal = strip->palette();
            pal.setColor(QPalette::Window, theme::kBg);
            strip->setPalette(pal);
        }
        auto* sl = new QVBoxLayout(strip);
        sl->setContentsMargins(0, 4, 0, 4);
        sl->setSpacing(2);
        sl->setAlignment(Qt::AlignHCenter | Qt::AlignTop);
        static const struct {
            Icon icon;
            const char* name;
            const char* tip;
            int key;
        } toolDefs[] = {
            {Icon::Navigate, "Navigate", "Navigate — drag to pan, wheel to zoom, double-click to fit", Qt::Key_V},
            {Icon::Probe, "Probe", "Probe — click to place the crosshair and read values", Qt::Key_P},
            {Icon::Measure, "Measure", "Measure distance / angle", Qt::Key_M},
            {Icon::Roi, "ROI", "ROI — drag a box", Qt::Key_R},
            {Icon::Brush, "Paint labels", "Paint labels — needs a segmentation step", Qt::Key_B}};
        for (std::size_t i = 0; i < tools.size(); ++i) {
            auto* b = new GlyphButton(toolDefs[i].icon, 28, strip);
            const QKeySequence keys(toolDefs[i].key);
            b->setToolTip(QString::fromUtf8(toolDefs[i].tip) + shortcutSuffix(keys));
            b->setAccessibleName(QString::fromUtf8(toolDefs[i].name));
            b->setAccessibleDescription(b->toolTip());
            b->setCheckable(true);
            tools[i] = b;
            sl->addWidget(b, 0, Qt::AlignHCenter);
        }
        auto* stripRule = new widgets::Rule(theme::kRule, Qt::Horizontal, strip);
        stripRule->setFixedSize(20, theme::kRule);
        sl->addSpacing(6);
        sl->addWidget(stripRule, 0, Qt::AlignHCenter);
        sl->addSpacing(6);
        auto* zin = new GlyphButton(Icon::Plus, 28, strip);
        zin->setToolTip(QStringLiteral("Zoom in") + shortcutSuffix(QKeySequence(Qt::Key_Plus)));
        zin->setAccessibleName(QStringLiteral("Zoom in"));
        auto* zout = new GlyphButton(Icon::Minus, 28, strip);
        zout->setToolTip(QStringLiteral("Zoom out") + shortcutSuffix(QKeySequence(Qt::Key_Minus)));
        zout->setAccessibleName(QStringLiteral("Zoom out"));
        auto* zfit = new GlyphButton(Icon::Fit, 28, strip);
        zfit->setToolTip(QStringLiteral("Fit to view") + shortcutSuffix(QKeySequence(Qt::Key_0)));
        zfit->setAccessibleName(QStringLiteral("Fit to view"));
        for (GlyphButton* b : {zin, zout, zfit}) sl->addWidget(b, 0, Qt::AlignHCenter);
        sl->addStretch(1);
        zoomReadout = new VerticalLabel(strip);
        sl->addWidget(zoomReadout, 0, Qt::AlignHCenter);
        cl->addWidget(strip);

        stack = new QStackedWidget(canvas);
        buildOrthoPage(stack);
        stack->addWidget(orthoPage);

        // QOpenGLWidget is unsupported (and crashes at flush) on the
        // offscreen / minimal platforms used by headless runs and tests.
        const QString platform = QGuiApplication::platformName();
        if (platform != QLatin1String("offscreen") && platform != QLatin1String("minimal") &&
            platform != QLatin1String("vnc")) {
            volume = new VolumeView(stack);
            volumePage = volume;
        } else {
            auto* notice = new QLabel(QStringLiteral("3D rendering needs OpenGL, which the %1 platform does not provide").arg(platform), stack);
            notice->setAlignment(Qt::AlignCenter);
            widgets::setWidgetClass(notice, "onDark");
            volumePage = notice;
        }
        stack->addWidget(volumePage);

        comparePage = new QWidget(stack);
        auto* cgrid = new QHBoxLayout(comparePage);
        cgrid->setContentsMargins(0, 0, 0, 0);
        cgrid->setSpacing(2);
        cmpLeft = new SlicePane(SlicePane::Kind::Compare, comparePage);
        cmpRight = new SlicePane(SlicePane::Kind::Compare, comparePage);
        cmpLeft->setObjectName(QStringLiteral("compareRawPane"));
        cmpRight->setObjectName(QStringLiteral("compareStepPane"));
        cgrid->addWidget(cmpLeft, 1);
        cgrid->addWidget(cmpRight, 1);
        stack->addWidget(comparePage);
        cl->addWidget(stack, 1);
        root->addWidget(canvas, 1);

        root->addWidget(new widgets::Rule(theme::kRule, Qt::Horizontal, q));
        dims = new DimsStrip(q);
        root->addWidget(dims);

        // chrome wiring
        QObject::connect(modeSeg, &SegmentedControl::changed, q, [this](int i) {
            wb.setViewMode(i == 0 ? ViewMode::Ortho : i == 1 ? ViewMode::Volume
                                                             : ViewMode::Compare);
        });
        QObject::connect(labelsCheck, &QAbstractButton::clicked, q, [this] { wb.toggleLabels(); });
        QObject::connect(soloCheck, &QAbstractButton::clicked, q, [this] { wb.toggleSoloLabel(); });
        QObject::connect(crossCheck, &QAbstractButton::clicked, q, [this] { wb.toggleCrosshair(); });
        QObject::connect(boxCheck, &QAbstractButton::clicked, q, [this] {
            ViewState s = vs();
            s.boundingBox = !s.boundingBox;
            wb.setViewState(s);
        });
        static const ViewerTool toolIds[] = {ViewerTool::Navigate, ViewerTool::Probe, ViewerTool::Measure,
                                             ViewerTool::Roi, ViewerTool::Paint};
        for (std::size_t i = 0; i < tools.size(); ++i) {
            const ViewerTool t = toolIds[i];
            QObject::connect(tools[i], &QAbstractButton::clicked, q, [this, t] {
                if (t == ViewerTool::Paint && !paintAvailable()) {
                    refreshChrome();
                    return;
                }
                wb.setTool(t);
            });
        }
        QObject::connect(zin, &QAbstractButton::clicked, q, [this] { q->zoomIn(); });
        QObject::connect(zout, &QAbstractButton::clicked, q, [this] { q->zoomOut(); });
        QObject::connect(zfit, &QAbstractButton::clicked, q, [this] { q->fitToWindow(); });

        // keyboard: tool letters and brush size (the menus own the rest)
        auto key = [this](const QKeySequence& ks, std::function<void()> fn) {
            auto* sc = new QShortcut(ks, q);
            sc->setContext(Qt::WidgetWithChildrenShortcut);
            QObject::connect(sc, &QShortcut::activated, q, std::move(fn));
        };
        key(QKeySequence(Qt::Key_V), [this] { wb.setTool(ViewerTool::Navigate); });
        key(QKeySequence(Qt::Key_P), [this] { wb.setTool(ViewerTool::Probe); });
        key(QKeySequence(Qt::Key_M), [this] { wb.setTool(ViewerTool::Measure); });
        key(QKeySequence(Qt::Key_R), [this] { wb.setTool(ViewerTool::Roi); });
        // Escape clears the annotations in progress -- unless a run or a
        // task is active, when the window's Cancel action owns the key: two
        // live shortcuts on one key are "ambiguous" to Qt and neither fires.
        auto* escape = new QShortcut(QKeySequence(Qt::Key_Escape), q);
        escape->setContext(Qt::WidgetWithChildrenShortcut);
        QObject::connect(escape, &QShortcut::activated, q, [this] {
            measure.clear();
            roi = QRectF();
            pushAnnotations();
        });
        auto armEscape = [this, escape] { escape->setEnabled(!bridge.running() && !bridge.taskRunning()); };
        QObject::connect(&bridge, &WorkbenchBridge::runStateChanged, q, armEscape);
        QObject::connect(&bridge, &WorkbenchBridge::taskStarted, q, armEscape);
        QObject::connect(&bridge, &WorkbenchBridge::taskFinished, q, armEscape);
        armEscape();
        key(QKeySequence(Qt::Key_BracketLeft), [this] {
            ViewState s = vs();
            s.brushPx = std::max(viewer::kBrushMinPx, s.brushPx - viewer::kBrushStepPx);
            wb.setViewState(s);
        });
        key(QKeySequence(Qt::Key_BracketRight), [this] {
            ViewState s = vs();
            s.brushPx = std::min(viewer::kBrushMaxPx, s.brushPx + viewer::kBrushStepPx);
            wb.setViewState(s);
        });

        playTimer.setInterval(kPlayIntervalMs);
        QObject::connect(&playTimer, &QTimer::timeout, q, [this] {
            if (nt() <= 1) {
                q->setPlaying(false);
                return;
            }
            wb.setT((curT() + 1) % nt());
        });
        QObject::connect(dims, &DimsStrip::zRequested, q, [this](Index z) { wb.setZ(z); });
        QObject::connect(dims, &DimsStrip::tRequested, q, [this](Index t) { wb.setT(t); });
        QObject::connect(dims, &DimsStrip::playToggled, q, [this](bool on) { q->setPlaying(on); });

        // Tab order: the toolbar left to right, then the tool strip top to
        // bottom, then the panes, then the dims strip. The channel swatches
        // are rebuilt with the output, so refreshSwatches() re-links them.
        QWidget* chain[] = {modeSeg, labelsCheck, soloCheck, crossCheck, boxCheck, autoBtn, resetBtn,
                            tools[0], tools[1], tools[2], tools[3], tools[4], zin, zout,
                            zfit, xy, yz, xz, mip, dims};
        for (std::size_t i = 0; i + 1 < sizeof chain / sizeof chain[0]; ++i) QWidget::setTabOrder(chain[i], chain[i + 1]);
    }

    void ViewerWidget::Impl::connectSignals() {
        // pane events
        QObject::connect(xy, &SlicePane::pressed, q, [this](QPointF v, Qt::MouseButton b, Qt::KeyboardModifiers m) { onXYPressed(v, b, m); });
        QObject::connect(xy, &SlicePane::contextMenuRequested, q, [this](QPoint sp, QPointF) { showXYContextMenu(sp); });
        QObject::connect(xy, &SlicePane::dragged, q, [this](QPointF v, QPointF d, Qt::MouseButton b, Qt::KeyboardModifiers m) { onXYDragged(v, d, b, m); });
        QObject::connect(xy, &SlicePane::released, q, [this](QPointF v, Qt::MouseButton b, Qt::KeyboardModifiers m, bool moved) { onXYReleased(v, b, m, moved); });
        QObject::connect(xy, &SlicePane::doubleClicked, q, [this](QPointF, Qt::KeyboardModifiers) {
            if (vs().tool == ViewerTool::Navigate || vs().tool == ViewerTool::Probe) fit();
        });
        QObject::connect(xy, &SlicePane::wheeled, q, [this](QPointF s, double steps, Qt::KeyboardModifiers) {
            zoomAround(std::pow(kWheelZoomBase, steps), s);
        });
        QObject::connect(xy, &SlicePane::hovered, q, [this](QPointF v) { hover(SlicePane::Kind::XY, v); });
        QObject::connect(xy, &SlicePane::resized, q, [this] { layoutPanes(); dirty.xy = true; scheduleUpdate(); });

        // YZ / XZ: probe moves the crosshair (and z); navigate pans along the shared axis
        QObject::connect(yz, &SlicePane::pressed, q, [this](QPointF v, Qt::MouseButton b, Qt::KeyboardModifiers) {
            if (b == Qt::LeftButton && probe() && model.valid())
                wb.setCrosshair(curX(), clampIndex(v.y(), ny()), clampIndex(v.x(), nz()));
        });
        QObject::connect(yz, &SlicePane::dragged, q, [this](QPointF v, QPointF d, Qt::MouseButton b, Qt::KeyboardModifiers) {
            if (b == Qt::MiddleButton || (b == Qt::LeftButton && vs().tool == ViewerTool::Navigate))
                setZoomPan(vs().zoom, vs().panX, vs().panY + d.y());
            else if (b == Qt::LeftButton && probe() && model.valid())
                wb.setCrosshair(curX(), clampIndex(v.y(), ny()), clampIndex(v.x(), nz()));
        });
        QObject::connect(yz, &SlicePane::wheeled, q, [this](QPointF s, double steps, Qt::KeyboardModifiers) {
            zoomAround(std::pow(kWheelZoomBase, steps), QPointF(xy->width() / 2.0, s.y()));
        });
        QObject::connect(yz, &SlicePane::hovered, q, [this](QPointF v) { hover(SlicePane::Kind::YZ, v); });
        QObject::connect(yz, &SlicePane::resized, q, [this] { layoutPanes(); });

        QObject::connect(xz, &SlicePane::pressed, q, [this](QPointF v, Qt::MouseButton b, Qt::KeyboardModifiers) {
            if (b == Qt::LeftButton && probe() && model.valid())
                wb.setCrosshair(clampIndex(v.x(), nx()), curY(), clampIndex(v.y(), nz()));
        });
        QObject::connect(xz, &SlicePane::dragged, q, [this](QPointF v, QPointF d, Qt::MouseButton b, Qt::KeyboardModifiers) {
            if (b == Qt::MiddleButton || (b == Qt::LeftButton && vs().tool == ViewerTool::Navigate))
                setZoomPan(vs().zoom, vs().panX + d.x(), vs().panY);
            else if (b == Qt::LeftButton && probe() && model.valid())
                wb.setCrosshair(clampIndex(v.x(), nx()), curY(), clampIndex(v.y(), nz()));
        });
        QObject::connect(xz, &SlicePane::wheeled, q, [this](QPointF s, double steps, Qt::KeyboardModifiers) {
            zoomAround(std::pow(kWheelZoomBase, steps), QPointF(s.x(), xy->height() / 2.0));
        });
        QObject::connect(xz, &SlicePane::hovered, q, [this](QPointF v) { hover(SlicePane::Kind::XZ, v); });
        QObject::connect(xz, &SlicePane::resized, q, [this] { layoutPanes(); });
        QObject::connect(mip, &SlicePane::resized, q, [this] { layoutPanes(); dirty.mip = true; scheduleUpdate(); });
        QObject::connect(mip, &SlicePane::pressed, q, [this](QPointF v, Qt::MouseButton b, Qt::KeyboardModifiers) {
            if (b == Qt::LeftButton && probe() && model.valid())
                wb.setCrosshair(clampIndex(v.x(), nx()), clampIndex(v.y(), ny()), curZ());
        });
        for (SlicePane* p : {xy, yz, xz, mip, cmpLeft, cmpRight}) {
            QObject::connect(p, &SlicePane::exited, q, [this] {
                cursorText = QStringLiteral("cursor —");
                emit q->cursorChanged(cursorText);
            });
            // Arrow keys in a focused pane move the crosshair over that
            // pane's own axes; page up / down step the third one.
            const SlicePane::Kind kind = p->kind();
            QObject::connect(p, &SlicePane::keyNavigated, q, [this, kind](int dc, int dr, int dd) {
                if (!model.valid()) return;
                Index x = curX(), y = curY(), z = curZ();
                switch (kind) {
                    case SlicePane::Kind::YZ:
                        z += dc;
                        y += dr;
                        x += dd;
                        break;
                    case SlicePane::Kind::XZ:
                        x += dc;
                        z += dr;
                        y += dd;
                        break;
                    default:
                        x += dc;
                        y += dr;
                        z += dd;
                        break;
                }
                wb.setCrosshair(x, y, z);   // clamps all three
            });
        }

        // compare panes share the XY transform
        for (SlicePane* p : {cmpLeft, cmpRight}) {
            QObject::connect(p, &SlicePane::dragged, q, [this](QPointF v, QPointF d, Qt::MouseButton b, Qt::KeyboardModifiers) {
                if (b == Qt::MiddleButton || (b == Qt::LeftButton && vs().tool == ViewerTool::Navigate))
                    setZoomPan(vs().zoom, vs().panX + d.x(), vs().panY + d.y());
                else if (b == Qt::LeftButton && probe() && model.valid())
                    wb.setCrosshair(clampIndex(v.x(), nx()), clampIndex(v.y(), ny()), curZ());
            });
            QObject::connect(p, &SlicePane::pressed, q, [this](QPointF v, Qt::MouseButton b, Qt::KeyboardModifiers) {
                if (b == Qt::LeftButton && probe() && model.valid())
                    wb.setCrosshair(clampIndex(v.x(), nx()), clampIndex(v.y(), ny()), curZ());
            });
            const bool raw = p == cmpLeft;
            QObject::connect(p, &SlicePane::wheeled, q, [this, raw](QPointF s, double steps, Qt::KeyboardModifiers m) {
                // With View ▸ Sync Z / T off, shift + wheel over the raw pane
                // moves its own plane -- the point of switching the sync off.
                if (raw && !vs().syncZT && m.testFlag(Qt::ShiftModifier)) {
                    cmpZ = std::clamp<Index>(cmpZ + static_cast<Index>(steps > 0 ? 1 : -1), 0, std::max<Index>(nz() - 1, 0));
                    dirty.cmp = true;
                    layoutPanes();
                    scheduleUpdate();
                    return;
                }
                zoomAround(std::pow(kWheelZoomBase, steps), s);
            });
            QObject::connect(p, &SlicePane::doubleClicked, q, [this](QPointF, Qt::KeyboardModifiers) { fit(); });
            QObject::connect(p, &SlicePane::hovered, q, [this](QPointF v) { hover(SlicePane::Kind::Compare, v); });
            QObject::connect(p, &SlicePane::resized, q, [this] { layoutPanes(); dirty.cmp = true; scheduleUpdate(); });
        }

        // volume view
        if (volume) {
            QObject::connect(volume, &VolumeView::orientationChanged, q, [this](double yaw, double pitch) {
                ViewState s = vs();
                s.yaw = yaw;
                s.pitch = pitch;
                wb.setViewState(s);
            });
            QObject::connect(volume, &VolumeView::clipChanged, q, [this](double lo, double hi) {
                ViewState s = vs();
                s.clipZ = {lo, hi};
                wb.setViewState(s);
            });
        }

        // workbench
        QObject::connect(&bridge, &WorkbenchBridge::datasetChanged, q, [this] { rebuildOutput(); });
        QObject::connect(&bridge, &WorkbenchBridge::viewedStepChanged, q, [this] { rebuildOutput(); });
        QObject::connect(&bridge, &WorkbenchBridge::outputsChanged, q, [this] { rebuildOutput(); });
        QObject::connect(&bridge, &WorkbenchBridge::pipelineChanged, q, [this] { rebuildOutput(); });
        QObject::connect(&bridge, &WorkbenchBridge::runFinished, q, [this](bool, const QString&) { rebuildOutput(); });
        QObject::connect(&bridge, &WorkbenchBridge::stepChanged, q, [this](int index) {
            if (index == 0) refreshTiles();   // Load ▸ tile edited elsewhere
            if (index == wb.viewedIndex()) {
                if (previewing || wb.viewedIsLivePreview()) {
                    rebuildOutput();   // re-applies the preview window
                    return;
                }
                refreshChrome();
                dirty.vol = true;
                scheduleUpdate();
            }
        });
        QObject::connect(&bridge, &WorkbenchBridge::labelsChanged, q, [this](quint64) {
            ++labelsVersion;
            dirty.xy = dirty.xz = dirty.yz = dirty.cmp = dirty.vol = true;
            scheduleUpdate();
        });
        QObject::connect(&bridge, &WorkbenchBridge::viewStateChanged, q, [this] { applyViewStateDiff(vs()); });
        // While a run holds the pipeline the workbench refuses label edits
        // (Workbench::canEdit): the paint tools go with it.
        QObject::connect(&bridge, &WorkbenchBridge::runStarted, q, [this] {
            if (painting) wb.endPaintStroke();
            painting = false;
            refreshChrome();
        });
        QObject::connect(&bridge, &WorkbenchBridge::runFinished, q, [this](bool, const QString&) { refreshChrome(); });

        // the loader's results
        QObject::connect(&loader, &ViewerLoader::volumeReady, q,
                         [this](const ViewerLoader::Volume& v) { onVolumeReady(v); });
        QObject::connect(&loader, &ViewerLoader::reductionReady, q,
                         [this](const ViewerLoader::Reduction& r) { onReductionReady(r); });
    }

    // --- asynchronous volumes ------------------------------------------------------

    DisplayModel::VolumeState ViewerWidget::Impl::ensureVolumes(DisplayModel& m, Index t) {
        if (!m.valid()) return DisplayModel::VolumeState::TooLarge;
        bool wanted = false, tooLarge = false;
        for (Index c = 0; c < m.dims().c; ++c) {
            if (!vs().channelOn(c)) continue;
            switch (m.volumeState(c, t)) {
                case DisplayModel::VolumeState::Ready: break;
                case DisplayModel::VolumeState::Wanted:
                    if (failedVolumes.count({m.output().get(), c, t})) break;
                    wanted = true;
                    loader.prepare(m.output(), c, t);
                    break;
                case DisplayModel::VolumeState::TooLarge: tooLarge = true; break;
            }
        }
        if (tooLarge) return DisplayModel::VolumeState::TooLarge;
        return wanted ? DisplayModel::VolumeState::Wanted : DisplayModel::VolumeState::Ready;
    }

    void ViewerWidget::Impl::onVolumeReady(const ViewerLoader::Volume& v) {
        DisplayModel* target = nullptr;
        if (v.out == model.output()) target = &model;
        else if (v.out == rawModel.output()) target = &rawModel;
        if (!target) return;                                  // the viewer moved on: drop it
        if (target == &model && v.t != curT()) return;        // a time point ago: drop it too
        if (!v.ok) {
            failedVolumes.insert({v.out.get(), v.c, v.t});
            sliceNotice = QStringLiteral("could not read the volume");
            refreshHints();
            wb.logLine("Viewer: " + toStd(v.error));
            return;
        }
        if (ScopedTrace::enabled())
            qInfo("view volume c%lld t%lld ready in %lld us (%s)", static_cast<long long>(v.c), static_cast<long long>(v.t),
                  v.micros, v.volume ? "read" : "in memory");
        target->installVolume(v.c, v.t, v.volume, v.mip, v.lo, v.hi);
        // A new exact range can move the display window, so nothing is
        // spared; applyDirty clears the loading notice once every visible
        // channel has arrived.
        dirty = Dirty{};
        scheduleUpdate();
    }

    void ViewerWidget::Impl::onReductionReady(const ViewerLoader::Reduction& r) {
        if (!volume || r.key != volumeKey) return;
        if (ScopedTrace::enabled())
            qInfo("view 3d reduction of %d channels in %lld us", static_cast<int>(r.channels.size()), r.micros);
        volume->setTextures(r.key, r.channels, model.meta().voxelUm, nz(), ny(), nx());
    }

    // --- output --------------------------------------------------------------------

    void ViewerWidget::Impl::rebuildOutput() {
        // coalesce bursts (a run finishing emits several notifications)
        if (rebuildQueued) return;
        rebuildQueued = true;
        QTimer::singleShot(0, q, [this] {
            rebuildQueued = false;
            int actual = -1;
            std::shared_ptr<const StepOutput> out = wb.hasDataset() ? wb.displayOutput(&actual) : nullptr;
            const bool changed = out != model.output();
            model.setOutput(out);
            displayIndex = actual;
            // the raw side of Compare: the Load step's output, or the bare source
            std::shared_ptr<const StepOutput> raw = wb.hasDataset() ? wb.output(0) : nullptr;
            if (!raw && wb.hasDataset() && wb.source()) {
                auto so = std::make_shared<StepOutput>();
                so->meta = wb.dataset();
                so->source = wb.source();
                raw = so;
            }
            if (raw != rawModel.output()) {
                rawModel.setOutput(raw);
                dirty.cmp = true;
            }
            if (changed) {
                // results for the old output are no longer wanted
                loader.cancelAll();
                volumeKey = 0;
                sliceNotice.clear();
                failedVolumes.clear();
                dirty = Dirty{};
                measure.clear();
                roi = QRectF();
                annotations.clear();
                splitPending = false;
                mergeFirst = 0;
            }
            applyLivePreview();
            refreshSwatches();
            refreshChrome();
            refreshDims();
            layoutPanes();
            scheduleUpdate();
        });
    }

    // A live-preview step (Contrast) that has not run is shown on its input
    // through its own window and gamma, so edits update the panes at once.
    void ViewerWidget::Impl::applyLivePreview() {
        const bool was = previewing;
        previewing = wb.viewedIsLivePreview() && model.valid();
        if (!previewing) {
            if (was) {
                model.resetWindows();
                dirty = Dirty{};
            }
            return;
        }
        const Step& st = wb.pipeline().at(wb.viewedIndex());
        const StepInput in = model.output()->asInput();
        for (Index c = 0; c < model.dims().c; ++c) {
            const ContrastWindow w = contrastWindow(in, st.params, c, 8);
            model.setWindow(c, DisplayWindow{w.lo, w.hi, w.gamma});
        }
        dirty = Dirty{};
    }

    void ViewerWidget::Impl::refreshSwatches() {
        const DatasetMeta& m = model.meta();
        QString sig;
        std::vector<std::pair<QString, QColor>> items;
        if (model.valid()) {
            if (m.rgb) {
                items = {{QStringLiteral("R"), QColor(255, 80, 80)}, {QStringLiteral("G"), QColor(80, 255, 80)}, {QStringLiteral("B"), QColor(110, 110, 255)}};
            } else {
                for (Index c = 0; c < m.dims.c; ++c) {
                    if (static_cast<std::size_t>(c) < m.channels.size()) {
                        const ChannelInfo& ch = m.channels[static_cast<std::size_t>(c)];
                        items.emplace_back(fromStd(ch.shortName()), QColor(fromStd(ch.hexColor())));
                    } else {
                        items.emplace_back(QString::number(c), QColor(255, 255, 255));
                    }
                }
            }
        }
        for (const auto& it : items) sig += it.first + it.second.name() + QLatin1Char(';');
        if (sig != swatchSignature) {
            swatchSignature = sig;
            for (ChannelSwatch* s : swatches) delete s;
            swatches.clear();
            for (std::size_t c = 0; c < items.size(); ++c) {
                auto* s = new ChannelSwatch(items[c].first, items[c].second, swatchHost);
                const Index ci = static_cast<Index>(c);
                QObject::connect(s, &QAbstractButton::clicked, q, [this, ci] { wb.setChannelVisible(ci, !vs().channelOn(ci)); });
                swatchLayout->addWidget(s);
                swatches.push_back(s);
            }
        }
        for (std::size_t c = 0; c < swatches.size(); ++c) swatches[c]->setChecked(vs().channelOn(static_cast<Index>(c)));
        // keep the rebuilt swatches in the tab order, after the check boxes
        QWidget* previous = boxCheck;
        for (ChannelSwatch* sw : swatches) {
            QWidget::setTabOrder(previous, sw);
            previous = sw;
        }
        refreshTiles();
    }

    // "1 · tile_0_0", "2 · tile_0_1", …: the displayed output's tiles, current
    // = the tile it was read from. Hidden for single-tile data.
    void ViewerWidget::Impl::refreshTiles() {
        if (!tileHost) return;
        const DatasetMeta& m = model.meta();
        if (!model.valid() || !m.hasTiles()) {
            tileHost->hide();
            return;
        }
        QString sig;
        for (const TileInfo& t : m.tiles) sig += fromStd(t.name) + QLatin1Char(';');
        QSignalBlocker block(tileCombo);
        if (sig != tileSignature) {
            tileSignature = sig;
            tileCombo->clear();
            for (std::size_t i = 0; i < m.tiles.size(); ++i)
                tileCombo->addItem(QStringLiteral("%1 · %2").arg(i + 1).arg(fromStd(m.tiles[i].name)));
        }
        tileCombo->setCurrentIndex(std::clamp(static_cast<int>(m.tileIndex), 0, tileCombo->count() - 1));
        tileHost->show();
    }

    void ViewerWidget::Impl::refreshChrome() {
        const ViewState& s = vs();
        modeSeg->setCurrentIndex(s.mode == ViewMode::Ortho ? 0 : s.mode == ViewMode::Volume ? 1
                                                                                            : 2);
        stack->setCurrentWidget(s.mode == ViewMode::Ortho ? orthoPage : s.mode == ViewMode::Volume ? volumePage
                                                                                                   : comparePage);

        // "Viewing 05 Contrast rgb z48 y4096 x4096"
        const int viewed = wb.viewedIndex();
        QString html = QStringLiteral("<span style='color:%1'>Viewing</span>").arg(theme::kNeutral600.name());
        if (viewed >= 0 && viewed < wb.pipeline().size()) {
            const Step& st = wb.pipeline().at(viewed);
            html += QStringLiteral(" <span style='color:%1;font-weight:800'>%2 %3</span>")
                        .arg(theme::kAccent.name(), num2(viewed), fromStd(st.name).toHtmlEscaped());
            QString shape = fromStd(wb.outputMetaOf(viewed).shapeString());
            html += QStringLiteral(" <span style='color:%1'>%2</span>").arg(theme::kNeutral500.name(), shape);
            if (previewing)
                html += QStringLiteral(" <span style='color:%1'>· live preview on %2</span>")
                            .arg(theme::kNeutral500.name(), num2(displayIndex));
            else if (displayIndex >= 0 && displayIndex != viewed)
                html += QStringLiteral(" <span style='color:%1'>· not run yet, showing %2</span>")
                            .arg(theme::kNeutral500.name(), num2(displayIndex));
        } else if (!wb.hasDataset()) {
            html += QStringLiteral(" <span style='color:%1'>no dataset</span>").arg(theme::kNeutral500.name());
        }
        if (html != viewingHtml) {   // a pan drag must not relayout the toolbar
            viewingHtml = html;
            viewing->setText(html);
            // the label is the toolbar's give: what it loses to a narrow
            // window is still readable here
            QTextDocument plain;
            plain.setHtml(html);
            viewing->setToolTip(plain.toPlainText());
            viewing->setAccessibleName(plain.toPlainText());
        }

        labelsCheck->setChecked(s.labels);
        labelsCheck->setEnabled(model.hasLabels());
        soloCheck->setChecked(s.soloLabel);
        soloCheck->setEnabled(model.hasLabels());
        soloCheck->setCaption(s.soloLabel ? (s.selectedLabel ? QStringLiteral("label %1").arg(s.selectedLabel) : QStringLiteral("select a label"))
                                          : QString());
        crossCheck->setChecked(s.crosshair);
        crossCheck->setCaption(s.tool == ViewerTool::Probe ? QString() : QStringLiteral("locked"));
        crossCheck->setVisible(s.mode != ViewMode::Volume);
        boxCheck->setChecked(s.boundingBox);
        boxCheck->setVisible(s.mode == ViewMode::Volume);

        static const ViewerTool toolIds[] = {ViewerTool::Navigate, ViewerTool::Probe, ViewerTool::Measure,
                                             ViewerTool::Roi, ViewerTool::Paint};
        // A run owns the pipeline: the workbench refuses label edits while it
        // lasts (Workbench::canEdit), so the brush is dimmed rather than
        // silently swallowing strokes. Navigate / Probe / Measure / ROI stay.
        const bool paintOk = paintAvailable() && canPaint();
        for (std::size_t i = 0; i < tools.size(); ++i) {
            tools[i]->setChecked(s.tool == toolIds[i]);
            if (toolIds[i] != ViewerTool::Paint) continue;
            tools[i]->setEnabled(paintOk);
            tools[i]->setToolTip(canPaint()
                                     ? QStringLiteral("Paint labels — needs a segmentation step") +
                                           shortcutSuffix(QKeySequence(Qt::Key_B))
                                     : QStringLiteral("Paint labels — not while a run is in progress"));
        }
        zoomText = QStringLiteral("%1 %").arg(std::lround(s.zoom * 100.0));
        zoomReadout->setText(zoomText);
        emit q->zoomChanged(zoomText);
        refreshHints();
        setCursorFor(s.tool);
        if (!volume) return;
        volume->setBoundingBox(s.boundingBox);
        volume->setOrientation(s.yaw, s.pitch);
        volume->setClip(s.clipZ[0], s.clipZ[1]);
        // transfer function from a volume reconstruction step's parameters
        if (viewed >= 0 && viewed < wb.pipeline().size() && wb.pipeline().at(viewed).kind == "volrec") {
            const ParamSet& p = wb.pipeline().at(viewed).params;
            const std::string method = p.getString("method", "Ray casting");
            const bool isMip = method.find("ax") != std::string::npos || method.find("MIP") != std::string::npos;
            volume->setTransfer(static_cast<float>(p.getDouble("opacity_lo", 0.05)),
                                static_cast<float>(p.getDouble("opacity_hi", 0.6)),
                                static_cast<float>(p.getDouble("opacity", 0.9)),
                                static_cast<float>(p.getDouble("step", 0.5)), isMip);
            volume->setMethodText(fromStd(method));
        } else {
            volume->setTransfer(0.05f, 0.6f, 0.9f, 0.5f, false);
            volume->setMethodText(QStringLiteral("Ray casting"));
        }
    }

    void ViewerWidget::Impl::refreshHints() {
        const ViewState& s = vs();
        if (s.tool != ViewerTool::Measure && !measure.isEmpty()) {
            measure.clear();
            pushAnnotations();
        }
        if (s.tool != ViewerTool::Roi && !roi.isNull()) {
            roi = QRectF();
            pushAnnotations();
        }
        QString hint;
        switch (s.tool) {
            case ViewerTool::Navigate: hint = QStringLiteral("drag · pan   wheel · zoom   double-click · fit"); break;
            case ViewerTool::Probe: hint = QStringLiteral("click · move crosshair   drag · follow   wheel · zoom"); break;
            case ViewerTool::Measure:
                hint = QStringLiteral("click twice · distance   %1 click · angle   right-click · clear")
                           .arg(QKeySequence(Qt::SHIFT).toString(QKeySequence::NativeText).remove(QLatin1Char('+')));
                break;
            case ViewerTool::Roi: hint = QStringLiteral("drag · box   right-click · clear"); break;
            case ViewerTool::Paint: {
                if (!canPaint()) {
                    hint = QStringLiteral("label edits are paused while a run is in progress");
                    break;
                }
                QString what;
                switch (s.paintTool) {
                    case PaintTool::Brush: what = QStringLiteral("brush %1 px · alt · erase · [ ] · size").arg(s.brushPx); break;
                    case PaintTool::Erase: what = QStringLiteral("erase %1 px · [ ] · size").arg(s.brushPx); break;
                    case PaintTool::Fill: what = QStringLiteral("click · fill region with the selected label"); break;
                    case PaintTool::Pick: what = QStringLiteral("click · pick label"); break;
                    case PaintTool::Merge: what = mergeFirst ? QStringLiteral("click the label to merge into %1").arg(mergeFirst) : QStringLiteral("click first label · then the second"); break;
                    case PaintTool::Split: what = splitPending ? QStringLiteral("click the second seed") : QStringLiteral("click two seeds inside one label"); break;
                    case PaintTool::Delete: what = QStringLiteral("click · delete label"); break;
                    case PaintTool::Lasso: what = QStringLiteral("lasso not available · painting with the brush"); break;
                }
                hint = what + QStringLiteral(" · crosshair locked");
                break;
            }
        }
        if (model.volumeTooLarge()) {
            yz->setMessage(QStringLiteral("volume too large for re-slicing"));
            xz->setMessage(QStringLiteral("volume too large for re-slicing"));
            mip->setMessage(QStringLiteral("volume too large"));
        } else {
            // "Loading…" while the loader reads the volume these three need
            for (SlicePane* p : {yz, xz, mip}) p->setMessage(sliceNotice);
        }
        xy->setHint(hint);
        cmpRight->setHint(hint);
        const bool brush = s.tool == ViewerTool::Paint && (s.paintTool == PaintTool::Brush || s.paintTool == PaintTool::Erase || s.paintTool == PaintTool::Lasso);
        xy->setBrushCursor(brush, s.brushPx / 2.0);
        cmpRight->setBrushCursor(brush, s.brushPx / 2.0);
    }

    void ViewerWidget::Impl::setCursorFor(ViewerTool t) {
        Qt::CursorShape shape = Qt::CrossCursor;
        if (t == ViewerTool::Navigate) shape = Qt::OpenHandCursor;
        else if (t == ViewerTool::Paint) shape = Qt::BlankCursor;
        xy->setCursor(shape);
        cmpLeft->setCursor(t == ViewerTool::Navigate ? Qt::OpenHandCursor : Qt::CrossCursor);
        cmpRight->setCursor(shape);
        for (SlicePane* p : {yz, xz, mip}) p->setCursor(t == ViewerTool::Navigate ? Qt::OpenHandCursor : Qt::CrossCursor);
    }

    void ViewerWidget::Impl::refreshDims() {
        const DatasetMeta& m = model.meta();
        dims->setExtents(model.valid() ? nz() : 1, model.valid() ? nt() : 1, m.dz(), m.frameIntervalS);
        dims->setPosition(vs().z, vs().t);
        dims->setPlaying(playTimer.isActive());
    }

    // --- view state ------------------------------------------------------------------

    void ViewerWidget::Impl::applyViewStateDiff(const ViewState& s) {
        if (s.syncZT) {   // the compare plane follows until the sync is switched off
            cmpZ = curZ();
            cmpT = curT();
        }
        if (!havePrev) {
            prev = s;
            havePrev = true;
            dirty = Dirty{};
        } else {
            if (s.t != prev.t) dirty = Dirty{};
            if (s.z != prev.z) dirty.xy = dirty.cmp = true;
            if (s.cx != prev.cx || s.cy != prev.cy) dirty.xz = dirty.yz = true;
            if (s.channelVisible != prev.channelVisible) dirty = Dirty{};
            if (s.labels != prev.labels || s.labelOpacity != prev.labelOpacity || s.selectedLabel != prev.selectedLabel ||
                s.soloLabel != prev.soloLabel)
                dirty.xy = dirty.xz = dirty.yz = dirty.cmp = dirty.vol = true;
            if (s.mode != prev.mode) {
                if (s.mode == ViewMode::Volume) dirty.vol = true;
                if (s.mode == ViewMode::Compare) dirty.cmp = true;
            }
            if (s.syncZT != prev.syncZT) dirty.cmp = true;
            // yaw / pitch / clip / bounding box need no re-upload: the volume
            // view keeps its own orientation state and repaints itself.
            prev = s;
        }
        refreshChrome();
        refreshSwatches();
        dims->setPosition(s.z, s.t);
        layoutPanes();
        scheduleUpdate();
    }

    void ViewerWidget::Impl::scheduleUpdate() {
        if (updateQueued) return;
        updateQueued = true;
        QTimer::singleShot(0, q, [this] {
            updateQueued = false;
            applyDirty();
        });
    }

    void ViewerWidget::Impl::layoutPanes() {
        if (!model.valid()) return;
        const ScopedTrace trace("layoutPanes");
        const ViewState& s = vs();
        const double za = zAspect();
        // XY: fit x state.zoom, offset by the pan
        xy->setGrid(nx(), ny());   // keeps cols/rows current before fitView (the image keeps its origin)
        const SlicePane::View fitV = xy->fitView(1.0, 1.0);
        SlicePane::View v;
        v.zx = v.zy = fitV.zx * s.zoom;
        v.ox = (xy->width() - nx() * v.zx) / 2.0 + s.panX;
        v.oy = (xy->height() - ny() * v.zy) / 2.0 + s.panY;
        xy->setView(v);
        const Index cz = curZ();
        // YZ: rows y follow XY, cols z at the physical aspect, centred on z when wider than the pane
        {
            SlicePane::View w;
            w.zy = v.zy;
            w.oy = v.oy;
            w.zx = v.zx * za;
            const double ez = nz() * w.zx;
            w.ox = ez <= yz->width() ? (yz->width() - ez) / 2.0 : yz->width() / 2.0 - (cz + 0.5) * w.zx;
            yz->setView(w);
        }
        {
            SlicePane::View w;
            w.zx = v.zx;
            w.ox = v.ox;
            w.zy = v.zy * za;
            const double ez = nz() * w.zy;
            w.oy = ez <= xz->height() ? (xz->height() - ez) / 2.0 : xz->height() / 2.0 - (cz + 0.5) * w.zy;
            xz->setView(w);
        }
        mip->setGrid(nx(), ny());   // as for xy: fitView needs the grid, not last frame's content
        mip->setView(mip->fitView(1.0, 1.0));
        {
            // Both compare panes show the same physical field: the raw pane
            // scales its (coarser or finer) voxels by the voxel-size ratio so
            // a 64-pixel raw frame overlays a 128-pixel reconstruction.
            // Both panes need their grid before fitView: until content first
            // arrives (or after the shape changes) cols/rows are stale and
            // fitView falls back to 1 px per voxel, which laid the compare
            // panes out at the wrong scale entirely.
            cmpRight->setGrid(nx(), ny());
            if (rawModel.valid()) cmpLeft->setGrid(rawModel.dims().x, rawModel.dims().y);
            const SlicePane::View f = cmpRight->fitView(1.0, 1.0);
            SlicePane::View w;
            w.zx = w.zy = f.zx * s.zoom;
            w.ox = (cmpRight->width() - nx() * w.zx) / 2.0 + s.panX;
            w.oy = (cmpRight->height() - ny() * w.zy) / 2.0 + s.panY;
            cmpRight->setView(w);
            SlicePane::View l = w;
            if (rawModel.valid() && rawModel.meta().dx() > 0.0 && rawModel.meta().dy() > 0.0) {
                // screen pixels per raw voxel: a raw voxel twice as large as
                // the reconstruction's covers twice the screen
                l.zx = w.zx * rawModel.meta().dx() / model.meta().dx();
                l.zy = w.zy * rawModel.meta().dy() / model.meta().dy();
            }
            cmpLeft->setView(l);
        }
        // a coarser render is enough when the image is smaller than the pane
        const int wantFactor = std::max(1, static_cast<int>(std::floor(1.0 / std::max(v.zx, 1e-6))));
        if (wantFactor != xyFactor) {
            xyFactor = wantFactor;
            dirty.xy = true;
        }
        const int cmpWant = std::max(1, static_cast<int>(std::floor(1.0 / std::max(cmpRight->view().zx, 1e-6))));
        if (cmpWant != cmpFactor) {
            cmpFactor = cmpWant;
            dirty.cmp = true;
        }
        // The raw pane has its own scale: after a step that changes the voxel
        // size (SIM halves it) a raw voxel covers twice the screen, so it needs
        // half the sub-sampling. Sharing the reconstruction's factor rendered
        // it at half the resolution it deserved and magnified the result.
        const int cmpLeftWant = std::max(1, static_cast<int>(std::floor(1.0 / std::max(cmpLeft->view().zx, 1e-6))));
        if (cmpLeftWant != cmpLeftFactor) {
            cmpLeftFactor = cmpLeftWant;
            dirty.cmp = true;
        }
        const int mipWant = std::max(1, static_cast<int>(std::floor(1.0 / std::max(mip->view().zx, 1e-6))));
        if (mipWant != mipFactor) {
            mipFactor = mipWant;
            dirty.mip = true;
        }
        xy->setSmooth(v.zx * xyFactor * xy->devicePixelRatioF() < 1.0);
        cmpRight->setSmooth(cmpRight->view().zx * cmpFactor * cmpRight->devicePixelRatioF() < 1.0);
        cmpLeft->setSmooth(cmpLeft->view().zx * cmpLeftFactor * cmpLeft->devicePixelRatioF() < 1.0);
        // the view moved out of what was rendered (pan / zoom / resize): render again
        if (!dirty.xy && xy->hasContent() && !xyRegion.isEmpty() && !xyRegion.contains(visibleVoxels(xy, nx(), ny()))) {
            dirty.xy = true;
            scheduleUpdate();
        }
        if (!dirty.cmp && s.mode == ViewMode::Compare && cmpRight->hasContent() && !cmpRegion.isEmpty() &&
            (!cmpRegion.contains(visibleVoxels(cmpRight, nx(), ny())) ||
             (rawModel.valid() && !cmpLeftRegion.isEmpty() && !cmpLeftRegion.contains(visibleVoxels(cmpLeft, rawModel.dims().x, rawModel.dims().y))))) {
            dirty.cmp = true;
            scheduleUpdate();
        }
        // overlays that depend on the view
        const ViewState& st = vs();
        const bool locked = st.tool != ViewerTool::Probe;
        const QPointF cross(static_cast<double>(curX()), static_cast<double>(curY()));
        xy->setCrosshair(cross, st.crosshair, locked);
        yz->setCrosshair(QPointF(static_cast<double>(cz), static_cast<double>(curY())), st.crosshair, locked);
        xz->setCrosshair(QPointF(static_cast<double>(curX()), static_cast<double>(cz)), st.crosshair, locked);
        mip->setCrosshair(cross, st.crosshair, locked);
        QPointF rawCross = cross;
        double rawDx = model.meta().dx();
        if (rawModel.valid() && rawModel.meta().dx() > 0.0 && rawModel.meta().dy() > 0.0) {
            rawCross = QPointF(cross.x() * model.meta().dx() / rawModel.meta().dx(),
                               cross.y() * model.meta().dy() / rawModel.meta().dy());
            rawDx = rawModel.meta().dx();
        }
        cmpLeft->setCrosshair(rawCross, st.crosshair, locked);
        cmpRight->setCrosshair(cross, st.crosshair, locked);
        // View ▸ Scale bar: 0 µm per voxel hides it
        const double dx = st.scaleBar ? model.meta().dx() : 0.0;
        xy->setScaleBar(dx);
        cmpLeft->setScaleBar(st.scaleBar ? rawDx : 0.0);
        cmpRight->setScaleBar(dx);
        mip->setScaleBar(dx);
        const QString zt = QStringLiteral("z %1 / %2  t %3 / %4  %5 %")
                               .arg(cz)
                               .arg(nz() - 1)
                               .arg(curT())
                               .arg(nt() - 1)
                               .arg(std::lround(st.zoom * 100.0));
        xy->setTitle(QStringLiteral("XY  ") + zt);
        yz->setTitle(QStringLiteral("YZ"));
        xz->setTitle(QStringLiteral("XZ"));
        mip->setTitle(QStringLiteral("MIP · Z"));
        const QString rawZt = st.syncZT
                                  ? zt
                                  : QStringLiteral("z %1 / %2  t %3 / %4  unsynced")
                                        .arg(compareZ())
                                        .arg(nz() - 1)
                                        .arg(compareT())
                                        .arg(nt() - 1);
        cmpLeft->setTitle(QStringLiteral("01 Load · raw  ") + rawZt);
        const int viewed = wb.viewedIndex();
        const QString name = viewed >= 0 && viewed < wb.pipeline().size()
                                 ? num2(viewed) + QLatin1Char(' ') + fromStd(wb.pipeline().at(viewed).name)
                                 : QString();
        cmpRight->setTitle(name);
        pushAnnotations();
    }

    void ViewerWidget::Impl::applyDirty() {
        if (!model.valid()) {
            for (SlicePane* p : {xy, yz, xz, mip, cmpLeft, cmpRight}) p->clearContent();
            xy->setMessage(wb.hasDataset() ? QStringLiteral("Nothing to display") : QStringLiteral("Open a dataset (File ▸ Open dataset…)"));
            if (volume) volume->clearVolumes();
            dirty = Dirty{};
            return;
        }
        xy->setMessage({});
        const ViewState& s = vs();
        const Index t = curT(), z = curZ();
        // SIRIUS_TRACE_VIEW=1 prints what each pane costs (render, label overlay, hand-over)
        const bool trace = ScopedTrace::enabled();
        QElapsedTimer clock;
        auto lap = [&clock](qint64& into) {
            into = clock.nsecsElapsed() / 1000;
            clock.restart();
        };
        // The XZ / YZ re-slices and the MIP corner need a whole (c, t)
        // volume: ask for what is missing and say so, rather than reading
        // gigabytes on this thread.
        const bool needsVolume = s.mode == ViewMode::Ortho;
        if (needsVolume) {
            const DisplayModel::VolumeState vstate = ensureVolumes(model, t);
            const QString notice = vstate == DisplayModel::VolumeState::Wanted ? QStringLiteral("Loading…") : QString();
            if (notice != sliceNotice) {
                sliceNotice = notice;
                if (ScopedTrace::enabled())
                    qInfo("view slices %s", notice.isEmpty() ? "have their volume" : "waiting for the volume (Loading...)");
                refreshHints();
            }
            if (vstate != DisplayModel::VolumeState::Ready) {
                // keep whatever the panes already show and try again when the
                // volume lands (onVolumeReady re-schedules)
                dirty.xz = dirty.yz = dirty.mip = false;
            }
        }
        if (s.mode == ViewMode::Ortho) {
            if (dirty.xy) {
                qint64 r = 0, o = 0, c = 0;
                clock.start();
                xyRegion = renderRegion(xy, xyFactor, nx(), ny());
                model.renderXY(t, z, s, xyFactor, xyImg, xyRegion);
                lap(r);
                if (s.labels) model.overlayLabelsXY(t, z, xyFactor, s, xyImg, xyRegion);
                lap(o);
                xy->setContent(xyImg, xyFactor, nx(), ny(), xyRegion.topLeft());
                lap(c);
                if (trace) {
                    const QRect vis = visibleVoxels(xy, nx(), ny());
                    qInfo("view xy %dx%d f%d at %d,%d (visible %d,%d %dx%d · pane %dx%d · zx %.3f ox %.1f): render %lld us · labels %lld us · content %lld us",
                          xyImg.width(), xyImg.height(), xyFactor, xyRegion.x(), xyRegion.y(), vis.x(), vis.y(), vis.width(), vis.height(),
                          xy->width(), xy->height(), xy->view().zx, xy->view().ox, r, o, c);
                }
                dirty.xy = false;
            }
            if (dirty.xz) {
                qint64 r = 0, o = 0, c = 0;
                clock.start();
                model.renderXZ(t, curY(), s, xzImg);
                lap(r);
                if (s.labels) model.overlayLabelsXZ(t, curY(), s, xzImg);
                lap(o);
                xz->setContent(xzImg, 1, nx(), nz());
                lap(c);
                if (trace) qInfo("view xz %dx%d: render %lld us · labels %lld us · content %lld us", xzImg.width(), xzImg.height(), r, o, c);
                dirty.xz = false;
            }
            if (dirty.yz) {
                qint64 r = 0, o = 0, c = 0;
                clock.start();
                model.renderYZ(t, curX(), s, yzImg);
                lap(r);
                if (s.labels) model.overlayLabelsYZ(t, curX(), s, yzImg);
                lap(o);
                yz->setContent(yzImg, 1, nz(), ny());
                lap(c);
                if (trace) qInfo("view yz %dx%d: render %lld us · labels %lld us · content %lld us", yzImg.width(), yzImg.height(), r, o, c);
                dirty.yz = false;
            }
            if (dirty.mip) {
                qint64 r = 0, c = 0;
                clock.start();
                model.renderMIP(t, s, mipFactor, mipImg);
                lap(r);
                mip->setContent(mipImg, mipFactor, nx(), ny());
                lap(c);
                if (trace) qInfo("view mip %dx%d f%d: render %lld us · content %lld us", mipImg.width(), mipImg.height(), mipFactor, r, c);
                dirty.mip = false;
            }
            if (model.volumeTooLarge()) refreshHints();
        } else if (s.mode == ViewMode::Compare) {
            if (dirty.cmp) {
                if (rawModel.valid()) {
                    const Index rt = std::clamp<Index>(compareT(), 0, rawModel.dims().t - 1);
                    const Index rz = std::clamp<Index>(compareZ(), 0, rawModel.dims().z - 1);
                    cmpLeftRegion = renderRegion(cmpLeft, cmpLeftFactor, rawModel.dims().x, rawModel.dims().y);
                    rawModel.renderXY(rt, rz, s, cmpLeftFactor, cmpLeftImg, cmpLeftRegion);
                    cmpLeft->setContent(cmpLeftImg, cmpLeftFactor, rawModel.dims().x, rawModel.dims().y, cmpLeftRegion.topLeft());
                    if (trace)
                        qInfo("view cmp raw %dx%d f%d (zx %.3f) · step f%d (zx %.3f)", cmpLeftImg.width(), cmpLeftImg.height(),
                              cmpLeftFactor, cmpLeft->view().zx, cmpFactor, cmpRight->view().zx);
                } else {
                    cmpLeft->clearContent();
                }
                cmpRegion = renderRegion(cmpRight, cmpFactor, nx(), ny());
                model.renderXY(t, z, s, cmpFactor, cmpRightImg, cmpRegion);
                if (s.labels) model.overlayLabelsXY(t, z, cmpFactor, s, cmpRightImg, cmpRegion);
                cmpRight->setContent(cmpRightImg, cmpFactor, nx(), ny(), cmpRegion.topLeft());
                dirty.cmp = false;
            }
        } else {
            if (dirty.vol) {
                renderVolume();
                dirty.vol = false;
            }
        }
        layoutPanes();
    }

    void ViewerWidget::Impl::renderVolume() {
        if (!volume) return;
        const ViewState& s = vs();
        const Index t = curT();
        const DisplayModel::VolumeState vstate = ensureVolumes(model, t);
        if (vstate == DisplayModel::VolumeState::TooLarge) {
            volume->clearVolumes();
            volume->setPreparing(QStringLiteral("Volume too large to render"));
            volumeKey = 0;
            return;
        }
        if (vstate == DisplayModel::VolumeState::Wanted) {
            volume->setPreparing(QStringLiteral("Loading volume…"));
            return;   // the textures already up stay up until the new ones land
        }
        // The reduction to <= 256 texels per axis is a pass over every voxel:
        // it runs on the loader thread and paintGL only uploads the result.
        std::vector<ViewerLoader::Channel> chans;
        quint64 key = static_cast<quint64>(reinterpret_cast<std::uintptr_t>(model.output().get())) ^ (static_cast<quint64>(t) << 48);
        for (Index c = 0; c < model.dims().c; ++c) {
            if (!s.channelOn(c)) continue;
            const float* v = model.volumeIfReady(c, t);
            if (!v) continue;
            ViewerLoader::Channel ch;
            ch.out = model.output();
            ch.hold = model.volumeHold(c, t);
            ch.data = v;
            ch.z = nz();
            ch.y = ny();
            ch.x = nx();
            const DisplayWindow w = model.window(c, t);
            ch.lo = w.lo;
            ch.hi = w.hi;
            if (model.meta().rgb) {
                ch.color = {c == 0 ? 1.f : 0.f, c == 1 ? 1.f : 0.f, c == 2 ? 1.f : 0.f};
            } else if (static_cast<std::size_t>(c) < model.meta().channels.size()) {
                ch.color = model.meta().channels[static_cast<std::size_t>(c)].color;
            }
            key ^= (static_cast<quint64>(c + 1) * 0x9e3779b97f4a7c15ull) ^ static_cast<quint64>(std::hash<float>{}(w.lo) * 31 + std::hash<float>{}(w.hi));
            chans.push_back(ch);
        }
        if (chans.empty()) {
            volume->clearVolumes();
            volume->setPreparing(QString());
            volumeKey = 0;
        } else if (key != volumeKey) {
            volumeKey = key;
            volume->setPreparing(QStringLiteral("Preparing volume…"));
            loader.reduce(key, std::move(chans));
        }
        // labels ride along as their own texture, toggled with the Labels box
        const LabelVolume* L = s.labels ? model.labels() : nullptr;
        if (L && t < L->t()) {
            const std::uint32_t only = s.soloLabel ? s.selectedLabel : 0u;
            const quint64 lkey = (static_cast<quint64>(reinterpret_cast<std::uintptr_t>(L)) ^ (static_cast<quint64>(t + 1) << 40) ^
                                  (labelsVersion << 8) ^ (static_cast<quint64>(only) << 20)) |
                                 1;
            volume->setLabels(lkey, L->volume(t), L->z(), L->y(), L->x(), static_cast<float>(s.labelOpacity), only);
        } else {
            volume->clearLabels();
        }
    }

    // --- zoom / pan --------------------------------------------------------------------

    void ViewerWidget::Impl::setZoomPan(double zoom, double panX, double panY) {
        ViewState s = vs();
        s.zoom = std::clamp(zoom, kMinZoom, kMaxZoom);
        s.panX = panX;
        s.panY = panY;
        if (s.zoom == 1.0 && zoom == 1.0 && panX == 0.0 && panY == 0.0) {
            s.panX = s.panY = 0.0;
        }
        wb.setViewState(s);
    }

    void ViewerWidget::Impl::zoomAround(double factor, const QPointF& anchor) {
        if (!model.valid()) return;
        const ViewState& s = vs();
        const double newZoom = std::clamp(s.zoom * factor, kMinZoom, kMaxZoom);
        if (newZoom == s.zoom) return;
        const SlicePane::View v = xy->view();
        const double fz = v.zx / s.zoom;   // px per voxel at fit
        const QPointF voxel = xy->toVoxel(anchor);
        const double z2 = fz * newZoom;
        const double ox = anchor.x() - voxel.x() * z2, oy = anchor.y() - voxel.y() * z2;
        const double panX = ox - (xy->width() - nx() * z2) / 2.0;
        const double panY = oy - (xy->height() - ny() * z2) / 2.0;
        setZoomPan(newZoom, panX, panY);
    }

    void ViewerWidget::Impl::fit() { setZoomPan(1.0, 0.0, 0.0); }

    // View ▸ Sync Z / T across viewers. On (the default) every pane shows the
    // plane the dims strip points at. Off, the raw side of Compare keeps the
    // plane it was on -- a reference to scrub the processed side against --
    // and shift + wheel over it moves that plane on its own.
    Index ViewerWidget::Impl::compareZ() const {
        return vs().syncZT ? curZ() : std::clamp<Index>(cmpZ, 0, std::max<Index>(nz() - 1, 0));
    }

    Index ViewerWidget::Impl::compareT() const {
        return vs().syncZT ? curT() : std::clamp<Index>(cmpT, 0, std::max<Index>(nt() - 1, 0));
    }

    // --- tools ---------------------------------------------------------------------------

    std::uint32_t ViewerWidget::Impl::labelAt(Index z, Index y, Index x) const {
        const LabelVolume* L = model.labels();
        if (!L) return 0;
        const Index t = curT();
        if (t >= L->t() || z < 0 || z >= L->z() || y < 0 || y >= L->y() || x < 0 || x >= L->x()) return 0;
        return L->at(t, z, y, x);
    }

    void ViewerWidget::Impl::paintAt(const QPointF& v, bool erase) {
        if (!model.valid()) return;
        // a drag that leaves the image stops painting instead of stamping
        // along the border row or column
        if (!xy->inside(v)) return;
        const Index x = clampIndex(v.x(), nx()), y = clampIndex(v.y(), ny());
        wb.paintLabels(curZ(), y, x, erase);
    }

    void ViewerWidget::Impl::onXYPressed(const QPointF& v, Qt::MouseButton b, Qt::KeyboardModifiers m) {
        if (!model.valid() || b != Qt::LeftButton) return;
        const ViewState& s = vs();
        const Index x = clampIndex(v.x(), nx()), y = clampIndex(v.y(), ny()), z = curZ();
        switch (s.tool) {
            case ViewerTool::Navigate:
                xy->setCursor(Qt::ClosedHandCursor);
                break;
            case ViewerTool::Probe:
                if (xy->inside(v)) wb.setCrosshair(x, y, z);
                break;
            case ViewerTool::Measure: {
                // shift-click extends the last distance on this plane to an angle
                Annotation* last = annotations.empty() ? nullptr : &annotations.back();
                if (m.testFlag(Qt::ShiftModifier) && measure.isEmpty() && last &&
                    last->kind == SlicePane::Annotation::Kind::Measure && last->points.size() == 2 && last->t == curT() && last->z == z) {
                    last->points.push_back(v);
                } else {
                    measure.push_back(v);
                    if (measure.size() >= 2) commitMeasure();
                }
                pushAnnotations();
                break;
            }
            case ViewerTool::Roi:
                roiStart = v;
                roi = QRectF();
                pushAnnotations();
                break;
            case ViewerTool::Paint: {
                if (!canPaint()) return;   // a run holds the pipeline
                const bool erase = s.paintTool == PaintTool::Erase || m.testFlag(Qt::AltModifier);
                switch (s.paintTool) {
                    case PaintTool::Brush:
                    case PaintTool::Erase:
                    case PaintTool::Lasso:
                        // one stroke, one undo entry: opened here, closed on release
                        wb.beginPaintStroke();
                        painting = true;
                        lastPaint = v;
                        paintAt(v, erase);
                        break;
                    case PaintTool::Fill:
                        wb.fillLabel(z, y, x);
                        break;
                    case PaintTool::Pick: {
                        ViewState ns = s;
                        ns.selectedLabel = labelAt(z, y, x);
                        wb.setViewState(ns);
                        break;
                    }
                    case PaintTool::Merge: {
                        const std::uint32_t id = labelAt(z, y, x);
                        if (id == 0) break;
                        if (mergeFirst == 0 || mergeFirst == id) {
                            mergeFirst = id;
                            ViewState ns = s;
                            ns.selectedLabel = id;
                            wb.setViewState(ns);
                        } else {
                            wb.mergeLabels({mergeFirst, id});
                            mergeFirst = 0;
                        }
                        refreshHints();
                        break;
                    }
                    case PaintTool::Split: {
                        if (!splitPending) {
                            splitA = {z, y, x};
                            splitPending = labelAt(z, y, x) != 0;
                        } else {
                            const std::uint32_t id = labelAt(splitA[0], splitA[1], splitA[2]);
                            if (id != 0) wb.splitLabel(id, splitA, {z, y, x});
                            splitPending = false;
                        }
                        refreshHints();
                        break;
                    }
                    case PaintTool::Delete: {
                        const std::uint32_t id = labelAt(z, y, x);
                        if (id != 0) wb.deleteLabel(id);
                        break;
                    }
                }
                break;
            }
        }
    }

    void ViewerWidget::Impl::onXYDragged(const QPointF& v, const QPointF& delta, Qt::MouseButton b, Qt::KeyboardModifiers m) {
        if (!model.valid()) return;
        const ViewState& s = vs();
        if (b == Qt::MiddleButton || (b == Qt::LeftButton && s.tool == ViewerTool::Navigate)) {
            setZoomPan(s.zoom, s.panX + delta.x(), s.panY + delta.y());
            return;
        }
        if (b != Qt::LeftButton) return;
        switch (s.tool) {
            case ViewerTool::Probe:
                if (xy->inside(v)) wb.setCrosshair(clampIndex(v.x(), nx()), clampIndex(v.y(), ny()), curZ());
                break;
            case ViewerTool::Roi:
                roi = QRectF(roiStart, v).normalized();
                pushAnnotations();
                break;
            case ViewerTool::Paint:
                if (painting && canPaint()) {
                    const ScopedTrace dragTrace("drag: paint handling");
                    const bool erase = s.paintTool == PaintTool::Erase || m.testFlag(Qt::AltModifier);
                    // stamp along the path so fast strokes stay continuous
                    const double spacing = std::max(1.0, s.brushPx / 4.0);
                    const QPointF d = v - lastPaint;
                    const double len = std::hypot(d.x(), d.y());
                    const int n = std::max(1, static_cast<int>(len / spacing));
                    for (int i = 1; i <= n; ++i) paintAt(lastPaint + d * (static_cast<double>(i) / n), erase);
                    lastPaint = v;
                }
                break;
            default:
                break;
        }
    }

    void ViewerWidget::Impl::onXYReleased(const QPointF&, Qt::MouseButton b, Qt::KeyboardModifiers, bool) {
        if (b == Qt::LeftButton && vs().tool == ViewerTool::Navigate) xy->setCursor(Qt::OpenHandCursor);
        if (b == Qt::LeftButton && vs().tool == ViewerTool::Roi) {
            // a box of at least one voxel becomes an annotation; a click does nothing
            if (roi.width() >= 1.0 && roi.height() >= 1.0) {
                Annotation a;
                a.kind = SlicePane::Annotation::Kind::Roi;
                a.rect = roi;
                a.t = curT();
                a.z = curZ();
                annotations.push_back(a);
            }
            roi = QRectF();
            pushAnnotations();
        }
        if (painting) wb.endPaintStroke();
        painting = false;
    }

    void ViewerWidget::Impl::commitMeasure() {
        Annotation a;
        a.kind = SlicePane::Annotation::Kind::Measure;
        a.points = measure;
        a.t = curT();
        a.z = curZ();
        annotations.push_back(a);
        measure.clear();
    }

    void ViewerWidget::Impl::clearAnnotations() {
        annotations.clear();
        measure.clear();
        roi = QRectF();
        pushAnnotations();
    }

    void ViewerWidget::Impl::pushAnnotations() {
        QVector<SlicePane::Annotation> out;
        for (const Annotation& a : annotations) {
            if (a.t != curT() || a.z != curZ()) continue;   // other planes keep theirs
            SlicePane::Annotation pa;
            pa.kind = a.kind;
            pa.points = a.points;
            pa.rect = a.rect;
            pa.text = a.kind == SlicePane::Annotation::Kind::Measure ? measureText(a.points) : roiText(a.rect);
            out.push_back(pa);
        }
        if (!measure.isEmpty()) {
            SlicePane::Annotation pa;
            pa.points = measure;
            pa.text = measureText(measure);
            pa.pending = true;
            out.push_back(pa);
        }
        if (!roi.isNull()) {
            SlicePane::Annotation pa;
            pa.kind = SlicePane::Annotation::Kind::Roi;
            pa.rect = roi;
            pa.text = roiText(roi);
            pa.pending = true;
            out.push_back(pa);
        }
        xy->setAnnotations(out);
    }

    void ViewerWidget::Impl::showXYContextMenu(const QPoint& screen) {
        QMenu menu(q);
        const bool any = !annotations.empty() || !measure.isEmpty() || !roi.isNull();
        QAction* clear = menu.addAction(QStringLiteral("Clear annotations"));
        clear->setEnabled(any);
        QAction* last = menu.addAction(QStringLiteral("Remove last annotation"));
        last->setEnabled(!annotations.empty());
        menu.addSeparator();
        QAction* fit = menu.addAction(QStringLiteral("Fit to window"));
        QAction* chosen = menu.exec(xy->mapToGlobal(screen));
        if (chosen == clear) clearAnnotations();
        else if (chosen == last) {
            annotations.pop_back();
            pushAnnotations();
        } else if (chosen == fit) q->fitToWindow();
    }

    QString ViewerWidget::Impl::roiText(const QRectF& r) const {
        const double dx = model.meta().dx(), dy = model.meta().dy();
        return QStringLiteral("%1 × %2 µm").arg(r.width() * dx, 0, 'f', 2).arg(r.height() * dy, 0, 'f', 2);
    }

    QString ViewerWidget::Impl::measureText(const QVector<QPointF>& points) const {
        if (points.size() < 2) return QString();
        const double dx = model.meta().dx(), dy = model.meta().dy();
        const QPointF a = points[0], b = points[1];
        const double um = std::hypot((b.x() - a.x()) * dx, (b.y() - a.y()) * dy);
        QString text = QStringLiteral("%1 µm").arg(um, 0, 'f', 2);
        if (points.size() >= 3) {
            const QPointF c = points[2];
            const double ux = (a.x() - b.x()) * dx, uy = (a.y() - b.y()) * dy;
            const double vx = (c.x() - b.x()) * dx, vy = (c.y() - b.y()) * dy;
            const double ang = std::acos(std::clamp((ux * vx + uy * vy) / std::max(1e-12, std::hypot(ux, uy) * std::hypot(vx, vy)), -1.0, 1.0));
            text += QStringLiteral("  ∠ %1°").arg(ang * 180.0 / sirius::kPi, 0, 'f', 1);
        }
        return text;
    }

    void ViewerWidget::Impl::hover(SlicePane::Kind kind, const QPointF& v) {
        if (!model.valid()) return;
        Index x = curX(), y = curY(), z = curZ();
        bool inside = true;
        switch (kind) {
            case SlicePane::Kind::XY:
            case SlicePane::Kind::MIP:
            case SlicePane::Kind::Compare:
                x = static_cast<Index>(std::floor(v.x()));
                y = static_cast<Index>(std::floor(v.y()));
                inside = x >= 0 && y >= 0 && x < nx() && y < ny();
                break;
            case SlicePane::Kind::XZ:
                x = static_cast<Index>(std::floor(v.x()));
                z = static_cast<Index>(std::floor(v.y()));
                inside = x >= 0 && z >= 0 && x < nx() && z < nz();
                break;
            case SlicePane::Kind::YZ:
                z = static_cast<Index>(std::floor(v.x()));
                y = static_cast<Index>(std::floor(v.y()));
                inside = z >= 0 && y >= 0 && z < nz() && y < ny();
                break;
        }
        if (!inside) {
            cursorText = QStringLiteral("cursor —");
        } else {
            Index c = 0;
            for (Index i = 0; i < model.dims().c; ++i)
                if (vs().channelOn(i)) {
                    c = i;
                    break;
                }
            std::optional<float> val;
            if (kind == SlicePane::Kind::XY || kind == SlicePane::Kind::Compare) val = model.valueAt(c, curT(), z, y, x);
            else if (const float* vol = model.volumeIfReady(c, curT())) val = vol[(z * ny() + y) * nx() + x];
            QString vtext = val ? QString::number(*val, 'g', 5) : QStringLiteral("—");
            if (const LabelVolume* L = model.labels(); L && vs().labels) {
                const std::uint32_t id = labelAt(z, y, x);
                if (id) vtext += QStringLiteral(" · label %1").arg(id);
            }
            cursorText = QStringLiteral("cursor %1, %2, %3 · %4").arg(x).arg(y).arg(z).arg(vtext);
        }
        emit q->cursorChanged(cursorText);
    }

    // -------------------------------------------------------------------------
    // ViewerWidget
    // -------------------------------------------------------------------------

    ViewerWidget::ViewerWidget(WorkbenchBridge& bridge, QWidget* parent)
        : QWidget(parent), impl_(std::make_unique<Impl>(this, bridge)) {
        setFocusPolicy(Qt::StrongFocus);
        impl_->build();
        impl_->connectSignals();
        impl_->rebuildOutput();
        impl_->applyViewStateDiff(impl_->vs());
    }

    ViewerWidget::~ViewerWidget() = default;

    void ViewerWidget::showEvent(QShowEvent* event) {
        QWidget::showEvent(event);
        // after the first layout pass, when the panes have their real sizes
        QTimer::singleShot(0, this, [this] { impl_->restoreOrthoSizes(); });
    }

    void ViewerWidget::zoomIn() {
        if (impl_->vs().mode == ViewMode::Volume) {
            if (impl_->volume) impl_->volume->setZoom(impl_->volume->zoom() * kButtonZoomFactor);
            return;
        }
        impl_->zoomAround(kButtonZoomFactor, QPointF(impl_->xy->width() / 2.0, impl_->xy->height() / 2.0));
    }

    void ViewerWidget::zoomOut() {
        if (impl_->vs().mode == ViewMode::Volume) {
            if (impl_->volume) impl_->volume->setZoom(impl_->volume->zoom() / kButtonZoomFactor);
            return;
        }
        impl_->zoomAround(1.0 / kButtonZoomFactor, QPointF(impl_->xy->width() / 2.0, impl_->xy->height() / 2.0));
    }

    void ViewerWidget::fitToWindow() {
        if (impl_->vs().mode == ViewMode::Volume && impl_->volume) impl_->volume->setZoom(1.0);
        impl_->fit();
    }

    void ViewerWidget::setPlaying(bool on) {
        if (on && impl_->nt() > 1) impl_->playTimer.start();
        else impl_->playTimer.stop();
        impl_->dims->setPlaying(impl_->playTimer.isActive());
    }

    bool ViewerWidget::playing() const { return impl_->playTimer.isActive(); }

    void ViewerWidget::autoContrast() {
        if (impl_->previewing) {   // the previewed step's own Auto
            const int i = impl_->wb.viewedIndex();
            if (auto up = impl_->wb.upstreamOutput(i))
                impl_->wb.setStepParams(i, contrastAutoParams(impl_->wb.pipeline().at(i).params, up->asInput()), "Auto contrast");
            return;
        }
        impl_->model.setWindowMode(DisplayModel::WindowMode::Auto);
        impl_->rawModel.setWindowMode(DisplayModel::WindowMode::Auto);
        impl_->dirty = Impl::Dirty{};
        impl_->scheduleUpdate();
    }

    void ViewerWidget::resetContrast() {
        if (impl_->previewing) {
            const int i = impl_->wb.viewedIndex();
            if (auto up = impl_->wb.upstreamOutput(i))
                impl_->wb.setStepParams(i, contrastResetParams(impl_->wb.pipeline().at(i).params, up->asInput()), "Reset contrast");
            return;
        }
        impl_->model.setWindowMode(DisplayModel::WindowMode::Full);
        impl_->rawModel.setWindowMode(DisplayModel::WindowMode::Full);
        // The full range is every voxel: ask for the volumes so the exact
        // one replaces the sampled stand-in as soon as it is read.
        impl_->ensureVolumes(impl_->model, impl_->curT());
        if (impl_->rawModel.valid()) impl_->ensureVolumes(impl_->rawModel, impl_->curT());
        impl_->dirty = Impl::Dirty{};
        impl_->scheduleUpdate();
    }

    QImage ViewerWidget::grabView() const {
        const ViewState& s = impl_->vs();
        if (s.mode == ViewMode::Volume && impl_->volume) return impl_->volume->grabImage();
        QWidget* page = s.mode == ViewMode::Ortho ? impl_->orthoPage : impl_->comparePage;
        return page->grab().toImage();
    }

    void ViewerWidget::syntheticStroke(const QPointF& fromVoxel, const QPointF& toVoxel, int moves) {
        SlicePane* pane = impl_->xy;
        if (!pane || !impl_->model.valid()) return;
        moves = std::max(1, moves);
        auto at = [pane](const QPointF& v) { return pane->toScreen(v); };
        auto send = [pane](QEvent::Type type, const QPointF& pos, Qt::MouseButton button, Qt::MouseButtons buttons) {
            QMouseEvent e(type, pos, pane->mapToGlobal(pos), button, buttons, Qt::NoModifier);
            QCoreApplication::sendEvent(pane, &e);
        };
        QElapsedTimer clock;
        clock.start();
        send(QEvent::MouseButtonPress, at(fromVoxel), Qt::LeftButton, Qt::LeftButton);
        QCoreApplication::processEvents();
        qint64 inSend = 0, inEvents = 0;
        QElapsedTimer part;
        for (int i = 1; i <= moves; ++i) {
            const double f = static_cast<double>(i) / moves;
            part.start();
            send(QEvent::MouseMove, at(fromVoxel + (toVoxel - fromVoxel) * f), Qt::NoButton, Qt::LeftButton);
            inSend += part.nsecsElapsed() / 1000;
            part.start();
            QCoreApplication::processEvents();
            inEvents += part.nsecsElapsed() / 1000;
        }
        qInfo("stroke: per move %lld us in the move handler, %lld us in the deferred events", inSend / moves, inEvents / moves);
        send(QEvent::MouseButtonRelease, at(toVoxel), Qt::LeftButton, Qt::NoButton);
        QCoreApplication::processEvents();
        QCoreApplication::sendPostedEvents();
        QCoreApplication::processEvents();
        qInfo("stroke: %d moves from (%.0f, %.0f) to (%.0f, %.0f) in %lld ms", moves, fromVoxel.x(), fromVoxel.y(), toVoxel.x(),
              toVoxel.y(), static_cast<long long>(clock.elapsed()));
    }

    void ViewerWidget::syntheticWheel(const QPointF& atVoxel, double steps) {
        SlicePane* pane = impl_->xy;
        if (!pane || !impl_->model.valid()) return;
        const QPointF pos = pane->toScreen(atVoxel);
        QElapsedTimer clock;
        clock.start();
        QWheelEvent e(pos, pane->mapToGlobal(pos), QPoint(), QPoint(0, static_cast<int>(std::lround(steps * 120.0))), Qt::NoButton,
                      Qt::NoModifier, Qt::NoScrollPhase, false);
        QCoreApplication::sendEvent(pane, &e);
        QCoreApplication::processEvents();
        QCoreApplication::sendPostedEvents();
        QCoreApplication::processEvents();
        qInfo("wheel: %.1f steps at (%.0f, %.0f) in %lld ms", steps, atVoxel.x(), atVoxel.y(), static_cast<long long>(clock.elapsed()));
    }

    QString ViewerWidget::cursorText() const { return impl_->cursorText; }
    QString ViewerWidget::zoomText() const { return impl_->zoomText; }

} // namespace sirius::app
