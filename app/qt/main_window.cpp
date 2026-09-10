#include "qt/main_window.hpp"

#include <algorithm>
#include <exception>
#include <fstream>
#include <vector>

#include <QAbstractSlider>
#include <QAction>
#include <QActionGroup>
#include <QApplication>
#include <QBoxLayout>
#include <QCloseEvent>
#include <QDesktopServices>
#include <QMimeData>
#include <QDragEnterEvent>
#include <QDockWidget>
#include <QPushButton>
#include <QWindow>
#include <QMouseEvent>
#include <QDir>
#include <QElapsedTimer>
#include <QFileDialog>
#include <QTimer>
#include <QFontMetrics>
#include <QFileInfo>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QTextEdit>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPainter>
#include <QProgressBar>
#include <QScreen>
#include <QSettings>
#include <QStatusBar>
#include <QUrl>

#include <sirius/device.hpp>
#include <sirius/tiff_io.hpp>

#include "core/export.hpp"
#include "qt/viewer/slice_pane.hpp"
#include "qt/dialogs/export_dialog.hpp"
#include "qt/dialogs/training_export_dialog.hpp"
#include "qt/dialogs/folder_dataset_dialog.hpp"
#include "qt/dialogs/model_hub_dialog.hpp"
#include "qt/dialogs/open_dataset_dialog.hpp"
#include "qt/dialogs/plugin_manager.hpp"
#include "qt/dialogs/preferences_dialog.hpp"
#include "qt/panels/assistant_panel.hpp"
#include "qt/panels/diagnostics_panel.hpp"
#include "qt/panels/help_window.hpp"
#include "qt/panels/log_panel.hpp"
#include "qt/panels/ops_panel.hpp"
#include "qt/panels/params_panel.hpp"
#include "qt/qt_strings.hpp"
#include "qt/shortcuts.hpp"
#include "qt/theme.hpp"
#include "qt/trace.hpp"
#include "qt/viewer/viewer_widget.hpp"
#include "qt/widgets/controls.hpp"

namespace sirius::app {

    namespace {

        constexpr const char* kIssuesUrl = "https://github.com/abcucberkeley/sirius/issues";

        // "✦ Assistant" toggle: 26 px, 1.5 px border, accent fill when open.
        // (The sparkle is painted, like every other icon: see widgets/icons.hpp.)
        class AssistantButton : public QAbstractButton {
        public:
            explicit AssistantButton(QWidget* parent) : QAbstractButton(parent) {
                setCheckable(true);
                setFixedHeight(26);
                setCursor(Qt::PointingHandCursor);
                setToolTip(withShortcut(QStringLiteral("Assistant"), keys::assistant()));
                setAccessibleName(QStringLiteral("Assistant"));
                setAccessibleDescription(QStringLiteral("Show or hide the assistant panel"));
                setFont(theme::heading(12));
                setFixedWidth(QFontMetrics(font()).horizontalAdvance(QStringLiteral("Assistant")) + 40);
            }

        protected:
            void paintEvent(QPaintEvent*) override {
                QPainter p(this);
                const bool on = isChecked();
                const QColor border = on ? theme::kAccent : (underMouse() ? theme::kAccent : theme::kText);
                p.setPen(QPen(border, 1.5));
                p.setBrush(on ? QBrush(theme::kAccent) : Qt::NoBrush);
                p.drawRect(QRectF(rect()).adjusted(0.75, 0.75, -0.75, -0.75));
                const QColor fg = on ? theme::kBg : theme::kText;
                widgets::drawIcon(p, QRectF(10, (height() - 13) / 2.0, 13, 13), widgets::Icon::Sparkle, fg);
                p.setPen(fg);
                p.setFont(theme::heading(12));
                p.drawText(rect().adjusted(27, 0, -10, 0), Qt::AlignVCenter | Qt::AlignLeft, QStringLiteral("Assistant"));
            }
            void enterEvent(QEnterEvent*) override { update(); }
            void leaveEvent(QEvent*) override { update(); }
        };

        // The title bar of a floating dock: a handle that moves the window
        // through the compositor (Qt's own drag moves nothing on Wayland),
        // Dock to put it back, and close. Docked panels carry their own
        // headers and show no title bar at all.
        class FloatingTitle : public QWidget {
        public:
            FloatingTitle(QDockWidget* dock, const QString& name) : QWidget(dock), dock_(dock) {
                setFixedHeight(30);
                setCursor(Qt::SizeAllCursor);
                setAutoFillBackground(true);
                QPalette pal = palette();
                pal.setColor(QPalette::Window, theme::kSurface);
                setPalette(pal);
                auto* h = new QHBoxLayout(this);
                h->setContentsMargins(12, 0, 6, 0);
                h->setSpacing(6);
                h->addWidget(widgets::label(name.toUpper(), 11, theme::kNeutral700, QFont::DemiBold, this));
                h->addWidget(widgets::label(QStringLiteral("drag to move · double-click or Dock to put back"), 11, theme::kNeutral600, -1, this), 1);
                auto* back = new QPushButton(QStringLiteral("Dock"), this);
                widgets::setButtonClass(back, "ghost small");
                back->setCursor(Qt::PointingHandCursor);
                QObject::connect(back, &QPushButton::clicked, dock, [dock] { dock->setFloating(false); });
                auto* close = new QPushButton(QStringLiteral("×"), this);
                widgets::setButtonClass(close, "ghost small");
                close->setCursor(Qt::PointingHandCursor);
                QObject::connect(close, &QPushButton::clicked, dock, &QDockWidget::close);
                h->addWidget(back);
                h->addWidget(close);
            }

        protected:
            void mousePressEvent(QMouseEvent* e) override {
                if (e->button() != Qt::LeftButton) return;
                if (QWindow* w = dock_->windowHandle(); w && w->startSystemMove()) return;
                drag_ = e->globalPosition().toPoint() - dock_->frameGeometry().topLeft();   // platforms without a system move
                dragging_ = true;
            }
            void mouseMoveEvent(QMouseEvent* e) override {
                if (dragging_) dock_->move(e->globalPosition().toPoint() - drag_);
            }
            void mouseReleaseEvent(QMouseEvent*) override { dragging_ = false; }
            void mouseDoubleClickEvent(QMouseEvent*) override { dock_->setFloating(false); }

        private:
            QDockWidget* dock_;
            QPoint drag_;
            bool dragging_ = false;
        };

        // The docked "title bar": a 10 px strip above the panel's own header
        // carrying the design's flat grip rule. Qt implements dock dragging in
        // QDockWidget's mouse handlers over whatever the title bar widget
        // occupies, so the strip has to have a height (a 0 px one left the
        // docks impossible to drag) and has to let the press through: it
        // ignores every button event so the dock widget below sees it and
        // starts its own drag, with drop indicators and all.
        class DockGrip : public QWidget {
        public:
            DockGrip(QDockWidget* dock, const QString& name) : QWidget(dock) {
                setFixedHeight(10);
                setCursor(Qt::SizeAllCursor);
                setToolTip(QStringLiteral("Drag to move the %1 panel; double-click to undock").arg(name.toLower()));
                setAccessibleName(QStringLiteral("%1 panel handle").arg(name));
            }

        protected:
            void paintEvent(QPaintEvent*) override {
                QPainter p(this);
                p.fillRect(rect(), theme::kBg);
                const int w = 30;
                p.fillRect(QRect((width() - w) / 2, 4, w, 2), hover_ ? theme::kNeutral600 : theme::kNeutral400);
            }
            void mousePressEvent(QMouseEvent* e) override { e->ignore(); }        // -> QDockWidget: drag
            void mouseDoubleClickEvent(QMouseEvent* e) override { e->ignore(); }   // -> QDockWidget: float
            void enterEvent(QEnterEvent*) override {
                hover_ = true;
                update();
            }
            void leaveEvent(QEvent*) override {
                hover_ = false;
                update();
            }

        private:
            bool hover_ = false;
        };

        QDockWidget* makeDock(const QString& name, QWidget* content, QMainWindow* window) {
            auto* dock = new QDockWidget(name, window);
            dock->setObjectName(name);
            dock->setWidget(content);
            dock->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable |
                              QDockWidget::DockWidgetClosable);
            // No title bar chrome while docked beyond the grip: the panels
            // carry their own headers.
            auto* title = new DockGrip(dock, name);
            dock->setTitleBarWidget(title);
            auto* floating = new FloatingTitle(dock, name);
            floating->hide();
            QObject::connect(dock, &QDockWidget::topLevelChanged, dock, [dock, title, floating](bool top) {
                dock->setTitleBarWidget(top ? static_cast<QWidget*>(floating) : title);
                (top ? static_cast<QWidget*>(floating) : title)->show();
                // shadow-lg while it floats, nothing while it is docked.
                if (top) widgets::applyShadow(dock, true);
                else widgets::clearShadow(dock);
                if (top) {
                    // a floating panel starts as a real window of a useful size
                    dock->setWindowFlags(dock->windowFlags() | Qt::Window);
                    dock->show();
                    if (dock->width() < 360 || dock->height() < 240) dock->resize(std::max(dock->width(), 480), std::max(dock->height(), 360));
                    dock->raise();
                    dock->activateWindow();
                }
            });
            return dock;
        }

        QString gpuText(const Workbench& wb) {
            if (!cudaAvailable()) return QStringLiteral("CPU only");
            try {
                const DeviceProperties p = deviceProperties(Device::cuda(wb.cudaDevice()));
                return QStringLiteral("GPU · %1 · %2 GB")
                    .arg(fromStd(p.name))
                    .arg(static_cast<double>(p.totalMemoryBytes) / (1024.0 * 1024.0 * 1024.0), 0, 'f', 1);
            } catch (const std::exception&) {
                return QStringLiteral("GPU");
            }
        }

    } // namespace

    struct MainWindow::Impl {
        MainWindow* self;
        WorkbenchBridge& bridge;
        bool unattended = false;   // see MainWindow::setUnattended
        // The history's size when the pipeline was last saved or loaded:
        // anything beyond it is unsaved (label edits are never saved by
        // File > Save; they leave through an export).
        std::size_t savedHistory = 0;

        ViewerWidget* viewer = nullptr;
        OpsPanel* ops = nullptr;
        ParamsPanel* params = nullptr;
        DiagnosticsPanel* diagnostics = nullptr;
        AssistantPanel* assistant = nullptr;
        LogPanel* log = nullptr;
        HelpWindow* help = nullptr;
        PluginManagerDialog* plugins = nullptr;   // created on first use
        QDockWidget* opsDock = nullptr;
        QDockWidget* paramsDock = nullptr;
        QDockWidget* diagDock = nullptr;
        QDockWidget* assistantDock = nullptr;
        QDockWidget* logDock = nullptr;

        QLabel* datasetLabel = nullptr;
        QLabel* gpuLabel = nullptr;
        AssistantButton* assistantButton = nullptr;

        QLabel* statusShape = nullptr;
        QLabel* statusDtype = nullptr;
        QLabel* statusZoom = nullptr;
        QLabel* statusCursor = nullptr;
        QWidget* statusProgress = nullptr;
        QProgressBar* progressBar = nullptr;
        QLabel* progressText = nullptr;
        QLabel* statusLog = nullptr;     // the last log line, cleared after a moment
        QTimer statusLogTimer;

        // A temporary QStatusBar message would hide the whole left block,
        // progress bar included, for as long as the worker keeps logging.
        // The line is a button onto the log dock: four seconds is not long
        // enough to read an error, and the full history lives there.
        void showLogLine(const QString& line, int ms) {
            const QFontMetrics fm(statusLog->font());
            statusLog->setText(fm.elidedText(line.simplified(), Qt::ElideRight, 640));
            statusLog->setToolTip(line.simplified() + QStringLiteral("\n\nClick to open the log (%1)")
                                                          .arg(shortcutText(keys::logDock())));
            statusLogTimer.start(ms);
        }

        void showLog() {
            logDock->show();
            logDock->raise();
            log->showLatest();
        }
        QElapsedTimer progressClock;   // since the run / task started, for the estimate
        QString progressMessage;       // the step's last message ("cellpose 2/4")

        // "53 % · ~40 s left · cellpose cpsam_v2" from the fraction, the clock
        // and the last message; the estimate waits until 3 % is done.
        void showProgress(double f, const QString& message) {
            statusProgress->show();
            progressBar->setValue(static_cast<int>(std::clamp(f, 0.0, 1.0) * 1000.0));
            if (!message.isEmpty()) progressMessage = message;
            QString text = QStringLiteral("%1 %").arg(static_cast<int>(f * 100.0 + 0.5));
            const double elapsed = progressClock.isValid() ? progressClock.elapsed() / 1000.0 : 0.0;
            if (f >= 0.03 && f < 1.0 && elapsed >= 2.0) {
                const double left = elapsed * (1.0 - f) / f;
                text += QStringLiteral(" · ~%1 left").arg(durationText(left));
            } else if (elapsed >= 2.0) {
                text += QStringLiteral(" · %1").arg(durationText(elapsed));
            }
            if (!progressMessage.isEmpty()) text += QStringLiteral(" · ") + progressMessage;
            progressText->setText(text);
        }

        static QString durationText(double seconds) {
            if (seconds < 60.0) return QStringLiteral("%1 s").arg(static_cast<int>(seconds + 0.5));
            const int m = static_cast<int>(seconds / 60.0), sec = static_cast<int>(seconds + 0.5) % 60;
            if (m < 60) return QStringLiteral("%1:%2 min").arg(m).arg(sec, 2, 10, QLatin1Char('0'));
            return QStringLiteral("%1 h %2 min").arg(m / 60).arg(m % 60);
        }
        QLabel* statusRight = nullptr;

        QMenu* recentMenu = nullptr;
        QAction* undo = nullptr;
        QAction* redo = nullptr;
        QAction* pasteParams = nullptr;
        QAction* cancelRun = nullptr;
        QAction* viewOrtho = nullptr;
        QAction* view3d = nullptr;
        QAction* viewCompare = nullptr;
        QAction* crosshair = nullptr;
        QAction* labels = nullptr;
        QAction* soloLabel = nullptr;
        QAction* recordSession = nullptr;
        QAction* scaleBar = nullptr;
        QAction* physicalZ = nullptr;
        QAction* syncZT = nullptr;
        QAction* backendCuda = nullptr;
        QAction* backendCpu = nullptr;
        QAction* backendHpc = nullptr;
        QAction* helpPage = nullptr;
        QAction* assistantAction = nullptr;
        QAction* logAction = nullptr;
        QAction* maximiseDiag = nullptr;
        int diagRestoreHeight = theme::kDiagnosticsH;   // before ⛶ took the viewer's room
        QAction* enableStep = nullptr;
        QAction* removeStep = nullptr;
        QAction* duplicateStep = nullptr;
        QAction* moveUp = nullptr;
        QAction* moveDown = nullptr;
        QAction* runSelected = nullptr;
        QAction* runAllAct = nullptr;
        QAction* runTo = nullptr;
        QAction* clearCache = nullptr;
        QAction* clearAll = nullptr;
        QAction* exportResult = nullptr;
        QAction* exportTraining = nullptr;
        QAction* exportPython = nullptr;
        QAction* exportFigure = nullptr;
        QAction* savePipeline = nullptr;
        QAction* savePipelineAs = nullptr;
        QAction* closeDataset = nullptr;
        std::vector<QAction*> labelEdits;   // label mutations, refused during a run
        QByteArray defaultState;
        QString lastDir;

        Impl(MainWindow* s, WorkbenchBridge& b) : self(s), bridge(b) {}
        Workbench& wb() { return bridge.wb(); }

        // --- building -----------------------------------------------------------
        // Space, H, L, B, E, the digits, Left / Right and Backspace are
        // window-wide shortcuts with no modifier. Qt asks the focus widget
        // first (ShortcutOverride) and every stock text entry claims the keys
        // it types, but a painted widget of ours might not, so the unmodified
        // ones check for themselves that no text field is being typed into.
        static bool typingSomewhere() {
            QWidget* w = QApplication::focusWidget();
            if (!w || !w->isEnabled()) return false;
            if (auto* line = qobject_cast<QLineEdit*>(w)) return !line->isReadOnly();
            if (auto* plain = qobject_cast<QPlainTextEdit*>(w)) return !plain->isReadOnly();
            if (auto* rich = qobject_cast<QTextEdit*>(w)) return !rich->isReadOnly();
            return w->inherits("QAbstractSpinBox");
        }

        // The focused widget uses the key itself (a button is pressed with
        // Space, a slider and a slice pane move with the arrows and the page
        // keys), so the window-wide shortcut on it stands back.
        static bool focusClaims(const QKeySequence& key) {
            if (typingSomewhere()) return true;
            if (key.count() != 1) return false;
            QWidget* w = QApplication::focusWidget();
            if (!w || !w->isEnabled()) return false;
            const int k = key[0].key();
            if (k == Qt::Key_Space || k == Qt::Key_Return || k == Qt::Key_Enter) return qobject_cast<QAbstractButton*>(w) != nullptr;
            const bool navigation = k == Qt::Key_Left || k == Qt::Key_Right || k == Qt::Key_Up || k == Qt::Key_Down ||
                                    k == Qt::Key_PageUp || k == Qt::Key_PageDown;
            return navigation && (qobject_cast<QAbstractSlider*>(w) != nullptr || qobject_cast<SlicePane*>(w) != nullptr);
        }

        static bool unmodified(const QKeySequence& key) {
            if (key.count() != 1) return false;
            return (key[0].keyboardModifiers() & ~Qt::KeypadModifier) == Qt::NoModifier;
        }

        QAction* action(QMenu* menu, const QString& text, const QKeySequence& key, std::function<void()> fn,
                        const QString& tip = {}) {
            auto* a = menu->addAction(text);
            if (!key.isEmpty()) a->setShortcut(key);
            a->setShortcutContext(Qt::WindowShortcut);
            if (!tip.isEmpty()) a->setStatusTip(tip);
            if (fn) {
                const bool guard = unmodified(key);
                QObject::connect(a, &QAction::triggered, self, [fn, guard, key] {
                    if (guard && focusClaims(key)) return;
                    fn();
                });
            }
            return a;
        }

        void buildMenus() {
            auto* bar = new QMenuBar(self);
            bar->setNativeMenuBar(false);
            bar->setFixedHeight(theme::kTitleBarH);
            self->setMenuBar(bar);

            // brand
            auto* brand = new QWidget(bar);
            auto* bl = new QHBoxLayout(brand);
            bl->setContentsMargins(14, 0, 10, 0);
            bl->setSpacing(8);
            bl->addWidget(widgets::colorChip(theme::kAccent, 12, 12, brand));
            bl->addWidget(widgets::heading(QStringLiteral("SIRIUS"), theme::kBrandPx, brand));
            bar->setCornerWidget(brand, Qt::TopLeftCorner);

            // right: dataset · GPU · ✦ Assistant
            auto* right = new QWidget(bar);
            auto* rl = new QHBoxLayout(right);
            rl->setContentsMargins(0, 0, 14, 0);
            rl->setSpacing(18);
            datasetLabel = widgets::label(QString(), 12, theme::kNeutral600, -1, right);
            gpuLabel = widgets::label(gpuText(wb()), 12, theme::kNeutral600, -1, right);
            assistantButton = new AssistantButton(right);
            rl->addWidget(datasetLabel);
            rl->addWidget(gpuLabel);
            rl->addWidget(assistantButton);
            bar->setCornerWidget(right, Qt::TopRightCorner);

            // File
            QMenu* file = bar->addMenu(QStringLiteral("File"));
            action(file, QStringLiteral("Open dataset…"), QKeySequence::Open, [this] { openDataset(); });
            action(file, QStringLiteral("Open folder as dataset…"), QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_O),
                   [this] { openFolderDataset(); });
            recentMenu = file->addMenu(QStringLiteral("Open recent"));
            closeDataset = action(file, QStringLiteral("Close dataset"), QKeySequence::Close, [this] { wb().closeDataset(); });
            file->addSeparator();
            savePipeline = action(file, QStringLiteral("Save pipeline"), QKeySequence::Save, [this] { savePipelineTo(fromStd(wb().pipelinePath())); });
            savePipelineAs = action(file, QStringLiteral("Save pipeline as…"), QKeySequence::SaveAs, [this] { savePipelineTo(QString()); });
            action(file, QStringLiteral("Load pipeline preset…"), QKeySequence(), [this] { loadPipeline(); });
            file->addSeparator();
            exportResult = action(file, QStringLiteral("Export result…"), QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_E),
                                  [this] { exportResultDialog(); });
            exportTraining = action(file, QStringLiteral("Export training data…"), QKeySequence(),
                                    [this] { exportTrainingDialog(); });
            exportTraining->setStatusTip(QStringLiteral("Write the labels of a step as instance masks, bounding boxes and a "
                                                        "semantic mask into a dataset folder"));
            exportPython = action(file, QStringLiteral("Export pipeline as Python script…"), QKeySequence(),
                                  [this] { exportPythonScript(); });
            file->addSeparator();
            recordSession = action(file, QStringLiteral("Record session…"), QKeySequence(), [this] { toggleRecording(); });
            recordSession->setStatusTip(QStringLiteral("Write what you do to a JSON-lines file: steps, parameter changes, "
                                                       "run results and label corrections"));
            exportFigure = action(file, QStringLiteral("Export figure (current view)…"), QKeySequence(Qt::ALT | Qt::CTRL | Qt::Key_E),
                                  [this] { exportFigureImage(); });
            file->addSeparator();
            action(file, QStringLiteral("Preferences…"), QKeySequence::Preferences, [this] { preferences(); });
            file->addSeparator();
            action(file, QStringLiteral("Quit"), QKeySequence::Quit, [this] { self->close(); });

            // Edit
            QMenu* edit = bar->addMenu(QStringLiteral("Edit"));
            undo = action(edit, QStringLiteral("Undo"), QKeySequence::Undo, [this] { wb().undo(); });
            redo = action(edit, QStringLiteral("Redo"), QKeySequence::Redo, [this] { wb().redo(); });
            edit->addSeparator();
            duplicateStep = action(edit, QStringLiteral("Duplicate step"), keys::duplicateStep(),
                                   [this] { wb().duplicateStep(wb().selectedIndex()); });
            removeStep = action(edit, QStringLiteral("Remove step"), keys::removeStep(),
                                [this] { removeStepAt(wb().selectedIndex()); });
            enableStep = action(edit, QStringLiteral("Enable / skip step"), keys::enableStep(), [this] {
                const int i = wb().selectedIndex();
                if (i > 0 && i < wb().pipeline().size()) wb().setStepEnabled(i, !wb().pipeline().at(i).enabled);
            });
            edit->addSeparator();
            moveUp = action(edit, QStringLiteral("Move step up"), keys::moveUp(),
                            [this] { wb().moveStep(wb().selectedIndex(), -1); });
            moveDown = action(edit, QStringLiteral("Move step down"), keys::moveDown(),
                              [this] { wb().moveStep(wb().selectedIndex(), +1); });
            edit->addSeparator();
            action(edit, QStringLiteral("Copy parameters"), QKeySequence::Copy, [this] { wb().copyParameters(wb().selectedIndex()); });
            pasteParams = action(edit, QStringLiteral("Paste parameters"), QKeySequence::Paste, [this] {
                if (!wb().pasteParameters(wb().selectedIndex()))
                    wb().logLine("Paste parameters: the copied parameters belong to another operation kind.");
            });

            // View
            QMenu* view = bar->addMenu(QStringLiteral("View"));
            auto* modes = new QActionGroup(self);
            viewOrtho = action(view, QStringLiteral("Ortho views"), QKeySequence(Qt::Key_1), [this] { wb().setViewMode(ViewMode::Ortho); });
            view3d = action(view, QStringLiteral("3D volume"), QKeySequence(Qt::Key_2), [this] { wb().setViewMode(ViewMode::Volume); });
            viewCompare = action(view, QStringLiteral("Compare raw vs. step"), QKeySequence(Qt::Key_3),
                                 [this] { wb().setViewMode(ViewMode::Compare); });
            for (QAction* a : {viewOrtho, view3d, viewCompare}) {
                a->setCheckable(true);
                modes->addAction(a);
            }
            view->addSeparator();
            crosshair = action(view, QStringLiteral("Crosshair"), QKeySequence(Qt::Key_H), [this] { wb().toggleCrosshair(); });
            crosshair->setCheckable(true);
            labels = action(view, QStringLiteral("Labels overlay"), QKeySequence(Qt::Key_L), [this] { wb().toggleLabels(); });
            labels->setCheckable(true);
            scaleBar = action(view, QStringLiteral("Scale bar"), QKeySequence(), [this] {
                ViewState s = wb().viewState();
                s.scaleBar = !s.scaleBar;
                wb().setViewState(s);
            });
            scaleBar->setCheckable(true);
            physicalZ = action(view, QStringLiteral("Physical z scaling"), QKeySequence(), [this] {
                ViewState s = wb().viewState();
                s.physicalZ = !s.physicalZ;
                wb().setViewState(s);
            });
            physicalZ->setCheckable(true);
            physicalZ->setStatusTip(QStringLiteral("Scale the XZ / YZ panes by the voxel aspect. Off draws one row per "
                                                   "plane, for checking the grid rather than the shape"));
            view->addSeparator();
            action(view, QStringLiteral("Zoom in"), QKeySequence(Qt::Key_Plus), [this] { viewer->zoomIn(); });
            action(view, QStringLiteral("Zoom out"), QKeySequence(Qt::Key_Minus), [this] { viewer->zoomOut(); });
            action(view, QStringLiteral("Fit to window"), QKeySequence(Qt::Key_0), [this] { viewer->fitToWindow(); });
            view->addSeparator();
            action(view, QStringLiteral("Auto contrast (display)"), QKeySequence(Qt::SHIFT | Qt::Key_A),
                   [this] { viewer->autoContrast(); });
            action(view, QStringLiteral("Reset contrast (display)"), QKeySequence(Qt::SHIFT | Qt::Key_R),
                   [this] { viewer->resetContrast(); });
            syncZT = action(view, QStringLiteral("Sync Z / T across viewers"), QKeySequence(), [this] {
                ViewState s = wb().viewState();
                s.syncZT = !s.syncZT;
                wb().setViewState(s);
            });
            syncZT->setCheckable(true);

            // Process
            QMenu* process = bar->addMenu(QStringLiteral("Process"));
            action(process, QStringLiteral("Add operation…"), QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_A), [this] {
                opsDock->show();
                ops->openAddMenu();
            });
            process->addSeparator();
            runAllAct = action(process, QStringLiteral("Run all enabled"), keys::runAll(), [this] { bridge.startRun(-1); });
            runSelected = action(process, QStringLiteral("Run selected step"), keys::runSelected(), [this] { runSelectedStep(); }, QStringLiteral("Run just this step; its input has to be computed already"));
            runTo = action(process, QStringLiteral("Run to selected step"), QKeySequence(), [this] { bridge.startRun(wb().selectedIndex()); }, QStringLiteral("Run every enabled step from the top down to this one"));
            cancelRun = action(process, QStringLiteral("Cancel"), QKeySequence(Qt::Key_Escape), [this] {
                if (bridge.running()) bridge.cancelRun();
                if (bridge.taskRunning()) bridge.cancelTask();
            });
            process->addSeparator();
            clearCache = action(process, QStringLiteral("Clear cache for step"), QKeySequence(), [this] { wb().clearCache(wb().selectedIndex()); });
            clearAll = action(process, QStringLiteral("Clear all caches"), QKeySequence(), [this] { wb().clearAllCaches(); });
            process->addSeparator();
            action(process, QStringLiteral("Reload plugins"), QKeySequence(), [this] { wb().loadPlugins(true); });
            process->addSeparator();
            auto* backends = new QActionGroup(self);
            backendCuda = action(process, QStringLiteral("Backend: CUDA"), QKeySequence(), [this] { wb().setBackend(Backend::Cuda); });
            backendCpu = action(process, QStringLiteral("Backend: CPU"), QKeySequence(), [this] { wb().setBackend(Backend::Cpu); });
            backendHpc = action(process, QStringLiteral("Backend: HPC (Slurm)"), QKeySequence(), [this] { wb().setBackend(Backend::Hpc); });
            for (QAction* a : {backendCuda, backendCpu, backendHpc}) {
                a->setCheckable(true);
                backends->addAction(a);
            }
            backendCuda->setEnabled(cudaAvailable());
            if (!cudaAvailable()) backendCuda->setStatusTip(QStringLiteral("No CUDA device available"));

            // Segment
            QMenu* segment = bar->addMenu(QStringLiteral("Segment"));
            action(segment, QStringLiteral("Load Torch model…"), QKeySequence(Qt::CTRL | Qt::Key_M), [this] { loadTorchModel(); });
            action(segment, QStringLiteral("Download model…"), QKeySequence(), [this] { modelHub(); });
            action(segment, QStringLiteral("Run segmentation"), QKeySequence(), [this] {
                const int i = segmentationStep();
                if (i < 0) wb().logLine("Run segmentation: add a segmentation step first (Process ▸ Add operation).");
                else bridge.startRun(i);
            });
            segment->addSeparator();
            action(segment, QStringLiteral("Paint labels"), QKeySequence(Qt::Key_B), [this] {
                wb().setTool(ViewerTool::Paint);
                wb().setPaintTool(PaintTool::Brush);
                if (!wb().viewState().labels) wb().toggleLabels();
            });
            action(segment, QStringLiteral("Erase"), QKeySequence(Qt::Key_E), [this] {
                wb().setTool(ViewerTool::Paint);
                wb().setPaintTool(PaintTool::Erase);
            });
            labelEdits.push_back(
                action(segment, QStringLiteral("Merge selected labels"), QKeySequence(Qt::CTRL | Qt::Key_G), [this] { mergeLabelsDialog(); }));
            action(segment, QStringLiteral("Split label"), QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_G), [this] {
                wb().setTool(ViewerTool::Paint);
                wb().setPaintTool(PaintTool::Split);
                wb().logLine("Split: click two seeds inside the label in the viewer.");
            });
            labelEdits.push_back(action(segment, QStringLiteral("Delete label"), QKeySequence(Qt::CTRL | Qt::Key_Backspace), [this] {
                const std::uint32_t id = wb().viewState().selectedLabel;
                if (id) wb().deleteLabel(id);
            }));
            segment->addSeparator();
            soloLabel = action(segment, QStringLiteral("Only selected label"), QKeySequence(Qt::Key_O), [this] { wb().toggleSoloLabel(); });
            soloLabel->setCheckable(true);
            soloLabel->setStatusTip(QStringLiteral("Draw only the selected label in the slices and in 3D; selecting a label jumps to it"));
            action(segment, QStringLiteral("Next flagged label"), QKeySequence(Qt::Key_Right), [this] { selectFlagged(true); });
            action(segment, QStringLiteral("Previous flagged label"), QKeySequence(Qt::Key_Left), [this] { selectFlagged(false); });
            segment->addSeparator();
            labelEdits.push_back(
                action(segment, QStringLiteral("Accept all reviewed"), QKeySequence(), [this] { wb().acceptAllReviewed(); }));
            segment->addSeparator();
            action(segment, QStringLiteral("Export labels…"), QKeySequence(), [this] { exportLabels(); });

            // Window
            QMenu* window = bar->addMenu(QStringLiteral("Window"));
            QAction* opsToggle = opsDock->toggleViewAction();
            opsToggle->setText(QStringLiteral("Operations"));
            opsToggle->setShortcut(QKeySequence(Qt::ALT | Qt::Key_1));
            QAction* paramsToggle = paramsDock->toggleViewAction();
            paramsToggle->setText(QStringLiteral("Parameters"));
            paramsToggle->setShortcut(QKeySequence(Qt::ALT | Qt::Key_2));
            QAction* diagToggle = diagDock->toggleViewAction();
            diagToggle->setText(QStringLiteral("Diagnostics"));
            diagToggle->setShortcut(QKeySequence(Qt::ALT | Qt::Key_3));
            window->addAction(opsToggle);
            window->addAction(paramsToggle);
            window->addAction(diagToggle);
            helpPage = action(window, QStringLiteral("Help page"), QKeySequence(Qt::ALT | Qt::Key_4), [this] { toggleHelp(); });
            helpPage->setCheckable(true);
            window->addSeparator();
            action(window, QStringLiteral("Float diagnostics"), QKeySequence(), [this] {
                diagDock->show();
                diagDock->setFloating(true);
            });
            // The ⛶ control of the diagnostics header, from the keyboard.
            maximiseDiag = action(window, QStringLiteral("Maximise diagnostics"), QKeySequence(), [this] {
                diagDock->show();
                diagDock->raise();
                diagnostics->setMaximized(!diagnostics->isMaximized());
            });
            maximiseDiag->setCheckable(true);
            maximiseDiag->setStatusTip(QStringLiteral("Let the diagnostics cover the viewer"));
            action(window, QStringLiteral("Reset layout"), QKeySequence(), [this] {
                self->restoreState(defaultState);
                wb().logLine("Layout reset to the default arrangement.");
            });
            // No dialog behind this one, so no ellipsis: it stores the
            // arrangement as it is, which is also what closing the window does.
            action(window, QStringLiteral("Save layout as default"), QKeySequence(), [this] {
                QSettings settings;
                settings.setValue(QStringLiteral("window/state"), self->saveState());
                settings.setValue(QStringLiteral("window/geometry"), self->saveGeometry());
                wb().logLine("Saved this layout: the next session opens with it.");
            });
            window->addSeparator();
            assistantAction = assistantDock->toggleViewAction();
            assistantAction->setText(QStringLiteral("Assistant"));
            assistantAction->setShortcut(keys::assistant());
            window->addAction(assistantAction);
            action(window, QStringLiteral("User operations…"), QKeySequence(Qt::ALT | Qt::Key_6), [this] { pluginManager(); });
            logAction = logDock->toggleViewAction();
            logAction->setText(QStringLiteral("Log"));
            logAction->setShortcut(keys::logDock());
            logAction->setStatusTip(QStringLiteral("Everything this session has reported"));
            window->addAction(logAction);

            // Help
            QMenu* helpMenu = bar->addMenu(QStringLiteral("Help"));
            action(helpMenu, QStringLiteral("Help for this step"), QKeySequence(Qt::Key_F1), [this] { showHelpForSelected(); });
            action(helpMenu, QStringLiteral("Sirius manual"), QKeySequence(), [this] {
                help->showManual();
                help->show();
                help->raise();
            });
            action(helpMenu, QStringLiteral("Keyboard shortcuts"), QKeySequence(Qt::CTRL | Qt::Key_Slash), [this] {
                help->showShortcuts();
                help->show();
                help->raise();
            });
            helpMenu->addSeparator();
            action(helpMenu, QStringLiteral("Operation plugin API"), QKeySequence(), [this] {
                help->showKind("plugin-api");
                help->show();
                help->raise();
            });
            action(helpMenu, QStringLiteral("Report a problem…"), QKeySequence(),
                   [] { QDesktopServices::openUrl(QUrl(QLatin1String(kIssuesUrl))); });
            helpMenu->addSeparator();
            action(helpMenu, QStringLiteral("About Sirius"), QKeySequence(), [this] { about(); });

            QObject::connect(assistantButton, &QAbstractButton::toggled, self, [this](bool on) { assistantDock->setVisible(on); });
            QObject::connect(assistantDock, &QDockWidget::visibilityChanged, self, [this](bool on) {
                QSignalBlocker b(assistantButton);
                assistantButton->setChecked(on);
                if (on) assistant->focusInput();
            });
            QObject::connect(recentMenu, &QMenu::aboutToShow, self, [this] { rebuildRecent(); });
        }

        void buildStatusBar() {
            auto* sb = self->statusBar();
            sb->setSizeGripEnabled(false);
            auto* left = new QWidget(sb);
            auto* ll = new QHBoxLayout(left);
            ll->setContentsMargins(14, 0, 0, 0);
            ll->setSpacing(24);
            statusShape = widgets::label(QString(), 11, theme::kNeutral600, -1, left);
            statusDtype = widgets::label(QString(), 11, theme::kNeutral600, -1, left);
            statusZoom = widgets::label(QStringLiteral("zoom 100 %"), 11, theme::kNeutral600, -1, left);
            statusCursor = widgets::label(QString(), 11, theme::kNeutral600, -1, left);
            // The status bar is nothing but numbers that change under a moving
            // cursor: tabular figures keep them from jittering.
            for (QLabel* l : {statusShape, statusDtype, statusZoom, statusCursor}) widgets::useTabularNumbers(l);
            statusProgress = new QWidget(left);
            auto* pl = new QHBoxLayout(statusProgress);
            pl->setContentsMargins(0, 0, 0, 0);
            pl->setSpacing(8);
            progressBar = new QProgressBar(statusProgress);
            progressBar->setRange(0, 1000);
            progressBar->setTextVisible(false);
            progressBar->setFixedSize(160, 4);
            progressText = widgets::label(QStringLiteral("0 %"), 11, theme::kText, -1, statusProgress);
            pl->addWidget(progressBar);
            pl->addWidget(progressText);
            statusProgress->hide();
            statusLog = widgets::label(QString(), 11, theme::kNeutral600, -1, left);
            statusLog->setMaximumWidth(660);
            // The line is here for four seconds; the whole history is one
            // click (or the Window ▸ Log shortcut) away.
            statusLog->setCursor(Qt::PointingHandCursor);
            statusLog->setAccessibleName(QStringLiteral("Last log line"));
            statusLog->setAccessibleDescription(QStringLiteral("Click to open the log"));
            statusLog->installEventFilter(self);
            statusLogTimer.setSingleShot(true);
            QObject::connect(&statusLogTimer, &QTimer::timeout, self, [this] { statusLog->clear(); });
            for (QWidget* w : {static_cast<QWidget*>(statusShape), static_cast<QWidget*>(statusDtype),
                               static_cast<QWidget*>(statusZoom), static_cast<QWidget*>(statusCursor), statusProgress,
                               static_cast<QWidget*>(statusLog)})
                ll->addWidget(w);
            ll->addStretch(1);
            sb->addWidget(left, 1);
            statusRight = widgets::label(QString(), 11, theme::kNeutral600, -1, sb);
            widgets::useTabularNumbers(statusRight);
            statusRight->setContentsMargins(0, 0, 14, 0);
            sb->addPermanentWidget(statusRight);
        }

        // --- state refresh ------------------------------------------------------------
        void refreshTitle() {
            const Workbench& w = wb();
            const QString name = w.hasDataset() ? fromStd(w.dataset().name) : QStringLiteral("no dataset");
            datasetLabel->setText(name);
            gpuLabel->setText(gpuText(w));
            QString title = QStringLiteral("SIRIUS");
            if (w.hasDataset()) title += QStringLiteral(" — ") + name;
            if (!w.pipelinePath().empty()) title += QStringLiteral(" · ") + QFileInfo(fromStd(w.pipelinePath())).fileName();
            self->setWindowTitle(title);
        }

        void refreshStatus() {
            const Workbench& w = wb();
            if (w.hasDataset()) {
                const DatasetMeta& m = w.dataset();
                statusShape->setText(fromStd(m.shapeString()));
                statusDtype->setText(QStringLiteral("%1 → float32").arg(QString::fromLatin1(toString(m.sourceType))));
            } else {
                statusShape->setText(QStringLiteral("no dataset"));
                statusDtype->clear();
            }
            statusZoom->setText(QStringLiteral("zoom %1").arg(viewer->zoomText()));
            statusCursor->setText(viewer->cursorText());
            const Pipeline& p = w.pipeline();
            const bool lazy = !w.output(0) || !w.output(0)->array;
            statusRight->setText(QStringLiteral("%1 of %2 steps enabled · %3 · %4 cached")
                                     .arg(p.enabledCount())
                                     .arg(p.size())
                                     .arg(lazy ? QStringLiteral("lazy") : QStringLiteral("in memory"))
                                     .arg(widgets::bytesText(w.cachedBytes())));
        }

        void refreshActions() {
            const ScopedTrace trace("refreshActions");
            Workbench& w = wb();   // history() is non-const
            const int i = w.selectedIndex();
            const Pipeline& p = w.pipeline();
            const bool stepOk = i >= 0 && i < p.size();
            const bool movable = stepOk && i > 0;
            const bool running = bridge.running();
            // The workbench refuses every edit while a run is active
            // (Workbench::canEdit): the menu has to say so rather than let
            // the user pick an item that quietly does nothing.
            const bool edit = w.canEdit();
            const QString frozen = QStringLiteral("Not while a run is in progress — cancel it (Esc) or wait");
            undo->setEnabled(edit && w.history().canUndo());
            undo->setText(w.history().canUndo() ? QStringLiteral("Undo %1").arg(fromStd(w.history().undoLabel()))
                                                : QStringLiteral("Undo"));
            redo->setEnabled(edit && w.history().canRedo());
            redo->setText(w.history().canRedo() ? QStringLiteral("Redo %1").arg(fromStd(w.history().redoLabel()))
                                                : QStringLiteral("Redo"));
            pasteParams->setEnabled(edit && w.hasCopiedParameters() && stepOk);
            duplicateStep->setEnabled(edit && movable);
            removeStep->setEnabled(edit && movable);
            enableStep->setEnabled(edit && movable);
            moveUp->setEnabled(edit && movable && i > 1);
            moveDown->setEnabled(edit && movable && i < p.size() - 1);
            for (QAction* a : {undo, redo, pasteParams, duplicateStep, removeStep, enableStep, moveUp, moveDown,
                               clearCache, clearAll, closeDataset})
                a->setStatusTip(edit ? QString() : frozen);
            for (QAction* a : labelEdits) {
                a->setEnabled(edit);
                a->setStatusTip(edit ? QString() : frozen);
            }
            runAllAct->setEnabled(w.hasDataset() && !running);
            runSelected->setEnabled(w.hasDataset() && !running && stepOk);
            runTo->setEnabled(w.hasDataset() && !running && stepOk);
            cancelRun->setEnabled(running || bridge.taskRunning());
            clearCache->setEnabled(edit && stepOk);
            clearAll->setEnabled(edit);
            exportResult->setEnabled(w.hasDataset() && !running);
            // It reads a step's output and its labels on the task thread, which
            // a run is free to replace underneath it, so it goes with the rest.
            exportTraining->setEnabled(w.hasDataset() && !running);
            exportPython->setEnabled(true);
            closeDataset->setEnabled(edit && w.hasDataset());
            savePipeline->setEnabled(true);
            const ViewState& v = w.viewState();
            viewOrtho->setChecked(v.mode == ViewMode::Ortho);
            view3d->setChecked(v.mode == ViewMode::Volume);
            viewCompare->setChecked(v.mode == ViewMode::Compare);
            crosshair->setChecked(v.crosshair);
            labels->setChecked(v.labels);
            soloLabel->setChecked(v.soloLabel);
            if (recordSession)
                recordSession->setText(w.recording()
                                           ? QStringLiteral("Stop recording (%1 events)").arg(w.recordedLines())
                                           : QStringLiteral("Record session…"));
            scaleBar->setChecked(v.scaleBar);
            physicalZ->setChecked(v.physicalZ);
            syncZT->setChecked(v.syncZT);
            backendCuda->setChecked(w.backend() == Backend::Cuda);
            backendCpu->setChecked(w.backend() == Backend::Cpu);
            backendHpc->setChecked(w.backend() == Backend::Hpc);
        }

        void rebuildRecent() {
            recentMenu->clear();
            const QStringList recent = OpenDatasetDialog::recentFiles();
            if (recent.isEmpty()) {
                recentMenu->addAction(QStringLiteral("No recent datasets"))->setEnabled(false);
                return;
            }
            for (const QString& path : recent) {
                QAction* a = recentMenu->addAction(QFileInfo(path).fileName());
                a->setToolTip(path);
                QObject::connect(a, &QAction::triggered, self, [this, path] { self->openDatasetPath(path); });
            }
            recentMenu->addSeparator();
            QAction* clear = recentMenu->addAction(QStringLiteral("Clear list"));
            QObject::connect(clear, &QAction::triggered, self, [] {
                QSettings settings;
                settings.remove(QStringLiteral("recent/datasets"));
            });
        }

        // --- commands -----------------------------------------------------------------
        // "Run selected step" against "Run to selected step". The executor
        // only ever walks the pipeline from the top, taking each step's fresh
        // cache entry as it passes (Executor::run), so the one thing that
        // separates the two is whether the steps above are already computed:
        // "Run to" recomputes whatever is stale, "Run selected" is the step
        // on its own and refuses -- out loud -- when its input is missing,
        // instead of quietly running the whole chain under the same label.
        void runSelectedStep() {
            const int i = wb().selectedIndex();
            const Pipeline& p = wb().pipeline();
            if (i < 0 || i >= p.size()) return;
            for (int j = 0; j < i; ++j) {
                if (j > 0 && !p.at(j).enabled) continue;
                if (wb().outputFresh(j)) continue;
                wb().logLine("Run selected step: step " + Step::number(j) + " " + p.at(j).name +
                             " above it is not computed yet, so this step has no input. "
                             "Use Process ▸ Run to selected step (or Run all enabled) instead.");
                return;
            }
            bridge.startRun(i);
        }

        // Backspace removes a step with no dialog in the way, which is right:
        // the removal is one undo entry. What was missing is the evidence, so
        // the log and the status bar name the step and point at Undo. A step
        // holding a computed output is the one case worth a question: undo
        // brings the step back but not its cache, and recomputing it can cost
        // minutes.
        void removeStepAt(int index) {
            const Pipeline& p = wb().pipeline();
            if (index <= 0 || index >= p.size() || !wb().canEdit()) return;
            const Step& step = p.at(index);
            const QString name = QStringLiteral("%1 %2").arg(fromStd(Step::number(index)), fromStd(step.name));
            const std::size_t cached = wb().executor().cachedBytesOf(step.id);
            if (cached > 64ull * 1024 * 1024) {
                const auto answer = QMessageBox::question(
                    self, QStringLiteral("Remove step"),
                    QStringLiteral("Remove step %1?\n\nIts computed output (%2) is discarded and has to be "
                                   "recomputed if you bring the step back.")
                        .arg(name, widgets::bytesText(cached)),
                    QMessageBox::Cancel | QMessageBox::Yes, QMessageBox::Cancel);
                if (answer != QMessageBox::Yes) return;
            }
            wb().removeStep(index);   // logs "Removed step …" itself
            wb().logLine("Edit ▸ Undo (" + toStd(shortcutText(keys::undo())) + ") brings step " + toStd(name) + " back.");
        }

        void openDataset() {
            OpenDatasetDialog dialog(bridge, self, wb().hasDataset() ? fromStd(wb().dataset().sourcePath) : QString());
            if (dialog.exec() != QDialog::Accepted) return;
            openWith(dialog.path(), dialog.options());
        }

        void openWith(const QString& path, const OpenOptions& options) {
            try {
                wb().openDataset(toStd(path), options);
                OpenDatasetDialog::addRecentFile(path);
                lastDir = QFileInfo(path).absolutePath();
            } catch (const std::exception& e) {
                wb().logLine(std::string("Open failed: ") + e.what());
                QMessageBox::warning(self, QStringLiteral("Open dataset"), QString::fromUtf8(e.what()));
            }
        }

        // Start or stop the session recording. The file is JSON lines, so it
        // is appended to and stays readable if the application is killed.
        void toggleRecording() {
            if (wb().recording()) {
                wb().stopRecording();
                refreshActions();
                return;
            }
            QString path = QFileDialog::getSaveFileName(self, QStringLiteral("Record this session to"),
                                                        lastDir.isEmpty() ? QString() : lastDir + QStringLiteral("/session.jsonl"),
                                                        QStringLiteral("Session recording (*.jsonl);;All files (*)"));
            if (path.isEmpty()) return;
            if (!path.contains(QLatin1Char('.'))) path += QStringLiteral(".jsonl");
            try {
                wb().startRecording(toStd(path));
            } catch (const std::exception& e) {
                QMessageBox::warning(self, QStringLiteral("Record session"), QString::fromUtf8(e.what()));
            }
            refreshActions();
        }

        void savePipelineTo(QString path) {
            if (path.isEmpty()) {
                path = QFileDialog::getSaveFileName(self, QStringLiteral("Save pipeline"), lastDir,
                                                    QStringLiteral("SIRIUS pipeline (*.sirius.toml *.toml)"));
                if (path.isEmpty()) return;
                if (!path.endsWith(QLatin1String(".toml"))) path += QStringLiteral(".sirius.toml");
            }
            try {
                wb().savePipeline(toStd(path));
                lastDir = QFileInfo(path).absolutePath();
                savedHistory = wb().history().size();
            } catch (const std::exception& e) {
                QMessageBox::warning(self, QStringLiteral("Save pipeline"), QString::fromUtf8(e.what()));
            }
            refreshTitle();
        }

        bool unsavedWork() { return wb().history().size() != savedHistory; }

        void loadPipeline() {
            const QString path = QFileDialog::getOpenFileName(self, QStringLiteral("Load pipeline"), lastDir,
                                                              QStringLiteral("SIRIUS pipeline (*.sirius.toml *.toml);;All files (*)"));
            if (!path.isEmpty()) self->openPipelinePath(path);
        }

        void exportResultDialog() {
            if (!wb().hasDataset()) return;
            ExportDialog dialog(bridge, self);
            if (dialog.exec() != QDialog::Accepted) return;
            const int step = dialog.stepIndex();
            ExportOptions options = dialog.options();
            std::shared_ptr<const StepOutput> out = wb().output(step);
            if (!out) {
                QMessageBox::information(self, QStringLiteral("Export"),
                                         QStringLiteral("Step %1 has not been computed yet. Run it first.").arg(fromStd(Step::number(step))));
                return;
            }
            const std::string pipelinePath = options.path + ".pipeline.toml";
            const bool sidecar = options.includePipeline;
            options.includePipeline = false;
            if (sidecar) {
                try {
                    wb().savePipeline(pipelinePath);
                } catch (const std::exception& e) {
                    wb().logLine(std::string("Pipeline sidecar: ") + e.what());
                }
            }
            // The labels are copied first: the task reads them on its thread
            // while the viewer may still paint into the step's volume.
            std::shared_ptr<const LabelVolume> labels = out->labels ? out->labels->clone() : nullptr;
            bridge.startTask(QStringLiteral("Export"), [out, labels, options](const WorkbenchBridge::TaskProgress& progress,
                                                                              const WorkbenchBridge::TaskCancelled& cancelled) {
                ArrayPtr array = out->asInput().materialize(progress);
                exportArray(*array, out->meta, labels.get(), options, progress, cancelled);
            });
        }

        void exportTrainingDialog() {
            if (!wb().hasDataset() || bridge.running()) return;
            TrainingExportDialog dialog(bridge, self);
            if (dialog.exec() != QDialog::Accepted) return;
            const int step = dialog.stepIndex();
            std::shared_ptr<const StepOutput> out = wb().output(step);
            if (!out || !out->labels || out->labels->empty()) {
                QMessageBox::information(self, QStringLiteral("Export training data"),
                                         QStringLiteral("Step %1 has no labels. Run a segmentation step first.").arg(fromStd(Step::number(step))));
                return;
            }
            TrainingExportOptions options = dialog.options();
            // the pipeline goes with the sample: it is how the labels were made
            options.provenance = {{"step", Step::number(step)},
                                  {"step_name", wb().pipeline().at(step).name},
                                  {"kind", wb().pipeline().at(step).kind},
                                  {"dataset", wb().hasDataset() ? wb().dataset().sourcePath : std::string()},
                                  {"pipeline", wb().pipeline().toJson()}};
            // a copy: the task reads on its thread while the viewer may paint
            std::shared_ptr<const LabelVolume> labels = out->labels->clone();
            bridge.startTask(QStringLiteral("Export training data"),
                             [out, labels, options](const WorkbenchBridge::TaskProgress& progress,
                                                    const WorkbenchBridge::TaskCancelled& cancelled) {
                                 ArrayPtr array = options.image || options.slices ? out->asInput().materialize(progress) : nullptr;
                                 const Array5 empty;
                                 exportTrainingData(array ? *array : empty, out->meta, *labels, options, progress, cancelled);
                             });
        }

        void exportPythonScript() {
            const QString path = QFileDialog::getSaveFileName(self, QStringLiteral("Export pipeline as Python"), lastDir,
                                                              QStringLiteral("Python (*.py)"));
            if (path.isEmpty()) return;
            const std::string script = wb().pipeline().toPythonScript(wb().hasDataset() ? wb().dataset().sourcePath : std::string());
            std::ofstream f(toStd(path));
            if (!f) {
                QMessageBox::warning(self, QStringLiteral("Export"), QStringLiteral("Cannot write %1").arg(path));
                return;
            }
            f << script;
            wb().logLine("Python script written to " + toStd(path));
        }

        void exportFigureImage() {
            const QImage image = viewer->grabView();
            if (image.isNull()) return;
            const QString path = QFileDialog::getSaveFileName(self, QStringLiteral("Export figure"), lastDir,
                                                              QStringLiteral("PNG (*.png);;TIFF (*.tif)"));
            if (path.isEmpty()) return;
            if (!image.save(path)) QMessageBox::warning(self, QStringLiteral("Export figure"), QStringLiteral("Cannot write %1").arg(path));
            else wb().logLine("Figure written to " + toStd(path));
        }

        void exportLabels() {
            std::shared_ptr<LabelVolume> labelsVol = wb().viewedLabels();
            if (!labelsVol || labelsVol->empty()) {
                wb().logLine("Export labels: the viewed step has no labels.");
                return;
            }
            const QString path = QFileDialog::getSaveFileName(self, QStringLiteral("Export labels"), lastDir,
                                                              QStringLiteral("TIFF (*.tif)"));
            if (path.isEmpty()) return;
            try {
                writeTiffStack<std::uint32_t>(toStd(path), labelsVol->view().asStack(), TiffCompression::Deflate);
                wb().logLine("Labels written to " + toStd(path));
            } catch (const std::exception& e) {
                QMessageBox::warning(self, QStringLiteral("Export labels"), QString::fromUtf8(e.what()));
            }
        }

        void preferences() {
            PreferencesDialog dialog(bridge, self);
            QObject::connect(&dialog, &PreferencesDialog::assistantSettingsChanged, self,
                             [this] { assistant->setSettings(AssistantSettings::load()); });
            dialog.exec();
            refreshTitle();
        }

        int segmentationStep() const {
            const Pipeline& p = bridge.wb().pipeline();
            for (int i = p.size() - 1; i >= 0; --i)
                if (p.at(i).op().info().producesLabels && p.at(i).kind != "threshold") return i;
            for (int i = p.size() - 1; i >= 0; --i)
                if (p.at(i).op().info().producesLabels) return i;
            return -1;
        }

        // The segmentation step the model goes to: the selected one, else the
        // first, else a new one.
        int segmentationStepOrNew() {
            int i = segmentationStep();
            if (i < 0 || wb().pipeline().at(i).kind != "seg") {
                if (!findOperation("seg")) return -1;
                i = wb().pipeline().indexOf(wb().addStep("seg"));
            }
            return i;
        }

        void modelHub() {
            ModelHubDialog dialog(bridge, self);
            if (dialog.exec() != QDialog::Accepted || dialog.chosenModel().isEmpty()) return;
            const int i = segmentationStepOrNew();
            if (i < 0) return;
            wb().setStepParam(i, "model", toStd(dialog.chosenModel()));
            wb().select(i);
        }

        // A folder with a manifest opens directly; otherwise the pattern
        // dialog builds one first.
        void openFolderDataset() {
            const QString folder = QFileDialog::getExistingDirectory(self, QStringLiteral("Open folder as dataset"), lastDir);
            if (folder.isEmpty()) return;
            lastDir = folder;
            if (isFolderDataset(toStd(folder))) {
                openWith(folder, OpenOptions{});
                return;
            }
            FolderDatasetDialog dialog(bridge, folder, self);
            dialog.exec();
        }

        void pluginManager(const QString& file = {}) {
            if (!plugins) {
                plugins = new PluginManagerDialog(bridge, self);
                plugins->setAttribute(Qt::WA_DeleteOnClose, false);
            }
            if (!file.isEmpty()) plugins->openFile(file);
            plugins->show();
            plugins->raise();
            plugins->activateWindow();
        }

        void loadTorchModel() {
            int i = segmentationStep();
            if (i < 0 || wb().pipeline().at(i).kind != "seg") {
                if (!findOperation("seg")) return;
                i = wb().pipeline().indexOf(wb().addStep("seg"));
            }
            const Step& s = wb().pipeline().at(i);
            std::string pathKey;
            for (const ParamSpec& spec : s.op().info().params)
                if (spec.type == ParamType::Path) {
                    pathKey = spec.key;
                    break;
                }
            const QString path = QFileDialog::getOpenFileName(self, QStringLiteral("Load Torch model"), lastDir,
                                                              QStringLiteral("TorchScript / ONNX (*.pt *.pth *.ts *.onnx);;All files (*)"));
            if (path.isEmpty() || pathKey.empty()) return;
            wb().setStepParam(i, pathKey, toStd(path));
            wb().select(i);
        }

        void mergeLabelsDialog() {
            bool ok = false;
            const QString text = QInputDialog::getText(self, QStringLiteral("Merge labels"),
                                                       QStringLiteral("Label ids to merge (comma-separated)"), QLineEdit::Normal,
                                                       QString::number(wb().viewState().selectedLabel), &ok);
            if (!ok) return;
            std::vector<std::uint32_t> ids;
            for (const QString& part : text.split(QLatin1Char(','), Qt::SkipEmptyParts)) {
                bool num = false;
                const uint v = part.trimmed().toUInt(&num);
                if (num && v > 0) ids.push_back(v);
            }
            if (ids.size() >= 2) wb().mergeLabels(ids);
            else wb().logLine("Merge labels: give at least two label ids.");
        }

        void selectFlagged(bool forward) {
            const std::uint32_t id = wb().nextFlaggedLabel(forward);
            if (!id) {
                wb().logLine("No flagged labels.");
                return;
            }
            ViewState s = wb().viewState();
            s.selectedLabel = id;
            s.labels = true;
            wb().setViewState(s);
        }

        void toggleHelp() {
            if (help->isVisible()) help->hide();
            else showHelpForSelected();
        }

        void showHelpForSelected() {
            const int i = wb().selectedIndex();
            if (i >= 0 && i < wb().pipeline().size()) help->showKind(wb().pipeline().at(i).kind);
            help->show();
            help->raise();
        }

        void about() {
            QMessageBox::about(self, QStringLiteral("About SIRIUS"),
                               QStringLiteral("<b>SIRIUS</b> — Structured Illumination Reconstruction and Image Utility Suite<br>"
                                              "Microscopy processing workbench.<br><br>"
                                              "CPU, CUDA and HPC backends · TIFF, OME-TIFF%1 · Qt %2<br>"
                                              "<a href=\"https://github.com/abcucberkeley/sirius\">github.com/abcucberkeley/sirius</a>")
                                   .arg(zarrSupported() ? QStringLiteral(", zarr / N5 (TensorStore)") : QString(), QLatin1String(qVersion())));
        }
    };

    MainWindow::MainWindow(WorkbenchBridge& bridge, QWidget* parent)
        : QMainWindow(parent), impl_(std::make_unique<Impl>(this, bridge)) {
        setObjectName(QStringLiteral("MainWindow"));
        setAcceptDrops(true);   // datasets, folders, pipelines and plugins open by being dropped
        resize(1600, 960);
        setDockOptions(QMainWindow::AnimatedDocks | QMainWindow::AllowNestedDocks | QMainWindow::AllowTabbedDocks);
        setDockNestingEnabled(true);
        Impl& d = *impl_;

        d.viewer = new ViewerWidget(bridge, this);
        setCentralWidget(d.viewer);

        d.ops = new OpsPanel(bridge, this);
        d.params = new ParamsPanel(bridge, this);
        d.diagnostics = new DiagnosticsPanel(bridge, this);
        d.assistant = new AssistantPanel(bridge, this);
        d.log = new LogPanel(bridge, this);
        d.help = new HelpWindow(bridge, this);
        d.help->hide();

        d.opsDock = makeDock(QStringLiteral("Operations"), d.ops, this);
        d.paramsDock = makeDock(QStringLiteral("Parameters"), d.params, this);
        d.diagDock = makeDock(QStringLiteral("Diagnostics"), d.diagnostics, this);
        d.assistantDock = makeDock(QStringLiteral("Assistant"), d.assistant, this);
        d.logDock = makeDock(QStringLiteral("Log"), d.log, this);
        d.diagnostics->setDock(d.diagDock);
        addDockWidget(Qt::LeftDockWidgetArea, d.opsDock);
        addDockWidget(Qt::RightDockWidgetArea, d.paramsDock);
        addDockWidget(Qt::RightDockWidgetArea, d.assistantDock);
        splitDockWidget(d.paramsDock, d.assistantDock, Qt::Horizontal);
        addDockWidget(Qt::BottomDockWidgetArea, d.diagDock);
        // The log shares the bottom area with the diagnostics as a tab: same
        // place, one click away, and it keeps the viewer its full height.
        addDockWidget(Qt::BottomDockWidgetArea, d.logDock);
        tabifyDockWidget(d.diagDock, d.logDock);
        d.diagDock->raise();
        d.assistantDock->hide();
        // Sizing a hidden dock here makes the first layout pass reserve its
        // width and grow the window past 1600 px; the assistant dock is
        // sized when it is first shown instead (toggleAssistant).
        resizeDocks({d.opsDock, d.paramsDock}, {theme::kOpsDockW, theme::kParamsDockW}, Qt::Horizontal);
        resizeDocks({d.diagDock}, {theme::kDiagnosticsH}, Qt::Vertical);
        setCorner(Qt::BottomLeftCorner, Qt::LeftDockWidgetArea);
        setCorner(Qt::BottomRightCorner, Qt::RightDockWidgetArea);

        d.buildMenus();
        d.buildStatusBar();
        d.defaultState = saveState();

        // panel <-> window wiring
        connect(d.ops, &OpsPanel::exportRequested, this, [this] { impl_->exportResultDialog(); });
        connect(d.ops, &OpsPanel::managePluginsRequested, this, [this] { impl_->pluginManager(); });
        connect(d.params, &ParamsPanel::helpRequested, this, [this](bool open) {
            if (open) impl_->showHelpForSelected();
            else impl_->help->hide();
        });
        connect(d.help, &HelpWindow::visibilityChanged, this, [this](bool visible) {
            impl_->params->setHelpOpen(visible);
            impl_->helpPage->setChecked(visible);
        });
        connect(d.assistant, &AssistantPanel::closeRequested, this, [this] { impl_->assistantDock->hide(); });
        connect(d.ops, &OpsPanel::removeStepRequested, this, [this](int i) { impl_->removeStepAt(i); });
        connect(d.params, &ParamsPanel::removeStepRequested, this, [this](int i) { impl_->removeStepAt(i); });
        connect(d.params, &ParamsPanel::runStepRequested, this, [this] { impl_->runSelectedStep(); });
        // ⛶: the diagnostics take the viewer's room until they are restored.
        connect(d.diagnostics, &DiagnosticsPanel::maximizedChanged, this, [this](bool on) {
            Impl& d = *impl_;
            if (d.maximiseDiag) d.maximiseDiag->setChecked(on);
            if (on) {
                d.diagRestoreHeight = std::max(d.diagDock->height(), theme::kDiagnosticsH);
                d.diagDock->show();
                d.diagDock->raise();
                if (QWidget* c = centralWidget()) c->hide();
                resizeDocks({d.diagDock}, {std::max(height() - theme::kTitleBarH - theme::kStatusBarH, 200)}, Qt::Vertical);
            } else {
                if (QWidget* c = centralWidget()) c->show();
                resizeDocks({d.diagDock}, {d.diagRestoreHeight}, Qt::Vertical);
            }
        });
        connect(d.viewer, &ViewerWidget::cursorChanged, this, [this](const QString& t) { impl_->statusCursor->setText(t); });
        connect(d.viewer, &ViewerWidget::zoomChanged, this, [this](const QString& t) {
            impl_->statusZoom->setText(QStringLiteral("zoom %1").arg(t));
        });
        connect(&bridge, &WorkbenchBridge::selectionChanged, this, [this] {
            impl_->refreshActions();
            if (impl_->help->isVisible()) impl_->showHelpForSelected();
        });
        auto refreshAll = [this] {
            impl_->refreshTitle();
            impl_->refreshStatus();
            impl_->refreshActions();
        };
        connect(&bridge, &WorkbenchBridge::datasetChanged, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::pipelineChanged, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::stepChanged, this, [this](int) { impl_->refreshStatus(); impl_->refreshActions(); });
        connect(&bridge, &WorkbenchBridge::viewStateChanged, this, [this] { impl_->refreshActions(); });
        connect(&bridge, &WorkbenchBridge::outputsChanged, this, [this] { impl_->refreshStatus(); });
        connect(&bridge, &WorkbenchBridge::historyChanged, this, [this] { impl_->refreshActions(); });
        connect(&bridge, &WorkbenchBridge::backendChanged, this, [this] {
            impl_->refreshActions();
            impl_->refreshTitle();
        });
        connect(&bridge, &WorkbenchBridge::runStarted, this, [this] {
            impl_->progressClock.start();
            impl_->progressMessage.clear();
            impl_->showProgress(0.0, QString());
            impl_->refreshActions();
        });
        connect(&bridge, &WorkbenchBridge::runProgress, this, [this](double f, int step, const QString& msg) {
            QString message = msg;
            if (step >= 0 && step < impl_->wb().pipeline().size()) {
                const QString name = fromStd(impl_->wb().pipeline().at(step).name);
                message = message.isEmpty() ? name : name + QStringLiteral(" · ") + message;
            }
            impl_->showProgress(f, message);
        });
        connect(&bridge, &WorkbenchBridge::runFinished, this, [this](bool ok, const QString& error) {
            impl_->statusProgress->hide();
            refreshAllLater();
            // A cancelled run arrives with an empty error (workbench_bridge),
            // so there is no message text to recognise here.
            if (!ok && !error.isEmpty()) QMessageBox::warning(this, QStringLiteral("Run failed"), error);
        });
        connect(&bridge, &WorkbenchBridge::taskStarted, this, [this](const QString& name) {
            impl_->progressClock.start();
            impl_->progressMessage = name;
            impl_->showProgress(0.0, QString());
            impl_->refreshActions();
        });
        connect(&bridge, &WorkbenchBridge::taskProgress, this, [this](double f, const QString& msg) {
            impl_->showProgress(f, msg);
        });
        connect(&bridge, &WorkbenchBridge::taskFinished, this, [this](bool ok, const QString& error) {
            impl_->statusProgress->hide();
            impl_->refreshActions();
            if (!ok && !error.isEmpty()) QMessageBox::warning(this, impl_->bridge.taskLabel(), error);
        });
        connect(&bridge, &WorkbenchBridge::logged, this, [this](const QString& line) { impl_->showLogLine(line, 4000); });
        // Both edges of the run: the workbench refuses edits between them, and
        // every menu item and panel control has to follow.
        connect(&bridge, &WorkbenchBridge::runStateChanged, this, [this] { impl_->refreshActions(); });

        // The dock arrangement first, the window size second: restoreState()
        // hands the dock areas their saved extents and the layout then fits
        // the window around them, so running it after restoreGeometry() can
        // leave the window shorter than the size that was just restored.
        QSettings settings;
        if (settings.contains(QStringLiteral("window/state"))) restoreState(settings.value(QStringLiteral("window/state")).toByteArray());
        if (settings.contains(QStringLiteral("window/geometry")))
            restoreGeometry(settings.value(QStringLiteral("window/geometry")).toByteArray());
        refreshAll();
    }

    MainWindow::~MainWindow() = default;

    void MainWindow::refreshAllLater() {
        impl_->refreshTitle();
        impl_->refreshStatus();
        impl_->refreshActions();
    }

    // A plain folder of files (no manifest, not a zarr / N5 store) goes through
    // the pattern dialog that builds the manifest; everything else opens directly.
    void MainWindow::openDatasetPath(const QString& path) {
        const QFileInfo info(path);
        if (info.isDir() && !isFolderDataset(toStd(path))) {
            const QDir dir(path);
            bool store = false;
            for (const char* marker : {".zarray", ".zgroup", "zarr.json", "attributes.json"})
                if (dir.exists(QString::fromLatin1(marker))) store = true;
            if (!store) {
                impl_->lastDir = path;
                // window-modal but not blocking: this also runs before the
                // event loop when the folder comes from the command line
                auto* dialog = new FolderDatasetDialog(impl_->bridge, path, this);
                dialog->setAttribute(Qt::WA_DeleteOnClose);
                dialog->open();
                return;
            }
        }
        impl_->openWith(path, OpenOptions{});
    }

    ViewerWidget& MainWindow::viewer() { return *impl_->viewer; }

    void MainWindow::openPipelinePath(const QString& path) {
        try {
            impl_->wb().loadPipeline(toStd(path));
            impl_->savedHistory = impl_->wb().history().size();   // a loaded pipeline is a saved one
            impl_->lastDir = QFileInfo(path).absolutePath();
        } catch (const std::exception& e) {
            impl_->wb().logLine(std::string("Load pipeline failed: ") + e.what());
            QMessageBox::warning(this, QStringLiteral("Load pipeline"), QString::fromUtf8(e.what()));
        }
        refreshAllLater();
    }

    void MainWindow::runAll() { impl_->bridge.startRun(-1); }

    void MainWindow::setUnattended(bool on) { impl_->unattended = on; }

    void MainWindow::askAssistant(const QString& text) {
        impl_->assistantDock->show();
        impl_->assistant->ask(text);
    }

    bool MainWindow::eventFilter(QObject* watched, QEvent* event) {
        if (watched == impl_->statusLog && event->type() == QEvent::MouseButtonRelease) {
            impl_->showLog();
            return true;
        }
        return QMainWindow::eventFilter(watched, event);
    }

    namespace {
        // What a dropped path is, by extension and by what is inside a folder.
        enum class DropKind { None,
                              Dataset,
                              Folder,
                              Pipeline,
                              Plugin };

        DropKind kindOfDrop(const QString& path) {
            const QFileInfo info(path);
            if (info.isDir()) return DropKind::Folder;
            if (!info.isFile()) return DropKind::None;
            const QString name = info.fileName().toLower();
            if (name.endsWith(QStringLiteral(".sirius.toml"))) return DropKind::Pipeline;
            if (name.endsWith(QStringLiteral(".py"))) return DropKind::Plugin;
            static const char* datasets[] = {".tif", ".tiff", ".ome.tif", ".ome.tiff", ".zarr", ".n5", ".sir5"};
            for (const char* ext : datasets)
                if (name.endsWith(QString::fromLatin1(ext))) return DropKind::Dataset;
            return DropKind::None;
        }

        // The paths of a drop we know what to do with, in the order they came.
        QStringList droppablePaths(const QMimeData* mime) {
            QStringList out;
            if (!mime || !mime->hasUrls()) return out;
            for (const QUrl& url : mime->urls()) {
                if (!url.isLocalFile()) continue;
                const QString path = url.toLocalFile();
                if (kindOfDrop(path) != DropKind::None) out << path;
            }
            return out;
        }
    } // namespace

    // A drop is an edit: the workbench refuses every one of them while a run
    // holds the pipeline (Workbench::canEdit), so the cursor has to say so
    // rather than take the drop and answer it with an error box.
    bool MainWindow::canAcceptDrop(const QMimeData* mime) const {
        return impl_->wb().canEdit() && !droppablePaths(mime).isEmpty();
    }

    void MainWindow::dragEnterEvent(QDragEnterEvent* event) {
        if (!canAcceptDrop(event->mimeData())) {
            event->ignore();
            return;
        }
        event->setDropAction(Qt::LinkAction);   // "open this", not "move it here"
        event->accept();
    }

    void MainWindow::dragMoveEvent(QDragMoveEvent* event) {
        if (!canAcceptDrop(event->mimeData())) {
            event->ignore();
            return;
        }
        event->setDropAction(Qt::LinkAction);
        event->accept();
    }

    void MainWindow::dropPaths(const QStringList& paths) {
        QStringList absolute;
        for (const QString& path : paths)
            if (!path.isEmpty()) absolute << QFileInfo(path).absoluteFilePath();
        openDroppedPaths(absolute);
    }

    void MainWindow::dropEvent(QDropEvent* event) {
        const QStringList paths = droppablePaths(event->mimeData());
        if (paths.isEmpty()) return;
        event->acceptProposedAction();
        // Opening reads the file and can raise a dialog (the folder pattern
        // dialog, an error box). Both belong after the drop has been answered:
        // while this handler runs, the application the files were dragged from
        // is blocked waiting for it.
        QMetaObject::invokeMethod(
            this, [this, paths] { openDroppedPaths(paths); }, Qt::QueuedConnection);
    }

    void MainWindow::openDroppedPaths(const QStringList& paths) {
        if (paths.isEmpty()) return;
        if (!impl_->wb().canEdit()) {
            // logLine reaches the status bar's log line, so this is said as
            // well as shown by the cursor that refused the drop.
            impl_->wb().logLine("A run is in progress: cancel it (Esc) or wait before opening " + toStd(paths.first()) + ".");
            return;
        }
        // Several dataset files from one folder at once are what a folder
        // dataset is for, so offer that rather than opening one and dropping
        // the rest. Files from different folders have no such folder to open.
        const QString folder = QFileInfo(paths.first()).absolutePath();
        const bool manyDatasets = paths.size() > 1 && std::all_of(paths.begin(), paths.end(), [&folder](const QString& p) {
                                      return kindOfDrop(p) == DropKind::Dataset && QFileInfo(p).absolutePath() == folder;
                                  });
        if (manyDatasets) {
            impl_->wb().logLine("Dropped " + std::to_string(paths.size()) + " files: opening " + toStd(folder) +
                                " as a folder dataset.");
            openDatasetPath(folder);
            return;
        }
        for (const QString& path : paths) {
            switch (kindOfDrop(path)) {
                case DropKind::Pipeline:
                    openPipelinePath(path);
                    break;
                case DropKind::Plugin:
                    impl_->pluginManager(path);
                    break;
                case DropKind::Folder:
                case DropKind::Dataset:
                    openDatasetPath(path);
                    break;
                case DropKind::None:
                    // only reachable through --drop: a real drop is filtered
                    // by droppablePaths before it gets here
                    impl_->wb().logLine("Nothing to open in " + toStd(path) +
                                        ": expected a TIFF / zarr / N5, a folder, a .sirius.toml or a .py operation.");
                    break;
            }
            // one dataset at a time: a second would replace the first
            if (kindOfDrop(path) == DropKind::Dataset || kindOfDrop(path) == DropKind::Folder) break;
        }
    }

    void MainWindow::closeEvent(QCloseEvent* event) {
        // Edits since the last save go with the window: say so first (not
        // when nobody is at the keyboard; see setUnattended).
        if (!impl_->unattended && impl_->unsavedWork()) {
            const auto answer = QMessageBox::question(
                this, QStringLiteral("Quit"),
                QStringLiteral("The pipeline has unsaved changes, and label edits are only kept by an export. Quit anyway?"),
                QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
            if (answer != QMessageBox::Yes) {
                event->ignore();
                return;
            }
        }
        if (impl_->bridge.running()) {
            // Unattended (--screenshot, --quit-after, scripting): there is
            // nobody to answer, and a modal question here would hold the
            // application open for ever.
            if (impl_->unattended) {
                impl_->bridge.cancelRun();
            } else {
                const auto answer = QMessageBox::question(this, QStringLiteral("Quit"),
                                                          QStringLiteral("A run is in progress. Cancel it and quit?"));
                if (answer != QMessageBox::Yes) {
                    event->ignore();
                    return;
                }
                impl_->bridge.cancelRun();
            }
        }
        // Not while the diagnostics cover the viewer: that would save a
        // hidden central widget as the layout to open with.
        impl_->diagnostics->setMaximized(false);
        QSettings settings;
        // A size the screen imposed is not a size the user chose. On a
        // display smaller than the window wants (a laptop, or the small
        // virtual screen of a headless run) Qt clamps the window to the
        // available area, and saving that is how a window comes back shorter
        // than it was left on the big monitor. The arrangement of the docks
        // is worth keeping either way.
        const QRect available = screen() ? screen()->availableGeometry() : QRect();
        const QRect frame = frameGeometry();
        const bool deliberate = isMaximized() || isFullScreen();   // a real choice, and Qt records it as one
        const bool squeezed = !deliberate && !available.isEmpty() &&
                              (frame.height() >= available.height() - 2 || frame.width() >= available.width() - 2);
        if (!squeezed) settings.setValue(QStringLiteral("window/geometry"), saveGeometry());
        settings.setValue(QStringLiteral("window/state"), saveState());
        impl_->help->hide();
        QMainWindow::closeEvent(event);
    }

} // namespace sirius::app
