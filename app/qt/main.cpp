// sirius-app: the SIRIUS microscopy workbench (docs/design/README.md).
//
//   sirius-app [--dataset stack.tif] [--pipeline steps.sirius.toml] [--run] [files...]
//
// Everything can also be opened from the File menu; --run runs every
// enabled step as soon as the window is up. Files named without an option
// open as though dropped on the window, which is what a file manager's
// "Open with" passes (app/linux/sirius-app.desktop).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <functional>
#include <optional>

#include <QApplication>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QAction>
#include <QDir>
#include <QEventLoop>
#include <QDialog>
#include <QFileInfo>
#include <QIcon>
#include <QMenu>
#include <QDockWidget>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QKeySequence>
#include <QSettings>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QTimer>

#include "core/app_paths.hpp"
#include "core/help_pages.hpp"
#include "core/operation.hpp"
#include "core/ops/builtin.hpp"
#include "core/tool_api.hpp"
#include "core/workbench.hpp"
#include "qt/dialogs/preferences_dialog.hpp"
#include "qt/main_window.hpp"
#include "qt/viewer/viewer_widget.hpp"
#include "qt/worker_launcher.hpp"
#include "qt/qt_strings.hpp"
#include "qt/secret_store.hpp"
#include "qt/theme.hpp"
#include "qt/workbench_bridge.hpp"

#include <sirius/device.hpp>

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QCoreApplication::setApplicationName(QStringLiteral("sirius-app"));
    QCoreApplication::setOrganizationName(QStringLiteral("sirius"));
    QCoreApplication::setApplicationVersion(QStringLiteral("0.2"));
    // The window's app id on Wayland and its WM_CLASS on X11: how a desktop
    // matches the window to sirius-app.desktop for its icon and name.
    QGuiApplication::setDesktopFileName(QStringLiteral("sirius-app"));
    {
        // PNGs rather than the SVG: an SVG icon needs Qt's SVG plugin, which
        // a deployment may not carry (app/qt/resources/icons/app/README.md)
        QIcon icon;
        for (int px : {16, 24, 32, 48, 64, 128, 256})
            icon.addFile(QStringLiteral(":/icons/app/sirius-app-%1.png").arg(px), QSize(px, px));
        QApplication::setWindowIcon(icon);
    }
    // the help pages, the worker and the plugins are found relative to it
    sirius::app::setApplicationDirectory(sirius::app::toStd(QCoreApplication::applicationDirPath()));
    sirius::app::theme::applyTheme(app);

    QCommandLineParser parser;
    parser.setApplicationDescription(QStringLiteral("SIRIUS microscopy processing workbench"));
    parser.addHelpOption();
    parser.addVersionOption();
    const QCommandLineOption datasetOpt(QStringLiteral("dataset"), QStringLiteral("Dataset to open (TIFF / OME-TIFF / zarr)"),
                                        QStringLiteral("path"));
    const QCommandLineOption pipelineOpt(QStringLiteral("pipeline"), QStringLiteral("Pipeline file (.sirius.toml)"),
                                         QStringLiteral("path"));
    const QCommandLineOption runOpt(QStringLiteral("run"), QStringLiteral("Run every enabled step once the window is up"));
    // Developer aids: grab the window to a PNG (after the run when --run is
    // given) and quit; or just quit after a delay, for headless smoke tests.
    const QCommandLineOption screenshotOpt(QStringLiteral("screenshot"),
                                           QStringLiteral("Save a screenshot of the window to <path> and quit"),
                                           QStringLiteral("path"));
    const QCommandLineOption quitAfterOpt(QStringLiteral("quit-after"), QStringLiteral("Quit after <ms> milliseconds"),
                                          QStringLiteral("ms"));
    // Scripting for smoke tests: tool calls through the assistant's typed API
    // ({"name": "set_view", "args": {"mode": "3d"}}) and menu actions by
    // their text ("Assistant"), applied once the run (if any) has finished.
    const QCommandLineOption toolOpt(QStringLiteral("tool"), QStringLiteral("Call a tool of the assistant API (JSON, repeatable)"),
                                     QStringLiteral("json"));
    const QCommandLineOption recordOpt(QStringLiteral("record"), QStringLiteral("Record this session to a JSON-lines file"),
                                       QStringLiteral("path"));
    const QCommandLineOption dropOpt(QStringLiteral("drop"), QStringLiteral("Act as though this path were dropped on the window (repeatable)"),
                                     QStringLiteral("path"));
    const QCommandLineOption strokeOpt(QStringLiteral("stroke"), QStringLiteral("Drag on the XY pane: x0,y0,x1,y1,moves in voxels (repeatable, after the tools)"),
                                       QStringLiteral("spec"));
    const QCommandLineOption wheelOpt(QStringLiteral("wheel"), QStringLiteral("Wheel on the XY pane (the step pane in Compare): x,y,steps in voxels (repeatable, before the strokes)"),
                                      QStringLiteral("spec"));
    const QCommandLineOption actionOpt(QStringLiteral("action"), QStringLiteral("Trigger a menu action by its text (repeatable)"),
                                       QStringLiteral("text"));
    const QCommandLineOption keyOpt(QStringLiteral("key"), QStringLiteral("Focus a widget by its accessible or object name and press a key: \"Z plane=Right\" (repeatable)"),
                                    QStringLiteral("name=key"));
    const QCommandLineOption askOpt(QStringLiteral("ask"), QStringLiteral("Send a message to the assistant"), QStringLiteral("text"));
    // A settings file of this run's own. Without it every headless run reads
    // and writes the settings of whoever is logged in: their dock widths,
    // backend, cache policy and assistant settings. That makes a screenshot
    // say as much about the developer's window as about the code, and a
    // scripted run saves its own layout back over theirs.
    const QCommandLineOption settingsOpt(QStringLiteral("settings"),
                                         QStringLiteral("Keep settings in <dir> instead of the user's own; "
                                                        "'scratch' uses a new one, removed when the run ends"),
                                         QStringLiteral("dir"));
    const QCommandLineOption settleOpt(QStringLiteral("settle"), QStringLiteral("Milliseconds to wait before the screenshot (default 600)"),
                                       QStringLiteral("ms"));
    const QCommandLineOption listCudaOpt(QStringLiteral("list-cuda"),
                                         QStringLiteral("Print visible CUDA devices and exit"));
    parser.addOptions({datasetOpt, pipelineOpt, runOpt, screenshotOpt, quitAfterOpt, toolOpt, actionOpt, keyOpt, askOpt,
                       settleOpt, strokeOpt, wheelOpt, dropOpt, recordOpt, settingsOpt, listCudaOpt});
    parser.addPositionalArgument(QStringLiteral("files"), QStringLiteral("Datasets or pipeline files to open, as though dropped on the window"),
                                 QStringLiteral("[files...]"));
    parser.process(app);
    if (parser.isSet(listCudaOpt)) {
        const int n = sirius::cudaDeviceCount();
        std::printf("cuda devices: %d\n", n);
        for (int i = 0; i < n; ++i) {
            try {
                const auto p = sirius::deviceProperties(sirius::Device::cuda(i));
                std::printf("  %d: %s  %.1f GB  sm_%d%d\n", i, p.name.c_str(),
                            static_cast<double>(p.totalMemoryBytes) / (1024.0 * 1024.0 * 1024.0), p.computeMajor,
                            p.computeMinor);
            } catch (const std::exception& e) {
                std::printf("  %d: (%s)\n", i, e.what());
            }
        }
        return n > 0 ? 0 : 1;
    }
    const QStringList files = parser.positionalArguments();
    const bool filesHavePipeline = std::any_of(files.begin(), files.end(), [](const QString& f) { return f.endsWith(QStringLiteral(".toml"), Qt::CaseInsensitive); });

    // Before anything reads a setting: PreferencesDialog::applyStored below and
    // the window's saved layout both use a default-constructed QSettings, so
    // pointing the default format and path at a directory of our own moves the
    // settings, and the secret store is pointed at the same directory (on
    // Windows its secrets are in QSettings already; elsewhere they are a file
    // that would otherwise stay in the user's ~/.sirius).
    // Declared before everything that saves settings on the way out (the
    // window's layout), so a scratch directory goes last, when main returns.
    std::optional<QTemporaryDir> scratchSettings;
    if (parser.isSet(settingsOpt)) {
        QString dir = parser.value(settingsOpt);
        if (dir == QLatin1String("scratch")) {
            scratchSettings.emplace(QDir(QStandardPaths::writableLocation(QStandardPaths::TempLocation))
                                        .filePath(QStringLiteral("sirius-settings-XXXXXX")));
            if (!scratchSettings->isValid()) {
                qCritical("cannot create a scratch settings directory: %s", qPrintable(scratchSettings->errorString()));
                return 2;
            }
            dir = scratchSettings->path();
        } else if (!QDir().mkpath(dir)) {
            qCritical("cannot create the settings directory %s", qPrintable(dir));
            return 2;
        }
        QSettings::setDefaultFormat(QSettings::IniFormat);
        QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, dir);
        sirius::app::secrets::setStoreDirectory(dir);
        qInfo("settings: %s", qPrintable(QSettings().fileName()));
    }

    sirius::app::registerBuiltinOperations();

    // Per-process scratch for the disk cache and worker files.
    const QString tmp = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
    const QString scratch = QDir(tmp).filePath(QStringLiteral("sirius-%1").arg(QCoreApplication::applicationPid()));
    QDir().mkpath(scratch);

    sirius::app::Workbench workbench(std::filesystem::path(sirius::app::toStd(scratch)));
    sirius::app::PreferencesDialog::applyStored(workbench);
    // Steps that need Python (Torch models) get a worker spawned on demand.
    // Declared before the bridge: the bridge's destructor joins the run
    // thread, which may be inside the launcher starting that worker.
    sirius::app::WorkerLauncher launcher;
    sirius::app::WorkbenchBridge bridge(workbench);
    QObject::connect(&launcher, &sirius::app::WorkerLauncher::logged, &bridge,
                     [&workbench](const QString& line) { workbench.logLine("worker: " + sirius::app::toStd(line)); });
    workbench.setLocalWorkerLauncher([&launcher] { return launcher.connect(); });
    auto syncWorkerDevice = [&] {
        if (workbench.backend() != sirius::app::Backend::Cuda) launcher.setDevice(QStringLiteral("cpu"));
        else if (workbench.cudaDevice() < 0) launcher.setDevice(QStringLiteral("cuda"));
        else launcher.setDevice(QStringLiteral("cuda:%1").arg(workbench.cudaDevice()));
    };
    syncWorkerDevice();
    QObject::connect(&bridge, &sirius::app::WorkbenchBridge::backendChanged, &bridge, syncWorkerDevice);
    // a gated model's token goes with the request that downloads it
    workbench.setHubTokenProvider([] { return sirius::app::toStd(sirius::app::secrets::read(QStringLiteral("hub/token")).trimmed()); });
    sirius::app::MainWindow window(bridge);
    // User operations come from the Python worker. A pipeline given on the
    // command line may use them, so load them first in that case; otherwise
    // after the window is up so start-up stays quick.
    if (parser.isSet(pipelineOpt) || filesHavePipeline) workbench.loadPlugins(false);
    else QTimer::singleShot(400, &window, [&workbench] { workbench.loadPlugins(false); });
    if (parser.isSet(pipelineOpt)) window.openPipelinePath(parser.value(pipelineOpt));
    // recording starts before anything scripted happens, so the run is in it
    if (parser.isSet(recordOpt)) {
        try {
            workbench.startRecording(sirius::app::toStd(parser.value(recordOpt)));
        } catch (const std::exception& e) {
            qWarning("--record: %s", e.what());
        }
    }
    window.show();
    // Opened from inside the event loop so that a dialog it raises (an error
    // box, the folder pattern dialog) does not block the scripting timers.
    if (parser.isSet(datasetOpt)) {
        const QString dataset = parser.value(datasetOpt);
        QTimer::singleShot(0, &window, [&window, dataset] { window.openDatasetPath(dataset); });
    }
    if (!files.isEmpty()) QTimer::singleShot(0, &window, [&window, files] { window.dropPaths(files); });
    const QStringList toolCalls = parser.values(toolOpt);
    const QStringList actions = parser.values(actionOpt);
    sirius::app::ToolApi tools(workbench);
    // get_help answers from the pages the application reads, as it does for the assistant
    tools.setHelpHook([](const std::string& kind) { return sirius::app::loadHelpPage(kind).markdown; });
    // Scripted runs block on the worker thread the way the assistant does.
    tools.setRunHook([&bridge](int target) {
        QEventLoop loop;
        bool ok = false;
        QString error;
        QObject::connect(&bridge, &sirius::app::WorkbenchBridge::runFinished, &loop, [&](bool good, const QString& err) {
            ok = good;
            error = err;
            loop.quit();
        });
        if (!bridge.startRun(target)) return nlohmann::json{{"ok", false}, {"error", "the run could not start (see the log)"}};
        loop.exec();
        return nlohmann::json{{"ok", ok}, {"error", sirius::app::toStd(error)}};
    });
    // Scripted steps run in the order they were written on the command line.
    // Grouping them by kind -- every tool, then every action -- meant a
    // get_state after an --action reported the state before it, which is not
    // what anyone writing the line intends and made the widgets hard to test.
    // QCommandLineParser does not keep the order, so it is read back off argv.
    auto letTheWindowCatchUp = [&bridge] {
        // let the window react (repaint, refresh) between steps, as it would
        // between a user's actions
        QCoreApplication::processEvents(QEventLoop::AllEvents, 200);
        QCoreApplication::sendPostedEvents();
        QCoreApplication::processEvents(QEventLoop::AllEvents, 200);
        // A dataset opens on a worker thread and is installed when that task
        // ends: the next scripted step waits for it, as a user would for the
        // progress bar, rather than find no dataset yet.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(600);
        while (bridge.taskRunning() && std::chrono::steady_clock::now() < deadline)
            QCoreApplication::processEvents(QEventLoop::AllEvents | QEventLoop::WaitForMoreEvents, 50);
        QCoreApplication::processEvents(QEventLoop::AllEvents, 50);
    };
    auto runTool = [&](const QString& call) {
        const QJsonObject j = QJsonDocument::fromJson(call.toUtf8()).object();
        const nlohmann::json args = nlohmann::json::parse(QJsonDocument(j.value(QStringLiteral("args")).toObject()).toJson().constData());
        const nlohmann::json r = tools.call(sirius::app::toStd(j.value(QStringLiteral("name")).toString()), args);
        workbench.logLine("tool " + sirius::app::toStd(j.value(QStringLiteral("name")).toString()) + " → " + r.dump().substr(0, 200));
        qInfo("tool %s -> %s", qPrintable(j.value(QStringLiteral("name")).toString()), r.dump(2).c_str());
        letTheWindowCatchUp();
    };
    auto runAction = [&](const QString& text) {
        for (QAction* a : window.findChildren<QAction*>())
            if (a->text().remove(QLatin1Char('&')) == text || a->text().remove(QLatin1Char('&')).startsWith(text + QChar(0x2026))) {
                a->trigger();
                letTheWindowCatchUp();
                return;
            }
        workbench.logLine("no action named " + sirius::app::toStd(text));
        qWarning("no action named %s", qPrintable(text));
    };
    // A key press as a user makes it: to the focused widget, through the
    // application -- so a shortcut the key matches is asked about first --
    // with the dock the widget sits in raised so that it can take the focus.
    auto pressKey = [&](const QString& spec) {
        const int eq = spec.indexOf(QLatin1Char('='));
        const QString target = spec.left(eq);
        const QKeySequence key(eq < 0 ? QString() : spec.mid(eq + 1));
        QWidget* widget = nullptr;
        for (QWidget* top : QApplication::topLevelWidgets()) {
            for (QWidget* w : top->findChildren<QWidget*>())
                if (!widget && (w->accessibleName() == target || w->objectName() == target)) widget = w;
        }
        if (!widget || key.isEmpty()) {
            qWarning("--key %s: no widget of that name, or no key", qPrintable(spec));
            return;
        }
        for (QWidget* w = widget; w; w = w->parentWidget())
            if (auto* dock = qobject_cast<QDockWidget*>(w)) {
                dock->show();
                dock->raise();
            }
        widget->window()->activateWindow();
        letTheWindowCatchUp();
        widget->setFocus(Qt::OtherFocusReason);
        letTheWindowCatchUp();
        const QKeyCombination combo = key[0];
        const Qt::KeyboardModifiers mods = combo.keyboardModifiers();
        QString text;
        if (combo.key() >= Qt::Key_Space && combo.key() <= Qt::Key_AsciiTilde && !(mods & (Qt::ControlModifier | Qt::AltModifier | Qt::MetaModifier))) {
            text = QChar(static_cast<char16_t>(combo.key()));
            if (!(mods & Qt::ShiftModifier)) text = text.toLower();
        }
        QWidget* receiver = QApplication::focusWidget() ? QApplication::focusWidget() : widget;
        qInfo("key %s to %s: focus %s", qPrintable(key.toString()), qPrintable(target), receiver == widget ? "yes" : "no");
        QKeyEvent press(QEvent::KeyPress, combo.key(), mods, text);
        QCoreApplication::sendEvent(receiver, &press);
        QKeyEvent release(QEvent::KeyRelease, combo.key(), mods, text);
        QCoreApplication::sendEvent(receiver, &release);
        letTheWindowCatchUp();
    };
    auto script = [&] {
        letTheWindowCatchUp();   // a --dataset still loading
        const QStringList argv = QCoreApplication::arguments();
        for (int i = 1; i < argv.size(); ++i) {
            QString name = argv[i];
            if (!name.startsWith(QLatin1String("--"))) continue;
            name.remove(0, 2);
            QString value;
            const int eq = name.indexOf(QLatin1Char('='));
            if (eq >= 0) {
                value = name.mid(eq + 1);
                name.truncate(eq);
            } else if (i + 1 < argv.size()) {
                value = argv[i + 1];
            }
            if (name == QLatin1String("tool")) runTool(value);
            else if (name == QLatin1String("action")) runAction(value);
            else if (name == QLatin1String("key")) pressKey(value);
            else if (name == QLatin1String("drop")) {
                window.dropPaths({value});
                letTheWindowCatchUp();
            } else if (name == QLatin1String("wheel")) {
                const QStringList v = value.split(QLatin1Char(','));
                if (v.size() == 3) window.viewer().syntheticWheel(QPointF(v[0].toDouble(), v[1].toDouble()), v[2].toDouble());
                letTheWindowCatchUp();
            } else if (name == QLatin1String("stroke")) {
                const QStringList v = value.split(QLatin1Char(','));
                if (v.size() == 5)
                    window.viewer().syntheticStroke(QPointF(v[0].toDouble(), v[1].toDouble()), QPointF(v[2].toDouble(), v[3].toDouble()),
                                                    v[4].toInt());
                letTheWindowCatchUp();
            } else if (name == QLatin1String("ask")) {
                window.askAssistant(value);
            }
        }
    };
    const bool scripted = !toolCalls.isEmpty() || !actions.isEmpty() || parser.isSet(keyOpt) || parser.isSet(askOpt) || parser.isSet(strokeOpt) || parser.isSet(wheelOpt) || parser.isSet(dropOpt);
    const bool headless = scripted || parser.isSet(screenshotOpt);
    // An interactive --run just starts; a headless one (below) also decides
    // the exit code and when the window is grabbed.
    if (parser.isSet(runOpt) && !headless) QTimer::singleShot(0, &window, &sirius::app::MainWindow::runAll);
    // Nobody is at the keyboard in any of these modes, so the window must not
    // ask whether to cancel a running job on the way out (see
    // MainWindow::setUnattended).
    if (headless || parser.isSet(quitAfterOpt)) window.setUnattended(true);
    const int settle = parser.isSet(settleOpt) ? parser.value(settleOpt).toInt() : 600;
    // What the process exits with: 1 when a headless run failed (or was
    // still going when the deadline struck), 2 when it could not start.
    int exitCode = 0;
    // Declared here, not in the block below: the timers armed there fire
    // inside exec(), long after the block's own locals are gone.
    std::function<void()> grabWhenIdle;
    bool finished = false;
    if (headless) {
        const QString path = parser.value(screenshotOpt);
        // A scripted run (--tool run) blocks in a nested event loop in which
        // the grab timer still fires; the grab waits for it up to this long
        // so the picture shows the result rather than the middle of the run.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(600);
        auto grab = [&window, &app, &bridge, &exitCode, path] {
                // size report: which widget dictates the window's minimum
            QString report = QStringLiteral("window %1x%2 min %3x%4").arg(window.width()).arg(window.height()).arg(window.minimumSizeHint().width()).arg(window.minimumSizeHint().height());
            if (QWidget* c = window.centralWidget())
                report += QStringLiteral(" central-min %1x%2").arg(c->minimumSizeHint().width()).arg(c->minimumSizeHint().height());
            for (QDockWidget* d : window.findChildren<QDockWidget*>())
                report += QStringLiteral(" %1-min %2x%3").arg(d->objectName()).arg(d->widget() ? d->widget()->minimumSizeHint().width() : -1).arg(d->widget() ? d->widget()->minimumSizeHint().height() : -1);
            qInfo("%s", qPrintable(report));
            window.grab().save(path);
                // a dialog opened by --action is grabbed beside the window
            if (QWidget* modal = QApplication::activeModalWidget()) {
                QFileInfo fi(path);
                modal->grab().save(fi.path() + QLatin1Char('/') + fi.completeBaseName() + QStringLiteral("-dialog.") + fi.suffix());
            }
                // tool windows and non-modal dialogs (the plugin manager) beside it too.
                // A dialog under a modal question is "-dialog-2": it used to be saved
                // as "-dialog" as well, over the picture of the question itself.
            int dialogs = QApplication::activeModalWidget() ? 1 : 0;
            for (QWidget* top : QApplication::topLevelWidgets())
                if (top != &window && top->isVisible() && top->isWindow() && !qobject_cast<QMenu*>(top) &&
                    top != QApplication::activeModalWidget() && (top->windowType() == Qt::Tool || qobject_cast<QDialog*>(top))) {
                    QFileInfo fi(path);
                    QString tag = QStringLiteral("-tool.");
                    if (top->windowType() != Qt::Tool)
                        tag = ++dialogs == 1 ? QStringLiteral("-dialog.") : QStringLiteral("-dialog-%1.").arg(dialogs);
                    top->grab().save(fi.path() + QLatin1Char('/') + fi.completeBaseName() + tag + fi.suffix());
                }
            while (QWidget* modal = QApplication::activeModalWidget()) modal->close();   // let exec() return
            if (bridge.running()) {   // a slow step must not hold the exit
                bridge.cancelRun();
                if (exitCode == 0) exitCode = 1;
            }
            app.quit();
        };
        grabWhenIdle = [&window, &bridge, &grabWhenIdle, grab, settle, deadline] {
            // a run, or a dataset still loading: the picture waits for either
            if ((bridge.running() || bridge.taskRunning()) && std::chrono::steady_clock::now() < deadline) {
                QTimer::singleShot(settle, &window, grabWhenIdle);
                return;
            }
            grab();
        };
        auto finish = [&window, &script, &finished, &grabWhenIdle, path, scripted, settle] {
            if (finished) return;   // a scripted run's own runFinished lands here too
            finished = true;
            // Arm the grab before scripting: a modal dialog opened by an
            // action runs its own event loop, in which the timer still fires.
            if (!path.isEmpty()) QTimer::singleShot(settle, &window, grabWhenIdle);
            if (scripted) script();
        };
        if (parser.isSet(runOpt)) {
            QObject::connect(&bridge, &sirius::app::WorkbenchBridge::runFinished, &window,
                             [&window, &exitCode, finish](bool ok, const QString&) {
                                 if (!ok && exitCode == 0) exitCode = 1;
                                 QTimer::singleShot(300, &window, finish);
                             });
            QTimer::singleShot(0, &window, [&window, &bridge, &exitCode, finish] {
                window.runAll();
                if (!bridge.running()) {   // could not start: no dataset, a validation error
                    exitCode = 2;
                    QTimer::singleShot(300, &window, finish);
                }
            });
            QTimer::singleShot(600000, &window, [&exitCode, finish] {   // never hang a headless run
                if (exitCode == 0) exitCode = 1;
                finish();
            });
        } else {
            QTimer::singleShot(1200, &window, finish);
        }
    }
    if (parser.isSet(quitAfterOpt)) QTimer::singleShot(parser.value(quitAfterOpt).toInt(), &app, &QApplication::quit);
    const int rc = QApplication::exec();
    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::path(sirius::app::toStd(scratch)), ec);
    return rc != 0 ? rc : exitCode;
}
