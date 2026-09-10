// sirius-app: the SIRIUS microscopy workbench (docs/design/README.md).
//
//   sirius-app [--dataset stack.tif] [--pipeline steps.sirius.toml] [--run]
//
// Everything can also be opened from the File menu; --run runs every
// enabled step as soon as the window is up.

#include <chrono>
#include <filesystem>
#include <functional>

#include <QApplication>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QAction>
#include <QDir>
#include <QEventLoop>
#include <QDialog>
#include <QFileInfo>
#include <QMenu>
#include <QDockWidget>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStandardPaths>
#include <QTimer>

#include "core/operation.hpp"
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

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QCoreApplication::setApplicationName(QStringLiteral("sirius-app"));
    QCoreApplication::setOrganizationName(QStringLiteral("sirius"));
    QCoreApplication::setApplicationVersion(QStringLiteral("0.2"));
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
    const QCommandLineOption wheelOpt(QStringLiteral("wheel"), QStringLiteral("Wheel on the XY pane: x,y,steps in voxels (repeatable, before the strokes)"),
                                      QStringLiteral("spec"));
    const QCommandLineOption actionOpt(QStringLiteral("action"), QStringLiteral("Trigger a menu action by its text (repeatable)"),
                                       QStringLiteral("text"));
    const QCommandLineOption askOpt(QStringLiteral("ask"), QStringLiteral("Send a message to the assistant"), QStringLiteral("text"));
    const QCommandLineOption settleOpt(QStringLiteral("settle"), QStringLiteral("Milliseconds to wait before the screenshot (default 600)"),
                                       QStringLiteral("ms"));
    parser.addOptions({datasetOpt, pipelineOpt, runOpt, screenshotOpt, quitAfterOpt, toolOpt, actionOpt, askOpt, settleOpt, strokeOpt, wheelOpt, dropOpt, recordOpt});
    parser.process(app);

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
    // a gated model's token goes with the request that downloads it
    workbench.setHubTokenProvider([] { return sirius::app::toStd(sirius::app::secrets::read(QStringLiteral("hub/token")).trimmed()); });
    sirius::app::MainWindow window(bridge);
    // User operations come from the Python worker. A pipeline given on the
    // command line may use them, so load them first in that case; otherwise
    // after the window is up so start-up stays quick.
    if (parser.isSet(pipelineOpt)) workbench.loadPlugins(false);
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
    const QStringList toolCalls = parser.values(toolOpt);
    const QStringList actions = parser.values(actionOpt);
    sirius::app::ToolApi tools(workbench);
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
    auto letTheWindowCatchUp = [] {
        // let the window react (repaint, refresh) between steps, as it would
        // between a user's actions
        QCoreApplication::processEvents(QEventLoop::AllEvents, 200);
        QCoreApplication::sendPostedEvents();
        QCoreApplication::processEvents(QEventLoop::AllEvents, 200);
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
    auto script = [&] {
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
    const bool scripted = !toolCalls.isEmpty() || !actions.isEmpty() || parser.isSet(askOpt) || parser.isSet(strokeOpt) || parser.isSet(wheelOpt) || parser.isSet(dropOpt);
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
                // tool windows and non-modal dialogs (the plugin manager) beside it too
            for (QWidget* top : QApplication::topLevelWidgets())
                if (top != &window && top->isVisible() && top->isWindow() && !qobject_cast<QMenu*>(top) &&
                    top != QApplication::activeModalWidget() && (top->windowType() == Qt::Tool || qobject_cast<QDialog*>(top))) {
                    QFileInfo fi(path);
                    const QString tag = top->windowType() == Qt::Tool ? QStringLiteral("-tool.") : QStringLiteral("-dialog.");
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
            if (bridge.running() && std::chrono::steady_clock::now() < deadline) {
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
