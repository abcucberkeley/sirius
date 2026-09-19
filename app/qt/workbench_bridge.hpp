#ifndef SIRIUS_APP_WORKBENCH_BRIDGE_HPP
#define SIRIUS_APP_WORKBENCH_BRIDGE_HPP

// The one QObject between the Qt-free Workbench and the widgets: forwards
// the workbench's observer callbacks as signals, runs RunJobs on a worker
// thread and reports their progress on the GUI thread. Every panel takes a
// WorkbenchBridge& and talks to `wb()` directly for reads and edits.
//
// A run's job executes entirely on the worker thread, including the start
// of the Python worker it may need; the GUI thread only polls the job's
// progress until the queued onJobFinished() folds the finished job back
// into the workbench. While the run is active the workbench refuses edits
// (Workbench::canEdit), so the panels can keep calling it as usual.

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include <QObject>
#include <QString>
#include <QThread>
#include <QTimer>

#include "core/workbench.hpp"

namespace sirius::app {

    class WorkbenchBridge : public QObject {
        Q_OBJECT
    public:
        explicit WorkbenchBridge(Workbench& wb, QObject* parent = nullptr);
        ~WorkbenchBridge() override;

        Workbench& wb() noexcept { return wb_; }
        const Workbench& wb() const noexcept { return wb_; }

        // Starts a run of step `target` (-1 = all) on the worker thread; false
        // (with a log line) when nothing can run or a run is active.
        bool startRun(int target = -1);
        void cancelRun();
        bool running() const noexcept { return wb_.running(); }

        // Any other long task (an export, a probe) on the same worker thread:
        // `task` receives a progress callback and a cancellation query and
        // may throw; the outcome arrives as taskFinished. One task at a time;
        // false when one is already running.
        using TaskProgress = std::function<void(double, const std::string&)>;
        using TaskCancelled = std::function<bool()>;
        using Task = std::function<void(const TaskProgress&, const TaskCancelled&)>;
        bool startTask(const QString& label, Task task);
        void cancelTask();
        bool taskRunning() const noexcept { return taskActive_.load(); }
        QString taskLabel() const { return taskLabel_; }

        // Decode on the worker thread, then install the dataset on the GUI
        // thread. False when a run or another task is already active.
        bool openDatasetAsync(const std::string& path, OpenOptions options);

    signals:
        void datasetChanged();
        void pipelineChanged();
        void stepChanged(int index);
        void selectionChanged();
        void viewedStepChanged();
        void viewStateChanged();
        void outputsChanged();
        void labelsChanged(quint64 stepId);
        void runStarted();
        void runProgress(double fraction, int stepIndex, const QString& message);
        void runFinished(bool ok, const QString& error);
        void historyChanged();
        void backendChanged();
        void operationsChanged();   // plugins (re)loaded: the operation registry changed
        void logged(const QString& line);
        void taskStarted(const QString& label);
        void taskProgress(double fraction, const QString& message);
        void taskFinished(bool ok, const QString& error);
        // A run started or finished: what the workbench refuses to edit
        // changed with it (Workbench::canEdit).
        void runStateChanged();

    private:
        // The Workbench::Observer lives in a relay object so its callbacks can
        // share names with the signals they forward to.
        struct Relay;
        friend struct Relay;
        void onRunStateChanged();

        void pollProgress();
        void onJobFinished();
        void onTaskFinished();

        Workbench& wb_;
        std::unique_ptr<Relay> relay_;
        QThread worker_;
        QObject* dispatcher_ = nullptr;      // lives on worker_
        QTimer progressTimer_;
        std::shared_ptr<RunJob> job_;

        // task state (written on the worker thread, read on the GUI thread)
        std::atomic<bool> taskActive_{false};
        std::atomic<bool> taskCancel_{false};
        std::atomic<double> taskFraction_{0.0};
        std::mutex taskMutex_;
        std::string taskMessage_;
        std::string taskError_;
        QString taskLabel_;
        // What a finished task leaves for the GUI thread to do, run from
        // onTaskFinished (openDatasetAsync: install the decoded dataset).
        // Guarded by taskMutex_.
        std::function<void()> taskCompletion_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_WORKBENCH_BRIDGE_HPP
