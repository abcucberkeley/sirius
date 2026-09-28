#ifndef SIRIUS_IMGUI_BRIDGE_HPP
#define SIRIUS_IMGUI_BRIDGE_HPP

// The one object between the GUI-free Workbench and the panels: it observes
// the workbench, runs RunJobs and other long tasks on a worker thread and
// brings their outcome back to the GUI thread. Every panel takes a Bridge&
// and talks to `wb()` directly for reads and edits. (The Qt application's
// WorkbenchBridge, without the QObject.)
//
// Immediate mode changes how a change reaches a panel. A panel that draws
// straight from the workbench needs no notification at all: next frame it
// draws the new state. What is expensive to derive (a rendered slice, a
// texture, a parsed page) is cached against the *revision* of what it was
// derived from: `bridge.rev().outputs` moves whenever
// Workbench::Observer::outputsChanged fires, and so on for every observer
// callback. The few places that must react once, when something happens (a
// failed run raises a box), connect a callback to the matching Signal.
//
// Threads: everything here is called on the GUI thread, except post() and
// wake(), which any thread may call. A run's job executes entirely on the
// worker thread, including the start of the Python worker it may need; the
// GUI thread reads the job's progress every frame (update()) and folds the
// finished job back into the workbench. While the run is active the
// workbench refuses edits (Workbench::canEdit), so the panels can keep
// calling it as usual.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/workbench.hpp"

namespace sirius::app::gui {

    // A list of callbacks. connect() while emitting is allowed; the new
    // callback is first called by the next emit.
    template <class... Args>
    class Signal {
    public:
        using Slot = std::function<void(Args...)>;
        int connect(Slot slot) {
            slots_.emplace_back(++last_, std::move(slot));
            return last_;
        }
        void disconnect(int id) {
            for (auto& s : slots_)
                if (s.first == id) s.second = nullptr;
        }
        void emit(Args... args) const {
            const std::size_t n = slots_.size();   // slots connected meanwhile wait for the next emit
            for (std::size_t i = 0; i < n && i < slots_.size(); ++i) {
                const Slot slot = slots_[i].second;   // a copy: the slot may disconnect itself
                if (slot) slot(args...);
            }
        }

    private:
        std::vector<std::pair<int, Slot>> slots_;
        int last_ = 0;
    };

    // How often each observer callback fired. Compare against a remembered
    // value to know that what a cache was built from has changed.
    struct Revisions {
        std::uint64_t dataset = 1;
        std::uint64_t pipeline = 1;       // steps added / removed / moved / edited
        std::uint64_t step = 1;           // one step's params / name / cache / enabled (any step)
        std::uint64_t selection = 1;
        std::uint64_t viewedStep = 1;
        std::uint64_t viewState = 1;
        std::uint64_t outputs = 1;        // cached outputs / freshness
        std::uint64_t labels = 1;         // label voxels edited (any step)
        std::uint64_t runState = 1;       // a run started or finished
        std::uint64_t history = 1;
        std::uint64_t backend = 1;
        std::uint64_t operations = 1;     // plugins (re)loaded: the operation registry changed
        std::uint64_t log = 1;
        // Anything that changes what the pipeline looks like to a panel.
        std::uint64_t anyPipeline() const noexcept { return pipeline + step + selection + operations; }
    };

    class Bridge {
    public:
        explicit Bridge(Workbench& wb);
        ~Bridge();
        Bridge(const Bridge&) = delete;
        Bridge& operator=(const Bridge&) = delete;

        Workbench& wb() noexcept { return wb_; }
        const Workbench& wb() const noexcept { return wb_; }
        const Revisions& rev() const noexcept { return rev_; }

        // --- runs ------------------------------------------------------------
        // Starts a run of step `target` (-1 = all) on the worker thread; false
        // (with a log line) when nothing can run or a run is active.
        bool startRun(int target = -1);
        void cancelRun();
        bool running() const noexcept { return wb_.running(); }
        // Progress of the active run, read from the job every frame.
        double runFraction() const noexcept { return runFraction_; }
        int runStep() const noexcept { return runStep_; }
        const std::string& runMessage() const noexcept { return runMessage_; }

        // --- tasks -----------------------------------------------------------
        // Any other long task (an export, a probe) on the same worker thread:
        // `task` receives a progress callback and a cancellation query and may
        // throw; the outcome arrives as taskFinished. One task at a time;
        // false when one is already running or a run is active.
        using TaskProgress = std::function<void(double, const std::string&)>;
        using TaskCancelled = std::function<bool()>;
        using Task = std::function<void(const TaskProgress&, const TaskCancelled&)>;
        // `completion` runs on the GUI thread once the task finished without
        // an error and was not cancelled (it may throw; that fails the task).
        bool startTask(const std::string& label, Task task, std::function<void()> completion = {});
        void cancelTask();
        bool taskRunning() const noexcept { return taskActive_.load(); }
        const std::string& taskLabel() const noexcept { return taskLabel_; }
        double taskFraction() const noexcept { return taskFraction_.load(); }
        std::string taskMessage() const;
        bool busy() const noexcept { return running() || taskRunning(); }

        // Decode on the worker thread, then install the dataset on the GUI
        // thread. False when a run or another task is already active.
        bool openDatasetAsync(const std::string& path, OpenOptions options);

        // --- the GUI thread's queue ------------------------------------------
        // Runs `fn` on the GUI thread at the start of the next frame, outside
        // of any Dear ImGui window. Callable from any thread.
        void post(std::function<void()> fn);
        // Asks the main loop for another frame (it sleeps while nothing
        // happens). Callable from any thread. The application installs the
        // function that does it.
        void wake() const;
        void setWaker(std::function<void()> waker) { waker_ = std::move(waker); }
        // Once per frame, before the frame is built: runs the posted
        // functions, reads the progress and folds finished jobs back in.
        void update();

        // --- signals (GUI thread) ----------------------------------------------
        Signal<> datasetChanged;
        Signal<> pipelineChanged;
        Signal<int> stepChanged;
        Signal<> selectionChanged;
        Signal<> viewedStepChanged;
        Signal<> viewStateChanged;
        Signal<> outputsChanged;
        Signal<StepId> labelsChanged;
        Signal<> runStarted;
        Signal<bool, const std::string&> runFinished;      // ok, error ("" when cancelled)
        Signal<> runStateChanged;                          // both edges of a run
        Signal<> historyChanged;
        Signal<> backendChanged;
        Signal<> operationsChanged;
        Signal<const std::string&> logged;
        Signal<const std::string&> taskStarted;
        Signal<bool, const std::string&> taskFinished;     // ok, error

    private:
        struct Relay;
        friend struct Relay;
        void onJobFinished();
        void onTaskFinished();
        void enqueue(std::function<void()> job);   // onto the worker thread
        void workerLoop();

        Workbench& wb_;
        std::unique_ptr<Relay> relay_;
        Revisions rev_;
        std::function<void()> waker_;

        // the worker thread and its queue
        std::thread worker_;
        std::mutex queueMutex_;
        std::condition_variable queueReady_;
        std::deque<std::function<void()>> queue_;
        bool quit_ = false;

        // posted to the GUI thread
        std::mutex postedMutex_;
        std::vector<std::function<void()>> posted_;

        // run state
        std::shared_ptr<RunJob> job_;
        std::atomic<bool> jobDone_{false};
        double runFraction_ = 0.0;
        int runStep_ = -1;
        std::string runMessage_;

        // task state (written on the worker thread, read on the GUI thread)
        std::atomic<bool> taskActive_{false};
        std::atomic<bool> taskDone_{false};
        std::atomic<bool> taskCancel_{false};
        std::atomic<double> taskFraction_{0.0};
        mutable std::mutex taskMutex_;
        std::string taskMessage_;
        std::string taskError_;
        std::string taskLabel_;
        // What a finished task leaves for the GUI thread to do. Guarded by taskMutex_.
        std::function<void()> taskCompletion_;
    };

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_BRIDGE_HPP
