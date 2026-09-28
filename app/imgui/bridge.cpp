#include "imgui/bridge.hpp"

#include <exception>

#include "core/cancel.hpp"

namespace sirius::app::gui {

    struct Bridge::Relay final : Workbench::Observer {
        Bridge& b;
        explicit Relay(Bridge& bridge) : b(bridge) {}
        void datasetChanged() override {
            ++b.rev_.dataset;
            b.datasetChanged.emit();
        }
        void pipelineChanged() override {
            ++b.rev_.pipeline;
            b.pipelineChanged.emit();
        }
        void stepChanged(int index) override {
            ++b.rev_.step;
            b.stepChanged.emit(index);
        }
        void selectionChanged() override {
            ++b.rev_.selection;
            b.selectionChanged.emit();
        }
        void viewedStepChanged() override {
            ++b.rev_.viewedStep;
            b.viewedStepChanged.emit();
        }
        void viewStateChanged() override {
            ++b.rev_.viewState;
            b.viewStateChanged.emit();
        }
        void outputsChanged() override {
            ++b.rev_.outputs;
            b.outputsChanged.emit();
        }
        void labelsChanged(StepId id) override {
            ++b.rev_.labels;
            b.labelsChanged.emit(id);
        }
        void runStateChanged() override {
            ++b.rev_.runState;
            if (b.wb_.running()) b.runStarted.emit();
            // Both edges: the menus and panels grey out every edit the
            // workbench will refuse while the run is active.
            b.runStateChanged.emit();
        }
        void historyChanged() override {
            ++b.rev_.history;
            b.historyChanged.emit();
        }
        void backendChanged() override {
            ++b.rev_.backend;
            b.backendChanged.emit();
        }
        void operationsChanged() override {
            ++b.rev_.operations;
            b.operationsChanged.emit();
        }
        void logged(const std::string& line) override {
            ++b.rev_.log;
            b.logged.emit(line);
        }
    };

    Bridge::Bridge(Workbench& wb) : wb_(wb), relay_(std::make_unique<Relay>(*this)) {
        wb_.addObserver(relay_.get());
        worker_ = std::thread([this] { workerLoop(); });
    }

    Bridge::~Bridge() {
        wb_.removeObserver(relay_.get());
        if (job_) job_->cancel();
        taskCancel_.store(true);   // a load in flight stops at its next progress call and installs nothing
        {
            const std::lock_guard<std::mutex> g(queueMutex_);
            quit_ = true;
        }
        queueReady_.notify_all();
        // The queued jobs finish on the worker thread before it ends, so
        // nothing touches the workbench after this returns.
        if (worker_.joinable()) worker_.join();
    }

    void Bridge::workerLoop() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(queueMutex_);
                queueReady_.wait(lock, [this] { return quit_ || !queue_.empty(); });
                if (queue_.empty()) return;   // quit_, and nothing left to finish
                job = std::move(queue_.front());
                queue_.pop_front();
            }
            job();
        }
    }

    void Bridge::enqueue(std::function<void()> job) {
        {
            const std::lock_guard<std::mutex> g(queueMutex_);
            queue_.push_back(std::move(job));
        }
        queueReady_.notify_one();
    }

    void Bridge::post(std::function<void()> fn) {
        {
            const std::lock_guard<std::mutex> g(postedMutex_);
            posted_.push_back(std::move(fn));
        }
        wake();
    }

    void Bridge::wake() const {
        // Called under the lock, so that once setWaker(nullptr) has returned
        // no wake is running or can start, and none overlaps glfwTerminate in
        // ~App. The waker (glfwPostEmptyEvent) never calls back into the
        // Bridge, so holding the lock cannot deadlock.
        const std::lock_guard<std::mutex> g(wakerMutex_);
        if (waker_) waker_();
    }

    void Bridge::setWaker(std::function<void()> waker) {
        // The old waker is destroyed once the lock is released: whatever its
        // captures do when they go then runs without the lock held.
        std::function<void()> old;
        {
            const std::lock_guard<std::mutex> g(wakerMutex_);
            old.swap(waker_);
            waker_ = std::move(waker);
        }
    }

    void Bridge::update() {
        std::vector<std::function<void()>> posted;
        {
            const std::lock_guard<std::mutex> g(postedMutex_);
            posted.swap(posted_);
        }
        for (auto& fn : posted)
            if (fn) fn();

        if (job_) {
            RunProgress& p = job_->progress();
            runFraction_ = p.fraction.load();
            runStep_ = p.stepIndex.load();
            runMessage_ = p.messageCopy();
            if (jobDone_.exchange(false)) onJobFinished();
        }
        if (taskActive_.load() && taskDone_.exchange(false)) onTaskFinished();
    }

    // --- runs ------------------------------------------------------------------

    bool Bridge::startRun(int target) {
        if (wb_.running()) {
            wb_.logLine("A run is already in progress.");
            return false;
        }
        if (taskActive_.load()) {
            wb_.logLine("Wait for " + taskLabel_ + " to finish.");
            return false;
        }
        std::shared_ptr<RunJob> job = wb_.createRun(target);
        if (!job) return false;
        job_ = job;
        jobDone_.store(false);
        runFraction_ = 0.0;
        runStep_ = job->target();
        runMessage_ = "Starting\xE2\x80\xA6";
        enqueue([this, job] {
            job->execute();
            jobDone_.store(true);
            wake();
        });
        return true;
    }

    void Bridge::cancelRun() {
        if (job_) job_->cancel();
        wb_.cancelRun();
    }

    void Bridge::onJobFinished() {
        const std::shared_ptr<RunJob> job = std::move(job_);
        job_.reset();
        if (!job) return;
        // set by the worker thread after execute() returned: the job's results
        // are published (RunJob::finished) before this runs
        const bool ok = job->finished() && job->succeeded();
        // A cancelled run finished the way the user asked it to: it carries no
        // error for the window to show.
        const bool cancelled = job->finished() && job->wasCancelled();
        const std::string error = cancelled ? std::string() : job->finished() ? job->error()
                                                                              : std::string("the run did not finish");
        wb_.finishRun(job);
        runFraction_ = 0.0;
        runStep_ = -1;
        runMessage_.clear();
        runFinished.emit(ok, error);
    }

    // --- tasks -----------------------------------------------------------------

    bool Bridge::startTask(const std::string& label, Task task, std::function<void()> completion) {
        if (wb_.running()) {
            wb_.logLine("A run is already in progress.");
            return false;
        }
        if (taskActive_.load()) {
            wb_.logLine("Another task is still running: " + taskLabel_);
            return false;
        }
        taskActive_.store(true);
        taskDone_.store(false);
        taskCancel_.store(false);
        taskFraction_.store(0.0);
        {
            const std::lock_guard<std::mutex> g(taskMutex_);
            taskMessage_.clear();
            taskError_.clear();
            taskCompletion_ = std::move(completion);
        }
        taskLabel_ = label;
        taskStarted.emit(label);
        enqueue([this, task = std::move(task)] {
            try {
                task(
                    [this](double f, const std::string& m) {
                        taskFraction_.store(f);
                        {
                            const std::lock_guard<std::mutex> g(taskMutex_);
                            taskMessage_ = m;
                        }
                        wake();
                    },
                    [this] { return taskCancel_.load(); });
            } catch (const std::exception& e) {
                const std::lock_guard<std::mutex> g(taskMutex_);
                if (!isCancellation(e)) taskError_ = e.what();
            } catch (...) {
                const std::lock_guard<std::mutex> g(taskMutex_);
                taskError_ = "unknown error";
            }
            taskDone_.store(true);
            wake();
        });
        return true;
    }

    void Bridge::cancelTask() { taskCancel_.store(true); }

    std::string Bridge::taskMessage() const {
        const std::lock_guard<std::mutex> g(taskMutex_);
        return taskMessage_;
    }

    void Bridge::onTaskFinished() {
        std::string error;
        std::function<void()> completion;
        {
            const std::lock_guard<std::mutex> g(taskMutex_);
            error = taskError_;
            completion.swap(taskCompletion_);
        }
        // The GUI-thread half of the task, unless it failed or was cancelled
        // (a cancelled load must not replace the dataset on screen).
        if (completion && error.empty() && !taskCancel_.load()) {
            try {
                completion();
            } catch (const std::exception& e) {
                if (!isCancellation(e)) error = e.what();
            }
        }
        taskActive_.store(false);
        // taskLabel_ is kept: the window titles its "task failed" box with
        // taskLabel(). The next startTask() overwrites it.
        if (error.empty() && taskCancel_.load()) wb_.logLine(taskLabel_ + ": cancelled");
        else if (error.empty()) wb_.logLine(taskLabel_ + ": done");
        else wb_.logLine(taskLabel_ + ": " + error);
        taskFinished.emit(error.empty(), error);
    }

    bool Bridge::openDatasetAsync(const std::string& path, OpenOptions options) {
        if (wb_.running()) {
            wb_.logLine("A run is in progress: cancel it or wait before opening a dataset.");
            return false;
        }
        options.progress = {};
        return startTask("Loading dataset", [this, path, options](const TaskProgress& progress, const TaskCancelled& cancelled) {
            OpenOptions o = options;
            o.progress = [&](double f, const std::string& m) {
                if (cancelled()) throw CancelledError{};
                progress(f, m);
            };
            OpenResult opened = sirius::app::openDataset(path, o);
            if (cancelled()) throw CancelledError{};
            o.progress = {};
            // The dataset is installed on the GUI thread, from onTaskFinished.
            auto result = std::make_shared<OpenResult>(std::move(opened));
            const std::lock_guard<std::mutex> g(taskMutex_);
            taskCompletion_ = [this, result, path, o] { wb_.adoptDataset(std::move(*result), path, o); };
        });
    }

} // namespace sirius::app::gui
