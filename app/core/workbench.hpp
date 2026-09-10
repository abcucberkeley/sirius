#ifndef SIRIUS_APP_WORKBENCH_HPP
#define SIRIUS_APP_WORKBENCH_HPP

// The session: one dataset, one pipeline, the executor's cached outputs,
// the undo history and the viewer state, behind a single Qt-free facade that
// the widgets, the assistant's tool API and the tests all drive the same
// way. Every edit goes through here so it is undoable and observed.
//
// Threading: the workbench is single-threaded (the GUI thread). Runs are
// prepared here as RunJob objects, executed by the caller on a worker
// thread (RunJob::execute is self-contained: it also obtains the Python or
// HPC worker there, so the GUI never waits for a process to start) and
// folded back in with finishRun() on the GUI thread; progress is read from
// the job's atomics, the results only once finished() is true.
//
// While a run is active the pipeline, the dataset, the caches, the history
// and the label volumes are frozen: every such edit is refused with a log
// line and a false / zero / early return (canEdit() says so up front), so
// the worker thread sees the same pipeline and outputs from start to end.
// Selection, viewing and view-state changes stay allowed.

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "core/array_source.hpp"
#include "core/dataset.hpp"
#include "core/executor.hpp"
#include "core/history.hpp"
#include "core/labels.hpp"
#include "core/session_log.hpp"
#include "core/operation.hpp"
#include "core/pipeline.hpp"
#include "core/rpc.hpp"

namespace sirius::app {

    enum class ViewMode { Ortho,
                          Volume,
                          Compare };
    enum class ViewerTool { Navigate,
                            Probe,
                            Measure,
                            Roi,
                            Paint };
    enum class PaintTool { Brush,
                           Erase,
                           Fill,
                           Pick,
                           Merge,
                           Split,
                           Delete,
                           Lasso };

    const char* toString(ViewMode m) noexcept;       // "ortho" "3d" "compare"
    const char* toString(ViewerTool t) noexcept;     // "nav" "probe" "measure" "roi" "paint"
    const char* toString(PaintTool t) noexcept;
    std::optional<ViewMode> viewModeFromString(const std::string& s) noexcept;
    std::optional<ViewerTool> viewerToolFromString(const std::string& s) noexcept;
    std::optional<PaintTool> paintToolFromString(const std::string& s) noexcept;

    struct ViewState {
        ViewMode mode = ViewMode::Ortho;
        ViewerTool tool = ViewerTool::Probe;
        PaintTool paintTool = PaintTool::Brush;
        int brushPx = 18;
        bool paint3d = true;
        Index z = 0, t = 0;
        Index cx = 0, cy = 0;                   // crosshair voxel (x, y)
        bool crosshair = true;
        bool labels = false;
        bool boundingBox = true;
        bool scaleBar = true;
        bool syncZT = true;
        std::vector<bool> channelVisible;       // resized to the viewed output's channels
        double zoom = 1.0;                      // 1 = fit
        double panX = 0.0, panY = 0.0;          // screen pixels
        double yaw = 35.0, pitch = 22.0;        // 3D
        std::array<double, 2> clipZ{0.0, 1.0};
        double labelOpacity = 0.45;
        std::uint32_t selectedLabel = 0;
        bool soloLabel = false;                 // draw only the selected label (slices and 3D)
        // The ortho panes scale z by the voxel aspect, so what is on screen is
        // physically proportioned -- which is what you want of a result and not
        // what you want when checking the grid a reconstruction was built on.
        // Off draws one row per plane.
        bool physicalZ = true;

        bool channelOn(Index c) const noexcept {
            return c < 0 || static_cast<std::size_t>(c) >= channelVisible.size() || channelVisible[static_cast<std::size_t>(c)];
        }
        nlohmann::json toJson() const;
        static ViewState fromJson(const nlohmann::json& j);
        static ViewState fromJson(const nlohmann::json& j, const ViewState& base);
    };

    struct RunProgress {
        std::atomic<double> fraction{0.0};
        std::atomic<int> stepIndex{-1};
        std::mutex mutex;
        std::string message;                    // guarded by mutex
        std::string messageCopy() {
            std::lock_guard<std::mutex> g(mutex);
            return message;
        }
        void set(double f, int step, const std::string& m) {
            fraction.store(f);
            stepIndex.store(step);
            std::lock_guard<std::mutex> g(mutex);
            message = m;
        }
    };

    class Workbench;

    struct RemoteConfig {
        std::string host = "localhost";
        int port = 7645;
        std::string token;
    };

    // Connects to (starting when needed) the local Python worker; installed
    // by the Qt layer, called on the thread that executes the run.
    using LocalWorkerLauncher = std::function<std::unique_ptr<RemoteWorker>()>;

    // One run, prepared on the GUI thread, executed anywhere.
    //
    // Contract: execute() runs once, on any thread; while it runs only
    // progress(), cancel() and finished() may be called from elsewhere. The
    // results (error, reports, output, seconds) are published by the
    // release store in finished() and may be read only after finished()
    // returned true on the reading thread: reading them earlier throws
    // std::logic_error.
    class RunJob {
    public:
        // Pipeline snapshot, target step index and context are fixed at creation.
        int target() const noexcept { return target_; }
        const Pipeline& pipeline() const noexcept { return pipeline_; }
        RunProgress& progress() noexcept { return progress_; }
        void cancel() noexcept { cancelled_.store(true); }
        bool cancelled() const noexcept { return cancelled_.load(); }

        // Blocking; never throws (errors land in error()). Obtains the worker
        // the run needs first (the Python worker, or the HPC connection).
        void execute();
        bool finished() const noexcept { return finished_.load(std::memory_order_acquire); }
        bool succeeded() const { return finished() && error_.empty(); }
        bool wasCancelled() const {
            requireFinished("wasCancelled");
            return cancelledResult_;
        }
        const std::string& error() const {
            requireFinished("error");
            return error_;
        }
        const std::vector<StepReport>& reports() const {
            requireFinished("reports");
            return reports_;
        }
        std::shared_ptr<const StepOutput> output() const {
            requireFinished("output");
            return output_;
        }
        double seconds() const {
            requireFinished("seconds");
            return seconds_;
        }

    private:
        friend class Workbench;
        void requireFinished(const char* what) const;
        void connectWorker();                          // on the executing thread

        Pipeline pipeline_;
        int target_ = 0;
        Executor* executor_ = nullptr;
        StepContext ctx_;
        Backend backend_ = Backend::Cpu;
        bool needsWorker_ = false;                     // a step wants the Python worker
        LocalWorkerLauncher launcher_;
        RemoteConfig remoteConfig_;                    // Backend::Hpc
        std::unique_ptr<RemoteWorker> ownedRemote_;    // the job's connection
        RunProgress progress_;
        std::atomic<bool> cancelled_{false};
        // written by execute() before the release store to finished_
        std::atomic<bool> finished_{false};
        bool cancelledResult_ = false;
        std::string error_;
        std::string workerNote_;                       // "Local worker: cuda", for the log
        std::vector<StepReport> reports_;
        std::shared_ptr<const StepOutput> output_;
        double seconds_ = 0.0;
    };

    class Workbench {
    public:
        // What changed; the Qt layer forwards these as signals. Called on the
        // GUI thread, never during a run's worker execution.
        class Observer {
        public:
            virtual ~Observer() = default;
            virtual void datasetChanged() {}
            virtual void pipelineChanged() {}                 // steps added / removed / moved / edited
            virtual void stepChanged(int /*index*/) {}        // one step's params / name / cache / enabled
            virtual void selectionChanged() {}
            virtual void viewedStepChanged() {}
            virtual void viewStateChanged() {}
            virtual void outputsChanged() {}                  // cached outputs / freshness
            virtual void labelsChanged(StepId /*id*/) {}      // label voxels edited
            virtual void runStateChanged() {}                 // started / finished
            virtual void historyChanged() {}
            virtual void backendChanged() {}
            virtual void operationsChanged() {}               // plugins (re)loaded
            virtual void logged(const std::string& /*line*/) {}
        };

        explicit Workbench(std::filesystem::path scratchDir);
        ~Workbench();
        Workbench(const Workbench&) = delete;
        Workbench& operator=(const Workbench&) = delete;

        void addObserver(Observer* o);
        void removeObserver(Observer* o);

        // --- dataset ---------------------------------------------------------
        void openDataset(const std::string& path, const OpenOptions& options = {});

        // --- session recording ------------------------------------------------
        // Writes what the user does to a JSON-lines file: the dataset, every
        // step and parameter change with its old and new value, what each run
        // produced, and every label correction. Meant to be replayed, or used
        // as training data for a model that learns which settings a person
        // reaches for on which data.
        void startRecording(const std::string& path);
        void stopRecording();
        bool recording() const { return session_.recording(); }
        std::string recordingPath() const { return session_.path().string(); }
        std::uint64_t recordedLines() const { return session_.lines(); }
        // Anything the caller wants in the record (a UI action, an export).
        void recordEvent(const std::string& event, const nlohmann::json& fields = nlohmann::json::object());
        void setDataset(std::shared_ptr<ArraySource> source);   // tests, scripted data
        void closeDataset();
        bool hasDataset() const noexcept { return static_cast<bool>(source_); }
        const DatasetMeta& dataset() const noexcept { return datasetMeta_; }
        std::shared_ptr<ArraySource> source() const noexcept { return source_; }

        // --- pipeline (every mutation is one undo entry) ---------------------
        // False while a run is active: every edit below (pipeline, dataset,
        // caches, history, labels) is then refused with a log line.
        bool canEdit() const noexcept { return !running(); }
        const Pipeline& pipeline() const noexcept { return pipeline_; }
        StepId addStep(const std::string& kind, int at = -1);        // 0 when refused
        void removeStep(int index);
        bool moveStep(int index, int delta);
        StepId duplicateStep(int index);
        void setStepEnabled(int index, bool on);
        // Write a named preset of the step's operation into it: an ordinary
        // undoable parameter change, so everything stays editable afterwards.
        // False when the step or the preset does not exist, and also when a
        // run is in progress -- which is logged, as every other refused edit
        // is. A caller that needs to tell those apart asks canEdit() first.
        bool applyPreset(int index, const std::string& presetName);

        void setStepParams(int index, const ParamSet& params, const std::string& label = {},
                           const std::string& mergeKey = {});
        void setStepParam(int index, const std::string& key, const ParamValue& value, const std::string& mergeKey = {});
        void setStepCache(int index, CachePolicy policy);
        void renameStep(int index, const std::string& name);
        void replacePipeline(const Pipeline& p, const std::string& label);
        void loadPipeline(const std::string& path);   // also opens the dataset the Load step names
        // OpenOptions equivalent to a Load step's parameters (page order, voxel size, SIM layout).
        static OpenOptions openOptionsFromLoadParams(const ParamSet& loadParams);
        void savePipeline(const std::string& path) const;
        void loadExamplePipeline();
        std::string pipelinePath() const noexcept { return pipelinePath_; }
        // Copy / paste parameters between steps of the same kind.
        void copyParameters(int index);
        bool pasteParameters(int index);
        bool hasCopiedParameters() const noexcept { return static_cast<bool>(clipboard_); }

        // Per-step description for the UI without running anything.
        std::string stepSummary(int index) const;
        Validation stepValidation(int index) const;
        DatasetMeta inputMetaOf(int index) const;      // meta arriving at the step
        DatasetMeta outputMetaOf(int index) const;     // meta leaving it (predicted)
        std::size_t estimatedBytesOf(int index) const;

        // --- selection & viewing --------------------------------------------
        int selectedIndex() const noexcept { return selected_; }
        int viewedIndex() const noexcept { return viewed_; }
        void select(int index);
        void view(int index);
        const ViewState& viewState() const noexcept { return view_; }
        void setViewState(const ViewState& s);          // notifies when changed
        void setViewMode(ViewMode m);
        void setTool(ViewerTool t);
        void setPaintTool(PaintTool t);
        void setZ(Index z);
        void setT(Index t);
        void setCrosshair(Index x, Index y, Index z);
        void setChannelVisible(Index c, bool on);
        void toggleCrosshair();
        void toggleLabels();
        void toggleSoloLabel();                 // only the selected label is drawn
        // Select a label and put the crosshair and z on it (its bounding box centre).
        void focusLabel(std::uint32_t id);
        bool centreOnLabel(std::uint32_t id);   // crosshair and z to its bounding box centre; false when unknown

        // --- outputs ---------------------------------------------------------
        // Last computed output of step `index` (fresh or stale), or null.
        std::shared_ptr<const StepOutput> output(int index) const;
        bool outputFresh(int index) const;
        // What the viewer should draw for the viewed step: its output if it
        // has one, else the nearest computed upstream output (Load's lazy
        // source at worst). `actualIndex` reports which one it is.
        std::shared_ptr<const StepOutput> displayOutput(int* actualIndex = nullptr) const;
        // Nearest computed output upstream of step `index` (the step's input).
        std::shared_ptr<const StepOutput> upstreamOutput(int index, int* actualIndex = nullptr) const;
        // True while the viewed step is shown as a live preview on its input
        // (OpInfo::livePreview and not run or stale).
        bool viewedIsLivePreview() const;
        // Diagnostics of the selected step: the last run's, or the
        // operation's live preview when it offers one.
        Diagnostics selectedDiagnostics() const;
        Diagnostics diagnosticsOf(int index) const;   // the same for any step, without selecting it
        void clearCache(int index);
        void clearAllCaches();
        std::size_t cachedBytes() const;

        // --- running ---------------------------------------------------------
        Backend backend() const noexcept { return backend_; }
        void setBackend(Backend b);
        int cudaDevice() const noexcept { return cudaDevice_; }
        void setCudaDevice(int index);
        const RemoteConfig& remoteConfig() const noexcept { return remote_; }
        void setRemoteConfig(RemoteConfig c);
        // Steps that need the Python worker (Operation::remoteCapable) get a
        // local worker from this launcher when the backend is not HPC; the
        // Qt layer installs one that spawns app/python/sirius_worker. A run
        // job calls it on its own thread; loadPlugins calls it here.
        using WorkerLauncher = LocalWorkerLauncher;
        void setLocalWorkerLauncher(WorkerLauncher launcher) { launcher_ = std::move(launcher); }
        // Starts the local worker (through the launcher, synchronously: this
        // blocks until the worker answers), registers the user operations it
        // finds (app/python/sirius_worker/plugins.py) and logs the outcome;
        // returns the number registered. `reload` re-imports the files.
        // Refused (0) while a run is active: the registry is in use.
        int loadPlugins(bool reload);
        struct PluginInfo {
            std::string kind, name, file, error;   // error non-empty when the file did not load
        };
        const std::vector<PluginInfo>& plugins() const noexcept { return plugins_; }
        const std::vector<std::string>& pluginDirs() const noexcept { return pluginDirs_; }
        // A job for step `target` (or the last step when -1); null with a log
        // line when nothing can run. The caller executes it and calls
        // finishRun once it is finished (a job that never executed is
        // logged as abandoned).
        std::shared_ptr<RunJob> createRun(int target = -1);
        void finishRun(const std::shared_ptr<RunJob>& job);
        bool running() const noexcept { return static_cast<bool>(activeRun_); }
        std::shared_ptr<RunJob> activeRun() const noexcept { return activeRun_; }
        void cancelRun();

        // --- labels (undoable, on the viewed output) -------------------------
        // The volume shown for the viewed step. It belongs to that step's
        // output (a step that carries its input's labels through owns a
        // copy-on-write view of them, see labels.hpp), so an edit here never
        // changes another step's cached labels.
        std::shared_ptr<LabelVolume> viewedLabels() const;
        // One brush stroke = one undo entry: call beginPaintStroke() on press,
        // paintLabels() on every move and endPaintStroke() on release. The
        // label statistics are brought up to date at the end of the stroke
        // (every other edit updates them at once); a stroke still open is
        // ended by the next stroke, edit or statistics query.
        void beginPaintStroke();
        void paintLabels(Index z, Index y, Index x, bool erase);          // uses brush size / label
        void endPaintStroke();
        void fillLabel(Index z, Index y, Index x);
        void mergeLabels(const std::vector<std::uint32_t>& ids);
        void splitLabel(std::uint32_t id, std::array<Index, 3> a, std::array<Index, 3> b);
        void deleteLabel(std::uint32_t id);
        void setLabelReviewed(std::uint32_t id, bool reviewed);
        void acceptAllReviewed();
        std::uint32_t nextFlaggedLabel(bool forward);

        // --- history & log ---------------------------------------------------
        History& history() noexcept { return history_; }
        void undo();
        void redo();
        const std::vector<std::string>& log() const noexcept { return log_; }
        void logLine(const std::string& line);

        const std::filesystem::path& scratchDir() const noexcept { return executor_.scratchDir(); }
        Executor& executor() noexcept { return executor_; }

    private:
        struct Snapshot {
            nlohmann::json pipeline;
            int selected = 1, viewed = 1;
            ViewState view;
        };
        Snapshot snapshot() const;
        void restore(const Snapshot& s);
        void pushEdit(const std::string& label, const Snapshot& before, const std::string& mergeKey = {});
        void pushCommand(Command c);      // every history push goes through here
        void notify(void (Observer::*fn)());
        void notifyStep(int index);
        void notifyLabels(StepId id);
        void clampSelection();
        void onStepSelected(int index);
        Diagnostics previewDiagnostics(int index) const;
        // True (with a log line naming `what`) when a run is active.
        bool refuseIfRunning(const char* what);
        void installDataset(std::shared_ptr<ArraySource> source, DatasetMeta meta, std::string note);
        // The labels of step `id` as the executor holds them now, or null.
        std::shared_ptr<LabelVolume> labelsOf(StepId id) const;
        // The viewed labels and the step they belong to (0 when none).
        std::shared_ptr<LabelVolume> editableLabels(StepId* id);
        // A label edit on step `id`: every step below it consumed (or
        // carried) the labels as they were, so their outputs are stale.
        void staleBelow(StepId id);
        // The statistics describe one time point: bring them to the one on
        // screen (a time series after tracking keeps its ids across t).
        void syncLabelStats();
        // Undo / redo of a label edit: applies `diff` to the labels of step
        // `id` if they are still the volume the edit was made on, else a
        // logged no-op (the step was re-run or removed since).
        void applyLabelDiff(StepId id, const std::weak_ptr<LabelVolume>& target, const LabelDiff& diff, bool forward);
        void pushLabelCommand(const std::string& label, const std::string& mergeKey, StepId id,
                              const std::shared_ptr<LabelVolume>& labels, std::shared_ptr<LabelDiff> diff);
        void recordLabelDiff(const std::string& label, StepId id, const std::shared_ptr<LabelVolume>& labels,
                             LabelDiff diff);

        std::vector<Observer*> observers_;
        std::shared_ptr<ArraySource> source_;
        DatasetMeta datasetMeta_;
        Pipeline pipeline_;
        std::string pipelinePath_;
        Executor executor_;
        History history_;
        SessionLog session_;
        ViewState view_;
        int selected_ = 1;
        int viewed_ = 1;
        Backend backend_ = Backend::Cuda;
        int cudaDevice_ = 0;
        RemoteConfig remote_;
        std::shared_ptr<RunJob> activeRun_;
        std::vector<std::string> log_;
        std::optional<std::pair<std::string, ParamSet>> clipboard_;
        std::shared_ptr<const StepOutput> loadOutput_;   // the Load step's lazy output
        WorkerLauncher launcher_;
        std::vector<PluginInfo> plugins_;
        std::vector<std::string> pluginDirs_;
        // The first "before" of the merge group the top history entry belongs
        // to (History::mergesWith decides whether it still applies).
        std::optional<std::pair<std::string, Snapshot>> mergeFirst_;
        int strokeCounter_ = 0;
        LabelDiff strokeDiff_;                           // the open stroke so far
        StepId strokeStep_ = 0;
        std::shared_ptr<LabelVolume> strokeLabels_;      // the volume the open stroke edits
        bool strokeOpen_ = false;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_WORKBENCH_HPP
