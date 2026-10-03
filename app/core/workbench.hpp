#ifndef SIRIUS_APP_WORKBENCH_HPP
#define SIRIUS_APP_WORKBENCH_HPP

// The session: one dataset, one pipeline, the executor's cached outputs,
// the undo history and the viewer state, behind a single GUI-free facade that
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
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "core/array_source.hpp"
#include "core/dataset.hpp"
#include "core/executor.hpp"
#include "core/history.hpp"
#include "core/labels.hpp"
#include "core/session_log.hpp"
#include "core/operation.hpp"
#include "core/ops/contrast.hpp"
#include "core/pipeline.hpp"
#include "core/rpc.hpp"
#include "core/tracks.hpp"
#include "core/worker_error.hpp"

namespace sirius::app {

    class RemoteDatasets;

    enum class ViewMode { Ortho,
                          Volume,
                          Compare };
    enum class ViewerTool { Navigate,
                            Probe,
                            Measure,
                            Roi,
                            Paint,
                            // places the points of a Prompt step (params.hpp)
                            Prompt };
    enum class PaintTool { Brush,
                           Erase,
                           Fill,
                           Pick,
                           Merge,
                           Split,
                           Delete,
                           Lasso };
    // What a drag (or a click) with the Prompt tool places. A box is the
    // default: one box gives the best mask of any single prompt the prompt
    // decoder was measured on (median IoU .73, a centre click .61).
    enum class PromptMode { Box,
                            Click,
                            Scribble };

    const char* toString(ViewMode m) noexcept;       // "ortho" "3d" "compare"
    const char* toString(ViewerTool t) noexcept;     // "nav" "probe" "measure" "roi" "paint" "prompt"
    const char* toString(PaintTool t) noexcept;
    const char* toString(PromptMode m) noexcept;    // "box" "click" "scribble"
    std::optional<ViewMode> viewModeFromString(const std::string& s) noexcept;
    std::optional<ViewerTool> viewerToolFromString(const std::string& s) noexcept;
    std::optional<PaintTool> paintToolFromString(const std::string& s) noexcept;
    std::optional<PromptMode> promptModeFromString(const std::string& s) noexcept;

    struct ViewState {
        ViewMode mode = ViewMode::Ortho;
        ViewerTool tool = ViewerTool::Probe;
        PaintTool paintTool = PaintTool::Brush;
        PromptMode promptMode = PromptMode::Box;
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
        // Tracked labels: each track's centroid path over the slices, and
        // whether the crosshair stays on the selected track as t changes.
        bool trajectories = true;
        bool followTrack = false;
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
        // > 0: reach host:port through this local SOCKS5 proxy, the SSH
        // connection of a cluster session (core/cluster.hpp), which resolves
        // the compute node's name on the cluster.
        int socksPort = 0;
        // How a connection is made instead of host:port when set: the cluster
        // session's connectWorker (its endpoint as it is now, after a
        // reconnect too), an in-process engine in the tests.
        std::function<std::unique_ptr<RemoteWorker>(const std::function<bool()>& cancelled)> connect;
        // What the HPC worker said of itself when the host connected to it
        // (`known`): the "engine" block of SIRIUS's engine (sirius-cli serve,
        // core/engine_server.hpp), null for the Python worker. Not known
        // (sirius-cli --hpc): the run finds out when it connects.
        bool known = false;
        nlohmann::json engine;
        // Known to have no engine (`known`, a null `engine`): why, in the
        // words the run buttons show ("HPC: no SIRIUS engine on the cluster
        // \xE2\x80\x94 open Cluster to fix"); "" = kHpcNoEngine.
        std::string noEngine;
        // Where the results computed there are held, for the log and the ops
        // row: "fiona · n0042 · job 4711".
        std::string where;

        // SIRIUS's engine (known so far): the run goes to it whole.
        bool hasEngine() const { return known && engine.is_object(); }
        // A connection as configured.
        std::unique_ptr<RemoteWorker> open(const std::function<bool()>& cancelled = {}) const;
    };

    // A file of this computer a run on the cluster node needs there: the
    // dataset, a flat-field image, a PSF. Uploaded only when the user said so.
    struct UploadFile {
        std::string path;            // as the pipeline names it
        std::uint64_t bytes = 0;
        std::string stamp;           // size and modification time (fileStamp)
    };

    // Connects to (starting when needed) the local Python worker; installed
    // by the host (the window, sirius-cli), called on the thread that
    // executes the run. It throws WorkerStartError when the worker cannot
    // start, which the run and the plugin load keep for the host.
    using LocalWorkerLauncher = std::function<std::unique_ptr<RemoteWorker>()>;

    // Whether a run may start now, and when not, why: one reason for every
    // run entry point (the Run buttons, the Process menu, the shortcuts, the
    // assistant's and the agents' run tool), shown where the button is
    // rather than refused after it is pressed. The HPC backend runs nothing
    // without SIRIUS's engine on the cluster: none started, the engine
    // missing, the engine gone, the job ended, the connection lost.
    struct RunGate {
        bool enabled = true;
        std::string why;
    };
    inline constexpr const char* kHpcNoEngine = "HPC: no SIRIUS engine on the cluster \xE2\x80\x94 open Cluster to fix";
    // The refusal of a built-in step on a cluster job without SIRIUS's engine:
    // "Step 02 Contrast needs SIRIUS's C++ engine on the cluster, and this job
    // has none: open Cluster ▸ Job ▸ More options, set Engine builds folder,
    // then Restart worker."
    std::string noEngineRefusal(int index, const std::string& stepName);

    // Why the last createRun() returned null, for a caller that answers
    // with more than the log line (the headless tools map it to an error
    // code). None after a createRun() that made a job.
    struct RunRefusal {
        enum class Kind { None,
                          Running,
                          NoDataset,
                          Invalid,
                          NoLauncher,
                          // the HPC job runs the Python worker only: built-in steps cannot run there
                          NoEngine,
                          // the HPC job's engine is another SIRIUS whose operations differ
                          EngineMismatch,
                          // files of this computer have to be uploaded first: ask (`uploads`), never by default
                          NeedsUpload };
        Kind kind = Kind::None;
        int step = -1;               // Invalid / NoEngine: the step that cannot run (its index, 0 = Load)
        std::string message;         // the line that was logged
        std::vector<UploadFile> uploads;   // NeedsUpload: what would be sent, and how much
    };

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
        // After finished(): the worker failure the run stopped on, if that was
        // the reason (a local worker that did not start). error() then reads
        // "Worker unavailable: <its message> <its hint>". Empty for every
        // other failure, and for a run that was cancelled.
        const std::optional<WorkerStartError>& workerFailure() const {
            requireFinished("workerFailure");
            return workerFailure_;
        }
        // True when the run went to SIRIUS's engine on the cluster node.
        bool ranOnEngine() const {
            requireFinished("ranOnEngine");
            return onEngine_;
        }

    private:
        friend class Workbench;
        void requireFinished(const char* what) const;
        void connectWorker();                          // on the executing thread
        // Backend::Hpc with SIRIUS's engine: the whole run on the node (uploads
        // first), the outputs seeded here as handles (NodeOutputSource).
        void executeOnEngine();
        void upload(const UploadFile& f, std::map<std::string, std::string>& nodePaths);

        Pipeline pipeline_;
        int target_ = 0;
        Executor* executor_ = nullptr;
        StepContext ctx_;
        Backend backend_ = Backend::Cpu;
        bool needsWorker_ = false;                     // a step wants the Python worker
        LocalWorkerLauncher launcher_;
        RemoteConfig remoteConfig_;                    // Backend::Hpc
        std::string workerHint_;                       // Workbench::setWorkerHint, when the job was made
        std::unique_ptr<RemoteWorker> ownedRemote_;    // the job's connection
        std::shared_ptr<RemoteDatasets> nodeDatasets_; // what node outputs are drawn through
        std::vector<UploadFile> uploads_;              // what the user agreed to upload
        std::map<std::string, std::string> nodePaths_; // uploaded: this computer's path -> the node's (in and out)
        std::string engineSession_;                    // the engine's session, from its result
        std::string where_;                            // the node that answered: "fiona · n0042 · job 4711"
        bool onEngine_ = false;
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
        std::optional<WorkerStartError> workerFailure_;
    };

    class Workbench {
    public:
        // What changed; the GUI forwards these to its panels. Called on the
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
        // Install an already-opened result as the dataset. The GUI decodes on
        // the worker thread, then calls this on the GUI thread.
        void adoptDataset(OpenResult opened, const std::string& path, const OpenOptions& options = {});

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
        // 0 when refused. seedParams false keeps the operation's defaults
        // instead of seeding them from the step's current input (the tool API
        // does that: an automatic window follows its input when it runs).
        StepId addStep(const std::string& kind, int at = -1, bool seedParams = true);
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

        // --- tracks (tracked labels, core/tracks.hpp) ----------------------
        // One row per track of the viewed labels; empty when they are not
        // tracked. Distances in microns from the viewed output's voxel size.
        std::vector<TrackSummary> viewedTrackSummaries() const;
        // Select a track and bring it into view: the time point moves to the
        // nearest one the track exists in, crosshair and z onto its centroid.
        // False when the viewed labels have no such track.
        bool focusTrack(std::uint32_t id);
        void setFollowTrack(bool on);

        // --- outputs ---------------------------------------------------------
        // Last computed output of step `index` (fresh or stale), or null.
        std::shared_ptr<const StepOutput> output(int index) const;
        bool outputFresh(int index) const;
        // What the viewer should draw for the viewed step: its output if it
        // has one, else the nearest computed upstream output (Load's lazy
        // source at worst). `actualIndex` reports which one it is.
        std::shared_ptr<const StepOutput> displayOutput(int* actualIndex = nullptr) const;
        // The metadata of that output, or the viewed step's predicted output
        // when nothing is on screen yet: what z, t and the crosshair are
        // clamped to. A step that has not run is shown on its input, whose
        // shape can differ from the step's own (a crop, a SIM reconstruction).
        DatasetMeta displayedMeta() const;
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
        // CUDA ordinal, or kAllCudaDevices (-1) to round-robin every volume
        // across all visible GPUs.
        static constexpr int kAllCudaDevices = -1;
        int cudaDevice() const noexcept { return cudaDevice_; }
        void setCudaDevice(int index);
        // Backend::Hpc: the worker job's GPU or its CPU, sent with every
        // request (StepContext::hpcDevice), so a switch needs no new job.
        // Session state, not the pipeline's: no undo entry, and no step goes
        // stale (a result does not depend on where it was computed).
        HpcDevice hpcDevice() const noexcept { return hpcDevice_; }
        void setHpcDevice(HpcDevice d);
        const RemoteConfig& remoteConfig() const noexcept { return remote_; }
        void setRemoteConfig(RemoteConfig c);
        // What the worker beside the engine on the node says about a model on
        // the cluster ("cluster://host/path"): its model_info, asked once and
        // remembered; nullopt while it is asked, or with `error` set when it
        // cannot be (no engine, the worker's refusal).
        std::optional<nlohmann::json> clusterModelInfo(const std::string& clusterPath, std::string* error = nullptr) const;
        // Whether a run may start (RunGate above); createRun refuses with
        // the same reason, so a caller that did not ask first is told so.
        RunGate runGate() const;
        // Every result held by the engine session `session` went away (its job
        // ended: `reason` says how): the steps keep their diagnostics, are no
        // longer fresh, and say so. "" = every engine session. Returns the
        // number of steps affected.
        int nodeOutputsGone(const std::string& session, const std::string& reason);
        // The engine session the results here were computed by ("" none).
        const std::string& engineSession() const noexcept { return engineSession_; }
        // The files of this computer a run to `target` on the HPC engine
        // would have to upload, and whether the user agreed to each (the
        // run asks for the rest: RunRefusal::Kind::NeedsUpload).
        std::vector<UploadFile> filesToUpload(int target) const;
        void allowUploads(const std::vector<UploadFile>& files);
        // "node A100", "this computer · CPU": where step `index`'s last output
        // was computed, "" when it has none; and the reason its data is gone.
        std::string placementOf(int index) const;
        // Answers of the engine on the node (previews, validations) arrive on
        // a thread of their own: `wake` (any thread) asks the host to call
        // poll() on its thread, which tells the observers. True when any arrived.
        void setWakeHandler(std::function<void()> wake);
        bool poll();

        // --- the Contrast step's window on its input (core/ops/contrast.hpp) ---
        // What the parameter panel and the viewer read off the input of
        // Contrast step `index`: computed here from a few planes of an input
        // on this computer; for one that stays on the cluster, by the engine
        // there, from its preview (step_preview), so not a plane comes here.
        // nullopt without an input, or while the node is asked (poll()).
        // On the node every one of these is one measurement of the input
        // (its histograms, the automatic window, the data range), the same
        // whatever min / max / gamma are: a window dragged asks nothing.
        std::optional<ContrastWindow> contrastWindowOf(int index, const ParamSet& params, Index c, bool wantRange) const;
        // The parameters behind the Auto and Reset buttons, the same way.
        std::optional<ParamSet> contrastAutoOf(int index, const ParamSet& current) const;
        std::optional<ParamSet> contrastResetOf(int index, const ParamSet& current) const;
        // The Auto and Reset buttons of Contrast step `index`: applied at once
        // (an undoable parameter change) for an input on this computer; for
        // one on the cluster the node measures it, and poll() applies the
        // answer when it arrives -- nothing to press again. False (and
        // contrastError says why) when it cannot be done at all.
        enum class ContrastAction { Auto,
                                    Reset };
        bool requestContrast(int index, ContrastAction action);
        // The request of step `index` the node is answering, if any, and for
        // how long it has been asked.
        struct ContrastRequest {
            ContrastAction action = ContrastAction::Auto;
            double seconds = 0.0;
        };
        std::optional<ContrastRequest> contrastRequest(int index) const;
        // Why the last request of step `index` failed ("" when it did not);
        // cleared by the next one.
        std::string contrastError(int index) const;
        // The input of Contrast step `index` is on the cluster and the node
        // has not answered for it yet (its window, its histograms).
        bool contrastMeasuring(int index) const;
        // Steps that need the Python worker (Operation::needsWorker) get a
        // local worker from this launcher when the backend is not HPC; the
        // GUI installs one that spawns app/python/sirius_worker. A run
        // job calls it on its own thread; loadPlugins calls it here.
        using WorkerLauncher = LocalWorkerLauncher;
        void setLocalWorkerLauncher(WorkerLauncher launcher) { launcher_ = std::move(launcher); }
        // The Hugging Face token a run hands the steps that download models
        // (StepContext::hubToken); the GUI reads it from the secret
        // store when a run is created.
        void setHubTokenProvider(std::function<std::string()> provider) { hubToken_ = std::move(provider); }
        // Starts the local worker (through the launcher, synchronously: this
        // blocks until the worker answers), registers the user operations it
        // finds (app/python/sirius_worker/plugins.py) and logs the outcome;
        // returns the number registered. `reload` re-imports the files.
        // Refused (0, and pluginError() says so) while a run is active: the
        // registry is in use. The plugins loaded before stay registered.
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
        const RunRefusal& lastRunRefusal() const noexcept { return lastRunRefusal_; }
        // Why the last loadPlugins() found none: its message ("" after one
        // that reached the worker, never after a refused one), and the
        // worker start failure behind it.
        const std::string& pluginError() const noexcept { return pluginError_; }   // "" after a successful loadPlugins
        const std::optional<WorkerStartError>& pluginWorkerFailure() const noexcept { return pluginWorkerFailure_; }
        // The host's next step when a worker is unavailable ("Preferences ...
        // sets the interpreter" in the window), appended to the generic
        // worker errors of runs, plugin loads and the missing launcher. A
        // WorkerStartError brings a hint of its own, which wins. The core
        // names no window or command of its own: "" (the default) adds
        // nothing.
        void setWorkerHint(std::string hint) { workerHint_ = std::move(hint); }   // appended to generic worker errors; "" = none

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
        // The planes a "Paint in 3D" stroke reaches above and below the one
        // painted, for a brush of `brushPx`: what the panel says it paints.
        static int paintZRadius(int brushPx) noexcept;
        void fillLabel(Index z, Index y, Index x);
        // On tracked labels (LabelVolume::tracked) a merge or a delete
        // applies to every time point: the id is the object's whole life.
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
        // createRun's null: logs `line` and keeps it as lastRunRefusal().
        std::shared_ptr<RunJob> refuseRun(RunRefusal::Kind kind, int step, const std::string& line);
        void installDataset(std::shared_ptr<ArraySource> source, DatasetMeta meta, std::string note);
        // Open `path` with `options`; the Load step's parameters become
        // `loadParams` with the options written into them.
        void openDatasetAs(const std::string& path, const OpenOptions& options, ParamSet loadParams);
        void installOpened(OpenResult opened, const std::string& path, const OpenOptions& options, ParamSet loadParams);
        // Seeds the executor with loadOutput_ when the Load step's parameters
        // are still the ones it was opened with (otherwise the step re-runs).
        void seedLoadOutput();
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
        bool followSelectedTrack();              // crosshair and z onto the selected track at t; false when it is not there
        // Undo / redo of a label edit: applies `diff` to the labels of step
        // `id` if they are still the volume the edit was made on, else a
        // logged no-op (the step was re-run or removed since).
        void applyLabelDiff(StepId id, const std::weak_ptr<LabelVolume>& target, const LabelDiff& diff, bool forward);
        void pushLabelCommand(const std::string& label, const std::string& mergeKey, StepId id,
                              const std::shared_ptr<LabelVolume>& labels, std::shared_ptr<LabelDiff> diff);
        void recordLabelDiff(const std::string& label, StepId id, const std::shared_ptr<LabelVolume>& labels,
                             LabelDiff diff);
        // One undo entry for an edit that touched several time points (a
        // merge or a delete on tracked labels).
        void recordLabelDiffs(const std::string& label, StepId id, const std::shared_ptr<LabelVolume>& labels,
                              std::vector<LabelDiff> diffs);

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
        HpcDevice hpcDevice_ = HpcDevice::Gpu;
        RemoteConfig remote_;
        std::shared_ptr<RunJob> activeRun_;
        std::vector<std::string> log_;
        std::optional<std::pair<std::string, ParamSet>> clipboard_;
        std::shared_ptr<const StepOutput> loadOutput_;   // the Load step's lazy output
        ParamSet loadOutputParams_;                      // the Load parameters it was opened with
        WorkerLauncher launcher_;
        std::function<std::string()> hubToken_;
        std::vector<PluginInfo> plugins_;
        std::vector<std::string> pluginDirs_;
        RunRefusal lastRunRefusal_;
        std::string pluginError_;
        std::optional<WorkerStartError> pluginWorkerFailure_;
        std::string workerHint_;
        // The engine on the node: the connection its outputs are drawn
        // through, and the answers it gave to previews and validations.
        struct EngineLink;
        std::shared_ptr<EngineLink> engine_;
        std::string engineSession_;
        std::set<std::string> uploadConsent_;            // "path\nstamp" the user agreed to upload
        std::map<std::string, std::string> nodePaths_;   // uploaded: this computer's path -> the node's
        // The pipeline as the node knows it: uploaded files under their node paths.
        nlohmann::json nodePipelineJson() const;
        // The engine's answer to `method` (step_preview, step_validate), or
        // nullopt while it is asked; `error` set when it failed.
        std::optional<nlohmann::json> askEngine(const std::string& method, const nlohmann::json& params, std::string* error,
                                                std::vector<rpc::Tensor>* tensors = nullptr, bool urgent = false) const;
        // The input of step `index` stays on the cluster, and the engine there answers for it.
        bool inputOnCluster(int index) const;
        // ... and that input is the step's own, computed (fresh): the node can preview the step on it.
        bool previewedOnNode(int index) const;
        // The node's measurement of the input of Contrast step `index` (its
        // preview with an automatic window and gamma 1: the histograms'
        // lo / hi are the automatic window, binLo / binHi the data range);
        // nullopt while asked; throws its error, or why it cannot be asked.
        std::optional<Diagnostics> nodeContrast(int index, const ParamSet& params, bool urgent = false) const;
        // Applies the answers of the pending contrast requests that arrived.
        void settleContrastRequests();
        struct PendingContrast {
            ContrastAction action = ContrastAction::Auto;
            std::chrono::steady_clock::time_point since;
            const ArraySource* source = nullptr;   // the dataset it was asked for
        };
        std::map<StepId, PendingContrast> contrastPending_;
        std::map<StepId, std::string> contrastErrors_;
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
