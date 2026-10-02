#ifndef SIRIUS_IMGUI_DIALOGS_HPP
#define SIRIUS_IMGUI_DIALOGS_HPP

// The dialogs of the application, as factories. Each returns a Dialog for
// App::showDialog(); what the dialog collects arrives through the callback
// the factory takes, called
// on the GUI thread when the dialog is accepted (never when it is
// cancelled). One file per dialog implements its factory.

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "core/array_source.hpp"
#include "core/export.hpp"
#include "core/python_env.hpp"
#include "core/training_export.hpp"
#include "core/worker_error.hpp"
#include "core/workbench.hpp"
#include "imgui/app.hpp"

namespace sirius::app::gui {

    // File ▸ Open dataset…: path + Browse, the facts the file reports, and --
    // when the metadata does not settle them -- how the pages map onto
    // (c, t, z), the voxel size, the channel names and the raw SIM layout,
    // with the recent files below. A folder without a manifest is handed to
    // the folder dialog (which opens the dataset itself); `accepted` is then
    // not called.
    std::shared_ptr<Dialog> makeOpenDatasetDialog(App& app, const std::string& initialPath,
                                                  std::function<void(const std::string& path, const OpenOptions& options)> accepted);

    // "Open folder as dataset": a folder of TIFF stacks described by a
    // filename pattern with named groups, previewed live, then opened (the
    // dialog opens the dataset itself through App::openWith).
    std::shared_ptr<Dialog> makeFolderDatasetDialog(App& app, const std::string& folder);

    // File ▸ Export result…: format list, source step, range, pixel type,
    // the container's knobs, destination and sidecars.
    std::shared_ptr<Dialog> makeExportDialog(App& app,
                                             std::function<void(int stepIndex, const ExportOptions& options)> accepted);

    // File ▸ Export training data…
    std::shared_ptr<Dialog> makeTrainingExportDialog(
        App& app, std::function<void(int stepIndex, const TrainingExportOptions& options)> accepted);

    // Process ▸ Connect to cluster…: the profile, Connect, the checklist of
    // the steps (SSH login, checks, submit, queue, start, hello) and what
    // each found or why it failed; Disconnect. Not modal. (cluster_dialog.cpp)
    std::shared_ptr<Dialog> makeClusterDialog(App& app);
    // The cluster's files through the SSH session: path bar, Up, Home,
    // recent folders; `chosen` gets "cluster://<host>/<path>" of a file (or
    // of the folder shown, with `folders`). `extension` (".sif") lists only
    // the files that end so, besides the folders.
    std::shared_ptr<Dialog> makeClusterBrowser(App& app, const std::string& start, bool folders,
                                               std::function<void(const std::string& clusterPath)> chosen,
                                               const std::string& extension = {});

    // File ▸ Preferences…: default backend and CUDA device, the HPC worker
    // connection, the Python interpreter for the local worker and SIRIUS's
    // own Python environment, and the assistant provider. Values live in the
    // settings; the workbench and the assistant panel are updated on OK. The
    // environment's buttons act at once, without waiting for OK.
    std::shared_ptr<Dialog> makePreferencesDialog(App& app);
    // Applies the stored preferences to a fresh workbench at start-up.
    void applyStoredPreferences(Workbench& wb);

    // What "Set up Python for SIRIUS" is opened for: a worker that could not
    // start (`failure`), or a button of Preferences ▸ Compute, which names
    // the environment's state (Absent: set it up; Outdated: update it;
    // Incomplete or Broken: repair it; Ready: recreate it).
    struct PythonEnvRequest {
        std::optional<WorkerStartError> failure;
        pyenv::State state = pyenv::State::Absent;
        std::string problem;       // what the status said is wrong (Outdated, Broken)
        bool useUv = true;         // the setting "worker/useUv", or Preferences' unsaved checkbox
        // On the GUI thread after the dialog changed the environment or the
        // interpreter the worker runs: once a setup it started has ended
        // (however it ended, and whether or not the dialog is still open).
        std::function<void()> finished;
    };
    // Plans the setup (the interpreters found, uv) off the GUI thread, says
    // what would be downloaded and where, and runs the setup as a Bridge task
    // only when its button is pressed; then reloads the plugins.
    std::shared_ptr<Dialog> makePythonEnvDialog(App& app, PythonEnvRequest request);
    // The worker could not start in a way a setup would fix: shows the dialog
    // at most once per session, never when unattended nor when turned off
    // ("worker/offerEnvironment"), and otherwise logs one line.
    // $SIRIUS_PYTHON_OFFER=always shows it whatever else holds, =never never
    // does (screenshots and tests). GUI thread.
    void offerPythonEnvironment(App& app, const WorkerStartError& error);

    // The thread a dialog runs work on that starts a Python (planning a
    // setup, Preferences' Check): a moment usually, but as long as a probe's
    // timeout when an interpreter hangs. A dialog that closes meanwhile does
    // not wait for it: the thread is handed over and joined once it has
    // ended. The work must therefore stop reaching the dialog and the Bridge
    // once the dialog is gone, which the state it shares with the dialog
    // tells it. GUI thread.
    class DialogThread {
    public:
        DialogThread() = default;
        ~DialogThread();
        DialogThread(const DialogThread&) = delete;
        DialogThread& operator=(const DialogThread&) = delete;
        // Joins the last work first: start again only once it has answered.
        void start(std::function<void()> work);

    private:
        std::thread thread_;
        std::shared_ptr<std::atomic<bool>> ended_;
    };
    // Joins the threads that closed dialogs left behind; main() calls it once
    // the application is gone.
    void finishDialogThreads();

    // Models for the steps that need one: the local cache, Hugging Face,
    // model families, foundation bundles. `chosen` receives the model spec
    // ("/path/model.pt", "hf:repo/name:file.onnx", "cellpose:cyto3", ...).
    // `bundles`: open on the registry of foundation bundles.
    std::shared_ptr<Dialog> makeModelHubDialog(App& app, bool bundles, std::function<void(const std::string& model)> chosen);

    // Window ▸ User operations…: the plugin folders and files, their load
    // status and a code editor. Not modal; one instance, which the
    // application keeps and raises. `openFile` shows a plugin file in the
    // editor.
    class PluginManager : public Dialog {
    public:
        virtual void openFile(const std::string& path) = 0;
    };
    std::shared_ptr<PluginManager> makePluginManager(App& app);

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_DIALOGS_HPP
