#ifndef SIRIUS_IMGUI_DIALOGS_HPP
#define SIRIUS_IMGUI_DIALOGS_HPP

// The dialogs of the application, as factories. Each returns a Dialog for
// App::showDialog(); what the Qt application read off the dialog after
// exec() returned arrives through the callback the factory takes, called
// on the GUI thread when the dialog is accepted (never when it is
// cancelled). One file per dialog implements its factory.

#include <functional>
#include <memory>
#include <string>

#include "core/array_source.hpp"
#include "core/export.hpp"
#include "core/training_export.hpp"
#include "core/workbench.hpp"
#include "imgui/app.hpp"

namespace sirius::app::gui {

    // File ▸ Open dataset…: path + Browse, the facts the file reports, and --
    // when the metadata does not settle them -- how the pages map onto
    // (c, t, z), the voxel size, the channel names and the raw SIM layout,
    // with the recent files below. A folder without a manifest is handed to
    // the folder dialog (which opens the dataset itself); `accepted` is then
    // not called. (app/qt/dialogs/open_dataset_dialog.cpp)
    std::shared_ptr<Dialog> makeOpenDatasetDialog(App& app, const std::string& initialPath,
                                                  std::function<void(const std::string& path, const OpenOptions& options)> accepted);

    // "Open folder as dataset": a folder of TIFF stacks described by a
    // filename pattern with named groups, previewed live, then opened (the
    // dialog opens the dataset itself through App::openWith).
    // (app/qt/dialogs/folder_dataset_dialog.cpp)
    std::shared_ptr<Dialog> makeFolderDatasetDialog(App& app, const std::string& folder);

    // File ▸ Export result…: format list, source step, range, pixel type,
    // the container's knobs, destination and sidecars.
    // (app/qt/dialogs/export_dialog.cpp)
    std::shared_ptr<Dialog> makeExportDialog(App& app,
                                             std::function<void(int stepIndex, const ExportOptions& options)> accepted);

    // File ▸ Export training data… (app/qt/dialogs/training_export_dialog.cpp)
    std::shared_ptr<Dialog> makeTrainingExportDialog(
        App& app, std::function<void(int stepIndex, const TrainingExportOptions& options)> accepted);

    // File ▸ Preferences…: default backend and CUDA device, the HPC worker
    // connection, the Python interpreter for the local worker, and the
    // assistant provider. Values live in the settings; the workbench and the
    // assistant panel are updated on OK. (app/qt/dialogs/preferences_dialog.cpp)
    std::shared_ptr<Dialog> makePreferencesDialog(App& app);
    // Applies the stored preferences to a fresh workbench at start-up.
    void applyStoredPreferences(Workbench& wb);

    // Models for the steps that need one: the local cache, Hugging Face,
    // model families, foundation bundles. `chosen` receives the model spec
    // ("/path/model.pt", "hf:repo/name:file.onnx", "cellpose:cyto3", ...).
    // `bundles`: open on the registry of foundation bundles.
    // (app/qt/dialogs/model_hub_dialog.cpp)
    std::shared_ptr<Dialog> makeModelHubDialog(App& app, bool bundles, std::function<void(const std::string& model)> chosen);

    // Window ▸ User operations…: the plugin folders and files, their load
    // status and a code editor. Not modal; one instance, which the
    // application keeps and raises. `openFile` shows a plugin file in the
    // editor. (app/qt/dialogs/plugin_manager.cpp)
    class PluginManager : public Dialog {
    public:
        virtual void openFile(const std::string& path) = 0;
    };
    std::shared_ptr<PluginManager> makePluginManager(App& app);

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_DIALOGS_HPP
