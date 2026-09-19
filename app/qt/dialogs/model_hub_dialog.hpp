#ifndef SIRIUS_APP_MODEL_HUB_DIALOG_HPP
#define SIRIUS_APP_MODEL_HUB_DIALOG_HPP

// Models for the steps that need one: search Hugging Face, download a
// TorchScript / ONNX file into the local model cache, pick a model family the
// worker's Python packages provide (Cellpose, micro-SAM), or choose a
// foundation bundle (.ltb) from a registry directory. The chosen model spec is
// what the step's "model" parameter accepts.

#include <QDialog>
#include <QString>

#include "qt/workbench_bridge.hpp"

namespace sirius::app {

    class ModelHubDialog : public QDialog {
        Q_OBJECT
    public:
        explicit ModelHubDialog(WorkbenchBridge& bridge, QWidget* parent = nullptr);
        ~ModelHubDialog() override;

        // Model spec to put into a step ("/path/model.pt", "hf:repo/name:file.onnx",
        // "cellpose:cyto3", "microsam:vit_b_lm", "/path/model.ltb"); empty when cancelled.
        QString chosenModel() const;

        // Open on the registry of foundation bundles, for a step whose model is
        // a .ltb: the other tabs offer files such a step cannot use.
        void showBundles();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_MODEL_HUB_DIALOG_HPP
