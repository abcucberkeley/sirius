#ifndef SIRIUS_APP_FOLDER_DATASET_DIALOG_HPP
#define SIRIUS_APP_FOLDER_DATASET_DIALOG_HPP

// "Open folder as dataset": a folder of TIFF stacks (one per channel / time
// point / tile) described by a filename pattern with named groups, previewed
// live, saved as a manifest (core/manifest.hpp) — in the folder or anywhere
// else — and opened. A folder that already has a manifest opens directly;
// this dialog then serves to inspect or redo the mapping.

#include <QDialog>
#include <QString>

#include "qt/workbench_bridge.hpp"

namespace sirius::app {

    class FolderDatasetDialog : public QDialog {
        Q_OBJECT
    public:
        FolderDatasetDialog(WorkbenchBridge& bridge, const QString& folder, QWidget* parent = nullptr);
        ~FolderDatasetDialog() override;

        // Accepting writes the manifest and opens the dataset through the workbench.
        QString folder() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_FOLDER_DATASET_DIALOG_HPP
