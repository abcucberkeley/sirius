#ifndef SIRIUS_APP_FOLDER_DATASET_DIALOG_HPP
#define SIRIUS_APP_FOLDER_DATASET_DIALOG_HPP

// "Open folder as dataset": a folder of TIFF stacks described by a filename
// pattern with named groups, previewed live, then opened. The pattern is
// remembered (last used, and per folder) so the next acquisition of the same
// layout does not need a new regex. An existing manifest can be loaded to
// recover its pattern; a sidecar is written only when the mapping is new or
// the TIFF folder cannot hold one (then a local cache file with files_folder).

#include <QDialog>
#include <QString>

#include "qt/workbench_bridge.hpp"

namespace sirius::app {

    class FolderDatasetDialog : public QDialog {
        Q_OBJECT
    public:
        FolderDatasetDialog(WorkbenchBridge& bridge, const QString& folder, QWidget* parent = nullptr);
        ~FolderDatasetDialog() override;

        // Accepting opens the dataset. A manifest is written when needed.
        QString folder() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_FOLDER_DATASET_DIALOG_HPP
