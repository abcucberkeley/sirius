#ifndef SIRIUS_APP_OPEN_DATASET_DIALOG_HPP
#define SIRIUS_APP_OPEN_DATASET_DIALOG_HPP

// File ▸ Open dataset…: path + Browse, the facts the file reports, and --
// when the metadata does not settle them -- how the pages map onto
// (c, t, z), the voxel size, the channel names and the raw SIM layout.
// A recent-files table (after the earlier design's Datasets browser) sits
// below for one-click reopening. "Folder…" picks a folder of TIFF files:
// one with a manifest (core/manifest.hpp) opens like a file; one without
// goes through the FolderDatasetDialog, which needs the workbench bridge.

#include <memory>

#include <QDialog>
#include <QString>
#include <QStringList>

#include "core/array_source.hpp"

namespace sirius::app {

    class WorkbenchBridge;

    class OpenDatasetDialog : public QDialog {
        Q_OBJECT
    public:
        explicit OpenDatasetDialog(QWidget* parent = nullptr, const QString& initialPath = {});
        // With the bridge, a folder without a manifest is described (and
        // opened) in a FolderDatasetDialog; the open dialog then closes as
        // rejected so the caller does not open the folder a second time.
        explicit OpenDatasetDialog(WorkbenchBridge& bridge, QWidget* parent = nullptr, const QString& initialPath = {});
        ~OpenDatasetDialog() override;

        void setBridge(WorkbenchBridge* bridge);

        QString path() const;
        OpenOptions options() const;

        // QSettings-backed recent list shared with File ▸ Open recent.
        static QStringList recentFiles();
        static void addRecentFile(const QString& path);

        void openAsOneStack();

    public slots:
        void accept() override;

    private:
        void probe();
        void updatePageCheck();
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_OPEN_DATASET_DIALOG_HPP
