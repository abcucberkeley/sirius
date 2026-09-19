#ifndef SIRIUS_APP_FAST_FILE_DIALOG_HPP
#define SIRIUS_APP_FAST_FILE_DIALOG_HPP

// Qt's non-native QFileDialog builds a QFileSystemModel of the whole
// machine, MIME-sniffs files for icons and watches directories. On NFS /
// Vast a folder of stacks (or the "Computer" sidebar) freezes the GUI.
// These helpers keep the dialog on a small writable place, skip watches
// and content icons, and do not list TIFF stacks when picking a folder.

#include <QDir>
#include <QFileDialog>
#include <QFileIconProvider>
#include <QFileInfo>
#include <QFileSystemModel>
#include <QStandardPaths>
#include <QUrl>
#include <QWidget>

namespace sirius::app {

    inline QFileIconProvider* genericFileIconProvider() {
        class GenericFileIconProvider : public QFileIconProvider {
        public:
            QIcon icon(const QFileInfo& info) const override {
                return QFileIconProvider::icon(info.isDir() ? Folder : File);
            }
        };
        return new GenericFileIconProvider;
    }

    // Home, plus cluster scratch `$HOME` maps onto when it exists.
    inline QList<QUrl> fastSidebarUrls(const QString& extra = {}) {
        QList<QUrl> urls;
        auto add = [&](const QString& p) {
            if (p.isEmpty() || !QDir(p).exists()) return;
            const QUrl u = QUrl::fromLocalFile(QDir(p).absolutePath());
            if (!urls.contains(u)) urls.append(u);
        };
        add(QStandardPaths::writableLocation(QStandardPaths::HomeLocation));
        add(QDir::homePath());
        add(QStringLiteral("/clusterfs/nvme2/Users/") + QDir::home().dirName());
        add(extra);
        return urls;
    }

    inline QString fastWritableDirectory() {
        const QString nvme = QStringLiteral("/clusterfs/nvme2/Users/") + QDir::home().dirName();
        if (QDir(nvme).exists()) return nvme;
        const QString home = QStandardPaths::writableLocation(QStandardPaths::HomeLocation);
        return home.isEmpty() ? QDir::homePath() : home;
    }

    inline void configureFastFileDialog(QFileDialog& dialog, const QString& directory) {
        dialog.setOption(QFileDialog::DontUseNativeDialog);
        dialog.setOption(QFileDialog::DontUseCustomDirectoryIcons);
        dialog.setOption(QFileDialog::DontResolveSymlinks);
        dialog.setViewMode(QFileDialog::List);
        dialog.setIconProvider(genericFileIconProvider());
        dialog.setSidebarUrls(fastSidebarUrls(directory));

        if (auto* model = dialog.findChild<QFileSystemModel*>()) {
            model->setOption(QFileSystemModel::DontWatchForChanges);
            model->setOption(QFileSystemModel::DontUseCustomDirectoryIcons);
            model->setOption(QFileSystemModel::DontResolveSymlinks);
            model->setIconProvider(dialog.iconProvider());
            if (dialog.fileMode() == QFileDialog::Directory)
                model->setFilter(QDir::AllDirs | QDir::Drives | QDir::NoDot);
        }
        if (!directory.isEmpty()) dialog.setDirectory(directory);
    }

    inline QString getExistingDirectoryFast(QWidget* parent, const QString& caption, const QString& start) {
        QString dir = start;
        if (dir.isEmpty() || !QDir(dir).exists()) dir = fastWritableDirectory();
        QFileDialog dialog(parent, caption);
        dialog.setAcceptMode(QFileDialog::AcceptOpen);
        dialog.setFileMode(QFileDialog::Directory);
        dialog.setOption(QFileDialog::ShowDirsOnly);
        configureFastFileDialog(dialog, dir);
        if (dialog.exec() != QDialog::Accepted) return {};
        const QStringList files = dialog.selectedFiles();
        return files.isEmpty() ? QString() : files.front();
    }

    inline QString getOpenFileNameFast(QWidget* parent, const QString& caption, const QString& start,
                                       const QString& filter) {
        QString dir = start;
        const QFileInfo info(start);
        if (info.isFile()) dir = info.absolutePath();
        if (dir.isEmpty() || !QDir(dir).exists()) dir = fastWritableDirectory();
        QFileDialog dialog(parent, caption);
        dialog.setAcceptMode(QFileDialog::AcceptOpen);
        dialog.setFileMode(QFileDialog::ExistingFile);
        dialog.setNameFilter(filter);
        configureFastFileDialog(dialog, dir);
        if (info.isFile()) dialog.selectFile(info.fileName());
        if (dialog.exec() != QDialog::Accepted) return {};
        const QStringList files = dialog.selectedFiles();
        return files.isEmpty() ? QString() : files.front();
    }

    inline QString getSaveFileNameFast(QWidget* parent, const QString& caption, const QString& directory,
                                       const QString& fileName, const QString& filter) {
        QString dir = directory;
        if (dir.isEmpty() || !QDir(dir).exists()) dir = fastWritableDirectory();
        QFileDialog dialog(parent, caption);
        dialog.setAcceptMode(QFileDialog::AcceptSave);
        dialog.setFileMode(QFileDialog::AnyFile);
        dialog.setNameFilter(filter);
        dialog.setDefaultSuffix(QStringLiteral("toml"));
        configureFastFileDialog(dialog, dir);
        if (!fileName.isEmpty()) dialog.selectFile(fileName);
        if (dialog.exec() != QDialog::Accepted) return {};
        const QStringList files = dialog.selectedFiles();
        return files.isEmpty() ? QString() : files.front();
    }

} // namespace sirius::app

#endif // SIRIUS_APP_FAST_FILE_DIALOG_HPP
