#include "qt/dialogs/plugin_manager.hpp"

#include <algorithm>
#include <map>
#include <vector>

#include <QBoxLayout>
#include <QCloseEvent>
#include <QDesktopServices>
#include <QDir>
#include <QEvent>
#include <QFile>
#include <QFileInfo>
#include <QInputDialog>
#include <QKeyEvent>
#include <QKeySequence>
#include <QLabel>
#include <QMessageBox>
#include <QPainter>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QResizeEvent>
#include <QStyledItemDelegate>
#include <QTextStream>
#include <QTreeWidget>
#include <QUrl>

#include "core/ops/plugin.hpp"
#include "qt/qt_strings.hpp"
#include "qt/theme.hpp"
#include "qt/widgets/code_editor.hpp"
#include "qt/widgets/controls.hpp"

namespace sirius::app {

    using widgets::CaptionLabel;
    using widgets::CodeEditor;
    using widgets::Rule;

    namespace {

        // Item data roles on the tree.
        enum Role {
            kPathRole = Qt::UserRole + 1,   // file path (file rows) or directory path (directory rows)
            kKindRole,                      // operation kind (file rows)
            kStatusRole,                    // Status below
            kErrorRole,                     // first line of the load error
        };
        enum Status { kDirectory = 0,
                      kMissingDirectory,
                      kLoaded,
                      kFailed,
                      kSkipped };

        constexpr int kDirRowHeight = 26;
        constexpr int kFileRowHeight = 28;
        constexpr int kErrorRowHeight = 44;

        // Rows painted from theme:: tokens: directory captions, file rows with
        // name · kind · status glyph, and a second accent line for load errors.
        class PluginRowDelegate : public QStyledItemDelegate {
        public:
            using QStyledItemDelegate::QStyledItemDelegate;

            QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override {
                const int status = index.data(kStatusRole).toInt();
                int h = kFileRowHeight;
                if (status == kDirectory || status == kMissingDirectory) h = kDirRowHeight;
                else if (status == kFailed) h = kErrorRowHeight;
                return {option.rect.width(), h};
            }

            void paint(QPainter* p, const QStyleOptionViewItem& option, const QModelIndex& index) const override {
                const QRect r = option.rect;
                const int status = index.data(kStatusRole).toInt();
                const bool selected = option.state & QStyle::State_Selected;
                const bool hover = option.state & QStyle::State_MouseOver;
                p->save();
                if (selected) p->fillRect(r, theme::kSurface);
                else if (hover) p->fillRect(r, theme::kNeutral200);
                if (status == kDirectory || status == kMissingDirectory) {
                    p->fillRect(QRect(r.left(), r.top(), r.width(), 1), theme::kDivider);
                    QFont f = theme::caption();
                    p->setFont(f);
                    p->setPen(status == kMissingDirectory ? theme::kNeutral400 : theme::kNeutral600);
                    QString text = index.data(Qt::DisplayRole).toString();
                    if (status == kMissingDirectory) text += QStringLiteral("  (not created)");
                    const QRect tr = r.adjusted(10, 0, -10, 0);
                    p->drawText(tr, Qt::AlignVCenter | Qt::AlignLeft,
                                QFontMetrics(f).elidedText(captionCase(text), Qt::ElideMiddle, tr.width()));
                    p->restore();
                    return;
                }
                if (selected) p->fillRect(QRect(r.left(), r.top(), 3, r.height()), theme::kAccent);
                // status glyph at the right
                const QRect glyphRect(r.right() - 26, r.top(), 20, kFileRowHeight);
                if (status == kLoaded)
                    widgets::drawIcon(*p, QRectF(glyphRect).adjusted(4, 4, -4, -4), widgets::Icon::Check, theme::kNeutral600);
                else if (status == kFailed)
                    widgets::drawIcon(*p, QRectF(glyphRect).adjusted(4, 4, -4, -4), widgets::Icon::Close, theme::kAccent);
                // name, then the kind in caption style
                const QFont nameFont = theme::heading(12);
                const QFont kindFont = theme::caption();
                const QString name = index.data(Qt::DisplayRole).toString();
                const QString kind = captionCase(index.data(kKindRole).toString());
                const int left = r.left() + 14;
                const int right = glyphRect.left() - 6;
                const QRect line1(left, r.top(), right - left, kFileRowHeight);
                const int kindW = kind.isEmpty() ? 0 : QFontMetrics(kindFont).horizontalAdvance(kind) + 8;
                p->setFont(nameFont);
                p->setPen(theme::kText);
                const QString shownName = QFontMetrics(nameFont).elidedText(name, Qt::ElideRight, std::max(20, line1.width() - kindW));
                p->drawText(line1, Qt::AlignVCenter | Qt::AlignLeft, shownName);
                if (!kind.isEmpty()) {
                    const int nameW = QFontMetrics(nameFont).horizontalAdvance(shownName);
                    p->setFont(kindFont);
                    p->setPen(theme::kNeutral600);
                    p->drawText(QRect(left + nameW + 8, r.top() + 1, std::max(0, right - left - nameW - 8), kFileRowHeight),
                                Qt::AlignVCenter | Qt::AlignLeft, kind);
                }
                if (status == kFailed) {
                    const QFont errFont = theme::font(theme::kSmallPx);
                    p->setFont(errFont);
                    p->setPen(theme::kAccent);
                    const QRect line2(left, r.top() + kFileRowHeight - 6, r.right() - 10 - left, r.bottom() - (r.top() + kFileRowHeight - 6));
                    p->drawText(line2, Qt::AlignTop | Qt::AlignLeft,
                                QFontMetrics(errFont).elidedText(index.data(kErrorRole).toString(), Qt::ElideRight, line2.width()));
                }
                p->restore();
            }
        };

        // Lets the dialog veto Close (window button) and Escape while the
        // editor holds unsaved changes; QDialog's own handlers cannot be
        // overridden through the fixed header.
        class CloseGuard : public QObject {
        public:
            CloseGuard(QObject* parent, std::function<bool()> mayClose) : QObject(parent), mayClose_(std::move(mayClose)) {}

        protected:
            bool eventFilter(QObject* watched, QEvent* e) override {
                if (e->type() == QEvent::Close) {
                    if (!mayClose_()) {
                        e->ignore();
                        return true;
                    }
                } else if (e->type() == QEvent::KeyPress) {
                    auto* k = static_cast<QKeyEvent*>(e);
                    if (k->key() == Qt::Key_Escape && !mayClose_()) return true;
                }
                return QObject::eventFilter(watched, e);
            }

        private:
            std::function<bool()> mayClose_;
        };

        // A one-line label that elides its text (middle) to whatever width the
        // layout gives it, so the file name at the end stays readable.
        QString cleanDir(const QString& dir) { return QDir::cleanPath(QDir(dir).absolutePath()); }
        QString cleanFile(const QString& file) { return QDir::cleanPath(QFileInfo(file).absoluteFilePath()); }

        QString identifier(QString name) {
            name = name.trimmed().toLower();
            name.replace(QRegularExpression(QStringLiteral("[^a-z0-9_]+")), QStringLiteral("_"));
            name.remove(QRegularExpression(QStringLiteral("^_+|_+$")));
            if (!name.isEmpty() && name.at(0).isDigit()) name.prepend(QLatin1Char('_'));
            return name;
        }

        QString titleCase(const QString& kind) {
            QStringList words = kind.split(QLatin1Char('_'), Qt::SkipEmptyParts);
            for (QString& w : words) w[0] = w[0].toUpper();
            return words.join(QLatin1Char(' '));
        }

        QString pluginTemplate(const QString& kind, const QString& title) {
            QString t = QStringLiteral(R"PY("""%TITLE%: a SIRIUS user operation.

Edit STEP (the name and the parameters the step card shows) and run() (what
it does). Save reloads the plugin; a file that fails to load shows its error
here and in the add menu. plugins/README.md beside the app has the full
contract.
"""

import numpy as np

STEP = {
    "kind": "%KIND%",          # unique id, stored in saved pipelines
    "name": "%TITLE%",         # shown in the add menu and on the step card
    "group": "User",
    "params": [
        {"key": "offset", "label": "Offset", "type": "double", "default": 0.0, "min": -1e6, "max": 1e6,
         "unit": "counts", "help": "Value subtracted from every voxel"},
        {"key": "clip", "label": "Clip negatives", "type": "bool", "default": True,
         "help": "Set values below 0 to 0"},
    ],
    "separable_over_t": True,   # frames are independent: the app may split the run over t
}


def run(data, params, meta, ctx):
    """# %TITLE%

    Markdown shown as the operation's help. `data` is a float32 array of
    shape (c, t, z, y, x); `meta` carries dims and voxel size; return an
    array with the same layout plus a dict with a one-line "summary" and
    optional "facts" for the step card.
    """
    offset = float(params["offset"])
    c, t = data.shape[:2]
    out = np.empty_like(data, dtype=np.float32)
    n = c * t
    k = 0
    for ci in range(c):
        for ti in range(t):
            if ctx.cancelled():
                raise RuntimeError("cancelled")
            vol = data[ci, ti]
            # --- the operation: replace with your own ---
            out[ci, ti] = vol - offset
            k += 1
            ctx.progress(k / n, f"channel {ci} t {ti}")
    if params["clip"]:
        np.maximum(out, 0.0, out=out)
    return out, {"summary": f"%TITLE% offset {offset:g}", "facts": {"Offset": f"{offset:g} counts"}}
)PY");
            t.replace(QStringLiteral("%KIND%"), kind);
            t.replace(QStringLiteral("%TITLE%"), title);
            return t;
        }

    } // namespace

    struct PluginManagerDialog::Impl {
        PluginManagerDialog* dialog;
        WorkbenchBridge& bridge;

        QTreeWidget* tree = nullptr;
        QPushButton* newButton = nullptr;
        QPushButton* openFolderButton = nullptr;
        QPushButton* deleteButton = nullptr;
        QPushButton* reloadButton = nullptr;

        widgets::ElidedLabel* pathLabel = nullptr;
        CaptionLabel* modifiedMark = nullptr;
        QLabel* readOnlyNote = nullptr;
        CodeEditor* editor = nullptr;
        QWidget* banner = nullptr;
        QPlainTextEdit* bannerText = nullptr;
        QPushButton* saveButton = nullptr;
        QPushButton* revertButton = nullptr;

        QString currentPath;    // file shown in the editor, empty when none
        bool modified = false;
        bool loading = false;   // setPlainText in progress: ignore textChanged
        bool selecting = false; // tree being rebuilt / re-selected programmatically

        Impl(PluginManagerDialog* d, WorkbenchBridge& b) : dialog(d), bridge(b) {}

        Workbench& wb() { return bridge.wb(); }
        QString userDir() const { return cleanDir(fromStd(userPluginDirectory(false))); }

        // --- tree ----------------------------------------------------------------
        QTreeWidgetItem* itemForPath(const QString& path) const {
            for (int d = 0; d < tree->topLevelItemCount(); ++d) {
                QTreeWidgetItem* dir = tree->topLevelItem(d);
                if (dir->data(0, kPathRole).toString() == path) return dir;
                for (int f = 0; f < dir->childCount(); ++f)
                    if (dir->child(f)->data(0, kPathRole).toString() == path) return dir->child(f);
            }
            return nullptr;
        }

        void refreshTree() {
            QString keep = currentPath;
            if (QTreeWidgetItem* cur = tree->currentItem(); cur && cur->data(0, kStatusRole).toInt() <= kMissingDirectory)
                keep = cur->data(0, kPathRole).toString();

            selecting = true;
            tree->clear();

            // directories: the user folder first, then what the worker searched
            std::vector<QString> dirs;
            auto addDir = [&dirs](const QString& d) {
                if (d.isEmpty()) return;
                if (std::find(dirs.begin(), dirs.end(), d) == dirs.end()) dirs.push_back(d);
            };
            addDir(userDir());
            for (const std::string& d : wb().pluginDirs()) addDir(cleanDir(fromStd(d)));

            // plugin files the worker reported, keyed by path
            std::map<QString, Workbench::PluginInfo> known;
            for (const Workbench::PluginInfo& p : wb().plugins()) {
                const QString file = cleanFile(fromStd(p.file));
                known[file] = p;
                addDir(cleanDir(QFileInfo(file).absolutePath()));
            }

            for (const QString& dir : dirs) {
                const bool exists = QDir(dir).exists();
                auto* dirItem = new QTreeWidgetItem(tree);
                QString shown = dir;
                const QString home = QDir::homePath();
                if (shown.startsWith(home)) shown = QStringLiteral("~") + shown.mid(home.size());
                dirItem->setText(0, shown);
                dirItem->setToolTip(0, dir == userDir() ? QStringLiteral("Your plugins folder\n%1").arg(dir) : dir);
                dirItem->setData(0, kPathRole, dir);
                dirItem->setData(0, kStatusRole, exists ? kDirectory : kMissingDirectory);
                dirItem->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);

                // every .py on disk (helpers included) merged with the worker's view
                std::map<QString, QString> files;   // path -> file name
                if (exists)
                    for (const QFileInfo& fi : QDir(dir).entryInfoList({QStringLiteral("*.py")}, QDir::Files, QDir::Name))
                        files[cleanFile(fi.absoluteFilePath())] = fi.fileName();
                for (const auto& [path, info] : known)
                    if (cleanDir(QFileInfo(path).absolutePath()) == dir) files[path] = QFileInfo(path).fileName();

                std::vector<std::pair<QString, QString>> sorted(files.begin(), files.end());
                std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
                    return a.second.compare(b.second, Qt::CaseInsensitive) < 0;
                });
                for (const auto& [path, fileName] : sorted) {
                    auto* item = new QTreeWidgetItem(dirItem);
                    item->setData(0, kPathRole, path);
                    item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
                    auto it = known.find(path);
                    if (it == known.end()) {
                        item->setText(0, fileName);
                        item->setData(0, kStatusRole, kSkipped);
                        item->setData(0, kKindRole, fileName.startsWith(QLatin1Char('_')) ? QStringLiteral("helper") : QStringLiteral("not loaded"));
                        item->setToolTip(0, fileName.startsWith(QLatin1Char('_'))
                                                ? QStringLiteral("Files starting with '_' are not loaded as operations")
                                                : QStringLiteral("Not loaded yet: click Reload"));
                        continue;
                    }
                    const Workbench::PluginInfo& p = it->second;
                    const QString name = fromStd(p.name);
                    item->setText(0, name.isEmpty() ? fileName : name);
                    item->setData(0, kKindRole, fromStd(p.kind));
                    if (p.error.empty()) {
                        item->setData(0, kStatusRole, kLoaded);
                        item->setToolTip(0, QStringLiteral("%1\nLoaded as \"%2\"").arg(path, fromStd(p.kind)));
                    } else {
                        const QString err = fromStd(p.error);
                        item->setData(0, kStatusRole, kFailed);
                        item->setData(0, kErrorRole, err.section(QLatin1Char('\n'), 0, 0).trimmed());
                        item->setToolTip(0, err);
                    }
                }
                dirItem->setExpanded(true);
            }

            if (QTreeWidgetItem* again = keep.isEmpty() ? nullptr : itemForPath(keep)) tree->setCurrentItem(again);
            selecting = false;
            updateButtons();
        }

        QString selectedDirectory() const {
            QTreeWidgetItem* cur = tree->currentItem();
            if (!cur) return userDir();
            if (cur->data(0, kStatusRole).toInt() <= kMissingDirectory) return cur->data(0, kPathRole).toString();
            return cleanDir(QFileInfo(cur->data(0, kPathRole).toString()).absolutePath());
        }

        void updateButtons() {
            QTreeWidgetItem* cur = tree->currentItem();
            const bool fileRow = cur && cur->data(0, kStatusRole).toInt() >= kLoaded;
            deleteButton->setEnabled(fileRow);
            const bool editable = !currentPath.isEmpty() && !editor->isReadOnly();
            saveButton->setEnabled(editable && modified);
            revertButton->setEnabled(!currentPath.isEmpty());   // also re-reads a file changed outside
        }

        // --- editor --------------------------------------------------------------
        void setModified(bool on) {
            modified = on;
            modifiedMark->setVisible(on);
            updateButtons();
        }

        void updateBanner() {
            QString error;
            if (!currentPath.isEmpty())
                for (const Workbench::PluginInfo& p : wb().plugins())
                    if (cleanFile(fromStd(p.file)) == currentPath && !p.error.empty()) error = fromStd(p.error);
            bannerText->setPlainText(error);
            // one to six lines tall; the scrollbar takes the rest of a long traceback
            const int lines = std::clamp(bannerText->document()->blockCount(), 1, 6);
            bannerText->setFixedHeight(lines * bannerText->fontMetrics().lineSpacing() + 10);
            banner->setVisible(!error.isEmpty());
        }

        void showNothing() {
            loading = true;
            currentPath.clear();
            editor->setPlainText({});
            editor->setReadOnly(true);
            loading = false;
            pathLabel->setFullText(QStringLiteral("No file selected"));
            readOnlyNote->hide();
            setModified(false);
            updateBanner();
        }

        void loadFile(const QString& path) {
            QFile f(path);
            if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
                QMessageBox::warning(dialog, QStringLiteral("User operations"),
                                     QStringLiteral("Could not read %1:\n%2").arg(path, f.errorString()));
                return;
            }
            const QString text = QString::fromUtf8(f.readAll());
            f.close();
            loading = true;
            currentPath = path;
            editor->setPlainText(text);
            const QFileInfo fi(path);
            const bool writable = fi.isWritable();
            editor->setReadOnly(!writable);
            loading = false;
            editor->document()->setModified(false);
            editor->moveCursor(QTextCursor::Start);
            pathLabel->setFullText(path);
            readOnlyNote->setVisible(!writable);
            setModified(false);
            updateBanner();
        }

        bool save() {
            if (currentPath.isEmpty() || editor->isReadOnly()) return false;
            QFile f(currentPath);
            if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
                QMessageBox::warning(dialog, QStringLiteral("User operations"),
                                     QStringLiteral("Could not write %1:\n%2").arg(currentPath, f.errorString()));
                return false;
            }
            f.write(editor->toPlainText().toUtf8());
            f.close();
            setModified(false);
            wb().loadPlugins(true);
            refreshTree();
            updateBanner();
            return true;
        }

        void revert() {
            if (currentPath.isEmpty()) return;
            loadFile(currentPath);
        }

        // False when the user cancelled; saves or discards otherwise.
        bool confirmDiscard() {
            if (!modified || currentPath.isEmpty()) return true;
            QMessageBox box(dialog);
            box.setWindowTitle(QStringLiteral("User operations"));
            box.setText(QStringLiteral("%1 has unsaved changes.").arg(QFileInfo(currentPath).fileName()));
            box.setInformativeText(QStringLiteral("Save them before switching?"));
            QPushButton* saveBtn = box.addButton(QStringLiteral("Save"), QMessageBox::AcceptRole);
            QPushButton* discard = box.addButton(QStringLiteral("Discard"), QMessageBox::DestructiveRole);
            box.addButton(QStringLiteral("Cancel"), QMessageBox::RejectRole);
            box.setDefaultButton(saveBtn);
            box.exec();
            if (box.clickedButton() == saveBtn) return save();
            if (box.clickedButton() == discard) {
                setModified(false);
                return true;
            }
            return false;
        }

        void onCurrentItemChanged(QTreeWidgetItem* current, QTreeWidgetItem* previous) {
            updateButtons();
            if (selecting || !current) return;
            if (current->data(0, kStatusRole).toInt() <= kMissingDirectory) return;   // directory: keep the editor
            const QString path = current->data(0, kPathRole).toString();
            if (path == currentPath) return;
            if (!confirmDiscard()) {
                selecting = true;
                tree->setCurrentItem(previous);
                selecting = false;
                updateButtons();
                return;
            }
            loadFile(path);
        }

        // --- actions -------------------------------------------------------------
        void newPlugin() {
            if (!confirmDiscard()) return;
            bool ok = false;
            const QString raw = QInputDialog::getText(dialog, QStringLiteral("New user operation"),
                                                      QStringLiteral("Name (becomes <name>.py in your plugins folder):"),
                                                      QLineEdit::Normal, QStringLiteral("my_filter"), &ok);
            if (!ok) return;
            const QString kind = identifier(raw);
            if (kind.isEmpty()) {
                QMessageBox::warning(dialog, QStringLiteral("New user operation"),
                                     QStringLiteral("The name needs at least one letter or digit."));
                return;
            }
            const QString dir = cleanDir(fromStd(userPluginDirectory(true)));
            if (!QDir(dir).exists()) {
                QMessageBox::warning(dialog, QStringLiteral("New user operation"),
                                     QStringLiteral("Could not create the plugins folder %1.").arg(dir));
                return;
            }
            const QString path = cleanFile(dir + QLatin1Char('/') + kind + QStringLiteral(".py"));
            if (QFileInfo::exists(path)) {
                QMessageBox::information(dialog, QStringLiteral("New user operation"),
                                         QStringLiteral("%1 already exists; opening it instead.").arg(path));
                wb().loadPlugins(true);
                refreshTree();
                dialog->openFile(path);
                return;
            }
            QFile f(path);
            if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
                QMessageBox::warning(dialog, QStringLiteral("New user operation"),
                                     QStringLiteral("Could not write %1:\n%2").arg(path, f.errorString()));
                return;
            }
            f.write(pluginTemplate(kind, titleCase(kind)).toUtf8());
            f.close();
            wb().loadPlugins(true);
            refreshTree();
            dialog->openFile(path);
        }

        void openFolder() {
            QString dir = selectedDirectory();
            if (!QDir(dir).exists()) {
                if (dir == userDir()) dir = cleanDir(fromStd(userPluginDirectory(true)));
                if (!QDir(dir).exists()) {
                    QMessageBox::information(dialog, QStringLiteral("User operations"),
                                             QStringLiteral("%1 does not exist.").arg(dir));
                    return;
                }
                refreshTree();
            }
            QDesktopServices::openUrl(QUrl::fromLocalFile(dir));
        }

        void deleteCurrent() {
            QTreeWidgetItem* cur = tree->currentItem();
            if (!cur || cur->data(0, kStatusRole).toInt() < kLoaded) return;
            const QString path = cur->data(0, kPathRole).toString();
            const auto answer = QMessageBox::question(
                dialog, QStringLiteral("Delete user operation"),
                QStringLiteral("Delete %1?\n\nThis removes the file\n%2").arg(cur->text(0), path),
                QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel);
            if (answer != QMessageBox::Yes) return;
            if (!QFile::remove(path)) {
                QMessageBox::warning(dialog, QStringLiteral("Delete user operation"),
                                     QStringLiteral("Could not delete %1.").arg(path));
                return;
            }
            if (path == currentPath) showNothing();
            wb().loadPlugins(true);
            refreshTree();
        }

        void reload() {
            wb().loadPlugins(true);
            refreshTree();
            updateBanner();
        }
    };

    PluginManagerDialog::PluginManagerDialog(WorkbenchBridge& bridge, QWidget* parent)
        : QDialog(parent), impl_(std::make_unique<Impl>(this, bridge)) {
        setWindowTitle(QStringLiteral("User operations"));
        resize(900, 600);
        setMinimumSize(720, 460);
        setSizeGripEnabled(true);
        QPalette pal = palette();
        pal.setColor(QPalette::Window, theme::kBg);
        setPalette(pal);
        setAutoFillBackground(true);

        auto* root = new QVBoxLayout(this);
        root->setContentsMargins(22, 18, 22, 18);
        root->setSpacing(10);

        auto* head = new QHBoxLayout();
        head->setSpacing(12);
        head->addWidget(widgets::heading(QStringLiteral("User operations"), theme::kH4Px, this));
        auto* blurb = widgets::label(
            QStringLiteral("Python files in these folders become operations in the add menu. Save reloads them."),
            theme::kSmallPx, theme::kNeutral600, -1, this);
        head->addWidget(blurb, 1, Qt::AlignVCenter);
        root->addLayout(head);
        root->addWidget(new Rule(2, Qt::Horizontal, this));

        auto* body = new QHBoxLayout();
        body->setContentsMargins(0, 0, 0, 0);
        body->setSpacing(0);
        root->addLayout(body, 1);

        // --- left: folders and files
        auto* left = new QWidget(this);
        left->setFixedWidth(300);
        auto* ll = new QVBoxLayout(left);
        ll->setContentsMargins(0, 0, 14, 0);
        ll->setSpacing(8);
        impl_->tree = new QTreeWidget(left);
        impl_->tree->setColumnCount(1);
        impl_->tree->setHeaderHidden(true);
        impl_->tree->setRootIsDecorated(false);
        impl_->tree->setIndentation(0);
        impl_->tree->setItemsExpandable(false);
        impl_->tree->setExpandsOnDoubleClick(false);
        impl_->tree->setSelectionMode(QAbstractItemView::SingleSelection);
        impl_->tree->setMouseTracking(true);
        impl_->tree->setUniformRowHeights(false);
        impl_->tree->setItemDelegate(new PluginRowDelegate(impl_->tree));
        impl_->tree->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        impl_->tree->setFocusPolicy(Qt::StrongFocus);
        ll->addWidget(impl_->tree, 1);

        auto* lb = new QHBoxLayout();
        lb->setSpacing(6);
        auto smallButton = [left](const QString& text, const QString& tip) {
            auto* b = new QPushButton(text, left);
            widgets::setButtonClass(b, "small");
            b->setToolTip(tip);
            b->setCursor(Qt::PointingHandCursor);
            return b;
        };
        impl_->newButton = smallButton(QStringLiteral("New"), QStringLiteral("Create a plugin from a template in your plugins folder"));
        impl_->openFolderButton = smallButton(QStringLiteral("Open folder"), QStringLiteral("Show the selected folder in the file manager"));
        impl_->deleteButton = smallButton(QStringLiteral("Delete"), QStringLiteral("Delete the selected plugin file"));
        impl_->reloadButton = smallButton(QStringLiteral("Reload"), QStringLiteral("Re-import every plugin"));
        lb->addWidget(impl_->newButton);
        lb->addWidget(impl_->openFolderButton);
        lb->addWidget(impl_->deleteButton);
        lb->addStretch(1);
        lb->addWidget(impl_->reloadButton);
        ll->addLayout(lb);
        body->addWidget(left);
        body->addWidget(new Rule(2, Qt::Vertical, this));

        // --- right: editor
        auto* right = new QWidget(this);
        auto* rl = new QVBoxLayout(right);
        rl->setContentsMargins(14, 0, 0, 0);
        rl->setSpacing(8);

        auto* header = new QHBoxLayout();
        header->setSpacing(10);
        impl_->pathLabel = new widgets::ElidedLabel(right, Qt::ElideMiddle);
        widgets::setWidgetClass(impl_->pathLabel, "mono");
        impl_->pathLabel->setFullText(QStringLiteral("No file selected"));
        header->addWidget(impl_->pathLabel, 1);
        impl_->modifiedMark = new CaptionLabel(QStringLiteral("modified"), right);
        impl_->modifiedMark->setColor(theme::kAccentText);
        impl_->modifiedMark->hide();
        header->addWidget(impl_->modifiedMark, 0, Qt::AlignRight);
        rl->addLayout(header);

        impl_->readOnlyNote = widgets::label(
            QStringLiteral("Read-only: this file is not writable. Copy it into your plugins folder to change it."),
            theme::kSmallPx, theme::kNeutral600, -1, right);
        impl_->readOnlyNote->setWordWrap(true);
        impl_->readOnlyNote->hide();
        rl->addWidget(impl_->readOnlyNote);

        auto* editorFrame = new QFrame(right);
        editorFrame->setFrameShape(QFrame::NoFrame);
        auto* efl = new QVBoxLayout(editorFrame);
        efl->setContentsMargins(0, 0, 0, 0);
        efl->setSpacing(0);
        efl->addWidget(new Rule(2, Qt::Horizontal, editorFrame));
        impl_->editor = new CodeEditor(editorFrame);
        impl_->editor->setPlaceholderText(QStringLiteral("Select a file on the left, or click New."));
        impl_->editor->setReadOnly(true);
        efl->addWidget(impl_->editor, 1);
        efl->addWidget(new Rule(2, Qt::Horizontal, editorFrame));
        rl->addWidget(editorFrame, 1);

        impl_->banner = new QWidget(right);
        impl_->banner->setAutoFillBackground(true);
        {
            QPalette p = impl_->banner->palette();
            p.setColor(QPalette::Window, theme::kSurface);
            impl_->banner->setPalette(p);
        }
        auto* bl = new QHBoxLayout(impl_->banner);
        bl->setContentsMargins(0, 0, 10, 0);
        bl->setSpacing(10);
        auto* edge = new Rule(3, Qt::Vertical, impl_->banner);
        edge->setColor(theme::kAccent);
        bl->addWidget(edge);
        auto* bv = new QVBoxLayout();
        bv->setContentsMargins(0, 8, 0, 8);
        bv->setSpacing(4);
        auto* bcap = new CaptionLabel(QStringLiteral("Did not load"), impl_->banner);
        bcap->setColor(theme::kAccentText);
        bv->addWidget(bcap);
        impl_->bannerText = new QPlainTextEdit(impl_->banner);
        impl_->bannerText->setReadOnly(true);
        impl_->bannerText->setFrameShape(QFrame::NoFrame);
        impl_->bannerText->setLineWrapMode(QPlainTextEdit::WidgetWidth);
        impl_->bannerText->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        impl_->bannerText->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        widgets::setWidgetClass(impl_->bannerText, "banner");
        impl_->bannerText->document()->setDocumentMargin(0);
        bv->addWidget(impl_->bannerText);
        bl->addLayout(bv, 1);
        impl_->banner->hide();
        rl->addWidget(impl_->banner);

        auto* rb = new QHBoxLayout();
        rb->setSpacing(8);
        impl_->saveButton = new QPushButton(QStringLiteral("Save"), right);
        widgets::setButtonClass(impl_->saveButton, "primary");
        impl_->saveButton->setShortcut(QKeySequence::Save);
        impl_->saveButton->setToolTip(QStringLiteral("Write the file and reload plugins (Ctrl+S)"));
        impl_->revertButton = new QPushButton(QStringLiteral("Revert"), right);
        widgets::setButtonClass(impl_->revertButton, "secondary");
        impl_->revertButton->setToolTip(QStringLiteral("Discard the changes and reload the file from disk"));
        auto* hint = widgets::label(QStringLiteral("Tab indents · Ctrl+/ comments · Ctrl+S saves"), theme::kSmallPx,
                                    theme::kNeutral500, -1, right);
        auto* close = new QPushButton(QStringLiteral("Close"), right);
        widgets::setButtonClass(close, "ghost");
        rb->addWidget(impl_->saveButton);
        rb->addWidget(impl_->revertButton);
        rb->addSpacing(8);
        rb->addWidget(hint);
        rb->addStretch(1);
        rb->addWidget(close);
        rl->addLayout(rb);
        body->addWidget(right, 1);

        // --- wiring
        Impl* impl = impl_.get();
        connect(impl->tree, &QTreeWidget::currentItemChanged, this,
                [impl](QTreeWidgetItem* cur, QTreeWidgetItem* prev) { impl->onCurrentItemChanged(cur, prev); });
        connect(impl->editor, &QPlainTextEdit::textChanged, this, [impl] {
            if (!impl->loading && !impl->currentPath.isEmpty() && !impl->modified) impl->setModified(true);
        });
        connect(impl->newButton, &QPushButton::clicked, this, [impl] { impl->newPlugin(); });
        connect(impl->openFolderButton, &QPushButton::clicked, this, [impl] { impl->openFolder(); });
        connect(impl->deleteButton, &QPushButton::clicked, this, [impl] { impl->deleteCurrent(); });
        connect(impl->reloadButton, &QPushButton::clicked, this, [impl] { impl->reload(); });
        connect(impl->saveButton, &QPushButton::clicked, this, [impl] { impl->save(); });
        connect(impl->revertButton, &QPushButton::clicked, this, [impl] { impl->revert(); });
        connect(close, &QPushButton::clicked, this, [this, impl] {
            if (impl->confirmDiscard()) reject();
        });
        connect(&bridge, &WorkbenchBridge::operationsChanged, this, [impl] {
            impl->refreshTree();
            impl->updateBanner();
        });
        installEventFilter(new CloseGuard(this, [impl] { return impl->confirmDiscard(); }));

        impl->refreshTree();
        impl->updateButtons();
    }

    PluginManagerDialog::~PluginManagerDialog() = default;

    void PluginManagerDialog::openFile(const QString& path) {
        Impl* impl = impl_.get();
        const QString clean = cleanFile(path);
        QTreeWidgetItem* item = impl->itemForPath(clean);
        if (!item) {
            impl->refreshTree();
            item = impl->itemForPath(clean);
        }
        if (item) {
            impl->tree->setCurrentItem(item);   // loads through onCurrentItemChanged (asks about unsaved edits)
            impl->tree->scrollToItem(item);
            return;
        }
        if (clean == impl->currentPath || !QFileInfo::exists(clean) || !impl->confirmDiscard()) return;
        impl->selecting = true;
        impl->tree->setCurrentItem(nullptr);
        impl->selecting = false;
        impl->loadFile(clean);
    }

} // namespace sirius::app
