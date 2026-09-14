#include "qt/dialogs/open_dataset_dialog.hpp"

#include <cmath>
#include <exception>

#include <QBoxLayout>
#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDialogButtonBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileInfo>
#include <QGridLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QTableWidget>
#include <QTimer>

#include "core/manifest.hpp"
#include "qt/dialogs/folder_dataset_dialog.hpp"
#include "qt/fast_file_dialog.hpp"
#include "qt/qt_strings.hpp"
#include "qt/theme.hpp"
#include "qt/widgets/controls.hpp"
#include "qt/workbench_bridge.hpp"

namespace sirius::app {

    using widgets::CaptionLabel;
    using widgets::Rule;

    namespace {
        constexpr int kMaxRecent = 12;

        // TIFF files directly in a folder (what a manifest would describe).
        int tiffCount(const QString& folder) { return static_cast<int>(tiffNamesInOrder(toStd(folder)).size()); }

        QString fileFilter() {
            QStringList exts;
            for (const std::string& e : readableExtensions()) exts << QStringLiteral("*") + fromStd(e);
            if (exts.isEmpty()) exts << QStringLiteral("*.tif") << QStringLiteral("*.tiff");
            return QStringLiteral("Datasets (%1);;SIRIUS dataset (*.toml);;All files (*)").arg(exts.join(QLatin1Char(' ')));
        }
    } // namespace

    struct OpenDatasetDialog::Impl {
        QLineEdit* path = nullptr;
        QLabel* facts = nullptr;
        QLabel* error = nullptr;
        QWidget* layoutBox = nullptr;
        QComboBox* order = nullptr;
        QSpinBox* c = nullptr;
        QSpinBox* t = nullptr;
        QSpinBox* z = nullptr;
        QLabel* pageCheck = nullptr;
        QDoubleSpinBox* vx = nullptr;
        QDoubleSpinBox* vy = nullptr;
        QDoubleSpinBox* vz = nullptr;
        QLineEdit* channels = nullptr;
        QCheckBox* sim = nullptr;
        QSpinBox* dirs = nullptr;
        QSpinBox* phases = nullptr;
        QCheckBox* fastSi = nullptr;
        QComboBox* readAs = nullptr;
        QTableWidget* recent = nullptr;
        QPushButton* open = nullptr;
        QPushButton* oneStack = nullptr;   // a folder of TIFFs, a frame per file
        QTimer probeTimer;
        DatasetMeta probed;
        bool probeOk = false;
        bool dimsFromMetadata = false;
        bool isFolder = false;          // a folder with a manifest: nothing to override
        Index pages = 0;
        WorkbenchBridge* bridge = nullptr;
    };

    OpenDatasetDialog::OpenDatasetDialog(WorkbenchBridge& bridge, QWidget* parent, const QString& initialPath)
        : OpenDatasetDialog(parent, initialPath) {
        impl_->bridge = &bridge;
    }

    void OpenDatasetDialog::setBridge(WorkbenchBridge* bridge) { impl_->bridge = bridge; }

    OpenDatasetDialog::OpenDatasetDialog(QWidget* parent, const QString& initialPath)
        : QDialog(parent), impl_(std::make_unique<Impl>()) {
        setWindowTitle(QStringLiteral("Open dataset"));
        setMinimumWidth(640);
        auto* root = new QVBoxLayout(this);
        root->setContentsMargins(22, 18, 22, 18);
        root->setSpacing(14);
        root->addWidget(widgets::heading(QStringLiteral("Open dataset"), theme::kH4Px, this));

        // path
        auto* pathRow = new QHBoxLayout();
        pathRow->setSpacing(6);
        impl_->path = new QLineEdit(initialPath, this);
        impl_->path->setPlaceholderText(QStringLiteral("/data/…/stack.tif, dataset.zarr or a folder of TIFF files"));
        auto* browse = new QPushButton(QStringLiteral("Browse"), this);
        widgets::setButtonClass(browse, "secondary small");
        auto* browseFolder = new QPushButton(QStringLiteral("Folder…"), this);
        widgets::setButtonClass(browseFolder, "secondary small");
        browseFolder->setToolTip(QStringLiteral("A folder of TIFF files, one per channel / time point / tile, described by a %1 manifest")
                                     .arg(QLatin1String(DatasetManifest::kFileName)));
        auto* browseDir = new QPushButton(QStringLiteral("Directory…"), this);
        widgets::setButtonClass(browseDir, "secondary small");
        browseDir->setToolTip(QStringLiteral("zarr / N5 stores are directories"));
        browseDir->setVisible(zarrSupported());
        pathRow->addWidget(impl_->path, 1);
        pathRow->addWidget(browse);
        pathRow->addWidget(browseFolder);
        pathRow->addWidget(browseDir);
        root->addLayout(pathRow);

        impl_->facts = widgets::label(QString(), 12, theme::kNeutral600, -1, this);
        impl_->facts->setWordWrap(true);
        root->addWidget(impl_->facts);
        impl_->error = widgets::label(QString(), 11, theme::kAccentText, -1, this);
        impl_->error->setWordWrap(true);
        impl_->error->hide();
        root->addWidget(impl_->error);

        // layout / metadata overrides
        impl_->layoutBox = new QWidget(this);
        auto* lb = new QVBoxLayout(impl_->layoutBox);
        lb->setContentsMargins(0, 0, 0, 0);
        lb->setSpacing(10);
        lb->addWidget(new Rule(2, Qt::Horizontal, impl_->layoutBox));
        lb->addWidget(new CaptionLabel(QStringLiteral("Page layout"), impl_->layoutBox));
        auto* grid = new QGridLayout();
        grid->setHorizontalSpacing(10);
        grid->setVerticalSpacing(8);
        auto fieldLabel = [this](const QString& t) { return widgets::label(t, 11, theme::kNeutral600, -1, this); };
        impl_->order = new QComboBox(impl_->layoutBox);
        for (const char* o : {"czt", "ctz", "zct", "ztc", "tcz", "tzc"})
            impl_->order->addItem(QStringLiteral("%1 (%2 fastest)").arg(QLatin1String(o)).arg(QLatin1Char(o[0])),
                                  QString::fromLatin1(o));
        impl_->order->setToolTip(QStringLiteral("Which axis changes fastest from page to page (ImageJ hyperstacks: c, then z, then t)"));
        impl_->c = new QSpinBox(impl_->layoutBox);
        impl_->c->setRange(1, 64);
        impl_->t = new QSpinBox(impl_->layoutBox);
        impl_->t->setRange(1, 1000000);
        impl_->z = new QSpinBox(impl_->layoutBox);
        impl_->z->setRange(0, 1000000);
        impl_->z->setSpecialValueText(QStringLiteral("auto"));
        impl_->z->setToolTip(QStringLiteral("0 = derived from the page count"));
        grid->addWidget(fieldLabel(QStringLiteral("Order")), 0, 0);
        grid->addWidget(impl_->order, 1, 0);
        grid->addWidget(fieldLabel(QStringLiteral("Channels (c)")), 0, 1);
        grid->addWidget(impl_->c, 1, 1);
        grid->addWidget(fieldLabel(QStringLiteral("Time points (t)")), 0, 2);
        grid->addWidget(impl_->t, 1, 2);
        grid->addWidget(fieldLabel(QStringLiteral("Planes (z)")), 0, 3);
        grid->addWidget(impl_->z, 1, 3);
        lb->addLayout(grid);
        impl_->pageCheck = widgets::label(QString(), 11, theme::kNeutral600, -1, impl_->layoutBox);
        lb->addWidget(impl_->pageCheck);

        lb->addWidget(new CaptionLabel(QStringLiteral("Metadata"), impl_->layoutBox));
        auto* mg = new QGridLayout();
        mg->setHorizontalSpacing(10);
        mg->setVerticalSpacing(8);
        auto voxel = [this] {
            auto* s = new QDoubleSpinBox(this);
            s->setRange(0.0001, 1000.0);
            s->setDecimals(4);
            s->setSuffix(QStringLiteral(" µm"));
            return s;
        };
        impl_->vx = voxel();
        impl_->vy = voxel();
        impl_->vz = voxel();
        // typical widefield sampling until a file says otherwise
        impl_->vx->setValue(0.1);
        impl_->vy->setValue(0.1);
        impl_->vz->setValue(0.2);
        mg->addWidget(fieldLabel(QStringLiteral("Voxel x")), 0, 0);
        mg->addWidget(impl_->vx, 1, 0);
        mg->addWidget(fieldLabel(QStringLiteral("Voxel y")), 0, 1);
        mg->addWidget(impl_->vy, 1, 1);
        mg->addWidget(fieldLabel(QStringLiteral("Voxel z")), 0, 2);
        mg->addWidget(impl_->vz, 1, 2);
        impl_->channels = new QLineEdit(this);
        impl_->channels->setPlaceholderText(QStringLiteral("488 α-actinin, 640 Mitochondria"));
        impl_->channels->setToolTip(QStringLiteral("Comma-separated channel names; a leading number is the emission wavelength"));
        mg->addWidget(fieldLabel(QStringLiteral("Channel names")), 2, 0, 1, 3);
        mg->addWidget(impl_->channels, 3, 0, 1, 3);
        lb->addLayout(mg);

        auto* simRow = new QHBoxLayout();
        simRow->setSpacing(10);
        impl_->sim = new QCheckBox(QStringLiteral("Raw SIM acquisition: z holds directions × phases × planes"), this);
        impl_->dirs = new QSpinBox(this);
        impl_->dirs->setRange(1, 9);
        impl_->dirs->setValue(3);
        impl_->dirs->setPrefix(QStringLiteral("dirs "));
        impl_->phases = new QSpinBox(this);
        impl_->phases->setRange(1, 15);
        impl_->phases->setValue(5);
        impl_->phases->setPrefix(QStringLiteral("phases "));
        impl_->fastSi = new QCheckBox(QStringLiteral("fast SI order"), this);
        simRow->addWidget(impl_->sim, 1);
        simRow->addWidget(impl_->dirs);
        simRow->addWidget(impl_->phases);
        simRow->addWidget(impl_->fastSi);
        lb->addLayout(simRow);
        root->addWidget(impl_->layoutBox);

        auto* readRow = new QHBoxLayout();
        readRow->addWidget(fieldLabel(QStringLiteral("Read as")));
        impl_->readAs = new QComboBox(this);
        impl_->readAs->addItem(QStringLiteral("Lazy (planes on demand)"));
        impl_->readAs->addItem(QStringLiteral("Full load to RAM"));
        impl_->readAs->setCurrentIndex(1);
        readRow->addWidget(impl_->readAs, 1);
        root->addLayout(readRow);

        // recent
        root->addWidget(new Rule(2, Qt::Horizontal, this));
        root->addWidget(new CaptionLabel(QStringLiteral("Recent datasets"), this));
        impl_->recent = new QTableWidget(0, 3, this);
        impl_->recent->setHorizontalHeaderLabels({QStringLiteral("Name"), QStringLiteral("Format"), QStringLiteral("Modified")});
        impl_->recent->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
        impl_->recent->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
        impl_->recent->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
        impl_->recent->verticalHeader()->hide();
        impl_->recent->setSelectionBehavior(QAbstractItemView::SelectRows);
        impl_->recent->setSelectionMode(QAbstractItemView::SingleSelection);
        impl_->recent->setEditTriggers(QAbstractItemView::NoEditTriggers);
        impl_->recent->setShowGrid(false);
        impl_->recent->setMaximumHeight(150);
        for (const QString& f : recentFiles()) {
            const int r = impl_->recent->rowCount();
            impl_->recent->insertRow(r);
            const QFileInfo fi(f);
            auto* name = new QTableWidgetItem(fi.fileName());
            name->setToolTip(f);
            name->setData(Qt::UserRole, f);
            impl_->recent->setItem(r, 0, name);
            const QString format = fi.isDir() && isFolderDataset(toStd(f)) ? QStringLiteral("folder")
                                   : fi.suffix().isEmpty()                 ? QStringLiteral("dir")
                                                                           : fi.suffix();
            impl_->recent->setItem(r, 1, new QTableWidgetItem(format));
            impl_->recent->setItem(r, 2, new QTableWidgetItem(fi.exists() ? fi.lastModified().toString(QStringLiteral("MMM d HH:mm")) : QStringLiteral("missing")));
        }
        root->addWidget(impl_->recent);

        // buttons
        auto* buttons = new QHBoxLayout();
        buttons->addStretch(1);
        auto* cancel = new QPushButton(QStringLiteral("Cancel"), this);
        widgets::setButtonClass(cancel, "ghost");
        impl_->oneStack = new QPushButton(QStringLiteral("Open as one stack"), this);
        widgets::setButtonClass(impl_->oneStack, "secondary");
        impl_->oneStack->hide();
        impl_->oneStack->setToolTip(QStringLiteral("Every TIFF in the folder as one time point of one stack, in name order"));
        impl_->open = new QPushButton(QStringLiteral("Open"), this);
        widgets::setButtonClass(impl_->open, "primary");
        impl_->open->setDefault(true);
        buttons->addWidget(cancel);
        buttons->addWidget(impl_->oneStack);
        buttons->addWidget(impl_->open);
        root->addLayout(buttons);

        connect(cancel, &QPushButton::clicked, this, &QDialog::reject);
        connect(impl_->open, &QPushButton::clicked, this, &QDialog::accept);
        connect(impl_->oneStack, &QPushButton::clicked, this, [this] { openAsOneStack(); });
        connect(browse, &QPushButton::clicked, this, [this] {
            const QString start = impl_->path->text().isEmpty() ? QString() : QFileInfo(impl_->path->text()).absolutePath();
            const QString f = getOpenFileNameFast(this, QStringLiteral("Open dataset"), start, fileFilter());
            if (!f.isEmpty()) impl_->path->setText(f);
        });
        connect(browseDir, &QPushButton::clicked, this, [this] {
            const QString d = getExistingDirectoryFast(this, QStringLiteral("Open zarr / N5 store"), QString());
            if (!d.isEmpty()) impl_->path->setText(d);
        });
        connect(browseFolder, &QPushButton::clicked, this, [this] {
            QString start = impl_->path->text();
            if (!start.isEmpty()) {
                const QFileInfo fi(start);
                // Parent of a TIFF folder: listing the folder itself stats
                // every stack on Vast/NFS and freezes the picker.
                start = fi.isDir() ? fi.absolutePath() : QFileInfo(fi.absolutePath()).absolutePath();
            }
            const QString d = getExistingDirectoryFast(this, QStringLiteral("Open folder of TIFF files"), start);
            if (d.isEmpty()) return;
            if (isFolderDataset(toStd(d)) || !impl_->bridge) {
                impl_->path->setText(d);   // the probe reports the manifest, or its absence
                return;
            }
            // no manifest yet: describe the files; the folder dialog opens the
            // dataset itself, so close without asking the caller to open it again
            FolderDatasetDialog describe(*impl_->bridge, d, this);
            const bool opened = describe.exec() == QDialog::Accepted;
            impl_->path->setText(d);
            if (opened) {
                addRecentFile(d);
                reject();
            }
        });
        connect(impl_->recent, &QTableWidget::itemSelectionChanged, this, [this] {
            const auto items = impl_->recent->selectedItems();
            if (!items.isEmpty()) impl_->path->setText(impl_->recent->item(items.first()->row(), 0)->data(Qt::UserRole).toString());
        });
        connect(impl_->recent, &QTableWidget::itemDoubleClicked, this, [this](QTableWidgetItem*) { accept(); });
        impl_->probeTimer.setSingleShot(true);
        impl_->probeTimer.setInterval(250);
        connect(&impl_->probeTimer, &QTimer::timeout, this, [this] { probe(); });
        connect(impl_->path, &QLineEdit::textChanged, this, [this] {
            // What was probed describes the previous path: nothing opens
            // until this one is probed, or a recent row double-clicked within
            // the probe's delay opened the new file with the old one's layout.
            impl_->probeOk = false;
            impl_->open->setEnabled(false);
            impl_->pageCheck->clear();
            impl_->probeTimer.start();
        });
        auto pageCheck = [this] { updatePageCheck(); };
        connect(impl_->c, qOverload<int>(&QSpinBox::valueChanged), this, pageCheck);
        connect(impl_->t, qOverload<int>(&QSpinBox::valueChanged), this, pageCheck);
        connect(impl_->z, qOverload<int>(&QSpinBox::valueChanged), this, pageCheck);
        connect(impl_->sim, &QCheckBox::toggled, this, [this](bool on) {
            impl_->dirs->setEnabled(on);
            impl_->phases->setEnabled(on);
            impl_->fastSi->setEnabled(on);
        });
        impl_->dirs->setEnabled(false);
        impl_->phases->setEnabled(false);
        impl_->fastSi->setEnabled(false);
        impl_->open->setEnabled(false);
        impl_->probeTimer.start();
    }

    // A folder of TIFFs with no manifest, read the plain way: every file one
    // time point of one stack, in name order. The manifest is written beside
    // the files, as the Folder dialog does, so the folder opens directly from
    // then on and the reading can be corrected by hand.
    void OpenDatasetDialog::openAsOneStack() {
        const QString folder = impl_->path->text().trimmed();
        if (folder.isEmpty() || !impl_->bridge) return;
        // The manifest is written before the dataset is opened, so the refusal
        // has to come first: otherwise a run in progress leaves the file behind
        // for a dataset that never opened.
        if (!impl_->bridge->wb().canEdit()) {
            QMessageBox::information(this, QStringLiteral("Open as one stack"),
                                     QStringLiteral("A run is in progress: cancel it (Esc) or wait before opening a dataset."));
            return;
        }
        const std::filesystem::path dir(toStd(folder));
        DatasetManifest manifest;
        try {
            manifest = manifestOfOneStack(dir);
        } catch (const std::exception& e) {
            QMessageBox::warning(this, QStringLiteral("Open as one stack"), QString::fromUtf8(e.what()));
            return;
        }
        if (manifest.files.empty()) {
            QMessageBox::information(this, QStringLiteral("Open as one stack"),
                                     QStringLiteral("No TIFF files in %1.").arg(folder));
            return;
        }
        const std::filesystem::path manifestPath = dir / DatasetManifest::kFileName;
        // A file's own pages become z; the files become t. That is right for a
        // time series and wrong for a stack saved a plane per file, and the two
        // look identical from the names, so say which reading is about to be
        // taken when the files are single planes.
        QString caution;
        try {
            const DatasetMeta first = probeDataset((dir / manifest.files.front().path).string());
            if (first.dims.z <= 1 && first.dims.t <= 1)
                caution = QStringLiteral(" Each file holds a single plane, so this gives %1 time points of one plane. If "
                                         "these are instead the planes of one stack, combine them into one TIFF first: a "
                                         "folder maps files to channels, tiles and time points, and takes z from the pages "
                                         "inside each file.")
                              .arg(manifest.files.size());
        } catch (const std::exception&) {
            // unreadable first file: the open below will say so properly
        }
        QMessageBox box(this);
        box.setIcon(QMessageBox::Question);
        box.setWindowTitle(QStringLiteral("Open as one stack"));
        box.setText(QStringLiteral("Read the %1 files as one stack, one time point each?").arg(manifest.files.size()));
        box.setInformativeText(QStringLiteral("In name order, %1 first and %2 last.%3 A %4 is written beside the files so the "
                                              "folder opens directly from then on; edit it, or use Folder…, for channels, "
                                              "tiles or another order.")
                                   .arg(fromStd(manifest.files.front().path), fromStd(manifest.files.back().path), caution,
                                        QLatin1String(DatasetManifest::kFileName)));
        QPushButton* go = box.addButton(QStringLiteral("Open"), QMessageBox::AcceptRole);
        box.addButton(QMessageBox::Cancel);
        box.setDefaultButton(go);
        box.exec();
        if (box.clickedButton() != go) return;
        try {
            manifest.save(manifestPath);
            OpenOptions options;
            options.tile = 0;
            options.readAll = true;
            if (!impl_->bridge->openDatasetAsync(toStd(folder), options)) {
                QMessageBox::warning(this, QStringLiteral("Open as one stack"),
                                     QStringLiteral("Another task is still running: cancel it or wait."));
                return;
            }
        } catch (const std::exception& e) {
            QMessageBox::warning(this, QStringLiteral("Open as one stack"), QString::fromUtf8(e.what()));
            return;
        }
        addRecentFile(folder);
        reject();   // the dataset is open; the caller must not open it again
    }

    OpenDatasetDialog::~OpenDatasetDialog() = default;

    // The facts, the page layout and the metadata fields for the path as it
    // is now. Runs 250 ms after the last edit of the path, and at once when
    // the dialog is accepted before that.
    void OpenDatasetDialog::probe() {
        const QString p = impl_->path->text().trimmed();
        impl_->probeOk = false;
        impl_->isFolder = false;
        impl_->error->hide();
        impl_->oneStack->hide();
        impl_->layoutBox->setVisible(true);
        if (p.isEmpty() || !QFileInfo::exists(p)) {
            impl_->facts->setText(p.isEmpty() ? QStringLiteral("Choose a TIFF / OME-TIFF file, a zarr / N5 store or a folder of TIFF files.")
                                              : QStringLiteral("No such file or directory."));
            impl_->open->setEnabled(false);
            return;
        }
        const bool folder = isManifestDataset(toStd(p));
        if (!folder && QFileInfo(p).isDir()) {
            // a folder of TIFFs without a manifest is not yet a dataset
            const int tiffs = tiffCount(p);
            if (tiffs > 0) {
                impl_->facts->setText(QStringLiteral("Folder of %1 TIFF file(s) without a %2 manifest. "
                                                     "Open as one stack reads them in name order, one time point each. "
                                                     "%3")
                                          .arg(tiffs)
                                          .arg(QLatin1String(DatasetManifest::kFileName),
                                               impl_->bridge ? QStringLiteral("For channels, tiles or another order, use Folder….")
                                                             : QStringLiteral("For anything else, File ▸ Open folder….")));
                impl_->open->setEnabled(false);
                impl_->oneStack->setVisible(impl_->bridge != nullptr);
                updatePageCheck();
                return;
            }
        }
        try {
            impl_->probed = probeDataset(toStd(p));
            impl_->probeOk = true;
            impl_->isFolder = folder;
            impl_->layoutBox->setVisible(!folder);   // the manifest settles layout, voxel size and channels
            const DatasetMeta& m = impl_->probed;
            impl_->pages = m.dims.planes();
            impl_->dimsFromMetadata = folder || m.format != "tiff";   // plain TIFF: the page mapping is the user's call
            QString facts = QStringLiteral("%1 · %2 · %3 · %4 · %5 channel(s)")
                                .arg(fromStd(m.format), fromStd(m.shapeString()), QString::fromLatin1(toString(m.sourceType)),
                                     widgets::bytesText(m.bytesOnDisk))
                                .arg(m.channels.size());
            if (m.hasTiles()) facts += QStringLiteral(" · %1 tiles").arg(m.tiles.size());
            impl_->facts->setText(facts);
            {
                QSignalBlocker b1(impl_->c), b2(impl_->t), b3(impl_->z);
                impl_->c->setValue(static_cast<int>(m.dims.c));
                impl_->t->setValue(static_cast<int>(m.dims.t));
                impl_->z->setValue(static_cast<int>(m.dims.z));
            }
            if (m.voxelUm[0] > 0.0) impl_->vx->setValue(m.voxelUm[0]);
            if (m.voxelUm[1] > 0.0) impl_->vy->setValue(m.voxelUm[1]);
            if (m.voxelUm[2] > 0.0) impl_->vz->setValue(m.voxelUm[2]);
            QStringList names;
            for (const ChannelInfo& ch : m.channels) {
                QString n = fromStd(ch.label);
                if (ch.wavelengthNm > 0) n.prepend(QString::number(static_cast<int>(std::lround(ch.wavelengthNm))) + QLatin1Char(' '));
                names << n;
            }
            impl_->channels->setText(names.join(QStringLiteral(", ")));
            impl_->sim->setChecked(m.sim.present);
            impl_->dirs->setValue(m.sim.ndirs);
            impl_->phases->setValue(m.sim.nphases);
            impl_->fastSi->setChecked(m.sim.fastSi);
            impl_->open->setEnabled(true);
        } catch (const std::exception& e) {
            impl_->facts->clear();
            impl_->error->setText(QString::fromUtf8(e.what()));
            impl_->error->show();
            impl_->open->setEnabled(false);
        }
        updatePageCheck();
    }

    void OpenDatasetDialog::accept() {
        // A path picked or typed a moment ago is probed now, so the options
        // that go with it are its own; one that does not open is not accepted
        // (the Open button is disabled for it too).
        if (impl_->probeTimer.isActive()) {
            impl_->probeTimer.stop();
            probe();
        }
        if (!impl_->probeOk || !impl_->open->isEnabled()) return;
        QDialog::accept();
    }

    void OpenDatasetDialog::updatePageCheck() {
        if (!impl_->probeOk) {
            impl_->pageCheck->clear();
            return;
        }
        if (impl_->isFolder) {
            impl_->pageCheck->clear();
            impl_->open->setEnabled(true);
            return;
        }
        const Index c = impl_->c->value(), t = impl_->t->value();
        Index z = impl_->z->value();
        const Index pages = impl_->pages;
        if (z == 0) z = (c * t > 0 && pages % (c * t) == 0) ? pages / (c * t) : 0;
        const bool ok = z > 0 && c * t * z == pages;
        impl_->pageCheck->setText(ok ? QStringLiteral("%1 pages = c%2 × t%3 × z%4").arg(pages).arg(c).arg(t).arg(z)
                                     : QStringLiteral("%1 pages do not divide into c%2 × t%3 × z%4").arg(pages).arg(c).arg(t).arg(impl_->z->value()));
        QPalette p = impl_->pageCheck->palette();
        p.setColor(QPalette::WindowText, ok ? theme::kNeutral600 : theme::kAccent);
        impl_->pageCheck->setPalette(p);
        impl_->open->setEnabled(ok);
    }

    QString OpenDatasetDialog::path() const { return impl_->path->text().trimmed(); }

    OpenOptions OpenDatasetDialog::options() const {
        OpenOptions o;
        if (impl_->isFolder) {
            // everything else comes from the manifest; start on the first tile
            o.readAll = impl_->readAs->currentIndex() == 1;
            o.tile = 0;
            return o;
        }
        PageOrder po;
        po.order = toStd(impl_->order->currentData().toString());
        po.c = impl_->c->value();
        po.t = impl_->t->value();
        po.z = impl_->z->value();
        const DatasetMeta& m = impl_->probed;
        if (!impl_->dimsFromMetadata || po.c != m.dims.c || po.t != m.dims.t || (po.z != 0 && po.z != m.dims.z))
            o.pageOrder = po;
        const std::array<double, 3> voxel{impl_->vx->value(), impl_->vy->value(), impl_->vz->value()};
        if (voxel != m.voxelUm) o.voxelUm = voxel;
        // channels: "488 name, 640 other"
        std::vector<ChannelInfo> channels;
        for (const QString& part : impl_->channels->text().split(QLatin1Char(','), Qt::SkipEmptyParts)) {
            ChannelInfo ch;
            QString s = part.trimmed();
            bool isNum = false;
            const int space = s.indexOf(QLatin1Char(' '));
            const QString head = space < 0 ? s : s.left(space);
            const double nm = head.toDouble(&isNum);
            if (isNum && nm > 100.0) {
                ch.wavelengthNm = nm;
                s = space < 0 ? QString() : s.mid(space + 1).trimmed();
            }
            ch.label = toStd(s);
            ch.color = colorForWavelength(ch.wavelengthNm);
            channels.push_back(ch);
        }
        bool sameChannels = channels.size() == m.channels.size();
        for (std::size_t i = 0; sameChannels && i < channels.size(); ++i)
            sameChannels = channels[i].label == m.channels[i].label && channels[i].wavelengthNm == m.channels[i].wavelengthNm;
        if (!channels.empty() && !sameChannels) o.channels = channels;
        SimLayout sim;
        sim.present = impl_->sim->isChecked();
        sim.ndirs = impl_->dirs->value();
        sim.nphases = impl_->phases->value();
        sim.fastSi = impl_->fastSi->isChecked();
        if (sim.present != m.sim.present || sim.ndirs != m.sim.ndirs || sim.nphases != m.sim.nphases || sim.fastSi != m.sim.fastSi)
            o.sim = sim;
        o.readAll = impl_->readAs->currentIndex() == 1;
        return o;
    }

    QStringList OpenDatasetDialog::recentFiles() {
        QSettings settings;
        return settings.value(QStringLiteral("recent/datasets")).toStringList();
    }

    void OpenDatasetDialog::addRecentFile(const QString& path) {
        QSettings settings;
        QStringList list = recentFiles();
        list.removeAll(path);
        list.prepend(path);
        while (list.size() > kMaxRecent) list.removeLast();
        settings.setValue(QStringLiteral("recent/datasets"), list);
    }

} // namespace sirius::app
