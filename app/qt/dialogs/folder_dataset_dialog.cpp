#include "qt/dialogs/folder_dataset_dialog.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <QAction>
#include <QBoxLayout>
#include <QBrush>
#include <QCheckBox>
#include <QColorDialog>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileInfo>
#include <QGridLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMessageBox>
#include <QPainter>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QTableWidget>
#include <QTimer>

#include "core/array_source.hpp"
#include "core/manifest.hpp"
#include "qt/fast_file_dialog.hpp"
#include "qt/qt_strings.hpp"
#include "qt/theme.hpp"
#include "qt/widgets/controls.hpp"

namespace sirius::app {

    using widgets::CaptionLabel;
    using widgets::ClickRow;
    using widgets::Rule;
    using widgets::SegmentedControl;

    namespace {
        constexpr int kMaxPreviewRows = 32;

        // Preset patterns for the layouts acquisition software writes most
        // often. The combo keeps the chosen name so the pick is visible.
        struct Preset {
            const char* label;
            const char* pattern;
            FilenameRule::Positions positions;
            const char* example;   // tooltip; kept off the combo so the list stays clickable
        };

        const Preset kPresets[] = {
            // ABC AOLLS / LabVIEW SPIM (Aang_Foundation, Scan_Iter_*_CamA_*_000x_000y_000z_0000t.tif)
            {"AOLLS",
             R"(Scan_Iter_\d+_\d+_\d+_\d+_(?P<channel>Cam[A-Z])_ch\d+_CAM\d+_stack\d+_\d+nm_\d+msec_\d+msecAbs_(?P<x>\d+)x_(?P<y>\d+)y_(?P<z>\d+)z_(?P<t>\d+)t\.tiff?$)",
             FilenameRule::Positions::GridIndex, "Scan_Iter_…_CamA_ch0_…_000x_000y_000z_0000t.tif"},
            {"channel · t · x · y",
             R"(^.*_c(?P<channel>[^_]+)_t(?P<t>\d+)_x(?P<x>\d+)_y(?P<y>\d+)\.tiff?$)", FilenameRule::Positions::GridIndex,
             "stack_c488_t0_x1_y2.tif"},
            {"channel · t", R"(^.*_ch(?P<channel>\d+)_t(?P<t>\d+)\.tiff?$)", FilenameRule::Positions::None,
             "stack_ch0_t003.tif"},
            {"channel prefix", R"(^(?P<channel>[^_]+)_.*\.tiff?$)", FilenameRule::Positions::None, "488_cell.tif"},
            {"Micro-Manager positions", R"(^.*_MMStack_Pos(?P<tile>\d+)\.ome\.tiff?$)", FilenameRule::Positions::None,
             "run_MMStack_Pos3.ome.tif"},
            {"stage coordinates",
             R"(^.*_c(?P<channel>[^_]+)_X(?P<x>-?[\d.]+)_Y(?P<y>-?[\d.]+)\.tiff?$)", FilenameRule::Positions::Microns,
             "stack_c488_X1200.5_Y-30.0.tif"},
        };

        bool looksLikeAollsScanIter(const std::string& name) {
            return name.size() > 10 && name.compare(0, 10, "Scan_Iter_") == 0 &&
                   name.find("msecAbs_") != std::string::npos && name.find("Cam") != std::string::npos;
        }

        int indexOfPresetPattern(const QString& pat) {
            for (int i = 0; i < static_cast<int>(std::size(kPresets)); ++i)
                if (pat == QString::fromLatin1(kPresets[i].pattern)) return i + 1;
            return 0;
        }

        std::filesystem::path canonicalPath(const std::filesystem::path& p) {
            std::error_code ec;
            const std::filesystem::path c = std::filesystem::weakly_canonical(p, ec);
            return ec ? std::filesystem::absolute(p) : c;
        }

        // First of the group aliases that matched, else empty.
        QString group(const FilenameMatch& m, std::initializer_list<const char*> names) {
            for (const char* n : names) {
                const auto it = m.groups.find(n);
                if (it != m.groups.end()) return fromStd(it->second);
            }
            return {};
        }

        // "488" < "561" < "640" numerically, everything else by name.
        bool tokenLess(const std::string& a, const std::string& b) {
            bool na = false, nb = false;
            const double da = fromStd(a).toDouble(&na);
            const double db = fromStd(b).toDouble(&nb);
            if (na && nb) return da < db;
            if (na != nb) return na;
            return a < b;
        }

        QColor chipColor(const ChannelInfo& ch) { return QColor(fromStd(ch.hexColor())); }

        constexpr auto kLastManifestDirKey = "folderDataset/lastManifestDir";

        QString manifestBrowseDirectory(const QString& dest, const QString& tiffFolder) {
            const QString destDir = QFileInfo(dest).absolutePath();
            const QString tiffAbs = QDir(tiffFolder).absolutePath();
            if (!destDir.isEmpty() && QDir(destDir).absolutePath() != tiffAbs && QDir(destDir).exists())
                return destDir;
            const QString last = QSettings().value(QLatin1String(kLastManifestDirKey)).toString();
            if (!last.isEmpty() && QDir(last).absolutePath() != tiffAbs && QDir(last).exists()) return last;
            return fastWritableDirectory();
        }

        // The tiles as the pattern places them: grid cells or scaled stage
        // positions, with a one-line summary underneath.
        class TileMap : public QWidget {
        public:
            explicit TileMap(QWidget* parent) : QWidget(parent) { setFixedSize(188, 132); }

            void setTiles(std::vector<QPointF> pts, bool grid, const QString& note) {
                pts_ = std::move(pts);
                grid_ = grid;
                note_ = note;
                update();
            }

        protected:
            void paintEvent(QPaintEvent*) override {
                QPainter p(this);
                p.setRenderHint(QPainter::Antialiasing);
                p.setPen(QPen(theme::kDivider, 1));
                p.setBrush(theme::kSurface);
                p.drawRect(QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5));
                p.setFont(theme::font(11));
                p.setPen(theme::kNeutral600);
                p.drawText(QRect(0, height() - 20, width(), 18), Qt::AlignCenter, note_);
                if (pts_.empty()) return;
                double minx = pts_[0].x(), maxx = minx, miny = pts_[0].y(), maxy = miny;
                for (const QPointF& q : pts_) {
                    minx = std::min(minx, q.x());
                    maxx = std::max(maxx, q.x());
                    miny = std::min(miny, q.y());
                    maxy = std::max(maxy, q.y());
                }
                const QRectF area(10.0, 10.0, width() - 20.0, height() - 36.0);
                QColor fill = theme::kAccent;
                fill.setAlpha(40);
                if (grid_) {
                    const int nx = static_cast<int>(maxx - minx) + 1;
                    const int ny = static_cast<int>(maxy - miny) + 1;
                    const double cell = std::clamp(std::min(area.width() / nx, area.height() / ny), 4.0, 26.0);
                    const double ox = area.left() + (area.width() - cell * nx) / 2.0;
                    const double oy = area.top() + (area.height() - cell * ny) / 2.0;
                    p.setPen(QPen(theme::kAccent, 1));
                    p.setBrush(fill);
                    for (const QPointF& q : pts_) {
                        const QRectF r(ox + (q.x() - minx) * cell, oy + (q.y() - miny) * cell, cell, cell);
                        p.drawRect(r.adjusted(1.0, 1.0, -1.0, -1.0));
                    }
                } else {
                    const double sx = maxx > minx ? (area.width() - 8.0) / (maxx - minx) : 0.0;
                    const double sy = maxy > miny ? (area.height() - 8.0) / (maxy - miny) : 0.0;
                    double s = 0.0;
                    if (sx > 0.0 && sy > 0.0) s = std::min(sx, sy);
                    else s = std::max(sx, sy);
                    const double w = s * (maxx - minx), h = s * (maxy - miny);
                    const double ox = area.left() + (area.width() - w) / 2.0, oy = area.top() + (area.height() - h) / 2.0;
                    p.setPen(QPen(theme::kAccent, 1));
                    p.setBrush(theme::kAccent);
                    for (const QPointF& q : pts_)
                        p.drawRect(QRectF(ox + (q.x() - minx) * s - 3.0, oy + (q.y() - miny) * s - 3.0, 6.0, 6.0));
                }
            }

        private:
            std::vector<QPointF> pts_;
            bool grid_ = true;
            QString note_;
        };
    } // namespace

    struct FolderDatasetDialog::Impl {
        FolderDatasetDialog* q;
        WorkbenchBridge& bridge;
        QString folder;
        std::vector<std::string> names;              // the folder's TIFF files, sorted
        std::optional<DatasetManifest> existing;     // a manifest already in the folder

        QLineEdit* pattern = nullptr;
        QPushButton* presets = nullptr;
        QLineEdit* manifestPath = nullptr;
        QLabel* status = nullptr;
        QTableWidget* preview = nullptr;
        SegmentedControl* positions = nullptr;
        QDoubleSpinBox* overlap = nullptr;
        QDoubleSpinBox* vx = nullptr;
        QDoubleSpinBox* vy = nullptr;
        QDoubleSpinBox* vz = nullptr;
        QDoubleSpinBox* interval = nullptr;
        QLineEdit* acquisition = nullptr;
        QCheckBox* sim = nullptr;
        QSpinBox* dirs = nullptr;
        QSpinBox* phases = nullptr;
        QCheckBox* fastSi = nullptr;
        QTableWidget* channels = nullptr;
        QLabel* channelsNote = nullptr;
        TileMap* tileMap = nullptr;
        QPushButton* save = nullptr;
        QTimer previewTimer;

        std::vector<FilenameMatch> matches;
        bool patternOk = false;
        int matchedCount = 0;

        struct Chan {
            ChannelInfo info;
            bool customColor = false;   // chosen here or preloaded: not re-derived from the wavelength
            QWidget* chip = nullptr;
        };
        std::map<std::string, Chan> chans;   // by channel token
        std::vector<std::string> tokens;     // channel table rows, in order
        bool fillingChannels = false;

        Impl(FolderDatasetDialog* self, WorkbenchBridge& b) : q(self), bridge(b) {}

        FilenameRule::Positions positionsMode() const {
            switch (positions->currentIndex()) {
                case 1: return FilenameRule::Positions::GridIndex;
                case 2: return FilenameRule::Positions::Microns;
                default: return FilenameRule::Positions::None;
            }
        }

        FilenameRule rule() const;
        void runPreview();
        void refreshChannels();
        void refreshTileMap();
        void saveAndOpen();
    };

    FolderDatasetDialog::FolderDatasetDialog(WorkbenchBridge& bridge, const QString& folder, QWidget* parent)
        : QDialog(parent), impl_(std::make_unique<Impl>(this, bridge)) {
        impl_->folder = QDir(folder).absolutePath();
        setWindowTitle(QStringLiteral("Open folder as dataset"));
        setMinimumWidth(720);
        resize(720, 780);

        impl_->names = tiffNamesInOrder(std::filesystem::path(toStd(impl_->folder)));
        const std::filesystem::path manifestPath = std::filesystem::path(toStd(impl_->folder)) / DatasetManifest::kFileName;
        std::error_code ec;
        if (std::filesystem::exists(manifestPath, ec)) {
            try {
                impl_->existing = DatasetManifest::load(manifestPath);
            } catch (const std::exception&) {
                impl_->existing.reset();
            }
        }

        auto* root = new QVBoxLayout(this);
        root->setContentsMargins(22, 18, 22, 18);
        root->setSpacing(12);
        root->addWidget(widgets::heading(QStringLiteral("Open folder as dataset"), theme::kH4Px, this));
        auto fieldLabel = [this](const QString& t) { return widgets::label(t, 11, theme::kNeutral600, -1, this); };

        // folder + file count
        auto* folderRow = new QHBoxLayout();
        folderRow->setSpacing(8);
        auto* folderEdit = new QLineEdit(impl_->folder, this);
        folderEdit->setReadOnly(true);
        folderEdit->setToolTip(impl_->folder);
        folderRow->addWidget(folderEdit, 1);
        folderRow->addWidget(widgets::label(QStringLiteral("%1 TIFF file(s)").arg(impl_->names.size()), 12, theme::kNeutral600, -1, this));
        root->addLayout(folderRow);

        auto* manifestRow = new QHBoxLayout();
        manifestRow->setSpacing(8);
        impl_->manifestPath = new QLineEdit(this);
        impl_->manifestPath->setText(impl_->folder + QLatin1Char('/') + QLatin1String(DatasetManifest::kFileName));
        impl_->manifestPath->setToolTip(QStringLiteral("Where to write the manifest. Does not have to be inside the TIFF folder."));
        auto* browseManifest = new QPushButton(QStringLiteral("Browse"), this);
        widgets::setButtonClass(browseManifest, "secondary small");
        browseManifest->setToolTip(QStringLiteral(
            "Choose where to write the manifest. Opens outside the TIFF folder so the dialog does not list hundreds of stacks"));
        manifestRow->addWidget(fieldLabel(QStringLiteral("Manifest")), 0);
        manifestRow->addWidget(impl_->manifestPath, 1);
        manifestRow->addWidget(browseManifest);
        root->addLayout(manifestRow);

        // pattern
        root->addWidget(new Rule(2, Qt::Horizontal, this));
        root->addWidget(new CaptionLabel(QStringLiteral("Filename pattern"), this));
        auto* patternRow = new QHBoxLayout();
        patternRow->setSpacing(8);
        impl_->pattern = new QLineEdit(this);
        impl_->pattern->setFont(theme::mono(13));
        impl_->pattern->setPlaceholderText(QString::fromLatin1(kPresets[0].pattern));
        impl_->pattern->setToolTip(QStringLiteral("Regular expression matched against each file name; named groups pick out the channel, time point and tile"));
        impl_->presets = new QPushButton(QStringLiteral("Presets"), this);
        widgets::setButtonClass(impl_->presets, "secondary small");
        impl_->presets->setToolTip(QStringLiteral("Common layouts; pick one and adjust"));
        auto* presetMenu = new QMenu(impl_->presets);
        presetMenu->setToolTipsVisible(true);
        for (int i = 0; i < static_cast<int>(std::size(kPresets)); ++i) {
            const Preset& p = kPresets[i];
            auto* action = presetMenu->addAction(QString::fromUtf8(p.label));
            action->setToolTip(QString::fromUtf8(p.example));
            action->setData(i);
        }
        impl_->presets->setMenu(presetMenu);
        patternRow->addWidget(impl_->pattern, 1);
        patternRow->addWidget(impl_->presets);
        root->addLayout(patternRow);
        auto* hint = widgets::label(
            QStringLiteral("Named groups, written (?P<name>…): channel · t · tile · x · y · z. x / y / z are grid indices or stage "
                           "coordinates (see Positions); files the pattern does not match are left out."),
            11, theme::kNeutral600, -1, this);
        hint->setWordWrap(true);
        root->addWidget(hint);
        impl_->status = widgets::label(QString(), 12, theme::kNeutral600, -1, this);
        impl_->status->setWordWrap(true);
        root->addWidget(impl_->status);

        // live preview
        impl_->preview = new QTableWidget(0, 7, this);
        impl_->preview->setHorizontalHeaderLabels({QStringLiteral("File"), QStringLiteral("Channel"), QStringLiteral("t"),
                                                   QStringLiteral("Tile"), QStringLiteral("x"), QStringLiteral("y"), QStringLiteral("z")});
        impl_->preview->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
        for (int c = 1; c < 7; ++c) impl_->preview->horizontalHeader()->setSectionResizeMode(c, QHeaderView::ResizeToContents);
        impl_->preview->verticalHeader()->hide();
        impl_->preview->setSelectionMode(QAbstractItemView::NoSelection);
        impl_->preview->setEditTriggers(QAbstractItemView::NoEditTriggers);
        impl_->preview->setShowGrid(false);
        impl_->preview->setMinimumHeight(140);
        root->addWidget(impl_->preview, 1);

        // positions
        root->addWidget(new Rule(2, Qt::Horizontal, this));
        auto* posRow = new QHBoxLayout();
        posRow->setSpacing(12);
        posRow->addWidget(new CaptionLabel(QStringLiteral("Positions"), this));
        impl_->positions = new SegmentedControl({QStringLiteral("None"), QStringLiteral("Grid indices"), QStringLiteral("Micrometres")}, this);
        impl_->positions->setOptionToolTip(0, QStringLiteral("One tile, or tiles without known positions"));
        impl_->positions->setOptionToolTip(1, QStringLiteral("x / y / z are column, row and layer indices of a grid"));
        impl_->positions->setOptionToolTip(2, QStringLiteral("x / y / z are stage coordinates in micrometres"));
        impl_->positions->setCurrentIndex(1);
        posRow->addWidget(impl_->positions);
        posRow->addWidget(fieldLabel(QStringLiteral("Overlap")));
        impl_->overlap = new QDoubleSpinBox(this);
        impl_->overlap->setRange(0.0, 90.0);
        impl_->overlap->setDecimals(1);
        impl_->overlap->setSingleStep(1.0);
        impl_->overlap->setSuffix(QStringLiteral(" %"));
        impl_->overlap->setValue(10.0);
        impl_->overlap->setToolTip(QStringLiteral("Grid indices: neighbouring tiles overlap by this fraction of their size"));
        posRow->addWidget(impl_->overlap);
        posRow->addStretch(1);
        root->addLayout(posRow);

        // metadata
        root->addWidget(new CaptionLabel(QStringLiteral("Metadata"), this));
        auto* mg = new QGridLayout();
        mg->setHorizontalSpacing(10);
        mg->setVerticalSpacing(6);
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
        impl_->vx->setValue(0.1);
        impl_->vy->setValue(0.1);
        impl_->vz->setValue(0.2);
        impl_->interval = new QDoubleSpinBox(this);
        impl_->interval->setRange(0.0, 1.0e6);
        impl_->interval->setDecimals(3);
        impl_->interval->setSuffix(QStringLiteral(" s"));
        impl_->interval->setSpecialValueText(QStringLiteral("unknown"));
        impl_->interval->setToolTip(QStringLiteral("Time between consecutive time points; 0 = unknown"));
        impl_->acquisition = new QLineEdit(this);
        impl_->acquisition->setPlaceholderText(QStringLiteral("widefield, 3D-SIM, confocal…"));
        mg->addWidget(fieldLabel(QStringLiteral("Voxel x")), 0, 0);
        mg->addWidget(impl_->vx, 1, 0);
        mg->addWidget(fieldLabel(QStringLiteral("Voxel y")), 0, 1);
        mg->addWidget(impl_->vy, 1, 1);
        mg->addWidget(fieldLabel(QStringLiteral("Voxel z")), 0, 2);
        mg->addWidget(impl_->vz, 1, 2);
        mg->addWidget(fieldLabel(QStringLiteral("Frame interval")), 0, 3);
        mg->addWidget(impl_->interval, 1, 3);
        mg->addWidget(fieldLabel(QStringLiteral("Acquisition")), 0, 4);
        mg->addWidget(impl_->acquisition, 1, 4);
        mg->setColumnStretch(4, 1);
        root->addLayout(mg);

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
        root->addLayout(simRow);

        // channels + tile map
        root->addWidget(new Rule(2, Qt::Horizontal, this));
        auto* lower = new QHBoxLayout();
        lower->setSpacing(16);
        auto* chanCol = new QVBoxLayout();
        chanCol->setSpacing(6);
        chanCol->addWidget(new CaptionLabel(QStringLiteral("Channels"), this));
        impl_->channels = new QTableWidget(0, 4, this);
        impl_->channels->setHorizontalHeaderLabels({QStringLiteral("Token"), QStringLiteral("Label"), QStringLiteral("Wavelength (nm)"), QStringLiteral("Colour")});
        impl_->channels->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
        impl_->channels->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
        impl_->channels->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
        impl_->channels->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
        impl_->channels->verticalHeader()->hide();
        impl_->channels->setSelectionMode(QAbstractItemView::NoSelection);
        impl_->channels->setShowGrid(false);
        impl_->channels->setMaximumHeight(150);
        impl_->channels->setToolTip(QStringLiteral("Double-click a label or wavelength to edit; click the colour to change it"));
        chanCol->addWidget(impl_->channels);
        impl_->channelsNote = widgets::label(QStringLiteral("No channel group in the pattern: the files hold one channel."), 11,
                                             theme::kNeutral600, -1, this);
        impl_->channelsNote->setWordWrap(true);
        chanCol->addWidget(impl_->channelsNote);
        chanCol->addStretch(1);
        lower->addLayout(chanCol, 1);
        auto* mapCol = new QVBoxLayout();
        mapCol->setSpacing(6);
        mapCol->addWidget(new CaptionLabel(QStringLiteral("Tiles"), this));
        impl_->tileMap = new TileMap(this);
        mapCol->addWidget(impl_->tileMap);
        mapCol->addStretch(1);
        lower->addLayout(mapCol);
        root->addLayout(lower);

        // footer
        auto* buttons = new QHBoxLayout();
        buttons->addStretch(1);
        auto* cancel = new QPushButton(QStringLiteral("Cancel"), this);
        widgets::setButtonClass(cancel, "ghost");
        impl_->save = new QPushButton(QStringLiteral("Save manifest & open"), this);
        widgets::setButtonClass(impl_->save, "primary");
        impl_->save->setDefault(true);
        impl_->save->setEnabled(false);
        impl_->save->setToolTip(QStringLiteral("Writes the manifest to the path above and opens the dataset"));
        buttons->addWidget(cancel);
        buttons->addWidget(impl_->save);
        root->addLayout(buttons);

        // --- behaviour -------------------------------------------------------
        connect(cancel, &QPushButton::clicked, this, &QDialog::reject);
        auto applyPreset = [this](int i) {
            if (i < 0 || i >= static_cast<int>(std::size(kPresets))) return;
            const Preset& p = kPresets[i];
            QSignalBlocker blockPattern(impl_->pattern);
            QSignalBlocker blockPos(impl_->positions);
            impl_->pattern->setText(QString::fromLatin1(p.pattern));
            impl_->positions->setCurrentIndex(p.positions == FilenameRule::Positions::GridIndex ? 1
                                              : p.positions == FilenameRule::Positions::Microns  ? 2
                                                                                                 : 0);
            impl_->overlap->setEnabled(p.positions == FilenameRule::Positions::GridIndex);
            impl_->presets->setText(QString::fromUtf8(p.label));
            impl_->previewTimer.stop();
            impl_->runPreview();
        };
        connect(presetMenu, &QMenu::triggered, this, [applyPreset](QAction* a) {
            if (a) applyPreset(a->data().toInt());
        }, Qt::QueuedConnection);
        connect(browseManifest, &QPushButton::clicked, this, [this] {
            const QString dest = impl_->manifestPath->text().trimmed();
            const QFileInfo destInfo(dest);
            QString name = destInfo.fileName();
            if (name.isEmpty()) name = QLatin1String(DatasetManifest::kFileName);
            const QString dir = manifestBrowseDirectory(dest, impl_->folder);
            if (QDir(destInfo.absolutePath()).absolutePath() == QDir(impl_->folder).absolutePath()) {
                const QString folderName = QFileInfo(impl_->folder).fileName();
                if (!folderName.isEmpty()) name = folderName + QStringLiteral(".toml");
            }
            QString chosen = getSaveFileNameFast(
                this, QStringLiteral("Save dataset manifest"), dir, name,
                QStringLiteral("SIRIUS dataset (*.toml);;All files (*)"));
            if (chosen.isEmpty()) return;
            if (!chosen.endsWith(QLatin1String(".toml"), Qt::CaseInsensitive)) chosen += QStringLiteral(".toml");
            impl_->manifestPath->setText(chosen);
            QSettings().setValue(QLatin1String(kLastManifestDirKey), QFileInfo(chosen).absolutePath());
        });
        impl_->previewTimer.setSingleShot(true);
        impl_->previewTimer.setInterval(150);
        connect(&impl_->previewTimer, &QTimer::timeout, this, [this] { impl_->runPreview(); });
        connect(impl_->pattern, &QLineEdit::textChanged, this, [this] {
            impl_->previewTimer.start();
            const int i = indexOfPresetPattern(impl_->pattern->text());
            impl_->presets->setText(i > 0 ? QString::fromUtf8(kPresets[i - 1].label) : QStringLiteral("Presets"));
        });
        connect(impl_->positions, &SegmentedControl::changed, this, [this](int i) {
            impl_->overlap->setEnabled(i == 1);
            impl_->refreshTileMap();
        });
        connect(impl_->sim, &QCheckBox::toggled, this, [this](bool on) {
            impl_->dirs->setEnabled(on);
            impl_->phases->setEnabled(on);
            impl_->fastSi->setEnabled(on);
        });
        impl_->dirs->setEnabled(false);
        impl_->phases->setEnabled(false);
        impl_->fastSi->setEnabled(false);
        connect(impl_->channels, &QTableWidget::itemChanged, this, [this](QTableWidgetItem* it) {
            if (impl_->fillingChannels || !it) return;
            const int r = it->row();
            if (r < 0 || r >= static_cast<int>(impl_->tokens.size())) return;
            Impl::Chan& ch = impl_->chans[impl_->tokens[static_cast<std::size_t>(r)]];
            if (it->column() == 1) {
                ch.info.label = toStd(it->text().trimmed());
            } else if (it->column() == 2) {
                bool ok = false;
                const double nm = it->text().trimmed().toDouble(&ok);
                ch.info.wavelengthNm = ok && nm > 0.0 ? nm : 0.0;
                if (!ch.customColor) {
                    ch.info.color = colorForWavelength(ch.info.wavelengthNm);
                    if (ch.chip) widgets::setChipColor(ch.chip, chipColor(ch.info));
                }
            }
        });
        connect(impl_->save, &QPushButton::clicked, this, [this] { impl_->saveAndOpen(); });

        // preload from the folder's manifest
        {
            QSignalBlocker blockPattern(impl_->pattern);
            QSignalBlocker blockPos(impl_->positions);
            if (impl_->existing) {
                const DatasetManifest& m = *impl_->existing;
                impl_->vx->setValue(m.voxelUm[0] > 0.0 ? m.voxelUm[0] : 0.1);
                impl_->vy->setValue(m.voxelUm[1] > 0.0 ? m.voxelUm[1] : 0.1);
                impl_->vz->setValue(m.voxelUm[2] > 0.0 ? m.voxelUm[2] : 0.2);
                impl_->interval->setValue(m.frameIntervalS);
                impl_->acquisition->setText(fromStd(m.acquisition));
                impl_->sim->setChecked(m.sim.present);
                impl_->dirs->setValue(m.sim.ndirs);
                impl_->phases->setValue(m.sim.nphases);
                impl_->fastSi->setChecked(m.sim.fastSi);
                bool anyGrid = false, anyPos = false;
                for (const TileInfo& t : m.tiles) {
                    anyGrid = anyGrid || t.gridIndex[1] != 0 || t.gridIndex[2] != 0;
                    anyPos = anyPos || t.positionUm[1] != 0.0 || t.positionUm[2] != 0.0;
                }
                impl_->positions->setCurrentIndex(m.tiles.size() <= 1 ? 0 : anyGrid ? 1 : anyPos ? 2 : 0);
                impl_->overlap->setEnabled(impl_->positions->currentIndex() == 1);
                if (!m.pattern.empty()) impl_->pattern->setText(fromStd(m.pattern));
            } else {
                // No manifest yet: if the files are ABC AOLLS Scan_Iter names,
                // start from that preset instead of the Micro-Manager ones.
                const bool aolls = std::any_of(impl_->names.begin(), impl_->names.end(), looksLikeAollsScanIter);
                if (aolls) {
                    impl_->pattern->setText(QString::fromLatin1(kPresets[0].pattern));
                    impl_->positions->setCurrentIndex(1);
                    impl_->overlap->setEnabled(true);
                    impl_->presets->setText(QString::fromUtf8(kPresets[0].label));
                }
            }
        }
        impl_->status->setText(QStringLiteral("Matching files…"));
        QTimer::singleShot(0, this, [this] { impl_->runPreview(); });
    }

    FolderDatasetDialog::~FolderDatasetDialog() = default;

    QString FolderDatasetDialog::folder() const { return impl_->folder; }

    FilenameRule FolderDatasetDialog::Impl::rule() const {
        FilenameRule r;
        r.pattern = toStd(pattern->text().trimmed());
        r.positions = positionsMode();
        r.overlapFraction = overlap->value() / 100.0;
        r.voxelUm = {vx->value(), vy->value(), vz->value()};
        r.frameIntervalS = interval->value();
        r.sim.present = sim->isChecked();
        r.sim.ndirs = dirs->value();
        r.sim.nphases = phases->value();
        r.sim.fastSi = fastSi->isChecked();
        r.acquisition = toStd(acquisition->text().trimmed());
        for (const std::string& token : tokens) {
            const auto it = chans.find(token);
            if (it != chans.end()) r.channelInfo[token] = it->second.info;
        }
        return r;
    }

    // Match the pattern against the folder, fill the preview and derive the
    // channel table and tile map from what matched.
    void FolderDatasetDialog::Impl::runPreview() {
        const std::string pat = toStd(pattern->text().trimmed());
        matches.clear();
        patternOk = false;
        matchedCount = 0;
        QString error;
        if (!pat.empty()) {
            try {
                matches = matchFilenames(names, pat);
                patternOk = true;
            } catch (const std::exception& e) {
                error = QString::fromUtf8(e.what());
                matches.clear();
            }
        }
        if (matches.empty())
            for (const std::string& n : names) matches.push_back(FilenameMatch{n, false, {}});
        std::set<QString> times, tiles;
        for (const FilenameMatch& m : matches) {
            if (!m.matched) continue;
            ++matchedCount;
            times.insert(group(m, {"t", "time"}));
            tiles.insert(group(m, {"tile"}) + QLatin1Char('|') + group(m, {"x", "col"}) + QLatin1Char('|') + group(m, {"y", "row"}) +
                         QLatin1Char('|') + group(m, {"z"}));
        }

        // preview rows
        preview->setUpdatesEnabled(false);
        const int rows = static_cast<int>(std::min<std::size_t>(matches.size(), kMaxPreviewRows));
        preview->setRowCount(rows);
        for (int r = 0; r < rows; ++r) {
            const FilenameMatch& m = matches[static_cast<std::size_t>(r)];
            const QString cells[7] = {fromStd(m.file),
                                      m.matched ? group(m, {"channel", "c"}) : QStringLiteral("no match"),
                                      group(m, {"t", "time"}),
                                      group(m, {"tile"}),
                                      group(m, {"x", "col"}),
                                      group(m, {"y", "row"}),
                                      group(m, {"z"})};
            for (int c = 0; c < 7; ++c) {
                auto* it = new QTableWidgetItem(cells[c]);
                it->setForeground(QBrush(m.matched ? theme::kText : theme::kNeutral500));
                if (c > 0) it->setTextAlignment(Qt::AlignCenter);
                preview->setItem(r, c, it);
            }
        }
        preview->setUpdatesEnabled(true);

        refreshChannels();
        refreshTileMap();

        // status
        QString s;
        if (!error.isEmpty()) {
            s = QStringLiteral("Pattern error: %1").arg(error);
        } else if (names.empty()) {
            s = QStringLiteral("The folder holds no TIFF files.");
        } else if (pat.empty()) {
            s = QStringLiteral("Enter a pattern or pick a preset.");
        } else {
            s = QStringLiteral("%1 of %2 file(s) match").arg(matchedCount).arg(names.size());
            if (matchedCount > 0)
                s += QStringLiteral(" · %1 channel(s) · %2 time point(s) · %3 tile(s)")
                         .arg(std::max<std::size_t>(tokens.size(), 1))
                         .arg(times.size())
                         .arg(tiles.size());
            if (matches.size() > static_cast<std::size_t>(kMaxPreviewRows)) s += QStringLiteral(" · first %1 shown").arg(kMaxPreviewRows);
        }
        status->setText(s);
        QPalette p = status->palette();
        const bool bad = !error.isEmpty() || (patternOk && matchedCount == 0);
        p.setColor(QPalette::WindowText, bad ? theme::kAccent : theme::kNeutral600);
        status->setPalette(p);
        save->setEnabled(patternOk && matchedCount > 0);
    }

    // One row per channel token the pattern found, seeded from the existing
    // manifest (by label, short name or wavelength) or from the token itself.
    void FolderDatasetDialog::Impl::refreshChannels() {
        std::vector<std::string> found;
        for (const FilenameMatch& m : matches) {
            if (!m.matched) continue;
            const QString t = group(m, {"channel", "c"});
            if (t.isEmpty()) continue;
            const std::string s = toStd(t);
            if (std::find(found.begin(), found.end(), s) == found.end()) found.push_back(s);
        }
        std::sort(found.begin(), found.end(), tokenLess);
        if (found == tokens) return;
        tokens = found;
        channels->setVisible(!tokens.empty());
        channelsNote->setVisible(tokens.empty());

        fillingChannels = true;
        for (auto& entry : chans) entry.second.chip = nullptr;   // the cell widgets go with the rows
        channels->setRowCount(0);
        channels->setRowCount(static_cast<int>(tokens.size()));
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            const std::string& token = tokens[i];
            auto it = chans.find(token);
            if (it == chans.end()) {
                Chan ch;
                ch.info.label = token;
                bool num = false;
                const double nm = fromStd(token).toDouble(&num);
                if (num && nm > 100.0) ch.info.wavelengthNm = nm;
                bool seeded = false;
                if (existing) {
                    const std::vector<ChannelInfo>& ex = existing->channels;
                    for (const ChannelInfo& c : ex) {
                        if (c.label == token || c.shortName() == token ||
                            (ch.info.wavelengthNm > 0.0 && std::abs(c.wavelengthNm - ch.info.wavelengthNm) < 0.5)) {
                            ch.info = c;
                            seeded = true;
                            break;
                        }
                    }
                    if (!seeded && ex.size() == tokens.size()) {
                        ch.info = ex[i];
                        seeded = true;
                    }
                }
                if (seeded) ch.customColor = true;
                else ch.info.color = colorForWavelength(ch.info.wavelengthNm);
                it = chans.emplace(token, ch).first;
            }
            Chan& ch = it->second;
            const int r = static_cast<int>(i);
            auto* tok = new QTableWidgetItem(fromStd(token));
            tok->setFlags(tok->flags() & ~Qt::ItemIsEditable);
            tok->setForeground(QBrush(theme::kNeutral600));
            channels->setItem(r, 0, tok);
            channels->setItem(r, 1, new QTableWidgetItem(fromStd(ch.info.label)));
            auto* nm = new QTableWidgetItem(ch.info.wavelengthNm > 0.0 ? QString::number(static_cast<int>(std::lround(ch.info.wavelengthNm)))
                                                                       : QString());
            nm->setTextAlignment(Qt::AlignCenter);
            channels->setItem(r, 2, nm);
            auto* row = new ClickRow(channels);
            row->setTopRule(0);
            auto* rl = new QHBoxLayout(row);
            rl->setContentsMargins(10, 0, 10, 0);
            ch.chip = widgets::colorChip(chipColor(ch.info), 14, 14, row);
            rl->addWidget(ch.chip, 0, Qt::AlignCenter);
            row->setToolTip(QStringLiteral("Display colour"));
            row->setCursor(Qt::PointingHandCursor);
            QObject::connect(row, &ClickRow::clicked, q, [this, token] {
                const auto ct = chans.find(token);
                if (ct == chans.end()) return;
                const QColor c = QColorDialog::getColor(chipColor(ct->second.info), q,
                                                        QStringLiteral("Colour for channel %1").arg(fromStd(token)));
                if (!c.isValid()) return;
                ct->second.info.color = colorFromHex(toStd(c.name()));
                ct->second.customColor = true;
                if (ct->second.chip) widgets::setChipColor(ct->second.chip, c);
            });
            channels->setCellWidget(r, 3, row);
        }
        fillingChannels = false;
    }

    void FolderDatasetDialog::Impl::refreshTileMap() {
        const FilenameRule::Positions mode = positionsMode();
        std::vector<QPointF> pts;
        std::set<QString> tiles;
        bool anyXY = false;
        for (const FilenameMatch& m : matches) {
            if (!m.matched) continue;
            const QString x = group(m, {"x", "col"}), y = group(m, {"y", "row"});
            const QString key = group(m, {"tile"}) + QLatin1Char('|') + x + QLatin1Char('|') + y + QLatin1Char('|') + group(m, {"z"});
            if (!tiles.insert(key).second) continue;
            if (x.isEmpty() && y.isEmpty()) continue;
            anyXY = true;
            bool okx = false, oky = false;
            const double px = x.toDouble(&okx), py = y.toDouble(&oky);
            pts.emplace_back(okx ? px : 0.0, oky ? py : 0.0);
        }
        QString note;
        if (matchedCount == 0) {
            pts.clear();
            note = QStringLiteral("—");
        } else if (mode == FilenameRule::Positions::None) {
            pts.clear();
            note = tiles.size() > 1 ? QStringLiteral("%1 tiles, no positions").arg(tiles.size()) : QStringLiteral("single tile");
        } else if (!anyXY) {
            pts.clear();
            note = tiles.size() > 1 ? QStringLiteral("%1 tiles · no x / y groups").arg(tiles.size()) : QStringLiteral("single tile");
        } else if (mode == FilenameRule::Positions::GridIndex) {
            double minx = pts[0].x(), maxx = minx, miny = pts[0].y(), maxy = miny;
            for (const QPointF& p : pts) {
                minx = std::min(minx, p.x());
                maxx = std::max(maxx, p.x());
                miny = std::min(miny, p.y());
                maxy = std::max(maxy, p.y());
            }
            note = QStringLiteral("%1 tiles · %2 × %3 grid")
                       .arg(pts.size())
                       .arg(static_cast<int>(maxy - miny) + 1)
                       .arg(static_cast<int>(maxx - minx) + 1);
        } else {
            note = QStringLiteral("%1 positions (µm)").arg(pts.size());
        }
        tileMap->setTiles(std::move(pts), mode == FilenameRule::Positions::GridIndex, note);
    }

    // Build the manifest from the rule, write it (asking before replacing a
    // file that is there) and open the dataset through the workbench.
    void FolderDatasetDialog::Impl::saveAndOpen() {
        const std::filesystem::path folderPath(toStd(folder));
        QString destText = manifestPath->text().trimmed();
        if (destText.isEmpty()) destText = folder + QLatin1Char('/') + QLatin1String(DatasetManifest::kFileName);
        std::filesystem::path dest(toStd(destText));
        std::error_code ec;
        if (std::filesystem::is_directory(dest, ec)) dest /= DatasetManifest::kFileName;
        if (dest.extension().empty()) dest += ".toml";
        DatasetManifest manifest;
        std::vector<std::string> unmatched;
        try {
            manifest = manifestFromFolder(folderPath, rule(), &unmatched);
        } catch (const std::exception& e) {
            QMessageBox::warning(q, QStringLiteral("Open folder as dataset"),
                                 QStringLiteral("The files do not form a dataset:\n%1").arg(QString::fromUtf8(e.what())));
            return;
        }
        if (canonicalPath(dest.parent_path()) != canonicalPath(folderPath))
            manifest.filesFolder = canonicalPath(folderPath).string();
        if (std::filesystem::exists(dest, ec)) {
            const auto answer = QMessageBox::question(
                q, QStringLiteral("Replace manifest"),
                QStringLiteral("Replace the existing file?\n%1").arg(fromStd(dest.string())),
                QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel);
            if (answer != QMessageBox::Yes) return;
        }
        std::filesystem::create_directories(dest.parent_path(), ec);
        try {
            manifest.save(dest);
        } catch (const std::exception& e) {
            QMessageBox::warning(q, QStringLiteral("Save manifest"),
                                 QStringLiteral("Could not write %1:\n%2").arg(fromStd(dest.string()), QString::fromUtf8(e.what())));
            return;
        }
        OpenOptions options;
        options.tile = 0;
        options.readAll = true;
        if (!bridge.openDatasetAsync(dest.string(), options)) {
            QMessageBox::warning(q, QStringLiteral("Open folder as dataset"),
                                 QStringLiteral("The manifest was saved but another task is still running: cancel it or wait, then open the dataset."));
            return;
        }
        q->accept();
    }

} // namespace sirius::app
