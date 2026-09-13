#include "qt/dialogs/export_dialog.hpp"

#include <algorithm>
#include <cstring>

#include <QBoxLayout>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QVector>

#include "qt/qt_strings.hpp"
#include "qt/theme.hpp"
#include "qt/widgets/controls.hpp"

namespace sirius::app {

    using widgets::CaptionLabel;
    using widgets::ClickRow;

    namespace {

        struct FormatRow {
            QString name, note;
            ExportFormat format;
            bool tiled, pyramid;   // presets applied when the row is chosen
            int pyramidLevels;
        };

        const QVector<FormatRow>& formatRows() {
            static const QVector<FormatRow> rows = {
                {QStringLiteral("OME-TIFF"), QStringLiteral("Single file · broad compatibility"), ExportFormat::Tiff, false, false, 1},
                {QStringLiteral("Tiled TIFF"), QStringLiteral("512² tiles · random access"), ExportFormat::Tiff, true, false, 1},
                {QStringLiteral("Pyramidal OME-TIFF"), QStringLiteral("Multi-resolution · viewers & QuPath"), ExportFormat::Tiff, true, true, 5},
                {QStringLiteral("OME-Zarr"), QStringLiteral("Chunked · cloud & Dask friendly"), ExportFormat::Zarr, true, true, 5},
                {QStringLiteral("N5"), QStringLiteral("BigDataViewer · Fiji"), ExportFormat::N5, true, true, 5},
                {QStringLiteral("Raw float32"), QStringLiteral("No metadata · fastest"), ExportFormat::Raw, false, false, 1},
            };
            return rows;
        }

        QLabel* fieldLabel(const QString& t, QWidget* parent) { return widgets::label(t, 11, theme::kNeutral600, -1, parent); }

        QWidget* field(const QString& label, QWidget* editor, QWidget* parent) {
            auto* w = new QWidget(parent);
            auto* l = new QVBoxLayout(w);
            l->setContentsMargins(0, 0, 0, 0);
            l->setSpacing(4);
            l->addWidget(fieldLabel(label, w));
            l->addWidget(editor);
            return w;
        }

    } // namespace

    struct ExportDialog::Impl {
        WorkbenchBridge& bridge;
        QVector<ClickRow*> rows;
        QVector<QLabel*> sizes;
        int format = 0;
        QComboBox* step = nullptr;
        QSpinBox* t0 = nullptr;
        QSpinBox* t1 = nullptr;
        QSpinBox* z0 = nullptr;
        QSpinBox* z1 = nullptr;
        QLineEdit* channels = nullptr;
        QComboBox* dtype = nullptr;
        QComboBox* scaling = nullptr;
        QDoubleSpinBox* rangeLo = nullptr;
        QDoubleSpinBox* rangeHi = nullptr;
        QWidget* rangeBox = nullptr;
        QComboBox* compression = nullptr;
        QSpinBox* level = nullptr;
        QCheckBox* predictor = nullptr;
        QCheckBox* tiled = nullptr;
        QSpinBox* tileW = nullptr;
        QSpinBox* tileH = nullptr;
        QCheckBox* bigTiff = nullptr;
        QCheckBox* omeXml = nullptr;
        QWidget* tiffBox = nullptr;
        QSpinBox* pyramid = nullptr;
        QSpinBox* downsample = nullptr;
        QWidget* pyramidBox = nullptr;
        QComboBox* zarrVersion = nullptr;
        QLineEdit* chunk = nullptr;
        QComboBox* codec = nullptr;
        QSpinBox* zarrLevel = nullptr;
        QCheckBox* shard = nullptr;
        QCheckBox* ngff = nullptr;
        QWidget* zarrBox = nullptr;
        QLineEdit* destination = nullptr;
        QCheckBox* sidecarPipeline = nullptr;
        QCheckBox* sidecarLabels = nullptr;
        QLabel* problem = nullptr;
        QPushButton* exportBtn = nullptr;
        QLabel* stale = nullptr;
        Dims5 dims;
        // Whether the user set the t / z range. Until then it follows the
        // chosen step's extents; it used to keep the first step's, so a
        // switch from a 14-plane step to a 135-plane one exported 0..14.
        bool tEdited = false, zEdited = false;

        explicit Impl(WorkbenchBridge& b) : bridge(b) {}
    };

    ExportDialog::ExportDialog(WorkbenchBridge& bridge, QWidget* parent)
        : QDialog(parent), impl_(std::make_unique<Impl>(bridge)) {
        setWindowTitle(QStringLiteral("Export result"));
        // 640 px is the design's width to open at, not a cage: the format
        // notes and the destination path are the kind of thing a user widens.
        setSizeGripEnabled(true);
        const Workbench& wb = bridge.wb();
        auto* root = new QVBoxLayout(this);
        root->setContentsMargins(22, 18, 22, 18);
        root->setSpacing(14);
        root->addWidget(widgets::heading(QStringLiteral("Export result"), theme::kH4Px, this));

        auto* body = new QGridLayout();
        body->setHorizontalSpacing(18);
        body->setColumnStretch(0, 1);
        body->setColumnStretch(1, 1);

        // --- left: formats
        auto* left = new QWidget(this);
        auto* ll = new QVBoxLayout(left);
        ll->setContentsMargins(0, 0, 0, 0);
        ll->setSpacing(2);
        auto* cap = new CaptionLabel(QStringLiteral("Format"), left);
        cap->setContentsMargins(0, 0, 0, 6);
        ll->addWidget(cap);
        int i = 0;
        for (const FormatRow& f : formatRows()) {
            auto* row = new ClickRow(left);
            row->setTopRule(0);
            row->setHoverable(false);
            auto* rl = new QHBoxLayout(row);
            rl->setContentsMargins(10, 8, 10, 8);
            rl->setSpacing(10);
            auto* texts = new QVBoxLayout();
            texts->setContentsMargins(0, 0, 0, 0);
            texts->setSpacing(0);
            texts->addWidget(widgets::heading(f.name, 13, row));
            const bool available = exportFormatAvailable(f.format);
            texts->addWidget(widgets::label(available ? f.note : QStringLiteral("not available in this build"), 11,
                                            theme::kNeutral600, -1, row));
            rl->addLayout(texts, 1);
            auto* size = widgets::label(QString(), 11, theme::kNeutral600, -1, row);
            rl->addWidget(size, 0, Qt::AlignTop);
            row->setEnabled(available);
            if (!available) row->setToolTip(QStringLiteral("Rebuild with SIRIUS_ENABLE_TENSORSTORE=ON for zarr / N5"));
            connect(row, &ClickRow::clicked, this, [this, i] { selectFormat(i); });
            ll->addWidget(row);
            impl_->rows.push_back(row);
            impl_->sizes.push_back(size);
            ++i;
        }
        ll->addStretch(1);
        body->addWidget(left, 0, 0);

        // --- right: options
        auto* right = new QWidget(this);
        auto* rl = new QVBoxLayout(right);
        rl->setContentsMargins(0, 0, 0, 0);
        rl->setSpacing(12);

        impl_->step = new QComboBox(right);
        const Pipeline& p = wb.pipeline();
        for (int s = 0; s < p.size(); ++s) {
            QString label = fromStd(Step::number(s) + " " + p.at(s).name);
            if (!wb.output(s)) label += QStringLiteral("  (not computed)");
            else if (!wb.outputFresh(s)) label += QStringLiteral("  (out of date)");
            impl_->step->addItem(label);
        }
        impl_->step->setCurrentIndex(std::max(0, wb.viewedIndex()));
        rl->addWidget(field(QStringLiteral("From step"), impl_->step, right));
        impl_->stale = widgets::label(QString(), 11, theme::kAccentText, -1, right);
        impl_->stale->setWordWrap(true);
        impl_->stale->hide();
        rl->addWidget(impl_->stale);

        auto* rangeGrid = new QGridLayout();
        rangeGrid->setHorizontalSpacing(6);
        rangeGrid->setVerticalSpacing(4);
        auto spin = [right](const QString& prefix) {
            auto* s = new QSpinBox(right);
            s->setRange(0, 1000000);
            s->setPrefix(prefix);
            return s;
        };
        impl_->t0 = spin(QStringLiteral("t "));
        impl_->t1 = spin(QStringLiteral("to "));
        impl_->z0 = spin(QStringLiteral("z "));
        impl_->z1 = spin(QStringLiteral("to "));
        impl_->channels = new QLineEdit(right);
        impl_->channels->setPlaceholderText(QStringLiteral("all channels (or 0, 2)"));
        rangeGrid->addWidget(fieldLabel(QStringLiteral("Range"), right), 0, 0, 1, 4);
        rangeGrid->addWidget(impl_->t0, 1, 0);
        rangeGrid->addWidget(impl_->t1, 1, 1);
        rangeGrid->addWidget(impl_->z0, 1, 2);
        rangeGrid->addWidget(impl_->z1, 1, 3);
        rangeGrid->addWidget(impl_->channels, 2, 0, 1, 4);
        rl->addLayout(rangeGrid);

        auto* dtGrid = new QGridLayout();
        dtGrid->setHorizontalSpacing(10);
        impl_->dtype = new QComboBox(right);
        for (PixelType t : {PixelType::UInt8, PixelType::Int8, PixelType::UInt16, PixelType::Int16, PixelType::UInt32,
                            PixelType::Int32, PixelType::Float32, PixelType::Float64})
            impl_->dtype->addItem(QString::fromLatin1(toString(t)), static_cast<int>(t));
        impl_->dtype->setCurrentIndex(6);
        impl_->scaling = new QComboBox(right);
        impl_->scaling->addItem(QStringLiteral("cast (no rescale)"), static_cast<int>(ExportScaling::Cast));
        impl_->scaling->addItem(QStringLiteral("rescale min – max"), static_cast<int>(ExportScaling::MinMax));
        impl_->scaling->addItem(QStringLiteral("rescale fixed range"), static_cast<int>(ExportScaling::FixedRange));
        impl_->scaling->addItem(QStringLiteral("rescale percentiles"), static_cast<int>(ExportScaling::Percentile));
        dtGrid->addWidget(field(QStringLiteral("Dtype"), impl_->dtype, right), 0, 0);
        dtGrid->addWidget(field(QStringLiteral("Scaling"), impl_->scaling, right), 0, 1);
        rl->addLayout(dtGrid);

        impl_->rangeBox = new QWidget(right);
        auto* rb = new QHBoxLayout(impl_->rangeBox);
        rb->setContentsMargins(0, 0, 0, 0);
        rb->setSpacing(10);
        impl_->rangeLo = new QDoubleSpinBox(right);
        impl_->rangeLo->setRange(-1e12, 1e12);
        impl_->rangeLo->setDecimals(3);
        impl_->rangeLo->setValue(0.1);
        impl_->rangeHi = new QDoubleSpinBox(right);
        impl_->rangeHi->setRange(-1e12, 1e12);
        impl_->rangeHi->setDecimals(3);
        impl_->rangeHi->setValue(99.9);
        rb->addWidget(field(QStringLiteral("Low"), impl_->rangeLo, right));
        rb->addWidget(field(QStringLiteral("High"), impl_->rangeHi, right));
        rl->addWidget(impl_->rangeBox);

        // TIFF box
        impl_->tiffBox = new QWidget(right);
        auto* tb = new QGridLayout(impl_->tiffBox);
        tb->setContentsMargins(0, 0, 0, 0);
        tb->setHorizontalSpacing(10);
        tb->setVerticalSpacing(8);
        impl_->compression = new QComboBox(right);
        impl_->compression->addItem(QStringLiteral("none"), static_cast<int>(TiffCompression::None));
        impl_->compression->addItem(QStringLiteral("LZW"), static_cast<int>(TiffCompression::Lzw));
        impl_->compression->addItem(QStringLiteral("Deflate (zlib)"), static_cast<int>(TiffCompression::Deflate));
        impl_->compression->setCurrentIndex(2);
        impl_->level = new QSpinBox(right);
        impl_->level->setRange(1, 9);
        impl_->level->setValue(6);
        impl_->level->setPrefix(QStringLiteral("level "));
        tb->addWidget(field(QStringLiteral("Compression"), impl_->compression, right), 0, 0);
        tb->addWidget(field(QStringLiteral(" "), impl_->level, right), 0, 1);
        impl_->predictor = new QCheckBox(QStringLiteral("Predictor"), right);
        impl_->tiled = new QCheckBox(QStringLiteral("Tiled"), right);
        impl_->tileW = new QSpinBox(right);
        impl_->tileW->setRange(16, 8192);
        impl_->tileW->setSingleStep(16);
        impl_->tileW->setValue(512);
        impl_->tileH = new QSpinBox(right);
        impl_->tileH->setRange(16, 8192);
        impl_->tileH->setSingleStep(16);
        impl_->tileH->setValue(512);
        auto* tileRow = new QHBoxLayout();
        tileRow->setSpacing(6);
        tileRow->addWidget(impl_->tiled);
        tileRow->addWidget(impl_->tileW);
        tileRow->addWidget(widgets::label(QStringLiteral("×"), 12, theme::kNeutral600, -1, right));
        tileRow->addWidget(impl_->tileH);
        tb->addLayout(tileRow, 1, 0, 1, 2);
        impl_->bigTiff = new QCheckBox(QStringLiteral("BigTIFF"), right);
        impl_->bigTiff->setChecked(true);
        impl_->omeXml = new QCheckBox(QStringLiteral("OME-XML"), right);
        impl_->omeXml->setChecked(true);
        auto* flags = new QHBoxLayout();
        flags->addWidget(impl_->predictor);
        flags->addWidget(impl_->bigTiff);
        flags->addWidget(impl_->omeXml);
        tb->addLayout(flags, 2, 0, 1, 2);
        rl->addWidget(impl_->tiffBox);

        // pyramid box (TIFF + zarr)
        impl_->pyramidBox = new QWidget(right);
        auto* pb = new QHBoxLayout(impl_->pyramidBox);
        pb->setContentsMargins(0, 0, 0, 0);
        pb->setSpacing(10);
        impl_->pyramid = new QSpinBox(right);
        impl_->pyramid->setRange(1, 12);
        impl_->pyramid->setSpecialValueText(QStringLiteral("none"));
        impl_->downsample = new QSpinBox(right);
        impl_->downsample->setRange(2, 8);
        impl_->downsample->setPrefix(QStringLiteral("÷ "));
        pb->addWidget(field(QStringLiteral("Pyramid levels"), impl_->pyramid, right));
        pb->addWidget(field(QStringLiteral("Downsample"), impl_->downsample, right));
        rl->addWidget(impl_->pyramidBox);

        // zarr box
        impl_->zarrBox = new QWidget(right);
        auto* zb = new QGridLayout(impl_->zarrBox);
        zb->setContentsMargins(0, 0, 0, 0);
        zb->setHorizontalSpacing(10);
        zb->setVerticalSpacing(8);
        impl_->zarrVersion = new QComboBox(right);
        impl_->zarrVersion->addItem(QStringLiteral("zarr v3"), 3);
        impl_->zarrVersion->addItem(QStringLiteral("zarr v2"), 2);
        impl_->chunk = new QLineEdit(QStringLiteral("1, 1, 16, 512, 512"), right);
        impl_->chunk->setToolTip(QStringLiteral("Chunk shape over c, t, z, y, x"));
        impl_->codec = new QComboBox(right);
        for (const char* c : {"blosc-zstd", "blosc-lz4", "zstd", "gzip", "none"}) impl_->codec->addItem(QLatin1String(c));
        impl_->zarrLevel = new QSpinBox(right);
        impl_->zarrLevel->setRange(0, 22);
        impl_->zarrLevel->setValue(3);
        impl_->zarrLevel->setPrefix(QStringLiteral("level "));
        impl_->shard = new QCheckBox(QStringLiteral("Shard chunks (zarr v3)"), right);
        impl_->ngff = new QCheckBox(QStringLiteral("OME-NGFF multiscales metadata"), right);
        impl_->ngff->setChecked(true);
        zb->addWidget(field(QStringLiteral("Store"), impl_->zarrVersion, right), 0, 0);
        zb->addWidget(field(QStringLiteral("Chunk (c, t, z, y, x)"), impl_->chunk, right), 0, 1);
        zb->addWidget(field(QStringLiteral("Codec"), impl_->codec, right), 1, 0);
        zb->addWidget(field(QStringLiteral(" "), impl_->zarrLevel, right), 1, 1);
        zb->addWidget(impl_->shard, 2, 0);
        zb->addWidget(impl_->ngff, 2, 1);
        rl->addWidget(impl_->zarrBox);

        // destination
        auto* destRow = new QHBoxLayout();
        destRow->setSpacing(6);
        impl_->destination = new QLineEdit(right);
        auto* browse = new QPushButton(QStringLiteral("Browse"), right);
        widgets::setButtonClass(browse, "secondary small");
        destRow->addWidget(impl_->destination, 1);
        destRow->addWidget(browse);
        auto* destField = new QWidget(right);
        auto* dl = new QVBoxLayout(destField);
        dl->setContentsMargins(0, 0, 0, 0);
        dl->setSpacing(4);
        dl->addWidget(fieldLabel(QStringLiteral("Destination"), destField));
        dl->addLayout(destRow);
        rl->addWidget(destField);
        impl_->sidecarPipeline = new QCheckBox(QStringLiteral("Include pipeline sidecar (.pipeline.toml)"), right);
        impl_->sidecarPipeline->setChecked(true);
        impl_->sidecarLabels = new QCheckBox(QStringLiteral("Include labels sidecar"), right);
        rl->addWidget(impl_->sidecarPipeline);
        rl->addWidget(impl_->sidecarLabels);
        impl_->problem = widgets::label(QString(), 11, theme::kAccentText, -1, right);
        impl_->problem->setWordWrap(true);
        impl_->problem->hide();
        rl->addWidget(impl_->problem);
        rl->addStretch(1);
        body->addWidget(right, 0, 1);
        root->addLayout(body);

        // actions
        auto* actions = new QHBoxLayout();
        actions->addStretch(1);
        auto* cancel = new QPushButton(QStringLiteral("Cancel"), this);
        widgets::setButtonClass(cancel, "ghost");
        impl_->exportBtn = new QPushButton(QStringLiteral("Export"), this);
        widgets::setButtonClass(impl_->exportBtn, "primary");
        impl_->exportBtn->setDefault(true);
        actions->addWidget(cancel);
        actions->addWidget(impl_->exportBtn);
        root->addLayout(actions);

        connect(cancel, &QPushButton::clicked, this, &QDialog::reject);
        connect(impl_->exportBtn, &QPushButton::clicked, this, &QDialog::accept);
        connect(browse, &QPushButton::clicked, this, [this] {
            const ExportOptions o = options();
            const bool dir = o.format == ExportFormat::Zarr || o.format == ExportFormat::N5;
            QString chosen;
            if (dir) chosen = QFileDialog::getSaveFileName(this, QStringLiteral("Export store"), impl_->destination->text(),
                                                           QStringLiteral("Stores (*.zarr *.n5)"), nullptr, QFileDialog::ShowDirsOnly);
            else chosen = QFileDialog::getSaveFileName(this, QStringLiteral("Export file"), impl_->destination->text());
            if (!chosen.isEmpty()) impl_->destination->setText(chosen);
        });
        connect(impl_->step, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) { refresh(); });
        connect(impl_->scaling, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) { refresh(); });
        connect(impl_->dtype, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) { refresh(); });
        connect(impl_->compression, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) { refresh(); });
        connect(impl_->tiled, &QCheckBox::toggled, this, [this](bool) { refresh(); });
        connect(impl_->pyramid, qOverload<int>(&QSpinBox::valueChanged), this, [this](int) { refresh(); });
        connect(impl_->zarrVersion, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) { refresh(); });
        connect(impl_->destination, &QLineEdit::textChanged, this, [this](const QString&) { refresh(); });
        for (QSpinBox* s : {impl_->t0, impl_->t1, impl_->z0, impl_->z1})
            connect(s, qOverload<int>(&QSpinBox::valueChanged), this, [this, s](int) {
                // refresh() moves them under a signal blocker: this is the user
                (s == impl_->t0 || s == impl_->t1 ? impl_->tEdited : impl_->zEdited) = true;
                refresh();
            });

        // defaults from the dataset
        const DatasetMeta& ds = wb.dataset();
        QString base = fromStd(ds.name);
        if (base.isEmpty()) base = QStringLiteral("result");
        QString dir = QFileInfo(fromStd(ds.sourcePath)).absolutePath();
        if (dir.isEmpty() || dir == QLatin1String(".")) dir = QDir::homePath();
        impl_->destination->setText(dir + QLatin1Char('/') + base + QStringLiteral("_processed"));
        selectFormat(0);
        resize(640, std::max(sizeHint().height(), 480));
    }

    ExportDialog::~ExportDialog() = default;

    void ExportDialog::selectFormat(int i) {
        if (i < 0 || i >= impl_->rows.size() || !impl_->rows[i]->isEnabled()) return;
        impl_->format = i;
        const FormatRow& f = formatRows()[i];
        impl_->tiled->setChecked(f.tiled);
        impl_->pyramid->setValue(f.pyramid ? f.pyramidLevels : 1);
        impl_->downsample->setValue(2);
        impl_->omeXml->setChecked(f.name.contains(QStringLiteral("OME")));
        // keep the destination's stem, swap the extension
        QString dest = impl_->destination->text();
        for (const char* ext : {".ome.tif", ".ome.tiff", ".tif", ".tiff", ".zarr", ".n5", ".raw"})
            if (dest.endsWith(QLatin1String(ext), Qt::CaseInsensitive)) {
                dest.chop(static_cast<int>(strlen(ext)));
                break;
            }
        ExportOptions o = options();
        impl_->destination->setText(dest + fromStd(exportExtension(o)));
        refresh();
    }

    void ExportDialog::refresh() {
        const Workbench& wb = impl_->bridge.wb();
        const int step = impl_->step->currentIndex();
        DatasetMeta meta = step >= 0 ? wb.outputMetaOf(step) : wb.dataset();
        if (auto out = wb.output(step)) meta = out->meta;
        impl_->dims = meta.dims;
        {
            const QSignalBlocker b0(impl_->t0), b1(impl_->t1), b2(impl_->z0), b3(impl_->z1);
            impl_->t1->setMaximum(static_cast<int>(meta.dims.t));
            impl_->t0->setMaximum(static_cast<int>(std::max<Index>(meta.dims.t - 1, 0)));
            impl_->z1->setMaximum(static_cast<int>(meta.dims.z));
            impl_->z0->setMaximum(static_cast<int>(std::max<Index>(meta.dims.z - 1, 0)));
            // the whole of the chosen step until the user narrows it; a range
            // the user set is kept (clamped to the new step) when the step changes
            if (!impl_->tEdited) impl_->t0->setValue(0);
            if (!impl_->tEdited || impl_->t1->value() == 0) impl_->t1->setValue(static_cast<int>(meta.dims.t));
            if (!impl_->zEdited) impl_->z0->setValue(0);
            if (!impl_->zEdited || impl_->z1->value() == 0) impl_->z1->setValue(static_cast<int>(meta.dims.z));
        }
        // The file gets the step's last output, while the sidecar records the
        // pipeline as it is now: after a parameter edit or an undo the two do
        // not belong together, and nothing said so.
        const bool stale = wb.output(step) && !wb.outputFresh(step);
        impl_->stale->setText(stale ? QStringLiteral("The parameters changed since step %1 was computed: the export writes that earlier "
                                                     "result, and a pipeline sidecar would record the current parameters, which did not "
                                                     "produce it. Run the step again for a matching pair.")
                                          .arg(fromStd(Step::number(step)))
                                    : QString());
        impl_->stale->setVisible(stale);

        for (int i = 0; i < impl_->rows.size(); ++i) {
            impl_->rows[i]->setSelected(i == impl_->format);
            ExportOptions probe = options();
            const FormatRow& f = formatRows()[i];
            probe.format = f.format;
            probe.tiff.tiled = f.tiled;
            probe.tiff.pyramidLevels = f.pyramid ? f.pyramidLevels : 1;
            probe.zarr.pyramidLevels = f.pyramid ? f.pyramidLevels : 1;
            impl_->sizes[i]->setText(widgets::bytesText(estimateExportBytes(meta.dims, probe)));
        }
        const ExportOptions o = options();
        const bool tiff = o.format == ExportFormat::Tiff;
        const bool zarr = o.format == ExportFormat::Zarr || o.format == ExportFormat::N5;
        impl_->tiffBox->setVisible(tiff);
        impl_->zarrBox->setVisible(zarr);
        impl_->pyramidBox->setVisible(tiff || zarr);
        impl_->tileW->setEnabled(impl_->tiled->isChecked());
        impl_->tileH->setEnabled(impl_->tiled->isChecked());
        impl_->level->setEnabled(o.tiff.compression == TiffCompression::Deflate);
        impl_->predictor->setEnabled(o.tiff.compression != TiffCompression::None);
        impl_->shard->setEnabled(o.format == ExportFormat::Zarr && o.zarr.zarrVersion == 3);
        impl_->zarrVersion->setEnabled(o.format == ExportFormat::Zarr);
        impl_->rangeBox->setVisible(o.scaling == ExportScaling::FixedRange || o.scaling == ExportScaling::Percentile);
        impl_->exportBtn->setText(QStringLiteral("Export %1").arg(formatRows()[impl_->format].name));
        const std::string problem = validateExport(o, meta.dims);
        impl_->problem->setText(fromStd(problem));
        impl_->problem->setVisible(!problem.empty());
        impl_->exportBtn->setEnabled(problem.empty() && !impl_->destination->text().trimmed().isEmpty());
        impl_->sidecarLabels->setEnabled(wb.output(step) && wb.output(step)->labels);
    }

    int ExportDialog::stepIndex() const { return impl_->step->currentIndex(); }

    ExportOptions ExportDialog::options() const {
        ExportOptions o;
        const FormatRow& f = formatRows()[impl_->format];
        o.format = f.format;
        o.path = toStd(impl_->destination->text().trimmed());
        o.dtype = static_cast<PixelType>(impl_->dtype->currentData().toInt());
        o.scaling = static_cast<ExportScaling>(impl_->scaling->currentData().toInt());
        if (o.scaling == ExportScaling::FixedRange) {
            o.rangeLo = impl_->rangeLo->value();
            o.rangeHi = impl_->rangeHi->value();
        } else if (o.scaling == ExportScaling::Percentile) {
            o.percentileLo = impl_->rangeLo->value();
            o.percentileHi = impl_->rangeHi->value();
        }
        o.range.t0 = impl_->t0->value();
        o.range.t1 = impl_->t1->value() > 0 ? impl_->t1->value() : -1;
        o.range.z0 = impl_->z0->value();
        o.range.z1 = impl_->z1->value() > 0 ? impl_->z1->value() : -1;
        for (const QString& part : impl_->channels->text().split(QLatin1Char(','), Qt::SkipEmptyParts)) {
            bool ok = false;
            const int c = part.trimmed().toInt(&ok);
            if (ok) o.range.channels.push_back(c);
        }
        o.tiff.tiled = impl_->tiled->isChecked();
        o.tiff.tileWidth = impl_->tileW->value();
        o.tiff.tileHeight = impl_->tileH->value();
        o.tiff.compression = static_cast<TiffCompression>(impl_->compression->currentData().toInt());
        o.tiff.compressionLevel = impl_->level->value();
        o.tiff.predictor = impl_->predictor->isChecked();
        o.tiff.bigTiff = impl_->bigTiff->isChecked();
        o.tiff.omeXml = impl_->omeXml->isChecked();
        o.tiff.pyramidLevels = impl_->pyramid->value();
        o.tiff.downsample = impl_->downsample->value();
        o.zarr.zarrVersion = impl_->zarrVersion->currentData().toInt();
        {
            const QStringList parts = impl_->chunk->text().split(QLatin1Char(','), Qt::SkipEmptyParts);
            for (int i = 0; i < 5 && i < parts.size(); ++i) {
                bool ok = false;
                const int v = parts[i].trimmed().toInt(&ok);
                if (ok && v > 0) o.zarr.chunk[static_cast<std::size_t>(i)] = v;
            }
        }
        o.zarr.codec = toStd(impl_->codec->currentText());
        o.zarr.level = impl_->zarrLevel->value();
        o.zarr.shard = impl_->shard->isChecked();
        o.zarr.pyramidLevels = impl_->pyramid->value();
        o.zarr.downsample = impl_->downsample->value();
        o.zarr.omeNgff = impl_->ngff->isChecked();
        o.includePipeline = impl_->sidecarPipeline->isChecked();
        o.includeLabels = impl_->sidecarLabels->isChecked() && impl_->sidecarLabels->isEnabled();
        return o;
    }

} // namespace sirius::app
