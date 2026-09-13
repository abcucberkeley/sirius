#include "qt/dialogs/training_export_dialog.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>

#include <QBoxLayout>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>

#include "qt/qt_strings.hpp"
#include "qt/theme.hpp"
#include "qt/widgets/controls.hpp"

namespace sirius::app {

    namespace {

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

    struct TrainingExportDialog::Impl {
        explicit Impl(WorkbenchBridge& b) : bridge(b) {}
        WorkbenchBridge& bridge;
        QComboBox* step = nullptr;
        QLineEdit* directory = nullptr;
        QLineEdit* sample = nullptr;
        QCheckBox* image = nullptr;
        QCheckBox* instances = nullptr;
        QCheckBox* semantic = nullptr;
        QCheckBox* boxes = nullptr;
        QCheckBox* slices = nullptr;
        QSpinBox* minVoxels = nullptr;
        QComboBox* dtype = nullptr;
        QComboBox* scaling = nullptr;
        QLabel* summary = nullptr;
        QLabel* problem = nullptr;
        QLabel* stale = nullptr;
        QPushButton* exportBtn = nullptr;

        // boundingBoxes() walks every voxel of every time point. refresh()
        // runs on every keystroke in the folder field and on every checkbox,
        // so the count is kept until the step or the size filter moves.
        std::shared_ptr<const StepOutput> countedOut;
        int countedStep = -1;
        std::uint64_t countedMin = 0;
        QString countedSummary;
    };

    TrainingExportDialog::TrainingExportDialog(WorkbenchBridge& bridge, QWidget* parent)
        : QDialog(parent), impl_(std::make_unique<Impl>(bridge)) {
        setWindowTitle(QStringLiteral("Export training data"));
        const Workbench& wb = bridge.wb();

        auto* root = new QVBoxLayout(this);
        root->setContentsMargins(20, 20, 20, 20);
        root->setSpacing(12);

        auto* intro = widgets::label(QStringLiteral("Writes one sample folder per export — instance masks, a semantic mask and "
                                                    "bounding boxes — and appends it to index.jsonl, so a folder collects the "
                                                    "output of many runs into one training set."),
                                     11, theme::kNeutral600, -1, this);
        intro->setWordWrap(true);
        root->addWidget(intro);

        impl_->step = new QComboBox(this);
        const Pipeline& p = wb.pipeline();
        for (int s = 0; s < p.size(); ++s) {
            QString label = fromStd(Step::number(s) + " " + p.at(s).name);
            const auto out = wb.output(s);
            if (!out) label += QStringLiteral("  (not computed)");
            else if (!out->labels || out->labels->empty()) label += QStringLiteral("  (no labels)");
            else if (!wb.outputFresh(s)) label += QStringLiteral("  (out of date)");
            impl_->step->addItem(label);
        }
        impl_->step->setCurrentIndex(std::max(0, wb.viewedIndex()));
        root->addWidget(field(QStringLiteral("Labels from step"), impl_->step, this));
        impl_->stale = widgets::label(QString(), 11, theme::kAccentText, -1, this);
        impl_->stale->setWordWrap(true);
        impl_->stale->hide();
        root->addWidget(impl_->stale);

        auto* destRow = new QHBoxLayout();
        impl_->directory = new QLineEdit(this);
        impl_->directory->setPlaceholderText(QStringLiteral("dataset folder"));
        auto* browse = new QPushButton(QStringLiteral("Browse…"), this);
        widgets::setButtonClass(browse, "ghost");
        destRow->addWidget(impl_->directory, 1);
        destRow->addWidget(browse);
        auto* destWidget = new QWidget(this);
        destWidget->setLayout(destRow);
        destRow->setContentsMargins(0, 0, 0, 0);
        root->addWidget(field(QStringLiteral("Dataset folder"), destWidget, this));

        impl_->sample = new QLineEdit(this);
        impl_->sample->setPlaceholderText(QStringLiteral("sample name"));
        root->addWidget(field(QStringLiteral("Sample"), impl_->sample, this));

        root->addWidget(fieldLabel(QStringLiteral("Write"), this));
        impl_->image = new QCheckBox(QStringLiteral("Image (image.tif)"), this);
        impl_->instances = new QCheckBox(QStringLiteral("Instance masks (instances.tif, one id per object)"), this);
        impl_->semantic = new QCheckBox(QStringLiteral("Semantic mask (semantic.tif, one id per class)"), this);
        impl_->boxes = new QCheckBox(QStringLiteral("Bounding boxes (boxes.json, 3D and per plane)"), this);
        impl_->slices = new QCheckBox(QStringLiteral("2D slices (slices/: one 8-bit plane and one YOLO file each)"), this);
        for (QCheckBox* c : {impl_->image, impl_->instances, impl_->semantic, impl_->boxes}) c->setChecked(true);
        for (QCheckBox* c : {impl_->image, impl_->instances, impl_->semantic, impl_->boxes, impl_->slices}) root->addWidget(c);

        impl_->minVoxels = new QSpinBox(this);
        impl_->minVoxels->setRange(1, 1000000);
        impl_->minVoxels->setValue(1);
        impl_->minVoxels->setSuffix(QStringLiteral(" voxels"));
        impl_->dtype = new QComboBox(this);
        for (PixelType t : {PixelType::UInt8, PixelType::UInt16, PixelType::Float32})
            impl_->dtype->addItem(QString::fromLatin1(toString(t)), static_cast<int>(t));
        impl_->dtype->setCurrentIndex(1);
        impl_->scaling = new QComboBox(this);
        impl_->scaling->addItem(QStringLiteral("rescale percentiles"), static_cast<int>(ExportScaling::Percentile));
        impl_->scaling->addItem(QStringLiteral("rescale min – max"), static_cast<int>(ExportScaling::MinMax));
        impl_->scaling->addItem(QStringLiteral("cast (no rescale)"), static_cast<int>(ExportScaling::Cast));
        auto* opts = new QHBoxLayout();
        opts->addWidget(field(QStringLiteral("Smallest object"), impl_->minVoxels, this));
        opts->addWidget(field(QStringLiteral("Image pixel type"), impl_->dtype, this));
        opts->addWidget(field(QStringLiteral("Image scaling"), impl_->scaling, this), 1);
        root->addLayout(opts);

        impl_->summary = widgets::label(QString(), 11, theme::kNeutral600, -1, this);
        impl_->summary->setWordWrap(true);
        root->addWidget(impl_->summary);
        impl_->problem = widgets::label(QString(), 11, theme::kAccentText, -1, this);
        impl_->problem->setWordWrap(true);
        impl_->problem->hide();
        root->addWidget(impl_->problem);
        root->addStretch(1);

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
            const QString chosen = QFileDialog::getExistingDirectory(this, QStringLiteral("Training dataset folder"), impl_->directory->text());
            if (!chosen.isEmpty()) impl_->directory->setText(chosen);
        });
        connect(impl_->step, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) { refresh(); });
        connect(impl_->directory, &QLineEdit::textChanged, this, [this](const QString&) { refresh(); });
        connect(impl_->minVoxels, qOverload<int>(&QSpinBox::valueChanged), this, [this](int) { refresh(); });
        for (QCheckBox* c : {impl_->image, impl_->instances, impl_->semantic, impl_->boxes, impl_->slices})
            connect(c, &QCheckBox::toggled, this, [this](bool) { refresh(); });

        // defaults from the dataset: the folder beside it, the step as the name
        const DatasetMeta& ds = wb.dataset();
        QString dir = QFileInfo(fromStd(ds.sourcePath)).absolutePath();
        if (dir.isEmpty() || dir == QLatin1String(".")) dir = QDir::homePath();
        impl_->directory->setText(dir + QStringLiteral("/training-data"));
        QString base = fromStd(ds.name);
        if (base.isEmpty()) base = QStringLiteral("sample");
        impl_->sample->setText(base);
        refresh();
        resize(620, std::max(sizeHint().height(), 520));
    }

    TrainingExportDialog::~TrainingExportDialog() = default;

    int TrainingExportDialog::stepIndex() const { return impl_->step->currentIndex(); }

    void TrainingExportDialog::refresh() {
        const Workbench& wb = impl_->bridge.wb();
        const int step = impl_->step->currentIndex();
        const auto out = step >= 0 ? wb.output(step) : nullptr;
        const LabelVolume* labels = out ? out->labels.get() : nullptr;

        QString summary;
        if (labels != nullptr && !labels->empty()) {
            const std::uint64_t minVoxels = static_cast<std::uint64_t>(impl_->minVoxels->value());
            if (out != impl_->countedOut || step != impl_->countedStep || minVoxels != impl_->countedMin) {
                const ClassTable classes = classTable(*labels);
                std::uint64_t objects = 0;
                for (Index t = 0; t < labels->t(); ++t) objects += boundingBoxes(*labels, t, classes, minVoxels).size();
                impl_->countedSummary = QStringLiteral("%1 object%2 over %3 time point%4, %5 class%6.")
                                            .arg(objects)
                                            .arg(objects == 1 ? QString() : QStringLiteral("s"))
                                            .arg(labels->t())
                                            .arg(labels->t() == 1 ? QString() : QStringLiteral("s"))
                                            .arg(classes.size())
                                            .arg(classes.size() == 1 ? QString() : QStringLiteral("es"));
                impl_->countedOut = out;
                impl_->countedStep = step;
                impl_->countedMin = minVoxels;
            }
            summary = impl_->countedSummary;
            if (impl_->slices->isChecked())
                summary += QStringLiteral(" The slice output writes %1 plane files.").arg(labels->t() * labels->z() * 2);
        }
        impl_->summary->setText(summary);
        // The sample keeps the pipeline as it is now as its provenance, and
        // after a parameter edit or an undo that is not the one that made
        // these labels.
        const bool stale = labels != nullptr && !labels->empty() && !wb.outputFresh(step);
        impl_->stale->setText(stale ? QStringLiteral("The parameters changed since step %1 was computed: these labels come from the "
                                                     "earlier parameters, and the sample's provenance records the current pipeline, "
                                                     "which did not make them. Run the step again for a matching record.")
                                          .arg(fromStd(Step::number(step)))
                                    : QString());
        impl_->stale->setVisible(stale);

        std::string problem;
        if (!out) problem = "step " + Step::number(step) + " has not been computed yet; run it first";
        else if (labels == nullptr || labels->empty()) problem = "step " + Step::number(step) + " produced no labels; segment first";
        else problem = validateTrainingExport(options(), *labels);
        impl_->problem->setText(fromStd(problem));
        impl_->problem->setVisible(!problem.empty());
        impl_->exportBtn->setEnabled(problem.empty());
        impl_->dtype->setEnabled(impl_->image->isChecked() || impl_->slices->isChecked());
        impl_->scaling->setEnabled(impl_->image->isChecked());
    }

    TrainingExportOptions TrainingExportDialog::options() const {
        TrainingExportOptions o;
        o.directory = toStd(impl_->directory->text().trimmed());
        o.sample = toStd(impl_->sample->text().trimmed());
        if (o.sample.empty()) o.sample = "sample";
        o.image = impl_->image->isChecked();
        o.instances = impl_->instances->isChecked();
        o.semantic = impl_->semantic->isChecked();
        o.boxes = impl_->boxes->isChecked();
        o.slices = impl_->slices->isChecked();
        o.minVoxels = static_cast<std::uint64_t>(impl_->minVoxels->value());
        o.imageDtype = static_cast<PixelType>(impl_->dtype->currentData().toInt());
        o.scaling = static_cast<ExportScaling>(impl_->scaling->currentData().toInt());
        return o;
    }

} // namespace sirius::app
