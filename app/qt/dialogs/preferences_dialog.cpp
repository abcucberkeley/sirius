#include "qt/dialogs/preferences_dialog.hpp"

#include <algorithm>
#include <exception>

#include <QBoxLayout>
#include <QCheckBox>
#include <QComboBox>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QTabWidget>

#include <sirius/device.hpp>

#include "qt/panels/assistant_panel.hpp"
#include "qt/panels/llm_client.hpp"
#include "qt/qt_strings.hpp"
#include "qt/secret_store.hpp"
#include "qt/theme.hpp"
#include "qt/widgets/controls.hpp"

namespace sirius::app {

    using widgets::CaptionLabel;
    using widgets::Rule;

    namespace {
        QWidget* field(const QString& label, QWidget* editor, QWidget* parent) {
            auto* w = new QWidget(parent);
            auto* l = new QVBoxLayout(w);
            l->setContentsMargins(0, 0, 0, 0);
            l->setSpacing(4);
            l->addWidget(widgets::label(label, 11, theme::kNeutral600, -1, w));
            l->addWidget(editor);
            return w;
        }

        QString deviceLabel(int i) {
            try {
                const DeviceProperties p = deviceProperties(Device::cuda(i));
                return QStringLiteral("cuda:%1 · %2 · %3 GB")
                    .arg(i)
                    .arg(fromStd(p.name))
                    .arg(static_cast<double>(p.totalMemoryBytes) / (1024.0 * 1024.0 * 1024.0), 0, 'f', 1);
            } catch (const std::exception&) {
                return QStringLiteral("cuda:%1").arg(i);
            }
        }
    } // namespace

    struct PreferencesDialog::Impl {
        WorkbenchBridge& bridge;

        // The server's model list into the dropdown, keeping whatever is
        // typed; for Ollama the models held in memory are marked, since
        // the first answer from any other one waits for a load.
        void refreshModelList() {
            const QString keep = model->currentText().trimmed();
            const QString base = baseUrl->text().trimmed(), key = apiKey->text();
            const bool ollama = provider->currentData().toString() == QLatin1String("ollama");
            listedModels.clear();
            refreshModels->setEnabled(false);
            modelNote->setText(QStringLiteral("Asking %1 for its models…").arg(base));
            client.fetchModels(base, key, [this, keep, base, ollama](QStringList ids, QString error) {
                refreshModels->setEnabled(true);
                if (ids.isEmpty()) {
                    modelNote->setText(error.isEmpty() ? QStringLiteral("%1 lists no models.").arg(base)
                                                       : QStringLiteral("Cannot list the models at %1 (%2); type a name.").arg(base, error));
                    return;
                }
                ids.sort(Qt::CaseInsensitive);
                listedModels = ids;
                model->clear();
                model->addItems(ids);
                if (!keep.isEmpty()) model->setCurrentText(keep);
                else model->setCurrentIndex(0);
                modelNote->setText(QStringLiteral("%1 model(s) at %2.").arg(ids.size()).arg(base));
                if (!ollama) return;
                client.fetchLoadedModels(base, [this, ids](QStringList loaded, QString) {
                    if (loaded.isEmpty()) {
                        modelNote->setText(modelNote->text() +
                                           QStringLiteral(" None is loaded yet: the first answer waits for a load, a minute or two for a large model."));
                        return;
                    }
                    modelNote->setText(modelNote->text() + QStringLiteral(" In memory now: %1.").arg(loaded.join(QStringLiteral(", "))));
                });
            });
        }
        QComboBox* backend = nullptr;
        QComboBox* device = nullptr;
        QLineEdit* host = nullptr;
        QSpinBox* port = nullptr;
        QLineEdit* token = nullptr;
        QLineEdit* python = nullptr;
        QLineEdit* hfToken = nullptr;
        QComboBox* provider = nullptr;
        QLineEdit* baseUrl = nullptr;
        QComboBox* model = nullptr;          // editable: the server's list, or any name typed
        QPushButton* refreshModels = nullptr;
        QLabel* modelNote = nullptr;
        QStringList listedModels;   // what the server said it has, empty until it answers
        LlmClient client;
        QLineEdit* apiKey = nullptr;
        QCheckBox* askFirst = nullptr;
        explicit Impl(WorkbenchBridge& b) : bridge(b) {}
    };

    PreferencesDialog::PreferencesDialog(WorkbenchBridge& bridge, QWidget* parent)
        : QDialog(parent), impl_(std::make_unique<Impl>(bridge)) {
        setWindowTitle(QStringLiteral("Preferences"));
        setMinimumWidth(560);
        const Workbench& wb = bridge.wb();
        QSettings settings;
        auto* root = new QVBoxLayout(this);
        root->setContentsMargins(22, 18, 22, 18);
        root->setSpacing(14);
        root->addWidget(widgets::heading(QStringLiteral("Preferences"), theme::kH4Px, this));

        auto* tabs = new QTabWidget(this);
        tabs->setDocumentMode(true);

        // --- Compute
        auto* compute = new QWidget(tabs);
        auto* cl = new QVBoxLayout(compute);
        cl->setContentsMargins(0, 14, 0, 0);
        cl->setSpacing(12);
        impl_->backend = new QComboBox(compute);
        impl_->backend->addItems({QStringLiteral("CUDA"), QStringLiteral("CPU"), QStringLiteral("HPC (remote worker)")});
        impl_->backend->setCurrentIndex(static_cast<int>(wb.backend()));
        impl_->device = new QComboBox(compute);
        const int n = cudaDeviceCount();
        for (int i = 0; i < n; ++i) impl_->device->addItem(deviceLabel(i), i);
        if (n == 0) {
            impl_->device->addItem(QStringLiteral("no CUDA device"), 0);
            impl_->device->setEnabled(false);
        }
        impl_->device->setCurrentIndex(std::max(0, std::min(wb.cudaDevice(), n - 1)));
        auto* g = new QGridLayout();
        g->setHorizontalSpacing(10);
        g->addWidget(field(QStringLiteral("Default backend"), impl_->backend, compute), 0, 0);
        g->addWidget(field(QStringLiteral("CUDA device"), impl_->device, compute), 0, 1);
        cl->addLayout(g);
        cl->addWidget(new Rule(2, Qt::Horizontal, compute));
        cl->addWidget(new CaptionLabel(QStringLiteral("HPC worker"), compute));
        auto* hg = new QGridLayout();
        hg->setHorizontalSpacing(10);
        impl_->host = new QLineEdit(fromStd(wb.remoteConfig().host), compute);
        impl_->port = new QSpinBox(compute);
        impl_->port->setRange(1, 65535);
        impl_->port->setValue(wb.remoteConfig().port);
        impl_->token = new QLineEdit(fromStd(wb.remoteConfig().token), compute);
        impl_->token->setEchoMode(QLineEdit::Password);
        hg->addWidget(field(QStringLiteral("Host"), impl_->host, compute), 0, 0);
        hg->addWidget(field(QStringLiteral("Port"), impl_->port, compute), 0, 1);
        hg->addWidget(field(QStringLiteral("Token"), impl_->token, compute), 1, 0, 1, 2);
        cl->addLayout(hg);
        auto* hpcNote = widgets::label(
            QStringLiteral("Launch the worker on the cluster with app/python/slurm/sirius_worker.sbatch and forward its port "
                           "(ssh -L). The same worker runs Torch models locally."),
            11, theme::kNeutral600, -1, compute);
        hpcNote->setWordWrap(true);
        cl->addWidget(hpcNote);
        cl->addWidget(new Rule(2, Qt::Horizontal, compute));
        impl_->python = new QLineEdit(settings.value(QStringLiteral("worker/python"), QStringLiteral("python3")).toString(), compute);
        impl_->python->setToolTip(QStringLiteral("Interpreter with numpy (and torch for segmentation); SIRIUS_PYTHON overrides"));
        cl->addWidget(field(QStringLiteral("Python for the local worker"), impl_->python, compute));
        impl_->hfToken = new QLineEdit(secrets::read(QStringLiteral("hub/token")), compute);
        impl_->hfToken->setEchoMode(QLineEdit::Password);
        impl_->hfToken->setToolTip(QStringLiteral("Access token for gated or private Hugging Face repositories (huggingface.co ▸ Settings ▸ "
                                                  "Access Tokens); sent with each request that downloads a model"));
        cl->addWidget(field(QStringLiteral("Hugging Face access token (optional)"), impl_->hfToken, compute));
        cl->addStretch(1);
        tabs->addTab(compute, QStringLiteral("Compute"));

        // --- Assistant
        auto* assistant = new QWidget(tabs);
        auto* al = new QVBoxLayout(assistant);
        al->setContentsMargins(0, 14, 0, 0);
        al->setSpacing(12);
        const AssistantSettings as = AssistantSettings::load();
        impl_->provider = new QComboBox(assistant);
        impl_->provider->addItem(QStringLiteral("Ollama (local)"), QStringLiteral("ollama"));
        impl_->provider->addItem(QStringLiteral("OpenRouter"), QStringLiteral("openrouter"));
        impl_->provider->addItem(QStringLiteral("Custom OpenAI-compatible"), QStringLiteral("custom"));
        impl_->provider->setCurrentIndex(std::max(0, impl_->provider->findData(as.provider)));
        impl_->baseUrl = new QLineEdit(as.baseUrl, assistant);
        impl_->model = new QComboBox(assistant);
        impl_->model->setEditable(true);
        impl_->model->setInsertPolicy(QComboBox::NoInsert);
        impl_->model->lineEdit()->setPlaceholderText(QStringLiteral("e.g. llama3.1:8b or anthropic/claude-sonnet-4"));
        impl_->model->setToolTip(QStringLiteral("The models the server lists (Refresh asks it again), or any name typed here"));
        if (!as.model.isEmpty()) {
            impl_->model->addItem(as.model);
            impl_->model->setCurrentText(as.model);
        }
        impl_->refreshModels = new QPushButton(QStringLiteral("Refresh"), assistant);
        impl_->refreshModels->setToolTip(QStringLiteral("Ask the server at the base URL which models it offers"));
        impl_->modelNote = widgets::label(QString(), 11, theme::kNeutral600, -1, assistant);
        impl_->modelNote->setWordWrap(true);
        impl_->apiKey = new QLineEdit(as.apiKey, assistant);
        impl_->apiKey->setEchoMode(QLineEdit::Password);
        impl_->askFirst = new QCheckBox(QStringLiteral("Ask before acting (the assistant proposes, you confirm)"), assistant);
        impl_->askFirst->setChecked(as.askBeforeActing);
        auto* ag = new QGridLayout();
        ag->setHorizontalSpacing(10);
        ag->addWidget(field(QStringLiteral("Provider"), impl_->provider, assistant), 0, 0);
        auto* modelRow = new QWidget(assistant);
        auto* mh = new QHBoxLayout(modelRow);
        mh->setContentsMargins(0, 0, 0, 0);
        mh->setSpacing(6);
        mh->addWidget(impl_->model, 1);
        mh->addWidget(impl_->refreshModels);
        ag->addWidget(field(QStringLiteral("Model"), modelRow, assistant), 0, 1);
        ag->addWidget(field(QStringLiteral("Base URL"), impl_->baseUrl, assistant), 1, 0, 1, 2);
        ag->addWidget(impl_->modelNote, 2, 0, 1, 2);
        ag->addWidget(field(QStringLiteral("API key"), impl_->apiKey, assistant), 3, 0, 1, 2);
        al->addLayout(ag);
        al->addWidget(impl_->askFirst);
        auto* note = widgets::label(
            QStringLiteral("Ollama needs a model with tool calling (ollama pull llama3.1). OpenRouter keys start with sk-or-; "
                           "the key is stored in the application settings."),
            11, theme::kNeutral600, -1, assistant);
        note->setWordWrap(true);
        al->addWidget(note);
        al->addStretch(1);
        tabs->addTab(assistant, QStringLiteral("Assistant"));
        root->addWidget(tabs, 1);
        // the tab last looked at comes back first (the assistant's settings
        // get revisited far more often than the compute ones)
        tabs->setCurrentIndex(std::clamp(settings.value(QStringLiteral("prefs/tab"), 0).toInt(), 0, tabs->count() - 1));
        connect(this, &QDialog::finished, this, [tabs](int) { QSettings().setValue(QStringLiteral("prefs/tab"), tabs->currentIndex()); });

        connect(impl_->provider, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) {
            const QString p = impl_->provider->currentData().toString();
            if (p == QLatin1String("ollama")) impl_->baseUrl->setText(QStringLiteral("http://localhost:11434/v1"));
            else if (p == QLatin1String("openrouter")) impl_->baseUrl->setText(QStringLiteral("https://openrouter.ai/api/v1"));
            impl_->apiKey->setEnabled(p != QLatin1String("ollama"));
            impl_->refreshModelList();
        });
        impl_->apiKey->setEnabled(as.provider != QLatin1String("ollama"));
        connect(impl_->refreshModels, &QPushButton::clicked, this, [this] { impl_->refreshModelList(); });
        impl_->refreshModelList();   // the list on opening, without blocking the dialog

        auto* actions = new QHBoxLayout();
        actions->addStretch(1);
        auto* cancel = new QPushButton(QStringLiteral("Cancel"), this);
        widgets::setButtonClass(cancel, "ghost");
        auto* ok = new QPushButton(QStringLiteral("Save"), this);
        widgets::setButtonClass(ok, "primary");
        ok->setDefault(true);
        actions->addWidget(cancel);
        actions->addWidget(ok);
        root->addLayout(actions);
        connect(cancel, &QPushButton::clicked, this, &QDialog::reject);
        connect(ok, &QPushButton::clicked, this, [this] {
            apply();
            accept();
        });
    }

    PreferencesDialog::~PreferencesDialog() = default;

    void PreferencesDialog::apply() {
        Workbench& wb = impl_->bridge.wb();
        QSettings settings;
        settings.setValue(QStringLiteral("compute/backend"), impl_->backend->currentIndex());
        settings.setValue(QStringLiteral("compute/cudaDevice"), impl_->device->currentData().toInt());
        settings.setValue(QStringLiteral("hpc/host"), impl_->host->text().trimmed());
        settings.setValue(QStringLiteral("hpc/port"), impl_->port->value());
        QStringList notStored;
        if (!secrets::write(QStringLiteral("hpc/token"), impl_->token->text())) notStored << QStringLiteral("the HPC token");
        settings.setValue(QStringLiteral("worker/python"), impl_->python->text().trimmed());
        if (!secrets::write(QStringLiteral("hub/token"), impl_->hfToken->text().trimmed())) notStored << QStringLiteral("the Hugging Face token");
        AssistantSettings as;
        as.provider = impl_->provider->currentData().toString();
        as.baseUrl = impl_->baseUrl->text().trimmed();
        as.model = LlmClient::resolveModel(impl_->model->currentText(), impl_->listedModels);
        as.apiKey = impl_->apiKey->text();
        as.askBeforeActing = impl_->askFirst->isChecked();
        as.save();
        wb.setBackend(static_cast<Backend>(impl_->backend->currentIndex()));
        wb.setCudaDevice(impl_->device->currentData().toInt());
        RemoteConfig rc;
        rc.host = toStd(impl_->host->text().trimmed());
        rc.port = impl_->port->value();
        rc.token = toStd(impl_->token->text());
        wb.setRemoteConfig(rc);
        emit assistantSettingsChanged();
        if (!notStored.isEmpty())
            QMessageBox::warning(this, QStringLiteral("Preferences"),
                                 QStringLiteral("Could not store %1 in the secret store: it will not be there at the next launch.")
                                     .arg(notStored.join(QStringLiteral(" and "))));
    }

    void PreferencesDialog::applyStored(Workbench& wb) {
        QSettings settings;
        const int backend = settings.value(QStringLiteral("compute/backend"), cudaAvailable() ? 0 : 1).toInt();
        wb.setBackend(static_cast<Backend>(std::max(0, std::min(backend, 2))));
        wb.setCudaDevice(settings.value(QStringLiteral("compute/cudaDevice"), 0).toInt());
        RemoteConfig rc;
        rc.host = toStd(settings.value(QStringLiteral("hpc/host"), QStringLiteral("localhost")).toString());
        rc.port = settings.value(QStringLiteral("hpc/port"), 7645).toInt();
        rc.token = toStd(secrets::read(QStringLiteral("hpc/token")));
        wb.setRemoteConfig(rc);
    }

} // namespace sirius::app
