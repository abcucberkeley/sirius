#include "qt/panels/params_panel.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <vector>

#include <QBoxLayout>
#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QGridLayout>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMouseEvent>
#include <QPainter>
#include <QPointer>
#include <QPushButton>
#include <QSlider>

#include <limits>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QToolButton>

#include <sirius/device.hpp>

#include "qt/qt_strings.hpp"
#include "qt/shortcuts.hpp"
#include "core/ops/builtin.hpp"
#include "qt/dialogs/model_hub_dialog.hpp"
#include "qt/theme.hpp"
#include "qt/widgets/controls.hpp"

namespace sirius::app {

    using widgets::CaptionLabel;
    using widgets::GlyphButton;
    using widgets::Rule;
    using widgets::SegmentedControl;

    namespace {

        // core's formatBytes for the number, an em dash for "not known yet".
        QString bytesText(std::size_t bytes) {
            return bytes == 0 ? QStringLiteral("—") : widgets::bytesText(bytes);
        }

        QString pixelTypeName(PixelType t) { return QString::fromLatin1(toString(t)); }

        // "label above input" field: 11 px neutral-600 label, then the editor.
        QWidget* field(const QString& label, QWidget* editor, QWidget* parent) {
            auto* w = new QWidget(parent);
            auto* l = new QVBoxLayout(w);
            l->setContentsMargins(0, 0, 0, 0);
            l->setSpacing(4);
            if (!label.isEmpty()) l->addWidget(widgets::label(label, 11, theme::kNeutral600, -1, w));
            l->addWidget(editor);
            return w;
        }

        bool isNumeric(ParamType t) {
            return t == ParamType::Int || t == ParamType::Double || t == ParamType::Channel;
        }

        // Einsum axis tiles: kept = outlined, reduced = accent-filled with
        // the reduction name underneath. Click toggles.
        class AxisTiles : public QWidget {
        public:
            explicit AxisTiles(QWidget* parent) : QWidget(parent) {
                setFixedHeight(46);
                setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
                setMouseTracking(true);
            }
            void setState(const QString& kept, const QString& reduction) {
                kept_ = kept;
                reduction_ = reduction;
                update();
            }
            QString kept() const { return kept_; }
            std::function<void(const QString&)> onChanged;

        protected:
            void paintEvent(QPaintEvent*) override {
                QPainter p(this);
                const int n = 5;
                const int w = (width() - 2 * (n - 1)) / n;
                for (int i = 0; i < n; ++i) {
                    const QChar ax = QLatin1Char("ctzyx"[i]);
                    const bool keep = kept_.contains(ax);
                    const QRect r(i * (w + 2), 0, i == n - 1 ? width() - i * (w + 2) : w, height());
                    p.setPen(QPen(keep ? (hover_ == i ? theme::kAccent : theme::kDivider) : theme::kAccent, 1.5));
                    p.setBrush(keep ? Qt::NoBrush : QBrush(theme::kAccent));
                    p.drawRect(QRectF(r).adjusted(0.75, 0.75, -0.75, -0.75));
                    p.setPen(keep ? theme::kText : theme::kBg);
                    p.setFont(theme::heading(16));
                    p.drawText(r.adjusted(0, 6, 0, -14), Qt::AlignHCenter | Qt::AlignTop, QString(ax));
                    QFont sf = theme::font(9);
                    sf.setLetterSpacing(QFont::PercentageSpacing, 108.0);
                    p.setFont(sf);
                    p.drawText(r.adjusted(0, 0, 0, -5), Qt::AlignHCenter | Qt::AlignBottom, captionCase(keep ? QStringLiteral("keep") : reduction_));
                }
            }
            void mousePressEvent(QMouseEvent* e) override {
                const int n = 5;
                const int w = (width() - 2 * (n - 1)) / n;
                const int i = std::clamp(e->pos().x() / (w + 2), 0, n - 1);
                const QChar ax = QLatin1Char("ctzyx"[i]);
                QString k;
                for (QChar c : QStringLiteral("ctzyx")) {
                    const bool on = kept_.contains(c);
                    if ((c == ax) != on) k += c;   // toggle the clicked axis
                }
                kept_ = k;
                update();
                if (onChanged) onChanged(kept_);
            }
            void mouseMoveEvent(QMouseEvent* e) override {
                const int n = 5;
                const int w = (width() - 2 * (n - 1)) / n;
                hover_ = std::clamp(e->pos().x() / (w + 2), 0, n - 1);
                update();
            }
            void leaveEvent(QEvent*) override {
                hover_ = -1;
                update();
            }

        private:
            QString kept_ = QStringLiteral("ctzyx");
            QString reduction_ = QStringLiteral("mean");
            int hover_ = -1;
        };

    } // namespace

    struct ParamsPanel::Impl {
        WorkbenchBridge& bridge;
        ParamsPanel* panel;

        CaptionLabel* kicker = nullptr;
        QLabel* state = nullptr;
        QLabel* name = nullptr;
        GlyphButton* help = nullptr;
        QScrollArea* scroll = nullptr;
        QWidget* body = nullptr;
        QVBoxLayout* bodyLayout = nullptr;
        SegmentedControl* backend = nullptr;
        SegmentedControl* cache = nullptr;
        QLabel* cacheSize = nullptr;
        QLabel* cacheNote = nullptr;
        QLabel* backendNote = nullptr;     // "no HPC implementation…" when it applies
        QPushButton* run = nullptr;
        QPushButton* view = nullptr;
        QPushButton* remove = nullptr;
        QLabel* validation = nullptr;

        int builtFor = -1;
        std::string builtVisibility;                 // step index the body was built for
        std::string builtKind;
        std::map<std::string, std::function<void(const ParamSet&)>> updaters;   // key -> refresh from params
        bool updating = false;

        Impl(WorkbenchBridge& b, ParamsPanel* p) : bridge(b), panel(p) {}

        int index() const { return bridge.wb().selectedIndex(); }
        const Step* step() const {
            const Pipeline& p = bridge.wb().pipeline();
            const int i = index();
            return i >= 0 && i < p.size() ? &p.at(i) : nullptr;
        }

        void setParam(const std::string& key, ParamValue v, bool merge) {
            if (updating) return;
            bridge.wb().setStepParam(index(), key, std::move(v), merge ? key : std::string());
        }

        // A file, colour or model dialog runs an event loop of its own, in
        // which a run finishing or another selection rebuilds this form --
        // deleting the editor that opened the dialog -- or points it at
        // another step. What the dialog returns applies only while the step
        // it was opened for is still the selected one, and reaches the
        // editor only if the editor is still there.
        StepId selectedId() const {
            const Step* st = step();
            return st ? st->id : 0;
        }

        // --- generic editors -----------------------------------------------------
        QWidget* editor(const ParamSpec& s, const ParamSet& params, const DatasetMeta& input, QWidget* parent) {
            const std::string key = s.key;
            switch (s.type) {
                case ParamType::Bool: {
                    auto* box = new QCheckBox(fromStd(s.label), parent);
                    box->setChecked(params.getBool(key));
                    box->setToolTip(fromStd(s.help));
                    box->setEnabled(!s.readOnly);
                    QObject::connect(box, &QCheckBox::toggled, panel, [this, key](bool on) { setParam(key, on, false); });
                    updaters[key] = [box, key](const ParamSet& p) {
                        QSignalBlocker b(box);
                        box->setChecked(p.getBool(key));
                    };
                    return box;
                }
                case ParamType::Int:
                case ParamType::Channel: {
                    if (s.type == ParamType::Channel) {
                        auto* combo = new QComboBox(parent);
                        for (std::size_t c = 0; c < input.channels.size(); ++c)
                            combo->addItem(fromStd(input.channels[c].shortName() + " " + input.channels[c].label));
                        if (combo->count() == 0) combo->addItem(QStringLiteral("ch 0"));
                        combo->setCurrentIndex(std::clamp(static_cast<int>(params.getInt(key)), 0, combo->count() - 1));
                        combo->setToolTip(fromStd(s.help));
                        QObject::connect(combo, qOverload<int>(&QComboBox::currentIndexChanged), panel,
                                         [this, key](int i) { setParam(key, static_cast<std::int64_t>(i), false); });
                        updaters[key] = [combo, key](const ParamSet& p) {
                            QSignalBlocker b(combo);
                            combo->setCurrentIndex(std::clamp(static_cast<int>(p.getInt(key)), 0, combo->count() - 1));
                        };
                        return combo;
                    }
                    auto* spin = new QSpinBox(parent);
                    const int lo = std::isfinite(s.min) ? static_cast<int>(s.min) : -1000000000;
                    const int hi = std::isfinite(s.max) ? static_cast<int>(s.max) : 1000000000;
                    spin->setRange(lo, hi);
                    if (s.step > 0) spin->setSingleStep(static_cast<int>(s.step));
                    if (!s.unit.empty()) spin->setSuffix(QStringLiteral(" ") + fromStd(s.unit));
                    spin->setValue(static_cast<int>(params.getInt(key)));
                    spin->setToolTip(fromStd(s.help));
                    spin->setReadOnly(s.readOnly);
                    spin->setKeyboardTracking(false);
                    widgets::useTabularNumbers(spin);
                    QObject::connect(spin, qOverload<int>(&QSpinBox::valueChanged), panel,
                                     [this, key](int v) { setParam(key, static_cast<std::int64_t>(v), true); });
                    updaters[key] = [spin, key](const ParamSet& p) {
                        QSignalBlocker b(spin);
                        spin->setValue(static_cast<int>(p.getInt(key)));
                    };
                    return spin;
                }
                case ParamType::Double: {
                    auto* spin = new QDoubleSpinBox(parent);
                    const double lo = std::isfinite(s.min) ? s.min : -1e12;
                    const double hi = std::isfinite(s.max) ? s.max : 1e12;
                    spin->setRange(lo, hi);
                    const double def = params.getDouble(key);
                    int decimals = s.decimals;
                    if (decimals < 0) {
                        const double mag = std::abs(def) > 0 ? std::abs(def) : (s.step > 0 ? s.step : 1.0);
                        decimals = mag >= 100 ? 1 : mag >= 1  ? 2
                                                : mag >= 0.01 ? 4
                                                              : 6;
                    }
                    spin->setDecimals(decimals);
                    widgets::useTabularNumbers(spin);
                    spin->setSingleStep(s.step > 0 ? s.step : std::pow(10.0, -std::max(decimals - 1, 0)));
                    if (!s.unit.empty()) spin->setSuffix(QStringLiteral(" ") + fromStd(s.unit));
                    spin->setValue(def);
                    spin->setToolTip(fromStd(s.help));
                    spin->setReadOnly(s.readOnly);
                    spin->setKeyboardTracking(false);
                    QObject::connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), panel,
                                     [this, key](double v) { setParam(key, v, true); });
                    updaters[key] = [spin, key](const ParamSet& p) {
                        QSignalBlocker b(spin);
                        spin->setValue(p.getDouble(key));
                    };
                    return spin;
                }
                case ParamType::Choice: {
                    auto* combo = new QComboBox(parent);
                    for (const std::string& c : s.choices) combo->addItem(fromStd(c));
                    combo->setCurrentText(fromStd(params.getString(key)));
                    combo->setToolTip(fromStd(s.help));
                    combo->setEnabled(!s.readOnly);
                    QObject::connect(combo, &QComboBox::currentTextChanged, panel,
                                     [this, key](const QString& t) { setParam(key, toStd(t), false); });
                    updaters[key] = [combo, key](const ParamSet& p) {
                        QSignalBlocker b(combo);
                        combo->setCurrentText(fromStd(p.getString(key)));
                    };
                    return combo;
                }
                case ParamType::Path: {
                    auto* row = new QWidget(parent);
                    auto* l = new QHBoxLayout(row);
                    l->setContentsMargins(0, 0, 0, 0);
                    l->setSpacing(6);
                    auto* edit = new QLineEdit(fromStd(params.getString(key)), row);
                    edit->setPlaceholderText(s.directory ? QStringLiteral("directory…") : QStringLiteral("file…"));
                    edit->setToolTip(fromStd(s.help));
                    edit->setReadOnly(s.readOnly);
                    auto* browse = new QPushButton(QStringLiteral("Browse"), row);
                    widgets::setButtonClass(browse, "secondary small");
                    browse->setEnabled(!s.readOnly);
                    l->addWidget(edit, 1);
                    l->addWidget(browse);
                    const bool dir = s.directory;
                    const QString filter = fromStd(s.fileFilter);
                    QObject::connect(edit, &QLineEdit::editingFinished, panel,
                                     [this, key, edit] { setParam(key, toStd(edit->text()), false); });
                    QObject::connect(browse, &QPushButton::clicked, panel, [this, key, edit, dir, filter] {
                        const QString start = edit->text().isEmpty() ? QString() : QFileInfo(edit->text()).absolutePath();
                        const QPointer<QLineEdit> editor(edit);
                        const StepId forStep = selectedId();
                        const QString path = dir ? QFileDialog::getExistingDirectory(panel, QStringLiteral("Choose directory"), start)
                                                 : QFileDialog::getOpenFileName(panel, QStringLiteral("Choose file"), start,
                                                                                filter.isEmpty() ? QStringLiteral("All files (*)") : filter);
                        if (path.isEmpty() || selectedId() != forStep) return;
                        if (editor) editor->setText(path);
                        setParam(key, toStd(path), false);
                    });
                    updaters[key] = [edit, key](const ParamSet& p) {
                        QSignalBlocker b(edit);
                        edit->setText(fromStd(p.getString(key)));
                    };
                    return row;
                }
                case ParamType::String:
                case ParamType::DoubleList:
                case ParamType::StringList:
                case ParamType::Axes: {
                    auto* edit = new QLineEdit(fromStd(toDisplayString(*params.find(key))), parent);
                    edit->setToolTip(fromStd(s.help));
                    edit->setReadOnly(s.readOnly);
                    if (s.type == ParamType::DoubleList) edit->setPlaceholderText(QStringLiteral("z, y, x"));
                    const ParamType type = s.type;
                    QObject::connect(edit, &QLineEdit::editingFinished, panel, [this, key, edit, type] {
                        const std::string text = toStd(edit->text());
                        if (type == ParamType::DoubleList) {
                            ParamSet tmp;
                            tmp.set("v", text);
                            setParam(key, tmp.getDoubleList("v"), false);
                        } else if (type == ParamType::StringList) {
                            ParamSet tmp;
                            tmp.set("v", text);
                            setParam(key, tmp.getStringList("v"), false);
                        } else {
                            setParam(key, text, false);
                        }
                    });
                    updaters[key] = [edit, key](const ParamSet& p) {
                        QSignalBlocker b(edit);
                        if (const ParamValue* v = p.find(key)) edit->setText(fromStd(toDisplayString(*v)));
                    };
                    return edit;
                }
            }
            return new QWidget(parent);
        }

        // Generic form: numeric fields in pairs, everything else full width.
        void buildGeneric(const std::vector<ParamSpec>& specs, const ParamSet& params, const DatasetMeta& input,
                          QVBoxLayout* into, bool includeAdvanced, const std::vector<std::string>& skip = {}) {
            QGridLayout* grid = nullptr;
            int col = 0;
            std::vector<const ParamSpec*> advanced;
            std::string group;
            for (const ParamSpec& s : specs) {
                if (std::find(skip.begin(), skip.end(), s.key) != skip.end()) continue;
                // a field the current mode ignores is not shown at all, not
                // even folded away under "More parameters"
                if (!s.visibleFor(params)) continue;
                if (s.advanced && !includeAdvanced) {
                    advanced.push_back(&s);
                    continue;
                }
                if (!s.group.empty() && s.group != group) {
                    group = s.group;
                    grid = nullptr;
                    into->addWidget(new Rule(2, Qt::Horizontal, body));
                    into->addWidget(new CaptionLabel(fromStd(group), body));
                }
                if (s.readOnly && (s.type == ParamType::String || isNumeric(s.type))) {
                    auto* row = new QWidget(body);
                    auto* rl = new QHBoxLayout(row);
                    rl->setContentsMargins(0, 0, 0, 0);
                    rl->addWidget(widgets::label(fromStd(s.label), 12, theme::kNeutral600, -1, row));
                    rl->addStretch(1);
                    const ParamValue* v = params.find(s.key);
                    rl->addWidget(widgets::label(v ? fromStd(toDisplayString(*v)) : QString(), 12, theme::kText, -1, row));
                    into->addWidget(row);
                    grid = nullptr;
                    continue;
                }
                if (isNumeric(s.type)) {
                    if (!grid) {
                        auto* host = new QWidget(body);
                        grid = new QGridLayout(host);
                        grid->setContentsMargins(0, 0, 0, 0);
                        grid->setHorizontalSpacing(10);
                        grid->setVerticalSpacing(12);
                        into->addWidget(host);
                        col = 0;
                    }
                    grid->addWidget(field(fromStd(s.label), editor(s, params, input, body), body), col / 2, col % 2);
                    ++col;
                    continue;
                }
                grid = nullptr;
                QWidget* ed = editor(s, params, input, body);
                if (s.type == ParamType::Bool) into->addWidget(ed);
                else into->addWidget(field(fromStd(s.label), ed, body));
            }
            if (!advanced.empty()) {
                auto* more = new QPushButton(QStringLiteral("More parameters…"), body);
                widgets::setButtonClass(more, "link");
                into->addWidget(more, 0, Qt::AlignLeft);
                auto* host = new QWidget(body);
                auto* hl = new QVBoxLayout(host);
                hl->setContentsMargins(0, 0, 0, 0);
                hl->setSpacing(12);
                std::vector<ParamSpec> adv;
                for (const ParamSpec* s : advanced) {
                    ParamSpec c = *s;
                    c.advanced = false;
                    adv.push_back(c);
                }
                // build eagerly (hidden) so updaters exist for every key
                buildGenericInto(adv, params, input, hl);
                host->hide();
                into->addWidget(host);
                QObject::connect(more, &QPushButton::clicked, panel, [host, more] {
                    host->setVisible(!host->isVisible());
                    more->setText(host->isVisible() ? QStringLiteral("Fewer parameters") : QStringLiteral("More parameters…"));
                });
            }
        }

        void buildGenericInto(const std::vector<ParamSpec>& specs, const ParamSet& params, const DatasetMeta& input,
                              QVBoxLayout* into) {
            buildGeneric(specs, params, input, into, true);
        }

        // --- kind decorations ------------------------------------------------------
        void factsTable(const std::vector<std::pair<QString, QString>>& rows, QVBoxLayout* into) {
            auto* host = new QWidget(body);
            auto* grid = new QGridLayout(host);
            grid->setContentsMargins(0, 0, 0, 0);
            grid->setHorizontalSpacing(0);
            grid->setVerticalSpacing(0);
            grid->setColumnMinimumWidth(0, 90);
            grid->setColumnStretch(1, 1);
            int r = 0;
            for (const auto& [k, v] : rows) {
                auto* kl = widgets::label(k, 12, theme::kNeutral600, -1, host);
                kl->setContentsMargins(0, 6, 0, 6);
                auto* vl = widgets::label(v, 12, theme::kText, -1, host);
                vl->setContentsMargins(0, 6, 0, 6);
                vl->setWordWrap(true);
                grid->addWidget(kl, r, 0);
                grid->addWidget(vl, r, 1);
                ++r;
                auto* rule = new Rule(1, Qt::Horizontal, host);
                grid->addWidget(rule, r, 0, 1, 2);
                ++r;
            }
            into->addWidget(new Rule(2, Qt::Horizontal, body));
            into->addWidget(host);
        }

        void channelList(const DatasetMeta& meta, QVBoxLayout* into) {
            if (meta.channels.empty()) return;
            auto* host = new QWidget(body);
            auto* l = new QVBoxLayout(host);
            l->setContentsMargins(0, 0, 0, 0);
            l->setSpacing(0);
            l->addWidget(widgets::label(QStringLiteral("Channels"), 11, theme::kNeutral600, -1, host));
            for (const ChannelInfo& ch : meta.channels) {
                auto* row = new QWidget(host);
                auto* rl = new QHBoxLayout(row);
                rl->setContentsMargins(0, 5, 0, 5);
                rl->setSpacing(10);
                rl->addWidget(widgets::colorChip(QColor(fromStd(ch.hexColor())), 10, 10, row));
                auto* nm = widgets::label(ch.wavelengthNm > 0 ? QString::number(static_cast<int>(std::lround(ch.wavelengthNm))) : QStringLiteral("—"),
                                          12, theme::kText, -1, row);
                nm->setFixedWidth(36);
                rl->addWidget(nm);
                rl->addWidget(widgets::label(fromStd(ch.label), 12, theme::kText, -1, row), 1);
                l->addWidget(row);
                l->addWidget(new Rule(1, Qt::Horizontal, host));
            }
            into->addWidget(host);
        }

        void buildLoad(const Step& step, const OpInfo& info, QVBoxLayout* into) {
            const Workbench& wb = bridge.wb();
            const DatasetMeta& ds = wb.dataset();
            std::vector<std::string> done;
            // the path field first, then facts, channels, then the rest
            for (const ParamSpec& s : info.params)
                if (s.type == ParamType::Path) {
                    into->addWidget(field(fromStd(s.label), editor(s, step.params, ds, body), body));
                    done.push_back(s.key);
                    break;
                }
            if (wb.hasDataset()) {
                std::vector<std::pair<QString, QString>> facts{
                    {QStringLiteral("Shape"), fromStd(ds.shapeString())},
                    {QStringLiteral("Acquisition"), ds.acquisition.empty() ? QStringLiteral("—") : fromStd(ds.acquisition)},
                    {QStringLiteral("Voxel"), fromStd(ds.voxelString())},
                    {QStringLiteral("Dtype"), pixelTypeName(ds.sourceType) + QStringLiteral(" · ") + bytesText(ds.bytesOnDisk)}};
                if (ds.hasTiles()) {
                    // grid extent from the tiles' grid indices (rows × columns, layers when > 1)
                    Index rows = 0, cols = 0, layers = 0;
                    for (const TileInfo& t : ds.tiles) {
                        layers = std::max(layers, t.gridIndex[0] + 1);
                        rows = std::max(rows, t.gridIndex[1] + 1);
                        cols = std::max(cols, t.gridIndex[2] + 1);
                    }
                    QString tiles = QString::number(ds.tiles.size());
                    if (rows * cols > 1) tiles += QStringLiteral(" · %1 × %2 grid").arg(rows).arg(cols);
                    if (layers > 1) tiles += QStringLiteral(" · %1 layers").arg(layers);
                    facts.emplace_back(QStringLiteral("Tiles"), tiles);
                }
                factsTable(facts, into);
                channelList(ds, into);
                // the tile chooser, bound to Load ▸ tile like the viewer toolbar's
                const bool hasTileParam = std::any_of(info.params.begin(), info.params.end(),
                                                      [](const ParamSpec& s) { return s.key == "tile"; });
                if (ds.hasTiles() && hasTileParam) {
                    const std::string key = "tile";
                    auto* combo = new QComboBox(body);
                    for (std::size_t i = 0; i < ds.tiles.size(); ++i)
                        combo->addItem(QStringLiteral("%1 · %2").arg(i + 1).arg(fromStd(ds.tiles[i].name)));
                    combo->setCurrentIndex(std::clamp(static_cast<int>(step.params.getInt(key)), 0, combo->count() - 1));
                    combo->setToolTip(QStringLiteral("Which tile of the multi-file dataset the pipeline reads; the viewed step is re-run on it"));
                    QObject::connect(combo, qOverload<int>(&QComboBox::currentIndexChanged), panel, [this, key](int i) {
                        setParam(key, static_cast<std::int64_t>(i), false);
                        if (!bridge.running()) bridge.startRun(bridge.wb().viewedIndex());
                    });
                    updaters[key] = [combo, key](const ParamSet& p) {
                        QSignalBlocker b(combo);
                        combo->setCurrentIndex(std::clamp(static_cast<int>(p.getInt(key)), 0, combo->count() - 1));
                    };
                    into->addWidget(field(QStringLiteral("Tile"), combo, body));
                    done.push_back(key);
                }
            } else {
                auto* hint = widgets::label(QStringLiteral("No dataset loaded. Choose a file above or use File ▸ Open dataset…"), 12,
                                            theme::kNeutral600, -1, body);
                hint->setWordWrap(true);
                into->addWidget(hint);
            }
            buildGeneric(info.params, step.params, ds, into, false, done);
        }

        void buildEinsum(const Step& step, const OpInfo& info, const DatasetMeta& input, QVBoxLayout* into) {
            const ParamSpec* axesSpec = nullptr;
            const ParamSpec* redSpec = nullptr;
            for (const ParamSpec& s : info.params) {
                if (!axesSpec && s.type == ParamType::Axes) axesSpec = &s;
                if (!redSpec && s.type == ParamType::Choice &&
                    std::find(s.choices.begin(), s.choices.end(), "mean") != s.choices.end())
                    redSpec = &s;
            }
            if (!axesSpec) {
                buildGeneric(info.params, step.params, input, into, false);
                return;
            }
            const std::string axesKey = axesSpec->key;
            const std::string redKey = redSpec ? redSpec->key : std::string();
            auto* tiles = new AxisTiles(body);
            auto* expr = new QLabel(body);
            expr->setFont(theme::mono(15));
            expr->setContentsMargins(12, 10, 12, 10);
            expr->setAutoFillBackground(true);
            QPalette ep = expr->palette();
            ep.setColor(QPalette::Window, theme::kSurface);
            expr->setPalette(ep);
            auto refreshExpr = [expr](const QString& kept) {
                QString out;
                for (QChar c : QStringLiteral("ctzyx"))
                    if (kept.contains(c)) out += c;
                expr->setText(QStringLiteral("ctzyx -> %1").arg(out.isEmpty() ? QStringLiteral("·") : out));
            };
            auto normalizeKept = [](const std::string& raw) {
                QString kept;
                for (QChar c : QStringLiteral("ctzyx"))
                    if (raw.find(c.toLatin1()) != std::string::npos) kept += c;
                return kept;
            };
            const QString kept0 = normalizeKept(step.params.getString(axesKey, "ctzyx"));
            const QString red0 = fromStd(redKey.empty() ? std::string("mean") : step.params.getString(redKey, "mean"));
            tiles->setState(kept0, red0);
            refreshExpr(kept0);
            tiles->onChanged = [this, axesKey, refreshExpr](const QString& kept) {
                refreshExpr(kept);
                setParam(axesKey, toStd(kept), false);
            };
            into->addWidget(field(fromStd(axesSpec->label.empty() ? std::string("Axes — click to keep or reduce") : axesSpec->label), tiles, body));
            updaters[axesKey] = [tiles, refreshExpr, normalizeKept, axesKey, redKey](const ParamSet& p) {
                const QString kept = normalizeKept(p.getString(axesKey, "ctzyx"));
                tiles->setState(kept, fromStd(redKey.empty() ? std::string("mean") : p.getString(redKey, "mean")));
                refreshExpr(kept);
            };
            std::vector<std::string> done{axesKey};
            if (redSpec) {
                QStringList opts;
                for (const std::string& c : redSpec->choices) opts << fromStd(c);
                auto* seg = new SegmentedControl(opts, body);
                seg->setTileMode(true);
                seg->setCurrentText(red0);
                QObject::connect(seg, &SegmentedControl::changed, panel, [this, redKey, seg, tiles](int) {
                    tiles->setState(tiles->kept(), seg->currentText());
                    setParam(redKey, toStd(seg->currentText()), false);
                });
                updaters[redKey] = [seg, redKey](const ParamSet& p) { seg->setCurrentText(fromStd(p.getString(redKey))); };
                into->addWidget(field(fromStd(redSpec->label), seg, body));
                done.push_back(redKey);
            }
            into->addWidget(field(QStringLiteral("Expression"), expr, body));
            buildGeneric(info.params, step.params, input, into, false, done);
        }

        void buildSim(const Step& step, const OpInfo& info, const DatasetMeta& input, QVBoxLayout* into) {
            std::vector<std::string> done;
            // first Choice = mode -> segmented control
            for (const ParamSpec& s : info.params) {
                if (s.type == ParamType::Choice && !s.advanced) {
                    bool modeLike = false;
                    for (const std::string& c : s.choices)
                        if (c.find("stimate") != std::string::npos || c.find("anual") != std::string::npos) modeLike = true;
                    if (!modeLike) continue;
                    QStringList opts;
                    for (const std::string& c : s.choices) opts << fromStd(c);
                    auto* seg = new SegmentedControl(opts, body);
                    seg->setCurrentText(fromStd(step.params.getString(s.key)));
                    const std::string key = s.key;
                    QObject::connect(seg, &SegmentedControl::changed, panel,
                                     [this, key, seg](int) { setParam(key, toStd(seg->currentText()), false); });
                    updaters[key] = [seg, key](const ParamSet& p) { seg->setCurrentText(fromStd(p.getString(key))); };
                    into->addWidget(seg, 0, Qt::AlignLeft);
                    done.push_back(key);
                    break;
                }
            }
            buildGeneric(info.params, step.params, input, into, false, done);
            const Diagnostics d = bridge.wb().selectedDiagnostics();
            if (!d.warnings.empty()) {
                into->addWidget(new Rule(2, Qt::Horizontal, body));
                for (const std::string& w : d.warnings) {
                    auto* note = widgets::label(fromStd(w), 11, theme::kNeutral600, -1, body);
                    note->setWordWrap(true);
                    into->addWidget(note);
                }
            }
        }

        // The foundation step's model is a bundle, and a bundle is picked from
        // the registry rather than found on disk: what distinguishes two .ltb
        // files is inside them (task, voxel size, the thresholds they were
        // validated at), and a file dialog shows none of it.
        void buildFoundation(const Step& step, const OpInfo& info, const DatasetMeta& input, QVBoxLayout* into) {
            std::vector<std::string> done;
            for (const ParamSpec& s : info.params) {
                if (s.type != ParamType::Path || s.advanced) continue;
                auto* row = new QWidget(body);
                auto* rl = new QHBoxLayout(row);
                rl->setContentsMargins(0, 0, 0, 0);
                rl->setSpacing(6);
                QWidget* pathEditor = editor(s, step.params, input, row);
                auto* pick = new QPushButton(QStringLiteral("Bundles…"), row);
                widgets::setButtonClass(pick, "secondary small");
                pick->setToolTip(QStringLiteral("Choose from the bundles in the registry, with what each was trained "
                                                "for and calibrated at"));
                rl->addWidget(pathEditor, 1);
                rl->addWidget(pick);
                const std::string key = s.key;
                QObject::connect(pick, &QPushButton::clicked, panel, [this, key, pathEditor] {
                    ModelHubDialog dialog(bridge, panel);
                    dialog.showBundles();
                    if (dialog.exec() != QDialog::Accepted || dialog.chosenModel().isEmpty()) return;
                    const QString chosen = dialog.chosenModel();
                    if (auto* edit = pathEditor->findChild<QLineEdit*>()) edit->setText(chosen);
                    setParam(key, toStd(chosen), false);
                });
                into->addWidget(field(fromStd(s.label), row, body));
                done.push_back(s.key);
                break;
            }
            buildGeneric(info.params, step.params, input, into, false, done);
        }

        void buildSeg(const Step& step, const OpInfo& info, const DatasetMeta& input, QVBoxLayout* into) {
            std::vector<std::string> done;
            for (const ParamSpec& s : info.params)
                if (s.type == ParamType::Path && !s.advanced) {
                    // the path editor (file + Browse) plus "Hub…": Hugging Face
                    // downloads and the local model cache in one dialog
                    auto* row = new QWidget(body);
                    auto* rl = new QHBoxLayout(row);
                    rl->setContentsMargins(0, 0, 0, 0);
                    rl->setSpacing(6);
                    QWidget* pathEditor = editor(s, step.params, input, row);
                    auto* hub = new QPushButton(QStringLiteral("Hub…"), row);
                    widgets::setButtonClass(hub, "secondary small");
                    hub->setToolTip(QStringLiteral("Search Hugging Face, pick a Cellpose / micro-SAM model, or a cached file"));
                    rl->addWidget(pathEditor, 1);
                    rl->addWidget(hub);
                    const std::string key = s.key;
                    QObject::connect(hub, &QPushButton::clicked, panel, [this, key, pathEditor] {
                        const QPointer<QWidget> editor(pathEditor);
                        const StepId forStep = selectedId();
                        ModelHubDialog dialog(bridge, panel);
                        if (dialog.exec() != QDialog::Accepted || dialog.chosenModel().isEmpty() || selectedId() != forStep) return;
                        const QString chosen = dialog.chosenModel();
                        if (auto* edit = editor ? editor->findChild<QLineEdit*>() : nullptr) edit->setText(chosen);
                        setParam(key, toStd(chosen), false);
                    });
                    into->addWidget(field(fromStd(s.label), row, body));
                    done.push_back(s.key);
                    break;
                }
            const Diagnostics d = bridge.wb().selectedDiagnostics();
            QString facts;
            for (const DiagnosticFact& f : d.facts) {
                if (!facts.isEmpty()) facts += QStringLiteral(" · ");
                facts += f.key.empty() ? fromStd(f.value) : fromStd(f.key + " " + f.value);
            }
            if (!facts.isEmpty()) {
                auto* l = widgets::label(facts, 11, theme::kNeutral600, -1, body);
                l->setWordWrap(true);
                into->addWidget(l);
            }
            buildGeneric(info.params, step.params, input, into, false, done);
            auto* row = new QWidget(body);
            auto* rl = new QHBoxLayout(row);
            rl->setContentsMargins(0, 10, 0, 0);
            rl->addWidget(widgets::label(QStringLiteral("Label opacity"), 12, theme::kNeutral600, -1, row));
            rl->addStretch(1);
            const int pct = static_cast<int>(std::lround(bridge.wb().viewState().labelOpacity * 100.0));
            rl->addWidget(widgets::label(QStringLiteral("%1 %").arg(pct), 12, theme::kText, -1, row));
            into->addWidget(new Rule(2, Qt::Horizontal, body));
            into->addWidget(row);
        }

        void buildMerge(const Step& step, const OpInfo& info, const DatasetMeta& input, QVBoxLayout* into) {
            // per-channel rows with a colour chip; a StringList spec (if any)
            // receives the hex colours, otherwise the chips are informative.
            const ParamSpec* colorSpec = nullptr;
            for (const ParamSpec& s : info.params)
                if (s.type == ParamType::StringList) {
                    colorSpec = &s;
                    break;
                }
            std::vector<std::string> colors = colorSpec ? step.params.getStringList(colorSpec->key) : std::vector<std::string>{};
            auto* host = new QWidget(body);
            auto* hl = new QVBoxLayout(host);
            hl->setContentsMargins(0, 0, 0, 0);
            hl->setSpacing(0);
            for (std::size_t c = 0; c < input.channels.size(); ++c) {
                const ChannelInfo& ch = input.channels[c];
                const QString hex = c < colors.size() && !colors[c].empty() ? fromStd(colors[c]) : fromStd(ch.hexColor());
                auto* row = new QWidget(host);
                auto* grid = new QGridLayout(row);
                grid->setContentsMargins(0, 6, 0, 6);
                grid->setHorizontalSpacing(10);
                grid->setColumnMinimumWidth(0, 44);
                grid->setColumnStretch(1, 1);
                grid->setColumnMinimumWidth(2, 90);
                grid->addWidget(widgets::label(ch.wavelengthNm > 0 ? QString::number(static_cast<int>(std::lround(ch.wavelengthNm))) : QStringLiteral("—"),
                                               12, theme::kText, -1, row),
                                0, 0);
                grid->addWidget(widgets::label(fromStd(ch.label), 12, theme::kText, -1, row), 0, 1);
                auto* chips = new QWidget(row);
                auto* cl = new QHBoxLayout(chips);
                cl->setContentsMargins(0, 0, 0, 0);
                cl->setSpacing(2);
                QWidget* chip = widgets::colorChip(QColor(hex), 44, 22, chips);
                auto* pick = new GlyphButton(QStringLiteral("…"), 22, chips);
                pick->setSize(44, 22);
                pick->setToolTip(QStringLiteral("Choose display colour"));
                cl->addWidget(chip);
                cl->addWidget(pick);
                grid->addWidget(chips, 0, 2);
                const std::size_t ci = c;
                const std::string colorKey = colorSpec ? colorSpec->key : std::string();
                std::vector<std::string> defaults;
                for (const ChannelInfo& other : input.channels) defaults.push_back(other.hexColor());
                QObject::connect(pick, &QAbstractButton::clicked, panel, [this, chip, ci, colorKey, hex, defaults] {
                    const QPointer<QWidget> swatch(chip);
                    const StepId forStep = selectedId();
                    const QColor chosen = QColorDialog::getColor(QColor(hex), panel, QStringLiteral("Channel colour"));
                    if (!chosen.isValid() || selectedId() != forStep) return;
                    if (swatch) widgets::setChipColor(swatch, chosen);
                    if (colorKey.empty()) return;
                    const Step* st = this->step();
                    if (!st) return;
                    std::vector<std::string> cols = st->params.getStringList(colorKey);
                    cols.resize(std::max(cols.size(), defaults.size()));
                    for (std::size_t k = 0; k < cols.size(); ++k)
                        if (cols[k].empty() && k < defaults.size()) cols[k] = defaults[k];
                    cols[ci] = toStd(chosen.name(QColor::HexRgb));
                    setParam(colorKey, cols, false);
                });
                hl->addWidget(row);
                hl->addWidget(new Rule(1, Qt::Horizontal, host));
            }
            into->addWidget(host);
            std::vector<std::string> done;
            if (colorSpec) done.push_back(colorSpec->key);
            buildGeneric(info.params, step.params, input, into, false, done);
        }

        // Contrast: window mode, percentiles or min / max sliders over the
        // input's range, gamma, Auto / Reset. Every edit is a parameter
        // change, so the viewer's live preview follows it and it is undoable.
        void buildContrast(const Step& step, const OpInfo& info, const DatasetMeta& input, QVBoxLayout* into) {
            std::vector<std::string> done{"min", "max", "gamma"};
            auto specOf = [&](const char* key) -> const ParamSpec* {
                for (const ParamSpec& sp : info.params)
                    if (sp.key == key) return &sp;
                return nullptr;
            };
            // the input's intensity range, for the slider extents
            double dataMin = 0.0, dataMax = 1.0;
            std::shared_ptr<const StepOutput> upstream = bridge.wb().upstreamOutput(index());
            if (upstream) {
                const StepInput in = upstream->asInput();
                float mn = std::numeric_limits<float>::infinity(), mx = -mn;
                for (Index c = 0; c < in.meta.dims.c; ++c) {
                    const ContrastWindow w = contrastWindow(in, step.params, c, 8, true);
                    mn = std::min(mn, w.dataMin);
                    mx = std::max(mx, w.dataMax);
                }
                if (mn < mx) {
                    dataMin = mn;
                    dataMax = mx;
                }
            }
            const double span = dataMax - dataMin;
            const int decimals = span >= 100.0 ? 1 : span >= 10.0 ? 2
                                                 : span >= 1.0    ? 3
                                                                  : 4;

            // manual min / max: slider + spin box over the data range
            auto* manualBox = new QWidget(body);   // min / max sliders over the data range
            auto* mv = new QVBoxLayout(manualBox);
            mv->setContentsMargins(0, 0, 0, 0);
            mv->setSpacing(10);
            auto sliderRow = [&](const char* key, const QString& label, double lo, double hi, int dec, double value) {
                auto* row = new QWidget(manualBox);
                auto* h = new QHBoxLayout(row);
                h->setContentsMargins(0, 0, 0, 0);
                h->setSpacing(8);
                auto* slider = new QSlider(Qt::Horizontal, row);
                slider->setRange(0, 1000);
                auto* spin = new QDoubleSpinBox(row);
                spin->setRange(-1e12, 1e12);   // the slider spans the data; typed values are not clamped
                spin->setDecimals(dec);
                spin->setSingleStep((hi - lo) / 200.0);
                spin->setValue(value);
                spin->setFixedWidth(96);
                widgets::useTabularNumbers(spin);
                slider->setValue(static_cast<int>(std::lround((value - lo) / (hi - lo) * 1000.0)));
                h->addWidget(slider, 1);
                h->addWidget(spin);
                const std::string k = key;
                auto commit = [this, k](double value, bool merge) {
                    const Step* st = this->step();
                    if (!st) return;
                    ParamSet np = st->params;
                    if (!(np.getDouble("max", 0.0) > np.getDouble("min", 0.0)))   // leave automatic: pin the other bound
                        if (auto up = bridge.wb().upstreamOutput(index())) {
                            const ContrastWindow eff = contrastWindow(up->asInput(), np, 0, 8);
                            np.set("min", static_cast<double>(eff.lo));
                            np.set("max", static_cast<double>(eff.hi));
                        }
                    np.set(k, value);
                    bridge.wb().setStepParams(index(), np, "Step " + Step::number(index()) + " · " + (k == "min" ? "Min" : "Max"),
                                              merge ? k : std::string());
                };
                QObject::connect(slider, &QSlider::valueChanged, panel, [this, lo, hi, spin, commit](int v) {
                    if (updating) return;
                    const double value = lo + (hi - lo) * v / 1000.0;
                    QSignalBlocker b(spin);
                    spin->setValue(value);
                    commit(value, true);   // one undo entry per drag
                });
                QObject::connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), panel, [this, lo, hi, slider, commit](double v) {
                    if (updating) return;
                    QSignalBlocker b(slider);
                    slider->setValue(static_cast<int>(std::lround(std::clamp((v - lo) / (hi - lo), 0.0, 1.0) * 1000.0)));
                    commit(v, false);
                });
                updaters[k] = [this, slider, spin, k, lo, hi](const ParamSet& p) {
                    double v = p.getDouble(k, lo);
                    if (!(p.getDouble("max", 0.0) > p.getDouble("min", 0.0)))   // automatic: show the resolved window
                        if (auto up = bridge.wb().upstreamOutput(index())) {
                            const ContrastWindow eff = contrastWindow(up->asInput(), p, 0, 8);
                            v = k == "min" ? eff.lo : eff.hi;
                        }
                    QSignalBlocker b1(slider), b2(spin);
                    spin->setValue(v);
                    slider->setValue(static_cast<int>(std::lround(std::clamp((v - lo) / (hi - lo), 0.0, 1.0) * 1000.0)));
                };
                mv->addWidget(field(label, row, manualBox));
            };
            // an empty window (the default) is automatic: show what it resolves to
            double curMin = step.params.getDouble("min", 0.0), curMax = step.params.getDouble("max", 0.0);
            if (!(curMax > curMin) && upstream) {
                const ContrastWindow eff = contrastWindow(upstream->asInput(), step.params, 0, 8);
                curMin = eff.lo;
                curMax = eff.hi;
            }
            sliderRow("min", QStringLiteral("Min"), dataMin, dataMax, decimals, curMin);
            sliderRow("max", QStringLiteral("Max"), dataMin, dataMax, decimals, curMax);
            into->addWidget(manualBox);

            // gamma: slider (0.1 .. 5) with the generic spin box
            if (const ParamSpec* g = specOf("gamma")) {
                auto* row = new QWidget(body);
                auto* h = new QHBoxLayout(row);
                h->setContentsMargins(0, 0, 0, 0);
                h->setSpacing(8);
                auto* slider = new QSlider(Qt::Horizontal, row);
                slider->setRange(10, 500);   // hundredths
                slider->setValue(static_cast<int>(std::lround(step.params.getDouble("gamma", 1.0) * 100.0)));
                QWidget* spin = editor(*g, step.params, input, row);
                spin->setFixedWidth(96);
                h->addWidget(slider, 1);
                h->addWidget(spin);
                QObject::connect(slider, &QSlider::valueChanged, panel, [this](int v) {
                    if (updating) return;
                    setParam("gamma", v / 100.0, true);
                });
                updaters["gamma#slider"] = [slider](const ParamSet& p) {
                    QSignalBlocker b(slider);
                    slider->setValue(static_cast<int>(std::lround(p.getDouble("gamma", 1.0) * 100.0)));
                };
                into->addWidget(field(fromStd(g->label), row, body));
            }

            // Auto / Reset
            auto* buttons = new QWidget(body);
            auto* bh = new QHBoxLayout(buttons);
            bh->setContentsMargins(0, 0, 0, 0);
            bh->setSpacing(8);
            auto* autoBtn = new QPushButton(QStringLiteral("Auto"), buttons);
            auto* resetBtn = new QPushButton(QStringLiteral("Reset"), buttons);
            widgets::setButtonClass(autoBtn, "secondary small");
            widgets::setButtonClass(resetBtn, "ghost small");
            autoBtn->setToolTip(QStringLiteral("Min / max on the input's percentiles (see More parameters)"));
            resetBtn->setToolTip(QStringLiteral("Min / max over the input's full range, gamma 1"));
            QObject::connect(autoBtn, &QPushButton::clicked, panel, [this] {
                const Step* st = this->step();
                auto up = bridge.wb().upstreamOutput(index());
                if (st && up) bridge.wb().setStepParams(index(), contrastAutoParams(st->params, up->asInput()), "Auto contrast");
            });
            QObject::connect(resetBtn, &QPushButton::clicked, panel, [this] {
                const Step* st = this->step();
                auto up = bridge.wb().upstreamOutput(index());
                if (st && up) bridge.wb().setStepParams(index(), contrastResetParams(st->params, up->asInput()), "Reset contrast");
            });
            bh->addWidget(autoBtn);
            bh->addWidget(resetBtn);
            bh->addStretch(1);
            into->addWidget(buttons);

            buildGeneric(info.params, step.params, input, into, false, done);
        }

        // --- rebuild --------------------------------------------------------------
        // The values every visibility rule of this step depends on. The panel
        // is rebuilt when one of them moves, since that is what decides which
        // fields exist at all; refreshing the values alone would leave the old
        // set on screen.
        std::string visibilitySignature(const OpInfo& info, const ParamSet& params) const {
            std::string out;
            for (const ParamSpec& s : info.params)
                for (const ParamSpec::Visibility& rule : s.visibility)
                    if (const ParamValue* v = params.find(rule.key)) out += rule.key + "=" + toDisplayString(*v) + ";";
            return out;
        }

        // Offered above the fields by any operation that has them. Choosing
        // one writes its values into the step -- an ordinary undoable
        // parameter change -- and the control returns to its caption, because
        // what a step holds afterwards is a set of values and not a mode.
        void buildPresets(const Step& step, const OpInfo& info, QVBoxLayout* into) {
            if (info.presets.empty()) return;
            auto* combo = new QComboBox(body);
            combo->addItem(QStringLiteral("Start from…"));
            for (const ParamPreset& preset : info.presets) {
                combo->addItem(fromStd(preset.name), fromStd(preset.name));
                combo->setItemData(combo->count() - 1, fromStd(preset.summary), Qt::ToolTipRole);
            }
            combo->setToolTip(QStringLiteral("Fill the fields below for a kind of structure; every one stays editable"));
            const int index = builtFor;
            connect(combo, qOverload<int>(&QComboBox::currentIndexChanged), panel, [this, combo, index](int row) {
                if (updating || row <= 0) return;
                const QString name = combo->itemData(row).toString();
                QSignalBlocker block(combo);
                combo->setCurrentIndex(0);
                bridge.wb().applyPreset(index, toStd(name));
            });
            into->addWidget(field(QStringLiteral("Preset"), combo, body));
            (void)step;
        }

        void rebuild() {
            updaters.clear();
            QWidget* old = scroll->takeWidget();
            if (old) old->deleteLater();
            body = new QWidget(scroll);
            body->setObjectName(QStringLiteral("Panel"));
            bodyLayout = new QVBoxLayout(body);
            bodyLayout->setContentsMargins(18, 0, 18, 16);
            bodyLayout->setSpacing(12);
            const Workbench& wb = bridge.wb();
            const Step* st = step();
            builtFor = index();
            if (!st) {
                bodyLayout->addWidget(widgets::label(QStringLiteral("No step selected."), 12, theme::kNeutral600, -1, body));
                bodyLayout->addStretch(1);
                scroll->setWidget(body);
                return;
            }
            builtKind = st->kind;
            const OpInfo& info = st->op().info();
            builtVisibility = visibilitySignature(info, st->params);
            const DatasetMeta input = wb.inputMetaOf(builtFor);
            buildPresets(*st, info, bodyLayout);
            if (st->kind == "load") buildLoad(*st, info, bodyLayout);
            else if (st->kind == "einsum") buildEinsum(*st, info, input, bodyLayout);
            else if (st->kind == "sim") buildSim(*st, info, input, bodyLayout);
            else if (st->kind == "seg") buildSeg(*st, info, input, bodyLayout);
            else if (st->kind == "foundation") buildFoundation(*st, info, input, bodyLayout);
            else if (st->kind == "merge") buildMerge(*st, info, input, bodyLayout);
            else if (st->kind == "contrast") buildContrast(*st, info, input, bodyLayout);
            else buildGeneric(info.params, st->params, input, bodyLayout, false);
            validation = widgets::label(QString(), 11, theme::kAccentText, -1, body);
            validation->setWordWrap(true);
            validation->hide();
            bodyLayout->addWidget(validation);
            bodyLayout->addStretch(1);
            scroll->setWidget(body);
            refreshValues();
        }

        void refreshValues() {
            const Step* st = step();
            if (!st) return;
            updating = true;
            for (auto& [key, fn] : updaters) fn(st->params);
            updating = false;
            const Validation v = bridge.wb().stepValidation(index());
            QString text;
            for (const std::string& e : v.errors) text += (text.isEmpty() ? "" : "\n") + fromStd(e);
            for (const std::string& w : v.warnings) text += (text.isEmpty() ? "" : "\n") + fromStd(w);
            if (validation) {
                validation->setText(text);
                validation->setVisible(!text.isEmpty());
                QPalette p = validation->palette();
                p.setColor(QPalette::WindowText, v.ok() ? theme::kNeutral600 : theme::kAccentText);
                validation->setPalette(p);
            }
        }

        void refreshHeader() {
            const Workbench& wb = bridge.wb();
            const Step* st = step();
            if (!st) {
                kicker->setText(QStringLiteral("Step"));
                state->clear();
                name->setText(QStringLiteral("—"));
                run->setEnabled(false);
                view->setEnabled(false);
                remove->hide();
                cacheSize->clear();
                backendNote->hide();
                return;
            }
            const int i = index();
            kicker->setText(QStringLiteral("Step %1 · %2").arg(fromStd(Step::number(i)), fromStd(st->op().info().kindLabel)));
            state->setText(st->enabled ? QStringLiteral("enabled") : QStringLiteral("skipped"));
            QPalette sp = state->palette();
            sp.setColor(QPalette::WindowText, st->enabled ? theme::kAccentText : theme::kNeutral600);
            state->setPalette(sp);
            name->setText(fromStd(st->name));
            name->setToolTip(QStringLiteral("Double-click to rename"));
            const bool running = bridge.running();
            const bool busy = running || bridge.taskRunning();
            // Every parameter edit is refused while a run holds the pipeline
            // (Workbench::canEdit), so the whole form goes with it rather
            // than accepting values that are dropped.
            const bool editable = wb.canEdit() && !bridge.taskRunning();
            scroll->setEnabled(editable);
            cache->setEnabled(editable);
            run->setEnabled(!busy && wb.hasDataset());
            view->setEnabled(true);
            remove->setVisible(!st->pinned);
            remove->setEnabled(editable);
            const QString frozen = QStringLiteral("Not while a run or load is in progress — cancel it (Esc) or wait");
            scroll->setToolTip(editable ? QString() : frozen);
            cache->setToolTip(editable ? QString() : frozen);
            remove->setToolTip(editable ? QString() : frozen);
            // backend / cache
            backend->setCurrentIndex(static_cast<int>(wb.backend()));
            backend->setOptionEnabled(0, cudaAvailable());
            backend->setOptionToolTip(0, cudaAvailable() ? QStringLiteral("Run on the selected CUDA device")
                                                         : QStringLiteral("No CUDA device is available in this build / machine"));
            backend->setOptionToolTip(2, QStringLiteral("Run on the remote worker (Preferences ▸ HPC)"));
            // Only operations the Python worker implements (OpInfo::remoteCapable)
            // are handed to the remote worker; the C++ ones run here whatever
            // the backend says, so the panel says which one this is.
            const bool remote = st->op().info().remoteCapable;
            backendNote->setText(wb.backend() != Backend::Hpc
                                     ? QString()
                                     : (remote ? QStringLiteral("Runs on the HPC worker.")
                                               : QStringLiteral("%1 has no HPC implementation: this step runs on this "
                                                                "machine even with the HPC backend selected.")
                                                     .arg(fromStd(st->op().info().name))));
            backendNote->setVisible(!backendNote->text().isEmpty());
            cache->setCurrentIndex(static_cast<int>(st->cache));
            cacheSize->setText(QStringLiteral("≈ %1").arg(bytesText(wb.estimatedBytesOf(i))));
            static const char* notes[] = {"Fastest scrubbing; evicted first when GPU/RAM fills.",
                                          "Survives restarts; written to the zarr scratch directory. Best for slow steps like reconstruction.",
                                          "Nothing stored; recomputed from the previous step on demand. Good for cheap steps."};
            cacheNote->setText(QString::fromUtf8(notes[static_cast<int>(st->cache)]));
        }

        void refresh() {
            refreshHeader();
            const Step* st = step();
            const bool sameShape = st && builtFor == index() && builtKind == st->kind &&
                                   builtVisibility == visibilitySignature(st->op().info(), st->params);
            if (!sameShape) rebuild();
            else refreshValues();
        }
    };

    ParamsPanel::ParamsPanel(WorkbenchBridge& bridge, QWidget* parent)
        : QWidget(parent), impl_(std::make_unique<Impl>(bridge, this)) {
        setObjectName(QStringLiteral("Panel"));
        // 320 px is what it asks for; fields shrink, so the dock may be
        // dragged narrower (see OpsPanel).
        setMinimumWidth(200);
        setAccessibleName(QStringLiteral("Parameters"));
        auto* root = new QVBoxLayout(this);
        root->setContentsMargins(0, 0, 0, 0);
        root->setSpacing(0);

        // header
        auto* header = new QWidget(this);
        auto* hl = new QVBoxLayout(header);
        hl->setContentsMargins(18, 14, 18, 10);
        hl->setSpacing(4);
        auto* kickRow = new QHBoxLayout();
        kickRow->setContentsMargins(0, 0, 0, 0);
        impl_->kicker = new CaptionLabel(QStringLiteral("Step"), header);
        // The accent at 10 / 11 px is 3.8:1 on this background; the darkened
        // one (theme::kAccentText) is the same red at 5.2:1.
        impl_->kicker->setColor(theme::kAccentText);
        impl_->state = widgets::label(QString(), 11, theme::kAccentText, -1, header);
        kickRow->addWidget(impl_->kicker);
        kickRow->addStretch(1);
        kickRow->addWidget(impl_->state);
        hl->addLayout(kickRow);
        auto* nameRow = new QHBoxLayout();
        nameRow->setContentsMargins(0, 0, 0, 0);
        nameRow->setSpacing(10);
        impl_->name = widgets::heading(QStringLiteral("—"), theme::kH4Px, header);
        impl_->name->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
        impl_->name->installEventFilter(this);
        impl_->help = new GlyphButton(widgets::Icon::Help, 24, header);
        impl_->help->setGlyphPx(13);
        impl_->help->setToolTip(withShortcut(QStringLiteral("Explain this step"), keys::helpForStep()));
        impl_->help->setAccessibleName(QStringLiteral("Help for this step"));
        nameRow->addWidget(impl_->name, 1);
        nameRow->addWidget(impl_->help);
        hl->addLayout(nameRow);
        root->addWidget(header);

        // body
        impl_->scroll = new QScrollArea(this);
        impl_->scroll->setWidgetResizable(true);
        impl_->scroll->setFrameShape(QFrame::NoFrame);
        impl_->scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        root->addWidget(impl_->scroll, 1);

        // backend
        auto* sections = new QWidget(this);
        auto* sl = new QVBoxLayout(sections);
        sl->setContentsMargins(18, 0, 18, 0);
        sl->setSpacing(0);
        sl->addWidget(new Rule(2, Qt::Horizontal, sections));
        auto* backendBox = new QWidget(sections);
        auto* bl = new QVBoxLayout(backendBox);
        bl->setContentsMargins(0, 16, 0, 16);
        bl->setSpacing(8);
        bl->addWidget(new CaptionLabel(QStringLiteral("Backend"), backendBox));
        impl_->backend = new SegmentedControl({QStringLiteral("CUDA"), QStringLiteral("CPU"), QStringLiteral("HPC")}, backendBox);
        impl_->backend->setTileMode(true);
        bl->addWidget(impl_->backend);
        impl_->backendNote = widgets::label(QString(), 12, theme::kNeutral600, -1, backendBox);
        impl_->backendNote->setWordWrap(true);
        impl_->backendNote->hide();
        bl->addWidget(impl_->backendNote);
        sl->addWidget(backendBox);
        sl->addWidget(new Rule(2, Qt::Horizontal, sections));
        auto* cacheBox = new QWidget(sections);
        auto* cl = new QVBoxLayout(cacheBox);
        cl->setContentsMargins(0, 16, 0, 16);
        cl->setSpacing(8);
        auto* cacheHead = new QHBoxLayout();
        cacheHead->setContentsMargins(0, 0, 0, 0);
        cacheHead->addWidget(new CaptionLabel(QStringLiteral("Cache output"), cacheBox));
        cacheHead->addStretch(1);
        impl_->cacheSize = widgets::label(QString(), 12, theme::kNeutral700, -1, cacheBox);
        cacheHead->addWidget(impl_->cacheSize);
        cl->addLayout(cacheHead);
        impl_->cache = new SegmentedControl({QStringLiteral("Memory"), QStringLiteral("Disk"), QStringLiteral("Recompute")}, cacheBox);
        impl_->cache->setTileMode(true);
        impl_->cache->setOptionToolTip(0, QStringLiteral("Cached in GPU/RAM"));
        impl_->cache->setOptionToolTip(1, QStringLiteral("Cached on disk (zarr scratch)"));
        impl_->cache->setOptionToolTip(2, QStringLiteral("Recomputed on demand"));
        cl->addWidget(impl_->cache);
        impl_->cacheNote = widgets::label(QString(), 12, theme::kNeutral600, -1, cacheBox);
        impl_->cacheNote->setWordWrap(true);
        cl->addWidget(impl_->cacheNote);
        sl->addWidget(cacheBox);
        root->addWidget(sections);

        // footer
        root->addWidget(new Rule(2, Qt::Horizontal, this));
        auto* footer = new QWidget(this);
        auto* fl = new QHBoxLayout(footer);
        fl->setContentsMargins(18, 14, 18, 14);
        fl->setSpacing(8);
        impl_->run = new QPushButton(QStringLiteral("Run step"), footer);
        widgets::setButtonClass(impl_->run, "primary");
        impl_->run->setToolTip(withShortcut(QStringLiteral("Run this step; its input has to be computed already"),
                                            keys::runSelected()));
        impl_->view = new QPushButton(QStringLiteral("View"), footer);
        widgets::setButtonClass(impl_->view, "secondary");
        impl_->view->setToolTip(QStringLiteral("Show this step's output in the viewer"));
        impl_->remove = new QPushButton(QStringLiteral("Remove"), footer);
        widgets::setButtonClass(impl_->remove, "ghost");
        fl->addWidget(impl_->run, 1);
        fl->addWidget(impl_->view);
        fl->addWidget(impl_->remove);
        root->addWidget(footer);

        connect(impl_->help, &QAbstractButton::clicked, this, [this] { emit helpRequested(!impl_->help->isActive()); });
        connect(impl_->backend, &SegmentedControl::changed, this, [this](int i) {
            impl_->bridge.wb().setBackend(static_cast<Backend>(i));
        });
        connect(impl_->cache, &SegmentedControl::changed, this, [this](int i) {
            impl_->bridge.wb().setStepCache(impl_->index(), static_cast<CachePolicy>(i));
        });
        connect(impl_->run, &QPushButton::clicked, this, [this] { emit runStepRequested(impl_->index()); });
        connect(impl_->view, &QPushButton::clicked, this, [this] { impl_->bridge.wb().view(impl_->index()); });
        connect(impl_->remove, &QPushButton::clicked, this, [this] { emit removeStepRequested(impl_->index()); });

        connect(&bridge, &WorkbenchBridge::selectionChanged, this, [this] { impl_->refresh(); });
        connect(&bridge, &WorkbenchBridge::pipelineChanged, this, [this] { impl_->refresh(); });
        connect(&bridge, &WorkbenchBridge::datasetChanged, this, [this] {
            impl_->builtFor = -1;
            impl_->refresh();
        });
        connect(&bridge, &WorkbenchBridge::stepChanged, this, [this](int index) {
            if (index == impl_->index()) impl_->refresh();
        });
        connect(&bridge, &WorkbenchBridge::backendChanged, this, [this] { impl_->refreshHeader(); });
        connect(&bridge, &WorkbenchBridge::outputsChanged, this, [this] { impl_->refreshHeader(); });
        connect(&bridge, &WorkbenchBridge::runStarted, this, [this] { impl_->refreshHeader(); });
        connect(&bridge, &WorkbenchBridge::runStateChanged, this, [this] { impl_->refreshHeader(); });
        connect(&bridge, &WorkbenchBridge::runFinished, this, [this] {
            impl_->builtFor = -1;   // diagnostics-driven notes may have changed
            impl_->refresh();
        });
        connect(&bridge, &WorkbenchBridge::taskStarted, this, [this] { impl_->refreshHeader(); });
        connect(&bridge, &WorkbenchBridge::taskFinished, this, [this] { impl_->refreshHeader(); });
        impl_->refresh();
    }

    ParamsPanel::~ParamsPanel() = default;

    QSize ParamsPanel::sizeHint() const { return {theme::kParamsDockW, QWidget::sizeHint().height()}; }

    void ParamsPanel::setHelpOpen(bool open) { impl_->help->setActive(open); }

    bool ParamsPanel::eventFilter(QObject* watched, QEvent* event) {
        if (watched == impl_->name && event->type() == QEvent::MouseButtonDblClick) {
            const Step* st = impl_->step();
            if (st) {
                bool ok = false;
                const QString text = QInputDialog::getText(this, QStringLiteral("Rename step"), QStringLiteral("Name"),
                                                           QLineEdit::Normal, fromStd(st->name), &ok);
                if (ok) impl_->bridge.wb().renameStep(impl_->index(), toStd(text));
            }
            return true;
        }
        return QWidget::eventFilter(watched, event);
    }

} // namespace sirius::app
