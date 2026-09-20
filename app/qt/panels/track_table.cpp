#include "qt/panels/track_table.hpp"

#include "core/labels.hpp"   // labelColor

#include <QApplication>
#include <QCheckBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QItemSelectionModel>
#include <QLabel>
#include <QSortFilterProxyModel>
#include <QVBoxLayout>

#include "qt/theme.hpp"
#include "qt/widgets/controls.hpp"

namespace sirius::app {

    namespace {
        // Sort key: numbers as numbers, not as the text in the cell.
        constexpr int kSortRole = Qt::UserRole + 1;

        QString number(double v, int decimals) { return QString::number(v, 'f', decimals); }
    } // namespace

    // --- model ----------------------------------------------------------------------

    TrackTableModel::TrackTableModel(QObject* parent) : QAbstractTableModel(parent) {}

    void TrackTableModel::setTracks(std::vector<TrackSummary> tracks) {
        beginResetModel();
        tracks_ = std::move(tracks);
        rows_.clear();
        for (std::size_t i = 0; i < tracks_.size(); ++i) rows_[tracks_[i].id] = static_cast<int>(i);
        endResetModel();
    }

    std::uint32_t TrackTableModel::idAt(int row) const {
        return row >= 0 && row < static_cast<int>(tracks_.size()) ? tracks_[static_cast<std::size_t>(row)].id : 0u;
    }

    int TrackTableModel::rowOf(std::uint32_t id) const {
        const auto it = rows_.find(id);
        return it == rows_.end() ? -1 : it->second;
    }

    int TrackTableModel::rowCount(const QModelIndex& parent) const {
        return parent.isValid() ? 0 : static_cast<int>(tracks_.size());
    }

    int TrackTableModel::columnCount(const QModelIndex& parent) const { return parent.isValid() ? 0 : ColumnCount; }

    QVariant TrackTableModel::headerData(int section, Qt::Orientation o, int role) const {
        if (o != Qt::Horizontal) return {};
        static const char* names[ColumnCount] = {"ID", "FRAMES", "PRESENT", "GAPS", "µm / FRAME", "NET µm", "PARENT", "CHILDREN"};
        static const char* tips[ColumnCount] = {
            "Track id: the label id it has in every frame",
            "First and last frame the track is present in",
            "Frames the track is present in",
            "Frames missing between its first and last: where it was lost and picked up again",
            "Centroid path length divided by the frames it spans",
            "Distance from the first centroid to the last",
            "The track it divided from, when the tracker reported one",
            "The tracks it divided into",
        };
        if (section < 0 || section >= ColumnCount) return {};
        if (role == Qt::DisplayRole) return QString::fromUtf8(names[section]);
        if (role == Qt::ToolTipRole) return QString::fromUtf8(tips[section]);
        return {};
    }

    QVariant TrackTableModel::data(const QModelIndex& index, int role) const {
        if (!index.isValid() || index.row() < 0 || index.row() >= static_cast<int>(tracks_.size())) return {};
        const TrackSummary& s = tracks_[static_cast<std::size_t>(index.row())];
        switch (role) {
            case Qt::DisplayRole:
                switch (index.column()) {
                    case Id: return QStringLiteral("%1").arg(s.id, 4, 10, QChar('0'));
                    case Frames: return QStringLiteral("%1 – %2").arg(s.first).arg(s.last);
                    case Present: return QString::number(s.frames);
                    case Gaps: return s.gaps ? QString::number(s.gaps) : QStringLiteral("—");
                    case Speed: return s.last > s.first ? number(s.umPerFrame, 2) : QStringLiteral("—");
                    case Net: return number(s.netUm, 2);
                    case Parent: return s.parent ? QString::number(s.parent) : QStringLiteral("—");
                    case Children: {
                        if (s.children.empty()) return QStringLiteral("—");
                        QStringList ids;
                        for (std::uint32_t c : s.children) ids << QString::number(c);
                        return ids.join(QStringLiteral(", "));
                    }
                    default: return {};
                }
            case kSortRole:
                switch (index.column()) {
                    case Id: return static_cast<qulonglong>(s.id);
                    case Frames: return static_cast<qlonglong>(s.first);
                    case Present: return static_cast<qlonglong>(s.frames);
                    case Gaps: return static_cast<qlonglong>(s.gaps);
                    case Speed: return s.umPerFrame;
                    case Net: return s.netUm;
                    case Parent: return static_cast<qulonglong>(s.parent);
                    case Children: return static_cast<qulonglong>(s.children.size());
                    default: return {};
                }
            case Qt::DecorationRole:
                return index.column() == Id ? QVariant(chip(s.id)) : QVariant();
            case Qt::ForegroundRole:
                // a gap is the first thing to look at: where identity may have been lost
                if (index.column() == Gaps && s.gaps > 0) return QBrush(theme::kAccentText);
                return {};
            case Qt::UserRole:
                return static_cast<uint>(s.id);
            default:
                return {};
        }
    }

    const QPixmap& TrackTableModel::chip(std::uint32_t id) const {
        const auto c = labelColor(id);
        const QRgb key = QColor::fromRgbF(c[0], c[1], c[2]).rgb();
        auto it = chips_.find(key);
        if (it == chips_.end()) {
            const qreal dpr = qApp ? qApp->devicePixelRatio() : 1.0;
            QPixmap pm(qRound(10 * dpr), qRound(10 * dpr));
            pm.setDevicePixelRatio(dpr);
            pm.fill(QColor(key));
            it = chips_.emplace(key, pm).first;
        }
        return it->second;
    }

    // --- widget ---------------------------------------------------------------------

    TrackTable::TrackTable(QWidget* parent) : QWidget(parent) {
        auto* v = new QVBoxLayout(this);
        v->setContentsMargins(0, 0, 0, 0);
        v->setSpacing(4);

        auto* head = new QHBoxLayout;
        head->setContentsMargins(14, 6, 14, 0);
        caption_ = new QLabel(this);
        caption_->setFont(theme::font(theme::kSmallPx));
        caption_->setTextFormat(Qt::PlainText);
        follow_ = new QCheckBox(QStringLiteral("Follow selected track"), this);
        follow_->setFont(theme::font(theme::kSmallPx));
        follow_->setToolTip(QStringLiteral("Keep the selected track centred while the time point changes"));
        head->addWidget(caption_, 1);
        head->addWidget(follow_);
        v->addLayout(head);

        model_ = new TrackTableModel(this);
        proxy_ = new QSortFilterProxyModel(this);
        proxy_->setSourceModel(model_);
        proxy_->setSortRole(kSortRole);
        view_ = new QTableView(this);
        view_->setModel(proxy_);
        view_->setFrameShape(QFrame::NoFrame);
        view_->setShowGrid(false);
        view_->setEditTriggers(QAbstractItemView::NoEditTriggers);
        view_->setSelectionMode(QAbstractItemView::SingleSelection);
        view_->setSelectionBehavior(QAbstractItemView::SelectRows);
        view_->setSortingEnabled(true);
        view_->sortByColumn(TrackTableModel::Id, Qt::AscendingOrder);
        view_->setAccessibleName(QStringLiteral("Tracks"));
        view_->setObjectName(QStringLiteral("trackTable"));   // the toolbar's Tracks toggle has the accessible name too
        view_->setAccessibleDescription(
            QStringLiteral("One row per track: id, frames, frames present, gaps, speed, net displacement, parent, children"));
        view_->verticalHeader()->setVisible(false);
        view_->verticalHeader()->setDefaultSectionSize(22);
        view_->horizontalHeader()->setStretchLastSection(true);
        view_->horizontalHeader()->setHighlightSections(false);
        view_->horizontalHeader()->setDefaultAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        view_->horizontalHeader()->setSectionResizeMode(QHeaderView::Interactive);
        view_->setFont(theme::tabular(theme::font(theme::kSmallPx)));
        widgets::setWidgetClass(view_, "dense");
        // fixed widths: ResizeToContents would measure every row
        const QFontMetrics fm(theme::font(theme::kSmallPx));
        view_->setColumnWidth(TrackTableModel::Id, fm.horizontalAdvance(QStringLiteral("00000")) + 46);
        view_->setColumnWidth(TrackTableModel::Frames, fm.horizontalAdvance(QStringLiteral("0000 – 0000")) + 24);
        view_->setColumnWidth(TrackTableModel::Present, fm.horizontalAdvance(QStringLiteral("PRESENT")) + 24);
        view_->setColumnWidth(TrackTableModel::Gaps, fm.horizontalAdvance(QStringLiteral("GAPS")) + 28);
        view_->setColumnWidth(TrackTableModel::Speed, fm.horizontalAdvance(QStringLiteral("µm / FRAME")) + 24);
        view_->setColumnWidth(TrackTableModel::Net, fm.horizontalAdvance(QStringLiteral("000.00")) + 30);
        view_->setColumnWidth(TrackTableModel::Parent, fm.horizontalAdvance(QStringLiteral("PARENT")) + 24);
        v->addWidget(view_, 1);

        connect(view_->selectionModel(), &QItemSelectionModel::selectionChanged, this, [this] {
            if (updating_) return;
            const std::uint32_t id = selectedTrack();
            if (id) emit trackChosen(id);
        });
        // a click on the row already selected still means "take me there"
        connect(view_, &QTableView::clicked, this, [this](const QModelIndex& index) {
            if (updating_ || !index.isValid()) return;
            const std::uint32_t id = model_->idAt(proxy_->mapToSource(index).row());
            if (id && id == selectedTrack()) emit trackChosen(id);
        });
        connect(follow_, &QCheckBox::toggled, this, [this](bool on) {
            if (!updating_) emit followToggled(on);
        });
        refreshCaption();
    }

    void TrackTable::setTracks(std::vector<TrackSummary> tracks) {
        const std::uint32_t selected = selectedTrack();
        updating_ = true;
        model_->setTracks(std::move(tracks));
        updating_ = false;
        setSelectedTrack(selected);
        refreshCaption();
    }

    void TrackTable::setSelectedTrack(std::uint32_t id) {
        updating_ = true;
        const int row = id ? model_->rowOf(id) : -1;
        if (row < 0) {
            view_->clearSelection();
        } else {
            const QModelIndex at = proxy_->mapFromSource(model_->index(row, 0));
            view_->selectRow(at.row());
            view_->scrollTo(at, QAbstractItemView::EnsureVisible);
        }
        updating_ = false;
    }

    void TrackTable::setFollow(bool on) {
        updating_ = true;
        follow_->setChecked(on);
        updating_ = false;
    }

    std::uint32_t TrackTable::selectedTrack() const {
        const QModelIndexList rows = view_->selectionModel()->selectedRows();
        if (rows.isEmpty()) return 0;
        return model_->idAt(proxy_->mapToSource(rows.first()).row());
    }

    void TrackTable::refreshCaption() {
        const std::vector<TrackSummary>& tracks = model_->tracks();
        if (tracks.empty()) {
            caption_->setText(QStringLiteral("No tracks: run a tracking step to fill this table."));
            return;
        }
        Index gapped = 0;
        for (const TrackSummary& s : tracks)
            if (s.gaps > 0) ++gapped;
        QString text = QStringLiteral("%1 tracks · %2 with gaps").arg(tracks.size()).arg(gapped);
        const Index divisions = countDivisions(tracks);
        if (divisions > 0)
            text += QStringLiteral(" · %1 divisions (tracker's estimate, not verified)").arg(divisions);
        caption_->setText(text);
    }

} // namespace sirius::app
