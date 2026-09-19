#ifndef SIRIUS_APP_QT_PANELS_TRACK_TABLE_HPP
#define SIRIUS_APP_QT_PANELS_TRACK_TABLE_HPP

// The track table of a tracked label volume: one row per track with the
// columns a reviewer sorts by to find the broken ones -- where it starts and
// ends, how many frames it is missing, how fast and how far it moves, and
// its parent and children. Like the diagnostics cells it knows nothing about
// the workbench: it shows a vector of TrackSummary and reports which track
// was chosen, so it can be driven offscreen.
//
// Division counts come from the model's geometric rule, which under-calls on
// real detections (docs/foundation_model_integration.md, section 4), and the
// caption says so rather than presenting them as a measurement.

#include <cstdint>
#include <map>
#include <vector>

#include <QAbstractTableModel>
#include <QPixmap>
#include <QTableView>
#include <QWidget>

#include "core/tracks.hpp"

class QCheckBox;
class QLabel;
class QSortFilterProxyModel;

namespace sirius::app {

    class TrackTableModel : public QAbstractTableModel {
        Q_OBJECT
    public:
        enum Column { Id, Frames, Present, Gaps, Speed, Net, Parent, Children, ColumnCount };

        explicit TrackTableModel(QObject* parent = nullptr);
        void setTracks(std::vector<TrackSummary> tracks);
        const std::vector<TrackSummary>& tracks() const noexcept { return tracks_; }
        std::uint32_t idAt(int row) const;
        int rowOf(std::uint32_t id) const;

        int rowCount(const QModelIndex& parent = QModelIndex()) const override;
        int columnCount(const QModelIndex& parent = QModelIndex()) const override;
        QVariant headerData(int section, Qt::Orientation o, int role) const override;
        QVariant data(const QModelIndex& index, int role) const override;

    private:
        const QPixmap& chip(std::uint32_t id) const;
        std::vector<TrackSummary> tracks_;
        std::map<std::uint32_t, int> rows_;
        mutable std::map<QRgb, QPixmap> chips_;
    };

    class TrackTable : public QWidget {
        Q_OBJECT
    public:
        explicit TrackTable(QWidget* parent = nullptr);

        // A new set of tracks keeps the sort order and the selected id.
        void setTracks(std::vector<TrackSummary> tracks);
        void setSelectedTrack(std::uint32_t id);   // 0 clears; emits nothing
        void setFollow(bool on);                   // emits nothing
        std::uint32_t selectedTrack() const;
        const TrackTableModel& model() const noexcept { return *model_; }

    signals:
        void trackChosen(std::uint32_t id);
        void followToggled(bool on);

    private:
        void refreshCaption();

        TrackTableModel* model_ = nullptr;
        QSortFilterProxyModel* proxy_ = nullptr;
        QTableView* view_ = nullptr;
        QLabel* caption_ = nullptr;
        QCheckBox* follow_ = nullptr;
        bool updating_ = false;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_QT_PANELS_TRACK_TABLE_HPP
