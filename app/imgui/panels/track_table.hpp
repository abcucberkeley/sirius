#ifndef SIRIUS_IMGUI_PANELS_TRACK_TABLE_HPP
#define SIRIUS_IMGUI_PANELS_TRACK_TABLE_HPP

// The track table of a tracked label volume: one row per track with the
// columns a reviewer sorts by to find the broken ones -- where it starts and
// ends, how many frames it is missing, how fast and how far it moves, and
// its parent and children. Like the diagnostics cells it knows nothing about
// the workbench: it shows a vector of TrackSummary and reports which track
// was chosen. (app/qt/panels/track_table.cpp)
//
// Division counts come from the model's geometric rule, which under-calls on
// real detections (docs/foundation_model_integration.md, section 4), and the
// caption says so rather than presenting them as a measurement.
//
// Immediate mode: the selection is not the table's own. The caller passes the
// selected track (the view state's selected label) every frame and hears
// back what the user chose or toggled.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "core/tracks.hpp"

namespace sirius::app::gui {

    class TrackTable {
    public:
        enum Column { Id,
                      Frames,
                      Present,
                      Gaps,
                      Speed,
                      Net,
                      Parent,
                      Children,
                      ColumnCount };

        // A new set of tracks keeps the sort order (the selection is the caller's).
        void setTracks(std::vector<TrackSummary> tracks);
        const std::vector<TrackSummary>& tracks() const noexcept { return tracks_; }
        // Index into tracks(), -1 when no track has this id.
        int rowOf(std::uint32_t id) const;

        struct Events {
            std::uint32_t chosen = 0;         // a row was clicked (also the one already selected)
            std::optional<bool> follow;       // "Follow selected track" toggled
        };
        // Fills the rest of the current window: the caption line with the
        // follow toggle, then the table. `selected` is highlighted (0: none)
        // and scrolled into view when it changes.
        Events draw(std::uint32_t selected, bool follow);

    private:
        void resort();
        std::string caption() const;

        std::vector<TrackSummary> tracks_;
        std::vector<int> order_;              // display row -> tracks_ index
        int sortColumn_ = Id;
        bool sortDescending_ = false;
        bool sortDirty_ = true;
        std::uint32_t shownSelection_ = 0;    // the selection last scrolled to
        std::string caption_;
    };

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_PANELS_TRACK_TABLE_HPP
