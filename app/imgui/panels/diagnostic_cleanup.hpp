#ifndef SIRIUS_IMGUI_PANELS_DIAGNOSTIC_CLEANUP_HPP
#define SIRIUS_IMGUI_PANELS_DIAGNOSTIC_CLEANUP_HPP

// The segmentation cleanup page of the diagnostics dock: the 4 x 2 tool grid
// with the brush size and "Paint in 3D", the label table (one row per label:
// id with its colour chip, class, voxels, confidence, flag, and the merge /
// split / delete links) and the review queue. (The SegmentCleanupView of
// app/qt/panels/diagnostics_panel.cpp.)
//
// Tens of thousands of labels are common: the table reads the statistics of
// the viewed label volume in place and draws only the rows on screen, so a
// refresh after every stroke costs nothing.

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "core/labels.hpp"

namespace sirius::app::gui {

    class App;

    class SegmentCleanupView {
    public:
        explicit SegmentCleanupView(App& app) : app_(app) {}

        // The labels on show (the viewed step's), after anything that may
        // have replaced them or edited them. The table's selection becomes
        // the view state's label alone, as a refresh of the Qt table did.
        void setLabels(std::shared_ptr<LabelVolume> labels);
        // Fills the rest of the current window.
        void draw();

    private:
        void drawTools(float x0, float x1, float y0, float y1, bool editable);
        void drawTable(float x0, float x1, float y0, float y1);
        void drawQueue(float x0, float x1, float y0, float y1, bool editable);
        // The table's rows in display order (indices into the statistics).
        void refreshOrder();
        // A click on a row: plain, Ctrl (toggle) or Shift (range).
        void clickRow(int displayRow, std::uint32_t id);
        void act(const std::string& link, std::uint32_t id);

        App& app_;
        std::shared_ptr<LabelVolume> labels_;

        // selection (label ids) and the row the view state names
        std::set<std::uint32_t> selected_;
        std::uint32_t primary_ = 0;          // what we last wrote to ViewState::selectedLabel
        std::uint32_t seenSelected_ = 0;     // ViewState::selectedLabel as last seen
        int anchorRow_ = -1;                 // Shift+click extends from here
        bool scrollToSelected_ = false;

        // display order
        std::vector<int> order_;
        const LabelVolume* orderedFor_ = nullptr;
        std::size_t orderedSize_ = 0;
        std::uint64_t orderedRev_ = 0;
        int sortColumn_ = -1;                // -1: the statistics' own order
        bool sortDescending_ = false;
    };

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_PANELS_DIAGNOSTIC_CLEANUP_HPP
