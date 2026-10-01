#ifndef SIRIUS_IMGUI_VIEWER_HPP
#define SIRIUS_IMGUI_VIEWER_HPP

// The central panel: viewer toolbar (Ortho | 3D | Compare, "Viewing 05
// Contrast …", Labels / Crosshair, channel swatches), the tool strip, the
// ortho / 3D / compare views and the dims strip (Z, T with play). It draws
// whatever wb().displayOutput() returns and writes every interaction back
// into the workbench's ViewState, so the assistant and the menus see the
// same state the mouse produces.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sirius::app::gui {

    class App;

    class Viewer {
    public:
        explicit Viewer(App& app);
        ~Viewer();
        Viewer(const Viewer&) = delete;
        Viewer& operator=(const Viewer&) = delete;

        // The panel's content, inside the central window the application began.
        void draw();

        // Menu actions call these; they route through the workbench.
        void zoomIn();
        void zoomOut();
        void fitToWindow();
        void setPlaying(bool on);
        bool playing() const;
        void autoContrast();      // display windows back to the percentile auto window
        void resetContrast();     // display windows to the full data range

        // The current view as an image (Export figure): RGBA, rows top to
        // bottom. False when there is nothing to grab.
        bool grabView(std::vector<std::uint8_t>& rgba, int& width, int& height);

        // Status bar readouts: "cursor x, y, z · value" and "100 %".
        std::string cursorText() const;
        std::string zoomText() const;
        // A volume being read for the re-slices / the 3D view: the status bar
        // shows it while no run or task has the progress bar.
        bool loading() const;
        double loadFraction() const;
        std::string loadMessage() const;
        // True while the viewer wants frames without input (play, a load, a
        // Prompt step waiting to re-run).
        bool animating() const;
        // The points of Prompt step `stepId` changed outside the viewer (the
        // parameters panel's list): it re-runs as after a click, once the
        // edits pause.
        void promptsEdited(std::uint64_t stepId);

        // Scripting / testing: mouse input on the XY pane, positions in
        // voxels. --stroke and --wheel on the command line use these.
        void syntheticStroke(double x0, double y0, double x1, double y1, int moves);
        void syntheticWheel(double x, double y, double steps);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_VIEWER_HPP
