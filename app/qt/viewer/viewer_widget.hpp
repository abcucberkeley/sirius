#ifndef SIRIUS_APP_VIEWER_WIDGET_HPP
#define SIRIUS_APP_VIEWER_WIDGET_HPP

// The central widget: viewer toolbar (Ortho | 3D | Compare, "Viewing 05
// Contrast …", Labels / Crosshair, channel swatches), the tool strip, the
// ortho / 3D / compare views and the dims strip (Z, T with play). It draws
// whatever wb().displayOutput() returns and writes every interaction back
// into the workbench's ViewState, so the assistant and the menus see the
// same state the mouse produces.

#include <QWidget>

#include "qt/workbench_bridge.hpp"

namespace sirius::app {

    class ViewerWidget : public QWidget {
        Q_OBJECT
    public:
        explicit ViewerWidget(WorkbenchBridge& bridge, QWidget* parent = nullptr);
        ~ViewerWidget() override;

        // Menu actions call these; they route through the workbench.
        void zoomIn();
        void zoomOut();
        void fitToWindow();
        void setPlaying(bool on);
        bool playing() const;
        // Recompute the per-channel display windows (View ▸ Auto contrast).
        void autoContrast();      // display windows back to the percentile auto window
        void resetContrast();     // display windows to the full data range
        // Grab the current view as an image (Export figure).
        QImage grabView() const;
        // Cursor readout for the status bar: "cursor x, y, z · value"
        QString cursorText() const;
        QString zoomText() const;         // "100 %"

        // Scripting / testing: real mouse events on the XY pane, positions in
        // voxels; events are processed between the moves as they would be
        // for a user. --stroke and --wheel on the command line use these.
        void syntheticStroke(const QPointF& fromVoxel, const QPointF& toVoxel, int moves);
        void syntheticWheel(const QPointF& atVoxel, double steps);

    signals:
        void cursorChanged(const QString& text);
        void zoomChanged(const QString& text);
        void loadStarted();
        void loadProgress(double fraction, const QString& message);
        void loadFinished();

    protected:
        // The ortho splitters take their saved (or design) balance once the
        // panes have real geometry.
        void showEvent(QShowEvent* event) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_VIEWER_WIDGET_HPP
