#ifndef SIRIUS_IMGUI_VIEWER_SLICE_PANE_HPP
#define SIRIUS_IMGUI_VIEWER_SLICE_PANE_HPP

// One pane of the ortho / compare layouts: draws a rendered slice image on
// the viewer ground with a per-axis view transform (display pixels per voxel
// and the position of voxel (0, 0) in the pane), and the overlays of the
// design -- corner label, scale bar, tool hint, crosshair, brush outline,
// measure and ROI marks, track paths. The pane knows nothing about tools: it
// reports mouse events in voxel coordinates through its callbacks and the
// viewer decides what they mean. A pane that was clicked has the keyboard:
// the arrow keys walk the crosshair over its own two axes, page up / down
// step the third (the viewer asks keyNavigation() for them).
//
// Coordinates: "local" positions are display pixels from the pane's
// top-left corner; voxel positions are columns / rows of the pane's plane.

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <imgui.h>

#include <sirius/buffer.hpp>

#include "imgui/gl.hpp"
#include "imgui/theme.hpp"
#include "imgui/viewer/display_model.hpp"
#include "imgui/viewer/track_overlay.hpp"

namespace sirius::app::gui {

    // Text drawn like the prototype's overlay labels (11 px, viewer text
    // colour, optional opacity), for use over the viewer ground. `pos` is the
    // top-left corner in display pixels. Returns the width drawn.
    float drawOverlayText(ImDrawList* dl, ImVec2 pos, const std::string& text, bool bold = false, float opacity = 1.0f,
                          float designPx = 11);
    // `text` word-wrapped and centred in (min, max), each line on its own.
    void drawCenteredWrapped(ImDrawList* dl, ImVec2 min, ImVec2 max, const std::string& text, float designPx, ImU32 color);
    // A rectangle outline in dashes (4 on, 2 off, in pens).
    void dashedOutline(ImDrawList* dl, ImVec2 a, ImVec2 b, ImU32 color, float thickness);

    class SlicePane {
    public:
        enum class Kind { XY,
                          YZ,
                          XZ,
                          MIP,
                          Compare };

        // Display pixels per voxel along the columns / rows, and where voxel
        // (0, 0) sits in the pane.
        struct View {
            double zx = 1.0, zy = 1.0;
            double ox = 0.0, oy = 0.0;
        };

        // Annotations in voxel coordinates of this pane's plane: measurements
        // (1..3 points: mark, distance, angle) and ROI boxes. Pending ones
        // (still being drawn) are painted lighter.
        struct Annotation {
            enum class Kind { Measure,
                              Roi };
            Kind kind = Kind::Measure;
            std::vector<DPoint> points;
            DRect rect;
            std::string text;
            bool pending = false;
        };

        SlicePane(Kind kind, std::string name);
        SlicePane(const SlicePane&) = delete;
        SlicePane& operator=(const SlicePane&) = delete;
        Kind kind() const noexcept { return kind_; }
        const std::string& name() const noexcept { return name_; }

        // The rendered image covers part of a (cols, rows) voxel grid with
        // `factor` voxels per image pixel, starting at voxel (originX,
        // originY) (the viewer renders the visible region plus a margin, not
        // the whole plane). Uploaded to the pane's texture at once: call on
        // the GUI thread.
        void setContent(const Image& img, int factor, Index cols, Index rows, int originX = 0, int originY = 0);
        int originX() const noexcept { return originX_; }
        int originY() const noexcept { return originY_; }
        // The grid alone (fitView needs it before the first content arrives).
        void setGrid(Index cols, Index rows) {
            cols_ = cols;
            rows_ = rows;
        }
        void clearContent();
        bool hasContent() const noexcept { return hasImage_; }
        Index cols() const noexcept { return cols_; }
        Index rows() const noexcept { return rows_; }

        void setView(const View& v) { view_ = v; }
        const View& view() const noexcept { return view_; }
        // View that fits a grid whose voxels are (ax, ay) units in size.
        View fitView(double ax, double ay) const;
        DPoint toVoxel(const DPoint& local) const;
        DPoint toScreen(const DPoint& voxel) const;       // local
        ImVec2 toScreenAbs(const DPoint& voxel) const;    // display pixels of the window
        bool inside(const DPoint& voxel) const;

        // --- geometry ----------------------------------------------------------
        // Where the pane is this frame (display pixels); true when its size changed.
        bool place(ImVec2 min, ImVec2 max);
        ImVec2 min() const noexcept { return min_; }
        ImVec2 max() const noexcept { return max_; }
        double width() const noexcept { return static_cast<double>(max_.x - min_.x); }
        double height() const noexcept { return static_cast<double>(max_.y - min_.y); }
        bool placed() const noexcept { return max_.x > min_.x && max_.y > min_.y; }

        // --- overlays ---------------------------------------------------------
        void setTitle(const std::string& title) { title_ = title; }     // "XY  z 24 / 47 ..."
        void setHint(const std::string& hint) { hint_ = hint; }         // tool hint, bottom-left
        void setScaleBar(double umPerVoxel) { umPerVoxel_ = umPerVoxel; }   // 0 hides it
        void setCrosshair(const DPoint& voxel, bool visible, bool locked) {
            cross_ = voxel;
            crossVisible_ = visible;
            crossLocked_ = locked;
        }
        void setBrushCursor(bool on, double radiusVoxels) {
            brush_ = on;
            brushRadius_ = radiusVoxels;
        }
        void setAnnotations(std::vector<Annotation> annotations) { annotations_ = std::move(annotations); }
        // The prompts of a Prompt step, in this pane's voxel coordinates.
        // On the pane's plane: an object point is a filled accent disc with a
        // light ring and a plus, a background point a dark disc with a light
        // ring and a minus; a box an accent rectangle and a scribble an accent
        // stroke (light when it marks background), all edged in dark so they
        // read on bright data as on dark. Off the plane, where they project:
        // small, faint, dashed. Pending (being drawn): dashed and light.
        struct PromptMark {
            enum class Shape { Point,
                               Box,
                               Stroke };
            Shape shape = Shape::Point;
            DPoint a, b;                  // a point's voxel; a box's corners, b exclusive
            std::vector<DPoint> stroke;   // a scribble's voxels
            bool object = true;
            bool inPlane = true;
            bool pending = false;
        };
        void setPromptMarks(std::vector<PromptMark> marks) { promptMarks_ = std::move(marks); }
        // Trajectories of tracked labels (track_overlay.hpp), drawn over the
        // image and under the annotations; null draws none. The paths are
        // shared with the viewer, which rebuilds them only when the labels change.
        void setTracks(std::shared_ptr<const std::vector<TrackPath>> paths, const TrackPaintOptions& options) {
            tracks_ = std::move(paths);
            trackOptions_ = options;
        }
        void setMessage(const std::string& text) { message_ = text; }   // centred notice ("volume too large")
        void setSmooth(bool smooth) { smooth_ = smooth; }
        // The mouse cursor over the pane, idle and while a button is held.
        void setCursor(ImGuiMouseCursor idle, ImGuiMouseCursor held) {
            cursor_ = idle;
            cursorHeld_ = held;
        }
        DPoint lastMouse() const noexcept { return mouse_; }
        bool mouseInside() const noexcept { return mouseIn_; }

        // --- input -------------------------------------------------------------
        // Mouse events, in this pane's voxels (the wheel and the context menu
        // in local pixels / window pixels). Buttons are ImGuiMouseButton;
        // modifiers are ImGuiMod_* flags.
        std::function<void(DPoint)> onHover;
        std::function<void()> onExit;
        std::function<void(DPoint, int, ImGuiKeyChord)> onPress;
        std::function<void(DPoint, DPoint, int, ImGuiKeyChord)> onDrag;   // voxel, local delta
        std::function<void(DPoint, int, ImGuiKeyChord, bool)> onRelease;  // moved
        std::function<void(DPoint, ImGuiKeyChord)> onDoubleClick;
        std::function<void(DPoint, double, ImGuiKeyChord)> onWheel;       // local position, steps
        std::function<void(int, int, int)> onKeyNavigate;                 // columns, rows, depth
        std::function<void(ImVec2, DPoint)> onContextMenu;                // right press: window position, voxel

        // Submits the pane's hit area at its place and turns this frame's
        // mouse into the callbacks above. True when it was pressed this frame
        // (the pane takes the keyboard).
        bool input();
        // The arrow / page keys (shift: ten at a time; with Ctrl, Alt or
        // Super they are the window's) for a pane that has the keyboard; the
        // viewer calls this only then (and claims the plain keys from the
        // window's menu actions meanwhile).
        void keyNavigation();
        bool hovered() const noexcept { return hovered_; }
        bool dragging() const noexcept { return button_ >= 0; }
        // Scripting: the press / drag / release a mouse would have produced,
        // at local positions (the same callbacks, the same bookkeeping).
        void synthPress(const DPoint& local, int button);
        void synthMove(const DPoint& local);
        void synthRelease(const DPoint& local);

        // Draws the image and the overlays into the pane's rectangle.
        void draw(ImDrawList* dl);

    private:
        Kind kind_;
        std::string name_;
        Texture texture_;
        bool hasImage_ = false;
        int imageW_ = 0, imageH_ = 0;
        int factor_ = 1;
        int originX_ = 0, originY_ = 0;   // voxel of the image's top-left corner
        Index cols_ = 0, rows_ = 0;
        View view_;
        ImVec2 min_{0, 0}, max_{0, 0};
        std::string title_, hint_, message_;
        double umPerVoxel_ = 0.0;
        DPoint cross_;
        bool crossVisible_ = false, crossLocked_ = false;
        bool brush_ = false;
        double brushRadius_ = 0.0;
        std::vector<Annotation> annotations_;
        std::vector<PromptMark> promptMarks_;
        std::shared_ptr<const std::vector<TrackPath>> tracks_;
        TrackPaintOptions trackOptions_;
        bool smooth_ = false;
        ImGuiMouseCursor cursor_ = ImGuiMouseCursor_Arrow, cursorHeld_ = ImGuiMouseCursor_Arrow;
        DPoint mouse_{-1, -1};
        bool mouseIn_ = false;
        bool hovered_ = false;
        int button_ = -1;                 // the button a drag is made with
        bool releaseAfterDouble_ = false;   // the second click of a double-click still has to come up
        DPoint pressPos_, lastDrag_;
        bool moved_ = false;
    };

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_VIEWER_SLICE_PANE_HPP
