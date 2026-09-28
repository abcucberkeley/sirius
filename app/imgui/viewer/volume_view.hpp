#ifndef SIRIUS_IMGUI_VIEWER_VOLUME_VIEW_HPP
#define SIRIUS_IMGUI_VIEWER_VOLUME_VIEW_HPP

// The 3D layout: a ray-cast rendering of the current time point (OpenGL 3.3
// core), rendered off screen into a RenderTarget and drawn as an image.
// Every visible channel arrives as an 8-bit brick of its windowed
// intensities, already down-sampled to at most 256 voxels per axis by
// ViewerLoader (the reduction is a pass over the whole volume and must not
// happen inside a frame), is uploaded as a 3D texture and composited front
// to back through a linear opacity ramp (or as a maximum-intensity
// projection) in the channel's colour. The bounding box, the corner label,
// the view presets, the yaw / pitch sliders and the Z clip range are drawn
// or laid over the rendering exactly as in the design.
// (app/qt/viewer/volume_view.*)
//
// The overlay controls live in child windows of their own, so a figure
// grabbed from the viewer's draw list shows the rendering without them --
// as QOpenGLWidget::grabFramebuffer did.

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <imgui.h>

#include <sirius/buffer.hpp>

#include "imgui/gl.hpp"
#include "imgui/viewer/viewer_loader.hpp"

namespace sirius::app::gui {

    class VolumeView {
    public:
        VolumeView();
        ~VolumeView();   // GL objects: on the GUI thread, with the context current
        VolumeView(const VolumeView&) = delete;
        VolumeView& operator=(const VolumeView&) = delete;

        // The reduced bricks of the visible channels, with the full-resolution
        // grid they came from (the box keeps the physical aspect). `key`
        // identifies the (output, t, channels, windows) combination; the
        // textures are uploaded only when it changes.
        void setTextures(std::uint64_t key, std::vector<ReducedVolume> channels, const std::array<double, 3>& voxelUm, Index nz,
                         Index ny, Index nx);
        void clearVolumes();
        bool hasVolumes() const noexcept { return !textures_.empty(); }
        // Drawn instead of "No volume to render" while the loader is reading
        // or reducing what this view will show next.
        void setPreparing(const std::string& text) { preparing_ = text; }

        // Instance labels of the same (z, y, x) grid, composited over the
        // volume in their palette colours; `key` changes with every edit.
        // `owner` keeps the voxels `labels` points into alive until the next
        // setLabels / clearLabels: a paint stroke (copy-on-write) or a re-run
        // may otherwise free them between the call and the next upload.
        void setLabels(std::uint64_t key, std::shared_ptr<const void> owner, const std::uint32_t* labels, Index z, Index y, Index x,
                       float opacity, std::uint32_t only = 0);
        void clearLabels();
        bool hasLabels() const noexcept { return labels_ != nullptr; }

        void setOrientation(double yawDeg, double pitchDeg) { applyOrientation(yawDeg, pitchDeg, false); }
        double yaw() const noexcept { return yaw_; }
        double pitch() const noexcept { return pitch_; }
        void setClip(double lo, double hi);
        void setBoundingBox(bool on) { box_ = on; }
        void setZoom(double zoom);
        double zoom() const noexcept { return zoom_; }
        // Transfer function: opacity ramps from 0 at `lo` to `alpha` at `hi`
        // (normalized intensity), sampled every `stepVoxels`; `mip` switches
        // to a maximum projection.
        void setTransfer(float lo, float hi, float alpha, float stepVoxels, bool mip);
        void setMethodText(const std::string& text) { method_ = text; }   // "Ray casting"

        // From dragging / the sliders / the presets, the clip bar, the wheel.
        std::function<void(double, double)> orientationChanged;
        std::function<void(double, double)> clipChanged;
        std::function<void(double)> zoomChanged;

        // Renders (when anything changed) and draws the view in (min, max)
        // of the current window, with its overlays and controls.
        void draw(ImVec2 min, ImVec2 max);
        // The rendering alone, RGBA rows top to bottom; false before the first one.
        bool grabImage(std::vector<std::uint8_t>& rgba, int& width, int& height);

    private:
        struct Gl;
        bool initGl();
        void uploadTextures();
        void uploadLabels();
        void render(int width, int height);
        void drawControls(ImVec2 min, ImVec2 max);
        void applyOrientation(double yaw, double pitch, bool emitSignal);

        std::unique_ptr<Gl> gl_;
        std::vector<ReducedVolume> textures_;
        Index vz_ = 0, vy_ = 0, vx_ = 0;   // full-resolution grid of the bricks
        std::string preparing_;
        std::uint64_t key_ = 0, uploadedKey_ = 0;
        const std::uint32_t* labels_ = nullptr;
        std::shared_ptr<const void> labelsOwner_;   // what labels_ points into
        Index lz_ = 0, ly_ = 0, lx_ = 0;
        std::uint64_t labelsKey_ = 0, uploadedLabelsKey_ = 0;
        float labelOpacity_ = 0.45f;
        std::uint32_t labelOnly_ = 0;   // non-zero: that label alone
        std::array<double, 3> voxelUm_{0.1, 0.1, 0.2};
        double yaw_ = 35.0, pitch_ = 22.0, zoom_ = 1.0;
        double clipLo_ = 0.0, clipHi_ = 1.0;
        bool box_ = true;
        float tfLo_ = 0.05f, tfHi_ = 0.6f, tfAlpha_ = 0.9f, stepVoxels_ = 0.5f;
        bool mip_ = false;
        std::string method_ = "Ray casting";
        bool glTried_ = false, glOk_ = false;
        std::string glError_;
        bool dragging_ = false;
        std::string renderedState_;   // what the target holds; re-rendered when it differs
    };

} // namespace sirius::app::gui

#endif // SIRIUS_IMGUI_VIEWER_VOLUME_VIEW_HPP
