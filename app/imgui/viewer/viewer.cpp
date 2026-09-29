#include "imgui/viewer/viewer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <set>
#include <tuple>

#include <imgui.h>
#include <imgui_impl_opengl3.h>

#include <sirius/constants.hpp>

#include "core/array_source.hpp"
#include "core/labels.hpp"
#include "core/operation.hpp"
#include "core/ops/contrast.hpp"
#include "core/tracks.hpp"
#include "imgui/app.hpp"
#include "imgui/gl.hpp"
#include "imgui/settings.hpp"
#include "imgui/strings.hpp"
#include "imgui/theme.hpp"
#include "imgui/viewer/dims_strip.hpp"
#include "imgui/viewer/display_model.hpp"
#include "imgui/viewer/slice_pane.hpp"
#include "imgui/viewer/trace.hpp"
#include "imgui/viewer/track_overlay.hpp"
#include "imgui/viewer/viewer_constants.hpp"
#include "imgui/viewer/viewer_loader.hpp"
#include "imgui/viewer/volume_view.hpp"
#include "imgui/widgets/controls.hpp"

namespace sirius::app::gui {

    using theme::px;

    namespace {
        using viewer::kButtonZoomFactor;
        using viewer::kMaxZoom;
        using viewer::kMinZoom;
        using viewer::kPlayIntervalMs;
        using viewer::kWheelZoomBase;

        // The ortho splitters' saved balance: the side column's width and the
        // bottom row's height, in design pixels.
        const char* const kOrthoRowsKey = "viewer/orthoRows";
        const char* const kOrthoColsKey = "viewer/orthoCols";

        const char* const kNoCursor = "cursor \xE2\x80\x94";   // "cursor —"

        // "(V)", "(Shift+A)": the shortcut a tooltip advertises, in the
        // platform's own notation.
        std::string shortcutSuffix(ImGuiKeyChord keys) {
            const std::string text = shortcutText(keys);
            return text.empty() ? std::string() : " (" + text + ")";
        }

        std::string num2(int index) { return Step::number(index); }

        Index clampIndex(double v, Index n) {
            return std::clamp<Index>(static_cast<Index>(std::floor(v)), 0, std::max<Index>(n - 1, 0));
        }

        ImU32 colorFromHexString(const std::string& hex) {
            const std::array<float, 3> c = colorFromHex(hex);
            return theme::fromFloat(c);
        }

        // The paint tools that draw a brush outline over the XY pane (Lasso
        // paints with the brush); the others are single clicks.
        bool brushLike(const ViewState& s) {
            return s.tool == ViewerTool::Paint &&
                   (s.paintTool == PaintTool::Brush || s.paintTool == PaintTool::Erase || s.paintTool == PaintTool::Lasso);
        }

        // One coloured run of the "Viewing 05 Contrast …" line.
        struct Run {
            std::string text;
            ImU32 color;
            theme::Weight weight = theme::Weight::Regular;
        };

        // Text drawn rotated to run upwards from `bottomLeft` (the tool
        // strip's zoom readout): the glyphs' tops point left.
        void drawVerticalText(ImDrawList* dl, ImVec2 bottomLeft, float thickness, const std::string& s, float designPx, ImU32 color) {
            const ImVec2 ts = theme::textSize(s, designPx);
            const int first = dl->VtxBuffer.Size;
            // laid out horizontally at the origin, then turned: (u, v) -> (x0 + v, y1 - u)
            // at the origin nothing is inside the clip rectangle and the glyphs
            // would be culled: lay them out without one
            dl->PushClipRect(ImVec2(-1e6f, -1e6f), ImVec2(1e6f, 1e6f), false);
            widgets::drawText(dl, ImVec2(0, 0), s, designPx, color);
            dl->PopClipRect();
            const float dv = std::floor((thickness - ts.y) * 0.5f);
            for (int i = first; i < dl->VtxBuffer.Size; ++i) {
                ImVec2& p = dl->VtxBuffer[i].pos;
                const float u = p.x, v = p.y;
                p = ImVec2(bottomLeft.x + dv + v, bottomLeft.y - u);
            }
        }
    } // namespace

    // -------------------------------------------------------------------------
    // Impl
    // -------------------------------------------------------------------------

    struct Viewer::Impl {
        App& app;
        Bridge& bridge;
        Workbench& wb;

        // views
        SlicePane xy{SlicePane::Kind::XY, "xyPane"};
        SlicePane yz{SlicePane::Kind::YZ, "yzPane"};
        SlicePane xz{SlicePane::Kind::XZ, "xzPane"};
        SlicePane mip{SlicePane::Kind::MIP, "mipPane"};
        SlicePane cmpLeft{SlicePane::Kind::Compare, "compareRawPane"};
        SlicePane cmpRight{SlicePane::Kind::Compare, "compareStepPane"};
        VolumeView volume;
        DimsStrip dims;
        // The ortho grid's balance: the YZ / MIP column's width and the XZ /
        // MIP row's height, design pixels; the first layout sets the design's
        // targets (or the saved balance) and raises orthoLoaded.
        float orthoCol = 0.0f, orthoRow = 0.0f;
        bool orthoLoaded = false;
        bool orthoSaved = false;
        SlicePane* focused = nullptr;   // the pane that has the keyboard
        bool contextMenuPending = false;

        // rendering state
        DisplayModel model, rawModel;
        // Volumes, projections, exact ranges and the 3D bricks are produced
        // here, never on the GUI thread.
        ViewerLoader loader;
        std::uint64_t volumeKey = 0;    // the reduction the 3D view is waiting for
        std::uint64_t shownVolumeKey = 0;   // the reduction whose bricks the 3D view has
        std::string sliceNotice;        // "Loading 37%" for the panes that need a volume
        bool loadActive = false;        // volume decode in flight; drives the status bar
        double loadFrac = 0.0;
        // (output, c, t) reads that threw: not asked for again until the
        // displayed output changes.
        std::set<std::tuple<const StepOutput*, Index, Index>> failedVolumes;
        int displayIndex = -1;
        Image xyImg, xzImg, yzImg, mipImg, cmpLeftImg, cmpRightImg;
        int xyFactor = 1, xzFactor = 1, yzFactor = 1, mipFactor = 1, cmpFactor = 1, cmpLeftFactor = 1;
        // Rendered voxel regions: the visible part plus a margin. Panning out
        // of them, or a factor change, re-renders.
        RectI xyRegion, xzRegion, yzRegion, cmpRegion, cmpLeftRegion;
        // One image pixel per this many voxels, from the finer of the two
        // pane axes: the one with more screen pixels per voxel, which on XZ /
        // YZ is z, stretched by the voxel aspect. The factor applies to both
        // axes, so taking it from the coarser one (std::min, as this did)
        // dropped z planes that had a screen pixel each: at 0.15 / 0.75 um,
        // zoomed out, four planes of five.
        static int paneFactor(const SlicePane::View& v) { return paneFactorFor(v.zx, v.zy); }
        static int paneFactorFor(double zx, double zy) {
            const double p = std::max(zx, zy);
            return std::max(1, static_cast<int>(std::floor(1.0 / std::max(p, 1e-6))));
        }
        // The voxels a pane shows now (no margin), for containment checks.
        static RectI visibleVoxels(const SlicePane& pane, Index cols, Index rows) {
            const DPoint a = pane.toVoxel(DPoint(0, 0));
            const DPoint b = pane.toVoxel(DPoint(pane.width(), pane.height()));
            const int x0 = std::clamp(static_cast<int>(std::floor(std::min(a.x, b.x))), 0, static_cast<int>(std::max<Index>(cols - 1, 0)));
            const int y0 = std::clamp(static_cast<int>(std::floor(std::min(a.y, b.y))), 0, static_cast<int>(std::max<Index>(rows - 1, 0)));
            const int x1 = std::clamp(static_cast<int>(std::ceil(std::max(a.x, b.x))), x0 + 1, std::max(x0 + 1, static_cast<int>(cols)));
            const int y1 = std::clamp(static_cast<int>(std::ceil(std::max(a.y, b.y))), y0 + 1, std::max(y0 + 1, static_cast<int>(rows)));
            return RectI{x0, y0, x1 - x0, y1 - y0};
        }
        // What to render: the visible voxels grown by a quarter of their size
        // on every side, aligned to the factor; the whole plane when that is
        // about as big anyway.
        static RectI renderRegion(const SlicePane& pane, int factor, Index cols, Index rows) {
            if (cols <= 0 || rows <= 0) return RectI{};
            const RectI vis = visibleVoxels(pane, cols, rows);
            const int mx = std::max(factor, vis.w / 4), my = std::max(factor, vis.h / 4);
            int x0 = std::max(0, vis.x - mx), y0 = std::max(0, vis.y - my);
            int x1 = std::min(static_cast<int>(cols), vis.x + vis.w + mx);
            int y1 = std::min(static_cast<int>(rows), vis.y + vis.h + my);
            x0 = (x0 / factor) * factor;
            y0 = (y0 / factor) * factor;
            x1 = std::min(static_cast<int>(cols), ((x1 + factor - 1) / factor) * factor);
            y1 = std::min(static_cast<int>(rows), ((y1 + factor - 1) / factor) * factor);
            const RectI r{x0, y0, x1 - x0, y1 - y0};
            const RectI whole{0, 0, static_cast<int>(cols), static_cast<int>(rows)};
            return static_cast<double>(r.w) * r.h >= 0.7 * static_cast<double>(cols) * static_cast<double>(rows) ? whole : r;
        }
        // `want` raised until the region rendered at that factor fits in a
        // texture: one longer than GL_MAX_TEXTURE_SIZE on a side is not
        // created at all. The region, not the plane, is the bound: a plane
        // wider than the limit would otherwise be sub-sampled at every zoom.
        // The whole plane fits at ceil(n / limit), and no region is larger.
        static int textureFactor(const SlicePane& pane, int want, Index cols, Index rows) {
            const int limit = maxTextureSize();
            const int whole = static_cast<int>(std::max<Index>(1, (std::max(cols, rows) + limit - 1) / limit));
            int f = std::max(want, 1);
            for (; f < whole; ++f) {
                const RectI r = renderRegion(pane, f, cols, rows);
                if ((r.w + f - 1) / f <= limit && (r.h + f - 1) / f <= limit) break;
            }
            return f;
        }
        struct Dirty {
            bool xy = true, xz = true, yz = true, mip = true, cmp = true, vol = true;
        } dirty;
        bool updateQueued = false;
        ViewState prev;
        bool havePrev = false;

        // What the workbench has done since the last frame: the revisions
        // seen, and the steps whose parameters changed (the bridge's
        // stepChanged, which names the step).
        Revisions seen;
        std::vector<int> changedSteps;
        int stepConnection = 0;

        // interaction: annotations live on the plane (t, z) they were drawn
        // on; a measurement or ROI in progress is shown lighter until committed.
        struct Annotation {
            SlicePane::Annotation::Kind kind = SlicePane::Annotation::Kind::Measure;
            std::vector<DPoint> points;
            DRect rect;
            Index t = 0, z = 0;
        };
        std::vector<Annotation> annotations;
        std::vector<DPoint> measure;   // pending measurement
        DRect roi;                     // pending box
        DPoint roiStart;
        std::string measureText(const std::vector<DPoint>& points) const;
        std::string roiText(const DRect& r) const;
        void pushAnnotations();        // to the XY pane, filtered by the current plane
        void commitMeasure();
        void clearAnnotations();
        void drawXYContextMenu();
        std::array<Index, 3> splitA{};
        bool splitPending = false;
        std::uint32_t mergeFirst = 0;
        DPoint lastPaint;
        bool painting = false;
        std::uint64_t labelsVersion = 0;   // bumps on every label edit: the 3D label texture follows
        // Trajectory paths per plane, rebuilt when the labels (or the solo
        // track) change and handed to the panes on every update.
        struct TrackPaths {
            // weak: a strong reference here would make every label edit copy
            // the index (LabelVolume shares it copy-on-write)
            std::weak_ptr<const TrackIndex> index;
            std::uint64_t version = ~std::uint64_t{0};
            std::uint32_t only = 0;
            std::shared_ptr<const std::vector<TrackPath>> xy, xz, yz;
        } trackCache;
        void refreshTracks();
        void followTrackIntoView();   // zoomed in and following: keep the crosshair on screen
        bool followPending = false;
        bool playing = false;
        double playLast = 0.0;
        std::string cursorText = kNoCursor;
        std::string zoomText = "100 %";

        // chrome state
        float rightGroupW = 0.0f;   // the toolbar's right-hand group, as wide as it was last frame

        // what grabView replays: the draw list and the part of it the views cover
        ImDrawList* grabList = nullptr;
        ImVec2 grabMin{0, 0}, grabMax{0, 0};
        std::uint64_t grabFrame = 0;

        Impl(App& a)
            : app(a), bridge(a.bridge()), wb(a.wb()),
              loader([&b = a.bridge()](std::function<void()> fn) { b.post(std::move(fn)); }) {}

        // --- helpers ---------------------------------------------------------
        const ViewState& vs() const { return wb.viewState(); }
        Index nx() const { return model.dims().x; }
        Index ny() const { return model.dims().y; }
        Index nz() const { return model.dims().z; }
        Index nt() const { return model.dims().t; }
        double zAspect() const {
            if (!vs().physicalZ) return 1.0;   // voxel grid: one row per plane
            const auto& v = model.meta().voxelUm;
            return v[0] > 0.0 && v[2] > 0.0 ? v[2] / v[0] : 1.0;
        }
        Index curZ() const { return std::clamp<Index>(vs().z, 0, std::max<Index>(nz() - 1, 0)); }
        Index curT() const { return std::clamp<Index>(vs().t, 0, std::max<Index>(nt() - 1, 0)); }
        Index curX() const { return std::clamp<Index>(vs().cx, 0, std::max<Index>(nx() - 1, 0)); }
        Index curY() const { return std::clamp<Index>(vs().cy, 0, std::max<Index>(ny() - 1, 0)); }
        bool probe() const { return vs().tool == ViewerTool::Probe; }

        bool paintAvailable() const {
            if (model.hasLabels()) return true;
            for (const Step& s : wb.pipeline().steps()) {
                if (!s.enabled) continue;
                if (const Operation* op = findOperation(s.kind))
                    if (op->info().producesLabels || op->info().needsLabels) return true;
            }
            return false;
        }

        void connect();
        void sync();                   // what the workbench did since the last look
        void rebuildOutput();          // display output changed
        void applyLivePreview();       // window / gamma of a previewed step
        bool previewing = false;
        // The output whose preview window could not be computed, logged
        // once: every rebuild tries again.
        std::weak_ptr<const StepOutput> previewFailed;
        void refreshChrome();
        void refreshDims();
        void refreshHints();
        void applyViewStateDiff(const ViewState& s);
        void scheduleUpdate() { updateQueued = true; }
        void applyPending();           // sync, then render what is dirty
        void applyDirty();
        void layoutPanes();
        void renderVolume();
        // Asks the loader for whatever (c, t) volumes the visible channels
        // still need; the aggregate state of those channels.
        DisplayModel::VolumeState ensureVolumes(DisplayModel& m, Index t);
        // Drops the reads of time points no longer on screen (nor read
        // ahead for play), so they do not hold up the one that is.
        void retainVolumes();
        // Visible channels: volume is in RAM (slices can draw), MIP is cached.
        void volumeReadiness(const DisplayModel& m, Index t, bool& haveVol, bool& haveMip) const;
        void onVolumeReady(const ViewerLoader::Volume& v);
        void onReductionReady(const ViewerLoader::Reduction& r);
        void onVolumeProgress(double fraction, const std::string& message);
        void beginLoad();
        void endLoad();
        bool canPaint() const { return wb.canEdit(); }
        // Compare's own plane when View ▸ Sync Z / T is off.
        Index compareZ() const;
        Index compareT() const;
        // The raw pane has voxels of its own size (layoutPanes scales it by
        // the voxel-size ratio): the raw plane at compareZ()'s physical
        // position, and a point of the raw pane in the step's voxels.
        Index compareRawZ() const;
        DPoint rawToStep(const DPoint& rawVoxel) const;
        Index cmpZ = 0, cmpT = 0;      // the raw pane's plane while unsynced
        void setZoomPan(double zoom, double panX, double panY);
        // `anchor` is in the local coordinates of `pane` (the XY pane when
        // null) and stays over the same point of the data.
        void zoomAround(double factor, const DPoint& anchor, const SlicePane* pane = nullptr);
        void fit();
        void setCursorFor(const ViewState& s);
        void setPlaying(bool on);

        // tools
        void onXYPressed(const DPoint& v, int b, ImGuiKeyChord m);
        void onXYDragged(const DPoint& v, const DPoint& delta, int b, ImGuiKeyChord m);
        void onXYReleased(const DPoint& v, int b, ImGuiKeyChord m, bool moved);
        void paintAt(const DPoint& v, bool erase);
        std::uint32_t labelAt(Index z, Index y, Index x) const;
        void hover(SlicePane::Kind kind, const DPoint& v, bool rawPane = false);

        // drawing
        void draw();
        void layoutOrtho(ImVec2 min, ImVec2 max);
        void orthoSplitters(ImVec2 min, ImVec2 max);
        void drawToolbar(ImVec2 min, ImVec2 max);
        void drawToolStrip(ImVec2 min, ImVec2 max);
        void handleKeys();
        bool grab(std::vector<std::uint8_t>& rgba, int& width, int& height);
    };

    // --- wiring ------------------------------------------------------------------------

    void Viewer::Impl::connect() {
        // XY: the tools
        xy.onPress = [this](DPoint v, int b, ImGuiKeyChord m) { onXYPressed(v, b, m); };
        xy.onContextMenu = [this](ImVec2, DPoint) { contextMenuPending = true; };
        xy.onDrag = [this](DPoint v, DPoint d, int b, ImGuiKeyChord m) { onXYDragged(v, d, b, m); };
        xy.onRelease = [this](DPoint v, int b, ImGuiKeyChord m, bool moved) { onXYReleased(v, b, m, moved); };
        xy.onDoubleClick = [this](DPoint, ImGuiKeyChord) {
            if (vs().tool == ViewerTool::Navigate || vs().tool == ViewerTool::Probe) fit();
        };
        xy.onWheel = [this](DPoint s, double steps, ImGuiKeyChord) { zoomAround(std::pow(kWheelZoomBase, steps), s); };
        xy.onHover = [this](DPoint v) { hover(SlicePane::Kind::XY, v); };

        // YZ / XZ: probe moves the crosshair (and z); navigate pans along the shared axis
        yz.onPress = [this](DPoint v, int b, ImGuiKeyChord) {
            if (b == ImGuiMouseButton_Left && probe() && model.valid()) wb.setCrosshair(curX(), clampIndex(v.y, ny()), clampIndex(v.x, nz()));
        };
        yz.onDrag = [this](DPoint v, DPoint d, int b, ImGuiKeyChord) {
            if (b == ImGuiMouseButton_Middle || (b == ImGuiMouseButton_Left && vs().tool == ViewerTool::Navigate))
                setZoomPan(vs().zoom, vs().panX, vs().panY + d.y);
            else if (b == ImGuiMouseButton_Left && probe() && model.valid())
                wb.setCrosshair(curX(), clampIndex(v.y, ny()), clampIndex(v.x, nz()));
        };
        yz.onWheel = [this](DPoint s, double steps, ImGuiKeyChord) {
            zoomAround(std::pow(kWheelZoomBase, steps), DPoint(xy.width() / 2.0, s.y));
        };
        yz.onHover = [this](DPoint v) { hover(SlicePane::Kind::YZ, v); };

        xz.onPress = [this](DPoint v, int b, ImGuiKeyChord) {
            if (b == ImGuiMouseButton_Left && probe() && model.valid()) wb.setCrosshair(clampIndex(v.x, nx()), curY(), clampIndex(v.y, nz()));
        };
        xz.onDrag = [this](DPoint v, DPoint d, int b, ImGuiKeyChord) {
            if (b == ImGuiMouseButton_Middle || (b == ImGuiMouseButton_Left && vs().tool == ViewerTool::Navigate))
                setZoomPan(vs().zoom, vs().panX + d.x, vs().panY);
            else if (b == ImGuiMouseButton_Left && probe() && model.valid())
                wb.setCrosshair(clampIndex(v.x, nx()), curY(), clampIndex(v.y, nz()));
        };
        xz.onWheel = [this](DPoint s, double steps, ImGuiKeyChord) {
            zoomAround(std::pow(kWheelZoomBase, steps), DPoint(s.x, xy.height() / 2.0));
        };
        xz.onHover = [this](DPoint v) { hover(SlicePane::Kind::XZ, v); };
        mip.onPress = [this](DPoint v, int b, ImGuiKeyChord) {
            if (b == ImGuiMouseButton_Left && probe() && model.valid()) wb.setCrosshair(clampIndex(v.x, nx()), clampIndex(v.y, ny()), curZ());
        };
        mip.onHover = [this](DPoint v) { hover(SlicePane::Kind::MIP, v); };

        for (SlicePane* p : {&xy, &yz, &xz, &mip, &cmpLeft, &cmpRight}) {
            p->onExit = [this] { cursorText = kNoCursor; };
            // Arrow keys in a focused pane move the crosshair over that
            // pane's own axes; page up / down step the third one.
            const SlicePane::Kind kind = p->kind();
            p->onKeyNavigate = [this, kind](int dc, int dr, int dd) {
                if (!model.valid()) return;
                Index x = curX(), y = curY(), z = curZ();
                switch (kind) {
                    case SlicePane::Kind::YZ:
                        z += dc;
                        y += dr;
                        x += dd;
                        break;
                    case SlicePane::Kind::XZ:
                        x += dc;
                        z += dr;
                        y += dd;
                        break;
                    default:
                        x += dc;
                        y += dr;
                        z += dd;
                        break;
                }
                wb.setCrosshair(x, y, z);   // clamps all three
            };
        }

        // compare panes share the XY transform; the raw pane's voxels are
        // the raw data's, so what it reports goes through rawToStep()
        for (SlicePane* p : {&cmpLeft, &cmpRight}) {
            const bool raw = p == &cmpLeft;
            p->onDrag = [this, raw](DPoint v, DPoint d, int b, ImGuiKeyChord) {
                if (b == ImGuiMouseButton_Middle || (b == ImGuiMouseButton_Left && vs().tool == ViewerTool::Navigate)) {
                    setZoomPan(vs().zoom, vs().panX + d.x, vs().panY + d.y);
                } else if (b == ImGuiMouseButton_Left && probe() && model.valid()) {
                    const DPoint sv = raw ? rawToStep(v) : v;
                    wb.setCrosshair(clampIndex(sv.x, nx()), clampIndex(sv.y, ny()), curZ());
                }
            };
            p->onPress = [this, raw](DPoint v, int b, ImGuiKeyChord) {
                if (b != ImGuiMouseButton_Left || !probe() || !model.valid()) return;
                const DPoint sv = raw ? rawToStep(v) : v;
                wb.setCrosshair(clampIndex(sv.x, nx()), clampIndex(sv.y, ny()), curZ());
            };
            p->onWheel = [this, raw, p](DPoint s, double steps, ImGuiKeyChord m) {
                // With View ▸ Sync Z / T off, shift + wheel over the raw pane
                // moves its own plane -- the point of switching the sync off.
                if (raw && !vs().syncZT && (m & ImGuiMod_Shift)) {
                    cmpZ = std::clamp<Index>(cmpZ + static_cast<Index>(steps > 0 ? 1 : -1), 0, std::max<Index>(nz() - 1, 0));
                    dirty.cmp = true;
                    layoutPanes();
                    scheduleUpdate();
                    return;
                }
                zoomAround(std::pow(kWheelZoomBase, steps), s, p);
            };
            p->onDoubleClick = [this](DPoint, ImGuiKeyChord) { fit(); };
            p->onHover = [this, raw](DPoint v) { hover(SlicePane::Kind::Compare, v, raw); };
        }

        // volume view
        volume.orientationChanged = [this](double yaw, double pitch) {
            ViewState s = vs();
            s.yaw = yaw;
            s.pitch = pitch;
            wb.setViewState(s);
        };
        volume.clipChanged = [this](double lo, double hi) {
            ViewState s = vs();
            s.clipZ = {lo, hi};
            wb.setViewState(s);
        };

        // dims strip
        dims.zRequested = [this](Index z) { wb.setZ(z); };
        dims.tRequested = [this](Index t) { wb.setT(t); };
        dims.playToggled = [this](bool on) { setPlaying(on); };

        // the loader's results
        loader.volumeReady = [this](const ViewerLoader::Volume& v) { onVolumeReady(v); };
        loader.reductionReady = [this](const ViewerLoader::Reduction& r) { onReductionReady(r); };
        loader.volumeProgress = [this](double f, const std::string& msg) { onVolumeProgress(f, msg); };

        // which step changed, when one did (Load ▸ tile, a live preview's window)
        stepConnection = bridge.stepChanged.connect([this](int index) { changedSteps.push_back(index); });
    }

    // The bridge's revisions tell what moved since the last look, and the
    // handler for each change runs.
    void Viewer::Impl::sync() {
        const Revisions r = bridge.rev();
        bool rebuild = r.dataset != seen.dataset || r.viewedStep != seen.viewedStep || r.outputs != seen.outputs || r.pipeline != seen.pipeline;
        if (r.runState != seen.runState) {
            // While a run holds the pipeline the workbench refuses label edits
            // (Workbench::canEdit): the paint tools go with it.
            if (wb.running()) {
                if (painting) wb.endPaintStroke();
                painting = false;
            } else {
                rebuild = true;   // a finished run
            }
            refreshChrome();
        }
        if (r.labels != seen.labels) {
            labelsVersion += r.labels - seen.labels;
            dirty.xy = dirty.xz = dirty.yz = dirty.cmp = dirty.vol = true;
            scheduleUpdate();
        }
        std::vector<int> steps;
        steps.swap(changedSteps);
        for (int index : steps) {
            // (index 0, Load ▸ tile edited elsewhere: the toolbar reads it each frame)
            if (index == wb.viewedIndex()) {
                if (previewing || wb.viewedIsLivePreview()) {
                    rebuild = true;   // re-applies the preview window
                    continue;
                }
                refreshChrome();
                dirty.vol = true;
                scheduleUpdate();
            }
        }
        const bool viewChanged = r.viewState != seen.viewState;
        seen = r;
        if (viewChanged) applyViewStateDiff(vs());
        if (rebuild) rebuildOutput();
        if (followPending) {
            followPending = false;
            followTrackIntoView();
        }
    }

    // --- asynchronous volumes ------------------------------------------------------

    DisplayModel::VolumeState Viewer::Impl::ensureVolumes(DisplayModel& m, Index t) {
        if (!m.valid()) return DisplayModel::VolumeState::TooLarge;
        bool wanted = false, tooLarge = false;
        for (Index c = 0; c < m.dims().c; ++c) {
            if (!vs().channelOn(c)) continue;
            switch (m.volumeState(c, t)) {
                case DisplayModel::VolumeState::Ready: break;
                case DisplayModel::VolumeState::Wanted:
                    if (failedVolumes.count({m.output().get(), c, t})) break;
                    wanted = true;
                    if (loader.prepare(m.output(), c, t) && !m.output()->array) beginLoad();
                    break;
                case DisplayModel::VolumeState::TooLarge: tooLarge = true; break;
            }
        }
        if (tooLarge) return DisplayModel::VolumeState::TooLarge;
        return wanted ? DisplayModel::VolumeState::Wanted : DisplayModel::VolumeState::Ready;
    }

    void Viewer::Impl::retainVolumes() {
        // The loader reads first in first out: every frame scrubbed or played
        // past would be read in full before the one the user stopped on.
        std::vector<Index> keep{curT()};
        if (playing && nt() > 1) keep.push_back((curT() + 1) % nt());
        const bool shared = rawModel.output() == model.output();
        if (shared) keep.push_back(compareT());   // one output, one set of reads
        loader.retain(model.output().get(), keep);
        if (!shared) loader.retain(rawModel.output().get(), {compareT()});
        // a dropped read no longer counts: when it was the last one, nothing else ends the load
        if (!loader.busy()) endLoad();
    }

    void Viewer::Impl::volumeReadiness(const DisplayModel& m, Index t, bool& haveVol, bool& haveMip) const {
        haveVol = haveMip = true;
        if (!m.valid()) {
            haveVol = haveMip = false;
            return;
        }
        for (Index c = 0; c < m.dims().c; ++c) {
            if (!vs().channelOn(c)) continue;
            if (!m.volumeIfReady(c, t)) haveVol = false;
            if (!m.mipIfReady(c, t)) haveMip = false;
        }
    }

    void Viewer::Impl::onVolumeReady(const ViewerLoader::Volume& v) {
        // Both models show the Load output while it is the one viewed (or
        // the viewed step has not run): the read serves both.
        const bool forModel = v.out == model.output(), forRaw = v.out == rawModel.output();
        if (!forModel && !forRaw) {
            if (!loader.busy()) endLoad();
            return;   // the viewer moved on: drop it
        }
        const bool onScreen = (forModel && v.t == curT()) || (forRaw && v.t == compareT());
        if (!v.ok) {
            failedVolumes.insert({v.out.get(), v.c, v.t});
            sliceNotice = "could not read the volume";
            refreshHints();
            wb.logLine("Viewer: " + v.error);
            if (onScreen) {   // the panes waiting for it show that instead
                dirty = Dirty{};
                scheduleUpdate();
            }
            if (!loader.busy()) endLoad();
            return;
        }
        if (ScopedTrace::enabled())
            std::fprintf(stderr, "view volume c%lld t%lld ready in %lld us (%s)\n", static_cast<long long>(v.c), static_cast<long long>(v.t),
                         v.micros, v.volume ? "read" : "in memory");
        if (forModel) {
            std::shared_ptr<Buffer<float>> vol = v.volume;
            // A late lazy read must not evict the time point on screen; its MIP
            // is still worth keeping for the next loop of play. The exception is
            // the frame play asked for ahead of time: dropping that one made play
            // read every frame of a lazy source twice.
            const Index shown = curT();
            const bool readAhead = playing && nt() > 1 && v.t == (shown + 1) % nt();
            if (v.t != shown && !readAhead) vol.reset();
            model.installVolume(v.c, v.t, std::move(vol), v.mip, v.lo, v.hi, v.t == shown ? Index{-1} : shown);
        }
        if (forRaw) {
            // The raw pane draws planes: the projection and the exact range
            // are what it wants. With one output the volume stays with the
            // model alone, or the raw model would keep up to 3 GiB alive
            // after the model moved on to another output.
            const Index shown = compareT();
            rawModel.installVolume(v.c, v.t, (forModel || v.t != shown) ? nullptr : v.volume, v.mip, v.lo, v.hi,
                                   v.t == shown ? Index{-1} : shown);
        }
        // Cache MIPs for every t (play loops). Only the current frame needs a
        // redraw; an older job that finished late is still worth keeping.
        if (onScreen) {
            dirty = Dirty{};
            scheduleUpdate();
        }
        if (!loader.busy()) endLoad();
    }

    void Viewer::Impl::onVolumeProgress(double fraction, const std::string& message) {
        if (!loadActive) beginLoad();
        std::string tail = message;
        if (startsWith(tail, "reading ")) tail = tail.substr(8);
        sliceNotice = format("Loading %d%%", static_cast<int>(std::clamp(fraction, 0.0, 1.0) * 100.0 + 0.5));
        if (!tail.empty()) sliceNotice += " \xC2\xB7 " + tail;
        refreshHints();
        // Only while the 3D view waits for this frame: play's read-ahead or
        // the raw pane's read would otherwise write over a finished rendering.
        bool haveVol = false, haveMip = false;
        volumeReadiness(model, curT(), haveVol, haveMip);
        if (!haveVol) volume.setPreparing(sliceNotice);
        loadFrac = fraction;
    }

    void Viewer::Impl::beginLoad() {
        if (loadActive) return;
        loadActive = true;
        loadFrac = 0.0;
        if (sliceNotice.empty()) sliceNotice = "Loading\xE2\x80\xA6";
    }

    void Viewer::Impl::endLoad() {
        if (!loadActive) return;
        loadActive = false;
    }

    void Viewer::Impl::onReductionReady(const ViewerLoader::Reduction& r) {
        if (r.key != volumeKey) return;
        if (ScopedTrace::enabled())
            std::fprintf(stderr, "view 3d reduction of %d channels in %lld us\n", static_cast<int>(r.channels.size()), r.micros);
        volume.setTextures(r.key, r.channels, model.meta().voxelUm, nz(), ny(), nx());
        shownVolumeKey = r.key;
    }

    // --- output --------------------------------------------------------------------

    void Viewer::Impl::rebuildOutput() {
        int actual = -1;
        std::shared_ptr<const StepOutput> out = wb.hasDataset() ? wb.displayOutput(&actual) : nullptr;
        const bool changed = out != model.output();
        model.setOutput(out);
        displayIndex = actual;
        // the raw side of Compare: the Load step's output, or the bare source
        std::shared_ptr<const StepOutput> raw = wb.hasDataset() ? wb.output(0) : nullptr;
        if (!raw && wb.hasDataset() && wb.source()) {
            auto so = std::make_shared<StepOutput>();
            so->meta = wb.dataset();
            so->source = wb.source();
            raw = so;
        }
        if (raw != rawModel.output()) {
            rawModel.setOutput(raw);
            dirty.cmp = true;
        }
        if (changed) {
            // results for the old output are no longer wanted
            loader.cancelAll();
            endLoad();
            volumeKey = 0;
            // The 3D labels' owner is the old output, image array and all:
            // holding it until the 3D view next renders kept it alive.
            volume.clearLabels();
            // The re-slices of the old output are other data: they must not
            // stand in while the new volume loads (a new time point of the
            // same output keeps its predecessor's until then).
            for (SlicePane* p : {&xz, &yz, &mip}) p->clearContent();
            xzRegion = yzRegion = RectI{};
            sliceNotice.clear();
            failedVolumes.clear();
            dirty = Dirty{};
            measure.clear();
            roi = DRect{};
            annotations.clear();
            splitPending = false;
            mergeFirst = 0;
        }
        applyLivePreview();
        refreshChrome();
        refreshDims();
        layoutPanes();
        scheduleUpdate();
    }

    // A live-preview step (Contrast) that has not run is shown on its input
    // through its own window and gamma, so edits update the panes at once.
    void Viewer::Impl::applyLivePreview() {
        const bool was = previewing;
        previewing = wb.viewedIsLivePreview() && model.valid();
        if (!previewing) {
            if (was) {
                model.resetWindows();
                dirty = Dirty{};
            }
            return;
        }
        const Step& st = wb.pipeline().at(wb.viewedIndex());
        const StepInput in = model.output()->asInput();
        try {
            for (Index c = 0; c < model.dims().c; ++c) {
                const ContrastWindow w = contrastWindow(in, st.params, c, 8);
                model.setWindow(c, DisplayWindow{w.lo, w.hi, w.gamma});
            }
        } catch (const std::exception& e) {
            // The automatic window samples planes of the input, which a lazy
            // source can fail to read: the panes keep their own windows.
            model.resetWindows();
            if (previewFailed.lock() != model.output()) {
                previewFailed = model.output();
                wb.logLine(std::string("Viewer: contrast preview: ") + e.what());
            }
        }
        dirty = Dirty{};
    }

    void Viewer::Impl::refreshChrome() {
        const ViewState& s = vs();
        zoomText = format("%ld %%", std::lround(s.zoom * 100.0));
        refreshHints();
        setCursorFor(s);
        volume.setBoundingBox(s.boundingBox);
        volume.setOrientation(s.yaw, s.pitch);
        volume.setClip(s.clipZ[0], s.clipZ[1]);
        // transfer function from a volume reconstruction step's parameters
        const int viewed = wb.viewedIndex();
        if (viewed >= 0 && viewed < wb.pipeline().size() && wb.pipeline().at(viewed).kind == "volrec") {
            const ParamSet& p = wb.pipeline().at(viewed).params;
            const std::string method = p.getString("method", "Ray casting");
            const bool isMip = method.find("ax") != std::string::npos || method.find("MIP") != std::string::npos;
            volume.setTransfer(static_cast<float>(p.getDouble("opacity_lo", 0.05)), static_cast<float>(p.getDouble("opacity_hi", 0.6)),
                               static_cast<float>(p.getDouble("opacity", 0.9)), static_cast<float>(p.getDouble("step", 0.5)), isMip);
            volume.setMethodText(method);
        } else {
            volume.setTransfer(0.05f, 0.6f, 0.9f, 0.5f, false);
            volume.setMethodText("Ray casting");
        }
    }

    void Viewer::Impl::refreshHints() {
        const ViewState& s = vs();
        if (s.tool != ViewerTool::Measure && !measure.empty()) {
            measure.clear();
            pushAnnotations();
        }
        if (s.tool != ViewerTool::Roi && !roi.isNull()) {
            roi = DRect{};
            pushAnnotations();
        }
        std::string hint;
        switch (s.tool) {
            case ViewerTool::Navigate: hint = "drag \xC2\xB7 pan   wheel \xC2\xB7 zoom   double-click \xC2\xB7 fit"; break;
            case ViewerTool::Probe: hint = "click \xC2\xB7 move crosshair   drag \xC2\xB7 follow   wheel \xC2\xB7 zoom"; break;
            case ViewerTool::Measure:
#ifdef __APPLE__
                hint = "click twice \xC2\xB7 distance   \xE2\x87\xA7 click \xC2\xB7 angle   right-click \xC2\xB7 clear";
#else
                hint = "click twice \xC2\xB7 distance   Shift click \xC2\xB7 angle   right-click \xC2\xB7 clear";
#endif
                break;
            case ViewerTool::Roi: hint = "drag \xC2\xB7 box   right-click \xC2\xB7 clear"; break;
            case ViewerTool::Paint: {
                if (!canPaint()) {
                    hint = "label edits are paused while a run is in progress";
                    break;
                }
                std::string what;
                switch (s.paintTool) {
                    case PaintTool::Brush: what = format("brush %d px \xC2\xB7 alt \xC2\xB7 erase \xC2\xB7 [ ] \xC2\xB7 size", s.brushPx); break;
                    case PaintTool::Erase: what = format("erase %d px \xC2\xB7 [ ] \xC2\xB7 size", s.brushPx); break;
                    case PaintTool::Fill: what = "click \xC2\xB7 fill region with the selected label"; break;
                    case PaintTool::Pick: what = "click \xC2\xB7 pick label"; break;
                    case PaintTool::Merge:
                        what = mergeFirst ? format("click the label to merge into %u", mergeFirst) : "click first label \xC2\xB7 then the second";
                        break;
                    case PaintTool::Split: what = splitPending ? "click the second seed" : "click two seeds inside one label"; break;
                    case PaintTool::Delete: what = "click \xC2\xB7 delete label"; break;
                    case PaintTool::Lasso: what = "lasso not available \xC2\xB7 painting with the brush"; break;
                }
                hint = what + " \xC2\xB7 crosshair locked";
                break;
            }
        }
        if (model.volumeTooLarge()) {
            yz.setMessage("volume too large for re-slicing");
            xz.setMessage("volume too large for re-slicing");
            mip.setMessage("volume too large");
        } else {
            bool haveVol = false, haveMip = false;
            volumeReadiness(model, curT(), haveVol, haveMip);
            xz.setMessage(haveVol ? std::string() : sliceNotice);
            yz.setMessage(haveVol ? std::string() : sliceNotice);
            mip.setMessage(haveMip ? std::string() : sliceNotice);
        }
        xy.setHint(hint);
        cmpRight.setHint(hint);
        const bool brush = brushLike(s);
        xy.setBrushCursor(brush, s.brushPx / 2.0);
        cmpRight.setBrushCursor(brush, s.brushPx / 2.0);
    }

    // Dear ImGui has no open / closed hand or cross cursors: navigate shows
    // the four-way arrow, the brush hides the pointer (its outline is the
    // cursor), the other tools, the paint tools that are a click among them,
    // keep the arrow.
    void Viewer::Impl::setCursorFor(const ViewState& s) {
        const ViewerTool t = s.tool;
        ImGuiMouseCursor shape = ImGuiMouseCursor_Arrow;
        if (t == ViewerTool::Navigate) shape = ImGuiMouseCursor_ResizeAll;
        else if (brushLike(s)) shape = ImGuiMouseCursor_None;
        xy.setCursor(shape, shape);
        const ImGuiMouseCursor plain = t == ViewerTool::Navigate ? ImGuiMouseCursor_ResizeAll : ImGuiMouseCursor_Arrow;
        cmpLeft.setCursor(plain, plain);
        cmpRight.setCursor(shape, shape);
        for (SlicePane* p : {&yz, &xz, &mip}) p->setCursor(plain, plain);
    }

    void Viewer::Impl::refreshDims() {
        const DatasetMeta& m = model.meta();
        dims.setExtents(model.valid() ? nz() : 1, model.valid() ? nt() : 1, m.dz(), m.frameIntervalS);
        dims.setPosition(vs().z, vs().t);
        dims.setPlaying(playing);
    }

    // --- view state ------------------------------------------------------------------

    void Viewer::Impl::applyViewStateDiff(const ViewState& s) {
        if (s.syncZT) {   // the compare plane follows until the sync is switched off
            cmpZ = curZ();
            cmpT = curT();
        }
        if (!havePrev) {
            prev = s;
            havePrev = true;
            dirty = Dirty{};
        } else {
            if (s.t != prev.t) {
                dirty = Dirty{};
                // the frame left behind is read no further, and its progress
                // is not the new frame's
                retainVolumes();
                sliceNotice.clear();
            }
            if (s.z != prev.z) dirty.xy = dirty.cmp = true;
            if (s.cx != prev.cx) dirty.yz = true;
            if (s.cy != prev.cy) dirty.xz = true;
            if (s.channelVisible != prev.channelVisible) dirty = Dirty{};
            if (s.labels != prev.labels || s.labelOpacity != prev.labelOpacity || s.selectedLabel != prev.selectedLabel ||
                s.soloLabel != prev.soloLabel)
                dirty.xy = dirty.xz = dirty.yz = dirty.cmp = dirty.vol = true;
            if (s.mode != prev.mode) {
                if (s.mode == ViewMode::Volume) dirty.vol = true;
                if (s.mode == ViewMode::Compare) dirty.cmp = true;
            }
            if (s.syncZT != prev.syncZT) dirty.cmp = true;
            if (s.followTrack && (s.t != prev.t || s.cx != prev.cx || s.cy != prev.cy || !prev.followTrack)) followPending = true;
            // yaw / pitch / clip / bounding box need no re-upload: the volume
            // view keeps its own orientation state and re-renders itself.
            prev = s;
        }
        refreshChrome();
        dims.setPosition(s.z, s.t);
        layoutPanes();
        scheduleUpdate();
    }

    void Viewer::Impl::applyPending() {
        sync();
        if (updateQueued) {
            updateQueued = false;
            applyDirty();
        }
    }

    void Viewer::Impl::layoutPanes() {
        if (!model.valid() || !xy.placed()) return;
        const ScopedTrace trace("layoutPanes");
        const ViewState& s = vs();
        const double za = zAspect();
        // XY: fit x state.zoom, offset by the pan
        xy.setGrid(nx(), ny());   // keeps cols/rows current before fitView (the image keeps its origin)
        const SlicePane::View fitV = xy.fitView(1.0, 1.0);
        SlicePane::View v;
        v.zx = v.zy = fitV.zx * s.zoom;
        v.ox = (xy.width() - static_cast<double>(nx()) * v.zx) / 2.0 + s.panX;
        v.oy = (xy.height() - static_cast<double>(ny()) * v.zy) / 2.0 + s.panY;
        xy.setView(v);
        const Index cz = curZ();
        // YZ: rows y follow XY, cols z at the physical aspect, centred on z when wider than the pane
        yz.setGrid(nz(), ny());
        {
            SlicePane::View w;
            w.zy = v.zy;
            w.oy = v.oy;
            w.zx = v.zx * za;
            const double ez = static_cast<double>(nz()) * w.zx;
            w.ox = ez <= yz.width() ? (yz.width() - ez) / 2.0 : yz.width() / 2.0 - (static_cast<double>(cz) + 0.5) * w.zx;
            yz.setView(w);
        }
        xz.setGrid(nx(), nz());
        {
            SlicePane::View w;
            w.zx = v.zx;
            w.ox = v.ox;
            w.zy = v.zy * za;
            const double ez = static_cast<double>(nz()) * w.zy;
            w.oy = ez <= xz.height() ? (xz.height() - ez) / 2.0 : xz.height() / 2.0 - (static_cast<double>(cz) + 0.5) * w.zy;
            xz.setView(w);
        }
        mip.setGrid(nx(), ny());   // as for xy: fitView needs the grid, not last frame's content
        mip.setView(mip.fitView(1.0, 1.0));
        {
            // Both compare panes show the same physical field: the raw pane
            // scales its (coarser or finer) voxels by the voxel-size ratio so
            // a 64-pixel raw frame overlays a 128-pixel reconstruction.
            // Both panes need their grid before fitView: until content first
            // arrives (or after the shape changes) cols/rows are stale and
            // fitView falls back to 1 px per voxel, which laid the compare
            // panes out at the wrong scale entirely.
            cmpRight.setGrid(nx(), ny());
            if (rawModel.valid()) cmpLeft.setGrid(rawModel.dims().x, rawModel.dims().y);
            const SlicePane::View f = cmpRight.fitView(1.0, 1.0);
            SlicePane::View w;
            w.zx = w.zy = f.zx * s.zoom;
            w.ox = (cmpRight.width() - static_cast<double>(nx()) * w.zx) / 2.0 + s.panX;
            w.oy = (cmpRight.height() - static_cast<double>(ny()) * w.zy) / 2.0 + s.panY;
            cmpRight.setView(w);
            SlicePane::View l = w;
            if (rawModel.valid() && rawModel.meta().dx() > 0.0 && rawModel.meta().dy() > 0.0) {
                // screen pixels per raw voxel: a raw voxel twice as large as
                // the reconstruction's covers twice the screen
                l.zx = w.zx * rawModel.meta().dx() / model.meta().dx();
                l.zy = w.zy * rawModel.meta().dy() / model.meta().dy();
            }
            cmpLeft.setView(l);
        }
        // a coarser render is enough when the image is smaller than the pane;
        // a region too long for a texture is sub-sampled further
        const int xyZoom = std::max(1, static_cast<int>(std::floor(1.0 / std::max(v.zx, 1e-6))));
        const int wantFactor = textureFactor(xy, xyZoom, nx(), ny());
        if (wantFactor != xyFactor) {
            xyFactor = wantFactor;
            dirty.xy = true;
        }
        const int cmpZoom = std::max(1, static_cast<int>(std::floor(1.0 / std::max(cmpRight.view().zx, 1e-6))));
        const int cmpWant = textureFactor(cmpRight, cmpZoom, nx(), ny());
        if (cmpWant != cmpFactor) {
            cmpFactor = cmpWant;
            dirty.cmp = true;
        }
        // The raw pane has its own scale: after a step that changes the voxel
        // size (SIM halves it) a raw voxel covers twice the screen, so it needs
        // half the sub-sampling. Sharing the reconstruction's factor rendered
        // it at half the resolution it deserved and magnified the result.
        const int cmpLeftZoom = std::max(1, static_cast<int>(std::floor(1.0 / std::max(cmpLeft.view().zx, 1e-6))));
        const int cmpLeftWant =
            rawModel.valid() ? textureFactor(cmpLeft, cmpLeftZoom, rawModel.dims().x, rawModel.dims().y) : cmpLeftZoom;
        if (cmpLeftWant != cmpLeftFactor) {
            cmpLeftFactor = cmpLeftWant;
            dirty.cmp = true;
        }
        // the MIP is the whole plane fitted to the pane: at most twice the pane's size
        const int mipWant = std::max(1, static_cast<int>(std::floor(1.0 / std::max(mip.view().zx, 1e-6))));
        if (mipWant != mipFactor) {
            mipFactor = mipWant;
            dirty.mip = true;
        }
        // XZ / YZ stretch z by the voxel aspect, so the factor that suits z
        // can leave a long x or y at one texel per voxel, too long for a texture
        const int xzWant = textureFactor(xz, paneFactor(xz.view()), nx(), nz());
        if (xzWant != xzFactor) {
            xzFactor = xzWant;
            dirty.xz = true;
        }
        const int yzWant = textureFactor(yz, paneFactor(yz.view()), nz(), ny());
        if (yzWant != yzFactor) {
            yzFactor = yzWant;
            dirty.yz = true;
        }
        xy.setSmooth(v.zx * xyFactor < 1.0);
        xz.setSmooth(std::min(xz.view().zx, xz.view().zy) * xzFactor < 1.0);
        yz.setSmooth(std::min(yz.view().zx, yz.view().zy) * yzFactor < 1.0);
        cmpRight.setSmooth(cmpRight.view().zx * cmpFactor < 1.0);
        cmpLeft.setSmooth(cmpLeft.view().zx * cmpLeftFactor < 1.0);
        // the view moved out of what was rendered (pan / zoom / resize): render again
        if (!dirty.xy && xy.hasContent() && !xyRegion.empty() && !xyRegion.contains(visibleVoxels(xy, nx(), ny()))) {
            dirty.xy = true;
            scheduleUpdate();
        }
        if (!dirty.xz && xz.hasContent() && !xzRegion.empty() && !xzRegion.contains(visibleVoxels(xz, nx(), nz()))) {
            dirty.xz = true;
            scheduleUpdate();
        }
        if (!dirty.yz && yz.hasContent() && !yzRegion.empty() && !yzRegion.contains(visibleVoxels(yz, nz(), ny()))) {
            dirty.yz = true;
            scheduleUpdate();
        }
        if (!dirty.cmp && s.mode == ViewMode::Compare && cmpRight.hasContent() && !cmpRegion.empty() &&
            (!cmpRegion.contains(visibleVoxels(cmpRight, nx(), ny())) ||
             (rawModel.valid() && !cmpLeftRegion.empty() &&
              !cmpLeftRegion.contains(visibleVoxels(cmpLeft, rawModel.dims().x, rawModel.dims().y))))) {
            dirty.cmp = true;
            scheduleUpdate();
        }
        // overlays that depend on the view
        const bool locked = s.tool != ViewerTool::Probe;
        const DPoint cross(static_cast<double>(curX()), static_cast<double>(curY()));
        xy.setCrosshair(cross, s.crosshair, locked);
        yz.setCrosshair(DPoint(static_cast<double>(cz), static_cast<double>(curY())), s.crosshair, locked);
        xz.setCrosshair(DPoint(static_cast<double>(curX()), static_cast<double>(cz)), s.crosshair, locked);
        mip.setCrosshair(cross, s.crosshair, locked);
        DPoint rawCross = cross;
        double rawDx = model.meta().dx();
        if (rawModel.valid() && rawModel.meta().dx() > 0.0 && rawModel.meta().dy() > 0.0) {
            rawCross = DPoint(cross.x * model.meta().dx() / rawModel.meta().dx(), cross.y * model.meta().dy() / rawModel.meta().dy());
            rawDx = rawModel.meta().dx();
        }
        cmpLeft.setCrosshair(rawCross, s.crosshair, locked);
        cmpRight.setCrosshair(cross, s.crosshair, locked);
        // View ▸ Scale bar: 0 µm per voxel hides it
        const double dx = s.scaleBar ? model.meta().dx() : 0.0;
        xy.setScaleBar(dx);
        cmpLeft.setScaleBar(s.scaleBar ? rawDx : 0.0);
        cmpRight.setScaleBar(dx);
        mip.setScaleBar(dx);
        const std::string zt = format("z %lld / %lld  t %lld / %lld  %ld %%", static_cast<long long>(cz), static_cast<long long>(nz() - 1),
                                      static_cast<long long>(curT()), static_cast<long long>(nt() - 1), std::lround(s.zoom * 100.0));
        xy.setTitle("XY  " + zt);
        yz.setTitle("YZ");
        xz.setTitle("XZ");
        mip.setTitle("MIP \xC2\xB7 Z");
        // the raw pane's own plane and extent (a step may resample z)
        const Index rawNz = rawModel.valid() ? rawModel.dims().z : nz();
        const std::string rawZt = format("z %lld / %lld  t %lld / %lld  %s", static_cast<long long>(compareRawZ()), static_cast<long long>(rawNz - 1),
                                         static_cast<long long>(compareT()), static_cast<long long>(nt() - 1),
                                         s.syncZT ? format("%ld %%", std::lround(s.zoom * 100.0)).c_str() : "unsynced");
        cmpLeft.setTitle("01 Load \xC2\xB7 raw  " + rawZt);
        const int viewed = wb.viewedIndex();
        const std::string name = viewed >= 0 && viewed < wb.pipeline().size() ? num2(viewed) + " " + wb.pipeline().at(viewed).name : std::string();
        cmpRight.setTitle(name);
        pushAnnotations();
    }

    void Viewer::Impl::applyDirty() {
        if (!model.valid()) {
            for (SlicePane* p : {&xy, &yz, &xz, &mip, &cmpLeft, &cmpRight}) p->clearContent();
            xy.setMessage(wb.hasDataset() ? "Nothing to display" : "Open a dataset (File \xE2\x96\xB8 Open dataset\xE2\x80\xA6)");
            volume.clearVolumes();
            volume.clearLabels();
            shownVolumeKey = 0;
            dirty = Dirty{};
            return;
        }
        // The panes have no size (before their first frame, or squeezed to
        // nothing): render when layoutOrtho gives XY a size again, which
        // queues the update. The dirty flags wait until then; queuing one
        // here redrew every frame while nothing could be shown.
        if (!xy.placed()) return;
        xy.setMessage({});
        const ViewState& s = vs();
        const Index t = curT(), z = curZ();
        // SIRIUS_TRACE_VIEW=1 prints what each pane costs (render, label overlay, hand-over)
        const bool trace = ScopedTrace::enabled();
        TraceClock clock;
        // XZ / YZ need the (c, t) volume in RAM; the MIP corner also needs a
        // z-projection. Play used to wait for the projection (a full pass over
        // ~10^8 voxels) before drawing the slices, so the movie stuttered.
        const bool needsVolume = s.mode == ViewMode::Ortho;
        bool haveVol = false, haveMip = false;
        if (needsVolume) {
            retainVolumes();
            volumeReadiness(model, t, haveVol, haveMip);
            const DisplayModel::VolumeState vstate = ensureVolumes(model, t);
            // An in-memory volume small enough is projected inside
            // ensureVolumes: it is ready now.
            if (vstate == DisplayModel::VolumeState::Ready && (!haveVol || !haveMip)) {
                const bool hadMip = haveMip;
                volumeReadiness(model, t, haveVol, haveMip);
                if (haveMip && !hadMip) dirty.mip = true;
            }
            // Wanted: a read is on its way. Ready without the volume: a
            // channel's read failed (failedVolumes). TooLarge says so itself
            // (refreshHints).
            const bool wanted = vstate == DisplayModel::VolumeState::Wanted;
            std::string notice;
            if (!haveVol && wanted) notice = sliceNotice.empty() ? std::string("Loading\xE2\x80\xA6") : sliceNotice;
            else if (!haveVol && vstate == DisplayModel::VolumeState::Ready) notice = "could not read the volume";
            if (notice != sliceNotice) {
                sliceNotice = notice;
                if (trace)
                    std::fprintf(stderr, "view slices %s\n",
                                 haveVol ? (haveMip ? "ready" : "have volume, MIP pending") : "waiting for the volume (Loading...)");
                refreshHints();
            }
            // What the panes show until then: the previous frame of this
            // output while a read is coming, nothing when none is (the old
            // slices no longer followed the crosshair, and a region of
            // another grid had layoutPanes ask for a render every frame).
            if (!haveVol) {
                dirty.xz = dirty.yz = false;
                if (!wanted) {
                    xz.clearContent();
                    yz.clearContent();
                    xzRegion = yzRegion = RectI{};
                }
            }
            if (!haveMip) {
                dirty.mip = false;
                if (!wanted) mip.clearContent();
            }
            if (playing && nt() > 1) ensureVolumes(model, (t + 1) % nt());
        }
        if (s.mode == ViewMode::Ortho) {
            if (dirty.xy) {
                clock.start();
                xyRegion = renderRegion(xy, xyFactor, nx(), ny());
                model.renderXY(t, z, s, xyFactor, xyImg, xyRegion);
                const long long r = clock.restart();
                if (s.labels) model.overlayLabelsXY(t, z, xyFactor, s, xyImg, xyRegion);
                const long long o = clock.restart();
                xy.setContent(xyImg, xyFactor, nx(), ny(), xyRegion.x, xyRegion.y);
                const long long c = clock.restart();
                if (trace) {
                    const RectI vis = visibleVoxels(xy, nx(), ny());
                    std::fprintf(stderr,
                                 "view xy %dx%d f%d at %d,%d (visible %d,%d %dx%d \xC2\xB7 pane %.0fx%.0f \xC2\xB7 zx %.3f ox %.1f): render %lld us \xC2\xB7 "
                                 "labels %lld us \xC2\xB7 content %lld us\n",
                                 xyImg.width, xyImg.height, xyFactor, xyRegion.x, xyRegion.y, vis.x, vis.y, vis.w, vis.h, xy.width(), xy.height(),
                                 xy.view().zx, xy.view().ox, r, o, c);
                }
                dirty.xy = false;
            }
            if (dirty.xz) {
                clock.start();
                xzRegion = renderRegion(xz, xzFactor, nx(), nz());
                model.renderXZ(t, curY(), s, xzFactor, xzImg, xzRegion);
                const long long r = clock.restart();
                if (s.labels) model.overlayLabelsXZ(t, curY(), xzFactor, s, xzImg, xzRegion);
                const long long o = clock.restart();
                xz.setContent(xzImg, xzFactor, nx(), nz(), xzRegion.x, xzRegion.y);
                const long long c = clock.restart();
                if (trace)
                    std::fprintf(stderr, "view xz %dx%d f%d at %d,%d (zx %.3f zy %.3f): render %lld us \xC2\xB7 labels %lld us \xC2\xB7 content %lld us\n",
                                 xzImg.width, xzImg.height, xzFactor, xzRegion.x, xzRegion.y, xz.view().zx, xz.view().zy, r, o, c);
                dirty.xz = false;
            }
            if (dirty.yz) {
                clock.start();
                yzRegion = renderRegion(yz, yzFactor, nz(), ny());
                model.renderYZ(t, curX(), s, yzFactor, yzImg, yzRegion);
                const long long r = clock.restart();
                if (s.labels) model.overlayLabelsYZ(t, curX(), yzFactor, s, yzImg, yzRegion);
                const long long o = clock.restart();
                yz.setContent(yzImg, yzFactor, nz(), ny(), yzRegion.x, yzRegion.y);
                const long long c = clock.restart();
                if (trace)
                    std::fprintf(stderr, "view yz %dx%d f%d at %d,%d (zx %.3f zy %.3f): render %lld us \xC2\xB7 labels %lld us \xC2\xB7 content %lld us\n",
                                 yzImg.width, yzImg.height, yzFactor, yzRegion.x, yzRegion.y, yz.view().zx, yz.view().zy, r, o, c);
                dirty.yz = false;
            }
            if (dirty.mip) {
                clock.start();
                model.renderMIP(t, s, mipFactor, mipImg);
                const long long r = clock.restart();
                mip.setContent(mipImg, mipFactor, nx(), ny());
                const long long c = clock.restart();
                if (trace)
                    std::fprintf(stderr, "view mip %dx%d f%d: render %lld us \xC2\xB7 content %lld us\n", mipImg.width, mipImg.height, mipFactor, r, c);
                dirty.mip = false;
            }
            if (model.volumeTooLarge()) refreshHints();
        } else if (s.mode == ViewMode::Compare) {
            if (dirty.cmp) {
                if (rawModel.valid()) {
                    const Index rt = std::clamp<Index>(compareT(), 0, rawModel.dims().t - 1);
                    const Index rz = compareRawZ();
                    cmpLeftRegion = renderRegion(cmpLeft, cmpLeftFactor, rawModel.dims().x, rawModel.dims().y);
                    rawModel.renderXY(rt, rz, s, cmpLeftFactor, cmpLeftImg, cmpLeftRegion);
                    cmpLeft.setContent(cmpLeftImg, cmpLeftFactor, rawModel.dims().x, rawModel.dims().y, cmpLeftRegion.x, cmpLeftRegion.y);
                    if (trace)
                        std::fprintf(stderr, "view cmp raw %dx%d f%d (zx %.3f) \xC2\xB7 step f%d (zx %.3f)\n", cmpLeftImg.width, cmpLeftImg.height,
                                     cmpLeftFactor, cmpLeft.view().zx, cmpFactor, cmpRight.view().zx);
                } else {
                    cmpLeft.clearContent();
                }
                cmpRegion = renderRegion(cmpRight, cmpFactor, nx(), ny());
                model.renderXY(t, z, s, cmpFactor, cmpRightImg, cmpRegion);
                if (s.labels) model.overlayLabelsXY(t, z, cmpFactor, s, cmpRightImg, cmpRegion);
                cmpRight.setContent(cmpRightImg, cmpFactor, nx(), ny(), cmpRegion.x, cmpRegion.y);
                dirty.cmp = false;
            }
        } else {
            if (dirty.vol) {
                renderVolume();
                dirty.vol = false;
            }
        }
        refreshTracks();
        layoutPanes();
    }

    void Viewer::Impl::refreshTracks() {
        const ViewState& s = vs();
        const LabelVolume* labels = model.labels();
        std::shared_ptr<const TrackIndex> index =
            labels && labels->tracked() && s.labels && s.trajectories && s.mode != ViewMode::Volume ? labels->tracks() : nullptr;
        const std::uint32_t only = s.soloLabel ? s.selectedLabel : 0u;
        const bool sameIndex = !trackCache.index.owner_before(index) && !index.owner_before(trackCache.index) &&
                               (index != nullptr) == !trackCache.index.expired();
        if (!sameIndex || labelsVersion != trackCache.version || only != trackCache.only) {
            trackCache.index = index;
            trackCache.version = labelsVersion;
            trackCache.only = only;
            if (index) {
                trackCache.xy = std::make_shared<const std::vector<TrackPath>>(trackPaths(*index, TrackPlane::XY, only));
                trackCache.xz = std::make_shared<const std::vector<TrackPath>>(trackPaths(*index, TrackPlane::XZ, only));
                trackCache.yz = std::make_shared<const std::vector<TrackPath>>(trackPaths(*index, TrackPlane::YZ, only));
            } else {
                trackCache.xy = trackCache.xz = trackCache.yz = nullptr;
            }
        }
        // The slices show the tracks within a nucleus or so of the plane on
        // screen (kTrackSliceUm, converted per axis); the projection shows all.
        constexpr double kTrackSliceUm = 4.0;
        const auto& um = model.meta().voxelUm;   // x, y, z
        const auto voxels = [&](double axisUm) { return axisUm > 0.0 ? kTrackSliceUm / axisUm : 4.0; };
        TrackPaintOptions all;
        all.t = curT();
        all.selected = s.selectedLabel;
        TrackPaintOptions nearZ = all, nearY = all, nearX = all;
        nearZ.depth = static_cast<double>(curZ()) + 0.5, nearZ.depthRange = voxels(um[2]);
        nearY.depth = static_cast<double>(curY()) + 0.5, nearY.depthRange = voxels(um[1]);
        nearX.depth = static_cast<double>(curX()) + 0.5, nearX.depthRange = voxels(um[0]);
        xy.setTracks(trackCache.xy, nearZ);
        mip.setTracks(trackCache.xy, all);
        xz.setTracks(trackCache.xz, nearY);
        yz.setTracks(trackCache.yz, nearX);
        cmpRight.setTracks(s.mode == ViewMode::Compare ? trackCache.xy : nullptr, nearZ);
    }

    void Viewer::Impl::followTrackIntoView() {
        const ViewState& s = vs();
        if (!s.followTrack || !model.valid() || s.mode != ViewMode::Ortho || s.zoom <= 1.0) return;
        // only when the crosshair nears the edge: recentring on every frame
        // would make the image swim under a track that barely moves
        const DPoint at = xy.toScreen(DPoint(static_cast<double>(curX()) + 0.5, static_cast<double>(curY()) + 0.5));
        const double w = xy.width(), h = xy.height();
        if (at.x >= w * 0.2 && at.x <= w * 0.8 && at.y >= h * 0.2 && at.y <= h * 0.8) return;
        const double zx = xy.view().zx, zy = xy.view().zy;
        setZoomPan(s.zoom, (static_cast<double>(nx()) / 2.0 - static_cast<double>(curX()) - 0.5) * zx,
                   (static_cast<double>(ny()) / 2.0 - static_cast<double>(curY()) - 0.5) * zy);
    }

    void Viewer::Impl::renderVolume() {
        const ViewState& s = vs();
        const Index t = curT();
        retainVolumes();
        const DisplayModel::VolumeState vstate = ensureVolumes(model, t);
        if (vstate == DisplayModel::VolumeState::TooLarge) {
            volume.clearVolumes();
            volume.setPreparing("Volume too large to render");
            volumeKey = shownVolumeKey = 0;
            return;
        }
        if (vstate == DisplayModel::VolumeState::Wanted) {
            volume.setPreparing(sliceNotice.empty() ? std::string("Loading volume\xE2\x80\xA6") : sliceNotice);
            return;   // the textures already up stay up until the new ones land
        }
        // The reduction to <= 256 texels per axis is a pass over every voxel:
        // it runs on the loader thread and the frame only uploads the result.
        std::vector<ViewerLoader::Channel> chans;
        // FNV-1a over the output, t and each visible channel's window, in
        // order: XOR-ing a term per channel cancelled two channels with one
        // window (a live preview gives every channel the same), and the key
        // then ignored the window.
        std::uint64_t key = 0xcbf29ce484222325ull;
        const auto mix = [&key](std::uint64_t v) { key = (key ^ v) * 0x100000001b3ull; };
        const auto bits = [](float f) {
            std::uint32_t u = 0;
            std::memcpy(&u, &f, sizeof u);
            return static_cast<std::uint64_t>(u);
        };
        mix(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(model.output().get())));
        mix(static_cast<std::uint64_t>(t));
        for (Index c = 0; c < model.dims().c; ++c) {
            if (!s.channelOn(c)) continue;
            const float* v = model.volumeIfReady(c, t);
            if (!v) continue;
            ViewerLoader::Channel ch;
            ch.out = model.output();
            ch.hold = model.volumeHold(c, t);
            ch.data = v;
            ch.z = nz();
            ch.y = ny();
            ch.x = nx();
            const DisplayWindow w = model.window(c, t);
            ch.lo = w.lo;
            ch.hi = w.hi;
            if (model.meta().rgb) {
                ch.color = {c == 0 ? 1.f : 0.f, c == 1 ? 1.f : 0.f, c == 2 ? 1.f : 0.f};
            } else if (static_cast<std::size_t>(c) < model.meta().channels.size()) {
                ch.color = model.meta().channels[static_cast<std::size_t>(c)].color;
            }
            mix(static_cast<std::uint64_t>(c + 1));
            mix(bits(w.lo));
            mix(bits(w.hi));
            chans.push_back(ch);
        }
        if (chans.empty()) {
            volume.clearVolumes();
            volume.setPreparing(std::string());
            volumeKey = shownVolumeKey = 0;
        } else if (key != volumeKey) {
            volumeKey = key;
            volume.setPreparing("Preparing volume\xE2\x80\xA6");
            loader.reduce(key, std::move(chans));
        } else {
            // The same bricks as before: up already, when the text is left
            // from a read of another frame (back in 3D after Ortho stepped
            // through time), or still being reduced.
            volume.setPreparing(shownVolumeKey == key ? std::string() : std::string("Preparing volume\xE2\x80\xA6"));
        }
        // labels ride along as their own texture, toggled with the Labels box
        const LabelVolume* L = s.labels ? model.labels() : nullptr;
        if (L && t < L->t()) {
            const std::uint32_t only = s.soloLabel ? s.selectedLabel : 0u;
            const std::uint64_t lkey = (static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(L)) ^ (static_cast<std::uint64_t>(t + 1) << 40) ^
                                        (labelsVersion << 8) ^ (static_cast<std::uint64_t>(only) << 20)) |
                                       1;
            // the view keeps the volume (and so its voxels) alive between the
            // call and its upload; a stroke's copy-on-write must not free them
            const std::shared_ptr<const StepOutput> owner = model.output();
            volume.setLabels(lkey, std::shared_ptr<const void>(owner, owner ? owner->labels.get() : nullptr), L->volume(t), L->z(), L->y(), L->x(),
                             static_cast<float>(s.labelOpacity), only);
        } else {
            volume.clearLabels();
        }
    }

    // --- zoom / pan --------------------------------------------------------------------

    void Viewer::Impl::setZoomPan(double zoom, double panX, double panY) {
        ViewState s = vs();
        s.zoom = std::clamp(zoom, kMinZoom, kMaxZoom);
        s.panX = panX;
        s.panY = panY;
        if (s.zoom == 1.0 && zoom == 1.0 && panX == 0.0 && panY == 0.0) {
            s.panX = s.panY = 0.0;
        }
        wb.setViewState(s);
    }

    void Viewer::Impl::zoomAround(double factor, const DPoint& anchor, const SlicePane* pane) {
        if (!model.valid()) return;
        const ViewState& s = vs();
        const double newZoom = std::clamp(s.zoom * factor, kMinZoom, kMaxZoom);
        if (newZoom == s.zoom) return;
        // Compare lays both of its panes out from the step pane (layoutPanes):
        // its view and size are what zoom and pan mean there, and the raw
        // pane shares its origin, so a point of either pane is the same
        // point of the step pane. Using the XY pane -- hidden in Compare,
        // with a stale view and another size -- let the image slide away
        // from under the cursor.
        const SlicePane& ref = pane == &cmpLeft || pane == &cmpRight ? cmpRight : xy;
        const SlicePane::View v = ref.view();
        const double fz = v.zx / s.zoom;   // px per voxel at fit
        const DPoint voxel = ref.toVoxel(anchor);
        const double z2 = fz * newZoom;
        const double ox = anchor.x - voxel.x * z2, oy = anchor.y - voxel.y * z2;
        const double panX = ox - (ref.width() - static_cast<double>(nx()) * z2) / 2.0;
        const double panY = oy - (ref.height() - static_cast<double>(ny()) * z2) / 2.0;
        setZoomPan(newZoom, panX, panY);
    }

    void Viewer::Impl::fit() { setZoomPan(1.0, 0.0, 0.0); }

    void Viewer::Impl::setPlaying(bool on) {
        playing = on && nt() > 1;
        playLast = ImGui::GetTime();
        dims.setPlaying(playing);
        if (playing) app.requestRedraw();
    }

    // View ▸ Sync Z / T across viewers. On (the default) every pane shows the
    // plane the dims strip points at. Off, the raw side of Compare keeps the
    // plane it was on -- a reference to scrub the processed side against --
    // and shift + wheel over it moves that plane on its own.
    Index Viewer::Impl::compareZ() const { return vs().syncZT ? curZ() : std::clamp<Index>(cmpZ, 0, std::max<Index>(nz() - 1, 0)); }

    Index Viewer::Impl::compareT() const { return vs().syncZT ? curT() : std::clamp<Index>(cmpT, 0, std::max<Index>(nt() - 1, 0)); }

    // The raw plane holding the middle of the step's plane compareZ(): a step
    // that resamples z (10x coarser, say) has other plane numbers than the
    // raw data, and the raw pane showed raw plane 5 beside step plane 5.
    Index Viewer::Impl::compareRawZ() const {
        if (!rawModel.valid()) return compareZ();
        const Index rawNz = std::max<Index>(rawModel.dims().z, 1);
        const double dzStep = model.meta().dz(), dzRaw = rawModel.meta().dz();
        Index z = compareZ();
        if (dzStep > 0.0 && dzRaw > 0.0 && model.valid()) z = static_cast<Index>(std::floor((static_cast<double>(z) + 0.5) * dzStep / dzRaw));
        return std::clamp<Index>(z, 0, rawNz - 1);
    }

    DPoint Viewer::Impl::rawToStep(const DPoint& v) const {
        if (!rawModel.valid() || !model.valid()) return v;
        const double sx = model.meta().dx(), sy = model.meta().dy(), rx = rawModel.meta().dx(), ry = rawModel.meta().dy();
        if (sx <= 0.0 || sy <= 0.0 || rx <= 0.0 || ry <= 0.0) return v;
        return {v.x * rx / sx, v.y * ry / sy};
    }

    // --- tools ---------------------------------------------------------------------------

    std::uint32_t Viewer::Impl::labelAt(Index z, Index y, Index x) const {
        const LabelVolume* L = model.labels();
        if (!L) return 0;
        const Index t = curT();
        if (t >= L->t() || z < 0 || z >= L->z() || y < 0 || y >= L->y() || x < 0 || x >= L->x()) return 0;
        return L->at(t, z, y, x);
    }

    void Viewer::Impl::paintAt(const DPoint& v, bool erase) {
        if (!model.valid()) return;
        // a drag that leaves the image stops painting instead of stamping
        // along the border row or column
        if (!xy.inside(v)) return;
        const Index x = clampIndex(v.x, nx()), y = clampIndex(v.y, ny());
        wb.paintLabels(curZ(), y, x, erase);
    }

    void Viewer::Impl::onXYPressed(const DPoint& v, int b, ImGuiKeyChord m) {
        if (!model.valid() || b != ImGuiMouseButton_Left) return;
        const ViewState& s = vs();
        const Index x = clampIndex(v.x, nx()), y = clampIndex(v.y, ny()), z = curZ();
        switch (s.tool) {
            case ViewerTool::Navigate: break;
            case ViewerTool::Probe:
                if (xy.inside(v)) wb.setCrosshair(x, y, z);
                break;
            case ViewerTool::Measure: {
                // shift-click extends the last distance on this plane to an angle
                Annotation* last = annotations.empty() ? nullptr : &annotations.back();
                if ((m & ImGuiMod_Shift) && measure.empty() && last && last->kind == SlicePane::Annotation::Kind::Measure &&
                    last->points.size() == 2 && last->t == curT() && last->z == z) {
                    last->points.push_back(v);
                } else {
                    measure.push_back(v);
                    if (measure.size() >= 2) commitMeasure();
                }
                pushAnnotations();
                break;
            }
            case ViewerTool::Roi:
                roiStart = v;
                roi = DRect{};
                pushAnnotations();
                break;
            case ViewerTool::Paint: {
                if (!canPaint()) return;   // a run holds the pipeline
                // A click beside the image edits nothing: Fill, Pick, Merge,
                // Split and Delete would act on the edge voxel it clamps to
                // (a fill of the whole background). A stroke may start there
                // and drag in.
                if (!brushLike(s) && !xy.inside(v)) return;
                const bool erase = s.paintTool == PaintTool::Erase || (m & ImGuiMod_Alt);
                switch (s.paintTool) {
                    case PaintTool::Brush:
                    case PaintTool::Erase:
                    case PaintTool::Lasso:
                        // one stroke, one undo entry: opened here, closed on release
                        wb.beginPaintStroke();
                        painting = true;
                        lastPaint = v;
                        paintAt(v, erase);
                        break;
                    case PaintTool::Fill: wb.fillLabel(z, y, x); break;
                    case PaintTool::Pick: {
                        ViewState ns = s;
                        ns.selectedLabel = labelAt(z, y, x);
                        wb.setViewState(ns);
                        break;
                    }
                    case PaintTool::Merge: {
                        const std::uint32_t id = labelAt(z, y, x);
                        if (id == 0) break;
                        if (mergeFirst == 0 || mergeFirst == id) {
                            mergeFirst = id;
                            ViewState ns = s;
                            ns.selectedLabel = id;
                            wb.setViewState(ns);
                        } else {
                            wb.mergeLabels({mergeFirst, id});
                            mergeFirst = 0;
                        }
                        refreshHints();
                        break;
                    }
                    case PaintTool::Split: {
                        if (!splitPending) {
                            splitA = {z, y, x};
                            splitPending = labelAt(z, y, x) != 0;
                        } else {
                            // Both seeds must lie in the one label, which the
                            // core refuses by throwing. The labels shown can
                            // lag the workbench's for a frame (an undo, a
                            // re-run), so the catch stays as well.
                            const std::uint32_t id = labelAt(splitA[0], splitA[1], splitA[2]);
                            if (id != 0 && labelAt(z, y, x) == id) {
                                try {
                                    wb.splitLabel(id, splitA, {z, y, x});
                                } catch (const std::exception& e) {
                                    wb.logLine(std::string("Split: ") + e.what());
                                }
                            } else {
                                wb.logLine("Split: both seeds must lie inside one label.");
                            }
                            splitPending = false;
                        }
                        refreshHints();
                        break;
                    }
                    case PaintTool::Delete: {
                        const std::uint32_t id = labelAt(z, y, x);
                        if (id != 0) wb.deleteLabel(id);
                        break;
                    }
                }
                break;
            }
        }
    }

    void Viewer::Impl::onXYDragged(const DPoint& v, const DPoint& delta, int b, ImGuiKeyChord m) {
        if (!model.valid()) return;
        const ViewState& s = vs();
        if (b == ImGuiMouseButton_Middle || (b == ImGuiMouseButton_Left && s.tool == ViewerTool::Navigate)) {
            setZoomPan(s.zoom, s.panX + delta.x, s.panY + delta.y);
            return;
        }
        if (b != ImGuiMouseButton_Left) return;
        switch (s.tool) {
            case ViewerTool::Probe:
                if (xy.inside(v)) wb.setCrosshair(clampIndex(v.x, nx()), clampIndex(v.y, ny()), curZ());
                break;
            case ViewerTool::Roi:
                roi = DRect::spanning(roiStart, v);
                pushAnnotations();
                break;
            case ViewerTool::Paint:
                if (painting && canPaint()) {
                    const ScopedTrace dragTrace("drag: paint handling");
                    const bool erase = s.paintTool == PaintTool::Erase || (m & ImGuiMod_Alt);
                    // stamp along the path so fast strokes stay continuous
                    const double spacing = std::max(1.0, s.brushPx / 4.0);
                    const DPoint d = v - lastPaint;
                    const double len = std::hypot(d.x, d.y);
                    const int n = std::max(1, static_cast<int>(len / spacing));
                    for (int i = 1; i <= n; ++i) paintAt(lastPaint + d * (static_cast<double>(i) / n), erase);
                    lastPaint = v;
                }
                break;
            default: break;
        }
    }

    void Viewer::Impl::onXYReleased(const DPoint&, int b, ImGuiKeyChord, bool) {
        if (b == ImGuiMouseButton_Left && vs().tool == ViewerTool::Roi) {
            // a box of at least one voxel becomes an annotation; a click does nothing
            if (roi.width() >= 1.0 && roi.height() >= 1.0) {
                Annotation a;
                a.kind = SlicePane::Annotation::Kind::Roi;
                a.rect = roi;
                a.t = curT();
                a.z = curZ();
                annotations.push_back(a);
            }
            roi = DRect{};
            pushAnnotations();
        }
        if (painting) wb.endPaintStroke();
        painting = false;
    }

    void Viewer::Impl::commitMeasure() {
        Annotation a;
        a.kind = SlicePane::Annotation::Kind::Measure;
        a.points = measure;
        a.t = curT();
        a.z = curZ();
        annotations.push_back(a);
        measure.clear();
    }

    void Viewer::Impl::clearAnnotations() {
        annotations.clear();
        measure.clear();
        roi = DRect{};
        pushAnnotations();
    }

    void Viewer::Impl::pushAnnotations() {
        std::vector<SlicePane::Annotation> out;
        for (const Annotation& a : annotations) {
            if (a.t != curT() || a.z != curZ()) continue;   // other planes keep theirs
            SlicePane::Annotation pa;
            pa.kind = a.kind;
            pa.points = a.points;
            pa.rect = a.rect;
            pa.text = a.kind == SlicePane::Annotation::Kind::Measure ? measureText(a.points) : roiText(a.rect);
            out.push_back(pa);
        }
        if (!measure.empty()) {
            SlicePane::Annotation pa;
            pa.points = measure;
            pa.text = measureText(measure);
            pa.pending = true;
            out.push_back(pa);
        }
        if (!roi.isNull()) {
            SlicePane::Annotation pa;
            pa.kind = SlicePane::Annotation::Kind::Roi;
            pa.rect = roi;
            pa.text = roiText(roi);
            pa.pending = true;
            out.push_back(pa);
        }
        xy.setAnnotations(std::move(out));
    }

    // The XY pane's right-click menu, in the look of the application's menus
    // (2 px ink border, 27 px rows, the label 28 px in). The items are
    // evaluated when chosen, so a run that finished while the menu was open
    // (and cleared the annotations) cannot act on a stale state.
    void Viewer::Impl::drawXYContextMenu() {
        if (contextMenuPending) {
            contextMenuPending = false;
            ImGui::OpenPopup("##xyContext");
        }
        ImGui::PushStyleColor(ImGuiCol_Border, theme::kText);
        ImGui::PushStyleColor(ImGuiCol_PopupBg, theme::kBg);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, theme::kNeutral200);
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, theme::kNeutral300);
        ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, theme::crispPen(2));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, px(0, 4));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
        if (ImGui::BeginPopup("##xyContext")) {
            auto item = [](const char* label, bool enabled) {
                const float h = theme::snap(px(27));
                const float left = px(28), right = px(12);
                const float w = std::max(px(240), left + theme::textSize(label, 12).x + right);
                ImGui::BeginDisabled(!enabled);
                const ImVec2 min = ImGui::GetCursorScreenPos();
                const bool clicked = ImGui::Selectable((std::string("##") + label).c_str(), false, ImGuiSelectableFlags_None, ImVec2(w, h));
                const bool hovered = ImGui::IsItemHovered();
                ImGui::EndDisabled();
                const ImVec2 max(min.x + std::max(w, ImGui::GetItemRectSize().x), min.y + h);
                const ImU32 fg = !enabled ? theme::kNeutral500 : (hovered ? theme::kAccentText : theme::kText);
                widgets::drawTextIn(ImGui::GetWindowDrawList(), ImVec2(min.x + left, min.y), max, label, 12, fg, theme::Weight::Regular, 0.0f, 0.5f);
                return clicked && enabled;
            };
            const bool any = !annotations.empty() || !measure.empty() || !roi.isNull();
            if (item("Clear annotations", any)) clearAnnotations();
            if (item("Remove last annotation", !annotations.empty()) && !annotations.empty()) {
                annotations.pop_back();
                pushAnnotations();
            }
            ImGui::Dummy(px(0, 4));
            const ImVec2 p = ImGui::GetCursorScreenPos();
            ImGui::GetWindowDrawList()->AddRectFilled(p, ImVec2(p.x + ImGui::GetContentRegionAvail().x, p.y + theme::crispPen(1)), theme::kDivider);
            ImGui::Dummy(ImVec2(0, theme::crispPen(1) + px(4)));
            if (item("Fit to window", true)) {
                if (vs().mode == ViewMode::Volume) volume.setZoom(1.0);
                fit();
            }
            ImGui::EndPopup();
        }
        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(4);
    }

    std::string Viewer::Impl::roiText(const DRect& r) const {
        const double dx = model.meta().dx(), dy = model.meta().dy();
        return format("%.2f \xC3\x97 %.2f \xC2\xB5m", r.width() * dx, r.height() * dy);
    }

    std::string Viewer::Impl::measureText(const std::vector<DPoint>& points) const {
        if (points.size() < 2) return std::string();
        const double dx = model.meta().dx(), dy = model.meta().dy();
        const DPoint a = points[0], b = points[1];
        const double um = std::hypot((b.x - a.x) * dx, (b.y - a.y) * dy);
        std::string text = format("%.2f \xC2\xB5m", um);
        if (points.size() >= 3) {
            const DPoint c = points[2];
            const double ux = (a.x - b.x) * dx, uy = (a.y - b.y) * dy;
            const double vx = (c.x - b.x) * dx, vy = (c.y - b.y) * dy;
            const double ang =
                std::acos(std::clamp((ux * vx + uy * vy) / std::max(1e-12, std::hypot(ux, uy) * std::hypot(vx, vy)), -1.0, 1.0));
            text += format("  \xE2\x88\xA0 %.1f\xC2\xB0", ang * 180.0 / sirius::kPi);
        }
        return text;
    }

    void Viewer::Impl::hover(SlicePane::Kind kind, const DPoint& v, bool rawPane) {
        if (!model.valid()) return;
        // The raw side of Compare reads the raw data under the cursor, in
        // the raw data's own voxels: it used to look the step's output up at
        // those coordinates, a value from somewhere else entirely.
        if (rawPane && rawModel.valid()) {
            const Dims5& d = rawModel.dims();
            const Index x = static_cast<Index>(std::floor(v.x)), y = static_cast<Index>(std::floor(v.y));
            if (x < 0 || y < 0 || x >= d.x || y >= d.y) {
                cursorText = kNoCursor;
            } else {
                Index c = 0;
                for (Index i = 0; i < d.c; ++i)
                    if (vs().channelOn(i)) {
                        c = i;
                        break;
                    }
                const Index z = compareRawZ(), t = std::clamp<Index>(compareT(), 0, std::max<Index>(d.t - 1, 0));
                const std::optional<float> val = rawModel.valueAt(c, t, z, y, x);
                cursorText = format("cursor %lld, %lld, %lld \xC2\xB7 %s \xC2\xB7 raw", static_cast<long long>(x), static_cast<long long>(y),
                                    static_cast<long long>(z), val ? format("%.5g", static_cast<double>(*val)).c_str() : "\xE2\x80\x94");
            }
            return;
        }
        Index x = curX(), y = curY(), z = curZ();
        bool inside = true;
        switch (kind) {
            case SlicePane::Kind::XY:
            case SlicePane::Kind::MIP:
            case SlicePane::Kind::Compare:
                x = static_cast<Index>(std::floor(v.x));
                y = static_cast<Index>(std::floor(v.y));
                inside = x >= 0 && y >= 0 && x < nx() && y < ny();
                break;
            case SlicePane::Kind::XZ:
                x = static_cast<Index>(std::floor(v.x));
                z = static_cast<Index>(std::floor(v.y));
                inside = x >= 0 && z >= 0 && x < nx() && z < nz();
                break;
            case SlicePane::Kind::YZ:
                z = static_cast<Index>(std::floor(v.x));
                y = static_cast<Index>(std::floor(v.y));
                inside = z >= 0 && y >= 0 && z < nz() && y < ny();
                break;
        }
        if (!inside) {
            cursorText = kNoCursor;
            return;
        }
        Index c = 0;
        for (Index i = 0; i < model.dims().c; ++i)
            if (vs().channelOn(i)) {
                c = i;
                break;
            }
        std::optional<float> val;
        if (kind == SlicePane::Kind::XY || kind == SlicePane::Kind::Compare) val = model.valueAt(c, curT(), z, y, x);
        else if (const float* vol = model.volumeIfReady(c, curT())) val = vol[(z * ny() + y) * nx() + x];
        std::string vtext = val ? format("%.5g", static_cast<double>(*val)) : std::string("\xE2\x80\x94");
        if (model.labels() && vs().labels) {
            const std::uint32_t id = labelAt(z, y, x);
            if (id) vtext += format(" \xC2\xB7 label %u", id);
        }
        cursorText = format("cursor %lld, %lld, %lld \xC2\xB7 %s", static_cast<long long>(x), static_cast<long long>(y), static_cast<long long>(z),
                            vtext.c_str());
    }

    // --- drawing -----------------------------------------------------------------------------

    // "Grid 1fr 220px / 1fr 170px, 2 px gaps on neutral-900" -- with gaps the
    // user drags, so the design values are where the balance starts. The two
    // rows share one column balance, so XY stays above XZ and YZ above MIP.
    void Viewer::Impl::layoutOrtho(ImVec2 min, ImVec2 max) {
        const float gap = theme::snap(px(static_cast<float>(viewer::kPaneGap)));
        const float s = std::max(theme::scale(), 0.01f);
        const float w = max.x - min.x - gap, h = max.y - min.y - gap;
        if (!orthoLoaded) {
            // the saved balance, else the design's targets
            const nlohmann::json cols = settings().value(kOrthoColsKey), rows = settings().value(kOrthoRowsKey);
            orthoCol = cols.is_number() ? cols.get<float>() : static_cast<float>(viewer::kYzWidth);
            orthoRow = rows.is_number() ? rows.get<float>() : static_cast<float>(viewer::kXzHeight);
            orthoLoaded = true;
        }
        const float sideMin = px(static_cast<float>(viewer::kSidePaneMin)), mainMin = px(static_cast<float>(viewer::kMainPaneMin));
        auto side = [&](float want, float total) {
            if (total <= sideMin + mainMin) return std::max(1.0f, std::floor(total * sideMin / (sideMin + mainMin)));
            return theme::snap(std::clamp(want, sideMin, total - mainMin));
        };
        const float right = side(orthoCol * s, w), bottom = side(orthoRow * s, h);
        const float x1 = max.x - right - gap, y1 = max.y - bottom - gap;
        const auto resized = [&](SlicePane& p, ImVec2 a, ImVec2 b) { return p.place(a, b); };
        // XY getting a size back (from none: squeezed, or the first frame)
        // is what renders the frame applyDirty had to leave
        const bool xyWasPlaced = xy.placed();
        if (resized(xy, min, ImVec2(x1, y1)) || xy.placed() != xyWasPlaced) {
            layoutPanes();
            dirty.xy = dirty.xz = dirty.yz = true;
            scheduleUpdate();
        }
        if (resized(yz, ImVec2(x1 + gap, min.y), ImVec2(max.x, y1))) {
            layoutPanes();
            dirty.yz = true;
            scheduleUpdate();
        }
        if (resized(xz, ImVec2(min.x, y1 + gap), ImVec2(x1, max.y))) {
            layoutPanes();
            dirty.xz = true;
            scheduleUpdate();
        }
        if (resized(mip, ImVec2(x1 + gap, y1 + gap), max)) {
            layoutPanes();
            dirty.mip = true;
            scheduleUpdate();
        }
    }

    void Viewer::Impl::orthoSplitters(ImVec2 min, ImVec2 max) {
        const float gap = theme::snap(px(static_cast<float>(viewer::kPaneGap)));
        const float s = std::max(theme::scale(), 0.01f);
        const float grip = px(3);
        // The balance stays within what layoutOrtho can show: pointer travel
        // past an edge would have to be dragged back before the gap moved.
        const float sideMin = px(static_cast<float>(viewer::kSidePaneMin)), mainMin = px(static_cast<float>(viewer::kMainPaneMin));
        const auto within = [&](float want, float total) { return std::clamp(want, sideMin / s, std::max(sideMin, total - mainMin) / s); };
        bool released = false;
        // the column gap
        {
            const float x = xy.max().x;
            ImGui::SetCursorScreenPos(ImVec2(x - grip, min.y));
            ImGui::InvisibleButton("##orthoCols", ImVec2(gap + 2 * grip, max.y - min.y));
            if (ImGui::IsItemHovered() || ImGui::IsItemActive()) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
            if (ImGui::IsItemActive() && ImGui::GetIO().MouseDelta.x != 0.0f)
                orthoCol = within(orthoCol - ImGui::GetIO().MouseDelta.x / s, max.x - min.x - gap);
            released = released || ImGui::IsItemDeactivated();
        }
        // the row gap
        {
            const float y = xy.max().y;
            ImGui::SetCursorScreenPos(ImVec2(min.x, y - grip));
            ImGui::InvisibleButton("##orthoRows", ImVec2(max.x - min.x, gap + 2 * grip));
            if (ImGui::IsItemHovered() || ImGui::IsItemActive()) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
            if (ImGui::IsItemActive() && ImGui::GetIO().MouseDelta.y != 0.0f)
                orthoRow = within(orthoRow - ImGui::GetIO().MouseDelta.y / s, max.y - min.y - gap);
            released = released || ImGui::IsItemDeactivated();
        }
        if (released) {
            // what the layout made of the drag, not the raw pointer travel
            orthoCol = static_cast<float>(yz.width()) / s;
            orthoRow = static_cast<float>(xz.height()) / s;
            settings().set(kOrthoColsKey, std::round(orthoCol));
            settings().set(kOrthoRowsKey, std::round(orthoRow));
        }
    }

    void Viewer::Impl::handleKeys() {
        const ImGuiIO& io = ImGui::GetIO();
        const bool ours = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) ||
                          ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows);
        if (!ours || io.WantTextInput || ImGui::IsAnyItemActive() || app.dialogOpen() ||
            ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel))
            return;
        // keyboard: tool letters and brush size (the menus own the rest)
        if (ImGui::IsKeyChordPressed(ImGuiKey_V)) wb.setTool(ViewerTool::Navigate);
        if (ImGui::IsKeyChordPressed(ImGuiKey_P)) wb.setTool(ViewerTool::Probe);
        if (ImGui::IsKeyChordPressed(ImGuiKey_M)) wb.setTool(ViewerTool::Measure);
        if (ImGui::IsKeyChordPressed(ImGuiKey_R)) wb.setTool(ViewerTool::Roi);
        // Escape clears the annotations in progress -- unless a run or a
        // task is active, when the window's Cancel action owns the key.
        const bool pending = !measure.empty() || !roi.isNull();
        if (pending && !bridge.busy()) app.claimKey(ImGuiKey_Escape);
        if (ImGui::IsKeyChordPressed(ImGuiKey_Escape) && !bridge.busy()) {
            measure.clear();
            roi = DRect{};
            pushAnnotations();
        }
        if (ImGui::IsKeyChordPressed(ImGuiKey_LeftBracket)) {
            ViewState s = vs();
            s.brushPx = std::max(viewer::kBrushMinPx, s.brushPx - viewer::kBrushStepPx);
            wb.setViewState(s);
        }
        if (ImGui::IsKeyChordPressed(ImGuiKey_RightBracket)) {
            ViewState s = vs();
            s.brushPx = std::min(viewer::kBrushMaxPx, s.brushPx + viewer::kBrushStepPx);
            wb.setViewState(s);
        }
        if (focused && ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
            // a pane with the keyboard walks the crosshair with the arrows:
            // Segment ▸ Next / Previous flagged label (left / right) stand back
            for (ImGuiKeyChord k : {ImGuiKeyChord(ImGuiKey_LeftArrow), ImGuiKeyChord(ImGuiKey_RightArrow), ImGuiKeyChord(ImGuiKey_UpArrow),
                                    ImGuiKeyChord(ImGuiKey_DownArrow), ImGuiKeyChord(ImGuiKey_PageUp), ImGuiKeyChord(ImGuiKey_PageDown)})
                app.claimKey(k);
            focused->keyNavigation();
        }
    }

    void Viewer::Impl::drawToolbar(ImVec2 min, ImVec2 max) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ViewState& s = vs();
        const float cy = (min.y + max.y) * 0.5f;
        const float spacing = px(14);
        float x = min.x + px(14);

        // Ortho | 3D | Compare
        static const std::vector<std::string> modes = {"Ortho", "3D", "Compare"};
        int mode = s.mode == ViewMode::Ortho ? 0 : (s.mode == ViewMode::Volume ? 1 : 2);
        ImGui::SetCursorScreenPos(ImVec2(x, theme::snap(cy - px(13))));
        if (widgets::segmented("##viewMode", modes, &mode))
            wb.setViewMode(mode == 0 ? ViewMode::Ortho : (mode == 1 ? ViewMode::Volume : ViewMode::Compare));
        x += widgets::segmentedWidth(modes) + spacing;

        // the right-hand group, where it ended up last frame
        const float rightStart = std::max(x, max.x - px(14) - rightGroupW);

        // "Viewing 05 Contrast rgb z48 y4096 x4096": the one element that
        // yields when the toolbar is tight, so "Ortho | 3D | Compare" is never
        // clipped; what it loses is in its tooltip
        {
            std::vector<Run> runs = {{"Viewing", theme::kNeutral600}};
            const int viewed = wb.viewedIndex();
            if (viewed >= 0 && viewed < wb.pipeline().size()) {
                const Step& st = wb.pipeline().at(viewed);
                runs.push_back({" " + num2(viewed) + " " + st.name, theme::kAccentText, theme::Weight::ExtraBold});
                runs.push_back({" " + wb.outputMetaOf(viewed).shapeString(), theme::kNeutral500});
                if (previewing) runs.push_back({" \xC2\xB7 live preview on " + num2(displayIndex), theme::kNeutral500});
                else if (displayIndex >= 0 && displayIndex != viewed)
                    runs.push_back({" \xC2\xB7 not run yet, showing " + num2(displayIndex), theme::kNeutral500});
            } else if (!wb.hasDataset()) {
                runs.push_back({" no dataset", theme::kNeutral500});
            }
            std::string plain;
            for (const Run& r : runs) plain += r.text;
            const float room = rightStart - spacing - x;
            float rx = x;
            for (const Run& r : runs) {
                const float left = x + room - rx;
                if (left <= 0.0f) break;
                const std::string shown = widgets::elideText(r.text, left, 12, r.weight);
                const ImVec2 ts = theme::textSize(shown, 12, r.weight);
                widgets::drawText(dl, ImVec2(rx, cy - ts.y * 0.5f), shown, 12, r.color, r.weight);
                rx += ts.x;
                if (shown != r.text) break;
            }
            if (room > 1.0f) {
                ImGui::SetCursorScreenPos(ImVec2(x, min.y));
                ImGui::Dummy(ImVec2(room, max.y - min.y));
                widgets::tooltip(plain);
            }
        }

        // Labels, Solo, Tracks, Crosshair / Bounding box, swatches, tile, Auto, Reset
        float rx = rightStart;
        auto at = [&](float h) { ImGui::SetCursorScreenPos(ImVec2(rx, theme::snap(cy - h * 0.5f))); };
        auto next = [&](float gap) { rx = ImGui::GetItemRectMax().x + gap; };
        const float checkH = std::max(theme::snap(px(14)), theme::textSize("Ag", 12).y) + px(4);
        const bool hasLabels = model.hasLabels();

        at(checkH);
        if (widgets::tokenCheck("Labels", s.labels, nullptr, hasLabels)) wb.toggleLabels();
        next(spacing);

        at(checkH);
        const std::string soloCaption = s.soloLabel ? (s.selectedLabel ? format("label %u", s.selectedLabel) : std::string("select a label")) : std::string();
        if (widgets::tokenCheck("Solo", s.soloLabel, soloCaption.empty() ? nullptr : soloCaption.c_str(), hasLabels)) wb.toggleSoloLabel();
        widgets::tooltip("Show only the selected label, in the slices and in 3D; selecting a label jumps to it" + shortcutSuffix(ImGuiKey_O));
        next(spacing);

        {
            const LabelVolume* labels = model.labels();
            const bool tracked = labels && labels->tracked() && labels->tracks();
            if (tracked && s.mode != ViewMode::Volume) {
                at(checkH);
                if (widgets::tokenCheck("Tracks", s.trajectories, tracked && s.followTrack ? "following" : nullptr, s.labels)) {
                    ViewState ns = vs();
                    ns.trajectories = !ns.trajectories;
                    wb.setViewState(ns);
                }
                widgets::tooltip("Draw each track's path over time on tracked labels: solid up to this time "
                                 "point, faint after it, dotted where the track is missing from a frame");
                next(spacing);
            }
        }
        if (s.mode != ViewMode::Volume) {
            at(checkH);
            if (widgets::tokenCheck("Crosshair", s.crosshair, s.tool == ViewerTool::Probe ? nullptr : "locked")) wb.toggleCrosshair();
            next(spacing);
        } else {
            at(checkH);
            if (widgets::tokenCheck("Bounding box", s.boundingBox)) {
                ViewState ns = vs();
                ns.boundingBox = !ns.boundingBox;
                wb.setViewState(ns);
            }
            next(spacing);
        }

        // channel swatches
        if (model.valid()) {
            const DatasetMeta& m = model.meta();
            std::vector<std::pair<std::string, ImU32>> items;
            if (m.rgb) {
                items = {{"R", theme::rgb(255, 80, 80)}, {"G", theme::rgb(80, 255, 80)}, {"B", theme::rgb(110, 110, 255)}};
            } else {
                for (Index c = 0; c < m.dims.c; ++c) {
                    if (static_cast<std::size_t>(c) < m.channels.size()) {
                        const ChannelInfo& ch = m.channels[static_cast<std::size_t>(c)];
                        items.emplace_back(ch.shortName(), colorFromHexString(ch.hexColor()));
                    } else {
                        items.emplace_back(std::to_string(c), theme::rgb(255, 255, 255));
                    }
                }
            }
            for (std::size_t c = 0; c < items.size(); ++c) {
                at(px(22));
                ImGui::PushID(static_cast<int>(c));
                const Index ci = static_cast<Index>(c);
                if (widgets::channelSwatch("##swatch", items[c].first, items[c].second, s.channelOn(ci), items[c].first))
                    wb.setChannelVisible(ci, !vs().channelOn(ci));
                ImGui::PopID();
                next(c + 1 < items.size() ? px(4) : spacing);
            }

            // tile chooser: only for multi-file datasets with more than one tile
            if (m.hasTiles()) {
                rx += px(6);   // the tile chooser's own margin
                const std::string cap = captionCase("Tile");
                at(theme::textSize(cap, 10).y);
                widgets::caption("Tile");
                next(px(4));
                std::vector<std::string> tiles;
                float widest = 0.0f;
                for (std::size_t i = 0; i < m.tiles.size(); ++i) {
                    tiles.push_back(format("%zu \xC2\xB7 %s", i + 1, m.tiles[i].name.c_str()));
                    widest = std::max(widest, theme::textSize(tiles.back(), 13).x);
                }
                int current = std::clamp(static_cast<int>(m.tileIndex), 0, static_cast<int>(tiles.size()) - 1);
                widgets::FieldOpts fo;
                fo.height = 26;
                fo.width = widest / std::max(theme::scale(), 0.01f) + 40.0f;
                at(px(26));
                if (widgets::combo("##tile", &current, tiles, fo)) {
                    bool ok = true;
                    try {
                        wb.setStepParam(0, "tile", static_cast<std::int64_t>(current));
                    } catch (const std::exception& e) {
                        wb.logLine(std::string("Tile: ") + e.what());
                        ok = false;
                    }
                    // switching tiles is navigation: show the new tile without a manual run
                    if (ok && !bridge.running()) bridge.startRun(wb.viewedIndex());
                }
                widgets::tooltip("Which tile of the multi-file dataset the pipeline reads (Load \xE2\x96\xB8 tile); the viewed step is re-run on it");
                next(spacing);
            }
        }

        // display contrast: the auto percentile window or the full data range
        widgets::ButtonOpts bo;
        bo.kind = widgets::ButtonKind::Ghost;
        bo.small = true;
        const float buttonH = std::max(px(14), theme::textSize("Ag", 12, theme::Weight::SemiBold).y) + 2 * px(4) + 2 * px(theme::kBorder);
        at(buttonH);
        bo.tooltip = "Auto contrast (display): window on the 0.1\xE2\x80\x93"
                     "99.9 percentiles" +
                     shortcutSuffix(ImGuiMod_Shift | ImGuiKey_A);
        if (widgets::button("Auto##contrast", bo)) app.viewer().autoContrast();
        next(spacing);
        at(buttonH);
        bo.tooltip = "Reset the display window to the full data range" + shortcutSuffix(ImGuiMod_Shift | ImGuiKey_R);
        if (widgets::button("Reset##contrast", bo)) app.viewer().resetContrast();
        const float groupW = ImGui::GetItemRectMax().x - rightStart;
        if (std::abs(groupW - rightGroupW) > 0.5f) {
            rightGroupW = groupW;
            app.requestRedraw();
        }
    }

    void Viewer::Impl::drawToolStrip(ImVec2 min, ImVec2 max) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(min, max, theme::kBg);
        const ViewState& s = vs();
        const float cx = theme::snap((min.x + max.x) * 0.5f);
        const float b = px(28);
        float y = min.y + px(4);
        static const struct {
            Icon icon;
            const char* id;
            const char* tip;
            ImGuiKey key;
            ViewerTool tool;
        } toolDefs[] = {
            {Icon::Navigate, "##navigate", "Navigate \xE2\x80\x94 drag to pan, wheel to zoom, double-click to fit", ImGuiKey_V, ViewerTool::Navigate},
            {Icon::Probe, "##probe", "Probe \xE2\x80\x94 click to place the crosshair and read values", ImGuiKey_P, ViewerTool::Probe},
            {Icon::Measure, "##measure", "Measure distance / angle", ImGuiKey_M, ViewerTool::Measure},
            {Icon::Roi, "##roi", "ROI \xE2\x80\x94 drag a box", ImGuiKey_R, ViewerTool::Roi},
            {Icon::Brush, "##paint", "Paint labels \xE2\x80\x94 needs a segmentation step", ImGuiKey_B, ViewerTool::Paint}};
        // A run owns the pipeline: the workbench refuses label edits while it
        // lasts (Workbench::canEdit), so the brush is dimmed rather than
        // silently swallowing strokes. Navigate / Probe / Measure / ROI stay.
        const bool paintOk = paintAvailable() && canPaint();
        for (const auto& t : toolDefs) {
            ImGui::SetCursorScreenPos(ImVec2(cx - b * 0.5f, y));
            widgets::GlyphOpts o;
            o.active = s.tool == t.tool;
            o.tooltip = std::string(t.tip) + shortcutSuffix(t.key);
            if (t.tool == ViewerTool::Paint) {
                o.enabled = paintOk;
                if (!canPaint()) o.tooltip = "Paint labels \xE2\x80\x94 not while a run is in progress";
            }
            if (widgets::glyphButton(t.id, t.icon, 28, o)) {
                if (t.tool != ViewerTool::Paint || paintAvailable()) wb.setTool(t.tool);
            }
            y += b + px(2);
        }
        // the rule, then + / - / fit
        y += px(8);
        dl->AddRectFilled(ImVec2(cx - px(10), theme::snap(y)), ImVec2(cx + px(10), theme::snap(y) + theme::crispPen(theme::kRule)), theme::kDivider);
        y += theme::crispPen(theme::kRule) + px(10);
        struct ZoomButton {
            Icon icon;
            const char* id;
            std::string tip;
            int which;
        };
        // the tool tips name the keys: + - 0
        const ZoomButton zooms[] = {{Icon::Plus, "##zoomIn", "Zoom in (+)", 0},
                                    {Icon::Minus, "##zoomOut", "Zoom out (-)", 1},
                                    {Icon::Fit, "##zoomFit", "Fit to view" + shortcutSuffix(ImGuiKey_0), 2}};
        for (const ZoomButton& z : zooms) {
            ImGui::SetCursorScreenPos(ImVec2(cx - b * 0.5f, y));
            widgets::GlyphOpts o;
            o.tooltip = z.tip;
            if (widgets::glyphButton(z.id, z.icon, 28, o)) {
                if (z.which == 0) app.viewer().zoomIn();
                else if (z.which == 1) app.viewer().zoomOut();
                else app.viewer().fitToWindow();
            }
            y += b + px(2);
        }
        // the zoom readout: 10 px text running upwards, at the bottom
        const float thick = px(14);
        if (max.y - px(4) - px(60) > y) drawVerticalText(dl, ImVec2(cx - thick * 0.5f, max.y - px(4)), thick, zoomText, 10, theme::kNeutral600);
    }

    void Viewer::Impl::draw() {
        // Below the window's tab or title bar: the window has no padding, so
        // the cursor starts where the viewer's own room does.
        const ImVec2 wmin = ImGui::GetCursorScreenPos();
        const ImVec2 wmax(ImGui::GetWindowPos().x + ImGui::GetWindowSize().x, ImGui::GetWindowPos().y + ImGui::GetWindowSize().y);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ViewState& s0 = vs();

        // play: ~8 fps, looping; the frames keep coming while it plays
        if (playing) {
            const double now = ImGui::GetTime();
            if (now - playLast >= kPlayIntervalMs / 1000.0) {
                playLast = now;
                if (nt() <= 1) setPlaying(false);
                else wb.setT((curT() + 1) % nt());
            }
            app.requestRedraw();
        }
        sync();

        // regions: toolbar, rule, canvas, rule, dims strip
        const float rule = theme::crispPen(theme::kRule);
        const float barH = theme::snap(px(theme::kViewerToolbarH));
        const float dimsH = dims.height();
        const ImVec2 barMin = wmin, barMax(wmax.x, wmin.y + barH);
        const ImVec2 dimsMin(wmin.x, wmax.y - dimsH), dimsMax = wmax;
        const ImVec2 canvasMin(wmin.x, barMax.y + rule), canvasMax(wmax.x, dimsMin.y - rule);
        const float g = theme::snap(px(2));
        const ImVec2 stripMin(canvasMin.x + g, canvasMin.y + g), stripMax(stripMin.x + theme::snap(px(theme::kToolStripW)), canvasMax.y - g);
        const ImVec2 stackMin(stripMax.x + g, canvasMin.y + g), stackMax(std::max(stripMax.x + g + 1.0f, canvasMax.x - g), std::max(canvasMin.y + g + 1.0f, canvasMax.y - g));

        // every page has the stack's geometry (the hidden ones as well)
        layoutOrtho(stackMin, stackMax);
        {
            const float half = std::floor((stackMax.x - stackMin.x - g) * 0.5f);
            const bool leftResized = cmpLeft.place(stackMin, ImVec2(stackMin.x + half, stackMax.y));
            const bool rightResized = cmpRight.place(ImVec2(stackMin.x + half + g, stackMin.y), stackMax);
            if (leftResized || rightResized) {
                layoutPanes();
                dirty.cmp = true;
                scheduleUpdate();
            }
        }

        // input on the visible page
        bool pressed = false;
        const ViewMode mode = s0.mode;
        if (mode == ViewMode::Ortho) {
            for (SlicePane* p : {&xy, &yz, &xz, &mip})
                if (p->input()) {
                    focused = p;
                    pressed = true;
                }
            orthoSplitters(stackMin, stackMax);
        } else if (mode == ViewMode::Compare) {
            for (SlicePane* p : {&cmpLeft, &cmpRight})
                if (p->input()) {
                    focused = p;
                    pressed = true;
                }
        }
        if (!pressed && (ImGui::IsMouseClicked(ImGuiMouseButton_Left) || ImGui::IsMouseClicked(ImGuiMouseButton_Right))) focused = nullptr;
        // A pane on a page no longer shown (the mode changed by a key, a menu,
        // the toolbar or a tool, with no click) gives the keyboard back: it
        // kept the arrows from the window's actions for a hidden crosshair.
        bool onPage = false;   // 3D has no slice pane
        if (mode == ViewMode::Ortho) onPage = focused == &xy || focused == &yz || focused == &xz || focused == &mip;
        else if (mode == ViewMode::Compare) onPage = focused == &cmpLeft || focused == &cmpRight;
        if (!onPage) focused = nullptr;
        handleKeys();

        // what the input did, then render what is dirty
        applyPending();

        // the canvas and the views
        dl->AddRectFilled(canvasMin, canvasMax, theme::kNeutral900);
        if (vs().mode == ViewMode::Ortho) {
            for (SlicePane* p : {&xy, &yz, &xz, &mip}) p->draw(dl);
        } else if (vs().mode == ViewMode::Compare) {
            cmpLeft.draw(dl);
            cmpRight.draw(dl);
        } else {
            volume.draw(stackMin, stackMax);
        }
        grabList = dl;
        grabMin = stackMin;
        grabMax = stackMax;
        grabFrame = app.frameCount();
        drawXYContextMenu();

        // chrome: toolbar, rules, tool strip, dims strip
        drawToolbar(barMin, barMax);
        dl->AddRectFilled(ImVec2(wmin.x, barMax.y), ImVec2(wmax.x, barMax.y + rule), theme::kDivider);
        drawToolStrip(stripMin, stripMax);
        dl->AddRectFilled(ImVec2(wmin.x, dimsMin.y - rule), ImVec2(wmax.x, dimsMin.y), theme::kDivider);
        dims.draw(dimsMin, dimsMax);

        // a render that asked for another pass (the region moved out of what
        // was rendered, a pane that had no size yet)
        if (updateQueued) app.requestRedraw();
    }

    bool Viewer::Impl::grab(std::vector<std::uint8_t>& rgba, int& width, int& height) {
        // The views as the last frame drew them: that frame's draw list,
        // replayed into a target the size of the views -- the overlays, the
        // crosshair and the labels with them.
        // The 3D view's controls live in child windows and stay out.
        const bool fresh = grabList && app.frameCount() == grabFrame + 1 && grabMax.x > grabMin.x && grabMax.y > grabMin.y;
        if (!fresh) {
            if (vs().mode == ViewMode::Volume) return volume.grabImage(rgba, width, height);
            return false;
        }
        const ImGuiIO& io = ImGui::GetIO();
        ImDrawData dd;
        dd.Valid = true;
        dd.CmdLists.push_back(grabList);
        dd.CmdListsCount = 1;
        dd.TotalVtxCount = grabList->VtxBuffer.Size;
        dd.TotalIdxCount = grabList->IdxBuffer.Size;
        dd.DisplayPos = grabMin;
        dd.DisplaySize = ImVec2(grabMax.x - grabMin.x, grabMax.y - grabMin.y);
        dd.FramebufferScale = io.DisplayFramebufferScale;
        dd.OwnerViewport = ImGui::GetMainViewport();
        const int w = std::max(1, static_cast<int>(dd.DisplaySize.x * dd.FramebufferScale.x));
        const int h = std::max(1, static_cast<int>(dd.DisplaySize.y * dd.FramebufferScale.y));
        RenderTarget target;
        if (!target.begin(w, h)) return false;
        GLfloat clear[4] = {0, 0, 0, 0};
        glGetFloatv(GL_COLOR_CLEAR_VALUE, clear);
        const ImVec4 ground = theme::vec(theme::kNeutral900);
        glClearColor(ground.x, ground.y, ground.z, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glClearColor(clear[0], clear[1], clear[2], clear[3]);
        ImGui_ImplOpenGL3_RenderDrawData(&dd);
        target.end();
        rgba = target.readRgba();
        for (std::size_t i = 3; i < rgba.size(); i += 4) rgba[i] = 255;
        width = w;
        height = h;
        return !rgba.empty();
    }

    // -------------------------------------------------------------------------
    // Viewer
    // -------------------------------------------------------------------------

    Viewer::Viewer(App& app) : impl_(std::make_unique<Impl>(app)) {
        impl_->connect();
        impl_->seen = impl_->bridge.rev();
        impl_->rebuildOutput();
        impl_->applyViewStateDiff(impl_->vs());
    }

    Viewer::~Viewer() { impl_->bridge.stepChanged.disconnect(impl_->stepConnection); }

    void Viewer::draw() { impl_->draw(); }

    void Viewer::zoomIn() {
        if (impl_->vs().mode == ViewMode::Volume) {
            impl_->volume.setZoom(impl_->volume.zoom() * kButtonZoomFactor);
            return;
        }
        const SlicePane& pane = impl_->vs().mode == ViewMode::Compare ? impl_->cmpRight : impl_->xy;
        impl_->zoomAround(kButtonZoomFactor, DPoint(pane.width() / 2.0, pane.height() / 2.0), &pane);
    }

    void Viewer::zoomOut() {
        if (impl_->vs().mode == ViewMode::Volume) {
            impl_->volume.setZoom(impl_->volume.zoom() / kButtonZoomFactor);
            return;
        }
        const SlicePane& pane = impl_->vs().mode == ViewMode::Compare ? impl_->cmpRight : impl_->xy;
        impl_->zoomAround(1.0 / kButtonZoomFactor, DPoint(pane.width() / 2.0, pane.height() / 2.0), &pane);
    }

    void Viewer::fitToWindow() {
        if (impl_->vs().mode == ViewMode::Volume) impl_->volume.setZoom(1.0);
        impl_->fit();
    }

    void Viewer::setPlaying(bool on) { impl_->setPlaying(on); }

    bool Viewer::playing() const { return impl_->playing; }

    void Viewer::autoContrast() {
        Impl& d = *impl_;
        if (d.previewing) {   // the previewed step's own Auto
            const int i = d.wb.viewedIndex();
            // the window samples planes of the input, which a lazy source can fail to read
            try {
                if (auto up = d.wb.upstreamOutput(i))
                    d.wb.setStepParams(i, contrastAutoParams(d.wb.pipeline().at(i).params, up->asInput()), "Auto contrast");
            } catch (const std::exception& e) {
                d.wb.logLine(std::string("Auto contrast: ") + e.what());
            }
            return;
        }
        d.model.setWindowMode(DisplayModel::WindowMode::Auto);
        d.rawModel.setWindowMode(DisplayModel::WindowMode::Auto);
        d.dirty = Impl::Dirty{};
        d.scheduleUpdate();
    }

    void Viewer::resetContrast() {
        Impl& d = *impl_;
        if (d.previewing) {
            const int i = d.wb.viewedIndex();
            // the range reads planes of the input, which a lazy source can fail to read
            try {
                if (auto up = d.wb.upstreamOutput(i))
                    d.wb.setStepParams(i, contrastResetParams(d.wb.pipeline().at(i).params, up->asInput()), "Reset contrast");
            } catch (const std::exception& e) {
                d.wb.logLine(std::string("Reset contrast: ") + e.what());
            }
            return;
        }
        d.model.setWindowMode(DisplayModel::WindowMode::Full);
        d.rawModel.setWindowMode(DisplayModel::WindowMode::Full);
        // The full range is every voxel: ask for the volumes so the exact
        // one replaces the sampled stand-in as soon as it is read.
        d.retainVolumes();
        d.ensureVolumes(d.model, d.curT());
        // One output shown by both (the Load step viewed, or a step not run
        // yet) is read once, for both (onVolumeReady). A lazy one asked for
        // again was read in full on every click: the raw model never keeps
        // its volume, so it always wants one. Its exact range comes with the
        // projection: the raw pane asks only while it lacks that, as it does
        // at a time point of its own (Sync Z/T off) that the model's read
        // does not cover. That is the time point retainVolumes() keeps for it.
        const auto rawHasRanges = [&d] {
            for (Index c = 0; c < d.rawModel.dims().c; ++c)
                if (d.vs().channelOn(c) && !d.rawModel.mipIfReady(c, d.compareT())) return false;
            return true;
        };
        if (d.rawModel.valid() && (d.rawModel.output() != d.model.output() || !rawHasRanges()))
            d.ensureVolumes(d.rawModel, d.compareT());
        d.dirty = Impl::Dirty{};
        d.scheduleUpdate();
    }

    bool Viewer::grabView(std::vector<std::uint8_t>& rgba, int& width, int& height) { return impl_->grab(rgba, width, height); }

    std::string Viewer::cursorText() const { return impl_->cursorText; }
    std::string Viewer::zoomText() const { return impl_->zoomText; }
    bool Viewer::loading() const { return impl_->loadActive; }
    double Viewer::loadFraction() const { return impl_->loadFrac; }
    std::string Viewer::loadMessage() const { return impl_->loadActive ? impl_->sliceNotice : std::string(); }
    bool Viewer::animating() const { return impl_->playing || impl_->loadActive; }

    // Scripting: the press, moves and release a mouse would make on the XY
    // pane, through the pane's own bookkeeping and the same handlers; what
    // they changed is applied between the moves, as between frames.
    void Viewer::syntheticStroke(double x0, double y0, double x1, double y1, int moves) {
        Impl& d = *impl_;
        SlicePane& pane = d.xy;
        if (!d.model.valid()) return;
        moves = std::max(1, moves);
        const DPoint from(x0, y0), to(x1, y1);
        TraceClock clock, part;
        pane.synthPress(pane.toScreen(from), ImGuiMouseButton_Left);
        d.applyPending();
        long long inSend = 0, inEvents = 0;
        for (int i = 1; i <= moves; ++i) {
            const double f = static_cast<double>(i) / moves;
            part.start();
            pane.synthMove(pane.toScreen(from + (to - from) * f));
            inSend += part.restart();
            d.applyPending();
            inEvents += part.micros();
        }
        std::fprintf(stderr, "stroke: per move %lld us in the move handler, %lld us in the deferred events\n", inSend / moves, inEvents / moves);
        pane.synthRelease(pane.toScreen(to));
        d.applyPending();
        std::fprintf(stderr, "stroke: %d moves from (%.0f, %.0f) to (%.0f, %.0f) in %lld ms\n", moves, x0, y0, x1, y1, clock.micros() / 1000);
        std::fflush(stderr);
    }

    void Viewer::syntheticWheel(double x, double y, double steps) {
        Impl& d = *impl_;
        // the XY pane, or in Compare the step's pane (the XY pane is hidden there)
        SlicePane& pane = d.vs().mode == ViewMode::Compare ? d.cmpRight : d.xy;
        if (!d.model.valid() || !pane.onWheel) return;
        const DPoint pos = pane.toScreen(DPoint(x, y));
        TraceClock clock;
        pane.onWheel(pos, steps, ImGuiMod_None);
        d.applyPending();
        const DPoint under = pane.toVoxel(pos);   // where the anchor ended up: the same voxel when zoom keeps it
        std::fprintf(stderr, "wheel: %.1f steps at (%.0f, %.0f) on %s in %lld ms; under the cursor now (%.2f, %.2f)\n", steps, x, y,
                     pane.name().c_str(), clock.micros() / 1000, under.x, under.y);
        std::fflush(stderr);
    }

} // namespace sirius::app::gui
