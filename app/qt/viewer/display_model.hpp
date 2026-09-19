#ifndef SIRIUS_APP_VIEWER_DISPLAY_MODEL_HPP
#define SIRIUS_APP_VIEWER_DISPLAY_MODEL_HPP

// Turns one step output into the pixels the panes draw: per-channel display
// windows (robust percentiles, computed once per output and channel), the
// additive channel blend into an RGB image, the XZ / YZ re-slices and the
// z maximum projection, and the label overlay. Lazy (on-disk) outputs are
// read plane by plane for XY and as one (c, t) volume, cached, for the
// re-slices; everything else reads straight out of the in-memory array.
//
// Nothing here ever reads a whole volume: volumeState() says whether one is
// in memory, has to be produced (by ViewerLoader, off the GUI thread) or is
// too large to hold at all, and installVolume() takes the loader's result.
// The re-slices and the MIP draw what they have and the viewer shows a
// loading state for the rest.
//
// All buffers persist between calls so scrubbing through a stack allocates
// nothing; a change of output, t or channel invalidates only what depends
// on it.

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <QImage>
#include <QRect>

#include "core/byte_budget_lru.hpp"
#include "core/operation.hpp"
#include "core/workbench.hpp"

namespace sirius::app {

    struct DisplayWindow {
        float lo = 0.0f;
        float hi = 1.0f;
        float gamma = 1.0f;   // display = ((v - lo) / (hi - lo)) ^ (1 / gamma)
    };

    class DisplayModel {
    public:
        // Bytes of one (c, t) volume kept for the re-slices; above it the panes
        // that need a whole volume report "too large" instead of reading it.
        static constexpr std::size_t kVolumeCacheLimit = std::size_t{3} << 30;   // 3 GiB
        // The projections of the time points already played, least recently
        // shown dropped first (64 frames of one 2048 x 2048 channel).
        static constexpr std::size_t kMipCacheLimit = std::size_t{1} << 30;      // 1 GiB

        void setOutput(std::shared_ptr<const StepOutput> out);
        std::shared_ptr<const StepOutput> output() const noexcept { return out_; }
        bool valid() const noexcept;
        const DatasetMeta& meta() const noexcept { return meta_; }
        const Dims5& dims() const noexcept { return meta_.dims; }
        bool hasLabels() const noexcept;
        const LabelVolume* labels() const noexcept;

        // --- windows ---------------------------------------------------------
        // Auto: robust percentiles (0.1 / 99.9) of a few sampled planes;
        // Full: the (c, t) volume's minimum and maximum, nothing clipped.
        enum class WindowMode { Auto,
                                Full };
        DisplayWindow window(Index c, Index t);
        void setWindow(Index c, DisplayWindow w);   // an explicit window (live previews)
        void setWindowMode(WindowMode m);
        WindowMode windowMode() const noexcept { return windowMode_; }
        void resetWindows();

        // --- data (cached) ---------------------------------------------------
        // The (y, x) plane; null when it cannot be read. One plane of a lazy
        // source is a small read the GUI thread can afford.
        const float* plane(Index c, Index t, Index z);
        // The (z, y, x) volume and its z maximum projection, or null when
        // they are not in memory yet. Neither ever touches the disk.
        const float* volumeIfReady(Index c, Index t) const;
        const float* mipIfReady(Index c, Index t) const;
        // Keeps a volume alive while another thread reads it (the 3D
        // reduction); null for an in-memory output, whose StepOutput owns it.
        std::shared_ptr<const Buffer<float>> volumeHold(Index c, Index t) const;

        // What the re-slices, the MIP corner and the 3D view can do with
        // (c, t) right now. Wanted asks the caller to have a ViewerLoader
        // produce it; a cheap projection of an in-memory volume is done here
        // and reports Ready.
        enum class VolumeState { Ready,
                                 Wanted,
                                 TooLarge };
        VolumeState volumeState(Index c, Index t);
        // The loader's result: the volume (null for an in-memory output), its
        // projection and its exact range. `shownT` is the time point on
        // screen when `t` is another one (play reading ahead): its volumes are
        // kept, and the read-ahead volume is kept beside them only while both
        // fit in kVolumeCacheLimit. -1: `t` is the one shown.
        void installVolume(Index c, Index t, std::shared_ptr<Buffer<float>> volume,
                           std::shared_ptr<Buffer<float>> mip, float lo, float hi, Index shownT = -1);
        bool volumeTooLarge() const noexcept;
        std::optional<float> valueAt(Index c, Index t, Index z, Index y, Index x);
        void dropVolumeCaches();

        // --- rendering -------------------------------------------------------
        // `factor` sub-samples the plane (image pixel = factor voxels).
        // Every renderer resizes `img` as needed (Format_RGB32) and blends the
        // visible channels of `vs` additively with their colours.
        // `region` (voxels, empty = the whole plane) limits the render to the
        // part of the plane on screen: the cost follows the window, not the data.
        void renderXY(Index t, Index z, const ViewState& vs, int factor, QImage& img, const QRect& region = QRect());
        void renderXZ(Index t, Index y, const ViewState& vs, int factor, QImage& img, const QRect& region = QRect());   // rows z, cols x
        void renderYZ(Index t, Index x, const ViewState& vs, int factor, QImage& img, const QRect& region = QRect());   // rows y, cols z
        void renderMIP(Index t, const ViewState& vs, int factor, QImage& img);

        // Label overlay on an image produced by the matching renderer.
        void overlayLabelsXY(Index t, Index z, int factor, const ViewState& vs, QImage& img, const QRect& region = QRect());
        void overlayLabelsXZ(Index t, Index y, int factor, const ViewState& vs, QImage& img, const QRect& region = QRect());
        void overlayLabelsYZ(Index t, Index x, int factor, const ViewState& vs, QImage& img, const QRect& region = QRect());

    private:
        struct Key {
            Index c, t;
            bool operator<(const Key& o) const noexcept { return c < o.c || (c == o.c && t < o.t); }
        };
        struct PlaneKey {
            Index c = -1, t = -1, z = -1;
            bool operator==(const PlaneKey& o) const noexcept { return c == o.c && t == o.t && z == o.z; }
        };
        struct ChannelPlane {
            const float* data = nullptr;
            Index rowStride = 0;    // floats between rows
            Index colStride = 1;    // floats between columns
            std::array<int, 3> tint{256, 256, 256};
            DisplayWindow window;
            std::shared_ptr<const std::array<std::uint8_t, 256>> lut;   // gamma table, null for gamma 1
        };

        DisplayWindow computeWindow(Index c, Index t);
        std::vector<ChannelPlane> visibleChannels(const ViewState& vs, Index t);
        void blend(std::vector<ChannelPlane> chans, Index rows, Index cols, int factor, QImage& img);
        // `only` (non-zero) draws that label alone: the solo view.
        void overlay(const std::uint32_t* lab, Index rows, Index cols, Index rowStride, Index colStride, int factor,
                     float opacity, std::uint32_t selected, std::uint32_t only, QImage& img);
        static std::array<int, 3> tintOf(const DatasetMeta& m, Index c, bool rgb);

        // A projection of an in-memory volume up to this many voxels is
        // computed inline: a few milliseconds, not worth a thread hop.
        static constexpr Index kInlineProjectVoxels = Index{8} << 20;
        // Keep at most one time point of the heavy (c, t) volumes (the 3 GiB
        // cap). Play must not throw the projections away or every frame
        // re-projects 10^8 voxels, but a projection is 16 MB at 2048 x 2048, so
        // they are kept up to kMipCacheLimit rather than for the whole movie
        // (two channels, 500 frames: 32 GB). The ranges are a few bytes each
        // and all stay.
        void evictOtherTimePoints(Index t, Index alsoKeep = -1);
        void storeMip(const Key& key, std::shared_ptr<Buffer<float>> mip);

        std::shared_ptr<const StepOutput> out_;
        DatasetMeta meta_;
        std::map<Index, DisplayWindow> windows_;          // per channel
        WindowMode windowMode_ = WindowMode::Auto;
        // Shared so a loader thread can hold a volume the model has evicted.
        std::map<Key, std::shared_ptr<Buffer<float>>> volumes_;   // lazy sources only
        std::map<Key, std::shared_ptr<Buffer<float>>> mips_;
        mutable ByteBudgetLru<Key> mipBudget_{kMipCacheLimit};   // touched by mipIfReady
        struct Range {
            float lo = 0.0f, hi = 1.0f;
        };
        std::map<Key, Range> ranges_;                     // exact, from the loader
        bool tooLarge_ = false;
        // one cached plane per channel for lazy sources without a cached volume
        std::map<Index, std::pair<PlaneKey, Buffer<float>>> planes_;
        std::vector<std::uint32_t> labelScratch_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_VIEWER_DISPLAY_MODEL_HPP
