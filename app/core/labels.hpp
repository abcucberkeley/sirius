#ifndef SIRIUS_APP_LABELS_HPP
#define SIRIUS_APP_LABELS_HPP

// Instance label volumes: (t, z, y, x) uint32 with 0 = background, produced by
// the segmentation steps and edited in the viewer's paint mode. Statistics
// per label feed the review table (voxels, confidence, flags) and the review
// queue; every edit returns the voxels it changed so the history can undo it.
//
// Ownership rule: a LabelVolume owns its statistics and shares its voxels
// copy-on-write. share() makes a second volume over the same voxels (the
// executor carries an input's labels through a step that does not produce
// its own this way), and the first mutating call on either volume -- the
// non-const volume() / plane() accessors, the edits, apply() -- gives that
// volume a private copy while the voxels are still shared. An edit through
// one step's output therefore never reaches another step's cached output,
// and an unmodified carry-through costs nothing. Sharing is decided by the
// thread doing the mutation (shared_ptr::use_count); one volume must not be
// written from two threads at once, which the workbench guarantees by
// refusing label edits while a run is active.

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <sirius/buffer.hpp>

#include "core/array.hpp"

namespace sirius::app {

    struct LabelStats {
        std::uint32_t id = 0;
        std::string cls = "object";
        Index voxels = 0;
        double confidence = 1.0;                // mean foreground probability, 1 when unknown
        std::array<Index, 6> bbox{};            // z0, z1, y0, y1, x0, x1 (half open)
        bool touchesBorder = false;
        std::vector<std::string> flags;         // "low conf", "small", "touching border", "merged?"
        bool reviewed = false;

        std::string flagText() const;           // first flag or ""
    };

    // Voxel diff of one edit: linear indices into one (z, y, x) volume of
    // time point t, with the values before and after.
    struct LabelDiff {
        Index t = 0;
        std::vector<Index> indices;
        std::vector<std::uint32_t> before;
        std::vector<std::uint32_t> after;

        bool empty() const noexcept { return indices.empty(); }
    };

    struct LabelFlagRules {
        double lowConfidence = 0.6;
        Index minVoxels = 0;                    // 0 = median / 8
        bool flagBorder = true;
        double sizeOutlierFactor = 4.0;         // > factor x median volume -> "merged?"
    };

    class LabelVolume {
    public:
        LabelVolume();
        LabelVolume(Index t, Index z, Index y, Index x);   // zero filled

        Index t() const noexcept { return t_; }
        Index z() const noexcept { return z_; }
        Index y() const noexcept { return y_; }
        Index x() const noexcept { return x_; }
        Index volumeSize() const noexcept { return z_ * y_ * x_; }
        bool empty() const noexcept { return data_->empty(); }
        // True while another volume shares these voxels (share()).
        bool sharesVoxels() const noexcept { return data_.use_count() > 1; }

        // The mutable accessors detach a shared copy first (see the header note).
        std::uint32_t* volume(Index t);                          // (z, y, x)
        const std::uint32_t* volume(Index t) const noexcept;
        std::uint32_t* plane(Index t, Index z);
        const std::uint32_t* plane(Index t, Index z) const noexcept;
        std::uint32_t at(Index t, Index z, Index y, Index x) const noexcept;
        BufferView<const std::uint32_t> view() const noexcept { return data_->view(); }

        // Highest id handed out so far: monotonic, so ids never collide with
        // labels that an undo may bring back. After a dense relabel
        // (cleanup) call resetMaxLabel() to make it the highest id present.
        std::uint32_t maxLabel() const noexcept;
        void resetMaxLabel() noexcept;

        // --- statistics ---------------------------------------------------
        // Recomputed from the voxels; `probabilities` (same (z, y, x) as one
        // volume, optional) gives per-label mean confidence -- without them a
        // label already known keeps the confidence it had (a cleanup or a
        // crop after a segmentation does not make every object certain),
        // a new one starts at 1. Class and review mark are kept the same
        // way, and the flags are refreshed with the rules of the last
        // applyFlags(). The statistics describe one time point, the last
        // one computed (statsT()).
        void recomputeStats(Index t, const float* probabilities = nullptr);
        // Brings the statistics up to date after an edit, touching only the
        // labels the diff changed (each is rescanned within its bounding
        // box): what the workbench calls at the end of a brush stroke and
        // after every other edit, undo and redo. Falls back to a full
        // recompute when the diff is for another time point than the
        // statistics, or when none were computed yet. Confidence and class
        // of a known label are kept; new labels start at 1.0 / "object".
        // The flags are refreshed with the rules of the last applyFlags().
        void updateStats(const LabelDiff& diff);
        Index statsT() const noexcept { return statsT_; }   // -1: none computed
        const std::vector<LabelStats>& stats() const noexcept { return stats_; }
        std::vector<LabelStats>& stats() noexcept { return stats_; }
        const LabelStats* statsOf(std::uint32_t id) const noexcept;
        void applyFlags(const LabelFlagRules& rules);
        Index reviewedCount() const noexcept;
        Index flaggedCount(const std::string& flag) const noexcept;

        // --- edits (all return the voxels they changed) -------------------
        // Paint a ball (radius in x/y voxels, zRadius planes) of `label`;
        // `erase` paints 0. `onlyLabel` restricts the change to voxels of that label.
        LabelDiff paint(Index t, Index cz, Index cy, Index cx, double radius, Index zRadius,
                        std::uint32_t label, std::uint32_t onlyLabel = 0);
        // Fill the connected region of the label under (z, y, x) with `label` (flood fill, 6-connected).
        LabelDiff fill(Index t, Index z, Index y, Index x, std::uint32_t label);
        LabelDiff merge(Index t, const std::vector<std::uint32_t>& ids);   // all into the smallest id
        LabelDiff remove(Index t, std::uint32_t id);
        // Split `id` into two by a watershed from the two seeds (distance transform);
        // the new part gets maxLabel() + 1.
        LabelDiff split(Index t, std::uint32_t id, std::array<Index, 3> seedA, std::array<Index, 3> seedB);
        // Re-apply / revert a diff.
        void apply(const LabelDiff& diff, bool forward);

        // A deep copy: its own voxels from the start.
        std::shared_ptr<LabelVolume> clone() const;
        // A volume over the same voxels (and a copy of the statistics) that
        // takes a private copy on its first write; see the header note.
        std::shared_ptr<LabelVolume> share() const;

    private:
        void detach();                       // own the voxels before writing
        LabelStats* mutableStatsOf(std::uint32_t id) noexcept;

        Index t_ = 0, z_ = 0, y_ = 0, x_ = 0;
        std::shared_ptr<Buffer<std::uint32_t>> data_;   // never null
        std::vector<LabelStats> stats_;
        Index statsT_ = -1;
        std::optional<LabelFlagRules> flagRules_;       // the last applyFlags()
        std::uint32_t maxLabel_ = 0;
    };

    using LabelsPtr = std::shared_ptr<const LabelVolume>;

    // --- algorithms --------------------------------------------------------

    // 6-connected components of mask > 0 on a (z, y, x) volume; returns the
    // label count. `out` receives 1..n.
    std::uint32_t connectedComponents(const std::uint8_t* mask, Index z, Index y, Index x, std::uint32_t* out);

    // Marker-based watershed on `landscape` (higher = ridge) starting from
    // `seeds` (non-zero label voxels), restricted to mask > 0. Priority-flood.
    void watershed(const float* landscape, const std::uint8_t* mask, Index z, Index y, Index x,
                   std::uint32_t* labels /* in: seeds, out: result */);

    // Seeds for a watershed of a foreground probability map: local maxima of
    // the distance transform of fg > threshold, at least `minDistance` apart.
    std::uint32_t distanceSeeds(const std::uint8_t* mask, Index z, Index y, Index x, double minDistance,
                                std::uint32_t* out);

    // Euclidean distance transform of mask > 0 (distance to the nearest 0), 3D.
    void distanceTransform(const std::uint8_t* mask, Index z, Index y, Index x, float* out);

    // Morphological reconstruction of `marker` by dilation under `mask`
    // (marker <= mask elementwise), 6-connected, in place on `marker`.
    void reconstructByDilation(float* marker, const float* mask, Index z, Index y, Index x);

    // Watershed seeds from the h-maxima of `values` inside mask > 0: the
    // regional maxima that stand at least `h` above the surrounding
    // landscape. A lumpy or elongated object whose distance map has several
    // shallow bumps yields one seed instead of one per bump, which is the
    // usual cause of over-segmentation. Returns the seed count; `out` gets
    // 1..n at the seed voxels and 0 elsewhere.
    std::uint32_t hMaximaSeeds(const float* values, const std::uint8_t* mask, Index z, Index y, Index x, double h,
                               std::uint32_t* out);

    // Separable 3D Gaussian with reflected borders, truncated at three sigma,
    // in place on `v` (z, y, x); `tmp` is scratch the caller can reuse.
    void gaussianVolume(std::vector<float>& v, Index z, Index y, Index x, double sx, double sy, double sz,
                        std::vector<float>& tmp);

    // Watershed seeds from a 3D scale-space blob detector: the scale-normalised
    // Laplacian of Gaussian is evaluated at `scales` widths between sigmaMin
    // and sigmaMax (voxels in x and y, divided by `zAspect` in z so the width
    // is the same physical size along every axis), and its strongest response
    // over the scales peaks once at the centre of each round object whatever
    // its size. That is what the distance map cannot do: it seeds by shape, so
    // objects of different sizes need different settings. Peaks are accepted
    // strongest first and suppress others within the width that found them.
    // Returns the seed count; `out` gets 1..n at the seed voxels.
    std::uint32_t logBlobSeeds(const float* values, const std::uint8_t* mask, Index z, Index y, Index x, double zAspect,
                               double sigmaMin, double sigmaMax, int scales, std::uint32_t* out);

    // --- filters and thresholds -------------------------------------------

    // 3x3 median of one plane, in place. Removes shot noise without moving an
    // edge, which a Gaussian does; the usual first step on a noisy stack.
    void medianFilterPlane(float* plane, Index y, Index x, std::vector<float>& tmp);

    // Perona-Malik anisotropic diffusion of one plane, in place: `iterations`
    // explicit steps with the exponential conductance exp(-(|grad| / k)^2), so
    // flat regions smooth and edges do not. `k` is a fraction (0..1) of the
    // plane's intensity range, which keeps the setting scale free. lambda is
    // fixed at 0.25, the stability limit for four neighbours.
    void anisotropicDiffusionPlane(float* plane, Index y, Index x, int iterations, double k, std::vector<float>& tmp);

    // Triangle (Zack) threshold: the value furthest from the line joining the
    // histogram's peak to its far end. Made for the skewed histogram of a
    // fluorescence image, where most of the field is background and Otsu's
    // two-class assumption puts the cut too high.
    float triangleThreshold(const float* values, Index n);

    // Li's minimum cross-entropy threshold, found by the iterative fixed point
    // of the original paper. Keeps dim objects Otsu discards.
    float liThreshold(const float* values, Index n);

    // Yen's threshold: the cut that maximises Yen's entropic correlation
    // criterion on the histogram. Between Otsu and the triangle in practice --
    // it tolerates a background that outweighs the signal without giving the
    // whole tail away.
    float yenThreshold(const float* values, Index n);

    // Isodata (Ridler-Calvard): iterate t = (mean below t + mean above t) / 2
    // from the image mean until it settles. The oldest of the automatic
    // thresholds and still a reasonable default on a clean bimodal image.
    float isodataThreshold(const float* values, Index n);

    // Hysteresis: keep every 6-connected component of `low` that contains at
    // least one voxel of `high`, write it to `out` (may alias `low`). A
    // filament that fades below the cut stays whole as long as part of it is
    // clearly above. Returns the number of voxels kept.
    Index hysteresisMask(const std::uint8_t* high, const std::uint8_t* low, Index z, Index y, Index x, std::uint8_t* out);

    // Rolling-ball background, subtracted in place from one plane.
    //
    // A ball of `radius` is rolled along the underside of the intensity
    // surface and the surface its top traces is the background: a grey opening
    // with a hemispherical structuring element, which follows a curved,
    // uneven background where the box top-hat can only follow a flat one.
    // Everything narrower than the ball survives.
    //
    // The ball is run on a decimated copy and the background interpolated back
    // up, as ImageJ does, because a full-resolution ball of radius r costs
    // (2r+1)^2 per pixel: the shrink factor is radius / 10, capped at 8, and a
    // radius under 15 runs at full resolution.
    void rollingBallPlane(float* plane, Index y, Index x, double radius, std::vector<float>& scratch);

    // Fill background enclosed by foreground in 3D: every 0-component that
    // does not reach the volume border becomes 1, unless it is larger than
    // `maxVoxels` (0 = no limit). The per-plane fill cannot close a cavity
    // that no single plane encloses, which is most of them in a stack.
    // Returns the number of voxels filled.
    Index fillHoles3D(std::uint8_t* mask, Index z, Index y, Index x, Index maxVoxels);

    // Thin a binary volume to its centrelines, in place.
    //
    // Topological thinning: border voxels are deleted, one of the six
    // directions at a time, but only where deleting one changes neither the
    // object's connectivity nor the background's -- a "simple" point, which
    // here is tested directly (one 26-connected component of the object in the
    // 26-neighbourhood, one 6-connected component of the background in the
    // 18-neighbourhood touching the centre) rather than through a lookup
    // table. Voxels with a single neighbour are kept, so a curve is thinned to
    // a line instead of eroded away. What is left is one voxel thick, runs
    // down the middle of the structure, and has the same topology as it had:
    // the centreline of a filament, and the length of one measured on it.
    // Returns the number of voxels that remain.
    //
    // `poll` is called once per direction of every pass: a thinning runs the
    // whole volume many times over, and a step that cannot be cancelled until
    // it finishes is a step that cannot be cancelled. Throw from it to stop.
    Index skeletonize3D(std::uint8_t* mask, Index z, Index y, Index x, const std::function<void()>& poll = {});

    // Grow every label outwards into the background, up to `distance` voxels
    // measured in x / y pixels -- a step in z costs `zAspect` of them, the
    // planes being that much further apart -- nearest label first;
    // a voxel equidistant from two labels is left as background so the two do
    // not fuse. What closes the gap an opening or a watershed line left
    // behind without changing which object a voxel belongs to. Returns the
    // number of voxels claimed.
    Index expandLabels(std::uint32_t* labels, Index z, Index y, Index x, double distance, double zAspect);

    // Central-difference gradient magnitude of a (z, y, x) volume, with z
    // scaled by `zAspect` so the gradient is physical. The landscape a
    // boundary watershed floods: its ridges are the object edges, which is
    // what separates touching objects that have a visible boundary but no
    // waist for the distance transform to find.
    void gradientMagnitude(const float* values, Index z, Index y, Index x, double zAspect, float* out);

    // Morphological Chan-Vese (Marquez-Neila et al.), one plane, in place on
    // `mask`. Each iteration moves the contour towards the two-region fit of
    // the image -- inside mean against outside mean, no edge and no shape
    // assumption -- then smooths it with `smoothing` rounds of the alternating
    // sup-inf / inf-sup operators over the four line elements, which is the
    // morphological stand-in for the curvature term. This is the level-set
    // (snake) refinement without the PDE: it fixes a threshold that leaks or
    // pinches, on filaments as readily as on round objects.
    void morphologicalChanVesePlane(const float* image, std::uint8_t* mask, Index y, Index x, int iterations, int smoothing,
                                    std::vector<std::uint8_t>& tmp);

    // --- shape filters -----------------------------------------------------

    struct ShapeFilter {
        Index minVoxels = 0;             // 0 = off
        Index maxVoxels = 0;             // 0 = off
        double minFill = 0.0;            // voxels / bounding box volume, 0 = off
        double maxElongation = 0.0;      // longest bounding box side / shortest, 0 = off
        bool dropBorder = false;         // objects touching the x / y border
    };

    // Removes the objects a filter rejects and relabels 1..n densely.
    // Returns the label count that remains.
    std::uint32_t filterLabelsByShape(std::uint32_t* labels, Index z, Index y, Index x, const ShapeFilter& filter);

    // Drop components smaller than minVoxels, relabel 1..n densely.
    std::uint32_t removeSmall(std::uint32_t* labels, Index n, Index minVoxels);
    // Drop components smaller than minVoxels; the other ids are kept as they
    // are. Returns the label count that remains.
    std::uint32_t dropSmall(std::uint32_t* labels, Index n, Index minVoxels);

    // Distinct colour for a label id (the design's 7-colour label palette, cycled).
    std::array<float, 3> labelColor(std::uint32_t id) noexcept;

} // namespace sirius::app

#endif // SIRIUS_APP_LABELS_HPP
