#ifndef SIRIUS_APP_OPS_COMMON_HPP
#define SIRIUS_APP_OPS_COMMON_HPP

// What every operation implementation needs: the (c, t) volume loops that
// report progress and honour cancellation, the small formatters its summary
// is built from, and the two standard Diagnostics.

#include <array>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "core/operation.hpp"

namespace sirius::app {

    // Runs `fn(c, t, progress)` for every (c, t) volume, reporting progress
    // 0..1 and honouring cancellation.
    void forEachVolume(const DatasetMeta& meta, const StepContext& ctx,
                       const std::function<void(Index c, Index t)>& fn);
    // Like forEachVolume, but when the context asks for every GPU the (c, t)
    // volumes run in parallel (one OpenMP thread per device) and `fn` is
    // given the device for that volume.
    void forEachVolumeOnGpus(const DatasetMeta& meta, const StepContext& ctx,
                             const std::function<void(Index c, Index t, Device device)>& fn);
    // "3 angles · 5 phases" style joining with " · ".
    std::string joinSummary(std::initializer_list<std::string> parts);
    // Channel label "488 α-actinin" for summaries.
    std::string channelName(const DatasetMeta& meta, Index c);
    // "12.8 GB", "412 MB"
    std::string formatBytes(std::uint64_t bytes);
    std::string formatNumber(double v, int decimals);
    // Otsu's threshold of `n` values (256-bin histogram over the finite ones; NaN and +-inf ignored).
    float otsuThreshold(const float* values, Index n);
    // "~9 s" for `bytes` at a nominal throughput.
    std::string estimatedTime(std::uint64_t bytes, double bytesPerSecond);
    // Input / Output thumbnails, summary and cost facts (the "Einsum / other" panel).
    Diagnostics genericDiagnostics(const StepInput& input, const StepOutput& output, const std::string& summary,
                                   double bytesPerSecond = 2.0e9);
    std::shared_ptr<Array5> allocateLike(const DatasetMeta& meta);
    // Label table + review-queue facts (the segmentation panel's data).
    Diagnostics labelDiagnostics(const LabelVolume& labels, const std::string& summary);

    // One time point of a Prompt step as the worker is asked it: the joint
    // `objects` form (docs/foundation_model_integration.md section 7), one
    // entry per object that has an object prompt (a box, an object point or
    // scribble) on that time point, in ascending id, holding ALL of its
    // prompts there -- its background points too, which is what makes a
    // corrective click refine that object's mask rather than ask for another.
    // Only `objects` is sent, so the worker labels the i-th object's mask
    // i + 1, and applyPromptIds turns that into the object's own id: an
    // object keeps its label (and its colour) across re-runs. An object with
    // only background prompts on the time point names nothing there and is
    // not sent.
    struct FramePrompt {
        struct Stroke {
            std::vector<std::array<double, 3>> points;   // a few, evenly spaced along the stroke
            int label = 1;
        };
        struct Object {
            std::uint32_t id = 0;
            std::optional<std::array<double, 6>> box;    // (x0, y0, z0, x1, y1, z1); its first box
            std::size_t boxes = 0;                        // how many it was given (validatePrompts: one)
            std::vector<std::array<double, 3>> points;   // (x, y, z), in the list's order
            std::vector<int> pointLabels;                 // 1 object, 0 background
            std::vector<Stroke> scribbles;
            std::vector<std::size_t> placed;              // its prompts' indices in the step's list
        };
        std::vector<Object> objects;                      // sent, in the worker's order
        std::vector<std::uint32_t> backgroundOnly;        // on this time point, not sent
        bool empty() const noexcept { return objects.empty(); }
        // the label the worker's i-th mask becomes
        std::vector<std::uint32_t> ids() const;
    };
    // A scribble is sent as at most this many points, evenly spaced along the
    // stroke and including both ends: the prompt decoder was trained on a few
    // points per stroke (three, at random), and a stroke's every voxel would
    // make one decoder call no better and much slower.
    inline constexpr std::size_t kScribblePointsSent = 8;
    std::vector<std::array<double, 3>> scribbleSample(const std::vector<std::array<double, 3>>& stroke, std::size_t at_most = kScribblePointsSent);
    FramePrompt framePrompt(const std::vector<Prompt>& prompts, Index t);
    // The worker's `objects` parameter for one time point:
    // [{"box", "points", "point_labels", "scribbles"}, ...], each key only
    // when the object has one.
    nlohmann::json promptObjectsJson(const FramePrompt& frame);
    // Masks numbered as the worker returned them (i + 1 for the i-th object)
    // renumbered to the objects' ids, in place; an id beyond the list becomes 0.
    void applyPromptIds(std::uint32_t* labels, Index n, const FramePrompt& frame);
    // The planes a 2-D model (micro-SAM) answers an object in, worked out as
    // its worker does: those of its points and scribble points, or the
    // middle of its box when it has neither. More than one is refused.
    std::vector<Index> promptPlanes(const FramePrompt::Object& object);
    // Every prompt inside the image and on one of its time points (the
    // worker refuses one outside the volume, and one placed on other data,
    // a dataset swapped under the step, would otherwise be a failed run
    // rather than a line in the panel), and at most one box per object and
    // time point. No prompts, an object with only background prompts (not
    // sent), or only background prompts at all, is a warning: the step runs.
    void validatePrompts(const std::vector<Prompt>& prompts, const DatasetMeta& in, Validation& v);
    // "2 objects · 1 box · 3 object points · 1 background point": what a run was given.
    std::string promptCounts(const std::vector<Prompt>& prompts);

    // One object of a Prompt step, as the Parameters panel lists it.
    struct PromptObject {
        std::uint32_t id = 0;
        std::vector<std::size_t> prompts;   // its prompts' indices in the list, in order
        std::vector<Index> times;           // the time points it has prompts on, ascending
        std::size_t boxes = 0, points = 0, scribbles = 0;   // object prompts
        std::size_t corrections = 0;                         // background prompts
        bool sent() const noexcept { return boxes + points + scribbles > 0; }
    };
    // The objects of `prompts`, in ascending id.
    std::vector<PromptObject> promptObjects(const std::vector<Prompt>& prompts);
    // "box + 2 points + 1 correction"
    std::string promptObjectText(const PromptObject& object);
    // The id a new object gets: one above the highest in the list.
    std::uint32_t nextPromptObject(const std::vector<Prompt>& prompts);

    // What a click with the viewer's Prompt tool does, decided without the
    // GUI. `at` is the voxel clicked on time point `t`, `maskUnder` the label
    // the step's last result has there (0: none), `positive` false for a
    // background click (Alt, right button), `forceNew` true with Shift.
    //   object click: inside the mask of an object of this time point, a
    //     point added to that object (to grow it); elsewhere, or with
    //     forceNew, a new object.
    //   background click: added to the object whose mask is under it, else
    //     to the nearest object of this time point (promptDistance to its
    //     prompts, the lower id on a tie); with no object, nothing, and
    //     `why` says so.
    // `planar` (a 2-D model, micro-SAM) counts only the objects whose
    // prompts lie on the clicked plane: all of an object's must share one.
    struct PromptClick {
        enum class Action { None,
                            NewObject,
                            AddTo };
        Action action = Action::None;
        std::uint32_t object = 0;   // AddTo: which; NewObject: the id it gets
        std::string why;            // None: the reason, for the viewer's hint
    };
    PromptClick promptClickTarget(const std::vector<Prompt>& prompts, Index t, const std::array<double, 3>& at, std::uint32_t maskUnder,
                                  bool positive, bool forceNew, bool planar = false);
    // `prompts` without prompt k; when that was its object's last object
    // prompt on its time point, the object's background prompts there go too.
    std::vector<Prompt> removePrompt(const std::vector<Prompt>& prompts, std::size_t k);
    // `prompts` without object `id`, on every time point.
    std::vector<Prompt> removePromptObject(const std::vector<Prompt>& prompts, std::uint32_t id);

    // The model's score of each object's mask (the worker's mask_scores, in
    // the order sent), appended to a "Mask scores" fact: "#3 0.81, #5 0.70",
    // with "t 2: " before a time point's when the data has more than one;
    // promptScores reads them back from a step's diagnostics, for time point t.
    void appendPromptScores(std::string& fact, const FramePrompt& frame, const nlohmann::json& maskScores, Index t, bool manyTimes);
    std::vector<std::pair<std::uint32_t, double>> promptScores(const Diagnostics& diagnostics, Index t);

    // The "device" of a request to the Python worker: where this run was
    // asked to go, as the launcher names it when it starts a local worker.
    // "auto" would mean the worker's own, fixed when its process started
    // and kept for the session, so a backend or GPU chosen since never
    // reached it. The HPC worker is told the session's choice
    // (StepContext::hpcDevice), its job's GPU or its CPU, with every
    // request, so a switch needs no new job.
    inline std::string workerDevice(const StepContext& ctx) {
        if (ctx.backend == Backend::Cpu) return "cpu";
        if (ctx.backend == Backend::Cuda)
            return ctx.device.isCuda() && ctx.device.index >= 0 ? "cuda:" + std::to_string(ctx.device.index) : std::string("cuda");
        return ctx.hpcDevice == HpcDevice::Cpu ? "cpu" : "cuda";
    }

} // namespace sirius::app

#endif // SIRIUS_APP_OPS_COMMON_HPP
